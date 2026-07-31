"""Regression tests for a revision the repository does not hold (#71).

A SHA that resolves to nothing used to produce a blank message and an empty file
list, which is byte for byte what a commit that exists and touched no files
produces. A rater reading that report sees strong evidence the commit did
nothing, and `not-a-refactoring` is the reasonable label to write from it.

These build throwaway git repos, so git runs for real but no network does: the
question under test is what git says about a revision, and mocking that away
would test the mock.
"""
import io, os, shutil, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import inspect_commit as ic

# Well-formed and 40 characters, which is the point: `git rev-parse` echoes such
# a string back unchanged whether or not the object is there, so a check that
# does not peel it to a commit passes on it.
ABSENT = "545b0ea518" + "0" * 30
DS = "someone/a-dataset"


def git(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True,
                   capture_output=True, text=True)


def rev_parse(repo, rev="HEAD"):
    return subprocess.run(["git", "-C", repo, "rev-parse", rev],
                          capture_output=True, text=True).stdout.strip()


class MissingRevisionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        git(self.tmp, "init", "-q", ".")
        git(self.tmp, "config", "user.email", "t@example.com")
        git(self.tmp, "config", "user.name", "t")
        self._clone = ic.clone
        ic.clone = lambda ds: self.tmp
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def tearDown(self):
        ic.clone = self._clone

    def commit(self, name=None, msg="c"):
        """A commit that adds `name`, or an empty one when `name` is None."""
        if name is None:
            git(self.tmp, "commit", "-q", "--allow-empty", "-m", msg)
        else:
            with open(os.path.join(self.tmp, name), "w") as f:
                f.write("a,b,c\n1,2,3\n")
            git(self.tmp, "add", "-A")
            git(self.tmp, "commit", "-qm", msg)
        return rev_parse(self.tmp)

    def run_inspect(self, sha):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            ic.inspect(DS, sha, False, False)
        return out.getvalue(), err.getvalue()


class TestResolveCommit(MissingRevisionTest):
    def test_a_commit_in_the_repo_resolves_to_its_full_sha(self):
        sha = self.commit("data.csv")
        self.assertEqual(ic.resolve_commit(self.tmp, sha), sha)

    def test_an_abbreviation_resolves_to_the_full_sha(self):
        sha = self.commit("data.csv")
        self.assertEqual(ic.resolve_commit(self.tmp, sha[:10]), sha)

    def test_a_well_formed_sha_the_repo_does_not_hold_is_none(self):
        """The #71 case: 40 hex characters that name no object here. rev-parse
        without --verify echoes this straight back, which is why the check
        cannot be a spelling check."""
        self.commit("data.csv")
        self.assertIsNone(ic.resolve_commit(self.tmp, ABSENT))

    def test_a_truncated_sha_that_matches_nothing_is_none(self):
        sha = self.commit("data.csv")
        # Same length as a usable abbreviation, one character off.
        wrong = ("f" if sha[0] != "f" else "e") + sha[1:10]
        self.assertIsNone(ic.resolve_commit(self.tmp, wrong))

    def test_an_object_that_is_not_a_commit_is_none(self):
        """A tree or blob SHA is a revision git resolves happily, and analysis
        of it would report nothing changed."""
        self.commit("data.csv")
        tree = rev_parse(self.tmp, "HEAD^{tree}")
        self.assertIsNone(ic.resolve_commit(self.tmp, tree))

    def test_an_annotated_tag_peels_to_the_commit_it_points_at(self):
        sha = self.commit("data.csv")
        git(self.tmp, "tag", "-a", "v1", "-m", "release")
        self.assertEqual(ic.resolve_commit(self.tmp, "v1"), sha)


class TestMissingCommitMessage(unittest.TestCase):
    def test_it_names_the_commit_the_dataset_and_where_to_check(self):
        msg = ic.missing_commit(DS, ABSENT)
        self.assertIn(ABSENT, msg)
        self.assertIn(DS, msg)
        self.assertIn(ic.CANDIDATES, msg)

    def test_it_denies_the_reading_it_exists_to_prevent(self):
        """The whole defect is that the empty report reads as `changed
        nothing`, so the message has to say that is not what happened."""
        msg = ic.missing_commit(DS, ABSENT)
        self.assertIn("not a commit that changed no files", msg)


class TestInspectOnAMissingRevision(MissingRevisionTest):
    def test_it_exits_non_zero_rather_than_reporting_an_empty_commit(self):
        self.commit("data.csv")
        with self.assertRaises(SystemExit) as e:
            self.run_inspect(ABSENT)
        code = e.exception.code
        self.assertIsNotNone(code)          # sys.exit(str) exits 1
        self.assertNotEqual(code, 0)
        self.assertIn("not found", str(code))

    def test_nothing_is_printed_before_the_check(self):
        """A header and a blank `message:` ahead of the error would leave the
        misleading half of the old output on screen anyway."""
        self.commit("data.csv")
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit):
                ic.inspect(DS, ABSENT, False, False)
        self.assertEqual(out.getvalue(), "")

    def test_a_commit_that_changed_nothing_still_reports_none(self):
        """The other half of #71: `(none)` has to keep meaning what it says, or
        the fix would only move the ambiguity."""
        self.commit("data.csv")
        empty = self.commit(None, msg="nothing to see")
        out, _ = self.run_inspect(empty)
        self.assertIn("file changes:", out)
        self.assertIn("(none)", out)
        self.assertIn("nothing to see", out)

    def test_a_real_commit_is_analyzed_as_before(self):
        sha = self.commit("data.csv", msg="add data")
        out, _ = self.run_inspect(sha)
        self.assertIn("add data", out)
        self.assertIn("data.csv", out)
        self.assertNotIn("(none)", out)

    def test_the_two_outcomes_do_not_look_alike(self):
        """Stated as the comparison the issue makes: the report for a missing
        revision and the report for an empty commit must not coincide."""
        self.commit("data.csv")
        empty_out, _ = self.run_inspect(self.commit(None))
        with self.assertRaises(SystemExit) as e:
            self.run_inspect(ABSENT)
        self.assertNotEqual(empty_out.strip(), str(e.exception.code).strip())


if __name__ == "__main__":
    unittest.main()
