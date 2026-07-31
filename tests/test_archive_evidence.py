"""Tests for the evidence archive generator (#65).

The constraint that matters most is inherited from #34, #55, #59 and #72: what
the generator writes for a failure must not be indistinguishable from what it
writes for an empty result. These tests hold the record layer to that, and they
run no network: git runs for real in throwaway repos, and everything that would
touch the Hub is stubbed at the seam archive_evidence already exposes.
"""
import io, os, shutil, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import archive_evidence as ae
import inspect_commit as ic

DS = "someone/a-dataset"
ABSENT = "545b0ea518" + "0" * 30

ROW = {"DatasetID": DS, "CommitId": ABSENT, "tentative_type": "cleaning",
       "log_message": "tidy up"}

REACHABLE = ic.RepoStatus(ic.REACHABLE, "HTTP 200")
GONE = ic.RepoStatus(ic.GONE, "HTTP 401")
MOVED = ic.RepoStatus(ic.MOVED, "HTTP 301", moved_to="elsewhere/a-dataset")


def git(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True,
                   capture_output=True, text=True)


class RecordRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_filename_matches_the_clone_cache_mapping(self):
        self.assertEqual(ae.evidence_filename(DS, ABSENT),
                         f"someone__a-dataset__{ABSENT[:10]}.txt")

    def test_header_fields_survive_the_round_trip(self):
        name = ae.write_record(self.tmp, ROW, ae.ARCHIVED, REACHABLE, "body",
                               stamp="2026-07-31T00:00:00Z")
        rec = ae.parse_record(os.path.join(self.tmp, name))
        self.assertEqual(rec["status"], ae.ARCHIVED)
        self.assertEqual(rec["dataset"], DS)
        self.assertEqual(rec["commit"], ABSENT)
        self.assertNotIn("renamed_to", rec)

    def test_a_rename_is_recorded_as_provenance(self):
        """Reading one repo when the corpus names another is a fact the paper
        should be able to state (#65)."""
        name = ae.write_record(self.tmp, ROW, ae.ARCHIVED, MOVED, "body")
        rec = ae.parse_record(os.path.join(self.tmp, name))
        self.assertEqual(rec["renamed_to"], "elsewhere/a-dataset")

    def test_the_body_is_kept_verbatim_below_the_separator(self):
        body = "# report\nfile changes:\n  (none)"
        name = ae.write_record(self.tmp, ROW, ae.ARCHIVED, REACHABLE, body)
        text = open(os.path.join(self.tmp, name), encoding="utf-8").read()
        self.assertIn(ae.SEP + "\n" + body, text)
        self.assertTrue(text.endswith("\n"))


class UnreachableRecord(unittest.TestCase):
    def record(self, probe=REACHABLE, tree="HTTP 404"):
        return ae.unreachable_record(DS, ABSENT, probe,
                                     ["`git fetch origin` completed; commit "
                                      "still absent"], tree)

    def test_it_denies_the_reading_it_exists_to_prevent(self):
        rec = self.record()
        self.assertIn("NOTHING WAS ANALYZED", rec)
        self.assertIn("not a commit that", rec)

    def test_both_checks_appear_with_their_outcomes(self):
        """A later reader must see what was established, not a bare verdict."""
        rec = self.record()
        self.assertIn("rev-parse", rec)
        self.assertIn("tree endpoint", rec)
        self.assertIn("HTTP 404", rec)
        self.assertIn("fetch origin", rec)

    def test_a_live_repo_whose_tree_404s_is_named_a_rewrite(self):
        self.assertIn("rewritten", self.record(REACHABLE, "HTTP 404"))

    def test_a_gone_repo_is_not_called_a_rewrite(self):
        rec = self.record(GONE, "HTTP 401")
        self.assertNotIn("rewritten", rec)
        self.assertIn(ic.GONE, rec)

    def test_an_unlandable_probe_claims_less(self):
        """A tree check that did not land decides nothing; the record must not
        promote a local miss into a confirmed rewrite."""
        rec = self.record(REACHABLE, "ConnectionError")
        self.assertNotIn("rewritten", rec)
        self.assertIn("does not confirm", rec)


class NoCloneRecord(unittest.TestCase):
    def test_it_carries_the_rater_facing_explanation(self):
        rec = ae.no_clone_record(DS, GONE)
        self.assertIn("NOTHING WAS ANALYZED", rec)
        self.assertIn("deleted or made private", rec)


class FailureAndEmptyDoNotLookAlike(unittest.TestCase):
    """The #72 constraint at the archive layer: the record for a commit that
    exists and touched nothing, and the record for a commit that cannot be
    read, must not coincide in either status or body."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        git(self.tmp, "init", "-q", ".")
        git(self.tmp, "config", "user.email", "t@example.com")
        git(self.tmp, "config", "user.name", "t")
        git(self.tmp, "commit", "-q", "--allow-empty", "-m", "first")
        git(self.tmp, "commit", "-q", "--allow-empty", "-m", "nothing to see")
        self._clone = ic.clone
        ic.clone = lambda ds: self.tmp
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(shutil.rmtree, self.out, True)

    def tearDown(self):
        ic.clone = self._clone

    def empty_commit_report(self):
        sha = subprocess.run(["git", "-C", self.tmp, "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            ic.inspect(DS, sha, False, False)
        return sha, out.getvalue()

    def test_statuses_and_bodies_differ(self):
        sha, report = self.empty_commit_report()
        row = dict(ROW, CommitId=sha)
        kept = ae.write_record(self.out, row, ae.ARCHIVED, REACHABLE, report)
        lost = ae.write_record(self.out, ROW, ae.UNREACHABLE, REACHABLE,
                               ae.unreachable_record(DS, ABSENT, REACHABLE,
                                                     [], "HTTP 404"))
        kept_rec = ae.parse_record(os.path.join(self.out, kept))
        lost_rec = ae.parse_record(os.path.join(self.out, lost))
        self.assertEqual(kept_rec["status"], ae.ARCHIVED)
        self.assertEqual(lost_rec["status"], ae.UNREACHABLE)
        kept_text = open(os.path.join(self.out, kept), encoding="utf-8").read()
        self.assertIn("(none)", kept_text)
        self.assertNotIn("NOTHING WAS ANALYZED", kept_text)


class IndexCoversEveryRow(unittest.TestCase):
    def test_a_row_with_no_record_is_listed_not_dropped(self):
        out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out, True)
        other = dict(ROW, DatasetID="someone/other", CommitId="b" * 40)
        ae.write_record(out, ROW, ae.UNREACHABLE, MOVED, "body")
        ae.build_index([ROW, other], out)
        lines = open(os.path.join(out, ae.INDEX), encoding="utf-8").read()
        self.assertIn(f"{DS},{ABSENT},cleaning,{ae.UNREACHABLE},"
                      f"moved (HTTP 301),elsewhere/a-dataset,", lines)
        self.assertIn("someone/other,bbbb", lines)
        self.assertIn("not-generated", lines)
        self.assertTrue(lines.endswith("\n"))


class ArchiveDatasetFlow(unittest.TestCase):
    """The branch logic over a real local repo, with the Hub stubbed out."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        git(self.tmp, "init", "-q", ".")
        git(self.tmp, "config", "user.email", "t@example.com")
        git(self.tmp, "config", "user.name", "t")
        git(self.tmp, "commit", "-q", "--allow-empty", "-m", "only commit")
        self.sha = subprocess.run(["git", "-C", self.tmp, "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
        self._saved = (ic.repo_status, ic.clone, ae.run_inspect, ae.tree_status)
        ic.repo_status = lambda ds: REACHABLE
        ic.clone = lambda ds: self.tmp
        ae.run_inspect = lambda ds, sha: (ae.ARCHIVED, "# a report\n")
        ae.tree_status = lambda ds, sha: "HTTP 404"
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(shutil.rmtree, self.out, True)

    def tearDown(self):
        ic.repo_status, ic.clone, ae.run_inspect, ae.tree_status = self._saved

    def rows(self, sha):
        return [dict(ROW, CommitId=sha)]

    def run_it(self, sha, force=False):
        out = io.StringIO()
        with redirect_stdout(out):
            return ae.archive_dataset(DS, self.rows(sha), self.out, force,
                                      total=1)

    def test_a_held_commit_is_archived(self):
        self.assertEqual(self.run_it(self.sha), [ae.ARCHIVED])

    def test_a_commit_nobody_holds_is_recorded_unreachable(self):
        """The repo has no origin, so both fetches fail for real; the record
        must still land, carrying what was tried."""
        self.assertEqual(self.run_it(ABSENT), [ae.UNREACHABLE])
        path = os.path.join(self.out, ae.evidence_filename(DS, ABSENT))
        text = open(path, encoding="utf-8").read()
        self.assertIn("fetch origin", text)
        self.assertIn("rewritten", text)

    def test_an_existing_record_is_kept_and_its_status_reported(self):
        self.run_it(ABSENT)
        ae.run_inspect = lambda ds, sha: (_ for _ in ()).throw(
            AssertionError("must not regenerate"))
        self.assertEqual(self.run_it(ABSENT), [ae.UNREACHABLE])


if __name__ == "__main__":
    unittest.main()
