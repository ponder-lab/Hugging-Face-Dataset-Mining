"""Regression tests for the --download path on LFS-tracked files.

These build throwaway git repos whose blobs are literal LFS-pointer text. With
no LFS remote to fetch from, `git lfs smudge` echoes the pointer back and exits
0, which is exactly the situation the helper used to mishandle. No network and
no real LFS server are needed.
"""
import io, os, shutil, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import inspect_commit as ic

POINTER = "version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n"
PLAIN = "a,b,c\n1,2,3\n"


def git(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True,
                   capture_output=True, text=True)


def commit_file(repo, text, msg):
    with open(os.path.join(repo, "data.csv"), "w") as f:
        f.write(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)


def rev_parse(repo, rev="HEAD"):
    return subprocess.run(["git", "-C", repo, "rev-parse", rev],
                          capture_output=True, text=True).stdout.strip()


def status(repo):
    return subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                          capture_output=True, text=True).stdout


class LfsDownloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        git(self.tmp, "init", "-q", ".")
        git(self.tmp, "config", "user.email", "t@example.com")
        git(self.tmp, "config", "user.name", "t")
        self._clone = ic.clone
        ic.clone = lambda ds: self.tmp

    def tearDown(self):
        ic.clone = self._clone
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, parent_text, child_text):
        commit_file(self.tmp, parent_text, "parent")
        commit_file(self.tmp, child_text, "child")
        return rev_parse(self.tmp)

    def run_inspect(self, sha):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            ic.inspect("fake/ds", sha, True, False)
        return out.getvalue(), err.getvalue()

    def test_unfetchable_pointer_is_unresolved_not_none(self):
        """smudge exits 0 and echoes the pointer back; that is not a CSV header."""
        sha = self.build(PLAIN, POINTER.format(oid="dead", size=1))
        self.assertIs(ic.header(self.tmp, sha, "data.csv", True), ic.UNRESOLVED)

    def test_plain_parent_lfs_child_does_not_crash(self):
        """Regression: raised TypeError: 'NoneType' object is not iterable."""
        sha = self.build(PLAIN, POINTER.format(oid="dead", size=1))
        _, err = self.run_inspect(sha)  # must not raise
        self.assertIn("could not fetch", err)

    def test_lfs_both_sides_warns_instead_of_reporting_no_change(self):
        """Regression: printed nothing, which read as 'no column change'."""
        sha = self.build(POINTER.format(oid="1111", size=111),
                         POINTER.format(oid="2222", size=222))
        out, err = self.run_inspect(sha)
        self.assertIn("could not fetch", err)
        self.assertNotIn("column change", out)

    def test_inspect_leaves_the_cached_clone_clean(self):
        """Regression: checkout staged the parent's blob and never restored it."""
        sha = self.build(POINTER.format(oid="1111", size=111),
                         POINTER.format(oid="2222", size=222))
        self.run_inspect(sha)
        self.assertEqual(status(self.tmp), "")

    def test_path_deleted_at_head_leaves_no_untracked_file(self):
        """Regression (Copilot review on #37): restoring via `checkout HEAD --`
        failed when the path no longer exists at HEAD, stranding an untracked
        file. Nothing touches the working tree now."""
        sha = self.build(POINTER.format(oid="1111", size=111),
                         POINTER.format(oid="2222", size=222))
        git(self.tmp, "rm", "-q", "data.csv")
        git(self.tmp, "commit", "-qm", "later: delete data.csv")
        self.run_inspect(sha)
        self.assertEqual(status(self.tmp), "", "stranded a working-tree file")

    def test_absent_blob_reports_absent(self):
        """A path that does not exist at the revision is ABSENT, not UNRESOLVED."""
        sha = self.build(PLAIN, POINTER.format(oid="dead", size=1))
        self.assertIs(ic.header(self.tmp, sha, "no_such_file.csv", True), ic.ABSENT)

    def test_no_download_flag_still_flags_lfs(self):
        """Without --download an LFS file is flagged, not fetched."""
        sha = self.build(PLAIN, POINTER.format(oid="dead", size=1))
        self.assertIs(ic.header(self.tmp, sha, "data.csv", False), ic.NO_DOWNLOAD)

    def test_real_csv_still_diffs_columns(self):
        """The non-LFS path is untouched."""
        sha = self.build("a,b\n1,2\n", "a,b,c\n1,2,3\n")
        out, _ = self.run_inspect(sha)
        self.assertIn("column change", out)
        self.assertIn("'c'", out)


if __name__ == "__main__":
    unittest.main()
