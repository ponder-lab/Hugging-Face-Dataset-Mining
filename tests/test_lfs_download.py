"""Regression tests for the --download path on LFS-tracked files.

These build throwaway git repos whose blobs are literal LFS-pointer text, so
`git lfs pull` finds no remote and materializes nothing. That is exactly the
situation the helper used to mishandle, and it needs no network and no real
LFS remote to reproduce.
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


def build_repo(tmp, parent_text, child_text):
    """Two commits touching data.csv; returns the child sha."""
    git(tmp, "init", "-q", ".")
    git(tmp, "config", "user.email", "t@example.com")
    git(tmp, "config", "user.name", "t")
    for text, msg in ((parent_text, "parent"), (child_text, "child")):
        with open(os.path.join(tmp, "data.csv"), "w") as f:
            f.write(text)
        git(tmp, "add", "-A")
        git(tmp, "commit", "-qm", msg)
    return subprocess.run(["git", "-C", tmp, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


class LfsDownloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._clone = ic.clone
        ic.clone = lambda ds: self.tmp

    def tearDown(self):
        ic.clone = self._clone
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_inspect(self, sha):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            ic.inspect("fake/ds", sha, True, False)
        return out.getvalue(), err.getvalue()

    def test_unmaterialized_pointer_is_unresolved_not_none(self):
        """A pointer that lfs pull cannot fetch must not parse as a CSV header."""
        sha = build_repo(self.tmp, PLAIN, POINTER.format(oid="dead", size=1))
        self.assertIs(ic.header(self.tmp, sha, "data.csv", True), ic.UNRESOLVED)

    def test_plain_parent_lfs_child_does_not_crash(self):
        """Regression: used to raise TypeError: 'NoneType' object is not iterable."""
        sha = build_repo(self.tmp, PLAIN, POINTER.format(oid="dead", size=1))
        _, err = self.run_inspect(sha)  # must not raise
        self.assertIn("not materialized", err)

    def test_lfs_both_sides_warns_instead_of_reporting_no_change(self):
        """Regression: used to print nothing, reading as 'no column change'."""
        sha = build_repo(self.tmp,
                         POINTER.format(oid="1111", size=111),
                         POINTER.format(oid="2222", size=222))
        out, err = self.run_inspect(sha)
        self.assertIn("not materialized", err)
        self.assertNotIn("column change", out)

    def test_inspect_leaves_the_cached_clone_clean(self):
        """Regression: checkout staged the parent's blob and never restored it."""
        sha = build_repo(self.tmp,
                         POINTER.format(oid="1111", size=111),
                         POINTER.format(oid="2222", size=222))
        self.run_inspect(sha)
        status = subprocess.run(["git", "-C", self.tmp, "status", "--porcelain"],
                                capture_output=True, text=True).stdout
        self.assertEqual(status, "", f"clone left dirty: {status!r}")

    def test_absent_blob_reports_absent(self):
        """checkout fails and lfs pull succeeds: must be ABSENT, not a parsed pointer."""
        sha = build_repo(self.tmp, PLAIN, POINTER.format(oid="dead", size=1))
        self.assertIs(ic.load_lfs_pointer(self.tmp, "no_such_file.csv", sha), ic.ABSENT)

    def test_real_csv_still_diffs_columns(self):
        """The non-LFS path is untouched."""
        sha = build_repo(self.tmp, "a,b\n1,2\n", "a,b,c\n1,2,3\n")
        out, _ = self.run_inspect(sha)
        self.assertIn("column change", out)
        self.assertIn("'c'", out)


if __name__ == "__main__":
    unittest.main()
