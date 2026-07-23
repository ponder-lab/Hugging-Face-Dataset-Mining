"""Regression tests for the --download path on LFS-tracked files.

These build throwaway git repos whose blobs are literal LFS-pointer text, which
is what a GIT_LFS_SKIP_SMUDGE clone stores. Content for such a blob is fetched
from the Hub's resolve endpoint, so the HTTP layer is mocked here: no network
and no real LFS server are needed.
"""
import io, os, shutil, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock

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


class FakeResponse:
    """Minimal stand-in for a streamed requests response.

    Counts the bytes handed out so a test can assert we stop at the header
    instead of pulling a whole dataset down.
    """

    def __init__(self, body=b"", status_code=206):
        self.status_code = status_code
        self._body = io.BytesIO(body)
        self.consumed = 0

    def iter_content(self, chunk_size):
        while True:
            chunk = self._body.read(chunk_size)
            if not chunk:
                return
            self.consumed += len(chunk)
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def serving(body=b"", status_code=206):
    """Patch the HTTP layer to serve one response; yields the mocked get."""
    return mock.patch.object(ic.requests, "get",
                             return_value=FakeResponse(body, status_code))


class LfsDownloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        git(self.tmp, "init", "-q", ".")
        git(self.tmp, "config", "user.email", "t@example.com")
        git(self.tmp, "config", "user.name", "t")
        self._clone = ic.clone
        ic.clone = lambda ds: self.tmp
        # Retries back off with time.sleep; no test should wait on a real clock.
        sleep = mock.patch.object(ic.time, "sleep")
        sleep.start()
        self.addCleanup(sleep.stop)

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

    def test_pointer_content_is_fetched_by_revision(self):
        """#44: a Xet-backed blob has no classic LFS object, but the resolve
        endpoint still serves it. The pointer blob must not stop us."""
        sha = self.build(PLAIN, POINTER.format(oid="dead", size=1))
        with serving(b"x,y,z\n1,2,3\n") as get:
            self.assertEqual(ic.header("fake/ds", self.tmp, sha, "data.csv", True),
                             ["x", "y", "z"])
        url = get.call_args.args[0]
        self.assertIn("/datasets/fake/ds/resolve/", url)
        self.assertTrue(url.endswith(f"/{sha}/data.csv"), url)

    def test_only_the_header_is_requested(self):
        """A Range request, so we never ask the CDN for the whole payload."""
        sha = self.build(PLAIN, POINTER.format(oid="dead", size=1))
        with serving(b"x,y\n") as get:
            ic.header("fake/ds", self.tmp, sha, "data.csv", True)
        rng = get.call_args.kwargs["headers"]["Range"]
        self.assertEqual(rng, f"bytes=0-{ic.HEADER_READ_CAP - 1}")
        self.assertTrue(get.call_args.kwargs["stream"])
        self.assertTrue(get.call_args.kwargs["timeout"])

    def test_header_read_stops_at_newline_not_at_cap(self):
        """Even if the server ignores the range, reading stops at the header."""
        header = b"a,b,c\n"
        response = FakeResponse(header + b"0" * (4 * ic.HEADER_READ_CAP), 200)
        with mock.patch.object(ic.requests, "get", return_value=response):
            result = ic.fetch_header("fake/ds", "abc123", "data.csv")
        self.assertEqual(result, ["a", "b", "c"])
        self.assertLess(response.consumed, ic.HEADER_READ_CAP,
                        f"streamed {response.consumed} bytes for a header")

    def test_newline_free_payload_is_capped_and_not_csv(self):
        """A blob that never emits a newline (e.g. binary/parquet) is bounded by
        the cap and reported not_csv, not crashed. Its 1 MiB single 'field' would
        otherwise trip csv's field-size limit inside inspect()."""
        response = FakeResponse(b"z" * (4 * ic.HEADER_READ_CAP), 200)
        with mock.patch.object(ic.requests, "get", return_value=response):
            result = ic.fetch_header("fake/ds", "abc123", "data.csv")
        self.assertLessEqual(response.consumed, ic.HEADER_READ_CAP + 8192)
        self.assertIsInstance(result, ic.Unread)
        self.assertEqual(result.kind, "not_csv")

    def test_gated_status_is_access_and_not_retried(self):
        """401/403 means private or gated: a distinct, terminal disposition."""
        with serving(b"", 403) as get:
            result = ic.fetch_header("fake/ds", "abc123", "data.csv")
        self.assertEqual(result.kind, "access")
        self.assertFalse(result.retryable)
        self.assertEqual(get.call_count, 1)

    def test_missing_blob_404_is_content_absent_and_not_retried(self):
        """404 means no bytes for this revision: terminal, a retry cannot help."""
        with serving(b"", 404) as get:
            result = ic.fetch_header("fake/ds", "abc123", "data.csv")
        self.assertEqual(result.kind, "content_absent")
        self.assertFalse(result.retryable)
        self.assertEqual(get.call_count, 1)

    def test_server_error_is_transport_and_is_retried(self):
        """5xx is a transient transport error: retried with backoff before giving
        up, then reported as a retryable disposition (#47)."""
        with serving(b"", 503) as get:
            result = ic.fetch_header("fake/ds", "abc123", "data.csv")
        self.assertEqual(result.kind, "transport")
        self.assertTrue(result.retryable)
        self.assertEqual(get.call_count, ic.RETRIES)

    def test_network_failure_is_transport_and_is_retried(self):
        """A dropped connection is retried, then surfaces as retryable transport,
        never taking inspect() down with it."""
        with mock.patch.object(ic.requests, "get",
                               side_effect=ic.requests.RequestException("boom")) as get:
            result = ic.fetch_header("fake/ds", "abc123", "data.csv")
        self.assertEqual(result.kind, "transport")
        self.assertTrue(result.retryable)
        self.assertEqual(get.call_count, ic.RETRIES)

    def test_pointer_echoed_back_is_content_absent_not_none(self):
        """If the endpoint hands back the pointer text, that is not a header: the
        object was never uploaded for this revision (#47), a terminal outcome."""
        with serving(POINTER.format(oid="dead", size=1).encode()) as get:
            result = ic.fetch_header("fake/ds", "abc123", "data.csv")
        self.assertEqual(result.kind, "content_absent")
        self.assertFalse(result.retryable)
        self.assertEqual(get.call_count, 1)

    def test_token_is_sent_when_the_environment_has_one(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_secret"}):
            with serving(b"a,b\n") as get:
                ic.fetch_header("fake/ds", "abc123", "data.csv")
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"],
                         "Bearer hf_secret")

    def test_no_token_header_without_one(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")}
        with mock.patch.dict(os.environ, env, clear=True):
            with serving(b"a,b\n") as get:
                ic.fetch_header("fake/ds", "abc123", "data.csv")
        self.assertNotIn("Authorization", get.call_args.kwargs["headers"])

    def test_plain_parent_lfs_child_does_not_crash(self):
        """Regression: raised TypeError: 'NoneType' object is not iterable."""
        sha = self.build(PLAIN, POINTER.format(oid="dead", size=1))
        with serving(b"", 404):
            _, err = self.run_inspect(sha)  # must not raise
        self.assertIn("no content", err)
        self.assertIn("retry will not help", err)

    def test_lfs_both_sides_warns_instead_of_reporting_no_change(self):
        """Regression: printed nothing, which read as 'no column change'."""
        sha = self.build(POINTER.format(oid="1111", size=111),
                         POINTER.format(oid="2222", size=222))
        with serving(b"", 404):
            out, err = self.run_inspect(sha)
        self.assertIn("no content", err)
        self.assertNotIn("column change", out)

    def test_modified_csv_with_same_columns_says_so(self):
        """#48: an unchanged column set is stated explicitly, not left as silence
        that reads the same as 'the tool did not look'."""
        sha = self.build("a,b,c\n1,2,3\n", "a,b,c\n9,9,9\n")
        with mock.patch.object(ic.requests, "get",
                               side_effect=AssertionError("fetched")):
            out, _ = self.run_inspect(sha)
        self.assertIn("no column-set change", out)
        self.assertNotIn("column change:", out)

    def test_unfetchable_parent_warns_and_skips_the_diff(self):
        """Child resolves, parent does not: say so rather than diffing halves."""
        sha = self.build(POINTER.format(oid="1111", size=111),
                         POINTER.format(oid="2222", size=222))
        responses = [FakeResponse(b"a,b,c\n"), FakeResponse(b"", 404)]
        with mock.patch.object(ic.requests, "get", side_effect=responses):
            out, err = self.run_inspect(sha)
        self.assertIn("could not read parent columns", err)
        self.assertNotIn("column change", out)

    def test_inspect_leaves_the_cached_clone_clean(self):
        """Regression: checkout staged the parent's blob and never restored it."""
        sha = self.build(POINTER.format(oid="1111", size=111),
                         POINTER.format(oid="2222", size=222))
        with serving(b"", 404):
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
        with serving(b"", 404):
            self.run_inspect(sha)
        self.assertEqual(status(self.tmp), "", "stranded a working-tree file")

    def test_absent_blob_reports_absent(self):
        """A path that does not exist at the revision is ABSENT, not UNRESOLVED."""
        sha = self.build(PLAIN, POINTER.format(oid="dead", size=1))
        self.assertIs(ic.header("fake/ds", self.tmp, sha, "no_such_file.csv", True),
                      ic.ABSENT)

    def test_no_download_flag_does_not_hit_the_network(self):
        """Without --download an LFS file is flagged, not fetched."""
        sha = self.build(PLAIN, POINTER.format(oid="dead", size=1))
        with mock.patch.object(ic.requests, "get",
                               side_effect=AssertionError("fetched")) as get:
            self.assertIs(ic.header("fake/ds", self.tmp, sha, "data.csv", False),
                          ic.NO_DOWNLOAD)
        get.assert_not_called()

    def test_real_csv_still_diffs_columns(self):
        """The non-LFS path is untouched, and needs no network."""
        sha = self.build("a,b\n1,2\n", "a,b,c\n1,2,3\n")
        with mock.patch.object(ic.requests, "get",
                               side_effect=AssertionError("fetched")):
            out, _ = self.run_inspect(sha)
        self.assertIn("column change", out)
        self.assertIn("'c'", out)


if __name__ == "__main__":
    unittest.main()
