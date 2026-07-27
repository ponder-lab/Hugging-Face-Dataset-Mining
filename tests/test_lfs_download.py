"""Regression tests for the --download path on LFS-tracked files.

These build throwaway git repos whose blobs are literal LFS-pointer text, which
is what a GIT_LFS_SKIP_SMUDGE clone stores. Content for such a blob is fetched
from the Hub's resolve endpoint, so the HTTP layer is mocked here: no network
and no real LFS server are needed.
"""
import gzip, io, os, random, shutil, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import inspect_commit as ic

POINTER = "version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n"
PLAIN = "a,b,c\n1,2,3\n"


def gzipped(data):
    """`data` as a gzip member, the way a .csv.gz blob arrives."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as f:
        f.write(data)
    return buf.getvalue()


def git(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True,
                   capture_output=True, text=True)


def commit_file(repo, text, msg):
    with open(os.path.join(repo, "data.csv"), "w") as f:
        f.write(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)


def commit_files(repo, files, msg):
    """Commit a {name: bytes} map, so a test can name the format under test."""
    for name, data in files.items():
        with open(os.path.join(repo, name), "wb") as f:
            f.write(data)
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


def serving_each(*bodies):
    """Patch the HTTP layer to serve one response per call, in order.

    A single FakeResponse cannot be reused: its body is a stream, and the second
    read of it comes back empty.
    """
    return mock.patch.object(ic.requests, "get",
                             side_effect=[FakeResponse(b) for b in bodies])


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

    def run_inspect(self, sha, download=True, show_rows=False):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            ic.inspect("fake/ds", sha, download, show_rows)
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

    def test_newline_free_payload_is_capped_and_not_tabular(self):
        """A blob that never emits a newline (e.g. binary/parquet) is bounded by
        the cap and reported not_tabular, not crashed. Its 1 MiB single 'field'
        would otherwise trip csv's field-size limit inside inspect()."""
        response = FakeResponse(b"z" * (4 * ic.HEADER_READ_CAP), 200)
        with mock.patch.object(ic.requests, "get", return_value=response):
            result = ic.fetch_header("fake/ds", "abc123", "data.csv")
        self.assertLessEqual(response.consumed, ic.HEADER_READ_CAP + 8192)
        self.assertIsInstance(result, ic.Unread)
        self.assertEqual(result.kind, "not_tabular")

    def test_truncated_header_is_refused_rather_than_parsed(self):
        """#55: a read cut short with no line break holds part of a first line.
        Parsing it would compare two truncated headers, find them equal, and
        report an unchanged column set: the #48 trap in its most convincing
        form, since the truncated header parses cleanly."""
        body = b",".join(b"col%d" % i for i in range(4 * ic.HEADER_READ_CAP))
        response = FakeResponse(body, 200)
        with mock.patch.object(ic.requests, "get", return_value=response):
            result = ic.fetch_header("fake/ds", "abc123", "data.csv")
        self.assertIsInstance(result, ic.Unread)
        self.assertEqual(result.kind, "not_tabular")
        self.assertIn("no line break", result.detail)
        self.assertIn("no line break", ic.explain(result, "data.csv", "abc123"))

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


class RowSampleTest(LfsDownloadTest):
    """#52: --show_rows must read the content, not the pointer sitting in git.

    The clone is made with GIT_LFS_SKIP_SMUDGE, so for an LFS-tracked file the
    local blob is pointer text. Diffing that locally printed an oid and a size;
    where smudge could still reach a classic LFS object it instead pulled the
    entire payload, which is the download --download exists to bound.
    """

    def lfs_pair(self):
        """A commit whose parent and child are both LFS pointers of equal columns."""
        return self.build(POINTER.format(oid="1111", size=111),
                          POINTER.format(oid="2222", size=222))

    def test_rows_come_from_the_hub_not_the_local_pointer(self):
        sha = self.lfs_pair()
        header = b"a,b,c\n"
        with serving_each(header, header,
                          header + b"1,2,3\n", header + b"9,9,9\n"):
            out, err = self.run_inspect(sha, show_rows=True)
        self.assertIn("9,9,9", out)
        self.assertNotIn("oid sha256", out, "printed the pointer instead of the rows")
        self.assertNotIn("diff --git", out, "shelled out to git diff for an LFS blob")
        self.assertEqual(err, "")

    def test_no_ansi_escapes_in_the_output(self):
        """Output is read through a pipe and pasted into a sheet as often as it is
        read on a terminal, so it carries no color."""
        sha = self.lfs_pair()
        header = b"a,b,c\n"
        with serving_each(header, header,
                          header + b"1,2,3\n", header + b"9,9,9\n"):
            out, _ = self.run_inspect(sha, show_rows=True)
        self.assertNotIn("\x1b[", out)

    def test_no_blank_line_between_every_line(self):
        sha = self.lfs_pair()
        header = b"a,b,c\n"
        with serving_each(header, header,
                          header + b"1,2,3\n", header + b"9,9,9\n"):
            out, _ = self.run_inspect(sha, show_rows=True)
        self.assertNotIn("\n\n+9,9,9", out)
        self.assertNotIn("+9,9,9\n\n", out)

    def test_identical_sample_says_so_rather_than_printing_nothing(self):
        """The #48 trap one level down: an empty diff of the head reads as 'no
        change' when it only means 'no change this far into the file'."""
        sha = self.lfs_pair()
        same = b"a,b,c\n1,2,3\n"
        with serving_each(b"a,b,c\n", b"a,b,c\n", same, same):
            out, _ = self.run_inspect(sha, show_rows=True)
        self.assertIn("identical at both revisions", out)
        self.assertIn("further into the file", out)

    def test_pointer_size_delta_is_reported_without_fetching(self):
        """The sizes are in the pointers git already holds, so a changed payload
        can be shown even when the head sample matches."""
        sha = self.lfs_pair()
        same = b"a,b,c\n1,2,3\n"
        with serving_each(b"a,b,c\n", b"a,b,c\n", same, same):
            out, _ = self.run_inspect(sha, show_rows=True)
        self.assertIn("payload size 111 -> 222 bytes (+111)", out)
        self.assertIn("the data did change", out)

    def test_equal_sizes_are_not_claimed_as_equal_content(self):
        """A shuffle keeps the byte count identical."""
        sha = self.build(POINTER.format(oid="1111", size=555),
                         POINTER.format(oid="2222", size=555))
        with serving_each(b"a,b\n", b"a,b\n", b"a,b\n1,2\n", b"a,b\n3,4\n"):
            out, _ = self.run_inspect(sha, show_rows=True)
        self.assertIn("payload size unchanged at 555 bytes", out)
        self.assertIn("equal size is not equal content", out)

    def test_row_sample_asks_for_only_the_sample(self):
        """A ranged request, so a 10 MB CSV costs a few kilobytes to sample."""
        sha = self.lfs_pair()
        header = b"a,b\n"
        big = header + b"x,y\n" * (4 * ic.ROW_READ_CAP)
        responses = [FakeResponse(header), FakeResponse(header),
                     FakeResponse(big), FakeResponse(big)]
        with mock.patch.object(ic.requests, "get", side_effect=responses):
            self.run_inspect(sha, show_rows=True)
        for r in responses[2:]:
            self.assertLess(r.consumed, ic.ROW_READ_CAP,
                            f"streamed {r.consumed} bytes for a {ic.ROW_SAMPLE_LINES}"
                            f"-line sample")

    def test_sample_is_capped_at_the_declared_number_of_lines(self):
        sha = self.lfs_pair()
        header = b"a,b\n"
        body = header + b"".join(b"%d,%d\n" % (i, i) for i in range(200))
        with serving_each(header, header, body, body + b"9,9\n"):
            out, _ = self.run_inspect(sha, show_rows=True)
        self.assertIn(f"first {ic.ROW_SAMPLE_LINES} lines", out)
        self.assertNotIn("100,100", out)

    def test_unreadable_side_warns_and_shows_no_rows(self):
        sha = self.lfs_pair()
        responses = [FakeResponse(b"a,b\n"), FakeResponse(b"a,b\n"),
                     FakeResponse(b"", 404), FakeResponse(b"a,b\n1,2\n")]
        with mock.patch.object(ic.requests, "get", side_effect=responses):
            out, err = self.run_inspect(sha, show_rows=True)
        self.assertIn("rows not shown", err)
        self.assertNotIn("head sample", out)

    def test_lfs_rows_without_the_download_flag_touch_no_network(self):
        """--show_rows does not become a back door around --download: the file is
        flagged as needing one, and nothing is fetched."""
        sha = self.lfs_pair()
        with mock.patch.object(ic.requests, "get",
                               side_effect=AssertionError("fetched")) as get:
            out, _ = self.run_inspect(sha, download=False, show_rows=True)
        get.assert_not_called()
        self.assertIn("download a version to inspect", out)
        self.assertNotIn("head sample", out)

    def test_changed_column_set_still_gets_a_row_sample(self):
        """Passing --show_rows and getting silence back is its own defect."""
        sha = self.lfs_pair()
        with serving_each(b"a,b,c\n", b"a,b\n", b"a,b\n1,2\n", b"a,b,c\n1,2,3\n"):
            out, _ = self.run_inspect(sha, show_rows=True)
        self.assertIn("column change:", out)
        self.assertIn("head sample", out)

    def test_plain_file_diffs_locally_without_the_network(self):
        """A file git actually holds needs no HTTP at all."""
        sha = self.build("a,b\n1,2\n", "a,b\n9,9\n")
        with mock.patch.object(ic.requests, "get",
                               side_effect=AssertionError("fetched")):
            out, _ = self.run_inspect(sha, show_rows=True)
        self.assertIn("diff --git", out)
        self.assertIn("+9,9", out)
        self.assertNotIn("\x1b[", out)

    def test_long_local_diff_reports_what_it_cut(self):
        """Silent truncation reads as a short diff, a different claim."""
        rows = "".join(f"{i},{i}\n" for i in range(200))
        sha = self.build("a,b\n" + rows, "a,b\n" + rows.replace(",", ",9"))
        with mock.patch.object(ic.requests, "get",
                               side_effect=AssertionError("fetched")):
            out, _ = self.run_inspect(sha, show_rows=True)
        self.assertIn("further lines not shown", out)
        self.assertIn(f"cap is {ic.DIFF_MAX_LINES}", out)


class FormatCoverageTest(LfsDownloadTest):
    """#55: three commits were reported as LFS-opaque that were nothing of the
    kind. The analysis loop gated on a `.csv` suffix, so a `.tsv` sitting in git
    in plain sight and a `.csv.gz` whose content the Hub serves on request were
    both passed over, and passed over in silence.
    """

    def build_files(self, parent, child):
        commit_files(self.tmp, parent, "parent")
        commit_files(self.tmp, child, "child")
        return rev_parse(self.tmp)

    def offline(self):
        """Assert no HTTP happens at all: git already holds this content."""
        return mock.patch.object(ic.requests, "get",
                                 side_effect=AssertionError("fetched"))

    def test_tsv_column_removal_is_reported(self):
        """oskarvanderwal/SHADR: an unnamed pandas index column dropped from a
        plain .tsv that needs no download whatsoever."""
        sha = self.build_files({"test.tsv": b"\tindex\tinput\n0\t0\tx\n"},
                               {"test.tsv": b"index\tinput\n0\tx\n"})
        with self.offline():
            out, _ = self.run_inspect(sha)
        self.assertIn("column change", out)
        self.assertIn("removed: ['']", out)

    def test_tsv_columns_are_split_on_tabs_not_commas(self):
        """A TSV read with a comma delimiter is one column wide at both
        revisions, which compares equal and reads as an unchanged column set."""
        sha = self.build_files({"test.tsv": b"a\tb\n1\t2\n"},
                               {"test.tsv": b"a\tb\tc\n1\t2\t3\n"})
        with self.offline():
            out, _ = self.run_inspect(sha)
        self.assertIn("added:   ['c']", out)
        self.assertNotIn("no column-set change", out)

    def test_gzipped_csv_held_by_git_is_read_without_the_network(self):
        """Compression is not opacity: git has the bytes, they just need
        inflating before a header can be read out of them."""
        sha = self.build_files(
            {"train.csv.gz": gzipped(b"id,text,sentiment_group\n1,x,g\n")},
            {"train.csv.gz": gzipped(b"id,text\n1,x\n")})
        with self.offline():
            out, _ = self.run_inspect(sha)
        self.assertIn("removed: ['sentiment_group']", out)

    def test_gzipped_lfs_csv_is_read_over_the_hub(self):
        """agentlans/twitter-sentiment-meta-analysis: LFS-tracked and gzipped.
        The resolve endpoint serves it; the suffix gate is what hid it."""
        sha = self.build_files(
            {"train.csv.gz": POINTER.format(oid="1111", size=111).encode()},
            {"train.csv.gz": POINTER.format(oid="2222", size=222).encode()})
        # The child header is read first, then the parent's.
        with serving_each(gzipped(b"id,text\n1,x\n"),
                          gzipped(b"id,text,sentiment_group\n1,x,g\n")):
            out, err = self.run_inspect(sha)
        self.assertIn("removed: ['sentiment_group']", out)
        self.assertEqual(err, "")

    def test_gzipped_row_sample_shows_rows_not_compressed_bytes(self):
        """`git diff` on a gzipped blob reports only that the bytes differ, so
        --show_rows goes through the same decompressing read either way."""
        sha = self.build_files({"train.csv.gz": gzipped(b"a,b\n1,2\n")},
                               {"train.csv.gz": gzipped(b"a,b\n9,9\n")})
        with self.offline():
            out, _ = self.run_inspect(sha, show_rows=True)
        self.assertIn("+9,9", out)
        self.assertNotIn("Binary files", out)
        self.assertNotIn("\x1b[", out)

    def test_corrupt_gzip_from_the_hub_is_a_format_finding(self):
        """Bytes that claim to be gzip and are not: a format outcome, not a
        transport one, so it is terminal and not retried."""
        sha = self.build_files(
            {"train.csv.gz": POINTER.format(oid="1111", size=111).encode()},
            {"train.csv.gz": POINTER.format(oid="2222", size=222).encode()})
        with serving(ic.GZIP_MAGIC + b"not really gzip" * 40) as get:
            _, err = self.run_inspect(sha)
        self.assertIn("delimited text", err)
        self.assertIn("corrupt gzip stream", err)
        self.assertEqual(get.call_count, 1)

    def test_gzip_spanning_several_chunks_loses_nothing_at_the_seams(self):
        """The stream is decompressed chunk by chunk, so a header longer than one
        chunk is reassembled across iterations. Incompressible bytes force the
        multi-chunk path: a header that fits in one chunk would not exercise it,
        and bytes dropped at a seam would still parse, just into wrong columns."""
        rnd = random.Random(0).randbytes(64 * 1024)
        wide = bytes(b for b in rnd if b not in (10, ord(",")))  # one long field
        body = gzipped(wide + b"\n" + b"second line\n")
        self.assertGreater(len(body), 4 * 8192, "payload fits in one chunk")
        with mock.patch.object(ic.requests, "get",
                               return_value=FakeResponse(body, 206)):
            lines = ic._fetch("fake/ds", "abc123", "d.csv.gz", 1, ic.HEADER_READ_CAP)
        self.assertEqual(lines, [wide.decode("utf-8", errors="replace")])

    def test_unread_format_is_listed_rather_than_passed_over(self):
        """The defect behind all three reports: a modified file the tool never
        opened left exactly the impression of one it opened and found clean."""
        sha = self.build_files({"data.parquet": b"PAR1\x00old\x00PAR1"},
                               {"data.parquet": b"PAR1\x00new\x00PAR1"})
        with self.offline():
            out, _ = self.run_inspect(sha, show_rows=True)
        self.assertIn("not analyzed:", out)
        self.assertIn("data.parquet", out)
        self.assertIn("whether the contents of these changed", out)

    def test_read_and_unread_formats_are_both_reported(self):
        """One commit touching both: the CSV is analyzed and the parquet is
        named, so neither result is inferred from the other's presence."""
        sha = self.build_files(
            {"data.csv": b"a,b\n1,2\n", "data.parquet": b"PAR1\x00old"},
            {"data.csv": b"a,b,c\n1,2,3\n", "data.parquet": b"PAR1\x00new"})
        with self.offline():
            out, _ = self.run_inspect(sha)
        self.assertIn("added:   ['c']", out)
        self.assertIn("not analyzed:", out)
        self.assertIn("data.parquet", out)
        self.assertNotIn("data.csv   [format", out)

    def test_nothing_is_claimed_unread_when_every_file_was_read(self):
        sha = self.build_files({"data.csv": b"a,b\n1,2\n"},
                               {"data.csv": b"a,b,c\n1,2,3\n"})
        with self.offline():
            out, _ = self.run_inspect(sha)
        self.assertNotIn("not analyzed", out)


if __name__ == "__main__":
    unittest.main()
