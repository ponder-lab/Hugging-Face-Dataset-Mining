import gzip, io, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import inspect_commit as ic


def gzipped(data):
    """`data` as a gzip member, the way a .csv.gz blob arrives."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as f:
        f.write(data)
    return buf.getvalue()


class TestLfsStatus(unittest.TestCase):
    def _patch_blob(self, blob):
        self._orig = ic.show_blob
        ic.show_blob = lambda *a: blob

    def tearDown(self):
        if hasattr(self, "_orig"):
            ic.show_blob = self._orig

    def test_pointer_is_lfs(self):
        self._patch_blob(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
        self.assertEqual(ic.lfs_status("repo", "rev", "f.csv"), "lfs")

    def test_real_content_is_plain(self):
        self._patch_blob(b"a,b,c\n1,2,3\n")
        self.assertEqual(ic.lfs_status("repo", "rev", "f.csv"), "plain")

    def test_absent_blob_is_none(self):
        self._patch_blob(None)
        self.assertIsNone(ic.lfs_status("repo", "rev", "missing.csv"))

    def test_undecodable_payload_is_plain_not_a_crash(self):
        """A gzipped or binary blob is bytes that are not valid UTF-8. It is by
        that fact not a pointer, and must not blow up the check."""
        self._patch_blob(b"\x1f\x8b\x08\x00\xff\xfe\xfd")
        self.assertEqual(ic.lfs_status("repo", "rev", "f.csv.gz"), "plain")


class TestParsing(unittest.TestCase):
    def test_lfs_pointer_detected(self):
        ptr = "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12\n"
        self.assertTrue(ic.is_lfs_pointer(ptr))
        self.assertIsNone(ic.parse_header(ptr))   # data absent for LFS files

    def test_plain_csv_header(self):
        self.assertFalse(ic.is_lfs_pointer("a,b,c\n1,2,3\n"))
        self.assertEqual(ic.parse_header('"Name","Prompt",Votes\nx,y,z'),
                         ["Name", "Prompt", "Votes"])

    def test_empty_file(self):
        self.assertEqual(ic.parse_header(""), [])

    def test_tab_delimited_header(self):
        """#55: a TSV split on commas comes back as one column, which then
        compares equal at both revisions and reads as an unchanged column set."""
        self.assertEqual(ic.parse_header("index\tinput\toutput\n0\tx\ty", "\t"),
                         ["index", "input", "output"])
        self.assertEqual(ic.parse_header("index\tinput\toutput\n0\tx\ty"),
                         ["index\tinput\toutput"])

    def test_comma_inside_a_tsv_field_is_not_a_separator(self):
        self.assertEqual(ic.parse_header("a,b\tc\n1\t2", "\t"), ["a,b", "c"])


class TestDelimiterForPath(unittest.TestCase):
    """#55: the format a path names, which decides whether it is read at all."""

    def test_known_formats(self):
        self.assertEqual(ic.delimiter("train.csv"), ",")
        self.assertEqual(ic.delimiter("test.tsv"), "\t")
        self.assertEqual(ic.delimiter("data.tab"), "\t")

    def test_gz_names_the_compression_not_the_format(self):
        self.assertEqual(ic.delimiter("train.csv.gz"), ",")
        self.assertEqual(ic.delimiter("test.tsv.gz"), "\t")

    def test_case_is_ignored(self):
        self.assertEqual(ic.delimiter("TRAIN.CSV.GZ"), ",")

    def test_nested_path_reads_its_own_suffix(self):
        self.assertEqual(
            ic.delimiter("test/humap3_test_feature_matrix_20220625.csv.gz"), ",")

    def test_unread_formats_are_none(self):
        for path in ("data.parquet", "sheet.xlsx", "notes.md", "archive.gz",
                     "model.bin", "csv_notes.txt"):
            self.assertIsNone(ic.delimiter(path), path)

    def test_compression_is_recognized_by_suffix(self):
        self.assertTrue(ic.is_compressed("train.csv.gz"))
        self.assertFalse(ic.is_compressed("train.csv"))


class TestGunzipHead(unittest.TestCase):
    def test_plain_bytes_pass_through(self):
        self.assertEqual(ic.gunzip_head(b"a,b,c\n", 1024), b"a,b,c\n")

    def test_gzip_is_decompressed(self):
        self.assertEqual(ic.gunzip_head(gzipped(b"a,b,c\n1,2,3\n"), 1024),
                         b"a,b,c\n1,2,3\n")

    def test_truncated_stream_yields_what_it_reached(self):
        """The normal case: we hold the head of the compressed bytes only."""
        body = b"".join(b"%d,%d\n" % (i, i) for i in range(5000))
        head = ic.gunzip_head(gzipped(b"a,b\n" + body)[:512], 1 << 20)
        self.assertTrue(head.startswith(b"a,b\n"))
        self.assertLess(len(head), len(body))

    def test_output_is_bounded_by_the_cap(self):
        """A small compressed payload must not expand without limit."""
        self.assertEqual(len(ic.gunzip_head(gzipped(b"x" * (1 << 20)), 64)), 64)

    def test_corrupt_stream_yields_nothing(self):
        self.assertEqual(ic.gunzip_head(ic.GZIP_MAGIC + b"garbage" * 10, 1024), b"")

if __name__ == "__main__":
    unittest.main()
