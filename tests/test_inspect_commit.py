import csv, gzip, io, os, sys, unittest
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


class TestSkipReason(unittest.TestCase):
    """#84: why a modified file went unopened, which the "not analyzed" list
    states out loud. Git hands over a path holding non-ASCII bytes escaped and
    wrapped in double quotes unless told otherwise, and that stand-in fails the
    suffix test for a reason that is not about the file's format at all. Calling
    it an unread format is a false statement about a table this tool reads
    perfectly well once it is handed the real name, and it sends whoever follows
    the pointer to #43, which is about formats and has nothing to do with it."""

    def test_a_genuinely_unread_format_still_points_at_43(self):
        for path in ("notes.md", "data.parquet", "model.bin"):
            self.assertIn("#43", ic.skip_reason(path), path)

    def test_an_escaped_name_is_not_reported_as_a_format(self):
        escaped = r'"\345\274\200\345\205\263/train/0000.csv"'
        reason = ic.skip_reason(escaped)
        self.assertIn("#84", reason)
        self.assertNotIn("#43", reason)
        self.assertNotIn("format", reason)

    def test_a_real_non_ascii_name_is_not_mistaken_for_an_escaped_one(self):
        # Decoded, the path reads normally and never reaches the skipped list;
        # if it somehow does, the format pointer is the honest one.
        self.assertIn("#43", ic.skip_reason("开关/train/notes.md"))

    def test_a_quote_must_open_and_close_to_count_as_escaping(self):
        # A name that merely contains a quote is not git's escaped form.
        self.assertIn("#43", ic.skip_reason('say"what.md'))


class TestSniffDelimiter(unittest.TestCase):
    """#79: the separator a revision is actually written with, which the file
    name only claims. A `.csv` that separates on `;` read on commas splits the
    header inside its quoted fields, and the column diff then compares a sound
    reading of one revision against a shredded reading of the other."""

    # The commit that found it: djulian13/east-slavic-swadesh-lists @ a4c5651f,
    # message "Split by ;". The column names contain commas, which is the reason
    # the author moved off them.
    SEMI = ['Concept;"Meɡra (North, Russian, Russia)";"Beloɡornoje (South, Russian, Russia)"',
            "eye;ɡlˠas;ɡlˠas", "ear;uxo;uxo"]
    COMMA = ['Concept,"Meɡra (North, Russian, Russia)","Beloɡornoje (South, Russian, Russia)"',
             "eye,ɡlˠas,ɡlˠas", "ear,uxo,uxo"]

    def test_semicolon_csv_is_not_read_on_commas(self):
        self.assertEqual(ic.sniff_delimiter(self.SEMI, ","), ";")

    def test_the_same_file_before_the_split_still_reads_on_commas(self):
        """The parent of that commit. Both revisions are sniffed, because the
        change of separator is the finding and needs a before to be one."""
        self.assertEqual(ic.sniff_delimiter(self.COMMA, ","), ",")

    def test_semicolons_without_quoting_are_found_too(self):
        """No commas anywhere to shred the header: read on commas this file is
        one column wide at both revisions, which compares equal and reads as an
        unchanged column set (the #48 trap, arrived at through #55's door)."""
        self.assertEqual(ic.sniff_delimiter(["a;b;c", "1;2;3"], ","), ";")

    def test_a_coherent_name_is_not_talked_out_of_its_delimiter(self):
        """A .tsv whose fields contain commas: commas are the more popular
        character, and tabs are still what separates the fields."""
        self.assertEqual(ic.sniff_delimiter(["a,b\tc", "1,2\t3", "x,y\tz"], "\t"),
                         "\t")

    def test_an_ordinary_csv_is_left_alone(self):
        self.assertEqual(ic.sniff_delimiter(["a,b,c", "1,2,3"], ","), ",")

    def test_pipe_separated(self):
        self.assertEqual(ic.sniff_delimiter(["a|b|c", "1|2|3"], ","), "|")

    def test_a_single_column_file_keeps_the_named_delimiter(self):
        """One column is not evidence of a separator, and guessing one here
        would split a value that merely contains it."""
        self.assertEqual(ic.sniff_delimiter(["text", "hello", "world"], ","), ",")
        self.assertEqual(ic.sniff_delimiter(["text", "hello, world"], ","), ",")

    def test_an_empty_sample_decides_nothing(self):
        self.assertEqual(ic.sniff_delimiter([], ","), ",")
        self.assertEqual(ic.sniff_delimiter(["", "  "], "\t"), "\t")

    def test_coherence_is_what_a_candidate_has_to_survive(self):
        self.assertTrue(ic.coherent(["a,b", "1,2"], ","))
        self.assertFalse(ic.coherent(["a,b", "1,2,3"], ","))  # ragged
        self.assertFalse(ic.coherent(["a,b", "1,2"], ";"))    # one column wide


class TestSampleLines(unittest.TestCase):
    """The read is capped, so its tail can be half a row, and half a row is
    short by a field or two under the very separator that is correct."""

    def test_a_trailing_fragment_is_dropped(self):
        self.assertEqual(ic.sample_lines("a,b\n1,2\n3,"), ["a,b", "1,2"])

    def test_a_complete_read_loses_nothing(self):
        self.assertEqual(ic.sample_lines("a,b\n1,2\n"), ["a,b", "1,2"])

    def test_a_lone_line_is_kept(self):
        self.assertEqual(ic.sample_lines("a,b"), ["a,b"])

    def test_the_sample_is_bounded(self):
        text = "".join(f"{i},{i}\n" for i in range(100))
        self.assertEqual(len(ic.sample_lines(text)), ic.SNIFF_LINES)


class TestReadHeader(unittest.TestCase):
    """A column list means nothing without the separator that produced it, so
    the two come back together (#79)."""

    def test_columns_carry_their_delimiter(self):
        h = ic.read_header(["a;b;c", "1;2;3"], "train.csv")
        self.assertEqual(h.columns, ["a", "b", "c"])
        self.assertEqual(h.delim, ";")

    def test_a_format_we_cannot_read_is_an_unread_not_a_crash(self):
        wide = "x" * (2 * csv.field_size_limit())
        h = ic.read_header([wide], "data.csv")
        self.assertIsInstance(h, ic.Unread)
        self.assertEqual(h.kind, "not_tabular")


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
