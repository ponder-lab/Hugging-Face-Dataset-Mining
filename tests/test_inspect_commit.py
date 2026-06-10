import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import inspect_commit as ic

class TestParsing(unittest.TestCase):
    def test_lfs_pointer_detected(self):
        ptr = "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12\n"
        self.assertTrue(ic.is_lfs_pointer(ptr))
        self.assertIsNone(ic.parse_csv_header(ptr))   # data absent for LFS files

    def test_plain_csv_header(self):
        self.assertFalse(ic.is_lfs_pointer("a,b,c\n1,2,3\n"))
        self.assertEqual(ic.parse_csv_header('"Name","Prompt",Votes\nx,y,z'),
                         ["Name", "Prompt", "Votes"])

    def test_empty_file(self):
        self.assertEqual(ic.parse_csv_header(""), [])

if __name__ == "__main__":
    unittest.main()
