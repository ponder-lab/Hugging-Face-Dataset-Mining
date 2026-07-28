"""Regression tests for what a failed clone reports (#64).

Every cause used to come back as one `clone failed` line, so a dataset deleted
upstream was indistinguishable from a broken tool (#63). These cover the probe
that tells them apart and the message each outcome produces. The HTTP layer is
mocked and git is never run, so no network is needed.
"""
import os, shutil, sys, tempfile, unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import inspect_commit as ic

import requests

DS = "someone/a-dataset"


class FakeHead:
    """Minimal stand-in for the response to a HEAD on the metadata endpoint."""

    def __init__(self, status_code, location=None):
        self.status_code = status_code
        self.headers = {"Location": location} if location else {}


def heading(status_code, location=None):
    """Patch the HTTP layer to answer one probe."""
    return mock.patch.object(ic.requests, "head",
                             return_value=FakeHead(status_code, location))


class TestMovedTarget(unittest.TestCase):
    def test_absolute_redirect_names_the_new_id(self):
        self.assertEqual(
            ic.moved_target("https://huggingface.co/api/datasets/prism-oncology/novae"),
            "prism-oncology/novae")

    def test_relative_redirect(self):
        self.assertEqual(ic.moved_target("/api/datasets/albertgong1/stackexchange"),
                         "albertgong1/stackexchange")

    def test_query_string_is_dropped(self):
        self.assertEqual(ic.moved_target("/api/datasets/a/b?full=true"), "a/b")

    def test_unrecognized_location_is_none(self):
        self.assertIsNone(ic.moved_target("https://example.com/elsewhere"))
        self.assertIsNone(ic.moved_target(""))
        self.assertIsNone(ic.moved_target("/api/datasets/"))


class TestRepoStatus(unittest.TestCase):
    def test_served_repo_is_reachable(self):
        with heading(200):
            self.assertEqual(ic.repo_status(DS).kind, ic.REACHABLE)

    def test_rename_is_moved_and_carries_the_new_id(self):
        with heading(307, "https://huggingface.co/api/datasets/new-owner/thing"):
            s = ic.repo_status(DS)
        self.assertEqual(s.kind, ic.MOVED)
        self.assertEqual(s.moved_to, "new-owner/thing")

    def test_rename_without_a_usable_location_still_reports_the_move(self):
        with heading(308, "https://example.com/somewhere-else"):
            s = ic.repo_status(DS)
        self.assertEqual(s.kind, ic.MOVED)
        self.assertIsNone(s.moved_to)

    def test_anonymous_401_is_gone(self):
        """What the Hub answers for a repo deleted or made private (#63), and,
        as measured on the live endpoint, for one that never existed under this
        ID (#67). The three are indistinguishable from outside, and the message
        says so rather than picking one."""
        with heading(401):
            self.assertEqual(ic.repo_status(DS).kind, ic.GONE)

    def test_404_is_gone(self):
        with heading(404):
            self.assertEqual(ic.repo_status(DS).kind, ic.GONE)

    def test_403_is_restricted_not_gone(self):
        """A token can get into a restricted repo; nothing gets into a deleted
        one, so these must not share a message."""
        with heading(403):
            self.assertEqual(ic.repo_status(DS).kind, ic.RESTRICTED)

    def test_server_error_is_unknown(self):
        with heading(503):
            self.assertEqual(ic.repo_status(DS).kind, ic.UNKNOWN)

    def test_probe_that_does_not_land_is_unknown(self):
        with mock.patch.object(ic.requests, "head",
                               side_effect=requests.ConnectionError("no route")):
            s = ic.repo_status(DS)
        self.assertEqual(s.kind, ic.UNKNOWN)
        self.assertEqual(s.detail, "ConnectionError")

    def test_redirects_are_not_followed(self):
        """The redirect is the finding, so following it would discard the answer."""
        with heading(200) as head:
            ic.repo_status(DS)
        self.assertFalse(head.call_args.kwargs["allow_redirects"])

    def test_token_is_sent_when_set(self):
        for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            with self.subTest(var=var):
                with mock.patch.dict(os.environ, {var: "tok"}, clear=True):
                    with heading(200) as head:
                        ic.repo_status(DS)
                self.assertEqual(head.call_args.kwargs["headers"]["Authorization"],
                                 "Bearer tok")

    def test_no_token_sends_no_auth_header(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with heading(200) as head:
                ic.repo_status(DS)
        self.assertNotIn("Authorization", head.call_args.kwargs["headers"])


class TestCloneFailureMessage(unittest.TestCase):
    def test_gone_names_the_cause_and_the_action(self):
        msg = ic.clone_failure(DS, ic.RepoStatus(ic.GONE, "HTTP 401"))
        self.assertIn(DS, msg)
        self.assertIn("HTTP 401", msg)
        self.assertIn("deleted or made private", msg)
        self.assertIn("skip", msg)

    def test_gone_admits_the_id_may_never_have_existed(self):
        """A 401 covers a corrupted DatasetID too, and there the recommended
        action inverts: the dataset is live under its real ID, so skipping on
        this message alone drops a candidate that is still verifiable (#67)."""
        msg = ic.clone_failure(DS, ic.RepoStatus(ic.GONE, "HTTP 401"))
        self.assertIn("no dataset ever went by this ID", msg)
        self.assertIn(ic.CANDIDATES, msg)

    def test_gone_does_not_order_a_skip_before_the_id_is_checked(self):
        """The claim and the instruction have to agree: checking the ID comes
        first, so the word `skip` may not stand on its own here."""
        msg = ic.clone_failure(DS, ic.RepoStatus(ic.GONE, "HTTP 401"))
        self.assertLess(msg.index("check the DatasetID"), msg.index("skipping"))

    def test_restricted_points_at_the_token(self):
        msg = ic.clone_failure(DS, ic.RepoStatus(ic.RESTRICTED, "HTTP 403"))
        self.assertIn("HF_TOKEN", msg)
        self.assertNotIn("deleted", msg)

    def test_moved_names_the_new_id(self):
        msg = ic.clone_failure(
            DS, ic.RepoStatus(ic.MOVED, "HTTP 307", moved_to="new/thing"))
        self.assertIn("new/thing", msg)

    def test_reachable_says_the_failure_is_ours(self):
        """The one case that really is a defect on our side must read as one."""
        msg = ic.clone_failure(DS, ic.RepoStatus(ic.REACHABLE, "HTTP 200"))
        self.assertIn("transport or local git failure", msg)

    def test_unknown_does_not_blame_the_dataset(self):
        msg = ic.clone_failure(DS, ic.RepoStatus(ic.UNKNOWN, "ConnectionError"))
        self.assertIn("could not be asked why", msg)
        self.assertNotIn("deleted", msg)

    def test_every_disposition_is_distinct(self):
        msgs = [ic.clone_failure(DS, ic.RepoStatus(k, "detail"))
                for k in (ic.GONE, ic.RESTRICTED, ic.MOVED, ic.REACHABLE, ic.UNKNOWN)]
        self.assertEqual(len(set(msgs)), len(msgs))


class TestClone(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        cache = mock.patch.object(ic, "CACHE", self.tmp)
        cache.start()
        self.addCleanup(cache.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _run_clone(self, returncode=0):
        """Patch out git and return the mock, so no test shells out to the network."""
        return mock.patch.object(ic.subprocess, "run",
                                 return_value=mock.Mock(returncode=returncode))

    def test_gone_dataset_exits_without_attempting_a_clone(self):
        """Cloning a repo the Hub refuses anonymously only produces git's
        suppressed credential prompt, which is the confusing part of #63."""
        with heading(401), self._run_clone() as run:
            with self.assertRaises(SystemExit) as e:
                ic.clone(DS)
        self.assertIn("deleted or made private", str(e.exception.code))
        run.assert_not_called()

    def test_restricted_dataset_exits_without_attempting_a_clone(self):
        with heading(403), self._run_clone() as run:
            with self.assertRaises(SystemExit) as e:
                ic.clone(DS)
        self.assertIn("HF_TOKEN", str(e.exception.code))
        run.assert_not_called()

    def test_renamed_dataset_is_cloned_and_the_rename_reported(self):
        """git follows the redirect, so this is provenance, not a failure."""
        err = StringIO()
        with heading(307, "/api/datasets/new-owner/thing"), self._run_clone():
            with redirect_stderr(err):
                path = ic.clone(DS)
        self.assertEqual(path, os.path.join(self.tmp, "someone__a-dataset"))
        self.assertIn("new-owner/thing", err.getvalue())
        self.assertIn("renamed", err.getvalue())

    def test_reachable_dataset_clones_quietly(self):
        err = StringIO()
        with heading(200), self._run_clone():
            with redirect_stderr(err):
                path = ic.clone(DS)
        self.assertEqual(path, os.path.join(self.tmp, "someone__a-dataset"))
        self.assertEqual(err.getvalue(), "")

    def test_unknown_probe_does_not_block_the_clone(self):
        """A probe that cannot reach the Hub must not decide the clone will fail."""
        with mock.patch.object(ic.requests, "head",
                               side_effect=requests.ConnectionError("no route")):
            with self._run_clone() as run:
                ic.clone(DS)
        run.assert_called_once()

    def test_git_failure_on_a_served_dataset_reads_as_ours(self):
        with heading(200), self._run_clone(returncode=128):
            with self.assertRaises(SystemExit) as e:
                ic.clone(DS)
        self.assertIn("transport or local git failure", str(e.exception.code))

    def test_cached_clone_needs_neither_probe_nor_git(self):
        os.makedirs(os.path.join(self.tmp, "someone__a-dataset", ".git"))
        with mock.patch.object(ic.requests, "head",
                               side_effect=AssertionError("probed a cached repo")):
            with self._run_clone() as run:
                path = ic.clone(DS)
        self.assertEqual(path, os.path.join(self.tmp, "someone__a-dataset"))
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
