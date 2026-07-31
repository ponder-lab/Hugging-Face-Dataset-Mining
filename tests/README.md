# Tests and Gold Set

- `gold_set.csv`—hand-labeled data refactorings, one row per refactoring. It is the development oracle the staged commit classifiers are built and scored against (https://github.com/ponder-lab/Hugging-Face-Dataset-Mining/issues/11), the source of the (open-coded) refactoring taxonomy, and a regression fixture for `analysis/inspect_commit.py`. It carries labels only, no features: a classifier resolves each `DatasetID`/`CommitId` pair to its evidence itself.
- `test_inspect_commit.py`—unit tests for the helper's parsing logic: pointer detection, header parsing per delimiter, the format a path names, and gzip decompression.
- `test_lfs_download.py`—regression tests for the `--download` and `--show_rows` paths, over throwaway git repos and a mocked HTTP layer. No network and no LFS server are needed to run them.
- `test_missing_revision.py`—regression tests for a commit the repository does not hold (https://github.com/ponder-lab/Hugging-Face-Dataset-Mining/issues/71), which used to print the same empty report as a commit that exists and changed no files. They cover the revision check, the message it produces, and that `(none)` still means what it says. Over throwaway git repos, so git runs for real and no network does.
- `test_clone_failure.py`—regression tests for the disposition a failed clone reports (https://github.com/ponder-lab/Hugging-Face-Dataset-Mining/issues/64): a dataset deleted or made private upstream, one that is access-restricted, one that was renamed, and a failure that is ours rather than the Hub's. The HTTP layer is mocked and git is never run.
- `test_archive_evidence.py`—tests for the evidence archive generator (https://github.com/ponder-lab/Hugging-Face-Dataset-Mining/issues/65): the record round trip, the two-check record for a commit nobody holds any more, rename provenance, and that the index lists every row rather than dropping the ones with no record. Over throwaway git repos with the Hub stubbed out, so git runs for real and no network does.

Run the unit tests:

    python -m unittest discover -s tests
