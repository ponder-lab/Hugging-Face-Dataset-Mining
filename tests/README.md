# Tests and Gold Set

- `gold_set.csv`—hand-labeled data refactorings, one row per refactoring. It is the development oracle the staged commit classifiers are built and scored against (https://github.com/ponder-lab/Hugging-Face-Dataset-Mining/issues/11), the source of the (open-coded) refactoring taxonomy, and a regression fixture for `analysis/inspect_commit.py`. It carries labels only, no features: a classifier resolves each `DatasetID`/`CommitId` pair to its evidence itself.
- `test_inspect_commit.py`—unit tests for the helper's parsing logic: pointer detection, header parsing per delimiter, the format a path names, and gzip decompression.
- `test_lfs_download.py`—regression tests for the `--download` and `--show_rows` paths, over throwaway git repos and a mocked HTTP layer. No network and no LFS server are needed to run them.
- `test_clone_failure.py`—regression tests for the disposition a failed clone reports (https://github.com/ponder-lab/Hugging-Face-Dataset-Mining/issues/64): a dataset deleted or made private upstream, one that is access-restricted, one that was renamed, and a failure that is ours rather than the Hub's. The HTTP layer is mocked and git is never run.

Run the unit tests:

    python -m unittest discover -s tests
