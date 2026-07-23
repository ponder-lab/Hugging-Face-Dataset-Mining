# Tests and Gold Set

- `gold_set.csv`—hand-labeled data refactorings, one row per refactoring. It is the development oracle the staged commit classifiers are built and scored against (https://github.com/ponder-lab/Hugging-Face-Dataset-Mining/issues/11), the source of the (open-coded) refactoring taxonomy, and a regression fixture for `analysis/inspect_commit.py`. It carries labels only, no features: a classifier resolves each `DatasetID`/`CommitId` pair to its evidence itself.
- `test_inspect_commit.py`—unit tests for the helper's parsing logic.

Run the unit tests:

    python -m unittest discover -s tests
