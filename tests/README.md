# Tests and Gold Set

- `gold_set.csv`—37 hand-labeled refactorings across 33 commits. It is the scoring oracle the staged commit classifiers are evaluated against, the source of the (open-coded) refactoring taxonomy, and a regression fixture for `analysis/inspect_commit.py`.
- `test_inspect_commit.py`—unit tests for the helper's parsing logic.

Run the unit tests:

    python -m unittest discover -s tests
