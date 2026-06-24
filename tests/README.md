# Tests and Gold Set

- `gold_set.csv`—a small set of hand-labeled commits: a regression fixture for the inspection helper and reference/sample data. Rows marked `TODO` are a worklist to label (use `analysis/inspect_commit.py <dataset> <commit>`).
- `test_inspect_commit.py`—unit tests for the helper's parsing logic.

Run the unit tests:

    python -m unittest discover -s tests
