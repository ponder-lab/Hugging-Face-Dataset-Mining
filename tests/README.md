# Tests and gold set

- `gold_set.csv` — hand-labeled commits used as the **oracle** for the classifier,
  the seed of the **inter-rater/ground-truth** set, and a **regression** fixture.
  One verified negative is seeded; rows marked `TODO` are a worklist to label
  (use `tools/inspect_commit.py <dataset> <commit>`). Labeling these defines the
  taxonomy (open-code as you go; new types are expected).
- `test_inspect_commit.py` — unit tests for the helper's parsing logic.

Run the unit tests:

    python -m unittest discover -s tests
