# Tests and Gold Set

- `gold_set.csv`—hand-labeled data refactorings, one row per refactoring. It is the development oracle the staged commit classifiers are built and scored against (https://github.com/ponder-lab/Hugging-Face-Dataset-Mining/issues/11), the source of the (open-coded) refactoring taxonomy, and a regression fixture for `analysis/inspect_commit.py`. It carries labels only, no features: a classifier resolves each `DatasetID`/`CommitId` pair to its evidence itself.
- `test_inspect_commit.py`—unit tests for the helper's parsing logic: pointer detection, header parsing per delimiter, the format a path names, and gzip decompression.
- `test_lfs_download.py`—regression tests for the `--download` and `--show_rows` paths, over throwaway git repos and a mocked HTTP layer. No network and no LFS server are needed to run them.
- `test_missing_revision.py`—regression tests for a commit the repository does not hold (https://github.com/ponder-lab/Hugging-Face-Dataset-Mining/issues/71), which used to print the same empty report as a commit that exists and changed no files. They cover the revision check, the message it produces, and that `(none)` still means what it says. Over throwaway git repos, so git runs for real and no network does.
- `test_clone_failure.py`—regression tests for the disposition a failed clone reports (https://github.com/ponder-lab/Hugging-Face-Dataset-Mining/issues/64): a dataset deleted or made private upstream, one that is access-restricted, one that was renamed, and a failure that is ours rather than the Hub's. The HTTP layer is mocked and git is never run.
- `test_archive_evidence.py`—tests for the evidence archive generator (https://github.com/ponder-lab/Hugging-Face-Dataset-Mining/issues/65): the record round trip, the two-check record for a commit nobody holds any more, rename provenance, and that the index lists every row rather than dropping the ones with no record. Over throwaway git repos with the Hub stubbed out, so git runs for real and no network does.

Run the unit tests:

    python -m unittest discover -s tests

## `verified_set.csv`

The independently verified candidate labels, in the same five columns as `gold_set.csv`, so both load the same way.

Two annotators labeled an overlapping subset of `data/message_refactoring_candidates.csv` from each commit's change, working from a written protocol. This file is their raw output: **one row per (annotator, commit, refactoring)**, 417 rows over 66 commits that carry both annotators. It is deliberately not deduplicated to one label per commit. An agreement computation needs both rows, and a classifier scored against a single reconciled label would be scored against a judgment no annotator made on their own.

- `verified_by` is `R1` or `R2` rather than a name. The annotators are credited on the paper; what this file avoids is pairing a named person with two hundred individual judgments, several of which the study itself documents as mistaken. Nothing downstream needs the identity, only that the two identifiers differ.
- Gold-set commits are excluded automatically: any row whose `(DatasetID, CommitId)` appears in `gold_set.csv` is dropped at export. A rater labeling a worked example is not producing an independent verification, and three such rows had leaked into the pool. The exclusion is a rule rather than a remembered list because the leak recurs whenever the gold set grows.
- Where a change could not be read, `label` carries a disposition rather than a guessed type: `absent` (the data was never uploaded for that revision), `inaccessible` (the repository or commit is gone or gated), or `opaque` (a binary or archive payload the tooling declines to parse).

A commit can appear twice for one annotator when they recorded more than one distinct refactoring on it; the protocol asks for a row each rather than a collapsed label.
