# Investigating Data Dependency Refactorings and Technical Debt in Machine Learning (ML) Systems

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20465381-blue)](https://doi.org/10.5281/zenodo.20465381)

NYU GSTEM 2025 project at CUNY Hunter College.

## Instructions

Install the dependencies first with `pip install -r requirements.txt`.

1. Run `python mining/1getHFdatasets.py` to extract datasets from Hugging Face. Filters such as modularity and file format can be adjusted by editing, adding, or removing lines such as those below:
   ```python
   is_tabular = 'modality:tabular' in tags
   is_csv = 'format:csv' in tags
   # Saves the datasets in "filtered_datasets.json"
   ```
1. Use `python mining/2getHFcommits.py` to extract more information, including the commit logs, from all the datasets included in `filtered_datasets.json`. The script extracts `datasetId`, `tags`, `downloads`, `likes`, `lastModified`, `created_at`, `commits`; saves this information in `FilteredHFDatasets.csv`.
1. Run `python mining/3HFcommitFormatting.py FilteredHFDatasets.csv outputFilename.csv`. Formats all previously extracted commits into separate rows; includes `DatasetID`, `CommitId`, `Authors`, `Date`, `Log message`, and `message`.

## Evidence Archive

Upstream repositories drift after mining: within weeks of the June 2026 sweep, some candidate datasets were deleted or renamed on the Hub, and a few had their histories rewritten so that the mined commit no longer exists anywhere. `data/evidence/` therefore holds one committed file per row of `data/message_refactoring_candidates.csv`: the report `analysis/inspect_commit.py` produces for that commit (commit message, file-level change list with LFS flags, the field separator each revision was read under, column-header diff, row samples), frozen at generation time. Every label is reproducible from this repository alone, with no network.

A commit that can no longer be read appears in the archive carrying the reason, never omitted, and the record states which checks established the loss. `data/evidence/INDEX.csv` lists every row with its disposition and, where the Hub reported a rename, the name the dataset now goes by alongside the mined one. Regenerate with `python analysis/archive_evidence.py`; existing records are kept unless `--force` is passed.

The full git histories behind these records are archived separately on Zenodo as a snapshot of the clone cache, captured 2026-07-31 ([10.5281/zenodo.21727324](https://doi.org/10.5281/zenodo.21727324)). Access to the snapshot is restricted because it aggregates 270 third-party datasets, each governed by its own upstream license; the in-repo evidence records above are the open form of the same facts.

## Preservation Check

A data refactoring is a change that preserves the dataset's information content: every fact recoverable before the change is recoverable after it, and nothing is introduced that was not already derivable. `analysis/preservation_check.py` tests that mechanically over a labeled set, comparing the multiset of rows a commit's touched files hold before and after, projected onto the columns the two revisions share. Row order is irrelevant by construction, so a commit that only reorders rows or columns comes out preserving without a special case. It reads each revision's separator through `inspect_commit.py` rather than reimplementing that rule, so the two tools cannot drift apart on what a file says.

```
python analysis/preservation_check.py --set tests/verified_set.csv --out data/preservation.csv
```

It settles the negative and defers the rest. A commit reported `preserves` has the same rows under the same names at both revisions. Beyond that it separates a fall in the row count (`rows-dropped`) from a rewrite that leaves the count intact (`values-rewritten`), because those ask different questions, and it hands `adds-columns` and `columns-lost` back with the column names attached, since whether a column is derivable from what was already there is a question about meaning. A rename arrives as one of each and is the usual reason a `columns-lost` verdict is not a loss.

Two limits are worth stating before the output is used. Roughly half of the labeled corpus holds its data in Git LFS or a format this check does not read; those commits are reported as unread rather than passed over, but they are not evidence of anything. And because of that, a commit that relocates rows into a file the check cannot open is indistinguishable from one that deletes them, so `rows-dropped` is biased toward false positives on exactly the larger, better-organized datasets. Every verdict is a starting point for a human ruling, not a ruling.

## Provenance, License, and Citation

This tool was developed by **Ayla Zhang**, a high-school student (Thomas Jefferson High School for Science and Technology) participating in NYU GSTEM (Summer 2025), under the mentorship of **Raffi Khatchadourian** (CUNY Hunter College), as a preliminary study of data-dependency refactorings and technical debt in machine learning systems.

- The **Hugging Face mining** in this repository is original to this work.
- The **GitHub-side commit analysis** reuses the dataset of Tang et al., "An Empirical Study of Refactorings and Technical Debt in Machine Learning Systems," ICSE 2021.
- This material is based upon work supported by the National Science Foundation under Grant No. CCF-2343750. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
- Licensed under the **MIT License** (see [`LICENSE`](LICENSE)).
- Please cite using [`CITATION.cff`](CITATION.cff).

This is a preliminary research prototype; the mining methodology (keyword filtering plus manual inspection) is exploratory and not exhaustive.
