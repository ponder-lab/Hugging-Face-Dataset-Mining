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

## Preservation Screen

A data refactoring is one that preserves the dataset's information content: every fact recoverable before the change is recoverable after it, and nothing is introduced that was not already derivable. **That judgment is made by human annotators against the written criterion, and disagreements are adjudicated against the diff.** `analysis/preservation_check.py` does not make it and does not attempt to.

What the script does is the arithmetic such a judgment rests on, which is where careful readers reliably fail. It compares the multiset of rows a commit's touched files hold before and after, projected onto the columns the two revisions share, so row order is irrelevant and a reordering comes out unchanged without a special case. Three commits in this corpus were read wrongly by a person and corrected by that arithmetic: a split recorded as dropping rows that is an exact 768-row partition, a deduplication that looked like the corpus's largest deletion and retains every removed row in a file the same commit adds, and the separator change in #79.

```
python analysis/preservation_check.py <dataset> <commit>
```

Most of the criterion is outside its reach, because derivability is about meaning. An added column holding a computed ratio and one holding a model's output look identical here; a vanished column may be a rename or a deletion; a rewritten value may be a normalization or a loss. Those come back as `adds-columns`, `columns-lost` and `values-rewritten`, with the column names attached, and each is a question handed to a person. Deduplication is a case where the script is simply wrong by the criterion: it reports `rows-dropped` while every distinct fact survives.

So `preserves` means only that the rows and column names it could read are identical at both revisions, which is a sufficient condition for no loss rather than the criterion. Counts it prints are commits screened, not preservation decided, and no aggregate of them belongs in a result.

LFS-tracked files are streamed from the Hub and discarded rather than stored, since the corpus keeps 106 GB behind its pointers and only the rows are needed. Revisions are resolved to concrete SHAs first, because the Hub does not accept git revision expressions and parent-side fetches otherwise 404 silently. `PRESERVATION_LFS_CAP` bounds what one file may pull, default 100 MB against a 12.7 GB tail, and anything turned away is named rather than folded into a total.

## Provenance, License, and Citation

This tool was developed by **Ayla Zhang**, a high-school student (Thomas Jefferson High School for Science and Technology) participating in NYU GSTEM (Summer 2025), under the mentorship of **Raffi Khatchadourian** (CUNY Hunter College), as a preliminary study of data-dependency refactorings and technical debt in machine learning systems.

- The **Hugging Face mining** in this repository is original to this work.
- The **GitHub-side commit analysis** reuses the dataset of Tang et al., "An Empirical Study of Refactorings and Technical Debt in Machine Learning Systems," ICSE 2021.
- This material is based upon work supported by the National Science Foundation under Grant No. CCF-2343750. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
- Licensed under the **MIT License** (see [`LICENSE`](LICENSE)).
- Please cite using [`CITATION.cff`](CITATION.cff).

This is a preliminary research prototype; the mining methodology (keyword filtering plus manual inspection) is exploratory and not exhaustive.
