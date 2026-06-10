# Investigating Data Dependency Refactorings and Technical Debt in Machine Learning (ML) Systems

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20465381-blue)](https://doi.org/10.5281/zenodo.20465381)

NYU GSTEM 2025 project at CUNY Hunter College.

## Instructions

1. Run `1getHFdatasets.py` to extract datasets from Hugging Face. Filters such as modularity and file format can be adjusted by editing, adding, or removing lines such as those below:
   ```python
   is_tabular = 'modality:tabular' in tags
   is_csv = 'format:csv' in tags
   # Saves the datasets in "filtered_datasets.json"
   ```
1. Use `2getHFcommits.py` to extract more information, including the commit logs, from all the datasets included in `filtered_datasets.json`. The script extracts `datasetId`, `tags`, `downloads`, `likes`, `lastModified`, `created_at`, `commits`; saves this information in `FilteredHFDatasets.csv`.
1. Run `python 3HFcommitFormatting.py FilteredHFDatasets.csv outputFilename.csv`. Formats all previously extracted commits into separate rows; includes `DatasetID`, `CommitId`, `Authors`, `Date`, `Log message`, and `message`.

## Provenance, License, and Citation

This tool was developed by **Ayla Zhang**, a high-school student (Thomas Jefferson High School for Science and Technology) participating in NYU GSTEM (Summer 2025), under the mentorship of **Raffi Khatchadourian** (CUNY Hunter College), as a preliminary study of data-dependency refactorings and technical debt in machine learning systems.

- The **Hugging Face mining** in this repository is original to this work.
- The **GitHub-side commit analysis** reuses the dataset of Tang et al., "An Empirical Study of Refactorings and Technical Debt in Machine Learning Systems," ICSE 2021.
- This material is based upon work supported by the National Science Foundation under Grant No. CCF-2343750. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
- Licensed under the **MIT License** (see [`LICENSE`](LICENSE)).
- Please cite using [`CITATION.cff`](CITATION.cff).

This is a preliminary research prototype; the mining methodology (keyword filtering plus manual inspection) is exploratory and not exhaustive.
