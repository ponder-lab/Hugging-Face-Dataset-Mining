# Export

Regenerates the committed label CSVs from the annotation sources they come from.

A CSV committed here is a snapshot of a source held outside this repository. A snapshot with no committed generator cannot be checked or refreshed by anyone who does not already have the pipeline on their machine, which is the gap this directory exists to close.

## `verified_set.py`

Writes `tests/verified_set.csv`: both annotators' rows, in the gold-set schema.

```
export GOOGLE_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)
python3 export/verified_set.py <source-id-a> <source-id-b> \
    <handles.json> <adjudications.csv> tests/verified_set.csv
```

Three inputs are named on the command line rather than committed, and all three for the same reason: this repository is public.

- **Source identifiers.** The annotation sources are held outside this repository deliberately, and naming one here would defeat that.
- **`handles.json`.** Maps annotator name to the handle published in the CSV. Committing it would undo the substitution it performs.
- **`adjudications.csv`.** The settled disposition per commit, used to normalize `absent`, `inaccessible` and `opaque` where a row still carries a coarser word.

The script reads the source live rather than an exported copy. An export cannot detect that its source has moved, and scoring against stale labels is a failure that does not announce itself; reading the source turns the same drift into a row-count mismatch at export time.

An annotator name with no entry in `handles.json` is a hard error rather than a pass-through, so a new rater cannot have their full name written into a published file by omission.
