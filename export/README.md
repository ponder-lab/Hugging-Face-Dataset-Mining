# Export

Regenerates the committed label CSVs from the annotation sheets they come from.

The sheets are the source of truth. A CSV committed here is a snapshot of them, and a snapshot with no committed generator cannot be checked or refreshed by anyone who does not already have the pipeline on their machine. That is how `tests/gold_set.csv` ended up with no way to reproduce it.

## `verified_set.py`

Writes `tests/verified_set.csv`: both annotators' rows, in the gold-set schema.

```
export GOOGLE_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)
python3 export/verified_set.py <sheet-id-a> <sheet-id-b> \
    <handles.json> <adjudications.csv> tests/verified_set.csv
```

Three inputs are named on the command line rather than committed, and all three for the same reason: this repository is public.

- **Sheet identifiers.** Naming an annotator's working sheet in a repository they may one day read is not something to do by accident.
- **`handles.json`.** Maps annotator name to the handle published in the CSV. Committing it would undo the substitution it performs.
- **`adjudications.csv`.** The settled disposition per commit, used to normalize `absent`, `inaccessible` and `opaque` where a sheet still carries an older word.

The script fetches the sheets live rather than reading an exported file. A stale export cannot detect that its source has changed, and scoring against last month's labels is a failure that does not announce itself; a live fetch turns the same drift into a row-count mismatch. On its first live run it found a `CommitId` that had been blanked in one sheet, which would otherwise have dropped that commit out of every join silently.

An annotator name with no entry in `handles.json` is a hard error rather than a pass-through, so a new rater cannot have their full name written into a published file by omission.
