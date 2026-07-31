#!/usr/bin/env python3
"""Export both annotators' verified sheets to one CSV, in the gold-set schema.

WHAT THIS PRODUCES

One row per (annotator, commit, refactoring), carrying the gold set's five
columns and nothing else:

    DatasetID, CommitId, label, verified_by, notes

`log_message` is dropped for symmetry with the gold set, and the debt fields are
dropped because they are a separate judgment on a separate axis; a consumer that
wants them should read the sheets. Column order is irrelevant downstream, since
the repository loads candidate CSVs with `csv.DictReader`.

Both annotators' rows survive for overlap commits. `verified_by` distinguishes
them, which is what keeps the agreement subset recoverable from this file alone.

TWO FILTERS, AND BOTH ARE THE POINT

1. GOLD-SET EXCLUSION, applied automatically rather than from a list. Any row
   whose (DatasetID, CommitId) appears in tests/gold_set.csv is dropped. Gold
   commits are worked examples the annotators were meant never to see; three
   leaked into the rater pool anyway, and a rater labeling a worked example is
   not producing an independent verification. Doing this by rule rather than by
   remembered list is the durable fix: the leak recurs whenever the gold set
   grows, and a list is only correct until the next sync.

2. DISPOSITION NORMALIZATION. A row that could not be judged carries one of
   three words from clarification 7 of the protocol -- `absent`, `inaccessible`,
   `opaque` -- in place of a label, never a guessed type. Older rows predate the
   distinction and say `LFS-opaque` or a bare `opaque` for all three cases;
   scripts/adjudications.csv holds the settled disposition for every such row and
   wins over what the sheet says.

WHAT THIS IS NOT

Not the catalog. This file is the raw verified labels, two rows deep where the
annotators overlap, which is what an agreement computation and a classifier's
scoring set both need. One adjudicated label per commit is a different artifact
and is built elsewhere.

NAMES ARE REPLACED WITH HANDLES

`verified_by` is emitted as a handle rather than a full name, matching the form
`tests/gold_set.csv` already uses (`miridonoso`, `khatchad`). The two
secondary-school annotators have no GitHub account, so their institutional
identifiers stand in; an account minted to fill a CSV column would be the
throwaway pattern the onboarding checklist warns about, and a handle is what the
gold set's convention actually calls for.

The substitution is driven by the name in the sheet, not by the order the sheets
are passed, so re-running with the arguments swapped cannot silently reassign
anyone. A name absent from the map is a hard error rather than a pass-through:
passing an unrecognized name straight into a published file is the failure this
map exists to prevent.

The map itself lives outside this repository, in a JSON file named on the command
line, because this repository is public and a committed map would undo the
substitution it performs.

Usage:

    export GOOGLE_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)
    python3 export/verified_set.py <sheet-id-a> <sheet-id-b> \
        <handles.json> <adjudications.csv> <out.csv>

`handles.json` maps annotator name to published handle. `adjudications.csv` is
the settled disposition per commit. Both live outside this repository and are
named on the command line, for the same reason the sheet identifiers are.

Sheet identifiers are arguments rather than constants, as in label_map.py and
kappa_denominator.py, so that neither verifier's sheet is named in a repository
either of them might one day read. The sheets are fetched live: a pre-exported
file would go stale silently, and scoring a classifier against last month's
labels is the failure this whole pipeline keeps running into.
"""
import csv
import io
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request

GOLD = ("https://raw.githubusercontent.com/ponder-lab/"
        "Hugging-Face-Dataset-Mining/main/tests/gold_set.csv")
VALUES_URL = "https://sheets.googleapis.com/v4/spreadsheets/{sid}/values/{rng}"
RANGE = "candidates!A:H"
CAND = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "data", "message_refactoring_candidates.csv")
DISPOSITIONS = ("absent", "inaccessible", "opaque")

# Annotator name as it appears in the sheet -> the handle published in the CSV.
# Held OUTSIDE this repository, in a JSON file passed on the command line, for the
# same reason the sheet identifiers are: this repository is public, and a file
# mapping a handle back to a full name would undo the substitution it performs.
# A name with no handle on file is a hard error rather than a pass-through.
FIELDS = ["DatasetID", "CommitId", "label", "verified_by", "notes"]


def access_token():
    token = os.environ.get("GOOGLE_ACCESS_TOKEN")
    if token:
        return token
    try:
        out = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sys.exit("no token: set GOOGLE_ACCESS_TOKEN or install the gcloud CLI")


def rows(sheet_id, token):
    """Yield the sheet's rows, live, padded to the full column count."""
    url = VALUES_URL.format(sid=sheet_id, rng=urllib.parse.quote(RANGE))
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req) as response:
        for r in json.load(response).get("values", [])[1:]:
            yield list(r) + [""] * (8 - len(r))


def main(sheet_a, sheet_b, handles_path, adj_path, out):
    token = access_token()
    handles = json.load(open(handles_path))
    gold = {(r["DatasetID"].strip(), r["CommitId"].strip())
            for r in csv.DictReader(io.StringIO(
                urllib.request.urlopen(GOLD).read().decode()))}
    valid = {(r["DatasetID"].strip(), r["CommitId"].strip())
             for r in csv.DictReader(open(CAND))}
    disp = {}
    for r in csv.DictReader(open(adj_path)):
        if r["disposition"].strip():
            disp[(r["DatasetID"].strip(), r["CommitId"].strip())] = \
                r["disposition"].strip()

    kept, dropped_gold, unjoinable, normalized = [], 0, [], 0
    raters = {}
    for sheet_id in (sheet_a, sheet_b):
        for r in rows(sheet_id, token):
            key = (r[0].strip(), r[1].strip())
            if key not in valid:
                unjoinable.append(key)
                continue
            if key in gold:
                dropped_gold += 1
                continue
            label = r[3].strip()
            if key in disp:
                if label != disp[key]:
                    normalized += 1
                label = disp[key]
            elif label.lower() in ("lfs-opaque", "opaque"):
                # A stale disposition with no adjudication on file. Left as
                # written rather than guessed at, and reported below.
                normalized += 0
            name = r[4].strip()
            if name not in handles:
                sys.exit(f"no handle for annotator {name!r}; add it to "
                         f"{handles_path} rather than letting a name reach a "
                         f"published file")
            raters[handles[name]] = name
            kept.append({"DatasetID": r[0].strip(), "CommitId": r[1].strip(),
                         "label": label, "verified_by": handles[name],
                         "notes": r[5].strip()})

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(kept)

    stale = sorted({(r["DatasetID"], r["CommitId"]) for r in kept
                    if r["label"].lower() not in ("",)
                    and r["label"].lower().replace("lfs-", "") in DISPOSITIONS
                    and (r["DatasetID"], r["CommitId"]) not in disp})

    print(f"rows written              {len(kept)}")
    print(f"  handles                 {', '.join(sorted(raters))}")
    print(f"  dropped, gold-set row   {dropped_gold}")
    print(f"  dropped, unjoinable key {len(unjoinable)}")
    for k in unjoinable:
        print(f"      {k[0]} @ {k[1][:12]}")
    print(f"  dispositions normalized {normalized}")
    if stale:
        print(f"\n  disposition words with no adjudication on file: {len(stale)}")
        for d, c in stale:
            print(f"      {d} @ {c[:12]}")
    dual = {}
    for r in kept:
        dual.setdefault((r["DatasetID"], r["CommitId"]), set()).add(r["verified_by"])
    print(f"\ncommits carrying two annotators: "
          f"{sum(1 for v in dual.values() if len(v) > 1)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("usage: verified_set.py <sheet-id-a> <sheet-id-b> "
                 "<handles.json> <adjudications.csv> <out.csv>")
    sys.exit(main(*sys.argv[1:]))
