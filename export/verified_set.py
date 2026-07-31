#!/usr/bin/env python3
"""Export the verified candidate labels to one CSV, in the gold-set schema.

WHAT THIS PRODUCES

One row per (annotator, commit, refactoring), carrying the gold set's five
columns and nothing else:

    DatasetID, CommitId, label, verified_by, notes

`log_message` is dropped for symmetry with the gold set, and the debt fields are
dropped because they are a separate judgment on a separate axis. Column order is
irrelevant downstream, since the repository loads candidate CSVs with
`csv.DictReader`.

Both annotators' rows survive for overlap commits. `verified_by` distinguishes
them, which is what keeps the agreement subset recoverable from this file alone.

TWO FILTERS, AND BOTH ARE THE POINT

1. GOLD-SET EXCLUSION, applied automatically rather than from a list. Any row
   whose (DatasetID, CommitId) appears in tests/gold_set.csv is dropped. Gold
   commits are worked examples the annotators were meant never to see, and a
   rater labeling a worked example is not producing an independent verification.
   Doing this by rule rather than by remembered list is the durable fix: the
   exclusion has to hold whenever the gold set grows, and a list is only correct
   until the next time it does.

2. DISPOSITION NORMALIZATION. A row that could not be judged carries one of
   three words from clarification 7 of the protocol -- `absent`, `inaccessible`,
   `opaque` -- in place of a label, never a guessed type. Older rows predate the
   distinction and use a coarser word for all three cases; the adjudications file
   named on the command line holds the settled disposition and wins.

WHAT THIS IS NOT

Not the catalog. This file is the raw verified labels, two rows deep where the
annotators overlap, which is what an agreement computation and a classifier's
scoring set both need. One adjudicated label per commit is a different artifact
and is built elsewhere.

NAMES ARE REPLACED WITH HANDLES

`verified_by` is emitted as a handle rather than a full name, matching the form
`tests/gold_set.csv` already uses. Two of the annotators are secondary-school
students with no GitHub account, so their institutional identifiers stand in; an
account minted to fill a CSV column would be the throwaway pattern the onboarding
checklist warns about.

The substitution is keyed on the recorded annotator name rather than on argument
order, so re-running with the arguments swapped cannot silently reassign anyone.
A name with no handle on file is a hard error rather than a pass-through, since
letting an unrecognized name through is the failure the map exists to prevent.

WHAT IS NOT COMMITTED, AND WHY

Every input except the candidate list is named on the command line and held
outside this repository, which is public:

  - the annotation source identifiers,
  - the name-to-handle map, which would undo the substitution above,
  - the adjudications file.

The annotation source is read live rather than from an exported copy. An export
cannot detect that its source has moved, and scoring against stale labels is a
failure that does not announce itself; reading the source turns the same drift
into a row-count mismatch at export time.

Usage:

    export GOOGLE_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)
    python3 export/verified_set.py <source-id-a> <source-id-b> \\
        <handles.json> <adjudications.csv> <out.csv>
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

# The name-to-handle map is held OUTSIDE this repository, in a JSON file passed on
# the command line: this repository is public, and a committed map would undo the
# substitution it performs.
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


def rows(source_id, token):
    """Yield the source's rows, live, padded to the full column count."""
    url = VALUES_URL.format(sid=source_id, rng=urllib.parse.quote(RANGE))
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req) as response:
        for r in json.load(response).get("values", [])[1:]:
            yield list(r) + [""] * (8 - len(r))


def main(source_a, source_b, handles_path, adj_path, out):
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
    for source_id in (source_a, source_b):
        for r in rows(source_id, token):
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
        sys.exit("usage: verified_set.py <source-id-a> <source-id-b> "
                 "<handles.json> <adjudications.csv> <out.csv>")
    sys.exit(main(*sys.argv[1:]))
