#!/usr/bin/env python3
"""Decide mechanically whether a labeled commit preserves the dataset's information content.

WHY THIS EXISTS

Our annotation protocol defines a data refactoring as a change that preserves the
dataset's information content: every fact recoverable from the dataset before the
change is recoverable after it, and no fact is introduced that was not already
derivable from what the dataset held. Whether real commits fail that test is an
empirical question, and it had been answered from two examples.

Both commits cited for that claim dissolved when checked by hand on 2026-07-31:

  djulian13/east-slavic-swadesh-lists @ a4c5651f  changes the field separator from
    `,` to `;` because the column names contain commas. The inspection tool read
    the child on commas anyway and reported six malformed columns that do not
    exist (ponder-lab/Hugging-Face-Dataset-Mining#79). Nothing was lost.

  khoaguin/pima-indians-diabetes-database @ a19fc7ccdb  splits one file into two.
    The parent holds 768 data rows and the children hold 768 between them, every
    one matching apart from a stray carriage return. It is an exact partition.

Two hand checks are not a result, and the axis should not stand or fall on
whichever cases someone happened to look at twice. This runs the test over every
commit in a labeled set and reports what it finds, so the answer is a measurement
rather than a recollection.

WHAT IT CAN AND CANNOT DECIDE

It compares the multiset of data rows held by the whole tree before and after the
commit, not file by file. A dataset's information content does not live in any
one file, and comparing per file cannot see a split, a merge, or a rename without
being told which file became which. Comparing trees needs to be told nothing.

Because it is a multiset, row order is irrelevant by construction, so a commit
that only reorders rows or columns comes out preserving without a special case.

Rows are compared projected onto the columns the two revisions share. Without
that, adding one column would rewrite every row tuple and every commit that adds
a column would report total loss. With it, the check answers one question well:
did any row that was there before stop being there? That is the half of
derivability a machine can settle.

A tuple that is gone can mean a deletion or a rewrite, and lumping the two under
one verdict was the first version's mistake: most of what it called lost rows was
a commit that rewrote values while the row count never moved. So `rows-dropped`
is reserved for a fall in the count, and `values-rewritten` names the other case,
which asks a different question. Whether a rewritten value is derivable from the
one it replaced is a question about meaning, not about counting.

It cannot decide derivability at all, for the same reason: a column holding a
model's predictions and a column holding a computed ratio look identical here.
Commits that add columns come back as `adds-columns` and commits that drop one
as `columns-lost`, with the column names in the output, so that finishing the
judgment does not require opening the diff again. A rename shows up as one of
each and is the most common reason a `columns-lost` verdict is not a loss.

What it settles on its own is the negative: a commit reported `preserves` has the
same rows under the same names at both revisions, and no reading of the diff will
turn that into an information loss.

It reads the separator out of each revision's own bytes, by importing that logic
from inspect_commit rather than reimplementing it. Deriving the separator from
the file name is the defect above, and a checker carrying its own second copy of
the rule would eventually disagree with the tool the labels were formed under.

USAGE

  python3 analysis/preservation_check.py --set tests/verified_set.csv
  python3 analysis/preservation_check.py djulian13/east-slavic-swadesh-lists a4c5651f
  python3 analysis/preservation_check.py --set tests/verified_set.csv --out data/preservation.csv

Clones land in ~/.cache/hf-dataset-clones, shared with the inspection tool, with
LFS payloads left on the server. Files whose bytes are not present locally are
reported as skipped rather than passed over, because a check that goes quiet on
what it could not read is indistinguishable from one that found nothing wrong.
"""
import argparse
import csv
import gzip
import io
import os
import subprocess
import sys
import zlib
from collections import Counter

# The separator logic lives in inspect_commit and is shared rather than copied.
# Two implementations of "which byte separates these fields" is how the corpus
# ends up with two answers for one file, which is the defect this check exists
# to keep out (#79).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_commit import (  # noqa: E402
    SNIFF_LINES, coherent, delimiter, sniff_delimiter,
)

CACHE = os.path.expanduser("~/.cache/hf-dataset-clones")
# A tree can hold more data than belongs in memory. Rows past this are not read,
# and a file that hits it is named in the output rather than silently truncated.
MAX_ROWS_PER_FILE = 400_000
TABULAR_SUFFIXES = (".csv", ".tsv", ".tab")
LFS_MARKER = b"version https://git-lfs.github.com/spec/v1"


def run(args, **kwargs):
    return subprocess.run(args, capture_output=True, **kwargs)


def clone(dataset):
    """Local clone of `dataset`, fetched if absent. Returns a path or None."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, dataset.replace("/", "__"))
    if os.path.isdir(os.path.join(path, ".git")):
        return path
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    done = run(["git", "clone", "--quiet",
                f"https://huggingface.co/datasets/{dataset}", path], env=env)
    return path if done.returncode == 0 else None


def blob(repo, rev, path):
    """Bytes of `path` at `rev`, or None if the revision does not hold it."""
    done = run(["git", "-C", repo, "show", f"{rev}:{path}"])
    return done.stdout if done.returncode == 0 else None


def tree_files(repo, rev):
    """Every path held at `rev`, or None if `rev` is not in the repository."""
    done = run(["git", "-C", repo, "ls-tree", "-r", "--name-only", rev])
    if done.returncode != 0:
        return None
    return [p for p in done.stdout.decode("utf-8", "replace").splitlines() if p]


def is_tabular(path):
    name = path.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return name.endswith(TABULAR_SUFFIXES)


def decompress(path, data):
    if not path.lower().endswith(".gz"):
        return data
    try:
        return gzip.decompress(data)
    except (OSError, EOFError, zlib.error):
        return None


def read_table(repo, rev, path):
    """(columns, rows) for one tabular file, or ('skip', reason)."""
    data = blob(repo, rev, path)
    if data is None:
        return None, "absent"
    if data.startswith(LFS_MARKER):
        return None, "lfs-pointer"
    data = decompress(path, data)
    if data is None:
        return None, "undecompressable"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
        except UnicodeDecodeError:
            return None, "undecodable"
    # The suffix proposes and the bytes dispose, exactly as inspect_commit does,
    # so a file whose separator changed is read correctly at both revisions.
    named = delimiter(path)
    if named is None:
        return None, "unreadable-format"
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None, "empty"
    delim = sniff_delimiter(lines, named)
    if not coherent(lines[:SNIFF_LINES], delim):
        return None, "no-delimiter"
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    try:
        header = next(reader)
    except StopIteration:
        return None, "empty"
    rows = []
    for i, row in enumerate(reader):
        if i >= MAX_ROWS_PER_FILE:
            return None, f"over-{MAX_ROWS_PER_FILE}-rows"
        rows.append(row)
    return (header, rows), delim


def changed_paths(repo, sha):
    """Tabular paths the commit touched, both sides of any rename.

    Comparing whole trees was the first attempt and it was wrong. A dataset that
    holds unrelated tables -- per-client splits, mock alongside private -- pools
    rows from incompatible schemas into one bag, and the bag then disagrees with
    itself for reasons the commit had nothing to do with. Unchanged files are
    identical across the two revisions by definition and can only add noise.

    Rename detection is off on purpose: a rename has to appear as a deletion and
    an addition so that both names reach the comparison, which is what lets a
    split or a merge be read without being told which file became which.
    """
    done = run(["git", "-C", repo, "diff", "--name-only", "--no-renames",
                f"{sha}~1", sha])
    if done.returncode != 0:
        return None
    return [p for p in done.stdout.decode("utf-8", "replace").splitlines()
            if p and is_tabular(p)]


def tree_content(repo, rev, paths):
    """Columns and rows held by `paths` at `rev`.

    Returns (columns, tables, skipped). `columns` is the union over the files,
    since a fact recorded in any file of the dataset is recoverable from it.
    """
    columns, tables, skipped = [], [], []
    for path in sorted(paths):
        result, detail = read_table(repo, rev, path)
        if result is None:
            # A file added by the commit is absent from the parent, and one the
            # commit deleted is absent from the child. That is the change being
            # measured, not a failure to read it, and counting it as a skip
            # marked almost every add and delete as unverified.
            if detail != "absent":
                skipped.append(f"{path}:{detail}")
            continue
        header, rows = result
        for name in header:
            if name not in columns:
                columns.append(name)
        tables.append((path, header, rows))
    return columns, tables, skipped


ABSENT = "\x00absent"


def projected(tables, keep):
    """Multiset of row tuples restricted to the columns in `keep`.

    Restricting is what lets the check survive a commit that adds a column: the
    question asked is whether the facts already recorded are still recorded, and
    a new column is not evidence either way about the old ones.

    Every tuple has one slot per name in `keep`, in that order, so that rows from
    files with different headers stay comparable. Building the tuple from only
    the columns a given file happens to carry produced tuples of different
    lengths, which never compare equal and reported total loss on commits that
    lost nothing.

    The residual limit: a column carried by one revision's files and not the
    other's fills with ABSENT, so a commit that moves rows between files with
    genuinely different headers can still report loss where none occurred. That
    is the same shape as the LFS blindness in the module docstring, and it fails
    in the same direction, toward a false `rows-dropped`.
    """
    bag = Counter()
    for _path, header, rows in tables:
        index = [header.index(c) if c in header else None for c in keep]
        if all(i is None for i in index):
            continue
        for row in rows:
            bag[tuple(row[i].strip() if i is not None and i < len(row)
                      else ABSENT for i in index)] += 1
    return bag


def check(dataset, sha):
    """Verdict dict for one commit."""
    out = {"dataset": dataset, "commit": sha, "verdict": "", "detail": "",
           "rows_before": "", "rows_after": "", "rows_changed": "",
           "columns_lost": "", "columns_added": "", "skipped": ""}
    repo = clone(dataset)
    if repo is None:
        out["verdict"] = "unreachable"
        out["detail"] = "dataset could not be cloned"
        return out
    if tree_files(repo, sha) is None:
        out["verdict"] = "unreachable"
        out["detail"] = "commit not in repository"
        return out

    paths = changed_paths(repo, sha)
    if paths is None:
        out["verdict"] = "no-parent"
        out["detail"] = "root commit, nothing to compare against"
        return out
    if not paths:
        out["verdict"] = "no-tabular-change"
        out["detail"] = "the commit touched no file this check can read"
        return out

    before_cols, before, skip_b = tree_content(repo, f"{sha}~1", paths)
    after_cols, after, skip_a = tree_content(repo, sha, paths)

    skipped = sorted(set(skip_b or []) | set(skip_a or []))
    out["skipped"] = "; ".join(skipped)
    if not before and not after:
        out["verdict"] = "unreadable"
        out["detail"] = "no readable tabular file at either revision"
        return out

    shared = [c for c in before_cols if c in after_cols]
    lost_cols = [c for c in before_cols if c not in after_cols]
    added_cols = [c for c in after_cols if c not in before_cols]
    out["columns_lost"] = "; ".join(lost_cols)
    out["columns_added"] = "; ".join(added_cols)

    bag_before = projected(before, shared)
    bag_after = projected(after, shared)
    out["rows_before"] = sum(projected(before, before_cols).values())
    out["rows_after"] = sum(projected(after, after_cols).values())
    missing = bag_before - bag_after

    # A tuple that is gone can mean two unrelated things, and calling both "rows
    # lost" put value rewrites and deletions in one bucket. If the row count did
    # not fall, nothing was deleted; the values inside the shared columns were
    # rewritten, and the question becomes whether the new value is derivable from
    # the old, which is not the question row counting answers.
    count_before = sum(bag_before.values())
    count_after = sum(bag_after.values())
    out["rows_changed"] = sum(missing.values())

    if not shared and before_cols:
        out["verdict"] = "no-shared-columns"
        out["detail"] = "column names share nothing; compare by hand"
    elif missing and count_after < count_before:
        out["verdict"] = "rows-dropped"
        out["detail"] = (f"{count_before - count_after} fewer row(s) after, "
                         f"{sum(missing.values())} tuple(s) not carried over")
    elif missing:
        out["verdict"] = "values-rewritten"
        out["detail"] = (f"row count holds at {count_after}, but "
                         f"{sum(missing.values())} row(s) changed value in a shared column")
    elif lost_cols:
        out["verdict"] = "columns-lost"
        out["detail"] = f"every row survives, but {len(lost_cols)} column(s) do not"
    elif added_cols:
        out["verdict"] = "adds-columns"
        out["detail"] = "no row lost; derivability of the added columns needs a human"
    else:
        out["verdict"] = "preserves"
        out["detail"] = "same rows, same columns"
    if skipped and out["verdict"] in ("preserves", "adds-columns"):
        out["detail"] += f"; {len(skipped)} file(s) unread, so this is not a clean bill"
    return out


def load_set(path):
    """(dataset, sha) pairs from a labeled-set CSV, deduplicated, order kept."""
    pairs, seen = [], set()
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            keys = {k.lower().replace("_", "").replace(" ", ""): k for k in row}
            ds = row.get(keys.get("datasetid") or keys.get("dataset") or "", "")
            sha = row.get(keys.get("commitid") or keys.get("commit") or "", "")
            ds, sha = (ds or "").strip(), (sha or "").strip()
            if not ds or not sha or (ds, sha) in seen:
                continue
            seen.add((ds, sha))
            pairs.append((ds, sha))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("commit", nargs="?")
    parser.add_argument("--set", dest="labeled_set",
                        help="CSV of labeled commits to sweep")
    parser.add_argument("--out", help="write the per-commit verdicts here")
    args = parser.parse_args()

    if args.labeled_set:
        pairs = load_set(args.labeled_set)
    elif args.dataset and args.commit:
        pairs = [(args.dataset, args.commit)]
    else:
        parser.error("give a dataset and commit, or --set")

    results = []
    for i, (dataset, sha) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {dataset} @ {sha[:10]}", file=sys.stderr, flush=True)
        result = check(dataset, sha)
        results.append(result)
        print(f"    {result['verdict']}: {result['detail']}", file=sys.stderr, flush=True)

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nwrote {args.out}", file=sys.stderr)

    tally = Counter(r["verdict"] for r in results)
    print("\n=== verdicts ===")
    for verdict, count in tally.most_common():
        print(f"{count:5d}  {verdict}")
    unread = sum(1 for r in results if r["skipped"])
    print(f"\n{unread} commit(s) had at least one file this check could not read.")


if __name__ == "__main__":
    main()
