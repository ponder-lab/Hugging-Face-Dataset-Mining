#!/usr/bin/env python3
"""Check the arithmetic behind a preservation judgment. It does not make one.

WHAT THIS IS FOR

Whether a change preserved a dataset's information content is a semantic
judgment, and in this study it is made by human annotators against a written
criterion, with disagreements adjudicated against the diff. That is the
instrument. This script is not a second one, and nothing it prints is a verdict
on preservation.

What it does is the arithmetic underneath such a judgment, which is where people
reliably fail. Nobody can eyeball whether 615 rows plus 153 rows account for all
768, whether the parts of a split union to the whole, or whether 4,473 rows
dropped from one file reappear in another the same commit added. Each of those
was got wrong by a careful reader on this corpus before it was checked:

  khoaguin/pima-indians-diabetes-database @ a19fc7ccdb  was recorded as dropping
    rows. It is an exact 768-row partition.

  imageomics/Heliconius-Collection_Cambridge-Butterfly @ 6ef14cc5  looked like the
    largest deletion in the corpus. Every removed row is retained in a master file
    the same commit adds.

  djulian13/east-slavic-swadesh-lists @ a4c5651f  was read by two annotators as
    shredding its column names. It changed its field separator (#79).

WHAT IT CANNOT DO, WHICH IS MOST OF THE CRITERION

The criterion asks whether every fact recoverable before is recoverable after, and
whether anything introduced was already derivable. Derivability is about meaning,
and nothing here sees meaning:

  - An added column holding a computed ratio and one holding a model's output are
    indistinguishable to this script. It reports `adds-columns` and stops.
  - A column that disappears may be a rename, which preserves, or a deletion,
    which may not. It reports `columns-lost` and stops.
  - A rewritten value may be a normalization or a loss. It reports
    `values-rewritten` and stops.
  - Deduplication removes rows while preserving every distinct fact, so a
    `rows-dropped` verdict is WRONG by the criterion whenever duplicates are what
    went. That has already happened here once.

So `preserves` means the narrow, literal thing: the rows and column names it could
read are the same at both revisions. That is a sufficient condition for no loss,
not the criterion itself. Every other verdict is a question handed to a person,
and the counts it prints are commits screened, never preservation decided. Read a
percentage of them as a percentage of nothing in particular.

USAGE

  python3 analysis/preservation_check.py <dataset> <commit>
  python3 analysis/preservation_check.py --set tests/verified_set.csv --out /tmp/screen.csv

It reads each revision's separator through inspect_commit rather than
reimplementing that rule, so the two cannot drift apart on what a file says.
LFS-tracked files are streamed from the Hub and discarded rather than stored, and
revisions are resolved to concrete SHAs first, because the Hub does not accept git
revision expressions and every parent-side fetch otherwise 404s. Files it declines
to read are named, never folded into a silent total.
"""
import argparse
import csv
import gzip
import io
import os
import gzip as _gzip
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import time
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
RESOLVE = "https://huggingface.co/datasets/{ds}/resolve/{rev}/{path}"
# An LFS payload is streamed and discarded, never stored: the check needs the rows,
# not the file, and the corpus's LFS content is 106 GB against a few megabytes of
# row digests. The cap is on bytes pulled for ONE file, and what it turns away is
# named in the output rather than folded into the ordinary unreadable count.
LFS_BYTE_CAP = int(os.environ.get("PRESERVATION_LFS_CAP", 100 * 1024 * 1024))
LFS_TIMEOUT = 180
LFS_ATTEMPTS = 4          # a sweep makes hundreds of requests and the Hub throttles
LFS_BACKOFF = 3.0         # seconds, doubled per retry
# Real corpora hold cells far past csv's 128 KB default, and hitting one raises
# rather than returning, so one pathological file used to end the whole sweep.
csv.field_size_limit(1 << 31)
# A tree can hold more data than belongs in memory. Rows past this are not read,
# and a file that hits it is named in the output rather than silently truncated.
MAX_ROWS_PER_FILE = 2_000_000
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


_REV_CACHE = {}


def resolve_rev(repo, rev):
    """A concrete commit SHA for `rev`, or None.

    The Hub's resolve endpoint speaks commit SHAs and branch names, not git
    revision expressions, so `<sha>~1` is a 404 there while being perfectly
    valid locally. Every parent-side LFS fetch in one sweep failed this way and
    was recorded as the file being missing, which is the half of the comparison
    that decides whether rows were relocated or destroyed.
    """
    key = (repo, rev)
    if key not in _REV_CACHE:
        done = run(["git", "-C", repo, "rev-parse", rev])
        _REV_CACHE[key] = (done.stdout.decode().strip()
                           if done.returncode == 0 else None)
    return _REV_CACHE[key]


def lfs_size(pointer):
    """Payload size recorded in an LFS pointer, or None."""
    for line in pointer.decode("utf-8", "replace").splitlines():
        if line.startswith("size "):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def lfs_stream(dataset, rev, path, cap):
    """Bytes of an LFS-tracked file, pulled from the Hub and never written to disk.

    Only the pointer is in the repository, so a commit whose data lives in LFS
    was previously unreadable, and that is not a neutral gap: a commit that moves
    rows into an LFS file looks exactly like one that deletes them. Half of this
    corpus is in that state, and the bias runs toward reporting loss on the
    best-organized datasets, which is where reorganization is likeliest.
    """
    url = RESOLVE.format(ds=dataset, rev=rev,
                         path=urllib.parse.quote(path))
    req = urllib.request.Request(url, headers={"User-Agent": "preservation-check"})
    # The status code is recorded rather than collapsed into the exception name.
    # Reporting a throttled 429 and a genuinely missing 404 under one label made
    # a self-inflicted gap indistinguishable from a fact about the corpus, and a
    # fifth of one sweep's "unreadable" files turned out to be the former.
    for attempt in range(LFS_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=LFS_TIMEOUT) as response:
                out, total = [], 0
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > cap:
                        return None, "lfs-over-cap"
                    out.append(chunk)
            return b"".join(out), None
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < LFS_ATTEMPTS - 1:
                time.sleep(LFS_BACKOFF * (2 ** attempt))
                continue
            return None, f"lfs-http-{exc.code}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if attempt < LFS_ATTEMPTS - 1:
                time.sleep(LFS_BACKOFF * (2 ** attempt))
                continue
            return None, f"lfs-fetch-failed({type(exc).__name__})"
    return None, "lfs-fetch-failed(retries-exhausted)"


def read_table(repo, rev, path, dataset=None):
    """(columns, rows) for one tabular file, or ('skip', reason)."""
    data = blob(repo, rev, path)
    if data is None:
        return None, "absent"
    if data.startswith(LFS_MARKER):
        if dataset is None:
            return None, "lfs-pointer"
        size = lfs_size(data)
        if size is not None and size > LFS_BYTE_CAP:
            return None, "lfs-over-cap"
        concrete = resolve_rev(repo, rev)
        if concrete is None:
            return None, "rev-unresolvable"
        data, why = lfs_stream(dataset, concrete, path, LFS_BYTE_CAP)
        if data is None:
            return None, why
    data = decompress(path, data)
    if data is None:
        return None, "undecompressable"
    if b"\x00" in data[:8192]:
        # A .csv holding NUL bytes is a binary payload under a text name, most
        # often parquet. That is not a delimiter the sniffer failed to find.
        return None, "binary-content"
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
    try:
        for i, row in enumerate(reader):
            if i >= MAX_ROWS_PER_FILE:
                return None, f"over-{MAX_ROWS_PER_FILE}-rows"
            rows.append(row)
    except csv.Error as exc:
        return None, f"csv-error({str(exc)[:40]})"
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


def tree_content(repo, rev, paths, dataset=None):
    """Columns and rows held by `paths` at `rev`.

    Returns (columns, tables, skipped). `columns` is the union over the files,
    since a fact recorded in any file of the dataset is recoverable from it.
    """
    columns, tables, skipped, absent = [], [], [], 0
    for path in sorted(paths):
        result, detail = read_table(repo, rev, path, dataset)
        if result is None:
            # A file added by the commit is absent from the parent, and one the
            # commit deleted is absent from the child. That is the change being
            # measured, not a failure to read it, and counting it as a skip
            # marked almost every add and delete as unverified.
            if detail != "absent":
                skipped.append(f"{path}:{detail}")
            else:
                absent += 1
            continue
        header, rows = result
        for name in header:
            if name not in columns:
                columns.append(name)
        tables.append((path, header, rows))
    return columns, tables, skipped, absent


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

    before_cols, before, skip_b, absent_b = tree_content(repo, f"{sha}~1", paths, dataset)
    after_cols, after, skip_a, absent_a = tree_content(repo, sha, paths, dataset)

    skipped = sorted(set(skip_b or []) | set(skip_a or []))
    out["skipped"] = "; ".join(skipped)
    if not before and not after:
        out["verdict"] = "unreadable"
        out["detail"] = "no readable tabular file at either revision"
        return out
    if not before and not skip_b and absent_b:
        # Every path the commit touched is new. There was no prior state, so
        # preservation is not a question this commit can fail, and filing it
        # under unreadable would count a non-question as a coverage gap.
        out["verdict"] = "adds-files"
        out["detail"] = f"{absent_b} file(s) added; no prior state to preserve"
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
        try:
            result = check(dataset, sha)
        except Exception as exc:  # noqa: BLE001
            # A sweep over a few hundred repositories meets malformed data that
            # no amount of anticipation covers, and losing every completed
            # verdict to the last one is worse than recording the failure.
            result = {"dataset": dataset, "commit": sha, "verdict": "check-failed",
                      "detail": f"{type(exc).__name__}: {str(exc)[:120]}",
                      "rows_before": "", "rows_after": "", "rows_changed": "",
                      "columns_lost": "", "columns_added": "", "skipped": ""}
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
