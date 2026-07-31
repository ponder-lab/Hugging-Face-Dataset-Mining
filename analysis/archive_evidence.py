#!/usr/bin/env python3
"""Evidence archive for the candidate set (#65): one committed file per row.

Verification re-clones each dataset from Hugging Face on demand, so a candidate
stays checkable only as long as its upstream repo does. That assumption fails
in practice: repos get deleted, renamed, and history-rewritten within weeks of
mining. This script freezes, for every row of the candidate CSV, the report
`analysis/inspect_commit.py` produces (with --download --show_rows), so that
every label is reproducible from this repository alone, with no network.

Each row gets exactly one file under data/evidence/, failures included. A row
whose commit can no longer be read appears carrying the reason, rather than
being omitted: an archive that silently drops it says "we did not try" and "we
could not read this" in the same voice, which is the lesson of #34, #55, #59
and #72 one layer up.

Deciding that a commit is gone takes two checks rather than one, because a
stale local clone is not the same thing as a deleted commit:
  1. `git rev-parse <sha>^{commit}` against a full local clone, after fetching;
  2. the Hub tree endpoint at that revision.
A repository that answers 200 while its tree endpoint 404s at the mined
revision has rewritten history, and that is the case recorded by name.

Usage:
  python analysis/archive_evidence.py              # every row; existing files kept
  python analysis/archive_evidence.py --force      # regenerate everything
  python analysis/archive_evidence.py --limit N    # first N rows, to measure size
  python analysis/archive_evidence.py --dataset D  # only rows of one dataset
"""
import argparse, csv, os, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inspect_commit as ic

# Held repo-relative as well as resolved, for the same reason inspect_commit
# holds its CSV path both ways: text that names this directory should name it
# the way the repo does.
EVIDENCE = "data/evidence"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       *EVIDENCE.split("/"))
INSPECT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "inspect_commit.py")
INDEX = "INDEX.csv"
TREE = "https://huggingface.co/api/datasets/{ds}/tree/{rev}"

WORKERS = 4        # datasets archived concurrently; rows within one dataset
                   # run in order, so no two workers touch the same clone
ROW_TIMEOUT = 600  # seconds per inspect run; a hang must not stall the sweep

# Row dispositions, written on the `status:` line of every record and echoed
# into INDEX.csv. One tag per kind of outcome, so the index can be filtered
# without parsing prose.
ARCHIVED = "archived"                  # the report was generated and stored
UNREACHABLE = "commit-unreachable"     # both checks say the commit is gone
REPO_GONE = "repo-gone"                # no clone: deleted, private, or never there
REPO_RESTRICTED = "repo-restricted"    # no clone: access-controlled
ERROR = "error"                        # our side failed; a rerun may differ

SEP = "=" * 60  # divides the provenance header from the archived report


def run(*a):
    return subprocess.run(a, capture_output=True, text=True, errors="replace")


def evidence_filename(ds, sha):
    """File name for one row, matching the clone cache's `/` -> `__` mapping."""
    return f"{ds.replace('/', '__')}__{sha[:10]}.txt"


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def header_lines(row, status, probe, stamp):
    """Provenance header for one record.

    The mined identity comes from the CSV, not from the repo read: when the Hub
    reports a rename, the corpus names one repo and the report below reads
    another, and the paper should be able to state both (#65). `hub:` records
    where the repo stood at generation time, which is a sweep result in its own
    right and changes meaning as upstream drifts.
    """
    lines = [
        f"dataset:    {row['DatasetID']}",
        f"commit:     {row['CommitId']}",
        f"type:       {row['tentative_type']}",
        f"message:    {row['log_message'].splitlines()[0] if row['log_message'] else ''}",
        f"status:     {status}",
        f"hub:        {probe.kind} ({probe.detail})" if probe else "hub:        (not probed)",
    ]
    if probe and probe.kind == ic.MOVED:
        lines.append(f"renamed_to: {probe.moved_to or '(new name not reported)'}")
    lines.append(f"generated:  {stamp} by analysis/archive_evidence.py, "
                 f"wrapping inspect_commit.py --download --show_rows")
    return lines


def write_record(out_dir, row, status, probe, body, stamp=None):
    """Write one row's record atomically and return its file name.

    Atomic so an interrupted sweep leaves no truncated file behind for the
    skip-if-exists rerun to mistake for a finished one.
    """
    name = evidence_filename(row["DatasetID"], row["CommitId"])
    text = "\n".join(header_lines(row, status, probe, stamp or utc_stamp()))
    text += "\n" + SEP + "\n" + body
    if not text.endswith("\n"):
        text += "\n"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(path + ".tmp", path)
    return name


def parse_record(path):
    """The header fields of a record, as a dict, without reading the body."""
    fields = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == SEP:
                break
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def tree_status(ds, sha):
    """What the Hub tree endpoint answers at this revision, as `HTTP nnn`.

    This is the second of the two checks: the local clone says what our copy
    holds, this says what the Hub still serves. A probe that does not land
    returns the exception name and decides nothing.
    """
    try:
        r = requests.get(TREE.format(ds=ds, rev=sha), headers=ic._auth_headers(),
                         timeout=ic.TIMEOUT, stream=True)
        r.close()
        return f"HTTP {r.status_code}"
    except requests.RequestException as e:
        return type(e).__name__


def unreachable_record(ds, sha, probe, fetch_notes, tree):
    """Body of the record for a commit neither we nor the Hub can produce.

    Written to deny the reading it exists to prevent: an absent report must
    never leave the impression of a commit that changed nothing (#71, #72).
    Both checks appear with their outcomes, so a later reader can see what was
    actually established rather than trusting a bare verdict.
    """
    lines = [
        f"NOTHING WAS ANALYZED: commit {sha} cannot be read from "
        f"{ds} any more.",
        "",
        "This is a revision no repository will produce, not a commit that",
        "changed no files. Two checks, because a stale local clone is not the",
        "same thing as a deleted commit:",
        f"  1. local full clone: `git rev-parse {sha[:10]}^{{commit}}` finds no "
        f"commit",
    ]
    lines += [f"     {note}" for note in fetch_notes]
    lines.append(f"  2. Hub tree endpoint at this revision: {tree}")
    lines.append("")
    if tree == "HTTP 404" and probe and probe.kind == ic.REACHABLE:
        lines.append(
            f"The repository itself still answers on the Hub "
            f"({probe.detail}) and still serves a history; that history no "
            f"longer contains this commit. Upstream history was rewritten "
            f"after mining.")
    elif probe and probe.kind in (ic.GONE, ic.RESTRICTED):
        lines.append(
            f"The repository is no longer readable either ({probe.kind}, "
            f"{probe.detail}), so the commit went down with it.")
    else:
        lines.append(
            f"The Hub's answer ({tree}) does not confirm the rewrite on its "
            f"own; what stands is that a full local clone, fetched at "
            f"generation time, does not hold the commit. Retry the sweep to "
            f"settle it.")
    return "\n".join(lines)


def no_clone_record(ds, status):
    """Body of the record for a dataset no clone could be had of.

    The prose is inspect_commit's clone_failure, so a reader of the archive
    sees the same explanation a rater at the terminal would.
    """
    return ("NOTHING WAS ANALYZED: the dataset could not be cloned.\n\n"
            + ic.clone_failure(ds, status))


def try_clone(ds):
    """(path, None) for a clone of ds, or (None, why) when none can be had."""
    try:
        return ic.clone(ds), None
    except SystemExit as e:
        return None, str(e.code)


def reach_commit(ds, repo, sha):
    """Try to make `sha` resolvable in `repo`; notes on what was tried.

    Returns (resolved, fetch_notes). The clone may simply be stale, so fetch
    the branch refs first; the commit may also live outside them (the Hub keeps
    refs git clone does not copy), so ask for the SHA itself second.
    """
    notes = []
    for what, args in (("`git fetch origin`", ("fetch", "--quiet", "origin")),
                       (f"`git fetch origin {sha[:10]}`",
                        ("fetch", "--quiet", "origin", sha))):
        r = run("git", "-C", repo, *args)
        outcome = "completed" if r.returncode == 0 else \
            f"failed ({(r.stderr.strip().splitlines() or ['no output'])[-1]})"
        notes.append(f"{what} {outcome}; commit still absent")
        if ic.resolve_commit(repo, sha):
            return True, notes
    return False, notes


def run_inspect(ds, sha):
    """One row's report, stderr merged in stream order, or an error body.

    A subprocess rather than an in-process call: inspect() writes to real
    stdout/stderr and exits on failure, and merging the two streams at the pipe
    keeps each [warn] next to the file it warns about.
    """
    try:
        r = subprocess.run(
            [sys.executable, INSPECT, ds, sha, "--download", "--show_rows"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", timeout=ROW_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        out = e.output or ""
        return ERROR, (f"NOTHING WAS ANALYZED in time: inspect_commit.py did "
                       f"not finish within {ROW_TIMEOUT}s and was stopped. "
                       f"Partial output follows.\n\n{out}")
    if r.returncode != 0:
        return ERROR, (f"NOTHING WAS ANALYZED: inspect_commit.py exited "
                       f"{r.returncode}. Its output follows.\n\n{r.stdout}")
    return ARCHIVED, r.stdout


_print_lock = threading.Lock()
_done = [0]


def progress(total, status, ds, sha):
    with _print_lock:
        _done[0] += 1
        print(f"[{_done[0]}/{total}] {status:20} {ds} @ {sha[:10]}")


def archive_dataset(ds, rows, out_dir, force, total):
    """Archive every row of one dataset, in order, sharing one clone."""
    results = []
    probe = ic.repo_status(ds)
    repo, clone_err = try_clone(ds)
    for row in rows:
        sha = row["CommitId"]
        path = os.path.join(out_dir, evidence_filename(ds, sha))
        if not force and os.path.exists(path):
            status = parse_record(path).get("status", ERROR)
            progress(total, f"kept ({status})", ds, sha)
            results.append(status)
            continue
        if repo is None:
            status = {ic.GONE: REPO_GONE,
                      ic.RESTRICTED: REPO_RESTRICTED}.get(probe.kind, ERROR)
            write_record(out_dir, row, status, probe, no_clone_record(ds, probe))
        elif ic.resolve_commit(repo, sha) is None:
            resolved, notes = reach_commit(ds, repo, sha)
            if resolved:
                status, body = run_inspect(ds, sha)
                write_record(out_dir, row, status, probe, body)
            else:
                status = UNREACHABLE
                write_record(out_dir, row, status, probe,
                             unreachable_record(ds, sha, probe, notes,
                                                tree_status(ds, sha)))
        else:
            status, body = run_inspect(ds, sha)
            write_record(out_dir, row, status, probe, body)
        progress(total, status, ds, sha)
        results.append(status)
    return results


def load_rows():
    with open(ic.CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_index(rows, out_dir):
    """INDEX.csv: one line per candidate row, in candidate-CSV order.

    Rebuilt from the record files themselves rather than from this run's
    in-memory results, so a partial run indexes what is actually on disk. A row
    with no record yet is listed as such, never dropped: the index must not
    say "we did not try" and "we could not read this" in the same voice.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, INDEX), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["DatasetID", "CommitId", "tentative_type", "status", "hub",
                    "renamed_to", "evidence_file"])
        for row in rows:
            name = evidence_filename(row["DatasetID"], row["CommitId"])
            path = os.path.join(out_dir, name)
            if os.path.exists(path):
                rec = parse_record(path)
                w.writerow([row["DatasetID"], row["CommitId"],
                            row["tentative_type"], rec.get("status", ""),
                            rec.get("hub", ""), rec.get("renamed_to", ""),
                            name])
            else:
                w.writerow([row["DatasetID"], row["CommitId"],
                            row["tentative_type"], "not-generated", "", "", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="regenerate records that already exist")
    ap.add_argument("--limit", type=int,
                    help="only the first N rows of the candidate CSV")
    ap.add_argument("--dataset", help="only rows of this dataset ID")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()

    rows = load_rows()
    picked = rows
    if a.dataset:
        picked = [r for r in picked if r["DatasetID"] == a.dataset]
    if a.limit:
        picked = picked[:a.limit]

    # Grouped by dataset so one worker owns one clone; rows of the same dataset
    # never race on git state.
    groups = {}
    for row in picked:
        groups.setdefault(row["DatasetID"], []).append(row)

    statuses = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(archive_dataset, ds, g, OUT_DIR, a.force,
                               len(picked))
                   for ds, g in groups.items()]
        for fut in futures:
            statuses.extend(fut.result())

    # The index always covers the whole CSV, whatever subset this run touched.
    build_index(rows, OUT_DIR)

    print("\nby status:")
    for tag in sorted(set(statuses)):
        print(f"  {tag}: {statuses.count(tag)}")
    sizes = [os.path.getsize(os.path.join(OUT_DIR, f))
             for f in os.listdir(OUT_DIR) if f.endswith(".txt")]
    if sizes:
        print(f"\n{len(sizes)} records, {sum(sizes):,} bytes total, "
              f"mean {sum(sizes) // len(sizes):,}, max {max(sizes):,}")


if __name__ == "__main__":
    main()
