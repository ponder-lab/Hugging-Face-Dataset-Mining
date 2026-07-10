#!/usr/bin/env python3
"""Verification helper for message-visible data-refactoring candidates.

Given a Hugging Face dataset and a commit, show enough to decide whether the
commit is a genuine data refactoring, WITHOUT downloading large LFS payloads.

Usage:
  python analysis/inspect_commit.py <dataset_id> <commit_sha>
  python analysis/inspect_commit.py --list [--type TYPE]   # list candidates from the CSV

What you see:
  - the commit message
  - file-level changes (rename/add/delete/modify) for data files
  - for MODIFIED non-LFS CSVs, the column-header diff vs the parent commit
  - LFS-tracked files are flagged (download a version to inspect those)

Renames/adds/deletes are visible from Git history alone (no download). Only
in-file changes to LFS-stored files (often parquet) need an actual download.
"""
import argparse, csv, os, subprocess, sys

CACHE = os.path.expanduser("~/.cache/hf-dataset-clones")
CSV = os.path.join(os.path.dirname(__file__), "..", "data", "message_refactoring_candidates.csv")

NO_DOWNLOAD = object()

def run(*a):
    return subprocess.run(a, capture_output=True, text=True, errors="replace")

def clone(ds):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, ds.replace("/", "__"))
    if not os.path.isdir(os.path.join(path, ".git")):
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
        if subprocess.run(["git","clone","--quiet",
                           f"https://huggingface.co/datasets/{ds}",path], env=env).returncode:
            sys.exit(f"clone failed for {ds}")
    return path

def show(repo,*a): return run("git","-C",repo,*a).stdout

def is_lfs_pointer(text):
    """True if file content is a Git LFS pointer (actual data not present)."""
    return text[:25].startswith("version https://git-lfs")

def parse_csv_header(text):
    """Column names from CSV text, or None if it is an LFS pointer.""" 

    if(is_lfs_pointer(text)):
        return None
    
    line = text.split("\n",1)[0]

    if not line.strip(): return []
    return next(csv.reader([line]))

ABSENT = object()  # blob not present at a revision (distinct from an LFS pointer)
UNRESOLVED = object()  # an LFS pointer whose content we could not fetch

HEADER_READ_CAP = 1 << 20  # bytes; a CSV header line is tiny, cap so a binary
                           # blob (e.g. parquet) with no early newline cannot
                           # stream unbounded into memory

def load_lfs_pointer(repo, pointer):
    """First line of the content behind an LFS pointer, or UNRESOLVED.

    `git lfs smudge` takes the pointer on stdin and writes the content to
    stdout, so we never touch the working tree or the index. It also exits 0
    and echoes the pointer straight back when it cannot fetch the object, so
    the caller must check the output rather than the exit status.
    """
    # GIT_TERMINAL_PROMPT=0 so a smudge against an auth-requiring LFS remote
    # fails fast instead of blocking on a credential prompt (matches clone()).
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    p = subprocess.Popen(["git", "-C", repo, "lfs", "smudge"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, env=env)
    try:
        p.stdin.write(pointer.encode())
        p.stdin.close()   # signal EOF so smudge produces its output
        # readline(cap) returns as soon as the first newline arrives, and reads
        # no more than HEADER_READ_CAP bytes if one never does. So a normal CSV
        # header returns immediately, and a newline-free binary payload cannot
        # hang or balloon memory.
        raw = p.stdout.readline(HEADER_READ_CAP) or b""
        line = raw.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    except OSError:
        return UNRESOLVED
    finally:
        p.kill()          # we only ever want the header; do not stream a whole dataset
        # Close both pipes on every path. If write() raised above, stdin is
        # still open; leaving it open leaks an fd across repeated calls.
        for stream in (p.stdin, p.stdout):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        p.wait()

    # Smudge failed if what came back is the pointer we sent in. Never let
    # that read as "this file has no columns".
    if not line or is_lfs_pointer(line):
        return UNRESOLVED

    # A resolved blob need not be parseable CSV: a binary payload (e.g. parquet)
    # yields a huge single "field" that trips csv's field-size limit. Treat any
    # unparseable header as unresolved rather than crashing inspect().
    try:
        return parse_csv_header(line)
    except csv.Error:
        return UNRESOLVED

def header(repo, rev, path,download):
    r = run("git", "-C", repo, "show", f"{rev}:{path}")
    if r.returncode != 0:
        return ABSENT  # git could not read the blob at this revision
    if is_lfs_pointer(r.stdout):
        if download:
            return load_lfs_pointer(repo, r.stdout)
        else:
            return NO_DOWNLOAD

    return parse_csv_header(r.stdout)

def inspect(ds, sha, download, show_rows):
    repo = clone(ds)
    print(f"# {ds} @ {sha[:10]}")
    print("message:", show(repo,"log","-1","--pretty=%s",sha).strip(), "\n")
    parent = show(repo,"rev-parse",f"{sha}^").strip()
    ns = show(repo,"show","--name-status","--find-renames","--pretty=format:",sha).strip()
    print("file changes:\n" + (ns or "  (none)"))
    for line in ns.splitlines():
        p = line.split("\t")
        if not p[0].startswith("M") or not p[-1].lower().endswith(".csv"): continue
        path = p[-1]
        h = header(repo, sha, path, download)
        if h is ABSENT:
            print(f"  [warn] git could not read {path} at {sha[:10]}; skipping", file=sys.stderr)
            continue

        if h is NO_DOWNLOAD:
            print(f"\n[{path}] LFS-tracked -> download a version to inspect the data change")
            continue

        if h is UNRESOLVED:
            print(f"  [warn] could not fetch LFS content for {path} at {sha[:10]} "
                  f"(is the LFS remote reachable?); cannot compare columns", file=sys.stderr)
            continue

        pc = header(repo, parent, path,download) if parent else None
        if parent and not isinstance(pc, list):
            # ABSENT/NO_DOWNLOAD/UNRESOLVED on the parent: we cannot diff columns.
            # Say so rather than silently falling through to "no column change".
            print(f"  [warn] could not read parent columns for {path}; "
                  f"column diff skipped", file=sys.stderr)
        if isinstance(pc, list) and set(h) != set(pc):
            print(f"\n[{path}] column change:")
            print(f"  removed: {sorted(set(pc)-set(h))}")
            print(f"  added:   {sorted(set(h)-set(pc))}")
        elif p[0] == 'M' and show_rows:
            print()
            r = run("git","-C", repo,"-c", "diff.lfs.textconv=cat","diff","--color-words","--textconv",parent,sha, "--",path)
            output = r.stdout.splitlines()
            for line in output[:50]:
                print(line)
                print()

def list_candidates(typ=None):
    with open(CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if typ and row["tentative_type"] != typ: continue
            print(f"{row['tentative_type']:32} {row['DatasetID']:42} {row['CommitId'][:10]}  {row['log_message'][:55]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?"); ap.add_argument("commit", nargs="?")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--type")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--show_rows", action="store_true")
    a = ap.parse_args()
    if a.list: list_candidates(a.type)
    elif a.dataset and a.commit: inspect(a.dataset, a.commit,a.download,a.show_rows)
    else: ap.print_help()
