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

Renames/adds/deletes are visible from git history alone (no download). Only
in-file changes to LFS-stored files (often parquet) need an actual download.
"""
import argparse, csv, os, subprocess, sys

CACHE = os.path.expanduser("~/.cache/hf-dataset-clones")
CSV = os.path.join(os.path.dirname(__file__), "..", "data", "message_refactoring_candidates.csv")

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
    if is_lfs_pointer(text): return None
    line = text.split("\n",1)[0]
    if not line.strip(): return []
    return next(csv.reader([line]))

def header(repo, rev, path):
    r = run("git", "-C", repo, "show", f"{rev}:{path}")
    if r.returncode != 0:
        return None  # blob absent at this revision; not comparable
    return parse_csv_header(r.stdout)

def inspect(ds, sha):
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
        h = header(repo, sha, path)
        if h is None:
            print(f"\n[{path}] LFS-tracked -> download a version to inspect the data change"); continue
        pc = header(repo, parent, path) if parent else None
        if pc is not None and set(h) != set(pc):
            print(f"\n[{path}] column change:")
            print(f"  removed: {sorted(set(pc)-set(h))}")
            print(f"  added:   {sorted(set(h)-set(pc))}")

def list_candidates(typ=None):
    with open(CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if typ and row["tentative_type"] != typ: continue
            print(f"{row['tentative_type']:32} {row['DatasetID']:42} {row['CommitId'][:10]}  {row['log_message'][:55]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?"); ap.add_argument("commit", nargs="?")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--type")
    a = ap.parse_args()
    if a.list: list_candidates(a.type)
    elif a.dataset and a.commit: inspect(a.dataset, a.commit)
    else: ap.print_help()
