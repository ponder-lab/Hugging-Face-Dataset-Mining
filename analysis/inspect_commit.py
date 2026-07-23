#!/usr/bin/env python3
"""Verification helper for message-visible data-refactoring candidates.

Given a Hugging Face dataset and a commit, show enough to decide whether the
commit is a genuine data refactoring, WITHOUT downloading large LFS payloads.

Usage:
  python analysis/inspect_commit.py <dataset_id> <commit_sha>
  python analysis/inspect_commit.py --list [--type TYPE]   # list candidates from the CSV

What you see:
  - the commit message
  - file-level changes (rename/add/delete/modify), each flagged if the file is
    stored in Git LFS at that commit
  - for MODIFIED non-LFS CSVs, the column-header diff vs the parent commit
  - LFS-tracked files are flagged (download a version to inspect those)

Renames/adds/deletes are visible from Git history alone (no download). Only
in-file changes to LFS-stored files (often parquet) need an actual download,
and that download is a ranged HTTP read of the header row, not a full payload.
"""
import argparse, csv, os, subprocess, sys, time

import requests

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

def lfs_status(repo, rev, path):
    """Whether the blob at rev:path is stored in Git LFS.

    Returns "lfs" if the blob is a pointer, "plain" if it is the real content,
    or None if the blob is absent at that revision. Needs no download: under the
    GIT_LFS_SKIP_SMUDGE clone the stored blob IS the pointer text, so `git show`
    reveals LFS tracking without fetching the payload.
    """
    r = run("git", "-C", repo, "show", f"{rev}:{path}")
    if r.returncode != 0:
        return None
    return "lfs" if is_lfs_pointer(r.stdout) else "plain"

def parse_csv_header(text):
    """Column names from CSV text, or None if it is an LFS pointer.""" 

    if(is_lfs_pointer(text)):
        return None
    
    line = text.split("\n",1)[0]

    if not line.strip(): return []
    return next(csv.reader([line]))

class Unread:
    """Why a header could not be turned into a column list, kept distinct so a
    rater can act on it.

    `kind` is a stable tag inspect() dispatches on; `retryable` says whether trying
    again could change the outcome (a transient transport error) or not (the bytes
    are simply not there); `detail` is a short specific such as an HTTP status.
    """
    __slots__ = ("kind", "retryable", "detail")

    def __init__(self, kind, retryable=False, detail=""):
        self.kind = kind
        self.retryable = retryable
        self.detail = detail

    def __repr__(self):
        return f"Unread({self.kind!r}, retryable={self.retryable}, detail={self.detail!r})"

# Dispositions with no per-call detail are shared singletons; callers and tests
# rely on the `is` identity.
ABSENT = Unread("absent")            # git has no blob for this path at this revision
NO_DOWNLOAD = Unread("no_download")  # LFS-tracked and --download was not passed

HEADER_READ_CAP = 1 << 20  # bytes; a CSV header line is tiny, cap so a binary
                           # blob (e.g. parquet) with no early newline cannot
                           # stream unbounded into memory

RESOLVE = "https://huggingface.co/datasets/{ds}/resolve/{rev}/{path}"
TIMEOUT = 30   # seconds, per HTTP request
RETRIES = 3    # attempts for a transient transport error
BACKOFF = 0.5  # seconds before the first retry, doubled after each failed attempt

def _fetch_once(url, headers):
    """One attempt at the header: a column list, or an Unread saying why not."""
    try:
        with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
            code = r.status_code
            if code not in (200, 206):
                # 401/403: private or gated, a retry cannot get in. 404: no blob
                # for this path at this revision, terminal. 5xx/429 and anything
                # else: a server-side or throttling hiccup a retry could clear.
                if code in (401, 403):
                    return Unread("access", detail=f"HTTP {code}")
                if code == 404:
                    return Unread("content_absent", detail="HTTP 404")
                return Unread("transport", retryable=True, detail=f"HTTP {code}")
            # Stop at the first newline: a normal CSV header returns after one
            # chunk, and a newline-free binary payload (e.g. parquet) cannot
            # stream unbounded into memory.
            buf = b""
            for chunk in r.iter_content(8192):
                buf += chunk
                if b"\n" in buf or len(buf) >= HEADER_READ_CAP:
                    break
    except requests.RequestException as e:
        return Unread("transport", retryable=True, detail=type(e).__name__)

    line = buf[:HEADER_READ_CAP].split(b"\n", 1)[0].decode("utf-8", errors="replace")

    # Nothing came back, or the bytes are the LFS pointer itself: the object was
    # never materialized for this revision (a Xet-migrated repo can serve the
    # pointer text as the file's content, #47). Absent bytes will not appear on a
    # retry, so this is terminal, not transport.
    if not line or is_lfs_pointer(line):
        return Unread("content_absent", detail="Hub served an LFS pointer")

    # A resolved blob need not be parseable CSV: a binary payload (e.g. parquet)
    # yields a huge single "field" that trips csv's field-size limit. That is a
    # format we do not read here (see #43), not a fetch failure.
    try:
        return parse_csv_header(line)
    except csv.Error:
        return Unread("not_csv")

def fetch_header(ds, rev, path):
    """First line of a file at a revision on the Hub, or an Unread saying why not.

    Reads over HTTP rather than through `git lfs smudge`. Repos migrated to Xet
    storage serve no classic LFS object, so smudge cannot fetch them; it exits 0
    and echoes the pointer straight back, which read as "LFS is broken for this
    repo" (#44). The resolve endpoint serves both storage backends, redirecting
    to whichever CDN holds the content.

    A transient transport error (5xx, throttling, a dropped connection) is retried
    with backoff before giving up; a terminal outcome (absent bytes, a gated repo,
    a non-CSV payload) is returned at once (#47). Only the header row is wanted, so
    this is a Range request, capped again client-side in case the server ignores
    the range.
    """
    url = RESOLVE.format(ds=ds, rev=rev, path=path)
    # An auth-requiring repo fails fast with 401 rather than prompting; a token
    # in the environment is used if one is there (matches the mining scripts).
    headers = {"Range": f"bytes=0-{HEADER_READ_CAP - 1}"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(RETRIES):
        out = _fetch_once(url, headers)
        if not (isinstance(out, Unread) and out.retryable):
            return out
        if attempt < RETRIES - 1:
            time.sleep(BACKOFF * (2 ** attempt))
    return out

def explain(u, path, sha):
    """A rater-facing line for a header we could not read, keyed by disposition."""
    at = f"{path} at {sha[:10]}"
    if u.kind == "content_absent":
        return (f"no content for {at} on the Hub ({u.detail}); the object was never "
                f"uploaded for this revision, so a retry will not help")
    if u.kind == "access":
        return (f"cannot read {at} ({u.detail}); the repo looks private or gated, "
                f"set HF_TOKEN or HUGGINGFACE_HUB_TOKEN")
    if u.kind == "transport":
        return (f"could not fetch {at} ({u.detail}) after {RETRIES} tries; this is a "
                f"transport error, a later retry may succeed")
    if u.kind == "not_csv":
        return (f"{at} did not resolve to a readable CSV header; likely a non-CSV "
                f"payload (see #43), not a fetch failure")
    return f"could not read {at}"

def header(ds, repo, rev, path, download):
    r = run("git", "-C", repo, "show", f"{rev}:{path}")
    if r.returncode != 0:
        return ABSENT  # git could not read the blob at this revision
    if is_lfs_pointer(r.stdout):
        if download:
            return fetch_header(ds, rev, path)
        else:
            return NO_DOWNLOAD

    return parse_csv_header(r.stdout)

def inspect(ds, sha, download, show_rows):
    repo = clone(ds)
    print(f"# {ds} @ {sha[:10]}")
    print("message:", show(repo,"log","-1","--pretty=%s",sha).strip(), "\n")
    parent = show(repo,"rev-parse",f"{sha}^").strip()
    ns = show(repo,"show","--name-status","--find-renames","--pretty=format:",sha).strip()
    print("file changes:")
    if not ns:
        print("  (none)")
    for line in ns.splitlines():
        p = line.split("\t")
        status, path = p[0], p[-1]
        # a delete leaves no blob at sha; inspect the parent side instead
        rev = parent if (status.startswith("D") and parent) else sha
        print(line + ("   [stored in Git LFS]" if lfs_status(repo, rev, path) == "lfs" else ""))
    for line in ns.splitlines():
        p = line.split("\t")
        if not p[0].startswith("M") or not p[-1].lower().endswith(".csv"): continue
        path = p[-1]
        h = header(ds, repo, sha, path, download)
        if isinstance(h, Unread):
            # Each disposition gets its own line: a rater must be able to tell a
            # retryable transport error from bytes that are simply not there (#47).
            if h.kind == "absent":
                print(f"  [warn] git could not read {path} at {sha[:10]}; skipping",
                      file=sys.stderr)
            elif h.kind == "no_download":
                print(f"\n[{path}] LFS-tracked -> download a version to inspect the "
                      f"data change")
            else:
                print(f"  [warn] {explain(h, path, sha)}; cannot compare columns",
                      file=sys.stderr)
            continue

        pc = header(ds, repo, parent, path, download) if parent else None
        if parent and not isinstance(pc, list):
            # Any Unread on the parent: we cannot diff columns. Say so rather than
            # silently falling through to "no column change".
            print(f"  [warn] could not read parent columns for {path}; "
                  f"column diff skipped", file=sys.stderr)
        if isinstance(pc, list) and set(h) != set(pc):
            print(f"\n[{path}] column change:")
            print(f"  removed: {sorted(set(pc)-set(h))}")
            print(f"  added:   {sorted(set(h)-set(pc))}")
        elif isinstance(pc, list):
            # #48: the column sets match. Say so rather than printing nothing, which
            # reads the same as "not looked at". The blob may still differ in values
            # (a column recomputed in place), which a header-only diff cannot see.
            print(f"\n[{path}] no column-set change (values may still differ)")
            if p[0] == 'M' and show_rows:
                print()
                r = run("git", "-C", repo, "-c", "diff.lfs.textconv=cat", "diff",
                        "--color-words", "--textconv", parent, sha, "--", path)
                for line in r.stdout.splitlines()[:50]:
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
