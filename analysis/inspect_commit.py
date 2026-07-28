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
  - for MODIFIED tabular files, the column-header diff vs the parent commit.
    Tabular means CSV, TSV or TAB, each optionally gzipped
  - LFS-tracked files are flagged (download a version to inspect those)
  - with --show_rows, the head of the file at both revisions, for the case where
    the column set holds still and only the values move
  - a list of modified files left unanalyzed because their format is not one of
    the above, so that silence is never mistaken for "the data did not change"
  - when the dataset cannot be cloned at all, why: deleted or made private,
    restricted, renamed, or a failure on our side rather than the Hub's

Renames/adds/deletes are visible from Git history alone (no download). Only
in-file changes to LFS-stored files (often parquet) need an actual download, and
that download is a ranged HTTP read of the first few lines, not a full payload.
"""
import argparse, csv, difflib, os, subprocess, sys, time, zlib

import requests

CACHE = os.path.expanduser("~/.cache/hf-dataset-clones")
# Held repo-relative as well as resolved, because a message that sends a rater to
# this table should name it the way the repo does, not by an absolute path out of
# whichever checkout happened to print it.
CANDIDATES = "data/message_refactoring_candidates.csv"
CSV = os.path.join(os.path.dirname(__file__), "..", *CANDIDATES.split("/"))

def run(*a):
    return subprocess.run(a, capture_output=True, text=True, errors="replace")

def clone(ds):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, ds.replace("/", "__"))
    if os.path.isdir(os.path.join(path, ".git")):
        return path  # already cloned, so neither probe nor network is needed

    # Asked before cloning, so a dataset that is gone is named as such instead of
    # arriving as git's suppressed credential prompt, which reads as a broken
    # tool (#63). One small request ahead of a clone that moves megabytes.
    status = repo_status(ds)
    if status.kind in (GONE, RESTRICTED):
        sys.exit(clone_failure(ds, status))
    if status.kind == MOVED:
        # git follows the redirect on its own, so this is provenance rather than
        # an error: the ID recorded in the corpus and the repo actually read here
        # are no longer the same name.
        where = f" to {status.moved_to}" if status.moved_to else ""
        print(f"[note] {ds} was renamed on the Hub{where}; git follows the "
              f"redirect, so the corpus names one repo and this reads another",
              file=sys.stderr)

    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    if subprocess.run(["git","clone","--quiet",
                       f"https://huggingface.co/datasets/{ds}",path], env=env).returncode:
        sys.exit(clone_failure(ds, status))
    return path

def show(repo,*a): return run("git","-C",repo,*a).stdout

def show_blob(repo, rev, path):
    """Raw bytes of the blob at rev:path, or None if git holds none there.

    Bytes rather than text, unlike run(): a gzipped table is an ordinary case
    here (#55), and decoding one as text before it is decompressed destroys it.
    """
    r = subprocess.run(["git", "-C", repo, "show", f"{rev}:{path}"],
                       capture_output=True)
    return None if r.returncode else r.stdout

def is_lfs_pointer(text):
    """True if file content is a Git LFS pointer (actual data not present)."""
    return text[:25].startswith("version https://git-lfs")

def looks_like_pointer(blob):
    """is_lfs_pointer for raw bytes. Pointer text is ASCII, so a payload that
    fails to decode is by that fact not a pointer."""
    return is_lfs_pointer(blob[:64].decode("utf-8", errors="replace"))

# Formats read here, and the delimiter each one separates fields with. A format
# outside this table is not a fetch failure, it is a file inspect() reports as
# unanalyzed rather than passing over silently (#55).
DELIMITERS = {".csv": ",", ".tsv": "\t", ".tab": "\t"}
GZIP_SUFFIX = ".gz"
GZIP_MAGIC = b"\x1f\x8b"

def is_compressed(path):
    return path.lower().endswith(GZIP_SUFFIX)

def delimiter(path):
    """Field delimiter for a path this tool can read, or None if it cannot.

    A trailing .gz names the compression, not the format, so it comes off before
    the format suffix is read: train.csv.gz is a comma-separated file that
    happens to be gzipped, and gating on ".csv" alone skipped it outright (#55).
    """
    name = path.lower()
    if name.endswith(GZIP_SUFFIX):
        name = name[:-len(GZIP_SUFFIX)]
    for suffix, delim in DELIMITERS.items():
        if name.endswith(suffix):
            return delim
    return None

def gunzip_head(data, cap):
    """Decompressed head of `data` bounded by `cap`, or `data` if it is not gzip.

    The input is normally a truncated stream: we hold only the head of the
    compressed bytes and decompress as far as they reach. zlib stops at the end
    of what it was handed without complaint, so a short read is expected here
    rather than an error; only a corrupt stream raises, and that yields nothing.
    """
    if not data.startswith(GZIP_MAGIC):
        return data
    try:
        return zlib.decompressobj(zlib.MAX_WBITS | 16).decompress(data, cap)
    except zlib.error:
        return b""

def lfs_status(repo, rev, path):
    """Whether the blob at rev:path is stored in Git LFS.

    Returns "lfs" if the blob is a pointer, "plain" if it is the real content,
    or None if the blob is absent at that revision. Needs no download: under the
    GIT_LFS_SKIP_SMUDGE clone the stored blob IS the pointer text, so `git show`
    reveals LFS tracking without fetching the payload.
    """
    blob = show_blob(repo, rev, path)
    if blob is None:
        return None
    return "lfs" if looks_like_pointer(blob) else "plain"

def parse_header(text, delim=","):
    """Column names from the first line of delimited text, or None for a pointer.

    `delim` comes from the path's format (see delimiter): splitting a TSV on
    commas returns the whole line as one column, which then compares equal at
    both revisions and reads as "no column-set change" (#55).
    """
    if(is_lfs_pointer(text)):
        return None

    line = text.split("\n",1)[0]

    if not line.strip(): return []
    return next(csv.reader([line], delimiter=delim))

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
ROW_READ_CAP = 1 << 20     # bytes; same ceiling for the row sample, which stops
                           # at ROW_SAMPLE_LINES newlines long before this in
                           # any ordinary CSV
ROW_SAMPLE_LINES = 20      # lines read per revision for --show_rows
DIFF_MAX_LINES = 50        # lines of diff printed before we say what we cut

RESOLVE = "https://huggingface.co/datasets/{ds}/resolve/{rev}/{path}"
API = "https://huggingface.co/api/datasets/{ds}"
API_PATH = "/api/datasets/"  # the segment a rename redirect carries the new ID after
TIMEOUT = 30   # seconds, per HTTP request
RETRIES = 3    # attempts for a transient transport error
BACKOFF = 0.5  # seconds before the first retry, doubled after each failed attempt

def _auth_headers():
    """Bearer header from the environment, or {} when no token is set.

    Either variable name works, matching the mining scripts.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}

# Where a dataset repo stands on the Hub. Kept apart so a clone that does not
# happen can say which of these it was, rather than reporting every cause with
# one flat line (#64).
REACHABLE = "reachable"    # the Hub serves it; a failed clone is then ours to fix
MOVED = "moved"            # renamed; git follows the redirect, but the ID changed
GONE = "gone"              # deleted, private, or never there: no anonymous read
RESTRICTED = "restricted"  # present but access-controlled; a token may get in
UNKNOWN = "unknown"        # the probe itself did not land, so it decides nothing

class RepoStatus:
    """Outcome of a repo_status probe.

    `kind` is one of the tags above, which clone() and clone_failure() dispatch
    on; `moved_to` is the new dataset ID for a rename, else None; `detail` is a
    short specific such as an HTTP status.
    """
    __slots__ = ("kind", "detail", "moved_to")

    def __init__(self, kind, detail="", moved_to=None):
        self.kind = kind
        self.detail = detail
        self.moved_to = moved_to

    def __repr__(self):
        return (f"RepoStatus({self.kind!r}, detail={self.detail!r}, "
                f"moved_to={self.moved_to!r})")

def moved_target(location):
    """Dataset ID a rename redirect points at, or None if it names something else."""
    i = location.find(API_PATH)
    if i < 0:
        return None
    return location[i + len(API_PATH):].split("?")[0].strip("/") or None

def repo_status(ds):
    """Where `ds` stands on the Hub, established without cloning it.

    Reads the metadata endpoint, which settles the question in one small request.
    A 401 there is what the Hub answers an anonymous caller for a repo that was
    deleted, made private, or never created under this ID at all, and it cannot
    tell the three apart: whether a repo exists is itself withheld from a caller
    who could not read it anyway (#67). Gating is a different animal and does not
    land here at all: a gated dataset still answers 200 on this endpoint and
    refuses only at the content.

    Sending a token changes the code but not what it establishes, which is why
    401 and 404 share one disposition below rather than being split (#67).

    Redirects are not followed, because the redirect itself is the finding: it
    carries the name the dataset now goes by.

    A probe that does not land returns UNKNOWN and so decides nothing on its own;
    the clone is attempted regardless, and the Hub is never given the last word
    on whether our own git can do its job.
    """
    try:
        r = requests.head(API.format(ds=ds), headers=_auth_headers(),
                          timeout=TIMEOUT, allow_redirects=False)
    except requests.RequestException as e:
        return RepoStatus(UNKNOWN, type(e).__name__)
    code = r.status_code
    if code in (301, 302, 303, 307, 308):
        return RepoStatus(MOVED, f"HTTP {code}",
                          moved_to=moved_target(r.headers.get("Location", "")))
    if code in (401, 404):
        # Measured against the live endpoint (#67): the code tracks who is
        # asking, not what is wrong with the repo. Anonymously every unreadable
        # dataset answers 401, deleted and private and never-created alike, and
        # 404 means only that the path is no dataset route at all. Hand the Hub
        # a token and all of them answer 404, a repo known to have existed when
        # the corpus was mined included. Keying a disposition on which code came
        # back would therefore report our own auth state as a fact about the
        # dataset, so both land on GONE and the message names every cause.
        return RepoStatus(GONE, f"HTTP {code}")
    if code == 403:
        return RepoStatus(RESTRICTED, f"HTTP {code}")
    if 200 <= code < 300:
        return RepoStatus(REACHABLE, f"HTTP {code}")
    return RepoStatus(UNKNOWN, f"HTTP {code}")

def clone_failure(ds, status):
    """Why the clone did not happen, in terms a rater can act on (#64).

    Every cause used to arrive as the same `clone failed` line, under git's
    suppressed credential prompt, which reads as a broken tool whichever of these
    actually went wrong.
    """
    if status.kind == GONE:
        return (f"cannot read {ds} on the Hub ({status.detail}): the dataset was "
                f"deleted or made private since the corpus was mined, or else no "
                f"dataset ever went by this ID. Nothing here tells the three "
                f"apart, so check the DatasetID against {CANDIDATES} before "
                f"skipping the candidate: a corrupted key names a repo that was "
                f"never there, while the dataset it should have named is still "
                f"live. A token gets in only if you hold access to this repo; "
                f"gating would look different.")
    if status.kind == RESTRICTED:
        return (f"access to {ds} is restricted ({status.detail}); set HF_TOKEN or "
                f"HUGGINGFACE_HUB_TOKEN, or accept the dataset's terms on the Hub")
    if status.kind == MOVED:
        where = f" to {status.moved_to}" if status.moved_to else ""
        return (f"clone failed for {ds}, which the Hub reports was renamed{where} "
                f"({status.detail}); retry under the new name")
    if status.kind == REACHABLE:
        return (f"clone failed for {ds}, though the Hub serves the dataset "
                f"({status.detail}); this is a transport or local git failure, "
                f"not a dataset that went away")
    return (f"clone failed for {ds}, and the Hub could not be asked why "
            f"({status.detail}); check the network, then retry")

def _fetch_once(url, headers, max_lines, cap):
    """One attempt at the first `max_lines` lines: a list of them, or an Unread."""
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
            # Stop at the max_lines-th newline: a header (max_lines=1) or a short
            # row sample returns after one chunk, and a newline-free binary
            # payload (e.g. parquet) cannot stream unbounded into memory.
            buf, whole, gz, first = b"", True, None, True
            for chunk in r.iter_content(8192):
                if first:
                    # Sniffed once, at the very start: a gzip member begins with
                    # the magic, and a plain payload whose interior happens to
                    # carry those two bytes is not compressed.
                    first = False
                    if chunk.startswith(GZIP_MAGIC):
                        gz = zlib.decompressobj(zlib.MAX_WBITS | 16)
                if gz is not None:
                    # Decompressed as it streams, so the newline count below sees
                    # rows rather than compressed bytes, and bounded by the same
                    # cap, so a small payload cannot expand without limit.
                    try:
                        chunk = gz.decompress(chunk, max(1, cap - len(buf)))
                    except zlib.error:
                        return Unread("not_tabular", detail="corrupt gzip stream")
                buf += chunk
                if buf.count(b"\n") >= max_lines or len(buf) >= cap:
                    whole = False
                    break
    except requests.RequestException as e:
        return Unread("transport", retryable=True, detail=type(e).__name__)

    if not whole and b"\n" not in buf:
        # A read cut short with no line break anywhere in it holds part of a
        # first line and no way to tell how much is missing. Returning it would
        # be the #48 trap in its most convincing form: a truncated header parses
        # cleanly and compares equal to the other revision's truncated header.
        return Unread("not_tabular",
                      detail=f"no line break in the first {cap:,} bytes")

    lines = buf[:cap].decode("utf-8", errors="replace").split("\n")
    # A read we cut short can end mid-line, and half a row is not a row. Keep it
    # when it is all we have: a payload with no newline inside the cap is a
    # not_csv finding for the caller, not an empty response.
    if not whole and len(lines) > 1 and lines[-1]:
        lines.pop()
    lines = lines[:max_lines]

    # Nothing came back, or the bytes are the LFS pointer itself: the object was
    # never materialized for this revision (a Xet-migrated repo can serve the
    # pointer text as the file's content, #47). Absent bytes will not appear on a
    # retry, so this is terminal, not transport.
    if not lines or not lines[0]:
        return Unread("content_absent", detail="empty response")
    if is_lfs_pointer(lines[0]):
        return Unread("content_absent", detail="Hub served an LFS pointer")
    return lines

def _fetch(ds, rev, path, max_lines, cap):
    """First `max_lines` lines of a file at a revision on the Hub, or an Unread.

    Reads over HTTP rather than through `git lfs smudge`. Repos migrated to Xet
    storage serve no classic LFS object, so smudge cannot fetch them; it exits 0
    and echoes the pointer straight back, which read as "LFS is broken for this
    repo" (#44). The resolve endpoint serves both storage backends, redirecting
    to whichever CDN holds the content.

    A transient transport error (5xx, throttling, a dropped connection) is retried
    with backoff before giving up; a terminal outcome (absent bytes, a gated repo)
    is returned at once (#47). Only the head of the file is wanted, so this is a
    Range request, capped again client-side in case the server ignores the range.
    """
    url = RESOLVE.format(ds=ds, rev=rev, path=path)
    # An auth-requiring repo fails fast with 401 rather than prompting; a token
    # in the environment is used if one is there.
    headers = {"Range": f"bytes=0-{cap - 1}", **_auth_headers()}
    for attempt in range(RETRIES):
        out = _fetch_once(url, headers, max_lines, cap)
        if not (isinstance(out, Unread) and out.retryable):
            return out
        if attempt < RETRIES - 1:
            time.sleep(BACKOFF * (2 ** attempt))
    return out

def fetch_header(ds, rev, path):
    """Column names from a file's first line on the Hub, or an Unread."""
    out = _fetch(ds, rev, path, 1, HEADER_READ_CAP)
    if isinstance(out, Unread):
        return out
    # A resolved blob need not be parseable as a table: a binary payload (e.g.
    # parquet) yields a huge single "field" that trips csv's field-size limit.
    # That is a format we do not read here (see #43), not a fetch failure.
    try:
        return parse_header(out[0], delimiter(path) or ",")
    except csv.Error:
        return Unread("not_tabular", detail="first line did not parse")

def fetch_rows(ds, rev, path):
    """Head of a file on the Hub as a list of lines, or an Unread."""
    return _fetch(ds, rev, path, ROW_SAMPLE_LINES, ROW_READ_CAP)

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
    if u.kind == "not_tabular":
        why = f" ({u.detail})" if u.detail else ""
        return (f"{at} did not resolve to readable delimited text{why}; likely a "
                f"format this tool does not read (see #43), not a fetch failure")
    return f"could not read {at}"

def header(ds, repo, rev, path, download):
    blob = show_blob(repo, rev, path)
    if blob is None:
        return ABSENT  # git could not read the blob at this revision
    if looks_like_pointer(blob):
        if download:
            return fetch_header(ds, rev, path)
        else:
            return NO_DOWNLOAD

    # git holds the content itself, so no download is needed even when it is
    # compressed: gunzip_head is a no-op on anything that is not gzip.
    text = gunzip_head(blob, HEADER_READ_CAP).decode("utf-8", errors="replace")
    try:
        return parse_header(text, delimiter(path) or ",")
    except csv.Error:
        return Unread("not_tabular", detail="first line did not parse")

def pointer_size(repo, rev, path):
    """Payload size recorded in the LFS pointer at rev:path, or None.

    The pointer is in git, so a size change is evidence the payload changed that
    costs not one byte of download. Equal sizes prove nothing either way.
    """
    blob = show_blob(repo, rev, path)
    if blob is None or not looks_like_pointer(blob):
        return None
    for line in blob.decode("utf-8", errors="replace").splitlines():
        if line.startswith("size "):
            try:
                return int(line.split(None, 1)[1])
            except ValueError:
                return None
    return None

def print_capped(lines, indent="  "):
    """Print at most DIFF_MAX_LINES lines, and say so when there were more.

    Silent truncation reads as a short diff, which is a different claim.
    """
    for line in lines[:DIFF_MAX_LINES]:
        print(indent + line)
    if len(lines) > DIFF_MAX_LINES:
        print(f"{indent}... {len(lines) - DIFF_MAX_LINES} further lines not shown "
              f"(cap is {DIFF_MAX_LINES})")

def local_diff(repo, parent, sha, path):
    """Row diff for a file git actually holds. No network, no color, no textconv.

    Passing --no-textconv matters: git enables textconv drivers by default for
    `git diff`, and .gitattributes on a Hub dataset sets diff=lfs, so a driver
    could reach for the payload behind our back.
    """
    r = run("git", "-C", repo, "diff", "--no-color", "--no-textconv",
            parent, sha, "--", path)
    lines = r.stdout.splitlines()
    if not lines:
        print("  no textual difference in the file")
        return
    print_capped(lines)

def head_lines(ds, repo, rev, path):
    """First ROW_SAMPLE_LINES lines of rev:path, or an Unread.

    Reads the local blob where git holds the content and the Hub where git holds
    only an LFS pointer, decompressing a gzipped payload either way.
    """
    blob = show_blob(repo, rev, path)
    if blob is None:
        return ABSENT
    if looks_like_pointer(blob):
        return fetch_rows(ds, rev, path)
    text = gunzip_head(blob, ROW_READ_CAP).decode("utf-8", errors="replace")
    return text.split("\n")[:ROW_SAMPLE_LINES]

def row_sample(ds, repo, parent, sha, path):
    """Show the head of `path` at both revisions, for a values-only change.

    LFS-tracked blobs are read over the resolve endpoint, the same path
    fetch_header takes. The local clone cannot answer this: it is cloned under
    GIT_LFS_SKIP_SMUDGE, so `git diff` there prints the pointer's oid and size
    (#52), or, where smudge can still reach a classic LFS object, quietly pulls
    the whole payload, which is the download this tool exists to avoid.

    A gzipped file goes the same way even when git holds it in full, because
    `git diff` on compressed bytes reports only that they differ.

    What comes back is the head of the file, not a diff of it. A value change
    below the sample does not appear here, so an identical sample is reported as
    an identical sample rather than as "no change" (the #48 trap, one level down).

    Only reached once both headers read, which for an LFS-tracked file means
    that --download was passed: without it header() returns NO_DOWNLOAD and
    inspect() has already said so.
    """
    lfs = any(lfs_status(repo, rev, path) == "lfs" for rev in (parent, sha))
    if not lfs and not is_compressed(path):
        local_diff(repo, parent, sha, path)
        return

    if lfs:
        old_size, new_size = pointer_size(repo, parent, path), pointer_size(repo, sha, path)
        if old_size is not None and new_size is not None:
            if old_size != new_size:
                print(f"  payload size {old_size:,} -> {new_size:,} bytes "
                      f"({new_size - old_size:+,}): the data did change")
            else:
                print(f"  payload size unchanged at {old_size:,} bytes "
                      f"(equal size is not equal content)")

    old, new = head_lines(ds, repo, parent, path), head_lines(ds, repo, sha, path)
    for rev, rows in ((parent, old), (sha, new)):
        if isinstance(rows, Unread):
            print(f"  [warn] {explain(rows, path, rev)}; rows not shown",
                  file=sys.stderr)
    if isinstance(old, Unread) or isinstance(new, Unread):
        return

    diff = list(difflib.unified_diff(old, new, fromfile=f"{path}@{parent[:10]}",
                                     tofile=f"{path}@{sha[:10]}", lineterm=""))
    if not diff:
        print(f"  first {len(old)} lines identical at both revisions; whatever "
              f"changed is further into the file than this sample reaches")
        return
    print(f"  first {ROW_SAMPLE_LINES} lines at each revision (head sample, not a "
          f"diff of the whole file):")
    print_capped(diff)

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
    skipped = []
    for line in ns.splitlines():
        p = line.split("\t")
        if not p[0].startswith("M"): continue
        path = p[-1]
        if delimiter(path) is None:
            # Not a format we can read. Collected rather than dropped: a modified
            # file the tool never opened must not leave the same impression as one
            # it opened and found unchanged (#55).
            skipped.append(path)
            continue
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
        # Both headers read, so both revisions resolve and a row sample is worth
        # asking for. This runs for a changed column set too: a rater who passed
        # --show_rows should not get silence back.
        if show_rows and isinstance(pc, list):
            row_sample(ds, repo, parent, sha, path)

    if skipped:
        # Every modified file the tool declined to open is named. A README and a
        # parquet land here alike: the point is not to classify them but to keep
        # "not looked at" from reading as "looked at and found unchanged".
        print("\nnot analyzed:")
        for path in skipped:
            print(f"  {path}   [format this tool does not read, see #43]")
        print("  Nothing above says whether the contents of these changed.")

def list_candidates(typ=None):
    with open(CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if typ and row["tentative_type"] != typ: continue
            print(f"{row['tentative_type']:32} {row['DatasetID']:42} {row['CommitId'][:10]}  {row['log_message'][:55]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?"); ap.add_argument("commit", nargs="?")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--type")
    ap.add_argument("--download", action="store_true",
                    help="read LFS-tracked content over the Hub's resolve endpoint")
    ap.add_argument("--show_rows", action="store_true",
                    help=f"show the first {ROW_SAMPLE_LINES} lines of each modified "
                         f"table at both revisions; needs --download for LFS-tracked "
                         f"files")
    a = ap.parse_args()
    if a.list: list_candidates(a.type)
    elif a.dataset and a.commit: inspect(a.dataset, a.commit,a.download,a.show_rows)
    else: ap.print_help()
