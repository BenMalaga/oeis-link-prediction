"""Idempotent data fetch + ingest for the OEIS link-prediction project (P3).

All sources verified live 2026-06-10 (see data/README.md). Everything here is free
and anonymous: no API keys, no signups, $0.

Usage (from the project root):

    python -m src.fetch_data bulk                 # stripped.gz + names.gz (~39 MB total)
    python -m src.fetch_data oeisdata --confirm   # ~610 MB shallow clone (main-phase only)
    python -m src.fetch_data bfiles A000045 A000108 ...   # per-candidate b-files, 10 s spacing
    python -m src.fetch_data spotcheck A000045    # single-sequence JSON (xref field) for QA
    python -m src.fetch_data parse                # parse local dumps -> data/parsed/

Design rules baked in (do not relax):
  * Idempotent: every fetch skips if the target already exists (use --force to refetch).
  * Polite: oeis.org gets a browser UA and >= 10 s spacing per robots.txt Crawl-Delay.
    robots.txt also Disallows /search, the JSON API is for spot checks ONLY, never bulk.
  * Bulk xref ground truth comes from the oeisdata git repo (%Y lines), NOT the API.
  * b-files are fetched individually over plain HTTP (no Git-LFS), per-candidate only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PARSED_DIR = DATA_DIR / "parsed"
BFILE_DIR = DATA_DIR / "bfiles"
OEISDATA_DIR = DATA_DIR / "oeisdata"
SNAPSHOT_FILE = DATA_DIR / "SNAPSHOT.txt"

# ------------------------------------------------------------------- verified facts
# Verified 2026-06-10 by direct request (see data/README.md). Sizes drift daily as the
# OEIS grows, so they are used as sanity floors, not exact-match checks.

USER_AGENT = (
    "Mozilla/5.0 (oeis-link-prediction research; contact: benmalaga03@gmail.com)"
)
OEIS_CRAWL_DELAY_S = 10  # robots.txt Crawl-Delay: 10, non-negotiable

BULK_FILES = {
    # name -> (url, min_expected_bytes as sanity floor on 2026-06-10 sizes)
    "stripped.gz": ("https://oeis.org/stripped.gz", 30_000_000),  # 31,655,637 B on 2026-06-10
    "names.gz": ("https://oeis.org/names.gz", 7_000_000),  # 7,499,817 B on 2026-06-10
}

OEISDATA_REPO = "https://github.com/oeis/oeisdata"  # ~610 MB git-side WITHOUT LFS objects
BFILE_URL = "https://oeis.org/A{num:06d}/b{num:06d}.txt"  # verified 200, plain text, no LFS
SEQ_RAW_URL = "https://raw.githubusercontent.com/oeis/oeisdata/main/seq/A{prefix:03d}/A{num:06d}.seq"
JSON_URL = "https://oeis.org/A{num:06d}?fmt=json"  # single object; includes "xref" field

A_NUMBER_RE = re.compile(r"^A(\d{6,7})$")

_last_oeis_request_ts: float = 0.0


# ------------------------------------------------------------------------- helpers

def log(msg: str) -> None:
    print(f"[fetch_data] {msg}", flush=True)


def parse_a_number(a: str) -> int:
    m = A_NUMBER_RE.match(a.strip())
    if not m:
        raise ValueError(f"Not an A-number: {a!r} (expected e.g. A000045)")
    return int(m.group(1))


def _polite_oeis_get(url: str) -> bytes:
    """GET from oeis.org honoring Crawl-Delay: 10 across ALL calls in this process."""
    global _last_oeis_request_ts
    wait = OEIS_CRAWL_DELAY_S - (time.monotonic() - _last_oeis_request_ts)
    if wait > 0:
        log(f"crawl-delay: sleeping {wait:.1f}s before {url}")
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    _last_oeis_request_ts = time.monotonic()
    return body


def _download(url: str, dest: Path, min_bytes: int = 0, force: bool = False) -> bool:
    """Download url -> dest atomically. Returns True if a download happened."""
    if dest.exists() and not force:
        log(f"skip (exists): {dest.relative_to(PROJECT_ROOT)}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"GET {url}")
    body = _polite_oeis_get(url) if "oeis.org" in url else _plain_get(url)
    if len(body) < min_bytes:
        raise RuntimeError(
            f"{url} returned {len(body)} B < sanity floor {min_bytes} B, "
            "truncated download or upstream change; not writing."
        )
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(body)
    tmp.rename(dest)
    log(f"wrote {dest.relative_to(PROJECT_ROOT)} ({len(body):,} B)")
    return True


def _plain_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _record_snapshot(line: str) -> None:
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with SNAPSHOT_FILE.open("a") as fh:
        fh.write(f"{stamp}  {line}\n")


# --------------------------------------------------------------------- bulk fetch

def fetch_bulk(force: bool = False) -> None:
    """stripped.gz + names.gz (~39 MB total). Regenerated daily ~03:00-05:00 UTC;
    record the fetch date, results must be pinned to a snapshot date."""
    for name, (url, floor) in BULK_FILES.items():
        if _download(url, DATA_DIR / name, min_bytes=floor, force=force):
            _record_snapshot(f"fetched {name} from {url}")


# ------------------------------------------------------------------ oeisdata clone

def fetch_oeisdata(confirm: bool = False) -> None:
    """Shallow-clone github.com/oeis/oeisdata for the %Y xref ground truth.

    Cloned with --no-checkout: the pack alone is ~341 MB (measured 2026-06-10),
    while a full checkout would materialize ~400k tiny .seq files plus ~600k
    files/** LFS pointer stubs, at APFS's 4 KB block size that is >1.5 GB of
    allocated disk for ~600 MB of bytes. All parsing streams blobs straight from
    the pack via `git archive` / `git show` (see src/build_graph.py), so no
    working tree is ever needed. b-files stay out via GIT_LFS_SKIP_SMUDGE=1
    (.gitattributes pins LFS to files/** only). --confirm gates the download.
    """
    if OEISDATA_DIR.exists():
        log(f"skip (exists): {OEISDATA_DIR.relative_to(PROJECT_ROOT)}")
        return
    if not confirm:
        log(
            "REFUSING to clone oeisdata (~341 MB pack) without --confirm. "
            "This is a main-phase download; see data/README.md."
        )
        return
    import os

    cmd = [
        "git", "clone", "--depth", "1", "--no-checkout",
        OEISDATA_REPO, str(OEISDATA_DIR),
    ]
    log(" ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    record_oeisdata_snapshot()


def record_oeisdata_snapshot() -> str:
    """Record the oeisdata HEAD SHA + upstream sync time in data/SNAPSHOT.txt
    (idempotent: skips if this SHA is already recorded). Returns the SHA."""
    sha = subprocess.run(
        ["git", "-C", str(OEISDATA_DIR), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    sync_time = subprocess.run(
        ["git", "-C", str(OEISDATA_DIR), "show", "HEAD:time.txt"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if SNAPSHOT_FILE.exists() and sha in SNAPSHOT_FILE.read_text():
        return sha
    _record_snapshot(f"oeisdata commit {sha} (upstream sync time.txt: {sync_time})")
    log(f"oeisdata commit {sha} recorded in {SNAPSHOT_FILE.name}")
    return sha


def fetch_seq_file(a_number: str, force: bool = False) -> Path:
    """Fetch a single .seq file from raw.githubusercontent (free, no clone needed).
    Useful for spot checks before the full oeisdata clone exists."""
    num = parse_a_number(a_number)
    dest = DATA_DIR / "seq_samples" / f"A{num:06d}.seq"
    url = SEQ_RAW_URL.format(prefix=num // 1000, num=num)
    _download(url, dest, min_bytes=10, force=force)
    return dest


# ----------------------------------------------------------------------- b-files

def fetch_bfiles(a_numbers: list[str], force: bool = False) -> None:
    """Per-candidate b-files over plain HTTP (no LFS), one request per 10 s.

    Fine for the planned >=25-candidate verification set (~minutes); NOT a bulk path.
    Some sequences have no b-file (404), recorded and skipped, not fatal.
    """
    BFILE_DIR.mkdir(parents=True, exist_ok=True)
    for a in a_numbers:
        num = parse_a_number(a)
        dest = BFILE_DIR / f"b{num:06d}.txt"
        url = BFILE_URL.format(num=num)
        try:
            _download(url, dest, min_bytes=2, force=force)
        except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
            if e.code == 404:
                log(f"no b-file for A{num:06d} (404), verification must fall back "
                    "to stripped.gz terms and be labeled accordingly")
            else:
                raise


# ------------------------------------------------------------------ API spot check

def spotcheck(a_number: str) -> dict:
    """Fetch ONE sequence as JSON (https://oeis.org/A{n}?fmt=json) for QA of the
    parsed %Y graph. The response is a single object including an 'xref' field.

    NOTE (verified 2026-06-10): the /search endpoint returns a BARE JSON ARRAY now
    (10/page, &start=N, no count field), old {results,count} parsing code breaks.
    robots.txt Disallows /search; use this per-sequence form, sparingly.
    """
    num = parse_a_number(a_number)
    body = _polite_oeis_get(JSON_URL.format(num=num))
    obj = json.loads(body)
    log(f"A{num:06d}: name={obj.get('name', '')[:60]!r} "
        f"xref_lines={len(obj.get('xref', []) or [])}")
    return obj


# ------------------------------------------------------------------------ parsing

@dataclass
class Sequence:
    a_number: int
    terms: list[int]
    name: str = ""
    xrefs: list[int] = field(default_factory=list)  # parsed %Y targets (raw, unstratified)
    dead: bool = False  # in names.gz but not stripped.gz (dead/recycled entry)


def parse_stripped(path: Path | None = None) -> dict[int, list[int]]:
    """stripped.gz -> {a_number: [terms]}. 396,449 rows on the 2026-06-10 snapshot."""
    path = path or DATA_DIR / "stripped.gz"
    out: dict[int, list[int]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("A"):
                continue  # header/comment lines
            a_str, _, terms_str = line.partition(" ")
            num = int(a_str[1:])
            # terms field looks like ",1,1,2,3,5,8,", strip commas at both ends
            terms = [int(t) for t in terms_str.strip().strip(",").split(",") if t]
            out[num] = terms
    log(f"parsed stripped: {len(out):,} sequences")
    return out


def parse_names(path: Path | None = None) -> dict[int, str]:
    """names.gz -> {a_number: name}. 396,756 rows on the 2026-06-10 snapshot, 307 MORE
    than stripped.gz. The surplus = dead/recycled entries (e.g. 'Duplicate of A...'):
    these feed the duplicate-merge stratum and must NOT be silently dropped."""
    path = path or DATA_DIR / "names.gz"
    out: dict[int, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("A"):
                continue
            a_str, _, name = line.partition(" ")
            out[int(a_str[1:])] = name.strip()
    log(f"parsed names: {len(out):,} entries")
    return out


_XREF_TARGET_RE = re.compile(r"A(\d{6,7})")


def parse_xrefs_from_seq_dir(seq_dir: Path | None = None) -> dict[int, list[int]]:
    """Extract %Y cross-reference targets from the oeisdata clone.

    Returns {source_a_number: [target_a_numbers]}, the RAW directed graph
    (deduped per source). The clone is --no-checkout, so blobs are streamed from
    the pack via src.build_graph.iter_seq_files (no working tree required); a
    checked-out seq/ dir is used as a fallback if one exists.

    Stratification (duplicate/transform/see-also/contextual) happens downstream in
    the analysis code using names.gz heuristics + %Y line text, per the pre-registration
    """
    seq_dir = seq_dir or OEISDATA_DIR / "seq"
    graph: dict[int, list[int]] = {}
    n_files = 0
    if seq_dir.exists():  # legacy checked-out clone
        for seq_file in sorted(seq_dir.glob("A*/A*.seq")):
            n_files += 1
            src = int(seq_file.stem[1:])
            targets: list[int] = []
            with seq_file.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("%Y"):
                        targets.extend(
                            int(m) for m in _XREF_TARGET_RE.findall(line) if int(m) != src
                        )
            if targets:
                graph[src] = sorted(set(targets))
    elif OEISDATA_DIR.exists():  # --no-checkout clone: stream from the pack
        from .build_graph import iter_seq_files

        for src, text in iter_seq_files():
            n_files += 1
            targets = [
                int(m)
                for line in text.splitlines()
                if line.startswith("%Y")
                for m in _XREF_TARGET_RE.findall(line)
                if int(m) != src
            ]
            if targets:
                graph[src] = sorted(set(targets))
    else:
        raise FileNotFoundError(
            f"{OEISDATA_DIR} missing, run `python -m src.fetch_data oeisdata --confirm` "
            "first (~341 MB pack, main-phase download)."
        )
    log(f"parsed %Y graph: {n_files:,} .seq files, {len(graph):,} sources with xrefs, "
        f"{sum(len(v) for v in graph.values()):,} directed edges")
    return graph


def parse_all(out_dir: Path | None = None) -> None:
    """Join stripped + names (+ xrefs if oeisdata is present) and write parsed JSONL.

    Join is on A-number; entries present only in names.gz are flagged dead=True.
    """
    out_dir = out_dir or PARSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    terms = parse_stripped()
    names = parse_names()
    try:
        xrefs = parse_xrefs_from_seq_dir()
    except FileNotFoundError as e:
        log(f"WARNING: {e}")
        log("writing corpus WITHOUT xrefs, fine for feature work, not for the benchmark")
        xrefs = {}

    only_names = sorted(set(names) - set(terms))
    log(f"join: {len(terms):,} with terms, {len(only_names):,} name-only (dead/recycled)")

    out_path = out_dir / "corpus.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for num in sorted(set(terms) | set(names)):
            rec = {
                "a_number": num,
                "terms": terms.get(num, []),
                "name": names.get(num, ""),
                "xrefs": xrefs.get(num, []),
                "dead": num not in terms,
            }
            fh.write(json.dumps(rec) + "\n")
    log(f"wrote {out_path.relative_to(PROJECT_ROOT)}")

    # TODO(main-phase): emit the stratified benchmark splits here once the
    # stratification heuristics are locked in PRE_REGISTRATION.md:
    #   - collapse symmetric/reciprocal xref pairs BEFORE the 10% edge holdout
    #   - negative sampling must exclude adjacent A-numbers (same author/family leak)
    #   - strata: duplicate-of / transform-of / see-also / contextual-constant
    # Nothing here may run against held-out edges before pre-registration is locked.


# ---------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fetch_data", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bulk", help="stripped.gz + names.gz (~39 MB)")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("oeisdata", help="shallow-clone oeisdata (~610 MB; gated)")
    p.add_argument("--confirm", action="store_true",
                   help="acknowledge the ~610 MB main-phase download")

    p = sub.add_parser("seq", help="fetch single .seq file(s) from raw.githubusercontent")
    p.add_argument("a_numbers", nargs="+")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("bfiles", help="fetch b-files per candidate (10 s spacing)")
    p.add_argument("a_numbers", nargs="+")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("spotcheck", help="single-sequence JSON for xref QA (rate-limited)")
    p.add_argument("a_number")

    sub.add_parser("parse", help="parse local dumps -> data/parsed/corpus.jsonl")

    args = ap.parse_args(argv)
    if args.cmd == "bulk":
        fetch_bulk(force=args.force)
    elif args.cmd == "oeisdata":
        fetch_oeisdata(confirm=args.confirm)
    elif args.cmd == "seq":
        for a in args.a_numbers:
            fetch_seq_file(a, force=args.force)
    elif args.cmd == "bfiles":
        fetch_bfiles(args.a_numbers, force=args.force)
    elif args.cmd == "spotcheck":
        spotcheck(args.a_number)
    elif args.cmd == "parse":
        parse_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
