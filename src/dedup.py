"""Dedup exclusion sets vs Sequence Machine and LODA (PRE_REGISTRATION.md 5.7).

Every H2 candidate pair must be checked against the two public machine-discovery
projects BEFORE being claimed or submitted; already-known relations are dropped
from the H2 count and logged. This module builds the exclusion data:

  python -m src.dedup loda            # full LODA relation list (transient clone)
  python -m src.dedup loda --keep-clone
  python -m src.dedup sm A000045 ...  # per-candidate Sequence Machine programs
  python -m src.dedup report          # sizes of the built exclusion sets

LODA (https://github.com/loda-lang/loda-programs, Apache-2.0 programs):
  one .asm per mined sequence under oeis/<prefix>/A######.asm; cross-sequence
  use is the `seq` opcode ("seq $2,45" = call A000045). We shallow-clone with
  --no-checkout (pack only), stream blobs via `git archive`, extract
  (program, called) pairs, then DELETE the clone (disk budget), the derived
  parquet + A-number list are kept in data/dedup/.

Sequence Machine (https://sequencedb.net; data mirror
https://github.com/jonmaiga/sequence-machine-data, CC-BY-SA-4.0):
  per-sequence <A>.programs.json under oeis/A###/ with conjectured formulas.
  The full repo is ~690 MB (too large for this project's disk budget), so the
  exclusion check is PER-CANDIDATE: fetch the .programs.json for each
  candidate's endpoints from raw.githubusercontent.com at claim time and scan
  for mentions of the other endpoint. That is exact for our <=~100-pair H2
  candidate set and verified live 2026-06-10.

No model evaluation happens here; this is discovery-phase support tooling.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

from .fetch_data import DATA_DIR, PROJECT_ROOT, log as _log, parse_a_number

DEDUP_DIR = DATA_DIR / "dedup"
LODA_CLONE = DEDUP_DIR / "loda-programs"
LODA_REPO = "https://github.com/loda-lang/loda-programs"
LODA_RELATIONS = DEDUP_DIR / "loda_relations.parquet"
LODA_ANUMBERS = DEDUP_DIR / "loda_anumbers.txt"
SM_DIR = DEDUP_DIR / "sequence-machine"
SM_RAW_URL = (
    "https://raw.githubusercontent.com/jonmaiga/sequence-machine-data/master/"
    "oeis/A{prefix:03d}/A{num:06d}.programs.json"
)
SIZES_JSON = PROJECT_ROOT / "results" / "dedup_sizes.json"

SEQ_OP_RE = re.compile(r"^\s*seq\s+\$\d+\s*,\s*(\d+)", re.M)
A_REF_RE = re.compile(r"A(\d{6,7})")


# ------------------------------------------------------------------- LODA


def build_loda(keep_clone: bool = False, force: bool = False) -> dict:
    import pandas as pd

    if LODA_RELATIONS.exists() and not force:
        _log("dedup: LODA relations exist; --force to rebuild")
        return loda_sizes()
    DEDUP_DIR.mkdir(parents=True, exist_ok=True)
    if not LODA_CLONE.exists():
        cmd = ["git", "clone", "--depth", "1", "--no-checkout",
               LODA_REPO, str(LODA_CLONE)]
        _log("dedup: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
    sha = subprocess.run(
        ["git", "-C", str(LODA_CLONE), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()

    t0 = time.monotonic()
    progs: list[int] = []
    rel_src: list[int] = []
    rel_dst: list[int] = []
    proc = subprocess.Popen(
        ["git", "-C", str(LODA_CLONE), "archive", "--format=tar", "HEAD", "oeis"],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    n = 0
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".asm"):
                    continue
                stem = member.name.rsplit("/", 1)[-1][:-4]  # "A000045"
                try:
                    a = int(stem[1:])
                except ValueError:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                text = fh.read().decode("utf-8", errors="replace")
                progs.append(a)
                for m in SEQ_OP_RE.findall(text):
                    rel_src.append(a)
                    rel_dst.append(int(m))
                n += 1
                if n % 50_000 == 0:
                    _log(f"dedup:   ... {n:,} programs in {time.monotonic()-t0:.0f}s")
    finally:
        proc.stdout.close()
        if proc.wait() != 0:
            raise RuntimeError(f"git archive exited {proc.returncode}")

    rel = pd.DataFrame({"program": pd.array(rel_src, dtype="int32"),
                        "calls": pd.array(rel_dst, dtype="int32")})
    rel.to_parquet(LODA_RELATIONS, engine="pyarrow", index=False)
    LODA_ANUMBERS.write_text("\n".join(f"A{a:06d}" for a in sorted(progs)) + "\n")
    (DEDUP_DIR / "LODA_SNAPSHOT.txt").write_text(
        f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}  loda-programs commit {sha}\n")
    _log(f"dedup: LODA: {len(progs):,} programs, {len(rel):,} seq-call relations "
         f"({time.monotonic()-t0:.0f}s); commit {sha[:12]}")
    if not keep_clone:
        shutil.rmtree(LODA_CLONE)
        _log("dedup: deleted transient LODA clone (disk budget)")
    return loda_sizes()


def loda_sizes() -> dict:
    import pandas as pd

    out: dict = {}
    if LODA_RELATIONS.exists():
        rel = pd.read_parquet(LODA_RELATIONS)
        pairs = {(min(a, b), max(a, b)) for a, b in zip(rel["program"], rel["calls"])}
        out["loda_programs"] = sum(1 for _ in LODA_ANUMBERS.open())
        out["loda_seq_call_relations_directed"] = int(len(rel))
        out["loda_seq_call_pairs_undirected"] = len(pairs)
    return out


def loda_exclusion_pairs() -> set[tuple[int, int]]:
    """Undirected (u,v) pairs already related by a LODA seq-call."""
    import pandas as pd

    rel = pd.read_parquet(LODA_RELATIONS)
    return {(min(a, b), max(a, b)) for a, b in zip(rel["program"], rel["calls"])}


# -------------------------------------------------------- Sequence Machine


def fetch_sm_programs(a_number: str, force: bool = False):
    """Fetch one sequence's Sequence Machine programs.json (None on 404 = SM
    has no conjectured programs for it)."""
    num = parse_a_number(a_number)
    SM_DIR.mkdir(parents=True, exist_ok=True)
    dest = SM_DIR / f"A{num:06d}.programs.json"
    if dest.exists() and not force:
        return json.loads(dest.read_text())
    url = SM_RAW_URL.format(prefix=num // 1000, num=num)
    req = urllib.request.Request(url, headers={"User-Agent": "oeis-link-prediction"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        if e.code == 404:
            dest.with_suffix(".missing").write_text("404\n")
            return None
        raise
    dest.write_bytes(body)
    time.sleep(1)  # gentle on raw.githubusercontent
    return json.loads(body)


def sm_mentions(a_number: str) -> set[int]:
    """A-numbers mentioned anywhere in this sequence's SM programs (the
    candidate-time exclusion test: pair (u,v) is KNOWN if v in sm_mentions(u)
    or u in sm_mentions(v) or (u,v) in loda_exclusion_pairs())."""
    obj = fetch_sm_programs(a_number)
    if obj is None:
        return set()
    return {int(m) for m in A_REF_RE.findall(json.dumps(obj))}


# ------------------------------------------------------------------ report


def report() -> dict:
    sizes = loda_sizes()
    sm_files = list(SM_DIR.glob("*.programs.json")) if SM_DIR.exists() else []
    sizes["sm_strategy"] = "per-candidate fetch (full repo ~690 MB > disk budget)"
    sizes["sm_programs_cached"] = len(sm_files)
    SIZES_JSON.parent.mkdir(parents=True, exist_ok=True)
    SIZES_JSON.write_text(json.dumps(sizes, indent=2) + "\n")
    _log(f"dedup: {json.dumps(sizes)}")
    return sizes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dedup", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("loda")
    p.add_argument("--keep-clone", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("sm")
    p.add_argument("a_numbers", nargs="+")
    sub.add_parser("report")
    args = ap.parse_args(argv)
    if args.cmd == "loda":
        build_loda(keep_clone=args.keep_clone, force=args.force)
        report()
    elif args.cmd == "sm":
        for a in args.a_numbers:
            ment = sm_mentions(a)
            _log(f"dedup: SM {a}: {len(ment)} A-number mentions")
        report()
    elif args.cmd == "report":
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
