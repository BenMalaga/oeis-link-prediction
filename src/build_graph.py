"""Stage-1 ingest: build the OEIS %Y cross-reference edge list + descriptive stats.

Idempotent, from scratch (run from the project root, inside .venv):

    python -m src.build_graph              # ensure data -> extract -> stats
    python -m src.build_graph --force      # re-extract + recompute even if outputs exist

What it does (and does NOT do):
  1. Ensures stripped.gz + names.gz are present (delegates to src.fetch_data).
  2. Ensures the oeisdata clone is present (--depth 1 --no-checkout, LFS skipped;
     ~341 MB pack, approved main-phase download) and records the commit SHA +
     upstream sync time in data/SNAPSHOT.txt.
  3. Streams every seq/**/A*.seq blob straight out of the git pack (`git archive`,
     no working tree) and extracts ALL %Y cross-reference lines into a clean
     edge list -> data/xrefs.parquet, plus per-sequence metadata (%K keywords,
     term counts) -> data/seq_meta.parquet.
  4. Computes the descriptive aggregates the pre-registration draft marked
     [UNVERIFIED] -> results/data_stats.json + results/data_stats.md.

SCIENTIFIC-INTEGRITY GUARD: the %Y graph is LABEL data for the link-prediction
benchmark. This module only *builds* it and reports aggregate counts. It draws
no train/test split, trains no model, and computes no retrieval metric. Nothing
may be evaluated against held-out edges before PRE_REGISTRATION.md is locked and
committed (the pre-registration).

Edge-list schema (data/xrefs.parquet), one row per A-number MENTION in a %Y line:
    source       int32   A-number of the .seq file the line came from
    target       int32   A-number mentioned in the line (self-mentions dropped)
    y_line_idx   int16   0-based index of the %Y line within the source entry
    link_type    str     line-level parse: duplicate | transform | Cf. | See also | other
    paren        str     parenthetical attached to THIS target mention, if any
                         (e.g. "complement" from "A001690 (complement)"), the
                         target-attributed cue used for stratification
    stratum      str     DRAFT edge stratum s1|s2|s3|s4 (heuristics v0 below,
                         NOT locked; frozen + audited only at pre-reg lock)
    raw_text     str     the %Y line text (after the "%Y A###### " prefix)

Draft stratum heuristics v0 (pre-reg draft section 5.1). %Y lines often list
dozens of targets ("Cf. A001622 (phi), A039834 (signed), A001690 (complement),
..."), so a line-level keyword must NOT be smeared over every target on the
line: a transform cue counts only when target-attributed (in that mention's
parenthetical) or when the line is "focused" (<= 3 targets, e.g. "Partial sums
of A000108."). Priority per mention:
    s1 duplicate-of: line/paren matches S1_RE, or either endpoint's name says
       "duplicate of A<other endpoint>" (names.gz).
    s2 (paren): this mention's parenthetical matches a transform keyword from
       the draft battery (binomial/Euler/Moebius transform, partial sums,
       differences, bisection, complement, signed/unsigned version,
       offset/shift, constant multiple, decimation).
    s4 contextual: either endpoint is a constant-type sequence (keyword `cons`
       or name starting "Decimal expansion"/"Continued fraction expansion"),
       e.g. Fibonacci <-> phi (A000045 <-> A001622), presumed not term-derivable.
    s2 (focused line): line text matches a transform keyword AND the line has
       <= 3 targets.
    s3 see-also: everything else (generic "Cf."/family links).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import tarfile
import time
from collections.abc import Iterator
from pathlib import Path

from .fetch_data import (
    DATA_DIR,
    OEISDATA_DIR,
    PROJECT_ROOT,
    fetch_bulk,
    fetch_oeisdata,
    log as _log,
    record_oeisdata_snapshot,
)

RESULTS_DIR = PROJECT_ROOT / "results"
XREFS_PARQUET = DATA_DIR / "xrefs.parquet"
SEQ_META_PARQUET = DATA_DIR / "seq_meta.parquet"
STATS_JSON = RESULTS_DIR / "data_stats.json"
STATS_MD = RESULTS_DIR / "data_stats.md"

A_RE = re.compile(r"A(\d{6,7})")
# A-number mention plus an optional immediately-following parenthetical, which
# editors use for target-attributed cues: "A001690 (complement)", "A001622 (phi)".
# The lookahead keeps parentheticals that themselves contain A-numbers (e.g.
# "(see A000032)") out of the capture so those mentions are still extracted.
A_PAREN_RE = re.compile(r"A(\d{6,7})(?:\s*\((?![^()]*A\d{6})([^()]{1,120})\))?")

# ----------------------------------------------------- draft classifiers (v0)
# These are DRAFT heuristics for descriptive counts only. The locked versions
# (frozen wording + 100-edge hand-labeled audit) are produced at pre-reg lock.

S1_RE = re.compile(r"duplicate of|essentially the same|essentially a duplicate", re.I)
S2_RE = re.compile(
    r"binomial transform|euler transform|m(?:oe|ö|o)bius transform"
    r"|partial sums?|first differences?|bisections?|complement"
    r"|signed version|unsigned version|decimations?|offset|shift"
    r"|constant multiple",
    re.I,
)
CF_RE = re.compile(r"^cf\b", re.I)
SEE_ALSO_RE = re.compile(r"^see also\b", re.I)
DUP_NAME_RE = re.compile(r"duplicate of A(\d{6,7})", re.I)
CONST_NAME_RE = re.compile(
    r"^(?:decimal expansion|continued fraction expansion)", re.I
)


def classify_line(text: str) -> str:
    """Line-level link type: duplicate | transform | Cf. | See also | other."""
    if S1_RE.search(text):
        return "duplicate"
    if S2_RE.search(text):
        return "transform"
    if CF_RE.match(text):
        return "Cf."
    if SEE_ALSO_RE.match(text):
        return "See also"
    return "other"


def log(msg: str) -> None:
    _log(f"build_graph: {msg}")


# ------------------------------------------------------------ pack streaming

def iter_seq_files() -> Iterator[tuple[int, str]]:
    """Yield (a_number, full_text) for every seq/**/A*.seq blob at oeisdata HEAD.

    Streams `git archive` output through a tar pipe, works on the --no-checkout
    clone, touches no working tree, and never materializes 400k small files.
    """
    if not OEISDATA_DIR.exists():
        raise FileNotFoundError(
            f"{OEISDATA_DIR} missing, run `python -m src.fetch_data oeisdata --confirm`."
        )
    proc = subprocess.Popen(
        ["git", "-C", str(OEISDATA_DIR), "archive", "--format=tar", "HEAD", "seq"],
        stdout=subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".seq"):
                    continue
                stem = member.name.rsplit("/", 1)[-1][:-len(".seq")]  # "A000045"
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                yield int(stem[1:]), fh.read().decode("utf-8", errors="replace")
    finally:
        proc.stdout.close()
        if proc.wait() != 0:
            raise RuntimeError(f"git archive exited {proc.returncode}")


def list_bfile_numbers() -> set[int]:
    """A-numbers that have a b-file in oeisdata's files/** tree (LFS pointers,
    listed via ls-tree, never downloaded)."""
    out = subprocess.run(
        ["git", "-C", str(OEISDATA_DIR), "ls-tree", "-r", "--name-only", "HEAD", "files"],
        check=True, capture_output=True, text=True,
    ).stdout
    bfile_re = re.compile(r"/b(\d+)\.txt$")
    return {int(m.group(1)) for line in out.splitlines() if (m := bfile_re.search(line))}


# ----------------------------------------------------------- bulk dump parses

def parse_stripped_lengths() -> dict[int, int]:
    """stripped.gz -> {a_number: n_terms} (terms themselves not retained, this
    stage needs presence + length only, which keeps peak RAM well under 1 GB)."""
    out: dict[int, int] = {}
    with gzip.open(DATA_DIR / "stripped.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("A"):
                continue
            a_str, _, terms_str = line.partition(" ")
            out[int(a_str[1:])] = sum(1 for t in terms_str.strip().strip(",").split(",") if t)
    return out


def parse_names() -> dict[int, str]:
    out: dict[int, str] = {}
    with gzip.open(DATA_DIR / "names.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("A"):
                continue
            a_str, _, name = line.partition(" ")
            out[int(a_str[1:])] = name.strip()
    return out


# --------------------------------------------------------------- extraction

def extract_xrefs(force: bool = False) -> None:
    """Stream all .seq blobs -> data/xrefs.parquet + data/seq_meta.parquet."""
    import pandas as pd

    if XREFS_PARQUET.exists() and SEQ_META_PARQUET.exists() and not force:
        log(f"skip extraction (exists): {XREFS_PARQUET.name}, {SEQ_META_PARQUET.name}")
        return

    t0 = time.monotonic()
    src_l: list[int] = []
    tgt_l: list[int] = []
    idx_l: list[int] = []
    type_l: list[str] = []
    paren_l: list[str] = []
    nline_l: list[int] = []  # number of targets on this mention's line
    raw_l: list[str] = []
    meta_a: list[int] = []
    meta_kw: list[str] = []
    meta_ny: list[int] = []
    n_files = 0
    n_self = 0

    for a_num, text in iter_seq_files():
        n_files += 1
        y_idx = 0
        keywords: list[str] = []
        for line in text.splitlines():
            if line.startswith("%Y"):
                # "%Y A000045 Cf. A001622 (phi), ..." -> strip tag + self A-number
                body = line[2:].strip()
                _, _, rest = body.partition(" ")
                rest = rest.strip()
                ltype = classify_line(rest)
                mentions = A_PAREN_RE.findall(rest)
                n_line = sum(1 for m, _ in mentions if int(m) != a_num)
                for m, paren in mentions:
                    t = int(m)
                    if t == a_num:
                        n_self += 1
                        continue
                    src_l.append(a_num)
                    tgt_l.append(t)
                    idx_l.append(y_idx)
                    type_l.append(ltype)
                    paren_l.append(paren)
                    nline_l.append(n_line)
                    raw_l.append(rest)
                y_idx += 1
            elif line.startswith("%K"):
                body = line[2:].strip()
                _, _, kw = body.partition(" ")
                keywords.append(kw.strip())
        meta_a.append(a_num)
        meta_kw.append(",".join(keywords))
        meta_ny.append(y_idx)
        if n_files % 100_000 == 0:
            log(f"  ... {n_files:,} .seq files in {time.monotonic() - t0:.0f}s")

    log(f"streamed {n_files:,} .seq files in {time.monotonic() - t0:.0f}s; "
        f"{len(src_l):,} raw target mentions ({n_self:,} self-mentions dropped)")

    edges = pd.DataFrame(
        {
            "source": pd.array(src_l, dtype="int32"),
            "target": pd.array(tgt_l, dtype="int32"),
            "y_line_idx": pd.array(idx_l, dtype="int16"),
            "link_type": type_l,
            "paren": paren_l,
            "raw_text": raw_l,
        }
    )
    meta = pd.DataFrame(
        {
            "a_number": pd.array(meta_a, dtype="int32"),
            "keywords": meta_kw,
            "n_y_lines": pd.array(meta_ny, dtype="int16"),
        }
    )

    # Draft edge-level stratum (heuristics v0, see module docstring).
    log("assigning draft strata (v0 heuristics; NOT the locked classifier)")
    names = parse_names()
    dup_of = {
        a: int(m.group(1)) for a, nm in names.items() if (m := DUP_NAME_RE.search(nm))
    }
    const_like = {
        a for a, nm in names.items() if CONST_NAME_RE.match(nm)
    } | {
        int(a) for a, kw in zip(meta_a, meta_kw)
        if "cons" in kw.split(",")
    }
    strata: list[str] = []
    for s, t, lt, paren, n_line in zip(src_l, tgt_l, type_l, paren_l, nline_l):
        if lt == "duplicate" or S1_RE.search(paren) or dup_of.get(s) == t or dup_of.get(t) == s:
            strata.append("s1")
        elif paren and S2_RE.search(paren):
            strata.append("s2")  # target-attributed transform cue
        elif s in const_like or t in const_like:
            strata.append("s4")
        elif lt == "transform" and n_line <= 3:
            strata.append("s2")  # focused line, e.g. "Partial sums of A000108."
        else:
            strata.append("s3")
    edges["stratum"] = strata

    edges.to_parquet(XREFS_PARQUET, engine="pyarrow", index=False)
    meta.to_parquet(SEQ_META_PARQUET, engine="pyarrow", index=False)
    log(f"wrote {XREFS_PARQUET.relative_to(PROJECT_ROOT)} "
        f"({XREFS_PARQUET.stat().st_size:,} B, {len(edges):,} rows)")
    log(f"wrote {SEQ_META_PARQUET.relative_to(PROJECT_ROOT)} "
        f"({SEQ_META_PARQUET.stat().st_size:,} B, {len(meta):,} rows)")


# -------------------------------------------------------------------- stats

def _summary(series) -> dict:
    """Degree-distribution summary: mean/quantiles/max as plain Python types."""
    import numpy as np

    arr = np.asarray(series)
    q = np.quantile(arr, [0.5, 0.9, 0.99])
    return {
        "mean": round(float(arr.mean()), 3),
        "median": float(q[0]),
        "p90": float(q[1]),
        "p99": float(q[2]),
        "max": int(arr.max()),
    }


def compute_stats() -> dict:
    """Descriptive aggregates only (counts/degrees/strata). No splits, no models."""
    import numpy as np
    import pandas as pd

    log("computing stats")
    sha = record_oeisdata_snapshot()
    n_terms = parse_stripped_lengths()
    names = parse_names()
    stripped_set = set(n_terms)
    name_only = sorted(set(names) - stripped_set)
    n_allocated = sum(1 for a in name_only if "allocated" in names[a].lower())
    dup_names_live = sum(
        1 for a, nm in names.items() if a in stripped_set and "duplicate of" in nm.lower()
    )
    tl = np.array(sorted(n_terms.values()))

    meta = pd.read_parquet(SEQ_META_PARQUET, engine="pyarrow")
    kw_lists = meta["keywords"].fillna("").str.split(",")
    kw_dead = meta.loc[[("dead" in k) for k in kw_lists], "a_number"]
    kw_counts = {
        kw: int(sum(1 for k in kw_lists if kw in k))
        for kw in ("dead", "dup", "dupe", "cons", "base", "fini", "sign", "nonn")
    }
    dead_set = set(kw_dead.tolist())

    edges = pd.read_parquet(
        XREFS_PARQUET, engine="pyarrow",
        columns=["source", "target", "link_type", "stratum"],
    )
    n_raw = len(edges)
    link_type_counts = edges["link_type"].value_counts().to_dict()

    # Dedup directed: keep the strongest stratum label per (source, target).
    rank = {"s1": 0, "s2": 1, "s4": 2, "s3": 3}
    edges["rank"] = edges["stratum"].map(rank).astype("int8")
    ded = (
        edges.sort_values("rank")
        .drop_duplicates(["source", "target"], keep="first")
        .loc[:, ["source", "target", "rank"]]
    )
    n_directed = len(ded)

    # Undirected collapse: canonical (u, v) = (min, max); symmetric if both
    # directions exist in the deduped directed graph.
    u = np.minimum(ded["source"].to_numpy(), ded["target"].to_numpy())
    v = np.maximum(ded["source"].to_numpy(), ded["target"].to_numpy())
    und = pd.DataFrame({"u": u, "v": v, "rank": ded["rank"].to_numpy()})
    g = und.groupby(["u", "v"], sort=False)
    pair_count = g.size()
    pair_rank = g["rank"].min()
    n_undirected = len(pair_count)
    n_symmetric = int((pair_count == 2).sum())
    n_asymmetric = int((pair_count == 1).sum())

    inv_rank = {0: "s1", 1: "s2", 2: "s4", 3: "s3"}
    strata_directed = {
        inv_rank[r]: int(c) for r, c in ded["rank"].value_counts().items()
    }
    strata_undirected = {
        inv_rank[r]: int(c) for r, c in pair_rank.value_counts().items()
    }

    # Restriction to sequences with terms in stripped.gz (the usable benchmark pool).
    pairs = pair_rank.reset_index()
    in_stripped_u = pairs["u"].isin(stripped_set)
    in_stripped_v = pairs["v"].isin(stripped_set)
    both = pairs[in_stripped_u & in_stripped_v]
    n_und_stripped = len(both)
    strata_und_stripped = {
        inv_rank[r]: int(c) for r, c in both["rank"].value_counts().items()
    }
    ded_strip_mask = ded["source"].isin(stripped_set) & ded["target"].isin(stripped_set)
    n_dir_stripped = int(ded_strip_mask.sum())
    n_edges_touching_dead = int(
        (ded["source"].isin(dead_set) | ded["target"].isin(dead_set)).sum()
    )

    # Degree distributions (zeros included over the full stripped.gz universe).
    out_deg = ded.groupby("source").size()
    in_deg = ded.groupby("target").size()
    und_deg = pd.concat([
        both.groupby("u").size(), both.groupby("v").size()
    ]).groupby(level=0).sum()
    stripped_idx = pd.Index(sorted(stripped_set))
    out_full = out_deg.reindex(stripped_idx, fill_value=0)
    in_full = in_deg.reindex(stripped_idx, fill_value=0)
    und_full = und_deg.reindex(stripped_idx, fill_value=0)
    n_isolated = int((und_full == 0).sum())

    # b-file coverage (from the LFS pointer tree, nothing downloaded).
    bfiles = list_bfile_numbers()
    n_bfiles_stripped = len(bfiles & stripped_set)

    stats = {
        "snapshot": {
            "date": "2026-06-10",
            "oeisdata_commit": sha,
            "stripped_rows": len(n_terms),
            "names_rows": len(names),
            "names_only_no_terms": len(name_only),
            "names_only_allocated_placeholders": n_allocated,
            "live_names_containing_duplicate_of": dup_names_live,
            "seq_files_in_oeisdata": int(len(meta)),
            "terms_per_seq": {
                "min": int(tl.min()), "median": float(np.median(tl)),
                "mean": round(float(tl.mean()), 2), "max": int(tl.max()),
                "n_lt_8_terms": int((tl < 8).sum()),
            },
        },
        "keywords": {
            "seq_with_keyword_line": int((meta["keywords"].fillna("") != "").sum()),
            "counts": kw_counts,
            "dead_also_in_stripped": int(len(dead_set & stripped_set)),
        },
        "edges": {
            "raw_target_mentions": n_raw,
            "deduped_directed": n_directed,
            "undirected_pairs": n_undirected,
            "symmetric_pairs_both_directions": n_symmetric,
            "asymmetric_pairs_one_direction": n_asymmetric,
            "deduped_directed_both_endpoints_in_stripped": n_dir_stripped,
            "undirected_pairs_both_endpoints_in_stripped": n_und_stripped,
            "deduped_directed_touching_dead": n_edges_touching_dead,
            "link_type_counts_raw_mentions": {
                str(k): int(v) for k, v in link_type_counts.items()
            },
        },
        "strata_draft_v0": {
            "note": ("heuristics v0, descriptive only; classifier frozen + "
                     "100-edge audit happens at pre-registration lock"),
            "deduped_directed": strata_directed,
            "undirected_pairs": strata_undirected,
            "undirected_pairs_both_in_stripped": strata_und_stripped,
        },
        "degrees_over_stripped_universe": {
            "note": "deduped directed graph; zeros included for all stripped.gz seqs",
            "out_degree": _summary(out_full),
            "in_degree": _summary(in_full),
            "undirected_degree_stripped_only_edges": _summary(und_full),
            "isolated_in_undirected_stripped_graph": n_isolated,
        },
        "bfiles": {
            "bfiles_in_oeisdata_tree": len(bfiles),
            "stripped_seqs_with_bfile": n_bfiles_stripped,
            "fraction_of_stripped": round(n_bfiles_stripped / len(n_terms), 4),
        },
    }
    return stats


def write_stats(stats: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_JSON.write_text(json.dumps(stats, indent=2) + "\n")
    log(f"wrote {STATS_JSON.relative_to(PROJECT_ROOT)}")

    e, s, d, b = (stats["edges"], stats["strata_draft_v0"],
                  stats["degrees_over_stripped_universe"], stats["bfiles"])
    snap = stats["snapshot"]
    lines = [
        "# OEIS xref-graph descriptive stats (Stage 1 ingest)",
        "",
        f"Snapshot **{snap['date']}**, oeisdata commit `{snap['oeisdata_commit'][:12]}`. "
        "Aggregates only, no splits drawn, no models run (pre-reg not yet locked).",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Sequences in stripped.gz | {snap['stripped_rows']:,} |",
        f"| Rows in names.gz | {snap['names_rows']:,} |",
        f"| .seq files in oeisdata | {snap['seq_files_in_oeisdata']:,} |",
        f"| Raw %Y target mentions | {e['raw_target_mentions']:,} |",
        f"| Deduped directed edges | {e['deduped_directed']:,} |",
        f"| Undirected pairs | {e['undirected_pairs']:,} |",
        f"| ...symmetric (both directions) | {e['symmetric_pairs_both_directions']:,} |",
        f"| ...asymmetric (one direction) | {e['asymmetric_pairs_one_direction']:,} |",
        f"| Undirected pairs, both endpoints in stripped.gz | "
        f"{e['undirected_pairs_both_endpoints_in_stripped']:,} |",
        f"| b-files in oeisdata tree | {b['bfiles_in_oeisdata_tree']:,} "
        f"({b['fraction_of_stripped']:.1%} of stripped) |",
        "",
        "## Draft strata (v0 heuristics; frozen + audited only at pre-reg lock)",
        "",
        "| Stratum | Deduped directed | Undirected | Undirected, both in stripped |",
        "|---|---|---|---|",
    ]
    for k in ("s1", "s2", "s3", "s4"):
        lines.append(
            f"| {k} | {s['deduped_directed'].get(k, 0):,} "
            f"| {s['undirected_pairs'].get(k, 0):,} "
            f"| {s['undirected_pairs_both_in_stripped'].get(k, 0):,} |"
        )
    lines += [
        "",
        "## Degrees (deduped directed graph; zeros included over all stripped.gz)",
        "",
        "| Degree | mean | median | p90 | p99 | max |",
        "|---|---|---|---|---|---|",
    ]
    for label, key in (("out", "out_degree"), ("in", "in_degree"),
                       ("undirected (stripped-only edges)",
                        "undirected_degree_stripped_only_edges")):
        m = d[key]
        lines.append(f"| {label} | {m['mean']} | {m['median']:g} | {m['p90']:g} "
                     f"| {m['p99']:g} | {m['max']:,} |")
    lines += [
        "",
        f"Isolated sequences (no stripped-to-stripped xref at all): "
        f"{d['isolated_in_undirected_stripped_graph']:,}",
        "",
    ]
    STATS_MD.write_text("\n".join(lines))
    log(f"wrote {STATS_MD.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build_graph", description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-extract parquet outputs even if they exist")
    args = ap.parse_args(argv)

    fetch_bulk()                       # idempotent: skips if present
    fetch_oeisdata(confirm=True)       # idempotent: skips if present (approved download)
    extract_xrefs(force=args.force)
    write_stats(compute_stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
