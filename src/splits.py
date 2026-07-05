"""Pre-registered train/val/test edge splits + negatives (PRE_REGISTRATION.md 5.2).

    python -m src.splits          # refuses to run unless PRE_REGISTRATION.md exists
    python -m src.splits --force  # rebuild

INTEGRITY GUARD: this module materializes the held-out test edges. It refuses
to run before PRE_REGISTRATION.md exists (locked + committed). Nothing in this
module *evaluates* anything: it writes the splits and a manifest of counts +
SHA256 hashes, and never prints test-edge contents.

Frozen procedure (seed 42, numpy default_rng):
  1. Pair table: mention-level strata (stratum_v1) -> strongest label per
     directed edge -> undirected pairs (u<v) with strongest pair label.
  2. Eligibility: both endpoints in stripped.gz; neither endpoint has %K dead.
     Pairs with an endpoint having <8 terms stay, flagged `lt8_any` (reported
     separately at evaluation, per pre-reg section 5.2/6.2).
  3. Per stratum (s1,s2,s3,s4 in order), permute eligible pairs (sorted by
     (u,v) first, single rng): test = first 10%, val = next 9%, train = rest.
  4. Negatives: 5 per positive, per split (train, then val, then test, same
     rng). Universe: eligible nodes (in stripped, not dead). Constraints:
     u<v, |u-v|>20 (same-author/family adjacency leak guard), pair not in the
     FULL undirected xref pair set (not just the split), no duplicates within
     or across splits.
  5. Outputs: data/splits/{train,val,test}.parquet (columns u, v, stratum,
     label 1/0, lt8_any), data/splits/MANIFEST.json + committed copy at
     results/splits_manifest.json (counts per split x stratum x label,
     eligibility funnel, seed, SHA256 of each parquet, code + data versions).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .build_graph import SEQ_META_PARQUET, parse_stripped_lengths
from .fetch_data import DATA_DIR, PROJECT_ROOT, log as _log
from .stratify import XREFS_V1_PARQUET, pair_table

SPLITS_DIR = DATA_DIR / "splits"
MANIFEST = SPLITS_DIR / "MANIFEST.json"
MANIFEST_COPY = PROJECT_ROOT / "results" / "splits_manifest.json"
PREREG = PROJECT_ROOT / "PRE_REGISTRATION.md"

SEED = 42
TEST_FRAC = 0.10
VAL_FRAC = 0.09           # of the stratum total (~10% of the post-test remainder)
NEG_PER_POS = 5
ADJ_EXCLUSION = 20        # |u - v| <= 20 forbidden for negatives
MIN_TERMS = 8


def _pair_key(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u.astype(np.int64) * (1 << 20) + v.astype(np.int64)


def eligible_pairs() -> tuple[pd.DataFrame, dict, set[int], np.ndarray]:
    """Apply the frozen eligibility rules; return (pairs, funnel, nodes, all_keys)."""
    n_terms = parse_stripped_lengths()
    meta = pd.read_parquet(SEQ_META_PARQUET)
    kw = meta["keywords"].fillna("").str.split(",")
    dead = {int(a) for a, k in zip(meta["a_number"], kw) if "dead" in k}

    pairs = pair_table("stratum_v1")
    funnel = {"undirected_pairs": int(len(pairs))}
    all_keys = _pair_key(pairs["u"].to_numpy(), pairs["v"].to_numpy())

    sset = set(n_terms)
    m_str = pairs["u"].isin(sset) & pairs["v"].isin(sset)
    pairs = pairs[m_str]
    funnel["dropped_not_in_stripped"] = int((~m_str).sum())

    m_dead = pairs["u"].isin(dead) | pairs["v"].isin(dead)
    pairs = pairs[~m_dead]
    funnel["dropped_touching_dead"] = int(m_dead.sum())
    funnel["eligible_pairs"] = int(len(pairs))

    lt8 = {a for a, n in n_terms.items() if n < MIN_TERMS}
    pairs = pairs.assign(
        lt8_any=(pairs["u"].isin(lt8) | pairs["v"].isin(lt8)).astype("int8")
    )
    funnel["eligible_pairs_lt8_flagged"] = int(pairs["lt8_any"].sum())

    nodes = sset - dead
    funnel["negative_universe_nodes"] = len(nodes)
    return pairs.reset_index(drop=True), funnel, nodes, all_keys


def draw_splits(pairs: pd.DataFrame, rng: np.random.Generator):
    parts = {"train": [], "val": [], "test": []}
    for s in ("s1", "s2", "s3", "s4"):
        block = pairs[pairs["stratum"] == s].sort_values(["u", "v"]).reset_index(drop=True)
        perm = rng.permutation(len(block))
        n_test = int(len(block) * TEST_FRAC)
        n_val = int(len(block) * VAL_FRAC)
        parts["test"].append(block.iloc[perm[:n_test]])
        parts["val"].append(block.iloc[perm[n_test:n_test + n_val]])
        parts["train"].append(block.iloc[perm[n_test + n_val:]])
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def draw_negatives(n: int, nodes: np.ndarray, forbidden: set[int],
                   rng: np.random.Generator) -> pd.DataFrame:
    """Rejection-sample n negative pairs (u<v) honoring the frozen constraints."""
    got_u: list[np.ndarray] = []
    got_v: list[np.ndarray] = []
    need = n
    seen = forbidden  # mutated: accepted keys are added (no dupes across calls)
    while need > 0:
        m = max(int(need * 1.5), 1024)
        a = nodes[rng.integers(0, len(nodes), m)]
        b = nodes[rng.integers(0, len(nodes), m)]
        u = np.minimum(a, b)
        v = np.maximum(a, b)
        ok = (v - u) > ADJ_EXCLUSION
        u, v = u[ok], v[ok]
        keys = _pair_key(u, v)
        # drop in-batch duplicates, then anything already seen/forbidden
        _, first_idx = np.unique(keys, return_index=True)
        u, v, keys = u[np.sort(first_idx)], v[np.sort(first_idx)], keys[np.sort(first_idx)]
        fresh = np.fromiter((k not in seen for k in keys), bool, len(keys))
        u, v, keys = u[fresh], v[fresh], keys[fresh]
        take = min(need, len(u))
        got_u.append(u[:take])
        got_v.append(v[:take])
        seen.update(keys[:take].tolist())
        need -= take
    return pd.DataFrame({
        "u": np.concatenate(got_u).astype("int32"),
        "v": np.concatenate(got_v).astype("int32"),
    })


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(force: bool = False) -> dict:
    if not PREREG.exists():
        raise SystemExit(
            "REFUSING to draw splits: PRE_REGISTRATION.md is not locked/committed. "
            "(the pre-registration)"
        )
    if MANIFEST.exists() and not force:
        _log("splits: skip (exists); --force to rebuild")
        return json.loads(MANIFEST.read_text())

    t0 = time.monotonic()
    pairs, funnel, node_set, all_keys = eligible_pairs()
    _log(f"splits: eligibility funnel {funnel}")
    rng = np.random.default_rng(SEED)
    splits = draw_splits(pairs, rng)

    nodes = np.array(sorted(node_set), dtype=np.int64)
    forbidden = set(all_keys.tolist())  # ALL known pairs, not just eligible
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "seed": SEED,
        "test_frac": TEST_FRAC,
        "val_frac": VAL_FRAC,
        "neg_per_pos": NEG_PER_POS,
        "adjacency_exclusion": ADJ_EXCLUSION,
        "min_terms_flag": MIN_TERMS,
        "eligibility_funnel": funnel,
        "stratum_column": "stratum_v1",
        "snapshot_date": "2026-06-10",
        "files": {},
        "counts": {},
    }
    for name in ("train", "val", "test"):
        pos = splits[name].copy()
        pos["label"] = np.int8(1)
        neg = draw_negatives(NEG_PER_POS * len(pos), nodes, forbidden, rng)
        neg["stratum"] = "neg"
        neg["lt8_any"] = np.int8(0)
        neg["label"] = np.int8(0)
        df = pd.concat([pos, neg], ignore_index=True)[
            ["u", "v", "stratum", "label", "lt8_any"]
        ]
        df["u"] = df["u"].astype("int32")
        df["v"] = df["v"].astype("int32")
        path = SPLITS_DIR / f"{name}.parquet"
        df.to_parquet(path, engine="pyarrow", index=False)
        manifest["files"][name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(path),
            "rows": int(len(df)),
        }
        by = (
            df.groupby(["stratum", "label"]).size()
            .reset_index(name="n")
            .to_dict(orient="records")
        )
        manifest["counts"][name] = by
        _log(f"splits: {name}: {len(pos):,} pos + {len(neg):,} neg")

    try:
        manifest["code_commit"] = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        manifest["code_commit"] = "unknown"
    manifest["xrefs_v1_sha256"] = sha256(XREFS_V1_PARQUET)
    manifest["wall_seconds"] = round(time.monotonic() - t0, 1)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    MANIFEST_COPY.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_COPY.write_text(json.dumps(manifest, indent=2) + "\n")
    _log(f"splits: wrote {MANIFEST.relative_to(PROJECT_ROOT)} "
         f"(+ copy in results/) in {manifest['wall_seconds']}s")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="splits", description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    build(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
