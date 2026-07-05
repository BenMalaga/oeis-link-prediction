"""SMOKE check: cheapest baseline features on <= 200 TRAIN-split pairs.

    python -m src.smoke [--n 200]

Proves the pipeline runs end-to-end (splits -> terms -> pair features) on a
tiny budget. INTEGRITY: reads the TRAIN split ONLY (never val/test), computes
descriptive aggregates over train positives, this is not a retrieval
evaluation and produces no benchmark metric. Writes results/smoke_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from .features import load_terms_subset, pair_features, sequence_features
from .fetch_data import PROJECT_ROOT, log as _log
from .splits import SPLITS_DIR

OUT = PROJECT_ROOT / "results" / "smoke_check.json"
SEED = 42


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="smoke", description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args(argv)

    t0 = time.monotonic()
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    pos = train[train["label"] == 1]
    rng = np.random.default_rng(SEED)
    take = pos.iloc[np.sort(rng.choice(len(pos), size=min(args.n, len(pos)),
                                       replace=False))]
    _log(f"smoke: {len(take)} TRAIN positive pairs (train rows: {len(train):,})")

    need = set(take["u"].tolist()) | set(take["v"].tolist())
    t_load = time.monotonic()
    terms = load_terms_subset(need)
    _log(f"smoke: loaded terms for {len(terms):,} sequences "
         f"in {time.monotonic()-t_load:.1f}s")

    t_feat = time.monotonic()
    rows = []
    for u, v, stratum in zip(take["u"], take["v"], take["stratum"]):
        a, b = terms.get(int(u)), terms.get(int(v))
        if not a or not b:
            continue
        f = pair_features(a, b)
        f["stratum"] = stratum
        rows.append(f)
    feat_dt = time.monotonic() - t_feat
    df = pd.DataFrame(rows)
    # one per-sequence feature vector as a shape sanity check
    any_terms = next(iter(terms.values()))
    vec = sequence_features(any_terms)

    summary = {
        "n_pairs": int(len(df)),
        "split_source": "train ONLY (no val/test read)",
        "seed": SEED,
        "wall_seconds_total": round(time.monotonic() - t0, 2),
        "wall_seconds_pair_features": round(feat_dt, 2),
        "per_pair_ms": round(1000 * feat_dt / max(len(df), 1), 2),
        "seq_feature_dim": int(vec.shape[0]),
        "train_aggregates": {
            "mean_term_set_jaccard": round(float(df["term_set_jaccard"].mean()), 4),
            "mean_ngram3_jaccard": round(float(df["ngram3_jaccard"].mean()), 4),
            "frac_exact_subseq": round(float(df["exact_subseq"].mean()), 4),
            "frac_affine_match": round(float(df["affine_match"].mean()), 4),
            "by_stratum_frac_affine": {
                k: round(float(g["affine_match"].mean()), 4)
                for k, g in df.groupby("stratum")
            },
        },
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    _log(f"smoke: wrote {OUT.relative_to(PROJECT_ROOT)}: "
         f"{json.dumps(summary['train_aggregates'])}")
    _log(f"smoke: TOTAL wall {summary['wall_seconds_total']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
