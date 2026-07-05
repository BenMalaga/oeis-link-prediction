"""Pre-registration lock audit: sample undirected pairs per stratum (seeded) for
hand-labeling of the stratification classifier's precision.

    python -m src.audit_strata sample                       # v0 dev sample (seed 42)
    python -m src.audit_strata sample --column stratum_v1 --seed 43 --blind \
        --out results/strata_audit2_blind.csv               # v1 confirmatory sample
    python -m src.audit_strata report results/strata_audit_sample_labeled.csv ...

This is LABEL-data QA only: it draws audit samples of edge classifications for
review. It does NOT draw the train/test holdout (different procedure, different
artifact; the holdout is drawn only after PRE_REGISTRATION.md is locked), trains
no model, and computes no retrieval metric.

Sampling frame: undirected pairs with both endpoints in stripped.gz, stratum =
strongest (min-rank) label over all directed mentions of the pair, exactly as in
src.build_graph.compute_stats / src.stratify.pair_table. 25 pairs per stratum.

--blind writes the labeling CSV with the classifier's stratum and the v0
link_type column REMOVED and rows shuffled, plus a separate key CSV, so the
auditor labels without seeing the machine's answer.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .build_graph import XREFS_PARQUET, parse_names, parse_stripped_lengths
from .fetch_data import PROJECT_ROOT

RANK = {"s1": 0, "s2": 1, "s4": 2, "s3": 3}
PER_STRATUM = 25


def build_sample(column: str = "stratum", seed: int = 42,
                 per_stratum: int = PER_STRATUM) -> pd.DataFrame:
    if column == "stratum":
        path = XREFS_PARQUET
    else:
        from .stratify import XREFS_V1_PARQUET as path  # type: ignore
    edges = pd.read_parquet(
        path,
        columns=["source", "target", "link_type", "paren", "raw_text", column],
    )
    edges["rank"] = edges[column].map(RANK).astype("int8")
    # Strongest label per directed (source, target), then per undirected pair,
    # mirrors the pair-level collapse used everywhere else.
    ded = edges.sort_values("rank").drop_duplicates(["source", "target"], keep="first")
    u = np.minimum(ded["source"].to_numpy(), ded["target"].to_numpy())
    v = np.maximum(ded["source"].to_numpy(), ded["target"].to_numpy())
    ded = ded.assign(u=u, v=v)
    rep = ded.sort_values("rank").drop_duplicates(["u", "v"], keep="first")

    stripped = set(parse_stripped_lengths())
    rep = rep[rep["u"].isin(stripped) & rep["v"].isin(stripped)]

    names = parse_names()
    rng = np.random.default_rng(seed)
    parts = []
    for s in ("s1", "s2", "s3", "s4"):
        pool = rep[rep[column] == s]
        idx = rng.choice(len(pool), size=min(per_stratum, len(pool)), replace=False)
        parts.append(pool.iloc[np.sort(idx)])
    sample = pd.concat(parts, ignore_index=True)
    sample["name_source"] = [names.get(a, "") for a in sample["source"]]
    sample["name_target"] = [names.get(a, "") for a in sample["target"]]
    sample["hand_label"] = ""
    sample = sample.rename(columns={column: "stratum"})
    cols = ["u", "v", "source", "target", "stratum", "link_type", "paren",
            "raw_text", "name_source", "name_target", "hand_label"]
    return sample[cols]


def cmd_sample(args) -> int:
    sample = build_sample(column=args.column, seed=args.seed)
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.blind:
        key = sample[["u", "v", "stratum", "link_type"]]
        key_path = out.with_name(out.stem + "_key.csv")
        key.to_csv(key_path, index=False)
        blind = sample.drop(columns=["stratum", "link_type"])
        blind = blind.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        blind.to_csv(out, index=False)
        print(f"wrote BLIND {out} ({len(blind)} rows) + key {key_path}")
    else:
        sample.to_csv(out, index=False)
        print(f"wrote {out} ({len(sample)} rows; seed {args.seed}, "
              f"{PER_STRATUM}/stratum, column {args.column})")
    return 0


def cmd_report(args) -> int:
    for path in args.labeled:
        df = pd.read_csv(PROJECT_ROOT / path)
        if "stratum" not in df.columns:  # blinded file: join the key back in
            key = pd.read_csv(
                (PROJECT_ROOT / path).with_name(
                    (PROJECT_ROOT / path).stem.replace("_labeled", "") + "_key.csv")
            )
            df = df.merge(key[["u", "v", "stratum"]], on=["u", "v"], how="left")
        df["correct"] = df["stratum"] == df["hand_label"]
        print(f"== {path}")
        print(df.groupby("stratum")["correct"].agg(["sum", "count", "mean"]))
        print(pd.crosstab(df["stratum"], df["hand_label"]))
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="audit_strata", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--column", default="stratum", choices=["stratum", "stratum_v1"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--out", default="results/strata_audit_sample.csv")
    p.set_defaults(func=cmd_sample)
    p = sub.add_parser("report")
    p.add_argument("labeled", nargs="+")
    p.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
