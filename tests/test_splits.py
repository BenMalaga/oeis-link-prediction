"""Unit tests for the split-drawing core on a synthetic graph (no real data,
no PRE_REGISTRATION gating, these exercise draw_splits/draw_negatives only)."""

import numpy as np
import pandas as pd
import pytest

from src.splits import (
    ADJ_EXCLUSION,
    _pair_key,
    draw_negatives,
    draw_splits,
)


@pytest.fixture
def toy_pairs():
    rng = np.random.default_rng(7)
    rows = []
    used = set()
    # 400 synthetic undirected pairs across 4 strata over nodes 1..5000
    while len(rows) < 400:
        u, v = sorted(rng.integers(1, 5001, 2).tolist())
        if u == v or (u, v) in used:
            continue
        used.add((u, v))
        rows.append((u, v, ("s1", "s2", "s3", "s4")[len(rows) % 4], 0))
    return pd.DataFrame(rows, columns=["u", "v", "stratum", "lt8_any"])


def test_split_sizes_and_disjointness(toy_pairs):
    rng = np.random.default_rng(42)
    splits = draw_splits(toy_pairs, rng)
    n = sum(len(s) for s in splits.values())
    assert n == len(toy_pairs)
    keys = [set(map(tuple, s[["u", "v"]].to_numpy())) for s in splits.values()]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])
    # 10% of each 100-pair stratum -> 10 test pairs per stratum
    test = splits["test"]
    assert (test["stratum"].value_counts() == 10).all()
    val = splits["val"]
    assert (val["stratum"].value_counts() == 9).all()


def test_split_determinism(toy_pairs):
    a = draw_splits(toy_pairs, np.random.default_rng(42))
    b = draw_splits(toy_pairs, np.random.default_rng(42))
    for k in ("train", "val", "test"):
        pd.testing.assert_frame_equal(
            a[k].reset_index(drop=True), b[k].reset_index(drop=True))


def test_negatives_respect_constraints(toy_pairs):
    nodes = np.arange(1, 5001, dtype=np.int64)
    forbidden = set(
        _pair_key(toy_pairs["u"].to_numpy(), toy_pairs["v"].to_numpy()).tolist())
    n_forbidden_before = len(forbidden)
    neg = draw_negatives(500, nodes, forbidden, np.random.default_rng(42))
    assert len(neg) == 500
    assert (neg["v"] - neg["u"] > ADJ_EXCLUSION).all()          # adjacency guard
    keys = _pair_key(neg["u"].to_numpy(), neg["v"].to_numpy())
    assert len(set(keys.tolist())) == 500                        # no duplicates
    # none of the negatives is a known positive pair
    pos_keys = _pair_key(toy_pairs["u"].to_numpy(), toy_pairs["v"].to_numpy())
    assert not set(keys.tolist()) & set(pos_keys.tolist())
    # accepted negatives were added to the forbidden set (cross-split no-dupe)
    assert len(forbidden) == n_forbidden_before + 500


def test_negatives_disjoint_across_calls(toy_pairs):
    nodes = np.arange(1, 5001, dtype=np.int64)
    forbidden = set()
    rng = np.random.default_rng(0)
    a = draw_negatives(300, nodes, forbidden, rng)
    b = draw_negatives(300, nodes, forbidden, rng)
    ka = set(_pair_key(a["u"].to_numpy(), a["v"].to_numpy()).tolist())
    kb = set(_pair_key(b["u"].to_numpy(), b["v"].to_numpy()).tolist())
    assert not ka & kb
