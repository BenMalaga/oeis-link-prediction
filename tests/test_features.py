"""Unit tests for term-only feature extractors on toy sequences."""

import numpy as np
import pytest

from src.features import (
    K_PROFILE,
    affine_match,
    exact_subsequence_match,
    feature_names,
    ngram_jaccard,
    pair_features,
    sequence_features,
    term_set_jaccard,
)

FIB = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
NATURALS = list(range(1, 17))
POW2 = [2**i for i in range(16)]
ALT = [(-1) ** i * (i + 1) for i in range(16)]  # 1, -2, 3, -4, ...


# ------------------------------------------------------------- vector shape


def test_vector_shape_and_names():
    v = sequence_features(FIB)
    assert v.shape == (K_PROFILE + 34,)
    assert len(feature_names()) == v.shape[0]
    assert np.all(np.isfinite(v))


def test_empty_and_short_sequences():
    assert np.all(sequence_features([]) == 0)
    v = sequence_features([5])
    assert np.isfinite(v).all()


# --------------------------------------------------------------- semantics


def test_growth_slope_pow2():
    v = sequence_features(POW2)
    names = feature_names()
    slope = v[names.index("growth_slope")]
    assert slope == pytest.approx(np.log(2), rel=1e-6)


def test_parity_profile_even_sequence():
    v = sequence_features([2, 4, 6, 8, 10, 12, 14, 16])
    names = feature_names()
    assert v[names.index("mod2_0")] == 1.0
    assert v[names.index("frac_even")] == 1.0


def test_sign_alternation():
    v = sequence_features(ALT)
    names = feature_names()
    assert v[names.index("sign_alt")] == 1.0
    assert v[names.index("frac_neg")] == pytest.approx(0.5)


def test_residue_histogram_mod3():
    v = sequence_features([3, 6, 9, 12, 15, 18, 21, 24])
    names = feature_names()
    assert v[names.index("mod3_0")] == 1.0
    assert v[names.index("mod3_1")] == 0.0


# -------------------------------------------------------------- pair level


def test_jaccard_identical():
    assert term_set_jaccard(FIB, FIB) == 1.0
    assert ngram_jaccard(FIB, FIB) == 1.0


def test_jaccard_disjoint():
    assert term_set_jaccard([1, 2, 3], [7, 8, 9]) == 0.0


def test_exact_subsequence_shifted():
    # Fibonacci with the first two terms dropped is a contiguous run of FIB
    assert exact_subsequence_match(FIB[2:], FIB)
    assert not exact_subsequence_match(NATURALS, POW2)


def test_affine_match_constant_multiple():
    doubled = [2 * t for t in FIB]
    got = affine_match(FIB, doubled)
    assert got is not None
    c, d, s = got
    assert (c, d, s) == (2, 0, 0)


def test_affine_match_shift_and_offset():
    # b(n) = 3*a(n+2) - 1
    b = [3 * t - 1 for t in FIB[2:]]
    got = affine_match(FIB, b)
    assert got is not None
    c, d, s = got
    assert c == 3 and d == -1 and s == 2


def test_affine_match_rejects_unrelated():
    assert affine_match(NATURALS, POW2) is None


def test_pair_features_bundle():
    f = pair_features(FIB, [2 * t for t in FIB])
    assert f["affine_match"] == 1.0
    assert 0.0 <= f["term_set_jaccard"] <= 1.0
    assert set(f) == {"term_set_jaccard", "ngram3_jaccard",
                      "exact_subseq", "affine_match"}
