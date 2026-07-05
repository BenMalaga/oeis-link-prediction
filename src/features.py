"""Term-only baseline feature extractors (PRE_REGISTRATION.md section 5.3-5.4).

Two layers, both computed from raw integer terms ONLY (never names/comments):

1. Per-sequence magnitude-robust feature vector (`sequence_features`):
   signed-log term profile, log-growth fit, finite-difference signatures,
   residue histograms mod small primes, parity/sign patterns, term ratios.

2. Pair-level classical-baseline primitives (`term_set_jaccard`,
   `ngram_jaccard`, `exact_subsequence_match`, `affine_match`): the building
   blocks of the strengthened classical baseline (exact/affine subsequence
   match + transform battery at leading-terms level). `term_set_jaccard` /
   `ngram_jaccard` are the cheap retrieval-style overlap features ("term
   overlap / string kernel" style).

Everything is pure-Python/NumPy, laptop-scale, and unit-tested on toy
sequences in tests/test_features.py. No model training happens here.
"""

from __future__ import annotations

import gzip
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

K_PROFILE = 32                  # leading terms used for profile features
SMALL_PRIMES = (2, 3, 5, 7)     # residue histograms: 2+3+5+7 = 17 dims
MIN_OVERLAP = 8                 # minimum aligned terms for subsequence/affine match


# ----------------------------------------------------------- per-sequence


def _slog(x: float) -> float:
    """Signed log1p compression: keeps sign, tames OEIS's huge magnitudes."""
    return math.copysign(math.log1p(abs(x)), x)


def sequence_features(terms: list[int], k: int = K_PROFILE) -> np.ndarray:
    """Fixed-length magnitude-robust feature vector for one sequence.

    Layout (see feature_names()):
      [0:k)        signed-log profile of first k terms (0-padded)
      [k:k+2)      n_terms (capped at 348), n_used (= min(n, k))
      [k+2:k+5)    log-growth OLS fit on |a_i|>0: slope, intercept, residual RMS
      [k+5:k+9)    diff1 slog mean/std, diff2 slog mean/std
      [k+9:k+26)   residue histograms a_i mod p, p in (2,3,5,7), normalized
      [k+26:k+31)  frac_negative, frac_zero, frac_even, sign_alternation,
                   parity_alternation
      [k+31:k+34)  median log-ratio, std log-ratio, frac_nondecreasing
    """
    n = len(terms)
    used = terms[:k]
    out = np.zeros(k + 34, dtype=np.float64)
    if n == 0:
        return out
    out[0:len(used)] = [_slog(t) for t in used]
    out[k] = min(n, 348)
    out[k + 1] = len(used)

    # log-growth OLS on positive magnitudes
    xs = [i for i, t in enumerate(used) if t != 0]
    if len(xs) >= 2:
        ys = [math.log(abs(used[i])) for i in xs]
        xarr = np.array(xs, dtype=float)
        yarr = np.array(ys, dtype=float)
        slope, intercept = np.polyfit(xarr, yarr, 1)
        resid = yarr - (slope * xarr + intercept)
        out[k + 2] = slope
        out[k + 3] = intercept
        out[k + 4] = float(np.sqrt(np.mean(resid**2)))

    # finite-difference signatures
    if len(used) >= 2:
        d1 = [_slog(b - a) for a, b in zip(used, used[1:])]
        out[k + 5] = float(np.mean(d1))
        out[k + 6] = float(np.std(d1))
        if len(used) >= 3:
            dd = [b - a for a, b in zip(used, used[1:])]
            d2 = [_slog(b - a) for a, b in zip(dd, dd[1:])]
            out[k + 7] = float(np.mean(d2))
            out[k + 8] = float(np.std(d2))

    # residue histograms mod small primes
    pos = k + 9
    for p in SMALL_PRIMES:
        for t in used:
            out[pos + (t % p)] += 1.0
        out[pos:pos + p] /= len(used)
        pos += p

    # sign / parity patterns
    out[k + 26] = sum(1 for t in used if t < 0) / len(used)
    out[k + 27] = sum(1 for t in used if t == 0) / len(used)
    out[k + 28] = sum(1 for t in used if t % 2 == 0) / len(used)
    if len(used) >= 2:
        signs = [0 if t == 0 else (1 if t > 0 else -1) for t in used]
        out[k + 29] = sum(1 for a, b in zip(signs, signs[1:]) if a * b < 0) / (len(used) - 1)
        pars = [t % 2 for t in used]
        out[k + 30] = sum(1 for a, b in zip(pars, pars[1:]) if a != b) / (len(used) - 1)

    # ratios
    ratios = [
        math.log(abs(b) / abs(a))
        for a, b in zip(used, used[1:])
        if a != 0 and b != 0
    ]
    if ratios:
        out[k + 31] = float(np.median(ratios))
        out[k + 32] = float(np.std(ratios))
    if len(used) >= 2:
        out[k + 33] = sum(1 for a, b in zip(used, used[1:]) if b >= a) / (len(used) - 1)
    return out


def feature_names(k: int = K_PROFILE) -> list[str]:
    names = [f"slog_{i}" for i in range(k)]
    names += ["n_terms", "n_used", "growth_slope", "growth_intercept", "growth_rms",
              "diff1_mean", "diff1_std", "diff2_mean", "diff2_std"]
    for p in SMALL_PRIMES:
        names += [f"mod{p}_{r}" for r in range(p)]
    names += ["frac_neg", "frac_zero", "frac_even", "sign_alt", "parity_alt",
              "logratio_med", "logratio_std", "frac_nondec"]
    return names


# ----------------------------------------------------------------- pair level


def term_set_jaccard(a: list[int], b: list[int]) -> float:
    """Cheapest overlap feature: Jaccard of the term SETS (order-free)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ngram_jaccard(a: list[int], b: list[int], n: int = 3) -> float:
    """String-kernel-style overlap: Jaccard of contiguous term n-grams."""
    ga = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    gb = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    if not ga and not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def exact_subsequence_match(a: list[int], b: list[int],
                            min_overlap: int = MIN_OVERLAP) -> bool:
    """True iff the first >=min_overlap terms of one sequence appear as a
    contiguous run anywhere in the other (offset/truncation-robust)."""
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < min_overlap:
        return False
    probe = short[:max(min_overlap, min(len(short), len(long_)))]
    m = len(probe)
    for s in range(len(long_) - min_overlap + 1):
        w = long_[s:s + m]
        if len(w) >= min_overlap and w == probe[:len(w)]:
            return True
    return False


def affine_match(a: list[int], b: list[int], max_shift: int = 8,
                 min_overlap: int = MIN_OVERLAP):
    """Search for (c, d, s) with b[i] = c*a[i+s] + d (exact, rational c) over the
    full overlap for some shift |s| <= max_shift. Returns (c, d, s) or None.

    Covers the battery's offset/shift + constant-multiple + affine relations at
    the leading-terms level. c=1, d=0 reduces to a shifted exact match.
    """
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            aa = a[s:]
            bb = b
        else:
            aa = a
            bb = b[-s:]
        m = min(len(aa), len(bb))
        if m < min_overlap:
            continue
        aa, bb = aa[:m], bb[:m]
        # find two positions with distinct a-values to solve c, d
        i0 = 0
        i1 = next((i for i in range(1, m) if aa[i] != aa[i0]), None)
        if i1 is None:  # constant a: need constant b too
            if all(x == bb[0] for x in bb):
                return (Fraction(0), Fraction(bb[0]), s)
            continue
        c = Fraction(bb[i1] - bb[i0], aa[i1] - aa[i0])
        d = Fraction(bb[i0]) - c * aa[i0]
        if c == 0:
            continue
        if all(c * x + d == y for x, y in zip(aa, bb)):
            return (c, d, s)
    return None


def pair_features(a: list[int], b: list[int]) -> dict[str, float]:
    """Bundle of the classical pair features (cheap first, then affine)."""
    aff = affine_match(a, b)
    return {
        "term_set_jaccard": term_set_jaccard(a, b),
        "ngram3_jaccard": ngram_jaccard(a, b, 3),
        "exact_subseq": float(exact_subsequence_match(a, b)),
        "affine_match": float(aff is not None),
    }


# ------------------------------------------------------------- corpus loading


def load_terms_subset(a_numbers: set[int],
                      stripped_path: Path | None = None) -> dict[int, list[int]]:
    """Stream stripped.gz and return terms ONLY for the requested A-numbers
    (keeps peak RAM tiny vs parsing all 396k sequences)."""
    from .fetch_data import DATA_DIR

    path = stripped_path or DATA_DIR / "stripped.gz"
    out: dict[int, list[int]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("A"):
                continue
            a_str, _, terms_str = line.partition(" ")
            num = int(a_str[1:])
            if num in a_numbers:
                out[num] = [int(t) for t in terms_str.strip().strip(",").split(",") if t]
                if len(out) == len(a_numbers):
                    break
    return out
