"""Unit tests for the frozen v1 stratifier on synthetic %Y lines.

Each case replicates a failure mode found (or a behavior confirmed) in the
100-edge development audit of 2026-06-10.
"""

import pytest

from src.stratify import StratifierV1, split_sentences, strip_parens

NAMES = {
    1: "Tribonacci numbers.",
    2: "First differences of A000213, also twice A000001.",
    3: "Numbers whose runs-resistance is maximal.",
    4: "Decimal expansion of 4 - Pi.",
    5: "Positions of 0 in A000007; complement of A000006.",
    6: "Positions of 1 in A000007.",
    9: "Duplicate of A000003.",
}
CONST = {4}


@pytest.fixture
def clf():
    return StratifierV1(NAMES, CONST)


def one(clf, source, raw, target, paren=""):
    return clf.classify_line(source, raw, [(target, paren)])[0]


# ---------------------------------------------------------------- s1 rules


def test_paren_attributed_s1(clf):
    assert one(clf, 10, "Cf. A000020 (essentially the same).", 20,
               "essentially the same") == "s1"


def test_direct_adjacency_s1(clf):
    raw = "Cf. A000011, A000012; essentially the same as A000013."
    assert one(clf, 10, raw, 13) == "s1"
    # ... but the OTHER targets on the line must NOT inherit the cue
    assert one(clf, 10, raw, 11) == "s3"


def test_essentially_same_other_sentence_not_smeared(clf):
    raw = "Cf. A000030, A000031. Essentially the same as A000096."
    assert one(clf, 10, raw, 30) == "s3"
    assert one(clf, 10, raw, 31) == "s3"
    assert one(clf, 10, raw, 96) == "s1"


def test_row_sums_of_is_not_s1(clf):
    raw = "Essentially the same as row sums of A000128."
    assert one(clf, 10, raw, 128) == "s3"


def test_cluster_sentence_s1(clf):
    raw = ("The following sequences are all essentially the same, with A000031 "
           "as the parent: A000031, A000032 (s(n)-1), A000033 (s(n)-2). "
           "The first differences of A000031 essentially matches A000061.")
    assert one(clf, 31, raw, 32) == "s1"
    # target named only in the later (first differences) sentence: NOT s1
    assert one(clf, 31, raw, 61) == "s2"  # only target in that sentence + diff cue


def test_dup_name_s1(clf):
    assert one(clf, 9, "Cf. A000003.", 3) == "s1"
    assert one(clf, 3, "Cf. A000009.", 9) == "s1"  # symmetric


# ---------------------------------------------------------------- s2 rules


def test_paren_attributed_s2(clf):
    assert one(clf, 10, "Cf. A000022 (complement).", 22, "complement") == "s2"


def test_focused_sentence_s2(clf):
    assert one(clf, 10, "Partial sums: A000023.", 23) == "s2"
    assert one(clf, 10, "First differences are in A000024.", 24) == "s2"


def test_third_party_complement_not_smeared(clf):
    # "X is A1, complement A2": the cue binds A1<->A2, NOT source<->A2.
    raw = "The case of partitions is A000041, complement A000042."
    assert one(clf, 10, raw, 41) == "s3"
    assert one(clf, 10, raw, 42) == "s3"


def test_cue_inside_paren_does_not_leak_to_sentence(clf):
    # paren belongs to A000051 only; A000050 must not become s2
    raw = "Cf. A000050, A000051 (partial sums)."
    assert one(clf, 10, raw, 50) == "s3"
    assert one(clf, 10, raw, 51, paren="partial sums") == "s2"


def test_name_based_transform_cue(clf):
    # NAMES[2] says "First differences of A000213, also twice A000001"
    assert one(clf, 2, "Cf. A000001.", 1) == "s2"
    assert one(clf, 1, "Cf. A000002.", 2) == "s2"  # symmetric
    assert one(clf, 5, "Cf. A000006.", 6) == "s2"  # "complement of A000006"


# ---------------------------------------------------------------- s4 / s3


def test_const_endpoint_s4(clf):
    assert one(clf, 10, "Cf. A000004.", 4) == "s4"


def test_s1_beats_s4(clf):
    raw = "Essentially the same as A000004."
    assert one(clf, 10, raw, 4) == "s1"


def test_plain_cf_is_s3(clf):
    assert one(clf, 10, "Cf. A000077, A000078.", 77) == "s3"


# ------------------------------------------------------------- text helpers


def test_split_sentences():
    s = split_sentences("Partial sums give A000001. a(n) = A000002(n, 0).")
    assert len(s) == 2


def test_strip_parens_nested():
    assert "x" not in strip_parens("a (x (y) z) b")
