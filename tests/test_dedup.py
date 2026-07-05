"""Unit tests for dedup parsers (no network)."""

from src.dedup import A_REF_RE, SEQ_OP_RE

LODA_ASM = """\
; A007318: Pascal's triangle read by rows.
; Submitted by miner
; 1,1,1,1,2,1

mov $2,$0
seq $2,45
lpb $0
  sub $0,1
  seq $0, 000142
lpe
mul $1,$2
"""


def test_seq_op_extraction():
    calls = [int(m) for m in SEQ_OP_RE.findall(LODA_ASM)]
    assert calls == [45, 142]


def test_seq_op_ignores_comments_and_other_ops():
    text = "; seq $1,999 in a comment is still matched only at line start\nmov $1,5\n"
    # the regex is anchored at line start (^\s*seq) so the comment must NOT match
    assert SEQ_OP_RE.findall(text) == []


def test_a_ref_extraction_from_sm_json():
    blob = '{"ix": "a(n) = A000045(n) + A000032(n-1)", "px": ["A000142"]}'
    assert {int(m) for m in A_REF_RE.findall(blob)} == {45, 32, 142}
