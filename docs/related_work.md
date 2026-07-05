# Related work

Machine learning on the OEIS has an established lineage. The specific task studied
here, predicting OEIS cross-references (xrefs) from raw integer terms alone, framed as a
held-out link-prediction/retrieval benchmark, is, to our knowledge, new. Each item below
was checked against its primary source.

## Learning on integer sequences

**FACT** (Belcák, Kastrati, Schenker & Wattenhofer, NeurIPS 2022 Datasets & Benchmarks,
arXiv:2209.09543) established the "OEIS as ML benchmark" genre, with organic and synthetic
abstraction-learning tasks over integer sequences. It contains no cross-reference or
link-prediction task; we build on its framing of OEIS as a substrate for benchmarking
mathematical pattern learning.

**IntSeqBERT** (Nakasho, arXiv:2603.05556) is the closest representation-learning work: a
91.5M-parameter dual-stream encoder trained on 274,705 OEIS sequences using log-magnitude
and sin/cos modulo embeddings (mod 2–101), a feature recipe convergent with ours. Its task,
however, is masked-element and next-term prediction, not inter-sequence relation prediction.
We cite it as independent validation of the input representation and include a comparable
encoder baseline; our contribution differs in task, evaluation design, and the released
benchmark artifact.

The 2016 Kaggle **Integer Sequence Learning** competition (OEIS-derived next-term
prediction) and the public Hugging Face OEIS corpora (christopher/oeis;
RenaudGaudron/oeis-sequences-benchmark; N8Programs/oeis-enhanced and oeis-massive) are all
next-term-prediction or language-model-training resources; none provides a relation/xref
benchmark. More recently, O'Malley et al. (arXiv:2411.04372) benchmark large language
models on generating Python code that computes individual OEIS sequences, sorted into easy
and hard tiers: again a per-sequence generation task, with no notion of inter-sequence
relations or cross-references.

## Automated discovery of OEIS relations

Two active systems mine OEIS for relations through program synthesis rather than learned
embeddings, and form the discovery-loop prior art for our H2 (proposing editor-missed
cross-references):

- **The Sequence Machine** (sequencedb.net): over 1.3M machine-generated sequences with
  automatically conjectured relations. Any candidate relation we propose is deduplicated
  against its conjecture set before being counted as novel.
- **LODA** (loda-lang.org): an actively maintained open-source ecosystem that mines integer
  sequence programs daily; its coverage list is a second mandatory dedup target.

The program-synthesis lineage of Gauthier & Urban (and the Alien Coding line of work) is
engaged in the same context: these systems learn *generating programs* for sequences, whereas
we test whether *relations between* sequences are recoverable from surface terms alone.

A visible 2026 wave of papers proving individual OEIS-recorded conjectures (e.g.
arXiv:2606.09913 and the cluster of short proofs of Mathar-conjecture identities in math.CO,
April–June 2026) sits adjacent: it validates community interest in machine-assisted OEIS
mathematics but involves no learning-based relation prediction.

## Positioning

To our knowledge, no prior work (i) defines a held-out link-prediction task over OEIS
cross-references, (ii) restricts inputs to raw integer terms (no names, formulas, comments,
or program text), or (iii) releases a reproducible benchmark for it. That combination is
this project's contribution. The pre-registered hypotheses and thresholds are in
[`PRE_REGISTRATION.md`](../PRE_REGISTRATION.md).
