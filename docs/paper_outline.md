# Paper outline (skeleton; no results yet)

**Working title:** *Term-Only Link Prediction on the OEIS Cross-Reference Graph:
a Pre-Registered Benchmark*

**Target:** arXiv `cs.LG` (primary), `math.CO` (cross-list). Venue candidates:
MATH-AI-style workshop; *Experimental Mathematics*; *INTEGERS*.

**Status:** outline only. The study is pre-registered
([`../PRE_REGISTRATION.md`](../PRE_REGISTRATION.md), locked 2026-06-10) and the
benchmark-construction pipeline is built; **no model has been trained and no
held-out metric has been computed**. Every value in §7 is a placeholder.

---

## Abstract (skeleton)

The OEIS cross-links related integer sequences through hand-curated
cross-references. We ask whether those links are recoverable from the raw
integer terms alone, with no names, comments, formulas, or programs. We release the
first held-out link-prediction benchmark over the OEIS cross-reference graph,
with edges stratified by relation type (duplicate, transform, family,
contextual) under a frozen, blind-audited classifier, and pre-registered
hypotheses, thresholds, and analysis plan committed before any split was drawn.
We compare a term-only learned embedding against a strengthened classical
baseline (search-style n-gram retrieval plus an exact/affine subsequence and
transform battery). [Results: TBD.] We additionally mine top-ranked unlinked
pairs, verify candidate relations exactly on full b-file data, deduplicate them
against existing machine-discovery projects, and propose surviving candidates
as OEIS edits. [Discovery results: TBD.]

## 1. Introduction

- The OEIS as a hand-curated knowledge graph: ~396k sequences, ~1.18M
  undirected cross-reference pairs (pinned 2026-06-10 snapshot).
- The question: how much of that editorial graph is encoded in the terms
  themselves? Framing as held-out link prediction makes it quantitative.
- Why stratification is essential: cross-references are heterogeneous, and many
  are not term-derivable even in principle (e.g. a sequence linked to the
  decimal expansion of an associated constant). An unstratified recall number
  is either unreachable or gameable.
- Contributions:
  1. A reusable, pre-registered link-prediction / retrieval benchmark over the
     OEIS cross-reference graph (frozen splits, stratified edges, audited
     strata, manifest with hashes).
  2. A term-only embedding evaluated against a deliberately strengthened
     classical baseline, with per-stratum, pre-registered decision rules.
  3. A discovery pipeline: candidate unlinked pairs verified exactly on b-file
     data under a frozen transform battery, deduplicated against prior
     machine-discovery projects, and submitted as OEIS edits.
- Note the two-sided design: under the pre-registration, a clean negative on
  H1 is itself an informative, publishable result about how much
  non-term-level editorial knowledge the cross-reference graph encodes.

## 2. Related work

(Engage each; see [`related_work.md`](related_work.md) for the verified survey.)

- **OEIS as an ML benchmark:** FACT (Belcák et al., NeurIPS 2022 D&B,
  arXiv:2209.09543), abstraction/next-term tasks; the 2016 Kaggle integer
  sequence competition; Hugging Face OEIS corpora (next-term / LM training).
  None frames the cross-reference graph as a held-out link-prediction task.
- **Sequence representation learning:** IntSeqBERT (arXiv:2603.05556),
  modulo-spectrum + log-magnitude embeddings for masked-element / next-term
  prediction; cite as convergent validation of magnitude-robust,
  residue-based feature design; our task (retrieval over the xref graph)
  differs.
- **Machine discovery over OEIS:** the Sequence Machine (sequencedb.net) and
  LODA (loda-lang.org) auto-conjecture relations and mine programs at scale;
  Gauthier & Urban's program-synthesis line. These motivate the mandatory
  dedup step (§5.7 of the pre-registration) for every discovery claim.
- **Link prediction generally:** classical graph link-prediction literature;
  our setting differs in that node features (the terms) are the only input,
  and graph structure is never used as a feature.

## 3. Data

(Method source: PRE_REGISTRATION.md §3, §6; `src/build_graph.py`,
`src/fetch_data.py`.)

- Pinned snapshot 2026-06-10: `stripped.gz` (396,449 sequences' leading terms;
  median 39 terms), `names.gz`, and the `oeisdata` git mirror at a pinned
  commit; `%Y` lines give the cross-reference graph (~1.175M undirected pairs
  with both endpoints in the term corpus).
- License: CC-BY-SA-4.0 (the released benchmark inherits share-alike + OEIS
  attribution).
- Locked exclusions, each logged with counts: "allocated" placeholder
  A-numbers; `%K dead` sequences; edges with an endpoint lacking a data line;
  sequences with < 8 terms are excluded from encoder training but retained as
  retrieval targets and reported separately.

## 4. Benchmark construction

### 4.1 Edge stratification (frozen classifier v1)

(Method source: PRE_REGISTRATION.md §5.1; `src/stratify.py`;
`tests/test_stratify.py`.)

- Four strata: **S1** duplicate / essentially-the-same; **S2** transform-of
  (the frozen battery's relation types); **S3** see-also / family; **S4**
  contextual / constants. The headline hypothesis applies to S1+S2 only.
- Classifier = sentence-scoped regex rules over `%Y` mention text and entry
  names (rule order normative, unit-tested); pair stratum = strongest label
  over all mentions.
- Label quality quantified by hand audits completed before lock, including a
  blind confirmatory audit on a fresh sample (per-stratum precision reported;
  known residual S2 impurities disclosed and carried into interpretation).
- The strata are *defined* by the frozen classifier; it is not revised
  post-lock (deviation policy: PRE_REGISTRATION.md §9).

### 4.2 Splits and negatives

(Method source: PRE_REGISTRATION.md §5.2; `src/splits.py`.)

- Undirected pair table (u<v), symmetric/reciprocal references collapsed
  before the draw (no mirror-edge leakage).
- Per stratum, seeded permutation (numpy `default_rng(42)`): test 10%, val 9%,
  train 81%. Val is for model selection only; the test split is not read until
  the pre-registered evaluation.
- Negatives: 5 per positive per split; constraints: u<v, |u−v| > 20
  (same-author/family adjacency guard), not any known cross-reference pair in
  the *full* graph, no duplicates within or across splits.
- A committed manifest records counts per split × stratum × label, the
  eligibility funnel, seeds, and SHA256 hashes of the split files.

## 5. Methods

### 5.1 Term-only features

(Method source: PRE_REGISTRATION.md §5.3; `src/features.py`.)

Magnitude-robust per-sequence features computed from raw terms only:
signed-log term profile, log-growth OLS fit, first/second finite-difference
signatures, residue histograms mod {2,3,5,7}, sign/parity patterns, and
log-ratio statistics.

### 5.2 Learned model

Small contrastive encoder (CPU-scale) or gradient-boosted trees on pair
features, with exact / approximate nearest-neighbour retrieval. Architecture
and hyperparameters are chosen on train/val only and frozen in writing before
the single test evaluation (PRE_REGISTRATION.md §5.3).

### 5.3 Strengthened classical baseline

(Method source: PRE_REGISTRATION.md §5.4; `src/features.py` primitives.)

Search-style candidate shortlisting by shared term 3-grams (inverted index),
plus exact/affine subsequence probes of the query *and of each battery
transform of the query* at the leading-terms level; shortlist ranked by
battery/affine match, then 3-gram Jaccard, then term-set Jaccard. The baseline
is deliberately strengthened beyond exact subsequence match so the comparison
is not a strawman.

## 6. Evaluation protocol

(Method source: PRE_REGISTRATION.md §2, §5.5.)

- Primary metric: **top-10 recall per stratum** on held-out edges (an edge
  counts as recovered if either endpoint retrieves the other in its top 10
  over the full eligible candidate set; training edges are not masked out).
- Secondary: top-1 / top-100 recall, MRR, the unstratified aggregate (labeled
  as secondary), short-sequence (< 8 terms) edges reported separately.
- Uncertainty: 95% bootstrap confidence intervals (1,000 resamples over
  held-out edges); any stratum with < 100 held-out edges flagged underpowered.
- Pre-registered decision rules (H1 retrieval thresholds per stratum, H2
  verified-discovery count) are stated verbatim in PRE_REGISTRATION.md §2 and
  are not restated with numbers here to keep this outline metric-free.

## 7. Results: ALL PLACEHOLDERS (nothing computed yet)

### 7.1 H1: per-stratum held-out retrieval

| Stratum | Method | Top-10 recall [95% CI] | Top-1 | Top-100 | MRR |
|---|---|---|---|---|---|
| S1 | term-only embedding | TBD | TBD | TBD | TBD |
| S1 | classical baseline | TBD | TBD | TBD | TBD |
| S2 | term-only embedding | TBD | TBD | TBD | TBD |
| S2 | classical baseline | TBD | TBD | TBD | TBD |

- H1 decision per the pre-registered rule: **TBD**.
- Unstratified aggregate (secondary, labeled as such): TBD.
- Short-sequence (`lt8`) edges, reported separately: TBD.

### 7.2 Descriptive results on S3/S4 (do not decide H1)

- TBD.

### 7.3 H2: discovery funnel

| Stage | Count |
|---|---|
| Top-ranked currently-unlinked pairs mined | TBD |
| After dedup (LODA, Sequence Machine, existing entries) | TBD |
| Pass exact b-file verification (frozen battery) | TBD |
| Lacking b-file on either side (reported separately, weaker) | TBD |

- H2 decision per the pre-registered rule: **TBD**.
- Example verified (conjectural) relations: TBD.

### 7.4 Error analysis

- Which relation types the embedding recovers vs misses; failure cases; effect
  of sequence length and growth rate. TBD.

## 8. Discussion

- What the result (positive or null) says about how much of the OEIS editorial
  graph is term-derivable.
- The benchmark as the reusable artifact: frozen splits + manifest enable
  future encoders to be compared on equal footing.
- Relation to next-term/masked-prediction benchmarks: complementary axes of
  "understanding" an integer sequence.

## 9. Limitations (anticipated reviewer objections)

1. **Heuristic strata.** The strata are defined by a frozen regex classifier,
   not ground truth; the blind audit quantifies precision, and S2 in
   particular carries disclosed impurities (count-complements, array
   extractions, sibling bisections). Conclusions are conditioned on the
   audited label quality.
2. **Open-world negatives.** Unlinked pairs are unlabeled, not guaranteed
   unrelated; some sampled negatives may be true (missing) relations. This
   deflates measured precision and is partially what H2 exploits.
3. **Leakage risk in negatives.** The |u−v| > 20 adjacency guard blocks the
   most obvious same-author/family leakage but cannot rule out all of it.
4. **Baseline strength.** The classical baseline, though strengthened (search
   semantics + transform battery), operates at the leading-terms level; a
   reviewer may argue a still-stronger non-learned system (e.g. a
   full-b-file or Superseeker-class pipeline) belongs in the comparison.
5. **Leading-terms ceiling.** Both systems see only the leading terms
   (median ~39); relations that manifest only deep in the sequence are
   invisible to both, capping attainable recall in ways the strata only
   partially capture.
6. **Verification is not proof.** Exact agreement on all overlapping b-file
   terms under the frozen battery remains conjectural; every surviving
   candidate is labeled as such absent a generating-function proof.
7. **Closed battery.** The 10-transform battery (no compositions beyond one
   transform plus offset) misses relation types outside it; H2 counts are a
   lower bound on discoverable relations but also battery-relative.
8. **Single snapshot of an evolving corpus.** The graph and dedup sets are
   pinned to 2026-06-10; the OEIS and the machine-discovery projects grow
   daily, so absolute counts are snapshot-relative.
9. **Editorial ground truth.** The cross-reference graph itself is incomplete
   and stylistically inconsistent across editors; "recall of %Y" is recall of
   a curated, imperfect target, not of mathematical relatedness.
10. **External dependency for edit acceptance.** OEIS review latency and
    draft limits are outside our control, which is why edit *acceptance* is a
    stretch goal rather than a decision criterion (see §10).

## 10. Artifacts and deliverables

- **Benchmark release:** frozen train/val/test splits + manifest (counts,
  seeds, SHA256), stratifier code, and feature extractors, all CC-BY-SA-4.0 with
  OEIS attribution; archived with a DOI.
- **Code + model:** full pipeline, pinned dependencies, single reproduction
  path; trained encoder weights.
- **Candidate relations:** ranked CSV of verified, deduplicated, conjectural
  relations with the exact verifying transform per pair.
- **OEIS edits, an explicit deliverable.** Verified candidates are submitted
  as OEIS edits (duplicate merges prioritized; small batches respecting
  editorial limits), and the paper ships a public log of submitted and
  accepted edits. Accepted edits are concrete, externally validated
  contributions of the method, reported as outcomes even though acceptance is
  not a pre-registered decision criterion.

## Reproducibility statement

Pre-registration committed before any split was drawn (`src/splits.py` refuses
to run without it); seeds fixed; data snapshot pinned by date and commit SHA;
split manifest carries SHA256 hashes; pinned `requirements.txt`; all
exclusions logged with counts; deviations, if any, recorded in a dated
`DEVIATIONS.md` and labeled exploratory.

## References (to be completed at writeup)

FACT (arXiv:2209.09543) · IntSeqBERT (arXiv:2603.05556) · Sequence Machine ·
LODA · Gauthier & Urban / Alien Coding (pin exact IDs) · Kaggle Integer
Sequence Learning (2016) · OEIS (oeis.org; CC-BY-SA-4.0) · classical
link-prediction references.
