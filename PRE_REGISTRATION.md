# PRE-REGISTRATION: OEIS Term-Only Link Prediction (LOCKED)

**Locked:** 2026-06-10 (the git commit introducing this file is the lock timestamp).
**Supersedes:** `PRE_REGISTRATION_DRAFT.md` (drafted + iterated 2026-06-10; deleted at
lock, full history in git).
**Status at lock:** no train/val/test split drawn, no embedding trained, no retrieval
metric computed, no model evaluated. The splits are drawn by `src/splits.py` only
AFTER this file is committed (the script refuses to run otherwise).

---

## 1. Research question

Can a model trained on **only the raw integer terms** of OEIS sequences (no names,
comments, formulas, or programs) rediscover the human-curated cross-reference (%Y)
graph as a held-out link-prediction task, and surface new, b-file-verifiable
candidate relations the editors missed?

## 2. Hypotheses and thresholds (locked verbatim)

- **H1 (benchmark headline).** A term-only learned embedding achieves **top-10
  recall ≥ 50%** on held-out cross-reference edges **within the term-derivable
  strata S1 and S2** (defined §5.1), versus **< 10%** for the strengthened
  classical baseline (§5.4). H1 is evaluated and reported **per stratum**; the
  unstratified aggregate is reported only as a secondary, explicitly-labeled
  number. S3/S4 results are reported descriptively and do NOT decide H1.
- **H2 (discovery, controllable).** **≥ 25** top-ranked currently-unlinked pairs
  pass verification on **full b-file data** under the frozen transform battery
  (§5.6), after dedup against Sequence Machine and LODA (§5.7). All such pairs
  are labeled **conjectural** unless a symbolic proof via known generating
  functions exists. OEIS-edit acceptance is a stretch goal, not a dependency.

**Falsification:** H1 fails if term-only top-10 recall < 50% on both S1 and S2,
or if the strengthened baseline reaches ≥ 10% on them (collapsing the claimed
gap). H2 fails if fewer than 25 deduped candidates survive b-file verification.
**A clean null on H1 is publishable** (it would say the OEIS xref graph encodes
mostly non-term-level editorial knowledge).

## 3. Data (verified, pinned snapshot 2026-06-10)

| Source | Verified fact |
|---|---|
| `stripped.gz` | 31,655,637 B; **396,449** rows; last A-number A396899; "Last Modified: June 10 04:25 UTC 2026" |
| `names.gz` | 7,499,817 B; **396,756** rows; the 307 surplus rows are all "allocated" placeholders (excluded, §6.1) |
| `oeisdata` | commit **`d42170c973069f33f1ad396f7741ea81e6b08b26`** (sync 2026-06-10T03:00:13-04:00); 396,449 `.seq` files; only `files/**` (b-files) is LFS; 214,206 b-file pointers (54.0%) |
| %Y graph | 1,473,000 raw mentions → 1,459,880 deduped directed edges → **1,175,100 undirected pairs** (1,175,043 with both endpoints in stripped.gz) |
| Terms | min 1 / median 39 / mean 44.4 / max 348 per sequence; 12,626 sequences < 8 terms |
| Keywords | `%K` on all 396,449; `dead` = 1,947 (ALL present in stripped.gz → actively filtered, §6.4) |
| License | CC-BY-SA-4.0; released benchmark/model inherit SA + OEIS attribution |

## 4. Pipeline (summary)

Parse dumps → stratify edges with the FROZEN v1 classifier (§5.1) → draw frozen
splits (§5.2) → features/encoder (§5.3) → evaluate vs baseline on held-out edges
(§5.4-5.5) → mine + verify + dedup candidates (§5.6-5.7).

## 5. Analysis plan (locked)

### 5.1 Edge stratification: classifier v1, FROZEN

Strata: **S1** duplicate/essentially-the-same · **S2** transform-of (battery §5.6)
· **S3** see-also/family · **S4** contextual/constants. H1 applies to S1+S2 only.

The classifier is the code at `src/stratify.py` as of the lock commit (regexes +
rule order are normative; unit-tested in `tests/test_stratify.py`). Rules, per
%Y mention: (1) name-based duplicate ("Duplicate of A..." naming the other
endpoint) → S1; (2) mention-attached parenthetical s1-cue → S1; (3) "all
essentially the same" cluster sentence containing the target → S1; (4) s1-cue
immediately preceding the target's A-number → S1; (5) paren s2-cue (battery
fingerprints) → S2; (6) name-based transform cue naming the other endpoint
("First differences of A...", "complement of A...", "twice A...") → S2;
(7) constant-type endpoint (name starts with decimal/continued-fraction/base
expansion, or keyword `cons` without `core`) → S4; (8) sentence-scoped s2-cue
where the target is the only A-number in its paren-stripped sentence → S2;
(9) else S3. Pair stratum = strongest label over all mentions (S1>S2>S4>S3).
Textual cues are scoped to sentences with parentheticals stripped, never
smeared across multi-target lines (v0's failure mode).

**Hand audits (completed BEFORE lock; label data only, no outcome data involved):**

| Audit | Sample | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| v0 development audit | 25/stratum, seed 42 (`results/strata_audit_sample_labeled.csv`) | 13/25 | 11/25 | 22/25 | 22/25 |
| v1 on dev sample (tuning, optimistically biased) | same 100 pairs | 12/15 | 13/14 | 44/49 | 22/22 |
| **v1 confirmatory (BLIND, fresh)** | 25/stratum, seed 43, classifier label hidden + rows shuffled (`results/strata_audit2_labeled.csv`) | **23/25 (92%)** | **19/25 (76%)** | **25/25 (100%)** | **24/25 (96%)** |

The strata are DEFINED by the frozen classifier; the audits quantify what they
contain. Known residual S2 impurities (from the blind audit): count-complements,
diagonal/array extraction, and sibling-bisections-of-a-common-parent read as
transforms. These are disclosed and the 76% S2 precision is carried into
interpretation; the classifier is NOT revised after lock (deviation policy §9).

**Frozen v1 stratum counts** (undirected pairs, both endpoints in stripped.gz;
`results/strata_v1_counts.json`): **S1 1,103 · S2 10,049 · S3 1,107,838 ·
S4 56,053**.

### 5.2 Splits (drawn post-lock by `src/splits.py`; seed 42; frozen)

- Pair table: strongest v1 label per undirected pair (u<v).
- **Eligibility:** both endpoints in stripped.gz; neither endpoint `%K dead`.
  Pairs with a <8-term endpoint stay, flagged `lt8_any`, reported separately.
- Collapse symmetric/reciprocal %Y pairs BEFORE the draw (done by the
  undirected collapse, no mirror-edge leakage).
- Per stratum (s1,s2,s3,s4 order), permute with `numpy default_rng(42)`:
  **test = 10%**, **val = 9%**, **train = 81%**. Val is for model selection /
  early stopping only; **test is not read until the pre-registered evaluation**
  (only counts + SHA256 hashes appear in the manifest).
- **Negatives:** 5 per positive per split (train, then val, then test, same
  rng stream). Universe: stripped minus dead. Constraints: u<v; **|u−v| > 20**
  (same-author/family adjacency guard); not ANY known undirected %Y pair (full
  graph, not just eligible); no duplicates within or across splits.
- Outputs + manifest (counts per split×stratum×label, eligibility funnel,
  seeds, SHA256s): `data/splits/MANIFEST.json`, committed copy
  `results/splits_manifest.json`.

### 5.3 Features / model (term-only)

Magnitude-robust features (frozen extractors in `src/features.py`): signed-log
term profile, log-growth OLS fit, finite-difference signatures, residue
histograms mod {2,3,5,7}, parity/sign patterns, term ratios. Model: small
contrastive encoder (CPU, M2/8GB) or LightGBM on pair features; FAISS/exact
top-k retrieval. **Choice + hyperparameters are made on train/val only and
frozen in writing before the test evaluation.** Cite IntSeqBERT
(arXiv:2603.05556) as convergent feature-design validation.

### 5.4 Strengthened classical baseline (frozen; no strawman)

For a query q, candidates are shortlisted by shared term 3-grams (inverted
index; OEIS-search-like semantics) plus exact/affine subsequence probes of q
**and of each battery transform of q** (Sequence-Machine-style battery at the
leading-terms level, battery = §5.6, offsets |s| ≤ 8, ≥ 8 aligned terms).
Shortlist ranked by: battery/affine match (binary) → 3-gram Jaccard → term-set
Jaccard. Primitives live in `src/features.py`.

### 5.5 Metrics

Primary: **top-10 recall per stratum** on held-out edges, a held-out edge
(u,v) counts as recovered if v ranks in u's top-10 or u in v's top-10 over the
full eligible-node candidate set (excluding only u itself; train edges are NOT
masked out of the ranking). Secondary: top-1/top-100 recall, MRR, unstratified
aggregate (labeled as such), lt8 edges reported separately. Uncertainty: 95%
bootstrap CIs (1,000 resamples over held-out edges). **Any stratum with < 100
held-out edges is flagged underpowered.** Expected held-out counts: S1 ≈ 110,
S2 ≈ 1,004 (not triggered).

### 5.6 H2 verification: frozen transform battery

A candidate pair passes only if one of these relations holds EXACTLY on **all
overlapping b-file terms** (typically hundreds; SymPy/PARI exact arithmetic),
with offset shifts |s| ≤ 8 allowed for each:

1. identity (up to offset/truncation); 2. negation / signed version;
3. affine: b = c·a + d (rational c ≠ 0, includes constant multiples);
4. first differences / partial sums; 5. second differences;
6. bisection (even/odd) and decimation a(kn+φ), k ≤ 3;
7. binomial transform and inverse; 8. Euler transform and inverse;
9. Möbius (Dirichlet) transform and inverse; 10. set complement (for
monotone increasing sequences read as sets).

No compositions beyond [one transform] ∘ [offset]. Pairs lacking a b-file on
either side are reported separately and labeled weaker. Everything is labeled
**conjectural** absent a generating-function proof.

### 5.7 Dedup (mandatory; built Stage 3)

- **LODA** (loda-programs commit `f16d13a435ed`, fetched 2026-06-10):
  **152,917** mined programs; **61,400** directed `seq`-call relations;
  **59,717 undirected known-relation pairs** (`data/dedup/loda_relations.parquet`,
  regenerable via `python -m src.dedup loda`).
- **Sequence Machine** (sequencedb.net; data mirror
  `jonmaiga/sequence-machine-data`, ~690 MB > disk budget): **per-candidate**
  check, fetch each endpoint's `*.programs.json`, scan for mentions of the
  other endpoint (verified live 2026-06-10).
- A candidate pair is "already known" (dropped from the H2 count, logged) if it
  is in the LODA pair set, or either endpoint's SM programs mention the other,
  or the relation is in the pair's OEIS entries already.

## 6. Exclusions (locked; every exclusion logged with counts)

1. The 307 "allocated" placeholder A-numbers, excluded everywhere.
2. Sequences with < 8 terms (12,626), excluded from encoder training; retained
   as retrieval targets; their held-out edges reported separately.
3. S3/S4 edges, excluded from H1's headline; reported descriptively.
4. `%K dead` sequences (1,947, all present in stripped.gz), actively filtered
   from the corpus and from split eligibility; name-based "Duplicate of"
   evidence still feeds S1 via the classifier.
5. Any edge whose endpoint lacks a stripped.gz data line.

## 7. Power / feasibility

396,449 sequences; ~1.175M undirected pairs. Test holdout ≈ 117.5k pairs
overall; S1 ≈ 110 held-out edges → 95% CI half-width ≤ ~9.5 pp (adequate to
separate 50% from 10%); S2 ≈ 1,004 → ≤ ~3.1 pp. Laptop-scale throughout; $0
spend; b-file fetches at Crawl-Delay 10 (~35 min for ~200 fetches).

## 8. Peek status at lock (what has and has not been observed)

Observed (label data + aggregates only): bulk-dump row counts and term-length
stats; the %Y edge list with degree/symmetry/stratum AGGREGATES; two hand
audits of 100 edge classifications each; LODA/SM dedup-set sizes. **Not
observed:** any train/val/test split contents, any embedding, any retrieval or
ranking metric, any feature value on any %Y pair. H1/H2 thresholds are
unchanged from the pre-data draft. Prior-art survey as of 2026-06-10
(`docs/related_work.md`): no prior work with the link-prediction benchmark
framing was found.

## 9. Deviation policy

Any post-lock change is recorded in a `DEVIATIONS.md` with date and reason, and
affected analyses are labeled exploratory. This file is never edited in place.

## 10. Lock checklist: all complete

- [x] oeisdata cloned + SHA pinned (`data/SNAPSHOT.txt`).
- [x] %Y edges + per-stratum counts measured (`results/data_stats.json`,
      `results/strata_v1_counts.json`).
- [x] Stratification heuristics FROZEN (v1, `src/stratify.py`) after a labeled
      100-edge development audit; precision CONFIRMED on a fresh blind
      100-edge sample (seed 43).
- [x] Baseline + transform battery frozen (§5.4/§5.6).
- [x] `%K` keyword availability confirmed; exclusion rules locked.
- [x] Dedup exclusion sets built/verified (LODA full; SM per-candidate).
- [x] This file committed BEFORE `src/splits.py` draws any split; no model
      evaluation before that commit (verified: none exists).
