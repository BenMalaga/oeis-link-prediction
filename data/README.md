# Data: fetch, don't commit

Raw data is **not** committed. Sources and pinned facts: [`../PRE_REGISTRATION.md`](../PRE_REGISTRATION.md) §3.
Canonical fetch path: `python -m src.fetch_data --help` (idempotent; wraps everything below).
Stage-1 one-shot (fetch + clone + edge list + stats): `python -m src.build_graph`.

**Pinned snapshot for this project: 2026-06-10.** stripped.gz/names.gz are regenerated
daily, so all results must reference this snapshot date and the row counts below.
Record the `oeisdata` commit SHA at clone time in `data/SNAPSHOT.txt` (the repo syncs
daily from OEIS; `time.txt` inside it carries the sync timestamp).

**Verified live 2026-06-10** (HEAD + full download + row counts; re-verified by local
gzcat row-count on the downloaded files):

```bash
# Bulk terms + names (regenerated daily ~03:00-05:00 UTC; ~31.7 MB + ~7.5 MB).
# Default curl UA worked on 2026-06-10; keep a browser UA for robustness against
# intermittent Cloudflare filtering.
curl -A "Mozilla/5.0" https://oeis.org/stripped.gz -o data/stripped.gz   # 31,655,637 B; 396,449 seq rows; last A-number A396899
curl -A "Mozilla/5.0" https://oeis.org/names.gz   -o data/names.gz       # 7,499,817 B; 396,756 name rows (307 extra = "allocated" placeholders, join on A-number)

# Full sequence pages incl. %Y cross-references (ground truth). seq/ is plain text in
# 397 subdirs (seq/A000/A000045.seq … seq/A396/); only files/** (b-files) is Git-LFS.
# DONE 2026-06-10 (Stage 1): cloned with --no-checkout, since the pack alone is ~341 MB,
# while a full checkout would materialize ~400k tiny .seq files + ~600k LFS pointer
# stubs (>1.5 GB of 4 KB APFS blocks). All parsing streams blobs from the pack via
# `git archive` (src/build_graph.py); no working tree is ever created:
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --no-checkout https://github.com/oeis/oeisdata data/oeisdata

# b-files for candidate verification: fetch INDIVIDUALLY from oeis.org (no LFS needed),
# honoring robots.txt Crawl-Delay: 10 (one request per 10 s):
curl -A "Mozilla/5.0" https://oeis.org/A000045/b000045.txt -o data/bfiles/b000045.txt

# JSON API (spot checks only; robots.txt disallows /search for crawlers; NOT a bulk path):
#   https://oeis.org/search?q=id:A000045&fmt=json  → bare JSON array, max 10 results/page, &start=N paginates, no count field
#   https://oeis.org/A000045?fmt=json              → single sequence object (includes "xref" field)
```

## Verified snapshot facts (2026-06-10)

| File | Bytes | Rows | Notes |
|---|---|---|---|
| `stripped.gz` | 31,655,637 | 396,449 sequence rows | header `Last Modified: June 10 04:25 UTC 2026`; last A-number A396899 |
| `names.gz` | 7,499,817 | 396,756 name rows | 307 more than stripped.gz; **measured 2026-06-10: all 307 are "allocated for &lt;editor&gt;" placeholders** (reserved, unfilled A-numbers) → exclude. The duplicate-merge signal lives in the **1,262 live** sequences whose name contains "duplicate of". Join on A-number. |
| `oeisdata` repo | 341 MB (pack only, `--no-checkout`, LFS skipped) | **396,449 `.seq` files** (= stripped.gz rows; the 307 placeholders have no `.seq`) | **cloned 2026-06-10, commit `d42170c973069f33f1ad396f7741ea81e6b08b26`**, upstream sync 2026-06-10T03:00:13-04:00 (see `SNAPSHOT.txt`); `.gitattributes` confirms only `files/**` (b-files) is LFS; tree carries **214,206 b-file pointers** (54.0% of sequences) |
| b-file `b000045.txt` | 429,385 | n/a | direct HTTP 200 from oeis.org, no LFS |

## Stage-1 ingest outputs (derived, regenerable, NOT committed)

Built by `python -m src.build_graph` (idempotent; `--force` to rebuild) on the
2026-06-10 snapshot. Parsing streams `.seq` blobs straight from the git pack
(`git archive`); no working tree is ever checked out.

| File | Size | Rows | Contents |
|---|---|---|---|
| `xrefs.parquet` | ~24 MB | 1,473,000 | one row per %Y target mention: `source`, `target`, `y_line_idx`, `link_type`, `paren` (target-attributed parenthetical), draft `stratum` (v0), `raw_text` |
| `seq_meta.parquet` | ~2.7 MB | 396,449 | per sequence: `%K` keyword string, number of %Y lines |

Headline counts (full table: `../results/data_stats.json` + `.md`): 1,473,000 raw
%Y mentions (5,714 self-mentions dropped) → 1,459,880 deduped directed edges →
1,175,100 undirected pairs (284,780 symmetric, 890,320 one-directional); 1,175,043
pairs have both endpoints in stripped.gz. Draft-v0 strata (undirected, both in
stripped): S1 1,914 · S2 15,483 · S3 1,098,252 · S4 59,394. `%K` present for all
396,449 entries; all 1,947 `dead`-keyword sequences appear in stripped.gz with
terms (must be actively excluded). QA: A000045's parquet targets exactly match
both a direct re-parse of its `.seq` blob and the live oeis.org JSON API (89
distinct targets, verified 2026-06-10).

**Integrity note:** the %Y graph is *label* data. Stage 1 built it and reported
aggregates only, with no train/test split, no model, no retrieval metric until
`PRE_REGISTRATION.md` is locked and committed.

## Stage-3 derived outputs (regenerable, NOT committed)

| File | Built by | Contents |
|---|---|---|
| `xrefs_v1.parquet` (~24 MB) | `python -m src.stratify` | mention table + FROZEN v1 `stratum_v1` column (classifier locked in `PRE_REGISTRATION.md` §5.1) |
| `splits/{train,val,test}.parquet` (~56 MB) | `python -m src.splits` (refuses to run pre-lock; drawn at lock commit `7ce9324`, seed 42) | pre-registered splits: 951,155 / 105,681 / 117,424 positive pairs + 5:1 negatives; counts + SHA256s in `splits/MANIFEST.json` (committed copy: `../results/splits_manifest.json`) |
| `dedup/loda_relations.parquet` + `loda_anumbers.txt` | `python -m src.dedup loda` | LODA exclusion set (see Dedup targets below) |

**Test split discipline:** `data/splits/test.parquet` is written but its contents
are not read by any analysis until the pre-registered evaluation; only counts +
hashes are surfaced. Smoke checks (`python -m src.smoke`) read TRAIN only.

The OEIS Download wiki page (approved rev. 2025-09-16) lists exactly three bulk options:
stripped.gz, names.gz, and the oeisdata repo. **There is no other bulk dump**; the JSON
API at robots.txt-polite pace (Crawl-Delay 10, Disallow /search) would take ~46 days for
396k sequences, so never use it for bulk.

## Alternative ingest (optional, do not pin to it)

`christopher/oeis` on Hugging Face (parquet full dump, CC-BY-SA-4.0, updated 2026-01-19)
could simplify ingest, but pin results to our own stripped.gz/oeisdata snapshot for
reproducibility.

## Dedup targets (mandatory before any H2 claim/submission)

- Sequence Machine: https://sequencedb.net/ (1,328,459 machine-generated sequences with
  auto-conjectured relations, live 2026-06-10)
- LODA: https://github.com/loda-lang/loda-programs (mining active daily; last push
  2026-06-09; programs Apache-2.0)

**Built 2026-06-10 (Stage 3, `python -m src.dedup`):**

- **LODA exclusion set** (`data/dedup/loda_relations.parquet` + `loda_anumbers.txt`,
  regenerable): shallow `--no-checkout` clone of loda-programs (commit
  `f16d13a435ed`, ~60 MB pack, **deleted after extraction** for the disk budget),
  all `seq`-opcode calls extracted from streamed `.asm` blobs →
  **152,917 mined programs; 61,400 directed seq-call relations; 59,717 undirected
  known-relation pairs**. Snapshot in `data/dedup/LODA_SNAPSHOT.txt`.
- **Sequence Machine**: full data mirror `github.com/jonmaiga/sequence-machine-data`
  is ~690 MB (> this project's disk budget), so the exclusion check is
  **per-candidate**: `python -m src.dedup sm A000045 ...` fetches each candidate
  endpoint's `*.programs.json` from raw.githubusercontent.com and scans for
  mentions of the other endpoint (verified live 2026-06-10; fetched files cached
  under `data/dedup/sequence-machine/`). Exact for the ≤~100-pair H2 candidate
  set; a pair is "already known" if either endpoint's SM programs mention the
  other, or the pair is in the LODA set.

## License

**CC-BY-SA-4.0** (EULA rev. approved 2023-02-24, primary source
https://oeis.org/wiki/The_OEIS_End-User_License_Agreement (note the wiki 403s generic
fetchers; use a browser UA); restated on the Download wiki page approved 2025-09-16 and in
oeisdata/LICENSE). Attribution must credit "The On-Line Encyclopedia of Integer Sequences"
with a URL to https://oeis.org/ or the specific sequence. The released benchmark/model
inherits SA + attribution. Non-share-alike uses only by special arrangement with the OEIS
Foundation. (The old "CC BY-NC?" concern is stale, that was the pre-2018 license;
commercial use and derivatives are permitted.)
