# OEIS xref-graph descriptive stats (Stage 1 ingest)

Snapshot **2026-06-10**, oeisdata commit `d42170c97306`. Aggregates only, no splits drawn, no models run (pre-reg not yet locked).

| Quantity | Value |
|---|---|
| Sequences in stripped.gz | 396,449 |
| Rows in names.gz | 396,756 |
| .seq files in oeisdata | 396,449 |
| Raw %Y target mentions | 1,473,000 |
| Deduped directed edges | 1,459,880 |
| Undirected pairs | 1,175,100 |
| ...symmetric (both directions) | 284,780 |
| ...asymmetric (one direction) | 890,320 |
| Undirected pairs, both endpoints in stripped.gz | 1,175,043 |
| b-files in oeisdata tree | 214,206 (54.0% of stripped) |

## Draft strata (v0 heuristics; frozen + audited only at pre-reg lock)

| Stratum | Deduped directed | Undirected | Undirected, both in stripped |
|---|---|---|---|
| s1 | 2,529 | 1,914 | 1,914 |
| s2 | 17,953 | 15,483 | 15,483 |
| s3 | 1,364,412 | 1,098,308 | 1,098,252 |
| s4 | 74,986 | 59,395 | 59,394 |

## Degrees (deduped directed graph; zeros included over all stripped.gz)

| Degree | mean | median | p90 | p99 | max |
|---|---|---|---|---|---|
| out | 3.682 | 2 | 8 | 25 | 127 |
| in | 3.682 | 1 | 8 | 34 | 10,089 |
| undirected (stripped-only edges) | 5.928 | 3 | 12 | 45 | 10,100 |

Isolated sequences (no stripped-to-stripped xref at all): 67,379
