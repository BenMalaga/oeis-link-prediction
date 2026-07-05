"""Edge stratification, classifier v1 (the version FROZEN at pre-registration lock).

    python -m src.stratify            # xrefs.parquet -> data/xrefs_v1.parquet + counts
    python -m src.stratify --force

Strata (PRE_REGISTRATION.md section 5.1):
    s1  duplicate-of / essentially-the-same      (term-derivable; H1 headline)
    s2  transform-of (battery, section 5.4/5.6)  (term-derivable; H1 headline)
    s3  see-also / same-family                   (partially derivable; reported)
    s4  contextual / constants                   (presumed NOT term-derivable)

Why v1 (what the 100-edge development audit of v0 found, 2026-06-10):
  v0 matched cue words at LINE level, so "Essentially the same as A096365." smeared
  s1 onto every other target on the line (v0 s1 precision 13/25), and "X is A1,
  complement A2" smeared s2 onto pairs that are not transforms of the source
  (v0 s2 precision 11/25). v1 scopes every textual cue to (a) the mention's own
  parenthetical, (b) the SENTENCE containing the mention, with parentheticals
  stripped, and only when the mention is the ONLY A-number in that sentence,
  or (c) an explicit cue-immediately-before-target adjacency / "all essentially
  the same" cluster sentence. It also adds NAME-based transform cues ("First
  differences of A000213", "complement of A284676", "twice A000073") and stops
  treating `core` sequences (A000012, A000035, ...) as constants.

These rules are FROZEN as of the lock; any later change is a logged deviation.

Like all of Stage 1-3 this is LABEL-data processing only: no split is read here,
no model is trained, no retrieval metric is computed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

from .build_graph import (
    SEQ_META_PARQUET,
    XREFS_PARQUET,
    parse_names,
)
from .fetch_data import DATA_DIR, PROJECT_ROOT, log as _log

XREFS_V1_PARQUET = DATA_DIR / "xrefs_v1.parquet"
COUNTS_JSON = PROJECT_ROOT / "results" / "strata_v1_counts.json"

RANK = {"s1": 0, "s2": 1, "s4": 2, "s3": 3}  # "strongest label wins" order (frozen)

# ------------------------------------------------------------- frozen v1 regexes

# s1 cues.
S1_CUE_RE = re.compile(
    r"duplicate of|essentially the same|essentially a duplicate", re.I
)
# "The following sequences are all essentially the same ... A003151, A001951, ..."
S1_CLUSTER_RE = re.compile(r"all essentially the same", re.I)
# cue immediately before THE target ("Essentially the same as A132732"), so that
# "essentially the same as row sums of A128715" does NOT mark A128715 as s1.
S1_DIRECT_TMPL = (
    r"(?:essentially the same as|essentially a duplicate of|duplicate of)"
    r"\s+(?:the\s+)?{a}\b"
)
DUP_NAME_RE = re.compile(r"duplicate of A(\d{6,7})", re.I)

# s2 cues: the transform battery's textual fingerprints (battery frozen in
# PRE_REGISTRATION.md section 5.4/5.6).
S2_CUE_RE = re.compile(
    r"binomial transform|euler transform|m(?:oe|ö|o)bius transform"
    r"|partial sums?|first differences?|second differences?|bisections?"
    r"|complement|signed version|unsigned version|decimations?|offset|shift"
    r"|constant multiple|negated|negative of|twice|half",
    re.I,
)
# NAME-based transform cue: "<transform> [of] A######" inside an endpoint's NAME,
# pointing at the other endpoint (e.g. A354784 "First differences of A000213,
# also twice A000073." -> (A354784, A000073) is s2).
NAME_XFORM_TMPL = (
    r"(?:first differences?|second differences?|partial sums?"
    r"|binomial transform|inverse binomial transform|euler transform"
    r"|m(?:oe|ö|o)bius transform|inverse m(?:oe|ö|o)bius transform"
    r"|bisections?|complement|twice|signed version|unsigned version|negative)"
    r"(?:\s+of|\s+are in|\s+gives?|\s+in|\s*:)?\s+(?:the\s+)?{a}\b"
)

# s4 (constant-type endpoint): name says it is an expansion of a constant, OR it
# carries keyword `cons` WITHOUT `core` (A000012/A000035 carry cons+core and are
# NOT constants in context -- the development audit's s4 false positives).
CONST_NAME_RE = re.compile(
    r"^(?:decimal expansion|continued fraction expansion"
    r"|expansion of .{0,60} in base|binary expansion)",
    re.I,
)

A_TOKEN_RE = re.compile(r"A(\d{6,7})")
SENT_SPLIT_RE = re.compile(r"(?<=\.)\s+")
PAREN_RE = re.compile(r"\([^()]*\)")


def a_str(n: int) -> str:
    return f"A{n:06d}" if n < 10**6 else f"A{n:07d}"


def strip_parens(text: str, rounds: int = 3) -> str:
    """Remove (possibly nested, up to `rounds`) parenthetical groups."""
    for _ in range(rounds):
        new = PAREN_RE.sub(" ", text)
        if new == text:
            break
        text = new
    return text


def split_sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT_RE.split(text) if s.strip()]


# ----------------------------------------------------------------- classifier


NAME_XFORM_GLOBAL = re.compile(
    NAME_XFORM_TMPL.format(a=r"A(\d{6,7})"), re.I
)


class StratifierV1:
    """Frozen v1 mention classifier. Context = names + dup-of map + const set.

    The decision RULES are frozen; for speed, name-based cues are precomputed
    in one pass over names.gz (identical result to per-pair regex search).
    """

    def __init__(self, names: dict[int, str], const_like: set[int]):
        self.names = names
        self.const_like = const_like
        self.dup_of: dict[int, int] = {}
        # name_refs[holder] = set of A-numbers named in a transform cue in
        # holder's NAME ("First differences of A000213, also twice A000073").
        self.name_refs: dict[int, set[int]] = {}
        for a, nm in names.items():
            m = DUP_NAME_RE.search(nm)
            if m:
                self.dup_of[a] = int(m.group(1))
            refs = {int(x) for x in NAME_XFORM_GLOBAL.findall(nm)}
            if refs:
                self.name_refs[a] = refs

    # -- per-line preprocessing (one line may carry many mentions) -------------

    def _line_context(self, raw_text: str):
        sents = split_sentences(raw_text)
        stripped = [strip_parens(s) for s in sents]
        sent_targets = [
            {int(m) for m in A_TOKEN_RE.findall(s)} for s in stripped
        ]
        return sents, stripped, sent_targets

    def line_cue_sets(self, source: int, raw_text: str):
        """Compute, ONCE per line, the target sets each text rule fires for.

        Returns (cluster_or_direct_s1_targets, sentence_scoped_s2_targets).
        Pure performance refactor of the frozen rules: identical decisions to
        evaluating each rule per mention (covered by tests/test_stratify.py).
        """
        s1_targets: set[int] = set()
        s2_targets: set[int] = set()
        has_s1 = bool(S1_CUE_RE.search(raw_text))
        has_s2 = bool(S2_CUE_RE.search(raw_text))
        if not (has_s1 or has_s2):
            return s1_targets, s2_targets
        _, stripped, sent_targets = self._line_context(raw_text)
        if has_s1:
            line_stripped = strip_parens(raw_text)
            for m in S1_DIRECT_GLOBAL.finditer(line_stripped):
                s1_targets.add(int(m.group(1)))
            for s, tset in zip(stripped, sent_targets):
                if S1_CLUSTER_RE.search(s):
                    s1_targets |= tset
        if has_s2:
            for s, tset in zip(stripped, sent_targets):
                only = tset - {source}
                if len(only) == 1 and S2_CUE_RE.search(s):
                    s2_targets |= only
        return s1_targets, s2_targets

    def classify_line(
        self, source: int, raw_text: str, mentions: list[tuple[int, str]]
    ) -> list[str]:
        """mentions = [(target, paren), ...] for one %Y line -> stratum each."""
        s1_line, s2_line = self.line_cue_sets(source, raw_text)
        out: list[str] = []
        for target, paren in mentions:
            out.append(
                self._classify_mention(source, target, paren, s1_line, s2_line)
            )
        return out

    def _classify_mention(
        self,
        source: int,
        target: int,
        paren: str,
        s1_line: set[int],
        s2_line: set[int],
    ) -> str:
        # NOTE on sentence scoping: a target can appear in several sentences;
        # cue sets are unions over sentences, exactly as the per-mention rules.
        # ---- s1
        if self.dup_of.get(source) == target or self.dup_of.get(target) == source:
            return "s1"
        if paren and S1_CUE_RE.search(paren):
            return "s1"
        if target in s1_line:
            return "s1"
        # ---- s2 (target-attributed)
        if paren and S2_CUE_RE.search(paren):
            return "s2"
        if (target in self.name_refs.get(source, _EMPTY)
                or source in self.name_refs.get(target, _EMPTY)):
            return "s2"
        # ---- s4 (constant-type endpoint)
        if source in self.const_like or target in self.const_like:
            return "s4"
        # ---- s2 (sentence-scoped, target is the only A-number in the sentence)
        if target in s2_line:
            return "s2"
        return "s3"


_EMPTY: frozenset[int] = frozenset()
S1_DIRECT_GLOBAL = re.compile(
    S1_DIRECT_TMPL.format(a=r"A(\d{6,7})"), re.I
)


def build_const_like(names: dict[int, str], meta) -> set[int]:
    """Constant-type endpoints: CONST name pattern, or keyword cons w/o core."""
    const = {a for a, nm in names.items() if CONST_NAME_RE.match(nm)}
    kw_lists = meta["keywords"].fillna("").str.split(",")
    for a, kws in zip(meta["a_number"].tolist(), kw_lists):
        if "cons" in kws and "core" not in kws:
            const.add(int(a))
    return const


# ------------------------------------------------------------------- pipeline


def restratify(force: bool = False) -> None:
    import pandas as pd

    if XREFS_V1_PARQUET.exists() and not force:
        _log(f"stratify: skip (exists): {XREFS_V1_PARQUET.name}")
        return
    t0 = time.monotonic()
    _log("stratify: loading xrefs + names + meta")
    edges = pd.read_parquet(XREFS_PARQUET)
    names = parse_names()
    meta = pd.read_parquet(SEQ_META_PARQUET)
    clf = StratifierV1(names, build_const_like(names, meta))

    _log(f"stratify: classifying {len(edges):,} mentions (run-length per %Y line)")
    n = len(edges)
    strata = [""] * n
    src_arr = edges["source"].to_numpy()
    yidx_arr = edges["y_line_idx"].to_numpy()
    tgt_arr = edges["target"].to_numpy()
    par_arr = edges["paren"].tolist()
    raw_arr = edges["raw_text"].tolist()
    # mentions of one %Y line are contiguous rows (written that way by
    # build_graph.extract_xrefs): iterate line runs without a groupby
    i = 0
    n_lines = 0
    while i < n:
        j = i + 1
        while j < n and src_arr[j] == src_arr[i] and yidx_arr[j] == yidx_arr[i]:
            j += 1
        source = int(src_arr[i])
        s1_line, s2_line = clf.line_cue_sets(source, raw_arr[i])
        for k in range(i, j):
            strata[k] = clf._classify_mention(
                source, int(tgt_arr[k]), par_arr[k], s1_line, s2_line
            )
        n_lines += 1
        if n_lines % 200_000 == 0:
            _log(f"stratify:   ... {n_lines:,} lines / {j:,} mentions "
                 f"in {time.monotonic()-t0:.0f}s")
        i = j
    edges["stratum_v1"] = strata
    edges.to_parquet(XREFS_V1_PARQUET, engine="pyarrow", index=False)
    _log(
        f"stratify: wrote {XREFS_V1_PARQUET.name} in {time.monotonic()-t0:.0f}s "
        f"({XREFS_V1_PARQUET.stat().st_size:,} B)"
    )


def pair_table(column: str = "stratum_v1"):
    """Deduped undirected pairs with strongest stratum label (frozen RANK)."""
    import numpy as np
    import pandas as pd

    edges = pd.read_parquet(XREFS_V1_PARQUET, columns=["source", "target", column])
    edges["rank"] = edges[column].map(RANK).astype("int8")
    ded = edges.sort_values("rank").drop_duplicates(["source", "target"], keep="first")
    u = np.minimum(ded["source"].to_numpy(), ded["target"].to_numpy())
    v = np.maximum(ded["source"].to_numpy(), ded["target"].to_numpy())
    und = pd.DataFrame({"u": u, "v": v, "rank": ded["rank"].to_numpy()})
    pairs = und.groupby(["u", "v"], sort=True)["rank"].min().reset_index()
    inv = {r: s for s, r in RANK.items()}
    pairs["stratum"] = pairs["rank"].map(inv)
    return pairs.drop(columns=["rank"])


def write_counts() -> dict:
    import pandas as pd

    from .build_graph import parse_stripped_lengths

    pairs = pair_table()
    stripped = parse_stripped_lengths()
    sset = set(stripped)
    both = pairs[pairs["u"].isin(sset) & pairs["v"].isin(sset)]
    counts = {
        "column": "stratum_v1",
        "undirected_pairs": int(len(pairs)),
        "undirected_pairs_both_in_stripped": int(len(both)),
        "strata_undirected_both_in_stripped": {
            k: int(v) for k, v in both["stratum"].value_counts().sort_index().items()
        },
    }
    COUNTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    COUNTS_JSON.write_text(json.dumps(counts, indent=2) + "\n")
    _log(f"stratify: wrote {COUNTS_JSON.relative_to(PROJECT_ROOT)}: "
         f"{counts['strata_undirected_both_in_stripped']}")
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stratify", description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    restratify(force=args.force)
    write_counts()
    return 0


if __name__ == "__main__":
    sys.exit(main())
