#!/usr/bin/env python3
"""
rank.py — single entry point for Redrob Hackathon submission.

Usage:
    python rank.py --candidates ./data/candidates.jsonl.gz \
                   --out ./team_xxx.csv

Constraints (per submission_spec.md Section 3):
    - Runtime  ≤ 5 min wall-clock
    - RAM      ≤ 16 GB
    - CPU only, no GPU
    - No network / external API calls
    - Output   exactly 100 rows + header, UTF-8 CSV
"""

import argparse
import csv
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── local pipeline modules ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from bm25 import build_index, tokenize, JD_QUERY
from scorer import final_score
from reasoning import generate_reasoning

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rank")

# ── constants ─────────────────────────────────────────────────────────────────
BM25_PREFILTER_K  = 3000   # candidates passed to full scorer after BM25
TOP_K             = 100    # final submission size (spec requirement)
WALL_CLOCK_BUDGET = 280    # seconds — leave 20s headroom under 5-min limit


# ═════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_candidates(path: Path) -> list[dict]:
    """Load candidates.jsonl, .jsonl.gz, or .json array → list[dict]."""
    """Load candidates.jsonl or candidates.jsonl.gz → list[dict]."""
    log.info(f"Loading candidates from {path} ...")
    t0 = time.perf_counter()

    candidates = []
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as f:
        raw = f.read().strip()

    # Support both JSON array [...] and JSONL (one object per line)
    if raw.startswith("["):
        candidates = json.loads(raw)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    elapsed = time.perf_counter() - t0
    log.info(f"Loaded {len(candidates):,} candidates in {elapsed:.1f}s")
    return candidates


def enforce_monotone(results: list[dict]) -> list[dict]:
    """
    Guarantee score[rank i] >= score[rank i+1].
    Spec: non-increasing scores required; ties broken by candidate_id asc.
    If a score inversion exists after sorting, clamp it down.
    """
    if not results:
        return results

    # Already sorted desc by (final, candidate_id asc) from score_all
    # Do a forward pass to enforce strict non-increase
    for i in range(1, len(results)):
        if results[i]["final"] > results[i - 1]["final"]:
            results[i]["final"] = results[i - 1]["final"]

    return results


def write_csv(results: list[dict],
              candidates_by_id: dict[str, dict],
              out_path: Path) -> None:
    """Write spec-compliant CSV: header + exactly 100 data rows, UTF-8."""
    assert len(results) == TOP_K, f"Expected {TOP_K} rows, got {len(results)}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, r in enumerate(results, start=1):
            cid = r["candidate_id"]
            candidate = candidates_by_id[cid]
            reasoning = generate_reasoning(rank, candidate, r['final'])

            writer.writerow([
                cid,
                rank,
                f"{r['final']:.6f}",
                reasoning,
            ])

    log.info(f"Wrote {out_path}  ({TOP_K} rows)")


# ═════════════════════════════════════════════════════════════════════════════
# Validation (mirrors validate_submission.py logic inline)
# ═════════════════════════════════════════════════════════════════════════════

def validate_output(out_path: Path, valid_ids: set[str]) -> bool:
    """Quick inline sanity check before returning. Returns True if clean."""
    errors = []

    with open(out_path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if rows[0] != ["candidate_id", "rank", "score", "reasoning"]:
        errors.append(f"Bad header: {rows[0]}")

    data = rows[1:]
    if len(data) != TOP_K:
        errors.append(f"Expected {TOP_K} data rows, got {len(data)}")

    seen_ids, seen_ranks = set(), set()
    prev_score = float("inf")

    for i, row in enumerate(data):
        if len(row) != 4:
            errors.append(f"Row {i+2}: expected 4 columns, got {len(row)}")
            continue
        cid, rank_s, score_s, _ = row

        if cid not in valid_ids:
            errors.append(f"Row {i+2}: unknown candidate_id {cid!r}")
        if cid in seen_ids:
            errors.append(f"Row {i+2}: duplicate candidate_id {cid!r}")
        seen_ids.add(cid)

        try:
            rank = int(rank_s)
            if not 1 <= rank <= 100:
                errors.append(f"Row {i+2}: rank {rank} out of range")
            if rank in seen_ranks:
                errors.append(f"Row {i+2}: duplicate rank {rank}")
            seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {i+2}: rank not int: {rank_s!r}")

        try:
            score = float(score_s)
            if score > prev_score + 1e-9:
                errors.append(
                    f"Row {i+2}: score {score} > prev {prev_score} (not monotone)"
                )
            prev_score = score
        except ValueError:
            errors.append(f"Row {i+2}: score not float: {score_s!r}")

    missing_ranks = set(range(1, 101)) - seen_ranks
    if missing_ranks:
        errors.append(f"Missing ranks: {sorted(missing_ranks)[:10]}")

    if errors:
        log.error("Output validation FAILED:")
        for e in errors:
            log.error(f"  {e}")
        return False

    log.info("Output validation passed ✓")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# Core pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(candidates: list[dict]) -> list[dict]:
    """
    Full ranking pipeline:
      1. BM25 pre-filter → top BM25_PREFILTER_K candidates
      2. Full scorer    → relevance + behavioral scores
      3. Sort + top-100
    """
    t0 = time.perf_counter()

    # ── Step 1: Build BM25 index ──────────────────────────────────────────
    log.info(f"Building BM25 index over {len(candidates):,} candidates ...")
    idx, bm25 = build_index(candidates)
    log.info(f"Index built in {time.perf_counter()-t0:.1f}s  "
             f"({len(idx.index):,} unique terms)")

    # ── Step 2: BM25 retrieval ────────────────────────────────────────────
    t1 = time.perf_counter()
    query_tokens = tokenize(JD_QUERY)
    log.info(f"Query tokens ({len(query_tokens)}): "
             f"{' '.join(query_tokens[:12])} ...")

    bm25_hits = bm25.retrieve(query_tokens, top_k=BM25_PREFILTER_K)
    hit_ids   = {doc_id for doc_id, _ in bm25_hits}

    # Always include BM25 misses for hard-gate-exempt checking;
    # add them at the back with score=0 so they can still be gated.
    # (Ensures we never miss a great candidate with unusual vocab.)
    all_hit_ids = hit_ids | {
        c["candidate_id"] for c in candidates
        if c["redrob_signals"].get("open_to_work_flag", False)
    }

    candidates_to_score = [
        c for c in candidates if c["candidate_id"] in all_hit_ids
    ]
    log.info(
        f"BM25 pre-filter: {len(bm25_hits)} hits + "
        f"{len(candidates_to_score)-len(bm25_hits)} open-to-work additions "
        f"→ {len(candidates_to_score)} to score  "
        f"(took {time.perf_counter()-t1:.1f}s)"
    )

    # ── Step 3: Full scoring ──────────────────────────────────────────────
    t2 = time.perf_counter()
    log.info("Running full scorer ...")

    scored = [final_score(c) for c in candidates_to_score]

    elapsed = time.perf_counter() - t2
    log.info(f"Scored {len(scored):,} candidates in {elapsed:.1f}s")

    # ── Step 4: Sort + select top-100 ─────────────────────────────────────
    # Primary: final score desc
    # Tie-break: candidate_id asc (per spec Section 3)
    scored.sort(key=lambda r: (-r["final"], r["candidate_id"]))

    # Log gate/honeypot stats
    gated     = sum(1 for r in scored if r["gate"] == "not_open_to_work")
    honeypots = sum(1 for r in scored if r["gate"] == "honeypot")
    work_mode = sum(1 for r in scored if r["gate"] == "work_mode_mismatch")
    salary    = sum(1 for r in scored if r["gate"] == "salary_too_high")
    active    = sum(1 for r in scored if r["gate"] is None)

    log.info(
        f"Gate stats — active: {active}  |  not_open_to_work: {gated}  |  "
        f"honeypots: {honeypots}  |  work_mode: {work_mode}  |  salary: {salary}"
    )

    if active < TOP_K:
        log.warning(
            f"Only {active} active candidates after gates — "
            f"filling remaining slots with best gated candidates"
        )
        # Fallback: include best gated candidates (excluding honeypots)
        # rather than producing fewer than 100 rows
        non_honeypot_gated = [
            r for r in scored
            if r["gate"] not in (None, "honeypot")
        ]
        scored = [r for r in scored if r["gate"] is None] + non_honeypot_gated
        log.warning(f"After fallback: {len(scored)} total candidates available")

    available = [r for r in scored if r["gate"] != "honeypot"]
    top100 = enforce_monotone(available[:TOP_K])

    total_elapsed = time.perf_counter() - t0
    log.info(
        f"Pipeline complete in {total_elapsed:.1f}s  "
        f"(budget: {WALL_CLOCK_BUDGET}s)"
    )
    if total_elapsed > WALL_CLOCK_BUDGET:
        log.warning(f"⚠ Exceeded budget by {total_elapsed - WALL_CLOCK_BUDGET:.0f}s")

    return top100


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Redrob Hackathon — candidate ranker"
    )
    p.add_argument(
        "--candidates",
        type=Path,
        default=Path("./data/candidates.jsonl.gz"),
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./team_xxx.csv"),
        help="Output CSV path (e.g. team_xxx.csv)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Score only first N candidates (for quick local testing)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wall_start = time.perf_counter()

    # ── Load ──────────────────────────────────────────────────────────────
    candidates = load_candidates(args.candidates)
    if args.sample:
        candidates = candidates[: args.sample]
        log.info(f"--sample mode: using first {args.sample} candidates")

    valid_ids = {c["candidate_id"] for c in candidates}
    candidates_by_id = {c["candidate_id"]: c for c in candidates}

    # ── Pipeline ──────────────────────────────────────────────────────────
    top100 = run_pipeline(candidates)

    # ── Guard: if fewer than 100 unique active candidates exist ───────────
    if len(top100) < TOP_K:
        log.error(
            f"Only {len(top100)} scoreable candidates in pool — "
            f"this sample is too small to produce {TOP_K} rows. "
            f"Run against the full 100K candidates.jsonl.gz"
        )
        sys.exit(1)

    # ── Write CSV ─────────────────────────────────────────────────────────
    write_csv(top100, candidates_by_id, args.out)

    # ── Validate ──────────────────────────────────────────────────────────
    ok = validate_output(args.out, valid_ids)
    if not ok:
        log.error("Submission invalid — fix errors before uploading")
        sys.exit(1)

    total = time.perf_counter() - wall_start
    log.info(f"Done. Total wall-clock: {total:.1f}s")
    log.info(f"Submit: {args.out.resolve()}")


if __name__ == "__main__":
    main()
