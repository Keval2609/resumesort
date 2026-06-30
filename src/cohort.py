
import logging
import numpy as np
from typing import Optional

from synergy import (
    RELATIVE_POOL_BOOST,
    _skill_names,
    _has_product_company_experience,
    _product_company_months,
    _count_retrieval_domain_skills,
    RECOMMENDATION_SKILLS,
    RETRIEVAL_SKILLS,
    RANKING_SKILLS,
)

log = logging.getLogger("cohort")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURABLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

COHORT_TOP_N = 300  # compute statistics over this many top candidates


# ═══════════════════════════════════════════════════════════════════════════════
# COHORT STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cohort_stats(
    scored_results: list[dict],
    candidates_by_id: dict[str, dict],
    top_n: int = COHORT_TOP_N,
) -> dict:
    """
    Compute median statistics over the top-N candidate pool.

    Returns dict with:
        median_recruiter_interest: float
        median_product_months: float
        median_retrieval_depth: float
        pool_size: int
    """
    pool = [
        r for r in scored_results[:top_n]
        if r.get("gate") is None
    ]

    if not pool:
        return {
            "median_recruiter_interest": 0.0,
            "median_product_months": 0.0,
            "median_retrieval_depth": 0.0,
            "pool_size": 0,
        }

    # Recruiter interest scores
    recruiter_scores = [r.get("recruiter_s", 0.0) for r in pool]

    # Product-company months (need candidate objects)
    product_months_list = []
    retrieval_depth_list = []
    for r in pool:
        cid = r["candidate_id"]
        candidate = candidates_by_id.get(cid)
        if candidate:
            product_months_list.append(_product_company_months(candidate))
            retrieval_depth_list.append(_count_retrieval_domain_skills(candidate))
        else:
            product_months_list.append(0)
            retrieval_depth_list.append(0)

    stats = {
        "median_recruiter_interest": float(np.median(recruiter_scores)),
        "median_product_months": float(np.median(product_months_list)),
        "median_retrieval_depth": float(np.median(retrieval_depth_list)),
        "pool_size": len(pool),
    }

    log.info(
        f"Cohort stats (n={stats['pool_size']}): "
        f"median_recruiter={stats['median_recruiter_interest']:.3f}, "
        f"median_product_mo={stats['median_product_months']:.0f}, "
        f"median_retrieval_depth={stats['median_retrieval_depth']:.1f}"
    )

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# COHORT BOOST
# ═══════════════════════════════════════════════════════════════════════════════

def apply_cohort_boost(
    result: dict,
    candidate: dict,
    cohort_stats: dict,
) -> dict:
    """
    Apply relative pool bonus if candidate dominates the cohort on
    multiple dimensions simultaneously.

    Conditions (ALL must be true):
      1. Product-company experience present
      2. Recommendation/retrieval/ranking evidence present
      3. Recruiter interest > cohort median

    Returns updated result dict with 'cohort_boost' key added.
    """
    if result.get("gate") is not None:
        result["cohort_boost"] = 0.0
        return result

    skill_names = _skill_names(candidate)
    has_product = _has_product_company_experience(candidate)
    has_rec_retrieval = bool(
        skill_names & (RECOMMENDATION_SKILLS | RETRIEVAL_SKILLS | RANKING_SKILLS)
    )
    recruiter_s = result.get("recruiter_s", 0.0)

    median_recruiter = cohort_stats.get("median_recruiter_interest", 0.0)

    if has_product and has_rec_retrieval and recruiter_s > median_recruiter:
        result["final"] = round(
            min(1.0, result["final"] + RELATIVE_POOL_BOOST), 6
        )
        result["cohort_boost"] = RELATIVE_POOL_BOOST
        log.debug(
            f"Applied cohort boost to {result['candidate_id']} "
            f"(recruiter_s={recruiter_s:.3f} > median={median_recruiter:.3f})"
        )
    else:
        result["cohort_boost"] = 0.0

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def cohort_rerank(
    scored_results: list[dict],
    candidates_by_id: dict[str, dict],
    top_n: int = COHORT_TOP_N,
) -> list[dict]:
    """
    Apply cohort-aware relative comparison to the top-N candidates.

    1. Compute pool statistics over top_n.
    2. Apply cohort boost to qualifying candidates.
    3. Re-sort by updated final score.

    Candidates beyond top_n pass through unchanged.
    """
    if not scored_results:
        return scored_results

    # Compute cohort statistics
    stats = compute_cohort_stats(scored_results, candidates_by_id, top_n)

    if stats["pool_size"] == 0:
        return scored_results

    # Split into cohort pool and tail
    pool = scored_results[:top_n]
    tail = scored_results[top_n:]

    # Apply cohort boosts
    boosted_count = 0
    for r in pool:
        cid = r["candidate_id"]
        candidate = candidates_by_id.get(cid)
        if candidate:
            apply_cohort_boost(r, candidate, stats)
            if r.get("cohort_boost", 0.0) > 0:
                boosted_count += 1

    # Re-sort pool
    pool.sort(key=lambda r: (-r["final"], r["candidate_id"]))

    log.info(f"Cohort reranking complete: {boosted_count} candidates boosted")

    return pool + tail
