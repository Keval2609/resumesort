#!/usr/bin/env python3
"""
reasoning.py — Stage 4-compliant reasoning column generator.

Stage 4 checks (10 random rows sampled):
  1. Specific facts    — must cite YOE, title, company, named skills, signal values
  2. JD connection     — connect to what the JD actually asks for
  3. Honest concerns   — acknowledge real gaps (notice, location, salary, exp level)
  4. No hallucination  — every claim must exist in the profile
  5. Variation         — no templated strings; each row must differ
  6. Rank consistency  — top ranks sound positive; bottom ranks sound cautious

Usage:
    from reasoning import generate_reasoning
    text = generate_reasoning(rank=1, candidate=c, score=0.91)
"""

from __future__ import annotations

from datetime import date, datetime

# ── JD reference constants ────────────────────────────────────────────────────
JD_REQUIRED_SKILLS_LOWER = {
    "sentence transformers", "sentence-transformers",
    "embeddings", "vector search", "hybrid search",
    "bge", "e5", "openai embeddings",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch",
    "information retrieval", "retrieval",
    "ranking", "recommendation systems",
    "learning to rank", "ndcg", "map", "mrr",
    "python",
    "hugging face transformers", "hugging face",
    "bm25", "nlp",
}
JD_NICE_SKILLS_LOWER = {
    "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning",
    "xgboost", "lightgbm", "mlflow", "mlops",
    "feature engineering", "scikit-learn",
    "pytorch", "tensorflow",
}
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree",
}
INDIA_TIER1_CITIES = {
    "pune", "noida", "hyderabad", "bangalore", "bengaluru",
    "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram",
}
JD_NOTICE_SOFT = 30
JD_NOTICE_HARD = 90
JD_MAX_SAL_LPA = 60.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        anchor = date(2026, 6, 13)
        return (anchor - d).days
    except Exception:
        return 9999


def _matched_required(candidate: dict) -> list[str]:
    return [
        s["name"] for s in candidate.get("skills", [])
        if s["name"].lower() in JD_REQUIRED_SKILLS_LOWER
        and s.get("proficiency") in ("intermediate", "advanced", "expert")
    ]


def _matched_nice(candidate: dict) -> list[str]:
    return [
        s["name"] for s in candidate.get("skills", [])
        if s["name"].lower() in JD_NICE_SKILLS_LOWER
    ]


def _product_months(candidate: dict) -> int:
    return sum(
        j.get("duration_months", 0)
        for j in candidate.get("career_history", [])
        if not any(cf in j.get("company", "").lower() for cf in CONSULTING_FIRMS)
    )


def _is_consulting_only(candidate: dict) -> bool:
    jobs = candidate.get("career_history", [])
    if not jobs:
        return False
    return all(
        any(cf in j.get("company", "").lower() for cf in CONSULTING_FIRMS)
        for j in jobs
    )


def _location_status(candidate: dict) -> tuple[bool, str]:
    profile = candidate.get("profile", {})
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
    in_tier1 = any(city in loc for city in INDIA_TIER1_CITIES)

    if in_tier1:
        return True, profile.get("location", loc)
    if country != "india":
        return relocate, f"{profile.get('location')} (abroad; relocate={relocate})"
    return relocate, f"{profile.get('location')} (non-metro; relocate={relocate})"


def _salary_concern(candidate: dict) -> str | None:
    sal = candidate.get("redrob_signals", {}).get("expected_salary_range_inr_lpa", {})
    mn, mx = sal.get("min", 0), sal.get("max", 0)
    if mx > JD_MAX_SAL_LPA * 1.30:
        return f"salary band {mn:.0f}–{mx:.0f}L may exceed JD budget"
    return None


def _engagement_facts(candidate: dict) -> dict:
    sig = candidate.get("redrob_signals", {})
    return {
        "open": sig.get("open_to_work_flag", False),
        "days_inactive": _days_since(sig.get("last_active_date", "2000-01-01")),
        "rr": sig.get("recruiter_response_rate", 0.0),
        "rt_hours": sig.get("avg_response_time_hours", 999),
        "apps": sig.get("applications_submitted_30d", 0),
        "views": sig.get("profile_views_received_30d", 0),
        "saved": sig.get("saved_by_recruiters_30d", 0),
        "github": sig.get("github_activity_score", -1),
        "notice": sig.get("notice_period_days", 0),
        "assessments": sig.get("skill_assessment_scores", {}),
        "interview_rate": sig.get("interview_completion_rate", 0),
    }


# ── Segment builders ──────────────────────────────────────────────────────────

def _build_skill_segment(required: list[str], nice: list[str]) -> str:
    """Segment 1: skill match with JD."""
    if required:
        parts = f"matches JD on: {', '.join(required[:3])}"
        if nice:
            parts += f"; bonus: {nice[0]}"
        return parts
    if nice:
        return f"no core IR/retrieval skills matched; adjacent: {', '.join(nice[:2])}"
    return "no direct ML/IR/retrieval skill match with JD"


def _build_career_segment(candidate: dict) -> str:
    """Segment 2: career background honesty."""
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)
    pm = _product_months(candidate)
    consulting_only = _is_consulting_only(candidate)

    base = f"{yoe:.1f}yr career, currently {title} at {company}"

    if consulting_only:
        base += "; entire career in services/consulting firms (JD disqualifier risk)"
    elif pm >= 36:
        base += f"; {pm//12}yr+ at product companies"
    elif pm > 0:
        base += f"; {pm}mo product-company experience"
    else:
        base += "; minimal product-company exposure"

    return base


def _build_engagement_segment(eng: dict) -> str:
    """Segment 3: engagement summary with concrete numbers."""
    parts = []

    # Availability
    if eng["open"]:
        parts.append("open-to-work flag set")
    if eng["days_inactive"] < 14:
        parts.append(f"active {eng['days_inactive']}d ago")
    elif eng["days_inactive"] < 60:
        parts.append(f"last active {eng['days_inactive']}d ago")
    else:
        parts.append(f"inactive {eng['days_inactive']}d (low hire probability)")

    # Responsiveness
    rr_pct = f"{eng['rr']:.0%}"
    if eng["rr"] >= 0.7:
        parts.append(f"high recruiter response rate ({rr_pct})")
    elif eng["rr"] <= 0.15:
        parts.append(f"very low response rate ({rr_pct})")
    else:
        parts.append(f"response rate {rr_pct}")

    # Notice
    notice = eng["notice"]
    if notice == 0:
        parts.append("immediate joiner")
    elif notice <= JD_NOTICE_SOFT:
        parts.append(f"notice {notice}d (within JD soft limit)")
    elif notice <= JD_NOTICE_HARD:
        parts.append(f"notice {notice}d (buyout possible)")
    else:
        parts.append(f"notice {notice}d (exceeds JD hard limit of {JD_NOTICE_HARD}d)")

    return "; ".join(parts)


def _build_signal_segment(eng: dict) -> str:
    """Segment 4: extra positive/negative signals."""
    parts = []
    if eng["github"] > 50:
        parts.append(f"strong GitHub activity ({eng['github']:.0f}/100)")
    elif eng["github"] > 20:
        parts.append(f"GitHub score {eng['github']:.0f}/100")
    if eng["assessments"]:
        top_skill = max(eng["assessments"], key=eng["assessments"].get)
        top_score = eng["assessments"][top_skill]
        parts.append(f"platform assessment: {top_skill} {top_score:.0f}/100")
    if eng["saved"] >= 10:
        parts.append(f"saved by {eng['saved']} recruiters/30d")
    if eng["interview_rate"] >= 0.8:
        parts.append(f"high interview completion ({eng['interview_rate']:.0%})")
    elif eng["interview_rate"] < 0.4:
        parts.append(f"low interview show-up rate ({eng['interview_rate']:.0%})")
    return "; ".join(parts) if parts else ""


def _build_concern_segment(
    candidate: dict,
    required: list[str],
    eng: dict,
    loc_ok: bool,
    loc_label: str,
) -> str:
    """Segment 5: honest concerns for Stage 4."""
    concerns = []
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    if yoe < 4:
        concerns.append(f"experience ({yoe:.1f}yr) below JD floor of 5yr")
    elif yoe > 12:
        concerns.append(f"overqualified at {yoe:.1f}yr; may not be IC-coded")

    if not required:
        concerns.append("core retrieval/ranking skills absent from profile")
    elif len(required) < 2:
        concerns.append(f"only {len(required)} core JD skill matched")

    if not loc_ok:
        concerns.append(f"location: {loc_label}")

    sal_issue = _salary_concern(candidate)
    if sal_issue:
        concerns.append(sal_issue)

    if eng["days_inactive"] > 90:
        concerns.append(f"profile inactive {eng['days_inactive']}d")

    if eng["rr"] < 0.20:
        concerns.append(f"recruiter response rate {eng['rr']:.0%}")

    if _is_consulting_only(candidate):
        concerns.append("all experience at consulting firms (JD explicitly cautious)")

    return ("Concerns: " + "; ".join(concerns) + ".") if concerns else ""


# ── Main public function ──────────────────────────────────────────────────────

def generate_reasoning(rank: int, candidate: dict, score: float) -> str:
    """
    Generate a Stage 4-compliant reasoning string.

    Guarantees:
    - Cites specific facts from this candidate (no hallucination)
    - Connects to the JD explicitly
    - Acknowledges genuine concerns
    - Tone matches rank (positive top, cautious bottom)
    - Max ~420 chars for CSV readability
    """
    required = _matched_required(candidate)
    nice = _matched_nice(candidate)
    eng = _engagement_facts(candidate)
    loc_ok, loc_label = _location_status(candidate)

    skill_seg = _build_skill_segment(required, nice)
    career_seg = _build_career_segment(candidate)
    engagement_seg = _build_engagement_segment(eng)
    signal_seg = _build_signal_segment(eng)
    concern_seg = _build_concern_segment(candidate, required, eng, loc_ok, loc_label)

    # Combine segments
    parts = [career_seg + ".", skill_seg + ".", engagement_seg + "."]
    if signal_seg:
        parts.append(signal_seg + ".")
    if concern_seg:
        parts.append(concern_seg)

    # Rank-tone adjustment
    if rank <= 10 and not concern_seg:
        parts.append("Strong overall fit for founding AI engineer role.")
    elif rank >= 85:
        parts.append("Marginal fit; included at tail of ranking.")

    reasoning = " ".join(parts)

    # Hard cap: truncate to 430 chars cleanly
    if len(reasoning) > 430:
        reasoning = reasoning[:427] + "..."

    return reasoning


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, sys

    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/project/sample_candidates.json"
    with open(path) as f:
        candidates = json.load(f)

    print(f"{'RANK':>4}  {'ID':12}  {'LEN':>4}  REASONING")
    print("-" * 100)
    for i, c in enumerate(candidates[:10], start=1):
        r = generate_reasoning(i, c, round(1.0 - i * 0.01, 3))
        print(f"{i:>4}  {c['candidate_id']:12}  {len(r):>4}  {r[:80]}...")
