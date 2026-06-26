#!/usr/bin/env python3

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



# ── Helpers ───────────────────────────────────────────────────────────────────

def _days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        anchor = date(2026, 6, 13)
        return (anchor - d).days
    except Exception:
        return -1


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


def _career_months(candidate: dict) -> int:
    return sum(
        j.get("duration_months", 0)
        for j in candidate.get("career_history", [])
    )


def _effective_yoe(candidate: dict) -> float:
    profile_yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    return max(profile_yoe, _career_months(candidate) / 12.0)


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
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
    in_tier1 = any(city in loc for city in INDIA_TIER1_CITIES)

    if in_tier1:
        return True, profile.get("location", loc)
    if country != "india":
        return relocate, f"{profile.get('location')} (abroad; relocate={relocate})"
    return relocate, f"{profile.get('location')} (non-metro; relocate={relocate})"

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
    career_months = _career_months(candidate)
    yoe = _effective_yoe(candidate)
    pm = min(_product_months(candidate), career_months)
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
    if eng["days_inactive"] == -1:
        parts.append("activity date not available")
    elif eng["days_inactive"] < 14:
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
    if eng["assessments"]:
        top_skill = max(eng["assessments"], key=eng["assessments"].get)
        top_score = eng["assessments"][top_skill]
        parts.append(f"platform assessment: {top_skill} {top_score:.0f}/100")
    if eng["interview_rate"] >= 0.8:
        parts.append(f"high interview completion ({eng['interview_rate']:.0%})")
    elif eng["interview_rate"] < 0.4:
        parts.append(f"low interview show-up rate ({eng['interview_rate']:.0%})")
    return "; ".join(parts) if parts else ""


# ── Score-derived concern engine ──────────────────────────────────────────────

# Weight = actual impact on final score (component_weight × parent_weight)
SCORE_LABELS = {
    "skills_s":    (0.315, "skill coverage"),        # 0.45 × 0.70
    "exp_s":       (0.175, "experience fit"),         # 0.25 × 0.70
    "title_s":     (0.105, "title alignment"),        # 0.15 × 0.70
    "readiness_s": (0.090, "availability/notice"),    # 0.30 × 0.30
    "recruiter_s": (0.075, "recruiter engagement"),   # 0.25 × 0.30
    "prof_s":      (0.060, "professionalism"),        # 0.20 × 0.30
    "trust_s":     (0.045, "trust signals"),          # 0.15 × 0.30
    "edu_s":       (0.056, "education"),              # 0.08 × 0.70
    "loc_s":       (0.035, "location fit"),           # 0.05 × 0.70
}

# Thresholds: below these the component is flagged as a concern
CONCERN_THRESHOLD = {
    "skills_s":    0.55,
    "exp_s":       0.55,
    "title_s":     0.50,
    "readiness_s": 0.50,
    "recruiter_s": 0.40,
    "prof_s":      0.45,
    "trust_s":     0.40,
    "edu_s":       0.40,
    "loc_s":       0.60,   # location threshold higher — JD is location-sensitive
}


def _concerns_from_scores(scores: dict, top_n: int = 2) -> list[tuple[str, float]]:
    """Returns [(key, val), ...] sorted by impact. Caller humanizes."""
    if not scores:
        return []

    gaps = []
    for key, (weight, _label) in SCORE_LABELS.items():
        val = scores.get(key, 1.0)
        threshold = CONCERN_THRESHOLD.get(key, 0.50)
        if val < threshold:
            gaps.append((weight * (1.0 - val), key, val))

    gaps.sort(reverse=True)
    return [(key, val) for _, key, val in gaps[:top_n]]


def _humanize_concern(
    key: str,
    val: float,
    candidate: dict,
    eng: dict,
) -> str | None:
    """
    Converts a weak score key into a recruiter-readable sentence.
    Returns None for 'exp_s' — handled separately by _yoe_concern.
    """
    if key == "exp_s":
        return None

    if key == "skills_s":
        return "limited coverage of core JD skills (embeddings, IR, vector search)"

    if key == "title_s":
        title = candidate.get("profile", {}).get("current_title", "current role")
        return f"current title ({title!r}) not aligned with ML/IR engineering"

    if key == "readiness_s":
        notice = eng["notice"]
        inactive = eng["days_inactive"]
        if notice > JD_NOTICE_HARD:
            return f"{notice}-day notice period exceeds the {JD_NOTICE_HARD}-day JD limit"
        if notice > JD_NOTICE_SOFT:
            return f"{notice}-day notice is above the preferred {JD_NOTICE_SOFT}-day window"
        if inactive > 0 and inactive > 90:
            return f"profile inactive for {inactive} days"
        return "low recent job-search activity"

    if key == "recruiter_s":
        return "low recruiter engagement in the past 30 days"

    if key == "prof_s":
        return f"recruiter response rate is low ({eng['rr']:.0%})"

    if key == "trust_s":
        return "limited verifiable profile signals"

    if key == "edu_s":
        return "educational background outside core CS/ML fields"

    if key == "loc_s":
        loc_ok, loc_label = _location_status(candidate)
        if not loc_ok:
            return f"located outside preferred cities ({loc_label}); relocation required"
        return "location is a minor consideration"

    return None


def _yoe_concern(candidate: dict, scores: dict, rank: int) -> str | None:
    """
    Explicit YOE concern — score-derived concerns can't distinguish
    underqualified vs overqualified vs low-depth from exp_s alone.
    Only fires if exp_s is actually weak.
    """
    yoe = _effective_yoe(candidate)
    exp_s = scores.get("exp_s", 1.0)

    # Only surface YOE concern if experience score is actually hurting
    if exp_s >= 0.65:
        return None

    if yoe < 5:
        concern = f"underqualified at {yoe:.1f}yr (JD: 5–9yr)"
        if rank <= 30:
            required = _matched_required(candidate)
            if required:
                return f"{concern}, offset partially by {required[0]} depth"
        return concern

    if yoe > 12:
        concern = f"overqualified at {yoe:.1f}yr — IC-fit risk"
        if rank <= 30:
            return f"{concern}, though product-company depth remains relevant"
        return concern

    # exp_s low but YOE is in range → low product-AI depth (handled by score label)
    return None


def generate_reasoning(
    rank: int,
    candidate: dict,
    score: float,
    scores: dict = None,        # ← full score dict from final_score()
) -> str:

    required = _matched_required(candidate)
    nice     = _matched_nice(candidate)
    eng      = _engagement_facts(candidate)
    loc_ok, loc_label = _location_status(candidate)

    career  = _build_career_segment(candidate)
    skills  = _build_skill_segment(required, nice)
    signals = _build_signal_segment(eng)

    # ── Score-derived concerns → humanized text ───────────────────────────
    raw_concerns = _concerns_from_scores(scores or {})
    concerns = []
    for key, val in raw_concerns:
        text = _humanize_concern(key, val, candidate, eng)
        if text:
            concerns.append(text)

    # ── Inject YOE concern if applicable ─────────────────────────────────
    yoe_c = _yoe_concern(candidate, scores or {}, rank)
    if yoe_c:
        concerns = [c for c in concerns if "experience fit" not in c]
        concerns.insert(0, yoe_c)

    parts = [career + ".", skills + "."]

    # ── Rank-tier framing ─────────────────────────────────────────────────
    strength = required[0] if required else "overall profile strength"

    if rank <= 10:
        if signals:
            parts.append(signals.capitalize() + ".")
        if not concerns:
            parts.append("Strong overall fit for the founding AI engineer role.")
        else:
            parts.append(
                f"{concerns[0].capitalize()}, but {strength} expertise "
                f"and product-company background justify the high ranking."
            )

    elif rank <= 30:
        if signals:
            parts.append(signals.capitalize() + ".")
        if not concerns:
            parts.append("Solid fit; ranked here on overall profile strength.")
        else:
            parts.append(
                f"Strong candidate; note: {concerns[0]}."
            )

    elif rank <= 50:
        if signals:
            parts.append(signals.capitalize() + ".")
        if concerns:
            parts.append(f"Main concern: {'; '.join(concerns)}.")

    elif rank <= 84:
        if concerns:
            parts.append(f"Ranked here due to: {'; '.join(concerns)}.")
        else:
            parts.append(
                "Borderline match — weaker than higher-ranked candidates overall."
            )

    else:
        if concerns:
            parts.append(
                f"Near cutoff — {'; '.join(concerns)}."
            )
        else:
            parts.append(
                "Included at tail as marginal match across multiple dimensions."
            )

    reasoning = " ".join(parts)
    if len(reasoning) > 340:
        reasoning = reasoning[:337].rstrip() + "..."

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
