
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

# ── Segment builders ──────────────────────────────────────────────────────────

def _build_skill_segment(required: list[str], nice: list[str]) -> str:
    """Segment 2: JD Match."""
    if required:
        formatted_req = [_format_skill_name(s) for s in required]
        skills_str = ", ".join(formatted_req[:2])
        if len(formatted_req) > 2:
            skills_str += f" and {formatted_req[2]}"
        return f"Experience with {skills_str} aligns well with the JD"
    if nice:
        formatted_nice = [_format_skill_name(s) for s in nice]
        skills_str = " and ".join(formatted_nice[:2])
        return f"Shows adjacent experience with {skills_str}, but limited evidence of core retrieval skills"
    return "Limited evidence of direct ML/IR/retrieval skills matching the JD requirements"


def _get_domain(required: list[str]) -> str:
    req_lower = [r.lower() for r in required]
    if "ndcg" in req_lower and "map" in req_lower:
        return "ranking evaluation"
    if "bm25" in req_lower and "faiss" in req_lower:
        return "retrieval systems"
    if "elasticsearch" in req_lower and "opensearch" in req_lower:
        return "search platforms"
    if "learning to rank" in req_lower:
        return "ranking systems"
    if "vector search" in req_lower:
        return "vector search"
    if "recommendation systems" in req_lower:
        return "recommendation engines"
    if "information retrieval" in req_lower:
        return "retrieval systems"
    return "core engineering"


def _build_career_segment(candidate: dict, required: list[str], rank: int) -> str:
    """Segment 1: Candidate Facts."""
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    yoe = _effective_yoe(candidate)
    pm = min(_product_months(candidate), _career_months(candidate))
    consulting_only = _is_consulting_only(candidate)

    domain = _get_domain(required)
    title_company = f"{title} at {company}" if company else title
    
    var = rank % 3
    if var == 0:
        starter = f"{title_company}"
    elif var == 1:
        starter = f"Profile shows {title_company}"
    else:
        starter = f"Evidence includes {title_company}"

    if consulting_only:
        return f"{starter} with {yoe:.1f} years of experience at consulting/services firms"
    elif pm >= 24:
        return f"{starter} with {yoe:.1f} years building {domain} in a product environment"
    else:
        return f"{starter} with {yoe:.1f} years of experience"


def _format_skill_name(skill: str) -> str:
    overrides = {
        "faiss": "FAISS",
        "lora": "LoRA",
        "nlp": "NLP",
        "llm": "LLM",
        "aws": "AWS",
        "gcp": "GCP",
        "ml": "ML",
        "ai": "AI",
        "ndcg": "NDCG",
        "bm25": "BM25",
        "mrr": "MRR",
        "map": "MAP"
    }
    return overrides.get(skill.lower(), skill.title())


def _build_engagement_segment(eng: dict, required: list[str], rank: int, has_concerns: bool) -> str:
    """Segment 3/Signals."""
    facts = []
    
    inactive = eng.get("days_inactive", -1)
    if inactive >= 0:
        if inactive == 0:
            facts.append("active today")
        else:
            facts.append(f"active {inactive} days ago")
            
    rr = eng.get("rr", 0.0)
    if rr > 0:
        facts.append(f"recruiter response rate {int(rr * 100)}%")
        
    notice = eng.get("notice", 0)
    if notice > 0:
        facts.append(f"notice period {notice} days")
        
    ir = eng.get("interview_rate", 0.0)
    if ir > 0:
        facts.append(f"interview completion {int(ir * 100)}%")
        
    if eng.get("assessments"):
        top_skill = max(eng["assessments"], key=eng["assessments"].get)
        top_score = eng["assessments"][top_skill]
        facts.append(f"{_format_skill_name(top_skill)} assessment: {top_score:.0f}/100")
        
    if not facts:
        return ""
        
    if len(facts) > 2:
        facts = facts[:2]
        
    return " and ".join(facts)


# ── Score-derived concern engine ──────────────────────────────────────────────

SCORE_LABELS = {
    "skills_s":    (0.315, "skill coverage"),
    "exp_s":       (0.175, "experience fit"),
    "title_s":     (0.105, "title alignment"),
    "readiness_s": (0.090, "availability/notice"),
    "recruiter_s": (0.075, "recruiter engagement"),
    "prof_s":      (0.060, "professionalism"),
    "trust_s":     (0.045, "trust signals"),
    "edu_s":       (0.056, "education"),
    "loc_s":       (0.035, "location fit"),
}

CONCERN_THRESHOLD = {
    "skills_s":    0.55,
    "exp_s":       0.55,
    "title_s":     0.50,
    "readiness_s": 0.50,
    "recruiter_s": 0.40,
    "prof_s":      0.45,
    "trust_s":     0.40,
    "edu_s":       0.40,
    "loc_s":       0.60,
}


def _concerns_from_scores(scores: dict, top_n: int = 2) -> list[tuple[str, float]]:
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
    if key == "exp_s":
        return None

    if key == "skills_s":
        return "limited evidence of core retrieval or ranking systems"

    if key == "title_s":
        title = candidate.get("profile", {}).get("current_title", "current role")
        return f"current title ({title}) is broader than the JD requires"

    if key == "readiness_s":
        notice = eng["notice"]
        inactive = eng["days_inactive"]
        if notice > JD_NOTICE_HARD:
            return f"the {notice}-day notice period exceeds preferences"
        if notice > JD_NOTICE_SOFT:
            return f"the {notice}-day notice period is a minor trade-off"
        if inactive > 0 and inactive > 90:
            return f"profile has been inactive for {inactive} days"
        return "job-search activity is low"

    if key == "recruiter_s":
        return "recruiter engagement is currently low"

    if key == "prof_s":
        return "recruiter response rate is relatively low"

    if key == "trust_s":
        return "there are limited verifiable profile signals"

    if key == "edu_s":
        return "educational background is outside core CS/ML fields"

    if key == "loc_s":
        loc_ok, loc_label = _location_status(candidate)
        if not loc_ok:
            return "relocation may be required"
        return "location is a minor consideration"

    return None


def _yoe_concern(candidate: dict, scores: dict, rank: int) -> str | None:
    yoe = _effective_yoe(candidate)
    exp_s = scores.get("exp_s", 1.0)

    if exp_s >= 0.65:
        return None

    if yoe < 5:
        return f"experience is slightly below the preferred range ({yoe:.1f} years)"

    if yoe > 12:
        return f"extensive experience ({yoe:.1f} years) poses a slight IC-fit risk"

    return None


def generate_reasoning(
    rank: int,
    candidate: dict,
    score: float,
    scores: dict = None,
) -> str:

    required = _matched_required(candidate)
    nice     = _matched_nice(candidate)
    eng      = _engagement_facts(candidate)

    career_str = _build_career_segment(candidate, required, rank)
    skills_str = _build_skill_segment(required, nice)

    raw_concerns = _concerns_from_scores(scores or {})
    concerns = []
    for key, val in raw_concerns:
        text = _humanize_concern(key, val, candidate, eng)
        if text:
            concerns.append(text)

    yoe_c = _yoe_concern(candidate, scores or {}, rank)
    if yoe_c:
        concerns = [c for c in concerns if "experience" not in c]
        concerns.insert(0, yoe_c)

    signals_str = _build_engagement_segment(eng, required, rank, bool(concerns))

    parts = [career_str + ".", skills_str + "."]
    
    if signals_str:
        parts.append(signals_str.capitalize() + ".")

    if concerns:
        # Join multiple concerns naturally
        if len(concerns) > 1:
            concerns_joined = ", and ".join(concerns[:2])
        else:
            concerns_joined = concerns[0]
        
        parts.append(f"However, {concerns_joined}.")

    # Clean up multiple periods, ensure spacing
    reasoning = " ".join(parts).replace("..", ".").replace(" .", ".")

    return reasoning



# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, sys

    path = sys.argv[1] if len(sys.argv) > 1 else "./data/candidates.jsonl.gz"
    from pathlib import Path
    sample_path = Path(path)
    if sample_path.suffix == ".gz":
        import gzip
        opener = gzip.open
    else:
        opener = open

    try:
        with opener(sample_path, "rt", encoding="utf-8") as f:
            raw = f.read().strip()
            if raw.startswith("["):
                candidates = json.loads(raw)
            else:
                candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except Exception as e:
        print(f"Skipping smoke test: could not load candidates from {sample_path}")
        candidates = []

    print(f"{'RANK':>4}  {'ID':12}  {'LEN':>4}  REASONING")
    print("-" * 100)
    for i, c in enumerate(candidates[:10], start=1):
        r = generate_reasoning(i, c, round(1.0 - i * 0.01, 3))
        print(f"{i:>4}  {c['candidate_id']:12}  {len(r):>4}  {r[:80]}...")
