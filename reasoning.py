
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
        skills_str = ", ".join(required[:2])
        if len(required) > 2:
            skills_str += f" and {required[2]}"
        return f"Experience with {skills_str} aligns closely with the JD"
    if nice:
        skills_str = " and ".join(nice[:2])
        return f"Shows some adjacent experience with {skills_str}, but lacks core retrieval skills"
    return "Lacks direct ML/IR/retrieval skills matching the JD requirements"


def _get_domain(required: list[str]) -> str:
    if "Learning to Rank" in required:
        return "ranking systems"
    if "Vector Search" in required:
        return "vector search"
    if "Recommendation Systems" in required:
        return "recommendation systems"
    if "Elasticsearch" in required or "OpenSearch" in required:
        return "search platform"
    if "Information Retrieval" in required:
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
    
    product_phrases = [
        "at product companies",
        "in product-driven environments",
        "at scaled product organizations"
    ]
    phrase = product_phrases[(rank - 1) % len(product_phrases)]

    if consulting_only:
        return f"{title_company} with {yoe:.1f} years of experience at consulting/services firms"
    elif pm >= 24:
        return f"{title_company} with {yoe:.1f} years building {domain} {phrase}"
    else:
        return f"{title_company} with {yoe:.1f} years of experience"


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
    parts = []
    
    if eng.get("rr", 0) >= 0.6 and eng.get("days_inactive", -1) >= 0 and eng.get("days_inactive", -1) <= 30:
        parts.append("strong recruiter engagement")
    
    borderline = rank > 50 or has_concerns or len(required) < 3
    if borderline and eng.get("assessments"):
        top_skill = max(eng["assessments"], key=eng["assessments"].get)
        top_score = eng["assessments"][top_skill]
        if top_score >= 80:
            parts.append(f"a strong {_format_skill_name(top_skill)} assessment ({top_score:.0f}/100)")
            
    if eng.get("interview_rate", 0) >= 0.8:
        parts.append("consistent interview participation")
        
    return " and ".join(parts[:2])


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


# ── Varied sentence templates per rank tier ───────────────────────────────────

_TOP20_CLEAN = [
    "The ranking is further supported by {a} {mod} {domain} background{and_signals}.",
    "This strong profile is backed by {mod} {domain} experience{and_signals}.",
    "Ranked highly, with {a} {mod} {domain} background{and_signals} adding confidence.",
]

_TOP20_CONCERN = [
    "Although {concern}, {a} {mod} {domain} background{and_signals} {supports} a top-tier ranking.",
    "{A} {mod} {domain} background{and_signals} {helps} offset the fact that {concern}, making this one of the strongest matches.",
    "While {concern}, {a} {mod} {domain} background{and_signals} {places} the candidate among the best overall fits.",
    "{A} {mod} {domain} background{and_signals} {outweighs} the fact that {concern}.",
    "While {concern}, this profile remains highly competitive due to {mod} {domain} experience{and_signals}.",
]

_TOP50_CLEAN = [
    "{A} {mod} {domain} background{and_signals} {reinforces} this match.",
    "A well-rounded profile, further supported by {mod} {domain} experience{and_signals}.",
]

_TOP50_CONCERN = [
    "{A} {mod} {domain} background{and_signals} {is_are} a strong plus, though {concern} places this candidate below higher-ranked profiles.",
    "A solid candidate backed by {a} {mod} {domain} background{and_signals}, but {concern} is a trade-off.",
    "While {a} {mod} {domain} background{and_signals} {is_are} positive, {concern} keeps the ranking in the mid-tier.",
]

_TAIL_CONCERN = [
    "Included near the cutoff because {concerns}.",
    "Placed below stronger matches since {concerns}.",
]

_TAIL_CLEAN = [
    "Included near the cutoff as adjacent experience may still be transferable.",
    "Marginal match overall, as stronger candidates demonstrate deeper retrieval stack experience.",
]


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

    concern_text = concerns[0] if concerns else ""
    concerns_joined = " and ".join(concerns) if concerns else ""
    
    domain = _get_domain(required)
    and_signals = f" and {signals_str}" if signals_str else ""

    modifiers = [
        "solid",
        "strong",
        "deep",
        "proven",
        "extensive",
        "hands-on",
        "demonstrated",
        "well-established",
        "production"
    ]
    mod = modifiers[(rank - 1) % len(modifiers)]

    idx = rank - 1
    
    is_plural = bool(signals_str)
    
    format_args = {
        "mod": mod,
        "domain": domain,
        "and_signals": and_signals,
        "concern": concern_text,
        "a": "an" if mod == "extensive" else "a",
        "A": "An" if mod == "extensive" else "A",
        "supports": "support" if is_plural else "supports",
        "helps": "help" if is_plural else "helps",
        "places": "place" if is_plural else "places",
        "outweighs": "outweigh" if is_plural else "outweighs",
        "reinforces": "reinforce" if is_plural else "reinforces",
        "is_are": "are" if is_plural else "is"
    }

    if rank <= 20:
        if not concerns:
            tpl = _TOP20_CLEAN[idx % len(_TOP20_CLEAN)]
            parts.append(tpl.format(**format_args))
        else:
            tpl = _TOP20_CONCERN[idx % len(_TOP20_CONCERN)]
            parts.append(tpl.format(**format_args))

    elif rank <= 50:
        if not concerns:
            tpl = _TOP50_CLEAN[idx % len(_TOP50_CLEAN)]
            parts.append(tpl.format(**format_args))
        else:
            tpl = _TOP50_CONCERN[idx % len(_TOP50_CONCERN)]
            parts.append(tpl.format(**format_args))

    else:
        if concerns:
            tpl = _TAIL_CONCERN[idx % len(_TAIL_CONCERN)]
            parts.append(tpl.format(concerns=concerns_joined))
        else:
            tpl = _TAIL_CLEAN[idx % len(_TAIL_CLEAN)]
            parts.append(tpl)

    # Clean up multiple periods, ensure spacing
    reasoning = " ".join(parts).replace("..", ".").replace(" .", ".")

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
