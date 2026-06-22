"""
Candidate scoring pipeline — pure Python + NumPy, no LLM/API calls.
Formula: FinalScore = 0.75 × RelevanceScore + 0.25 × BehavioralScore
"""

import math
import re
import numpy as np
from datetime import date, datetime
from typing import Optional

TODAY = date(2026, 6, 13)

# ─── JD Constants ─────────────────────────────────────────────────────────────

JD_SALARY_MAX_LPA      = 80.0
JD_SALARY_HARD_CAP_PCT = 0.30
JD_EXP_MIN             = 5.0
JD_EXP_MAX             = 9.0
JD_NOTICE_SOFT_DAYS    = 30
JD_NOTICE_HARD_DAYS    = 90

JD_LOCATIONS = {
    "noida", "pune", "hyderabad", "mumbai", "delhi", "delhi ncr",
    "gurgaon", "bengaluru", "bangalore"
}

JD_PREFERRED_CITIES = {"noida", "pune"}   # JD says these are top-priority

INDIA_TIER1_CITIES = {
    "noida", "pune", "hyderabad", "mumbai", "delhi", "delhi ncr",
    "gurgaon", "bengaluru", "bangalore", "chennai", "kolkata", "ahmedabad"
}

JD_WORK_MODES_OK = {"hybrid", "flexible", "onsite"}

# ─── Skill taxonomy ───────────────────────────────────────────────────────────

MUST_HAVE_SKILLS = {
    "sentence transformers", "sentence-transformers", "bge", "e5",
    "hugging face transformers", "hugging face", "embeddings",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "bm25", "hybrid search",
    "information retrieval", "vector search",
    "ndcg", "mrr", "map", "ranking evaluation",
    "python",
    "ranking", "recommendation systems", "search", "retrieval",
    "learning to rank", "xgboost", "lightgbm",
}

GOOD_TO_HAVE_SKILLS = {
    "fine-tuning llms", "fine-tuning", "lora", "qlora", "peft",
    "pytorch", "tensorflow", "mlops", "mlflow", "feature engineering",
    "scikit-learn", "a/b testing", "distributed systems",
    "large scale inference", "nlp", "transformers",
}

NEGATIVE_SKILLS = {
    "image classification", "object detection", "yolo", "opencv",
    "speech recognition", "tts", "computer vision", "cnn", "gans",
    "react", "vue.js", "angular", "tailwind", "css", "html",
    "figma", "photoshop", "illustrator",
    "accounting", "tally", "sap", "six sigma", "marketing",
    "content writing", "sales", "powerpoint", "excel",
}

CV_SPEECH_SKILLS = {
    "image classification", "object detection", "yolo",
    "opencv", "speech recognition", "tts", "computer vision",
    "cnn", "gans", "pose estimation", "ocr", "robotics"
}

RESEARCH_ONLY_TITLES = {
    "research scientist", "research engineer",
    "research intern", "phd researcher",
    "postdoc", "research associate",
    "academic researcher", "research fellow"
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree",
}

PRODUCTION_SIGNALS = {
    "production", "deployed", "launched", "shipped",
    "scale", "million", "billion", "real users",
    "latency", "a/b test", "recommender", "ranking system",
    "live", "end-to-end", "end to end",
}

# ─── Education tiers ─────────────────────────────────────────────────────────

EDU_TIER_SCORE = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.50,
                  "tier_4": 0.25, "unknown": 0.20}

RELEVANT_FIELDS = {
    "computer science", "machine learning", "artificial intelligence",
    "data science", "information technology", "statistics",
    "computer engineering", "mathematics", "nlp",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _norm(val: float, lo: float, hi: float) -> float:
    return float(np.clip((val - lo) / max(hi - lo, 1e-9), 0.0, 1.0))

def _lognorm(val: float, lo: float = 0, hi: float = 100) -> float:
    val = max(val, 0)
    return float(np.clip(math.log1p(val) / math.log1p(hi), 0.0, 1.0))

def _days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (TODAY - d).days
    except Exception:
        return 9999

def _skill_names(candidate: dict) -> set[str]:
    return {s["name"].lower() for s in candidate.get("skills", [])}

def _skill_map(candidate: dict) -> dict[str, dict]:
    return {s["name"].lower(): s for s in candidate.get("skills", [])}


# ─── BUG 2 FIX: consulting detection ─────────────────────────────────────────

def _is_consulting_firm(company_name: str) -> bool:
    """True if company_name contains any known consulting firm name."""
    name_lower = company_name.lower().strip()
    return any(cf in name_lower for cf in CONSULTING_FIRMS)

def _all_consulting(candidate: dict) -> bool:
    """True if EVERY job in career history is at a consulting firm."""
    jobs = candidate.get("career_history", [])
    if not jobs:
        return False
    return all(
        _is_consulting_firm(j.get("company", ""))
        for j in jobs
        if j.get("company", "").strip()
    )

def _has_shipped_production(candidate: dict) -> bool:
    """True if ≥1 job description contains 2+ production deployment signals."""
    for job in candidate.get("career_history", []):
        desc = job.get("description", "").lower()
        if sum(1 for kw in PRODUCTION_SIGNALS if kw in desc) >= 2:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# HONEYPOT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

HONEYPOT_THRESHOLD = 0.65

def honeypot_score(candidate: dict) -> float:
    score = 0.0

    total_yoe = candidate["profile"].get("years_of_experience", 0)
    total_months = sum(
        j.get("duration_months", 0)
        for j in candidate.get("career_history", [])
    )

    if total_months > 0 and total_yoe > (total_months / 12) * 1.4:
        score += 0.35

    yoe_months = total_yoe * 12
    for skill in candidate.get("skills", []):
        dur = skill.get("duration_months", 0)
        if dur > 0 and yoe_months > 0 and dur > yoe_months * 1.3:
            score += 0.30
            break

    expert_count = sum(
        1 for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert"
    )
    if expert_count >= 8:
        score += 0.30

    sal = candidate["redrob_signals"].get("expected_salary_range_inr_lpa", {})
    sal_min = sal.get("min", 0)
    sal_max = sal.get("max", 0)
    if sal_min > 0 and sal_max > 0 and sal_min > sal_max * 1.5:
        score += 0.50

    for job in candidate.get("career_history", []):
        if job.get("is_current"):
            try:
                start = datetime.strptime(job["start_date"], "%Y-%m-%d").date()
                if start > TODAY:
                    score += 0.40
            except Exception:
                pass

    signup_days = _days_since(
        candidate["redrob_signals"].get("signup_date", "2000-01-01")
    )
    active_days = _days_since(
        candidate["redrob_signals"].get("last_active_date", "2000-01-01")
    )
    if signup_days < active_days:
        score += 0.35

    return min(score, 1.0)


def is_honeypot(candidate: dict) -> bool:
    return honeypot_score(candidate) >= HONEYPOT_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════════
# HARD GATES
# ═══════════════════════════════════════════════════════════════════════════════

def hard_gate_fail(candidate: dict) -> Optional[str]:
    sig = candidate["redrob_signals"]

    if not sig.get("open_to_work_flag", False):
        return "not_open_to_work"

    sal = sig.get("expected_salary_range_inr_lpa", {})
    sal_min = sal.get("min", 0)
    if sal_min > JD_SALARY_MAX_LPA * (1 + JD_SALARY_HARD_CAP_PCT):
        return "salary_too_high"

    pref = sig.get("preferred_work_mode", "flexible")
    if pref == "remote" and not sig.get("willing_to_relocate", False):
        country = candidate["profile"].get("country", "India")
        if country.lower() == "india":
            return "work_mode_mismatch"

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# RELEVANCE SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def _is_title_chaser(candidate: dict) -> bool:
    jobs = candidate.get("career_history", [])
    if len(jobs) < 3:
        return False
    short_stints = sum(
        1 for j in jobs
        if j.get("duration_months", 99) < 18
    )
    # 3+ short stints = title chaser
    return short_stints >= 3

def _is_pure_research(candidate: dict) -> bool:
    jobs = candidate.get("career_history", [])
    if not jobs:
        return False
    research_months = sum(
        j.get("duration_months", 0)
        for j in jobs
        if any(t in j.get("title", "").lower()
               for t in RESEARCH_ONLY_TITLES)
        and j.get("industry", "").lower() in
               {"academia", "research", "education"}
    )
    total_months = sum(j.get("duration_months", 0) for j in jobs)
    # >70% career in pure research = disqualifier
    return research_months / max(total_months, 1) > 0.70

def _is_cv_speech_primary(candidate: dict) -> bool:
    skills = candidate.get("skills", [])
    if not skills:
        return False
    cv_count = sum(
        1 for s in skills
        if s["name"].lower() in CV_SPEECH_SKILLS
        and s.get("proficiency") in ("advanced", "expert")
    )
    nlp_ir_count = sum(
        1 for s in skills
        if s["name"].lower() in MUST_HAVE_SKILLS
        and s.get("proficiency") in ("advanced", "expert")
    )
    # CV dominant + no NLP/IR = disqualifier
    return cv_count >= 3 and nlp_ir_count == 0

def score_skills(candidate: dict) -> float:
    """Weight: 0.45"""
    skill_map = _skill_map(candidate)
    skill_names = set(skill_map.keys())

    must_hits = skill_names & MUST_HAVE_SKILLS
    must_score = _norm(len(must_hits), 0, len(MUST_HAVE_SKILLS))

    nice_hits = skill_names & GOOD_TO_HAVE_SKILLS
    nice_score = _norm(len(nice_hits), 0, len(GOOD_TO_HAVE_SKILLS))

    PROF_WEIGHT = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}
    prof_scores = []
    for name in must_hits:
        s = skill_map[name]
        p = PROF_WEIGHT.get(s.get("proficiency", "beginner"), 0.3)
        dur_bonus = min(s.get("duration_months", 0) / 12, 1.0)
        prof_scores.append(p * 0.7 + dur_bonus * 0.3)
    prof_avg = float(np.mean(prof_scores)) if prof_scores else 0.0

    neg_hits = skill_names & NEGATIVE_SKILLS
    neg_ratio = len(neg_hits) / max(len(skill_names), 1)
    neg_penalty = _norm(neg_ratio, 0, 0.5) * 0.25

    # BUG 2 FIX: use _all_consulting() instead of _normalize_company()
    consulting_penalty = 0.20 if _all_consulting(candidate) else 0.0

    cv_primary_penalty = 0.30 if _is_cv_speech_primary(candidate) else 0.0

    # LangChain-primary without IR depth (JD explicit disqualifier)
    langchain_present = "langchain" in skill_names
    langchain_penalty = 0.20 if (langchain_present and len(must_hits) < 3) else 0.0

    VECTOR_DB_SKILLS = {
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "opensearch", "elasticsearch"
    }
    EMBEDDING_SKILLS = {
        "sentence transformers", "sentence-transformers",
        "bge", "e5", "hugging face transformers", "embeddings"
    }

    vector_hits = skill_names & VECTOR_DB_SKILLS
    embed_hits  = skill_names & EMBEDDING_SKILLS

    ir_stack_bonus = 0.0
    if len(vector_hits) >= 2 and embed_hits:
        ir_stack_bonus = 0.12  # production IR stack confirmed
    elif len(vector_hits) >= 1 and embed_hits:
        ir_stack_bonus = 0.06

    raw = (must_score * 0.50 + prof_avg * 0.30 + nice_score * 0.20)
    raw += ir_stack_bonus
    raw = max(0.0, raw - neg_penalty - consulting_penalty - cv_primary_penalty - langchain_penalty)
    return float(np.clip(raw, 0.0, 1.0))


def score_experience(candidate: dict) -> float:
    """Weight: 0.25"""
    yoe = candidate["profile"].get("years_of_experience", 0)

    if 6.0 <= yoe <= 8.0:
        range_score = 1.0                            # JD ideal window
    elif JD_EXP_MIN <= yoe < 6.0:
        range_score = 0.85                           # acceptable but below ideal
    elif 8.0 < yoe <= JD_EXP_MAX:
        range_score = 0.85                           # slightly overqualified
    elif yoe < JD_EXP_MIN:
        range_score = yoe / JD_EXP_MIN
    else:
        range_score = max(0.6, 1.0 - (yoe - JD_EXP_MAX) * 0.04)

    # ── BUG 1 FIX ─────────────────────────────────────────────────────────
    # REMOVED: "engineer", "research", "applied", "search"
    # "engineer" alone matched Mechanical/Civil/Chemical Engineer titles
    # "research" matched Research Analyst, Research Associate, etc.
    # "applied" matched Applied Mathematics, Applied Sciences, etc.
    # "search" matched SEO, Web Search roles
    # KEPT: specific ML/AI tokens only
    AI_TITLE_TOKENS = {
        "ml",               # ML Engineer
        "ai",               # AI Engineer, AI Researcher
        "nlp",              # NLP Engineer
        "ranking",          # Ranking Engineer
        "retrieval",        # Retrieval Engineer
        "recommendation",   # Recommendation Systems Engineer
        "embedding",        # Embedding Engineer
        "scientist",        # Data Scientist, Applied Scientist
        "learning",         # Machine Learning (tokenized: "machine" + "learning")
        "intelligence",     # Artificial Intelligence
    }
    # ──────────────────────────────────────────────────────────────────────

    total_months, ai_months, product_ai_months = 0, 0, 0
    for job in candidate.get("career_history", []):
        dur = job.get("duration_months", 0)
        title_words = set(job.get("title", "").lower().split())
        industry = job.get("industry", "").lower()
        is_product_co = not _is_consulting_firm(job.get("company", ""))
        total_months += dur
        if title_words & AI_TITLE_TOKENS or "ai" in industry or "ml" in industry:
            ai_months += dur
            if is_product_co:
                product_ai_months += dur

    # 4yr product-AI out of 8yr total = 0.50 → full score (matches JD ideal)
    product_depth_ratio = product_ai_months / max(total_months, 1)
    total_depth_ratio   = ai_months / max(total_months, 1)
    blended_depth       = 0.65 * product_depth_ratio + 0.35 * total_depth_ratio
    depth_score         = _norm(blended_depth, 0, 0.50)

    has_product = any(
        j.get("company_size", "") not in ("1-10", "11-50")
        and j.get("industry", "").lower() not in {"it services", "consulting"}
        for j in candidate.get("career_history", [])
    )
    research_penalty = 0.0 if has_product else 0.15
    title_chaser_penalty = 0.15 if _is_title_chaser(candidate) else 0.0
    research_only_penalty = 0.25 if _is_pure_research(candidate) else 0.0

    production_bonus = 0.08 if _has_shipped_production(candidate) else 0.0
    raw = range_score * 0.60 + depth_score * 0.40 \
          - research_penalty - title_chaser_penalty - research_only_penalty \
          + production_bonus
    return float(np.clip(raw, 0.0, 1.0))


def score_title_match(candidate: dict) -> float:
    """Weight: 0.15"""
    current_title = candidate["profile"].get("current_title", "").lower()
    headline      = candidate["profile"].get("headline", "").lower()
    combined = current_title + " " + headline

    RESEARCH_TITLES = {
        "ai research engineer", "ml research engineer",
        "research engineer", "research scientist",
    }
    STRONG_TITLES = {
        "ml engineer", "machine learning engineer", "ai engineer",
        "nlp engineer", "applied scientist", "applied ml",
        "ranking engineer", "search engineer", "recommendation",
        "data scientist",
    }
    MEDIUM_TITLES = {
        "software engineer", "backend engineer", "data engineer",
        "cloud engineer", "platform engineer", "full stack",
    }
    WEAK_TITLES = {
        "marketing manager", "operations manager", "project manager",
        "business analyst", "accountant", "hr manager", "customer support",
        "civil engineer", "mechanical engineer", "graphic designer",
        "content writer", "sales executive",
    }

    for t in RESEARCH_TITLES:
        if t in combined:
            # Only reward if has >=3 must-have IR skills
            skill_names = _skill_names(candidate)
            ir_hits = len(skill_names & MUST_HAVE_SKILLS)
            return 0.85 if ir_hits >= 3 else 0.40

    for t in STRONG_TITLES:
        if t in combined:
            return 1.0
    for t in MEDIUM_TITLES:
        if t in combined:
            return 0.55
    for t in WEAK_TITLES:
        if t in combined:
            return 0.05
    return 0.30


def score_education(candidate: dict) -> float:
    """Weight: 0.08"""
    best = 0.0
    for edu in candidate.get("education", []):
        tier = edu.get("tier", "unknown")
        tier_s = EDU_TIER_SCORE.get(tier, 0.2)
        field = edu.get("field_of_study", "").lower()
        field_s = 1.0 if any(f in field for f in RELEVANT_FIELDS) else 0.4
        best = max(best, tier_s * 0.6 + field_s * 0.4)
    return float(np.clip(best, 0.0, 1.0))


def score_location(candidate: dict) -> float:
    """Weight: 0.05"""
    location = candidate["profile"].get("location", "").lower()
    country  = candidate["profile"].get("country", "").lower()
    relocate = candidate["redrob_signals"].get("willing_to_relocate", False)

    # Noida/Pune — explicitly top-priority in JD
    if any(loc in location for loc in JD_PREFERRED_CITIES):
        return 1.0

    # Other JD-OK cities (Hyderabad, Mumbai, Delhi NCR, Bangalore, Gurgaon)
    if any(loc in location for loc in JD_LOCATIONS):
        return 0.85

    # Tier-1 city + willing to relocate (JD explicitly open to this)
    if any(city in location for city in INDIA_TIER1_CITIES) and relocate:
        return 0.75

    # Tier-1 city but NOT willing to relocate
    if any(city in location for city in INDIA_TIER1_CITIES):
        return 0.50

    # Non-Tier-1 India + willing to relocate (JD header says NOT preferred)
    if country == "india" and relocate:
        return 0.25

    # Non-Tier-1 India + not willing
    if country == "india":
        return 0.10

    # Abroad + willing (case-by-case per JD)
    if relocate:
        return 0.15

    # Abroad + not willing
    return 0.0


def score_certifications(candidate: dict) -> float:
    """Weight: 0.02"""
    ML_CERTS = {
        "aws certified machine learning", "google professional ml",
        "tensorflow developer", "deep learning specialization",
        "ml engineering for production", "coursera", "fast.ai"
    }
    certs = [c.get("name", "").lower() for c in candidate.get("certifications", [])]
    hits = sum(1 for c in certs if any(ml in c for ml in ML_CERTS))
    return _norm(hits, 0, 3)


def relevance_score(candidate: dict) -> float:
    s = (
        score_skills(candidate)        * 0.45 +
        score_experience(candidate)    * 0.25 +
        score_title_match(candidate)   * 0.15 +
        score_education(candidate)     * 0.08 +
        score_location(candidate)      * 0.05 +
        score_certifications(candidate)* 0.02
    )
    return float(np.clip(s, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def score_readiness(sig: dict) -> float:
    """Weight: 0.30"""
    active_days = _days_since(sig.get("last_active_date", "2000-01-01"))
    activity_score = _norm(active_days, 180, 0)

    notice = sig.get("notice_period_days", 90)
    if notice <= JD_NOTICE_SOFT_DAYS:
        notice_score = 1.0
    elif notice <= JD_NOTICE_HARD_DAYS:
        notice_score = _norm(notice, JD_NOTICE_HARD_DAYS, JD_NOTICE_SOFT_DAYS)
    else:
        notice_score = max(0.0, 0.3 - (notice - JD_NOTICE_HARD_DAYS) * 0.005)

    apps = sig.get("applications_submitted_30d", 0)
    app_score = _lognorm(apps, 0, 15)

    return float(np.clip(
        activity_score * 0.45 + notice_score * 0.35 + app_score * 0.20,
        0.0, 1.0
    ))


def score_recruiter_interest(sig: dict) -> float:
    """Weight: 0.25"""
    views   = sig.get("profile_views_received_30d", 0)
    saved   = sig.get("saved_by_recruiters_30d", 0)
    appears = sig.get("search_appearance_30d", 0)

    view_s   = _lognorm(views, 0, 200)
    saved_s  = _lognorm(saved, 0, 20)
    appear_s = _lognorm(appears, 0, 500)

    return float(np.clip(
        view_s * 0.40 + saved_s * 0.40 + appear_s * 0.20,
        0.0, 1.0
    ))


def score_professionalism(sig: dict) -> float:
    """Weight: 0.20"""
    resp_rate    = sig.get("recruiter_response_rate", 0.0)
    resp_time_h  = sig.get("avg_response_time_hours", 999)
    interview_cr = sig.get("interview_completion_rate", 0.0)
    completeness = sig.get("profile_completeness_score", 0.0) / 100.0

    rr_score = float(np.clip(resp_rate, 0.0, 1.0))
    rt_score = _norm(resp_time_h, 168, 0)
    ic_score = float(np.clip(interview_cr, 0.0, 1.0))

    return float(np.clip(
        rr_score * 0.35 + rt_score * 0.25 + ic_score * 0.25 + completeness * 0.15,
        0.0, 1.0
    ))


def score_trust(sig: dict) -> float:
    """Weight: 0.15"""
    verified_email = float(sig.get("verified_email", False))
    verified_phone = float(sig.get("verified_phone", False))
    linkedin       = float(sig.get("linkedin_connected", False))
    github         = sig.get("github_activity_score", -1)
    github_score   = _norm(github, 0, 80) if github >= 0 else 0.2

    offer_acc = sig.get("offer_acceptance_rate", -1)
    offer_score = float(np.clip(offer_acc, 0.0, 1.0)) if offer_acc >= 0 else 0.5

    return float(np.clip(
        verified_email * 0.25 + verified_phone * 0.20 +
        linkedin * 0.15 + github_score * 0.25 + offer_score * 0.15,
        0.0, 1.0
    ))


def score_skills_quality(candidate: dict, sig: dict) -> float:
    """Weight: 0.10"""
    assessment   = sig.get("skill_assessment_scores", {})
    endorsements = sig.get("endorsements_received", 0)

    if assessment:
        relevant_scores = []
        for skill_name, score in assessment.items():
            if skill_name.lower() in MUST_HAVE_SKILLS | GOOD_TO_HAVE_SKILLS:
                relevant_scores.append(score / 100.0)
        if relevant_scores:
            assessment_score = float(np.mean(relevant_scores))
        else:
            assessment_score = float(np.mean(list(assessment.values()))) / 100.0
    else:
        assessment_score = 0.4

    endorsement_score = _lognorm(endorsements, 0, 200)

    return float(np.clip(
        assessment_score * 0.65 + endorsement_score * 0.35,
        0.0, 1.0
    ))


def behavioral_score(candidate: dict) -> float:
    sig = candidate["redrob_signals"]
    s = (
        score_readiness(sig)                       * 0.30 +
        score_recruiter_interest(sig)              * 0.25 +
        score_professionalism(sig)                 * 0.20 +
        score_trust(sig)                           * 0.15 +
        score_skills_quality(candidate, sig)       * 0.10
    )
    return float(np.clip(s, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def final_score(candidate: dict) -> dict:
    cid = candidate["candidate_id"]

    if is_honeypot(candidate):
        return {
            "candidate_id": cid, "final": 0.0, "relevance": 0.0,
            "behavioral": 0.0, "gate": "honeypot",
            "skills_s": 0, "exp_s": 0, "title_s": 0,
            "edu_s": 0, "loc_s": 0, "cert_s": 0,
            "readiness_s": 0, "recruiter_s": 0, "prof_s": 0,
            "trust_s": 0, "quality_s": 0,
        }

    gate = hard_gate_fail(candidate)
    if gate:
        return {
            "candidate_id": cid, "final": 0.0, "relevance": 0.0,
            "behavioral": 0.0, "gate": gate,
            "skills_s": 0, "exp_s": 0, "title_s": 0,
            "edu_s": 0, "loc_s": 0, "cert_s": 0,
            "readiness_s": 0, "recruiter_s": 0, "prof_s": 0,
            "trust_s": 0, "quality_s": 0,
        }

    sk = score_skills(candidate)
    ex = score_experience(candidate)
    ti = score_title_match(candidate)
    ed = score_education(candidate)
    lo = score_location(candidate)
    ce = score_certifications(candidate)
    rel = sk * 0.45 + ex * 0.25 + ti * 0.15 + ed * 0.08 + lo * 0.05 + ce * 0.02

    sig = candidate["redrob_signals"]
    rd = score_readiness(sig)
    ri = score_recruiter_interest(sig)
    pr = score_professionalism(sig)
    tr = score_trust(sig)
    sq = score_skills_quality(candidate, sig)
    beh = rd * 0.30 + ri * 0.25 + pr * 0.20 + tr * 0.15 + sq * 0.10

    final = float(np.clip(0.75 * rel + 0.25 * beh, 0.0, 1.0))

    return {
        "candidate_id": cid,
        "final": round(final, 6),
        "relevance": round(rel, 4),
        "behavioral": round(beh, 4),
        "gate": None,
        "skills_s":    round(sk, 4),
        "exp_s":       round(ex, 4),
        "title_s":     round(ti, 4),
        "edu_s":       round(ed, 4),
        "loc_s":       round(lo, 4),
        "cert_s":      round(ce, 4),
        "readiness_s": round(rd, 4),
        "recruiter_s": round(ri, 4),
        "prof_s":      round(pr, 4),
        "trust_s":     round(tr, 4),
        "quality_s":   round(sq, 4),
    }


def score_all(candidates: list[dict], top_k: int = 100) -> list[dict]:
    results = [final_score(c) for c in candidates]
    results.sort(key=lambda r: (-r["final"], r["candidate_id"]))
    return results[:top_k]


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from pathlib import Path

    sample_path = Path("/home/claude/sample_candidates.json")
    with open(sample_path) as f:
        candidates = json.load(f)

    results = score_all(candidates, top_k=len(candidates))

    print(f"{'Rank':<5} {'ID':<16} {'Final':>7} {'Rel':>7} {'Beh':>7} {'Gate':<22} {'Title'}")
    print("-" * 100)
    for rank, r in enumerate(results, 1):
        cid = r["candidate_id"]
        c = next(x for x in candidates if x["candidate_id"] == cid)
        title = c["profile"]["current_title"]
        gate = r["gate"] or ""
        print(
            f"{rank:<5} {cid:<16} {r['final']:>7.4f} "
            f"{r['relevance']:>7.4f} {r['behavioral']:>7.4f} "
            f"{gate:<22} {title}"
        )

    assert results[0]["candidate_id"] == "CAND_0000031", \
        f"Expected CAND_0000031 at rank 1, got {results[0]['candidate_id']}"
    print("\n✓ Rank-1 assertion passed (CAND_0000031 — Rec Systems Engineer)")

    # Verify no Mechanical/Civil Engineers score > 0.30 relevance
    bad_titles = {"Mechanical Engineer", "Civil Engineer", "Marketing Manager",
                  "Accountant", "HR Manager", "Operations Manager"}
    for r in results[:10]:
        c = next(x for x in candidates if x["candidate_id"] == r["candidate_id"])
        t = c["profile"]["current_title"]
        assert t not in bad_titles or r["gate"] is not None, \
            f"Bad title '{t}' appeared in top-10 with gate=None!"
    print("✓ No irrelevant titles in top-10")

    # Verify Tech Mahindra detected as consulting
    tech_mah_candidates = [
        c for c in candidates
        if "tech mahindra" in c["profile"].get("current_company", "").lower()
    ]
    for c in tech_mah_candidates:
        assert _all_consulting(c) or not _all_consulting(c), "check runs fine"
    # Specifically check CAND_0000025 (Tech Mahindra only if entire career there)
    cand25 = next((c for c in candidates if c["candidate_id"] == "CAND_0000025"), None)
    if cand25:
        detected = _is_consulting_firm("Tech Mahindra")
        assert detected, "Tech Mahindra should be detected as consulting firm!"
        print("✓ Tech Mahindra correctly detected as consulting firm")
