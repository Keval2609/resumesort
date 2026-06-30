
import logging
import re
from typing import Optional

log = logging.getLogger("synergy")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURABLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Positive synergy bonuses ──────────────────────────────────────────────────
SYNERGY_VECTOR_PRODUCT      = 0.05   # vector search + product company + ≥5yr
SYNERGY_RETRIEVAL_PRODUCT   = 0.04   # retrieval + product company + 5–9yr
SYNERGY_RANKING_STACK       = 0.03   # ranking + recommendation + python
SYNERGY_SEARCH_STACK        = 0.03   # sentence transformers + FAISS + elasticsearch

# ── Production evidence ──────────────────────────────────────────────────────
PRODUCTION_EVIDENCE_BONUS   = 0.04   # product company + shipped retrieval/rec system
PRODUCTION_EVIDENCE_GENERIC = 0.02   # product company + shipped any ML system

# ── Career evidence ──────────────────────────────────────────────────────────
CAREER_EVIDENCE_BONUS       = 0.03   # max career evidence bonus
CAREER_OWNERSHIP_WEIGHT     = 0.40   # weight for ownership signals
CAREER_PROGRESSION_WEIGHT   = 0.30   # weight for career progression
CAREER_TENURE_WEIGHT        = 0.30   # weight for product-company tenure

# ── Keyword stuffing ─────────────────────────────────────────────────────────
KEYWORD_STUFFING_PENALTY    = 0.08   # penalty when title_s < 0.30 and skills_s > 0.90

# ── Negative synergies ───────────────────────────────────────────────────────
PENALTY_CONSULTING_NO_PRODUCT   = 0.02
PENALTY_KEYWORD_TITLE_MISMATCH  = 0.02
PENALTY_HIGH_EXP_NO_RETRIEVAL   = 0.015
PENALTY_JUNIOR_NO_CORE          = 0.015
NEGATIVE_SYNERGY_CAP            = 0.05   # total negative synergy never exceeds this

# ── Relative pool ────────────────────────────────────────────────────────────
RELATIVE_POOL_BOOST         = 0.02

# ═══════════════════════════════════════════════════════════════════════════════
# SKILL / DOMAIN TAXONOMIES (for synergy checks)
# ═══════════════════════════════════════════════════════════════════════════════

VECTOR_SEARCH_SKILLS = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "vector search", "vector database", "ann search", "hnsw",
}

RETRIEVAL_SKILLS = {
    "information retrieval", "retrieval", "bm25", "hybrid search",
    "dense retrieval", "semantic search", "opensearch", "elasticsearch",
    "retrieval pipeline", "search relevance", "retrieval optimization",
}

RANKING_SKILLS = {
    "ranking", "learning to rank", "ndcg", "mrr", "map",
    "ranking evaluation", "candidate ranking", "reranking",
}

RECOMMENDATION_SKILLS = {
    "recommendation systems", "recommendation engine", "recommender",
    "personalization", "collaborative filtering",
}

EMBEDDING_SKILLS = {
    "sentence transformers", "sentence-transformers", "bge", "e5",
    "embeddings", "hugging face transformers", "hugging face",
}

CORE_RETRIEVAL_SKILLS = (
    VECTOR_SEARCH_SKILLS | RETRIEVAL_SKILLS | RANKING_SKILLS
    | RECOMMENDATION_SKILLS | EMBEDDING_SKILLS
)

# ── Production ownership domains (job description text matching) ──────────────
PRODUCTION_OWNERSHIP_PATTERNS = {
    "retrieval_system": [
        r"(?:built|designed|architected|owned|led|shipped)\b.{0,40}\b(?:retrieval|search)\s+(?:system|pipeline|platform|engine|infrastructure)",
        r"(?:retrieval|search)\s+(?:system|pipeline|platform|engine)\b.{0,40}\b(?:production|deployed|launched|shipped|live|scale)",
    ],
    "ranking_system": [
        r"(?:built|designed|architected|owned|led|shipped)\b.{0,40}\b(?:ranking|relevance)\s+(?:system|model|pipeline|engine)",
        r"(?:ranking|relevance)\s+(?:system|model|pipeline)\b.{0,40}\b(?:production|deployed|launched|shipped|live|scale)",
    ],
    "recommendation_engine": [
        r"(?:built|designed|architected|owned|led|shipped)\b.{0,40}\b(?:recommend|personalization)\w*\s+(?:system|engine|platform|pipeline)",
        r"(?:recommend|personalization)\w*\s+(?:system|engine|platform)\b.{0,40}\b(?:production|deployed|launched|shipped|live|scale)",
    ],
    "search_platform": [
        r"(?:built|designed|architected|owned|led|shipped)\b.{0,40}\b(?:search)\s+(?:platform|infrastructure|service|engine)",
        r"(?:search)\s+(?:platform|infrastructure|service)\b.{0,40}\b(?:production|deployed|launched|shipped|live|users|scale)",
    ],
    "vector_search": [
        r"(?:built|designed|architected|owned|led|shipped)\b.{0,40}\b(?:vector|embedding|semantic)\s+(?:search|index|retrieval|store)",
        r"(?:vector|embedding|semantic)\s+(?:search|index|retrieval)\b.{0,40}\b(?:production|deployed|launched|shipped|live|scale)",
    ],
}

# ── Career progression title hierarchy ────────────────────────────────────────
SENIORITY_LEVELS = {
    "intern": 0, "trainee": 0, "fresher": 0,
    "junior": 1, "associate": 1,
    "engineer": 2, "developer": 2, "analyst": 2,
    "senior": 3, "lead": 4, "staff": 4, "principal": 5,
    "manager": 4, "director": 5, "head": 5, "vp": 6,
    "founding": 5, "co-founder": 6, "cto": 6,
}

# ── Consulting firms (shared with scorer.py) ──────────────────────────────────
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _skill_names(candidate: dict) -> set[str]:
    """Extract lowercase skill names from candidate."""
    return {s["name"].lower() for s in candidate.get("skills", [])}


def _is_consulting_firm(company_name: str) -> bool:
    name_lower = company_name.lower().strip()
    return any(cf in name_lower for cf in CONSULTING_FIRMS)


def _all_consulting(candidate: dict) -> bool:
    """True if every job in career history is at a consulting firm."""
    jobs = candidate.get("career_history", [])
    if not jobs:
        return False
    return all(
        _is_consulting_firm(j.get("company", ""))
        for j in jobs
        if j.get("company", "").strip()
    )


def _has_product_company_experience(candidate: dict) -> bool:
    """True if at least one job is at a non-consulting company."""
    for job in candidate.get("career_history", []):
        company = job.get("company", "")
        industry = job.get("industry", "").lower()
        if (company.strip()
                and not _is_consulting_firm(company)
                and industry not in {"it services", "consulting"}):
            return True
    return False


def _product_company_months(candidate: dict) -> int:
    """Total months at product (non-consulting) companies."""
    return sum(
        j.get("duration_months", 0)
        for j in candidate.get("career_history", [])
        if j.get("company", "").strip()
        and not _is_consulting_firm(j.get("company", ""))
        and j.get("industry", "").lower() not in {"it services", "consulting"}
    )


def _effective_yoe(candidate: dict) -> float:
    profile_yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    career_months = sum(
        j.get("duration_months", 0)
        for j in candidate.get("career_history", [])
    )
    return max(profile_yoe, career_months / 12.0)


def _count_retrieval_domain_skills(candidate: dict) -> int:
    """Count how many core retrieval/ranking/search skills the candidate has."""
    names = _skill_names(candidate)
    return len(names & CORE_RETRIEVAL_SKILLS)


# ═══════════════════════════════════════════════════════════════════════════════
# 1A. POSITIVE SYNERGY BONUSES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_synergy_bonus(candidate: dict, scores: dict) -> float:
    """
    Reward complete skill/experience combinations.
    Only fires when ALL conditions in a combination are met.
    Returns total additive bonus.
    """
    skill_names = _skill_names(candidate)
    yoe = _effective_yoe(candidate)
    has_product = _has_product_company_experience(candidate)
    bonus = 0.0

    # ── Retrieval Stack: vector search + product company + ≥5yr ───────────
    has_vector = bool(skill_names & VECTOR_SEARCH_SKILLS)
    if has_vector and has_product and yoe >= 5.0:
        bonus += SYNERGY_VECTOR_PRODUCT
        log.debug("Applied retrieval-stack synergy (vector + product + 5yr+)")

    # ── Product Retrieval: retrieval + product company + 5–9yr ────────────
    has_retrieval = bool(skill_names & RETRIEVAL_SKILLS)
    if has_retrieval and has_product and 5.0 <= yoe <= 9.0:
        bonus += SYNERGY_RETRIEVAL_PRODUCT
        log.debug("Applied product-retrieval synergy (retrieval + product + 5-9yr)")

    # ── Ranking Stack: ranking + recommendation + python ──────────────────
    has_ranking = bool(skill_names & RANKING_SKILLS)
    has_rec = bool(skill_names & RECOMMENDATION_SKILLS)
    has_python = "python" in skill_names
    if has_ranking and has_rec and has_python:
        bonus += SYNERGY_RANKING_STACK
        log.debug("Applied ranking-stack synergy (ranking + rec + python)")

    # ── Search Stack: sentence transformers + FAISS + elasticsearch ───────
    has_embeddings = bool(skill_names & EMBEDDING_SKILLS)
    has_faiss = "faiss" in skill_names
    has_es = bool(skill_names & {"elasticsearch", "opensearch"})
    if has_embeddings and has_faiss and has_es:
        bonus += SYNERGY_SEARCH_STACK
        log.debug("Applied search-stack synergy (embeddings + FAISS + ES)")

    return bonus


# ═══════════════════════════════════════════════════════════════════════════════
# 1B. PRODUCTION EVIDENCE BONUS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_production_evidence_bonus(candidate: dict) -> float:
    """
    Reward candidates who shipped production retrieval/rec/search systems
    at product companies.  Evaluates achievement quality, not just keywords.
    """
    if not _has_product_company_experience(candidate):
        return 0.0

    # Check job descriptions at product companies for ownership patterns
    domain_hits = set()
    for job in candidate.get("career_history", []):
        company = job.get("company", "")
        if _is_consulting_firm(company):
            continue

        desc = job.get("description", "").lower()
        if not desc:
            continue

        for domain, patterns in PRODUCTION_OWNERSHIP_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, desc):
                    domain_hits.add(domain)
                    break

    if not domain_hits:
        return 0.0

    # Multiple domain hits indicate deeper production ownership
    if len(domain_hits) >= 2:
        bonus = PRODUCTION_EVIDENCE_BONUS
        log.debug(f"Applied production evidence bonus for {domain_hits}")
    else:
        bonus = PRODUCTION_EVIDENCE_GENERIC
        log.debug(f"Applied generic production bonus for {domain_hits}")

    return bonus


# ═══════════════════════════════════════════════════════════════════════════════
# 1C. CAREER EVIDENCE BONUS
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_ownership_score(candidate: dict) -> float:
    """
    Evaluate quality of career achievements — prioritize retrieval/search/rec
    ownership over generic engineering work.

    Returns 0.0–1.0.
    """
    # High-value ownership patterns (retrieval/ranking/rec/search specific)
    HIGH_VALUE_PATTERNS = [
        r"(?:built|designed|architected|owned|led)\b.{0,50}\b(?:retrieval|ranking|recommendation|search|embedding|vector)\b",
        r"(?:end.to.end|e2e|full.stack)\b.{0,30}\b(?:retrieval|ranking|recommendation|search)\b",
        r"(?:improved|optimized|increased)\b.{0,30}\b(?:ndcg|mrr|relevance|precision|recall|latency)\b",
        r"(?:production|deployed|shipped|launched)\b.{0,30}\b(?:model|pipeline|system|engine|platform)\b",
        r"(?:millions?|billions?|10k|100k|1m)\b.{0,20}\b(?:users?|queries|requests|documents)\b",
    ]

    total_high_value = 0
    for job in candidate.get("career_history", []):
        desc = job.get("description", "").lower()
        if not desc:
            continue
        for pattern in HIGH_VALUE_PATTERNS:
            if re.search(pattern, desc):
                total_high_value += 1
                break  # one hit per job is enough

    # Normalize: 0 hits → 0.0, 1 hit → 0.5, 2+ hits → 1.0
    if total_high_value >= 2:
        return 1.0
    elif total_high_value == 1:
        return 0.5
    return 0.0


def _compute_progression_score(candidate: dict) -> float:
    """
    Evaluate career progression — increasing seniority across jobs.
    Returns 0.0–1.0.
    """
    jobs = candidate.get("career_history", [])
    if len(jobs) < 2:
        return 0.3  # can't evaluate progression with 1 job

    levels = []
    for job in jobs:
        title = job.get("title", "").lower()
        best_level = -1
        for token, level in SENIORITY_LEVELS.items():
            if token in title:
                best_level = max(best_level, level)
        if best_level >= 0:
            levels.append(best_level)

    if len(levels) < 2:
        return 0.3

    # Check if generally ascending (later jobs = higher level)
    # Jobs are typically newest-first in career_history
    ascending_pairs = 0
    total_pairs = 0
    for i in range(len(levels) - 1):
        total_pairs += 1
        if levels[i] >= levels[i + 1]:  # newer job >= older job = progression
            ascending_pairs += 1

    if total_pairs == 0:
        return 0.3

    return min(1.0, ascending_pairs / total_pairs)


def _compute_tenure_score(candidate: dict) -> float:
    """
    Evaluate product-company tenure depth.
    ≥36 months at product companies → full score.
    Returns 0.0–1.0.
    """
    months = _product_company_months(candidate)
    if months >= 48:
        return 1.0
    elif months >= 36:
        return 0.85
    elif months >= 24:
        return 0.6
    elif months >= 12:
        return 0.35
    return 0.0


def compute_career_evidence_bonus(candidate: dict) -> float:
    """
    Combined career evidence score based on ownership quality,
    career progression, and product-company tenure.
    Returns additive bonus in [0, CAREER_EVIDENCE_BONUS].
    """
    ownership = _compute_ownership_score(candidate)
    progression = _compute_progression_score(candidate)
    tenure = _compute_tenure_score(candidate)

    combined = (
        ownership * CAREER_OWNERSHIP_WEIGHT
        + progression * CAREER_PROGRESSION_WEIGHT
        + tenure * CAREER_TENURE_WEIGHT
    )

    bonus = combined * CAREER_EVIDENCE_BONUS

    if bonus > 0.005:
        log.debug(
            f"Career evidence bonus: {bonus:.4f} "
            f"(ownership={ownership:.2f}, progression={progression:.2f}, "
            f"tenure={tenure:.2f})"
        )

    return bonus


# ═══════════════════════════════════════════════════════════════════════════════
# 1D. KEYWORD STUFFING PENALTY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_keyword_stuffing_penalty(title_score: float,
                                      skills_score: float) -> float:
    """
    Penalize profiles with high skill scores but unrelated titles.
    Only triggers on obvious keyword stuffing.
    Returns additive penalty (positive number to subtract).
    """
    if title_score < 0.30 and skills_score > 0.90:
        log.debug(
            f"Keyword stuffing detected: title_s={title_score:.2f}, "
            f"skills_s={skills_score:.2f}"
        )
        return KEYWORD_STUFFING_PENALTY
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 1E. NEGATIVE SYNERGIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_negative_synergies(candidate: dict, scores: dict) -> float:
    """
    Apply conservative penalties for poor evidence combinations.
    Total penalty is capped at NEGATIVE_SYNERGY_CAP to avoid
    excessively suppressing borderline candidates.
    Returns total penalty (positive number to subtract).
    """
    skill_names = _skill_names(candidate)
    title_s = scores.get("title_s", 0.5)
    skills_s = scores.get("skills_s", 0.5)
    exp_s = scores.get("exp_s", 0.5)
    yoe = _effective_yoe(candidate)
    penalty = 0.0

    # ── Consulting-only career with no product experience ─────────────────
    if _all_consulting(candidate) and not _has_product_company_experience(candidate):
        penalty += PENALTY_CONSULTING_NO_PRODUCT
        log.debug("Negative synergy: consulting-only career")

    # ── Strong skill list but completely unrelated title ───────────────────
    if skills_s > 0.70 and title_s < 0.30:
        penalty += PENALTY_KEYWORD_TITLE_MISMATCH
        log.debug("Negative synergy: skills/title mismatch")

    # ── High experience but no retrieval/search evidence ──────────────────
    retrieval_count = _count_retrieval_domain_skills(candidate)
    if yoe > 8.0 and retrieval_count < 2:
        penalty += PENALTY_HIGH_EXP_NO_RETRIEVAL
        log.debug("Negative synergy: high YOE but no retrieval depth")

    # ── Junior with missing core retrieval skills ─────────────────────────
    must_have_core = skill_names & (RETRIEVAL_SKILLS | RANKING_SKILLS | VECTOR_SEARCH_SKILLS)
    if yoe < 3.0 and len(must_have_core) < 2:
        penalty += PENALTY_JUNIOR_NO_CORE
        log.debug("Negative synergy: junior with missing core skills")

    # Cap total negative synergy
    capped = min(penalty, NEGATIVE_SYNERGY_CAP)
    if capped < penalty:
        log.debug(
            f"Negative synergy capped: {penalty:.4f} → {capped:.4f}"
        )

    return capped


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def apply_synergies(base_score: float, candidate: dict,
                    scores: dict) -> dict:
    """
    Orchestrate all synergy computations and return adjusted score.

    Uses ADDITIVE scoring:
        final = base + synergy_bonus + production_bonus + career_bonus
                - keyword_penalty - negative_synergies

    Returns dict with:
        adjusted_score: float — clamped to [0, 1]
        synergy_bonus: float  — total positive synergy
        production_bonus: float
        career_bonus: float
        penalties: float      — total penalties applied
    """
    # Positive bonuses
    synergy = compute_synergy_bonus(candidate, scores)
    production = compute_production_evidence_bonus(candidate)
    career = compute_career_evidence_bonus(candidate)

    # Penalties
    keyword_pen = compute_keyword_stuffing_penalty(
        scores.get("title_s", 0.5),
        scores.get("skills_s", 0.5),
    )
    negative_syn = compute_negative_synergies(candidate, scores)
    total_penalties = keyword_pen + negative_syn

    # Additive combination
    adjusted = (
        base_score
        + synergy
        + production
        + career
        - total_penalties
    )

    adjusted = max(0.0, min(1.0, adjusted))

    total_bonus = synergy + production + career
    if total_bonus > 0.005 or total_penalties > 0.005:
        log.debug(
            f"Synergy result: base={base_score:.4f} "
            f"+bonus={total_bonus:.4f} -penalty={total_penalties:.4f} "
            f"= {adjusted:.4f}"
        )

    return {
        "adjusted_score": adjusted,
        "synergy_bonus": round(synergy, 6),
        "production_bonus": round(production, 6),
        "career_bonus": round(career, 6),
        "penalties": round(total_penalties, 6),
    }
