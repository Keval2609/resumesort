
import logging
import numpy as np
from typing import Optional

log = logging.getLogger("reranker")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURABLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SEMANTIC_WEIGHT         = 0.15   # hybrid = SCORER_WEIGHT * scorer + SEMANTIC_WEIGHT * semantic
SCORER_WEIGHT           = 0.85
TOP_N_TO_RERANK         = 300    # only rerank this many from scorer output
MIN_SCORE_FOR_RERANK    = 0.30   # candidates below this are excluded from benefit
SCORE_WINDOW_FOR_RERANK = 0.08   # max score gap for reordering (prevents wild jumps)
MODEL_NAME              = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_MODEL_NAME     = "BAAI/bge-small-en-v1.5"

# ── JD embedding text ────────────────────────────────────────────────────────
JD_EMBEDDING_TEXT = (
    "Senior AI Engineer specializing in information retrieval, ranking, "
    "and recommendation systems. Expertise in vector search, dense retrieval, "
    "semantic search, FAISS, Elasticsearch, sentence transformers, BM25, "
    "hybrid search, learning to rank, NDCG, MRR evaluation. "
    "Production ML systems at product companies. "
    "Python, embeddings, fine-tuning, MLOps. "
    "5-9 years experience, founding team member."
)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (lazy, with graceful fallback)
# ═══════════════════════════════════════════════════════════════════════════════

_model = None
_model_loaded = False


import sys

def _load_model():
    """
    Lazy-load sentence-transformers model.
    Returns the model. Exits with non-zero if unavailable.
    """
    global _model, _model_loaded
    if _model_loaded:
        return _model

    _model_loaded = True

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Exiting.")
        sys.exit(1)

    log.info("Loading embedding model from local cache...\n")

    try:
        _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        log.info(f"Loaded:\n{MODEL_NAME}\n\nSemantic reranking enabled.")
        return _model
    except Exception as e_primary:
        log.info(f"Primary model not found locally.\n\nLoading fallback model...")
        try:
            _model = SentenceTransformer(FALLBACK_MODEL_NAME, local_files_only=True)
            log.info(f"Loaded:\n{FALLBACK_MODEL_NAME}")
            return _model
        except Exception as e_fallback:
            log.error("No local embedding model found.\n\nSemantic reranking disabled.\n\nRun:\n\npython download_models.py\n")
            sys.stderr.write(
                "ERROR: No local embedding model found.\n\n"
                "Semantic reranking is required to reproduce the submitted ranking.\n\n"
                "Please run:\n\n"
                "python download_models.py\n\n"
                "and retry.\n"
            )
            sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_candidate_text_for_embedding(candidate: dict) -> str:
    """
    Build compact text representation for embedding.
    Shorter than BM25 text — focused on semantic meaning.
    """
    profile = candidate.get("profile", {})
    parts = []

    # Title and headline (most semantically meaningful)
    title = profile.get("current_title", "")
    headline = profile.get("headline", "")
    if title:
        parts.append(title)
    if headline and headline != title:
        parts.append(headline)

    # Skills (comma-separated)
    skills = [s["name"] for s in candidate.get("skills", [])[:15]]
    if skills:
        parts.append(", ".join(skills))

    # Latest job description (truncated for efficiency)
    jobs = candidate.get("career_history", [])
    if jobs:
        latest_desc = jobs[0].get("description", "")
        if latest_desc:
            parts.append(latest_desc[:250])

    # Summary (truncated)
    summary = profile.get("summary", "")
    if summary:
        parts.append(summary[:150])

    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING CACHE
# ═══════════════════════════════════════════════════════════════════════════════

_embedding_cache: dict[str, np.ndarray] = {}
_query_embedding: Optional[np.ndarray] = None


def _get_query_embedding(model) -> np.ndarray:
    """Compute and cache the JD query embedding."""
    global _query_embedding
    if _query_embedding is None:
        _query_embedding = model.encode(
            JD_EMBEDDING_TEXT,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return _query_embedding


def _get_candidate_embeddings(
    model,
    candidates: list[dict],
) -> dict[str, np.ndarray]:
    """
    Compute candidate embeddings, using cache for already-seen candidates.
    Returns dict mapping candidate_id → embedding vector.
    """
    global _embedding_cache

    # Separate cached vs uncached
    uncached_ids = []
    uncached_texts = []
    for c in candidates:
        cid = c["candidate_id"]
        if cid not in _embedding_cache:
            uncached_ids.append(cid)
            uncached_texts.append(build_candidate_text_for_embedding(c))

    # Batch encode uncached
    if uncached_texts:
        log.info(f"Encoding {len(uncached_texts)} candidate embeddings")
        embeddings = model.encode(
            uncached_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        for cid, emb in zip(uncached_ids, embeddings):
            _embedding_cache[cid] = emb

    # Return all requested
    result = {}
    for c in candidates:
        cid = c["candidate_id"]
        result[cid] = _embedding_cache[cid]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# COSINE SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (assumes normalized)."""
    return float(np.dot(a, b))


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC RERANKING
# ═══════════════════════════════════════════════════════════════════════════════

def semantic_rerank(
    scored_results: list[dict],
    candidates_by_id: dict[str, dict],
    top_n: int = TOP_N_TO_RERANK,
) -> list[dict]:
    """
    Rerank top-N candidates using hybrid scorer + semantic similarity.

    Constraints:
      - Only reranks top_n candidates.
      - Candidates with score < MIN_SCORE_FOR_RERANK get no semantic benefit.
      - Candidates can only be reordered within SCORE_WINDOW_FOR_RERANK
        of their original position's score.
      - Remaining candidates (beyond top_n) pass through unchanged.

    Returns the full list, re-sorted.
    """
    model = _load_model()
    if model is None:
        log.info("Semantic reranking skipped (no model available)")
        return scored_results

    if len(scored_results) == 0:
        return scored_results

    # Split into rerank pool and tail
    rerank_pool = scored_results[:top_n]
    tail = scored_results[top_n:]

    # Filter out gated/honeypot candidates from reranking benefit
    eligible = [r for r in rerank_pool if r.get("gate") is None]
    ineligible = [r for r in rerank_pool if r.get("gate") is not None]

    if not eligible:
        return scored_results

    # Get candidate objects for eligible
    eligible_candidates = []
    for r in eligible:
        cid = r["candidate_id"]
        if cid in candidates_by_id:
            eligible_candidates.append(candidates_by_id[cid])

    if not eligible_candidates:
        return scored_results

    # Compute embeddings
    query_emb = _get_query_embedding(model)
    cand_embeddings = _get_candidate_embeddings(model, eligible_candidates)

    # Normalize scorer scores within the pool for blending
    scores = [r["final"] for r in eligible]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    score_range = max(max_score - min_score, 1e-9)

    # Compute hybrid scores
    for r in eligible:
        cid = r["candidate_id"]
        scorer_norm = (r["final"] - min_score) / score_range

        if r["final"] < MIN_SCORE_FOR_RERANK:
            # Weak candidates: no semantic benefit
            r["_hybrid"] = r["final"]
            r["semantic_sim"] = 0.0
            continue

        if cid in cand_embeddings:
            sim = _cosine_similarity(query_emb, cand_embeddings[cid])
            # Clamp similarity to [0, 1]
            sim = max(0.0, min(1.0, sim))
        else:
            sim = 0.0

        hybrid_norm = SCORER_WEIGHT * scorer_norm + SEMANTIC_WEIGHT * sim
        # Map back to original score range
        hybrid = min_score + hybrid_norm * score_range

        r["_hybrid"] = hybrid
        r["semantic_sim"] = round(sim, 4)

    # Sort by hybrid score
    eligible.sort(key=lambda r: (-r["_hybrid"], r["candidate_id"]))

    # ── Enforce narrow-window constraint ──────────────────────────────────
    # A candidate cannot jump past another if their original score
    # difference exceeds SCORE_WINDOW_FOR_RERANK
    original_order = {r["candidate_id"]: i for i, r in enumerate(
        sorted(eligible, key=lambda r: (-r["final"], r["candidate_id"]))
    )}

    # Simple enforcement: if a candidate jumped more than SCORE_WINDOW
    # past a candidate that was originally above it, swap them back
    stable = True
    for i in range(len(eligible)):
        for j in range(i + 1, min(i + 10, len(eligible))):
            r_i = eligible[i]
            r_j = eligible[j]
            # If j was originally above i, and their score gap was large,
            # this reordering is invalid
            if (original_order[r_j["candidate_id"]] < original_order[r_i["candidate_id"]]
                    and r_j["final"] - r_i["final"] > SCORE_WINDOW_FOR_RERANK):
                eligible[i], eligible[j] = eligible[j], eligible[i]
                stable = False

    if not stable:
        log.debug("Applied narrow-window constraint corrections")

    # Update final scores to hybrid scores (preserve monotonicity)
    for r in eligible:
        r["final"] = round(r["_hybrid"], 6)
        if "_hybrid" in r:
            del r["_hybrid"]

    # Recombine: eligible + ineligible (sorted), then tail
    combined_pool = eligible + ineligible
    combined_pool.sort(key=lambda r: (-r["final"], r["candidate_id"]))

    result = combined_pool + tail
    log.info(f"Semantic reranking complete: reranked {len(eligible)} candidates")

    return result
