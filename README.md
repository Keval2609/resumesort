# Redrob Hackathon — Intelligent Candidate Ranking

**Team:** SignalOverNoise-108  
**Challenge:** Intelligent Candidate Discovery & Ranking  
**Target Role:** Senior AI Engineer — Founding Team  

---

## Overview

Deterministic, fully local candidate ranking pipeline for 100K profiles.  
No LLMs, no GPUs, no network calls during ranking.

Pipeline: **BM25 pre-filter → Hard gates → Honeypot detection → Full scoring → Ranked CSV**  
Formula: `FinalScore = 0.75 × RelevanceScore + 0.25 × BehavioralScore`

---

## Repository Structure

```
resumesort/
├── rank.py                   # Main CLI — orchestrates the full pipeline
├── app.py                    # Streamlit sandbox (≤100 candidates, interactive)
├── bm25.py                   # BM25 inverted index for fast pre-filtering
├── scorer.py                 # Core scorer (Relevance + Behavioral) — production version
├── honeypot.py               # Synthetic profile detector (standalone CLI + library)
├── reasoning.py              # Stage 4-compliant reasoning generator
├── submission_metadata.yaml  # Team metadata, methodology, compute declarations
├── requirements.txt          # Python dependencies (numpy, pandas, streamlit)
└── .gitignore
```

---

## Quickstart

### 1. Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Full Pipeline (CLI)

```bash
python rank.py --candidates ./candidates.jsonl --out ./SignalOverNoise-108.csv
```

Produces a spec-compliant CSV: 100 rows, header, UTF-8, monotone scores.  
**Runtime:** ~2 min on a 16 GB CPU-only machine.

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--candidates` | `./data/candidates.jsonl.gz` | Path to `.jsonl` or `.jsonl.gz` |
| `--out` | `./final_output.csv` | Output CSV path |
| `--sample N` | — | Score only first N candidates (local testing) |

### 3. Interactive Sandbox (Streamlit)

```bash
streamlit run app.py
```

Accepts a `.json` array or `.jsonl` file (≤100 candidates). Runs the full pipeline and outputs a downloadable ranked CSV.  
Live deployment: [resumesort-signalovernoise-108.streamlit.app](https://resumesort-signalovernoise-108.streamlit.app/)

### 4. Validate Output

```bash
python validate_submission.py SignalOverNoise-108.csv
```

Checks: 100 rows, correct header, unique ranks 1–100, unique candidate IDs, monotone scores, valid CAND_ IDs.

---

## Pipeline Architecture

### Stage 1 — BM25 Pre-filter

All 100K profiles are flattened into weighted text blobs. High-value fields (title, headline) are repeated 3×; core ML skills get extra TF weight. An in-memory inverted index retrieves the **top 5,000 BM25 candidates** against the JD query.

### Stage 2 — Hard Gates

Candidates are immediately excluded (score = 0.0) if they fail any gate:

| Gate | Condition | Effect |
|------|-----------|--------|
| Not open to work | `open_to_work_flag = False` | Score = 0.0 (excluded) |
| Work mode mismatch | `preferred_work_mode = remote` AND `willing_to_relocate = False` AND `country = India` | Score = 0.0 (excluded) |

### Stage 3 — Honeypot Detection

`scorer.py` runs an independent honeypot scorer (threshold = **0.65**). Flagged candidates are excluded before ranking. Detection signals include:

- Skill `duration_months` > (total YOE in months + 48) (allows a 4-year buffer for self-taught developers)
- Chronological career span >> declared YOE (handles concurrent roles without false positives)  
- Future `start_date` on current job  
- `last_active_date` before `signup_date`  
- Unrealistic junior expertise (e.g., ≥8 expert-level skills on a junior profile acts as a supporting signal)

> `honeypot.py` standalone CLI uses a lower threshold of **0.40** for broader audit use. The ranking pipeline always uses 0.65 from `scorer.py`.

### Stage 4 — Scoring

**Final Score = 0.75 × RelevanceScore + 0.25 × BehavioralScore**

#### Relevance Score (75%)

| Component | Weight | Key logic |
|-----------|--------|-----------|
| Skills | 45% | Must-have (sentence-transformers, FAISS, Qdrant, NDCG, Python, etc.) + good-to-have; Elite bonus for advanced IR/Ranking skills; penalties for CV/speech-primary, all-consulting career, irrelevant skills |
| Experience | 25% | JD range 5–9 yr (with linear decay penalty for >9 yr); depth ratio (AI/ML months ÷ total months); retrieval-specific shipping bonus; penalties for title-chasers (3+ stints <18mo), pure-research career (>70%), no product company |
| Title match | 15% | Tier 1 (1.0): IR/Ranking/Search/Recommendation. Tier 2 (0.90): ML/NLP/Applied Scientist. Tier 3 (0.75): AI/Data Scientist. Tier 4: SWE/Backend/Data Engineer. Weak: Marketing/Civil/etc. |
| Education | 8% | Institution tier × field relevance (CS, ML, AI, Stats, NLP) |
| Location | 5% | JD cities (1.0) → Tier-1 + willing to relocate (0.75) → Tier-1 no relocation (0.50) → Non-Tier-1 India + relocate (0.25) → non-Tier-1 India (0.10) → Abroad + relocate (0.15) → Abroad no relocation (0.0) |
| Certifications | 2% | AWS ML, Google Professional ML, TF Developer, Deep Learning Specialization, etc. |

#### Behavioral Score (25%)

| Component | Weight | Key signals |
|-----------|--------|-------------|
| Readiness | 30% | Last active date, notice period (soft=30d, hard=90d), applications/30d |
| Recruiter interest | 25% | Profile views, saves, search appearances (all /30d) |
| Professionalism | 20% | Recruiter response rate, avg response time, interview completion rate |
| Trust | 15% | Verified email/phone, LinkedIn linked, GitHub activity score, offer acceptance rate |
| Skills quality | 10% | Platform assessment scores on must-have skills + endorsement count |

### Stage 5 — Reasoning Generation

`reasoning.py` generates a unique, fact-grounded justification per candidate (Stage 4 compliance):

- Cites specific YOE, title, company, named skills
- Connects to JD requirements explicitly
- Acknowledges real concerns (notice period, location, inactivity)
- Handled gracefully for null values (e.g., "activity date not available" for missing signals)
- Tone matches rank (positive top-10, cautious bottom-15)
- No templated endings — each row differs

---

## Key Design Decisions

**Why BM25 over embeddings for pre-filter?**  
Embedding 100K profiles in <5 min on CPU is not feasible. BM25 over a weighted text blob is ~10× faster, zero-dependency, and retrieves a high-recall top-5000 candidate set that the full scorer then refines.

**Why 5000 for BM25 cutoff?**  
Empirical: the full scorer runs ~5,000 candidates in ~90s. Going beyond risks the 5-min wall-clock budget.

**Why `_is_consulting_firm()` instead of regex normalization?**  
The previous `_normalize_company()` approach silently missed multi-word firms like "Tech Mahindra" (normalized to "tech_mahindra", not matched against "techmahindra"). Substring matching on raw lowercase fixes this correctly.

**Why separate `honeypot.py` and inline `honeypot_score()` in `scorer.py`?**  
`honeypot.py` is the full audit tool (11 detectors, probabilistic combination, CLI). The inline version in `scorer.py` is a faster subset tuned for the ranking pipeline's false-positive budget (~2% flag rate in top-5000). Both strictly share logical constraints like chronological span and self-taught skill buffers.

**Why use chronological span for career duration?**  
Summing job durations penalizes candidates with concurrent roles (e.g., advisors, moonlighters). Calculating chronological span (`latest - earliest`) provides an accurate measure of total career time to prevent false positives in honeypot detection.

**How do we handle self-taught candidates?**  
The honeypot detector includes a +48-month buffer for skill duration vs. YOE to account for self-taught engineers who developed skills prior to their formal career start, preventing them from being incorrectly disqualified.

---

## Compute Constraints (Spec-Compliant)

| Constraint | Limit | Status |
|------------|-------|--------|
| Runtime | ≤ 5 min wall-clock | ~2 min ✓ |
| RAM | ≤ 16 GB | ~2 GB peak ✓ |
| Compute | CPU only | No GPU ✓ |
| Network | Off | No external calls ✓ |
| Disk | ≤ 5 GB intermediate | Negligible ✓ |

---

## AI Tools Declaration

Used **Claude** and **Antigravity** for architecture planning, code review, and debugging.  
No candidate data was sent to any external LLM.  
All ranking logic is deterministic Python — zero LLM calls during `rank.py` execution.
