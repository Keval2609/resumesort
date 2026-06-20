# Redrob Hackathon — Intelligent Candidate Ranking

**Team:** SignalOverNoise-108  
**Challenge:** Intelligent Candidate Discovery & Ranking  
**Target Role:** Senior AI Engineer — Founding Team  

---

## 🎯 Overview

This repository contains our submission for the Redrob Hackathon. We've built a deterministic, fully local, and highly efficient candidate ranking pipeline. It ranks up to 100K candidates without relying on external LLM APIs, GPUs, or network calls during the ranking process. 

The pipeline uses a combination of BM25 pre-filtering, rigorous honeypot detection, and a carefully weighted scoring mechanism to identify the best candidates based on Relevance (70%) and Behavioral (30%) factors.

---

## 📁 Repository Structure

```
resumesort/
├── rank.py                  # Main CLI entry point — orchestrates the entire ranking pipeline
├── app.py                   # Streamlit interactive sandbox for testing candidate batches
├── bm25.py                  # Custom TF-IDF/BM25 inverted index for fast pre-filtering
├── scorer.py                # Core scoring logic (Relevance + Behavioral metrics)
├── honeypot.py              # Synthetic/impossible profile detection
├── reasoning.py             # Stage 4-compliant justification generator for candidate ranks
├── validate_submission.py   # Strict CSV output validator
├── submission_metadata.yaml # Metadata, compute declaration, and methodology summary
├── requirements.txt         # Python dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Ranking Pipeline (CLI)

To rank the full 100K candidates dataset and generate the final output CSV:

```bash
python rank.py --candidates ./candidates.jsonl --out ./SignalOverNoise-108.csv
```

**What this command does:**
1. Loads all candidates.
2. Builds a BM25 index and retrieves the top 5,000 matches.
3. Applies Hard Gates (e.g., open-to-work, salary expectations).
4. Runs Honeypot Detection to flag synthetic profiles (score = 0).
5. Computes Relevance and Behavioral scores for the remaining candidates.
6. Selects the top 100 candidates, guarantees score monotonicity, and generates reasoning.
7. Writes the results to a CSV and validates the output format.

**Performance:** Completes in ~2.1 minutes on a standard 16 GB CPU-only machine.

### 3. Run the Interactive Sandbox (Streamlit)

We also provide a Streamlit app to interactively explore the ranking logic on smaller datasets (≤ 100 candidates):

```bash
streamlit run app.py
```

---

## 🧠 Pipeline Architecture

### 1. BM25 Pre-filtering
To efficiently handle 100K profiles, we first flatten each candidate's high-value fields (titles, core skills, summaries) into a weighted text blob. We then build an in-memory inverted index and retrieve the top 5,000 candidates using the BM25 algorithm against our tailored Job Description query.

### 2. Hard Gates
Candidates are instantly excluded or heavily penalized if they fail basic criteria:
- Not open to work (`Score = 0`)
- Work mode hard mismatch (e.g., remote-only but requires relocation)

### 3. Honeypot Detection
Our `honeypot.py` script rigorously identifies synthetic or impossible profiles (e.g., 5 expert skills with < 3 years of experience, time-traveling career dates, or 0-duration skills with 20+ endorsements). Candidates exceeding the threshold are excluded (`Score = 0`).

### 4. Scoring Formula
**Final Score = 0.70 × Relevance + 0.30 × Behavioral**

**Relevance Score (70%):**
- **Skills (45%):** Keyword matching against Must-Have (e.g., Sentence Transformers, BM25) and Good-To-Have skills, with penalties for irrelevant focuses (e.g., pure Computer Vision).
- **Experience (25%):** Penalizes pure research or title-chasers; rewards applied ML/AI depth in product companies.
- **Title / Career (15%):** Exact or strong partial matches to "AI Engineer", "ML Engineer", etc.
- **Education (8%):** Based on institution tier and relevance of study field.
- **Location (5%):** Proximity to preferred tier-1 cities and willingness to relocate.
- **Certifications (2%):** Bonus for relevant ML certifications.

**Behavioral Score (30%):**
- **Readiness (30%):** Notice period length and recent platform activity.
- **Recruiter Signals (25%):** Profile views, searches, and saves.
- **Professionalism (20%):** Recruiter response rates and interview completion rates.
- **Trust (15%):** Verified contacts, linked GitHub, and offer acceptance history.
- **Skills Quality (10%):** Endorsements and platform skill assessment scores.

### 5. Reasoning Generation
For the final Top 100, `reasoning.py` dynamically builds a Stage 4-compliant summary citing specific candidate facts, JD alignment, and honest concerns (e.g., salary bands, location issues), ensuring top-ranked candidates have positive tones while lower-ranked ones highlight genuine risks.

---

## ✅ Validation & Testing

Ensure your output meets all challenge constraints:

```bash
python validate_submission.py SignalOverNoise-108.csv
```
*(Checks for exactly 100 rows, strict header format, monotonically decreasing scores, and valid CAND_ IDs).*

---

## 🤖 Compute & AI Declarations

- **Hardware:** Developed and tested on Windows 11 Pro (4 cores, 16 GB RAM).
- **Inference:** **CPU ONLY.** No GPUs were used during ranking.
- **Network:** **100% Offline.** No external API or LLM calls are made during the execution of `rank.py`.
- **AI Tools:** Antigravity AI and Claude were utilized exclusively for architecture planning, code review, and script generation. Candidate data was strictly processed locally.
