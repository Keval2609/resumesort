# Redrob Hackathon — Intelligent Candidate Ranking

**Team:** Antigravity-Redrob  
**Challenge:** Intelligent Candidate Discovery & Ranking  
**Target role:** Senior AI Engineer — Founding Team  

---

## Repo structure

```
resumesort/
├── rank.py              # Main pipeline script — reads candidates, coordinates scoring and output
├── bm25.py              # TF-IDF/BM25 retrieval system
├── scorer.py            # Scoring function with behavioral and trust modifiers
├── output.py            # CSV writer — converts ranked JSON to submission CSV
├── honeypot.py          # Honeypot detection — flags impossible profiles
├── reasoning.py         # Stage 4 reasoning generator
├── requirements.txt
├── submission_metadata.yaml
├── .gitignore
│
├── scripts/
│   ├── precompute.py    # Optional: pre-build BM25 index / TF-IDF cache
│   └── explore.py       # EDA helpers used during development
│
├── tests/
│   ├── test_honeypot.py
│   ├── test_output.py
│   └── test_reasoning.py
│
├── artifacts/           # Pre-computed index files (gitignored if large)
│   └── .gitkeep
│
└── docs/
    └── approach.md      # Design notes
```

---

## Reproduce command

```bash
python rank.py --candidates ./candidates.jsonl --out ./final_output.csv
```

This single command:
1. Loads all 100K candidates
2. Runs honeypot detection (flags ~80 impossible profiles → score = 0)
3. Scores every candidate with the weighted formula
4. Selects top 100, generates reasoning, writes CSV
5. Validates format before exit

**Runtime:** ~3.6 min on 16 GB CPU-only machine  
**No network calls during ranking**

---

## Pipeline overview

```
candidates.jsonl
       │
       ▼
honeypot.py          Filter impossible profiles (score = 0.0)
       │
       ▼
rank.py  ──────────── Score each candidate
  │  FinalScore = 0.70 × RelevanceScore + 0.30 × BehavioralScore
  │
  │  RelevanceScore sub-weights:
  │    Skills           0.45
  │    Experience       0.25
  │    Title / Career   0.15
  │    Education        0.08
  │    Location         0.05
  │    Certifications   0.02
  │
  │  BehavioralScore sub-weights:
  │    Readiness        0.30
  │    Recruiter signals 0.25
  │    Professionalism  0.20
  │    Trust            0.15
  │    Skills quality   0.10
  │
  ▼
reasoning.py         Generate Stage 4-compliant per-candidate reasoning
       │
       ▼
output.py            Write final_output.csv (100 rows, validated)
```

---

## Hard gates (automatic exclusion)

| Condition | Action |
|---|---|
| `open_to_work_flag = false` | Score = 0 (excluded) |
| Honeypot detected | Score = 0 (excluded) |
| Notice period > 180d | Score = 0 |
| Work mode hard mismatch | Score multiplier × 0.3 |
| Salary > 130% of JD max | Score multiplier × 0.5 |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Validate before submitting

```bash
python validate_submission.py final_output.csv
```

---

## Run tests

```bash
python tests/test_honeypot.py
python tests/test_output.py
python tests/test_reasoning.py
```

---

## Compute declaration

- CPU only (no GPU during ranking)
- No external API calls during ranking
- ~3.6 min wall-clock on 16 GB RAM (Windows 11)
- No intermediate state > 5 GB

---

## AI tools used

- Antigravity AI (code fixes, performance optimization, and running scripts)
- Claude (architecture discussion, code review)  
- All candidate data processed locally; no profile data sent to any LLM API
