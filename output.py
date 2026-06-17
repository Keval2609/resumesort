#!/usr/bin/env python3
"""
output.py — Generate submission CSV from ranked candidates.

Usage:
    python output.py --ranked ranked_candidates.json --out team_xxx.csv
    python output.py --ranked ranked_candidates.json --out team_xxx.csv --validate

Input (ranked_candidates.json): list of dicts from rank.py, sorted best-first.
Each dict must have: candidate_id, final_score, candidate (full profile dict).
"""

import argparse
import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path


# ── JD constants (for reasoning generation) ──────────────────────────────────
JD_MAX_SALARY_LPA = 60.0
JD_NOTICE_SOFT_DAYS = 30
JD_NOTICE_HARD_DAYS = 90
REQUIRED_SKILLS = {
    "sentence transformers", "bge", "e5", "embeddings", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "weaviate",
    "hybrid search", "vector search", "bm25", "information retrieval",
    "ranking", "recommendation systems", "nlp", "python",
    "hugging face transformers", "sentence-transformers",
    "learning to rank", "ndcg", "map", "mrr", "retrieval",
    "fine-tuning llms", "lora", "qlora", "peft",
}
TIER1_COMPANIES = {
    "google", "meta", "microsoft", "amazon", "apple", "uber", "swiggy",
    "zomato", "flipkart", "razorpay", "cred", "meesho", "phonepe",
    "openai", "deepmind", "anthropic", "nvidia",
}
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree",
}
INDIA_TIER1_CITIES = {
    "pune", "noida", "hyderabad", "bangalore", "bengaluru",
    "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram",
}


# ── Reasoning generator ───────────────────────────────────────────────────────

def _days_since(date_str: str) -> int:
    """Days since a date string (YYYY-MM-DD)."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        anchor = date(2026, 6, 13)
        return (anchor - d).days
    except Exception:
        return 999


def _matched_skills(candidate: dict) -> list[str]:
    """Skills from candidate that match JD requirements."""
    skills = candidate.get("skills", [])
    matched = []
    for s in skills:
        name = s.get("name", "").lower()
        if any(req in name for req in REQUIRED_SKILLS):
            matched.append(s["name"])
    return matched


def _career_summary(candidate: dict) -> str:
    """Short career description."""
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Engineer")
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")
    return f"{title} with {yoe:.1f} yrs at {company}"


def _product_company_exp(candidate: dict) -> int:
    """Months spent at non-consulting product companies."""
    months = 0
    for job in candidate.get("career_history", []):
        company = job.get("company", "").lower()
        if not any(c in company for c in CONSULTING_FIRMS):
            months += job.get("duration_months", 0)
    return months


def _location_match(candidate: dict) -> bool:
    location = candidate.get("profile", {}).get("location", "").lower()
    country = candidate.get("profile", {}).get("country", "").lower()
    if country != "india":
        return candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
    return any(city in location for city in INDIA_TIER1_CITIES)


def _salary_concern(candidate: dict) -> str | None:
    sig = candidate.get("redrob_signals", {})
    sal_max = sig.get("expected_salary_range_inr_lpa", {}).get("max", 0)
    if sal_max > JD_MAX_SALARY_LPA * 1.30:
        return f"salary expectation ({sal_max:.0f}L) may exceed budget"
    return None


def _notice_concern(candidate: dict) -> str | None:
    notice = candidate.get("redrob_signals", {}).get("notice_period_days", 0)
    if notice > JD_NOTICE_HARD_DAYS:
        return f"long notice period ({notice}d)"
    elif notice > JD_NOTICE_SOFT_DAYS:
        return f"notice period {notice}d (buyout needed)"
    return None


def _engagement_summary(candidate: dict) -> str:
    sig = candidate.get("redrob_signals", {})
    rr = sig.get("recruiter_response_rate", 0)
    days_inactive = _days_since(sig.get("last_active_date", "2000-01-01"))
    open_flag = sig.get("open_to_work_flag", False)

    parts = []
    if open_flag:
        parts.append("open to work")
    if days_inactive < 30:
        parts.append(f"active {days_inactive}d ago")
    elif days_inactive < 90:
        parts.append(f"semi-active ({days_inactive}d)")
    else:
        parts.append(f"inactive {days_inactive}d")

    parts.append(f"response rate {rr:.0%}")
    return "; ".join(parts)


def generate_reasoning(rank: int, candidate: dict, score: float) -> str:
    """
    Generate specific, honest, non-templated reasoning.
    References actual profile facts + JD connection.
    """
    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})

    career = _career_summary(candidate)
    matched = _matched_skills(candidate)
    product_months = _product_company_exp(candidate)
    loc_ok = _location_match(candidate)
    engagement = _engagement_summary(candidate)
    salary_issue = _salary_concern(candidate)
    notice_issue = _notice_concern(candidate)
    github = sig.get("github_activity_score", -1)
    assessments = sig.get("skill_assessment_scores", {})
    yoe = profile.get("years_of_experience", 0)

    # Build the core statement
    if matched:
        skills_str = ", ".join(matched[:3])
        core = f"{career}; JD-relevant skills: {skills_str}"
    else:
        core = f"{career}; limited direct ML/IR skill overlap with JD"

    # Add product company context
    if product_months >= 36:
        core += f"; {product_months//12}yr+ product-company ML experience"
    elif product_months > 0:
        core += f"; {product_months}mo product-company experience (limited)"
    else:
        core += "; career primarily at consulting/services firms"

    # Add engagement
    core += f". {engagement.capitalize()}"

    # Add positive signals
    extras = []
    if github > 50:
        extras.append(f"strong GitHub activity ({github:.0f})")
    elif github > 20:
        extras.append(f"moderate GitHub ({github:.0f})")
    if assessments:
        top_score = max(assessments.values())
        top_skill = max(assessments, key=assessments.get)
        extras.append(f"assessment: {top_skill} {top_score:.0f}/100")
    if loc_ok:
        loc = profile.get("location", "")
        extras.append(f"location match ({loc})")

    # Add concerns
    concerns = []
    if salary_issue:
        concerns.append(salary_issue)
    if notice_issue:
        concerns.append(notice_issue)
    if not loc_ok:
        concerns.append(f"location mismatch ({profile.get('location','')}; relocate={sig.get('willing_to_relocate',False)})")
    if yoe < 4:
        concerns.append(f"below JD experience floor ({yoe:.1f}yr)")
    elif yoe > 12:
        concerns.append(f"overqualified risk ({yoe:.1f}yr)")

    # Assemble
    parts = [core]
    if extras:
        parts.append("; ".join(extras) + ".")
    if concerns:
        parts.append("Concerns: " + "; ".join(concerns) + ".")

    reasoning = " ".join(parts)

    # Rank-tone check: top ranks should sound positive, bottom ranks honest
    if rank <= 10 and not concerns:
        pass  # already positive
    if rank >= 80 and not concerns:
        reasoning += " Included as marginal fit; skill overlap is indirect."

    # Truncate if too long (keep under ~400 chars for CSV readability)
    if len(reasoning) > 450:
        reasoning = reasoning[:447] + "..."

    return reasoning


# ── Output writer ─────────────────────────────────────────────────────────────

def write_submission(
    ranked: list[dict],
    out_path: str,
    team_name: str = "team_xxx",
) -> Path:
    """
    Write submission CSV.

    ranked: list of dicts, each with:
        - candidate_id: str
        - final_score: float
        - candidate: full candidate profile dict

    Returns Path to written file.
    """
    if len(ranked) < 100:
        raise ValueError(f"Need ≥100 ranked candidates, got {len(ranked)}")

    # Take top 100
    top100 = ranked[:100]

    # Enforce non-increasing scores; adjust tiny floating point drift
    for i in range(1, len(top100)):
        if top100[i]["final_score"] > top100[i - 1]["final_score"]:
            top100[i]["final_score"] = top100[i - 1]["final_score"]

    # Tie-break: same score → sort by candidate_id ascending
    # We need to re-sort by (score DESC, candidate_id ASC) to get stable order
    top100.sort(key=lambda x: (-round(x["final_score"], 6), x["candidate_id"]))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank_idx, item in enumerate(top100, start=1):
        cid = item["candidate_id"]
        score = round(item["final_score"], 6)
        candidate = item.get("candidate", {})
        reasoning = generate_reasoning(rank_idx, candidate, score)

        rows.append({
            "candidate_id": cid,
            "rank": rank_idx,
            "score": f"{score:.6f}",
            "reasoning": reasoning,
        })

    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[output] Written {len(rows)} rows → {out}")
    return out


# ── Inline validator (mirrors validate_submission.py logic) ───────────────────

def quick_validate(path: Path) -> list[str]:
    """Fast local validation before submitting."""
    errors = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["candidate_id", "rank", "score", "reasoning"]:
                errors.append(f"Header mismatch: {reader.fieldnames}")
            rows = list(reader)
    except Exception as e:
        return [str(e)]

    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    seen_ids, seen_ranks = set(), set()
    by_rank = []

    for i, row in enumerate(rows):
        cid = row.get("candidate_id", "").strip()
        rank_s = row.get("rank", "").strip()
        score_s = row.get("score", "").strip()
        reasoning = row.get("reasoning", "").strip()

        if not cid or not cid.startswith("CAND_") or len(cid) != 12:
            errors.append(f"Row {i+2}: bad candidate_id '{cid}'")
        if cid in seen_ids:
            errors.append(f"Row {i+2}: duplicate id {cid}")
        seen_ids.add(cid)

        try:
            rank = int(rank_s)
            if rank in seen_ranks:
                errors.append(f"Row {i+2}: duplicate rank {rank}")
            seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {i+2}: invalid rank '{rank_s}'")
            rank = None

        try:
            score = float(score_s)
        except ValueError:
            errors.append(f"Row {i+2}: invalid score '{score_s}'")
            score = None

        if not reasoning:
            errors.append(f"Row {i+2}: empty reasoning (will hurt Stage 4)")

        if rank is not None and score is not None:
            by_rank.append((rank, score, cid))

    by_rank.sort()
    for i in range(len(by_rank) - 1):
        r1, s1, _ = by_rank[i]
        r2, s2, _ = by_rank[i + 1]
        if s1 < s2:
            errors.append(f"Score increased: rank {r1} ({s1}) < rank {r2} ({s2})")

    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"Missing ranks: {sorted(missing)[:10]}")

    return errors


# ── Stats printer ─────────────────────────────────────────────────────────────

def print_stats(path: Path) -> None:
    """Print summary stats for the written CSV."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    scores = [float(r["score"]) for r in rows]
    print(f"\n── Submission stats ──────────────────────────────")
    print(f"  Rows         : {len(rows)}")
    print(f"  Score range  : {min(scores):.4f} – {max(scores):.4f}")
    print(f"  Score @rank1 : {scores[0]:.4f}")
    print(f"  Score @rank10: {scores[9]:.4f}")
    print(f"  Score @rank50: {scores[49]:.4f}")
    print(f"  Score @rank100:{scores[99]:.4f}")
    print(f"  Unique IDs   : {len({r['candidate_id'] for r in rows})}")
    print(f"  Unique ranks : {len({r['rank'] for r in rows})}")
    avg_reasoning_len = sum(len(r["reasoning"]) for r in rows) / len(rows)
    print(f"  Avg reasoning: {avg_reasoning_len:.0f} chars")

    # Check monotonicity
    violations = sum(
        1 for i in range(len(scores) - 1) if scores[i] < scores[i + 1]
    )
    print(f"  Score mono?  : {'✓' if violations == 0 else f'✗ {violations} violations'}")
    print(f"──────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate submission CSV from rank.py output")
    parser.add_argument(
        "--ranked",
        required=True,
        help="JSON file output by rank.py (list of ranked dicts)",
    )
    parser.add_argument(
        "--out",
        default="team_xxx.csv",
        help="Output CSV path (default: team_xxx.csv)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run inline validator after writing",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print submission stats after writing",
    )
    args = parser.parse_args()

    # Load ranked output from rank.py
    print(f"[output] Loading ranked candidates from {args.ranked} ...")
    with open(args.ranked, encoding="utf-8") as f:
        ranked = json.load(f)
    print(f"[output] Loaded {len(ranked)} ranked candidates")

    # Write CSV
    out_path = write_submission(ranked, args.out)

    # Stats
    if args.stats:
        print_stats(out_path)

    # Validate
    if args.validate:
        print(f"[output] Validating {out_path} ...")
        errors = quick_validate(out_path)
        if errors:
            print(f"\n[output] ✗ Validation FAILED ({len(errors)} issues):")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        else:
            print("[output] ✓ Validation passed")

    print(f"\n[output] Done → {out_path}")


if __name__ == "__main__":
    main()
