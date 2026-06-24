#!/usr/bin/env python3

from __future__ import annotations
from datetime import date, datetime
from typing import Any

HONEYPOT_THRESHOLD = 0.40


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _today():
    return date(2026, 6, 13)  # Match scorer.py


# ── Individual detectors ─────────────────────────────────────────────────────

def _check_salary_inversion(c):
    sal = c.get("redrob_signals", {}).get("expected_salary_range_inr_lpa", {})
    mn, mx = sal.get("min", 0), sal.get("max", 0)
    if mn > 0 and mx > 0 and mn > mx * 1.05:
        return 0.60, f"salary inverted (min={mn} > max={mx})"
    return 0.0, ""


def _check_active_before_signup(c):
    sig = c.get("redrob_signals", {})
    signup = _parse_date(sig.get("signup_date"))
    active = _parse_date(sig.get("last_active_date"))
    if signup and active and active < signup:
        delta = (signup - active).days
        return 0.80, f"last_active ({active}) before signup ({signup}) by {delta}d"
    return 0.0, ""


def _check_skill_duration_exceeds_yoe(c):
    yoe = c.get("profile", {}).get("years_of_experience", 0)
    cap = yoe * 12 + 48   # allow 4yr buffer for self-taught
    violations = []
    for s in c.get("skills", []):
        dur = s.get("duration_months", 0)
        if dur > cap:
            violations.append(f"{s['name']}({dur}mo)")
    if not violations:
        return 0.0, ""
    weight = min(0.15 + 0.05 * len(violations), 0.25)
    return weight, f"skill dur > cap ({cap:.0f}mo): {', '.join(violations[:3])}"


def _check_career_duration_exceeds_yoe(c):
    yoe = c.get("profile", {}).get("years_of_experience", 0)
    jobs = c.get("career_history", [])
    if not jobs:
        return 0.0, ""

    # Chronological span — handles moonlighters / concurrent roles
    spans = []
    for j in jobs:
        start = _parse_date(j.get("start_date"))
        if not start:
            continue
        end = _today() if j.get("is_current") else (_parse_date(j.get("end_date")) or _today())
        spans.append((start, end))

    if spans:
        earliest = min(s[0] for s in spans)
        latest   = max(s[1] for s in spans)
        chron_months = (latest - earliest).days / 30.44
    else:
        chron_months = sum(j.get("duration_months", 0) for j in jobs)

    cap = yoe * 12 + 24
    if chron_months > cap * 1.5:
        return 0.50, f"career span {chron_months:.0f}mo >> YOE {yoe}yr"
    return 0.0, ""


def _check_expert_on_junior(c):
    yoe = c.get("profile", {}).get("years_of_experience", 0)
    if yoe >= 5:
        return 0.0, ""
    expert_n = sum(
        1 for s in c.get("skills", [])
        if s.get("proficiency") in ("expert", "advanced")
    )
    if expert_n >= 5 and yoe < 3:
        return 0.15, f"{expert_n} advanced/expert skills on {yoe}yr profile"
    if expert_n >= 8 and yoe < 5:
        return 0.12, f"{expert_n} advanced/expert skills on {yoe}yr profile"
    return 0.0, ""


def _check_future_end_date(c):
    today = _today()
    for job in c.get("career_history", []):
        if job.get("is_current"):
            continue
        ed = _parse_date(job.get("end_date"))
        if ed and ed > today:
            return 0.70, (
                f"non-current job '{job.get('title')}@{job.get('company')}' "
                f"has future end_date {ed}"
            )
    return 0.0, ""


def _check_low_completeness_high_expert(c):
    comp = c.get("redrob_signals", {}).get("profile_completeness_score", 100)
    expert_n = sum(
        1 for s in c.get("skills", [])
        if s.get("proficiency") == "expert"
    )
    if comp < 30 and expert_n >= 3:
        return 0.35, f"completeness {comp:.0f}% but {expert_n} expert skills"
    return 0.0, ""


def _check_endorsements_zero_duration(c):
    suspicious = [
        s["name"]
        for s in c.get("skills", [])
        if s.get("duration_months", 1) == 0 and s.get("endorsements", 0) >= 20
    ]
    if suspicious:
        return 0.45, f"high endorsements on 0-dur skills: {suspicious}"
    return 0.0, ""


def _check_expert_vs_low_assessment(c):
    assessments = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    mismatches = [
        f"{s['name']}(score={assessments[s['name']]:.0f})"
        for s in c.get("skills", [])
        if s.get("proficiency") == "expert"
        and s["name"] in assessments
        and assessments[s["name"]] < 30
    ]
    if mismatches:
        return 0.15, f"expert claim but low assessment: {mismatches}"
    return 0.0, ""


def _check_offer_acceptance_gt1(c):
    oar = c.get("redrob_signals", {}).get("offer_acceptance_rate", -1)
    if oar > 1.0:
        return 0.80, f"offer_acceptance_rate > 1.0: {oar}"
    return 0.0, ""


def _check_closed_but_spamming(c):
    sig = c.get("redrob_signals", {})
    if not sig.get("open_to_work_flag", True) and sig.get("applications_submitted_30d", 0) > 20:
        return 0.25, f"open_to_work=False but {sig['applications_submitted_30d']} apps/30d"
    return 0.0, ""


_DETECTORS = [
    _check_salary_inversion,
    _check_active_before_signup,
    _check_skill_duration_exceeds_yoe,
    _check_career_duration_exceeds_yoe,
    _check_expert_on_junior,
    _check_future_end_date,
    _check_low_completeness_high_expert,
    _check_endorsements_zero_duration,
    _check_expert_vs_low_assessment,
    _check_offer_acceptance_gt1,
    _check_closed_but_spamming,
]


# ── Public API ────────────────────────────────────────────────────────────────

def score_honeypot(candidate: dict) -> tuple[float, list[str]]:
    """Returns (honeypot_score 0..1, reasons list)."""
    combined = 0.0
    reasons = []
    for detector in _DETECTORS:
        weight, reason = detector(candidate)
        if weight > 0:
            combined = 1.0 - (1.0 - combined) * (1.0 - weight)
            reasons.append(f"[{weight:.2f}] {reason}")
    return round(combined, 4), reasons


def is_honeypot(candidate: dict, threshold: float = HONEYPOT_THRESHOLD) -> bool:
    score, _ = score_honeypot(candidate)
    return score >= threshold


def filter_honeypots(
    candidates: list[dict],
    threshold: float = HONEYPOT_THRESHOLD,
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Returns (clean_list, flagged_list)."""
    clean, flagged = [], []
    for c in candidates:
        score, reasons = score_honeypot(c)
        if score >= threshold:
            c["_honeypot_score"] = score
            c["_honeypot_reasons"] = reasons
            flagged.append(c)
            if verbose:
                cid = c.get("candidate_id", "?")
                print(f"[honeypot] {cid} score={score:.3f}")
                for r in reasons:
                    print(f"           {r}")
        else:
            clean.append(c)
    return clean, flagged


def annotate_honeypot_scores(candidates: list[dict]) -> list[dict]:
    """Add _honeypot_score/_honeypot_reasons to every candidate in-place."""
    for c in candidates:
        score, reasons = score_honeypot(c)
        c["_honeypot_score"] = score
        c["_honeypot_reasons"] = reasons
    return candidates


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json, gzip

    if len(sys.argv) < 2:
        print("Usage: python honeypot.py candidates.jsonl[.gz] [--verbose]")
        sys.exit(1)

    path = sys.argv[1]
    verbose = "--verbose" in sys.argv

    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            candidates = [json.loads(line) for line in f if line.strip()]
    elif path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            candidates = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, encoding="utf-8") as f:
            candidates = json.load(f)

    clean, flagged = filter_honeypots(candidates, verbose=verbose)

    print(f"\nTotal    : {len(candidates)}")
    print(f"Clean    : {len(clean)}")
    print(f"Honeypots: {len(flagged)}")
    print(f"Rate     : {len(flagged)/max(len(candidates),1)*100:.2f}%")

    if not verbose:
        flagged.sort(key=lambda c: c["_honeypot_score"], reverse=True)
        print("\nTop flagged:")
        for c in flagged[:10]:
            print(f"  {c['candidate_id']}  score={c['_honeypot_score']:.3f}")
            for r in c["_honeypot_reasons"][:2]:
                print(f"    {r}")
