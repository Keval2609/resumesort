"""
app.py — Redrob Hackathon Sandbox (Team_Dev_108)
Streamlit interface for candidate ranking pipeline.
Runs end-to-end: upload candidates → rank → download CSV.
"""

import csv
import io
import json
import pandas as pd
import streamlit as st

from bm25 import JD_QUERY, build_index, tokenize
from reasoning import generate_reasoning
from scorer import final_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Redrob Candidate Ranker")
st.caption("Team_Dev_108 | Senior AI Engineer — Founding Team")

st.markdown("""
**Pipeline:** BM25 pre-filter → Hard gates → Honeypot detection → Full scoring → Ranked CSV  
**Formula:** `FinalScore = 0.75 × RelevanceScore + 0.25 × BehavioralScore`  
> Sandbox accepts ≤100 candidates. Full 100K run happens locally.
""")

st.divider()

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload candidates file (.json array or .jsonl)",
    type=["json", "jsonl"],
    help="Each candidate must match the Redrob candidate schema.",
)

if uploaded is None:
    st.info("Upload a candidates file to begin.")
    st.stop()

# ── Parse ─────────────────────────────────────────────────────────────────────
try:
    raw = uploaded.read().decode("utf-8").strip()
    if raw.startswith("["):
        candidates = json.loads(raw)
    else:
        candidates = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
except Exception as e:
    st.error(f"Failed to parse file: {e}")
    st.stop()

if len(candidates) > 100:
    st.warning(f"Loaded {len(candidates)} candidates — truncating to 100 for sandbox.")
    candidates = candidates[:100]
else:
    st.success(f"Loaded {len(candidates)} candidates.")

# ── Run button ────────────────────────────────────────────────────────────────
if not st.button("▶ Run Ranking Pipeline", type="primary"):
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────
with st.spinner("Building BM25 index..."):
    idx, bm25 = build_index(candidates)
    query_tokens = tokenize(JD_QUERY)
    bm25_hits = bm25.retrieve(query_tokens, top_k=len(candidates))
    hit_ids = {doc_id for doc_id, _ in bm25_hits}
    to_score = [c for c in candidates if c["candidate_id"] in hit_ids]

st.write(f"BM25 shortlist: **{len(to_score)}** candidates")

with st.spinner("Scoring candidates..."):
    scored = [final_score(c) for c in to_score]
    scored.sort(key=lambda r: (-r["final"], r["candidate_id"]))

    # Gate & honeypot stats
    honeypots  = sum(1 for r in scored if r["gate"] == "honeypot")
    not_open   = sum(1 for r in scored if r["gate"] == "not_open_to_work")
    salary_out = sum(1 for r in scored if r["gate"] == "salary_too_high")
    mode_fail  = sum(1 for r in scored if r["gate"] == "work_mode_mismatch")
    active     = sum(1 for r in scored if r["gate"] is None)

    available  = [r for r in scored if r["gate"] != "honeypot"]
    top_k      = min(100, len(available))
    top_results = available[:top_k]

# ── Stats ─────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Pipeline Stats")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Active (scored)", active)
col2.metric("Honeypots filtered", honeypots)
col3.metric("Not open-to-work", not_open)
col4.metric("Salary too high", salary_out)
col5.metric("Work mode mismatch", mode_fail)

if honeypots > 0:
    hp_rate = honeypots / len(candidates) * 100
    color = "🔴" if hp_rate > 10 else "🟢"
    st.write(f"{color} Honeypot rate: **{hp_rate:.1f}%** (limit: 10%)")

# ── Generate CSV ──────────────────────────────────────────────────────────────
cand_by_id = {c["candidate_id"]: c for c in candidates}

output = io.StringIO()
writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
writer.writerow(["candidate_id", "rank", "score", "reasoning"])

rows_for_table = []
with st.spinner("Generating reasoning strings..."):
    for rank, r in enumerate(top_results, 1):
        cid = r["candidate_id"]
        reasoning = generate_reasoning(rank, cand_by_id[cid], r["final"])
        writer.writerow([cid, rank, f"{r['final']:.6f}", reasoning])
        rows_for_table.append({
            "rank":         rank,
            "candidate_id": cid,
            "score":        round(r["final"], 4),
            "relevance":    r.get("relevance", 0),
            "behavioral":   r.get("behavioral", 0),
            "gate":         r.get("gate") or "✓",
            "reasoning_preview": reasoning[:80] + "...",
        })

csv_bytes = output.getvalue().encode("utf-8")

# ── Results table ─────────────────────────────────────────────────────────────
st.divider()
st.subheader(f"Top {len(top_results)} Rankings")

import pandas as pd
df = pd.DataFrame(rows_for_table)
st.dataframe(df, use_container_width=True, height=400)

# ── Download ──────────────────────────────────────────────────────────────────
st.divider()
st.download_button(
    label="⬇️ Download Ranked CSV",
    data=csv_bytes,
    file_name="final_output.csv",
    mime="text/csv",
    type="primary",
)

st.caption(
    "Submission spec: 100 rows · UTF-8 · ranks 1–100 · monotone non-increasing scores"
)