"""
Inverted Index + BM25 implementation for candidate ranking.
CPU-only, no external APIs, fits 100K candidates in <16GB RAM.
"""

import math
import json
import re
from collections import defaultdict
from typing import Optional


# ─── Tokenizer ───────────────────────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","and","or","of","in","to","for","with","on","at","by","from",
    "is","are","was","were","be","been","have","has","had","not","no","i","my",
    "we","our","their","its","this","that","which","who","as","at","but","if"
}

def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ─── Inverted Index ───────────────────────────────────────────────────────────

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, dict[str, int]] = defaultdict(dict)  # term -> {doc_id: freq}
        self.doc_lengths: dict[str, int] = {}
        self.doc_count: int = 0
        self.avg_doc_length: float = 0.0

    def add_document(self, doc_id: str, tokens: list[str]):
        freq = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        for term, count in freq.items():
            self.index[term][doc_id] = count
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1

    def finalize(self):
        total = sum(self.doc_lengths.values())
        self.avg_doc_length = total / self.doc_count if self.doc_count else 1.0

    def df(self, term: str) -> int:
        """Document frequency of term."""
        return len(self.index.get(term, {}))


# ─── BM25 Scorer ─────────────────────────────────────────────────────────────

class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def idf(self, term: str) -> float:
        df = self.index.df(term)
        if df == 0:
            return 0.0
        N = self.index.doc_count
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list[str], doc_id: str) -> float:
        dl = self.index.doc_lengths.get(doc_id, 0)
        avgdl = self.index.avg_doc_length
        total = 0.0
        for term in set(query_tokens):
            idf = self.idf(term)
            if idf == 0:
                continue
            tf = self.index.index.get(term, {}).get(doc_id, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
            total += idf * (numerator / denominator)
        return total

    def retrieve(
        self,
        query_tokens: list[str],
        top_k: int = 200,
        candidate_ids: Optional[set] = None
    ) -> list[tuple[str, float]]:
        """
        Efficient retrieval: only score docs containing at least one query term.
        Returns list of (doc_id, score) sorted descending.
        """
        candidate_docs: set[str] = set()
        for term in set(query_tokens):
            posting = self.index.index.get(term, {})
            if candidate_ids:
                candidate_docs.update(candidate_ids & posting.keys())
            else:
                candidate_docs.update(posting.keys())

        scores = [
            (doc_id, self.score(query_tokens, doc_id))
            for doc_id in candidate_docs
        ]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ─── Candidate Text Builder ───────────────────────────────────────────────────

def build_candidate_text(candidate: dict) -> str:
    """
    Flatten a candidate JSON into a single weighted text blob.
    High-value fields are repeated for TF amplification.
    """
    parts = []
    p = candidate.get("profile", {})

    # High-weight fields (repeated 3x)
    for _ in range(3):
        parts.append(p.get("headline", ""))
        parts.append(p.get("current_title", ""))

    # Medium-weight (repeated 2x)
    for _ in range(2):
        parts.append(p.get("summary", ""))
        parts.append(p.get("current_industry", ""))

    # Skills (repeated 2x for core ML skills)
    CORE_SKILLS = {
        "python","embeddings","faiss","pinecone","weaviate","qdrant","milvus",
        "elasticsearch","opensearch","sentence-transformers","bge","e5",
        "hugging face","transformers","ranking","retrieval","recommendation",
        "ndcg","mrr","map","bm25","hybrid search","vector","information retrieval",
        "fine-tuning","lora","peft","xgboost","lightgbm","scikit-learn",
        "pytorch","tensorflow","mlflow","feature engineering","a/b testing"
    }
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "")
        parts.append(name)
        if name.lower() in CORE_SKILLS or proficiency in ("advanced", "expert"):
            parts.append(name)  # extra weight

    # Career history
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("industry", ""))
        parts.append(job.get("description", ""))

    # Education
    for edu in candidate.get("education", []):
        parts.append(edu.get("field_of_study", ""))
        parts.append(edu.get("degree", ""))

    # Certifications
    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", ""))

    return " ".join(filter(None, parts))


# ─── JD Query Builder ────────────────────────────────────────────────────────

JD_QUERY = """
senior ai engineer machine learning ranking retrieval recommendation systems
embeddings sentence transformers bge e5 hugging face transformers
vector database faiss pinecone weaviate qdrant milvus opensearch elasticsearch
hybrid search python production deployment applied ml
ndcg mrr map evaluation framework a/b testing offline online
fine-tuning lora peft xgboost lightgbm learning to rank
information retrieval nlp feature engineering mlops
product company startup founding team
"""


# ─── Build Index from candidates ─────────────────────────────────────────────

def build_index(candidates: list[dict]) -> tuple[InvertedIndex, BM25]:
    idx = InvertedIndex()
    for c in candidates:
        doc_id = c["candidate_id"]
        text = build_candidate_text(c)
        tokens = tokenize(text)
        idx.add_document(doc_id, tokens)
    idx.finalize()
    bm25 = BM25(idx)
    return idx, bm25


# ─── Quick sanity test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal smoke test with 3 fake candidates
    dummy = [
        {
            "candidate_id": "CAND_0000001",
            "profile": {
                "headline": "ML Engineer ranking retrieval systems",
                "current_title": "ML Engineer",
                "summary": "Built embedding-based retrieval and ranking systems at scale.",
                "current_industry": "AI/ML",
            },
            "skills": [
                {"name": "FAISS", "proficiency": "expert", "endorsements": 50},
                {"name": "Sentence Transformers", "proficiency": "expert", "endorsements": 40},
                {"name": "Python", "proficiency": "expert", "endorsements": 60},
            ],
            "career_history": [
                {"title": "ML Engineer", "industry": "Tech",
                 "description": "Led ranking and retrieval at a product company using FAISS and BGE embeddings."}
            ],
            "education": [{"degree": "B.Tech", "field_of_study": "Computer Science"}],
            "certifications": [],
        },
        {
            "candidate_id": "CAND_0000002",
            "profile": {
                "headline": "Marketing Manager driving business outcomes",
                "current_title": "Marketing Manager",
                "summary": "Led marketing campaigns and brand strategy.",
                "current_industry": "FMCG",
            },
            "skills": [
                {"name": "Marketing", "proficiency": "expert", "endorsements": 70},
                {"name": "SEO", "proficiency": "advanced", "endorsements": 30},
            ],
            "career_history": [
                {"title": "Marketing Manager", "industry": "FMCG",
                 "description": "Managed campaigns and content strategy."}
            ],
            "education": [{"degree": "MBA", "field_of_study": "Marketing"}],
            "certifications": [],
        },
        {
            "candidate_id": "CAND_0000003",
            "profile": {
                "headline": "NLP Engineer information retrieval",
                "current_title": "NLP Engineer",
                "summary": "Built vector search and hybrid retrieval pipelines using Elasticsearch and Qdrant.",
                "current_industry": "AI/ML",
            },
            "skills": [
                {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 45},
                {"name": "Qdrant", "proficiency": "advanced", "endorsements": 35},
                {"name": "Python", "proficiency": "expert", "endorsements": 55},
            ],
            "career_history": [
                {"title": "NLP Engineer", "industry": "AI/ML",
                 "description": "Information retrieval, hybrid search, NDCG evaluation."}
            ],
            "education": [{"degree": "M.Tech", "field_of_study": "Machine Learning"}],
            "certifications": [],
        },
    ]

    idx, bm25 = build_index(dummy)
    query_tokens = tokenize(JD_QUERY)
    results = bm25.retrieve(query_tokens, top_k=10)

    print("BM25 retrieval results:")
    for doc_id, score in results:
        print(f"  {doc_id}  score={score:.4f}")

    assert results[0][0] != "CAND_0000002", "Marketing manager should NOT rank first!"
    print("\nAll checks passed.")
