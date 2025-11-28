"""
Standalone NDCG evaluator for AutoDDG (BM25 only).

Reads:
    - baseline JSONL (descriptions)
    - metadata JSON (original descriptions)
    - queries.txt
    - relevance_matrix.csv

Computes BM25 NDCG@10 and @20 for:
    hs, ufd, sfd, ufd_nyc, sfd_nyc, original

Outputs:
    ndcg_results.json
"""

import json
import math
from typing import Dict, List
import pandas as pd
from collections import Counter


# ======================================
#  Utils: DCG / NDCG
# ======================================

def dcg(scores: List[int], k: int) -> float:
    s = scores[:k]
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(s))


def ndcg(scores: List[int], k: int) -> float:
    if not scores:
        return 0.0
    ideal = sorted(scores, reverse=True)
    d_ideal = dcg(ideal, k)
    return dcg(scores, k) / d_ideal if d_ideal > 0 else 0.0


# ======================================
#  BM25 Implementation
# ======================================

def bm25_score(query_tokens, doc_tokens, avgdl, N, doc_freq, k1=1.5, b=0.75):
    score = 0.0
    tf = Counter(doc_tokens)
    doc_len = len(doc_tokens)

    for term in query_tokens:
        if term not in tf:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        score += idf * ((tf[term] * (k1 + 1)) /
                        (tf[term] + k1 * (1 - b + b * (doc_len / avgdl))))
    return score


def bm25_rank(query: str, documents: Dict[str, str]) -> List[str]:
    q_toks = query.lower().split()
    tok_docs = {k: v.lower().split() for k, v in documents.items()}

    N = len(tok_docs)
    avgdl = sum(len(t) for t in tok_docs.values()) / N

    doc_freq = {}
    for tks in tok_docs.values():
        for term in set(tks):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    scored = []
    for ds_id, toks in tok_docs.items():
        s = bm25_score(q_toks, toks, avgdl, N, doc_freq)
        scored.append((ds_id, s))

    return [ds for ds, _ in sorted(scored, key=lambda x: x[1], reverse=True)]


# ======================================
#  NDCG for one method
# ======================================

def compute_ndcg_for_method(
    queries: List[str],
    dataset_records: Dict[str, Dict[str, str]],
    relevance: Dict[str, Dict[str, int]],
    method_col: str,
    k: int,
):
    ndcg_scores = []

    for qi, q in enumerate(queries, start=1):
        docs = {ds: dataset_records[ds].get(method_col, "") for ds in dataset_records}
        ranked = bm25_rank(q, docs)

        rel_scores = [
            relevance.get(f"query_{qi}", {}).get(ds, 0)
            for ds in ranked
        ]

        ndcg_scores.append(ndcg(rel_scores, k))

    return sum(ndcg_scores) / len(ndcg_scores)


# ======================================
#  Main Pipeline
# ======================================

def run_ndcg_evaluation(
    baseline_jsonl: str,
    metadata_json: str,      # ← NEW
    queries_txt: str,
    relevance_csv: str,
    output_json: str,
):
    # ----------------------------------------------------
    # Load metadata JSON (original descriptions)
    # ----------------------------------------------------
    with open(metadata_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    original_map = {item["id"]: item.get("description", "") for item in metadata}

    # ----------------------------------------------------
    # Load baseline JSONL
    # ----------------------------------------------------
    dataset_records = {}

    with open(baseline_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ds = rec["dataset_id"]

            dataset_records[ds] = {
                "hs": rec.get("HandS", ""),
                "ufd": rec.get("ufd", ""),
                "sfd": rec.get("sfd", ""),
                "ufd_nyc": rec.get("ufd_nyc", ""),
                "sfd_nyc": rec.get("sfd_nyc", ""),
                "original": original_map.get(ds, "")  # ← correct source
            }

    baseline_ids = set(dataset_records.keys())

    # ----------------------------------------------------
    # Load queries
    # ----------------------------------------------------
    with open(queries_txt, "r", encoding="utf-8") as f:
        queries = [q.strip() for q in f if q.strip()]

    # ----------------------------------------------------
    # Load relevance
    # ----------------------------------------------------
    rel_df = pd.read_csv(relevance_csv)
    rel_ids = set(rel_df["dataset_id"].astype(str))

    if not rel_ids.issubset(baseline_ids):
        missing = rel_ids - baseline_ids
        print("[WARNING] baseline JSONL does NOT contain all dataset_ids from relevance CSV.")
        print(f"Missing count: {len(missing)}")
        print(f"Example missing: {list(missing)[:10]}\n")

    relevance = {}
    for col in rel_df.columns:
        if col == "dataset_id":
            continue
        qid = col
        relevance[qid] = {}
        for _, row in rel_df.iterrows():
            relevance[qid][str(row["dataset_id"])] = int(row[col])

    # ----------------------------------------------------
    # Compute NDCG (BM25 only)
    # ----------------------------------------------------
    methods = ["hs", "ufd", "sfd", "ufd_nyc", "sfd_nyc", "original"]
    results = {}

    for m in methods:
        print(f"Computing {m} ...")
        ndcg10 = compute_ndcg_for_method(queries, dataset_records, relevance, m, 10)
        ndcg20 = compute_ndcg_for_method(queries, dataset_records, relevance, m, 20)

        results[m] = {
            "bm25@10": ndcg10,
            "bm25@20": ndcg20,
        }

    # ----------------------------------------------------
    # Save final output
    # ----------------------------------------------------
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved NDCG results →", output_json)


# ======================================
#  CLI
# ======================================

if __name__ == "__main__":
    run_ndcg_evaluation(
        baseline_jsonl="../../outputs/baseline_autoddg_descriptions.jsonl",
        metadata_json="../../outputs/metadata_registry.json",
        queries_txt="queries.txt",
        relevance_csv="relevance_matrix.csv",
        output_json="ndcg_eval_results.json"
    )
