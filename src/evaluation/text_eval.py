import json
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from dotenv import load_dotenv
import pandas as pd

import google.generativeai as genai
from google.generativeai import GenerativeModel

# -----------------------
# Init nltk & Gemini
# -----------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"
model = GenerativeModel(MODEL_NAME)

# -----------------------
# Prompts
# -----------------------
EVAL_PROMPT = """
Evaluate the dataset description on (0–10):

1. COMPLETENESS
2. CONCISENESS
3. READABILITY

Return ONLY JSON:
{"completeness": int, "conciseness": int, "readability": int}
"""

FAITH_PROMPT = """
Evaluate FAITHFULNESS (0–10) of the description.

A faithful description MUST only include information inferable from:
1. SAMPLE
2. CONTENT PROFILE
3. SEMANTIC PROFILE

Return ONLY JSON:
{"faithfulness": int}
"""

# -----------------------
# Gemini helpers
# -----------------------
def call_json(system_prompt, user_prompt):
    raw = model.generate_content(
        system_prompt + "\n\n" + user_prompt,
        generation_config={"temperature": 0, "response_mime_type": "application/json"}
    )
    return json.loads(raw.text)

# -----------------------
# Text similarity
# -----------------------
def compute_similarity(reference, candidate):
    if not reference or not candidate:
        return {k: None for k in [
            "meteor", "rouge1", "rouge2", "rougeL",
            "bertscore_precision", "bertscore_recall", "bertscore_f1"
        ]}

    ref_tok = word_tokenize(reference)
    cand_tok = word_tokenize(candidate)
    meteor = meteor_score([ref_tok], cand_tok)

    rouge = rouge_scorer.RougeScorer(
        ["rouge1","rouge2","rougeL"], use_stemmer=True
    ).score(reference, candidate)

    P, R, F1 = bertscore([candidate], [reference], lang="en", verbose=False)

    return {
        "meteor": meteor,
        "rouge1": rouge["rouge1"].fmeasure,
        "rouge2": rouge["rouge2"].fmeasure,
        "rougeL": rouge["rougeL"].fmeasure,
        "bertscore_precision": float(P[0]),
        "bertscore_recall": float(R[0]),
        "bertscore_f1": float(F1[0]),
    }


# -----------------------
# Load metadata original descriptions
# -----------------------
def load_original(metadata_path):
    out = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for item in meta:
        out[item["id"]] = item.get("description", "")
    return out


# -----------------------
# MAIN EVALUATION with RESUME
# -----------------------
def evaluate_all(
    baseline_jsonl,
    metadata_path,
    queries_txt,
    relevance_csv,
    output_jsonl
):
    # -------------------------------------
    # Load existing results (Resume)
    # -------------------------------------
    done_ids = set()
    if os.path.exists(output_jsonl):
        print(f"[RESUME] Loading existing results from {output_jsonl}")
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                done_ids.add(rec["dataset_id"])
        print(f"[RESUME] Found {len(done_ids)} completed datasets.")
    else:
        print("[RESUME] No existing results, starting fresh.")

    # Open output file in append mode
    fout = open(output_jsonl, "a", encoding="utf-8")

    # -------------------------------------
    # Load metadata original descriptions
    # -------------------------------------
    original_map = load_original(metadata_path)

    # -------------------------------------
    # Iterate baseline
    # -------------------------------------
    with open(baseline_jsonl, "r", encoding="utf-8") as fin:

        for line in fin:
            if not line.strip():
                continue

            rec = json.loads(line)
            ds_id = rec["dataset_id"]

            # Skip if already evaluated
            if ds_id in done_ids:
                #print(f"[SKIP] {ds_id} already processed.")
                continue

            print(f"=== Evaluating {ds_id} ===")

            sample = rec.get("sample", "")
            content = rec.get("content_profile", {})
            semantic = rec.get("semantic_profile", {})

            ufd = rec.get("ufd", "")
            sfd = rec.get("sfd", "")
            ufd_nyc = rec.get("ufd_nyc", "")
            sfd_nyc = rec.get("sfd_nyc", "")
            hs = rec.get("HandS", "")

            original_desc = original_map.get(ds_id, "")

            # --------------------
            # Gemini ref-free eval
            # --------------------
            def eval_QA(text):
                return call_json(EVAL_PROMPT, text)

            def eval_faith(text):
                MAX_CHARS = 50000
                sample_trunc = sample[:MAX_CHARS]
                content_trunc = json.dumps(content, indent=2)[:MAX_CHARS]
                semantic_trunc = json.dumps(semantic, indent=2)[:MAX_CHARS]
                block = (
                    "SAMPLE:\n" + sample_trunc + "\n\n" +
                    "CONTENT:\n" + content_trunc + "\n\n" +
                    "SEMANTIC:\n" + semantic_trunc + "\n\n" +
                    "DESC:\n" + text
                )
                return call_json(FAITH_PROMPT, block).get("faithfulness", None)

            def build_ref_free(desc):
                out = eval_QA(desc)
                out["faithfulness"] = eval_faith(desc)
                return out

            # Might raise API error → resume protects us
            try:
                rf_hs = build_ref_free(hs)
                rf_ufd = build_ref_free(ufd)
                rf_sfd = build_ref_free(sfd)
                rf_ufd_nyc = build_ref_free(ufd_nyc)
                rf_sfd_nyc = build_ref_free(sfd_nyc)
                rf_original = build_ref_free(original_desc) if original_desc else {}
            except Exception as e:
                print(f"[ERROR] LLM failed on {ds_id}: {e}")
                print("[HINT] Resume will pick up from here next run.")
                fout.close()
                raise e

            # --------------------
            # Similarity
            # --------------------
            sim_hs = compute_similarity(original_desc, hs)
            sim_ufd = compute_similarity(original_desc, ufd)
            sim_sfd = compute_similarity(original_desc, sfd)
            sim_ufd_nyc = compute_similarity(original_desc, ufd_nyc)
            sim_sfd_nyc = compute_similarity(original_desc, sfd_nyc)

            # --------------------
            # WRITE RESULT
            # --------------------
            out = {
                "dataset_id": ds_id,
                "hs": {"ref_free": rf_hs, "similarity": sim_hs},
                "ufd": {"ref_free": rf_ufd, "similarity": sim_ufd},
                "sfd": {"ref_free": rf_sfd, "similarity": sim_sfd},
                "ufd_nyc": {"ref_free": rf_ufd_nyc, "similarity": sim_ufd_nyc},
                "sfd_nyc": {"ref_free": rf_sfd_nyc, "similarity": sim_sfd_nyc},
                "original": {"ref_free": rf_original},
            }

            fout.write(json.dumps(out) + "\n")
            fout.flush()

            print(f"[DONE] {ds_id}")

    fout.close()
    print("Saved →", output_jsonl)


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    evaluate_all(
        baseline_jsonl="../../outputs/baseline_autoddg_descriptions.jsonl",
        metadata_path="../../outputs/metadata_registry.json",
        queries_txt="queries.txt",
        relevance_csv="relevance_matrix.csv",
        output_jsonl="text_eval_results.jsonl"
    )
