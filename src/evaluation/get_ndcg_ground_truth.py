"""
File: run_relevance_auto_csv.py

Description:
    Generate a relevance matrix (datasets × queries) using Gemini 2.5 flash.
    LLM sees more metadata: category, agency, columns, description, etc.

Output:
    relevance_matrix.csv
"""
import random
import os
import csv
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# ============================================================
#                 LOAD API KEY & CONFIGURE MODEL
# ============================================================

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"


# ============================================================
#            STEP 1 — LOAD METADATA & PICK DATASETS
# ============================================================

def load_metadata(path=None):
  
    if path is None:

        # .parent = src/evaluation
        # .parent.parent = src
        # .parent.parent.parent =  autoddg-nyc
        base_dir = Path(__file__).resolve().parent.parent.parent
        path = base_dir / "outputs" / "metadata_registry.json"

    print(f"[DEBUG] Loading metadata from: {path}")  

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_datasets(metadata, target_count=50, seed=42):
    """
    Deterministic random selection:
        - Shuffle metadata using a fixed seed
        - Take first target_count
        - Fully reproducible
    """
    random.seed(seed)
    shuffled = metadata[:]        # copy
    random.shuffle(shuffled)
    return shuffled[:target_count]


# ============================================================
#          STEP 2 — GENERATE 20 DETERMINISTIC QUERIES
# ============================================================

def generate_queries(metadata=None, num_queries=20):

    queries = []
    

    base_dir = Path(__file__).resolve().parent

    queries_path = base_dir / "queries.txt"

    print(f"[DEBUG] Loading queries from: {queries_path}") 

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    return queries[:num_queries]



# ============================================================
#                STEP 3 — GEMINI RELEVANCE JUDGE
# ============================================================

JUDGE_PROMPT = """
You are rating whether a dataset in NYC open data is relevant to a user query.

Return ONLY one character:
0 = not relevant
1 = partially relevant
2 = highly relevant

Return ONLY: 0 or 1 or 2
Do NOT return any words, explanations, or JSON.
"""

def judge_relevance(query, ds):
    """LLM sees full metadata: name, description, category, agency, columns."""

    name = ds.get("name", "")
    desc = ds.get("description", "")
    category = ds.get("category", "")
    agency = ds.get("agency", "")
    columns = ds.get("columns", [])

    metadata_text = f"""
Dataset Name: {name}
Category: {category}
Agency: {agency}
Columns: {', '.join(columns)}

Description:
{desc}
"""

    model = genai.GenerativeModel(MODEL_NAME)

    user_prompt = f"""
Query:
{query}

Dataset Metadata:
{metadata_text}

Rate relevance: output only 0 or 1 or 2.
"""

    resp = model.generate_content(
        JUDGE_PROMPT + "\n\n" + user_prompt,
        generation_config={
            "temperature": 0,
            "response_mime_type": "text/plain"   # ensure single-digit output
        }
    )

    text = resp.text.strip()

    # extract single digit
    digits = [c for c in text if c in "012"]
    if not digits:
        return 0
    return int(digits[0])


# ============================================================
#                        MAIN LOOP
# ============================================================

def main():
    metadata = load_metadata()
    selected = choose_datasets(metadata, target_count=50)
    queries = generate_queries(metadata, num_queries=20)

    print(f"Selected datasets: {len(selected)}")
    print(f"Generated queries: {len(queries)}")

    output_path = "relevance_matrix.csv"

    header = ["dataset_id"] + [f"query_{i+1}" for i in range(len(queries))]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ds in selected:
            dsid = ds["id"]
            print(f"\n===== DATASET {dsid} =====")
            print(f" Description: {ds['description']}")
            print(f"\n==========================")
            row = [dsid]

            for i, query in enumerate(queries):
                print(f"  Query {i+1}: {query}")
                score = judge_relevance(query, ds)
                row.append(score)

                print(f"    → relevance = {score}")

            writer.writerow(row)

    print(f"\nDone! Saved to {output_path}\n")


# ============================================================
# Start
# ============================================================
if __name__ == "__main__":
    main()
