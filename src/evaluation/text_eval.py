"""
File: src/evaluation/text_eval.py
Description:
    Evaluates the qualitative aspects of the descriptions using LLM as a judge.
    Metrics: Readability, Faithfulness, Completeness, Conciseness.
    Updated: Now includes 'original' description for fair comparison.
"""

import sys
from pathlib import Path

# ================================================================
# Path Setup
# ================================================================
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

import json
import time
import re
import statistics
from baseline.llm_client import call_llm


# ================================================================
# Path Configuration
# ================================================================
BASE_DIR = src_dir.parent
METADATA_PATH = BASE_DIR / "outputs" / "metadata_registry.json"
BASELINE_PATH = BASE_DIR / "outputs" / "0_baseline_autoddg_descriptions.jsonl"
NYC_PATH = BASE_DIR / "outputs" / "stage2_async_nyc_descriptions.jsonl"
OUTPUT_JSON_PATH = BASE_DIR / "outputs" / "text_eval_results.json"


# ================================================================
# Prompt
# ================================================================
EVAL_PROMPT_TEMPLATE = """
You are an expert data documentation judge. Rate the following dataset description on 4 criteria (1-5 scale):

1. Readability: Is it easy to understand?
2. Faithfulness: Does it align with the provided metadata (columns, title)?
3. Completeness: Does it cover the key information?
4. Conciseness: Is it free of fluff?

Dataset Title: {title}
Columns: {columns}

Description to Evaluate:
"{description}"

Output ONLY a JSON object with integer scores, like this:
{{
  "readability": 4,
  "faithfulness": 5,
  "completeness": 3,
  "conciseness": 4
}}
"""

# ================================================================
# Helpers
# ================================================================
def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except:
        return None

# ================================================================
# Main Logic
# ================================================================
def run_text_evaluation(limit=10):
    print("-" * 60)
    print(" Running Text Quality Evaluation (Original + Baseline + NYC)")
    print("-" * 60)

    # 1. Load metadata (original / H&S)
    if not METADATA_PATH.exists():
        print(f"[ERROR] Metadata not found at {METADATA_PATH}")
        return
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    meta_map = {item["id"]: item for item in metadata}

    # 2. Load baseline AutoDDG descriptions (ufd, sfd)
    if not BASELINE_PATH.exists():
        print(f"[ERROR] Baseline descriptions not found at {BASELINE_PATH}")
        return

    baseline_records = []
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                baseline_records.append(json.loads(line))
    baseline_map = {rec["dataset_id"]: rec for rec in baseline_records}

    # 3. Load NYC AutoDDG descriptions (ufd_nyc, sfd_nyc)
    if not NYC_PATH.exists():
        print(f"[WARN] NYC descriptions not found at {NYC_PATH} - NYC methods will be skipped.")
        nyc_map = {}
    else:
        nyc_records = []
        with open(NYC_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    nyc_records.append(json.loads(line))
        nyc_map = {rec["dataset_id"]: rec for rec in nyc_records}

    # 4. Methods to evaluate
    methods = ["original", "ufd", "sfd", "ufd_nyc", "sfd_nyc"]
    scores = {
        m: {
            "readability": [],
            "faithfulness": [],
            "completeness": [],
            "conciseness": [],
        }
        for m in methods
    }

    def get_description(ds_id: str, method: str) -> str:
        # Original human-written description from metadata (H&S)
        if method == "original":
            return meta_map.get(ds_id, {}).get("description", "")

        # Baseline AutoDDG descriptions
        if method in ("ufd", "sfd"):
            return baseline_map.get(ds_id, {}).get(method, "")

        # NYC AutoDDG descriptions
        if method in ("ufd_nyc", "sfd_nyc"):
            return nyc_map.get(ds_id, {}).get(method, "")

        return ""

    # Limit which datasets we evaluate based on the baseline list
    processing_records = baseline_records[:limit]

    print(f"[INFO] Evaluating {len(processing_records)} datasets (x {len(methods)} methods each)...")

    for i, rec in enumerate(processing_records):
        ds_id = rec["dataset_id"]
        meta = meta_map.get(ds_id, {})
        title = meta.get("name", "Unknown")

        # Robust column parsing
        raw_cols = meta.get("columns", [])
        col_list = []
        for c in raw_cols:
            if isinstance(c, dict):
                col_list.append(c.get("name", str(c)))
            else:
                col_list.append(str(c))
        cols = ", ".join(col_list)

        print(f" [{i+1}/{len(processing_records)}] Evaluating {ds_id}...")

        for method in methods:
            desc_text = get_description(ds_id, method)

            # Skip empty descriptions
            if not desc_text or len(desc_text) < 5:
                continue

            prompt = EVAL_PROMPT_TEMPLATE.format(
                title=title,
                columns=cols,
                description=desc_text,
            )

            try:
                # Call LLM
                resp = call_llm(prompt, temperature=0.0)
                rating = extract_json(resp)

                if rating:
                    for key in scores[method]:
                        val = rating.get(key, 3)
                        scores[method][key].append(val)
                else:
                    pass

            except Exception as e:
                print(f"   [ERROR] LLM call failed for {method}: {e}")

            time.sleep(0.5)

    # 5. Aggregate
    final_results = {}
    print("\n" + "=" * 30)
    print(" FINAL TEXT SCORES (Average)")
    print("=" * 30)

    for m in methods:
        final_results[m] = {}
        print(f" Method: {m}")
        for metric, values in scores[m].items():
            if values:
                avg = statistics.mean(values)
            else:
                avg = 0
            final_results[m][metric] = avg
            print(f"   - {metric:12s}: {avg:.2f}")

    # 6. Save
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("-" * 60)
    print(f" Saved results -> {OUTPUT_JSON_PATH}")
    print("-" * 60)

if __name__ == "__main__":
    run_text_evaluation(limit=20)