"""
File: baseline_autoddg.py
Description:
    Main runner for the Baseline AutoDDG pipeline.
      - Load datasets listed in metadata_registry.json
      - Build data-driven content profiles (no LLM)
      - Build semantic profiles using Gemini (column-limited)
      - Generate H+S (Header + Sample) description via LLM
      - Generate UFD and SFD descriptions via LLM
      - [NEW] Generate AutoDDG-NYC descriptions (Domain Customization)
      - Append results to outputs/baseline_autoddg_descriptions.jsonl

    Features:
      - Skips datasets already processed (resume support)
      - Optional --max_datasets batching
      - Stops immediately on Gemini quota errors
      - Automatic absolute path resolution

    Usage:
        python src/baseline_autoddg.py
"""

import argparse
import json
import time
import pandas as pd
from pathlib import Path

# Import baseline functions
from baseline.descriptions_autoddg import sample_rows, generate_ufd, generate_sfd
from baseline.profiling_autoddg import profile_dataset
from baseline.semantic_autoddg import build_semantic_profile
from baseline.llm_client import call_llm

# [Week 3] Import NYC customization functions
from baseline.descriptions_nyc import generate_ufd_nyc, generate_sfd_nyc


# ================================================================
# File paths (Fixed with Absolute Paths)
# ================================================================

# Determine the project root directory regardless of where the script is run
# Current file is in src/baseline_autoddg.py -> parent=src -> parent.parent=root
BASE_DIR = Path(__file__).resolve().parent.parent

# Define absolute paths for outputs
OUTPUT_PATH = BASE_DIR / "outputs" / "baseline_autoddg_descriptions.jsonl"
REGISTRY_PATH = BASE_DIR / "outputs" / "metadata_registry.json"
RUNTIME_LOG_PATH = BASE_DIR / "outputs" / "baseline_autoddg_runtime.jsonl"

# The relevance matrix is in the project root directory
RELEVANCE_MATRIX_PATH = BASE_DIR / "relevance_matrix.csv"

# Define the base data directory
DATA_DIR_ROOT = BASE_DIR / "data"


# ================================================================
# Helpers
# ================================================================

def load_processed_ids(path: Path) -> set:
    """Return dataset_ids already processed (resume support)."""
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("dataset_id"):
                    ids.add(rec["dataset_id"])
            except:
                pass
    return ids


def derive_topic_from_metadata(item: dict) -> str:
    """Prefer category, fallback to first 2–3 words of name."""
    if item.get("category"):
        return item["category"]
    name = item.get("name", "")
    parts = name.split()
    return " ".join(parts[:3]) if parts else "Unknown dataset"


def is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("quota" in msg) or ("429" in msg) or ("exceeded" in msg)


def log_runtime(dataset_id: str, duration: float):
    RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "dataset_id": dataset_id,
            "runtime_sec": round(duration, 3)
        }) + "\n")


# ================================================================
# H+S (Header + Sample) LLM generation
# ================================================================

def generate_hs_via_llm(headers, sample):
    """
    LLM-based Header + Sample baseline.
    Uses same logic as evaluator version.
    """
    header_list = ", ".join(headers)
    prompt = f"""
Generate a short dataset description using ONLY:

Headers:
{header_list}

Sample:
{sample}

Return plain text. Do NOT use JSON.
"""
    return call_llm(prompt, temperature=0.0)


# ================================================================
# Main Runner
# ================================================================

def run_baseline_autoddg(max_datasets: int | None = None, test_mode: bool = False):
    # Load registry
    if not REGISTRY_PATH.exists():
        print(f"[ERROR] Registry file missing at {REGISTRY_PATH}")
        return

    registry = json.load(REGISTRY_PATH.open("r", encoding="utf-8"))

    # ----------------------------
    # TEST MODE: filter registry
    # ----------------------------
    if test_mode:
        print(f"[DEBUG] Reading relevance matrix from: {RELEVANCE_MATRIX_PATH}")
        
        if not RELEVANCE_MATRIX_PATH.exists():
            print("[ERROR] Relevance matrix file not found!")
            print("Please run 'python src/evaluation/get_ndcg_ground_truth.py' first.")
            return

        rel_df = pd.read_csv(RELEVANCE_MATRIX_PATH)

        relevance_ids = set(rel_df["dataset_id"].astype(str))

        print("[DEBUG] relevance dataset count =", len(relevance_ids))
        
        # === Mask registry: only keep datasets appearing in relevance ===
        original_registry_size = len(registry)
        registry = [item for item in registry if str(item.get("id")) in relevance_ids]

        print(f"[DEBUG] registry filtered: {original_registry_size} -> {len(registry)}")

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed = load_processed_ids(OUTPUT_PATH)
    print(f"Found {len(processed)} datasets already processed.\n")

    print("-" * 60)
    print(" Running Baseline AutoDDG")
    print("-" * 60)

    done_this_run = 0
    failed = 0

    for item in registry:
        ds_id = item["id"]
        title = item.get("name", ds_id)
        topic = derive_topic_from_metadata(item)

        if ds_id in processed:
            continue

        if max_datasets is not None and done_this_run >= max_datasets:
            break

        # [FIX] Try to find the CSV file in multiple possible locations
        # Strategy 1: data/csv_files/ID.csv
        csv_path = DATA_DIR_ROOT / "csv_files" / f"{ds_id}.csv"
        
        # Strategy 2: data/ID.csv (based on your screenshot)
        if not csv_path.exists():
             csv_path = DATA_DIR_ROOT / f"{ds_id}.csv"
        
        # Strategy 3: outputs/ID.csv (fallback)
        if not csv_path.exists():
            csv_path = BASE_DIR / "outputs" / f"{ds_id}.csv"
        
        if not csv_path.exists():
            print(f"[WARN] Missing CSV for {ds_id} (Checked data/ and data/csv_files/)")
            failed += 1
            continue

        print(f"\nProcessing [{ds_id}] {title}")

        try:
            start = time.time()

            # ---------------------------------------------------------
            # 1) Content Profile (non-LLM)
            # ---------------------------------------------------------
            content_profile = profile_dataset(
                csv_path=csv_path,
                dataset_id=ds_id,
                max_rows=item.get("sample_row_count")
            )

            # ---------------------------------------------------------
            # 2) Sample rows
            # ---------------------------------------------------------
            sample_text = sample_rows(csv_path)

            # ---------------------------------------------------------
            # 3) Semantic Profile (LLM)
            # ---------------------------------------------------------
            semantic_profile = build_semantic_profile(
                csv_path=csv_path,
                content_profile=content_profile,
                title=title,
                description=item.get("description", ""),
                topic=topic,
                max_semantic_columns=12
            )

            time.sleep(1)

            # ---------------------------------------------------------
            # 4) H+S baseline (LLM)
            # ---------------------------------------------------------
            headers = [col["name"] for col in content_profile["columns"]]
            hs_desc = generate_hs_via_llm(headers, sample_text)

            time.sleep(1)

            # ---------------------------------------------------------
            # 5) UFD / SFD (LLM - Baseline)
            # ---------------------------------------------------------
            ufd = generate_ufd(
                csv_path=csv_path,
                content_profile=content_profile,
                semantic_profile=semantic_profile,
                topic=topic,
            )

            time.sleep(1)

            sfd = generate_sfd(topic=topic, ufd=ufd)

            time.sleep(1)

            # ---------------------------------------------------------
            # 6) AutoDDG-NYC (Week 3: Domain Customization)
            # ---------------------------------------------------------
            # Use the specialized functions to generate NYC-tailored descriptions.
            ufd_nyc = generate_ufd_nyc(
                csv_path=csv_path,
                content_profile=content_profile,
                semantic_profile=semantic_profile,
                topic=topic,
                metadata=item
            )

            time.sleep(1)

            sfd_nyc = generate_sfd_nyc(ufd_nyc=ufd_nyc, metadata=item)

            # ---------------------------------------------------------
            # 7) Save Record
            # ---------------------------------------------------------
            record = {
                "dataset_id": ds_id,
                "title": title,
                "topic": topic,

                "sample": sample_text,
                "content_profile": content_profile,
                "semantic_profile": semantic_profile,

                # HandS before UFD
                "HandS": hs_desc,

                "ufd": ufd,
                "sfd": sfd,

                # Real NYC versions
                "ufd_nyc": ufd_nyc,
                "sfd_nyc": sfd_nyc
            }

            with OUTPUT_PATH.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

            duration = time.time() - start
            log_runtime(ds_id, duration)

            print(f"✓ Done [{ds_id}]")
            done_this_run += 1

        except Exception as e:
            if is_quota_error(e):
                print("=" * 60)
                print("Gemini quota exhausted. Stopping.")
                print("=" * 60)
                raise e
            failed += 1
            print(f"[ERROR] Failed {ds_id}: {e}")

    print("-" * 60)
    print(" Baseline AutoDDG Completed")
    print(f"   New processed: {done_this_run}")
    print(f"   Failed       : {failed}")
    print(f" Output saved to {OUTPUT_PATH}")
    print("-" * 60)


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_datasets", type=int, default=None)
    args = parser.parse_args()

    run_baseline_autoddg(max_datasets=args.max_datasets, test_mode=True)
