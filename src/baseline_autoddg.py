"""
File: baseline_autoddg.py
Description:
    Main runner for the Baseline AutoDDG pipeline.
      - Load datasets listed in metadata_registry.json
      - Build data-driven content profiles (no LLM)
      - Build semantic profiles using Gemini (column-limited)
      - Generate UFD and SFD descriptions
      - Append results to data/0_baseline_autoddg_descriptions.jsonl

    Features:
      - Skips datasets already processed (resume support)
      - Optional --max_datasets batching
      - Stops immediately on Gemini quota errors

    Usage:
        python src/baseline_autoddg.py
        python src/baseline_autoddg.py --max_datasets 50
"""


import argparse
import json
from pathlib import Path
import time

from baseline.profiling_autoddg import profile_dataset
from baseline.semantic_autoddg import build_semantic_profile
from baseline.descriptions_autoddg import generate_ufd, generate_sfd


# Paths
OUTPUT_PATH = Path("data/0_baseline_autoddg_descriptions.jsonl")
REGISTRY_PATH = Path("data/metadata_registry.json")
RUNTIME_LOG_PATH = Path("data/0_baseline_autoddg_runtime.jsonl")


# Helper functions
def load_processed_ids(output_path: Path) -> set:
    """
    Read existing JSONL and return the set of dataset_ids already processed.
    """
    processed = set()
    if not output_path.exists():
        return processed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                did = rec.get("dataset_id")
                if did:
                    processed.add(did)
            except json.JSONDecodeError:
                # ignore malformed lines
                continue

    return processed


def derive_topic_from_metadata(item: dict) -> str:
    """
    Gets topic from registry metadata.
    Use 'category' if present, otherwise use first 2–3 words of 'name'
    """
    category = item.get("category")
    if category:
        return category

    title = item.get("name") or ""
    words = title.split()
    if not words:
        return "Unknown dataset"
    return " ".join(words[:3])


def is_quota_error(e: Exception) -> bool:
    """
    Detect Gemini quota errors from the exception message.
    """
    msg = str(e).lower()
    if "quota" in msg:
        return True
    if "429" in msg:
        return True
    if "exceeded your current quota" in msg:
        return True
    return False

def log_runtime(dataset_id: str, duration_sec: float):
    """
    Append a runtime record to data/0_baseline_autoddg_runtime.jsonl.
    """
    RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset_id": dataset_id,
        "runtime_sec": round(duration_sec, 4),
    }
    with RUNTIME_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# Main runner
def run_baseline_autoddg(max_datasets: int | None = None):
    # Check registry file
    if not REGISTRY_PATH.exists():
        print(f"Error: registry not found at {REGISTRY_PATH}")
        print("  Make sure you have run `download_from_registry.py` or pulled the file from git.")
        return

    # Load registry
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    # Ensure output dir exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load any already-processed dataset_ids to support resume
    processed_ids = load_processed_ids(OUTPUT_PATH)
    print(f" Found {len(processed_ids)} datasets already processed.\n")

    print("-" * 60)
    print(" Running Baseline AutoDDG")
    print("-" * 60)

    done_this_run = 0
    skipped_existing = 0
    failed = 0

    for item in registry:
        dataset_id = item["id"]
        title = item.get("name", dataset_id)
        short_title = title if len(title) <= 60 else title[:57] + "..."

        # Skip already-processed dataset without printing anything
        if dataset_id in processed_ids:
            skipped_existing += 1
            continue

        print(f"\nProcessing [{dataset_id}] {short_title}")

        # Respect max_datasets limit for this run
        if max_datasets is not None and done_this_run >= max_datasets:
            break

        description = item.get("description", "") or ""
        csv_path = Path(item.get("local_path", f"data/{dataset_id}.csv"))
        sample_rows_limit = item.get("sample_row_count", None)

        if not csv_path.exists():
            print(f"[WARN] Missing CSV for {dataset_id}: {csv_path}")
            failed += 1
            continue

        try:
            start_time = time.time()   # <--- START TIMER

            # 1) Content profile (non-LLM)
            content_profile = profile_dataset(
                csv_path=csv_path,
                dataset_id=dataset_id,
                max_rows=sample_rows_limit,
            )

            # 2) Topic from metadata (no LLM call)
            topic = derive_topic_from_metadata(item)

            # 3) Semantic profile (LLM; quota-aware inside build_semantic_profile)
            semantic_profile = build_semantic_profile(
                csv_path=csv_path,
                content_profile=content_profile,
                title=title,
                description=description,
                topic=topic,
                max_semantic_columns=12,  # adjust up/down for more/less LLM usage
            )

            # 4) Final descriptions (UFD + SFD) via LLM
            ufd = generate_ufd(
                csv_path=csv_path,
                content_profile=content_profile,
                semantic_profile=semantic_profile,
                topic=topic,
            )
            sfd = generate_sfd(topic=topic, ufd=ufd)

            # 5) Append record to JSONL
            record = {
                "dataset_id": dataset_id,
                "title": title,
                "topic": topic,
                "ufd": ufd,
                "sfd": sfd,
            }

            with OUTPUT_PATH.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

            # END TIMER & LOG
            duration = time.time() - start_time
            log_runtime(dataset_id, duration)

            print(f"✓ Done [{dataset_id}]")

            done_this_run += 1

        except Exception as e:
            if is_quota_error(e):
                print("\n" + "=" * 60)
                print(f" STOPPING: Gemini quota exceeded on dataset {dataset_id}")
                print("  You may resume later; completed records are already saved.")
                print("=" * 60 + "\n")
                raise(e)
                # Exit the function immediately so we don't keep spamming errors
                break

            print(f"[ERROR] Failed on {dataset_id}: {e}")
            failed += 1

    print("-" * 60)
    print(" Baseline AutoDDG Completed (this run)")
    print(f"   New datasets processed : {done_this_run}")
    print(f"   Already processed skip : {skipped_existing}")
    print(f"   Failed                 : {failed}")
    print(f" Output: {OUTPUT_PATH}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Baseline AutoDDG over metadata_registry.json."
    )
    parser.add_argument(
        "--max_datasets",
        type=int,
        default=None,
        help="Maximum number of datasets to process in this run.",
    )
    args = parser.parse_args()

    run_baseline_autoddg(max_datasets=args.max_datasets)
