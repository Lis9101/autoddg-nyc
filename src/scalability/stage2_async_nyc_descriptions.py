"""
Stage 2: Asynchronous NYC Descriptions (AutoDDG-NYC, scalable version)

Design:
  - Input: Spark-based Stage 1 profiles from:
        outputs/stage1_spark_profiles.jsonl

        Each record (simplified) looks like:
        {
          "dataset_id": "8wbx-tsch",
          "status": "ok",
          "row_count": 12345,
          "column_count": 12,
          "columns": ["col1", "col2", ...]
        }

  - This stage:
      * Filters to status=="ok" datasets.
      * For each dataset (concurrently, with bounded concurrency):
          - Finds the CSV file (same logic as baseline_autoddg).
          - Recomputes the full Pandas-based content_profile using
                baseline.profiling_autoddg.profile_dataset
            so the pipeline matches the original baseline.
          - Builds a semantic profile with
                baseline.semantic_autoddg.build_semantic_profile
          - Generates NYC-specific UFD/SFD via
                baseline.descriptions_nyc.generate_ufd_nyc / generate_sfd_nyc
      * Appends results to:
            outputs/stage2_async_nyc_descriptions.jsonl

  - Features:
      * Bounded concurrency via asyncio (default concurrency=5).
      * Resume support: skips datasets already processed with status=="ok"
        in the Stage 2 output file.
      * Logs runtime and throughput (#ok datasets per hour).

Usage:
    python src/scalability/stage2_async_nyc_descriptions.py \
        --max_datasets 50 \
        --concurrency 5
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

# ----- Path setup -----
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]        # .../src
ROOT_DIR = THIS_FILE.parents[2]       # .../autoddg-nyc

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ----- Imports from baseline pipeline -----
from baseline.profiling_autoddg import profile_dataset
from baseline.semantic_autoddg import build_semantic_profile
from baseline.descriptions_nyc import generate_ufd_nyc, generate_sfd_nyc

# ----- Project paths -----
STAGE1_PROFILES_PATH = ROOT_DIR / "outputs" / "stage1_spark_profiles.jsonl"
REGISTRY_PATH = ROOT_DIR / "outputs" / "metadata_registry.json"
OUTPUT_PATH = ROOT_DIR / "outputs" / "stage2_async_nyc_descriptions.jsonl"
DATA_DIR_ROOT = ROOT_DIR / "data"


# ================================================================
# Helpers: file loading
# ================================================================

def load_stage1_profiles(max_datasets: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load Stage 1 Spark profiles and filter to status=="ok".
    """
    if not STAGE1_PROFILES_PATH.exists():
        raise FileNotFoundError(
            f"Stage 1 profiles not found at {STAGE1_PROFILES_PATH}. "
            "Run stage1_spark_profiling.py first."
        )

    profiles: List[Dict[str, Any]] = []
    with STAGE1_PROFILES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok":
                profiles.append(rec)

    if max_datasets is not None:
        profiles = profiles[:max_datasets]

    return profiles


def load_registry_map() -> Dict[str, Dict[str, Any]]:
    """
    Load metadata_registry.json and return a dict keyed by dataset_id (string).
    """
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Registry file not found at {REGISTRY_PATH}. "
            "Make sure the registry has been generated."
        )

    registry = json.load(REGISTRY_PATH.open("r", encoding="utf-8"))
    reg_map: Dict[str, Dict[str, Any]] = {}
    for item in registry:
        ds_id = str(item.get("id"))
        if ds_id:
            reg_map[ds_id] = item
    return reg_map


def load_stage2_processed_ids(path: Path) -> set[str]:
    """
    Resume support: datasets already processed with status=="ok"
    in Stage 2 output will be skipped on subsequent runs.
    """
    if not path.exists():
        return set()

    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ds_id = rec.get("dataset_id")
            status = rec.get("status")
            if ds_id and status == "ok":
                ids.add(str(ds_id))
    return ids


def find_csv_path(dataset_id: str) -> Path:
    """
    Same CSV resolution strategy as baseline_autoddg.py:
      1) data/csv_files/<id>.csv
      2) data/<id>.csv
      3) outputs/<id>.csv
    """
    # Strategy 1
    csv_path = DATA_DIR_ROOT / "csv_files" / f"{dataset_id}.csv"

    # Strategy 2
    if not csv_path.exists():
        csv_path = DATA_DIR_ROOT / f"{dataset_id}.csv"

    # Strategy 3
    if not csv_path.exists():
        csv_path = ROOT_DIR / "outputs" / f"{dataset_id}.csv"

    return csv_path


# ================================================================
# Core per-dataset processing (sync, run in executor)
# ================================================================

def process_dataset_sync(
    profile_obj: Dict[str, Any],
    registry_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Synchronous processing for a single dataset, to be run in a thread executor.

    Steps:
      - Look up metadata in registry_map.
      - Resolve CSV path.
      - Recompute Pandas-based content_profile (profile_dataset) so the
        pipeline matches baseline_autoddg logic.
      - Build semantic_profile (build_semantic_profile).
      - Generate NYC-specific UFD + SFD descriptions (generate_ufd_nyc / generate_sfd_nyc).
    """

    dataset_id = profile_obj["dataset_id"]
    result: Dict[str, Any] = {"dataset_id": dataset_id}

    # If Stage 1 reported non-ok, just propagate
    if profile_obj.get("status") != "ok":
        result["status"] = profile_obj.get("status", "error")
        result["error"] = profile_obj.get("error")
        return result

    # Metadata lookup
    meta = registry_map.get(dataset_id, {})
    metadata = meta.get("metadata", {})
    title = meta.get("name", "")
    description = meta.get("description", "")
    topic = meta.get("topic", "")
    sample_row_count = meta.get("sample_row_count", None)

    # CSV path
    csv_path = find_csv_path(dataset_id)
    if not csv_path.exists():
        msg = f"CSV not found for dataset_id={dataset_id} (looked under data/ and outputs/)"
        result["status"] = "missing_csv"
        result["error"] = msg
        return result

    try:
        # 1) Data-driven content profile (Pandas) — same as baseline
        content_profile = profile_dataset(
            csv_path=csv_path,
            dataset_id=dataset_id,
            max_rows=sample_row_count,
        )

        # 2) Semantic profile (LLM)
        semantic_profile = build_semantic_profile(
            csv_path=csv_path,
            content_profile=content_profile,
            title=title,
            description=description or "",
            topic=topic or "",
            max_semantic_columns=12,
        )

        # 3) NYC-specific descriptions (AutoDDG-NYC)
        ufd_nyc = generate_ufd_nyc(
            csv_path,
            content_profile,
            semantic_profile,
            topic or "",
            metadata,
        )
        sfd_nyc = generate_sfd_nyc(ufd_nyc, metadata)

        result.update(
            {
                "status": "ok",
                "title": title,
                "topic": topic,
                "metadata": metadata,
                "content_profile": content_profile,
                "semantic_profile": semantic_profile,
                "ufd_nyc": ufd_nyc,
                "sfd_nyc": sfd_nyc,
            }
        )
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# ================================================================
# Async wrapper + driver
# ================================================================

async def process_dataset_async(
    profile_obj: Dict[str, Any],
    registry_map: Dict[str, Dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """
    Async wrapper: run process_dataset_sync in a thread pool, with
    bounded concurrency via a semaphore.
    """
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            process_dataset_sync,
            profile_obj,
            registry_map,
        )


async def run_stage2_async(
    max_datasets: Optional[int] = None,
    concurrency: int = 5,
):
    # Load Stage 1 Spark profiles
    profiles = load_stage1_profiles(max_datasets=max_datasets)
    if not profiles:
        print("[WARN] No Stage 1 profiles loaded (status==\"ok\"). Nothing to do.")
        return

    # Resume support: skip datasets already processed as ok in Stage 2
    processed_ids = load_stage2_processed_ids(OUTPUT_PATH)
    pending_profiles = [p for p in profiles if p["dataset_id"] not in processed_ids]

    total = len(pending_profiles)
    print(f"[INFO] Stage 2 starting with {len(profiles)} dataset profiles.")
    print(f"[INFO] Remaining datasets to process in Stage 2: {total}")

    if total == 0:
        print("[INFO] All datasets already processed in Stage 2. Exiting.")
        return

    # Load registry map
    registry_map = load_registry_map()

    # Prepare async tasks
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        process_dataset_async(p, registry_map, sem)
        for p in pending_profiles
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    num_ok = 0
    num_error = 0
    num_missing = 0
    completed = 0

    t0 = time.time()

    with OUTPUT_PATH.open("a", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            completed += 1

            ds_id = res.get("dataset_id")
            status = res.get("status", "error")

            if status == "ok":
                num_ok += 1
            elif status == "missing_csv":
                num_missing += 1
            else:
                num_error += 1

            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()

            print(f"[INFO] Completed {completed}/{total} (id={ds_id}, status={status})")

    t1 = time.time()
    wall_clock = t1 - t0
    throughput = (num_ok / wall_clock * 3600) if wall_clock > 0 and num_ok > 0 else 0.0

    print("--------------------------------------------------------")
    print(" Stage 2 (Async NYC Descriptions) Completed")
    print(f"   Successful (ok)   : {num_ok}")
    print(f"   Missing CSV       : {num_missing}")
    print(f"   Errors            : {num_error}")
    print(f"   Total datasets    : {total}")
    #print(f"   Wall-clock time   : {wall_clock:.1f} seconds")
    #print(f"   Throughput        : {throughput:.1f} datasets/hour (ok only)")
    print(f"   Output written to : {OUTPUT_PATH}")
    print("--------------------------------------------------------")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_datasets",
        type=int,
        default=None,
        help="Optional cap on number of Stage 1 profiles to process.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max number of datasets processed concurrently.",
    )
    args = parser.parse_args()

    asyncio.run(run_stage2_async(max_datasets=args.max_datasets, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
