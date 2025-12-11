"""
Stage 1: Spark-based Profiling for AutoDDG / AutoDDG-NYC.

Description:
  - Uses PySpark to compute minimal dataset profiles (row_count, column_count, column names).
  - Writes one JSON record per dataset to outputs/stage1_spark_profiles.jsonl.
  - Kept intentionally minimal to avoid Windows stability issues with Spark integration.

"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException

# ----- Path setup -----
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]        # .../src
ROOT_DIR = THIS_FILE.parents[2]       # .../autoddg-nyc

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ----- Project paths -----
OUTPUT_PATH = ROOT_DIR / "outputs" / "stage1_spark_profiles.jsonl"
REGISTRY_PATH = ROOT_DIR / "outputs" / "metadata_registry.json"
DATA_DIR_ROOT = ROOT_DIR / "data"


# ================================================================
# Helpers
# ================================================================

def load_registry(max_datasets: Optional[int] = None) -> List[Dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Registry file not found at {REGISTRY_PATH}. "
            "Please run the registry generation script first."
        )
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if max_datasets is not None:
        registry = registry[:max_datasets]
    return registry


def find_csv_path(dataset_id: str) -> Path:
    """
    Same search strategy as baseline_autoddg.py:
      1) data/csv_files/<id>.csv
      2) data/<id>.csv
      3) outputs/<id>.csv
    """
    # 1) data/csv_files/<id>.csv
    csv_path = DATA_DIR_ROOT / "csv_files" / f"{dataset_id}.csv"

    # 2) data/<id>.csv
    if not csv_path.exists():
        csv_path = DATA_DIR_ROOT / f"{dataset_id}.csv"

    # 3) outputs/<id>.csv
    if not csv_path.exists():
        csv_path = ROOT_DIR / "outputs" / f"{dataset_id}.csv"

    return csv_path


def load_processed_ids(path: Path) -> set[str]:
    """
    Resume support: read existing JSONL output and collect dataset_ids
    that are already profiled successfully (status == "ok").
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


def profile_with_spark(
    spark: SparkSession,
    dataset_id: str,
    csv_path: Path,
) -> Dict[str, Any]:
    """
    Use Spark DataFrame API to profile ONE dataset.

    Returns a dict:
      {
        "dataset_id": ...,
        "status": "ok",
        "row_count": ...,
        "column_count": ...,
        "columns": [...]
      }
    """
    result: Dict[str, Any] = {"dataset_id": dataset_id}

    try:
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(csv_path))
        )
    except AnalysisException as e:
        result["status"] = "error"
        result["error"] = f"Spark failed to read CSV: {e}"
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Unexpected error reading CSV: {e}"
        return result

    try:
        row_count = df.count()
        cols = df.columns
        col_count = len(cols)

        result.update(
            {
                "status": "ok",
                "row_count": row_count,
                "column_count": col_count,
                "columns": cols,
            }
        )
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Error during Spark profiling: {e}"

    return result


# ================================================================
# Main runner
# ================================================================

def run_spark_profiling(max_datasets: Optional[int] = None):
    registry = load_registry(max_datasets=max_datasets)

    if not registry:
        print("[WARN] Registry is empty. Nothing to profile.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = load_processed_ids(OUTPUT_PATH)
    print(f"[INFO] Found {len(processed_ids)} datasets already profiled (status=ok).")

    remaining = [item for item in registry if str(item.get("id")) not in processed_ids]
    print(f"[INFO] Remaining datasets to profile with Spark: {len(remaining)}")

    if not remaining:
        print("[INFO] All datasets in registry are already profiled. Nothing to do.")
        return

    # Start Spark ONCE, reuse for all datasets
    spark = (
        SparkSession.builder
        .appName("AutoDDG-Stage1-SparkProfiling-Minimal")
        .getOrCreate()
    )

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(remaining, 1):
        ds_id = str(item.get("id"))
        if idx == 1 or idx % 10 == 0:
            print(f"[INFO] Spark profiling dataset {idx}/{len(remaining)} (id={ds_id})...")

        csv_path = find_csv_path(ds_id)
        if not csv_path.exists():
            results.append(
                {
                    "dataset_id": ds_id,
                    "status": "missing_csv",
                    "error": f"CSV not found at {csv_path}",
                }
            )
            continue

        rec = profile_with_spark(spark, ds_id, csv_path)
        results.append(rec)

    # Stop Spark
    spark.stop()

    # Append results to JSONL
    with OUTPUT_PATH.open("a", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    num_ok = sum(1 for r in results if r.get("status") == "ok")
    num_missing = sum(1 for r in results if r.get("status") == "missing_csv")
    num_error = sum(1 for r in results if r.get("status") == "error")

    print("--------------------------------------------------------")
    print(" Stage 1 (Spark Profiling) Completed")
    print(f"   Successful profiles : {num_ok}")
    print(f"   Missing CSVs        : {num_missing}")
    print(f"   Errors              : {num_error}")
    print(f"   Output written to   : {OUTPUT_PATH}")
    print("--------------------------------------------------------")


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_datasets",
        type=int,
        default=None,
        help="Optional cap on the number of datasets profiled from the registry.",
    )
    args = parser.parse_args()

    run_spark_profiling(max_datasets=args.max_datasets)
