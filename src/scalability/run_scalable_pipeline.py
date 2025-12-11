"""
Driver for the scalable AutoDDG-NYC pipeline.

This script runs:
  1) Stage 1 (Spark-based profiling)
  2) Stage 2 (async LLM NYC descriptions)

and measures:
  - Stage 1 runtime
  - Stage 2 runtime
  - Total runtime
  - Throughput (newly generated descriptions per hour, based on
    NEW ok records added in this run)

It also appends a summary record to:
  outputs/scalable_pipeline_runtime.jsonl

Usage:
    python src/scalability/run_scalable_pipeline.py \
        --max_datasets 50 \
        --concurrency 5

You can skip Stage 1 (e.g. if already done) with:
    python src/scalability/run_scalable_pipeline.py \
        --max_datasets 50 \
        --concurrency 5 \
        --skip_stage1
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Set

# ----- Path setup -----
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]        # .../src
ROOT_DIR = THIS_FILE.parents[2]       # .../autoddg-nyc

STAGE1_SCRIPT = SRC_DIR / "scalability" / "stage1_spark_profiling.py"
STAGE2_SCRIPT = SRC_DIR / "scalability" / "stage2_async_nyc_descriptions.py"

STAGE2_OUTPUT = ROOT_DIR / "outputs" / "stage2_async_nyc_descriptions.jsonl"
RUNTIME_OUTPUT = ROOT_DIR / "outputs" / "scalable_pipeline_runtime.jsonl"


def load_ok_ids_from_stage2(path: Path) -> Set[str]:
    """
    Return the set of dataset_ids with status=="ok" from the Stage 2 output file.
    Used to compute how many new datasets were processed in THIS run.
    """
    ids: Set[str] = set()
    if not path.exists():
        return ids

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok":
                ds_id = rec.get("dataset_id")
                if ds_id:
                    ids.add(str(ds_id))
    return ids


def run_stage1(max_datasets: int | None) -> float:
    """
    Run Stage 1 script as a subprocess and return its wall-clock runtime in seconds.
    """
    cmd = [sys.executable, str(STAGE1_SCRIPT)]
    if max_datasets is not None:
        cmd.extend(["--max_datasets", str(max_datasets)])

    print(f"[RUN] Stage 1 command: {' '.join(cmd)}")
    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    elapsed = end - start
    print(f"[DONE] Stage 1 completed in {elapsed:.1f} seconds.")
    return elapsed


def run_stage2(max_datasets: int | None, concurrency: int) -> tuple[float, int]:
    """
    Run Stage 2 script as a subprocess and return:
      - runtime in seconds
      - number of NEW ok datasets produced in this run
    """
    before_ids = load_ok_ids_from_stage2(STAGE2_OUTPUT)

    cmd = [sys.executable, str(STAGE2_SCRIPT)]
    if max_datasets is not None:
        cmd.extend(["--max_datasets", str(max_datasets)])
    cmd.extend(["--concurrency", str(concurrency)])

    print(f"[RUN] Stage 2 command: {' '.join(cmd)}")
    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    elapsed = end - start

    after_ids = load_ok_ids_from_stage2(STAGE2_OUTPUT)
    new_ids = after_ids - before_ids
    num_new = len(new_ids)

    print(f"[DONE] Stage 2 completed in {elapsed:.1f} seconds.")
    print(f"[INFO] New ok datasets in this run: {num_new}")
    return elapsed, num_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_datasets",
        type=int,
        default=None,
        help="Optional cap on number of datasets processed in Stage 1/2.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max number of datasets processed concurrently in Stage 2.",
    )
    parser.add_argument(
        "--skip_stage1",
        action="store_true",
        help="If set, do not run Stage 1 (use existing stage1_spark_profiles.jsonl).",
    )

    args = parser.parse_args()

    RUNTIME_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Stage 1
    stage1_time: float | None = None
    if args.skip_stage1:
        print("[INFO] Skipping Stage 1 (as requested).")
    else:
        stage1_time = run_stage1(args.max_datasets)

    # Stage 2
    stage2_time, num_new = run_stage2(args.max_datasets, args.concurrency)

    total_time = time.time() - total_start
    throughput = (num_new / total_time * 3600) if total_time > 0 and num_new > 0 else 0.0

    print("======================================================")
    print(" Scalable Pipeline Summary")
    print("------------------------------------------------------")
    if stage1_time is not None:
        print(f"  Stage 1 runtime       : {stage1_time:.1f} seconds")
    else:
        print("  Stage 1 runtime       : (skipped)")
    print(f"  Stage 2 runtime       : {stage2_time:.1f} seconds")
    print(f"  Total pipeline runtime: {total_time:.1f} seconds")
    print(f"  New ok datasets       : {num_new}")
    print(f"  Throughput (this run) : {throughput:.1f} datasets/hour")
    print("======================================================")

    # Write summary record for later analysis
    record = {
        "timestamp": time.time(),
        "max_datasets": args.max_datasets,
        "concurrency": args.concurrency,
        "skip_stage1": args.skip_stage1,
        "stage1_runtime_seconds": stage1_time,
        "stage2_runtime_seconds": stage2_time,
        "total_runtime_seconds": total_time,
        "new_ok_datasets": num_new,
        "throughput_datasets_per_hour": throughput,
    }

    with RUNTIME_OUTPUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
