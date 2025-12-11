import json
from pathlib import Path
import sys
# ----- Path setup -----
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]        # .../src
ROOT_DIR = THIS_FILE.parents[2]       # .../autoddg-nyc

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_PATH = ROOT_DIR / "outputs" / "stage2_async_nyc_descriptions.jsonl"
INPUT_PATH = ROOT_DIR / "outputs" / "stage2_raw.jsonl"

INPUT = Path(INPUT_PATH)
OUTPUT = Path(OUTPUT_PATH)

with open(INPUT, "r", encoding="utf-8") as f_in, \
     open(OUTPUT, "w", encoding="utf-8") as f_out:

    for line in f_in:
        line = line.strip()
        if not line:
            continue

        try:
            rec = json.loads(line)
        except Exception as e:
            print("Skipping broken line:", e)
            continue

        slim = {
            "dataset_id": rec.get("dataset_id"),
            "title": rec.get("title"),
            "ufd_nyc": rec.get("ufd_nyc", ""),
            "sfd_nyc": rec.get("sfd_nyc", "")
        }

        f_out.write(json.dumps(slim, ensure_ascii=False) + "\n")

print(f"Created slim Stage 2 file at {OUTPUT}")
