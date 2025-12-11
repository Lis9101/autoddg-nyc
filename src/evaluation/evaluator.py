"""
File: src/evaluation/evaluator.py
Description:
    Generates a Radar Plot comparing methods.
    Fixed: Moved 'angles' definition before plotting loop to fix UnboundLocalError.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path

# ================================================================
# Path Configuration
# ================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

NDCG_PATH = OUTPUTS_DIR / "ndcg_eval_results.json"
TEXT_EVAL_PATH = OUTPUTS_DIR / "text_eval_results.json"

OUTPUT_PLOT_PATH = OUTPUTS_DIR / "eval_radar.png"
OUTPUT_TABLE_PATH = OUTPUTS_DIR / "eval_results.json"


# ================================================================
# Loaders
# ================================================================
def load_json(path):
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ================================================================
# Plotting Logic
# ================================================================
def make_radar_chart(ndcg_data, text_data):
    print("-" * 60)
    print(" Generating Radar Plot...")
    print("-" * 60)

    labels = [
        "Search (NDCG)", 
        "Readability", 
        "Faithfulness", 
        "Completeness", 
        "Conciseness"
    ]
    N = len(labels)

    # [FIX] Calculate angles HERE (Before the plotting loop)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Define methods to plot
    methods_to_plot = {
        "original": {"label": "Original",   "color": "blue",   "style": "dotted"},
        "ufd":      {"label": "UFD",        "color": "green",  "style": "solid"},
        "sfd":      {"label": "SFD",        "color": "purple", "style": "dashdot"},
        "ufd_nyc":  {"label": "UFD-NYC",    "color": "red",    "style": "solid"},
        "sfd_nyc":  {"label": "SFD-NYC",    "color": "brown",  "style": "dashdot"},
    }

    # Helper to handle "hs" vs "HandS" mismatch
    def get_text_metric(method_key, metric):
        if method_key in text_data:
            return text_data[method_key].get(metric, 0)
        if method_key == "hs" and "HandS" in text_data:
            return text_data["HandS"].get(metric, 0)
        return 0

    # Prepare data
    plot_data = {}

    for m in methods_to_plot.keys():
        # 1. Get Search Score
        ndcg_val = ndcg_data.get(m, {}).get("bm25@10", 0)
        if ndcg_val == 0:
             ndcg_val = ndcg_data.get(m, {}).get("bm25@20", 0)

        # 2. Get Text Scores
        readability  = get_text_metric(m, "readability") / 5.0
        faithfulness = get_text_metric(m, "faithfulness") / 5.0
        completeness = get_text_metric(m, "completeness") / 5.0
        conciseness  = get_text_metric(m, "conciseness") / 5.0

        if ndcg_val == 0 and readability == 0:
            # Skip if absolutely no data found
            continue

        values = [ndcg_val, readability, faithfulness, completeness, conciseness]
        values += values[:1] # Close the loop
        
        plot_data[m] = values

    # Draw Plot
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for m, values in plot_data.items():
        config = methods_to_plot.get(m, {})
        
        # Highlight UFD-NYC
        lw = 3 if m == "ufd_nyc" else 2
        alpha = 0.15 if m == "ufd_nyc" else 0.05
        
        ax.plot(angles, values, linewidth=lw, linestyle=config["style"], 
                label=config["label"], color=config["color"])
        ax.fill(angles, values, config["color"], alpha=alpha)

    # Setup Axes
    plt.xticks(angles[:-1], labels, size=11)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Method Comparison: AutoDDG-NYC vs Baselines', size=15, y=1.1)

    # Save
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f" [SAVED] Radar plot -> {OUTPUT_PLOT_PATH}")

    # Save Summary Table
    with open(OUTPUT_TABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(plot_data, f, indent=2)


def main():
    if not NDCG_PATH.exists() or not TEXT_EVAL_PATH.exists():
        print("[ERROR] Input files missing in 'outputs' folder.")
        return

    ndcg_data = load_json(NDCG_PATH)
    text_data = load_json(TEXT_EVAL_PATH)

    make_radar_chart(ndcg_data, text_data)
    print("-" * 60)
    print(" Done!")
    print("-" * 60)

if __name__ == "__main__":
    main()