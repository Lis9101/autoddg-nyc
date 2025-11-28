import json
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
#   Load NDCG JSON
# -----------------------------
def load_ndcg(ndcg_path="ndcg_eval_results.json"):
    with open(ndcg_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
#   Load text_eval JSONL
# -----------------------------
def load_text_eval(path="text_eval_results.jsonl"):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# -----------------------------
#   Aggregate text_eval
# -----------------------------
def aggregate_text_eval(records):
    methods = ["original", "hs", "ufd", "ufd_nyc", "sfd", "sfd_nyc"]

    summary = {m: {"completeness": [], "conciseness": [], "readability": [], "faithfulness": []}
               for m in methods}

    for row in records:
        for m in methods:
            ref_free = row.get(m, {}).get("ref_free", {})
            for k in summary[m]:
                v = ref_free.get(k)
                if v is not None:
                    summary[m][k].append(v)

    # compute mean
    for m in methods:
        for k in summary[m]:
            arr = summary[m][k]
            summary[m][k] = float(np.mean(arr)) if arr else None

    return summary


# -----------------------------
#   Radar plot
# -----------------------------
def make_radar_plot(text_summary, ndcg_summary, out_path="../../outputs/eval_radar.png"):

    labels = [
        "Lexical Matching (BM25)",
        "Completeness",
        "Conciseness",
        "Readability",
        "Faithfulness"
    ]

    def metrics_normalized(method):
        """Return metrics normalized to [0,1]."""
        return [
            ndcg_summary[method]["bm25@20"],                  # Already in 0–1
            text_summary[method]["completeness"] / 10.0,      # Normalize
            text_summary[method]["conciseness"] / 10.0,
            text_summary[method]["readability"] / 10.0,
            text_summary[method]["faithfulness"] / 10.0,
        ]

    methods = ["original", "hs", "ufd", "ufd_nyc", "sfd", "sfd_nyc"]
    method_names = {
        "original": "Original",
        "hs": "H&S",
        "ufd": "UFD",
        "ufd_nyc": "UFD-NYC",
        "sfd": "SFD",
        "sfd_nyc": "SFD-NYC"
    }

    num_vars = len(labels)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    for m in methods:
        vals = metrics_normalized(m)
        vals = [0 if v is None else v for v in vals]
        vals += vals[:1]

        ax.plot(angles, vals, label=method_names[m], linewidth=2)
        ax.fill(angles, vals, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels([])

    # annotation
    plt.text(0.5, -0.15, "Scores normalized to [0, 1]", transform=ax.transAxes,
             ha="center", fontsize=12)

    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] Radar plot → {out_path}")


# -----------------------------
#   Save metrics table (normalized)
# -----------------------------
def save_metrics_table(text_summary, ndcg_summary, out_path="../../outputs/eval_results.json"):
    out = {}
    methods = ["original", "hs", "ufd", "ufd_nyc", "sfd", "sfd_nyc"]

    for m in methods:
        out[m] = {
            "lexical_matching_bm25": round(ndcg_summary[m]["bm25@20"], 4),
            "completeness": round(text_summary[m]["completeness"] / 10.0, 4),
            "conciseness": round(text_summary[m]["conciseness"] / 10.0, 4),
            "readability": round(text_summary[m]["readability"] / 10.0, 4),
            "faithfulness": round(text_summary[m]["faithfulness"] / 10.0, 4),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[SAVED] Metrics table → {out_path}")


# -----------------------------
#   Main
# -----------------------------
if __name__ == "__main__":

    ndcg = load_ndcg("ndcg_eval_results.json")
    text_records = load_text_eval("text_eval_results.jsonl")

    text_summary = aggregate_text_eval(text_records)

    make_radar_plot(text_summary, ndcg)

    save_metrics_table(text_summary, ndcg)
