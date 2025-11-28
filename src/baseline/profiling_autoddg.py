"""
File: profiling_autoddg.py
Description:
    Data-driven (non-LLM) profiling used by Baseline AutoDDG.

    Includes:
        - infer_column_profile()
        - profile_dataset()
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any

def infer_column_profile(series: pd.Series, max_examples: int = 5) -> Dict[str, Any]:
    col = series.dropna()

    if pd.api.types.is_numeric_dtype(col):
        coarse_type = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(col):
        coarse_type = "datetime"
    elif pd.api.types.is_bool_dtype(col):
        coarse_type = "boolean"
    else:
        coarse_type = "text"

    stats = {}
    if coarse_type == "numeric":
        try:
            desc = col.describe(percentiles=[0.25, 0.5, 0.75])
            stats = {
                "min": float(desc.get("min", float("nan"))),
                "max": float(desc.get("max", float("nan"))),
                "mean": float(desc.get("mean", float("nan"))),
                "std": float(desc.get("std", 0)),
                "p25": float(desc.get("25%", float("nan"))),
                "p50": float(desc.get("50%", float("nan"))),
                "p75": float(desc.get("75%", float("nan"))),
            }
        except Exception:
            stats = {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "p25": None,
                "p50": None,
                "p75": None,
            }

    elif coarse_type == "datetime":
        stats = {"min": str(col.min()), "max": str(col.max())}

    else:
        vc = col.value_counts().head(10)
        stats = {"top_values": [{"value": str(v), "count": int(c)} for v, c in vc.items()]}

    return {
        "name": series.name,
        "inferred_dtype": str(series.dtype),
        "coarse_type": coarse_type,
        "num_non_null": int(col.shape[0]),
        "num_unique": int(col.nunique()),
        "null_fraction": float(series.isna().mean()),
        "stats": stats,
        "example_values": [str(v) for v in col.head(max_examples)],
    }

def profile_dataset(csv_path: Path, dataset_id: str, max_rows=None) -> Dict[str, Any]:
    df = pd.read_csv(csv_path, nrows=max_rows)
    return {
        "dataset_id": dataset_id,
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": [infer_column_profile(df[c]) for c in df.columns],
    }
