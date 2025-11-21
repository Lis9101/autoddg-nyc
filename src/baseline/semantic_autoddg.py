"""
File: semantic_autoddg.py
Description:
    Semantic profiling (LLM-based) for Baseline AutoDDG.
    Includes:
        - semantic_profile_column()
        - build_semantic_profile()
    Quota-aware:
        - Only profiles a subset of columns per dataset (max_semantic_columns).
        - Topic is provided by the caller (no extra LLM calls).
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from baseline.llm_client import call_llm


SEMANTIC_COLUMN_PROMPT = """You are assisting with semantic profiling of table columns.

Column name: {col_name}
Inferred coarse type: {coarse_type}
Example values (strings): {examples}

Return a STRICT JSON object with the following fields:
- is_temporal
- temporal_resolution
- is_spatial
- spatial_resolution
- entity_type
- domain_specific_type
- function_or_usage

Only output valid JSON, no extra text or explanation.
"""


def sample_rows(csv_path: Path, n: int = 5) -> str:
    df = pd.read_csv(csv_path, nrows=n)
    return df.to_csv(index=False)


def semantic_profile_column(name: str, coarse_type: str, examples: List[str]) -> Dict[str, Any]:
    prompt = SEMANTIC_COLUMN_PROMPT.format(
        col_name=name,
        coarse_type=coarse_type,
        examples=", ".join(examples[:10]),
    )
    raw = call_llm(prompt, temperature=0.0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: record raw text, mark as unknown
        return {
            "_raw": raw,
            "is_temporal": False,
            "temporal_resolution": "unknown",
            "is_spatial": False,
            "spatial_resolution": "unknown",
            "entity_type": "unknown",
            "domain_specific_type": "general",
            "function_or_usage": "unknown",
        }


def _select_columns_for_semantics(content_profile: Dict[str, Any],
                                  max_semantic_columns: int) -> List[str]:
    """
    Heuristic:
      - ignore very sparse columns (null_fraction >= 0.8)
      - ignore ID-like columns (num_unique / num_rows > 0.9)
      - among remaining, pick up to max_semantic_columns with lowest null_fraction
    Returns list of column names to send to LLM.
    """
    num_rows = max(1, content_profile.get("num_rows", 1))

    candidates = []
    for col in content_profile["columns"]:
        null_frac = col.get("null_fraction", 0.0)
        nunique = col.get("num_unique", 0)
        uniqueness_ratio = nunique / num_rows

        too_sparse = null_frac >= 0.8
        is_id_like = uniqueness_ratio > 0.9

        if not too_sparse and not is_id_like:
            candidates.append(col)

    # Sort so the densest columns (lowest null_fraction) are first
    candidates.sort(key=lambda c: c.get("null_fraction", 0.0))

    selected = candidates[:max_semantic_columns]
    return [c["name"] for c in selected]


def build_semantic_profile(
    csv_path: Path,
    content_profile: Dict[str, Any],
    title: str,
    description: str,
    topic: str,
    max_semantic_columns: int = 5,
) -> Dict[str, Any]:
    """
    Build semantic profile for a dataset.

    Quota tricks:
      - Topic is passed in (no LLM call).
      - Only a subset of columns are semantically profiled with LLM.
      - Non-profiled columns get a lightweight "semantic" stub.
    """
    selected_names = set(
        _select_columns_for_semantics(content_profile, max_semantic_columns)
    )

    semantic_columns = []
    for col in content_profile["columns"]:
        name = col["name"]
        coarse_type = col["coarse_type"]
        examples = col.get("example_values", [])

        if name in selected_names:
            sem = semantic_profile_column(name, coarse_type, examples)
            semantic_columns.append({
                "name": name,
                "coarse_type": coarse_type,
                "example_values": examples,
                "semantic": sem,
                "semantic_profiled": True,
            })
        else:
            # Skipped to save quota; still include basic info.
            semantic_columns.append({
                "name": name,
                "coarse_type": coarse_type,
                "example_values": examples,
                "semantic": {
                    "is_temporal": False,
                    "temporal_resolution": "unknown",
                    "is_spatial": False,
                    "spatial_resolution": "unknown",
                    "entity_type": "unknown",
                    "domain_specific_type": "general",
                    "function_or_usage": "unknown",
                    "skipped_for_quota": True,
                },
                "semantic_profiled": False,
            })

    return {
        "dataset_id": content_profile["dataset_id"],
        "title": title,
        "description": description,
        "topic": topic,
        "columns": semantic_columns,
    }
