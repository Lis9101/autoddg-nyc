"""
File: semantic_autoddg.py  (UPDATED: merged LLM call version)
Description:
    Semantic profiling (LLM-based) for Baseline AutoDDG, modified to use **ONE**
    LLM call per dataset instead of per-column.

    Includes:
        - sample_rows()
        - build_semantic_profile()  (now uses merged prompt)
        - _parse_multi_column_output()  (NEW)
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import re

from baseline.llm_client import call_llm


# -------------------------------------------------------------------
# New merged-column semantic prompt template
# -------------------------------------------------------------------
MERGED_SEMANTIC_PROMPT = """
You are performing semantic profiling for MULTIPLE columns of a table.

For EACH column, extract:

- is_temporal (true/false)
- temporal_resolution (e.g., second/minute/hour/day/month/year/unknown)
- is_spatial (true/false)
- spatial_resolution (e.g., point/region/address/unknown)
- entity_type (e.g., person, location, event, measurement, id, etc.)
- domain_specific_type (e.g., climate, transportation, finance, general, etc.)
- function_or_usage (e.g., identifier, measurement, descriptor, category, etc.)

Return STRICT JSON of the following structure:

{{
  "columns": [
    {{
      "name": "...",
      "semantic": {{
         "is_temporal": ...,
         "temporal_resolution": "...",
         "is_spatial": ...,
         "spatial_resolution": "...",
         "entity_type": "...",
         "domain_specific_type": "...",
         "function_or_usage": "..."
      }}
    }},
    ...
  ]
}}

Now analyze the following columns:

{column_blocks}

Output ONLY the JSON. No explanation.
"""


def sample_rows(csv_path: Path, n: int = 5) -> str:
    df = pd.read_csv(csv_path, nrows=n)
    return df.to_csv(index=False)


# -------------------------------------------------------------------
# Helper: parse multi-column JSON output from GPT
# -------------------------------------------------------------------

def _parse_multi_column_output(raw: str) -> Dict[str, Dict[str, Any]]:
    """
    Robust JSON extractor:
      - Removes surrounding text
      - Extracts the first {...} block
      - Returns {column_name: semantic_dict}
    """
    if not raw:
        return {}

    # ---- 1) Extract the JSON object between the FIRST '{' and LAST '}' ----
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    json_str = raw[start:end+1].strip()

    # ---- 2) Try to parse ----
    try:
        data = json.loads(json_str)
    except Exception:
        # If broken, try removing trailing commas, common GPT issue
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*\]", "]", json_str)
        try:
            data = json.loads(json_str)
        except Exception:
            print("[WARN] Failed to parse LLM JSON:", raw[:200], "...")
            return {}

    # ---- 3) Convert to {name: semantic_dict} format ----
    result = {}
    for col in data.get("columns", []):
        name = col.get("name")
        sem = col.get("semantic", {})
        if name:
            result[name] = sem

    return result



# -------------------------------------------------------------------
# Build merged prompt for all selected columns
# -------------------------------------------------------------------
def _build_merged_prompt(columns: List[Dict[str, Any]]) -> str:
    blocks = []
    for col in columns:
        name = col["name"]
        coarse = col["coarse_type"]
        examples = col.get("example_values", [])

        # truncate huge examples
        ex_list = []
        for e in examples[:10]:
            s = str(e)
            if len(s) > 300:
                s = s[:300] + "...[truncated]"
            ex_list.append(s)

        block = f"""
Column name: {name}
Coarse type: {coarse}
Example values: {ex_list}
"""
        blocks.append(block)

    column_blocks = "\n".join(blocks)
    return MERGED_SEMANTIC_PROMPT.format(column_blocks=column_blocks)


# -------------------------------------------------------------------
# Main function: merged semantic profiling (ONE LLM CALL)
# -------------------------------------------------------------------
def build_semantic_profile(
    csv_path: Path,
    content_profile: Dict[str, Any],
    title: str,
    description: str,
    topic: str,
    max_semantic_columns: int = 5,
) -> Dict[str, Any]:

    # Step 1: select columns (same logic as before)
    num_rows = max(1, content_profile.get("num_rows", 1))

    # simple reuse: choose first `max_semantic_columns` non-ID-like columns
    candidates = []
    for col in content_profile["columns"]:
        null_frac = col.get("null_fraction", 0.0)
        nunique = col.get("num_unique", 0)
        uniqueness_ratio = nunique / num_rows

        too_sparse = null_frac >= 0.8
        is_id_like = uniqueness_ratio > 0.9

        if not too_sparse and not is_id_like:
            candidates.append(col)

    selected = candidates[:max_semantic_columns]

    # Step 2: Build merged prompt
    prompt = _build_merged_prompt(selected)

    # Step 3: Call LLM ONCE
    raw = call_llm(prompt, temperature=0.0)
    parsed = _parse_multi_column_output(raw)

    # Step 4: Construct output columns (preserve structure)
    semantic_columns = []
    for col in content_profile["columns"]:
        name = col["name"]
        examples = col.get("example_values", [])
        coarse = col["coarse_type"]

        if name in parsed:
            sem = parsed[name]
            profiled = True
        else:
            # fallback stub
            sem = {
                "is_temporal": False,
                "temporal_resolution": "unknown",
                "is_spatial": False,
                "spatial_resolution": "unknown",
                "entity_type": "unknown",
                "domain_specific_type": "general",
                "function_or_usage": "unknown",
                "skipped_for_quota": True,
            }
            profiled = False

        semantic_columns.append({
            "name": name,
            "coarse_type": coarse,
            "example_values": examples,
            "semantic": sem,
            "semantic_profiled": profiled,
        })

    return {
        "dataset_id": content_profile["dataset_id"],
        "title": title,
        "description": description,
        "topic": topic,
        "columns": semantic_columns,
    }
