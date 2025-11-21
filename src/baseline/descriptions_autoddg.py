"""
File: descriptions_autoddg.py
Description:
    Generates UFD and SFD for Baseline AutoDDG using LLM.
"""

import json
from pathlib import Path

import pandas as pd

from baseline.llm_client import call_llm


# --- sampling settings ---

MAX_SAMPLE_ROWS = 5          # how many rows to show
MAX_SAMPLE_COLS = 10         # max number of columns to include in the sample
MAX_COL_CELL_CHARS = 2000    # if a column has cells longer than this, we drop it from the sample


def sample_rows(csv_path: Path, n: int = MAX_SAMPLE_ROWS) -> str:
    """
    Read a small sample of the CSV and only include columns whose
    cell contents are not "too big" in the first n rows.

    If all columns are too large, we return a small note instead of
    a giant sample.
    """
    df = pd.read_csv(csv_path, nrows=n)

    if df.empty:
        return ""

    # Work with strings to measure lengths
    as_str = df.astype(str)

    # For each column, look at the max cell length in the sample
    col_max_len = as_str.apply(lambda col: col.str.len().max())

    # Keep only "safe" columns whose max cell length is below the threshold
    safe_cols = [col for col in df.columns if col_max_len[col] <= MAX_COL_CELL_CHARS]

    if not safe_cols:
        # All columns are huge; don't pass raw data into the prompt
        return (
            "# Sample omitted: all columns have very large cell values "
            f"in the first {n} rows.\n"
        )

    # Optionally cap the number of columns we include
    if len(safe_cols) > MAX_SAMPLE_COLS:
        safe_cols = safe_cols[:MAX_SAMPLE_COLS]

    df_safe = df[safe_cols]

    return df_safe.to_csv(index=False)



UFD_PROMPT = """You are generating a USER-FOCUSED description for a tabular dataset.

Dataset topic: {topic}

You are given:
- A small CSV sample:
{sample}

- A data-driven content profile (JSON):
{content_profile}

- A semantic profile (JSON):
{semantic_profile}

Write 5–10 natural, readable sentences describing what the dataset is about,
what key columns represent, and how someone might use it.
"""


SFD_PROMPT = """You are generating a SEARCH-FOCUSED description for a tabular dataset.

Dataset topic: {topic}

User-focused description:
{ufd}

Create the following sections in plain text:

Dataset Overview:
  2–4 sentences summarizing the dataset.

Related Topics:
  - bullet list of related domains and themes.

Concepts and Synonyms:
  - bullet list of important terms, phrases, and synonyms.

Applications and Use Cases:
  - bullet list of plausible ways to use this dataset.
"""


def generate_ufd(
    csv_path: Path,
    content_profile: dict,
    semantic_profile: dict,
    topic: str,
) -> str:
    prompt = UFD_PROMPT.format(
        topic=topic,
        sample=sample_rows(csv_path),
        content_profile=json.dumps(content_profile, ensure_ascii=False),
        semantic_profile=json.dumps(semantic_profile, ensure_ascii=False),
    )
    return call_llm(prompt, temperature=0.3)


def generate_sfd(topic: str, ufd: str) -> str:
    prompt = SFD_PROMPT.format(topic=topic, ufd=ufd)
    return call_llm(prompt, temperature=0.3)
