"""
File: descriptions_autoddg.py
Description:
    Generates UFD and SFD for Baseline AutoDDG using LLM.
"""

import json
from pathlib import Path

import pandas as pd

from baseline.llm_client import call_llm


def sample_rows(csv_path: Path, n: int = 5) -> str:
    df = pd.read_csv(csv_path, nrows=n)
    return df.to_csv(index=False)


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
