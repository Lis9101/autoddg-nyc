# Baseline AutoDDG + NYC specific prompting 

This directory contains the baseline implementation of the AutoDDG pipeline.
These components provide dataset profiling, semantic enrichment, and description generation using a single-dataset, sequential workflow.


#### **profiling_autoddg.py**

Computes column-level statistics and content summaries using Pandas.
Used to generate the structured input for semantic profiling and description generation.

#### **semantic_autoddg.py**

Builds semantic profiles for each column by calling the LLM (via `llm_client`).
Extracts properties such as temporal/spatial flags, entity type, and domain signals.

#### **descriptions_autoddg.py**

Generates baseline dataset descriptions (UFD/SFD) using generic prompts without NYC-specific context.

#### **descriptions_nyc.py**

Extends the baseline description generator with NYC-specific prompting (UFD-NYC / SFD-NYC), incorporating local entities, geographic context, and domain relevance.

#### **llm_client.py**

A thin wrapper around the LLM API (Gemini).
Handles request formatting, parsing, and basic error handling.