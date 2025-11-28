# Contents

## Week 1

- `data_collector.py` - Bulk downloader. Collects NYC Open Data datasets (tabular only), saves
  CSV samples, and builds the metadata registry.
- `download_from_registry.py` - Team sync tool. Restores missing CSV files using the shared
  `metadata_registry.json`.
- `pipeline_test.py`- End-to-end test to verify Socrata + Gemini APIs.
- `check_models.py`- Lists Gemini models available to your API key.
- `metadata_registry.json`: Shared dataset registry created in Week 1.

## Week 2

Baseline AutoDDG implementation.

- `baseline/profiling_autoddg.py` - Generates Content Summaries (non-LLM): column types, null ratios, unique counts, stats, sample values.
- `baseline/semantic_autoddg.py` - Generates Semantic Summaries (LLM): temporal/spatial detection, entity types, usage roles. Includes column filtering to reduce Gemini quota usage.
- `baseline/descriptions_autoddg.py` - Produces UFD (User-Focused Description) and SFD (Search-Focused Description).
- `baseline/llm_client.py` - Central Gemini client with the gemini-2.0-flash model.
- `baseline_autoddg.py` - Main Baseline AutoDDG runner. Loads each dataset, builds summaries, generates UFD+SFD, and writes results to:
  - `data/0_baseline_autoddg_descriptions.jsonl`
  - `data/0_baseline_autoddg_runtime.jsonl`

Supports resume (skips completed datasets) and stops on quota errors.

### Run baseline AutoDDG

```
python src/baseline_autoddg.py
```
