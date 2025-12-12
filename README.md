# AutoDDG-NYC: Generating Scalable Descriptions for NYC Open Data

## Team: Jet2Holiday
* Shuhua Li
* Yang Han
* Yueli He
* Yang Zheng

## Project Overview
This project addresses the "discoverability crisis" in open data portals. We aim to adapt the AutoDDG framework to the NYC Open Data domain, using LLMs to automatically generate clear and consistent descriptions for tabular datasets.


## Methodology

1. **Data Collection:** Retrieve dataset metadata and CSVs from the Socrata API.
2. **Data Profiling:** Use PySpark DataFrame API for scalable schema + content profiling across thousands of datasets.
3. **LLM Description Generation:** Generate semantic descriptions with Gemini 2.5 Flash, using an asynchronous pipeline with bounded concurrency.
4. **Evaluation:** Compare original, baseline, and NYC-specific descriptions using metrics such as NDCG.

## Setup

1. Clone this repository.
2. Create a `.env` file:

   ```
   SOCRATA_APP_TOKEN=...
   GEMINI_API_KEY=...
   ```
3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

The project uses **Gemini 2.5 Flash** for all LLM-based generation.

## Running the Pipeline

### Download datasets

```
python src/download_from_registry.py
```

### Run the AutoDDG (slow)

Generates UFD/SFD and UFD-NYC/SFD-NYC descriptions for all datasets.

```
python src/baseline_autoddg.py
```

### Run the scalable AutoDDG-NYC (fast)

Runs only with NYC-specific prompting:

```
python src/scalability/run_scalable_pipeline.py \
        --max_datasets 2000 \
        --concurrency 20
```

### Run evaluation

```
python src/evaluation/evaluator.py
```

For details on individual folders, see [**Contents.md**](contents.md).