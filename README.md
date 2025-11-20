# AutoDDG-NYC: Generating Scalable Descriptions for NYC Open Data

## Team: Jet2Holiday
* Shuhua Li
* Yang Han
* Yueli He
* Yang Zheng

## Project Overview
This project addresses the "discoverability crisis" in open data portals. We aim to adapt the AutoDDG framework to the NYC Open Data domain, using LLMs to automatically generate clear and consistent descriptions for tabular datasets.

## Methodology
1. **Data Collection:** Fetching datasets via Socrata API.
2. **Data Profiling:** Using PySpark for scalable content summarization.
3. **LLM Generation:** Generating semantic descriptions with OpenAI API.
4. **Evaluation:** Measuring quality via ROUGE scores and human evaluation.

## Setup
1. Clone the repository.
2. Create a `.env` file with your `SOCRATA_APP_TOKEN` and `OPENAI_API_KEY`.
3. Install dependencies via `pip install -r requirements.txt`.