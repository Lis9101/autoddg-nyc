# Contents
> The repo contains some legacy files after merging branch, this document will talk about what are the important files and how to run to reproduce same result.

All scripts are included in the `src/` folder, all generated output are saved to the `output/` folder, the `data/` folder contains the result from week 1 & 2 (baseline AutoDDG generation).

The `/baseline` folder contains the code for building the baseline AutoDDG. `descriptions_nyc.py` in this folder contains the the new nyc-specific prompts for the LLM.  Read [whatshere.md](src\baseline\whatshere.md) in this folder for details.


The `/evaluation` folder contains the code for evaluation. Read [evaluation.txt](src\evaluation\evaluation.txt) in this folder for details.

The `/scalability` folder contains the code for building the scalability pipeline. Read [scalability.md](src\scalability\scalability.md) in this folder for more details.

---

Here're the important files in the `outputs` folder:
- `metadata_registry.json` - contains metadata for all 2000 datasets, including original descriptions
- `0_baseline_autoddg_descrioptions.jsonl` - contains generated descriptions for 2000 datasets using the baseline AutoDDG (ufd and sfd). This is generated from the baseline_autoddg branch. 
- `stage_2_async_nyc_descriptions.jsonl` - contains generated descriptions for 2000 datasets using the AutoDDG with nyc-specific prompts (ufd_nyc and sfd_nyc). This is generated from the scalable pipeline (scalability branch).
- `eval_radar.png` - contains the radar graph for final evaluation results. The current graph is generated based on the three json/jsonl files listed above. 
  To reevaluate, run `python evaluator.py` in the evaluation folder.

### Dataset downloads
Currently we are using the Top 2000 datasets from NYC Open Data for experimentation. 
To re-download the same dataset, run 
```
python src/download_from_registry.py
```
Make sure to have `metadata_registry.json` in output folder and Socrata API key set up in your local environment.

### Run AutoDDG (takes a long time)
This will take in all dataset csv files in the local path and generate description with the baseline  and NYC-specific prompting. 
```
python src/baseline_autoddg.py
```
This will take a long time and generate a huge file. Output is stored to [outputs/baseline_autoddg_descriptions.jsonl](outputs/baseline_autoddg_descriptions.jsonl).
Current version of this file on repo is only the head of the large file.

### Run AutoDDG-NYC fast

```
python src/scalability/run_scalable_pipeline.py \
        --max_datasets 2000 \
        --concurrency 20
```
Please keep in mind that this only genereate descriptions with the NYC-specific prompts, we didn't include the baseline version in the scalable scripts (we just reused results from previous branch). The logic would be the same, only change is switch off the use of `descriptions_nyc.py` to `descriptions_autoddg.py` in `stage2_async_nyc_descriptions.py`.