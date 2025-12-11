# Contents

The `/baseline` folder contains the code for building the baseline AutoDDG. `descriptions_nyc.py` in this folder contains the the new nyc-specific prompts for the llm.
The `/evaluation` folder contains the code for evaluation.
The `/scalability` folder contains the code for building the scalability pipeline.

All generated results are saved to outputs folder. Here're the important files

- `metadata_registry` - contains metadata for all 2000 datasets, including original description
- `0_baseline_autoddg_descrioptions.jsonl` - contains generated dataset description using the baseline AutoDDG (ufd and sfd)
- `stage_2_async_nyc_descriptions.jsonl` - contains generated dataset description using the AutoDDG with nyc-specific prompts (ufd_nyc and sfd_nyc)
- `eval_radar.png` - contains the radar graph for final evaluation results.

### AutoDDG fast

```
python src/run_scalable_pipeline.py
```

### Run AutoDDG (takes a long time)

```
python src/baseline_autoddg.py
```
