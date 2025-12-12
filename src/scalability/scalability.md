# What's in this folder

We implemented a scalable pipeline with two stages to improve performance and support large-scale dataset processing:

- **Stage 1: Profiling with Spark**
  Stage 1 loads each dataset as a PySpark DataFrame and uses DataFrame API operations (count, columns) to compute minimal metadata in parallel. The output (`stage1_spark_profiles.jsonl`) provides structured, minimal profiles that feed into Stage 2.

- **Stage 2: Asynchronous LLM Descriptions**
  Generates dataset descriptions outside Spark using an asynchronous Python service with bounded concurrency. Multiple API calls are issued concurrently while respecting rate limits and cost constraints. Outputs are written to a large file `stage2_raw.jsonl`, which is then post-processed by `create_stage2_slim.py` to keep only the dataset ID, title, and generated descriptions.

This folder contains:

1. `run_scalable_pipeline.py` - the driver that runs both stages
2. `stage1_spark_profiling.py` - spark code and logic
3. `stage2_async_nyc_descriptions.py` - asynchronous LLM calls code and logic
4. `create_stage2_slim.py` - convert the large output from stage2 to a smaller size that can be uploaded to github

---

## Observations

We compare **average runtime per dataset** and **throughput (datasets/hour)** between the original pipeline (`output/baseline_autoddg_runtime.jsonl`) and the scalable version (`output/scalable_pipeline_runtime.jsonl`).

For the first 100 NYC Open Data datasets, the original pipeline averages **~20.38 seconds per dataset**, corresponding to **~177 datasets/hour** (fully sequential).

The scalable pipeline shows major improvements. Average runtime per dataset decreases and throughput increases as concurrency and batch size grow:

| Concurrency | # Processed Datasets| Total Runtime (s) | Avg Runtime per Dataset (s) | Throughput (datasets/hr) |
| ----------- | ------ | ----------------: | --------------------------: | -----------------------: |
| 5           | 50     |            264.06 |                    **5.28** |                **682.7** |
| 5           | 100    |            481.24 |                    **4.81** |                **748.1** |
| 5           | 497    |           2301.01 |                    **4.63** |                **777.57** |
| 10          | 50     |            160.92 |                    **3.22** |               **1118.6** |
| 10          | 500    |           1219.27 |                    **2.44** |               **1476.3** |
| 20          | 50     |            170.43 |                    **3.40** |               **1035.0** |
| 20          | 500    |           1005.85 |                    **2.01** |               **1789.5** |

**Key takeaway:**
The scalable pipeline is much faster than the original. It increases throughput from ~177 datasets/hour to almost 1800 datasets/hour, and it reduces the average time per dataset from ~20 seconds to about 2 seconds when concurrency is high. This demonstrates that the redesigned pipeline scales efficiently as workload and parallelism increase.
