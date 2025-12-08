# What's in this folder

We implemented a scalable pipeline with two stages to improve performance and support large-scale dataset processing:

- **Stage 1: Profiling with Spark**
  Runs data profiling in parallel across datasets using PySpark, producing structured summaries efficiently.

- **Stage 2: Asynchronous LLM Descriptions**
  Generates dataset descriptions outside Spark using an asynchronous Python service with bounded concurrency. Multiple API calls are issued concurrently while respecting rate limits and cost constraints.

This folder contains three scripts:

1. `stage1_spark_profiling.py`
2. `stage2_async_nyc_descriptions.py`
3. `run_scalable_pipeline.py` — the driver that runs both stages

---

## Observations

We compare **average runtime per dataset** and **throughput (datasets/hour)** between the original pipeline and the scalable version.

For the first 100 NYC Open Data datasets, the original pipeline averages **~20.38 seconds per dataset**, corresponding to **~177 datasets/hour** (fully sequential).

The scalable pipeline shows major improvements. Average runtime per dataset decreases and throughput increases as concurrency and batch size grow:

| Concurrency | New OK | Total Runtime (s) | Avg Runtime per Dataset (s) | Throughput (datasets/hr) |
| ----------- | ------ | ----------------: | --------------------------: | -----------------------: |
| 5           | 50     |            264.06 |                    **5.28** |                **682.7** |
| 5           | 100    |            481.24 |                    **4.81** |                **748.1** |
| 10          | 50     |            160.92 |                    **3.22** |               **1118.6** |
| 10          | 500    |           1219.27 |                    **2.44** |               **1476.3** |
| 20          | 500    |           1005.85 |                    **2.01** |               **1789.5** |

**Key takeaway:**
The scalable pipeline is much faster than the original. It increases throughput from ~177 datasets/hour to almost 1800 datasets/hour, and it reduces the average time per dataset from ~20 seconds to about 2 seconds when concurrency is high. This demonstrates that the redesigned pipeline scales efficiently as workload and parallelism increase.
