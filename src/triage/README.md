# `triage/` – Sampling Strategies for LLM‑Powered Data Triage

This module contains small, composable sampling utilities that you can plug
into an LLM‑powered labeling pipeline.


> You have a **large pool** of text examples (e.g., wildlife ads, NYC Open
> Data descriptions, or benchmark datasets), and a **limited LLM budget** for
> labeling.  You want to choose *which* examples to send to the LLM so that a
> down‑stream classifier trains well.


---

## Overview

All samplers implement a tiny, common interface:

```python
import pandas as pd
from triage import BaseSampler

def sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
    ...
```

* `df` is your candidate pool as a pandas `DataFrame`.
* `n` is the desired batch size.
* The return value is a **new** `DataFrame` with the sampled rows.

Implemented strategies:

- `RandomSampler` – simple uniform baseline.
- `KeywordSampler` – prioritizes rows whose text contains a given keyword list.
- `StratifiedSampler` – label‑aware sampling for imbalanced datasets.
- `ClusterDiverseSampler` – encourages diversity across pre‑computed clusters.

You can mix and match these samplers or add your own by inheriting from
`BaseSampler`.

---

## Installation

The samplers only depend on packages that are already in the root
`requirements.txt`:

- `pandas`
- `numpy`

Once the repo environment is set up, you can import the module from the project
root:

```python
# From the project root:
# python -m pip install -r requirements.txt

from src.triage import (
    RandomSampler,
    KeywordSampler,
    StratifiedSampler,
    ClusterDiverseSampler,
)
```

If you prefer running scripts from inside `src/`, you can also rely on the
relative import:

```python
from triage import RandomSampler
```

(As long as `src/` is on `PYTHONPATH`.)

---

## 1. Random baseline – `RandomSampler`

**File:** `random_sampler.py`

Uniform random selection is a surprisingly strong baseline. It is also useful
for comparing against more complex strategies.

```python
import pandas as pd
from triage import RandomSampler

df = pd.read_csv("data/wildlife_ads.csv")

sampler = RandomSampler(random_state=42)
batch = sampler.sample(df, n=500)  # returns a DataFrame
```

* Sampling is done **without replacement**.
* If `n` is larger than `len(df)`, you simply get all rows.

---

## 2. Keyword‑biased sampling – `KeywordSampler`

**File:** `keyword_sampler.py`

Use this when you have a high‑precision keyword list (e.g., species names,
product types, domain terms) and want to bias labeling toward likely positives.

```python
from triage import KeywordSampler

sampler = KeywordSampler(
    keywords=["ivory", "tiger", "rhino", "leopard"],
    text_column="title",   # or "description"
    add_score_column=True, # attach keyword_score to the output
)

batch = sampler.sample(df, n=300)
```

Internally, the sampler:

1. Counts how many keywords appear in each row (case‑insensitive).
2. Samples primarily from rows with non‑zero counts.
3. If needed, fills remaining slots with uniform random examples.

This is a cheap way to approximate an "importance sampler" without training a
model first.

---

## 3. Label‑aware sampling – `StratifiedSampler`

**File:** `stratified_sampler.py`

If you already have *some* labels (for example, from a previous LLM pass or a
weak heuristic classifier), you can build more balanced batches by stratifying
over a label column.

```python
from triage import StratifiedSampler

sampler = StratifiedSampler(
    label_column="weak_label",  # e.g. 0/1, or a string label
    min_per_label=10,
)

batch = sampler.sample(df, n=200)
```

The strategy is two‑stage:

1. Try to allocate at least `min_per_label` examples per label (if available).
2. Use the remaining budget on a proportionally stratified sample from the
   leftover pool.

This is useful for:

- Constructing evaluation sets for imbalanced data.
- Oversampling rare classes in a second labeling round.

---

## 4. Cluster‑diverse sampling – `ClusterDiverseSampler`

**File:** `cluster_diverse_sampler.py`

This is a simple, LTS‑style *diversity* sampler: it assumes you already have a
`cluster_id` column that groups similar examples together (e.g., using k‑means
on embeddings) and tries to cover many clusters in each batch.

```python
from triage import ClusterDiverseSampler

sampler = ClusterDiverseSampler(
    cluster_column="cluster_id",
    max_per_cluster=10,
)

batch = sampler.sample(df, n=400)
```

Conceptually:

- First round: take one representative from as many clusters as possible.
- Second round: keep cycling over clusters, adding more rows up to
  `max_per_cluster` per cluster, until the budget `n` is exhausted or there are
  no more unused rows.

This gives you batches that are **diverse** across clusters, which can reduce
redundancy in LLM labeling.

---

## 5. Implementing your own sampler

To add a new strategy (e.g., full Lean‑To‑Sample with a bandit policy), create
a new file (e.g. `lts_sampler.py`) and inherit from `BaseSampler`:

```python
from dataclasses import dataclass
from triage import BaseSampler
import pandas as pd

@dataclass
class MyCoolSampler(BaseSampler):
    some_hyperparam: float = 1.0

    def sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        n = self._normalize_n(df, n)
        if n == 0:
            return df.iloc[0:0].copy()

        # TODO: implement your logic here
        # use self._rng for randomness to keep things reproducible
        return df.sample(
            n=n,
            replace=False,
            random_state=self._rng.integers(0, 1_000_000),
        )
```

Then re‑export it in `__init__.py` so it can be imported as
`from triage import MyCoolSampler`.

---

## 6. How to use

1. Load the candidate pool into a `DataFrame`.
2. Run one or more samplers to select a batch.
3. Send that batch to an LLM for labeling.
4. Train / update a classifier on the labeled subset.
5. Log which sampler and configuration produced which batch so you can compare
   performance across strategies.

Because the module is independent of any particular LLM or dataset, you can use
it for:

- Wildlife trafficking ads.
- NYC Open Data tables.
- Standard text benchmarks (Reuters, WebKB, 20 Newsgroups, Emotions, ...).
