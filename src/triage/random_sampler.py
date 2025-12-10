"""Uniform random sampler for data triage.

This is the simplest reasonable baseline: pick examples uniformly at random
from the pool.  In many LLM‑triage settings this is surprisingly strong and
provides a good reference point for more sophisticated strategies.

Usage
-----

.. code-block:: python

    from triage import RandomSampler
    import pandas as pd

    df = pd.read_csv("data/wildlife_ads.csv")
    sampler = RandomSampler(random_state=42)
    batch = sampler.sample(df, n=500)

"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import BaseSampler


@dataclass
class RandomSampler(BaseSampler):
    """Uniform random sampler.

    This sampler ignores all columns and simply draws ``n`` rows uniformly
    at random (without replacement) from the input ``DataFrame``.

    Notes
    -----
    * If ``n`` is larger than the number of rows, all rows are returned.
    * Sampling is done **without replacement** so you never see duplicates
      within a single batch.
    """

    def sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        n = self._normalize_n(df, n)
        if n == 0:
            # Return an empty DataFrame with the same columns/index type
            return df.iloc[0:0].copy()

        return df.sample(n=n, replace=False, random_state=self._rng.integers(0, 1_000_000))
