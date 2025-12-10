"""triage
=================

Sampling utilities for *data triage* experiments.

This package provides small, composable samplers that you can plug into an
LLM-powered labeling pipeline.  Each sampler takes a pandas ``DataFrame`` and
returns a sampled ``DataFrame`` with the same columns.

The design is intentionally lightweight:

* No direct LLM dependencies
* Works on any tabular / text dataset (wildlife ads, NYC Open Data, etc.)
* Easy to extend with custom strategies

The main entrypoints are:

* :class:`triage.random_sampler.RandomSampler`
* :class:`triage.keyword_sampler.KeywordSampler`
* :class:`triage.stratified_sampler.StratifiedSampler`
* :class:`triage.cluster_diverse_sampler.ClusterDiverseSampler`

Example
-------

.. code-block:: python

    import pandas as pd
    from triage import RandomSampler, KeywordSampler

    df = pd.read_csv("data/wildlife_ads.csv")

    # 1) Pure random sampling
    sampler = RandomSampler(random_state=42)
    batch = sampler.sample(df, n=200)

    # 2) Keyword-biased sampling over the same pool
    kw_sampler = KeywordSampler(
        keywords=["ivory", "tiger", "rhino", "leopard"],
        text_column="title",  # or "description"
        random_state=42,
    )
    keyword_batch = kw_sampler.sample(df, n=200)

Both ``batch`` and ``keyword_batch`` can then be sent to an LLM for labeling.

You can combine these strategies manually (run one after another and
concatenate) or define your own sampler by inheriting from
:class:`triage.base.BaseSampler`.

"""

from .base import BaseSampler
from .random_sampler import RandomSampler
from .keyword_sampler import KeywordSampler
from .stratified_sampler import StratifiedSampler
from .cluster_diverse_sampler import ClusterDiverseSampler

__all__ = [
    "BaseSampler",
    "RandomSampler",
    "KeywordSampler",
    "StratifiedSampler",
    "ClusterDiverseSampler",
]
