"""Base classes for data triage samplers.

The idea of *data triage* is to select a small, informative subset of a much
larger pool of unlabeled (or weakly labeled) examples that you will send to an
LLM for labeling.  Once labeled, you can train a cheaper classifier on this
subset.

This module defines :class:`BaseSampler`, a tiny interface that all concrete
samplers implement.  The interface is intentionally simple:

* Input:  a :class:`pandas.DataFrame` (your candidate pool)
* Output: a new :class:`pandas.DataFrame` containing only the sampled rows

Design goals
------------

* **Small surface area.**  One method, :meth:`BaseSampler.sample`.
* **Pure sampling.**  No in-place mutation of the input ``DataFrame``.
* **Reproducible.**  Shared ``random_state`` handling across samplers.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BaseSampler(ABC):
    """Abstract base class for all samplers.

    Parameters
    ----------
    random_state:
        Optional integer seed to make sampling reproducible.  If provided,
        samplers will derive a :class:`numpy.random.Generator` from it.
    """

    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        # We create a dedicated Generator so multiple samplers can share the
        # same seed without interfering with global RNG state.
        self._rng: np.random.Generator = np.random.default_rng(self.random_state)

    @abstractmethod
    def sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Return a sampled subset of ``df``.

        Implementations **must not** modify ``df`` in place.

        Parameters
        ----------
        df:
            Candidate pool.  Can contain any columns; samplers will look only
            at the columns they need (e.g., ``text_column`` or ``label_column``).
        n:
            Desired number of rows in the returned sample.  If ``n`` is larger
            than ``len(df)``, the sampler should gracefully fall back to
            returning all rows (without raising an error).

        Returns
        -------
        pandas.DataFrame
            A new ``DataFrame`` containing the sampled rows.
        """

    # Helper for subclasses
    def _normalize_n(self, df: pd.DataFrame, n: int) -> int:
        """Clamp ``n`` to ``[0, len(df)]`` and return it.

        This makes it safe to ask for more samples than rows without dealing
        with ``ValueError`` from :meth:`pandas.DataFrame.sample`.
        """
        if n <= 0:
            return 0
        return min(n, len(df))
