"""Keyword‑biased sampler.

Many triage tasks have a small list of *high‑precision* keywords (e.g., names
of species or product types) that correlate with the positive class.  This
sampler uses such keywords to bias sampling **toward** potentially relevant
examples while still keeping some randomness.

High‑level idea
---------------

1. Compute a simple keyword count for each row.
2. Prioritize rows with non‑zero counts.
3. If there are more matching rows than you need, randomly sample from them.
4. If there are fewer, take all of them and fill the remaining budget
   from the rest of the pool uniformly at random.

This is intentionally lightweight and does **not** require any model
probabilities or external libraries.

The text column can be the ad *title*, *description*, or any column that
contains the main text you want to search.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import pandas as pd

from .base import BaseSampler


def _normalize_keywords(keywords: Iterable[str]) -> List[str]:
    # Lower‑case and strip whitespace to make matching more robust.
    return [k.strip().lower() for k in keywords if k.strip()]


@dataclass
class KeywordSampler(BaseSampler):
    """Sampler that prioritizes rows containing specific keywords.

    Parameters
    ----------
    keywords:
        Iterable of keyword strings (e.g., species names or domain terms).
        Matching is case‑insensitive and uses a simple substring check.
    text_column:
        Name of the column that contains the text to search over
        (e.g., ``"title"`` or ``"description"``).
    add_score_column:
        If ``True``, the returned ``DataFrame`` will include a new column
        ``"keyword_score"`` with the number of keyword matches in each row.
        This can be useful later for analysis or debugging.
    """

    keywords: Iterable[str] = field(default_factory=list)
    text_column: str = "text"
    add_score_column: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self._keywords = _normalize_keywords(self.keywords)

    def _score_row(self, text: str) -> int:
        """Return the number of keywords that appear in ``text`` (case‑insensitive)."""
        if not isinstance(text, str):
            return 0
        lower = text.lower()
        return sum(1 for kw in self._keywords if kw in lower)

    def sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if not self._keywords:
            # Fall back to pure random if no keywords are provided.
            from .random_sampler import RandomSampler

            return RandomSampler(random_state=self.random_state).sample(df, n)

        n = self._normalize_n(df, n)
        if n == 0:
            return df.iloc[0:0].copy()

        if self.text_column not in df.columns:
            raise KeyError(
                f"text_column='{self.text_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        # Compute keyword score for each row.
        scores = df[self.text_column].map(self._score_row)
        matching_mask = scores > 0

        matching = df[matching_mask]
        non_matching = df[~matching_mask]

        selected_frames = []
        remaining = n

        if len(matching) > 0:
            # If we have more matching rows than we need, sample from them.
            take = min(len(matching), remaining)
            selected_frames.append(
                matching.sample(
                    n=take,
                    replace=False,
                    random_state=self._rng.integers(0, 1_000_000),
                )
            )
            remaining -= take

        if remaining > 0 and len(non_matching) > 0:
            # Fill remaining budget randomly from the rest of the pool.
            take = min(len(non_matching), remaining)
            selected_frames.append(
                non_matching.sample(
                    n=take,
                    replace=False,
                    random_state=self._rng.integers(0, 1_000_000),
                )
            )

        if not selected_frames:
            # This happens only when df is empty.
            return df.iloc[0:0].copy()

        result = pd.concat(selected_frames, axis=0)

        if self.add_score_column:
            # Align scores with the sampled index.
            result = result.copy()
            result["keyword_score"] = scores.loc[result.index]

        return result
