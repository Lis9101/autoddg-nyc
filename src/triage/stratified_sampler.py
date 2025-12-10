"""Stratified sampler for imbalanced labels.

If your pool already has *some* labels (e.g., a weak heuristic label, or an
earlier round of LLM annotations) you can use them to ensure that your next
batch covers all classes.

Typical use cases:

* Down‑stream evaluation: build a balanced validation set.
* Second‑round labeling: oversample rare classes so the classifier improves.

This is a simple, transparent alternative to more complex active‑learning
schemes.  It does **not** require model probabilities.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import BaseSampler


@dataclass
class StratifiedSampler(BaseSampler):
    """Sample rows in a label‑aware, approximately stratified way.

    Parameters
    ----------
    label_column:
        Column name that contains the labels or group IDs you want to
        stratify over.  This can be a true label, a weak label, or even a
        coarse cluster ID.
    min_per_label:
        Minimum number of examples to *attempt* to draw for each label.
        If the dataset does not have enough rows for a label, all available
        rows are used.
    """

    label_column: str = "label"
    min_per_label: int = 1

    def sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if self.label_column not in df.columns:
            raise KeyError(
                f"label_column='{self.label_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        n = self._normalize_n(df, n)
        if n == 0:
            return df.iloc[0:0].copy()

        # Group dataframe by label.
        grouped = df.groupby(self.label_column)

        # First pass: guarantee ``min_per_label`` samples where possible.
        pieces = []
        remaining = n

        for label, group in grouped:
            if remaining <= 0:
                break

            # How many can we take from this label?
            take = min(len(group), self.min_per_label, remaining)
            if take <= 0:
                continue

            sampled = group.sample(
                n=take,
                replace=False,
                random_state=self._rng.integers(0, 1_000_000),
            )
            pieces.append(sampled)
            remaining -= len(sampled)

        # Second pass: if we still have budget, fill remaining slots with a
        # proportionally stratified sample from the full pool (excluding
        # anything we've already selected).
        if remaining > 0:
            already_selected_idx = (
                pd.concat(pieces).index if pieces else pd.Index([], dtype=df.index.dtype)
            )
            leftover = df.drop(index=already_selected_idx)

            if len(leftover) > 0:
                # Compute proportional allocation.
                label_counts = leftover[self.label_column].value_counts()
                label_probs = label_counts / label_counts.sum()

                # Draw label ids according to their frequency, then sample
                # one row per drawn label.  This is a simple and reasonably
                # efficient approximate stratified scheme.
                labels_drawn = self._rng.choice(
                    label_counts.index.to_numpy(),
                    size=min(remaining, len(leftover)),
                    replace=True,
                    p=label_probs.to_numpy(),
                )

                # For each drawn label, sample one row (without replacement)
                # from that label's subset until we exhaust either budget or
                # available rows.
                leftover_groups = leftover.groupby(self.label_column)

                sampled_indices = []
                for label in labels_drawn:
                    group = leftover_groups.get_group(label)
                    # Only sample from rows not yet used for this label.
                    available_idx = group.index.difference(sampled_indices)
                    if len(available_idx) == 0:
                        continue
                    idx = self._rng.choice(available_idx.to_numpy())
                    sampled_indices.append(idx)

                    if len(sampled_indices) >= remaining:
                        break

                if sampled_indices:
                    pieces.append(df.loc[sampled_indices])

        if not pieces:
            # This happens only if df is empty.
            return df.iloc[0:0].copy()

        return pd.concat(pieces, axis=0)
