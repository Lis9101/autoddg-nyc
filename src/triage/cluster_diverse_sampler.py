"""Cluster‑diverse sampler.

This sampler assumes that each row has already been assigned to a *cluster*
(e.g., via k‑means on text embeddings).  It then draws a batch that covers
as many different clusters as possible, similar in spirit to the diversity
component of Lean‑To‑Sample (LTS) style algorithms.

It is **not** a full multi‑armed bandit implementation, but it provides a
simple, inspectable way to get diverse batches without pulling in extra
dependencies.

Typical pipeline
----------------

1. Compute embeddings for your text (title/description).
2. Run your favorite clustering method (k‑means, spectral, etc.).
3. Store cluster ids in a column, e.g. ``cluster_id``.
4. Use :class:`ClusterDiverseSampler` to select batches for LLM labeling.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import BaseSampler


@dataclass
class ClusterDiverseSampler(BaseSampler):
    """Sampler that tries to cover many clusters.

    Parameters
    ----------
    cluster_column:
        Column name that contains integer cluster identifiers.
    max_per_cluster:
        Upper bound on how many rows to draw from a single cluster.  This
        avoids a single very‑large cluster dominating the batch.
    """

    cluster_column: str = "cluster_id"
    max_per_cluster: int = 10

    def sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if self.cluster_column not in df.columns:
            raise KeyError(
                f"cluster_column='{self.cluster_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        n = self._normalize_n(df, n)
        if n == 0:
            return df.iloc[0:0].copy()

        grouped = df.groupby(self.cluster_column)

        # We will take up to ``max_per_cluster`` rows from each cluster, but we
        # also limit ourselves to the requested batch size ``n``.
        pieces = []
        remaining = n

        # Shuffle cluster order for fairness.
        cluster_ids = list(grouped.groups.keys())
        self._rng.shuffle(cluster_ids)

        # First round: one example per cluster (if available).
        round_indices = []

        for cid in cluster_ids:
            if remaining <= 0:
                break
            group = grouped.get_group(cid)
            # Sample a single representative row from this cluster.
            idx = self._rng.choice(group.index.to_numpy())
            round_indices.append(idx)
            remaining -= 1

        if round_indices:
            pieces.append(df.loc[round_indices])

        # Second round: if we still have budget, keep cycling over clusters and
        # add more rows up to ``max_per_cluster`` per cluster.
        if remaining > 0:
            used_counts = (
                pieces[0]
                .groupby(self.cluster_column)
                .size()
                .reindex(cluster_ids, fill_value=0)
            )

            # Precompute group indices for speed.
            group_indices = {
                cid: grouped.get_group(cid).index.to_numpy() for cid in cluster_ids
            }

            # Track which indices we have already used.
            used_indices = set(round_indices)

            # Simple round‑robin over clusters.
            while remaining > 0:
                progress = False
                for cid in cluster_ids:
                    if remaining <= 0:
                        break
                    if used_counts.loc[cid] >= self.max_per_cluster:
                        continue

                    candidates = np.array(
                        [idx for idx in group_indices[cid] if idx not in used_indices]
                    )
                    if len(candidates) == 0:
                        continue

                    idx = self._rng.choice(candidates)
                    used_indices.add(idx)
                    used_counts.loc[cid] += 1
                    remaining -= 1
                    progress = True

                    if remaining <= 0:
                        break

                if not progress:
                    # No clusters could provide additional samples.
                    break

            # Combine any extra rows we just selected.
            extra_indices = [idx for idx in used_indices if idx not in round_indices]
            if extra_indices:
                pieces.append(df.loc[extra_indices])

        if not pieces:
            return df.iloc[0:0].copy()

        return pd.concat(pieces, axis=0)
