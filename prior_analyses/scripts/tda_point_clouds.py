#!/usr/bin/env python
"""Helpers for topological data analysis (TDA) of two SciVer point clouds.

Cloud A: claim/evidence pair embeddings (e.g. `pair_text_vec`, 384-dim) read from the
         local Qdrant store. Cosine-normalized, so cosine distance is natural.
Cloud B: the 12 fully-scored difficulty dimensions from the faithfulness dataset
         (817 x 12 integer scores). Standardized, so Euclidean distance is natural.

Because the two clouds live in different ambient spaces and scales, each distance
matrix is normalized by its own maximum so the filtration axes are comparable in
[0, 1]. Comparing topology across different spaces is heuristic; treat the diagrams
as shape summaries rather than ground truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

DEFAULT_QDRANT_PATH = Path("data/processed/qdrant_sciver")
DEFAULT_FAITHFULNESS = Path("data/processed/faithfulness_dataset.parquet")
COMPLETE_DIMS = [
    "dim_AS", "dim_AT", "dim_CL", "dim_GS", "dim_KNf", "dim_MA",
    "dim_MCr", "dim_MCu", "dim_QLl", "dim_QLq", "dim_VL", "dim_VO",
]


def load_embedding_cloud(
    qdrant_path: Path = DEFAULT_QDRANT_PATH,
    collection: str = "sciver_pairs",
    vector: str = "pair_text_vec",
) -> np.ndarray:
    """Pull every pair's named vector from Qdrant into an (n, d) array."""
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(qdrant_path))
    try:
        rows, offset = [], None
        while True:
            points, offset = client.scroll(
                collection, limit=512, offset=offset, with_vectors=True, with_payload=False
            )
            for p in points:
                vec = (p.vector or {}).get(vector)
                if vec is not None:
                    rows.append(vec)
            if offset is None:
                break
    finally:
        client.close()
    return np.asarray(rows, dtype=float)


def load_difficulty_cloud(
    parquet_path: Path = DEFAULT_FAITHFULNESS, standardize: bool = True
) -> np.ndarray:
    """Load the 12 complete difficulty dimensions as an (n, 12) array."""
    df = pd.read_parquet(parquet_path)
    X = df[COMPLETE_DIMS].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("Unexpected NaNs in the 12 complete dimensions.")
    return StandardScaler().fit_transform(X) if standardize else X


def subsample(X: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Take a reproducible random subset of rows (keeps TDA tractable and N matched)."""
    if n >= len(X):
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


def normalized_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    """Pairwise distances scaled by their maximum, so the filtration spans [0, 1]."""
    D = pairwise_distances(X, metric=metric)
    dmax = D.max()
    return D / dmax if dmax > 0 else D


def persistence(D: np.ndarray, maxdim: int = 1, thresh: float | None = None) -> list[np.ndarray]:
    """Persistent homology from a precomputed distance matrix; returns [H0, H1, ...]."""
    from ripser import ripser

    kwargs = {"distance_matrix": True, "maxdim": maxdim}
    if thresh is not None:
        kwargs["thresh"] = thresh
    return ripser(D, **kwargs)["dgms"]


def finite_bars(dgm: np.ndarray) -> np.ndarray:
    """Drop bars with infinite death (e.g. the H0 essential class)."""
    if len(dgm) == 0:
        return dgm
    return dgm[np.isfinite(dgm[:, 1])]


def betti_curve(dgm: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Betti number of one homology dimension as a function of filtration value."""
    if len(dgm) == 0:
        return np.zeros_like(grid)
    births, deaths = dgm[:, 0], dgm[:, 1]
    deaths = np.where(np.isfinite(deaths), deaths, np.inf)
    return np.array([(np.sum((births <= e) & (deaths > e))) for e in grid])


def significant_betti(dgms: list[np.ndarray], min_persistence: float) -> dict[str, int]:
    """Count long-lived features per dimension (a scale-robust 'discrete Betti' summary)."""
    out = {}
    for k, dgm in enumerate(dgms):
        bars = finite_bars(dgm)
        pers = bars[:, 1] - bars[:, 0] if len(bars) else np.array([])
        out[f"b{k}"] = int(np.sum(pers >= min_persistence))
    return out
