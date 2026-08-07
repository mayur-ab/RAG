"""Document embedding clustering for semantic collections (numpy K-Means)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.logging_config import logger
from config.settings import settings


def _choose_k(n: int) -> int:
    if n <= settings.DOC_CLUSTER_MIN_DOCS:
        return max(1, n // 2)
    auto = max(settings.DOC_CLUSTER_MIN_K, int(math.sqrt(n)))
    return min(settings.DOC_CLUSTER_MAX_K, auto, n)


def kmeans(
    vectors: np.ndarray,
    k: int,
    max_iters: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run K-Means. Returns (labels, centroids)."""
    n, d = vectors.shape
    if k >= n:
        return np.arange(n), vectors.copy()

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    centroids = vectors[indices].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        dists = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centroids[j] = vectors[mask].mean(axis=0)
            else:
                centroids[j] = vectors[rng.integers(0, n)]

    return labels, centroids


def cluster_documents(
    items: List[Dict[str, Any]],
    k: Optional[int] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """Group documents by embedding similarity."""
    if not items:
        return {}

    vectors = np.array([item["embedding"] for item in items], dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    n = len(items)
    k = k or _choose_k(n)
    k = max(1, min(k, n))

    labels, _ = kmeans(vectors, k)
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(items[idx])

    logger.info("Clustered %s documents into %s groups (k=%s)", n, len(grouped), k)
    return grouped
