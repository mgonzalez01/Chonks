"""Candidate subsystem groupings from the folder summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from chonks.store import Store

# 0.30 cosine distance is about 0.70 cosine similarity.
DEFAULT_SUBSYSTEM_DISTANCE_THRESHOLD = 0.30


def suggest_subsystems(
    store: "Store",
    distance_threshold: float = DEFAULT_SUBSYSTEM_DISTANCE_THRESHOLD,
    min_cluster_size: int = 2,
) -> list[list[str]]:
    """Clusters folder summaries by cosine distance into candidate subsystem groupings.
    """
    paths = store.get_all_folder_paths()
    if len(paths) < 2:
        return []

    embeddings: list[list[float]] = []
    valid_paths: list[str] = []
    for p in paths:
        s = store.get_folder_summary(p)
        if s and s["summary_embedding"]:
            embeddings.append(s["summary_embedding"])
            valid_paths.append(p)
    if len(valid_paths) < 2:
        return []

    # scipy over HDBSCAN: avoids a heavyweight dep for a one-shot CLI command.
    from scipy.cluster.hierarchy import fcluster, linkage  # noqa: PLC0415
    from scipy.spatial.distance import pdist  # noqa: PLC0415

    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-8)

    condensed = pdist(arr, metric="cosine")
    linkage_matrix = linkage(condensed, method="average")
    cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")

    by_cluster: dict[int, list[str]] = defaultdict(list)
    for path, c in zip(valid_paths, cluster_labels):
        by_cluster[int(c)].append(path)

    return [sorted(p) for p in by_cluster.values() if len(p) >= min_cluster_size]
