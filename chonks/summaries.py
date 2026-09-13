"""
Per-folder structural summaries for hierarchical retrieval, embedded and stored in `folder_summaries`.
Structural only, no LLM: substitutes the existing folder hierarchy for RAPTOR's clustering+LLM tree.
"""

from __future__ import annotations

import hashlib
import logging
import posixpath
from collections import defaultdict
from typing import TYPE_CHECKING, Callable

import numpy as np

from chonks.repomap import compute_pagerank_global

if TYPE_CHECKING:
    from chonks.store import Store

logger = logging.getLogger(__name__)

TOP_SYMBOLS_PER_FOLDER = 12

# Caps embed request size below the embedding server's per-request limit.
SUMMARY_EMBED_BATCH = 64

# 0.30 cosine distance is about 0.70 cosine similarity.
DEFAULT_SUBSYSTEM_DISTANCE_THRESHOLD = 0.30


def _aggregate_hash(content_hashes: list[str]) -> str:
    """Hash of a folder's file hashes, sorted so member order can't change the result."""
    h = hashlib.sha256()
    for fh in sorted(content_hashes):
        h.update(fh.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def _format_folder_summary(
    folder: str,
    file_paths: list[str],
    folder_chunks: list[dict],
    scores: dict[str, float],
) -> str:
    """Text embedded for the folder; format is chosen so different folder kinds embed distinctly."""
    parent_token = posixpath.basename(folder) if folder not in ("", ".") else "(root)"

    ext_counts: dict[str, int] = defaultdict(int)
    for p in file_paths:
        ext = posixpath.splitext(p)[1].lower() or "(none)"
        ext_counts[ext] += 1
    total = len(file_paths)
    ext_dist = ", ".join(
        f"{ext} {round(100 * c / total)}%"
        for ext, c in sorted(ext_counts.items(), key=lambda kv: -kv[1])
    )

    ranked = sorted(folder_chunks, key=lambda c: -scores.get(c["id"], 0.0))
    top_symbols = [c["name"] for c in ranked if c["name"]][:TOP_SYMBOLS_PER_FOLDER]
    symbols_line = ", ".join(top_symbols) if top_symbols else "(no named symbols)"

    return (
        f"{folder}/  [{parent_token}]\n"
        f"  files: {total} ({ext_dist})\n"
        f"  top symbols: {symbols_line}"
    )


def build_folder_summaries(
    store: "Store",
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> dict[str, int]:
    """Refreshes `folder_summaries`; skips folders whose content_hash is unchanged."""
    files = store.get_all_files()
    if not files:
        for stale in store.get_all_folder_paths():
            store.delete_folder_summary(stale)
        store.commit()
        return {"refreshed": 0, "pruned": 0}

    files_by_folder: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        folder = posixpath.dirname(f["path"]) or "."
        files_by_folder[folder].append(f)

    chunks_by_folder: dict[str, list[dict]] = defaultdict(list)
    for c in store.get_named_chunks_meta():
        folder = posixpath.dirname(c["path"]) or "."
        chunks_by_folder[folder].append(c)

    scores = compute_pagerank_global(store)

    pending: list[tuple[str, str, str]] = []  # (folder, summary_text, content_hash)
    for folder, folder_files in files_by_folder.items():
        content_hash = _aggregate_hash([f["content_hash"] for f in folder_files])
        existing = store.get_folder_summary(folder)
        if existing is not None and existing["content_hash"] == content_hash:
            continue
        summary = _format_folder_summary(
            folder,
            [f["path"] for f in folder_files],
            chunks_by_folder.get(folder, []),
            scores,
        )
        pending.append((folder, summary, content_hash))

    # Prune before embedding: a failed embed call must not leave stale rows.
    indexed_folders = set(files_by_folder.keys())
    pruned = 0
    for stored in store.get_all_folder_paths():
        if stored not in indexed_folders:
            store.delete_folder_summary(stored)
            pruned += 1

    refreshed = 0
    if pending:
        for i in range(0, len(pending), SUMMARY_EMBED_BATCH):
            batch = pending[i:i + SUMMARY_EMBED_BATCH]
            texts = [p[1] for p in batch]
            try:
                embeddings = embed_fn(texts)
            except Exception as e:
                logger.error(
                    "Failed to embed folder-summary batch (%d items): %s — "
                    "remaining %d summaries skipped this run",
                    len(batch), e, len(pending) - i - len(batch),
                )
                break
            for (folder, summary, content_hash), emb in zip(batch, embeddings):
                store.upsert_folder_summary(folder, summary, emb, content_hash)
                refreshed += 1

    store.commit()
    return {"refreshed": refreshed, "pruned": pruned}


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
