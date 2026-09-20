"""PageRank computation over chunk_refs: live compute, persistence with a
stale-batch skip gate, and the read path used at query time."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx

from chonks.core.edges import DEFAULT_EDGE_TYPE_WEIGHTS, _PAGERANK_STALE_META_KEY

if TYPE_CHECKING:
    from chonks.storage.store import Store

logger = logging.getLogger("repomap")

# PageRank has no incremental algorithm, so a small batch skips the live
# recompute and reuses stale scores; churn accumulates in the meta key
# (_PAGERANK_STALE_META_KEY) until it crosses this fraction, forcing a refresh.
_PAGERANK_REFRESH_MIN_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _compute_pagerank_live(
    store: "Store",
    chunks: list[dict] | None = None,
    edge_type_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Live PageRank over chunk_refs; expensive, index time should call
    persist_pagerank so query time reads persisted scores instead. Unknown
    edge_type weights default to 1.0, never erroring on an old or new DB."""
    if chunks is None:
        chunks = store.get_named_chunks_meta()
    if not chunks:
        return {}
    weights = edge_type_weights if edge_type_weights is not None else DEFAULT_EDGE_TYPE_WEIGHTS
    G = nx.DiGraph()
    for c in chunks:
        G.add_node(c["id"])
    for from_id, to_id, edge_type in store.get_all_refs_typed():
        # chunk_refs' PK is (from_id, to_id), one edge_type per pair, so
        # there's no duplicate-edge weight to accumulate here.
        G.add_edge(from_id, to_id, weight=weights.get(edge_type, 1.0))
    try:
        return nx.pagerank(G, alpha=0.85, max_iter=100, weight="weight")
    except nx.exception.PowerIterationFailedConvergence:
        return {c["id"]: 1.0 for c in chunks}


def persist_pagerank(
    store: "Store",
    *,
    changed_ids: set[str] | list[str] | None = None,
    deleted_ids: set[str] | list[str] | None = None,
    force: bool = False,
    edge_type_weights: dict[str, float] | None = None,
) -> int:
    """Computes PageRank live and persists it. When changed/deleted batches
    are small, skips the recompute and reuses stale scores (churn accumulates
    in meta) rather than paying a full-graph solve on every incremental index."""
    if not force and changed_ids is not None and deleted_ids is not None:
        existing = store.load_pagerank()
        if existing:
            total = store.count_chunks()
            stale = int(store.get_meta(_PAGERANK_STALE_META_KEY) or 0)
            batch = stale + len(changed_ids) + len(deleted_ids)
            if batch <= _PAGERANK_REFRESH_MIN_FRACTION * max(total, 1):
                store.set_meta(_PAGERANK_STALE_META_KEY, str(batch))
                logger.info(
                    "PageRank refresh skipped (cumulative batch=%d, N=%d) — "
                    "reusing persisted scores.",
                    batch, total,
                )
                return len(existing)
            logger.info(
                "PageRank refresh triggered (cumulative batch=%d, N=%d) — "
                "recomputing live.",
                batch, total,
            )

    scores = _compute_pagerank_live(store, edge_type_weights=edge_type_weights)
    store.save_pagerank(scores)
    store.set_meta(_PAGERANK_STALE_META_KEY, "0")
    return len(scores)


def compute_pagerank_global(
    store: "Store", edge_type_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Reads persisted PageRank from chunk_pagerank; never writes the DB, so
    a pre-persistence DB falls back to a live compute instead. Re-index to
    persist scores and skip that fallback on future requests."""
    persisted = store.load_pagerank()
    if persisted:
        return persisted
    chunks = store.get_named_chunks_meta()
    if not chunks:
        # Legitimately empty, not pre-migration: skip the live-compute
        # fallback and the stale-DB warning, or a zero-chunk repo would send
        # a user chasing a no-op re-index every query.
        return {}
    logger.warning(
        "chunk_pagerank table is empty — this DB predates persisted "
        "PageRank; falling back to a live (slow) compute. Re-index to "
        "persist scores and avoid this on future requests."
    )
    return _compute_pagerank_live(store, chunks, edge_type_weights=edge_type_weights)
