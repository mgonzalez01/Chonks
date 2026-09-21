"""The passes that run after the embed stage of an indexing run."""

import logging
import time

import httpx

from chonks.index.graph.hierarchy import rebuild_hierarchy
from chonks.index.graph.knn import build_neighbors
from chonks.index.graph.refs import build_refs
from chonks.index.graph.pagerank import persist_pagerank
from chonks.index.summaries import build_folder_summaries

logger = logging.getLogger("chonks.chunker")


def run_post_index_passes(store, embedder, *, force, indexed, pruned, changed_chunk_ids,
                          deleted_chunk_ids, deleted_chunk_names, edge_type_weights,
                          cap_mentions_fanout, associated_top_frac):
    """Returns the elapsed seconds of each pass: (fts, refs, knn, summaries, pagerank, hierarchy)."""
    # After a full force re-index, rebuild FTS statistics from scratch.
    # Repeated partial updates cause term-frequency drift; a rebuild corrects it.
    fts_elapsed = 0.0
    if force and indexed > 0:
        _phase_t0 = time.monotonic()
        store.rebuild_fts()
        fts_elapsed = time.monotonic() - _phase_t0

    # Timed independently: on a large corpus these passes can roughly double
    # wall time beyond embed_elapsed alone.
    refs_elapsed = 0.0
    knn_elapsed = 0.0
    summaries_elapsed = 0.0
    pagerank_elapsed = 0.0
    hierarchy_elapsed = 0.0

    # build_refs takes the incremental path unless --force (None/None/None
    # forces the bulk path; mirrors build_neighbors below).
    if (
        indexed > 0
        or pruned > 0
        or deleted_chunk_ids
        or changed_chunk_ids
    ):
        _phase_t0 = time.monotonic()
        ref_count = build_refs(
            store,
            changed_ids=None if force else changed_chunk_ids,
            deleted_ids=None if force else deleted_chunk_ids,
            deleted_names=None if force else deleted_chunk_names,
            cap_mentions=cap_mentions_fanout,
            associated_top_frac=associated_top_frac,
        )
        refs_elapsed = time.monotonic() - _phase_t0
        logger.info("Rebuilt %d cross-reference edges.", ref_count)

        # build_neighbors takes the incremental path unless --force, same
        # convention as build_refs above.
        _phase_t0 = time.monotonic()
        neighbor_count = build_neighbors(
            store,
            changed_ids=None if force else changed_chunk_ids,
            deleted_ids=None if force else deleted_chunk_ids,
        )
        knn_elapsed = time.monotonic() - _phase_t0
        logger.info("Built %d k-NN neighbour edges.", neighbor_count)

        # Must run AFTER build_refs/build_neighbors, not before: their
        # incremental paths rely on this run's own dangling rows, so purging
        # first would desync them from a full rebuild. Gated to run once per DB.
        if force or not store.get_meta("orphan_sweep_v1"):
            orphan_neighbors = store.purge_orphan_neighbors()
            orphan_refs = store.purge_orphan_refs()
            if orphan_neighbors:
                logger.info("Purged %d orphan chunk_neighbors edge(s).", orphan_neighbors)
            if orphan_refs:
                logger.info("Purged %d orphan chunk_refs edge(s).", orphan_refs)
            store.set_meta("orphan_sweep_v1", "1")

        # Full rebuild every run: cheap relative to refs/kNN (see
        # rebuild_hierarchy), so no incremental path to keep in sync.
        _phase_t0 = time.monotonic()
        hierarchy = rebuild_hierarchy(store)
        hierarchy_elapsed = time.monotonic() - _phase_t0
        logger.info("Built hierarchy: %d nodes, %d contains edges.",
                    hierarchy["nodes"], hierarchy["edges"])

        # Computed here, not at query time. None ids force a full
        # recompute (same convention as build_refs/build_neighbors); 
        # with no true incremental algorithm, None here skips it, not just cheapens it.
        _phase_t0 = time.monotonic()
        pagerank_count = persist_pagerank(
            store,
            changed_ids=None if force else changed_chunk_ids,
            deleted_ids=None if force else deleted_chunk_ids,
            force=force,
            edge_type_weights=edge_type_weights,
        )
        pagerank_elapsed = time.monotonic() - _phase_t0
        logger.info("Persisted %d PageRank scores.", pagerank_count)

        # Refresh per-folder structural summaries (incremental: only folders
        # whose aggregate content_hash changed get re-embedded).
        _phase_t0 = time.monotonic()
        try:
            with httpx.Client() as client:
                summary_stats = build_folder_summaries(
                    store,
                    lambda texts: embedder.embed_documents(texts, client),
                )
            logger.info(
                "Folder summaries: refreshed %d, pruned %d.",
                summary_stats["refreshed"], summary_stats["pruned"],
            )
        except Exception as e:
            logger.error("Folder summary generation failed: %s", e)
        summaries_elapsed = time.monotonic() - _phase_t0
    return fts_elapsed, refs_elapsed, knn_elapsed, summaries_elapsed, pagerank_elapsed, hierarchy_elapsed
