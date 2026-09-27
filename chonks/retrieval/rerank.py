"""Hybrid search reordered by a reranker: its rank over hybrid's top candidates
is fused with hybrid's own rank by weighted reciprocal rank."""

import logging
import time
from typing import Any

import httpx

from chonks.embed.reranker import Reranker, rerank_document
from chonks.retrieval.searcher import Searcher

RERANK_CANDIDATES = 50
RERANK_WEIGHT = 2.0
K_RRF = 60

logger = logging.getLogger("chonks.rerank")


def rerank_skipped_note(error: Exception) -> str:
    return (f"reranking was skipped because the reranker failed ({type(error).__name__}); "
            "these results are in hybrid order")


def fuse_ranks(chunks: list[dict[str, Any]], scores: list[float]) -> list[dict[str, Any]]:
    """`chunks` in hybrid order, reordered by 1/(K + hybrid rank) + w/(K + rerank rank).
    `scores` covers the first len(scores) chunks; the rest keep only their hybrid term."""
    by_score = sorted(range(len(scores)), key=lambda i: -scores[i])
    rerank_rank = {i: r for r, i in enumerate(by_score, 1)}
    fused = []
    for i, c in enumerate(chunks):
        c = dict(c)
        r = rerank_rank.get(i)
        c["_rank_hybrid"] = i + 1
        c["_rank_rerank"] = r
        c["_rerank_score"] = scores[i] if r is not None else None
        c["_score"] = 1 / (K_RRF + i + 1) + (RERANK_WEIGHT / (K_RRF + r) if r is not None else 0.0)
        fused.append(c)
    fused.sort(key=lambda c: (-c["_score"], c["_rank_hybrid"]))
    return fused


def reranked_hybrid(
    searcher: Searcher,
    reranker: Reranker,
    query: str,
    top_k: int = 50,
    path_prefix: str | None = None,
    chunk_kind: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """(chunks, note). Hybrid's top RERANK_CANDIDATES fused with the reranker's
    order; when the reranker fails, hybrid order and a note saying so."""
    chunks = searcher.hybrid(query, max(top_k, RERANK_CANDIDATES), path_prefix, chunk_kind=chunk_kind)
    docs = [rerank_document(c["path"], c.get("name"), c["content"]) for c in chunks[:RERANK_CANDIDATES]]
    t0 = time.perf_counter()
    try:
        with httpx.Client() as client:
            scores = reranker.rerank(query, docs, client)
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("rerank  candidates=%d  secs=%.2f  outcome=skipped  error=%s",
                       len(docs), time.perf_counter() - t0, e)
        return chunks[:top_k], rerank_skipped_note(e)
    logger.info("rerank  candidates=%d  secs=%.2f  outcome=ok", len(docs), time.perf_counter() - t0)
    return fuse_ranks(chunks, scores)[:top_k], None
