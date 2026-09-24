"""
searcher.py: thin query layer on top of Store. Semantic/FTS/regex search,
RRF hybrid fusion, and folder-blend re-rank.
"""

import posixpath
import re
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from chonks.embed.client import (
    EMBED_QUERY_TOKEN_BUDGET,
    QUERY_CHARS_PER_TOKEN,
    Embedder,
    truncate_query_text,
)
from chonks.retrieval.query_reformulate import DEFAULT_REFORMULATE_QUERY, augment_query
from chonks.storage.store import Store

# FTS5 treats punctuation and bare AND/OR/NOT as operators, so free text
# raises a syntax error. hybrid() sanitises before querying FTS; raw fts()
# keeps the FTS5-syntax contract for power users.
_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _sanitize_fts_query(query: str) -> str:
    """Quote each word token so FTS5 treats it as a literal, not an operator.
    Multiple quoted literals stay implicit AND.
    """
    tokens = _FTS_TOKEN_RE.findall(query)
    return " ".join(f'"{t}"' for t in tokens)


# unicode61 keeps camelCase tokens whole, so query "fileCap" won't match
# indexed `file_cap` unless split into word parts.
_CAMEL_SPLIT_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_HAS_CAMEL_RE = re.compile(r"[a-z][A-Z]")


def _split_identifier(token: str) -> list[str]:
    """Word-parts of a camelCase/PascalCase token, else [].
    snake_case is excluded on purpose: it's already split at index time, and
    re-splitting it here measurably hurts ranking (adds only noise)."""
    if "_" in token or not _HAS_CAMEL_RE.search(token):
        return []
    return [p for p in _CAMEL_SPLIT_RE.findall(token) if len(p) > 1]


def _or_fts_query(query: str) -> str:
    """OR-joined variant of _sanitize_fts_query, incl. camelCase word-parts.
    Tokens deduped case-insensitively to avoid double-counting bm25."""
    seen: set[str] = set()
    toks: list[str] = []
    for t in _FTS_TOKEN_RE.findall(query):
        for piece in (t, *_split_identifier(t)):
            k = piece.lower()
            if k not in seen:
                seen.add(k)
                toks.append(piece)
    return " OR ".join(f'"{t}"' for t in toks)


# FTS starvation fallback: if implicit-AND finds <top_k hits, retry
# OR-joined but admit a hit only if it's in the semantic pool AND ranks
# <= OR_FALLBACK_MAX_RANK. Unrestricted OR admission measurably hurt ranking.
OR_FALLBACK_MAX_RANK = 20

# α weights chunk similarity, β weights folder-summary similarity. No
# hard-filtering, so a failed re-rank still degrades to plain semantic search.
DEFAULT_BLEND_ALPHA = 1.0
DEFAULT_BLEND_BETA  = 0.2

# Oversample factor for folder-blend re-rank before truncating to top_k.
_FOLDER_BLEND_OVERSAMPLE = 4

# Oversample factor for the per-file cap: lets a chunk skipped for exceeding
# its cap be backfilled from a different-file chunk. No effect if file_cap==0.
_FILE_CAP_OVERSAMPLE = 4

# 0 = off: plain rank-ordered top_k truncation.
DEFAULT_FILE_CAP = 0


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector has zero norm."""
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _int8_cosine(query_vec: list[float], blob: bytes) -> float:
    """True cosine(query, chunk) from a stored int8 blob. sqlite-vec ranks
    chunk_vecs by raw L2 (~117-142), not cosine; order matches but the scale
    doesn't, so compute cosine directly instead of trusting that distance."""
    return _cosine(query_vec, np.frombuffer(blob, dtype=np.int8))


def _cap_per_file(
    chunks: list[dict[str, Any]], top_k: int, file_cap: int,
) -> list[dict[str, Any]]:
    """Truncate rank-ordered `chunks` to `top_k`, capped at `file_cap` per
    path. Skips (never reorders) chunks over cap so slots backfill from the
    next-ranked, different-file chunk; input should be oversampled."""
    if file_cap <= 0:
        return chunks[:top_k]
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for c in chunks:
        if len(selected) >= top_k:
            break
        path = c.get("path", "")
        if counts.get(path, 0) >= file_cap:
            continue
        counts[path] = counts.get(path, 0) + 1
        selected.append(c)
    return selected


class Searcher:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        reformulate_query: bool = DEFAULT_REFORMULATE_QUERY,
    ):
        self._store    = store
        self._embedder = embedder
        # Default OFF. Only affects the SEMANTIC embed side, never FTS.
        self._reformulate_query = reformulate_query

    @property
    def store(self) -> Store:
        """Expose the underlying store for read-only consumers (e.g. research.py
        graph traversal) without committing to private-attribute access."""
        return self._store

    def query_truncated(self, query: str) -> bool:
        """Whether `query` would be truncated on the embed side; FTS is never
        truncated. Checks the RAW query only, so this is an approximate
        diagnostic, not exactly what gets embedded when reformulation is on."""
        budget = getattr(self._embedder, "query_token_budget", EMBED_QUERY_TOKEN_BUDGET)
        return truncate_query_text(query, budget)[1]

    def _augmented_query(self, query: str) -> str:
        """Appends extracted identifiers to `query`, budget-aware so the
        appended tail survives the embedder's own truncation. No-op (returns
        `query` unchanged) when reformulate_query is off (default)."""
        if not self._reformulate_query:
            return query
        budget = getattr(self._embedder, "query_token_budget", EMBED_QUERY_TOKEN_BUDGET)
        budget_chars = int(budget * QUERY_CHARS_PER_TOKEN)
        return augment_query(query, enabled=True, budget_chars=budget_chars)

    def _embed_query(self, query: str, client: httpx.Client) -> list[float]:
        # Tighter timeout than batch indexing (30s vs 120s): interactive
        # queries should fail fast on a hung server.
        query = self._augmented_query(query)
        return self._embedder.embed_queries([query], client, timeout=30.0)[0]

    def embed_query(self, query: str, client: httpx.Client | None = None) -> list[float]:
        """Public wrapper over `_embed_query` owning the httpx client if none
        is passed. For callers needing the raw vector (e.g. research's
        reranker) rather than a result set."""
        _client = client or httpx.Client()
        try:
            return self._embed_query(query, _client)
        finally:
            if client is None:
                _client.close()

    def semantic(
        self,
        query: str,
        top_k: int = 50,
        path_prefix: str | None = None,
        client: httpx.Client | None = None,
        min_score: float | None = None,
        folder_blend: tuple[float, float] | None = None,
        chunk_kind: str | None = None,
        file_cap: int = DEFAULT_FILE_CAP,
    ) -> list[dict[str, Any]]:
        """Vector search over chunk embeddings. min_score filters against
        the *final* score: the blended score when folder_blend is set, not
        raw chunk similarity."""
        _client = client or httpx.Client()
        try:
            vec = self._embed_query(query, _client)
        finally:
            if client is None:
                _client.close()

        if folder_blend is None:
            fetch_k = top_k * _FILE_CAP_OVERSAMPLE if file_cap > 0 else top_k
            candidates = self._store.search_semantic(vec, fetch_k, path_prefix, chunk_kind=chunk_kind)
            results = _cap_per_file(candidates, top_k, file_cap)
        else:
            alpha, beta = folder_blend
            oversample = top_k * _FOLDER_BLEND_OVERSAMPLE
            if file_cap > 0:
                oversample = max(oversample, top_k * _FILE_CAP_OVERSAMPLE)
            candidates = self._store.search_semantic(
                vec, oversample, path_prefix, chunk_kind=chunk_kind,
            )
            if candidates:
                folder_paths = list({
                    posixpath.dirname(c["path"]) or "." for c in candidates
                })
                folder_embs = self._store.get_folder_embeddings(folder_paths)
                folder_sims: dict[str, float] = {}
                for fp, emb in folder_embs.items():
                    # Stale folder-embedding dim (model/dim changed, summary
                    # not regenerated) would raise ValueError in np.dot.
                    # Skip it; falls back to folder_sim = 0.0 below.
                    if len(emb) != len(vec):
                        continue
                    folder_sims[fp] = _cosine(vec, emb)
                chunk_blobs = self._store.get_int8_embeddings_by_ids(
                    [c["id"] for c in candidates]
                )
                for c in candidates:
                    folder = posixpath.dirname(c["path"]) or "."
                    blob = chunk_blobs.get(c["id"])
                    chunk_sim = _int8_cosine(vec, blob) if blob else 0.0
                    folder_sim = folder_sims.get(folder, 0.0)
                    c["_chunk_sim"] = chunk_sim
                    c["_folder_sim"] = folder_sim
                    c["_blended_score"] = alpha * chunk_sim + beta * folder_sim
                candidates.sort(key=lambda c: -c["_blended_score"])
            results = _cap_per_file(candidates, top_k, file_cap)

        if min_score is not None:
            if folder_blend is not None:
                results = [c for c in results if c.get("_blended_score", 0.0) >= min_score]
            else:
                blobs = self._store.get_int8_embeddings_by_ids([c["id"] for c in results])
                results = [
                    c for c in results
                    if (b := blobs.get(c["id"])) is not None
                    and _int8_cosine(vec, b) >= min_score
                ]

        return results

    def fts(
        self,
        query: str,
        top_k: int = 50,
        path_prefix: str | None = None,
        chunk_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        # Qualified symbols (`Foo::Bar`) can hit FTS5 as operators; retry
        # sanitised (same as hybrid()) instead of raising 500.
        try:
            return self._store.search_fts(query, top_k, path_prefix, chunk_kind=chunk_kind)
        except sqlite3.OperationalError:
            safe = _sanitize_fts_query(query)
            if not safe:
                return []
            return self._store.search_fts(safe, top_k, path_prefix, chunk_kind=chunk_kind)

    def keyword(
        self,
        query: str,
        top_k: int = 50,
        path_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        """FTS for a freeform query: chunks with every word, then chunks with any."""
        and_query = _sanitize_fts_query(query)
        hits = self.fts(and_query, top_k, path_prefix) if and_query else []
        or_query = _or_fts_query(query)
        if len(hits) < top_k and or_query:
            seen = {h["id"] for h in hits}
            more = [h for h in self.fts(or_query, top_k, path_prefix) if h["id"] not in seen]
            hits += more[: top_k - len(hits)]
        return hits

    def regex(
        self,
        pattern: str,
        top_k: int = 50,
        path_prefix: str | None = None,
        chunk_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._store.search_regex(pattern, top_k, path_prefix, chunk_kind=chunk_kind)

    def hybrid(
        self,
        query: str,
        top_k: int = 50,
        path_prefix: str | None = None,
        client: httpx.Client | None = None,
        chunk_kind: str | None = None,
        file_cap: int = DEFAULT_FILE_CAP,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion of semantic and FTS branches (K_RRF=60,
        Cormack et al. 2009). file_cap needs no extra oversampling here:
        each branch already fetches 2*top_k unconditionally."""
        K_RRF = 60
        oversample = 2 * top_k

        # Only FTS needs the freeform-to-bareword translation.
        fts_query = _sanitize_fts_query(query)

        _client = client or httpx.Client()
        try:
            sem_results = self.semantic(query, oversample, path_prefix, client=_client, chunk_kind=chunk_kind)
            # RRF degrades to semantic-only if sanitisation empties the query.
            fts_results = self.fts(fts_query, oversample, path_prefix, chunk_kind=chunk_kind) if fts_query else []
        finally:
            if client is None:
                _client.close()

        sem_ranks = {c["id"]: i + 1 for i, c in enumerate(sem_results)}
        fts_ranks = {c["id"]: i + 1 for i, c in enumerate(fts_results)}

        # OR ranks go into fts_ranks only, never fts_results: chunks_by_id
        # below is built from fts_results, so this keeps OR-only chunks from
        # ever becoming output rows.
        if fts_query and len(fts_results) < top_k:
            or_query = _or_fts_query(query)
            if or_query:
                or_results = self.fts(or_query, oversample, path_prefix, chunk_kind=chunk_kind)
                n_and = len(fts_results)
                for i, c in enumerate(or_results[:OR_FALLBACK_MAX_RANK]):
                    cid = c["id"]
                    if cid in sem_ranks and cid not in fts_ranks:
                        fts_ranks[cid] = n_and + i + 1

        # Prefer the semantic-branch copy when both exist (same row data either way).
        chunks_by_id: dict[str, dict[str, Any]] = {}
        for c in sem_results:
            chunks_by_id[c["id"]] = dict(c)
        for c in fts_results:
            chunks_by_id.setdefault(c["id"], dict(c))

        fused: list[dict[str, Any]] = []
        for cid, chunk in chunks_by_id.items():
            sem_rank = sem_ranks.get(cid)
            fts_rank = fts_ranks.get(cid)
            score = 0.0
            if sem_rank is not None:
                score += 1.0 / (K_RRF + sem_rank)
            if fts_rank is not None:
                score += 1.0 / (K_RRF + fts_rank)
            chunk["_score"] = score
            chunk["_rank_semantic"] = sem_rank
            chunk["_rank_fts"] = fts_rank
            chunk.pop("distance", None)
            chunk.pop("fts_rank", None)
            fused.append(chunk)

        fused.sort(key=lambda c: (-c["_score"], c["id"]))
        return _cap_per_file(fused, top_k, file_cap)
