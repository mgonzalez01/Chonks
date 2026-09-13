"""
searcher.py: thin query layer on top of Store. Semantic/FTS/regex search,
RRF hybrid fusion, folder-blend re-rank, and result formatting for LLM
consumption.
"""

import posixpath
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from chonks.embedder import (
    EMBED_QUERY_TOKEN_BUDGET,
    QUERY_CHARS_PER_TOKEN,
    Embedder,
    truncate_query_text,
)
from chonks.query_reformulate import DEFAULT_REFORMULATE_QUERY, augment_query
from chonks.store import Store

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

# TAU: cosine to count as "same wall". Both a naively high (>=0.9) and low
# (<=0.75) value fail: confirmed dupes rarely exceed ~0.9, and single-topic
# results are correlated enough to false-positive at low thresholds.
NEAR_DUP_TAU = 0.85
NEAR_DUP_WALL_SHARE_THRESHOLD = 0.6
# Below this many vector-bearing chunks, "largest connected component" is
# noise, not signal.
NEAR_DUP_MIN_RESULTS = 5
# O(n^2) pairwise cosine; cap keeps it cheap even at TOP_K_MAX=200.
NEAR_DUP_MAX_CHUNKS = 50


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


def rank_files(chunks: list[dict[str, Any]], top_m: int = 3) -> list[dict[str, Any]]:
    """File score = best chunk's `_score` (max, not sum). Summing double-
    counts correlated same-file near-duplicate evidence and measurably hurt
    accuracy in testing. Order: score DESC, then first-appearance."""
    scores: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    best_rank: dict[str, int] = {}
    order: list[str] = []
    for i, c in enumerate(chunks):
        path = c.get("path", "")
        score = c.get("_score")
        score = score if score is not None else 0.0
        if path not in scores:
            scores[path] = []
            counts[path] = 0
            best_rank[path] = i + 1
            order.append(path)
        scores[path].append(score)
        counts[path] += 1

    ranked = []
    for path in order:
        ranked.append({
            "path":      path,
            "score":     max(scores[path]),
            "n_chunks":  counts[path],
            "best_rank": best_rank[path],
        })

    # Final tiebreak: input (first-appearance) order. n_chunks is payload
    # only, not a scoring factor.
    ranked.sort(key=lambda f: -f["score"])
    return ranked


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

    def detect_near_dup_wall(self, chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Detect a near-duplicate "wall": largest connected component over
        pairwise cosine >= NEAR_DUP_TAU. Observation only, never reorders or
        drops chunks. Returns None below NEAR_DUP_MIN_RESULTS vectors."""
        ids = [c["id"] for c in chunks[:NEAR_DUP_MAX_CHUNKS]]
        vectors = self._store.get_vectors_for_chunks(ids)
        ids = [i for i in ids if i in vectors]
        n = len(ids)
        if n < NEAR_DUP_MIN_RESULTS:
            return None

        mat = np.stack([vectors[i] for i in ids])
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0  # zero vector: similarity 0 to everything, avoid /0
        normed = mat / norms[:, None]
        sims = normed @ normed.T

        # Union-find over the TAU-threshold graph. n is capped at
        # NEAR_DUP_MAX_CHUNKS, so the naive O(n^2) edge scan is negligible.
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] >= NEAR_DUP_TAU:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj

        component_sizes = Counter(find(i) for i in range(n))
        wall_size = max(component_sizes.values())
        return {
            "wall_share": wall_size / n,
            "wall_size": wall_size,
            "tau": NEAR_DUP_TAU,
        }

    def format_results(self, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "_No results._"
        parts: list[str] = []
        for i, c in enumerate(chunks, 1):
            header = f"[{i}] {c['path']}:{c['start_line']}-{c['end_line']}"
            if c.get("name"):
                header += f"  ({c['name']})"
            if "_score" in c:
                # Hybrid (RRF) result: show fused score and per-branch ranks.
                rank_bits: list[str] = []
                if c.get("_rank_semantic") is not None:
                    rank_bits.append(f"sem#{c['_rank_semantic']}")
                if c.get("_rank_fts") is not None:
                    rank_bits.append(f"fts#{c['_rank_fts']}")
                ranks = f"  [{','.join(rank_bits)}]" if rank_bits else ""
                header += f"  rrf={c['_score']:.4f}{ranks}"
            elif "_blended_score" in c:
                # Folder-blend re-ranked semantic result: show blend components
                # so the consumer can see why a chunk surfaced (chunk vs folder).
                header += (
                    f"  blend={c['_blended_score']:.4f} "
                    f"[chunk={c.get('_chunk_sim', 0.0):.3f},"
                    f"folder={c.get('_folder_sim', 0.0):.3f}]"
                )
            else:
                # Explicit None check, not `distance or fts_rank`: distance
                # == 0.0 (perfect match) is falsy and would hide the score.
                if (d := c.get("distance")) is not None:
                    header += f"  score={d:.4f}"
                elif (r := c.get("fts_rank")) is not None:
                    header += f"  score={r:.4f}"
            parts.append(f"{header}\n```\n{c['content']}\n```")
        return "\n\n".join(parts)
