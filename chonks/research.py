"""research.py: deep research retrieval, returns ranked chunks for the outer
LLM to synthesize. See DOCS.md for more."""

import logging
import re
import time
from collections import defaultdict
from typing import Any

import numpy as np

from chonks.core.edges import DEFAULT_EDGE_TYPE_WEIGHTS, edge_provenance
from chonks.searcher import Searcher
from chonks.store import Store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "max_iterations":            3,
    "convergence_threshold":     0.15,
    "oversample_factor":         3,
    "max_candidates":            500,
    "top_k":                     50,
    "iteration_timeout_s":       30,
    "graph_seeds_per_iter":      20,
    "graph_neighbours_per_seed": 5,
    "hub_indegree_max":          50,
    "structural_weight":         0.5,
    "structural_seed_n":         30,
    "edge_type_weights":         DEFAULT_EDGE_TYPE_WEIGHTS,
    # Confidence gate (LARGER, arXiv:2605.16352): below this fraction of the
    # top score, a candidate can't seed graph expansion. 0.0 = off.
    "graph_seed_min_rel_score":  0.0,
    # Interleave selector, ON by default; interleave=False is an internal
    # off-switch kept for eval baselines. See _select_final_interleave for
    # what interleave_reserved_n and interleave_file_cap each do.
    "interleave_reserved_n":     40,
    "interleave_file_cap":       3,
    # Header/impl paired-file expansion, off by default pending a further
    # ablation. See _graph_expand for the companion-selection rule.
    "expand_paired_files":       False,
}

# Cap on chunks a single paired-file companion contributes to expansion.
# See expand_paired_files above.
PAIRED_CHUNKS_PER_FILE = 5

# Paired expansion / boost-donation trigger set: the top N files of the
# ranked pool by best-chunk order (not chunk-level; see _graph_expand's
# paired-file block for why).
PAIRED_TRIGGER_FILES = 20

# Regex patterns for extracting symbol names, in match order. PascalCase
# (index _PASCAL_IDX) needs extra filtering, see _extract_symbols.
_SYMBOL_PATTERNS = [
    re.compile(r'\b([A-Z][A-Za-z0-9_]+(?:::[A-Za-z0-9_]+)+)'),  # Qualified::Name
    re.compile(r'\b([A-Z][A-Za-z0-9_]{2,})\b'),                  # PascalCase (filtered)
    re.compile(r'\b([a-z][A-Za-z0-9_]{3,}(?:Manager|Handler|Controller|Service|Factory|Builder|Provider|Processor|Runner|Worker|Loader|Parser))\b'),
]
_PASCAL_IDX = 1
_MULTI_CAP_RE = re.compile(r'[A-Z][a-z0-9_]*[A-Z]')  # ≥2 uppercase letters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_symbols(chunks: list[dict]) -> list[str]:
    """Pull candidate symbol names from chunk content via regex. Single-cap
    PascalCase hits only count if already a known chunk name (filters
    docstring words like Returns/Note); multi-cap names always pass."""
    known_names = {c["name"] for c in chunks if c.get("name")}
    seen: set[str] = set()
    symbols: list[str] = []
    for chunk in chunks:
        text = (chunk.get("name") or "") + " " + chunk["content"]
        for i, pat in enumerate(_SYMBOL_PATTERNS):
            for m in pat.finditer(text):
                sym = m.group(1)
                if sym in seen:
                    continue
                if i == _PASCAL_IDX and not _MULTI_CAP_RE.match(sym) and sym not in known_names:
                    continue
                seen.add(sym)
                symbols.append(sym)
        if len(symbols) > 30:
            break
    return symbols[:20]


def _dedup(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            out.append(c)
    return out


def _rank_and_cap(chunks: list[dict], max_candidates: int) -> list[dict]:
    """Dedup, sort by score descending, then cap; order matters."""
    deduped = _dedup(chunks)
    deduped.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return deduped[:max_candidates]


def _top_score(chunks: list[dict]) -> float:
    """Average score of the top-5 chunks (already ranked)."""
    scores = [c.get("_score", 0.0) for c in chunks[:5]]
    return sum(scores) / len(scores) if scores else 0.0


def _score_candidates(
    candidates: list[dict],
    query_vec: list[float] | None,
    store: Store,
) -> None:
    """Assign a cosine `_score` in [-1, 1] to each candidate lacking one,
    recomputed from its stored int8 embedding
    (raw L2, not comparable to cosine on the same scale)."""
    pending = [c for c in candidates if "_score" not in c]
    if not pending:
        return

    q = np.asarray(query_vec, dtype=np.float32) if query_vec is not None else None
    q_norm = float(np.linalg.norm(q)) if q is not None else 0.0
    if q is None or q_norm == 0.0:
        # No query vector (embedder down): score 0 so the pool degrades to
        # insertion order instead of a poisoned ranking.
        for c in pending:
            c["_score"] = 0.0
        return

    emb = store.get_int8_embeddings_by_ids([c["id"] for c in pending])
    for c in pending:
        blob = emb.get(c["id"])
        if blob is None:
            c["_score"] = 0.0  # no stored vector: keep, but never above a hit
            continue
        v = np.frombuffer(blob, dtype=np.int8).astype(np.float32)
        denom = float(np.linalg.norm(v)) * q_norm
        c["_score"] = float(np.dot(v, q) / denom) if denom else 0.0


def _apply_structural_boost(
    candidates: list[dict],
    store: Store,
    *,
    beta: float,
    seed_n: int,
    edge_type_weights: dict[str, float] | None = None,
    expand_paired_files: bool = False,
) -> list[dict]:
    """Re-rank by query relevance plus structural proximity to top seeds:
    final = query_cosine + beta * max(edge_weight * seed_cosine) over adjacent seeds.
    Mutates `_score` in place and re-sorts; no-op when beta <= 0."""
    if not candidates or beta <= 0:
        candidates.sort(key=lambda c: c.get("_score", 0.0), reverse=True)
        return candidates

    weights = edge_type_weights if edge_type_weights is not None else DEFAULT_EDGE_TYPE_WEIGHTS

    ranked = sorted(candidates, key=lambda c: c.get("_score", 0.0), reverse=True)
    seed_score = {c["id"]: c.get("_score", 0.0) for c in ranked[:seed_n]}
    seed_ids = list(seed_score)

    # prox/donor: track proximity and which seed/edge produced it, for
    # explaining why a chunk was boosted. Pure bookkeeping, doesn't affect
    # score or order.
    prox: dict[str, float] = {}
    donor: dict[str, tuple[str, str]] = {}
    def bump(cid: str, s: float, donor_id: str, et: str) -> None:
        if s > prox.get(cid, 0.0):
            prox[cid] = s
            donor[cid] = (donor_id, et)

    # seed -> c  (c is referenced by a seed)
    for s, c, edge_type in store.get_refs_for_chunks_typed(seed_ids):
        if s in seed_score:
            bump(c, weights.get(edge_type, 1.0) * seed_score[s], s, edge_type)
    # c -> seed  (c references a seed)
    for c, s, edge_type in store.get_refs_to_chunks_typed(seed_ids):
        if s in seed_score:
            bump(c, weights.get(edge_type, 1.0) * seed_score[s], s, edge_type)
    # semantic k-NN neighbours of seeds (no edge_type, always weight 1.0)
    for s in seed_ids:
        for nid, _dist in store.get_neighbors(s):
            bump(nid, seed_score[s], s, "semantic")

    # Header/impl adjacency, weight 1.0 like semantic. Without this, paired
    # chunks are admitted but invisible to the interleave selector, since
    # _struct_boost is its only proximity signal.
    comp_score: dict[str, float] = {}
    comp_donor: dict[str, str] = {}
    if expand_paired_files:
        # File-level donors (top PAIRED_TRIGGER_FILES files by best-chunk
        # order), not the top-seed_n CHUNKS, which would miss a prominent
        # file whose best chunk sits below the chunk cutoff.
        seed_path_score: dict[str, float] = {}
        seed_path_donor: dict[str, str] = {}
        for c in ranked:
            p = c.get("path")
            if not p:
                continue
            if p not in seed_path_score:
                if len(seed_path_score) >= PAIRED_TRIGGER_FILES:
                    continue
                seed_path_score[p] = c.get("_score", 0.0)
                seed_path_donor[p] = c["id"]
        for sp, comps in store.get_paired_files(list(seed_path_score)).items():
            for cp in comps:
                if seed_path_score[sp] > comp_score.get(cp, 0.0):
                    comp_score[cp] = seed_path_score[sp]
                    comp_donor[cp] = seed_path_donor[sp]

    for c in candidates:
        # _struct_boost records raw structural proximity (pre-beta) for the
        # interleave selector; it's bookkeeping only and doesn't affect
        # _score or ordering here.
        prox_score = prox.get(c["id"], 0.0)
        path_score = comp_score.get(c.get("path"), 0.0)
        struct = max(prox_score, path_score)
        c["_struct_boost"] = struct
        c["_score"] = c.get("_score", 0.0) + beta * struct
        # Only set _struct_anchor for candidates without _graph_expand's own
        # _evidence, so the renderer can explain a bare structural bump
        # ("wired to X via calls").
        if struct > 0.0 and not c.get("_evidence"):
            if prox_score >= path_score:
                donor_id, et = donor.get(c["id"], (None, None))
            else:
                donor_id, et = comp_donor.get(c.get("path"), None), "paired"
            if donor_id is not None:
                c["_struct_anchor"] = {"anchor_id": donor_id, "edge_type": et}
    candidates.sort(key=lambda c: c["_score"], reverse=True)
    return candidates


def _graph_expand(
    store: Store,
    seed_chunks: list[dict],
    *,
    seeds_per_iter: int,
    neighbours_per_seed: int,
    path_prefix: str | None,
    hub_indegree_max: int = 0,
    edge_type_weights: dict[str, float] | None = None,
    expand_paired_files: bool = False,
) -> list[dict]:
    """Walk chunk_neighbors and chunk_refs from top candidates for coupled
    chunks. Caller must dedupe against the global pool. An edge_type
    weighted <= 0 is excluded from expansion entirely; this is a membership gate, not a score."""
    if not seed_chunks:
        return []

    weights = edge_type_weights if edge_type_weights is not None else DEFAULT_EDGE_TYPE_WEIGHTS

    seeds = seed_chunks[:seeds_per_iter]
    seed_ids = {c["id"] for c in seeds}
    expansion_ids: set[str] = set()

    # admitted_via: how each id first entered the pool. First writer wins
    # per id (deterministic iteration order), for a stable tie-break when a
    # chunk is admitted via multiple routes.
    admitted_via: dict[str, dict] = {}

    # Semantic k-NN neighbours
    for c in seeds:
        for nid, _dist in store.get_neighbors(c["id"], limit=neighbours_per_seed):
            if nid not in seed_ids:
                expansion_ids.add(nid)
                admitted_via.setdefault(nid, {"origin": "semantic", "anchor_id": c["id"]})

    # Structural reference edges (outgoing from seeds)
    for from_id, to_id, edge_type in store.get_refs_for_chunks_typed(list(seed_ids)):
        if to_id not in seed_ids and weights.get(edge_type, 1.0) > 0:
            expansion_ids.add(to_id)
            admitted_via.setdefault(
                to_id, {"origin": "graph", "anchor_id": from_id, "edge_type": edge_type},
            )

    # Placed before the hub filter below so paired hits get the same
    # hub-indegree filtering as any other expansion target. Trigger is
    # FILE-level, not chunk seeds; see PAIRED_TRIGGER_FILES.
    if expand_paired_files:
        seed_names_by_path: dict[str, set[str]] = defaultdict(set)
        seed_paths: list[str] = []
        seed_id_by_path: dict[str, str] = {}
        seen_paths: set[str] = set()
        for c in seed_chunks:
            path = c.get("path")
            if not path:
                continue
            if path not in seen_paths:
                if len(seen_paths) >= PAIRED_TRIGGER_FILES:
                    continue
                seen_paths.add(path)
                seed_paths.append(path)
            seed_id_by_path.setdefault(path, c["id"])
            name = c.get("name")
            if name:
                seed_names_by_path[path].add(name)

        if seed_paths:
            companions = store.get_paired_files(seed_paths)
            for seed_path, companion_paths in companions.items():
                seed_names = list(seed_names_by_path.get(seed_path, ()))
                anchor_id = seed_id_by_path.get(seed_path)
                for companion_path in companion_paths:
                    name_hits = (
                        store.get_chunks_by_path_and_names(
                            companion_path, seed_names, limit=PAIRED_CHUNKS_PER_FILE,
                        )
                        if seed_names else []
                    )
                    if name_hits:
                        for hit in name_hits:
                            if hit["id"] not in seed_ids:
                                expansion_ids.add(hit["id"])
                                admitted_via.setdefault(
                                    hit["id"],
                                    {"origin": "paired", "anchor_id": anchor_id, "edge_type": "paired"},
                                )
                    else:
                        top = store.get_top_pagerank_chunk_for_path(companion_path)
                        if top is not None and top["id"] not in seed_ids:
                            expansion_ids.add(top["id"])
                            admitted_via.setdefault(
                                top["id"],
                                {"origin": "paired", "anchor_id": anchor_id, "edge_type": "paired"},
                            )

    if not expansion_ids:
        return []

    # Drop expansion targets with high in-degree (base classes, autogen
    # binding roots): referenced by everything, so pulling them in floods
    # the pool with noise. Seeds are never filtered.
    if hub_indegree_max:
        indeg = store.ref_indegrees(list(expansion_ids))
        expansion_ids = {i for i in expansion_ids if indeg.get(i, 0) <= hub_indegree_max}
        if not expansion_ids:
            return []

    new_chunks = store.get_chunks_by_ids(list(expansion_ids))
    for c in new_chunks:
        c["_evidence"] = admitted_via.get(c["id"])
    if path_prefix:
        # Normalize like Store.search_semantic (strip trailing slash) so
        # scope matches; a raw startswith on "foo/" would miss files
        # search_semantic's LIKE 'foo%' matched, silently dropping in-scope chunks.
        normalized_prefix = path_prefix.rstrip("/\\")
        new_chunks = [c for c in new_chunks if c["path"].startswith(normalized_prefix)]
    return new_chunks


def _select_final_interleave(
    candidates: list[dict],
    *,
    top_k: int,
    reserved_n: int,
    file_cap: int,
) -> list[dict]:
    """Replaces `candidates[:top_k]` truncation with two anti-burial levers
    (file_cap, reserved_n). Never displaces or reorders a file already
    selected; rescues only spend slack, they don't cost a found file."""
    if not candidates:
        return []

    # Blended stream with the per-file cap applied.
    capped: list[dict] = []
    file_counts: dict[str, int] = {}
    for c in candidates:
        path = c.get("path", "")
        if file_cap and file_counts.get(path, 0) >= file_cap:
            continue
        capped.append(c)
        file_counts[path] = file_counts.get(path, 0) + 1
    natural = capped[:top_k]
    natural_ids = {c["id"] for c in natural}

    # Split the capped window into first-occurrence chunks (each sets a
    # file's rank, so protect these, in order) and redundant duplicate-file
    # chunks (sacrificable without changing any file rank).
    seen_paths: set[str] = set()
    firsts: list[dict] = []
    redundant: list[dict] = []
    for c in natural:
        path = c.get("path", "")
        if path in seen_paths:
            redundant.append(c)
        else:
            seen_paths.add(path)
            firsts.append(c)

    # Reserved rescues: top structurally-adjacent candidates the capped window
    # missed AND whose file is not already represented; only a NEW file earns a
    # rescue slot (a duplicate-file rescue would not move any file rank).
    reserved = sorted(
        (c for c in candidates
         if c.get("_struct_boost", 0.0) > 0.0
         and c["id"] not in natural_ids
         and c.get("path", "") not in seen_paths),
        key=lambda c: c["_struct_boost"],
        reverse=True,
    )
    rescues: list[dict] = []
    rescue_paths: set[str] = set()
    for c in reserved:
        if len(rescues) >= max(0, reserved_n):
            break
        p = c.get("path", "")
        if p in rescue_paths:
            continue  # one rescue slot per new file
        rescue_paths.add(p)
        rescues.append(c)

    if not rescues:
        return natural  # levers changed nothing, hand back the capped order

    # Keep every first-occurrence file in full (never demoted or dropped),
    # then fill remaining slack with rescues, then redundant chunks.
    final = firsts[:top_k]
    if len(final) < top_k:
        final += rescues[: top_k - len(final)]
    if len(final) < top_k:
        final += redundant[: top_k - len(final)]
    return final[:top_k]


def _result_connections(store: Store, ids: list[str], cap: int = 40) -> list[dict]:
    """Typed edges among `ids` where both endpoints are in `ids`; the graph
    within what's returned, not the full expansion frontier. Deduped and
    capped at `cap`, order deterministic."""
    if not ids:
        return []
    id_set = set(ids)
    rows = sorted(
        (row for row in store.get_refs_for_chunks_typed(ids) if row[1] in id_set),
        key=lambda r: (r[0], r[1], r[2]),
    )
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for from_id, to_id, edge_type in rows:
        key = (from_id, to_id)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "from_id":    from_id,
            "to_id":      to_id,
            "edge_type":  edge_type,
            "provenance": edge_provenance(edge_type),
        })
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
# Main research function
# ---------------------------------------------------------------------------

def deep_research(
    query: str,
    searcher: Searcher,
    cfg: dict | None = None,
    path_prefix: str | None = None,
    interleave: bool = True,
) -> dict[str, Any]:
    """Iterative candidate collection; returns ranked chunks for the outer LLM
    to synthesize. `degraded="semantic_unavailable"` means expansion scoring
    is a neutral fallback, not real relevance; None on the healthy path."""
    cfg = {**_DEFAULTS, **(cfg or {})}
    top_k                = cfg["top_k"]
    oversample           = cfg["oversample_factor"]
    max_iters            = cfg["max_iterations"]
    graph_seed_min_rel   = cfg["graph_seed_min_rel_score"]
    conv_threshold       = cfg["convergence_threshold"]
    max_candidates       = cfg["max_candidates"]
    graph_seeds_per_iter = cfg["graph_seeds_per_iter"]
    graph_neighbours     = cfg["graph_neighbours_per_seed"]
    hub_indegree_max     = cfg["hub_indegree_max"]
    edge_type_weights    = cfg["edge_type_weights"]
    expand_paired_files  = cfg["expand_paired_files"]

    store = searcher.store

    # Embed the query once up front and reuse it across every iteration's
    # scoring pass instead of re-embedding each time.
    try:
        query_vec: list[float] | None = searcher.embed_query(query)
    except Exception as e:
        logger.warning("query embedding failed; expansion scoring degraded: %s", e)
        query_vec = None
    # query_vec None means embedder is down; mark degraded so callers know.
    degraded: str | None = "semantic_unavailable" if query_vec is None else None

    all_candidates: list[dict] = []
    prev_score = 0.0
    iterations = 0

    for iteration in range(max_iters):
        iterations += 1
        iter_start = time.monotonic()

        logger.debug("iteration %d/%d", iteration + 1, max_iters)

        # searcher.semantic manages its own httpx client when none is passed;
        # we let it own the lifecycle rather than hold a client open across
        # iterations that never use it.
        if iteration == 0:
            fetch_k = top_k * oversample
            try:
                hits = searcher.semantic(query, fetch_k, path_prefix)
                for h in hits:
                    h["_origin"]    = "seed"
                    h["_iteration"] = 0
                all_candidates.extend(hits)
            except Exception as e:
                logger.warning("semantic seed failed: %s", e)

        _score_candidates(all_candidates, query_vec, store)
        all_candidates = _rank_and_cap(all_candidates, max_candidates)

        # Gate only removes low scorers from the EXPANSION SEED set; they
        # stay in the pool and can still be returned, just can't pull in
        # their structural neighbourhood. Relative to top score, not absolute.
        expand_seeds = all_candidates
        if graph_seed_min_rel > 0 and all_candidates:
            top_score = all_candidates[0].get("_score", 0.0)
            if top_score > 0:
                floor = graph_seed_min_rel * top_score
                expand_seeds = [c for c in all_candidates
                                if c.get("_score", 0.0) >= floor]

        try:
            graph_hits = _graph_expand(
                store,
                expand_seeds,
                seeds_per_iter=graph_seeds_per_iter,
                neighbours_per_seed=graph_neighbours,
                path_prefix=path_prefix,
                hub_indegree_max=hub_indegree_max,
                edge_type_weights=edge_type_weights,
                expand_paired_files=expand_paired_files,
            )
            for h in graph_hits:
                # No _score here, _score_candidates ranks it by query cosine.
                h.setdefault("_origin",    "graph")
                h.setdefault("_iteration", iteration)
            all_candidates.extend(graph_hits)
            logger.debug("graph expansion added %d candidates", len(graph_hits))
        except Exception as e:
            logger.warning("graph expansion failed: %s", e)

        symbols = _extract_symbols(all_candidates[:30])
        logger.debug("extracted %d symbols for regex expansion", len(symbols))

        # Complementary to graph traversal: catches symbol mentions in
        # non-boundary chunks (graph only links named chunks) and
        # freshly-added files whose neighbours haven't been rebuilt yet.
        for sym in symbols[:10]:
            try:
                hits = searcher.regex(sym, top_k=10, path_prefix=path_prefix)
                for h in hits:
                    h.setdefault("_origin",    "regex")
                    h.setdefault("_iteration", iteration)
                all_candidates.extend(hits)
            except Exception as e:
                logger.warning("regex expansion failed for %r: %s", sym, e)

        _score_candidates(all_candidates, query_vec, store)
        all_candidates = _rank_and_cap(all_candidates, max_candidates)

        # Convergence: stop if top-5 average score stops improving.
        current_score = _top_score(all_candidates)
        delta = current_score - prev_score
        logger.debug("score=%.4f  delta=%.4f", current_score, delta)

        if iteration > 0 and delta < conv_threshold:
            logger.debug("converged")
            break

        prev_score = current_score

        elapsed = time.monotonic() - iter_start
        if elapsed > cfg["iteration_timeout_s"]:
            logger.warning("iteration %d timed out after %.1fs", iteration + 1, elapsed)
            break

    # Final re-rank: blend query relevance with structural proximity to the
    # top seeds. See _apply_structural_boost for why.
    all_candidates = _apply_structural_boost(
        all_candidates, store,
        beta=cfg["structural_weight"], seed_n=cfg["structural_seed_n"],
        edge_type_weights=edge_type_weights,
        expand_paired_files=expand_paired_files,
    )
    # interleave=False is the eval baseline / escape hatch: plain
    # blended-score truncation instead of the anti-burial selector.
    if interleave:
        final_chunks = _select_final_interleave(
            all_candidates,
            top_k=top_k,
            reserved_n=cfg["interleave_reserved_n"],
            file_cap=cfg["interleave_file_cap"],
        )
    else:
        final_chunks = all_candidates[:top_k]

    connections = _result_connections(store, [c["id"] for c in final_chunks])

    return {
        "chunks":      final_chunks,
        "count":       len(final_chunks),
        "iterations":  iterations,
        "connections": connections,
        "degraded":    degraded,
    }
