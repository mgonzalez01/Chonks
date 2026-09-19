"""trace_path: bidirectional BFS between two symbols over chunk_refs (and,
optionally, chunk_neighbors as a semantic fallback)."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from chonks.core.edges import edge_provenance

if TYPE_CHECKING:
    from chonks.store import Store

_ASSOCIATED_RANK = 1  # typed edges (calls/imports/inherits/xlang) rank 0
_MENTIONS_RANK = 2   # plain mentions ranks below associated
_SEMANTIC_RANK = 3   # semantic k-NN edges are the least preferred, fallback-only


def _edge_rank(edge_type: str) -> int:
    if edge_type == "mentions":
        return _MENTIONS_RANK
    if edge_type == "associated":
        return _ASSOCIATED_RANK
    return 0


def _adjacency(
    store: "Store",
    frontier_ids: list[str],
    max_fanout: int,
    include_semantic: bool,
) -> dict[str, list[str]]:
    """Batched BFS-step neighbor lookup, fanout-capped and prioritizing typed
    edges over associated/mentions over semantic k-NN (see _edge_rank).
    Direction is re-derived later; here only "is X reachable from Y" matters."""
    frontier_set = set(frontier_ids)
    per_node: dict[str, dict[str, int]] = defaultdict(dict)

    def _consider(node: str, neighbor: str, rank: int) -> None:
        if neighbor == node:
            return
        best = per_node[node]
        if neighbor not in best or rank < best[neighbor]:
            best[neighbor] = rank

    for from_id, to_id, edge_type in store.get_refs_for_chunks_typed(frontier_ids):
        if from_id in frontier_set:
            _consider(from_id, to_id, _edge_rank(edge_type))
    for from_id, to_id, edge_type in store.get_refs_to_chunks_typed(frontier_ids):
        if to_id in frontier_set:
            _consider(to_id, from_id, _edge_rank(edge_type))

    if include_semantic:
        for chunk_id, neighbor_id, _distance in store.get_neighbor_edges_touching(frontier_ids):
            if chunk_id in frontier_set:
                _consider(chunk_id, neighbor_id, _SEMANTIC_RANK)
            if neighbor_id in frontier_set:
                _consider(neighbor_id, chunk_id, _SEMANTIC_RANK)

    result: dict[str, list[str]] = {}
    for node, candidates in per_node.items():
        ordered = sorted(candidates.items(), key=lambda kv: kv[1])[:max_fanout]
        result[node] = [nb for nb, _rank in ordered]
    return result


def _bidirectional_bfs(
    store: "Store",
    from_ids: list[str],
    to_ids: list[str],
    max_depth: int,
    max_fanout: int,
    include_semantic: bool,
) -> list[str] | None:
    """Bidirectional BFS over chunk_refs (+ chunk_neighbors when
    include_semantic). Only parent pointers are tracked; hop direction and
    edge_type are re-derived afterward for just the winning path's small hop count."""
    from_set, to_set = set(from_ids), set(to_ids)
    common = from_set & to_set
    if common:
        return [sorted(common)[0]]  # same chunk defines both names, 0-hop path

    visited_fwd: dict[str, str | None] = {i: None for i in from_ids}
    visited_bwd: dict[str, str | None] = {i: None for i in to_ids}
    frontier_fwd = set(from_ids)
    frontier_bwd = set(to_ids)

    depth = 0
    meeting: str | None = None
    while frontier_fwd and frontier_bwd and depth < max_depth:
        if len(frontier_fwd) <= len(frontier_bwd):
            adj = _adjacency(store, list(frontier_fwd), max_fanout, include_semantic)
            next_frontier: set[str] = set()
            for node in frontier_fwd:
                for nb in adj.get(node, ()):
                    if nb not in visited_fwd:
                        visited_fwd[nb] = node
                        next_frontier.add(nb)
                    if nb in visited_bwd and meeting is None:
                        meeting = nb
            frontier_fwd = next_frontier
        else:
            adj = _adjacency(store, list(frontier_bwd), max_fanout, include_semantic)
            next_frontier = set()
            for node in frontier_bwd:
                for nb in adj.get(node, ()):
                    if nb not in visited_bwd:
                        visited_bwd[nb] = node
                        next_frontier.add(nb)
                    if nb in visited_fwd and meeting is None:
                        meeting = nb
            frontier_bwd = next_frontier
        depth += 1
        if meeting is not None:
            break

    if meeting is None:
        return None

    # Walk the forward tree's parent pointers from the meeting node back to a
    # from_ids root, giving source -> ... -> meeting.
    fwd_chain: list[str] = []
    node: str | None = meeting
    while node is not None:
        fwd_chain.append(node)
        node = visited_fwd.get(node)
    fwd_chain.reverse()

    # Walk the backward tree's parent pointers, which point toward the
    # to_ids root, giving (node after meeting) -> ... -> target.
    bwd_chain: list[str] = []
    node = visited_bwd.get(meeting)
    while node is not None:
        bwd_chain.append(node)
        node = visited_bwd.get(node)

    return fwd_chain + bwd_chain


def _hop_direction(store: "Store", u: str, v: str) -> tuple[str, str]:
    """Resolves (edge_type, direction) for display, checked structural-then-
    semantic to match the traversal's own preference. 'backward' means the
    path walks a chunk_refs edge against its arrow."""
    best_type: str | None = None
    for from_id, to_id, edge_type in store.get_refs_for_chunks_typed([u]):
        if from_id == u and to_id == v:
            if best_type is None or _edge_rank(edge_type) < _edge_rank(best_type):
                best_type = edge_type
    if best_type is not None:
        return best_type, "forward"

    for from_id, to_id, edge_type in store.get_refs_for_chunks_typed([v]):
        if from_id == v and to_id == u:
            if best_type is None or _edge_rank(edge_type) < _edge_rank(best_type):
                best_type = edge_type
    if best_type is not None:
        return best_type, "backward"

    return "semantic", "semantic"


def trace_path(
    store: "Store",
    from_symbol: str,
    to_symbol: str,
    *,
    max_depth: int = 6,
    max_fanout: int = 25,
    include_semantic: bool = False,
) -> dict:
    """Shortest path from `from_symbol` to `to_symbol` over chunk_refs and,
    as an include_semantic=True fallback only, chunk_neighbors. Returns
    {"found", "hops": [...], "used_semantic"} or {"found": False, "error"} (and
    "reason": "unknown_symbol" when a symbol does not exist)."""
    from_ids = store.resolve_symbol_chunk_ids(from_symbol)
    to_ids = store.resolve_symbol_chunk_ids(to_symbol)

    if not from_ids:
        return {"found": False, "error": f"Unknown symbol: {from_symbol!r}", "reason": "unknown_symbol", "used_semantic": False}
    if not to_ids:
        return {"found": False, "error": f"Unknown symbol: {to_symbol!r}", "reason": "unknown_symbol", "used_semantic": False}

    path = _bidirectional_bfs(store, from_ids, to_ids, max_depth, max_fanout, include_semantic=False)
    used_semantic = False
    if path is None and include_semantic:
        path = _bidirectional_bfs(store, from_ids, to_ids, max_depth, max_fanout, include_semantic=True)
        used_semantic = True

    if path is None:
        return {
            "found": False,
            "error": f"No path found from {from_symbol!r} to {to_symbol!r} within max_depth={max_depth}"
                     + ("" if include_semantic else " (structural edges only; retry with include_semantic=True)"),
            "used_semantic": include_semantic,
        }

    meta = {c["id"]: c for c in store.get_chunks_by_ids(path)}
    hops = []
    for u, v in zip(path, path[1:]):
        edge_type, direction = _hop_direction(store, u, v)
        cu, cv = meta.get(u, {"id": u}), meta.get(v, {"id": v})

        def _display(c: dict) -> dict:
            return {
                "chunk_id":   c.get("id"),
                "path":       c.get("path"),
                "name":       c.get("name"),
                "chunk_type": c.get("chunk_type"),
                "start_line": c.get("start_line"),
                "end_line":   c.get("end_line"),
            }

        hops.append({
            "from_chunk": _display(cu),
            "to_chunk":   _display(cv),
            "edge_type":  edge_type,
            "direction":  direction,
            "provenance": edge_provenance(edge_type),
        })

    return {"found": True, "hops": hops, "depth": len(hops), "used_semantic": used_semantic}
