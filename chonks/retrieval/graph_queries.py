"""Shaped graph answers: usages, outgoing references and impact, with their note text."""

import re
from typing import Any

from chonks.core.batching import batched
from chonks.core.edges import _COLLAPSE_RANK, _HUB_EDGE_TYPES, _MAX_CROSS_LANG_OCCURRENCES, edge_provenance

# Unscoped get_hubs scans all of chunk_refs under the store lock, which can
# block every other request for minutes on a huge corpus. Above this size
# the global branch requires a path_prefix instead.
_HUBS_GLOBAL_MAX_CHUNKS = 100_000


def _provenance_rollup(edge_types: dict[str, int]) -> dict[str, int]:
    """Roll up {edge_type: count} into {provenance: count}; shared by
    get_hubs' precomputed and live paths so both use the same rollup."""
    out: dict[str, int] = {}
    for et, n in edge_types.items():
        prov = edge_provenance(et)
        out[prov] = out.get(prov, 0) + n
    return out


def _symbol_miss_note(store, name: str) -> str:
    """Diagnostic for a resolve_symbol_chunk_ids miss. A qualified query
    gets no suffix fallback, so this suggests the bare last component
    only when that bare name actually resolves to something."""
    if "::" in name or "." in name:
        bare = re.split(r"::|\.", name)[-1]
        if bare and bare != name and store.resolve_symbol_chunk_ids(bare):
            return f"symbol not found — try the bare name {bare!r}"
    return "symbol not found"


def _fts_scan_for_name(store, name: str, path_prefix: str | None,
                       limit: int | None) -> list[dict[str, Any]]:
    """FTS content-scan fallback for find_usages/get_impact when a name
    exceeds _MAX_CROSS_LANG_OCCURRENCES. A pre-filter, not a proven
    reference; callers must label these as content matches, not edges."""
    phrase = '"' + name.replace('"', '""') + '"'
    rows = store.search_fts(phrase, top_k=limit or 50, path_prefix=path_prefix)
    return [
        {
            "chunk_id": r["id"], "path": r["path"], "name": r.get("name"),
            "chunk_type": r.get("chunk_type"), "start_line": r["start_line"],
            "end_line": r["end_line"], "origin": "fts_scan",
        }
        for r in rows
    ]


def find_usages(store, name: str, path_prefix: str | None = None,
                limit: int | None = None) -> dict[str, Any]:
    """Who references `name`. Empty `results` doesn't mean "no callers":
    see `note` for the unresolved-name and above-cap cases. `limit`
    truncates AFTER sorting by edge quality, not alphabetically."""
    chunk_ids = store.resolve_symbol_chunk_ids(name)
    if not chunk_ids:
        return {"results": [], "note": _symbol_miss_note(store, name), "content_matches": []}

    edges = store.get_refs_to_chunks_typed(chunk_ids)
    from_ids = sorted({from_id for from_id, _to_id, _et in edges})
    if not from_ids:
        if len(chunk_ids) > _MAX_CROSS_LANG_OCCURRENCES:
            note = (
                f"{name!r} has {len(chunk_ids)} definers, above the "
                f"edge-indexing cap ({_MAX_CROSS_LANG_OCCURRENCES}) — reference "
                "edges are not indexed for this name; falling back to an FTS content scan"
            )
            content_matches = _fts_scan_for_name(store, name, path_prefix, limit)
            return {"results": [], "note": note, "content_matches": content_matches}
        return {"results": [], "note": None, "content_matches": []}

    # Collapse to the least-uncertain edge type when a chunk has more
    # than one into the resolution set (see _COLLAPSE_RANK).
    edge_type_by_from: dict[str, str] = {}
    for from_id, _to_id, et in edges:
        cur = edge_type_by_from.get(from_id)
        if cur is None or (
            (_COLLAPSE_RANK.get(et, 3), et) < (_COLLAPSE_RANK.get(cur, 3), cur)
        ):
            edge_type_by_from[from_id] = et

    results: list[dict[str, Any]] = []
    with store._lock:
        for batch in batched(from_ids, 900):
            placeholders = ",".join("?" * len(batch))
            sql = (f"SELECT id AS chunk_id, path, name, chunk_type, start_line, end_line "
                   f"FROM chunks WHERE id IN ({placeholders})")
            args: list = list(batch)
            if path_prefix:
                sql += " AND path LIKE ? ESCAPE '\\'"
                args.append(store._like_escape(path_prefix.rstrip("/\\")) + "%")
            results.extend(dict(r) for r in store._conn.execute(sql, args).fetchall())
    for r in results:
        et = edge_type_by_from.get(r["chunk_id"], "mentions")
        r["edge_type"] = et
        r["provenance"] = edge_provenance(et)
    results.sort(key=lambda r: (_COLLAPSE_RANK.get(r["edge_type"], 3), r["path"], r["start_line"]))
    note = None
    if limit and len(results) > limit:
        omitted = results[limit:]
        omitted_xlang = sum(1 for r in omitted if r["edge_type"] == "xlang")
        omitted_associated = sum(1 for r in omitted if r["edge_type"] == "associated")
        omitted_mentions = sum(1 for r in omitted if r["edge_type"] == "mentions")
        omitted_typed = len(omitted) - omitted_xlang - omitted_associated - omitted_mentions
        note = (
            f"limit={limit} truncated {len(omitted)} result(s): "
            f"{omitted_typed} typed, {omitted_xlang} xlang, "
            f"{omitted_associated} associated, {omitted_mentions} mentions"
        )
        results = results[:limit]
    return {"results": results, "note": note, "content_matches": []}


def find_outgoing(store, name: str, path_prefix: str | None = None,
                  limit: int | None = None) -> dict[str, Any]:
    """Forward twin of find_usages (self-refs among definers excluded).
    Zero outgoing edges is a valid empty result. `limit` truncates
    AFTER sorting by edge quality, not alphabetically."""
    chunk_ids = store.resolve_symbol_chunk_ids(name)
    if not chunk_ids:
        return {"results": [], "note": _symbol_miss_note(store, name)}

    definer_set = set(chunk_ids)
    edges = store.get_refs_from_chunks_typed(chunk_ids)
    to_ids = sorted({to_id for _from_id, to_id, _et in edges if to_id not in definer_set})
    if not to_ids:
        return {"results": [], "note": None}

    edge_type_by_to: dict[str, str] = {}
    for _from_id, to_id, et in edges:
        if to_id in definer_set:
            continue
        cur = edge_type_by_to.get(to_id)
        if cur is None or (
            (_COLLAPSE_RANK.get(et, 3), et) < (_COLLAPSE_RANK.get(cur, 3), cur)
        ):
            edge_type_by_to[to_id] = et

    results: list[dict[str, Any]] = []
    with store._lock:
        for batch in batched(to_ids, 900):
            placeholders = ",".join("?" * len(batch))
            sql = (f"SELECT id AS chunk_id, path, name, chunk_type, start_line, end_line "
                   f"FROM chunks WHERE id IN ({placeholders})")
            args: list = list(batch)
            if path_prefix:
                sql += " AND path LIKE ? ESCAPE '\\'"
                args.append(store._like_escape(path_prefix.rstrip("/\\")) + "%")
            results.extend(dict(r) for r in store._conn.execute(sql, args).fetchall())
    for r in results:
        et = edge_type_by_to.get(r["chunk_id"], "mentions")
        r["edge_type"] = et
        r["provenance"] = edge_provenance(et)
    results.sort(key=lambda r: (_COLLAPSE_RANK.get(r["edge_type"], 3), r["path"], r["start_line"]))
    if limit:
        results = results[:limit]
    return {"results": results, "note": None}


def get_impact(store, name: str, path_prefix: str | None = None,
               limit: int = 20, rank_by: str = "pagerank_sum") -> dict[str, Any]:
    """Blast radius of `name`, aggregated by referencing file, with a
    `note` matching find_usages' diagnostics. `rank_by="pagerank_sum"`
    degrades cleanly to count-DESC when chunk_pagerank is empty."""
    if rank_by not in ("pagerank_sum", "count"):
        raise ValueError(
            f"invalid rank_by {rank_by!r} — must be 'pagerank_sum' or 'count'"
        )
    def_chunk_ids = store.resolve_symbol_chunk_ids(name)
    if not def_chunk_ids:
        return {
            "symbol": name, "definitions": [], "total_references": 0,
            "by_edge_type": {}, "by_provenance": {}, "rank_by": rank_by,
            "files": [], "files_total": 0,
            "note": _symbol_miss_note(store, name),
        }

    definitions: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    with store._lock:
        for batch in batched(def_chunk_ids, 900):
            placeholders = ",".join("?" * len(batch))
            rows = store._conn.execute(
                f"SELECT path, name, chunk_type FROM chunks WHERE id IN ({placeholders})",
                batch,
            ).fetchall()
            definitions.extend(dict(r) for r in rows)

            sql = (
                "SELECT cr.from_id AS chunk_id, cr.edge_type AS edge_type, "
                "c.path AS path, c.name AS name, c.chunk_type AS chunk_type, "
                "c.start_line AS start_line, COALESCE(pr.score, 0.0) AS pagerank "
                "FROM chunk_refs cr "
                "JOIN chunks c ON c.id = cr.from_id "
                "LEFT JOIN chunk_pagerank pr ON pr.chunk_id = cr.from_id "
                f"WHERE cr.to_id IN ({placeholders})"
            )
            args: list = list(batch)
            if path_prefix:
                sql += " AND c.path LIKE ? ESCAPE '\\'"
                args.append(store._like_escape(path_prefix.rstrip("/\\")) + "%")
            edges.extend(dict(r) for r in store._conn.execute(sql, args).fetchall())

    definitions.sort(key=lambda d: (d["path"], d["name"] or ""))

    by_edge_type: dict[str, int] = {}
    for e in edges:
        by_edge_type[e["edge_type"]] = by_edge_type.get(e["edge_type"], 0) + 1

    by_provenance: dict[str, int] = {}
    for et, n in by_edge_type.items():
        prov = edge_provenance(et)
        by_provenance[prov] = by_provenance.get(prov, 0) + n

    files: dict[str, dict[str, Any]] = {}
    for e in edges:
        f = files.setdefault(e["path"], {
            "path": e["path"], "count": 0, "edge_types": {}, "_chunks": {},
        })
        f["count"] += 1
        f["edge_types"][e["edge_type"]] = f["edge_types"].get(e["edge_type"], 0) + 1
        f["_chunks"][e["chunk_id"]] = (e["name"], e["chunk_type"], e["start_line"], e["pagerank"])

    file_list: list[dict[str, Any]] = []
    for f in files.values():
        chunks = f.pop("_chunks")
        f["pagerank_sum"] = sum(v[3] for v in chunks.values())
        referrers = sorted(chunks.values(), key=lambda v: (-v[3], v[2]))[:3]
        f["top_referrers"] = [
            {"name": n, "chunk_type": ct, "start_line": sl}
            for n, ct, sl, _pr in referrers
        ]
        file_list.append(f)

    if rank_by == "count":
        file_list.sort(key=lambda f: (-f["count"], -f["pagerank_sum"], f["path"]))
    else:
        file_list.sort(key=lambda f: (-f["pagerank_sum"], -f["count"], f["path"]))
    files_total = len(file_list)

    note = None
    if not edges and len(def_chunk_ids) > _MAX_CROSS_LANG_OCCURRENCES:
        note = (
            f"{name!r} has {len(def_chunk_ids)} definers, above the "
            f"edge-indexing cap ({_MAX_CROSS_LANG_OCCURRENCES}) — reference "
            "edges are not indexed for this name; try find_usages, which falls "
            "back to an FTS content scan for this case"
        )

    return {
        "symbol": name,
        "definitions": definitions,
        "total_references": len(edges),
        "by_edge_type": by_edge_type,
        "by_provenance": by_provenance,
        "rank_by": rank_by,
        "files": file_list[:limit],
        "files_total": files_total,
        "note": note,
    }


def get_hubs(store, path_prefix: str | None = None, limit: int = 20,
             edge_types: list[str] | None = None) -> dict[str, Any]:
    """Named chunks ranked by in-degree, then PageRank, then path for
    determinism. Dispatches on chunk_indegree: precomputed when
    possible, else a live fallback gated by _get_hubs_live's guard."""
    # [] normalizes to None: the live path's `is not None` check would
    # otherwise filter every type out on an empty list.
    edge_types_set: set[str] | None = None
    if edge_types:
        edge_types_set = set(edge_types)
        invalid = edge_types_set - _HUB_EDGE_TYPES
        if invalid:
            raise ValueError(
                f"invalid edge_types {sorted(invalid)!r} — valid types are "
                f"{sorted(_HUB_EDGE_TYPES)}"
            )

    if store.has_indegree():
        return _get_hubs_precomputed(store, path_prefix, limit, edge_types_set)
    return _get_hubs_live(store, path_prefix, limit, edge_types_set)


def _get_hubs_precomputed(
    store, path_prefix: str | None, limit: int, edge_types_set: set[str] | None,
) -> dict[str, Any]:
    """get_hubs via chunk_indegree: one aggregate query covers both
    scoped and unscoped cases, since this table stays small regardless
    of corpus size (unlike chunk_refs)."""
    rows = store.hub_indegree_rows(path_prefix, limit, edge_types_set)
    if not rows:
        return {"hubs": []}

    ids = [r["id"] for r in rows]
    breakdown: dict[str, dict[str, int]] = {}
    with store._lock:
        for batch in batched(ids, 900):
            ph = ",".join("?" * len(batch))
            type_clauses = [f"chunk_id IN ({ph})"]
            type_params: list[Any] = list(batch)
            if edge_types_set:
                ph2 = ",".join("?" * len(edge_types_set))
                type_clauses.append(f"edge_type IN ({ph2})")
                type_params.extend(sorted(edge_types_set))
            for r in store._conn.execute(
                f"SELECT chunk_id, edge_type, n FROM chunk_indegree WHERE "
                f"{' AND '.join(type_clauses)}",
                type_params,
            ):
                breakdown.setdefault(r["chunk_id"], {})[r["edge_type"]] = r["n"]

    return {"hubs": [{
        "path": r["path"],
        "name": r["name"],
        "chunk_type": r["chunk_type"],
        "start_line": r["start_line"],
        "in_degree": r["in_degree"],
        "pagerank": r["pagerank"],
        "edge_types": breakdown.get(r["id"], {}),
        "by_provenance": _provenance_rollup(breakdown.get(r["id"], {})),
    } for r in rows]}


def _get_hubs_live(
    store, path_prefix: str | None, limit: int, edge_types_set: set[str] | None,
) -> dict[str, Any]:
    """Live fallback for get_hubs when chunk_indegree is empty (old DB).
    edge_types_set is filtered in Python, not pushed into SQL."""
    if path_prefix:
        # RANGE, not LIKE: chunks is wide, so an unindexed LIKE scan
        # reads the whole corpus off disk. U+10FFFF bounds the range
        # to exactly the lo-prefixed paths.
        lo = path_prefix.rstrip("/\\")
        with store._lock:
            candidates = [dict(r) for r in store._conn.execute(
                "SELECT id, path, name, chunk_type, start_line FROM chunks"
                " WHERE name IS NOT NULL AND path >= ? AND path < ?",
                (lo, lo + "\U0010FFFF"),
            ).fetchall()]
        if not candidates:
            return {"hubs": []}
        ids = [c["id"] for c in candidates]
        edge_counts = store._indegree_by_type(ids)
        pagerank = store.get_pagerank_for_chunks(ids)
        hubs: list[dict[str, Any]] = []
        for c in candidates:
            types = edge_counts.get(c["id"])
            if not types:
                continue
            if edge_types_set is not None:
                types = {t: n for t, n in types.items() if t in edge_types_set}
                if not types:
                    continue
            hubs.append({
                "path": c["path"],
                "name": c["name"],
                "chunk_type": c["chunk_type"],
                "start_line": c["start_line"],
                "in_degree": sum(types.values()),
                "pagerank": pagerank.get(c["id"], 0.0),
                "edge_types": types,
                "by_provenance": _provenance_rollup(types),
            })
        hubs.sort(key=lambda h: (-h["in_degree"], -h["pagerank"], h["path"], h["start_line"]))
        return {"hubs": hubs[:limit]}

    # Whole-corpus branch. Guarded by _HUBS_GLOBAL_MAX_CHUNKS: the
    # GROUP BY over chunk_refs holds the store lock and can take minutes
    # on a huge corpus, blocking every other request.
    if store.count_chunks() > _HUBS_GLOBAL_MAX_CHUNKS:
        raise ValueError(
            "global /hubs on a corpus over "
            f"{_HUBS_GLOBAL_MAX_CHUNKS} chunks requires precomputed "
            "in-degree, and this DB doesn't have it yet (chunk_indegree "
            "is empty) — re-index (or run the graph rebuild) to populate "
            "it, or pass a path_prefix (or @subsystem) to scope the "
            "request in the meantime"
        )
    rows = store.hub_edge_type_rows()
    by_id: dict[str, dict[str, int]] = {}
    for r in rows:
        if edge_types_set is not None and r["edge_type"] not in edge_types_set:
            continue
        by_id.setdefault(r["to_id"], {})[r["edge_type"]] = r["n"]
    by_degree: dict[int, list[str]] = {}
    for cid, types in by_id.items():
        by_degree.setdefault(sum(types.values()), []).append(cid)

    hubs = []
    for degree in sorted(by_degree, reverse=True):
        ids = by_degree[degree]
        meta: dict[str, dict[str, Any]] = {}
        with store._lock:
            for batch in batched(ids, 900):
                ph = ",".join("?" * len(batch))
                for r in store._conn.execute(
                    "SELECT id, path, name, chunk_type, start_line FROM chunks"
                    f" WHERE name IS NOT NULL AND id IN ({ph})", batch,
                ):
                    meta[r["id"]] = dict(r)
        pagerank = store.get_pagerank_for_chunks(list(meta))
        group = [{
            "path": meta[cid]["path"],
            "name": meta[cid]["name"],
            "chunk_type": meta[cid]["chunk_type"],
            "start_line": meta[cid]["start_line"],
            "in_degree": degree,
            "pagerank": pagerank.get(cid, 0.0),
            "edge_types": by_id[cid],
            "by_provenance": _provenance_rollup(by_id[cid]),
        } for cid in ids if cid in meta]
        group.sort(key=lambda h: (-h["pagerank"], h["path"], h["start_line"]))
        hubs.extend(group)
        if len(hubs) >= limit:
            break
    return {"hubs": hubs[:limit]}
