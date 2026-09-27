"""Shaped graph answers: usages, outgoing references and impact, with their note text."""

from collections import Counter
from typing import Any

from chonks.core.batching import batched
from chonks.core.edges import CALLS, _COLLAPSE_RANK, _HUB_EDGE_TYPES, _MAX_CROSS_LANG_OCCURRENCES, edge_provenance
from chonks.core.symbols import FORWARD_DECLARATION
from chonks.retrieval.callers import (
    Reached, Targets, last_component, owners_by_chunk, reaching_edges, resolve_targets, whole_word,
)

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


def _only_forward_declared(store, name: str) -> bool:
    rows = store.find_symbols(name)
    return bool(rows) and all(r["kind"] == FORWARD_DECLARATION for r in rows)


def _forward_declared_note(name: str) -> str:
    return (f"{name!r} is only forward-declared in the indexed code (its definition "
            "is outside it), so no reference edges point at it")


def _symbol_miss_note(store, targets: Targets) -> str:
    """Diagnostic for a name that resolves to no chunk. A qualified query
    gets no suffix fallback, so this suggests the bare last component
    only when that bare name actually resolves to something."""
    name = targets.name
    if _only_forward_declared(store, name):
        return _forward_declared_note(name) + "; find_usages lists its content matches"
    if targets.lookup is not None and targets.lookup.classes:
        note = f"{' / '.join(targets.lookup.classes)} has no method {targets.member!r}"
    else:
        note = "symbol not found"
    if "::" in name or "." in name:
        bare = targets.member
        if bare and bare != name and store.resolve_symbol_chunk_ids(bare):
            return f"{note} — try the bare name {bare!r}"
    return note


def _lookup_notes(targets: Targets) -> list[str]:
    """How the class model read a qualified name: the classes a partial owner
    matched, and the base a method is inherited from."""
    found = targets.lookup
    if not targets.checked:
        return []
    owner = targets.name[:-len(targets.member)].rstrip(":.")
    notes = []
    if found.guessed or len(found.classes) > 1:
        notes.append(f"{owner!r} matched {', '.join(found.classes)}")
    if found.classes and not set(found.found_in) & set(found.classes):
        via = ", ".join(f"{c}::{targets.member}" for c in found.found_in)
        notes.append(f"{owner} does not define {targets.member!r}; it inherits {via}, whose usages these are")
    return notes


def _fts_scan_for_name(store, name: str, path_prefix: str | None,
                       limit: int | None) -> list[dict[str, Any]]:
    """FTS content-scan fallback for find_usages/get_impact when a name
    exceeds _MAX_CROSS_LANG_OCCURRENCES. A pre-filter, not a proven
    reference; callers must label these as content matches, not edges."""
    phrase = '"' + name.replace('"', '""') + '"'
    # FTS splits on `_`, so the phrase for wl_surface also matches
    # wl_surface_commit; keep whole-word hits only, over a wider fetch.
    word = whole_word(name)
    rows = store.search_fts(phrase, top_k=max((limit or 50) * 20, 1000), path_prefix=path_prefix)
    rows = [r for r in rows if word.search(r.get("content") or "")][: limit or 50]
    return [
        {
            "chunk_id": r["id"], "path": r["path"], "name": r.get("name"),
            "chunk_type": r.get("chunk_type"), "start_line": r["start_line"],
            "end_line": r["end_line"], "origin": "fts_scan",
        }
        for r in rows
    ]


def _above_cap_note(name: str, definers: int, fallback: str) -> str:
    return (f"{name!r} has {definers} definers, above the edge-indexing cap "
            f"({_MAX_CROSS_LANG_OCCURRENCES}) — reference edges are not indexed for this name; {fallback}")


def _reaching(store, targets: Targets, edges: list[tuple[str, str, str]],
              path_prefix: str | None) -> tuple[list[Reached], dict[str, dict], int]:
    """The `edges` into `targets` its callers' references explain, from callers
    under `path_prefix`; with those callers' chunk rows and how many linked
    callers were left out."""
    spans = {r["chunk_id"]: r for r in store.get_chunk_spans(sorted({f for f, _t, _et in edges}), path_prefix)}
    kept = reaching_edges(store, targets, [e for e in edges if e[0] in spans], spans)
    reached = {r.from_id for r in kept}
    return kept, spans, len(set(spans) - reached)


def _left_out_note(left_out: int, targets: Targets) -> str | None:
    if not left_out:
        return None
    return (f"{left_out} linked chunk(s) left out: they reach the defining chunk through another "
            f"name it defines, or their {targets.member!r} calls resolve to another definition")


def find_definitions(store, name: str) -> list[dict[str, Any]]:
    """find_symbols rows; for a method the class model qualifies, the symbol
    rows of its definitions."""
    targets = resolve_targets(store, name)
    if not targets.checked:
        return store.find_symbols(name)
    return [r for r in store.get_symbols_by_chunk_ids(targets.chunk_ids)
            if r["name"] == name or last_component(r["name"]) == targets.member]


def find_usages(store, name: str, path_prefix: str | None = None,
                limit: int | None = None) -> dict[str, Any]:
    """Who references `name`. Empty `results` doesn't mean "no callers":
    see `note` for the unresolved-name and above-cap cases. `limit`
    truncates AFTER sorting by edge quality, not alphabetically."""
    targets = resolve_targets(store, name)
    chunk_ids = targets.chunk_ids
    if not chunk_ids:
        if _only_forward_declared(store, name):
            note = _forward_declared_note(name) + "; falling back to an FTS content scan"
            return {"results": [], "note": note, "by_class": [],
                    "content_matches": _fts_scan_for_name(store, name, path_prefix, limit)}
        return {"results": [], "note": _symbol_miss_note(store, targets), "by_class": [], "content_matches": []}

    edges = store.get_refs_to_chunks_typed(chunk_ids)
    if not edges:
        if len(chunk_ids) > _MAX_CROSS_LANG_OCCURRENCES:
            note = _above_cap_note(name, len(chunk_ids), "falling back to an FTS content scan")
            content_matches = _fts_scan_for_name(store, name, path_prefix, limit)
            return {"results": [], "note": note, "by_class": [], "content_matches": content_matches}
        return {"results": [], "note": None, "by_class": [], "content_matches": []}

    kept, spans, left_out = _reaching(store, targets, edges, path_prefix)
    # Collapse to the least-uncertain edge type when a chunk has more
    # than one into the resolution set (see _COLLAPSE_RANK).
    edge_type_by_from: dict[str, str] = {}
    verified: set[str] = set()
    for r in kept:
        cur = edge_type_by_from.get(r.from_id)
        if cur is None or (
            (_COLLAPSE_RANK.get(r.edge_type, 3), r.edge_type) < (_COLLAPSE_RANK.get(cur, 3), cur)
        ):
            edge_type_by_from[r.from_id] = r.edge_type
        if r.verified:
            verified.add(r.from_id)

    owners = {} if targets.checked else owners_by_chunk(store, chunk_ids, targets.member)
    classes: dict[str, set[str]] = {}
    for r in kept:
        if r.verified and owners.get(r.to_id):
            classes.setdefault(r.from_id, set()).update(owners[r.to_id])

    results: list[dict[str, Any]] = []
    for from_id, et in edge_type_by_from.items():
        row = dict(spans[from_id])
        row["edge_type"] = et
        row["provenance"] = edge_provenance(et)
        if targets.checked and et == CALLS and from_id not in verified:
            row["unverified"] = True
        if from_id in classes:
            row["classes"] = sorted(classes[from_id])
        row.pop("language", None)
        results.append(row)
    results.sort(key=lambda r: (_COLLAPSE_RANK.get(r["edge_type"], 3), bool(r.get("unverified")),
                                r["path"], r["start_line"]))

    by_class = Counter(c for cs in classes.values() for c in cs)
    notes = _lookup_notes(targets)
    unverified = sum(1 for r in results if r.get("unverified"))
    if unverified:
        notes.append(f"{unverified} caller(s) unverified: the class model cannot tell which class "
                     f"their {targets.member!r} call reaches")
    if len(by_class) > 1:
        top = ", ".join(f"{c} {n}" for c, n in by_class.most_common(5))
        more = f", +{len(by_class) - 5} more" if len(by_class) > 5 else ""
        example = by_class.most_common(1)[0][0]
        notes.append(f"{name!r} is a method of {len(by_class)} classes (callers: {top}{more}); "
                     f"name one, as in {example}::{targets.member}, for its callers alone")
    if (left := _left_out_note(left_out, targets)) is not None:
        notes.append(left)
    if limit and len(results) > limit:
        omitted = results[limit:]
        omitted_xlang = sum(1 for r in omitted if r["edge_type"] == "xlang")
        omitted_associated = sum(1 for r in omitted if r["edge_type"] == "associated")
        omitted_mentions = sum(1 for r in omitted if r["edge_type"] == "mentions")
        omitted_typed = len(omitted) - omitted_xlang - omitted_associated - omitted_mentions
        notes.append(
            f"limit={limit} truncated {len(omitted)} result(s): "
            f"{omitted_typed} typed, {omitted_xlang} xlang, "
            f"{omitted_associated} associated, {omitted_mentions} mentions"
        )
        results = results[:limit]
    return {
        "results": results, "note": "\n".join(notes) or None,
        "by_class": [{"class": c, "callers": n} for c, n in by_class.most_common()],
        "content_matches": [],
    }


def _shared_chunk_note(store, targets: Targets) -> str | None:
    """Outgoing edges belong to whole chunks; names the other definitions
    sharing a chunk with this one."""
    others = sorted({r["name"] for r in store.get_symbols_by_chunk_ids(targets.chunk_ids)
                     if last_component(r["name"]) != targets.member})
    if not others:
        return None
    shown = ", ".join(others[:5]) + (f", +{len(others) - 5} more" if len(others) > 5 else "")
    return (f"{targets.name!r} shares its chunk with {shown}; outgoing references are the whole "
            "chunk's, since recorded calls carry no line")


def find_outgoing(store, name: str, path_prefix: str | None = None,
                  limit: int | None = None) -> dict[str, Any]:
    """Forward twin of find_usages (self-refs among definers excluded).
    Zero outgoing edges is a valid empty result. `limit` truncates
    AFTER sorting by edge quality, not alphabetically."""
    targets = resolve_targets(store, name)
    chunk_ids = targets.chunk_ids
    if not chunk_ids:
        return {"results": [], "note": _symbol_miss_note(store, targets)}

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
    for r in store.get_chunk_spans(to_ids, path_prefix):
        r.pop("language", None)
        et = edge_type_by_to.get(r["chunk_id"], "mentions")
        r["edge_type"] = et
        r["provenance"] = edge_provenance(et)
        results.append(r)
    results.sort(key=lambda r: (_COLLAPSE_RANK.get(r["edge_type"], 3), r["path"], r["start_line"]))
    if limit:
        results = results[:limit]
    notes = [*_lookup_notes(targets), _shared_chunk_note(store, targets)]
    return {"results": results, "note": "\n".join(n for n in notes if n) or None}


def get_impact(store, name: str, path_prefix: str | None = None,
               limit: int = 20, rank_by: str = "pagerank_sum") -> dict[str, Any]:
    """Blast radius of `name`, aggregated by referencing file, with a
    `note` matching find_usages' diagnostics. `rank_by="pagerank_sum"`
    degrades cleanly to count-DESC when chunk_pagerank is empty."""
    if rank_by not in ("pagerank_sum", "count"):
        raise ValueError(
            f"invalid rank_by {rank_by!r} — must be 'pagerank_sum' or 'count'"
        )
    targets = resolve_targets(store, name)
    def_chunk_ids = targets.chunk_ids
    if not def_chunk_ids:
        return {
            "symbol": name, "definitions": [], "total_references": 0,
            "by_edge_type": {}, "by_provenance": {}, "rank_by": rank_by,
            "files": [], "files_total": 0,
            "note": _symbol_miss_note(store, targets),
        }

    definitions = [{"path": r["path"], "name": r["name"], "chunk_type": r["chunk_type"]}
                   for r in store.get_chunk_spans(def_chunk_ids)]
    definitions.sort(key=lambda d: (d["path"], d["name"] or ""))

    kept, spans, left_out = _reaching(store, targets, store.get_refs_to_chunks_typed(def_chunk_ids), path_prefix)
    pagerank = store.get_pagerank_for_chunks(sorted({r.from_id for r in kept}))
    edges = [{"chunk_id": r.from_id, "edge_type": r.edge_type, "path": spans[r.from_id]["path"],
              "name": spans[r.from_id]["name"], "chunk_type": spans[r.from_id]["chunk_type"],
              "start_line": spans[r.from_id]["start_line"], "pagerank": pagerank.get(r.from_id, 0.0)}
             for r in kept]

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

    notes = _lookup_notes(targets)
    if not edges and not left_out and len(def_chunk_ids) > _MAX_CROSS_LANG_OCCURRENCES:
        notes.append(_above_cap_note(name, len(def_chunk_ids), "try find_usages, which falls "
                                     "back to an FTS content scan for this case"))
    if (left := _left_out_note(left_out, targets)) is not None:
        notes.append(left)

    return {
        "symbol": name,
        "definitions": definitions,
        "total_references": len(edges),
        "by_edge_type": by_edge_type,
        "by_provenance": by_provenance,
        "rank_by": rank_by,
        "files": file_list[:limit],
        "files_total": files_total,
        "note": "\n".join(notes) or None,
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
