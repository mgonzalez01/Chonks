"""Tests for precomputed in-degree (chunk_indegree): get_hubs
served from the table vs. the old-DB live fallback, the edge_types filter,
get_impact's rank_by, and the >100k-chunk guard on the live fallback.

test_build_refs_incremental.py already proves chunk_indegree stays exact
across every incremental build_refs mutation shape; this file covers the
query-side surface (get_hubs/get_impact) built on top of it.
"""
import tempfile
from collections import Counter

import pytest

from chonks.index.graph.refs import build_refs
from chonks.retrieval.graph_queries import get_hubs, get_impact
from chonks.store import Store


def _mk_store() -> Store:
    return Store(tempfile.mktemp(suffix=".db"))


def _embed() -> list[float]:
    return [0.1, 0.2, 0.3, 0.4]


def _chunk(
    id, path, name=None, content="", metadata=None,
    start_line=1, end_line=1, language="cpp",
):
    return {
        "id": id, "path": path, "language": language, "chunk_type": "function_definition",
        "name": name, "start_line": start_line, "end_line": end_line,
        "content": content, "metadata": metadata,
    }


def _insert(store: Store, chunks: list[dict]) -> None:
    store.insert_chunks(chunks, [_embed() for _ in chunks])


def _seed_hub_corpus(store: Store) -> None:
    """Base: r1 (calls), r2 (calls), r3 (mentions) -> in_degree=3, {calls:2, mentions:1}
    Util: r3 (mentions)                          -> in_degree=1, {mentions:1}
    """
    chunks = [
        _chunk("hub", "src/base.cpp", name="Base", content="class Base {};"),
        _chunk("util", "src/util.cpp", name="Util", content="class Util {};"),
        _chunk("r1", "src/a.cpp", name="a1", content="void a1() { Base(); }",
               metadata={"calls": ["Base"], "imports": [], "inherits": []}),
        _chunk("r2", "src/b.cpp", name="b1", content="void b1() { Base(); }",
               metadata={"calls": ["Base"], "imports": [], "inherits": []}),
        _chunk("r3", "other/c.cpp", name="c1", content="void c1() { Base(); Util(); }"),
    ]
    _insert(store, chunks)
    build_refs(store)


def _indegree_table(store: Store) -> dict[tuple[str, str], int]:
    rows = store._conn.execute("SELECT chunk_id, edge_type, n FROM chunk_indegree").fetchall()
    return {(r["chunk_id"], r["edge_type"]): r["n"] for r in rows}


# ---------------------------------------------------------------------------
# Bulk build_refs populates chunk_indegree exactly
# ---------------------------------------------------------------------------

def test_bulk_build_refs_populates_indegree_matching_recompute():
    store = _mk_store()
    _seed_hub_corpus(store)

    edges = store.get_all_refs_typed()
    want = dict(Counter((to_id, edge_type) for (_f, to_id, edge_type) in edges))
    assert _indegree_table(store) == want
    assert want[("hub", "calls")] == 2
    assert want[("hub", "mentions")] == 1
    assert want[("util", "mentions")] == 1
    store.close()


def test_build_refs_on_empty_corpus_clears_indegree():
    store = _mk_store()
    _seed_hub_corpus(store)
    assert _indegree_table(store)  # sanity: populated

    # Deleting every chunk and rebuilding refs on an empty corpus must clear
    # chunk_indegree too (clear_refs, called unconditionally before the
    # `if not chunks` early-return, now also clears chunk_indegree).
    for path in ("src/base.cpp", "src/util.cpp", "src/a.cpp", "src/b.cpp", "other/c.cpp"):
        store.delete_file(path)
    store.commit()
    written = build_refs(store)
    assert written == 0
    assert _indegree_table(store) == {}
    store.close()


def test_build_refs_never_populated_indegree_forces_full_rebuild_not_partial_delta():
    """Regression: an old DB with chunk_refs populated via raw insert_refs
    (chunk_indegree never went through save_indegree) must force the full
    rebuild on the next build_refs call, not take the scoped incremental
    path, which would leave chunk_indegree permanently incomplete while
    has_indegree() reports True."""
    store = _mk_store()
    chunks = [
        _chunk("hub", "src/base.cpp", name="Base", content="class Base {};"),
        _chunk("hub2", "src/other.cpp", name="Other", content="class Other {};"),
    ] + [
        _chunk(f"r{i}", f"src/r{i}.cpp", name=f"fn{i}", content="void fn() { Base(); }")
        for i in range(20)
    ]
    _insert(store, chunks)
    # Raw insert bypasses build_refs, so chunk_refs is populated but
    # chunk_indegree stays empty, like an old/never-rebuilt DB.
    store.insert_refs([("r0", "hub2", "calls")])
    store.commit()
    assert store.count_refs() > 0
    assert not store.has_indegree()

    new_chunk = _chunk("new1", "src/new1.cpp", name="newfn", content="void newfn() { Base(); }")
    _insert(store, [new_chunk])
    build_refs(store, changed_ids={"new1"}, deleted_ids=set(), deleted_names=set())

    edges = store.get_all_refs_typed()
    want = dict(Counter((to_id, edge_type) for (_f, to_id, edge_type) in edges))
    assert _indegree_table(store) == want
    store.close()


# ---------------------------------------------------------------------------
# get_hubs: precomputed (chunk_indegree) vs. live fallback must agree exactly
# ---------------------------------------------------------------------------

def test_hubs_precomputed_matches_live_fallback_global_and_scoped():
    store = _mk_store()
    _seed_hub_corpus(store)

    precomputed_global = get_hubs(store)
    precomputed_scoped = get_hubs(store, path_prefix="src/")
    assert precomputed_global["hubs"], "fixture should produce global hubs"
    assert precomputed_scoped["hubs"], "fixture should produce scoped hubs"

    with store._lock:
        store._conn.execute("DELETE FROM chunk_indegree")
        store._conn.commit()

    live_global = get_hubs(store)
    live_scoped = get_hubs(store, path_prefix="src/")

    assert precomputed_global == live_global
    assert precomputed_scoped == live_scoped
    store.close()


def test_hubs_precomputed_ranks_by_indegree_then_pagerank():
    store = _mk_store()
    _seed_hub_corpus(store)
    store.save_pagerank({"hub": 0.9, "util": 0.1})

    result = get_hubs(store)
    assert [h["name"] for h in result["hubs"]] == ["Base", "Util"]
    assert result["hubs"][0]["in_degree"] == 3
    assert result["hubs"][0]["edge_types"] == {"calls": 2, "mentions": 1}
    assert result["hubs"][0]["pagerank"] == 0.9
    store.close()


def test_hubs_precomputed_by_provenance_rollup():
    store = _mk_store()
    _seed_hub_corpus(store)

    result = get_hubs(store)
    by_name = {h["name"]: h for h in result["hubs"]}
    assert by_name["Base"]["by_provenance"] == {"extracted": 2, "inferred": 1}
    assert by_name["Util"]["by_provenance"] == {"inferred": 1}
    store.close()


def test_hubs_live_fallback_by_provenance_rollup():
    """Same by_provenance rollup, on the live (_get_hubs_live) path once
    chunk_indegree is emptied."""
    store = _mk_store()
    _seed_hub_corpus(store)
    with store._lock:
        store._conn.execute("DELETE FROM chunk_indegree")
        store._conn.commit()

    scoped = {h["name"]: h for h in get_hubs(store, path_prefix="src/")["hubs"]}
    assert scoped["Base"]["by_provenance"] == {"extracted": 2, "inferred": 1}

    unscoped = {h["name"]: h for h in get_hubs(store)["hubs"]}
    assert unscoped["Base"]["by_provenance"] == {"extracted": 2, "inferred": 1}
    assert unscoped["Util"]["by_provenance"] == {"inferred": 1}
    store.close()


# ---------------------------------------------------------------------------
# edge_types filter
# ---------------------------------------------------------------------------

def test_hubs_edge_types_filter_sums_only_selected_types():
    store = _mk_store()
    _seed_hub_corpus(store)

    result = get_hubs(store, edge_types=["calls"])
    names = {h["name"]: h for h in result["hubs"]}
    assert names["Base"]["in_degree"] == 2
    assert names["Base"]["edge_types"] == {"calls": 2}
    assert "Util" not in names
    store.close()


def test_hubs_edge_types_filter_accepts_associated():
    """'associated' (PMI-selected high-signal mentions) must be accepted
    by the edge_types filter, not rejected as an unknown edge type."""
    store = _mk_store()
    _insert(store, [
        _chunk("hub", "src/base.cpp", name="Base", content="class Base {};"),
        _chunk("r1", "src/a.cpp", name="a1", content="void a1() { Base(); }"),
    ])
    store.insert_refs([("r1", "hub", "associated")])
    store.save_indegree({("hub", "associated"): 1})

    result = get_hubs(store, edge_types=["associated"])
    names = {h["name"]: h for h in result["hubs"]}
    assert names["Base"]["in_degree"] == 1
    assert names["Base"]["edge_types"] == {"associated": 1}
    store.close()


def test_hubs_edge_types_filter_matches_between_precomputed_and_live():
    store = _mk_store()
    _seed_hub_corpus(store)
    precomputed = get_hubs(store, edge_types=["mentions"])

    with store._lock:
        store._conn.execute("DELETE FROM chunk_indegree")
        store._conn.commit()
    live = get_hubs(store, edge_types=["mentions"])

    assert precomputed == live
    assert {h["name"] for h in precomputed["hubs"]} == {"Base", "Util"}
    store.close()


def test_hubs_edge_types_invalid_raises():
    store = _mk_store()
    _seed_hub_corpus(store)
    with pytest.raises(ValueError, match="edge_types"):
        get_hubs(store, edge_types=["bogus"])
    store.close()


def test_hubs_edge_types_invalid_raises_on_live_fallback_too():
    store = _mk_store()
    _seed_hub_corpus(store)
    with store._lock:
        store._conn.execute("DELETE FROM chunk_indegree")
        store._conn.commit()
    with pytest.raises(ValueError, match="edge_types"):
        get_hubs(store, edge_types=["bogus"])
    store.close()


# ---------------------------------------------------------------------------
# Old-DB fallback: empty chunk_indegree still works, guard still fires
# ---------------------------------------------------------------------------

def test_hubs_old_db_fallback_still_works_with_empty_indegree_table(tmp_path):
    store = Store(tmp_path / "old.db")
    store.insert_chunks(
        [_chunk("hub", "src/base.cpp", "Base"), _chunk("r1", "src/a.cpp", "a1")],
        [_embed(), _embed()],
    )
    store.insert_refs([("r1", "hub", "calls")])
    # No build_refs call: chunk_indegree was never populated, like a DB
    # indexed before chunk_indegree existed.
    assert _indegree_table(store) == {}

    result = get_hubs(store)
    assert [h["name"] for h in result["hubs"]] == ["Base"]
    assert result["hubs"][0]["in_degree"] == 1


def test_hubs_global_guard_still_fires_on_empty_indegree_table(tmp_path, monkeypatch):
    import chonks.retrieval.graph_queries as graph_queries
    store = Store(tmp_path / "old.db")
    store.insert_chunks(
        [_chunk("hub", "src/base.cpp", "Base"), _chunk("r1", "src/a.cpp", "a1")],
        [_embed(), _embed()],
    )
    store.insert_refs([("r1", "hub", "calls")])
    monkeypatch.setattr(graph_queries, "_HUBS_GLOBAL_MAX_CHUNKS", 1)

    with pytest.raises(ValueError, match="path_prefix"):
        get_hubs(store)
    assert get_hubs(store, path_prefix="src")["hubs"]  # scoped branch unaffected


# ---------------------------------------------------------------------------
# get_impact rank_by
# ---------------------------------------------------------------------------

def _impact_chunk(id, path, name, s=1):
    return {"id": id, "path": path, "language": "cpp", "chunk_type": "function",
            "name": name, "start_line": s, "end_line": s + 1, "content": f"{name}()"}


def _sym(path, name, chunk_id):
    return {"path": path, "name": name, "kind": "function", "language": "cpp",
            "start_line": 1, "end_line": 1, "chunk_id": chunk_id}


def test_impact_rank_by_count_orders_by_raw_reference_count():
    store = _mk_store()
    store.insert_chunks(
        [_impact_chunk("def", "widget.cpp", "Widget"),
         # fileA: 3 referencing chunks, low pagerank each.
         _impact_chunk("a1", "a.cpp", "one", s=1),
         _impact_chunk("a2", "a.cpp", "two", s=2),
         _impact_chunk("a3", "a.cpp", "three", s=3),
         # fileB: 1 referencing chunk, high pagerank.
         _impact_chunk("b1", "b.cpp", "only", s=1)],
        [_embed()] * 5,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", "def")])
    store.insert_refs([("a1", "def"), ("a2", "def"), ("a3", "def"), ("b1", "def")])
    store.save_pagerank({"a1": 0.01, "a2": 0.01, "a3": 0.01, "b1": 0.9})

    default = get_impact(store, "applyStep")
    assert default["rank_by"] == "pagerank_sum"
    assert [f["path"] for f in default["files"]] == ["b.cpp", "a.cpp"]

    by_count = get_impact(store, "applyStep", rank_by="count")
    assert by_count["rank_by"] == "count"
    assert [f["path"] for f in by_count["files"]] == ["a.cpp", "b.cpp"]
    assert [f["count"] for f in by_count["files"]] == [3, 1]
    store.close()


def test_impact_rank_by_invalid_raises():
    store = _mk_store()
    with pytest.raises(ValueError, match="rank_by"):
        get_impact(store, "applyStep", rank_by="bogus")
    store.close()


# ---------------------------------------------------------------------------
# purge_orphan_refs must not partially populate a never-built chunk_indegree
# ---------------------------------------------------------------------------

def test_purge_orphan_refs_skips_indegree_refresh_when_never_populated():
    """Same hazard as the build_refs regression above, via a different
    entry point: purge_orphan_refs must leave a never-built chunk_indegree
    empty, not write a partial delta that makes has_indegree() report True."""
    store = _mk_store()
    store.insert_chunks(
        [_chunk("c1", "a.cpp", "fn_a"), _chunk("c2", "b.cpp", "fn_b")],
        [_embed(), _embed()],
    )
    store.insert_refs([("c1", "c2"), ("c1", "ghost")])  # "ghost" was never inserted
    store.commit()
    assert not store.has_indegree()

    removed = store.purge_orphan_refs()
    assert removed == 1
    assert _indegree_table(store) == {}, "must stay empty, not partially populated"
    store.close()
