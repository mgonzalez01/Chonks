"""Tests for chunks_fts trigger-based alignment."""
import sqlite3

import numpy as np
import pytest
from chonks.chunking import CODE_LANGUAGES
from chonks.store import Store, _chunk_kind_clause


def _make_store(tmp_path) -> Store:
    return Store(tmp_path / "test.db")


def _insert_chunk_direct(store: Store, chunk_id: str, content: str, name: str | None = None) -> None:
    """Insert a chunk row directly into the chunks table, bypassing insert_chunks()."""
    with store._lock:
        store._conn.execute(
            "INSERT INTO chunks(id, path, content, name, indexed_at) VALUES(?,?,?,?,0.0)",
            (chunk_id, "test.py", content, name),
        )
        store._conn.commit()


def test_fts_trigger_on_direct_insert(tmp_path):
    store = _make_store(tmp_path)
    _insert_chunk_direct(store, "c1", "def greet_world(): pass", name="greet_world")

    results = store.search_fts("greet_world")
    ids = [r["id"] for r in results]
    assert "c1" in ids


def test_fts_name_weight_lifts_definition_over_body_repeats(tmp_path):
    """search_fts weights `name` above `content`: a chunk whose name IS the
    query term must outrank one that merely repeats it more in the body.
    Filler chunks keep bm25 corpus stats non-degenerate."""
    store = _make_store(tmp_path)
    _insert_chunk_direct(store, "def_chunk", "def target_fn(): return compute_value()", name="target_fn")
    _insert_chunk_direct(store, "body_chunk", "def runner(): target_fn() target_fn()", name="runner")
    for i in range(3):
        _insert_chunk_direct(store, f"filler{i}", f"def unrelated_{i}(): return {i}", name=f"unrelated_{i}")

    ids = [r["id"] for r in store.search_fts("target_fn")]
    assert ids[0] == "def_chunk", f"name match must rank first, got {ids}"
    store.close()


def test_fts_trigger_on_direct_delete(tmp_path):
    store = _make_store(tmp_path)
    _insert_chunk_direct(store, "c1", "def goodbye_world(): pass", name="goodbye_world")

    assert any(r["id"] == "c1" for r in store.search_fts("goodbye_world"))

    with store._lock:
        store._conn.execute("DELETE FROM chunks WHERE id = 'c1'")
        store._conn.commit()

    assert not any(r["id"] == "c1" for r in store.search_fts("goodbye_world"))


def _fake_embedding(dim: int = 4) -> list[float]:
    return [0.5] * dim


def test_insert_chunks_still_searchable(tmp_path):
    store = _make_store(tmp_path)
    chunk = {
        "id": "c1",
        "path": "foo.py",
        "language": "python",
        "chunk_type": "function",
        "name": "do_something",
        "start_line": 1,
        "end_line": 5,
        "content": "def do_something(): pass",
    }
    store.insert_chunks([chunk], [_fake_embedding()])

    results = store.search_fts("do_something")
    assert any(r["id"] == "c1" for r in results)


def test_insert_chunks_dedups_duplicate_ids_in_batch(tmp_path):
    """Regression: a duplicate id within one batch used to raise 'UNIQUE
    constraint failed on chunk_vecs'. The batch must collapse to one row,
    last write winning, with chunks and chunk_vecs staying consistent."""
    store = _make_store(tmp_path)
    base = {
        "id": "dup",
        "path": "same.cs",
        "language": "c_sharp",
        "chunk_type": "method",
        "name": "Foo",
        "start_line": 10,
        "end_line": 12,
        "content": "void Foo() {}",
    }
    second = {**base, "content": "void Foo() {} // last wins"}

    store.insert_chunks([base, second], [_fake_embedding(), _fake_embedding()])

    with store._lock:
        n_chunks = store._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE id='dup'").fetchone()[0]
        n_vecs = store._conn.execute(
            "SELECT COUNT(*) FROM chunk_vecs WHERE id='dup'").fetchone()[0]
        content = store._conn.execute(
            "SELECT content FROM chunks WHERE id='dup'").fetchone()[0]
    assert n_chunks == 1
    assert n_vecs == 1
    assert content == "void Foo() {} // last wins"


def test_delete_file_removes_from_fts(tmp_path):
    store = _make_store(tmp_path)
    chunk = {
        "id": "c1",
        "path": "bar.py",
        "language": "python",
        "chunk_type": "function",
        "name": "remove_me",
        "start_line": 1,
        "end_line": 5,
        "content": "def remove_me(): pass",
    }
    store.insert_chunks([chunk], [_fake_embedding()])
    store.upsert_file("bar.py", 100, 0.0, "hash1")
    store.commit()

    store.delete_file("bar.py")
    store.commit()

    assert not any(r["id"] == "c1" for r in store.search_fts("remove_me"))


def test_reindex_same_id_no_stale_fts(tmp_path):
    """Re-indexing an id via insert_chunks without a prior delete_file relies on
    recursive_triggers=ON, so the INSERT OR REPLACE conflict-delete fires the
    chunks_ad trigger and removes the old content's FTS terms."""
    store = _make_store(tmp_path)
    chunk = {
        "id": "c1",
        "path": "foo.py",
        "language": "python",
        "chunk_type": "function",
        "name": "alpha_term",
        "start_line": 1,
        "end_line": 5,
        "content": "def alpha_term(): pass",
    }
    store.insert_chunks([chunk], [_fake_embedding()])
    assert any(r["id"] == "c1" for r in store.search_fts("alpha_term"))

    chunk2 = {**chunk, "name": "beta_term", "content": "def beta_term(): pass"}
    store.insert_chunks([chunk2], [_fake_embedding()])

    assert any(r["id"] == "c1" for r in store.search_fts("beta_term"))
    assert not any(r["id"] == "c1" for r in store.search_fts("alpha_term")), \
        "stale FTS posting survived a same-id re-index"


def test_search_semantic_does_not_persist_dim(tmp_path):
    store = Store(tmp_path / "test.db")

    assert store._dim is None

    results = store.search_semantic([0.1, 0.2, 0.3, 0.4], top_k=5)

    assert results == []
    assert store._dim is None, (
        "search_semantic wrote embedding_dim from the query vector — "
        "this would cause a mismatch on subsequent insert_chunks with a different dim"
    )

    store.close()


# --- symbols table (decoupled symbol index) -------------------------------

def _sym(path, name, kind="function_definition", lang="cpp", s=1, e=2, chunk_id=None):
    return {"path": path, "name": name, "kind": kind, "language": lang,
            "start_line": s, "end_line": e, "chunk_id": chunk_id}


def test_symbols_insert_and_find(tmp_path):
    store = _make_store(tmp_path)
    store.insert_symbols([
        _sym("a.cpp", "Foo", kind="class_specifier", s=1, e=20, chunk_id="c1"),
        _sym("a.cpp", "bar", s=5, e=9, chunk_id="c1"),
        _sym("b.cpp", "bar", s=1, e=3, chunk_id="c2"),
    ])
    assert store.symbols_count() == 3
    hits = store.find_symbols("bar")
    assert {(h["path"], h["chunk_id"]) for h in hits} == {("a.cpp", "c1"), ("b.cpp", "c2")}
    assert [s["name"] for s in store.get_all_symbols("a.cpp")] == ["Foo", "bar"]


def test_symbols_name_chunks_skips_null(tmp_path):
    store = _make_store(tmp_path)
    store.insert_symbols([
        _sym("a.cpp", "Widget", chunk_id="c1"),
        _sym("a.cpp", "orphan", chunk_id=None),
    ])
    m = store.get_symbol_name_chunks()
    assert m == {"Widget": ["c1"]}, f"NULL chunk_id should be skipped: {m}"


def test_symbols_cascade_on_delete_file(tmp_path):
    store = _make_store(tmp_path)
    store.insert_symbols([_sym("a.cpp", "Foo", chunk_id="c1"), _sym("b.cpp", "Bar", chunk_id="c2")])
    store.delete_file("a.cpp")
    remaining = [s["name"] for s in store.get_all_symbols()]
    assert remaining == ["Bar"], f"delete_file must cascade to symbols: {remaining}"


# --- incremental k-NN id-list batching (>900 ids must not blow SQLITE_MAX_VARIABLE_NUMBER) ---

def test_neighbor_id_list_methods_batch_over_900_ids(tmp_path):
    """get_chunk_ids_by_neighbor / delete_neighbors_touching / delete_neighbors_from
    build a SQL IN(...) clause from a caller-supplied id list; each must batch
    it (mirroring get_int8_embeddings_by_ids) or blow past
    SQLITE_MAX_VARIABLE_NUMBER on stock sqlite builds. Uses 2000 ids (> 900)."""
    store = _make_store(tmp_path)
    n = 2000

    rows = [("hub", f"n{i}", float(i)) for i in range(n)]
    rows.append(("keep-me", "other-neighbor", 999999.0))
    store.insert_neighbors(rows)
    store.commit()

    assert store.count_neighbors() == n + 1

    ids = [f"n{i}" for i in range(n)]

    hit = store.get_chunk_ids_by_neighbor(ids)
    assert hit == {"hub"}

    deleted_from = store.delete_neighbors_from(ids)
    assert deleted_from == 0, "no row has chunk_id in ids yet"
    assert store.count_neighbors() == n + 1

    deleted_touching = store.delete_neighbors_touching(ids)
    assert deleted_touching == n
    assert store.count_neighbors() == 1
    remaining = store.get_all_neighbors()
    assert remaining == [("keep-me", "other-neighbor")]


def test_delete_neighbors_from_batches_over_900_ids(tmp_path):
    """delete_neighbors_from must delete rows whose chunk_id (source) is in a
    >900-id list, batching the IN-clause the same way."""
    store = _make_store(tmp_path)
    n = 1500
    rows = [(f"src{i}", "target", float(i)) for i in range(n)]
    rows.append(("other-src", "target", 1.0))
    store.insert_neighbors(rows)
    store.commit()
    assert store.count_neighbors() == n + 1

    ids = [f"src{i}" for i in range(n)]
    deleted = store.delete_neighbors_from(ids)
    assert deleted == n
    assert store.count_neighbors() == 1
    assert store.get_all_neighbors() == [("other-src", "target")]


# --- get_vectors_for_chunks (near-dup-wall detection input) ----------------

def test_get_vectors_for_chunks_returns_dequantized_arrays(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [
            {"id": "c1", "path": "a.py", "content": "def foo(): pass",
             "name": "foo", "start_line": 1, "end_line": 1},
            {"id": "c2", "path": "a.py", "content": "def bar(): pass",
             "name": "bar", "start_line": 2, "end_line": 2},
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    vectors = store.get_vectors_for_chunks(["c1", "c2"])
    assert set(vectors) == {"c1", "c2"}
    assert vectors["c1"].dtype == np.float32
    assert vectors["c1"].shape == (4,)
    # int8 'unit' quantization: c1's [1,0,0,0] and c2's [0,1,0,0] must stay
    # near-orthogonal after dequantization (cosine ~0), not collapse together.
    cos = float(np.dot(vectors["c1"], vectors["c2"]) /
                (np.linalg.norm(vectors["c1"]) * np.linalg.norm(vectors["c2"])))
    assert abs(cos) < 0.1, f"expected near-orthogonal, got cosine={cos}"


def test_get_vectors_for_chunks_omits_ids_with_no_vector_row(tmp_path):
    """Same contract as get_int8_embeddings_by_ids: an id absent from
    chunk_vecs is simply missing from the result, not an error or a zero
    vector."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [{"id": "c1", "path": "a.py", "content": "def foo(): pass",
          "name": "foo", "start_line": 1, "end_line": 1}],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    vectors = store.get_vectors_for_chunks(["c1", "ghost-id"])
    assert set(vectors) == {"c1"}


def test_get_vectors_for_chunks_empty_ids_returns_empty(tmp_path):
    store = _make_store(tmp_path)
    assert store.get_vectors_for_chunks([]) == {}


# ---------------------------------------------------------------------------
# chunk_refs.edge_type: typed edges
# ---------------------------------------------------------------------------

def test_edge_type_migration_on_old_schema_db(tmp_path):
    """An old DB whose chunk_refs table predates edge_type gets the column
    added in place, no need to drop and re-index just for this change."""
    import sqlite3
    db_path = tmp_path / "old.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO meta(key,value) VALUES('schema_version', '5')"
    )
    conn.execute(
        "CREATE TABLE chunk_refs (from_id TEXT NOT NULL, to_id TEXT NOT NULL, "
        "PRIMARY KEY (from_id, to_id))"
    )
    conn.execute("INSERT INTO chunk_refs(from_id, to_id) VALUES ('a', 'b')")
    conn.commit()
    conn.close()

    store = Store(db_path)
    cols = {row["name"] for row in
            store._conn.execute("PRAGMA table_info(chunk_refs)").fetchall()}
    assert "edge_type" in cols

    # Pre-existing row defaults to 'mentions', no backfill needed.
    row = store._conn.execute(
        "SELECT edge_type FROM chunk_refs WHERE from_id='a' AND to_id='b'"
    ).fetchone()
    assert row["edge_type"] == "mentions"
    store.close()


def test_save_pagerank_load_pagerank_roundtrip(tmp_path):
    store = _make_store(tmp_path)
    assert store.load_pagerank() == {}
    store.save_pagerank({"a": 0.5, "b": 0.25})
    assert store.load_pagerank() == {"a": 0.5, "b": 0.25}
    store.close()


def test_save_pagerank_second_call_fully_replaces_first(tmp_path):
    store = _make_store(tmp_path)
    store.save_pagerank({"a": 0.5, "b": 0.25})
    store.save_pagerank({"c": 0.9})
    assert store.load_pagerank() == {"c": 0.9}
    store.close()


def test_insert_refs_defaults_to_mentions(tmp_path):
    store = _make_store(tmp_path)
    store.insert_refs([("a", "b")])
    assert store.get_all_refs_typed() == [("a", "b", "mentions")]


def test_insert_refs_typed(tmp_path):
    store = _make_store(tmp_path)
    store.insert_refs([("a", "b", "calls"), ("a", "c", "imports"), ("a", "d", "inherits")])
    typed = {(f, t, e) for f, t, e in store.get_all_refs_typed()}
    assert typed == {("a", "b", "calls"), ("a", "c", "imports"), ("a", "d", "inherits")}
    assert set(store.get_all_refs()) == {("a", "b"), ("a", "c"), ("a", "d")}


# --- find_usages (chunk_refs incoming edges resolved via the symbol index) --

def _chunk(id, path, name, s=1, e=2):
    return {"id": id, "path": path, "language": "cpp", "chunk_type": "function",
            "name": name, "start_line": s, "end_line": e, "content": f"{name}()"}


def test_find_usages_resolves_via_symbol_index(tmp_path):
    store = _make_store(tmp_path)
    # "applyStep" is defined inside a folded chunk c1 (symbol name != chunk
    # name); this exercises resolution through the symbol index.
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "caller.cpp", "run", s=10, e=12),
         _chunk("c3", "other_caller.cpp", "helper", s=1, e=3)],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([("c2", "c1"), ("c3", "c1")])

    hits = store.find_usages("applyStep")["results"]
    assert {(h["path"], h["chunk_id"]) for h in hits} == {
        ("caller.cpp", "c2"), ("other_caller.cpp", "c3"),
    }
    assert [h["path"] for h in hits] == ["caller.cpp", "other_caller.cpp"]
    assert {h["provenance"] for h in hits} == {"inferred"}


def test_find_usages_row_provenance_by_edge_type(tmp_path):
    """An unknown edge_type (not in the provenance map) degrades to
    "inferred" rather than raising."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "caller.cpp", "run"),
         _chunk("c3", "mentioner.cpp", "helper"),
         _chunk("c4", "unknown_type.cpp", "weird")],
        [_fake_embedding()] * 4,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([
        ("c2", "c1", "calls"),
        ("c3", "c1", "mentions"),
        ("c4", "c1", "some_future_edge_type"),
    ])

    hits = {h["chunk_id"]: h["provenance"] for h in store.find_usages("applyStep")["results"]}
    assert hits == {"c2": "extracted", "c3": "inferred", "c4": "inferred"}


def test_find_usages_associated_ranks_between_xlang_and_mentions(tmp_path):
    """'associated' (PMI-selected high-signal mentions) sits between 'xlang'
    and plain 'mentions' in _COLLAPSE_RANK, and its provenance still reads
    'inferred' since it's a labelled subset of the mentions mechanism."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "a.cpp", "xlang_caller"),
         _chunk("c3", "b.cpp", "assoc_caller"),
         _chunk("c4", "c.cpp", "mentions_caller")],
        [_fake_embedding()] * 4,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([
        ("c2", "c1", "xlang"),
        ("c3", "c1", "associated"),
        ("c4", "c1", "mentions"),
    ])

    hits = store.find_usages("applyStep")["results"]
    assert [h["chunk_id"] for h in hits] == ["c2", "c3", "c4"]
    provenance = {h["chunk_id"]: h["provenance"] for h in hits}
    assert provenance["c3"] == "inferred"


def test_find_usages_unknown_edge_type_falls_back_below_associated(tmp_path):
    """Regression: _COLLAPSE_RANK's unknown-type fallback must track the
    current max rank (3, now that 'associated' occupies rank 2), not the old
    max of 2, or an unknown edge_type would tie 'associated' and flip order."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "zzz_assoc.cpp", "assoc_caller"),
         _chunk("c3", "aaa_unknown.cpp", "unknown_caller")],
        [_fake_embedding()] * 3,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([
        ("c2", "c1", "associated"),
        ("c3", "c1", "some_future_edge_type"),
    ])

    hits = store.find_usages("applyStep")["results"]
    assert [h["chunk_id"] for h in hits] == ["c2", "c3"]


def test_find_usages_quality_ordering_beats_alphabetical(tmp_path):
    """Mirrors test_find_outgoing_quality_ordering_beats_alphabetical for the
    incoming direction: a widely-referenced symbol with an uppercase doc path
    (sorts before code dirs in ASCII) mentioning it and a real code caller
    calling it must surface the typed caller first, and a small limit must
    keep it rather than the alphabetically-earlier doc mention."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "DOCS.md", "mentioned"),
         _chunk("c3", "zzz_caller.cpp", "called")],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([("c2", "c1", "mentions"), ("c3", "c1", "calls")])

    hits = store.find_usages("applyStep")["results"]
    assert [h["chunk_id"] for h in hits] == ["c3", "c2"]

    limited = store.find_usages("applyStep", limit=1)["results"]
    assert [h["chunk_id"] for h in limited] == ["c3"]


def test_find_usages_limit_truncation_notes_typed_vs_mentions(tmp_path):
    """When limit truncates results, the note must disambiguate the omitted
    count between typed edges and mentions (silent-lies rule)."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "a.cpp", "caller_a"),
         _chunk("c3", "b.cpp", "caller_b"),
         _chunk("c4", "c.md", "mention_c")],
        [_fake_embedding()] * 4,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([
        ("c2", "c1", "calls"),
        ("c3", "c1", "calls"),
        ("c4", "c1", "mentions"),
    ])

    result = store.find_usages("applyStep", limit=1)
    assert [h["chunk_id"] for h in result["results"]] == ["c2"]
    assert result["note"] == \
        "limit=1 truncated 2 result(s): 1 typed, 0 xlang, 0 associated, 1 mentions"


def test_find_usages_limit_truncation_notes_associated_separately_from_typed(tmp_path):
    """Regression: an omitted 'associated' row must land in its own bucket, not
    get silently folded into 'typed' by the omitted_typed subtraction, which
    predates the 'associated' edge_type."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "a.cpp", "caller_a"),
         _chunk("c3", "b.cpp", "caller_b")],
        [_fake_embedding()] * 3,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([
        ("c2", "c1", "calls"),
        ("c3", "c1", "associated"),
    ])

    result = store.find_usages("applyStep", limit=1)
    assert [h["chunk_id"] for h in result["results"]] == ["c2"]
    assert result["note"] == \
        "limit=1 truncated 1 result(s): 0 typed, 0 xlang, 1 associated, 0 mentions"


def test_find_usages_unknown_symbol_returns_empty(tmp_path):
    store = _make_store(tmp_path)
    result = store.find_usages("doesNotExist")
    assert result["results"] == []
    assert result["note"] == "symbol not found"
    assert result["content_matches"] == []


def test_find_usages_path_prefix_scopes_referencing_chunks(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "src/caller.cpp", "run"),
         _chunk("c3", "other/caller.cpp", "helper")],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([("c2", "c1"), ("c3", "c1")])

    hits = store.find_usages("applyStep", path_prefix="src/")["results"]
    assert [h["chunk_id"] for h in hits] == ["c2"]


def test_find_usages_limit(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "a.cpp", "run"),
         _chunk("c3", "b.cpp", "helper")],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([("c2", "c1"), ("c3", "c1")])

    hits = store.find_usages("applyStep", limit=1)["results"]
    assert len(hits) == 1


def test_find_usages_qualified_miss_suggests_bare_name(tmp_path):
    """A qualified miss ("Manager::Init") gets no suffix-match fallback in
    resolve_symbol_chunk_ids, unlike a bare query. The note should point at
    the bare last component when that one resolves."""
    store = _make_store(tmp_path)
    store.insert_chunks([_chunk("c1", "manager.cpp", "Manager")], [_fake_embedding()])
    store.insert_symbols([_sym("manager.cpp", "Init", chunk_id="c1")])

    result = store.find_usages("Manager::Init")
    assert result["results"] == []
    assert result["note"] == "symbol not found — try the bare name 'Init'"
    assert result["content_matches"] == []


def test_find_usages_qualified_miss_with_no_bare_match_is_generic(tmp_path):
    """A qualified miss with no resolvable bare name gets the plain note, not
    a suggestion pointing nowhere."""
    store = _make_store(tmp_path)
    result = store.find_usages("Nope::AlsoNope")
    assert result["note"] == "symbol not found"


def test_find_usages_ubiquitous_name_above_cap_notes_and_falls_back_to_fts(tmp_path):
    """build_refs skips edge creation for a name above
    _MAX_CROSS_LANG_OCCURRENCES (repomap/_shared.py), so find_usages resolves the
    symbol but walks chunk_refs into nothing. That must be disclosed, not
    indistinguishable from "no callers", and backed by a labeled FTS scan."""
    from chonks.repomap import _MAX_CROSS_LANG_OCCURRENCES
    store = _make_store(tmp_path)
    n = _MAX_CROSS_LANG_OCCURRENCES + 1
    # Definer content deliberately omits "init" literally, so the FTS scan
    # below is isolated to the one chunk that actually calls it.
    def_chunks = [
        {"id": f"def{i}", "path": f"def{i}.cpp", "language": "cpp",
         "chunk_type": "class", "name": f"Owner{i}", "start_line": 1, "end_line": 2,
         "content": f"class Owner{i} {{ }}"}
        for i in range(n)
    ]
    store.insert_chunks(def_chunks, [_fake_embedding()] * n)
    store.insert_symbols([_sym(f"def{i}.cpp", "init", chunk_id=f"def{i}") for i in range(n)])
    caller = {"id": "caller1", "path": "caller.cpp", "language": "cpp",
              "chunk_type": "function", "name": "run", "start_line": 1,
              "end_line": 3, "content": "void run() { init(); }"}
    store.insert_chunks([caller], [_fake_embedding()])

    result = store.find_usages("init")
    assert result["results"] == []
    assert f"{n} definers" in result["note"]
    assert str(_MAX_CROSS_LANG_OCCURRENCES) in result["note"]
    assert [m["chunk_id"] for m in result["content_matches"]] == ["caller1"]
    assert result["content_matches"][0]["origin"] == "fts_scan"


def test_get_impact_ubiquitous_name_above_cap_gets_note(tmp_path):
    """Same disclosure as find_usages, on the aggregate path."""
    from chonks.repomap import _MAX_CROSS_LANG_OCCURRENCES
    store = _make_store(tmp_path)
    n = _MAX_CROSS_LANG_OCCURRENCES + 1
    def_chunks = [
        {"id": f"def{i}", "path": f"def{i}.cpp", "language": "cpp",
         "chunk_type": "class", "name": f"Owner{i}", "start_line": 1, "end_line": 2,
         "content": f"class Owner{i} {{ }}"}
        for i in range(n)
    ]
    store.insert_chunks(def_chunks, [_fake_embedding()] * n)
    store.insert_symbols([_sym(f"def{i}.cpp", "init", chunk_id=f"def{i}") for i in range(n)])

    result = store.get_impact("init")
    assert result["total_references"] == 0
    assert f"{n} definers" in result["note"]


# --- find_outgoing (chunk_refs outgoing edges resolved via the symbol index) -

def test_find_outgoing_resolves_via_symbol_index(tmp_path):
    store = _make_store(tmp_path)
    # "applyStep" is defined inside a folded chunk c1 (symbol name != chunk name).
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "helper.cpp", "helperFn", s=10, e=12),
         _chunk("c3", "other.cpp", "otherFn", s=1, e=3)],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([("c1", "c2"), ("c1", "c3")])

    hits = store.find_outgoing("applyStep")["results"]
    assert {(h["path"], h["chunk_id"]) for h in hits} == {
        ("helper.cpp", "c2"), ("other.cpp", "c3"),
    }
    assert [h["path"] for h in hits] == ["helper.cpp", "other.cpp"]
    assert {h["provenance"] for h in hits} == {"inferred"}
    assert {h["name"] for h in hits} == {"helperFn", "otherFn"}


def test_find_outgoing_edge_type_collapse_picks_least_uncertain(tmp_path):
    """chunk_refs is PRIMARY KEY(from_id, to_id), so when two definer chunks of
    the same overloaded name both reach a target, only one edge_type can be
    stored; the least-uncertain type wins, same _COLLAPSE_RANK find_usages uses."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1a", "widget.cpp", "Widget"),
         _chunk("c1b", "widget2.cpp", "Widget2"),
         _chunk("c2", "target.cpp", "target")],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([
        _sym("widget.cpp", "applyStep", chunk_id="c1a"),
        _sym("widget2.cpp", "applyStep", chunk_id="c1b"),
    ])
    store.insert_refs([("c1a", "c2", "mentions"), ("c1b", "c2", "calls")])

    hits = {h["chunk_id"]: h["provenance"] for h in store.find_outgoing("applyStep")["results"]}
    assert hits == {"c2": "extracted"}


def test_find_outgoing_associated_ranks_between_xlang_and_mentions(tmp_path):
    """Mirrors test_find_usages_associated_ranks_between_xlang_and_mentions for
    the outgoing direction."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("src", "driver.cpp", "driver"),
         _chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "a.cpp", "xlang_target"),
         _chunk("c3", "b.cpp", "assoc_target")],
        [_fake_embedding()] * 4,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="src")])
    store.insert_refs([
        ("src", "c1", "xlang"),
        ("src", "c2", "associated"),
        ("src", "c3", "mentions"),
    ])

    hits = store.find_outgoing("applyStep")["results"]
    assert [h["chunk_id"] for h in hits] == ["c1", "c2", "c3"]
    provenance = {h["chunk_id"]: h["provenance"] for h in hits}
    assert provenance["c2"] == "inferred"


def test_find_outgoing_excludes_definer_set_self_edges(tmp_path):
    """A self-reference among the definer set itself (overloads referencing
    each other) is noise, not a real outgoing call, and must be excluded."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "widget.cpp", "Widget2"),
         _chunk("c3", "real_target.cpp", "target")],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([
        _sym("widget.cpp", "applyStep", chunk_id="c1"),
        _sym("widget.cpp", "applyStep", chunk_id="c2"),
    ])
    store.insert_refs([("c1", "c2"), ("c1", "c3")])

    hits = store.find_outgoing("applyStep")["results"]
    assert [h["chunk_id"] for h in hits] == ["c3"]


def test_find_outgoing_path_prefix_scopes_target_chunks(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "src/target.cpp", "run"),
         _chunk("c3", "other/target.cpp", "helper")],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([("c1", "c2"), ("c1", "c3")])

    hits = store.find_outgoing("applyStep", path_prefix="src/")["results"]
    assert [h["chunk_id"] for h in hits] == ["c2"]


def test_find_outgoing_limit(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "a.cpp", "run"),
         _chunk("c3", "b.cpp", "helper")],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([("c1", "c2"), ("c1", "c3")])

    hits = store.find_outgoing("applyStep", limit=1)["results"]
    assert len(hits) == 1


def test_find_outgoing_unknown_symbol_returns_empty(tmp_path):
    store = _make_store(tmp_path)
    result = store.find_outgoing("doesNotExist")
    assert result["results"] == []
    assert result["note"] == "symbol not found"


def test_find_outgoing_edge_type_field(tmp_path):
    """Each row carries the collapsed edge_type, not just the
    derived provenance label."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "a.cpp", "run"),
         _chunk("c3", "b.cpp", "helper")],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([("c1", "c2", "calls"), ("c1", "c3", "mentions")])

    hits = {h["chunk_id"]: h["edge_type"] for h in store.find_outgoing("applyStep")["results"]}
    assert hits == {"c2": "calls", "c3": "mentions"}


def test_find_outgoing_quality_ordering_beats_alphabetical(tmp_path):
    """Measured on the Godot corpus: outgoing edges skew heavily toward
    cross-language 'mentions' noise against typed edges, so a 'mentions' edge
    sorting alphabetically before a 'calls' edge must still order after it."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "aaa_mentions_target.cpp", "mentioned"),
         _chunk("c3", "zzz_calls_target.cpp", "called")],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([("c1", "c2", "mentions"), ("c1", "c3", "calls")])

    hits = store.find_outgoing("applyStep")["results"]
    assert [h["chunk_id"] for h in hits] == ["c3", "c2"]

    limited = store.find_outgoing("applyStep", limit=1)["results"]
    assert [h["chunk_id"] for h in limited] == ["c3"]


def test_find_outgoing_resolved_with_no_outgoing_edges_is_valid_empty(tmp_path):
    """A resolved symbol with zero outgoing edges is a valid empty result:
    note stays None, unlike a resolution miss."""
    store = _make_store(tmp_path)
    store.insert_chunks([_chunk("c1", "widget.cpp", "Widget")], [_fake_embedding()])
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])

    result = store.find_outgoing("applyStep")
    assert result["results"] == []
    assert result["note"] is None


# --- get_impact (find_usages aggregated by referencing file) ---------------

def test_get_impact_unknown_symbol_returns_empty_shape(tmp_path):
    store = _make_store(tmp_path)
    result = store.get_impact("doesNotExist")
    assert result == {
        "symbol": "doesNotExist", "definitions": [], "total_references": 0,
        "by_edge_type": {}, "by_provenance": {}, "rank_by": "pagerank_sum",
        "files": [], "files_total": 0,
        "note": "symbol not found",
    }


def test_get_impact_multiple_definitions_and_edge_types(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.cpp", "Widget"),
         _chunk("c1b", "b.cpp", "Gadget"),
         _chunk("c2", "caller.cpp", "run"),
         _chunk("c3", "other.cpp", "helper")],
        [_fake_embedding()] * 4,
    )
    store.insert_symbols([
        _sym("a.cpp", "applyStep", chunk_id="c1"),
        _sym("b.cpp", "applyStep", chunk_id="c1b"),
    ])
    store.insert_refs([("c2", "c1", "calls"), ("c3", "c1b", "mentions")])

    result = store.get_impact("applyStep")
    assert {(d["path"], d["name"]) for d in result["definitions"]} == {
        ("a.cpp", "Widget"), ("b.cpp", "Gadget"),
    }
    assert result["total_references"] == 2
    assert result["by_edge_type"] == {"calls": 1, "mentions": 1}
    assert result["by_provenance"] == {"extracted": 1, "inferred": 1}
    assert result["files_total"] == 2
    paths = {f["path"] for f in result["files"]}
    assert paths == {"caller.cpp", "other.cpp"}


def test_get_impact_by_provenance_sums_match_by_edge_type(tmp_path):
    """by_provenance's counts, summed, always equal by_edge_type's counts,
    summed: the rollup redistributes, never drops or double-counts."""
    from chonks.repomap import edge_provenance
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("def", "widget.cpp", "Widget"),
         _chunk("c1", "a.cpp", "one"), _chunk("c2", "b.cpp", "two"),
         _chunk("c3", "c.cpp", "three"), _chunk("c4", "d.cpp", "four")],
        [_fake_embedding()] * 5,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="def")])
    store.insert_refs([
        ("c1", "def", "calls"), ("c2", "def", "imports"),
        ("c3", "def", "mentions"), ("c4", "def", "xlang"),
    ])

    result = store.get_impact("applyStep")
    assert sum(result["by_provenance"].values()) == sum(result["by_edge_type"].values())
    expected: dict[str, int] = {}
    for et, n in result["by_edge_type"].items():
        expected[edge_provenance(et)] = expected.get(edge_provenance(et), 0) + n
    assert result["by_provenance"] == expected
    assert result["by_provenance"] == {"extracted": 2, "inferred": 1, "paired": 1}


def test_get_impact_path_prefix_scopes_referencing_files(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "widget.cpp", "Widget"),
         _chunk("c2", "src/caller.cpp", "run"),
         _chunk("c3", "other/caller.cpp", "helper")],
        [_fake_embedding()] * 3,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="c1")])
    store.insert_refs([("c2", "c1"), ("c3", "c1")])

    result = store.get_impact("applyStep", path_prefix="src/")
    assert [f["path"] for f in result["files"]] == ["src/caller.cpp"]
    assert result["total_references"] == 1


def test_get_impact_empty_pagerank_degrades_to_count_desc_path_asc(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("def", "widget.cpp", "Widget"),
         _chunk("c1", "z.cpp", "one", s=1),
         _chunk("c2", "a.cpp", "two", s=1),
         _chunk("c3", "a.cpp", "three", s=5)],
        [_fake_embedding()] * 4,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="def")])
    # z.cpp: 1 reference. a.cpp: 2 references (two distinct chunks) -> a.cpp
    # must rank first (higher count) despite z.cpp < a.cpp alphabetically.
    store.insert_refs([("c1", "def"), ("c2", "def"), ("c3", "def")])
    assert store.load_pagerank() == {}

    result = store.get_impact("applyStep")
    assert [f["path"] for f in result["files"]] == ["a.cpp", "z.cpp"]
    assert [f["count"] for f in result["files"]] == [2, 1]
    assert all(f["pagerank_sum"] == 0.0 for f in result["files"])


def test_get_impact_truncation_and_files_total(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("def", "widget.cpp", "Widget"),
         _chunk("c1", "a.cpp", "one"),
         _chunk("c2", "b.cpp", "two"),
         _chunk("c3", "c.cpp", "three")],
        [_fake_embedding()] * 4,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="def")])
    store.insert_refs([("c1", "def"), ("c2", "def"), ("c3", "def")])

    result = store.get_impact("applyStep", limit=2)
    assert result["files_total"] == 3
    assert len(result["files"]) == 2
    # deterministic tie-break (equal pagerank_sum=0, equal count=1) -> path ASC
    assert [f["path"] for f in result["files"]] == ["a.cpp", "b.cpp"]


def test_get_impact_top_referrers_ranked_by_pagerank_then_start_line(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("def", "widget.cpp", "Widget"),
         _chunk("r1", "a.cpp", "first", s=10),
         _chunk("r2", "a.cpp", "second", s=1),
         _chunk("r3", "a.cpp", "third", s=5)],
        [_fake_embedding()] * 4,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="def")])
    store.insert_refs([("r1", "def"), ("r2", "def"), ("r3", "def")])
    # All three referrers tie on pagerank (table empty -> 0.0 each), so the
    # top_referrers tie-break is start_line ASC.
    result = store.get_impact("applyStep")
    file_ = result["files"][0]
    assert [r["name"] for r in file_["top_referrers"]] == ["second", "third", "first"]


def test_get_impact_top_referrers_capped_at_three(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("def", "widget.cpp", "Widget"),
         _chunk("r1", "a.cpp", "one", s=1),
         _chunk("r2", "a.cpp", "two", s=2),
         _chunk("r3", "a.cpp", "three", s=3),
         _chunk("r4", "a.cpp", "four", s=4)],
        [_fake_embedding()] * 5,
    )
    store.insert_symbols([_sym("widget.cpp", "applyStep", chunk_id="def")])
    store.insert_refs([("r1", "def"), ("r2", "def"), ("r3", "def"), ("r4", "def")])
    result = store.get_impact("applyStep")
    assert len(result["files"][0]["top_referrers"]) == 3
    assert result["files"][0]["count"] == 4


# --- get_hubs (in-degree ranked structural hub listing) ---------------------

def test_get_hubs_ranks_by_indegree_then_pagerank(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("hub", "base.cpp", "Base"),
         _chunk("minor", "util.cpp", "Util"),
         _chunk("r1", "a.cpp", "a1"), _chunk("r2", "b.cpp", "b1"),
         _chunk("r3", "c.cpp", "c1")],
        [_fake_embedding()] * 5,
    )
    # "hub" referenced by 3 chunks, "minor" by 1.
    store.insert_refs([
        ("r1", "hub", "calls"), ("r2", "hub", "calls"), ("r3", "hub", "inherits"),
        ("r1", "minor"),
    ])
    store.save_pagerank({"hub": 0.9, "minor": 0.1})

    result = store.get_hubs()
    assert [h["path"] for h in result["hubs"]] == ["base.cpp", "util.cpp"]
    assert result["hubs"][0]["in_degree"] == 3
    assert result["hubs"][0]["edge_types"] == {"calls": 2, "inherits": 1}
    assert result["hubs"][0]["pagerank"] == 0.9


def test_get_hubs_path_prefix_scopes_hub_chunk_not_referrer(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("hub_in", "src/base.cpp", "Base"),
         _chunk("hub_out", "other/base.cpp", "Base2"),
         _chunk("r1", "caller.cpp", "caller")],
        [_fake_embedding()] * 3,
    )
    store.insert_refs([("r1", "hub_in"), ("r1", "hub_out")])

    result = store.get_hubs(path_prefix="src/")
    assert [h["path"] for h in result["hubs"]] == ["src/base.cpp"]


def test_get_hubs_excludes_nameless_chunks(tmp_path):
    store = _make_store(tmp_path)
    nameless = _chunk("hub", "base.cpp", None)
    store.insert_chunks(
        [nameless, _chunk("r1", "a.cpp", "a1"), _chunk("r2", "b.cpp", "b1")],
        [_fake_embedding()] * 3,
    )
    store.insert_refs([("r1", "hub"), ("r2", "hub")])

    result = store.get_hubs()
    assert result["hubs"] == []


def test_get_hubs_deterministic_tie_break_path_then_start_line(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("z1", "z.cpp", "zfn", s=1),
         _chunk("a2", "a.cpp", "afn2", s=5),
         _chunk("a1", "a.cpp", "afn1", s=1),
         _chunk("r1", "caller1.cpp", "c1"),
         _chunk("r2", "caller2.cpp", "c2")],
        [_fake_embedding()] * 5,
    )
    # All three hub candidates tie on in_degree=1 and pagerank=0.0 (unset).
    store.insert_refs([("r1", "z1"), ("r1", "a2"), ("r2", "a1")])

    result = store.get_hubs()
    assert [(h["path"], h["start_line"]) for h in result["hubs"]] == [
        ("a.cpp", 1), ("a.cpp", 5), ("z.cpp", 1),
    ]


def test_get_hubs_zero_indegree_chunks_excluded(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("lonely", "solo.cpp", "Solo")],
        [_fake_embedding()],
    )
    result = store.get_hubs()
    assert result["hubs"] == []


def test_get_hubs_limit(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("h1", "a.cpp", "a1"), _chunk("h2", "b.cpp", "b1"),
         _chunk("h3", "c.cpp", "c1"), _chunk("r1", "r.cpp", "r1")],
        [_fake_embedding()] * 4,
    )
    store.insert_refs([("r1", "h1"), ("r1", "h2"), ("r1", "h3")])
    result = store.get_hubs(limit=2)
    assert len(result["hubs"]) == 2


def test_find_symbols_bare_name_matches_qualified(tmp_path):
    """Observed in practice on a large corpus: out-of-line C++ definitions
    store 'Class::method'; a bare 'method' lookup must fall back to
    last-component suffix matching instead of zero-hitting an indexed symbol."""
    from chonks.store import Store
    store = Store(tmp_path / "t.db")
    try:
        store.insert_symbols([
            {"path": "a.cpp", "name": "AABB::encloses", "kind": "function_definition",
             "language": "cpp", "start_line": 1, "end_line": 5, "chunk_id": "c1"},
            {"path": "b.cs", "name": "Foo.Bar", "kind": "method_declaration",
             "language": "c_sharp", "start_line": 1, "end_line": 5, "chunk_id": "c2"},
            {"path": "c.cpp", "name": "encloses_something", "kind": "function_definition",
             "language": "cpp", "start_line": 9, "end_line": 12, "chunk_id": "c3"},
        ])
        hits = store.find_symbols("encloses")
        assert [h["name"] for h in hits] == ["AABB::encloses"]
        assert store.find_symbols("Bar")[0]["name"] == "Foo.Bar"
        assert store.find_symbols("AABB::encloses")[0]["chunk_id"] == "c1"
        assert store.find_symbols("Nope::encloses") == []
        assert store.find_symbols("enclose_") == []
        assert store.resolve_symbol_chunk_ids("encloses") == ["c1"]
    finally:
        store.close()


def test_like_metacharacters_in_path_prefix_dont_leak_scope(tmp_path):
    """Regression: get_all_symbols/find_symbols/find_usages built bare LIKE
    clauses with no ESCAPE, so '_' (a LIKE wildcard) in path_prefix 'a_b/'
    silently also matched the sibling directory 'axb/'."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a_b/x.cpp", "shared_fn"),
         _chunk("c2", "axb/x.cpp", "shared_fn")],
        [_fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([
        _sym("a_b/x.cpp", "shared_fn", chunk_id="c1"),
        _sym("axb/x.cpp", "shared_fn", chunk_id="c2"),
        # Same path as shared_fn above: isolates the prefix-name branch's own
        # escape from the path-filter escape, since only the NAME's '_'
        # differs between a literal match and a wildcard match here.
        _sym("a_b/x.cpp", "render_pass", chunk_id="c1"),
        _sym("a_b/x.cpp", "renderXpass", chunk_id="c1"),
    ])

    paths = {s["path"] for s in store.get_all_symbols(path_prefix="a_b/")}
    assert paths == {"a_b/x.cpp"}, f"'_' acted as a wildcard: {paths}"

    hits = store.find_symbols("shared_fn", path_prefix="a_b/")
    assert [h["path"] for h in hits] == ["a_b/x.cpp"], hits

    prefix_hits = store.find_symbols("shared", path_prefix="a_b/", prefix=True)
    assert [h["path"] for h in prefix_hits] == ["a_b/x.cpp"], prefix_hits

    render_hits = store.find_symbols("render_", path_prefix="a_b/", prefix=True)
    assert [h["name"] for h in render_hits] == ["render_pass"], (
        f"'_' in the prefix NAME acted as a wildcard, pulled in siblings: {render_hits}"
    )

    # find_usages: path_prefix scopes the REFERENCING chunks, not the
    # definition site, so scoping to 'a_b/' must exclude the axb/ sibling
    # caller rather than including it via the '_' wildcard.
    store.insert_chunks(
        [_chunk("caller_ab", "a_b/caller.cpp", "caller_in_ab"),
         _chunk("caller_x", "axb/caller.cpp", "caller_in_axb")],
        [_fake_embedding(), _fake_embedding()],
    )
    store.insert_refs([("caller_ab", "c1"), ("caller_x", "c1")])
    usage_hits = store.find_usages("shared_fn", path_prefix="a_b/")["results"]
    assert [h["path"] for h in usage_hits] == ["a_b/caller.cpp"], usage_hits


def test_purge_orphan_neighbors_and_refs(tmp_path):
    """purge_orphan_neighbors()/purge_orphan_refs() must delete only rows
    whose chunk_id/neighbor_id (from_id/to_id) is NOT in `chunks`; a row with
    both endpoints present is a live edge and must never be touched."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.cpp", "fn_a"), _chunk("c2", "b.cpp", "fn_b")],
        [_fake_embedding(), _fake_embedding()],
    )

    store.insert_neighbors([("c1", "c2", 0.1), ("c2", "c1", 0.1)])
    store.insert_refs([("c1", "c2")])

    # Orphan edges: one endpoint references a chunk id that was never inserted.
    store.insert_neighbors([("c1", "ghost", 0.9), ("ghost2", "c2", 0.9)])
    store.insert_refs([("c1", "ghost"), ("ghost2", "c2")])
    store.commit()

    removed_neighbors = store.purge_orphan_neighbors()
    removed_refs = store.purge_orphan_refs()
    store.commit()

    assert removed_neighbors == 2, removed_neighbors
    assert removed_refs == 2, removed_refs

    remaining_neighbors = {
        (r["chunk_id"], r["neighbor_id"])
        for r in store._conn.execute("SELECT chunk_id, neighbor_id FROM chunk_neighbors").fetchall()
    }
    assert remaining_neighbors == {("c1", "c2"), ("c2", "c1")}, remaining_neighbors

    remaining_refs = {
        (r["from_id"], r["to_id"])
        for r in store._conn.execute("SELECT from_id, to_id FROM chunk_refs").fetchall()
    }
    assert remaining_refs == {("c1", "c2")}, remaining_refs

    assert store.purge_orphan_neighbors() == 0
    assert store.purge_orphan_refs() == 0


# ---------------------------------------------------------------------------
# "touching" methods bind batch+batch: must slice at 450, not 900
# ---------------------------------------------------------------------------

def test_touching_methods_batch_over_500_ids_avoid_variable_overflow(tmp_path):
    """get_neighbor_edges_touching / delete_refs_touching /
    delete_xlang_refs_touching each build a WHERE x IN(batch) OR y IN(batch)
    clause, binding batch+batch params. At the old 900-id slice, 600 ids alone
    bind 1200 params, over SQLite's default SQLITE_MAX_VARIABLE_NUMBER (999).
    Pins SQLITE_LIMIT_VARIABLE_NUMBER to that default since some builds
    compile in a higher limit and wouldn't otherwise overflow here."""
    store = _make_store(tmp_path)
    store._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    n = 600
    ids = [f"id{i}" for i in range(n)]

    # get_neighbor_edges_touching: n rows touching `ids` via neighbor_id, plus
    # one unrelated row that must not be returned.
    store.insert_neighbors(
        [("hub", f"id{i}", float(i)) for i in range(n)] + [("other", "other-neighbor", 1.0)]
    )
    store.commit()
    edges = store.get_neighbor_edges_touching(ids)
    assert len(edges) == n
    assert all(nid in (fr, to) for fr, to, _d in edges for nid in (fr, to) if nid.startswith("id"))
    assert not any(fr == "other" or to == "other-neighbor" for fr, to, _d in edges)

    # delete_xlang_refs_touching: type-scoped, not id-scoped, so a
    # same-endpoint non-xlang row must survive.
    store.insert_refs([(f"id{i}", "hub", "xlang") for i in range(n)])
    store.insert_refs([("unrelated-from", "unrelated-to", "xlang")])
    store.insert_refs([("id0", "hub2", "calls")])
    store.commit()

    removed_xlang, _removed_xlang_to_ids = store.delete_xlang_refs_touching(ids)
    assert removed_xlang == n
    remaining = set(store.get_all_refs_typed())
    assert ("unrelated-from", "unrelated-to", "xlang") in remaining
    assert ("id0", "hub2", "calls") in remaining
    assert not any(e == "xlang" and (fr in ids or to in ids) for fr, to, e in remaining)

    removed, _removed_to_ids = store.delete_refs_touching(ids)
    assert removed == 1  # only the surviving ("id0","hub2","calls") row touches ids
    remaining2 = set(store.get_all_refs_typed())
    assert remaining2 == {("unrelated-from", "unrelated-to", "xlang")}


# ---------------------------------------------------------------------------
# delete_file / update_vectors: batch chunk_vecs+chunks IN-deletes
# ---------------------------------------------------------------------------

def test_delete_file_batches_over_900_chunks(tmp_path):
    """delete_file builds one `id IN (...)` over every chunk of a path for
    both the chunk_vecs and chunks deletes; unbatched, >900 chunks raises
    sqlite3.OperationalError. The variable-number limit is pinned only AFTER
    the setup insert, since insert_chunks has its own separate unbatched IN
    over `ids` that would otherwise trip the same ceiling during setup."""
    store = _make_store(tmp_path)
    n = 1200
    chunks = [
        {"id": f"c{i}", "path": "big.py", "language": "python", "chunk_type": "function",
         "name": f"fn_{i}", "start_line": i, "end_line": i, "content": f"def fn_{i}(): pass"}
        for i in range(n)
    ]
    store.insert_chunks(chunks, [_fake_embedding() for _ in range(n)])
    with store._lock:
        assert store._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE path='big.py'"
        ).fetchone()[0] == n
        assert store._conn.execute(
            "SELECT COUNT(*) FROM chunk_vecs WHERE id LIKE 'c%'"
        ).fetchone()[0] == n

    store._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    deleted_ids = store.delete_file("big.py")
    assert len(deleted_ids) == n

    with store._lock:
        assert store._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE path='big.py'"
        ).fetchone()[0] == 0
        assert store._conn.execute(
            "SELECT COUNT(*) FROM chunk_vecs WHERE id LIKE 'c%'"
        ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# update_vectors must never delete without a matching re-insert
# ---------------------------------------------------------------------------

def test_update_vectors_empty_embeddings_does_not_delete_existing_vectors(tmp_path):
    """update_vectors(ids=[...], embeddings=[]) must refuse (raise) rather
    than deleting the existing chunk_vecs rows without re-inserting anything,
    which would leave chunks rows without a matching chunk_vecs row."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("a", "a.cpp", "fnA"), _chunk("b", "b.cpp", "fnB")],
        [_fake_embedding(), _fake_embedding()],
    )
    with store._lock:
        before = store._conn.execute(
            "SELECT COUNT(*) FROM chunk_vecs WHERE id IN ('a','b')"
        ).fetchone()[0]
    assert before == 2

    with pytest.raises(ValueError):
        store.update_vectors(["a", "b"], [])

    with store._lock:
        after = store._conn.execute(
            "SELECT COUNT(*) FROM chunk_vecs WHERE id IN ('a','b')"
        ).fetchone()[0]
    assert after == 2, "existing vectors must survive a rejected mismatched-length update"


def test_get_hubs_global_guard_large_corpus(tmp_path, monkeypatch):
    """Unscoped get_hubs on a large corpus must refuse loudly (measured >10
    min at 385k chunks, store lock held for the whole walk) rather than wedge
    the server; scoped calls stay allowed."""
    import chonks.store as store_mod
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("h1", "src/core.cpp", "Core"), _chunk("h2", "src/user.cpp", "use")],
        [_fake_embedding()] * 2,
    )
    store.insert_refs([("h2", "h1", "calls")])
    monkeypatch.setattr(store_mod, "_HUBS_GLOBAL_MAX_CHUNKS", 1)
    with pytest.raises(ValueError, match="path_prefix"):
        store.get_hubs()
    assert store.get_hubs(path_prefix="src")["hubs"]  # scoped branch unaffected


# ---------------------------------------------------------------------------
# chunk_kind filter: code (AST-tier) vs docs (text-fallback)
# ---------------------------------------------------------------------------

def test_chunk_kind_clause_any_and_none_are_noop():
    assert _chunk_kind_clause(None) == ("", [])
    assert _chunk_kind_clause("any") == ("", [])


def test_chunk_kind_clause_invalid_raises():
    with pytest.raises(ValueError):
        _chunk_kind_clause("bogus")


def test_chunk_kind_clause_code_and_docs_use_the_same_language_set():
    code_clause, code_params = _chunk_kind_clause("code")
    docs_clause, docs_params = _chunk_kind_clause("docs")
    assert set(code_params) == set(docs_params) == CODE_LANGUAGES
    assert "IN" in code_clause and "NOT IN" not in code_clause
    assert "NOT IN" in docs_clause


def _kind_chunk(id, path, language, content, name=None, s=1, e=1):
    return {"id": id, "path": path, "language": language, "content": content,
            "name": name, "start_line": s, "end_line": e}


def test_search_fts_chunk_kind_code_and_docs_partition_results(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [
            _kind_chunk("code1", "a.py", "python", "def needle(): pass", name="needle"),
            _kind_chunk("docs1", "a.md", "md", "the needle in the haystack"),
        ],
        [_fake_embedding(), _fake_embedding()],
    )
    assert {r["id"] for r in store.search_fts("needle")} == {"code1", "docs1"}
    assert {r["id"] for r in store.search_fts("needle", chunk_kind="code")} == {"code1"}
    assert {r["id"] for r in store.search_fts("needle", chunk_kind="docs")} == {"docs1"}


def test_search_fts_chunk_kind_any_matches_default_behavior(tmp_path):
    """Regression guard: chunk_kind="any"/None must be byte-identical to
    omitting the parameter entirely; a no-docs corpus is enough to prove the
    filter clause is a true no-op in that case."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_kind_chunk("code1", "a.py", "python", "def needle(): pass", name="needle")],
        [_fake_embedding()],
    )
    default = store.search_fts("needle")
    assert default == store.search_fts("needle", chunk_kind="any")
    assert default == store.search_fts("needle", chunk_kind=None)


def test_search_regex_chunk_kind_code_and_docs_partition_results(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [
            _kind_chunk("code1", "a.py", "python", "needle"),
            _kind_chunk("docs1", "a.md", "md", "needle"),
        ],
        [_fake_embedding(), _fake_embedding()],
    )
    assert {r["id"] for r in store.search_regex("needle")} == {"code1", "docs1"}
    assert {r["id"] for r in store.search_regex("needle", chunk_kind="code")} == {"code1"}
    assert {r["id"] for r in store.search_regex("needle", chunk_kind="docs")} == {"docs1"}


def test_search_regex_chunk_kind_filters_before_top_k_truncation(tmp_path):
    """search_regex streams rows in insertion order and stops after top_k
    matches. chunk_kind="code" must still surface a code chunk inserted after
    5 docs chunks, proving the SQL-side filter runs before the top_k break."""
    store = _make_store(tmp_path)
    chunks = [_kind_chunk(f"d{i}", f"d{i}.md", "md", "needle") for i in range(5)]
    chunks.append(_kind_chunk("code1", "a.py", "python", "needle"))
    store.insert_chunks(chunks, [_fake_embedding()] * len(chunks))

    unfiltered = store.search_regex("needle", top_k=1)
    assert unfiltered and unfiltered[0]["language"] == "md"

    filtered = store.search_regex("needle", top_k=1, chunk_kind="code")
    assert filtered == [c for c in filtered if c["id"] == "code1"]
    assert filtered and filtered[0]["id"] == "code1"


def test_search_regex_chunk_kind_any_matches_default_behavior(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_kind_chunk("code1", "a.py", "python", "needle")], [_fake_embedding()],
    )
    default = store.search_regex("needle")
    assert default == store.search_regex("needle", chunk_kind="any")
    assert default == store.search_regex("needle", chunk_kind=None)


def test_search_semantic_chunk_kind_filters_before_top_k_truncation(tmp_path):
    """vec0 has no WHERE pushdown, so a fixed oversample factor can under-fill
    top_k when everything nearest is the wrong kind. This corpus makes
    unfiltered top_k=2 return ALL docs; chunk_kind="code" must still surface
    both code chunks, proving the widen-and-retry loop filters before truncating."""
    store = _make_store(tmp_path)
    query      = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    orthogonal = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    docs_chunks = [_kind_chunk(f"d{i}", f"docs/d{i}.md", "md", f"doc {i}") for i in range(10)]
    code_chunks = [
        _kind_chunk("c0", "src/a.py", "python", "def a(): pass"),
        _kind_chunk("c1", "src/b.py", "python", "def b(): pass"),
    ]
    store.insert_chunks(
        docs_chunks + code_chunks,
        [query] * len(docs_chunks) + [orthogonal] * len(code_chunks),
    )

    # Sanity check: unfiltered top_k=2 is all docs, confirming the setup is real.
    unfiltered = store.search_semantic(query, top_k=2)
    assert {c["id"] for c in unfiltered}.issubset({d["id"] for d in docs_chunks})

    filtered = store.search_semantic(query, top_k=2, chunk_kind="code")
    assert {c["id"] for c in filtered} == {"c0", "c1"}


def test_search_semantic_chunk_kind_any_matches_default_behavior(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("a", "a.cpp", "fnA"), _chunk("b", "b.cpp", "fnB")],
        [_fake_embedding(), _fake_embedding()],
    )
    query = _fake_embedding()
    default = store.search_semantic(query, top_k=5)
    assert default == store.search_semantic(query, top_k=5, chunk_kind="any")
    assert default == store.search_semantic(query, top_k=5, chunk_kind=None)


def test_chunk_kind_docs_includes_null_language(tmp_path):
    """language can be NULL for rows written by callers that never set it.
    chunk_kind="docs" must still catch it, chunk_kind="code" must exclude it."""
    store = _make_store(tmp_path)
    with store._lock:
        store._conn.execute(
            "INSERT INTO chunks(id, path, content, indexed_at) VALUES(?,?,?,0.0)",
            ("nulllang", "weird.xyz", "mystery content"),
        )
        store._conn.commit()

    assert any(c["id"] == "nulllang" for c in store.search_regex("mystery", chunk_kind="docs"))
    assert not any(c["id"] == "nulllang" for c in store.search_regex("mystery", chunk_kind="code"))


def test_chunk_kind_code_matches_ast_language_regardless_of_parse_health(tmp_path):
    """A parse-failed AST-tier file still stores its real language id; chunker
    only falls back to the raw extension when _lang_for_path returns None."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_kind_chunk("healed", "broken.cpp", "cpp", "unparseable garbage")],
        [_fake_embedding()],
    )
    assert store.search_regex("garbage", chunk_kind="code")
    assert not store.search_regex("garbage", chunk_kind="docs")


def test_hub_edge_types_and_collapse_rank_are_shared_with_core_edges():
    """Guards the store.py import: both names must be the same objects as
    the ones in chonks.core.edges, not copies."""
    import chonks.core.edges as edges
    import chonks.store as store_mod
    assert edges._HUB_EDGE_TYPES == frozenset(
        {"calls", "imports", "inherits", "mentions", "xlang", "associated"}
    )
    assert edges._COLLAPSE_RANK == {
        "calls": 0, "imports": 0, "inherits": 0,
        "xlang": 1,
        "associated": 2,
        "mentions": 3,
    }
    assert store_mod._HUB_EDGE_TYPES is edges._HUB_EDGE_TYPES
    assert store_mod._COLLAPSE_RANK is edges._COLLAPSE_RANK


def test_batched():
    from chonks.core.batching import batched
    assert list(batched([0, 1, 2, 3, 4], 2)) == [[0, 1], [2, 3], [4]]
    assert list(batched([0, 1, 2, 3], 2)) == [[0, 1], [2, 3]]
    assert list(batched([], 2)) == []


def test_skeleton_all_names_are_store_globals():
    """Guards the store.py star import: every name in skeleton.__all__ must
    land in chonks.store's namespace."""
    import chonks.core.skeleton as skeleton
    import chonks.store
    assert set(skeleton.__all__) <= set(vars(chonks.store))
