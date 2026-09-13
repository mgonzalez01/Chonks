"""graph_nodes/graph_edges (Store.rebuild_hierarchy) are deliberately
independent of chunk_refs/chunk_pagerank/chunk_indegree; these tests never
touch the chunk_refs graph."""
import tempfile

from chonks.store import Store


def _mk_store() -> Store:
    return Store(tempfile.mktemp(suffix=".db"))


def _embed() -> list[float]:
    return [0.1, 0.2, 0.3, 0.4]


def _chunk(id, path, name=None, content="chunk", start_line=1, end_line=1):
    return {
        "id": id, "path": path, "language": "python", "chunk_type": "function_definition",
        "name": name, "start_line": start_line, "end_line": end_line,
        "content": content, "metadata": None,
    }


def _seed_file(store: Store, path: str, chunk_ids: list[str]) -> None:
    store.upsert_file(path, size=1, mtime=0.0, content_hash="h")
    chunks = [_chunk(cid, path) for cid in chunk_ids]
    if chunks:
        store.insert_chunks(chunks, [_embed() for _ in chunks])
    store.commit()


# ---------------------------------------------------------------------------
# Basic rebuild over a small synthetic corpus
# ---------------------------------------------------------------------------

def _seed_small_corpus(store: Store) -> None:
    # root.py at repo root, plus a/b/c.py nested three dirs deep.
    _seed_file(store, "root.py", ["r1"])
    _seed_file(store, "a/b/c.py", ["c1", "c2"])


def test_rebuild_node_set_and_parent_chain():
    store = _mk_store()
    _seed_small_corpus(store)
    counts = store.rebuild_hierarchy()

    # dirs: ".", "a", "a/b" ; files: "root.py", "a/b/c.py"
    assert counts["nodes"] == 5

    rows = store._conn.execute(
        "SELECT id, kind, path, parent_id FROM graph_nodes"
    ).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}
    assert set(by_id) == {
        "dir:.", "dir:a", "dir:a/b", "file:root.py", "file:a/b/c.py",
    }

    assert by_id["dir:."]["kind"] == "dir"
    assert by_id["dir:."]["path"] == "."
    assert by_id["dir:."]["parent_id"] is None

    assert by_id["dir:a"]["kind"] == "dir"
    assert by_id["dir:a"]["path"] == "a"
    assert by_id["dir:a"]["parent_id"] == "dir:."

    assert by_id["dir:a/b"]["kind"] == "dir"
    assert by_id["dir:a/b"]["path"] == "a/b"
    assert by_id["dir:a/b"]["parent_id"] == "dir:a"

    assert by_id["file:root.py"]["kind"] == "file"
    assert by_id["file:root.py"]["path"] == "root.py"
    assert by_id["file:root.py"]["parent_id"] == "dir:."

    assert by_id["file:a/b/c.py"]["kind"] == "file"
    assert by_id["file:a/b/c.py"]["path"] == "a/b/c.py"
    assert by_id["file:a/b/c.py"]["parent_id"] == "dir:a/b"

    store.close()


def test_rebuild_contains_edges_and_children_ordering():
    store = _mk_store()
    _seed_small_corpus(store)
    counts = store.rebuild_hierarchy()

    # dir->dir: .->a, a->a/b (2)
    # dir->file: .->root.py, a/b->a/b/c.py (2)
    # file->chunk: root.py->r1, a/b/c.py->c1, a/b/c.py->c2 (3)
    assert counts["edges"] == 7

    assert store.get_graph_children("dir:.") == ["dir:a", "file:root.py"]
    assert store.get_graph_children("dir:a") == ["dir:a/b"]
    assert store.get_graph_children("dir:a/b") == ["file:a/b/c.py"]
    assert store.get_graph_children("file:root.py") == ["r1"]
    assert store.get_graph_children("file:a/b/c.py") == ["c1", "c2"]

    assert store.get_graph_children("dir:.", edge_type="calls") == []

    store.close()


def test_get_graph_node_missing_returns_none():
    store = _mk_store()
    _seed_small_corpus(store)
    store.rebuild_hierarchy()
    assert store.get_graph_node("dir:nonexistent") is None
    node = store.get_graph_node("dir:a")
    assert node == {"id": "dir:a", "kind": "dir", "path": "a", "parent_id": "dir:."}
    store.close()


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------

def test_rebuild_idempotent():
    store = _mk_store()
    _seed_small_corpus(store)
    first = store.rebuild_hierarchy()
    second = store.rebuild_hierarchy()
    assert first == second
    assert store.count_graph_nodes() == first["nodes"]
    assert store.count_graph_edges() == first["edges"]
    store.close()


# ---------------------------------------------------------------------------
# File deletion is reflected on the next rebuild
# ---------------------------------------------------------------------------

def test_rebuild_after_file_deletion_drops_its_nodes_and_edges():
    store = _mk_store()
    _seed_small_corpus(store)
    store.rebuild_hierarchy()

    store.delete_file("a/b/c.py")
    store.commit()
    counts = store.rebuild_hierarchy()

    node_ids = {r["id"] for r in store._conn.execute("SELECT id FROM graph_nodes").fetchall()}
    assert node_ids == {"dir:.", "file:root.py"}
    assert counts["nodes"] == 2

    edge_pairs = {
        (r["from_id"], r["to_id"])
        for r in store._conn.execute("SELECT from_id, to_id FROM graph_edges").fetchall()
    }
    assert edge_pairs == {("dir:.", "file:root.py"), ("file:root.py", "r1")}
    assert store.get_graph_node("dir:a") is None
    assert store.get_graph_node("dir:a/b") is None
    assert store.get_graph_node("file:a/b/c.py") is None

    store.close()


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------

def test_rebuild_on_empty_store_is_all_zeros():
    store = _mk_store()
    counts = store.rebuild_hierarchy()
    assert counts == {"nodes": 0, "edges": 0}
    assert store.count_graph_nodes() == 0
    assert store.count_graph_edges() == 0
    store.close()


# ---------------------------------------------------------------------------
# Header/impl paired-file edges (component 2)
# ---------------------------------------------------------------------------

def _paired_edges(store: Store) -> set[tuple[str, str]]:
    rows = store._conn.execute(
        "SELECT from_id, to_id FROM graph_edges WHERE edge_type = 'paired'"
    ).fetchall()
    return {(r["from_id"], r["to_id"]) for r in rows}


def test_same_dir_header_impl_pair_both_directions():
    store = _mk_store()
    _seed_file(store, "widget.h", ["h1"])
    _seed_file(store, "widget.cpp", ["c1"])
    store.rebuild_hierarchy()

    assert _paired_edges(store) == {
        ("file:widget.h", "file:widget.cpp"),
        ("file:widget.cpp", "file:widget.h"),
    }
    store.close()


def test_different_dir_same_stem_not_paired():
    store = _mk_store()
    _seed_file(store, "include/widget.h", ["h1"])
    _seed_file(store, "src/widget.cpp", ["c1"])
    store.rebuild_hierarchy()

    assert _paired_edges(store) == set()
    store.close()


def test_impl_impl_not_paired():
    store = _mk_store()
    _seed_file(store, "widget.c", ["c1"])
    _seed_file(store, "widget.cpp", ["c2"])
    store.rebuild_hierarchy()

    assert _paired_edges(store) == set()
    store.close()


def test_multi_impl_stem_pairs_header_to_each():
    store = _mk_store()
    _seed_file(store, "widget.h", ["h1"])
    _seed_file(store, "widget.c", ["c1"])
    _seed_file(store, "widget.cpp", ["c2"])
    store.rebuild_hierarchy()

    assert _paired_edges(store) == {
        ("file:widget.h", "file:widget.c"),
        ("file:widget.c", "file:widget.h"),
        ("file:widget.h", "file:widget.cpp"),
        ("file:widget.cpp", "file:widget.h"),
    }
    store.close()


def test_pairing_counts_included_in_edge_count():
    store = _mk_store()
    _seed_file(store, "widget.h", ["h1"])
    _seed_file(store, "widget.cpp", ["c1"])
    counts = store.rebuild_hierarchy()

    # dir->file: ".->widget.h", ".->widget.cpp" (2)
    # file->chunk: widget.h->h1, widget.cpp->c1 (2)
    # paired: h<->cpp, both directions (2)
    assert counts["edges"] == 6
    store.close()


def test_pairing_extension_case_insensitive():
    store = _mk_store()
    _seed_file(store, "widget.H", ["h1"])
    _seed_file(store, "widget.CPP", ["c1"])
    store.rebuild_hierarchy()

    assert _paired_edges(store) == {
        ("file:widget.H", "file:widget.CPP"),
        ("file:widget.CPP", "file:widget.H"),
    }
    store.close()


def test_get_paired_files_batched_lookup():
    store = _mk_store()
    _seed_file(store, "widget.h", ["h1"])
    _seed_file(store, "widget.c", ["c1"])
    _seed_file(store, "widget.cpp", ["c2"])
    _seed_file(store, "lonely.py", ["p1"])
    store.rebuild_hierarchy()

    result = store.get_paired_files(["widget.h", "widget.c", "widget.cpp", "lonely.py"])
    assert set(result["widget.h"]) == {"widget.c", "widget.cpp"}
    assert result["widget.c"] == ["widget.h"]
    assert result["widget.cpp"] == ["widget.h"]
    assert "lonely.py" not in result
    store.close()


def test_get_paired_files_empty_input():
    store = _mk_store()
    assert store.get_paired_files([]) == {}
    store.close()


def test_rebuild_pairing_idempotent():
    store = _mk_store()
    _seed_file(store, "widget.h", ["h1"])
    _seed_file(store, "widget.cpp", ["c1"])
    first = store.rebuild_hierarchy()
    second = store.rebuild_hierarchy()
    assert first == second
    assert _paired_edges(store) == {
        ("file:widget.h", "file:widget.cpp"),
        ("file:widget.cpp", "file:widget.h"),
    }
    store.close()
