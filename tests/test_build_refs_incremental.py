"""Equivalence tests for the incremental build_refs update.

Property under test, same shape as test_build_neighbors.py's incremental k-NN
suite: applying a batch of file changes incrementally must produce a
chunk_refs table byte-identical (same (from_id, to_id, edge_type) set) to a
from-scratch full rebuild on the same post-mutation corpus, including the
corpus-global xlang cap (_MAX_CROSS_LANG_OCCURRENCES), whose boundary can flip
for a name whose definition count wasn't touched directly but changed because
sibling definitions were added or removed elsewhere.
"""
import logging
import tempfile
from collections import Counter

import pytest

import chonks.index.graph.refs as repomap
from chonks.core.edges import _MAX_CROSS_LANG_OCCURRENCES
from chonks.index.graph.refs import build_refs
from chonks.storage.store import Store


@pytest.fixture(autouse=True)
def _force_incremental_path(monkeypatch):
    """Opens both incremental gates so these tiny fixtures deterministically
    take the incremental path instead of silently falling back to a full
    rebuild (which would still pass but stop testing the algorithm).
    Gate-behavior tests below restore the real values explicitly."""
    monkeypatch.setattr(repomap, "_REFS_INCREMENTAL_MAX_FRACTION", 1.1)
    monkeypatch.setattr(repomap, "_REFS_INCREMENTAL_MAX_REFERENCER_FRACTION", 1.1)


def _mk_store() -> Store:
    return Store(tempfile.mktemp(suffix=".db"))


def _embed() -> list[float]:
    return [0.1, 0.2, 0.3, 0.4]


def _chunk(
    id, path, name=None, language="cpp", content="", metadata=None,
    start_line=1, end_line=1, chunk_type="function_definition",
):
    return {
        "id": id, "path": path, "language": language, "chunk_type": chunk_type,
        "name": name, "start_line": start_line, "end_line": end_line,
        "content": content, "metadata": metadata,
    }


def _insert(store: Store, chunks: list[dict]) -> None:
    if chunks:
        store.insert_chunks(chunks, [_embed() for _ in chunks])


def _edges(store: Store) -> set[tuple[str, str, str]]:
    return set(store.get_all_refs_typed())


def _indegree_table(store: Store) -> dict[tuple[str, str], int]:
    rows = store._conn.execute("SELECT chunk_id, edge_type, n FROM chunk_indegree").fetchall()
    return {(r["chunk_id"], r["edge_type"]): r["n"] for r in rows}


def _assert_indegree_consistent(store: Store) -> None:
    want = dict(Counter((to_id, edge_type) for (_from_id, to_id, edge_type) in _edges(store)))
    assert _indegree_table(store) == want


def _assert_incremental_ran(caplog) -> None:
    """build_refs' gate before calling _build_refs_incremental can silently
    fall back to a full rebuild, which also makes got == want true, but as a
    tautology. Assert the incremental-path success log line instead of the
    absence of the fallback one."""
    assert any(
        "chunk_refs incremental update:" in r.message for r in caplog.records
    ), (
        "build_refs did not take the incremental path — fell back to a full "
        "rebuild, making the equivalence assertion below a tautology. "
        f"log records: {[r.message for r in caplog.records]}"
    )


def _delete_tracked(store: Store, path: str, deleted_ids: set[str], deleted_names: set[str]) -> None:
    """Mirrors chunker.py's capture-before-delete pattern."""
    deleted_names.update(store.get_names_for_path(path))
    deleted_ids.update(store.delete_file(path))


def _full_rebuild_edges(
    final_chunks: list[dict], *, cap_mentions: bool = False,
) -> set[tuple[str, str, str]]:
    """Ground truth: a fresh store seeded with exactly the final chunk set,
    full-rebuilt."""
    ref = _mk_store()
    _insert(ref, final_chunks)
    build_refs(ref, cap_mentions=cap_mentions)
    edges = _edges(ref)
    _assert_indegree_consistent(ref)
    ref.close()
    return edges


# ---------------------------------------------------------------------------
# Basic add / delete / modify equivalence
# ---------------------------------------------------------------------------

def test_incremental_pure_add_matches_full_rebuild():
    store = _mk_store()
    initial = [
        _chunk("a", "a.cpp", name="driver", content="void driver() { helper(); }"),
    ]
    _insert(store, initial)
    build_refs(store)

    new = [_chunk("b", "b.cpp", name="helper", content="void helper() {}")]
    _insert(store, new)
    changed_ids = {"b"}

    build_refs(store, changed_ids=changed_ids, deleted_ids=set(), deleted_names=set())
    got = _edges(store)
    _assert_indegree_consistent(store)
    want = _full_rebuild_edges(initial + new)
    assert got == want
    assert ("a", "b", "mentions") in got


def test_incremental_pure_delete_matches_full_rebuild():
    """Deleting a definer drops referencer edges pointing at it, but leaves
    edges to a surviving co-definer of the same name intact."""
    store = _mk_store()
    initial = [
        _chunk("a", "a.cpp", name="driver", content="void driver() { helper(); }"),
        _chunk("b1", "b1.cpp", name="helper", content="void helper() {}"),
        _chunk("b2", "b2.cpp", name="helper", content="void helper() {}"),
    ]
    _insert(store, initial)
    build_refs(store)
    before = _edges(store)
    assert ("a", "b1", "mentions") in before
    assert ("a", "b2", "mentions") in before

    deleted_ids: set[str] = set()
    deleted_names: set[str] = set()
    _delete_tracked(store, "b1.cpp", deleted_ids, deleted_names)

    build_refs(store, changed_ids=set(), deleted_ids=deleted_ids, deleted_names=deleted_names)
    got = _edges(store)
    _assert_indegree_consistent(store)
    remaining = [c for c in initial if c["id"] != "b1"]
    want = _full_rebuild_edges(remaining)
    assert got == want
    assert ("a", "b1", "mentions") not in got
    assert ("a", "b2", "mentions") in got


def test_incremental_modify_reuses_same_id_for_untouched_sibling_chunk():
    """A re-parse commonly regenerates the same content-addressed id for
    chunks whose content/start_line didn't change; that id lands in both
    deleted_ids and changed_ids and must still keep its outgoing edges."""
    store = _mk_store()
    # Filler chunks sized so batch/total stays under the 20% incremental
    # threshold, so the incremental path (not the fallback) is under test.
    filler = [
        _chunk(f"filler{i}", f"filler{i}.cpp", name=f"filler{i}", content=f"void filler{i}() {{}}")
        for i in range(20)
    ]
    initial = [
        _chunk("foo", "driver.cpp", name="foo", content="void foo() { helper(); }"),
        _chunk("bar_old", "driver.cpp", name="bar", content="void bar() {}", start_line=3),
        _chunk("helper", "helper.cpp", name="helper", content="void helper() {}"),
    ] + filler
    _insert(store, initial)
    build_refs(store)
    before = _edges(store)
    assert ("foo", "helper", "mentions") in before

    deleted_ids: set[str] = set()
    deleted_names: set[str] = set()
    _delete_tracked(store, "driver.cpp", deleted_ids, deleted_names)

    # Re-parse of driver.cpp: 'foo' is byte-identical (same id regenerated),
    # only 'bar' actually changed content -> new id.
    new_driver = [
        _chunk("foo", "driver.cpp", name="foo", content="void foo() { helper(); }"),
        _chunk("bar_new", "driver.cpp", name="bar", content="void bar() { helper(); }", start_line=3),
    ]
    _insert(store, new_driver)
    changed_ids = {"foo", "bar_new"}

    build_refs(store, changed_ids=changed_ids, deleted_ids=deleted_ids, deleted_names=deleted_names)
    got = _edges(store)
    _assert_indegree_consistent(store)
    final_chunks = new_driver + [initial[2]] + filler
    want = _full_rebuild_edges(final_chunks)
    assert got == want
    assert ("foo", "helper", "mentions") in got


def test_incremental_modify_matches_full_rebuild():
    """delete-then-insert of the same logical file, the mutation shape
    chunker.py actually performs on a changed file."""
    store = _mk_store()
    initial = [
        _chunk("a", "a.cpp", name="driver", content="void driver() { helper(); }"),
        _chunk("b", "b.cpp", name="helper", content="void helper() {}"),
    ]
    _insert(store, initial)
    build_refs(store)

    deleted_ids: set[str] = set()
    deleted_names: set[str] = set()
    _delete_tracked(store, "b.cpp", deleted_ids, deleted_names)

    new_b = [_chunk("b2", "b.cpp", name="helperRenamed", content="void helperRenamed() {}")]
    _insert(store, new_b)
    changed_ids = {"b2"}

    build_refs(store, changed_ids=changed_ids, deleted_ids=deleted_ids, deleted_names=deleted_names)
    got = _edges(store)
    _assert_indegree_consistent(store)
    final_chunks = [initial[0]] + new_b
    want = _full_rebuild_edges(final_chunks)
    assert got == want


# ---------------------------------------------------------------------------
# Typed edges (calls/imports/inherits) reverse resolution
# ---------------------------------------------------------------------------

def test_incremental_typed_edge_added_when_target_appears():
    store = _mk_store()
    initial = [
        _chunk("a", "a.cpp", name="driver", content="void driver() { helper(); }",
               metadata={"calls": ["helper"], "imports": [], "inherits": []}),
    ]
    _insert(store, initial)
    build_refs(store)
    before = _edges(store)
    assert not any(t == "helper" for (_f, t, _e) in before)  # nothing to resolve to yet

    new = [_chunk("b", "b.cpp", name="helper", content="void helper() {}")]
    _insert(store, new)
    build_refs(store, changed_ids={"b"}, deleted_ids=set(), deleted_names=set())
    got = _edges(store)
    _assert_indegree_consistent(store)
    want = _full_rebuild_edges(initial + new)
    assert got == want
    assert ("a", "b", "calls") in got


# ---------------------------------------------------------------------------
# Receiver/arity call-site discriminator: the incremental path must run the
# same _discriminate_definers logic as _build_graph's bulk pass, proven
# byte-identical to a from-scratch full rebuild.
# ---------------------------------------------------------------------------

def _chained_fillers() -> list[dict]:
    """20 filler chunks calling each other in a cycle, so the initial build
    leaves real edges (count_refs() > 0). Plain no-op fillers build to zero
    edges, which fails the incremental-path gate and silently routes every
    later build_refs call back through the full-rebuild path."""
    return [
        _chunk(
            f"filler{i}", f"filler{i}.py", name=f"filler{i}", language="python",
            content=f"def filler{i}():\n    filler{(i + 1) % 20}()",
            metadata={"calls": [{"name": f"filler{(i + 1) % 20}", "receiver": None, "arity": 0}]},
        )
        for i in range(20)
    ]


def test_incremental_calls_discriminator_matches_full_rebuild(caplog):
    """A name collision (two 'parse' definers) resolved via the calls-list
    receiver fingerprint must land on the same definer whether the batch
    introducing the definers is applied incrementally or as a full rebuild."""
    store = _mk_store()
    filler = _chained_fillers()
    initial = [
        _chunk("caller", "driver.py", name="driver", language="python",
               content="def driver():\n    self.parser.parse(1)",
               metadata={"calls": [{"name": "parse", "receiver": "parser", "arity": 1}]}),
    ] + filler
    _insert(store, initial)
    build_refs(store)
    assert store.count_refs() > 0 and store.has_indegree(), (
        "initial build produced no edges — the discriminator test below "
        "would silently fall back to a full rebuild instead of exercising "
        "the incremental path"
    )

    new = [
        _chunk("parser_cls", "parser.py", name="Parser", language="python",
               chunk_type="class_definition",
               content="class Parser:\n    def parse(self, a): pass"),
        _chunk("lexer_cls", "lexer.py", name="Lexer", language="python",
               chunk_type="class_definition",
               content="class Lexer:\n    def parse(self, a): pass"),
    ]
    symbol_rows = [
        {"path": "parser.py", "name": "parse", "kind": "function_definition",
         "language": "python", "start_line": 2, "end_line": 2, "chunk_id": "parser_cls"},
        {"path": "lexer.py", "name": "parse", "kind": "function_definition",
         "language": "python", "start_line": 2, "end_line": 2, "chunk_id": "lexer_cls"},
    ]
    _insert(store, new)
    store.insert_symbols(symbol_rows)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids={"parser_cls", "lexer_cls"}, deleted_ids=set(), deleted_names=set())
    _assert_incremental_ran(caplog)

    got = _edges(store)
    _assert_indegree_consistent(store)

    ref = _mk_store()
    _insert(ref, initial + new)
    ref.insert_symbols(symbol_rows)
    build_refs(ref)
    want = _edges(ref)
    _assert_indegree_consistent(ref)
    ref.close()

    assert got == want
    assert ("caller", "parser_cls", "calls") in got
    assert ("caller", "lexer_cls", "calls") not in got


def test_incremental_calls_discriminator_fallback_matches_full_rebuild(caplog):
    """When the receiver matches nothing and arity rules out every
    candidate, both paths fall back to the same full fan-out (recall
    preserved by construction, see _discriminate_definers)."""
    store = _mk_store()
    filler = _chained_fillers()
    initial = [
        _chunk("caller", "driver.py", name="driver", language="python",
               content="def driver():\n    nope.parse(999)",
               metadata={"calls": [{"name": "parse", "receiver": "nope", "arity": 999}]}),
    ] + filler
    _insert(store, initial)
    build_refs(store)
    assert store.count_refs() > 0 and store.has_indegree(), (
        "initial build produced no edges — the discriminator test below "
        "would silently fall back to a full rebuild instead of exercising "
        "the incremental path"
    )

    new = [
        _chunk("parser_cls", "parser.py", name="Parser", language="python",
               chunk_type="class_definition",
               content="class Parser:\n    def parse(self, a): pass"),
        _chunk("lexer_cls", "lexer.py", name="Lexer", language="python",
               chunk_type="class_definition",
               content="class Lexer:\n    def parse(self, a): pass"),
    ]
    symbol_rows = [
        {"path": "parser.py", "name": "parse", "kind": "function_definition",
         "language": "python", "start_line": 2, "end_line": 2, "chunk_id": "parser_cls"},
        {"path": "lexer.py", "name": "parse", "kind": "function_definition",
         "language": "python", "start_line": 2, "end_line": 2, "chunk_id": "lexer_cls"},
    ]
    _insert(store, new)
    store.insert_symbols(symbol_rows)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids={"parser_cls", "lexer_cls"}, deleted_ids=set(), deleted_names=set())
    _assert_incremental_ran(caplog)

    got = _edges(store)
    _assert_indegree_consistent(store)

    ref = _mk_store()
    _insert(ref, initial + new)
    ref.insert_symbols(symbol_rows)
    build_refs(ref)
    want = _edges(ref)
    _assert_indegree_consistent(ref)
    ref.close()

    assert got == want
    assert ("caller", "parser_cls", "calls") in got
    assert ("caller", "lexer_cls", "calls") in got


# ---------------------------------------------------------------------------
# Bare-alias for qualified names: the incremental twin of
# test_build_graph_qualified_definer_reachable_via_bare_alias (test_repomap.py).
# ---------------------------------------------------------------------------

def test_incremental_qualified_definer_add_reachable_via_bare_alias():
    store = _mk_store()
    initial = [
        _chunk("a", "a.cpp", name="driver", content="void driver() { applyStep(); }"),
    ]
    _insert(store, initial)
    build_refs(store)

    new = [_chunk("b", "b.cpp", name=None, content="class Widget {};")]
    _insert(store, new)
    store.insert_symbols([
        {"path": "b.cpp", "name": "Widget::applyStep", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "b"},
    ])
    build_refs(store, changed_ids={"b"}, deleted_ids=set(), deleted_names=set())
    got = _edges(store)
    _assert_indegree_consistent(store)

    ref = _mk_store()
    _insert(ref, initial + new)
    ref.insert_symbols([
        {"path": "b.cpp", "name": "Widget::applyStep", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "b"},
    ])
    build_refs(ref)
    want = _edges(ref)
    _assert_indegree_consistent(ref)
    ref.close()

    assert got == want
    assert ("a", "b", "mentions") in got


def test_incremental_deleting_qualified_definer_drops_only_its_own_bare_alias_edge():
    """A deleted chunk defining a qualified name must invalidate the bare
    alias too, while a surviving co-definer of the same bare alias keeps
    its edge."""
    store = _mk_store()
    a = _chunk("a", "foo.cpp", name=None, content="class Foo {};")
    b = _chunk("b", "baz.cpp", name=None, content="class Baz {};")
    r = _chunk("r", "r.cpp", name="driver", content="void driver() { bar(); }")
    _insert(store, [a, b, r])
    store.insert_symbols([
        {"path": "foo.cpp", "name": "Foo::bar", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "a"},
        {"path": "baz.cpp", "name": "Baz::bar", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "b"},
    ])
    build_refs(store)
    before = _edges(store)
    assert ("r", "a", "mentions") in before
    assert ("r", "b", "mentions") in before

    deleted_ids: set[str] = set()
    deleted_names: set[str] = set()
    _delete_tracked(store, "foo.cpp", deleted_ids, deleted_names)

    build_refs(store, changed_ids=set(), deleted_ids=deleted_ids, deleted_names=deleted_names)
    got = _edges(store)
    _assert_indegree_consistent(store)

    ref = _mk_store()
    _insert(ref, [b, r])
    ref.insert_symbols([
        {"path": "baz.cpp", "name": "Baz::bar", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "b"},
    ])
    build_refs(ref)
    want = _edges(ref)
    _assert_indegree_consistent(ref)
    ref.close()

    assert got == want
    assert ("r", "a", "mentions") not in got
    assert ("r", "b", "mentions") in got


def test_incremental_referencer_edit_preserves_edges_to_qualified_aliases_sharing_bare_name():
    """No chunk here is literally named "bsearch"; recomputing R's edges
    from scratch after an unrelated edit to R must still resolve both
    qualified aliases via the reverse bare -> qualified lookup
    (_bare_alias_definers), not just exact-match names."""
    store = _mk_store()
    q1 = _chunk("q1", "array.cpp", name=None, content="class Array {};")
    q2 = _chunk("q2", "vector.cpp", name=None, content="class Vector {};")
    r = _chunk("r", "r.cpp", name="driver", content="void driver() { bsearch(x); }")
    _insert(store, [q1, q2, r])
    store.insert_symbols([
        {"path": "array.cpp", "name": "Array::bsearch", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "q1"},
        {"path": "vector.cpp", "name": "Vector::bsearch", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "q2"},
    ])
    build_refs(store)
    before = _edges(store)
    assert ("r", "q1", "mentions") in before
    assert ("r", "q2", "mentions") in before

    deleted_ids: set[str] = set()
    deleted_names: set[str] = set()
    _delete_tracked(store, "r.cpp", deleted_ids, deleted_names)
    r2 = _chunk("r2", "r.cpp", name="driver",
                content="void driver() { /* edited */ bsearch(x); }")
    _insert(store, [r2])

    build_refs(store, changed_ids={"r2"}, deleted_ids=deleted_ids, deleted_names=deleted_names)
    got = _edges(store)
    _assert_indegree_consistent(store)

    ref = _mk_store()
    _insert(ref, [q1, q2, r2])
    ref.insert_symbols([
        {"path": "array.cpp", "name": "Array::bsearch", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "q1"},
        {"path": "vector.cpp", "name": "Vector::bsearch", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "q2"},
    ])
    build_refs(ref)
    want = _edges(ref)
    _assert_indegree_consistent(ref)
    ref.close()

    assert got == want
    assert ("r2", "q1", "mentions") in got
    assert ("r2", "q2", "mentions") in got


def _make_definers(n: int) -> list[dict]:
    return [
        _chunk(f"d{i}", f"d{i}.cpp", name=None, content=f"class C{i} {{}};")
        for i in range(n)
    ]


def _make_symbols(n: int) -> list[dict]:
    return [
        {"path": f"d{i}.cpp", "name": f"C{i}::make", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": f"d{i}"}
        for i in range(n)
    ]


def _d_edge_count(edges: set[tuple[str, str, str]]) -> int:
    return sum(1 for (_f, t, _e) in edges if t.startswith("d"))


def test_incremental_bare_alias_aggregate_at_cap_kept_on_referencer_edit():
    """_MAX_CROSS_LANG_OCCURRENCES qualified definers, each with one definer,
    all alias to the bare "make" key, sitting exactly at the cap. An
    unrelated edit to referencer R must still produce edges to every
    definer, proving the aggregate is evaluated fresh, not cached."""
    n = _MAX_CROSS_LANG_OCCURRENCES
    store = _mk_store()
    defs = _make_definers(n)
    r = _chunk("r", "r.cpp", name="driver", content="void driver() { make(); }")
    _insert(store, defs + [r])
    store.insert_symbols(_make_symbols(n))
    build_refs(store)
    before = _edges(store)
    assert _d_edge_count(before) == n

    deleted_ids: set[str] = set()
    deleted_names: set[str] = set()
    _delete_tracked(store, "r.cpp", deleted_ids, deleted_names)
    r2 = _chunk("r2", "r.cpp", name="driver",
                content="void driver() { /* edited */ make(); }")
    _insert(store, [r2])
    build_refs(store, changed_ids={"r2"}, deleted_ids=deleted_ids, deleted_names=deleted_names)
    got = _edges(store)
    _assert_indegree_consistent(store)

    ref = _mk_store()
    _insert(ref, defs + [r2])
    ref.insert_symbols(_make_symbols(n))
    build_refs(ref)
    want = _edges(ref)
    _assert_indegree_consistent(ref)
    ref.close()

    assert got == want
    assert _d_edge_count(got) == n


def test_incremental_bare_alias_aggregate_over_cap_dropped_on_referencer_edit():
    """One more qualified definer pushes the "make" bare-alias aggregate
    over the cap: a from-scratch rebuild then drops all aliases for that
    key. An unrelated referencer edit must reproduce this, not simply keep
    the previously-built, now-stale edges."""
    n = _MAX_CROSS_LANG_OCCURRENCES + 1
    store = _mk_store()
    defs = _make_definers(n)
    r = _chunk("r", "r.cpp", name="driver", content="void driver() { make(); }")
    _insert(store, defs + [r])
    store.insert_symbols(_make_symbols(n))
    build_refs(store)
    before = _edges(store)
    assert _d_edge_count(before) == 0

    deleted_ids: set[str] = set()
    deleted_names: set[str] = set()
    _delete_tracked(store, "r.cpp", deleted_ids, deleted_names)
    r2 = _chunk("r2", "r.cpp", name="driver",
                content="void driver() { /* edited */ make(); }")
    _insert(store, [r2])
    build_refs(store, changed_ids={"r2"}, deleted_ids=deleted_ids, deleted_names=deleted_names)
    got = _edges(store)
    _assert_indegree_consistent(store)

    ref = _mk_store()
    _insert(ref, defs + [r2])
    ref.insert_symbols(_make_symbols(n))
    build_refs(ref)
    want = _edges(ref)
    _assert_indegree_consistent(ref)
    ref.close()

    assert got == want
    assert _d_edge_count(got) == 0


def test_incremental_bare_alias_reappears_after_deletion_drops_aggregate_under_cap():
    """Starting over cap (aliases dropped, per the test above), deleting one
    definer brings the aggregate back to exactly the cap: the bare alias
    must reappear for the remaining definers. Here the definer side
    changes, not the referencer, so deleted_names must carry the alias
    correctly through the deletion path too."""
    n = _MAX_CROSS_LANG_OCCURRENCES + 1
    store = _mk_store()
    defs = _make_definers(n)
    r = _chunk("r", "r.cpp", name="driver", content="void driver() { make(); }")
    _insert(store, defs + [r])
    store.insert_symbols(_make_symbols(n))
    build_refs(store)
    before = _edges(store)
    assert _d_edge_count(before) == 0

    deleted_ids: set[str] = set()
    deleted_names: set[str] = set()
    _delete_tracked(store, "d0.cpp", deleted_ids, deleted_names)
    build_refs(store, changed_ids=set(), deleted_ids=deleted_ids, deleted_names=deleted_names)
    got = _edges(store)
    _assert_indegree_consistent(store)

    remaining_defs = defs[1:]
    ref = _mk_store()
    _insert(ref, remaining_defs + [r])
    ref.insert_symbols(_make_symbols(n)[1:])
    build_refs(ref)
    want = _edges(ref)
    _assert_indegree_consistent(ref)
    ref.close()

    assert got == want
    assert _d_edge_count(got) == n - 1
    assert not any(t == "d0" for (_f, t, _e) in got)


# ---------------------------------------------------------------------------
# xlang cap-boundary crossing
# ---------------------------------------------------------------------------

def _xlang_definer(i: int, lang: str) -> dict:
    return _chunk(f"x{i}", f"x{i}.src", name="Shared", language=lang,
                  content=f"void notUsed_{i}() {{}}")


def test_incremental_xlang_cap_crossing_upward_suppresses_edges():
    """8 definers (cap boundary) across 2 languages produce xlang edges.
    Adding a 9th pushes the name over the cap, so a from-scratch rebuild
    would emit no xlang edges at all; the incremental update must actively
    remove the previously-valid edges to match."""
    assert _MAX_CROSS_LANG_OCCURRENCES == 8
    store = _mk_store()
    initial = [_xlang_definer(i, "cpp" if i % 2 == 0 else "csharp") for i in range(8)]
    _insert(store, initial)
    build_refs(store)
    before = _edges(store)
    assert any(e == "xlang" for (_f, _t, e) in before), "fixture didn't produce xlang edges"

    new = [_xlang_definer(8, "cpp")]
    _insert(store, new)
    build_refs(store, changed_ids={"x8"}, deleted_ids=set(), deleted_names=set())
    got = _edges(store)
    _assert_indegree_consistent(store)
    want = _full_rebuild_edges(initial + new)
    assert got == want
    assert not any(e == "xlang" for (_f, _t, e) in got), "cap crossing must suppress all xlang edges for the name"


def test_incremental_xlang_cap_crossing_downward_restores_edges():
    """9 definers (over cap) produce no xlang edges. Deleting one back down
    to 8 must restore the full xlang pair set a from-scratch rebuild would
    produce."""
    store = _mk_store()
    initial = [_xlang_definer(i, "cpp" if i % 2 == 0 else "csharp") for i in range(9)]
    _insert(store, initial)
    build_refs(store)
    before = _edges(store)
    assert not any(e == "xlang" for (_f, _t, e) in before), "fixture should start over the cap"

    deleted_ids: set[str] = set()
    deleted_names: set[str] = set()
    _delete_tracked(store, "x8.src", deleted_ids, deleted_names)

    build_refs(store, changed_ids=set(), deleted_ids=deleted_ids, deleted_names=deleted_names)
    got = _edges(store)
    _assert_indegree_consistent(store)
    remaining = [c for c in initial if c["id"] != "x8"]
    want = _full_rebuild_edges(remaining)
    assert got == want
    assert any(e == "xlang" for (_f, _t, e) in got), "dropping below the cap must restore xlang edges"


# ---------------------------------------------------------------------------
# Step-3 sibling-expansion fixpoint: a second-hop definer id, only reachable
# through a name harvested off ANOTHER id's siblings, must have its own name
# harvested too, or its xlang edge gets wiped but never rebuilt.
# ---------------------------------------------------------------------------

def test_incremental_sibling_expansion_fixpoint_preserves_second_hop_xlang_edge():
    """Chain: editing P touches name Alpha + P's folded symbol Beta. Beta is
    also folded onto Q (own chunk name Gamma), so Q enters
    affected_target_ids directly (hop 1). A single sibling-expansion pass
    harvests Q's name Gamma and resolves it to R, but R is folded with
    symbol name Gamma too (its own chunk name is Delta), so R only enters
    affected_target_ids after that harvest already ran, and its own name
    Delta (paired in an xlang edge with T) is never re-harvested by a
    single pass. The fixpoint must re-harvest after each new id lands in
    affected_target_ids so the R<->T edge survives, matching a full rebuild."""
    store = _mk_store()
    p = _chunk("p", "P.cs", name="Alpha", language="c_sharp")
    q = _chunk("q", "Q.cpp", name="Gamma", language="cpp")
    r = _chunk("r", "R.cs", name="Delta", language="c_sharp")
    t = _chunk("t", "T.cpp", name="Delta", language="cpp")
    _insert(store, [p, q, r, t])
    store.insert_symbols([
        {"path": "P.cs", "name": "Beta", "kind": "method", "language": "c_sharp",
         "start_line": 1, "end_line": 1, "chunk_id": "p"},
        {"path": "Q.cpp", "name": "Beta", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "q"},
        {"path": "R.cs", "name": "Gamma", "kind": "method", "language": "c_sharp",
         "start_line": 1, "end_line": 1, "chunk_id": "r"},
    ])
    build_refs(store)
    before = _edges(store)
    assert any(e == "xlang" and {f, tgt} == {"r", "t"} for (f, tgt, e) in before), \
        "fixture didn't produce the R<->T xlang edge to begin with"

    deleted_ids: set[str] = set()
    deleted_names: set[str] = set()
    _delete_tracked(store, "P.cs", deleted_ids, deleted_names)

    p2 = _chunk("p2", "P.cs", name="Alpha", language="c_sharp", content="/* edited */")
    _insert(store, [p2])
    store.insert_symbols([
        {"path": "P.cs", "name": "Beta", "kind": "method", "language": "c_sharp",
         "start_line": 1, "end_line": 1, "chunk_id": "p2"},
    ])

    build_refs(store, changed_ids={"p2"}, deleted_ids=deleted_ids, deleted_names=deleted_names)
    got = _edges(store)
    _assert_indegree_consistent(store)

    ref = _mk_store()
    _insert(ref, [p2, q, r, t])
    ref.insert_symbols([
        {"path": "P.cs", "name": "Beta", "kind": "method", "language": "c_sharp",
         "start_line": 1, "end_line": 1, "chunk_id": "p2"},
        {"path": "Q.cpp", "name": "Beta", "kind": "method", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "q"},
        {"path": "R.cs", "name": "Gamma", "kind": "method", "language": "c_sharp",
         "start_line": 1, "end_line": 1, "chunk_id": "r"},
    ])
    build_refs(ref)
    want = _edges(ref)
    _assert_indegree_consistent(ref)
    ref.close()

    assert got == want
    assert any(e == "xlang" and {f, tgt} == {"r", "t"} for (f, tgt, e) in got), \
        "R<->T xlang edge must survive the incremental update (second-hop definer)"


# ---------------------------------------------------------------------------
# cap_mentions equivalence
# ---------------------------------------------------------------------------

def _mentions_definer(i: int) -> dict:
    return _chunk(f"m{i}", f"m{i}.cpp", name="Update", content="void notUsed() {}")


def test_incremental_cap_mentions_matches_full_rebuild_over_cap():
    """With cap_mentions enabled, adding a definer that pushes a name over
    the cap must suppress the caller's existing mentions edge, not just
    skip adding a new one, matching a from-scratch capped rebuild."""
    assert _MAX_CROSS_LANG_OCCURRENCES == 8
    store = _mk_store()
    initial = [_mentions_definer(i) for i in range(8)]
    initial.append(
        _chunk("caller", "caller.cpp", name="driver", content="void driver() { Update(); }"),
    )
    _insert(store, initial)
    build_refs(store, cap_mentions=True)
    before = _edges(store)
    assert ("caller", "m0", "mentions") in before, "fixture didn't produce a mentions edge at the boundary"

    new = [_mentions_definer(8)]
    _insert(store, new)
    build_refs(store, changed_ids={"m8"}, deleted_ids=set(), deleted_names=set(), cap_mentions=True)
    got = _edges(store)
    _assert_indegree_consistent(store)
    want = _full_rebuild_edges(initial + new, cap_mentions=True)
    assert got == want
    assert not any(e == "mentions" for (_f, _t, e) in got), \
        "cap crossing must suppress all mentions edges for the over-cap name"


def test_incremental_cap_mentions_flag_off_keeps_uncapped_edges():
    """The same over-cap fixture without cap_mentions must keep the full,
    uncapped mentions-edge set: the flag defaults off and must not change
    existing behaviour."""
    store = _mk_store()
    initial = [_mentions_definer(i) for i in range(8)]
    initial.append(
        _chunk("caller", "caller.cpp", name="driver", content="void driver() { Update(); }"),
    )
    _insert(store, initial)
    build_refs(store)

    new = [_mentions_definer(8)]
    _insert(store, new)
    build_refs(store, changed_ids={"m8"}, deleted_ids=set(), deleted_names=set())
    got = _edges(store)
    _assert_indegree_consistent(store)
    want = _full_rebuild_edges(initial + new, cap_mentions=False)
    assert got == want
    assert sum(1 for (_f, _t, e) in got if e == "mentions") == 9, \
        "default (uncapped) behaviour must be unaffected"


# ---------------------------------------------------------------------------
# Fallback to full rebuild above the churn threshold
# ---------------------------------------------------------------------------

def test_incremental_falls_back_to_full_rebuild_over_threshold(monkeypatch):
    monkeypatch.setattr(repomap, "_REFS_INCREMENTAL_MAX_FRACTION", 0.20)
    store = _mk_store()
    initial = [_chunk(f"c{i}", f"c{i}.cpp", name=f"fn{i}", content=f"void fn{i}() {{}}")
               for i in range(20)]
    _insert(store, initial)
    build_refs(store)

    called = []
    original = repomap._build_refs_incremental

    def _spy(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    repomap._build_refs_incremental = _spy
    try:
        # 10 out of 20 changed -> 50% > the 20% threshold -> full rebuild.
        changed = {f"c{i}" for i in range(10)}
        build_refs(store, changed_ids=changed, deleted_ids=set(), deleted_names=set())
    finally:
        repomap._build_refs_incremental = original
    assert not called
    _assert_indegree_consistent(store)


def test_incremental_aborts_on_referencer_fanout_and_full_rebuild_matches(monkeypatch):
    """A batch small enough to pass the input-size gate can still touch a
    name so widely referenced that recomputing every referencer costs more
    than a full rebuild; the incremental path must detect that after
    computing the fan-out and fall through to the full rebuild."""
    monkeypatch.setattr(repomap, "_REFS_INCREMENTAL_MAX_REFERENCER_FRACTION", 0.25)
    store = _mk_store()
    # 1 hub + 19 spokes, every spoke references the hub -> changing just the
    # hub (5% batch, passes the 20% input gate) fans out to 19/20 = 95% of the
    # corpus, over the 25% referencer gate.
    initial = [_chunk("hub", "hub.cpp", name="hubfn", content="void hubfn() {}")]
    initial += [
        _chunk(f"s{i}", f"s{i}.cpp", name=f"fn{i}", content=f"void fn{i}() {{ hubfn(); }}")
        for i in range(19)
    ]
    _insert(store, initial)
    build_refs(store)

    results = []
    original = repomap._build_refs_incremental

    def _spy(*args, **kwargs):
        results.append(original(*args, **kwargs))
        return results[-1]

    repomap._build_refs_incremental = _spy
    try:
        build_refs(store, changed_ids={"hub"}, deleted_ids=set(), deleted_names=set())
    finally:
        repomap._build_refs_incremental = original

    assert results == [None], "incremental path should run, then abort on fan-out"
    assert _edges(store) == _full_rebuild_edges(initial)
    # The full rebuild's clear_refs + save_indegree subsumes any partial
    # state left by the aborted incremental attempt.
    _assert_indegree_consistent(store)


def test_incremental_empty_changed_and_deleted_is_noop():
    store = _mk_store()
    initial = [
        _chunk("a", "a.cpp", name="driver", content="void driver() { helper(); }"),
        _chunk("b", "b.cpp", name="helper", content="void helper() {}"),
    ]
    _insert(store, initial)
    build_refs(store)
    before = _edges(store)

    written = build_refs(store, changed_ids=set(), deleted_ids=set(), deleted_names=set())
    assert written == 0
    assert _edges(store) == before
    _assert_indegree_consistent(store)


def test_unnamed_definer_chunk_discriminated_identically_on_both_paths():
    """A symbol can resolve to an unnamed chunk (folded member in a residue
    chunk). Bulk builds id_to_chunk from named chunks only, so that target
    has no content there and arity is always compatible; incremental
    fetches it by id and can exclude it. Both paths must agree, on the
    strong verdict: parse(self, a, b, c) is arity-incompatible with
    parse(1), so the anon edge demotes to mentions on both."""
    def corpus():
        filler = [_chunk(f"filler{i}", f"filler{i}.py", name=f"filler{i}", language="python",
                  content=f"def filler{i}():\n    filler{(i+1)%20}()",
                  metadata={"calls": [{"name": f"filler{(i+1)%20}", "receiver": None, "arity": 0}]})
                  for i in range(20)]
        caller = _chunk("caller", "driver.py", name="driver", language="python",
                        content="def driver():\n    parse(1)",
                        metadata={"calls": [{"name": "parse", "receiver": None, "arity": 1}]})
        named = _chunk("named_cls", "parser.py", name="Parser", language="python",
                       chunk_type="class_definition",
                       content="class Parser:\n    def parse(self, a): pass")
        anon = _chunk("anon", "residue.py", name=None, language="python",
                      content="def parse(self, a, b, c): pass")
        return filler, caller, named, anon

    syms = [{"path": "parser.py", "name": "parse", "kind": "function_definition",
             "language": "python", "start_line": 2, "end_line": 2, "chunk_id": "named_cls"},
            {"path": "residue.py", "name": "parse", "kind": "function_definition",
             "language": "python", "start_line": 1, "end_line": 1, "chunk_id": "anon"}]

    filler, caller, named, anon = corpus()
    ref = _mk_store()
    _insert(ref, [caller] + filler + [named, anon])
    ref.insert_symbols(syms)
    build_refs(ref)
    want = sorted(e for e in _edges(ref) if e[0] == "caller")

    filler, caller, named, anon = corpus()
    st = _mk_store()
    _insert(st, [caller] + filler)
    build_refs(st)
    assert st.count_refs() > 0 and st.has_indegree()
    _insert(st, [named, anon])
    st.insert_symbols(syms)
    build_refs(st, changed_ids={"named_cls", "anon"}, deleted_ids=set(), deleted_names=set())
    got = sorted(e for e in _edges(st) if e[0] == "caller")

    assert got == want
    assert ("caller", "anon", "mentions") in want
    assert ("caller", "named_cls", "calls") in want
