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


def _delete_tracked(store: Store, path: str, deleted_ids: set[str], deleted_defs: dict) -> None:
    """Mirrors chonks/index/pipeline.py's capture-before-delete pattern."""
    deleted_defs.update(store.get_definitions([path]))
    deleted_ids.update(store.delete_file(path))


def _full_rebuild_edges(
    final_chunks: list[dict], *, cap_mentions: bool = False, associated_top_frac: float = 0.0,
) -> set[tuple[str, str, str]]:
    """Ground truth: a fresh store seeded with exactly the final chunk set,
    full-rebuilt."""
    ref = _mk_store()
    _insert(ref, final_chunks)
    build_refs(ref, cap_mentions=cap_mentions, associated_top_frac=associated_top_frac)
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

    build_refs(store, changed_ids=changed_ids, deleted_ids=set(), deleted_defs={})
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
    deleted_defs: dict = {}
    _delete_tracked(store, "b1.cpp", deleted_ids, deleted_defs)

    build_refs(store, changed_ids=set(), deleted_ids=deleted_ids, deleted_defs=deleted_defs)
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
    deleted_ids and changed_ids and must still keep its outgoing edges. A
    symbol row the re-parse adds to such a chunk still counts as a change."""
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
        _chunk("user", "user.cpp", name="user", content="void user() { fooAlias(); }"),
    ] + filler
    _insert(store, initial)
    build_refs(store)
    before = _edges(store)
    assert ("foo", "helper", "mentions") in before

    deleted_ids: set[str] = set()
    deleted_defs: dict = {}
    _delete_tracked(store, "driver.cpp", deleted_ids, deleted_defs)

    # Re-parse of driver.cpp: 'foo' is byte-identical (same id regenerated),
    # only 'bar' actually changed content -> new id.
    new_driver = [
        _chunk("foo", "driver.cpp", name="foo", content="void foo() { helper(); }"),
        _chunk("bar_new", "driver.cpp", name="bar", content="void bar() { helper(); }", start_line=3),
    ]
    _insert(store, new_driver)
    alias = {"path": "driver.cpp", "name": "fooAlias", "kind": "function_definition", "language": "cpp",
             "start_line": 1, "end_line": 1, "chunk_id": "foo"}
    store.insert_symbols([alias])
    changed_ids = {"foo", "bar_new"}

    build_refs(store, changed_ids=changed_ids, deleted_ids=deleted_ids, deleted_defs=deleted_defs)
    got = _edges(store)
    _assert_indegree_consistent(store)
    ref = _mk_store()
    _insert(ref, new_driver + initial[2:])
    ref.insert_symbols([alias])
    build_refs(ref)
    assert got == _edges(ref)
    assert ("foo", "helper", "mentions") in got
    assert ("user", "foo", "mentions") in got


def test_incremental_modify_matches_full_rebuild():
    """delete-then-insert of the same logical file, the mutation shape
    chonks/index/pipeline.py actually performs on a changed file."""
    store = _mk_store()
    initial = [
        _chunk("a", "a.cpp", name="driver", content="void driver() { helper(); }"),
        _chunk("b", "b.cpp", name="helper", content="void helper() {}"),
    ]
    _insert(store, initial)
    build_refs(store)

    deleted_ids: set[str] = set()
    deleted_defs: dict = {}
    _delete_tracked(store, "b.cpp", deleted_ids, deleted_defs)

    new_b = [_chunk("b2", "b.cpp", name="helperRenamed", content="void helperRenamed() {}")]
    _insert(store, new_b)
    changed_ids = {"b2"}

    build_refs(store, changed_ids=changed_ids, deleted_ids=deleted_ids, deleted_defs=deleted_defs)
    got = _edges(store)
    _assert_indegree_consistent(store)
    final_chunks = [initial[0]] + new_b
    want = _full_rebuild_edges(final_chunks)
    assert got == want


# ---------------------------------------------------------------------------
# Typed edges (calls/imports/inherits) reverse resolution
# ---------------------------------------------------------------------------

def test_incremental_typed_edges_reach_new_targets_and_bases_skip_constructors(caplog):
    store = _mk_store()
    initial = [
        _chunk("a", "a.cpp", name="driver", content="void driver() { helper(); }",
               metadata={"calls": ["helper"], "imports": [], "inherits": []}),
        _chunk("d", "derived.h", name="Derived", chunk_type="class_specifier",
               content="class Derived : public Base, public Box<int> {};",
               metadata={"calls": [], "imports": [], "inherits": ["Base", "Box"]}),
    ] + _chained_fillers()
    _insert(store, initial)
    build_refs(store)
    before = _edges(store)
    assert not any(t == "helper" for (_f, t, _e) in before)  # nothing to resolve to yet

    new = [
        _chunk("b", "b.cpp", name="helper", content="void helper() {}"),
        _chunk("base", "base.h", name="Base", chunk_type="class_specifier", content="class Base {\n  Base();\n};"),
        _chunk("ctor", "base.cpp", name="Base::Base", content="Base::Base() {}"),
        _chunk("box", "box.h", name="Box", chunk_type="template_declaration",
               content="template <class T>\nclass Box {};"),
    ]
    symbol_rows = [
        {"path": "base.h", "name": "Base", "kind": "class_specifier", "language": "cpp",
         "start_line": 1, "end_line": 3, "chunk_id": "base"},
        {"path": "base.cpp", "name": "Base::Base", "kind": "function_definition", "language": "cpp",
         "start_line": 1, "end_line": 1, "chunk_id": "ctor"},
        {"path": "box.h", "name": "Box", "kind": "class_specifier", "language": "cpp",
         "start_line": 2, "end_line": 2, "chunk_id": "box"},
    ]
    _insert(store, new)
    store.insert_symbols(symbol_rows)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids={c["id"] for c in new}, deleted_ids=set(), deleted_defs={})
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
    assert ("a", "b", "calls") in got
    assert {t for (f, t, e) in got if f == "d" and e == "inherits"} == {"base", "box"}


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
        build_refs(store, changed_ids={"parser_cls", "lexer_cls"}, deleted_ids=set(), deleted_defs={})
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
        build_refs(store, changed_ids={"parser_cls", "lexer_cls"}, deleted_ids=set(), deleted_defs={})
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
    build_refs(store, changed_ids={"b"}, deleted_ids=set(), deleted_defs={})
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
    deleted_defs: dict = {}
    _delete_tracked(store, "foo.cpp", deleted_ids, deleted_defs)

    build_refs(store, changed_ids=set(), deleted_ids=deleted_ids, deleted_defs=deleted_defs)
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
    deleted_defs: dict = {}
    _delete_tracked(store, "r.cpp", deleted_ids, deleted_defs)
    r2 = _chunk("r2", "r.cpp", name="driver",
                content="void driver() { /* edited */ bsearch(x); }")
    _insert(store, [r2])

    build_refs(store, changed_ids={"r2"}, deleted_ids=deleted_ids, deleted_defs=deleted_defs)
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
    deleted_defs: dict = {}
    _delete_tracked(store, "r.cpp", deleted_ids, deleted_defs)
    r2 = _chunk("r2", "r.cpp", name="driver",
                content="void driver() { /* edited */ make(); }")
    _insert(store, [r2])
    build_refs(store, changed_ids={"r2"}, deleted_ids=deleted_ids, deleted_defs=deleted_defs)
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
    deleted_defs: dict = {}
    _delete_tracked(store, "r.cpp", deleted_ids, deleted_defs)
    r2 = _chunk("r2", "r.cpp", name="driver",
                content="void driver() { /* edited */ make(); }")
    _insert(store, [r2])
    build_refs(store, changed_ids={"r2"}, deleted_ids=deleted_ids, deleted_defs=deleted_defs)
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
    changes, not the referencer, so deleted_defs must carry the alias
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
    deleted_defs: dict = {}
    _delete_tracked(store, "d0.cpp", deleted_ids, deleted_defs)
    build_refs(store, changed_ids=set(), deleted_ids=deleted_ids, deleted_defs=deleted_defs)
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
    build_refs(store, changed_ids={"x8"}, deleted_ids=set(), deleted_defs={})
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
    deleted_defs: dict = {}
    _delete_tracked(store, "x8.src", deleted_ids, deleted_defs)

    build_refs(store, changed_ids=set(), deleted_ids=deleted_ids, deleted_defs=deleted_defs)
    got = _edges(store)
    _assert_indegree_consistent(store)
    remaining = [c for c in initial if c["id"] != "x8"]
    want = _full_rebuild_edges(remaining)
    assert got == want
    assert any(e == "xlang" for (_f, _t, e) in got), "dropping below the cap must restore xlang edges"


# ---------------------------------------------------------------------------
# xlang through names the batch did not touch
# ---------------------------------------------------------------------------

def test_incremental_keeps_an_xlang_edge_through_a_name_the_batch_left_alone():
    """Editing `open_url` recomputes the Python `shell_open`, which calls it;
    its pairing with the C++ `shell_open` goes through an untouched name."""
    store = _mk_store()
    cpp = _chunk("cpp", "os.cpp", name="shell_open", content="Error shell_open(const String &p_uri) {}")
    py = _chunk("py", "serve.py", name="shell_open", language="python",
                content="def shell_open(url):\n    open_url(url)")
    helper = _chunk("helper", "util.py", name="open_url", language="python", content="def open_url(u):\n    pass")
    _insert(store, [cpp, py, helper])
    build_refs(store)
    assert ("py", "cpp", "xlang") in _edges(store)

    deleted_ids: set[str] = set()
    deleted_defs: dict = {}
    _delete_tracked(store, "util.py", deleted_ids, deleted_defs)
    helper2 = _chunk("helper2", "util.py", name="open_url", language="python",
                     content="def open_url(u):\n    return u")
    _insert(store, [helper2])
    build_refs(store, changed_ids={"helper2"}, deleted_ids=deleted_ids, deleted_defs=deleted_defs)
    got = _edges(store)
    _assert_indegree_consistent(store)
    assert got == _full_rebuild_edges([cpp, py, helper2])
    assert ("py", "cpp", "xlang") in got


def test_incremental_preserves_an_xlang_edge_two_shared_names_away():
    """P shares Beta with Q, Q shares Gamma with R, and R pairs with T across
    languages through Delta. Editing P leaves R<->T as a full rebuild has it."""
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
    deleted_defs: dict = {}
    _delete_tracked(store, "P.cs", deleted_ids, deleted_defs)

    p2 = _chunk("p2", "P.cs", name="Alpha", language="c_sharp", content="/* edited */")
    _insert(store, [p2])
    store.insert_symbols([
        {"path": "P.cs", "name": "Beta", "kind": "method", "language": "c_sharp",
         "start_line": 1, "end_line": 1, "chunk_id": "p2"},
    ])

    build_refs(store, changed_ids={"p2"}, deleted_ids=deleted_ids, deleted_defs=deleted_defs)
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
    assert any(e == "xlang" and {f, tgt} == {"r", "t"} for (f, tgt, e) in got)


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
    build_refs(store, changed_ids={"m8"}, deleted_ids=set(), deleted_defs={}, cap_mentions=True)
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
    build_refs(store, changed_ids={"m8"}, deleted_ids=set(), deleted_defs={})
    got = _edges(store)
    _assert_indegree_consistent(store)
    want = _full_rebuild_edges(initial + new, cap_mentions=False)
    assert got == want
    assert sum(1 for (_f, _t, e) in got if e == "mentions") == 9, \
        "default (uncapped) behaviour must be unaffected"


# ---------------------------------------------------------------------------
# associated ranking
# ---------------------------------------------------------------------------

def test_incremental_reranks_associated_pairs_corpus_wide(caplog):
    """r3 mentioning alpha lowers the PMI of (r1, alpha), so (r2, beta) takes
    the one associated slot though neither r1 nor r2 is in the batch. A run
    under another associated_top_frac, or over a graph built under older
    rules, rebuilds fully."""
    store = _mk_store()
    initial = [
        _chunk("da", "a.cpp", name="alpha", content="void alpha() {}"),
        _chunk("db", "b.cpp", name="beta", content="void beta() {}"),
        _chunk("r1", "r1.cpp", name="run1", content="void run1() { alpha(); }"),
        _chunk("r2", "r2.cpp", name="run2", content="void run2() { beta(); }"),
    ]
    _insert(store, initial)
    build_refs(store, associated_top_frac=0.5)
    assert ("r1", "da", "associated") in _edges(store)

    r3 = _chunk("r3", "r3.cpp", name="run3", content="void run3() { alpha(); }")
    _insert(store, [r3])
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids={"r3"}, deleted_ids=set(), deleted_defs={}, associated_top_frac=0.5)
    _assert_incremental_ran(caplog)
    got = _edges(store)
    _assert_indegree_consistent(store)
    assert got == _full_rebuild_edges(initial + [r3], associated_top_frac=0.5)
    assert ("r2", "db", "associated") in got
    assert ("r1", "da", "mentions") in got

    r4 = _chunk("r4", "r4.cpp", name="run4", content="void run4() { beta(); }")
    _insert(store, [r4])
    build_refs(store, changed_ids={"r4"}, deleted_ids=set(), deleted_defs={}, associated_top_frac=0.0)
    assert _edges(store) == _full_rebuild_edges(initial + [r3, r4])
    _assert_indegree_consistent(store)

    store.set_meta(repomap._REFS_PARAMS_META_KEY, "[false, 0.0]")
    r5 = _chunk("r5", "r5.cpp", name="run5", content="void run5() { beta(); }")
    _insert(store, [r5])
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids={"r5"}, deleted_ids=set(), deleted_defs={})
    assert any("full rebuild" in r.message for r in caplog.records)
    assert not any("chunk_refs incremental update:" in r.message for r in caplog.records)


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
        build_refs(store, changed_ids=changed, deleted_ids=set(), deleted_defs={})
    finally:
        repomap._build_refs_incremental = original
    assert not called
    _assert_indegree_consistent(store)


def test_incremental_gate_counts_the_chunks_a_batch_reaches(monkeypatch):
    """Changing a hub fans out to its 19 referencers, past the 25% gate, and
    falls through to a full rebuild. Changing a sibling in the hub's file, or
    one link of a chain of chunks that share names pairwise, stays incremental."""
    monkeypatch.setattr(repomap, "_REFS_INCREMENTAL_MAX_REFERENCER_FRACTION", 0.25)
    store = _mk_store()
    hub = [_chunk("hub", "hub.cpp", name="hubfn", content="void hubfn() {}"),
           _chunk("leaf", "hub.cpp", name="leaf", content="void leaf() {}", start_line=3)]
    spokes = [_chunk(f"s{i}", f"s{i}.cpp", name=f"fn{i}", content=f"void fn{i}() {{ hubfn(); }}")
              for i in range(19)]
    links = [_chunk(f"l{i}", f"l{i}.cpp", name=f"link{i}", content=f"void link{i}() {{}}") for i in range(20)]
    link_symbols = [{"path": f"l{i}.cpp", "name": f"link{i + 1}", "kind": "function_definition", "language": "cpp",
                     "start_line": 1, "end_line": 1, "chunk_id": f"l{i}"} for i in range(19)]
    _insert(store, hub + spokes + links)
    store.insert_symbols(link_symbols)
    build_refs(store)

    results = []
    original = repomap._build_refs_incremental

    def _spy(*args, **kwargs):
        results.append(original(*args, **kwargs))
        return results[-1]

    repomap._build_refs_incremental = _spy
    try:
        build_refs(store, changed_ids={"hub"}, deleted_ids=set(), deleted_defs={})
        deleted_ids: set[str] = set()
        deleted_defs: dict = {}
        _delete_tracked(store, "hub.cpp", deleted_ids, deleted_defs)
        leaf2 = _chunk("leaf2", "hub.cpp", name="leaf", content="void leaf() { return; }", start_line=3)
        _insert(store, [hub[0], leaf2])
        build_refs(store, changed_ids={"hub", "leaf2"}, deleted_ids=deleted_ids, deleted_defs=deleted_defs)
        deleted_ids, deleted_defs = set(), {}
        _delete_tracked(store, "l0.cpp", deleted_ids, deleted_defs)
        l0b = _chunk("l0b", "l0.cpp", name="link0", content="void link0() { return; }")
        _insert(store, [l0b])
        store.insert_symbols([{**link_symbols[0], "chunk_id": "l0b"}])
        build_refs(store, changed_ids={"l0b"}, deleted_ids=deleted_ids, deleted_defs=deleted_defs)
    finally:
        repomap._build_refs_incremental = original

    assert results[0] is None
    assert None not in results[1:]
    ref = _mk_store()
    _insert(ref, [hub[0], leaf2] + spokes + [l0b] + links[1:])
    ref.insert_symbols([{**link_symbols[0], "chunk_id": "l0b"}] + link_symbols[1:])
    build_refs(ref)
    assert _edges(store) == _edges(ref)
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

    written = build_refs(store, changed_ids=set(), deleted_ids=set(), deleted_defs={})
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
    build_refs(st, changed_ids={"named_cls", "anon"}, deleted_ids=set(), deleted_defs={})
    got = sorted(e for e in _edges(st) if e[0] == "caller")

    assert got == want
    assert ("caller", "anon", "mentions") in want
    assert ("caller", "named_cls", "calls") in want


def test_a_namespace_chunk_is_never_an_edge_target(caplog):
    """A namespace body of declarations is named after its namespace, and a
    namespace-wrapped codebase writes `engine::X` everywhere: every such
    chunk mentioned every namespace chunk. Full and incremental alike."""
    store = _mk_store()
    initial = [
        _chunk("ns1", "a.h", name="engine", chunk_type="declaration_list",
               content="namespace engine { class Clock; const int kTicks = 60; }"),
        _chunk("ns2", "b.h", name="engine", chunk_type="declaration_list",
               content="namespace engine { struct Frame; }"),
        _chunk("cs", "Tools.cs", name="Engine.Tools", language="c_sharp",
               chunk_type="namespace_declaration",
               content="namespace Engine.Tools { public delegate void Tick(); }"),
        _chunk("adv", "advance.cpp", name="advance", content="void advance() {}"),
        _chunk("run", "run.cpp", name="run", content="void run() { engine::advance(); Tools(); }"),
    ]
    _insert(store, initial)
    build_refs(store)
    scopes = {"ns1", "ns2", "cs"}
    assert ("run", "adv", "mentions") in _edges(store)
    assert not {e for e in _edges(store) if e[1] in scopes}

    new = [_chunk("run2", "run2.cpp", name="run2", content="void run2() { engine::advance(); }")]
    _insert(store, new)
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids={"run2"}, deleted_ids=set(), deleted_defs={})
    _assert_incremental_ran(caplog)
    got = _edges(store)
    _assert_indegree_consistent(store)
    assert got == _full_rebuild_edges(initial + new)
    assert not {e for e in got if e[1] in scopes}


def test_a_call_to_a_name_with_many_definers_stays_typed_incrementally(caplog):
    """Past the fan-out cap's threshold, the cap is the only thing that drops
    a typed edge: with it off, an incremental update keeps `calls`."""
    store = _mk_store()
    definers = [_chunk(f"d{i}", f"d{i}.cpp", name="tick", content=f"void tick() {{ step{i}(); }}")
                for i in range(9)]
    caller = _chunk("c1", "c1.cpp", name="run", content="void run() { tick(); }")
    caller["metadata"] = {"calls": ["tick"], "imports": [], "inherits": []}
    _insert(store, definers + [caller])
    build_refs(store)

    new = _chunk("c2", "c2.cpp", name="run2", content="void run2() { tick(); }")
    new["metadata"] = {"calls": ["tick"], "imports": [], "inherits": []}
    _insert(store, [new])
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids={"c2"}, deleted_ids=set(), deleted_defs={})
    _assert_incremental_ran(caplog)
    got = _edges(store)
    assert got == _full_rebuild_edges(definers + [caller, new])
    assert ("c2", "d0", "calls") in got


def test_an_incremental_call_takes_defaults_from_a_class_split_into_pieces(caplog):
    """The declaration sits in a later piece of the class, not the one
    named after it; the class's symbol span finds it."""
    store = _mk_store()
    corpus = [
        _chunk("thread_head", "thread.h", name="Thread", chunk_type="class_specifier",
               start_line=1, end_line=2, content="class Thread {\n  int id = 0;"),
        _chunk("thread_rest", "thread.h", name="<block>", chunk_type="module", start_line=3, end_line=4,
               content="  Error start(Callback p_cb, void *p_ud = nullptr);\n};"),
        _chunk("thread_start", "thread.cpp", name="Thread::start",
               content="Error Thread::start(Callback p_cb, void *p_ud) { return OK; }"),
        _chunk("timer_start", "timer.cpp", name="Timer::start", content="void Timer::start(int a, int b) {}"),
    ]
    symbols = [{"path": "thread.h", "name": "Thread", "kind": "class_specifier", "language": "cpp",
                "start_line": 1, "end_line": 4, "chunk_id": "thread_head"}]
    _insert(store, corpus)
    store.insert_symbols(symbols)
    build_refs(store)

    new = _chunk("c1", "c1.cpp", name="run", content="void run() { start(cb); }",
                 metadata={"calls": [{"name": "start", "receiver": None, "arity": 1}]})
    _insert(store, [new])
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids={"c1"}, deleted_ids=set(), deleted_defs={})
    _assert_incremental_ran(caplog)
    got = _edges(store)
    ref = _mk_store()
    _insert(ref, corpus + [new])
    ref.insert_symbols(symbols)
    build_refs(ref)
    assert got == _edges(ref)
    assert ("c1", "thread_start", "calls") in got
    assert ("c1", "timer_start", "calls") not in got


def _member(owner: str, name: str, kind: str, line: int, chunk_id: str, path: str,
            type_text: str | None = None) -> dict:
    return {"path": path, "language": "cpp", "owner": owner, "name": name, "kind": kind,
            "type_text": type_text, "arity_min": 0, "arity_max": 0, "variadic": 0, "flags": None,
            "line": line, "chunk_id": chunk_id}


def _class_span(name: str, path: str, start: int, end: int, chunk_id: str) -> dict:
    return {"path": path, "name": name, "kind": "class_specifier", "language": "cpp",
            "start_line": start, "end_line": end, "chunk_id": chunk_id}


def _with_class_model(store: Store, chunks: list[dict], model: dict[str, list[dict]],
                      spans: list[dict]) -> None:
    _insert(store, chunks)
    for path, rows in model.items():
        store.replace_class_model(path, rows, [], [])
    store.insert_symbols(spans)
    store.set_meta("class_model_version", "1")


_DRAW_CALL = {"name": "draw", "receiver": "s", "arity": 0, "racc": "->",
              "rhead": {"name": "s", "via": "param", "type": "Shape *"}}


def test_an_incremental_call_resolves_through_the_class_model(caplog):
    corpus = [
        _chunk("shape_h", "shape.h", name="Shape", chunk_type="class_specifier", start_line=1, end_line=4,
               content="class Shape {\npublic:\n  virtual void draw() = 0;\n};"),
        _chunk("canvas_draw", "canvas.cpp", name="Canvas::draw", content="void Canvas::draw() {}"),
    ]
    model = {"shape.h": [_member("Shape", "draw", "method_decl", 3, "shape_h", "shape.h")]}
    spans = [_class_span("Shape", "shape.h", 1, 4, "shape_h")]
    store = _mk_store()
    _with_class_model(store, corpus, model, spans)
    build_refs(store)

    new = _chunk("c1", "use.cpp", name="render", content="void render(Shape *s) { s->draw(); }",
                 metadata={"calls": [_DRAW_CALL]})
    _insert(store, [new])
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids={"c1"}, deleted_ids=set(), deleted_defs={})
    _assert_incremental_ran(caplog)
    got = _edges(store)
    ref = _mk_store()
    _with_class_model(ref, corpus + [new], model, spans)
    build_refs(ref)
    assert got == _edges(ref)
    assert ("c1", "shape_h", "calls") in got
    assert ("c1", "canvas_draw", "calls") not in got


def test_an_incremental_class_change_retargets_a_call_that_names_only_the_member(caplog):
    """The caller reaches Shape through a field and never names it; the
    declaration it now resolves to is a member of the changed class."""
    canvas = _chunk("canvas_h", "canvas.h", name="Canvas", chunk_type="class_specifier", start_line=1, end_line=3,
                    content="class Canvas {\n  Shape *surface;\n};")
    caller = _chunk("c1", "use.cpp", name="render", content="void render(Canvas *c) { c->surface->outline(); }",
                    metadata={"calls": [{"name": "outline", "receiver": "surface", "arity": 0, "racc": "->",
                                         "rhead": {"name": "c", "via": "param", "type": "Canvas *"},
                                         "rpath": ["surface"]}]})
    old_shape = _chunk("shape_h", "shape.h", name="Shape", chunk_type="class_specifier", start_line=1, end_line=3,
                       content="class Shape {\n  void draw();\n};")
    new_shape = _chunk("shape_h2", "shape.h", name="Shape", chunk_type="class_specifier", start_line=1, end_line=4,
                       content="class Shape {\n  void draw();\n  void outline();\n};")
    filler = [_chunk(f"filler{i}", f"filler{i}.cpp", name=f"filler{i}", content=f"void filler{i}() {{}}")
              for i in range(12)]
    canvas_rows = [_member("Canvas", "surface", "field", 2, "canvas_h", "canvas.h", "Shape *")]
    spans = [_class_span("Canvas", "canvas.h", 1, 3, "canvas_h")]
    store = _mk_store()
    _with_class_model(store, [canvas, caller, old_shape] + filler, {
        "canvas.h": canvas_rows, "shape.h": [_member("Shape", "draw", "method_decl", 2, "shape_h", "shape.h")],
    }, spans + [_class_span("Shape", "shape.h", 1, 3, "shape_h")])
    build_refs(store)

    deleted_ids: set[str] = set()
    deleted_defs: dict = {}
    _delete_tracked(store, "shape.h", deleted_ids, deleted_defs)
    new_rows = [_member("Shape", "draw", "method_decl", 2, "shape_h2", "shape.h"),
                _member("Shape", "outline", "method_decl", 3, "shape_h2", "shape.h")]
    new_span = _class_span("Shape", "shape.h", 1, 4, "shape_h2")
    _insert(store, [new_shape])
    store.replace_class_model("shape.h", new_rows, [], [])
    store.insert_symbols([new_span])
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids={"shape_h2"}, deleted_ids=deleted_ids, deleted_defs=deleted_defs)
    _assert_incremental_ran(caplog)
    got = _edges(store)
    ref = _mk_store()
    _with_class_model(ref, [canvas, caller, new_shape] + filler,
                      {"canvas.h": canvas_rows, "shape.h": new_rows}, spans + [new_span])
    build_refs(ref)
    assert got == _edges(ref)
    assert ("c1", "shape_h2", "calls") in got


def test_a_new_base_retargets_a_call_in_a_subclass_that_spells_neither_class():
    """L::run calls step(); L derives from B, and B now derives from X, which declares step."""
    x_h = _chunk("x_h", "x.h", name="X", chunk_type="class_specifier", start_line=1, end_line=3,
                 content="class X {\n  void step();\n};")
    l_h = _chunk("l_h", "l.h", name="L", chunk_type="class_specifier", start_line=1, end_line=3,
                 content="class L : public B {\n  void run();\n};")
    l_run = _chunk("l_run", "l.cpp", name="L::run", content="void L::run() { step(); }",
                   metadata={"calls": [{"name": "step", "receiver": None, "arity": 0, "cls": "L"}]})
    old_b = _chunk("b_h", "b.h", name="B", chunk_type="class_specifier", start_line=1, end_line=3,
                   content="class B {\n  int hp;\n};")
    new_b = _chunk("b_h2", "b.h", name="B", chunk_type="class_specifier", start_line=1, end_line=3,
                   content="class B : public X {\n  int hp;\n};")
    model = {"x.h": ([_member("X", "step", "method_decl", 2, "x_h", "x.h")], []),
             "l.h": ([_member("L", "run", "method_decl", 2, "l_h", "l.h")],
                     [{"path": "l.h", "owner": "L", "base": "B", "line": 1}])}
    spans = [_class_span("X", "x.h", 1, 3, "x_h"), _class_span("L", "l.h", 1, 3, "l_h")]

    def seed(store, b, b_bases):
        _insert(store, [x_h, l_h, l_run, b])
        for path, (rows, bases) in model.items():
            store.replace_class_model(path, rows, bases, [])
        store.replace_class_model("b.h", [_member("B", "hp", "field", 2, b["id"], "b.h", "int")], b_bases, [])
        store.insert_symbols(spans + [_class_span("B", "b.h", 1, 3, b["id"])])
        store.set_meta("class_model_version", "1")

    store = _mk_store()
    seed(store, old_b, [])
    build_refs(store)
    deleted_ids: set[str] = set()
    deleted_defs: dict = {}
    _delete_tracked(store, "b.h", deleted_ids, deleted_defs)
    new_bases = [{"path": "b.h", "owner": "B", "base": "X", "line": 1}]
    _insert(store, [new_b])
    store.replace_class_model("b.h", [_member("B", "hp", "field", 2, "b_h2", "b.h", "int")], new_bases, [])
    store.insert_symbols([_class_span("B", "b.h", 1, 3, "b_h2")])
    build_refs(store, changed_ids={"b_h2"}, deleted_ids=deleted_ids, deleted_defs=deleted_defs)
    ref = _mk_store()
    seed(ref, new_b, new_bases)
    build_refs(ref)
    assert _edges(store) == _edges(ref)
    assert ("l_run", "x_h", "calls") in _edges(ref)


_OUTLINE_CALLER = _chunk("c1", "use.cpp", name="render", content="void render(Server *s) { s->outline(); }",
                         metadata={"calls": [{"name": "outline", "receiver": "s", "arity": 0, "racc": "->",
                                              "rhead": {"name": "s", "via": "param", "type": "Server *"}}]})
_CANVAS_OUTLINE = _chunk("canvas_outline", "canvas.cpp", name="Canvas::outline", content="void Canvas::outline() {}")
_SERVER = _chunk("server_h", "server.h", name="Server", chunk_type="class_specifier", start_line=1, end_line=3,
                 content="class Server {\n  void outline();\n};")
_SERVER_ROWS = [_member("Server", "outline", "method_decl", 2, "server_h", "server.h")]
_SERVER_SPAN = _class_span("Server", "server.h", 1, 3, "server_h")
_OUTLINE_FILLER = [_chunk(f"filler{i}", f"filler{i}.cpp", name=f"filler{i}", content=f"void filler{i}() {{}}")
                   for i in range(12)]


def _with_server(store: Store, chunks: list[dict]) -> None:
    _with_class_model(store, chunks, {"server.h": _SERVER_ROWS}, [_SERVER_SPAN])


def test_an_incremental_class_that_settles_an_unchanged_call_drops_its_mentions(caplog):
    store = _mk_store()
    _with_class_model(store, [_OUTLINE_CALLER, _CANVAS_OUTLINE] + _OUTLINE_FILLER, {}, [])
    build_refs(store)
    assert ("c1", "canvas_outline", "mentions") in _edges(store)

    _with_server(store, [_SERVER])
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids={"server_h"}, deleted_ids=set(), deleted_defs={})
    _assert_incremental_ran(caplog)
    got = _edges(store)
    ref = _mk_store()
    _with_server(ref, [_OUTLINE_CALLER, _CANVAS_OUTLINE, _SERVER] + _OUTLINE_FILLER)
    build_refs(ref)
    assert got == _edges(ref)
    assert ("c1", "server_h", "calls") in got
    assert ("c1", "canvas_outline", "mentions") not in got


def test_an_incremental_class_removal_restores_the_mentions_of_an_unchanged_call(caplog):
    store = _mk_store()
    _with_server(store, [_OUTLINE_CALLER, _CANVAS_OUTLINE, _SERVER] + _OUTLINE_FILLER)
    build_refs(store)
    assert ("c1", "canvas_outline", "mentions") not in _edges(store)

    deleted_ids: set[str] = set()
    deleted_defs: dict = {}
    _delete_tracked(store, "server.h", deleted_ids, deleted_defs)
    with caplog.at_level(logging.INFO, logger="repomap"):
        build_refs(store, changed_ids=set(), deleted_ids=deleted_ids, deleted_defs=deleted_defs)
    _assert_incremental_ran(caplog)
    got = _edges(store)
    ref = _mk_store()
    _with_class_model(ref, [_OUTLINE_CALLER, _CANVAS_OUTLINE] + _OUTLINE_FILLER, {}, [])
    build_refs(ref)
    assert got == _edges(ref)
    assert ("c1", "canvas_outline", "mentions") in got
