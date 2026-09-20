"""Tests for repomap symbol-type prefix rendering."""
import pytest
from chonks.index.graph.hierarchy import rebuild_hierarchy
from chonks.retrieval.repomap import _format_map


def _chunk(chunk_type: str, name: str, path: str = "src/foo.py", start_line: int = 1) -> dict:
    return {
        "id": f"{path}:{name}",
        "path": path,
        "name": name,
        "chunk_type": chunk_type,
        "start_line": start_line,
        "content": None,
    }


def _scores(chunks: list[dict]) -> dict[str, float]:
    return {c["id"]: 1.0 for c in chunks}


def test_build_graph_expands_refs_from_symbol_index():
    """applyStep is folded into Sim's chunk and isn't a chunk name on its
    own, so it only resolves to an edge via the symbol index."""
    from chonks.index.graph.refs import _build_graph
    chunks = [
        {"id": "a", "name": "driver", "language": "cpp",
         "content": "void driver() { applyStep(x); }"},
        {"id": "b", "name": "Sim", "language": "cpp",
         "content": "class Sim { void applyStep(int){} };"},
    ]
    assert ("a", "b") not in _build_graph(chunks)
    assert ("a", "b") in _build_graph(chunks, {"applyStep": ["b"]}), \
        "symbol-index reference did not create a graph edge"


# ---------------------------------------------------------------------------
# Bare-alias resolution for qualified symbol names
# ---------------------------------------------------------------------------

def test_build_graph_qualified_definer_reachable_via_bare_alias():
    """A chunk whose only symbol is qualified ("Widget::applyStep") is still
    reachable from a bare content token ("applyStep()")."""
    from chonks.index.graph.refs import _build_graph
    chunks = [
        {"id": "a", "name": "driver", "language": "cpp",
         "content": "void driver() { applyStep(); }"},
        {"id": "b", "name": None, "language": "cpp", "content": "class Widget {};"},
    ]
    sym_map = {"Widget::applyStep": ["b"]}
    edges = _build_graph(chunks, sym_map)
    assert edges.get(("a", "b")) == "mentions", \
        "bare content token did not resolve to the qualified-name definer"


def test_build_graph_bare_alias_respects_min_name_len():
    """A qualified name whose bare last component is under _MIN_NAME_LEN
    gets no alias at all."""
    from chonks.core.edges import _MIN_NAME_LEN
    from chonks.index.graph.refs import _build_graph
    assert _MIN_NAME_LEN == 3
    chunks = [
        {"id": "a", "name": "driver", "language": "cpp",
         "content": "void driver() { x = ok; }"},
        {"id": "b", "name": None, "language": "cpp", "content": "class Foo {};"},
    ]
    sym_map = {"Foo::ok": ["b"]}  # bare alias "ok" is 2 chars, under _MIN_NAME_LEN
    edges = _build_graph(chunks, sym_map)
    assert ("a", "b") not in edges, \
        "an alias shorter than _MIN_NAME_LEN must not be registered"


def test_build_graph_bare_alias_ubiquity_cap_applies_post_aliasing():
    """Many distinct qualified definers that alias to the same bare name
    ("init") are aggregated before the ubiquity cap is checked, not
    evaluated per qualified name."""
    from chonks.core.edges import _MAX_CROSS_LANG_OCCURRENCES
    from chonks.index.graph.refs import _build_graph

    n_definers = _MAX_CROSS_LANG_OCCURRENCES + 1
    chunks = [
        {"id": "caller", "name": "driver", "language": "cpp",
         "content": "void driver() { init(); }"},
    ]
    chunks += [
        {"id": f"def{i}", "name": None, "language": "cpp", "content": f"class C{i} {{}};"}
        for i in range(n_definers)
    ]
    sym_map = {f"C{i}::init": [f"def{i}"] for i in range(n_definers)}
    edges = _build_graph(chunks, sym_map, cap_mentions=True)
    assert not any(k[0] == "caller" for k in edges), \
        "post-aliasing definer count over the cap must suppress all mentions edges"


def test_build_graph_bare_alias_no_duplicate_edge_when_both_forms_resolve():
    """A chunk whose own name and whose qualified symbol both alias to the
    same bare key must not be registered twice under that key."""
    from chonks.index.graph.refs import _build_graph
    chunks = [
        {"id": "a", "name": "driver", "language": "cpp",
         "content": "void driver() { bsearch(); }"},
        {"id": "b", "name": "bsearch", "language": "cpp", "content": "void bsearch(){}"},
    ]
    sym_map = {"Vector::bsearch": ["b"]}
    edges = _build_graph(chunks, sym_map)
    assert edges.get(("a", "b")) == "mentions"
    assert len(edges) == 1, "the same (from, to) pair must not appear more than once"


# ---------------------------------------------------------------------------
# Tree-sitter node types that must produce friendly prefixes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("node_type,expected_prefix", [
    ("function_definition", "function"),   # Python / C / C++
    ("class_definition", "class"),         # Python
    ("class_specifier", "class"),          # C++
    ("class_declaration", "class"),        # C#
    ("method_declaration", "method"),      # C# / Java
    ("struct_specifier", "struct"),        # C / C++
    ("enum_specifier", "enum"),            # C / C++
    ("union_specifier", "union"),          # C
    ("type_definition", "typedef"),        # C
    ("namespace_definition", "namespace"), # C++
    ("interface_declaration", "interface"),# C# / Java
    ("decorated_definition", "function"),  # Python decorated function
])
def test_node_type_renders_prefix(node_type, expected_prefix):
    chunks = [_chunk(node_type, "MySymbol")]
    scores = _scores(chunks)
    output = _format_map(chunks, scores, token_budget=None)
    assert f"  {expected_prefix} MySymbol" in output, (
        f"Expected prefix '{expected_prefix}' for node_type '{node_type}', got:\n{output}"
    )


def test_unknown_node_type_has_no_prefix():
    chunks = [_chunk("block", "some_block")]
    scores = _scores(chunks)
    output = _format_map(chunks, scores, token_budget=None)
    assert "  some_block  (line" in output
    assert "  block some_block" not in output


def test_friendly_name_legacy_names_have_no_prefix():
    """'function' is a friendly display name, not a tree-sitter node type,
    so it must not produce a double-prefix."""
    chunks = [_chunk("function", "legacy_fn")]
    scores = _scores(chunks)
    output = _format_map(chunks, scores, token_budget=None)
    assert "  function function" not in output


# ---------------------------------------------------------------------------
# build_repomap: full-symbol listing
# ---------------------------------------------------------------------------

def _sym(path, name, kind, *, start_line=1, end_line=2, chunk_id=None, lang="cpp"):
    return {"path": path, "name": name, "kind": kind, "language": lang,
            "start_line": start_line, "end_line": end_line, "chunk_id": chunk_id}


def test_build_repomap_lists_folded_class_methods(tmp_path):
    """Methods folded into a small whole-class chunk share that chunk's id
    and aren't chunk names of their own, but they're in the symbol index and
    must still show up in the map."""
    from chonks.storage.store import Store
    from chonks.retrieval.repomap import build_repomap

    store = Store(tmp_path / "t.db")
    store.insert_symbols([
        _sym("src/sim.cpp", "Sim", "class_specifier", start_line=1, end_line=30, chunk_id="c1"),
        _sym("src/sim.cpp", "applyStep", "function_definition", start_line=5, end_line=9, chunk_id="c1"),
        _sym("src/sim.cpp", "reset", "function_definition", start_line=11, end_line=14, chunk_id="c1"),
    ])

    out = build_repomap(store)
    assert "class Sim" in out
    assert "function applyStep" in out, f"folded method missing from map:\n{out}"
    assert "function reset" in out, f"folded method missing from map:\n{out}"
    store.close()


def test_build_repomap_empty_symbols_falls_back_to_chunks(tmp_path):
    """An old DB with no symbol table rows must still produce a map from
    chunk names rather than returning empty."""
    from chonks.storage.store import Store
    from chonks.retrieval.repomap import build_repomap

    store = Store(tmp_path / "t.db")
    with store._lock:
        store._conn.execute(
            "INSERT INTO chunks(id, path, language, chunk_type, name, start_line, "
            "end_line, content, indexed_at) VALUES(?,?,?,?,?,?,?,?,0.0)",
            ("c1", "src/foo.cpp", "cpp", "function_definition", "doThing", 1, 4, "void doThing(){}"),
        )
        store._conn.commit()
    assert store.symbols_count() == 0

    out = build_repomap(store)
    assert "src/foo.cpp" in out
    assert "function doThing" in out, f"fallback produced no symbols:\n{out}"
    store.close()


def test_node_type_prefix_covers_all_boundary_nodes():
    """Guards against _NODE_TYPE_PREFIX drifting from the chunker's
    _BOUNDARY_NODES, which was the root cause of the original bare renders."""
    from chonks.index.segment import _BOUNDARY_NODES
    from chonks.retrieval.repomap import _NODE_TYPE_PREFIX

    emitted = {nt for types in _BOUNDARY_NODES.values() for nt in types}
    emitted.add("cbuffer")  # HLSL cbuffer/tbuffer -> synthetic chunk_type (_is_cbuffer)
    missing = emitted - _NODE_TYPE_PREFIX.keys()
    assert not missing, (
        f"chunker emits these chunk_types but repomap has no kind label: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Typed edges (calls / imports / inherits)
# ---------------------------------------------------------------------------

def test_build_graph_emits_typed_edge_from_metadata():
    from chonks.index.graph.refs import _build_graph
    chunks = [
        {"id": "a", "name": "driver", "language": "cpp",
         "content": "void driver() { helper(); }",
         "metadata": {"calls": ["helper"], "imports": [], "inherits": []}},
        {"id": "b", "name": "helper", "language": "cpp",
         "content": "void helper() {}",
         "metadata": {"calls": [], "imports": [], "inherits": []}},
    ]
    edges = _build_graph(chunks)
    assert edges.get(("a", "b")) == "calls"


def test_build_graph_emits_typed_edge_from_metadata_c_across_files():
    """Resolution is purely by name, not by shared file, so a caller and
    callee in different .c files still resolve to a typed edge."""
    from chonks.index.graph.refs import _build_graph
    chunks = [
        {"id": "a", "name": "driver", "language": "c", "path": "src/driver.c",
         "content": "int driver(void) { return helper(1); }",
         "metadata": {"calls": ["helper"], "imports": [], "inherits": []}},
        {"id": "b", "name": "helper", "language": "c", "path": "src/helper.c",
         "content": "int helper(int x) { return x + 1; }",
         "metadata": {"calls": [], "imports": [], "inherits": []}},
    ]
    edges = _build_graph(chunks)
    assert edges.get(("a", "b")) == "calls"


def test_typed_edge_supersedes_mentions():
    """The typed pass runs after the content-scan pass, so it wins when both
    produce an edge for the same (from_id, to_id) pair."""
    from chonks.index.graph.refs import _build_graph
    chunks = [
        {"id": "a", "name": "driver", "language": "cpp",
         "content": "void driver() { helper(); }",  # content scan also sees "helper"
         "metadata": {"calls": ["helper"], "imports": [], "inherits": []}},
        {"id": "b", "name": "helper", "language": "cpp",
         "content": "void helper() {}", "metadata": None},
    ]
    edges = _build_graph(chunks)
    assert edges.get(("a", "b")) == "calls", \
        "typed edge must supersede the untyped mentions edge for the same pair"


def test_build_graph_no_metadata_falls_back_to_mentions():
    from chonks.index.graph.refs import _build_graph
    chunks = [
        {"id": "a", "name": "driver", "language": "hlsl",
         "content": "void driver() { helper(); }"},
        {"id": "b", "name": "helper", "language": "hlsl",
         "content": "void helper() {}"},
    ]
    edges = _build_graph(chunks)
    assert edges.get(("a", "b")) == "mentions"


def test_cap_mentions_suppresses_edges_for_ubiquitous_name():
    """cap_mentions=True mirrors the xlang/typed passes' existing cap: a
    mentioned name over _MAX_CROSS_LANG_OCCURRENCES definers gets no
    mentions edges at all."""
    from chonks.core.edges import _MAX_CROSS_LANG_OCCURRENCES
    from chonks.index.graph.refs import _build_graph

    n_definers = _MAX_CROSS_LANG_OCCURRENCES + 1
    chunks = [
        {"id": f"def{i}", "name": "Update", "language": "cpp", "content": "void Update(){}"}
        for i in range(n_definers)
    ]
    chunks.append(
        {"id": "caller", "name": "driver", "language": "cpp",
         "content": "void driver() { Update(); }"},
    )

    default_edges = _build_graph(chunks)
    assert any(k[0] == "caller" for k in default_edges), \
        "sanity: default behaviour still produces mentions edges for the over-cap name"

    capped_edges = _build_graph(chunks, cap_mentions=True)
    assert not any(k[0] == "caller" for k in capped_edges), \
        "cap_mentions=True must suppress ALL mentions edges for an over-cap name"


def test_cap_mentions_still_edges_within_cap_name():
    """The cap is a ceiling, not a blanket suppression."""
    from chonks.core.edges import _MAX_CROSS_LANG_OCCURRENCES
    from chonks.index.graph.refs import _build_graph

    n_definers = _MAX_CROSS_LANG_OCCURRENCES
    chunks = [
        {"id": f"def{i}", "name": "Helper", "language": "cpp", "content": "void Helper(){}"}
        for i in range(n_definers)
    ]
    chunks.append(
        {"id": "caller", "name": "driver", "language": "cpp",
         "content": "void driver() { Helper(); }"},
    )

    edges = _build_graph(chunks, cap_mentions=True)
    assert any(k[0] == "caller" for k in edges), \
        "within-cap name must still produce mentions edges when the flag is on"


def test_cap_mentions_does_not_affect_typed_edges():
    """Typed edges already have their own cap, independent of cap_mentions."""
    from chonks.index.graph.refs import _build_graph

    chunks = [
        {"id": "a", "name": "driver", "language": "cpp",
         "content": "void driver() { helper(); }",
         "metadata": {"calls": ["helper"], "imports": [], "inherits": []}},
        {"id": "b", "name": "helper", "language": "cpp",
         "content": "void helper() {}",
         "metadata": {"calls": [], "imports": [], "inherits": []}},
    ]
    edges_off = _build_graph(chunks)
    edges_on = _build_graph(chunks, cap_mentions=True)
    assert edges_off.get(("a", "b")) == "calls"
    assert edges_on.get(("a", "b")) == "calls"


# ---------------------------------------------------------------------------
# PMI-scored 'associated' promotion ("PMI-pruned mentions")
# ---------------------------------------------------------------------------

def test_classify_mentions_promotes_top_pmi_pairs_deterministically():
    """All four (chunk, name) pairs have exactly equal PMI, so the promoted
    set is decided by the tie-break (-PMI, name, chunk id), not by dict/set
    iteration order: "www" and "xxx" sort first and win."""
    from chonks.index.graph.refs import _classify_mentions

    referenced_by_chunk = {
        "refZ": {"zzz"},
        "refY": {"yyy"},
        "refX": {"xxx"},
        "refW": {"www"},
    }
    name_to_ids = {"zzz": ["defZ"], "yyy": ["defY"], "xxx": ["defX"], "www": ["defW"]}
    promoted = _classify_mentions(referenced_by_chunk, name_to_ids, associated_top_frac=0.5)
    assert promoted == {("refW", "www"), ("refX", "xxx")}


def test_classify_mentions_zero_frac_promotes_nothing():
    from chonks.index.graph.refs import _classify_mentions

    referenced_by_chunk = {"a": {"only_name"}}
    name_to_ids = {"only_name": ["def"]}
    assert _classify_mentions(referenced_by_chunk, name_to_ids, associated_top_frac=0.0) == set()


def test_classify_mentions_excludes_self_only_pairs_from_population():
    """A (chunk, name) pair whose only definer is the referencing chunk
    itself can never emit an edge, so it must not enter the PMI population
    at all (checked here at frac=1.0, which promotes everything live)."""
    from chonks.index.graph.refs import _classify_mentions

    referenced_by_chunk = {
        "solo": {"selfName"},
        "caller": {"liveName"},
    }
    name_to_ids = {"selfName": ["solo"], "liveName": ["def"]}
    promoted = _classify_mentions(referenced_by_chunk, name_to_ids, associated_top_frac=1.0)
    assert promoted == {("caller", "liveName")}


def test_classify_mentions_is_deterministic_across_repeated_calls():
    """Same tie-heavy population as the promotion test above, called twice
    with fresh dicts: must be byte-identical."""
    from chonks.index.graph.refs import _classify_mentions

    def _pairs():
        return {"refW": {"www"}, "refX": {"xxx"}, "refY": {"yyy"}, "refZ": {"zzz"}}

    name_to_ids = {"zzz": ["defZ"], "yyy": ["defY"], "xxx": ["defX"], "www": ["defW"]}
    first = _classify_mentions(_pairs(), name_to_ids, associated_top_frac=0.5)
    second = _classify_mentions(_pairs(), name_to_ids, associated_top_frac=0.5)
    assert first == second == {("refW", "www"), ("refX", "xxx")}


def test_build_graph_associated_top_frac_zero_is_bit_identical_to_mentions():
    """frac=0 (the default) must reproduce pre-PMI chunk_refs exactly, even
    on a corpus that would promote a pair at any positive frac."""
    from chonks.index.graph.refs import _build_graph

    chunks = [
        {"id": "def_rare", "name": "rareName", "language": "cpp",
         "content": "void rareName(){}"},
        {"id": "focus", "name": "focus", "language": "cpp",
         "content": "void focus(){ rareName(); }"},
    ]
    default_edges = _build_graph(chunks)
    explicit_zero_edges = _build_graph(chunks, associated_top_frac=0.0)
    assert default_edges == explicit_zero_edges == {("focus", "def_rare"): "mentions"}

    promoting_edges = _build_graph(chunks, associated_top_frac=1.0)
    assert promoting_edges == {("focus", "def_rare"): "associated"}, \
        "sanity: this corpus DOES promote at a positive frac, so the frac=0 " \
        "case above is a real assertion, not a vacuous one"


def test_build_graph_promotes_rare_pair_leaves_common_pairs_as_mentions():
    """A pair with high PMI (df=1, referenced only by focus) is promoted;
    four pairs with low PMI (df=4) are not, at a frac admitting exactly one
    of the five total pairs."""
    from chonks.index.graph.refs import _build_graph

    chunks = [
        {"id": "def_rare", "name": "rareName", "language": "cpp",
         "content": "void rareName(){}"},
        {"id": "def_common", "name": "commonName", "language": "cpp",
         "content": "void commonName(){}"},
        {"id": "focus", "name": "focus", "language": "cpp",
         "content": "void focus(){ rareName(); }"},
    ] + [
        {"id": f"spam{i}", "name": f"spam{i}", "language": "cpp",
         "content": "void spam(){ commonName(); }"}
        for i in range(4)
    ]
    edges = _build_graph(chunks, associated_top_frac=0.2)
    assert edges[("focus", "def_rare")] == "associated"
    for i in range(4):
        assert edges[(f"spam{i}", "def_common")] == "mentions"


def test_build_graph_cap_mentions_excludes_capped_name_from_pmi_population():
    """cap_mentions applies before PMI classification: a name skipped by
    the fan-out cap can never be promoted, even at associated_top_frac=1.0."""
    from chonks.core.edges import _MAX_CROSS_LANG_OCCURRENCES
    from chonks.index.graph.refs import _build_graph

    n_definers = _MAX_CROSS_LANG_OCCURRENCES + 1
    chunks = [
        {"id": f"def{i}", "name": "Update", "language": "cpp", "content": "void Update(){}"}
        for i in range(n_definers)
    ]
    chunks.append(
        {"id": "caller", "name": "driver", "language": "cpp",
         "content": "void driver() { Update(); }"},
    )
    edges = _build_graph(chunks, cap_mentions=True, associated_top_frac=1.0)
    assert not any(k[0] == "caller" for k in edges), \
        "capped-out name must be excluded from the PMI population, not just uncounted"


def test_typed_edge_supersedes_associated():
    """The typed pass runs last, so it must overwrite 'associated' too, not
    just plain 'mentions'."""
    from chonks.index.graph.refs import _build_graph

    chunks = [
        {"id": "a", "name": "driver", "language": "cpp",
         "content": "void driver() { helper(); }",
         "metadata": {"calls": ["helper"], "imports": [], "inherits": []}},
        {"id": "b", "name": "helper", "language": "cpp",
         "content": "void helper() {}", "metadata": None},
    ]
    # frac=1.0 promotes this pair's sole pair to 'associated' before the typed pass runs.
    edges = _build_graph(chunks, associated_top_frac=1.0)
    assert edges.get(("a", "b")) == "calls", \
        "typed edge must supersede 'associated' the same way it supersedes 'mentions'"


def test_xlang_edge_supersedes_associated():
    """The xlang pass runs after the mentions/associated pass, so a pair
    that's both a cross-language same-name pairing and a content-scan hit
    must end up 'xlang', not 'associated'."""
    from chonks.index.graph.refs import _build_graph

    chunks = [
        {"id": "a", "name": "Shared", "language": "cpp",
         "content": "void Shared(){ Alias(); }"},
        {"id": "b", "name": "Shared", "language": "c#",
         "content": "void Shared(){}"},
    ]
    sym_map = {"Alias": ["b"]}
    edges = _build_graph(chunks, sym_map, associated_top_frac=1.0)
    assert edges.get(("a", "b")) == "xlang", \
        "xlang edge must supersede 'associated' for the same (from, to) pair"


def test_build_graph_two_names_same_target_associated_wins_over_mentions():
    """Regression: two names both fanning out to the same target chunk used
    to have their (from, to) result decided by nondeterministic string-hash
    iteration order before the collision guard was added; the promoted
    classification must win regardless of emit order."""
    from chonks.index.graph.refs import _build_graph

    chunks = [
        {"id": "B", "name": "aaaName", "language": "cpp", "content": "void aaaName(){}"},
        {"id": "A", "name": "focus", "language": "cpp",
         "content": "void focus(){ aaaName(); zzzName(); }"},
    ] + [
        {"id": f"decoy{i}", "name": f"decoy{i}", "language": "cpp",
         "content": "void decoy(){ aaaName(); }"}
        for i in range(3)
    ]
    sym_map = {"zzzName": ["B"]}
    edges = _build_graph(chunks, sym_map, associated_top_frac=0.3)
    assert edges[("A", "B")] == "associated"


def test_build_graph_promoted_name_sorting_first_survives_later_mentions_write():
    """Mirror of the collision case above with sort order flipped: the
    promoted name is written first, and the later 'mentions' write for the
    same key must be refused by the collision guard, not silently downgrade it."""
    from chonks.index.graph.refs import _build_graph

    chunks = [
        {"id": "B", "name": "zzzName", "language": "cpp", "content": "void zzzName(){}"},
        {"id": "A", "name": "focus", "language": "cpp",
         "content": "void focus(){ zzzName(); aaaRare(); }"},
    ] + [
        {"id": f"decoy{i}", "name": f"decoy{i}", "language": "cpp",
         "content": "void decoy(){ zzzName(); }"}
        for i in range(3)
    ]
    sym_map = {"aaaRare": ["B"]}
    edges = _build_graph(chunks, sym_map, associated_top_frac=0.3)
    assert edges[("A", "B")] == "associated"


def test_build_refs_persists_typed_edges(tmp_path):
    from chonks.storage.store import Store
    from chonks.index.graph.refs import build_refs

    store = Store(tmp_path / "t.db")
    with store._lock:
        store._conn.execute(
            "INSERT INTO chunks(id, path, language, chunk_type, name, start_line, "
            "end_line, content, indexed_at, metadata) VALUES(?,?,?,?,?,?,?,?,0.0,?)",
            ("a", "a.cpp", "cpp", "function_definition", "driver", 1, 1,
             "void driver() { helper(); }",
             '{"calls":["helper"],"imports":[],"inherits":[]}'),
        )
        store._conn.execute(
            "INSERT INTO chunks(id, path, language, chunk_type, name, start_line, "
            "end_line, content, indexed_at, metadata) VALUES(?,?,?,?,?,?,?,?,0.0,?)",
            ("b", "a.cpp", "cpp", "function_definition", "helper", 3, 3,
             "void helper() {}", None),
        )
        store._conn.commit()

    n = build_refs(store)
    assert n >= 1
    typed = store.get_all_refs_typed()
    assert ("a", "b", "calls") in typed
    store.close()


def _seed_two_named_chunks(store) -> None:
    with store._lock:
        store._conn.execute(
            "INSERT INTO chunks(id, path, language, chunk_type, name, start_line, "
            "end_line, content, indexed_at, metadata) VALUES(?,?,?,?,?,?,?,?,0.0,?)",
            ("a", "a.cpp", "cpp", "function_definition", "driver", 1, 1,
             "void driver() { helper(); }",
             '{"calls":["helper"],"imports":[],"inherits":[]}'),
        )
        store._conn.execute(
            "INSERT INTO chunks(id, path, language, chunk_type, name, start_line, "
            "end_line, content, indexed_at, metadata) VALUES(?,?,?,?,?,?,?,?,0.0,?)",
            ("b", "a.cpp", "cpp", "function_definition", "helper", 3, 3,
             "void helper() {}", None),
        )
        store._conn.commit()


def test_persist_pagerank_writes_scores(tmp_path):
    from chonks.storage.store import Store
    from chonks.index.graph.pagerank import persist_pagerank
    from chonks.index.graph.refs import build_refs

    store = Store(tmp_path / "t.db")
    _seed_two_named_chunks(store)
    build_refs(store)

    n = persist_pagerank(store)
    assert n == 2
    persisted = store.load_pagerank()
    assert set(persisted.keys()) == {"a", "b"}
    assert all(isinstance(v, float) for v in persisted.values())
    store.close()


def test_compute_pagerank_global_reads_persisted_scores_without_recompute(tmp_path):
    from chonks.storage.store import Store
    from chonks.index.graph.pagerank import compute_pagerank_global
    from chonks.index.graph.refs import build_refs

    store = Store(tmp_path / "t.db")
    _seed_two_named_chunks(store)
    build_refs(store)

    fake_scores = {"a": 12.5, "b": -3.0}  # not valid pagerank output - proves no recompute
    store.save_pagerank(fake_scores)

    result = compute_pagerank_global(store)
    assert result == fake_scores
    store.close()


def test_compute_pagerank_global_falls_back_to_live_compute_on_old_db(tmp_path):
    from chonks.storage.store import Store
    from chonks.index.graph.pagerank import compute_pagerank_global
    from chonks.index.graph.refs import build_refs

    store = Store(tmp_path / "t.db")
    _seed_two_named_chunks(store)
    build_refs(store)
    assert store.load_pagerank() == {}  # old-DB condition: table empty

    result = compute_pagerank_global(store)
    assert set(result.keys()) == {"a", "b"}
    assert all(isinstance(v, float) for v in result.values())
    store.close()


def test_persist_pagerank_end_to_end_via_index_pipeline(tmp_path):
    from chonks.index.pipeline import index_paths
    from chonks.storage.store import Store

    class _FakeEmbedder:
        model = "fake"
        url = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    (tmp_path / "a.cpp").write_text(
        "void helper() {}\nvoid driver() { helper(); }\n"
    )

    store = Store(tmp_path / "test.db")
    embedder = _FakeEmbedder()
    index_paths([str(tmp_path)], store, embedder, root=tmp_path)

    first = store.load_pagerank()
    assert first, "chunk_pagerank not populated after indexing"

    (tmp_path / "b.cpp").write_text(
        "void extra() {}\nvoid caller() { extra(); }\n"
    )
    index_paths([str(tmp_path)], store, embedder, root=tmp_path)

    second = store.load_pagerank()
    assert second, "chunk_pagerank empty after re-index"
    assert set(second.keys()) != set(first.keys()), (
        "re-index after a file change did not refresh persisted PageRank scores"
    )
    store.close()


def test_persist_pagerank_skip_gate_is_cumulative(tmp_path):
    """Small batches skip the recompute, but skipped churn is banked in meta
    and eventually crosses the threshold, triggering a full refresh."""
    from chonks.storage.store import Store
    from chonks.index.graph.pagerank import persist_pagerank
    from chonks.index.graph.refs import build_refs
    from chonks.core.edges import _PAGERANK_STALE_META_KEY

    store = Store(tmp_path / "t.db")
    # 10 named chunks -> 20% threshold = 2: single-chunk batches skip twice,
    # the third (cumulative 3 > 2) must recompute.
    with store._lock:
        for i in range(10):
            store._conn.execute(
                "INSERT INTO chunks(id, path, language, chunk_type, name, "
                "start_line, end_line, content, indexed_at, metadata) "
                "VALUES(?,?,?,?,?,?,?,?,0.0,?)",
                (f"c{i}", "a.cpp", "cpp", "function_definition", f"fn{i}",
                 i, i, f"void fn{i}() {{}}", None),
            )
        store._conn.commit()
    build_refs(store)

    fake = {f"c{i}": 99.0 + i for i in range(10)}  # sentinel: recompute replaces these
    store.save_pagerank(fake)

    for expected_bank in ("1", "2"):
        n = persist_pagerank(store, changed_ids={"c0"}, deleted_ids=set())
        assert n == 10
        assert store.load_pagerank() == fake, "skip path must not recompute"
        assert store.get_meta(_PAGERANK_STALE_META_KEY) == expected_bank

    persist_pagerank(store, changed_ids={"c0"}, deleted_ids=set())
    refreshed = store.load_pagerank()
    assert refreshed != fake, "cumulative churn over threshold must recompute"
    assert store.get_meta(_PAGERANK_STALE_META_KEY) == "0"
    store.close()


# ---------------------------------------------------------------------------
# Typed-edge PageRank weighting
# ---------------------------------------------------------------------------

def _seed_three_named_chunks(store) -> None:
    """Edges are inserted directly via store.insert_refs so the graph shape
    and edge_type per pair are fully controlled."""
    with store._lock:
        for cid, name in (("a", "alpha"), ("b", "beta"), ("c", "gamma")):
            store._conn.execute(
                "INSERT INTO chunks(id, path, language, chunk_type, name, "
                "start_line, end_line, content, indexed_at, metadata) "
                "VALUES(?,?,?,?,?,?,?,?,0.0,?)",
                (cid, "a.cpp", "cpp", "function_definition", name, 1, 1,
                 f"void {name}() {{}}", None),
            )
        store._conn.commit()


def test_pagerank_edge_type_weighting_changes_ranking(tmp_path):
    from chonks.storage.store import Store
    from chonks.index.graph.pagerank import _compute_pagerank_live

    store = Store(tmp_path / "t.db")
    _seed_three_named_chunks(store)
    # a needs >1 outgoing edge: with only one, nx.pagerank's out-weight
    # normalization would cancel the edge weight out entirely.
    store.insert_refs([("a", "b", "calls"), ("a", "c", "mentions")])
    store.commit()

    baseline = _compute_pagerank_live(store)
    weighted = _compute_pagerank_live(
        store, edge_type_weights={"calls": 10.0, "mentions": 0.1},
    )

    assert baseline != weighted, "edge-type weighting must change the scores"
    assert weighted["b"] > baseline["b"]
    assert weighted["c"] < baseline["c"]
    store.close()


def test_pagerank_default_weights_are_identical_to_pre_weighting_behaviour(tmp_path):
    """Regression pin: edge_type_weights=None (every existing caller's
    default) must reproduce the exact same scores as an explicit all-1.0
    weight map."""
    from chonks.storage.store import Store
    from chonks.index.graph.pagerank import _compute_pagerank_live
    from chonks.core.edges import DEFAULT_EDGE_TYPE_WEIGHTS

    store = Store(tmp_path / "t.db")
    _seed_three_named_chunks(store)
    store.insert_refs([("a", "b", "calls"), ("c", "b", "mentions"), ("b", "a", "xlang")])
    store.commit()

    default = _compute_pagerank_live(store)
    explicit_all_ones = _compute_pagerank_live(store, edge_type_weights=DEFAULT_EDGE_TYPE_WEIGHTS)

    assert default.keys() == explicit_all_ones.keys()
    for k in default:
        assert default[k] == pytest.approx(explicit_all_ones[k], abs=1e-12)
    store.close()


def test_persist_pagerank_threads_edge_type_weights(tmp_path):
    from chonks.storage.store import Store
    from chonks.index.graph.pagerank import persist_pagerank, _compute_pagerank_live

    store = Store(tmp_path / "t.db")
    _seed_three_named_chunks(store)
    store.insert_refs([("a", "b", "calls"), ("c", "b", "mentions")])
    store.commit()

    persist_pagerank(store, edge_type_weights={"calls": 10.0, "mentions": 0.1})
    persisted = store.load_pagerank()
    live = _compute_pagerank_live(store, edge_type_weights={"calls": 10.0, "mentions": 0.1})
    for k in live:
        assert persisted[k] == pytest.approx(live[k], abs=1e-12)
    store.close()


def test_index_paths_forwards_edge_type_weights_to_pagerank(tmp_path, monkeypatch):
    """edge_type_weights must reach persist_pagerank, not be dropped on the
    floor - the plumbing chonks/ops/index_cmd.py's CLI/config path relies on."""
    import chonks.index.pipeline as chunker_mod
    import chonks.index.postindex as postindex_mod
    from chonks.storage.store import Store

    class _FakeEmbedder:
        model = "fake"
        url = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    captured = {}
    real_persist_pagerank = postindex_mod.persist_pagerank
    def _spy_persist_pagerank(store, **kwargs):
        captured.update(kwargs)
        return real_persist_pagerank(store, **kwargs)
    monkeypatch.setattr(postindex_mod, "persist_pagerank", _spy_persist_pagerank)

    (tmp_path / "a.cpp").write_text(
        "void helper() {}\nvoid driver() { helper(); }\n"
    )
    store = Store(tmp_path / "t.db")
    weights = {"calls": 50.0, "mentions": 0.01}
    chunker_mod.index_paths(
        [str(tmp_path)], store, _FakeEmbedder(), root=tmp_path,
        edge_type_weights=weights,
    )
    store.close()

    assert captured.get("edge_type_weights") == weights, (
        "edge_type_weights passed to index_paths did not reach persist_pagerank"
    )


def test_index_paths_forwards_cap_mentions_fanout_to_build_refs(tmp_path, monkeypatch):
    """Same plumbing shape as edge_type_weights above, for cap_mentions_fanout."""
    import chonks.index.pipeline as chunker_mod
    import chonks.index.postindex as postindex_mod
    from chonks.storage.store import Store

    class _FakeEmbedder:
        model = "fake"
        url = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    captured = {}
    real_build_refs = postindex_mod.build_refs
    def _spy_build_refs(store, **kwargs):
        captured.update(kwargs)
        return real_build_refs(store, **kwargs)
    monkeypatch.setattr(postindex_mod, "build_refs", _spy_build_refs)

    (tmp_path / "a.cpp").write_text(
        "void helper() {}\nvoid driver() { helper(); }\n"
    )
    store = Store(tmp_path / "t.db")
    chunker_mod.index_paths(
        [str(tmp_path)], store, _FakeEmbedder(), root=tmp_path,
        cap_mentions_fanout=True,
    )
    store.close()

    assert captured.get("cap_mentions") is True, (
        "cap_mentions_fanout passed to index_paths did not reach build_refs"
    )


def test_index_paths_forwards_associated_top_frac_to_build_refs(tmp_path, monkeypatch):
    """Same plumbing shape as cap_mentions_fanout above, for associated_top_frac."""
    import chonks.index.pipeline as chunker_mod
    import chonks.index.postindex as postindex_mod
    from chonks.storage.store import Store

    class _FakeEmbedder:
        model = "fake"
        url = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    captured = {}
    real_build_refs = postindex_mod.build_refs
    def _spy_build_refs(store, **kwargs):
        captured.update(kwargs)
        return real_build_refs(store, **kwargs)
    monkeypatch.setattr(postindex_mod, "build_refs", _spy_build_refs)

    (tmp_path / "a.cpp").write_text(
        "void helper() {}\nvoid driver() { helper(); }\n"
    )
    store = Store(tmp_path / "t.db")
    chunker_mod.index_paths(
        [str(tmp_path)], store, _FakeEmbedder(), root=tmp_path,
        associated_top_frac=0.5,
    )
    store.close()

    assert captured.get("associated_top_frac") == 0.5, (
        "associated_top_frac passed to index_paths did not reach build_refs"
    )


def test_format_map_truncation_appends_footer():
    """The output must say when the budget cut files - the LLM consumer
    can't otherwise tell 'whole codebase' from 'top slice that fit'."""
    chunks = [
        _chunk("function_definition", f"fn_{i}", path=f"src/file_{i}.py")
        for i in range(20)
    ]
    scores = {c["id"]: 1.0 for c in chunks}
    output = _format_map(chunks, scores, token_budget=30)  # fits ~a few files
    assert "[truncated: showing " in output
    assert "of 20 files" in output
    assert "path_prefix" in output and "token_budget" in output


def test_format_map_no_footer_when_everything_fits():
    chunks = [_chunk("function_definition", "fn_a"), _chunk("function_definition", "fn_b")]
    scores = _scores(chunks)
    for budget in (None, 100000):
        output = _format_map(chunks, scores, token_budget=budget)
        assert "[truncated" not in output


def test_repomap_endpoint_applies_default_budget_to_scoped_calls(monkeypatch):
    import chonks.serve.app as serve_app
    import chonks.serve.models as serve_models

    captured = {}
    def fake_build_repomap(store, path_prefix=None, query=None, token_budget=None):
        captured["budget"] = token_budget
        return "map"
    monkeypatch.setattr(serve_app, "build_repomap", fake_build_repomap)
    monkeypatch.setattr(serve_app, "_get_project",
                        lambda name: {"store": object(), "repomap_cfg": {}})

    serve_app.repomap(serve_models.RepomapRequest(path_prefix="src/gfx/"))
    assert captured["budget"] == 8000, "scoped call must not be unbudgeted"

    serve_app.repomap(serve_models.RepomapRequest(path_prefix="src/gfx/", token_budget=123))
    assert captured["budget"] == 123, "explicit request budget must override"


# ---------------------------------------------------------------------------
# Directory overview (graph v2)
# ---------------------------------------------------------------------------

def _seed_hierarchy(store, file_paths: list[str]) -> None:
    """Registers files and rebuilds graph_nodes/graph_edges, matching what
    the real index pipeline does before repomap runs."""
    for i, p in enumerate(file_paths):
        store.upsert_file(p, size=1, mtime=0.0, content_hash=f"h{i}")
    store.commit()
    rebuild_hierarchy(store)


def _seed_symbol(store, path, name, *, kind="function_definition", lang="python"):
    store.insert_symbols([_sym(path, name, kind, lang=lang)])


def test_dir_overview_unscoped_map_has_correct_subtree_counts(tmp_path):
    from chonks.storage.store import Store
    from chonks.retrieval.repomap import build_repomap

    store = Store(tmp_path / "t.db")
    files = ["core/a.py", "core/b.py", "core/sub/c.py", "docs/readme.py"]
    _seed_hierarchy(store, files)
    for p in files:
        _seed_symbol(store, p, "fn_" + p.replace("/", "_"))

    out = build_repomap(store)
    assert out.startswith("# Directory overview")

    # core/sub and docs tie on count (1), so alpha order ("core/sub" < "docs") applies.
    core_idx = out.index("core/ (3 files)")
    sub_idx = out.index("  core/sub/ (1 file)")
    docs_idx = out.index("docs/ (1 file)")
    assert core_idx < sub_idx < docs_idx

    assert "core/a.py" in out
    assert "core/sub/c.py" in out
    assert "docs/readme.py" in out
    store.close()


def test_dir_overview_scoped_map_restricted_no_false_prefix_match(tmp_path):
    """path_prefix='core' must not naive-string-match 'core_x', and must not
    show core's own line."""
    from chonks.storage.store import Store
    from chonks.retrieval.repomap import build_repomap

    store = Store(tmp_path / "t.db")
    files = ["core/a.py", "core/sub/c.py", "core_x/d.py"]
    _seed_hierarchy(store, files)
    for p in files:
        _seed_symbol(store, p, "fn_" + p.replace("/", "_"))

    out = build_repomap(store, path_prefix="core")
    assert "# Directory overview" in out
    assert "core/sub/ (1 file)" in out
    assert "core_x" not in out.split("\n\n", 1)[0]  # not in the overview block
    lines = out.split("\n\n", 1)[0].splitlines()
    assert "core/ (" not in "\n".join(lines)
    store.close()


def test_dir_overview_absent_without_hierarchy_rebuild(tmp_path):
    """A DB that never called rebuild_hierarchy() must produce the
    pre-hierarchy format, with no overview."""
    from chonks.storage.store import Store
    from chonks.retrieval.repomap import build_repomap

    store = Store(tmp_path / "t.db")
    _seed_symbol(store, "core/a.py", "fn_a")

    out = build_repomap(store)
    assert "# Directory overview" not in out
    assert "core/a.py" in out
    store.close()


def test_dir_overview_tight_budget_stays_under_quarter_and_map_keeps_a_block(tmp_path):
    from chonks.storage.store import Store
    from chonks.retrieval.repomap import build_repomap

    store = Store(tmp_path / "t.db")
    files = [f"dir{i}/file{i}.py" for i in range(15)]
    _seed_hierarchy(store, files)
    for p in files:
        _seed_symbol(store, p, "fn_" + p.replace("/", "_"))

    token_budget = 40  # budget_chars = 160, overview must stay <= 40 chars
    out = build_repomap(store, token_budget=token_budget)

    if out.startswith("# Directory overview"):
        overview_block, _, rest = out.partition("\n\n")
        overview_text = overview_block + "\n\n"
        assert len(overview_text) <= (token_budget * 4) // 4
    else:
        rest = out

    assert any(p in rest for p in files)
    store.close()


def test_dir_overview_truncation_marker_lists_omitted_subtrees():
    """The truncation marker lists the top omitted top-level directories by
    count, capped at 5."""
    chunks = []
    for i in range(20):
        group = f"grp{i % 8}"  # 8 top-level groups, uneven sizes across 20 files
        chunks.append(_chunk("function_definition", f"fn_{i}", path=f"{group}/file_{i}.py"))
    scores = {c["id"]: 1.0 for c in chunks}

    output = _format_map(chunks, scores, token_budget=20)  # small: forces truncation

    assert "[truncated: showing " in output
    assert "omitted subtrees:" in output
    assert "narrow path_prefix or raise token_budget]" in output

    marker = output[output.index("[truncated"):]
    subtree_section = marker.split("omitted subtrees: ", 1)[1].split(" — narrow", 1)[0]
    groups = subtree_section.split(", ")
    assert len(groups) <= 5
    for g in groups:
        assert "(" in g and "files)" in g


def test_truncation_marker_groups_relative_to_common_scope():
    """When every considered file shares a directory prefix, omitted groups
    are named relative to that prefix (and full-from-root), not bucketed
    under the scope itself."""
    chunks = []
    for i in range(20):
        sub = f"sub{i % 4}"  # all under core/: 4 second-level groups
        chunks.append(_chunk("function_definition", f"fn_{i}", path=f"core/{sub}/file_{i}.py"))
    scores = {c["id"]: 1.0 for c in chunks}

    output = _format_map(chunks, scores, token_budget=20)  # forces truncation

    marker = output[output.index("[truncated"):]
    subtree_section = marker.split("omitted subtrees: ", 1)[1].split(" — narrow", 1)[0]
    group_names = [g.split(" (")[0] for g in subtree_section.split(", ")]
    assert "core" not in group_names  # the scope itself is not a group
    assert all(name.startswith("core/sub") for name in group_names)


def test_dir_overview_depth_cap_excludes_level_3_but_rolls_up_count(tmp_path):
    """A dir 3 levels below the scope root isn't listed itself, but its
    files still count toward its level-2 ancestor's subtree total."""
    from chonks.storage.store import Store
    from chonks.retrieval.repomap import build_repomap

    store = Store(tmp_path / "t.db")
    files = ["a/b/c/d.py", "a/b/other.py"]
    _seed_hierarchy(store, files)
    for p in files:
        _seed_symbol(store, p, "fn_" + p.replace("/", "_"))

    out = build_repomap(store)
    overview = out.split("\n\n", 1)[0]
    assert "a/b/c/" not in overview  # depth-3 dir excluded
    assert "a/b/ (2 files)" in overview  # other.py (direct) + c/d.py (subtree)
    store.close()
