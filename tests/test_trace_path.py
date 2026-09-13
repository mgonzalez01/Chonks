from chonks.repomap import trace_path
from chonks.store import Store


def _make_store(tmp_path) -> Store:
    return Store(tmp_path / "test.db")


def _fake_embedding(dim: int = 4) -> list[float]:
    return [0.5] * dim


def _chunk(id, path, name, s=1, e=2):
    return {"id": id, "path": path, "language": "cpp", "chunk_type": "function",
            "name": name, "start_line": s, "end_line": e, "content": f"{name}()"}


def _sym(path, name, kind="function_definition", lang="cpp", s=1, e=2, chunk_id=None):
    return {"path": path, "name": name, "kind": kind, "language": lang,
            "start_line": s, "end_line": e, "chunk_id": chunk_id}


def _insert(store, chunks_defs, symbol_names, refs=(), neighbors=()):
    """chunks_defs: list of (id, path, name). symbol_names: {name: chunk_id}."""
    store.insert_chunks(
        [_chunk(cid, path, name) for cid, path, name in chunks_defs],
        [_fake_embedding() for _ in chunks_defs],
    )
    store.insert_symbols([
        _sym(path, sym_name, chunk_id=cid)
        for sym_name, cid in symbol_names.items()
        for (cid2, path, name) in chunks_defs if cid2 == cid
    ])
    if refs:
        store.insert_refs(list(refs))
    if neighbors:
        store.insert_neighbors(list(neighbors))


def test_trace_path_direct_edge(tmp_path):
    store = _make_store(tmp_path)
    _insert(
        store,
        [("a", "a.cpp", "callerFn"), ("b", "b.cpp", "calleeFn")],
        {"callerFn": "a", "calleeFn": "b"},
        refs=[("a", "b", "calls")],
    )
    result = trace_path(store, "callerFn", "calleeFn")
    assert result["found"] is True
    assert result["depth"] == 1
    assert len(result["hops"]) == 1
    hop = result["hops"][0]
    assert hop["from_chunk"]["chunk_id"] == "a"
    assert hop["to_chunk"]["chunk_id"] == "b"
    assert hop["edge_type"] == "calls"
    assert hop["direction"] == "forward"
    assert hop["provenance"] == "extracted"
    assert result["used_semantic"] is False


def test_trace_path_multi_hop(tmp_path):
    store = _make_store(tmp_path)
    _insert(
        store,
        [("a", "a.cpp", "fnA"), ("b", "b.cpp", "fnB"), ("c", "c.cpp", "fnC")],
        {"fnA": "a", "fnB": "b", "fnC": "c"},
        refs=[("a", "b", "calls"), ("b", "c", "calls")],
    )
    result = trace_path(store, "fnA", "fnC")
    assert result["found"] is True
    assert result["depth"] == 2
    chain = [result["hops"][0]["from_chunk"]["chunk_id"],
             result["hops"][0]["to_chunk"]["chunk_id"],
             result["hops"][1]["to_chunk"]["chunk_id"]]
    assert chain == ["a", "b", "c"]


def test_trace_path_backward_hop_direction(tmp_path):
    # The path can walk a chunk_refs edge against its arrow: direction must
    # reflect that ('backward'), not read as 'forward'.
    store = _make_store(tmp_path)
    _insert(
        store,
        [("a", "a.cpp", "fnA"), ("b", "b.cpp", "fnB")],
        {"fnA": "a", "fnB": "b"},
        refs=[("b", "a", "calls")],  # b calls a; path walks a -> b against the arrow
    )
    result = trace_path(store, "fnA", "fnB")
    assert result["found"] is True
    hop = result["hops"][0]
    assert hop["direction"] == "backward"
    assert hop["edge_type"] == "calls"


def test_trace_path_no_structural_path(tmp_path):
    store = _make_store(tmp_path)
    _insert(
        store,
        [("a", "a.cpp", "fnA"), ("b", "b.cpp", "fnB")],
        {"fnA": "a", "fnB": "b"},
    )
    result = trace_path(store, "fnA", "fnB")
    assert result["found"] is False
    assert "error" in result
    assert result["used_semantic"] is False


def test_trace_path_unknown_symbol(tmp_path):
    store = _make_store(tmp_path)
    _insert(
        store,
        [("a", "a.cpp", "fnA")],
        {"fnA": "a"},
    )
    result = trace_path(store, "doesNotExist", "fnA")
    assert result["found"] is False
    assert "doesNotExist" in result["error"]

    result2 = trace_path(store, "fnA", "alsoMissing")
    assert result2["found"] is False
    assert "alsoMissing" in result2["error"]


def test_trace_path_multiple_definitions_picks_reachable_one(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("d1", "unrelated.cpp", "Other"),
         _chunk("d2", "real.cpp", "Real"),
         _chunk("target", "target.cpp", "targetFn")],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([
        _sym("unrelated.cpp", "dup", chunk_id="d1"),
        _sym("real.cpp", "dup", chunk_id="d2"),
        _sym("target.cpp", "targetFn", chunk_id="target"),
    ])
    store.insert_refs([("d2", "target", "calls")])  # only d2 connects

    result = trace_path(store, "dup", "targetFn")
    assert result["found"] is True
    assert result["hops"][0]["from_chunk"]["chunk_id"] == "d2"


def test_trace_path_fanout_cap_prioritizes_typed_edges(tmp_path):
    # With max_fanout=1, the cap must keep the typed edge over an arbitrary
    # mentions edge, so the path is still found.
    store = _make_store(tmp_path)
    chunks = [_chunk("src", "src.cpp", "fnSrc"), _chunk("dst", "dst.cpp", "fnDst")]
    chunks += [_chunk(f"noise{i}", f"noise{i}.cpp", f"noise{i}") for i in range(5)]
    store.insert_chunks(chunks, [_fake_embedding() for _ in chunks])
    store.insert_symbols([
        _sym("src.cpp", "fnSrc", chunk_id="src"),
        _sym("dst.cpp", "fnDst", chunk_id="dst"),
    ] + [_sym(f"noise{i}.cpp", f"noise{i}", chunk_id=f"noise{i}") for i in range(5)])
    refs = [("src", f"noise{i}", "mentions") for i in range(5)]
    refs.append(("src", "dst", "calls"))
    store.insert_refs(refs)

    result = trace_path(store, "fnSrc", "fnDst", max_fanout=1)
    assert result["found"] is True
    assert result["hops"][0]["edge_type"] == "calls"


def test_trace_path_fanout_cap_prioritizes_typed_over_associated(tmp_path):
    # 'associated' is the PMI-promoted label, not raw 'mentions'; _edge_rank
    # must still rank it below typed edges or it ties with 'calls' at rank 0
    # and can evict the real target from a capped fanout.
    store = _make_store(tmp_path)
    chunks = [_chunk("src", "src.cpp", "fnSrc"), _chunk("dst", "dst.cpp", "fnDst")]
    chunks += [_chunk(f"noise{i}", f"noise{i}.cpp", f"noise{i}") for i in range(5)]
    store.insert_chunks(chunks, [_fake_embedding() for _ in chunks])
    store.insert_symbols([
        _sym("src.cpp", "fnSrc", chunk_id="src"),
        _sym("dst.cpp", "fnDst", chunk_id="dst"),
    ] + [_sym(f"noise{i}.cpp", f"noise{i}", chunk_id=f"noise{i}") for i in range(5)])
    refs = [("src", f"noise{i}", "associated") for i in range(5)]
    refs.append(("src", "dst", "calls"))
    store.insert_refs(refs)

    result = trace_path(store, "fnSrc", "fnDst", max_fanout=1)
    assert result["found"] is True
    assert result["hops"][0]["edge_type"] == "calls"


def test_trace_path_semantic_fallback_only_when_allowed(tmp_path):
    store = _make_store(tmp_path)
    _insert(
        store,
        [("a", "a.cpp", "fnA"), ("b", "b.cpp", "fnB")],
        {"fnA": "a", "fnB": "b"},
        neighbors=[("a", "b", 0.1)],
    )
    no_fallback = trace_path(store, "fnA", "fnB", include_semantic=False)
    assert no_fallback["found"] is False

    with_fallback = trace_path(store, "fnA", "fnB", include_semantic=True)
    assert with_fallback["found"] is True
    assert with_fallback["used_semantic"] is True
    hop = with_fallback["hops"][0]
    assert hop["edge_type"] == "semantic"
    assert hop["direction"] == "semantic"
    # Edge provenance: a semantic k-NN hop is always "inferred" provenance.
    assert hop["provenance"] == "inferred"


def test_trace_path_prefers_structural_over_semantic(tmp_path):
    store = _make_store(tmp_path)
    _insert(
        store,
        [("a", "a.cpp", "fnA"), ("b", "b.cpp", "fnB")],
        {"fnA": "a", "fnB": "b"},
        refs=[("a", "b", "calls")],
        neighbors=[("a", "b", 0.1)],
    )
    result = trace_path(store, "fnA", "fnB", include_semantic=True)
    assert result["found"] is True
    assert result["used_semantic"] is False
    assert result["hops"][0]["edge_type"] == "calls"


def test_trace_path_max_depth_cap(tmp_path):
    store = _make_store(tmp_path)
    chunks = [_chunk(f"c{i}", f"f{i}.cpp", f"fn{i}") for i in range(5)]
    store.insert_chunks(chunks, [_fake_embedding() for _ in chunks])
    store.insert_symbols([_sym(f"f{i}.cpp", f"fn{i}", chunk_id=f"c{i}") for i in range(5)])
    store.insert_refs([(f"c{i}", f"c{i+1}", "calls") for i in range(4)])  # chain of 4 hops

    assert trace_path(store, "fn0", "fn4", max_depth=1)["found"] is False
    assert trace_path(store, "fn0", "fn4", max_depth=4)["found"] is True


def test_trace_path_hop_provenance_mapping(tmp_path):
    # Each hop's provenance is edge_type through edge_provenance; an
    # edge_type unknown to the map degrades to 'inferred' rather than crashing.
    store = _make_store(tmp_path)
    _insert(
        store,
        [("a", "a.cpp", "fnA"), ("b", "b.cpp", "fnB")],
        {"fnA": "a", "fnB": "b"},
        refs=[("a", "b", "xlang")],
    )
    result = trace_path(store, "fnA", "fnB")
    assert result["found"] is True
    assert result["hops"][0]["edge_type"] == "xlang"
    assert result["hops"][0]["provenance"] == "paired"

    store2 = _make_store(tmp_path.parent / (tmp_path.name + "_2"))
    _insert(
        store2,
        [("a", "a.cpp", "fnA"), ("b", "b.cpp", "fnB")],
        {"fnA": "a", "fnB": "b"},
        refs=[("a", "b", "mystery_future_edge_type")],
    )
    result2 = trace_path(store2, "fnA", "fnB")
    assert result2["found"] is True
    assert result2["hops"][0]["provenance"] == "inferred"
