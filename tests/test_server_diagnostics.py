"""Tests for the /symbol and /usages endpoint miss diagnostics."""
import json

from chonks.chunking import segment_file
from chonks.searcher import Searcher
from chonks.store import Store

import chonks.server as server


def _fake_embedding(dim: int = 4) -> list[float]:
    return [0.5] * dim


class _NoEmbedder:
    def embed_queries(self, queries, client, timeout=30.0):
        raise AssertionError("embedder should not be called by these endpoints")


def _setup_default_project(monkeypatch, tmp_path) -> dict:
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **kw: None)
    server._projects.clear()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}))
    server.main(["--config", str(config_path), "--db", str(tmp_path / "x.db")])

    project = server._projects[server.DEFAULT_PROJECT]
    project["store"] = Store(project["db_path"])
    project["searcher"] = Searcher(project["store"], _NoEmbedder())
    return project


def test_symbol_endpoint_miss_gets_routing_note(monkeypatch, tmp_path):
    _setup_default_project(monkeypatch, tmp_path)
    req = server.SymbolRequest(name="m_somefield")
    body = json.loads(server.symbol(req).body)

    assert body["symbols"] == []
    assert "m_somefield" in body["note"]
    assert "codebase_search mode=fts" in body["note"]


def test_symbol_endpoint_hit_has_no_note(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    store.insert_chunks(
        [{"id": "c1", "path": "a.cpp", "language": "cpp", "chunk_type": "function",
          "name": "doThing", "start_line": 1, "end_line": 2, "content": "void doThing() {}"}],
        [_fake_embedding()],
    )
    store.insert_symbols([{
        "path": "a.cpp", "name": "doThing", "kind": "function_definition",
        "language": "cpp", "start_line": 1, "end_line": 2, "chunk_id": "c1",
    }])

    req = server.SymbolRequest(name="doThing")
    body = json.loads(server.symbol(req).body)
    assert len(body["symbols"]) == 1
    assert body["note"] is None


def test_usages_endpoint_qualified_miss_gets_note(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    store.insert_chunks(
        [{"id": "c1", "path": "manager.cpp", "language": "cpp", "chunk_type": "class",
          "name": "Manager", "start_line": 1, "end_line": 2, "content": "class Manager {}"}],
        [_fake_embedding()],
    )
    store.insert_symbols([{
        "path": "manager.cpp", "name": "Init", "kind": "function_definition",
        "language": "cpp", "start_line": 1, "end_line": 2, "chunk_id": "c1",
    }])

    req = server.UsagesRequest(name="Manager::Init")
    body = json.loads(server.usages(req).body)

    assert body["usages"] == []
    assert body["note"] == "symbol not found — try the bare name 'Init'"
    assert body["content_matches"] == []


def test_outgoing_endpoint_happy_path(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    store.insert_chunks(
        [{"id": "c1", "path": "widget.cpp", "language": "cpp", "chunk_type": "class",
          "name": "Widget", "start_line": 1, "end_line": 2, "content": "class Widget {}"},
         {"id": "c2", "path": "helper.cpp", "language": "cpp", "chunk_type": "function",
          "name": "helperFn", "start_line": 1, "end_line": 2, "content": "void helperFn() {}"}],
        [_fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([{
        "path": "widget.cpp", "name": "applyStep", "kind": "function_definition",
        "language": "cpp", "start_line": 1, "end_line": 2, "chunk_id": "c1",
    }])
    store.insert_refs([("c1", "c2")])

    req = server.OutgoingRequest(name="applyStep")
    body = json.loads(server.outgoing(req).body)

    assert body["count"] == 1
    assert body["outgoing"][0]["chunk_id"] == "c2"
    assert body["outgoing"][0]["provenance"] == "inferred"
    assert body["note"] is None


def test_outgoing_endpoint_miss_gets_note(monkeypatch, tmp_path):
    _setup_default_project(monkeypatch, tmp_path)
    req = server.OutgoingRequest(name="doesNotExist")
    body = json.loads(server.outgoing(req).body)

    assert body["outgoing"] == []
    assert body["note"] == "symbol not found"


def _seed_investigate_fixture(store) -> None:
    store.insert_chunks(
        [{"id": "c1", "path": "widget.cpp", "language": "cpp", "chunk_type": "class",
          "name": "Widget", "start_line": 1, "end_line": 2, "content": "class Widget {}"},
         {"id": "c2", "path": "caller.cpp", "language": "cpp", "chunk_type": "function",
          "name": "run", "start_line": 1, "end_line": 2, "content": "void run() { applyStep(); }"},
         {"id": "c3", "path": "helper.cpp", "language": "cpp", "chunk_type": "function",
          "name": "helperFn", "start_line": 1, "end_line": 2, "content": "void helperFn() {}"}],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([{
        "path": "widget.cpp", "name": "applyStep", "kind": "function_definition",
        "language": "cpp", "start_line": 1, "end_line": 2, "chunk_id": "c1",
    }])
    store.insert_refs([("c2", "c1"), ("c1", "c3")])


def test_investigate_endpoint_returns_all_four_legs(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    _seed_investigate_fixture(project["store"])

    req = server.InvestigateRequest(name="applyStep")
    body = json.loads(server.investigate(req).body)

    assert body["symbol"] == "applyStep"
    assert len(body["definitions"]) == 1
    assert body["usages"]["count"] == 1
    assert body["usages"]["usages"][0]["chunk_id"] == "c2"
    assert body["outgoing"]["count"] == 1
    assert body["outgoing"]["outgoing"][0]["chunk_id"] == "c3"
    assert body["impact"]["total_references"] == 1
    assert body["notes"] == {
        "definitions": None, "usages": None, "outgoing": None, "impact": None,
    }


def test_investigate_endpoint_attaches_definition_source_by_default(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    _seed_investigate_fixture(project["store"])

    req = server.InvestigateRequest(name="applyStep")
    body = json.loads(server.investigate(req).body)

    assert len(body["definitions"]) == 1
    assert body["definitions"][0]["source"] == "class Widget {}"


def test_investigate_endpoint_omits_source_when_disabled(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    _seed_investigate_fixture(project["store"])

    req = server.InvestigateRequest(name="applyStep", definition_source=False)
    body = json.loads(server.investigate(req).body)

    assert "source" not in body["definitions"][0]


def test_investigate_endpoint_truncates_source_at_small_budget(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    long_content = "class Widget {\n" + "  int x;\n" * 50 + "}"
    store.insert_chunks(
        [{"id": "c1", "path": "widget.cpp", "language": "cpp", "chunk_type": "class",
          "name": "Widget", "start_line": 1, "end_line": 52, "content": long_content}],
        [_fake_embedding()],
    )
    store.insert_symbols([{
        "path": "widget.cpp", "name": "applyStep", "kind": "function_definition",
        "language": "cpp", "start_line": 1, "end_line": 52, "chunk_id": "c1",
    }])

    req = server.InvestigateRequest(name="applyStep", definition_source_max_chars=200)
    body = json.loads(server.investigate(req).body)

    source = body["definitions"][0]["source"]
    assert source.startswith(long_content[:200])
    assert "truncated at 200 chars" in source
    assert "widget.cpp:1-52" in source


def test_investigate_endpoint_short_circuits_on_total_miss(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]

    def _boom(*a, **kw):
        raise AssertionError("should not be called on a total miss")

    monkeypatch.setattr(store, "find_usages", _boom)
    monkeypatch.setattr(store, "find_outgoing", _boom)
    monkeypatch.setattr(store, "get_impact", _boom)

    req = server.InvestigateRequest(name="doesNotExist")
    body = json.loads(server.investigate(req).body)

    assert body["symbol"] == "doesNotExist"
    assert body["definitions"] == []
    assert body["usages"] == {"usages": [], "count": 0}
    assert body["outgoing"] == {"outgoing": [], "count": 0}
    assert body["impact"] == {}
    note = body["notes"]["definitions"]
    assert note == body["notes"]["usages"] == body["notes"]["outgoing"] == body["notes"]["impact"]
    assert "doesNotExist" in note


def test_investigate_endpoint_respects_per_leg_limits(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    chunks = [{"id": "c1", "path": "widget.cpp", "language": "cpp", "chunk_type": "class",
               "name": "Widget", "start_line": 1, "end_line": 2, "content": "class Widget {}"}]
    for i in range(3):
        chunks.append({
            "id": f"caller{i}", "path": f"caller{i}.cpp", "language": "cpp",
            "chunk_type": "function", "name": f"run{i}", "start_line": 1, "end_line": 2,
            "content": f"void run{i}() {{ applyStep(); }}",
        })
        chunks.append({
            "id": f"callee{i}", "path": f"callee{i}.cpp", "language": "cpp",
            "chunk_type": "function", "name": f"helper{i}", "start_line": 1, "end_line": 2,
            "content": f"void helper{i}() {{}}",
        })
    store.insert_chunks(chunks, [_fake_embedding()] * len(chunks))
    store.insert_symbols([{
        "path": "widget.cpp", "name": "applyStep", "kind": "function_definition",
        "language": "cpp", "start_line": 1, "end_line": 2, "chunk_id": "c1",
    }])
    store.insert_refs(
        [(f"caller{i}", "c1") for i in range(3)] + [("c1", f"callee{i}") for i in range(3)]
    )

    req = server.InvestigateRequest(name="applyStep", usages_limit=1, outgoing_limit=2, impact_limit=1)
    body = json.loads(server.investigate(req).body)

    assert body["usages"]["count"] == 1
    assert body["outgoing"]["count"] == 2
    assert len(body["impact"]["files"]) == 1


def test_investigate_path_prefix_never_scopes_definitions(monkeypatch, tmp_path):
    """path_prefix scopes the fan-out legs only. A symbol defined outside
    the prefix must still resolve, not short-circuit to "symbol not
    found"."""
    project = _setup_default_project(monkeypatch, tmp_path)
    _seed_investigate_fixture(project["store"])

    req = server.InvestigateRequest(name="applyStep", path_prefix="caller")
    body = json.loads(server.investigate(req).body)

    assert len(body["definitions"]) == 1          # widget.cpp, outside the prefix
    assert body["notes"]["definitions"] is None
    assert body["usages"]["count"] == 1           # caller.cpp is inside the prefix
    assert body["outgoing"]["count"] == 0         # helper.cpp is outside it


def test_investigate_endpoint_stitches_split_definition_under_budget(monkeypatch, tmp_path):
    """A definition split across chunks: the symbol row's end_line outgrows
    the pointed-at chunk's own end_line, so its continuation chunk must be
    stitched in rather than silently cut at the first chunk's boundary."""
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    part1 = "void bigFn() {\n  int a = 1;\n  int b = 2;"
    part2 = "  return a + b;\n}"
    store.insert_chunks(
        [{"id": "c1", "path": "big.cpp", "language": "cpp", "chunk_type": "function",
          "name": "bigFn", "start_line": 1, "end_line": 5, "content": part1},
         {"id": "c2", "path": "big.cpp", "language": "cpp", "chunk_type": "function",
          "name": None, "start_line": 6, "end_line": 7, "content": part2}],
        [_fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([{
        "path": "big.cpp", "name": "bigFn", "kind": "function_definition",
        "language": "cpp", "start_line": 1, "end_line": 7, "chunk_id": "c1",
    }])

    req = server.InvestigateRequest(name="bigFn")
    body = json.loads(server.investigate(req).body)

    source = body["definitions"][0]["source"]
    assert source == part1 + "\n" + part2
    assert "chunk boundary" not in source
    assert "truncated" not in source


def test_investigate_endpoint_truncates_stitched_source_and_names_full_range(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    part1 = "void bigFn() {\n" + "  int a = 1;\n" * 10
    part2 = "  int b = 2;\n" * 9 + "  return a + b;\n}"
    store.insert_chunks(
        [{"id": "c1", "path": "big.cpp", "language": "cpp", "chunk_type": "function",
          "name": "bigFn", "start_line": 1, "end_line": 11, "content": part1},
         {"id": "c2", "path": "big.cpp", "language": "cpp", "chunk_type": "function",
          "name": None, "start_line": 12, "end_line": 22, "content": part2}],
        [_fake_embedding(), _fake_embedding()],
    )
    store.insert_symbols([{
        "path": "big.cpp", "name": "bigFn", "kind": "function_definition",
        "language": "cpp", "start_line": 1, "end_line": 22, "chunk_id": "c1",
    }])

    req = server.InvestigateRequest(name="bigFn", definition_source_max_chars=200)
    body = json.loads(server.investigate(req).body)

    source = body["definitions"][0]["source"]
    assert source.startswith((part1 + part2)[:200])
    assert "truncated at 200 chars" in source
    assert "big.cpp:1-22" in source


def test_investigate_endpoint_single_chunk_source_unchanged(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    _seed_investigate_fixture(project["store"])

    req = server.InvestigateRequest(name="applyStep")
    body = json.loads(server.investigate(req).body)

    assert body["definitions"][0]["source"] == "class Widget {}"


def test_investigate_endpoint_missing_continuation_chunk_gets_boundary_marker(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    part1 = "void bigFn() {\n  int a = 1;\n  int b = 2;"
    store.insert_chunks(
        [{"id": "c1", "path": "big.cpp", "language": "cpp", "chunk_type": "function",
          "name": "bigFn", "start_line": 1, "end_line": 5, "content": part1}],
        [_fake_embedding()],
    )
    store.insert_symbols([{
        "path": "big.cpp", "name": "bigFn", "kind": "function_definition",
        "language": "cpp", "start_line": 1, "end_line": 10, "chunk_id": "c1",
    }])

    req = server.InvestigateRequest(name="bigFn")
    body = json.loads(server.investigate(req).body)

    source = body["definitions"][0]["source"]
    assert source.startswith(part1)
    assert "chunk boundary at line 5" in source
    assert "definition continues to 10" in source
    assert "big.cpp:6-10" in source


def test_investigate_endpoint_stitches_real_chunker_split_without_duplication(monkeypatch, tmp_path):
    """Against the real chunker: continuation slices prepend overlap lines
    for context while start_line/end_line stay canonical, so a naive stitch
    would repeat lines at the seam."""
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]

    src_lines = ["def big_fn():"]
    for i in range(300):
        src_lines.append(f"    variable_{i} = {i} + compute_something({i})")
    src_lines.append("    return variable_0")
    src_text = "\n".join(src_lines) + "\n"
    segments = segment_file(src_text.encode(), "python", path="big.py")
    assert len(segments) > 1  # sanity: this function must actually get split

    chunks = [
        {"id": f"c{i}", "path": "big.py", "language": "python",
         "chunk_type": seg["chunk_type"], "name": seg["name"],
         "start_line": seg["start_line"], "end_line": seg["end_line"],
         "content": seg["content"]}
        for i, seg in enumerate(segments)
    ]
    store.insert_chunks(chunks, [_fake_embedding() for _ in chunks])
    store.insert_symbols([{
        "path": "big.py", "name": "big_fn", "kind": "function_definition",
        "language": "python", "start_line": segments[0]["start_line"],
        "end_line": segments[-1]["end_line"], "chunk_id": "c0",
    }])

    req = server.InvestigateRequest(name="big_fn", definition_source_max_chars=15000)
    body = json.loads(server.investigate(req).body)

    source = body["definitions"][0]["source"]
    expected = "\n".join(src_lines[segments[0]["start_line"] - 1:segments[-1]["end_line"]])
    assert source == expected
    assert "chunk boundary" not in source
    assert "truncated" not in source
