"""Notes and warnings for scopes and config entries that can't match anything."""
import collections
import json
import logging

import pytest

import chonks.serve.app as serve_app
import chonks.serve.main as serve_main
import chonks.serve.projects as serve_projects
from chonks.retrieval.scope_notes import config_warnings, excluded_scope_note
from chonks.retrieval.searcher import Searcher
from chonks.storage.store import Store

_BUILD_NOTE = "build/ is excluded from the index by config (exclude: build/), so nothing under it can match."


@pytest.mark.parametrize("path_prefix,include,excluded", [
    ("build/", [], True),
    ("build", [], True),
    ("build/sub/", [], True),
    ("src/", [], False),
    ("buildtools/", [], False),
    ("build/", ["build/gen/"], False),
    ("build/gen/", ["build/gen/"], False),
    (None, [], False),
], ids=["dir", "no-slash", "under", "other", "sibling-name", "include-beneath", "include-itself", "none"])
def test_excluded_scope_note(path_prefix, include, excluded):
    note = excluded_scope_note(path_prefix, ["build/"], include)
    assert (note is not None) == excluded
    if excluded:
        assert f"{path_prefix} is excluded from the index by config (exclude: build/)" in note


def test_config_warnings_name_missing_entries_and_excluded_scopes(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "build").mkdir()
    warnings = config_warnings(tmp_path, ["build/", "gone"], ["pkg@1.2.3/"], ["build/", "src/", "build/"])
    assert warnings == [
        f"exclude entry gone/ matches nothing under {tmp_path}",
        f"include entry pkg@1.2.3/ matches nothing under {tmp_path}",
        _BUILD_NOTE,
    ]
    assert config_warnings(None, ["gone/"], [], []) == []


class _NoEmbedder:
    def embed_queries(self, queries, client, timeout=30.0):
        raise AssertionError("keyword search never needs the embedder")


def _serve(monkeypatch, tmp_path, config: dict) -> dict:
    monkeypatch.setattr(serve_main.uvicorn, "run", lambda *a, **kw: None)
    serve_projects._projects.clear()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    serve_main.main(["--config", str(config_path), "--db", str(tmp_path / "x.db")])
    project = serve_projects._projects[serve_projects.DEFAULT_PROJECT]
    project["store"] = Store(project["db_path"])
    project["searcher"] = Searcher(project["store"], _NoEmbedder())
    return project


def test_search_and_research_say_when_the_scope_is_excluded(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    _serve(monkeypatch, tmp_path, {"codebase": str(tmp_path), "exclude": ["build/"]})
    monkeypatch.setattr(serve_app, "deep_research", lambda *a, **k: {
        "chunks": [], "count": 0, "iterations": 0, "connections": [], "degraded": None})
    with TestClient(serve_app.app) as client:
        excluded = client.post("/search", json={"query": "Renderer", "mode": "fts", "path_prefix": "build/"})
        included = client.post("/search", json={"query": "Renderer", "mode": "fts", "path_prefix": "src/"})
        research = client.post("/research", json={"query": "Renderer", "path_prefix": "build/"})
    assert excluded.json()["note"] == _BUILD_NOTE
    assert included.json()["note"] is None
    assert research.json()["note"] == _BUILD_NOTE


def test_status_warns_about_client_scopes_and_missing_entries(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    (tmp_path / "build").mkdir()
    _serve(monkeypatch, tmp_path, {"codebase": str(tmp_path), "exclude": ["build/"], "include": ["gone/"]})
    with TestClient(serve_app.app) as client:
        body = client.get("/status", params={"scope": ["build/", "src/"]}).json()
    assert body["warnings"] == [f"include entry gone/ matches nothing under {tmp_path}", _BUILD_NOTE]


def test_doctor_reports_config_problems(tmp_path):
    from chonks.ops.doctor import build_report
    from chonks.storage.readonly import _connect_readonly

    (tmp_path / "build").mkdir()
    Store(tmp_path / "test.db").close()
    config = {"codebase": str(tmp_path), "exclude": ["build/"], "subsystems": {"build": ["build/"]}}
    conn = _connect_readonly(str(tmp_path / "test.db"))
    try:
        report = build_report(conn, config, str(tmp_path / "test.db"))
    finally:
        conn.close()
    section = report.split("== Config ==")[1].split("\n== ")[0]
    assert section.strip() == f"!! {_BUILD_NOTE}"


def test_index_warns_about_config_problems_at_start(tmp_path, monkeypatch, caplog):
    import chonks.ops.index_cmd as index_cmd

    class FakeStore:
        def __init__(self, db, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stats(self): return {}
        def path_family_rows(self): return []
    monkeypatch.setattr(index_cmd, "Store", FakeStore)
    monkeypatch.setattr(index_cmd, "index_paths", lambda *a, **k: collections.defaultdict(int))

    src = tmp_path / "src"
    src.mkdir()
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"codebase": str(tmp_path), "include": ["pkg@1.2.3/"]}))
    with caplog.at_level(logging.WARNING, logger="chonks.chunker"):
        try:
            index_cmd.main(["--config", str(cfg), "--db", str(tmp_path / "x.db"), str(src)])
        except SystemExit:
            pass
    assert f"include entry pkg@1.2.3/ matches nothing under {tmp_path}" in caplog.messages
