import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import chonks.serve.app as serve_app
import chonks.serve.main as serve_main
import chonks.serve.models as serve_models
import chonks.serve.projects as serve_projects


def _run_main_with_config(monkeypatch, tmp_path, config: dict) -> None:
    # uvicorn.run stubbed so main() populates _projects and returns instead
    # of blocking on a real server.
    monkeypatch.setattr(serve_main.uvicorn, "run", lambda *a, **kw: None)
    serve_projects._projects.clear()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    serve_main.main(["--config", str(config_path), "--db", str(tmp_path / "x.db")])


def _default_db_for(monkeypatch, tmp_path, config: dict, *argv: str):
    monkeypatch.setattr(serve_main.uvicorn, "run", lambda *a, **kw: None)
    monkeypatch.chdir(tmp_path)
    serve_projects._projects.clear()
    (tmp_path / "config.json").write_text(json.dumps(config))
    serve_main.main(["--config", str(tmp_path / "config.json"), *argv])
    return serve_projects._projects[serve_projects.DEFAULT_PROJECT]["db_path"]


def test_db_flag_overrides_the_config_db(monkeypatch, tmp_path):
    got = _default_db_for(monkeypatch, tmp_path, {"db": "from_config.db"}, "--db", "from_flag.db")
    assert got == Path("from_flag.db")


def test_config_db_is_used_without_the_flag(monkeypatch, tmp_path):
    assert _default_db_for(monkeypatch, tmp_path, {"db": "from_config.db"}) == Path("from_config.db")


def test_db_defaults_without_flag_or_config(monkeypatch, tmp_path):
    assert _default_db_for(monkeypatch, tmp_path, {}) == Path(".db/chonks.db")


def test_default_project_picks_up_top_level_edge_type_weights(monkeypatch, tmp_path):
    weights = {"calls": 5.0, "mentions": 0.2}
    _run_main_with_config(monkeypatch, tmp_path, {"edge_type_weights": weights})
    assert serve_projects._projects[serve_projects.DEFAULT_PROJECT]["edge_type_weights"] == weights


def test_default_project_defaults_to_empty_when_unset(monkeypatch, tmp_path):
    _run_main_with_config(monkeypatch, tmp_path, {})
    assert serve_projects._projects[serve_projects.DEFAULT_PROJECT]["edge_type_weights"] == {}


def test_named_project_inherits_top_level_edge_type_weights(monkeypatch, tmp_path):
    weights = {"calls": 3.0}
    db_path = tmp_path / "proj.db"
    _run_main_with_config(monkeypatch, tmp_path, {
        "edge_type_weights": weights,
        "projects": {"proj": {"db": str(db_path)}},
    })
    assert serve_projects._projects["proj"]["edge_type_weights"] == weights


def test_named_project_can_override_top_level_edge_type_weights(monkeypatch, tmp_path):
    db_path = tmp_path / "proj.db"
    override = {"calls": 99.0}
    _run_main_with_config(monkeypatch, tmp_path, {
        "edge_type_weights": {"calls": 3.0},
        "projects": {"proj": {"db": str(db_path), "edge_type_weights": override}},
    })
    assert serve_projects._projects["proj"]["edge_type_weights"] == override


def test_project_language_plugins_is_refused(monkeypatch, tmp_path, capsys):
    # main() reconfigures logging with force=True, which drops caplog's
    # handler; the log stream is sys.stderr, which capsys owns.
    db_path = tmp_path / "proj.db"
    _run_main_with_config(monkeypatch, tmp_path, {
        "projects": {"proj": {"db": str(db_path), "language_plugins": ["whatever"]}},
    })
    assert "proj" not in serve_projects._projects
    err = capsys.readouterr().err
    assert "language_plugins" in err and "proj" in err


def test_research_endpoint_injects_edge_type_weights_into_cfg(monkeypatch, tmp_path):
    weights = {"calls": 7.0, "mentions": 0.3}
    _run_main_with_config(monkeypatch, tmp_path, {"edge_type_weights": weights})

    captured = {}
    def fake_deep_research(query, searcher, cfg=None, path_prefix=None, compact=False):
        captured["cfg"] = cfg
        return {"chunks": [], "count": 0, "iterations": 0}
    monkeypatch.setattr(serve_app, "deep_research", fake_deep_research)

    # Avoid opening a real Store/Searcher for the default project.
    project = serve_projects._projects[serve_projects.DEFAULT_PROJECT]
    project["store"] = object()
    project["searcher"] = object()

    req = serve_models.ResearchRequest(query="q")
    serve_app.research(req)
    assert captured["cfg"]["edge_type_weights"] == weights


def test_research_endpoint_request_level_edge_type_weights_overrides_config(monkeypatch, tmp_path):
    # Precedence: request > project config > all-1.0 default.
    config_weights = {"calls": 7.0, "mentions": 0.3}
    _run_main_with_config(monkeypatch, tmp_path, {"edge_type_weights": config_weights})

    captured = {}
    def fake_deep_research(query, searcher, cfg=None, path_prefix=None, compact=False):
        captured["cfg"] = cfg
        return {"chunks": [], "count": 0, "iterations": 0}
    monkeypatch.setattr(serve_app, "deep_research", fake_deep_research)

    project = serve_projects._projects[serve_projects.DEFAULT_PROJECT]
    project["store"] = object()
    project["searcher"] = object()

    override = {"calls": 8.0, "imports": 8.0, "inherits": 8.0, "xlang": 8.0, "mentions": 1.0}
    req = serve_models.ResearchRequest(query="q", edge_type_weights=override)
    serve_app.research(req)
    assert captured["cfg"]["edge_type_weights"] == override


def test_research_request_negative_edge_type_weight_rejected():
    with pytest.raises(ValidationError):
        serve_models.ResearchRequest(query="q", edge_type_weights={"calls": -1.0})


class _NoGraphStore:
    def get_neighbors(self, cid, limit=None):
        return []

    def get_refs_for_chunks_typed(self, ids):
        return []

    def get_refs_to_chunks_typed(self, ids):
        return []


class _OneHitSearcher:
    store = _NoGraphStore()

    def embed_query(self, q):
        return [1.0]

    def semantic(self, q, k, p):
        return [{"id": "ready", "_score": 0.5, "path": "scene/main/node.cpp", "name": "Node::_ready",
                 "start_line": 10, "end_line": 20, "content": "void Node::_ready() {}"}]

    def regex(self, sym, top_k=10, path_prefix=None):
        return []


def test_research_endpoint_compact_returns_chunks_without_content(monkeypatch, tmp_path):
    _run_main_with_config(monkeypatch, tmp_path, {})
    project = serve_projects._projects[serve_projects.DEFAULT_PROJECT]
    project["store"] = object()
    project["searcher"] = _OneHitSearcher()

    def post(body: dict) -> dict:
        return json.loads(serve_app.research(serve_models.ResearchRequest.model_validate(body)).body)

    full = post({"query": "q"})
    compact = post({"query": "q", "compact": True})
    assert full["chunks"][0]["content"] == "void Node::_ready() {}"
    assert compact["chunks"] == [{k: v for k, v in full["chunks"][0].items() if k != "content"}]
    assert compact["files"] == full["files"] != []




# ---- serve refuses to start on a dead embedder -----------------------------

def _main_with_probe(monkeypatch, tmp_path, probe_result, extra_args=()):
    monkeypatch.setattr(serve_main.uvicorn, "run", lambda *a, **kw: None)
    monkeypatch.setattr(serve_main, "probe_embedder", lambda embedder, timeout=10.0: probe_result)
    serve_projects._projects.clear()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"embed_url": "http://nowhere:1/v1/embeddings"}))
    serve_main.main(["--config", str(config_path), "--db", str(tmp_path / "x.db"), *extra_args])


def test_serve_exits_when_embedder_probe_fails(monkeypatch, tmp_path, capsys):
    # main() reconfigures logging with force=True, which drops caplog's
    # handler; the log stream is sys.stderr, which capsys owns.
    with pytest.raises(SystemExit) as exc:
        _main_with_probe(monkeypatch, tmp_path, "ConnectError: refused")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "http://nowhere:1/v1/embeddings" in err and "config.json embed_url" in err


def test_serve_allow_degraded_starts_with_warning(monkeypatch, tmp_path, capsys):
    _main_with_probe(monkeypatch, tmp_path, "ConnectError: refused", ["--allow-degraded"])
    assert "Starting anyway" in capsys.readouterr().err


def test_serve_starts_when_probe_ok(monkeypatch, tmp_path):
    _main_with_probe(monkeypatch, tmp_path, None)
    assert serve_projects.DEFAULT_PROJECT in serve_projects._projects


# The function of chonks.embed.client, not the stub that conftest sets on chonks.serve.main.
from chonks.embed.client import probe_embedder as _real_probe_embedder


def test_probe_embedder_reports_connection_failure():
    from chonks.embed.client import Embedder
    err = _real_probe_embedder(Embedder("http://127.0.0.1:1/v1/embeddings", "jina-code-x"), timeout=1.0)
    assert err is not None and ":" in err


def test_serve_allow_degraded_via_env(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHONKS_ALLOW_DEGRADED", "1")
    _main_with_probe(monkeypatch, tmp_path, "ConnectError: refused")
    assert "Starting anyway" in capsys.readouterr().err


def test_serve_env_zero_does_not_allow_degraded(monkeypatch, tmp_path):
    monkeypatch.setenv("CHONKS_ALLOW_DEGRADED", "0")
    with pytest.raises(SystemExit):
        _main_with_probe(monkeypatch, tmp_path, "ConnectError: refused")


def test_unparseable_explicit_config_is_not_also_reported_missing(monkeypatch, tmp_path, capsys):
    # main() reconfigures logging with force=True, which drops caplog's
    # handler; the log stream is sys.stderr, which capsys owns.
    monkeypatch.setattr(serve_main.uvicorn, "run", lambda *a, **kw: None)
    serve_projects._projects.clear()
    config_path = tmp_path / "config.json"
    config_path.write_text("{ broken")
    serve_main.main(["--config", str(config_path), "--db", str(tmp_path / "x.db")])
    err = capsys.readouterr().err
    assert "Failed to parse" in err
    assert "Config file not found" not in err


def test_rerank_url_is_off_by_default(monkeypatch, tmp_path):
    _run_main_with_config(monkeypatch, tmp_path, {"projects": {"proj": {"db": str(tmp_path / "p.db")}}})
    assert serve_projects._projects[serve_projects.DEFAULT_PROJECT]["reranker"] is None
    assert serve_projects._projects["proj"]["reranker"] is None


def test_top_level_rerank_url_reaches_every_project(monkeypatch, tmp_path):
    url = "http://localhost:11440/v1/rerank"
    _run_main_with_config(monkeypatch, tmp_path, {
        "rerank_url": url, "projects": {"proj": {"db": str(tmp_path / "p.db")}}})
    assert serve_projects._projects[serve_projects.DEFAULT_PROJECT]["reranker"].url == url
    assert serve_projects._projects["proj"]["reranker"].url == url


def test_project_rerank_url_overrides_or_turns_off_the_top_level_one(monkeypatch, tmp_path):
    _run_main_with_config(monkeypatch, tmp_path, {
        "rerank_url": "http://localhost:11440/v1/rerank",
        "projects": {
            "own": {"db": str(tmp_path / "a.db"), "rerank_url": "http://gpu:11440/v1/rerank"},
            "off": {"db": str(tmp_path / "b.db"), "rerank_url": None},
        },
    })
    assert serve_projects._projects["own"]["reranker"].url == "http://gpu:11440/v1/rerank"
    assert serve_projects._projects["off"]["reranker"] is None
