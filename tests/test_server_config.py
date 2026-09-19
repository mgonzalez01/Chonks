import json

import pytest
from pydantic import ValidationError

import chonks.server as server
import chonks.serve.app as serve_app


def _run_main_with_config(monkeypatch, tmp_path, config: dict) -> None:
    # uvicorn.run stubbed so main() populates _projects and returns instead
    # of blocking on a real server.
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **kw: None)
    server._projects.clear()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    server.main(["--config", str(config_path), "--db", str(tmp_path / "x.db")])


def test_default_project_picks_up_top_level_edge_type_weights(monkeypatch, tmp_path):
    weights = {"calls": 5.0, "mentions": 0.2}
    _run_main_with_config(monkeypatch, tmp_path, {"edge_type_weights": weights})
    assert server._projects[server.DEFAULT_PROJECT]["edge_type_weights"] == weights


def test_default_project_defaults_to_empty_when_unset(monkeypatch, tmp_path):
    _run_main_with_config(monkeypatch, tmp_path, {})
    assert server._projects[server.DEFAULT_PROJECT]["edge_type_weights"] == {}


def test_named_project_inherits_top_level_edge_type_weights(monkeypatch, tmp_path):
    weights = {"calls": 3.0}
    db_path = tmp_path / "proj.db"
    _run_main_with_config(monkeypatch, tmp_path, {
        "edge_type_weights": weights,
        "projects": {"proj": {"db": str(db_path)}},
    })
    assert server._projects["proj"]["edge_type_weights"] == weights


def test_named_project_can_override_top_level_edge_type_weights(monkeypatch, tmp_path):
    db_path = tmp_path / "proj.db"
    override = {"calls": 99.0}
    _run_main_with_config(monkeypatch, tmp_path, {
        "edge_type_weights": {"calls": 3.0},
        "projects": {"proj": {"db": str(db_path), "edge_type_weights": override}},
    })
    assert server._projects["proj"]["edge_type_weights"] == override


def test_research_endpoint_injects_edge_type_weights_into_cfg(monkeypatch, tmp_path):
    weights = {"calls": 7.0, "mentions": 0.3}
    _run_main_with_config(monkeypatch, tmp_path, {"edge_type_weights": weights})

    captured = {}
    def fake_deep_research(query, searcher, cfg=None, path_prefix=None):
        captured["cfg"] = cfg
        return {"chunks": [], "count": 0, "iterations": 0}
    monkeypatch.setattr(serve_app, "deep_research", fake_deep_research)

    # Avoid opening a real Store/Searcher for the default project.
    project = server._projects[server.DEFAULT_PROJECT]
    project["store"] = object()
    project["searcher"] = object()

    req = server.ResearchRequest(query="q")
    server.research(req)
    assert captured["cfg"]["edge_type_weights"] == weights


def test_research_endpoint_request_level_edge_type_weights_overrides_config(monkeypatch, tmp_path):
    # Precedence: request > project config > all-1.0 default.
    config_weights = {"calls": 7.0, "mentions": 0.3}
    _run_main_with_config(monkeypatch, tmp_path, {"edge_type_weights": config_weights})

    captured = {}
    def fake_deep_research(query, searcher, cfg=None, path_prefix=None):
        captured["cfg"] = cfg
        return {"chunks": [], "count": 0, "iterations": 0}
    monkeypatch.setattr(serve_app, "deep_research", fake_deep_research)

    project = server._projects[server.DEFAULT_PROJECT]
    project["store"] = object()
    project["searcher"] = object()

    override = {"calls": 8.0, "imports": 8.0, "inherits": 8.0, "xlang": 8.0, "mentions": 1.0}
    req = server.ResearchRequest(query="q", edge_type_weights=override)
    server.research(req)
    assert captured["cfg"]["edge_type_weights"] == override


def test_research_request_negative_edge_type_weight_rejected():
    with pytest.raises(ValidationError):
        server.ResearchRequest(query="q", edge_type_weights={"calls": -1.0})




# ---- serve refuses to start on a dead embedder -----------------------------

def _main_with_probe(monkeypatch, tmp_path, probe_result, extra_args=()):
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **kw: None)
    monkeypatch.setattr(server, "probe_embedder", lambda embedder, timeout=10.0: probe_result)
    server._projects.clear()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"embed_url": "http://nowhere:1/v1/embeddings"}))
    server.main(["--config", str(config_path), "--db", str(tmp_path / "x.db"), *extra_args])


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
    assert server.DEFAULT_PROJECT in server._projects


# Captured at import, before the autouse conftest stub replaces the attribute.
_real_probe_embedder = server.probe_embedder


def test_probe_embedder_reports_connection_failure():
    from chonks.embedder import Embedder
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
