import json

import pytest
from pydantic import ValidationError

import chonks.serve.app as serve_app
import chonks.serve.main as serve_main
import chonks.serve.models as serve_models
import chonks.serve.projects as serve_projects
from chonks.retrieval.searcher import Searcher
from chonks.storage.store import Store


class _NoEmbedder:
    def embed_queries(self, queries, client, timeout=30.0):
        raise AssertionError("find_by_message never needs the embedder")


def _setup_default_project(monkeypatch, tmp_path) -> dict:
    monkeypatch.setattr(serve_main.uvicorn, "run", lambda *a, **kw: None)
    serve_projects._projects.clear()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}))
    serve_main.main(["--config", str(config_path), "--db", str(tmp_path / "x.db")])

    project = serve_projects._projects[serve_projects.DEFAULT_PROJECT]
    project["store"] = Store(project["db_path"])
    project["searcher"] = Searcher(project["store"], _NoEmbedder())
    return project


def test_find_by_message_request_rejects_overlong_message():
    with pytest.raises(ValidationError):
        serve_models.FindByMessageRequest(message="x" * (serve_models._QUERY_MAX_LEN + 1))


def test_find_by_message_endpoint_resolves_format_hole_over_http(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    project = _setup_default_project(monkeypatch, tmp_path)
    project["store"].insert_chunks(
        [{
            "id": "c1",
            "path": "loader.py",
            "language": "python",
            "chunk_type": "function_definition",
            "name": "load_asset",
            "start_line": 1,
            "end_line": 2,
            "content": 'logger.error("failed to load %s: %d" % (name, code))',
            "literals": [("failed to load %s: %d", 2)],
        }],
        [[0.1, 0.2, 0.3, 0.4]],
        model="fake",
    )
    # insert_chunks alone doesn't set the literal-index-complete meta flag
    # (store.py's _LITERAL_INDEX_META_KEY); only a full index_paths run does.
    # Set it explicitly since this test drives the store directly.
    project["store"].set_meta("literal_index_version", "1")

    with TestClient(serve_app.app) as client:
        resp = client.post("/find_by_message", json={"message": "failed to load textures/hero.png: 404"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["results"][0]["match_kind"] == "skeleton"
    assert body["results"][0]["path"] == "loader.py"


def test_find_by_message_endpoint_overlong_message_rejected_422(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    _setup_default_project(monkeypatch, tmp_path)
    with TestClient(serve_app.app) as client:
        resp = client.post("/find_by_message", json={"message": "x" * 5000})
    assert resp.status_code == 422


def test_find_by_message_endpoint_empty_message_accepted_and_degrades_harmlessly(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    _setup_default_project(monkeypatch, tmp_path)
    with TestClient(serve_app.app) as client:
        resp = client.post("/find_by_message", json={"message": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"] == []
    assert body["count"] == 0


def test_find_by_message_endpoint_note_survives_the_jsonresponse_shape(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    project = _setup_default_project(monkeypatch, tmp_path)
    project["store"].insert_chunks(
        [{
            "id": "c1", "path": "a.py", "language": "python",
            "chunk_type": "function_definition", "name": "f",
            "start_line": 1, "end_line": 2,
            "content": 'log("totally unrelated literal text")',
            "literals": [("totally unrelated literal text", 1)],
        }],
        [[0.1, 0.2, 0.3, 0.4]],
        model="fake",
    )
    project["store"].set_meta("literal_index_version", "1")

    with TestClient(serve_app.app) as client:
        resp = client.post("/find_by_message", json={"message": "nothing here matches anything at all"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"] == []
    assert body.get("note")
