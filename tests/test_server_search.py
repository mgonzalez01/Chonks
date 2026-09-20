"""Tests for the /search endpoint's chunk_kind filter and docs_in_results
count."""
import json

import httpx
import pytest
from pydantic import ValidationError

import chonks.server as server
from chonks.retrieval.searcher import Searcher
from chonks.store import Store


def _fake_embedding(dim: int = 4) -> list[float]:
    return [0.5] * dim


class _NoEmbedder:
    def embed_queries(self, queries, client, timeout=30.0):
        raise AssertionError("embedder should not be called for fts/regex modes")


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


def test_search_request_rejects_invalid_chunk_kind():
    with pytest.raises(ValidationError):
        server.SearchRequest(query="q", chunk_kind="bogus")


def test_search_request_accepts_code_docs_any_and_none():
    for v in ("code", "docs", "any", None):
        server.SearchRequest(query="q", chunk_kind=v)  # must not raise


def test_search_endpoint_rejects_invalid_chunk_kind_over_http(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **kw: None)
    server._projects.clear()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}))
    server.main(["--config", str(config_path), "--db", str(tmp_path / "x.db")])

    with TestClient(server.app) as client:
        resp = client.post("/search", json={"query": "q", "chunk_kind": "bogus"})
    assert resp.status_code == 422


def test_search_endpoint_docs_in_results_counts_returned_docs_chunks(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    store.insert_chunks(
        [
            {"id": "code1", "path": "a.py", "language": "python",
             "content": "def needle(): pass", "name": "needle",
             "start_line": 1, "end_line": 1},
            {"id": "docs1", "path": "a.md", "language": "md",
             "content": "needle docs one", "start_line": 1, "end_line": 1},
            {"id": "docs2", "path": "b.md", "language": "md",
             "content": "needle docs two", "start_line": 1, "end_line": 1},
        ],
        [_fake_embedding(), _fake_embedding(), _fake_embedding()],
    )

    req = server.SearchRequest(query="needle", mode="fts", top_k=10)
    body = json.loads(server.search(req).body)

    assert body["count"] == 3
    assert body["docs_in_results"] == 2


def test_search_endpoint_chunk_kind_code_excludes_docs(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    store.insert_chunks(
        [
            {"id": "code1", "path": "a.py", "language": "python",
             "content": "def needle(): pass", "name": "needle",
             "start_line": 1, "end_line": 1},
            {"id": "docs1", "path": "a.md", "language": "md",
             "content": "needle docs", "start_line": 1, "end_line": 1},
        ],
        [_fake_embedding(), _fake_embedding()],
    )

    req = server.SearchRequest(query="needle", mode="fts", top_k=10, chunk_kind="code")
    body = json.loads(server.search(req).body)

    assert body["count"] == 1
    assert body["chunks"][0]["id"] == "code1"
    assert body["docs_in_results"] == 0


def test_search_endpoint_chunk_kind_docs_excludes_code(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    store.insert_chunks(
        [
            {"id": "code1", "path": "a.py", "language": "python",
             "content": "def needle(): pass", "name": "needle",
             "start_line": 1, "end_line": 1},
            {"id": "docs1", "path": "a.md", "language": "md",
             "content": "needle docs", "start_line": 1, "end_line": 1},
        ],
        [_fake_embedding(), _fake_embedding()],
    )

    req = server.SearchRequest(query="needle", mode="regex", top_k=10, chunk_kind="docs")
    body = json.loads(server.search(req).body)

    assert body["count"] == 1
    assert body["chunks"][0]["id"] == "docs1"
    assert body["docs_in_results"] == 1


# --- search.file_cap --------------------------------------------------------

def _wall_chunks_and_vecs():
    chunks = [
        {"id": "hot0", "path": "hot.py", "content": "def hot0(): pass", "name": "hot0", "start_line": 1, "end_line": 1},
        {"id": "hot1", "path": "hot.py", "content": "def hot1(): pass", "name": "hot1", "start_line": 2, "end_line": 2},
        {"id": "hot2", "path": "hot.py", "content": "def hot2(): pass", "name": "hot2", "start_line": 3, "end_line": 3},
        {"id": "b0",   "path": "b.py",   "content": "def b0(): pass",   "name": "b0",   "start_line": 1, "end_line": 1},
    ]
    vecs = [
        [1.00, 0.00, 0.0, 0.0],
        [0.95, 0.05, 0.0, 0.0],
        [0.90, 0.10, 0.0, 0.0],
        [0.80, 0.20, 0.0, 0.0],
    ]
    return chunks, vecs


class _FixedEmbedder:
    def __init__(self, vec):
        self._vec = vec

    def embed_queries(self, queries, client, timeout=30.0):
        return [self._vec for _ in queries]


def test_search_endpoint_rejects_file_cap_on_non_semantic_mode(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    req = server.SearchRequest(query="needle", mode="fts", file_cap=2)
    with pytest.raises(server.HTTPException) as exc_info:
        server.search(req)
    assert exc_info.value.status_code == 422


def test_search_endpoint_file_cap_request_overrides_config(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    project["search_cfg"] = {"file_cap": 0}
    project["searcher"] = Searcher(project["store"], _FixedEmbedder([1.0, 0.0, 0.0, 0.0]))
    chunks, vecs = _wall_chunks_and_vecs()
    project["store"].insert_chunks(chunks, vecs)

    req = server.SearchRequest(query="needle", mode="semantic", top_k=3, file_cap=1)
    body = json.loads(server.search(req).body)

    ids = [c["id"] for c in body["chunks"]]
    assert sum(1 for i in ids if i.startswith("hot")) == 1
    assert "b0" in ids


def test_search_endpoint_file_cap_falls_back_to_config(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    project["search_cfg"] = {"file_cap": 2}
    project["searcher"] = Searcher(project["store"], _FixedEmbedder([1.0, 0.0, 0.0, 0.0]))
    chunks, vecs = _wall_chunks_and_vecs()
    project["store"].insert_chunks(chunks, vecs)

    req = server.SearchRequest(query="needle", mode="semantic", top_k=3)
    body = json.loads(server.search(req).body)

    ids = [c["id"] for c in body["chunks"]]
    assert sum(1 for i in ids if i.startswith("hot")) == 2
    assert "b0" in ids


def test_search_endpoint_file_cap_default_off_matches_pre_file_cap_behaviour(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    project["searcher"] = Searcher(project["store"], _FixedEmbedder([1.0, 0.0, 0.0, 0.0]))
    chunks, vecs = _wall_chunks_and_vecs()
    project["store"].insert_chunks(chunks, vecs)

    req = server.SearchRequest(query="needle", mode="semantic", top_k=3)
    body = json.loads(server.search(req).body)

    ids = [c["id"] for c in body["chunks"]]
    assert ids == ["hot0", "hot1", "hot2"]


# --- near_dup response shape ------------------------------------------------

def test_search_endpoint_near_dup_null_below_min_results(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    store.insert_chunks(
        [{"id": "code1", "path": "a.py", "language": "python",
          "content": "def needle(): pass", "name": "needle",
          "start_line": 1, "end_line": 1}],
        [_fake_embedding()],
    )

    req = server.SearchRequest(query="needle", mode="fts", top_k=10)
    body = json.loads(server.search(req).body)

    assert "near_dup" in body
    assert body["near_dup"] is None


def test_search_endpoint_near_dup_fires_on_identical_vectors(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    vec = _fake_embedding()
    store.insert_chunks(
        [
            {"id": f"needle{i}", "path": f"a{i}.py", "language": "python",
             "content": "def needle(): pass", "name": "needle",
             "start_line": 1, "end_line": 1}
            for i in range(5)
        ],
        [vec] * 5,
    )

    req = server.SearchRequest(query="needle", mode="fts", top_k=10)
    body = json.loads(server.search(req).body)

    assert body["count"] == 5
    assert body["near_dup"] is not None
    assert body["near_dup"]["wall_share"] == 1.0
    assert body["near_dup"]["wall_size"] == 5
    from chonks.retrieval.results import NEAR_DUP_TAU
    assert body["near_dup"]["tau"] == NEAR_DUP_TAU


# --- search.files (tier 1) ---------------------------------------------------

def test_search_endpoint_files_key_reflects_aggregated_ranking(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    searcher = project["searcher"]

    canned = [
        {"id": "single1", "path": "single.py", "_score": 0.65, "start_line": 1, "end_line": 1, "content": "pass"},
        {"id": "multi1",  "path": "multi.py",  "_score": 0.60, "start_line": 1, "end_line": 1, "content": "pass"},
        {"id": "multi2",  "path": "multi.py",  "_score": 0.58, "start_line": 2, "end_line": 2, "content": "pass"},
        {"id": "multi3",  "path": "multi.py",  "_score": 0.57, "start_line": 3, "end_line": 3, "content": "pass"},
    ]
    monkeypatch.setattr(searcher, "hybrid", lambda *a, **kw: canned)

    req = server.SearchRequest(query="needle", mode="hybrid", top_k=10)
    body = json.loads(server.search(req).body)

    assert body["chunks"] == canned
    assert body["count"] == 4

    files = body["files"]
    assert [f["path"] for f in files] == ["single.py", "multi.py"]
    assert files[0]["score"] == pytest.approx(0.65)
    assert files[0]["n_chunks"] == 1
    assert files[0]["best_rank"] == 1
    assert files[1]["score"] == pytest.approx(0.60)
    assert files[1]["n_chunks"] == 3
    assert files[1]["best_rank"] == 2


def test_search_endpoint_near_dup_absent_key_never_happens(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    store = project["store"]
    store.insert_chunks(
        [
            {"id": f"c{i}", "path": f"a{i}.py", "language": "python",
             "content": f"def foo{i}(): pass", "name": f"foo{i}",
             "start_line": 1, "end_line": 1}
            for i in range(6)
        ],
        [_fake_embedding() for _ in range(6)],
    )
    for mode in ("fts", "regex"):
        req = server.SearchRequest(query="foo" if mode == "fts" else "foo\\d", mode=mode, top_k=10)
        body = json.loads(server.search(req).body)
        assert "near_dup" in body, f"mode={mode}"


# --- embedder-down degradation signalling ------------------------------------

class _UnreachableEmbedder:
    def embed_queries(self, queries, client, timeout=30.0):
        raise httpx.ConnectError("Connection refused", request=None)


def test_search_endpoint_semantic_503s_when_embedder_unreachable(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    project["searcher"] = Searcher(project["store"], _UnreachableEmbedder())

    req = server.SearchRequest(query="needle", mode="semantic", top_k=5)
    with pytest.raises(server.HTTPException) as exc_info:
        server.search(req)
    assert exc_info.value.status_code == 503
    assert "fts" in exc_info.value.detail.lower()


def test_search_endpoint_hybrid_503s_when_embedder_unreachable(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    project["searcher"] = Searcher(project["store"], _UnreachableEmbedder())

    req = server.SearchRequest(query="needle", mode="hybrid", top_k=5)
    with pytest.raises(server.HTTPException) as exc_info:
        server.search(req)
    assert exc_info.value.status_code == 503
    assert "fts" in exc_info.value.detail.lower()


def test_search_endpoint_fts_unaffected_by_unreachable_embedder(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    project["searcher"] = Searcher(project["store"], _UnreachableEmbedder())
    project["store"].insert_chunks(
        [{"id": "c1", "path": "a.py", "language": "python",
          "content": "def needle(): pass", "name": "needle",
          "start_line": 1, "end_line": 1}],
        [_fake_embedding()],
    )

    req = server.SearchRequest(query="needle", mode="fts", top_k=5)
    body = json.loads(server.search(req).body)
    assert body["count"] == 1
    assert body["chunks"][0]["id"] == "c1"


# --- query_truncated ----------------------------------------------------------

def test_search_endpoint_query_truncated_false_for_short_query(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    project["searcher"] = Searcher(project["store"], _FixedEmbedder([0.5, 0.5, 0.5, 0.5]))

    req = server.SearchRequest(query="needle", mode="semantic", top_k=5)
    body = json.loads(server.search(req).body)
    assert body["query_truncated"] is False


def test_search_endpoint_query_truncated_true_for_over_budget_semantic_query(monkeypatch, tmp_path):
    from chonks.embedder import Embedder

    project = _setup_default_project(monkeypatch, tmp_path)
    embedder = Embedder("http://x/v1/embeddings", "qwen3-embedding", query_token_budget=3)
    embedder.embed_queries = lambda queries, client, timeout=30.0: [[0.5, 0.5, 0.5, 0.5] for _ in queries]
    project["searcher"] = Searcher(project["store"], embedder)

    req = server.SearchRequest(query="a" * 100, mode="semantic", top_k=5)
    body = json.loads(server.search(req).body)
    assert body["query_truncated"] is True


def test_search_endpoint_query_truncated_true_for_over_budget_hybrid_query(monkeypatch, tmp_path):
    from chonks.embedder import Embedder

    project = _setup_default_project(monkeypatch, tmp_path)
    embedder = Embedder("http://x/v1/embeddings", "qwen3-embedding", query_token_budget=3)
    embedder.embed_queries = lambda queries, client, timeout=30.0: [[0.5, 0.5, 0.5, 0.5] for _ in queries]
    project["searcher"] = Searcher(project["store"], embedder)

    req = server.SearchRequest(query="a" * 100, mode="hybrid", top_k=5)
    body = json.loads(server.search(req).body)
    assert body["query_truncated"] is True


def test_search_endpoint_query_truncated_false_for_fts_and_regex_regardless_of_length(monkeypatch, tmp_path):
    project = _setup_default_project(monkeypatch, tmp_path)
    project["searcher"] = Searcher(project["store"], _NoEmbedder())

    for mode, query in (("fts", "a" * 100), ("regex", "a" * 100)):
        req = server.SearchRequest(query=query, mode=mode, top_k=5)
        body = json.loads(server.search(req).body)
        assert body["query_truncated"] is False, f"mode={mode}"
