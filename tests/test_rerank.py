"""Tests for the reranker client and the fusion of hybrid and reranker ranks."""
import json

import httpx
import pytest

from chonks.embed.reranker import (
    RERANK_DOC_MAX_CHARS,
    RERANK_QUERY_MAX_CHARS,
    Reranker,
    probe_reranker,
    rerank_document,
)
from chonks.retrieval.rerank import RERANK_CANDIDATES, fuse_ranks, reranked_hybrid
from chonks.retrieval.results import format_results


def _chunks(n: int) -> list[dict]:
    return [{"id": f"c{i}", "path": f"scene/node_{i}.cpp", "name": f"Node{i}", "content": f"void f{i}() {{}}",
             "start_line": 1, "end_line": 1, "_score": 1.0 / (60 + i + 1),
             "_rank_semantic": i + 1, "_rank_fts": None} for i in range(n)]


def _rerank_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _scores_by_index(scores):
    def handler(request):
        body = json.loads(request.content)
        results = [{"index": i, "relevance_score": scores[i]} for i in range(len(body["documents"]))]
        return httpx.Response(200, json={"results": sorted(results, key=lambda r: -r["relevance_score"])})
    return handler


def test_fuse_ranks_orders_by_hybrid_rank_plus_twice_the_rerank_rank():
    chunks = _chunks(3)
    fused = fuse_ranks(chunks, [0.1, 0.9, 0.5])
    assert [c["id"] for c in fused] == ["c1", "c0", "c2"]
    assert fused[0]["_score"] == pytest.approx(1 / 62 + 2 / 61)
    assert fused[1]["_score"] == pytest.approx(1 / 61 + 2 / 63)
    assert [(c["_rank_hybrid"], c["_rank_rerank"]) for c in fused] == [(2, 1), (1, 3), (3, 2)]
    assert fused[0]["_rerank_score"] == 0.9


def test_hybrid_first_stays_first_when_reranker_prefers_deep_candidates():
    chunks = _chunks(50)
    scores = [0.0] * 50
    scores[0], scores[39], scores[29] = 0.7, 0.9, 0.8
    fused = fuse_ranks(chunks, scores)
    assert fused[0]["id"] == "c0"
    assert fused[0]["_rank_rerank"] == 3


def test_chunks_past_the_scored_prefix_keep_hybrid_order_after_it():
    chunks = _chunks(5)
    fused = fuse_ranks(chunks, [0.1, 0.2, 0.3])
    assert [c["id"] for c in fused] == ["c2", "c1", "c0", "c3", "c4"]
    assert fused[3]["_rank_rerank"] is None and fused[3]["_rerank_score"] is None


def test_equal_rerank_scores_keep_hybrid_order():
    fused = fuse_ranks(_chunks(3), [0.5, 0.5, 0.5])
    assert [c["id"] for c in fused] == ["c0", "c1", "c2"]


def test_rerank_document_is_path_name_breadcrumb_then_content():
    assert rerank_document("core/os/os.cpp", "OS::get_name", "String OS::get_name()") == \
        "core/os/os.cpp :: OS::get_name\nString OS::get_name()"
    assert rerank_document("README.md", None, "text") == "README.md :: \ntext"


def test_reranker_returns_scores_in_document_order_and_cuts_query_and_documents():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"results": [
            {"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.2}]})

    scores = Reranker("http://reranker/v1/rerank").rerank(
        "q" * 5000, ["a" * 9000, "b"], _rerank_client(handler))
    assert scores == [0.2, 0.8]
    assert len(sent["query"]) == RERANK_QUERY_MAX_CHARS
    assert [len(d) for d in sent["documents"]] == [RERANK_DOC_MAX_CHARS, 1]
    assert sent["top_n"] == 2


def test_reranker_retries_once_with_halved_cuts_on_an_error_status():
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        if len(sent) == 1:
            return httpx.Response(500, json={"error": {"message": "input is too large to process"}})
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    scores = Reranker("http://reranker/v1/rerank").rerank("q" * 5000, ["a" * 9000], _rerank_client(handler))
    assert scores == [0.5]
    assert len(sent[1]["query"]) == RERANK_QUERY_MAX_CHARS // 2
    assert len(sent[1]["documents"][0]) == RERANK_DOC_MAX_CHARS // 2


def test_reranker_raises_after_the_retry_fails():
    client = _rerank_client(lambda request: httpx.Response(500, json={}))
    with pytest.raises(httpx.HTTPStatusError):
        Reranker("http://reranker/v1/rerank").rerank("q", ["a"], client)


def test_reranker_rejects_a_response_that_leaves_a_document_unscored():
    client = _rerank_client(lambda request: httpx.Response(200, json={"results": [
        {"index": 0, "relevance_score": 0.5}]}))
    with pytest.raises(ValueError):
        Reranker("http://reranker/v1/rerank").rerank("q", ["a", "b"], client)


def test_probe_reranker_reports_connection_failure():
    err = probe_reranker(Reranker("http://127.0.0.1:1/v1/rerank"), timeout=1.0)
    assert err.startswith("does not answer (Connect")


class _FixedScores:
    def __init__(self, *scores):
        self._scores = list(scores)

    def rerank(self, query, documents, client, *, timeout=30.0):
        return self._scores


@pytest.mark.parametrize("scores", [(0.9996, 0.0001), (4.5, -7.0)])
def test_probe_reranker_passes_when_the_relevant_document_scores_clearly_higher(scores):
    assert probe_reranker(_FixedScores(*scores)) is None


def test_probe_reranker_flags_near_zero_scores():
    assert "near zero" in probe_reranker(_FixedScores(1e-23, 2e-23))


@pytest.mark.parametrize("scores", [(0.51, 0.49), (0.2, 0.7)])
def test_probe_reranker_flags_an_unrelated_document_scored_as_high(scores):
    assert "about as high" in probe_reranker(_FixedScores(*scores))


class _Searcher:
    def __init__(self, chunks):
        self._chunks = chunks
        self.asked_top_k = None

    def hybrid(self, query, top_k=50, path_prefix=None, client=None, chunk_kind=None, file_cap=0):
        self.asked_top_k = top_k
        return [dict(c) for c in self._chunks[:top_k]]


class _ScoresReranker:
    def __init__(self, scores):
        self._scores = scores
        self.documents = None

    def rerank(self, query, documents, client, *, timeout=30.0):
        self.documents = documents
        return self._scores[:len(documents)]


class _DownReranker:
    def rerank(self, query, documents, client, *, timeout=30.0):
        raise httpx.ConnectError("Connection refused")


def test_reranked_hybrid_reranks_fifty_candidates_and_returns_top_k():
    searcher = _Searcher(_chunks(60))
    reranker = _ScoresReranker([i / 100 for i in range(60)])
    chunks, note = reranked_hybrid(searcher, reranker, "query", top_k=10)
    assert searcher.asked_top_k == RERANK_CANDIDATES
    assert len(reranker.documents) == RERANK_CANDIDATES
    assert reranker.documents[0] == "scene/node_0.cpp :: Node0\nvoid f0() {}"
    assert len(chunks) == 10 and note is None
    assert chunks[0]["id"] == "c49"


def test_reranked_hybrid_returns_hybrid_order_with_a_note_when_the_reranker_fails():
    searcher = _Searcher(_chunks(60))
    chunks, note = reranked_hybrid(searcher, _DownReranker(), "query", top_k=10)
    assert [c["id"] for c in chunks] == [f"c{i}" for i in range(10)]
    assert all("_rank_rerank" not in c for c in chunks)
    assert "ConnectError" in note and "hybrid order" in note


def test_format_results_shows_the_rerank_rank():
    fused = fuse_ranks(_chunks(2), [0.1, 0.9])
    assert "[sem#2,rerank#1]" in format_results(fused)
