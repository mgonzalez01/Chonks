import httpx

from chonks.index.embed_retry import _embed_isolating, _embed_one_isolating
from chonks.index.embed_retry import EMBED_MIN_CHARS

# Sized so a halve or two still lands above the floor; max_len=400 (> floor) means
# an oversized chunk can be truncated to fit, max_len below the floor means it
# never can (error instead).
_LONG = 1000
assert _LONG > 2 * EMBED_MIN_CHARS


def _http_400():
    req = httpx.Request("POST", "http://x/v1/embeddings")
    return httpx.HTTPStatusError("too long", request=req, response=httpx.Response(400, request=req))


def _http_503():
    req = httpx.Request("POST", "http://x/v1/embeddings")
    return httpx.HTTPStatusError("server down", request=req, response=httpx.Response(503, request=req))


def make_embed_docs(max_len, *, calls=None, transient_marker=None):
    """Fake embed_docs: 503 if any input contains `transient_marker` (non-truncatable),
    400 if any input exceeds `max_len` chars (truncatable), else a 1-d marker
    embedding [len(text)] per input."""
    def embed_docs(texts):
        if calls is not None:
            calls.append(list(texts))
        if transient_marker is not None and any(transient_marker in t for t in texts):
            raise _http_503()
        if any(len(t) > max_len for t in texts):
            raise _http_400()
        return [[float(len(t))] for t in texts]
    return embed_docs


def test_survivors_keep_full_length_only_culprit_truncated():
    embed_docs = make_embed_docs(max_len=400)
    chunks = [{"id": i} for i in range(4)]
    texts = ["ok", "fine", "X" * _LONG, "good"]
    ok_c, ok_e, trunc, errs = _embed_isolating(chunks, texts, embed_docs)

    assert sorted(c["id"] for c in ok_c) == [0, 1, 2, 3]
    assert trunc == 1
    assert errs == []
    by_id = {c["id"]: e[0] for c, e in zip(ok_c, ok_e)}
    assert by_id[0] == 2.0 and by_id[1] == 4.0 and by_id[3] == 4.0
    assert EMBED_MIN_CHARS <= by_id[2] <= 400 and by_id[2] < _LONG


def test_order_is_preserved():
    embed_docs = make_embed_docs(max_len=400)
    chunks = [{"id": i} for i in range(6)]
    texts = ["a", "b", "Z" * _LONG, "d", "e", "f"]
    ok_c, _e, _t, errs = _embed_isolating(chunks, texts, embed_docs)
    assert [c["id"] for c in ok_c] == [0, 1, 2, 3, 4, 5]
    assert errs == []


def test_unfixable_chunk_errors_without_taking_others_down():
    # max_len below the floor: no truncation can satisfy it, so it must error
    # while the innocent sibling still embeds at full length.
    embed_docs = make_embed_docs(max_len=10)
    chunks = [{"id": 0, "path": "a", "start_line": 1}, {"id": 1, "path": "b", "start_line": 2}]
    texts = ["ok", "B" * _LONG]
    ok_c, ok_e, trunc, errs = _embed_isolating(chunks, texts, embed_docs)
    assert [c["id"] for c in ok_c] == [0]
    assert ok_e[0][0] == 2.0
    assert trunc == 0
    assert len(errs) == 1 and errs[0][0]["id"] == 1


def test_transient_error_is_not_truncated():
    calls = []
    embed_docs = make_embed_docs(max_len=10_000, calls=calls, transient_marker="ZZZ")
    ok_c, ok_e, trunc, errs = _embed_one_isolating(
        {"id": 9, "path": "a", "start_line": 1}, "ZZZ" + "q" * _LONG, embed_docs
    )
    assert ok_c == [] and len(errs) == 1
    assert trunc == 0
    assert len(calls) == 1


def test_short_innocent_chunk_embedded_in_one_full_length_call():
    calls = []
    embed_docs = make_embed_docs(max_len=10_000, calls=calls)
    ok_c, ok_e, trunc, errs = _embed_one_isolating({"id": 5}, "hello", embed_docs)
    assert [c["id"] for c in ok_c] == [5]
    assert ok_e[0][0] == 5.0
    assert trunc == 0
    assert calls == [["hello"]]
