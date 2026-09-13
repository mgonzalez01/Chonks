"""Prefixes resolve from a per-model preset, overridable by explicit config,
and are applied on the document vs query side. The low-level `embed`
primitive stays prefix-free.
"""
from chonks.embedder import (
    CODERANK_QUERY_PREFIX,
    EMBED_QUERY_TOKEN_BUDGET,
    QUERY_CHARS_PER_TOKEN,
    Embedder,
    JINA_CODE_DOC_PREFIX,
    JINA_CODE_QUERY_PREFIX,
    resolve_prefixes,
    truncate_query_text,
)


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _FakeClient:
    """Captures the `input` list posted to /v1/embeddings; returns one vec per input."""

    def __init__(self):
        self.last_input = None

    def post(self, url, json, timeout):
        self.last_input = list(json["input"])
        data = [{"index": i, "embedding": [float(i)]} for i in range(len(json["input"]))]
        return _FakeResp({"data": data})


# ---- resolve_prefixes ------------------------------------------------------

def test_general_model_has_no_prefixes():
    assert resolve_prefixes("generic-embedding") == ("", "")


def test_jina_preset_matched_by_substring():
    assert resolve_prefixes("jina-code-embeddings-0.5b") == (
        JINA_CODE_QUERY_PREFIX,
        JINA_CODE_DOC_PREFIX,
    )


def test_jina_preset_case_insensitive():
    assert resolve_prefixes("JINA-CODE-Embeddings-1.5B") == (
        JINA_CODE_QUERY_PREFIX,
        JINA_CODE_DOC_PREFIX,
    )


def test_explicit_overrides_preset():
    assert resolve_prefixes("jina-code", query_prefix="Q: ", doc_prefix="D: ") == ("Q: ", "D: ")


def test_empty_string_forces_no_prefix_per_side():
    assert resolve_prefixes("jina-code", query_prefix="") == ("", JINA_CODE_DOC_PREFIX)


def test_none_means_use_preset():
    assert resolve_prefixes("jina-code", query_prefix=None, doc_prefix=None) == (
        JINA_CODE_QUERY_PREFIX,
        JINA_CODE_DOC_PREFIX,
    )


# ---- Embedder.embed_documents / embed_queries ------------------------------

def test_general_model_documents_unprefixed():
    e = Embedder("http://x/v1/embeddings", "generic-embedding")
    c = _FakeClient()
    e.embed_documents(["alpha", "beta"], c)
    assert c.last_input == ["alpha", "beta"]


def test_general_model_queries_unprefixed():
    e = Embedder("http://x/v1/embeddings", "generic-embedding")
    c = _FakeClient()
    e.embed_queries(["how does X work"], c)
    assert c.last_input == ["how does X work"]


def test_jina_documents_get_doc_prefix():
    e = Embedder("http://x/v1/embeddings", "jina-code-embeddings-0.5b")
    c = _FakeClient()
    e.embed_documents(["def f(): ...", "class C: ..."], c)
    assert c.last_input == [
        JINA_CODE_DOC_PREFIX + "def f(): ...",
        JINA_CODE_DOC_PREFIX + "class C: ...",
    ]


def test_jina_queries_get_query_prefix():
    e = Embedder("http://x/v1/embeddings", "jina-code-embeddings-0.5b")
    c = _FakeClient()
    e.embed_queries(["where is the renderer"], c)
    assert c.last_input == [JINA_CODE_QUERY_PREFIX + "where is the renderer"]


def test_force_empty_doc_prefix_overrides_jina_preset():
    e = Embedder("http://x/v1/embeddings", "jina-code-embeddings-0.5b", doc_prefix="")
    c = _FakeClient()
    e.embed_documents(["code"], c)
    assert c.last_input == ["code"]


def test_embed_primitive_is_never_prefixed():
    e = Embedder("http://x/v1/embeddings", "jina-code-embeddings-0.5b")
    c = _FakeClient()
    e.embed(["raw"], c)
    assert c.last_input == ["raw"]


# ---- indexing-concurrency defaults + plumbing contract ----

def test_indexing_concurrency_defaults_unchanged():
    import chonks.embedder as _e
    assert _e.EMBED_BATCH == 128
    assert _e.EMBED_INFLIGHT == 2


def test_index_paths_exposes_concurrency_params():
    import inspect
    import chonks.chunker as chunker
    import chonks.embedder as _e
    sig = inspect.signature(chunker.index_paths)
    assert sig.parameters["embed_batch"].default == _e.EMBED_BATCH
    assert sig.parameters["embed_inflight"].default == _e.EMBED_INFLIGHT


def test_reembed_all_exposes_batch_param():
    import inspect
    import chonks.chunker as chunker
    import chonks.embedder as _e
    sig = inspect.signature(chunker.reembed_all)
    assert sig.parameters["embed_batch"].default == _e.EMBED_BATCH


# ---- base64 embedding path (throughput) ----

def test_embed_requests_base64_encoding():
    captured = {}

    class _C:
        def post(self, url, json, timeout):
            captured.update(json)
            return _FakeResp({"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    Embedder("http://x", "generic-embedding").embed(["x"], _C())
    assert captured.get("encoding_format") == "base64"


def test_embed_decodes_base64_response():
    import base64
    import numpy as np

    vec = np.array([0.5, -0.25, 1.0, 0.0], dtype="<f4")
    blob = base64.b64encode(vec.tobytes()).decode()

    class _C:
        def post(self, url, json, timeout):
            return _FakeResp({"data": [{"index": 0, "embedding": blob}]})

    out = Embedder("http://x", "generic-embedding").embed(["hi"], _C())
    assert np.allclose(np.asarray(out[0], dtype=np.float32), [0.5, -0.25, 1.0, 0.0])


def test_embed_falls_back_to_float_array():
    import numpy as np

    class _C:
        def post(self, url, json, timeout):
            return _FakeResp({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})

    out = Embedder("http://x", "generic-embedding").embed(["hi"], _C())
    assert np.allclose(np.asarray(out[0], dtype=np.float32), [0.1, 0.2, 0.3])


def test_embed_preserves_input_order_with_base64():
    import base64
    import numpy as np

    def blob(x):
        return base64.b64encode(np.array([x], dtype="<f4").tobytes()).decode()

    class _C:
        def post(self, url, json, timeout):
            # return out of order to exercise the index sort
            return _FakeResp({"data": [
                {"index": 1, "embedding": blob(11.0)},
                {"index": 0, "embedding": blob(10.0)},
            ]})

    out = Embedder("http://x", "generic-embedding").embed(["a", "b"], _C())
    assert float(out[0][0]) == 10.0 and float(out[1][0]) == 11.0


# ---- CodeRankEmbed preset (asymmetric: query prefixed, documents bare) ------

def test_coderank_preset_query_only():
    assert resolve_prefixes("coderankembed") == (CODERANK_QUERY_PREFIX, "")


def test_coderank_documents_stay_bare():
    e = Embedder("http://x/v1/embeddings", "coderankembed")
    c = _FakeClient()
    e.embed_documents(["def f(): ..."], c)
    assert c.last_input == ["def f(): ..."]


def test_coderank_queries_get_query_prefix():
    e = Embedder("http://x/v1/embeddings", "coderankembed")
    c = _FakeClient()
    e.embed_queries(["where is the renderer"], c)
    assert c.last_input == [CODERANK_QUERY_PREFIX + "where is the renderer"]


# ---- Query-side token-budget truncation -------------------------------------

def test_truncate_query_text_short_query_untouched():
    text, truncated = truncate_query_text("where is the renderer")
    assert text == "where is the renderer"
    assert truncated is False


def test_truncate_query_text_exact_boundary_untouched():
    budget_chars = int(EMBED_QUERY_TOKEN_BUDGET * QUERY_CHARS_PER_TOKEN)
    text = "a" * budget_chars
    out, truncated = truncate_query_text(text)
    assert out == text
    assert truncated is False


def test_truncate_query_text_one_over_boundary_truncates():
    budget_chars = int(EMBED_QUERY_TOKEN_BUDGET * QUERY_CHARS_PER_TOKEN)
    text = "a" * (budget_chars + 1)
    out, truncated = truncate_query_text(text)
    assert len(out) == budget_chars
    assert truncated is True


def test_truncate_query_text_respects_custom_budget():
    text = "abcdefghij"
    out, truncated = truncate_query_text(text, token_budget=3)  # 3*2 = 6 chars
    assert out == "abcdef"
    assert truncated is True


def test_truncate_query_text_unicode_boundary_no_surrogate_split():
    # Python str indexing is per-codepoint, so slicing mid-cluster on a
    # multi-codepoint emoji still yields a valid string, not a lone
    # surrogate or decoding error.
    family = "\U0001F468\u200D\U0001F469\u200D\U0001F467"
    text = "x" * 5 + family
    out, truncated = truncate_query_text(text, token_budget=3)  # budget 6 chars
    assert truncated is True
    assert len(out) == 6
    out.encode("utf-8")  # must not raise


def test_truncate_query_text_non_ascii_char_count_not_bytes():
    # é is one Python str codepoint, not a UTF-8 byte-count trap.
    text = "café_lookup"
    out, truncated = truncate_query_text(text, token_budget=2)  # budget 4 chars
    assert out == "café"
    assert truncated is True


def test_embed_queries_truncates_before_sending():
    e = Embedder("http://x/v1/embeddings", "generic-embedding", query_token_budget=3)  # 6 chars
    c = _FakeClient()
    e.embed_queries(["abcdefghij"], c)
    assert c.last_input == ["abcdef"]


def test_embed_queries_short_query_unaffected_by_budget():
    e = Embedder("http://x/v1/embeddings", "generic-embedding", query_token_budget=100)
    c = _FakeClient()
    e.embed_queries(["short query"], c)
    assert c.last_input == ["short query"]


def test_embed_queries_truncation_applies_before_prefix():
    # Prefix is added AFTER truncation, so the prefix itself never eats into
    # the query's own truncation budget and always survives intact.
    e = Embedder("http://x/v1/embeddings", "jina-code-embeddings-0.5b", query_token_budget=3)
    c = _FakeClient()
    e.embed_queries(["abcdefghij"], c)
    assert c.last_input == [JINA_CODE_QUERY_PREFIX + "abcdef"]


def test_default_query_token_budget_used_when_unset():
    e = Embedder("http://x/v1/embeddings", "generic-embedding")
    assert e.query_token_budget == EMBED_QUERY_TOKEN_BUDGET


def test_explicit_query_token_budget_overrides_default():
    e = Embedder("http://x/v1/embeddings", "generic-embedding", query_token_budget=100)
    assert e.query_token_budget == 100


def test_embed_queries_retries_400_with_halved_budget():
    import httpx
    from chonks.embedder import Embedder

    calls = []

    class FakeResp:
        def __init__(self, fail):
            self.status_code = 400 if fail else 200
            self._fail = fail

        def raise_for_status(self):
            if self._fail:
                raise httpx.HTTPStatusError(
                    "400", request=httpx.Request("POST", "http://x"),
                    response=httpx.Response(400, request=httpx.Request("POST", "http://x")))

        def json(self):
            return {"data": [{"embedding": [0.0] * 4, "index": 0}]}

    class FakeClient:
        def post(self, url, json=None, timeout=None):
            calls.append(len(json["input"][0]))
            # fail until the input is under half the first attempt's length
            return FakeResp(len(json["input"][0]) > calls[0] // 2)

    emb = Embedder(url="http://x", model="plain", query_prefix="", doc_prefix="")
    dense = "x" * (emb.query_token_budget * 4)
    out = emb.embed_queries([dense], FakeClient())
    assert len(out) == 1 and len(calls) >= 2, calls
    assert calls[-1] < calls[0]


# ---- warn when the model name selects no prefix preset --------------------

def test_no_preset_model_warns(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="chonks.embedder"):
        Embedder("http://h/v1/embeddings", "generic-embedding")
    assert any("matches no instruction-prefix preset" in r.message for r in caplog.records)


def test_preset_model_does_not_warn(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="chonks.embedder"):
        Embedder("http://h/v1/embeddings", "jina-code-embeddings-0.5b")
    assert not [r for r in caplog.records if "prefix preset" in r.message]


def test_explicit_prefixes_silence_no_preset_warning(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="chonks.embedder"):
        Embedder("http://h/v1/embeddings", "generic-embedding", query_prefix="", doc_prefix="")
    assert not [r for r in caplog.records if "prefix preset" in r.message]


# ---- qwen3 preset: query instruction, documents bare ----------------------

def test_qwen3_preset_query_instruction_doc_bare():
    from chonks.embedder import QWEN3_QUERY_PREFIX
    assert resolve_prefixes("qwen3-embedding-0.6b") == (QWEN3_QUERY_PREFIX, "")
    assert resolve_prefixes("Qwen/Qwen3-Embedding-4B-GGUF") == (QWEN3_QUERY_PREFIX, "")


def test_qwen3_queries_get_instruction_documents_do_not():
    from chonks.embedder import QWEN3_QUERY_PREFIX
    seen = []
    class _C:
        def post(self, url, json, timeout):
            seen.append(json["input"])
            class R:
                def raise_for_status(self): pass
                def json(self): return {"data": [{"index": i, "embedding": [0.0]} for i in range(len(json["input"]))]}
            return R()
    e = Embedder("http://x/v1/embeddings", "qwen3-embedding-0.6b")
    e.embed_queries(["find the parser"], _C())
    e.embed_documents(["def parse(): pass"], _C())
    assert seen[0] == [QWEN3_QUERY_PREFIX + "find the parser"]
    assert seen[1] == ["def parse(): pass"]


def test_default_model_name_warns_about_qwen3_instruction(caplog):
    import logging
    from chonks.embedder import DEFAULT_EMBED_MODEL
    with caplog.at_level(logging.WARNING, logger="chonks.embedder"):
        Embedder("http://h/v1/embeddings", DEFAULT_EMBED_MODEL)
    assert any("embed_model is not set" in r.message for r in caplog.records)
