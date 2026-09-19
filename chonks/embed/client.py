"""OpenAI-compatible /v1/embeddings client, query truncation and prefix presets."""

import base64
import logging

import httpx
import numpy as np

# CLI-only defaults; library callers (server.py, MCP) build Embedder from config
# and never touch these, keeping a remote-embedder swap a pure config change.
DEFAULT_EMBED_URL   = "http://localhost:11437/v1/embeddings"
DEFAULT_EMBED_MODEL = "qwen3-embedding"   # ignored by llama-server but required by spec
# What `chonks init` writes; the name selects the prefix preset below.
# Servers ignore the model name field, Chonks doesn't.
RECOMMENDED_EMBED_MODEL = "jina-code-embeddings-0.5b"


# ---------------------------------------------------------------------------
# Query-side token budget
# ---------------------------------------------------------------------------
# Queries have no CHUNK_MAX-style guard, so an over-budget one dies with a
# bare HTTP 400 from llama-server. No client-side tokenizer, so the token
# budget converts to chars via a conservative, code-density chars-per-token ratio.
QUERY_CHARS_PER_TOKEN = 2.0

# Conservative default leaving headroom for a query instruction prefix added
# on top (see JINA_CODE_QUERY_PREFIX etc below). Override via
# Embedder(query_token_budget=...) or config key `embed_query_token_budget`.
EMBED_QUERY_TOKEN_BUDGET = 3800


def truncate_query_text(
    text: str, token_budget: int = EMBED_QUERY_TOKEN_BUDGET,
) -> tuple[str, bool]:
    """Truncate `text` to a char budget derived from `token_budget`. Returns
    (text, truncated); untouched if already within budget. Slicing is Unicode-safe
    but can land mid-token, that's fine since the alternative is HTTP 400."""
    budget_chars = int(token_budget * QUERY_CHARS_PER_TOKEN)
    if len(text) <= budget_chars:
        return text, False
    return text[:budget_chars], True


# ---------------------------------------------------------------------------
# Asymmetric instruction prefixes
# ---------------------------------------------------------------------------
# See DESIGN.md "The embedding model and its prefixes" for why. Only
# embed_documents/embed_queries apply prefixes; embed() itself is always
# prefix-free.
#
# jina-code-embeddings, nl2code task (natural-language query -> code snippet):
JINA_CODE_QUERY_PREFIX = "Find the most relevant code snippet given the following query:\n"
JINA_CODE_DOC_PREFIX   = "Candidate code snippet:\n"

# nomic-ai/CodeRankEmbed: query-side prefix only, drops SILENTLY if the
# model-name gate doesn't match (no error, just degraded ranking).
CODERANK_QUERY_PREFIX = "Represent this query for searching relevant code: "

# Qwen3-Embedding: query-side instruction only; llama-server already appends
# <|endoftext|> itself (add_special), so Chonks adds nothing extra on the doc
# side. Measured to help research recall on the panel, see eval/results/F1.
QWEN3_QUERY_PREFIX = (
    "Instruct: Given a code search query, retrieve relevant code snippets that answer the query\n"
    "Query: "
)

# Case-insensitive substring match on embed_model. Explicit embed_query_prefix/
# embed_doc_prefix override a preset; no match means symmetric (no prefix).
_PREFIX_PRESETS: dict[str, tuple[str, str]] = {
    "jina-code": (JINA_CODE_QUERY_PREFIX, JINA_CODE_DOC_PREFIX),
    "coderank": (CODERANK_QUERY_PREFIX, ""),
    "qwen3": (QWEN3_QUERY_PREFIX, ""),
}


logger = logging.getLogger("chonks.embedder")


def matched_prefix_preset(model: str) -> str | None:
    """The `_PREFIX_PRESETS` key that `model` selects (case-insensitive substring),
    or None when the model runs symmetric/prefix-free."""
    lo = (model or "").lower()
    for key in _PREFIX_PRESETS:
        if key in lo:
            return key
    return None


def resolve_prefixes(
    model: str,
    query_prefix: str | None = None,
    doc_prefix: str | None = None,
) -> tuple[str, str]:
    """Resolve (query_prefix, doc_prefix) for `model`. Precedence: explicit arg
    (including "" to force none) > name-matched preset > "" empty. Centralizing
    this makes swapping embedders a pure config change.
    """
    q, d = "", ""
    key = matched_prefix_preset(model)
    if key is not None:
        q, d = _PREFIX_PRESETS[key]
    if query_prefix is not None:
        q = query_prefix
    if doc_prefix is not None:
        d = doc_prefix
    return q, d


def _decode_embedding(emb) -> np.ndarray:
    """Decode one embedding: base64-encoded float32 bytes (zero-copy) or a plain
    float list, so a server that ignores the base64 request param still works."""
    if isinstance(emb, str):
        return np.frombuffer(base64.b64decode(emb), dtype="<f4")
    return np.asarray(emb, dtype=np.float32)


class Embedder:
    """Thin client for an OpenAI-compatible /v1/embeddings endpoint. The HTTP
    client is passed per call, since the indexer and searcher
    each want their own connection pool to the same target.
    """

    def __init__(
        self,
        url: str,
        model: str,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        query_token_budget: int | None = None,
    ):
        self.url   = url
        self.model = model
        # Resolve asymmetric instruction prefixes once (see `resolve_prefixes`):
        # None => use the per-model preset; "" => force no prefix; str => verbatim.
        self.query_prefix, self.doc_prefix = resolve_prefixes(model, query_prefix, doc_prefix)
        # Warn once: an unset embed_model defaults to a name that silently selects
        # the qwen3 prefix preset, which could prepend the wrong instruction for
        # whatever model the URL actually serves.
        if model == DEFAULT_EMBED_MODEL and query_prefix is None and doc_prefix is None:
            logger.warning(
                "embed_model is not set; using the built-in default %r, which applies the "
                "Qwen3 query instruction. If %s serves a different model, set embed_model "
                "to its name.", model, url,
            )
        elif (matched_prefix_preset(model) is None
                and query_prefix is None and doc_prefix is None):
            logger.warning(
                "embed_model=%r matches no instruction-prefix preset (%s): queries "
                "and documents embed without prefixes. If %s serves jina-code-embeddings, "
                "set embed_model to a name containing 'jina-code' (and re-index if the "
                "DB was built without prefixes). Set embed_query_prefix/embed_doc_prefix "
                "explicitly to silence this.",
                model, ", ".join(sorted(_PREFIX_PRESETS)), url,
            )
        # See `truncate_query_text`. None => the conservative default.
        self.query_token_budget = (
            query_token_budget if query_token_budget is not None else EMBED_QUERY_TOKEN_BUDGET
        )

    def embed(
        self,
        texts: list[str],
        client: httpx.Client,
        *,
        timeout: float = 120.0,
    ) -> list[np.ndarray]:
        """Call /v1/embeddings for a batch; returns float32 arrays in input order.
        Requests base64 encoding since a GIL-bound per-float JSON parse would cap
        index throughput; falls back if the server ignores that param."""
        resp = client.post(
            self.url,
            json={"model": self.model, "input": texts, "encoding_format": "base64"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # data is [{index, embedding}, ...]; sort by index to guarantee order
        data.sort(key=lambda x: x["index"])
        return [_decode_embedding(item["embedding"]) for item in data]

    def embed_documents(
        self,
        texts: list[str],
        client: httpx.Client,
        *,
        timeout: float = 120.0,
    ) -> list[list[float]]:
        """Embed document/passage text (index side). Prepends the model's document
        instruction prefix when one is configured; otherwise identical to `embed`."""
        if self.doc_prefix:
            texts = [self.doc_prefix + t for t in texts]
        return self.embed(texts, client, timeout=timeout)

    def embed_queries(
        self,
        texts: list[str],
        client: httpx.Client,
        *,
        timeout: float = 120.0,
    ) -> list[list[float]]:
        """Embed query text: truncates to query_token_budget then adds the query
        prefix. On an HTTP 400 (budget still too generous for dense text) halves the
        budget and retries, up to twice, before giving up; other errors propagate."""
        budget = self.query_token_budget
        for attempt in range(3):
            cut = [truncate_query_text(t, budget)[0] for t in texts]
            if self.query_prefix:
                cut = [self.query_prefix + t for t in cut]
            try:
                return self.embed(cut, client, timeout=timeout)
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 400 or attempt == 2:
                    raise
                budget //= 2


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def compress_for_embed(text: str, *, name: str | None = None, path: str | None = None) -> str:
    """Strip whitespace and collapse blank lines before embedding; prepend a
    path::name breadcrumb if given. The DB always stores the original content,
    this only changes what gets embedded.
    """
    lines = [line.strip() for line in text.splitlines()]
    out: list[str] = []
    prev_blank = False
    for line in lines:
        blank = not line
        if blank and prev_blank:
            continue
        out.append(line)
        prev_blank = blank
    compressed = "\n".join(out)
    parts = [p for p in (path, name) if p]
    if parts:
        return " :: ".join(parts) + "\n" + compressed
    return compressed
