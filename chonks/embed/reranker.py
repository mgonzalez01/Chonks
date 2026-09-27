"""Client for a llama.cpp /v1/rerank endpoint, and the text each document is scored as."""

import httpx

# At the client's two characters per token, a 2000-character query and a
# 6000-character document fit one 4096-token slot together with the prompt
# template. Denser text can still overflow the slot.
RERANK_QUERY_MAX_CHARS = 2000
RERANK_DOC_MAX_CHARS = 6000


def rerank_document(path: str, name: str | None, content: str) -> str:
    """The `path :: name` breadcrumb and the chunk's content, as the reranker scores it."""
    return f"{path} :: {name or ''}\n{content}"


class Reranker:
    """Thin client for a llama.cpp `llama-server --reranking` endpoint. The
    HTTP client is passed per call, as with Embedder."""

    def __init__(self, url: str):
        self.url = url

    def _post(self, query: str, documents: list[str], client: httpx.Client, timeout: float) -> list[float]:
        resp = client.post(
            self.url,
            json={"query": query, "documents": documents, "top_n": len(documents)},
            timeout=timeout,
        )
        resp.raise_for_status()
        scores: list[float | None] = [None] * len(documents)
        try:
            for item in resp.json()["results"]:
                scores[item["index"]] = float(item["relevance_score"])
        except (KeyError, TypeError, IndexError, ValueError) as e:
            raise ValueError(f"malformed rerank response: {e!r}") from e
        if any(s is None for s in scores):
            raise ValueError("rerank response left documents unscored")
        return scores

    def rerank(
        self,
        query: str,
        documents: list[str],
        client: httpx.Client,
        *,
        timeout: float = 30.0,
    ) -> list[float]:
        """Relevance score per document, in input order. Cuts the query and each
        document to their caps; on an HTTP error status (one pair over the slot
        fails the whole request) halves both caps and retries once."""
        if not documents:
            return []
        q_cap, d_cap = RERANK_QUERY_MAX_CHARS, RERANK_DOC_MAX_CHARS
        try:
            return self._post(query[:q_cap], [d[:d_cap] for d in documents], client, timeout)
        except httpx.HTTPStatusError:
            q_cap, d_cap = q_cap // 2, d_cap // 2
            return self._post(query[:q_cap], [d[:d_cap] for d in documents], client, timeout)


PROBE_QUERY = "read the contents of a file"
PROBE_RELEVANT = "def read_file(path):\n    with open(path) as f:\n        return f.read()"
PROBE_UNRELATED = "The recipe calls for two cups of flour and a pinch of salt."


def probe_reranker(reranker: Reranker, timeout: float = 10.0) -> str | None:
    """None when the reranker scores a relevant document clearly above an
    unrelated one, else a one-line reason."""
    try:
        with httpx.Client() as client:
            relevant, unrelated = reranker.rerank(
                PROBE_QUERY, [PROBE_RELEVANT, PROBE_UNRELATED], client, timeout=timeout)
    except Exception as e:  # noqa: BLE001, any failure is the same answer: not usable
        first_line = str(e).partition("\n")[0]
        return f"does not answer ({type(e).__name__}: {first_line})"
    scale = max(abs(relevant), abs(unrelated))
    if scale < 1e-6:
        return (f"scores every document near zero (relevant {relevant:.3g}, unrelated {unrelated:.3g}), "
                "so reranking reorders at random; the model file may lack its classification head")
    if relevant - unrelated < 0.1 * scale:
        return (f"scores an unrelated document ({unrelated:.3g}) about as high as a relevant one "
                f"({relevant:.3g}), so reranking reorders at random")
    return None
