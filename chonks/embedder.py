"""Old import path for chonks.embed.client and chonks.index.embed_retry."""

from chonks.embed.client import (
    CODERANK_QUERY_PREFIX,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_URL,
    EMBED_QUERY_TOKEN_BUDGET,
    Embedder,
    JINA_CODE_DOC_PREFIX,
    JINA_CODE_QUERY_PREFIX,
    QUERY_CHARS_PER_TOKEN,
    QWEN3_QUERY_PREFIX,
    RECOMMENDED_EMBED_MODEL,
    compress_for_embed as _compress_for_embed,
    matched_prefix_preset,
    resolve_prefixes,
    truncate_query_text,
)
from chonks.index.embed_retry import (
    EMBED_BATCH,
    EMBED_INFLIGHT,
    EMBED_MIN_CHARS,
    EMBED_TIMEOUT_CEILING_S,
    EMBED_TIMEOUT_FLOOR_S,
    EMBED_TIMEOUT_PER_ITEM_S,
    EMBED_WATCHDOG_SECS,
    _should_truncate_and_retry,
    compute_embed_timeout,
)
