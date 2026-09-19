"""Batch size, timeout and retry policy for the indexer's embedding calls."""

import httpx


# ---------------------------------------------------------------------------
# Embedding constants
# ---------------------------------------------------------------------------

# Defaults for indexing concurrency. Both are overridable via config keys
# (embed_batch / embed_inflight) or chunker.py flags (--embed-batch / --embed-inflight),
# threaded into index_paths, so there's no module-level mutable state.
EMBED_BATCH    = 128       # chunks per embedding API call (request size)
EMBED_INFLIGHT = 2         # concurrent in-flight batches; too low starves a fast
                           # embedder's GPU, see DEPLOY.md "Throughput tuning".

# Truncation floor (compressed chars) for the per-chunk retry; below this a
# rejected chunk is dropped as an error instead of embedded from a fragment
# too small to carry meaning.
EMBED_MIN_CHARS = 200

# Abort an index run if the embedder hasn't responded in this many seconds,
# measured per round trip so bisection recovery on a slow chunk isn't mistaken
# for a hang. config key `embed_watchdog_secs`.
EMBED_WATCHDOG_SECS = 300

# Per-request timeout scales with batch size: a fixed 120s can fire on a slow-but-alive
# large batch, which gets misclassified non-retryable (_should_truncate_and_retry) and
# drops the whole batch. See compute_embed_timeout.
EMBED_TIMEOUT_PER_ITEM_S = 1.0     # seconds budgeted per chunk in a batch
EMBED_TIMEOUT_FLOOR_S    = 120.0
EMBED_TIMEOUT_CEILING_S  = 900.0   # cap so a huge batch can't hang one request indefinitely


def compute_embed_timeout(batch_size: int) -> float:
    """Per-request timeout for a batch: linear in size, floored at
    EMBED_TIMEOUT_FLOOR_S, capped at EMBED_TIMEOUT_CEILING_S. Only the
    indexer's embed call site uses this; other callers keep the 120s default."""
    return min(EMBED_TIMEOUT_CEILING_S,
               max(EMBED_TIMEOUT_FLOOR_S, batch_size * EMBED_TIMEOUT_PER_ITEM_S))


def _should_truncate_and_retry(exc: BaseException) -> bool:
    """True for a 4xx (oversize-input, likely fixable by truncating). False for
    5xx/connectivity errors, truncating can't fix an unreachable or broken
    server, so the retry loop should bail instead of burning its budget.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return 400 <= exc.response.status_code < 500
    return False
