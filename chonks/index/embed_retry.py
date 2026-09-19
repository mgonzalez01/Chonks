"""Batch size, timeout and retry policy for the indexer's embedding calls."""

import logging

import httpx

logger = logging.getLogger("chonks.chunker")


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


def _chunk_path(chunk) -> str:
    """Path for logging. Works for both chunk dicts and sqlite3.Row."""
    try:
        return chunk["path"]
    except (KeyError, IndexError, TypeError):
        return "?"


def _embed_one_isolating(chunk, text, embed_docs):
    """Halves toward EMBED_MIN_CHARS only on a 4xx (oversize); a transient/5xx
    error is not truncated, since sending less won't help. Returns the
    embedded chunk or the failure with its exception."""
    full = len(text)
    limit = full
    last_exc: Exception | None = None
    while True:
        try:
            emb = embed_docs([text[:limit]])
            if limit < full:
                logger.warning("EMBED oversize chunk %s truncated %d->%d chars",
                               _chunk_path(chunk), full, limit)
            return [chunk], [emb[0]], (1 if limit < full else 0), []
        except Exception as exc:
            last_exc = exc
            if not _should_truncate_and_retry(exc):
                break
            nxt = limit // 2
            if nxt < EMBED_MIN_CHARS:   # don't embed from a meaningless fragment
                break
            limit = nxt
    return [], [], 0, [(chunk, last_exc)]


def _batch_label(chunks) -> str:
    """Compact descriptor of a batch's source files for the watchdog's abort
    diagnostic: name the first and count the rest rather than dumping every
    path into a terminal message."""
    paths = sorted({c.get("path") or "?" for c in chunks})
    if not paths:
        return "empty batch"
    extra = f" +{len(paths) - 1} more" if len(paths) > 1 else ""
    return f"{paths[0]}{extra}"


def _embed_isolating(chunks, texts, embed_docs):
    """Recovers a failed batch by bisecting into halves and retrying each at
    full length, instead of truncating everything. Call only after `chunks`
    already failed as one request."""
    if not chunks:
        return [], [], 0, []
    if len(chunks) == 1:
        return _embed_one_isolating(chunks[0], texts[0], embed_docs)
    mid = len(chunks) // 2
    ok_c: list = []
    ok_e: list = []
    trunc = 0
    errs: list = []
    for cs, ts in ((chunks[:mid], texts[:mid]), (chunks[mid:], texts[mid:])):
        if not cs:
            continue
        try:
            embs = embed_docs(ts)            # retry this half at FULL length
            ok_c += list(cs)
            ok_e += embs
        except Exception:
            c2, e2, t2, er2 = _embed_isolating(cs, ts, embed_docs)
            ok_c += c2
            ok_e += e2
            trunc += t2
            errs += er2
    return ok_c, ok_e, trunc, errs
