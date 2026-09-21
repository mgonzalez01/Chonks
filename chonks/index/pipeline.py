"""The indexing pipeline: scan, parse and embed threads, run state, worker pool and abort errors."""

import json
import logging
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import httpx
from tqdm import tqdm

from chonks.core.paths import (
    _to_stored_path,
    _normalize_prefixes,
    _path_allowed,
    _dir_should_prune,
    _dotdir_prefix,
    _stored_dirname,
)
from chonks.core.refresh import register_refresh
from chonks.index.embed_retry import (
    EMBED_BATCH,
    EMBED_INFLIGHT,
    EMBED_WATCHDOG_SECS,
    _should_truncate_and_retry,
    compute_embed_timeout,
)
from chonks.embed.client import Embedder, compress_for_embed as _compress_for_embed
from chonks.index.admission import (
    DEFAULT_DATA_BLOB_SIZE_LIMIT,
    DEFAULT_FALLBACK_EXTENSIONS,
    _file_hash,
    _is_oversize_data_blob,
)
from chonks.index.embed_retry import _batch_label, _embed_isolating
from chonks.index.macro_memo import load_macro_memo, persist_macro_memo
from chonks.index.postindex import run_post_index_passes
from chonks.index.progress import tqdm_reporter
from chonks.index.rows import (
    _chunk_id,
    _chunk_metadata,
    _file_metadata,
)
from chonks.index.segment import (
    CHUNKER_VERSION,
    _lang_for_path,
    segment_file,
)
from chonks.index.text_segment import segment_text_file
from chonks.languages import EXT_TO_LANG as _EXT_TO_LANG
from chonks.languages import language_set as _language_set
from chonks.storage.store import Store


def _refresh_from_registry() -> None:
    global _EXT_TO_LANG
    import chonks.languages as _languages
    _EXT_TO_LANG = _languages.EXT_TO_LANG


register_refresh(_refresh_from_registry)

logger = logging.getLogger("chonks.chunker")


_SENTINEL = object()  # pipeline end-of-stream marker


@dataclass
class RunState:
    """State shared by the threads of one index_paths run."""

    state: dict
    lock: threading.Lock
    done_event: threading.Event
    file_symbols: dict[str, list[dict]]
    worker_exc: list[Exception | None]
    changed_chunk_ids: set[str]
    deleted_chunk_ids: set[str]
    deleted_chunk_names: set[str]
    parse_q: queue.Queue
    embed_q: queue.Queue
    macro_file_counts: dict[str, int]
    unhealable_hashes: set[str]
    unhealable_order: list[str]
    new_unhealable: bool


class _DaemonPool:
    """Daemon-thread ThreadPoolExecutor stand-in: stdlib non-daemon workers
    would block process shutdown if one hangs inside the embedder HTTP call.
    Safe because a batch only counts once commit_phase writes it under the lock."""

    def __init__(self, max_workers: int):
        self._q: queue.Queue = queue.Queue()
        self._threads = [
            threading.Thread(target=self._worker, daemon=True,
                             name=f"embed-pool-{i}")
            for i in range(max_workers)
        ]
        for t in self._threads:
            t.start()

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            fut, fn, args = item
            try:
                fut.set_result(fn(*args))
            except BaseException as e:
                fut.set_exception(e)

    def submit(self, fn, *args):
        fut: Future = Future()
        self._q.put((fut, fn, args))
        return fut

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        for _ in self._threads:
            self._q.put(None)  # wake idle workers; wedged ones are daemon


class NoProgressError(RuntimeError):
    """Raised by index_paths' progress monitor when no embedding batch has
    committed within `no_progress_timeout` seconds, since a wedged embedder
    thread would otherwise hang the run silently forever."""


class EmbedderDownError(RuntimeError):
    """After EMBEDDER_DOWN_THRESHOLD consecutive whole-batch failures (non-4xx).
    A dead embedder fails fast, which resets the no-progress watchdog's clock
    every time, so this circuit breaker catches what that watchdog can't."""


# Non-4xx whole-batch failures before EmbedderDownError aborts. Counts only
# the outer failure, not _embed_isolating's nested per-chunk ones (those are
# normal). 5 balances tolerating a network blip against a fast abort.
EMBEDDER_DOWN_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Incremental indexer
# ---------------------------------------------------------------------------

# ------------------------------------------------------- parser thread
def parser_worker(rs: RunState, vocab: set[str]) -> None:
    state = rs.state
    lock = rs.lock
    file_symbols = rs.file_symbols
    worker_exc = rs.worker_exc
    parse_q = rs.parse_q
    embed_q = rs.embed_q
    macro_file_counts = rs.macro_file_counts
    unhealable_hashes = rs.unhealable_hashes
    unhealable_order = rs.unhealable_order
    try:
        while True:
            item = parse_q.get()
            if item is _SENTINEL:
                embed_q.put(_SENTINEL)
                return
            fpath, content_hash, stored_path = item
            try:
                with lock:
                    state["current_file"] = fpath.name
                src  = fpath.read_bytes()
                lang = _lang_for_path(fpath)
                _oc: dict = {}
                if lang is not None:
                    # Skip the heal sweep entirely for content already known
                    # unhealable from a prior run.
                    heal = content_hash not in unhealable_hashes
                    segs = segment_file(src, lang, path=stored_path,
                                        counters=_oc, macros=vocab, self_heal=heal)
                else:
                    # No grammar for this extension, admitted only because it's
                    # in fallback_extensions (scan_producer already filtered out
                    # anything else). Line-based slicing, no AST boundaries.
                    lang = fpath.suffix.lower().lstrip(".")
                    segs = segment_text_file(src, path=stored_path, counters=_oc)
                # coverage: low means content was silently dropped (parse
                # error, or Python module-level code that's never a boundary).
                covered  = sum(len(s["content"].encode("utf-8")) for s in segs)
                coverage = covered / max(1, len(src))
                with lock:
                    if _oc.get("oversize_chunks"):
                        state["oversize_chunks"] += _oc.get("oversize_chunks", 0)
                        state["oversize_files"]  += _oc.get("oversize_files", 0)
                    if _oc.get("parse_error"):
                        state["parse_error_files"] += 1
                    if _oc.get("macro_healed"):
                        state["macro_healed_files"] += 1
                    if _oc.get("error_salvaged"):
                        state["error_salvaged_files"] += 1
                        state["error_salvaged_nodes"] += _oc.get("error_salvaged_nodes", 0)
                    if _oc.get("literals_capped_chunks"):
                        state["literals_capped_chunks"] += _oc.get("literals_capped_chunks", 0)
                        state["literals_dropped"] += _oc.get("literals_dropped", 0)
                    # discovered_macros excludes ones already in `vocab`
                    # (those parse cleanly and never resurface here).
                    for _m in _oc.get("discovered_macros") or ():
                        macro_file_counts[_m] = macro_file_counts.get(_m, 0) + 1
                    # Only set when self_heal actually ran, so a memoized
                    # skip never re-adds an already-recorded hash.
                    if _oc.get("heal_unhealable") and content_hash not in unhealable_hashes:
                        unhealable_hashes.add(content_hash)
                        unhealable_order.append(content_hash)
                        rs.new_unhealable = True
                    if coverage < 0.70:
                        state["low_coverage_files"] += 1
                        if coverage < state["worst_coverage"]:
                            state["worst_coverage"]      = coverage
                            state["worst_coverage_file"] = stored_path
                stat = fpath.stat()
                n    = len(segs)
                file_metadata = _file_metadata(stored_path, lang)
                # file_symbols must be populated BEFORE the first chunk for
                # this file is enqueued: the embedder can complete the file
                # (popping file_symbols) as soon as the last chunk lands.
                chunk_ranges: list = []
                chunk_dicts: list = []
                for seg in segs:
                    cid = _chunk_id(stored_path, seg["start_line"], seg["content"])
                    chunk_ranges.append((seg["start_line"], seg["end_line"], cid))
                    chunk_dicts.append({
                        "id":           cid,
                        "path":         stored_path,
                        "language":     lang,
                        "chunk_type":   seg["chunk_type"],
                        "name":         seg["name"],
                        "start_line":   seg["start_line"],
                        "end_line":     seg["end_line"],
                        "content":      seg["content"],
                        "metadata":     _chunk_metadata(file_metadata, seg.get("refs")),
                        "literals":     seg.get("literals") or [],
                        # private metadata stripped before DB insert
                        "_fpath":        stored_path,
                        "_content_hash": content_hash,
                        "_file_total":   n,
                        "_stat":         stat,
                    })
                # Map each symbol to its containing chunk (ranges tile the
                # file). Handed to the embedder thread to write on commit.
                sym_rows = [
                    {
                        "path":       stored_path,
                        "name":       sym["name"],
                        "kind":       sym["kind"],
                        "language":   sym["language"],
                        "start_line": sym["start_line"],
                        "end_line":   sym["end_line"],
                        "chunk_id":   next((cc for cs, ce, cc in chunk_ranges
                                            if cs <= sym["start_line"] <= ce), None),
                    }
                    for sym in (_oc.get("symbols") or [])
                ]
                with lock:
                    if sym_rows:
                        file_symbols[stored_path] = sym_rows
                    state["files_parsed"]  += 1
                    state["chunks_queued"] += n
                # embed_q.put() must NOT be called under `lock`: commit_phase
                # needs `lock` to drain, so holding it here would deadlock.
                for cd in chunk_dicts:
                    embed_q.put(cd)
                logger.info("Parsed %s (%d chunks)", stored_path, n)
            except Exception as e:
                with lock:
                    state["files_parsed"] += 1
                    state["errors"]       += 1
                logger.error("PARSE ERROR %s: %s", fpath, e)
    except Exception as e:
        worker_exc[0] = e
        logger.error("Parser worker crashed: %s", e, exc_info=True)
        try:
            embed_q.put_nowait(_SENTINEL)  # unblock embedder so it can drain and exit
        except queue.Full:
            pass  # embedder also crashed or queue full; join timeout handles cleanup


# ------------------------------------------------------ embedder thread
def embedder_worker(rs: RunState, store: Store, embedder: Embedder, embed_batch: int, embed_inflight: int) -> None:
    state = rs.state
    lock = rs.lock
    done_event = rs.done_event
    file_symbols = rs.file_symbols
    worker_exc = rs.worker_exc
    changed_chunk_ids = rs.changed_chunk_ids
    embed_q = rs.embed_q
    # embed_phase runs in a thread pool so HTTP overlaps the GPU; commit_phase
    # runs serially here (single-writer). Futures drained FIFO to keep
    # insertion order matching submission order.
    file_embedded: dict[str, int]          = {}  # fpath → chunks embedded
    file_dropped: dict[str, int]           = {}  # fpath → chunks dropped (embed error)
    file_meta: dict[str, tuple]            = {}  # fpath → (hash, total, stat)
    batch: list[dict]                      = []

    def embed_phase(snapshot: list[dict], client: httpx.Client) -> tuple[list[dict], list[list[float]], int, list[tuple[dict, Exception | None]], int, Exception | None]:
        """Network phase (thread pool); only shared-state touch is the
        watchdog clock. Returns batch_exc, the top-level failure if any,
        for commit_phase's embedder-down circuit breaker."""
        texts = [_compress_for_embed(c["content"], name=c.get("name"), path=c.get("path")) for c in snapshot]

        def _embed_live(ts: list[str]) -> list[list[float]]:
            """Resets the watchdog clock on every round trip that RETURNS,
            success or exception. This is why the no-progress watchdog can't
            catch a fast-failing dead embedder (see EmbedderDownError)."""
            try:
                return embedder.embed_documents(ts, client, timeout=compute_embed_timeout(len(ts)))
            finally:
                with lock:
                    state["last_progress_ts"] = time.monotonic()

        with lock:
            state["inflight_batches"][id(snapshot)] = _batch_label(snapshot)
        try:
            return _embed_phase_inner(snapshot, texts, _embed_live)
        finally:
            with lock:
                state["inflight_batches"].pop(id(snapshot), None)

    def _embed_phase_inner(snapshot: list[dict], texts: list[str], _embed_live) -> tuple[list[dict], list[list[float]], int, list[tuple[dict, Exception | None]], int, Exception | None]:
        try:
            embeddings = _embed_live(texts)
            return snapshot, embeddings, 0, [], 0, None
        except Exception as e:
            logger.warning("EMBED batch failed (%d chunks), isolating: %s", len(snapshot), e)
            ok_chunks, embeddings, truncated, errors = _embed_isolating(
                snapshot, texts, _embed_live,
            )
            for chunk, exc in errors:
                logger.error("EMBED ERROR %s:%s: %s", chunk["path"], chunk["start_line"], exc)
            return ok_chunks, embeddings, truncated, errors, 1, e

    def _maybe_complete_file(fp: str) -> None:
        """A file completes once every chunk is embedded or dropped. Dropped
        chunks still count, or the files row for content with any
        unembeddable chunk would never get written."""
        total = file_meta[fp][1]
        if file_embedded.get(fp, 0) + file_dropped.get(fp, 0) != total:
            return
        ch, _, stat = file_meta[fp]
        store.upsert_file(fp, stat.st_size, stat.st_mtime, ch)
        # Refresh the file's decoupled symbols. delete-then-insert is
        # idempotent on re-index; written here on the single-writer
        # thread so symbols stay consistent with the file's chunks.
        with lock:
            rows = file_symbols.pop(fp, None)
        store.delete_symbols_for_path(fp)
        if rows:
            store.insert_symbols(rows)
        with lock:
            state["indexed"] += 1

    def commit_phase(chunks_to_insert: list[dict], embeddings: list[list[float]], truncated: int, errors: list[tuple[dict, Exception | None]], batch_failed: int = 0, batch_exc: Exception | None = None) -> None:
        """DB-write phase. Runs only on the embedder thread so inserts and
        file-completion bookkeeping stay strictly ordered."""
        # A batch reaching commit_phase at all, success or failure, is
        # forward progress; reset the no-progress watchdog clock.
        with lock:
            state["last_progress_ts"] = time.monotonic()
        if batch_failed:
            with lock:
                state["batch_failures"] += batch_failed

        # See EmbedderDownError. A 4xx (oversize chunk) doesn't count:
        # bisection handles it legitimately, so it shouldn't trip this streak.
        if chunks_to_insert or not batch_failed:
            with lock:
                state["consecutive_batch_failures"] = 0
        elif not _should_truncate_and_retry(batch_exc):
            with lock:
                state["consecutive_batch_failures"] += 1
                n = state["consecutive_batch_failures"]
            if n >= EMBEDDER_DOWN_THRESHOLD:
                raise EmbedderDownError(
                    f"{n} consecutive whole-batch embed failures against "
                    f"{embedder.url} (non-retryable: {batch_exc!r}) — "
                    f"embedder appears down. Aborting rather than silently "
                    f"dropping the rest of the corpus."
                )
        else:
            with lock:
                state["consecutive_batch_failures"] = 0

        touched: set[str] = set()

        # Drops processed FIRST: a file whose chunks ALL fail must still
        # complete, or its files row never gets written and its chunks
        # orphan permanently.
        if errors:
            with lock:
                state["errors"] += len(errors)
            for c, _exc in errors:
                fp = c["_fpath"]
                if fp not in file_meta:
                    file_meta[fp] = (c["_content_hash"], c["_file_total"], c["_stat"])
                file_dropped[fp] = file_dropped.get(fp, 0) + 1
                touched.add(fp)

        if not chunks_to_insert:
            if truncated:
                with lock:
                    state["truncated"] += truncated
            if touched:
                for fp in touched:
                    _maybe_complete_file(fp)
                store.commit()
            return

        clean = [{k: v for k, v in c.items() if not k.startswith("_")}
                 for c in chunks_to_insert]
        store.insert_chunks(clean, embeddings, model=embedder.model)
        with lock:
            changed_chunk_ids.update(c["id"] for c in clean)

        for c in chunks_to_insert:
            fp = c["_fpath"]
            if fp not in file_meta:
                file_meta[fp] = (c["_content_hash"], c["_file_total"], c["_stat"])
            file_embedded[fp] = file_embedded.get(fp, 0) + 1
            touched.add(fp)

        for fp in touched:
            _maybe_complete_file(fp)

        store.commit()
        with lock:
            state["chunks_embedded"] += len(chunks_to_insert)
            state["truncated"]       += truncated

    try:
        with httpx.Client() as client, _DaemonPool(max_workers=embed_inflight) as pool:
            inflight: deque = deque()

            def submit_flush() -> None:
                """Move the current batch into a pending future; block first
                on the oldest in-flight future if we're already at capacity,
                so HTTP work overlaps with the previous batch's commit."""
                nonlocal batch
                if not batch:
                    return
                while len(inflight) >= embed_inflight:
                    commit_phase(*inflight.popleft().result())
                inflight.append(pool.submit(embed_phase, batch, client))
                batch = []

            while True:
                try:
                    item = embed_q.get(timeout=0.05)
                except queue.Empty:
                    # Idle drain: commit finished batches now, not at the next
                    # flush, or a Ctrl+C during a long unchanged-scan tail
                    # loses embedding work the GPU already did.
                    while inflight and inflight[0].done():
                        commit_phase(*inflight.popleft().result())
                    continue
                if item is _SENTINEL:
                    submit_flush()
                    break
                batch.append(item)
                if len(batch) >= embed_batch:
                    submit_flush()

            while inflight:
                commit_phase(*inflight.popleft().result())
    except Exception as e:
        worker_exc[1] = e
        logger.error("Embedder worker crashed: %s", e, exc_info=True)
    finally:
        done_event.set()


# ------------------------------------------------------ scan producer
# Streams changed files into parse_q as the walk progresses. Orphan
# pruning runs after, since it needs the full set of paths seen.
def scan_producer(rs: RunState, paths: list[str | Path], root: Path | None, store: Store, force: bool,
                  excludes: list[str], includes: list[str], fallback_exts: set[str], data_limit: int) -> None:
    state = rs.state
    lock = rs.lock
    deleted_chunk_ids = rs.deleted_chunk_ids
    deleted_chunk_names = rs.deleted_chunk_names
    parse_q = rs.parse_q
    seen: set[str] = set()
    scanned_stored: set[str] = set()
    excluded_count = 0
    pruned_dirs = 0
    # Dot-directory subtrees containing a nested .git (stale repo copy or
    # worktree). Warn-only: doesn't change what's scanned or indexed.
    dotdir_repo_prefixes: dict[str, int] = {}
    try:
        def emit(fpath: Path) -> None:
            nonlocal excluded_count
            k = str(fpath)
            if k in seen:
                return
            seen.add(k)
            with lock:
                state["files_scanned"] += 1
            try:
                stored_path = _to_stored_path(fpath, root)
                if excludes and not _path_allowed(stored_path, excludes, includes):
                    excluded_count += 1
                    return
                if dotdir_repo_prefixes:
                    dd_prefix = _dotdir_prefix(_stored_dirname(stored_path))
                    if dd_prefix in dotdir_repo_prefixes:
                        dotdir_repo_prefixes[dd_prefix] += 1
                # Dedup by STORED path, not fpath: two on-disk paths (a
                # symlink/junction) can map to one stored path, and
                # processing both crashes on chunk_vecs' UNIQUE constraint.
                if stored_path in scanned_stored:
                    logger.warning(
                        "scan: stored path %r reached via a second on-disk "
                        "path (%s) — indexing once (junction/symlink or "
                        "overlapping include maps two files to one path)",
                        stored_path, fpath)
                    return
                # Added even if not re-indexed: prune treats absence as
                # deleted-from-disk, so a hash error below must not read as one.
                scanned_stored.add(stored_path)
                content_hash = _file_hash(fpath)
                stored_hash  = store.get_file_hash(stored_path)
                if not force and stored_hash == content_hash:
                    with lock:
                        state["skipped"] += 1
                    return
                if stored_hash is not None:
                    removed_names = store.get_names_for_path(stored_path)
                    removed_ids = store.delete_file(stored_path)
                    with lock:
                        deleted_chunk_ids.update(removed_ids)
                        deleted_chunk_names.update(removed_names)
                with lock:
                    state["to_index"] += 1
                parse_q.put((fpath, content_hash, stored_path))
            except Exception as e:
                with lock:
                    state["errors"] += 1
                logger.error("SCAN ERROR %s: %s", fpath, e)

        for p in paths:
            p = Path(p).resolve()
            if p.is_file():
                ext = p.suffix.lower()
                if ext in _EXT_TO_LANG or ext in fallback_exts:
                    if _is_oversize_data_blob(p, ext, data_limit):
                        with lock:
                            state["data_blob_skipped"] += 1
                    else:
                        emit(p)
                else:
                    with lock:
                        state["unsupported_ext_skipped"] += 1
            elif p.is_dir():
                # Single os.walk per top-level path replaces the original
                # rglob-per-extension loop, which traversed the whole tree
                # once per supported extension.
                for dirpath, dirnames, filenames in os.walk(p):
                    dir_path = Path(dirpath)
                    if excludes:
                        # Mutating dirnames in place (the documented way to
                        # stop os.walk descending) so excluded trees are never
                        # walked, not just filtered per-file. Include-aware.
                        kept = []
                        for d in dirnames:
                            child_stored = _to_stored_path(dir_path / d, root)
                            if _dir_should_prune(child_stored, excludes, includes):
                                pruned_dirs += 1
                            else:
                                kept.append(d)
                        dirnames[:] = kept
                    # A .git here means this subtree is itself a repo; flagged
                    # only if also under a dot-directory (stale worktree
                    # copy). Warn-only: never auto-excludes.
                    if ".git" in dirnames or ".git" in filenames:
                        dd_root = _dotdir_prefix(_to_stored_path(dir_path, root))
                        if dd_root is not None:
                            dotdir_repo_prefixes.setdefault(dd_root, 0)
                    for fname in filenames:
                        ext = Path(fname).suffix.lower()
                        if ext in _EXT_TO_LANG or ext in fallback_exts:
                            fpath = (dir_path / fname).resolve()
                            if _is_oversize_data_blob(fpath, ext, data_limit):
                                with lock:
                                    state["data_blob_skipped"] += 1
                            else:
                                emit(fpath)
                        else:
                            # Counted regardless of fallback_extensions config:
                            # this is the "gap is never silent"
                            # signal, not an opt-in diagnostic.
                            with lock:
                                state["unsupported_ext_skipped"] += 1
                            if dotdir_repo_prefixes:
                                dd_prefix = _dotdir_prefix(_to_stored_path(dir_path, root))
                                if dd_prefix in dotdir_repo_prefixes:
                                    dotdir_repo_prefixes[dd_prefix] += 1

        if dotdir_repo_prefixes:
            for dd_prefix, n in sorted(dotdir_repo_prefixes.items(), key=lambda kv: (-kv[1], kv[0])):
                if n == 0:
                    continue
                logger.warning(
                    "%d file%s under %s look%s like a nested repo copy "
                    "(contains .git) — add an exclude if this isn't "
                    'intended: config.json "exclude": ["%s"]',
                    n, "" if n == 1 else "s", dd_prefix, "s" if n == 1 else "", dd_prefix,
                )

        if excluded_count:
            if includes:
                logger.info("Excluded %d files matching %s (with includes %s)",
                            excluded_count, excludes, includes)
            else:
                logger.info("Excluded %d files matching %s", excluded_count, excludes)
        if pruned_dirs:
            # Visibility: how many excluded directories were never
            # descended into at all, vs. filtered file-by-file above.
            logger.info("Pruned %d excluded director%s from the walk (not descended)",
                        pruned_dirs, "y" if pruned_dirs == 1 else "ies")
        with lock:
            state["dirs_pruned"] = pruned_dirs
            data_blob_skipped = state["data_blob_skipped"]

        # Prune orphaned DB entries for files deleted from disk.
        # Scoped to directories/files that were actually scanned, so
        # indexing a subfolder never removes entries from other parts.
        pruned = 0
        for p in paths:
            resolved = Path(p).resolve()
            prefix = _to_stored_path(resolved, root)
            if prefix == ".":
                # relative_to yields "." for root itself; a naive "./"
                # prefix would never match a stored path, silently
                # disabling orphan-prune for the most common invocation.
                prefix = ""
            # Incremented per file, not once at the end: the watchdog's
            # liveness check reads this during a large prune to tell
            # "pruning steadily" from "frozen".
            if resolved.is_dir():
                for db_path in store.get_paths_under(prefix + "/" if prefix else ""):
                    if db_path not in scanned_stored:
                        removed_names = store.get_names_for_path(db_path)
                        removed_ids = store.delete_file(db_path)
                        with lock:
                            deleted_chunk_ids.update(removed_ids)
                            deleted_chunk_names.update(removed_names)
                            state["pruned"] += 1
                        pruned += 1
            else:
                if prefix not in scanned_stored and store.get_file_hash(prefix) is not None:
                    removed_names = store.get_names_for_path(prefix)
                    removed_ids = store.delete_file(prefix)
                    with lock:
                        deleted_chunk_ids.update(removed_ids)
                        deleted_chunk_names.update(removed_names)
                        state["pruned"] += 1
                    pruned += 1

        # Load-bearing: in the prune-only case (no chunks queued) the
        # embedder never calls store.commit(), so this is the only flush
        # of the producer-side deletes.
        store.commit()

        with lock:
            state["scan_complete"] = True
            unsupported = state["unsupported_ext_skipped"]
        logger.info("Scan complete: %d to index, %d unchanged, %d pruned.",
                    state["to_index"], state["skipped"], pruned)
        if unsupported:
            logger.info(
                "Skipped %d file(s) with unsupported extension (not in "
                "_EXT_TO_LANG or fallback_extensions).", unsupported,
            )
        if data_blob_skipped:
            logger.info(
                "Skipped %d oversize data file(s) (over the %d-byte "
                "data_blob_size_limit).", data_blob_skipped, data_limit,
            )
    finally:
        parse_q.put(_SENTINEL)


def index_paths(
    paths: list[str | Path],
    store: Store,
    embedder: Embedder,
    *,
    root: Path | None = None,
    force: bool = False,
    exclude: list[str] | None = None,
    include: list[str] | None = None,
    embed_batch: int = EMBED_BATCH,
    embed_inflight: int = EMBED_INFLIGHT,
    macros: set | None = None,
    fallback_extensions: list[str] | None = None,
    data_blob_size_limit: int | None = None,
    edge_type_weights: dict[str, float] | None = None,
    cap_mentions_fanout: bool = False,
    associated_top_frac: float = 0.0,
    no_progress_timeout: float = EMBED_WATCHDOG_SECS,
    reporter=None,
) -> dict[str, int]:
    """Three-stage pipeline (scan, parse, embed); returns run stats. Config
    params are in DOCS.md. `cap_mentions_fanout`/`associated_top_frac` need
    `--rebuild-graphs` to take effect on an existing index. `reporter` is a
    context manager that yields an object with `tick` and `finish`; None
    draws the tqdm bars."""
    excludes = _normalize_prefixes(exclude)
    includes = _normalize_prefixes(include)
    fallback_exts = {e.lower() for e in
                     (fallback_extensions if fallback_extensions is not None
                      else DEFAULT_FALLBACK_EXTENSIONS)}
    data_limit = (data_blob_size_limit if data_blob_size_limit is not None
                  else DEFAULT_DATA_BLOB_SIZE_LIMIT)
    t0 = time.monotonic()  # for the end-of-run chunks/s throughput line

    persisted, vocab, unhealable_order, unhealable_hashes, new_unhealable = load_macro_memo(store, macros)
    # Per-macro file count, checked against _MACRO_PERSIST_MIN_FILES at persist
    # time below.
    macro_file_counts: dict[str, int] = {}

    # Record/validate root in DB meta
    if root is not None:
        stored_root = store.get_meta("indexed_root")
        root_posix = root.as_posix()
        if stored_root is None:
            store.set_meta("indexed_root", root_posix)
        elif stored_root != root_posix:
            logger.warning("DB was indexed with root '%s', now using '%s'", stored_root, root_posix)

    # doctor.py's staleness sweep reads these back, so it prunes what the
    # indexer ignores instead of flagging mtime churn on them as stale.
    store.set_meta("exclude", json.dumps(excludes))
    store.set_meta("include", json.dumps(includes))

    # --------------------------------------------------------- shared state
    state = {
        "files_parsed":    0,
        "files_scanned":   0,       # unique files the producer has discovered (drives the bar pre-scan-complete)
        "chunks_queued":   0,
        "chunks_embedded": 0,
        "indexed":         0,
        "errors":          0,
        "truncated":       0,
        "batch_failures":  0,       # batches that failed the whole-batch embed and fell to bisection
        "consecutive_batch_failures": 0,  # top-level whole-batch failures in a row under a
                                           # non-4xx exception (circuit breaker,
                                           # EMBEDDER_DOWN_THRESHOLD); reset on any success
        "oversize_chunks": 0,       # chunks over CHUNK_MAX the chunker byte-split (long-line/data files)
        "oversize_files":  0,
        "parse_error_files":  0,    # tree-sitter reported a parse error AFTER self-heal (genuinely unparseable)
        "macro_healed_files": 0,    # C++ files whose macros were discovered + blanked to fix the parse
        "error_salvaged_files": 0,  # files where error-recovery boundary salvage fired
        "error_salvaged_nodes": 0,  # total glued boundary nodes salvaged across all files
        "literals_capped_chunks": 0,  # chunks whose literal count hit the per-chunk cap
        "literals_dropped": 0,        # total literals silently dropped by that cap
        "low_coverage_files": 0,    # <70% of file bytes reached any chunk (silent content loss)
        "worst_coverage":     1.0,  # lowest per-file coverage seen, for a pointer to the worst offender
        "worst_coverage_file": "",
        "skipped":         0,
        "to_index":        0,       # files producer pushed (known only after scan finishes)
        "unsupported_ext_skipped": 0,  # files never even queued: extension has no grammar
                                       # and isn't in fallback_extensions (visibility)
        "data_blob_skipped": 0,  # oversize machine-generated data file, guard hit
        "pruned":          0,
        "dirs_pruned":     0,  # excluded directories skipped without descending
        "scan_complete":   False,
        "current_file":    "",  # parser position, runs ahead of the embedder
        "last_progress_ts": time.monotonic(),  # last embedder response; watchdog clock
        # Batches currently inside an embedder round trip, keyed by batch identity
        # (several are in flight at once at embed_inflight > 1). See the
        # watchdog's abort message below for why this, not `current_file`.
        "inflight_batches": {},
    }
    lock       = threading.Lock()
    done_event = threading.Event()
    # Decoupled symbol rows per file, handed from the parser thread to the
    # embedder thread. Guarded by `lock`.
    file_symbols: dict[str, list[dict]] = {}
    worker_exc: list[Exception | None] = [None, None]  # [parser_exc, embedder_exc]
    # Chunk ids inserted/deleted this run, threaded to build_neighbors so it can
    # take the exact incremental k-NN update path. Guarded by `lock`: deletes
    # happen on the scan-producer thread, inserts on the embedder thread.
    changed_chunk_ids: set[str] = set()
    deleted_chunk_ids: set[str] = set()
    # Names deleted chunks/symbols defined, captured BEFORE each delete_file
    # call (those rows are gone by the time build_refs runs at the end).
    # Guarded by `lock`, same as the id sets.
    deleted_chunk_names: set[str] = set()

    parse_q: queue.Queue = queue.Queue(maxsize=64)    # (fpath, hash) → parser
    embed_q: queue.Queue = queue.Queue(maxsize=2000)  # chunk dicts  → embedder
    rs = RunState(
        state=state, lock=lock, done_event=done_event, file_symbols=file_symbols,
        worker_exc=worker_exc, changed_chunk_ids=changed_chunk_ids,
        deleted_chunk_ids=deleted_chunk_ids, deleted_chunk_names=deleted_chunk_names,
        parse_q=parse_q, embed_q=embed_q, macro_file_counts=macro_file_counts,
        unhealable_hashes=unhealable_hashes, unhealable_order=unhealable_order,
        new_unhealable=new_unhealable,
    )

    # ---------------------------------------------------- start threads
    t_parser   = threading.Thread(target=parser_worker,   args=(rs, vocab), daemon=True)
    t_embedder = threading.Thread(target=embedder_worker, args=(rs, store, embedder, embed_batch, embed_inflight), daemon=True)
    t_scanner  = threading.Thread(target=scan_producer,   args=(rs, paths, root, store, force, excludes, includes, fallback_exts, data_limit), daemon=True)
    t_parser.start()
    t_embedder.start()
    t_scanner.start()

    # ---------------------------------------------------- progress bars
    with (reporter if reporter is not None else tqdm_reporter()) as progress:

        _parser_sentinel_pushed = False
        # Scanner/parser liveness for the watchdog's idle branch below: any
        # movement in these counters proves the pipeline is alive even when
        # the embedder has nothing in flight.
        _last_counters = None
        _last_activity_ts = time.monotonic()

        while not done_event.is_set():
            # If the parser crashed without pushing its sentinel, nudge the embedder
            # so it can drain whatever is already in the queue and exit cleanly.
            if worker_exc[0] is not None and not t_parser.is_alive() and not _parser_sentinel_pushed:
                try:
                    embed_q.put_nowait(_SENTINEL)
                    _parser_sentinel_pushed = True
                except queue.Full:
                    pass  # retry next tick

            with lock:
                parsed      = state["files_parsed"]
                scanned     = state["files_scanned"]
                queued      = state["chunks_queued"]
                embedded    = state["chunks_embedded"]
                cur_file    = state["current_file"]
                scan_done   = state["scan_complete"]
                to_index    = state["to_index"]
                last_progress_ts = state["last_progress_ts"]
                pruned_ct   = state["pruned"]
                inflight    = sorted(state["inflight_batches"].values())

            # No-progress watchdog (see NoProgressError). Embedder-quiet alone
            # isn't sufficient: a healthy long unchanged-scan tail also shows
            # nothing in flight, so also require scanner/parser to be frozen.
            counters = (scanned, parsed, queued, embedded, pruned_ct)
            if counters != _last_counters:
                _last_counters = counters
                _last_activity_ts = time.monotonic()
            stalled_for  = time.monotonic() - last_progress_ts
            activity_for = time.monotonic() - _last_activity_ts
            if stalled_for > no_progress_timeout and (
                inflight or activity_for > no_progress_timeout
            ):
                # Reports inflight_batches, not current_file: the parser runs
                # far ahead of the embedder, so current_file would misdirect
                # a wedge diagnosis (except in the idle branch, where it IS the suspect).
                stuck_on = (", ".join(inflight) if inflight
                            else "no batch in flight — scanner/parser frozen")
                raise NoProgressError(
                    f"No embedder response in {stalled_for:.0f}s "
                    f"(timeout {no_progress_timeout:.0f}s). Last progress at "
                    f"{time.ctime(time.time() - stalled_for)}. Embedder: "
                    f"{embedder.url}. Embedding: {stuck_on} "
                    f"(chunks embedded={embedded} of {queued} queued; "
                    f"parser at {cur_file or 'unknown'}). "
                    f"The run appears wedged — aborting."
                )

            progress.tick(scanned, parsed, queued, embedded, cur_file, scan_done, to_index)

            time.sleep(0.1)

        # flush final counts
        with lock:
            parsed   = state["files_parsed"]
            queued   = state["chunks_queued"]
            embedded = state["chunks_embedded"]
        progress.finish(parsed, queued, embedded)

    # Timeout guards: if a worker crashed and left the scanner/parser blocked on a
    # full queue, a boundless join would hang. 10 s is generous for clean shutdown.
    t_scanner.join(timeout=10)
    t_parser.join(timeout=10)
    t_embedder.join(timeout=10)

    if worker_exc[0] is not None:
        raise RuntimeError(f"Parser worker crashed: {worker_exc[0]}") from worker_exc[0]
    if worker_exc[1] is not None:
        # EmbedderDownError is itself the diagnostic; re-raise it
        # directly instead of wrapping in a generic "Embedder worker crashed"
        # RuntimeError, so callers can catch/match it by type.
        if isinstance(worker_exc[1], EmbedderDownError):
            raise worker_exc[1]
        raise RuntimeError(f"Embedder worker crashed: {worker_exc[1]}") from worker_exc[1]

    persist_macro_memo(store, persisted, macro_file_counts, unhealable_order, rs.new_unhealable)

    # Wall-clock scan+parse+embed only, excluding post-processing below. The
    # number to watch when tuning embed_batch/embed_inflight/--parallel.
    embed_elapsed = time.monotonic() - t0
    with lock:
        chunks_embedded_total = state["chunks_embedded"]
    chunks_per_s = chunks_embedded_total / embed_elapsed if embed_elapsed > 0 else 0.0
    logger.info(
        "Embedded %d chunks in %.1fs (%.1f chunks/s).",
        chunks_embedded_total, embed_elapsed, chunks_per_s,
    )

    pruned = state["pruned"]

    # Only claim a clean chunker-version match when skipped==0 (every tracked
    # file reprocessed this run); otherwise record "mixed" rather than
    # silently claiming a false match (doctor.py reads this back).
    _prev_chunker_version = store.get_meta("chunker_version")
    if _prev_chunker_version is None or state["skipped"] == 0:
        store.set_meta("chunker_version", str(CHUNKER_VERSION))
    elif _prev_chunker_version != str(CHUNKER_VERSION):
        store.set_meta(
            "chunker_version",
            f"mixed: {_prev_chunker_version}+{CHUNKER_VERSION} "
            f"({state['skipped']} unchanged file(s) retain old chunk boundaries)",
        )

    # Same three-branch shape as chunker_version, and for the same reason:
    # files skipped as unchanged were chunked by whatever language set was
    # current when they were last indexed, not this run's.
    _cur_language_set = json.dumps(_language_set(), sort_keys=True, separators=(",", ":"))
    _prev_language_set = store.get_meta("language_set")
    if _prev_language_set is None or state["skipped"] == 0:
        store.set_meta("language_set", _cur_language_set)
    elif _prev_language_set != _cur_language_set:
        store.set_meta(
            "language_set",
            f"mixed: {_prev_language_set}+{_cur_language_set} "
            f"({state['skipped']} unchanged file(s) retain old chunk boundaries)",
        )

    # skipped==0 alone isn't sufficient here (a --force on a path subset also
    # yields it): also require indexed == tracked_file_count, so this run
    # demonstrably covered every file the DB tracks, not just what it targeted.
    if store.get_meta("literal_index_version") is not None or (
        state["skipped"] == 0 and state["indexed"] == store.tracked_file_count()
    ):
        store.set_meta("literal_index_version", "1")

    fts_elapsed, refs_elapsed, knn_elapsed, summaries_elapsed, pagerank_elapsed, hierarchy_elapsed = run_post_index_passes(
        store, embedder,
        force=force, indexed=state["indexed"], pruned=pruned,
        changed_chunk_ids=changed_chunk_ids, deleted_chunk_ids=deleted_chunk_ids,
        deleted_chunk_names=deleted_chunk_names, edge_type_weights=edge_type_weights,
        cap_mentions_fanout=cap_mentions_fanout, associated_top_frac=associated_top_frac,
    )

    total_elapsed = time.monotonic() - t0

    with lock:
        return {
            "indexed":         state["indexed"],
            "skipped":         state["skipped"],
            "errors":          state["errors"],
            "pruned":          pruned,
            "dirs_pruned":     state["dirs_pruned"],
            "truncated":       state["truncated"],
            "chunks_embedded": chunks_embedded_total,
            "elapsed_s":       round(embed_elapsed, 1),
            "chunks_per_s":    round(chunks_per_s, 1),
            "batch_failures":  state["batch_failures"],
            "oversize_chunks": state["oversize_chunks"],
            "oversize_files":  state["oversize_files"],
            "parse_error_files":   state["parse_error_files"],
            "macro_healed_files":  state["macro_healed_files"],
            "error_salvaged_files": state["error_salvaged_files"],
            "error_salvaged_nodes": state["error_salvaged_nodes"],
            "literals_capped_chunks": state["literals_capped_chunks"],
            "literals_dropped":       state["literals_dropped"],
            "low_coverage_files":  state["low_coverage_files"],
            "worst_coverage":      round(state["worst_coverage"], 2),
            "worst_coverage_file": state["worst_coverage_file"],
            "unsupported_ext_skipped": state["unsupported_ext_skipped"],
            "data_blob_skipped": state["data_blob_skipped"],
            "embed_elapsed_s":      round(embed_elapsed, 1),
            "fts_elapsed_s":        round(fts_elapsed, 1),
            "refs_elapsed_s":       round(refs_elapsed, 1),
            "knn_elapsed_s":        round(knn_elapsed, 1),
            "summaries_elapsed_s":  round(summaries_elapsed, 1),
            "pagerank_elapsed_s":   round(pagerank_elapsed, 1),
            "hierarchy_elapsed_s":  round(hierarchy_elapsed, 1),
            "total_elapsed_s":      round(total_elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Re-embed
# ---------------------------------------------------------------------------

def reembed_all(
    store: Store,
    embedder: Embedder,
    *,
    embed_batch: int = EMBED_BATCH,
) -> dict[str, int]:
    """Re-embed all chunks already in the DB using the current _compress_for_embed format.
    Only chunk_vecs is updated; the chunks table is untouched. Useful after changing
    the embedding format (e.g. adding breadcrumb prepending) without re-parsing source."""
    stats: dict[str, int] = {"reembedded": 0, "errors": 0}
    total = store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    logger.info("Re-embedding %d chunks from DB (vectors only)...", total)

    PAGE   = embed_batch
    offset = 0

    with httpx.Client() as client:
        with tqdm(total=total, desc="reembed", unit="chunk") as bar:
            while True:
                rows = store._conn.execute(
                    "SELECT id, path, name, start_line, content FROM chunks ORDER BY rowid LIMIT ? OFFSET ?",
                    (PAGE, offset),
                ).fetchall()
                if not rows:
                    break

                texts = [
                    _compress_for_embed(r["content"], name=r["name"], path=r["path"])
                    for r in rows
                ]
                ids = [r["id"] for r in rows]

                try:
                    embeddings = embedder.embed_documents(texts, client)
                    store.update_vectors(ids, embeddings, model=embedder.model)
                    stats["reembedded"] += len(rows)
                    bar.update(len(rows))
                except Exception as e:
                    logger.warning("EMBED batch failed at offset %d, isolating: %s", offset, e)
                    ok_rows, ok_embs, _trunc, errs = _embed_isolating(
                        rows, texts, lambda ts: embedder.embed_documents(ts, client),
                    )
                    if ok_rows:
                        store.update_vectors([r["id"] for r in ok_rows], ok_embs, model=embedder.model)
                        stats["reembedded"] += len(ok_rows)
                    for row, exc in errs:
                        stats["errors"] += 1
                        logger.error("EMBED ERROR %s:%s: %s", row['path'], row['start_line'], exc)
                    bar.update(len(rows))

                offset += PAGE

    return stats
