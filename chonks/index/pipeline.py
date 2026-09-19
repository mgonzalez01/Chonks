"""The indexing pipeline: worker pool, end-of-stream marker and abort errors."""

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass


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
