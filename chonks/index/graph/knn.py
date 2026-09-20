"""k-NN graph construction: backend detection (numpy/MLX/CUDA), the int8
embedding corpus wrapper, and bulk/incremental top-k builds."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from chonks.storage.store import Store

logger = logging.getLogger("repomap")

# Top-K semantic neighbours stored per chunk for GraphRAG traversal.
GRAPHRAG_TOP_K = 10

_NEIGHBOR_INSERT_BATCH = 10000

# Below this batch fraction, update the k-NN graph incrementally; above
# it, a full rebuild beats the per-row bookkeeping cost.
_NEIGHBOR_INCREMENTAL_MAX_FRACTION = 0.20

# How often (seconds) the incremental k-NN path logs progress, so a
# multi-minute silent stretch doesn't read as a hang.
_NEIGHBOR_PROGRESS_INTERVAL_S = 15.0

# Below this many displaced rows, per-row SQL is fast enough that bailing
# to a full rebuild would only add work.
_NEIGHBOR_INCREMENTAL_BAIL_MIN_ROWS = 1000

# On MLX/CUDA the full rebuild is fast enough that the incremental path's
# per-row SQL loses at a much lower churn fraction, so this gate is
# tighter than the numpy one above.
_NEIGHBOR_INCREMENTAL_MAX_FRACTION_DEVICE = 0.02

# Floor/fallback block size for the k-NN matmul; auto-scaled up when RAM
# allows (see _neighbor_block_rows).
_NEIGHBOR_MATMUL_BLOCK = 1024
# Fraction of *available* RAM to spend on one matmul block's working set.
_NEIGHBOR_RAM_FRACTION = 0.33
# Peak bytes per block row ~= N * this (sims array + negated copy +
# argpartition's int64 index array); same in both fast and exact mode.
_NEIGHBOR_BYTES_PER_ROW_ELEM = 16

# Mode picked by embedding dim: float32 dot products are exactly correct
# (bit-identical to int32) only when dim*127^2 < 2^24, i.e. dim <= 1040;
# above that a RAM budget decides fast (approximate) vs exact (dgemm).
_NEIGHBOR_DIM_SAFE_DOT_PRODUCT_MAX = 2 ** 24

# RAM-aware fallback for dim-unsafe corpora: if the one-shot float32
# corpus fits this fraction of free RAM, "fast" mode still wins; else
# fall back to exact mode's bounded per-block memory.
_NEIGHBOR_FULL_F32_RAM_FRACTION = 0.25
_NEIGHBOR_FULL_F32_MAX_BYTES = 512 * 1024 * 1024

# Sentinel for masking self-similarity in exact (int32) mode: must stay
# below any real dot product and survive negation without int32 overflow.
_NEIGHBOR_EXACT_SELF_MASK = -(1 << 30)

# Empirical device buffer ceiling for the MLX top-k path: mlx 0.32.0's
# argpartition silently returns garbage past ~3GB with no error, so
# topk_block sub-slices to stay well under this cap.
_MLX_TOPK_MAX_BUFFER_BYTES = 1 << 30

# CUDA twin of the MLX cap above; cupy has no known corruption cliff, but
# the sub-slice discipline is kept as cheap insurance either way.
_CUDA_TOPK_MAX_BUFFER_BYTES = 1 << 30


def _available_ram_bytes() -> int | None:
    """Best-effort available RAM; None when it can't be measured (e.g. macOS
    without psutil), so callers must handle that fallback themselves."""
    try:
        import psutil  # optional; most accurate + cross-platform
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:  # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:  # Windows
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = _MS()
        st.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys)
    except Exception:
        pass
    return None


def _neighbor_block_rows(n: int, override: int | None = None) -> int:
    """Rows per k-NN matmul block, RAM-scaled between the fixed floor and N."""
    if override and override > 0:
        return max(1, min(int(override), n))
    avail = _available_ram_bytes()
    if not avail:
        return min(_NEIGHBOR_MATMUL_BLOCK, n)
    budget = int(avail * _NEIGHBOR_RAM_FRACTION)
    rows = budget // max(1, n * _NEIGHBOR_BYTES_PER_ROW_ELEM)
    # Cap at N so a tiny corpus is one block, not a floor bigger than the corpus.
    return min(max(_NEIGHBOR_MATMUL_BLOCK, int(rows)), n)


# Below this many chunks the CPU k-NN is fast enough that the GPU hint is noise.
_KNN_CPU_HINT_MIN_N = 50_000


def _detect_knn_backend() -> str:
    """Soft probe for 'auto': 'cuda' if usable, else 'mlx' if usable, else 'numpy'.
    Never raises."""
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except Exception:  # noqa: BLE001
        pass
    return "numpy"



class _Corpus:
    """Holds the int8 embedding corpus and picks the fast/exact memory-speed
    mode, shared by the bulk, incremental, and _topk_for_rows callers.
    sims_block() never masks self-similarity; callers do that themselves."""

    def __init__(self, ids: list[str], blob: bytes, dim: int, block_rows: int | None = None):
        self.ids = ids
        self.n = len(ids)
        self.dim = dim
        self.m8 = np.frombuffer(blob, dtype=np.int8).reshape(self.n, dim)
        # Tier 1: dim-safe dims always use fast mode; sgemm is exact there at
        # any N (see the module-level comment above _NEIGHBOR_DIM_SAFE_DOT_PRODUCT_MAX).
        if dim * 127 * 127 < _NEIGHBOR_DIM_SAFE_DOT_PRODUCT_MAX:
            self.exact = False
        else:
            # Tier 2: dim-unsafe corpora pick exact mode based on whether the
            # one-shot float32 corpus fits a RAM budget, or a fixed byte
            # threshold when free RAM can't be measured.
            full_f32_bytes = self.n * dim * 4
            avail = _available_ram_bytes()
            if avail is not None:
                self.exact = full_f32_bytes >= avail * _NEIGHBOR_FULL_F32_RAM_FRACTION
            else:
                self.exact = full_f32_bytes >= _NEIGHBOR_FULL_F32_MAX_BYTES
        if self.exact:
            self.m = None
            self.mt = None
        else:
            # Whole corpus, converted once. Memory: int8 ≈ N·dim B, float32 ≈
            # 4·N·dim B (e.g. 260k × 768 ≈ 800 MB f32).
            self.m = self.m8.astype(np.float32)
            # Transpose VIEW: m.T swaps strides, and sgemm
            # consumes it natively via transB. np.ascontiguousarray(m.T)
            # would allocate a second full N x dim float32 buffer for no benefit.
            self.mt = self.m.T
        # MLX (CHONKS_KNN_BACKEND=mlx): gated on dim-safety, not just fast
        # mode, so results stay bit-identical to BLAS (validated, max diff
        # 0.0). Opt-in only; the incremental path (list row_idx) always stays on numpy.
        self._mx = None
        self._mxt = None
        self._mx_rank = None
        # CUDA (CHONKS_KNN_BACKEND=cuda): unlike MLX, also serves the
        # incremental path's matmuls. An explicit request is HARD: import or
        # device failure raises loudly instead of silently falling back to numpy.
        self._cp = None
        self._cpt = None
        self._cp_rank = None
        backend = os.environ.get("CHONKS_KNN_BACKEND") or "auto"
        explicit = backend != "auto"
        dim_safe = dim * 127 * 127 < _NEIGHBOR_DIM_SAFE_DOT_PRODUCT_MAX
        # 'auto': prefer a GPU backend when dim-safe and available; an
        # explicit value keeps its old contract (cuda hard, mlx soft).
        if backend == "auto":
            backend = _detect_knn_backend() if (not self.exact and dim_safe) else "numpy"
            if backend != "numpy":
                logger.info("k-NN backend auto-selected: %s (set CHONKS_KNN_BACKEND or "
                            "config knn_backend to force one)", backend)
        if not self.exact and dim_safe and backend == "mlx":
            try:
                import mlx.core as mx
                self._mx = mx.array(self.m)
                self._mxt = self._mx.T
                logger.info("k-NN matmul backend: MLX (Metal GPU), N=%d dim=%d", self.n, dim)
            except ImportError:
                logger.warning("CHONKS_KNN_BACKEND=mlx set but mlx not importable — using numpy")
        elif backend == "cuda":
            if not self.exact and dim_safe:
                try:
                    import cupy as cp
                except ImportError as e:
                    raise RuntimeError(
                        "CHONKS_KNN_BACKEND=cuda is set but cupy is not "
                        "importable. Install the optional extra "
                        "(`pip install 'chonks[cuda]'` / `uv sync --extra "
                        "cuda`) or unset CHONKS_KNN_BACKEND to use the numpy "
                        f"path. Import error: {e}"
                    ) from e
                try:
                    device_count = cp.cuda.runtime.getDeviceCount()
                except Exception as e:
                    raise RuntimeError(
                        "CHONKS_KNN_BACKEND=cuda is set but querying CUDA "
                        "devices failed (no driver, or no GPU present?). "
                        f"Unset CHONKS_KNN_BACKEND to use the numpy path. "
                        f"Underlying error: {e}"
                    ) from e
                if device_count < 1:
                    raise RuntimeError(
                        "CHONKS_KNN_BACKEND=cuda is set but no CUDA device "
                        "was detected (cupy.cuda.runtime.getDeviceCount() "
                        "== 0). Unset CHONKS_KNN_BACKEND to use the numpy "
                        "path."
                    )
                self._cp = cp.asarray(self.m)
                self._cpt = self._cp.T
                logger.info("k-NN matmul backend: CUDA (cupy), N=%d dim=%d", self.n, dim)
            else:
                logger.info(
                    "CHONKS_KNN_BACKEND=cuda set but this corpus isn't "
                    "GPU-safe (exact=%s, dim=%d) — using numpy, same gate "
                    "as the MLX backend.", self.exact, dim,
                )
        if (self._cp is None and self._mx is None and not (explicit and backend == "numpy")
                and self.n >= _KNN_CPU_HINT_MIN_N):
            logger.warning(
                "k-NN graph on CPU for N=%d chunks: expect tens of minutes on a large "
                "corpus. On an NVIDIA machine run `uv sync --extra cuda` and re-run; the "
                "GPU path is picked up automatically and gives a bit-identical result. "
                "On Apple silicon install the `mlx` package.", self.n,
            )
        # Block size also sets exact mode's column-chunk width; RAM is
        # measured after self.m is resident so the budget reflects what's free.
        self.block = _neighbor_block_rows(self.n, block_rows)

    def sims_block(self, row_idx) -> np.ndarray:
        """(len(row_idx), n) similarity matrix for the given global row
        index slice/list against the WHOLE current corpus."""
        if not self.exact:
            if self._cp is not None:
                # CUDA also serves the incremental caller (list row_idx),
                # unlike MLX (see __init__); the transfer back is lossless.
                import cupy as cp
                idx_dev = row_idx if isinstance(row_idx, slice) else cp.asarray(row_idx)
                return cp.asnumpy(self._cp[idx_dev] @ self._cpt)
            # Plain numpy: no device backend armed, or it's MLX (which uses
            # topk_block for the bulk path, stays numpy for incremental).
            return self.m[row_idx] @ self.mt
        rows64 = self.m8[row_idx].astype(np.float64)
        out = np.empty((rows64.shape[0], self.n), dtype=np.int32)
        for cs in range(0, self.n, self.block):
            ce = min(cs + self.block, self.n)
            cols64 = self.m8[cs:ce].astype(np.float64)
            # float64 holds every reachable dot-product magnitude exactly, so
            # this cast is lossless (see the module-level mode-selection comment).
            out[:, cs:ce] = (rows64 @ cols64.T).astype(np.int32)
        return out

    def mask_self(self, sims: np.ndarray, global_rows) -> None:
        """In-place: set each row's own-column entry so it never wins top-k."""
        sentinel = _NEIGHBOR_EXACT_SELF_MASK if self.exact else -np.inf
        for local_i, global_i in enumerate(global_rows):
            sims[local_i, global_i] = sentinel

    def topk_block(self, start: int, end: int, k: int, rank: np.ndarray):
        """The bulk path's per-block top-k, dispatched to whichever device
        backend is armed (MLX or CUDA/cupy). Not callable if neither is."""
        if self._mx is not None:
            return self._topk_block_mlx(start, end, k, rank)
        if self._cp is not None:
            return self._topk_block_cuda(start, end, k, rank)
        raise RuntimeError("topk_block called without an active device backend")

    def _topk_block_mlx(self, start: int, end: int, k: int, rank: np.ndarray):
        """GPU top-k for one block: matmul, mask, tie-break, and sort all stay
        on Metal, only (block x k) arrays cross back. Sub-sliced under
        _MLX_TOPK_MAX_BUFFER_BYTES: argpartition silently corrupts past that size."""
        import mlx.core as mx
        if self._mx_rank is None:
            self._mx_rank = mx.array(rank)
        sub = max(1, min(end - start, _MLX_TOPK_MAX_BUFFER_BYTES // (self.n * 8)))
        idx_parts: list[np.ndarray] = []
        sim_parts: list[np.ndarray] = []
        for s in range(start, end, sub):
            e = min(s + sub, end)
            sims = self._mx[s:e] @ self._mxt  # (sub, N) f32, exact ints
            sims[mx.arange(e - s), mx.arange(s, e)] = _NEIGHBOR_EXACT_SELF_MASK
            neg_keys = self._mx_rank[None, :] - sims.astype(mx.int64) * (self.n + 1)
            topk_idx = mx.argpartition(neg_keys, kth=k, axis=1)[:, :k]
            order = mx.argsort(mx.take_along_axis(neg_keys, topk_idx, axis=1), axis=1)
            topk_idx = mx.take_along_axis(topk_idx, order, axis=1)
            topk_sims = mx.take_along_axis(sims, topk_idx, axis=1)
            mx.eval(topk_idx, topk_sims)
            idx_parts.append(np.array(topk_idx))
            sim_parts.append(np.array(topk_sims))
        idx = idx_parts[0] if len(idx_parts) == 1 else np.concatenate(idx_parts)
        sims_out = sim_parts[0] if len(sim_parts) == 1 else np.concatenate(sim_parts)
        # A repeated column index means the GPU kernel corrupted this row.
        sorted_idx = np.sort(idx, axis=1)
        if idx.shape[1] > 1 and not (sorted_idx[:, 1:] > sorted_idx[:, :-1]).all():
            raise RuntimeError(
                "MLX k-NN top-k returned duplicate neighbor indices — GPU "
                "kernel corruption (known Metal large-buffer failure mode). "
                "Unset CHONKS_KNN_BACKEND to use the numpy path, and report "
                "N/dim/block so the buffer cap can be adjusted."
            )
        return idx, sims_out

    def _topk_block_cuda(self, start: int, end: int, k: int, rank: np.ndarray):
        """CUDA twin of `_topk_block_mlx`: same algorithm and determinism
        argument, cupy instead of Metal. No known corruption cliff here, but
        the same buffer cap and distinct-index check are kept as insurance."""
        import cupy as cp
        if self._cp_rank is None:
            self._cp_rank = cp.asarray(rank)
        sub = max(1, min(end - start, _CUDA_TOPK_MAX_BUFFER_BYTES // (self.n * 8)))
        idx_parts: list[np.ndarray] = []
        sim_parts: list[np.ndarray] = []
        for s in range(start, end, sub):
            e = min(s + sub, end)
            sims = self._cp[s:e] @ self._cpt  # (sub, N) f32, exact ints
            sims[cp.arange(e - s), cp.arange(s, e)] = _NEIGHBOR_EXACT_SELF_MASK
            neg_keys = self._cp_rank[None, :] - sims.astype(cp.int64) * (self.n + 1)
            topk_idx = cp.argpartition(neg_keys, kth=k, axis=1)[:, :k]
            order = cp.argsort(cp.take_along_axis(neg_keys, topk_idx, axis=1), axis=1)
            topk_idx = cp.take_along_axis(topk_idx, order, axis=1)
            topk_sims = cp.take_along_axis(sims, topk_idx, axis=1)
            idx_parts.append(cp.asnumpy(topk_idx))
            sim_parts.append(cp.asnumpy(topk_sims))
        idx = idx_parts[0] if len(idx_parts) == 1 else np.concatenate(idx_parts)
        sims_out = sim_parts[0] if len(sim_parts) == 1 else np.concatenate(sim_parts)
        # Same defensive check as the MLX path, see its comment.
        sorted_idx = np.sort(idx, axis=1)
        if idx.shape[1] > 1 and not (sorted_idx[:, 1:] > sorted_idx[:, :-1]).all():
            raise RuntimeError(
                "CUDA k-NN top-k returned duplicate neighbor indices — GPU "
                "kernel corruption or a cupy/BLAS argpartition tie-break "
                "mismatch. Unset CHONKS_KNN_BACKEND to use the numpy path, "
                "and report N/dim/block so this can be investigated."
            )
        return idx, sims_out


# ---------------------------------------------------------------------------
# Deterministic k-NN tie-break
# ---------------------------------------------------------------------------

# Every top-k site must break a distance tie the SAME way (smaller
# chunk_id wins), or the incremental-equals-full-rebuild contract breaks.

def _id_rank(ids: list[str]) -> np.ndarray:
    """rank[i] = position of ids[i] in ascending lexicographic order; must
    match Store.trim_neighbors' ORDER BY tie-break exactly."""
    n = len(ids)
    order = sorted(range(n), key=lambda i: ids[i])
    rank = np.empty(n, dtype=np.int64)
    for pos, idx in enumerate(order):
        rank[idx] = pos
    return rank


def _tie_break_keys(sims: np.ndarray, rank: np.ndarray, n: int) -> np.ndarray:
    """Sort key: descending similarity, ties toward the smaller chunk_id.
    float64 holds every reachable value exactly and propagates -inf (the
    self-mask sentinel) unchanged."""
    return sims.astype(np.float64) * (n + 1) - rank[None, :]


def _device_backend_armed(dim: int) -> bool:
    """Cheap pre-check of whether `_Corpus` would dispatch to a device
    backend for this dim, without allocating the corpus. Must mirror
    `_Corpus.__init__`'s dim-safety gate exactly."""
    backend = os.environ.get("CHONKS_KNN_BACKEND")
    if backend not in ("mlx", "cuda"):
        return False
    if dim * 127 * 127 >= _NEIGHBOR_DIM_SAFE_DOT_PRODUCT_MAX:
        return False
    if backend == "mlx":
        try:
            import mlx.core  # noqa: F401
        except ImportError:
            return False
    else:
        try:
            import cupy  # noqa: F401
        except ImportError:
            return False
    return True

def build_neighbors(
    store: "Store",
    k: int = GRAPHRAG_TOP_K,
    *,
    block_rows: int | None = None,
    changed_ids: set[str] | list[str] | None = None,
    deleted_ids: set[str] | list[str] | None = None,
) -> int:
    """Precomputes the k-NN graph via blocked matmul instead of N per-chunk
    sqlite-vec scans. Incremental path falls back to a full rebuild whenever
    k's corpus-size cap would differ before/after the mutation."""
    ids, blob, dim = store.get_all_int8_embeddings()
    n = len(ids)
    if n < 2:
        # 0 or 1 chunks → no neighbours possible. Clear so a shrunk corpus
        # doesn't leave stale edges, then return (also covers the empty case).
        store.clear_neighbors()
        store.commit()
        return 0

    if changed_ids is not None and deleted_ids is not None:
        n_before = n - len(changed_ids) + len(deleted_ids)
        batch = len(changed_ids) + len(deleted_ids)
        # On a device backend the incremental path's per-row SQL loses far
        # sooner than the numpy-tuned fraction; see the constant's docstring.
        device_backend = _device_backend_armed(dim)
        max_fraction = (
            _NEIGHBOR_INCREMENTAL_MAX_FRACTION_DEVICE if device_backend
            else _NEIGHBOR_INCREMENTAL_MAX_FRACTION
        )
        if (
            store.count_neighbors() > 0
            and batch <= max_fraction * max(n, 1)
            and n - 1 >= k
            and n_before - 1 >= k
        ):
            result = _build_neighbors_incremental(
                store, ids, blob, dim, k, set(changed_ids), set(deleted_ids),
                block_rows=block_rows, max_fraction=max_fraction,
            )
            if result is not None:
                return result
            # Mid-flight bail: the incremental path's partial writes are moot
            # since the full rebuild below starts with clear_neighbors().
        else:
            logger.info(
                "k-NN incremental update skipped (batch=%d, N=%d, backend=%s, "
                "max_fraction=%.3f) — falling back to full rebuild.",
                batch, n, "device" if device_backend else "numpy", max_fraction,
            )

    k = min(k, n - 1)  # argpartition requires kth < N
    logger.info("k-NN full rebuild: N=%d, k=%d.", n, k)

    store.clear_neighbors()
    corpus = _Corpus(ids, blob, dim, block_rows)
    block = corpus.block
    device_backend = corpus._mx is not None or corpus._cp is not None
    if device_backend:
        # MLX/CUDA sub-slices under its own buffer cap, so the numpy
        # working-set formula below would overstate peak memory.
        backend_name = "MLX" if corpus._mx is not None else "CUDA"
        cap = _MLX_TOPK_MAX_BUFFER_BYTES if corpus._mx is not None else _CUDA_TOPK_MAX_BUFFER_BYTES
        logger.info(
            "k-NN matmul block: %d rows (%s device buffers capped at ~%.1f GB "
            "per sub-slice, N=%d, mode=%s)",
            block, backend_name, cap / 1e9, n,
            "exact" if corpus.exact else "fast",
        )
    else:
        logger.info(
            "k-NN matmul block: %d rows (~%.1f GB peak working set, N=%d, mode=%s)",
            block, block * n * _NEIGHBOR_BYTES_PER_ROW_ELEM / 1e9, n,
            "exact" if corpus.exact else "fast",
        )

    inserted = 0
    pending: list[tuple[str, str, float]] = []
    inv_scale = 1.0 / (127.0 ** 2)
    rank = _id_rank(ids)

    bar = tqdm(
        range(0, n, block),
        desc="k-NN graph",
        unit="block",
        dynamic_ncols=True,
    )
    for start in bar:
        end = min(start + block, n)
        if device_backend:
            # Whole block's top-k on GPU; rows arrive pre-sorted by
            # tie-break key, so no per-row argsort here.
            topk_idx, topk_sims = corpus.topk_block(start, end, k, rank)
            for i in range(end - start):
                qid = ids[start + i]
                idx_row = topk_idx[i]
                sim_row = topk_sims[i]
                for pos in range(k):
                    dist = 1.0 - float(sim_row[pos]) * inv_scale
                    pending.append((qid, ids[int(idx_row[pos])], dist))
        else:
            # (block, N) dot products: sgemm in "fast" mode, dgemm in "exact" mode.
            sims = corpus.sims_block(slice(start, end))
            corpus.mask_self(sims, range(start, end))
            # argpartition's top-k is unsorted; re-sort by descending key
            # (see _tie_break_keys) for a deterministic insert order.
            keys = _tie_break_keys(sims, rank, n)
            topk_idx = np.argpartition(-keys, k, axis=1)[:, :k]
            for i in range(end - start):
                row = sims[i]
                key_row = keys[i]
                cand = topk_idx[i]
                cand = cand[np.argsort(-key_row[cand])]
                qid = ids[start + i]
                for j in cand:
                    dist = 1.0 - float(row[j]) * inv_scale
                    pending.append((qid, ids[int(j)], dist))
        if len(pending) >= _NEIGHBOR_INSERT_BATCH:
            store.insert_neighbors(pending)
            inserted += len(pending)
            pending = []

    if pending:
        store.insert_neighbors(pending)
        inserted += len(pending)
    store.commit()
    return inserted

def _topk_for_rows(
    corpus: "_Corpus", ids: list[str], row_idx: list[int], k: int,
    block: int, inv_scale: float, rank: np.ndarray, *, phase: str = "rows",
) -> list[tuple[str, str, float]]:
    """Exact top-k for given rows vs the whole corpus, using the bulk path's
    same tie-break (`rank`) so a recomputed row matches a from-scratch
    rebuild exactly. `phase` labels progress so silence isn't mistaken for a hang."""
    out: list[tuple[str, str, float]] = []
    n = len(ids)
    total = len(row_idx)
    t0 = time.monotonic()
    last_log = t0
    for start in range(0, total, block):
        chunk = row_idx[start:start + block]
        sims = corpus.sims_block(chunk)  # (len(chunk), N)
        corpus.mask_self(sims, chunk)
        keys = _tie_break_keys(sims, rank, n)
        topk_idx = np.argpartition(-keys, k, axis=1)[:, :k]
        for local_i, global_i in enumerate(chunk):
            row = sims[local_i]
            key_row = keys[local_i]
            cand = topk_idx[local_i]
            cand = cand[np.argsort(-key_row[cand])]
            qid = ids[global_i]
            for j in cand:
                dist = 1.0 - float(row[j]) * inv_scale
                out.append((qid, ids[int(j)], dist))
        now = time.monotonic()
        if now - last_log >= _NEIGHBOR_PROGRESS_INTERVAL_S:
            logger.info(
                "k-NN incremental %s: %d/%d rows recomputed (%.0fs elapsed).",
                phase, min(start + block, total), total, now - t0,
            )
            last_log = now
    return out

def _build_neighbors_incremental(
    store: "Store",
    ids: list[str],
    blob: bytes,
    dim: int,
    k: int,
    changed_ids: set[str],
    deleted_ids: set[str],
    *,
    block_rows: int | None = None,
    max_fraction: float = _NEIGHBOR_INCREMENTAL_MAX_FRACTION,
) -> int | None:
    """Incremental k-NN update, byte-identical to a full rebuild. Returns
    None (caller falls back to a full rebuild) when displaced-row fan-out
    from changed rows exceeds `max_fraction`, only knowable after the step-3 matmul."""
    n = len(ids)
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    inv_scale = 1.0 / (127.0 ** 2)
    rank = _id_rank(ids)

    # Step 1: damaged rows captured before their stale edges are deleted;
    # filtered to ids still present, not already changed or deleted.
    damaged_ids = {
        d for d in store.get_chunk_ids_by_neighbor(list(deleted_ids))
        if d in id_to_idx and d not in changed_ids and d not in deleted_ids
    }

    # Step 2: purge stale edges (outgoing rows of deleted chunks + dangling
    # edges pointing at them, including every damaged row's bad entry).
    removed = store.delete_neighbors_touching(list(deleted_ids))

    # States the workload up front: the passes below can run minutes with
    # no output on a large batch, and silence reads as a hang.
    logger.info(
        "k-NN incremental: N=%d, m=%d changed, a=%d damaged rows to repair — "
        "this can take several minutes on CPU.",
        n, len(changed_ids), len(damaged_ids),
    )

    corpus = _Corpus(ids, blob, dim, block_rows)
    block = corpus.block

    inserted = 0

    # Step 3: changed rows' own top-k, plus a check of whether they displace
    # an existing chunk's worst neighbour.
    changed_rows = sorted(id_to_idx[c] for c in changed_ids if c in id_to_idx)
    if changed_rows:
        own_topk = _topk_for_rows(
            corpus, ids, changed_rows, k, block, inv_scale, rank,
            phase="changed-rows top-k",
        )
        store.insert_neighbors(own_topk)
        inserted += len(own_topk)

        # Threshold per column is -inf for changed/damaged ids (handled
        # elsewhere), else the current worst neighbour's tie-break key: using
        # its id, not just distance, so an exact-tie candidate can still displace it.
        worst = store.get_neighbor_worst_distances()
        excluded = changed_ids | damaged_ids
        thresholds = np.full(n, -np.inf, dtype=np.float64)
        for j, cid in enumerate(ids):
            if cid in excluded:
                continue
            w = worst.get(cid)
            if not w:
                continue
            worst_dist, worst_id, _cnt = w
            wi = id_to_idx.get(worst_id)
            if wi is None:
                # Dangling stored neighbor (pre-existing orphan); leave
                # threshold at -inf so any candidate improves it instead of crashing the build.
                continue
            worst_sim = (1.0 - worst_dist) / inv_scale
            thresholds[j] = worst_sim * (n + 1) - rank[wi]

        improved: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        for start in range(0, len(changed_rows), block):
            chunk = changed_rows[start:start + block]
            sims = corpus.sims_block(chunk)  # (len(chunk), N)
            corpus.mask_self(sims, chunk)
            # NOTE: rows are incoming CANDIDATES, columns are TARGETS whose
            # worst neighbour might get displaced, the reverse of
            # _topk_for_rows' layout; tie-break must use the candidate's (row) rank.
            row_rank = rank[np.asarray(chunk)]
            keys = sims.astype(np.float64) * (n + 1) - row_rank[:, None]
            mask = keys > thresholds[None, :]
            rows_hit, cols_hit = np.nonzero(mask)
            for r, c in zip(rows_hit, cols_hit):
                qid = ids[chunk[int(r)]]
                cid = ids[int(c)]
                dist = 1.0 - float(sims[r, c]) * inv_scale
                improved[cid].append((cid, qid, dist))

        # Mid-flight bail, see the docstring: only the displaced count enters
        # the budget, since step 4's damaged-row repair is one cheap bulk
        # insert while every displaced row here costs individual insert+trim SQL.
        total_improved = len(improved)
        if total_improved > max(_NEIGHBOR_INCREMENTAL_BAIL_MIN_ROWS,
                                max_fraction * n):
            logger.info(
                "k-NN incremental bail: %d displaced rows exceed the %.1f%% "
                "budget (N=%d) — falling back to full rebuild.",
                total_improved, max_fraction * 100, n,
            )
            return None

        # Per-cid SQL (insert + trim) dominates step 3's wall time regardless
        # of matmul backend; progress logged so a long run doesn't read as a hang.
        t0 = time.monotonic()
        last_log = t0
        for i, (cid, rows) in enumerate(improved.items(), start=1):
            store.insert_neighbors(rows)
            inserted += len(rows)
            store.trim_neighbors(cid, k)
            now = time.monotonic()
            if now - last_log >= _NEIGHBOR_PROGRESS_INTERVAL_S:
                logger.info(
                    "k-NN incremental displaced-neighbour trim: %d/%d rows (%.0fs elapsed).",
                    i, total_improved, now - t0,
                )
                last_log = now

    # Step 4: damaged rows get a full recompute, restoring exact top-k for
    # rows that lost an entry to a deletion.
    if damaged_ids:
        store.delete_neighbors_from(list(damaged_ids))  # clear any leftover rows first
        damaged_rows = sorted(id_to_idx[d] for d in damaged_ids)
        recomputed = _topk_for_rows(
            corpus, ids, damaged_rows, k, block, inv_scale, rank,
            phase="damaged-rows repair",
        )
        store.insert_neighbors(recomputed)
        inserted += len(recomputed)

    store.commit()
    logger.info(
        "k-NN incremental update: N=%d, changed=%d, deleted=%d, damaged=%d, "
        "stale edges removed=%d, edges written=%d.",
        n, len(changed_ids), len(deleted_ids), len(damaged_ids), removed, inserted,
    )
    return inserted
