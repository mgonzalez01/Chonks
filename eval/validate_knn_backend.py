#!/usr/bin/env python
"""eval/validate_knn_backend.py: bit-identical validation of a
CHONKS_KNN_BACKEND device backend (mlx/cuda) against the numpy path.

CI has no GPU, so the unit tests only check code paths against a fake
cupy/mlx; this is the real acceptance gate, run by hand on real hardware.

    uv run python eval/validate_knn_backend.py --db path/to/real_corpus.db --backend cuda

Never mutates --db; rebuilds both a full graph and one incremental
add/delete step on scratch copies and byte-compares chunk_neighbors.
A backend that fails to construct (no CUDA device, missing cupy) is a
FAIL here, never a silent numpy fallback.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chonks.index.graph.knn import build_neighbors  # noqa: E402
from chonks.store import Store  # noqa: E402


def _ordered_edges(store: Store) -> list[tuple[str, str, float]]:
    """chunk_neighbors rows in INSERT order: device paths are expected to
    match numpy row-for-row, not just as a set."""
    return [
        (r["chunk_id"], r["neighbor_id"], r["distance"])
        for r in store._conn.execute(
            "SELECT chunk_id, neighbor_id, distance FROM chunk_neighbors ORDER BY rowid"
        ).fetchall()
    ]


def _compare(label: str, want: list[tuple[str, str, float]], got: list[tuple[str, str, float]]) -> bool:
    want_set = {(c, n): d for c, n, d in want}
    got_set = {(c, n): d for c, n, d in got}

    if set(want_set) != set(got_set):
        missing = set(want_set) - set(got_set)
        extra = set(got_set) - set(want_set)
        print(f"[{label}] FAIL — edge sets differ: {len(missing)} missing, {len(extra)} extra")
        for e in list(missing)[:5]:
            print(f"    missing: {e}")
        for e in list(extra)[:5]:
            print(f"    extra:   {e}")
        return False

    mismatches = [(k, want_set[k], got_set[k]) for k in want_set if want_set[k] != got_set[k]]
    if mismatches:
        print(f"[{label}] FAIL — {len(mismatches)} edges with a mismatched distance (not bit-identical)")
        for k, w, g in mismatches[:5]:
            print(f"    {k}: numpy={w!r} vs device={g!r}")
        return False

    order_differs = want != got
    print(
        f"[{label}] PASS — {len(want_set)} edges, bit-identical to numpy "
        f"({'insert order differs' if order_differs else 'insert order identical'})"
    )
    if order_differs:
        print(
            f"    [{label}] note: edge set + distances match exactly, but row insert "
            "order doesn't. Not fatal — chunk_neighbors.distance is only ever consumed "
            "as a sort key (see DOCS.md) — but the bulk/incremental device paths are "
            "designed to match numpy's insert order too, so this is worth a look."
        )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", required=True, help="Path to a real, already-indexed DB. Never modified.")
    ap.add_argument("--backend", choices=["mlx", "cuda"], required=True)
    ap.add_argument("--k", type=int, default=10, help="k for build_neighbors (default: 10)")
    ap.add_argument(
        "--incremental-batch", type=int, default=200,
        help="Files to delete + synthetic chunks to add for the incremental step (default: 200)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-scratch", action="store_true", help="Don't print a cleanup reminder path (they're always left on disk either way)")
    ap.add_argument(
        "--scratch-dir", default=None,
        help="Reuse this scratch dir across runs: a side whose scratch copy "
             "already carries a complete neighbor graph is NOT rebuilt (the "
             "numpy side can take 20+ min at 350k-chunk scale, do not re-pay it "
             "when only the device side failed, e.g. on missing CUDA DLLs).",
    )
    args = ap.parse_args()

    src = Path(args.db)
    if not src.exists():
        print(f"--db {src} does not exist", file=sys.stderr)
        return 2

    if args.scratch_dir:
        tmpdir = Path(args.scratch_dir)
        tmpdir.mkdir(parents=True, exist_ok=True)
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="chonks_knn_validate_"))
    np_db = tmpdir / "numpy.db"
    dev_db = tmpdir / f"{args.backend}.db"

    def _has_full_graph(path: Path) -> bool:
        """Reusable = scratch copy exists and neighbor count is exactly
        chunks×k; a crashed or partial build can't satisfy that."""
        if not path.exists():
            return False
        s = Store(str(path))
        try:
            n = s.count_chunks()
            return n > 1 and s.count_neighbors() == n * min(args.k, n - 1)
        finally:
            s.close()

    reuse_np = _has_full_graph(np_db)
    if not reuse_np:
        shutil.copy(src, np_db)
    # The device side is always rebuilt fresh; reusing it would defeat the
    # point of the comparison.
    shutil.copy(src, dev_db)
    print(f"scratch copies: {np_db}  {dev_db}")

    ok = True

    print("\n--- full rebuild ---")
    os.environ.pop("CHONKS_KNN_BACKEND", None)
    store_np = Store(str(np_db))
    if reuse_np:
        print("numpy side: reusing completed graph from --scratch-dir (skipped rebuild)")
    else:
        build_neighbors(store_np, k=args.k)
    want_full = _ordered_edges(store_np)
    print(f"numpy: {len(want_full)} edges")

    os.environ["CHONKS_KNN_BACKEND"] = args.backend
    store_dev = Store(str(dev_db))
    try:
        build_neighbors(store_dev, k=args.k)
    except RuntimeError as e:
        print(f"[full rebuild] FAIL — {args.backend} backend raised: {e}")
        return 1
    got_full = _ordered_edges(store_dev)
    print(f"{args.backend}: {len(got_full)} edges")
    ok = _compare("full rebuild", want_full, got_full) and ok

    print("\n--- incremental step ---")
    rng = random.Random(args.seed)
    paths = [r[0] for r in store_np._conn.execute("SELECT DISTINCT path FROM chunks").fetchall()]
    batch_n = args.incremental_batch
    if len(paths) < batch_n:
        batch_n = max(1, len(paths) // 10)
        print(f"corpus has only {len(paths)} distinct paths — reducing batch to {batch_n}")
    delete_paths = rng.sample(paths, batch_n)

    ids, blob, dim = store_np.get_all_int8_embeddings()
    np_rng = np.random.default_rng(args.seed)
    new_vecs = np_rng.standard_normal((batch_n, dim)).astype(np.float64)
    new_vecs /= np.linalg.norm(new_vecs, axis=1, keepdims=True)
    new_ids = [f"__validate_synthetic_{i}__" for i in range(batch_n)]
    new_chunks = [
        {
            "id": nid, "path": f"__validate_synthetic_{i}__.py",
            "content": f"def _validate_synthetic_{i}(): pass",
        }
        for i, nid in enumerate(new_ids)
    ]
    changed_ids = set(new_ids)

    # Apply the IDENTICAL mutation (same delete_paths, same synthetic
    # chunks/vectors) to both scratch DBs so the two backends update from
    # the same pre- and post-mutation corpus.
    os.environ.pop("CHONKS_KNN_BACKEND", None)
    deleted_ids_np: set[str] = set()
    for p in delete_paths:
        deleted_ids_np.update(store_np.delete_file(p))
    store_np.insert_chunks(new_chunks, new_vecs.tolist())
    build_neighbors(store_np, k=args.k, changed_ids=changed_ids, deleted_ids=deleted_ids_np)
    want_incr = _ordered_edges(store_np)
    print(f"numpy: {len(want_incr)} edges after +{len(new_ids)}/-{len(deleted_ids_np)}")

    os.environ["CHONKS_KNN_BACKEND"] = args.backend
    deleted_ids_dev: set[str] = set()
    for p in delete_paths:
        deleted_ids_dev.update(store_dev.delete_file(p))
    store_dev.insert_chunks(new_chunks, new_vecs.tolist())
    try:
        build_neighbors(store_dev, k=args.k, changed_ids=changed_ids, deleted_ids=deleted_ids_dev)
    except RuntimeError as e:
        print(f"[incremental] FAIL — {args.backend} backend raised: {e}")
        return 1
    got_incr = _ordered_edges(store_dev)
    print(f"{args.backend}: {len(got_incr)} edges after +{len(new_ids)}/-{len(deleted_ids_dev)}")
    ok = _compare("incremental step", want_incr, got_incr) and ok

    print("\n" + "=" * 60)
    print("OVERALL:", "PASS" if ok else "FAIL")
    print("=" * 60)
    print(f"scratch DBs left at {tmpdir} for inspection (not auto-deleted)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
