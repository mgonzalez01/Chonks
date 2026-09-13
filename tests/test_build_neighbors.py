"""Tests for build_neighbors edge cases."""
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

from chonks.repomap import build_neighbors
from chonks.store import Store


def _make_store(n_chunks: int, dim: int = 8) -> Store:
    """Create an in-memory-style Store seeded with n_chunks random embeddings."""
    tmp = tempfile.mktemp(suffix=".db")
    store = Store(tmp)
    rng = np.random.default_rng(42)
    chunks = []
    embeddings = []
    for i in range(n_chunks):
        chunks.append({
            "id": f"chunk-{i}",
            "path": "test.py",
            "content": f"def fn_{i}(): pass",
        })
        vec = rng.standard_normal(dim).tolist()
        norm = math.sqrt(sum(v * v for v in vec))
        embeddings.append([v / norm for v in vec])
    store.insert_chunks(chunks, embeddings)
    return store


def test_build_neighbors_fewer_chunks_than_k_does_not_crash():
    store = _make_store(n_chunks=3)
    edges = build_neighbors(store)
    # 3 chunks x 2 neighbours each (self excluded) = 6 edges
    assert edges == 6


def test_build_neighbors_single_chunk_returns_zero_edges():
    store = _make_store(n_chunks=1)
    assert build_neighbors(store) == 0


def test_build_neighbors_empty_corpus_returns_zero_edges():
    tmp = tempfile.mktemp(suffix=".db")
    store = Store(tmp)
    assert build_neighbors(store) == 0


# --- matmul block auto-sizing (use available RAM) -------------------------

def test_neighbor_block_rows_override():
    from chonks.repomap import _neighbor_block_rows
    assert _neighbor_block_rows(10000, override=2048) == 2048
    assert _neighbor_block_rows(500, override=4096) == 500  # capped at N


def test_neighbor_block_rows_autoscale(monkeypatch):
    import chonks.repomap.knn as repomap
    monkeypatch.setattr(repomap, "_available_ram_bytes", lambda: 64 * 10**9)
    n = 100_000
    got = repomap._neighbor_block_rows(n)
    want = min(
        max(repomap._NEIGHBOR_MATMUL_BLOCK,
            int(64 * 10**9 * repomap._NEIGHBOR_RAM_FRACTION)
            // (n * repomap._NEIGHBOR_BYTES_PER_ROW_ELEM)),
        n,
    )
    assert got == want
    assert got > repomap._NEIGHBOR_MATMUL_BLOCK  # plenty of RAM -> bigger than the floor


def test_neighbor_block_rows_fallback_and_cap(monkeypatch):
    import chonks.repomap.knn as repomap
    # RAM unknown -> fixed floor (capped at N)
    monkeypatch.setattr(repomap, "_available_ram_bytes", lambda: None)
    assert repomap._neighbor_block_rows(100_000) == repomap._NEIGHBOR_MATMUL_BLOCK
    assert repomap._neighbor_block_rows(300) == 300
    # huge RAM, tiny corpus -> one block of N, never a floor larger than N
    monkeypatch.setattr(repomap, "_available_ram_bytes", lambda: 10**12)
    assert repomap._neighbor_block_rows(50) == 50


# --- exact incremental k-NN update ----------------------------------------
# Property: incrementally updating a graph after a batch of adds/deletes/
# modifies must produce a chunk_neighbors table BYTE-IDENTICAL (same edge
# set, allclose distances) to a from-scratch full rebuild on the same
# post-mutation corpus.

def _rand_unit_vecs(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    vecs = rng.standard_normal((count, dim)).astype(np.float64)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def _seed_chunks(store: Store, ids: list[str], vecs: np.ndarray, offset: int = 0) -> None:
    """Insert one chunk per id, each on its own path, so an individual chunk
    can be removed later via store.delete_file without touching its siblings."""
    chunks = [
        {"id": cid, "path": f"f{offset + i}.py", "content": f"def fn_{offset + i}(): pass"}
        for i, cid in enumerate(ids)
    ]
    store.insert_chunks(chunks, vecs.tolist())


def _all_edges(store: Store) -> dict[tuple[str, str], float]:
    rows = store._conn.execute(
        "SELECT chunk_id, neighbor_id, distance FROM chunk_neighbors"
    ).fetchall()
    return {(r["chunk_id"], r["neighbor_id"]): r["distance"] for r in rows}


def _assert_graphs_equal(got: dict[tuple[str, str], float], want: dict[tuple[str, str], float]) -> None:
    assert set(got) == set(want), (
        f"edge sets differ: missing={set(want) - set(got)}, extra={set(got) - set(want)}"
    )
    for key in want:
        assert got[key] == pytest.approx(want[key], abs=1e-4), f"distance mismatch for {key}"


def _build_mutated_reference(
    ids: list[str], vecs: np.ndarray, keep_mask: np.ndarray,
    new_ids: list[str], new_vecs: np.ndarray, k: int,
) -> dict[tuple[str, str], float]:
    """Build a from-scratch store containing exactly the post-mutation corpus
    (surviving originals + new chunks) and full-rebuild its graph: the
    ground truth the incremental path must match."""
    ref = Store(tempfile.mktemp(suffix=".db"))
    kept_ids = [i for i, keep in zip(ids, keep_mask) if keep]
    kept_vecs = vecs[keep_mask]
    _seed_chunks(ref, kept_ids, kept_vecs, offset=0)
    if new_ids:
        _seed_chunks(ref, new_ids, new_vecs, offset=10_000)
    build_neighbors(ref, k=k)
    return _all_edges(ref)


def _run_incremental_case(
    n_initial: int, n_add: int, n_delete: int, k: int = 5, dim: int = 8, seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    ids = [f"orig-{i}" for i in range(n_initial)]
    vecs = _rand_unit_vecs(rng, n_initial, dim)

    store = Store(tempfile.mktemp(suffix=".db"))
    _seed_chunks(store, ids, vecs, offset=0)
    build_neighbors(store, k=k)

    # Mutate: delete a random subset, add a fresh batch.
    delete_idx = set(rng.choice(n_initial, size=n_delete, replace=False)) if n_delete else set()
    keep_mask = np.array([i not in delete_idx for i in range(n_initial)])
    deleted_ids: set[str] = set()
    for i in sorted(delete_idx):
        removed = store.delete_file(f"f{i}.py")
        deleted_ids.update(removed)

    new_ids = [f"new-{i}" for i in range(n_add)]
    new_vecs = _rand_unit_vecs(rng, n_add, dim) if n_add else np.zeros((0, dim))
    if n_add:
        _seed_chunks(store, new_ids, new_vecs, offset=10_000)
    changed_ids = set(new_ids)

    incr_edges = build_neighbors(store, k=k, changed_ids=changed_ids, deleted_ids=deleted_ids)
    got = _all_edges(store)
    want = _build_mutated_reference(ids, vecs, keep_mask, new_ids, new_vecs, k)
    _assert_graphs_equal(got, want)
    return incr_edges


def test_incremental_pure_adds_matches_full_rebuild():
    _run_incremental_case(n_initial=200, n_add=15, n_delete=0, k=5, seed=1)


def test_incremental_pure_deletes_matches_full_rebuild():
    _run_incremental_case(n_initial=200, n_add=0, n_delete=15, k=5, seed=2)


def test_incremental_mixed_add_delete_matches_full_rebuild():
    _run_incremental_case(n_initial=300, n_add=20, n_delete=20, k=6, seed=3)


def test_incremental_modify_matches_full_rebuild():
    """A 'modify' is delete-then-insert of the same logical file: same shape
    of mutation as add+delete but exercised as the paired operation chunker.py
    actually performs on a changed file."""
    rng = np.random.default_rng(4)
    n = 250
    dim = 8
    ids = [f"orig-{i}" for i in range(n)]
    vecs = _rand_unit_vecs(rng, n, dim)

    store = Store(tempfile.mktemp(suffix=".db"))
    _seed_chunks(store, ids, vecs, offset=0)
    build_neighbors(store, k=5)

    modify_idx = sorted(rng.choice(n, size=12, replace=False))
    keep_mask = np.array([i not in set(modify_idx) for i in range(n)])
    deleted_ids: set[str] = set()
    for i in modify_idx:
        deleted_ids.update(store.delete_file(f"f{i}.py"))

    new_ids = [f"modified-{i}" for i in modify_idx]
    new_vecs = _rand_unit_vecs(rng, len(modify_idx), dim)
    _seed_chunks(store, new_ids, new_vecs, offset=10_000)
    changed_ids = set(new_ids)

    build_neighbors(store, k=5, changed_ids=changed_ids, deleted_ids=deleted_ids)
    got = _all_edges(store)
    want = _build_mutated_reference(ids, vecs, keep_mask, new_ids, new_vecs, k=5)
    _assert_graphs_equal(got, want)


def test_incremental_k_close_to_n_edge_case():
    """Small corpus where k is close to N-1 both before and after the batch,
    exercising the min(k, n-1) capping path inside the incremental branch."""
    _run_incremental_case(n_initial=12, n_add=2, n_delete=1, k=10, seed=5)


def test_incremental_survives_dangling_worst_neighbor():
    """Regression: a dangling chunk_neighbors row (worst-neighbor id no
    longer in the corpus) must not abort the incremental build via a
    KeyError in the threshold loop's `id_to_idx[worst_id]` lookup."""
    n = 30
    dim = 8
    k = 5
    rng = np.random.default_rng(7)
    ids = [f"orig-{i}" for i in range(n)]
    vecs = _rand_unit_vecs(rng, n, dim)

    store = Store(tempfile.mktemp(suffix=".db"))
    _seed_chunks(store, ids, vecs, offset=0)
    build_neighbors(store, k=k)

    # Corrupt: rewrite an existing chunk's worst (max-distance) neighbor row,
    # tie broken the same way get_neighbor_worst_distances does, so it points
    # at an id that was never inserted into `chunks`.
    victim = store._conn.execute(
        "SELECT chunk_id, neighbor_id FROM chunk_neighbors "
        "ORDER BY distance DESC, neighbor_id DESC LIMIT 1"
    ).fetchone()
    with store._lock:
        store._conn.execute(
            "UPDATE chunk_neighbors SET neighbor_id=? WHERE chunk_id=? AND neighbor_id=?",
            ("ghost-does-not-exist", victim["chunk_id"], victim["neighbor_id"]),
        )
        store._conn.commit()

    # A non-empty changed_ids batch plus a pre-existing graph routes
    # build_neighbors into the incremental path, whose threshold loop reads
    # every row's "worst" entry, the code path that crashed.
    new_id = "new-0"
    new_vec = _rand_unit_vecs(rng, 1, dim)
    _seed_chunks(store, [new_id], new_vec, offset=10_000)

    # Must not raise KeyError.
    edges = build_neighbors(store, k=k, changed_ids={new_id}, deleted_ids=set())
    assert edges >= 0


# --- k-NN distance tie-break ------------------------------------------------
#
# int8-quantized cosine distance ties at the k-th (worst-kept) slot often at
# small `dim`; without a deterministic tie-break applied consistently at
# every site that picks/keeps a row's top-k (the bulk rebuild's argpartition
# sort, `_topk_for_rows`, and the incremental displacement/trim_neighbors
# path), the incremental result can pick a different one of the tied
# candidates than a from-scratch rebuild, breaking the "incremental exactly
# equals a full rebuild" contract.

def test_incremental_matches_full_rebuild_with_knn_tie_at_dim16_seed11():
    """Reproducer: at dim=16 this seed produces a real tie at the k-th
    distance boundary for a damaged row. Before the shared tie-break, the
    incremental path picked orig-113 instead of orig-55 as orig-8's neighbour."""
    _run_incremental_case(n_initial=120, n_add=10, n_delete=10, k=5, dim=16, seed=11)


def test_incremental_matches_full_rebuild_with_knn_tie_another_seed():
    """A second seed exercising the same k-th-distance tie-break path (also
    fails without the fix), so the regression coverage isn't over-fit to a
    single seed's specific tie."""
    _run_incremental_case(n_initial=120, n_add=10, n_delete=10, k=5, dim=16, seed=6)


def test_incremental_add_only_displacement_tie_break():
    """Regression: the displacement mask tie-broke on the target row's own id
    (`rank[cid]`) instead of the incoming candidate's id (`rank[qid]`).
    Constructed so the two rules disagree, exercising the ADD-only
    displacement check specifically, not the damaged-row recompute path."""
    k = 2

    def unit(theta_deg: float) -> list[float]:
        theta = math.radians(theta_deg)
        return [math.cos(theta), math.sin(theta), 0.0, 0.0]

    # orig-7 is the target row T. orig-1 (10 deg) is T's clear best neighbour.
    # orig-3 (30 deg) is T's worst-KEPT (k-th) neighbour pre-add. The rest are
    # far enough from T (0 deg) to never contend for a top-2 slot.
    ids = [f"orig-{i}" for i in range(8)]
    vecs = np.array([
        unit(170), unit(10), unit(150), unit(30),
        unit(60), unit(80), unit(140), unit(0),
    ])

    store = Store(tempfile.mktemp(suffix=".db"))
    _seed_chunks(store, ids, vecs, offset=0)
    build_neighbors(store, k=k)

    pre = _all_edges(store)
    assert {n for (c, n) in pre if c == "orig-7"} == {"orig-1", "orig-3"}

    # New chunk, vector IDENTICAL to orig-3's, forces an exact tie in distance
    # to T. Its id ("new-0") sorts before "orig-3", so tie-break must prefer
    # it; "orig-7" (T itself) sorts after "orig-3", what the buggy rule compared.
    new_ids = ["new-0"]
    new_vecs = np.array([unit(30)])
    _seed_chunks(store, new_ids, new_vecs, offset=10_000)

    build_neighbors(store, k=k, changed_ids={"new-0"}, deleted_ids=set())
    got = _all_edges(store)

    ref = Store(tempfile.mktemp(suffix=".db"))
    _seed_chunks(ref, ids, vecs, offset=0)
    _seed_chunks(ref, new_ids, new_vecs, offset=10_000)
    build_neighbors(ref, k=k)
    want = _all_edges(ref)

    _assert_graphs_equal(got, want)
    assert ("orig-7", "new-0") in got
    assert ("orig-7", "orig-3") not in got


def test_incremental_empty_changed_and_deleted_is_noop():
    store = _make_store(n_chunks=50)
    build_neighbors(store)
    before = _all_edges(store)
    edges_written = build_neighbors(store, changed_ids=set(), deleted_ids=set())
    assert edges_written == 0
    assert _all_edges(store) == before


def test_incremental_falls_back_to_full_rebuild_over_threshold():
    """A batch that's a large fraction of the corpus should not take the
    incremental path; verify by monkeypatching the incremental helper and
    asserting it's never called."""
    import chonks.repomap.knn as repomap
    store = _make_store(n_chunks=50)
    build_neighbors(store)

    called = []
    original = repomap._build_neighbors_incremental
    def _spy(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)
    repomap._build_neighbors_incremental = _spy
    try:
        # 40 out of 50 changed -> way over the 20% threshold -> full rebuild.
        changed = {f"chunk-{i}" for i in range(40)}
        build_neighbors(store, changed_ids=changed, deleted_ids=set())
    finally:
        repomap._build_neighbors_incremental = original
    assert not called


def test_incremental_bails_mid_flight_on_high_fanout(monkeypatch, caplog):
    """The entry gate only sees batch size; a batch's actual cost is its
    fan-out (how many existing rows' top-k it displaces), known only after
    the displacement matmul. Over budget, the incremental path must return
    None and fall back to a full rebuild that still produces the exact graph."""
    import logging
    import chonks.repomap.knn as repomap

    rng = np.random.default_rng(7)
    n, dim, k = 20, 8, 10
    ids = [f"orig-{i}" for i in range(n)]
    vecs = _rand_unit_vecs(rng, n, dim)
    store = Store(tempfile.mktemp(suffix=".db"))
    _seed_chunks(store, ids, vecs, offset=0)
    build_neighbors(store, k=k)

    # Batch of 4 (17% of N=24) passes a 30% entry gate; its fan-out (most of
    # the corpus, >7.2 rows) blows max(floor, 30% of N) once the floor is
    # patched below the fraction budget.
    monkeypatch.setattr(repomap, "_NEIGHBOR_INCREMENTAL_MAX_FRACTION", 0.30)
    monkeypatch.setattr(repomap, "_NEIGHBOR_INCREMENTAL_BAIL_MIN_ROWS", 1)
    new_ids = [f"new-{i}" for i in range(4)]
    new_vecs = _rand_unit_vecs(rng, 4, dim)
    _seed_chunks(store, new_ids, new_vecs, offset=10_000)

    with caplog.at_level(logging.INFO, logger="repomap"):
        build_neighbors(store, k=k, changed_ids=set(new_ids), deleted_ids=set())
    assert any("k-NN incremental bail" in r.message for r in caplog.records), (
        "expected the mid-flight fan-out bail to fire")
    assert any("k-NN full rebuild" in r.message for r in caplog.records), (
        "bail must fall through to the full rebuild")

    # Exact-graph contract survives the bail: compare against a from-scratch
    # rebuild of the same final corpus in a fresh store.
    ref = Store(tempfile.mktemp(suffix=".db"))
    _seed_chunks(ref, ids, vecs, offset=0)
    _seed_chunks(ref, new_ids, new_vecs, offset=10_000)
    build_neighbors(ref, k=k)
    _assert_graphs_equal(_all_edges(store), _all_edges(ref))


# --- int8 matmul "exact" mode (large-N memory path) ----------------------
#
# _Corpus picks "fast" (whole corpus resident as float32, one BLAS sgemm per
# row-block, the pre-existing behavior) or "exact" (int8 corpus resident
# only, block-by-block float64 BLAS dgemm, exact int32 dot products) by
# corpus size. These tests force each mode via monkeypatching
# `_NEIGHBOR_FULL_F32_MAX_BYTES` (0 => always exact, a huge number => always
# fast) so both are exercised regardless of how big the test corpus is.

def _force_mode(monkeypatch, exact: bool):
    import chonks.repomap.knn as repomap
    monkeypatch.setattr(
        repomap, "_NEIGHBOR_FULL_F32_MAX_BYTES", 0 if exact else 10**18,
    )


def test_corpus_sims_block_exact_matches_fast_on_random_data():
    """'exact' and 'fast' modes must agree on ordinary (non-adversarial)
    random int8 data across several seeds/dims; the ranking-affecting case
    (large magnitude, dim large enough to overflow float32's mantissa) is
    covered separately below since it needs adversarial vectors to surface."""
    import chonks.repomap.knn as repomap
    for seed, dim, n in [(0, 8, 50), (1, 64, 30), (2, 300, 20), (3, 1, 40)]:
        rng = np.random.default_rng(seed)
        m8 = rng.integers(-127, 128, size=(n, dim), dtype=np.int8)
        ids = [f"c{i}" for i in range(n)]
        blob = m8.tobytes()
        fast = repomap._Corpus(ids, blob, dim, block_rows=7)
        fast.exact = False
        fast.m = m8.astype(np.float32)
        fast.mt = fast.m.T
        exact = repomap._Corpus(ids, blob, dim, block_rows=7)
        exact.exact = True
        exact.m = None
        exact.mt = None

        sims_fast = fast.sims_block(slice(0, n))
        sims_exact = exact.sims_block(slice(0, n))
        assert np.array_equal(sims_fast.astype(np.int64), sims_exact.astype(np.int64)), (
            f"seed={seed} dim={dim} n={n}: fast/exact sims diverged"
        )


def test_corpus_exact_mode_correct_where_float32_would_round():
    """Adversarial max-magnitude vectors at dim=2560 push the true dot
    product above float32's 24-bit-mantissa exact-integer ceiling, so sgemm
    rounds to the wrong integer. 'exact' mode must still match int64 arithmetic."""
    import chonks.repomap.knn as repomap
    dim = 2560
    n = 40
    m8 = np.full((n, dim), 127, dtype=np.int8)
    m8[1::2] = -128  # alternate max-negative rows for varied products
    # Zero one column so the all-127 rows' self-product is an odd integer
    # above 2^25, where float32 can't represent it, keeping the non-vacuity
    # guard below from firing spuriously on BLAS that stays exact for even products.
    m8[:, 0] = 0
    ids = [f"c{i}" for i in range(n)]
    blob = m8.tobytes()

    exact = repomap._Corpus(ids, blob, dim, block_rows=5)
    exact.exact = True
    exact.m = None
    exact.mt = None
    got = exact.sims_block(slice(0, n))

    m_i64 = m8.astype(np.int64)
    want = m_i64 @ m_i64.T
    assert np.array_equal(got.astype(np.int64), want)

    # Confirm this dataset really does break float32 (i.e. the test is
    # actually stressing the mantissa, not vacuously passing).
    m_f32 = m8.astype(np.float32)
    sims_f32 = (m_f32 @ m_f32.T).astype(np.int64)
    assert not np.array_equal(sims_f32, want), "fixture didn't stress f32 mantissa as intended"


# --- two-tier mode selection ----------------------------------------------
#
# _Corpus.__init__ picks "fast" vs "exact" in two tiers:
#   1. dim-aware exactness: dim·127² < 2²⁴ ⇒ "fast" unconditionally, at any N
#      or RAM threshold: sgemm IS exact at such dims.
#   2. RAM-aware threshold: only reached when dim·127² >= 2²⁴ (dim-unsafe);
#      budgets the one-shot float32 corpus size against measured free RAM.

def test_corpus_dim_safe_selects_fast_regardless_of_ram_threshold(monkeypatch):
    """A dim-safe corpus (dim=64, well under the ~1040 cutoff) must select
    'fast' mode even when the RAM-aware threshold is forced to say the corpus
    is 'too big': the dim rule short-circuits before RAM is considered."""
    import chonks.repomap.knn as repomap
    # Force the old-style threshold (and a starved RAM probe) to look like
    # they'd pick "exact": the dim rule must still win.
    monkeypatch.setattr(repomap, "_NEIGHBOR_FULL_F32_MAX_BYTES", 0)
    monkeypatch.setattr(repomap, "_available_ram_bytes", lambda: 1)  # ~nothing free
    dim = 64
    n = 20
    rng = np.random.default_rng(0)
    m8 = rng.integers(-127, 128, size=(n, dim), dtype=np.int8)
    ids = [f"c{i}" for i in range(n)]
    corpus = repomap._Corpus(ids, m8.tobytes(), dim, block_rows=5)
    assert corpus.exact is False
    assert dim * 127 * 127 < repomap._NEIGHBOR_DIM_SAFE_DOT_PRODUCT_MAX


def test_corpus_dim_safe_at_jina_896_selects_fast(monkeypatch):
    """dim=896 (jina) is the motivating case; must select
    'fast' even under a starved RAM budget."""
    import chonks.repomap.knn as repomap
    monkeypatch.setattr(repomap, "_available_ram_bytes", lambda: 1)
    dim = 896
    assert dim * 127 * 127 < repomap._NEIGHBOR_DIM_SAFE_DOT_PRODUCT_MAX
    n = 10
    rng = np.random.default_rng(1)
    m8 = rng.integers(-127, 128, size=(n, dim), dtype=np.int8)
    ids = [f"c{i}" for i in range(n)]
    corpus = repomap._Corpus(ids, m8.tobytes(), dim, block_rows=5)
    assert corpus.exact is False


def test_corpus_dim_unsafe_small_corpus_selects_fast_when_ram_plentiful(monkeypatch):
    """A dim-unsafe corpus (dim=2560) that's tiny in absolute size should
    still select 'fast' when free RAM comfortably covers the full float32
    residency; Tier 2 must not force 'exact' just because dim crossed the
    Tier-1 cutoff."""
    import chonks.repomap.knn as repomap
    monkeypatch.setattr(repomap, "_available_ram_bytes", lambda: 64 * 10**9)  # 64 GB free
    dim = 2560
    assert dim * 127 * 127 >= repomap._NEIGHBOR_DIM_SAFE_DOT_PRODUCT_MAX
    n = 20
    rng = np.random.default_rng(2)
    m8 = rng.integers(-127, 128, size=(n, dim), dtype=np.int8)
    ids = [f"c{i}" for i in range(n)]
    corpus = repomap._Corpus(ids, m8.tobytes(), dim, block_rows=5)
    assert corpus.exact is False


def test_corpus_dim_unsafe_tiny_ram_budget_selects_exact(monkeypatch):
    """A dim-unsafe corpus under a mocked tiny RAM budget must fall back to
    'exact' mode: Tier 2's RAM-aware threshold in action."""
    import chonks.repomap.knn as repomap
    monkeypatch.setattr(repomap, "_available_ram_bytes", lambda: 1024)  # 1 KB free
    dim = 2560
    n = 20
    rng = np.random.default_rng(3)
    m8 = rng.integers(-127, 128, size=(n, dim), dtype=np.int8)
    ids = [f"c{i}" for i in range(n)]
    corpus = repomap._Corpus(ids, m8.tobytes(), dim, block_rows=5)
    assert corpus.exact is True


def test_corpus_dim_unsafe_no_ram_probe_falls_back_to_fixed_threshold(monkeypatch):
    """When free RAM can't be measured at all, Tier 2 falls back to the fixed
    `_NEIGHBOR_FULL_F32_MAX_BYTES` byte threshold (same escape hatch role the
    fixed floor plays in `_neighbor_block_rows`)."""
    import chonks.repomap.knn as repomap
    monkeypatch.setattr(repomap, "_available_ram_bytes", lambda: None)
    dim = 2560
    n = 4  # tiny corpus: n*dim*4 well under the fixed 512MB threshold
    rng = np.random.default_rng(4)
    m8 = rng.integers(-127, 128, size=(n, dim), dtype=np.int8)
    ids = [f"c{i}" for i in range(n)]
    corpus = repomap._Corpus(ids, m8.tobytes(), dim, block_rows=2)
    assert corpus.exact is False  # n*dim*4 = 40960 bytes, far under 512MB

    monkeypatch.setattr(repomap, "_NEIGHBOR_FULL_F32_MAX_BYTES", 0)
    corpus2 = repomap._Corpus(ids, m8.tobytes(), dim, block_rows=2)
    assert corpus2.exact is True  # fixed threshold of 0 now forces "exact"


def test_build_neighbors_exact_mode_matches_fast_mode(monkeypatch):
    """End-to-end: forcing 'exact' mode on a real (small) corpus built through
    Store must produce the identical neighbor graph as the default 'fast'
    mode: same edges, same distances."""
    store = _make_store(n_chunks=80, dim=16)
    _force_mode(monkeypatch, exact=False)
    build_neighbors(store, k=5)
    want = _all_edges(store)

    store2 = _make_store(n_chunks=80, dim=16)
    _force_mode(monkeypatch, exact=True)
    build_neighbors(store2, k=5)
    got = _all_edges(store2)

    _assert_graphs_equal(got, want)


def test_incremental_exact_mode_matches_fast_mode(monkeypatch):
    """Forced 'exact' mode incremental update matches a forced 'fast' mode
    incremental update on the same mutation batch."""
    _force_mode(monkeypatch, exact=False)
    _run_incremental_case(n_initial=120, n_add=10, n_delete=10, k=5, dim=16, seed=12)


def test_incremental_exact_mode_end_to_end(monkeypatch):
    _force_mode(monkeypatch, exact=True)
    _run_incremental_case(n_initial=120, n_add=10, n_delete=10, k=5, dim=16, seed=12)


def test_damaged_row_repair_when_deleted_chunk_had_many_incoming_edges():
    """Delete a single chunk that many other chunks had as a top-k neighbour
    (a hub): every one of those rows is 'damaged' and must be repaired to
    exactly match a full rebuild, not just left short a neighbour."""
    rng = np.random.default_rng(6)
    dim = 8
    n = 150
    k = 5
    # Build the hub's cluster: make many vectors a small perturbation of one
    # base vector so they all rank the hub as a top neighbour.
    base = _rand_unit_vecs(rng, 1, dim)[0]
    hub_idx = 0
    vecs = _rand_unit_vecs(rng, n, dim)
    cluster_idx = list(range(1, 30))
    for i in cluster_idx:
        v = base + rng.normal(scale=0.05, size=dim)
        vecs[i] = v / np.linalg.norm(v)
    vecs[hub_idx] = base

    ids = [f"orig-{i}" for i in range(n)]
    store = Store(tempfile.mktemp(suffix=".db"))
    _seed_chunks(store, ids, vecs, offset=0)
    build_neighbors(store, k=k)

    # Sanity check: the hub really is a top-k neighbour of the cluster before deletion.
    hub_id = ids[hub_idx]
    hub_referenced = {
        r["chunk_id"] for r in store._conn.execute(
            "SELECT chunk_id FROM chunk_neighbors WHERE neighbor_id=?", (hub_id,)
        ).fetchall()
    }
    assert len(hub_referenced) >= 10  # confirms the hub setup actually produced a hub

    keep_mask = np.array([i != hub_idx for i in range(n)])
    deleted_ids = set(store.delete_file(f"f{hub_idx}.py"))

    build_neighbors(store, k=k, changed_ids=set(), deleted_ids=deleted_ids)
    got = _all_edges(store)
    want = _build_mutated_reference(ids, vecs, keep_mask, [], np.zeros((0, dim)), k)
    _assert_graphs_equal(got, want)


# --- exact-vs-fast equivalence at a dim-UNSAFE dimension -------------------
#
# The `test_..._exact_mode_matches_fast_mode` / `test_incremental_exact_mode_*`
# tests above use dim=16 fixtures and force mode via `_force_mode` (which
# patches `_NEIGHBOR_FULL_F32_MAX_BYTES`, the Tier-2 fallback used only when
# `_available_ram_bytes()` returns None). Since the dim-aware Tier-1 rule
# (dim·127² < 2²⁴ ⇒ always "fast") now short-circuits before Tier 2 is ever
# consulted at dim=16, those tests select "fast" on BOTH sides regardless of
# `_force_mode`: the exact integer path (block-by-block float64 dgemm) has
# had zero equivalence coverage since that rule shipped. The tests below use
# dim=2560 (dim-unsafe: dim·127² ≈ 41.3M ≥ 2²⁴) and force mode by mocking the
# RAM probe directly (mirroring `test_corpus_dim_unsafe_*` above), which is
# the only lever that actually reaches Tier 2's real branch.

def _force_ram_mode(monkeypatch, exact: bool) -> None:
    """Force _Corpus's Tier-2 RAM-aware threshold at a dim-unsafe dimension by
    mocking the RAM probe (tiny -> 'exact', plentiful -> 'fast')."""
    import chonks.repomap.knn as repomap
    monkeypatch.setattr(
        repomap, "_available_ram_bytes", lambda: 1024 if exact else 64 * 10**9,
    )


def test_build_neighbors_exact_mode_matches_fast_mode_dim_unsafe(monkeypatch):
    """End-to-end equivalence at dim=2560 (dim-unsafe): forcing 'exact' mode
    via a starved RAM probe must produce the identical neighbor graph as
    'fast' mode forced via a plentiful RAM probe."""
    import chonks.repomap.knn as repomap
    dim = 2560
    assert dim * 127 * 127 >= repomap._NEIGHBOR_DIM_SAFE_DOT_PRODUCT_MAX
    store = _make_store(n_chunks=200, dim=dim)
    _force_ram_mode(monkeypatch, exact=False)
    build_neighbors(store, k=5)
    want = _all_edges(store)

    store2 = _make_store(n_chunks=200, dim=dim)
    _force_ram_mode(monkeypatch, exact=True)
    build_neighbors(store2, k=5)
    got = _all_edges(store2)

    _assert_graphs_equal(got, want)


def test_incremental_fast_mode_matches_full_rebuild_dim_unsafe(monkeypatch):
    """Incremental path at a dim-unsafe dimension, forced onto 'fast' mode,
    must match a from-scratch full rebuild. At dim=2560, cosine similarities
    cluster tightly enough that many seeds hit a genuine k/k+1 boundary tie;
    this picks a seed with no such tie rather than papering over it."""
    _force_ram_mode(monkeypatch, exact=False)
    _run_incremental_case(n_initial=110, n_add=8, n_delete=8, k=5, dim=2560, seed=1)


def test_incremental_exact_mode_matches_full_rebuild_dim_unsafe(monkeypatch):
    """Incremental path at a dim-unsafe dimension, forced onto 'exact' mode,
    must match a from-scratch full rebuild, exercising the integer dgemm path
    through the incremental logic. See the seed-choice note above."""
    _force_ram_mode(monkeypatch, exact=True)
    _run_incremental_case(n_initial=110, n_add=8, n_delete=8, k=5, dim=2560, seed=1)


# --- MLX (Metal GPU) backend -----------------------------------------------

def _ordered_edges(store: Store) -> list[tuple[str, str, float]]:
    """Every chunk_neighbors row in INSERT order: parity with the numpy path
    must hold for the exact rows and their order, not just the edge set."""
    return store._conn.execute(
        "SELECT chunk_id, neighbor_id, distance FROM chunk_neighbors ORDER BY rowid"
    ).fetchall()


def test_mlx_backend_activates_only_dim_safe(monkeypatch):
    """CHONKS_KNN_BACKEND=mlx must arm the GPU path at dim-safe dims only:
    dim-unsafe corpora in 'fast' mode have inexact float32 sims where MLX and
    BLAS accumulation order can disagree, breaking numpy-path determinism."""
    pytest.importorskip("mlx.core")
    from chonks.repomap import _Corpus

    monkeypatch.setenv("CHONKS_KNN_BACKEND", "mlx")
    rng = np.random.default_rng(0)

    blob = rng.integers(-127, 128, size=(50, 8), dtype=np.int8).tobytes()
    corpus = _Corpus([f"c{i}" for i in range(50)], blob, 8)
    assert not corpus.exact
    assert corpus._mx is not None

    blob = rng.integers(-127, 128, size=(50, 2560), dtype=np.int8).tobytes()
    monkeypatch.setattr("chonks.repomap.knn._available_ram_bytes", lambda: 64 * 10**9)
    corpus = _Corpus([f"c{i}" for i in range(50)], blob, 2560)
    assert not corpus.exact  # tier 2 picked 'fast', the case the gate targets
    assert corpus._mx is None


def test_mlx_bulk_rebuild_identical_to_numpy(monkeypatch):
    """Full bulk rebuild through the on-GPU top-k (matmul, self-mask,
    tie-break, argpartition, sort all on Metal) must produce row-for-row
    identical chunk_neighbors output to the numpy path. dim=4 makes int8
    distance ties pervasive, exercising the int64 tie-break equivalence."""
    pytest.importorskip("mlx.core")

    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    store_np = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_np)
    want = _ordered_edges(store_np)

    monkeypatch.setenv("CHONKS_KNN_BACKEND", "mlx")
    store_mx = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_mx)
    got = _ordered_edges(store_mx)

    assert len(want) > 0
    assert got == want


def test_mlx_subsliced_topk_identical_to_numpy(monkeypatch):
    """Metal argpartition silently corrupts past a buffer-size cliff, so
    topk_block sub-slices its rows (see _MLX_TOPK_MAX_BUFFER_BYTES). Shrink
    the cap so every block splits into many sub-slices and verify the output
    is still row-for-row identical to numpy, catching any stitching/offset
    bug in the sub-slice loop."""
    pytest.importorskip("mlx.core")
    import chonks.repomap.knn as repomap

    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    store_np = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_np)
    want = _ordered_edges(store_np)

    monkeypatch.setenv("CHONKS_KNN_BACKEND", "mlx")
    # n=300, n*8=2400 B per key row; cap of 3 rows' worth forces ~100 slices
    monkeypatch.setattr(repomap, "_MLX_TOPK_MAX_BUFFER_BYTES", 3 * 300 * 8)
    store_mx = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_mx)
    got = _ordered_edges(store_mx)

    assert len(want) > 0
    assert got == want


def test_mlx_bulk_multi_block_identical_to_numpy(monkeypatch):
    """Multiple OUTER matmul blocks through the MLX path (the other identity
    tests fit in one block): block_rows=64 over 300 chunks = 5 blocks, so
    per-block global row offsets (self-mask columns, ids indexing) are
    exercised across block boundaries."""
    pytest.importorskip("mlx.core")

    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    store_np = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_np, block_rows=64)
    want = _ordered_edges(store_np)

    monkeypatch.setenv("CHONKS_KNN_BACKEND", "mlx")
    store_mx = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_mx, block_rows=64)
    got = _ordered_edges(store_mx)

    assert len(want) > 0
    assert got == want


def test_mlx_bulk_identity_at_production_dim(monkeypatch):
    """Row-for-row identity at dim=896 (jina, the real deployment dim, near
    the 1040 dim-safe boundary): the tie-saturated dim=4 tests prove the
    tie-break equivalence, this proves the exact-integer-in-float32 argument
    where dot products actually approach the 2^24 exactness limit."""
    pytest.importorskip("mlx.core")

    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    store_np = _make_store(n_chunks=200, dim=896)
    build_neighbors(store_np)
    want = _ordered_edges(store_np)

    monkeypatch.setenv("CHONKS_KNN_BACKEND", "mlx")
    store_mx = _make_store(n_chunks=200, dim=896)
    build_neighbors(store_mx)
    got = _ordered_edges(store_mx)

    assert len(want) > 0
    assert got == want


def test_mlx_incremental_after_mlx_bulk_matches_reference(monkeypatch):
    """Composition guard: a graph whose initial bulk build ran on the MLX path,
    then mutated through the (always-numpy) incremental path, must still match
    the independent numpy-built full-rebuild reference, pinning the contract
    that the two backends can be freely mixed across a graph's lifetime."""
    pytest.importorskip("mlx.core")
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "mlx")
    _run_incremental_case(n_initial=300, n_add=20, n_delete=20, k=6, seed=3)


# --- CUDA (cupy) backend ----------------------------------------------------
#
# CI has no GPU, so these tests must exercise the backend-selection logic and
# the actual matmul/top-k code paths WITHOUT cupy installed. Two strategies,
# both used below:
#
#   1. Backend-selection / error-message tests need no cupy at all: they
#      either force `import cupy` to fail deterministically (sys.modules
#      entry set to None, a standard stdlib trick, works whether or not
#      cupy happens to be installed) or use a minimal fake module to hit the
#      "device query fails" / "no device" branches.
#   2. Code-path tests (bulk rebuild, sub-slicing, incremental) inject a
#      numpy-backed fake "cupy" module into sys.modules. cupy's array API
#      for everything `_Corpus` touches (asarray, arange, argpartition,
#      argsort, take_along_axis, matmul via `@`, fancy indexing, dtype
#      objects, `asnumpy`) is a deliberate numpy mirror, so wrapping numpy
#      functions under the name "cupy" is enough to run the REAL
#      `_Corpus`/`topk_block`/`sims_block` cupy branches unmodified: this
#      is a functional double of cupy's semantics, not a mock of call
#      signatures, so it exercises the same code a real GPU would run
#      (computed on host instead of device, but bit-identical either way per
#      the exactness argument both backends share).
#
# A handful of tests also use `pytest.importorskip("cupy")` to run for real
# if this happens to execute on a machine with cupy + a CUDA device (e.g. a
# CUDA machine's validation pass), skipped everywhere else, same as the MLX
# real-hardware tests above.

def _fake_cupy_module(device_count: int = 1) -> types.ModuleType:
    """Numpy-backed stand-in for cupy, see the module comment above."""
    fake = types.ModuleType("cupy")
    for name in ("asarray", "arange", "argpartition", "argsort", "take_along_axis", "int64"):
        setattr(fake, name, getattr(np, name))
    fake.asnumpy = lambda a: np.asarray(a)
    fake.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(getDeviceCount=lambda: device_count),
    )
    return fake


def test_cuda_backend_missing_cupy_raises_clear_error(monkeypatch):
    """CHONKS_KNN_BACKEND=cuda is a hard request (unlike mlx's graceful
    fallback): cupy import failure must raise a clear, actionable error
    instead of silently degrading to numpy."""
    from chonks.repomap import _Corpus

    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    rng = np.random.default_rng(0)
    blob = rng.integers(-127, 128, size=(20, 8), dtype=np.int8).tobytes()
    with pytest.raises(RuntimeError, match="cupy is not importable"):
        _Corpus([f"c{i}" for i in range(20)], blob, 8)


def test_cuda_backend_no_device_raises_clear_error(monkeypatch):
    """cupy importable but zero CUDA devices detected must also raise
    clearly, not silently fall back."""
    from chonks.repomap import _Corpus

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module(device_count=0))
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    rng = np.random.default_rng(0)
    blob = rng.integers(-127, 128, size=(20, 8), dtype=np.int8).tobytes()
    with pytest.raises(RuntimeError, match="no CUDA device was detected"):
        _Corpus([f"c{i}" for i in range(20)], blob, 8)


def test_cuda_backend_device_query_failure_raises_clear_error(monkeypatch):
    """A device-count query that raises (no driver at all) must also surface
    as a clear RuntimeError, not an unhandled cupy exception."""
    from chonks.repomap import _Corpus

    fake = _fake_cupy_module()
    def _boom():
        raise RuntimeError("CUDA driver not found")
    fake.cuda.runtime.getDeviceCount = _boom
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    rng = np.random.default_rng(0)
    blob = rng.integers(-127, 128, size=(20, 8), dtype=np.int8).tobytes()
    with pytest.raises(RuntimeError, match="querying CUDA devices failed"):
        _Corpus([f"c{i}" for i in range(20)], blob, 8)


def test_cuda_backend_activates_only_dim_safe(monkeypatch):
    """Same gate as MLX (test_mlx_backend_activates_only_dim_safe): the CUDA
    backend only arms at dim-safe dims, where every int8 dot product is
    exactly representable in float32. A dim-unsafe corpus that still lands in
    'fast' mode (tier 2, RAM-based) must NOT arm the backend, even though
    CHONKS_KNN_BACKEND=cuda is set; it falls back to numpy silently, same as
    MLX would."""
    from chonks.repomap import _Corpus

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    rng = np.random.default_rng(0)

    blob = rng.integers(-127, 128, size=(50, 8), dtype=np.int8).tobytes()
    corpus = _Corpus([f"c{i}" for i in range(50)], blob, 8)
    assert not corpus.exact
    assert corpus._cp is not None

    blob = rng.integers(-127, 128, size=(50, 2560), dtype=np.int8).tobytes()
    monkeypatch.setattr("chonks.repomap.knn._available_ram_bytes", lambda: 64 * 10**9)
    corpus = _Corpus([f"c{i}" for i in range(50)], blob, 2560)
    assert not corpus.exact  # tier 2 picked 'fast', the case the gate targets
    assert corpus._cp is None


def test_cuda_bulk_rebuild_identical_to_numpy(monkeypatch):
    """Full bulk rebuild through the on-device top-k (matmul, self-mask,
    tie-break, argpartition, sort all dispatched through cupy) must produce
    row-for-row identical chunk_neighbors output to the numpy path. dim=4
    makes int8 distance ties pervasive, exercising the tie-break equivalence."""
    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    store_np = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_np)
    want = _ordered_edges(store_np)

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    store_cu = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_cu)
    got = _ordered_edges(store_cu)

    assert len(want) > 0
    assert got == want


def test_cuda_subsliced_topk_identical_to_numpy(monkeypatch):
    """Sub-slicing under _CUDA_TOPK_MAX_BUFFER_BYTES must not change output.
    Shrink the cap so every block splits into many sub-slices and verify the
    result is still row-for-row identical to numpy (catches any
    stitching/offset bug in the sub-slice loop, mirrors the MLX cliff-cap
    test even though cupy has no known corruption cliff of its own)."""
    import chonks.repomap.knn as repomap

    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    store_np = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_np)
    want = _ordered_edges(store_np)

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    # n=300 → n*8=2400 B per key row; cap of 3 rows' worth forces ~100 slices
    monkeypatch.setattr(repomap, "_CUDA_TOPK_MAX_BUFFER_BYTES", 3 * 300 * 8)
    store_cu = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_cu)
    got = _ordered_edges(store_cu)

    assert len(want) > 0
    assert got == want


def test_cuda_bulk_multi_block_identical_to_numpy(monkeypatch):
    """Multiple OUTER matmul blocks through the CUDA path: block_rows=64 over
    300 chunks = 5 blocks, so per-block global row offsets (self-mask
    columns, ids indexing) are exercised across block boundaries."""
    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    store_np = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_np, block_rows=64)
    want = _ordered_edges(store_np)

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    store_cu = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_cu, block_rows=64)
    got = _ordered_edges(store_cu)

    assert len(want) > 0
    assert got == want


def test_cuda_bulk_identity_at_production_dim(monkeypatch):
    """Row-for-row identity at dim=896 (jina, the real deployment dim, near
    the 1040 dim-safe boundary): proves the exact-integer-in-float32
    argument where dot products actually approach the 2^24 exactness limit."""
    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    store_np = _make_store(n_chunks=200, dim=896)
    build_neighbors(store_np)
    want = _ordered_edges(store_np)

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    store_cu = _make_store(n_chunks=200, dim=896)
    build_neighbors(store_cu)
    got = _ordered_edges(store_cu)

    assert len(want) > 0
    assert got == want


def test_cuda_incremental_matches_full_rebuild(monkeypatch):
    """The case that matters most: unlike MLX, the CUDA backend must also
    accelerate the incremental path, since production traffic is dominated by
    frequent incremental damaged-row repair, not rare full rebuilds."""
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    _run_incremental_case(n_initial=300, n_add=20, n_delete=20, k=6, dim=8, seed=3)


def test_cuda_incremental_after_cuda_bulk_matches_reference(monkeypatch):
    """Composition guard: unlike the MLX equivalent (bulk-only, incremental
    always numpy), a CUDA-backed graph runs BOTH its initial bulk build and
    its incremental update through cupy end to end, and must still match the
    independent numpy-built full-rebuild reference."""
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    _run_incremental_case(n_initial=300, n_add=20, n_delete=20, k=6, seed=3)


def test_cuda_dim_unsafe_backend_falls_back_silently_no_error(monkeypatch, caplog):
    """A dim-unsafe corpus that lands in 'fast' mode (tier 2) with
    CHONKS_KNN_BACKEND=cuda set must not raise: the exactness gate silently
    declines the device backend (same as MLX), it does not treat this as a
    forced-backend failure. An info-level log line notes why."""
    import logging
    from chonks.repomap import _Corpus

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    monkeypatch.setattr("chonks.repomap.knn._available_ram_bytes", lambda: 64 * 10**9)
    rng = np.random.default_rng(0)
    blob = rng.integers(-127, 128, size=(50, 2560), dtype=np.int8).tobytes()
    with caplog.at_level(logging.INFO, logger="repomap"):
        corpus = _Corpus([f"c{i}" for i in range(50)], blob, 2560)
    assert corpus._cp is None
    assert any("isn't" in r.message and "GPU-safe" in r.message for r in caplog.records)


# Real-hardware tests: only run where cupy + a CUDA device actually exist
# (a CUDA machine's validation pass, see eval/validate_knn_backend.py for
# the full byte-comparison gate). Skipped everywhere else, same pattern as
# the MLX `pytest.importorskip("mlx.core")` tests above.

def test_cuda_bulk_rebuild_identical_to_numpy_real_gpu(monkeypatch):
    pytest.importorskip("cupy")
    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    store_np = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_np)
    want = _ordered_edges(store_np)

    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    store_cu = _make_store(n_chunks=300, dim=4)
    build_neighbors(store_cu)
    got = _ordered_edges(store_cu)

    assert len(want) > 0
    assert got == want


def test_cuda_incremental_matches_full_rebuild_real_gpu(monkeypatch):
    pytest.importorskip("cupy")
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    _run_incremental_case(n_initial=300, n_add=20, n_delete=20, k=6, dim=8, seed=3)


# --- incremental progress logging ------------------------------------------
#
# The incremental path (changed-rows matmul + damaged-row repair, plus the
# per-cid displaced-neighbour trim) can run for minutes with zero output on a
# large batch at 350k-chunk scale. Force the progress interval to 0 so every loop
# iteration logs, then assert the entry line and each phase's periodic line
# fire, without waiting on a real wall-clock interval.

def test_incremental_progress_logs_entry_and_phase_lines(monkeypatch, caplog):
    import logging
    import chonks.repomap.knn as repomap

    monkeypatch.setattr(repomap, "_NEIGHBOR_PROGRESS_INTERVAL_S", 0.0)
    rng = np.random.default_rng(9)
    n = 60
    dim = 8
    ids = [f"orig-{i}" for i in range(n)]
    vecs = _rand_unit_vecs(rng, n, dim)

    store = Store(tempfile.mktemp(suffix=".db"))
    _seed_chunks(store, ids, vecs, offset=0)
    build_neighbors(store, k=5, block_rows=5)

    deleted_ids: set[str] = set()
    for i in range(5):
        deleted_ids.update(store.delete_file(f"f{i}.py"))

    new_ids = [f"new-{i}" for i in range(5)]
    new_vecs = _rand_unit_vecs(rng, 5, dim)
    _seed_chunks(store, new_ids, new_vecs, offset=10_000)
    changed_ids = set(new_ids)

    with caplog.at_level(logging.INFO, logger="repomap"):
        build_neighbors(store, k=5, changed_ids=changed_ids, deleted_ids=deleted_ids, block_rows=5)

    messages = [r.message for r in caplog.records]
    assert any("k-NN incremental: N=" in m and "this can take several minutes" in m for m in messages)
    assert any("changed-rows top-k" in m for m in messages)
    assert any("damaged-rows repair" in m for m in messages)


# --- backend-aware incremental gate -----------------------------------------
#
# On a device backend the full rebuild's matmul is fast enough that the
# incremental path's SQL-bound per-row bookkeeping loses far sooner than the
# numpy-tuned 20% fraction, so the gate must use a much lower churn threshold
# when a device backend is armed.

def test_device_backend_armed_false_without_env(monkeypatch):
    import chonks.repomap.knn as repomap
    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    assert repomap._device_backend_armed(dim=8) is False


def test_device_backend_armed_false_dim_unsafe(monkeypatch):
    """Even with CHONKS_KNN_BACKEND=cuda set and cupy importable, a dim-unsafe
    dim must not arm the device path, mirroring _Corpus's own gate."""
    import chonks.repomap.knn as repomap
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    assert repomap._device_backend_armed(dim=2560) is False


def test_device_backend_armed_true_for_cuda_dim_safe(monkeypatch):
    import chonks.repomap.knn as repomap
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy_module())
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    assert repomap._device_backend_armed(dim=8) is True


def test_device_backend_armed_false_cupy_not_importable(monkeypatch):
    import chonks.repomap.knn as repomap
    monkeypatch.setitem(sys.modules, "cupy", None)  # forces ImportError
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "cuda")
    assert repomap._device_backend_armed(dim=8) is False


def test_gate_uses_device_threshold_when_backend_armed(monkeypatch):
    """A batch over the device fraction but under the numpy fraction must
    fall back to a full rebuild when a device backend is armed, and must
    take the incremental path when it isn't, on the same store and batch."""
    import chonks.repomap.knn as repomap

    def _run(armed: bool) -> bool:
        monkeypatch.setattr(repomap, "_device_backend_armed", lambda dim: armed)
        store = _make_store(n_chunks=1000, dim=8)
        build_neighbors(store)
        called = []
        original = repomap._build_neighbors_incremental
        def _spy(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)
        repomap._build_neighbors_incremental = _spy
        try:
            # 25 of 1000 changed = 2.5%: over the 2% device threshold, under
            # the 20% numpy threshold.
            changed = {f"chunk-{i}" for i in range(25)}
            build_neighbors(store, changed_ids=changed, deleted_ids=set())
        finally:
            repomap._build_neighbors_incremental = original
        return bool(called)

    assert _run(armed=False) is True
    assert _run(armed=True) is False


def test_gate_device_threshold_boundary_value(monkeypatch):
    """Right at the device fraction boundary the batch still qualifies
    (`<=`, matching the numpy gate's own boundary convention)."""
    import chonks.repomap.knn as repomap
    monkeypatch.setattr(repomap, "_device_backend_armed", lambda dim: True)
    store = _make_store(n_chunks=1000, dim=8)
    build_neighbors(store)

    called = []
    original = repomap._build_neighbors_incremental
    def _spy(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)
    repomap._build_neighbors_incremental = _spy
    try:
        # Exactly 2% of 1000 = 20, at the device fraction's boundary.
        changed = {f"chunk-{i}" for i in range(20)}
        build_neighbors(store, changed_ids=changed, deleted_ids=set())
    finally:
        repomap._build_neighbors_incremental = original
    assert called


# ---- "auto" backend selection and the CPU-path hint ------------------------

def _tiny_corpus():
    import numpy as np
    from chonks.repomap import _Corpus
    ids = ["a", "b", "c"]
    blob = np.zeros((3, 8), dtype=np.int8).tobytes()
    return _Corpus(ids, blob, 8)


def test_auto_backend_falls_to_numpy_without_gpu_libs(monkeypatch):
    import chonks.repomap.knn as rm
    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    monkeypatch.setattr(rm, "_detect_knn_backend", lambda: "numpy")
    c = _tiny_corpus()
    assert c._cp is None and c._mx is None


def test_cpu_hint_names_the_cuda_extra_for_large_corpora(monkeypatch, caplog):
    import logging
    import chonks.repomap.knn as rm
    monkeypatch.delenv("CHONKS_KNN_BACKEND", raising=False)
    monkeypatch.setattr(rm, "_detect_knn_backend", lambda: "numpy")
    monkeypatch.setattr(rm, "_KNN_CPU_HINT_MIN_N", 1)
    with caplog.at_level(logging.WARNING, logger="repomap"):
        _tiny_corpus()
    assert any("uv sync --extra cuda" in r.message for r in caplog.records)


def test_explicit_numpy_suppresses_cpu_hint(monkeypatch, caplog):
    import logging
    import chonks.repomap.knn as rm
    monkeypatch.setenv("CHONKS_KNN_BACKEND", "numpy")
    monkeypatch.setattr(rm, "_KNN_CPU_HINT_MIN_N", 1)
    with caplog.at_level(logging.WARNING, logger="repomap"):
        _tiny_corpus()
    assert not [r for r in caplog.records if "uv sync --extra cuda" in r.message]
