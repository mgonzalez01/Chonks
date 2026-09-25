"""Integration test for index_paths -> build_neighbors wiring. The incremental
k-NN path itself is unit tested in test_build_neighbors.py; what's untested
there is the chonks/index/pipeline.py plumbing that collects changed_ids/deleted_ids across
a real index run. Proves it's correct by cross-checking a real incremental
run against a full rebuild's ground truth on the same corpus."""
import hashlib
import logging
from collections import Counter
from unittest.mock import patch

import httpx
import pytest

from chonks.embed.client import compress_for_embed
from chonks.index.pipeline import EmbedderDownError, index_paths
from chonks.storage.store import Store

_DIM = 8


def _deterministic_vec(text: str) -> list[float]:
    # Hash-derived so distinct chunks get distinct, reproducible embeddings
    # (a real k-NN graph, not every chunk tying on an identical vector).
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(b - 127.5) / 127.5 for b in digest[:_DIM]]


class _FakeEmbedder:
    model = "fake"
    url   = "http://localhost:9999"

    def embed_documents(self, texts, client=None, **kw):
        return [_deterministic_vec(t) for t in texts]

    def embed_queries(self, texts, client=None, **kw):
        return [_deterministic_vec(t) for t in texts]


def _bulk_file(idx: int) -> str:
    # Several functions per file: bulks up the corpus so a handful of
    # changed/deleted chunks stays under build_neighbors'
    # _NEIGHBOR_INCREMENTAL_MAX_FRACTION threshold.
    funcs = []
    for j in range(4):
        funcs.append(
            f"def bulk_{idx}_{j}(x):\n"
            f"    total = 0\n"
            f"    for i in range(x + {idx}{j}):\n"
            f"        total += i * {idx} - {j}\n"
            f"    return total\n"
        )
    return "\n\n".join(funcs) + "\n"


def _small_file(tag: str) -> str:
    # Single-function file, so each mutated file contributes exactly one
    # chunk to the changed/deleted sets.
    return f"def {tag}(x):\n    return x + len('{tag}')\n"


def _all_edges(store: Store) -> dict:
    rows = store._conn.execute(
        "SELECT chunk_id, neighbor_id, distance FROM chunk_neighbors"
    ).fetchall()
    return {(r["chunk_id"], r["neighbor_id"]): r["distance"] for r in rows}


def _assert_graphs_equal(got: dict, want: dict) -> None:
    assert set(got) == set(want), (
        f"edge sets differ: missing={set(want) - set(got)}, extra={set(got) - set(want)}"
    )
    for key in want:
        assert got[key] == want[key] or abs(got[key] - want[key]) < 1e-4, (
            f"distance mismatch for {key}: {got[key]!r} vs {want[key]!r}"
        )


def _write_initial_tree(tmp_path) -> None:
    for i in range(7):
        (tmp_path / f"bulk_{i}.py").write_text(_bulk_file(i))
    (tmp_path / "to_modify.py").write_text(_small_file("orig_modify"))
    (tmp_path / "to_delete.py").write_text(_small_file("orig_delete"))


def test_index_paths_incremental_wiring_matches_full_rebuild(tmp_path, caplog):
    from chonks.index.graph.knn import build_neighbors

    _write_initial_tree(tmp_path)
    store = Store(tmp_path / "test.db")
    embedder = _FakeEmbedder()

    first = index_paths([str(tmp_path)], store, embedder, root=tmp_path)
    assert first["indexed"] > 0
    n0 = len(store.get_all_int8_embeddings()[0])
    assert n0 >= 15, f"corpus too small to exercise the incremental path safely: {n0}"

    (tmp_path / "to_modify.py").write_text(_small_file("modified_now"))
    (tmp_path / "to_delete.py").unlink()
    (tmp_path / "brand_new.py").write_text(_small_file("brand_new_fn"))

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="repomap"):
        second = index_paths([str(tmp_path)], store, embedder, root=tmp_path)
    assert second["indexed"] > 0
    assert second["pruned"] >= 1

    incremental_logged = any(
        "k-NN incremental update" in rec.message for rec in caplog.records
    )
    assert incremental_logged, (
        "build_neighbors did not take the incremental path on the second "
        "index_paths run — changed_ids/deleted_ids plumbing may be broken. "
        f"log records: {[r.message for r in caplog.records]}"
    )

    incremental_edges = _all_edges(store)

    # Ground truth: a full rebuild on the SAME (already-mutated) corpus.
    build_neighbors(store)
    full_rebuild_edges = _all_edges(store)

    _assert_graphs_equal(incremental_edges, full_rebuild_edges)


def test_index_paths_orphan_prune_matches_full_rebuild(tmp_path, caplog):
    # Deleting a file with no other changes exercises the orphan-prune call
    # site, not the changed-file call site.
    from chonks.index.graph.knn import build_neighbors

    _write_initial_tree(tmp_path)
    store = Store(tmp_path / "test.db")
    embedder = _FakeEmbedder()

    first = index_paths([str(tmp_path)], store, embedder, root=tmp_path)
    assert first["indexed"] > 0
    n0 = len(store.get_all_int8_embeddings()[0])
    assert n0 >= 15

    (tmp_path / "bulk_0.py").unlink()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="repomap"):
        second = index_paths([str(tmp_path)], store, embedder, root=tmp_path)
    assert second["pruned"] >= 1

    incremental_logged = any(
        "k-NN incremental update" in rec.message for rec in caplog.records
    )
    assert incremental_logged, (
        "build_neighbors did not take the incremental path on the orphan-prune "
        f"run. log records: {[r.message for r in caplog.records]}"
    )

    incremental_edges = _all_edges(store)

    build_neighbors(store)
    full_rebuild_edges = _all_edges(store)

    _assert_graphs_equal(incremental_edges, full_rebuild_edges)


def test_reindex_with_zero_completed_files_still_purges_deleted_edges(tmp_path):
    # Regression: the old gate was `indexed > 0 or pruned > 0`, which is False
    # when a file's old chunks are deleted but 0 files complete this run, so
    # graph-maintenance never ran and old edges dangled forever.
    import chonks.index.pipeline as pipeline_mod
    from chonks.index.pipeline import index_paths

    (tmp_path / "callee.py").write_text(
        "def target_function(x):\n    return x + 1\n"
    )
    (tmp_path / "caller.py").write_text(
        "def caller_function(y):\n    return target_function(y) * 2\n"
    )

    store = Store(tmp_path / "test.db")
    embedder = _FakeEmbedder()

    first = index_paths([str(tmp_path)], store, embedder, root=tmp_path)
    assert first["indexed"] == 2

    callee_chunk_id = next(
        r["chunk_id"] for r in store.get_all_symbols()
        if r["name"] == "target_function"
    )
    assert callee_chunk_id is not None

    refs_before = store._conn.execute(
        "SELECT 1 FROM chunk_refs WHERE from_id=? OR to_id=?",
        (callee_chunk_id, callee_chunk_id),
    ).fetchall()
    neighbors_before = store._conn.execute(
        "SELECT 1 FROM chunk_neighbors WHERE chunk_id=? OR neighbor_id=?",
        (callee_chunk_id, callee_chunk_id),
    ).fetchall()
    assert refs_before, "setup did not produce a chunk_refs edge to test cleanup of"
    assert neighbors_before, "setup did not produce a chunk_neighbors edge to test cleanup of"

    # Content change triggers the scan-time delete of the old chunk; the
    # re-parse below then fails, so no new chunk is ever queued to replace it.
    (tmp_path / "callee.py").write_text(
        "def target_function(x):\n    return x + 999\n"
    )
    real_segment_file = pipeline_mod.segment_file

    def failing_segment_file(src, lang, *, path=None, **kw):
        if path == "callee.py":
            raise RuntimeError("simulated parser crash")
        return real_segment_file(src, lang, path=path, **kw)

    with patch.object(pipeline_mod, "segment_file", side_effect=failing_segment_file):
        second = index_paths([str(tmp_path)], store, embedder, root=tmp_path)

    assert second["indexed"] == 0, (
        "test setup invariant broken: expected zero files to complete this run"
    )
    assert second["pruned"] == 0, (
        "test setup invariant broken: expected nothing pruned this run "
        "(callee.py still exists on disk, just failed to re-parse)"
    )
    assert second["errors"] >= 1

    refs_after = store._conn.execute(
        "SELECT 1 FROM chunk_refs WHERE from_id=? OR to_id=?",
        (callee_chunk_id, callee_chunk_id),
    ).fetchall()
    neighbors_after = store._conn.execute(
        "SELECT 1 FROM chunk_neighbors WHERE chunk_id=? OR neighbor_id=?",
        (callee_chunk_id, callee_chunk_id),
    ).fetchall()
    assert not refs_after, (
        "deleted chunk's chunk_refs edges were not purged when 0 files completed"
    )
    assert not neighbors_after, (
        "deleted chunk's chunk_neighbors edges were not purged when 0 files completed"
    )


def _function(name: str, extra: str = "") -> str:
    body = "".join(f"    step_{i} = x * {i} + len('{name}')\n" for i in range(12))
    return f"def {name}(x):\n{extra}{body}    return x\n"


def _write_module(root, stem: str, edited: bool = False) -> None:
    funcs = [_function(f"{stem}_alpha", "    x += 1\n" if edited else ""),
             _function(f"{stem}_beta"), _function(f"{stem}_gamma")]
    (root / f"{stem}.py").write_text("\n\n".join(funcs))


class _CountingEmbedder(_FakeEmbedder):
    def __init__(self, model="fake", doc_prefix="", salt=""):
        self.model, self.doc_prefix, self.salt = model, doc_prefix, salt
        self.sent: list[str] = []

    def embed_documents(self, texts, client=None, **kw):
        self.sent.extend(texts)
        return [_deterministic_vec(self.salt + t) for t in texts]


class _DownEmbedder(_CountingEmbedder):
    def embed_documents(self, texts, client=None, **kw):
        raise httpx.ConnectError("connection refused")


def _chunk_rows(store: Store):
    return store._conn.execute("SELECT id, name, content, path FROM chunks").fetchall()


def _embed_texts(store: Store) -> Counter:
    return Counter(compress_for_embed(r["content"], name=r["name"], path=r["path"])
                   for r in _chunk_rows(store))


def _vectors_by_text(store: Store) -> dict[str, bytes]:
    rows = _chunk_rows(store)
    blobs = store.get_int8_embeddings_by_ids([r["id"] for r in rows])
    return {compress_for_embed(r["content"], name=r["name"], path=r["path"]): blobs[r["id"]]
            for r in rows}


@pytest.fixture
def reuse_tree(tmp_path, monkeypatch):
    import chonks.index.postindex as postindex
    monkeypatch.setattr(postindex, "build_folder_summaries",
                        lambda store, embed_fn: {"refreshed": 0, "pruned": 0})
    for stem in ("calc", "grid", "route"):
        _write_module(tmp_path, stem)
    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _CountingEmbedder(), root=tmp_path)
    yield tmp_path, store
    store.close()


def test_force_on_unchanged_tree_embeds_nothing(reuse_tree):
    root, store = reuse_tree
    emb = _CountingEmbedder()
    stats = index_paths([str(root)], store, emb, root=root, force=True)
    assert emb.sent == []
    assert stats["chunks_reused"] == stats["chunks_embedded"] == 9


@pytest.mark.parametrize("force", [True, False])
def test_reindex_after_editing_one_chunk_embeds_only_that_chunk(reuse_tree, force):
    root, store = reuse_tree
    ids_before = {r["name"]: r["id"] for r in _chunk_rows(store)}
    texts_before = _embed_texts(store)
    _write_module(root, "calc", edited=True)
    emb = _CountingEmbedder()
    index_paths([str(root)], store, emb, root=root, force=force)
    assert len(emb.sent) == 1
    assert emb.sent == list((_embed_texts(store) - texts_before).elements())
    ids_after = {r["name"]: r["id"] for r in _chunk_rows(store)}
    assert ids_after["calc_beta"] != ids_before["calc_beta"]


def test_reused_vectors_equal_the_stored_ones(reuse_tree):
    root, store = reuse_tree
    stored = _vectors_by_text(store)
    index_paths([str(root)], store, _CountingEmbedder(salt="other"), root=root, force=True)
    assert _vectors_by_text(store) == stored


def test_force_with_another_model_embeds_everything(reuse_tree):
    root, store = reuse_tree
    texts = _embed_texts(store)
    emb = _CountingEmbedder(model="other-model")
    with pytest.raises(RuntimeError, match="model mismatch"):
        index_paths([str(root)], store, emb, root=root, force=True)
    assert Counter(emb.sent) == texts


def test_force_with_unknown_stored_model_embeds_everything(reuse_tree):
    root, store = reuse_tree
    texts = _embed_texts(store)
    with store._lock:
        store._conn.execute("DELETE FROM meta WHERE key='embedding_model'")
        store._conn.commit()
    emb = _CountingEmbedder()
    index_paths([str(root)], store, emb, root=root, force=True)
    assert Counter(emb.sent) == texts


def test_force_after_doc_prefix_change_embeds_everything(reuse_tree):
    root, store = reuse_tree
    texts = _embed_texts(store)
    emb = _CountingEmbedder(doc_prefix="Candidate code snippet:\n")
    index_paths([str(root)], store, emb, root=root, force=True)
    assert Counter(emb.sent) == texts


@pytest.mark.parametrize("doc_prefix", ["", "Candidate code snippet:\n"])
def test_vectors_left_mixed_by_a_partial_run_are_not_reused(reuse_tree, doc_prefix):
    root, store = reuse_tree
    _write_module(root, "calc", edited=True)
    index_paths([str(root)], store, _CountingEmbedder(doc_prefix="Candidate code snippet:\n"),
                root=root)
    emb = _CountingEmbedder(doc_prefix=doc_prefix)
    index_paths([str(root)], store, emb, root=root, force=True)
    assert Counter(emb.sent) == _embed_texts(store)


def test_force_on_a_db_without_the_embed_text_stamp_reuses_vectors(reuse_tree):
    root, store = reuse_tree
    with store._lock:
        store._conn.execute("DELETE FROM meta WHERE key='embed_text'")
        store._conn.commit()
    emb = _CountingEmbedder()
    index_paths([str(root)], store, emb, root=root, force=True)
    assert emb.sent == []


@pytest.mark.parametrize("embed_batch", [1, 3])
def test_dead_embedder_aborts_when_batches_also_hold_reused_chunks(reuse_tree, embed_batch):
    root, store = reuse_tree
    stems = ("calc", "grid", "route", "mesh", "tile", "zone")
    for stem in stems:
        _write_module(root, stem)
    index_paths([str(root)], store, _CountingEmbedder(), root=root)
    for stem in stems:
        _write_module(root, stem, edited=True)
    with pytest.raises(EmbedderDownError):
        index_paths([str(root)], store, _DownEmbedder(), root=root, embed_batch=embed_batch)
