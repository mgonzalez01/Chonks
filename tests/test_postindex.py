"""run_post_index_passes: the one-time orphan sweep, the pass timings and log lines."""
import logging
import time

import pytest

import chonks.index.postindex as postindex
from chonks.storage.store import Store


def _run(store, **kw):
    args = dict(force=False, indexed=0, pruned=0, changed_chunk_ids=set(),
                deleted_chunk_ids=set(), deleted_chunk_names=set(),
                edge_type_weights=None, cap_mentions_fanout=False, associated_top_frac=0.02)
    args.update(kw)
    return postindex.run_post_index_passes(store, object(), **args)


def test_orphan_sweep_runs_once_on_a_no_op_run(tmp_path, caplog):
    store = Store(tmp_path / "t.db")
    store._conn.execute("INSERT INTO chunk_neighbors VALUES ('ghost', 'gone', 0.1)")
    store._conn.commit()
    with caplog.at_level(logging.INFO, logger="chonks.chunker"):
        _run(store)
    assert "Checking for orphan edges." in [r.getMessage() for r in caplog.records]
    assert store._conn.execute("SELECT COUNT(*) FROM chunk_neighbors").fetchone()[0] == 0
    assert store.get_meta("orphan_sweep_v1") == "1"
    store.close()


def test_a_forced_run_skips_the_orphan_sweep(tmp_path, monkeypatch, stub_passes):
    monkeypatch.setattr(postindex, "build_folder_summaries", lambda *a, **k: {"refreshed": 0, "pruned": 0})
    store = Store(tmp_path / "t.db")
    store._conn.execute("INSERT INTO chunk_neighbors VALUES ('ghost', 'gone', 0.1)")
    store._conn.commit()
    _run(store, force=True, indexed=1)
    # The stubbed rebuilds leave the row, so only a sweep would remove it.
    assert store._conn.execute("SELECT COUNT(*) FROM chunk_neighbors").fetchone()[0] == 1
    assert store.get_meta("orphan_sweep_v1") == "1"
    store.close()


@pytest.fixture
def stub_passes(monkeypatch):
    monkeypatch.setattr(postindex, "build_refs", lambda *a, **k: 0)
    monkeypatch.setattr(postindex, "build_neighbors", lambda *a, **k: 0)
    monkeypatch.setattr(postindex, "rebuild_hierarchy", lambda *a, **k: {"nodes": 0, "edges": 0})
    monkeypatch.setattr(postindex, "persist_pagerank", lambda *a, **k: 0)


def test_failed_folder_summaries_report_no_duration(tmp_path, monkeypatch, stub_passes):
    def failing(*a, **k):
        time.sleep(0.02)
        raise RuntimeError("embedder down")
    monkeypatch.setattr(postindex, "build_folder_summaries", failing)
    store = Store(tmp_path / "t.db")
    summaries = _run(store, indexed=1)[3]
    assert summaries == 0.0
    store.close()


def test_successful_folder_summaries_report_a_duration(tmp_path, monkeypatch, stub_passes):
    def ok(*a, **k):
        time.sleep(0.02)
        return {"refreshed": 0, "pruned": 0}
    monkeypatch.setattr(postindex, "build_folder_summaries", ok)
    store = Store(tmp_path / "t.db")
    summaries = _run(store, indexed=1)[3]
    assert summaries >= 0.02
    store.close()


def test_each_pass_says_when_it_starts(tmp_path, monkeypatch, stub_passes, caplog):
    monkeypatch.setattr(postindex, "build_folder_summaries", lambda *a, **k: {"refreshed": 0, "pruned": 0})
    store = Store(tmp_path / "t.db")
    with caplog.at_level(logging.INFO, logger="chonks.chunker"):
        _run(store, force=True, indexed=1)
    store.close()
    lines = [r.getMessage() for r in caplog.records]
    expected = ["Rebuilding the full-text index.", "Rebuilt the full-text index in",
                "Building the folder hierarchy.", "Built hierarchy:",
                "Computing PageRank.", "Persisted 0 PageRank scores.",
                "Refreshing folder summaries.", "Folder summaries:"]
    found = [next(i for i, line in enumerate(lines) if line.startswith(e)) for e in expected]
    assert found == sorted(found), lines
