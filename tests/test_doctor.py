"""Tests for doctor.py's read-only index health report."""
import json
import sqlite3
import time
from pathlib import Path

from chonks.chunking import CHUNKER_VERSION
from chonks.doctor import _connect_readonly, build_report
from chonks.store import SCHEMA_VERSION, Store


def _fake_embedding(dim: int = 4) -> list[float]:
    return [0.5] * dim


def _chunk(id, path, name, language="python", s=1, e=5):
    return {"id": id, "path": path, "language": language, "chunk_type": "function_definition",
            "name": name, "start_line": s, "end_line": e, "content": f"{name}()"}


def _build_synthetic_store(tmp_path, root: Path | None = None) -> Store:
    store = Store(tmp_path / "test.db")
    chunks = [
        _chunk("a", "core/engine.py", "Engine"),
        _chunk("b", "core/utils.py", "Helper"),
        _chunk("c", "render/shader.cpp", "Shader", language="cpp"),
    ]
    store.insert_chunks(chunks, [_fake_embedding() for _ in chunks])
    store.insert_refs([
        ("a", "b", "calls"),
        ("a", "c", "mentions"),
    ])
    store.insert_neighbors([("a", "b", 0.1), ("b", "c", 0.2)])
    store.upsert_file("core/engine.py", 10, 0.0, "h1")
    store.upsert_file("core/utils.py", 10, 0.0, "h2")
    store.upsert_file("render/shader.cpp", 10, 0.0, "h3")
    store.set_meta("indexed_root", str(root) if root else "/repo/example")
    store.set_meta("chunker_version", str(CHUNKER_VERSION))
    store.commit()
    return store


def _report_for(db_path: Path, config: dict | None = None) -> str:
    conn = _connect_readonly(str(db_path))
    try:
        return build_report(conn, config or {}, str(db_path))
    finally:
        conn.close()


def test_schema_version_matches(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert f"schema_version: {SCHEMA_VERSION} (matches code)" in report
    assert "MISMATCH" not in report


def test_schema_version_mismatch_flagged(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    # Corrupt the stored schema_version directly with a plain rw connection:
    # Store's own raise-on-mismatch means we can't do this through Store.
    conn = sqlite3.connect(tmp_path / "test.db")
    conn.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION + 1),))
    conn.commit()
    conn.close()

    report = _report_for(tmp_path / "test.db")
    assert "SCHEMA MISMATCH" in report
    assert f"version {SCHEMA_VERSION + 1}" in report
    assert f"code expects {SCHEMA_VERSION}" in report


def test_chunker_version_recorded(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert f"chunker_version: {CHUNKER_VERSION} (matches code)" in report


def test_chunker_version_not_recorded(tmp_path):
    store = Store(tmp_path / "test.db")
    store.upsert_file("a.py", 1, 0.0, "h1")
    store.commit()
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "chunker_version: not recorded in this DB" in report


def test_vitals_section(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "files:            3" in report
    assert "chunks:           3" in report
    assert "embedding dim:    4" in report
    assert "indexed root:     /repo/example" in report


def test_edges_section(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "chunk_refs[calls]: 1" in report
    assert "chunk_refs[mentions]: 1" in report
    assert "chunk_neighbors: 2" in report


def test_parse_health_not_recorded(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "not recorded in this DB" in report
    assert "unsupported-extension histogram: not recorded in this DB" in report
    assert "oversize chunk count: not recorded in this DB" in report


def test_table_sizes_section(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "chunks: 3 rows" in report
    assert "files: 3 rows" in report
    assert "chunk_refs: 2 rows" in report
    assert "chunk_neighbors: 2 rows" in report


def test_table_sizes_section_lists_the_schema_tables(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    report = _report_for(tmp_path / "test.db")
    sizes = report.split("== Table sizes ==")[1].split("\n== ")[0]
    for table in ("chunk_pagerank", "chunk_indegree", "graph_nodes", "graph_edges", "chunk_literals"):
        assert f"\n{table}: " in sizes
    assert "_fts" not in sizes
    assert "chunk_vecs" not in sizes
    assert "table not present" not in sizes


def test_staleness_stale_when_disk_newer(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    stale_file = root / "touched.py"
    stale_file.write_text("x = 1\n")

    store = _build_synthetic_store(tmp_path, root=root)
    store.close()

    # Push the on-disk mtime into the future relative to indexed_at (which
    # upsert_file stamped at "now" during the build above).
    future = time.time() + 3600
    import os
    os.utime(stale_file, (future, future))

    report = _report_for(tmp_path / "test.db")
    assert "STALE" in report


def test_staleness_ok_when_index_newer(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "old.py").write_text("x = 1\n")
    past = time.time() - 3600
    import os
    os.utime(root / "old.py", (past, past))

    store = _build_synthetic_store(tmp_path, root=root)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "OK: index is at least as new as everything on disk." in report
    assert "STALE" not in report


def test_staleness_root_missing_on_disk(tmp_path):
    store = _build_synthetic_store(tmp_path, root=tmp_path / "nonexistent_root")
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "not found on this machine" in report


def test_staleness_ignores_git_metadata_churn(tmp_path):
    """Touching .git/FETCH_HEAD (a routine `git fetch`) must
    not flip a report to STALE when no actual source changed."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "old.py").write_text("x = 1\n")
    past = time.time() - 3600
    import os
    os.utime(root / "old.py", (past, past))

    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "FETCH_HEAD").write_text("abc123\n")
    future = time.time() + 3600
    os.utime(git_dir / "FETCH_HEAD", (future, future))

    store = _build_synthetic_store(tmp_path, root=root)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "STALE" not in report
    assert "OK: index is at least as new as everything on disk." in report


def test_staleness_honors_excluded_dirs_from_meta(tmp_path):
    """A directory the indexer itself was configured to skip (e.g. build/)
    must not make an otherwise-unchanged index look stale, when the DB
    recorded that exclude list at index time."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "old.py").write_text("x = 1\n")
    past = time.time() - 3600
    import os
    os.utime(root / "old.py", (past, past))

    build_dir = root / "build"
    build_dir.mkdir()
    (build_dir / "generated.py").write_text("z = 1\n")
    future = time.time() + 3600
    os.utime(build_dir / "generated.py", (future, future))

    store = _build_synthetic_store(tmp_path, root=root)
    store.set_meta("exclude", json.dumps(["build/"]))
    store.commit()
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "STALE" not in report
    assert "OK: index is at least as new as everything on disk." in report


def test_staleness_honors_excluded_dirs_from_config_fallback(tmp_path):
    """Older DBs that predate exclude/include meta persistence fall back to
    config.json's `exclude` list."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "old.py").write_text("x = 1\n")
    past = time.time() - 3600
    import os
    os.utime(root / "old.py", (past, past))

    node_modules = root / "node_modules"
    node_modules.mkdir()
    (node_modules / "pkg.js").write_text("var x = 1;\n")
    future = time.time() + 3600
    os.utime(node_modules / "pkg.js", (future, future))

    store = _build_synthetic_store(tmp_path, root=root)
    store.close()  # no exclude/include meta recorded -- pre-dates the fix

    report = _report_for(tmp_path / "test.db", config={"exclude": ["node_modules/"]})
    assert "STALE" not in report
    assert "OK: index is at least as new as everything on disk." in report


def test_schema_version_corrupted_does_not_crash(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    conn = sqlite3.connect(tmp_path / "test.db")
    conn.execute("UPDATE meta SET value=? WHERE key='schema_version'", ("not-a-number",))
    conn.commit()
    conn.close()

    report = _report_for(tmp_path / "test.db")
    assert "corrupted" in report.lower()


def test_chunker_version_mixed_reported(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.set_meta("chunker_version", f"mixed: 0+{CHUNKER_VERSION} (1 unchanged file(s) retain old chunk boundaries)")
    store.commit()
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "mixed:" in report
    assert "force re-index" in report


def _store_from_chunks(tmp_path, chunks: list[dict]) -> Store:
    """Like _build_synthetic_store but with caller-supplied chunk rows,
    used by the path-family tests below to control path/language shape."""
    store = Store(tmp_path / "test.db")
    store.insert_chunks(chunks, [_fake_embedding() for _ in chunks])
    store.set_meta("indexed_root", "/repo/example")
    store.commit()
    return store


def _bulk_chunks(prefix: str, path: str, language: str, n: int) -> list[dict]:
    """n chunks all under the same path (so they share one file), ids
    disambiguated by index, mirrors a docs mirror file with many chunks."""
    return [_chunk(f"{prefix}{i}", path, f"{prefix}{i}", language=language) for i in range(n)]


def test_path_family_breakdown_groups_by_top_level_dir(tmp_path):
    chunks = (
        _bulk_chunks("s", "src/a.py", "python", 3)
        + _bulk_chunks("d", "docs/x.html", "html", 2)
        + _bulk_chunks("r", "README.md", "md", 1)
    )
    store = _store_from_chunks(tmp_path, chunks)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "== Path families ==" in report
    assert "src" in report
    assert "docs" in report
    assert "(root)" in report


def test_path_family_dominance_warning_fires_on_docs_family(tmp_path):
    # docs/mirror.html alone = 84% of the corpus, 100% docs-kind within the
    # family: the sqlglot pdoc shape this issue is calibrated against.
    chunks = (
        _bulk_chunks("d", "docs/mirror.html", "html", 84)
        + _bulk_chunks("s", "src/a.py", "python", 16)
    )
    store = _store_from_chunks(tmp_path, chunks)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "WARNING" in report
    assert "docs/" in report
    assert "docs/mirror.html" in report
    assert '"exclude": ["docs/"]' in report


def test_path_family_no_warning_for_healthy_src_dominant_corpus(tmp_path):
    # src/ = 90% of the corpus, all code: normal code dominance must not warn.
    chunks = (
        _bulk_chunks("s", "src/a.py", "python", 60)
        + _bulk_chunks("s2", "src/b.py", "python", 30)
        + _bulk_chunks("d", "docs/x.md", "md", 10)
    )
    store = _store_from_chunks(tmp_path, chunks)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "WARNING" not in report


def test_path_family_corpus_wide_docs_warning(tmp_path):
    # No single family dominates, but docs-kind chunks are >= 50% of the
    # whole corpus once you add them up across families.
    chunks = []
    for i in range(10):
        chunks += _bulk_chunks(f"f{i}", f"fam{i}/x.md", "md", 6)
    chunks += _bulk_chunks("s", "src/a.py", "python", 40)
    store = _store_from_chunks(tmp_path, chunks)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "WARNING" in report
    assert "docs-kind chunks are" in report


def test_path_family_rollup_beyond_top_n(tmp_path):
    # 12 distinct top-level families: only the top 10 show individually,
    # the remaining 2 roll into a single "other" row.
    chunks = []
    for i in range(12):
        chunks += _bulk_chunks(f"f{i}", f"fam{i:02d}/x.py", "python", 12 - i)
    store = _store_from_chunks(tmp_path, chunks)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "other (2 families)" in report
    # The two smallest families (fam10, fam11 with 2 and 1 chunks) must not
    # get their own row.
    assert "fam10" not in report
    assert "fam11" not in report


def test_dotdir_section_no_dotdir_content(tmp_path):
    chunks = _bulk_chunks("s", "src/a.py", "python", 5)
    store = _store_from_chunks(tmp_path, chunks)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "== Dot-directories ==" in report
    assert "no chunks under dot-directories." in report
    assert "WARNING" not in report.split("== Dot-directories ==")[1].split("== Edges ==")[0]


def test_dotdir_section_reports_share_and_offenders(tmp_path):
    chunks = (
        _bulk_chunks("w", ".claude/worktrees/agent-a/foo.py", "python", 6)
        + _bulk_chunks("s", "src/a.py", "python", 4)
    )
    store = _store_from_chunks(tmp_path, chunks)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "6/10 chunks (60%) live under a dot-directory." in report
    assert ".claude/worktrees" in report
    assert 'exclude": [".claude/worktrees/"]' in report


def test_dotdir_section_rollup_beyond_top_n(tmp_path):
    chunks = []
    for i in range(7):
        chunks += _bulk_chunks(f"d{i}", f".dot{i:02d}/x.py", "python", 7 - i)
    store = _store_from_chunks(tmp_path, chunks)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "... and 2 more" in report


def test_dotdir_section_empty_db(tmp_path):
    store = Store(tmp_path / "test.db")
    store.set_meta("indexed_root", "/repo/example")
    store.commit()
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "== Dot-directories ==" in report
    assert "no chunks in this DB." in report.split("== Dot-directories ==")[1]


def test_read_only_connection_rejects_writes(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    conn = _connect_readonly(str(tmp_path / "test.db"))
    try:
        try:
            conn.execute("INSERT INTO meta(key, value) VALUES('x','y')")
            conn.commit()
            assert False, "expected a write against a read-only connection to fail"
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# --set-model: the one write doctor performs
# ---------------------------------------------------------------------------

def test_set_model_relabels_and_unblocks_store(tmp_path, capsys):
    """Index under one model name, hit the exact-match guard under a case
    variant, relabel via `chonks doctor --set-model`, and confirm the store
    accepts the new name without re-embedding."""
    import pytest
    from chonks.doctor import main

    db = tmp_path / "test.db"
    store = Store(db)
    store.insert_chunks([_chunk("a", "x.py", "A")], [_fake_embedding()], model="Qwen3-Embedding-0.6B")
    store.commit()
    store.close()

    store = Store(db)
    with pytest.raises(ValueError, match="model mismatch.*--set-model"):
        store.insert_chunks([_chunk("b", "y.py", "B")], [_fake_embedding()], model="qwen3-embedding-0.6b")
    store.close()

    main(["--db", str(db), "--set-model", "qwen3-embedding-0.6b"])
    out = capsys.readouterr().out
    assert "Qwen3-Embedding-0.6B -> qwen3-embedding-0.6b" in out
    assert "== Vitals ==" not in out  # no report on the write path

    store = Store(db)
    store.insert_chunks([_chunk("b", "y.py", "B")], [_fake_embedding()], model="qwen3-embedding-0.6b")
    store.commit()
    assert store.count_chunks() == 2
    store.close()

    conn = _connect_readonly(str(db))
    assert conn.execute("SELECT value FROM meta WHERE key='embedding_model'").fetchone()[0] == "qwen3-embedding-0.6b"
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
    conn.close()


def test_set_model_missing_db_errors(tmp_path):
    import pytest
    from chonks.doctor import main
    with pytest.raises(SystemExit):
        main(["--db", str(tmp_path / "nope.db"), "--set-model", "x"])
    assert not (tmp_path / "nope.db").exists()


def test_languages_section_lists_every_language(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    report = _report_for(tmp_path / "test.db")
    assert "== Languages ==" in report
    assert f"{'language':<12} {'refs':<5} {'literals':<9} {'macro':<6} {'pairing':<8} extensions" in report
    assert f"{'C++':<12} {'y':<5} {'y':<9} {'y':<6} {'y':<8} .cc .cpp .cu .cuh .cxx .h .hpp .hxx .inl .metal .mm" in report
    assert f"{'HLSL':<12} {'-':<5} {'-':<9} {'-':<6} {'-':<8} .fx .fxh .hlsl" in report
    assert report.endswith(
        "refs: typed calls/imports/inherits edges. literals: find_by_message. "
        "macro: C macro self-heal. pairing: header/impl pairing.\n"
    )
