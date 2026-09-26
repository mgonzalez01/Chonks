import json
import subprocess
import sys
from pathlib import Path

from chonks.ops.report import build_report
from chonks.storage.store import Store


def _fake_embedding(dim: int = 4) -> list[float]:
    return [0.5] * dim


def _chunk(id, path, name, language, chunk_type="function_definition", s=1, e=5):
    return {"id": id, "path": path, "language": language, "chunk_type": chunk_type,
            "name": name, "start_line": s, "end_line": e, "content": f"{name}()"}


def _build_synthetic_store(tmp_path) -> Store:
    # Spans two configured subsystems (core/, render/) plus one un-configured
    # top-level dir (plugin/) so both grouping paths get exercised.
    store = Store(tmp_path / "test.db")
    chunks = [
        _chunk("engine",  "core/engine.py",   "Engine",     "python"),
        _chunk("helper",  "core/utils.py",    "Helper",     "python"),
        _chunk("shader",  "render/shader.cpp", "Shader",     "cpp"),
        _chunk("pass_",   "render/pass.cpp",  "RenderPass", "cpp"),
        _chunk("bridge",  "plugin/bridge.py", "Bridge",     "python"),
    ]
    store.insert_chunks(chunks, [_fake_embedding() for _ in chunks])
    store.insert_refs([
        ("engine", "helper", "calls"),      # same subsystem (core -> core)
        ("engine", "shader", "calls"),      # cross-subsystem (core -> render)
        ("bridge", "shader", "xlang"),      # cross-language, both legs (as
        ("shader", "bridge", "xlang"),      # index.graph.refs._build_graph writes xlang symmetrically)
        ("shader", "pass_",  "mentions"),   # same subsystem, mentions-only (should be excluded)
        ("helper", "shader", "associated"), # cross-subsystem, PMI-associated (should be excluded)
    ])
    store.upsert_file("core/engine.py", 10, 0.0, "h1")
    store.upsert_file("core/utils.py", 10, 0.0, "h2")
    store.upsert_file("render/shader.cpp", 10, 0.0, "h3")
    store.upsert_file("render/pass.cpp", 10, 0.0, "h4")
    store.upsert_file("plugin/bridge.py", 10, 0.0, "h5")
    store.upsert_file("plugin/unparsed.dat", 10, 0.0, "h6")  # discovered, never chunked
    store.set_meta("indexed_root", "/repo/example")
    store.commit()
    return store


CONFIG = {"codebase": "example", "subsystems": {"core": ["core/"], "render": ["render/"]}}


def test_sections_present(tmp_path):
    store = _build_synthetic_store(tmp_path)
    try:
        report = build_report(store, CONFIG, str(tmp_path / "test.db"))
    finally:
        store.close()

    for heading in (
        "# INDEX_REPORT: example",
        "## God nodes",
        "### Per subsystem",
        "## Surprising connections",
        "### Cross-language edges",
        "### Cross-subsystem edges",
        "## Suggested questions",
        "## Index vitals",
    ):
        assert heading in report, f"missing section: {heading}"


def test_god_nodes_content(tmp_path):
    store = _build_synthetic_store(tmp_path)
    try:
        report = build_report(store, CONFIG, str(tmp_path / "test.db"))
    finally:
        store.close()

    god_section = report.split("## Surprising connections")[0]
    # Engine has the most out-edges, Shader the most in-edges.
    assert "**Engine**" in god_section
    assert "**Shader**" in god_section
    assert "#### core" in god_section
    assert "#### render" in god_section


def test_cross_language_edge_detected(tmp_path):
    store = _build_synthetic_store(tmp_path)
    try:
        report = build_report(store, CONFIG, str(tmp_path / "test.db"))
    finally:
        store.close()

    xlang_section = report.split("### Cross-language edges")[1].split("### Cross-subsystem edges")[0]
    assert "Bridge" in xlang_section
    assert "Shader" in xlang_section


def test_cross_subsystem_edge_detected_and_mentions_excluded(tmp_path):
    store = _build_synthetic_store(tmp_path)
    try:
        report = build_report(store, CONFIG, str(tmp_path / "test.db"))
    finally:
        store.close()

    cross_section = report.split("### Cross-subsystem edges")[1].split("## Suggested questions")[0]
    assert "Engine" in cross_section and "Shader" in cross_section and "calls" in cross_section
    # Shader -> RenderPass is same-subsystem and mentions-only: excluded.
    assert "RenderPass" not in cross_section
    # Helper -> Shader is 'associated' (PMI-selected mentions, not a real
    # dependency): same exclusion rule as plain mentions.
    assert "Helper" not in cross_section


def test_suggested_questions_reference_real_symbols(tmp_path):
    store = _build_synthetic_store(tmp_path)
    try:
        report = build_report(store, CONFIG, str(tmp_path / "test.db"))
    finally:
        store.close()

    q_section = report.split("## Suggested questions")[1].split("## Index vitals")[0]
    assert "Engine" in q_section
    assert "render" in q_section


def test_vitals_counts(tmp_path):
    store = _build_synthetic_store(tmp_path)
    try:
        report = build_report(store, CONFIG, str(tmp_path / "test.db"))
    finally:
        store.close()

    vitals = report.split("## Index vitals")[1]
    assert "**Files**: 6" in vitals
    assert "**Chunks**: 5" in vitals
    assert "5/6 files produced at least one chunk" in vitals


def test_deterministic_byte_identical(tmp_path):
    db_path = tmp_path / "test.db"
    store = _build_synthetic_store(tmp_path)
    store.close()

    store1 = Store(db_path)
    report1 = build_report(store1, CONFIG, str(db_path))
    store1.close()

    store2 = Store(db_path)
    report2 = build_report(store2, CONFIG, str(db_path))
    store2.close()

    assert report1 == report2


def test_no_config_falls_back_to_path_grouping(tmp_path):
    # With no `subsystems` in config, grouping falls back to each chunk's
    # top-level path segment; the per-subsystem god-node breakdown is omitted.
    store = _build_synthetic_store(tmp_path)
    try:
        report = build_report(store, {}, str(tmp_path / "test.db"))
    finally:
        store.close()

    assert "### Per subsystem" not in report
    assert "## Surprising connections" in report
    cross_section = report.split("### Cross-subsystem edges")[1].split("## Suggested questions")[0]
    assert "Engine" in cross_section and "Shader" in cross_section


def test_cli_writes_report(tmp_path):
    store = _build_synthetic_store(tmp_path)
    store.close()

    out_path = tmp_path / "INDEX_REPORT.md"
    result = subprocess.run(
        [sys.executable, "-m", "chonks.ops.report",
         "--db", str(tmp_path / "test.db"), "-o", str(out_path)],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("# INDEX_REPORT:")
    assert "## Index vitals" in text


def test_main_discovers_dot_chonks_json(tmp_path, monkeypatch):
    from chonks.ops.report import main

    store = _build_synthetic_store(tmp_path)
    store.close()
    (tmp_path / ".chonks.json").write_text(json.dumps({"db": str(tmp_path / "test.db")}))
    monkeypatch.chdir(tmp_path)

    out_path = tmp_path / "R.md"
    main(["-o", str(out_path)])
    text = out_path.read_text(encoding="utf-8")
    assert text.startswith("# INDEX_REPORT:")
    assert "## Index vitals" in text


def test_main_unreadable_config_warns_and_continues(tmp_path, monkeypatch, capsys):
    import pytest
    from chonks.ops.report import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text("{ broken")

    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    # the unreadable config candidate is now a warning, not a hard stop:
    # the run continues to the next requirement (a missing --db), so the
    # config gate no longer aborts a run that doesn't need that file.
    assert "Failed to parse config.json" in err
    assert "config not readable" not in err
    assert "--db required" in err
