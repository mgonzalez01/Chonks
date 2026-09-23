"""Tests for chonks/core/config.py."""
import json
import logging
from pathlib import Path

from chonks.core.config import load_config, resolve_codebase


def test_explicit_path_loads_and_turns_discovery_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"db": "decoy.db"}))
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"db": "explicit.db"}))
    loaded = load_config(str(explicit))
    assert loaded.data == {"db": "explicit.db"}
    assert loaded.path == explicit
    assert loaded.problems == ()


def test_discovery_order_first_parseable_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text("{ broken")
    (tmp_path / ".chonks.json").write_text(json.dumps({"db": "dot.db"}))
    loaded = load_config(None)
    assert loaded.data == {"db": "dot.db"}
    assert loaded.path == Path(".chonks.json")
    assert len(loaded.problems) == 1
    assert loaded.problems[0].path == "config.json"
    assert loaded.problems[0].missing is False


def test_discovery_finds_package_dir_candidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg_dir = tmp_path / "chonks"
    pkg_dir.mkdir()
    (pkg_dir / "config.json").write_text(json.dumps({"db": "pkg.db"}))
    loaded = load_config(None)
    assert loaded.data == {"db": "pkg.db"}
    assert loaded.path == Path("chonks/config.json")
    assert loaded.problems == ()


def test_nothing_found_returns_empty_without_problems(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loaded = load_config(None)
    assert loaded.data == {}
    assert loaded.path is None
    assert loaded.problems == ()


def test_explicit_missing_file_is_a_missing_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loaded = load_config("nope.json")
    assert loaded.data == {}
    assert loaded.path is None
    assert len(loaded.problems) == 1
    assert loaded.problems[0].path == "nope.json"
    assert loaded.problems[0].missing is True


def test_explicit_invalid_json_is_a_problem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bad.json").write_text("{ broken")
    loaded = load_config("bad.json")
    assert loaded.data == {}
    assert loaded.path is None
    assert len(loaded.problems) == 1
    assert loaded.problems[0].path == "bad.json"
    assert loaded.problems[0].missing is False


def test_reads_utf8(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_bytes('{"exclude": ["café/"]}'.encode("utf-8"))
    loaded = load_config(None)
    assert loaded.data == {"exclude": ["café/"]}
    assert loaded.problems == ()


def test_non_object_top_level_is_a_problem_and_discovery_moves_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(["db", "x.db"]))
    (tmp_path / ".chonks.json").write_text(json.dumps({"db": "dot.db"}))
    loaded = load_config(None)
    assert loaded.data == {"db": "dot.db"}
    assert len(loaded.problems) == 1
    assert loaded.problems[0].path == "config.json"
    assert loaded.problems[0].missing is False
    assert "object" in loaded.problems[0].reason


def test_explicit_non_object_top_level_returns_empty_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "explicit.json").write_text("42")
    loaded = load_config("explicit.json")
    assert loaded.data == {}
    assert loaded.path is None
    assert len(loaded.problems) == 1


def test_resolve_codebase_rejects_absolute_path_from_discovered_config(caplog):
    logger = logging.getLogger("test.chonks.core.config.discovered")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        result = resolve_codebase("/abs/path", explicit_config=False, logger=logger)
    assert result is None
    assert "Ignoring absolute 'codebase' path" in caplog.text


def test_resolve_codebase_accepts_absolute_path_from_explicit_config(tmp_path, caplog):
    logger = logging.getLogger("test.chonks.core.config.explicit")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        result = resolve_codebase(str(tmp_path), explicit_config=True, logger=logger)
    assert result == tmp_path.resolve()
    assert caplog.text == ""
