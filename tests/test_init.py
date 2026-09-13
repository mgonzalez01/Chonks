"""The interactive prompt loop is exercised via an injected `input_fn` list
rather than real stdin, and the embedder ping is stubbed via `post_fn`, so
no test here touches a live embedder.
"""
import json

from chonks.init import (
    JUNK_DIR_NAMES,
    ask,
    ask_yes_no,
    build_config,
    format_size,
    ping_embedder,
    render_mcp_add,
    render_mcp_json,
    run_wizard,
    scan_junk_candidates,
    write_config,
)


# ---- format_size ------------------------------------------------------

def test_format_size_bytes():
    assert format_size(500) == "500 B"


def test_format_size_kb_mb():
    assert format_size(2048) == "2.0 KB"
    assert format_size(5 * 1024 * 1024) == "5.0 MB"


# ---- scan_junk_candidates ----------------------------------------------

def test_scan_detects_known_junk_dirs_with_evidence(tmp_path):
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "a.js").write_text("x" * 10)
    (tmp_path / "node_modules" / "pkg" / "b.js").write_text("y" * 20)
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "c.py").write_text("z" * 5)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("real code")

    candidates = scan_junk_candidates(tmp_path)
    paths = {c["path"] for c in candidates}
    assert "node_modules/" in paths
    assert ".venv/" in paths
    assert "src/" not in paths

    nm = next(c for c in candidates if c["path"] == "node_modules/")
    assert nm["file_count"] == 2
    assert nm["size_bytes"] == 30


def test_scan_does_not_descend_into_matched_junk_dir(tmp_path):
    nested = tmp_path / "node_modules" / "some_pkg" / "node_modules"
    nested.mkdir(parents=True)
    (nested / "f.js").write_text("x")

    candidates = scan_junk_candidates(tmp_path)
    assert len(candidates) == 1
    assert candidates[0]["path"] == "node_modules/"
    assert candidates[0]["file_count"] == 1


def test_scan_ignores_dirs_not_in_junk_names(tmp_path):
    (tmp_path / "vendor_but_legit").mkdir()
    (tmp_path / "vendor_but_legit" / "f.txt").write_text("keep me")
    candidates = scan_junk_candidates(tmp_path)
    assert candidates == []


def test_scan_finds_junk_dir_at_any_depth(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "build"
    deep.mkdir(parents=True)
    (deep / "out.o").write_text("obj")
    candidates = scan_junk_candidates(tmp_path)
    assert candidates == [{"path": "a/b/c/build/", "file_count": 1, "size_bytes": 3}]


def test_junk_dir_names_include_game_engine_dirs():
    for name in ("DerivedData", "Intermediate", "Saved", ".nuget", "packages"):
        assert name in JUNK_DIR_NAMES


# ---- build_config / write_config ----------------------------------------

def test_build_config_appends_v1_embeddings_suffix():
    config = build_config(
        codebase="/abs/code", db="./.db/chonks.db",
        embed_url="http://localhost:11437", exclude=["node_modules/"],
    )
    assert config["embed_url"] == "http://localhost:11437/v1/embeddings"
    assert config["codebase"] == "/abs/code"
    assert config["exclude"] == ["node_modules/"]


def test_build_config_does_not_double_append_suffix():
    config = build_config(
        codebase="/x", db="d", embed_url="http://host:1/v1/embeddings", exclude=[],
    )
    assert config["embed_url"] == "http://host:1/v1/embeddings"


def test_build_config_fallback_extensions_omitted_when_none():
    config = build_config(codebase="/x", db="d", embed_url="http://h", exclude=[])
    assert "fallback_extensions" not in config


def test_build_config_fallback_extensions_included_when_given():
    config = build_config(
        codebase="/x", db="d", embed_url="http://h", exclude=[],
        fallback_extensions=[".md"],
    )
    assert config["fallback_extensions"] == [".md"]


def test_write_config_creates_valid_json(tmp_path):
    path = tmp_path / "config.json"
    config = build_config(codebase="/x", db="d", embed_url="http://h", exclude=[])
    written, msg = write_config(path, config)
    assert written
    assert json.loads(path.read_text()) == config


def test_write_config_refuses_to_clobber_without_force(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"existing": true}')
    written, msg = write_config(path, {"new": True}, force=False)
    assert not written
    assert "already exists" in msg
    assert json.loads(path.read_text()) == {"existing": True}


def test_write_config_overwrites_with_force(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"existing": true}')
    written, msg = write_config(path, {"new": True}, force=True)
    assert written
    assert json.loads(path.read_text()) == {"new": True}


# ---- render_mcp_json ------------------------------------------------------

def test_render_mcp_json_fills_absolute_paths(tmp_path):
    chonks_root = tmp_path / "Chonks"
    chonks_root.mkdir()
    db = tmp_path / "db.sqlite"
    config = tmp_path / "config.json"
    out = json.loads(render_mcp_json(chonks_root=chonks_root, db=db, config=config))
    env = out["mcpServers"]["chonks"]["env"]
    assert env["CHONKS_DB"] == str(db.resolve())
    assert env["CHONKS_CONFIG"] == str(config.resolve())
    assert env["CHONKS_SERVER_PY"] == str(chonks_root / "chonks" / "server.py")
    assert out["mcpServers"]["chonks"]["args"] == [
        str(chonks_root / "mcp-server" / "dist" / "index.js")
    ]


# ---- ping_embedder (stubbed, no live network) ---------------------------

class _FakeResp:
    def __init__(self, payload, status_ok=True):
        self._p = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("bad status")

    def json(self):
        return self._p


def test_ping_embedder_reachable_reports_dim():
    def fake_post(url, body, timeout):
        assert url.endswith("/v1/embeddings")
        return _FakeResp({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})

    result = ping_embedder("http://localhost:11437", post_fn=fake_post)
    assert result == {"reachable": True, "dim": 3, "error": None}


def test_ping_embedder_unreachable_reports_error():
    def fake_post(url, body, timeout):
        raise ConnectionError("connection refused")

    result = ping_embedder("http://localhost:11437", post_fn=fake_post)
    assert result["reachable"] is False
    assert result["dim"] is None
    assert "connection refused" in result["error"]


def test_ping_embedder_http_error_status():
    def fake_post(url, body, timeout):
        return _FakeResp({}, status_ok=False)

    result = ping_embedder("http://localhost:11437", post_fn=fake_post)
    assert result["reachable"] is False


# ---- ask / ask_yes_no prompt helpers --------------------------------------

def test_ask_returns_default_on_blank():
    inputs = iter([""])
    assert ask(lambda p: next(inputs), "prompt", default="fallback") == "fallback"


def test_ask_returns_typed_value():
    inputs = iter(["typed"])
    assert ask(lambda p: next(inputs), "prompt", default="fallback") == "typed"


def test_ask_yes_no_defaults():
    inputs = iter([""])
    assert ask_yes_no(lambda p: next(inputs), "prompt", default=True) is True


def test_ask_yes_no_explicit_no():
    inputs = iter(["n"])
    assert ask_yes_no(lambda p: next(inputs), "prompt", default=True) is False


# ---- run_wizard: --yes non-interactive path -------------------------------

def test_yes_mode_requires_codebase(capsys):
    rc = run_wizard(["--yes"], print_fn=lambda *a: None)
    assert rc == 2


def test_yes_mode_rejects_missing_codebase_dir(tmp_path):
    rc = run_wizard(
        ["--yes", "--codebase", str(tmp_path / "nope"),
         "--config", str(tmp_path / "config.json")],
        print_fn=lambda *a: None,
    )
    assert rc == 2


def test_yes_mode_writes_config_with_defaults(tmp_path):
    codebase = tmp_path / "code"
    codebase.mkdir()
    config_path = tmp_path / "config.json"
    rc = run_wizard(
        ["--yes", "--codebase", str(codebase), "--config", str(config_path)],
        print_fn=lambda *a: None,
    )
    assert rc == 0
    config = json.loads(config_path.read_text())
    assert config["codebase"] == str(codebase)
    assert config["exclude"] == []


def test_yes_mode_auto_exclude_applies_scan_suggestions(tmp_path):
    codebase = tmp_path / "code"
    (codebase / "node_modules").mkdir(parents=True)
    (codebase / "node_modules" / "f.js").write_text("x")
    config_path = tmp_path / "config.json"

    rc = run_wizard(
        ["--yes", "--codebase", str(codebase), "--config", str(config_path),
         "--auto-exclude"],
        print_fn=lambda *a: None,
    )
    assert rc == 0
    config = json.loads(config_path.read_text())
    assert "node_modules/" in config["exclude"]


def test_yes_mode_manual_exclude_flags_used_without_auto_exclude(tmp_path):
    codebase = tmp_path / "code"
    (codebase / "node_modules").mkdir(parents=True)
    (codebase / "node_modules" / "f.js").write_text("x")
    config_path = tmp_path / "config.json"

    rc = run_wizard(
        ["--yes", "--codebase", str(codebase), "--config", str(config_path),
         "--exclude", "custom/"],
        print_fn=lambda *a: None,
    )
    assert rc == 0
    config = json.loads(config_path.read_text())
    assert config["exclude"] == ["custom/"]


def test_yes_mode_respects_clobber_protection(tmp_path):
    codebase = tmp_path / "code"
    codebase.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text('{"existing": true}')

    rc = run_wizard(
        ["--yes", "--codebase", str(codebase), "--config", str(config_path)],
        print_fn=lambda *a: None,
    )
    assert rc == 1
    assert json.loads(config_path.read_text()) == {"existing": True}


def test_yes_mode_force_overwrites_existing_config(tmp_path):
    codebase = tmp_path / "code"
    codebase.mkdir()
    config_path = tmp_path / "config.json"
    config_path.write_text('{"existing": true}')

    rc = run_wizard(
        ["--yes", "--codebase", str(codebase), "--config", str(config_path), "--force"],
        print_fn=lambda *a: None,
    )
    assert rc == 0
    config = json.loads(config_path.read_text())
    assert config["codebase"] == str(codebase)


# ---- run_wizard: interactive path via injected input_fn -------------------

def test_interactive_flow_writes_config_end_to_end(tmp_path):
    codebase = tmp_path / "code"
    (codebase / "node_modules").mkdir(parents=True)
    (codebase / "node_modules" / "f.js").write_text("x")
    config_path = tmp_path / "config.json"

    answers = iter([
        str(codebase),   # codebase path
        "y",              # exclude node_modules/? yes
        "",               # extra excludes -> blank
        "y",              # use default fallback extensions
        "",               # db path -> default
        "http://localhost:11437",  # embed url
        "",               # embed model -> default (jina-code)
        "",               # run first index now? -> default (no)
    ])

    def fake_input(prompt):
        return next(answers)

    lines = []
    rc = run_wizard(
        ["--config", str(config_path)],
        input_fn=fake_input,
        print_fn=lambda *a: lines.append(" ".join(str(x) for x in a)),
    )
    assert rc == 0
    config = json.loads(config_path.read_text())
    assert config["codebase"] == str(codebase)
    assert "node_modules/" in config["exclude"]
    assert config["embed_url"] == "http://localhost:11437/v1/embeddings"


def test_interactive_flow_aborts_on_existing_config_declined(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"existing": true}')

    answers = iter(["n"])  # decline overwrite

    rc = run_wizard(
        ["--config", str(config_path)],
        input_fn=lambda p: next(answers),
        print_fn=lambda *a: None,
    )
    assert rc == 1
    assert json.loads(config_path.read_text()) == {"existing": True}


# ---- embed_model: the name selects the prefix preset ----------------------

def test_build_config_defaults_to_recommended_model():
    from chonks.embedder import RECOMMENDED_EMBED_MODEL, matched_prefix_preset
    config = build_config(codebase="/x", db="d", embed_url="http://h", exclude=[])
    assert config["embed_model"] == RECOMMENDED_EMBED_MODEL
    assert matched_prefix_preset(config["embed_model"]) == "jina-code"


def test_noninteractive_embed_model_flag(tmp_path):
    codebase = tmp_path / "code"; codebase.mkdir()
    config_path = tmp_path / "config.json"
    rc = run_wizard(
        ["--yes", "--codebase", str(codebase), "--config", str(config_path),
         "--embed-model", "qwen3-embedding"],
        print_fn=lambda *a: None,
    )
    assert rc == 0
    assert json.loads(config_path.read_text())["embed_model"] == "qwen3-embedding"


def test_ping_embedder_sends_configured_model():
    seen = {}
    def fake_post(url, body, timeout):
        seen.update(body)
        class R:
            def raise_for_status(self): pass
            def json(self): return {"data": [{"embedding": [0.0] * 4}]}
        return R()
    ping_embedder("http://h", post_fn=fake_post, model="my-model")
    assert seen["model"] == "my-model"


# ---- GPU install hint (pure) ----------------------------------------------

def test_gpu_hint_only_when_nvidia_smi_present_and_cupy_missing():
    from chonks.init import gpu_install_hint
    assert gpu_install_hint(which_fn=lambda n: None, cupy_importable=False) is None
    assert gpu_install_hint(which_fn=lambda n: "/usr/bin/nvidia-smi", cupy_importable=True) is None
    hint = gpu_install_hint(which_fn=lambda n: "/usr/bin/nvidia-smi", cupy_importable=False)
    assert hint and "uv sync --extra cuda" in hint


def test_render_mcp_add_is_one_command_with_absolute_paths(tmp_path):
    chonks_root = tmp_path / "Chonks"
    chonks_root.mkdir()
    db = tmp_path / "db.sqlite"
    config = tmp_path / "config.json"
    cmd = render_mcp_add(chonks_root=chonks_root, db=db, config=config)
    assert cmd.startswith("claude mcp add chonks -s user")
    assert f"CHONKS_DB={db.resolve()}" in cmd
    assert f"CHONKS_CONFIG={config.resolve()}" in cmd
    assert cmd.rstrip().endswith(f"-- node {chonks_root / 'mcp-server' / 'dist' / 'index.js'}")
