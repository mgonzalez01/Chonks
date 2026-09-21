#!/usr/bin/env python3
"""init.py: first-run setup wizard for Chonks, writes config.json.

`--auto-exclude` only takes effect together with `--yes`; in interactive
mode every suggested exclude is still confirmed one at a time regardless
of this flag.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from chonks.index.admission import DEFAULT_FALLBACK_EXTENSIONS
from chonks.embed.client import ping_embedder
from chonks.embed.client import RECOMMENDED_EMBED_MODEL

DEFAULT_EMBED_URL = "http://localhost:11437"
DEFAULT_DB_PATH = "./.db/chonks.db"
DEFAULT_CONFIG_PATH = "config.json"

# Junk-dir heuristic;
# Matched by exact basename, not substring, so e.g. "distribution/" is
# never swept up.
JUNK_DIR_NAMES = {
    "node_modules", ".venv", "venv", ".git", "__pycache__",
    "dist", "build", "target", ".nuget", "packages", ".next",
    "coverage", "DerivedData", "Intermediate", "Saved",
}


# --------------------------------------------------------------------------
# Pure logic: scanning, config building, serialization. No stdin/stdout.
# --------------------------------------------------------------------------

def expand_path(raw: str) -> Path:
    """Expand ~ and resolve to an absolute path (does not require existence)."""
    return Path(raw).expanduser().resolve()


def format_size(num_bytes: int) -> str:
    """Human-readable size, e.g. 1536 -> '1.5 KB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"  # unreachable, keeps type-checkers happy


def _dir_stats(path: Path) -> tuple[int, int]:
    """Total (file_count, size_bytes) under `path`, walked recursively.
    Best-effort: unreadable subpaths are skipped rather than raising."""
    count = 0
    size = 0
    for dirpath, _dirnames, filenames in os.walk(path, onerror=lambda e: None):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            count += 1
            try:
                size += fpath.stat().st_size
            except OSError:
                pass
    return count, size


def scan_junk_candidates(
    root: Path, junk_names: set[str] = JUNK_DIR_NAMES
) -> list[dict]:
    """Walk `root`, return one entry per junk-dir candidate:
    {"path": "<posix, relative to root>/", "file_count": int, "size_bytes": int}.
    Once matched, a candidate's subtree is not descended into further, so a
    nested duplicate (node_modules/ inside node_modules/) isn't double-counted.
    """
    candidates: list[dict] = []
    for dirpath, dirnames, _filenames in os.walk(root, onerror=lambda e: None):
        dir_path = Path(dirpath)
        keep = []
        for d in dirnames:
            if d in junk_names:
                child = dir_path / d
                count, size = _dir_stats(child)
                rel = child.relative_to(root).as_posix() + "/"
                candidates.append({
                    "path": rel,
                    "file_count": count,
                    "size_bytes": size,
                })
                # Don't descend: already fully accounted for above.
            else:
                keep.append(d)
        dirnames[:] = keep
    candidates.sort(key=lambda c: c["path"])
    return candidates


def build_config(
    *,
    codebase: str,
    db: str,
    embed_url: str,
    exclude: list[str],
    fallback_extensions: list[str] | None = None,
    embed_model: str = RECOMMENDED_EMBED_MODEL,
) -> dict:
    """Assemble the config.json dict. `embed_url` is the base server URL
    (e.g. http://localhost:11437); the /v1/embeddings suffix is appended
    here to match config.example.json's stored form."""
    base = embed_url.rstrip("/")
    if not base.endswith("/v1/embeddings"):
        base = base + "/v1/embeddings"
    config: dict = {
        "server_url": "http://localhost:11438",
        "db": db,
        "codebase": codebase,
        "embed_url": base,
        "embed_model": embed_model,
        "exclude": exclude,
    }
    if fallback_extensions is not None:
        config["fallback_extensions"] = fallback_extensions
    return config


def write_config(path: Path, config: dict, force: bool = False) -> tuple[bool, str]:
    """Write `config` as JSON to `path`. Refuses to clobber an existing file
    unless `force` is True. Returns (written, message)."""
    if path.exists() and not force:
        return False, f"{path} already exists — not overwritten (pass force to confirm)."
    path.write_text(json.dumps(config, indent=2) + "\n")
    return True, f"Wrote {path}"


def render_mcp_json(*, chonks_root: Path, db: Path, config: Path) -> str:
    """Project-scope `.mcp.json` block with absolute paths filled in;
    `render_mcp_add` is the command form Claude Code reads directly.
    `chonks_root` is the repo root, not the package dir itself.
    """
    server_py = chonks_root / "chonks" / "server.py"
    mcp_index = chonks_root / "mcp-server" / "dist" / "index.js"
    block = {
        "mcpServers": {
            "chonks": {
                "command": "node",
                "args": [str(mcp_index)],
                "env": {
                    "CHONKS_URL": "http://localhost:11438",
                    "CHONKS_SERVER_PY": str(server_py),
                    "CHONKS_DB": str(db.resolve()),
                    "CHONKS_CONFIG": str(config.resolve()),
                    "CHONKS_PY_CMD": "uv run python",
                },
            }
        }
    }
    return json.dumps(block, indent=2)


def render_mcp_add(*, chonks_root: Path, db: Path, config: Path, scope: str = "user") -> str:
    """The `claude mcp add` command for this install. Claude Code does not read
    an arbitrary JSON file; it reads what this command writes (~/.claude.json for
    user scope, .mcp.json at the project root for project scope)."""
    server_py = chonks_root / "chonks" / "server.py"
    mcp_index = chonks_root / "mcp-server" / "dist" / "index.js"
    lines = [
        f"claude mcp add chonks -s {scope} \\",
        "  -e CHONKS_URL=http://localhost:11438 \\",
        f"  -e CHONKS_SERVER_PY={server_py} \\",
        f"  -e CHONKS_DB={db.resolve()} \\",
        f"  -e CHONKS_CONFIG={config.resolve()} \\",
        "  -e \"CHONKS_PY_CMD=uv run python\" \\",
        f"  -- node {mcp_index}",
    ]
    return "\n".join(lines)


def gpu_install_hint(*, which_fn=shutil.which, cupy_importable: bool | None = None) -> str | None:
    """One line to print when an NVIDIA GPU is visible (nvidia-smi on PATH) but
    the CUDA k-NN extra is not installed. None otherwise. Pure for tests."""
    if not which_fn("nvidia-smi"):
        return None
    if cupy_importable is None:
        try:
            import cupy  # noqa: F401
            cupy_importable = True
        except Exception:  # noqa: BLE001
            cupy_importable = False
    if cupy_importable:
        return None
    return ("NVIDIA GPU detected but the CUDA k-NN extra is not installed. Run "
            "`uv sync --extra cuda`; the graph build then uses the GPU automatically "
            "(tens of minutes down to seconds on a large corpus).")


# --------------------------------------------------------------------------
# Prompt loop: thin wrappers over an injectable input function.
# --------------------------------------------------------------------------

def ask(input_fn, prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input_fn(f"{prompt}{suffix}: ").strip()
    return raw or (default or "")


def ask_yes_no(input_fn, prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input_fn(f"{prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


# --------------------------------------------------------------------------
# Wizard
# --------------------------------------------------------------------------

def run_wizard(argv: list[str] | None = None, input_fn=input, print_fn=print) -> int:
    parser = argparse.ArgumentParser(
        prog="chonks init",
        description="Interactive first-run setup wizard for Chonks — writes "
                     "config.json and prints the claude mcp add command.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Non-interactive: use flags/defaults, skip all prompts.",
    )
    parser.add_argument("--codebase", default=None, metavar="DIR")
    parser.add_argument("--db", default=None, metavar="FILE")
    parser.add_argument("--embed-url", default=None, metavar="URL")
    parser.add_argument(
        "--embed-model", default=None, metavar="NAME",
        help=f"Model name recorded in config.json (default: {RECOMMENDED_EMBED_MODEL}). "
             "The name selects the query/document instruction prefixes, so it must "
             "match the model the server actually runs.",
    )
    parser.add_argument(
        "--exclude", action="append", default=None, metavar="PREFIX",
        help="Exclude prefix to add (repeatable). Only used with --yes.",
    )
    parser.add_argument(
        "--auto-exclude", action="store_true",
        help="With --yes only: auto-apply scan-detected junk-dir suggestions "
             "instead of ignoring them. Has no effect in interactive mode — "
             "there every suggestion is confirmed one at a time regardless.",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, metavar="FILE",
        help=f"Where to write config.json (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing config file without prompting.",
    )
    args = parser.parse_args(argv)

    # __file__ is chonks/ops/init.py; parent.parent.parent is the repo root containing
    # the `chonks` package dir and the sibling `mcp-server/` project.
    chonks_root = Path(__file__).resolve().parent.parent.parent
    config_path = Path(args.config)

    if args.yes:
        return _run_noninteractive(args, config_path, chonks_root, print_fn)
    try:
        return _run_interactive(args, config_path, chonks_root, input_fn, print_fn)
    except (EOFError, KeyboardInterrupt):
        # No stdin (piped/CI invocation) or Ctrl-C mid-prompt: exit cleanly
        # instead of dumping a traceback. config.json may already exist if the
        # interrupt hit the post-write "run the first index?" prompt.
        print_fn("\nAborted. For scripted setup use: "
                 "chonks init --yes --codebase <dir> [--auto-exclude]")
        return 1


def _run_noninteractive(args, config_path: Path, chonks_root: Path, print_fn) -> int:
    if not args.codebase:
        print_fn("--yes requires --codebase")
        return 2
    codebase = expand_path(args.codebase)
    if not codebase.is_dir():
        print_fn(f"Codebase path does not exist or is not a directory: {codebase}")
        return 2

    exclude = list(args.exclude or [])
    if args.auto_exclude:
        for cand in scan_junk_candidates(codebase):
            if cand["path"] not in exclude:
                exclude.append(cand["path"])

    db = args.db or DEFAULT_DB_PATH
    embed_url = args.embed_url or DEFAULT_EMBED_URL
    embed_model = args.embed_model or RECOMMENDED_EMBED_MODEL

    config = build_config(
        codebase=str(codebase), db=db, embed_url=embed_url, exclude=exclude,
        embed_model=embed_model,
    )
    written, msg = write_config(config_path, config, force=args.force)
    print_fn(msg)
    if not written:
        return 1
    hint = gpu_install_hint()
    if hint:
        print_fn(hint)

    print_fn(render_mcp_add(chonks_root=chonks_root, db=Path(db), config=config_path))
    return 0


def _run_interactive(args, config_path: Path, chonks_root: Path, input_fn, print_fn) -> int:
    print_fn("Chonks setup\n" + "=" * 40)

    if config_path.exists() and not args.force:
        print_fn(f"\n{config_path} already exists.")
        if not ask_yes_no(input_fn, "Overwrite it?", default=False):
            print_fn("Aborted — existing config.json left untouched.")
            return 1

    # 1. codebase path + scan
    while True:
        raw = ask(input_fn, "\nPath to the codebase to index")
        if not raw:
            print_fn("A codebase path is required.")
            continue
        codebase = expand_path(raw)
        if not codebase.is_dir():
            print_fn(f"Not a directory: {codebase}")
            continue
        break

    print_fn(f"\nScanning {codebase} ...")
    candidates = scan_junk_candidates(codebase)
    exclude: list[str] = []
    if candidates:
        print_fn(f"Found {len(candidates)} candidate director{'y' if len(candidates)==1 else 'ies'} "
                  f"to consider excluding:\n")
        for cand in candidates:
            evidence = f"{cand['file_count']} files, {format_size(cand['size_bytes'])}"
            if ask_yes_no(input_fn, f"  Exclude {cand['path']} ({evidence})?", default=True):
                exclude.append(cand["path"])
    else:
        print_fn("No junk-dir candidates detected by name heuristics.")

    while True:
        extra = ask(input_fn, "\nAny additional exclude prefixes? (comma-separated, blank to skip)")
        if not extra:
            break
        for p in extra.split(","):
            p = p.strip()
            if p:
                exclude.append(p if p.endswith("/") else p + "/")
        break

    # 2. fallback extensions
    print_fn(f"\nDefault fallback extensions (indexed via line-based chunking, "
              f"no AST grammar): {', '.join(DEFAULT_FALLBACK_EXTENSIONS)}")
    fallback_extensions = DEFAULT_FALLBACK_EXTENSIONS
    if not ask_yes_no(input_fn, "Use this list?", default=True):
        raw = ask(input_fn, "Enter comma-separated extensions (with leading dots)", default="")
        fallback_extensions = [e.strip() for e in raw.split(",") if e.strip()]

    db = ask(input_fn, "\nDB path", default=DEFAULT_DB_PATH)

    # embed url + live ping
    embed_url = ask(input_fn, "Embedding server URL", default=DEFAULT_EMBED_URL)
    embed_model = ask(
        input_fn, "Embedding model name (selects the query/document prefixes; "
        "must match what the server runs)", default=RECOMMENDED_EMBED_MODEL,
    )
    print_fn(f"Pinging {embed_url} ...")
    result = ping_embedder(embed_url, model=embed_model)
    if result["reachable"]:
        print_fn(f"  Reachable. Embedding dim: {result['dim']}")
    else:
        print_fn(f"  WARNING: could not reach embedding server ({result['error']}).")
        print_fn("  You can continue setup, but indexing will fail until it's up.")
    hint = gpu_install_hint()
    if hint:
        print_fn(f"  {hint}")

    # 3. write config
    config = build_config(
        codebase=str(codebase), db=db, embed_url=embed_url,
        exclude=exclude, fallback_extensions=fallback_extensions,
        embed_model=embed_model,
    )
    written, msg = write_config(config_path, config, force=True)
    print_fn(f"\n{msg}")

    print_fn("\nRegister with Claude Code (one line per flag; PowerShell users replace the trailing \\ with `):\n")
    print_fn(render_mcp_add(chonks_root=chonks_root, db=Path(db), config=config_path))
    print_fn("\nThen /mcp inside Claude Code should list chonks. For a project-scoped .mcp.json instead, use -s project.")

    if ask_yes_no(input_fn, "\nRun the first index now?", default=False):
        import subprocess
        cmd = [
            sys.executable, "-m", "chonks.ops.index_cmd",
            str(codebase), "--db", db, "--config", str(config_path),
        ]
        print_fn(f"\n$ {' '.join(cmd)}")
        subprocess.run(cmd)

    return 0


def main(argv: list[str] | None = None) -> int:
    return run_wizard(argv)


if __name__ == "__main__":
    sys.exit(run_wizard())
