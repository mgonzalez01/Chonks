"""doctor.py: one-shot, read-only index health report for `chonks doctor`.

Deliberately doesn't open the DB through `store.Store`: Store's __init__
raises on a schema-version mismatch, which is exactly the condition doctor
exists to report. Talks to the DB directly over a read-only connection.
"""
import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path

from chonks.core.config import load_config
from chonks.core.paths import _dir_should_prune, _normalize_prefixes, _path_allowed, _to_stored_path
from chonks.index.plugins import load_plugins
from chonks.index.segment import CHUNKER_VERSION
from chonks.languages import describe as _describe_languages
from chonks.languages import language_set as _language_set
from chonks.ops.diagnostics import (
    dominance_warning,
    dotdir_breakdown,
    dotdir_total_share,
    family_breakdown,
)
from chonks.storage.readonly import _connect_readonly, set_embedding_model
from chonks.storage.schema import SCHEMA_DDL
from chonks.storage.schema import SCHEMA_VERSION

# The tables of the schema that are not virtual, in schema order. An fts5 table
# and chunk_vecs have no dbstat row under their own name, so they are not here.
_KNOWN_TABLES = re.findall(r"^\s*CREATE TABLE IF NOT EXISTS (\w+)", SCHEMA_DDL, re.M)


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return None  # no `meta` table at all: pre-dates this schema entirely
    return row["value"] if row else None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?", (name,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _schema_section(conn: sqlite3.Connection) -> str:
    stored = _get_meta(conn, "schema_version")
    lines = ["== Schema =="]
    if stored is None:
        lines.append("schema_version: NOT SET (pre-versioning DB or empty file)")
    else:
        try:
            stored_int = int(stored)
        except ValueError:
            lines.append(
                f"!! schema_version is corrupted (non-numeric value {stored!r}) — "
                f"drop the DB and re-index."
            )
            return "\n".join(lines)
        if stored_int != SCHEMA_VERSION:
            lines.append(
                f"!! SCHEMA MISMATCH: DB has version {stored}, code expects {SCHEMA_VERSION}. "
                f"Drop the DB and re-index."
            )
        else:
            lines.append(f"schema_version: {stored} (matches code)")
    return "\n".join(lines)


def _chunker_version_section(conn: sqlite3.Connection) -> str:
    stored = _get_meta(conn, "chunker_version")
    lines = ["== Chunker version =="]
    if stored is None:
        lines.append(
            "chunker_version: not recorded in this DB "
            "(indexed before chunker_version provenance was added)"
        )
    elif stored.startswith("mixed:"):
        lines.append(
            f"chunker_version: {stored} — this DB has a mix of chunk-boundary "
            f"conventions; a --force re-index will make it consistent again"
        )
    elif stored != str(CHUNKER_VERSION):
        lines.append(
            f"chunker_version: {stored} (code is at {CHUNKER_VERSION} — "
            f"chunk boundaries may differ from a fresh index; re-index to refresh)"
        )
    else:
        lines.append(f"chunker_version: {stored} (matches code)")

    stored_language_set = _get_meta(conn, "language_set")
    if stored_language_set is None:
        lines.append(
            "language_set: not recorded in this DB "
            "(indexed before language_set provenance was added)"
        )
    elif stored_language_set.startswith("mixed: "):
        lines.append(
            f"language_set: {stored_language_set} — this DB has a mix of "
            f"chunk-boundary conventions; a --force re-index will make it "
            f"consistent again"
        )
    else:
        current_language_set = json.dumps(_language_set(), sort_keys=True, separators=(",", ":"))
        if stored_language_set != current_language_set:
            lines.append(
                f"language_set: {stored_language_set} (code is at {current_language_set} — "
                f"chunk boundaries may differ from a fresh index; re-index to refresh)"
            )
        else:
            lines.append(f"language_set: {stored_language_set} (matches code)")
    return "\n".join(lines)


def _vitals_section(conn: sqlite3.Connection, db_path: str) -> str:
    files  = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    size_bytes = Path(db_path).stat().st_size if Path(db_path).exists() else 0
    newest = conn.execute("SELECT MAX(indexed_at) AS t FROM files").fetchone()["t"]
    dim    = _get_meta(conn, "embedding_dim")
    model  = _get_meta(conn, "embedding_model")
    root   = _get_meta(conn, "indexed_root")
    newest_str = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest)) if newest else "never"
    )
    return "\n".join([
        "== Vitals ==",
        f"files:            {files}",
        f"chunks:           {chunks}",
        f"db size:          {size_bytes / 1024 / 1024:.2f} MB",
        f"embedding dim:    {dim or 'unset'}",
        f"embedding model:  {model or 'unset'}",
        f"indexed root:     {root or 'unset'}",
        f"newest indexed:   {newest_str}",
    ])


def _staleness_section(conn: sqlite3.Connection, config: dict) -> str:
    """Compares newest `files.indexed_at` against newest on-disk mtime via one
    os.walk at doctor-run time, not a watcher. Reuses the indexer's own
    persisted exclude/include so ignored dirs (`.git/` always) don't
    manufacture a false STALE."""
    newest_indexed = conn.execute("SELECT MAX(indexed_at) AS t FROM files").fetchone()["t"]
    root = _get_meta(conn, "indexed_root") or config.get("codebase")
    lines = ["== Staleness =="]
    if not root:
        lines.append("indexed_root not recorded and no `codebase` in config — cannot check.")
        return "\n".join(lines)
    root_path = Path(root)
    if not root_path.is_dir():
        lines.append(f"indexed_root '{root}' not found on this machine — cannot check.")
        return "\n".join(lines)
    if newest_indexed is None:
        lines.append("DB has no indexed files yet — nothing to compare.")
        return "\n".join(lines)

    exclude_raw = _get_meta(conn, "exclude")
    include_raw = _get_meta(conn, "include")
    exclude_list = json.loads(exclude_raw) if exclude_raw else list(config.get("exclude") or [])
    include_list = json.loads(include_raw) if include_raw else list(config.get("include") or [])
    excludes = _normalize_prefixes(exclude_list)
    includes = _normalize_prefixes(include_list)

    newest_mtime = 0.0
    newest_mtime_path = ""
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        if excludes:
            dir_path = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if not _dir_should_prune(_to_stored_path(dir_path / d, root_path), excludes, includes)
            ]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if excludes and not _path_allowed(_to_stored_path(fpath, root_path), excludes, includes):
                continue
            try:
                mtime = fpath.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_mtime_path = str(fpath)

    lines.append(f"newest indexed_at:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest_indexed))}")
    if newest_mtime == 0.0:
        lines.append("no files found under indexed_root on disk.")
    else:
        lines.append(f"newest on-disk mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest_mtime))} ({newest_mtime_path})")
        if newest_mtime > newest_indexed:
            lines.append(
                "!! STALE: files on disk are newer than the index — re-index to pick up changes."
            )
        else:
            lines.append("OK: index is at least as new as everything on disk.")
    return "\n".join(lines)


_FAMILY_TOP_N = 10  # display cap; remainder rolls into a single "other" row


def _path_family_section(conn: sqlite3.Connection) -> str:
    """Chunk-share by top-level path family. Warns if one family dominates
    the index; see chonks.ops.diagnostics.family_breakdown / dominance_warning for
    the threshold."""
    rows = conn.execute("SELECT path, language FROM chunks").fetchall()
    total = len(rows)
    lines = ["== Path families =="]
    if total == 0:
        lines.append("no chunks in this DB.")
        return "\n".join(lines)

    stats = family_breakdown((r["path"], r["language"]) for r in rows)
    header = f"{'family':<24} {'chunks':>8} {'share':>7} {'files':>7} {'chunks/file':>12} {'docs-share':>11}"
    lines.append(header)

    shown, rest = stats[:_FAMILY_TOP_N], stats[_FAMILY_TOP_N:]
    for s in shown:
        share = s.chunks / total
        lines.append(
            f"{s.family:<24} {s.chunks:>8} {share:>6.0%} {s.files:>7} "
            f"{s.chunks_per_file:>12.1f} {s.docs_share:>10.0%}"
        )
    if rest:
        other_chunks = sum(s.chunks for s in rest)
        other_files = sum(s.files for s in rest)
        other_docs = sum(s.docs_chunks for s in rest)
        other_share = other_chunks / total
        other_cpf = other_chunks / other_files if other_files else 0.0
        other_docs_share = other_docs / other_chunks if other_chunks else 0.0
        lines.append(
            f"{'other (' + str(len(rest)) + ' families)':<24} {other_chunks:>8} {other_share:>6.0%} "
            f"{other_files:>7} {other_cpf:>12.1f} {other_docs_share:>10.0%}"
        )

    warning = dominance_warning(stats, total)
    if warning:
        lines.append("")
        lines.append(warning)
    return "\n".join(lines)


_DOTDIR_TOP_N = 5  # display cap; remainder rolls into a single "and N more" line


def _dotdir_section(conn: sqlite3.Connection) -> str:
    """% of the index under dot-directories, plus top offenders by chunk
    count. Visibility only; never changes what gets indexed. Grouping rule:
    chonks.ops.diagnostics.dotdir_breakdown."""
    rows = conn.execute("SELECT path, language FROM chunks").fetchall()
    total = len(rows)
    lines = ["== Dot-directories =="]
    if total == 0:
        lines.append("no chunks in this DB.")
        return "\n".join(lines)

    stats = dotdir_breakdown((r["path"], r["language"]) for r in rows)
    if not stats:
        lines.append("no chunks under dot-directories.")
        return "\n".join(lines)

    dotdir_chunks = sum(s.chunks for s in stats)
    share = dotdir_total_share(stats, total)
    lines.append(f"{dotdir_chunks}/{total} chunks ({share:.0%}) live under a dot-directory.")
    lines.append("")
    lines.append(f"{'prefix':<32} {'chunks':>8} {'files':>7}")
    shown, rest = stats[:_DOTDIR_TOP_N], stats[_DOTDIR_TOP_N:]
    for s in shown:
        lines.append(f"{s.prefix:<32} {s.chunks:>8} {s.files:>7}")
    if rest:
        lines.append(f"... and {len(rest)} more")

    lines.append("")
    lines.append(
        "WARNING: dot-directory content is often a nested repo copy, a "
        "venv, or editor/agent scratch space, not source — if this isn't "
        'intended, add an exclude in config.json, e.g. "exclude": '
        f'["{stats[0].prefix}/"]'
    )
    return "\n".join(lines)


def _edges_section(conn: sqlite3.Connection) -> str:
    lines = ["== Edges =="]
    if _table_exists(conn, "chunk_refs"):
        rows = conn.execute(
            "SELECT edge_type, COUNT(*) AS n FROM chunk_refs GROUP BY edge_type ORDER BY edge_type"
        ).fetchall()
        if rows:
            for r in rows:
                lines.append(f"chunk_refs[{r['edge_type']}]: {r['n']}")
        else:
            lines.append("chunk_refs: 0 (empty)")
    else:
        lines.append("chunk_refs: table not present")

    if _table_exists(conn, "chunk_neighbors"):
        n = conn.execute("SELECT COUNT(*) FROM chunk_neighbors").fetchone()[0]
        lines.append(f"chunk_neighbors: {n}")
    else:
        lines.append("chunk_neighbors: table not present")
    return "\n".join(lines)


def _parse_health_section(conn: sqlite3.Connection) -> str:
    """Parse-health diagnostics are computed per-run in chonks/index/pipeline.py but not
    persisted in the DB, so this reports what actually exists rather than
    inventing columns."""
    files_cols  = {r["name"] for r in conn.execute("PRAGMA table_info(files)").fetchall()}
    chunks_cols = {r["name"] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    diag_cols = {"parse_error", "coverage", "low_coverage", "oversize", "unsupported_ext"}
    found = sorted((files_cols | chunks_cols) & diag_cols)

    lines = ["== Parse health =="]
    if found:
        # A future schema might add one of these columns; report it generically
        # rather than guessing its meaning.
        for col in found:
            table = "files" if col in files_cols else "chunks"
            lines.append(f"{table}.{col}: column present (not summarized by doctor yet)")
    else:
        lines.append(
            "per-file parse_error/coverage diagnostics: not recorded in this DB "
            "(the indexer, chonks/index/pipeline.py, computes them per run but only reports them in the run "
            "summary printed at index time — nothing persists them to the DB)"
        )
    lines.append(
        "unsupported-extension histogram: not recorded in this DB (run-summary only)"
    )
    lines.append(
        "oversize chunk count: not recorded in this DB (run-summary only)"
    )
    lines.append(
        "data-blob-skipped count: not recorded in this DB (run-summary only)"
    )
    return "\n".join(lines)


def _table_sizes_section(conn: sqlite3.Connection) -> str:
    lines = ["== Table sizes =="]
    dbstat: dict[str, int] = {}
    try:
        for r in conn.execute("SELECT name, SUM(pgsize) AS bytes FROM dbstat GROUP BY name"):
            dbstat[r["name"]] = r["bytes"]
    except sqlite3.OperationalError:
        dbstat = {}  # dbstat vtab unavailable on this SQLite build, fall back to counts only
    indexes: dict[str, list[str]] = {}
    for r in conn.execute("SELECT name, tbl_name FROM sqlite_master WHERE type = 'index'"):
        indexes.setdefault(r["tbl_name"], []).append(r["name"])

    for table in _KNOWN_TABLES:
        if not _table_exists(conn, table):
            lines.append(f"{table}: table not present")
            continue
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if table in dbstat:
            line = f"{table}: {count} rows, ~{dbstat[table] / 1024:.1f} KB"
            table_indexes = indexes.get(table, [])
            if table_indexes:
                index_bytes = sum(dbstat.get(name, 0) for name in table_indexes)
                n = len(table_indexes)
                line += f" + ~{index_bytes / 1024:.1f} KB in {n} index{'es' if n != 1 else ''}"
            lines.append(line)
        else:
            lines.append(f"{table}: {count} rows (size n/a — dbstat unavailable)")
    return "\n".join(lines)


def _languages_section() -> str:
    """One row per registered language, with its capability flags."""
    lines = ["== Languages =="]
    lines.append(f"{'language':<12} {'refs':<5} {'literals':<9} {'macro':<6} {'pairing':<8} extensions")
    for row in _describe_languages():
        lines.append(
            f"{row['language']:<12} "
            f"{'y' if row['typed_refs'] else '-':<5} "
            f"{'y' if row['literals'] else '-':<9} "
            f"{'y' if row['macro_heal'] else '-':<6} "
            f"{'y' if row['pairing'] else '-':<8} "
            f"{' '.join(row['extensions'])}"
        )
    lines.append("")
    lines.append(
        "refs: typed calls/imports/inherits edges. literals: find_by_message. "
        "macro: C macro self-heal. pairing: header/impl pairing."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_report(conn: sqlite3.Connection, config: dict, db_path: str) -> str:
    sections = [
        _schema_section(conn),
        _chunker_version_section(conn),
        _vitals_section(conn, db_path),
        _staleness_section(conn, config),
        _path_family_section(conn),
        _dotdir_section(conn),
        _edges_section(conn),
        _parse_health_section(conn),
        _table_sizes_section(conn),
        _languages_section(),
    ]
    return "\n\n".join(sections) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="chonks doctor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None, help="Path to chunks DB (default: config.json's `db` key)")
    ap.add_argument("--config", default=None, help="Path to config.json (default: the first of ./config.json, ./chonks/config.json, ./.chonks.json that is present)")
    ap.add_argument(
        "--set-model", default=None, metavar="NAME",
        help="Rewrite the recorded embedding_model label to NAME and exit (no report). "
             "Only for a renamed-but-identical model; vectors are not touched.",
    )
    args = ap.parse_args(argv)

    loaded = load_config(args.config)
    if loaded.problems:
        problem = loaded.problems[0]
        ap.error(f"config not readable: {problem.path}: {problem.reason}")
    config = loaded.data
    load_plugins(config.get("language_plugins") or [], config.get("fallback_extensions"))
    # CLI flag overrides config key, same precedence as chunker/serve.
    args.db = args.db or config.get("db")
    if not args.db:
        ap.error("--db required (or set `db` in config.json)")
    if args.set_model is not None:
        if not Path(args.db).exists():
            ap.error(f"DB not found: {args.db}")
        previous = set_embedding_model(args.db, args.set_model)
        print(f"embedding_model: {previous or 'unset'} -> {args.set_model}")
        return
    if not Path(args.db).exists():
        ap.error(f"DB not found: {args.db} (run chonks index first)")
    conn = _connect_readonly(args.db)
    try:
        print(build_report(conn, config, args.db))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
