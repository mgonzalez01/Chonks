"""Connections to a store DB that do not go through Store."""

import sqlite3
from pathlib import Path


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    """Open the DB strictly read-only, mode=ro in the URI so a write raises
    rather than silently succeeding, and load sqlite-vec so vec0-backed
    tables stay queryable."""
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    # Same busy_timeout Store uses (store.py); without it a concurrent
    # indexing run's write lock raises immediately instead of doctor
    # retrying briefly.
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:
        pass  # optional: nothing here strictly requires vec0
    return conn


def set_embedding_model(db_path: str, model: str) -> str | None:
    """Rewrite the `embedding_model` label in meta. Returns the previous
    value (None if the DB never recorded one). Opens its own read-write
    connection; the report path stays on `_connect_readonly`."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        row = conn.execute("SELECT value FROM meta WHERE key='embedding_model'").fetchone()
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('embedding_model',?)", (model,)
        )
        conn.commit()
    finally:
        conn.close()
    return row[0] if row else None
