"""sqlite-vec storage for chunks, embeddings, and the ref/kNN graph. Schema
is documented in DOCS.md. Embedding dim is fixed on first insert and
validated against `meta` on every later open."""

import json
import logging
import posixpath
import re
import sqlite3
import struct
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec

from chonks.chunking import CODE_LANGUAGES
from chonks.core.edges import _COLLAPSE_RANK, _HUB_EDGE_TYPES, _MAX_CROSS_LANG_OCCURRENCES, edge_provenance

SCHEMA_VERSION = 5

# Unscoped get_hubs scans all of chunk_refs under the store lock, which can
# block every other request for minutes on a huge corpus. Above this size
# the global branch requires a path_prefix instead.
_HUBS_GLOBAL_MAX_CHUNKS = 100_000

# Header/impl pairing (rebuild_hierarchy) is same-dir only; cross-dir layouts
# (include/src) are not paired. Extension matching is case-insensitive.
HEADER_EXTS = {"h", "hh", "hpp", "hxx"}
IMPL_EXTS = {"c", "cc", "cpp", "cxx", "m", "mm"}


def _provenance_rollup(edge_types: dict[str, int]) -> dict[str, int]:
    """Roll up {edge_type: count} into {provenance: count}; shared by
    get_hubs' precomputed and live paths so both use the same rollup."""
    out: dict[str, int] = {}
    for et, n in edge_types.items():
        prov = edge_provenance(et)
        out[prov] = out.get(prov, 0) + n
    return out

# sqlite-vec's hard limit on k for a vec0 KNN query ("k value in knn query too large").
_VEC_KNN_MAX_K = 4096

# Weights `name` over `content` in FTS ranking so an exact-name hit doesn't
# lose to a longer chunk repeating the query tokens. Regression-gated on the
# LocBench panel, don't change without re-running it.
_FTS_NAME_WEIGHT = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack_f32(vec) -> bytes:
    """Pack floats into a little-endian float32 blob for sqlite-vec.
    Assumes a little-endian host."""
    return np.asarray(vec, dtype="<f4").tobytes()


def _unpack_f32(blob: bytes) -> list[float]:
    """Reverse of _pack_f32. 4 bytes per element."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a dict, parsing the JSON `metadata` column back
    into a dict for callers. Rows from queries that don't SELECT metadata are
    returned untouched."""
    d = dict(row)
    md = d.get("metadata")
    if md is not None and isinstance(md, str):
        try:
            d["metadata"] = json.loads(md)
        except (json.JSONDecodeError, ValueError):
            d["metadata"] = None
    return d


def _escape_like(s: str) -> str:
    """Escape SQL LIKE metacharacters so a literal '_' in a path prefix
    isn't treated as a single-char wildcard. Pair with ESCAPE '\\'."""
    s = s.replace("\\", "\\\\")  # escape existing backslashes first
    s = s.replace("%", r"\%")
    s = s.replace("_", r"\_")
    return s


# ---------------------------------------------------------------------------
# Literal & message index: skeleton generation + matching (find_by_message)
# ---------------------------------------------------------------------------

# Sentinel for a collapsed format hole. Must stay printable and FTS-safe; a
# raw NUL byte round-trips badly through FTS5 and JSON/HTTP.
_HOLE_SENTINEL = "␀*"

# printf-style hole. "%%" is protected before this runs so it's never
# swallowed as a hole itself.
_PRINTF_HOLE_RE = re.compile(r"%[-+0 #]*(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlLqjzt]*[diouxXeEfFgGaAcsp]")
# ${var} / $VAR (shell/template style). Matched before the generic brace
# pass below so "${name}" collapses to ONE hole, not "$" + a hole.
_DOLLAR_HOLE_RE = re.compile(r"\$\{[^{}]*\}|\$[A-Za-z_][A-Za-z0-9_]*")
# {}, {0}, {name}, {expr}, {0:D4}: python .format/f-string and C#
# interpolation holes. "{{"/"}}" are protected first.
_BRACE_HOLE_RE = re.compile(r"\{[^{}]*\}")


# _skeleton_match walks fragments with str.find instead of a regex, to
# avoid catastrophic backtracking on a repetitive message. Templates past
# this many holes have too little constant text left to verify, so they're excluded.
_SKELETON_MAX_HOLES = 8

# Skeleton tier only (exact/substring unaffected); belt-and-suspenders bound
# on top of the hole cap and work budget below.
_SKELETON_MAX_MESSAGE_LEN = 600

# Max chars a hole may span between two constant fragments.
_SKELETON_HOLE_GAP = 80

# Hard cap on str.find() calls per candidate skeleton; a repetitive message
# can make the fragment walk branch combinatorially otherwise. Exhaustion
# is reported as a note, never a silent miss.
_SKELETON_WORK_BUDGET = 2000

# Hard cap on str.find() calls across all candidates in one query; the
# per-candidate budget alone leaves total query cost unbounded.
_SKELETON_QUERY_BUDGET = 100_000


def _compute_skeleton(text: str) -> str | None:
    """Collapse format holes into _HOLE_SENTINEL. Returns None when no
    holes were found, or when there are more than _SKELETON_MAX_HOLES."""
    protected = (
        text.replace("%%", "\x02")
            .replace("{{", "\x03")
            .replace("}}", "\x04")
    )
    collapsed = _PRINTF_HOLE_RE.sub(_HOLE_SENTINEL, protected)
    collapsed = _DOLLAR_HOLE_RE.sub(_HOLE_SENTINEL, collapsed)
    collapsed = _BRACE_HOLE_RE.sub(_HOLE_SENTINEL, collapsed)
    collapsed = (
        collapsed.replace("\x02", "%")
                 .replace("\x03", "{")
                 .replace("\x04", "}")
    )
    if collapsed == text:
        return None
    if collapsed.count(_HOLE_SENTINEL) > _SKELETON_MAX_HOLES:
        return None
    return collapsed


def _is_message_shaped(text: str) -> bool:
    """True if `text` is phrase-shaped or long enough to be a real log
    line, not just a short word that's a coincidental substring match."""
    return " " in text or len(text) >= 20


def _passes_exact_gate(text: str, message: str) -> bool:
    return _is_message_shaped(text) or text.strip() == message.strip()


def _longest_skeleton_fragment(skeleton: str) -> str:
    return max(skeleton.split(_HOLE_SENTINEL), key=len, default="")


def _skeleton_match(
    skeleton: str, message: str, budget: int = _SKELETON_WORK_BUDGET,
) -> tuple[bool, bool, int]:
    """Match unanchored, like re.search, so extra context around the
    template still matches. Returns (matched, budget_exhausted, calls_used);
    hitting `budget` returns (False, True, calls_used) immediately."""
    parts = skeleton.split(_HOLE_SENTINEL)
    n = len(parts)
    msg_len = len(message)
    calls = 0
    next_lo = [0] * n
    cur_idx = [-1] * n
    ends = [0] * n

    level = 0
    while 0 <= level < n:
        frag = parts[level]
        flen = len(frag)
        lo = next_lo[level]
        hi = msg_len if level == 0 else min(ends[level - 1] + _SKELETON_HOLE_GAP, msg_len)
        if lo > hi:
            idx = -1
        elif frag == "":
            idx = lo
        else:
            if calls >= budget:
                return False, True, calls
            calls += 1
            idx = message.find(frag, lo, hi + flen)
            if idx > hi:
                idx = -1
        if idx == -1:
            level -= 1
            if level < 0:
                break
            next_lo[level] = cur_idx[level] + 1
            continue
        cur_idx[level] = idx
        ends[level] = idx + flen
        level += 1
        if level < n:
            next_lo[level] = ends[level - 1]
    return level == n, False, calls


def _skeleton_candidates(skeleton: str) -> list[str]:
    """The skeleton plus its first-line-only fallback, since a pasted
    message never carries a template's later lines. Returns just the
    skeleton itself when it's single-line."""
    first = _first_line(skeleton)
    return [skeleton] if first == skeleton else [skeleton, first]


def _first_line(text: str) -> str:
    """Text up to its first newline, or `text` unchanged if it has none.
    The exact/substring tiers' fallback for a multi-line literal, since a
    pasted message never carries its later lines."""
    idx = text.find("\n")
    return text if idx == -1 else text[:idx]


# SQL equivalent of _first_line(cl.text), for the tier-1a query. Kept as
# one string so the two definitions stay in sync.
_FIRST_LINE_SQL = "CASE WHEN instr(cl.text, char(10)) > 0 THEN substr(cl.text, 1, instr(cl.text, char(10)) - 1) ELSE cl.text END"


def _dedupe_literal_rows(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    """Collapse rows sharing (path, line, text): a split boundary node can
    attribute the same literal to several sibling chunks, which would
    otherwise crowd out other matches under `limit`."""
    groups: dict[tuple[str, int, str], list[sqlite3.Row]] = {}
    order: list[tuple[str, int, str]] = []
    for r in rows:
        k = (r["path"], r["line"] or 0, r["text"])
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)
    out = []
    for k in order:
        group = groups[k]
        line = k[1]
        contained = [r for r in group
                     if r["c_start_line"] is not None and r["c_end_line"] is not None
                     and r["c_start_line"] <= line <= r["c_end_line"]]
        out.append(contained[0] if contained else group[0])
    return out


# Means "every chunk in this DB was produced by literal-aware code", set
# only by a full/--force index_paths run, never by a single insert_chunks
# batch. Get this wrong and find_by_message silently looks empty.
_LITERAL_INDEX_META_KEY = "literal_index_version"


def _partial_index_note(conn: sqlite3.Connection) -> str:
    """Message for an unflagged DB (_LITERAL_INDEX_META_KEY): distinguishes
    never-indexed (chunk_literals empty) from partially-indexed (some rows)
    so the note doesn't misreport a real gap as a clean no-match."""
    has_literals = conn.execute("SELECT 1 FROM chunk_literals LIMIT 1").fetchone() is not None
    if has_literals:
        return (
            "literal index incomplete — this DB was only partially indexed "
            "by literal-extraction-aware code; some indexed files have no "
            "literals yet. Run `chonks index --force <paths>` for a full "
            "re-index to enable find_by_message for every file"
        )
    return (
        "index predates literal extraction — run `chonks index --force "
        "<paths>` (a plain re-index skips unchanged files and won't "
        "populate literals) to enable find_by_message"
    )

# FTS candidate-pool cap for the token/skeleton tier, relevance-ordered
# (bm25) so a hit cap only ever drops the least-relevant candidates.
_LITERAL_CANDIDATE_CAP = 20000

# Matches server.py's request-body cap so every accepted message gets the
# substring tier. Skipping above this length is announced, not silent.
_SUBSTRING_TIER_MAX_MESSAGE_LEN = 2000

# Caps the FTS token pool to the N longest tokens so the OR-query stays
# cheap. Exceeding this can crowd out the real literal's tokens; announced,
# not silently dropped.
_TOKEN_POOL_CAP = 12

_VALID_CHUNK_KINDS = frozenset({"code", "docs", "any"})


def _chunk_kind_clause(chunk_kind: str | None, column: str = "language") -> tuple[str, list[str]]:
    """SQL `AND ...` fragment + params for chunk_kind against `column`.
    docs includes NULL: text-fallback chunks and rows missing `language`
    must still count as docs, not be silently excluded."""
    if not chunk_kind or chunk_kind == "any":
        return "", []
    if chunk_kind not in _VALID_CHUNK_KINDS:
        raise ValueError(f"Invalid chunk_kind: {chunk_kind!r}. Use code|docs|any")
    placeholders = ",".join("?" * len(CODE_LANGUAGES))
    langs = list(CODE_LANGUAGES)
    if chunk_kind == "code":
        return f"AND {column} IN ({placeholders})", langs
    return f"AND ({column} IS NULL OR {column} NOT IN ({placeholders}))", langs


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._dim: int | None = None

        # check_same_thread=False so a worker thread (the indexer's embedder)
        # can share this connection. RLock, not Lock: insert_chunks calls
        # _set_dim calls _ensure_vec_table, re-entering on the same thread.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-32000")  # 32 MB
        # recursive_triggers ON: an INSERT OR REPLACE conflict-delete must
        # still fire the chunks_ad FTS trigger, or chunks_fts drifts out of sync.
        self._conn.execute("PRAGMA recursive_triggers=ON")
        # WAL is still single-writer; wait up to 5s for a held lock instead
        # of erroring immediately, to smooth server/indexer contention.
        self._conn.execute("PRAGMA busy_timeout=5000")

        # Load sqlite-vec extension
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        # Runs only from __init__ before any other thread can hold a reference
        # to this Store, but kept under the lock for uniformity with the rest
        # of the conn-touching surface.
        with self._lock:
            cur = self._conn
            cur.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS files (
                path         TEXT PRIMARY KEY,
                size         INTEGER,
                mtime        REAL,
                content_hash TEXT,
                indexed_at   REAL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                path        TEXT NOT NULL,
                language    TEXT,
                chunk_type  TEXT,
                name        TEXT,
                start_line  INTEGER,
                end_line    INTEGER,
                content     TEXT NOT NULL,
                indexed_at  REAL,
                metadata    TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
            CREATE INDEX IF NOT EXISTS idx_files_path  ON files(path);
            -- Backs build_refs' incremental "find chunks named X" lookup;
            -- without it, resolving a few touched names is a full scan.
            CREATE INDEX IF NOT EXISTS idx_chunks_name ON chunks(name);

            CREATE TABLE IF NOT EXISTS chunk_refs (
                from_id   TEXT NOT NULL,
                to_id     TEXT NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'mentions',
                PRIMARY KEY (from_id, to_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_refs_from ON chunk_refs(from_id);
            CREATE INDEX IF NOT EXISTS idx_chunk_refs_to   ON chunk_refs(to_id);

            -- Persisted PageRank, computed once at index time. Empty on a
            -- DB from before this existed; compute_pagerank_global falls
            -- back to a live compute in that case.
            CREATE TABLE IF NOT EXISTS chunk_pagerank (
                chunk_id TEXT PRIMARY KEY,
                score    REAL NOT NULL
            );

            -- Persisted in-degree, computed at graph-build time. Empty on a
            -- DB from before this existed; get_hubs falls back to a live
            -- GROUP BY, guarded by _HUBS_GLOBAL_MAX_CHUNKS, in that case.
            CREATE TABLE IF NOT EXISTS chunk_indegree (
                chunk_id  TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                n         INTEGER NOT NULL,
                PRIMARY KEY (chunk_id, edge_type)
            );

            -- File hierarchy (dir/file nodes + 'contains' edges) is kept out
            -- of chunk_refs so PageRank doesn't pick up hierarchy nodes and
            -- chunk_pagerank stays chunk-only, comparable to pre-v2 baselines.
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id        TEXT PRIMARY KEY,   -- "dir:<path>" | "file:<path>"
                kind      TEXT NOT NULL,      -- 'dir' | 'file'
                path      TEXT NOT NULL,
                parent_id TEXT                -- containing dir's node id; NULL at root
            );

            CREATE INDEX IF NOT EXISTS idx_graph_nodes_path   ON graph_nodes(path);
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_parent ON graph_nodes(parent_id);

            CREATE TABLE IF NOT EXISTS graph_edges (
                from_id   TEXT NOT NULL,
                to_id     TEXT NOT NULL,      -- node id, or a chunks.id for file->chunk
                edge_type TEXT NOT NULL,      -- 'contains' (Stage 1)
                PRIMARY KEY (from_id, to_id, edge_type)
            );

            CREATE INDEX IF NOT EXISTS idx_graph_edges_from ON graph_edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_to   ON graph_edges(to_id);

            CREATE TABLE IF NOT EXISTS folder_summaries (
                path              TEXT PRIMARY KEY,
                summary           TEXT NOT NULL,
                summary_embedding BLOB NOT NULL,
                content_hash      TEXT NOT NULL,
                generated_at      REAL
            );

            CREATE TABLE IF NOT EXISTS chunk_neighbors (
                chunk_id    TEXT NOT NULL,
                neighbor_id TEXT NOT NULL,
                distance    REAL NOT NULL,
                PRIMARY KEY (chunk_id, neighbor_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_neighbors_chunk ON chunk_neighbors(chunk_id);
            -- Reverse lookup for the incremental k-NN update: finds rows
            -- whose neighbor list contained a chunk that was just deleted,
            -- without a full table scan.
            CREATE INDEX IF NOT EXISTS idx_chunk_neighbors_neighbor ON chunk_neighbors(neighbor_id);

            -- Decoupled symbol index: one row per named boundary, independent
            -- of how the chunker packs content, so folded/merged/split chunks
            -- still have every symbol findable. chunk_id may be NULL.
            CREATE TABLE IF NOT EXISTS symbols (
                id          INTEGER PRIMARY KEY,
                path        TEXT NOT NULL,
                name        TEXT NOT NULL,
                kind        TEXT,
                language    TEXT,
                start_line  INTEGER NOT NULL,
                end_line    INTEGER,
                chunk_id    TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_symbols_name  ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_path  ON symbols(path);
            CREATE INDEX IF NOT EXISTS idx_symbols_chunk ON symbols(chunk_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                id UNINDEXED,
                name,
                content,
                content=chunks,
                content_rowid=rowid
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, id, name, content)
                VALUES (new.rowid, new.id, new.name, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, id, name, content)
                VALUES ('delete', old.rowid, old.id, old.name, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, id, name, content)
                VALUES ('delete', old.rowid, old.id, old.name, old.content);
                INSERT INTO chunks_fts(rowid, id, name, content)
                VALUES (new.rowid, new.id, new.name, new.content);
            END;

            -- CREATE TABLE IF NOT EXISTS alone migrates an old DB: no
            -- SCHEMA_VERSION bump, no forced re-index. Coverage is tracked
            -- by _LITERAL_INDEX_META_KEY, not by this table being empty.
            CREATE TABLE IF NOT EXISTS chunk_literals (
                chunk_id TEXT NOT NULL,
                text     TEXT NOT NULL,
                skeleton TEXT,
                line     INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_literals_chunk ON chunk_literals(chunk_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS literals_fts USING fts5(
                text,
                content=chunk_literals,
                content_rowid=rowid
            );

            CREATE TRIGGER IF NOT EXISTS chunk_literals_ai AFTER INSERT ON chunk_literals BEGIN
                INSERT INTO literals_fts(rowid, text) VALUES (new.rowid, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunk_literals_ad AFTER DELETE ON chunk_literals BEGIN
                INSERT INTO literals_fts(literals_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunk_literals_au AFTER UPDATE ON chunk_literals BEGIN
                INSERT INTO literals_fts(literals_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
                INSERT INTO literals_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
            """)
            self._conn.commit()

            # Adds edge_type to a pre-migration DB. Guarded by a pragma check
            # since sqlite has no ADD COLUMN IF NOT EXISTS. No backfill needed:
            # build_refs clears and rebuilds chunk_refs on every index run.
            cols = {row["name"] for row in
                    self._conn.execute("PRAGMA table_info(chunk_refs)").fetchall()}
            if "edge_type" not in cols:
                self._conn.execute(
                    "ALTER TABLE chunk_refs ADD COLUMN edge_type TEXT NOT NULL DEFAULT 'mentions'"
                )
                self._conn.commit()

            # Catches DB/code schema mismatch immediately at startup.
            stored_ver = self._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if stored_ver is None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION),),
                )
                self._conn.commit()
            elif int(stored_ver["value"]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"DB schema version {stored_ver['value']} does not match code "
                    f"version {SCHEMA_VERSION}. Drop the DB file and re-index."
                )

            # Restore dim from meta if already set
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='embedding_dim'"
            ).fetchone()
            if row:
                self._dim = int(row["value"])
                self._ensure_vec_table(self._dim)

    def _ensure_vec_table(self, dim: int) -> None:
        """Create the vec0 virtual table for the given dimension (once only)."""
        with self._lock:
            self._conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vecs USING vec0(
                    id        TEXT PRIMARY KEY,
                    embedding int8[{dim}]
                )
            """)
            self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))
            self._conn.commit()

    def _set_dim(self, dim: int, model: str | None = None) -> None:
        """Validate (and on first call, persist) the embedding dim and model.
        `model` is optional for back-compat with older callers; new callers
        should pass it so a silent dim-matches-but-model-differs swap is caught."""
        with self._lock:
            if self._dim is not None and self._dim != dim:
                raise ValueError(
                    f"Embedding dimension mismatch: DB has {self._dim}, got {dim}"
                )
            # Missing recorded model is treated as matching (old DB); a
            # mismatched one is refused, since that's a silent wrong-vectors bug.
            if model is not None:
                stored_model_row = self._conn.execute(
                    "SELECT value FROM meta WHERE key='embedding_model'"
                ).fetchone()
                if stored_model_row is None:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO meta(key,value) VALUES('embedding_model',?)",
                        (model,),
                    )
                elif stored_model_row["value"] != model:
                    raise ValueError(
                        f"Embedding model mismatch: DB was indexed with "
                        f"{stored_model_row['value']!r}, current config uses {model!r}. "
                        f"Drop the DB and re-index, revert the model setting, or if the "
                        f"model is the same and only its name changed: "
                        f"chonks doctor --db <path> --set-model {model!r}"
                    )
            if self._dim is None:
                self._dim = dim
                self._ensure_vec_table(dim)
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('embedding_dim',?)",
                    (str(dim),),
                )
            self._conn.commit()

    # ------------------------------------------------------------------
    # File change detection
    # ------------------------------------------------------------------

    def get_file_hash(self, path: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT content_hash FROM files WHERE path=?", (path,)
            ).fetchone()
            return row["content_hash"] if row else None

    def upsert_file(self, path: str, size: int, mtime: float, content_hash: str) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO files(path,size,mtime,content_hash,indexed_at)
                   VALUES(?,?,?,?,?)""",
                (path, size, mtime, content_hash, time.time()),
            )
            # Don't commit here: caller batches.

    def get_paths_under(self, prefix: str) -> list[str]:
        """Return all indexed paths that start with the given POSIX prefix."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path FROM files WHERE path LIKE ? ESCAPE '\\'",
                (_escape_like(prefix) + "%",),
            ).fetchall()
            return [r[0] for r in rows]

    def delete_file(self, path: str) -> list[str]:
        """Remove a file and its chunks, vectors, and FTS entries. Returns
        the deleted chunk ids so callers can update derived state (e.g. the
        k-NN graph) without a second query."""
        with self._lock:
            chunk_ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT id FROM chunks WHERE path=?", (path,)
                ).fetchall()
            ]
            for i in range(0, len(chunk_ids), 900):
                batch = chunk_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                self._conn.execute(
                    f"DELETE FROM chunk_vecs WHERE id IN ({placeholders})", batch
                )
                self._conn.execute(
                    f"DELETE FROM chunk_literals WHERE chunk_id IN ({placeholders})", batch
                )
                self._conn.execute(
                    f"DELETE FROM chunks WHERE id IN ({placeholders})", batch
                )
            self._conn.execute("DELETE FROM symbols WHERE path=?", (path,))
            self._conn.execute("DELETE FROM files WHERE path=?", (path,))
            return chunk_ids

    def get_names_for_path(self, path: str) -> set[str]:
        """Names defined at `path`. Must be called BEFORE delete_file(path):
        build_refs needs these names to know what a deletion affects, and
        they're gone once the rows are deleted."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT name FROM chunks WHERE path=? AND name IS NOT NULL "
                "UNION SELECT name FROM symbols WHERE path=?",
                (path, path),
            ).fetchall()
        return {r[0] for r in rows}

    # ------------------------------------------------------------------
    # Symbols (decoupled symbol index)
    # ------------------------------------------------------------------

    def insert_symbols(self, rows: list[dict[str, Any]]) -> None:
        """Insert symbol rows. Each row: {path, name, kind, language, start_line,
        end_line, chunk_id}. Caller is responsible for clearing the file's prior
        symbols first (see delete_symbols_for_path / delete_file)."""
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO symbols(path, name, kind, language, start_line, end_line, chunk_id) "
                "VALUES (:path, :name, :kind, :language, :start_line, :end_line, :chunk_id)",
                rows,
            )
            self._conn.commit()

    def delete_symbols_for_path(self, path: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM symbols WHERE path=?", (path,))
            self._conn.commit()

    def get_all_symbols(self, path_prefix: str | None = None) -> list[dict[str, Any]]:
        """Every symbol (optionally under a path prefix), for repomap listing."""
        sql = ("SELECT path, name, kind, language, start_line, end_line, chunk_id "
               "FROM symbols")
        args: list = []
        if path_prefix:
            sql += " WHERE path LIKE ? ESCAPE '\\'"
            args.append(self._like_escape(path_prefix.rstrip("/\\")) + "%")
        sql += " ORDER BY path, start_line"
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    # C++ symbols are qualified ("AABB::encloses") but callers query the bare
    # method name, so an exact miss falls back to a last-component suffix
    # match. LIKE wildcards in the name are escaped so '_' isn't a wildcard.

    @staticmethod
    def _like_escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def find_symbols(self, name: str, path_prefix: str | None = None,
                     prefix: bool = False) -> list[dict[str, Any]]:
        """Exact-name (or prefix) symbol lookup -> path:line + owning chunk_id.
        Bare names that miss exactly also match as the last component of a
        qualified name (see comment above)."""
        base = "SELECT path, name, kind, language, start_line, end_line, chunk_id FROM symbols WHERE "
        tail = " ORDER BY path, start_line"

        def run(where: str, args: list) -> list[dict[str, Any]]:
            sql, a = base + where, list(args)
            if path_prefix:
                sql += " AND path LIKE ? ESCAPE '\\'"
                a.append(self._like_escape(path_prefix.rstrip("/\\")) + "%")
            with self._lock:
                return [dict(r) for r in self._conn.execute(sql + tail, a).fetchall()]

        if prefix:
            return run(r"name LIKE ? ESCAPE '\'", [self._like_escape(name) + "%"])
        rows = run("name = ?", [name])
        if rows or "::" in name or "." in name:
            return rows
        esc = self._like_escape(name)
        return run(r"(name LIKE ? ESCAPE '\' OR name LIKE ? ESCAPE '\')",
                   [f"%::{esc}", f"%.{esc}"])

    def resolve_symbol_chunk_ids(self, name: str) -> list[str]:
        """Resolve a name to its defining chunk_id(s), same lookup and
        suffix fallback as find_symbols. Can return more than one id
        (overloads); callers needing "the" definition should try each."""
        with self._lock:
            def_rows = self._conn.execute(
                "SELECT DISTINCT chunk_id FROM symbols WHERE name = ? AND chunk_id IS NOT NULL",
                (name,),
            ).fetchall()
            if not def_rows and "::" not in name and "." not in name:
                esc = self._like_escape(name)
                def_rows = self._conn.execute(
                    r"SELECT DISTINCT chunk_id FROM symbols WHERE "
                    r"(name LIKE ? ESCAPE '\' OR name LIKE ? ESCAPE '\') "
                    r"AND chunk_id IS NOT NULL",
                    (f"%::{esc}", f"%.{esc}"),
                ).fetchall()
        return [r["chunk_id"] for r in def_rows]

    def _symbol_miss_note(self, name: str) -> str:
        """Diagnostic for a resolve_symbol_chunk_ids miss. A qualified query
        gets no suffix fallback, so this suggests the bare last component
        only when that bare name actually resolves to something."""
        if "::" in name or "." in name:
            bare = re.split(r"::|\.", name)[-1]
            if bare and bare != name and self.resolve_symbol_chunk_ids(bare):
                return f"symbol not found — try the bare name {bare!r}"
        return "symbol not found"

    def _fts_scan_for_name(self, name: str, path_prefix: str | None,
                           limit: int | None) -> list[dict[str, Any]]:
        """FTS content-scan fallback for find_usages/get_impact when a name
        exceeds _MAX_CROSS_LANG_OCCURRENCES. A pre-filter, not a proven
        reference; callers must label these as content matches, not edges."""
        phrase = '"' + name.replace('"', '""') + '"'
        rows = self.search_fts(phrase, top_k=limit or 50, path_prefix=path_prefix)
        return [
            {
                "chunk_id": r["id"], "path": r["path"], "name": r.get("name"),
                "chunk_type": r.get("chunk_type"), "start_line": r["start_line"],
                "end_line": r["end_line"], "origin": "fts_scan",
            }
            for r in rows
        ]

    def find_usages(self, name: str, path_prefix: str | None = None,
                    limit: int | None = None) -> dict[str, Any]:
        """Who references `name`. Empty `results` doesn't mean "no callers":
        see `note` for the unresolved-name and above-cap cases. `limit`
        truncates AFTER sorting by edge quality, not alphabetically."""
        chunk_ids = self.resolve_symbol_chunk_ids(name)
        if not chunk_ids:
            return {"results": [], "note": self._symbol_miss_note(name), "content_matches": []}

        edges = self.get_refs_to_chunks_typed(chunk_ids)
        from_ids = sorted({from_id for from_id, _to_id, _et in edges})
        if not from_ids:
            if len(chunk_ids) > _MAX_CROSS_LANG_OCCURRENCES:
                note = (
                    f"{name!r} has {len(chunk_ids)} definers, above the "
                    f"edge-indexing cap ({_MAX_CROSS_LANG_OCCURRENCES}) — reference "
                    "edges are not indexed for this name; falling back to an FTS content scan"
                )
                content_matches = self._fts_scan_for_name(name, path_prefix, limit)
                return {"results": [], "note": note, "content_matches": content_matches}
            return {"results": [], "note": None, "content_matches": []}

        # Collapse to the least-uncertain edge type when a chunk has more
        # than one into the resolution set (see _COLLAPSE_RANK).
        edge_type_by_from: dict[str, str] = {}
        for from_id, _to_id, et in edges:
            cur = edge_type_by_from.get(from_id)
            if cur is None or (
                (_COLLAPSE_RANK.get(et, 3), et) < (_COLLAPSE_RANK.get(cur, 3), cur)
            ):
                edge_type_by_from[from_id] = et

        results: list[dict[str, Any]] = []
        with self._lock:
            for i in range(0, len(from_ids), 900):
                batch = from_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                sql = (f"SELECT id AS chunk_id, path, name, chunk_type, start_line, end_line "
                       f"FROM chunks WHERE id IN ({placeholders})")
                args: list = list(batch)
                if path_prefix:
                    sql += " AND path LIKE ? ESCAPE '\\'"
                    args.append(self._like_escape(path_prefix.rstrip("/\\")) + "%")
                results.extend(dict(r) for r in self._conn.execute(sql, args).fetchall())
        for r in results:
            et = edge_type_by_from.get(r["chunk_id"], "mentions")
            r["edge_type"] = et
            r["provenance"] = edge_provenance(et)
        results.sort(key=lambda r: (_COLLAPSE_RANK.get(r["edge_type"], 3), r["path"], r["start_line"]))
        note = None
        if limit and len(results) > limit:
            omitted = results[limit:]
            omitted_xlang = sum(1 for r in omitted if r["edge_type"] == "xlang")
            omitted_associated = sum(1 for r in omitted if r["edge_type"] == "associated")
            omitted_mentions = sum(1 for r in omitted if r["edge_type"] == "mentions")
            omitted_typed = len(omitted) - omitted_xlang - omitted_associated - omitted_mentions
            note = (
                f"limit={limit} truncated {len(omitted)} result(s): "
                f"{omitted_typed} typed, {omitted_xlang} xlang, "
                f"{omitted_associated} associated, {omitted_mentions} mentions"
            )
            results = results[:limit]
        return {"results": results, "note": note, "content_matches": []}

    def find_outgoing(self, name: str, path_prefix: str | None = None,
                      limit: int | None = None) -> dict[str, Any]:
        """Forward twin of find_usages (self-refs among definers excluded).
        Zero outgoing edges is a valid empty result. `limit` truncates
        AFTER sorting by edge quality, not alphabetically."""
        chunk_ids = self.resolve_symbol_chunk_ids(name)
        if not chunk_ids:
            return {"results": [], "note": self._symbol_miss_note(name)}

        definer_set = set(chunk_ids)
        edges = self.get_refs_from_chunks_typed(chunk_ids)
        to_ids = sorted({to_id for _from_id, to_id, _et in edges if to_id not in definer_set})
        if not to_ids:
            return {"results": [], "note": None}

        edge_type_by_to: dict[str, str] = {}
        for _from_id, to_id, et in edges:
            if to_id in definer_set:
                continue
            cur = edge_type_by_to.get(to_id)
            if cur is None or (
                (_COLLAPSE_RANK.get(et, 3), et) < (_COLLAPSE_RANK.get(cur, 3), cur)
            ):
                edge_type_by_to[to_id] = et

        results: list[dict[str, Any]] = []
        with self._lock:
            for i in range(0, len(to_ids), 900):
                batch = to_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                sql = (f"SELECT id AS chunk_id, path, name, chunk_type, start_line, end_line "
                       f"FROM chunks WHERE id IN ({placeholders})")
                args: list = list(batch)
                if path_prefix:
                    sql += " AND path LIKE ? ESCAPE '\\'"
                    args.append(self._like_escape(path_prefix.rstrip("/\\")) + "%")
                results.extend(dict(r) for r in self._conn.execute(sql, args).fetchall())
        for r in results:
            et = edge_type_by_to.get(r["chunk_id"], "mentions")
            r["edge_type"] = et
            r["provenance"] = edge_provenance(et)
        results.sort(key=lambda r: (_COLLAPSE_RANK.get(r["edge_type"], 3), r["path"], r["start_line"]))
        if limit:
            results = results[:limit]
        return {"results": results, "note": None}

    def get_impact(self, name: str, path_prefix: str | None = None,
                   limit: int = 20, rank_by: str = "pagerank_sum") -> dict[str, Any]:
        """Blast radius of `name`, aggregated by referencing file, with a
        `note` matching find_usages' diagnostics. `rank_by="pagerank_sum"`
        degrades cleanly to count-DESC when chunk_pagerank is empty."""
        if rank_by not in ("pagerank_sum", "count"):
            raise ValueError(
                f"invalid rank_by {rank_by!r} — must be 'pagerank_sum' or 'count'"
            )
        def_chunk_ids = self.resolve_symbol_chunk_ids(name)
        if not def_chunk_ids:
            return {
                "symbol": name, "definitions": [], "total_references": 0,
                "by_edge_type": {}, "by_provenance": {}, "rank_by": rank_by,
                "files": [], "files_total": 0,
                "note": self._symbol_miss_note(name),
            }

        definitions: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        with self._lock:
            for i in range(0, len(def_chunk_ids), 900):
                batch = def_chunk_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT path, name, chunk_type FROM chunks WHERE id IN ({placeholders})",
                    batch,
                ).fetchall()
                definitions.extend(dict(r) for r in rows)

                sql = (
                    "SELECT cr.from_id AS chunk_id, cr.edge_type AS edge_type, "
                    "c.path AS path, c.name AS name, c.chunk_type AS chunk_type, "
                    "c.start_line AS start_line, COALESCE(pr.score, 0.0) AS pagerank "
                    "FROM chunk_refs cr "
                    "JOIN chunks c ON c.id = cr.from_id "
                    "LEFT JOIN chunk_pagerank pr ON pr.chunk_id = cr.from_id "
                    f"WHERE cr.to_id IN ({placeholders})"
                )
                args: list = list(batch)
                if path_prefix:
                    sql += " AND c.path LIKE ? ESCAPE '\\'"
                    args.append(self._like_escape(path_prefix.rstrip("/\\")) + "%")
                edges.extend(dict(r) for r in self._conn.execute(sql, args).fetchall())

        definitions.sort(key=lambda d: (d["path"], d["name"] or ""))

        by_edge_type: dict[str, int] = {}
        for e in edges:
            by_edge_type[e["edge_type"]] = by_edge_type.get(e["edge_type"], 0) + 1

        by_provenance: dict[str, int] = {}
        for et, n in by_edge_type.items():
            prov = edge_provenance(et)
            by_provenance[prov] = by_provenance.get(prov, 0) + n

        files: dict[str, dict[str, Any]] = {}
        for e in edges:
            f = files.setdefault(e["path"], {
                "path": e["path"], "count": 0, "edge_types": {}, "_chunks": {},
            })
            f["count"] += 1
            f["edge_types"][e["edge_type"]] = f["edge_types"].get(e["edge_type"], 0) + 1
            f["_chunks"][e["chunk_id"]] = (e["name"], e["chunk_type"], e["start_line"], e["pagerank"])

        file_list: list[dict[str, Any]] = []
        for f in files.values():
            chunks = f.pop("_chunks")
            f["pagerank_sum"] = sum(v[3] for v in chunks.values())
            referrers = sorted(chunks.values(), key=lambda v: (-v[3], v[2]))[:3]
            f["top_referrers"] = [
                {"name": n, "chunk_type": ct, "start_line": sl}
                for n, ct, sl, _pr in referrers
            ]
            file_list.append(f)

        if rank_by == "count":
            file_list.sort(key=lambda f: (-f["count"], -f["pagerank_sum"], f["path"]))
        else:
            file_list.sort(key=lambda f: (-f["pagerank_sum"], -f["count"], f["path"]))
        files_total = len(file_list)

        note = None
        if not edges and len(def_chunk_ids) > _MAX_CROSS_LANG_OCCURRENCES:
            note = (
                f"{name!r} has {len(def_chunk_ids)} definers, above the "
                f"edge-indexing cap ({_MAX_CROSS_LANG_OCCURRENCES}) — reference "
                "edges are not indexed for this name; try find_usages, which falls "
                "back to an FTS content scan for this case"
            )

        return {
            "symbol": name,
            "definitions": definitions,
            "total_references": len(edges),
            "by_edge_type": by_edge_type,
            "by_provenance": by_provenance,
            "rank_by": rank_by,
            "files": file_list[:limit],
            "files_total": files_total,
            "note": note,
        }

    def get_symbol_name_chunks(self) -> dict[str, list[str]]:
        """name -> [chunk_id] (NULL chunk_ids skipped), for build_refs name_set
        expansion so references to a merged-away name still create graph edges."""
        out: dict[str, list[str]] = {}
        with self._lock:
            for r in self._conn.execute(
                "SELECT name, chunk_id FROM symbols WHERE chunk_id IS NOT NULL"
            ).fetchall():
                out.setdefault(r["name"], []).append(r["chunk_id"])
        return out

    def symbols_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]

    def get_symbol_names_by_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        """Distinct symbol names on the given chunk ids, so build_refs
        doesn't drop a sibling name's edges when its chunk becomes an
        affected target for an unrelated touched name."""
        if not chunk_ids:
            return set()
        out: set[str] = set()
        with self._lock:
            for i in range(0, len(chunk_ids), 900):
                batch = chunk_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT DISTINCT name FROM symbols WHERE chunk_id IN ({placeholders})",
                    batch,
                ).fetchall()
                out.update(r[0] for r in rows)
        return out

    def get_symbol_chunk_ids_by_names(self, names: list[str]) -> dict[str, list[str]]:
        """name -> [chunk_id], scoped variant of get_symbol_name_chunks for
        the incremental build_refs update, so it avoids a full table scan."""
        if not names:
            return {}
        out: dict[str, list[str]] = defaultdict(list)
        with self._lock:
            for i in range(0, len(names), 900):
                batch = names[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT name, chunk_id FROM symbols "
                    f"WHERE chunk_id IS NOT NULL AND name IN ({placeholders})",
                    batch,
                ).fetchall()
                for r in rows:
                    out[r["name"]].append(r["chunk_id"])
        return dict(out)

    # ------------------------------------------------------------------
    # Chunk writes
    # ------------------------------------------------------------------

    def insert_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        *,
        model: str | None = None,
    ) -> None:
        """Insert a batch of chunks and embeddings atomically. `model` is
        recorded in `meta` on first insert and validated against it after."""
        assert len(chunks) == len(embeddings), "chunks/embeddings length mismatch"
        if not chunks:
            return

        # chunk_vecs has no INSERT OR REPLACE, so a repeated id in one batch
        # would crash after the pre-delete below. Dedupe, keeping the last
        # occurrence (matches the chunks-table INSERT OR REPLACE semantics).
        if len({c["id"] for c in chunks}) != len(chunks):
            deduped: dict[str, tuple[dict[str, Any], list[float]]] = {}
            dup_example: dict[str, Any] | None = None
            for c, e in zip(chunks, embeddings):
                if c["id"] in deduped and dup_example is None:
                    dup_example = c
                deduped[c["id"]] = (c, e)  # last write wins
            logging.getLogger(__name__).warning(
                "insert_chunks: collapsed %d duplicate chunk id(s) in one batch "
                "(same path:start_line:content emitted twice — a double-indexed "
                "file or a duplicate span). e.g. %s:%s",
                len(chunks) - len(deduped),
                dup_example.get("path") if dup_example else "?",
                dup_example.get("start_line") if dup_example else "?",
            )
            chunks = [c for c, _ in deduped.values()]
            embeddings = [e for _, e in deduped.values()]

        dim = len(embeddings[0])
        with self._lock:
            self._set_dim(dim, model=model)

            now = time.time()
            ids = [c["id"] for c in chunks]

            # vec0 doesn't support INSERT OR REPLACE: delete first. chunk_literals
            # is deleted+reinserted the same way so a direct re-insert (without
            # a preceding delete_file) stays correct too.
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"DELETE FROM chunk_vecs WHERE id IN ({placeholders})", ids
            )
            self._conn.execute(
                f"DELETE FROM chunk_literals WHERE chunk_id IN ({placeholders})", ids
            )

            for chunk, emb in zip(chunks, embeddings):
                cid = chunk["id"]
                md = chunk.get("metadata")
                md_json = json.dumps(md, separators=(",", ":")) if md else None
                self._conn.execute(
                    """INSERT OR REPLACE INTO chunks
                       (id,path,language,chunk_type,name,start_line,end_line,content,indexed_at,metadata)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        cid,
                        chunk["path"],
                        chunk.get("language"),
                        chunk.get("chunk_type"),
                        chunk.get("name"),
                        chunk.get("start_line"),
                        chunk.get("end_line"),
                        chunk["content"],
                        now,
                        md_json,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO chunk_vecs(id, embedding) VALUES(?, vec_quantize_int8(?, 'unit'))",
                    (cid, _pack_f32(emb)),
                )
                for lit in chunk.get("literals") or ():
                    text, line = lit[0], lit[1]
                    self._conn.execute(
                        "INSERT INTO chunk_literals(chunk_id, text, skeleton, line) VALUES(?,?,?,?)",
                        (cid, text, _compute_skeleton(text), line),
                    )

            # Does NOT set _LITERAL_INDEX_META_KEY: this is one batch, not
            # necessarily the whole DB. That flag is chunker.py's job, set
            # once at the end of a full index_paths run.
            self._conn.commit()

    def update_vectors(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        *,
        model: str | None = None,
    ) -> None:
        """Replace vectors for existing chunks; doesn't touch the chunks
        table. `ids` and `embeddings` must be the same length, or this
        deletes vectors it never re-inserts, leaving chunks with none."""
        if not ids:
            return
        if len(ids) != len(embeddings):
            raise ValueError(
                f"update_vectors: ids ({len(ids)}) and embeddings ({len(embeddings)}) "
                "must be the same length — refusing to delete vectors without a "
                "matching re-insert for every id."
            )
        self._set_dim(len(embeddings[0]), model=model)
        with self._lock:
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                self._conn.execute(f"DELETE FROM chunk_vecs WHERE id IN ({placeholders})", batch)
            for cid, emb in zip(ids, embeddings):
                self._conn.execute(
                    "INSERT INTO chunk_vecs(id, embedding) VALUES(?, vec_quantize_int8(?, 'unit'))",
                    (cid, _pack_f32(emb)),
                )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_semantic(
        self,
        query_embedding: list[float],
        top_k: int,
        path_prefix: str | None = None,
        chunk_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """KNN search via sqlite-vec, filtered by path prefix and/or
        chunk_kind. Returns chunk dicts with an added 'distance' field."""
        dim = len(query_embedding)
        with self._lock:
            if self._dim is None:
                return []
            if self._dim != dim:
                raise ValueError(
                    f"Embedding dimension mismatch: DB has {self._dim}, got {dim}"
                )

            kind_clause, kind_params = _chunk_kind_clause(chunk_kind)
            path_clause = "AND path LIKE ? ESCAPE '\\'" if path_prefix else ""

            # Path/kind filtering is a post-filter join (vec0 has no WHERE
            # pushdown), so a fixed oversample can under-fill top_k when the
            # filtered-out fraction is large. Widen k and retry until filled.
            total: int | None = None
            if kind_clause:
                total = self._conn.execute("SELECT COUNT(*) FROM chunk_vecs").fetchone()[0]
            fetch_k = top_k * 4 if (path_prefix or kind_clause) else top_k
            if total is not None:
                fetch_k = min(fetch_k, total) if total else fetch_k
            fetch_k = min(fetch_k, _VEC_KNN_MAX_K)

            while True:
                rows = self._conn.execute(
                    """
                    SELECT v.id, v.distance
                    FROM chunk_vecs v
                    WHERE v.embedding MATCH vec_quantize_int8(?, 'unit')
                      AND k = ?
                    ORDER BY v.distance
                    """,
                    (_pack_f32(query_embedding), fetch_k),
                ).fetchall()

                if not rows:
                    return []

                ids = [r["id"] for r in rows]
                dist_by_id = {r["id"]: r["distance"] for r in rows}

                placeholders = ",".join("?" * len(ids))
                params: list[Any] = list(ids)
                if path_prefix:
                    params.append(_escape_like(path_prefix.rstrip("/\\")) + "%")
                params.extend(kind_params)

                chunks = self._conn.execute(
                    f"""
                    SELECT * FROM chunks
                    WHERE id IN ({placeholders})
                    {path_clause}
                    {kind_clause}
                    """,
                    params,
                ).fetchall()

                if (
                    not kind_clause
                    or len(chunks) >= top_k
                    or fetch_k >= _VEC_KNN_MAX_K
                    or (total is not None and fetch_k >= total)
                ):
                    break
                fetch_k = min(fetch_k * 4, total, _VEC_KNN_MAX_K)

        results = []
        for row in chunks:
            d = _row_to_dict(row)
            d["distance"] = dist_by_id[d["id"]]
            results.append(d)

        results.sort(key=lambda x: x["distance"])
        return results[:top_k]

    def search_fts(
        self,
        query: str,
        top_k: int = 50,
        path_prefix: str | None = None,
        chunk_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """FTS5 keyword search. query is a standard FTS5 query string.
        chunk_kind ("code" | "docs" | "any"/None) filters before the SQL
        LIMIT, so a filtered search still returns up to top_k matches."""
        kind_clause, kind_params = _chunk_kind_clause(chunk_kind, column="c.language")
        path_clause = "AND c.path LIKE ? ESCAPE '\\'" if path_prefix else ""
        params: list[Any] = [query]
        if path_prefix:
            params.append(_escape_like(path_prefix.rstrip("/\\")) + "%")
        params.extend(kind_params)
        params.append(top_k)

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT c.*, bm25(chunks_fts, 0.0, {_FTS_NAME_WEIGHT}, 1.0) AS fts_rank
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.id
                WHERE chunks_fts MATCH ?
                  {path_clause}
                  {kind_clause}
                ORDER BY fts_rank
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [_row_to_dict(r) for r in rows]

    def search_regex(
        self,
        pattern: str,
        top_k: int = 50,
        path_prefix: str | None = None,
        chunk_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """Regex search, streamed cursor-side so memory stays O(top_k).
        No SQL LIMIT pushdown: the regex filter is Python-side, so a SQL
        LIMIT would cap the scan and miss matches past it."""
        import re

        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        kind_clause, kind_params = _chunk_kind_clause(chunk_kind)
        path_clause = "AND path LIKE ? ESCAPE '\\'" if path_prefix else ""
        params: list[Any] = []
        if path_prefix:
            params.append(_escape_like(path_prefix.rstrip("/\\")) + "%")
        params.extend(kind_params)

        with self._lock:
            cursor = self._conn.execute(
                f"SELECT * FROM chunks WHERE 1=1 {path_clause} {kind_clause}",
                params,
            )
            try:
                results: list[dict[str, Any]] = []
                for row in cursor:
                    if rx.search(row["content"]) or (row["name"] and rx.search(row["name"])):
                        results.append(_row_to_dict(row))
                        if len(results) >= top_k:
                            break
                return results
            finally:
                cursor.close()

    def find_by_message(self, message: str, limit: int = 20) -> dict[str, Any]:
        """Finds the source literal for a pasted runtime message, including
        through format holes (skeleton tier). Every cap, skip, or exclusion
        is reported via `note`; never a silent empty result."""
        try:
            with self._lock:
                has_chunks = self._conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None
                literal_index_live = self._conn.execute(
                    "SELECT 1 FROM meta WHERE key=?", (_LITERAL_INDEX_META_KEY,)
                ).fetchone() is not None
                if has_chunks and not literal_index_live:
                    return {
                        "results": [],
                        "truncated": False,
                        "note": _partial_index_note(self._conn),
                    }

                message = message or ""
                join = (
                    "SELECT cl.chunk_id, cl.text, cl.skeleton, cl.line, "
                    "c.path AS path, c.name AS name, "
                    "c.start_line AS c_start_line, c.end_line AS c_end_line "
                    "FROM chunk_literals cl JOIN chunks c ON c.id = cl.chunk_id "
                )

                # Tier 1a: literal is a substring of the message or vice
                # versa (covers an elided/truncated log fragment). Skipped
                # above _SUBSTRING_TIER_MAX_MESSAGE_LEN, announced below.
                tier1a_rows: list[sqlite3.Row] = []
                substring_tier_skipped = len(message) >= _SUBSTRING_TIER_MAX_MESSAGE_LEN
                if 0 < len(message) and not substring_tier_skipped:
                    where = (
                        "WHERE length(cl.text) >= 6 AND "
                        f"(instr(?, cl.text) > 0 OR instr(?, {_FIRST_LINE_SQL}) > 0)"
                    )
                    params: list[Any] = [message, message]
                    if len(message) >= 6:
                        where += f" OR instr(cl.text, ?) > 0 OR instr({_FIRST_LINE_SQL}, ?) > 0"
                        params.extend([message, message])
                    tier1a_rows = self._conn.execute(join + where, params).fetchall()

                # FTS candidate pool keyed off the message's rarest tokens,
                # quoted so words like AND/OR/NOT match literally instead of
                # parsing as FTS operators. Capped at _TOKEN_POOL_CAP, announced.
                all_tokens = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", message)),
                                     key=len, reverse=True)
                token_cap_applied = len(all_tokens) > _TOKEN_POOL_CAP
                tokens = all_tokens[:_TOKEN_POOL_CAP]
                candidate_rows: list[sqlite3.Row] = []
                candidates_truncated = False
                fts_degraded = False
                if tokens:
                    fts_query = " OR ".join(f'"{t}"' for t in tokens)
                    try:
                        rows = self._conn.execute(
                            "SELECT cl.chunk_id, cl.text, cl.skeleton, cl.line, "
                            "c.path AS path, c.name AS name, "
                            "c.start_line AS c_start_line, c.end_line AS c_end_line "
                            "FROM literals_fts "
                            "JOIN chunk_literals cl ON cl.rowid = literals_fts.rowid "
                            "JOIN chunks c ON c.id = cl.chunk_id "
                            "WHERE literals_fts MATCH ? "
                            "ORDER BY bm25(literals_fts) LIMIT ?",
                            (fts_query, _LITERAL_CANDIDATE_CAP + 1),
                        ).fetchall()
                        candidates_truncated = len(rows) > _LITERAL_CANDIDATE_CAP
                        candidate_rows = rows[:_LITERAL_CANDIDATE_CAP]
                    except sqlite3.OperationalError:
                        candidate_rows = []
                        fts_degraded = True  # malformed FTS query: degrade, don't 500

            # Lock released above. Verification runs here so a pathological
            # query can't block every other request on the store lock.

            seen: set[tuple[str, int, str]] = set()
            exact_rows: list[sqlite3.Row] = []

            def _key(row: sqlite3.Row) -> tuple[str, int, str]:
                return (row["chunk_id"], row["line"] or 0, row["text"])

            def _text_matches(text: str) -> bool:
                if text in message or (len(message) >= 6 and message in text):
                    return True
                first = _first_line(text)
                if first is text:
                    return False
                return first in message or (len(message) >= 6 and message in first)

            # Rows that substring-matched but failed _passes_exact_gate stay
            # eligible for the skeleton tier below (not fully rejected), but
            # count once into the note.
            low_confidence_filtered: set[tuple[str, int, str]] = set()

            for row in tier1a_rows:
                k = _key(row)
                if k in seen:
                    continue
                if _passes_exact_gate(row["text"], message):
                    seen.add(k)
                    exact_rows.append(row)
                else:
                    low_confidence_filtered.add(k)

            for row in candidate_rows:
                k = _key(row)
                if k in seen or len(row["text"]) < 6 or not _text_matches(row["text"]):
                    continue
                if _passes_exact_gate(row["text"], message):
                    seen.add(k)
                    exact_rows.append(row)
                else:
                    low_confidence_filtered.add(k)

            # Length/hole caps apply after the cheap prefilter so excluded
            # candidates still count into the note. Budget exhaustion wins
            # priority over a flat exclusion: "may exist" beats "excluded".
            message_too_long = len(message) > _SKELETON_MAX_MESSAGE_LEN
            skeleton_rows: list[sqlite3.Row] = []
            skeleton_excluded_length = 0
            skeleton_excluded_holes = 0
            skeleton_budget_exhausted = 0
            query_budget = _SKELETON_QUERY_BUDGET
            for row in candidate_rows:
                k = _key(row)
                if k in seen or not row["skeleton"]:
                    continue
                matched = False
                any_attempted = False
                length_blocked = False
                hole_blocked = False
                exhausted = False
                remaining_budget = min(_SKELETON_WORK_BUDGET, query_budget)
                for variant in _skeleton_candidates(row["skeleton"]):
                    fragment = _longest_skeleton_fragment(variant)
                    if len(fragment) < 6 or fragment not in message:
                        continue
                    if message_too_long:
                        length_blocked = True
                        continue
                    if variant.count(_HOLE_SENTINEL) > _SKELETON_MAX_HOLES:
                        hole_blocked = True
                        continue
                    any_attempted = True
                    if remaining_budget <= 0:
                        exhausted = True
                        continue
                    is_match, variant_exhausted, calls_used = _skeleton_match(
                        variant, message, remaining_budget,
                    )
                    remaining_budget -= calls_used
                    query_budget -= calls_used
                    if variant_exhausted:
                        exhausted = True
                        continue
                    if is_match:
                        matched = True
                        break
                if matched:
                    seen.add(k)
                    skeleton_rows.append(row)
                elif exhausted:
                    skeleton_budget_exhausted += 1
                elif length_blocked and not any_attempted:
                    skeleton_excluded_length += 1
                elif hole_blocked and not any_attempted:
                    skeleton_excluded_holes += 1

            exact_rows = _dedupe_literal_rows(exact_rows)
            skeleton_rows = _dedupe_literal_rows(skeleton_rows)

            exact_rows.sort(key=lambda r: (r["path"], r["line"] or 0))
            skeleton_rows.sort(key=lambda r: (
                -len(_longest_skeleton_fragment(r["skeleton"])), r["path"], r["line"] or 0
            ))

            def _hit(row: sqlite3.Row, kind: str) -> dict[str, Any]:
                out = {
                    "path": row["path"],
                    "line": row["line"],
                    "chunk_id": row["chunk_id"],
                    "name": row["name"],
                    "matched_literal": row["text"],
                    "match_kind": kind,
                }
                if kind == "skeleton":
                    out["skeleton"] = row["skeleton"]
                return out

            results = [_hit(r, "exact") for r in exact_rows]
            results += [_hit(r, "skeleton") for r in skeleton_rows]

            total = len(results)
            truncated = total > limit
            results = results[:limit]

            notes: list[str] = []
            if truncated:
                notes.append(f"{total - limit} additional match(es) not shown (limit={limit})")
            if fts_degraded:
                notes.append(
                    "message could not be parsed for the token/skeleton tier "
                    "(unusual characters) — only the direct-substring tier applied"
                )
            if low_confidence_filtered:
                notes.append(
                    f"{len(low_confidence_filtered)} low-confidence match(es) filtered "
                    "(a short literal appearing only as a coincidental substring, not "
                    "the message itself)"
                )
            if skeleton_excluded_length:
                notes.append(
                    f"{skeleton_excluded_length} candidate(s) excluded from skeleton "
                    f"verification (message over {_SKELETON_MAX_MESSAGE_LEN} chars) "
                    "— try a shorter/more specific excerpt"
                )
            if skeleton_excluded_holes:
                notes.append(
                    f"{skeleton_excluded_holes} candidate(s) excluded from skeleton "
                    f"verification (template has more than {_SKELETON_MAX_HOLES} format holes)"
                )
            if skeleton_budget_exhausted:
                notes.append(
                    f"verification budget exhausted for {skeleton_budget_exhausted} "
                    "candidate(s) — match may exist"
                )
            # These two caps are silent by nature (nothing was evaluated to
            # count), so announce them whenever the result is thin, or a
            # false "no matches" reads as a clean miss.
            if substring_tier_skipped and total < 3:
                notes.append(
                    f"message is {_SUBSTRING_TIER_MAX_MESSAGE_LEN}+ chars — the "
                    "direct-substring tier was skipped; a verbatim literal in this "
                    "message may have been missed"
                )
            if token_cap_applied and total < 3:
                notes.append(
                    f"message has more than {_TOKEN_POOL_CAP} distinct significant "
                    f"tokens — only the {_TOKEN_POOL_CAP} longest were used for the "
                    "token/skeleton candidate search; a match may have been missed"
                )
            if candidates_truncated:
                # Announced even when a hit was found anyway: a silent cap
                # must always say so, not just when it explains emptiness.
                if results:
                    notes.append(
                        f"token/skeleton candidate pool capped at the top "
                        f"{_LITERAL_CANDIDATE_CAP} most relevant literals for "
                        "these tokens — some lower-relevance literals were not considered"
                    )
                else:
                    notes.append(
                        f"no match in the top {_LITERAL_CANDIDATE_CAP} most relevant "
                        "literals for these tokens — try a more specific fragment"
                    )
            elif (not results and not fts_degraded and not skeleton_excluded_length
                  and not skeleton_excluded_holes and not skeleton_budget_exhausted
                  and not low_confidence_filtered
                  and not (substring_tier_skipped and total < 3)
                  and not (token_cap_applied and total < 3)):
                notes.append("no literal matches for this message")

            out: dict[str, Any] = {"results": results, "truncated": truncated}
            if notes:
                out["note"] = "; ".join(notes)
            return out
        except sqlite3.OperationalError as e:
            # A lookup must never 500: a capped result must say so, and a
            # crash is just as much a silent failure as an empty answer.
            return {
                "results": [],
                "truncated": False,
                "note": f"literal index unavailable ({e})",
            }

    def get_named_chunks(self, path_prefix: str | None = None) -> list[dict[str, Any]]:
        """Chunks with non-null names, including `content` and `metadata`
        for build_refs. Display-only callers should use get_named_chunks_meta
        instead, to avoid materialising every chunk's body."""
        path_clause = "AND path LIKE ? ESCAPE '\\'" if path_prefix else ""
        params: list[Any] = []
        if path_prefix:
            params.append(_escape_like(path_prefix.rstrip("/\\")) + "%")
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, path, language, name, chunk_type, start_line, end_line, content, metadata
                FROM chunks
                WHERE name IS NOT NULL
                  {path_clause}
                ORDER BY path, start_line
                """,
                params,
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_named_chunks_meta(self, path_prefix: str | None = None) -> list[dict[str, Any]]:
        """Like get_named_chunks but skips `content`, cutting peak memory a
        lot on a large corpus. `language` is kept for callers that colour
        by language."""
        path_clause = "AND path LIKE ? ESCAPE '\\'" if path_prefix else ""
        params: list[Any] = []
        if path_prefix:
            params.append(_escape_like(path_prefix.rstrip("/\\")) + "%")
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, path, language, name, chunk_type, start_line, end_line
                FROM chunks
                WHERE name IS NOT NULL
                  {path_clause}
                ORDER BY path, start_line
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chunk_defs_by_names(self, names: list[str]) -> dict[str, list[tuple[str, str]]]:
        """name -> [(chunk_id, language)] for build_refs' incremental name
        resolution, scoped to a candidate set via idx_chunks_name instead
        of a full-corpus scan."""
        if not names:
            return {}
        out: dict[str, list[tuple[str, str]]] = defaultdict(list)
        with self._lock:
            for i in range(0, len(names), 900):
                batch = names[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT name, id, language FROM chunks WHERE name IN ({placeholders})",
                    batch,
                ).fetchall()
                for r in rows:
                    out[r["name"]].append((r["id"], r["language"]))
        return dict(out)

    def get_chunks_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Bulk-fetch chunks by id. Order of returned chunks is not guaranteed
        to match `ids`; caller should re-order if needed. Batches at 900 to
        stay under SQLite's variable limit."""
        if not ids:
            return []
        results: list[dict[str, Any]] = []
        with self._lock:
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT * FROM chunks WHERE id IN ({placeholders})",
                    batch,
                ).fetchall()
                results.extend(_row_to_dict(r) for r in rows)
        return results

    def get_chunks_by_path_and_names(
        self, path: str, names: list[str], limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Chunks at `path` named in `names` (header/impl pairing by shared
        symbol name). Order not guaranteed; `limit` caps at the SQL level."""
        if not names:
            return []
        results: list[dict[str, Any]] = []
        with self._lock:
            for i in range(0, len(names), 900):
                batch = names[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                sql = (
                    f"SELECT * FROM chunks WHERE path = ? AND name IN ({placeholders})"
                )
                params: list[Any] = [path, *batch]
                if limit is not None:
                    sql += " LIMIT ?"
                    params.append(limit)
                rows = self._conn.execute(sql, params).fetchall()
                results.extend(_row_to_dict(r) for r in rows)
        return results

    def get_chunks_by_path_line_range(
        self, path: str, start_line: int, end_line: int,
    ) -> list[dict[str, Any]]:
        """Chunks at `path` overlapping (start_line, end_line]. Used to
        fetch continuation chunks when a symbol's span outgrows the one
        chunk_id it points at."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE path = ? AND end_line > ? AND start_line <= ? "
                "ORDER BY start_line",
                (path, start_line, end_line),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            file_count  = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            chunk_count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            db_size     = self.db_path.stat().st_size if self.db_path.exists() else 0
            newest_row  = self._conn.execute(
                "SELECT MAX(indexed_at) AS t FROM files"
            ).fetchone()
            root_row    = self._conn.execute(
                "SELECT value FROM meta WHERE key='indexed_root'"
            ).fetchone()
            model_row   = self._conn.execute(
                "SELECT value FROM meta WHERE key='embedding_model'"
            ).fetchone()
            return {
                "files":             file_count,
                "chunks":            chunk_count,
                "db_size_mb":        round(db_size / 1024 / 1024, 2),
                "embedding_dim":     self._dim,
                "embedding_model":   model_row["value"] if model_row else None,
                "indexed_root":      root_row["value"] if root_row else None,
                "never_indexed":     chunk_count == 0,
                "newest_indexed_at": newest_row["t"] if newest_row else None,
            }

    def language_counts(self) -> dict[str, int]:
        """Chunk count per language (language may be NULL for text-fallback
        chunks; reported under the None key)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT language, COUNT(*) AS n FROM chunks GROUP BY language"
            ).fetchall()
        return {r["language"]: r["n"] for r in rows}

    def path_family_rows(self) -> list[tuple[str, str | None]]:
        """(path, language) for every chunk; feeds chunking.family_breakdown
        for the dominance warning."""
        with self._lock:
            rows = self._conn.execute("SELECT path, language FROM chunks").fetchall()
        return [(r["path"], r["language"]) for r in rows]

    def parsed_file_count(self) -> int:
        """Number of indexed files that produced at least one chunk."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(DISTINCT path) FROM chunks"
            ).fetchone()[0]

    def tracked_file_count(self) -> int:
        """Row count in `files`. Used by chunker.py's literal-index
        completeness gate to distinguish "zero skips" from "covered every
        tracked file" (see _LITERAL_INDEX_META_KEY)."""
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    # ------------------------------------------------------------------
    # Semantic k-NN graph (GraphRAG)
    # ------------------------------------------------------------------

    def clear_neighbors(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunk_neighbors")

    def insert_neighbors(self, rows: list[tuple[str, str, float]]) -> None:
        """Bulk-insert (chunk_id, neighbor_id, distance) edges. Ignores duplicates."""
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO chunk_neighbors(chunk_id, neighbor_id, distance) VALUES(?,?,?)",
                rows,
            )

    def get_neighbors(self, chunk_id: str, limit: int | None = None) -> list[tuple[str, float]]:
        """Return semantic neighbors of a chunk, sorted by distance ascending."""
        sql = "SELECT neighbor_id, distance FROM chunk_neighbors WHERE chunk_id=? ORDER BY distance"
        params: list[Any] = [chunk_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [(r["neighbor_id"], r["distance"]) for r in rows]

    def get_neighbor_edges_touching(self, chunk_ids: list[str]) -> list[tuple[str, str, float]]:
        """k-NN edges touching `chunk_ids` on EITHER side. chunk_neighbors
        rows are stored one-directionally; both directions are returned
        here since the caller treats the edge as undirected."""
        if not chunk_ids:
            return []
        results: list[tuple[str, str, float]] = []
        with self._lock:
            # Sliced at 450 (not 900): this statement binds batch+batch, so a
            # 900-slice would bind up to 1800 params, over SQLite's default
            # SQLITE_MAX_VARIABLE_NUMBER (999) -> "too many SQL variables".
            for i in range(0, len(chunk_ids), 450):
                batch = chunk_ids[i:i + 450]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT chunk_id, neighbor_id, distance FROM chunk_neighbors "
                    f"WHERE chunk_id IN ({placeholders}) OR neighbor_id IN ({placeholders})",
                    batch + batch,
                ).fetchall()
                results.extend((r[0], r[1], r[2]) for r in rows)
        return results

    def count_neighbors(self) -> int:
        """Total edge count. Used to decide whether a graph exists at all
        before attempting an incremental update."""
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunk_neighbors").fetchone()[0]

    def count_chunks(self) -> int:
        """Total row count in chunks. Used by the incremental k-NN path to size
        the changed/deleted batch against the corpus and to detect whether the
        effective top-k was capped (small corpus) before or after the mutation."""
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_chunk_ids_by_neighbor(self, neighbor_ids: list[str]) -> set[str]:
        """chunk_ids whose neighbor list includes any of `neighbor_ids`:
        the "damaged" rows after a neighbor was deleted. Index lookup via
        idx_chunk_neighbors_neighbor, not a scan."""
        if not neighbor_ids:
            return set()
        out: set[str] = set()
        with self._lock:
            for i in range(0, len(neighbor_ids), 900):
                batch = neighbor_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT DISTINCT chunk_id FROM chunk_neighbors WHERE neighbor_id IN ({placeholders})",
                    batch,
                ).fetchall()
                out.update(r[0] for r in rows)
        return out

    def delete_neighbors_touching(self, ids: list[str]) -> int:
        """Delete every chunk_neighbors row referencing any of `ids`, as
        source or target. Returns rows deleted."""
        if not ids:
            return 0
        deleted = 0
        with self._lock:
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                cur = self._conn.execute(
                    f"DELETE FROM chunk_neighbors WHERE chunk_id IN ({placeholders})",
                    batch,
                )
                deleted += cur.rowcount if cur.rowcount is not None else 0
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                cur = self._conn.execute(
                    f"DELETE FROM chunk_neighbors WHERE neighbor_id IN ({placeholders})",
                    batch,
                )
                deleted += cur.rowcount if cur.rowcount is not None else 0
            return deleted

    def delete_neighbors_from(self, chunk_ids: list[str]) -> int:
        """Delete outgoing chunk_neighbors rows for `chunk_ids` only. Used
        before a full per-row recompute, so top-k inserts onto a clean slate."""
        if not chunk_ids:
            return 0
        deleted = 0
        with self._lock:
            for i in range(0, len(chunk_ids), 900):
                batch = chunk_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                cur = self._conn.execute(
                    f"DELETE FROM chunk_neighbors WHERE chunk_id IN ({placeholders})",
                    batch,
                )
                deleted += cur.rowcount if cur.rowcount is not None else 0
            return deleted

    def purge_orphan_neighbors(self) -> int:
        """Delete chunk_neighbors rows whose chunk_id or neighbor_id no
        longer exists in `chunks`. Self-heals a DB from before the
        graph-maintenance fix; returns rows deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM chunk_neighbors WHERE chunk_id NOT IN (SELECT id FROM chunks) "
                "OR neighbor_id NOT IN (SELECT id FROM chunks)"
            )
            return cur.rowcount if cur.rowcount is not None else 0

    def get_neighbor_worst_distances(self) -> dict[str, tuple[float, str, int]]:
        """{chunk_id: (worst_distance, worst_neighbor_id, count)}. Tie-break
        (largest neighbor_id) must match trim_neighbors' rule exactly, or
        an incremental update can diverge from a full rebuild."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT chunk_id, distance AS worst, neighbor_id AS worst_id, cnt
                FROM (
                    SELECT
                        chunk_id, distance, neighbor_id,
                        COUNT(*) OVER (PARTITION BY chunk_id) AS cnt,
                        ROW_NUMBER() OVER (
                            PARTITION BY chunk_id
                            ORDER BY distance DESC, neighbor_id DESC
                        ) AS rn
                    FROM chunk_neighbors
                )
                WHERE rn = 1
                """
            ).fetchall()
        return {r["chunk_id"]: (r["worst"], r["worst_id"], r["cnt"]) for r in rows}

    def trim_neighbors(self, chunk_id: str, k: int) -> int:
        """Keep the k closest neighbor rows for `chunk_id`, deleting the
        rest. Tie-break (smaller neighbor_id kept) must match the bulk
        rebuild's rule exactly, or incremental drifts from a full rebuild."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT neighbor_id FROM chunk_neighbors WHERE chunk_id=? "
                "ORDER BY distance, neighbor_id",
                (chunk_id,),
            ).fetchall()
            if len(rows) <= k:
                return 0
            excess = [r["neighbor_id"] for r in rows[k:]]
            placeholders = ",".join("?" * len(excess))
            self._conn.execute(
                f"DELETE FROM chunk_neighbors WHERE chunk_id=? AND neighbor_id IN ({placeholders})",
                [chunk_id] + excess,
            )
            return len(excess)

    def get_all_neighbors(self) -> list[tuple[str, str]]:
        """Every (chunk_id, neighbor_id) edge, distance omitted; use
        get_neighbors for a single chunk's distances."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, neighbor_id FROM chunk_neighbors"
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_all_chunk_ids(self) -> list[str]:
        """Return every chunk id in insertion order. Used by build_neighbors to
        iterate over the corpus."""
        with self._lock:
            rows = self._conn.execute("SELECT id FROM chunks ORDER BY rowid").fetchall()
        return [r["id"] for r in rows]

    def get_all_int8_embeddings(self) -> tuple[list[str], bytes, int]:
        """Bulk-load every chunk's int8 embedding as one packed blob, in
        the same order as `ids`. Lets build_neighbors compute all-pairs
        k-NN as one matmul instead of N per-query MATCH round-trips."""
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT id, embedding FROM chunk_vecs"
                ).fetchall()
            except sqlite3.OperationalError as e:
                # Only "table doesn't exist" is a legitimate empty corpus.
                # Any other error must propagate, or build_neighbors could
                # wipe a healthy graph thinking the corpus is empty.
                if "no such table" in str(e).lower():
                    return [], b"", self._dim or 0
                raise
        if not rows:
            return [], b"", self._dim or 0
        ids = [r["id"] for r in rows]
        blob = b"".join(r["embedding"] for r in rows)
        dim = len(rows[0]["embedding"])  # bytes per int8 vector == dim
        return ids, blob, dim

    def get_int8_embeddings_by_ids(self, ids: list[str]) -> dict[str, bytes]:
        """Raw int8 embedding blob per id that has one (missing ids are
        omitted). Cosine is scale-invariant, so callers use the blob
        directly, no dequantisation needed."""
        if not ids:
            return {}
        out: dict[str, bytes] = {}
        with self._lock:
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                try:
                    rows = self._conn.execute(
                        f"SELECT id, embedding FROM chunk_vecs WHERE id IN ({placeholders})",
                        batch,
                    ).fetchall()
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e).lower():
                        return {}
                    raise
                for r in rows:
                    out[r["id"]] = r["embedding"]
        return out

    def get_vectors_for_chunks(self, ids: list[str]) -> dict[str, np.ndarray]:
        """Dequantized float32 vectors per chunk id (missing ids omitted).
        Used by the near-duplicate detector: same int8 representation
        search itself uses, no embedder round-trip needed."""
        blobs = self.get_int8_embeddings_by_ids(ids)
        return {
            cid: np.frombuffer(blob, dtype=np.int8).astype(np.float32)
            for cid, blob in blobs.items()
        }

    # ------------------------------------------------------------------
    # Folder summaries
    # ------------------------------------------------------------------

    def get_all_files(self) -> list[dict[str, Any]]:
        """Return all (path, content_hash) rows from files."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, content_hash FROM files"
            ).fetchall()
        return [{"path": r["path"], "content_hash": r["content_hash"]} for r in rows]

    def get_folder_summary(self, path: str) -> dict[str, Any] | None:
        """Return the row for a folder, or None if not present.
        `summary_embedding` is returned as a `list[float]` (unpacked from BLOB)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT path, summary, summary_embedding, content_hash, generated_at "
                "FROM folder_summaries WHERE path=?",
                (path,),
            ).fetchone()
        if row is None:
            return None
        return {
            "path":              row["path"],
            "summary":           row["summary"],
            "summary_embedding": _unpack_f32(row["summary_embedding"]),
            "content_hash":      row["content_hash"],
            "generated_at":      row["generated_at"],
        }

    def upsert_folder_summary(
        self, path: str, summary: str, embedding: list[float], content_hash: str,
    ) -> None:
        """Insert or replace a folder's summary row. Embedding is stored as
        packed float32 (4 bytes per element)."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO folder_summaries
                   (path, summary, summary_embedding, content_hash, generated_at)
                   VALUES(?,?,?,?,?)""",
                (path, summary, _pack_f32(embedding), content_hash, time.time()),
            )

    def delete_folder_summary(self, path: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM folder_summaries WHERE path=?", (path,))

    def get_all_folder_paths(self) -> list[str]:
        """Return every folder path that currently has a summary."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT path FROM folder_summaries"
            ).fetchall()
        return [r["path"] for r in rows]

    def get_folder_embeddings(self, paths: list[str]) -> dict[str, list[float]]:
        """Bulk-fetch folder summary embeddings. Missing paths are absent from
        the result (caller should treat as folder_sim = 0). Batches the IN
        clause at 900 to stay under SQLite's variable limit."""
        if not paths:
            return {}
        result: dict[str, list[float]] = {}
        with self._lock:
            for i in range(0, len(paths), 900):
                batch = paths[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT path, summary_embedding FROM folder_summaries WHERE path IN ({placeholders})",
                    batch,
                ).fetchall()
                for r in rows:
                    result[r["path"]] = _unpack_f32(r["summary_embedding"])
        return result

    # ------------------------------------------------------------------
    # Cross-reference graph
    # ------------------------------------------------------------------

    def clear_refs(self) -> None:
        """Delete all cross-reference edges and their precomputed
        in-degree, which is always cleared alongside chunk_refs."""
        with self._lock:
            self._conn.execute("DELETE FROM chunk_refs")
            self._conn.execute("DELETE FROM chunk_indegree")

    def insert_refs(self, refs: list[tuple[str, str]] | list[tuple[str, str, str]]) -> None:
        """Bulk-insert (from_id, to_id[, edge_type]) edges. Ignores duplicates.
        A 2-tuple defaults edge_type to 'mentions' (table default) for callers
        that don't carry a type."""
        if not refs:
            return
        rows3 = [r if len(r) == 3 else (r[0], r[1], "mentions") for r in refs]
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO chunk_refs(from_id, to_id, edge_type) VALUES(?,?,?)",
                rows3,
            )

    def count_refs(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunk_refs").fetchone()[0]

    # ------------------------------------------------------------------
    # Graph v2 hierarchy (Stage 1)
    # ------------------------------------------------------------------

    def rebuild_hierarchy(self) -> dict[str, int]:
        """Full rebuild of graph_nodes + graph_edges, DELETE then re-derive
        from scratch every call, never incremental. Chunks stay out of
        graph_nodes so chunk_pagerank stays chunk-only and comparable."""
        with self._lock:
            self._conn.execute("DELETE FROM graph_edges")
            self._conn.execute("DELETE FROM graph_nodes")

            file_paths = [r["path"] for r in
                          self._conn.execute("SELECT path FROM files").fetchall()]

            dir_paths: set[str] = set()
            for fp in file_paths:
                d = posixpath.dirname(fp) or "."
                while True:
                    dir_paths.add(d)
                    if d == ".":
                        break
                    parent = posixpath.dirname(d) or "."
                    d = parent

            def _dir_parent_id(d: str) -> str | None:
                if d == ".":
                    return None
                parent = posixpath.dirname(d) or "."
                return "dir:" + parent

            dir_rows = [
                ("dir:" + d, "dir", d, _dir_parent_id(d))
                for d in dir_paths
            ]
            file_rows = [
                ("file:" + fp, "file", fp, "dir:" + (posixpath.dirname(fp) or "."))
                for fp in file_paths
            ]

            if dir_rows or file_rows:
                self._conn.executemany(
                    "INSERT INTO graph_nodes(id, kind, path, parent_id) VALUES(?,?,?,?)",
                    dir_rows + file_rows,
                )

            edge_rows = []
            for d in dir_paths:
                parent_id = _dir_parent_id(d)
                if parent_id is not None:
                    edge_rows.append((parent_id, "dir:" + d, "contains"))
            for fp in file_paths:
                parent_id = "dir:" + (posixpath.dirname(fp) or ".")
                edge_rows.append((parent_id, "file:" + fp, "contains"))

            if edge_rows:
                self._conn.executemany(
                    "INSERT INTO graph_edges(from_id, to_id, edge_type) VALUES(?,?,?)",
                    edge_rows,
                )

            self._conn.execute(
                "INSERT INTO graph_edges(from_id, to_id, edge_type) "
                "SELECT 'file:' || path, id, 'contains' FROM chunks"
            )

            # Group files by (dir, stem), pair every header against every
            # impl in that group. Both directions are emitted so a lookup
            # needs only one query direction (get_paired_files).
            by_dir_stem: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for fp in file_paths:
                d = posixpath.dirname(fp) or "."
                base = posixpath.basename(fp)
                stem, dot, ext = base.rpartition(".")
                if not dot:
                    continue  # no extension: nothing to pair on
                by_dir_stem[(d, stem)][ext.lower()].append(fp)

            paired_edge_rows = []
            for (_d, _stem), ext_map in by_dir_stem.items():
                headers = [fp for ext in HEADER_EXTS for fp in ext_map.get(ext, [])]
                impls = [fp for ext in IMPL_EXTS for fp in ext_map.get(ext, [])]
                for h in headers:
                    for impl in impls:
                        paired_edge_rows.append(("file:" + h, "file:" + impl, "paired"))
                        paired_edge_rows.append(("file:" + impl, "file:" + h, "paired"))

            if paired_edge_rows:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO graph_edges(from_id, to_id, edge_type) "
                    "VALUES(?,?,?)",
                    paired_edge_rows,
                )

            node_count = self._conn.execute(
                "SELECT COUNT(*) FROM graph_nodes"
            ).fetchone()[0]
            edge_count = self._conn.execute(
                "SELECT COUNT(*) FROM graph_edges"
            ).fetchone()[0]

            self._conn.commit()

        return {"nodes": node_count, "edges": edge_count}

    def get_graph_node(self, node_id: str) -> dict | None:
        """One graph_nodes row as a dict, or None if node_id doesn't exist."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, kind, path, parent_id FROM graph_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_graph_children(self, from_id: str, edge_type: str = "contains") -> list[str]:
        """to_ids of every graph_edges row out of `from_id`, ordered by to_id."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT to_id FROM graph_edges WHERE from_id = ? AND edge_type = ? "
                "ORDER BY to_id",
                (from_id, edge_type),
            ).fetchall()
        return [r["to_id"] for r in rows]

    def count_graph_nodes(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]

    def count_graph_edges(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]

    def get_graph_dirs(self) -> list[dict]:
        """Every graph_nodes row with kind='dir', as {id, path, parent_id}
        dicts ordered by path."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, path, parent_id FROM graph_nodes "
                "WHERE kind = 'dir' ORDER BY path"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_graph_dir_file_counts(self) -> dict[str, int]:
        """{parent_id: direct file count}, not a subtree total."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT parent_id, COUNT(*) AS n FROM graph_nodes "
                "WHERE kind = 'file' GROUP BY parent_id"
            ).fetchall()
        return {r["parent_id"]: r["n"] for r in rows}

    def get_paired_files(self, paths: list[str]) -> dict[str, list[str]]:
        """Batched header/impl companion lookup: `paths` are file paths,
        resolved via outgoing 'paired' graph_edges. Paths with no
        companion are absent from the result."""
        if not paths:
            return {}
        result: dict[str, list[str]] = defaultdict(list)
        id_to_path = {"file:" + p: p for p in paths}
        from_ids = list(id_to_path.keys())
        with self._lock:
            for i in range(0, len(from_ids), 900):
                batch = from_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"""SELECT ge.from_id AS from_id, gn.path AS to_path
                        FROM graph_edges ge
                        JOIN graph_nodes gn ON gn.id = ge.to_id
                        WHERE ge.edge_type = 'paired' AND ge.from_id IN ({placeholders})""",
                    batch,
                ).fetchall()
                for r in rows:
                    src_path = id_to_path.get(r["from_id"])
                    if src_path is not None:
                        result[src_path].append(r["to_path"])
        return dict(result)

    # ------------------------------------------------------------------
    # Persisted global PageRank
    # ------------------------------------------------------------------

    def save_pagerank(self, scores: dict[str, float]) -> None:
        """Replace chunk_pagerank with `scores` atomically (DELETE then
        bulk-insert in one transaction, no half-written table). Called
        once at index time, never from a query path."""
        with self._lock:
            self._conn.execute("DELETE FROM chunk_pagerank")
            self._conn.executemany(
                "INSERT INTO chunk_pagerank(chunk_id, score) VALUES(?,?)",
                list(scores.items()),
            )
            self._conn.commit()

    def load_pagerank(self) -> dict[str, float]:
        """Return the full persisted {chunk_id: score} PageRank map. Empty
        dict on a DB that predates persisted PageRank (or was never indexed)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, score FROM chunk_pagerank"
            ).fetchall()
        return {r["chunk_id"]: r["score"] for r in rows}

    def get_pagerank_for_chunks(self, ids: list[str]) -> dict[str, float]:
        """Scoped load_pagerank: {chunk_id: score} for `ids` only (missing
        ids are absent; caller treats as 0.0)."""
        if not ids:
            return {}
        out: dict[str, float] = {}
        with self._lock:
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT chunk_id, score FROM chunk_pagerank WHERE chunk_id IN ({placeholders})",
                    batch,
                ).fetchall()
                for r in rows:
                    out[r["chunk_id"]] = r["score"]
        return out

    def get_top_pagerank_chunk_for_path(self, path: str) -> dict[str, Any] | None:
        """Highest-PageRank chunk at `path`. Missing chunk_pagerank rows
        sort as 0.0 via LEFT JOIN rather than being excluded, so a file
        with no scored chunks still returns its first chunk."""
        with self._lock:
            row = self._conn.execute(
                """SELECT c.*, COALESCE(pr.score, 0.0) AS _pagerank
                   FROM chunks c
                   LEFT JOIN chunk_pagerank pr ON pr.chunk_id = c.id
                   WHERE c.path = ?
                   ORDER BY _pagerank DESC, c.start_line ASC
                   LIMIT 1""",
                (path,),
            ).fetchone()
        return _row_to_dict(row) if row is not None else None

    # ------------------------------------------------------------------
    # Persisted in-degree
    # ------------------------------------------------------------------

    def save_indegree(self, counts: dict[tuple[str, str], int]) -> None:
        """Replace chunk_indegree with `counts` atomically. Built from the
        in-memory refs list, never via GROUP BY over chunk_refs (that scan
        is the slow operation get_hubs' guard exists for)."""
        with self._lock:
            self._conn.execute("DELETE FROM chunk_indegree")
            self._conn.executemany(
                "INSERT INTO chunk_indegree(chunk_id, edge_type, n) VALUES(?,?,?)",
                [(cid, edge_type, n) for (cid, edge_type), n in counts.items()],
            )
            self._conn.commit()

    def has_indegree(self) -> bool:
        """True if chunk_indegree fully mirrors chunk_refs, not just
        nonempty. A scoped refresh_indegree delta is only safe when this
        is true; otherwise the untouched majority looks legitimately empty."""
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM chunk_indegree LIMIT 1"
            ).fetchone() is not None

    def refresh_indegree(self, chunk_ids: list[str]) -> None:
        """Recompute chunk_indegree for exactly `chunk_ids` from current
        chunk_refs. Only valid when has_indegree() is true first. Doesn't
        commit; caller batches with other writes."""
        if not chunk_ids:
            return
        ids = list(chunk_ids)
        with self._lock:
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                self._conn.execute(
                    f"DELETE FROM chunk_indegree WHERE chunk_id IN ({placeholders})",
                    batch,
                )
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT to_id, edge_type, COUNT(*) AS n FROM chunk_refs "
                    f"WHERE to_id IN ({placeholders}) GROUP BY to_id, edge_type",
                    batch,
                ).fetchall()
                if rows:
                    self._conn.executemany(
                        "INSERT INTO chunk_indegree(chunk_id, edge_type, n) VALUES(?,?,?)",
                        [(r["to_id"], r["edge_type"], r["n"]) for r in rows],
                    )

    def delete_refs_touching(self, ids: list[str]) -> tuple[int, set[str]]:
        """Delete chunk_refs rows where from_id OR to_id is in `ids`.
        Returns (rows_deleted, affected_to_ids), captured before the
        DELETE, so callers know which chunk_indegree rows to refresh."""
        if not ids:
            return 0, set()
        removed = 0
        affected_to_ids: set[str] = set()
        with self._lock:
            # Sliced at 450, not 900 (see get_neighbor_edges_touching: this
            # statement binds batch+batch).
            for i in range(0, len(ids), 450):
                batch = ids[i:i + 450]
                placeholders = ",".join("?" * len(batch))
                where = f"from_id IN ({placeholders}) OR to_id IN ({placeholders})"
                params = batch + batch
                affected_to_ids.update(
                    r[0] for r in self._conn.execute(
                        f"SELECT DISTINCT to_id FROM chunk_refs WHERE {where}", params,
                    ).fetchall()
                )
                cur = self._conn.execute(
                    f"DELETE FROM chunk_refs WHERE {where}", params,
                )
                removed += cur.rowcount
        return removed, affected_to_ids

    def delete_refs_from(self, ids: list[str]) -> tuple[int, set[str]]:
        """Delete chunk_refs rows whose from_id is in `ids`. Returns
        (rows_deleted, affected_to_ids), same contract as delete_refs_touching."""
        if not ids:
            return 0, set()
        removed = 0
        affected_to_ids: set[str] = set()
        with self._lock:
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                where = f"from_id IN ({placeholders})"
                affected_to_ids.update(
                    r[0] for r in self._conn.execute(
                        f"SELECT DISTINCT to_id FROM chunk_refs WHERE {where}", batch,
                    ).fetchall()
                )
                cur = self._conn.execute(
                    f"DELETE FROM chunk_refs WHERE {where}", batch,
                )
                removed += cur.rowcount
        return removed, affected_to_ids

    def delete_xlang_refs_touching(self, ids: list[str]) -> tuple[int, set[str]]:
        """Delete 'xlang' chunk_refs rows touching `ids`. Scoped to that
        edge_type so it never touches a mentions/calls/imports/inherits
        edge sharing the same endpoint. Same return contract as delete_refs_touching."""
        if not ids:
            return 0, set()
        removed = 0
        affected_to_ids: set[str] = set()
        with self._lock:
            # Sliced at 450, not 900 (see get_neighbor_edges_touching: this
            # statement binds batch+batch).
            for i in range(0, len(ids), 450):
                batch = ids[i:i + 450]
                placeholders = ",".join("?" * len(batch))
                where = (
                    f"edge_type='xlang' AND "
                    f"(from_id IN ({placeholders}) OR to_id IN ({placeholders}))"
                )
                params = batch + batch
                affected_to_ids.update(
                    r[0] for r in self._conn.execute(
                        f"SELECT DISTINCT to_id FROM chunk_refs WHERE {where}", params,
                    ).fetchall()
                )
                cur = self._conn.execute(
                    f"DELETE FROM chunk_refs WHERE {where}", params,
                )
                removed += cur.rowcount
        return removed, affected_to_ids

    def purge_orphan_refs(self) -> int:
        """Delete chunk_refs rows whose endpoint no longer exists in
        `chunks`; self-heals dangling edges. Also refreshes chunk_indegree
        for affected ids, but only when has_indegree() is already true."""
        where = (
            "from_id NOT IN (SELECT id FROM chunks) "
            "OR to_id NOT IN (SELECT id FROM chunks)"
        )
        with self._lock:
            affected_to_ids = {
                r[0] for r in self._conn.execute(
                    f"SELECT DISTINCT to_id FROM chunk_refs WHERE {where}"
                ).fetchall()
            }
            cur = self._conn.execute(f"DELETE FROM chunk_refs WHERE {where}")
            removed = cur.rowcount if cur.rowcount is not None else 0
        if affected_to_ids and self.has_indegree():
            self.refresh_indegree(list(affected_to_ids))
        return removed

    def get_chunk_ids_referencing(self, target_ids: list[str]) -> set[str]:
        """from_id values in chunk_refs whose to_id is in `target_ids`.
        Reverse lookup for the incremental build_refs update."""
        if not target_ids:
            return set()
        out: set[str] = set()
        with self._lock:
            for i in range(0, len(target_ids), 900):
                batch = target_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT DISTINCT from_id FROM chunk_refs WHERE to_id IN ({placeholders})",
                    batch,
                ).fetchall()
                out.update(r[0] for r in rows)
        return out

    def find_named_chunks_referencing(self, names: list[str]) -> list[dict[str, Any]]:
        """Named chunks whose content likely contains any of `names`, via
        chunks_fts. A fast pre-filter, not the final match: false positives
        are possible, so callers must re-verify with a word-boundary regex."""
        if not names:
            return []
        results: dict[str, dict[str, Any]] = {}
        with self._lock:
            for i in range(0, len(names), 200):
                batch = names[i:i + 200]
                phrase_query = " OR ".join(
                    '"' + n.replace('"', '""') + '"' for n in batch
                )
                rows = self._conn.execute(
                    """
                    SELECT c.id, c.path, c.language, c.name, c.chunk_type,
                           c.start_line, c.end_line, c.content, c.metadata
                    FROM chunks_fts f
                    JOIN chunks c ON c.id = f.id
                    WHERE chunks_fts MATCH ? AND c.name IS NOT NULL
                    """,
                    (phrase_query,),
                ).fetchall()
                for r in rows:
                    d = _row_to_dict(r)
                    results[d["id"]] = d
        return list(results.values())

    def get_all_refs(self) -> list[tuple[str, str]]:
        """Every (from_id, to_id) edge, type-agnostic; see get_all_refs_typed
        for the typed variant."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT from_id, to_id FROM chunk_refs"
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_all_refs_typed(self) -> list[tuple[str, str, str]]:
        """Return every (from_id, to_id, edge_type) edge in the table."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT from_id, to_id, edge_type FROM chunk_refs"
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def get_module_edge_type_counts(
        self, module_by_chunk: dict[str, str]
    ) -> list[tuple[str, str, str, int]]:
        """Aggregate chunk_refs into module-level (from, to, edge_type,
        count) rows via a SQL GROUP BY, since the table can be tens of
        millions of rows. Self-loops (same module both ends) are dropped."""
        if not module_by_chunk:
            return []
        with self._lock:
            self._conn.execute("DROP TABLE IF EXISTS temp.chunk_module")
            self._conn.execute(
                "CREATE TEMP TABLE chunk_module (id TEXT PRIMARY KEY, module TEXT NOT NULL)"
            )
            self._conn.executemany(
                "INSERT INTO chunk_module(id, module) VALUES(?,?)",
                list(module_by_chunk.items()),
            )
            rows = self._conn.execute(
                """
                SELECT cm1.module AS from_module, cm2.module AS to_module,
                       cr.edge_type AS edge_type, COUNT(*) AS n
                FROM chunk_refs cr
                JOIN chunk_module cm1 ON cr.from_id = cm1.id
                JOIN chunk_module cm2 ON cr.to_id = cm2.id
                WHERE cm1.module != cm2.module
                GROUP BY cm1.module, cm2.module, cr.edge_type
                """
            ).fetchall()
            self._conn.execute("DROP TABLE chunk_module")
        return [(r["from_module"], r["to_module"], r["edge_type"], r["n"]) for r in rows]

    def get_refs_for_chunks(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        """Return edges whose from_id is in chunk_ids. Batches to stay under
        SQLite's default variable limit (~999)."""
        if not chunk_ids:
            return []
        results: list[tuple[str, str]] = []
        with self._lock:
            for i in range(0, len(chunk_ids), 900):
                batch = chunk_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT from_id, to_id FROM chunk_refs WHERE from_id IN ({placeholders})",
                    batch,
                ).fetchall()
                results.extend((r[0], r[1]) for r in rows)
        return results

    def get_refs_to_chunks(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        """Edges whose to_id is in chunk_ids (incoming refs). Mirror of
        get_refs_for_chunks for the other direction."""
        if not chunk_ids:
            return []
        results: list[tuple[str, str]] = []
        with self._lock:
            for i in range(0, len(chunk_ids), 900):
                batch = chunk_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT from_id, to_id FROM chunk_refs WHERE to_id IN ({placeholders})",
                    batch,
                ).fetchall()
                results.extend((r[0], r[1]) for r in rows)
        return results

    def get_refs_for_chunks_typed(self, chunk_ids: list[str]) -> list[tuple[str, str, str]]:
        """Typed variant of get_refs_for_chunks: edges whose from_id is in
        chunk_ids, with edge_type. Used by trace_path's BFS to prefer typed
        edges (calls/imports/inherits) over 'mentions' when capping fanout."""
        if not chunk_ids:
            return []
        results: list[tuple[str, str, str]] = []
        with self._lock:
            for i in range(0, len(chunk_ids), 900):
                batch = chunk_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT from_id, to_id, edge_type FROM chunk_refs WHERE from_id IN ({placeholders})",
                    batch,
                ).fetchall()
                results.extend((r[0], r[1], r[2]) for r in rows)
        return results

    def get_refs_to_chunks_typed(self, chunk_ids: list[str]) -> list[tuple[str, str, str]]:
        """Typed variant of get_refs_to_chunks: edges whose to_id is in
        chunk_ids, with edge_type."""
        if not chunk_ids:
            return []
        results: list[tuple[str, str, str]] = []
        with self._lock:
            for i in range(0, len(chunk_ids), 900):
                batch = chunk_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT from_id, to_id, edge_type FROM chunk_refs WHERE to_id IN ({placeholders})",
                    batch,
                ).fetchall()
                results.extend((r[0], r[1], r[2]) for r in rows)
        return results

    def get_refs_from_chunks_typed(self, chunk_ids: list[str]) -> list[tuple[str, str, str]]:
        """Forward twin of get_refs_to_chunks_typed: edges whose from_id is in
        chunk_ids, with edge_type. Backs find_outgoing."""
        if not chunk_ids:
            return []
        results: list[tuple[str, str, str]] = []
        with self._lock:
            for i in range(0, len(chunk_ids), 900):
                batch = chunk_ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT from_id, to_id, edge_type FROM chunk_refs WHERE from_id IN ({placeholders})",
                    batch,
                ).fetchall()
                results.extend((r[0], r[1], r[2]) for r in rows)
        return results

    def ref_indegrees(self, ids: list[str]) -> dict[str, int]:
        """{id: incoming-edge count} for ids with any edge; ids with zero
        are omitted, caller treats missing as 0. Used to keep mega-hubs
        (base classes, generic types) out of graph expansion."""
        if not ids:
            return {}
        out: dict[str, int] = {}
        with self._lock:
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT to_id, COUNT(*) AS n FROM chunk_refs "
                    f"WHERE to_id IN ({placeholders}) GROUP BY to_id",
                    batch,
                ).fetchall()
                for r in rows:
                    out[r["to_id"]] = r["n"]
        return out

    def _indegree_by_type(self, ids: list[str]) -> dict[str, dict[str, int]]:
        """Like ref_indegrees but broken down by edge_type: {id: {edge_type:
        count}} for each id with any incoming edge. Backed by idx_chunk_refs_to.
        Base aggregate for get_hubs."""
        if not ids:
            return {}
        out: dict[str, dict[str, int]] = {}
        with self._lock:
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT to_id, edge_type, COUNT(*) AS n FROM chunk_refs "
                    f"WHERE to_id IN ({placeholders}) GROUP BY to_id, edge_type",
                    batch,
                ).fetchall()
                for r in rows:
                    out.setdefault(r["to_id"], {})[r["edge_type"]] = r["n"]
        return out

    def get_hubs(self, path_prefix: str | None = None, limit: int = 20,
                 edge_types: list[str] | None = None) -> dict[str, Any]:
        """Named chunks ranked by in-degree, then PageRank, then path for
        determinism. Dispatches on chunk_indegree: precomputed when
        possible, else a live fallback gated by _get_hubs_live's guard."""
        # [] normalizes to None: the live path's `is not None` check would
        # otherwise filter every type out on an empty list.
        edge_types_set: set[str] | None = None
        if edge_types:
            edge_types_set = set(edge_types)
            invalid = edge_types_set - _HUB_EDGE_TYPES
            if invalid:
                raise ValueError(
                    f"invalid edge_types {sorted(invalid)!r} — valid types are "
                    f"{sorted(_HUB_EDGE_TYPES)}"
                )

        if self.has_indegree():
            return self._get_hubs_precomputed(path_prefix, limit, edge_types_set)
        return self._get_hubs_live(path_prefix, limit, edge_types_set)

    def _get_hubs_precomputed(
        self, path_prefix: str | None, limit: int, edge_types_set: set[str] | None,
    ) -> dict[str, Any]:
        """get_hubs via chunk_indegree: one aggregate query covers both
        scoped and unscoped cases, since this table stays small regardless
        of corpus size (unlike chunk_refs)."""
        clauses = ["c.name IS NOT NULL"]
        params: list[Any] = []
        if edge_types_set:
            placeholders = ",".join("?" * len(edge_types_set))
            clauses.append(f"ci.edge_type IN ({placeholders})")
            params.extend(sorted(edge_types_set))
        if path_prefix:
            # RANGE, not LIKE: an ESCAPE clause disables SQLite's LIKE-prefix
            # index optimization (same trick as _get_hubs_live).
            lo = path_prefix.rstrip("/\\")
            clauses.append("c.path >= ? AND c.path < ?")
            params.extend([lo, lo + "\U0010FFFF"])
        where = " AND ".join(clauses)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT c.id AS id, c.path AS path, c.name AS name,
                       c.chunk_type AS chunk_type, c.start_line AS start_line,
                       SUM(ci.n) AS in_degree, COALESCE(pr.score, 0.0) AS pagerank
                FROM chunk_indegree ci
                JOIN chunks c ON c.id = ci.chunk_id
                LEFT JOIN chunk_pagerank pr ON pr.chunk_id = ci.chunk_id
                WHERE {where}
                GROUP BY ci.chunk_id
                ORDER BY in_degree DESC, pagerank DESC, c.path ASC, c.start_line ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        if not rows:
            return {"hubs": []}

        ids = [r["id"] for r in rows]
        breakdown: dict[str, dict[str, int]] = {}
        with self._lock:
            for i in range(0, len(ids), 900):
                batch = ids[i:i + 900]
                ph = ",".join("?" * len(batch))
                type_clauses = [f"chunk_id IN ({ph})"]
                type_params: list[Any] = list(batch)
                if edge_types_set:
                    ph2 = ",".join("?" * len(edge_types_set))
                    type_clauses.append(f"edge_type IN ({ph2})")
                    type_params.extend(sorted(edge_types_set))
                for r in self._conn.execute(
                    f"SELECT chunk_id, edge_type, n FROM chunk_indegree WHERE "
                    f"{' AND '.join(type_clauses)}",
                    type_params,
                ):
                    breakdown.setdefault(r["chunk_id"], {})[r["edge_type"]] = r["n"]

        return {"hubs": [{
            "path": r["path"],
            "name": r["name"],
            "chunk_type": r["chunk_type"],
            "start_line": r["start_line"],
            "in_degree": r["in_degree"],
            "pagerank": r["pagerank"],
            "edge_types": breakdown.get(r["id"], {}),
            "by_provenance": _provenance_rollup(breakdown.get(r["id"], {})),
        } for r in rows]}

    def _get_hubs_live(
        self, path_prefix: str | None, limit: int, edge_types_set: set[str] | None,
    ) -> dict[str, Any]:
        """Live fallback for get_hubs when chunk_indegree is empty (old DB).
        edge_types_set is filtered in Python, not pushed into SQL."""
        if path_prefix:
            # RANGE, not LIKE: chunks is wide, so an unindexed LIKE scan
            # reads the whole corpus off disk. U+10FFFF bounds the range
            # to exactly the lo-prefixed paths.
            lo = path_prefix.rstrip("/\\")
            with self._lock:
                candidates = [dict(r) for r in self._conn.execute(
                    "SELECT id, path, name, chunk_type, start_line FROM chunks"
                    " WHERE name IS NOT NULL AND path >= ? AND path < ?",
                    (lo, lo + "\U0010FFFF"),
                ).fetchall()]
            if not candidates:
                return {"hubs": []}
            ids = [c["id"] for c in candidates]
            edge_counts = self._indegree_by_type(ids)
            pagerank = self.get_pagerank_for_chunks(ids)
            hubs: list[dict[str, Any]] = []
            for c in candidates:
                types = edge_counts.get(c["id"])
                if not types:
                    continue
                if edge_types_set is not None:
                    types = {t: n for t, n in types.items() if t in edge_types_set}
                    if not types:
                        continue
                hubs.append({
                    "path": c["path"],
                    "name": c["name"],
                    "chunk_type": c["chunk_type"],
                    "start_line": c["start_line"],
                    "in_degree": sum(types.values()),
                    "pagerank": pagerank.get(c["id"], 0.0),
                    "edge_types": types,
                    "by_provenance": _provenance_rollup(types),
                })
            hubs.sort(key=lambda h: (-h["in_degree"], -h["pagerank"], h["path"], h["start_line"]))
            return {"hubs": hubs[:limit]}

        # Whole-corpus branch. Guarded by _HUBS_GLOBAL_MAX_CHUNKS: the
        # GROUP BY over chunk_refs holds the store lock and can take minutes
        # on a huge corpus, blocking every other request.
        if self.count_chunks() > _HUBS_GLOBAL_MAX_CHUNKS:
            raise ValueError(
                "global /hubs on a corpus over "
                f"{_HUBS_GLOBAL_MAX_CHUNKS} chunks requires precomputed "
                "in-degree, and this DB doesn't have it yet (chunk_indegree "
                "is empty) — re-index (or run the graph rebuild) to populate "
                "it, or pass a path_prefix (or @subsystem) to scope the "
                "request in the meantime"
            )
        with self._lock:
            rows = self._conn.execute(
                "SELECT to_id, edge_type, COUNT(*) AS n FROM chunk_refs"
                " GROUP BY to_id, edge_type"
            ).fetchall()
        by_id: dict[str, dict[str, int]] = {}
        for r in rows:
            if edge_types_set is not None and r["edge_type"] not in edge_types_set:
                continue
            by_id.setdefault(r["to_id"], {})[r["edge_type"]] = r["n"]
        by_degree: dict[int, list[str]] = {}
        for cid, types in by_id.items():
            by_degree.setdefault(sum(types.values()), []).append(cid)

        hubs = []
        for degree in sorted(by_degree, reverse=True):
            ids = by_degree[degree]
            meta: dict[str, dict[str, Any]] = {}
            with self._lock:
                for i in range(0, len(ids), 900):
                    batch = ids[i:i + 900]
                    ph = ",".join("?" * len(batch))
                    for r in self._conn.execute(
                        "SELECT id, path, name, chunk_type, start_line FROM chunks"
                        f" WHERE name IS NOT NULL AND id IN ({ph})", batch,
                    ):
                        meta[r["id"]] = dict(r)
            pagerank = self.get_pagerank_for_chunks(list(meta))
            group = [{
                "path": meta[cid]["path"],
                "name": meta[cid]["name"],
                "chunk_type": meta[cid]["chunk_type"],
                "start_line": meta[cid]["start_line"],
                "in_degree": degree,
                "pagerank": pagerank.get(cid, 0.0),
                "edge_types": by_id[cid],
                "by_provenance": _provenance_rollup(by_id[cid]),
            } for cid in ids if cid in meta]
            group.sort(key=lambda h: (-h["pagerank"], h["path"], h["start_line"]))
            hubs.extend(group)
            if len(hubs) >= limit:
                break
        return {"hubs": hubs[:limit]}

    def rebuild_fts(self) -> None:
        """Rebuild FTS5 indexes from their content tables. Run after a
        --force re-index: repeated partial updates skew bm25 stats, which
        literals_fts relies on for find_by_message's relevance cap."""
        with self._lock:
            self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            self._conn.execute("INSERT INTO literals_fts(literals_fts) VALUES('rebuild')")
            self._conn.commit()

    def commit(self) -> None:
        """Flush pending writes; prefer this over store._conn directly so
        it serializes through the same lock."""
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
