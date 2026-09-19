"""sqlite-vec storage for chunks, embeddings, and the ref/kNN graph. Schema
is documented in DOCS.md. Embedding dim is fixed on first insert and
validated against `meta` on every later open."""

import posixpath
import re
import sqlite3
from collections import defaultdict
from typing import Any

from chonks.core.batching import batched
from chonks.core.edges import _COLLAPSE_RANK, _HUB_EDGE_TYPES, edge_provenance
from chonks.core.skeleton import *
from chonks.languages import union as _lang_union
from chonks.storage.schema import SCHEMA_VERSION
from chonks.storage.store import Store as _RowStore, _chunk_kind_clause

# Unscoped get_hubs scans all of chunk_refs under the store lock, which can
# block every other request for minutes on a huge corpus. Above this size
# the global branch requires a path_prefix instead.
_HUBS_GLOBAL_MAX_CHUNKS = 100_000

# Header/impl pairing (rebuild_hierarchy) is same-dir only; cross-dir layouts
# (include/src) are not paired. Extension matching is case-insensitive.
HEADER_EXTS = _lang_union("header_exts")
IMPL_EXTS = _lang_union("impl_exts")


def _provenance_rollup(edge_types: dict[str, int]) -> dict[str, int]:
    """Roll up {edge_type: count} into {provenance: count}; shared by
    get_hubs' precomputed and live paths so both use the same rollup."""
    out: dict[str, int] = {}
    for et, n in edge_types.items():
        prov = edge_provenance(et)
        out[prov] = out.get(prov, 0) + n
    return out


# ---------------------------------------------------------------------------
# Literal & message index: skeleton generation + matching (find_by_message)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class Store(_RowStore):
    def find_usages(self, name: str, path_prefix: str | None = None,
                    limit: int | None = None) -> dict[str, Any]:
        from chonks.retrieval.graph_queries import find_usages
        return find_usages(self, name, path_prefix, limit=limit)

    def find_outgoing(self, name: str, path_prefix: str | None = None,
                      limit: int | None = None) -> dict[str, Any]:
        from chonks.retrieval.graph_queries import find_outgoing
        return find_outgoing(self, name, path_prefix, limit=limit)

    def get_impact(self, name: str, path_prefix: str | None = None,
                   limit: int = 20, rank_by: str = "pagerank_sum") -> dict[str, Any]:
        from chonks.retrieval.graph_queries import get_impact
        return get_impact(self, name, path_prefix, limit=limit, rank_by=rank_by)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Persisted in-degree
    # ------------------------------------------------------------------

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
            for batch in batched(ids, 900):
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
                for batch in batched(ids, 900):
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
