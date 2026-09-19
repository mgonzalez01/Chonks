"""sqlite-vec storage for chunks, embeddings, and the ref/kNN graph. Schema
is documented in DOCS.md. Embedding dim is fixed on first insert and
validated against `meta` on every later open."""

import posixpath
from collections import defaultdict
from typing import Any

from chonks.core.edges import _COLLAPSE_RANK, _HUB_EDGE_TYPES
from chonks.core.skeleton import *
from chonks.languages import union as _lang_union
from chonks.storage.schema import SCHEMA_VERSION
from chonks.storage.store import Store as _RowStore, _chunk_kind_clause

# Header/impl pairing (rebuild_hierarchy) is same-dir only; cross-dir layouts
# (include/src) are not paired. Extension matching is case-insensitive.
HEADER_EXTS = _lang_union("header_exts")
IMPL_EXTS = _lang_union("impl_exts")


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

    def find_by_message(self, message: str, limit: int = 20) -> dict[str, Any]:
        from chonks.retrieval.message_match import find_by_message
        return find_by_message(self, message, limit=limit)

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

    def get_hubs(self, path_prefix: str | None = None, limit: int = 20,
                 edge_types: list[str] | None = None) -> dict[str, Any]:
        from chonks.retrieval.graph_queries import get_hubs
        return get_hubs(self, path_prefix, limit=limit, edge_types=edge_types)
