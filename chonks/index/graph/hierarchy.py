"""Derivation of the dir and file hierarchy and the header/impl pairs."""

import posixpath
from collections import defaultdict

from chonks.languages import union as _lang_union

# Header/impl pairing (rebuild_hierarchy) is same-dir only; cross-dir layouts
# (include/src) are not paired. Extension matching is case-insensitive.
HEADER_EXTS = _lang_union("header_exts")
IMPL_EXTS = _lang_union("impl_exts")


def rebuild_hierarchy(store) -> dict[str, int]:
    """Full rebuild of graph_nodes + graph_edges, DELETE then re-derive
    from scratch every call, never incremental. Chunks stay out of
    graph_nodes so chunk_pagerank stays chunk-only and comparable."""
    with store._lock:
        store.clear_hierarchy()

        file_paths = store.all_file_paths()

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
            store.insert_graph_nodes(dir_rows + file_rows)

        edge_rows = []
        for d in dir_paths:
            parent_id = _dir_parent_id(d)
            if parent_id is not None:
                edge_rows.append((parent_id, "dir:" + d, "contains"))
        for fp in file_paths:
            parent_id = "dir:" + (posixpath.dirname(fp) or ".")
            edge_rows.append((parent_id, "file:" + fp, "contains"))

        if edge_rows:
            store.insert_graph_edges(edge_rows)

        store.insert_chunk_contains_edges()

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
            store.insert_graph_edges(paired_edge_rows, ignore_duplicates=True)

        node_count = store.count_graph_nodes()
        edge_count = store.count_graph_edges()

        store.commit()

    return {"nodes": node_count, "edges": edge_count}
