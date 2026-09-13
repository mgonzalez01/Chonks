"""repomap: structural map of the indexed codebase, ranked by PageRank over
chunk_refs. No tree-sitter re-parsing here; symbols and edges come from
tables already populated at index time.

Split by subsystem:
- ._shared:   constants and helpers shared across the other modules.
- .knn:       k-NN neighbor graph (backend detection, _Corpus, build_neighbors).
- .refs:      chunk_refs graph construction (mentions/xlang/typed edges, build_refs).
- .pagerank:  PageRank compute/persist/read.
- .render:    repo-map text rendering (build_repomap).
- .trace:     trace_path bidirectional BFS.

This module re-exports every name any production or test file imports
directly from `chonks.repomap`, including private (underscore) names that
tests reach into, so `from chonks.repomap import X` keeps working exactly
as it did when this was a single flat module.
"""

from __future__ import annotations

from ._shared import (
    DEFAULT_EDGE_TYPE_WEIGHTS,
    PROVENANCE_BY_EDGE_TYPE,
    _MAX_CROSS_LANG_OCCURRENCES,
    _MIN_NAME_LEN,
    _PAGERANK_STALE_META_KEY,
    edge_provenance,
)
from .knn import (
    GRAPHRAG_TOP_K,
    _Corpus,
    _available_ram_bytes,
    _build_neighbors_incremental,
    _detect_knn_backend,
    _device_backend_armed,
    _id_rank,
    _neighbor_block_rows,
    _tie_break_keys,
    _topk_for_rows,
    build_neighbors,
)
from .refs import (
    _arity_compatible,
    _bare_alias_definers,
    _build_graph,
    _build_refs_incremental,
    _call_entry_fields,
    _classify_mentions,
    _definer_qualifiers,
    _discriminate_definers,
    _name_keys,
    _resolve_names,
    _xlang_pairs_for_names,
    build_refs,
)
from .pagerank import (
    _compute_pagerank_live,
    compute_pagerank_global,
    persist_pagerank,
)
from .render import (
    _NODE_TYPE_PREFIX,
    _build_repomap_from_chunks,
    _dir_overview,
    _format_map,
    _summarize_omitted_subtrees,
    build_repomap,
)
from .trace import (
    _adjacency,
    _bidirectional_bfs,
    _edge_rank,
    _hop_direction,
    trace_path,
)

__all__ = [
    "DEFAULT_EDGE_TYPE_WEIGHTS",
    "PROVENANCE_BY_EDGE_TYPE",
    "edge_provenance",
    "build_neighbors",
    "build_refs",
    "persist_pagerank",
    "compute_pagerank_global",
    "build_repomap",
    "trace_path",
    "GRAPHRAG_TOP_K",
]
