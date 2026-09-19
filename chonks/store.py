"""sqlite-vec storage for chunks, embeddings, and the ref/kNN graph. Schema
is documented in DOCS.md. Embedding dim is fixed on first insert and
validated against `meta` on every later open."""

from typing import Any

from chonks.core.edges import _COLLAPSE_RANK, _HUB_EDGE_TYPES
from chonks.core.skeleton import *
from chonks.storage.schema import SCHEMA_VERSION
from chonks.storage.store import Store as _RowStore, _chunk_kind_clause


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
        from chonks.index.graph.hierarchy import rebuild_hierarchy
        return rebuild_hierarchy(self)

    def get_hubs(self, path_prefix: str | None = None, limit: int = 20,
                 edge_types: list[str] | None = None) -> dict[str, Any]:
        from chonks.retrieval.graph_queries import get_hubs
        return get_hubs(self, path_prefix, limit=limit, edge_types=edge_types)
