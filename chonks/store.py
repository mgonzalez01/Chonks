"""sqlite-vec storage for chunks, embeddings, and the ref/kNN graph. Schema
is documented in DOCS.md. Embedding dim is fixed on first insert and
validated against `meta` on every later open."""

from chonks.core.edges import _COLLAPSE_RANK, _HUB_EDGE_TYPES
from chonks.core.skeleton import *
from chonks.storage.schema import SCHEMA_VERSION
from chonks.storage.store import Store, _chunk_kind_clause
