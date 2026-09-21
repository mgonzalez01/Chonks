"""The edge-type universe: weights, provenance, and hub/collapse ranks."""

from __future__ import annotations

# Skips short noise tokens like "i" or "ok" when matching references.
_MIN_NAME_LEN = 3

# Caps cross-language edges per name: ubiquitous names (Initialize, Update)
# would otherwise create thousands of low-signal edges.
_MAX_CROSS_LANG_OCCURRENCES = 8

# PageRank has no incremental algorithm, so a small batch (see
# pagerank._PAGERANK_REFRESH_MIN_FRACTION) skips the live recompute and
# reuses stale scores; churn accumulates in this meta key until it crosses
# that fraction, forcing a refresh.
_PAGERANK_STALE_META_KEY = "pagerank_stale_chunks"

# All-1.0 defaults reproduce pre-weighting PageRank exactly; the
# associated/mentions split stays a ranking-inert display label until a
# caller sets these two weights apart.
DEFAULT_EDGE_TYPE_WEIGHTS: dict[str, float] = {
    "calls":      1.0,
    "imports":    1.0,
    "inherits":   1.0,
    "xlang":      1.0,
    "associated": 1.0,
    "mentions":   1.0,
}

# Provenance is derived from edge_type at query time.
# 'extracted' labels the fact's mechanism: an
# ambiguous name's fan-out edges are all still labelled extracted.
PROVENANCE_BY_EDGE_TYPE = {
    "calls": "extracted", "imports": "extracted", "inherits": "extracted",
    "contains": "extracted",
    "mentions": "inferred", "associated": "inferred", "semantic": "inferred",
    "xlang": "paired",
}


def edge_provenance(edge_type: str) -> str:
    """Map edge_type to provenance (extracted/inferred/paired); unknown
    types default to inferred so an old or new DB never crashes here."""
    return PROVENANCE_BY_EDGE_TYPE.get(edge_type, "inferred")

# When a chunk has edges of more than one type into the resolution set, the
# lowest rank wins (deterministic; SQL row order is not). Shared by
# find_usages and find_outgoing.
_COLLAPSE_RANK = {
    "calls": 0, "imports": 0, "inherits": 0,
    "xlang": 1,
    "associated": 2,
    "mentions": 3,
}

# Valid chunk_refs.edge_type values, for get_hubs' edge_types filter.
_HUB_EDGE_TYPES = frozenset(DEFAULT_EDGE_TYPE_WEIGHTS)
