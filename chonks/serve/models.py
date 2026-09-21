"""Request models of the HTTP server and their limits."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Server-side cap: blocks OOM/huge-JSON from an absurd client top_k.
TOP_K_MAX = 200

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

_QUERY_MAX_LEN  = 2000
_PREFIX_MAX_LEN = 500
_INDEX_PATHS_MAX = 100
_PROJECT_MAX_LEN = 100


class SearchRequest(BaseModel):
    query:        str            = Field(max_length=_QUERY_MAX_LEN)
    mode:         str            = "semantic"   # "semantic" | "fts" | "regex" | "hybrid"
    top_k:        int            = Field(default=50, ge=1, le=TOP_K_MAX)
    path_prefix:  str | None     = Field(default=None, max_length=_PREFIX_MAX_LEN)
    # Applies to the blended score, not raw cosine, when folder_blend is set.
    min_score:    float | None   = Field(default=None, ge=-1.0, le=1.0)
    folder_blend: bool           = False
    # None/"any" applies no filter; see store._chunk_kind_clause.
    chunk_kind:   str | None     = Field(default=None, max_length=10)
    # Backfills freed slots from other files rather than just truncating (stops one hot file's near-dup wall).
    file_cap:     int | None     = Field(default=None, ge=0, le=TOP_K_MAX)
    project:      str | None     = Field(default=None, max_length=_PROJECT_MAX_LEN)

    @field_validator("chunk_kind")
    @classmethod
    def _chunk_kind_valid(cls, v: str | None) -> str | None:
        if v is not None and v not in ("code", "docs", "any"):
            raise ValueError(f"chunk_kind must be one of code|docs|any, got {v!r}")
        return v


class ResearchRequest(BaseModel):
    query:       str            = Field(max_length=_QUERY_MAX_LEN)
    path_prefix: str | None     = Field(default=None, max_length=_PREFIX_MAX_LEN)
    project:     str | None     = Field(default=None, max_length=_PROJECT_MAX_LEN)
    # Per-request override; precedence is request > project config > all-1.0 default.
    edge_type_weights: dict[str, float] | None = None

    @field_validator("edge_type_weights")
    @classmethod
    def _edge_type_weights_non_negative(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is not None:
            for k, weight in v.items():
                if weight < 0:
                    raise ValueError(f"edge_type_weights[{k!r}] must be non-negative, got {weight}")
        return v


class RepomapRequest(BaseModel):
    path_prefix:  str | None    = Field(default=None, max_length=_PREFIX_MAX_LEN)
    query:        str | None    = Field(default=None, max_length=_QUERY_MAX_LEN)
    token_budget: int | None    = None  # None = repomap_cfg's token_budget (default 8000)
    project:      str | None    = Field(default=None, max_length=_PROJECT_MAX_LEN)


class IndexRequest(BaseModel):
    paths:   list[str]          = Field(max_length=_INDEX_PATHS_MAX)
    force:   bool               = False
    project: str | None         = Field(default=None, max_length=_PROJECT_MAX_LEN)


class SymbolRequest(BaseModel):
    name:        str            = Field(max_length=_QUERY_MAX_LEN)
    path_prefix: str | None     = Field(default=None, max_length=_PREFIX_MAX_LEN)
    prefix:      bool           = False  # prefix match instead of exact
    project:     str | None     = Field(default=None, max_length=_PROJECT_MAX_LEN)


class UsagesRequest(BaseModel):
    name:        str            = Field(max_length=_QUERY_MAX_LEN)
    path_prefix: str | None     = Field(default=None, max_length=_PREFIX_MAX_LEN)
    limit:       int | None     = Field(default=None, gt=0, le=1000)
    project:     str | None     = Field(default=None, max_length=_PROJECT_MAX_LEN)


class ImpactRequest(BaseModel):
    name:        str            = Field(max_length=_QUERY_MAX_LEN)
    path_prefix: str | None     = Field(default=None, max_length=_PREFIX_MAX_LEN)
    # None is tolerated: JSON clients send null for an omitted param.
    limit:       int | None     = Field(default=None, ge=1, le=100)
    # Invalid rank_by is validated in the endpoint and mapped to a 400, not here.
    rank_by:     str | None     = Field(default=None, max_length=32)
    project:     str | None     = Field(default=None, max_length=_PROJECT_MAX_LEN)


class OutgoingRequest(BaseModel):
    name:        str            = Field(max_length=_QUERY_MAX_LEN)
    path_prefix: str | None     = Field(default=None, max_length=_PREFIX_MAX_LEN)
    limit:       int | None     = Field(default=None, gt=0, le=1000)
    project:     str | None     = Field(default=None, max_length=_PROJECT_MAX_LEN)


class InvestigateRequest(BaseModel):
    name:          str            = Field(max_length=_QUERY_MAX_LEN)
    path_prefix:   str | None     = Field(default=None, max_length=_PREFIX_MAX_LEN)
    usages_limit:  int | None     = Field(default=30, gt=0, le=1000)
    outgoing_limit: int | None    = Field(default=30, gt=0, le=1000)
    impact_limit:  int | None     = Field(default=10, ge=1, le=100)
    # Default True: without inlined source, agents under-read and skipped the standalone tools.
    definition_source:            bool = True
    definition_source_max_chars:  int  = Field(default=4000, ge=200, le=20000)
    project:       str | None     = Field(default=None, max_length=_PROJECT_MAX_LEN)


class HubsRequest(BaseModel):
    path_prefix: str | None     = Field(default=None, max_length=_PREFIX_MAX_LEN)
    limit:       int | None     = Field(default=None, ge=1, le=100)
    # Invalid values are validated in the endpoint and mapped to a 400.
    edge_types:  list[str] | None = Field(default=None)
    project:     str | None     = Field(default=None, max_length=_PROJECT_MAX_LEN)


class FindByMessageRequest(BaseModel):
    message: str        = Field(max_length=_QUERY_MAX_LEN)
    limit:   int | None = Field(default=None, gt=0, le=100)
    project: str | None = Field(default=None, max_length=_PROJECT_MAX_LEN)


class TraceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_symbol:      str         = Field(max_length=_QUERY_MAX_LEN, alias="from")
    to_symbol:        str         = Field(max_length=_QUERY_MAX_LEN, alias="to")
    max_depth:        int         = Field(default=6, ge=1, le=20)
    include_semantic: bool        = False
    project:          str | None  = Field(default=None, max_length=_PROJECT_MAX_LEN)


