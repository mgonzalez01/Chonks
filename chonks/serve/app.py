"""The FastAPI app of the HTTP server and its routes."""

import importlib.metadata
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from chonks.chunker import index_paths
from chonks.index.segment import CODE_LANGUAGES
from chonks.retrieval.repomap import build_repomap
from chonks.retrieval.trace import trace_path
from chonks.research import deep_research
from chonks.retrieval import graph_queries, message_match
from chonks.retrieval.source import _attach_definition_source
from chonks.searcher import DEFAULT_BLEND_ALPHA, DEFAULT_BLEND_BETA, DEFAULT_FILE_CAP, rank_files
from chonks.serve.models import (
    FindByMessageRequest,
    HubsRequest,
    ImpactRequest,
    IndexRequest,
    InvestigateRequest,
    OutgoingRequest,
    RepomapRequest,
    ResearchRequest,
    SearchRequest,
    SymbolRequest,
    TraceRequest,
    UsagesRequest,
)
from chonks.serve.projects import (
    DEFAULT_PROJECT,
    _get_project,
    _projects,
    lifespan,
)

logger = logging.getLogger("chonks.server")

app = FastAPI(title="Chonks", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/search")
def search(req: SearchRequest) -> JSONResponse:
    project = _get_project(req.project)
    searcher = project["searcher"]

    # Reject these outside semantic mode.
    if req.mode != "semantic" and (
        req.min_score is not None or req.folder_blend or req.file_cap is not None
    ):
        raise HTTPException(
            422,
            "min_score, folder_blend, and file_cap are only valid with mode='semantic'.",
        )

    # A request-level file_cap, if given, overrides the project config.
    blend: tuple[float, float] | None = None
    file_cap = DEFAULT_FILE_CAP
    if req.mode == "semantic":
        scfg = project["search_cfg"]
        if req.folder_blend:
            alpha = float(scfg.get("blend_alpha", DEFAULT_BLEND_ALPHA))
            beta  = float(scfg.get("blend_beta",  DEFAULT_BLEND_BETA))
            blend = (alpha, beta)
        file_cap = req.file_cap if req.file_cap is not None else int(scfg.get("file_cap", DEFAULT_FILE_CAP))

    # fts/regex never embed, so query_truncated stays False for them.
    query_truncated = False

    if req.mode == "semantic":
        query_truncated = searcher.query_truncated(req.query)
        if query_truncated:
            logger.warning(
                "Query truncated to the embed token budget, "
                "len=%d chars; semantic ranking sees only the truncated text.",
                len(req.query),
            )
        try:
            chunks = searcher.semantic(
                req.query, req.top_k, req.path_prefix,
                min_score=req.min_score, folder_blend=blend, chunk_kind=req.chunk_kind,
                file_cap=file_cap,
            )
        # Narrow catch: a broader except would also swallow a real store bug as "embedder down".
        except httpx.HTTPError as e:
            raise HTTPException(
                503,
                f"Embedder unreachable, semantic search unavailable: {e}. "
                "Retry with mode='fts' (keyword search needs no embedding).",
            )
    elif req.mode == "fts":
        chunks = searcher.fts(req.query, req.top_k, req.path_prefix, chunk_kind=req.chunk_kind)
    elif req.mode == "regex":
        chunks = searcher.regex(req.query, req.top_k, req.path_prefix, chunk_kind=req.chunk_kind)
    elif req.mode == "hybrid":
        query_truncated = searcher.query_truncated(req.query)
        if query_truncated:
            logger.warning(
                "Query truncated to the embed token budget, "
                "len=%d chars; hybrid's semantic branch sees only the "
                "truncated text (FTS branch is unaffected).",
                len(req.query),
            )
        try:
            chunks = searcher.hybrid(req.query, req.top_k, req.path_prefix, chunk_kind=req.chunk_kind)
        except httpx.HTTPError as e:
            raise HTTPException(
                503,
                f"Embedder unreachable, hybrid search unavailable: {e}. "
                "Retry with mode='fts' (keyword search needs no embedding).",
            )
    else:
        raise HTTPException(400, f"Unknown mode: {req.mode!r}. Use semantic|fts|regex|hybrid")

    # Counts only the returned (post top_k) chunks, for the MCP layer's docs-drown-code nudge.
    docs_in_results = sum(1 for c in chunks if c.get("language") not in CODE_LANGUAGES)

    # None means too few vector chunks to evaluate; see Searcher.detect_near_dup_wall.
    near_dup = searcher.detect_near_dup_wall(chunks)

    return JSONResponse({
        "chunks":           chunks,
        "formatted":        searcher.format_results(chunks),
        "count":            len(chunks),
        "docs_in_results":  docs_in_results,
        "near_dup":         near_dup,
        "files":            rank_files(chunks),
        "query_truncated":  query_truncated,
    })


@app.post("/research")
def research(req: ResearchRequest) -> JSONResponse:
    """deep_research never raises on an embedder outage; it sets `degraded`
    instead, since seeds still rank by stored similarity."""
    project = _get_project(req.project)
    cfg = dict(project["research_cfg"] or {})
    # Top-level config key, not nested under `research`, so it must be set here explicitly.
    cfg.setdefault("edge_type_weights", project["edge_type_weights"])
    if req.edge_type_weights is not None:
        cfg["edge_type_weights"] = req.edge_type_weights
    result = deep_research(
        req.query,
        project["searcher"],
        cfg=cfg or None,
        path_prefix=req.path_prefix,
    )
    result["files"] = rank_files(result["chunks"])
    return JSONResponse(result)


@app.post("/repomap")
def repomap(req: RepomapRequest) -> JSONResponse:
    project = _get_project(req.project)

    scope = req.path_prefix or "(whole index)"
    logger.info(
        "codebase_map  project=%r  scope=%r  query=%r  budget=%s",
        req.project or DEFAULT_PROJECT, scope, req.query, req.token_budget,
    )

    # Applies even to scoped calls: one subtree can still be too large for the caller's context.
    budget = req.token_budget
    if budget is None:
        budget = project["repomap_cfg"].get("token_budget", 8000)

    map_text = build_repomap(
        project["store"],
        path_prefix=req.path_prefix,
        query=req.query,
        token_budget=budget,
    )
    return JSONResponse({"map": map_text})


@app.post("/symbol")
def symbol(req: SymbolRequest) -> JSONResponse:
    """Exact-name (or prefix) lookup against the decoupled symbol index; see
    DOCS.md for the field-level contract. A miss gets a `note` routing the
    caller to codebase_search mode=fts."""
    project = _get_project(req.project)
    rows = project["store"].find_symbols(req.name, req.path_prefix, prefix=req.prefix)
    logger.info("symbol  project=%r  name=%r  prefix=%s  hits=%d",
                req.project or DEFAULT_PROJECT, req.name, req.prefix, len(rows))
    note = None
    if not rows:
        note = (
            f"no named boundary matches {req.name!r} — the symbol index holds "
            "functions/classes/named AST boundaries; fields and locals live in "
            "chunk content, try codebase_search mode=fts"
        )
    return JSONResponse({"symbols": rows, "count": len(rows), "note": note})


@app.post("/usages")
def usages(req: UsagesRequest) -> JSONResponse:
    """Who references `name`. The above-cap case returns `content_matches`
    (FTS scan hits) as a separate list, not merged into `usages`."""
    project = _get_project(req.project)
    result = graph_queries.find_usages(project["store"], req.name, req.path_prefix, limit=req.limit)
    rows = result["results"]
    logger.info("usages  project=%r  name=%r  hits=%d",
                req.project or DEFAULT_PROJECT, req.name, len(rows))
    return JSONResponse({
        "usages": rows, "count": len(rows),
        "note": result["note"], "content_matches": result["content_matches"],
    })


@app.post("/impact")
def impact(req: ImpactRequest) -> JSONResponse:
    """Blast radius of `name`, aggregated by referencing file; see DOCS.md
    for the field-level contract and get_impact in chonks/retrieval/graph_queries.py
    for the ranking."""
    project = _get_project(req.project)
    try:
        result = graph_queries.get_impact(project["store"],
            req.name, req.path_prefix, limit=req.limit or 20,
            rank_by=req.rank_by or "pagerank_sum",
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    logger.info(
        "impact  project=%r  name=%r  total_references=%d  files=%d  files_total=%d",
        req.project or DEFAULT_PROJECT, req.name,
        result["total_references"], len(result["files"]), result["files_total"],
    )
    return JSONResponse(result)


@app.post("/outgoing")
def outgoing(req: OutgoingRequest) -> JSONResponse:
    """Forward counterpart to /usages. Zero outgoing edges is a valid empty
    result (note: None), distinct from a symbol-not-found miss."""
    project = _get_project(req.project)
    result = graph_queries.find_outgoing(project["store"], req.name, req.path_prefix, limit=req.limit)
    rows = result["results"]
    logger.info("outgoing  project=%r  name=%r  hits=%d",
                req.project or DEFAULT_PROJECT, req.name, len(rows))
    return JSONResponse({
        "outgoing": rows, "count": len(rows), "note": result["note"],
    })


@app.post("/investigate")
def investigate(req: InvestigateRequest) -> JSONResponse:
    """Composite endpoint fanning out to /symbol, /usages, /outgoing, /impact.
    path_prefix scopes only the three fan-out legs; the definition lookup stays
    unscoped so a scoped call can't false-negative on a symbol defined elsewhere."""
    project = _get_project(req.project)
    store = project["store"]
    definitions = store.find_symbols(req.name)
    if not definitions:
        note = (
            f"no named boundary matches {req.name!r} — the symbol index holds "
            "functions/classes/named AST boundaries; fields and locals live in "
            "chunk content, try codebase_search mode=fts"
        )
        logger.info("investigate  project=%r  name=%r  definitions=0 (short-circuit)",
                    req.project or DEFAULT_PROJECT, req.name)
        return JSONResponse({
            "symbol": req.name,
            "definitions": [],
            "usages": {"usages": [], "count": 0},
            "outgoing": {"outgoing": [], "count": 0},
            "impact": {},
            "notes": {
                "definitions": note, "usages": note, "outgoing": note, "impact": note,
            },
        })

    if req.definition_source:
        _attach_definition_source(store, definitions, req.definition_source_max_chars)

    usages_result = graph_queries.find_usages(store, req.name, req.path_prefix, limit=req.usages_limit or 30)
    outgoing_result = graph_queries.find_outgoing(store, req.name, req.path_prefix, limit=req.outgoing_limit or 30)
    impact_result = graph_queries.get_impact(store, req.name, req.path_prefix, limit=req.impact_limit or 10)

    logger.info(
        "investigate  project=%r  name=%r  definitions=%d  usages=%d  outgoing=%d  impact_files=%d",
        req.project or DEFAULT_PROJECT, req.name, len(definitions),
        len(usages_result["results"]), len(outgoing_result["results"]),
        len(impact_result["files"]),
    )
    return JSONResponse({
        "symbol": req.name,
        "definitions": definitions,
        "usages": {
            "usages": usages_result["results"], "count": len(usages_result["results"]),
            "content_matches": usages_result["content_matches"],
        },
        "outgoing": {
            "outgoing": outgoing_result["results"], "count": len(outgoing_result["results"]),
        },
        "impact": impact_result,
        "notes": {
            "definitions": None,
            "usages": usages_result["note"],
            "outgoing": outgoing_result["note"],
            "impact": impact_result["note"],
        },
    })


@app.post("/hubs")
def hubs(req: HubsRequest) -> JSONResponse:
    """path_prefix scopes the hub chunks themselves, not their referrers."""
    project = _get_project(req.project)
    try:
        result = graph_queries.get_hubs(project["store"],
            req.path_prefix, limit=req.limit or 20, edge_types=req.edge_types,
        )
    except ValueError as e:
        # 400 covers both the global-scope guard and an invalid edge_types value.
        raise HTTPException(400, str(e))
    logger.info("hubs  project=%r  path_prefix=%r  hits=%d",
                req.project or DEFAULT_PROJECT, req.path_prefix, len(result["hubs"]))
    return JSONResponse(result)


@app.post("/trace")
def trace(req: TraceRequest) -> JSONResponse:
    """How does `from` reach `to`; see DOCS.md for the field-level contract.
    A name with multiple definitions has every one tried."""
    project = _get_project(req.project)
    result = trace_path(
        project["store"], req.from_symbol, req.to_symbol,
        max_depth=req.max_depth, include_semantic=req.include_semantic,
    )
    logger.info(
        "trace  project=%r  from=%r  to=%r  found=%s  depth=%s",
        req.project or DEFAULT_PROJECT, req.from_symbol, req.to_symbol,
        result["found"], result.get("depth"),
    )
    if not result["found"] and result.get("reason") == "unknown_symbol":
        raise HTTPException(404, result["error"])
    return JSONResponse(result)


@app.post("/find_by_message")
def find_by_message(req: FindByMessageRequest) -> JSONResponse:
    """Given a runtime message/log line/error string a human actually saw,
    find the source literal that emitted it, including through format holes;
    see DOCS.md for the field-level contract."""
    project = _get_project(req.project)
    result = message_match.find_by_message(project["store"], req.message, limit=req.limit or 20)
    rows = result["results"]
    logger.info("find_by_message  project=%r  hits=%d  truncated=%s",
                req.project or DEFAULT_PROJECT, len(rows), result["truncated"])
    return JSONResponse({
        "results": rows, "count": len(rows),
        "truncated": result["truncated"], "note": result.get("note"),
    })


@app.post("/index")
def index(req: IndexRequest) -> JSONResponse:
    project = _get_project(req.project)
    root    = project["root"]
    exclude = project["exclude"]
    include = project["include"]

    lock = project["index_lock"]
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "An indexing operation is already in progress for this project.")

    try:
        # Blocks re-indexing arbitrary filesystem locations via HTTP.
        if root is not None:
            for p in req.paths:
                try:
                    Path(p).resolve().relative_to(root)
                except ValueError:
                    raise HTTPException(
                        400,
                        f"Path {p!r} is outside the configured codebase root "
                        f"{str(root)!r}. Update 'codebase' in config.json to allow it.",
                    )

        stats = index_paths(
            req.paths, project["store"], project["embedder"],
            root=root,
            force=req.force,
            exclude=exclude,
            include=include,
            embed_batch=project["embed_batch"],
            embed_inflight=project["embed_inflight"],
            edge_type_weights=project["edge_type_weights"] or None,
        )
    finally:
        lock.release()

    return JSONResponse(stats)


@app.get("/status")
def status() -> JSONResponse:
    """Returns the legacy flat shape when only the default project exists,
    for back-compat with existing clients; otherwise {"projects": {...}}."""
    project_stats: dict[str, dict] = {}
    for name, p in _projects.items():
        if p.get("store") is None:
            project_stats[name] = {"ready": False, "lazy": True}
        else:
            project_stats[name] = {"ready": True, **p["store"].stats()}

    if list(_projects.keys()) == [DEFAULT_PROJECT]:
        return JSONResponse(project_stats[DEFAULT_PROJECT])
    return JSONResponse({"projects": project_stats})


# Hand-curated, not derived from the route table: mechanical introspection is what /docs is for.
_API_INDEX_ENDPOINTS: list[dict[str, str]] = [
    {"method": "GET",  "path": "/",              "description": "This index."},
    {"method": "GET",  "path": "/status",        "description": "Health check and DB statistics."},
    {"method": "POST", "path": "/search",        "description": "Search chunks: semantic|fts|regex|hybrid."},
    {"method": "POST", "path": "/research",      "description": "Deep-research retrieval with typed-edge evidence."},
    {"method": "POST", "path": "/repomap",       "description": "Structural symbol map of a path prefix or the whole index."},
    {"method": "POST", "path": "/symbol",        "description": "Exact or prefix lookup against the named-boundary symbol index."},
    {"method": "POST", "path": "/usages",        "description": "Who references a symbol (incoming chunk_refs)."},
    {"method": "POST", "path": "/outgoing",      "description": "What a symbol references (outgoing chunk_refs)."},
    {"method": "POST", "path": "/investigate",   "description": "Composite definitions + usages + outgoing + impact for a symbol."},
    {"method": "POST", "path": "/impact",        "description": "Blast radius of a symbol, aggregated by referencing file."},
    {"method": "POST", "path": "/hubs",          "description": "Structurally load-bearing chunks, ranked by in-degree."},
    {"method": "POST", "path": "/trace",         "description": "Shortest structural path between two symbols."},
    {"method": "POST", "path": "/find_by_message", "description": "Find the source literal that emitted a runtime message."},
    {"method": "POST", "path": "/index",         "description": "Parse + embed paths into the index."},
]


@app.get("/")
def root() -> JSONResponse:
    """Must not touch any project's Store: this is the first thing a fresh client probes."""
    try:
        version = importlib.metadata.version("chonks")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return JSONResponse({
        "name":        "chonks",
        "version":     version,
        "description": "Local code RAG (semantic search + deep research) over source codebases.",
        "endpoints":   _API_INDEX_ENDPOINTS,
        "status_url":  "/status",
    })
