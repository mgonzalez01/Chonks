// The MCP tool contract: server instructions and the tool list with schemas.

import { SUBSYSTEMS } from "./subsystems.js";

// Prohibition-with-rationale phrasing measured 100% tool adoption vs 17%
// for recommendation-only (eval/results/F1/adoption_godot.json).
export const INSTRUCTIONS = `Chonks is a local code-RAG index over this codebase, exposed as nine tools. Use it instead of filesystem search, not after it.

Never run Grep, Glob, find, or any filesystem enumeration against the repository root. This codebase is indexed precisely because it is too large to walk: root-scoped text searches are slow, noisy, and can time out. Start every code search with the Chonks tools. If they are deferred (present by name only), load them before your first search — one call: ToolSearch with query \"select:mcp__chonks__codebase_search,mcp__chonks__codebase_research,mcp__chonks__codebase_map,mcp__chonks__find_symbol,mcp__chonks__find_usages,mcp__chonks__investigate,mcp__chonks__trace_path,mcp__chonks__find_by_message,mcp__chonks__codebase_status\". Do not fall back to Grep because they need loading. Once a tool returns a concrete file path, refine with scoped Grep or Read on that path — retrieval points at code; the file is the source of truth.

Routing: codebase_map, codebase_search, and codebase_research are for orientation — finding the right region from a question. Once you hold a concrete symbol name, switch to the precise tools: find_symbol (definition sites), find_usages (callers and references), trace_path (how A reaches B), investigate (definition + callers + outgoing calls for that symbol, in one call with source inlined). When the input is a runtime message/log line/error string instead of a symbol name, use find_by_message — it matches through format holes (%s, {name}), which text search structurally cannot do since the concrete value never appears in source. Prefer these over text search: they resolve names folded into merged chunks, skip comment/string false positives, and (find_by_message) match cases grep cannot match at all.

Three bindings, because skipping them measurably produces wrong or wasteful routes:
- Orientation questions (where does X live, where do I start, how is Y organized): your FIRST call is codebase_map with NO path_prefix. It opens with a directory overview of the whole tree; take path_prefix values for follow-up calls from that overview. Never guess directory names from prior knowledge — guessed prefixes routinely name directories that do not exist in this repository.
- Dependency questions (what does X depend on, what uses X, what breaks if X changes): find_symbol alone is an incomplete answer. After locating the definition, call find_usages (who uses X), investigate (what X calls — the only tool with the outgoing direction), or trace_path — reference lists are indexed and cost one call; reading the definition file tells you what X calls, never who calls X. Weigh the returned provenance labels: extracted edges are AST facts, inferred edges are name co-occurrence.
- "Where does this message/log line/error come from" questions: do not grep for the concrete string — if it came from a format string (%s, f-string, {name}), grep on the concrete value will not find the source line that emitted it. Call find_by_message with the message instead; it is the only tool that matches through format holes.

Caveats: every response carries an index-staleness header — files changed since the build need a re-index. Docs and config are indexed as nameless text chunks: they surface in search and research (often the best answer to why-questions) but are invisible to the symbol and graph tools. trace_path hops over mentions/associated edges are name-occurrence evidence, not proven calls — trust calls/imports/inherits chains more. These tools only address the default project: if config.json defines multiple projects, query the others via the HTTP API directly, and note a project with a configured codebase root rejects (400) any request path outside it.`;

export const TOOLS = [
  {
    name: "codebase_search",
    description:
      "Answers 'where is X' / 'which file handles Y' in one sub-second call against the pre-built " +
      "index — the first move for that kind of question, replacing a multi-turn Grep hunt when " +
      "you don't yet know the file. Returns ranked chunks (not prose). " +
      "Modes: semantic (concept-level, default), fts (exact keywords / boolean), " +
      "regex (pattern matching), hybrid (RRF fusion of semantic + fts — best when " +
      "the query mixes a concept and a specific symbol name, or you don't know which to pick). " +
      "Prefix the query with `@subsystem ` (e.g. `@rendering ShadowAtlas`) to scope " +
      "the search to a configured subsystem's paths. Available subsystems: " +
      (Object.keys(SUBSYSTEMS).length ? Object.keys(SUBSYSTEMS).map((n) => "@" + n).join(", ") : "(none configured)") +
      ". A code-oriented question (find an implementation, a call site, a naming pattern) " +
      "should pass chunk_kind:\"code\"; a why/architecture question should leave it \"any\" " +
      "— docs chunks are often the best answer to why-questions. mode:\"fts\" is the tool for " +
      "anything find_symbol can't see: member fields, local variables, string literals — " +
      "find_symbol only indexes named AST boundaries (functions/classes), so a field or " +
      "literal search belongs here. Results open with a `files:` line ranking the distinct " +
      "files hit, each role-tagged by path heuristic ([impl]/[test]/[docs]/[gen]) with a " +
      "×N chunk count when a file contributed more than one result; individual chunk " +
      "headers carry the same tag when not [impl].",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query. May start with `@subsystem ` to scope." },
        mode: { type: "string", enum: ["semantic", "fts", "regex", "hybrid"], description: "Search mode." },
        top_k: {
          type: "number",
          description:
            "Max results (1-100). Default 10 — right for targeted lookups " +
            "(a specific symbol/implementation); keep 20+ only for broad sweeps " +
            "where you expect to scan many candidates. Results re-enter your " +
            "context every turn, so oversized result sets compound in cost.",
        },
        path_prefix: { type: "string", description: "Alternative to @subsystem: explicit path prefix filter." },
        chunk_kind: {
          type: "string",
          enum: ["code", "docs", "any"],
          description:
            "Filter by chunk kind. \"code\": AST-parsed source only. \"docs\": text-fallback " +
            "chunks only (.md/.json/config/etc). \"any\" (default): no filter.",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "codebase_research",
    description:
      "Reach for this on a broad architectural or cross-subsystem question that one " +
      "codebase_search call won't resolve — a single-shot search under-collects on 'how does " +
      "X work across the system' style questions, where the real answer is scattered. Runs an " +
      "iterative candidate collection loop (semantic search + symbol extraction + regex " +
      "expansion) and returns the top ranked chunks; you synthesize the answer from them, this " +
      "tool does not produce prose. Slower than codebase_search (seconds). @subsystem prefixes " +
      "are stripped — research " +
      "runs unscoped to allow full-corpus symbol expansion. `scope` tunes structural " +
      "weighting: \"explore\" (neighbourhood/dependency questions — how does X work, what " +
      "depends on X, map this subsystem) trades single-target precision for ~2x " +
      "dependency-neighbourhood recall; \"lookup\" (default) is for finding a specific " +
      "symbol/implementation. Result chunk headers carry the same [test]/[docs]/[gen] " +
      "path-heuristic role tags codebase_search uses (untagged = impl).",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Research question or topic." },
        path_prefix: { type: "string", description: "Optional path prefix filter." },
        scope: {
          type: "string",
          enum: ["lookup", "explore"],
          description:
            "\"lookup\" (default): find a specific symbol/implementation. \"explore\": " +
            "neighbourhood/dependency questions — trades single-target precision for " +
            "~2x dependency-neighbourhood recall.",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "codebase_map",
    description:
      "Get oriented in an unfamiliar codebase or subtree before you start reading files — a " +
      "structural symbol map ranked by PageRank importance, the fast substitute for building " +
      "that mental model file-by-file yourself. " +
      "Output is token-budgeted, most important files first, whole-index or scoped; a " +
      "footer says when it was truncated — narrow path_prefix or raise token_budget to see deeper. " +
      "Pass hubs:true for \"what are the load-bearing chunks here\" — ranks chunks by incoming " +
      "reference count and PageRank instead of returning the token-budgeted map. Hubs mode's " +
      "per-hub edge_type breakdown carries a sibling `by_provenance` rollup: extracted (AST " +
      "fact — note: name-resolved, ambiguous names fan out to every definer), inferred (name " +
      "co-occurrence / embedding similarity), paired (cross-language name match).",
    inputSchema: {
      type: "object",
      properties: {
        path_prefix: { type: "string", description: "Scope to this directory prefix." },
        query: { type: "string", description: "Optional investigation context (reserved for future query-biased ranking)." },
        token_budget: { type: "number", description: "Override the output token budget (default 8000; applies to scoped calls too)." },
        hubs: { type: "boolean", description: "Return ranked hub chunks (by in-degree, then PageRank) instead of the token-budgeted map." },
        edge_types: {
          type: "array",
          items: { type: "string" },
          description: "hubs mode only: restrict ranking to these edge types — pass [\"calls\",\"imports\",\"inherits\"] to suppress mention-noise hubs.",
        },
      },
    },
  },
  {
    name: "codebase_status",
    description: "Call this first when a result set looks thin, empty, or off — confirms whether " +
                 "the index itself is the problem before you blame the query. Shows indexed " +
                 "file/chunk counts, DB size, embedding dimension, and indexed root.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "find_symbol",
    description:
      "Jump straight to where a function or class is defined, by name — sharper than grep or " +
      "codebase_search for this because it resolves against a decoupled symbol index, so it " +
      "still finds a method folded into a merged chunk that text search wouldn't surface by " +
      "name. Exact-name (or prefix) lookup covering every named boundary (functions, methods, " +
      "classes, structs). Returns path:line definition sites; a C/C++ type the index only " +
      "forward-declares (e.g. from an external SDK) returns its declarations, marked " +
      "(forward_declaration). Boundary-only scope: " +
      "member fields, local variables, and other non-boundary names are NOT indexed here — " +
      "a miss on one of those is expected, not a tool failure; use codebase_search mode=fts " +
      "instead (it searches chunk content). Use find_symbol when you know a function/class " +
      "name; use codebase_search for concept/fuzzy queries or anything below boundary scope.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Symbol name (exact unless prefix=true)." },
        path_prefix: { type: "string", description: "Scope to this directory prefix." },
        prefix: { type: "boolean", description: "Prefix match instead of exact." },
      },
      required: ["name"],
    },
  },
  {
    name: "find_usages",
    description:
      "Who calls or references a symbol, and how many — a question grep answers with false " +
      "positives (string matches in comments/unrelated identifiers) and no counts; this walks " +
      "the pre-built reference graph instead. The counterpart to find_symbol. Resolves the name to its " +
      "defining chunk(s) via the same symbol index (so a folded-class method or merged-chunk " +
      "member still resolves), then walks the chunk_refs graph backwards to every chunk that " +
      "references it, including cross-file and cross-language edges. Returns path:line call " +
      "sites. Use when you have a concrete symbol name and want its callers/references. " +
      "Pass mode:\"aggregate\" for \"which code depends on X / blast radius\" — returns a " +
      "file-level rollup (reference counts, edge-type breakdown, top referrers per file, and " +
      "a subsystem rollup when configured) instead of a flat per-usage list. Aggregate " +
      "counts and locations are exact AST-resolved facts from the pre-built reference " +
      "graph — do NOT re-verify them with grep; that repeats work the index already did. " +
      "Read a file only if you need the code itself, not to confirm the counts. " +
      "List mode returns at most 100 rows by default: on widely-used symbols, scope with " +
      "path_prefix (see the subsystem map's paths) or prefer aggregate — an unscoped listing " +
      "of a hot symbol is thousands of rows you don't want in context. List mode's render " +
      "lists typed edges (calls/imports/inherits/xlang) and 'associated' edges (high-PMI name " +
      "co-occurrence — stronger signal than plain mentions, still not a proven call) as " +
      "path:line rows, and collapses mentions-type rows (ordinary name co-occurrence, not a " +
      "proven caller) to a single count — use mode:\"aggregate\" for a per-file edge-type " +
      "breakdown that still accounts for the collapsed mentions instead of enumerating every " +
      "row. A zero-result response is not necessarily \"no callers\": qualified names " +
      "(Foo::Bar) that miss exact " +
      "resolution and names indexed above the edge cap both render as empty with a `note` " +
      "explaining why; the above-cap case and a type that is only forward-declared both " +
      "include labeled FTS content-scan matches (whole-word, not graph edges) as a " +
      "fallback. Edges carry a provenance label: extracted (AST " +
      "fact — note: name-resolved, ambiguous names fan out to every definer), inferred " +
      "(name co-occurrence / embedding similarity), paired (cross-language name match). " +
      "List mode's per-usage `provenance` and aggregate mode's `by_provenance` rollup " +
      "surface it.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Symbol name (exact match)." },
        path_prefix: { type: "string", description: "Scope results to referencing chunks under this directory prefix." },
        limit: { type: "number", description: "Cap the number of returned usages (list mode) or files (aggregate mode)." },
        mode: {
          type: "string",
          enum: ["list", "aggregate"],
          description: "\"list\" (default): flat path:line usages. \"aggregate\": file-level blast-radius rollup.",
        },
        rank_by: {
          type: "string",
          enum: ["pagerank_sum", "count"],
          description: "aggregate mode only: rank files by summed referrer PageRank (default) or raw reference count.",
        },
      },
      required: ["name"],
    },
  },
  {
    name: "find_by_message",
    description:
      "Find the source literal that emitted a runtime message/log line/error string you " +
      "already have — the entry point for 'this string appeared at runtime, where does it " +
      "come from', which grep/codebase_search structurally cannot answer when the message " +
      "came from a format string: a concrete 'failed to load foo.png: 404' never text-matches " +
      "the source line 'failed to load %s: %d' that produced it (or an f-string/interpolated " +
      "equivalent) because the concrete value never appears in source. find_by_message matches " +
      "THROUGH format holes (%s/%d, {name}, {0:D4}, ${var}) by checking the message against " +
      "each literal's hole-collapsed skeleton, so it resolves cases grep/codebase_search miss " +
      "entirely, not just faster. Also handles verbatim literals (assertion text, exact string " +
      "constants) via direct substring matching. Paste the whole message/log line, including " +
      "surrounding context (timestamps, log levels) — matching allows for it. Results are " +
      "ranked exact matches first, then template (format-hole) matches by how much constant " +
      "text anchored the match. Use this INSTEAD of grep/codebase_search when the input is a " +
      "message a human or a log actually produced, not a symbol name (use find_symbol for " +
      "that) or a general topic (use codebase_search for that).",
    inputSchema: {
      type: "object",
      properties: {
        message: { type: "string", description: "The runtime message/log line/error string to resolve to source." },
        limit: { type: "number", description: "Cap the number of returned matches (default 20)." },
      },
      required: ["message"],
    },
  },
  {
    name: "investigate",
    description:
      "Use this instead of calling find_symbol then find_usages separately when you need to " +
      "understand what a symbol is AND how it connects — one call returns definition site(s) " +
      "with inline source excerpts, incoming callers, and outgoing calls/uses, fanning out " +
      "server-side to the same underlying queries so it costs one round trip instead of " +
      "several, with the defining chunk's source already attached (no follow-up Read needed " +
      "just to see the definition). The outgoing direction (what " +
      "a symbol calls/uses) exists ONLY through this tool — there is no standalone " +
      "find_outgoing tool; it renders typed edges (calls/imports/inherits/xlang) and " +
      "'associated' edges (high-PMI name co-occurrence — stronger signal than plain " +
      "mentions, still not a proven call) as rows, and collapses mentions-type rows to a " +
      "count (frequently cross-language name-occurrence noise, not a call) — the incoming " +
      "leg (callers) applies the identical typed/associated/mentions render collapse. Each " +
      "incoming/outgoing row carries a provenance label: extracted " +
      "(AST fact — note: name-resolved, ambiguous names fan out to every definer), inferred " +
      "(name co-occurrence / embedding similarity), paired (cross-language name match). " +
      "usages_limit and outgoing_limit each cap their leg at 30 rows by default; " +
      "definition_source_max_chars caps the total inlined-source budget (split across " +
      "definitions, default 4000 chars) — raise it for a symbol you need to read in full, " +
      "or scope with path_prefix for a hot symbol. A leg with zero results renders its " +
      "`note` explaining why (symbol not found, or genuinely no callers/outgoing refs) " +
      "instead of silently rendering empty. If `name` doesn't resolve to any definition, " +
      "all legs come back empty with the same miss note. This tool does not report blast " +
      "radius (file-level aggregate impact) — use find_usages mode:\"aggregate\" for that.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Symbol name (exact match)." },
        path_prefix: { type: "string", description: "Scope usages/outgoing to this directory prefix." },
        usages_limit: { type: "number", description: "Cap incoming-callers rows (default 30)." },
        outgoing_limit: { type: "number", description: "Cap outgoing-calls rows (default 30)." },
        definition_source_max_chars: {
          type: "number",
          description: "Total inlined-source character budget, split across definitions (default 4000).",
        },
      },
      required: ["name"],
    },
  },
  {
    name: "trace_path",
    description:
      "How does symbol A reach symbol B — a query shape neither codebase_search (flat " +
      "retrieval) nor codebase_research (expands outward from seeds, no fixed destination) " +
      "handles. Resolves both names to chunk(s) via the same symbol index find_symbol uses " +
      "(every definition is tried), then runs a bidirectional search over the chunk_refs " +
      "graph (calls/imports/inherits/associated/mentions/xlang — 'associated' is a " +
      "high-PMI name co-occurrence, stronger signal than plain mentions, still not a " +
      "proven call) for the shortest connecting chain. " +
      "Structural paths are always preferred; set include_semantic to also fall back to the " +
      "chunk_neighbors semantic k-NN graph when no structural path exists. Returns the hop " +
      "chain as file:line + name, with each hop's edge type, direction, and provenance " +
      "label: extracted (AST fact — note: name-resolved, ambiguous names fan out to every " +
      "definer), inferred (name co-occurrence / embedding similarity, incl. semantic hops), " +
      "paired (cross-language name match). If `from`/`to` didn't resolve to an exact symbol " +
      "the response echoes which containing chunk it fell back to. The 'Path found' verdict " +
      "carries a weakest-link caveat whenever any hop is associated/mentions/semantic — " +
      "chains of only calls/imports/inherits/xlang keep the confident, caveat-free render.",
    inputSchema: {
      type: "object",
      properties: {
        from: { type: "string", description: "Starting symbol name (exact match)." },
        to: { type: "string", description: "Destination symbol name (exact match)." },
        max_depth: { type: "number", description: "Max hops to search (default 6)." },
        include_semantic: {
          type: "boolean",
          description: "Fall back to semantic k-NN edges if no structural path exists. Default false.",
        },
      },
      required: ["from", "to"],
    },
  },
];
