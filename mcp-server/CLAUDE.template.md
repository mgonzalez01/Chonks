# Chonks MCP — Usage Guide

Copy this file into the CLAUDE.md of the repository being indexed.

Local code-RAG index over this codebase, exposed as nine tools. Per-tool details are in the tool descriptions; this file is the workflow.

## Searching the codebase

**VERY IMPORTANT: NEVER `Grep`, `Glob`, `find`, or any other filesystem enumeration against the repo root.** This codebase is indexed precisely because it is too large to walk: root-scoped searches are slow, noisy, and can time out. Use the Chonks MCP tools first, always.

If the Chonks tools are deferred (present by name only), load them before your first search — one call: ToolSearch with query `"select:mcp__chonks__codebase_search,mcp__chonks__codebase_research,mcp__chonks__codebase_map,mcp__chonks__find_symbol,mcp__chonks__find_usages,mcp__chonks__investigate,mcp__chonks__trace_path,mcp__chonks__find_by_message,mcp__chonks__codebase_status"`. Do not fall back to Grep because they need loading.

Once Chonks gives you a concrete file path, refine with **scoped** `Grep` or `Read` on that path.

**Routing:** use `codebase_map` / `codebase_search` / `codebase_research` for orientation — finding the region. Once you have a concrete symbol name, switch to the precise tools: `find_symbol` (definitions), `find_usages` (callers/references), `trace_path` (how A reaches B), `find_by_message` (you have a runtime message/log line/error string, not a symbol name, and want the code that emits it — works when grep fails because the message came from a format string). Investigating a specific symbol end-to-end (definition + callers + outgoing calls) -> `investigate`. Prefer them over Grep — they resolve names that were folded into merged chunks and skip comment/string false positives. Retrieval points you at code, it doesn't replace reading it: results carry `path:line` citations — `Read` the file there; the chunk is a retrieval primitive, the file is the source of truth.

**Research size:** `codebase_research` with `compact: true` returns only `path:start-end` headers, about a tenth of the full output: use it to orient, then `Read` the lines you need. Leave it off when you want to read the code in the result itself.

**Caveats:**
- Every tool response is prefixed with an index-staleness header (`index: N chunks · built Xh (or Xd) ago`) — that's build time, not on-disk drift; files changed since then need a re-index.
- Docs/config are indexed as nameless `text` chunks: they surface in search/research (often best for *why* questions) but are invisible to `codebase_map`, `find_symbol`, `find_usages`, `investigate`, and `trace_path`.
- If `config.json` defines multiple projects, these tools address only the default one; query others via the HTTP API directly.
- `trace_path` `mentions` hops are name-occurrence evidence, not proven calls — trust chains of typed edges (`calls`/`imports`/`inherits`) more.
