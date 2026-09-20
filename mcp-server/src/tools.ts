// The nine MCP tool implementations.

import { RESEARCH_TIMEOUT_MS } from "./config.js";
import { httpPost, httpGet } from "./client.js";
import {
  type Chunk,
  formatChunks,
  formatFilesLine,
  rankFilesFromChunks,
  formatResearchChunks,
  formatConnections,
  docsNudgeFooter,
  nearDupFooter,
  renderImpact,
  renderHubs,
  weakestLinkCaveat,
  resolutionEcho,
  _formatTraceChunk,
} from "./render.js";
import { parseSubsystemPrefix, stripSubsystemPrefix } from "./subsystems.js";

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

function dedupeByIdKeepBest(chunks: Chunk[]): Chunk[] {
  const map = new Map<string, Chunk>();
  for (const c of chunks) {
    const existing = map.get(c.id);
    if (!existing || (c._score ?? 0) > (existing._score ?? 0)) {
      map.set(c.id, c);
    }
  }
  return Array.from(map.values()).sort((a, b) => (b._score ?? 0) - (a._score ?? 0));
}

export async function toolCodebaseSearch(args: any): Promise<string> {
  const rawQuery = String(args.query ?? "").trim();
  if (!rawQuery) throw new Error("query is required");
  const mode = String(args.mode ?? "semantic");
  // Default 10: measured ~1.2k tokens at 10 vs ~12.5k at 20. Agents
  // use the default, not the description's advice, so the default IS the policy.
  const topK = Math.min(Math.max(Number(args.top_k ?? 10), 1), 100);
  const chunkKind = typeof args.chunk_kind === "string" ? args.chunk_kind : undefined;

  const { paths, query, name } = parseSubsystemPrefix(rawQuery);

  if (!paths) {
    const pathPrefix = typeof args.path_prefix === "string" ? args.path_prefix : null;
    const res = await httpPost("/search", {
      query,
      mode,
      top_k: topK,
      path_prefix: pathPrefix,
      chunk_kind: chunkKind,
    });
    const scopeNote = pathPrefix ? `scope: ${pathPrefix}` : "scope: whole index";
    const count = res.count ?? (res.chunks?.length ?? 0);
    const nudge = docsNudgeFooter(chunkKind, res.docs_in_results, count) + nearDupFooter(res.near_dup, count);
    const filesLine = formatFilesLine(res.files ?? []);
    const filesBlock = filesLine ? `${filesLine}\n\n` : "";
    return `(${scopeNote} · ${count} results)\n\n${filesBlock}${formatChunks(res.chunks ?? [])}${nudge}`;
  }

  if (paths.length === 1) {
    const res = await httpPost("/search", {
      query,
      mode,
      top_k: topK,
      path_prefix: paths[0],
      chunk_kind: chunkKind,
    });
    const count = res.count ?? (res.chunks?.length ?? 0);
    const nudge = docsNudgeFooter(chunkKind, res.docs_in_results, count) + nearDupFooter(res.near_dup, count);
    const filesLine = formatFilesLine(res.files ?? []);
    const filesBlock = filesLine ? `${filesLine}\n\n` : "";
    return `(scope: @${name} → ${paths[0]} · ${count} results)\n\n${filesBlock}${formatChunks(res.chunks ?? [])}${nudge}`;
  }

  // Nudge uses pre-merge per-branch sums, an approximation against the
  // final merged set rather than re-deriving the backend's code/docs split.
  const responses = await Promise.all(
    paths.map((p) =>
      httpPost("/search", { query, mode, top_k: topK, path_prefix: p, chunk_kind: chunkKind }).catch((e) => {
        console.error(`[chonks-mcp] search failed for ${p}: ${(e as Error).message}`);
        return { chunks: [] };
      })
    )
  );
  const merged = dedupeByIdKeepBest(responses.flatMap((r) => r.chunks ?? []));
  const top = merged.slice(0, topK);
  const totalDocs = responses.reduce((acc, r) => acc + (r.docs_in_results ?? 0), 0);
  const totalCount = responses.reduce((acc, r) => acc + (r.count ?? (r.chunks?.length ?? 0)), 0);
  // Worst-case proxy: each branch's wall_share was computed pre-merge, and
  // re-running detection here would need raw vectors this layer doesn't have.
  const worstNearDup = responses.reduce((worst: { wall_share: number } | null, r: any) => {
    const nd = r.near_dup;
    if (!nd || typeof nd.wall_share !== "number") return worst;
    return !worst || nd.wall_share > worst.wall_share ? nd : worst;
  }, null);
  const nudge = docsNudgeFooter(chunkKind, totalDocs, totalCount) + nearDupFooter(worstNearDup, totalCount);
  const filesLine = formatFilesLine(rankFilesFromChunks(top));
  const filesBlock = filesLine ? `${filesLine}\n\n` : "";
  return `(scope: @${name} × ${paths.length} paths · ${top.length} results)\n\n${filesBlock}${formatChunks(top)}${nudge}`;
}

const RESEARCH_EXPLORE_EDGE_TYPE_WEIGHTS = { calls: 8, imports: 8, inherits: 8, xlang: 8, associated: 1, mentions: 1 };

export async function toolCodebaseResearch(args: any): Promise<string> {
  const rawQuery = String(args.query ?? "").trim();
  if (!rawQuery) throw new Error("query is required");
  // Research runs unscoped, so strip any @subsystem prefix and ignore it.
  // The iterative symbol-expansion loop benefits from full-corpus access.
  const query = stripSubsystemPrefix(rawQuery);
  const pathPrefix = typeof args.path_prefix === "string" ? args.path_prefix : null;
  const scope = args.scope === "explore" ? "explore" : "lookup";

  const body: Record<string, unknown> = { query, path_prefix: pathPrefix };
  if (scope === "explore") {
    body.edge_type_weights = RESEARCH_EXPLORE_EDGE_TYPE_WEIGHTS;
  }
  const res = await httpPost("/research", body, RESEARCH_TIMEOUT_MS);

  // Backend keeps answering (degraded) when the embedder is down; the result
  // looks normal but isn't, so flag it in the first line the agent reads.
  const degradedLine = res.degraded
    ? `DEGRADED (${res.degraded}): the embedding server was unreachable, so this result used ` +
      `keyword seeds and unscored expansion only. Treat it as partial; check the embedder and retry.\n\n`
    : "";

  const header = `Deep research returned ${res.count ?? 0} chunks after ${res.iterations ?? 0} iteration(s). ` +
                 `Synthesize an answer from these and cite file:line references.`;
  const files = (res.files ?? []).slice(0, 8).map((f: any) => f.path).join(", ");
  const filesLine = files ? `files: ${files}\n\n` : "";
  const chunks: Chunk[] = res.chunks ?? [];
  const connectionsSection = formatConnections(res.connections, chunks);
  return `${degradedLine}${header}\n\n${filesLine}${formatResearchChunks(chunks)}${connectionsSection}`;
}

export async function toolCodebaseMap(args: any): Promise<string> {
  const pathPrefix = typeof args.path_prefix === "string" ? args.path_prefix : null;
  const query = typeof args.query === "string" ? args.query : null;
  const tokenBudget = typeof args.token_budget === "number" ? args.token_budget : null;
  const hubs = args.hubs === true;
  const edgeTypes = Array.isArray(args.edge_types) ? args.edge_types : undefined;

  if (hubs) {
    const res = await httpPost("/hubs", { path_prefix: pathPrefix, edge_types: edgeTypes });
    return renderHubs(res);
  }

  const res = await httpPost("/repomap", {
    path_prefix: pathPrefix,
    query,
    token_budget: tokenBudget,
  });
  return res.map ?? "(empty map)";
}

export async function toolFindSymbol(args: any): Promise<string> {
  const name = typeof args.name === "string" ? args.name.trim() : "";
  if (!name) return "find_symbol requires a non-empty 'name'.";
  const pathPrefix = typeof args.path_prefix === "string" ? args.path_prefix : null;
  const prefix = args.prefix === true;
  const res = await httpPost("/symbol", { name, path_prefix: pathPrefix, prefix });
  const rows = (res.symbols ?? []) as any[];
  if (!rows.length) {
    const base = `No symbol ${prefix ? `prefixed "${name}"` : `named "${name}"`} found.`;
    return res.note ? `${base} ${res.note}` : base;
  }
  const lines = rows.map(
    (r) => `${r.name}  ${r.path}:${r.start_line}${r.kind ? `  (${r.kind})` : ""}`,
  );
  return `${res.count ?? rows.length} match(es):\n${lines.join("\n")}`;
}

export async function toolFindUsages(args: any): Promise<string> {
  const name = typeof args.name === "string" ? args.name.trim() : "";
  if (!name) return "find_usages requires a non-empty 'name'.";
  const pathPrefix = typeof args.path_prefix === "string" ? args.path_prefix : null;
  // undefined (not null) so JSON.stringify drops the key when the caller
  // omits limit: /impact rejects explicit nulls less gracefully than /usages.
  const limit = typeof args.limit === "number" ? args.limit : undefined;
  const mode = args.mode === "aggregate" ? "aggregate" : "list";
  const rankBy = typeof args.rank_by === "string" ? args.rank_by : undefined;

  if (mode === "aggregate") {
    const res = await httpPost("/impact", { name, path_prefix: pathPrefix, limit, rank_by: rankBy });
    return renderImpact(res);
  }

  // Default cap: /usages is unbounded server-side; measured 2.16MB of rows
  // from ONE unlimited call on a hot symbol.
  const effLimit = limit ?? 100;
  const res = await httpPost("/usages", { name, path_prefix: pathPrefix, limit: effLimit });
  const rows = (res.usages ?? []) as any[];
  const contentMatches = (res.content_matches ?? []) as any[];
  if (!rows.length) {
    // content_matches (FTS scan) is labeled distinctly so it can't be
    // mistaken for a typed graph edge.
    let out = res.note ? `No usages of "${name}" found. ${res.note}` : `No usages of "${name}" found.`;
    if (contentMatches.length) {
      const lines = contentMatches.map(
        (r) => `${r.path}:${r.start_line}${r.name ? `  ${r.name}` : ""}${r.chunk_type ? `  (${r.chunk_type})` : ""}`,
      );
      out += `\n\n${contentMatches.length} content match(es) (FTS scan, not graph edges):\n${lines.join("\n")}`;
    }
    return out;
  }
  // mentions-type rows collapse to a count; doc-mention noise otherwise
  // drowns real callers within the row cap.
  const typedRows = rows.filter((r) => r.edge_type !== "mentions");
  const mentionRows = rows.filter((r) => r.edge_type === "mentions");
  const lines = typedRows.map(
    (r) => `${r.path}:${r.start_line}${r.name ? `  ${r.name}` : ""}${r.chunk_type ? `  (${r.chunk_type})` : ""}` +
      `${r.provenance ? `  [${r.provenance}${r.edge_type ? `, ${r.edge_type}` : ""}]` : ""}`,
  );
  if (mentionRows.length) {
    lines.push(
      `${mentionRows.length} name-mention ref(s) omitted (name co-occurrence, ` +
      "not a proven caller — mode:\"aggregate\" rolls them up per file)",
    );
  }
  let out = `${res.count ?? rows.length} usage(s):\n${lines.join("\n")}`;
  if (res.note) {
    out += `\n${res.note}`;
  }
  if ((res.count ?? rows.length) >= effLimit) {
    out += `\n(showing first ${rows.length} — scope with path_prefix, raise limit, ` +
           `or use mode:"aggregate" for the file-level rollup)`;
  }
  return out;
}

export async function toolFindByMessage(args: any): Promise<string> {
  const message = typeof args.message === "string" ? args.message.trim() : "";
  if (!message) return "find_by_message requires a non-empty 'message'.";
  const limit = typeof args.limit === "number" ? args.limit : undefined;

  const res = await httpPost("/find_by_message", { message, limit });
  const rows = (res.results ?? []) as any[];
  if (!rows.length) {
    return res.note ? `No literal matches for "${message}". ${res.note}` : `No literal matches for "${message}".`;
  }
  // ␀* marks a format hole in the rendered skeleton (same sentinel the
  // index stores, see chonks/store.py's _HOLE_SENTINEL) so a skeleton hit
  // reads as a template, not a literal transcription of the pasted message.
  const lines = rows.map((r) => {
    const kind = r.match_kind === "skeleton" ? "template" : "exact";
    const literal = r.match_kind === "skeleton" && r.skeleton ? r.skeleton : r.matched_literal;
    return `${r.path}:${r.line}  [${kind}]  ${literal}${r.name ? `  (${r.name})` : ""}`;
  });
  let out = `${res.count ?? rows.length} match(es):\n${lines.join("\n")}`;
  if (res.note) {
    out += `\n${res.note}`;
  }
  return out;
}

// Inlines definition source: an A/B found agents treating the prior
// summary-only composite as sufficient and reading fewer files.
export async function toolInvestigate(args: any): Promise<string> {
  const name = typeof args.name === "string" ? args.name.trim() : "";
  if (!name) return "investigate requires a non-empty 'name'.";
  const pathPrefix = typeof args.path_prefix === "string" ? args.path_prefix : null;
  const usagesLimit = typeof args.usages_limit === "number" ? args.usages_limit : undefined;
  const outgoingLimit = typeof args.outgoing_limit === "number" ? args.outgoing_limit : undefined;
  const definitionSourceMaxChars =
    typeof args.definition_source_max_chars === "number" ? args.definition_source_max_chars : undefined;

  const res = await httpPost("/investigate", {
    name, path_prefix: pathPrefix,
    usages_limit: usagesLimit, outgoing_limit: outgoingLimit,
    // Smallest leg keeps the server-side short-circuit identical without
    // paying for a file rollup the render (below) never uses.
    impact_limit: 1,
    definition_source: true,
    definition_source_max_chars: definitionSourceMaxChars,
  });

  const definitions = (res.definitions ?? []) as any[];
  if (!definitions.length) {
    const note = res.notes?.definitions;
    return note ? `No symbol named "${name}" found. ${note}` : `No symbol named "${name}" found.`;
  }

  const lines: string[] = [`Investigate: "${name}"`, "", "Definition(s):"];
  for (const d of definitions) {
    lines.push(`${d.path}:${d.start_line}-${d.end_line}${d.kind ? `  (${d.kind})` : ""}`);
    if (d.source) {
      lines.push("```" + (d.language ?? ""), d.source, "```");
    }
    lines.push("");
  }

  lines.push("Incoming — callers/references:");
  const usages = (res.usages?.usages ?? []) as any[];
  const contentMatches = (res.usages?.content_matches ?? []) as any[];
  if (!usages.length) {
    lines.push(`  ${res.notes?.usages ?? "no incoming references found."}`);
    // Above-cap fallback: the note promises an FTS content scan, so dropping
    // its rows here would render the promise over nothing (a silent
    // lie). Same labeled-distinctly rendering toolFindUsages uses.
    if (contentMatches.length) {
      lines.push(`  ${contentMatches.length} content match(es) (FTS scan, not graph edges):`);
      for (const r of contentMatches) {
        lines.push(`    ${r.path}:${r.start_line}${r.name ? `  ${r.name}` : ""}${r.chunk_type ? `  (${r.chunk_type})` : ""}`);
      }
    }
  } else {
    // Typed edges listed; mentions-type rows collapse to a count, same
    // treatment as the outgoing leg below (see Store.find_usages).
    const typedUsages = usages.filter((r) => r.edge_type !== "mentions");
    const mentionUsages = usages.filter((r) => r.edge_type === "mentions");
    for (const r of typedUsages) {
      lines.push(
        `  ${r.path}:${r.start_line}${r.name ? `  ${r.name}` : ""}` +
        `${r.provenance ? `  [${r.provenance}${r.edge_type ? `, ${r.edge_type}` : ""}]` : ""}`,
      );
    }
    if (mentionUsages.length) {
      lines.push(
        `  ${mentionUsages.length} name-mention ref(s) omitted (name co-occurrence, ` +
        "not a proven caller — find_usages mode:\"aggregate\" rolls them up per file)",
      );
    }
    if (res.notes?.usages) {
      lines.push(`  ${res.notes.usages}`);
    }
  }

  lines.push("", "Outgoing — what it calls/uses:");
  const outgoing = (res.outgoing?.outgoing ?? []) as any[];
  if (!outgoing.length) {
    lines.push(`  ${res.notes?.outgoing ?? "no outgoing references found."}`);
  } else {
    // mentions-type rows collapse to one count line: on the Godot corpus
    // mentions dominate the raw edge count and drown the typed signal.
    const typedRows = outgoing.filter((r) => r.edge_type !== "mentions");
    const mentionRows = outgoing.filter((r) => r.edge_type === "mentions");
    for (const r of typedRows) {
      lines.push(
        `  ${r.path}:${r.start_line}${r.name ? `  ${r.name}` : ""}` +
        `${r.provenance ? `  [${r.provenance}${r.edge_type ? `, ${r.edge_type}` : ""}]` : ""}`,
      );
    }
    if (mentionRows.length) {
      lines.push(
        `  ${mentionRows.length} name-mention ref(s) omitted (name co-occurrence, ` +
        "frequently cross-language noise — POST /outgoing for the full list)",
      );
    }
  }

  // Steering line is measured, not decorative: the same A/B found agents
  // stopping early on this composite.
  lines.push(
    "",
    "Definition source is inline above; caller/callee context is not — Read " +
    "the cited caller sites before drawing conclusions about how this " +
    "symbol is used.",
  );
  return lines.join("\n");
}

export async function toolTracePath(args: any): Promise<string> {
  const from = typeof args.from === "string" ? args.from.trim() : "";
  const to = typeof args.to === "string" ? args.to.trim() : "";
  if (!from || !to) return "trace_path requires non-empty 'from' and 'to' symbol names.";
  const maxDepth = typeof args.max_depth === "number" ? args.max_depth : null;
  const includeSemantic = args.include_semantic === true;
  const res = await httpPost("/trace", {
    from, to, max_depth: maxDepth ?? undefined, include_semantic: includeSemantic,
  });
  if (!res.found) return res.error ?? `No path found from "${from}" to "${to}".`;
  const hops = (res.hops ?? []) as any[];
  if (!hops.length) {
    return `"${from}" and "${to}" resolve to the same chunk — no traversal needed.`;
  }
  const chain: string[] = [];
  hops.forEach((h, i) => {
    if (i === 0) chain.push(_formatTraceChunk(h.from_chunk));
    chain.push(`--${h.edge_type}(${h.direction},${h.provenance})-->`);
    chain.push(_formatTraceChunk(h.to_chunk));
  });
  const semanticNote = res.used_semantic ? " (fell back to semantic k-NN)" : "";
  const caveat = weakestLinkCaveat(hops);
  const echoLines = [
    resolutionEcho("from", from, hops[0].from_chunk),
    resolutionEcho("to", to, hops[hops.length - 1].to_chunk),
  ].filter(Boolean);
  if (caveat) echoLines.unshift(caveat);
  const echoBlock = echoLines.length ? `\n${echoLines.join("\n")}` : "";
  return `Path found (${res.depth} hop${res.depth === 1 ? "" : "s"})${semanticNote}:${echoBlock}\n${chain.join(" ")}`;
}

export async function toolCodebaseStatus(): Promise<string> {
  const s = await httpGet("/status");
  if (!s?.ready) {
    return "Backend reports: not ready.";
  }
  const files = s.files ?? "?";
  const chunks = s.chunks ?? "?";
  const dim = s.embedding_dim ?? "?";
  const sizeMb = s.db_size_mb !== undefined ? `${Number(s.db_size_mb).toFixed(2)} MB` : "?";
  const root = s.indexed_root ?? "(unknown)";
  return [
    `Chonks status:`,
    `  files:         ${files}`,
    `  chunks:        ${chunks}`,
    `  db size:       ${sizeMb}`,
    `  embedding dim: ${dim}`,
    `  indexed root:  ${root}`,
  ].join("\n");
}
