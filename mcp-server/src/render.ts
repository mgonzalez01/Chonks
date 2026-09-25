// Renders tool results into the text the MCP tools return.

import { SUBSYSTEMS, matchSubsystemForPath } from "./subsystems.js";

export type Chunk = {
  id: string;
  path: string;
  language?: string;
  chunk_type?: string;
  name?: string | null;
  start_line: number;
  end_line: number;
  // Absent from a compact /research response.
  content?: string;
  distance?: number;
  _score?: number;
  // Set on hybrid (RRF) results: 1-indexed rank within each branch, or null
  // when the chunk did not appear in that branch's top results.
  _rank_semantic?: number | null;
  _rank_fts?: number | null;
  // /research only: how this chunk was admitted into the pool. A seed or
  // plain semantic hit carries neither field, since it needs no justification.
  _evidence?: { origin: "graph" | "semantic" | "paired"; anchor_id: string; edge_type?: string } | null;
  _struct_anchor?: { anchor_id: string; edge_type: string } | null;
};

// Typed edge among two RETURNED /research chunks. See
// chonks/retrieval/research.py's _result_connections.
type Connection = { from_id: string; to_id: string; edge_type: string; provenance: string };

// Cheap path heuristic, not a content sniff. Same classifier
// drives both the files: summary line and each chunk header's role tag.
function classifyFileRole(path: string): "impl" | "test" | "docs" | "gen" {
  const p = path ?? "";
  const base = basenameOf(p);
  if (p.startsWith("tests/") || p.includes("/tests/") || base.startsWith("test_")) return "test";
  if (p.endsWith(".md") || p.startsWith("docs/") || p.includes("/docs/")) return "docs";
  if (p.startsWith("dist/") || p.includes("/dist/") || p.includes("generated")) return "gen";
  return "impl";
}

function formatChunkHeader(c: Chunk, i: number): string {
  const loc = `${c.path}:${c.start_line}-${c.end_line}`;
  const name = c.name ? ` (${c.name})` : "";
  const role = classifyFileRole(c.path);
  const roleTag = role !== "impl" ? ` [${role}]` : "";
  let scoreFrag = "";
  if (c._score !== undefined) {
    const rankBits: string[] = [];
    if (c._rank_semantic != null) rankBits.push(`sem#${c._rank_semantic}`);
    if (c._rank_fts != null) rankBits.push(`fts#${c._rank_fts}`);
    const ranks = rankBits.length ? `  [${rankBits.join(",")}]` : "";
    // Hybrid result: label as rrf to distinguish from per-branch scores.
    const label = rankBits.length ? "rrf" : "score";
    scoreFrag = `  ${label}=${c._score.toFixed(4)}${ranks}`;
  }
  return `[${i + 1}] ${loc}${name}${roleTag}${scoreFrag}`;
}

export function formatChunks(chunks: Chunk[]): string {
  if (!chunks.length) return "No results.";
  const parts: string[] = [];
  chunks.forEach((c, i) => {
    const lang = c.language ?? "";
    parts.push(`${formatChunkHeader(c, i)}\n\`\`\`${lang}\n${c.content}\n\`\`\``);
  });
  return parts.join("\n\n");
}

function basenameOf(path?: string | null): string {
  if (!path) return "?";
  const segs = path.split("/");
  return segs[segs.length - 1] || path;
}

// Renders /search's `files` field, which the tool used to compute and drop.
type FileRank = { path: string; score?: number; n_chunks?: number; best_rank?: number };

export function formatFilesLine(files: FileRank[], cap = 6): string {
  if (!files || !files.length) return "";
  const shown = files.slice(0, cap);
  const parts = shown.map((f) => {
    const role = classifyFileRole(f.path);
    const n = f.n_chunks ?? 1;
    const suffix = n > 1 ? ` ×${n}` : "";
    return `${f.path} [${role}${suffix}]`;
  });
  const more = files.length - shown.length;
  const tail = more > 0 ? ` · +${more} more` : "";
  return `files: ${parts.join(" · ")}${tail}`;
}

// Must mirror chonks/retrieval/results.py's rank_files ordering exactly (score DESC,
// first-appearance ASC tiebreak), since this recomputes it client-side for
// the merged multi-@subsystem case that has no single backend `files` field.
export function rankFilesFromChunks(chunks: Chunk[]): FileRank[] {
  const order: string[] = [];
  const scores = new Map<string, number>();
  const counts = new Map<string, number>();
  const bestRank = new Map<string, number>();
  chunks.forEach((c, i) => {
    const path = c.path ?? "";
    const score = c._score ?? 0;
    if (!scores.has(path)) {
      order.push(path);
      scores.set(path, score);
      counts.set(path, 0);
      bestRank.set(path, i + 1);
    }
    scores.set(path, Math.max(scores.get(path)!, score));
    counts.set(path, (counts.get(path) ?? 0) + 1);
  });
  return order
    .map((path) => ({
      path, score: scores.get(path)!, n_chunks: counts.get(path)!, best_rank: bestRank.get(path)!,
    }))
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
}

// Must mirror chonks/core/edges.py's PROVENANCE_BY_EDGE_TYPE exactly.
const EDGE_PROVENANCE: Record<string, string> = {
  calls: "extracted", imports: "extracted", inherits: "extracted", contains: "extracted",
  mentions: "inferred", associated: "inferred", semantic: "inferred", xlang: "paired",
};
function provenanceFor(edgeType?: string): string {
  if (!edgeType) return "inferred";
  return EDGE_PROVENANCE[edgeType] ?? "inferred";
}

// Research-only rendering: adds an evidence line formatChunks doesn't have.
export function formatResearchChunks(chunks: Chunk[]): string {
  if (!chunks.length) return "No results.";

  const idToRank = new Map<string, number>();
  const idToChunk = new Map<string, Chunk>();
  chunks.forEach((c, i) => {
    idToRank.set(c.id, i + 1);
    idToChunk.set(c.id, c);
  });

  const anchorLabel = (anchorId: string | null | undefined): string => {
    if (!anchorId) return "unknown";
    const anchor = idToChunk.get(anchorId);
    if (!anchor) return anchorId; // anchor didn't survive to the final returned set
    const label = anchor.name || basenameOf(anchor.path);
    const rank = idToRank.get(anchorId);
    return rank ? `${label} [#${rank}]` : label;
  };

  const parts: string[] = [];
  chunks.forEach((c, i) => {
    const lang = c.language ?? "";
    let evidenceLine = "";
    const ev = c._evidence;
    const sa = c._struct_anchor;
    if (ev?.origin === "graph") {
      evidenceLine = `\n    evidence: expanded via ${ev.edge_type}(${provenanceFor(ev.edge_type)}) from ${anchorLabel(ev.anchor_id)}`;
    } else if (ev?.origin === "semantic") {
      evidenceLine = `\n    evidence: semantic neighbor of ${anchorLabel(ev.anchor_id)}`;
    } else if (ev?.origin === "paired") {
      evidenceLine = `\n    evidence: header/impl pair of ${anchorLabel(ev.anchor_id)}`;
    } else if (sa) {
      evidenceLine = `\n    evidence: wired to ${anchorLabel(sa.anchor_id)} via ${sa.edge_type}`;
    }
    // else: a seed, omit the line, it needs no justification.
    parts.push(`${formatChunkHeader(c, i)}${evidenceLine}\n\`\`\`${lang}\n${c.content}\n\`\`\``);
  });
  return parts.join("\n\n");
}

export function formatResearchHeaders(chunks: Chunk[]): string {
  if (!chunks.length) return "No results.";
  return chunks.map((c, i) => formatChunkHeader(c, i)).join("\n");
}

// Returns "" (not null) when empty, so callers can append it directly.
export function formatConnections(connections: Connection[] | undefined, chunks: Chunk[]): string {
  if (!connections || !connections.length) return "";

  const idToRank = new Map<string, number>();
  const idToChunk = new Map<string, Chunk>();
  chunks.forEach((c, i) => {
    idToRank.set(c.id, i + 1);
    idToChunk.set(c.id, c);
  });

  type Line = { fileA: string; rankA?: number; fileB: string; rankB?: number; edgeType: string; provenance: string };
  const order: string[] = [];
  const rendered = new Map<string, Line>();
  const counts = new Map<string, number>();

  for (const conn of connections) {
    const a = idToChunk.get(conn.from_id);
    const b = idToChunk.get(conn.to_id);
    const fileA = a ? basenameOf(a.path) : conn.from_id;
    const fileB = b ? basenameOf(b.path) : conn.to_id;
    const key = `${fileA}\x00${fileB}\x00${conn.edge_type}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
    if (!rendered.has(key)) {
      order.push(key);
      rendered.set(key, {
        fileA, rankA: idToRank.get(conn.from_id),
        fileB, rankB: idToRank.get(conn.to_id),
        edgeType: conn.edge_type, provenance: conn.provenance,
      });
    }
  }

  const shown = order.slice(0, 10).map((key) => {
    const l = rendered.get(key)!;
    const n = counts.get(key)!;
    const suffix = n > 1 ? `  ×${n}` : "";
    const rankAFrag = l.rankA ? ` [#${l.rankA}]` : "";
    const rankBFrag = l.rankB ? ` [#${l.rankB}]` : "";
    return `  ${l.fileA}${rankAFrag} --${l.edgeType}(${l.provenance})--> ${l.fileB}${rankBFrag}${suffix}`;
  });
  const more = order.length > 10 ? `\n  (+${order.length - 10} more)` : "";
  return `\n\nCONNECTIONS\n${shown.join("\n")}${more}`;
}

// Only fires when chunk_kind was left at default; an explicit choice
// shouldn't get second-guessed.
export function docsNudgeFooter(chunkKindArg: unknown, docsInResults: unknown, count: unknown): string {
  if (chunkKindArg !== undefined && chunkKindArg !== "any") return "";
  const n = typeof docsInResults === "number" ? docsInResults : 0;
  const m = typeof count === "number" ? count : 0;
  if (m === 0 || n / m <= 2 / 3) return "";
  return `\n\n${n} of ${m} results are documentation chunks — pass chunk_kind:"code" to filter.`;
}

// Nudges an agent stuck reformulating into the same near-duplicate wall.
// Threshold must mirror the server's NEAR_DUP_WALL_SHARE_THRESHOLD.
const NEAR_DUP_FIRING_THRESHOLD = 0.6;

export function nearDupFooter(nearDup: unknown, count: unknown): string {
  if (!nearDup || typeof nearDup !== "object") return "";
  const wallShare = (nearDup as { wall_share?: unknown }).wall_share;
  const wallSize = (nearDup as { wall_size?: unknown }).wall_size;
  if (typeof wallShare !== "number" || wallShare < NEAR_DUP_FIRING_THRESHOLD) return "";
  const n = typeof wallSize === "number" ? wallSize : "most";
  const m = typeof count === "number" ? count : "the";
  return `\n\n${n} of ${m} results are near-duplicates of each other — try @subsystem scoping, ` +
    `a more specific query, or codebase_search with mode regex for idiom-shaped targets.`;
}

function formatEdgeTypeCounts(edgeTypes: Record<string, number> | undefined): string {
  return Object.entries(edgeTypes ?? {})
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}:${v}`)
    .join(" ");
}

export function renderImpact(res: any): string {
  const symbol = res.symbol;
  const definitions = (res.definitions ?? []) as any[];
  const totalRefs = res.total_references ?? 0;
  const edgeSummary = formatEdgeTypeCounts(res.by_edge_type).replace(/ /g, ", ") || "none";
  const provSummary = formatEdgeTypeCounts(res.by_provenance).replace(/ /g, ", ") || "none";
  const lines: string[] = [
    `Impact: "${symbol}" — ${totalRefs} reference(s) across ${definitions.length} definition(s) (${edgeSummary})`,
    `provenance: ${provSummary}`,
  ];
  // Definitions with paths so the agent can cite the definition site without
  // a follow-up find_symbol/Read round-trip.
  for (const d of definitions.slice(0, 3)) {
    lines.push(`defined: ${d.path} (${d.chunk_type})`);
  }
  if (definitions.length > 3) lines.push(`(+${definitions.length - 3} more definitions)`);

  const files = (res.files ?? []) as any[];
  if (!files.length) {
    lines.push("No referencing files.");
    if (res.note) lines.push(res.note);
  } else {
    for (const f of files) {
      const edgePart = formatEdgeTypeCounts(f.edge_types);
      // name@line: a citable location per referrer, so "verify by reading"
      // costs nothing since the citation is already in hand.
      const top = (f.top_referrers ?? [])
        .map((r: any) => `${r.name || r.chunk_type || "?"}@${r.start_line}`)
        .join(", ");
      lines.push(
        `${f.path} — ${f.count} refs${edgePart ? ` (${edgePart})` : ""}${top ? ` · top: ${top}` : ""}`,
      );
    }
    const filesTotal = res.files_total ?? files.length;
    if (filesTotal > files.length) {
      lines.push(`(showing ${files.length} of ${filesTotal} files)`);
    }
  }

  if (Object.keys(SUBSYSTEMS).length && files.length) {
    const rollup = new Map<string, { count: number; files: number }>();
    for (const f of files) {
      const sub = matchSubsystemForPath(f.path) ?? "(unmapped)";
      const cur = rollup.get(sub) ?? { count: 0, files: 0 };
      cur.count += f.count ?? 0;
      cur.files += 1;
      rollup.set(sub, cur);
    }
    const rows = Array.from(rollup.entries()).sort(
      (a, b) => b[1].count - a[1].count || a[0].localeCompare(b[0]),
    );
    lines.push("", "By subsystem:");
    for (const [subName, agg] of rows) {
      lines.push(`  ${subName}: ${agg.count} ref(s) across ${agg.files} file(s)`);
    }
  }

  lines.push(
    "",
    "Counts are exact AST-resolved reference edges, not similarity estimates — " +
    "re-verifying them via grep is redundant. Cite the name@line locations directly.",
  );
  return lines.join("\n");
}

export function renderHubs(res: any): string {
  const hubs = (res.hubs ?? []) as any[];
  if (!hubs.length) return "No hubs found.";
  return hubs
    .map((h) => {
      const edgePart = formatEdgeTypeCounts(h.edge_types);
      // by_provenance omitted here: derivable on sight from edge_types.
      const pr = typeof h.pagerank === "number" ? h.pagerank.toFixed(4) : "0.0000";
      return `${h.path}:${h.name} (${h.chunk_type}) — in:${h.in_degree} pr:${pr}${edgePart ? ` (${edgePart})` : ""}`;
    })
    .join("\n");
}

// Unknown/future edge_type defaults to the weakest tier (0), matching the
// backend's edge_provenance "degrade to least-certain" convention.
const HOP_TIER: Record<string, number> = {
  calls: 3, imports: 3, inherits: 3,
  xlang: 2,
  associated: 1,
  mentions: 0, semantic: 0,
};

// Returns "" for an all-typed chain, else a line toolTracePath prepends.
export function weakestLinkCaveat(hops: any[]): string {
  let weakest: any = null;
  let weakestTier = Infinity;
  for (const h of hops) {
    const tier = HOP_TIER[h.edge_type] ?? 0;
    if (tier < weakestTier) {
      weakestTier = tier;
      weakest = h;
    }
  }
  if (!weakest || weakestTier >= 2) return "";
  // provenance, not the tier default, decides if the co-occurrence claim is true.
  if (weakest.provenance === "extracted" || weakest.provenance === "paired") return "";
  const body = weakest.edge_type === "associated"
    ? "High-PMI name co-occurrence, stronger than plain mentions but still not a proven call chain; Read both endpoints before citing it."
    : "Name co-occurrence, not a proven call chain; Read both endpoints before citing it.";
  return `weakest link: ${weakest.edge_type} (${weakest.provenance}). ${body}`;
}

// Surfaces a silent containing-chunk substitution for a non-boundary name.
export function resolutionEcho(label: "from" | "to", requested: string, chunk: any): string {
  if (!chunk || !chunk.name || chunk.name === requested) return "";
  // Bare resolving to a qualified symbol's exact definition is not
  // containment (get_longest_axis -> AABB::get_longest_axis), so no echo.
  if (chunk.name.split(/::|\./).pop() === requested) return "";
  const loc = chunk.path ? `${chunk.path}:${chunk.start_line ?? "?"}` : "?";
  return `(${label} '${requested}' — resolved to containing chunk ${chunk.name}, ${loc})`;
}

export function _formatTraceChunk(c: any): string {
  if (!c) return "?";
  const loc = c.path ? `${c.path}:${c.start_line ?? "?"}` : "?";
  return c.name ? `${loc} ${c.name}` : loc;
}
