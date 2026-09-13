#!/usr/bin/env node
/**
 * chonks-mcp: MCP server exposing the Chonks HTTP backend as tools. Config env vars are documented in DEPLOY.md.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createServer as createHttpServer, IncomingMessage } from "node:http";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { spawn, ChildProcess } from "node:child_process";
import { dirname } from "node:path";
import { readFileSync, existsSync } from "node:fs";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const SERVER_URL = (process.env.CHONKS_URL ?? "http://localhost:11438").replace(/\/$/, "");
const RESEARCH_TIMEOUT_MS = Number(process.env.CHONKS_RESEARCH_TIMEOUT_MS ?? 900_000);
const CONFIG_PATH = process.env.CHONKS_CONFIG ?? null;
const SUBSYSTEMS_ENV = process.env.CHONKS_SUBSYSTEMS ?? null;

const RESEARCH_EXPLORE_EDGE_TYPE_WEIGHTS = { calls: 8, imports: 8, inherits: 8, xlang: 8, associated: 1, mentions: 1 };

const SERVER_PY = process.env.CHONKS_SERVER_PY ?? null;
const DB_PATH = process.env.CHONKS_DB ?? null;
const PY_CMD = (process.env.CHONKS_PY_CMD ?? "uv run python").trim();

// ---------------------------------------------------------------------------
// Subsystem map loading
// ---------------------------------------------------------------------------

type SubsystemMap = Record<string, string[]>;

function loadSubsystems(): SubsystemMap {
  if (SUBSYSTEMS_ENV) {
    try {
      const parsed = JSON.parse(SUBSYSTEMS_ENV);
      if (parsed && typeof parsed === "object") return normalizeSubsystems(parsed);
    } catch (e) {
      console.error(`[chonks-mcp] failed to parse CHONKS_SUBSYSTEMS: ${(e as Error).message}`);
    }
  }
  if (CONFIG_PATH) {
    if (!existsSync(CONFIG_PATH)) {
      // A set-but-missing path is usually a relative path resolving against an
      // unexpected cwd. Say so instead of silently running without @subsystem
      // support.
      console.error(
        `[chonks-mcp] CHONKS_CONFIG=${CONFIG_PATH} not found (cwd: ${process.cwd()}) — ` +
        `@subsystem prefixes disabled`);
      return {};
    }
    try {
      const raw = JSON.parse(readFileSync(CONFIG_PATH, "utf8"));
      if (raw?.subsystems && typeof raw.subsystems === "object") {
        return normalizeSubsystems(raw.subsystems);
      }
    } catch (e) {
      console.error(`[chonks-mcp] failed to load ${CONFIG_PATH}: ${(e as Error).message}`);
    }
  }
  return {};
}

function normalizeSubsystems(raw: Record<string, unknown>): SubsystemMap {
  const out: SubsystemMap = {};
  for (const [k, v] of Object.entries(raw)) {
    if (Array.isArray(v)) {
      const paths = v.filter((p): p is string => typeof p === "string" && p.length > 0);
      if (paths.length) out[k] = paths;
    } else if (typeof v === "string" && v.length > 0) {
      // Tolerate string form, but the backend expects arrays.
      out[k] = [v];
    }
  }
  return out;
}

const SUBSYSTEMS = loadSubsystems();

// ---------------------------------------------------------------------------
// Subsystem prefix parsing
// ---------------------------------------------------------------------------

/** paths is null when the query has no @subsystem_name prefix, or the name isn't a known subsystem. */
function parseSubsystemPrefix(query: string): { paths: string[] | null; query: string; name: string | null } {
  const match = query.match(/^@([A-Za-z_][A-Za-z0-9_-]*)\s+(.+)$/s);
  if (!match) return { paths: null, query, name: null };
  const [, name, rest] = match;
  const paths = SUBSYSTEMS[name];
  if (!paths) return { paths: null, query, name: null };
  return { paths, query: rest, name };
}

/** Strips a leading @subsystem_name whether or not it's a known subsystem. */
function stripSubsystemPrefix(query: string): string {
  const match = query.match(/^@[A-Za-z_][A-Za-z0-9_-]*\s+(.+)$/s);
  return match ? match[1] : query;
}

/** Reverse of parseSubsystemPrefix: first matching path prefix wins. */
function matchSubsystemForPath(path: string): string | null {
  for (const [name, prefixes] of Object.entries(SUBSYSTEMS)) {
    for (const prefix of prefixes) {
      // Segment-boundary match: "src/eng" must not claim "src/engineering/".
      const p = prefix.endsWith("/") ? prefix : prefix + "/";
      if (path === prefix || path.startsWith(p)) return name;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

async function httpPost(path: string, body: unknown, timeoutMs = 60_000): Promise<any> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const res = await fetch(`${SERVER_URL}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function httpGet(path: string, timeoutMs = 10_000): Promise<any> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${SERVER_URL}${path}`, { signal: ctrl.signal });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Backend lifecycle (optional)
// ---------------------------------------------------------------------------

let backend: ChildProcess | null = null;

// ---------------------------------------------------------------------------
// Backend respawn
// ---------------------------------------------------------------------------

const RESPAWN_MAX_ATTEMPTS = 5;
const RESPAWN_WINDOW_MS = 60_000;
const RESPAWN_BASE_DELAY_MS = 1_000;

let respawnTimestamps: number[] = [];
let respawnInFlight = false;

function scheduleRespawn(): void {
  // guards against a second exit event racing an in-progress respawn
  if (respawnInFlight) return;
  const now = Date.now();
  respawnTimestamps = respawnTimestamps.filter((t) => now - t < RESPAWN_WINDOW_MS);
  if (respawnTimestamps.length >= RESPAWN_MAX_ATTEMPTS) {
    console.error(
      `[chonks-mcp] backend crashed ${RESPAWN_MAX_ATTEMPTS}x within ${RESPAWN_WINDOW_MS / 1000}s; ` +
      `giving up on auto-respawn. Fix the backend (or restart this MCP server) to recover.`
    );
    return;
  }
  respawnTimestamps.push(now);
  const attempt = respawnTimestamps.length;
  const delay = RESPAWN_BASE_DELAY_MS * 2 ** (attempt - 1);
  console.error(`[chonks-mcp] backend exited unexpectedly; respawning in ${delay}ms (attempt ${attempt}/${RESPAWN_MAX_ATTEMPTS})`);
  respawnInFlight = true;
  setTimeout(() => {
    ensureBackend()
      .catch((e) => console.error(`[chonks-mcp] respawn attempt failed: ${(e as Error).message}`))
      .finally(() => {
        respawnInFlight = false;
      });
  }, delay);
}

async function ensureBackend(): Promise<void> {
  try {
    const status = await httpGet("/status", 1500);
    if (status?.ready !== false) return;
  } catch {
    // not reachable; fall through to spawn or fail below
  }

  if (!SERVER_PY) {
    throw new Error(
      `Chonks backend at ${SERVER_URL} is not reachable. ` +
      `Start it manually (\`chonks serve --db ...\` or \`uv run python -m chonks.server\`) ` +
      `or set CHONKS_SERVER_PY / CHONKS_DB to have this MCP server spawn it.`
    );
  }

  const [cmd, ...cmdRest] = PY_CMD.split(/\s+/);
  const args = [...cmdRest, SERVER_PY];
  if (DB_PATH) args.push("--db", DB_PATH);
  if (CONFIG_PATH) args.push("--config", CONFIG_PATH);

  // Must spawn with cwd = repo root, not the node process's cwd, or `uv run`
  // can't find pyproject.toml and runs in an env missing uvicorn/fastapi.
  const projectDir = dirname(dirname(SERVER_PY));
  console.error(`[chonks-mcp] spawning backend (cwd=${projectDir}): ${cmd} ${args.join(" ")}`);
  backend = spawn(cmd, args, {
    cwd: projectDir,
    stdio: ["ignore", "inherit", "inherit"],
    env: process.env,
  });
  backend.on("exit", (code, signal) => {
    console.error(`[chonks-mcp] backend exited code=${code} signal=${signal}`);
    backend = null;
    // We only get here on the spawn path (the external-backend case throws
    // before ever spawning), so SERVER_PY is always set, but check it
    // explicitly so this stays correct if that path ever changes.
    if (SERVER_PY) scheduleRespawn();
  });
  backend.on("error", (err) => {
    console.error(`[chonks-mcp] backend spawn error: ${err.message}`);
  });

  const deadline = Date.now() + 30_000;
  let lastErr = "";
  while (Date.now() < deadline) {
    try {
      const status = await httpGet("/status", 1000);
      if (status?.ready !== false) return;
    } catch (e) {
      lastErr = (e as Error).message;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`Backend did not become ready within 30s. Last error: ${lastErr}`);
}

function shutdownBackend(): void {
  if (!backend) return;
  try {
    backend.kill("SIGTERM");
  } catch (e) {
    console.error(`[chonks-mcp] failed to terminate backend: ${(e as Error).message}`);
  }
}

// ---------------------------------------------------------------------------
// Staleness header
// ---------------------------------------------------------------------------

const STATUS_CACHE_TTL_MS = 45_000;
let statusCache: { text: string | null; fetchedAt: number } | null = null;

function humanizeAge(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "unknown age";
  if (seconds < 90) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(seconds / 3600);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(seconds / 86_400);
  return `${days}d ago`;
}

/** Returns null on any failure; callers must degrade silently, no header at all. */
async function getStalenessHeader(): Promise<string | null> {
  const now = Date.now();
  if (statusCache && now - statusCache.fetchedAt < STATUS_CACHE_TTL_MS) {
    return statusCache.text;
  }
  let text: string | null = null;
  try {
    const s = await httpGet("/status", 5_000);
    if (s?.ready && typeof s.chunks === "number") {
      const chunks = s.chunks.toLocaleString("en-US");
      let age = "unknown age";
      if (typeof s.newest_indexed_at === "number") {
        age = humanizeAge(now / 1000 - s.newest_indexed_at);
      }
      text = `index: ${chunks} chunks · built ${age}`;
    }
  } catch {
    text = null;
  }
  statusCache = { text, fetchedAt: now };
  return text;
}

async function withStalenessHeader(text: string): Promise<string> {
  const header = await getStalenessHeader();
  return header ? `${header}\n\n${text}` : text;
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

type Chunk = {
  id: string;
  path: string;
  language?: string;
  chunk_type?: string;
  name?: string | null;
  start_line: number;
  end_line: number;
  content: string;
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
// chonks/research.py's _result_connections.
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

function formatChunks(chunks: Chunk[]): string {
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

function formatFilesLine(files: FileRank[], cap = 6): string {
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

// Must mirror chonks/searcher.py's rank_files ordering exactly (score DESC,
// first-appearance ASC tiebreak), since this recomputes it client-side for
// the merged multi-@subsystem case that has no single backend `files` field.
function rankFilesFromChunks(chunks: Chunk[]): FileRank[] {
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

// Must mirror chonks/repomap.py's PROVENANCE_BY_EDGE_TYPE exactly.
const EDGE_PROVENANCE: Record<string, string> = {
  calls: "extracted", imports: "extracted", inherits: "extracted", contains: "extracted",
  mentions: "inferred", associated: "inferred", semantic: "inferred", xlang: "paired",
};
function provenanceFor(edgeType?: string): string {
  if (!edgeType) return "inferred";
  return EDGE_PROVENANCE[edgeType] ?? "inferred";
}

// Research-only rendering: adds an evidence line formatChunks doesn't have.
function formatResearchChunks(chunks: Chunk[]): string {
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

// Returns "" (not null) when empty, so callers can append it directly.
function formatConnections(connections: Connection[] | undefined, chunks: Chunk[]): string {
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

// Only fires when chunk_kind was left at default; an explicit choice
// shouldn't get second-guessed.
function docsNudgeFooter(chunkKindArg: unknown, docsInResults: unknown, count: unknown): string {
  if (chunkKindArg !== undefined && chunkKindArg !== "any") return "";
  const n = typeof docsInResults === "number" ? docsInResults : 0;
  const m = typeof count === "number" ? count : 0;
  if (m === 0 || n / m <= 2 / 3) return "";
  return `\n\n${n} of ${m} results are documentation chunks — pass chunk_kind:"code" to filter.`;
}

// Nudges an agent stuck reformulating into the same near-duplicate wall.
// Threshold must mirror the server's NEAR_DUP_WALL_SHARE_THRESHOLD.
const NEAR_DUP_FIRING_THRESHOLD = 0.6;

function nearDupFooter(nearDup: unknown, count: unknown): string {
  if (!nearDup || typeof nearDup !== "object") return "";
  const wallShare = (nearDup as { wall_share?: unknown }).wall_share;
  const wallSize = (nearDup as { wall_size?: unknown }).wall_size;
  if (typeof wallShare !== "number" || wallShare < NEAR_DUP_FIRING_THRESHOLD) return "";
  const n = typeof wallSize === "number" ? wallSize : "most";
  const m = typeof count === "number" ? count : "the";
  return `\n\n${n} of ${m} results are near-duplicates of each other — try @subsystem scoping, ` +
    `a more specific query, or codebase_search with mode regex for idiom-shaped targets.`;
}

async function toolCodebaseSearch(args: any): Promise<string> {
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

async function toolCodebaseResearch(args: any): Promise<string> {
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

function formatEdgeTypeCounts(edgeTypes: Record<string, number> | undefined): string {
  return Object.entries(edgeTypes ?? {})
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}:${v}`)
    .join(" ");
}

function renderImpact(res: any): string {
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

function renderHubs(res: any): string {
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

async function toolCodebaseMap(args: any): Promise<string> {
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

async function toolFindSymbol(args: any): Promise<string> {
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

async function toolFindUsages(args: any): Promise<string> {
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

async function toolFindByMessage(args: any): Promise<string> {
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
async function toolInvestigate(args: any): Promise<string> {
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

// Unknown/future edge_type defaults to the weakest tier (0), matching the
// backend's edge_provenance "degrade to least-certain" convention.
const HOP_TIER: Record<string, number> = {
  calls: 3, imports: 3, inherits: 3,
  xlang: 2,
  associated: 1,
  mentions: 0, semantic: 0,
};

// Returns "" for an all-typed chain, else a line toolTracePath prepends.
function weakestLinkCaveat(hops: any[]): string {
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
function resolutionEcho(label: "from" | "to", requested: string, chunk: any): string {
  if (!chunk || !chunk.name || chunk.name === requested) return "";
  // Bare resolving to a qualified symbol's exact definition is not
  // containment (get_longest_axis -> AABB::get_longest_axis), so no echo.
  if (chunk.name.split(/::|\./).pop() === requested) return "";
  const loc = chunk.path ? `${chunk.path}:${chunk.start_line ?? "?"}` : "?";
  return `(${label} '${requested}' — resolved to containing chunk ${chunk.name}, ${loc})`;
}

async function toolTracePath(args: any): Promise<string> {
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

function _formatTraceChunk(c: any): string {
  if (!c) return "?";
  const loc = c.path ? `${c.path}:${c.start_line ?? "?"}` : "?";
  return c.name ? `${loc} ${c.name}` : loc;
}

async function toolCodebaseStatus(): Promise<string> {
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

// ---------------------------------------------------------------------------
// MCP server setup
// ---------------------------------------------------------------------------

// Prohibition-with-rationale phrasing measured 100% tool adoption vs 17%
// for recommendation-only (eval/results/F1/adoption_godot.json).
const INSTRUCTIONS = `Chonks is a local code-RAG index over this codebase, exposed as nine tools. Use it instead of filesystem search, not after it.

Never run Grep, Glob, find, or any filesystem enumeration against the repository root. This codebase is indexed precisely because it is too large to walk: root-scoped text searches are slow, noisy, and can time out. Start every code search with the Chonks tools. If they are deferred (present by name only), load them before your first search — one call: ToolSearch with query \"select:mcp__chonks__codebase_search,mcp__chonks__codebase_research,mcp__chonks__codebase_map,mcp__chonks__find_symbol,mcp__chonks__find_usages,mcp__chonks__investigate,mcp__chonks__trace_path,mcp__chonks__find_by_message,mcp__chonks__codebase_status\". Do not fall back to Grep because they need loading. Once a tool returns a concrete file path, refine with scoped Grep or Read on that path — retrieval points at code; the file is the source of truth.

Routing: codebase_map, codebase_search, and codebase_research are for orientation — finding the right region from a question. Once you hold a concrete symbol name, switch to the precise tools: find_symbol (definition sites), find_usages (callers and references), trace_path (how A reaches B), investigate (definition + callers + outgoing calls for that symbol, in one call with source inlined). When the input is a runtime message/log line/error string instead of a symbol name, use find_by_message — it matches through format holes (%s, {name}), which text search structurally cannot do since the concrete value never appears in source. Prefer these over text search: they resolve names folded into merged chunks, skip comment/string false positives, and (find_by_message) match cases grep cannot match at all.

Three bindings, because skipping them measurably produces wrong or wasteful routes:
- Orientation questions (where does X live, where do I start, how is Y organized): your FIRST call is codebase_map with NO path_prefix. It opens with a directory overview of the whole tree; take path_prefix values for follow-up calls from that overview. Never guess directory names from prior knowledge — guessed prefixes routinely name directories that do not exist in this repository.
- Dependency questions (what does X depend on, what uses X, what breaks if X changes): find_symbol alone is an incomplete answer. After locating the definition, call find_usages (who uses X), investigate (what X calls — the only tool with the outgoing direction), or trace_path — reference lists are indexed and cost one call; reading the definition file tells you what X calls, never who calls X. Weigh the returned provenance labels: extracted edges are AST facts, inferred edges are name co-occurrence.
- "Where does this message/log line/error come from" questions: do not grep for the concrete string — if it came from a format string (%s, f-string, {name}), grep on the concrete value will not find the source line that emitted it. Call find_by_message with the message instead; it is the only tool that matches through format holes.

Caveats: every response carries an index-staleness header — files changed since the build need a re-index. Docs and config are indexed as nameless text chunks: they surface in search and research (often the best answer to why-questions) but are invisible to the symbol and graph tools. trace_path hops over mentions/associated edges are name-occurrence evidence, not proven calls — trust calls/imports/inherits chains more. These tools only address the default project: if config.json defines multiple projects, query the others via the HTTP API directly, and note a project with a configured codebase root rejects (400) any request path outside it.`;

const TOOLS = [
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
      "classes, structs). Returns path:line definition sites. Boundary-only scope: " +
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
      "explaining why, and the above-cap case includes labeled FTS content-scan matches " +
      "(not graph edges) as a fallback. Edges carry a provenance label: extracted (AST " +
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

// HTTP mode builds one Server per request, so tool handlers must stay
// free of per-connection state. They are.
function buildServer(): Server {
  const server = new Server(
    { name: "chonks", version: "0.1.0" },
    { capabilities: { tools: {} }, instructions: INSTRUCTIONS },
  );
  server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  try {
    let text: string;
    switch (name) {
      case "codebase_search":
        text = await toolCodebaseSearch(args ?? {});
        break;
      case "codebase_research":
        text = await toolCodebaseResearch(args ?? {});
        break;
      case "codebase_map":
        text = await toolCodebaseMap(args ?? {});
        break;
      case "codebase_status":
        text = await toolCodebaseStatus();
        break;
      case "find_symbol":
        text = await toolFindSymbol(args ?? {});
        break;
      case "find_usages":
        text = await toolFindUsages(args ?? {});
        break;
      case "find_by_message":
        text = await toolFindByMessage(args ?? {});
        break;
      case "investigate":
        text = await toolInvestigate(args ?? {});
        break;
      case "trace_path":
        text = await toolTracePath(args ?? {});
        break;
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
    if (name !== "codebase_status") {
      text = await withStalenessHeader(text);
    }
    return { content: [{ type: "text", text }] };
  } catch (e) {
    const msg = (e as Error).message || String(e);
    return {
      content: [{ type: "text", text: `Error: ${msg}` }],
      isError: true,
    };
  }
  });
  return server;
}

// ---------------------------------------------------------------------------
// Shutdown handling
// ---------------------------------------------------------------------------

let shuttingDown = false;
function handleSignal(sig: NodeJS.Signals): void {
  if (shuttingDown) return;
  shuttingDown = true;
  console.error(`[chonks-mcp] received ${sig}, shutting down`);
  shutdownBackend();
  setTimeout(() => process.exit(0), 500);
}

process.on("SIGINT", handleSignal);
process.on("SIGTERM", handleSignal);
process.on("exit", () => shutdownBackend());

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  try {
    await ensureBackend();
  } catch (e) {
    console.error(`[chonks-mcp] ${(e as Error).message}`);
    // Continue anyway: individual tool calls will surface the error clearly,
    // which is more useful than a silent start failure.
  }
  if (HTTP_PORT !== null) {
    await startHttp(HTTP_PORT);
    return;
  }
  const transport = new StdioServerTransport();
  await buildServer().connect(transport);
  console.error(`[chonks-mcp] ready (backend: ${SERVER_URL})`);
}

// ---------------------------------------------------------------------------
// HTTP mode: no authentication on either hop. See DEPLOY.md for setup.
// ---------------------------------------------------------------------------

function parseHttpPort(): number | null {
  const i = process.argv.indexOf("--http");
  const raw = i >= 0 ? process.argv[i + 1] : process.env.CHONKS_MCP_HTTP_PORT;
  if (raw === undefined) return null;
  const n = Number(raw);
  if (!Number.isInteger(n) || n <= 0 || n > 65535) {
    throw new Error(`invalid --http port: ${raw}`);
  }
  return n;
}
const HTTP_PORT = parseHttpPort();
const HTTP_HOST = process.env.CHONKS_MCP_HOST ?? "127.0.0.1";
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);

// DNS-rebinding allowlist: loopback names always accepted, plus bind host
// and CHONKS_MCP_ALLOWED_HOSTS for LAN binds.
function buildAllowedHosts(port: number): string[] {
  const names = new Set<string>(["127.0.0.1", "localhost"]);
  if (HTTP_HOST !== "0.0.0.0") names.add(HTTP_HOST);
  const extra = (process.env.CHONKS_MCP_ALLOWED_HOSTS ?? "")
    .split(",").map((s) => s.trim()).filter(Boolean);
  for (const name of extra) names.add(name);
  const hosts: string[] = [];
  for (const name of names) {
    hosts.push(name, `${name}:${port}`);
  }
  return hosts;
}

async function startHttp(port: number): Promise<void> {
  if (!LOOPBACK_HOSTS.has(HTTP_HOST)) {
    console.error(
      `[chonks-mcp] WARNING: HTTP mode bound to ${HTTP_HOST} with no authentication. ` +
      "Anyone who can reach this port can query the index. Trusted networks only.");
    if (!process.env.CHONKS_MCP_ALLOWED_HOSTS) {
      console.error(
        "[chonks-mcp] WARNING: Host header validation only accepts loopback names. " +
        "Set CHONKS_MCP_ALLOWED_HOSTS to the hostname or IP clients will use, or requests will be rejected.");
    }
  }
  const allowedHosts = buildAllowedHosts(port);
  const httpServer = createHttpServer(async (req, res) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (url.pathname === "/healthz") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: true, backend: SERVER_URL }));
      return;
    }
    if (url.pathname !== "/mcp") {
      res.writeHead(404).end();
      return;
    }
    const server = buildServer();
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableDnsRebindingProtection: true,
      allowedHosts,
    });
    res.on("close", () => {
      transport.close().catch(() => {});
      server.close().catch(() => {});
    });
    try {
      await server.connect(transport);
      await transport.handleRequest(req, res);
    } catch (e) {
      console.error(`[chonks-mcp] request failed: ${(e as Error).message}`);
      if (!res.headersSent) {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: "internal error" }));
      }
    }
  });
  await new Promise<void>((resolve, reject) => {
    httpServer.once("error", reject);
    httpServer.listen(port, HTTP_HOST, () => resolve());
  });
  console.error(
    `[chonks-mcp] ready — HTTP mode on http://${HTTP_HOST}:${port}/mcp ` +
    `(backend: ${SERVER_URL}, no authentication)`);
}

main().catch((e) => {
  console.error(`[chonks-mcp] fatal: ${(e as Error).message}`);
  shutdownBackend();
  process.exit(1);
});
