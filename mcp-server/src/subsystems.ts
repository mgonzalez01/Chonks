// Subsystem map: loads @subsystem path prefixes from config and matches paths against them.

import { readFileSync, existsSync } from "node:fs";
import { CONFIG_PATH, SUBSYSTEMS_ENV } from "./config.js";

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

export const SUBSYSTEMS = loadSubsystems();

// ---------------------------------------------------------------------------
// Subsystem prefix parsing
// ---------------------------------------------------------------------------

/** paths is null when the query has no @subsystem_name prefix, or the name isn't a known subsystem. */
export function parseSubsystemPrefix(query: string): { paths: string[] | null; query: string; name: string | null } {
  const match = query.match(/^@([A-Za-z_][A-Za-z0-9_-]*)\s+(.+)$/s);
  if (!match) return { paths: null, query, name: null };
  const [, name, rest] = match;
  const paths = SUBSYSTEMS[name];
  if (!paths) return { paths: null, query, name: null };
  return { paths, query: rest, name };
}

/** Strips a leading @subsystem_name whether or not it's a known subsystem. */
export function stripSubsystemPrefix(query: string): string {
  const match = query.match(/^@[A-Za-z_][A-Za-z0-9_-]*\s+(.+)$/s);
  return match ? match[1] : query;
}

/** Reverse of parseSubsystemPrefix: first matching path prefix wins. */
export function matchSubsystemForPath(path: string): string | null {
  for (const [name, prefixes] of Object.entries(SUBSYSTEMS)) {
    for (const prefix of prefixes) {
      // Segment-boundary match: "src/eng" must not claim "src/engineering/".
      const p = prefix.endsWith("/") ? prefix : prefix + "/";
      if (path === prefix || path.startsWith(p)) return name;
    }
  }
  return null;
}
