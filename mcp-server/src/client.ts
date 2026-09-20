// HTTP client for the Chonks backend, plus the cached index-staleness header.

import { SERVER_URL } from "./config.js";

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

export async function httpPost(path: string, body: unknown, timeoutMs = 60_000): Promise<any> {
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

export async function httpGet(path: string, timeoutMs = 10_000): Promise<any> {
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

export async function withStalenessHeader(text: string): Promise<string> {
  const header = await getStalenessHeader();
  return header ? `${header}\n\n${text}` : text;
}
