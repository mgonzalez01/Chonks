#!/usr/bin/env node
// Starts the built MCP server (mcp-server/dist/index.js) over stdio, asks it
// for tools/list, and checks the answer against a recorded baseline.
//
// Run:   cd mcp-server && npm run build   (once, or after any src/ change)
//        node scripts/mcp-smoke.mjs
// `--record` rewrites scripts/mcp-smoke-baseline.json from the current build
// instead of checking against it.
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const entryPath = join(here, "..", "mcp-server", "dist", "index.js");
const baselinePath = join(here, "mcp-smoke-baseline.json");
const record = process.argv.includes("--record");

if (!existsSync(entryPath)) {
  console.error("FAIL: mcp-server/dist/index.js is missing; run `cd mcp-server && npm run build` first");
  process.exit(1);
}

// A whitelist, not the inherited environment: CHONKS_SERVER_PY would spawn a
// backend, CHONKS_CONFIG would change a tool description, and
// CHONKS_MCP_HTTP_PORT would start an HTTP listener instead of stdio.
const env = {
  PATH: process.env.PATH,
  HOME: process.env.HOME,
  CHONKS_URL: "http://127.0.0.1:9",
  CHONKS_SUBSYSTEMS: "{}",
};

const child = spawn(process.execPath, [entryPath], { env, stdio: ["pipe", "pipe", "pipe"] });

let killed = false;
function kill() {
  if (!killed) {
    killed = true;
    child.kill("SIGTERM");
  }
}

let buf = "";
const pending = new Map();
child.stdout.on("data", (chunk) => {
  buf += chunk;
  let i;
  while ((i = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    if (!line) continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      fail(`the server wrote a line to stdout that is not JSON: ${line.slice(0, 80)}`);
    }
    if (msg.id != null && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
    }
  }
});

function send(id, method, params) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`timeout on ${method}`));
    }, 20000);
    pending.set(id, (msg) => {
      clearTimeout(timer);
      resolve(msg);
    });
    child.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
  });
}

function notify(method, params) {
  child.stdin.write(JSON.stringify({ jsonrpc: "2.0", method, params }) + "\n");
}

function fail(message) {
  kill();
  console.error(`FAIL: ${message}`);
  process.exit(1);
}

function sortKeys(value) {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value && typeof value === "object") {
    const out = {};
    for (const key of Object.keys(value).sort()) out[key] = sortKeys(value[key]);
    return out;
  }
  return value;
}

function canonicalStringify(value) {
  return JSON.stringify(sortKeys(value), null, 2);
}

// Returns the first JSON path at which a and b differ, or null if they match.
function firstDiffPath(a, b, path) {
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b)) return path || "(root)";
    if (a.length !== b.length) return `${path}.length`;
    for (let i = 0; i < a.length; i++) {
      const diff = firstDiffPath(a[i], b[i], `${path}[${i}]`);
      if (diff) return diff;
    }
    return null;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])].sort();
    for (const key of keys) {
      const next = path ? `${path}.${key}` : key;
      if (!(key in a) || !(key in b)) return next;
      const diff = firstDiffPath(a[key], b[key], next);
      if (diff) return diff;
    }
    return null;
  }
  return a === b ? null : path || "(root)";
}

let init, list;
try {
  init = await send(1, "initialize", {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: { name: "mcp-smoke", version: "0" },
  });
  notify("notifications/initialized", {});
  list = await send(2, "tools/list", {});
} catch (err) {
  fail(err.message);
}
kill();

const tools = list.result.tools;
const actual = {
  serverInfo: init.result.serverInfo,
  instructions: init.result.instructions,
  tools: tools.map((t) => ({ name: t.name, description: t.description, inputSchema: t.inputSchema })),
};

if (tools.length !== 9) {
  fail(`${tools.length} tools, expected 9`);
}

if (record) {
  writeFileSync(baselinePath, canonicalStringify(actual) + "\n");
  console.log(`ok: recorded ${baselinePath}`);
  process.exit(0);
}

const expected = JSON.parse(readFileSync(baselinePath, "utf8"));
const diffPath = firstDiffPath(sortKeys(actual), sortKeys(expected), "");
if (diffPath) {
  fail(`tools/list drifted from baseline at ${diffPath}`);
}

console.log("ok: 9 tools, contract matches baseline");
