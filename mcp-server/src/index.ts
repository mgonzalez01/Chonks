#!/usr/bin/env node
/**
 * chonks-mcp: MCP server exposing the Chonks HTTP backend as tools. Config env vars are documented in DEPLOY.md.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createServer as createHttpServer } from "node:http";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { SERVER_URL } from "./config.js";
import { ensureBackend, shutdownBackend } from "./backend.js";
import { withStalenessHeader } from "./client.js";
import { INSTRUCTIONS, TOOLS } from "./contract.js";
import {
  toolCodebaseSearch,
  toolCodebaseResearch,
  toolCodebaseMap,
  toolCodebaseStatus,
  toolFindSymbol,
  toolFindUsages,
  toolFindByMessage,
  toolInvestigate,
  toolTracePath,
} from "./tools.js";

// ---------------------------------------------------------------------------
// MCP server setup
// ---------------------------------------------------------------------------

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
