// Backend process lifecycle: spawn, respawn on crash, and shutdown.

import { spawn, ChildProcess } from "node:child_process";
import { dirname } from "node:path";
import { SERVER_URL, SERVER_PY, DB_PATH, CONFIG_PATH, PY_CMD } from "./config.js";
import { httpGet } from "./client.js";

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

export async function ensureBackend(): Promise<void> {
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

export function shutdownBackend(): void {
  if (!backend) return;
  try {
    backend.kill("SIGTERM");
  } catch (e) {
    console.error(`[chonks-mcp] failed to terminate backend: ${(e as Error).message}`);
  }
}
