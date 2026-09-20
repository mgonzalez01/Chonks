// Environment-variable configuration: backend URL, timeouts, and paths.

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export const SERVER_URL = (process.env.CHONKS_URL ?? "http://localhost:11438").replace(/\/$/, "");
export const RESEARCH_TIMEOUT_MS = Number(process.env.CHONKS_RESEARCH_TIMEOUT_MS ?? 900_000);
export const CONFIG_PATH = process.env.CHONKS_CONFIG ?? null;
export const SUBSYSTEMS_ENV = process.env.CHONKS_SUBSYSTEMS ?? null;

export const SERVER_PY = process.env.CHONKS_SERVER_PY ?? null;
export const DB_PATH = process.env.CHONKS_DB ?? null;
export const PY_CMD = (process.env.CHONKS_PY_CMD ?? "uv run python").trim();
