#!/usr/bin/env bash
# macOS/Linux twin of scripts/host.ps1: serve an existing Chonks DB to the LAN
# as an HTTP MCP endpoint. Other machines run one `claude mcp add` and install nothing.
#
#   scripts/host.sh --db .db/chonks.db [--config config.json] [--port 11439] [--bind 0.0.0.0]
#                  [--llama-server /path/llama-server --embed-model jinaai/jina-code-embeddings-0.5b-GGUF]
#
# Default --bind is 0.0.0.0 with no authentication: trusted networks only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
DB=".db/chonks.db"; CONFIG="config.json"; PORT=11439; BIND="0.0.0.0"
LLAMA=""; EMBED_MODEL=""; EMBED_PORT=11437; LLAMA_ARGS="${LLAMA_ARGS:---parallel 8 --ctx-size 32768 -ngl 99}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --db) DB="$2"; shift 2;; --config) CONFIG="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --bind) BIND="$2"; shift 2;;
    --llama-server) LLAMA="$2"; shift 2;; --embed-model) EMBED_MODEL="$2"; shift 2;; --embed-port) EMBED_PORT="$2"; shift 2;; *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
for t in node uv; do command -v "$t" >/dev/null || { echo "$t not found on PATH" >&2; exit 1; }; done
[[ -f "$DB" ]] || { echo "DB not found: $DB — index first" >&2; exit 1; }
DB="$(cd "$(dirname "$DB")" && pwd)/$(basename "$DB")"
# Rebuild when missing or older than its sources, so a pulled fix is not served from a stale build.
if [[ ! -f mcp-server/dist/index.js || -n "$(find mcp-server/src mcp-server/package-lock.json -newer mcp-server/dist/index.js | head -1)" ]]; then
  (cd mcp-server && npm install --no-audit --no-fund >/dev/null && npm run build >/dev/null)
fi
export CHONKS_MCP_HTTP_PORT="$PORT" CHONKS_MCP_HOST="$BIND"
export CHONKS_URL="http://127.0.0.1:11438" CHONKS_SERVER_PY="$ROOT/chonks/server.py" CHONKS_DB="$DB" CHONKS_PY_CMD="uv run python"
[[ -f "$CONFIG" ]] && export CHONKS_CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
embedder_up() { curl -sf -m 2 "http://127.0.0.1:$EMBED_PORT/health" >/dev/null; }
LLAMA_PID=""
if [[ -n "$LLAMA" ]]; then
  [[ -n "$EMBED_MODEL" ]] || { echo "--embed-model required with --llama-server" >&2; exit 2; }
  if [[ -f "$EMBED_MODEL" ]]; then MODEL_ARG=(-m "$EMBED_MODEL"); else MODEL_ARG=(--hf-repo "$EMBED_MODEL"); fi
  echo "[host] starting embedder: $LLAMA ${MODEL_ARG[*]} --embedding --pooling last --port $EMBED_PORT $LLAMA_ARGS"
  # shellcheck disable=SC2086
  "$LLAMA" "${MODEL_ARG[@]}" --embedding --pooling last --port "$EMBED_PORT" --host 127.0.0.1 $LLAMA_ARGS &
  LLAMA_PID=$!
  trap '[[ -n "$LLAMA_PID" ]] && kill "$LLAMA_PID" 2>/dev/null' EXIT
  for _ in $(seq 1 150); do embedder_up && break; kill -0 "$LLAMA_PID" 2>/dev/null || { echo "llama-server exited" >&2; exit 1; }; sleep 2; done
  embedder_up || { echo "embedder did not answer within 5 min" >&2; exit 1; }
  echo "[host] embedder ready on :$EMBED_PORT"
elif ! embedder_up; then
  echo "[host] WARNING: no embedder on http://127.0.0.1:$EMBED_PORT — semantic/hybrid queries will fail (fts/regex/graph still work)" >&2
fi
H="$(hostname)"
echo; echo "[host] Chonks MCP (HTTP) on http://$H:$PORT/mcp"; echo "[host] other machines run:"
echo "  claude mcp add --transport http chonks http://$H:$PORT/mcp"
echo "[host] no authentication: anyone who can reach port $PORT can query the index. Trusted network only."; echo
node mcp-server/dist/index.js
