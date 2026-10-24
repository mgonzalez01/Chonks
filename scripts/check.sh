#!/usr/bin/env bash
# The checks a change has to pass (CONTRIBUTING.md): formatting, lint, tests, and the MCP server build and tests.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --locked ruff format --check
uv run --locked ruff check
uv run --locked pytest -q
(cd mcp-server && npm ci && npm test)
echo "All checks passed."
