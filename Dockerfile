# syntax=docker/dockerfile:1.7
# Two images from one file:
#   docker build --target backend -t chonks-backend .   # Python API (chonks serve / index)
#   docker build --target mcp     -t chonks-mcp .       # MCP proxy in HTTP mode
# docker-compose.yml wires them; see DEPLOY.md.

# ---------------------------------------------------------------- backend --
FROM python:3.12-slim AS backend
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY chonks ./chonks
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
VOLUME ["/data"]
EXPOSE 11438
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:11438/status',timeout=2)" || exit 1
ENTRYPOINT ["chonks"]
CMD ["serve", "--host", "0.0.0.0", "--db", "/data/chonks.db", "--config", "/data/config.json"]

# -------------------------------------------------------------- mcp build --
FROM node:22-alpine AS mcp-build
WORKDIR /app
COPY mcp-server/package.json mcp-server/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY mcp-server/tsconfig.json ./
COPY mcp-server/src ./src
RUN npm run build

# -------------------------------------------------------------------- mcp --
FROM node:22-alpine AS mcp
WORKDIR /app
ENV NODE_ENV=production CHONKS_MCP_HTTP_PORT=11439 CHONKS_MCP_HOST=0.0.0.0
COPY mcp-server/package.json mcp-server/package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund
COPY --from=mcp-build /app/dist ./dist
EXPOSE 11439
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
  CMD wget -qO- http://127.0.0.1:11439/healthz >/dev/null || exit 1
CMD ["node", "dist/index.js"]
