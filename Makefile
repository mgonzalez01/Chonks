.PHONY: setup init server mcp-build status host

setup:
	uv sync
	cd mcp-server && npm install && npm run build

init:
	uv run chonks init

server:
	uv run chonks serve --db .db/chonks.db --config config.json

mcp-build:
	cd mcp-server && npm run build

status:
	curl -s http://localhost:11438/status | python3 -m json.tool

host:
	scripts/host.sh --db .db/chonks.db --config config.json
