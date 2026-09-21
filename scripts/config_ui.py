#!/usr/bin/env python3
"""Tiny localhost-only web UI for editing a Chonks deployment's config.

    uv run python scripts/config_ui.py          # http://127.0.0.1:11440

Edits .env and config.json in the repo root, atomically, with a .bak backup
each save. Binds 127.0.0.1 only: an operator convenience, not a service.
Changes need a restart of whatever consumes them to take effect.
"""
from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
ENV_EXAMPLE_PATH = ROOT / ".env.example"
CONFIG_PATH = ROOT / "config.json"
PORT = int(os.environ.get("CHONKS_CONFIG_UI_PORT", "11440"))


def parse_env_example() -> list[tuple[str, str, str]]:
    """(key, default, comment) rows from .env.example, the schema source."""
    rows = []
    if not ENV_EXAMPLE_PATH.exists():
        return rows
    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)=([^#]*)(?:#\s*(.*))?$", line)
        if m:
            rows.append((m.group(1), m.group(2).strip(), (m.group(3) or "").strip()))
    return rows


def parse_env(path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line)
            if m:
                vals[m.group(1)] = m.group(2)
    return vals


def atomic_write(path: Path, content: str) -> None:
    if path.exists():
        path.with_suffix(path.suffix + ".bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Chonks deployment config</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; color:#222; }}
 h1 {{ font-size: 1.3rem; }} h2 {{ font-size: 1.05rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
 label {{ display:block; margin-top: .8rem; font-weight: 600; }}
 .doc {{ font-weight: 400; color: #666; font-size: .85rem; }}
 input[type=text] {{ width: 100%; padding: .35rem; font-family: ui-monospace, monospace; box-sizing: border-box; }}
 textarea {{ width: 100%; height: 320px; font-family: ui-monospace, monospace; font-size: .85rem; box-sizing: border-box; }}
 button {{ margin-top: 1rem; padding: .5rem 1.2rem; font-size: 1rem; }}
 .msg {{ padding: .6rem .9rem; border-radius: 6px; margin: 1rem 0; }}
 .ok {{ background: #e6f4e6; border: 1px solid #9c9; }}
 .err {{ background: #fae6e6; border: 1px solid #c99; white-space: pre-wrap; }}
 .note {{ color:#666; font-size:.85rem; }}
</style></head><body>
<h1>Chonks deployment config</h1>
<p class="note">Edits <code>.env</code> (docker compose) and <code>config.json</code> in
<code>{root}</code>. Saves are atomic with a <code>.bak</code>. Changes apply on the next
restart: <code>docker compose up -d</code> or re-running <code>host.ps1</code>.</p>
{message}
<form method="post" action="/save-env"><h2>.env — docker compose settings</h2>
{env_fields}
<button>Save .env</button></form>
<form method="post" action="/save-config"><h2>config.json — Chonks settings</h2>
<p class="doc">Raw JSON; validated before writing (a parse error writes nothing).</p>
<textarea name="config_json">{config_json}</textarea>
<button>Save config.json</button></form>
</body></html>"""


def render(message: str = "") -> str:
    schema = parse_env_example()
    current = parse_env(ENV_PATH)
    fields = []
    seen = set()
    for key, default, doc in schema:
        seen.add(key)
        val = current.get(key, default)
        fields.append(
            f'<label>{key} <span class="doc">{html.escape(doc)}</span></label>'
            f'<input type="text" name="{key}" value="{html.escape(val)}">')
    for key, val in current.items():  # keys present in .env but not the example
        if key not in seen:
            fields.append(
                f'<label>{key} <span class="doc">(not in .env.example)</span></label>'
                f'<input type="text" name="{key}" value="{html.escape(val)}">')
    cfg = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else "{\n}\n"
    return PAGE.format(root=html.escape(str(ROOT)), message=message,
                       env_fields="\n".join(fields), config_json=html.escape(cfg))


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: str, status: int = 200) -> None:
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/":
            self._send("not found", 404)
            return
        self._send(render())

    def _allowed_host(self, value: str) -> bool:
        return value in (f"127.0.0.1:{PORT}", f"localhost:{PORT}")

    def _origin_ok(self) -> bool:
        # Reject cross-site form POSTs and DNS-rebinding requests: only accept
        # this exact loopback origin/host, since the form has no CSRF token.
        origin = self.headers.get("Origin")
        if origin is not None:
            if origin not in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"):
                return False
        host = self.headers.get("Host")
        if not host or not self._allowed_host(host):
            return False
        return True

    def do_POST(self) -> None:  # noqa: N802
        if not self._origin_ok():
            self._send("forbidden", 403)
            return
        length = int(self.headers.get("Content-Length", "0"))
        form = urllib.parse.parse_qs(self.rfile.read(length).decode(), keep_blank_values=True)
        if self.path == "/save-env":
            bad = []
            for k, v in form.items():
                if not re.match(r"^[A-Z][A-Z0-9_]*$", k):
                    bad.append(f"invalid key: {k}")
                elif "\n" in v[0] or "\r" in v[0]:
                    bad.append(f"invalid value for {k}: contains a newline")
            if bad:
                self._send(render('<div class="msg err">.env NOT saved — ' +
                                  html.escape("; ".join(bad)) + "</div>"), 400)
                return
            lines = [f"{k}={v[0]}" for k, v in form.items()]
            atomic_write(ENV_PATH, "\n".join(lines) + "\n")
            self._send(render('<div class="msg ok">.env saved. Apply with: '
                              "<code>docker compose up -d</code> (or restart host.ps1).</div>"))
        elif self.path == "/save-config":
            raw = form.get("config_json", [""])[0]
            try:
                json.loads(raw)
            except json.JSONDecodeError as e:
                self._send(render(f'<div class="msg err">config.json NOT saved — invalid JSON:\n'
                                  f"{html.escape(str(e))}</div>"))
                return
            atomic_write(CONFIG_PATH, raw if raw.endswith("\n") else raw + "\n")
            self._send(render('<div class="msg ok">config.json saved. Apply by restarting '
                              "the backend (host.ps1 / docker compose restart backend). "
                              "A changed <code>embed_model</code> needs a --force re-index.</div>"))
        else:
            self._send("not found", 404)

    def log_message(self, fmt: str, *args) -> None:  # quiet
        pass


def main() -> None:
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Chonks config UI: http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
