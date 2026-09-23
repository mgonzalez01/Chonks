import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { execFile } from "node:child_process";

// Fake backend: path_prefix "a" answers, "b" (and "c") fail with HTTP 500.
function fakeBackend() {
  return createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      const { path_prefix } = JSON.parse(body || "{}");
      if (path_prefix !== "a") {
        res.writeHead(500).end("boom");
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ count: 1, chunks: [{
        id: "c1", path: "a/x.cpp", name: "f", language: "cpp", chunk_type: "function_definition",
        start_line: 1, end_line: 2, content: "void f() {}", _score: 1,
      }] }));
    });
  });
}

function runSearch(port, subsystems) {
  const script = "import { toolCodebaseSearch } from './dist/tools.js';" +
    "try { console.log(await toolCodebaseSearch({ query: '@render shadows' })); }" +
    "catch (e) { console.log('THREW: ' + e.message); }";
  return new Promise((resolve, reject) => {
    execFile(process.execPath, ["--input-type=module", "-e", script], {
      env: { ...process.env, CHONKS_URL: `http://127.0.0.1:${port}`, CHONKS_SUBSYSTEMS: JSON.stringify(subsystems) },
      timeout: 20000,
    }, (err, stdout) => (err ? reject(err) : resolve(stdout)));
  });
}

test("a failed @subsystem branch is reported as partial", async () => {
  const server = fakeBackend();
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  try {
    const out = await runSearch(server.address().port, { render: ["a", "b"] });
    assert.match(out, /^PARTIAL: search failed for 1 of 2 paths of @render \(b: HTTP 500/);
    assert.match(out, /1 results/);
  } finally {
    server.close();
  }
});

test("every @subsystem branch failing is an error, not an empty result", async () => {
  const server = fakeBackend();
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  try {
    const out = await runSearch(server.address().port, { render: ["b", "c"] });
    assert.match(out, /^THREW: search failed for every path of @render/);
  } finally {
    server.close();
  }
});
