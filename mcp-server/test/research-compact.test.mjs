import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { execFile } from "node:child_process";

const CHUNK = {
  id: "ready",
  path: "scene/main/node.cpp",
  name: "Node::_ready",
  language: "cpp",
  start_line: 10,
  end_line: 20,
  _score: 0.5,
};
const SOURCE = "void Node::_ready() {}";

// Fake backend: records each /research body and leaves content out when asked to.
function fakeBackend(bodies) {
  return createServer((req, res) => {
    let raw = "";
    req.on("data", (d) => (raw += d));
    req.on("end", () => {
      const body = JSON.parse(raw);
      bodies.push(body);
      const chunk = body.compact ? CHUNK : { ...CHUNK, content: SOURCE };
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({
        chunks: [chunk], count: 1, iterations: 1,
        connections: [{ from_id: "ready", to_id: "ready", edge_type: "calls", provenance: "extracted" }],
        files: [{ path: CHUNK.path }], note: null, degraded: null,
      }));
    });
  });
}

function run(port, call) {
  const script = `import * as t from './dist/tools.js'; console.log(await t.${call});`;
  return new Promise((resolve, reject) => {
    execFile(process.execPath, ["--input-type=module", "-e", script], {
      env: { ...process.env, CHONKS_URL: `http://127.0.0.1:${port}` },
      timeout: 20000,
    }, (err, stdout) => (err ? reject(err) : resolve(stdout)));
  });
}

async function withBackend(fn) {
  const bodies = [];
  const server = fakeBackend(bodies);
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  try {
    await fn(server.address().port, bodies);
  } finally {
    server.close();
  }
}

test("compact research asks for compact chunks and renders one header per chunk", async () => {
  await withBackend(async (port, bodies) => {
    const out = await run(port, "toolCodebaseResearch({ query: 'how does a node become ready', compact: true })");
    assert.equal(bodies[0].compact, true);
    assert.match(out, /files: scene\/main\/node\.cpp\n\n\[1\] scene\/main\/node\.cpp:10-20 \(Node::_ready\)  score=0\.5000\n$/);
    assert.doesNotMatch(out, /```|CONNECTIONS/);
  });
});

test("research without compact keeps its request and renders source", async () => {
  await withBackend(async (port, bodies) => {
    const out = await run(port, "toolCodebaseResearch({ query: 'how does a node become ready' })");
    assert.equal("compact" in bodies[0], false);
    assert.ok(out.includes(`[1] scene/main/node.cpp:10-20 (Node::_ready)  score=0.5000\n\`\`\`cpp\n${SOURCE}\n\`\`\``), out);
    assert.match(out, /CONNECTIONS/);
  });
});
