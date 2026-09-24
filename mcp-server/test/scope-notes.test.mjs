import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { execFile } from "node:child_process";

const NOTE = "build/ is excluded from the index by config (exclude: build/), so nothing under it can match.";

// Fake backend: records every request URL and answers with an excluded-scope note.
function fakeBackend(urls) {
  return createServer((req, res) => {
    urls.push(req.url);
    req.resume();
    req.on("end", () => {
      res.writeHead(200, { "content-type": "application/json" });
      if (req.url.startsWith("/status")) {
        res.end(JSON.stringify({ ready: true, files: 1, chunks: 1, warnings: [NOTE] }));
      } else {
        res.end(JSON.stringify({ count: 0, chunks: [], note: NOTE }));
      }
    });
  });
}

function run(port, call) {
  const script = `import * as t from './dist/tools.js'; console.log(await t.${call});`;
  return new Promise((resolve, reject) => {
    execFile(process.execPath, ["--input-type=module", "-e", script], {
      env: {
        ...process.env,
        CHONKS_URL: `http://127.0.0.1:${port}`,
        CHONKS_SUBSYSTEMS: JSON.stringify({ build: ["build/"], tools: ["bin/tools/"] }),
      },
      timeout: 20000,
    }, (err, stdout) => (err ? reject(err) : resolve(stdout)));
  });
}

async function withBackend(fn) {
  const urls = [];
  const server = fakeBackend(urls);
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  try {
    await fn(server.address().port, urls);
  } finally {
    server.close();
  }
}

test("a search into an excluded scope leads with the backend's note", async () => {
  await withBackend(async (port) => {
    const out = await run(port, "toolCodebaseSearch({ query: '@build Renderer' })");
    assert.ok(out.startsWith(`note: ${NOTE}\n`), out);
  });
});

test("status sends the subsystem paths and prints the backend's warnings", async () => {
  await withBackend(async (port, urls) => {
    const out = await run(port, "toolCodebaseStatus()");
    assert.equal(urls[0], "/status?scope=build%2F&scope=bin%2Ftools%2F");
    assert.match(out, new RegExp(`warnings:\\n    ${NOTE.replace(/[.()/]/g, "\\$&")}`));
  });
});

test("research says it ignored an @subsystem prefix", async () => {
  await withBackend(async (port) => {
    const out = await run(port, "toolCodebaseResearch({ query: '@build how are frames scheduled' })");
    assert.match(out, /so @build was ignored/);
  });
});
