import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";

test("an invalid --http port exits with the fatal line, not a stack trace", () => {
  const r = spawnSync(process.execPath, ["dist/index.js", "--http", "abc"], {
    env: { ...process.env, CHONKS_URL: "http://127.0.0.1:1" },
    encoding: "utf8",
    timeout: 20000,
  });
  assert.equal(r.status, 1, r.stderr);
  assert.match(r.stderr, /\[chonks-mcp\] fatal: invalid --http port: abc/);
  assert.doesNotMatch(r.stderr, /\n\s+at /);
});
