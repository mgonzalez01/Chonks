import { test } from "node:test";
import assert from "node:assert/strict";
import { partialSearchNote } from "../dist/fanout.js";

test("no failures, no note", () => {
  assert.equal(partialSearchNote("render", 3, []), "");
});

test("a failed branch is named and the result is marked partial", () => {
  const note = partialSearchNote("render", 3, [{ path: "servers/rendering", message: "HTTP 500" }]);
  assert.match(note, /^PARTIAL:/);
  assert.match(note, /1 of 3/);
  assert.match(note, /servers\/rendering: HTTP 500/);
  assert.match(note, /@render/);
});
