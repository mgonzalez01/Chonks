import { test } from "node:test";
import assert from "node:assert/strict";
import { buildAllowedHosts, isAllowedHost } from "../dist/hosts.js";

const allowed = buildAllowedHosts({
  port: 11439,
  bindHost: "0.0.0.0",
  extra: ["Build-Box", " ", ""],
  machineNames: ["DESKTOP-ABC", "192.168.1.83", "fe80::1"],
});

test("hostname matches regardless of case, with or without port", () => {
  for (const h of ["desktop-abc:11439", "DESKTOP-ABC:11439", "desktop-abc", "build-box:11439"]) {
    assert.ok(isAllowedHost(h, allowed), h);
  }
});

test("machine IPs and loopback names are allowed", () => {
  for (const h of ["192.168.1.83:11439", "[fe80::1]:11439", "127.0.0.1:11439", "localhost:11439", "[::1]:11439"]) {
    assert.ok(isAllowedHost(h, allowed), h);
  }
});

test("foreign, missing and wrong-port hosts are rejected", () => {
  for (const h of ["evil.example:11439", "desktop-abc:80", "", undefined]) {
    assert.ok(!isAllowedHost(h, allowed), String(h));
  }
});

test("a wildcard bind adds nothing; a specific bind host is allowed", () => {
  assert.ok(!isAllowedHost("0.0.0.0:11439", allowed));
  const bound = buildAllowedHosts({ port: 11439, bindHost: "10.0.0.5", extra: [], machineNames: [] });
  assert.ok(isAllowedHost("10.0.0.5:11439", bound));
});
