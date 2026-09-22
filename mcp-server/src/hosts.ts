import { hostname, networkInterfaces } from "node:os";

const WILDCARD_BINDS = new Set(["0.0.0.0", "::"]);

function hostForm(name: string): string {
  return name.includes(":") && !name.startsWith("[") ? `[${name}]` : name;
}

// This machine's hostname (full and short) and interface addresses. Allowing
// them keeps DNS-rebinding protection intact: a rebinding page's Host header
// carries the attacker's domain, never this machine's own name or IP.
export function localMachineNames(): string[] {
  const full = hostname();
  const names = [full, full.split(".")[0]];
  for (const addrs of Object.values(networkInterfaces())) {
    for (const a of addrs ?? []) names.push(a.address);
  }
  return names;
}

// Host-header allowlist, lowercased: Node's URL parser lowercases hostnames,
// so every fetch-based MCP client sends them lowercase.
export function buildAllowedHosts(opts: {
  port: number;
  bindHost: string;
  extra: string[];
  machineNames: string[];
}): Set<string> {
  const names = ["127.0.0.1", "localhost", "::1", ...opts.machineNames, ...opts.extra];
  if (!WILDCARD_BINDS.has(opts.bindHost)) names.push(opts.bindHost);
  const hosts = new Set<string>();
  for (const raw of names) {
    const name = hostForm(raw.trim().toLowerCase());
    if (!name) continue;
    hosts.add(name);
    hosts.add(`${name}:${opts.port}`);
  }
  return hosts;
}

export function isAllowedHost(header: string | undefined, allowed: Set<string>): boolean {
  return header !== undefined && allowed.has(header.toLowerCase());
}
