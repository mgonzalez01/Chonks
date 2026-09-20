"""eval/gold.py: keys gold targets by path+symbol (content-hash tiebreak), not
chunk id, since chunk ids shift on every re-index. Resolution never drops a key:
callers must report unresolved/ambiguous, not swallow them.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field


def content_hash(content: str) -> str:
    """Same style as chonks/index/rows.py's chunk id hash, but content-only so it is
    invariant to start_line drift and path (path is matched separately)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def stable_key(path: str, name: str | None, chunk_type: str | None, content: str) -> dict:
    """Builds the portable identity for a chunk. The hash is the match key;
    content itself isn't needed here (callers may still stash it separately)."""
    return {"path": path, "name": name, "chunk_type": chunk_type,
            "content_hash": content_hash(content)}


@dataclass
class Resolution:
    resolved: dict = field(default_factory=dict)   # key index -> [chunk_id, ...] (len 1)
    ambiguous: dict = field(default_factory=dict)  # key index -> [chunk_id, ...] (len > 1)
    unresolved: list = field(default_factory=list) # key indices with 0 matches

    def ids_for(self, i: int) -> list[str]:
        return self.resolved.get(i) or self.ambiguous.get(i) or []

    def all_ids(self, indices) -> list[str]:
        """Union of resolved/ambiguous ids for the given key indices, deduped and
        order-preserving."""
        seen, out = set(), []
        for i in indices:
            for cid in self.ids_for(i):
                if cid not in seen:
                    seen.add(cid)
                    out.append(cid)
        return out

    def report(self, keys: list[dict]) -> str:
        n = len(keys)
        lines = [f"gold resolution: {len(self.resolved)}/{n} resolved, "
                 f"{len(self.ambiguous)}/{n} ambiguous, "
                 f"{len(self.unresolved)}/{n} UNRESOLVED"]
        for i in self.unresolved:
            k = keys[i]
            lines.append(f"    UNRESOLVED: {k.get('path')}::{k.get('name')} "
                          f"({k.get('chunk_type')}, content_hash={k.get('content_hash')})")
        return "\n".join(lines)


def resolve_gold(con: sqlite3.Connection, keys: list[dict]) -> Resolution:
    """Resolves gold keys to chunk ids: by path+name, narrowed by chunk_type if
    that helps, then by content_hash if still ambiguous. Multiple survivors stay
    ambiguous, not dropped, since a decl/def sibling pair is still a valid hit.
    """
    con.row_factory = sqlite3.Row
    res = Resolution()
    for i, k in enumerate(keys):
        if k.get("name"):
            rows = con.execute(
                "SELECT id, chunk_type, content FROM chunks WHERE path=? AND name=?",
                (k["path"], k["name"]),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT id, chunk_type, content FROM chunks WHERE path=?",
                (k["path"],),
            ).fetchall()

        if k.get("chunk_type") and len(rows) > 1:
            narrowed = [r for r in rows if r["chunk_type"] == k["chunk_type"]]
            if narrowed:
                rows = narrowed

        if not rows:
            res.unresolved.append(i)
            continue
        if len(rows) == 1:
            res.resolved[i] = [rows[0]["id"]]
            continue

        want = k.get("content_hash")
        by_hash = [r for r in rows if want and content_hash(r["content"]) == want]
        if len(by_hash) == 1:
            res.resolved[i] = [by_hash[0]["id"]]
        else:
            res.ambiguous[i] = [r["id"] for r in rows]
    return res
