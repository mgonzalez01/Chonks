"""eval/relational_sample.py: samples gold sets of an anchor's real cross-subsystem
callees (excluding generic hubs like Variant/RefCounted) to test whether research's
graph expansion reaches them where hybrid text search misses them. No RNG: hub
pruning is a fixed in-degree cap, anchors are stride-picked across files.
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from gold import stable_key  # noqa: E402

SUBSYSTEMS = ["core", "scene", "servers"]
DEF_CTYPES = ("function_definition", "class_specifier", "struct_specifier",
              "class_definition", "template_declaration")


def subsystem_of(path: str, subsystems=SUBSYSTEMS) -> str | None:
    """Longest matching prefix from `subsystems`, or None. Longest-prefix-wins so a
    nested value like "src/gfx" isn't shadowed by its parent "src"."""
    best = None
    for s in subsystems:
        if path.startswith(s + "/") and (best is None or len(s) > len(best)):
            best = s
    return best


def clean_name(n: str | None) -> bool:
    return bool(n) and n[0].isalpha() and len(n) >= 3 and "operator" not in n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Chonks sqlite DB to sample targets from")
    ap.add_argument("--out", default="eval_data/relational_targets.json")
    ap.add_argument("--per-subsystem", type=int, default=8)
    ap.add_argument("--min-neighbours", type=int, default=3,
                    help="anchor must have at least this many cross-subsystem, "
                         "non-hub callees to be a real interaction")
    ap.add_argument("--hub-indegree-max", type=int, default=50,
                    help="exclude callees referenced by more than this many chunks "
                         "(generic infrastructure, not an A-specific interaction)")
    ap.add_argument("--language", default="cpp")
    ap.add_argument("--subsystems", default=",".join(SUBSYSTEMS),
                    help="Comma-separated path prefixes to stratify by (longest "
                         "prefix wins; nested values like src/gfx are fine — use "
                         "them when your tree has a single top-level root). Default "
                         "matches this repo's Godot test corpus (%s)." % ",".join(SUBSYSTEMS))
    args = ap.parse_args()
    subsystems = [s.strip() for s in args.subsystems.split(",") if s.strip()]

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    meta: dict[str, dict] = {}
    for r in con.execute("SELECT id, path, name, chunk_type, language, content FROM chunks WHERE name IS NOT NULL"):
        meta[r["id"]] = {"path": r["path"], "name": r["name"],
                         "ctype": r["chunk_type"], "lang": r["language"],
                         "content": r["content"]}

    out_edges: dict[str, set] = defaultdict(set)   # from_id -> {to_id}
    indeg: Counter = Counter()                      # to_id -> #refs in
    for r in con.execute("SELECT from_id, to_id FROM chunk_refs"):
        out_edges[r["from_id"]].add(r["to_id"])
        indeg[r["to_id"]] += 1

    def neighbourhood(a_id: str) -> list[str]:
        """A's real dependency set: definition callees only, hubs excluded, deduped
        by name so a name collision across files doesn't inflate the count."""
        a_name = meta[a_id]["name"]
        by_name: dict[str, str] = {}
        for c in out_edges.get(a_id, ()):
            m = meta.get(c)
            if (m and m["ctype"] in DEF_CTYPES
                    and indeg[c] <= args.hub_indegree_max
                    and clean_name(m["name"]) and m["name"] != a_name):
                by_name.setdefault(m["name"], c)   # first id per distinct name
        return list(by_name.values())

    targets = []
    for s1 in subsystems:
        cands = sorted(n for n in out_edges
                       if n in meta and meta[n]["lang"] == args.language
                       and meta[n]["ctype"] in DEF_CTYPES
                       and subsystem_of(meta[n]["path"], subsystems) == s1
                       and clean_name(meta[n]["name"]))
        stride = max(1, len(cands) // (args.per_subsystem * 12))
        picked, used_files = 0, set()
        for A in cands[::stride]:
            if picked >= args.per_subsystem:
                break
            if meta[A]["path"] in used_files:
                continue
            gold = neighbourhood(A)
            if len(gold) < args.min_neighbours:
                continue
            dom = Counter(subsystem_of(meta[c]["path"], subsystems) for c in gold).most_common(1)[0][0]
            used_files.add(meta[A]["path"])
            picked += 1
            targets.append({
                "a_key": stable_key(meta[A]["path"], meta[A]["name"], meta[A]["ctype"], meta[A]["content"]),
                "a_name": meta[A]["name"], "s1": s1,
                "neighbour_subsystem": dom,
                "gold": [stable_key(meta[c]["path"], meta[c]["name"], meta[c]["ctype"], meta[c]["content"])
                         for c in gold],
                "gold_names": [meta[c]["name"] for c in gold],
                "query": f"How does {meta[A]['name']} work and what does it depend on?",
            })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(targets, f, indent=2)

    print(f"sampled {len(targets)} relational targets -> {args.out}")
    for t in targets:
        print(f"  [{t['s1']}->{t['neighbour_subsystem']}] {t['a_name']}  "
              f"(+{len(t['gold'])}): {', '.join(t['gold_names'][:5])}"
              + (" ..." if len(t['gold']) > 5 else ""))


if __name__ == "__main__":
    main()
