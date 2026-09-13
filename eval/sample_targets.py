"""eval/sample_targets.py: samples stratified definition chunks as gold retrieval
targets: each target's gold set is itself plus same-name siblings (decl/def pairs),
so a hit on any of them counts. Deterministic since chunk ids are content-derived.
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from gold import stable_key  # noqa: E402

# tree-sitter chunk_type -> coarse kind we stratify on
KIND = {
    "function_definition": "function",
    "class_specifier":     "type",
    "struct_specifier":    "type",
    "class_definition":    "type",
    "template_declaration": "type",
}
TYPE_CTYPES = [ct for ct, k in KIND.items() if k == "type"]
ALL_DEF_CTYPES = list(KIND)          # every definition chunk_type (for same-name sets)
MAX_TARGET_SET = 6                   # names with more defs than this are too ambiguous
SUBSYSTEMS = ["core", "scene", "servers"]


def _pick_spread(rows, want):
    """Pick up to `want` rows spread across distinct files (one per file,
    deterministic stride) so a single big file can't dominate a cell."""
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["path"]].append(r)
    files = sorted(by_file)
    if not files:
        return []
    stride = max(1, len(files) // want)
    picked = []
    for i in range(0, len(files), stride):
        picked.append(by_file[files[i]][0])
        if len(picked) >= want:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Chonks sqlite DB to sample targets from")
    ap.add_argument("--out", default="eval_data/targets.json")
    ap.add_argument("--functions-per-subsystem", type=int, default=6)
    ap.add_argument("--types-per-subsystem", type=int, default=4)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-chars", type=int, default=5000)
    ap.add_argument("--language", default="cpp",
                    help="Restrict targets to this language. Default 'cpp'"
                         "this excludes the Python build scripts "
                         "(methods.py, *_builders.py) that live under core/.")
    ap.add_argument("--subsystems", default=",".join(SUBSYSTEMS),
                    help="Comma-separated top-level dirs to stratify by (matched "
                         "as a path prefix against each chunk's path). Default "
                         "matches this repo's Godot test corpus (%s) — pass your "
                         "own tree's top-level dirs for a BYO corpus." % ",".join(SUBSYSTEMS))
    args = ap.parse_args()
    subsystems = [s.strip() for s in args.subsystems.split(",") if s.strip()]

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    all_def_ph = ",".join("?" * len(ALL_DEF_CTYPES))
    cells = [
        ("function", args.functions_per_subsystem, ["function_definition"]),
        ("type",     args.types_per_subsystem,     TYPE_CTYPES),
    ]

    targets = []
    for sub in subsystems:
        for kind, want, ctypes in cells:
            ph = ",".join("?" * len(ctypes))
            rows = con.execute(
                f"""SELECT id, path, start_line, end_line, chunk_type, name, content
                    FROM chunks
                    WHERE name IS NOT NULL
                      AND language = ?
                      AND chunk_type IN ({ph})
                      AND length(content) BETWEEN ? AND ?
                      AND path LIKE ?
                    ORDER BY path, start_line""",
                (args.language, *ctypes, args.min_chars, args.max_chars, sub + "/%"),
            ).fetchall()
            for r in _pick_spread(rows, want):
                # Gold = all defs sharing this name (decl+def pairs); a hit on any
                # counts. Too many defs for one name means ambiguous: keep just this one.
                sibs = con.execute(
                    f"SELECT path, chunk_type, content FROM chunks "
                    f"WHERE name = ? AND language = ? AND chunk_type IN ({all_def_ph})",
                    (r["name"], args.language, *ALL_DEF_CTYPES),
                ).fetchall()
                gold = [stable_key(s["path"], r["name"], s["chunk_type"], s["content"])
                        for s in sibs] if 1 <= len(sibs) <= MAX_TARGET_SET else \
                       [stable_key(r["path"], r["name"], r["chunk_type"], r["content"])]
                targets.append({
                    "path": r["path"],
                    "start_line": r["start_line"], "end_line": r["end_line"],
                    "chunk_type": r["chunk_type"], "name": r["name"],
                    "subsystem": sub, "kind": kind,
                    "content": r["content"],
                    "gold": gold,
                })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(targets, f, indent=2)

    counts = Counter((t["subsystem"], t["kind"]) for t in targets)
    print(f"sampled {len(targets)} targets -> {args.out}")
    multi = sum(1 for t in targets if len(t["gold"]) > 1)
    print(f"  {multi} targets have >1 chunk in their relevant set (decl/def/sibling)")
    for (sub, kind), n in sorted(counts.items()):
        print(f"  {sub:8} {kind:9} {n}")
    print("\nexamples:")
    for t in targets[:12]:
        print(f"  [{t['subsystem']}/{t['kind']:8}] {t['name'][:36]:36} {t['path']}:{t['start_line']}")


if __name__ == "__main__":
    main()
