"""eval/run.py: runs the frozen query set against the live API, recording each
query's target rank per mode. Gold resolves against the DB at run time, so
unresolved entries are reported, not silently dropped (that would lie about coverage).
"""
import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from gold import resolve_gold  # noqa: E402


def post(base, path, payload, timeout):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def rank_of(chunks, target_ids):
    """Best (smallest) rank at which ANY chunk in the relevant set appears."""
    targets = set(target_ids)
    for i, c in enumerate(chunks):
        if c.get("id") in targets:
            return i + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="frozen query set (queries.json)")
    ap.add_argument("--db", required=True, help="Chonks sqlite DB currently loaded by --api, "
                                                 "used to resolve each query's gold identity to chunk ids")
    ap.add_argument("--out", default="eval_data/results.json")
    ap.add_argument("--api", default="http://127.0.0.1:11438")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--modes", default="semantic,fts,hybrid")
    ap.add_argument("--research", action="store_true", help="also run /research per query")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    queries = json.load(open(args.queries))
    modes = [m for m in args.modes.split(",") if m]
    if args.research:
        modes.append("research")

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    # gold identity per query: prefer the portable `gold` key list; fall back
    # to legacy bare target_id/target_ids for queries.json files predating
    # the gold rekey (still resolved against the CURRENT db by id lookup).
    all_keys, key_ranges = [], []
    for q in queries:
        gold = q.get("gold")
        start = len(all_keys)
        if gold:
            all_keys.extend(gold)
        key_ranges.append((start, len(all_keys)))
    resolution = resolve_gold(con, all_keys) if all_keys else None
    if resolution is not None:
        print(resolution.report(all_keys))

    def gold_ids_for(i, q):
        if q.get("gold"):
            start, end = key_ranges[i]
            return resolution.all_ids(range(start, end))
        # legacy fallback: bare chunk ids, valid only if sampled from this same db
        return q.get("target_ids") or ([q["target_id"]] if q.get("target_id") else [])

    results = []
    for i, q in enumerate(queries):
        rec = {k: q.get(k) for k in ("query_id", "subsystem", "kind", "variant", "text")}
        tids = gold_ids_for(i, q)
        rec["gold_resolved"] = len(tids) > 0
        rec["per_mode"] = {}
        for mode in modes:
            try:
                if mode == "research":
                    chunks = post(args.api, "/research", {"query": q["text"]}, args.timeout)["chunks"]
                else:
                    chunks = post(args.api, "/search",
                                  {"query": q["text"], "mode": mode, "top_k": args.top_k}, args.timeout)["chunks"]
                rec["per_mode"][mode] = {
                    "rank": rank_of(chunks, tids),
                    "error": False,
                    "top_ids": [c["id"] for c in chunks[:args.top_k]],
                }
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                rec["per_mode"][mode] = {"rank": None, "error": True, "top_ids": []}
        results.append(rec)
        ranks = "  ".join(f"{m}={rec['per_mode'][m]['rank'] or ('ERR' if rec['per_mode'][m]['error'] else '—')}" for m in modes)
        print(f"  [{i+1}/{len(queries)}] {q['variant']:9} {ranks}   {q['text'][:46]!r}")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"wrote {len(results)} results -> {args.out}")


if __name__ == "__main__":
    main()
