"""eval/token_footprint.py: measures whether Chonks saves lines read versus a
grep-and-read baseline, not just recall. Compares chonks (top-k set, or lines
up to the target) against baseline (file, or window), only over hits.
"""
import argparse
import json
import os
import sqlite3
import statistics
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from gold import resolve_gold  # noqa: E402

CHARS_PER_TOKEN = 4
WINDOW = 40


def post(base, path, payload, timeout=180):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def union_len(intervals):
    """Total integer points covered by a set of [start,end] line intervals."""
    if not intervals:
        return 0
    ivs = sorted(intervals)
    total = 0
    cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s <= ce + 1:
            ce = max(ce, e)
        else:
            total += ce - cs + 1
            cs, ce = s, e
    total += ce - cs + 1
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Chonks sqlite DB currently loaded by --api")
    ap.add_argument("--queries", required=True, help="frozen query set (queries.json)")
    ap.add_argument("--api", default="http://127.0.0.1:11438")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    # id -> (path, start, end); path -> [(start,end)]
    span = {}
    by_file = {}
    chars_per_line_num = chars_per_line_den = 0
    for r in con.execute("SELECT id, path, start_line, end_line, length(content) AS clen FROM chunks"):
        span[r["id"]] = (r["path"], r["start_line"], r["end_line"])
        by_file.setdefault(r["path"], []).append((r["start_line"], r["end_line"]))
        chars_per_line_num += r["clen"]
        chars_per_line_den += r["end_line"] - r["start_line"] + 1
    tok_per_line = (chars_per_line_num / chars_per_line_den) / CHARS_PER_TOKEN

    queries = json.load(open(args.queries))

    # Resolve each query's portable gold identity to chunk ids in THIS db.
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

    rows, misses, unresolved = [], 0, 0
    for i, q in enumerate(queries):
        tids = gold_ids_for(i, q)
        primary = next((t for t in tids if t in span), None)
        if primary is None:
            unresolved += 1
            continue
        tpath, tstart, tend = span[primary]

        chunks = post(args.api, "/search",
                      {"query": q["text"], "mode": "hybrid", "top_k": args.top_k})["chunks"]
        ret_ids = [c["id"] for c in chunks][:args.top_k]

        rank = next((i for i, cid in enumerate(ret_ids) if cid in set(tids)), None)
        if rank is None:
            misses += 1
            continue

        def spans_of(ids):
            by = {}
            for cid in ids:
                if cid in span:
                    p, s, e = span[cid]
                    by.setdefault(p, []).append((s, e))
            return sum(union_len(v) for v in by.values())

        chonks_10 = spans_of(ret_ids)
        chonks_tgt = spans_of(ret_ids[:rank + 1])
        base_file = union_len(by_file.get(tpath, []))
        base_window = (tend + WINDOW) - (tstart - WINDOW) + 1

        rows.append({"variant": q.get("variant", "?"),
                     "chonks_10": chonks_10, "chonks_tgt": chonks_tgt,
                     "base_file": base_file, "base_window": base_window})

    def med(key):
        return statistics.median(r[key] for r in rows)

    n = len(rows)
    print(f"=== context footprint (lines read; tokens ≈ lines × {tok_per_line:.1f}) ===")
    print(f"hits (target in top-{args.top_k}): {n}/{len(queries)}   misses (Chonks saved nothing): {misses}"
          f"   unresolved (gold didn't resolve against --db): {unresolved}\n")
    print(f"  {'':22}{'lines':>8}{'~tokens':>10}")
    for label, key in (("BASELINE  whole file", "base_file"),
                       ("BASELINE  ±%d window" % WINDOW, "base_window"),
                       ("CHONKS    top-%d set" % args.top_k, "chonks_10"),
                       ("CHONKS    to target", "chonks_tgt")):
        m = med(key)
        print(f"  {label:22}{m:8.0f}{m * tok_per_line:10.0f}")

    print("\n  savings ratio (median per-query baseline/chonks):")
    for blabel, bkey in (("whole file", "base_file"), ("±window", "base_window")):
        for clabel, ckey in (("top-k set", "chonks_10"), ("to target", "chonks_tgt")):
            ratios = [r[bkey] / r[ckey] for r in rows if r[ckey] > 0]
            cheaper = sum(1 for r in rows if r[bkey] > r[ckey])
            print(f"    baseline {blabel:10} / chonks {clabel:10}: "
                  f"{statistics.median(ratios):5.1f}x   (chonks cheaper in {cheaper}/{n})")

    print("\n  hit rate by variant:")
    for v in sorted(set(r["variant"] for r in rows) | {"verbatim", "paraphrase"}):
        tot = sum(1 for q in queries if q.get("variant") == v)
        hit = sum(1 for r in rows if r["variant"] == v)
        if tot:
            print(f"    {v:11} {hit}/{tot}")


if __name__ == "__main__":
    main()
