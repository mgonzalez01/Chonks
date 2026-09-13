"""eval/relational_run.py: scores research vs hybrid on gold-set coverage from
relational_sample.py. R@20 is what an agent would see, R@50 is what research's
expansion reached; a gap means results are ranked too low, not unreached.
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

KS = (10, 20, 50)


def post(base, path, payload, timeout):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def recall_at(ids, gold, k):
    top = set(ids[:k])
    return len(top & gold) / len(gold) if gold else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="relational_targets.json")
    ap.add_argument("--db", required=True, help="Chonks sqlite DB currently loaded by --api, "
                                                 "used to resolve gold identities to chunk ids")
    ap.add_argument("--out", default="eval_data/relational_results.json")
    ap.add_argument("--api", default="http://127.0.0.1:11438")
    ap.add_argument("--fetch-k", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--research-extra", default=None,
                    help="JSON object merged into every /research request body, e.g. "
                         "'{\"edge_type_weights\": {\"mentions\": 0.0}}' (edge-type ablations). "
                         "Recorded in the output file as research_extra.")
    args = ap.parse_args()

    targets = json.load(open(args.targets))
    research_extra = json.loads(args.research_extra) if args.research_extra else {}
    if not isinstance(research_extra, dict):
        sys.exit("--research-extra must be a JSON object")
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    # Resolve every a_key + gold key, all targets at once, then slice back out.
    all_keys, ranges = [], []
    for t in targets:
        start = len(all_keys)
        all_keys.append(t["a_key"])
        all_keys.extend(t["gold"])
        ranges.append((start, len(all_keys)))
    resolution = resolve_gold(con, all_keys)
    print(resolution.report(all_keys))

    modes = ["hybrid", "research"]
    results = []

    for i, t in enumerate(targets):
        a_start, a_end = ranges[i]
        a_ids = resolution.all_ids([a_start])          # a_key is always index a_start
        gold_ids = resolution.all_ids(range(a_start + 1, a_end))
        gold = set(gold_ids)
        a_id = a_ids[0] if a_ids else None
        rec = {"a_name": t["a_name"], "s1": t["s1"],
               "neighbour_subsystem": t["neighbour_subsystem"], "query": t["query"]}
        rec["gold_n"] = len(gold)
        rec["per_mode"] = {}
        for mode in modes:
            try:
                if mode == "research":
                    chunks = post(args.api, "/research", {"query": t["query"], **research_extra}, args.timeout)["chunks"]
                else:
                    chunks = post(args.api, "/search",
                                  {"query": t["query"], "mode": mode, "top_k": args.fetch_k}, args.timeout)["chunks"]
                ids = [c["id"] for c in chunks]
                rec["per_mode"][mode] = {
                    "recall": {k: recall_at(ids, gold, k) for k in KS},
                    "anchor_found": a_id in ids,
                    "error": False,
                }
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                rec["per_mode"][mode] = {"recall": {k: 0.0 for k in KS}, "anchor_found": False,
                                         "error": True, "msg": str(e)}
        results.append(rec)
        h, r = rec["per_mode"]["hybrid"], rec["per_mode"]["research"]
        print(f"  [{i+1}/{len(targets)}] {t['a_name'][:34]:34} "
              f"gold={len(gold)}  hybrid@20={h['recall'][20]:.2f}  "
              f"research@20={r['recall'][20]:.2f} @50={r['recall'][50]:.2f}")

    json.dump({"research_extra": research_extra, "results": results} if research_extra else results,
              open(args.out, "w"), indent=2)

    # ---- aggregate ----
    n = len(results) or 1
    def mean(mode, k):
        return sum(r["per_mode"][mode]["recall"][k] for r in results) / n
    def mean_anchor(mode):
        return sum(1 for r in results if r["per_mode"][mode]["anchor_found"]) / n

    print(f"\n=== neighbourhood coverage (n={len(results)}) ===")
    print(f"  {'mode':10} " + "  ".join(f"R@{k:<3}" for k in KS) + "   anchor")
    for mode in modes:
        print(f"  {mode:10} " + "  ".join(f"{mean(mode, k):5.2f}" for k in KS)
              + f"   {mean_anchor(mode):5.2f}")

    rb = mean("research", 50) - mean("research", 20)
    print(f"\n  research 'reached but buried' (R@50 - R@20): {rb:+.2f}")
    print(f"  research advantage over hybrid @20: {mean('research',20)-mean('hybrid',20):+.2f}  "
          f"@50: {mean('research',50)-mean('hybrid',50):+.2f}")
    wins = sum(1 for r in results if r["per_mode"]["research"]["recall"][20] > r["per_mode"]["hybrid"]["recall"][20])
    ties = sum(1 for r in results if r["per_mode"]["research"]["recall"][20] == r["per_mode"]["hybrid"]["recall"][20])
    print(f"  research > hybrid @20: {wins}/{len(results)}  (ties {ties})")
    print(f"wrote {len(results)} results -> {args.out}")


if __name__ == "__main__":
    main()
