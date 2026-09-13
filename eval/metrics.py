"""eval/metrics.py: Recall@k and MRR over run results, sliced by mode/variant/
kind/subsystem. Errored calls count as a miss, not excluded, since the agent
got nothing back either way.
"""
import argparse
import json
from collections import defaultdict

KS = (1, 5, 10, 20)


def consolidation(res, k=10):
    """Per-mode coverage and unique wins at top-k: a mode with zero unique wins
    is dominated and can be dropped."""
    modes = []
    for r in res:
        for m in r["per_mode"]:
            if m not in modes:
                modes.append(m)

    def hit(r, m):
        pm = r["per_mode"].get(m)
        return pm is not None and pm["rank"] is not None and pm["rank"] <= k

    n = len(res) or 1
    print(f"\n=== mode consolidation @k={k} (n={len(res)}) ===")
    print(f"  {'mode':10} {'cover':>6} {'unique':>7}")
    for m in modes:
        cover = sum(1 for r in res if hit(r, m))
        uniq = sum(1 for r in res if hit(r, m) and not any(hit(r, o) for o in modes if o != m))
        print(f"  {m:10} {cover / n:6.2f} {uniq:7d}")

    if {"semantic", "fts", "hybrid"} <= set(modes):
        parts = [r for r in res if hit(r, "semantic") or hit(r, "fts")]
        covered = sum(1 for r in parts if hit(r, "hybrid"))
        pure_only = sum(1 for r in res if (hit(r, "semantic") or hit(r, "fts")) and not hit(r, "hybrid"))
        denom = len(parts) or 1
        print(f"\n  hybrid covers {covered}/{denom} of (semantic ∪ fts) hits "
              f"·  pure-mode-only (hybrid misses): {pure_only}  (>0 => keep that pure mode)")
    if {"research", "hybrid"} <= set(modes):
        r_only = sum(1 for r in res if hit(r, "research") and not hit(r, "hybrid"))
        print(f"  research hits where hybrid misses: {r_only}  (research's marginal value)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval_data/results.json")
    ap.add_argument("--by", default="variant", choices=["variant", "kind", "subsystem", "none"])
    args = ap.parse_args()

    res = json.load(open(args.results))

    # preserve mode order as first seen
    modes = []
    for r in res:
        for m in r["per_mode"]:
            if m not in modes:
                modes.append(m)

    def slice_key(r):
        return "all" if args.by == "none" else r.get(args.by, "?")

    # (slice, mode) -> list of (rank|None)
    ranks = defaultdict(lambda: defaultdict(list))
    for r in res:
        sl = slice_key(r)
        for m in modes:
            pm = r["per_mode"].get(m)
            if pm is not None:
                ranks[sl][m].append(pm["rank"])

    cols = [f"R@{k}" for k in KS] + ["MRR"]
    for sl in sorted(ranks):
        print(f"\n[{args.by}={sl}]")
        print(f"  {'mode':10} " + "  ".join(f"{c:>5}" for c in cols) + "    n")
        for m in modes:
            rs = ranks[sl].get(m)
            if not rs:
                continue
            n = len(rs)
            recall = [sum(1 for x in rs if x is not None and x <= k) / n for k in KS]
            mrr = sum((1.0 / x) for x in rs if x is not None) / n
            vals = recall + [mrr]
            print(f"  {m:10} " + "  ".join(f"{v:5.2f}" for v in vals) + f"    {n}")

    consolidation(res, k=10)


if __name__ == "__main__":
    main()
