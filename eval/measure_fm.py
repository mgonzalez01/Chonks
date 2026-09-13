"""eval/measure_fm.py: scores file-set recall (FRR) per arm against git_gold's
co-change gold; the graph under test must never be the gold source, or scoring
becomes circular. Arms: search (hybrid baseline) vs research (graph expansion).
"""
import argparse
import json
import os
import statistics
import sys

import httpx


def _norm(path: str, case_sensitive: bool) -> str:
    p = path.replace("\\", "/")
    return p if case_sensitive else p.lower()


def files_of(chunks: list[dict], case_sensitive: bool) -> set[str]:
    return {_norm(c["path"], case_sensitive) for c in chunks if c.get("path")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--project", default=None)
    ap.add_argument("--case-sensitive", action="store_true")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    data = json.load(open(args.questions, encoding="utf-8"))
    if isinstance(data, dict):
        # Authoring agents tend to wrap the list ({"questions": [...], "flags":
        # ...}); accept any dict whose first list-of-dicts value is the set.
        for v in ([data.get(k) for k in ("questions", "candidates", "entries")]
                  + list(data.values())):
            if isinstance(v, list) and v and isinstance(v[0], dict):
                data = v
                break
    if not isinstance(data, list):
        print(f"could not find a question list in {args.questions}", file=sys.stderr)
        return 2
    questions = [q for q in data
                 if isinstance(q, dict) and q.get("question") and q.get("gold_files")]
    if not questions:
        print("no runnable questions (need question text + gold_files)", file=sys.stderr)
        return 2

    headers = {}
    client = httpx.Client(timeout=args.timeout, headers=headers)
    rows, recalls = [], {"search": [], "research": []}
    for q in questions:
        gold = {_norm(p, args.case_sensitive) for p in q["gold_files"]}
        row = {"id": q.get("id") or q.get("sha"), "n_gold": len(gold)}
        for arm, endpoint, payload in (
            ("search", "/search", {"query": q["question"], "mode": "hybrid",
                                   "top_k": args.top_k, "project": args.project}),
            ("research", "/research", {"query": q["question"], "project": args.project}),
        ):
            r = client.post(args.url.rstrip("/") + endpoint, json=payload)
            r.raise_for_status()
            got = files_of(r.json().get("chunks") or [], args.case_sensitive)
            hit = gold & got
            recall = len(hit) / len(gold)
            recalls[arm].append(recall)
            row[arm] = {"recall": round(recall, 4), "files_returned": len(got),
                        "missed": sorted(gold - got)}
        rows.append(row)
        print(f"{row['id']}: search {row['search']['recall']:.2f}  "
              f"research {row['research']['recall']:.2f}  (gold n={row['n_gold']})")

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nn={len(rows)} questions")
    for arm in ("search", "research"):
        vals = recalls[arm]
        print(f"  {arm:9s} mean recall@full {statistics.mean(vals):.3f}  "
              f"median {statistics.median(vals):.3f}")
    wins = sum(1 for a, b in zip(recalls["research"], recalls["search"]) if a > b)
    losses = sum(1 for a, b in zip(recalls["research"], recalls["search"]) if a < b)
    print(f"  research vs search: {wins}W / {len(rows)-wins-losses}T / {losses}L")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
