"""eval/agentic_questions.py: builds relational (from relational_targets.json)
and lookup (paraphrase variant from queries.json) questions with automatic gold,
no hand-authored ground truth since nobody can hand-judge this corpus.
"""
import argparse
import hashlib
import json


def _id_of(key_or_name: str) -> str:
    return hashlib.sha256(key_or_name.encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relational", required=True, help="relational_targets.json")
    ap.add_argument("--queries", required=True, help="queries.json")
    ap.add_argument("--out", default="eval_data/agentic_questions.json")
    ap.add_argument("--per-kind", type=int, default=12, help="cap per kind (cost control)")
    args = ap.parse_args()

    out = []
    for t in json.load(open(args.relational))[:args.per_kind]:
        out.append({
            "id": f"rel-{_id_of(t['a_name'])}",
            "question": t["query"],
            "gold": [t["a_key"], *t["gold"]],
            "gold_names": [t["a_name"], *t["gold_names"]],
            "kind": "relational",
        })
    # Dedupe by id BEFORE slicing: ids hash the symbol, so duplicate paraphrases
    # per target collide and the eval runner skips the repeat, silently shrinking
    # the set. First paraphrase per target wins so regenerating keeps ids stable.
    para = [q for q in json.load(open(args.queries)) if q.get("variant") == "paraphrase"]
    seen_ids: set[str] = set()
    deduped = []
    for q in para:
        qid = _id_of(q.get("symbol") or q["text"])
        if qid not in seen_ids:
            seen_ids.add(qid)
            deduped.append(q)
    for q in deduped[:args.per_kind]:
        # gold_names: prefer explicit symbol field, else derive from gold identities.
        # Must not silently produce empty names, or coverage flatlines and the metric lies.
        names = [q["symbol"]] if q.get("symbol") else \
                [g["name"] for g in (q.get("gold") or []) if g.get("name")]
        if not names:
            raise SystemExit(f"lookup query has no symbol and no named gold: {q['text'][:60]!r}")
        out.append({
            "id": f"look-{_id_of(q.get('symbol') or q['text'])}",
            "question": q["text"],
            "gold": q.get("gold") or [],
            "gold_names": names,
            "kind": "lookup",
        })

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {len(out)} questions -> {args.out}")
    for o in out:
        print(f"  [{o['kind']:10}] {o['question'][:70]}")


if __name__ == "__main__":
    main()
