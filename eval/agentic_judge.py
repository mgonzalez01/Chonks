"""eval/agentic_judge.py: grades agentic_eval.py's output.

Three verdicts per question: OBJECTIVE (cost/tokens/turns, read straight off
the run), GOLD COVERAGE (fraction of known-relevant symbol names in the
answer), and LLM JUDGE (pairwise, position-swapped: a win only counts if it
survives both A/B orderings, which cancels naive LLM-judge order bias).
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess

BASELINE_MCP = '{"mcpServers":{}}'

JUDGE_INSTR = (
    "You are comparing two answers to a developer's question about a codebase. "
    "Decide which answer is more correct, complete, and specific — it should name "
    "the right files/symbols and explain the actual mechanism. Reward grounded "
    "specifics; penalize vagueness, hand-waving, and wrong claims. If they are "
    "genuinely equivalent, say tie.\n\n"
    "Reply with ONLY a JSON object: {\"winner\": \"A\"|\"B\"|\"tie\", \"reason\": \"<one sentence>\"}"
)


_META_MARKER_RE = re.compile(
    r"^\s*(?:\*+\s*chonks note|#+\s*notes on the tools|\*\*?feedback\b)",
    re.IGNORECASE,
)


def strip_meta(answer):
    """Strips a trailing meta-feedback section (e.g. "*Chonks note:* ...")
    that agents append after their answer, so the judge scores the answer,
    not whichever arm complied with the feedback ask. Only strips a match
    starting in the last 40% of the text, so an early section sharing
    wording is never truncated."""
    if not answer:
        return answer
    lines = answer.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    cutoff = len(answer) * 0.6
    for i, line in enumerate(lines):
        is_marker = _META_MARKER_RE.match(line) is not None
        is_rule_then_marker = (
            line.strip() == "---" and i + 1 < len(lines)
            and _META_MARKER_RE.match(lines[i + 1]) is not None
        )
        if (is_marker or is_rule_then_marker) and offsets[i] >= cutoff:
            return "\n".join(lines[:i]).rstrip()
    return answer


def judge_once(question, ans_a, ans_b, model, timeout=180):
    ans_a = strip_meta(ans_a)
    ans_b = strip_meta(ans_b)
    prompt = (f"{JUDGE_INSTR}\n\nQUESTION:\n{question}\n\n"
              f"ANSWER A:\n{ans_a[:4000]}\n\nANSWER B:\n{ans_b[:4000]}")
    # Empty model omits the flag for gateways that can't resolve public model
    # ids. If that forces judge == arms model, note the self-preference caveat
    # when reporting; the swapped verdict is still usable.
    cmd = ["claude", "-p", prompt, "--output-format", "json",
           *(["--model", model] if model else []),
           "--max-turns", "1", "--permission-mode", "dontAsk",
           "--strict-mcp-config", "--mcp-config", BASELINE_MCP,
           "--allowedTools", ""]
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=timeout)
        result = json.loads(proc.stdout)["result"]
        m = re.search(r"\{.*\}", result, re.DOTALL)
        return json.loads(m.group(0))["winner"].lower()
    except Exception as e:
        return f"error:{type(e).__name__}"


def gold_coverage(answer, gold_names):
    if not gold_names:
        return None
    a = (answer or "").lower()
    hit = sum(1 for g in gold_names if g and g.lower() in a)
    return hit / len(gold_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir written by agentic_eval.py")
    ap.add_argument("--out", default=None, help="judge_results.json (default: <results>/judge_results.json)")
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="judge model (pin it); pass '' for the CLI's configured "
                         "default on gateway setups")
    ap.add_argument("--no-judge", action="store_true", help="objective + gold only, skip LLM judge")
    ap.add_argument("--chonks-arm", default="chonks",
                    help="arm name to pair against baseline (e.g. chonks_doc); "
                         "prior runs copy-renamed *.chonks_doc.json to *.chonks.json to fit "
                         "the old hardcoded glob")
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "judge_results.json")

    pairs = {}
    for f in glob.glob(os.path.join(args.results, "*.baseline.json")):
        qid = os.path.basename(f)[:-len(".baseline.json")]
        cf = os.path.join(args.results, f"{qid}.{args.chonks_arm}.json")
        if os.path.exists(cf):
            pairs[qid] = (json.load(open(f)), json.load(open(cf)))

    rows = []
    for qid, (b, c) in pairs.items():
        chonks_calls = sum(v for k, v in (c.get("tools_used") or {}).items() if "chonks" in k)
        row = {"qid": qid, "kind": b.get("kind"),
               "baseline_ok": b["ok"], "chonks_ok": c["ok"],
               "chonks_tool_calls": chonks_calls,  # did the agent actually USE chonks?
               "cost": {"baseline": b["cost_usd"], "chonks": c["cost_usd"]},
               "tokens_in": {"baseline": b["tokens_in"], "chonks": c["tokens_in"]},
               "turns": {"baseline": b["num_turns"], "chonks": c["num_turns"]},
               "gold": {"baseline": gold_coverage(b["answer"], b.get("gold_names")),
                        "chonks": gold_coverage(c["answer"], c.get("gold_names"))}}
        if not args.no_judge and b["ok"] and c["ok"]:
            # position-swap: o1 has baseline=A; o2 has chonks=A
            o1 = judge_once(b["question"], b["answer"], c["answer"], args.model)
            o2 = judge_once(b["question"], c["answer"], b["answer"], args.model)
            if o1 == "b" and o2 == "a":
                verdict = "chonks"
            elif o1 == "a" and o2 == "b":
                verdict = "baseline"
            else:
                verdict = "tie"  # inconsistent across swap, or genuine tie
            row["judge"] = {"verdict": verdict, "order1": o1, "order2": o2}
        rows.append(row)
        v = row.get("judge", {}).get("verdict", "—")
        print(f"  {qid:30} judge={v:8} "
              f"cost b/c={row['cost']['baseline']}/{row['cost']['chonks']}  "
              f"gold b/c={row['gold']['baseline']}/{row['gold']['chonks']}")

    json.dump(rows, open(out, "w"), indent=2)

    def med(sel):
        xs = [sel(r) for r in rows if sel(r) is not None]
        return statistics.median(xs) if xs else None
    def ratio(field):
        xs = [r[field]["chonks"] / r[field]["baseline"]
              for r in rows if r[field]["baseline"] and r[field]["chonks"]]
        return statistics.median(xs) if xs else None

    n = len(rows)
    adopted = sum(1 for r in rows if r.get("chonks_tool_calls", 0) > 0)
    print(f"\n=== agentic A/B  (n={n}) ===")
    print(f"  ADOPTION: chonks arm actually called a chonks tool in {adopted}/{n} "
          f"(if low, the agent ignored the MCP — value can't show)")
    print(f"  cost USD     baseline {med(lambda r: r['cost']['baseline'])}  "
          f"chonks {med(lambda r: r['cost']['chonks'])}   (chonks/baseline median {ratio('cost')})")
    print(f"  input tokens baseline {med(lambda r: r['tokens_in']['baseline'])}  "
          f"chonks {med(lambda r: r['tokens_in']['chonks'])}   (ratio {ratio('tokens_in')})")
    print(f"  turns        baseline {med(lambda r: r['turns']['baseline'])}  "
          f"chonks {med(lambda r: r['turns']['chonks'])}")
    gb = med(lambda r: r['gold']['baseline']); gc = med(lambda r: r['gold']['chonks'])
    print(f"  gold coverage baseline {gb}  chonks {gc}")
    if not args.no_judge:
        w = {"chonks": 0, "baseline": 0, "tie": 0}
        for r in rows:
            v = r.get("judge", {}).get("verdict")
            if v in w:
                w[v] += 1
        print(f"  judge wins   chonks {w['chonks']}  baseline {w['baseline']}  tie {w['tie']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
