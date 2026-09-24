"""eval/agentic_eval.py: extrinsic A/B, does an agent do better WITH Chonks?
Runs each question through the Claude Code CLI with baseline tools
(Read/Glob/Grep) and again with the Chonks MCP added, recording cost/tokens/
turns/answer per arm. Resumable: a completed result file is skipped.
Works on any codebase; grade output with agentic_judge.py."""
import argparse
import json
import os
import subprocess
from collections import Counter

# Empty MCP config for the baseline arm (the CLI requires the mcpServers key).
BASELINE_MCP = '{"mcpServers":{}}'

# Base instruction (all arms). Only the toolset + the optional nudge differ.
SYSTEM = (
    "You are answering a question about this codebase for a developer who is "
    "onboarding. Investigate with the tools you have, then give a concise, "
    "specific answer that names the key files and symbols as path:line citations. "
    "Do not modify any files. Stop once you can answer."
)
# Mirrors CLAUDE.md's steering text, so the A/B is grep vs Chonks-as-deployed.
NUDGE = (
    " This is a large, complex codebase. Prefer the Chonks MCP tools "
    "(codebase_search, codebase_research, codebase_map) to explore and understand "
    "how things work; use Grep/Read for the specifics of a file you've already located."
)

# All nine MCP tools. Under --permission-mode dontAsk, a tool missing from
# this list is silently DENIED, not prompted, so keep in sync with the MCP
# server's tool set or an arm quietly degrades to the tools it can reach.
_CHONKS = ("mcp__chonks__codebase_search,mcp__chonks__codebase_research,"
           "mcp__chonks__codebase_map,mcp__chonks__codebase_status,"
           "mcp__chonks__find_symbol,mcp__chonks__find_usages,"
           "mcp__chonks__trace_path,mcp__chonks__investigate,"
           "mcp__chonks__find_by_message")
BASELINE_TOOLS = "Read,Glob,Grep"               # file tools only
CHONKS_TOOLS = f"Read,Glob,Grep,{_CHONKS}"      # both, tests ADOPTION (agent chooses)
CHONKS_ONLY_TOOLS = f"Read,{_CHONKS}"           # no grep, tests CAPABILITY (forced)


def run_arm(question, *, source_root, mcp_config, allowed_tools, disallowed_tools, system, model,
            max_turns, max_budget, timeout):
    """One headless Claude Code run. Never raises on a failed agent run;
    captures the error in the result dict instead so the harness keeps going."""
    # stream-json to see every tool_use call, since which tools the agent
    # chose to use is itself a result. The final 'result' event still carries
    # the same usage/cost/answer the plain json format would.
    cmd = [
        "claude", "-p", question,
        "--output-format", "stream-json", "--verbose",
        # Empty model omits the flag (some gateways can't resolve public model
        # ids), but then the default can drift between runs, so pin when possible.
        *(["--model", model] if model else []),
        "--max-turns", str(max_turns),
        "--permission-mode", "dontAsk",
        "--strict-mcp-config", "--mcp-config", mcp_config,
        "--allowedTools", allowed_tools,
        "--append-system-prompt", system,
    ]
    if disallowed_tools:
        # Under --permission-mode dontAsk an allow-list alone doesn't restrict:
        # read-only Bash leaks through, so the forced arm must explicitly deny
        # grep/bash/glob/subagents or it bash-greps instead of using the MCP.
        cmd += ["--disallowedTools", disallowed_tools]
    if max_budget:
        cmd += ["--max-budget-usd", str(max_budget)]
    try:
        # text=True alone uses the locale codec (cp1252 on Windows); the
        # stream carries arbitrary UTF-8 from tool results, and one unmappable
        # byte there kills the reader thread and leaves proc.stdout None.
        proc = subprocess.run(cmd, cwd=source_root, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "answer": "", "cost_usd": None,
                "tokens_in": None, "tokens_out": None, "num_turns": None, "tools_used": {}}

    # tools_used counts call ATTEMPTS, not successes: a call rejected by
    # schema or MCP-side arg checks still emits a tool_use block, so a tool
    # can "appear used" while doing nothing. tool_errors tracks that gap.
    tools, final = [], None
    use_ids: dict = {}
    tool_errors: dict = {}
    error_samples: dict = {}
    # Full per-call trace: {name, input, result_excerpt, is_error}, in call
    # order, so post-hoc analysis of what ran doesn't depend on Claude Code's
    # own session transcripts under ~/.claude/projects/ (undocumented, no
    # retention guarantee). Excerpts are capped, not full results.
    call_log: list = []
    call_by_id: dict = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for b in ev.get("message", {}).get("content", []):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tools.append(b.get("name"))
                    use_ids[b.get("id")] = b.get("name")
                    entry = {"name": b.get("name"), "input": b.get("input"),
                             "result_excerpt": None, "is_error": None}
                    call_log.append(entry)
                    call_by_id[b.get("id")] = entry
        elif ev.get("type") == "user":
            for b in ev.get("message", {}).get("content", []):
                if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                    continue
                c = b.get("content")
                text = c if isinstance(c, str) else " ".join(
                    x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else str(c)
                entry = call_by_id.get(b.get("tool_use_id"))
                if entry is not None:
                    entry["result_excerpt"] = text[:1500]
                    entry["is_error"] = bool(b.get("is_error"))
                if b.get("is_error"):
                    name = use_ids.get(b.get("tool_use_id"), "?")
                    tool_errors[name] = tool_errors.get(name, 0) + 1
                    if name not in error_samples:
                        error_samples[name] = text[:200]
        elif ev.get("type") == "result":
            final = ev
    if final is None:
        return {"ok": False, "error": (proc.stderr or "no result event")[:300], "answer": "",
                "cost_usd": None, "tokens_in": None, "tokens_out": None, "num_turns": None,
                "tools_used": dict(Counter(tools)),
                "tool_errors": tool_errors, "tool_error_samples": error_samples,
                "tool_calls": call_log}
    u = final.get("usage", {}) or {}
    tokens_in = (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                 + u.get("cache_creation_input_tokens", 0))
    return {
        "ok": not final.get("is_error", False),
        "error": final.get("api_error_status") or final.get("terminal_reason"),
        "stop_reason": final.get("stop_reason"),  # 'end_turn' = answered; 'max_turns' = truncated
        "answer": final.get("result", ""),
        "cost_usd": final.get("total_cost_usd"),
        "tokens_in": tokens_in,
        "tokens_out": u.get("output_tokens"),
        "num_turns": final.get("num_turns"),
        "duration_ms": final.get("duration_ms"),
        "tools_used": dict(Counter(tools)),  # {tool_name: call_count}, ATTEMPTS (see loop comment)
        "tool_errors": tool_errors,          # {tool_name: errored_result_count}
        "tool_error_samples": error_samples, # {tool_name: first error text}
        "tool_calls": call_log,              # [{name, input, result_excerpt, is_error}] in call order
        "permission_denials": final.get("permission_denials"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="agentic_questions.json")
    ap.add_argument("--source-root", required=True, help="checkout the baseline arm greps/reads")
    ap.add_argument("--chonks-mcp", required=True, help="MCP config file (or inline JSON) launching the Chonks server")
    ap.add_argument("--out", required=True, help="output dir for per-question results (resumable)")
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="model for both arms; pass '' to use the CLI's configured "
                         "default (for gateway setups that can't resolve public ids)")
    ap.add_argument("--arms", default="baseline,chonks")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--max-budget-usd", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (smoke)")
    ap.add_argument("--extra-chonks-tools", default="",
                    help="comma-joined MCP tool names appended to the chonks arms' "
                         "allow-list, for a tool a newer MCP build ships that this "
                         "script's _CHONKS list doesn't know about yet. Under "
                         "--permission-mode dontAsk an absent tool is silently "
                         "DENIED, so an A/B across MCP builds with different tool "
                         "sets MUST pass the new tools here for the treatment arm "
                         "and omit them for the control build.")
    ap.add_argument("--nudge-file", default=None,
                    help="file whose content replaces the built-in one-line NUDGE for the "
                         "chonks_doc arm (e.g. mcp-server/CLAUDE.template.md, the documented workflow "
                         "guide). Measures the product-as-documented condition vs the cold floor.")
    args = ap.parse_args()

    if os.path.exists(args.chonks_mcp):
        chonks_mcp = open(args.chonks_mcp).read()
    else:
        chonks_mcp = args.chonks_mcp  # inline JSON
    arms = args.arms.split(",")
    doc_nudge = ("\n\n" + open(args.nudge_file).read()) if args.nudge_file else NUDGE
    # Standing disallow for every arm: subagent/web tools would dissolve the
    # arm boundary. Bash stays allowed in both arms.
    _NO_ORCHESTRATION = ("Task,Agent,Workflow,Monitor,ScheduleWakeup,TaskOutput,"
                         "TaskCreate,TaskUpdate,TaskStop,TaskList,SendMessage,"
                         "WebSearch,WebFetch")
    extra = ("," + args.extra_chonks_tools) if args.extra_chonks_tools else ""
    chonks_tools = CHONKS_TOOLS + extra
    chonks_only_tools = CHONKS_ONLY_TOOLS + extra
    arm_cfg = {
        # (mcp_config, allowed, disallowed, system_prompt)
        "baseline":    (BASELINE_MCP, BASELINE_TOOLS, _NO_ORCHESTRATION, SYSTEM),                       # grep only
        "chonks":      (chonks_mcp, chonks_tools, _NO_ORCHESTRATION, SYSTEM + NUDGE),                   # chonks as deployed (nudged)
        "chonks_doc":  (chonks_mcp, chonks_tools, _NO_ORCHESTRATION, SYSTEM + doc_nudge),               # + documented workflow guide (--nudge-file)
        "chonks_only": (chonks_mcp, chonks_only_tools, "Bash,Grep,Glob," + _NO_ORCHESTRATION, SYSTEM),  # forced (capability)
    }
    questions = json.load(open(args.questions))
    if args.limit:
        questions = questions[:args.limit]
    os.makedirs(args.out, exist_ok=True)

    for i, q in enumerate(questions):
        qid = q["id"]
        for arm in arms:
            path = os.path.join(args.out, f"{qid}.{arm}.json")
            if os.path.exists(path):
                # Only successful runs are final; a cached failure (bad MCP
                # path, unresolvable model id, timeout) re-runs on the next
                # invocation instead of demanding manual file deletion.
                try:
                    cached_ok = bool(json.load(open(path)).get("ok"))
                except (json.JSONDecodeError, OSError):
                    cached_ok = False
                if cached_ok:
                    print(f"  [{i+1}/{len(questions)}] {arm:8} {qid}  (cached)")
                    continue
                print(f"  [{i+1}/{len(questions)}] {arm:8} {qid}  (cached failure — retrying)")
            mcp, tools, disallow, system = arm_cfg[arm]
            print(f"  [{i+1}/{len(questions)}] {arm:8} {qid}  running...", flush=True)
            res = run_arm(q["question"], source_root=args.source_root, mcp_config=mcp,
                          allowed_tools=tools, disallowed_tools=disallow, system=system, model=args.model,
                          max_turns=args.max_turns, max_budget=args.max_budget_usd, timeout=args.timeout)
            res.update({"qid": qid, "arm": arm, "question": q["question"],
                        "gold": q.get("gold"), "gold_names": q.get("gold_names"),
                        "kind": q.get("kind"),
                        # '' means the CLI default answered; this records the
                        # setup, not the actual model used.
                        "model": args.model})
            json.dump(res, open(path, "w"), indent=2)
            cost = f"${res['cost_usd']:.3f}" if res["cost_usd"] is not None else "—"
            chonks_calls = sum(v for k, v in (res.get("tools_used") or {}).items() if "chonks" in k)
            print(f"      ok={res['ok']} stop={res.get('stop_reason')} turns={res['num_turns']} "
                  f"cost={cost} tok_in={res['tokens_in']} chonks_calls={chonks_calls} "
                  f"tools={res.get('tools_used')}"
                  + ("" if res["ok"] else f"  ERROR: {res['error']}"))

    print(f"\ndone -> {args.out}  (run agentic_judge.py to grade)")


if __name__ == "__main__":
    main()
