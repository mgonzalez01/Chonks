# Chonks retrieval eval

This harness benchmarks an embedding model on the retrieval task, measuring recall@k, MRR, and neighbourhood coverage on a corpus of your choosing. It has no hardcoded paths; the defaults match the Godot test corpus and are overridable, so any codebase can be indexed and measured with it.

The harness has two parts. The intrinsic retrieval eval (`sample_targets`, `run`, `metrics`, and the `relational_*` scripts) measures embedder and retrieval quality directly. The agentic A/B eval (`agentic_eval` and `agentic_judge`, described in the last section of this file) measures whether an agent does better with Chonks than without it; it costs real API spend and is a separate tool.

The numbers in the README were produced with this harness at a pinned commit and configuration. That record, together with the per-question result files, is in [`FREEZES.md`](FREEZES.md) and [`results/F1/`](results/F1/).

## Gold is portable rather than pinned to one DB

A gold target means that the chunk in which a query's answer lives is known in advance, so that recall and MRR need no judge; the check is simply whether that chunk comes back. The obvious way to record the chunk would be its DB row id. However, chunk ids are derived from `path:start_line:content` (see `chonks/index/rows.py`) and are therefore not stable across a re-index, since a chunker version bump, a source edit, or an embedder swap that triggers a fresh re-index all reassign them.

For this reason, gold is keyed on a portable identity instead: the `path`, the symbol `name`, and the `chunk_type`, with a content-hash tiebreak for ambiguous cases and for the rare unnamed chunk (see `eval/gold.py`). `sample_targets.py` and `relational_sample.py` freeze that identity into `targets.json` and `relational_targets.json`, and `run.py`, `relational_run.py`, and `token_footprint.py` resolve it back to chunk ids at run time against whichever DB is loaded. This is what allows one frozen question set to be reused across re-indexes and across embedders.

Resolution never fails silently. Every gold entry either resolves or appears in the printed report as `UNRESOLVED`, or as `ambiguous` when a name matches more than one chunk and the content does not disambiguate. Ambiguous entries are kept rather than dropped, and they count as a hit on any of their chunks, since a declaration and its definition legitimately share a name. This matters because a shrinking gold set would silently inflate recall by asking only the questions it can still answer; reporting the misses instead makes it visible when a corpus swap or a re-chunk has broken coverage.

## 0. Prerequisites

- `uv` and the Python dependencies, installed with `uv sync`.
- An embedding server reachable over HTTP (`embed_url` in `config.json`, or `--embed-url` on the CLI) that speaks the `/v1/embeddings` shape. Any model can be used, since comparing models is what the eval is for.
- A source codebase to index, in any language Chonks supports (see the root `DOCS.md`). It does not need to be large, but larger and denser corpora exercise the retrieval task more thoroughly. On small-file corpora every mode tends to look similar, and the differences in cost and recall appear on large-codebase-scale code.

## 1. Index the corpus

```bash
cp config.example.json my_config.json
# edit my_config.json: codebase, embed_url, embed_model, exclude/include

chonks index /path/to/your/codebase \
    --db my_corpus.db --config my_config.json
```

This parses, chunks, and embeds the corpus and builds the reference and neighbour graphs into `my_corpus.db`. Re-running with `--force` re-indexes everything, and re-running plain performs an incremental update after edits. The chunker details, such as exclude and include rules, subsystems, and macros, are covered in the root `DOCS.md`.

## 2. Serve it

```bash
chonks serve --db my_corpus.db --config my_config.json --port 11438
```

This has to keep running, since every step below talks to it over HTTP (`--api`, default `http://127.0.0.1:11438`).

A DB built by an older checkout can fail to serve with a `SCHEMA_VERSION` mismatch `RuntimeError`. The message says what to do, which is to drop the DB file and re-index with the current code.

## 3. Sample gold targets

```bash
uv run python eval/sample_targets.py --db my_corpus.db --out eval_data/targets.json
uv run python eval/relational_sample.py --db my_corpus.db --out eval_data/relational_targets.json
```

`targets.json` picks named definition chunks, stratified by subsystem and by kind (function or type), and each carries its `content` together with a portable `gold` identity. `relational_targets.json` picks anchors together with their real cross-subsystem dependency set, taken from `chunk_refs`, for the neighbourhood-coverage eval.

Both scripts default to `--language cpp` and `--subsystems core,scene,servers`, which matches the Godot test corpus. For a different corpus, `--language <lang>` and `--subsystems <prefix1,prefix2,...>` (comma-separated path prefixes, longest match wins) stratify by that tree instead, without any code edits. Nested prefixes are fine, so a tree with a single top-level root can be stratified one level deeper (`--subsystems src/gfx,src/audio,tools`). Vendored and third-party directories should be left out of the list, since questions generated from them do not measure retrieval on the code in question and their gold keys break when packages update.

## 4. Generate queries from the targets

There is no bundled query generator. Query text is written from each target's `content` in whatever way suits the experiment, for example by paraphrasing the symbol's purpose, or by using the symbol name verbatim as a sanity check. Each entry needs a `text` field, a `gold` field that carries the target's `gold` list forward unchanged, and whatever slicing metadata is wanted (`variant`, `subsystem`, `kind`), which `metrics.py --by` slices on. The result goes to `eval_data/queries.json`.

## 5. Run and score

```bash
uv run python eval/run.py \
    --queries eval_data/queries.json --db my_corpus.db --out eval_data/results.json
uv run python eval/metrics.py --results eval_data/results.json --by variant

uv run python eval/relational_run.py \
    --targets eval_data/relational_targets.json --db my_corpus.db \
    --out eval_data/relational_results.json
```

`run.py` prints the gold-resolution report before running any queries, and it should be checked for `UNRESOLVED` entries before the numbers are trusted. `metrics.py` reports Recall@{1,5,10,20} and MRR per mode (semantic, fts, hybrid, and research when enabled), sliced by `--by`, together with a mode-consolidation table that shows whether hybrid dominates the pure modes and whether `--research` adds marginal value.

`eval/token_footprint.py --db my_corpus.db --queries eval_data/queries.json` measures the context savings, that is, the lines and tokens Chonks returns compared with a baseline of grep followed by reading the files. This is the other half of the cost story, which recall alone does not capture.

## Swapping embedders

To compare embedders, `--embed-url` and `--embed-model` (or the `embed_url` and `embed_model` keys in `config.json`) are pointed at a different embedding server and the corpus is re-indexed from scratch into a new DB file. The vector dimension of the `chunk_vecs` table is fixed per DB at creation, a different model usually means a different dimension, and mixing dimensions in one DB is not supported. This is also the reason gold is keyed portably: the same `queries.json` runs unchanged against the new DB through `run.py --db my_new_corpus.db`, so that the index built with embedder A and the index built with embedder B answer the same question set, and their numbers can be compared fairly.

## Reference baseline: Godot (optional)

Godot (MIT) is a fully open corpus that can be used to check the harness, or as a floor baseline that others can reproduce exactly:

```bash
git clone https://github.com/godotengine/godot.git /tmp/godot-src
git -C /tmp/godot-src checkout 2c089e9bf0b8712d0bc444c2ceaf9c543ed9c777
chonks index /tmp/godot-src --db godot.db \
    --embed-url http://localhost:11437/v1/embeddings --embed-model <your-model>
chonks serve --db godot.db --port 11438
```

Steps 3 to 5 are then run against `godot.db`. Its small average file size makes cost-savings comparisons come out closer to neutral than on a large C++ codebase, so it should be treated as a regression floor rather than as the headline number.

## Agentic A/B eval: does an agent do better with Chonks?

The intrinsic evals measure retrieval in isolation. This eval measures the product claim itself, by running an agent on real questions over a codebase twice:

- **baseline**, with the built-in file tools only (`Read`, `Glob`, `Grep`)
- **chonks**, with the same tools plus the Chonks MCP server

It compares cost, tokens, and turns as well as answer quality, and it drives the real Claude Code CLI with the real MCP server, so that the path being tested is the one actually used rather than a simulation.

### Portable to any codebase

Nothing in it is specific to Godot, because the question set is generated from the index itself, without hand-authored answers, so it works on a corpus that nobody could hand-judge. Grading combines objective run metrics (cost and tokens), gold coverage derived from the index, and a pairwise judgement by a strong model.

### Prerequisites

- The `claude` CLI, authenticated (`claude -v`).
- The codebase indexed by Chonks, with its server running: `chonks serve --db <db> --config <cfg> --port <P>`.
- The Chonks MCP server built, with `npm run build` inside `mcp-server/`.
- An MCP config file that launches that server, passed as `--chonks-mcp`. Its shape is the `.mcp.json` block in [DOCS.md, Registering with Claude Code](../DEPLOY.md#6-registering-client-machines), with `CHONKS_DB` and `CHONKS_CONFIG` pointing at the eval corpus. The script also accepts the JSON inline.
- A source checkout for the baseline arm. On a codebase of your own, its path is passed directly. For a corpus that exists only as a DB, one can be reconstructed with `materialize_source.py`, whose fidelity caveat should be read first.

### Running it

The corpus must be indexed and served, and steps 1 to 4 above must have produced `queries.json` and `relational_targets.json`.

```bash
REPO=/path/to/Chonks
DATA=/path/to/your/eval_data          # queries.json, relational_targets.json, etc.

# 1. questions from the index-derived artifacts
uv run python $REPO/eval/agentic_questions.py \
    --relational $DATA/relational_targets.json --queries $DATA/queries.json \
    --out $DATA/agentic_questions.json

# 2. source tree to grep for the baseline arm.
#    On your own checkout, just point --source-root at it and skip this step.
#    If you only have a DB (no kept source — e.g. a reference corpus), reconstruct
#    one (read the fidelity caveat in materialize_source.py's docstring first):
uv run python $REPO/eval/materialize_source.py --db $DATA/corpus.db --out $DATA/src

# 3. run both arms (start with --limit for a cheap smoke)
uv run python $REPO/eval/agentic_eval.py \
    --questions $DATA/agentic_questions.json \
    --source-root $DATA/src \
    --chonks-mcp $DATA/mcp.eval.json \
    --out $DATA/agentic_runs --limit 2

# 4. grade (objective + gold + pairwise judge)
uv run python $REPO/eval/agentic_judge.py --results $DATA/agentic_runs
```

### How grading works (`agentic_judge.py`)

Each question receives three independent verdicts, layered from the most objective to the least, so that when they disagree it is clear which one to trust.

#### 1. Objective run metrics

These are the cost in USD (cache-aware, read from the CLI's own accounting), the input tokens, and the number of turns, per arm. Aggregates are medians rather than means, so that one research-heavy outlier run does not skew the headline ratio. Input tokens can mislead on their own, because a multi-turn arm re-reads its cached conversation prefix on every turn at about a tenth of the price, which inflates the raw token count while barely moving the cost. Cost rather than tokens should therefore be quoted.

#### 2. Gold coverage

This is the fraction of the question's known-relevant symbol names (the index-derived `gold_names`) that appear in the arm's final answer, checked as a case-insensitive substring and nothing more. The bluntness is deliberate, since the check is deterministic, free, immune to drift in the judge model, and works on a corpus that nobody can hand-grade. Its limits are equally blunt: an answer that names the symbol without understanding it scores, and a correct answer that paraphrases instead of naming does not. With a small `n` and few golds per question, the per-run values should be compared before the aggregate is trusted, because a single missed gold in one run can move the mean by several points.

#### 3. Pairwise LLM judge, blind and position-swapped

The judge sees the question and both answers, labelled only A and B and limited to the first 4,000 characters of each, so that very long answers are judged on their opening, and picks a winner or a tie. It is then asked again with A and B swapped, and a win counts only if it survives the swap:

```
order1 (baseline=A)   order2 (chonks=A)    verdict
B                     A                    chonks      (consistent)
A                     B                    baseline    (consistent)
anything else                              tie
```

The swap cancels position bias, which is the single largest known failure of naive LLM judging. One side effect should be kept in mind when reading the results: a tie covers two different situations, namely that the judge called both answers equivalent twice, or that it flip-flopped across the swap and gave no stable signal. `judge_results.json` records `order1` and `order2` per row, so the two can be told apart; `tie/tie` is real equivalence, whereas `A/A` or `B/B` is a flip-flop that the swap correctly neutralised. A run with many flip-flops means that the judge model cannot discriminate between these answers, and gold coverage is then the better signal.

**Judge model.** `agentic_eval.py` (the arms) and `agentic_judge.py` (the judge) share the same default, `--model claude-opus-4-8`, so by default the judge and the arms are the same model. To avoid self-preference bias, the arms model has to be passed explicitly and kept different from the judge's, for example by pinning the arms to Sonnet with `agentic_eval.py --model claude-sonnet-...` while leaving the judge at its default. The judge is an offline instrument, so the local-by-design constraint does not apply to it. On gatewayed deployments that cannot resolve public model ids, `--model ''` omits the flag and uses the CLI's configured default; the verdicts remain usable, since they are still blind and swapped, but it is then unknown which model judged, reruns are not comparable across gateway changes, and if the gateway forces the judge to be the arms' model, the self-preference caveat has to be noted when reporting. `--no-judge` skips this layer and keeps the two objective ones.

#### Adoption check

Before any of the above means anything, the summary reports how many runs of the chonks arm actually called a Chonks tool. If adoption is low, the agent ignored the MCP server and the A/B measured nothing, in which case the steering text needs fixing and the numbers should not be interpreted.

#### Reading a summary

```
ADOPTION: 10/10          ← the comparison is meaningful
cost ratio 0.32          ← the headline (median, cache-aware)
gold 0.775 vs 0.708      ← check per-run rows before quoting: is this a
                            systematic gap or one missed gold in one run?
judge: 0/0/10 ties       ← open judge_results.json: tie/tie (equivalence)
                            or flip-flops (judge can't discriminate)?
```

### Reproducibility and cost

- `--model` is pinned, and `--max-turns` and `--max-budget-usd` cap each run. Runs are resumable, since results are per-question files that are skipped on re-run.
- `--bare` would give cross-machine determinism by ignoring hooks and CLAUDE.md, but it also skips credential loading and fails with "Not logged in", so it is not used. Anything that a project CLAUDE.md or hooks inject loads symmetrically in both arms, so the A/B verdict stays fair and only the absolute cross-machine numbers drift. A fully clean run points `--source-root` at a tree without a project `CLAUDE.md` or `.claude/`.
- Each question is two agent runs plus two judge calls, which is real API spend, so `--limit` and the budget cap should be used until a smoke run looks right.
- All tools are read-only (`Read`, `Glob`, `Grep`, and the read-only MCP server), so nothing is written.

### Caveats

- A `--source-root` reconstructed with `materialize_source.py` has gaps, in the form of blank lines where no chunk covers them, commonly 70 to 75% line coverage. This mildly handicaps the baseline and flatters Chonks, so a real checkout should be used for a clean number.
- Question quality is capped by the index-derived gold, since `chunk_refs` is a name-match graph (see `relational_sample.py`). The judge dimension is independent of that.
