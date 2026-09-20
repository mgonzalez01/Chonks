# Measurement freezes

A freeze pins the exact system a set of published numbers describes. Every headline number in the README cites a freeze id. A ranking-touching change after a freeze opens the next freeze and re-runs the panel; it never silently updates a published number.

Ranking changes are gated by the panel, not by unit tests, and a panel result is reported per question (wins, ties, losses, k of n), never as a two-decimal delta below n=100.

## F1 — 2026-09-05

**Code.** Git tag `F1`, 946 tests passing, 8 skipped, `SCHEMA_VERSION = 5`, `CHUNKER_VERSION = 3`. Whether a later commit moved the ranking path is answered by `git diff F1 -- chonks/research.py chonks/searcher.py chonks/chunking.py`. The tag `F1` is the frozen system plus non-ranking changes made before publication (removal of a standalone graph viewer, the doctor `--set-model` flag, and comment and docstring rewrites); the ranking path itself is unchanged. `knn_backend` now defaults to auto-detection (cuda or mlx if installed, else numpy), which `eval/validate_knn_backend.py` shows is bit-identical to the numpy path.

**Addendum, 2026-09-20: code moved after F1.** After tag `F1` the code was split into layer packages (`chonks/core`, `languages`, `storage`, `embed`, `index`, `retrieval`, `serve`, `ops`); the old flat modules stay as import paths. `chonks/research.py` and `chonks/searcher.py` did not change: `git diff F1 -- chonks/research.py chonks/searcher.py` is empty. `chonks/chunking.py` did change: its language tables and per-language branches moved to `chonks/languages/` and its path-family diagnostics to `chonks/ops/diagnostics.py`, so `git diff F1 -- chonks/chunking.py` is large and does not by itself show a change in output. At commit `1d4149f`, the last one that touches `chonks/chunking.py` or `chonks/languages/`, the output was compared with `F1`: the segment, symbol and call-arity records of the test fixtures were equal; for 3,360 Godot files the 37,499 stored chunk rows and their symbols were equal; and `build_refs` over the F1 Godot index gave the same 8,517,698 `chunk_refs` rows. From there on, "did a later commit move the ranking path" is answered by `git diff F1 -- chonks/research.py chonks/searcher.py` plus `git diff 1d4149f -- chonks/chunking.py chonks/languages/`, and `tests/test_segment_golden.py` holds the segment output. Later commits change behaviour only outside the ranking path: the config loader (`chonks/core/config.py`), the table list of `chonks doctor`, and the `reason` field that `trace_path` gives for an unknown symbol.

**Embedder.** `jina-code-embeddings-0.5b`, GGUF F16 via llama-server with `--pooling last`, dim 896. The prefix preset keyed on a model name containing `jina-code` applies the query prefix `Find the most relevant code snippet given the following query:\n` and the document prefix `Candidate code snippet:\n`. llama-server is not bit-deterministic across calls, so the localization runs used a query-vector cache; four cached runs were byte-identical, and without the cache 8 of 15 panel rows jitter by one rank.

**Retrieval configuration.** The `_DEFAULTS` in `chonks/research.py`, all at default, with no overrides in the serving config:

```json
{"max_iterations": 3, "convergence_threshold": 0.15, "oversample_factor": 3,
 "max_candidates": 500, "top_k": 50, "iteration_timeout_s": 30,
 "graph_seeds_per_iter": 20, "graph_neighbours_per_seed": 5, "hub_indegree_max": 50,
 "structural_weight": 0.5, "structural_seed_n": 30,
 "edge_type_weights": {"calls": 1.0, "imports": 1.0, "inherits": 1.0, "xlang": 1.0, "associated": 1.0, "mentions": 1.0},
 "graph_seed_min_rel_score": 0.0, "interleave_reserved_n": 40, "interleave_file_cap": 3,
 "expand_paired_files": false}
```

Index-time settings were `cap_mentions_fanout` off, `associated_top_frac` 0.02 (PMI labels on), and the default `knn_backend`.

**Loc-Bench provenance.** The Loc-Bench instance ids and gold files come from Loc-Bench, the localization benchmark released with the LocAgent paper (Chen et al., 2025, arXiv:2503.09089); the Hugging Face dataset id is `czlll/Loc-Bench_V1`. The dataset page declares no licence and asks only that the paper be cited, which RELATED.md does; the accompanying LocAgent code is Apache-2.0. What ships here is limited to instance ids, gold file lists, and Chonks' own per-instance ranks, and does not include the issue texts.

**Corpora.** The index DBs are hundreds of megabytes each and are not shipped. They are rebuilt at tag `F1` from the sources below.

| corpus | instances | built | notes |
|---|---|---|---|
| Loc-Bench held-out | 100 | 2026-07-23 | one full index per instance; ids in `results/F1/n100_ids.txt`; chosen before any tuning and never used for it |
| Loc-Bench panel | 15 | 2026-08-13 | regression panel with a ±1 noise floor; no fingerprint, no PMI labels |
| Godot engine | 1 | checked out 2026-07-10 at commit `2c089e9bf0b8712d0bc444c2ceaf9c543ed9c777`, index built 2026-09-05 | 57,083 chunks and 2,688,321 refs (calls 782k, mentions 1.88M, associated 8.7k, inherits 13k); excludes `thirdparty/ misc/ doc/ .github/ editor/icons/ modules/mono/glue/`; fingerprint and PMI labels on |

**Harness.** The Loc-Bench adapters and the per-instance DBs live in a private evaluation repository. What ships here is the question sets, the gold, and the per-question outputs, under [`results/F1/`](results/F1/README.md). The scripts in this directory are the Godot side: `relational_run.py` measures neighbourhood coverage against a served instance over the 24 targets in `results/F1/relational_targets.json`, and `agentic_eval.py` with `agentic_judge.py` runs the 24 Godot questions in `results/F1/agentic_questions.json` with Sonnet 5 in both arms and an Opus blind judge. Adoption in the agentic runs means any Chonks tool call in the run. The shipped scripts' own defaults differ from this: both `agentic_eval.py` and `agentic_judge.py` default `--model` to `claude-opus-4-8`, so the arms' model has to be passed explicitly to reproduce the Sonnet-5-both-arms setup described here.

**Numbers established at F1.**

| measurement | result | file |
|---|---|---|
| Loc-Bench n=100, search | Acc@5 64, Acc@10 73 | `n100_F1_baseline.jsonl` |
| Loc-Bench n=100, research | Acc@5 64, Acc@10 74 | `n100_F1_baseline.jsonl` |
| Loc-Bench panel n=15, search / research | 10/11 and 7/10 at Acc@5/Acc@10 | `panel_F1_baseline.jsonl` |
| Loc-Bench panel n=15 with Qwen3-Embedding-0.6B and its query instruction, search / research | 7/10 and 9/9 at Acc@5/Acc@10 | `panel_qwen3_instruct.jsonl` |
| Loc-Bench panel n=15 with Qwen3-Embedding-0.6B, no query instruction, search / research | 7/10 and 7/10 at Acc@5/Acc@10 | `panel_qwen3_bare.jsonl` |
| Godot neighbourhood coverage, 24 anchors | hybrid R@10/20/50 0.11/0.19/0.32, research 0.14/0.21/0.39 | `relational_F1_baseline.json` |
| Godot feature-set coverage, 20 questions, 50-chunk budget | search 0.678, research 0.843, 12 wins / 8 ties / 0 losses | `fm_godot_F1.jsonl` |
| Adoption, 24 Godot questions | cold nudge 4/24; `CLAUDE.md` mandate 24/24 | `adoption_godot.json` |


**Ablation.** Masking the inferred `mentions` layer, which is 70% of the Godot graph's edges (1.88M of 2.69M), changes nothing on the Loc-Bench repositories (64/100 either way). On Godot it costs a quarter of the recovered neighbourhood, with research R@50 falling from 0.39 to 0.29, and the top-20 of `codebase_map` keeps 7 of 20 entries. Python's extracted call edges cover the neighbourhood, whereas C++ templates and member calls resolve to no typed edge and the inferred layer bridges the gap.

**Caveats.** Sample sizes are small. The two Qwen3 rows are the only numbers on this page not measured with the jina embedder; both used the same 15 instances and the same F1 code, one with Qwen3's query-side instruction and one without it, both with bare documents. 