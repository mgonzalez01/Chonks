# Freeze F1 result files

Per-question outputs behind every F1 number in [`../../FREEZES.md`](../../FREEZES.md). Question sets and gold are frozen inputs; the `*_140_baseline.*` files are the outputs of the runs the numbers were read from. The index DBs are not shipped (hundreds of MB each); they are rebuilt at git tag `F1` as FREEZES.md describes.

The Loc-Bench instance ids and gold files come from Loc-Bench, the localization benchmark released with the LocAgent paper (Chen et al., 2025, arXiv:2503.09089); the Hugging Face dataset id is `czlll/Loc-Bench_V1`. The dataset page declares no licence and asks only that the paper be cited, which RELATED.md does; the accompanying LocAgent code is Apache-2.0. What ships here is limited to instance ids, gold file lists, and Chonks' own per-instance ranks, and does not include the issue texts.

| file | what | feeds |
|---|---|---|
| `n100_ids.txt` | the 100 held-out Loc-Bench instance ids, chosen before any tuning | localization headline |
| `n100_140_baseline.jsonl` | one row per instance: gold files, per-mode Acc@5/Acc@10, per-file rank | localization headline (64/73, 64/74) |
| `panel_140_baseline.jsonl` | the 15-instance regression panel, same row shape | panel (10/11, 7/10) |
| `panel_qwen3_instruct.jsonl` | the same 15 instances re-indexed with Qwen3-Embedding-0.6B and its query instruction, same row shape | Qwen3 panel (7/10, 9/9), the README's Apache-2.0 alternative sentence |
| `panel_qwen3_bare.jsonl` | the same 15 instances with Qwen3-Embedding-0.6B and no query instruction, same row shape | the embedder comment's 7/15 bare research figure |
| `godot_gold_frozen.json` | 20 Godot commits with their touched files as gold | feature-set coverage |
| `fm_godot_questions.json` | the same 20 as `{id, question, gold_files}`, the harness input | feature-set coverage |
| `fm_godot_F1.jsonl` | per-question recall for `search` and `research`, with missed files | feature-set coverage (0.678 / 0.843, 12W/8T/0L) |
| `relational_targets.json` | 24 Godot anchors with their real dependency sets | neighbourhood coverage |
| `relational_140_baseline.json` | per-anchor R@10/20/50 for hybrid and research | neighbourhood coverage (0.39 R@50) |
| `agentic_questions.json` | the 24 Godot questions for the agent-in-the-loop runs | cost ratio |
| `adoption_godot.json` | per-run adoption for the two steering conditions on the 24 Godot questions | adoption 4/24 and 24/24 |

The Godot questions are verbatim commit messages. Contributor email addresses in their `Co-authored-by` trailers are replaced with `<email redacted>` in the shipped copies. The runs used the unredacted text; the only difference is that string. Contributor names in those trailers are retained, as public attribution carried over from Godot's own commit history.

Recompute the headline from the shipped rows:

```bash
python3 -c "import json;r=[json.loads(l) for l in open('n100_140_baseline.jsonl')];print({m:sum(x['modes'][m]['acc5'] for x in r) for m in ('search','research')})"
```

Recompute the adoption counts from `adoption_godot.json`:

```bash
python3 -c "import json;r=json.load(open('adoption_godot.json'))['runs'];from collections import Counter;print(Counter((x['condition'],x['adopted']) for x in r))"
```
