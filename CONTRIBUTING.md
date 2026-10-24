# Contributing

I designed Chonks, initially as an implementation of the cAST paper, then incrementally by reading further papers and testing its boundaries. I own every decision in DESIGN.md. Much of the code was written by coding agents working from my specifications and my reading of the linked papers and repositories. The tests pass. The smell test doesn't, everywhere. A lot of it grew organically and it shows. If you contribute, leave the parts you touch more readable than you found them, for the sake of our collective sanity.

Issues and pull requests are welcome. The project is maintained in spare time and is not for profit, so replies can take a while.

`scripts/check.sh` runs every check a change has to pass, the same ones CI runs: ruff over `chonks/`, the tests, and the MCP server build.

Before opening a pull request that changes ranking, chunking, or graph construction, please include the reasoning or measurement behind the change in the PR description. A change to a module under `chonks/languages/` is a chunking change. [eval/FREEZES.md](eval/FREEZES.md) records what's already been measured, worth checking before you start in case there's a relevant number to build on. That's enough.

## Where code goes

`chonks/` is split into layers, and each layer imports only the layers listed for it:

| Layer | Holds | May import |
|---|---|---|
| `ops` | the CLI | every layer |
| `serve` | the HTTP server | retrieval, index, resolve, storage, embed, core |
| `retrieval` | queries, ranking, response shapes | resolve, storage, embed, languages, core |
| `index` | parsing, embedding, graph building | resolve, storage, embed, languages, core |
| `resolve` | which definitions a name or call reaches, read from the stored class model | storage, languages, core |
| `storage` | SQLite | languages, core |
| `embed` | clients for the local model servers (embedding, reranking) | core |
| `languages` | one spec per language | core |
| `core` | shared constants and helpers | nothing in `chonks` |

The rules that keep those seams clean. `tests/test_layering.py` checks the first five; review checks the rest.

- Imports follow the table.
- Language knowledge lives in `chonks/languages/`: a language name or a grammar node type is spelled nowhere else. Other code reads it from the registry (`union`, `table`, `merged` in `chonks.languages`) and rebinds it in a `_refresh_from_registry` hook so plugins take effect. A new per-language fact is a new `LanguageSpec` field.
- Edge-type names are spelled only in `chonks/core/edges.py` and the language specs.
- A third-party package is imported only where it is wrapped: scipy in `ops/subsystems.py`, sqlite-vec in `storage/`, FastAPI in `serve/`, tree-sitter-language-pack in `index/segment.py`.
- Environment variables are read only in `ops/`, `serve/main.py` and `index/graph/knn.py`.
- New response fields and note text are built in `retrieval/`; `storage/` returns rows.
- New logic goes in Python. The MCP server in `mcp-server/` renders what the HTTP API returns.
- Code goes in the layer whose job it is, not the one whose imports happen to allow it. When a change fits no layer, the pull request proposes a new one: a row in the table above and in `tests/test_layering.py`.

The exception lists in `tests/test_layering.py` only shrink: fix an entry and delete it, and never add one without saying why in the pull request.

## Code and tests

- Before writing a helper, search for code that already does the job, and reuse or extend it.
- A function stays within 60 statements and 100 lines; `tests/test_function_size.py` checks it.
- A fix comes with a test that fails without it.
- Extend an existing test before adding one.
- A test's name says what it checks.

`tests/test_function_size.py` lists the functions already over the limits at their current size. That list only shrinks and its ceilings only go down.

## Comments

The code is the code: a comment says what the code can't say for itself.

1. Explain hard logic, not decisions or history. No "originally", no "we chose X over Y", no measurements, no note on who or what wrote a hunk.
2. Match the density of the code around it, which is low. A docstring is a line or two, and the best ones are just an example.
3. Contracts stay: what a caller has to do, or what must stay in step with code elsewhere.
4. A `# noqa` names its rule and what the code is for: `# noqa: BLE001, any failure is the same answer: not usable`.
5. "Why this rule is narrower than it looks" goes in a test. If someone widens the rule, a test fails; a comment just gets deleted.
6. No TODO or FIXME. Known bugs go in an issue.
7. How the code got this way, what was tried, and the measurement and its method go in the pull request description. Commit bodies follow the rules above: short flat sentences about what landed, a "why" only where the diff can't show it, and at most one line of numbers.
