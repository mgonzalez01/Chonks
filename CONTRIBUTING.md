# Contributing

I designed Chonks, initially as an implementation of the cAST paper, then incrementally by reading further papers and testing its boundaries. I own every decision in DESIGN.md. Much of the code was written by coding agents working from my specifications and my reading of the linked papers and repositories. The tests pass. The smell test doesn't, everywhere. A lot of it grew organically and it shows. If you contribute, leave the parts you touch more readable than you found them, for the sake of our collective sanity.

Issues and pull requests are welcome. The project is maintained in spare time and is not for profit, so replies can take a while.

Before opening a pull request that changes ranking, chunking, or graph construction, please include the reasoning or measurement behind the change in the PR description. A change to a module under `chonks/languages/` is a chunking change. [eval/FREEZES.md](eval/FREEZES.md) records what's already been measured, worth checking before you start in case there's a relevant number to build on. That's enough.

For everything else, `uv run pytest -q` and `cd mcp-server && npm run build` are the checks that CI runs.

## Where code goes

`chonks/` is split into layers, and each layer imports only the layers listed for it:

| Layer | Holds | May import |
|---|---|---|
| `ops` | the CLI | every layer |
| `serve` | the HTTP server | retrieval, index, storage, embed, core |
| `retrieval` | queries, ranking, response shapes | storage, embed, languages, core |
| `index` | parsing, embedding, graph building | storage, embed, languages, core |
| `storage` | SQLite | languages, core |
| `embed` | the embedding client | core |
| `languages` | one spec per language | core |
| `core` | shared constants and helpers | nothing in `chonks` |

The rules that keep those seams clean. `tests/test_layering.py` checks the first five.

- Imports follow the table.
- Language knowledge lives in `chonks/languages/`: a language name or a grammar node type is spelled nowhere else. Other code reads it from the registry (`union`, `table`, `merged` in `chonks.languages`) and rebinds it in a `_refresh_from_registry` hook so plugins take effect. A new per-language fact is a new `LanguageSpec` field.
- Edge-type names are spelled only in `chonks/core/edges.py` and the language specs.
- A third-party package is imported only where it is wrapped: scipy in `ops/subsystems.py`, sqlite-vec in `storage/`, FastAPI in `serve/`, tree-sitter-language-pack in `index/segment.py`.
- Environment variables are read only in `ops/`, `serve/main.py` and `index/graph/knn.py`.
- New response fields and note text are built in `retrieval/`; `storage/` returns rows.
- New logic goes in Python. The MCP server in `mcp-server/` renders what the HTTP API returns.

The exception lists in `tests/test_layering.py` only shrink: fix an entry and delete it, and never add one without saying why in the pull request.
