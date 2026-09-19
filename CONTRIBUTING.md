# Contributing

I designed Chonks, initially as an implementation of the cAST paper, then incrementally by reading further papers and testing its boundaries. I own every decision in DESIGN.md. Much of the code was written by coding agents working from my specifications and my reading of the linked papers and repositories. The tests pass. The smell test doesn't, everywhere. A lot of it grew organically and it shows. If you contribute, leave the parts you touch more readable than you found them, for the sake of our collective sanity.

Issues and pull requests are welcome. The project is maintained in spare time and is not for profit, so replies can take a while.

Before opening a pull request that changes ranking, chunking, or graph construction, please include the reasoning or measurement behind the change in the PR description. A change to a module under `chonks/languages/` is a chunking change. [eval/FREEZES.md](eval/FREEZES.md) records what's already been measured, worth checking before you start in case there's a relevant number to build on. That's enough.

For everything else, `uv run pytest -q` and `cd mcp-server && npm run build` are the checks that CI runs.
