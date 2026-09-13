# Contributing

Issues and pull requests are welcome. The project is maintained in spare time and is not for profit, so replies can take a while.

Before opening a pull request that changes ranking, chunking, or graph construction, please include the reasoning or measurement behind the change in the PR description. [eval/FREEZES.md](eval/FREEZES.md) records what's already been measured, worth checking before you start in case there's a relevant number to build on. That's enough.

For everything else, `uv run pytest -q` and `cd mcp-server && npm run build` are the checks that CI runs.
