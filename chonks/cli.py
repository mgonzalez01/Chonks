"""chonks/cli.py: `chonks` console-script entry point.

Dispatches `chonks <subcommand> [args...]` to each module's own main(),
argv unchanged. No flags were renamed, only the invocation
(e.g. `chunker.py` -> `chonks index`).
"""
from __future__ import annotations

import sys

# subcommand -> module providing a main(argv=None) callable.
_MODULES = {
    "init": "chonks.init",
    "index": "chonks.chunker",
    "serve": "chonks.server",
    "doctor": "chonks.doctor",
    "report": "chonks.report",
}

_USAGE = (
    "usage: chonks {init,index,serve,doctor,report} [args...]\n"
    "\n"
    "Subcommands:\n"
    "  init     interactive first-run setup wizard\n"
    "  index    index source files into a chunks DB\n"
    "  serve    run the HTTP search/research/index server\n"
    "  doctor   read-only index health report\n"
    "  report   generate INDEX_REPORT.md\n"
    "\n"
    "Run `chonks <subcommand> --help` for that subcommand's own flags.\n"
)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)

    if not argv:
        print(_USAGE, file=sys.stderr)
        sys.exit(1)

    if argv[0] in ("-h", "--help"):
        print(_USAGE)
        sys.exit(0)

    cmd, rest = argv[0], argv[1:]
    module_name = _MODULES.get(cmd)
    if module_name is None:
        print(_USAGE, file=sys.stderr)
        print(f"chonks: unknown command {cmd!r}", file=sys.stderr)
        sys.exit(2)

    if cmd == "init":
        # init.py exposes run_wizard, not main(), so it's dispatched separately.
        from chonks.init import run_wizard
        sys.exit(run_wizard(rest))

    import importlib
    module = importlib.import_module(module_name)
    module.main(rest)


if __name__ == "__main__":
    main()
