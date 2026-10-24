"""Config loading for the commands that print their warnings instead of logging them."""
import sys

from chonks.core.config import load_config


def load_cli_config(explicit: str | None) -> dict:
    """Load config.json, printing each file that failed to stderr."""
    loaded = load_config(explicit)
    for problem in loaded.problems:
        prefix = "Failed to load" if explicit else "Failed to parse"
        print(f"{prefix} {problem.path}: {problem.reason}", file=sys.stderr)
    return loaded.data
