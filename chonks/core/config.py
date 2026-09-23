"""The config.json loader. It does not log: a caller owns its logger name and its failure policy."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

CONFIG_SEARCH_PATHS = (
    Path("config.json"),
    Path("chonks/config.json"),
    Path(".chonks.json"),
)


@dataclass(frozen=True)
class ConfigProblem:
    path: str
    reason: str
    missing: bool


@dataclass(frozen=True)
class ConfigLoad:
    data: dict
    path: Path | None
    problems: tuple[ConfigProblem, ...]


def load_config(explicit: str | None) -> ConfigLoad:
    """Load the explicit file, or the first candidate in the working directory that parses."""
    candidates = [explicit] if explicit else [str(p) for p in CONFIG_SEARCH_PATHS]
    problems = []
    for candidate in candidates:
        if not explicit and not Path(candidate).exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as e:
            problems.append(ConfigProblem(candidate, str(e), True))
            continue
        except Exception as e:
            problems.append(ConfigProblem(candidate, str(e), False))
            continue
        if not isinstance(data, dict):
            reason = f"top level is a JSON {type(data).__name__}, not an object"
            problems.append(ConfigProblem(candidate, reason, False))
            continue
        return ConfigLoad(data, Path(candidate), tuple(problems))
    return ConfigLoad({}, None, tuple(problems))


def resolve_codebase(codebase: str | None, explicit_config: bool, logger: logging.Logger) -> Path | None:
    """Rejects an absolute `codebase` path from an auto-discovered config (silent-redirection risk)."""
    if not codebase:
        return None
    p = Path(codebase)
    if (p.root or p.drive) and not explicit_config:
        logger.warning(
            "Ignoring absolute 'codebase' path %r from auto-discovered "
            "config; pass --config explicitly to use it.",
            codebase,
        )
        return None
    return p.resolve()
