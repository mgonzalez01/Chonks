"""Notes that say when the index config keeps a scope, or a config entry,
from matching anything."""
from pathlib import Path

from chonks.core.paths import _missing_prefixes, _normalize_prefixes, _scope_exclusion


def excluded_scope_note(path_prefix: str | None, exclude: list[str], include: list[str]) -> str | None:
    if not path_prefix:
        return None
    rule = _scope_exclusion(path_prefix, _normalize_prefixes(exclude), _normalize_prefixes(include))
    if rule is None:
        return None
    return f"{path_prefix} is excluded from the index by config (exclude: {rule}), so nothing under it can match."


def config_warnings(root: str | Path | None, exclude: list[str], include: list[str],
                    scopes: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Exclude and include entries that match nothing on disk, and the scopes
    in `scopes` that the excludes keep out of the index."""
    warnings: list[str] = []
    if root is not None:
        for kind, entries in (("exclude", exclude), ("include", include)):
            for entry in _missing_prefixes(Path(root), _normalize_prefixes(entries)):
                warnings.append(f"{kind} entry {entry} matches nothing under {root}")
    for scope in dict.fromkeys(scopes):
        note = excluded_scope_note(scope, exclude, include)
        if note:
            warnings.append(note)
    return warnings
