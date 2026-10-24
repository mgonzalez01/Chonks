"""How a request's path_prefix reaches the index, and notes that say when a
scope, or a config entry, can't match anything."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from chonks.core.paths import (
    _missing_prefixes,
    _normalize_prefixes,
    _path_scope,
    _scope_exclusion,
    _scope_under_root,
    _spelling_key,
)

if TYPE_CHECKING:
    from chonks.storage.store import Store

_SUGGESTIONS = 3


def resolve_scope(
    store: Store, path_prefix: str | None, root: Path | None, exclude: list[str], include: list[str]
) -> tuple[str | None, str | None]:
    """The scope `path_prefix` names in the index, and a note when nothing
    indexed lies under it; (None, None) for the whole index."""
    scope = _path_scope(path_prefix)
    if scope is not None:
        scope = _scope_under_root(scope, root)
    if scope is None:
        return None, None
    excluded = _excluded_note(path_prefix, scope, exclude, include)
    if excluded or store.get_paths_in_scope(scope, limit=1):
        return scope, excluded
    return _respelled(scope, path_prefix, store.get_paths_under(""))


def _respelled(scope: str, typed: str, paths: list[str]) -> tuple[str, str | None]:
    """The one spelling `paths` give `scope` apart from case and Unicode form,
    else `scope` and a note naming indexed paths that start with it."""
    depth = scope.count("/") + 1
    deeper: dict[str, bool] = {}
    for path in paths:
        parts = path.split("/", depth)
        head = "/".join(parts[:depth])
        deeper[head] = deeper.get(head, False) or len(parts) > depth
    key = _spelling_key(scope)
    spellings: list[str] = []
    near: list[str] = []
    for head, has_more in deeper.items():
        folded = _spelling_key(head)
        if folded == key:
            spellings.append(head)
        if folded.startswith(key):
            near.append(head + "/" if has_more else head)
    if len(spellings) == 1:
        return spellings[0], None
    note = f"no indexed path under {typed}"
    if near:
        note += f"; did you mean {', '.join(sorted(near)[:_SUGGESTIONS])}?"
    return scope, note


def _excluded_note(
    typed: str | None, scope: str | None, exclude: list[str], include: list[str]
) -> str | None:
    if scope is None:
        return None
    rule = _scope_exclusion(scope, _normalize_prefixes(exclude), _normalize_prefixes(include))
    if rule is None:
        return None
    return f"{typed} is excluded from the index by config (exclude: {rule}), so nothing under it can match."


def excluded_scope_note(path_prefix: str | None, exclude: list[str], include: list[str]) -> str | None:
    return _excluded_note(path_prefix, _path_scope(path_prefix), exclude, include)


def config_warnings(
    root: str | Path | None, exclude: list[str], include: list[str], scopes: list[str] | tuple[str, ...] = ()
) -> list[str]:
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
