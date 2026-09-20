"""Registration point for callbacks that rebind a module-level value frozen at import from the language registry."""

from __future__ import annotations

from collections.abc import Callable

_HOOKS: list[Callable[[], None]] = []


def register_refresh(fn: Callable[[], None]) -> None:
    _HOOKS.append(fn)


def run_refreshes() -> None:
    for fn in _HOOKS:
        fn()
