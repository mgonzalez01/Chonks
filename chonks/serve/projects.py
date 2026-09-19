"""Project registry of the HTTP server: lazy Store opening and app lifespan."""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from chonks.query_reformulate import DEFAULT_REFORMULATE_QUERY
from chonks.searcher import Searcher
from chonks.store import Store

# Project registry, populated from config.json at startup.
# Built in two places (default project + multi-project loop in main()); a
# new key must be added to both or a project silently misses it.
_projects: dict[str, dict] = {}

DEFAULT_PROJECT = "default"

# Serialises first-touch Store opens: racing Store.__init__ calls on the same DB file are not safe.
_open_lock = threading.Lock()


def _open_store_if_needed(p: dict) -> None:
    """Opens the project's Store once; thread-safe for concurrent first-touch requests."""
    if p.get("store") is not None:
        return
    with _open_lock:
        if p.get("store") is None:  # re-check inside the lock
            p["store"]    = Store(p["db_path"])
            # Config-only, no per-request override, so the A/B toggles per project not per call.
            reformulate = bool((p["search_cfg"] or {}).get(
                "reformulate_query", DEFAULT_REFORMULATE_QUERY,
            ))
            p["searcher"] = Searcher(p["store"], p["embedder"], reformulate_query=reformulate)


def _get_project(name: str | None) -> dict:
    """Unknown name raises HTTPException 404; None resolves to the default project."""
    name = name or DEFAULT_PROJECT
    p = _projects.get(name)
    if p is None:
        raise HTTPException(
            404,
            f"Unknown project: {name!r}. Configured projects: {sorted(_projects)}",
        )
    _open_store_if_needed(p)
    return p


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Opens the default project eagerly to avoid first-request latency; named projects stay lazy.
    if DEFAULT_PROJECT in _projects:
        _open_store_if_needed(_projects[DEFAULT_PROJECT])
    yield
    for p in _projects.values():
        if p.get("store") is not None:
            p["store"].close()
