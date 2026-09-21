"""Shared fixtures.

Stubs the embedder probe as healthy so tests running `server.main()` don't dial
a real embedder; tests for the probe itself override this explicitly.
"""
import pytest


@pytest.fixture(autouse=True)
def _healthy_embedder_probe(monkeypatch):
    import chonks.serve.main as serve_main
    monkeypatch.setattr(serve_main, "probe_embedder", lambda embedder, timeout=10.0: None)
