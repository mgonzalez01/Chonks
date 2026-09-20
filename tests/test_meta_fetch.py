from unittest.mock import MagicMock, patch

import pytest

from chonks.index.graph.pagerank import compute_pagerank_global
from chonks.summaries import build_folder_summaries


def _make_meta_chunk(chunk_id: str, path: str, name: str) -> dict:
    return {
        "id": chunk_id,
        "path": path,
        "language": "python",
        "name": name,
        "chunk_type": "function",
        "start_line": 1,
        "end_line": 5,
    }


def test_compute_pagerank_global_uses_meta_fetch():
    store = MagicMock()
    store.load_pagerank.return_value = {}  # force the live-compute fallback
    store.get_named_chunks_meta.return_value = [_make_meta_chunk("c1", "a.py", "foo")]
    store.get_all_refs.return_value = []

    compute_pagerank_global(store)

    store.get_named_chunks_meta.assert_called_once()
    store.get_named_chunks.assert_not_called()


def test_build_folder_summaries_uses_meta_fetch(tmp_path):
    store = MagicMock()
    store.get_named_chunks_meta.return_value = [_make_meta_chunk("c1", "pkg/a.py", "bar")]
    store.get_named_chunks_meta.side_effect = None
    store.get_named_chunks_meta.return_value = [_make_meta_chunk("c1", "pkg/a.py", "bar")]
    store.load_pagerank.return_value = {}

    store.get_all_refs.return_value = []
    store.get_folder_summary.return_value = None

    store.get_files_with_hashes.return_value = [
        {"path": "pkg/a.py", "content_hash": "abc123"}
    ]

    embed_fn = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    store.dim = 4

    build_folder_summaries(store, embed_fn)

    store.get_named_chunks.assert_not_called()
