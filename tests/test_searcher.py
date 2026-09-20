"""Tests for _sanitize_fts_query."""
import pytest
from chonks.retrieval.results import (
    NEAR_DUP_MAX_CHUNKS,
    NEAR_DUP_MIN_RESULTS,
    NEAR_DUP_TAU,
    NEAR_DUP_WALL_SHARE_THRESHOLD,
    detect_near_dup_wall,
    rank_files,
)
from chonks.retrieval.searcher import Searcher, _sanitize_fts_query
from chonks.store import Store


def test_plain_words_are_quoted():
    assert _sanitize_fts_query("texture mapping") == '"texture" "mapping"'


def test_OR_token_becomes_quoted_literal():
    # "OR" as a bare word made FTS5 parse it as boolean operator → syntax error
    result = _sanitize_fts_query("OR mapping in renderer")
    assert result == '"OR" "mapping" "in" "renderer"'


def test_NOT_token_becomes_quoted_literal():
    result = _sanitize_fts_query("texture NOT")
    assert result == '"texture" "NOT"'


def test_AND_token_becomes_quoted_literal():
    result = _sanitize_fts_query("AND foo bar")
    assert result == '"AND" "foo" "bar"'


def test_punctuation_is_stripped():
    result = _sanitize_fts_query("what is this?")
    assert result == '"what" "is" "this"'


def test_all_punctuation_returns_empty():
    assert _sanitize_fts_query("???") == ""


def test_empty_string_returns_empty():
    assert _sanitize_fts_query("") == ""


def test_underscores_are_kept():
    result = _sanitize_fts_query("some_function call")
    assert result == '"some_function" "call"'


def test_non_ascii_token_survives_sanitization():
    """_FTS_TOKEN_RE used to be ASCII-only, so an accented identifier like
    café_lookup lost its "café_" prefix even though FTS5's unicode61
    tokenizer matches it. Widened to \\w+, it must survive whole."""
    result = _sanitize_fts_query("café_lookup")
    assert result == '"café_lookup"', f"non-ASCII token dropped: {result!r}"


def test_cjk_token_survives_sanitization():
    result = _sanitize_fts_query("检索 function")
    assert '"检索"' in result.split(), f"CJK token dropped: {result!r}"


class _FakeEmbedder:
    """Stand-in for chonks.embedder.Embedder: returns a fixed query vector,
    no network calls."""

    def __init__(self, vec):
        self._vec = vec

    def embed_queries(self, queries, client, timeout=30.0):
        return [self._vec for _ in queries]


def test_folder_blend_skips_stale_dim_folder_embedding(tmp_path):
    """A stale folder embedding whose dim differs from the query vector's
    used to reach np.dot directly and raise ValueError, crashing the search.
    It should be skipped, degrading folder_sim to 0.0."""
    store = Store(tmp_path / "test.db")
    store.insert_chunks(
        [{"id": "c1", "path": "pkg/mod.py", "content": "def foo(): pass",
          "name": "foo", "chunk_type": "function_definition",
          "start_line": 1, "end_line": 1}],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    # Stale: dim 2, doesn't match the store's dim-4 chunk embeddings.
    store.upsert_folder_summary("pkg", "summary", [0.5, 0.5], "hash1")

    searcher = Searcher(store, _FakeEmbedder([1.0, 0.0, 0.0, 0.0]))
    results = searcher.semantic("foo", top_k=5, folder_blend=(1.0, 0.2))

    assert results, "expected the chunk to come back, not raise"
    assert results[0]["id"] == "c1"
    assert results[0]["_folder_sim"] == 0.0, \
        "stale-dim folder embedding should degrade to folder_sim=0.0, not contribute a bogus score"
    store.close()


def test_min_score_filters_by_true_cosine_not_dead_scale(tmp_path):
    """Regression: chunk_vecs ranks by L2 distance (~117-142), not cosine in
    [0,2], so `1 - distance` used to return ~-127 and any min_score dropped
    everything. min_score must filter by true cosine instead."""
    store = Store(tmp_path / "test.db")
    # "similar" has cosine ~0.71, not 1.0, so its L2 distance is large (~95)
    # despite being a good match. A perfectly-aligned chunk would survive
    # even the broken scale, hiding the bug; this one doesn't.
    store.insert_chunks(
        [
            {"id": "similar", "path": "a.py", "content": "def a(): pass", "name": "a", "start_line": 1, "end_line": 1},
            {"id": "ortho",   "path": "b.py", "content": "def b(): pass", "name": "b", "start_line": 1, "end_line": 1},
        ],
        [[0.7, 0.7, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    )
    searcher = Searcher(store, _FakeEmbedder([1.0, 0.0, 0.0, 0.0]))
    assert len(searcher.semantic("q", top_k=5)) == 2
    # min_score=0.5: true cosine of "similar" is ~0.71 (kept), "ortho" ~0
    # (dropped). Under the old dead scale both had similarity ~-95, so this
    # assertion failed pre-fix.
    ids = [c["id"] for c in searcher.semantic("q", top_k=5, min_score=0.5)]
    assert ids == ["similar"], f"expected only the similar chunk, got {ids}"
    store.close()


def test_split_identifier_camelcase_only():
    """Only genuine mixed-case tokens split; snake_case is left alone because
    unicode61 already indexes its parts and the whole-token phrase matches, so
    splitting it adds only common-word noise."""
    from chonks.retrieval.searcher import _split_identifier
    assert _split_identifier("fileCap") == ["file", "Cap"]
    assert _split_identifier("getHTTPResponse") == ["get", "HTTP", "Response"]
    assert _split_identifier("parse_one") == []
    assert _split_identifier("my_tag") == []
    assert _split_identifier("single") == []
    assert _split_identifier("HTTP") == []


def test_hybrid_camelcase_query_reaches_snake_case_chunk(tmp_path):
    """A "fileCap" query tokenizes to "filecap", which doesn't match an
    indexed `file_cap` (tokens file,cap). The OR-fallback's camelCase split
    emits the parts so the file_cap chunk still gets an FTS rank."""
    store = Store(tmp_path / "test.db")
    store.insert_chunks(
        [
            {"id": "snake", "path": "a.py", "content": "def file_cap(): return 0", "name": "file_cap", "start_line": 1, "end_line": 1},
            {"id": "other", "path": "b.py", "content": "def unrelated(): pass", "name": "unrelated", "start_line": 1, "end_line": 1},
        ],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    searcher = Searcher(store, _FakeEmbedder([1.0, 0.0, 0.0, 0.0]))
    snake = next(c for c in searcher.hybrid("fileCap", top_k=5) if c["id"] == "snake")
    assert snake["_rank_fts"] is not None, \
        "camelCase split must give the snake_case chunk an FTS rank via the OR fallback"
    store.close()


def _mixed_kind_store(tmp_path, vec=(1.0, 0.0, 0.0, 0.0)):
    store = Store(tmp_path / "test.db")
    store.insert_chunks(
        [
            {"id": "code1", "path": "pkg/a.py", "language": "python",
             "content": "def needle(): pass", "name": "needle",
             "start_line": 1, "end_line": 1},
            {"id": "docs1", "path": "pkg/a.md", "language": "md",
             "content": "needle docs", "start_line": 1, "end_line": 1},
        ],
        [list(vec), list(vec)],
    )
    return store


def test_semantic_chunk_kind_filters_plain_path(tmp_path):
    vec = [1.0, 0.0, 0.0, 0.0]
    store = _mixed_kind_store(tmp_path, vec)
    searcher = Searcher(store, _FakeEmbedder(vec))
    results = searcher.semantic("needle", top_k=5, chunk_kind="code")
    assert {r["id"] for r in results} == {"code1"}
    store.close()


def test_semantic_chunk_kind_threads_through_folder_blend(tmp_path):
    """folder_blend oversamples via a separate store.search_semantic call;
    chunk_kind must reach that call too, not just the plain (no-blend) path."""
    vec = [1.0, 0.0, 0.0, 0.0]
    store = _mixed_kind_store(tmp_path, vec)
    searcher = Searcher(store, _FakeEmbedder(vec))
    results = searcher.semantic("needle", top_k=5, folder_blend=(1.0, 0.2), chunk_kind="code")
    assert {r["id"] for r in results} == {"code1"}
    store.close()


def test_hybrid_chunk_kind_filters_both_branches(tmp_path):
    """hybrid fuses candidates from both the semantic and FTS branches;
    chunk_kind must reach both, not just apply post-fusion."""
    vec = [1.0, 0.0, 0.0, 0.0]
    store = _mixed_kind_store(tmp_path, vec)
    searcher = Searcher(store, _FakeEmbedder(vec))
    results = searcher.hybrid("needle", top_k=5, chunk_kind="code")
    assert {r["id"] for r in results} == {"code1"}
    store.close()


def test_hybrid_chunk_kind_any_matches_default_behavior(tmp_path):
    vec = [1.0, 0.0, 0.0, 0.0]
    store = Store(tmp_path / "test.db")
    store.insert_chunks(
        [{"id": "code1", "path": "pkg/a.py", "language": "python",
          "content": "def needle(): pass", "name": "needle",
          "start_line": 1, "end_line": 1}],
        [vec],
    )
    searcher = Searcher(store, _FakeEmbedder(vec))
    default = searcher.hybrid("needle", top_k=5)
    assert default == searcher.hybrid("needle", top_k=5, chunk_kind="any")
    assert default == searcher.hybrid("needle", top_k=5, chunk_kind=None)
    store.close()


def test_near_dup_constants_match_calibration():
    """Pins calibrated values (see DOCS.md, POST /search): TAU=0.85 and
    WALL_SHARE_THRESHOLD=0.6 come from a calibration sweep against the LLVM
    `loadSectionContribs` wall episode. Do not adjust ad hoc."""
    assert NEAR_DUP_TAU == 0.85
    assert NEAR_DUP_WALL_SHARE_THRESHOLD == 0.6
    assert NEAR_DUP_MIN_RESULTS == 5
    assert NEAR_DUP_MAX_CHUNKS == 50

def _insert_vecs(store: Store, vecs: dict[str, list[float]]) -> None:
    """Insert one chunk per (id, vector) pair. insert_chunks quantizes the
    float vectors the same way a real embedding would."""
    ids = list(vecs)
    store.insert_chunks(
        [
            {"id": cid, "path": f"{cid}.py", "content": f"def {cid}(): pass",
             "name": cid, "start_line": 1, "end_line": 1}
            for cid in ids
        ],
        [vecs[cid] for cid in ids],
    )


def _chunk_dicts(ids: list[str]) -> list[dict]:
    return [{"id": i} for i in ids]


def test_near_dup_wall_fires_on_identical_vectors(tmp_path):
    store = Store(tmp_path / "test.db")
    v = [0.9, 0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    ids = [f"wall{i}" for i in range(5)]
    _insert_vecs(store, {cid: v for cid in ids})

    searcher = Searcher(store, embedder=None)
    result = detect_near_dup_wall(searcher.store, _chunk_dicts(ids))

    assert result is not None
    assert result["wall_share"] == 1.0
    assert result["wall_size"] == 5
    assert result["tau"] == NEAR_DUP_TAU
    assert result["wall_share"] >= NEAR_DUP_WALL_SHARE_THRESHOLD, "must clear the firing threshold"
    store.close()


def test_near_dup_wall_diverse_vectors_dont_fire(tmp_path):
    store = Store(tmp_path / "test.db")
    ids = [f"diverse{i}" for i in range(5)]
    # Standard basis vectors in dim 5: pairwise cosine 0 (orthogonal).
    vecs = {ids[i]: [1.0 if j == i else 0.0 for j in range(5)] for i in range(5)}
    _insert_vecs(store, vecs)

    searcher = Searcher(store, embedder=None)
    result = detect_near_dup_wall(searcher.store, _chunk_dicts(ids))

    assert result is not None
    assert result["wall_share"] == pytest.approx(1 / 5)
    assert result["wall_size"] == 1
    assert result["wall_share"] < NEAR_DUP_WALL_SHARE_THRESHOLD, "must NOT clear the firing threshold"
    store.close()


def test_near_dup_wall_skips_below_min_results(tmp_path):
    store = Store(tmp_path / "test.db")
    ids = [f"c{i}" for i in range(NEAR_DUP_MIN_RESULTS - 1)]
    v = [1.0, 0.0, 0.0, 0.0]
    _insert_vecs(store, {cid: v for cid in ids})

    searcher = Searcher(store, embedder=None)
    assert detect_near_dup_wall(searcher.store, _chunk_dicts(ids)) is None
    store.close()


def test_near_dup_wall_empty_chunks_returns_none(tmp_path):
    store = Store(tmp_path / "test.db")
    searcher = Searcher(store, embedder=None)
    assert detect_near_dup_wall(searcher.store, []) is None
    store.close()


def test_near_dup_wall_skips_missing_vector_rows_gracefully(tmp_path):
    """A chunk id with no chunk_vecs row (e.g. an fts/regex hit whose vector
    was never computed) must be skipped, not crash the detector, and doesn't
    count toward NEAR_DUP_MIN_RESULTS."""
    store = Store(tmp_path / "test.db")
    v = [0.9, 0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    ids = [f"wall{i}" for i in range(5)]
    _insert_vecs(store, {cid: v for cid in ids})
    searcher = Searcher(store, embedder=None)

    result = detect_near_dup_wall(searcher.store, _chunk_dicts(ids + ["ghost-id"]))
    assert result is not None
    assert result["wall_share"] == 1.0
    assert result["wall_size"] == 5

    few_ids = ids[:3]
    result2 = detect_near_dup_wall(searcher.store, _chunk_dicts(few_ids + ["ghost-a", "ghost-b"]))
    assert result2 is None
    store.close()


# ---------------------------------------------------------------------------
# search.file_cap
# ---------------------------------------------------------------------------

def _file_cap_wall_store(tmp_path):
    """A hot file (hot.py) with 5 chunks ranked ahead of one chunk each from
    three distinct files (b/c/d.py): the near-duplicate-wall shape this
    reports, where a plain top_k=5 truncation returns nothing but hot.py."""
    store = Store(tmp_path / "test.db")
    store.insert_chunks(
        [
            {"id": "hot0", "path": "hot.py", "content": "def hot0(): pass", "name": "hot0", "start_line": 1, "end_line": 1},
            {"id": "hot1", "path": "hot.py", "content": "def hot1(): pass", "name": "hot1", "start_line": 2, "end_line": 2},
            {"id": "hot2", "path": "hot.py", "content": "def hot2(): pass", "name": "hot2", "start_line": 3, "end_line": 3},
            {"id": "hot3", "path": "hot.py", "content": "def hot3(): pass", "name": "hot3", "start_line": 4, "end_line": 4},
            {"id": "hot4", "path": "hot.py", "content": "def hot4(): pass", "name": "hot4", "start_line": 5, "end_line": 5},
            {"id": "b0",   "path": "b.py",   "content": "def b0(): pass",   "name": "b0",   "start_line": 1, "end_line": 1},
            {"id": "c0",   "path": "c.py",   "content": "def c0(): pass",   "name": "c0",   "start_line": 1, "end_line": 1},
            {"id": "d0",   "path": "d.py",   "content": "def d0(): pass",   "name": "d0",   "start_line": 1, "end_line": 1},
        ],
        [
            [1.00, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # hot0: sim 1.00
            [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # hot1
            [0.90, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # hot2
            [0.85, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # hot3
            [0.80, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # hot4
            [0.70, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # b0
            [0.60, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # c0
            [0.50, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # d0
        ],
    )
    query_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    searcher = Searcher(store, _FakeEmbedder(query_vec))
    return store, searcher


def test_file_cap_off_matches_uncapped_behaviour(tmp_path):
    store, searcher = _file_cap_wall_store(tmp_path)
    results = searcher.semantic("q", top_k=5)
    assert [c["id"] for c in results] == ["hot0", "hot1", "hot2", "hot3", "hot4"]
    store.close()


def test_file_cap_limits_per_file_and_backfills_distinct_files(tmp_path):
    store, searcher = _file_cap_wall_store(tmp_path)
    results = searcher.semantic("q", top_k=5, file_cap=3)
    ids = [c["id"] for c in results]
    assert ids.count("hot0") + ids.count("hot1") + ids.count("hot2") == 3
    assert sum(1 for i in ids if i.startswith("hot")) == 3, \
        "no more than file_cap chunks from hot.py"
    assert {"b0", "c0"}.issubset(set(ids)), "backfilled slots must come from distinct files"
    assert "hot3" not in ids and "hot4" not in ids, \
        "the sacrificed hot.py chunks are the weakest (tail) ones, not the strongest"
    store.close()


def test_file_cap_larger_than_diversity_degrades_to_uncapped(tmp_path):
    store, searcher = _file_cap_wall_store(tmp_path)
    uncapped = searcher.semantic("q", top_k=5)
    capped = searcher.semantic("q", top_k=5, file_cap=10)
    assert [c["id"] for c in capped] == [c["id"] for c in uncapped]
    store.close()


def test_file_cap_preserves_rank_order(tmp_path):
    store, searcher = _file_cap_wall_store(tmp_path)
    results = searcher.semantic("q", top_k=5, file_cap=3)
    ids = [c["id"] for c in results]
    assert ids == ["hot0", "hot1", "hot2", "b0", "c0"]
    store.close()


def test_hybrid_file_cap_off_matches_uncapped_behaviour(tmp_path):
    """hybrid()'s default (file_cap=0) must reproduce the pre-fix plain-RRF
    ranking exactly: eval calls hybrid(query, top_k=50) with no file_cap and
    depends on byte-identical output."""
    store, searcher = _file_cap_wall_store(tmp_path)
    # Query token absent from every chunk's content, so the FTS branch
    # contributes nothing and the fused ranking collapses to pure semantic-RRF
    # order, reusing the semantic file_cap wall fixture directly.
    results = searcher.hybrid("zzzznotpresent", top_k=5)
    assert [c["id"] for c in results] == ["hot0", "hot1", "hot2", "hot3", "hot4"]
    store.close()


def test_hybrid_file_cap_limits_per_file_and_backfills_distinct_files(tmp_path):
    """hybrid(file_cap=3) caps hot.py's near-duplicate chunks the same way
    semantic() does. Previously hybrid had no such protection, even though
    it's the mode most callers use."""
    store, searcher = _file_cap_wall_store(tmp_path)
    results = searcher.hybrid("zzzznotpresent", top_k=5, file_cap=3)
    ids = [c["id"] for c in results]
    assert sum(1 for i in ids if i.startswith("hot")) == 3, \
        "no more than file_cap chunks from hot.py"
    assert {"b0", "c0"}.issubset(set(ids)), "backfilled slots must come from distinct files"
    assert "hot3" not in ids and "hot4" not in ids
    store.close()


# --- rank_files (tier 1) -----------------------------------------

def _chunk(path: str, score: float | None) -> dict:
    c = {"path": path}
    if score is not None:
        c["_score"] = score
    return c


def test_best_chunk_ordering_not_sum_aggregation():
    """Measured: sum-of-top-m aggregation dropped search Acc@5 from .64 to
    .40 because same-file chunks are near-duplicates, double-counting
    correlated evidence. File order follows each file's BEST chunk, not the
    sum."""
    chunks = [
        _chunk("other1.py", 0.90),
        _chunk("other2.py", 0.80),
        _chunk("other3.py", 0.70),
        _chunk("other4.py", 0.66),
        _chunk("single.py", 0.65),   # rank 5
        _chunk("multi.py",  0.60),   # rank 6
        _chunk("other5.py", 0.59),
        _chunk("multi.py",  0.58),   # rank 8
        _chunk("multi.py",  0.57),   # rank 9
    ]
    files = {f["path"]: f for f in rank_files(chunks)}
    ranked_paths = [f["path"] for f in rank_files(chunks)]
    assert ranked_paths.index("single.py") < ranked_paths.index("multi.py")
    assert files["multi.py"]["score"] == pytest.approx(0.60)
    assert files["multi.py"]["n_chunks"] == 3
    assert files["multi.py"]["best_rank"] == 6
    assert files["single.py"]["score"] == pytest.approx(0.65)
    # best-chunk ordering reproduces first-appearance order exactly
    assert ranked_paths == ["other1.py", "other2.py", "other3.py", "other4.py",
                            "single.py", "multi.py", "other5.py"]


def test_volume_never_beats_quality():
    """Chunk volume is payload, not evidence: count bonuses were measured
    strictly harmful in the pilot."""
    volume_chunks = [_chunk("volume.py", 0.10) for _ in range(10)]
    quality_chunks = [_chunk("quality.py", 0.90)]
    chunks = volume_chunks + quality_chunks
    ranked = rank_files(chunks)
    assert ranked[0]["path"] == "quality.py"
    assert ranked[0]["score"] == pytest.approx(0.90)
    files = {f["path"]: f for f in ranked}
    assert files["volume.py"]["score"] == pytest.approx(0.10)
    assert files["volume.py"]["n_chunks"] == 10


def test_missing_score_treated_as_zero_no_keyerror():
    """Graph/regex-expanded chunks often lack `_score` entirely; rank_files
    must not KeyError and treats them as contributing 0.0."""
    chunks = [
        _chunk("scored.py", 0.5),
        _chunk("graph_origin.py", None),  # no _score key at all
    ]
    files = {f["path"]: f for f in rank_files(chunks)}
    assert files["graph_origin.py"]["score"] == 0.0
    assert files["graph_origin.py"]["n_chunks"] == 1


def test_tiebreak_equal_best_score_earlier_appearance_wins():
    chunks = [
        _chunk("first.py", 0.45),
        _chunk("second.py", 0.45),
        _chunk("second.py", 0.45),
    ]
    ranked = rank_files(chunks)
    assert [f["path"] for f in ranked] == ["first.py", "second.py"]


def test_tiebreak_equal_score_equal_chunks_earlier_rank_wins():
    chunks = [
        _chunk("later.py", 0.5),   # rank 1
        _chunk("earlier.py", 0.5), # rank 2, but appears again below
        _chunk("earlier.py", 0.0),
        _chunk("later.py", 0.0),
    ]
    # both files: best score 0.5, so first appearance decides: later.py (rank 1)
    ranked = rank_files(chunks)
    assert [f["path"] for f in ranked] == ["later.py", "earlier.py"]


# --- hybrid FTS starvation fallback -------------------------

import math

from chonks.retrieval.searcher import OR_FALLBACK_MAX_RANK, _or_fts_query


def test_or_fts_query_dedupes_case_insensitively():
    assert _or_fts_query("Wedged wedged embedder") == '"Wedged" OR "embedder"'


def test_or_fts_query_all_punctuation_is_empty():
    assert _or_fts_query("???") == ""


def _vec_for_cos(c):
    """Unit vector at cosine c to the query direction [1, 0, 0, 0]."""
    return [c, math.sqrt(1.0 - c * c), 0.0, 0.0]


def _starved_store(tmp_path):
    """Store where the sanitised AND query starves ('flurble' matches
    nothing), the gold chunk sits at the bottom of the semantic pool, and a
    lexically-loud noise chunk sits outside the semantic pool (top_k=3,
    oversample 6)."""
    store = Store(tmp_path / "test.db")
    chunks, vecs = [], []
    for i, c in enumerate([0.95, 0.85, 0.75, 0.65, 0.55]):
        chunks.append({"id": f"filler{i+1}", "path": f"f{i+1}.py",
                       "content": "alpha beta gamma", "name": f"f{i+1}",
                       "start_line": 1, "end_line": 1})
        vecs.append(_vec_for_cos(c))
    chunks.append({"id": "gold", "path": "gold.py",
                   "content": "wedged embedder give up", "name": "gold",
                   "start_line": 1, "end_line": 1})
    vecs.append(_vec_for_cos(0.40))
    chunks.append({"id": "injected", "path": "noise.py",
                   "content": "wedged embedder give up wedged embedder give up",
                   "name": "noise", "start_line": 1, "end_line": 1})
    vecs.append(_vec_for_cos(0.05))
    store.insert_chunks(chunks, vecs)
    return store


def test_hybrid_starved_and_falls_back_to_gated_or(tmp_path):
    """Query where one token ('flurble') matches nothing starves the AND
    branch; pre-fix hybrid collapsed to semantic-only and missed the gold
    chunk (semantic rank 6). The OR fallback must lift it back in."""
    store = _starved_store(tmp_path)
    searcher = Searcher(store, _FakeEmbedder([1.0, 0.0, 0.0, 0.0]))
    results = searcher.hybrid("flurble wedged embedder give up", top_k=3)
    assert results[0]["id"] == "gold", [r["id"] for r in results]
    assert results[0]["_rank_fts"] is not None
    store.close()


def test_hybrid_or_fallback_never_injects_semantically_absent_chunks(tmp_path):
    """Regression guard: a naive fallback that wholesale-replaced fts_results
    with or_results let 'injected' (OR's top hit, no semantic support) tie
    into the top-3. Guarded structurally by chunks_by_id; this test catches
    that."""
    store = _starved_store(tmp_path)
    searcher = Searcher(store, _FakeEmbedder([1.0, 0.0, 0.0, 0.0]))
    results = searcher.hybrid("flurble wedged embedder give up", top_k=3)
    ids = [r["id"] for r in results]
    assert "injected" not in ids, ids
    assert ids == ["gold", "filler1", "filler2"]
    store.close()


def test_hybrid_or_fallback_caps_admitted_or_ranks(tmp_path):
    """Only the OR branch's top OR_FALLBACK_MAX_RANK hits contribute; the
    long tail of weak OR matches is dropped, not down-weighted."""
    store = Store(tmp_path / "test.db")
    n = OR_FALLBACK_MAX_RANK + 2
    chunks, vecs = [], []
    for i in range(n):
        chunks.append({"id": f"c{i:02}", "path": f"c{i:02}.py",
                       "content": "wedged " + f"word{i} " * (i + 1),
                       "name": f"c{i:02}", "start_line": 1, "end_line": 1})
        vecs.append(_vec_for_cos(0.9 - 0.02 * i))
    store.insert_chunks(chunks, vecs)
    searcher = Searcher(store, _FakeEmbedder([1.0, 0.0, 0.0, 0.0]))
    results = searcher.hybrid("flurble wedged", top_k=n)
    with_fts = [r for r in results if r["_rank_fts"] is not None]
    assert len(with_fts) == OR_FALLBACK_MAX_RANK, len(with_fts)
    store.close()


def test_hybrid_healthy_and_path_never_runs_or_fallback(tmp_path):
    """When the AND branch already returns >= top_k results, no second FTS
    query fires."""
    store = Store(tmp_path / "test.db")
    store.insert_chunks(
        [{"id": "c1", "path": "a.py", "content": "wedged embedder",
          "name": "a", "start_line": 1, "end_line": 1}],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    searcher = Searcher(store, _FakeEmbedder([1.0, 0.0, 0.0, 0.0]))
    seen_queries = []
    orig_fts = searcher.fts

    def spy(query, *a, **kw):
        seen_queries.append(query)
        return orig_fts(query, *a, **kw)

    searcher.fts = spy
    # healthy: every token matches -> AND returns 1 >= top_k=1
    searcher.hybrid("wedged embedder", top_k=1)
    assert seen_queries == ['"wedged" "embedder"'], seen_queries
    # starved: 'flurble' matches nothing -> AND empty -> one OR retry
    seen_queries.clear()
    searcher.hybrid("flurble wedged", top_k=1)
    assert seen_queries == ['"flurble" "wedged"', '"flurble" OR "wedged"'], seen_queries
    store.close()


# ---- query_truncated -------------------------------------------------------

def test_query_truncated_false_for_short_query(tmp_path):
    store = Store(tmp_path / "test.db")
    searcher = Searcher(store, _FakeEmbedder([1.0, 0.0, 0.0, 0.0]))
    assert searcher.query_truncated("where is the renderer") is False
    store.close()


def test_query_truncated_true_for_over_budget_query(tmp_path):
    from chonks.embedder import Embedder

    store = Store(tmp_path / "test.db")
    embedder = Embedder("http://x/v1/embeddings", "qwen3-embedding", query_token_budget=3)
    searcher = Searcher(store, embedder)
    assert searcher.query_truncated("a" * 100) is True
    store.close()


def test_query_truncated_uses_embedder_configured_budget(tmp_path):
    from chonks.embedder import Embedder

    store = Store(tmp_path / "test.db")
    embedder = Embedder("http://x/v1/embeddings", "qwen3-embedding", query_token_budget=1000)
    searcher = Searcher(store, embedder)
    # 100 chars is well under a 1000-token (2000-char) budget.
    assert searcher.query_truncated("a" * 100) is False
    store.close()


def test_query_truncated_falls_back_to_default_for_embedder_without_attribute(tmp_path):
    store = Store(tmp_path / "test.db")
    searcher = Searcher(store, _FakeEmbedder([1.0, 0.0, 0.0, 0.0]))
    assert not hasattr(searcher._embedder, "query_token_budget")
    assert searcher.query_truncated("short") is False
    store.close()


# ---- reformulate_query wiring ------------------------------------------------

class _CapturingEmbedder:
    """Records the exact text `embed_queries` is called with, so a test can
    assert whether reformulation reached the embed side."""

    def __init__(self, vec, query_token_budget=None):
        self._vec = vec
        self.seen_queries: list[str] = []
        if query_token_budget is not None:
            self.query_token_budget = query_token_budget

    def embed_queries(self, queries, client, timeout=30.0):
        self.seen_queries.extend(queries)
        return [self._vec for _ in queries]


def test_reformulate_off_by_default_embeds_raw_query(tmp_path):
    store = Store(tmp_path / "test.db")
    embedder = _CapturingEmbedder([1.0, 0.0, 0.0, 0.0])
    searcher = Searcher(store, embedder)
    searcher.embed_query("the getUserById call is slow")
    assert embedder.seen_queries == ["the getUserById call is slow"]
    store.close()


def test_reformulate_on_appends_identifiers_to_embedded_text(tmp_path):
    store = Store(tmp_path / "test.db")
    embedder = _CapturingEmbedder([1.0, 0.0, 0.0, 0.0])
    searcher = Searcher(store, embedder, reformulate_query=True)
    searcher.embed_query("the getUserById call is slow")
    assert len(embedder.seen_queries) == 1
    seen = embedder.seen_queries[0]
    assert seen.startswith("the getUserById call is slow")
    assert "getUserById" in seen
    assert seen != "the getUserById call is slow"
    store.close()


def test_reformulate_on_is_noop_for_identifier_free_prose(tmp_path):
    store = Store(tmp_path / "test.db")
    embedder = _CapturingEmbedder([1.0, 0.0, 0.0, 0.0])
    searcher = Searcher(store, embedder, reformulate_query=True)
    searcher.embed_query("memory climbs over time")
    assert embedder.seen_queries == ["memory climbs over time"]
    store.close()


def test_reformulate_reaches_semantic_search(tmp_path):
    """semantic() (and therefore hybrid()'s semantic branch) must route
    through the same `_embed_query` path `embed_query` uses."""
    store = Store(tmp_path / "test.db")
    store.insert_chunks(
        [{"id": "c1", "path": "pkg/mod.py", "content": "def foo(): pass",
          "name": "foo", "chunk_type": "function_definition",
          "start_line": 1, "end_line": 1}],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    embedder = _CapturingEmbedder([1.0, 0.0, 0.0, 0.0])
    searcher = Searcher(store, embedder, reformulate_query=True)
    searcher.semantic("check my_helper_fn behavior", top_k=5)
    assert "my_helper_fn" in embedder.seen_queries[0]
    store.close()


def test_reformulate_does_not_reach_fts_branch(tmp_path):
    """hybrid()'s FTS branch sanitizes only the raw query; reformulation
    never reaches it. Extracted identifiers are always substrings of the raw
    query, so appending them would add zero new FTS tokens anyway."""
    store = Store(tmp_path / "test.db")
    store.insert_chunks(
        [{"id": "c1", "path": "pkg/mod.py", "content": "def my_helper_fn(): pass",
          "name": "my_helper_fn", "chunk_type": "function_definition",
          "start_line": 1, "end_line": 1}],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    embedder = _CapturingEmbedder([1.0, 0.0, 0.0, 0.0])
    searcher = Searcher(store, embedder, reformulate_query=True)
    seen_fts_queries: list[str] = []
    orig_search_fts = store.search_fts

    def _spy_fts(query, *args, **kwargs):
        seen_fts_queries.append(query)
        return orig_search_fts(query, *args, **kwargs)

    store.search_fts = _spy_fts
    searcher.hybrid("check my_helper_fn behavior", top_k=5)
    assert seen_fts_queries, "expected the FTS branch to run"
    assert "Identifiers:" not in seen_fts_queries[0]
    store.close()


def test_reformulate_survives_query_budget_truncation(tmp_path):
    """On a query long enough to hit the embed truncation cutoff, the
    appended identifier must still land in the embedded text, not get
    sliced off."""
    from chonks.embedder import QUERY_CHARS_PER_TOKEN

    store = Store(tmp_path / "test.db")
    budget_tokens = 50
    budget_chars = int(budget_tokens * QUERY_CHARS_PER_TOKEN)
    embedder = _CapturingEmbedder([1.0, 0.0, 0.0, 0.0], query_token_budget=budget_tokens)
    searcher = Searcher(store, embedder, reformulate_query=True)

    long_prose = "the response gets slower and slower every time. " * 5
    query = long_prose + "check my_distinctive_identifier next"
    assert len(query) > budget_chars, "test query must exceed the embed budget"

    searcher.embed_query(query)
    seen = embedder.seen_queries[0]
    assert len(seen) <= budget_chars
    assert "my_distinctive_identifier" in seen
    store.close()


def test_hybrid_fts_branch_sees_full_query_text_even_when_over_embed_budget(tmp_path):
    """Only the embed side truncates; the FTS branch must see the full
    query text (verified by spying on the fts() call)."""
    from chonks.embedder import Embedder

    store = Store(tmp_path / "test.db")
    store.insert_chunks(
        [{"id": "c1", "path": "a.py", "content": "needle haystack",
          "name": "a", "start_line": 1, "end_line": 1}],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    embedder = Embedder("http://x/v1/embeddings", "qwen3-embedding", query_token_budget=3)
    embedder.embed_queries = lambda texts, client, timeout=30.0: [[1.0, 0.0, 0.0, 0.0]]
    searcher = Searcher(store, embedder)

    long_query = "needle " + "filler " * 50
    seen_queries = []
    orig_fts = searcher.fts

    def spy(query, *a, **kw):
        seen_queries.append(query)
        return orig_fts(query, *a, **kw)

    searcher.fts = spy
    searcher.hybrid(long_query, top_k=5)
    assert seen_queries[0] == _sanitize_fts_query(long_query)
    store.close()
