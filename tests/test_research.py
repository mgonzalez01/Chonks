"""Tests for research candidate pool ranking."""
import numpy as np

from chonks.retrieval.research import _apply_structural_boost, _dedup, _rank_and_cap, _score_candidates


def _make_chunk(id: str, score: float) -> dict:
    return {"id": id, "_score": score}


class _FakeStore:
    """Minimal store stub for the research ranking helpers.

    embeddings: {id: int list} returned as int8 bytes (as chunk_vecs serves them).
    edges:      [(from_id, to_id)] or [(from_id, to_id, edge_type)]. A 2-tuple
                defaults to edge_type "mentions", matching untyped-edge tests
                written before edge_type weighting.
    neighbors:  {id: [(neighbor_id, dist)]} for get_neighbors.
    """
    def __init__(self, embeddings: dict[str, list[int]] | None = None,
                 edges: list[tuple[str, str]] | list[tuple[str, str, str]] | None = None,
                 neighbors: dict[str, list[tuple[str, float]]] | None = None):
        self._embeddings = embeddings or {}
        self._edges = [(e[0], e[1], e[2] if len(e) > 2 else "mentions") for e in (edges or [])]
        self._neighbors = neighbors or {}

    def get_int8_embeddings_by_ids(self, ids):
        return {
            i: np.asarray(self._embeddings[i], dtype=np.int8).tobytes()
            for i in ids if i in self._embeddings
        }

    def get_refs_for_chunks(self, ids):
        s = set(ids)
        return [(f, t) for f, t, _ in self._edges if f in s]

    def get_refs_to_chunks(self, ids):
        s = set(ids)
        return [(f, t) for f, t, _ in self._edges if t in s]

    def get_refs_for_chunks_typed(self, ids):
        s = set(ids)
        return [(f, t, ty) for f, t, ty in self._edges if f in s]

    def get_refs_to_chunks_typed(self, ids):
        s = set(ids)
        return [(f, t, ty) for f, t, ty in self._edges if t in s]

    def get_neighbors(self, cid, limit=None):
        return self._neighbors.get(cid, [])


def test_rank_and_cap_keeps_highest_scored_when_pool_exceeds_cap():
    weak = [_make_chunk(str(i), 0.1) for i in range(5)]
    strong = [_make_chunk("strong_1", 0.9), _make_chunk("strong_2", 0.8)]
    all_candidates = weak + strong

    result = _rank_and_cap(all_candidates, max_candidates=5)

    ids = [c["id"] for c in result]
    assert "strong_1" in ids, "highest-scored expansion hit should survive the cap"
    assert "strong_2" in ids, "second-highest expansion hit should survive the cap"


def test_rank_and_cap_result_is_sorted_descending():
    chunks = [_make_chunk("a", 0.3), _make_chunk("b", 0.9), _make_chunk("c", 0.5)]
    result = _rank_and_cap(chunks, max_candidates=3)
    scores = [c["_score"] for c in result]
    assert scores == sorted(scores, reverse=True)


def test_rank_and_cap_deduplicates():
    chunks = [_make_chunk("x", 0.5), _make_chunk("x", 0.5), _make_chunk("y", 0.3)]
    result = _rank_and_cap(chunks, max_candidates=10)
    assert len(result) == 2


def test_rank_and_cap_respects_cap():
    chunks = [_make_chunk(str(i), float(i) / 10) for i in range(20)]
    result = _rank_and_cap(chunks, max_candidates=7)
    assert len(result) == 7


# --- _score_candidates: the seed-eviction fix --------------------------------

def test_candidate_scored_by_query_cosine_from_embedding():
    query = [1.0, 0.0, 0.0]
    aligned = {"id": "a"}      # cosine 1.0
    orthogonal = {"id": "o"}   # cosine 0.0
    opposite = {"id": "p"}     # cosine -1.0
    store = _FakeStore({"a": [127, 0, 0], "o": [0, 127, 0], "p": [-127, 0, 0]})
    _score_candidates([aligned, orthogonal, opposite], query, store)
    assert abs(aligned["_score"] - 1.0) < 1e-3
    assert abs(orthogonal["_score"] - 0.0) < 1e-3
    assert abs(opposite["_score"] + 1.0) < 1e-3


def test_vec0_distance_field_is_ignored_for_scoring():
    # Regression: scoring used to read the raw int8 L2 'distance' from vec0
    # through a cosine-shaped transform, pinning the seed's score near 0.
    query = [1.0, 0.0, 0.0]
    seed = {"id": "seed", "distance": 92.0}
    store = _FakeStore({"seed": [127, 0, 0]})
    _score_candidates([seed], query, store)
    assert abs(seed["_score"] - 1.0) < 1e-3


def test_strongest_seed_is_not_evicted_by_expansion():
    # Regression: seeds used to score ~0 (L2 distance through a cosine-shaped
    # transform) while expansion got a flat 0.5, so _rank_and_cap dropped
    # the real seeds.
    query = [1.0, 0.0, 0.0]
    seed = {"id": "seed", "distance": 90.0}
    expansion = [{"id": f"exp{i}", "_origin": "graph"} for i in range(10)]
    store = _FakeStore({"seed": [127, 0, 0],
                        **{f"exp{i}": [80, 90, 0] for i in range(10)}})
    candidates = [seed] + expansion

    _score_candidates(candidates, query, store)
    assert seed["_score"] > expansion[0]["_score"]

    ranked = _rank_and_cap(candidates, max_candidates=5)
    assert ranked[0]["id"] == "seed"


def test_relevant_expansion_can_outrank_a_weak_seed():
    # Symmetric guard: the fix must not bury all expansion, a genuinely
    # relevant expanded chunk should still beat a weak seed.
    query = [1.0, 0.0, 0.0]
    seed = {"id": "seed", "distance": 90.0}
    exp = {"id": "exp", "_origin": "graph"}
    store = _FakeStore({"seed": [0, 127, 0], "exp": [127, 0, 0]})
    _score_candidates([seed, exp], query, store)
    assert exp["_score"] > seed["_score"]


def test_missing_embedding_scores_zero():
    exp = {"id": "exp", "_origin": "graph"}
    _score_candidates([exp], [1.0, 0.0, 0.0], _FakeStore({}))  # no embedding available
    assert exp["_score"] == 0.0


def test_no_query_vector_scores_everything_zero():
    # Embedder down, query_vec None: pool degrades to insertion order
    # rather than a poisoned ranking.
    seed = {"id": "seed", "distance": 90.0}
    exp = {"id": "exp", "_origin": "graph"}
    _score_candidates([seed, exp], None, _FakeStore({"seed": [127, 0, 0], "exp": [127, 0, 0]}))
    assert seed["_score"] == 0.0
    assert exp["_score"] == 0.0


def test_already_scored_candidates_untouched():
    c = {"id": "x", "_score": 0.42}
    _score_candidates([c], [1.0, 0.0, 0.0], _FakeStore({"x": [127, 0, 0]}))
    assert c["_score"] == 0.42


# --- _apply_structural_boost: exploration ranking ----------------------------

def test_structural_boost_lifts_seed_adjacent_neighbour_over_generic():
    # A glue neighbour (lower query cosine, wired to the anchor) must
    # outrank a generic high-cosine chunk that just echoes the query.
    anchor = {"id": "A", "_score": 0.70}
    neighbour = {"id": "N", "_score": 0.55}
    generic = {"id": "G", "_score": 0.65}
    out = _apply_structural_boost([anchor, neighbour, generic],
                                  _FakeStore(edges=[("A", "N")]), beta=0.5, seed_n=30)
    ids = [c["id"] for c in out]
    assert ids.index("N") < ids.index("G")
    assert abs(next(c["_score"] for c in out if c["id"] == "N") - (0.55 + 0.5 * 0.70)) < 1e-9


def test_structural_boost_counts_incoming_edges():
    anchor = {"id": "A", "_score": 0.70}
    caller = {"id": "C", "_score": 0.50}
    out = _apply_structural_boost([anchor, caller],
                                  _FakeStore(edges=[("C", "A")]), beta=0.5, seed_n=30)
    assert abs(next(c["_score"] for c in out if c["id"] == "C") - (0.50 + 0.5 * 0.70)) < 1e-9


def test_structural_boost_counts_knn_neighbours():
    anchor = {"id": "A", "_score": 0.70}
    kin = {"id": "K", "_score": 0.40}
    out = _apply_structural_boost([anchor, kin],
                                  _FakeStore(neighbors={"A": [("K", 0.1)]}), beta=0.5, seed_n=30)
    assert abs(next(c["_score"] for c in out if c["id"] == "K") - (0.40 + 0.5 * 0.70)) < 1e-9


def test_structural_boost_only_top_seeds_donate():
    # Edges from a non-seed (outside top-seed_n) must not boost anything.
    anchor = {"id": "A", "_score": 0.70}
    low = {"id": "L", "_score": 0.10}
    other = {"id": "O", "_score": 0.20}
    out = _apply_structural_boost([anchor, low, other],
                                  _FakeStore(edges=[("L", "O")]), beta=0.5, seed_n=1)
    assert next(c["_score"] for c in out if c["id"] == "O") == 0.20


def test_structural_boost_noop_when_beta_zero():
    a = {"id": "A", "_score": 0.3}
    b = {"id": "B", "_score": 0.9}
    out = _apply_structural_boost([a, b], _FakeStore(edges=[("B", "A")]), beta=0.0, seed_n=30)
    assert [c["id"] for c in out] == ["B", "A"]
    assert a["_score"] == 0.3 and b["_score"] == 0.9


# --- _apply_structural_boost: edge_type weighting ---------------------------

def test_structural_boost_scales_by_edge_type_weight():
    """The proximity bump on a chunk_refs edge is scaled by that edge_type's
    configured weight: a 'calls' edge with weight 10 contributes 10x what
    it would at the default weight of 1.0."""
    anchor = {"id": "A", "_score": 0.70}
    callee = {"id": "N", "_score": 0.55}
    out = _apply_structural_boost(
        [anchor, callee], _FakeStore(edges=[("A", "N", "calls")]),
        beta=0.5, seed_n=30, edge_type_weights={"calls": 10.0, "mentions": 1.0},
    )
    boosted = next(c["_score"] for c in out if c["id"] == "N")
    assert abs(boosted - (0.55 + 0.5 * 10.0 * 0.70)) < 1e-9


def test_structural_boost_default_weights_match_unweighted_behaviour():
    """Omitting edge_type_weights (None) must be bit-identical to explicitly
    passing all-1.0 weights."""
    anchor = {"id": "A", "_score": 0.70}
    neighbour = {"id": "N", "_score": 0.55}
    baseline = _apply_structural_boost(
        [dict(anchor), dict(neighbour)], _FakeStore(edges=[("A", "N", "calls")]),
        beta=0.5, seed_n=30,
    )
    weighted = _apply_structural_boost(
        [dict(anchor), dict(neighbour)], _FakeStore(edges=[("A", "N", "calls")]),
        beta=0.5, seed_n=30,
        edge_type_weights={"calls": 1.0, "imports": 1.0, "inherits": 1.0,
                            "xlang": 1.0, "mentions": 1.0},
    )
    assert [c["_score"] for c in baseline] == [c["_score"] for c in weighted]


def test_structural_boost_paired_adjacency_only_when_enabled():
    """A candidate in a seed's paired companion file gets the seed's proximity
    (weight 1.0, like semantic), but only under expand_paired_files: without
    it, paired-expanded headers are admitted but buried at selection time."""
    anchor = {"id": "A", "_score": 0.70, "path": "m/sky.cpp"}
    header = {"id": "H", "_score": 0.30, "path": "m/sky.h"}

    store = _FakeStore()
    store.get_paired_files = lambda paths: (
        {"m/sky.cpp": ["m/sky.h"]} if "m/sky.cpp" in paths else {})
    out = _apply_structural_boost([anchor, header], store, beta=0.5, seed_n=30,
                                  expand_paired_files=True)
    h = next(c for c in out if c["id"] == "H")
    assert abs(h["_score"] - (0.30 + 0.5 * 0.70)) < 1e-9
    assert abs(h["_struct_boost"] - 0.70) < 1e-9

    # Flag off (default): must never call get_paired_files.
    store2 = _FakeStore()
    def _boom(paths):
        raise AssertionError("get_paired_files called with flag off")
    store2.get_paired_files = _boom
    anchor2 = {"id": "A", "_score": 0.70, "path": "m/sky.cpp"}
    header2 = {"id": "H", "_score": 0.30, "path": "m/sky.h"}
    out2 = _apply_structural_boost([anchor2, header2], store2, beta=0.5, seed_n=30)
    h2 = next(c for c in out2 if c["id"] == "H")
    assert h2["_score"] == 0.30 and h2["_struct_boost"] == 0.0


def test_paired_trigger_is_file_level_not_chunk_level():
    """A partner file prominent by FILE rank but whose best chunk is below
    the chunk-seed line must still trigger pairing: both expansion trigger
    and boost donation collapse to top-N files.

    Two high chunks from file X push P's only chunk to chunk-rank 3; with
    seeds_per_iter=2 a chunk-level trigger would never see P.
    """
    from chonks.retrieval.research import _graph_expand
    x1 = {"id": "x1", "_score": 0.9, "path": "m/x.cpp", "name": "foo"}
    x2 = {"id": "x2", "_score": 0.8, "path": "m/x.cpp", "name": "bar"}
    p1 = {"id": "p1", "_score": 0.5, "path": "m/p.cpp", "name": "baz"}
    store = _FakeStore()
    calls = []
    store.get_paired_files = lambda paths: (calls.append(sorted(paths)) or
                                            {"m/p.cpp": ["m/p.h"]})
    store.get_chunks_by_path_and_names = lambda path, names, limit=None: (
        [{"id": "ph1", "path": "m/p.h", "name": "baz"}] if path == "m/p.h" else [])
    store.get_chunks_by_ids = lambda ids: [
        {"id": i, "path": "m/p.h", "name": "baz"} for i in ids]
    out = _graph_expand(store, [x1, x2, p1], seeds_per_iter=2,
                        neighbours_per_seed=0, path_prefix=None,
                        expand_paired_files=True)
    assert calls and "m/p.cpp" in calls[0]
    assert any(c["id"] == "ph1" for c in out)


# --- _apply_structural_boost: _struct_anchor donor bookkeeping -------------

def test_structural_boost_sets_struct_anchor_to_argmax_donor():
    """A candidate bumped by two seeds keeps _struct_anchor pointing at the
    donor that produced the argmax proximity."""
    weak_anchor = {"id": "A", "_score": 0.30}
    strong_anchor = {"id": "B", "_score": 0.70}
    neighbour = {"id": "N", "_score": 0.10}
    out = _apply_structural_boost(
        [weak_anchor, strong_anchor, neighbour],
        _FakeStore(edges=[("A", "N", "calls"), ("B", "N", "calls")]),
        beta=0.5, seed_n=30,
    )
    n = next(c for c in out if c["id"] == "N")
    assert n["_struct_anchor"] == {"anchor_id": "B", "edge_type": "calls"}


def test_structural_boost_never_overwrites_existing_evidence():
    """A candidate that already carries _evidence (set by _graph_expand) must
    not get a _struct_anchor even when it also gets a structural bump.
    _evidence is the authoritative admission record."""
    anchor = {"id": "A", "_score": 0.70}
    graph_hit = {"id": "N", "_score": 0.10,
                 "_evidence": {"origin": "graph", "anchor_id": "A", "edge_type": "calls"}}
    out = _apply_structural_boost(
        [anchor, graph_hit], _FakeStore(edges=[("A", "N", "calls")]),
        beta=0.5, seed_n=30,
    )
    n = next(c for c in out if c["id"] == "N")
    assert "_struct_anchor" not in n
    assert n["_evidence"] == {"origin": "graph", "anchor_id": "A", "edge_type": "calls"}


def test_structural_boost_no_struct_anchor_when_no_boost():
    """A candidate with no structural adjacency gets neither a boost nor a
    _struct_anchor key."""
    anchor = {"id": "A", "_score": 0.70}
    isolated = {"id": "I", "_score": 0.20}
    out = _apply_structural_boost([anchor, isolated], _FakeStore(), beta=0.5, seed_n=30)
    i = next(c for c in out if c["id"] == "I")
    assert "_struct_anchor" not in i


def test_structural_boost_low_weight_mentions_edge_contributes_less():
    """A down-weighted 'mentions' edge contributes less boost than the same
    edge at weight 1.0: the effect the sweep is meant to tune."""
    anchor = {"id": "A", "_score": 0.70}
    neighbour = {"id": "N", "_score": 0.55}
    out = _apply_structural_boost(
        [anchor, neighbour], _FakeStore(edges=[("A", "N", "mentions")]),
        beta=0.5, seed_n=30, edge_type_weights={"calls": 10.0, "mentions": 0.1},
    )
    boosted = next(c["_score"] for c in out if c["id"] == "N")
    assert abs(boosted - (0.55 + 0.5 * 0.1 * 0.70)) < 1e-9


# --- _graph_expand: edge_type weight gate ------------------------------------

def test_graph_expand_excludes_zero_weighted_edge_type():
    from chonks.retrieval.research import _graph_expand
    seeds = [{"id": "A", "_score": 0.9}]
    store = _FakeStore(edges=[("A", "N", "mentions")])
    out = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=5,
                         path_prefix=None, edge_type_weights={"mentions": 0.0})
    assert out == []


def test_graph_expand_includes_edge_when_weight_positive_or_default():
    from chonks.retrieval.research import _graph_expand

    class _StoreWithChunks(_FakeStore):
        def get_chunks_by_ids(self, ids):
            return [{"id": i, "path": "x.py"} for i in ids]

    seeds = [{"id": "A", "_score": 0.9}]
    store = _StoreWithChunks(edges=[("A", "N", "mentions")])
    out_default = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=5,
                                 path_prefix=None)
    out_positive = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=5,
                                  path_prefix=None, edge_type_weights={"mentions": 1.0})
    assert [c["id"] for c in out_default] == ["N"]
    assert [c["id"] for c in out_positive] == ["N"]


# --- _graph_expand: evidence trail --------------------------------------------

def test_graph_expand_attaches_evidence_for_structural_admission():
    """A chunk pulled in via a typed chunk_refs edge carries _evidence with
    origin="graph", the admitting seed as anchor_id, and the edge_type."""
    from chonks.retrieval.research import _graph_expand

    class _StoreWithChunks(_FakeStore):
        def get_chunks_by_ids(self, ids):
            return [{"id": i, "path": "x.py"} for i in ids]

    seeds = [{"id": "A", "_score": 0.9}]
    store = _StoreWithChunks(edges=[("A", "N", "calls")])
    out = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=0,
                         path_prefix=None)
    assert len(out) == 1
    assert out[0]["_evidence"] == {"origin": "graph", "anchor_id": "A", "edge_type": "calls"}


def test_graph_expand_attaches_evidence_for_semantic_admission():
    """A chunk pulled in via semantic k-NN carries _evidence with
    origin="semantic" and the admitting seed as anchor_id, no edge_type."""
    from chonks.retrieval.research import _graph_expand

    class _StoreWithChunks(_FakeStore):
        def get_chunks_by_ids(self, ids):
            return [{"id": i, "path": "x.py"} for i in ids]

    seeds = [{"id": "A", "_score": 0.9}]
    store = _StoreWithChunks(neighbors={"A": [("K", 0.1)]})
    out = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=5,
                         path_prefix=None)
    assert len(out) == 1
    assert out[0]["_evidence"] == {"origin": "semantic", "anchor_id": "A"}


def test_graph_expand_evidence_first_writer_wins():
    """A chunk reachable both by semantic k-NN and a structural edge from the
    same seed set keeps whichever route admitted it first (semantic runs
    before the structural loop in _graph_expand)."""
    from chonks.retrieval.research import _graph_expand

    class _StoreWithChunks(_FakeStore):
        def get_chunks_by_ids(self, ids):
            return [{"id": i, "path": "x.py"} for i in ids]

    seeds = [{"id": "A", "_score": 0.9}]
    store = _StoreWithChunks(edges=[("A", "N", "calls")], neighbors={"A": [("N", 0.1)]})
    out = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=5,
                         path_prefix=None)
    assert len(out) == 1
    assert out[0]["_evidence"]["origin"] == "semantic"


def test_graph_expand_paired_evidence_uses_seed_anchor():
    """A companion chunk admitted via paired-file expansion carries
    origin="paired" with anchor_id set to a seed chunk id of the paired
    path."""
    from chonks.retrieval.research import _graph_expand
    seeds = [{"id": "A", "_score": 0.9, "path": "widget.h", "name": "Widget"}]
    store = _PairedStore(
        paired={"widget.h": ["widget.cpp"]},
        by_name={("widget.cpp", "Widget"): "impl_chunk"},
    )
    out = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=5,
                         path_prefix=None, expand_paired_files=True)
    assert len(out) == 1
    assert out[0]["_evidence"] == {"origin": "paired", "anchor_id": "A", "edge_type": "paired"}


def test_graph_expand_normalizes_path_prefix_like_search_semantic():
    """Regression: _graph_expand used to scope expanded chunks with a raw
    startswith(path_prefix), while Store.search_semantic (scoping the seed
    chunks) normalizes by stripping trailing slashes before a LIKE match.
    Both must apply the same rstrip('/\\') so expansion scope agrees with
    seed scope, e.g. for prefix 'foo/' against a path like 'foobar.py'."""
    from chonks.retrieval.research import _graph_expand

    class _StoreWithChunks(_FakeStore):
        def get_chunks_by_ids(self, ids):
            return [{"id": i, "path": "foobar.py"} for i in ids]

    seeds = [{"id": "A", "_score": 0.9}]
    store = _StoreWithChunks(edges=[("A", "N", "mentions")])

    out = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=5,
                         path_prefix="foo/")

    assert [c["id"] for c in out] == ["N"], (
        "expansion scope diverged from search_semantic's normalized prefix "
        f"match: {out}"
    )


def test_graph_seed_confidence_gate_filters_low_scorers(monkeypatch):
    """graph_seed_min_rel_score must drop low-scoring candidates from the
    expansion seed set only (the pool itself is untouched). Gate off (0.0)
    passes everything."""
    import chonks.retrieval.research as research

    captured = {}
    def fake_expand(store, seed_chunks, **kw):
        captured["seeds"] = [c["id"] for c in seed_chunks]
        return []
    monkeypatch.setattr(research, "_graph_expand", fake_expand)
    monkeypatch.setattr(research, "_score_candidates", lambda *a, **k: None)
    monkeypatch.setattr(research, "_extract_symbols", lambda c: [])
    monkeypatch.setattr(research, "_apply_structural_boost",
                        lambda cands, store, **kw: cands)

    class FakeSearcher:
        class store:
            @staticmethod
            def get_refs_for_chunks_typed(ids):
                return []
        def embed_query(self, q):
            return [0.0]
        def semantic(self, q, k, p):
            return [
                {"id": "hi", "_score": 0.8, "content": "", "path": "a"},
                {"id": "lo", "_score": 0.3, "content": "", "path": "b"},
            ]
        def regex(self, *a, **k):
            return []

    for thr, expected in [(0.0, {"hi", "lo"}), (0.7, {"hi"})]:
        captured.clear()
        research.deep_research(
            "q", FakeSearcher(),
            cfg={"max_iterations": 1, "graph_seed_min_rel_score": thr},
        )
        assert set(captured["seeds"]) == expected, (thr, captured["seeds"])


# --- _graph_expand: header/impl paired-file expansion -----------------------

class _PairedStore(_FakeStore):
    """_FakeStore plus the lookups paired-file expansion needs.

    paired: {path: [companion_path, ...]} for get_paired_files.
    by_name: {(path, name): chunk_id} for get_chunks_by_path_and_names; a
             name not present yields no rows, exercising the fallback path.
    top_pagerank: {path: chunk_id} for get_top_pagerank_chunk_for_path.
    """
    def __init__(self, *a, paired=None, by_name=None, top_pagerank=None, **kw):
        super().__init__(*a, **kw)
        self._paired = paired or {}
        self._by_name = by_name or {}
        self._top_pagerank = top_pagerank or {}
        self.paired_calls = 0

    def get_paired_files(self, paths):
        self.paired_calls += 1
        return {p: self._paired[p] for p in paths if p in self._paired}

    def get_chunks_by_path_and_names(self, path, names, limit=None):
        hits = [
            {"id": cid, "path": path}
            for name in names
            if (cid := self._by_name.get((path, name))) is not None
        ]
        return hits[:limit] if limit is not None else hits

    def get_top_pagerank_chunk_for_path(self, path):
        cid = self._top_pagerank.get(path)
        return {"id": cid, "path": path} if cid is not None else None

    def get_chunks_by_ids(self, ids):
        return [{"id": i, "path": "expanded"} for i in ids]


def test_graph_expand_paired_files_off_by_default_no_call():
    """expand_paired_files defaults to False: get_paired_files must never
    be called and no paired chunk enters the pool."""
    from chonks.retrieval.research import _graph_expand
    seeds = [{"id": "A", "_score": 0.9, "path": "widget.h", "name": "Widget"}]
    store = _PairedStore(
        paired={"widget.h": ["widget.cpp"]},
        by_name={("widget.cpp", "Widget"): "impl_chunk"},
    )
    out = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=5,
                         path_prefix=None)
    assert out == []
    assert store.paired_calls == 0


def test_graph_expand_paired_files_same_name_match():
    """expand_paired_files=True: a companion chunk sharing the seed's name
    enters the expansion pool."""
    from chonks.retrieval.research import _graph_expand
    seeds = [{"id": "A", "_score": 0.9, "path": "widget.h", "name": "Widget"}]
    store = _PairedStore(
        paired={"widget.h": ["widget.cpp"]},
        by_name={("widget.cpp", "Widget"): "impl_chunk"},
    )
    out = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=5,
                         path_prefix=None, expand_paired_files=True)
    assert store.paired_calls == 1
    assert [c["id"] for c in out] == ["impl_chunk"]


def test_graph_expand_paired_files_fallback_to_top_pagerank():
    """No name match in the companion -> fall back to its single
    highest-PageRank chunk."""
    from chonks.retrieval.research import _graph_expand
    seeds = [{"id": "A", "_score": 0.9, "path": "widget.h", "name": "Widget"}]
    store = _PairedStore(
        paired={"widget.h": ["widget.cpp"]},
        by_name={},
        top_pagerank={"widget.cpp": "top_chunk"},
    )
    out = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=5,
                         path_prefix=None, expand_paired_files=True)
    assert [c["id"] for c in out] == ["top_chunk"]


def test_graph_expand_paired_files_cap_respected():
    """Companion contributes at most PAIRED_CHUNKS_PER_FILE chunks."""
    from chonks.retrieval.research import PAIRED_CHUNKS_PER_FILE, _graph_expand
    seed_names = [f"Sym{i}" for i in range(PAIRED_CHUNKS_PER_FILE + 3)]
    seeds = [{"id": "A", "_score": 0.9, "path": "widget.h", "name": seed_names[0]}]
    seeds = [
        {"id": f"A{i}", "_score": 0.9, "path": "widget.h", "name": n}
        for i, n in enumerate(seed_names)
    ]
    by_name = {("widget.cpp", n): f"impl_{n}" for n in seed_names}
    store = _PairedStore(paired={"widget.h": ["widget.cpp"]}, by_name=by_name)
    out = _graph_expand(store, seeds, seeds_per_iter=20, neighbours_per_seed=5,
                         path_prefix=None, expand_paired_files=True)
    assert len(out) == PAIRED_CHUNKS_PER_FILE


# --- _select_final_interleave: anti-burial selector --------------------------

def _c(id, path, score, boost=0.0):
    return {"id": id, "path": path, "_score": score, "_struct_boost": boost}


def test_interleave_file_cap_frees_slots_for_buried_connected_file():
    """Lever (b): a wall of one file's near-duplicate chunks is capped so a
    graph-connected file sitting just past top_k is lifted into the window."""
    from chonks.retrieval.research import _select_final_interleave
    wall = [_c(f"w{i}", "wall.py", 0.9 - i * 0.01) for i in range(5)]
    connected = _c("g", "gold.py", 0.4, boost=0.5)
    cands = wall + [connected]
    out = _select_final_interleave(cands, top_k=3, reserved_n=0, file_cap=2)
    paths = [c["path"] for c in out]
    assert paths.count("wall.py") == 2
    assert "gold.py" in paths


def test_interleave_reserved_rescues_deeply_buried_connected_file():
    """Lever (a): a structurally-connected chunk far below the cutoff is
    appended into the window's slack when the cap alone can't reach it.
    Rescues fill slack, never displace a found file, so the window needs
    room (fewer distinct files than top_k)."""
    from chonks.retrieval.research import _select_final_interleave
    filler = [_c(f"f{i}_{j}", f"f{i}.py", 0.9 - (i * 2 + j) * 0.01)
              for i in range(3) for j in range(2)]
    deep = _c("d", "deep.py", 0.1, boost=0.8)
    cands = filler + [deep]
    top = [c["path"] for c in _select_final_interleave(cands, top_k=5, reserved_n=0, file_cap=3)]
    assert "deep.py" not in top
    with_res = [c["path"] for c in _select_final_interleave(cands, top_k=5, reserved_n=5, file_cap=3)]
    assert "deep.py" in with_res


def test_interleave_never_demotes_a_file_already_in_window():
    """Non-regression guarantee: every first-occurrence file keeps its rank
    relative to the others; rescues can only append new files at the tail."""
    from chonks.retrieval.research import _select_final_interleave
    a = _c("a", "a.py", 0.9)
    b = _c("b", "b.py", 0.8)
    c = _c("c", "c.py", 0.7)
    rescue = _c("r", "r.py", 0.05, boost=0.9)
    out = _select_final_interleave([a, b, c, rescue], top_k=4, reserved_n=5, file_cap=3)
    order = [x["path"] for x in out]
    assert order.index("a.py") < order.index("b.py") < order.index("c.py") < order.index("r.py")


def test_interleave_returns_natural_order_when_levers_off():
    from chonks.retrieval.research import _select_final_interleave
    cands = [_c("a", "a.py", 0.9), _c("b", "a.py", 0.8), _c("c", "b.py", 0.7)]
    out = _select_final_interleave(cands, top_k=3, reserved_n=0, file_cap=0)
    assert [x["id"] for x in out] == ["a", "b", "c"]


# --- _result_connections: typed subgraph among returned chunks -------------

def test_result_connections_includes_edge_with_both_endpoints_returned():
    from chonks.retrieval.research import _result_connections
    store = _FakeStore(edges=[("A", "B", "calls")])
    out = _result_connections(store, ["A", "B"])
    assert out == [{"from_id": "A", "to_id": "B", "edge_type": "calls", "provenance": "extracted"}]


def test_result_connections_excludes_edge_to_a_chunk_outside_the_returned_set():
    from chonks.retrieval.research import _result_connections
    store = _FakeStore(edges=[("A", "B", "calls"), ("A", "OUTSIDE", "calls")])
    out = _result_connections(store, ["A", "B"])
    assert out == [{"from_id": "A", "to_id": "B", "edge_type": "calls", "provenance": "extracted"}]


def test_result_connections_dedupes_same_pair_deterministically():
    from chonks.retrieval.research import _result_connections

    class _MultiEdgeStore(_FakeStore):
        def get_refs_for_chunks_typed(self, ids):
            return [("A", "B", "mentions"), ("A", "B", "calls")]

    out = _MultiEdgeStore()
    result = _result_connections(out, ["A", "B"])
    # sorted by (from_id, to_id, edge_type) before dedup: "calls" < "mentions"
    assert result == [{"from_id": "A", "to_id": "B", "edge_type": "calls", "provenance": "extracted"}]


def test_result_connections_caps_after_dedupe():
    from chonks.retrieval.research import _result_connections

    class _ManyEdgesStore(_FakeStore):
        def get_refs_for_chunks_typed(self, ids):
            return [(f"n{i}", f"n{i+1}", "calls") for i in range(10)]

    ids = [f"n{i}" for i in range(11)]
    out = _result_connections(_ManyEdgesStore(), ids, cap=3)
    assert len(out) == 3
    assert out == sorted(out, key=lambda e: (e["from_id"], e["to_id"]))


def test_result_connections_empty_ids_returns_empty():
    from chonks.retrieval.research import _result_connections
    out = _result_connections(_FakeStore(edges=[("A", "B", "calls")]), [])
    assert out == []


# --- degraded-state signalling ----------------------------------------------

def test_deep_research_seeds_from_keywords_when_the_embedder_is_down(monkeypatch):
    import chonks.retrieval.research as research

    monkeypatch.setattr(research, "_graph_expand", lambda store, seeds, **kw: [])
    monkeypatch.setattr(research, "_extract_symbols", lambda c: [])
    monkeypatch.setattr(research, "_apply_structural_boost",
                         lambda cands, store, **kw: cands)
    monkeypatch.setattr(research, "_result_connections", lambda store, ids, **kw: [])

    class FakeSearcher:
        class store:
            @staticmethod
            def get_refs_for_chunks_typed(ids):
                return []
        def embed_query(self, q):
            raise ConnectionError("embedder unreachable")
        def semantic(self, q, k, p):
            raise ConnectionError("embedder unreachable")
        def keyword(self, q, k, p):
            return [{"id": "hi", "content": "", "path": "a"}]
        def regex(self, *a, **k):
            return []

    result = research.deep_research(
        "q", FakeSearcher(), cfg={"max_iterations": 1},
    )
    assert result["degraded"] == "semantic_unavailable"
    assert [c["id"] for c in result["chunks"]] == ["hi"]


def test_deep_research_degraded_none_when_query_embed_succeeds():
    """Healthy path: degraded is the explicit not-degraded value (None),
    not merely absent from the payload."""
    import chonks.retrieval.research as research

    class FakeSearcher:
        class store:
            @staticmethod
            def get_refs_for_chunks_typed(ids):
                return []
        def embed_query(self, q):
            return [1.0, 0.0]
        def semantic(self, q, k, p):
            return [{"id": "hi", "_score": 0.8, "content": "", "path": "a"}]
        def regex(self, *a, **k):
            return []

    import unittest.mock as mock
    with mock.patch.object(research, "_graph_expand", lambda store, seeds, **kw: []), \
         mock.patch.object(research, "_extract_symbols", lambda c: []), \
         mock.patch.object(research, "_apply_structural_boost", lambda cands, store, **kw: cands), \
         mock.patch.object(research, "_result_connections", lambda store, ids, **kw: []):
        result = research.deep_research(
            "q", FakeSearcher(), cfg={"max_iterations": 1},
        )
    assert result["degraded"] is None
    assert "degraded" in result


# --- compact output ----------------------------------------------------------

def test_deep_research_compact_drops_content_only_from_the_returned_chunks(monkeypatch):
    import chonks.retrieval.research as research

    monkeypatch.setattr(research, "_graph_expand", lambda store, seeds, **kw: [])
    monkeypatch.setattr(research, "_apply_structural_boost", lambda cands, store, **kw: cands)
    regex_calls = []

    class FakeSearcher:
        store = _FakeStore(edges=[("draw", "flush", "calls")])
        def embed_query(self, q):
            return [1.0, 0.0]
        def semantic(self, q, k, p):
            return [
                {"id": "draw", "_score": 0.9, "path": "servers/rendering/canvas.cpp", "name": "draw",
                 "start_line": 10, "end_line": 30, "content": "void draw() { RenderingServer::flush(); }"},
                {"id": "flush", "_score": 0.8, "path": "servers/rendering/server.cpp", "name": "flush",
                 "start_line": 5, "end_line": 9, "content": "void flush() {}"},
            ]
        def regex(self, sym, top_k=10, path_prefix=None):
            regex_calls.append(sym)
            return []

    full = research.deep_research("q", FakeSearcher(), cfg={"max_iterations": 1})
    full_symbols = list(regex_calls)
    regex_calls.clear()
    compact = research.deep_research("q", FakeSearcher(), cfg={"max_iterations": 1}, compact=True)

    assert regex_calls == full_symbols
    assert "RenderingServer::flush" in regex_calls
    assert all(c["content"] for c in full["chunks"])
    assert compact["chunks"] == [{k: v for k, v in c.items() if k != "content"} for c in full["chunks"]]
    assert compact["connections"] == full["connections"] != []
