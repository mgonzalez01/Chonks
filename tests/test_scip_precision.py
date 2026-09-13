"""Pure-logic tests over hand-built SCIP-shaped data, plus one end-to-end pass
over a real scip-python fixture (tests/data/scip_precision/), checked-in
`scip print --json` output from a 2-file toy project. No network or CLI calls;
scip_precision.py's toolchain-fetch code is exercised by hand, not here."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import scip_precision as sp  # noqa: E402

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "scip_precision")


# ----------------------------------------------------------------------
# line conversion / local-symbol detection
# ----------------------------------------------------------------------

def test_scip_line_to_chunk_line_converts_0_based_to_1_based():
    assert sp.scip_line_to_chunk_line(0) == 1
    assert sp.scip_line_to_chunk_line(5) == 6


def test_is_local_symbol_requires_exact_prefix():
    assert sp.is_local_symbol("local 0")
    assert sp.is_local_symbol("local 42")
    assert not sp.is_local_symbol("localthing")  # real symbol fragment, not a local marker
    assert not sp.is_local_symbol("scip-python python pkg 0.0.1 a/helper().")


# ----------------------------------------------------------------------
# iter_occurrences / classify_occurrences on hand-built SCIP JSON
# ----------------------------------------------------------------------

def _occ(line0, symbol, is_def, end_line0=None):
    rng = [line0, 0, 1] if end_line0 is None else [line0, 0, end_line0, 1]
    return {"range": rng, "symbol": symbol, "symbol_roles": 1 if is_def else 8}


def test_iter_occurrences_extracts_path_line_symbol_roles_incl_multiline_range():
    scip_json = {
        "documents": [
            {"relative_path": "a.py", "occurrences": [
                _occ(0, "sym-a", is_def=True),
                _occ(2, "sym-a", is_def=False),
            ]},
            {"relative_path": "b.py", "occurrences": [
                _occ(9, "sym-b", is_def=True, end_line0=11),  # 4-element multi-line range
            ]},
        ]
    }
    out = list(sp.iter_occurrences(scip_json))
    assert out == [
        ("a.py", 1, "sym-a", True),
        ("a.py", 3, "sym-a", False),
        ("b.py", 10, "sym-b", True),  # range[0] still the line even for a 4-element range
    ]


def test_classify_occurrences_splits_local_defs_and_references():
    occurrences = [
        ("a.py", 1, "sym-a", True),
        ("a.py", 5, "sym-a", True),   # 2nd definition site -> fan-out
        ("b.py", 1, "sym-a", False),
        ("b.py", 2, "local 0", True),
        ("b.py", 2, "local 0", False),
        ("b.py", 3, "sym-external", False),  # never defined anywhere -> stays external in GT stage
    ]
    definitions, reference_sites, excluded_local = sp.classify_occurrences(occurrences)
    assert definitions == {"sym-a": [("a.py", 1), ("a.py", 5)]}
    assert reference_sites == [("b.py", 1, "sym-a"), ("b.py", 3, "sym-external")]
    assert excluded_local == 2


# ----------------------------------------------------------------------
# ChunkMapper: unindexed file / no covering chunk / multi-chunk overlap /
# memoized exclusion counting
# ----------------------------------------------------------------------

def _mapper(rows):
    return sp.ChunkMapper(sp.build_chunk_index(rows))


def test_chunk_mapper_resolves_line_within_a_single_chunk():
    m = _mapper([("c1", "a.py", 1, 5), ("c2", "a.py", 6, 10)])
    assert m.resolve("a.py", 3) == "c1"
    assert m.resolve("a.py", 7) == "c2"
    assert m.excluded_unindexed_file == 0
    assert m.excluded_no_chunk_for_line == 0
    assert m.excluded_multi_chunk_match == 0


def test_chunk_mapper_counts_unindexed_file_and_never_double_counts_repeat_lookups():
    m = _mapper([("c1", "a.py", 1, 5)])
    assert m.resolve("missing.py", 1) is None
    assert m.resolve("missing.py", 1) is None  # same (path, line) again
    assert m.resolve("missing.py", 2) is None  # different line, same missing file
    assert m.excluded_unindexed_file == 2      # one per unique (path, line), not per call
    assert m.excluded_unindexed_file_by_file == {"missing.py": 2}


def test_chunk_mapper_counts_no_chunk_for_line_when_file_indexed_but_line_uncovered():
    m = _mapper([("c1", "a.py", 1, 5), ("c2", "a.py", 10, 15)])
    assert m.resolve("a.py", 7) is None  # gap between chunks
    assert m.resolve("a.py", 7) is None  # memoized, still one count
    assert m.excluded_no_chunk_for_line == 1
    assert m.excluded_unindexed_file == 0


def test_chunk_mapper_never_silently_picks_one_of_multiple_overlapping_chunks():
    m = _mapper([("c1", "a.py", 1, 10), ("c2", "a.py", 5, 15)])  # overlap at lines 5-10
    assert m.resolve("a.py", 7) is None
    assert m.excluded_multi_chunk_match == 1


# ----------------------------------------------------------------------
# build_ground_truth: external exclusion, intra-chunk exclusion, fan-out vs
# same-file-nearest strict
# ----------------------------------------------------------------------

def test_build_ground_truth_excludes_external_symbols_with_no_definition():
    definitions = {}  # nothing defines "sym-x" anywhere
    reference_sites = [("a.py", 3, "sym-x")]
    mapper = _mapper([("c1", "a.py", 1, 10)])
    gt = sp.build_ground_truth(definitions, reference_sites, mapper)
    assert gt["gt_fanout"] == set()
    assert gt["gt_strict"] == set()
    assert gt["excluded_external"] == 1


def test_build_ground_truth_excludes_intra_chunk_references():
    definitions = {"sym-a": [("a.py", 2)]}
    reference_sites = [("a.py", 4, "sym-a")]  # same chunk (1-10) as its definition
    mapper = _mapper([("c1", "a.py", 1, 10)])
    gt = sp.build_ground_truth(definitions, reference_sites, mapper)
    assert gt["gt_fanout"] == set()
    assert gt["excluded_intra_chunk_fanout"] == 1
    assert gt["excluded_intra_chunk_strict"] == 1


def test_build_ground_truth_fanout_keeps_every_definition_site_strict_keeps_one():
    # sym-a defined twice (c2 and c3); referenced once from c1.
    definitions = {"sym-a": [("b.py", 2), ("c.py", 2)]}
    reference_sites = [("a.py", 2, "sym-a")]
    mapper = _mapper([
        ("c1", "a.py", 1, 5),
        ("c2", "b.py", 1, 5),
        ("c3", "c.py", 1, 5),
    ])
    gt = sp.build_ground_truth(definitions, reference_sites, mapper)
    assert gt["gt_fanout"] == {("c1", "c2"), ("c1", "c3")}
    assert len(gt["gt_strict"]) == 1  # neither def is same-file as the ref -> nearest by line wins
    assert gt["gt_strict"] <= gt["gt_fanout"]


def test_build_ground_truth_strict_prefers_same_file_definition_over_nearer_cross_file():
    # Strict must pick the same-file definition even though a cross-file one
    # is closer by line distance.
    definitions = {"sym-a": [("a.py", 1), ("b.py", 19)]}
    reference_sites = [("a.py", 20, "sym-a")]
    mapper = _mapper([
        ("c1", "a.py", 15, 25),
        ("c2", "a.py", 1, 10),
        ("c3", "b.py", 15, 25),
    ])
    gt = sp.build_ground_truth(definitions, reference_sites, mapper)
    assert gt["gt_fanout"] == {("c1", "c2"), ("c1", "c3")}
    assert gt["gt_strict"] == {("c1", "c2")}  # same-file def wins over the line-nearer cross-file one


def test_build_ground_truth_strict_picks_nearest_among_same_file_definitions():
    definitions = {"sym-a": [("a.py", 2), ("a.py", 18)]}
    reference_sites = [("a.py", 20, "sym-a")]
    mapper = _mapper([
        ("c1", "a.py", 20, 25),
        ("c2", "a.py", 1, 5),
        ("c3", "a.py", 16, 19),
    ])
    gt = sp.build_ground_truth(definitions, reference_sites, mapper)
    assert gt["gt_strict"] == {("c1", "c3")}  # line 18 (dist 2) beats line 2 (dist 18)


# ----------------------------------------------------------------------
# score_precision_recall: hand-built 5-chunk corpus with known overlap
# ----------------------------------------------------------------------

def test_score_precision_recall_hand_built_corpus():
    # c5 is in z.py, present in chunks but unmapped (z.py isn't in the
    # SCIP index's document set).
    chunk_id_to_path = {"c1": "x.py", "c2": "x.py", "c3": "y.py", "c4": "y.py", "c5": "z.py"}
    mapped_paths = {"x.py", "y.py"}
    gt_fanout = {("c1", "c3"), ("c4", "c2"), ("c2", "c4"), ("c2", "c1")}

    chunk_refs_rows = [
        ("c1", "c3", "calls"),       # hit, scoreable
        ("c2", "c3", "calls"),       # miss, scoreable
        ("c1", "c5", "calls"),       # unscoreable (c5 unmapped)
        ("c4", "c2", "imports"),     # hit, scoreable
        ("c2", "c4", "mentions"),    # hit, scoreable
        ("c3", "c1", "mentions"),    # miss, scoreable
        ("c2", "c1", "associated"),  # hit, scoreable
        ("c3", "c4", "inherits"),    # miss, scoreable
        ("c5", "c1", "xlang"),       # unscoreable (c5 unmapped)
    ]

    result = sp.score_precision_recall(chunk_refs_rows, chunk_id_to_path, mapped_paths, gt_fanout)
    precision, recall = result["precision"], result["recall"]

    assert precision["calls"] == {"hit": 1, "scoreable": 2, "unscoreable": 1, "ratio": 0.5}
    assert precision["imports"] == {"hit": 1, "scoreable": 1, "unscoreable": 0, "ratio": 1.0}
    assert precision["inherits"] == {"hit": 0, "scoreable": 1, "unscoreable": 0, "ratio": 0.0}
    assert precision["xlang"] == {"hit": 0, "scoreable": 0, "unscoreable": 1, "ratio": None}
    assert precision["associated"] == {"hit": 1, "scoreable": 1, "unscoreable": 0, "ratio": 1.0}
    assert precision["mentions"] == {"hit": 1, "scoreable": 2, "unscoreable": 0, "ratio": 0.5}

    assert recall["any"] == {"hit": 4, "total": 4, "ratio": 1.0}
    assert recall["typed_only"] == {"hit": 2, "total": 4, "ratio": 0.5}
    assert recall["typed_plus_associated"] == {"hit": 3, "total": 4, "ratio": 0.75}


def test_score_precision_recall_reports_edge_type_with_zero_edges_as_none_not_missing():
    result = sp.score_precision_recall([], {}, set(), set())
    for edge_type in sp.EDGE_TYPES:
        assert result["precision"][edge_type] == {"hit": 0, "scoreable": 0, "unscoreable": 0, "ratio": None}
    for key in ("any", "typed_only", "typed_plus_associated"):
        assert result["recall"][key] == {"hit": 0, "total": 0, "ratio": None}


# ----------------------------------------------------------------------
# End-to-end over the real scip-python fixture (no subprocess): pins the
# pure-logic pipeline against actual pyright/scip-python occurrence shapes,
# not just hand-synthesized ones.
# ----------------------------------------------------------------------

def _load_toy_fixture():
    with open(os.path.join(_DATA_DIR, "toy_index.json")) as f:
        return json.load(f)


def test_definition_role_is_bitfield_not_equality():
    # symbol_roles is a bitfield (Definition=1 can combine with other bits,
    # e.g. Import=8 -> 9); `roles == 1` would miss this and escapes every
    # other test in this file (mutation-verified).
    payload = {"documents": [{"relative_path": "x.py", "occurrences": [
        {"range": [0, 0, 5], "symbol": "scip-python python p 1 x/combo().", "symbol_roles": 9},
        {"range": [3, 0, 5], "symbol": "scip-python python p 1 x/combo().", "symbol_roles": 8},
    ]}]}
    occs = list(sp.iter_occurrences(payload))
    assert [(line, is_def) for _p, line, _s, is_def in occs] == [(1, True), (4, False)]


def test_real_fixture_occurrence_and_role_shape():
    # Sanity-checks the fixture itself hasn't silently changed shape (e.g. a
    # scip-python version bump) before trusting the end-to-end test below.
    scip_json = _load_toy_fixture()
    occurrences = list(sp.iter_occurrences(scip_json))
    assert len(occurrences) == 18
    definitions, reference_sites, excluded_local = sp.classify_occurrences(occurrences)
    assert excluded_local == 2  # the `f = Foo()` local, def + ref
    assert len(reference_sites) == 8
    assert "scip-python python eval-toy 0.0.1 a/helper()." in definitions
    assert definitions["scip-python python eval-toy 0.0.1 a/helper()."] == [("a.py", 1)]


def test_real_fixture_end_to_end_ground_truth():
    # Chunks below deliberately tile each file completely (toy_a.py has 7
    # lines, toy_b.py has 6), so there are no mapping gaps to confound this.
    scip_json = _load_toy_fixture()
    occurrences = list(sp.iter_occurrences(scip_json))
    definitions, reference_sites, excluded_local = sp.classify_occurrences(occurrences)

    chunk_rows = [
        ("A1", "a.py", 1, 2),  # def helper / return
        ("A2", "a.py", 3, 7),  # blank lines + class Foo (def, method, its return)
        ("B1", "b.py", 1, 3),  # import + blanks
        ("B2", "b.py", 4, 6),  # def use() + body
    ]
    mapper = sp.ChunkMapper(sp.build_chunk_index(chunk_rows))
    gt = sp.build_ground_truth(definitions, reference_sites, mapper)

    # x-param reference inside helper() is intra-chunk (both in A1) -> excluded.
    assert gt["excluded_intra_chunk_fanout"] == 1
    assert gt["excluded_intra_chunk_strict"] == 1
    assert gt["excluded_external"] == 0
    assert mapper.excluded_unindexed_file == 0
    assert mapper.excluded_no_chunk_for_line == 0
    assert mapper.excluded_multi_chunk_match == 0

    expected = {("A2", "A1"), ("B1", "A1"), ("B1", "A2"), ("B2", "A2"), ("B2", "A1")}
    assert gt["gt_fanout"] == expected
    # No symbol here has more than one definition site, so strict == fanout.
    assert gt["gt_strict"] == expected
