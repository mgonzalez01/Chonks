"""Tests for the receiver/arity call-site discriminator:
chonks.repomap._discriminate_definers and its supporting helpers, plus
their wiring into _build_graph's typed 'calls' pass.
"""
from chonks.repomap import (
    _arity_compatible,
    _build_graph,
    _call_entry_fields,
    _definer_qualifiers,
    _discriminate_definers,
)


# ---------------------------------------------------------------------------
# _call_entry_fields: normalizing a calls-list entry
# ---------------------------------------------------------------------------

def test_call_entry_fields_dict():
    assert _call_entry_fields({"name": "foo", "receiver": "x", "arity": 2}) == ("foo", "x", 2)


def test_call_entry_fields_dict_none_receiver_and_arity():
    assert _call_entry_fields({"name": "foo", "receiver": None, "arity": None}) == ("foo", None, None)


def test_call_entry_fields_backward_compat_plain_string():
    """A legacy chunk's bare-name calls-list string (never migrated)
    normalizes to (name, None, None), i.e. 'no evidence'."""
    assert _call_entry_fields("foo") == ("foo", None, None)


# ---------------------------------------------------------------------------
# _definer_qualifiers: qualifier candidates for a definer chunk
# ---------------------------------------------------------------------------

def test_definer_qualifiers_scoped_cpp_name():
    chunk = {"name": "Ns::Cls::method", "chunk_type": "function_definition"}
    assert _definer_qualifiers(chunk) == {"Cls"}


def test_definer_qualifiers_dotted_single_level():
    chunk = {"name": "Foo.bar", "chunk_type": "function_definition"}
    assert _definer_qualifiers(chunk) == {"Foo"}


def test_definer_qualifiers_class_like_chunk_contributes_own_name():
    """A whole-class chunk stands in as the 'owning class' of any method
    resolved through it."""
    chunk = {"name": "Parser", "chunk_type": "class_definition"}
    assert _definer_qualifiers(chunk) == {"Parser"}


def test_definer_qualifiers_plain_function_no_qualifier():
    chunk = {"name": "helper", "chunk_type": "function_definition"}
    assert _definer_qualifiers(chunk) == set()


def test_definer_qualifiers_unnamed_chunk_empty():
    assert _definer_qualifiers({"name": None, "chunk_type": "function_definition"}) == set()
    assert _definer_qualifiers({}) == set()


# ---------------------------------------------------------------------------
# _arity_compatible
# ---------------------------------------------------------------------------

def test_arity_compatible_exact_and_default_range():
    chunk = {"name": "f", "content": "def f(self, a, b=1):\n    pass", "language": "python"}
    cache: dict = {}
    assert _arity_compatible("id1", chunk, "f", 1, cache) is True   # b defaulted
    assert _arity_compatible("id1", chunk, "f", 2, cache) is True   # both given
    assert _arity_compatible("id1", chunk, "f", 0, cache) is False  # a is required
    assert _arity_compatible("id1", chunk, "f", 3, cache) is False  # too many


def test_arity_compatible_variadic_only_bounds_the_minimum():
    chunk = {"name": "f", "content": "def f(self, a, *args):\n    pass", "language": "python"}
    cache: dict = {}
    assert _arity_compatible("id1", chunk, "f", 1, cache) is True
    assert _arity_compatible("id1", chunk, "f", 50, cache) is True
    assert _arity_compatible("id1", chunk, "f", 0, cache) is False


def test_arity_compatible_missing_signature_always_compatible():
    """No recognizable 'name(...)' in content: 'missing info', never excluded."""
    chunk = {"name": "Foo", "content": "class Foo:\n    pass", "language": "python"}
    cache: dict = {}
    assert _arity_compatible("id1", chunk, "Foo", 0, cache) is True
    assert _arity_compatible("id1", chunk, "Foo", 99, cache) is True


def test_arity_compatible_caches_per_id():
    """Cache is keyed by (id, name), not content identity or id alone, so
    two different ids with identical content don't collide into one entry."""
    cache: dict = {}
    empty_chunk = {"name": "", "content": "", "language": "python"}
    _arity_compatible("id1", empty_chunk, "f", 0, cache)
    _arity_compatible("id2", empty_chunk, "f", 0, cache)
    assert set(cache) == {("id1", "f"), ("id2", "f")}


def test_arity_compatible_anchors_on_called_name_not_chunk_own_name():
    """A chunk can hold multiple definitions (e.g. '__init__' and 'strip'
    merged into one chunk by the symbol index). Anchoring on the chunk's
    own name would parse the wrong signature; must anchor on the called name."""
    chunk = {
        "name": "__init__",
        "content": "def __init__(self):\n    pass\ndef strip(self, x):\n    pass",
        "language": "python",
    }
    cache: dict = {}
    assert _arity_compatible("id1", chunk, "strip", 1, cache) is True
    assert _arity_compatible("id1", chunk, "strip", 2, cache) is False
    assert _arity_compatible("id1", chunk, "__init__", 0, cache) is True
    assert set(cache) == {("id1", "strip"), ("id1", "__init__")}


# ---------------------------------------------------------------------------
# _discriminate_definers: table-driven survivor sets
# ---------------------------------------------------------------------------

def _def_chunk(name: str, chunk_type: str = "function_definition",
               content: str = "", language: str = "python") -> dict:
    return {"name": name, "chunk_type": chunk_type, "content": content, "language": language}


# A 3-definer pool: two share the qualifier 'Parser' (different arity), one
# is qualified 'Lexer' (same bare name). cpp-shaped so the arity anchor can
# find each signature.
_ID_TO_CHUNK = {
    "p1": _def_chunk("Parser::parse", content="void Parser::parse(int a) {}", language="cpp"),
    "p3": _def_chunk("Parser::parse", content="void Parser::parse(int a, int b, int c) {}", language="cpp"),
    "l2": _def_chunk("Lexer::parse", content="void Lexer::parse(int a, int b) {}", language="cpp"),
}
_ALL_IDS = ["p1", "l2", "p3"]


def test_discriminate_owner_match_narrows_alone():
    """Receiver given, arity not: the receiver alone decides."""
    got = _discriminate_definers(_ALL_IDS, "parse", "Parser", None, _ID_TO_CHUNK, {})
    assert got == ["p1", "p3"]


def test_discriminate_owner_match_case_insensitive_suffix_fallback():
    """Exact-case match finds nothing ('parser' != 'Parser'), so it retries
    as a case-insensitive SUFFIX match: matches the common
    instance-variable-named-after-its-class idiom (self.parser.parse())."""
    got = _discriminate_definers(_ALL_IDS, "parse", "parser", None, _ID_TO_CHUNK, {})
    assert got == ["p1", "p3"]


def test_discriminate_case_insensitive_suffix_match_does_not_exclude_true_definer():
    """Proof this is SUFFIX, not EQUALITY, matching: 'parser' is a suffix
    of both 'Parser' and 'ArgumentParser', so equality would wrongly drop
    the true 'ArgumentParser' definer while suffix matching keeps both."""
    id_to_chunk = {
        "true": _def_chunk("ArgumentParser::parse", content="void ArgumentParser::parse(int a) {}", language="cpp"),
        "decoy": _def_chunk("Parser::parse", content="void Parser::parse(int a) {}", language="cpp"),
    }
    got = _discriminate_definers(["true", "decoy"], "parse", "parser", None, id_to_chunk, {})
    assert got == ["true", "decoy"]


def test_discriminate_owner_match_suffix_fallback_excludes_unrelated_qualifier():
    """The suffix fallback still excludes a qualifier that isn't a suffix
    match at all: 'lexer' doesn't end with 'parser'."""
    got = _discriminate_definers(_ALL_IDS, "parse", "parser", None, _ID_TO_CHUNK, {})
    assert "l2" not in got


def test_discriminate_owner_and_arity_combine_to_intersection():
    got = _discriminate_definers(_ALL_IDS, "parse", "Parser", 1, _ID_TO_CHUNK, {})
    assert got == ["p1"]


def test_discriminate_owner_survivors_kept_when_arity_intersection_empties():
    """Owner match narrows to a real set, but arity conflicts with all of
    them: the owner match is trusted over arity rather than collapsing to
    nothing."""
    got = _discriminate_definers(_ALL_IDS, "parse", "Parser", 999, _ID_TO_CHUNK, {})
    assert got == ["p1", "p3"]


def test_discriminate_owner_applies_finds_nothing_falls_to_arity_alone():
    got = _discriminate_definers(_ALL_IDS, "parse", "NoSuchQualifier", 2, _ID_TO_CHUNK, {})
    assert got == ["l2"]


def test_discriminate_no_evidence_fans_out_fully():
    got = _discriminate_definers(_ALL_IDS, "parse", None, None, _ID_TO_CHUNK, {})
    assert got == _ALL_IDS


def test_discriminate_both_discriminators_empty_falls_back_to_full_fanout():
    """Receiver matches nothing and arity is incompatible with every
    candidate: falls back to the full candidate set rather than dropping
    the edge entirely (not a general recall guarantee, see the docstring)."""
    got = _discriminate_definers(_ALL_IDS, "parse", "NoSuchQualifier", 999, _ID_TO_CHUNK, {})
    assert got == _ALL_IDS


def test_discriminate_is_deterministic_and_preserves_ids_order():
    ids = ["l2", "p3", "p1"]  # deliberately not _ALL_IDS' order
    first = _discriminate_definers(ids, "parse", "Parser", None, _ID_TO_CHUNK, {})
    second = _discriminate_definers(ids, "parse", "Parser", None, _ID_TO_CHUNK, {})
    assert first == second == ["p3", "p1"]  # follows `ids`' order, not _ALL_IDS'


def test_discriminate_single_candidate_ids_list_unaffected():
    got = _discriminate_definers(["p1"], "parse", "NoMatch", 999, _ID_TO_CHUNK, {})
    assert got == ["p1"]


# ---------------------------------------------------------------------------
# Wiring into _build_graph's typed 'calls' pass
# ---------------------------------------------------------------------------

def test_build_graph_calls_discriminator_narrows_a_real_collision():
    chunks = [
        {"id": "caller", "name": "driver", "language": "python",
         "content": "def driver():\n    self.parser.parse(1)",
         "metadata": {"calls": [{"name": "parse", "receiver": "parser", "arity": 1}]}},
        {"id": "parser_cls", "name": "Parser", "chunk_type": "class_definition",
         "language": "python", "content": "class Parser:\n    def parse(self, a): pass"},
        {"id": "lexer_cls", "name": "Lexer", "chunk_type": "class_definition",
         "language": "python", "content": "class Lexer:\n    def parse(self, a): pass"},
    ]
    sym_map = {"parse": ["parser_cls", "lexer_cls"]}
    edges = _build_graph(chunks, sym_map)
    assert edges.get(("caller", "parser_cls")) == "calls"
    # lexer_cls may still pick up a lower-precedence 'mentions' edge from
    # the content-scan pass; the discriminator only narrows 'calls'.
    assert edges.get(("caller", "lexer_cls")) != "calls"


def test_build_graph_calls_no_evidence_still_fans_out_fully():
    chunks = [
        {"id": "caller", "name": "driver", "language": "python",
         "content": "def driver():\n    parse(1)",
         "metadata": {"calls": [{"name": "parse", "receiver": None, "arity": 1}]}},
        {"id": "a", "name": "Parser", "chunk_type": "class_definition",
         "language": "python", "content": "class Parser:\n    def parse(self, a): pass"},
        {"id": "b", "name": "Lexer", "chunk_type": "class_definition",
         "language": "python", "content": "class Lexer:\n    def parse(self, a): pass"},
    ]
    sym_map = {"parse": ["a", "b"]}
    edges = _build_graph(chunks, sym_map)
    assert edges.get(("caller", "a")) == "calls"
    assert edges.get(("caller", "b")) == "calls"


def test_build_graph_calls_backward_compat_plain_string_fans_out_fully():
    chunks = [
        {"id": "caller", "name": "driver", "language": "python",
         "content": "def driver():\n    self.parser.parse(1)",
         "metadata": {"calls": ["parse"]}},
        {"id": "a", "name": "Parser", "chunk_type": "class_definition",
         "language": "python", "content": "class Parser:\n    def parse(self, a): pass"},
        {"id": "b", "name": "Lexer", "chunk_type": "class_definition",
         "language": "python", "content": "class Lexer:\n    def parse(self, a): pass"},
    ]
    sym_map = {"parse": ["a", "b"]}
    edges = _build_graph(chunks, sym_map)
    assert edges.get(("caller", "a")) == "calls"
    assert edges.get(("caller", "b")) == "calls"


def test_build_graph_imports_and_inherits_unaffected_by_discriminator():
    """Only 'calls' entries carry a fingerprint. imports/inherits must keep
    fanning out as before, never routed through _discriminate_definers."""
    chunks = [
        {"id": "caller", "name": "driver", "language": "python",
         "content": "class driver(Base): pass",
         "metadata": {"inherits": ["Base"]}},
        {"id": "a", "name": "Base", "chunk_type": "class_definition",
         "language": "python", "content": "class Base: pass"},
        {"id": "b", "name": "Base", "chunk_type": "class_definition",
         "language": "cpp", "content": "class Base {};"},
    ]
    edges = _build_graph(chunks)
    assert edges.get(("caller", "a")) == "inherits"
    assert edges.get(("caller", "b")) == "inherits"
