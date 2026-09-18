"""Registry views equal the F1 chunking.py tables and today's literals."""

import importlib.util
import subprocess
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

import chonks.chunking as chunking
import chonks.languages as languages
import chonks.repomap.refs as refs
import chonks.repomap.render as render
import chonks.store as store
from chonks.languages import _ast
from chonks.languages.spec import LanguageSpec


def _load_f1_chunking():
    # Same loader as tests/test_refs_equivalence.py, pinned to the F1 tag.
    proc = subprocess.run(
        ["git", "show", "F1:chonks/chunking.py"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        pytest.skip("F1 tag unavailable (detached/shallow checkout)")
    spec = importlib.util.spec_from_loader("chunking_f1_reference", loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chunking_f1_reference"] = mod
    exec(compile(proc.stdout, "chunking_f1_reference.py", "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def f1():
    return _load_f1_chunking()


def _normalize(value):
    """Dataclasses and functions from two copies of a module never compare
    equal; compare by name and fields."""
    if is_dataclass(value) and not isinstance(value, type):
        cls_name = type(value).__name__.lstrip("_")
        return (cls_name, tuple(_normalize(getattr(value, f.name)) for f in fields(value)))
    if isinstance(value, (dict, MappingProxyType)):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return frozenset(_normalize(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_normalize(v) for v in value)
    if callable(value) and hasattr(value, "__name__"):
        return value.__name__.lstrip("_")
    return value


_TABLE_CASES = {
    "_BOUNDARY_NODES": lambda: languages.table("boundary_nodes"),
    "_CONTAINER_NODES": lambda: languages.table("container_nodes"),
    "_SALVAGE_ELIGIBLE_BOUNDARY_TYPES": lambda: languages.table("salvage_nodes"),
    "_EXT_TO_LANG": lambda: languages.EXT_TO_LANG,
    "CODE_LANGUAGES": lambda: languages.CODE_LANGUAGES,
    "LANG_REFS_SPECS": lambda: languages.table("refs_spec"),
    "_REFS_LANGS": lambda: set(languages.table("refs_spec")),
    "_LITERAL_SPECS": lambda: languages.table("literals"),
    "_DEF_SIGNATURE_KEYWORD": lambda: languages.table("def_signature_keyword"),
    "_COMMENT_PREFIXES": lambda: languages.table("line_comment_prefixes"),
    "_MACRO_LANGS": lambda: languages.flags("c_macro_self_heal"),
}


@pytest.mark.parametrize("name", sorted(_TABLE_CASES))
def test_table_matches_f1(name, f1):
    old = getattr(f1, name)
    assert _normalize(_TABLE_CASES[name]()) == _normalize(old)
    assert _normalize(getattr(chunking, name)) == _normalize(old)


_UNION_CASES = {
    "chunking._CHAIN_BASE_FIELD": (
        lambda: languages.merged("chain_base_fields"), lambda: chunking._CHAIN_BASE_FIELD),
    "_ast.TERMINAL_IDENTIFIER_TYPES": (
        lambda: languages.union("identifier_leaf_types"), lambda: set(_ast.TERMINAL_IDENTIFIER_TYPES)),
    "chunking._VARIADIC_CALL_ARG_TYPES": (
        lambda: languages.union("variadic_arg_types"), lambda: chunking._VARIADIC_CALL_ARG_TYPES),
    "chunking._KEYWORD_CALL_ARG_TYPES": (
        lambda: languages.union("keyword_arg_types"), lambda: chunking._KEYWORD_CALL_ARG_TYPES),
    "refs._CLASS_LIKE_CHUNK_TYPES": (
        lambda: languages.union("class_like_chunk_types"), lambda: refs._CLASS_LIKE_CHUNK_TYPES),
    "render._NODE_TYPE_PREFIX": (
        lambda: languages.merged("kind_labels", languages.GENERIC_KIND_LABELS),
        lambda: render._NODE_TYPE_PREFIX),
    "store.HEADER_EXTS": (
        lambda: languages.union("header_exts"), lambda: store.HEADER_EXTS),
    "store.IMPL_EXTS": (
        lambda: languages.union("impl_exts"), lambda: store.IMPL_EXTS),
}


@pytest.mark.parametrize("name", sorted(_UNION_CASES))
def test_union_matches_live_table(name):
    registry_fn, live_fn = _UNION_CASES[name]
    assert _normalize(registry_fn()) == _normalize(live_fn())


def test_grammar_names():
    overrides = {
        s: languages.get(s).grammar for s in languages.CODE_LANGUAGES
        if languages.get(s).grammar != s
    }
    assert overrides == {"c_sharp": "csharp"}


def test_validate_duplicate_extension():
    dup_a = LanguageSpec(name="dup_a", grammar="dup_a", extensions=frozenset({".dup"}),
                          boundary_nodes=frozenset())
    dup_b = LanguageSpec(name="dup_b", grammar="dup_b", extensions=frozenset({".dup"}),
                          boundary_nodes=frozenset())
    with pytest.raises(ValueError, match="dup_b"):
        languages.Registry((dup_a, dup_b))


def test_validate_label_conflict():
    conflict_a = LanguageSpec(name="conflict_a", grammar="conflict_a", extensions=frozenset({".ca"}),
                               boundary_nodes=frozenset({"widget"}), kind_labels={"widget": "one"})
    conflict_b = LanguageSpec(name="conflict_b", grammar="conflict_b", extensions=frozenset({".cb"}),
                               boundary_nodes=frozenset({"widget"}), kind_labels={"widget": "two"})
    with pytest.raises(ValueError, match="conflict_b"):
        languages.Registry((conflict_a, conflict_b))


def test_validate_missing_kind_label():
    unlabeled = LanguageSpec(name="unlabeled", grammar="unlabeled", extensions=frozenset({".ul"}),
                             boundary_nodes=frozenset({"gizmo"}))
    with pytest.raises(ValueError, match="unlabeled"):
        languages.Registry((unlabeled,))
