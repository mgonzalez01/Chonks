"""Registry validation, describe(), and the literals that the registry must equal."""

from dataclasses import fields, is_dataclass
from types import MappingProxyType

import pytest

import chonks.languages as languages
from chonks.languages.spec import LanguageSpec


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


# The engine used to consult these as cross-language unions instead of the
# per-language spec fields they duplicated. They are gone now; what's worth
# pinning is that the union of the per-language fields still covers exactly
# what each literal used to hold, member for member, so a future spec edit
# can't silently drop one.
_UNION_CASES = {
    "chain_base_fields": (
        lambda: languages.merged("chain_base_fields"),
        lambda: {
            "field_expression": "argument",
            "attribute": "object",
            "member_access_expression": "expression",
            "qualified_name": "qualifier",
        },
    ),
    "identifier_leaf_types": (
        lambda: languages.union("identifier_leaf_types"),
        lambda: {"identifier", "field_identifier", "type_identifier", "namespace_identifier"},
    ),
    "variadic_arg_types": (
        lambda: languages.union("variadic_arg_types"),
        lambda: {"list_splat", "dictionary_splat", "parameter_pack_expansion"},
    ),
    "keyword_arg_types": (
        lambda: languages.union("keyword_arg_types"),
        lambda: {"keyword_argument"},
    ),
}


@pytest.mark.parametrize("name", sorted(_UNION_CASES))
def test_union_matches_former_literal(name):
    registry_fn, expected_fn = _UNION_CASES[name]
    assert _normalize(registry_fn()) == _normalize(expected_fn())


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


@pytest.mark.parametrize("spec", languages._discover(), ids=lambda spec: spec.name)
def test_every_boundary_and_container_has_a_name_rule(spec):
    # Both become chunks, and a chunk is named only through a name rule.
    named = {t for rule in spec.name_rules for t in rule.types}
    unnamed = (spec.boundary_nodes | spec.container_nodes) - named
    assert not unnamed, f"{spec.name}: {sorted(unnamed)} have no name rule"


def test_describe_has_one_row_per_spec():
    rows = languages.describe()
    assert [row["name"] for row in rows] == [s.name for s in languages.REGISTRY.specs]
    expected_keys = [
        "name", "language", "extensions", "boundaries", "typed_refs",
        "literals", "arity_keyword", "macro_heal", "pairing",
    ]
    for row in rows:
        assert list(row.keys()) == expected_keys


def test_every_spec_has_a_unique_display_name():
    names = [s.display_name for s in languages.REGISTRY.specs]
    assert all(names)
    assert len(set(names)) == len(names)


def test_describe_row_values():
    rows = {row["name"]: row for row in languages.describe()}
    assert rows["cpp"] == {
        "name": "cpp",
        "language": "C++",
        "extensions": (".cc", ".cpp", ".cu", ".cuh", ".cxx", ".h", ".hpp", ".hxx", ".inl", ".metal", ".mm"),
        "boundaries": ("class_specifier", "function_definition", "struct_specifier", "template_declaration"),
        "typed_refs": True,
        "literals": True,
        "arity_keyword": False,
        "macro_heal": True,
        "pairing": True,
    }
    assert rows["hlsl"]["typed_refs"] is True
    assert rows["hlsl"]["literals"] is True
    assert rows["python"]["arity_keyword"] is True
    assert rows["python"]["macro_heal"] is False
    assert rows["python"]["pairing"] is False
