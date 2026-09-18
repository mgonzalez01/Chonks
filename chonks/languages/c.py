"""C language plugin data."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._naming import cpp_function_declarator_name, tag_specifier_name, typedef_name
from .cpp import INCLUDE, classify_param
from .spec import Field, LanguageSpec, LiteralSpec, NameRule

if TYPE_CHECKING:
    from tree_sitter import Node

# A bare tag reference/forward decl shares this node type with a real
# definition; without the body check below it spawns a spurious chunk.
_C_TAG_TYPES = {"struct_specifier", "union_specifier", "enum_specifier"}


def is_tag_definition(node: Node, src: bytes) -> bool:
    return node.type in _C_TAG_TYPES and node.child_by_field_name("body") is not None


C = LanguageSpec(
    name="c",
    grammar="c",
    extensions=frozenset({".c"}),
    # struct/union/enum_specifier also match bare tag refs and forward decls;
    # _is_boundary's C guard requires a body to treat one as a real boundary.
    boundary_nodes=frozenset({
        "function_definition", "struct_specifier", "union_specifier",
        "enum_specifier", "type_definition",
    }),
    salvage_nodes=frozenset({"function_definition"}),
    boundary_filters={t: is_tag_definition for t in _C_TAG_TYPES},
    name_rules=(
        NameRule(("function_definition",), cpp_function_declarator_name),
        NameRule(("struct_specifier", "union_specifier", "enum_specifier"), tag_specifier_name),
        NameRule(("type_definition",), typedef_name),
    ),
    c_macro_self_heal=True,
    # c is the grammar cpp is a superset of, so preproc_include/call_expression
    # reuse cpp's rules as-is. No inherits rule: C has no base classes.
    refs_spec={
        "preproc_include": INCLUDE,
        "call_expression": Field("function", "calls"),
    },
    chain_base_fields={"field_expression": "argument"},
    identifier_leaf_types=frozenset({"identifier", "field_identifier", "type_identifier"}),
    classify_param=classify_param,
    empty_param_spellings=frozenset({"void"}),
    kind_labels={
        "function_definition": "function",
        "struct_specifier": "struct",
        "enum_specifier": "enum",
        "union_specifier": "union",
        "type_definition": "typedef",
    },
    literals=LiteralSpec(leaf_types=("string_literal",), concat_types=("concatenated_string",)),
)

LANGUAGES = (C,)
