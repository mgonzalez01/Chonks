"""C language plugin data."""
from __future__ import annotations

from .cpp import INCLUDE
from .spec import Field, LanguageSpec, LiteralSpec

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
    c_macro_self_heal=True,
    # c is the grammar cpp is a superset of, so preproc_include/call_expression
    # reuse cpp's rules as-is. No inherits rule: C has no base classes.
    refs_spec={
        "preproc_include": INCLUDE,
        "call_expression": Field("function", "calls"),
    },
    chain_base_fields={"field_expression": "argument"},
    identifier_leaf_types=frozenset({"identifier", "field_identifier", "type_identifier"}),
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
