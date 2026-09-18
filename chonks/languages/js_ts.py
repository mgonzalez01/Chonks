"""JavaScript/TypeScript/TSX language plugin data."""
from __future__ import annotations

from .spec import LanguageSpec, LiteralSpec

# typescript/tsx are supersets of the javascript grammar, so their boundary
# sets layer on javascript's core plus TS-only declaration types.
JS_CORE_BOUNDARIES = frozenset({
    "function_declaration", "generator_function_declaration",
    "method_definition", "class_declaration",
})
TS_EXTRA_BOUNDARIES = frozenset({
    "interface_declaration", "enum_declaration",
    "type_alias_declaration", "abstract_class_declaration",
})

JAVASCRIPT = LanguageSpec(
    name="javascript",
    grammar="javascript",
    extensions=frozenset({".js", ".jsx", ".mjs", ".cjs"}),
    boundary_nodes=JS_CORE_BOUNDARIES,
    salvage_nodes=frozenset({
        "function_declaration", "generator_function_declaration", "method_definition",
    }),
    kind_labels={
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "lexical_declaration": "function",
        "variable_declaration": "function",
    },
    literals=LiteralSpec(leaf_types=("string", "template_string"), plus_type="binary_expression"),
)

TYPESCRIPT = LanguageSpec(
    name="typescript",
    grammar="typescript",
    extensions=frozenset({".ts"}),
    boundary_nodes=JS_CORE_BOUNDARIES | TS_EXTRA_BOUNDARIES,
    # internal_module/module aren't boundary types; listed here only so an
    # EMPTY TS/TSX namespace still becomes its own chunk instead of vanishing.
    container_nodes=frozenset({"internal_module", "module"}),
    salvage_nodes=frozenset({
        "function_declaration", "generator_function_declaration", "method_definition",
    }),
    kind_labels={
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "lexical_declaration": "function",
        "variable_declaration": "function",
        "abstract_class_declaration": "class",
        "type_alias_declaration": "type",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
    },
    literals=LiteralSpec(leaf_types=("string", "template_string"), plus_type="binary_expression"),
)

TSX = LanguageSpec(
    name="tsx",
    grammar="tsx",
    extensions=frozenset({".tsx"}),
    boundary_nodes=JS_CORE_BOUNDARIES | TS_EXTRA_BOUNDARIES,
    container_nodes=frozenset({"internal_module", "module"}),
    salvage_nodes=frozenset({
        "function_declaration", "generator_function_declaration", "method_definition",
    }),
    kind_labels={
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "lexical_declaration": "function",
        "variable_declaration": "function",
        "abstract_class_declaration": "class",
        "type_alias_declaration": "type",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
    },
    literals=LiteralSpec(leaf_types=("string", "template_string"), plus_type="binary_expression"),
)

LANGUAGES = (JAVASCRIPT, TYPESCRIPT, TSX)
