"""JavaScript/TypeScript/TSX language plugin data."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._naming import arrow_var_name, name_field
from .spec import LanguageSpec, LiteralSpec, NameRule

if TYPE_CHECKING:
    from tree_sitter import Node

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


def is_arrow_var_decl(node: Node, src: bytes) -> bool:
    """True for `const NAME = (...) => ...;` style declarations: JS/TS has no
    dedicated node type for a named function assigned to a variable."""
    if node.type not in ("lexical_declaration", "variable_declaration"):
        return False
    for child in node.children:
        if child.type == "variable_declarator":
            value = child.child_by_field_name("value")
            if value is not None and value.type in ("arrow_function", "function_expression"):
                return True
    return False


# JS/TS/TSX: function/class/method and TS-only declarations all expose
# their name via the 'name' field.
JS_NAME_RULES = (
    NameRule(
        ("function_declaration", "generator_function_declaration",
         "class_declaration", "method_definition"),
        name_field,
    ),
    NameRule(("lexical_declaration", "variable_declaration"), arrow_var_name),
)
TS_NAME_RULES = (
    NameRule(
        ("function_declaration", "generator_function_declaration",
         "class_declaration", "method_definition",
         "interface_declaration", "enum_declaration",
         "type_alias_declaration", "abstract_class_declaration",
         "internal_module", "module"),
        name_field,
    ),
    NameRule(("lexical_declaration", "variable_declaration"), arrow_var_name),
)

JAVASCRIPT = LanguageSpec(
    name="javascript",
    grammar="javascript",
    version="1",
    extensions=frozenset({".js", ".jsx", ".mjs", ".cjs"}),
    boundary_nodes=JS_CORE_BOUNDARIES,
    salvage_nodes=frozenset({
        "function_declaration", "generator_function_declaration", "method_definition",
    }),
    extra_boundary=is_arrow_var_decl,
    name_rules=JS_NAME_RULES,
    kind_labels={
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "lexical_declaration": "function",
        "variable_declaration": "function",
    },
    literals=LiteralSpec(leaf_types=("string", "template_string"), plus_type="binary_expression"),
    display_name="JavaScript",
)

TYPESCRIPT = LanguageSpec(
    name="typescript",
    grammar="typescript",
    version="1",
    extensions=frozenset({".ts"}),
    boundary_nodes=JS_CORE_BOUNDARIES | TS_EXTRA_BOUNDARIES,
    # internal_module/module aren't boundary types; listed here only so an
    # EMPTY TS/TSX namespace still becomes its own chunk instead of vanishing.
    container_nodes=frozenset({"internal_module", "module"}),
    salvage_nodes=frozenset({
        "function_declaration", "generator_function_declaration", "method_definition",
    }),
    extra_boundary=is_arrow_var_decl,
    name_rules=TS_NAME_RULES,
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
    display_name="TypeScript",
)

TSX = LanguageSpec(
    name="tsx",
    grammar="tsx",
    version="1",
    extensions=frozenset({".tsx"}),
    boundary_nodes=JS_CORE_BOUNDARIES | TS_EXTRA_BOUNDARIES,
    container_nodes=frozenset({"internal_module", "module"}),
    salvage_nodes=frozenset({
        "function_declaration", "generator_function_declaration", "method_definition",
    }),
    extra_boundary=is_arrow_var_decl,
    name_rules=TS_NAME_RULES,
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
    display_name="TSX",
)

LANGUAGES = (JAVASCRIPT, TYPESCRIPT, TSX)
