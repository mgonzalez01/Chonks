"""C# language plugin data."""
from __future__ import annotations

from ._ast import name_last_or_text, name_text
from ._naming import name_field
from .spec import ChildSpec, Children, Field, LanguageSpec, LiteralSpec, NameRule, NestedSpec, NOT_HANDLED

BASES = Children((
    ChildSpec(("base_list",), "inherits", nested=NestedSpec(("identifier",), name_text)),
))


def classify_param(part: str) -> "str | object":
    if part.startswith("params "):
        return "variadic"
    return NOT_HANDLED


C_SHARP = LanguageSpec(
    name="c_sharp",
    # tree-sitter-language-pack uses 'csharp' but we use 'c_sharp' internally
    grammar="csharp",
    extensions=frozenset({".cs"}),
    boundary_nodes=frozenset({
        "method_declaration", "constructor_declaration", "destructor_declaration",
        "property_declaration", "class_declaration", "struct_declaration",
        "interface_declaration", "enum_declaration", "operator_declaration",
        "conversion_operator_declaration",
    }),
    container_nodes=frozenset({"namespace_declaration", "class_declaration", "struct_declaration"}),
    salvage_nodes=frozenset({
        "method_declaration", "constructor_declaration", "destructor_declaration",
        "operator_declaration", "conversion_operator_declaration",
    }),
    # C#: all declaration types have a 'name' field
    name_rules=(
        NameRule((
            "method_declaration", "constructor_declaration", "destructor_declaration",
            "property_declaration", "class_declaration", "struct_declaration",
            "interface_declaration", "enum_declaration",
            "operator_declaration", "conversion_operator_declaration",
        ), name_field),
    ),
    refs_spec={
        "using_directive": Children((
            ChildSpec(("identifier", "qualified_name"), "imports", name_fn=name_last_or_text),
        )),
        "invocation_expression": Field("function", "calls"),
        "object_creation_expression": Children((
            ChildSpec(("identifier", "generic_name", "qualified_name"), "calls",
                      name_fn=name_last_or_text, break_outer=True),
        )),
        "class_declaration": BASES,
        "struct_declaration": BASES,
        "interface_declaration": BASES,
    },
    chain_base_fields={
        "member_access_expression": "expression",
        "qualified_name": "qualifier",
    },
    identifier_leaf_types=frozenset({"identifier"}),
    class_like_chunk_types=frozenset({"class_declaration", "struct_declaration", "interface_declaration"}),
    classify_param=classify_param,
    kind_labels={
        "class_declaration": "class",
        "method_declaration": "method",
        "interface_declaration": "interface",
        "struct_declaration": "struct",
        "enum_declaration": "enum",
        "namespace_declaration": "namespace",
        "destructor_declaration": "destructor",
        "property_declaration": "property",
        "operator_declaration": "operator",
        "conversion_operator_declaration": "operator",
        "constructor_declaration": "method",
    },
    literals=LiteralSpec(
        leaf_types=("string_literal", "verbatim_string_literal", "interpolated_string_expression"),
        plus_type="binary_expression"),
)

LANGUAGES = (C_SHARP,)
