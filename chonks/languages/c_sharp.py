"""C# language plugin data."""
from __future__ import annotations

from ._ast import name_last_or_text, name_text
from ._naming import name_field
from .spec import ChildSpec, Children, Field, LanguageSpec, LiteralSpec, NameRule, NestedSpec, NOT_HANDLED

BASES = Children((
    ChildSpec(("base_list",), "inherits", nested=NestedSpec(("identifier",), name_text)),
))


def operator_name(node, src: bytes) -> str | None:
    """`operator+` like C++; a conversion operator is named by its target type."""
    if node.type == "operator_declaration":
        op = node.child_by_field_name("operator")
        return f"operator{name_text(op, src)}" if op else None
    target = node.child_by_field_name("type")
    return f"operator {name_text(target, src)}" if target else None


def classify_param(part: str) -> "str | object":
    if part.startswith("params "):
        return "variadic"
    return NOT_HANDLED


C_SHARP = LanguageSpec(
    name="c_sharp",
    # tree-sitter-language-pack uses 'csharp' but we use 'c_sharp' internally
    grammar="csharp",
    version="2",
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
    # Operators have no 'name' field; every other declaration type does.
    name_rules=(
        NameRule((
            "method_declaration", "constructor_declaration", "destructor_declaration",
            "property_declaration", "class_declaration", "struct_declaration",
            "interface_declaration", "enum_declaration",
        ), name_field),
        NameRule(("operator_declaration", "conversion_operator_declaration"), operator_name),
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
    # Call arity counts every `argument` node, named (`x: 1`) or positional, so
    # a C# arity is the full argument count, which is what the arity filter
    # wants. Python's arity skips keyword arguments, so the two languages mean
    # different things by `arity`.
    literals=LiteralSpec(
        leaf_types=("string_literal", "verbatim_string_literal", "interpolated_string_expression"),
        plus_type="binary_expression"),
    display_name="C#",
)

LANGUAGES = (C_SHARP,)
