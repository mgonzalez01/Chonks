"""C# language plugin data."""
from __future__ import annotations

from ._ast import name_last_or_text, name_text, terminal_identifier
from ._braces import outside_bodies
from ._naming import name_field
from .spec import ChildSpec, Children, Field, LanguageSpec, LiteralSpec, NameRule, NestedSpec, NOT_HANDLED

_TYPE_ARGUMENTS = frozenset({"type_argument_list"})


def base_name(node, src: bytes) -> str | None:
    """`IList<T>` -> 'IList', `Ns.Foo<Bar>` -> 'Foo'."""
    return terminal_identifier(node, src, C_SHARP.identifier_leaf_types, _TYPE_ARGUMENTS)


# `(x)`, `x!`, `await x`, `(T)x`: the operand is the last named child.
_OPERAND_WRAPPERS = frozenset({"parenthesized_expression", "postfix_unary_expression", "await_expression",
                               "cast_expression"})


def type_receiver(call_node, callee, src: bytes) -> "str | None | object":
    """`Ns.Foo<Bar>.Make()` -> 'Foo'. A call result as the base, as in
    `GetNode<Player>("p")!.Hide()`, is left to the generic receiver."""
    field = C_SHARP.chain_base_fields.get(callee.type)
    base = inner = callee.child_by_field_name(field) if field else None
    while inner is not None and inner.type in _OPERAND_WRAPPERS and inner.named_children:
        inner = inner.named_children[-1]
    if inner is None or inner.type == "invocation_expression":
        return NOT_HANDLED
    return base_name(base, src)


BASES = Children((
    ChildSpec(("base_list",), "inherits", nested=NestedSpec(("identifier", "generic_name"), base_name)),
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
    version="4",
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
            "interface_declaration", "enum_declaration", "namespace_declaration",
        ), name_field),
        NameRule(("operator_declaration", "conversion_operator_declaration"), operator_name),
    ),
    refs_spec={
        "using_directive": Children((
            ChildSpec(("identifier", "qualified_name"), "imports", name_fn=name_last_or_text),
        )),
        "invocation_expression": Field("function", "calls"),
        "object_creation_expression": Children((
            ChildSpec(("identifier", "generic_name", "qualified_name"), "calls", name_fn=base_name,
                      break_outer=True, construct=True),
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
    call_receiver=type_receiver,
    class_like_chunk_types=frozenset({"class_declaration", "struct_declaration", "interface_declaration"}),
    constructor_chunk_types=frozenset({"constructor_declaration"}),
    scope_chunk_types=frozenset({"namespace_declaration"}),
    doc_comment_prefixes=("///", "/**"),
    signature_at=outside_bodies(frozenset({"class", "struct", "interface", "record", "enum", "namespace"})),
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
