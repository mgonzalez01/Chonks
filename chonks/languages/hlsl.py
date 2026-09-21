"""HLSL language plugin data."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._ast import name_text
from ._naming import cpp_function_declarator_name, tag_specifier_name
from .spec import LanguageSpec, LiteralSpec, NameRule

if TYPE_CHECKING:
    from tree_sitter import Node


def is_cbuffer(node: Node, src: bytes) -> bool:
    """tree-sitter-hlsl parses `cbuffer Foo {...}` as a plain `declaration`
    node (type_identifier "cbuffer"/"tbuffer" + identifier), not its own type."""
    if node.type != "declaration":
        return False
    for child in node.children:
        if child.type == "type_identifier":
            text = src[child.start_byte:child.end_byte].decode(errors="replace")
            return text in ("cbuffer", "tbuffer")
    return False


def cbuffer_name(node: Node, src: bytes) -> str | None:
    if not is_cbuffer(node, src):
        return None
    # declaration → identifier child
    for child in node.children:
        if child.type == "identifier":
            return name_text(child, src)
    return None


def cbuffer_chunk_type(node: Node, src: bytes) -> str | None:
    if is_cbuffer(node, src):
        return "cbuffer"
    return None


HLSL = LanguageSpec(
    name="hlsl",
    grammar="hlsl",
    version="2",
    extensions=frozenset({".hlsl", ".fx", ".fxh"}),
    # cbuffer/tbuffer are found with `is_cbuffer()` because tree-sitter-hlsl
    # parses them as `declaration` nodes, not a dedicated node type.
    boundary_nodes=frozenset({"function_definition", "struct_specifier"}),
    salvage_nodes=frozenset({"function_definition"}),  # cbuffer/tbuffer: see the salvage_extra field below
    salvage_extra=is_cbuffer,
    extra_boundary=is_cbuffer,
    synthetic_chunk_type=cbuffer_chunk_type,
    name_rules=(
        NameRule(("declaration",), cbuffer_name),
        NameRule(("function_definition",), cpp_function_declarator_name),
        NameRule(("struct_specifier",), tag_specifier_name),
    ),
    chain_base_fields={"field_expression": "argument"},
    identifier_leaf_types=frozenset({"identifier", "field_identifier", "type_identifier"}),
    kind_labels={
        "function_definition": "function",
        "struct_specifier": "struct",
        "cbuffer": "cbuffer",
    },
    # Same C-preprocessor-family grammar as cpp: string_literal/raw_string_literal
    # leaves, concatenated_string for adjacent juxtaposition. No plus_type: HLSL
    # has no "+"-based string concatenation to detect, same as c/cpp.
    literals=LiteralSpec(leaf_types=("string_literal", "raw_string_literal"),
                         concat_types=("concatenated_string",)),
    display_name="HLSL",
)

LANGUAGES = (HLSL,)
