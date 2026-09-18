"""HLSL language plugin data."""
from __future__ import annotations

from .spec import LanguageSpec

HLSL = LanguageSpec(
    name="hlsl",
    grammar="hlsl",
    extensions=frozenset({".hlsl", ".fx", ".fxh"}),
    # cbuffer/tbuffer are detected via _is_cbuffer() because tree-sitter-hlsl
    # parses them as `declaration` nodes, not a dedicated node type.
    boundary_nodes=frozenset({"function_definition", "struct_specifier"}),
    salvage_nodes=frozenset({"function_definition"}),  # cbuffer/tbuffer: see the _is_cbuffer branch in chunking.py
    chain_base_fields={"field_expression": "argument"},
    identifier_leaf_types=frozenset({"identifier", "field_identifier", "type_identifier"}),
    kind_labels={
        "function_definition": "function",
        "struct_specifier": "struct",
        "cbuffer": "cbuffer",
    },
)

LANGUAGES = (HLSL,)
