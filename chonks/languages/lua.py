"""Lua language plugin data."""
from __future__ import annotations

from .spec import LanguageSpec, LiteralSpec

LUA = LanguageSpec(
    name="lua",
    grammar="lua",
    # Best-effort tier; unmapped, cross-language C++/Lua binding surfaces
    # (e.g. cocos2d-x) go invisible to co-change pairing.
    extensions=frozenset({".lua"}),
    # Lua's grammar gives named and `local` functions the same node type
    # (function_declaration); anonymous function_definition is left
    # unboundaried since it has no name field.
    boundary_nodes=frozenset({"function_declaration"}),
    salvage_nodes=frozenset({"function_declaration"}),
    # BUG preserved: '--' comments fall to the C-style default prefixes.
    # A fix moves chunk boundaries and needs a CHUNKER_VERSION bump.
    line_comment_prefixes=None,
    kind_labels={"function_declaration": "function"},
    literals=LiteralSpec(leaf_types=("string",), plus_type="binary_expression", plus_op=".."),
)

LANGUAGES = (LUA,)
