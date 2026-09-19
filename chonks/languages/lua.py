"""Lua language plugin data."""
from __future__ import annotations

import re

from ._naming import name_field
from .spec import LanguageSpec, LiteralSpec, NameRule, NOT_HANDLED

# lua long-bracket [[...]]/[=[...]=]: tree-sitter-lua parses these as
# ordinary "string" nodes, not a distinct leaf type.
_LUA_LONG_BRACKET_RE = re.compile(r"^\[(=*)\[")


def literal_wrapper(raw: str) -> "tuple[str, str] | object":
    m = _LUA_LONG_BRACKET_RE.match(raw)
    if m:
        close = "]" + m.group(1) + "]"
        if raw.endswith(close) and len(raw) >= m.end() + len(close):
            return raw[m.end():-len(close)], "raw"
        return raw, "raw"
    return NOT_HANDLED


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
    # Lua's 'name' field is an identifier, or a dot_index_expression for the
    # `M.foo` module-table idiom; name_text() handles both.
    name_rules=(
        NameRule(("function_declaration",), name_field),
    ),
    # BUG preserved: '--' comments fall to the C-style default prefixes.
    # A fix moves chunk boundaries and needs a CHUNKER_VERSION bump.
    line_comment_prefixes=None,
    kind_labels={"function_declaration": "function"},
    literal_wrapper=literal_wrapper,
    literals=LiteralSpec(leaf_types=("string",), plus_type="binary_expression", plus_op=".."),
    display_name="Lua",
)

LANGUAGES = (LUA,)
