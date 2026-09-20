"""GDScript language plugin data."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._ast import name_last_or_text, terminal_identifier
from ._naming import name_field
from .spec import ChildSpec, Children, LanguageSpec, LiteralSpec, NameRule

if TYPE_CHECKING:
    from tree_sitter import Node

CALL = Children((
    ChildSpec(("identifier",), "calls", break_outer=True),
))


def call_receiver(call_node: Node, callee: Node, src: bytes) -> str | None:
    """gdscript's 'a.b.c()' is one flat 'attribute' node, not a nested
    chain, so the receiver is the last named sibling before `call_node`."""
    parent = call_node.parent
    if parent is None or parent.type != "attribute":
        return None
    prev: "Node | None" = None
    for c in parent.children:
        # tree-sitter Node identity isn't stable under `is`; compare `.id`.
        if c.id == call_node.id:
            break
        if c.is_named:
            prev = c
    return terminal_identifier(prev, src)


GDSCRIPT = LanguageSpec(
    name="gdscript",
    grammar="gdscript",
    version="1",
    extensions=frozenset({".gd"}),
    boundary_nodes=frozenset({"function_definition", "class_definition"}),
    salvage_nodes=frozenset({"function_definition"}),
    call_receiver=call_receiver,
    name_rules=(
        NameRule(("function_definition", "class_definition"), name_field),
    ),
    refs_spec={
        "call": CALL,
        "attribute_call": CALL,
        "extends_statement": Children((
            ChildSpec(("type",), "inherits", name_fn=name_last_or_text),
        )),
    },
    identifier_leaf_types=frozenset({"identifier"}),
    # BUG preserved: '#' comments fall to the C-style default prefixes.
    # A fix moves chunk boundaries and needs a CHUNKER_VERSION bump.
    line_comment_prefixes=None,
    def_signature_keyword=r"\bfunc\s+",
    class_like_chunk_types=frozenset({"class_definition"}),
    kind_labels={"function_definition": "function", "class_definition": "class"},
    literals=LiteralSpec(leaf_types=("string",)),
    display_name="GDScript",
)

LANGUAGES = (GDSCRIPT,)
