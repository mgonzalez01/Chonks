"""Node-text and identifier helpers shared by the refs DSL and language specs."""
from __future__ import annotations

from typing import TYPE_CHECKING, Collection

if TYPE_CHECKING:
    from tree_sitter import Node


def last_identifier(node: Node, src: bytes) -> str | None:
    """Rightmost identifier-like leaf of a dotted/member chain (a.b.c):
    the referenced name, not the receiver it's called through."""
    for child in reversed(node.children):
        if child.type in ("identifier", "field_identifier", "type_identifier"):
            return src[child.start_byte:child.end_byte].decode(errors="replace")
        # recurse into nested chains (field_expression/member_access_expression/attribute)
        if child.is_named:
            found = last_identifier(child, src)
            if found:
                return found
    return None


def terminal_identifier(n: "Node | None", src: bytes, identifier_leaf_types: Collection[str]) -> str | None:
    """`n`'s own text if it's one of the caller's language's identifier-leaf
    types, else the rightmost such leaf in its subtree."""
    if n is None:
        return None
    if n.type in identifier_leaf_types:
        return name_text(n, src)
    return last_identifier(n, src)


def name_text(n: Node, src: bytes) -> str:
    return src[n.start_byte:n.end_byte].decode(errors="replace")


def name_last_or_text(n: Node, src: bytes) -> str | None:
    return last_identifier(n, src) or name_text(n, src)


def name_call_target(n: Node, src: bytes) -> str | None:
    # The called *name* is always the rightmost identifier-like leaf of
    # whatever expression is being invoked (bare name, member access,
    # qualified/namespaced name, ...).
    if n.type in ("identifier", "field_identifier"):
        return name_text(n, src)
    return last_identifier(n, src)


def name_strip_angle_brackets(n: Node, src: bytes) -> str:
    return name_text(n, src).strip("<>")
