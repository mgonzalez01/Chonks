"""Node-text and identifier helpers shared by the refs DSL and language specs."""
from __future__ import annotations

from typing import TYPE_CHECKING, Collection

if TYPE_CHECKING:
    from tree_sitter import Node


# `f<T>(x)` in C++ and C#: the type arguments are not the called name.
_TYPE_ARGUMENT_LISTS = frozenset({"template_argument_list", "type_argument_list"})


def last_identifier(node: Node, src: bytes, skip: Collection[str] = ()) -> str | None:
    """Rightmost identifier-like leaf of a dotted/member chain (a.b.c):
    the referenced name, not the receiver it's called through. Children
    whose type is in `skip` are not searched."""
    for child in reversed(node.children):
        if child.type in skip:
            continue
        if child.type in ("identifier", "field_identifier", "type_identifier"):
            return src[child.start_byte:child.end_byte].decode(errors="replace")
        # recurse into nested chains (field_expression/member_access_expression/attribute)
        if child.is_named:
            found = last_identifier(child, src, skip)
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
    return last_identifier(n, src, _TYPE_ARGUMENT_LISTS)


def name_strip_angle_brackets(n: Node, src: bytes) -> str:
    return name_text(n, src).strip("<>")


def has_body(node: Node, src: bytes) -> bool:
    return node.child_by_field_name("body") is not None


def is_forward_declaration(node: Node, src: bytes) -> bool:
    """`struct X;` on its own. `struct stat st;` and `struct X *f();` are
    uses of the type, not declarations of it."""
    nxt = node.next_sibling
    return not has_body(node, src) and nxt is not None and nxt.type == ";"


def is_opaque_typedef(node: Node, src: bytes) -> bool:
    """`typedef struct X X;`: names a type whose layout is defined elsewhere."""
    target = node.child_by_field_name("type")
    return (target is not None and target.type in ("struct_specifier", "class_specifier",
                                                   "union_specifier")
            and not has_body(target, src))
