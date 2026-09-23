"""Per-node-type name-extraction hooks, shared or language-owned, and the engine that walks them."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._ast import name_text

if TYPE_CHECKING:
    from tree_sitter import Node

    from .spec import LanguageSpec


def name_field(node: Node, src: bytes) -> str | None:
    n = node.child_by_field_name("name")
    if n:
        return name_text(n, src)
    return None


# Declarator layers that wrap the name: `int* f()`, `V& f()`, `int (*f())(int)`.
_WRAPPER_DECLARATORS = frozenset({
    "function_declarator", "pointer_declarator", "reference_declarator",
    "parenthesized_declarator", "attributed_declarator",
})


def _operator_cast_name(node: Node, src: bytes) -> str:
    # `operator int*() const` -> `operator int*`: the text up to the parameter list.
    end = node.end_byte
    target = node.child_by_field_name("type")
    stack = [c for c in node.children if c != target]
    while stack:
        n = stack.pop()
        if n.type == "parameter_list":
            end = min(end, n.start_byte)
        stack.extend(n.children)
    return " ".join(src[node.start_byte:end].decode(errors="replace").split())


# C / C++ / HLSL function_definition: type declarator body. The declarator
# field holds the name under any number of wrapper layers.
def cpp_function_declarator_name(node: Node, src: bytes) -> str | None:
    decl = node.child_by_field_name("declarator")
    if decl is None:
        # fallback: find first function_declarator among children
        for c in node.children:
            if c.type == "function_declarator":
                decl = c
                break
    # A real definition has a function_declarator on the way down (or is a
    # conversion operator). Without one, this is error recovery reading
    # `MACRO class Foo {` as a function: leave it unnamed so salvage runs.
    is_function = False
    while decl is not None and decl.type in _WRAPPER_DECLARATORS:
        is_function = is_function or decl.type == "function_declarator"
        inner = decl.child_by_field_name("declarator")
        if inner is None:
            # reference_declarator and parenthesized_declarator have no field
            named = [c for c in decl.children if c.is_named]
            inner = named[0] if named else None
        decl = inner
    if decl is None:
        return None
    if decl.type == "operator_cast":
        return _operator_cast_name(decl, src)
    if decl.type == "qualified_identifier":
        name = decl
        while name is not None and name.type == "qualified_identifier":
            name = name.child_by_field_name("name")
        if name is not None and name.type == "operator_cast":
            return src[decl.start_byte:name.start_byte].decode(errors="replace") + _operator_cast_name(name, src)
    if not is_function:
        return None
    # For qualified names like ShadowMap::Render, return the full qualified text
    return name_text(decl, src)


# C++: class_specifier, struct_specifier, namespace_definition
# C: struct_specifier, union_specifier, enum_specifier (only ever reach
# here as a real definition, see c's `is_tag_definition` boundary_filters entry)
def tag_specifier_name(node: Node, src: bytes) -> str | None:
    n = node.child_by_field_name("name")
    if n:
        return name_text(n, src)
    # fallback: first type_identifier child
    for c in node.children:
        if c.type == "type_identifier":
            return name_text(c, src)
    return None


# C typedef: walk down through pointer/array/function-pointer layers to
# the type_identifier leaf, don't assume it's the direct child.
def typedef_name(node: Node, src: bytes) -> str | None:
    n = node.child_by_field_name("declarator")
    seen = 0
    while n is not None and n.type != "type_identifier" and seen < 8:
        nxt = n.child_by_field_name("declarator")
        if nxt is None:
            named = [c for c in n.children if c.is_named]
            nxt = named[0] if len(named) == 1 else None
        n = nxt
        seen += 1
    if n is not None:
        return name_text(n, src)
    return None


def namespace_name(node: Node, src: bytes) -> str | None:
    n = node.child_by_field_name("name")
    if n:
        return name_text(n, src)
    for c in node.children:
        if c.type in ("namespace_identifier", "identifier"):
            return name_text(c, src)
    return None


def decorated_name(node: Node, src: bytes) -> str | None:
    # F1 re-entered the whole _extract_name chain here; decorated_definition's
    # only inner node types are the ones name_field resolves.
    # decorated_definition → definition child
    defn = node.child_by_field_name("definition")
    if defn:
        return name_field(defn, src)
    return None


def template_inner_name(node: Node, src: bytes) -> str | None:
    # F1 re-entered the whole _extract_name chain here; a template's inner
    # child is only ever one of the two node types checked below.
    for c in node.children:
        if c.type == "function_definition":
            return cpp_function_declarator_name(c, src)
        if c.type in ("class_specifier", "struct_specifier"):
            return tag_specifier_name(c, src)
    return None


def namespace_body_name(node: Node, src: bytes) -> str | None:
    """A namespace body chunked on its own is named by its namespace."""
    parent = node.parent
    if parent is not None and parent.type == "namespace_definition":
        return namespace_name(parent, src)
    return None


def arrow_var_name(node: Node, src: bytes) -> str | None:
    # Boundary is the whole declaration statement (`is_arrow_var_decl`); name
    # lives on the variable_declarator whose value is the function.
    for c in node.children:
        if c.type == "variable_declarator":
            value = c.child_by_field_name("value")
            if value is not None and value.type in ("arrow_function", "function_expression"):
                n = c.child_by_field_name("name")
                if n:
                    return name_text(n, src)
    return None


def extract_name(spec: LanguageSpec, node: Node, src: bytes) -> str | None:
    for rule in spec.name_rules:
        if node.type in rule.types:
            result = rule.fn(node, src)
            if result is not None:
                return result
    return None
