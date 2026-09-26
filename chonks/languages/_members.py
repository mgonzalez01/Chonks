"""C++ class-model readers: the members a declaration names, with type, arity and flags."""
from __future__ import annotations

from typing import TYPE_CHECKING

from chonks.core.symbols import FIELD, METHOD_DECLARATION, METHOD_DEFINITION

from ._ast import name_text
from ._naming import _operator_cast_name
from .spec import Member

if TYPE_CHECKING:
    from tree_sitter import Node


def strip_template_args(text: str) -> str:
    """`Foo<Bar<int>>::Inner` -> `Foo::Inner`, whitespace removed."""
    out: list[str] = []
    depth = 0
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0 and not ch.isspace():
            out.append(ch)
    return "".join(out)


def name_without_template_args(node: Node, src: bytes) -> str | None:
    return strip_template_args(name_text(node, src)) or None


def class_name(node: Node, src: bytes) -> str | None:
    n = node.child_by_field_name("name")
    return name_without_template_args(n, src) if n is not None else None


def using_namespace_name(node: Node, src: bytes) -> str | None:
    """`using namespace llvm::sys;` -> `llvm::sys`; None for `using Base::f;`."""
    if not any(c.type == "namespace" for c in node.children):
        return None
    target = next((c for c in node.named_children
                   if c.type in ("identifier", "namespace_identifier", "qualified_identifier")), None)
    return "".join(name_text(target, src).split()) if target is not None else None


_FUNCTION = "function_declarator"
_POINTERS = frozenset({"pointer_declarator", "reference_declarator"})
_MARKS = frozenset({"*", "&", "&&"})
_PASS_THROUGH = frozenset({
    "parenthesized_declarator", "attributed_declarator", "array_declarator", "init_declarator",
})
_CAST = "operator_cast"
_QUALIFIED = "qualified_identifier"
_NAMES = frozenset({"field_identifier", "identifier", "destructor_name", "operator_name"})
_CV = frozenset({"const", "volatile"})
_EMPTY_PARAMS = frozenset({"void"})


def _inner(decl: Node) -> Node | None:
    inner = decl.child_by_field_name("declarator")
    if inner is None:
        named = decl.named_children
        inner = named[0] if named else None
    return inner


def _unwrap(decl: Node | None) -> tuple[Node | None, str, Node | None, bool]:
    """(name node, `*`/`&` marks on the declared type, outermost function
    declarator, whether that declarator is a function pointer's)."""
    marks = ""
    func = None
    fn_pointer = False
    while decl is not None:
        if decl.type == _FUNCTION:
            if func is None:
                func = decl
        elif decl.type in _POINTERS:
            if func is not None:
                fn_pointer = True
            elif decl.children[0].type in _MARKS:
                marks += decl.children[0].type
        elif decl.type not in _PASS_THROUGH:
            return decl, marks, func, fn_pointer
        decl = _inner(decl)
    return None, marks, func, fn_pointer


def _member_name(n: Node, src: bytes) -> str | None:
    """`operator ==` -> `operator==`; `operator new` keeps its space."""
    if n.type == _CAST:
        return _operator_cast_name(n, src)
    if n.type not in _NAMES:
        return None
    text = " ".join(name_text(n, src).split())
    if n.type == "operator_name":
        symbol = text.removeprefix("operator").lstrip()
        if symbol and not (symbol[0].isalnum() or symbol[0] == "_"):
            return "operator" + symbol.replace(" ", "")
    return text


def _qualifier(n: Node, src: bytes) -> tuple[tuple[str, ...], Node | None]:
    """`List<T>::Element::erase` -> (("List", "Element"), the `erase` node)."""
    parts: list[str] = []
    while n is not None and n.type == _QUALIFIED:
        scope = n.child_by_field_name("scope")
        if scope is not None:
            parts.append(name_without_template_args(scope, src) or "")
        n = n.child_by_field_name("name")
    return tuple(parts), n


def _base_type(node: Node, src: bytes) -> str | None:
    t = node.child_by_field_name("type")
    if t is None:
        return None
    if t.child_by_field_name("body") is not None:
        n = t.child_by_field_name("name")
        return name_text(n, src) if n is not None else None
    parts = [name_text(c, src) for c in node.children
             if c == t or (c.type == "type_qualifier" and name_text(c, src) in _CV)]
    return " ".join(" ".join(parts).split())


def _signature(name_node: Node, func: Node | None) -> Node | None:
    """The node holding the parameters and the trailing qualifiers."""
    if name_node.type == _CAST:
        return name_node.child_by_field_name("declarator")
    return func


def _arity(sig: Node | None, src: bytes) -> tuple[int, int, bool] | None:
    params = sig.child_by_field_name("parameters") if sig is not None else None
    if params is None:
        return None
    required = defaulted = 0
    variadic = False
    for c in params.children:
        if c.type in ("variadic_parameter_declaration", "..."):
            variadic = True
        elif c.type == "optional_parameter_declaration":
            defaulted += 1
        elif c.type == "parameter_declaration" and name_text(c, src) not in _EMPTY_PARAMS:
            required += 1
    return required, required + defaulted, variadic


def _return_type(node: Node, name_node: Node, marks: str, sig: Node | None,
                 src: bytes) -> str | None:
    if name_node.type == _CAST:
        target = name_node.child_by_field_name("type")
        return name_text(target, src) if target is not None else None
    for c in sig.children if sig is not None else ():
        if c.type == "trailing_return_type":
            return " ".join(name_text(c, src).removeprefix("->").split())
    base = _base_type(node, src)
    return f"{base} {marks}" if base and marks else base


def _flags(node: Node, sig: Node | None, pure: bool, src: bytes) -> tuple[str, ...]:
    own = {c.type if c.type == "virtual" else name_text(c, src)
           for c in node.children if c.type in ("virtual", "storage_class_specifier")}
    trailing = {name_text(c, src) for c in (sig.children if sig is not None else ())
                if c.type in ("type_qualifier", "virtual_specifier")}
    return tuple(f for f, on in (("virtual", "virtual" in own), ("pure", pure),
                                 ("static", "static" in own), ("const", "const" in trailing),
                                 ("override", "override" in trailing)) if on)


def _method(node: Node, name: str, name_node: Node, marks: str, func: Node | None,
            src: bytes, scope: tuple[str, ...] = ()) -> Member:
    sig = _signature(name_node, func)
    value = node.child_by_field_name("default_value")
    pure = value is not None and name_text(value, src) == "0"
    kind = METHOD_DEFINITION if node.child_by_field_name("body") is not None else METHOD_DECLARATION
    return Member(name, kind, name_node.start_point[0] + 1,
                  type_text=_return_type(node, name_node, marks, sig, src),
                  arity=_arity(sig, src), flags=_flags(node, sig, pure, src), scope=scope)


def _field(node: Node, name: str, name_node: Node, marks: str, func: Node | None,
           decl: Node, src: bytes) -> Member:
    if func is not None:
        t = node.child_by_field_name("type")
        start = t.start_byte if t is not None else decl.start_byte
        text = (src[start:name_node.start_byte] + src[name_node.end_byte:decl.end_byte]).decode(
            errors="replace")
        type_text = " ".join(text.split())
    else:
        base = _base_type(node, src)
        type_text = f"{base} {marks}" if base and marks else base
    return Member(name, FIELD, name_node.start_point[0] + 1, type_text=type_text,
                  flags=_flags(node, None, False, src))


def read_member(node: Node, src: bytes, owner_name: str) -> list[Member]:
    """Fields, method declarations and inline method definitions of one
    declaration in a class body. A typeless declaration is a constructor,
    destructor or conversion, or else a macro call (`GDCLASS(Node, Object);`)."""
    typed = node.child_by_field_name("type") is not None
    out: list[Member] = []
    for decl in node.children_by_field_name("declarator"):
        name_node, marks, func, fn_pointer = _unwrap(decl)
        if name_node is None:
            continue
        name = _member_name(name_node, src)
        if name is None:
            continue
        if not typed and name_node.type not in (_CAST, "destructor_name") and name != owner_name:
            continue
        if name_node.type == _CAST or (func is not None and not fn_pointer):
            out.append(_method(node, name, name_node, marks, func, src))
        elif node.child_by_field_name("body") is None:
            out.append(_field(node, name, name_node, marks, func, decl, src))
    return out


def read_definition(node: Node, src: bytes, owner_name: str) -> list[Member]:
    """The member a qualified function definition outside its class defines
    (`void ns::Foo::bar() {}`), with the written qualifier as `scope`."""
    name_node, marks, func, fn_pointer = _unwrap(node.child_by_field_name("declarator"))
    if name_node is None or name_node.type != _QUALIFIED or fn_pointer:
        return []
    scope, last = _qualifier(name_node, src)
    if not scope or last is None or not all(scope):
        return []
    name = _member_name(last, src)
    if name is None or (func is None and last.type != _CAST):
        return []
    return [_method(node, name, last, marks, func, src, scope)]
