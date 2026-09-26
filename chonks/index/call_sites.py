"""Call-site evidence for a `calls` entry: how the callee is reached (racc), the
receiver chain's root (rhead) with its declared type when the enclosing function
states it, the member steps from root to callee (rpath), and the calling class (cls)."""

from typing import NamedTuple

from tree_sitter import Node

from chonks.core.refresh import register_refresh
from chonks.languages import table as _lang_table
from chonks.languages._ast import name_call_target as _name_call_target
from chonks.languages.spec import CallSiteSpec as _CallSiteSpec

_CALL_SITE_SPECS = _lang_table("call_sites")

_HEAD_TEXT_MAX = 64
_TYPE_TEXT_MAX = 160
_TYPE_LOOKUP_DEPTH = 4


class CallSites:
    """Caches over one tree, keyed by node id. Holding the root keeps the
    tree alive, so no other tree can reuse its ids while this is cached."""
    __slots__ = ("cs", "src", "root", "blocks", "scopes", "scope_owner", "classes", "evidence")

    def __init__(self, cs: _CallSiteSpec, src: bytes, root: Node):
        self.cs = cs
        self.src = src
        self.root = root
        self.blocks: dict[int, list] = {}
        self.scopes: dict[int, list] = {}
        self.scope_owner: dict[int, Node | None] = {}
        self.classes: dict[int | None, str | None] = {}
        self.evidence: dict[int, dict] = {}


_last_sites: list[CallSites] = []


def sites_for(node: Node, lang: str, src: bytes) -> CallSites | None:
    """The caches for `node`'s tree, reused across calls on the same tree;
    None for a language without call-site facts."""
    cs = _CALL_SITE_SPECS.get(lang)
    if cs is None:
        return None
    root = node
    while root.parent is not None:
        root = root.parent
    if _last_sites:
        last = _last_sites[0]
        if last.src is src and last.cs is cs and last.root.id == root.id:
            return last
    sites = CallSites(cs, src, root)
    _last_sites[:] = [sites]
    return sites


def _text(n: Node, src: bytes) -> str:
    return " ".join(src[n.start_byte:n.end_byte].decode(errors="replace").split())


def _bare_name(n: Node, sites: CallSites) -> str:
    wrapper = sites.cs.type_argument_wrappers.get(n.type)
    inner = n.child_by_field_name(wrapper[0]) if wrapper is not None else None
    return _text(inner if inner is not None else n, sites.src)


def _member_name(n: Node, sites: CallSites) -> str:
    return _name_call_target(n, sites.src) or _text(n, sites.src)


def _first_named(n: Node) -> "Node | None":
    return next((c for c in n.children if c.is_named), None)


def _inner_declarator(d: Node, sites: CallSites) -> "Node | None":
    inner = d.child_by_field_name(sites.cs.declarator_field)
    return inner if inner is not None else _first_named(d)


def _declared_names(d: "Node | None", sites: CallSites) -> list[tuple[str, str | None]]:
    """(name, type suffix) under a declarator's pointer/reference/array
    layers; a structured binding's names get no type (suffix None)."""
    cs = sites.cs
    suffix = ""
    for _ in range(16):
        if d is None:
            return []
        if d.type in cs.declarator_names:
            return [(_text(d, sites.src), suffix)]
        if d.type in cs.binding_declarators:
            return [(_text(c, sites.src), None) for c in d.children if c.type in cs.declarator_names]
        marker = cs.declarators.get(d.type)
        if marker is None:
            return []
        suffix += marker
        d = _inner_declarator(d, sites)
    return []


def _written_type(owner: Node, suffix: str, sites: CallSites) -> str | None:
    cs = sites.cs
    tn = owner.child_by_field_name(cs.type_field)
    if tn is None:
        return None
    if tn.type in cs.classes:
        name = tn.child_by_field_name(cs.name_field)
        if name is None:
            return None
        tn = name
    quals = [q for c in owner.children
             if c.type in cs.type_qualifiers and c.start_byte < tn.start_byte
             and (q := _text(c, sites.src)) in cs.type_qualifiers[c.type]]
    written = " ".join(quals + [_text(tn, sites.src)])
    if len(written) > _TYPE_TEXT_MAX:
        return None
    return f"{written} {suffix}" if suffix else written


class _Declared(NamedTuple):
    name: str
    via: str
    owner: Node            # the declaration, parameter or loop holding the type
    value: "Node | None"   # initializer
    suffix: str | None     # None: declared without a type


def _declared_in(owner: Node, via: str, sites: CallSites) -> list[_Declared]:
    cs = sites.cs
    declarators = owner.children_by_field_name(cs.declarator_field)
    out = []
    for d in declarators:
        value = d.child_by_field_name(cs.value_field)
        if value is None and len(declarators) == 1:
            value = owner.child_by_field_name(cs.value_field)
        out.extend(_Declared(name, via, owner, value, suffix) for name, suffix in _declared_names(d, sites))
    return out


def _declared_type(decl: _Declared, sites: CallSites, depth: int) -> str | None:
    if decl.suffix is None:
        return None
    tn = decl.owner.child_by_field_name(sites.cs.type_field)
    if tn is not None and tn.type in sites.cs.inferred_types:
        return _value_type(decl.value, sites, depth + 1, inferred=True)
    return _written_type(decl.owner, decl.suffix, sites)


def _block_declarations(block: Node, sites: CallSites) -> list[tuple[int, _Declared]]:
    cached = sites.blocks.get(block.id)
    if cached is not None:
        return cached
    cs = sites.cs
    out: list[tuple[int, _Declared]] = []

    def scan(n: Node) -> None:
        for c in n.children:
            if c.type in cs.declarations:
                out.extend((c.start_byte, d) for d in _declared_in(c, "local", sites))
            elif c.type in cs.transparent_blocks:
                scan(c)

    scan(block)
    sites.blocks[block.id] = out
    return out


def _scoped_declarations(stmt: Node, fields: tuple[str, ...], sites: CallSites) -> list[_Declared]:
    cached = sites.scopes.get(stmt.id)
    if cached is not None:
        return cached
    cs = sites.cs
    out: list[_Declared] = []
    stack = [s for f in fields if (s := stmt.child_by_field_name(f)) is not None]
    while stack:
        n = stack.pop()
        if n.type in cs.declarations:
            out.extend(_declared_in(n, "local", sites))
        elif n.type not in cs.blocks and n.type not in cs.parameter_owners:
            stack.extend(c for c in n.children if c.is_named)
    sites.scopes[stmt.id] = out
    return out


def _parameter_list(owner: Node, sites: CallSites) -> "Node | None":
    cs = sites.cs
    field = cs.parameter_owners[owner.type]
    d = owner if field is None else owner.child_by_field_name(field)
    for _ in range(8):
        if d is None:
            return None
        params = d.child_by_field_name(cs.parameters_field)
        if params is not None:
            return params
        d = _inner_declarator(d, sites)
    return None


def _lookup(name: str, at: Node, sites: CallSites, depth: int) -> tuple[str, str | None] | None:
    """(via, declared type) of the innermost declaration of `name` visible at
    `at` within its function, or None."""
    cs = sites.cs
    prev, p = at, at.parent
    while p is not None:
        found: _Declared | None = None
        if p.type in cs.blocks:
            for start, decl in _block_declarations(p, sites):
                if start >= prev.start_byte:
                    break
                if decl.name == name:
                    found = decl
        elif p.type in cs.range_loops:
            outside = p.child_by_field_name(cs.range_loops[p.type])
            if outside is None or outside.id != prev.id:
                found = next((d for d in _declared_in(p, "local", sites) if d.name == name), None)
        elif p.type in cs.declaration_scopes:
            found = next((d for d in _scoped_declarations(p, cs.declaration_scopes[p.type], sites)
                          if d.name == name and d.owner.end_byte <= at.start_byte), None)
        if found is None and p.type in cs.parameter_owners:
            params = _parameter_list(p, sites)
            found = next((d for param in (params.children if params is not None else ())
                          if param.type in cs.parameters
                          for d in _declared_in(param, "param", sites) if d.name == name), None)
        if found is not None:
            return found.via, _declared_type(found, sites, depth)
        if p.type in cs.functions:
            return None
        prev, p = p, p.parent
    return None


def _type_argument_result(callee: "Node | None", sites: CallSites) -> str | None:
    """T for a call like `static_cast<T>(x)` or `make_unique<T>()`."""
    cs = sites.cs
    n = callee
    while n is not None and n.type in cs.scope_access:
        n = n.child_by_field_name(cs.scope_access[n.type][1])
    if n is None or n.type not in cs.type_argument_wrappers:
        return None
    name_field, args_field = cs.type_argument_wrappers[n.type]
    name, args = n.child_by_field_name(name_field), n.child_by_field_name(args_field)
    if name is None or args is None or not cs.type_argument_calls.fullmatch(_text(name, sites.src)):
        return None
    named = [c for c in args.children if c.is_named]
    return _text(named[0], sites.src) if len(named) == 1 else None


def _value_type(v: "Node | None", sites: CallSites, depth: int, inferred: bool) -> str | None:
    """Type an expression states itself (cast, new, T{...}, cast-like template
    call). For an `auto` initializer also a bare call's callee and a name's
    declared type."""
    cs = sites.cs
    while v is not None and v.type in cs.parenthesized:
        v = _first_named(v)
    if v is None or depth > _TYPE_LOOKUP_DEPTH:
        return None
    t = v.type
    for table, suffix in ((cs.casts, ""), (cs.allocations, cs.allocation_suffix), (cs.typed_literals, "")):
        if t in table:
            tn = v.child_by_field_name(table[t])
            return _text(tn, sites.src) + suffix if tn is not None else None
    if t in cs.calls:
        callee = v.child_by_field_name(cs.calls[t])
        typed = _type_argument_result(callee, sites)
        if typed is not None or not inferred or callee is None:
            return typed
        if callee.type in cs.names or callee.type in cs.scope_access or callee.type in cs.type_argument_wrappers:
            return _text(callee, sites.src)
        return None
    if inferred and t in cs.names:
        found = _lookup(_text(v, sites.src), v, sites, depth + 1)
        return found[1] if found is not None else None
    return None


def _scope_text(n: Node, sites: CallSites) -> str | None:
    """`ns::X` for `ns::X::f`: the qualifier as written, or None."""
    cs = sites.cs
    if n.type not in cs.scope_access:
        return None
    cur = n
    while True:
        name = cur.child_by_field_name(cs.scope_access[cur.type][1])
        if name is None or name.type not in cs.scope_access:
            break
        cur = name
    if name is None:
        return None
    scope = " ".join(sites.src[n.start_byte:name.start_byte].decode(errors="replace").split())
    scope = scope.removesuffix(cs.scope_operator).strip().removeprefix(cs.scope_operator).strip()
    return scope or None


def _head(name: str, via: str, typ: str | None) -> dict:
    return {"name": name, "via": via, "type": typ}


def _receiver_chain(obj: "Node | None", sites: CallSites) -> tuple[dict, list[str]]:
    """(head, steps) of a receiver expression, steps in source order."""
    cs, src = sites.cs, sites.src
    steps: list[str] = []
    n = obj
    head = _head("", "unknown", None)
    while n is not None:
        t = n.type
        if t in cs.parenthesized:
            n = _first_named(n)
            continue
        if t in cs.member_access:
            obj_field, _op, member_field = cs.member_access[t]
            member = n.child_by_field_name(member_field)
            steps.append(_member_name(member, sites) if member is not None else "")
            n = n.child_by_field_name(obj_field)
            continue
        if t in cs.subscripts:
            steps.append("[]")
            n = n.child_by_field_name(cs.subscripts[t])
            continue
        if t in cs.dereferences:
            operand, op_field, op = cs.dereferences[t]
            op_node = n.child_by_field_name(op_field)
            if op_node is not None and _text(op_node, src) == op:
                steps.append(op)
                n = n.child_by_field_name(operand)
                continue
        if t in cs.calls:
            callee = n.child_by_field_name(cs.calls[t])
            if callee is None:
                break
            typed = _type_argument_result(callee, sites)
            if typed is not None:
                head = _head(_member_name(callee, sites) + "()", "unknown", typed)
                break
            if callee.type in cs.member_access:
                obj_field, _op, member_field = cs.member_access[callee.type]
                member = callee.child_by_field_name(member_field)
                steps.append((_member_name(member, sites) if member is not None else "") + "()")
                n = callee.child_by_field_name(obj_field)
                continue
            scope = _scope_text(callee, sites)
            called = _member_name(callee, sites) + "()"
            if scope is not None:
                steps.append(called)
                head = _head(scope, "type", scope)
            else:
                head = _head(called, "unknown", None)
            break
        if t in cs.scope_access:
            scope = _scope_text(n, sites)
            if scope is not None:
                steps.append(_member_name(n, sites))
                head = _head(scope, "type", scope)
                break
        if t in cs.this_expressions:
            head = _head(_text(n, src), "this", None)
            break
        if t in cs.names:
            name = _text(n, src)
            found = _lookup(name, n, sites, 0)
            head = _head(name, *found) if found is not None else _head(name, "unknown", None)
            break
        head = _head(_text(n, src)[:_HEAD_TEXT_MAX], "unknown", _value_type(n, sites, 0, inferred=False))
        break
    steps.reverse()
    return head, steps


def _scope_parts(q: Node, sites: CallSites) -> list[str]:
    """Qualifier parts of `A<T>::B::f`, template arguments stripped: ['A', 'B']."""
    cs = sites.cs
    parts = []
    cur: Node | None = q
    while cur is not None and cur.type in cs.scope_access:
        scope_field, name_field = cs.scope_access[cur.type]
        scope = cur.child_by_field_name(scope_field)
        if scope is not None:
            parts.append(_bare_name(scope, sites))
        cur = cur.child_by_field_name(name_field)
    return parts


def _qualified_class(c: Node, sites: CallSites) -> str | None:
    """`ns::Outer::Inner` for a class node: enclosing named namespaces and
    classes, template arguments stripped."""
    cs = sites.cs
    parts: list[str] = []
    n: Node | None = c
    while n is not None:
        if n.type in cs.classes or n.type in cs.namespaces:
            name = n.child_by_field_name(cs.name_field)
            if name is not None:
                own = _scope_parts(name, sites) + [_bare_name(_last_name(name, sites), sites)]
                parts[:0] = own
            elif n is c:
                return None
        n = n.parent
    return "::".join(parts) or None


def _last_name(n: Node, sites: CallSites) -> Node:
    cs = sites.cs
    while n.type in cs.scope_access:
        nxt = n.child_by_field_name(cs.scope_access[n.type][1])
        if nxt is None:
            break
        n = nxt
    return n


def _definition_qualifier(fn: Node, sites: CallSites) -> list[str]:
    """['ns', 'Thread'] for `void ns::Thread::start()`, [] when unqualified."""
    cs = sites.cs
    d = fn.child_by_field_name(cs.declarator_field)
    for _ in range(8):
        if d is None:
            return []
        if d.type in cs.scope_access:
            return _scope_parts(d, sites)
        d = _inner_declarator(d, sites)
    return []


def _enclosing_scope(node: Node, sites: CallSites) -> "Node | None":
    """The nearest function or class above `node`, memoized for every node walked."""
    cs = sites.cs
    walked = []
    p = node.parent
    while p is not None:
        if p.id in sites.scope_owner:
            p = sites.scope_owner[p.id]
            break
        if p.type in cs.functions or p.type in cs.classes:
            break
        walked.append(p.id)
        p = p.parent
    for i in walked:
        sites.scope_owner[i] = p
    return p


def _calling_class(node: Node, sites: CallSites) -> str | None:
    cs = sites.cs
    p = _enclosing_scope(node, sites)
    key = p.id if p is not None else None
    if key in sites.classes:
        return sites.classes[key]
    result = None
    if p is not None and p.type in cs.classes:
        result = _qualified_class(p, sites)
    elif p is not None:
        qualifier = _definition_qualifier(p, sites)
        if qualifier:
            namespaces = []
            q = p.parent
            while q is not None:
                if q.type in cs.namespaces and (name := q.child_by_field_name(cs.name_field)) is not None:
                    namespaces.insert(0, _text(name, sites.src))
                q = q.parent
            result = "::".join(namespaces + qualifier)
        else:
            q = _enclosing_scope(p, sites)
            if q is not None and q.type in cs.classes:
                result = _qualified_class(q, sites)
    sites.classes[key] = result
    return result


def call_evidence(call_node: Node, callee: Node, sites: CallSites) -> dict:
    """The racc/rhead/rpath/cls keys of one call's fingerprint."""
    cs = sites.cs
    cached = sites.evidence.get(call_node.id)
    if cached is not None:
        return cached
    ev: dict = {}
    sites.evidence[call_node.id] = ev
    if callee.type in cs.member_access:
        obj_field, op_field, _member = cs.member_access[callee.type]
        op = callee.child_by_field_name(op_field)
        if op is not None:
            ev["racc"] = _text(op, sites.src)
        head, steps = _receiver_chain(callee.child_by_field_name(obj_field), sites)
        ev["rhead"] = head
        if steps:
            ev["rpath"] = steps
    else:
        scope = _scope_text(callee, sites)
        if scope is not None:
            ev["racc"] = cs.scope_operator
            ev["rhead"] = _head(scope, "type", scope)
    cls = _calling_class(call_node, sites)
    if cls:
        ev["cls"] = cls
    return ev


def _refresh_from_registry() -> None:
    global _CALL_SITE_SPECS
    _CALL_SITE_SPECS = _lang_table("call_sites")
    _last_sites.clear()


register_refresh(_refresh_from_registry)
