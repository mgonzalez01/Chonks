"""Class model: the members and bases of each class, read from the segmenter's parse tree."""

from operator import itemgetter

from tree_sitter import Node

from chonks.core.refresh import register_refresh
from chonks.index.refs_extract import _apply_rule
from chonks.languages import table as _lang_table
from chonks.languages.spec import ClassModelSpec, Member

_CLASS_MODELS = _lang_table("class_model")


def _refresh_from_registry() -> None:
    global _CLASS_MODELS
    _CLASS_MODELS = _lang_table("class_model")


register_refresh(_refresh_from_registry)

# A scope is (qualified name parts, whether the innermost part is a class).
Scope = tuple[tuple[str, ...], bool]


def collect(root: Node, lang: str, src: bytes) -> dict[str, list[dict]]:
    """"members", "bases" and "using_namespaces" rows of the tree, in line
    order. Like the symbol walk it recurses into boundaries, but not into
    function bodies."""
    model: dict[str, list[dict]] = {"members": [], "bases": [], "using_namespaces": []}
    spec: ClassModelSpec | None = _CLASS_MODELS.get(lang)  # type: ignore[assignment]
    if spec is None:
        return model
    stack: list[tuple[Node, Scope]] = [(root, ((), False))]
    while stack:
        node, scope = stack.pop()
        inner = _visit(node, scope, spec, src, lang, model)
        if inner is not None:
            stack.extend((child, inner) for child in node.children)
    line = itemgetter("line")
    return {key: sorted(rows, key=line) for key, rows in model.items()}


def _row(owner: str, m: Member, lang: str) -> dict:
    arity = m.arity or (None, None, None)
    return {
        "owner": owner, "name": m.name, "kind": m.kind, "type_text": m.type_text,
        "arity_min": arity[0], "arity_max": arity[1],
        "variadic": None if m.arity is None else int(arity[2]),
        "flags": ",".join(m.flags) or None, "language": lang, "line": m.line,
    }


def _read_body(body: Node, spec: ClassModelSpec, src: bytes, owner: str,
               owner_name: str, lang: str, out: list[dict]) -> None:
    for child in body.children:
        if child.type in spec.transparent:
            _read_body(child, spec, src, owner, owner_name, lang, out)
            continue
        read = spec.members.get(child.type)
        if read is not None:
            out.extend(_row(owner, m, lang) for m in read(child, src, owner_name))


def _visit(node: Node, scope: Scope, spec: ClassModelSpec, src: bytes, lang: str,
           model: dict[str, list[dict]]) -> Scope | None:
    """Records what `node` declares and returns the scope of its children,
    None for a function, whose body declares no members. An anonymous class
    in a class body (a union of fields) adds its members to the enclosing class."""
    parts, in_class = scope
    sep = spec.separator
    name_fn = spec.using_namespaces.get(node.type)
    if name_fn is not None:
        name = name_fn(node, src)
        if name and not in_class:
            model["using_namespaces"].append(
                {"scope": sep.join(parts) or None, "namespace": name, "line": node.start_point[0] + 1})
        return scope
    name_fn = spec.namespaces.get(node.type)
    if name_fn is not None:
        name = name_fn(node, src)
        return (parts + tuple(name.split(sep)) if name else parts), False
    name_fn = spec.classes.get(node.type)
    if name_fn is not None:
        body = node.child_by_field_name(spec.body_field)
        if body is None:
            return scope
        name = name_fn(node, src)
        if name:
            parts = parts + tuple(name.split(sep))
        elif not in_class:
            return scope
        owner = sep.join(parts)
        _read_body(body, spec, src, owner, parts[-1], lang, model["members"])
        if name:
            refs: dict[str, list[str]] = {"bases": []}
            _apply_rule(node, spec.bases, refs, src, lang)
            line = node.start_point[0] + 1
            model["bases"].extend({"owner": owner, "base": b, "line": line} for b in refs["bases"])
        return parts, True
    read = spec.functions.get(node.type)
    if read is not None:
        if not in_class:
            for m in read(node, src, ""):
                model["members"].append(_row(sep.join(parts + m.scope), m, lang))
        return None
    return scope
