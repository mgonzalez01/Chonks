"""Call targets from the class model: the receiver's class from call-site
evidence, then the called method's rows in that class or its bases."""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

from chonks.core.refresh import register_refresh
from chonks.core.symbols import FIELD, METHOD_DECLARATION, METHOD_DEFINITION
from chonks.languages import get_or_none as _lang_spec, table as _lang_table
from chonks.languages.spec import ClassModelSpec

if TYPE_CHECKING:
    from chonks.storage.store import Store

_CLASS_MODELS = _lang_table("class_model")
_CLASS_MODEL_META_KEY = "class_model_version"
_EVIDENCE_KEYS = ("racc", "rhead", "cls")
_CALL_STEP = "()"
_SUBSCRIPT_STEP = "[]"
_MAX_BASE_DEPTH = 16


def _refresh_from_registry() -> None:
    global _CLASS_MODELS
    _CLASS_MODELS = _lang_table("class_model")


register_refresh(_refresh_from_registry)

# Outcomes of resolve(); the first six carry targets.
TYPED = "members_typed"
CALLING_CLASS = "members_cls"
STATIC = "members_static"
DECL_ONLY = "members_decl_only"
NO_FIT = "members_nofit"
FIELD_CALL = "members_field"
NOT_IN_CLASS = "not_in_class"
NO_CLASS = "no_class"
UNTYPED = "untyped"
NOT_A_CLASS = "static_not_class"
BARE_NOT_IN_CLASS = "bare_not_in_class"

# Targets of a call whose evidence finds its class without the method, or
# names a class the model does not have: [] links nothing, None takes the name index.
UNRESOLVED_TARGETS: dict[str, "list[str] | None"] = {NOT_IN_CLASS: [], NO_CLASS: []}


class Resolution(NamedTuple):
    targets: list[str] | None
    outcome: str
    guessed: bool = False  # a name matched only the end of a class's qualified name


class Context(NamedTuple):
    """Where a name is written: scopes to try innermost first ("" is the
    global one), then the namespaces using-directives make visible."""
    scopes: tuple[str, ...]
    usings: tuple[str, ...]


class _Row(NamedTuple):
    owner: str
    name: str
    kind: str
    type_text: str | None
    arity_min: int | None
    arity_max: int | None
    variadic: int | None
    line: int
    chunk_id: str | None
    path: str


def _row(r: dict) -> _Row:
    intern = sys.intern
    return _Row(intern(r["owner"]), intern(r["name"]), intern(r["kind"]), r["type_text"], r["arity_min"],
                r["arity_max"], r["variadic"], r["line"], r["chunk_id"] and intern(r["chunk_id"]),
                intern(r["path"]))


class _Receiver(NamedTuple):
    classes: tuple[str, ...]
    pointers: int
    args: tuple[str, ...] = ()  # template arguments as written
    args_at: Context | None = None  # where they were written
    args_through: "_Receiver | None" = None  # what their own template parameters stand for


def has_evidence(entry: object) -> bool:
    return isinstance(entry, dict) and any(k in entry for k in _EVIDENCE_KEYS)


def _class_kinds() -> list[str]:
    return sorted({t for lang in _CLASS_MODELS for t in _lang_spec(lang).class_like_chunk_types})


def members_index(store: "Store") -> "MemberIndex | None":
    """None for an index written before the class model."""
    if store.get_meta(_CLASS_MODEL_META_KEY) is None:
        return None
    return MemberIndex(store, _class_kinds())


def _parents(name: str, sep: str) -> tuple[str, ...]:
    """`a::b::C` -> ("a::b", "a", "")."""
    parts = name.split(sep)
    return tuple(sep.join(parts[:i]) for i in range(len(parts) - 1, -1, -1))


def _fits(row: _Row, arity: int | None) -> bool:
    if arity is None or row.arity_min is None:
        return True
    return arity >= row.arity_min and (bool(row.variadic) or arity <= row.arity_max)


def _targets(hits: list[tuple[str, _Row]], arity: int | None, how: str) -> Resolution:
    """Definitions that fit the arity, else declarations that do, else every definition.
    A definition outside its class also fits through a fitting declaration of its width."""
    methods = [r for _c, r in hits if r.kind != FIELD]
    if not methods:
        return Resolution([], FIELD_CALL)
    defs = [r for r in methods if r.kind == METHOD_DEFINITION]
    decls = [r for r in methods if r.kind == METHOD_DECLARATION]
    fit_decls = [r for r in decls if _fits(r, arity)]
    widths = {(r.arity_max, r.variadic) for r in fit_decls}
    fit_defs = [r for r in defs if _fits(r, arity) or (r.arity_max, r.variadic) in widths]
    if fit_defs:
        rows, outcome = fit_defs, how
    elif fit_decls:
        rows, outcome = fit_decls, DECL_ONLY
    else:
        rows, outcome = defs or decls, NO_FIT
    ids = list(dict.fromkeys(r.chunk_id for r in rows if r.chunk_id))
    return Resolution(ids, outcome) if ids else Resolution(None, outcome)


def _unresolved(outcome: str) -> Resolution:
    return Resolution(UNRESOLVED_TARGETS.get(outcome), outcome)


class MemberIndex:
    """The class model read through the store, memoized per class and file."""

    def __init__(self, store: "Store", class_kinds: list[str]):
        self._store = store
        self.known = store.get_class_owners(class_kinds)
        self._by_last: dict[str, dict[str, list[str]]] = {}
        self._rows: dict[str, dict[str, list[_Row]]] = {}
        self._bases: dict[str, list[str]] = {}
        self._complete: dict[str, bool] = {}
        self._usings: dict[str, list[dict]] = {}
        self._qualified: dict[tuple[str, Context], tuple[list[str], bool]] = {}
        self._contexts: dict[tuple, Context] = {}
        self._file_scopes: dict[str, tuple[str, ...]] = {}
        self.guessed = False

    # ---------------------------------------------------------------- tables

    def _last(self, sep: str) -> dict[str, list[str]]:
        got = self._by_last.get(sep)
        if got is None:
            got = defaultdict(list)
            for owner in sorted(self.known):
                got[owner.rsplit(sep, 1)[-1]].append(owner)
            self._by_last[sep] = got
        return got

    def usings(self, path: str) -> list[dict]:
        got = self._usings.get(path)
        if got is None:
            got = self._usings[path] = self._store.get_using_namespaces([path])
        return got

    def _remap(self, row: _Row, sep: str) -> set[str]:
        """The classes a member defined outside its class under a shorter
        owner may belong to, through its file's using-directives."""
        owner = row.owner
        out: set[str] = set()
        for u in self.usings(row.path):
            if u["line"] > row.line:
                break
            scope, ns = u["scope"], u["namespace"]
            if scope and not owner.startswith(scope + sep):
                continue
            rest = owner[len(scope) + len(sep):] if scope else owner
            if ns.startswith(sep):
                out.add(ns[len(sep):] + sep + rest)
            else:
                out.add(ns + sep + rest)
                if scope:
                    out.add(scope + sep + ns + sep + rest)
        return out

    def rows(self, owner: str, sep: str) -> dict[str, list[_Row]]:
        """name -> member rows of `owner`, including those a file's
        using-directive qualifies: `void Foo::f() {}` after `using namespace ns;`."""
        got = self._rows.get(owner)
        if got is None:
            parts = owner.split(sep)
            shorter = [sep.join(parts[i:]) for i in range(1, len(parts))]
            got = defaultdict(list)
            for r in map(_row, self._store.get_members_by_owners([owner, *shorter])):
                if r.owner == owner or (r.owner not in self.known and owner in self._remap(r, sep)):
                    got[r.name].append(r)
            self._rows[owner] = got
        return got

    def bases(self, owner: str, sep: str) -> list[str]:
        got = self._bases.get(owner)
        if got is None:
            got = []
            complete = owner in self.known
            for b in self._store.get_bases([owner]):
                at = self._context(_parents(owner, sep), b["path"], b["line"], owner, sep)
                found = self.qualify(b["base"], at, sep)
                complete = complete and bool(found)
                got.extend(q for q in found if q != owner and q not in got)
            self._bases[owner] = got
            self._complete[owner] = complete
        return got

    def complete(self, classes: tuple[str, ...] | list[str], sep: str) -> bool:
        """Whether the model has the classes and every base above them."""
        seen: set[str] = set()
        todo = list(classes)
        while todo:
            c = todo.pop()
            if c in seen:
                continue
            seen.add(c)
            todo.extend(self.bases(c, sep))
            if not self._complete[c]:
                return False
        return True

    # ---------------------------------------------------------------- names

    def _context(self, scopes: tuple[str, ...], path: str, line: int, within: str | None,
                 sep: str) -> Context:
        """`within` is the enclosing class or namespace, None when unknown;
        a directive inside a namespace applies only within it."""
        usings = self.usings(path)
        visible = next((i for i, u in enumerate(usings) if u["line"] > line), len(usings))
        key = (scopes, path, visible, within)
        got = self._contexts.get(key)
        if got is None:
            names: list[str] = []
            for u in usings[:visible]:
                scope, ns = u["scope"], u["namespace"]
                if scope and within is not None and within != scope and not within.startswith(scope + sep):
                    continue
                if ns.startswith(sep):
                    names.append(ns[len(sep):])
                    continue
                if scope:
                    names.append(scope + sep + ns)
                names.append(ns)
            got = self._contexts[key] = Context(scopes, tuple(dict.fromkeys(names)))
        return got

    def _class_context(self, owner: str, path: str, line: int, sep: str) -> Context:
        """Names written in a member of `owner`: its own scope and its bases'
        (nested types), then the enclosing scopes."""
        scopes = (owner, *self.bases(owner, sep), *_parents(owner, sep))
        return self._context(tuple(dict.fromkeys(scopes)), path, line, owner, sep)

    def qualify(self, name: str, at: Context, sep: str) -> list[str]:
        """The known classes `name` denotes where it is written: the innermost scope's,
        else the using-directives', else every class ending with it (a guess)."""
        key = (name, at)
        got = self._qualified.get(key)
        if got is None:
            classes: list[str] = []
            for scope in at.scopes:
                q = f"{scope}{sep}{name}" if scope else name
                if q in self.known:
                    classes = [q]
                    break
            if not classes:
                classes = [q for ns in at.usings if (q := f"{ns}{sep}{name}") in self.known]
            guess = False
            if not classes:
                tail = sep + name
                classes = [k for k in self._last(sep).get(name.rsplit(sep, 1)[-1], ()) if k.endswith(tail)]
                guess = bool(classes)
            got = self._qualified[key] = (classes, guess)
        self.guessed = self.guessed or got[1]
        return got[0]

    def file_scopes(self, path: str, sep: str) -> tuple[str, ...]:
        """The namespaces a file declares classes in, innermost first, for
        code whose enclosing namespace the evidence does not say."""
        got = self._file_scopes.get(path)
        if got is None:
            names = {p for owner in self._store.get_member_owners_by_path(path)
                     for p in _parents(owner, sep) if p and p not in self.known}
            got = self._file_scopes[path] = (*sorted(names, key=lambda n: (-n.count(sep), n)), "")
        return got

    def _calling_classes(self, cls: str, path: str, line: int, sep: str) -> list[str]:
        """The calling class. An out-of-class qualifier is looked up from its enclosing
        namespaces outward, so every split of `cls` into namespaces and qualifier is tried."""
        parts = cls.split(sep)
        for k in range(1, len(parts) + 1):
            for i in range(len(parts) - k, -1, -1):
                q = sep.join(parts[:i] + parts[-k:])
                if q in self.known:
                    return [q]
        return self.qualify(cls, self._context(("",), path, line, cls, sep), sep)

    # ---------------------------------------------------------------- lookup

    def find(self, classes: tuple[str, ...] | list[str], name: str, sep: str) -> list[tuple[str, _Row]]:
        """(class, row) for `name` in the first of `classes` and their bases,
        breadth-first, that has it."""
        seen = set(classes)
        layer = list(classes)
        for _ in range(_MAX_BASE_DEPTH):
            if not layer:
                break
            hits = [(c, r) for c in layer for r in self.rows(c, sep).get(name, ())]
            if hits:
                return hits
            nxt = []
            for c in layer:
                for b in self.bases(c, sep):
                    if b not in seen:
                        seen.add(b)
                        nxt.append(b)
            layer = nxt
        return []

    def _type_of(self, text: str | None, at: Context, spec: ClassModelSpec,
                 through: _Receiver | None = None) -> _Receiver | None:
        """The receiver a written type makes. A name that is no class, reached
        through a class written with one template argument, is that argument."""
        ref = spec.type_ref(text) if text and spec.type_ref is not None else None
        if ref is None:
            return None
        sep = spec.separator
        classes = self.qualify(ref.name, at, sep)
        if classes:
            return _Receiver(tuple(classes), ref.pointers, ref.args, at, through)
        if (through is not None and len(through.args) == 1 and through.args_at is not None
                and sep not in ref.name and not ref.args):
            inner = self._type_of(through.args[0], through.args_at, spec, through.args_through)
            if inner is not None:
                return inner._replace(pointers=inner.pointers + ref.pointers)
        return None

    def _member_types(self, hits: list[tuple[str, _Row]], spec: ClassModelSpec,
                      through: _Receiver | None) -> _Receiver | None:
        states = []
        for c, r in hits:
            at = self._class_context(c, r.path, r.line, spec.separator)
            st = self._type_of(r.type_text, at, spec, through)
            if st is not None:
                states.append(st)
        if not states:
            return None
        classes = tuple(dict.fromkeys(c for st in states for c in st.classes))
        return states[0]._replace(classes=classes)

    def _arrow(self, recv: _Receiver, spec: ClassModelSpec) -> tuple[bool, _Receiver | None]:
        """Whether the class has `operator->`, and the receiver it returns."""
        member = spec.operator_members.get(spec.pointer_access or "")
        hits = [h for h in self.find(recv.classes, member, spec.separator) if h[1].kind != FIELD] \
            if member else []
        st = self._member_types(hits, spec, recv)
        return bool(hits), st._replace(pointers=max(st.pointers - 1, 0)) if st is not None else None

    def _member(self, recv: _Receiver, name: str, spec: ClassModelSpec) -> list[tuple[str, _Row]]:
        """`name` reached from `recv` without knowing whether `.` or `->`
        wrote it: a class value without the member goes through its `operator->`."""
        hits = self.find(recv.classes, name, spec.separator)
        if hits or recv.pointers:
            return hits
        _has, arrow = self._arrow(recv, spec)
        return self.find(arrow.classes, name, spec.separator) if arrow is not None else []

    def _step(self, recv: _Receiver, step: str, spec: ClassModelSpec) -> _Receiver | None:
        if step.endswith(_CALL_STEP):
            hits = [h for h in self._member(recv, step[:-len(_CALL_STEP)], spec) if h[1].kind != FIELD]
            return self._member_types(hits, spec, recv)
        member = spec.operator_members.get(step)
        if member is not None:
            if recv.pointers:
                return recv._replace(pointers=recv.pointers - 1)
            hits = [h for h in self.find(recv.classes, member, spec.separator) if h[1].kind != FIELD]
            if not hits:
                # A field declared as an array has its element's type.
                return recv if step == _SUBSCRIPT_STEP else None
            return self._member_types(hits, spec, recv)
        hits = [h for h in self._member(recv, step, spec) if h[1].kind == FIELD]
        return self._member_types(hits, spec, recv)

    def _enclosing(self, classes: list[str], sep: str) -> list[str]:
        """The calling classes, then the classes they are nested in."""
        out = list(classes)
        for c in classes:
            out.extend(p for p in _parents(c, sep) if p in self.known and p not in out)
        return out

    def _function_result(self, text: str, at: Context, cls: list[str],
                         spec: ClassModelSpec) -> _Receiver | None:
        """What `f()` or `X::f()` gives: a temporary when it names a class,
        else the return type of that method (a bare `f` in the calling class)."""
        ref = spec.type_ref(text) if spec.type_ref is not None else None
        if ref is None:
            return None
        sep = spec.separator
        classes = self.qualify(ref.name, at, sep)
        if classes:
            return _Receiver(tuple(classes), 0, ref.args, at)
        owner, _, name = ref.name.rpartition(sep)
        owners = (self.qualify(owner, at, sep) or [owner]) if owner else self._enclosing(cls, sep)
        for c in owners:
            hits = [h for h in self.find([c], name, sep) if h[1].kind != FIELD]
            if hits:
                return self._member_types(hits, spec, None)
        return None

    def _head(self, head: dict, at: Context, cls: list[str],
              spec: ClassModelSpec) -> tuple[_Receiver | None, str]:
        via, typ, name = head.get("via"), head.get("type"), head.get("name") or ""
        sep = spec.separator
        if via == "this":
            return (_Receiver(tuple(cls), 1, (), None), CALLING_CLASS) if cls else (None, NO_CLASS)
        if typ:
            recv = self._type_of(typ, at, spec) or self._function_result(typ, at, cls, spec)
            if recv is not None:
                return recv, TYPED
            ref = spec.type_ref(typ) if via == "type" and spec.type_ref is not None else None
            if ref is not None:
                # A namespace: its functions defined as `T ns::f()` are rows of `ns`.
                return _Receiver((ref.name,), 0, (), None), TYPED
            return None, NO_CLASS
        if via != "unknown":
            return None, UNTYPED
        if name.endswith(_CALL_STEP):
            recv = self._function_result(name[:-len(_CALL_STEP)], at, cls, spec)
            return (recv, CALLING_CLASS) if recv is not None else (None, UNTYPED)
        if name.isidentifier() and cls:
            for c in self._enclosing(cls, sep):
                hits = [h for h in self.find([c], name, sep) if h[1].kind == FIELD]
                if hits:
                    recv = self._member_types(hits, spec, None)
                    return (recv, CALLING_CLASS) if recv is not None else (None, NO_CLASS)
        return None, UNTYPED

    def resolve(self, name: str, arity: int | None, entry: dict, caller: dict) -> Resolution | None:
        """The targets a call's evidence finds, or None without evidence."""
        self.guessed = False
        found = self._resolve(name, arity, entry, caller)
        return found._replace(guessed=self.guessed) if found is not None else None

    def _resolve(self, name: str, arity: int | None, entry: dict, caller: dict) -> Resolution | None:
        spec: ClassModelSpec | None = _CLASS_MODELS.get(caller.get("language") or "")  # type: ignore[assignment]
        if spec is None or spec.type_ref is None or not has_evidence(entry):
            return None
        sep = spec.separator
        racc, head, cls_text = entry.get("racc"), entry.get("rhead"), entry.get("cls")
        path, line = caller.get("path") or "", caller.get("start_line") or 0
        cls = self._calling_classes(cls_text, path, line, sep) if cls_text else []
        if cls:
            at = self._class_context(cls[0], path, line, sep)
        else:
            scopes = _parents(cls_text, sep) if cls_text else self.file_scopes(path, sep)
            at = self._context(scopes, path, line, cls_text, sep)

        if racc is None:
            for c in self._enclosing(cls, sep):
                hits = self.find([c], name, sep)
                if hits:
                    return _targets(hits, arity, CALLING_CLASS)
            return Resolution(None, BARE_NOT_IN_CLASS)
        if not isinstance(head, dict):
            return None
        if racc == sep:
            ref = spec.type_ref(head.get("type") or "")
            classes = self.qualify(ref.name, at, sep) if ref is not None else []
            if not classes:
                return Resolution(None, NOT_A_CLASS)
            nested = [f"{c}{sep}{name}" for c in classes if f"{c}{sep}{name}" in self.known]
            if nested:
                # `X::Y(...)` constructs the nested class X::Y.
                hits = self.find(nested, name, sep)
                return _targets(hits, arity, STATIC) if hits else Resolution(None, NOT_A_CLASS)
            hits = self.find(classes, name, sep)
            if not hits:
                return _unresolved(NOT_IN_CLASS if self.complete(classes, sep) else NO_CLASS)
            return _targets(hits, arity, STATIC)

        recv, how = self._head(head, at, cls, spec)
        if recv is None:
            return _unresolved(how)
        for step in entry.get("rpath") or ():
            recv = self._step(recv, step, spec)
            if recv is None:
                return _unresolved(NO_CLASS)
        if racc == spec.pointer_access and not recv.pointers:
            has_arrow, arrow = self._arrow(recv, spec)
            if has_arrow and arrow is None:
                return _unresolved(NO_CLASS)
            recv = arrow or recv
        hits = self.find(recv.classes, name, sep)
        if not hits:
            return _unresolved(NOT_IN_CLASS if self.complete(recv.classes, sep) else NO_CLASS)
        return _targets(hits, arity, how)
