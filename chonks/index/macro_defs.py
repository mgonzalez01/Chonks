"""What a project's own source says about a name: a #define (and what it
stands for) or a type, read as text, with no preprocessor. For a name the
project defines, the table decides how it is rewritten before parsing:
hidden (it stands for nothing or attributes), unwrapped (attributes around
its one argument), replaced (it stands for a type) or never touched (a type,
or a macro that writes a declaration). Self-heal still guesses the macros
the project does not define, such as an SDK's or engine's."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from chonks.index.macro_heal import _MACRO_NAME

# How a name must be treated when self-heal would blank it.
VETO = "veto"            # a type, or a macro that writes a declaration: never blank
TYPE = "type"            # a macro that stands for a type: replace, don't blank
CODE = "code"            # expands to statements or expressions: blank only where it helps
WRAPPER = "wrapper"      # LOCAL(type) -> static type: drop the name and parens, keep the argument
UNKNOWN = "unknown"      # not decidable from the project: today's heuristics
ATTRIBUTE = "attribute"  # empty or attributes only: blanking is exact

# Most cautious first; a name with several bodies (#ifdef branches) takes
# the most cautious class among them.
_CAUTION = {VETO: 0, TYPE: 1, CODE: 2, UNKNOWN: 3, ATTRIBUTE: 4}

_DEFINE = re.compile(
    rb"^(?:\xef\xbb\xbf)?[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(\([^)]*\))?[ \t]*((?:[^\n]*\\\r?\n)*[^\n]*)",
    re.M)
_TYPE_DEF = re.compile(
    rb"\b(?:class|struct|union|enum(?:[ \t]+class)?)[ \t]+(?:\[\[[^\]]*\]\][ \t]*)?"
    rb"([A-Za-z_]\w*)[ \t]*(?:final[ \t]*)?[:{]")
_TYPEDEF = re.compile(rb"\btypedef\b[^;{}]*?\b([A-Za-z_]\w*)[ \t]*;")
_USING = re.compile(rb"\busing[ \t]+([A-Za-z_]\w*)[ \t]*=")
_COMMENT = re.compile(rb"/\*.*?\*/|//[^\n]*", re.S)
_ATTRIBUTE_CALL = re.compile(
    r"\b(?:__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\)|__declspec\s*\([^()]*\)"
    r"|_Pragma\s*\([^()]*\)|__pragma\s*\([^()]*\)|alignas\s*\([^()]*\))|\[\[[^\]]*\]\]")
_TOKEN = re.compile(r"[A-Za-z_]\w*|\S")

_ATTRIBUTE_WORDS = frozenset({
    "inline", "static", "extern", "__inline", "__inline__", "__forceinline",
    "__cdecl", "__stdcall", "__fastcall", "__thiscall", "__vectorcall",
    "virtual", "constexpr", "explicit", "noexcept", "__restrict", "restrict",
})
_TYPE_WORDS = frozenset({
    "void", "bool", "_Bool", "char", "short", "int", "long", "float", "double",
    "signed", "unsigned", "const", "volatile", "wchar_t", "char8_t", "char16_t",
    "char32_t", "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t",
    "uint64_t", "struct", "union", "enum",
})
_QUALIFIERS = frozenset({"const", "volatile", "signed", "unsigned", "struct", "union", "enum"})
_MAX_CHAIN = 4
# Bump when read_definitions extracts something different: the index
# caches its records per file under this version.
SCAN_VERSION = "2"
# Only macro-shaped names are hidden, the shape self-heal guesses from: a
# lowercase `#define local static` may be an ordinary identifier elsewhere.
_MACRO_SHAPE = re.compile(_MACRO_NAME)


@dataclass(frozen=True)
class DefinitionTable:
    classes: dict[str, str] = field(default_factory=dict)
    # For TYPE names whose type fits in the name's own length.
    substitutions: dict[str, str] = field(default_factory=dict)
    # Names self-heal must never blank: types, macros that write a
    # declaration, and TYPE macros with no substitution that fits.
    vetoes: frozenset[str] = frozenset()
    # Function-like macros that only add attributes around their one
    # argument; unwrapped in every file before parsing.
    wrappers: frozenset[str] = frozenset()
    # Object-like macros that stand for nothing or attributes only, in
    # every #ifdef branch; hidden in every file before parsing.
    attributes: frozenset[str] = frozenset()
    # Names with a function-like #define: an attribute passed to one of
    # them is its argument, left alone.
    function_like: frozenset[str] = frozenset()

    def vetoed(self, names) -> set[str]:
        return set(names) & self.vetoes


EMPTY = DefinitionTable()


def read_definitions(src: bytes) -> dict:
    """One file's #defines and type names: {"d": [[name, params or None,
    body]], "t": [type names]}. This is what the index caches per file (see
    SCAN_VERSION), so it stays plain JSON."""
    defines = []
    if b"define" in src:
        for m in _DEFINE.finditer(src):
            body = re.sub(rb"\\\r?\n", b" ", m.group(3))
            body = _COMMENT.sub(b" ", body).strip().decode("latin-1")
            params = ([p.strip() for p in m.group(2).decode("latin-1")[1:-1].split(",") if p.strip()]
                      if m.group(2) else None)
            defines.append([m.group(1).decode("latin-1"), params, body])
    types = set()
    for rx, word in ((_TYPE_DEF, b""), (_TYPEDEF, b"typedef"), (_USING, b"using")):
        if word in src:
            types.update(m.group(1).decode("latin-1") for m in rx.finditer(src))
    return {"d": defines, "t": sorted(types)}


def build_table(records) -> DefinitionTable:
    """The table for a set of files, from their read_definitions records."""
    bodies: dict[str, list[tuple[list[str] | None, str]]] = {}
    types: set[str] = set()
    for rec in records:
        for name, params, body in rec["d"]:
            bodies.setdefault(name, []).append((params, body))
        types.update(rec["t"])
    return _build(bodies, types)


def scan_definitions(paths) -> DefinitionTable:
    """Reads each file once, with no cache."""
    records = []
    for path in paths:
        try:
            records.append(read_definitions(Path(path).read_bytes()))
        except OSError:
            continue
    return build_table(records)


def _build(bodies: dict, types: set[str]) -> DefinitionTable:
    memo: dict[str, str] = {}

    def classify(name: str, depth: int, seen: frozenset) -> str:
        if name in types:
            return VETO
        if name in memo:
            return memo[name]
        if name not in bodies:
            return UNKNOWN
        if depth > _MAX_CHAIN or name in seen:
            return UNKNOWN
        cls = min((_classify_body(params, body, lambda n: classify(n, depth + 1, seen | {name}), types)
                   for params, body in bodies[name]), key=_CAUTION.__getitem__)
        memo[name] = cls
        return cls

    classes = {n: classify(n, 0, frozenset()) for n in set(bodies) | types}
    wrappers = frozenset(n for n, cls in classes.items()
                         if cls == UNKNOWN and _is_wrapper(n, bodies, classes, 0))
    for n in wrappers:
        classes[n] = WRAPPER
    substitutions = {}
    for name, cls in classes.items():
        if cls != TYPE:
            continue
        texts = sorted({_type_text(body, types, classes) for params, body in bodies.get(name, ())
                        if params is None}, key=len)
        if texts and texts[0] and len(texts[0]) <= len(name):
            substitutions[name] = texts[0]
    # A wrapper is vetoed too: blanking it with its argument deletes the type.
    vetoes = frozenset(n for n, cls in classes.items()
                       if cls in (VETO, WRAPPER) or (cls == TYPE and n not in substitutions))
    attributes = frozenset(n for n, cls in classes.items() if cls == ATTRIBUTE and _MACRO_SHAPE.fullmatch(n))
    function_like = frozenset(n for n, bs in bodies.items() if any(params is not None for params, _ in bs))
    return DefinitionTable(classes, substitutions, vetoes, wrappers, attributes, function_like)


def _is_wrapper(name: str, bodies: dict, classes: dict[str, str], depth: int) -> bool:
    """Every body takes one parameter, uses it exactly once, and otherwise
    holds only attributes, `extern "C"` or another wrapper around it:
    `#define LOCAL(type) static type`, `#define OF(args) args`."""
    if depth > _MAX_CHAIN or name not in bodies:
        return False
    for params, body in bodies[name]:
        if params is None or len(params) != 1:
            return False
        p = params[0]
        tokens = _TOKEN.findall(_ATTRIBUTE_CALL.sub(" ", body).replace('extern "C"', "extern"))
        if tokens.count(p) != 1:
            return False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == p or tok in _ATTRIBUTE_WORDS or classes.get(tok) == ATTRIBUTE:
                i += 1
            elif tokens[i + 1:i + 4] == ["(", p, ")"] and _is_wrapper(tok, bodies, classes, depth + 1):
                i += 4
            else:
                return False
    return True


def _classify_body(params, body: str, classify_name, types: set[str]) -> str:
    text = _ATTRIBUTE_CALL.sub(" ", body)
    tokens = _TOKEN.findall(text)
    if params is not None:
        # A function-like macro that places two of its parameters side by
        # side (`type name`) writes a declaration: blanking it deletes the
        # declared name. Others keep today's handling.
        ps = set(params)
        if any(a in ps and b in ps for a, b in zip(tokens, tokens[1:])):
            return VETO
        return UNKNOWN
    if not tokens:
        return ATTRIBUTE
    kinds = set()
    for tok in tokens:
        if tok in _ATTRIBUTE_WORDS:
            kinds.add(ATTRIBUTE)
        elif tok in _TYPE_WORDS or tok in types or tok in ("*", "&"):
            kinds.add(TYPE)
        elif tok[0].isalpha() or tok[0] == "_":
            kinds.add(classify_name(tok))
        else:
            kinds.add(CODE)  # operators, literals, braces
    if VETO in kinds:
        return VETO
    if CODE in kinds:
        return CODE
    if TYPE in kinds:
        return TYPE if UNKNOWN not in kinds else UNKNOWN
    if UNKNOWN in kinds:
        return UNKNOWN
    return ATTRIBUTE


def _type_text(body: str, types: set[str], classes: dict[str, str]) -> str:
    """The type a TYPE macro stands for, with attribute words dropped, so
    it parses in place of the name."""
    keep = [t for t in _TOKEN.findall(_ATTRIBUTE_CALL.sub(" ", body))
            if t in _TYPE_WORDS or t in types or t in ("*", "&") or classes.get(t) == TYPE]
    if not keep or all(t in _QUALIFIERS for t in keep):
        return ""
    return " ".join(keep).replace(" *", "*").replace(" &", "&")
