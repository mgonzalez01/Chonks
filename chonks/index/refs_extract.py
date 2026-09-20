"""Typed reference (calls/imports/inherits) and literal extraction from AST nodes."""

import re

from tree_sitter import Node

from chonks.core.refresh import register_refresh
from chonks.languages import get_or_none as _lang_spec, table as _lang_table
from chonks.languages._ast import terminal_identifier as _terminal_identifier
from chonks.languages.spec import (
    FieldChildren as _FieldChildren,
    Field as _Field,
    Children as _Children,
    NodeRule as _NodeRule,
    LiteralSpec as _LiteralSpec,
    NOT_HANDLED as _NOT_HANDLED,
)

# Typed reference extraction (calls/imports/inherits): implemented for cpp,
# c_sharp, gdscript, python, c; other languages fall back to untyped
# 'mentions' edges in build_refs.

# Cap per-list name count so a pathological chunk can't blow up the metadata JSON row size.
_REFS_MAX_NAMES = 200

# Bounds one name's own receiver/arity fan-out so it can't consume many
# of the 200 name slots by itself (dedup is by full fingerprint, not name).
_REFS_MAX_CALL_VARIANTS_PER_NAME = 8


# Call-site fingerprint: receiver is the IMMEDIATE token adjacent to the
# called name, not the chain's root ('a.b.c()' -> 'b'), matching repomap's
# definer-qualifier split.

# Left-recursive grammars only: chain nests on the base/left side, so the
# base field's rightmost leaf is the immediate receiver. cpp's
# qualified_identifier is right-recursive and handled by the cpp spec's
# call_receiver hook.
_CHAIN_BASE_FIELD: dict[str, str] = {
    "field_expression": "argument",
    "attribute": "object",
    "member_access_expression": "expression",
    "qualified_name": "qualifier",
}


def _call_receiver(callee: Node, src: bytes) -> str | None:
    """The immediate receiver of a call's callee expression (see the chain
    rule in this section's header comment). None for a bare/free callee."""
    base_field = _CHAIN_BASE_FIELD.get(callee.type)
    if base_field is None:
        return None
    return _terminal_identifier(callee.child_by_field_name(base_field), src)


def _receiver(call_node: Node, callee: Node, src: bytes, lang: str) -> str | None:
    spec = _lang_spec(lang)
    if spec is not None and spec.call_receiver is not None:
        result = spec.call_receiver(call_node, callee, src)
        if result is not _NOT_HANDLED:
            return result
    return _call_receiver(callee, src)


# A spread/pack/varargs marker makes the positional count meaningless.
_VARIADIC_CALL_ARG_TYPES = {"list_splat", "dictionary_splat", "parameter_pack_expansion"}
# Keyword args are real but not positional; don't count or force wildcard.
_KEYWORD_CALL_ARG_TYPES = {"keyword_argument"}
# A named `comment` node between arguments would otherwise inflate arity
# by 1 per comment; must be skipped, not counted.
_TRIVIA_CALL_ARG_TYPES = {"comment"}


def _call_arity(args_node: "Node | None") -> int | None:
    """Positional argument count, or None (wildcard) when a variadic marker
    or ANY keyword argument is present, since a keyword arg can satisfy a
    required param the positional count wouldn't reflect."""
    if args_node is None:
        return 0
    count = 0
    saw_keyword = False
    for c in args_node.children:
        if not c.is_named:
            continue
        if c.type in _TRIVIA_CALL_ARG_TYPES:
            continue
        if c.type in _VARIADIC_CALL_ARG_TYPES:
            return None
        if c.type in _KEYWORD_CALL_ARG_TYPES:
            saw_keyword = True
            continue
        count += 1
    return None if saw_keyword else count


# Declarative per-language specs, a table of NodeRules walked generically
# by _apply_rule below. Adding a language means writing a spec, not a new
# branch of the walker.
LANG_REFS_SPECS = _lang_table("refs_spec")

_REFS_LANGS = set(LANG_REFS_SPECS)


def _call_entry_name(entry: "str | dict") -> str | None:
    """Bare called name from a calls-list entry: a fingerprint dict or a
    legacy bare string."""
    return entry["name"] if isinstance(entry, dict) else entry


def _add_call_entry(lst: list, entry: "str | dict") -> None:
    """Dedups on (name, receiver, arity) but caps on distinct NAMES. Shared
    by extraction and _merge_refs; both must use this or a merge undoes the cap."""
    if entry in lst:
        return
    name = _call_entry_name(entry)
    if not name:
        return
    same_name = 0
    distinct_names: set[str] = set()
    for e in lst:
        n = _call_entry_name(e)
        distinct_names.add(n)
        if n == name:
            same_name += 1
    if same_name >= _REFS_MAX_CALL_VARIANTS_PER_NAME:
        return
    if name not in distinct_names and len(distinct_names) >= _REFS_MAX_NAMES:
        return
    lst.append(entry)


def _apply_rule(n: Node, rule: "_NodeRule", refs: dict[str, list[str]], src: bytes, lang: str) -> None:
    def add(bucket: str, name: str | None) -> None:
        lst = refs[bucket]
        if name and name not in lst and len(lst) < _REFS_MAX_NAMES:
            lst.append(name)

    def add_call(name: str | None, receiver: str | None, arity: int | None) -> None:
        # New entries are fingerprint dicts, not bare strings (legacy shape).
        if not name:
            return
        _add_call_entry(refs["calls"], {"name": name, "receiver": receiver, "arity": arity})

    if isinstance(rule, _Field):
        target = n.child_by_field_name(rule.field)
        if target is not None:
            if rule.bucket == "calls":
                add_call(
                    rule.name_fn(target, src),
                    _receiver(n, target, src, lang),
                    _call_arity(n.child_by_field_name("arguments")),
                )
            else:
                add(rule.bucket, rule.name_fn(target, src))
    elif isinstance(rule, _FieldChildren):
        target = n.child_by_field_name(rule.field)
        if target is not None:
            for c in target.children:
                if c.type in rule.types:
                    add(rule.bucket, rule.name_fn(c, src))
    elif isinstance(rule, _Children):
        for c in n.children:
            matched_break_outer = False
            for spec in rule.specs:
                if c.type not in spec.types:
                    continue
                if spec.nested is not None:
                    for cc in c.children:
                        if cc.type in spec.nested.types:
                            add(spec.bucket, spec.nested.name_fn(cc, src))
                            if spec.nested.break_inner:
                                break
                elif spec.bucket == "calls":
                    # gdscript's chain is flat, so `c` itself carries no receiver;
                    # the gdscript call_receiver hook scans the siblings instead.
                    receiver = _receiver(n, c, src, lang)
                    add_call(
                        spec.name_fn(c, src),
                        receiver,
                        _call_arity(n.child_by_field_name("arguments")),
                    )
                else:
                    add(spec.bucket, spec.name_fn(c, src))
                matched_break_outer = spec.break_outer
                break  # elif semantics: first matching spec per child wins
            if matched_break_outer:
                break


# Literal extraction: string literals per chunk for find_by_message, via a
# declarative per-language _LiteralSpec walked by _collect_literals below.
_LITERALS_MIN_LEN = 6  # trivial-skip: stripped text shorter than this is noise
_HAS_ALPHA_RE = re.compile(r"[A-Za-z]")

# One prefix pattern covers every language's string-prefix convention
# (python f"/r", cpp L"/u8", c_sharp $"/@") without a per-language table.
_LITERAL_PREFIX_RE = re.compile(r"^[A-Za-z&]{0,3}")
_LITERAL_QUOTES = ('"""', "'''", '"', "'", "`")

# Unrecognized escapes (\d, \uXXXX, ...) are left as written.
_ESCAPE_MAP = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'", "\\": "\\", "0": "\0"}
_ESCAPE_RE = re.compile(r"\\(?:x[0-9A-Fa-f]{2}|.)", re.DOTALL)


def _decode_escapes(text: str) -> str:
    """Decodes known escapes only; best-effort text-matching aid, not a
    real string-literal parser, so an unrecognized escape is left as-is."""
    def repl(m: re.Match) -> str:
        s = m.group(0)
        if s[1] in "xX" and len(s) == 4:
            return chr(int(s[2:], 16))
        return _ESCAPE_MAP.get(s[1], s)
    return _ESCAPE_RE.sub(repl, text)


def _strip_literal_wrapper(raw: str, lang: str) -> tuple[str, str]:
    """Returns (text, mode): 'normal' decodes escapes, 'raw' never does
    (python r"...", lua [[...]], or an unrecognized wrapper), 'verbatim'
    (C# @"...") only decodes "" -> "."""
    spec = _lang_spec(lang)
    if spec is not None and spec.literal_wrapper is not None:
        result = spec.literal_wrapper(raw)
        if result is not _NOT_HANDLED:
            return result

    m = _LITERAL_PREFIX_RE.match(raw)
    idx = m.end() if m else 0
    prefix = raw[:idx]
    verbatim = False
    while idx < len(raw) and raw[idx] in "$@":
        verbatim = verbatim or raw[idx] == "@"
        idx += 1
    rest = raw[idx:]
    for q in _LITERAL_QUOTES:
        if rest.startswith(q) and rest.endswith(q) and len(rest) >= 2 * len(q):
            text = rest[len(q):-len(q)]
            if spec is not None and spec.raw_string_prefix is not None and spec.raw_string_prefix(prefix):
                return text, "raw"
            return text, ("verbatim" if verbatim else "normal")
    return raw, "raw"


def _leaf_literal_text(n: Node, src: bytes, lang: str) -> str:
    """Leaf text, NOT trailing-stripped: a concat piece's own trailing
    space matters when joined to the next piece. Strip once, in
    _collect_literals, on the fully assembled literal instead."""
    if n.type == "raw_string_literal":
        content = next((c for c in n.children if c.type == "raw_string_content"), None)
        return src[content.start_byte:content.end_byte].decode(errors="replace") if content else ""
    raw = src[n.start_byte:n.end_byte].decode(errors="replace")
    text, mode = _strip_literal_wrapper(raw, lang)
    if mode == "normal":
        text = _decode_escapes(text)
    elif mode == "verbatim":
        text = text.replace('""', '"')
    return text


_LITERAL_SPECS = _lang_table("literals")


def _refresh_from_registry() -> None:
    global LANG_REFS_SPECS, _REFS_LANGS, _LITERAL_SPECS
    LANG_REFS_SPECS = _lang_table("refs_spec")
    _REFS_LANGS = set(LANG_REFS_SPECS)
    _LITERAL_SPECS = _lang_table("literals")


register_refresh(_refresh_from_registry)


def _plus_chain_pieces(n: Node, spec: _LiteralSpec, src: bytes, lang: str) -> list[str] | None:
    """Joined text pieces if the WHOLE chain is pure string literals, else
    None so an impure chain's literal pieces are picked up individually."""
    if not any(not c.is_named and c.type == spec.plus_op for c in n.children):
        return None
    named = [c for c in n.children if c.is_named]
    if len(named) != 2:
        return None
    pieces: list[str] = []
    for operand in named:
        if operand.type in spec.leaf_types:
            pieces.append(_leaf_literal_text(operand, src, lang))
        elif operand.type == spec.plus_type:
            sub = _plus_chain_pieces(operand, spec, src, lang)
            if sub is None:
                return None
            pieces.extend(sub)
        else:
            return None
    return pieces


def _is_trivial_literal(text: str) -> bool:
    return len(text) < _LITERALS_MIN_LEN or not _HAS_ALPHA_RE.search(text)


# Per-node memo so a node walked twice in one run doesn't double-count the
# cap/drop tally. Cleared alongside _literal_cap_state.
_literal_extract_cache: dict[int, list[tuple[str, int]]] = {}


def _collect_literals(node: Node, lang: str, src: bytes) -> list[tuple[str, int]]:
    """(text, line) per string literal, decoded so find_by_message matches
    runtime text, not escapes. Only a trailing DECODED newline is stripped."""
    spec = _LITERAL_SPECS.get(lang)
    if spec is None:
        return []
    cached = _literal_extract_cache.get(id(node))
    if cached is not None:
        return cached
    out: list[tuple[str, int]] = []

    def emit(text: str, line: int) -> None:
        text = text.rstrip("\r\n")
        if not _is_trivial_literal(text):
            out.append((text, line))

    def walk(n: Node) -> None:
        # is_named guard needed: TS's `predefined_type` reuses "string" as
        # an unnamed type-keyword token, not a string value.
        if spec.concat_types and n.type in spec.concat_types:
            pieces = [_leaf_literal_text(c, src, lang) for c in n.children
                      if c.is_named and c.type in spec.leaf_types]
            if pieces:
                emit("".join(pieces), n.start_point[0] + 1)
                return  # consumed, don't re-walk the joined pieces individually
        if spec.plus_type and n.type == spec.plus_type:
            pieces = _plus_chain_pieces(n, spec, src, lang)
            if pieces is not None:
                emit("".join(pieces), n.start_point[0] + 1)
                return
            # impure chain: fall through so pure leaf operands still collect
        if n.is_named and n.type in spec.leaf_types:
            if not (spec.skip_fn and spec.skip_fn(n)):
                emit(_leaf_literal_text(n, src, lang), n.start_point[0] + 1)
            return  # leaf's children are quote/content tokens, not literals
        for child in n.children:
            walk(child)

    walk(node)
    # Capped/counted once per node (memo above) so a node walked twice
    # doesn't double-count.
    if len(out) > _REFS_MAX_NAMES:
        _literal_cap_state["capped_chunks"] += 1
        _literal_cap_state["dropped"] += len(out) - _REFS_MAX_NAMES
        out = out[:_REFS_MAX_NAMES]
    _literal_extract_cache[id(node)] = out
    return out


# Module-level, but safe: segment_file/segment_text_file/segment_markdown
# run on one sequential parser thread.
_literal_cap_state = {"capped_chunks": 0, "dropped": 0}


def _extract_refs(node: Node, lang: str, src: bytes) -> dict[str, list[str]]:
    """{"calls", "imports", "inherits", "literals"} lists. "calls" entries
    are fingerprint dicts; a legacy bare string is treated as receiver=None,
    arity=None so old stored data still resolves."""
    refs: dict[str, list[str]] = {"calls": [], "imports": [], "inherits": [], "literals": []}

    spec = LANG_REFS_SPECS.get(lang)
    if spec is not None:
        def walk(n: Node) -> None:
            rule = spec.get(n.type)
            if rule is not None:
                _apply_rule(n, rule, refs, src, lang)
            for child in n.children:
                walk(child)

        walk(node)

    # _collect_literals already caps/tallies; this loop only adds dedup.
    lst = refs["literals"]
    for text, line in _collect_literals(node, lang, src):
        encoded = f"{line}\x1f{text}"
        if encoded not in lst:
            lst.append(encoded)

    return refs


_EMPTY_REFS: dict[str, list[str]] = {"calls": [], "imports": [], "inherits": [], "literals": []}
_REFS_KEYS = ("calls", "imports", "inherits", "literals")


def _merge_refs(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict[str, list[str]]:
    """Union two refs dicts, deduped, capped at _REFS_MAX_NAMES per list.
    "calls" must go through `_add_call_entry`, not the generic loop below,
    or a merge silently re-truncates by raw entry count and undoes the cap."""
    out: dict[str, list[str]] = {}
    for key in _REFS_KEYS:
        merged = list(a.get(key) or [])
        if key == "calls":
            for entry in b.get(key) or []:
                _add_call_entry(merged, entry)
        else:
            for name in b.get(key) or []:
                if name not in merged and len(merged) < _REFS_MAX_NAMES:
                    merged.append(name)
        out[key] = merged
    return out


def _decode_literals(encoded: list[str]) -> list[tuple[str, int]]:
    """Reverse the "<line>\\x1f<text>" packing _extract_refs uses internally
    (see its docstring) back into the public (text, line) record shape."""
    out = []
    for s in encoded:
        line_str, _, text = s.partition("\x1f")
        out.append((text, int(line_str)))
    return out
