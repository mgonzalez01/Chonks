"""Structural macro discovery and blanking to self-heal tree-sitter parse errors."""

import bisect
import re

from tree_sitter import Node

from chonks.core.refresh import register_refresh
from chonks.languages import flags as _lang_flags

# Macro self-heal: tree-sitter-cpp can't parse engine macros (UCLASS/
# Q_OBJECT/...), dropping the class name. Discovery is purely structural
# (no hardcoded names), so C's own annotation macros benefit too.

_MACRO_LANGS = _lang_flags("c_macro_self_heal")


def _refresh_from_registry() -> None:
    global _MACRO_LANGS
    _MACRO_LANGS = _lang_flags("c_macro_self_heal")


register_refresh(_refresh_from_registry)

_MAX_MACRO_CANDIDATES = 48  # cap trial-reparses per pass on pathological files
# A leading macro (UCLASS) only surfaces once the export macro beside the
# class name is blanked, so healing needs multiple passes.
_MAX_MACRO_PASSES = 5
_MACRO_KEEP = {"NULL", "TRUE", "FALSE"}

# Qt's signals:/slots: aren't ALL_CAPS, so treated as a fixed set. Bare
# form blanks NAME+colon; qualified form blanks only NAME, leaving the
# colon public/private's own access_specifier needs to close.
_QT_ACCESS_SPECIFIERS = {"signals", "slots", "Q_SIGNALS", "Q_SLOTS"}


def _qt_specifier_res(specifiers) -> tuple[re.Pattern, re.Pattern]:
    """Qualified/bare Qt specifier regexes scoped to the given names only."""
    alt = "|".join(re.escape(n) for n in specifiers)
    qualified = re.compile(
        r"\b(?:public|private|protected)\b[ \t]*\b(?:" + alt + r")\b")
    bare = re.compile(r"\b(?:" + alt + r")\b[ \t]*:")
    return qualified, bare


# ALL_CAPS, optionally wrapped in underscores (_FORCE_INLINE_, __API__), or
# `_Capital..._`, a name C reserves for the implementation, which is what
# annotation macros use (SAL: _In_, _Out_writes_(n)).
_MACRO_NAME = r"(?:_*[A-Z][A-Z0-9_]{2,}|_[A-Z][A-Za-z0-9_]*_)"


def _is_macro_name(t: str) -> bool:
    return bool(re.fullmatch(_MACRO_NAME, t)) and t not in _MACRO_KEEP


def _defined_type_names(text: str) -> set[str]:
    """Names this file defines as types. An ALL_CAPS type (AABB, RID) looks
    like a macro and can pass the error-count check by coincidence."""
    names = set(re.findall(
        r"\b(?:class|struct|union|enum(?:\s+class)?)\s+(?:\[\[[^\]]*\]\]\s*)?(\w+)\s*(?:final\s*)?[:{]",
        text))
    names |= set(re.findall(r"\btypedef\b[^;{}]*?\b(\w+)\s*;", text))
    names |= set(re.findall(r"\busing\s+(\w+)\s*=", text))
    return names


def _blank_macros(src: bytes, names) -> bytes:
    """Replaces each macro NAME(...) with spaces, preserving byte length
    and newlines so tree-sitter offsets stay valid against the ORIGINAL
    bytes. latin-1 keeps one char = one byte for regex spans."""
    if not names:
        return src
    out = bytearray(src)
    text = src.decode("latin-1")

    def blank(start: int, end: int) -> None:
        for i in range(start, end):
            if out[i] not in (0x0A, 0x0D):  # keep newlines so line numbers don't shift
                out[i] = 0x20

    specifiers = {n for n in names if n in _QT_ACCESS_SPECIFIERS}
    if specifiers:
        qualified_re, bare_re = _qt_specifier_res(specifiers)
        qualified_spans = []
        for m in qualified_re.finditer(text):
            # blank only the trailing NAME token, not the public/private/protected
            # keyword or the colon that belongs to it
            name_start = m.end()
            while name_start > m.start() and not text[name_start - 1].isspace():
                name_start -= 1
            blank(name_start, m.end())
            qualified_spans.append((m.start(), m.end()))
        for m in bare_re.finditer(text):
            if any(s <= m.start() < e for s, e in qualified_spans):
                continue  # already handled by the qualified form above
            blank(m.start(), m.end())

    plain = [n for n in names if n not in _QT_ACCESS_SPECIFIERS]
    if plain:
        pat = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in plain) + r")\b[ \t]*")
        for m in pat.finditer(text):
            blank(m.start(), _args_end(text, m.end()))
    return bytes(out)


def _args_end(text: str, i: int) -> int:
    """End of a balanced (...) starting at text[i], or i when there is none,
    so API_AVAILABLE(macos(11.0), ios(14.0)) blanks whole."""
    if i >= len(text) or text[i] != "(":
        return i
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        elif text[j] in ";{}":
            break
    return i


def _count_errors(root: Node) -> int:
    n, stack = 0, [root]
    while stack:
        node = stack.pop()
        if node.is_error or node.is_missing:
            n += 1
        stack.extend(node.children)
    return n


def _errors_with_ranges(root: Node) -> tuple[int, list[tuple[int, int]]]:
    """_count_errors plus byte ranges, one walk instead of two since both
    are needed per self-heal pass. Ranges are RAW; margin applied later."""
    n = 0
    ranges: list[tuple[int, int]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_error or node.is_missing:
            n += 1
            ranges.append((node.start_byte, node.end_byte))
        stack.extend(node.children)
    return n, ranges


def _count_errors_upto(root: Node, threshold: int) -> int:
    """Bails once count reaches `threshold` (admission only needs "< base").
    Uses a TreeCursor, not a children-stack, since the cursor visits nodes
    in source order so a real error is reached fast enough for the bail to help."""
    n = 0
    cursor = root.walk()
    reached_end = False
    while not reached_end:
        node = cursor.node
        if node.is_error or node.is_missing:
            n += 1
            if n >= threshold:
                return n
        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue
        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_end = True
            elif cursor.goto_next_sibling():
                retracing = False
    return n


_ERROR_PROXIMITY_MARGIN = 200  # bytes; see _near_error_ranges


def _near_error_ranges(cands: set, parse_src: bytes,
                       error_ranges: list[tuple[int, int]],
                       margin: int = _ERROR_PROXIMITY_MARGIN) -> set:
    """Drops candidates with no occurrence within `margin` bytes of an
    error, since those can't reduce the error count and aren't worth a
    trial reparse. Most _discover_macros tells have no proximity scoping."""
    if not error_ranges or not cands:
        return set()
    merged: list[tuple[int, int]] = []
    for s, e in sorted(error_ranges):
        s, e = s - margin, e + margin
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    starts = [s for s, _ in merged]

    def overlaps(a: int, b: int) -> bool:
        i = bisect.bisect_right(starts, b) - 1
        return i >= 0 and merged[i][1] >= a

    kept: set = set()
    for name in cands:
        pat = re.compile(rb"\b" + re.escape(name.encode("latin-1")) + rb"\b")
        for m in pat.finditer(parse_src):
            if overlaps(m.start(), m.end()):
                kept.add(name)
                break
    return kept


def _class_head_macros(text: str) -> set[str]:
    """A macro in a class definition head, `class _WARN_UNUSED_ HashSet {` or
    `class API_AVAILABLE(...) X :`. `struct TYPE var;` (a use) does not match."""
    return {m.group(1) for m in re.finditer(
        r"\b(?:class|struct)\s+(" + _MACRO_NAME + r")\s*(?:\([^;{}]*?\)\s*)?\s[A-Za-z_]\w*\s*(?:final\s*)?[:{]",
        text) if _is_macro_name(m.group(1))} - _defined_type_names(text)


def _clean_type_uses(root: Node, src: bytes) -> dict[str, int]:
    """How often each name is the type of a declaration that parsed without
    error (`RID render_target;`): a real type, not a macro."""
    uses: dict[str, int] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node.has_error:
            stack.extend(node.children)
            continue
        if node.type == "type_identifier" and node.parent is not None \
                and node.parent.child_by_field_name("type") == node:
            name = src[node.start_byte:node.end_byte].decode("latin-1")
            uses[name] = uses.get(name, 0) + 1
        stack.extend(node.children)
    return uses


def _drop_redundant(parser, parse_src: bytes, root: Node, admitted: set, counts: dict) -> set:
    """In `_FORCE_INLINE_ RID get()` blanking either name fixes the line, so
    both pass the error-count test on their own. Keep only the names the
    set needs: weakest first, and on a tie the one that already parses as
    a type elsewhere goes first, so a real type never rides along."""
    if len(admitted) < 2:
        return admitted
    type_uses = _clean_type_uses(root, parse_src)
    keep = set(admitted)
    full = _count_errors(parser.parse(_blank_macros(parse_src, keep)).root_node)
    for name in sorted(admitted, key=lambda m: (-counts[m], -type_uses.get(m, 0), m)):
        without = keep - {name}
        if without and _count_errors(parser.parse(_blank_macros(parse_src, without)).root_node) <= full:
            keep = without
    return keep


def _prune_healed(parser, base_src: bytes, healed: set) -> set:
    """Heal passes admit in rounds, so a name admitted early can be made
    unnecessary by one found later: in libjpeg, JDIMENSION (a real typedef)
    fixed a few errors before LOCAL(void), the actual macro, was found.
    Drops every name the final set does not need, likely types first."""
    if len(healed) < 2:
        return healed
    type_uses = _clean_type_uses(parser.parse(base_src).root_node, base_src)
    keep = set(healed)
    full = _count_errors(parser.parse(_blank_macros(base_src, keep)).root_node)
    for name in sorted(healed, key=lambda m: (-type_uses.get(m, 0), m)):
        without = keep - {name}
        if without and _count_errors(parser.parse(_blank_macros(base_src, without)).root_node) <= full:
            keep = without
    return keep


def _count_class_bodies(root: Node) -> int:
    n, stack = 0, [root]
    while stack:
        node = stack.pop()
        if node.type in ("class_specifier", "struct_specifier") and node.child_by_field_name("body"):
            n += 1
        stack.extend(node.children)
    return n


def _heal_class_heads(parser, root: Node, parse_src: bytes, healed: set) -> tuple[Node, bytes]:
    """`class _WARN_UNUSED_ HashSet {...}` often parses with no error at all,
    as a function named HashSet, so the error-count test never sees it.
    Admits a class-head macro when blanking it adds a class body and no
    error. Adds each admitted name to `healed`."""
    heads = _class_head_macros(parse_src.decode("latin-1")) - healed
    if not heads:
        return root, parse_src
    errors, bodies = _count_errors(root), _count_class_bodies(root)
    for name in sorted(heads):
        trial_src = _blank_macros(parse_src, {name})
        trial = parser.parse(trial_src).root_node
        trial_errors, trial_bodies = _count_errors(trial), _count_class_bodies(trial)
        if trial_errors <= errors and trial_bodies > bodies:
            healed.add(name)
            root, parse_src, errors, bodies = trial, trial_src, trial_errors, trial_bodies
    return root, parse_src


def _discover_macros(root: Node, src: bytes) -> set[str]:
    """Candidate macro names from structural tells (no hardcoded names).
    Over-discovery costs only trial reparses: callers keep a name only when
    blanking it lowers the error count and the other admitted names do not
    already cover it."""
    text = src.decode("latin-1")
    found: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if (node.type == "call_expression" and node.parent is not None
                and node.parent.type == "expression_statement" and node.children):
            callee = node.children[0]
            if callee.type == "identifier":
                t = src[callee.start_byte:callee.end_byte].decode("latin-1")
                if _is_macro_name(t):
                    found.add(t)
        if node.is_error or node.is_missing:
            span = src[node.start_byte:node.end_byte].decode("latin-1")
            # Require a following '(': a bare ALL_CAPS type (RID) blanked here
            # could coincidentally drop the error count and erase a real name.
            for tok in re.findall(_MACRO_NAME + r"(?=\s*\()", span):
                if _is_macro_name(tok):
                    found.add(tok)
        stack.extend(node.children)
    for m in re.finditer(r"\b(?:class|struct)\s+(" + _MACRO_NAME + r")(?:\s*\(|\s+[A-Za-z_]\w*)", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    for m in re.finditer(r"(?<![\w])(" + _MACRO_NAME + r")\b\s*(?:\([^()]*\))?\s*(?:class|struct)\b", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    # Suffix macro after a signature (`f() const _LIFETIME_BOUND_ {`).
    for m in re.finditer(r"\)[ \t]*(?:(?:const|noexcept|override|final)[ \t]*)*(" + _MACRO_NAME + r")[ \t]*(?:\([^()]*\))?[ \t]*(?=[{;])", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    # Calling-convention macro between the return type and the name
    # (`ULONG STDMETHODCALLTYPE AddRef(`). Without it only the type is a
    # candidate, and blanking the type alone also fixes the line.
    for m in re.finditer(r"\b[A-Za-z_]\w*[ \t*&]+(" + _MACRO_NAME + r")[ \t]+[A-Za-z_]\w*[ \t]*\(", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    # Annotation before a parameter (`Get(_Out_ UINT32 *p)`), `_Capital..._`
    # shape only, so an ALL_CAPS parameter type is never proposed here.
    for m in re.finditer(r"[(,][ \t]*(_[A-Z][A-Za-z0-9_]*_)[ \t]*(?:\([^()]*\))?[ \t]+[A-Za-z_]", text):
        found.add(m.group(1))
    # Prefix macro with no arguments at the start of a declaration
    # (`_FORCE_INLINE_ void f()`), which none of the tells above see.
    for m in re.finditer(r"(?m)^[ \t]*(?:(?:static|inline|virtual|constexpr)[ \t]+)*(" + _MACRO_NAME + r")[ \t]+(?=[A-Za-z_~])", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    # Standalone macro line (GENERATED_BODY(); Q_OBJECT) parses as a bogus
    # MISSING ';' declaration, not an ERROR node, so (a)/(d) above miss it.
    # `\r?$`: without it this never fires on CRLF (Windows) files.
    for m in re.finditer(r"(?m)^[ \t]*(" + _MACRO_NAME + r")[ \t]*(?:\([^()]*\))?[ \t]*;?[ \t]*\r?$", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    return found - _defined_type_names(text)
