"""Structural macro discovery and blanking to self-heal tree-sitter parse errors."""

import bisect
import re

from tree_sitter import Node

from chonks.languages import flags as _lang_flags

# Macro self-heal: tree-sitter-cpp can't parse engine macros (UCLASS/
# Q_OBJECT/...), dropping the class name. Discovery is purely structural
# (no hardcoded names), so C's own annotation macros benefit too.

_MACRO_LANGS = _lang_flags("c_macro_self_heal")
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


def _is_macro_name(t: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", t)) and t not in _MACRO_KEEP


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
        pat = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in plain) + r")\b[ \t]*(?:\([^()]*\))?")
        for m in pat.finditer(text):
            blank(m.start(), m.end())
    return bytes(out)


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


def _discover_macros(root: Node, src: bytes) -> set[str]:
    """Candidate macro names via 4 structural tells (no hardcoded names;
    over-discovery is harmless since callers validate by error-count
    decrease before ever blanking one)."""
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
            for tok in re.findall(r"[A-Z][A-Z0-9_]{2,}(?=\s*\()", span):
                if _is_macro_name(tok):
                    found.add(tok)
        stack.extend(node.children)
    for m in re.finditer(r"\b(?:class|struct)\s+([A-Z][A-Z0-9_]{2,})\s+[A-Za-z_]\w*", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    for m in re.finditer(r"\b([A-Z][A-Z0-9_]{2,})\b\s*(?:\([^()]*\))?\s*(?:class|struct)\b", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    # Standalone macro line (GENERATED_BODY(); Q_OBJECT) parses as a bogus
    # MISSING ';' declaration, not an ERROR node, so (a)/(d) above miss it.
    # `\r?$`: without it this never fires on CRLF (Windows) files.
    for m in re.finditer(r"(?m)^[ \t]*([A-Z][A-Z0-9_]{2,})[ \t]*(?:\([^()]*\))?[ \t]*;?[ \t]*\r?$", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    return found
