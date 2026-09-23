"""AST boundary detection, size-based splitting/merging, and segment_file."""

import logging
import re
import warnings
from pathlib import Path
from typing import Any

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from chonks.core.refresh import register_refresh
from chonks.core.symbols import FORWARD_DECLARATION
from chonks.index.macro_heal import (
    _MACRO_LANGS,
    _MAX_MACRO_CANDIDATES,
    _MAX_MACRO_PASSES,
    _QT_ACCESS_SPECIFIERS,
    _blank_macros,
    _count_errors_upto,
    _discover_macros,
    _errors_with_ranges,
    _near_error_ranges,
)
from chonks.index.refs_extract import (
    _EMPTY_REFS,
    _decode_literals,
    _extract_refs,
    _literal_cap_state,
    _literal_extract_cache,
    _merge_refs,
)
from chonks.languages import (
    EXT_TO_LANG as _EXT_TO_LANG,
    get_or_none as _lang_spec,
    table as _lang_table,
)
from chonks.languages._naming import extract_name as _extract_name_by_rules

logger = logging.getLogger("chunking")

# Bump when a change here would alter chunk boundaries for already-indexed
# content. Provenance only (meta table, doctor.py); nothing reads it back
# to gate behavior.
CHUNKER_VERSION = 4

# Chunk-size constants: CHUNK_TARGET is UTF-8 characters; CHUNK_MIN/MAX are
# UTF-8 bytes. Don't mix the two when comparing.

CHUNK_TARGET = 4500   # characters, line-fallback slice target
CHUNK_MIN    = 300    # bytes, below this merge forward into the next segment
CHUNK_MAX    = 6000   # bytes, hard ceiling; forcible split above this

# AST-boundary chunks are self-contained (cAST) and get no overlap; only
# line-based fallback chunks get FALLBACK_OVERLAP_LINES of prior-chunk tail
# prepended, since a split there can land mid-context.
FALLBACK_OVERLAP_LINES = 5

# tree-sitter 0.25 raises ValueError on timeout; without this a malformed or
# adversarial file can loop the parser indefinitely.
PARSE_TIMEOUT_MICROS = 30_000_000  # 30 seconds

# language -> structural node types that become chunk boundaries.
_BOUNDARY_NODES = _lang_table("boundary_nodes")

# Container nodes: recurse through them to find inner boundaries.
# If a container yields no inner boundaries, fall back to treating it as one chunk.
_CONTAINER_NODES = _lang_table("container_nodes")


def _lang_for_path(path: Path) -> str | None:
    return _EXT_TO_LANG.get(path.suffix.lower())


# Text-fallback chunks get `language` set to the raw extension instead, never
# a member of this set. store._chunk_kind_clause uses that split to
# discriminate "code" vs "docs" chunk_kind.
CODE_LANGUAGES: frozenset[str] = frozenset(_EXT_TO_LANG.values())


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _is_boundary(node: Node, lang: str, src: bytes) -> bool:
    boundaries = _BOUNDARY_NODES.get(lang, set())
    spec = _lang_spec(lang)
    if node.type in boundaries:
        if spec is not None:
            node_filter = spec.boundary_filters.get(node.type)
            if node_filter is not None:
                return node_filter(node, src)
        return True
    if spec is not None and spec.extra_boundary is not None:
        return spec.extra_boundary(node, src)
    return False


def _is_forward_declaration(node: Node, lang: str, src: bytes) -> bool:
    spec = _lang_spec(lang)
    pred = spec.forward_declarations.get(node.type) if spec is not None else None
    return pred is not None and pred(node, src)


# Salvage-eligible: statement-body types only. Member-body types
# (class/struct) are excluded, or an errored class explodes into one
# chunk per method (methods are always nested there).
_SALVAGE_ELIGIBLE_BOUNDARY_TYPES = _lang_table("salvage_nodes")


def _is_salvage_eligible(node: Node, lang: str, src: bytes) -> bool:
    """True for a has_error boundary type safe to recurse into (see module
    comment above). cbuffer/tbuffer qualify too: their body holds only
    simple fields, never nested boundary types."""
    spec = _lang_spec(lang)
    if spec is not None and spec.salvage_extra is not None and spec.salvage_extra(node, src):
        return True
    return node.type in _SALVAGE_ELIGIBLE_BOUNDARY_TYPES.get(lang, set())


def _salvage_errored_boundary(node: Node, lang: str, src: bytes,
                              counters: dict | None) -> list[Node] | None:
    """Pulls clean (error-free) boundaries out of a has_error node's
    children. `node` itself is always kept alongside them, never replaced.
    Returns None (caller keeps `node` whole) when nothing clean is found."""
    inner: list[Node] = []
    for child in node.children:
        if child.is_named:
            inner.extend(_collect_boundaries(child, lang, src, counters))

    clean = [n for n in inner if not n.has_error]
    if not clean:
        return None

    if counters is not None:
        counters["error_salvaged"] = 1
        counters["error_salvaged_nodes"] = counters.get("error_salvaged_nodes", 0) + 1

    return [node] + clean


def _extract_name(node: Node, lang: str, src: bytes) -> str | None:
    """Primary symbol name from a boundary node, or None if not found."""
    spec = _lang_spec(lang)
    if spec is None:
        return None
    return _extract_name_by_rules(spec, node, src)


# ---------------------------------------------------------------------------
# Segmentation (cAST algorithm)
# ---------------------------------------------------------------------------

def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode(errors="replace")


class _Segment:
    """A contiguous region of source code that will become one chunk."""
    __slots__ = ("start_byte", "end_byte", "start_line", "end_line",
                 "chunk_type", "name", "refs")

    def __init__(self, node: Node, src: bytes, lang: str):
        self.start_byte = node.start_byte
        self.end_byte   = node.end_byte
        self.start_line = node.start_point[0] + 1
        self.end_line   = node.end_point[0] + 1
        spec = _lang_spec(lang)
        synthetic = (spec.synthetic_chunk_type(node, src)
                     if spec is not None and spec.synthetic_chunk_type is not None else None)
        # F1 ran the cbuffer check for every language. This gives the same
        # result, because no other language uses "declaration" as a
        # boundary or a container type.
        self.chunk_type = synthetic if synthetic is not None else node.type
        self.name       = _extract_name(node, lang, src)
        self.refs       = _extract_refs(node, lang, src)

    def size(self, src: bytes) -> int:
        return self.end_byte - self.start_byte

    def content(self, src: bytes) -> str:
        return src[self.start_byte:self.end_byte].decode(errors="replace")


def _collect_boundaries(node: Node, lang: str, src: bytes,
                        counters: dict | None = None) -> list[Node]:
    """Structural boundary nodes in source order. `counters` goes ONLY at
    the top-level (segment_file) call; internal re-walks omit it to avoid
    double-counting the error-salvage tally."""
    if _is_boundary(node, lang, src):
        if node.has_error and _is_salvage_eligible(node, lang, src):
            salvaged = _salvage_errored_boundary(node, lang, src, counters)
            if salvaged is not None:
                return salvaged
        return [node]

    containers = _CONTAINER_NODES.get(lang, set())
    is_container = node.type in containers

    inner: list[Node] = []
    for child in node.children:
        if child.is_named:
            inner.extend(_collect_boundaries(child, lang, src, counters))

    if is_container and not inner:
        # Container with no structural children: chunk it as a whole
        return [node]

    return inner


def _split_large_node(node: Node, lang: str, src: bytes) -> list[_Segment]:
    """Recursively splits an oversized node by its children's boundaries,
    falling back to line-slicing if none exist."""
    # node itself is already a boundary, so collect from its children instead
    # (_collect_boundaries(node) would just return [node]).
    children: list[Node] = []
    for child in node.children:
        if child.is_named:
            children.extend(_collect_boundaries(child, lang, src))

    if len(children) > 1:
        segs: list = []
        # Header chunk = parent's signature up to the first child boundary,
        # so the container itself stays retrievable and never overlaps members.
        header_raw = src[node.start_byte:children[0].start_byte].decode(errors="replace")
        # rstrip: a trailing whitespace-only fragment would make h_end land
        # on the child's start_line, so adjacent chunks wouldn't share a line.
        header = header_raw.rstrip()
        if header.strip().strip("{}").strip():
            node_seg = _Segment(node, src, lang)
            h_start = node.start_point[0] + 1
            if len(header.encode("utf-8")) > CHUNK_MAX:
                segs.extend(_line_slice_oversized(
                    header, h_start, node_seg.name, node_seg.refs,
                    chunk_type=node_seg.chunk_type))
            else:
                h_end = h_start + header.count("\n")
                segs.append(_SyntheticSegment(
                    header, h_start, h_end,
                    chunk_type=node_seg.chunk_type, name=node_seg.name, refs=node_seg.refs))
        for c in children:
            child_seg = _Segment(c, src, lang)
            if child_seg.size(src) > CHUNK_MAX:
                segs.extend(_split_large_node(c, lang, src))
            else:
                segs.append(child_seg)
        # merge any still-tiny segments before returning
        return _merge_small(segs, src)

    # One oversized child: recurse into it so its slices keep its OWN name,
    # not the parent's. Byte-range check guards against infinite recursion.
    if len(children) == 1:
        c = children[0]
        if (c.start_byte, c.end_byte) != (node.start_byte, node.end_byte) \
                and _Segment(c, src, lang).size(src) > CHUNK_MAX:
            return _split_large_node(c, lang, src)

    # No sub-boundaries: slice by CHUNK_TARGET chars, but pieces still inherit
    # the boundary's name/type so it stays findable, not an anonymous "block".
    node_seg  = _Segment(node, src, lang)   # source of the name/type to propagate
    node_name = node_seg.name
    node_type = node_seg.chunk_type
    node_refs = node_seg.refs
    text = _node_text(node, src)
    lines = text.splitlines(keepends=True)
    segments: list = []
    buf: list[str] = []
    buf_start_line = node.start_point[0] + 1
    char_count = 0
    prev_tail: list[str] = []  # overlap prefix for the next segment

    for line in lines:
        buf.append(line)
        char_count += len(line)
        if char_count >= CHUNK_TARGET:
            content = "".join(prev_tail + buf)
            segments.append(_SyntheticSegment(
                content, buf_start_line, buf_start_line + len(buf) - 1,
                name=node_name, chunk_type=node_type, refs=node_refs,
            ))
            prev_tail = buf[-FALLBACK_OVERLAP_LINES:] if FALLBACK_OVERLAP_LINES > 0 else []
            buf_start_line += len(buf)
            buf = []
            char_count = 0

    if buf:
        content = "".join(prev_tail + buf)
        segments.append(_SyntheticSegment(
            content, buf_start_line, buf_start_line + len(buf) - 1,
            name=node_name, chunk_type=node_type, refs=node_refs,
        ))

    return segments if segments else [_Segment(node, src, lang)]


class _SyntheticSegment:
    """Fallback segment for raw text slices (no AST node). Carries optional
    name/chunk_type/refs so a merge of small named segments (_merge_small)
    can keep an identity instead of going anonymous."""
    __slots__ = ("_content", "start_line", "end_line", "chunk_type", "name", "refs")

    def __init__(self, content: str, start_line: int, end_line: int,
                 *, chunk_type: str = "block", name: str | None = None,
                 refs: dict[str, list[str]] | None = None):
        self._content  = content
        self.start_line = start_line
        self.end_line   = end_line
        self.chunk_type = chunk_type
        self.name       = name
        self.refs       = refs if refs is not None else _EMPTY_REFS

    def content(self, src: bytes) -> str:
        return self._content

    def size(self, src: bytes) -> int:
        # Return UTF-8 byte count to match _Segment.size(), so merge-time
        # comparisons against CHUNK_MIN / CHUNK_MAX use a single unit.
        return len(self._content.encode("utf-8"))


def _merge_small(segs: list, src: bytes) -> list:
    """Coalesces consecutive sub-CHUNK_MIN segments. BOTH sides must be
    trivia: absorbing a substantive boundary into a small one would erase
    its name from the symbol graph. Never merges past CHUNK_MAX."""
    if not segs:
        return segs

    merged: list = [segs[0]]
    for seg in segs[1:]:
        prev = merged[-1]
        prev_size = prev.size(src)
        # +1 for the "\n" inserted between contents on merge.
        merged_size = prev_size + 1 + seg.size(src)
        # seg.start_line > prev.end_line excludes salvage's deliberate
        # container/nested-boundary overlap; merging those would corrupt
        # the synthetic segment's line span.
        if (prev_size < CHUNK_MIN and seg.size(src) < CHUNK_MIN
                and merged_size <= CHUNK_MAX and seg.start_line > prev.end_line):
            pc = prev.content(src)
            # Only add a separator when prev's content doesn't already end in "\n".
            combined = pc + ("" if pc.endswith("\n") else "\n") + seg.content(src)
            # "<module>" residue is filler; never let it override a real
            # boundary's name.
            if prev.chunk_type == "module" and seg.name and seg.chunk_type != "module":
                m_name, m_type = seg.name, seg.chunk_type
            else:
                m_name = prev.name or seg.name
                m_type = prev.chunk_type if prev.name else seg.chunk_type
            m_refs = _merge_refs(prev.refs, seg.refs)
            syn = _SyntheticSegment(combined, prev.start_line, seg.end_line,
                                    chunk_type=m_type, name=m_name, refs=m_refs)
            merged[-1] = syn
        else:
            merged.append(seg)
    return merged


# Identity-free trivia (lone '#if'/'#endif', bare 'namespace X {') has no
# name of its own, so it's safe to fold into a neighbour without touching
# ITS name, unlike _merge_small's guard, which protects the opposite.
# BUG preserved: both patterns are C and C++ syntax, but they run for every
# language. A fix moves chunk boundaries and needs a CHUNKER_VERSION bump.
_PREPROC_CONDITIONAL_RE = re.compile(r"^#\s*(if|ifdef|ifndef|elif|else|endif)\b")
_CONTAINER_OPEN_RE = re.compile(r'^(namespace\s+[\w:]+\s*\{|extern\s+"C"\s*\{)\s*$')
# Braces/parens/semicolons/commas only (a lone "}", "};", "} {", …).
_PUNCT_ONLY_RE = re.compile(r"[{}();,\s]*$")


def _span_end_line(start_line: int, content: str) -> int:
    """end_line from `content`'s own newline count, not a neighbour's raw
    tree-sitter end_line, since the latter can claim a blank-line gap that
    isn't in `content` and overlap the following chunk's start_line."""
    n = content.count("\n")
    if content.endswith("\n"):
        return start_line + n - 1 if n else start_line
    return start_line + n


def _is_identity_free(text: str, lang: str) -> bool:
    """True iff every non-blank line is a preproc conditional, punctuation,
    a comment, or a bare container opening. Deliberately NOT #include/
    #define/#pragma: false positives fold real identity into a neighbour."""
    prefixes = _COMMENT_PREFIXES.get(lang, _COMMENT_PREFIXES_DEFAULT)
    spec = _lang_spec(lang)
    body = text if spec is not None and not spec.block_comments \
        else re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    saw_line = False
    for raw_line in body.splitlines():
        s = raw_line.strip()
        if not s:
            continue
        saw_line = True
        if _PREPROC_CONDITIONAL_RE.match(s):
            continue
        if _CONTAINER_OPEN_RE.match(s):
            continue
        # comment, possibly preceded by brace/punctuation ('} // namespace Foo')
        core = s.lstrip("{}();, \t")
        # Bare "*" must be alone or followed by space/'/'/EOL (doc-comment
        # continuation), or it misclassifies a real '*ptr = val;' statement.
        def _is_comment_prefixed(x: str) -> bool:
            for pfx in prefixes:
                if pfx == "*":
                    if re.match(r"^\*(\s|/|$)", x):
                        return True
                elif x.startswith(pfx):
                    return True
            return False
        if _is_comment_prefixed(core) or _is_comment_prefixed(s):
            continue
        if _PUNCT_ONLY_RE.fullmatch(s):
            continue
        return False
    return saw_line  # an all-blank fragment has nothing to classify, not identity-free


def _absorb_identity_free_fragments(segs: list, src: bytes, lang: str) -> list:
    """Folds an identity-free fragment (see module comment above) into a
    neighbour. BACKWARD by default; FORWARD for an unclosed leading guard,
    since '#if' introduces what follows it. Skips if it would break CHUNK_MAX."""
    if not segs:
        return segs

    def is_leading_guard(text: str) -> bool:
        has_open = re.search(r"^#\s*(if|ifdef|ifndef)\b", text, re.M) is not None
        has_close = re.search(r"^#\s*endif\b", text, re.M) is not None
        if has_open and not has_close:
            return True
        # A bare container opening introduces what follows it, so it belongs
        # forward, not appended after an unrelated preceding chunk.
        return any(_CONTAINER_OPEN_RE.match(ln.strip()) for ln in text.splitlines())

    out: list = list(segs)
    i = 0
    while i < len(out):
        seg = out[i]
        if seg.size(src) >= CHUNK_MIN or not _is_identity_free(seg.content(src), lang):
            i += 1
            continue

        text = seg.content(src)
        prev = out[i - 1] if i > 0 else None
        nxt = out[i + 1] if i + 1 < len(out) else None

        def absorb_backward() -> bool:
            if prev is None or prev.size(src) + 1 + seg.size(src) > CHUNK_MAX:
                return False
            pc = prev.content(src)
            combined = pc + ("" if pc.endswith("\n") else "\n") + text
            end_line = _span_end_line(prev.start_line, combined)
            out[i - 1] = _SyntheticSegment(combined, prev.start_line, end_line,
                                           chunk_type=prev.chunk_type, name=prev.name,
                                           refs=prev.refs)
            del out[i]
            return True

        def absorb_forward() -> bool:
            if nxt is None or seg.size(src) + 1 + nxt.size(src) > CHUNK_MAX:
                return False
            nc = nxt.content(src)
            combined = text + ("" if text.endswith("\n") else "\n") + nc
            end_line = _span_end_line(seg.start_line, combined)
            out[i + 1] = _SyntheticSegment(combined, seg.start_line, end_line,
                                           chunk_type=nxt.chunk_type, name=nxt.name,
                                           refs=nxt.refs)
            del out[i]
            return True

        if is_leading_guard(text):
            absorbed = absorb_forward() or absorb_backward()
        else:
            absorbed = absorb_backward() or absorb_forward()
        if not absorbed:
            i += 1  # neither neighbour could take it without breaking the ceiling
        # else: re-examine index i without advancing (absorption shrinks
        # `out` by one, so this can't loop forever).
    return out


def _hardwrap_text(text: str) -> list[str]:
    """Splits into <= CHUNK_MAX byte pieces on character boundaries, never
    mid-codepoint. Last resort for content (e.g. one giant line) a line
    split can't break."""
    if len(text.encode("utf-8")) <= CHUNK_MAX:
        return [text]
    pieces: list[str] = []
    buf: list[str] = []
    size = 0
    for ch in text:
        cb = len(ch.encode("utf-8"))
        if size + cb > CHUNK_MAX and buf:
            pieces.append("".join(buf))
            buf, size = [], 0
        buf.append(ch)
        size += cb
    if buf:
        pieces.append("".join(buf))
    return pieces


def _enforce_ceiling(segs: list, src: bytes, path: str | None = None,
                     counters: dict | None = None) -> list:
    """Hard-splits any segment still over CHUNK_MAX. Only fires for
    pathological input (one giant line) a line split can't break; logs the
    file so data/generated/minified blobs are easy to spot."""
    out: list = []
    oversize = 0
    extra_pieces = 0
    for s in segs:
        if s.size(src) <= CHUNK_MAX:
            out.append(s)
            continue
        oversize += 1
        pieces = _hardwrap_text(s.content(src))
        extra_pieces += len(pieces)
        # Apportion the line span per piece by newline count, or every piece
        # would inherit the parent's full span, breaking path:line citations.
        line = s.start_line
        for p in pieces:
            n = p.count("\n")
            if n == 0:
                # No newline at all: advance a virtual line by 1 anyway, or
                # every zero-newline piece reports the same duplicate span.
                next_line = line + 1
                end_line = next_line
            elif p.endswith("\n"):
                # n complete lines from `line`; no -1 here double-counts the
                # boundary line across this piece and the next.
                next_line = line + n
                end_line = next_line - 1
            else:
                # Ends mid-line: that last line continues in the next piece,
                # so the next piece must start on this SAME line.
                next_line = line + n
                end_line = next_line
            out.append(_SyntheticSegment(p, line, end_line,
                                         name=s.name, chunk_type=s.chunk_type, refs=s.refs))
            line = next_line
    if oversize:
        logger.warning(
            "chunking: %s had %d oversize chunk(s) (> %d B) hard-split into %d byte-windows "
            "— likely a data/generated/minified file; consider excluding it",
            path or "<unknown>", oversize, CHUNK_MAX, extra_pieces,
        )
        if counters is not None:
            counters["oversize_chunks"] = counters.get("oversize_chunks", 0) + oversize
            counters["oversize_files"] = counters.get("oversize_files", 0) + 1
    return out


def _finalize(segs: list, src: bytes, path: str | None = None,
              counters: dict | None = None) -> list[dict[str, Any]]:
    """Enforce the size ceiling, then emit chunk dicts (dropping whitespace-only)."""
    # Read + always reset the literal-cap tally, even without `counters`, so
    # it doesn't leak into the next file.
    capped_chunks = _literal_cap_state["capped_chunks"]
    dropped = _literal_cap_state["dropped"]
    _literal_cap_state["capped_chunks"] = 0
    _literal_cap_state["dropped"] = 0
    _literal_extract_cache.clear()
    if counters is not None and (capped_chunks or dropped):
        counters["literals_capped_chunks"] = counters.get("literals_capped_chunks", 0) + capped_chunks
        counters["literals_dropped"] = counters.get("literals_dropped", 0) + dropped

    return [
        {
            "chunk_type": seg.chunk_type,
            "name": seg.name,
            "start_line": seg.start_line,
            "end_line": seg.end_line,
            "content": seg.content(src),
            "refs": seg.refs,
            "literals": _decode_literals(seg.refs.get("literals", [])),
        }
        for seg in _enforce_ceiling(segs, src, path, counters)
        if seg.content(src).strip()
    ]


def _line_slice_oversized(content: str, start_line: int, name: str | None,
                          refs: dict[str, list[str]], chunk_type: str = "module") -> list:
    """Line-slices an oversized residue/header span with REAL per-piece
    line numbers, so no piece starts mid-statement or duplicates another's
    span, unlike the byte-window _hardwrap_text fallback."""
    lines = content.splitlines(keepends=True)
    out: list = []
    buf: list[str] = []
    buf_start = start_line
    char_count = 0
    ln = start_line
    for line in lines:
        buf.append(line)
        char_count += len(line)
        if char_count >= CHUNK_TARGET:
            out.append(_SyntheticSegment("".join(buf), buf_start, ln,
                                         chunk_type=chunk_type, name=name, refs=refs))
            buf, buf_start, char_count = [], ln + 1, 0
        ln += 1
    if buf and "".join(buf).strip():
        out.append(_SyntheticSegment("".join(buf), buf_start, ln - 1,
                                     chunk_type=chunk_type, name=name, refs=refs))
    return out


def _collect_module_residue(root: Node, lang: str, src: bytes) -> list:
    """Captures top-level spans no boundary node covers (imports, module
    docstring, `if __name__`). Without this, python's empty container set
    drops all module-level code, e.g. a 247-line argparse CLI."""
    segs: list = []
    run: list[Node] = []

    def flush() -> None:
        if not run:
            return
        content = src[run[0].start_byte:run[-1].end_byte].decode(errors="replace")
        if content.strip():
            # Module-level residue is where imports live for most languages;
            # without extracting refs here, 'imports' never populates.
            refs = _EMPTY_REFS
            for n in run:
                refs = _merge_refs(refs, _extract_refs(n, lang, src))
            start_line = run[0].start_point[0] + 1
            end_line = run[-1].end_point[0] + 1
            # Line-sliced here, not deferred to _finalize's byte-window
            # fallback, which would stamp every piece with the same span.
            if len(content.encode("utf-8")) > CHUNK_MAX:
                segs.extend(_line_slice_oversized(content, start_line, "<module>", refs))
            else:
                segs.append(_SyntheticSegment(
                    content, start_line, end_line,
                    chunk_type="module", name="<module>", refs=refs))
        run.clear()

    spec = _lang_spec(lang)
    for child in root.children:
        if not child.is_named:
            continue
        if _collect_boundaries(child, lang, src):
            flush()
            continue
        if spec is not None and spec.module_residue_split is not None \
                and spec.module_residue_split(child, src):
            flush()
            content = _node_text(child, src)
            if content.strip():
                m_start = child.start_point[0] + 1
                m_end = child.end_point[0] + 1
                m_refs = _extract_refs(child, lang, src)
                # Same line-slicing as the generic residue run above.
                if len(content.encode("utf-8")) > CHUNK_MAX:
                    segs.extend(_line_slice_oversized(content, m_start, "<module>:__main__", m_refs))
                else:
                    segs.append(_SyntheticSegment(
                        content, m_start, m_end,
                        chunk_type="module", name="<module>:__main__", refs=m_refs))
            continue
        run.append(child)
    flush()
    return segs


def _fill_coverage_gaps(segs: list, src: bytes) -> list:
    """Completeness backstop: emits chunks for non-blank lines no existing
    segment covers, e.g. a split node's own orchestration code between its
    child boundaries. chunk_type='module' so _merge_small treats it as filler."""
    if not segs:
        return []
    lines = src.decode(errors="replace").splitlines(keepends=True)
    n = len(lines)
    if n == 0:
        return []
    covered = bytearray(n + 2)  # 1-based line flags
    for s in segs:
        for ln in range(max(1, s.start_line), min(n, s.end_line) + 1):
            covered[ln] = 1

    gaps: list = []
    ln = 1
    while ln <= n:
        if covered[ln] or not lines[ln - 1].strip():
            ln += 1
            continue
        start = ln
        while ln <= n and not covered[ln]:
            ln += 1
        end = ln - 1
        while end > start and not lines[end - 1].strip():
            end -= 1
        # Skip spans that are only braces / punctuation / whitespace (a lone "}").
        if not "".join(lines[start - 1:end]).strip().strip("{}").strip():
            continue
        name = "<module>" if not lines[start - 1][:1].isspace() else "<block>"
        buf: list[str] = []
        buf_start = start
        char_count = 0
        for i in range(start, end + 1):
            buf.append(lines[i - 1])
            char_count += len(lines[i - 1])
            if char_count >= CHUNK_TARGET:
                gaps.append(_SyntheticSegment("".join(buf), buf_start, i,
                                              chunk_type="module", name=name))
                buf, buf_start, char_count = [], i + 1, 0
        if buf and "".join(buf).strip():
            gaps.append(_SyntheticSegment("".join(buf), buf_start, end,
                                          chunk_type="module", name=name))
    return gaps


# ---------------------------------------------------------------------------
# Symbol index (decoupled from chunk packing)
# ---------------------------------------------------------------------------

def _collect_symbols_from_root(root: Node, lang: str, src: bytes) -> list[dict[str, Any]]:
    """Every named boundary, recursing INTO boundaries (unlike
    _collect_boundaries, which stops at the first). So a folded class's
    methods are still recorded, independent of how chunking packed them."""
    out: list[dict[str, Any]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if _is_boundary(node, lang, src):
            kind = node.type
        elif _is_forward_declaration(node, lang, src):
            kind = FORWARD_DECLARATION
        else:
            kind = None
        if kind is not None:
            name = _extract_name(node, lang, src)
            if name:
                out.append({
                    "name":       name,
                    "kind":       kind,
                    "language":   lang,
                    "start_line": node.start_point[0] + 1,
                    "end_line":   node.end_point[0] + 1,
                })
        stack.extend(node.children)
    return out


# Line-comment prefixes per language (block comments /* */ handled separately).
_COMMENT_PREFIXES = _lang_table("line_comment_prefixes")
_COMMENT_PREFIXES_DEFAULT = ("//", "/*", "*/", "*")  # every language without its own prefixes
# Divider/comment punctuation: a span of only these (+ whitespace) has no text.
_DIVIDER_RE = re.compile(r"[/*#=\-_~<>|+.\s]")


def _refresh_from_registry() -> None:
    global _EXT_TO_LANG, _BOUNDARY_NODES, _CONTAINER_NODES, CODE_LANGUAGES
    global _SALVAGE_ELIGIBLE_BOUNDARY_TYPES, _COMMENT_PREFIXES, _MACRO_LANGS
    import chonks.languages as _languages
    import chonks.index.macro_heal as _macro_heal
    _EXT_TO_LANG = _languages.EXT_TO_LANG
    _BOUNDARY_NODES = _lang_table("boundary_nodes")
    _CONTAINER_NODES = _lang_table("container_nodes")
    CODE_LANGUAGES = frozenset(_EXT_TO_LANG.values())
    _SALVAGE_ELIGIBLE_BOUNDARY_TYPES = _lang_table("salvage_nodes")
    _COMMENT_PREFIXES = _lang_table("line_comment_prefixes")
    _MACRO_LANGS = _macro_heal._MACRO_LANGS


register_refresh(_refresh_from_registry)


def _is_comment_only(text: str, lang: str) -> bool:
    """True if every non-blank line is a comment or a pure-punctuation divider."""
    prefixes = _COMMENT_PREFIXES.get(lang, _COMMENT_PREFIXES_DEFAULT)
    spec = _lang_spec(lang)
    if spec is None or spec.block_comments:
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)  # strip block comments
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(prefixes):
            continue
        if not _DIVIDER_RE.sub("", s):  # only divider punctuation
            continue
        return False
    return True


def _comment_has_text(text: str) -> bool:
    """True if a comment carries real words (a label/doc), not just a divider rule."""
    return bool(_DIVIDER_RE.sub("", text))


def _attach_or_drop_comments(segs: list, src: bytes, lang: str) -> list:
    """A small labeled comment attaches FORWARD to the boundary it labels; a
    pure divider is dropped; a large block (>= CHUNK_MIN) stays standalone
    so it doesn't pollute a function's chunk."""
    out: list = []
    pending = None
    for seg in segs:
        text = seg.content(src)
        if _is_comment_only(text, lang):
            if not _comment_has_text(text):
                continue  # pure divider → drop
            if seg.size(src) >= CHUNK_MIN:
                if pending is not None:
                    out.append(pending)
                    pending = None
                out.append(seg)  # big comment block → standalone
                continue
            if pending is None:
                pending = seg
            else:  # coalesce consecutive small comments
                pending = _SyntheticSegment(
                    pending.content(src) + "\n" + text, pending.start_line, seg.end_line,
                    chunk_type=pending.chunk_type, name=pending.name)
            continue
        if pending is not None:  # attach held comment to this real boundary
            seg = _SyntheticSegment(
                pending.content(src) + "\n" + seg.content(src), pending.start_line, seg.end_line,
                chunk_type=seg.chunk_type, name=seg.name)
            pending = None
        out.append(seg)
    if pending is not None:
        # A trailing comment has no forward boundary to attach to; attach
        # it BACKWARD to the last segment instead, unless that breaks CHUNK_MAX.
        if out and out[-1].size(src) + 1 + pending.size(src) <= CHUNK_MAX:
            last = out[-1]
            lc = last.content(src)
            combined = lc + ("" if lc.endswith("\n") else "\n") + pending.content(src)
            # _span_end_line, not pending's raw end_line: a blank-line gap
            # between them isn't part of `combined` and would overstate the span.
            end_line = _span_end_line(last.start_line, combined)
            out[-1] = _SyntheticSegment(
                combined, last.start_line, end_line,
                chunk_type=last.chunk_type, name=last.name, refs=last.refs)
        else:
            out.append(pending)
    return out


def _dedup_segments(segs: list, src: bytes) -> list:
    """Drops later segments sharing (start_line, content) with an earlier
    one. Needed because salvage + split can independently re-derive the
    same nested boundary, producing two chunks with the same _chunk_id."""
    seen: set[tuple[int, str]] = set()
    out: list = []
    for seg in segs:
        key = (seg.start_line, seg.content(src))
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)
    return out


def segment_file(src: bytes, lang: str, *, path: str | None = None,
                 counters: dict | None = None, macros: set | None = None,
                 self_heal: bool = True) -> list[dict[str, Any]]:
    """Parses src and returns segment dicts. No chunk exceeds CHUNK_MAX
    (enforced by _finalize). `macros` pre-blanks known engine macros;
    `self_heal` additionally discovers and blanks unknown ones on the fly."""
    # Reset in case a prior file's segment_file call raised before reaching
    # _finalize's own reset.
    _literal_cap_state["capped_chunks"] = 0
    _literal_cap_state["dropped"] = 0
    _literal_extract_cache.clear()

    _LANG_PACK_NAME = _lang_table("grammar")
    parser = get_parser(_LANG_PACK_NAME.get(lang, lang))
    # timeout_micros is deprecated in 0.25, but progress_callback (its
    # replacement) is silently ignored for bytestring source.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        parser.timeout_micros = PARSE_TIMEOUT_MICROS

    # Parse a possibly-blanked buffer for STRUCTURE; content/names below
    # still read from the ORIGINAL src (blanking preserves byte offsets).
    parse_src = _blank_macros(src, macros) if macros else src
    root = parser.parse(parse_src).root_node

    # Discover macros, validate by strict error-count decrease (a real call
    # like ASSERT(x) is rejected since blanking it doesn't help), blank,
    # reparse, repeat (see _MAX_MACRO_PASSES for why one pass isn't enough).
    if self_heal and lang in _MACRO_LANGS and root.has_error:
        healed: set = set()
        for _ in range(_MAX_MACRO_PASSES):
            if not root.has_error:
                break
            base, error_ranges = _errors_with_ranges(root)
            # sorted() pins order before truncating, or a >cap file picks a
            # different subset per run. Qt specifiers unioned in AFTER truncation.
            discovered = sorted(_discover_macros(root, parse_src) - healed)[:_MAX_MACRO_CANDIDATES]
            cands = (set(discovered) | _QT_ACCESS_SPECIFIERS) - healed
            testable = _near_error_ranges(cands, parse_src, error_ranges)
            admitted = {m for m in testable
                        if _count_errors_upto(parser.parse(_blank_macros(parse_src, {m})).root_node,
                                              base) < base}
            if not admitted:
                break
            healed |= admitted
            parse_src = _blank_macros(parse_src, healed)
            root = parser.parse(parse_src).root_node
        if healed and counters is not None:
            counters.setdefault("discovered_macros", set()).update(healed)
            counters["macro_healed"] = 1
        elif not healed and counters is not None:
            # Sweep found nothing to heal; chonks/index/pipeline.py persists this by content
            # hash so unchanged unhealable files skip the sweep next time.
            counters["heal_unhealable"] = 1

    # Post-heal: parse_error now means genuinely unparseable, not "had a
    # macro we could heal".
    if counters is not None and root.has_error:
        counters["parse_error"] = 1

    if counters is not None:
        counters["symbols"] = _collect_symbols_from_root(root, lang, src)

    boundary_nodes = _collect_boundaries(root, lang, src, counters)

    # No structural nodes at all (e.g. a pure header): line-split by
    # CHUNK_TARGET with the same fallback overlap as _split_large_node.
    if not boundary_nodes:
        text = src.decode(errors="replace")
        lines = text.splitlines(keepends=True)
        segs: list = []
        buf: list[str] = []
        buf_start = 1
        char_count = 0
        prev_tail: list[str] = []
        for line in lines:
            buf.append(line)
            char_count += len(line)
            if char_count >= CHUNK_TARGET:
                content = "".join(prev_tail + buf)
                segs.append(_SyntheticSegment(content, buf_start, buf_start + len(buf) - 1))
                prev_tail = buf[-FALLBACK_OVERLAP_LINES:] if FALLBACK_OVERLAP_LINES > 0 else []
                buf_start += len(buf)
                buf = []
                char_count = 0
        if buf:
            content = "".join(prev_tail + buf)
            segs.append(_SyntheticSegment(content, buf_start, buf_start + len(buf) - 1))
        return _finalize(segs, src, path, counters)

    # A salvaged container's nested boundaries can be re-derived again by
    # _split_large_node below; precompute their spans and skip the
    # top-level duplicate rather than double-emit the same _chunk_id.
    split_results: dict[int, list] = {}
    covered_spans: set[tuple[int, int]] = set()
    for node in boundary_nodes:
        if _Segment(node, src, lang).size(src) > CHUNK_MAX:
            split_results[id(node)] = _split_large_node(node, lang, src)
            for child in node.children:
                if child.is_named:
                    for b in _collect_boundaries(child, lang, src):
                        covered_spans.add((b.start_byte, b.end_byte))

    raw_segs: list = []
    for node in boundary_nodes:
        if id(node) in split_results:
            raw_segs.extend(split_results[id(node)])
        elif (node.start_byte, node.end_byte) in covered_spans:
            continue  # already represented via an oversize sibling's split
        else:
            raw_segs.append(_Segment(node, src, lang))

    # Defense in depth for any collision the span-level filter above missed.
    raw_segs = _dedup_segments(raw_segs, src)

    # Fold in module-level residue; only re-sort if any was added, so a
    # boundary-only file keeps its already-canonical order.
    residue_segs = _collect_module_residue(root, lang, src)
    if residue_segs:
        raw_segs.extend(residue_segs)

    gap_segs = _fill_coverage_gaps(raw_segs, src)
    if gap_segs:
        raw_segs.extend(gap_segs)

    if residue_segs or gap_segs:
        raw_segs.sort(key=lambda s: s.start_line)
        # Comment-only spans come only from residue/gaps; attach forward.
        raw_segs = _attach_or_drop_comments(raw_segs, src, lang)

    # Absorb identity-free trivia BEFORE _merge_small, a separate pass since
    # the two rules' guards are opposite (see _absorb_identity_free_fragments).
    raw_segs = _absorb_identity_free_fragments(raw_segs, src, lang)
    raw_segs = _merge_small(raw_segs, src)
    return _finalize(raw_segs, src, path, counters)
