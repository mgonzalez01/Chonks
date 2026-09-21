"""Non-AST text and markdown segmentation fallback."""

import re
from typing import Any

from chonks.index.segment import CHUNK_TARGET, FALLBACK_OVERLAP_LINES, _SyntheticSegment, _finalize, _merge_small

_MARKDOWN_EXTS = (".md", ".markdown")

# Commonmark's trailing '#' closing sequence ("## Title ##") is stripped in
# _heading_name, not here, so this capture can stay simple/lazy.
_ATX_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})(?:[ \t]+(.+?))?[ \t]*$")
# Toggles fence state on any ``` or ~~~ line so '#' inside one isn't a heading.
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def _is_markdown_path(path: str | None) -> bool:
    return path is not None and path.lower().endswith(_MARKDOWN_EXTS)


def _heading_name(raw: str | None) -> str | None:
    """Strips Commonmark's trailing '#' closing sequence. Bare '#' (empty
    heading) returns None."""
    text = raw or ""
    text = re.sub(r"[ \t]+#+[ \t]*$", "", text).strip()
    return text or None


def _split_markdown_sections(lines: list[str]) -> list[tuple[int, int, str | None]]:
    """(start_line, end_line, name) spans at ATX heading boundaries,
    ignoring '#' inside fenced code blocks."""
    sections: list[tuple[int, int, str | None]] = []
    current_start = 1
    current_name: str | None = None
    in_fence = False
    fence_char = ""
    for i, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n")
        fence_m = _FENCE_RE.match(line)
        if fence_m:
            char = fence_m.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_char = char
            elif char == fence_char:
                # Only a same-character fence closes the block (CommonMark).
                in_fence = False
                fence_char = ""
            continue
        if in_fence:
            continue
        m = _ATX_HEADING_RE.match(line)
        if not m:
            continue
        if i > current_start:
            sections.append((current_start, i - 1, current_name))
        current_start = i
        current_name = _heading_name(m.group(2))
    sections.append((current_start, len(lines), current_name))
    return sections


def _slice_section(lines: list[str], name: str | None) -> list:
    """Line-slices one section into <= CHUNK_TARGET pieces, all keeping its
    name. Overlap is section-local: a different section's trailing context
    isn't useful here."""
    segs: list = []
    if not lines:
        return segs
    buf: list[str] = []
    buf_start = 1
    char_count = 0
    prev_tail: list[str] = []
    for line in lines:
        buf.append(line)
        char_count += len(line)
        if char_count >= CHUNK_TARGET:
            content = "".join(prev_tail + buf)
            segs.append(_SyntheticSegment(
                content, buf_start, buf_start + len(buf) - 1,
                chunk_type="text", name=name))
            prev_tail = buf[-FALLBACK_OVERLAP_LINES:] if FALLBACK_OVERLAP_LINES > 0 else []
            buf_start += len(buf)
            buf = []
            char_count = 0
    if buf:
        content = "".join(prev_tail + buf)
        segs.append(_SyntheticSegment(
            content, buf_start, buf_start + len(buf) - 1,
            chunk_type="text", name=name))
    return segs


def _segment_markdown(src: bytes, text_src: str) -> list:
    """Heading-aware split for .md/.markdown, named after heading text
    instead of the generic fallback's nameless line-tiling."""
    lines = text_src.splitlines(keepends=True)
    segs: list = []
    for start, end, name in _split_markdown_sections(lines):
        sec_lines = lines[start - 1:end]
        for seg in _slice_section(sec_lines, name):
            # _slice_section's line numbers are section-local (start at 1);
            # rebase them onto the file's absolute line numbers.
            seg.start_line += start - 1
            seg.end_line += start - 1
            segs.append(seg)
    return _merge_small(segs, src)


def segment_text_file(src: bytes, *, path: str | None = None,
                      counters: dict | None = None) -> list[dict[str, Any]]:
    """Non-AST fallback for config-allowlisted extensions with no grammar.
    Markdown is the one exception, split at headings; everything else is
    nameless line-tiling."""
    text_src = src.decode(errors="replace")
    if _is_markdown_path(path):
        return _finalize(_segment_markdown(src, text_src), src, path, counters)
    lines = text_src.splitlines(keepends=True)
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
            segs.append(_SyntheticSegment(
                content, buf_start, buf_start + len(buf) - 1,
                chunk_type="text", name=None))
            prev_tail = buf[-FALLBACK_OVERLAP_LINES:] if FALLBACK_OVERLAP_LINES > 0 else []
            buf_start += len(buf)
            buf = []
            char_count = 0
    if buf:
        content = "".join(prev_tail + buf)
        segs.append(_SyntheticSegment(
            content, buf_start, buf_start + len(buf) - 1,
            chunk_type="text", name=None))
    return _finalize(segs, src, path, counters)
