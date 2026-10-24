"""Chunk ids, the metadata column of a chunk row, and the chunk a symbol row points at."""

import hashlib
from bisect import bisect_right
from collections.abc import Callable

from chonks.core.edges import TYPED_EDGE_TYPES
from chonks.languages import get_or_none as _lang_spec


def _chunk_id(path: str, start_line: int, content: str) -> str:
    # Hash the full content, not a truncated prefix, so two chunks at the same
    # path:start_line that differ only later still get different ids. Otherwise
    # INSERT OR REPLACE would silently overwrite one with the other.
    key = f"{path}:{start_line}:{content}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _file_metadata(stored_path: str, lang: str) -> dict | None:
    """File-level context not derivable from chunk content alone (only Python
    module dotted-path today). Open JSON column, so new keys need no migration."""
    spec = _lang_spec(lang)
    if spec is not None and spec.file_metadata is not None:
        return spec.file_metadata(stored_path)
    return None


def _chunk_metadata(file_metadata: dict | None, refs: dict | None) -> dict | None:
    """Merge file-level metadata with a chunk's AST-derived refs. Only
    non-empty ref lists are added, so untouched chunks keep the column as
    before typed edges existed."""
    md = dict(file_metadata) if file_metadata else {}
    for key in TYPED_EDGE_TYPES:
        vals = (refs or {}).get(key)
        if vals:
            md[key] = vals
    return md or None


def _start_col(content: str, start_line: int, end_line: int, lines: list[bytes]) -> int:
    """Byte column on `start_line` where a chunk's text begins, or the line's end when its
    first line is not there verbatim. A line-fallback slice carries overlap lines above it."""
    parts = content.encode("utf-8").split(b"\n")
    if len(parts) > 1 and not parts[-1]:
        parts.pop()
    first = parts[max(0, len(parts) - (end_line - start_line + 1))]
    line = lines[start_line - 1] if 0 < start_line <= len(lines) else b""
    if line.endswith(first):
        return len(line) - len(first)
    found = line.find(first)
    return found if found >= 0 else len(line)


def _chunk_locator(src: bytes, chunks: list[tuple[int, int, str, str]]) -> Callable[[int, int], str | None]:
    """locate(line, col) -> id of the latest-starting chunk holding that position, else of the
    next chunk starting on that line. Chunks can share a line, and an errored node's chunk
    spans the chunks salvaged from inside it."""
    lines = src.split(b"\n")
    spans = sorted(
        (((s, _start_col(text, s, e, lines)), e, cid) for s, e, text, cid in chunks), key=lambda span: span[0]
    )
    starts = [start for start, _, _ in spans]

    def locate(line: int, col: int) -> str | None:
        j = bisect_right(starts, (line, col))
        for i in range(j - 1, -1, -1):
            if line <= spans[i][1]:
                return spans[i][2]
        return spans[j][2] if j < len(spans) and starts[j][0] == line else None

    return locate
