"""Chunk ids and the metadata column of a chunk row."""

import hashlib

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
    for key in ("calls", "imports", "inherits"):
        vals = (refs or {}).get(key)
        if vals:
            md[key] = vals
    return md or None
