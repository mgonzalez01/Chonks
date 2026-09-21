"""Which files the indexer admits: extension tables and size guards."""

import hashlib
from pathlib import Path


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:  # 64KB is an arbitrary streaming-read size, not a format constant
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


DEFAULT_FALLBACK_EXTENSIONS = [
    ".html", ".vue", ".svelte", ".md", ".markdown",
    ".yaml", ".yml", ".toml", ".json",
]

# .md is excluded from the size guard (often the best doc, never gated);
# .html is included since at this size it's a generated bundle, not
# hand-written docs.
DATA_BLOB_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".html"}

# Size threshold (bytes) above which a DATA_BLOB_EXTENSIONS file is skipped
# rather than chunked.
DEFAULT_DATA_BLOB_SIZE_LIMIT = 256 * 1024

# Second size-guard family: bundles that look like real source by size alone,
# so this also requires line density over MINIFIED_AVG_LINE_LEN (minified
# output only).
MINIFIED_GUARD_EXTENSIONS = {".js", ".mjs", ".cjs", ".css"}
MINIFIED_AVG_LINE_LEN = 500


def _is_oversize_data_blob(fpath: Path, ext: str, limit: int) -> bool:
    """True if `fpath` should be skipped under the data-blob policy instead
    of queued for chunking. `limit` <= 0 disables the guard entirely."""
    if limit <= 0:
        return False
    is_data = ext in DATA_BLOB_EXTENSIONS
    is_bundle_ext = ext in MINIFIED_GUARD_EXTENSIONS
    if not (is_data or is_bundle_ext):
        return False
    try:
        size = fpath.stat().st_size
    except OSError:
        return False
    if size <= limit:
        return False
    if is_data:
        return True
    # Sample the first 64KB rather than the whole file: density is uniform in
    # minified output, and reading a 50MB bundle just to decide to skip it
    # defeats the point.
    try:
        with open(fpath, "rb") as f:
            sample = f.read(65536)
    except OSError:
        return False
    lines = max(1, sample.count(b"\n"))
    return len(sample) / lines > MINIFIED_AVG_LINE_LEN
