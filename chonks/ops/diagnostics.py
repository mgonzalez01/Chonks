"""Path-family and dot-directory chunk-share diagnostics for the index report and doctor."""

from dataclasses import dataclass
from typing import Any, Iterable

from chonks.languages import CODE_LANGUAGES


# A healthy repo can be 90%+ one family; only warn when ALSO docs-kind.
FAMILY_DOCS_SHARE_WARN = 0.80    # (a) docs-share *within* the dominant family
FAMILY_TOTAL_SHARE_WARN = 0.40   # (a) that family's share of the whole corpus
CORPUS_DOCS_SHARE_WARN = 0.50    # (b) corpus-wide docs share


@dataclass
class FamilyStats:
    """One row of the path-family chunk-share breakdown."""
    family: str
    chunks: int
    files: int
    docs_chunks: int
    top_file: str | None
    top_file_chunks: int

    @property
    def docs_share(self) -> float:
        return self.docs_chunks / self.chunks if self.chunks else 0.0

    @property
    def chunks_per_file(self) -> float:
        return self.chunks / self.files if self.files else 0.0


def path_family(path: str) -> str:
    """First path segment, or "(root)" for a file with no directory
    component. `path` must be posix-style, relative to the indexed root."""
    head, sep, _ = path.partition("/")
    return head if sep else "(root)"


def family_breakdown(rows: Iterable[tuple[str, str | None]]) -> list[FamilyStats]:
    """Group chunk rows by top-level path family, sorted chunks DESC then
    family ASC for deterministic output."""
    by_family: dict[str, dict[str, Any]] = {}
    for path, language in rows:
        fam = path_family(path)
        entry = by_family.setdefault(
            fam, {"chunks": 0, "files": set(), "docs_chunks": 0, "file_counts": {}}
        )
        entry["chunks"] += 1
        entry["files"].add(path)
        if language not in CODE_LANGUAGES:
            entry["docs_chunks"] += 1
        entry["file_counts"][path] = entry["file_counts"].get(path, 0) + 1

    stats: list[FamilyStats] = []
    for fam, entry in by_family.items():
        top_file, top_file_chunks = None, 0
        for p, n in entry["file_counts"].items():
            if n > top_file_chunks:
                top_file, top_file_chunks = p, n
        stats.append(FamilyStats(
            family=fam,
            chunks=entry["chunks"],
            files=len(entry["files"]),
            docs_chunks=entry["docs_chunks"],
            top_file=top_file,
            top_file_chunks=top_file_chunks,
        ))
    stats.sort(key=lambda s: (-s.chunks, s.family))
    return stats


def dominance_warning(stats: list[FamilyStats], total_chunks: int) -> str | None:
    """Warns when a family is docs-heavy and large, or the corpus overall
    leans docs-kind (thresholds above). `stats` must be chunks-DESC."""
    if total_chunks == 0:
        return None

    for s in stats:
        family_share = s.chunks / total_chunks
        if s.docs_share >= FAMILY_DOCS_SHARE_WARN and family_share >= FAMILY_TOTAL_SHARE_WARN:
            return (
                f"WARNING: '{s.family}/' is {family_share:.0%} of the index "
                f"({s.chunks}/{total_chunks} chunks), {s.docs_share:.0%} of which is "
                f"docs-kind content. Largest file: {s.top_file} ({s.top_file_chunks} chunks). "
                f"If this is generated or vendored content, exclude it — config.json: "
                f'"exclude": ["{s.family}/"]'
            )

    total_docs = sum(s.docs_chunks for s in stats)
    corpus_docs_share = total_docs / total_chunks
    if corpus_docs_share >= CORPUS_DOCS_SHARE_WARN:
        return (
            f"WARNING: docs-kind chunks are {corpus_docs_share:.0%} of the whole index "
            f"({total_docs}/{total_chunks}). Consider excluding generated or vendored "
            f'docs in config.json, e.g. "exclude": ["<path>/"]'
        )
    return None


@dataclass
class DotDirStats:
    """One row of the dot-directory chunk-share breakdown."""
    prefix: str
    chunks: int
    files: int


def dotdir_breakdown(rows: Iterable[tuple[str, str | None]]) -> list[DotDirStats]:
    """Chunk/file counts by dot-dir prefix, catches a self-index run
    ingesting stale nested repo copies. ".git" is excluded as VCS metadata."""
    by_prefix: dict[str, dict[str, Any]] = {}
    for path, _language in rows:
        parts = path.split("/")
        if not parts[0].startswith(".") or parts[0] == ".git":
            continue
        # Drop the filename first: a file inside a dot-dir must group
        # under the dir, not under "<dir>/<file>".
        dir_parts = parts[:-1] or parts[:1]
        prefix = "/".join(dir_parts[:2])
        entry = by_prefix.setdefault(prefix, {"chunks": 0, "files": set()})
        entry["chunks"] += 1
        entry["files"].add(path)

    stats = [
        DotDirStats(prefix=p, chunks=e["chunks"], files=len(e["files"]))
        for p, e in by_prefix.items()
    ]
    stats.sort(key=lambda s: (-s.chunks, s.prefix))
    return stats


def dotdir_total_share(stats: list[DotDirStats], total_chunks: int) -> float:
    """Fraction of `total_chunks` accounted for by dotdir_breakdown's output."""
    if total_chunks == 0:
        return 0.0
    return sum(s.chunks for s in stats) / total_chunks
