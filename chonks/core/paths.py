"""Path normalization and filtering helpers for chunking."""
import os
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

_IGNORE_CASE = sys.platform in ("win32", "darwin")


def _fold(path: str) -> str:
    return path.lower() if _IGNORE_CASE else path


def _to_stored_path(fpath: Path, root: Path | None) -> str:
    """Portable forward-slash path, relative to root if provided."""
    if root is not None:
        try:
            rel = fpath.relative_to(root)
            return rel.as_posix()
        except ValueError:
            pass  # fpath outside root, fall through to absolute
    return fpath.as_posix()


def _spelling_key(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _same_entry(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _is_under(path: Path, root: Path) -> bool:
    """Whether `path` is `root` or lies under it, counting another case or
    Unicode form of root that the file system opens as root."""
    head = Path(*path.parts[:len(root.parts)])
    return head == root or (_spelling_key(str(head)) == _spelling_key(str(root)) and _same_entry(head, root))


def _spelled_as_listed(paths: list[Path], root: Path | None) -> list[Path]:
    """Each resolved path with every name as its directory lists it, after root's
    own spelling for a path under root, so a walk would spell it the same."""
    listings: dict[Path, set[str]] = {}
    folded: dict[Path, dict[str, list[str]]] = {}

    def listed(parent: Path, name: str) -> str:
        if parent not in listings:
            try:
                listings[parent] = set(os.listdir(parent))
            except OSError:
                listings[parent] = set()
        if name in listings[parent]:
            return name
        if parent not in folded:
            folded[parent] = {}
            for n in listings[parent]:
                folded[parent].setdefault(_spelling_key(n), []).append(n)
        candidates = folded[parent].get(_spelling_key(name), ())
        return next((n for n in candidates if _same_entry(parent / n, parent / name)), name)

    out = []
    for path in paths:
        spelled = root if root is not None and _is_under(path, root) else Path(path.anchor)
        for name in path.parts[len(spelled.parts):]:
            spelled = spelled / listed(spelled, name)
        out.append(spelled)
    return out


def _normalize_prefixes(prefixes: list[str] | None) -> list[str]:
    """Normalize path prefixes to forward-slash form with a trailing slash,
    so "vendor" and "vendor/" both match paths under vendor/ but never a
    sibling file like "external_thing.cpp"."""
    if not prefixes:
        return []
    out: list[str] = []
    for raw in prefixes:
        p = raw.replace("\\", "/").strip()
        if not p:
            continue
        if not p.endswith("/"):
            p = p + "/"
        out.append(p)
    return out


@lru_cache(maxsize=None)
def _segment_patterns(entry: str) -> tuple[re.Pattern[str], ...]:
    """One pattern per segment of `entry`: `*` matches within the segment, the rest is literal."""
    return tuple(re.compile("[^/]*".join(re.escape(part) for part in seg.split("*")))
                 for seg in entry.rstrip("/").split("/"))


def _segments_agree(parts: list[str], entry: str) -> bool:
    return all(p.fullmatch(s) for p, s in zip(_segment_patterns(entry), parts))


def _entry_matches(path: str, entry: str) -> bool:
    """Whether `path` is `entry` or lies under it; both folded."""
    if "*" not in entry:
        return path == entry[:-1] or path.startswith(entry)
    parts = path.split("/")
    return len(parts) >= len(_segment_patterns(entry)) and _segments_agree(parts, entry)


def _entry_reaches_into(dir_path: str, entry: str) -> bool:
    """Whether `entry` names `dir_path` or something under it; both folded."""
    if "*" not in entry:
        return entry.startswith(dir_path + "/")
    parts = dir_path.split("/")
    return len(parts) <= len(_segment_patterns(entry)) and _segments_agree(parts, entry)


def _specificity(entry: str) -> tuple[int, int]:
    """Deeper entries win, then those with more literal characters; for plain
    prefixes matching one path this orders them by length."""
    return entry.count("/"), len(entry) - entry.count("*")


def _most_specific_match(stored_path: str, prefixes: list[str]) -> str | None:
    """Most specific entry in `prefixes` matching `stored_path`, or None.
    Case is ignored where the file system ignores it by default."""
    path = _fold(stored_path)
    best: str | None = None
    for p in prefixes:
        if _entry_matches(path, _fold(p)):
            if best is None or _specificity(p) > _specificity(best):
                best = p
    return best


def _path_allowed(stored_path: str, excludes: list[str], includes: list[str]) -> bool:
    """Allowed unless an exclude matches, unless a strictly more specific
    include rescues it. Includes are exception markers only, never an
    allowlist on their own."""
    excl = _most_specific_match(stored_path, excludes)
    if excl is None:
        return True
    incl = _most_specific_match(stored_path, includes)
    if incl is None:
        return False
    return _specificity(incl) > _specificity(excl)


def _dir_should_prune(dir_stored_path: str, excludes: list[str], includes: list[str]) -> bool:
    if _path_allowed(dir_stored_path, excludes, includes):
        return False
    probe = _fold(dir_stored_path)
    return not any(_entry_reaches_into(probe, _fold(inc)) for inc in includes)


def _scope_exclusion(path_prefix: str, excludes: list[str], includes: list[str]) -> str | None:
    """The exclude entry that keeps everything under path_prefix out of the
    index, or None when some of it is indexed."""
    scopes = _normalize_prefixes([path_prefix])
    if not scopes:
        return None
    scope = scopes[0][:-1]
    if not _dir_should_prune(scope, excludes, includes):
        return None
    return _most_specific_match(scope, excludes)


def _names_something(root: Path, entry: str) -> bool:
    if "*" not in entry:
        return (root / entry.rstrip("/")).exists()
    return _matches_below(root, entry.rstrip("/").split("/"), _segment_patterns(_fold(entry)))


def _matches_below(d: Path, segments: list[str], patterns: tuple[re.Pattern[str], ...]) -> bool:
    """Whether `segments` name something under `d`; stops at the first hit."""
    if not segments:
        return d.exists()
    if "*" not in segments[0]:
        return _matches_below(d / segments[0], segments[1:], patterns[1:])
    try:
        names = os.listdir(d)
    except OSError:
        return False
    return any(patterns[0].fullmatch(_fold(name)) and _matches_below(d / name, segments[1:], patterns[1:])
               for name in names)


def _missing_prefixes(root: Path, prefixes: list[str]) -> list[str]:
    """Entries of `prefixes` that name nothing on disk under root."""
    return [p for p in prefixes if not _names_something(root, p)]


def _dotdir_prefix(stored_dir_path: str) -> str | None:
    """First two segments of a dot-directory path; ".git" doesn't count (VCS,
    not pollution)."""
    parts = [p for p in stored_dir_path.split("/") if p]
    if not parts or not parts[0].startswith(".") or parts[0] == ".git":
        return None
    return "/".join(parts[:2]) if len(parts) > 1 else parts[0]


def _stored_dirname(stored_path: str) -> str:
    """Directory portion of a forward-slash stored path, "" if root-level."""
    head, sep, _ = stored_path.rpartition("/")
    return head if sep else ""
