"""Path normalization and filtering helpers for chunking."""
from pathlib import Path


def _to_stored_path(fpath: Path, root: Path | None) -> str:
    """Portable forward-slash path, relative to root if provided."""
    if root is not None:
        try:
            rel = fpath.relative_to(root)
            return rel.as_posix()
        except ValueError:
            pass  # fpath outside root, fall through to absolute
    return fpath.as_posix()


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


def _longest_prefix_match(stored_path: str, prefixes: list[str]) -> str | None:
    """Longest prefix in `prefixes` matching `stored_path` (path equals the
    prefix's directory or starts with it), or None."""
    best: str | None = None
    for p in prefixes:
        if stored_path == p[:-1] or stored_path.startswith(p):
            if best is None or len(p) > len(best):
                best = p
    return best


def _path_allowed(stored_path: str, excludes: list[str], includes: list[str]) -> bool:
    """Allowed unless an exclude matches, unless a strictly more specific
    include rescues it. Includes are exception markers only, never an
    allowlist on their own."""
    excl = _longest_prefix_match(stored_path, excludes)
    if excl is None:
        return True
    incl = _longest_prefix_match(stored_path, includes)
    if incl is None:
        return False
    return len(incl) > len(excl)


def _dir_should_prune(dir_stored_path: str, excludes: list[str], includes: list[str]) -> bool:
    if _path_allowed(dir_stored_path, excludes, includes):
        return False
    probe = dir_stored_path + "/"
    for inc in includes:
        if inc == probe or inc.startswith(probe):
            return False
    return True


def _dotdir_prefix(stored_dir_path: str) -> str | None:
    """First two segments of a dot-directory path; ".git" doesn't count (VCS,
    not pollution). Grouping must match chunking.dotdir_breakdown so this and
    doctor's report describe the same bucket."""
    parts = [p for p in stored_dir_path.split("/") if p]
    if not parts or not parts[0].startswith(".") or parts[0] == ".git":
        return None
    return "/".join(parts[:2]) if len(parts) > 1 else parts[0]


def _stored_dirname(stored_path: str) -> str:
    """Directory portion of a forward-slash stored path, "" if root-level."""
    head, sep, _ = stored_path.rpartition("/")
    return head if sep else ""
