"""Guards the plan's layer boundaries over today's flat chonks/ layout."""
import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHONKS_ROOT = _REPO_ROOT / "chonks"

_LAYER_PACKAGE_NAMES = (
    "core", "languages", "storage", "embed", "index", "retrieval", "serve", "ops",
)

_LAYER_MAP: dict[str, str] = {
    "chonks": "core",
    "chonks.chunker": "index",
    "chonks.chunking": "index",
    "chonks.cli": "ops",
    "chonks.doctor": "ops",
    "chonks.embedder": "embed",
    "chonks.init": "ops",
    "chonks.query_reformulate": "retrieval",
    "chonks.report": "ops",
    "chonks.repomap": "index",
    "chonks.repomap._shared": "core",
    "chonks.repomap.knn": "index",
    "chonks.repomap.pagerank": "index",
    "chonks.repomap.refs": "index",
    "chonks.repomap.render": "retrieval",
    "chonks.repomap.trace": "retrieval",
    "chonks.research": "retrieval",
    "chonks.searcher": "retrieval",
    "chonks.server": "serve",
    "chonks.store": "storage",
    "chonks.summaries": "index",
}

_ALLOWED_DIRECTIONS: dict[str, set[str]] = {
    "ops":       {"serve", "retrieval", "index", "storage", "embed", "languages"},
    "serve":     {"retrieval", "index", "storage", "embed"},
    "retrieval": {"storage", "embed", "languages", "core"},
    "index":     {"storage", "embed", "languages", "core"},
    "storage":   {"languages", "core"},
    "embed":     {"core"},
    "languages": {"core"},
    "core":      set(),
}

_KNOWN_INVERSIONS: set[tuple[str, str]] = {
    ("chonks.chunking", "chonks.ops.diagnostics"),
    ("chonks.repomap", "chonks.repomap.render"),
    ("chonks.repomap", "chonks.repomap.trace"),
    ("chonks.repomap.render", "chonks.repomap.pagerank"),
    ("chonks.research", "chonks.repomap"),
    ("chonks.store", "chonks.repomap"),
}


def _module_name(path: Path) -> str:
    rel = path.relative_to(_REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _resolve_relative(current: str, dots: int, tail: str, is_init: bool) -> str:
    own_package = current if is_init else (
        current.rsplit(".", 1)[0] if "." in current else ""
    )
    pkg_parts = own_package.split(".") if own_package else []
    strip = dots - 1
    kept = pkg_parts[: len(pkg_parts) - strip] if strip <= len(pkg_parts) else []
    return ".".join(kept + tail.split(".")) if tail else ".".join(kept)


def _module_imports(path: Path) -> set[str]:
    mod = _module_name(path)
    is_init = path.name == "__init__.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    skip: set[ast.AST] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            skip.update(ast.walk(node))

    targets: set[str] = set()
    for node in ast.walk(tree):
        if node in skip:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "chonks" or alias.name.startswith("chonks."):
                    targets.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                resolved = _resolve_relative(mod, node.level, node.module or "", is_init)
            else:
                resolved = node.module or ""
            if resolved == "chonks" or resolved.startswith("chonks."):
                targets.add(resolved)
    targets.discard(mod)
    return targets


def _all_chonks_files() -> list[Path]:
    return sorted(p for p in _CHONKS_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _layer_of(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) >= 2 and parts[1] in _LAYER_PACKAGE_NAMES:
        return parts[1]
    return _LAYER_MAP.get(module)


_MODULES = [_module_name(p) for p in _all_chonks_files()]
_EDGES = [
    (_module_name(path), imported)
    for path in _all_chonks_files()
    for imported in sorted(_module_imports(path))
]


@pytest.mark.parametrize("module", _MODULES)
def test_every_module_has_a_layer(module):
    assert _layer_of(module) is not None


@pytest.mark.parametrize("importer,imported", _EDGES)
def test_import_direction_is_allowed_or_known(importer, imported):
    importer_layer = _layer_of(importer)
    imported_layer = _layer_of(imported)
    if importer_layer == imported_layer:
        return
    if imported_layer in _ALLOWED_DIRECTIONS.get(importer_layer, set()):
        return
    assert (importer, imported) in _KNOWN_INVERSIONS
