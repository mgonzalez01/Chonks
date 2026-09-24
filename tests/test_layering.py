"""Guards the seams between chonks/ layers: import direction, where third-party
packages and environment reads live, and where language and edge-type literals live."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHONKS_ROOT = _REPO_ROOT / "chonks"

_LAYER_PACKAGE_NAMES = (
    "core", "languages", "storage", "embed", "index", "retrieval", "serve", "ops",
)

_LAYER_MAP: dict[str, str] = {
    "chonks": "core",
    "chonks.server": "ops",
}

# What each layer may import, not what it imports today.
_ALLOWED_DIRECTIONS: dict[str, set[str]] = {
    "ops":       {"serve", "retrieval", "index", "storage", "embed", "languages", "core"},
    "serve":     {"retrieval", "index", "storage", "embed", "core"},
    "retrieval": {"storage", "embed", "languages", "core"},
    "index":     {"storage", "embed", "languages", "core"},
    "storage":   {"languages", "core"},
    "embed":     {"core"},
    "languages": {"core"},
    "core":      set(),
}

_KNOWN_INVERSIONS: set[tuple[str, str]] = {
    ("chonks.retrieval.repomap", "chonks.index.graph.pagerank"),
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
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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


@pytest.mark.parametrize("inversion", sorted(_KNOWN_INVERSIONS), ids="->".join)
def test_known_inversion_still_exists(inversion):
    assert inversion in _EDGES, f"{inversion} no longer imports; drop it from _KNOWN_INVERSIONS"


@pytest.mark.parametrize("module", ["chonks.storage.store"])
def test_storage_import_is_light(module):
    code = (
        f"import sys, {module}; "
        "print(sorted({'networkx', 'tqdm', 'tree_sitter'} & set(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


_TREES = {_rel(p): ast.parse(p.read_text(encoding="utf-8"), filename=str(p)) for p in _all_chonks_files()}

# Each package is imported only where it is wrapped.
_CONFINED_PACKAGES: dict[str, tuple[str, ...]] = {
    "scipy": ("chonks/ops/subsystems.py",),
    "sqlite_vec": ("chonks/storage/",),
    "fastapi": ("chonks/serve/",),
    "tree_sitter_language_pack": ("chonks/index/segment.py",),
}


def _top_level_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("package", sorted(_CONFINED_PACKAGES))
def test_third_party_package_stays_where_it_is_wrapped(package):
    allowed = _CONFINED_PACKAGES[package]
    offenders = [rel for rel, tree in _TREES.items()
                 if package in _top_level_imports(tree) and not rel.startswith(allowed)]
    assert offenders == []


_ENV_READERS = ("chonks/ops/", "chonks/serve/main.py", "chonks/index/graph/knn.py")


def _reads_environment(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")
                and isinstance(node.value, ast.Name) and node.value.id == "os"):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "os" and any(
                alias.name in ("environ", "getenv") for alias in node.names):
            return True
    return False


def test_environment_is_read_only_at_entry_points():
    offenders = [rel for rel, tree in _TREES.items()
                 if _reads_environment(tree) and not rel.startswith(_ENV_READERS)]
    assert offenders == []


# Language names and the node types a spec chunks by are spelled out only in
# chonks/languages/; edge types only there and in chonks/core/edges.py. Other
# modules read them from the registry or from core.edges.
_LANGUAGE_HOMES = ("chonks/languages/",)
_EDGE_TYPE_HOMES = ("chonks/languages/", "chonks/core/edges.py")
# Chunk types chonks assigns itself, whatever a grammar calls its nodes.
_OWN_CHUNK_TYPES = frozenset({"module"})
# Literals that predate this check. Remove entries; never add them.
_KNOWN_LITERALS: set[tuple[str, str]] = {
    ("chonks/index/plugins.py", "javascript"),
    ("chonks/index/macro_heal.py", "class_specifier"),
    ("chonks/index/macro_heal.py", "struct_specifier"),
    *{("chonks/index/graph/refs.py", t)
      for t in ("associated", "calls", "imports", "inherits", "mentions", "xlang")},
    *{("chonks/index/refs_extract.py", t) for t in ("calls", "imports", "inherits")},
    *{("chonks/index/rows.py", t) for t in ("calls", "imports", "inherits")},
    *{("chonks/ops/report.py", t) for t in ("associated", "mentions", "xlang")},
    *{("chonks/retrieval/graph_queries.py", t) for t in ("associated", "mentions", "xlang")},
    *{("chonks/retrieval/trace.py", t) for t in ("associated", "mentions")},
    ("chonks/storage/store.py", "mentions"),
}


def _language_literals() -> frozenset[str]:
    from dataclasses import fields

    from chonks.languages import REGISTRY
    names: set[str] = set()
    for spec in REGISTRY.specs:
        names.add(spec.name)
        for f in fields(spec):
            if f.name.endswith(("_nodes", "_chunk_types")):
                names |= getattr(spec, f.name)
        names |= set(spec.kind_labels) | set(spec.boundary_filters) | set(spec.forward_declarations)
    return frozenset(names - _OWN_CHUNK_TYPES)


def _edge_type_literals() -> frozenset[str]:
    from chonks.core.edges import DEFAULT_EDGE_TYPE_WEIGHTS
    return frozenset(DEFAULT_EDGE_TYPE_WEIGHTS)


def _literals_outside_their_home() -> set[tuple[str, str]]:
    watched = ((_language_literals(), _LANGUAGE_HOMES), (_edge_type_literals(), _EDGE_TYPE_HOMES))
    found: set[tuple[str, str]] = set()
    for rel, tree in _TREES.items():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            for names, homes in watched:
                if node.value in names and not rel.startswith(homes):
                    found.add((rel, node.value))
    return found


def test_language_and_edge_type_literals_stay_home():
    assert _literals_outside_their_home() - _KNOWN_LITERALS == set()


@pytest.mark.parametrize("literal", sorted(_KNOWN_LITERALS), ids=":".join)
def test_known_literal_still_exists(literal):
    assert literal in _literals_outside_their_home(), f"{literal} is gone; drop it from _KNOWN_LITERALS"
