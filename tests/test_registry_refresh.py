"""Every module-level value derived from the language registry must be rebound
by its module's `_refresh_from_registry` hook, and every hook must correspond
to a real derivation. Also proves the nine hooks actually track a language
added to the registry after import."""

import ast
from pathlib import Path

import pytest

import chonks.core.refresh as refresh
import chonks.languages as languages
from chonks.languages.spec import Field, LanguageSpec, LiteralSpec

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHONKS_ROOT = _REPO_ROOT / "chonks"
_SOURCE_MODULE = "chonks.languages"

_TRIGGER_NAMES = {
    "REGISTRY", "EXT_TO_LANG", "CODE_LANGUAGES", "_EXT_TO_LANG", "LANGUAGES",
    "_lang_table", "_lang_union", "_lang_flags", "_lang_merged", "language_set",
}


def _module_name(path: Path) -> str:
    rel = path.relative_to(_REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_import_module(current: str, node: ast.ImportFrom, is_init: bool) -> str:
    if not node.level:
        return node.module or ""
    own_package = current if is_init else (
        current.rsplit(".", 1)[0] if "." in current else ""
    )
    pkg_parts = own_package.split(".") if own_package else []
    strip = node.level - 1
    kept = pkg_parts[: len(pkg_parts) - strip] if strip <= len(pkg_parts) else []
    tail = node.module or ""
    return ".".join(kept + tail.split(".")) if tail else ".".join(kept)


class _ModuleInfo:
    def __init__(self, path: Path):
        self.path = path
        self.name = _module_name(path)
        self.tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.is_init = path.name == "__init__.py"
        self.assigns: dict[str, ast.expr] = {}
        self.funcdefs: set[str] = set()
        self.imports: list[tuple[str, str, str]] = []  # (source_module, orig_name, local_name)
        self.has_refresh_func = False
        self.registers_hook = False
        self.refresh_func_node: ast.AST | None = None
        self._scan()

    def _scan(self) -> None:
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcdefs.add(node.name)
                if node.name == "_refresh_from_registry":
                    self.has_refresh_func = True
                    self.refresh_func_node = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.assigns[target.id] = node.value
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.value is not None:
                    self.assigns[node.target.id] = node.value
            elif isinstance(node, ast.ImportFrom):
                src = _resolve_import_module(self.name, node, self.is_init)
                for alias in node.names:
                    local = alias.asname or alias.name
                    self.imports.append((src, alias.name, local))
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id == "register_refresh":
                    self.registers_hook = True


def _all_chonks_files() -> list[Path]:
    return sorted(p for p in _CHONKS_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


_MODULE_INFOS: dict[str, _ModuleInfo] = {
    info.name: info for info in (_ModuleInfo(p) for p in _all_chonks_files())
}


def _mentions_name(expr: ast.expr, names: set[str]) -> bool:
    return any(isinstance(node, ast.Name) and node.id in names for node in ast.walk(expr))


def _compute_frozen_names() -> dict[str, set[str]]:
    """For every module, the module-level names whose value is derived from
    the registry: a direct assignment mentioning one of the trigger names (or
    a name already known derived in the same module), or a `from ... import`
    of a name that another module's pass already marked derived, or that is
    itself one of the trigger names."""
    frozen: dict[str, set[str]] = {m: set() for m in _MODULE_INFOS}
    changed = True
    while changed:
        changed = False
        for name, info in _MODULE_INFOS.items():
            local_known = _TRIGGER_NAMES | frozen[name]
            for target, value in info.assigns.items():
                if target in frozen[name]:
                    continue
                if _mentions_name(value, local_known):
                    frozen[name].add(target)
                    changed = True
            for src, orig, local in info.imports:
                if local in frozen[name]:
                    continue
                src_info = _MODULE_INFOS.get(src)
                if src_info is not None and orig in src_info.funcdefs:
                    continue  # a live accessor function, not a frozen value
                is_derived = orig in _TRIGGER_NAMES or (
                    src_info is not None and orig in frozen.get(src, set())
                )
                if is_derived:
                    frozen[name].add(local)
                    changed = True
    return frozen


_FROZEN_NAMES = _compute_frozen_names()

_MODULES_NEEDING_HOOKS = sorted(
    m for m, names in _FROZEN_NAMES.items() if names and m != _SOURCE_MODULE
)

_MODULES_WITH_HOOKS = sorted(
    m for m, info in _MODULE_INFOS.items() if info.has_refresh_func and m != _SOURCE_MODULE
)


def _rebind_targets(func_node: ast.AST) -> tuple[set[str], set[str]]:
    globals_declared: set[str] = set()
    assigned: set[str] = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Global):
            globals_declared.update(node.names)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                assigned.add(node.target.id)
    return globals_declared, assigned


@pytest.mark.parametrize("module_name", _MODULES_NEEDING_HOOKS)
def test_module_with_frozen_registry_derivation_has_a_complete_hook(module_name):
    info = _MODULE_INFOS[module_name]
    names = _FROZEN_NAMES[module_name]
    assert info.has_refresh_func, (
        f"{module_name} derives {sorted(names)} from the language registry at "
        "import time and defines no _refresh_from_registry()"
    )
    assert info.registers_hook, (
        f"{module_name} defines _refresh_from_registry but never calls "
        "register_refresh(_refresh_from_registry) at module level"
    )
    globals_declared, assigned = _rebind_targets(info.refresh_func_node)
    missing_global = names - globals_declared
    missing_assign = names - assigned
    assert not missing_global, (
        f"{module_name}._refresh_from_registry does not declare global for "
        f"{sorted(missing_global)}"
    )
    assert not missing_assign, (
        f"{module_name}._refresh_from_registry does not rebind {sorted(missing_assign)}"
    )


@pytest.mark.parametrize("module_name", _MODULES_WITH_HOOKS)
def test_module_with_a_hook_has_a_frozen_registry_derivation(module_name):
    assert _FROZEN_NAMES.get(module_name), (
        f"{module_name} defines _refresh_from_registry but the AST walk found "
        "no registry-derived module-level name for it to rebind"
    )


# ---------------------------------------------------------------------------
# Behavioural: the hooks actually track a language added after import.
# ---------------------------------------------------------------------------

_ZIG_SPEC = LanguageSpec(
    name="zig",
    grammar="zig",
    extensions=frozenset({".zig"}),
    boundary_nodes=frozenset({"fn_decl"}),
    header_exts=frozenset({"zigh"}),
    kind_labels={"fn_decl": "function"},
    line_comment_prefixes=("//",),
    c_macro_self_heal=True,
    def_signature_keyword="fn",
    refs_spec={"call_expression": Field(field="function", bucket="calls")},
    literals=LiteralSpec(leaf_types=("string_literal",)),
)


def _rebind_registry(registry: "languages.Registry") -> None:
    languages.REGISTRY = registry
    languages.EXT_TO_LANG = registry.ext_to_lang
    languages.CODE_LANGUAGES = registry.code_languages


@pytest.fixture
def synthetic_zig_registry():
    original_specs = languages.REGISTRY.specs
    _rebind_registry(languages.Registry(original_specs + (_ZIG_SPEC,)))
    refresh.run_refreshes()
    try:
        yield
    finally:
        _rebind_registry(languages.Registry(original_specs))
        refresh.run_refreshes()


def test_all_frozen_derivations_track_a_late_loaded_language(synthetic_zig_registry):
    import chonks.index.graph.call_resolve as call_resolve
    import chonks.index.graph.hierarchy as hierarchy
    import chonks.index.macro_heal as macro_heal
    import chonks.index.pipeline as pipeline
    import chonks.index.refs_extract as refs_extract
    import chonks.index.segment as segment
    import chonks.ops.diagnostics as diagnostics
    import chonks.retrieval.repomap as repomap
    import chonks.serve.app as app
    import chonks.storage.store as store

    assert segment._lang_for_path(Path("a.zig")) == "zig"
    assert segment._EXT_TO_LANG.get(".zig") == "zig"
    assert "zig" in segment._BOUNDARY_NODES and "fn_decl" in segment._BOUNDARY_NODES["zig"]
    assert "zig" in segment._CONTAINER_NODES
    assert "zig" in segment.CODE_LANGUAGES
    assert "zig" in segment._SALVAGE_ELIGIBLE_BOUNDARY_TYPES
    assert segment._COMMENT_PREFIXES.get("zig") == ("//",)
    assert "zig" in segment._MACRO_LANGS

    assert "zig" in refs_extract.LANG_REFS_SPECS
    assert "zig" in refs_extract._REFS_LANGS
    assert "zig" in refs_extract._LITERAL_SPECS

    assert "zig" in macro_heal._MACRO_LANGS

    assert "zigh" in hierarchy.HEADER_EXTS

    assert call_resolve._DEF_SIGNATURE_KEYWORD.get("zig") == "fn"

    assert ".zig" in pipeline._EXT_TO_LANG

    assert "zig" in store.CODE_LANGUAGES
    assert "zig" in diagnostics.CODE_LANGUAGES
    assert "zig" in app.CODE_LANGUAGES

    assert repomap._NODE_TYPE_PREFIX.get("fn_decl") == "function"

    # One pass of run_refreshes() must be enough: app's CODE_LANGUAGES is read
    # from segment, and segment's _MACRO_LANGS is read from macro_heal, so
    # each must already agree with its upstream after a single run.
    assert app.CODE_LANGUAGES == segment.CODE_LANGUAGES
    assert segment._MACRO_LANGS == macro_heal._MACRO_LANGS
