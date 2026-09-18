"""Language plugin registry: discovers chonks/languages/*.py modules and validates their specs into shared views."""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .spec import Children, LanguageSpec

# no language owns this node type
GENERIC_KIND_LABELS = {"function_declarator": "function"}

_REFS_BUCKETS = frozenset({"calls", "imports", "inherits"})


class Registry:
    """Validated collection of LanguageSpec rows with derived, read-only views."""

    def __init__(self, specs: tuple[LanguageSpec, ...]):
        self.specs = specs
        self.by_name: dict[str, LanguageSpec] = {}
        ext_to_lang: dict[str, str] = {}
        for spec in specs:
            if spec.name in self.by_name:
                raise ValueError(f"{spec.name}: duplicate spec name")
            self.by_name[spec.name] = spec
            for ext in spec.extensions:
                if not ext.startswith(".") or ext != ext.lower():
                    raise ValueError(
                        f"{spec.name}: extension {ext!r} must be lower-case with a leading dot")
                if ext in ext_to_lang:
                    raise ValueError(
                        f"{spec.name}: extension {ext!r} already owned by {ext_to_lang[ext]!r}")
                ext_to_lang[ext] = spec.name
            for attr in ("header_exts", "impl_exts"):
                for entry in getattr(spec, attr):
                    if entry.startswith(".") or entry != entry.lower():
                        raise ValueError(
                            f"{spec.name}: {attr} entry {entry!r} must be dotless lower-case")
            if spec.refs_spec is not None:
                for rule in spec.refs_spec.values():
                    children = rule.specs if isinstance(rule, Children) else (rule,)
                    bad = {child.bucket for child in children} - _REFS_BUCKETS
                    if bad:
                        raise ValueError(
                            f"{spec.name}: refs_spec bucket {sorted(bad)!r} "
                            f"not in {sorted(_REFS_BUCKETS)!r}")
            if not spec.class_like_chunk_types <= spec.boundary_nodes:
                raise ValueError(
                    f"{spec.name}: class_like_chunk_types must be a subset of boundary_nodes")

        self.ext_to_lang = MappingProxyType(ext_to_lang)
        self.code_languages = frozenset(ext_to_lang.values())

        # After all specs: a label can come from another spec or from
        # GENERIC_KIND_LABELS. Container nodes need no label.
        labels = self.merged("kind_labels", GENERIC_KIND_LABELS)
        for spec in specs:
            missing = spec.boundary_nodes - set(labels)
            if missing:
                raise ValueError(
                    f"{spec.name}: boundary_nodes {sorted(missing)!r} missing a kind_labels entry")

    def get(self, name: str) -> LanguageSpec:
        return self.by_name[name]

    def table(self, attr: str) -> Mapping[str, object]:
        return MappingProxyType({
            spec.name: value for spec in self.specs
            if (value := getattr(spec, attr)) is not None
        })

    def union(self, attr: str) -> frozenset[str]:
        result: frozenset[str] = frozenset()
        for spec in self.specs:
            result |= getattr(spec, attr)
        return result

    def merged(self, attr: str, base: "Mapping[str, str] | None" = None) -> Mapping[str, str]:
        result: dict[str, str] = dict(base) if base else {}
        for spec in self.specs:
            for key, value in getattr(spec, attr).items():
                if key in result and result[key] != value:
                    raise ValueError(
                        f"{spec.name}: {attr} key {key!r} conflicts "
                        f"({value!r} != {result[key]!r})")
                result[key] = value
        return MappingProxyType(result)

    def flags(self, attr: str) -> frozenset[str]:
        return frozenset(spec.name for spec in self.specs if getattr(spec, attr))


def _discover() -> tuple[LanguageSpec, ...]:
    names = sorted(
        name for _, name, _ in pkgutil.iter_modules(__path__)
        if not name.startswith("_") and name != "spec"
    )
    specs: list[LanguageSpec] = []
    for name in names:
        module = importlib.import_module(f"{__name__}.{name}")
        specs.extend(module.LANGUAGES)
    return tuple(specs)


REGISTRY = Registry(_discover())

EXT_TO_LANG = REGISTRY.ext_to_lang
CODE_LANGUAGES = REGISTRY.code_languages


def get(name: str) -> LanguageSpec:
    return REGISTRY.get(name)


def table(attr: str) -> Mapping[str, object]:
    return REGISTRY.table(attr)


def union(attr: str) -> frozenset[str]:
    return REGISTRY.union(attr)


def merged(attr: str, base: "Mapping[str, str] | None" = None) -> Mapping[str, str]:
    return REGISTRY.merged(attr, base)


def flags(attr: str) -> frozenset[str]:
    return REGISTRY.flags(attr)


def lang_for_path(path: Path) -> str | None:
    return EXT_TO_LANG.get(path.suffix.lower())
