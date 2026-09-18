"""Refs-DSL dataclasses describing how to walk each language's tree-sitter grammar."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ._ast import name_call_target, name_text

if TYPE_CHECKING:
    from tree_sitter import Node


NameFn = Callable[["Node", bytes], "str | None"]


@dataclass(frozen=True)
class NestedSpec:
    types: tuple[str, ...]
    name_fn: NameFn
    break_inner: bool = False


@dataclass(frozen=True)
class ChildSpec:
    types: tuple[str, ...]
    bucket: str
    name_fn: NameFn = name_text
    nested: "NestedSpec | None" = None
    break_outer: bool = False


@dataclass(frozen=True)
class Field:
    field: str
    bucket: str
    name_fn: NameFn = name_call_target


@dataclass(frozen=True)
class FieldChildren:
    field: str
    types: tuple[str, ...]
    bucket: str
    name_fn: NameFn = name_text


@dataclass(frozen=True)
class Children:
    specs: tuple[ChildSpec, ...]


NodeRule = "Field | FieldChildren | Children"


@dataclass(frozen=True)
class LiteralSpec:
    leaf_types: tuple[str, ...]      # node types that are themselves one string literal
    concat_types: tuple[str, ...] = ()  # adjacent-juxtaposition container types;
                                         # grammar guarantees pure leaf children, no purity check
    plus_type: str | None = None     # binary +/concat node type; only joined if pure literals
    plus_op: str = "+"               # the concat operator token inside plus_type
    skip_fn: "Callable[[Node], bool] | None" = None  # leaf-level exclusion (e.g. docstrings)
