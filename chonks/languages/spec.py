"""Refs-DSL dataclasses describing how to walk each language's tree-sitter grammar."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping

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


NOT_HANDLED = object()  # returned by a hook to make the engine use its generic code

NodePred = Callable[["Node", bytes], bool]


@dataclass(frozen=True)
class NameRule:
    """Node types and the function that names them. Rules are tried in
    order; the first non-None result wins."""
    types: tuple[str, ...]
    fn: NameFn


@dataclass(frozen=True)
class LanguageSpec:
    # identity
    name: str  # never rename
    grammar: str
    # kw_only: a defaulted field can't sit ahead of the required extensions/
    # boundary_nodes fields in a dataclass's positional __init__ otherwise.
    version: str = field(default="1", kw_only=True)
    extensions: frozenset[str]  # lower-case, leading dot; unique across registry
    # tier 1: segmentation
    boundary_nodes: frozenset[str]
    container_nodes: frozenset[str] = frozenset()
    salvage_nodes: frozenset[str] = frozenset()
    salvage_extra: NodePred | None = None  # checked before salvage_nodes
    boundary_filters: Mapping[str, NodePred] = field(default_factory=dict)
    forward_declarations: Mapping[str, NodePred] = field(default_factory=dict)
    extra_boundary: NodePred | None = None
    synthetic_chunk_type: NameFn | None = None
    name_rules: tuple[NameRule, ...] = ()
    kind_labels: Mapping[str, str] = field(default_factory=dict)  # must cover boundary_nodes
    line_comment_prefixes: tuple[str, ...] | None = None  # None = C-style default
    block_comments: bool = True
    module_residue_split: NodePred | None = None
    c_macro_self_heal: bool = False
    # tier 2: typed refs, receivers, arity
    refs_spec: Mapping[str, "NodeRule"] | None = None
    chain_base_fields: Mapping[str, str] = field(default_factory=dict)
    identifier_leaf_types: frozenset[str] = frozenset()
    call_receiver: "Callable[[Node, Node, bytes], str | None | object] | None" = None
    variadic_arg_types: frozenset[str] = frozenset()
    keyword_arg_types: frozenset[str] = frozenset()
    class_like_chunk_types: frozenset[str] = frozenset()
    # Chunks that only scope other code: named, but never an edge target.
    scope_chunk_types: frozenset[str] = frozenset()
    def_signature_keyword: str | None = None
    classify_param: "Callable[[str], str | object] | None" = None
    strip_implicit_params: "Callable[[list[str]], list[str]] | None" = None
    empty_param_spellings: frozenset[str] = frozenset()
    # tier 2: literals
    literals: "LiteralSpec | None" = None
    literal_wrapper: "Callable[[str], tuple[str, str] | object] | None" = None
    # called with the literal's letter prefix; True = raw mode
    raw_string_prefix: "Callable[[str], bool] | None" = None
    # tier 3: file level, outside chunking
    file_metadata: "Callable[[str], dict | None] | None" = None
    header_exts: frozenset[str] = frozenset()  # dotless lower-case, validated separately
    impl_exts: frozenset[str] = frozenset()  # may list exts not in extensions
    display_name: str = ""
