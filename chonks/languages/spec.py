"""Refs-DSL dataclasses describing how to walk each language's tree-sitter grammar."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping

from ._ast import name_call_target, name_text

if TYPE_CHECKING:
    import re

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


@dataclass(frozen=True)
class CallSiteSpec:
    """Node facts for a call's receiver chain, the declarations that type its
    head, and the class the call sits in."""
    # receiver chains: node type -> field names
    member_access: Mapping[str, tuple[str, str, str]]  # (object, operator, member)
    scope_access: Mapping[str, tuple[str, str]]        # (scope, name)
    scope_operator: str
    calls: Mapping[str, str]                           # callee field
    type_argument_wrappers: Mapping[str, tuple[str, str]]  # (bare name, type arguments)
    parenthesized: frozenset[str]
    subscripts: Mapping[str, str]                      # object field
    dereferences: Mapping[str, tuple[str, str, str]]   # (operand, operator field, operator)
    this_expressions: frozenset[str]
    names: frozenset[str]
    # expressions that state their own type
    casts: Mapping[str, str]                           # type field
    allocations: Mapping[str, str]                     # type field; result is `<type><allocation_suffix>`
    allocation_suffix: str
    typed_literals: Mapping[str, str]                  # type field
    # callee names whose one type argument is the result type
    type_argument_calls: "re.Pattern[str]"
    # declarations
    functions: frozenset[str]                          # a declaration lookup stops here
    parameter_owners: Mapping[str, "str | None"]       # field leading to the parameter list; None: on the node
    parameters_field: str
    parameters: frozenset[str]
    blocks: frozenset[str]
    transparent_blocks: frozenset[str]                 # their declarations count in the enclosing block
    declarations: frozenset[str]
    declaration_scopes: Mapping[str, tuple[str, ...]]  # statement -> fields whose declarations it scopes
    range_loops: Mapping[str, str]                     # the field that does not see the loop variable
    declarators: Mapping[str, str]                     # wrapper -> suffix it adds to the type
    declarator_names: frozenset[str]
    binding_declarators: frozenset[str]                # declare several untyped names
    inferred_types: frozenset[str]
    type_qualifiers: Mapping[str, frozenset[str]]      # node type -> the words that belong to the type
    type_field: str
    declarator_field: str
    value_field: str
    # calling class
    classes: frozenset[str]
    namespaces: frozenset[str]
    name_field: str


NOT_HANDLED = object()  # returned by a hook to make the engine use its generic code

NodePred = Callable[["Node", bytes], bool]


@dataclass(frozen=True)
class NameRule:
    """Node types and the function that names them. Rules are tried in
    order; the first non-None result wins."""
    types: tuple[str, ...]
    fn: NameFn


@dataclass(frozen=True)
class Member:
    """One class member a declaration names. `scope` is the class qualifier
    written on an out-of-class definition (`Foo::bar` -> ("Foo",))."""
    name: str
    kind: str
    line: int
    type_text: str | None = None
    arity: "tuple[int, int, bool] | None" = None  # (required, required + defaulted, variadic)
    flags: tuple[str, ...] = ()
    scope: tuple[str, ...] = ()


MemberFn = Callable[["Node", bytes, str], "list[Member]"]  # (node, src, class name)


@dataclass(frozen=True)
class TypeRef:
    """A written type as the class it names: `const ns::Vector<Ref<Foo>> *` ->
    name `ns::Vector`, args ("Ref<Foo>",), pointers 1."""
    name: str
    args: tuple[str, ...] = ()
    pointers: int = 0


@dataclass(frozen=True)
class ClassModelSpec:
    """Node facts for the class model: the members and bases of each class."""
    namespaces: Mapping[str, NameFn]  # nodes that open a named scope
    classes: Mapping[str, NameFn]  # class nodes, named without template arguments
    members: Mapping[str, MemberFn]  # member declarations in a class body
    # Function definitions: one outside a class body may define a member
    # (`scope` set); nothing inside a function is recorded.
    functions: Mapping[str, MemberFn]
    bases: Children  # applied to a class node, bucket "bases"
    # Using-directives: the namespace one names (`using namespace ns;`), else None.
    using_namespaces: Mapping[str, NameFn] = field(default_factory=dict)
    transparent: frozenset[str] = frozenset()  # read through in a class body
    body_field: str = "body"
    separator: str = "::"
    # Resolving calls through the model: the class a written type names, the
    # access that goes through a pointer, and the member that overloads an
    # access or a receiver step (`->` -> `operator->`, `[]` -> `operator[]`).
    type_ref: "Callable[[str], TypeRef | None] | None" = None
    pointer_access: str | None = None
    operator_members: Mapping[str, str] = field(default_factory=dict)


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
    call_sites: "CallSiteSpec | None" = None
    variadic_arg_types: frozenset[str] = frozenset()
    keyword_arg_types: frozenset[str] = frozenset()
    class_like_chunk_types: frozenset[str] = frozenset()
    # Chunks that only scope other code: named, but never an edge target.
    scope_chunk_types: frozenset[str] = frozenset()
    # Other languages whose definitions this one's calls, imports and bases
    # can name; its own always can.
    typed_ref_languages: frozenset[str] = frozenset()
    def_signature_keyword: str | None = None
    classify_param: "Callable[[str], str | object] | None" = None
    strip_implicit_params: "Callable[[list[str]], list[str]] | None" = None
    empty_param_spellings: frozenset[str] = frozenset()
    # For methods defined outside their class: whether the `name(` at this
    # offset of a class chunk declares a member, so its defaults count.
    member_declaration_at: "Callable[[str, int], bool] | None" = None
    class_model: "ClassModelSpec | None" = None
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
