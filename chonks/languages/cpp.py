"""C++ language plugin data."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._ast import name_last_or_text, name_strip_angle_brackets, name_text, terminal_identifier
from ._naming import cpp_function_declarator_name, namespace_name, tag_specifier_name, template_inner_name
from .spec import (
    ChildSpec, Children, Field, LanguageSpec, LiteralSpec, NameRule, NestedSpec, NOT_HANDLED,
)

if TYPE_CHECKING:
    from tree_sitter import Node


def qualified_receiver(call_node: Node, callee: Node, src: bytes) -> "str | None | object":
    """Immediate qualifier of 'Ns::Cls::method' -> 'Cls'. Right-recursive
    grammar: follow 'name' down to the deepest qualified_identifier, then
    read its 'scope'."""
    if callee.type != "qualified_identifier":
        return NOT_HANDLED
    cur = callee
    while True:
        nxt = cur.child_by_field_name("name")
        if nxt is not None and nxt.type == "qualified_identifier":
            cur = nxt
        else:
            break
    return terminal_identifier(cur.child_by_field_name("scope"), src, CPP.identifier_leaf_types)


def classify_param(part: str) -> "str | object":
    if part == "..." or part.endswith("..."):
        return "variadic"
    return NOT_HANDLED


INCLUDE = Children((
    ChildSpec(("string_literal",), "imports",
              nested=NestedSpec(("string_content",), name_text)),
    ChildSpec(("system_lib_string",), "imports", name_fn=name_strip_angle_brackets),
))
BASES = Children((
    ChildSpec(("base_class_clause",), "inherits",
              nested=NestedSpec(("type_identifier", "qualified_identifier"), name_last_or_text)),
))

CPP = LanguageSpec(
    name="cpp",
    grammar="cpp",
    version="2",
    display_name="C++",
    # .h stays on "cpp" not "c": remapping would churn boundaries across
    # every existing C++ corpus (see chonks/index/pipeline.py's content-hash skip).
    # CUDA/Obj-C++/Metal ride the cpp grammar best-effort; unmapped, whole
    # GPU/macOS backends silently land in unsupported_ext_skipped.
    extensions=frozenset({
        ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".inl", ".cu", ".cuh", ".mm", ".metal",
    }),
    boundary_nodes=frozenset({
        "function_definition", "class_specifier", "struct_specifier", "template_declaration",
    }),
    container_nodes=frozenset({"namespace_definition", "declaration_list"}),
    salvage_nodes=frozenset({"function_definition"}),
    name_rules=(
        NameRule(("function_definition",), cpp_function_declarator_name),
        NameRule(("class_specifier", "struct_specifier", "union_specifier", "enum_specifier"),
                 tag_specifier_name),
        NameRule(("namespace_definition",), namespace_name),
        NameRule(("template_declaration",), template_inner_name),
    ),
    kind_labels={
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "enum_specifier": "enum",
        "namespace_definition": "namespace",
        "template_declaration": "template",
    },
    c_macro_self_heal=True,
    refs_spec={
        "preproc_include": INCLUDE,
        "call_expression": Field("function", "calls"),
        "class_specifier": BASES,
        "struct_specifier": BASES,
    },
    chain_base_fields={"field_expression": "argument"},
    identifier_leaf_types=frozenset({
        "identifier", "field_identifier", "type_identifier", "namespace_identifier",
    }),
    call_receiver=qualified_receiver,
    variadic_arg_types=frozenset({"parameter_pack_expansion"}),
    class_like_chunk_types=frozenset({"class_specifier", "struct_specifier"}),
    classify_param=classify_param,
    empty_param_spellings=frozenset({"void"}),
    literals=LiteralSpec(leaf_types=("string_literal", "raw_string_literal"),
                         concat_types=("concatenated_string",)),
    header_exts=frozenset({"h", "hh", "hpp", "hxx"}),
    impl_exts=frozenset({"c", "cc", "cpp", "cxx", "m", "mm"}),
)

LANGUAGES = (CPP,)
