"""C++ language plugin data."""
from __future__ import annotations

from ._ast import name_last_or_text, name_strip_angle_brackets, name_text
from .spec import ChildSpec, Children, Field, LanguageSpec, LiteralSpec, NestedSpec

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
    display_name="C++",
    # .h stays on "cpp" not "c": remapping would churn boundaries across
    # every existing C++ corpus (see chunker.py's content-hash skip).
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
    variadic_arg_types=frozenset({"parameter_pack_expansion"}),
    class_like_chunk_types=frozenset({"class_specifier", "struct_specifier"}),
    literals=LiteralSpec(leaf_types=("string_literal", "raw_string_literal"),
                         concat_types=("concatenated_string",)),
    header_exts=frozenset({"h", "hh", "hpp", "hxx"}),
    impl_exts=frozenset({"c", "cc", "cpp", "cxx", "m", "mm"}),
)

LANGUAGES = (CPP,)
