"""Python language plugin data."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._ast import name_text
from .spec import ChildSpec, Children, Field, FieldChildren, LanguageSpec, LiteralSpec, NestedSpec

if TYPE_CHECKING:
    from tree_sitter import Node


def is_python_docstring_position(n: Node) -> bool:
    """True when `n` is the first statement of a module/function/class
    body. tree-sitter-python gives docstrings no expression_statement
    wrapper, so position must be read off the parent chain instead."""
    parent = n.parent
    if parent is None or not parent.named_children or parent.named_children[0] != n:
        return False
    if parent.type == "module":
        return True
    return parent.type == "block" and parent.parent is not None and \
        parent.parent.type in ("function_definition", "class_definition")


IMPORT = Children((
    ChildSpec(("dotted_name",), "imports"),
    ChildSpec(("aliased_import",), "imports",
              nested=NestedSpec(("dotted_name",), name_text, break_inner=True)),
))

PYTHON = LanguageSpec(
    name="python",
    grammar="python",
    # .pyi is Python's cross-language interface layer; dropping it starves
    # xlang pairing of its highest-precision anchor.
    extensions=frozenset({".py", ".pyi"}),
    boundary_nodes=frozenset({"function_definition", "class_definition", "decorated_definition"}),
    salvage_nodes=frozenset({"function_definition"}),
    refs_spec={
        "import_statement": IMPORT,
        "import_from_statement": IMPORT,
        "call": Field("function", "calls"),
        "class_definition": FieldChildren("superclasses", ("identifier",), "inherits"),
    },
    line_comment_prefixes=("#",),
    block_comments=False,
    chain_base_fields={"attribute": "object"},
    identifier_leaf_types=frozenset({"identifier"}),
    variadic_arg_types=frozenset({"list_splat", "dictionary_splat"}),
    keyword_arg_types=frozenset({"keyword_argument"}),
    class_like_chunk_types=frozenset({"class_definition"}),
    def_signature_keyword=r"\bdef\s+",
    kind_labels={
        "function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "function",
    },
    literals=LiteralSpec(leaf_types=("string",), concat_types=("concatenated_string",),
                         plus_type="binary_operator", skip_fn=is_python_docstring_position),
)

LANGUAGES = (PYTHON,)
