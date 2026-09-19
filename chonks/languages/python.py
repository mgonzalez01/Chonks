"""Python language plugin data."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._ast import name_text
from ._naming import decorated_name, name_field
from .spec import (
    ChildSpec, Children, Field, FieldChildren, LanguageSpec, LiteralSpec, NameRule, NestedSpec,
    NOT_HANDLED,
)

if TYPE_CHECKING:
    from tree_sitter import Node


_PYTHON_SELF_NAMES = {"self", "cls", "__self", "__cls"}


def is_main_block(node: Node, src: bytes) -> bool:
    """True for a Python `if __name__ == "__main__":` top-level block."""
    if node.type != "if_statement":
        return False
    cond = node.child_by_field_name("condition")
    if cond is None:
        return False
    text = src[cond.start_byte:cond.end_byte].decode(errors="replace")
    return "__name__" in text and "__main__" in text


def strip_implicit_params(parts: list[str]) -> list[str]:
    """Drops an implicit 'self'/'cls'/'__self' first param: a bound call
    site never spells it out, so counting it undercounts every call by one."""
    if not parts:
        return parts
    bare = parts[0].split(":", 1)[0].strip()
    if bare in _PYTHON_SELF_NAMES:
        return parts[1:]
    return parts


def classify_param(part: str) -> "str | object":
    if part == "/":
        return "marker"
    if part.startswith("*"):
        return "variadic"
    return NOT_HANDLED


def file_metadata(stored_path: str) -> dict | None:
    """File-level context not derivable from chunk content alone (only Python
    module dotted-path today). Open JSON column, so new keys need no migration."""
    if stored_path.endswith(".py"):
        return {"module": stored_path[:-3].replace("/", ".")}
    return None


def is_raw_prefix(prefix: str) -> bool:
    return "r" in prefix.lower()


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
    name_rules=(
        NameRule(("function_definition", "async_function_def", "class_definition"), name_field),
        NameRule(("decorated_definition",), decorated_name),
    ),
    module_residue_split=is_main_block,
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
    classify_param=classify_param,
    strip_implicit_params=strip_implicit_params,
    kind_labels={
        "function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "function",
    },
    literals=LiteralSpec(leaf_types=("string",), concat_types=("concatenated_string",),
                         plus_type="binary_operator", skip_fn=is_python_docstring_position),
    raw_string_prefix=is_raw_prefix,
    file_metadata=file_metadata,
    display_name="Python",
)

LANGUAGES = (PYTHON,)
