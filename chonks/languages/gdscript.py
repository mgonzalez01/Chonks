"""GDScript language plugin data."""
from __future__ import annotations

from ._ast import name_last_or_text
from .spec import ChildSpec, Children, LanguageSpec, LiteralSpec

CALL = Children((
    ChildSpec(("identifier",), "calls", break_outer=True),
))

GDSCRIPT = LanguageSpec(
    name="gdscript",
    grammar="gdscript",
    extensions=frozenset({".gd"}),
    boundary_nodes=frozenset({"function_definition", "class_definition"}),
    salvage_nodes=frozenset({"function_definition"}),
    refs_spec={
        "call": CALL,
        "attribute_call": CALL,
        "extends_statement": Children((
            ChildSpec(("type",), "inherits", name_fn=name_last_or_text),
        )),
    },
    identifier_leaf_types=frozenset({"identifier"}),
    # BUG preserved: '#' comments fall to the C-style default prefixes.
    # A fix moves chunk boundaries and needs a CHUNKER_VERSION bump.
    line_comment_prefixes=None,
    def_signature_keyword=r"\bfunc\s+",
    class_like_chunk_types=frozenset({"class_definition"}),
    kind_labels={"function_definition": "function", "class_definition": "class"},
    literals=LiteralSpec(leaf_types=("string",)),
)

LANGUAGES = (GDSCRIPT,)
