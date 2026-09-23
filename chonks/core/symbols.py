"""Symbol-index vocabulary shared by the chunker, the store and the query layer."""

# Kind for a declaration that defines nothing (`class X;`). Lookups fall
# back to these only when a name has no definition; the graph never
# targets them.
FORWARD_DECLARATION = "forward_declaration"
