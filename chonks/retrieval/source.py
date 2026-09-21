"""Source text of a definition, stitched from its chunks."""


def _canonical_chunk_content(chunk: dict) -> str:
    """Strips the fallback-split overlap; without this, stitching chunks together
    duplicates lines at every seam."""
    content = chunk.get("content") or ""
    lines = content.splitlines(keepends=True)
    canonical_count = chunk["end_line"] - chunk["start_line"] + 1
    overlap = len(lines) - canonical_count
    if overlap > 0:
        lines = lines[overlap:]
    return "".join(lines)


def _attach_definition_source(store, definitions: list[dict], max_chars: int) -> None:
    """Mutates `definitions` in place, adding "source". A definition split across
    multiple chunks gets continuations stitched in; a missing continuation is marked
    with an explicit boundary rather than silently truncated."""
    chunk_ids = [d["chunk_id"] for d in definitions if d.get("chunk_id")]
    if not chunk_ids:
        return
    chunks_by_id = {c["id"]: c for c in store.get_chunks_by_ids(chunk_ids)}
    per_def_budget = max_chars if len(definitions) <= 1 else max(max_chars // len(definitions), 500)
    for d in definitions:
        chunk = chunks_by_id.get(d.get("chunk_id"))
        if not chunk:
            continue
        content = chunk.get("content") or ""
        chunk_end, def_end = chunk.get("end_line"), d.get("end_line")
        if chunk_end is not None and def_end is not None and def_end > chunk_end:
            covered_to = chunk_end
            for cont in store.get_chunks_by_path_line_range(d["path"], chunk_end, def_end):
                if cont["start_line"] > covered_to + 1:
                    break
                if content and not content.endswith("\n"):
                    content += "\n"
                content += _canonical_chunk_content(cont)
                covered_to = max(covered_to, cont["end_line"])
            if covered_to < def_end:
                content += (
                    f"\n… [chunk boundary at line {covered_to} — definition continues to "
                    f"{def_end}; Read {d['path']}:{covered_to + 1}-{def_end}]"
                )
        if len(content) > per_def_budget:
            d["source"] = (
                content[:per_def_budget]
                + f"\n… [truncated at {per_def_budget} chars — Read "
                  f"{d['path']}:{d['start_line']}-{d['end_line']} for the rest]"
            )
        else:
            d["source"] = content
