"""repo-map text rendering: directory overview, per-file symbol listing
ranked by PageRank, and truncation-marker summarization."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import networkx as nx

from chonks.core.refresh import register_refresh
from chonks.languages import GENERIC_KIND_LABELS, merged as _lang_merged

if TYPE_CHECKING:
    from chonks.storage.store import Store

_NODE_TYPE_PREFIX = _lang_merged("kind_labels", GENERIC_KIND_LABELS)


def _refresh_from_registry() -> None:
    global _NODE_TYPE_PREFIX
    _NODE_TYPE_PREFIX = _lang_merged("kind_labels", GENERIC_KIND_LABELS)


register_refresh(_refresh_from_registry)

_CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# Directory overview (graph v2)
# ---------------------------------------------------------------------------

_DIR_OVERVIEW_MAX_LINES = 20
_DIR_OVERVIEW_MAX_DEPTH = 2


def _dir_overview(store: "Store", path_prefix: str | None, budget_chars: int | None) -> str:
    """Directory overview from graph_nodes; returns "" on a DB that
    predates rebuild_hierarchy() (no dir rows), or if nothing fits
    even at the top level."""
    dirs = store.get_graph_dirs()
    if not dirs:
        return ""

    direct_counts = store.get_graph_dir_file_counts()  # {dir_id: direct file count}
    id_to_parent = {d["id"]: d["parent_id"] for d in dirs}

    # Roll each dir's direct file count up its parent_id chain into a
    # subtree total. Dir count is small, so a simple per-dir walk is fine.
    subtree: dict[str, int] = defaultdict(int)
    for dir_id, n in direct_counts.items():
        cur: str | None = dir_id
        while cur is not None:
            subtree[cur] += n
            cur = id_to_parent.get(cur)

    prefix = (path_prefix or "").rstrip("/") or None

    def _depth(path: str) -> int | None:
        """Levels below the scope root, or None if `path` isn't under scope."""
        if prefix is None:
            if path == ".":
                return 0
            return len(path.split("/"))
        if path == prefix:
            return 0
        if not path.startswith(prefix + "/"):
            return None
        rel = path[len(prefix) + 1:]
        return len(rel.split("/"))

    candidates: list[tuple[int, str, int]] = []
    for d in dirs:
        path = d["path"]
        if prefix is not None and not (path == prefix or path.startswith(prefix + "/")):
            continue
        depth = _depth(path)
        if depth is None or depth == 0 or depth > _DIR_OVERVIEW_MAX_DEPTH:
            continue
        n = subtree.get(d["id"], 0)
        if n == 0:
            continue
        candidates.append((depth, path, n))

    if not candidates:
        return ""

    candidates.sort(key=lambda t: (-t[2], t[1]))

    def _render(cands: list[tuple[int, str, int]]) -> str:
        total = len(cands)
        shown = cands[:_DIR_OVERVIEW_MAX_LINES]
        lines = [
            f"{'  ' * (depth - 1)}{path}/ ({n} file{'s' if n != 1 else ''})"
            for depth, path, n in shown
        ]
        if total > _DIR_OVERVIEW_MAX_LINES:
            lines.append(f"... +{total - _DIR_OVERVIEW_MAX_LINES} more directories")
        return "# Directory overview\n" + "\n".join(lines) + "\n\n"

    text = _render(candidates)

    if budget_chars is not None:
        limit = budget_chars // 4
        if len(text) > limit:
            # Drop the deepest level first (level 2), keep only immediate
            # children of the scope root.
            shallow = [c for c in candidates if c[0] == 1]
            if not shallow:
                return ""
            text = _render(shallow)
            if len(text) > limit:
                return ""

    return text


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_map(
    chunks: list[dict],
    scores: dict[str, float],
    token_budget: int | None,
) -> str:
    """Formats chunks grouped by file, files ranked by aggregate PageRank;
    stops adding files once the token budget is reached."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_file[c["path"]].append(c)

    file_scores = {
        path: sum(scores.get(c["id"], 0.0) for c in file_chunks)
        for path, file_chunks in by_file.items()
    }

    budget_chars = token_budget * _CHARS_PER_TOKEN if token_budget else None
    char_count = 0
    blocks: list[str] = []
    included_paths: list[str] = []

    for file_path in sorted(by_file, key=lambda p: -file_scores[p]):
        file_chunks = sorted(by_file[file_path], key=lambda c: -scores.get(c["id"], 0.0))

        symbol_lines: list[str] = []
        for c in file_chunks:
            friendly = _NODE_TYPE_PREFIX.get(c["chunk_type"] or "")
            type_prefix = friendly + " " if friendly else ""
            symbol_lines.append(f"  {type_prefix}{c['name']}  (line {c['start_line']})")

        block = file_path + "\n" + "\n".join(symbol_lines)
        block_chars = len(block) + 2  # +2 for separator newline

        if budget_chars is not None and char_count + block_chars > budget_chars:
            if not blocks:
                blocks.append(block)  # always include at least one file
                included_paths.append(file_path)
            break

        blocks.append(block)
        included_paths.append(file_path)
        char_count += block_chars

    out = "\n\n".join(blocks)
    if len(blocks) < len(by_file):
        # The consumer is usually an LLM agent: without this marker it cannot
        # distinguish "the whole codebase" from "the top slice that fit".
        included = set(included_paths)
        omitted_paths = [p for p in by_file if p not in included]
        marker = (
            f"[truncated: showing {len(blocks)} of {len(by_file)} files "
            "by importance"
        )
        subtree_note = _summarize_omitted_subtrees(omitted_paths, list(by_file))
        if subtree_note:
            marker += f" — omitted subtrees: {subtree_note}"
        marker += " — narrow path_prefix or raise token_budget]"
        out += "\n\n" + marker
    return out


def _summarize_omitted_subtrees(
    omitted_paths: list[str], all_paths: list[str], top_n: int = 5,
) -> str:
    """Groups omitted paths by directory level below the map's scope (the
    common prefix of all_paths, not the literal first segment, or a scoped
    map's groups collapse into one bucket) for the truncation marker."""

    def _dir_segments(path: str) -> list[str]:
        return path.split("/")[:-1]

    scope = _dir_segments(all_paths[0]) if all_paths else []
    for p in all_paths[1:]:
        segs = _dir_segments(p)
        i = 0
        while i < len(scope) and i < len(segs) and scope[i] == segs[i]:
            i += 1
        scope = scope[:i]
        if not scope:
            break

    group_counts: Counter[str] = Counter()
    for path in omitted_paths:
        rel = path.split("/")[len(scope):]
        if len(rel) > 1:
            key = "/".join(scope + [rel[0]])
        else:
            key = "/".join(scope) or "."
        group_counts[key] += 1

    top_groups = sorted(group_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    return ", ".join(
        f"{name} ({n} file{'s' if n != 1 else ''})" for name, n in top_groups
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_repomap(
    store: "Store",
    path_prefix: str | None = None,
    query: str | None = None,
    token_budget: int | None = None,
) -> str:
    """Builds a repo-map string ranked by PageRank; symbols with no covering
    chunk fall back to their file's mean score, then a global mean. Falls
    back to chunk-name listing on a pre-symbol-index DB so the map is never empty."""
    if store.symbols_count() == 0:
        return _build_repomap_from_chunks(store, path_prefix, token_budget)

    symbols = store.get_all_symbols(path_prefix)
    if not symbols:
        scope = f"under '{path_prefix}'" if path_prefix else "in the index"
        return f"No named symbols found {scope}."

    from chonks.index.graph.pagerank import compute_pagerank_global
    pr = compute_pagerank_global(store)

    # Per-file mean of known (covered-by-a-chunk) symbol scores, used as the
    # fallback for symbols whose chunk_id is None or absent from PageRank.
    file_known: dict[str, list[float]] = defaultdict(list)
    all_known: list[float] = []
    for s in symbols:
        cid = s.get("chunk_id")
        if cid is not None and cid in pr:
            file_known[s["path"]].append(pr[cid])
            all_known.append(pr[cid])
    global_fallback = (sum(all_known) / len(all_known)) if all_known else 1.0
    file_fallback = {p: sum(v) / len(v) for p, v in file_known.items()}

    # Adapt symbol rows to the shape _format_map consumes (kind -> chunk_type),
    # keying scores by a per-symbol synthetic id so two symbols sharing one
    # chunk_id are ranked and displayed independently.
    display: list[dict] = []
    scores: dict[int, float] = {}
    for i, s in enumerate(symbols):
        cid = s.get("chunk_id")
        if cid is not None and cid in pr:
            score = pr[cid]
        else:
            score = file_fallback.get(s["path"], global_fallback)
        scores[i] = score
        display.append({
            "id": i,
            "path": s["path"],
            "name": s["name"],
            "chunk_type": s["kind"],
            "start_line": s["start_line"],
        })

    budget_chars = token_budget * _CHARS_PER_TOKEN if token_budget else None
    overview = _dir_overview(store, path_prefix, budget_chars)

    map_token_budget = token_budget
    if token_budget is not None and overview:
        map_token_budget = max(100, token_budget - len(overview) // _CHARS_PER_TOKEN)

    map_text = _format_map(display, scores, map_token_budget)
    return overview + map_text if overview else map_text


def _build_repomap_from_chunks(
    store: "Store",
    path_prefix: str | None,
    token_budget: int | None,
) -> str:
    """Legacy chunk-name listing, the fallback for DBs older than the symbol
    table (symbols_count()==0). Uses the content-free chunk fetch since the
    output is names and line numbers only, never chunk bodies."""
    chunks = store.get_named_chunks_meta(path_prefix)
    if not chunks:
        scope = f"under '{path_prefix}'" if path_prefix else "in the index"
        return f"No named symbols found {scope}."

    G = nx.DiGraph()
    for c in chunks:
        G.add_node(c["id"])

    chunk_id_set = {c["id"] for c in chunks}
    if path_prefix is None:
        edges = store.get_all_refs()
    else:
        edges = store.get_refs_for_chunks([c["id"] for c in chunks])

    for from_id, to_id in edges:
        if from_id in chunk_id_set and to_id in chunk_id_set:
            G.add_edge(from_id, to_id)

    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100)
    except nx.exception.PowerIterationFailedConvergence:
        # Fall back to uniform scores, map still useful, just unranked.
        pr = {c["id"]: 1.0 for c in chunks}

    return _format_map(chunks, pr, token_budget)
