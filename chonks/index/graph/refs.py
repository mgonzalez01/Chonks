"""chunk_refs graph construction: mentions/xlang/typed-edge extraction, the
receiver/arity call-site discriminator, and the incremental update path."""

from __future__ import annotations

import heapq
import logging
import math
import re
import time
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Iterator

from chonks.index.refs_extract import _call_entry_name
from chonks.index.graph.call_resolve import _call_entry_fields, _discriminate_definers

from chonks.core.edges import _MAX_CROSS_LANG_OCCURRENCES, _MIN_NAME_LEN

if TYPE_CHECKING:
    from chonks.store import Store

logger = logging.getLogger("repomap")

_WORD_RE = re.compile(r"\b[A-Za-z_]\w+\b")

# Below this batch fraction, update chunk_refs incrementally; above it, a
# full rebuild is cheaper than per-referencer bookkeeping.
_REFS_INCREMENTAL_MAX_FRACTION = 0.20

# Second gate on actual referencer fan-out, not just batch size: a small
# batch can still touch names common enough to fan out to most of the
# corpus, which costs more than a full rebuild.
_REFS_INCREMENTAL_MAX_REFERENCER_FRACTION = 0.25


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _name_keys(name: str) -> tuple[str, ...]:
    """Bare last-component alias for `name` ("Foo::bar" -> "bar"), so a
    definer indexed only under a qualified name stays reachable from a bare
    content token. Applies no cap itself; callers must aggregate-cap the fold."""
    if "::" not in name and "." not in name:
        return (name,)
    bare = re.split(r"::|\.", name)[-1]
    if not bare or bare == name or len(bare) < _MIN_NAME_LEN:
        return (name,)
    return (name, bare)


def _classify_mentions(
    referenced_by_chunk: dict[str, set[str]],
    name_to_ids: dict[str, list[str]],
    associated_top_frac: float,
) -> set[tuple[str, str]]:
    """PMI-ranks (chunk, referenced name) pairs and returns the top
    `associated_top_frac` fraction to relabel 'associated'. Self-only pairs
    (never emit an edge) are excluded from ranking so they can't skew PMI."""
    if associated_top_frac <= 0:
        return set()
    live: dict[str, set[str]] = {}
    for cid, refs in referenced_by_chunk.items():
        kept = {name for name in refs if any(t != cid for t in name_to_ids[name])}
        if kept:
            live[cid] = kept
    df: Counter[str] = Counter()
    for refs in live.values():
        df.update(refs)
    total = sum(len(refs) for refs in live.values())
    if total == 0:
        return set()
    k = int(associated_top_frac * total)
    if k <= 0:
        return set()

    def _pairs() -> Iterator[tuple[float, str, str]]:
        for cid, refs in live.items():
            n_a = len(refs)
            for name in refs:
                yield (-math.log2(total / (n_a * df[name])), name, cid)

    top = heapq.nsmallest(k, _pairs())
    return {(cid, name) for _neg_pmi, name, cid in top}


def _build_graph(
    chunks: list[dict], sym_map: dict[str, list[str]] | None = None,
    *, cap_mentions: bool = False, associated_top_frac: float = 0.0,
    extra_definers: list[dict] | None = None,
) -> dict[tuple[str, str], str]:
    """Builds {(from_id, to_id): edge_type} from mentions, xlang pairing, and
    typed AST facts; typed edges are added last so they supersede a mentions
    or xlang edge for the same pair. Bare-alias fold is capped all-or-nothing per key."""
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    name_to_langs: dict[str, set[str]] = defaultdict(set)
    id_to_lang: dict[str, str] = {}

    # Direct registrations: a name under itself. Never touched by the bare-alias fold below.
    for c in chunks:
        name = c["name"]
        if name and len(name) >= _MIN_NAME_LEN:
            cid = c["id"]
            name_to_ids[name].append(cid)
            lang = c.get("language") or ""
            if lang:
                name_to_langs[name].add(lang)
                id_to_lang[cid] = lang

    # Adds decoupled symbol-index targets so a reference to a folded/merged
    # method still creates an edge to its owning chunk; capped per name like
    # the cross-language path.
    if sym_map:
        for name, ids in sym_map.items():
            if len(name) < _MIN_NAME_LEN or len(ids) > _MAX_CROSS_LANG_OCCURRENCES:
                continue
            existing = set(name_to_ids[name])
            for cid in ids:
                if cid not in existing:
                    name_to_ids[name].append(cid)
                    existing.add(cid)

    # Bare-alias fold: collect alias ids per bare key without mutating
    # name_to_ids, then fold in only if the aggregate (direct + alias) stays
    # within the cap. All-or-nothing per key. See _name_keys.
    alias_ids: dict[str, list[str]] = defaultdict(list)
    alias_seen: dict[str, set[str]] = defaultdict(set)
    alias_langs: dict[str, set[str]] = defaultdict(set)

    def _add_alias(bare: str, cid: str, lang: str) -> None:
        if cid not in alias_seen[bare]:
            alias_seen[bare].add(cid)
            alias_ids[bare].append(cid)
        if lang:
            alias_langs[bare].add(lang)

    for c in chunks:
        name = c["name"]
        if name and len(name) >= _MIN_NAME_LEN:
            keys = _name_keys(name)
            if len(keys) > 1:
                _add_alias(keys[1], c["id"], c.get("language") or "")

    if sym_map:
        for name, ids in sym_map.items():
            if len(name) < _MIN_NAME_LEN or len(ids) > _MAX_CROSS_LANG_OCCURRENCES:
                continue
            keys = _name_keys(name)
            if len(keys) > 1:
                for cid in ids:
                    _add_alias(keys[1], cid, "")

    for bare, aids in alias_ids.items():
        direct = set(name_to_ids.get(bare, ()))
        if len(direct | set(aids)) > _MAX_CROSS_LANG_OCCURRENCES:
            continue
        existing = set(name_to_ids[bare])
        for cid in aids:
            if cid not in existing:
                name_to_ids[bare].append(cid)
                existing.add(cid)
        if alias_langs.get(bare):
            name_to_langs[bare] |= alias_langs[bare]

    name_set = set(name_to_ids)

    edges: dict[tuple[str, str], str] = {}

    # Two-sweep mentions pass: cap_mentions applies in sweep 1, so a
    # capped-out name never enters the PMI population _classify_mentions ranks.
    referenced_by_chunk: dict[str, set[str]] = {}
    for c in chunks:
        content = c["content"] or ""
        own_name = c["name"]
        cid = c["id"]
        refs: set[str] = set()
        for word in _WORD_RE.findall(content):
            if word in name_set and word != own_name:
                if cap_mentions and len(name_to_ids[word]) > _MAX_CROSS_LANG_OCCURRENCES:
                    continue
                refs.add(word)
        referenced_by_chunk[cid] = refs

    associated_pairs = _classify_mentions(referenced_by_chunk, name_to_ids, associated_top_frac)

    # sorted(refs) pins iteration order; the edges.get(key)=='associated'
    # guard makes the result order-independent anyway, since a later
    # 'mentions' write must never downgrade an already-'associated' pair.
    for cid, refs in referenced_by_chunk.items():
        for ref_name in sorted(refs):
            edge_type = "associated" if (cid, ref_name) in associated_pairs else "mentions"
            for target_id in name_to_ids[ref_name]:
                if target_id != cid:
                    key = (cid, target_id)
                    if edges.get(key) == "associated":
                        continue
                    edges[key] = edge_type

    # Cross-language bidirectional edges between same-name definitions.
    for name, langs in name_to_langs.items():
        if len(langs) < 2:
            continue
        ids = name_to_ids[name]
        if len(ids) > _MAX_CROSS_LANG_OCCURRENCES:
            continue
        for i, id_a in enumerate(ids):
            lang_a = id_to_lang.get(id_a)
            for id_b in ids[i + 1:]:
                lang_b = id_to_lang.get(id_b)
                if lang_a != lang_b:
                    edges[(id_a, id_b)] = "xlang"
                    edges[(id_b, id_a)] = "xlang"

    # Typed edges (calls/imports/inherits) are added last, superseding any
    # mentions/xlang edge for the same pair. The fan-out cap here mirrors the
    # mentions pass, so a typed fact is never dropped in favor of a noisier one.
    id_to_chunk = {c["id"]: c for c in chunks}
    for c in extra_definers or ():
        id_to_chunk.setdefault(c["id"], c)
    arity_cache: dict[tuple[str, str], "tuple[int, int, bool] | None"] = {}
    for c in chunks:
        md = c.get("metadata") or {}
        cid = c["id"]
        for edge_type in ("calls", "imports", "inherits"):
            for raw in md.get(edge_type) or ():
                if edge_type == "calls":
                    name, receiver, arity = _call_entry_fields(raw)
                else:
                    name, receiver, arity = raw, None, None
                ids = name_to_ids.get(name)
                if not ids or (cap_mentions and len(ids) > _MAX_CROSS_LANG_OCCURRENCES):
                    continue
                targets = (
                    _discriminate_definers(ids, name, receiver, arity, id_to_chunk, arity_cache)
                    if edge_type == "calls" else ids
                )
                for target_id in targets:
                    if target_id != cid:
                        edges[(cid, target_id)] = edge_type

    return edges


def build_refs(
    store: "Store",
    *,
    changed_ids: set[str] | list[str] | None = None,
    deleted_ids: set[str] | list[str] | None = None,
    deleted_names: set[str] | list[str] | None = None,
    cap_mentions: bool = False,
    associated_top_frac: float = 0.0,
) -> int:
    """Computes chunk_refs and keeps chunk_indegree exactly in sync (get_hubs
    reads indegree, not a live GROUP BY). Falls back from the incremental
    path to a full rebuild whenever churn or preconditions exceed its gates."""
    if changed_ids is not None and deleted_ids is not None:
        total = store.count_chunks()
        batch = len(changed_ids) + len(deleted_ids)
        # has_indegree() gates on chunk_indegree already mirroring ALL of
        # chunk_refs; otherwise take the full rebuild below, a one-time
        # re-derivation, not the O(chunk_refs) scan get_hubs' guard avoids.
        if (
            store.count_refs() > 0
            and store.has_indegree()
            and batch <= _REFS_INCREMENTAL_MAX_FRACTION * max(total, 1)
        ):
            written = _build_refs_incremental(
                store, set(changed_ids), set(deleted_ids), set(deleted_names or ()),
                cap_mentions=cap_mentions,
            )
            if written is not None:
                return written
            # None: referencer fan-out exceeded the blast-radius gate (already
            # logged); fall through to the full rebuild below.
        else:
            logger.info(
                "chunk_refs incremental update skipped (batch=%d, N=%d) — "
                "falling back to full rebuild.",
                batch, total,
            )

    chunks = store.get_named_chunks()
    store.clear_refs()
    if not chunks:
        store.commit()
        return 0
    sym_map = store.get_symbol_name_chunks()
    named_ids = {c["id"] for c in chunks}
    unnamed_definer_ids = sorted(
        {cid for ids in sym_map.values() for cid in ids} - named_ids
    )
    edges = _build_graph(
        chunks, sym_map,
        cap_mentions=cap_mentions, associated_top_frac=associated_top_frac,
        extra_definers=store.get_chunks_by_ids(unnamed_definer_ids) if unnamed_definer_ids else None,
    )
    refs = [(u, v, t) for (u, v), t in edges.items()]
    store.insert_refs(refs)
    # Indegree computed from `refs` already in memory, not a live GROUP BY
    # over chunk_refs (see get_hubs' guard): that scan is the >10-min
    # operation being avoided.
    indegree_counts = Counter((v, t) for (u, v), t in edges.items())
    store.save_indegree(dict(indegree_counts))
    store.commit()
    logger.info("chunk_refs full rebuild: %d named chunks, %d edges.", len(chunks), len(refs))
    return len(refs)


def _bare_alias_definers(
    store: "Store", bare_names: set[str],
) -> dict[str, list[tuple[str, str]]]:
    """Reverse of `_name_keys`, corpus-wide: finds every qualified definer
    elsewhere whose bare last component matches."""
    if not bare_names:
        return {}
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for c in store.get_named_chunks_meta():
        name = c.get("name")
        if not name:
            continue
        keys = _name_keys(name)
        if len(keys) < 2 or keys[1] not in bare_names:
            continue
        out[keys[1]].append((c["id"], c.get("language") or ""))
    for name, ids in store.get_symbol_name_chunks().items():
        keys = _name_keys(name)
        if len(keys) < 2 or keys[1] not in bare_names or len(ids) > _MAX_CROSS_LANG_OCCURRENCES:
            continue
        for cid in ids:
            out[keys[1]].append((cid, ""))
    return dict(out)


def _resolve_names(
    store: "Store", names: set[str] | list[str],
) -> tuple[dict[str, list[str]], dict[str, set[str]], dict[str, str]]:
    """Mirrors `_build_graph`'s name_to_ids construction, scoped to a
    candidate set, replicating its quirks (e.g. symbol-only ids carry no
    language) exactly so the incremental result matches a full rebuild bit-for-bit."""
    names = [n for n in names if n and len(n) >= _MIN_NAME_LEN]
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    name_to_langs: dict[str, set[str]] = defaultdict(set)
    id_to_lang: dict[str, str] = {}
    if not names:
        return name_to_ids, name_to_langs, id_to_lang

    for name, rows in store.get_chunk_defs_by_names(names).items():
        for cid, lang in rows:
            if cid not in name_to_ids[name]:
                name_to_ids[name].append(cid)
            if lang:
                name_to_langs[name].add(lang)
                id_to_lang[cid] = lang

    for name, ids in store.get_symbol_chunk_ids_by_names(names).items():
        if len(ids) > _MAX_CROSS_LANG_OCCURRENCES:
            continue
        existing = set(name_to_ids[name])
        for cid in ids:
            if cid not in existing:
                name_to_ids[name].append(cid)
                existing.add(cid)

    # Bare-alias fold, aggregate-capped; see _bare_alias_definers.
    bare_names = {n for n in names if "::" not in n and "." not in n}
    for bare, pairs in _bare_alias_definers(store, bare_names).items():
        direct = set(name_to_ids.get(bare, ()))
        alias_cids = {cid for cid, _lang in pairs}
        if len(direct | alias_cids) > _MAX_CROSS_LANG_OCCURRENCES:
            continue
        existing = set(name_to_ids[bare])
        langs: set[str] = set()
        for cid, lang in pairs:
            if cid not in existing:
                name_to_ids[bare].append(cid)
                existing.add(cid)
            if lang:
                langs.add(lang)
        if langs:
            name_to_langs[bare] |= langs

    return name_to_ids, name_to_langs, id_to_lang

def _xlang_pairs_for_names(
    name_to_ids: dict[str, list[str]],
    name_to_langs: dict[str, set[str]],
    id_to_lang: dict[str, str],
    names: set[str],
) -> list[tuple[str, str]]:
    """Cross-language bidirectional pairing, scoped to `names`; same cap
    and same-language-skip rules as `_build_graph`."""
    pairs: list[tuple[str, str]] = []
    for name in names:
        langs = name_to_langs.get(name, set())
        if len(langs) < 2:
            continue
        ids = name_to_ids.get(name, [])
        if len(ids) > _MAX_CROSS_LANG_OCCURRENCES:
            continue
        for i, id_a in enumerate(ids):
            lang_a = id_to_lang.get(id_a)
            for id_b in ids[i + 1:]:
                lang_b = id_to_lang.get(id_b)
                if lang_a != lang_b:
                    pairs.append((id_a, id_b))
    return pairs

def _build_refs_incremental(
    store: "Store",
    changed_ids: set[str],
    deleted_ids: set[str],
    deleted_names: set[str],
    *,
    cap_mentions: bool = False,
) -> int | None:
    """Incremental chunk_refs update, an exact equivalent of a full rebuild
    (same edge set and types) when it doesn't return None. Returns None when
    referencer fan-out exceeds the gate; caller must then fall back to a full rebuild."""
    removed = 0
    indegree_dirty: set[str] = set(deleted_ids)
    if deleted_ids:
        removed_n, deleted_to_ids = store.delete_refs_touching(list(deleted_ids))
        removed += removed_n
        indegree_dirty |= deleted_to_ids

    changed_ids_named: set[str] = set()
    changed_own_names: set[str] = set()
    if changed_ids:
        for c in store.get_chunks_by_ids(list(changed_ids)):
            if c.get("name"):
                changed_ids_named.add(c["id"])
                changed_own_names.add(c["name"])
        changed_own_names |= store.get_symbol_names_by_chunk_ids(list(changed_ids))

    # Alias-expanded: a changed/deleted qualified name must also touch its
    # bare alias, or a bare-indexed sibling definer would miss the update.
    touched_names: set[str] = set()
    for n in set(deleted_names) | changed_own_names:
        if n and len(n) >= _MIN_NAME_LEN:
            touched_names.update(_name_keys(n))

    if not touched_names and not changed_ids_named:
        if indegree_dirty:
            store.refresh_indegree(list(indegree_dirty))
        store.commit()
        logger.info(
            "chunk_refs incremental update: no touched names, stale edges removed=%d.",
            removed,
        )
        return 0

    def _possible_target_ids(names: set[str]) -> set[str]:
        if not names:
            return set()
        names_l = list(names)
        ids: set[str] = set()
        for rows in store.get_chunk_defs_by_names(names_l).values():
            ids.update(cid for cid, _lang in rows)
        for cids in store.get_symbol_chunk_ids_by_names(names_l).values():
            ids.update(cids)
        # Reverse alias direction: a bare name here may also alias some
        # other, not-otherwise-touched qualified name's definer elsewhere.
        bare_names = {n for n in names if "::" not in n and "." not in n}
        for pairs in _bare_alias_definers(store, bare_names).values():
            ids.update(cid for cid, _lang in pairs)
        return ids

    affected_target_ids = _possible_target_ids(touched_names)

    # Hoisted above the loop: cheap, and the early-bail check below needs it
    # on every iteration, not just at the final gate.
    total_chunks = store.count_chunks()
    _max_fanout = _REFS_INCREMENTAL_MAX_REFERENCER_FRACTION * max(total_chunks, 1)

    def _fanout_blown_out() -> bool:
        # affected_target_ids is a subset of the eventual referencer_ids, so
        # bailing here early skips the expensive sibling-expansion and FTS
        # work once the final gate is already certain to fire.
        if len(affected_target_ids) > _max_fanout:
            logger.info(
                "chunk_refs incremental update aborted early: affected-target "
                "fan-out %d of %d chunks (touched_names=%d) exceeds %.0f%% — "
                "falling back to full rebuild.",
                len(affected_target_ids), total_chunks, len(touched_names),
                _REFS_INCREMENTAL_MAX_REFERENCER_FRACTION * 100,
            )
            return True
        return False

    if _fanout_blown_out():
        return None

    # Loops sibling expansion to a fixpoint: a second-hop definer added only
    # via a sibling name must also contribute its own names, or its xlang
    # edges get deleted by the target-scoped delete but never rebuilt.
    while affected_target_ids:
        sibling_names = store.get_symbol_names_by_chunk_ids(list(affected_target_ids))
        for c in store.get_chunks_by_ids(list(affected_target_ids)):
            if c.get("name"):
                sibling_names.add(c["name"])
        # Alias-expanded, same reason as touched_names above: a sibling name
        # that happens to be qualified must also contribute its bare alias.
        expanded_siblings: set[str] = set()
        for n in sibling_names:
            if n and len(n) >= _MIN_NAME_LEN:
                expanded_siblings.update(_name_keys(n))
        new_names = expanded_siblings - touched_names
        if not new_names:
            break
        touched_names |= new_names
        affected_target_ids |= _possible_target_ids(touched_names)
        if _fanout_blown_out():
            return None

    # Step 4: referencers whose outgoing edges get recomputed.
    referencer_ids: set[str] = set(changed_ids_named)
    if affected_target_ids:
        referencer_ids |= store.get_chunk_ids_referencing(list(affected_target_ids))
    if touched_names:
        referencer_ids |= {
            row["id"] for row in store.find_named_chunks_referencing(list(touched_names))
        }
    # Excludes ids deleted and not resurrected this batch. A changed file's
    # re-parse can regenerate the SAME content-addressed id, landing it in
    # both sets; that id is still live and must stay eligible as a referencer.
    referencer_ids -= (set(deleted_ids) - set(changed_ids))

    # Blast-radius gate: everything above is cheap; per-referencer
    # recomputation is what costs, so bail to a full rebuild past this fraction.
    if len(referencer_ids) > _max_fanout:
        logger.info(
            "chunk_refs incremental update aborted: referencer fan-out %d of %d "
            "chunks (touched_names=%d) exceeds %.0f%% — falling back to full rebuild.",
            len(referencer_ids), total_chunks, len(touched_names),
            _REFS_INCREMENTAL_MAX_REFERENCER_FRACTION * 100,
        )
        return None

    ref_chunks = [c for c in store.get_chunks_by_ids(list(referencer_ids)) if c.get("name")] \
        if referencer_ids else []
    ref_ids = [c["id"] for c in ref_chunks]

    # Step 5: candidate words to resolve: referencer content/metadata names,
    # plus touched_names (so xlang resolution has them with zero referencers).
    candidate_words: set[str] = set(touched_names)
    for c in ref_chunks:
        content = c.get("content") or ""
        own_name = c.get("name")
        for w in _WORD_RE.findall(content):
            if w != own_name and len(w) >= _MIN_NAME_LEN:
                candidate_words.add(w)
        md = c.get("metadata") or {}
        for edge_type in ("calls", "imports", "inherits"):
            # 'calls' entries are fingerprint dicts, 'imports'/'inherits'
            # plain strings; _call_entry_name handles both so this loop needn't branch.
            for raw in md.get(edge_type) or ():
                name = _call_entry_name(raw)
                if name:
                    candidate_words.add(name)

    local_name_to_ids, local_name_to_langs, local_id_to_lang = _resolve_names(
        store, candidate_words,
    )

    # Wipes every referencer's full outgoing set (type-agnostic delete, the
    # fresh edges below cover all three types) plus every 'xlang' edge
    # touching an affected target (covers definer-only pairs).
    if ref_ids:
        removed_n, from_to_ids = store.delete_refs_from(ref_ids)
        removed += removed_n
        indegree_dirty |= from_to_ids
    if affected_target_ids:
        removed_n, xlang_to_ids = store.delete_xlang_refs_touching(list(affected_target_ids))
        removed += removed_n
        indegree_dirty |= xlang_to_ids

    edges: dict[tuple[str, str], str] = {}

    # mentions (content scan), lowest precedence.
    for c in ref_chunks:
        content = c.get("content") or ""
        own_name = c.get("name")
        referenced = {
            w for w in _WORD_RE.findall(content)
            if w in local_name_to_ids and w != own_name
        }
        for ref_name in referenced:
            ids = local_name_to_ids[ref_name]
            if cap_mentions and len(ids) > _MAX_CROSS_LANG_OCCURRENCES:
                continue
            for target_id in ids:
                if target_id != c["id"]:
                    edges[(c["id"], target_id)] = "mentions"

    # xlang, middle precedence.
    xlang_pairs = _xlang_pairs_for_names(
        local_name_to_ids, local_name_to_langs, local_id_to_lang, touched_names,
    )
    for id_a, id_b in xlang_pairs:
        edges[(id_a, id_b)] = "xlang"
        edges[(id_b, id_a)] = "xlang"

    # Typed edges, highest precedence, run through the same
    # _discriminate_definers as _build_graph's bulk pass; id_to_chunk here
    # only needs 'calls' target chunks, not the whole corpus.
    call_target_ids: set[str] = set()
    for c in ref_chunks:
        md = c.get("metadata") or {}
        for raw in md.get("calls") or ():
            name = _call_entry_name(raw)
            call_target_ids.update(local_name_to_ids.get(name) or ())
    id_to_chunk = (
        {row["id"]: row for row in store.get_chunks_by_ids(list(call_target_ids))}
        if call_target_ids else {}
    )
    arity_cache: dict[tuple[str, str], "tuple[int, int, bool] | None"] = {}

    for c in ref_chunks:
        md = c.get("metadata") or {}
        for edge_type in ("calls", "imports", "inherits"):
            for raw in md.get(edge_type) or ():
                if edge_type == "calls":
                    name, receiver, arity = _call_entry_fields(raw)
                else:
                    name, receiver, arity = raw, None, None
                ids = local_name_to_ids.get(name)
                if not ids or len(ids) > _MAX_CROSS_LANG_OCCURRENCES:
                    continue
                targets = (
                    _discriminate_definers(ids, name, receiver, arity, id_to_chunk, arity_cache)
                    if edge_type == "calls" else ids
                )
                for target_id in targets:
                    if target_id != c["id"]:
                        edges[(c["id"], target_id)] = edge_type

    refs = [(u, v, t) for (u, v), t in edges.items()]
    store.insert_refs(refs)
    inserted = len(refs)
    indegree_dirty |= {v for (_u, v) in edges}

    if indegree_dirty:
        store.refresh_indegree(list(indegree_dirty))
    store.commit()
    logger.info(
        "chunk_refs incremental update: changed=%d, deleted=%d, touched_names=%d, "
        "referencers=%d, stale edges removed=%d, edges written=%d.",
        len(changed_ids), len(deleted_ids), len(touched_names), len(referencer_ids),
        removed, inserted,
    )
    return inserted
