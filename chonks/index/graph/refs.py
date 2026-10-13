"""chunk_refs graph construction: mentions/xlang/typed-edge extraction, the
receiver/arity call-site discriminator, and the incremental update path."""

from __future__ import annotations

import heapq
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable
from contextlib import closing
from typing import TYPE_CHECKING, Iterator

from chonks.index.progress import PassLog
from chonks.index.refs_extract import _call_entry_name
from chonks.index.graph.call_resolve import _call_entry_fields, _discriminate_definers, _owner_class_kinds
from chonks.resolve.members import MemberIndex, members_index, settled_targets

from chonks.core.edges import (
    ASSOCIATED, CALLS, INHERITS, MENTIONS, TYPED_EDGE_TYPES, XLANG, _MAX_CROSS_LANG_OCCURRENCES, _MIN_NAME_LEN,
)
from chonks.core.refresh import register_refresh
from chonks.languages import table as _lang_table, union as _lang_union
from chonks.languages._members import strip_template_args

if TYPE_CHECKING:
    from chonks.storage.store import Store

_SCOPE_CHUNK_TYPES = _lang_union("scope_chunk_types")
_TYPED_REF_LANGUAGES = _lang_table("typed_ref_languages")
_CLASS_LIKE_KINDS = _lang_union("class_like_chunk_types")


def _refresh_from_registry() -> None:
    global _SCOPE_CHUNK_TYPES, _TYPED_REF_LANGUAGES, _CLASS_LIKE_KINDS
    _SCOPE_CHUNK_TYPES = _lang_union("scope_chunk_types")
    _TYPED_REF_LANGUAGES = _lang_table("typed_ref_languages")
    _CLASS_LIKE_KINDS = _lang_union("class_like_chunk_types")


def _typed_targets(ids: list[str], lang: str, id_to_chunk: dict[str, dict]) -> list[str]:
    """The definers in `ids` whose language a `lang` chunk's calls, imports
    and bases can name."""
    reach = _TYPED_REF_LANGUAGES.get(lang, frozenset()) | {lang}
    return [i for i in ids if (id_to_chunk.get(i) or {}).get("language") in reach]


def _typed_refs(
    c: dict, name_to_ids: dict[str, list[str]], id_to_chunk: dict[str, dict], arity_cache: dict,
    owner_decls: "Callable[[str], Iterable[dict]] | None", members: MemberIndex | None, cap_mentions: bool,
    class_holders: "Callable[[str, list[str]], list[str]] | None" = None,
) -> tuple[dict[str, str], set[str]]:
    """The chunk's typed edges as {to_id: edge_type}, and the names every call
    of which the class model settles. A base links only the chunks
    `class_holders` keeps. A target reached under several types keeps the
    first in TYPED_EDGE_TYPES, so a call into a base's chunk stays a call."""
    cid = c["id"]
    md = c.get("metadata") or {}
    lang = c.get("language") or ""
    out: dict[str, str] = {}
    settled: set[str] = set()
    unsettled: set[str] = set()
    for edge_type in TYPED_EDGE_TYPES:
        for raw in md.get(edge_type) or ():
            resolution = None
            if edge_type == CALLS:
                name, receiver, arity = _call_entry_fields(raw)
                if members is not None and isinstance(raw, dict):
                    resolution = members.resolve(name, arity, raw, c)
                (unsettled if settled_targets(resolution) is None else settled).add(name)
            else:
                name, receiver, arity = raw, None, None
            if resolution is not None and resolution.targets is not None:
                targets = resolution.targets
            else:
                ids = _typed_targets(name_to_ids.get(name) or [], lang, id_to_chunk)
                if edge_type == INHERITS and class_holders is not None:
                    ids = class_holders(name, ids)
                # The mentions pass's cap, so a typed fact is never dropped in favor of a noisier one.
                if not ids or (cap_mentions and len(ids) > _MAX_CROSS_LANG_OCCURRENCES):
                    continue
                targets = (_discriminate_definers(ids, name, receiver, arity, id_to_chunk, arity_cache, owner_decls)
                           if edge_type == CALLS else ids)
            for target_id in targets:
                if target_id != cid:
                    out.setdefault(target_id, edge_type)
    return out, settled - unsettled


register_refresh(_refresh_from_registry)

logger = logging.getLogger("repomap")

_WORD_RE = re.compile(r"\b[A-Za-z_]\w+\b")

# Below this batch fraction, update chunk_refs incrementally; above it, a
# full rebuild is cheaper than per-referencer bookkeeping.
_REFS_INCREMENTAL_MAX_FRACTION = 0.20

# Second gate, on the chunks whose outgoing edges the batch can change: past
# this fraction of the corpus, recomputing them costs about as much as a full rebuild.
_REFS_INCREMENTAL_MAX_REFERENCER_FRACTION = 0.25

# The parameters chunk_refs and chunk_mention_names were built with; the
# incremental path applies only under the same ones.
_REFS_PARAMS_META_KEY = "refs_params"

# Bumped when a full rebuild writes different edges for the same chunks, so a
# graph built under older rules is rebuilt rather than patched.
_REFS_RULES = 3


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


def _name_index(
    chunks: list[dict], sym_map: dict[str, list[str]] | None,
) -> tuple[dict[str, list[str]], dict[str, set[str]], dict[str, str]]:
    """name -> definer ids, name -> their languages, and id -> language, from
    chunk names, the symbol index and the bare-alias fold, which is capped
    all-or-nothing per key. Reads only id, name, language and chunk_type."""
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    name_to_langs: dict[str, set[str]] = defaultdict(set)
    id_to_lang: dict[str, str] = {}

    # Direct registrations: a name under itself. Never touched by the bare-alias fold below.
    for c in chunks:
        name = c["name"]
        if name and len(name) >= _MIN_NAME_LEN and c.get("chunk_type") not in _SCOPE_CHUNK_TYPES:
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
        if name and len(name) >= _MIN_NAME_LEN and c.get("chunk_type") not in _SCOPE_CHUNK_TYPES:
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

    return dict(name_to_ids), dict(name_to_langs), id_to_lang


def _mentioned_names(
    c: dict, name_to_ids: dict[str, list[str]], settled: Collection[str], cap_mentions: bool,
) -> set[str]:
    """The names `c`'s content spells that reach some other chunk, less its
    own name and the names the class model settles for it."""
    cid, own_name = c["id"], c["name"]
    out: set[str] = set()
    for word in set(_WORD_RE.findall(c["content"] or "")):
        ids = name_to_ids.get(word)
        if not ids or word == own_name or word in settled:
            continue
        if cap_mentions and len(ids) > _MAX_CROSS_LANG_OCCURRENCES:
            continue
        if any(t != cid for t in ids):
            out.add(word)
    return out


def _classify_mentions(
    mentions: dict[str, set[str]], associated_top_frac: float,
) -> set[tuple[str, str]]:
    """PMI-ranks (chunk, mentioned name) pairs and returns the top
    `associated_top_frac` fraction to relabel 'associated'."""
    if associated_top_frac <= 0:
        return set()
    df: Counter[str] = Counter()
    for names in mentions.values():
        df.update(names)
    total = sum(len(names) for names in mentions.values())
    if total == 0:
        return set()
    k = int(associated_top_frac * total)
    if k <= 0:
        return set()

    def _pairs() -> Iterator[tuple[float, str, str]]:
        for cid, names in mentions.items():
            n_a = len(names)
            for name in names:
                yield (-math.log2(total / (n_a * df[name])), name, cid)

    top = heapq.nsmallest(k, _pairs())
    return {(cid, name) for _neg_pmi, name, cid in top}


def _xlang_targets(
    name_to_ids: dict[str, list[str]], name_to_langs: dict[str, set[str]],
    id_to_lang: dict[str, str], names: Iterable[str],
) -> dict[str, set[str]]:
    """id -> the ids it pairs with through one of `names`: definers of one
    name in two or more languages, under the cap, in different languages."""
    out: dict[str, set[str]] = defaultdict(set)
    for name in names:
        if len(name_to_langs.get(name, ())) < 2:
            continue
        ids = name_to_ids[name]
        if len(ids) > _MAX_CROSS_LANG_OCCURRENCES:
            continue
        for id_a in ids:
            lang_a = id_to_lang.get(id_a)
            out[id_a].update(id_b for id_b in ids if id_to_lang.get(id_b) != lang_a)
    return out


def _source_edges(
    cid: str, mentioned: Iterable[str], associated: set[tuple[str, str]],
    name_to_ids: dict[str, list[str]], xlang: Iterable[str], typed: dict[str, str],
) -> dict[str, str]:
    """`cid`'s outgoing edges as {to_id: edge_type}. A typed edge supersedes
    xlang, xlang supersedes associated, associated supersedes mentions."""
    out: dict[str, str] = {}
    for name in mentioned:
        edge_type = ASSOCIATED if (cid, name) in associated else MENTIONS
        for target_id in name_to_ids[name]:
            if target_id != cid and out.get(target_id) != ASSOCIATED:
                out[target_id] = edge_type
    for target_id in xlang:
        out[target_id] = XLANG
    out.update(typed)
    return out


def _graph(
    chunks: list[dict], sym_map: dict[str, list[str]] | None = None,
    *, cap_mentions: bool = False, associated_top_frac: float = 0.0,
    extra_definers: list[dict] | None = None, log: PassLog | None = None,
    owner_decls: "Callable[[str], Iterable[dict]] | None" = None,
    members: MemberIndex | None = None,
    class_holders: "Callable[[str, list[str]], list[str]] | None" = None,
) -> tuple[dict[tuple[str, str], str], dict[str, set[str]]]:
    """{(from_id, to_id): edge_type} from mentions, xlang pairing and typed
    AST facts, and each chunk's mentioned names. A name every call of which
    the class model settles makes no mentions. `log` reports each step on a
    full rebuild."""
    name_to_ids, name_to_langs, id_to_lang = _name_index(chunks, sym_map)

    # Typed facts come first: the names a chunk's calls settle stay out of its mentions.
    id_to_chunk = {c["id"]: c for c in chunks}
    for c in extra_definers or ():
        id_to_chunk.setdefault(c["id"], c)
    arity_cache: dict[tuple[str, str], "tuple[int, int, bool] | None"] = {}
    typed: dict[str, dict[str, str]] = {}
    settled: dict[str, set[str]] = {}
    t0 = time.monotonic()
    for i, c in enumerate(chunks):
        if log:
            log.progress("typed edges", i, len(chunks))
        got, names = _typed_refs(c, name_to_ids, id_to_chunk, arity_cache, owner_decls, members, cap_mentions,
                                 class_holders)
        if got:
            typed[c["id"]] = got
        if names:
            settled[c["id"]] = names
    if log:
        log.info("typed edges in %.1fs", time.monotonic() - t0)
        t0 = time.monotonic()

    mentions: dict[str, set[str]] = {}
    for i, c in enumerate(chunks):
        if log:
            log.progress("name scan", i, len(chunks))
        mentions[c["id"]] = _mentioned_names(c, name_to_ids, settled.get(c["id"], ()), cap_mentions)
    if log:
        log.info("name scan done in %.1fs", time.monotonic() - t0)
        t0 = time.monotonic()
    associated = _classify_mentions(mentions, associated_top_frac)
    if log:
        log.info("scored mentions in %.1fs", time.monotonic() - t0)
        t0 = time.monotonic()

    xlang = _xlang_targets(name_to_ids, name_to_langs, id_to_lang, name_to_langs)
    sources = [*mentions, *(cid for cid in xlang if cid not in mentions)]
    edges: dict[tuple[str, str], str] = {}
    for i, cid in enumerate(sources):
        if log:
            log.progress("edges", i, len(sources))
        out = _source_edges(cid, mentions.get(cid, ()), associated, name_to_ids,
                            xlang.get(cid, ()), typed.get(cid, {}))
        for target_id, edge_type in out.items():
            edges[(cid, target_id)] = edge_type
    if log:
        log.info("%d edges in %.1fs", len(edges), time.monotonic() - t0)
    return edges, mentions


def _build_graph(chunks: list[dict], sym_map: dict[str, list[str]] | None = None, **kw) -> dict[tuple[str, str], str]:
    """The edges of `_graph`."""
    return _graph(chunks, sym_map, **kw)[0]


def _owner_decls(store: "Store") -> "Callable[[str], list[dict]]":
    """Every chunk inside the classes of a name, so the member declarations
    of a class split into pieces are all found."""
    kinds = _owner_class_kinds()
    memo: dict[str, list[dict]] = {}

    def lookup(qualifier: str) -> list[dict]:
        if qualifier not in memo:
            memo[qualifier] = [
                c for path, _lang, start, end in store.get_symbol_spans(qualifier, kinds)
                for c in store.get_chunks_by_path_line_range(path, start - 1, end)
            ]
        return memo[qualifier]

    return lookup


def _class_name(name: str) -> str:
    """`ns::Vector<T>` -> `Vector`."""
    return re.split(r"::|\.", strip_template_args(name))[-1]


def _class_holders(store: "Store") -> "Callable[[str, list[str]], list[str]]":
    """The chunks among `ids` holding a class-like symbol named `name`."""
    held: dict[str, set[str]] = {}

    def holders(name: str, ids: list[str]) -> list[str]:
        todo = [i for i in ids if i not in held]
        held.update((i, set()) for i in todo)
        for row in store.get_symbols_by_chunk_ids(todo) if todo else ():
            if row["kind"] in _CLASS_LIKE_KINDS:
                held[row["chunk_id"]].add(_class_name(row["name"]))
        bare = _class_name(name)
        return [i for i in ids if bare in held[i]]

    return holders


def _refs_params(cap_mentions: bool, associated_top_frac: float) -> str:
    return json.dumps([_REFS_RULES, bool(cap_mentions), float(associated_top_frac)])


def build_refs(
    store: "Store",
    *,
    changed_ids: set[str] | list[str] | None = None,
    deleted_ids: set[str] | list[str] | None = None,
    deleted_defs: dict[tuple[str, str | None], frozenset[tuple]] | None = None,
    cap_mentions: bool = False,
    associated_top_frac: float = 0.0,
) -> int:
    """Computes chunk_refs and keeps chunk_indegree exactly in sync (get_hubs
    reads indegree, not a live GROUP BY). Falls back from the incremental
    path to a full rebuild whenever churn or preconditions exceed its gates.
    `deleted_defs` is what the deleted files defined, from Store.get_definitions."""
    if changed_ids is not None and deleted_ids is not None:
        total = store.count_chunks()
        batch = len(changed_ids) + len(deleted_ids)
        params = _refs_params(cap_mentions, associated_top_frac)
        built_with = store.get_meta(_REFS_PARAMS_META_KEY)
        # has_indegree() gates on chunk_indegree already mirroring ALL of
        # chunk_refs; otherwise take the full rebuild below, a one-time
        # re-derivation, not the O(chunk_refs) scan get_hubs' guard avoids.
        if (
            store.count_refs() > 0
            and store.has_indegree()
            and built_with == params
            and batch <= _REFS_INCREMENTAL_MAX_FRACTION * max(total, 1)
        ):
            written = _build_refs_incremental(
                store, set(changed_ids), set(deleted_ids), deleted_defs or {},
                cap_mentions=cap_mentions, associated_top_frac=associated_top_frac,
            )
            if written is not None:
                return written
        else:
            logger.info(
                "chunk_refs incremental update skipped (batch=%d, N=%d, parameters %s, "
                "graph built with %s) — falling back to full rebuild.",
                batch, total, params, built_with,
            )

    log = PassLog(logger, "chunk_refs")
    log.info("full rebuild, loading named chunks")
    t0 = time.monotonic()
    chunks = store.get_named_chunks()
    store.clear_refs()
    if not chunks:
        store.commit()
        return 0
    log.info("loaded %d named chunks in %.1fs", len(chunks), time.monotonic() - t0)
    sym_map = store.get_symbol_name_chunks()
    named_ids = {c["id"] for c in chunks}
    unnamed_definer_ids = sorted(
        {cid for ids in sym_map.values() for cid in ids} - named_ids
    )
    edges, mentions = _graph(
        chunks, sym_map,
        cap_mentions=cap_mentions, associated_top_frac=associated_top_frac,
        extra_definers=store.get_chunks_by_ids(unnamed_definer_ids) if unnamed_definer_ids else None,
        log=log, owner_decls=_owner_decls(store), members=members_index(store),
        class_holders=_class_holders(store),
    )
    refs = [(u, v, t) for (u, v), t in edges.items()]
    log.info("writing %d edges", len(refs))
    t0 = time.monotonic()
    store.insert_refs(refs)
    if associated_top_frac > 0:
        store.insert_mention_names([(cid, name) for cid, names in mentions.items() for name in names])
    # Indegree computed from `refs` already in memory, not a live GROUP BY
    # over chunk_refs (see get_hubs' guard): that scan is the >10-min
    # operation being avoided.
    indegree_counts = Counter((v, t) for (u, v), t in edges.items())
    store.save_indegree(dict(indegree_counts))
    store.set_meta(_REFS_PARAMS_META_KEY, _refs_params(cap_mentions, associated_top_frac))
    log.info("wrote edges in %.1fs", time.monotonic() - t0)
    logger.info("chunk_refs full rebuild: %d named chunks, %d edges.", len(chunks), len(refs))
    return len(refs)


def _touched_names(
    store: "Store", changed_ids: set[str], deleted_defs: dict[tuple[str, str | None], frozenset[tuple]],
) -> set[str] | None:
    """The names, with their bare aliases, of the definitions the batch
    removed, added or altered; a chunk re-inserted under its old id with the
    same rows touches nothing. None when a row tied to no chunk changed, a
    class's bases or a file's using-directives: calls anywhere can resolve
    through those."""
    paths = {path for path, _cid in deleted_defs} | {c["path"] for c in store.get_chunk_spans(list(changed_ids))}
    after = store.get_definitions(sorted(paths))
    names: set[str] = set()
    for key in deleted_defs.keys() | after.keys():
        old, new = deleted_defs.get(key, frozenset()), after.get(key, frozenset())
        if old != new:
            if key[1] is None:
                return None
            names.update(row[1] for row in old | new)
    return {key for n in names if n and len(n) >= _MIN_NAME_LEN for key in _name_keys(n)}


def _xlang_endpoints(
    names: set[str], chunks: list[dict], sym_map: dict[str, list[str]], changed_ids: set[str],
) -> set[str]:
    """Every chunk that can pair across languages through one of `names`,
    before the batch or after it. A share of a name's definers is left out
    when its definers outside `changed_ids` already put it over the cap."""
    direct: dict[str, set[str]] = defaultdict(set)
    alias: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        name = c["name"]
        if not name or len(name) < _MIN_NAME_LEN or c.get("chunk_type") in _SCOPE_CHUNK_TYPES:
            continue
        keys = _name_keys(name)
        if keys[0] in names:
            direct[keys[0]].add(c["id"])
        if len(keys) > 1 and keys[1] in names:
            alias[keys[1]].add(c["id"])
    alias_symbols: dict[str, set[str]] = defaultdict(set)
    for name, ids in sym_map.items():
        keys = _name_keys(name)
        if len(keys) > 1 and keys[1] in names:
            alias_symbols[keys[1]].update(ids)

    out: set[str] = set()
    for name in names:
        ids = direct[name]
        if len(ids - changed_ids) > _MAX_CROSS_LANG_OCCURRENCES:
            continue
        out |= ids
        symbols = sym_map.get(name, ())
        if sum(cid not in changed_ids for cid in symbols) <= _MAX_CROSS_LANG_OCCURRENCES:
            out.update(symbols)
        if len((ids | alias[name]) - changed_ids) <= _MAX_CROSS_LANG_OCCURRENCES:
            out |= alias[name] | alias_symbols[name]
    return out


def _spelled(c: dict) -> set[str]:
    """The words of a chunk's content and the names its calls, imports and bases record."""
    md = c.get("metadata") or {}
    return set(_WORD_RE.findall(c["content"] or "")) | {
        _call_entry_name(raw) for edge_type in TYPED_EDGE_TYPES for raw in md.get(edge_type) or ()
    }


def _build_refs_incremental(
    store: "Store",
    changed_ids: set[str],
    deleted_ids: set[str],
    deleted_defs: dict[tuple[str, str | None], frozenset[tuple]],
    *,
    cap_mentions: bool = False,
    associated_top_frac: float = 0.0,
) -> int | None:
    """Brings chunk_refs to what a full rebuild would write: recomputes the
    outgoing edges of every chunk the batch can reach and writes the
    difference. Returns None, having written nothing, past the gate."""
    if not changed_ids and not deleted_ids:
        return 0
    gone = deleted_ids - changed_ids
    total = store.count_chunks()
    chunks = store.get_named_chunks_meta()
    sym_map = store.get_symbol_name_chunks()
    name_to_ids, name_to_langs, id_to_lang = _name_index(chunks, sym_map)

    # A chunk's edges change only through its own content, a name it spells
    # or defines whose definitions changed, or the PMI ranking.
    touched = _touched_names(store, changed_ids, deleted_defs)
    if touched is None:
        logger.info("chunk_refs incremental update aborted: class bases or using-directives "
                    "changed — falling back to full rebuild.")
        return None
    base = changed_ids | _xlang_endpoints(touched, chunks, sym_map, changed_ids)
    limit = _REFS_INCREMENTAL_MAX_REFERENCER_FRACTION * max(total, 1)
    rows: dict[str, dict] = {}
    n_sources = len(base)
    if n_sources <= limit:
        with closing(store.iter_named_chunks_referencing(sorted(touched))) as referencers:
            for c in referencers:
                if _spelled(c) & touched:
                    rows[c["id"]] = c
                    if c["id"] not in base:
                        n_sources += 1
                        if n_sources > limit:
                            break
    if n_sources > limit:
        logger.info(
            "chunk_refs incremental update aborted: over %d of %d chunks to recompute "
            "(touched_names=%d), past %.0f%% — falling back to full rebuild.",
            limit, total, len(touched), _REFS_INCREMENTAL_MAX_REFERENCER_FRACTION * 100,
        )
        return None
    sources = rows.keys() | base
    rows.update((c["id"], c) for c in store.get_chunks_by_ids([i for i in sources if i not in rows]))
    named = [c for c in rows.values() if c["name"] is not None]

    members = members_index(store)
    typed_ids = {
        target_id for c in named for edge_type in TYPED_EDGE_TYPES
        for raw in (c.get("metadata") or {}).get(edge_type) or ()
        for target_id in name_to_ids.get(_call_entry_name(raw)) or ()
    }
    id_to_chunk = {c["id"]: c for c in store.get_chunks_by_ids(list(typed_ids))}
    arity_cache: dict[tuple[str, str], "tuple[int, int, bool] | None"] = {}
    owner_decls = _owner_decls(store)
    class_holders = _class_holders(store)
    typed: dict[str, dict[str, str]] = {}
    mentions: dict[str, set[str]] = {}
    for c in named:
        typed[c["id"]], settled = _typed_refs(
            c, name_to_ids, id_to_chunk, arity_cache, owner_decls, members, cap_mentions, class_holders)
        mentions[c["id"]] = _mentioned_names(c, name_to_ids, settled, cap_mentions)
    xlang = _xlang_targets(name_to_ids, name_to_langs, id_to_lang,
                           [name for name, ids in name_to_ids.items() if not sources.isdisjoint(ids)])

    # The ranking is corpus-wide: a chunk outside `sources` keeps its edges
    # and names, but a pair of it can cross the cut either way.
    associated: set[tuple[str, str]] = set()
    relabelled: dict[str, set[str]] = {}
    if associated_top_frac > 0:
        before = store.get_mention_names()
        promoted_before = _classify_mentions(before, associated_top_frac)
        after = {cid: names for cid, names in before.items() if cid not in gone and cid not in sources}
        after.update((cid, names) for cid, names in mentions.items() if names)
        associated = _classify_mentions(after, associated_top_frac)
        relabelled = {cid: after[cid] for cid, _name in associated ^ promoted_before
                      if cid in after and cid not in sources}

    removed_n, gone_targets = store.delete_refs_touching(list(gone))
    stored = store.get_refs_from(sorted(sources | relabelled.keys()))
    new_out: dict[str, dict[str, str]] = {
        cid: _source_edges(cid, mentions.get(cid, ()), associated, name_to_ids,
                           xlang.get(cid, ()), typed.get(cid, {}))
        for cid in sources
    }
    for cid, names in relabelled.items():
        mention_types = _source_edges(cid, names, associated, name_to_ids, (), {})
        new_out[cid] = {
            to_id: mention_types.get(to_id, t) if t in (MENTIONS, ASSOCIATED) else t
            for to_id, t in stored.get(cid, {}).items()
        }
    removed: list[tuple[str, str]] = []
    written: list[tuple[str, str, str]] = []
    for cid, out in new_out.items():
        old = stored.get(cid, {})
        removed.extend((cid, to_id) for to_id, t in old.items() if out.get(to_id) != t)
        written.extend((cid, to_id, t) for to_id, t in out.items() if old.get(to_id) != t)

    store.delete_refs(removed)
    store.insert_refs(written)
    if associated_top_frac > 0:
        store.delete_mention_names(sorted(gone | sources))
        store.insert_mention_names([(cid, name) for cid, names in mentions.items() for name in names])
    store.refresh_indegree(sorted(
        gone | gone_targets | {to_id for _cid, to_id in removed} | {to_id for _cid, to_id, _t in written}))
    store.commit()
    logger.info(
        "chunk_refs incremental update: changed=%d, deleted=%d, touched_names=%d, "
        "recomputed=%d, relabelled=%d, rows removed=%d, rows written=%d.",
        len(changed_ids), len(deleted_ids), len(touched), len(sources), len(relabelled),
        removed_n + len(removed), len(written),
    )
    return len(written)
