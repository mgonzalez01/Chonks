"""The chunks that define a name, and the edges into them its callers'
recorded references explain: `chunk_refs` carries no callee name, and one
chunk can define several."""

import re
from collections import defaultdict
from typing import NamedTuple

from chonks.core.edges import ASSOCIATED, CALLS, MENTIONS, TYPED_EDGE_TYPES, XLANG
from chonks.resolve.members import OUTSIDE_CLASSES, Lookup, settled_targets, shared_members_index

_QUALIFIER = re.compile(r"::|\.")


def last_component(name: str) -> str:
    return _QUALIFIER.split(name)[-1] or name


def whole_word(name: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")


class Targets(NamedTuple):
    name: str
    member: str  # the name a caller's recorded reference uses
    chunk_ids: list[str]
    lookup: Lookup | None  # the class model's reading of a qualified name

    @property
    def checked(self) -> bool:
        """Whether callers are checked against the class model's targets: a
        class's method, not a namespace's function."""
        return self.lookup is not None and bool(self.lookup.targets) and bool(self.lookup.classes)


class Reached(NamedTuple):
    from_id: str
    to_id: str
    edge_type: str
    verified: bool | None  # for a calls edge: whether one of its calls resolves to to_id


def resolve_targets(store, name: str) -> Targets:
    """The chunks defining `name`. A qualified method name goes through the class
    model, as the graph build resolves calls, plus any symbol spelled that way;
    any other name through the symbol index, a bare one with its qualified
    definitions (`Array::size` for `size`)."""
    index = shared_members_index(store)
    found = index.lookup(name) if index is not None else None
    if found is not None and found.targets:
        ids = list(dict.fromkeys([*found.targets, *store.resolve_symbol_chunk_ids(name)]))
    else:
        ids = store.resolve_symbol_chunk_ids(name, with_qualified=True)
    return Targets(name, last_component(name), ids, found)


def owners_by_chunk(store, chunk_ids: list[str], member: str) -> dict[str, list[str]]:
    """chunk id -> the classes whose `member` the chunk defines."""
    out: dict[str, list[str]] = defaultdict(list)
    for r in store.get_members_by_chunk_ids(chunk_ids, member):
        if r["owner"] not in out[r["chunk_id"]]:
            out[r["chunk_id"]].append(r["owner"])
    return out


def reaching_edges(store, targets: Targets, edges: list[tuple[str, str, str]],
                   callers: dict[str, dict]) -> list[Reached]:
    """The edges in `edges` (into `targets`) that the caller's references to
    the name explain; `callers` holds each source chunk's row. A typed edge needs
    a reference to the name, or to a member the chunk defines for a class of that
    name, and a call the class model resolves must resolve to the edge's chunk;
    for a class's method, one whose evidence places it outside every class does not.
    A mention is dropped when the caller never spells the name, defines it
    without calling it, or calls it only as something the class model resolves elsewhere."""
    member = targets.member
    to_ids = sorted({t for _f, t, _et in edges})
    allowed: dict[str, set[str]] = {t: {targets.name, member} for t in to_ids}
    for r in store.get_members_by_chunk_ids(to_ids):
        if last_component(r["owner"]) == member:
            allowed[r["chunk_id"]].add(r["name"])

    named = sorted({f for f, _t, et in edges if et in (MENTIONS, ASSOCIATED, XLANG)})
    word = whole_word(member)
    spelled = {cid for cid, content in store.get_contents_containing(named, member) if word.search(content)}
    mentioning = sorted({f for f, _t, et in edges if et in (MENTIONS, ASSOCIATED) and f in spelled})

    reading: dict[str, set[str]] = defaultdict(set)
    for f, _t, et in edges:
        if et in TYPED_EDGE_TYPES:
            reading[et].add(f)
    reading[CALLS].update(mentioning)
    names = sorted(set().union(*allowed.values()))
    index = shared_members_index(store)
    refs: dict[tuple[str, str], list[tuple[str, list[str] | None, list[str] | None]]] = defaultdict(list)
    for key, ids in reading.items():
        for row in store.get_ref_entries(sorted(ids), key, names):
            entry = row["entry"]
            ref = entry.get("name") if isinstance(entry, dict) else entry
            placed = heard = None
            if key == CALLS and index is not None and isinstance(entry, dict):
                resolution = index.resolve(ref, entry.get("arity"), entry, row)
                if resolution is not None:
                    if targets.checked and resolution.outcome in OUTSIDE_CLASSES:
                        placed = heard = []
                    else:
                        placed, heard = resolution.targets, settled_targets(resolution)
            refs[(row["id"], key)].append((ref, placed, heard))

    silent = [f for f in mentioning if not any(ref == member for ref, _p, _h in refs.get((f, CALLS), ()))]
    definers = {cid for cid in silent if last_component(callers[cid]["name"] or "") == member}
    definers.update(r["chunk_id"] for r in store.get_symbols_by_chunk_ids(silent)
                    if last_component(r["name"]) == member)
    definers.update(r["chunk_id"] for r in store.get_members_by_chunk_ids(silent, member))

    kept: list[Reached] = []
    for f, t, et in edges:
        if et in TYPED_EDGE_TYPES:
            found = [got for ref, got, _h in refs.get((f, et), ()) if ref in allowed[t]]
            if any(got is not None and t in got for got in found):
                kept.append(Reached(f, t, et, True if et == CALLS else None))
            elif any(got is None for got in found):
                kept.append(Reached(f, t, et, False if et == CALLS else None))
        elif et in (MENTIONS, ASSOCIATED, XLANG):
            if f not in spelled or (et != XLANG and f in definers):
                continue
            found = [got for ref, _p, got in refs.get((f, CALLS), ()) if ref == member] if et != XLANG else []
            if not found or not all(got is not None and t not in got for got in found):
                kept.append(Reached(f, t, et, None))
        else:
            kept.append(Reached(f, t, et, None))
    return kept
