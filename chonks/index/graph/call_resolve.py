"""Receiver and arity discrimination of the definers of a called name."""

from chonks.chunking import _definer_param_arity
from chonks.languages import union as _lang_union


# ---------------------------------------------------------------------------
# Receiver/arity call-site discriminator
# ---------------------------------------------------------------------------

# Chunk types that stand in for "this chunk IS a class" in
# _definer_qualifiers; omits types whose call sites never carry a
# receiver fingerprint anyway (JS/TS/lua).
_CLASS_LIKE_CHUNK_TYPES = _lang_union("class_like_chunk_types")


def _call_entry_fields(entry: "str | dict") -> tuple[str | None, str | None, int | None]:
    """Normalizes one calls-list entry to (name, receiver, arity); a plain
    string (pre-fingerprint format) becomes (name, None, None), i.e.
    unconditional fan-out for that entry."""
    if isinstance(entry, str):
        return entry, None, None
    if isinstance(entry, dict):
        return entry.get("name"), entry.get("receiver"), entry.get("arity")
    return None, None, None


def _definer_qualifiers(chunk: dict) -> set[str]:
    """Qualifier candidates for a definer, checked against a call's receiver
    token: the chunk's own qualifier prefix, plus its own name when the
    chunk IS the class boundary (its folded methods have no separate name)."""
    name = chunk.get("name") or ""
    out: set[str] = set()
    if not name:
        return out
    dotted = name.replace("::", ".")
    if "." in dotted:
        prefix = dotted.rsplit(".", 1)[0]
        qualifier = prefix.rsplit(".", 1)[-1]
        if qualifier:
            out.add(qualifier)
    if chunk.get("chunk_type") in _CLASS_LIKE_CHUNK_TYPES:
        out.add(name)
    return out


def _arity_compatible(cid: str, chunk: dict, name: str, call_arity: int, arity_cache: dict) -> bool:
    """True if `chunk` could accept `call_arity` positional args (exact
    match or variadic widening); True when no signature is found. Keyed on
    `name`, not the chunk's own name, since a folded chunk holds several signatures."""
    key = (cid, name)
    if key not in arity_cache:
        arity_cache[key] = _definer_param_arity(
            chunk.get("content") or "", name, chunk.get("language") or "",
        )
    sig = arity_cache[key]
    if sig is None:
        return True
    min_required, max_params, has_variadic = sig
    if has_variadic:
        return call_arity >= min_required
    return min_required <= call_arity <= max_params


def _discriminate_definers(
    ids: list[str],
    name: str,
    receiver: str | None,
    arity: int | None,
    id_to_chunk: dict[str, dict],
    arity_cache: dict,
) -> list[str]:
    """Narrows a name collision's definer set by receiver-qualifier match
    and/or arity compatibility (owner match wins when both apply). NOT a
    recall-safe filter: a qualifier collision can silently drop the true definer."""
    owner_survivors: list[str] | None = None
    if receiver is not None:
        exact = [i for i in ids if receiver in _definer_qualifiers(id_to_chunk.get(i) or {})]
        if exact:
            owner_survivors = exact
        else:
            receiver_l = receiver.lower()
            suffix = [
                i for i in ids
                if any(q.lower().endswith(receiver_l) for q in _definer_qualifiers(id_to_chunk.get(i) or {}))
            ]
            if suffix:
                owner_survivors = suffix

    arity_survivors: list[str] | None = None
    if arity is not None:
        arity_survivors = [
            i for i in ids
            if _arity_compatible(i, id_to_chunk.get(i) or {}, name, arity, arity_cache)
        ]

    # owner_survivors is None or non-empty; "found nothing" and "doesn't
    # apply" fall through to (b) alone identically.
    if owner_survivors:
        if arity_survivors is not None:
            keep = set(arity_survivors)
            inter = [i for i in owner_survivors if i in keep]
            if inter:
                return inter
        return owner_survivors
    if arity_survivors is not None:
        return arity_survivors if arity_survivors else ids
    return ids
