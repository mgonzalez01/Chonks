"""Receiver and arity discrimination of the definers of a called name."""

import re

from chonks.languages import get_or_none as _lang_spec, table as _lang_table, union as _lang_union
from chonks.languages.spec import NOT_HANDLED as _NOT_HANDLED


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


# Definer arity: a text scan over the already-stored `content`Best-effort: None ("missing info") is always
# treated as compatible, so a parse miss costs precision.

# Keyword right before a definition's name (skippable past an unrelated
# same-named call); empty string for languages without one.
_DEF_SIGNATURE_KEYWORD = _lang_table("def_signature_keyword")


def _find_signature_param_texts(content: str, name: str, lang: str) -> list[str]:
    """Param-list text for EVERY same-named signature in `content`, not
    just the first, since Python @overload stacks can differ in arity.
    A match whose parens never balance is skipped, not fatal."""
    if not name:
        return []
    keyword = _DEF_SIGNATURE_KEYWORD.get(lang, "")
    pattern = re.compile(keyword + re.escape(name) + r"\s*\(")
    out: list[str] = []
    pos = 0
    while True:
        m = pattern.search(content, pos)
        if m is None:
            break
        start = m.end() - 1  # index of the anchor '('
        depth = 0
        end = None
        for i in range(start, len(content)):
            ch = content[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            # Skip this unbalanced match; advance past '(' (not `start`)
            # or the next search re-matches it and loops forever.
            pos = start + 1
            continue
        out.append(content[start + 1:end])
        pos = end + 1
    return out


def _split_top_level_params(text: str) -> list[str]:
    """Splits on top-level commas only: bracket-depth-aware, and quote-aware
    so `msg="a, b"` doesn't split. Not a real parser; rare edge cases
    (lambda defaults, '<'/'>' comparisons) can still miscount."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    quote: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            cur.append(ch)
            if ch == "\\" and i + 1 < n:
                cur.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            cur.append(ch)
            i += 1
            continue
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _classify_param(part: str, lang: str) -> str:
    """One parameter's role: 'marker' (not a real param, e.g. python's
    '/'), 'variadic', 'default', or 'required'."""
    spec = _lang_spec(lang)
    if spec is not None and spec.classify_param is not None:
        result = spec.classify_param(part)
        if result is not _NOT_HANDLED:
            return result
    if "=" in part:
        return "default"
    return "required"


def _definer_param_arity(content: str, name: str, lang: str) -> "tuple[int, int, bool] | None":
    """(min_required, max_params, has_variadic) or None (always compatible).
    Overloads WIDEN together (recall over precision); C#'s `this` isn't stripped."""
    params_texts = _find_signature_param_texts(content, name, lang)
    if not params_texts:
        return None
    spec = _lang_spec(lang)
    combined_min: int | None = None
    combined_max = 0
    combined_variadic = False
    for params_text in params_texts:
        parts = _split_top_level_params(params_text)
        if spec is not None and spec.strip_implicit_params is not None:
            parts = spec.strip_implicit_params(parts)
        elif spec is not None and len(parts) == 1 and parts[0] in spec.empty_param_spellings:
            # c/cpp's explicit zero-args spelling ('int f(void)') is the
            # same as an empty parameter list, not a param named 'void'.
            parts = []
        min_required = 0
        max_params = 0
        has_variadic = False
        for part in parts:
            kind = _classify_param(part, lang)
            if kind == "marker":
                continue
            if kind == "variadic":
                has_variadic = True
                continue
            max_params += 1
            if kind == "required":
                min_required += 1
        combined_min = min_required if combined_min is None else min(combined_min, min_required)
        combined_max = max(combined_max, max_params)
        combined_variadic = combined_variadic or has_variadic
    return (combined_min if combined_min is not None else 0, combined_max, combined_variadic)
