"""
query_reformulate.py: rule-based (no LLM) query augmentation.

Pulls identifier-bearing terms out of symptom-language issue text and
appends them to the query, never replaces it: raw query text always stays
verbatim at the front. Pure, no I/O; ships wired but DEFAULT-OFF until the
LocBench panel measures it (see searcher.py's `_augmented_query`).
"""

import re

# Off until the LocBench panel A/Bs it.
DEFAULT_REFORMULATE_QUERY = False

# Cap on appended identifiers. Extraction is first-seen order, so this keeps
# the earliest, usually most relevant, terms and just stops early.
MAX_EXTRACTED_TERMS = 20

# Plain text, not markdown: the embed side sees ordinary query content, not
# a doc structure to render.
_APPEND_SEPARATOR = "\n\nIdentifiers: "

# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------
# Backtick/fence markers are punctuation, not word characters, so they never
# block a \b boundary: an identifier inside a code span matches the same as
# one written bare in prose. No separate fenced pass is needed.

# UPPER_SNAKE constants: MAX_ITERATIONS, HTTP_TIMEOUT_S.
_UPPER_SNAKE_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")

# snake_case identifiers: parse_one, embed_query_token_budget.
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

# Dotted / path-like fragments: foo.bar.baz, dir/file.py, scene/3d/light_3d.cpp.
# Requires at least one '.' or '/' separator so a lone word never matches here.
_DOTTED_PATH_RE = re.compile(r"\b[A-Za-z0-9_-]+(?:[./][A-Za-z0-9_-]+)+\b")

# Dotted-fragment rejects: prose abbreviations ("e.g", "i.e"), where every
# segment is a single letter, and bare version numbers ("3.10"), digits and
# separators only. Neither is a code identifier.
_ALL_SINGLE_LETTER_SEGMENTS_RE = re.compile(r"^[A-Za-z]([./][A-Za-z])+$")
_VERSIONISH_RE = re.compile(r"^[0-9]+([./][0-9]+)+$")


def _dotted_fragment_ok(token: str) -> bool:
    return not (_ALL_SINGLE_LETTER_SEGMENTS_RE.match(token)
                or _VERSIONISH_RE.match(token))

# CamelCase / PascalCase candidates: any bare word, filtered below to those
# with a lower->upper transition, so "Google" doesn't count but "getUserById"
# does (same rule as searcher.py's _split_identifier FTS fallback).
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*\b")
_HAS_CASE_TRANSITION_RE = re.compile(r"[a-z][A-Z]")


def extract_identifiers(text: str, max_terms: int = MAX_EXTRACTED_TERMS) -> list[str]:
    """Pull candidate identifiers out of `text`, first-seen order, deduplicated,
    capped at `max_terms`. Longest match wins at each position, so a dotted
    fragment like "foo.bar_baz.py" absorbs its overlapping "bar_baz" token
    rather than emitting both.
    """
    spans: list[tuple[int, int, str]] = []
    for pattern in (_UPPER_SNAKE_RE, _SNAKE_RE):
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), m.group()))
    for m in _DOTTED_PATH_RE.finditer(text):
        if _dotted_fragment_ok(m.group()):
            spans.append((m.start(), m.end(), m.group()))
    for m in _WORD_RE.finditer(text):
        token = m.group()
        if _HAS_CASE_TRANSITION_RE.search(token):
            spans.append((m.start(), m.end(), token))

    # Longest match wins at each starting position; a shorter span fully
    # covered by an already-claimed range is dropped.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    terms: list[str] = []
    seen: set[str] = set()
    claimed_until = -1
    for start, end, token in spans:
        if start < claimed_until:
            continue
        claimed_until = end
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def augment_query(
    query: str,
    *,
    enabled: bool = DEFAULT_REFORMULATE_QUERY,
    max_terms: int = MAX_EXTRACTED_TERMS,
    budget_chars: int | None = None,
) -> str:
    """Append `query`'s extracted identifiers to itself; a no-op when disabled
    or no identifiers are found. When `budget_chars` would be exceeded, the
    RAW query is truncated instead of the tail, so the identifier tail this
    function exists to add is never the part silently cut off.
    """
    if not enabled:
        return query
    terms = extract_identifiers(query, max_terms)
    if not terms:
        return query
    tail = _APPEND_SEPARATOR + " ".join(terms)
    augmented = query + tail
    if budget_chars is None or len(augmented) <= budget_chars:
        return augmented
    raw_budget = max(0, budget_chars - len(tail))
    return query[:raw_budget] + tail
