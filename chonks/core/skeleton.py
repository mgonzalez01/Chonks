"""Skeleton generation and matching for the literal & message index."""
import re
import sqlite3

# A star import skips underscore names unless __all__ lists them.
__all__ = [
    "_HOLE_SENTINEL",
    "_PRINTF_HOLE_RE",
    "_DOLLAR_HOLE_RE",
    "_BRACE_HOLE_RE",
    "_SKELETON_MAX_HOLES",
    "_SKELETON_MAX_MESSAGE_LEN",
    "_SKELETON_HOLE_GAP",
    "_SKELETON_WORK_BUDGET",
    "_SKELETON_QUERY_BUDGET",
    "_compute_skeleton",
    "_is_message_shaped",
    "_passes_exact_gate",
    "_longest_skeleton_fragment",
    "_skeleton_match",
    "_skeleton_candidates",
    "_first_line",
    "_FIRST_LINE_SQL",
    "_dedupe_literal_rows",
]


# Sentinel for a collapsed format hole. Must stay printable and FTS-safe; a
# raw NUL byte round-trips badly through FTS5 and JSON/HTTP.
_HOLE_SENTINEL = "␀*"

# printf-style hole. "%%" is protected before this runs so it's never
# swallowed as a hole itself.
_PRINTF_HOLE_RE = re.compile(r"%[-+0 #]*(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlLqjzt]*[diouxXeEfFgGaAcsp]")
# ${var} / $VAR (shell/template style). Matched before the generic brace
# pass below so "${name}" collapses to ONE hole, not "$" + a hole.
_DOLLAR_HOLE_RE = re.compile(r"\$\{[^{}]*\}|\$[A-Za-z_][A-Za-z0-9_]*")
# {}, {0}, {name}, {expr}, {0:D4}: python .format/f-string and C#
# interpolation holes. "{{"/"}}" are protected first.
_BRACE_HOLE_RE = re.compile(r"\{[^{}]*\}")


# _skeleton_match walks fragments with str.find instead of a regex, to
# avoid catastrophic backtracking on a repetitive message. Templates past
# this many holes have too little constant text left to verify, so they're excluded.
_SKELETON_MAX_HOLES = 8

# Skeleton tier only (exact/substring unaffected); belt-and-suspenders bound
# on top of the hole cap and work budget below.
_SKELETON_MAX_MESSAGE_LEN = 600

# Max chars a hole may span between two constant fragments.
_SKELETON_HOLE_GAP = 80

# Hard cap on str.find() calls per candidate skeleton; a repetitive message
# can make the fragment walk branch combinatorially otherwise. Exhaustion
# is reported as a note, never a silent miss.
_SKELETON_WORK_BUDGET = 2000

# Hard cap on str.find() calls across all candidates in one query; the
# per-candidate budget alone leaves total query cost unbounded.
_SKELETON_QUERY_BUDGET = 100_000


def _compute_skeleton(text: str) -> str | None:
    """Collapse format holes into _HOLE_SENTINEL. Returns None when no
    holes were found, or when there are more than _SKELETON_MAX_HOLES."""
    protected = (
        text.replace("%%", "\x02")
            .replace("{{", "\x03")
            .replace("}}", "\x04")
    )
    collapsed = _PRINTF_HOLE_RE.sub(_HOLE_SENTINEL, protected)
    collapsed = _DOLLAR_HOLE_RE.sub(_HOLE_SENTINEL, collapsed)
    collapsed = _BRACE_HOLE_RE.sub(_HOLE_SENTINEL, collapsed)
    collapsed = (
        collapsed.replace("\x02", "%")
                 .replace("\x03", "{")
                 .replace("\x04", "}")
    )
    if collapsed == text:
        return None
    if collapsed.count(_HOLE_SENTINEL) > _SKELETON_MAX_HOLES:
        return None
    return collapsed


def _is_message_shaped(text: str) -> bool:
    """True if `text` is phrase-shaped or long enough to be a real log
    line, not just a short word that's a coincidental substring match."""
    return " " in text or len(text) >= 20


def _passes_exact_gate(text: str, message: str) -> bool:
    return _is_message_shaped(text) or text.strip() == message.strip()


def _longest_skeleton_fragment(skeleton: str) -> str:
    return max(skeleton.split(_HOLE_SENTINEL), key=len, default="")


def _skeleton_match(
    skeleton: str, message: str, budget: int = _SKELETON_WORK_BUDGET,
) -> tuple[bool, bool, int]:
    """Match unanchored, like re.search, so extra context around the
    template still matches. Returns (matched, budget_exhausted, calls_used);
    hitting `budget` returns (False, True, calls_used) immediately."""
    parts = skeleton.split(_HOLE_SENTINEL)
    n = len(parts)
    msg_len = len(message)
    calls = 0
    next_lo = [0] * n
    cur_idx = [-1] * n
    ends = [0] * n

    level = 0
    while 0 <= level < n:
        frag = parts[level]
        flen = len(frag)
        lo = next_lo[level]
        hi = msg_len if level == 0 else min(ends[level - 1] + _SKELETON_HOLE_GAP, msg_len)
        if lo > hi:
            idx = -1
        elif frag == "":
            idx = lo
        else:
            if calls >= budget:
                return False, True, calls
            calls += 1
            idx = message.find(frag, lo, hi + flen)
            if idx > hi:
                idx = -1
        if idx == -1:
            level -= 1
            if level < 0:
                break
            next_lo[level] = cur_idx[level] + 1
            continue
        cur_idx[level] = idx
        ends[level] = idx + flen
        level += 1
        if level < n:
            next_lo[level] = ends[level - 1]
    return level == n, False, calls


def _skeleton_candidates(skeleton: str) -> list[str]:
    """The skeleton plus its first-line-only fallback, since a pasted
    message never carries a template's later lines. Returns just the
    skeleton itself when it's single-line."""
    first = _first_line(skeleton)
    return [skeleton] if first == skeleton else [skeleton, first]


def _first_line(text: str) -> str:
    """Text up to its first newline, or `text` unchanged if it has none.
    The exact/substring tiers' fallback for a multi-line literal, since a
    pasted message never carries its later lines."""
    idx = text.find("\n")
    return text if idx == -1 else text[:idx]


# SQL equivalent of _first_line(cl.text), for the tier-1a query. Kept as
# one string so the two definitions stay in sync. Split as BLOB: substr on
# TEXT stops at a NUL, and a decoded C literal can contain one.
_FIRST_LINE_SQL = (
    "CAST(CASE WHEN instr(CAST(cl.text AS BLOB), X'0A') > 0 "
    "THEN substr(CAST(cl.text AS BLOB), 1, instr(CAST(cl.text AS BLOB), X'0A') - 1) "
    "ELSE CAST(cl.text AS BLOB) END AS TEXT)"
)


def _dedupe_literal_rows(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    """Collapse rows sharing (path, line, text): a split boundary node can
    attribute the same literal to several sibling chunks, which would
    otherwise crowd out other matches under `limit`."""
    groups: dict[tuple[str, int, str], list[sqlite3.Row]] = {}
    order: list[tuple[str, int, str]] = []
    for r in rows:
        k = (r["path"], r["line"] or 0, r["text"])
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)
    out = []
    for k in order:
        group = groups[k]
        line = k[1]
        contained = [r for r in group
                     if r["c_start_line"] is not None and r["c_end_line"] is not None
                     and r["c_start_line"] <= line <= r["c_end_line"]]
        out.append(contained[0] if contained else group[0])
    return out
