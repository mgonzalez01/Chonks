"""Finds the source literal for a pasted runtime message (find_by_message)."""

import re
import sqlite3
from typing import Any

from chonks.core.skeleton import (
    _FIRST_LINE_SQL,
    _HOLE_SENTINEL,
    _SKELETON_MAX_HOLES,
    _SKELETON_MAX_MESSAGE_LEN,
    _SKELETON_QUERY_BUDGET,
    _SKELETON_WORK_BUDGET,
    _dedupe_literal_rows,
    _first_line,
    _longest_skeleton_fragment,
    _passes_exact_gate,
    _skeleton_candidates,
    _skeleton_match,
)

# Means "every chunk in this DB was produced by literal-aware code", set
# only by a full/--force index_paths run, never by a single insert_chunks
# batch. Get this wrong and find_by_message silently looks empty.
_LITERAL_INDEX_META_KEY = "literal_index_version"


def _partial_index_note(conn: sqlite3.Connection) -> str:
    """Message for an unflagged DB (_LITERAL_INDEX_META_KEY): distinguishes
    never-indexed (chunk_literals empty) from partially-indexed (some rows)
    so the note doesn't misreport a real gap as a clean no-match."""
    has_literals = conn.execute("SELECT 1 FROM chunk_literals LIMIT 1").fetchone() is not None
    if has_literals:
        return (
            "literal index incomplete — this DB was only partially indexed "
            "by literal-extraction-aware code; some indexed files have no "
            "literals yet. Run `chonks index --force <paths>` for a full "
            "re-index to enable find_by_message for every file"
        )
    return (
        "index predates literal extraction — run `chonks index --force "
        "<paths>` (a plain re-index skips unchanged files and won't "
        "populate literals) to enable find_by_message"
    )

# FTS candidate-pool cap for the token/skeleton tier, relevance-ordered
# (bm25) so a hit cap only ever drops the least-relevant candidates.
_LITERAL_CANDIDATE_CAP = 20000

# Matches server.py's request-body cap so every accepted message gets the
# substring tier. Skipping above this length is announced, not silent.
_SUBSTRING_TIER_MAX_MESSAGE_LEN = 2000

# Caps the FTS token pool to the N longest tokens so the OR-query stays
# cheap. Exceeding this can crowd out the real literal's tokens; announced,
# not silently dropped.
_TOKEN_POOL_CAP = 12


def find_by_message(store, message: str, limit: int = 20) -> dict[str, Any]:
    """Finds the source literal for a pasted runtime message, including
    through format holes (skeleton tier). Every cap, skip, or exclusion
    is reported via `note`; never a silent empty result."""
    try:
        with store._lock:
            has_chunks = store._conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None
            literal_index_live = store._conn.execute(
                "SELECT 1 FROM meta WHERE key=?", (_LITERAL_INDEX_META_KEY,)
            ).fetchone() is not None
            if has_chunks and not literal_index_live:
                return {
                    "results": [],
                    "truncated": False,
                    "note": _partial_index_note(store._conn),
                }

            message = message or ""
            join = (
                "SELECT cl.chunk_id, cl.text, cl.skeleton, cl.line, "
                "c.path AS path, c.name AS name, "
                "c.start_line AS c_start_line, c.end_line AS c_end_line "
                "FROM chunk_literals cl JOIN chunks c ON c.id = cl.chunk_id "
            )

            # Tier 1a: literal is a substring of the message or vice
            # versa (covers an elided/truncated log fragment). Skipped
            # above _SUBSTRING_TIER_MAX_MESSAGE_LEN, announced below.
            tier1a_rows: list[sqlite3.Row] = []
            substring_tier_skipped = len(message) >= _SUBSTRING_TIER_MAX_MESSAGE_LEN
            if 0 < len(message) and not substring_tier_skipped:
                # An empty first line (text starting with a newline) is a
                # substring of every message, so it never counts.
                where = (
                    "WHERE length(cl.text) >= 6 AND "
                    f"(instr(?, cl.text) > 0 OR ({_FIRST_LINE_SQL} <> '' "
                    f"AND instr(?, {_FIRST_LINE_SQL}) > 0))"
                )
                params: list[Any] = [message, message]
                if len(message) >= 6:
                    where += f" OR instr(cl.text, ?) > 0 OR instr({_FIRST_LINE_SQL}, ?) > 0"
                    params.extend([message, message])
                tier1a_rows = store._conn.execute(join + where, params).fetchall()

            # FTS candidate pool keyed off the message's rarest tokens,
            # quoted so words like AND/OR/NOT match literally instead of
            # parsing as FTS operators. Capped at _TOKEN_POOL_CAP, announced.
            all_tokens = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", message)),
                                 key=len, reverse=True)
            token_cap_applied = len(all_tokens) > _TOKEN_POOL_CAP
            tokens = all_tokens[:_TOKEN_POOL_CAP]
            candidate_rows: list[sqlite3.Row] = []
            candidates_truncated = False
            fts_degraded = False
            if tokens:
                fts_query = " OR ".join(f'"{t}"' for t in tokens)
                try:
                    rows = store._conn.execute(
                        "SELECT cl.chunk_id, cl.text, cl.skeleton, cl.line, "
                        "c.path AS path, c.name AS name, "
                        "c.start_line AS c_start_line, c.end_line AS c_end_line "
                        "FROM literals_fts "
                        "JOIN chunk_literals cl ON cl.rowid = literals_fts.rowid "
                        "JOIN chunks c ON c.id = cl.chunk_id "
                        "WHERE literals_fts MATCH ? "
                        "ORDER BY bm25(literals_fts) LIMIT ?",
                        (fts_query, _LITERAL_CANDIDATE_CAP + 1),
                    ).fetchall()
                    candidates_truncated = len(rows) > _LITERAL_CANDIDATE_CAP
                    candidate_rows = rows[:_LITERAL_CANDIDATE_CAP]
                except sqlite3.OperationalError:
                    candidate_rows = []
                    fts_degraded = True  # malformed FTS query: degrade, don't 500

        # Lock released above. Verification runs here so a pathological
        # query can't block every other request on the store lock.

        seen: set[tuple[str, int, str]] = set()
        exact_rows: list[sqlite3.Row] = []

        def _key(row: sqlite3.Row) -> tuple[str, int, str]:
            return (row["chunk_id"], row["line"] or 0, row["text"])

        def _text_matches(text: str) -> bool:
            if text in message or (len(message) >= 6 and message in text):
                return True
            first = _first_line(text)
            if first is text or not first:
                return False
            return first in message or (len(message) >= 6 and message in first)

        # Rows that substring-matched but failed _passes_exact_gate stay
        # eligible for the skeleton tier below (not fully rejected), but
        # count once into the note.
        low_confidence_filtered: set[tuple[str, int, str]] = set()

        for row in tier1a_rows:
            k = _key(row)
            if k in seen:
                continue
            if _passes_exact_gate(row["text"], message):
                seen.add(k)
                exact_rows.append(row)
            else:
                low_confidence_filtered.add(k)

        for row in candidate_rows:
            k = _key(row)
            if k in seen or len(row["text"]) < 6 or not _text_matches(row["text"]):
                continue
            if _passes_exact_gate(row["text"], message):
                seen.add(k)
                exact_rows.append(row)
            else:
                low_confidence_filtered.add(k)

        # Length/hole caps apply after the cheap prefilter so excluded
        # candidates still count into the note. Budget exhaustion wins
        # priority over a flat exclusion: "may exist" beats "excluded".
        message_too_long = len(message) > _SKELETON_MAX_MESSAGE_LEN
        skeleton_rows: list[sqlite3.Row] = []
        skeleton_excluded_length = 0
        skeleton_excluded_holes = 0
        skeleton_budget_exhausted = 0
        query_budget = _SKELETON_QUERY_BUDGET
        for row in candidate_rows:
            k = _key(row)
            if k in seen or not row["skeleton"]:
                continue
            matched = False
            any_attempted = False
            length_blocked = False
            hole_blocked = False
            exhausted = False
            remaining_budget = min(_SKELETON_WORK_BUDGET, query_budget)
            for variant in _skeleton_candidates(row["skeleton"]):
                fragment = _longest_skeleton_fragment(variant)
                if len(fragment) < 6 or fragment not in message:
                    continue
                if message_too_long:
                    length_blocked = True
                    continue
                if variant.count(_HOLE_SENTINEL) > _SKELETON_MAX_HOLES:
                    hole_blocked = True
                    continue
                any_attempted = True
                if remaining_budget <= 0:
                    exhausted = True
                    continue
                is_match, variant_exhausted, calls_used = _skeleton_match(
                    variant, message, remaining_budget,
                )
                remaining_budget -= calls_used
                query_budget -= calls_used
                if variant_exhausted:
                    exhausted = True
                    continue
                if is_match:
                    matched = True
                    break
            if matched:
                seen.add(k)
                skeleton_rows.append(row)
            elif exhausted:
                skeleton_budget_exhausted += 1
            elif length_blocked and not any_attempted:
                skeleton_excluded_length += 1
            elif hole_blocked and not any_attempted:
                skeleton_excluded_holes += 1

        exact_rows = _dedupe_literal_rows(exact_rows)
        skeleton_rows = _dedupe_literal_rows(skeleton_rows)

        exact_rows.sort(key=lambda r: (r["path"], r["line"] or 0))
        skeleton_rows.sort(key=lambda r: (
            -len(_longest_skeleton_fragment(r["skeleton"])), r["path"], r["line"] or 0
        ))

        def _hit(row: sqlite3.Row, kind: str) -> dict[str, Any]:
            out = {
                "path": row["path"],
                "line": row["line"],
                "chunk_id": row["chunk_id"],
                "name": row["name"],
                "matched_literal": row["text"],
                "match_kind": kind,
            }
            if kind == "skeleton":
                out["skeleton"] = row["skeleton"]
            return out

        results = [_hit(r, "exact") for r in exact_rows]
        results += [_hit(r, "skeleton") for r in skeleton_rows]

        total = len(results)
        truncated = total > limit
        results = results[:limit]

        notes: list[str] = []
        if truncated:
            notes.append(f"{total - limit} additional match(es) not shown (limit={limit})")
        if fts_degraded:
            notes.append(
                "message could not be parsed for the token/skeleton tier "
                "(unusual characters) — only the direct-substring tier applied"
            )
        if low_confidence_filtered:
            notes.append(
                f"{len(low_confidence_filtered)} low-confidence match(es) filtered "
                "(a short literal appearing only as a coincidental substring, not "
                "the message itself)"
            )
        if skeleton_excluded_length:
            notes.append(
                f"{skeleton_excluded_length} candidate(s) excluded from skeleton "
                f"verification (message over {_SKELETON_MAX_MESSAGE_LEN} chars) "
                "— try a shorter/more specific excerpt"
            )
        if skeleton_excluded_holes:
            notes.append(
                f"{skeleton_excluded_holes} candidate(s) excluded from skeleton "
                f"verification (template has more than {_SKELETON_MAX_HOLES} format holes)"
            )
        if skeleton_budget_exhausted:
            notes.append(
                f"verification budget exhausted for {skeleton_budget_exhausted} "
                "candidate(s) — match may exist"
            )
        # These two caps are silent by nature (nothing was evaluated to
        # count), so announce them whenever the result is thin, or a
        # false "no matches" reads as a clean miss.
        if substring_tier_skipped and total < 3:
            notes.append(
                f"message is {_SUBSTRING_TIER_MAX_MESSAGE_LEN}+ chars — the "
                "direct-substring tier was skipped; a verbatim literal in this "
                "message may have been missed"
            )
        if token_cap_applied and total < 3:
            notes.append(
                f"message has more than {_TOKEN_POOL_CAP} distinct significant "
                f"tokens — only the {_TOKEN_POOL_CAP} longest were used for the "
                "token/skeleton candidate search; a match may have been missed"
            )
        if candidates_truncated:
            # Announced even when a hit was found anyway: a silent cap
            # must always say so, not just when it explains emptiness.
            if results:
                notes.append(
                    f"token/skeleton candidate pool capped at the top "
                    f"{_LITERAL_CANDIDATE_CAP} most relevant literals for "
                    "these tokens — some lower-relevance literals were not considered"
                )
            else:
                notes.append(
                    f"no match in the top {_LITERAL_CANDIDATE_CAP} most relevant "
                    "literals for these tokens — try a more specific fragment"
                )
        elif (not results and not fts_degraded and not skeleton_excluded_length
              and not skeleton_excluded_holes and not skeleton_budget_exhausted
              and not low_confidence_filtered
              and not (substring_tier_skipped and total < 3)
              and not (token_cap_applied and total < 3)):
            notes.append("no literal matches for this message")

        out: dict[str, Any] = {"results": results, "truncated": truncated}
        if notes:
            out["note"] = "; ".join(notes)
        return out
    except sqlite3.OperationalError as e:
        # A lookup must never 500: a capped result must say so, and a
        # crash is just as much a silent failure as an empty answer.
        return {
            "results": [],
            "truncated": False,
            "note": f"literal index unavailable ({e})",
        }
