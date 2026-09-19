"""The persisted macro vocabulary and unhealable-content memo: limits, fingerprint, load, persist."""

import hashlib
import json
import logging

logger = logging.getLogger("chonks.chunker")


# A macro must heal >= this many files to be persisted; otherwise a one-off
# false-admit from a single weird file poisons every future index.
_MACRO_PERSIST_MIN_FILES = 2

# Bound on the persisted unhealable-content hash set. NOT cleared on --force,
# since surviving repeat force-reindexes of a partly-unhealable corpus is the
# point; FIFO-capped so it can't grow unbounded.
_UNHEALABLE_HASH_CAP = 100_000


def _vocab_fingerprint(vocab: set[str]) -> str:
    """Invalidates the unhealable-content memo when the vocab changes: a file
    healing depends on which macros are pre-blanked, so a memo built under a
    narrower vocab must be dropped once the vocab grows."""
    return hashlib.sha256("\n".join(sorted(vocab)).encode()).hexdigest()


def load_macro_memo(store, macros):
    """Reads the macro vocab and the unhealable-content memo from meta. Returns
    (persisted, vocab, unhealable_order, unhealable_hashes, new_unhealable)."""
    # Persisted macro vocab, pre-blanked so self-heal's trial-reparse loop
    # doesn't rediscover it every run (measured ~37x parse overhead avoided).
    persisted: set[str] = set(json.loads(store.get_meta("macro_vocab") or "[]"))
    # Seed is applied every run but NOT persisted, so a typo never sticks.
    vocab: set[str] = (persisted | set(macros)) if macros else set(persisted)

    # Content hashes a prior run's heal sweep admitted nothing for; skips the
    # trial-reparse sweep for them (vocab is still pre-blanked). Order kept
    # for the FIFO cap at persist time.
    unhealable_order: list[str] = list(json.loads(store.get_meta("unhealable_hashes") or "[]"))
    unhealable_hashes: set[str] = set(unhealable_order)
    new_unhealable = False  # only rewrite meta if something changed this run

    # Drop the memo if the persisted vocab changed since it was built (see
    # _vocab_fingerprint), else a file could stay wrongly memoized unhealable.
    # Compared against `persisted`, not `vocab`: `vocab` also carries this
    # run's non-persisted seed, which would cause spurious invalidation.
    if unhealable_hashes:
        stored_fp  = store.get_meta("unhealable_vocab_fingerprint")
        current_fp = _vocab_fingerprint(persisted)
        if stored_fp != current_fp:
            logger.info(
                "macro vocab changed since unhealable memo was built — "
                "clearing %d memoized entries", len(unhealable_hashes)
            )
            unhealable_order = []
            unhealable_hashes = set()
            new_unhealable = True  # force the (now-empty) memo + new fingerprint to persist
    return persisted, vocab, unhealable_order, unhealable_hashes, new_unhealable


def persist_macro_memo(store, persisted, macro_file_counts, unhealable_order, new_unhealable):
    """Writes the macro vocab and, when it changed, the unhealable-content memo to meta."""
    # `persisted` carried forward unconditionally: its members were
    # pre-blanked, so they can't reappear in macro_file_counts (see
    # _MACRO_PERSIST_MIN_FILES for the filter on new ones).
    qualifying = {m for m, c in macro_file_counts.items()
                  if c >= _MACRO_PERSIST_MIN_FILES}
    if qualifying:
        # Persist only the durable set + newly-qualifying macros, NOT the runtime
        # seed (which is layered on at load time), so a config seed never bakes in.
        store.set_meta("macro_vocab", json.dumps(sorted(persisted | qualifying)))

    # Persist the unhealable-content memo. FIFO-capped, not cleared on
    # --force, see _UNHEALABLE_HASH_CAP. Order is oldest-first, so a plain
    # negative-index slice keeps the most-recently-seen entries when over cap.
    if new_unhealable:
        if len(unhealable_order) > _UNHEALABLE_HASH_CAP:
            unhealable_order = unhealable_order[-_UNHEALABLE_HASH_CAP:]
        store.set_meta("unhealable_hashes", json.dumps(unhealable_order))
        # Fingerprint of the vocab this memo was (re)built under, the same
        # `persisted | qualifying` set persisted above as meta['macro_vocab'],
        # so the next run's load-time check (above) can detect vocab growth.
        store.set_meta("unhealable_vocab_fingerprint",
                       _vocab_fingerprint(persisted | qualifying))
