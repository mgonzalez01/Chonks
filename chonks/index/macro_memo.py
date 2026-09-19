"""Limits and fingerprint of the persisted macro vocabulary and unhealable-content memo."""

import hashlib


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
