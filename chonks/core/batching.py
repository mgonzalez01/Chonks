"""Fixed-size slices of a sequence, for SQL statements with a bound-parameter limit."""

from __future__ import annotations

from collections.abc import Iterator, Sequence


def batched(items: Sequence, n: int) -> Iterator[Sequence]:
    for i in range(0, len(items), n):
        yield items[i:i + n]
