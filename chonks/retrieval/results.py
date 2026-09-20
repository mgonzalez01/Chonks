"""File ranking, near-duplicate-wall detection, and result formatting."""

from collections import Counter
from typing import Any

import numpy as np

# TAU: cosine to count as "same wall". Both a naively high (>=0.9) and low
# (<=0.75) value fail: confirmed dupes rarely exceed ~0.9, and single-topic
# results are correlated enough to false-positive at low thresholds.
NEAR_DUP_TAU = 0.85
NEAR_DUP_WALL_SHARE_THRESHOLD = 0.6
# Below this many vector-bearing chunks, "largest connected component" is
# noise, not signal.
NEAR_DUP_MIN_RESULTS = 5
# O(n^2) pairwise cosine; cap keeps it cheap even at TOP_K_MAX=200.
NEAR_DUP_MAX_CHUNKS = 50


def rank_files(chunks: list[dict[str, Any]], top_m: int = 3) -> list[dict[str, Any]]:
    """File score = best chunk's `_score` (max, not sum). Summing double-
    counts correlated same-file near-duplicate evidence and measurably hurt
    accuracy in testing. Order: score DESC, then first-appearance."""
    scores: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    best_rank: dict[str, int] = {}
    order: list[str] = []
    for i, c in enumerate(chunks):
        path = c.get("path", "")
        score = c.get("_score")
        score = score if score is not None else 0.0
        if path not in scores:
            scores[path] = []
            counts[path] = 0
            best_rank[path] = i + 1
            order.append(path)
        scores[path].append(score)
        counts[path] += 1

    ranked = []
    for path in order:
        ranked.append({
            "path":      path,
            "score":     max(scores[path]),
            "n_chunks":  counts[path],
            "best_rank": best_rank[path],
        })

    # Final tiebreak: input (first-appearance) order. n_chunks is payload
    # only, not a scoring factor.
    ranked.sort(key=lambda f: -f["score"])
    return ranked


def detect_near_dup_wall(store, chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Detect a near-duplicate "wall": largest connected component over
    pairwise cosine >= NEAR_DUP_TAU. Observation only, never reorders or
    drops chunks. Returns None below NEAR_DUP_MIN_RESULTS vectors."""
    ids = [c["id"] for c in chunks[:NEAR_DUP_MAX_CHUNKS]]
    vectors = store.get_vectors_for_chunks(ids)
    ids = [i for i in ids if i in vectors]
    n = len(ids)
    if n < NEAR_DUP_MIN_RESULTS:
        return None

    mat = np.stack([vectors[i] for i in ids])
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0  # zero vector: similarity 0 to everything, avoid /0
    normed = mat / norms[:, None]
    sims = normed @ normed.T

    # Union-find over the TAU-threshold graph. n is capped at
    # NEAR_DUP_MAX_CHUNKS, so the naive O(n^2) edge scan is negligible.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= NEAR_DUP_TAU:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    component_sizes = Counter(find(i) for i in range(n))
    wall_size = max(component_sizes.values())
    return {
        "wall_share": wall_size / n,
        "wall_size": wall_size,
        "tau": NEAR_DUP_TAU,
    }


def format_results(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "_No results._"
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c['path']}:{c['start_line']}-{c['end_line']}"
        if c.get("name"):
            header += f"  ({c['name']})"
        if "_score" in c:
            # Hybrid (RRF) result: show fused score and per-branch ranks.
            rank_bits: list[str] = []
            if c.get("_rank_semantic") is not None:
                rank_bits.append(f"sem#{c['_rank_semantic']}")
            if c.get("_rank_fts") is not None:
                rank_bits.append(f"fts#{c['_rank_fts']}")
            ranks = f"  [{','.join(rank_bits)}]" if rank_bits else ""
            header += f"  rrf={c['_score']:.4f}{ranks}"
        elif "_blended_score" in c:
            # Folder-blend re-ranked semantic result: show blend components
            # so the consumer can see why a chunk surfaced (chunk vs folder).
            header += (
                f"  blend={c['_blended_score']:.4f} "
                f"[chunk={c.get('_chunk_sim', 0.0):.3f},"
                f"folder={c.get('_folder_sim', 0.0):.3f}]"
            )
        else:
            # Explicit None check, not `distance or fts_rank`: distance
            # == 0.0 (perfect match) is falsy and would hide the score.
            if (d := c.get("distance")) is not None:
                header += f"  score={d:.4f}"
            elif (r := c.get("fts_rank")) is not None:
                header += f"  score={r:.4f}"
        parts.append(f"{header}\n```\n{c['content']}\n```")
    return "\n\n".join(parts)
