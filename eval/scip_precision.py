"""eval/scip_precision.py: precision/recall of chunk_refs edges against an
independent SCIP oracle, so scoring isn't circular the way graph-derived
gold would be.

  uv run python eval/scip_precision.py --repo <checkout> --db <chonks.db> \
      [--scip <index.scip>] [--out report.json]

Toolchain auto-fetches into --cache-dir. The scip CLI is never looked up on
PATH: `scip` also names an unrelated MIP solver, and a PATH hit there would
silently run the wrong binary.
"""
import argparse
import json
import os
import platform
import shutil
import sqlite3
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SCIP_CLI_REPO = "sourcegraph/scip"   # redirects to scip-code/scip; kept as the canonical name
SCIP_CLI_VERSION = "v0.9.0"
SCIP_PYTHON_VERSION = "0.6.6"

# edge_type values this eval scores against (mirrors store.py's _HUB_EDGE_TYPES).
# Always reported even at zero edges, so absence reads as "0/0", not a missing row.
EDGE_TYPES = ("calls", "imports", "inherits", "xlang", "associated", "mentions")
TYPED_EDGE_TYPES = frozenset({"calls", "imports", "inherits", "xlang"})

_PLATFORM_TRIPLETS = {
    ("Darwin", "arm64"):  "darwin-arm64",
    ("Darwin", "x86_64"): "darwin-amd64",
    ("Linux", "x86_64"):  "linux-amd64",
    ("Linux", "aarch64"): "linux-arm64",
}


# ----------------------------------------------------------------------
# SCIP JSON -> occurrences (pure)
# ----------------------------------------------------------------------

def scip_line_to_chunk_line(scip_line_0based: int) -> int:
    """SCIP ranges are 0-based, chunks.start_line/end_line are 1-based.
    Every scip-line -> chunk-line conversion goes through this function."""
    return scip_line_0based + 1


def is_local_symbol(symbol: str) -> bool:
    """SCIP locals are 'local <N>'. Match "local " with a space, not
    startswith("local"), so a real symbol named e.g. 'localthing' doesn't
    false-positive as local."""
    return symbol.startswith("local ")


def iter_occurrences(scip_json: dict):
    """Yields (path, line_1based, symbol, is_definition) for every occurrence
    in a `scip print --json` payload. is_definition tests the role bit with
    &, not ==, since symbol_roles is a bitfield that can combine roles."""
    for doc in scip_json.get("documents", []):
        path = doc["relative_path"]
        for occ in doc.get("occurrences", []):
            line0 = occ["range"][0]  # range is [line, ...]: 3 or 4 elements
            roles = occ.get("symbol_roles", 0)
            yield path, scip_line_to_chunk_line(line0), occ["symbol"], bool(roles & 1)


def classify_occurrences(occurrences):
    """Splits occurrences into (definitions, reference_sites, excluded_local).
    Local symbols drop out here; external-reference exclusion happens later
    since it needs the full `definitions` dict across all documents first."""
    definitions: dict[str, list[tuple[str, int]]] = defaultdict(list)
    reference_sites: list[tuple[str, int, str]] = []
    excluded_local = 0
    for path, line1, symbol, is_def in occurrences:
        if is_local_symbol(symbol):
            excluded_local += 1
            continue
        if is_def:
            definitions[symbol].append((path, line1))
        else:
            reference_sites.append((path, line1, symbol))
    return definitions, reference_sites, excluded_local


# ----------------------------------------------------------------------
# chunk mapping (pure)
# ----------------------------------------------------------------------

def build_chunk_index(chunk_rows):
    """chunk_rows: iterable of (id, path, start_line, end_line).
    Returns dict[path -> sorted list of (start_line, end_line, id)]."""
    idx: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for cid, path, start, end in chunk_rows:
        idx[path].append((start, end, cid))
    for path in idx:
        idx[path].sort()
    return idx


class ChunkMapper:
    """Memoized (path, line_1based) -> chunk_id resolver.

    The cache is for correctness, not just speed: the same occurrence gets
    looked up once per reference site and once per definition site of every
    symbol referencing it, so without it an unmapped occurrence would be
    double- or triple-counted."""

    def __init__(self, chunk_index: dict[str, list[tuple[int, int, str]]]):
        self._idx = chunk_index
        self._cache: dict[tuple[str, int], str | None] = {}
        self.excluded_unindexed_file = 0
        self.excluded_unindexed_file_by_file: Counter = Counter()
        self.excluded_multi_chunk_match = 0
        self.excluded_no_chunk_for_line = 0

    def resolve(self, path: str, line1: int) -> str | None:
        key = (path, line1)
        if key in self._cache:
            return self._cache[key]
        result = self._resolve_uncached(path, line1)
        self._cache[key] = result
        return result

    def _resolve_uncached(self, path: str, line1: int) -> str | None:
        chunks = self._idx.get(path)
        if chunks is None:
            self.excluded_unindexed_file += 1
            self.excluded_unindexed_file_by_file[path] += 1
            return None
        matches = [cid for (s, e, cid) in chunks if s <= line1 <= e]
        if not matches:
            self.excluded_no_chunk_for_line += 1
            return None
        if len(matches) > 1:
            # Chunks tile a file with no overlap; never silently pick one
            # if that invariant is broken.
            self.excluded_multi_chunk_match += 1
            return None
        return matches[0]


# ----------------------------------------------------------------------
# ground truth construction (pure)
# ----------------------------------------------------------------------

def _pick_strict_def(ref_path: str, ref_line: int,
                      def_locs: list[tuple[str, int]]) -> tuple[str, int]:
    """Picks one definition per reference for gt_strict: same-file
    definitions win over cross-file ones, then nearest line. Ties keep
    list order since min() is stable."""
    same_file = [d for d in def_locs if d[0] == ref_path]
    pool = same_file if same_file else def_locs
    return min(pool, key=lambda d: abs(d[1] - ref_line))


def build_ground_truth(definitions: dict[str, list[tuple[str, int]]],
                        reference_sites: list[tuple[str, int, str]],
                        mapper: ChunkMapper) -> dict:
    """{(ref_chunk, def_chunk)} pairs over reference/definition occurrences
    that both map to a chunk. gt_fanout is the PRIMARY metric (chunk_refs
    fans out the same way); gt_strict is comparison-only."""
    gt_fanout: set[tuple[str, str]] = set()
    gt_strict: set[tuple[str, str]] = set()
    excluded_external = 0
    excluded_intra_chunk_fanout = 0
    excluded_intra_chunk_strict = 0

    for path, line1, symbol in reference_sites:
        def_locs = definitions.get(symbol)
        if not def_locs:
            excluded_external += 1
            continue
        ref_chunk = mapper.resolve(path, line1)
        if ref_chunk is None:
            continue

        for dpath, dline in def_locs:
            def_chunk = mapper.resolve(dpath, dline)
            if def_chunk is None:
                continue
            if def_chunk == ref_chunk:
                excluded_intra_chunk_fanout += 1
                continue
            gt_fanout.add((ref_chunk, def_chunk))

        dpath, dline = _pick_strict_def(path, line1, def_locs)
        def_chunk = mapper.resolve(dpath, dline)
        if def_chunk is None:
            continue
        if def_chunk == ref_chunk:
            excluded_intra_chunk_strict += 1
            continue
        gt_strict.add((ref_chunk, def_chunk))

    return {
        "gt_fanout": gt_fanout,
        "gt_strict": gt_strict,
        "excluded_external": excluded_external,
        "excluded_intra_chunk_fanout": excluded_intra_chunk_fanout,
        "excluded_intra_chunk_strict": excluded_intra_chunk_strict,
    }


# ----------------------------------------------------------------------
# scoring (pure)
# ----------------------------------------------------------------------

def score_precision_recall(chunk_refs_rows, chunk_id_to_path: dict[str, str],
                            mapped_paths: set[str], gt_fanout: set[tuple[str, str]]) -> dict:
    """precision(edge_type): edges touching a file the SCIP oracle doesn't
    cover are excluded from the denominator, not counted as wrong. recall is
    reported for any/typed_only/typed_plus_associated against gt_fanout."""
    by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for from_id, to_id, edge_type in chunk_refs_rows:
        by_type[edge_type].append((from_id, to_id))

    precision = {}
    for edge_type in EDGE_TYPES:
        hit = 0
        scoreable = 0
        unscoreable = 0
        for fid, tid in by_type.get(edge_type, ()):
            fpath = chunk_id_to_path.get(fid)
            tpath = chunk_id_to_path.get(tid)
            if fpath not in mapped_paths or tpath not in mapped_paths:
                unscoreable += 1
                continue
            scoreable += 1
            if (fid, tid) in gt_fanout:
                hit += 1
        precision[edge_type] = {
            "hit": hit, "scoreable": scoreable, "unscoreable": unscoreable,
            "ratio": (hit / scoreable) if scoreable else None,
        }

    any_pairs: set[tuple[str, str]] = set()
    typed_pairs: set[tuple[str, str]] = set()
    typed_assoc_pairs: set[tuple[str, str]] = set()
    for edge_type, pairs in by_type.items():
        s = set(pairs)
        any_pairs |= s
        if edge_type in TYPED_EDGE_TYPES:
            typed_pairs |= s
        if edge_type in TYPED_EDGE_TYPES or edge_type == "associated":
            typed_assoc_pairs |= s

    def _recall(hit_pairs: set[tuple[str, str]]) -> dict:
        total = len(gt_fanout)
        hit = len(gt_fanout & hit_pairs)
        return {"hit": hit, "total": total, "ratio": (hit / total) if total else None}

    recall = {
        "any": _recall(any_pairs),
        "typed_only": _recall(typed_pairs),
        "typed_plus_associated": _recall(typed_assoc_pairs),
    }
    return {"precision": precision, "recall": recall}


# ----------------------------------------------------------------------
# toolchain (network/subprocess, not exercised by tests)
# ----------------------------------------------------------------------

def _scip_platform_triplet() -> str:
    key = (platform.system(), platform.machine())
    if key not in _PLATFORM_TRIPLETS:
        raise RuntimeError(
            f"no scip CLI release known for platform {key} — see "
            f"https://github.com/{SCIP_CLI_REPO}/releases for available assets"
        )
    return _PLATFORM_TRIPLETS[key]


def ensure_scip_cli(cache_dir: Path, version: str = SCIP_CLI_VERSION) -> Path:
    """Downloads the scip CLI into cache_dir; never looked up on PATH
    (name collides with an unrelated MIP solver, see module docstring)."""
    bin_path = cache_dir / f"scip-cli-{version}" / "scip"
    if bin_path.exists():
        return bin_path
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    triplet = _scip_platform_triplet()
    url = f"https://github.com/{SCIP_CLI_REPO}/releases/download/{version}/scip-{triplet}.tar.gz"
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "scip.tar.gz"
        urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extract("scip", td, filter="data")
        shutil.move(str(Path(td) / "scip"), bin_path)
    mode = bin_path.stat().st_mode
    bin_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


def run_scip_python(repo: Path, out_scip: Path, cache_dir: Path,
                     version: str = SCIP_PYTHON_VERSION) -> None:
    """Generates a .scip index for `repo` via `npm exec`. NPM_CONFIG_PREFIX
    is pinned to cache_dir so this never touches system npm global state."""
    npm_prefix = cache_dir / "npm-prefix"
    (npm_prefix / "lib").mkdir(parents=True, exist_ok=True)
    (npm_prefix / "bin").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["npm_config_prefix"] = str(npm_prefix)
    cmd = [
        "npm", "exec", "--yes", f"@sourcegraph/scip-python@{version}", "--",
        "index", "--project-name", "eval", "--project-version", "0.0.1",
        "--output", str(out_scip), ".",
    ]
    subprocess.run(cmd, cwd=str(repo), env=env, check=True)


def load_scip_json(scip_cli: Path, scip_path: Path) -> dict:
    proc = subprocess.run(
        [str(scip_cli), "print", "--json", str(scip_path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def ensure_scip_index(repo: Path, scip_arg: str | None, cache_dir: Path) -> Path:
    if scip_arg:
        return Path(scip_arg).resolve()
    cached = repo.parent / f"{repo.name}.scip"
    if not cached.exists():
        run_scip_python(repo, cached, cache_dir)
    return cached


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------

def evaluate(repo: str, db: str, scip_arg: str | None, cache_dir: Path) -> dict:
    repo_p = Path(repo).resolve()
    scip_path = ensure_scip_index(repo_p, scip_arg, cache_dir)
    scip_cli = ensure_scip_cli(cache_dir)
    scip_json = load_scip_json(scip_cli, scip_path)

    occurrences = list(iter_occurrences(scip_json))
    definitions, reference_sites, excluded_local = classify_occurrences(occurrences)
    doc_paths = {doc["relative_path"] for doc in scip_json.get("documents", [])}

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)  # read-only, never writes to db
    con.row_factory = sqlite3.Row
    try:
        chunk_rows = con.execute(
            "SELECT id, path, start_line, end_line FROM chunks"
        ).fetchall()
        refs_rows = con.execute(
            "SELECT from_id, to_id, edge_type FROM chunk_refs"
        ).fetchall()
    finally:
        con.close()

    chunk_id_to_path = {r["id"]: r["path"] for r in chunk_rows}
    chunk_paths = set(chunk_id_to_path.values())
    chunk_index = build_chunk_index(
        (r["id"], r["path"], r["start_line"], r["end_line"]) for r in chunk_rows
    )
    mapper = ChunkMapper(chunk_index)
    gt = build_ground_truth(definitions, reference_sites, mapper)
    mapped_paths = doc_paths & chunk_paths

    chunk_refs_rows = [(r["from_id"], r["to_id"], r["edge_type"]) for r in refs_rows]
    scored = score_precision_recall(chunk_refs_rows, chunk_id_to_path, mapped_paths, gt["gt_fanout"])

    n_def_occ = sum(1 for *_, is_def in occurrences if is_def)
    n_ref_occ = len(occurrences) - n_def_occ

    return {
        "repo": str(repo_p),
        "db": str(Path(db).resolve()),
        "scip_index": str(scip_path),
        "counts": {
            "documents": len(scip_json.get("documents", [])),
            "occurrences": len(occurrences),
            "definition_occurrences": n_def_occ,
            "reference_occurrences": n_ref_occ,
            "chunks": len(chunk_rows),
            "chunk_refs_edges": len(chunk_refs_rows),
            "mapped_files": len(mapped_paths),
            "scip_only_files": len(doc_paths - chunk_paths),
            "chunks_only_files": len(chunk_paths - doc_paths),
        },
        "exclusions": {
            "excluded_local": excluded_local,
            "excluded_external": gt["excluded_external"],
            "excluded_unindexed_file_occurrences": mapper.excluded_unindexed_file,
            "excluded_unindexed_file_files": dict(mapper.excluded_unindexed_file_by_file),
            "excluded_multi_chunk_match": mapper.excluded_multi_chunk_match,
            "excluded_no_chunk_for_line": mapper.excluded_no_chunk_for_line,
            "excluded_intra_chunk_fanout": gt["excluded_intra_chunk_fanout"],
            "excluded_intra_chunk_strict": gt["excluded_intra_chunk_strict"],
        },
        "ground_truth": {
            "gt_fanout": len(gt["gt_fanout"]),
            "gt_strict": len(gt["gt_strict"]),
        },
        "precision": scored["precision"],
        "recall": scored["recall"],
    }


def print_report(report: dict) -> None:
    c, x, g = report["counts"], report["exclusions"], report["ground_truth"]
    print(f"\n=== SCIP precision/recall :: {report['repo']} ===")
    print(f"db={report['db']}")
    print(f"scip_index={report['scip_index']}")
    print(f"documents={c['documents']} occurrences={c['occurrences']} "
          f"(defs={c['definition_occurrences']} refs={c['reference_occurrences']}) "
          f"chunks={c['chunks']} chunk_refs_edges={c['chunk_refs_edges']}")
    print(f"mapped_files={c['mapped_files']} scip_only_files={c['scip_only_files']} "
          f"chunks_only_files={c['chunks_only_files']}")
    print("exclusions: "
          f"local={x['excluded_local']} external={x['excluded_external']} "
          f"unindexed_file_occ={x['excluded_unindexed_file_occurrences']} "
          f"({len(x['excluded_unindexed_file_files'])} files) "
          f"multi_chunk_match={x['excluded_multi_chunk_match']} "
          f"no_chunk_for_line={x['excluded_no_chunk_for_line']} "
          f"intra_chunk[fanout={x['excluded_intra_chunk_fanout']} "
          f"strict={x['excluded_intra_chunk_strict']}]")
    print(f"ground truth: gt_fanout={g['gt_fanout']} gt_strict={g['gt_strict']}  "
          f"(scoring below uses gt_fanout)")

    print(f"\n  {'edge_type':12} {'hit':>7} {'scoreable':>10} {'unscoreable':>12} {'precision':>10}")
    for edge_type in EDGE_TYPES:
        p = report["precision"][edge_type]
        ratio = f"{p['ratio']:.3f}" if p["ratio"] is not None else "  n/a"
        print(f"  {edge_type:12} {p['hit']:7d} {p['scoreable']:10d} {p['unscoreable']:12d} {ratio:>10}")

    print(f"\n  {'recall':20} {'hit':>7} {'total':>7} {'ratio':>8}")
    for key in ("any", "typed_only", "typed_plus_associated"):
        r = report["recall"][key]
        ratio = f"{r['ratio']:.3f}" if r["ratio"] is not None else "  n/a"
        print(f"  {key:20} {r['hit']:7d} {r['total']:7d} {ratio:>8}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="git checkout that was indexed into --db")
    ap.add_argument("--db", required=True, help="chonks sqlite DB (read-only)")
    ap.add_argument("--scip", default=None,
                    help="pre-built .scip index; if omitted, one is generated and "
                         "cached next to --repo as <repo>.scip")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "chonks-scip-eval"),
                    help="toolchain download cache (scip CLI binary + npm prefix)")
    args = ap.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    report = evaluate(args.repo, args.db, args.scip, cache_dir)
    print_report(report)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
