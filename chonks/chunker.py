"""
chunker.py: command line of the indexer. The pipeline (scan, parse, embed,
store) is in chonks/index/pipeline.py. AST segmentation lives in chunking.py,
the embedding client in embedder.py. Supported languages and fallback rules
are documented in DOCS.md.
"""

import json
import logging
import os
import sys
from pathlib import Path

import httpx
from tqdm import tqdm

from chonks.chunking import dominance_warning, family_breakdown
from chonks.core.paths import (
    _dir_should_prune,
    _dotdir_prefix,
)
from chonks.embedder import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_URL,
    EMBED_BATCH,
    EMBED_INFLIGHT,
    EMBED_WATCHDOG_SECS,
    Embedder,
)
from chonks.index.admission import DEFAULT_FALLBACK_EXTENSIONS, _file_hash
from chonks.index.embed_retry import _embed_isolating, _embed_one_isolating
from chonks.index.pipeline import EmbedderDownError, NoProgressError, index_paths, reembed_all
from chonks.index.rows import _chunk_id
from chonks.repomap import build_neighbors, build_refs, persist_pagerank
from chonks.store import Store
from chonks.summaries import build_folder_summaries

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Index source files into a Chonks sqlite-vec database."
    )
    parser.add_argument(
        "paths", nargs="*", metavar="PATH",
        help="Files or directories to index (recursively). Not required with --reembed.",
    )
    parser.add_argument(
        "--reembed", action="store_true",
        help="Re-embed all chunks already in the DB without re-parsing source files. "
             "Updates vectors only; use after changing the embedding format.",
    )
    parser.add_argument(
        "--suggest-subsystems", action="store_true",
        help="Cluster folder summaries by cosine similarity and print suggested "
             "`subsystems` groupings ready to paste into config.json. No indexing.",
    )
    parser.add_argument(
        "--rebuild-graphs", action="store_true",
        help="Re-run the post-index passes (build_refs + build_neighbors + "
             "build_folder_summaries) against the existing DB without re-parsing "
             "or re-embedding any source files. Use to recover after a graph-build "
             "failure that left chunks indexed but graphs empty.",
    )
    parser.add_argument(
        "--rebuild-knn", action="store_true",
        help="Like --rebuild-graphs but skips build_refs: re-runs only "
             "build_neighbors + PageRank + build_folder_summaries against the existing "
             "DB. build_refs dominates wall clock at scale (~25min/385k chunks vs ~8min "
             "for kNN) and is unneeded when chunk_refs is already persisted and only "
             "chunk_neighbors is stale/empty. Falls back to the full --rebuild-graphs "
             "chain, with a warning, if chunk_refs is empty (nothing to reuse).",
    )
    parser.add_argument(
        "--subsystem-threshold", type=float, default=None, metavar="DIST",
        help="Cosine-distance cut threshold for --suggest-subsystems (default: 0.30).",
    )
    parser.add_argument(
        "--db", default=None, metavar="FILE",
        help="Path to the sqlite DB file (created if absent). "
             "Default: config.json's `db` key, else .db/chonks.db",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-index all files even if unchanged.",
    )
    parser.add_argument(
        "--root", default=None, metavar="DIR",
        help="Root directory to store paths relative to (for portable DBs). "
             "Defaults to the common ancestor of the indexed paths.",
    )
    parser.add_argument(
        "--embed-url", default=None, metavar="URL",
        help=f"Embedding server URL. Precedence: this flag > config.embed_url > "
             f"{DEFAULT_EMBED_URL}",
    )
    parser.add_argument(
        "--embed-model", default=None, metavar="NAME",
        help=f"Embedding model name (sent in /v1/embeddings request body). "
             f"Precedence: this flag > config.embed_model > {DEFAULT_EMBED_MODEL}",
    )
    parser.add_argument(
        "--embed-query-prefix", default=None, metavar="STR",
        help="Instruction prefix prepended to QUERY text before embedding; overrides "
             "the per-model preset. Pass '' to force no prefix. Code models like "
             "jina-code-embeddings need asymmetric query/doc prefixes; Qwen3-Embedding "
             "needs a query-side instruction only. The preset handles both.",
    )
    parser.add_argument(
        "--embed-doc-prefix", default=None, metavar="STR",
        help="Instruction prefix prepended to DOCUMENT (chunk) text at index time; "
             "overrides the per-model preset. Pass '' to force no prefix.",
    )
    parser.add_argument(
        "--embed-batch", default=None, type=int, metavar="N",
        help=f"Chunks per embedding request. Precedence: this flag > config.embed_batch "
             f"> {EMBED_BATCH}. Raise for small/fast embedders to keep the GPU fed.",
    )
    parser.add_argument(
        "--embed-inflight", default=None, type=int, metavar="N",
        help=f"Concurrent in-flight embedding batches. Precedence: this flag > "
             f"config.embed_inflight > {EMBED_INFLIGHT}. Raise (with llama-server "
             f"--parallel) for small/fast embedders; the default suits a heavy 4B.",
    )
    parser.add_argument(
        "--config", default=None, metavar="FILE",
        help="Path to config.json. Used to load `exclude`, `include`, and `codebase` root.",
    )
    parser.add_argument(
        "--exclude", action="append", default=None, metavar="PREFIX",
        help="Path prefix (relative to root) to skip during scan. Repeat for multiple.",
    )
    parser.add_argument(
        "--include", action="append", default=None, metavar="PREFIX",
        help="Path prefix that overrides matching excludes when strictly more "
             "specific (e.g. --exclude tmp/ --include tmp/git/a/). Repeat for multiple.",
    )
    parser.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging verbosity (default: info)",
    )
    args = parser.parse_args(argv)

    # Route log records through tqdm.write() so they appear above progress bars
    # without tearing through the in-place bar redraw.
    class _TqdmHandler(logging.StreamHandler):
        def emit(self, record: logging.LogRecord) -> None:
            tqdm.write(self.format(record), file=sys.stderr)

    handler = _TqdmHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S"
    ))
    logging.root.setLevel(args.log_level.upper())
    logging.root.addHandler(handler)

    # ------------------------------------------------------------------ config
    config: dict = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.warning("Failed to load %s: %s", args.config, e)
    else:
        # Auto-discover config.json in cwd
        for candidate in ("config.json", ".chonks.json"):
            if Path(candidate).exists():
                try:
                    with open(candidate, "r") as f:
                        config = json.load(f)
                    logger.info("Loaded config from %s", candidate)
                    break
                except Exception as e:
                    logger.warning("Failed to parse %s: %s", candidate, e)

    # Merge CLI excludes/includes with config (CLI appended, both honored)
    exclude_list: list[str] = list(config.get("exclude") or [])
    if args.exclude:
        exclude_list.extend(args.exclude)
    include_list: list[str] = list(config.get("include") or [])
    if args.include:
        include_list.extend(args.include)

    macros_seed: set[str] = set(config.get("macros") or [])
    # Config-only settings (no CLI flag); see DOCS.md's config.json reference
    # for each. None means "not set", so index_paths applies its own default.
    fallback_extensions = config.get("fallback_extensions")
    data_blob_size_limit = config.get("data_blob_size_limit")
    edge_type_weights = config.get("edge_type_weights")
    cap_mentions_fanout = bool(config.get("cap_mentions_fanout", False))
    associated_top_frac = float(config.get("associated_top_frac", 0.02))
    no_progress_timeout = config.get("embed_watchdog_secs") or EMBED_WATCHDOG_SECS

    # Validated here, not left to _Corpus's silent numpy fallback, so a config
    # typo doesn't silently disable acceleration. Env var wins when set.
    knn_backend = config.get("knn_backend")
    if knn_backend is not None:
        if knn_backend not in ("auto", "numpy", "mlx", "cuda"):
            parser.error(
                f"config.json knn_backend must be auto|numpy|mlx|cuda, got {knn_backend!r}"
            )
        if knn_backend != "auto" and not os.environ.get("CHONKS_KNN_BACKEND"):
            os.environ["CHONKS_KNN_BACKEND"] = knn_backend

    if args.rebuild_graphs and args.rebuild_knn:
        parser.error("--rebuild-graphs and --rebuild-knn are mutually exclusive")

    if (not args.reembed and not args.suggest_subsystems and not args.rebuild_graphs
            and not args.rebuild_knn and not args.paths):
        parser.error(
            "provide paths to index, or use --reembed / --suggest-subsystems / "
            "--rebuild-graphs / --rebuild-knn"
        )

    # Same precedence as embedder settings below, and the same key `chonks
    # serve` honors, so both agree on which db they read/write.
    args.db = args.db or config.get("db") or ".db/chonks.db"

    # Embedder configuration: CLI flag > config key > module default. Resolved
    # once here and threaded down, so there's no module-level mutable state.
    embed_url   = args.embed_url   or config.get("embed_url")   or DEFAULT_EMBED_URL
    embed_model = args.embed_model or config.get("embed_model") or DEFAULT_EMBED_MODEL
    # Prefixes: CLI flag (incl. "") > config key > per-model preset (resolved inside
    # Embedder). None here means "not set on flag or config" => let the preset apply.
    query_prefix = (args.embed_query_prefix if args.embed_query_prefix is not None
                    else config.get("embed_query_prefix"))
    doc_prefix   = (args.embed_doc_prefix if args.embed_doc_prefix is not None
                    else config.get("embed_doc_prefix"))
    embedder    = Embedder(embed_url, embed_model,
                           query_prefix=query_prefix, doc_prefix=doc_prefix)
    # Indexing concurrency: CLI flag > config key > default. Raise for small/fast embedders.
    embed_batch    = (args.embed_batch if args.embed_batch is not None
                      else (config.get("embed_batch") or EMBED_BATCH))
    embed_inflight = (args.embed_inflight if args.embed_inflight is not None
                      else (config.get("embed_inflight") or EMBED_INFLIGHT))

    if args.rebuild_graphs or args.rebuild_knn:
        with Store(args.db) as store:
            print(f"DB: {args.db}")
            print(f"Embedding server: {embedder.url} ({embedder.model}) "
                  f"(only needed if folder summaries need refresh)\n")
            skip_refs = args.rebuild_knn
            if skip_refs and store.count_refs() == 0:
                print("chunk_refs is empty — --rebuild-knn has nothing to reuse; "
                      "falling back to the full --rebuild-graphs chain.")
                skip_refs = False
            if skip_refs:
                print("Skipping build_refs (--rebuild-knn): chunk_refs already persisted.")
            else:
                ref_count = build_refs(
                    store, cap_mentions=cap_mentions_fanout,
                    associated_top_frac=associated_top_frac,
                )
                print(f"Rebuilt {ref_count} cross-reference edges.")
            neighbor_count = build_neighbors(store)
            print(f"Built {neighbor_count} k-NN neighbour edges.")
            pagerank_count = persist_pagerank(
                store, force=True, edge_type_weights=edge_type_weights,
            )
            print(f"Persisted {pagerank_count} PageRank scores.")
            hierarchy = store.rebuild_hierarchy()
            print(f"Built hierarchy: {hierarchy['nodes']} nodes, "
                  f"{hierarchy['edges']} contains edges.")
            try:
                with httpx.Client() as client:
                    summary_stats = build_folder_summaries(
                        store,
                        lambda texts: embedder.embed_documents(texts, client),
                    )
                print(
                    f"Folder summaries: refreshed {summary_stats['refreshed']}, "
                    f"pruned {summary_stats['pruned']}."
                )
            except Exception as e:
                print(f"Folder summary regeneration failed (embedding server reachable?): {e}")
            print(json.dumps(store.stats(), indent=2))
        sys.exit(0)

    if args.suggest_subsystems:
        from chonks.summaries import (
            DEFAULT_SUBSYSTEM_DISTANCE_THRESHOLD,
            suggest_subsystems,
        )
        threshold = args.subsystem_threshold or DEFAULT_SUBSYSTEM_DISTANCE_THRESHOLD
        with Store(args.db) as store:
            clusters = suggest_subsystems(store, distance_threshold=threshold)
        if not clusters:
            print("No subsystem clusters found at threshold "
                  f"{threshold:.3f}. (Either fewer than 2 folders are indexed, "
                  "or all are too dissimilar.)", file=sys.stderr)
        else:
            # Suggested format mirrors the `subsystems` map shape from config.json;
            # names are placeholders the user should rewrite to be meaningful.
            output = {
                f"cluster_{i:02d}": paths
                for i, paths in enumerate(clusters, 1)
            }
            print(json.dumps({"subsystems": output}, indent=2))
        sys.exit(0)

    if args.reembed:
        with Store(args.db) as store:
            print(f"DB: {args.db}")
            print(f"Embedding server: {embedder.url} ({embedder.model})\n")
            result = reembed_all(store, embedder, embed_batch=embed_batch)
            print(f"\nDone — reembedded {result['reembedded']}, errors {result['errors']}")
            print(json.dumps(store.stats(), indent=2))
    else:
        # Resolve root: explicit flag > config.codebase > common ancestor of paths
        if args.root:
            root = Path(args.root).resolve()
        elif config.get("codebase"):
            root = Path(config["codebase"]).resolve()
        else:
            resolved = [Path(p).resolve() for p in args.paths]
            try:
                parts_list = [p.parts for p in resolved]
                common = []
                for parts in zip(*parts_list):
                    if len(set(parts)) == 1:
                        common.append(parts[0])
                    else:
                        break
                root = Path(*common) if common else resolved[0].parent
            except Exception:
                root = resolved[0].parent if resolved else Path.cwd()

        with Store(args.db) as store:
            print(f"DB: {args.db}")
            print(f"Root: {root}")
            print(f"Embedding server: {embedder.url} ({embedder.model})")
            print(f"Paths: {args.paths}\n")

            result = index_paths(
                args.paths, store, embedder,
                root=root,
                force=args.force,
                exclude=exclude_list,
                include=include_list,
                embed_batch=embed_batch,
                embed_inflight=embed_inflight,
                macros=macros_seed,
                fallback_extensions=fallback_extensions,
                data_blob_size_limit=data_blob_size_limit,
                edge_type_weights=edge_type_weights,
                cap_mentions_fanout=cap_mentions_fanout,
                associated_top_frac=associated_top_frac,
                no_progress_timeout=no_progress_timeout,
            )

            cov_note = ""
            if result["low_coverage_files"]:
                cov_note = (f" (worst: {result['worst_coverage_file']} at "
                            f"{int(result['worst_coverage'] * 100)}%)")
            print(
                f"\nDone in {result['total_elapsed_s']}s "
                f"(embed {result['embed_elapsed_s']}s, "
                + (f"fts {result['fts_elapsed_s']}s, " if result['fts_elapsed_s'] else "")
                + f"refs {result['refs_elapsed_s']}s, "
                f"knn {result['knn_elapsed_s']}s, pagerank {result['pagerank_elapsed_s']}s, "
                f"hierarchy {result['hierarchy_elapsed_s']}s, "
                f"summaries {result['summaries_elapsed_s']}s) "
                f"— indexed {result['indexed']} files, "
                f"{result['chunks_embedded']} chunks ({result['chunks_per_s']} chunks/s); "
                f"skipped {result['skipped']}, pruned {result['pruned']}, "
                f"dirs_pruned {result['dirs_pruned']}, "
                f"oversize {result['oversize_chunks']} (in {result['oversize_files']} files), "
                f"parse_errors {result['parse_error_files']}, "
                f"macro_healed {result['macro_healed_files']}, "
                f"error_salvaged {result['error_salvaged_files']}, "
                f"low_coverage {result['low_coverage_files']}{cov_note}, "
                f"batch_failures {result['batch_failures']}, truncated {result['truncated']}, "
                f"errors {result['errors']}, "
                f"unsupported_ext_skipped {result['unsupported_ext_skipped']}, "
                f"data_blob_skipped {result['data_blob_skipped']}, "
                f"literals_dropped {result['literals_dropped']} "
                f"(in {result['literals_capped_chunks']} chunks)"
            )
            print(json.dumps(store.stats(), indent=2))

            # Same rule as doctor.py's breakdown; run here so it surfaces
            # before the *next* run wastes an embed budget on it.
            family_stats = family_breakdown(store.path_family_rows())
            total_family_chunks = sum(s.chunks for s in family_stats)
            warning = dominance_warning(family_stats, total_family_chunks)
            if warning:
                print(f"\n{warning}")


if __name__ == "__main__":
    main()
