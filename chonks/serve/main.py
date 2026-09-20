"""FastAPI server for search/research/index/status. Run: chonks serve
[--db PATH] [--port PORT] [--host ADDR] [--config PATH]. Endpoint list: DOCS.md.
No `projects` config map: `project` must be omitted or "default" for the legacy single-DB path."""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

import uvicorn

from chonks.core.config import load_config, resolve_codebase
from chonks.embed.client import probe_embedder
from chonks.embed.client import DEFAULT_EMBED_MODEL, DEFAULT_EMBED_URL, Embedder
from chonks.index.embed_retry import EMBED_BATCH, EMBED_INFLIGHT
from chonks.serve.app import app
from chonks.serve.projects import (
    DEFAULT_PROJECT,
    _projects,
)

logger = logging.getLogger("chonks.server")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Chonks server")
    parser.add_argument("--db",        default=".db/chonks.db",
                        help="Path to sqlite DB file (used for the default project; "
                             "ignored if config defines a 'projects' map without 'default').")
    parser.add_argument("--port",      type=int, default=11438)
    parser.add_argument("--host",      default="127.0.0.1",
                        help="Bind address (default 127.0.0.1). Use 0.0.0.0 to expose on the LAN.")
    parser.add_argument("--config",    default=None, help="Path to config.json")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Logging verbosity (default: info)")
    parser.add_argument("--allow-degraded", action="store_true",
                        help="Start even if the default project's embedder does not answer "
                             "the boot probe. Without it, serve refuses to start: "
                             "an unreachable embedder otherwise degrades every semantic query "
                             "to keyword-only with nothing visible downstream. "
                             "CHONKS_ALLOW_DEGRADED=1 in the environment is equivalent.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )

    if args.host not in ("127.0.0.1", "::1", "localhost"):
        logger.warning(
            "Binding to %s (non-loopback). The HTTP API has no auth; anyone who "
            "can reach this port can search, read source, and trigger re-indexing. "
            "Restrict access at the network layer (firewall/VPN) or front it with "
            "an authenticating reverse proxy.",
            args.host,
        )

    loaded = load_config(args.config)
    for problem in loaded.problems:
        if not problem.missing:
            logger.warning("Failed to parse %s: %s", Path(problem.path), problem.reason)
    if loaded.path is not None:
        logger.info("Loaded config from %s", loaded.path)
    elif args.config:
        logger.warning("Config file not found: %s", args.config)
    else:
        logger.warning("No config.json found — using defaults")
    full = loaded.data
    explicit_config = args.config is not None

    # ---- Default project (legacy single-DB shape) ----
    # config's `projects["default"]`, if present, overrides this block entirely.
    default_research = full.get("research", {})
    default_repomap  = full.get("repomap", {})
    default_search   = full.get("search", {})
    # Top-level key, not nested under research/repomap; {} means unset (all-1.0 fallback elsewhere).
    default_edge_type_weights = full.get("edge_type_weights", {})
    default_exclude  = list(full.get("exclude") or [])
    default_include  = list(full.get("include") or [])
    default_root     = resolve_codebase(full.get("codebase"), explicit_config, logger)
    default_db       = Path(full.get("db") or args.db)
    # Per-project so one deployment can mix a LAN GPU and a localhost embedder.
    default_embed_url   = full.get("embed_url",   DEFAULT_EMBED_URL)
    default_embed_model = full.get("embed_model", DEFAULT_EMBED_MODEL)
    # Absent means Embedder applies its per-model preset; set explicitly (incl. "") to override.
    default_embed_query_prefix = full.get("embed_query_prefix")
    default_embed_doc_prefix   = full.get("embed_doc_prefix")
    # Absent falls back to Embedder's EMBED_QUERY_TOKEN_BUDGET.
    default_embed_query_token_budget = full.get("embed_query_token_budget")
    default_embed_batch    = full.get("embed_batch")    or EMBED_BATCH
    default_embed_inflight = full.get("embed_inflight") or EMBED_INFLIGHT

    _projects[DEFAULT_PROJECT] = {
        "db_path":      default_db,
        "root":         default_root,
        "exclude":      default_exclude,
        "include":      default_include,
        "research_cfg": default_research,
        "repomap_cfg":  default_repomap,
        "search_cfg":   default_search,
        "edge_type_weights": default_edge_type_weights,
        "embedder":     Embedder(default_embed_url, default_embed_model,
                                 query_prefix=default_embed_query_prefix,
                                 doc_prefix=default_embed_doc_prefix,
                                 query_token_budget=default_embed_query_token_budget),
        "embed_batch":    default_embed_batch,
        "embed_inflight": default_embed_inflight,
        "store":        None,
        "searcher":     None,
        "index_lock":   threading.Lock(),
    }

    # ---- Additional named projects ----
    # `db`/`codebase` are per-project, never inherited: sharing a DB across projects corrupts multi-tenancy.
    for name, pcfg in (full.get("projects") or {}).items():
        if not isinstance(pcfg, dict):
            logger.warning("Project %r config is not a dict; skipping.", name)
            continue
        db = pcfg.get("db")
        if not db:
            logger.warning("Project %r missing required 'db' field; skipping.", name)
            continue
        _projects[name] = {
            "db_path":      Path(db),
            # Bypasses the auto-discovery absolute-path guard: writing an entry is opting in.
            "root":         Path(pcfg["codebase"]).resolve() if pcfg.get("codebase") else None,
            "exclude":      list(pcfg.get("exclude") or default_exclude),
            "include":      list(pcfg.get("include") or default_include),
            "research_cfg": pcfg.get("research", default_research),
            "repomap_cfg":  pcfg.get("repomap",  default_repomap),
            "search_cfg":   pcfg.get("search",   default_search),
            "edge_type_weights": pcfg.get("edge_type_weights", default_edge_type_weights),
            "embedder":     Embedder(
                pcfg.get("embed_url",   default_embed_url),
                pcfg.get("embed_model", default_embed_model),
                query_prefix=pcfg.get("embed_query_prefix", default_embed_query_prefix),
                doc_prefix=pcfg.get("embed_doc_prefix", default_embed_doc_prefix),
                query_token_budget=pcfg.get("embed_query_token_budget", default_embed_query_token_budget),
            ),
            "embed_batch":    pcfg.get("embed_batch")    or default_embed_batch,
            "embed_inflight": pcfg.get("embed_inflight") or default_embed_inflight,
            "store":        None,
            "searcher":     None,
            "index_lock":   threading.Lock(),
        }

    if len(_projects) > 1:
        logger.info("Multi-project mode: %s", sorted(_projects))

    # Must fail loudly here: a dead embedder otherwise degrades silently to a still-plausible FTS-only response.
    embed_source = ("config.json embed_url" if "embed_url" in full
                    else "built-in default (no embed_url in config)")
    err = probe_embedder(_projects[DEFAULT_PROJECT]["embedder"])
    if err is not None:
        msg = (f"Embedder probe failed: {err}\n"
               f"  url:    {default_embed_url}  [{embed_source}]\n"
               f"  model:  {default_embed_model}")
        # Env var equivalent of --allow-degraded: compose can't append an optional CLI flag.
        allow_degraded = args.allow_degraded or (
            os.environ.get("CHONKS_ALLOW_DEGRADED", "").strip().lower() in ("1", "true", "yes"))
        if allow_degraded:
            logger.warning("%s\nStarting anyway (--allow-degraded): semantic and hybrid "
                           "search will fail, research will run degraded.", msg)
        else:
            logger.error("%s\nRefusing to start. Start the embedding server, fix "
                         "embed_url, or pass --allow-degraded for keyword-only serving.", msg)
            sys.exit(2)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
