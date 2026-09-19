"""chunking.py: tree-sitter AST segmentation (cAST-style). Pure,
dependency-light core of the indexer, bytes + language tag in, chunk dicts
out, no I/O/embedding/DB. Languages: _EXT_TO_LANG; text fallback: segment_text_file."""

import bisect
import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from chonks.languages import (
    EXT_TO_LANG as _EXT_TO_LANG,
    flags as _lang_flags,
    get_or_none as _lang_spec,
    table as _lang_table,
)
from chonks.languages._ast import (
    terminal_identifier as _terminal_identifier,
)
from chonks.languages._naming import extract_name as _extract_name_by_rules
from chonks.languages.spec import (
    FieldChildren as _FieldChildren,
    Field as _Field,
    Children as _Children,
    NodeRule as _NodeRule,
    LiteralSpec as _LiteralSpec,
    NOT_HANDLED as _NOT_HANDLED,
)

logger = logging.getLogger("chunking")

# Bump when a change here would alter chunk boundaries for already-indexed
# content. Provenance only (meta table, doctor.py); nothing reads it back
# to gate behavior.
CHUNKER_VERSION = 3

# Chunk-size constants: CHUNK_TARGET is UTF-8 characters; CHUNK_MIN/MAX are
# UTF-8 bytes. Don't mix the two when comparing.

CHUNK_TARGET = 4500   # characters, line-fallback slice target
CHUNK_MIN    = 300    # bytes, below this merge forward into the next segment
CHUNK_MAX    = 6000   # bytes, hard ceiling; forcible split above this

# AST-boundary chunks are self-contained (cAST) and get no overlap; only
# line-based fallback chunks get FALLBACK_OVERLAP_LINES of prior-chunk tail
# prepended, since a split there can land mid-context.
FALLBACK_OVERLAP_LINES = 5

# tree-sitter 0.25 raises ValueError on timeout; without this a malformed or
# adversarial file can loop the parser indefinitely.
PARSE_TIMEOUT_MICROS = 30_000_000  # 30 seconds

# file extension -> (tree-sitter language, structural node types that
# become chunk boundaries).
_BOUNDARY_NODES = _lang_table("boundary_nodes")

# Container nodes: recurse through them to find inner boundaries.
# If a container yields no inner boundaries, fall back to treating it as one chunk.
_CONTAINER_NODES = _lang_table("container_nodes")


def _lang_for_path(path: Path) -> str | None:
    return _EXT_TO_LANG.get(path.suffix.lower())


# Text-fallback chunks get `language` set to the raw extension instead, never
# a member of this set. store._chunk_kind_clause uses that split to
# discriminate "code" vs "docs" chunk_kind.
CODE_LANGUAGES: frozenset[str] = frozenset(_EXT_TO_LANG.values())

# A healthy repo can be 90%+ one family; only warn when ALSO docs-kind.
FAMILY_DOCS_SHARE_WARN = 0.80    # (a) docs-share *within* the dominant family
FAMILY_TOTAL_SHARE_WARN = 0.40   # (a) that family's share of the whole corpus
CORPUS_DOCS_SHARE_WARN = 0.50    # (b) corpus-wide docs share


@dataclass
class FamilyStats:
    """One row of the path-family chunk-share breakdown."""
    family: str
    chunks: int
    files: int
    docs_chunks: int
    top_file: str | None
    top_file_chunks: int

    @property
    def docs_share(self) -> float:
        return self.docs_chunks / self.chunks if self.chunks else 0.0

    @property
    def chunks_per_file(self) -> float:
        return self.chunks / self.files if self.files else 0.0


def path_family(path: str) -> str:
    """First path segment, or "(root)" for a file with no directory
    component. `path` must be posix-style, relative to the indexed root."""
    head, sep, _ = path.partition("/")
    return head if sep else "(root)"


def family_breakdown(rows: Iterable[tuple[str, str | None]]) -> list[FamilyStats]:
    """Group chunk rows by top-level path family, sorted chunks DESC then
    family ASC for deterministic output."""
    by_family: dict[str, dict[str, Any]] = {}
    for path, language in rows:
        fam = path_family(path)
        entry = by_family.setdefault(
            fam, {"chunks": 0, "files": set(), "docs_chunks": 0, "file_counts": {}}
        )
        entry["chunks"] += 1
        entry["files"].add(path)
        if language not in CODE_LANGUAGES:
            entry["docs_chunks"] += 1
        entry["file_counts"][path] = entry["file_counts"].get(path, 0) + 1

    stats: list[FamilyStats] = []
    for fam, entry in by_family.items():
        top_file, top_file_chunks = None, 0
        for p, n in entry["file_counts"].items():
            if n > top_file_chunks:
                top_file, top_file_chunks = p, n
        stats.append(FamilyStats(
            family=fam,
            chunks=entry["chunks"],
            files=len(entry["files"]),
            docs_chunks=entry["docs_chunks"],
            top_file=top_file,
            top_file_chunks=top_file_chunks,
        ))
    stats.sort(key=lambda s: (-s.chunks, s.family))
    return stats


def dominance_warning(stats: list[FamilyStats], total_chunks: int) -> str | None:
    """Warns when a family is docs-heavy and large, or the corpus overall
    leans docs-kind (thresholds above). `stats` must be chunks-DESC."""
    if total_chunks == 0:
        return None

    for s in stats:
        family_share = s.chunks / total_chunks
        if s.docs_share >= FAMILY_DOCS_SHARE_WARN and family_share >= FAMILY_TOTAL_SHARE_WARN:
            return (
                f"WARNING: '{s.family}/' is {family_share:.0%} of the index "
                f"({s.chunks}/{total_chunks} chunks), {s.docs_share:.0%} of which is "
                f"docs-kind content. Largest file: {s.top_file} ({s.top_file_chunks} chunks). "
                f"If this is generated or vendored content, exclude it — config.json: "
                f'"exclude": ["{s.family}/"]'
            )

    total_docs = sum(s.docs_chunks for s in stats)
    corpus_docs_share = total_docs / total_chunks
    if corpus_docs_share >= CORPUS_DOCS_SHARE_WARN:
        return (
            f"WARNING: docs-kind chunks are {corpus_docs_share:.0%} of the whole index "
            f"({total_docs}/{total_chunks}). Consider excluding generated or vendored "
            f'docs in config.json, e.g. "exclude": ["<path>/"]'
        )
    return None


@dataclass
class DotDirStats:
    """One row of the dot-directory chunk-share breakdown."""
    prefix: str
    chunks: int
    files: int


def dotdir_breakdown(rows: Iterable[tuple[str, str | None]]) -> list[DotDirStats]:
    """Chunk/file counts by dot-dir prefix, catches a self-index run
    ingesting stale nested repo copies. ".git" is excluded as VCS metadata."""
    by_prefix: dict[str, dict[str, Any]] = {}
    for path, _language in rows:
        parts = path.split("/")
        if not parts[0].startswith(".") or parts[0] == ".git":
            continue
        # Drop the filename first: a file inside a dot-dir must group
        # under the dir, not under "<dir>/<file>".
        dir_parts = parts[:-1] or parts[:1]
        prefix = "/".join(dir_parts[:2])
        entry = by_prefix.setdefault(prefix, {"chunks": 0, "files": set()})
        entry["chunks"] += 1
        entry["files"].add(path)

    stats = [
        DotDirStats(prefix=p, chunks=e["chunks"], files=len(e["files"]))
        for p, e in by_prefix.items()
    ]
    stats.sort(key=lambda s: (-s.chunks, s.prefix))
    return stats


def dotdir_total_share(stats: list[DotDirStats], total_chunks: int) -> float:
    """Fraction of `total_chunks` accounted for by dotdir_breakdown's output."""
    if total_chunks == 0:
        return 0.0
    return sum(s.chunks for s in stats) / total_chunks


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _is_boundary(node: Node, lang: str, src: bytes) -> bool:
    boundaries = _BOUNDARY_NODES.get(lang, set())
    spec = _lang_spec(lang)
    if node.type in boundaries:
        if spec is not None:
            node_filter = spec.boundary_filters.get(node.type)
            if node_filter is not None:
                return node_filter(node, src)
        return True
    if spec is not None and spec.extra_boundary is not None:
        return spec.extra_boundary(node, src)
    return False


# Salvage-eligible: statement-body types only. Member-body types
# (class/struct) are excluded, or an errored class explodes into one
# chunk per method (methods are always nested there).
_SALVAGE_ELIGIBLE_BOUNDARY_TYPES = _lang_table("salvage_nodes")


def _is_salvage_eligible(node: Node, lang: str, src: bytes) -> bool:
    """True for a has_error boundary type safe to recurse into (see module
    comment above). cbuffer/tbuffer qualify too: their body holds only
    simple fields, never nested boundary types."""
    spec = _lang_spec(lang)
    if spec is not None and spec.salvage_extra is not None and spec.salvage_extra(node, src):
        return True
    return node.type in _SALVAGE_ELIGIBLE_BOUNDARY_TYPES.get(lang, set())


def _salvage_errored_boundary(node: Node, lang: str, src: bytes,
                              counters: dict | None) -> list[Node] | None:
    """Pulls clean (error-free) boundaries out of a has_error node's
    children. `node` itself is always kept alongside them, never replaced.
    Returns None (caller keeps `node` whole) when nothing clean is found."""
    inner: list[Node] = []
    for child in node.children:
        if child.is_named:
            inner.extend(_collect_boundaries(child, lang, src, counters))

    clean = [n for n in inner if not n.has_error]
    if not clean:
        return None

    if counters is not None:
        counters["error_salvaged"] = 1
        counters["error_salvaged_nodes"] = counters.get("error_salvaged_nodes", 0) + 1

    return [node] + clean


def _extract_name(node: Node, lang: str, src: bytes) -> str | None:
    """Primary symbol name from a boundary node, or None if not found."""
    spec = _lang_spec(lang)
    if spec is None:
        return None
    return _extract_name_by_rules(spec, node, src)


# Typed reference extraction (calls/imports/inherits): implemented for cpp,
# c_sharp, gdscript, python, c; other languages fall back to untyped
# 'mentions' edges in build_refs.

# Cap per-list name count so a pathological chunk can't blow up the metadata JSON row size.
_REFS_MAX_NAMES = 200

# Bounds one name's own receiver/arity fan-out so it can't consume many
# of the 200 name slots by itself (dedup is by full fingerprint, not name).
_REFS_MAX_CALL_VARIANTS_PER_NAME = 8


# Call-site fingerprint: receiver is the IMMEDIATE token adjacent to the
# called name, not the chain's root ('a.b.c()' -> 'b'), matching repomap's
# definer-qualifier split.

# Left-recursive grammars only: chain nests on the base/left side, so the
# base field's rightmost leaf is the immediate receiver. cpp's
# qualified_identifier is right-recursive and handled by the cpp spec's
# call_receiver hook.
_CHAIN_BASE_FIELD: dict[str, str] = {
    "field_expression": "argument",
    "attribute": "object",
    "member_access_expression": "expression",
    "qualified_name": "qualifier",
}


def _call_receiver(callee: Node, src: bytes) -> str | None:
    """The immediate receiver of a call's callee expression (see the chain
    rule in this section's header comment). None for a bare/free callee."""
    base_field = _CHAIN_BASE_FIELD.get(callee.type)
    if base_field is None:
        return None
    return _terminal_identifier(callee.child_by_field_name(base_field), src)


def _receiver(call_node: Node, callee: Node, src: bytes, lang: str) -> str | None:
    spec = _lang_spec(lang)
    if spec is not None and spec.call_receiver is not None:
        result = spec.call_receiver(call_node, callee, src)
        if result is not _NOT_HANDLED:
            return result
    return _call_receiver(callee, src)


# A spread/pack/varargs marker makes the positional count meaningless.
_VARIADIC_CALL_ARG_TYPES = {"list_splat", "dictionary_splat", "parameter_pack_expansion"}
# Keyword args are real but not positional; don't count or force wildcard.
_KEYWORD_CALL_ARG_TYPES = {"keyword_argument"}
# A named `comment` node between arguments would otherwise inflate arity
# by 1 per comment; must be skipped, not counted.
_TRIVIA_CALL_ARG_TYPES = {"comment"}


def _call_arity(args_node: "Node | None") -> int | None:
    """Positional argument count, or None (wildcard) when a variadic marker
    or ANY keyword argument is present, since a keyword arg can satisfy a
    required param the positional count wouldn't reflect."""
    if args_node is None:
        return 0
    count = 0
    saw_keyword = False
    for c in args_node.children:
        if not c.is_named:
            continue
        if c.type in _TRIVIA_CALL_ARG_TYPES:
            continue
        if c.type in _VARIADIC_CALL_ARG_TYPES:
            return None
        if c.type in _KEYWORD_CALL_ARG_TYPES:
            saw_keyword = True
            continue
        count += 1
    return None if saw_keyword else count


# Declarative per-language specs, a table of NodeRules walked generically
# by _apply_rule below. Adding a language means writing a spec, not a new
# branch of the walker.
LANG_REFS_SPECS = _lang_table("refs_spec")

_REFS_LANGS = set(LANG_REFS_SPECS)


def _call_entry_name(entry: "str | dict") -> str | None:
    """Bare called name from a calls-list entry: a fingerprint dict or a
    legacy bare string."""
    return entry["name"] if isinstance(entry, dict) else entry


def _add_call_entry(lst: list, entry: "str | dict") -> None:
    """Dedups on (name, receiver, arity) but caps on distinct NAMES. Shared
    by extraction and _merge_refs; both must use this or a merge undoes the cap."""
    if entry in lst:
        return
    name = _call_entry_name(entry)
    if not name:
        return
    same_name = 0
    distinct_names: set[str] = set()
    for e in lst:
        n = _call_entry_name(e)
        distinct_names.add(n)
        if n == name:
            same_name += 1
    if same_name >= _REFS_MAX_CALL_VARIANTS_PER_NAME:
        return
    if name not in distinct_names and len(distinct_names) >= _REFS_MAX_NAMES:
        return
    lst.append(entry)


def _apply_rule(n: Node, rule: "_NodeRule", refs: dict[str, list[str]], src: bytes, lang: str) -> None:
    def add(bucket: str, name: str | None) -> None:
        lst = refs[bucket]
        if name and name not in lst and len(lst) < _REFS_MAX_NAMES:
            lst.append(name)

    def add_call(name: str | None, receiver: str | None, arity: int | None) -> None:
        # New entries are fingerprint dicts, not bare strings (legacy shape).
        if not name:
            return
        _add_call_entry(refs["calls"], {"name": name, "receiver": receiver, "arity": arity})

    if isinstance(rule, _Field):
        target = n.child_by_field_name(rule.field)
        if target is not None:
            if rule.bucket == "calls":
                add_call(
                    rule.name_fn(target, src),
                    _receiver(n, target, src, lang),
                    _call_arity(n.child_by_field_name("arguments")),
                )
            else:
                add(rule.bucket, rule.name_fn(target, src))
    elif isinstance(rule, _FieldChildren):
        target = n.child_by_field_name(rule.field)
        if target is not None:
            for c in target.children:
                if c.type in rule.types:
                    add(rule.bucket, rule.name_fn(c, src))
    elif isinstance(rule, _Children):
        for c in n.children:
            matched_break_outer = False
            for spec in rule.specs:
                if c.type not in spec.types:
                    continue
                if spec.nested is not None:
                    for cc in c.children:
                        if cc.type in spec.nested.types:
                            add(spec.bucket, spec.nested.name_fn(cc, src))
                            if spec.nested.break_inner:
                                break
                elif spec.bucket == "calls":
                    # gdscript's chain is flat, so `c` itself carries no receiver;
                    # the gdscript call_receiver hook scans the siblings instead.
                    receiver = _receiver(n, c, src, lang)
                    add_call(
                        spec.name_fn(c, src),
                        receiver,
                        _call_arity(n.child_by_field_name("arguments")),
                    )
                else:
                    add(spec.bucket, spec.name_fn(c, src))
                matched_break_outer = spec.break_outer
                break  # elif semantics: first matching spec per child wins
            if matched_break_outer:
                break


# Literal extraction: string literals per chunk for find_by_message, via a
# declarative per-language _LiteralSpec walked by _collect_literals below.
_LITERALS_MIN_LEN = 6  # trivial-skip: stripped text shorter than this is noise
_HAS_ALPHA_RE = re.compile(r"[A-Za-z]")

# One prefix pattern covers every language's string-prefix convention
# (python f"/r", cpp L"/u8", c_sharp $"/@") without a per-language table.
_LITERAL_PREFIX_RE = re.compile(r"^[A-Za-z&]{0,3}")
_LITERAL_QUOTES = ('"""', "'''", '"', "'", "`")

# Unrecognized escapes (\d, \uXXXX, ...) are left as written.
_ESCAPE_MAP = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'", "\\": "\\", "0": "\0"}
_ESCAPE_RE = re.compile(r"\\(?:x[0-9A-Fa-f]{2}|.)", re.DOTALL)


def _decode_escapes(text: str) -> str:
    """Decodes known escapes only; best-effort text-matching aid, not a
    real string-literal parser, so an unrecognized escape is left as-is."""
    def repl(m: re.Match) -> str:
        s = m.group(0)
        if s[1] in "xX" and len(s) == 4:
            return chr(int(s[2:], 16))
        return _ESCAPE_MAP.get(s[1], s)
    return _ESCAPE_RE.sub(repl, text)


def _strip_literal_wrapper(raw: str, lang: str) -> tuple[str, str]:
    """Returns (text, mode): 'normal' decodes escapes, 'raw' never does
    (python r"...", lua [[...]], or an unrecognized wrapper), 'verbatim'
    (C# @"...") only decodes "" -> "."""
    spec = _lang_spec(lang)
    if spec is not None and spec.literal_wrapper is not None:
        result = spec.literal_wrapper(raw)
        if result is not _NOT_HANDLED:
            return result

    m = _LITERAL_PREFIX_RE.match(raw)
    idx = m.end() if m else 0
    prefix = raw[:idx]
    verbatim = False
    while idx < len(raw) and raw[idx] in "$@":
        verbatim = verbatim or raw[idx] == "@"
        idx += 1
    rest = raw[idx:]
    for q in _LITERAL_QUOTES:
        if rest.startswith(q) and rest.endswith(q) and len(rest) >= 2 * len(q):
            text = rest[len(q):-len(q)]
            if spec is not None and spec.raw_string_prefix is not None and spec.raw_string_prefix(prefix):
                return text, "raw"
            return text, ("verbatim" if verbatim else "normal")
    return raw, "raw"


def _leaf_literal_text(n: Node, src: bytes, lang: str) -> str:
    """Leaf text, NOT trailing-stripped: a concat piece's own trailing
    space matters when joined to the next piece. Strip once, in
    _collect_literals, on the fully assembled literal instead."""
    if n.type == "raw_string_literal":
        content = next((c for c in n.children if c.type == "raw_string_content"), None)
        return src[content.start_byte:content.end_byte].decode(errors="replace") if content else ""
    raw = src[n.start_byte:n.end_byte].decode(errors="replace")
    text, mode = _strip_literal_wrapper(raw, lang)
    if mode == "normal":
        text = _decode_escapes(text)
    elif mode == "verbatim":
        text = text.replace('""', '"')
    return text


_LITERAL_SPECS = _lang_table("literals")


def _plus_chain_pieces(n: Node, spec: _LiteralSpec, src: bytes, lang: str) -> list[str] | None:
    """Joined text pieces if the WHOLE chain is pure string literals, else
    None so an impure chain's literal pieces are picked up individually."""
    if not any(not c.is_named and c.type == spec.plus_op for c in n.children):
        return None
    named = [c for c in n.children if c.is_named]
    if len(named) != 2:
        return None
    pieces: list[str] = []
    for operand in named:
        if operand.type in spec.leaf_types:
            pieces.append(_leaf_literal_text(operand, src, lang))
        elif operand.type == spec.plus_type:
            sub = _plus_chain_pieces(operand, spec, src, lang)
            if sub is None:
                return None
            pieces.extend(sub)
        else:
            return None
    return pieces


def _is_trivial_literal(text: str) -> bool:
    return len(text) < _LITERALS_MIN_LEN or not _HAS_ALPHA_RE.search(text)


# Per-node memo so a node walked twice in one run doesn't double-count the
# cap/drop tally. Cleared alongside _literal_cap_state.
_literal_extract_cache: dict[int, list[tuple[str, int]]] = {}


def _collect_literals(node: Node, lang: str, src: bytes) -> list[tuple[str, int]]:
    """(text, line) per string literal, decoded so find_by_message matches
    runtime text, not escapes. Only a trailing DECODED newline is stripped."""
    spec = _LITERAL_SPECS.get(lang)
    if spec is None:
        return []
    cached = _literal_extract_cache.get(id(node))
    if cached is not None:
        return cached
    out: list[tuple[str, int]] = []

    def emit(text: str, line: int) -> None:
        text = text.rstrip("\r\n")
        if not _is_trivial_literal(text):
            out.append((text, line))

    def walk(n: Node) -> None:
        # is_named guard needed: TS's `predefined_type` reuses "string" as
        # an unnamed type-keyword token, not a string value.
        if spec.concat_types and n.type in spec.concat_types:
            pieces = [_leaf_literal_text(c, src, lang) for c in n.children
                      if c.is_named and c.type in spec.leaf_types]
            if pieces:
                emit("".join(pieces), n.start_point[0] + 1)
                return  # consumed, don't re-walk the joined pieces individually
        if spec.plus_type and n.type == spec.plus_type:
            pieces = _plus_chain_pieces(n, spec, src, lang)
            if pieces is not None:
                emit("".join(pieces), n.start_point[0] + 1)
                return
            # impure chain: fall through so pure leaf operands still collect
        if n.is_named and n.type in spec.leaf_types:
            if not (spec.skip_fn and spec.skip_fn(n)):
                emit(_leaf_literal_text(n, src, lang), n.start_point[0] + 1)
            return  # leaf's children are quote/content tokens, not literals
        for child in n.children:
            walk(child)

    walk(node)
    # Capped/counted once per node (memo above) so a node walked twice
    # doesn't double-count.
    if len(out) > _REFS_MAX_NAMES:
        _literal_cap_state["capped_chunks"] += 1
        _literal_cap_state["dropped"] += len(out) - _REFS_MAX_NAMES
        out = out[:_REFS_MAX_NAMES]
    _literal_extract_cache[id(node)] = out
    return out


# Module-level, but safe: segment_file/segment_text_file/segment_markdown
# run on one sequential parser thread.
_literal_cap_state = {"capped_chunks": 0, "dropped": 0}


def _extract_refs(node: Node, lang: str, src: bytes) -> dict[str, list[str]]:
    """{"calls", "imports", "inherits", "literals"} lists. "calls" entries
    are fingerprint dicts; a legacy bare string is treated as receiver=None,
    arity=None so old stored data still resolves."""
    refs: dict[str, list[str]] = {"calls": [], "imports": [], "inherits": [], "literals": []}

    spec = LANG_REFS_SPECS.get(lang)
    if spec is not None:
        def walk(n: Node) -> None:
            rule = spec.get(n.type)
            if rule is not None:
                _apply_rule(n, rule, refs, src, lang)
            for child in n.children:
                walk(child)

        walk(node)

    # _collect_literals already caps/tallies; this loop only adds dedup.
    lst = refs["literals"]
    for text, line in _collect_literals(node, lang, src):
        encoded = f"{line}\x1f{text}"
        if encoded not in lst:
            lst.append(encoded)

    return refs


# Definer arity: a text scan over the already-stored `content`Best-effort: None ("missing info") is always
# treated as compatible, so a parse miss costs precision.

# Keyword right before a definition's name (skippable past an unrelated
# same-named call); empty string for languages without one.
_DEF_SIGNATURE_KEYWORD = _lang_table("def_signature_keyword")


def _find_signature_param_texts(content: str, name: str, lang: str) -> list[str]:
    """Param-list text for EVERY same-named signature in `content`, not
    just the first, since Python @overload stacks can differ in arity.
    A match whose parens never balance is skipped, not fatal."""
    if not name:
        return []
    keyword = _DEF_SIGNATURE_KEYWORD.get(lang, "")
    pattern = re.compile(keyword + re.escape(name) + r"\s*\(")
    out: list[str] = []
    pos = 0
    while True:
        m = pattern.search(content, pos)
        if m is None:
            break
        start = m.end() - 1  # index of the anchor '('
        depth = 0
        end = None
        for i in range(start, len(content)):
            ch = content[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            # Skip this unbalanced match; advance past '(' (not `start`)
            # or the next search re-matches it and loops forever.
            pos = start + 1
            continue
        out.append(content[start + 1:end])
        pos = end + 1
    return out


def _split_top_level_params(text: str) -> list[str]:
    """Splits on top-level commas only: bracket-depth-aware, and quote-aware
    so `msg="a, b"` doesn't split. Not a real parser; rare edge cases
    (lambda defaults, '<'/'>' comparisons) can still miscount."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    quote: str | None = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote is not None:
            cur.append(ch)
            if ch == "\\" and i + 1 < n:
                cur.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            cur.append(ch)
            i += 1
            continue
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _classify_param(part: str, lang: str) -> str:
    """One parameter's role: 'marker' (not a real param, e.g. python's
    '/'), 'variadic', 'default', or 'required'."""
    spec = _lang_spec(lang)
    if spec is not None and spec.classify_param is not None:
        result = spec.classify_param(part)
        if result is not _NOT_HANDLED:
            return result
    if "=" in part:
        return "default"
    return "required"


def _definer_param_arity(content: str, name: str, lang: str) -> "tuple[int, int, bool] | None":
    """(min_required, max_params, has_variadic) or None (always compatible).
    Overloads WIDEN together (recall over precision); C#'s `this` isn't stripped."""
    params_texts = _find_signature_param_texts(content, name, lang)
    if not params_texts:
        return None
    spec = _lang_spec(lang)
    combined_min: int | None = None
    combined_max = 0
    combined_variadic = False
    for params_text in params_texts:
        parts = _split_top_level_params(params_text)
        if spec is not None and spec.strip_implicit_params is not None:
            parts = spec.strip_implicit_params(parts)
        elif spec is not None and len(parts) == 1 and parts[0] in spec.empty_param_spellings:
            # c/cpp's explicit zero-args spelling ('int f(void)') is the
            # same as an empty parameter list, not a param named 'void'.
            parts = []
        min_required = 0
        max_params = 0
        has_variadic = False
        for part in parts:
            kind = _classify_param(part, lang)
            if kind == "marker":
                continue
            if kind == "variadic":
                has_variadic = True
                continue
            max_params += 1
            if kind == "required":
                min_required += 1
        combined_min = min_required if combined_min is None else min(combined_min, min_required)
        combined_max = max(combined_max, max_params)
        combined_variadic = combined_variadic or has_variadic
    return (combined_min if combined_min is not None else 0, combined_max, combined_variadic)


# ---------------------------------------------------------------------------
# Segmentation (cAST algorithm)
# ---------------------------------------------------------------------------

def _node_text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode(errors="replace")


_EMPTY_REFS: dict[str, list[str]] = {"calls": [], "imports": [], "inherits": [], "literals": []}
_REFS_KEYS = ("calls", "imports", "inherits", "literals")


def _merge_refs(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict[str, list[str]]:
    """Union two refs dicts, deduped, capped at _REFS_MAX_NAMES per list.
    "calls" must go through `_add_call_entry`, not the generic loop below,
    or a merge silently re-truncates by raw entry count and undoes the cap."""
    out: dict[str, list[str]] = {}
    for key in _REFS_KEYS:
        merged = list(a.get(key) or [])
        if key == "calls":
            for entry in b.get(key) or []:
                _add_call_entry(merged, entry)
        else:
            for name in b.get(key) or []:
                if name not in merged and len(merged) < _REFS_MAX_NAMES:
                    merged.append(name)
        out[key] = merged
    return out


class _Segment:
    """A contiguous region of source code that will become one chunk."""
    __slots__ = ("start_byte", "end_byte", "start_line", "end_line",
                 "chunk_type", "name", "refs")

    def __init__(self, node: Node, src: bytes, lang: str):
        self.start_byte = node.start_byte
        self.end_byte   = node.end_byte
        self.start_line = node.start_point[0] + 1
        self.end_line   = node.end_point[0] + 1
        spec = _lang_spec(lang)
        synthetic = (spec.synthetic_chunk_type(node, src)
                     if spec is not None and spec.synthetic_chunk_type is not None else None)
        # F1 ran the cbuffer check for every language. This gives the same
        # result, because no other language uses "declaration" as a
        # boundary or a container type.
        self.chunk_type = synthetic if synthetic is not None else node.type
        self.name       = _extract_name(node, lang, src)
        self.refs       = _extract_refs(node, lang, src)

    def size(self, src: bytes) -> int:
        return self.end_byte - self.start_byte

    def content(self, src: bytes) -> str:
        return src[self.start_byte:self.end_byte].decode(errors="replace")


def _collect_boundaries(node: Node, lang: str, src: bytes,
                        counters: dict | None = None) -> list[Node]:
    """Structural boundary nodes in source order. `counters` goes ONLY at
    the top-level (segment_file) call; internal re-walks omit it to avoid
    double-counting the error-salvage tally."""
    if _is_boundary(node, lang, src):
        if node.has_error and _is_salvage_eligible(node, lang, src):
            salvaged = _salvage_errored_boundary(node, lang, src, counters)
            if salvaged is not None:
                return salvaged
        return [node]

    containers = _CONTAINER_NODES.get(lang, set())
    is_container = node.type in containers

    inner: list[Node] = []
    for child in node.children:
        if child.is_named:
            inner.extend(_collect_boundaries(child, lang, src, counters))

    if is_container and not inner:
        # Container with no structural children: chunk it as a whole
        return [node]

    return inner


def _split_large_node(node: Node, lang: str, src: bytes) -> list[_Segment]:
    """Recursively splits an oversized node by its children's boundaries,
    falling back to line-slicing if none exist."""
    # node itself is already a boundary, so collect from its children instead
    # (_collect_boundaries(node) would just return [node]).
    children: list[Node] = []
    for child in node.children:
        if child.is_named:
            children.extend(_collect_boundaries(child, lang, src))

    if len(children) > 1:
        segs: list = []
        # Header chunk = parent's signature up to the first child boundary,
        # so the container itself stays retrievable and never overlaps members.
        header_raw = src[node.start_byte:children[0].start_byte].decode(errors="replace")
        # rstrip: a trailing whitespace-only fragment would make h_end land
        # on the child's start_line, so adjacent chunks wouldn't share a line.
        header = header_raw.rstrip()
        if header.strip().strip("{}").strip():
            node_seg = _Segment(node, src, lang)
            h_start = node.start_point[0] + 1
            if len(header.encode("utf-8")) > CHUNK_MAX:
                segs.extend(_line_slice_oversized(
                    header, h_start, node_seg.name, node_seg.refs,
                    chunk_type=node_seg.chunk_type))
            else:
                h_end = h_start + header.count("\n")
                segs.append(_SyntheticSegment(
                    header, h_start, h_end,
                    chunk_type=node_seg.chunk_type, name=node_seg.name, refs=node_seg.refs))
        for c in children:
            child_seg = _Segment(c, src, lang)
            if child_seg.size(src) > CHUNK_MAX:
                segs.extend(_split_large_node(c, lang, src))
            else:
                segs.append(child_seg)
        # merge any still-tiny segments before returning
        return _merge_small(segs, src)

    # One oversized child: recurse into it so its slices keep its OWN name,
    # not the parent's. Byte-range check guards against infinite recursion.
    if len(children) == 1:
        c = children[0]
        if (c.start_byte, c.end_byte) != (node.start_byte, node.end_byte) \
                and _Segment(c, src, lang).size(src) > CHUNK_MAX:
            return _split_large_node(c, lang, src)

    # No sub-boundaries: slice by CHUNK_TARGET chars, but pieces still inherit
    # the boundary's name/type so it stays findable, not an anonymous "block".
    node_seg  = _Segment(node, src, lang)   # source of the name/type to propagate
    node_name = node_seg.name
    node_type = node_seg.chunk_type
    node_refs = node_seg.refs
    text = _node_text(node, src)
    lines = text.splitlines(keepends=True)
    segments: list = []
    buf: list[str] = []
    buf_start_line = node.start_point[0] + 1
    char_count = 0
    prev_tail: list[str] = []  # overlap prefix for the next segment

    for line in lines:
        buf.append(line)
        char_count += len(line)
        if char_count >= CHUNK_TARGET:
            content = "".join(prev_tail + buf)
            segments.append(_SyntheticSegment(
                content, buf_start_line, buf_start_line + len(buf) - 1,
                name=node_name, chunk_type=node_type, refs=node_refs,
            ))
            prev_tail = buf[-FALLBACK_OVERLAP_LINES:] if FALLBACK_OVERLAP_LINES > 0 else []
            buf_start_line += len(buf)
            buf = []
            char_count = 0

    if buf:
        content = "".join(prev_tail + buf)
        segments.append(_SyntheticSegment(
            content, buf_start_line, buf_start_line + len(buf) - 1,
            name=node_name, chunk_type=node_type, refs=node_refs,
        ))

    return segments if segments else [_Segment(node, src, lang)]


class _SyntheticSegment:
    """Fallback segment for raw text slices (no AST node). Carries optional
    name/chunk_type/refs so a merge of small named segments (_merge_small)
    can keep an identity instead of going anonymous."""
    __slots__ = ("_content", "start_line", "end_line", "chunk_type", "name", "refs")

    def __init__(self, content: str, start_line: int, end_line: int,
                 *, chunk_type: str = "block", name: str | None = None,
                 refs: dict[str, list[str]] | None = None):
        self._content  = content
        self.start_line = start_line
        self.end_line   = end_line
        self.chunk_type = chunk_type
        self.name       = name
        self.refs       = refs if refs is not None else _EMPTY_REFS

    def content(self, src: bytes) -> str:
        return self._content

    def size(self, src: bytes) -> int:
        # Return UTF-8 byte count to match _Segment.size(), so merge-time
        # comparisons against CHUNK_MIN / CHUNK_MAX use a single unit.
        return len(self._content.encode("utf-8"))


def _merge_small(segs: list, src: bytes) -> list:
    """Coalesces consecutive sub-CHUNK_MIN segments. BOTH sides must be
    trivia: absorbing a substantive boundary into a small one would erase
    its name from the symbol graph. Never merges past CHUNK_MAX."""
    if not segs:
        return segs

    merged: list = [segs[0]]
    for seg in segs[1:]:
        prev = merged[-1]
        prev_size = prev.size(src)
        # +1 for the "\n" inserted between contents on merge.
        merged_size = prev_size + 1 + seg.size(src)
        # seg.start_line > prev.end_line excludes salvage's deliberate
        # container/nested-boundary overlap; merging those would corrupt
        # the synthetic segment's line span.
        if (prev_size < CHUNK_MIN and seg.size(src) < CHUNK_MIN
                and merged_size <= CHUNK_MAX and seg.start_line > prev.end_line):
            pc = prev.content(src)
            # Only add a separator when prev's content doesn't already end in "\n".
            combined = pc + ("" if pc.endswith("\n") else "\n") + seg.content(src)
            # "<module>" residue is filler; never let it override a real
            # boundary's name.
            if prev.chunk_type == "module" and seg.name and seg.chunk_type != "module":
                m_name, m_type = seg.name, seg.chunk_type
            else:
                m_name = prev.name or seg.name
                m_type = prev.chunk_type if prev.name else seg.chunk_type
            m_refs = _merge_refs(prev.refs, seg.refs)
            syn = _SyntheticSegment(combined, prev.start_line, seg.end_line,
                                    chunk_type=m_type, name=m_name, refs=m_refs)
            merged[-1] = syn
        else:
            merged.append(seg)
    return merged


# Identity-free trivia (lone '#if'/'#endif', bare 'namespace X {') has no
# name of its own, so it's safe to fold into a neighbour without touching
# ITS name, unlike _merge_small's guard, which protects the opposite.
_PREPROC_CONDITIONAL_RE = re.compile(r"^#\s*(if|ifdef|ifndef|elif|else|endif)\b")
_CONTAINER_OPEN_RE = re.compile(r'^(namespace\s+[\w:]+\s*\{|extern\s+"C"\s*\{)\s*$')
# Braces/parens/semicolons/commas only (a lone "}", "};", "} {", …).
_PUNCT_ONLY_RE = re.compile(r"[{}();,\s]*$")


def _span_end_line(start_line: int, content: str) -> int:
    """end_line from `content`'s own newline count, not a neighbour's raw
    tree-sitter end_line, since the latter can claim a blank-line gap that
    isn't in `content` and overlap the following chunk's start_line."""
    n = content.count("\n")
    if content.endswith("\n"):
        return start_line + n - 1 if n else start_line
    return start_line + n


def _is_identity_free(text: str, lang: str) -> bool:
    """True iff every non-blank line is a preproc conditional, punctuation,
    a comment, or a bare container opening. Deliberately NOT #include/
    #define/#pragma: false positives fold real identity into a neighbour."""
    prefixes = _COMMENT_PREFIXES.get(lang, _COMMENT_PREFIXES_DEFAULT)
    spec = _lang_spec(lang)
    body = text if spec is not None and not spec.block_comments \
        else re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    saw_line = False
    for raw_line in body.splitlines():
        s = raw_line.strip()
        if not s:
            continue
        saw_line = True
        if _PREPROC_CONDITIONAL_RE.match(s):
            continue
        if _CONTAINER_OPEN_RE.match(s):
            continue
        # comment, possibly preceded by brace/punctuation ('} // namespace Foo')
        core = s.lstrip("{}();, \t")
        # Bare "*" must be alone or followed by space/'/'/EOL (doc-comment
        # continuation), or it misclassifies a real '*ptr = val;' statement.
        def _is_comment_prefixed(x: str) -> bool:
            for pfx in prefixes:
                if pfx == "*":
                    if re.match(r"^\*(\s|/|$)", x):
                        return True
                elif x.startswith(pfx):
                    return True
            return False
        if _is_comment_prefixed(core) or _is_comment_prefixed(s):
            continue
        if _PUNCT_ONLY_RE.fullmatch(s):
            continue
        return False
    return saw_line  # an all-blank fragment has nothing to classify, not identity-free


def _absorb_identity_free_fragments(segs: list, src: bytes, lang: str) -> list:
    """Folds an identity-free fragment (see module comment above) into a
    neighbour. BACKWARD by default; FORWARD for an unclosed leading guard,
    since '#if' introduces what follows it. Skips if it would break CHUNK_MAX."""
    if not segs:
        return segs

    def is_leading_guard(text: str) -> bool:
        has_open = re.search(r"^#\s*(if|ifdef|ifndef)\b", text, re.M) is not None
        has_close = re.search(r"^#\s*endif\b", text, re.M) is not None
        if has_open and not has_close:
            return True
        # A bare container opening introduces what follows it, so it belongs
        # forward, not appended after an unrelated preceding chunk.
        return any(_CONTAINER_OPEN_RE.match(ln.strip()) for ln in text.splitlines())

    out: list = list(segs)
    i = 0
    while i < len(out):
        seg = out[i]
        if seg.size(src) >= CHUNK_MIN or not _is_identity_free(seg.content(src), lang):
            i += 1
            continue

        text = seg.content(src)
        prev = out[i - 1] if i > 0 else None
        nxt = out[i + 1] if i + 1 < len(out) else None

        def absorb_backward() -> bool:
            if prev is None or prev.size(src) + 1 + seg.size(src) > CHUNK_MAX:
                return False
            pc = prev.content(src)
            combined = pc + ("" if pc.endswith("\n") else "\n") + text
            end_line = _span_end_line(prev.start_line, combined)
            out[i - 1] = _SyntheticSegment(combined, prev.start_line, end_line,
                                           chunk_type=prev.chunk_type, name=prev.name,
                                           refs=prev.refs)
            del out[i]
            return True

        def absorb_forward() -> bool:
            if nxt is None or seg.size(src) + 1 + nxt.size(src) > CHUNK_MAX:
                return False
            nc = nxt.content(src)
            combined = text + ("" if text.endswith("\n") else "\n") + nc
            end_line = _span_end_line(seg.start_line, combined)
            out[i + 1] = _SyntheticSegment(combined, seg.start_line, end_line,
                                           chunk_type=nxt.chunk_type, name=nxt.name,
                                           refs=nxt.refs)
            del out[i]
            return True

        if is_leading_guard(text):
            absorbed = absorb_forward() or absorb_backward()
        else:
            absorbed = absorb_backward() or absorb_forward()
        if not absorbed:
            i += 1  # neither neighbour could take it without breaking the ceiling
        # else: re-examine index i without advancing (absorption shrinks
        # `out` by one, so this can't loop forever).
    return out


def _hardwrap_text(text: str) -> list[str]:
    """Splits into <= CHUNK_MAX byte pieces on character boundaries, never
    mid-codepoint. Last resort for content (e.g. one giant line) a line
    split can't break."""
    if len(text.encode("utf-8")) <= CHUNK_MAX:
        return [text]
    pieces: list[str] = []
    buf: list[str] = []
    size = 0
    for ch in text:
        cb = len(ch.encode("utf-8"))
        if size + cb > CHUNK_MAX and buf:
            pieces.append("".join(buf))
            buf, size = [], 0
        buf.append(ch)
        size += cb
    if buf:
        pieces.append("".join(buf))
    return pieces


def _enforce_ceiling(segs: list, src: bytes, path: str | None = None,
                     counters: dict | None = None) -> list:
    """Hard-splits any segment still over CHUNK_MAX. Only fires for
    pathological input (one giant line) a line split can't break; logs the
    file so data/generated/minified blobs are easy to spot."""
    out: list = []
    oversize = 0
    extra_pieces = 0
    for s in segs:
        if s.size(src) <= CHUNK_MAX:
            out.append(s)
            continue
        oversize += 1
        pieces = _hardwrap_text(s.content(src))
        extra_pieces += len(pieces)
        # Apportion the line span per piece by newline count, or every piece
        # would inherit the parent's full span, breaking path:line citations.
        line = s.start_line
        for p in pieces:
            n = p.count("\n")
            if n == 0:
                # No newline at all: advance a virtual line by 1 anyway, or
                # every zero-newline piece reports the same duplicate span.
                next_line = line + 1
                end_line = next_line
            elif p.endswith("\n"):
                # n complete lines from `line`; no -1 here double-counts the
                # boundary line across this piece and the next.
                next_line = line + n
                end_line = next_line - 1
            else:
                # Ends mid-line: that last line continues in the next piece,
                # so the next piece must start on this SAME line.
                next_line = line + n
                end_line = next_line
            out.append(_SyntheticSegment(p, line, end_line,
                                         name=s.name, chunk_type=s.chunk_type, refs=s.refs))
            line = next_line
    if oversize:
        logger.warning(
            "chunking: %s had %d oversize chunk(s) (> %d B) hard-split into %d byte-windows "
            "— likely a data/generated/minified file; consider excluding it",
            path or "<unknown>", oversize, CHUNK_MAX, extra_pieces,
        )
        if counters is not None:
            counters["oversize_chunks"] = counters.get("oversize_chunks", 0) + oversize
            counters["oversize_files"] = counters.get("oversize_files", 0) + 1
    return out


def _decode_literals(encoded: list[str]) -> list[tuple[str, int]]:
    """Reverse the "<line>\\x1f<text>" packing _extract_refs uses internally
    (see its docstring) back into the public (text, line) record shape."""
    out = []
    for s in encoded:
        line_str, _, text = s.partition("\x1f")
        out.append((text, int(line_str)))
    return out


def _finalize(segs: list, src: bytes, path: str | None = None,
              counters: dict | None = None) -> list[dict[str, Any]]:
    """Enforce the size ceiling, then emit chunk dicts (dropping whitespace-only)."""
    # Read + always reset the literal-cap tally, even without `counters`, so
    # it doesn't leak into the next file.
    capped_chunks = _literal_cap_state["capped_chunks"]
    dropped = _literal_cap_state["dropped"]
    _literal_cap_state["capped_chunks"] = 0
    _literal_cap_state["dropped"] = 0
    _literal_extract_cache.clear()
    if counters is not None and (capped_chunks or dropped):
        counters["literals_capped_chunks"] = counters.get("literals_capped_chunks", 0) + capped_chunks
        counters["literals_dropped"] = counters.get("literals_dropped", 0) + dropped

    return [
        {
            "chunk_type": seg.chunk_type,
            "name": seg.name,
            "start_line": seg.start_line,
            "end_line": seg.end_line,
            "content": seg.content(src),
            "refs": seg.refs,
            "literals": _decode_literals(seg.refs.get("literals", [])),
        }
        for seg in _enforce_ceiling(segs, src, path, counters)
        if seg.content(src).strip()
    ]


def _line_slice_oversized(content: str, start_line: int, name: str | None,
                          refs: dict[str, list[str]], chunk_type: str = "module") -> list:
    """Line-slices an oversized residue/header span with REAL per-piece
    line numbers, so no piece starts mid-statement or duplicates another's
    span, unlike the byte-window _hardwrap_text fallback."""
    lines = content.splitlines(keepends=True)
    out: list = []
    buf: list[str] = []
    buf_start = start_line
    char_count = 0
    ln = start_line
    for line in lines:
        buf.append(line)
        char_count += len(line)
        if char_count >= CHUNK_TARGET:
            out.append(_SyntheticSegment("".join(buf), buf_start, ln,
                                         chunk_type=chunk_type, name=name, refs=refs))
            buf, buf_start, char_count = [], ln + 1, 0
        ln += 1
    if buf and "".join(buf).strip():
        out.append(_SyntheticSegment("".join(buf), buf_start, ln - 1,
                                     chunk_type=chunk_type, name=name, refs=refs))
    return out


def _collect_module_residue(root: Node, lang: str, src: bytes) -> list:
    """Captures top-level spans no boundary node covers (imports, module
    docstring, `if __name__`). Without this, python's empty container set
    drops all module-level code, e.g. a 247-line argparse CLI."""
    segs: list = []
    run: list[Node] = []

    def flush() -> None:
        if not run:
            return
        content = src[run[0].start_byte:run[-1].end_byte].decode(errors="replace")
        if content.strip():
            # Module-level residue is where imports live for most languages;
            # without extracting refs here, 'imports' never populates.
            refs = _EMPTY_REFS
            for n in run:
                refs = _merge_refs(refs, _extract_refs(n, lang, src))
            start_line = run[0].start_point[0] + 1
            end_line = run[-1].end_point[0] + 1
            # Line-sliced here, not deferred to _finalize's byte-window
            # fallback, which would stamp every piece with the same span.
            if len(content.encode("utf-8")) > CHUNK_MAX:
                segs.extend(_line_slice_oversized(content, start_line, "<module>", refs))
            else:
                segs.append(_SyntheticSegment(
                    content, start_line, end_line,
                    chunk_type="module", name="<module>", refs=refs))
        run.clear()

    spec = _lang_spec(lang)
    for child in root.children:
        if not child.is_named:
            continue
        if _collect_boundaries(child, lang, src):
            flush()
            continue
        if spec is not None and spec.module_residue_split is not None \
                and spec.module_residue_split(child, src):
            flush()
            content = _node_text(child, src)
            if content.strip():
                m_start = child.start_point[0] + 1
                m_end = child.end_point[0] + 1
                m_refs = _extract_refs(child, lang, src)
                # Same line-slicing as the generic residue run above.
                if len(content.encode("utf-8")) > CHUNK_MAX:
                    segs.extend(_line_slice_oversized(content, m_start, "<module>:__main__", m_refs))
                else:
                    segs.append(_SyntheticSegment(
                        content, m_start, m_end,
                        chunk_type="module", name="<module>:__main__", refs=m_refs))
            continue
        run.append(child)
    flush()
    return segs


def _fill_coverage_gaps(segs: list, src: bytes) -> list:
    """Completeness backstop: emits chunks for non-blank lines no existing
    segment covers, e.g. a split node's own orchestration code between its
    child boundaries. chunk_type='module' so _merge_small treats it as filler."""
    if not segs:
        return []
    lines = src.decode(errors="replace").splitlines(keepends=True)
    n = len(lines)
    if n == 0:
        return []
    covered = bytearray(n + 2)  # 1-based line flags
    for s in segs:
        for ln in range(max(1, s.start_line), min(n, s.end_line) + 1):
            covered[ln] = 1

    gaps: list = []
    ln = 1
    while ln <= n:
        if covered[ln] or not lines[ln - 1].strip():
            ln += 1
            continue
        start = ln
        while ln <= n and not covered[ln]:
            ln += 1
        end = ln - 1
        while end > start and not lines[end - 1].strip():
            end -= 1
        # Skip spans that are only braces / punctuation / whitespace (a lone "}").
        if not "".join(lines[start - 1:end]).strip().strip("{}").strip():
            continue
        name = "<module>" if not lines[start - 1][:1].isspace() else "<block>"
        buf: list[str] = []
        buf_start = start
        char_count = 0
        for i in range(start, end + 1):
            buf.append(lines[i - 1])
            char_count += len(lines[i - 1])
            if char_count >= CHUNK_TARGET:
                gaps.append(_SyntheticSegment("".join(buf), buf_start, i,
                                              chunk_type="module", name=name))
                buf, buf_start, char_count = [], i + 1, 0
        if buf and "".join(buf).strip():
            gaps.append(_SyntheticSegment("".join(buf), buf_start, end,
                                          chunk_type="module", name=name))
    return gaps


# Macro self-heal: tree-sitter-cpp can't parse engine macros (UCLASS/
# Q_OBJECT/...), dropping the class name. Discovery is purely structural
# (no hardcoded names), so C's own annotation macros benefit too.

_MACRO_LANGS = _lang_flags("c_macro_self_heal")
_MAX_MACRO_CANDIDATES = 48  # cap trial-reparses per pass on pathological files
# A leading macro (UCLASS) only surfaces once the export macro beside the
# class name is blanked, so healing needs multiple passes.
_MAX_MACRO_PASSES = 5
_MACRO_KEEP = {"NULL", "TRUE", "FALSE"}

# Qt's signals:/slots: aren't ALL_CAPS, so treated as a fixed set. Bare
# form blanks NAME+colon; qualified form blanks only NAME, leaving the
# colon public/private's own access_specifier needs to close.
_QT_ACCESS_SPECIFIERS = {"signals", "slots", "Q_SIGNALS", "Q_SLOTS"}


def _qt_specifier_res(specifiers) -> tuple[re.Pattern, re.Pattern]:
    """Qualified/bare Qt specifier regexes scoped to the given names only."""
    alt = "|".join(re.escape(n) for n in specifiers)
    qualified = re.compile(
        r"\b(?:public|private|protected)\b[ \t]*\b(?:" + alt + r")\b")
    bare = re.compile(r"\b(?:" + alt + r")\b[ \t]*:")
    return qualified, bare


def _is_macro_name(t: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", t)) and t not in _MACRO_KEEP


def _blank_macros(src: bytes, names) -> bytes:
    """Replaces each macro NAME(...) with spaces, preserving byte length
    and newlines so tree-sitter offsets stay valid against the ORIGINAL
    bytes. latin-1 keeps one char = one byte for regex spans."""
    if not names:
        return src
    out = bytearray(src)
    text = src.decode("latin-1")

    def blank(start: int, end: int) -> None:
        for i in range(start, end):
            if out[i] not in (0x0A, 0x0D):  # keep newlines so line numbers don't shift
                out[i] = 0x20

    specifiers = {n for n in names if n in _QT_ACCESS_SPECIFIERS}
    if specifiers:
        qualified_re, bare_re = _qt_specifier_res(specifiers)
        qualified_spans = []
        for m in qualified_re.finditer(text):
            # blank only the trailing NAME token, not the public/private/protected
            # keyword or the colon that belongs to it
            name_start = m.end()
            while name_start > m.start() and not text[name_start - 1].isspace():
                name_start -= 1
            blank(name_start, m.end())
            qualified_spans.append((m.start(), m.end()))
        for m in bare_re.finditer(text):
            if any(s <= m.start() < e for s, e in qualified_spans):
                continue  # already handled by the qualified form above
            blank(m.start(), m.end())

    plain = [n for n in names if n not in _QT_ACCESS_SPECIFIERS]
    if plain:
        pat = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in plain) + r")\b[ \t]*(?:\([^()]*\))?")
        for m in pat.finditer(text):
            blank(m.start(), m.end())
    return bytes(out)


def _count_errors(root: Node) -> int:
    n, stack = 0, [root]
    while stack:
        node = stack.pop()
        if node.is_error or node.is_missing:
            n += 1
        stack.extend(node.children)
    return n


def _errors_with_ranges(root: Node) -> tuple[int, list[tuple[int, int]]]:
    """_count_errors plus byte ranges, one walk instead of two since both
    are needed per self-heal pass. Ranges are RAW; margin applied later."""
    n = 0
    ranges: list[tuple[int, int]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_error or node.is_missing:
            n += 1
            ranges.append((node.start_byte, node.end_byte))
        stack.extend(node.children)
    return n, ranges


def _count_errors_upto(root: Node, threshold: int) -> int:
    """Bails once count reaches `threshold` (admission only needs "< base").
    Uses a TreeCursor, not a children-stack, since the cursor visits nodes
    in source order so a real error is reached fast enough for the bail to help."""
    n = 0
    cursor = root.walk()
    reached_end = False
    while not reached_end:
        node = cursor.node
        if node.is_error or node.is_missing:
            n += 1
            if n >= threshold:
                return n
        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue
        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_end = True
            elif cursor.goto_next_sibling():
                retracing = False
    return n


_ERROR_PROXIMITY_MARGIN = 200  # bytes; see _near_error_ranges


def _near_error_ranges(cands: set, parse_src: bytes,
                       error_ranges: list[tuple[int, int]],
                       margin: int = _ERROR_PROXIMITY_MARGIN) -> set:
    """Drops candidates with no occurrence within `margin` bytes of an
    error, since those can't reduce the error count and aren't worth a
    trial reparse. Most _discover_macros tells have no proximity scoping."""
    if not error_ranges or not cands:
        return set()
    merged: list[tuple[int, int]] = []
    for s, e in sorted(error_ranges):
        s, e = s - margin, e + margin
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    starts = [s for s, _ in merged]

    def overlaps(a: int, b: int) -> bool:
        i = bisect.bisect_right(starts, b) - 1
        return i >= 0 and merged[i][1] >= a

    kept: set = set()
    for name in cands:
        pat = re.compile(rb"\b" + re.escape(name.encode("latin-1")) + rb"\b")
        for m in pat.finditer(parse_src):
            if overlaps(m.start(), m.end()):
                kept.add(name)
                break
    return kept


def _discover_macros(root: Node, src: bytes) -> set[str]:
    """Candidate macro names via 4 structural tells (no hardcoded names;
    over-discovery is harmless since callers validate by error-count
    decrease before ever blanking one)."""
    text = src.decode("latin-1")
    found: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if (node.type == "call_expression" and node.parent is not None
                and node.parent.type == "expression_statement" and node.children):
            callee = node.children[0]
            if callee.type == "identifier":
                t = src[callee.start_byte:callee.end_byte].decode("latin-1")
                if _is_macro_name(t):
                    found.add(t)
        if node.is_error or node.is_missing:
            span = src[node.start_byte:node.end_byte].decode("latin-1")
            # Require a following '(': a bare ALL_CAPS type (RID) blanked here
            # could coincidentally drop the error count and erase a real name.
            for tok in re.findall(r"[A-Z][A-Z0-9_]{2,}(?=\s*\()", span):
                if _is_macro_name(tok):
                    found.add(tok)
        stack.extend(node.children)
    for m in re.finditer(r"\b(?:class|struct)\s+([A-Z][A-Z0-9_]{2,})\s+[A-Za-z_]\w*", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    for m in re.finditer(r"\b([A-Z][A-Z0-9_]{2,})\b\s*(?:\([^()]*\))?\s*(?:class|struct)\b", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    # Standalone macro line (GENERATED_BODY(); Q_OBJECT) parses as a bogus
    # MISSING ';' declaration, not an ERROR node, so (a)/(d) above miss it.
    # `\r?$`: without it this never fires on CRLF (Windows) files.
    for m in re.finditer(r"(?m)^[ \t]*([A-Z][A-Z0-9_]{2,})[ \t]*(?:\([^()]*\))?[ \t]*;?[ \t]*\r?$", text):
        if _is_macro_name(m.group(1)):
            found.add(m.group(1))
    return found


# ---------------------------------------------------------------------------
# Symbol index (decoupled from chunk packing)
# ---------------------------------------------------------------------------

def _collect_symbols_from_root(root: Node, lang: str, src: bytes) -> list[dict[str, Any]]:
    """Every named boundary, recursing INTO boundaries (unlike
    _collect_boundaries, which stops at the first). So a folded class's
    methods are still recorded, independent of how chunking packed them."""
    out: list[dict[str, Any]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if _is_boundary(node, lang, src):
            name = _extract_name(node, lang, src)
            if name:
                out.append({
                    "name":       name,
                    "kind":       node.type,
                    "language":   lang,
                    "start_line": node.start_point[0] + 1,
                    "end_line":   node.end_point[0] + 1,
                })
        stack.extend(node.children)
    return out


# Line-comment prefixes per language (block comments /* */ handled separately).
_COMMENT_PREFIXES = _lang_table("line_comment_prefixes")
_COMMENT_PREFIXES_DEFAULT = ("//", "/*", "*/", "*")  # cpp / c / c_sharp / hlsl
# Divider/comment punctuation: a span of only these (+ whitespace) has no text.
_DIVIDER_RE = re.compile(r"[/*#=\-_~<>|+.\s]")


def _is_comment_only(text: str, lang: str) -> bool:
    """True if every non-blank line is a comment or a pure-punctuation divider."""
    prefixes = _COMMENT_PREFIXES.get(lang, _COMMENT_PREFIXES_DEFAULT)
    spec = _lang_spec(lang)
    if spec is None or spec.block_comments:
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)  # strip block comments
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(prefixes):
            continue
        if not _DIVIDER_RE.sub("", s):  # only divider punctuation
            continue
        return False
    return True


def _comment_has_text(text: str) -> bool:
    """True if a comment carries real words (a label/doc), not just a divider rule."""
    return bool(_DIVIDER_RE.sub("", text))


def _attach_or_drop_comments(segs: list, src: bytes, lang: str) -> list:
    """A small labeled comment attaches FORWARD to the boundary it labels; a
    pure divider is dropped; a large block (>= CHUNK_MIN) stays standalone
    so it doesn't pollute a function's chunk."""
    out: list = []
    pending = None
    for seg in segs:
        text = seg.content(src)
        if _is_comment_only(text, lang):
            if not _comment_has_text(text):
                continue  # pure divider → drop
            if seg.size(src) >= CHUNK_MIN:
                if pending is not None:
                    out.append(pending)
                    pending = None
                out.append(seg)  # big comment block → standalone
                continue
            if pending is None:
                pending = seg
            else:  # coalesce consecutive small comments
                pending = _SyntheticSegment(
                    pending.content(src) + "\n" + text, pending.start_line, seg.end_line,
                    chunk_type=pending.chunk_type, name=pending.name)
            continue
        if pending is not None:  # attach held comment to this real boundary
            seg = _SyntheticSegment(
                pending.content(src) + "\n" + seg.content(src), pending.start_line, seg.end_line,
                chunk_type=seg.chunk_type, name=seg.name)
            pending = None
        out.append(seg)
    if pending is not None:
        # A trailing comment has no forward boundary to attach to; attach
        # it BACKWARD to the last segment instead, unless that breaks CHUNK_MAX.
        if out and out[-1].size(src) + 1 + pending.size(src) <= CHUNK_MAX:
            last = out[-1]
            lc = last.content(src)
            combined = lc + ("" if lc.endswith("\n") else "\n") + pending.content(src)
            # _span_end_line, not pending's raw end_line: a blank-line gap
            # between them isn't part of `combined` and would overstate the span.
            end_line = _span_end_line(last.start_line, combined)
            out[-1] = _SyntheticSegment(
                combined, last.start_line, end_line,
                chunk_type=last.chunk_type, name=last.name, refs=last.refs)
        else:
            out.append(pending)
    return out


_MARKDOWN_EXTS = (".md", ".markdown")

# Commonmark's trailing '#' closing sequence ("## Title ##") is stripped in
# _heading_name, not here, so this capture can stay simple/lazy.
_ATX_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})(?:[ \t]+(.+?))?[ \t]*$")
# Toggles fence state on any ``` or ~~~ line so '#' inside one isn't a heading.
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def _is_markdown_path(path: str | None) -> bool:
    return path is not None and path.lower().endswith(_MARKDOWN_EXTS)


def _heading_name(raw: str | None) -> str | None:
    """Strips Commonmark's trailing '#' closing sequence. Bare '#' (empty
    heading) returns None."""
    text = raw or ""
    text = re.sub(r"[ \t]+#+[ \t]*$", "", text).strip()
    return text or None


def _split_markdown_sections(lines: list[str]) -> list[tuple[int, int, str | None]]:
    """(start_line, end_line, name) spans at ATX heading boundaries,
    ignoring '#' inside fenced code blocks."""
    sections: list[tuple[int, int, str | None]] = []
    current_start = 1
    current_name: str | None = None
    in_fence = False
    fence_char = ""
    for i, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n")
        fence_m = _FENCE_RE.match(line)
        if fence_m:
            char = fence_m.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_char = char
            elif char == fence_char:
                # Only a same-character fence closes the block (CommonMark).
                in_fence = False
                fence_char = ""
            continue
        if in_fence:
            continue
        m = _ATX_HEADING_RE.match(line)
        if not m:
            continue
        if i > current_start:
            sections.append((current_start, i - 1, current_name))
        current_start = i
        current_name = _heading_name(m.group(2))
    sections.append((current_start, len(lines), current_name))
    return sections


def _slice_section(lines: list[str], name: str | None) -> list:
    """Line-slices one section into <= CHUNK_TARGET pieces, all keeping its
    name. Overlap is section-local: a different section's trailing context
    isn't useful here."""
    segs: list = []
    if not lines:
        return segs
    buf: list[str] = []
    buf_start = 1
    char_count = 0
    prev_tail: list[str] = []
    for line in lines:
        buf.append(line)
        char_count += len(line)
        if char_count >= CHUNK_TARGET:
            content = "".join(prev_tail + buf)
            segs.append(_SyntheticSegment(
                content, buf_start, buf_start + len(buf) - 1,
                chunk_type="text", name=name))
            prev_tail = buf[-FALLBACK_OVERLAP_LINES:] if FALLBACK_OVERLAP_LINES > 0 else []
            buf_start += len(buf)
            buf = []
            char_count = 0
    if buf:
        content = "".join(prev_tail + buf)
        segs.append(_SyntheticSegment(
            content, buf_start, buf_start + len(buf) - 1,
            chunk_type="text", name=name))
    return segs


def _segment_markdown(src: bytes, text_src: str) -> list:
    """Heading-aware split for .md/.markdown, named after heading text
    instead of the generic fallback's nameless line-tiling."""
    lines = text_src.splitlines(keepends=True)
    segs: list = []
    for start, end, name in _split_markdown_sections(lines):
        sec_lines = lines[start - 1:end]
        for seg in _slice_section(sec_lines, name):
            # _slice_section's line numbers are section-local (start at 1);
            # rebase them onto the file's absolute line numbers.
            seg.start_line += start - 1
            seg.end_line += start - 1
            segs.append(seg)
    return _merge_small(segs, src)


def segment_text_file(src: bytes, *, path: str | None = None,
                      counters: dict | None = None) -> list[dict[str, Any]]:
    """Non-AST fallback for config-allowlisted extensions with no grammar.
    Markdown is the one exception, split at headings; everything else is
    nameless line-tiling."""
    text_src = src.decode(errors="replace")
    if _is_markdown_path(path):
        return _finalize(_segment_markdown(src, text_src), src, path, counters)
    lines = text_src.splitlines(keepends=True)
    segs: list = []
    buf: list[str] = []
    buf_start = 1
    char_count = 0
    prev_tail: list[str] = []
    for line in lines:
        buf.append(line)
        char_count += len(line)
        if char_count >= CHUNK_TARGET:
            content = "".join(prev_tail + buf)
            segs.append(_SyntheticSegment(
                content, buf_start, buf_start + len(buf) - 1,
                chunk_type="text", name=None))
            prev_tail = buf[-FALLBACK_OVERLAP_LINES:] if FALLBACK_OVERLAP_LINES > 0 else []
            buf_start += len(buf)
            buf = []
            char_count = 0
    if buf:
        content = "".join(prev_tail + buf)
        segs.append(_SyntheticSegment(
            content, buf_start, buf_start + len(buf) - 1,
            chunk_type="text", name=None))
    return _finalize(segs, src, path, counters)


def _dedup_segments(segs: list, src: bytes) -> list:
    """Drops later segments sharing (start_line, content) with an earlier
    one. Needed because salvage + split can independently re-derive the
    same nested boundary, producing two chunks with the same _chunk_id."""
    seen: set[tuple[int, str]] = set()
    out: list = []
    for seg in segs:
        key = (seg.start_line, seg.content(src))
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)
    return out


def segment_file(src: bytes, lang: str, *, path: str | None = None,
                 counters: dict | None = None, macros: set | None = None,
                 self_heal: bool = True) -> list[dict[str, Any]]:
    """Parses src and returns segment dicts. No chunk exceeds CHUNK_MAX
    (enforced by _finalize). `macros` pre-blanks known engine macros;
    `self_heal` additionally discovers and blanks unknown ones on the fly."""
    # Reset in case a prior file's segment_file call raised before reaching
    # _finalize's own reset.
    _literal_cap_state["capped_chunks"] = 0
    _literal_cap_state["dropped"] = 0
    _literal_extract_cache.clear()

    _LANG_PACK_NAME = _lang_table("grammar")
    parser = get_parser(_LANG_PACK_NAME.get(lang, lang))
    # timeout_micros is deprecated in 0.25, but progress_callback (its
    # replacement) is silently ignored for bytestring source.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        parser.timeout_micros = PARSE_TIMEOUT_MICROS

    # Parse a possibly-blanked buffer for STRUCTURE; content/names below
    # still read from the ORIGINAL src (blanking preserves byte offsets).
    parse_src = _blank_macros(src, macros) if macros else src
    root = parser.parse(parse_src).root_node

    # Discover macros, validate by strict error-count decrease (a real call
    # like ASSERT(x) is rejected since blanking it doesn't help), blank,
    # reparse, repeat (see _MAX_MACRO_PASSES for why one pass isn't enough).
    if self_heal and lang in _MACRO_LANGS and root.has_error:
        healed: set = set()
        for _ in range(_MAX_MACRO_PASSES):
            if not root.has_error:
                break
            base, error_ranges = _errors_with_ranges(root)
            # sorted() pins order before truncating, or a >cap file picks a
            # different subset per run. Qt specifiers unioned in AFTER truncation.
            discovered = sorted(_discover_macros(root, parse_src) - healed)[:_MAX_MACRO_CANDIDATES]
            cands = (set(discovered) | _QT_ACCESS_SPECIFIERS) - healed
            testable = _near_error_ranges(cands, parse_src, error_ranges)
            admitted = {m for m in testable
                        if _count_errors_upto(parser.parse(_blank_macros(parse_src, {m})).root_node,
                                              base) < base}
            if not admitted:
                break
            healed |= admitted
            parse_src = _blank_macros(parse_src, healed)
            root = parser.parse(parse_src).root_node
        if healed and counters is not None:
            counters.setdefault("discovered_macros", set()).update(healed)
            counters["macro_healed"] = 1
        elif not healed and counters is not None:
            # Sweep found nothing to heal; chunker.py persists this by content
            # hash so unchanged unhealable files skip the sweep next time.
            counters["heal_unhealable"] = 1

    # Post-heal: parse_error now means genuinely unparseable, not "had a
    # macro we could heal".
    if counters is not None and root.has_error:
        counters["parse_error"] = 1

    if counters is not None:
        counters["symbols"] = _collect_symbols_from_root(root, lang, src)

    boundary_nodes = _collect_boundaries(root, lang, src, counters)

    # No structural nodes at all (e.g. a pure header): line-split by
    # CHUNK_TARGET with the same fallback overlap as _split_large_node.
    if not boundary_nodes:
        text = src.decode(errors="replace")
        lines = text.splitlines(keepends=True)
        segs: list = []
        buf: list[str] = []
        buf_start = 1
        char_count = 0
        prev_tail: list[str] = []
        for line in lines:
            buf.append(line)
            char_count += len(line)
            if char_count >= CHUNK_TARGET:
                content = "".join(prev_tail + buf)
                segs.append(_SyntheticSegment(content, buf_start, buf_start + len(buf) - 1))
                prev_tail = buf[-FALLBACK_OVERLAP_LINES:] if FALLBACK_OVERLAP_LINES > 0 else []
                buf_start += len(buf)
                buf = []
                char_count = 0
        if buf:
            content = "".join(prev_tail + buf)
            segs.append(_SyntheticSegment(content, buf_start, buf_start + len(buf) - 1))
        return _finalize(segs, src, path, counters)

    # A salvaged container's nested boundaries can be re-derived again by
    # _split_large_node below; precompute their spans and skip the
    # top-level duplicate rather than double-emit the same _chunk_id.
    split_results: dict[int, list] = {}
    covered_spans: set[tuple[int, int]] = set()
    for node in boundary_nodes:
        if _Segment(node, src, lang).size(src) > CHUNK_MAX:
            split_results[id(node)] = _split_large_node(node, lang, src)
            for child in node.children:
                if child.is_named:
                    for b in _collect_boundaries(child, lang, src):
                        covered_spans.add((b.start_byte, b.end_byte))

    raw_segs: list = []
    for node in boundary_nodes:
        if id(node) in split_results:
            raw_segs.extend(split_results[id(node)])
        elif (node.start_byte, node.end_byte) in covered_spans:
            continue  # already represented via an oversize sibling's split
        else:
            raw_segs.append(_Segment(node, src, lang))

    # Defense in depth for any collision the span-level filter above missed.
    raw_segs = _dedup_segments(raw_segs, src)

    # Fold in module-level residue; only re-sort if any was added, so a
    # boundary-only file keeps its already-canonical order.
    residue_segs = _collect_module_residue(root, lang, src)
    if residue_segs:
        raw_segs.extend(residue_segs)

    gap_segs = _fill_coverage_gaps(raw_segs, src)
    if gap_segs:
        raw_segs.extend(gap_segs)

    if residue_segs or gap_segs:
        raw_segs.sort(key=lambda s: s.start_line)
        # Comment-only spans come only from residue/gaps; attach forward.
        raw_segs = _attach_or_drop_comments(raw_segs, src, lang)

    # Absorb identity-free trivia BEFORE _merge_small, a separate pass since
    # the two rules' guards are opposite (see _absorb_identity_free_fragments).
    raw_segs = _absorb_identity_free_fragments(raw_segs, src, lang)
    raw_segs = _merge_small(raw_segs, src)
    return _finalize(raw_segs, src, path, counters)
