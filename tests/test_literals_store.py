"""Literal & message index, store-side tests:
skeleton generation, find_by_message's two matching tiers, chunk_literals
lifecycle (insert / re-index / delete), and the pre-literal-DB graceful note.
"""

import pytest

from chonks.retrieval.message_match import find_by_message
from chonks.storage.store import Store
from chonks.core.skeleton import _compute_skeleton, _longest_skeleton_fragment, _skeleton_match


def _make_store(tmp_path) -> Store:
    return Store(tmp_path / "test.db")


def _mark_complete(store: Store) -> None:
    """Most tests here drive Store directly (insert_chunks), which no longer
    sets _LITERAL_INDEX_META_KEY on its own (see that flag's docstring in
    store.py). Call this to mark the DB as a completed full index run. The
    two pre_literal_db/genuinely_empty_db tests deliberately skip this."""
    store.set_meta("literal_index_version", "1")


def _chunk(cid: str, path: str, content: str, literals: list[tuple[str, int]],
           start_line: int = 1, name: str | None = None) -> dict:
    return {
        "id": cid,
        "path": path,
        "language": "python",
        "chunk_type": "function_definition",
        "name": name or cid,
        "start_line": start_line,
        "end_line": start_line + content.count("\n") + 1,
        "content": content,
        "literals": literals,
    }


_DIM = 8


def _embed() -> list[float]:
    return [0.1] * _DIM


# ---------------------------------------------------------------------------
# Skeleton generation (table-driven)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("no holes here at all", None),
    ("failed to load %s: %d", "failed to load ␀*: ␀*"),
    ("progress %03.2f%% done", "progress ␀*% done"),
    ("literal percent 100%% only", "literal percent 100% only"),
    ("hello {name}, you have {0} items", "hello ␀*, you have ␀* items"),
    ("format spec {0:D4} here", "format spec ␀* here"),
    ("empty braces {} hole", "empty braces ␀* hole"),
    ("escaped {{literal braces}} stay", "escaped {literal braces} stay"),
    ("shell var $NAME here", "shell var ␀* here"),
    ("template ${expr} form", "template ␀* form"),
    ("truncated field %.*s tail", "truncated field ␀* tail"),
    ("star width %-*.3f value", "star width ␀* value"),
    ("value %*d here", "value ␀* here"),
])
def test_compute_skeleton_table_driven(text, expected):
    assert _compute_skeleton(text) == expected


def test_compute_skeleton_percent_hole():
    assert _compute_skeleton("failed to load %s: %d") == "failed to load ␀*: ␀*"


def test_compute_skeleton_percent_precision_and_literal_percent():
    # %03.2f is a hole; the trailing %% is a literal percent sign, not a hole.
    assert _compute_skeleton("progress %03.2f%% done") == "progress ␀*% done"


def test_compute_skeleton_literal_percent_only_no_real_hole():
    assert _compute_skeleton("literal percent 100%% only") == "literal percent 100% only"


def test_compute_skeleton_brace_holes_named_and_positional():
    assert _compute_skeleton("hello {name}, you have {0} items") == "hello ␀*, you have ␀* items"


def test_compute_skeleton_brace_with_format_spec():
    assert _compute_skeleton("format spec {0:D4} here") == "format spec ␀* here"


def test_compute_skeleton_empty_braces():
    assert _compute_skeleton("empty braces {} hole") == "empty braces ␀* hole"


def test_compute_skeleton_escaped_braces_stay_literal():
    assert _compute_skeleton("escaped {{literal braces}} stay") == "escaped {literal braces} stay"


def test_compute_skeleton_dollar_var_and_dollar_brace():
    assert _compute_skeleton("shell var $NAME here") == "shell var ␀* here"
    assert _compute_skeleton("template ${expr} form") == "template ␀* form"


def test_compute_skeleton_no_holes_returns_none():
    assert _compute_skeleton("no holes here at all") is None


def test_longest_skeleton_fragment_picks_max_length_constant_run():
    skeleton = "short ␀* a much longer constant fragment here ␀* tail"
    assert _longest_skeleton_fragment(skeleton) == " a much longer constant fragment here "


def test_skeleton_match_matches_with_pasted_context_slack():
    """The walk is unanchored (like the old re.search), so extra leading/
    trailing context (timestamps, log levels) still matches."""
    skeleton = _compute_skeleton("failed to load %s: %d")
    assert _skeleton_match(skeleton, "failed to load config.json: 404")[0] is True
    assert _skeleton_match(
        skeleton, "[2026-07-28 ERROR] failed to load config.json: 404 (retry 1)"
    )[0] is True
    assert _skeleton_match(skeleton, "completely unrelated message")[0] is False


# ---------------------------------------------------------------------------
# Matching tiers
# ---------------------------------------------------------------------------

def test_exact_tier_matches_verbatim_literal_substring(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.py", 'raise ValueError("unexpected token in stream")',
                [("unexpected token in stream", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "crash: unexpected token in stream at pos 5")
    assert len(result["results"]) == 1
    assert result["results"][0]["match_kind"] == "exact"
    assert result["results"][0]["matched_literal"] == "unexpected token in stream"
    assert result["results"][0]["path"] == "a.py"


def test_skeleton_tier_matches_through_format_hole(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "loader.py",
                'logger.error("failed to load %s: %d" % (name, code))',
                [("failed to load %s: %d", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "failed to load config.json: 404")
    assert len(result["results"]) == 1
    hit = result["results"][0]
    assert hit["match_kind"] == "skeleton"
    assert hit["skeleton"] == "failed to load ␀*: ␀*"
    assert hit["matched_literal"] == "failed to load %s: %d"


def test_exact_beats_skeleton_when_both_present(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [
            _chunk("c_exact", "a.py", 'log("connection refused by remote host")',
                   [("connection refused by remote host", 1)]),
            _chunk("c_skel", "b.py", 'log("connection refused by %s" % host)',
                   [("connection refused by %s", 1)], start_line=5),
        ],
        [_embed(), _embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "connection refused by remote host")
    kinds = [r["match_kind"] for r in result["results"]]
    assert kinds[0] == "exact"
    assert "skeleton" not in kinds or kinds.index("exact") < (
        kinds.index("skeleton") if "skeleton" in kinds else len(kinds)
    )


def test_single_word_literal_is_not_a_coincidental_exact_hit(tmp_path):
    """A bare single-word literal that's a substring of an unrelated pasted
    message must not count as an exact hit."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "config.py", 'x = config["embedder"]', [("embedder", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "the embedder should not be called for fts modes")
    assert result["results"] == []


def test_skeleton_fragment_prefilter_rejects_unrelated_message(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("very specific constant phrase %s" % x)',
                [("very specific constant phrase %s", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "this message shares no constant fragment")
    assert result["results"] == []
    assert result["note"] == "no literal matches for this message"


def test_find_by_message_genuine_miss_returns_honest_empty_note(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("totally unrelated literal text")',
                [("totally unrelated literal text", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "nothing here matches anything at all")
    assert result["results"] == []
    assert result["truncated"] is False
    assert "note" in result and result["note"]


def test_find_by_message_truncation_note(tmp_path):
    store = _make_store(tmp_path)
    chunks = [
        _chunk(f"c{i}", f"f{i}.py", 'log("shared constant phrase for matching test")',
               [("shared constant phrase for matching test", 1)], start_line=i)
        for i in range(5)
    ]
    store.insert_chunks(chunks, [_embed()] * 5, model="test")
    _mark_complete(store)
    result = find_by_message(store, "shared constant phrase for matching test", limit=2)
    assert len(result["results"]) == 2
    assert result["truncated"] is True
    assert "note" in result and "limit=2" in result["note"]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_literals_die_with_their_chunk_on_delete_file(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("distinctive literal marker text")',
                [("distinctive literal marker text", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    assert find_by_message(store, "distinctive literal marker text")["results"]
    store.delete_file("a.py")
    result = find_by_message(store, "distinctive literal marker text")
    assert result["results"] == []
    # chunks table is empty too, so this must not be misread as the
    # pre-literal-extraction case.
    assert result["note"] == "no literal matches for this message"


def test_reindexing_a_changed_file_drops_stale_literals(tmp_path):
    """The incremental path must not leak stale literals: delete_file then
    insert_chunks (the real indexer's pattern) must not leave the old
    literal findable."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("the original stale message text")',
                [("the original stale message text", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    assert find_by_message(store, "the original stale message text")["results"]

    store.delete_file("a.py")
    store.insert_chunks(
        [_chunk("c2", "a.py", 'log("a brand new replacement message text")',
                [("a brand new replacement message text", 1)])],
        [_embed()], model="test",
    )

    stale = find_by_message(store, "the original stale message text")
    assert stale["results"] == []
    fresh = find_by_message(store, "a brand new replacement message text")
    assert len(fresh["results"]) == 1


def test_insert_chunks_replace_of_same_id_drops_old_literals(tmp_path):
    """insert_chunks itself (not just delete_file) must not accumulate stale
    literal rows for a re-inserted chunk id: it deletes chunk_literals for
    the id before re-inserting."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("first version literal text")',
                [("first version literal text", 1)])],
        [_embed()], model="test",
    )
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("second version literal text")',
                [("second version literal text", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    assert find_by_message(store, "first version literal text")["results"] == []
    assert len(find_by_message(store, "second version literal text")["results"]) == 1


# ---------------------------------------------------------------------------
# Pre-literal-DB graceful note
# ---------------------------------------------------------------------------

def test_pre_literal_db_returns_clear_note_not_exception_or_empty_silence(tmp_path):
    store = _make_store(tmp_path)
    with store._lock:
        store._conn.execute(
            "INSERT INTO chunks(id,path,language,chunk_type,name,start_line,end_line,content,indexed_at) "
            "VALUES('c1','a.py','python','function_definition','f',1,2,'def f(): pass',0)"
        )
        store._conn.commit()

    result = find_by_message(store, "anything at all")
    assert result["results"] == []
    assert "re-index" in result["note"]
    assert "literal extraction" in result["note"]


def test_genuinely_empty_db_is_not_mistaken_for_pre_literal_db(tmp_path):
    store = _make_store(tmp_path)
    result = find_by_message(store, "anything at all")
    assert result["results"] == []
    assert "predates" not in (result.get("note") or "")


def test_non_literal_language_db_is_not_mistaken_for_pre_literal_db(tmp_path):
    """A DB indexed entirely in a language with no literal spec (Go, Rust,
    ...) is also chunk_literals-empty, but the meta-flag discriminator
    (not chunk_literals' emptiness) decides this - see _LITERAL_INDEX_META_KEY."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [{
            "id": "c1", "path": "main.go", "language": "go",
            "chunk_type": "function_declaration", "name": "main",
            "start_line": 1, "end_line": 3,
            "content": 'func main() {\n\tfmt.Println("hello")\n}',
            "literals": [],  # go has no _LITERAL_SPECS entry
        }],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "anything at all")
    assert result["results"] == []
    assert "predates" not in (result.get("note") or "")


def test_single_token_literal_matches_when_message_is_the_literal_verbatim(tmp_path):
    """A short literal found as a coincidental substring of a longer message
    is filtered (see test_single_word_literal_is_not_a_coincidental_exact_hit),
    but a pasted message that IS the literal verbatim must still hit."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "s3.py", 'raise S3Error("NoSuchBucket")', [("NoSuchBucket", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "NoSuchBucket")
    assert len(result["results"]) == 1
    assert result["results"][0]["matched_literal"] == "NoSuchBucket"
    assert result["results"][0]["match_kind"] == "exact"


def test_reverse_substring_message_is_elided_fragment_of_longer_literal(tmp_path):
    """A truncated pasted log fragment, shorter than the literal that
    emitted it, must still resolve (the "or vice versa" exact-tier direction)."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.py", 'raise TypeError("expected a mapping, got a list instead")',
                [("expected a mapping, got a list instead", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "expected a mapping, got")
    assert len(result["results"]) == 1
    assert result["results"][0]["matched_literal"] == "expected a mapping, got a list instead"


def test_uppercase_and_or_not_in_message_does_not_break_fts_matching(tmp_path):
    """FTS5 treats bare uppercase AND/OR/NOT as query operators; a pasted
    message containing one uppercase must not lose the whole tier to a
    swallowed MATCH syntax error."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "io.py", 'raise IOError("could NOT open file %s for writing" % path)',
                [("could NOT open file %s for writing", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "could NOT open file /var/log/app.log for writing")
    assert len(result["results"]) == 1
    assert result["results"][0]["match_kind"] == "skeleton"
    assert "note" not in result or "no literal matches" not in (result.get("note") or "")


def test_large_candidate_pool_does_not_silently_drop_the_true_match(tmp_path):
    """Many decoys share tokens with the message; the true target is
    inserted last, so an unordered/truncated pool would drop it."""
    store = _make_store(tmp_path)
    chunks = [
        _chunk(f"decoy{i}", f"decoy{i}.py",
               f'log("shared token decoy number {i} in this file")',
               [(f"shared token decoy number {i} in this file", 1)], start_line=1)
        for i in range(600)
    ]
    chunks.append(
        _chunk("target", "target.py",
               'log("shared token real emitting line for the message")',
               [("shared token real emitting line for the message", 1)], start_line=1)
    )
    store.insert_chunks(chunks, [_embed()] * len(chunks), model="test")
    _mark_complete(store)
    result = find_by_message(store,
        "shared token real emitting line for the message", limit=1000
    )
    matched = {r["matched_literal"] for r in result["results"]}
    assert "shared token real emitting line for the message" in matched


def test_boundary_split_chunks_do_not_duplicate_the_same_literal_hit(tmp_path):
    """A chunker oversize-split can attribute the same (path, line, text)
    literal to several sibling chunks; results must dedupe to one hit,
    preferring the chunk whose line span actually contains the literal's line."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [
            # Sibling split piece that does NOT contain line 50.
            _chunk("piece_a", "big.py", "x = 1\n" * 10,
                   [("distinctive split literal message text", 50)],
                   start_line=1),
            # The piece that actually contains line 50.
            _chunk("piece_b", "big.py", "y = 2\n" * 10,
                   [("distinctive split literal message text", 50)],
                   start_line=45),
        ],
        [_embed(), _embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "distinctive split literal message text")
    assert len(result["results"]) == 1
    assert result["results"][0]["chunk_id"] == "piece_b"


# ---------------------------------------------------------------------------
# End-to-end: real indexer wiring (chonks/index/refs_extract.py -> chonks/index/pipeline.py -> store.py)
# ---------------------------------------------------------------------------

class _FakeEmbedder:
    model = "fake"
    url = "http://localhost:9999"

    def embed_documents(self, texts, client=None, **kw):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_queries(self, texts, client=None, **kw):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def test_index_paths_populates_chunk_literals_end_to_end(tmp_path):
    """The full pipeline, not just the individually-unit-tested pieces,
    resolves a format-hole message."""
    from chonks.index.pipeline import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "loader.py").write_text(
        "def load_asset(name, code):\n"
        '    logger.error("failed to load %s: %d" % (name, code))\n'
    )

    store = Store(tmp_path / "test.db")
    index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir)

    result = find_by_message(store, "failed to load textures/hero.png: 404")
    store.close()

    assert len(result["results"]) == 1
    hit = result["results"][0]
    assert hit["match_kind"] == "skeleton"
    assert hit["path"] == "loader.py"
    assert hit["name"] == "load_asset"


def test_cpp_raw_string_literal_end_to_end(tmp_path):
    """raw_string_literal (R"(...)") was previously absent from
    _LITERAL_SPECS['cpp'].leaf_types, so segment_file returned zero literals
    for this function; the point here is that it's extracted at all."""
    from chonks.index.pipeline import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "archive.cpp").write_text(
        'void open_archive(const char* path) {\n'
        '    const char* s = R"(could not open archive %s)";\n'
        '}\n'
    )

    store = Store(tmp_path / "test.db")
    index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir)

    result = find_by_message(store, "could not open archive %s")
    store.close()

    assert len(result["results"]) == 1
    hit = result["results"][0]
    assert hit["match_kind"] == "exact"
    assert hit["matched_literal"] == "could not open archive %s"
    assert hit["path"] == "archive.cpp"


# ---------------------------------------------------------------------------
# Skeleton tier cost guards (ReDoS)
# ---------------------------------------------------------------------------

def test_over_hole_cap_skeleton_never_stored():
    text = "expected %s, " * 25 + "ok"
    assert _compute_skeleton(text) is None


def test_at_hole_cap_skeleton_still_stored():
    """The cap excludes anything over the limit, not at it."""
    text = "expected %s, " * 8 + "ok"
    assert _compute_skeleton(text) is not None


def test_pathological_high_hole_query_returns_promptly_with_honest_note(tmp_path):
    """The ReDoS this guards against took >25s / >170s pre-fix; must return
    in well under 1 second with an honest empty-result note, not a hang."""
    import time

    store = _make_store(tmp_path)
    text = "expected %s, " * 25 + "ok"
    store.insert_chunks(
        [_chunk("c1", "chunker.py", 'log(f"' + text + '")', [(text, 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    message = "expected 3, " * 29
    t0 = time.monotonic()
    result = find_by_message(store, message)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"took {elapsed:.2f}s — hole cap did not exclude the pathological skeleton"
    assert result["results"] == []
    assert "note" in result and result["note"]


def test_skeleton_budget_is_per_query_not_per_candidate(tmp_path, monkeypatch):
    """The work budget bounds the whole find_by_message call, not each
    candidate row (pre-fix: candidate cap x per-row budget = 40M find calls
    ~= 7.6s at saturation)."""
    import chonks.retrieval.message_match as message_match

    store = _make_store(tmp_path)
    text = "ab,%s," * 8 + "ZZZEND"
    rows = [
        _chunk(f"c{i}", f"f{i}.c", f'printf("{text}");', [(text, 1)])
        for i in range(4)
    ]
    store.insert_chunks(rows, [_embed() for _ in rows], model="test")
    _mark_complete(store)
    monkeypatch.setattr(message_match, "_SKELETON_QUERY_BUDGET", 40)
    # fragment ",ZZZEND" passes the prefilter but is 90 chars past the last
    # hole's 80-char gap window, so every landing-point combination dead-ends.
    result = find_by_message(store, "ab,1,ab,2,ab,3,ab,4,ab,5,ab,6,ab,7,ab,8," + "z" * 90 + ",ZZZEND")
    assert result["results"] == []
    assert result.get("note") and "budget exhausted" in result["note"], result.get("note")
    assert any(ch.isdigit() and int(ch) >= 2 for ch in result["note"] if ch.isdigit()), result["note"]


def test_legacy_over_cap_skeleton_row_excluded_at_query_time_and_announced(tmp_path):
    """A DB written before the storage-time hole cap existed can still carry
    an over-cap skeleton row; find_by_message must re-apply the cap at query
    time, not just trust what extraction stored."""
    import time
    from chonks.core.skeleton import _HOLE_SENTINEL

    store = _make_store(tmp_path)
    _mark_complete(store)
    literal_text = "expected %s, " * 25 + "ok"
    skeleton = ("expected " + _HOLE_SENTINEL + ", ") * 25 + "ok"
    with store._lock:
        store._conn.execute(
            "INSERT INTO chunks(id,path,language,chunk_type,name,start_line,end_line,content,indexed_at) "
            "VALUES('c1','chunker.py','python','function','f',1,2,'x',0)"
        )
        store._conn.execute(
            "INSERT INTO chunk_literals(chunk_id, text, skeleton, line) VALUES(?,?,?,?)",
            ("c1", literal_text, skeleton, 2),
        )
        store._conn.commit()

    message = "expected 3, " * 29
    t0 = time.monotonic()
    result = find_by_message(store, message)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0
    assert result["results"] == []
    assert "8 format holes" in result["note"]


# ---------------------------------------------------------------------------
# Skeleton tier cost guards: budgeted fragment walk (replaces the ReDoS-prone
# regex). Timings below go through the real extractor, not a hand-built
# skeleton string.
# ---------------------------------------------------------------------------

def test_budgeted_walk_short_fragment_8_hole_real_extractor_stays_under_50ms(tmp_path):
    """A single-char constant fragment between holes gives each hole many
    candidate landing points in its 80-char gap window, making backtracking
    genuinely combinatorial. _SKELETON_WORK_BUDGET must bound it regardless:
    well under 50ms with the honest budget-exhausted note."""
    import time
    from chonks.index.segment import segment_file

    src = (
        b'void trace_all(int a) {\n'
        b'    fprintf(stderr, "trace,%s,%s,%s,%s,%s,%s,%s,%s,END\\n", '
        b'a, a, a, a, a, a, a, a);\n'
        b'}\n'
    )
    seg = next(s for s in segment_file(src, "c", path="trace.c") if s["literals"])
    assert seg["literals"][0][0] == "trace,%s,%s,%s,%s,%s,%s,%s,%s,END"

    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "trace.c", seg["content"], seg["literals"],
                start_line=seg["start_line"], name=seg["name"])],
        [_embed()], model="test",
    )
    _mark_complete(store)

    # Comma/digit-dense, under the message-length cap, never contains "END" -
    # forces the walk to exhaust its backtracking budget before concluding no-match.
    message = ("trace," + "9," * 290)[:599]
    assert "END" not in message

    t0 = time.perf_counter()
    result = find_by_message(store, message)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50, f"took {elapsed_ms:.2f}ms — budget did not bound the walk"
    assert result["results"] == []
    assert "verification budget exhausted" in result["note"]


def test_budgeted_walk_25_hole_real_extractor_stays_under_50ms(tmp_path):
    """25 holes exceeds _SKELETON_MAX_HOLES, so no skeleton is stored (see
    test_over_hole_cap_skeleton_never_stored); this pins that the real
    extraction pipeline produces that shape end-to-end and resolves
    near-instantly, not the >25s/>170s pre-fix hang."""
    import time
    from chonks.index.segment import segment_file

    n = 25
    fmt = "expected %s, " * n + "ok"
    src = (
        "def f(a):\n"
        f'    return "{fmt}" % ({", ".join(["a"] * n)})\n'
    ).encode()
    seg = next(s for s in segment_file(src, "python", path="t.py") if s["literals"])
    text = seg["literals"][0][0]
    assert text.count("%s") == n

    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "t.py", seg["content"], seg["literals"],
                start_line=seg["start_line"], name=seg["name"])],
        [_embed()], model="test",
    )
    _mark_complete(store)

    message = "expected 3, " * (n + 4)
    t0 = time.perf_counter()
    result = find_by_message(store, message)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50, f"took {elapsed_ms:.2f}ms"
    assert result["results"] == []
    assert "note" in result and result["note"]


def test_message_over_length_cap_excludes_skeleton_tier_and_announces(tmp_path):
    """A message beyond _SKELETON_MAX_MESSAGE_LEN skips skeleton
    verification (exact/substring tiers are unaffected)."""
    store = _make_store(tmp_path)
    text = "expected %s, expected %s, expected %s, ok"  # 3 holes, well under the hole cap
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("' + text + '")', [(text, 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    message = ("expected 3, " * 60)[:650]
    assert len(message) > 600
    result = find_by_message(store, message)
    assert result["results"] == []
    assert "600 chars" in result["note"]


def test_message_under_length_cap_skeleton_tier_still_works(tmp_path):
    store = _make_store(tmp_path)
    text = "failed to load %s: %d"
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("' + text + '")', [(text, 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "failed to load config.json: 404")
    assert len(result["results"]) == 1
    assert result["results"][0]["match_kind"] == "skeleton"


def test_skeleton_verification_does_not_hold_the_store_lock(tmp_path):
    """The lock is released before any Python-side verification runs: patch
    _skeleton_match to sleep, then confirm a second thread's unrelated store
    call completes quickly while the first is still "verifying"."""
    import threading
    import time
    import chonks.retrieval.message_match as message_match

    store = _make_store(tmp_path)
    text = "slow %s pattern"
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("' + text + '")', [(text, 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)

    real_skeleton_match = message_match._skeleton_match
    verifying = threading.Event()
    release = threading.Event()

    def slow_skeleton_match(skeleton, message, budget=message_match._SKELETON_WORK_BUDGET):
        verifying.set()
        release.wait(timeout=5)
        return real_skeleton_match(skeleton, message, budget)

    message_match._skeleton_match = slow_skeleton_match
    try:
        query_thread = threading.Thread(
            target=find_by_message, args=(store, "slow xyz pattern")
        )
        query_thread.start()
        assert verifying.wait(timeout=2), "skeleton verification never started"

        t0 = time.monotonic()
        store.get_meta("literal_index_version")
        unblocked_elapsed = time.monotonic() - t0
        assert unblocked_elapsed < 1.0, (
            "a concurrent store call was blocked — skeleton verification is "
            "still holding self._lock"
        )
    finally:
        release.set()
        message_match._skeleton_match = real_skeleton_match
        query_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Multi-line literal matching (escape decoding)
# ---------------------------------------------------------------------------

def test_multiline_literal_matches_by_first_line(tmp_path):
    """chonks/index/refs_extract.py keeps interior newlines in a decoded multi-line literal;
    a pasted single-line log line only ever carries the first line."""
    store = _make_store(tmp_path)
    text = "first line of the message\nsecond line never gets pasted alone"
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("' + text + '")', [(text, 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "first line of the message")
    assert len(result["results"]) == 1
    assert result["results"][0]["matched_literal"] == text


def test_multiline_skeleton_matches_by_first_line(tmp_path):
    """The first-line fallback applies to the skeleton tier too: a pasted
    single-line message never carries a multi-line template's later, static
    lines, so the old full-skeleton walk always failed there."""
    store = _make_store(tmp_path)
    text = "first line failure %s\nsecond line detail"
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("' + text + '")', [(text, 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store, "first line failure boom")
    assert len(result["results"]) == 1
    hit = result["results"][0]
    assert hit["match_kind"] == "skeleton"
    assert hit["matched_literal"] == text


def test_multiline_skeleton_matches_by_first_line_end_to_end(tmp_path):
    """Same case as test_multiline_skeleton_matches_by_first_line but through
    the real extractor, where \\n escapes get decoded to real newlines."""
    from chonks.index.pipeline import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "loader.py").write_text(
        "def load(name):\n"
        '    logger.error("first line failure %s\\nsecond line detail\\n" % name)\n'
    )

    store = Store(tmp_path / "test.db")
    index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir)

    result = find_by_message(store, "first line failure boom")
    store.close()

    assert len(result["results"]) == 1
    hit = result["results"][0]
    assert hit["match_kind"] == "skeleton"
    assert hit["path"] == "loader.py"


# ---------------------------------------------------------------------------
# Announced omissions
# ---------------------------------------------------------------------------

def test_low_confidence_exact_gate_filter_is_announced(tmp_path):
    """_passes_exact_gate filters a short unshaped literal found only as a
    coincidental substring; that filtering must be announced, not silently
    dropped."""
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "s3.py", 'raise S3Error("NoSuchBucket")', [("NoSuchBucket", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    result = find_by_message(store,
        "botocore.exceptions.ClientError: An error occurred (NoSuchBucket) "
        "when calling the HeadBucket operation"
    )
    assert result["results"] == []
    assert "low-confidence" in result["note"]


def test_verbatim_literal_survives_a_long_identifier_dense_message(tmp_path):
    """The substring tier used to be skipped above 200 chars, so a verbatim
    literal buried in a longer identifier-dense paste fell through to the
    token/skeleton tier, where the FTS pool's 12-longest-tokens cap could
    also crowd it out. _SUBSTRING_TIER_MAX_MESSAGE_LEN now covers this length."""
    store = _make_store(tmp_path)
    text = "connection refused by remote host"
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("' + text + '")', [(text, 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    decoys = " ".join(f"identifierTokenNumber{i}Extended" for i in range(12))
    message = f"{decoys} {text} {decoys}"
    assert 200 <= len(message) < 2000
    result = find_by_message(store, message)
    assert len(result["results"]) == 1
    assert result["results"][0]["matched_literal"] == text


def test_substring_tier_skip_announced_for_overlong_message(tmp_path):
    """A message at/above _SUBSTRING_TIER_MAX_MESSAGE_LEN skips the
    substring tier; when the token/skeleton tier also can't rescue it, the
    skip must be announced, not absorbed into a clean 'no literal matches'."""
    store = _make_store(tmp_path)
    text = "a specific literal phrase that does not recur"
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("' + text + '")', [(text, 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    message = "x " * 1000  # 2000 chars, no token >= 3 chars at all
    assert len(message) >= 2000
    result = find_by_message(store, message)
    assert result["results"] == []
    assert "direct-substring tier was skipped" in result["note"]


def test_token_pool_cap_announced_when_a_short_fragment_is_crowded_out(tmp_path, monkeypatch):
    """A message with more distinct significant tokens than _TOKEN_POOL_CAP
    can crowd the literal's own shorter discriminating token out of the FTS
    candidate query; that must be announced, not look like a clean no-match."""
    import chonks.retrieval.message_match as message_match

    monkeypatch.setattr(message_match, "_TOKEN_POOL_CAP", 2)
    store = _make_store(tmp_path)
    text = "shortfrag %s here"
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("' + text + '")', [(text, 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    # "shortfrag" loses the length-sort race against two longer decoys for
    # the (patched) 2-token cap, so it never reaches the FTS candidate pool.
    message = "reallyverylongdecoytokenone reallyverylongdecoytokentwo shortfrag boom here"
    result = find_by_message(store, message)
    assert result["results"] == []
    assert "only the 2 longest were used" in result["note"]


def test_candidate_pool_truncation_announced_even_with_results(tmp_path, monkeypatch):
    """candidates_truncated must be announced whenever it happens, not only
    when it also causes an empty result."""
    import chonks.retrieval.message_match as message_match

    monkeypatch.setattr(message_match, "_LITERAL_CANDIDATE_CAP", 2)
    store = _make_store(tmp_path)
    chunks = [
        _chunk(f"c{i}", f"f{i}.py", f'log("shared candidate pool token number {i}")',
               [(f"shared candidate pool token number {i}", 1)], start_line=1)
        for i in range(5)
    ]
    store.insert_chunks(chunks, [_embed()] * 5, model="test")
    _mark_complete(store)
    result = find_by_message(store, "shared candidate pool token number 0")
    assert "candidate pool capped" in (result.get("note") or "")


# ---------------------------------------------------------------------------
# FTS maintenance parity
# ---------------------------------------------------------------------------

def test_rebuild_fts_also_rebuilds_literals_fts(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("distinctive rebuild fts marker text")',
                [("distinctive rebuild fts marker text", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    store.rebuild_fts()
    result = find_by_message(store, "distinctive rebuild fts marker text")
    assert len(result["results"]) == 1


# ---------------------------------------------------------------------------
# Partial-index honesty
# ---------------------------------------------------------------------------

def test_partial_incremental_run_leaves_flag_unset_and_notes_incomplete(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("literal from the one touched file")',
                [("literal from the one touched file", 1)])],
        [_embed()], model="test",
    )
    # Deliberately do NOT call _mark_complete: insert_chunks alone must
    # never set the flag (this is the bug being regression-tested).
    assert store.get_meta("literal_index_version") is None
    result = find_by_message(store, "literal from an untouched file, never indexed")
    assert result["results"] == []
    assert "incomplete" in result["note"]
    assert "--force" in result["note"]


def test_insert_chunks_alone_never_sets_the_completeness_flag(tmp_path):
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.py", 'log("some literal text here")',
                [("some literal text here", 1)])],
        [_embed()], model="test",
    )
    assert store.get_meta("literal_index_version") is None


@pytest.mark.parametrize("text", [
    "one line", "first\nsecond", "first\nsecond\nthird", "\nleading newline",
    "trailing\n", "first\r\nsecond", "héllo wörld\nsecond", "",
])
def test_first_line_sql_matches_first_line(text):
    """The SQL first-line expression must agree with the Python one."""
    import sqlite3

    from chonks.core.skeleton import _FIRST_LINE_SQL, _first_line

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE cl (text TEXT)")
    conn.execute("INSERT INTO cl (text) VALUES (?)", (text,))
    result = conn.execute(f"SELECT {_FIRST_LINE_SQL} FROM cl").fetchone()[0]
    assert result == _first_line(text)


def _store_with_leading_newline_literal(tmp_path) -> Store:
    store = _make_store(tmp_path)
    store.insert_chunks(
        [_chunk("c1", "a.py", 'print("\\nOperation cancelled.")',
                [("\nOperation cancelled.", 1)])],
        [_embed()], model="test",
    )
    _mark_complete(store)
    return store


@pytest.mark.parametrize("message", ["zzz", "the job was cancelled by an admin"])
def test_empty_first_line_matches_no_unrelated_message(tmp_path, message):
    """A literal starting with a newline has an empty first line, which is a
    substring of every message; it must not count as a match."""
    store = _store_with_leading_newline_literal(tmp_path)
    result = find_by_message(store, message)
    assert result["results"] == []


def test_leading_newline_literal_still_matches_its_own_text(tmp_path):
    store = _store_with_leading_newline_literal(tmp_path)
    result = find_by_message(store, "Operation cancelled.")
    assert [r["matched_literal"] for r in result["results"]] == ["\nOperation cancelled."]
