"""Tests for chunker.py correctness fixes."""
import json
import threading
import time
from unittest.mock import MagicMock, patch

from chonks.store import Store


class _FakeEmbedder:
    model = "fake"
    url   = "http://localhost:9999"
    def embed_documents(self, texts, client=None, **kw):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    def embed_queries(self, texts, client=None, **kw):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


# A minimal C++ source whose engine annotation macros (UCLASS/GENERATED_BODY)
# break tree-sitter parsing and are recovered by segment_file's self-heal.
def _uclass_src(name: str) -> str:
    return (
        f"UCLASS()\n"
        f"class {name} : public AActor {{\n"
        f"    GENERATED_BODY()\n"
        f"public:\n"
        f"    void DoThing();\n"
        f"}};\n"
    )


def test_ext_filter_is_case_insensitive(tmp_path):
    """Directory scan must pick up .CPP/.H (Windows-origin files)."""
    from chonks.chunker import index_paths

    cpp_file = tmp_path / "HELLO.CPP"
    cpp_file.write_text("int main() { return 0; }\n")

    class _FakeEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store = Store(tmp_path / "test.db")
    embedder = _FakeEmbedder()

    result = index_paths(
        [str(tmp_path)],
        store,
        embedder,
        root=tmp_path,
    )
    store.close()

    assert result["indexed"] + result["errors"] >= 1, (
        ".CPP file was silently skipped (extension case-sensitivity bug)"
    )


def test_index_populates_decoupled_symbols(tmp_path):
    """End-to-end: indexing a small class (folded into ONE chunk) still populates
    the symbols table with the class and each method, mapped to the owning chunk."""
    from chonks.chunker import index_paths

    (tmp_path / "widget.cpp").write_text(
        "class Widget {\npublic:\n"
        "  int get_x() const { return x_; }\n"
        "  void set_x(int v) { x_ = v; }\n"
        "private:\n  int x_ = 0;\n};\n"
    )

    class _FakeEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)

    names = {s["name"] for s in store.get_all_symbols()}
    assert {"Widget", "get_x", "set_x"} <= names, f"symbols not populated end-to-end: {names}"
    hits = store.find_symbols("get_x")
    assert hits and hits[0]["chunk_id"], "symbol not mapped to its owning chunk_id"
    store.close()


def test_index_concurrent_returns_409(tmp_path):
    from fastapi.testclient import TestClient
    import chonks.server as server
    barrier = threading.Barrier(2)
    release = threading.Event()

    def slow_index(*args, **kwargs):
        barrier.wait()
        release.wait()
        return {"indexed": 0, "skipped": 0, "errors": 0, "pruned": 0, "truncated": 0}

    server._projects.clear()
    # Seed the project dict as main() would, including index_lock
    server._projects[server.DEFAULT_PROJECT] = {
        "db_path":      tmp_path / "test.db",
        "root":         None,
        "exclude":      [],
        "include":      [],
        "research_cfg": {},
        "repomap_cfg":  {},
        "search_cfg":   {},
        "edge_type_weights": {},
        "embedder":     MagicMock(),
        "embed_batch":    128,
        "embed_inflight": 2,
        "store":        MagicMock(),
        "searcher":     None,
        "index_lock":   threading.Lock(),
    }

    with patch("chonks.serve.app.index_paths", side_effect=slow_index):
        client = TestClient(server.app, raise_server_exceptions=False)

        results = []

        def do_index():
            r = client.post("/index", json={"paths": [str(tmp_path)]})
            results.append(r.status_code)

        t = threading.Thread(target=do_index)
        t.start()
        barrier.wait()

        r2 = client.post("/index", json={"paths": [str(tmp_path)]})
        assert r2.status_code == 409, f"Expected 409, got {r2.status_code}: {r2.text}"

        release.set()
        t.join()


def test_macro_vocab_persists_when_recurring(tmp_path):
    """A macro that heals >= _MACRO_PERSIST_MIN_FILES files this run is written
    to meta['macro_vocab'] so future runs pre-blank it instead of rediscovering."""
    from chonks.chunker import index_paths

    (tmp_path / "a.cpp").write_text(_uclass_src("AThing"))
    (tmp_path / "b.cpp").write_text(_uclass_src("BThing"))

    store = Store(tmp_path / "test.db")
    result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    # Both files needed self-heal on this first run (vocab started empty).
    assert result["macro_healed_files"] == 2

    vocab = set(json.loads(store.get_meta("macro_vocab") or "[]"))
    store.close()
    assert {"UCLASS", "GENERATED_BODY"} <= vocab, (
        f"recurring macros not persisted: {vocab}"
    )


def test_macro_vocab_persists_on_crlf_sources(tmp_path):
    """Same as above but with Windows line endings, written as bytes so the
    fixture is CRLF on every platform: discovery tell (e), the only tell that
    sees in-body GENERATED_BODY-style macros, must match lines ending \\r\\n."""
    from chonks.chunker import index_paths

    (tmp_path / "a.cpp").write_bytes(_uclass_src("AThing").replace("\n", "\r\n").encode())
    (tmp_path / "b.cpp").write_bytes(_uclass_src("BThing").replace("\n", "\r\n").encode())

    store = Store(tmp_path / "test.db")
    result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    assert result["macro_healed_files"] == 2

    vocab = set(json.loads(store.get_meta("macro_vocab") or "[]"))
    store.close()
    assert {"UCLASS", "GENERATED_BODY"} <= vocab, (
        f"recurring macros not persisted on CRLF sources: {vocab}"
    )


def test_macro_vocab_recurrence_filter_excludes_single_file(tmp_path):
    """A macro healing only ONE file is a candidate false-admit and must not be
    persisted; the recurrence filter (_MACRO_PERSIST_MIN_FILES) drops it."""
    from chonks.chunker import index_paths

    # Single file: each healed macro heals exactly one file -> below threshold.
    (tmp_path / "only.cpp").write_text(_uclass_src("Solo"))

    store = Store(tmp_path / "test.db")
    result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    assert result["macro_healed_files"] == 1

    persisted = store.get_meta("macro_vocab")
    store.close()
    # Nothing qualified, so the vocab key was never written.
    assert persisted is None, f"single-file macro leaked into vocab: {persisted}"


def test_second_run_loads_persisted_vocab_and_preblanks(tmp_path):
    """A second index run loads meta['macro_vocab'] and pre-blanks those macros,
    so the files parse cleanly without paying the self-heal trial-reparse cost
    (observable as macro_healed_files dropping to 0)."""
    from chonks.chunker import index_paths

    (tmp_path / "a.cpp").write_text(_uclass_src("AThing"))
    (tmp_path / "b.cpp").write_text(_uclass_src("BThing"))

    store = Store(tmp_path / "test.db")
    first = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    assert first["macro_healed_files"] == 2  # discovered + healed on first pass

    # Force re-index the same files: vocab is now persisted and pre-blanked, so
    # no file should need self-heal this run.
    second = index_paths([str(tmp_path)], store, _FakeEmbedder(),
                         root=tmp_path, force=True)
    store.close()
    assert second["macro_healed_files"] == 0, (
        "persisted vocab was not pre-blanked on the second run "
        f"(macro_healed_files={second['macro_healed_files']})"
    )


def test_macros_param_preblanks_first_run(tmp_path):
    """A caller-supplied `macros` set is pre-blanked on the very first run, so
    even a single file skips self-heal (config override path)."""
    from chonks.chunker import index_paths

    (tmp_path / "only.cpp").write_text(_uclass_src("Solo"))

    store = Store(tmp_path / "test.db")
    result = index_paths([str(tmp_path)], store, _FakeEmbedder(),
                         root=tmp_path, macros={"UCLASS", "GENERATED_BODY"})
    store.close()
    assert result["macro_healed_files"] == 0, (
        "explicit macros= override did not pre-blank on first run"
    )


def test_unhealable_hash_persists_and_skips_heal_sweep_on_second_run(tmp_path):
    """A file whose heal sweep ran and admitted nothing gets its content hash
    persisted to meta['unhealable_hashes']. A later run, even with --force on
    unchanged content, must skip the heal sweep entirely (one parse, not ~25)."""
    import tree_sitter
    from chonks.chunker import index_paths

    # Irreducible parse error, no macro-shaped tokens anywhere near it: the heal
    # sweep runs on the first pass and admits nothing.
    (tmp_path / "broken.cpp").write_text(
        "class Broken\n"       # missing ';' -> permanent parse error
        "class Other {\n"
        "    void run() { do_thing(); }\n"
        "};\n"
    )

    store = Store(tmp_path / "test.db")
    first = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    assert first["parse_error_files"] == 1
    assert first["macro_healed_files"] == 0

    hashes = json.loads(store.get_meta("unhealable_hashes") or "[]")
    assert len(hashes) == 1, f"unhealable content hash not persisted: {hashes}"

    orig_parse = tree_sitter.Parser.parse
    calls = {"n": 0}

    def counting_parse(self, *a, **kw):
        calls["n"] += 1
        return orig_parse(self, *a, **kw)

    with patch.object(tree_sitter.Parser, "parse", counting_parse):
        second = index_paths([str(tmp_path)], store, _FakeEmbedder(),
                             root=tmp_path, force=True)
    store.close()

    assert second["parse_error_files"] == 1, "memoized run must report the same diagnostics"
    assert calls["n"] == 1, (
        f"heal sweep ran again on memoized-unhealable content ({calls['n']} parse calls, want 1)"
    )


def test_unhealable_memo_invalidated_when_vocab_grows(tmp_path):
    """The unhealable-content memo is keyed on content hash only, but whether
    self-heal succeeds also depends on the pre-blanked vocab. A fingerprint of
    that vocab is persisted alongside the memo; if it changes, the memo drops."""
    import tree_sitter
    from chonks.chunker import index_paths

    # NOT_A_FIX(1) is a discoverable macro-call candidate (unlike the plain
    # "class Broken" fixture, which discovers zero), but blanking it doesn't
    # repair the real error, so the sweep runs to completion and admits nothing.
    (tmp_path / "broken.cpp").write_text(
        "NOT_A_FIX(1);\n"
        "class Broken\n"
        "class Other {\n"
        "    void run() { do_thing(); }\n"
        "};\n"
    )

    store = Store(tmp_path / "test.db")
    first = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    assert first["parse_error_files"] == 1
    assert first["macro_healed_files"] == 0

    hashes_before = json.loads(store.get_meta("unhealable_hashes") or "[]")
    assert len(hashes_before) == 1, f"unhealable content hash not persisted: {hashes_before}"
    fp_before = store.get_meta("unhealable_vocab_fingerprint")
    assert fp_before is not None, "vocab fingerprint must be persisted alongside the memo"

    # Simulate the persisted macro vocab growing across unrelated files/runs:
    # mutate meta['macro_vocab'] directly rather than staging a second
    # multi-file run just to get a recurrence-qualifying macro admitted.
    vocab = json.loads(store.get_meta("macro_vocab") or "[]")
    store.set_meta("macro_vocab", json.dumps(sorted(set(vocab) | {"SOME_NEW_MACRO"})))

    orig_parse = tree_sitter.Parser.parse
    calls = {"n": 0}

    def counting_parse(self, *a, **kw):
        calls["n"] += 1
        return orig_parse(self, *a, **kw)

    with patch.object(tree_sitter.Parser, "parse", counting_parse):
        second = index_paths([str(tmp_path)], store, _FakeEmbedder(),
                             root=tmp_path, force=True)

    assert second["parse_error_files"] == 1, "file is still genuinely unparseable"
    assert calls["n"] > 1, (
        "heal sweep did not re-run after the persisted macro vocab changed "
        f"({calls['n']} parse call(s)) — the memo should have been invalidated"
    )

    hashes_after = json.loads(store.get_meta("unhealable_hashes") or "[]")
    assert len(hashes_after) == 1, "memo should be rebuilt after invalidation"
    fp_after = store.get_meta("unhealable_vocab_fingerprint")
    store.close()
    assert fp_after != fp_before, "fingerprint must be updated to reflect the new vocab"


def test_scan_admits_ts_js_extensions(tmp_path):
    from chonks.chunker import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "a.ts").write_text("export function f(x: number) { return x + 1; }\n")
    (src_dir / "b.tsx").write_text("export function App() { return null; }\n")
    (src_dir / "c.js").write_text("export function g(x) { return x * 2; }\n")
    (src_dir / "d.jsx").write_text("export function H() { return null; }\n")

    store = Store(tmp_path / "test.db")
    result = index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir)
    store.close()

    assert result["indexed"] == 4, result
    assert result["unsupported_ext_skipped"] == 0


def test_scan_admits_fallback_extensions_by_default(tmp_path):
    from chonks.chunker import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "index.html").write_text("<html>\n<body>hi</body>\n</html>\n")
    (src_dir / "README.md").write_text("# Title\n\nSome docs.\n")

    store = Store(tmp_path / "test.db")
    result = index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir)
    store.close()

    assert result["indexed"] == 2, result
    assert result["unsupported_ext_skipped"] == 0


def test_fallback_extensions_config_can_disable_path(tmp_path):
    from chonks.chunker import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "index.html").write_text("<html><body>hi</body></html>\n")

    store = Store(tmp_path / "test.db")
    result = index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir,
                         fallback_extensions=[])
    store.close()

    assert result["indexed"] == 0
    assert result["unsupported_ext_skipped"] == 1


def test_unsupported_extension_counted_regardless_of_config(tmp_path):
    """A file with an extension in neither _EXT_TO_LANG nor fallback_extensions
    must be tallied in unsupported_ext_skipped, so the gap is never silent."""
    from chonks.chunker import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (src_dir / "real.py").write_text("def f():\n    return 1\n")

    store = Store(tmp_path / "test.db")
    result = index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir)
    store.close()

    assert result["indexed"] == 1  # only real.py
    assert result["unsupported_ext_skipped"] == 1  # image.png


def test_oversize_json_skipped_and_counted(tmp_path):
    """A .json file over the data_blob_size_limit is skipped rather than
    line-sliced into hundreds of ranking-competing chunks."""
    from chonks.chunker import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    big = json.dumps({"items": list(range(50_000))})
    (src_dir / "dump.json").write_text(big)
    assert len(big.encode()) > 256 * 1024, "test setup: file must exceed the default limit"
    (src_dir / "real.py").write_text("def f():\n    return 1\n")

    store = Store(tmp_path / "test.db")
    result = index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir)
    store.close()

    assert result["indexed"] == 1  # only real.py
    assert result["data_blob_skipped"] == 1  # dump.json


def test_small_json_still_indexed(tmp_path):
    """A .json file under the size limit keeps current behaviour: chunked via
    the line-based fallback, not affected by the data-blob guard."""
    from chonks.chunker import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "config.json").write_text(json.dumps({"a": 1, "b": 2}))

    store = Store(tmp_path / "test.db")
    result = index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir)
    store.close()

    assert result["indexed"] == 1
    assert result["data_blob_skipped"] == 0


def test_minified_js_bundle_skipped_but_legit_big_js_kept(tmp_path):
    """The minified-bundle guard: an oversize .js file is skipped only when
    minified-dense (avg line length beyond MINIFIED_AVG_LINE_LEN). An equally
    big but normally-formatted .js file is untouched; size alone never gates."""
    from chonks.chunker import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    # ~300KB on one line: a minifier's output shape.
    minified = "var a=1;" * 40_000
    (src_dir / "bundle.min.js").write_text(minified)
    assert len(minified.encode()) > 256 * 1024
    # Same weight, hand-written shape: short lines.
    legit = "".join(f"function f_{i}() {{\n  return {i};\n}}\n" for i in range(9_000))
    (src_dir / "app.js").write_text(legit)
    assert len(legit.encode()) > 256 * 1024

    store = Store(tmp_path / "test.db")
    result = index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir)
    store.close()

    assert result["data_blob_skipped"] == 1  # bundle.min.js only
    assert result["indexed"] == 1            # app.js chunked normally


def test_oversize_html_skipped_small_html_kept(tmp_path):
    """.html joins the size-gated data set (vendored web_dist bundles); a
    small hand-written HTML page keeps current fallback behaviour."""
    from chonks.chunker import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "huge.html").write_text("<div>x</div>\n" * 25_000)
    (src_dir / "page.html").write_text("<html><body>hello</body></html>\n")

    store = Store(tmp_path / "test.db")
    result = index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir)
    store.close()

    assert result["data_blob_skipped"] == 1  # huge.html
    assert result["indexed"] == 1            # page.html


def test_data_blob_size_limit_zero_restores_old_behavior(tmp_path):
    """data_blob_size_limit=0 disables the guard entirely: every data file
    gets chunked regardless of size, matching pre-guard behaviour."""
    from chonks.chunker import index_paths

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    big = json.dumps({"items": list(range(50_000))})
    (src_dir / "dump.json").write_text(big)

    store = Store(tmp_path / "test.db")
    result = index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir,
                         data_blob_size_limit=0)
    store.close()

    assert result["indexed"] == 1  # dump.json chunked, not skipped
    assert result["data_blob_skipped"] == 0


def test_config_seed_not_persisted(tmp_path):
    """A caller/config `macros` seed is applied at runtime but must not be baked
    into meta['macro_vocab']; only auto-discovered, recurring macros are durable.
    Prevents a typo'd or later-removed seed sticking in the DB forever."""
    from chonks.chunker import index_paths

    # UCLASS/GENERATED_BODY recur across 2 files -> they qualify and persist.
    (tmp_path / "a.cpp").write_text(_uclass_src("AThing"))
    (tmp_path / "b.cpp").write_text(_uclass_src("BThing"))

    store = Store(tmp_path / "test.db")
    # Seed a macro that does NOT appear in the corpus, so it can only enter the
    # persisted vocab via the (now-fixed) seed-baking path.
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path,
                macros={"BOGUS_SEED_MACRO"})
    vocab = set(json.loads(store.get_meta("macro_vocab") or "[]"))
    store.close()
    assert {"UCLASS", "GENERATED_BODY"} <= vocab, f"discovered macros not persisted: {vocab}"
    assert "BOGUS_SEED_MACRO" not in vocab, f"config seed wrongly baked into the DB: {vocab}"


# --------------------------------------------------------------------------
# Exclude-dir walk pruning
# --------------------------------------------------------------------------

def test_excluded_dir_is_pruned_from_walk_not_just_filtered(tmp_path):
    """A user-configured exclude prefix on a directory (e.g. node_modules/)
    must stop os.walk from descending into it at all, not merely filter its
    files out one by one after the fact. Zero chunks from the excluded tree,
    and the walk-prune counter reflects the skipped directory."""
    from chonks.chunker import index_paths

    nm = tmp_path / "node_modules" / "some_pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("export function f(x) { return x; }\n")
    (tmp_path / "real.py").write_text("def f():\n    return 1\n")

    store = Store(tmp_path / "test.db")
    result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path,
                         exclude=["node_modules/"])
    stored_paths = store.get_paths_under("")
    store.close()

    assert result["indexed"] == 1  # only real.py
    assert not any(p.startswith("node_modules/") for p in stored_paths)
    assert result["dirs_pruned"] >= 1


def test_include_rescues_deep_subtree_under_pruned_dir(tmp_path):
    """exclude=["tmp/"] + include=["tmp/git/a/"] must still index the deeper,
    more-specific include even though tmp/ itself is excluded; the walk-prune
    optimization must not short-circuit the existing include-override contract."""
    from chonks.chunker import index_paths

    other = tmp_path / "tmp" / "other"
    other.mkdir(parents=True)
    (other / "x.py").write_text("def x():\n    return 1\n")

    kept = tmp_path / "tmp" / "git" / "a"
    kept.mkdir(parents=True)
    (kept / "y.py").write_text("def y():\n    return 2\n")

    store = Store(tmp_path / "test.db")
    result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path,
                         exclude=["tmp/"], include=["tmp/git/a/"])
    stored_paths = store.get_paths_under("")
    store.close()

    assert result["indexed"] == 1  # only y.py
    assert "tmp/git/a/y.py" in stored_paths
    assert not any(p.startswith("tmp/other") for p in stored_paths)
    # tmp/other/ has no rescuing include underneath it, so it's prunable;
    # tmp/ and tmp/git/ and tmp/git/a/ must NOT be pruned (include reaches in).
    assert result["dirs_pruned"] >= 1


def test_dir_should_prune_helper():
    """Direct unit coverage of the include-aware prune decision."""
    from chonks.chunker import _dir_should_prune

    excludes = ["tmp/"]
    includes = ["tmp/git/a/"]

    # On the path to the include: never prune.
    assert _dir_should_prune("tmp", excludes, includes) is False
    assert _dir_should_prune("tmp/git", excludes, includes) is False
    assert _dir_should_prune("tmp/git/a", excludes, includes) is False
    # Sibling with no rescuing include: prune.
    assert _dir_should_prune("tmp/other", excludes, includes) is True
    # No matching exclude at all: never prune, regardless of includes.
    assert _dir_should_prune("src", excludes, includes) is False


def test_include_rescues_subtree_nested_below_include_prefix(tmp_path):
    """Regression: exclude=["vendor/"] + include=["vendor/keep_me/"] must
    rescue everything nested under vendor/keep_me/, not just that dir itself.
    _dir_should_prune previously missed the case of being INSIDE a rescued subtree."""
    from chonks.chunker import index_paths, _dir_should_prune

    junk = tmp_path / "vendor" / "junk"
    junk.mkdir(parents=True)
    (junk / "x.py").write_text("def x():\n    return 1\n")

    deep = tmp_path / "vendor" / "keep_me" / "deep"
    deep.mkdir(parents=True)
    (deep / "lib.py").write_text("def y():\n    return 2\n")

    store = Store(tmp_path / "test.db")
    result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path,
                         exclude=["vendor/"], include=["vendor/keep_me/"])
    stored_paths = store.get_paths_under("")
    store.close()

    assert result["indexed"] == 1  # only lib.py
    assert "vendor/keep_me/deep/lib.py" in stored_paths
    assert not any(p.startswith("vendor/junk") for p in stored_paths)
    # vendor/junk/ has no rescuing include underneath it, so it's prunable.
    assert result["dirs_pruned"] >= 1

    # Direct unit coverage of the nested-below-include case.
    excludes = ["vendor/"]
    includes = ["vendor/keep_me/"]
    assert _dir_should_prune("vendor/keep_me/deep", excludes, includes) is False
    assert _dir_should_prune("vendor/keep_me", excludes, includes) is False
    assert _dir_should_prune("vendor/junk", excludes, includes) is True


def test_index_paths_reports_per_phase_timing(tmp_path):
    """The summary used to report only embed-phase elapsed time, silently
    omitting build_refs/build_neighbors/folder-summary time (real wall time was
    ~2x the printed number). Must now break out each phase plus a grand total."""
    from chonks.chunker import index_paths

    (tmp_path / "a.py").write_text("def a():\n    return 1\n")
    (tmp_path / "b.py").write_text("def b():\n    return a()\n")

    store = Store(tmp_path / "test.db")
    result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    store.close()

    # New, additive keys; must not clobber the pre-existing `elapsed_s`/
    # `chunks_per_s` contract that CLI/HTTP callers already parse.
    for key in (
        "embed_elapsed_s", "fts_elapsed_s", "refs_elapsed_s", "knn_elapsed_s",
        "summaries_elapsed_s", "total_elapsed_s",
    ):
        assert key in result, f"missing phase-timing key: {key}"
        assert isinstance(result[key], float)
        assert result[key] >= 0.0

    # embed_elapsed_s is still the historical elapsed_s value (back-compat).
    assert result["embed_elapsed_s"] == result["elapsed_s"]

    # The grand total must cover at least the sum of the phases we can
    # attribute; it's measured from the same t0 as embed_elapsed_s and stops
    # after the last post-pass.
    phase_sum = (
        result["embed_elapsed_s"] + result["fts_elapsed_s"]
        + result["refs_elapsed_s"] + result["knn_elapsed_s"]
        + result["summaries_elapsed_s"]
    )
    assert result["total_elapsed_s"] >= phase_sum - 0.05  # rounding slack


def test_chunker_version_stamped_clean_on_full_run(tmp_path):
    """A run that reprocesses every file (nothing skipped) is a clean stamp of
    the current code version, no matter what was there before."""
    from chonks.chunker import index_paths
    from chonks.chunking import CHUNKER_VERSION

    (tmp_path / "a.py").write_text("def a():\n    return 1\n")
    store = Store(tmp_path / "test.db")
    store.set_meta("chunker_version", "0")  # simulate an older DB
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    assert store.get_meta("chunker_version") == str(CHUNKER_VERSION)
    store.close()


def test_chunker_version_mixed_when_incremental_run_skips_files(tmp_path):
    """A small incremental run that leaves most files untouched must not
    silently overwrite a stale chunker_version with a clean match; some
    chunks in the DB still carry the old boundaries."""
    from chonks.chunker import index_paths
    from chonks.chunking import CHUNKER_VERSION

    (tmp_path / "a.py").write_text("def a():\n    return 1\n")
    (tmp_path / "b.py").write_text("def b():\n    return 2\n")
    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)

    # Simulate the DB having been created under an older chunker version, then
    # touch only one of the two files and re-run without --force so the other
    # file's content hash matches and it gets skipped.
    store.set_meta("chunker_version", "0")
    (tmp_path / "a.py").write_text("def a():\n    return 999\n")
    result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)

    assert result["skipped"] >= 1, "expected b.py to be skipped as unchanged"
    stamped = store.get_meta("chunker_version")
    assert stamped != str(CHUNKER_VERSION)
    assert stamped.startswith("mixed:")
    assert f"0+{CHUNKER_VERSION}" in stamped
    store.close()


def test_chunker_version_self_heals_on_force_reindex(tmp_path):
    """A --force run always reprocesses every file, so it can always clean up
    a previously-mixed chunker_version stamp."""
    from chonks.chunker import index_paths
    from chonks.chunking import CHUNKER_VERSION

    (tmp_path / "a.py").write_text("def a():\n    return 1\n")
    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    store.set_meta("chunker_version", "mixed: 0+1 (1 unchanged file(s) retain old chunk boundaries)")

    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path, force=True)
    assert store.get_meta("chunker_version") == str(CHUNKER_VERSION)
    store.close()


def test_literal_index_flag_set_clean_on_full_run(tmp_path):
    """A run that reprocesses every file (nothing skipped) sets the literal
    index completeness flag; mirrors test_chunker_version_stamped_clean_
    on_full_run for the literal_index_version meta key."""
    from chonks.chunker import index_paths

    (tmp_path / "a.py").write_text('def a():\n    x = "a distinctive literal in file a"\n')
    store = Store(tmp_path / "test.db")
    assert store.get_meta("literal_index_version") is None
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    assert store.get_meta("literal_index_version") == "1"
    store.close()


def test_literal_index_flag_stays_unset_on_partial_incremental_run(tmp_path):
    """Build a normal DB, strip it to a genuine pre-upgrade state (no
    literals, no flag), then an incremental run touching only one of two
    files must leave the flag unset, not look like a clean no-match."""
    from chonks.chunker import index_paths

    (tmp_path / "a.py").write_text('def a():\n    x = "hello from a, a distinctive literal"\n')
    (tmp_path / "b.py").write_text('def b():\n    y = "hello from b, another distinctive literal"\n')
    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)

    # Simulate "this DB predates literal extraction" on top of otherwise
    # normal files/content-hash bookkeeping (a real upgrade's starting state).
    with store._lock:
        store._conn.execute("DELETE FROM chunk_literals")
        store._conn.execute("DELETE FROM meta WHERE key='literal_index_version'")
        store._conn.commit()

    # Touch only a.py; a plain (non-force) run skips b.py as unchanged.
    (tmp_path / "a.py").write_text('def a():\n    x = "hello from a, a distinctive literal, updated"\n')
    result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)

    assert result["indexed"] == 1
    assert result["skipped"] == 1
    assert store.get_meta("literal_index_version") is None

    # b.py's (untouched) message must get the honest "incomplete" note, not
    # a bare no-match; chunk_literals now has SOME rows (from a.py).
    note = store.find_by_message("hello from b, another distinctive literal")["note"]
    assert "incomplete" in note
    assert "--force" in note
    store.close()


def test_literal_index_flag_self_heals_on_force_reindex(tmp_path):
    """--force always reprocesses every file, so it can establish the
    completeness flag even from an unflagged, partially-covered DB, and the
    previously-unreachable file's message resolves afterward."""
    from chonks.chunker import index_paths

    (tmp_path / "a.py").write_text('def a():\n    x = "hello from a, a distinctive literal"\n')
    (tmp_path / "b.py").write_text('def b():\n    y = "hello from b, another distinctive literal"\n')
    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    with store._lock:
        store._conn.execute("DELETE FROM chunk_literals")
        store._conn.execute("DELETE FROM meta WHERE key='literal_index_version'")
        store._conn.commit()

    result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path, force=True)

    assert result["skipped"] == 0
    assert store.get_meta("literal_index_version") == "1"
    hit = store.find_by_message("hello from b, another distinctive literal")
    assert len(hit["results"]) == 1
    store.close()


def test_literal_index_flag_stays_unset_when_force_covers_only_a_subset(tmp_path):
    """`--force <one-of-two-tracked-paths>` is the gate's blind spot:
    state["skipped"] == 0 is trivially true when the run only got one file,
    even though the DB tracks two. The flag must stay unset."""
    from chonks.chunker import index_paths

    (tmp_path / "a.py").write_text('def a():\n    x = "hello from a, a distinctive literal"\n')
    (tmp_path / "b.py").write_text('def b():\n    y = "hello from b, another distinctive literal"\n')
    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    with store._lock:
        store._conn.execute("DELETE FROM chunk_literals")
        store._conn.execute("DELETE FROM meta WHERE key='literal_index_version'")
        store._conn.commit()

    # force=True, but scoped to a.py only; b.py stays tracked in the DB,
    # untouched this run.
    result = index_paths([str(tmp_path / "a.py")], store, _FakeEmbedder(),
                          root=tmp_path, force=True)

    assert result["skipped"] == 0  # force never skips what it's handed
    assert result["indexed"] == 1  # only a.py was in scope this run
    assert store.tracked_file_count() == 2  # b.py is still a tracked file
    assert store.get_meta("literal_index_version") is None

    note = store.find_by_message("hello from b, another distinctive literal")["note"]
    assert "incomplete" in note
    assert "--force" in note
    store.close()


def test_literal_index_flag_once_set_survives_a_later_partial_run(tmp_path):
    """Once the flag is set (a prior full/--force run), a later small
    incremental run must not unset it; untouched files' chunks are still
    accurately literal-covered from before."""
    from chonks.chunker import index_paths

    (tmp_path / "a.py").write_text('def a():\n    x = "hello from a, a distinctive literal"\n')
    (tmp_path / "b.py").write_text('def b():\n    y = "hello from b, another distinctive literal"\n')
    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    assert store.get_meta("literal_index_version") == "1"

    (tmp_path / "a.py").write_text('def a():\n    x = "hello from a, a distinctive literal, updated"\n')
    result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)

    assert result["skipped"] == 1  # b.py unchanged
    assert store.get_meta("literal_index_version") == "1"  # stays set
    store.close()


def test_exclude_include_persisted_to_meta(tmp_path):
    """doctor.py's staleness sweep needs the excludes a DB was actually built
    with, not just whatever's in a config.json that may no longer exist;
    index_paths must persist them."""
    from chonks.chunker import index_paths

    (tmp_path / "src.py").write_text("x = 1\n")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "dep.py").write_text("y = 2\n")

    store = Store(tmp_path / "test.db")
    index_paths(
        [str(tmp_path)], store, _FakeEmbedder(), root=tmp_path,
        exclude=["vendor/"], include=["vendor/keep/"],
    )
    assert json.loads(store.get_meta("exclude")) == ["vendor/"]
    assert json.loads(store.get_meta("include")) == ["vendor/keep/"]
    store.close()


def test_index_db_resolution_prefers_cli_then_config(tmp_path, monkeypatch):
    """`chonks index --config cfg.json` must write the config's db,
    not the argparse default .db/chonks.db; an explicit --db still wins."""
    import json
    import chonks.chunker as chunker

    cfg = tmp_path / "config.json"
    cfg_db = tmp_path / "from-config.db"
    json.dump({"db": str(cfg_db)}, open(cfg, "w"))

    captured = {}
    class FakeStore:
        def __init__(self, db): captured["db"] = str(db)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stats(self): return {}
        def path_family_rows(self): return []
    monkeypatch.setattr(chunker, "Store", FakeStore)
    import collections
    monkeypatch.setattr(chunker, "index_paths",
                        lambda *a, **k: collections.defaultdict(int))

    src = tmp_path / "src"; src.mkdir()
    for argv, expected in [
        (["--config", str(cfg), str(src)], str(cfg_db)),
        (["--config", str(cfg), "--db", str(tmp_path / "cli.db"), str(src)], str(tmp_path / "cli.db")),
    ]:
        captured.clear()
        try:
            chunker.main(argv)
        except SystemExit:
            pass
        assert captured.get("db") == expected, (argv, captured)


def _bulk_chunks(prefix: str, path: str, language: str, n: int) -> list[dict]:
    return [
        {"id": f"{prefix}{i}", "path": path, "language": language,
         "chunk_type": "text", "name": f"{prefix}{i}", "start_line": i, "end_line": i,
         "content": "x"}
        for i in range(n)
    ]


def test_index_summary_warns_on_docs_dominated_family(tmp_path, monkeypatch, capsys):
    """The end-of-index one-liner must fire the same dominance
    warning doctor.py's breakdown section does, sourced from the real DB
    state left behind by index_paths (not a re-derivation of it)."""
    import collections
    import chonks.chunker as chunker

    def fake_index_paths(paths, store, embedder, **kwargs):
        chunks = (
            _bulk_chunks("d", "docs/mirror.html", "html", 84)
            + _bulk_chunks("s", "src/a.py", "python", 16)
        )
        store.insert_chunks(chunks, [[0.1, 0.2, 0.3, 0.4] for _ in chunks])
        store.commit()
        return collections.defaultdict(int)

    monkeypatch.setattr(chunker, "index_paths", fake_index_paths)

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")

    try:
        chunker.main(["--db", str(tmp_path / "test.db"), str(src)])
    except SystemExit:
        pass

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "docs/" in out
    assert '"exclude": ["docs/"]' in out


def test_index_summary_silent_for_healthy_src_dominant_corpus(tmp_path, monkeypatch, capsys):
    """Normal code dominance (src/ = 90% of chunks, all code) must not warn."""
    import collections
    import chonks.chunker as chunker

    def fake_index_paths(paths, store, embedder, **kwargs):
        chunks = (
            _bulk_chunks("s", "src/a.py", "python", 90)
            + _bulk_chunks("d", "docs/x.md", "md", 10)
        )
        store.insert_chunks(chunks, [[0.1, 0.2, 0.3, 0.4] for _ in chunks])
        store.commit()
        return collections.defaultdict(int)

    monkeypatch.setattr(chunker, "index_paths", fake_index_paths)

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")

    try:
        chunker.main(["--db", str(tmp_path / "test.db"), str(src)])
    except SystemExit:
        pass

    out = capsys.readouterr().out
    assert "WARNING" not in out


def _padded_func(name: str, marker: str) -> str:
    """A single top-level function padded well past CHUNK_MIN (300 bytes) so
    it stays its own chunk instead of merging with a neighbour."""
    lines = [f"def {name}(x):", f"    # {marker}"]
    for i in range(12):
        lines.append(f"    x = x + {i}  # padding line {i} to push this well past CHUNK_MIN bytes")
    lines.append("    return x")
    return "\n".join(lines) + "\n"


class _AlwaysOkEmbedder:
    model = "fake"
    url   = "http://localhost:9999"
    def embed_documents(self, texts, client=None, **kw):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    def embed_queries(self, texts, client=None, **kw):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def test_dropped_chunk_still_completes_file(tmp_path):
    """Regression (chunker.py commit_phase): a file with one chunk that
    permanently fails to embed and one that succeeds must still get a
    completed `files` row, so a later edit's delete actually fires instead
    of leaving the surviving chunk orphaned forever."""
    from chonks.chunker import _file_hash, index_paths

    class _PartialFailEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            if any("DROPME" in t for t in texts):
                # Not an httpx.HTTPStatusError -> _should_truncate_and_retry
                # returns False -> permanent drop, not a transient retry.
                raise RuntimeError("simulated permanent embed failure")
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    src_file = tmp_path / "partial.py"
    src_file.write_text(_padded_func("good_fn", "GOODMARK") + "\n" + _padded_func("bad_fn", "DROPME"))

    store = Store(tmp_path / "test.db")
    embedder = _PartialFailEmbedder()
    result = index_paths([str(tmp_path)], store, embedder, root=tmp_path)

    assert result["errors"] >= 1, "setup invariant broken: expected the DROPME chunk to error"

    stored_hash = store.get_file_hash("partial.py")
    assert stored_hash is not None, "files row was never written for a file with a dropped chunk"
    assert stored_hash == _file_hash(src_file)

    rows = store._conn.execute(
        "SELECT name FROM chunks WHERE path=?", ("partial.py",)
    ).fetchall()
    names = {r["name"] for r in rows}
    assert "good_fn" in names, f"surviving chunk missing: {names}"
    assert "bad_fn" not in names, f"dropped chunk should never be inserted: {names}"

    symbol_names = {
        s["name"] for s in store.get_all_symbols() if s["path"] == "partial.py"
    }
    assert "good_fn" in symbol_names, f"symbols not written for completed file: {symbol_names}"

    src_file.write_text("def replaced_fn(z):\n    return z\n")
    index_paths([str(tmp_path)], store, _AlwaysOkEmbedder(), root=tmp_path, force=False)

    rows_after = store._conn.execute(
        "SELECT name FROM chunks WHERE path=?", ("partial.py",)
    ).fetchall()
    names_after = {r["name"] for r in rows_after}
    assert "good_fn" not in names_after, (
        "old surviving chunk was orphaned — the later edit's delete was skipped "
        f"(stored_hash never got set): {names_after}"
    )
    assert "replaced_fn" in names_after
    store.close()


def test_file_symbols_set_before_any_chunk_is_enqueued(tmp_path, monkeypatch):
    """Regression: parser_worker used to set file_symbols[path] AFTER
    enqueuing the file's chunks, letting the embedder complete and drop the
    symbols first. Hooks queue.Queue.put to assert ordering deterministically."""
    import sys
    import queue as queue_module

    from chonks.chunker import index_paths

    (tmp_path / "solo.py").write_text("def solo_fn(x):\n    return x + 1\n")
    target_path = "solo.py"

    violations: list[object] = []
    real_queue_put = queue_module.Queue.put

    def hooked_put(self, item, *a, **kw):
        # Only react to embed_q chunk dicts for our target file; parse_q
        # items are (fpath, hash, stored_path) tuples, and _SENTINEL is a
        # bare object(), neither of which match this shape.
        if isinstance(item, dict) and item.get("path") == target_path and "content" in item:
            caller_frame = sys._getframe(1)  # parser_worker's own frame
            file_symbols = caller_frame.f_locals.get("file_symbols")
            if not (file_symbols and target_path in file_symbols):
                violations.append(dict(file_symbols) if file_symbols else file_symbols)
        return real_queue_put(self, item, *a, **kw)

    monkeypatch.setattr(queue_module.Queue, "put", hooked_put)

    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)

    assert violations == [], (
        f"chunk(s) for {target_path!r} were enqueued onto embed_q before "
        f"file_symbols[{target_path!r}] was populated — the embedder can "
        f"complete the file first and drop its symbols: {violations}"
    )

    symbol_names = {
        s["name"] for s in store.get_all_symbols() if s["path"] == target_path
    }
    assert "solo_fn" in symbol_names, f"symbols not written at all: {symbol_names}"
    store.close()


def test_knn_backend_config_key(tmp_path, monkeypatch):
    """`knn_backend` in config.json arms the backend for the run (bridged via
    CHONKS_KNN_BACKEND, which _Corpus reads); an already-set env var wins,
    and an unknown value is rejected at startup instead of silently running numpy."""
    import json
    import chonks.chunker as chunker

    class FakeStore:
        def __init__(self, db): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def stats(self): return {}
        def path_family_rows(self): return []
    monkeypatch.setattr(chunker, "Store", FakeStore)
    import collections
    monkeypatch.setattr(chunker, "index_paths",
                        lambda *a, **k: collections.defaultdict(int))
    src = tmp_path / "src"; src.mkdir()

    import os
    import pytest

    # NOT monkeypatch-managed: main() writes CHONKS_KNN_BACKEND into os.environ
    # directly, which monkeypatch.delenv can't unwind for an initially-absent
    # key; pop explicitly around every run so nothing leaks into later tests.
    def run(cfg_dict, env_value):
        cfg = tmp_path / "config.json"
        json.dump({"db": str(tmp_path / "t.db"), **cfg_dict}, open(cfg, "w"))
        os.environ.pop("CHONKS_KNN_BACKEND", None)
        if env_value is not None:
            os.environ["CHONKS_KNN_BACKEND"] = env_value
        try:
            chunker.main(["--config", str(cfg), str(src)])
            return os.environ.get("CHONKS_KNN_BACKEND")
        except SystemExit as e:
            if e.code not in (None, 0):
                raise
            return os.environ.get("CHONKS_KNN_BACKEND")
        finally:
            pass

    try:
        assert run({"knn_backend": "cuda"}, None) == "cuda"        # config arms it
        assert run({"knn_backend": "cuda"}, "mlx") == "mlx"        # env wins
        assert run({"knn_backend": "numpy"}, None) == "numpy"     # explicit numpy forces CPU under the auto default
        assert run({"knn_backend": "auto"}, None) is None          # auto = the default, no bridge needed
        assert run({}, None) is None                               # absent = untouched
        with pytest.raises(SystemExit):                            # typo rejected loudly
            run({"knn_backend": "cudda"}, None)
    finally:
        os.environ.pop("CHONKS_KNN_BACKEND", None)


def test_watchdog_aborts_on_wedged_embedder(tmp_path):
    """An embedder that hangs forever inside embed_documents (no
    exception, no timeout of its own) must trip the no-progress watchdog
    instead of hanging the run forever. A tiny no_progress_timeout makes the
    stall observable in test time without actually waiting minutes."""
    import pytest

    from chonks.chunker import NoProgressError, index_paths

    (tmp_path / "solo.py").write_text("def solo_fn(x):\n    return x + 1\n")

    class _WedgedEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            threading.Event().wait()  # never returns, simulates a hung embedder
            return []
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store = Store(tmp_path / "test.db")
    with pytest.raises(NoProgressError, match="localhost:9999") as excinfo:
        index_paths(
            [str(tmp_path)], store, _WedgedEmbedder(),
            root=tmp_path, no_progress_timeout=0.3,
        )
    store.close()

    # The diagnostic must name the in-flight file and carry the embedded/queued
    # counters; without them a recurrence of a past multi-minute silent wedge
    # can't be told apart from a startup-wedge.
    msg = str(excinfo.value)
    assert "solo.py" in msg
    assert "embedded=" in msg and "queued" in msg


def test_watchdog_names_the_embedding_file_not_the_parser_position(tmp_path):
    """The abort must name what the EMBEDDER is stuck on, not `current_file`
    (the parser runs ahead across embed_q, so it's an arbitrary distance past
    the wedge). Only one file's content wedges the embedder here."""
    import pytest

    from chonks.chunker import NoProgressError, index_paths

    # Only this file's content hangs the embedder. Everything else drains, so
    # the parser (and the embedder) move well past it before the wedge bites.
    (tmp_path / "wedge_target.py").write_text("def wedge_fn(x):\n    return WEDGE_SENTINEL\n")
    for i in range(12):
        (tmp_path / f"ordinary_{i:02d}.py").write_text(f"def m{i}(x):\n    return x + {i}\n")

    class _SelectivelyWedgedEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            if any("WEDGE_SENTINEL" in t for t in texts):
                threading.Event().wait()  # never returns, for this batch only
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store = Store(tmp_path / "test.db")
    with pytest.raises(NoProgressError) as excinfo:
        index_paths(
            [str(tmp_path)], store, _SelectivelyWedgedEmbedder(),
            root=tmp_path, embed_batch=1, no_progress_timeout=1.0,
        )
    store.close()

    msg = str(excinfo.value)
    embedding_part = msg.split("Embedding:")[1].split("(chunks embedded")[0]
    assert "wedge_target.py" in embedding_part, (
        f"did not name the file the embedder is actually stuck on: {msg}"
    )


def test_watchdog_does_not_fire_on_slow_but_moving_batches(tmp_path):
    """A batch that's merely slow, not wedged, must never trip the watchdog
    as long as each one completes and resets the clock. Each file embeds
    with a per-batch delay comfortably under the timeout."""
    from chonks.chunker import index_paths

    for i in range(4):
        (tmp_path / f"f{i}.py").write_text(f"def fn_{i}(x):\n    return x + {i}\n")

    class _SlowEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            time.sleep(0.05)  # well under the watchdog timeout below
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store = Store(tmp_path / "test.db")
    result = index_paths(
        [str(tmp_path)], store, _SlowEmbedder(),
        root=tmp_path, embed_batch=1, no_progress_timeout=1.0,
    )
    store.close()

    assert result["indexed"] == 4


def test_watchdog_does_not_fire_during_bisection_recovery(tmp_path):
    """Bisection recovery on a pathological chunk can spend many legitimate
    round trips inside one embed_phase, never reaching commit_phase.
    Resetting the clock only on commit killed a productive recovery as a hang."""
    from chonks.chunker import index_paths

    (tmp_path / "crc.py").write_text(
        "CRC_TABLE = [\n" + "".join(f"    {i},\n" for i in range(64)) + "]\n"
    )
    for i in range(3):
        (tmp_path / f"f{i}.py").write_text(f"def fn_{i}(x):\n    return x + {i}\n")

    class _OffenderEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            time.sleep(0.08)  # each round trip is fast; the SEQUENCE is long
            if any("CRC_TABLE" in t for t in texts):
                raise RuntimeError("400: input exceeds per-slot token budget")
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store = Store(tmp_path / "test.db")
    result = index_paths(
        [str(tmp_path)], store, _OffenderEmbedder(),
        root=tmp_path, embed_batch=8, no_progress_timeout=0.3,
    )
    store.close()

    # The three healthy files still index; the offender is dropped, not fatal.
    assert result["indexed"] == 4
    assert result["errors"] >= 1


def test_watchdog_tolerates_long_scan_tail_with_idle_embedder(tmp_path):
    """The embedder-quiet clock alone must not abort a healthy run: an
    incremental run whose one changed file embeds early leaves the embedder
    idle while the scanner grinds through a long unchanged tail. Progress
    on the scan alone must keep the run alive."""
    from chonks.chunker import index_paths

    # "a_" sorts first so the walk hits the (soon-to-be) changed file before
    # the unchanged tail on the second run.
    (tmp_path / "a_changed.py").write_text("def head(x):\n    return x\n")
    for i in range(18):
        (tmp_path / f"tail_{i:02d}.py").write_text(f"def t{i}(x):\n    return x + {i}\n")

    class _InstantEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    # Run 1: index everything at full speed so run 2 has an unchanged tail.
    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _InstantEmbedder(), root=tmp_path)
    store.close()

    class _SlowHashStore(Store):
        """Each per-file freshness lookup stalls, stretching the scan far past
        the watchdog timeout while the liveness counters keep moving."""
        def get_file_hash(self, path):
            time.sleep(0.1)
            return super().get_file_hash(path)

    # Run 2: one changed file up front, then 18 unchanged files at 0.1s each,
    # ~1.8s of embedder silence against a 0.4s timeout.
    (tmp_path / "a_changed.py").write_text("def head(x):\n    return x * 2\n")
    store = _SlowHashStore(tmp_path / "test.db")
    result = index_paths(
        [str(tmp_path)], store, _InstantEmbedder(),
        root=tmp_path, embed_batch=1, no_progress_timeout=0.4,
    )
    chunk_count = store.stats()["chunks"]
    store.close()

    assert result["indexed"] == 1
    assert result["skipped"] == 18
    assert chunk_count == 19


def test_watchdog_still_aborts_when_whole_pipeline_freezes(tmp_path):
    """The idle branch must not become a blind spot: with nothing in flight
    AND the scanner/parser counters frozen past the timeout (here the scan
    wedges permanently inside a freshness lookup), the watchdog still aborts
    instead of hanging forever."""
    import pytest

    from chonks.chunker import NoProgressError, index_paths

    for i in range(3):
        (tmp_path / f"f{i}.py").write_text(f"def fn_{i}(x):\n    return x + {i}\n")

    class _WedgedScanStore(Store):
        def get_file_hash(self, path):
            threading.Event().wait()  # never returns, simulates a wedged scan
            return None

    class _InstantEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store = _WedgedScanStore(tmp_path / "test.db")
    with pytest.raises(NoProgressError, match="scanner/parser frozen"):
        index_paths(
            [str(tmp_path)], store, _InstantEmbedder(),
            root=tmp_path, no_progress_timeout=0.4,
        )
    store.close()


def test_watchdog_tolerates_long_prune_pass(tmp_path):
    """The orphan-prune sweep after the walk is real work that moves neither
    the scan nor the embed counters; it must register as liveness via the
    per-file pruned count, not get aborted as a frozen pipeline."""
    import os

    from chonks.chunker import index_paths

    for i in range(8):
        (tmp_path / f"gone_{i}.py").write_text(f"def g{i}(x):\n    return x + {i}\n")
    (tmp_path / "keeper.py").write_text("def keep(x):\n    return x\n")

    class _InstantEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _InstantEmbedder(), root=tmp_path)
    store.close()

    for i in range(8):
        os.remove(tmp_path / f"gone_{i}.py")

    class _SlowDeleteStore(Store):
        """Each orphan delete stalls, stretching the prune pass far past the
        watchdog timeout with no scan/parse/embed counter movement."""
        def delete_file(self, path):
            time.sleep(0.15)
            return super().delete_file(path)

    # 8 deletes x 0.15s = ~1.2s of prune against a 0.4s timeout; nothing to
    # embed after keeper.py is skipped as unchanged. Must complete, not abort.
    store = _SlowDeleteStore(tmp_path / "test.db")
    result = index_paths(
        [str(tmp_path)], store, _InstantEmbedder(),
        root=tmp_path, no_progress_timeout=0.4,
    )
    remaining = store.stats()["chunks"]
    store.close()

    assert result["pruned"] == 8
    assert remaining == 1


def test_circuit_breaker_aborts_on_persistently_dead_embedder(tmp_path):
    """An embedder that is actually DOWN fails every batch whole; pre-fix each
    failure bisected and dropped silently, and the no-progress watchdog can't
    catch it since a fast failure resets its clock like a fast success."""
    import httpx
    import pytest

    from chonks.chunker import EmbedderDownError, index_paths

    for i in range(20):
        (tmp_path / f"f{i}.py").write_text(f"def fn_{i}(x):\n    return x + {i}\n")

    class _DeadEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            raise httpx.ConnectError("connection refused")
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store = Store(tmp_path / "test.db")
    try:
        with pytest.raises(EmbedderDownError, match="localhost:9999"):
            index_paths(
                [str(tmp_path)], store, _DeadEmbedder(),
                root=tmp_path, embed_batch=1,
            )
    finally:
        store.close()


def test_circuit_breaker_does_not_trip_on_4xx_oversize_batch(tmp_path):
    """Circuit-breaker guard rail: a batch failing on legitimate oversize
    content (HTTP 4xx) must not trip the dead-embedder breaker, even on every
    batch. The breaker is for a genuinely unreachable embedder, not a tight context window."""
    import httpx

    from chonks.chunker import index_paths

    for i in range(20):
        (tmp_path / f"f{i}.py").write_text(f"def fn_{i}(x):\n    return x + {i}\n")

    def _http_400():
        req = httpx.Request("POST", "http://x/v1/embeddings")
        return httpx.HTTPStatusError(
            "context length exceeded", request=req,
            response=httpx.Response(400, request=req),
        )

    class _OversizeEverythingEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, **kw):
            raise _http_400()  # every chunk looks oversize to this "model"
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store = Store(tmp_path / "test.db")
    try:
        result = index_paths(
            [str(tmp_path)], store, _OversizeEverythingEmbedder(),
            root=tmp_path, embed_batch=4,
        )
    finally:
        store.close()

    # Every chunk is unembeddable (dropped after hitting EMBED_MIN_CHARS), but
    # the run completes rather than aborting; no EmbedderDownError raised.
    assert result["errors"] == 20
    assert result["indexed"] == 20  # each file "completes" with zero surviving chunks


def test_embed_timeout_scales_with_batch_size(tmp_path):
    """A fixed 120s timeout regardless of --embed-batch means a large batch
    against a heavy model can exceed it and get misclassified as a dead
    connection. The call site must scale timeout with request size."""
    import pytest

    from chonks.embedder import (
        EMBED_TIMEOUT_CEILING_S,
        EMBED_TIMEOUT_FLOOR_S,
        EMBED_TIMEOUT_PER_ITEM_S,
        compute_embed_timeout,
    )

    # Small batch: floored at the historical fixed default, never worse off.
    assert compute_embed_timeout(1) == EMBED_TIMEOUT_FLOOR_S
    assert compute_embed_timeout(10) == EMBED_TIMEOUT_FLOOR_S

    # A batch large enough to exceed the floor scales linearly with size.
    big = int(EMBED_TIMEOUT_FLOOR_S / EMBED_TIMEOUT_PER_ITEM_S) + 200
    assert compute_embed_timeout(big) == pytest.approx(big * EMBED_TIMEOUT_PER_ITEM_S)

    # Capped so a pathological batch size can't hang a request indefinitely.
    assert compute_embed_timeout(10_000_000) == EMBED_TIMEOUT_CEILING_S


def test_embed_documents_called_with_scaled_timeout(tmp_path):
    """Same contract as test_embed_timeout_scales_with_batch_size, but pinned
    at the actual call site (chunker.embed_phase), proving the override is
    threaded through, not just defined."""
    from chonks.chunker import index_paths
    from chonks.embedder import compute_embed_timeout

    (tmp_path / "solo.py").write_text("def solo_fn(x):\n    return x + 1\n")

    _NO_TIMEOUT_GIVEN = object()
    seen_timeouts = []

    class _TimeoutRecordingEmbedder:
        model = "fake"
        url   = "http://localhost:9999"
        def embed_documents(self, texts, client=None, timeout=_NO_TIMEOUT_GIVEN, **kw):
            seen_timeouts.append((len(texts), timeout))
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        def embed_queries(self, texts, client=None, **kw):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _TimeoutRecordingEmbedder(), root=tmp_path)
    store.close()

    assert seen_timeouts, "embed_documents was never called"
    # index_paths also calls embed_documents for folder-summary text (see
    # build_folder_summaries) with no timeout override; that's out of scope.
    # What matters is the main chunk-embed call site (embed_phase/_embed_live).
    scaled_calls = [(n, t) for n, t in seen_timeouts if t is not _NO_TIMEOUT_GIVEN]
    assert scaled_calls, f"no call site passed an explicit timeout: {seen_timeouts}"
    for n, timeout in scaled_calls:
        assert timeout == compute_embed_timeout(n)


def test_rebuild_knn_skips_build_refs_when_refs_present(tmp_path, monkeypatch):
    """--rebuild-knn must route to build_neighbors + pagerank + folder
    summaries without re-running build_refs when chunk_refs is already
    persisted; build_refs dominates wall clock at scale for no benefit."""
    import pytest

    import chonks.chunker as chunker

    calls: list[str] = []
    monkeypatch.setattr(chunker, "build_refs",
                         lambda *a, **k: calls.append("build_refs") or 0)
    monkeypatch.setattr(chunker, "build_neighbors",
                         lambda *a, **k: calls.append("build_neighbors") or 0)
    monkeypatch.setattr(chunker, "persist_pagerank",
                         lambda *a, **k: calls.append("persist_pagerank") or 0)
    monkeypatch.setattr(chunker, "build_folder_summaries",
                         lambda *a, **k: calls.append("build_folder_summaries")
                         or {"refreshed": 0, "pruned": 0})

    store = Store(tmp_path / "test.db")
    store.insert_refs([("a", "b")])  # chunk_refs non-empty
    store.commit()
    store.close()

    with pytest.raises(SystemExit) as exc:
        chunker.main(["--db", str(tmp_path / "test.db"), "--rebuild-knn"])
    assert exc.value.code in (None, 0)

    assert calls == ["build_neighbors", "persist_pagerank", "build_folder_summaries"], (
        f"--rebuild-knn must skip build_refs when chunk_refs is populated: {calls}"
    )


def test_rebuild_knn_falls_back_to_full_chain_when_refs_empty(tmp_path, monkeypatch):
    """--rebuild-knn has nothing to reuse when chunk_refs is empty (the
    normal --rebuild-graphs case, or a never-graphed DB); it must fall back
    to running build_refs too rather than leaving the graph unbuilt."""
    import pytest

    import chonks.chunker as chunker

    calls: list[str] = []
    monkeypatch.setattr(chunker, "build_refs",
                         lambda *a, **k: calls.append("build_refs") or 0)
    monkeypatch.setattr(chunker, "build_neighbors",
                         lambda *a, **k: calls.append("build_neighbors") or 0)
    monkeypatch.setattr(chunker, "persist_pagerank",
                         lambda *a, **k: calls.append("persist_pagerank") or 0)
    monkeypatch.setattr(chunker, "build_folder_summaries",
                         lambda *a, **k: calls.append("build_folder_summaries")
                         or {"refreshed": 0, "pruned": 0})

    store = Store(tmp_path / "test.db")  # chunk_refs empty
    store.close()

    with pytest.raises(SystemExit) as exc:
        chunker.main(["--db", str(tmp_path / "test.db"), "--rebuild-knn"])
    assert exc.value.code in (None, 0)

    assert calls == ["build_refs", "build_neighbors", "persist_pagerank",
                      "build_folder_summaries"]


def test_rebuild_graphs_and_rebuild_knn_are_mutually_exclusive(tmp_path):
    import pytest

    import chonks.chunker as chunker

    with pytest.raises(SystemExit):
        chunker.main(["--db", str(tmp_path / "test.db"),
                      "--rebuild-graphs", "--rebuild-knn"])


def test_symlinked_file_does_not_duplicate_index_entry(tmp_path, caplog):
    """scan_producer's emit() guards against a symlink making one real file
    reachable via two on-disk paths, which would double-index it. Locks in
    the observable guarantee: a symlinked duplicate is indexed exactly once."""
    import os
    import pytest

    from chonks.chunker import index_paths

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_file = real_dir / "foo.py"
    real_file.write_text("def foo():\n    return 1\n")

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    link = other_dir / "link.py"
    try:
        os.symlink(real_file, link)
    except (OSError, NotImplementedError):
        pytest.skip("os.symlink not permitted on this platform")

    assert link.resolve() == real_file.resolve(), (
        "test setup invalid: symlink does not resolve back to the real file"
    )

    store = Store(tmp_path / "test.db")
    with caplog.at_level("WARNING"):
        result = index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)

    # The core regression guarantee: the file is scanned/indexed exactly once,
    # never twice, no matter which on-disk route emit() saw first.
    assert result["indexed"] == 1, (
        f"expected exactly one file indexed, got {result['indexed']} "
        f"(symlink dedup regressed): {result}"
    )

    file_rows = store._conn.execute("SELECT path FROM files").fetchall()
    assert len(file_rows) == 1, f"duplicate file rows: {[r['path'] for r in file_rows]}"

    chunk_rows = store._conn.execute("SELECT id, path FROM chunks").fetchall()
    chunk_ids = [r["id"] for r in chunk_rows]
    assert len(chunk_ids) == len(set(chunk_ids)), (
        f"duplicate chunk ids from double-indexing the symlinked file: {chunk_ids}"
    )
    paths_seen = {r["path"] for r in chunk_rows}
    assert paths_seen == {"real/foo.py"}, (
        f"expected chunks stored once under real/foo.py, got: {paths_seen}"
    )

    # This construction exercises the earlier `seen`-set dedup, not the
    # `scanned_stored` branch; that branch's specific warning never fires here.
    scanned_stored_warnings = [
        r.message for r in caplog.records
        if "second on-disk path" in r.message
    ]
    assert scanned_stored_warnings == [], (
        "unexpected: the scanned_stored branch fired for this construction "
        f"({scanned_stored_warnings}) — if this starts passing with a "
        "non-empty list, the branch has become reachable and this test "
        "should be updated to assert on it directly"
    )
    store.close()
