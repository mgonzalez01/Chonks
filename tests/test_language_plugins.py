"""`chonks.index.plugins.load_plugins`: the config-driven registry extension
point, its extension collision rule, and its all-or-nothing rebind."""

import logging
import sys
import types

import pytest

import chonks.core.refresh as refresh
import chonks.languages as languages
from chonks.index.plugins import load_plugins
from chonks.languages.spec import LanguageSpec
from chonks.storage.store import Store


class _FakeEmbedder:
    model = "fake"
    url = "http://localhost:9999"

    def embed_documents(self, texts, client=None, **kw):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_queries(self, texts, client=None, **kw):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest.fixture(autouse=True)
def _restore_language_registry():
    """Plugin loading mutates process-global state (chonks.languages.REGISTRY
    and every module the AST completeness test in test_registry_refresh.py
    tracks). Undoing it here, rather than trusting each test to clean up
    after itself, is what keeps this file's tests from leaking into whatever
    runs after them."""
    original = languages.REGISTRY
    yield
    if languages.REGISTRY is not original:
        languages.REGISTRY = original
        languages.EXT_TO_LANG = original.ext_to_lang
        languages.CODE_LANGUAGES = original.code_languages
        refresh.run_refreshes()


def _install_module(monkeypatch, name, languages_tuple=None, with_languages=True):
    module = types.ModuleType(name)
    if with_languages:
        module.LANGUAGES = languages_tuple or ()
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _spec(name, ext, **kw):
    return LanguageSpec(
        name=name,
        grammar=kw.pop("grammar", "python"),
        extensions=frozenset({ext}),
        boundary_nodes=frozenset({"decl"}),
        kind_labels={"decl": "function"},
        **kw,
    )


# ---------------------------------------------------------------------------
# Neutrality: the empty case does nothing and logs nothing.
# ---------------------------------------------------------------------------

def test_empty_list_does_nothing(caplog):
    original = languages.REGISTRY
    caplog.set_level(logging.DEBUG)
    load_plugins([])
    assert languages.REGISTRY is original
    assert caplog.records == []


def test_empty_list_does_not_even_touch_fallback_extensions(caplog):
    caplog.set_level(logging.DEBUG)
    load_plugins([], fallback_extensions=[".made_up"])
    assert caplog.records == []


# ---------------------------------------------------------------------------
# A synthetic plugin, loaded end to end: admitted by pipeline's gate, and a
# real file in it produces a real chunk through a real tree-sitter grammar.
# ---------------------------------------------------------------------------

_PLUGIN_ZIG_SPEC = LanguageSpec(
    name="acme_zig",
    grammar="zig",
    extensions=frozenset({".zig"}),
    boundary_nodes=frozenset({"Decl"}),
    kind_labels={"Decl": "function"},
)


def test_synthetic_plugin_admitted_and_chunks(monkeypatch, tmp_path):
    _install_module(monkeypatch, "acme_chonks_zig", (_PLUGIN_ZIG_SPEC,))

    load_plugins(["acme_chonks_zig"])

    assert languages.REGISTRY.get("acme_zig") is _PLUGIN_ZIG_SPEC
    assert languages.EXT_TO_LANG[".zig"] == "acme_zig"

    import chonks.index.pipeline as pipeline
    assert ".zig" in pipeline._EXT_TO_LANG, (
        "pipeline's admission gate did not pick up the plugin's extension"
    )

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "add.zig").write_text(
        "fn add(a: i32, b: i32) i32 {\n    return a + b;\n}\n"
    )
    store = Store(tmp_path / "test.db")
    from chonks.index.pipeline import index_paths
    result = index_paths([str(src_dir)], store, _FakeEmbedder(), root=src_dir)

    assert result["unsupported_ext_skipped"] == 0, (
        "the plugin's file landed in unsupported_ext_skipped"
    )
    rows = store._conn.execute(
        "SELECT language, chunk_type FROM chunks WHERE path LIKE '%add.zig'"
    ).fetchall()
    store.close()
    assert rows, "no chunk was produced for the plugin-language file"
    assert all(language == "acme_zig" for language, _ in rows)


# ---------------------------------------------------------------------------
# The collision rule: one extension set at a time, plus the .js trap.
# ---------------------------------------------------------------------------

def test_rejects_extension_in_default_fallback_extensions(monkeypatch):
    # .vue is only in DEFAULT_FALLBACK_EXTENSIONS, not in any other admission set.
    _install_module(monkeypatch, "bad_vue_plugin", (_spec("acme_vue", ".vue"),))
    with pytest.raises(ValueError, match=r"\.vue"):
        load_plugins(["bad_vue_plugin"])


def test_rejects_extension_in_data_blob_extensions(monkeypatch):
    _install_module(monkeypatch, "bad_json_plugin", (_spec("acme_json", ".json"),))
    with pytest.raises(ValueError, match=r"\.json"):
        load_plugins(["bad_json_plugin"])


def test_rejects_extension_in_markdown_exts(monkeypatch):
    _install_module(monkeypatch, "bad_markdown_plugin", (_spec("acme_md", ".markdown"),))
    with pytest.raises(ValueError, match=r"\.markdown"):
        load_plugins(["bad_markdown_plugin"])


def test_rejects_css_even_though_no_built_in_owns_it(monkeypatch):
    # .css is in MINIFIED_GUARD_EXTENSIONS but owned by no language spec.
    # The rejection must not depend on Registry's duplicate-ownership check.
    assert ".css" not in languages.EXT_TO_LANG
    _install_module(monkeypatch, "bad_css_plugin", (_spec("acme_css", ".css"),))
    with pytest.raises(ValueError, match=r"\.css"):
        load_plugins(["bad_css_plugin"])


def test_rejects_plugin_claiming_js_despite_built_in_allowlist(monkeypatch):
    # .js/.mjs/.cjs are allowlisted for the built-in "javascript" spec only.
    # A plugin under any other name claiming .js is still refused.
    _install_module(monkeypatch, "bad_js_plugin", (_spec("acme_js", ".js"),))
    with pytest.raises(ValueError, match=r"\.js"):
        load_plugins(["bad_js_plugin"])


def test_rejects_extension_in_effective_configured_fallback_extensions(monkeypatch):
    # .proto is not in DEFAULT_FALLBACK_EXTENSIONS or any other admission set,
    # but a user who configured it into fallback_extensions is still protected.
    assert ".proto" not in languages.EXT_TO_LANG
    _install_module(monkeypatch, "bad_proto_plugin", (_spec("acme_proto", ".proto"),))
    with pytest.raises(ValueError, match=r"\.proto"):
        load_plugins(["bad_proto_plugin"], fallback_extensions=[".proto"])


def test_extension_outside_every_admission_set_is_not_rejected_by_collision_rule(monkeypatch):
    # Sanity check on the rejection tests above: an extension the collision
    # rule has no opinion on loads cleanly.
    _install_module(monkeypatch, "ok_plugin", (_spec("acme_ok", ".acmeok"),))
    load_plugins(["ok_plugin"])
    assert languages.EXT_TO_LANG[".acmeok"] == "acme_ok"


# ---------------------------------------------------------------------------
# All-or-nothing: the third of four plugins is rejected, none is applied.
# ---------------------------------------------------------------------------

def test_third_of_four_rejected_leaves_original_registry(monkeypatch):
    original = languages.REGISTRY

    _install_module(monkeypatch, "p1_plugin", (_spec("acme_p1", ".acmep1"),))
    _install_module(monkeypatch, "p2_plugin", (_spec("acme_p2", ".acmep2"),))
    _install_module(monkeypatch, "p3_plugin", (_spec("acme_p3", ".css"),))  # rejected
    _install_module(monkeypatch, "p4_plugin", (_spec("acme_p4", ".acmep4"),))

    with pytest.raises(ValueError, match=r"\.css"):
        load_plugins(["p1_plugin", "p2_plugin", "p3_plugin", "p4_plugin"])

    assert languages.REGISTRY is original
    assert ".acmep1" not in languages.EXT_TO_LANG
    assert ".acmep4" not in languages.EXT_TO_LANG


# ---------------------------------------------------------------------------
# A module with no LANGUAGES gets a named diagnosis.
# ---------------------------------------------------------------------------

def test_module_missing_languages_attribute_names_module_and_attribute(monkeypatch):
    _install_module(monkeypatch, "no_languages_plugin", with_languages=False)
    with pytest.raises(ValueError, match=r"no_languages_plugin.*LANGUAGES"):
        load_plugins(["no_languages_plugin"])
