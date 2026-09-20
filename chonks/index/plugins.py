"""Loads third-party language modules named in the `language_plugins` config
key into the process-global language registry."""

import importlib
import logging

from chonks.core.refresh import run_refreshes
from chonks.index.admission import (
    DATA_BLOB_EXTENSIONS,
    DEFAULT_FALLBACK_EXTENSIONS,
    MINIFIED_GUARD_EXTENSIONS,
)
from chonks.index.text_segment import _MARKDOWN_EXTS

logger = logging.getLogger("chonks.chunker")

# The one built-in overlap between a language spec and an admission set:
# .js/.mjs/.cjs are owned by the javascript spec (chonks/languages/js_ts.py)
# and are also in MINIFIED_GUARD_EXTENSIONS. Keyed on (extension, spec name),
# not the bare extension: a plugin spec named anything other than
# "javascript" claiming one of these extensions is still rejected below,
# and would in any case collide with the built-in spec's own ownership of
# that extension when the combined registry is built.
_ALLOWED_OVERLAP_PAIRS = {
    (".js", "javascript"),
    (".mjs", "javascript"),
    (".cjs", "javascript"),
}


def load_plugins(names, fallback_extensions=None) -> None:
    """Imports each dotted module name in `names`, validates every spec in
    its `LANGUAGES` against the indexer's admission rules, and — only if
    every plugin passes — rebuilds and rebinds the process-global language
    registry (`chonks.languages.REGISTRY`, `EXT_TO_LANG`, `CODE_LANGUAGES`)
    and runs every module's `_refresh_from_registry` hook.

    `names` empty (the default, no `language_plugins` configured) does
    nothing at all: no import, no registry rebuild, no log record.

    A rejection anywhere in `names` — a missing `LANGUAGES` attribute, an
    extension reserved by the indexer's admission rules, or a spec the
    registry itself refuses (duplicate name, duplicate extension, and the
    other checks in `chonks.languages.Registry.__init__`) — leaves the
    process on the original registry object; nothing is rebound until every
    plugin in `names` has passed.

    `fallback_extensions` is the effective `config["fallback_extensions"]`
    (or `None` to use the built-in default only), checked in addition to
    `DEFAULT_FALLBACK_EXTENSIONS` so a plugin can't claim an extension a
    user has configured into the fallback path even when that extension
    isn't in the shipped default list.
    """
    if not names:
        return

    import chonks.languages as languages

    reserved = (
        set(DEFAULT_FALLBACK_EXTENSIONS)
        | DATA_BLOB_EXTENSIONS
        | MINIFIED_GUARD_EXTENSIONS
        | set(_MARKDOWN_EXTS)
    )
    if fallback_extensions:
        reserved |= {ext.lower() for ext in fallback_extensions}

    plugin_specs = []
    for module_name in names:
        module = importlib.import_module(module_name)
        if not hasattr(module, "LANGUAGES"):
            raise ValueError(
                f"{module_name}: plugin module has no LANGUAGES attribute")
        for spec in module.LANGUAGES:
            for ext in spec.extensions:
                if ext in reserved and (ext, spec.name) not in _ALLOWED_OVERLAP_PAIRS:
                    raise ValueError(
                        f"{spec.name}: extension {ext!r} is reserved by the "
                        f"indexer's admission rules and cannot be claimed by "
                        f"a plugin (from {module_name!r})")
            plugin_specs.append(spec)

    new_registry = languages.Registry(tuple(languages.REGISTRY.specs) + tuple(plugin_specs))

    languages.REGISTRY = new_registry
    languages.EXT_TO_LANG = new_registry.ext_to_lang
    languages.CODE_LANGUAGES = new_registry.code_languages
    run_refreshes()
    logger.info("Loaded language plugin module(s): %s", ", ".join(names))
