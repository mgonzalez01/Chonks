# Working on Chonks

Before adding or moving code under `chonks/`, read "Where code goes" in CONTRIBUTING.md and put the change in the layer and extension point it names. A per-language fact is a `LanguageSpec` field in `chonks/languages/`, read elsewhere through the registry.

`uv run pytest -q tests/test_layering.py` checks the mechanical rules in about a second. Never add an entry to an exception list in that test without saying why in the pull request.

A change to ranking, chunking or graph construction needs the measurement behind it; CONTRIBUTING.md says what a pull request should include.
