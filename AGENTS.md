# Working on Chonks

Before adding or moving code under `chonks/`, read "Where code goes" and "Code and tests" in CONTRIBUTING.md and put the change in the layer and extension point it names. A per-language fact is a `LanguageSpec` field in `chonks/languages/`, read elsewhere through the registry. `uv run pytest -q tests/test_layering.py` checks the layer rules in about a second.

Before writing a comment, a commit message or a PR description, read "Comments" in CONTRIBUTING.md. A change to ranking, chunking or graph construction carries the measurement behind it in the PR description.

Before finishing, run `scripts/check.sh`. Never add an entry to an exception list in `tests/test_layering.py` without saying why in the pull request, and never add an entry or raise a ceiling in `tests/test_function_size.py`.
