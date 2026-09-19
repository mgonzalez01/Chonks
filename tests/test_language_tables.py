import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "gen_language_tables.py"

spec = importlib.util.spec_from_file_location("gen_language_tables", SCRIPT_PATH)
gen_language_tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_language_tables)


def _between_markers(text, name):
    begin = f"<!-- generated:{name}:begin -->"
    end = f"<!-- generated:{name}:end -->"
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    return text[start:stop].strip("\n")


def test_docs_extensions_table_matches_generator():
    text = (REPO_ROOT / "DOCS.md").read_text()
    assert _between_markers(text, "language-extensions") == gen_language_tables.extensions_table()


def test_docs_boundaries_table_matches_generator():
    text = (REPO_ROOT / "DOCS.md").read_text()
    assert _between_markers(text, "language-boundaries") == gen_language_tables.boundaries_table()


def test_readme_ast_cell_matches_generator():
    text = (REPO_ROOT / "README.md").read_text()
    line = next(line for line in text.split("\n") if line.startswith("| AST-aware"))
    assert line.split("|")[2].strip() == gen_language_tables.readme_ast_cell()


def test_render_raises_on_missing_marker():
    with pytest.raises(ValueError):
        gen_language_tables.render("no markers", "x", "body")


def test_check_mode_reports_clean_tree():
    try:
        result = gen_language_tables.main(["--check"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        assert result is None
