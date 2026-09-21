"""Generate the language tables in DOCS.md and README.md from chonks.languages.describe()."""
from __future__ import annotations

import sys
from pathlib import Path

from chonks.languages import describe

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = REPO_ROOT / "DOCS.md"
README_PATH = REPO_ROOT / "README.md"


def _rows() -> list[dict[str, object]]:
    return sorted(describe(), key=lambda row: row["language"].lower())


def extensions_table() -> str:
    lines = ["| Language | Extensions |", "|---|---|"]
    for row in _rows():
        extensions = " ".join(row["extensions"])
        lines.append(f"| {row['language']} | {extensions} |")
    return "\n".join(lines)


def boundaries_table() -> str:
    lines = ["| Language | Extensions | Detected boundaries |", "|---|---|---|"]
    for row in _rows():
        extensions = " ".join(row["extensions"])
        boundaries = ", ".join(row["boundaries"])
        lines.append(f"| {row['language']} | {extensions} | {boundaries} |")
    return "\n".join(lines)


def readme_ast_cell() -> str:
    cells = []
    for row in _rows():
        extensions = " ".join(row["extensions"])
        cells.append(f"`{extensions}` ({row['language']})")
    return " · ".join(cells)


def render(text: str, name: str, body: str) -> str:
    begin = f"<!-- generated:{name}:begin -->"
    end = f"<!-- generated:{name}:end -->"
    start_index = text.find(begin)
    end_index = text.find(end)
    if start_index == -1 or end_index == -1:
        raise ValueError(f"marker {name!r} not found")
    start_index += len(begin)
    return text[:start_index] + "\n" + body + "\n" + text[end_index:]


def main(argv=None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    check = "--check" in args

    docs_text = DOCS_PATH.read_text()
    new_docs_text = render(docs_text, "language-extensions", extensions_table())
    new_docs_text = render(new_docs_text, "language-boundaries", boundaries_table())

    readme_text = README_PATH.read_text()
    readme_lines = readme_text.split("\n")
    cell = readme_ast_cell()
    for i, line in enumerate(readme_lines):
        if line.startswith("| AST-aware"):
            parts = line.split("|")
            parts[2] = f" {cell} "
            readme_lines[i] = "|".join(parts)
            break
    new_readme_text = "\n".join(readme_lines)

    changed = []
    if new_docs_text != docs_text:
        changed.append(DOCS_PATH)
    if new_readme_text != readme_text:
        changed.append(README_PATH)

    if check:
        for path in changed:
            print(path)
        if changed:
            sys.exit(1)
        return

    if new_docs_text != docs_text:
        DOCS_PATH.write_text(new_docs_text)
    if new_readme_text != readme_text:
        README_PATH.write_text(new_readme_text)


if __name__ == "__main__":
    main()
