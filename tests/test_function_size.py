"""Keeps every function in chonks/ within a size budget of statements and lines."""
import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAX_STATEMENTS = 60
_MAX_LINES = 100

# Functions over a limit, with the (statements, lines) they may not grow past.
# See CONTRIBUTING.md.
_KNOWN_LARGE: dict[str, tuple[int, int]] = {
    "chonks/index/call_sites.py:_receiver_chain": (66, 68),
    "chonks/index/graph/knn.py:_build_neighbors_incremental": (71, 139),
    "chonks/index/graph/knn.py:build_neighbors": (66, 127),
    "chonks/index/graph/refs.py:_build_refs_incremental": (73, 118),
    "chonks/index/pipeline.py:embedder_worker": (37, 211),
    "chonks/index/pipeline.py:index_paths": (130, 351),
    "chonks/index/pipeline.py:parser_worker": (87, 147),
    "chonks/index/pipeline.py:scan_producer": (102, 211),
    "chonks/index/postindex.py:run_post_index_passes": (43, 103),
    "chonks/index/segment.py:segment_file": (111, 169),
    "chonks/ops/index_cmd.py:main": (143, 353),
    "chonks/ops/init.py:_run_interactive": (64, 94),
    "chonks/retrieval/graph_queries.py:find_usages": (62, 87),
    "chonks/retrieval/message_match.py:find_by_message": (139, 276),
    "chonks/retrieval/research.py:_graph_expand": (65, 113),
    "chonks/retrieval/research.py:deep_research": (84, 165),
    "chonks/serve/main.py:main": (67, 171),
}

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _statements(node: ast.AST) -> int:
    """Statements under node; a nested function or class counts as one."""
    count = 0
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt):
            count += 1
        if not isinstance(child, (*_FUNCTIONS, ast.ClassDef)):
            count += _statements(child)
    return count


def _functions(node: ast.AST, prefix: str = ""):
    """Yields (qualified name, node) for every function under node."""
    for child in ast.iter_child_nodes(node):
        scope = prefix
        if isinstance(child, (*_FUNCTIONS, ast.ClassDef)):
            scope = f"{prefix}{child.name}."
            if isinstance(child, _FUNCTIONS):
                yield f"{prefix}{child.name}", child
        yield from _functions(child, scope)


_SIZES = {
    f"{path.relative_to(_REPO_ROOT).as_posix()}:{name}": (_statements(fn), fn.end_lineno - fn.lineno + 1)
    for path in sorted((_REPO_ROOT / "chonks").rglob("*.py"))
    for name, fn in _functions(ast.parse(path.read_text(encoding="utf-8")))
}


def _over_a_limit(size: tuple[int, int]) -> bool:
    return size[0] > _MAX_STATEMENTS or size[1] > _MAX_LINES


def test_functions_stay_within_the_size_limits():
    offenders = [f"{name} {size}" for name, size in _SIZES.items()
                 if _over_a_limit(size) and name not in _KNOWN_LARGE]
    assert offenders == [], (
        f"over {_MAX_STATEMENTS} statements or {_MAX_LINES} lines; split it; see CONTRIBUTING.md")


@pytest.mark.parametrize("name", sorted(_KNOWN_LARGE))
def test_known_large_function_does_not_grow(name):
    statements, lines = _SIZES.get(name, (0, 0))
    max_statements, max_lines = _KNOWN_LARGE[name]
    assert statements <= max_statements and lines <= max_lines, (
        f"{name} grew to ({statements}, {lines}); split it; see CONTRIBUTING.md")


@pytest.mark.parametrize("name", sorted(_KNOWN_LARGE))
def test_known_large_function_is_still_over_a_limit(name):
    size = _SIZES.get(name, (0, 0))
    assert _over_a_limit(size), (
        f"{name} is gone or within the limits; drop it from _KNOWN_LARGE")
