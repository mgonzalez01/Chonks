"""A nested repo copy or venv under a dot-directory (e.g.
.claude/worktrees/<hash>/) gets silently indexed today since no excludes ship
as hardcoded defaults."""
import logging
from pathlib import Path

from chonks.core.paths import _dotdir_prefix
from chonks.index.pipeline import index_paths
from chonks.ops.diagnostics import dotdir_breakdown, dotdir_total_share
from chonks.storage.store import Store


# --------------------------------------------------------------------------
# chunking.dotdir_breakdown / dotdir_total_share
# --------------------------------------------------------------------------

def test_dotdir_breakdown_ignores_non_dot_paths():
    rows = [("src/a.py", "python"), ("docs/x.md", "md")]
    assert dotdir_breakdown(rows) == []


def test_dotdir_breakdown_groups_two_segments_deep():
    rows = [
        (".claude/worktrees/agent-a/foo.py", "python"),
        (".claude/worktrees/agent-a/bar.py", "python"),
        (".claude/worktrees/agent-b/foo.py", "python"),
        (".claude/skills/deploy.md", "md"),
    ]
    stats = dotdir_breakdown(rows)
    by_prefix = {s.prefix: s for s in stats}
    assert by_prefix[".claude/worktrees"].chunks == 3
    assert by_prefix[".claude/worktrees"].files == 3
    assert by_prefix[".claude/skills"].chunks == 1


def test_dotdir_breakdown_single_segment_dotdir():
    rows = [(".venv/lib.py", "python")] * 4
    stats = dotdir_breakdown(rows)
    assert stats[0].prefix == ".venv"
    assert stats[0].chunks == 4


def test_dotdir_breakdown_excludes_git_metadata():
    rows = [(".git/hooks/pre-commit", None)] * 3
    assert dotdir_breakdown(rows) == []


def test_dotdir_breakdown_sorted_desc_by_chunks_then_prefix():
    rows = [(".b/1.py", "python")] * 2 + [(".a/1.py", "python")] * 5
    stats = dotdir_breakdown(rows)
    assert [s.prefix for s in stats] == [".a", ".b"]


def test_dotdir_breakdown_ties_sorted_alphabetically():
    rows = [(".b/1.py", "python")] * 3 + [(".a/1.py", "python")] * 3
    stats = dotdir_breakdown(rows)
    assert [s.prefix for s in stats] == [".a", ".b"]


def test_dotdir_total_share():
    rows = [(".claude/x.py", "python")] * 2 + [("src/a.py", "python")] * 8
    stats = dotdir_breakdown(rows)
    assert dotdir_total_share(stats, 10) == 0.2


def test_dotdir_total_share_empty_corpus():
    assert dotdir_total_share([], 0) == 0.0


def test_dotdir_total_share_no_dotdir_content():
    stats = dotdir_breakdown([("src/a.py", "python")])
    assert dotdir_total_share(stats, 1) == 0.0


# --------------------------------------------------------------------------
# chunker._dotdir_prefix
# --------------------------------------------------------------------------

def test_dotdir_prefix_none_for_ordinary_path():
    assert _dotdir_prefix("src/engine.py") is None


def test_dotdir_prefix_none_for_root_file():
    assert _dotdir_prefix("README.md") is None


def test_dotdir_prefix_none_for_git_metadata():
    assert _dotdir_prefix(".git/config") is None
    assert _dotdir_prefix(".git") is None


def test_dotdir_prefix_two_segments_deep():
    assert _dotdir_prefix(".claude/worktrees/agent-a/chonks") == ".claude/worktrees"


def test_dotdir_prefix_single_segment_when_dotdir_has_no_children():
    assert _dotdir_prefix(".venv") == ".venv"


# --------------------------------------------------------------------------
# chunker.index_paths: end-to-end nested-.git-under-dotdir warning
# --------------------------------------------------------------------------

_DIM = 4


class _FakeEmbedder:
    model = "fake"
    url = "http://localhost:9999"

    def embed_documents(self, texts, client=None, **kw):
        return [[0.1] * _DIM for _ in texts]

    def embed_queries(self, texts, client=None, **kw):
        return [[0.1] * _DIM for _ in texts]


def test_index_warns_on_nested_git_under_dotdir(tmp_path, caplog):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main():\n    pass\n")

    nested = root / ".claude" / "worktrees" / "agent-x" / "chonks"
    nested.mkdir(parents=True)
    (nested / ".git").mkdir()
    (nested / "a.py").write_text("def a():\n    pass\n")
    (nested / "b.py").write_text("def b():\n    pass\n")

    store = Store(tmp_path / "test.db")
    try:
        with caplog.at_level(logging.WARNING, logger="chonks.chunker"):
            index_paths([root], store, _FakeEmbedder(), root=root)
    finally:
        store.close()

    warnings = [r.message for r in caplog.records if "nested repo copy" in r.message]
    assert len(warnings) == 1
    assert ".claude/worktrees" in warnings[0]
    assert "2 files" in warnings[0]
    assert '"exclude": [".claude/worktrees"]' in warnings[0]


def test_index_no_warning_without_nested_git(tmp_path, caplog):
    root = tmp_path / "repo"
    root.mkdir()
    dotdir = root / ".claude" / "skills"
    dotdir.mkdir(parents=True)
    (dotdir / "notes.py").write_text("x = 1\n")

    store = Store(tmp_path / "test.db")
    try:
        with caplog.at_level(logging.WARNING, logger="chonks.chunker"):
            index_paths([root], store, _FakeEmbedder(), root=root)
    finally:
        store.close()

    assert not [r.message for r in caplog.records if "nested repo copy" in r.message]


def test_index_still_indexes_dotdir_content_by_default(tmp_path):
    # Visibility fix, not a filtering one: still indexed unless the caller
    # configures an exclude (warn, not skip).
    root = tmp_path / "repo"
    root.mkdir()
    nested = root / ".claude" / "worktrees" / "agent-x"
    nested.mkdir(parents=True)
    (nested / ".git").mkdir()
    (nested / "a.py").write_text("def a():\n    pass\n")

    store = Store(tmp_path / "test.db")
    try:
        index_paths([root], store, _FakeEmbedder(), root=root)
        assert store.get_file_hash(".claude/worktrees/agent-x/a.py") is not None
    finally:
        store.close()
