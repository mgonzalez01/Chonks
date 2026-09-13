"""Regression context: a committed docs/ mirror (sqlglot's pdoc output) was
84% of the index and poisoned retrieval, with nothing surfacing it."""
from chonks.chunking import dominance_warning, family_breakdown, path_family


def test_path_family_top_level_dir():
    assert path_family("docs/search.js") == "docs"
    assert path_family("docs/sub/dir/file.py") == "docs"


def test_path_family_root_file():
    assert path_family("README.md") == "(root)"


def test_family_breakdown_groups_and_counts():
    rows = [
        ("src/a.py", "python"),
        ("src/b.py", "python"),
        ("src/sub/c.py", "python"),
        ("docs/x.html", "html"),
        ("README.md", "md"),
    ]
    stats = family_breakdown(rows)
    by_name = {s.family: s for s in stats}
    assert by_name["src"].chunks == 3
    assert by_name["src"].files == 3
    assert by_name["src"].docs_chunks == 0
    assert by_name["docs"].chunks == 1
    assert by_name["docs"].docs_chunks == 1
    assert by_name["(root)"].chunks == 1
    assert by_name["(root)"].docs_chunks == 1


def test_family_breakdown_sorted_desc_by_chunks():
    rows = [("a/1.py", "python")] * 2 + [("b/1.py", "python")] * 5
    stats = family_breakdown(rows)
    assert [s.family for s in stats] == ["b", "a"]


def test_family_breakdown_ties_sorted_alphabetically():
    rows = [("b/1.py", "python")] * 3 + [("a/1.py", "python")] * 3
    stats = family_breakdown(rows)
    assert [s.family for s in stats] == ["a", "b"]


def test_family_breakdown_top_file_within_family():
    rows = [("docs/big.html", "html")] * 10 + [("docs/small.html", "html")] * 2
    stats = family_breakdown(rows)
    docs = stats[0]
    assert docs.top_file == "docs/big.html"
    assert docs.top_file_chunks == 10


def test_family_breakdown_empty_rows():
    assert family_breakdown([]) == []


def test_family_breakdown_null_language_counts_as_docs():
    # language=None (text-fallback chunks) must count as docs-kind, same
    # contract as store._chunk_kind_clause.
    rows = [("notes/plan.txt", None)] * 3
    stats = family_breakdown(rows)
    assert stats[0].docs_chunks == 3


def test_dominance_warning_fires_on_docs_dominated_family():
    rows = [("docs/mirror.html", "html")] * 84 + [("src/a.py", "python")] * 16
    stats = family_breakdown(rows)
    warning = dominance_warning(stats, sum(s.chunks for s in stats))
    assert warning is not None
    assert "docs/" in warning
    assert "84%" in warning
    assert "docs/mirror.html" in warning
    assert '"exclude": ["docs/"]' in warning


def test_dominance_warning_silent_on_healthy_src_dominant_corpus():
    rows = [("src/a.py", "python")] * 90 + [("docs/x.md", "md")] * 10
    stats = family_breakdown(rows)
    warning = dominance_warning(stats, sum(s.chunks for s in stats))
    assert warning is None


def test_dominance_warning_family_share_alone_is_not_enough():
    # A 90%-share family that's mostly code fails the docs_share leg regardless of size.
    rows = [("src/a.py", "python")] * 90 + [("src/README.md", "md")] * 5 + [("docs/x.md", "md")] * 5
    stats = family_breakdown(rows)
    warning = dominance_warning(stats, sum(s.chunks for s in stats))
    assert warning is None


def test_dominance_warning_docs_share_alone_is_not_enough():
    # 100% docs but too small a slice of the corpus to matter.
    rows = [("docs/x.md", "md")] * 5 + [("src/a.py", "python")] * 95
    stats = family_breakdown(rows)
    warning = dominance_warning(stats, sum(s.chunks for s in stats))
    assert warning is None


def test_dominance_warning_fires_on_corpus_wide_docs_share():
    # Each family is under the 40%-family-share trigger, but the corpus as a
    # whole leans docs-kind (>= 50%).
    rows = []
    for i in range(10):
        rows += [(f"fam{i}/x.md", "md")] * 6  # 60 docs chunks total, 6% each
    rows += [("src/a.py", "python")] * 40  # 40 code chunks (40%)
    stats = family_breakdown(rows)
    total = sum(s.chunks for s in stats)
    warning = dominance_warning(stats, total)
    assert warning is not None
    assert "docs-kind chunks are" in warning
    assert "60%" in warning


def test_dominance_warning_none_on_empty_corpus():
    assert dominance_warning([], 0) is None
