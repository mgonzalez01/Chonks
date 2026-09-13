"""For .md/.markdown, segment_text_file splits at ATX heading boundaries and
names each chunk after its heading, instead of the nameless line-slicing the
rest of the text fallback uses."""
from chonks.chunking import CHUNK_MIN, CHUNK_TARGET, segment_text_file

# Pushes each section past CHUNK_MIN so _merge_small doesn't fold adjacent
# sections together (that's covered on its own below).
_PAD = "padding line to push this section past the merge-forward floor\n" * 8


def test_heading_split_basic():
    md = (
        f"# Title\n\n{_PAD}\n"
        f"## Sub One\n\n{_PAD}\n"
        f"## Sub Two\n\n{_PAD}"
    )
    segs = segment_text_file(md.encode(), path="doc.md")
    names = [s["name"] for s in segs]
    assert names == ["Title", "Sub One", "Sub Two"]
    for s in segs:
        assert s["chunk_type"] == "text"


def test_heading_section_spans_heading_line_through_next_heading():
    body_a = [f"line a{i} padded well past the merge-forward floor of bytes\n" for i in range(20)]
    body_b = [f"line b{i} padded well past the merge-forward floor of bytes\n" for i in range(20)]
    lines = ["# A\n"] + body_a + ["## B\n"] + body_b
    md = "".join(lines)
    segs = segment_text_file(md.encode(), path="doc.md")
    a = next(s for s in segs if s["name"] == "A")
    b = next(s for s in segs if s["name"] == "B")
    assert a["start_line"] == 1
    assert a["end_line"] == 21
    assert b["start_line"] == 22
    assert a["content"] == "".join(lines[:21])


def test_preamble_before_first_heading_is_nameless():
    md = f"Some intro text.\n{_PAD}\n# First Heading\n\n{_PAD}"
    segs = segment_text_file(md.encode(), path="doc.md")
    assert segs[0]["name"] is None
    assert segs[0]["chunk_type"] == "text"
    assert segs[0]["start_line"] == 1
    assert segs[1]["name"] == "First Heading"


def test_no_preamble_chunk_when_file_starts_with_heading():
    md = "# Title\n\nBody.\n"
    segs = segment_text_file(md.encode(), path="doc.md")
    # No leading nameless chunk: the first heading owns line 1.
    assert segs[0]["name"] == "Title"
    assert segs[0]["start_line"] == 1


def test_hash_inside_fenced_code_block_is_not_a_heading():
    md = (
        "# Real Heading\n\n"
        "```python\n"
        "# this is a comment, not a heading\n"
        "## neither is this\n"
        "```\n\n"
        "Some trailing text.\n"
    )
    segs = segment_text_file(md.encode(), path="doc.md")
    names = [s["name"] for s in segs]
    assert names == ["Real Heading"]
    assert "# this is a comment, not a heading" in segs[0]["content"]
    assert "## neither is this" in segs[0]["content"]


def test_tilde_fence_also_guards_headings():
    md = "# Heading\n\n~~~\n# not a heading\n~~~\n"
    segs = segment_text_file(md.encode(), path="doc.md")
    assert [s["name"] for s in segs] == ["Heading"]


def test_mismatched_fence_character_does_not_close_the_block():
    # CommonMark requires a closing fence to match the opening character, so a
    # ~~~ line does not close a ``` fence.
    md = (
        "# Real Heading\n\n"
        f"{_PAD}\n"
        "```\n"
        + "some code line\n" * 5
        + "~~~\n"
        "## sneaky heading not really\n\n"
        f"{_PAD}\n"
        "```\n\n"
        "trailing text\n"
    )
    segs = segment_text_file(md.encode(), path="doc.md")
    names = [s["name"] for s in segs]
    assert names == ["Real Heading"]
    assert "## sneaky heading not really" in segs[0]["content"]


def test_unclosed_fence_at_eof_swallows_rest_of_file_without_crashing():
    # A dangling fence with no closer: everything after it stays inside the
    # open section, even lines that look like headings.
    md = "# Heading\n\n```\n## looks like a heading but isn't\ntrailing code\n"
    segs = segment_text_file(md.encode(), path="doc.md")
    assert [s["name"] for s in segs] == ["Heading"]
    assert "## looks like a heading but isn't" in segs[0]["content"]
    assert "trailing code" in segs[0]["content"]


def test_heading_with_trailing_closing_hashes_is_stripped():
    md = "## Section Title ##\n\nBody.\n"
    segs = segment_text_file(md.encode(), path="doc.md")
    assert segs[0]["name"] == "Section Title"


def test_oversized_section_is_line_sliced_and_keeps_name():
    body = "filler line of section text\n" * 400  # comfortably over CHUNK_TARGET chars
    md = f"# Big Section\n\n{body}"
    segs = segment_text_file(md.encode(), path="doc.md")
    assert len(segs) > 1
    assert all(s["name"] == "Big Section" for s in segs)
    assert all(s["chunk_type"] == "text" for s in segs)
    assert segs[0]["start_line"] == 1
    assert segs[-1]["end_line"] == len(md.splitlines())
    for s in segs:
        assert len(s["content"].encode("utf-8")) <= CHUNK_TARGET * 4


def test_multiple_oversized_sections_each_keep_their_own_name():
    body = "filler line of section text\n" * 400
    md = f"# First Big\n\n{body}\n# Second Big\n\n{body}"
    segs = segment_text_file(md.encode(), path="doc.md")
    names = {s["name"] for s in segs}
    assert names == {"First Big", "Second Big"}
    assert sum(1 for s in segs if s["name"] == "First Big") > 1
    assert sum(1 for s in segs if s["name"] == "Second Big") > 1


def test_tiny_sections_merge_forward_below_chunk_min():
    # Sections under CHUNK_MIN merge via the same _merge_small trivia rule the
    # AST engine uses elsewhere.
    md = "# A\nx\n# B\ny\n# C\nz\n"
    assert len(md.encode("utf-8")) < CHUNK_MIN
    segs = segment_text_file(md.encode(), path="doc.md")
    assert len(segs) == 1
    assert segs[0]["name"] == "A"


def test_non_markdown_extension_unaffected_by_heading_logic():
    # Proves the heading path only fires for .md/.markdown extensions.
    text = "# not a heading, this is a shell comment\necho hi\n" * 5
    segs_txt = segment_text_file(text.encode(), path="script.sh")
    segs_none = segment_text_file(text.encode(), path=None)
    for segs in (segs_txt, segs_none):
        assert segs
        for s in segs:
            assert s["chunk_type"] == "text"
            assert s["name"] is None


def test_markdown_extension_case_insensitive_and_markdown_suffix():
    md = "# Title\n\nBody.\n"
    for p in ("README.MD", "doc.markdown", "doc.MARKDOWN"):
        segs = segment_text_file(md.encode(), path=p)
        assert segs[0]["name"] == "Title", p
