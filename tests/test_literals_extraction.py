"""Literal & message index extraction tests (chunking.py side).

Covers per-language string literal extraction: interpolation holes,
adjacent-string concatenation, escape decoding, and the per-chunk cap."""

from chonks.chunking import segment_file


def _literals_by_name(segs: list[dict]) -> dict[str, list[tuple[str, int]]]:
    return {s["name"]: s["literals"] for s in segs if s["literals"]}


# ---------------------------------------------------------------------------
# Per-language extraction
# ---------------------------------------------------------------------------

def test_python_fstring_holes_survive_as_literal_markup():
    """f-string interpolation expressions ({name}, {code:d}) are kept
    verbatim in the extracted text (they become skeleton holes downstream,
    in store.py); the literal fragments around them survive untouched."""
    src = b'''
def load(name, code):
    logger.error(f"failed to load {name}: {code:d}")
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits["load"] == [("failed to load {name}: {code:d}", 3)]


def test_python_implicit_concat_joins_adjacent_pieces():
    """Two adjacent string literals in the same expression (python implicit
    concatenation) join into ONE record."""
    src = b'''
def f():
    msg = "hello there, " "friend of mine"
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits["f"] == [("hello there, friend of mine", 3)]


def test_python_pure_plus_chain_joins_but_impure_chain_does_not():
    """A '+' chain of only string literals joins into one record; a chain
    with a non-literal operand leaves the pure literal pieces standalone.
    Each function is padded past CHUNK_MIN so they don't get merged into
    one chunk."""
    filler = "\n".join(f"    _ = {i}  # padding line {i} to clear CHUNK_MIN" for i in range(20))
    src = (
        "def pure():\n"
        '    msg = "hello " + "there " + "friend"\n'
        f"{filler}\n"
        "\n"
        "def impure(code):\n"
        '    msg = "error code: " + str(code)\n'
        f"{filler}\n"
    ).encode()
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits["pure"] == [("hello there friend", 2)]
    assert lits["impure"] == [("error code: ", 25)]


def test_cpp_adjacent_string_literals_join():
    """C/C++ juxtaposed string literals ("a" "b") are one token-pair the
    grammar itself groups as concatenated_string; must join into one record."""
    src = b'''
void load() {
    const char* msg = "failed to load asset" " from disk";
    log(msg);
}
'''
    segs = segment_file(src, "cpp", path="t.cpp")
    lits = _literals_by_name(segs)
    assert lits["load"] == [("failed to load asset from disk", 3)]


def test_cpp_prefixed_wide_string_strips_prefix_and_quotes():
    src = b'''
void f() {
    const wchar_t* s = L"wide string literal value";
}
'''
    segs = segment_file(src, "cpp", path="t.cpp")
    lits = _literals_by_name(segs)
    assert lits["f"] == [("wide string literal value", 3)]


def test_cpp_raw_string_literal_stripped_and_never_decoded():
    """R"(...)": the wrapper is stripped (grammar's own raw_string_content
    child, see chunking.py's _leaf_literal_text) but the content is never
    escape-decoded: a literal backslash-n here is two real characters, not
    a newline."""
    src = b'''
void f() {
    const char* s = R"(could not open archive %s)";
}
'''
    segs = segment_file(src, "cpp", path="t.cpp")
    lits = _literals_by_name(segs)
    assert lits["f"] == [("could not open archive %s", 3)]


def test_cpp_raw_string_literal_custom_delimiter():
    """R"delim(...)delim": a custom delimiter (needed when the content
    itself contains a bare `)"`) is stripped the same way."""
    src = b'''
void f() {
    const char* s = R"delim(text (with) parens and a )" inside)delim";
}
'''
    segs = segment_file(src, "cpp", path="t.cpp")
    lits = _literals_by_name(segs)
    assert lits["f"] == [('text (with) parens and a )" inside', 3)]


def test_c_sharp_interpolated_string_holes_survive():
    src = b'''
class X {
    void Load(string name, int code) {
        var s = $"failed to load {name}: {code:D4}";
    }
}
'''
    segs = segment_file(src, "c_sharp", path="t.cs")
    lits = _literals_by_name(segs)
    assert lits["X"] == [("failed to load {name}: {code:D4}", 4)]


def test_c_sharp_verbatim_string_strips_at_prefix():
    src = b'''
class X {
    void Load() {
        var r = @"raw path value here";
    }
}
'''
    segs = segment_file(src, "c_sharp", path="t.cs")
    lits = _literals_by_name(segs)
    assert lits["X"] == [("raw path value here", 4)]


def test_gdscript_percent_format_literal_kept_verbatim():
    """GDScript's % formatting operator doesn't produce an interpolation AST
    node; the %s/%d markup is already plain string content, so no special
    hole handling is needed to preserve it."""
    src = b'''
func load(name):
    var s = "failed to load %s: %d" % [name, 0]
'''
    segs = segment_file(src, "gdscript", path="t.gd")
    lits = _literals_by_name(segs)
    assert lits["load"] == [("failed to load %s: %d", 3)]


def test_javascript_template_literal_holes_survive():
    src = b'''
function load(name, code) {
    const s = `failed to load ${name}: ${code}`;
}
'''
    segs = segment_file(src, "javascript", path="t.js")
    lits = _literals_by_name(segs)
    assert lits["load"] == [("failed to load ${name}: ${code}", 3)]


def test_typescript_template_literal_holes_survive():
    src = b'''
function load(name: string, code: number): void {
    const s = `failed to load ${name}: ${code}`;
}
'''
    segs = segment_file(src, "typescript", path="t.ts")
    lits = _literals_by_name(segs)
    assert lits["load"] == [("failed to load ${name}: ${code}", 3)]


def test_comment_text_never_becomes_a_literal():
    """AST string-node extraction never regex-scans raw content: a comment
    that merely looks like a string literal must never surface as one."""
    src = b'''
def f():
    # msg = "this lives in a comment, not code"
    return 1
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits == {}


def test_hlsl_unsupported_language_yields_no_literals():
    """A language with no _LITERAL_SPECS entry falls back to empty literals,
    same shape as the calls/imports/inherits fallback."""
    src = b'''
struct VSOut { float4 pos : SV_Position; };
VSOut main(float3 p : POSITION) { VSOut o; o.pos = float4(p, 1); return o; }
'''
    segs = segment_file(src, "hlsl", path="t.hlsl")
    for seg in segs:
        assert seg["literals"] == []


def test_lua_string_literal_and_concat_join():
    """lua's concatenation operator is `..`, not `+`; an impure chain
    (literal .. identifier) leaves the pure literal piece standalone, same
    semantics as the other languages' impure '+' chains."""
    src = b'''
function load(name)
  print("failed to load "..name)
end
'''
    segs = segment_file(src, "lua", path="t.lua")
    lits = _literals_by_name(segs)
    assert lits["load"] == [("failed to load ", 3)]


def test_lua_pure_concat_chain_joins():
    src = b'''
function greet()
  local msg = "hello there, "..'friend of mine'
end
'''
    segs = segment_file(src, "lua", path="t.lua")
    lits = _literals_by_name(segs)
    assert lits["greet"] == [("hello there, friend of mine", 3)]


def test_python_docstrings_are_not_extracted_as_literals():
    """Module/function/class docstrings are AST string nodes but are prose,
    not runtime-emitted messages, so they must not surface as
    find_by_message-searchable literals."""
    src = b'''"""module docstring long enough to pass the floor"""

def f():
    """function docstring long enough to pass the floor"""
    print("hello world runtime message")


class C:
    """class docstring long enough to pass the floor"""
    pass
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    all_texts = [text for texts in lits.values() for text, _line in texts]
    assert all_texts == ["hello world runtime message"]


def test_python_non_docstring_bare_string_statement_still_skipped_by_position_only():
    """The docstring-position filter is positional (first statement of a
    module/function/class body), not any bare string-expression statement:
    a bare string elsewhere is still extracted."""
    src = b'''
def f():
    x = 1
    "not a docstring, not first statement, still runtime-visible text"
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits["f"] == [("not a docstring, not first statement, still runtime-visible text", 4)]


# ---------------------------------------------------------------------------
# Trivial-skip filter
# ---------------------------------------------------------------------------

def test_trivial_literals_skipped_short_and_no_alpha():
    """len(text) < 6 after stripping, or no alphabetic character, are both
    dropped as noise (punctuation-only separators, single letters, etc.)."""
    src = b'''
def f():
    a = "hi"          # len 2, dropped
    b = "12345"       # len 5, dropped
    c = "------"      # len 6 but no alpha, dropped
    d = "123456"      # len 6, no alpha, dropped
    e = "hello!"      # len 6, has alpha, KEPT
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits["f"] == [("hello!", 7)]


# ---------------------------------------------------------------------------
# Per-chunk cap
# ---------------------------------------------------------------------------

def test_per_chunk_literal_cap_at_200():
    """A single chunk with more than 200 qualifying literals is capped at
    200: the cap is a hard ceiling on the stored record, not merely a
    display truncation."""
    lines = [f'    x{i} = "literal number {i:04d} value"' for i in range(210)]
    src = ("def f():\n" + "\n".join(lines) + "\n").encode()
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert len(lits["f"]) == 200


def test_per_chunk_literal_cap_announces_drop_in_counters():
    """The cap is a stats-visible cap: dropped literals are counted in the
    run-summary counters dict, like other caps (oversize chunks, etc.),
    not merely dropped with no trace."""
    lines = [f'    x{i} = "literal number {i:04d} value"' for i in range(210)]
    src = ("def f():\n" + "\n".join(lines) + "\n").encode()
    counters: dict = {}
    segment_file(src, "python", path="t.py", counters=counters)
    assert counters["literals_capped_chunks"] == 1
    assert counters["literals_dropped"] == 10  # 210 - 200


def test_no_cap_drop_counters_when_under_cap():
    """A file with a handful of literals must not report a phantom cap event."""
    src = b'''
def f():
    x = "just one literal here"
'''
    counters: dict = {}
    segment_file(src, "python", path="t.py", counters=counters)
    assert "literals_capped_chunks" not in counters
    assert "literals_dropped" not in counters


# ---------------------------------------------------------------------------
# Escape-sequence decoding
# ---------------------------------------------------------------------------

def test_cpp_trailing_newline_escape_decoded_and_stripped():
    """A C literal ending in a decoded \\n must resolve to text a pasted
    single-line log message would actually contain: the trailing newline
    is stripped, not left as two literal source characters `\\` `n`."""
    src = b'''
void f(const char *path) {
    fprintf(stderr, "failed to open texture %s\\n", path);
}
'''
    segs = segment_file(src, "cpp", path="t.cpp")
    lits = _literals_by_name(segs)
    assert lits["f"] == [("failed to open texture %s", 3)]


def test_cpp_embedded_escaped_quote_decoded():
    """An embedded \\" must decode to a real quote character so the stored
    text matches a pasted concrete message, not the raw three-character
    source `\\` `"` `%`."""
    src = b'''
void f(const char *name) {
    fprintf(stderr, "user \\"%s\\" not found in registry", name);
}
'''
    segs = segment_file(src, "cpp", path="t.cpp")
    lits = _literals_by_name(segs)
    assert lits["f"] == [('user "%s" not found in registry', 3)]


def test_python_raw_string_not_decoded():
    """python r"..." raw strings must not have escapes decoded: the
    backslash-n stays two literal characters, exactly as tools relying on a
    raw string (regexes, Windows paths) expect."""
    src = br'''
def f():
    x = r"raw literal with backslash-n: \n stays literal"
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits["f"] == [(r"raw literal with backslash-n: \n stays literal", 3)]


def test_python_raw_bytes_string_not_decoded():
    """python rb"..." (raw bytes) is also raw: the 'r' in either prefix
    order (rb/br) must suppress decoding."""
    src = br'''
def f():
    x = rb"raw bytes with backslash-n: \n stays literal"
    y = br"same but prefix order swapped: \t also stays literal"
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits["f"] == [
        (r"raw bytes with backslash-n: \n stays literal", 3),
        (r"same but prefix order swapped: \t also stays literal", 4),
    ]


def test_python_normal_string_escapes_decoded():
    """A non-raw python string DOES decode standard escapes."""
    src = br'''
def f():
    x = "line one\nline two decoded properly"
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits["f"] == [("line one\nline two decoded properly", 3)]


def test_csharp_verbatim_string_only_decodes_doubled_quote():
    """C# @"..." verbatim strings decode only "" -> ": a literal
    backslash-n stays two characters (this is exactly what @-strings are
    for: unescaped Windows paths)."""
    src = br'''
class C {
    void F() {
        string a = @"path\to\file with backslash-n: \n stays literal and ""quoted"" becomes quoted";
    }
}
'''
    segs = segment_file(src, "c_sharp", path="t.cs")
    lits = _literals_by_name(segs)
    assert lits["C"] == [
        (r'path\to\file with backslash-n: \n stays literal and "quoted" becomes quoted', 4),
    ]


def test_csharp_regular_string_escapes_decoded():
    """A normal (non-@) C# string decodes standard escapes same as C/python."""
    src = br'''
class C {
    void F() {
        string a = "line one\nline two decoded";
    }
}
'''
    segs = segment_file(src, "c_sharp", path="t.cs")
    lits = _literals_by_name(segs)
    assert lits["C"] == [("line one\nline two decoded", 4)]


def test_lua_long_bracket_string_not_decoded():
    """lua [[...]] long-bracket strings are raw: no escape processing at
    all, matching lua's own language semantics for that literal kind."""
    src = b'''
function load(name)
    print([[raw bracket text with backslash-n: \\n stays literal]])
end
'''
    segs = segment_file(src, "lua", path="t.lua")
    lits = _literals_by_name(segs)
    assert lits["load"] == [(r"raw bracket text with backslash-n: \n stays literal", 3)]


def test_lua_regular_string_escapes_decoded():
    """An ordinary lua 'quoted'/"quoted" string DOES decode escapes."""
    src = b'''
function load(name)
    print("line one\\nline two decoded")
end
'''
    segs = segment_file(src, "lua", path="t.lua")
    lits = _literals_by_name(segs)
    assert lits["load"] == [("line one\nline two decoded", 3)]


def test_interior_newline_kept_only_trailing_one_stripped():
    """A decoded literal with an interior newline stays multi-line; only a
    trailing decoded newline/carriage-return is stripped (store.py's
    matcher separately tries the first line alone for messages like this)."""
    src = br'''
def f():
    x = "first line here\nsecond line here\n"
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits["f"] == [("first line here\nsecond line here", 3)]


def test_unknown_escape_passed_through_unchanged():
    """An escape this decoder doesn't recognize (\\d, \\p, ...) is left
    exactly as written: never guessed at, never dropped, never an error."""
    src = br'''
def f():
    x = "unrecognized escape stays as-is: \d and \p both untouched here"
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits["f"] == [
        (r"unrecognized escape stays as-is: \d and \p both untouched here", 3)
    ]


def test_hex_escape_decoded():
    """\\xNN (exactly two hex digits) decodes to the corresponding byte."""
    src = br'''
def f():
    x = "hex escape decodes: \x41\x42\x43 becomes ABC here"
'''
    segs = segment_file(src, "python", path="t.py")
    lits = _literals_by_name(segs)
    assert lits["f"] == [("hex escape decodes: ABC becomes ABC here", 3)]
