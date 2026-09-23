"""The definition table: what a project's own #defines and types say about a
name, and how macro self-heal uses it."""
import pytest

from chonks.index.macro_defs import ATTRIBUTE, CODE, TYPE, UNKNOWN, VETO, WRAPPER, scan_definitions
from chonks.index.macro_heal import _blank_attributes, _blank_macros, _unwrap_macros
from chonks.index.segment import segment_file


def _table(tmp_path, *sources):
    paths = []
    for i, text in enumerate(sources):
        p = tmp_path / f"f{i}.h"
        p.write_text(text)
        paths.append(p)
    return scan_definitions(paths)


@pytest.mark.parametrize("source,name,cls", [
    ("#define _FORCE_INLINE_ __attribute__((always_inline)) inline\n", "_FORCE_INLINE_", ATTRIBUTE),
    ("#define EXPORT\n", "EXPORT", ATTRIBUTE),
    ("#define BOOLISH int\n", "BOOLISH", TYPE),
    ("#define LOCK_METHOD MutexLock _lock_(_mutex_)\n", "LOCK_METHOD", CODE),
    ("#define LIMIT (1 << 4)\n", "LIMIT", CODE),
    ("#define DECL(type, name) type name\n", "DECL", VETO),
    ("#define CHECK(x) if (!(x)) return\n", "CHECK", UNKNOWN),
    ("#define CALLCONV WINAPI\n", "CALLCONV", UNKNOWN),
    ("#define LOCAL(type) static type\n", "LOCAL", WRAPPER),
    ("#define OF(args) args\n", "OF", WRAPPER),
    ("#define TWICE(x) x + x\n", "TWICE", UNKNOWN),
    ("class RID {\n};\n", "RID", VETO),
    ("typedef unsigned int JDIMENSION;\n", "JDIMENSION", VETO),
], ids=["attribute", "empty", "type", "statement", "expression", "declaration-writer",
        "call-shaped", "external", "wrapper", "argument-list-wrapper", "parameter-used-twice",
        "class", "typedef"])
def test_classification(tmp_path, source, name, cls):
    assert _table(tmp_path, source).classes[name] == cls


def test_conflicting_definitions_take_the_most_cautious_reading(tmp_path):
    table = _table(tmp_path,
                   "#ifdef WIN\n#define API __declspec(dllexport)\n#else\n#define API\n#endif\n"
                   "#define MIXED\n#define MIXED int\n"
                   "#define BYTE xxh_u8\n",
                   "typedef unsigned char xxh_u8;\nstruct BYTE { int x; };\n")
    assert table.classes["API"] == ATTRIBUTE
    assert table.classes["MIXED"] == TYPE
    assert table.classes["BYTE"] == VETO, "a name also defined as a type is a type"


def test_chains_resolve_and_cycles_stop(tmp_path):
    table = _table(tmp_path,
                   "#define OUTER MIDDLE\n#define MIDDLE INNER\n#define INNER inline\n"
                   "#define LOOP_A LOOP_B\n#define LOOP_B LOOP_A\n")
    assert table.classes["OUTER"] == ATTRIBUTE
    assert table.classes["LOOP_A"] == UNKNOWN


def test_type_macro_is_substituted_in_place(tmp_path):
    table = _table(tmp_path, "#define BOOLISH int\n")
    assert table.substitutions == {"BOOLISH": "int"}
    src = b"static BOOLISH\nflag(void);\n"
    out = _blank_macros(src, {"BOOLISH"}, table.substitutions)
    assert len(out) == len(src) and out.count(b"\n") == src.count(b"\n")
    assert out.startswith(b"static int    ")


def test_type_macro_longer_than_its_name_is_never_blanked(tmp_path):
    table = _table(tmp_path, "#define U8 unsigned char\n")
    assert table.classes["U8"] == TYPE
    assert "U8" not in table.substitutions and "U8" in table.vetoes


def test_a_type_the_project_defines_is_never_hidden(tmp_path):
    table = _table(tmp_path, "class RID {\n  int id;\n};\n")
    src = (b"class Buffers {\n"
           b"  _FORCE_INLINE_ RID get_render_target() const { return target; }\n"
           b"};\n")
    counters: dict = {}
    segment_file(src, "cpp", counters=counters, definitions=table)
    assert "RID" not in counters.get("discovered_macros", set())


def test_declaration_writing_macro_is_left_alone(tmp_path):
    table = _table(tmp_path, "#define PNG_FUNCTION(type, name, args, attributes) attributes type name args\n")
    src = b"PNG_FUNCTION(void, png_free, (void *p), PNG_EMPTY)\n{\n  release(p);\n}\n"
    counters: dict = {}
    segment_file(src, "c", counters=counters, definitions=table)
    assert "PNG_FUNCTION" not in counters.get("discovered_macros", set())


def test_macro_that_expands_to_code_can_still_be_healed(tmp_path):
    # extern "C" { ... } spelled as a pair of macros: blanking both is what
    # repairs the file, so the table must not stop it.
    table = _table(tmp_path, '#define BEGIN_C extern "C" {\n#define END_C }\n')
    assert table.classes["BEGIN_C"] == CODE and "BEGIN_C" not in table.vetoes
    src = b"BEGIN_C\nint area(int w, int h);\nint volume(int w, int h, int d);\nEND_C\n"
    counters: dict = {}
    segment_file(src, "cpp", counters=counters, definitions=table)
    assert {"BEGIN_C", "END_C"} <= counters.get("discovered_macros", set())


def test_nested_wrapper_and_extern_c(tmp_path):
    table = _table(tmp_path, '#define API_ATTR __attribute__((visibility("default")))\n'
                             '#define EXPORT(t) extern "C" API_ATTR t\n'
                             '#define EXPORT_DEF(t) EXPORT(t)\n')
    assert table.wrappers == {"EXPORT", "EXPORT_DEF"}
    assert table.wrappers <= table.vetoes, "blanking a wrapper with its argument deletes the type"


def test_unwrap_keeps_the_argument_and_the_layout():
    src = b"#if LOCAL(x)\n#endif\nLOCAL(unsigned int)\nf(void) { return 0; }\n"
    out = _unwrap_macros(src, {"LOCAL"})
    assert len(out) == len(src) and out.count(b"\n") == src.count(b"\n")
    assert out.startswith(b"#if LOCAL(x)\n"), "directive lines are the preprocessor's"
    assert b"      unsigned int \nf(void)" in out
    continued = b"#if defined(A) && \\\n    LOCAL(x)\n#endif\n"
    assert _unwrap_macros(continued, {"LOCAL"}) == continued, "continuation lines belong to the directive"


def test_blanking_leaves_directive_lines_alone():
    # Blanking the name in its own multi-line #define leaves a nameless
    # define whose body leaks out as code.
    src = (b"#define CHECK(n) \\\n  do { if (n) return; } \\\n  while (0)\n"
           b"#if defined(CHECK)\n#endif\nvoid f(int n) { CHECK(n); }\n")
    out = _blank_macros(src, {"CHECK"})
    assert out.startswith(src[:src.index(b"void")])
    assert b"void f(int n) {         ; }" in out


def test_wrapper_saved_as_a_macro_no_longer_hides_functions(tmp_path):
    # libjpeg's shape. With LOCAL saved as a macro to blank, the return type
    # goes with it and both functions vanish; the table unwraps it instead.
    table = _table(tmp_path, "#define LOCAL(type) static type\n#define INLINE inline\n")
    src = (b"INLINE\nLOCAL(void)\ninit_pass(j_ptr cinfo, int row,\n          int col)\n{\n  cinfo->pass = row + col;\n}\n\n"
           b"INLINE\nLOCAL(int)\nread_row(j_ptr cinfo, int row,\n         int col)\n{\n  return cinfo->row[row] + col;\n}\n")
    saved, with_table = {}, {}
    segment_file(src, "c", counters=saved, macros={"LOCAL", "INLINE"})
    segment_file(src, "c", counters=with_table, macros={"INLINE"}, definitions=table)
    names = lambda c: {s["name"] for s in c["symbols"]}
    assert not {"init_pass", "read_row"} & names(saved)
    assert {"init_pass", "read_row"} <= names(with_table)


def test_attribute_macros_are_hidden_before_parsing(tmp_path):
    table = _table(tmp_path, "#ifdef _WIN32\n#define API __declspec(dllexport)\n#else\n#define API\n#endif\n"
                             "#define CALLCONV\n#define local static\n")
    assert table.attributes == {"API", "CALLCONV"}, "lowercase names may be identifiers elsewhere"
    src = b"API int CALLCONV area(int w, int h) { return w * h; }\nint (CALLCONV *hook)(int);\n"
    counters: dict = {}
    segment_file(src, "c", counters=counters, definitions=table)
    assert "area" in {s["name"] for s in counters["symbols"]}
    assert not counters.get("parse_error") and not counters.get("discovered_macros")


def test_attribute_macro_keeps_the_parenthesis_after_it():
    out = _blank_attributes(b"int (CALLCONV *hook)(int);\n#if defined(CALLCONV)\n#endif\n", {"CALLCONV"})
    assert out == b"int (         *hook)(int);\n#if defined(CALLCONV)\n#endif\n"


def test_attribute_macro_inside_another_macros_arguments_is_left_alone():
    src = b"PNG_FUNCTION(void *, zalloc, (int n), PNG_ALLOCATED)\n{ return 0; }\nPNG_ALLOCATED void *f(void);\n"
    out = _blank_attributes(src, {"PNG_ALLOCATED"}, {"PNG_FUNCTION"})
    assert out.startswith(b"PNG_FUNCTION(void *, zalloc, (int n), PNG_ALLOCATED)\n")
    assert out.endswith(b"              void *f(void);\n")


def test_a_type_before_parentheses_is_not_a_macro_call():
    src = b"typedef HRESULT(WINAPI *set_description)(HANDLE, PCWSTR);\n"
    out = _blank_attributes(src, {"WINAPI"}, {"PNG_FUNCTION"})
    assert out == b"typedef HRESULT(       *set_description)(HANDLE, PCWSTR);\n"


def test_a_byte_order_mark_does_not_hide_the_first_directive(tmp_path):
    # Files saved by Visual Studio often start with a UTF-8 byte-order mark,
    # right before the include guard. Its name, an empty #define, was then
    # blanked in `#ifndef SHAPES_H`.
    src = b"\xef\xbb\xbf#ifndef SHAPES_H\n#define SHAPES_H\nint area(int w, int h) { return w * h; }\n#endif\n"
    (tmp_path / "shapes.h").write_bytes(src)
    (tmp_path / "api.h").write_bytes(b"\xef\xbb\xbf#define EXPORT\n")
    table = scan_definitions([tmp_path / "shapes.h", tmp_path / "api.h"])
    assert {"SHAPES_H", "EXPORT"} <= table.attributes
    counters: dict = {}
    segment_file(src, "c", counters=counters, definitions=table)
    assert not counters.get("parse_error")


def test_a_wrapper_the_file_also_defines_as_a_function_keeps_the_function(tmp_path):
    # One #if branch defines the function, the other a macro that passes its
    # argument through. Unwrapping the definition deleted the function.
    src = (b"#ifdef VT_ENABLED\nfloat pack(float feedback)\n{\n    return feedback * 2.0f;\n}\n"
           b"#else\n#define pack(feedback) feedback\n#endif\n")
    table = _table(tmp_path, src.decode())
    assert "pack" in table.wrappers
    counters: dict = {}
    segment_file(src, "c", counters=counters, definitions=table)
    assert "pack" in {s["name"] for s in counters["symbols"]}
    hlsl = b"float4 pack(float4 feedback) : SV_Target\n{\n    return feedback;\n}\n"
    assert _unwrap_macros(hlsl, {"pack"}) == hlsl, "an HLSL semantic sits between the parameters and the body"
