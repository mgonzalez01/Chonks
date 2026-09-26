"""test_chunking.py: segment_file's CHUNK_MAX ceiling guarantee against
noise tables and giant single lines, and named-method preservation."""
import re
from pathlib import Path

import pytest

from chonks.index.rows import _chunk_id
from chonks.index.macro_heal import _blank_macros
from chonks.index.segment import segment_file, CHUNK_MAX


def _sizes(segs):
    return [len(s["content"].encode("utf-8")) for s in segs]


def test_oversized_class_keeps_method_names():
    """A class over CHUNK_MAX whose methods are individually tiny used to merge
    into anonymous 'block' chunks, dropping every name. The merge must now keep the
    first named boundary's identity."""
    methods = "\n".join(
        f"  int method_{i}(int x, int y) {{ return x*{i} + y - {i}; }}" for i in range(140)
    )
    src = f"class Big {{\npublic:\n{methods}\n}};\n".encode()
    assert len(src) > CHUNK_MAX, "test setup: class must exceed CHUNK_MAX to hit the split path"

    segs = segment_file(src, "cpp")
    names = [s["name"] for s in segs if s["name"]]
    assert names, "merged method chunks went anonymous"
    assert any(n.startswith("method_") for n in names), f"no method names survived: {names}"
    # merged chunks keep a real type, not the anonymous 'block'
    assert any(s["chunk_type"] != "block" for s in segs)
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_split_big_method_keeps_its_name():
    """Split-path analogue of the merge-fix regression above: the pieces must
    keep the method's name, not go anonymous 'block'."""
    body = "\n".join(f"    total += compute_{i}(x) * {i};" for i in range(400))
    src = f"int applyStep(int x) {{\n    int total = 0;\n{body}\n    return total;\n}}\n".encode()
    assert len(src) > CHUNK_MAX, "test setup: method must exceed CHUNK_MAX"

    segs = segment_file(src, "cpp")
    assert len(segs) > 1, "expected the big method to split"
    assert {s["name"] for s in segs} == {"applyStep"}, \
        f"split pieces lost the method name: {[s['name'] for s in segs]}"
    assert all(s["chunk_type"] != "block" for s in segs)
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_split_method_pieces_carry_only_their_own_calls():
    body = "\n".join(f"    total += compute_{i}(x) * {i};" for i in range(400))
    src = f"int applyStep(int x) {{\n    int total = 0;\n{body}\n    return total;\n}}\n".encode()
    segs = segment_file(src, "cpp")
    assert len(segs) > 1
    seen: set[str] = set()
    for s in segs:
        called = {e["name"] for e in s["refs"]["calls"]}
        own_lines = src.decode().splitlines()[s["start_line"] - 1:s["end_line"]]
        assert called == {f"compute_{i}" for i in range(400) if any(f"compute_{i}(" in ln for ln in own_lines)}
        seen |= called
    assert seen == {f"compute_{i}" for i in range(400)}


def test_split_class_header_keeps_its_bases_but_not_member_calls():
    m = []
    for i in range(8):
        b = "\n".join(f"    acc += f{j}(x) * {j};" for j in range(40))
        m.append(f"  int method_{i}(int x) {{\n    int acc = 0;\n{b}\n    return acc;\n  }}")
    src = ("class Store : public Base {\npublic:\n" + "\n".join(m) + "\n};\n").encode()
    header = next(s for s in segment_file(src, "cpp") if s["name"] == "Store")
    assert header["refs"]["inherits"] == ["Base"]
    assert header["refs"]["calls"] == []


def test_giant_single_line_is_hard_split_no_monster():
    line = "static const float noise[] = {" + ",".join(str(i) for i in range(50_000)) + "};\n"
    src = line.encode()
    assert len(src) > CHUNK_MAX

    segs = segment_file(src, "cpp", path="data/noise.cpp")
    sizes = _sizes(segs)
    assert sizes, "no chunks produced"
    assert max(sizes) <= CHUNK_MAX, f"oversize chunk slipped through: {max(sizes)} > {CHUNK_MAX}"
    assert len(segs) > 1


def test_giant_single_line_content_is_preserved():
    line = "x" * (CHUNK_MAX * 5) + "\n"
    segs = segment_file(line.encode(), "cpp", path="blob.cpp")
    joined = "".join(s["content"] for s in segs)
    assert joined.replace("\n", "") == ("x" * (CHUNK_MAX * 5))
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_small_function_named_and_under_ceiling():
    src = b"int add(int a, int b) {\n    return a + b;\n}\n"
    segs = segment_file(src, "cpp")
    assert any(s["name"] == "add" for s in segs)
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_oversize_counters_reported():
    line = "static const float n[] = {" + ",".join(str(i) for i in range(50_000)) + "};\n"
    counters: dict = {}
    segment_file(line.encode(), "cpp", path="n.cpp", counters=counters)
    assert counters.get("oversize_chunks", 0) >= 1
    assert counters.get("oversize_files", 0) == 1


def test_normal_file_reports_no_oversize():
    counters: dict = {}
    segment_file(b"int add(int a, int b) {\n    return a + b;\n}\n", "cpp", counters=counters)
    assert counters.get("oversize_chunks", 0) == 0
    assert counters.get("oversize_files", 0) == 0


def test_merge_floor_keeps_substantive_methods_unglued():
    """Merge-floor fix: a sub-CHUNK_MIN accessor must not absorb the substantive
    method after it, the cross-language merge bug that stole the method's name
    and blurred its embedding. A class over CHUNK_MAX forces split->merge."""
    parts = ["class Sim {"]
    for i in range(8):
        parts.append(f"  int get_{i}() const {{ return m_{i}; }}")          # trivia < CHUNK_MIN
        body = "\n".join(f"    acc += step_{j}(in) * w_{j};" for j in range(30))
        parts.append(
            f"  float method_{i}(const Input& in) {{\n    float acc = 0;\n{body}\n    return acc;\n  }}")
    parts.append("};\n")
    src = "\n".join(parts).encode()
    assert len(src) > CHUNK_MAX, "setup: class must exceed CHUNK_MAX to hit split->merge"

    segs = segment_file(src, "cpp")
    names = {s["name"] for s in segs if s["name"]}
    missing = [f"method_{i}" for i in range(8) if f"method_{i}" not in names]
    assert not missing, f"substantive methods glued away by the merge: {missing}"
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_parse_error_counter_set_on_macro_cpp():
    """With self-heal disabled, engine annotation macros make tree-sitter-cpp report
    a parse error and the counter flags the file. (With self-heal on, the default,
    those macros are blanked and the parse error is cleared; see the heal tests.)"""
    src = (b"UCLASS()\n"
           b"class MYGAME_API AActor : public Base {\n"
           b"    GENERATED_BODY()\n"
           b"    void Tick(float dt) { time += dt; }\n"
           b"};\n")
    counters: dict = {}
    segment_file(src, "cpp", path="a.cpp", counters=counters, self_heal=False)
    assert counters.get("parse_error") == 1


def test_parse_error_counter_absent_on_clean_cpp():
    counters: dict = {}
    segment_file(b"class A {\n  void f() { return; }\n};\n", "cpp", counters=counters)
    assert "parse_error" not in counters


def test_single_oversized_inner_method_keeps_its_name():
    """An oversized container with exactly one oversized child boundary must
    recurse into that child so line-slices carry the inner method's name. A C#
    class with one giant method used to line-slice as 'Calculator' instead."""
    body = "\n".join(f"        sum += Step{i}(x) * {i};" for i in range(400))
    src = (
        "public class Calculator {\n"
        "    public int MegaCompute(int x) {\n"
        "        int sum = 0;\n"
        f"{body}\n"
        "        return sum;\n"
        "    }\n"
        "}\n"
    ).encode()
    assert len(src) > CHUNK_MAX, "setup: class must exceed CHUNK_MAX to hit the split path"

    segs = segment_file(src, "c_sharp")
    assert len(segs) > 1, "expected the giant inner method to split"
    names = {s["name"] for s in segs if s["name"]}
    assert "MegaCompute" in names, f"inner method name lost: {names}"
    assert "Calculator" not in names, f"slices mis-named after the parent class: {names}"
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_oversized_class_emits_header_chunk():
    """A class over CHUNK_MAX must also emit a synthetic header chunk (class
    signature + leading access specifiers) alongside the member chunks, with
    no content duplicated between the two."""
    m = []
    for i in range(8):
        b = "\n".join(f"    acc += f{j}(x) * {j};" for j in range(40))
        m.append(f"  int method_{i}(int x) {{\n    int acc = 0;\n{b}\n    return acc;\n  }}")
    src = ("class Store {\npublic:\n" + "\n".join(m) + "\n};\n").encode()
    assert len(src) > CHUNK_MAX, "setup: class must exceed CHUNK_MAX to hit the split path"

    segs = segment_file(src, "cpp")
    names = [s["name"] for s in segs]
    assert "Store" in names, f"no class-header chunk emitted: {names}"
    assert any(n and n.startswith("method_") for n in names), f"member chunks lost: {names}"
    total = sum(len(s["content"].encode("utf-8")) for s in segs)
    assert total <= len(src), f"content duplicated: {total} bytes > src {len(src)} bytes"
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_python_module_residue_captured():
    """Module-level code outside any boundary must be captured as '<module>' /
    '<module>:__main__' residue chunks, with the def still chunked exactly once."""
    big_main = "".join(
        f'    parser.add_argument("--opt{i}", help="option {i} configures behavior")\n'
        for i in range(40)
    )
    helper = "def helper(x):\n" + "".join(
        f"    x = x + {i}  # accumulate step {i} of the helper\n" for i in range(12)
    ) + "    return x\n"
    src = (
        '"""Module docstring explaining the CLI tool and its purpose in detail."""\n'
        "import os\n" "import sys\n" "import json\n" "from pathlib import Path\n" "\n"
        "DEFAULT_TIMEOUT = 30\n" "MAX_RETRIES = 5\n"
        "CONFIG = {'alpha': 1, 'beta': 2, 'gamma': 3, 'delta': 4, 'epsilon': 5}\n" "\n"
        + helper + "\n"
        'if __name__ == "__main__":\n'
        "    import argparse\n"
        "    parser = argparse.ArgumentParser()\n"
        + big_main +
        "    args = parser.parse_args()\n"
        "    sys.exit(helper(args))\n"
    ).encode()

    segs = segment_file(src, "python")
    names = [s["name"] for s in segs]
    assert names.count("helper") == 1, f"def chunked {names.count('helper')} times: {names}"
    assert "<module>" in names, f"no <module> residue chunk: {names}"
    assert "<module>:__main__" in names, f"no <module>:__main__ residue chunk: {names}"

    total = sum(len(s["content"].encode("utf-8")) for s in segs)
    coverage = total / len(src)
    def_bytes = sum(len(s["content"].encode("utf-8")) for s in segs if s["name"] == "helper")
    baseline = def_bytes / len(src)
    assert coverage > baseline + 0.3, f"coverage {coverage:.3f} not well above baseline {baseline:.3f}"
    assert coverage > 0.8, f"module coverage too low: {coverage:.3f}"
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_residue_does_not_double_count_boundaries():
    src = (
        "def a(x):\n    return x + 1\n\n"
        "class B:\n    def m(self):\n        return 2\n\n"
        "def c(y):\n    return y * 2\n"
    ).encode()
    segs = segment_file(src, "python")
    names = [s["name"] for s in segs]
    assert "<module>" not in names, f"spurious residue chunk: {names}"
    total = sum(len(s["content"].encode("utf-8")) for s in segs)
    assert total <= len(src), f"content duplicated: {total} > {len(src)}"


def test_oversized_class_header_survives_leading_includes():
    """Synthetic-header x merge interaction: a class preceded by a license comment
    and #includes (which become '<module>' residue) must not have its header
    chunk demoted to '<module>' when it coalesces with that residue."""
    m = []
    for i in range(8):
        b = "\n".join(f"    acc += f{j}(x) * {j};" for j in range(40))
        m.append(f"  int method_{i}(int x) {{\n    int acc = 0;\n{b}\n    return acc;\n  }}")
    src = (
        "// Copyright 2026 Example Corp.\n"
        "#include <vector>\n#include <string>\n\n"
        "class Store {\npublic:\n" + "\n".join(m) + "\n};\n"
    ).encode()
    assert len(src) > CHUNK_MAX

    segs = segment_file(src, "cpp")
    names = [s["name"] for s in segs]
    assert "Store" in names, f"class name demoted by leading-include residue: {names}"
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_module_residue_does_not_demote_real_name():
    """Residue x merge interaction: a small module's def must keep its real name;
    the '<module>' residue must not win the merge and erase it."""
    src = b'"""Module doc."""\nimport os\nimport sys\n\ndef f(x):\n    return x + 1\n'
    segs = segment_file(src, "python")
    names = [s["name"] for s in segs]
    assert "f" in names, f"real def name demoted to '<module>' residue: {names}"


def test_coverage_completeness_wrapper_body_captured():
    """A top-level wrapper containing a boundary (an if-__name__ block holding a
    nested class) must not drop its own body. Regression: a CLI whose `if __name__`
    block held a nested _TqdmHandler was dropping that CLI; the handler now sits
    inside `chonks/ops/index_cmd.py`'s `main`."""
    src = (
        "import sys\n\n"
        'if __name__ == "__main__":\n'
        "    class _Fmt:\n"
        "        def emit(self):\n"
        "            return 1\n"
        "    SETUP_MARKER = configure(sys.argv)\n"
        "    run_cli(SETUP_MARKER)\n"
    ).encode()
    segs = segment_file(src, "python")
    joined = "\n".join(s["content"] for s in segs)
    assert "SETUP_MARKER" in joined, "wrapper body (CLI) dropped by a coverage gap"
    assert "run_cli" in joined


def test_coverage_completeness_split_inter_child_captured():
    """A function over CHUNK_MAX splits into its nested defs; the orchestration
    code between/after those defs must still be captured."""
    inner = []
    for k in range(4):
        body = "\n".join(f"        v += step_{j}(v)" for j in range(120))
        inner.append(f"    def worker_{k}():\n        v = 0\n{body}\n        return v")
    orchestration = "\n".join(f"    ORCH_LINE_{k} = worker_{k}()" for k in range(4))
    src = ("def driver():\n" + "\n".join(inner) + "\n" + orchestration + "\n    return 0\n").encode()
    assert len(src) > CHUNK_MAX, "setup: driver must exceed CHUNK_MAX to split into nested defs"

    segs = segment_file(src, "python")
    joined = "\n".join(s["content"] for s in segs)
    for k in range(4):
        assert f"ORCH_LINE_{k}" in joined, f"orchestration line {k} dropped between split children"
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_split_class_member_lines_are_in_every_chunk_spanning_them():
    fields = "\n".join(f"    int light_mask_{i} = {i};" for i in range(20))
    decls = "\n".join(
        f"    void draw_shape_{i}(const Point2 &p_from, const Point2 &p_to, float p_width = -1.0);"
        for i in range(70))
    src = (
        "class Canvas {\n"
        "    struct Data {\n"
        "        int index = 0;\n"
        "    };\n"
        f"{fields}\n"
        "    bool is_dirty() const { return dirty; }\n"
        f"{decls}\n"
        "    int get_id() const { return id; }\n"
        "};\n"
    ).encode()
    assert len(src) > CHUNK_MAX

    segs = segment_file(src, "cpp")
    own = [{ln.strip() for ln in s["content"].splitlines()} for s in segs]
    for n, line in enumerate(src.decode().splitlines(), start=1):
        if not re.search(r"\w", line):
            continue
        spanning = [i for i, s in enumerate(segs) if s["start_line"] <= n <= s["end_line"]]
        assert spanning, f"line {n} is in no chunk: {line.strip()}"
        for i in spanning:
            assert line.strip() in own[i], (
                f"chunk {segs[i]['name']} {segs[i]['start_line']}-{segs[i]['end_line']} "
                f"spans line {n} but lacks it: {line.strip()}")
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_macro_self_heal_recovers_class_name():
    """Macro self-heal: clean engine C++ (UCLASS + GENERATED_BODY) trips tree-sitter;
    after discovering + blanking the macros, the class name is recovered, the file no
    longer counts as a parse_error, and the discovered vocabulary is reported."""
    src = (
        "UCLASS()\n"
        "class AMyActor : public AActor {\n"
        "    GENERATED_BODY()\n"
        "public:\n"
        "    void Tick(float dt) { time += dt; }\n"
        "private:\n"
        "    float time = 0.0f;\n"
        "};\n"
    ).encode()
    counters: dict = {}
    segs = segment_file(src, "cpp", counters=counters)
    names = {s["name"] for s in segs if s["name"]}
    assert "AMyActor" in names, f"class name not recovered after heal: {names}"
    assert "parse_error" not in counters, "healed file should not count as parse_error"
    assert counters.get("macro_healed") == 1
    assert {"UCLASS", "GENERATED_BODY"} <= counters.get("discovered_macros", set())


def test_macro_self_heal_preserves_original_macro_text():
    """Content fidelity: blanking is only for the PARSE; stored chunk content is
    read from the ORIGINAL bytes, so the macro lines survive verbatim."""
    src = (
        "UCLASS()\n"
        "class AThing : public AActor {\n"
        "    GENERATED_BODY()\n"
        "    void Run() { go(); }\n"
        "};\n"
    ).encode()
    joined = "\n".join(s["content"] for s in segment_file(src, "cpp"))
    assert "UCLASS()" in joined and "GENERATED_BODY()" in joined, "original macro text dropped"


def test_macro_self_heal_validation_rejects_plain_call():
    """The validation gate: a real call like ASSERT(x) (which already parses) must
    not be admitted as a macro; only tokens whose blanking strictly reduces the
    parse-error count are kept."""
    src = (
        "UCLASS()\n"
        "class AThing : public AActor {\n"
        "    GENERATED_BODY()\n"
        "    void f(int x) { ASSERT(x > 0); }\n"
        "};\n"
    ).encode()
    counters: dict = {}
    segment_file(src, "cpp", counters=counters)
    disc = counters.get("discovered_macros", set())
    assert "UCLASS" in disc, f"real macro not discovered: {disc}"
    assert "ASSERT" not in disc, f"plain call wrongly admitted as a macro: {disc}"


def test_blank_macros_preserves_length_and_newlines():
    """The offset-safety invariant: blanking must preserve byte length and newline
    positions, or every tree-sitter offset against the original bytes would shift."""
    src = b"UCLASS()\nclass A {\n  UPROPERTY(EditAnywhere) int x;\n};\n"
    out = _blank_macros(src, {"UCLASS", "UPROPERTY"})
    assert len(out) == len(src)
    assert [i for i, b in enumerate(src) if b == 0x0A] == [i for i, b in enumerate(out) if b == 0x0A]
    assert b"UCLASS" not in out and b"UPROPERTY" not in out
    assert b"class A" in out and b"int x;" in out  # non-macro content untouched


def test_macro_self_heal_does_not_blank_bare_type():
    """Validation-gate hardening: a bare ALL_CAPS *type* (RID) must not be admitted
    as a macro just because blanking it coincidentally drops the error count."""
    # mangled-style: a forward decl with its trailing ';' stripped breaks the parse
    # for a NON-macro reason; RID here is a real return type, not a macro.
    src = (b"class RenderingDevice\n"      # missing ';'
           b"class Store {\n"
           b"    RID get_id() const { return id; }\n"
           b"};\n")
    counters: dict = {}
    segment_file(src, "cpp", counters=counters)
    assert "RID" not in counters.get("discovered_macros", set()), "bare type wrongly blanked as a macro"


@pytest.mark.parametrize("src,name", [
    (b"class StringBuilder {\n  _FORCE_INLINE_ operator String() const { return s; }\n};\n",
     "operator String"),
    # The class-head macro's error lands at the closing brace, far from it.
    (b"class _WARN_UNUSED_ HashSet {\n  int x;\n" + b"  int pad;\n" * 40 +
     b"  _FORCE_INLINE_ int g() const { return x; }\n};\n", "HashSet"),
    (b"class API_AVAILABLE(macos(11.0), ios(14.0)) Surface {\n  int x;\n};\n", "Surface"),
    (b"class It {\n  It begin() const _LIFETIME_BOUND_ { return *this; }\n};\n", "It::begin"),
], ids=["prefix", "class-head", "class-head-nested-args", "suffix"])
def test_macro_self_heal_finds_attribute_macros(src, name):
    counters: dict = {}
    segment_file(src, "cpp", counters=counters)
    names = {sym["name"] for sym in counters["symbols"]}
    assert name in names or name.split("::")[-1] in names, names
    assert "parse_error" not in counters


def test_macro_self_heal_never_blanks_a_type_the_file_defines():
    src = (b"struct AABB { int x; };\n"
           b"class RenderingDevice\n"      # missing ';', a non-macro parse error
           b"class Store {\n"
           b"    AABB get_aabb() const { return a; }\n"
           b"};\n")
    from chonks.index.macro_heal import _discover_macros
    from tree_sitter_language_pack import get_parser
    assert "AABB" not in _discover_macros(get_parser("cpp").parse(src).root_node, src)
    counters: dict = {}
    segment_file(src, "cpp", counters=counters)
    assert "AABB" not in counters.get("discovered_macros", set())


def test_macro_self_heal_does_not_admit_the_type_beside_a_macro():
    # Blanking RID also fixes `_FORCE_INLINE_ RID get_target()`, but only
    # _FORCE_INLINE_ is needed; RID is a real type the file uses cleanly.
    src = (b"class Buffers {\n"
           b"  RID render_target;\n"
           b"  _FORCE_INLINE_ RID get_render_target() const { return render_target; }\n"
           b"};\n")
    counters: dict = {}
    segment_file(src, "cpp", counters=counters)
    assert counters.get("discovered_macros") == {"_FORCE_INLINE_"}


@pytest.mark.parametrize("src,macros", [
    (b"class Ref {\n  ULONG count;\n  ULONG STDMETHODCALLTYPE AddRef() { return ++count; }\n};\n",
     {"STDMETHODCALLTYPE"}),
    (b"class Client {\n  UINT32 frames;\n  HRESULT Get(_Out_ UINT32 *p) { *p = frames; return 0; }\n};\n",
     {"_Out_"}),
], ids=["calling-convention", "sal-annotation"])
def test_macro_self_heal_blames_the_macro_not_the_type(src, macros):
    counters: dict = {}
    segment_file(src, "cpp", counters=counters)
    assert counters.get("discovered_macros") == macros
    assert "parse_error" not in counters


def test_macro_self_heal_drops_early_admissions_a_later_macro_makes_unneeded():
    # libjpeg shape: a name admitted in an early pass is unneeded once
    # LOCAL(void), the real macro, is found in a later one.
    src = (b"INLINE\nLOCAL(void)\nupsample(j_decompress_ptr cinfo,\n                _JSAMPIMAGE input_buf,\n"
           b"                JDIMENSION in_row_group_ctr)\n{\n  JDIMENSION col;\n  for (col = 0; col < 4; col++) { }\n}\n") * 3
    counters: dict = {}
    segment_file(src, "c", counters=counters)
    assert counters.get("discovered_macros") == {"LOCAL"}
    assert "parse_error" not in counters


def test_blank_macros_blanks_nested_arguments():
    src = b"class API_AVAILABLE(macos(11.0), ios(14.0)) Surface {};\n"
    out = _blank_macros(src, {"API_AVAILABLE"})
    assert len(out) == len(src)
    assert out.split() == [b"class", b"Surface", b"{};"]


def test_macro_self_heal_clears_parse_error_for_qt_signals_slots():
    """Regression: Q_OBJECT/Q_PROPERTY are ALL_CAPS and already healed, but the
    lowercase `signals:`/`slots:` Qt access-specifier idiom is not ALL_CAPS, so
    the old code never blanked it and parse_error stayed set regardless."""
    src = (
        "class MyWidget : public QObject {\n"
        "    Q_OBJECT\n"
        "public:\n"
        "    explicit MyWidget(QObject *parent = nullptr);\n"
        "    int value() const;\n"
        "\n"
        "signals:\n"
        "    void valueChanged(int newValue);\n"
        "\n"
        "public slots:\n"
        "    void setValue(int v);\n"
        "\n"
        "private Q_SLOTS:\n"
        "    void internalSlot();\n"
        "\n"
        "private:\n"
        "    int m_value;\n"
        "};\n"
    ).encode()
    counters: dict = {}
    segs = segment_file(src, "cpp", counters=counters)
    names = {s["name"] for s in segs if s["name"]}
    assert "MyWidget" in names, f"class name not recovered after heal: {names}"
    assert "parse_error" not in counters, \
        f"Qt signals/slots heal left a false parse_error: {counters}"
    assert counters.get("macro_healed") == 1
    assert {"Q_OBJECT", "signals", "slots", "Q_SLOTS"} <= counters.get("discovered_macros", set())


def test_macro_self_heal_qt_signals_slots_persisted_vocab_stays_clean():
    """The persisted macro-vocab path (macros pre-blanked before self-heal even
    runs) must clear the same Qt signals/slots idiom; otherwise a second run
    would regress to a false parse_error once the macros are merely pre-blanked."""
    src = (
        "class MyWidget : public QObject {\n"
        "    Q_OBJECT\n"
        "signals:\n"
        "    void valueChanged(int newValue);\n"
        "public slots:\n"
        "    void setValue(int v);\n"
        "};\n"
    ).encode()
    vocab = {"Q_OBJECT", "signals", "slots", "Q_SIGNALS", "Q_SLOTS"}
    counters: dict = {}
    segs = segment_file(src, "cpp", counters=counters, macros=vocab, self_heal=False)
    names = {s["name"] for s in segs if s["name"]}
    assert "MyWidget" in names
    assert "parse_error" not in counters, \
        f"pre-blanked vocab path left a false parse_error: {counters}"


def test_blank_macros_qt_specifier_only_blanks_requested_names():
    """Regression: _blank_macros's Qt-specific path used to gate on `names` but
    then blank matches from the full fixed 4-name alternation regardless of
    what was requested. Requesting only 'signals' must leave 'slots:' untouched."""
    src = b"class A {\npublic slots:\n    void f();\n};\n"
    out = _blank_macros(src, {"signals"})
    assert b"slots" in out, "requesting 'signals' must not blank an unrelated 'slots:'"
    assert len(out) == len(src)


def test_macro_self_heal_qt_specifiers_survive_macro_dense_truncation():
    """Regression: _QT_ACCESS_SPECIFIERS used to be unioned into the candidate
    set before the sorted(...)[:_MAX_MACRO_CANDIDATES] truncation, so a
    macro-dense file could sort signals:/slots: off the list before heal tried them."""
    macro_lines = "\n".join(f"    MACRO_{i:02d};" for i in range(60))
    src = (
        "class MyWidget : public QObject {\n"
        "    Q_OBJECT\n"
        f"{macro_lines}\n"
        "public:\n"
        "    explicit MyWidget(QObject *parent = nullptr);\n"
        "\n"
        "signals:\n"
        "    void valueChanged(int newValue);\n"
        "\n"
        "public slots:\n"
        "    void setValue(int v);\n"
        "\n"
        "private:\n"
        "    int m_value;\n"
        "};\n"
    ).encode()
    counters: dict = {}
    segs = segment_file(src, "cpp", counters=counters)
    names = {s["name"] for s in segs if s["name"]}
    assert "MyWidget" in names, f"class name not recovered after heal: {names}"
    assert "parse_error" not in counters, \
        f"macro-dense file starved Qt specifiers out of the heal: {counters}"
    assert {"signals", "slots"} <= counters.get("discovered_macros", set())


def test_collect_symbols_records_folded_class_methods():
    """The decouple: a small class is emitted as ONE chunk (methods folded), but
    the symbol index records the class and each method independent of packing."""
    src = (b"class Widget {\npublic:\n"
           b"  int get_x() const { return x_; }\n"
           b"  void set_x(int v) { x_ = v; }\n"
           b"private:\n  int x_ = 0;\n};\n")
    counters: dict = {}
    segs = segment_file(src, "cpp", counters=counters)
    chunk_names = {s["name"] for s in segs if s["name"]}
    sym_names = {s["name"] for s in counters["symbols"]}
    assert chunk_names == {"Widget"}, f"small class should fold to one chunk: {chunk_names}"
    assert "Widget" in sym_names and {"get_x", "set_x"} <= sym_names, \
        f"folded methods missing from symbol index: {sym_names}"


def test_collect_symbols_after_macro_heal():
    """Symbols are collected from the healed tree, so a macro-broken class and its
    methods are recorded under their real names."""
    src = (b"UCLASS()\nclass AThing : public AActor {\n"
           b"    GENERATED_BODY()\n    void Run() { go(); }\n};\n")
    counters: dict = {}
    segment_file(src, "cpp", counters=counters)
    sym_names = {s["name"] for s in counters["symbols"]}
    assert "AThing" in sym_names and "Run" in sym_names, f"heal symbols missing: {sym_names}"


# ---------------------------------------------------------------------------
# Cheaper heal admission (early-bail error count, error-proximity
# prefilter) plus unhealable-content memoization. Equivalence is required:
# a healed file's output must be byte-identical; only the cost may change.
# ---------------------------------------------------------------------------

def test_macro_self_heal_unhealable_counter_set_when_nothing_admitted():
    """When the heal sweep runs to completion but admits nothing, the
    heal_unhealable counter flags it so chonks/index/pipeline.py can persist the content hash and
    skip the sweep on future runs. A genuinely-broken file with no macro-shaped
    tokens near the error has nothing for the sweep to admit."""
    src = (
        b"class Broken\n"       # missing ';' -> permanent, irreducible parse error
        b"class Other {\n"
        b"    void run() { do_thing(); }\n"
        b"};\n"
    )
    counters: dict = {}
    segment_file(src, "cpp", counters=counters)
    assert counters.get("parse_error") == 1
    assert "macro_healed" not in counters
    assert counters.get("heal_unhealable") == 1


def test_macro_self_heal_healed_file_does_not_set_unhealable_counter():
    """The flip side: a file the heal sweep DOES fix must never set
    heal_unhealable, since chonks/index/pipeline.py would otherwise wrongly memoize it and skip
    self-heal on a later run where it's actually needed again."""
    src = (
        "UCLASS()\n"
        "class AMyActor : public AActor {\n"
        "    GENERATED_BODY()\n"
        "};\n"
    ).encode()
    counters: dict = {}
    segment_file(src, "cpp", counters=counters)
    assert counters.get("macro_healed") == 1
    assert "heal_unhealable" not in counters


def test_macro_self_heal_admission_prefilter_does_not_change_outcome():
    """Stress test for the error-proximity prefilter: many ALL_CAPS macro-shaped
    calls that already parse cleanly, far from the real UCLASS/GENERATED_BODY
    error, must be dropped without a trial reparse; the real macros still heal."""
    noise = "\n".join(f"void noise_{i}() {{ NOISE_CALL_{i:02d}(); }}" for i in range(30))
    src = (
        f"{noise}\n"
        "UCLASS()\n"
        "class AMyActor : public AActor {\n"
        "    GENERATED_BODY()\n"
        "public:\n"
        "    void Tick(float dt) { time += dt; }\n"
        "private:\n"
        "    float time = 0.0f;\n"
        "};\n"
    ).encode()
    counters: dict = {}
    segs = segment_file(src, "cpp", counters=counters)
    joined = "\n".join(s["content"] for s in segs)
    assert "AMyActor" in joined, "class recovered by heal must survive in chunk content"
    assert "parse_error" not in counters
    assert counters.get("macro_healed") == 1
    disc = counters.get("discovered_macros", set())
    assert {"UCLASS", "GENERATED_BODY"} <= disc
    assert not any(m.startswith("NOISE_CALL") for m in disc), (
        f"clean, error-distant NOISE_CALL_* calls wrongly admitted as macros: {disc}"
    )


def _bigfunc(tag, lang="cpp"):
    if lang == "python":
        body = "\n".join(f"    acc += step_{tag}_{j}(x)" for j in range(22))
        return f"def func_{tag}(x):\n    acc = 0\n{body}\n    return acc"
    body = "\n".join(f"    acc += step_{tag}_{j}(x);" for j in range(22))
    return f"void func_{tag}(int x) {{\n    int acc = 0;\n{body}\n}}"


def test_pure_divider_dropped():
    src = (_bigfunc("A") + "\n\n//==================================//\n\n" + _bigfunc("B") + "\n").encode()
    segs = segment_file(src, "cpp")
    assert {s["name"] for s in segs} == {"func_A", "func_B"}
    assert not any("====" in s["content"] for s in segs), "pure divider survived as a chunk"


def test_labeled_banner_attaches_forward():
    banner = "//--------------------------//\n//   Rendering section        //\n//--------------------------//"
    src = (_bigfunc("A") + "\n\n" + banner + "\n\n" + _bigfunc("B") + "\n").encode()
    segs = segment_file(src, "cpp")
    assert {s["name"] for s in segs} == {"func_A", "func_B"}, "banner became its own chunk"
    funcB = next(s for s in segs if s["name"] == "func_B")
    assert "Rendering section" in funcB["content"], "banner text not carried with the function"


def test_doc_comment_rides_with_function():
    src = ("// Computes the weighted accumulation for the frame.\n" + _bigfunc("A") + "\n").encode()
    segs = segment_file(src, "cpp")
    a = next(s for s in segs if s["name"] == "func_A")
    assert "Computes the weighted accumulation" in a["content"], "doc-comment not attached"
    assert all(s["name"] for s in segs), "a standalone comment chunk slipped through"


def test_big_comment_block_left_standalone():
    block = "\n".join(f"// old line {i} of a disabled implementation kept around here" for i in range(12))
    src = (_bigfunc("A") + "\n\n" + block + "\n\n" + _bigfunc("B") + "\n").encode()
    segs = segment_file(src, "cpp")
    funcB = next(s for s in segs if s["name"] == "func_B")
    assert "old line 0" not in funcB["content"], "big comment block polluted the function chunk"


def test_python_leading_comment_attaches():
    src = ("# helper that accumulates the steps\n" + _bigfunc("A", "python") + "\n").encode()
    segs = segment_file(src, "python")
    a = next(s for s in segs if s["name"] == "func_A")
    assert "helper that accumulates" in a["content"]


def test_normal_multiline_table_splits_cleanly():
    """Control case for the hard-wrap tests below: one-value-per-line splits via
    the normal line path, no hard-wrap needed."""
    table = "static const int t[] = {\n" + "\n".join(f"  {i}," for i in range(20_000)) + "\n};\n"
    segs = segment_file(table.encode(), "cpp", path="t.cpp")
    assert all(z <= CHUNK_MAX for z in _sizes(segs))
    assert len(segs) > 10


# ---------------------------------------------------------------------------
# Oversized module residue must be line-sliced, not byte-window
# hard-wrapped with duplicated (wrong) line spans.
# ---------------------------------------------------------------------------

def test_oversized_preamble_residue_gets_distinct_correct_spans():
    """An oversized top-level residue run used to be emitted as ONE segment
    and byte-window hard-split, with every piece stamped with the WHOLE run's
    line span. Must now line-slice into distinct, correctly-spanned chunks."""
    includes = "".join(
        f'#include "very_long_generated_header_name_{i:04d}.h"\n' for i in range(400)
    )
    src = (includes + "int add(int a, int b) {\n    return a + b;\n}\n").encode()
    assert len(includes.encode("utf-8")) > CHUNK_MAX, "setup: preamble must exceed CHUNK_MAX"

    segs = segment_file(src, "cpp", path="preamble.cpp")
    modules = [s for s in segs if s["chunk_type"] == "module"]
    assert len(modules) > 1, "oversized preamble did not split into multiple module chunks"
    spans = [(s["start_line"], s["end_line"]) for s in modules]
    assert len(spans) == len(set(spans)), f"duplicate line spans survived: {spans}"
    for s in modules:
        assert s["content"].lstrip().startswith("#include"), \
            f"module piece starts mid-line: {s['content'][:40]!r}"
    assert any(s["name"] == "add" for s in segs)
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_oversized_synthetic_class_header_gets_distinct_correct_spans():
    """The synthetic container-header chunk has the same byte-window failure
    mode when it's itself over CHUNK_MAX: it must line-slice into distinct
    spans, not duplicate its own full range across every hard-wrap piece."""
    banner = "\n".join(f"// license banner line {i} with some extra padding text here" for i in range(150))
    methods = "\n".join(
        f"  int method_{i}(int x, int y) {{ return x*{i} + y - {i}; }}" for i in range(140)
    )
    src = f"class Big {{\n{banner}\npublic:\n{methods}\n}};\n".encode()
    assert len(src) > CHUNK_MAX, "setup: class must exceed CHUNK_MAX to hit the split path"

    segs = segment_file(src, "cpp")
    headers = [s for s in segs if s["name"] == "Big"]
    assert len(headers) > 1, f"expected the oversized header to line-slice, got {len(headers)}"
    spans = [(s["start_line"], s["end_line"]) for s in headers]
    assert len(spans) == len(set(spans)), f"duplicate header spans survived: {spans}"
    # Every piece's first content line must equal the actual source line at
    # its claimed start_line (not checked against a fixed prefix, since a
    # later piece may have absorbed the first small method via merge).
    src_lines = src.decode().splitlines()
    for h in headers:
        first_line = h["content"].splitlines()[0]
        assert first_line == src_lines[h["start_line"] - 1], \
            f"header piece starts mid-line: {first_line!r} != source line {h['start_line']}: {src_lines[h['start_line'] - 1]!r}"
    assert headers[0]["content"].lstrip().startswith("class Big")
    assert all(z <= CHUNK_MAX for z in _sizes(segs))


def test_enforce_ceiling_apportions_spans_not_duplicates():
    """_enforce_ceiling's byte-window hard-wrap used to stamp EVERY piece with
    the original segment's full line span. Must now apportion spans by
    counting newlines in each piece, so pieces get distinct starts."""
    from chonks.index.segment import _enforce_ceiling, _SyntheticSegment

    lines = [f"long_line_{i:03d} = " + "x" * 400 + "\n" for i in range(20)]
    text = "".join(lines)
    seg = _SyntheticSegment(text, start_line=100, end_line=119, chunk_type="module", name="<module>")
    assert seg.size(b"") > CHUNK_MAX, "setup: segment must exceed CHUNK_MAX to hit the hard-wrap path"

    out = _enforce_ceiling([seg], b"")
    assert len(out) > 1, "expected the oversized segment to hard-split"
    starts = [s.start_line for s in out]
    assert len(set(starts)) > 1, "every piece still stamped with the same start line"
    assert starts == sorted(starts), "piece start lines must be non-decreasing"
    assert all(100 <= s.start_line <= 120 and 100 <= s.end_line <= 120 for s in out), \
        "apportioned spans escaped the original segment's line range"


# ---------------------------------------------------------------------------
# Identity-free trivia fragments must be absorbed into a substantive
# neighbour, not survive as standalone orphan chunks.
# ---------------------------------------------------------------------------

def test_file_tail_absorbed_into_last_function_keeps_name():
    """A file-tail '} // namespace' fragment must be absorbed into the last
    function chunk, which keeps its own name, not demoted to '<module>'."""
    src = (
        "namespace foo {\n"
        + _bigfunc("A") + "\n"
        "} // namespace foo\n"
        "#endif\n"
    ).encode()
    segs = segment_file(src, "cpp")
    names = [s["name"] for s in segs]
    assert names == ["func_A"], f"tail fragment survived as its own orphan chunk: {names}"
    funcA = segs[0]
    assert "// namespace foo" in funcA["content"]
    assert "#endif" in funcA["content"]


def test_leading_guard_absorbed_into_function_below():
    """A lone '#if !X' guard directly above a function (with no matching
    '#endif' in the same fragment) is absorbed FORWARD into the function it
    guards, which keeps its own name."""
    src = (
        "#if !SOME_CONDITION\n"
        + _bigfunc("A") + "\n"
        "#endif\n"
    ).encode()
    segs = segment_file(src, "cpp")
    names = [s["name"] for s in segs]
    assert names == ["func_A"], f"guard fragment survived as its own orphan chunk: {names}"
    assert "#if !SOME_CONDITION" in segs[0]["content"]
    assert "#endif" in segs[0]["content"]


def test_leading_comment_keeps_the_function_refs():
    for lang, comment in (("cpp", "// Adds the steps."), ("python", "# Adds the steps.")):
        src = (_bigfunc("A", lang) + "\n\n" + comment + "\n" + _bigfunc("B", lang) + "\n").encode()
        segs = segment_file(src, lang)
        b = next(s for s in segs if s["name"] == "func_B")
        assert comment in b["content"]
        called = {e["name"] for e in b["refs"]["calls"]}
        assert "step_B_0" in called, lang


def test_trailing_namespace_comment_attaches_backward():
    """_attach_or_drop_comments: a trailing small labeled comment must attach
    BACKWARD to the last emitted segment, the blind spot distinct from the
    forward-attach case tested elsewhere."""
    src = (_bigfunc("A") + "\n// namespace foo\n").encode()
    segs = segment_file(src, "cpp")
    names = [s["name"] for s in segs]
    assert names == ["func_A"], f"trailing comment survived as its own orphan chunk: {names}"
    assert "// namespace foo" in segs[0]["content"]


def test_is_identity_free_does_not_misclassify_pointer_dereference():
    """Code-review follow-up: a pointer-dereference statement like '*ptr = val;'
    starts with the same '*' used to detect C-doc-comment continuation lines
    ('* blah'), but must not be classified identity-free."""
    from chonks.index.segment import _is_identity_free

    assert _is_identity_free("*ptr = 5;\n", "cpp") is False
    assert _is_identity_free("*out++ = value;\n", "cpp") is False
    # Genuine doc-comment continuation lines must still be recognized.
    assert _is_identity_free("* blah\n", "cpp") is True
    assert _is_identity_free("*/\n", "cpp") is True


def test_enforce_ceiling_zero_newline_pieces_get_distinct_spans():
    """Code-review follow-up: a monster line with no newline at all must not
    degenerate into the original duplicate-span bug where every piece is
    stamped with the identical (line, line) span."""
    from chonks.index.segment import _enforce_ceiling, _SyntheticSegment

    text = "x = " + "a" * 20000 + ";"
    seg = _SyntheticSegment(text, start_line=50, end_line=50, chunk_type="block", name="blob")
    assert seg.size(b"") > 6000

    out = _enforce_ceiling([seg], b"")
    assert len(out) > 1, "expected the monster line to hard-split"
    spans = [(s.start_line, s.end_line) for s in out]
    assert len(spans) == len(set(spans)), f"duplicate spans on zero-newline pieces: {spans}"


def test_enforce_ceiling_no_off_by_one_boundary_overlap():
    """Code-review follow-up: apportioning spans by newline count must not
    double-count the boundary line when a piece ends exactly on '\\n', that
    used to make two consecutive pieces both claim the same line."""
    from chonks.index.segment import _enforce_ceiling, _SyntheticSegment

    text = ("x" * 50 + "\n") * 300
    seg = _SyntheticSegment(text, start_line=100, end_line=100 + 300, chunk_type="block", name="blob")
    out = _enforce_ceiling([seg], text.encode())
    assert len(out) > 1
    spans = [(s.start_line, s.end_line) for s in out]
    assert len(spans) == len(set(spans)), f"duplicate spans: {spans}"
    for a, b in zip(out, out[1:]):
        assert a.end_line <= b.start_line, f"piece spans overlap by more than the shared boundary: {a.end_line} -> {b.start_line}"


def test_absorb_across_blank_line_gap_keeps_span_content_honest():
    """Code-review follow-up: absorbing an identity-free fragment across a
    blank-line gap must not stamp the merged chunk with the fragment's raw
    end_line, which can overlap the next chunk's start_line."""
    src = (
        _bigfunc("A") + "\n"
        "\n\n\n\n\n"
        "#endif\n"
        + _bigfunc("B") + "\n"
    ).encode()
    segs = segment_file(src, "cpp")
    names = [s["name"] for s in segs]
    assert names == ["func_A", "func_B"], f"unexpected chunk set: {names}"
    a, b = segs[0], segs[1]
    assert a["end_line"] < b["start_line"], (
        f"absorbed-fragment span overlaps the following chunk: "
        f"func_A end={a['end_line']} func_B start={b['start_line']}"
    )
    content_lines = a["content"].count("\n") + (0 if a["content"].endswith("\n") else 1)
    span_len = a["end_line"] - a["start_line"] + 1
    assert span_len == content_lines, (
        f"span claims {span_len} lines but content has {content_lines}: {a['start_line']}-{a['end_line']}"
    )


def test_bare_namespace_opening_absorbed_forward_not_backward():
    """Code-review follow-up: a bare container-opening fragment like
    'namespace Foo {' introduces the member that follows it and must attach
    FORWARD into that member, not backward into an unrelated preceding chunk."""
    src = (
        _bigfunc("A") + "\n\n"
        "namespace Foo {\n\n"
        + _bigfunc("B") + "\n\n"
        "}\n"
    ).encode()
    segs = segment_file(src, "cpp")
    names = [s["name"] for s in segs]
    assert names == ["func_A", "func_B"], f"unexpected chunk set: {names}"
    func_a, func_b = segs[0], segs[1]
    assert "namespace Foo {" not in func_a["content"], (
        "namespace opening wrongly absorbed backward into the preceding function"
    )
    assert "namespace Foo {" in func_b["content"], (
        "namespace opening was not absorbed forward into the function it introduces"
    )


def test_forward_absorbed_namespace_opening_keeps_the_function_end_line():
    body = _bigfunc("B")
    src = ("namespace Foo {\n\n" + body + "\n\n}\n").encode()
    segs = segment_file(src, "cpp")
    func_b = next(s for s in segs if s["name"] == "func_B")
    assert "namespace Foo {" in func_b["content"]
    assert func_b["end_line"] == 2 + len(body.split("\n"))


def test_small_named_function_not_merged_into_large_neighbor():
    """Regression pin: a small named function next to a large one must stay
    its own chunk. Identity-free absorption must not weaken this, since a
    small function's body isn't one of _is_identity_free's narrow shapes."""
    src = ("int get_x() { return 1; }\n\n" + _bigfunc("B") + "\n").encode()
    segs = segment_file(src, "cpp")
    names = {s["name"] for s in segs}
    assert names == {"get_x", "func_B"}, f"small function got merged/renamed away: {names}"
    assert len(segs) == 2


# ---------------------------------------------------------------------------
# Typed reference extraction: calls / imports / inherits (issue: typed edges)
# ---------------------------------------------------------------------------

def _refs(segs, name):
    seg = next(s for s in segs if s["name"] == name)
    return seg["refs"]


def _call_names(refs) -> set:
    """Bare names out of a calls list: the fingerprint change turned each entry
    into a {'name','receiver','arity'} dict, so this recovers the old
    `set(refs["calls"])` shape for pre-existing tests."""
    return {e["name"] if isinstance(e, dict) else e for e in refs["calls"]}


def _call_entry(refs, name: str) -> dict:
    """The single calls-list entry for a given bare name; asserts there's
    exactly one, since every test that uses this picks a distinct call."""
    matches = [e for e in refs["calls"] if isinstance(e, dict) and e["name"] == name]
    assert len(matches) == 1, f"expected exactly one {name!r} call entry, got {matches}"
    return matches[0]


def test_extract_refs_python():
    src = b'''
import os
from collections import defaultdict

class Foo(Base):
    def run(self):
        helper()
        self.other.call(x)
'''
    segs = segment_file(src, "python", path="t.py")
    refs = _refs(segs, "Foo")
    assert _call_names(refs) == {"helper", "call"}
    assert "os" in refs["imports"] and "collections" in refs["imports"]
    assert refs["inherits"] == ["Base"]
    # imports/inherits stay plain strings; only 'calls' gains the fingerprint.
    assert all(isinstance(x, str) for x in refs["imports"])
    assert all(isinstance(x, str) for x in refs["inherits"])
    # helper() is a free call (no receiver); self.other.call(x) -> receiver
    # is the IMMEDIATE token before the called name ('other'), not 'self'.
    assert _call_entry(refs, "helper") == {"name": "helper", "receiver": None, "arity": 0}
    assert _call_entry(refs, "call") == {"name": "call", "receiver": "other", "arity": 1}


def test_extract_refs_cpp():
    src = b'''
#include <foo.h>
#include "bar.h"

class Foo : public Base, protected IOther {
public:
    void run() { helper(); this->other->call(x); Bar::Static(); }
};
'''
    segs = segment_file(src, "cpp", path="t.cpp")
    refs = _refs(segs, "Foo")
    assert _call_names(refs) == {"helper", "call", "Static"}
    assert set(refs["imports"]) == {"foo.h", "bar.h"}
    assert set(refs["inherits"]) == {"Base", "IOther"}
    assert _call_entry(refs, "helper") == {"name": "helper", "receiver": None, "arity": 0, "cls": "Foo"}
    assert _call_entry(refs, "call") == {
        "name": "call", "receiver": "other", "arity": 1, "racc": "->",
        "rhead": {"name": "this", "via": "this", "type": None}, "rpath": ["other"], "cls": "Foo",
    }
    # Bar::Static() is a single-level qualified call: receiver is the whole
    # (only) qualifier, 'Bar'.
    assert _call_entry(refs, "Static") == {
        "name": "Static", "receiver": "Bar", "arity": 0, "racc": "::",
        "rhead": {"name": "Bar", "via": "type", "type": "Bar"}, "cls": "Foo",
    }


def test_extract_refs_c_sharp():
    src = b'''
using System;
using Godot;

namespace Foo {
class Bar : Node, IThing {
    void Run() {
        Helper();
        this.other.Call(x);
        new Widget();
    }
}
}
'''
    segs = segment_file(src, "c_sharp", path="t.cs")
    refs = _refs(segs, "Bar")
    assert _call_names(refs) >= {"Helper", "Call", "Widget"}
    assert set(refs["imports"]) == {"System", "Godot"}
    assert set(refs["inherits"]) == {"Node", "IThing"}
    assert _call_entry(refs, "Helper") == {"name": "Helper", "receiver": None, "arity": 0}
    assert _call_entry(refs, "Call") == {"name": "Call", "receiver": "other", "arity": 1}
    assert _call_entry(refs, "Widget") == {"name": "Widget", "receiver": None, "arity": 0}


def test_extract_refs_gdscript():
    src = b'''
extends Node2D

func _ready():
    var x = get_node("Bar")
    x.do_thing()
'''
    segs = segment_file(src, "gdscript", path="t.gd")
    refs = _refs(segs, "_ready")
    assert _call_names(refs) >= {"get_node", "do_thing"}
    assert refs["inherits"] == ["Node2D"]
    assert _call_entry(refs, "get_node") == {"name": "get_node", "receiver": None, "arity": 1}
    assert _call_entry(refs, "do_thing") == {"name": "do_thing", "receiver": "x", "arity": 0}


def test_extract_refs_hlsl_calls_and_includes():
    """HLSL has C's call and #include nodes. A constructor such as
    float4(p, 1) is a call too; nothing defines float4, so it never becomes
    an edge."""
    src = b'''#include "Packages/Core/Common.hlsl"
struct VSOut { float4 pos : SV_Position; };
VSOut main(float3 p : POSITION) { VSOut o; o.pos = float4(Warp(p), 1) * _Noise.Sample(s, p.xy); return o; }
'''
    segs = segment_file(src, "hlsl", path="t.hlsl")
    assert [i for seg in segs for i in seg["refs"]["imports"]] == ["Packages/Core/Common.hlsl"]
    refs = {"calls": [c for seg in segs for c in seg["refs"]["calls"]]}
    assert _call_names(refs) == {"float4", "Warp", "Sample"}
    assert _call_entry(refs, "Sample") == {"name": "Sample", "receiver": "_Noise", "arity": 2}


def test_extract_refs_unsupported_language_is_empty():
    """A language with no refs_spec falls back to mentions only."""
    src = b"local function helper() end\nlocal function main() helper() end\n"
    for seg in segment_file(src, "lua", path="t.lua"):
        assert (seg["refs"]["calls"], seg["refs"]["imports"], seg["refs"]["inherits"]) == ([], [], [])


def test_extract_refs_capped_at_max_names():
    """A chunk that calls hundreds of distinct names doesn't blow up the list."""
    calls = "\n".join(f"    fn_{i}();" for i in range(300))
    src = f"void driver() {{\n{calls}\n}}\n".encode()
    segs = segment_file(src, "cpp", path="t.cpp")
    refs = _refs(segs, "driver")
    from chonks.index.refs_extract import _REFS_MAX_NAMES
    assert len(refs["calls"]) <= _REFS_MAX_NAMES


def test_extract_refs_cap_is_on_distinct_names_not_raw_entries():
    """The "calls" cap must be on distinct call NAMES, not raw entry count.
    If it counted raw entries, 'hot' fanned out across 20 receivers would burn
    20 slots and push 20 of the 199 'other_*' names out entirely."""
    from chonks.index.refs_extract import _REFS_MAX_NAMES, _REFS_MAX_CALL_VARIANTS_PER_NAME
    hot_calls = "\n".join(f"    r{i}.hot();" for i in range(20))
    other_calls = "\n".join(f"    other_{i}();" for i in range(199))
    src = f"void driver() {{\n{hot_calls}\n{other_calls}\n}}\n".encode()
    segs = segment_file(src, "cpp", path="t.cpp")
    refs = _refs(segs, "driver")
    names = _call_names(refs)
    assert len(names) == _REFS_MAX_NAMES  # 199 other_* + 1 hot == 200, none dropped
    assert all(f"other_{i}" in names for i in range(199))
    assert "hot" in names
    hot_entries = [e for e in refs["calls"] if isinstance(e, dict) and e["name"] == "hot"]
    assert 1 <= len(hot_entries) <= _REFS_MAX_CALL_VARIANTS_PER_NAME


# ---------------------------------------------------------------------------
# Call-site fingerprint: receiver/arity
# ---------------------------------------------------------------------------
# Chain rule: the receiver is the IMMEDIATE token adjacent to the called
# name, 'a.b.c.d()' -> 'c', not 'a' or 'a.b.c'. A definer's own qualifier
# splits the same way (chonks.index.graph.call_resolve._definer_qualifiers).

def test_call_receiver_is_immediate_not_root_cpp():
    src = b'''
void driver() {
    a.b.c.d(1);
    obj->w->x->y->z();
}
'''
    segs = segment_file(src, "cpp", path="t.cpp")
    refs = _refs(segs, "driver")
    assert _call_entry(refs, "d") == {
        "name": "d", "receiver": "c", "arity": 1, "racc": ".",
        "rhead": {"name": "a", "via": "unknown", "type": None}, "rpath": ["b", "c"],
    }
    assert _call_entry(refs, "z") == {
        "name": "z", "receiver": "y", "arity": 0, "racc": "->",
        "rhead": {"name": "obj", "via": "unknown", "type": None}, "rpath": ["w", "x", "y"],
    }


def test_call_receiver_is_immediate_not_root_python():
    src = b'''
def driver():
    a.b.c.d(1)
'''
    segs = segment_file(src, "python", path="t.py")
    refs = _refs(segs, "driver")
    assert _call_entry(refs, "d") == {"name": "d", "receiver": "c", "arity": 1}


def test_call_receiver_is_immediate_not_root_gdscript():
    src = b'''
func driver():
    a.b.c.d(1)
'''
    segs = segment_file(src, "gdscript", path="t.gd")
    refs = _refs(segs, "driver")
    assert _call_entry(refs, "d") == {"name": "d", "receiver": "c", "arity": 1}


def test_call_receiver_cpp_scoped_qualified_chain():
    """cpp's qualified_identifier (Ns::Cls::method) is right-recursive, unlike
    the left-recursive field_expression the other chain tests exercise, and
    gets its own walk (_cpp_qualified_immediate_receiver)."""
    src = b'''
void driver() {
    A::B::C::method(1, 2);
}
'''
    segs = segment_file(src, "cpp", path="t.cpp")
    refs = _refs(segs, "driver")
    assert _call_entry(refs, "method") == {
        "name": "method", "receiver": "C", "arity": 2, "racc": "::",
        "rhead": {"name": "A::B::C", "via": "type", "type": "A::B::C"},
    }


def test_templated_call_names_the_function_not_its_type_argument():
    cpp = b'''
void driver(Node *p) {
    Object::cast_to<GraphFrame>(p);
    make_ref<Texture>(1);
}
'''
    refs = _refs(segment_file(cpp, "cpp", path="t.cpp"), "driver")
    assert _call_names(refs) >= {"cast_to", "make_ref"}
    assert not _call_names(refs) & {"GraphFrame", "Texture"}
    cs = b"class A { void Driver() { var r = GetComponent<Rigidbody>(); Foo.Bar<Baz>(1); } }\n"
    refs = segment_file(cs, "c_sharp", path="t.cs")[0]["refs"]
    assert _call_names(refs) >= {"GetComponent", "Bar"}
    assert not _call_names(refs) & {"Rigidbody", "Baz"}


def test_call_arity_varargs_and_defaults_wildcard_python():
    src = b'''
def driver():
    foo(1, 2, *args)
    bar(1, 2, **kw)
    baz(1, key=2)
    qux()
'''
    segs = segment_file(src, "python", path="t.py")
    refs = _refs(segs, "driver")
    assert _call_entry(refs, "foo")["arity"] is None, "*args at the call site -> wildcard"
    assert _call_entry(refs, "bar")["arity"] is None, "**kw at the call site -> wildcard"
    # a keyword argument makes the positional count a LOWER BOUND, not the
    # call's true total (it may satisfy a required param the count doesn't
    # reflect) -> wildcard, same as *args/**kw.
    assert _call_entry(refs, "baz")["arity"] is None, "keyword arg present -> wildcard"
    assert _call_entry(refs, "qux")["arity"] == 0


def test_call_arity_cpp_pack_expansion_wildcard():
    src = b'''
void driver() {
    bar(1, 2, args...);
}
'''
    segs = segment_file(src, "cpp", path="t.cpp")
    refs = _refs(segs, "driver")
    assert _call_entry(refs, "bar")["arity"] is None


# A comment token between two real arguments is a NAMED node in every
# LANG_REFS_SPECS grammar's argument list; _call_arity used to count it as
# a real argument, inflating arity (see _TRIVIA_CALL_ARG_TYPES).

def test_call_arity_skips_comment_python():
    src = b'''
def driver():
    f(1,  # note
      2)
'''
    segs = segment_file(src, "python", path="t.py")
    refs = _refs(segs, "driver")
    assert _call_entry(refs, "f")["arity"] == 2


def test_call_arity_skips_comment_cpp():
    src = b'''
void driver() {
    f(1, /*c*/ 2);
}
'''
    segs = segment_file(src, "cpp", path="t.cpp")
    refs = _refs(segs, "driver")
    assert _call_entry(refs, "f")["arity"] == 2


def test_call_arity_skips_comment_c():
    src = b'''
void driver() {
    f(1, /*c*/ 2);
}
'''
    segs = segment_file(src, "c", path="t.c")
    refs = _refs(segs, "driver")
    assert _call_entry(refs, "f")["arity"] == 2


def test_call_arity_skips_comment_c_sharp():
    src = b'''
class B {
    void Driver() {
        Foo(1, /*c*/ 2);
    }
}
'''
    segs = segment_file(src, "c_sharp", path="t.cs")
    refs = _refs(segs, "B")
    assert _call_entry(refs, "Foo")["arity"] == 2


def test_call_arity_skips_comment_gdscript():
    src = b'''
func driver():
    f(1,  # note
      2)
'''
    segs = segment_file(src, "gdscript", path="t.gd")
    refs = _refs(segs, "driver")
    assert _call_entry(refs, "f")["arity"] == 2


def test_definer_param_arity_python():
    from chonks.index.graph.call_resolve import _definer_param_arity
    # self isn't counted ("python self accounting").
    assert _definer_param_arity(
        "def f(self, a, b, *args, **kw):\n    pass", "f", "python") == (2, 2, True)
    assert _definer_param_arity("def g():\n    pass", "g", "python") == (0, 0, False)
    # a default widens the compatible range without raising the minimum.
    assert _definer_param_arity(
        "def h(self, a, b=1):\n    pass", "h", "python") == (1, 2, False)
    # no recognizable signature -> "missing info", never a wrong count.
    assert _definer_param_arity("class Foo:\n    pass", "Foo", "python") is None
    assert _definer_param_arity("", "f", "python") is None


def test_definer_param_arity_cpp_and_c_sharp():
    from chonks.index.graph.call_resolve import _definer_param_arity
    assert _definer_param_arity("void f(int a, int b, ...) {}", "f", "cpp") == (2, 2, True)
    assert _definer_param_arity("void h(int a=1, int b=2) {}", "h", "cpp") == (0, 2, False)
    assert _definer_param_arity(
        "void f(int a, int b, params int[] c) {}", "f", "c_sharp") == (2, 2, True)


def test_definer_param_arity_c_void_is_zero_params():
    """C/cpp's explicit zero-args spelling 'int f(void)' is not a required
    parameter literally named 'void'; it must parse the same as 'int f()'."""
    from chonks.index.graph.call_resolve import _definer_param_arity
    assert _definer_param_arity("int f(void) {}", "f", "c") == (0, 0, False)
    assert _definer_param_arity("int f() {}", "f", "c") == (0, 0, False)


def test_definer_param_arity_python_self_annotated_and_dunder():
    """Python self-accounting must also strip an annotated 'self: Foo' and
    the positional-only '__self' spelling stdlib .pyi stubs use, not just
    the bare 'self'/'cls' literal."""
    from chonks.index.graph.call_resolve import _definer_param_arity
    assert _definer_param_arity(
        "def m(self: Foo, a):\n    pass", "m", "python") == (1, 1, False)
    assert _definer_param_arity(
        "def log(__self, __level, __message):\n    pass", "log", "python") == (2, 2, False)


def test_definer_param_arity_python_multiple_overloads_widen_not_gate():
    """A chunk holding several @overload stubs above the real implementation
    must not let the FIRST stub's arity gate every call; the combined range
    is widened across every same-named signature found."""
    from chonks.index.graph.call_resolve import _definer_param_arity
    content = (
        "@t.overload\n"
        "def f(self, a):\n    ...\n"
        "@t.overload\n"
        "def f(self, a, b):\n    ...\n"
        "def f(self, a, b=None):\n    return a\n"
    )
    # arity 1 (only the first overload) and arity 2 (the second/third) must
    # both be accepted -- neither is excluded by anchoring on a single stub.
    assert _definer_param_arity(content, "f", "python") == (1, 2, False)


def test_definer_param_arity_python_positional_only_marker_not_counted():
    """The bare '/' positional-only-parameter divider (PEP 570) is a syntax
    marker, not a parameter; it must not inflate min/max the way a real
    required parameter would."""
    from chonks.index.graph.call_resolve import _definer_param_arity
    assert _definer_param_arity(
        "def f(self, a, /, b):\n    pass", "f", "python") == (2, 2, False)


def test_find_signature_param_texts_unbalanced_match_is_skipped_not_scan_ending():
    """One match whose parens never balance (a truncated/oversize-split
    fragment) must not abort the whole scan; a later, complete same-named
    signature in the same content is still found and parsed."""
    from chonks.index.graph.call_resolve import _definer_param_arity, _find_signature_param_texts
    content = "def f(a, b\n\ndef f(c):\n    pass"
    assert _find_signature_param_texts(content, "f", "python") == ["c"]
    assert _definer_param_arity(content, "f", "python") == (1, 1, False)


def test_calls_entries_are_dicts_not_backward_compat_strings():
    """Fresh extraction always produces the fingerprint dict shape. Backward
    compat for legacy plain-string entries is a repomap-side READ concern
    (see chonks.index.graph.call_resolve._call_entry_fields), not something the extractor re-creates."""
    src = b"def driver():\n    helper()\n"
    segs = segment_file(src, "python", path="t.py")
    refs = _refs(segs, "driver")
    assert refs["calls"] == [{"name": "helper", "receiver": None, "arity": 0}]


# ---------------------------------------------------------------------------
# Error-recovery boundary salvage
# ---------------------------------------------------------------------------
# tree-sitter's error recovery can glue a malformed region, including a
# clean neighbouring construct, into one boundary node that was returned
# as-is. These repros force that glue with a genuinely unbalanced delimiter.

def test_fx_repro_yields_multiple_chunks_not_one():
    """Repro: a small .fx file used to collapse into ONE chunk because a
    missing closing brace glued the technique10 block into the function's
    boundary node, which was returned whole without recursion."""
    src = b"""cbuffer PerFrame : register(b0)
{
    float4x4 View;
    float4x4 Projection;
    float4x4 ViewProjection;
    float4x4 InverseView;
    float4x4 InverseProjection;
    float3   CameraPosition;
    float3   CameraDirection;
    float3   SunDirection;
    float3   SunColor;
    float2   ScreenSize;
    float2   InvScreenSize;
    float    TimeSeconds;
    float    DeltaTime;
}

struct VSInput
{
    float3 Position       : POSITION;
    float3 Normal         : NORMAL;
    float3 Tangent        : TANGENT;
    float3 Bitangent      : BINORMAL;
    float2 TexCoord0      : TEXCOORD0;
    float2 TexCoord1      : TEXCOORD1;
    float4 Color          : COLOR0;
    float4 BoneWeights    : BLENDWEIGHT;
    uint4  BoneIndices    : BLENDINDICES;
};

float4 MainPS(VSInput input) : SV_Target
{
    float3 n = normalize(input.Normal);
    float3 lightDir = normalize(float3(0.3, 0.7, 0.2));
    float diffuse = max(dot(n, lightDir), 0.0);
    float3 albedo = input.Color.rgb;
    float3 litColor = albedo * diffuse * SunColor;
    float rim = pow(1.0 - saturate(dot(n, CameraDirection)), 4.0);
    litColor += rim * 0.15;
    return float4(litColor, input.Color.a);

technique10 RenderTech
{
    pass P0
    {
        SetVertexShader(CompileShader(vs_4_0, MainVS()));
        SetPixelShader(CompileShader(ps_4_0, MainPS()));
        SetRasterizerState(DefaultRS);
        SetDepthStencilState(DefaultDSS, 0);
        SetBlendState(NoBlend, float4(0,0,0,0), 0xFFFFFFFF);
    }
}
"""
    counters: dict = {}
    segs = segment_file(src, "hlsl", path="shader.fx", counters=counters)
    types = {s["chunk_type"] for s in segs}

    assert len(segs) > 1, f"still glued into one chunk: {segs}"
    assert "cbuffer" in types, f"cbuffer boundary lost: {types}"
    assert "struct_specifier" in types, f"struct boundary lost: {types}"
    assert "function_definition" in types, f"function boundary lost: {types}"
    assert counters.get("error_salvaged") == 1
    assert counters.get("error_salvaged_nodes", 0) >= 1

    # MainPS is the real, wanted construct and must keep its own name and
    # type. Salvage must not drop the errored container in favour of what's
    # glued inside it, that would rename/anonymize MainPS.
    main_ps = [s for s in segs if s["name"] == "MainPS"]
    assert main_ps, f"MainPS lost its name entirely: {segs}"
    assert main_ps[0]["chunk_type"] == "function_definition"
    assert "normalize(input.Normal)" in main_ps[0]["content"], \
        "MainPS chunk doesn't actually contain MainPS's own body"


def test_camelcase_block_macro_does_not_swallow_neighbor_function():
    """A camelCase macro with a statement-block argument is invalid C++ and
    unhealable (self-heal only auto-discovers ALL_CAPS names). When error
    recovery glues it to a neighbouring function, that function must still
    surface as its own intact, correctly-named chunk."""
    src = b"""namespace engine {

void firstFunction(int a, int b, int c) {
    int total = a + b + c;
    total *= 2;
    total -= a;
    total += computeAdjustment(a, b, c);
    total = clampValue(total, 0, 1000);
    doSomething(total);
    logValue("first", total);
    notifyListeners(total);
    recordMetric("first.total", total);

someMacro(Name, int x; void f();)

void secondFunction(int b, int c, int d) {
    int total = b + c + d;
    total *= 3;
    total -= b;
    total += computeAdjustment(b, c, d);
    total = clampValue(total, 0, 1000);
    doSomethingElse(total);
    logValue("second", total);
    notifyListeners(total);
    recordMetric("second.total", total);
}

}
"""
    counters: dict = {}
    segs = segment_file(src, "cpp", path="engine.cpp", counters=counters)

    assert len(segs) > 1, f"still glued into one chunk: {segs}"
    second = [s for s in segs if s["name"] == "secondFunction"]
    assert second, f"secondFunction never surfaced as its own chunk: {segs}"
    assert second[0]["chunk_type"] == "function_definition"
    joined = "\n".join(s["content"] for s in segs)
    assert "doSomething(total)" in joined, "firstFunction's body was dropped"
    assert "doSomethingElse(total)" in joined, "secondFunction's body was dropped"
    assert counters.get("error_salvaged") == 1


def test_class_with_one_broken_method_still_chunks_as_one_class_churn_guard():
    """Churn guard: class/struct/interface boundaries are excluded from
    salvage-recursion (see _is_salvage_eligible) since members are always
    there, error or no error. One unparseable method must not explode the
    class into per-member chunks, and error_salvaged must not fire."""
    src = b"""class Widget {
public:
    Widget() : value_(0) {}

    int compute(int x) {
        int y = x + 1
        return y * value_;
    }

    void reset() {
        value_ = 0;
    }

    int value() const { return value_; }

private:
    int value_;
};
"""
    counters: dict = {}
    segs = segment_file(src, "cpp", path="widget.cpp", counters=counters)

    assert len(segs) == 1, f"class exploded into per-member chunks (churn): {segs}"
    assert segs[0]["chunk_type"] == "class_specifier"
    assert segs[0]["name"] == "Widget"
    assert "error_salvaged" not in counters


def test_legal_nested_function_keeps_outer_name_python():
    """Salvage regression: the premise that a statement body can never nest a
    function/class is false for Python/JS/TS nested functions. Salvage must
    not replace an errored outer function with a clean inner one nested inside."""
    src = b"""def outer(a,b):
    x = a+b
    def inner(c):
        return c*2
    y=inner(x
    return y
"""
    counters: dict = {}
    segs = segment_file(src, "python", counters=counters)

    outer = [s for s in segs if s["name"] == "outer"]
    inner = [s for s in segs if s["name"] == "inner"]
    assert outer, f"outer's name was dropped/overwritten: {segs}"
    assert inner, f"inner never surfaced as its own chunk: {segs}"
    assert outer[0]["chunk_type"] == "function_definition"
    assert "x = a+b" in outer[0]["content"]
    assert "return y" in outer[0]["content"]
    assert counters.get("error_salvaged") == 1


def test_legal_local_struct_keeps_outer_function_name_cpp():
    """Same regression family as above, C++ flavour: a local struct declared
    inside a function is legal nesting, not evidence the function's own name
    is bogus."""
    src = b"""void configure(int n){
    struct LocalConfig{
        int a;
        int b;
    };
    total=total+n
}
"""
    counters: dict = {}
    segs = segment_file(src, "cpp", counters=counters)

    outer = [s for s in segs if s["name"] == "configure"]
    inner = [s for s in segs if s["name"] == "LocalConfig"]
    assert outer, f"configure's name was dropped/overwritten: {segs}"
    assert inner, f"LocalConfig never surfaced as its own chunk: {segs}"
    assert outer[0]["chunk_type"] == "function_definition"
    assert inner[0]["chunk_type"] == "struct_specifier"
    assert counters.get("error_salvaged") == 1


def test_error_salvage_counters_reported():
    """Counter test: segment_file's `counters` dict reports both the per-file
    flag and a total salvaged-node count, mirroring macro_healed's pattern."""
    src = b"""cbuffer PerFrame : register(b0)
{
    float4x4 View;
}

float4 MainPS() : SV_Target
{
    return float4(1,1,1,1);

technique10 RenderTech
{
    pass P0
    {
        SetVertexShader(CompileShader(vs_4_0, MainVS()));
    }
}
"""
    counters: dict = {}
    segment_file(src, "hlsl", path="t.fx", counters=counters)
    assert counters.get("error_salvaged") == 1
    assert counters.get("error_salvaged_nodes", 0) >= 1


def test_error_salvage_counter_absent_on_clean_file():
    counters: dict = {}
    segment_file(b"int add(int a, int b) {\n    return a + b;\n}\n", "cpp", counters=counters)
    assert "error_salvaged" not in counters
    assert "error_salvaged_nodes" not in counters


def test_salvaged_oversize_container_does_not_duplicate_nested_boundaries():
    """Salvage dedup: an oversize errored container pulls clean nested
    boundaries out as top-level entries, but the oversize-split path
    re-walks the same container and can re-emit them as duplicate chunk ids."""
    padding = "\n".join(
        f"    total += computeStep(a, b, c, {i}) * {i};" for i in range(150)
    )
    src = f"""namespace engine {{

void firstFunction(int a, int b, int c) {{
    int total = a + b + c;
    total *= 2;
    total -= a;
{padding}
    total += computeAdjustment(a, b, c);

someMacro(Name, int x; void f();)

void secondFunction(int b, int c, int d) {{
    int total = b + c + d;
    total *= 3;
    doSomethingElse(total);
}}

void thirdFunction(int b, int c, int d) {{
    int total = b - c - d;
    total *= 5;
    doSomethingThird(total);
}}

}}
""".encode()
    assert len(src) > CHUNK_MAX, "test setup: firstFunction's glued span must exceed CHUNK_MAX"

    counters: dict = {}
    segs = segment_file(src, "cpp", path="engine.cpp", counters=counters)
    assert counters.get("error_salvaged") == 1, \
        "salvage never fired — test setup didn't reproduce the glue"

    joined = "\n".join(s["content"] for s in segs)
    assert "doSomethingElse" in joined, "secondFunction's body was dropped"
    assert "doSomethingThird" in joined, "thirdFunction's body was dropped"

    seen: dict[tuple[int, str], int] = {}
    for s in segs:
        key = (s["start_line"], s["content"])
        seen[key] = seen.get(key, 0) + 1
    dupes = [k for k, v in seen.items() if v > 1]
    assert not dupes, f"duplicate (start_line, content) segments: {[d[0] for d in dupes]}"

    ids = [_chunk_id("engine.cpp", s["start_line"], s["content"]) for s in segs]
    assert len(ids) == len(set(ids)), "duplicate chunk ids from within-file boundary collision"


# ---------------------------------------------------------------------------
# C language support
# ---------------------------------------------------------------------------

def _c_boundaries(src: bytes):
    """Low-level boundary walk (bypasses merge/split packing) so structural
    detection can be asserted independent of size-driven chunk packing."""
    from chonks.index.segment import _collect_boundaries, _extract_name
    from tree_sitter_language_pack import get_parser
    root = get_parser("c").parse(src).root_node
    return [(n.type, _extract_name(n, "c", src)) for n in _collect_boundaries(root, "c", src)]


def test_c_boundary_kinds_and_names():
    """Every construct C support asks for: function, struct, enum, union,
    typedef, static function, K&R-style parameter declarations."""
    src = b'''
struct Point { int x; int y; };

enum Color { RED, GREEN, BLUE };

union Value { int i; float f; };

typedef struct Point PointT;

static int helper(int a, int b) {
    return a + b;
}

int add(a, b)
    int a;
    int b;
{
    return a + b;
}
'''
    boundaries = _c_boundaries(src)
    assert ("struct_specifier", "Point") in boundaries
    assert ("enum_specifier", "Color") in boundaries
    assert ("union_specifier", "Value") in boundaries
    assert ("type_definition", "PointT") in boundaries
    assert ("function_definition", "helper") in boundaries
    assert ("function_definition", "add") in boundaries, \
        "K&R-style parameter declarations must still parse and name cleanly"


def test_c_anonymous_typedef_struct_gets_single_named_chunk():
    """The common C idiom `typedef struct { ... } Name;` must produce exactly
    ONE named boundary (the typedef alias), not a duplicate anonymous-struct
    boundary; the struct is nested inside the type_definition boundary."""
    src = b"typedef struct { int w; int h; } Size;\n"
    assert _c_boundaries(src) == [("type_definition", "Size")]


def test_c_function_pointer_typedef_name():
    """A function-pointer typedef's `declarator` field is wrapped in several
    layers (function_declarator -> parenthesized_declarator ->
    pointer_declarator); the alias must resolve to the leaf type_identifier."""
    src = b"typedef void (*FuncPtr)(int);\n"
    assert _c_boundaries(src) == [("type_definition", "FuncPtr")]


def test_c_bare_tag_reference_is_not_a_boundary():
    """A bare struct/union/enum type reference (forward decl, opaque pointer)
    must not become its own boundary, only a body-carrying definition does.
    C leans on this idiom far more than C++, so it needs explicit coverage."""
    src = b'''
struct Foo;
struct Foo *global_ptr;
extern struct Foo named_global;
'''
    boundaries = _c_boundaries(src)
    assert boundaries == [], f"bare tag references must not become boundaries: {boundaries}"


def test_c_bare_tag_reference_excluded_from_symbol_index():
    """Same guard, exercised through the decoupled symbol index (recurses INTO
    boundaries, unlike _collect_boundaries): a parameter/local variable using
    a named struct type must not spawn a spurious duplicate struct symbol."""
    from chonks.index.segment import _collect_symbols_from_root
    from tree_sitter_language_pack import get_parser
    src = b'''
struct Point { int x; int y; };
void takes_ptr(struct Point *p) {
    struct Point local;
    local.x = p->x;
}
'''
    root = get_parser("c").parse(src).root_node
    syms = _collect_symbols_from_root(root, "c", src)
    names_kinds = [(s["name"], s["kind"]) for s in syms]
    assert names_kinds.count(("Point", "struct_specifier")) == 1, \
        f"expected exactly one Point definition, got {names_kinds}"
    assert ("takes_ptr", "function_definition") in names_kinds


def test_extract_refs_c():
    src = b'''
#include <foo.h>
#include "bar.h"

int helper(int x);

int driver(int x) {
    int r = helper(x);
    return process(r);
}
'''
    segs = segment_file(src, "c", path="t.c")
    refs = _refs(segs, "driver")
    assert _call_names(refs) == {"helper", "process"}
    assert set(refs["imports"]) == {"foo.h", "bar.h"}
    assert refs["inherits"] == [], "C has no inheritance concept"
    assert _call_entry(refs, "helper") == {"name": "helper", "receiver": None, "arity": 1}
    assert _call_entry(refs, "process") == {"name": "process", "receiver": None, "arity": 1}


def test_parse_error_counter_set_on_macro_c():
    """With self-heal disabled, a leading annotation-style macro (the same
    shape as C++ engine macros, plain-C flavored) makes tree-sitter-c report a
    parse error; mirrors test_parse_error_counter_set_on_macro_cpp."""
    src = b'''
MY_EXPORT_ANNOTATION
int foo(void) {
    return 0;
}
'''
    counters: dict = {}
    segment_file(src, "c", path="a.c", counters=counters, self_heal=False)
    assert counters.get("parse_error") == 1


def test_parse_error_counter_absent_on_clean_c():
    counters: dict = {}
    segment_file(b"struct A { int x; };\nint f(void) { return 0; }\n", "c", counters=counters)
    assert "parse_error" not in counters


def test_macro_self_heal_recovers_function_name_c():
    """Macro self-heal (enabled for "c", fully generic, no cpp-specific
    machinery involved) recovers the function name after a leading
    annotation macro breaks the raw parse, same as cpp's UCLASS case."""
    src = b'''
MY_EXPORT_ANNOTATION
int foo(void) {
    return 0;
}
'''
    counters: dict = {}
    segs = segment_file(src, "c", path="a.c", counters=counters)
    names = {s["name"] for s in segs if s["name"]}
    assert "foo" in names, f"function name not recovered after heal: {names}"
    assert "parse_error" not in counters, "healed file should not count as parse_error"
    assert counters.get("macro_healed") == 1
    assert "MY_EXPORT_ANNOTATION" in counters.get("discovered_macros", set())


def test_collect_symbols_after_macro_heal_c_openssl_style():
    """Messy-C acceptance fixture: OpenSSL's DECLARE_ASN1_FUNCTIONS /
    IMPLEMENT_ASN1_FUNCTIONS macro idiom, which tree-sitter-c can't parse raw.
    Heal must recover the typedef, struct, and function under real names."""
    src = b'''
#include <stdio.h>

typedef struct st_MyType MyType;

DECLARE_ASN1_FUNCTIONS(MyType)

struct st_MyType {
    int id;
    char *name;
};

IMPLEMENT_ASN1_FUNCTIONS(MyType)

static int MyType_new(MyType **out) {
    *out = calloc(1, sizeof(MyType));
    return *out != NULL;
}
'''
    from tree_sitter_language_pack import get_parser
    assert get_parser("c").parse(src).root_node.has_error, \
        "test setup: the OpenSSL-style macro must break the raw parse"

    counters: dict = {}
    segment_file(src, "c", path="asn1.c", counters=counters)
    assert counters.get("macro_healed") == 1
    assert "parse_error" not in counters, "healed file should not count as parse_error"
    assert {"DECLARE_ASN1_FUNCTIONS", "IMPLEMENT_ASN1_FUNCTIONS"} <= counters.get("discovered_macros", set())

    sym_names_kinds = {(s["name"], s["kind"]) for s in counters["symbols"]}
    assert ("MyType", "type_definition") in sym_names_kinds
    assert ("st_MyType", "struct_specifier") in sym_names_kinds
    assert ("MyType_new", "function_definition") in sym_names_kinds


def test_cuda_and_stub_extensions_map_to_ast_languages():
    from chonks.index.segment import _lang_for_path
    from pathlib import Path
    assert _lang_for_path(Path("k.cu")) == "cpp"
    assert _lang_for_path(Path("k.cuh")) == "cpp"
    assert _lang_for_path(Path("m.pyi")) == "python"
    assert _lang_for_path(Path("k.mm")) == "cpp"
    assert _lang_for_path(Path("k.metal")) == "cpp"


def test_cuda_kernel_chunks_via_cpp_grammar():
    src = (
        b"__global__ void add_kernel(const float* a, const float* b, float* out, int n) {\n"
        b"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        b"    if (i < n) { out[i] = a[i] + b[i]; }\n"
        b"}\n\n"
        b"void launch_add(const float* a, const float* b, float* out, int n) {\n"
        b"    add_kernel<<<(n + 255) / 256, 256>>>(a, b, out, n);\n"
        b"}\n"
    )
    segs = segment_file(src, "cpp", path="kernels/add.cu")
    names = {s["name"] for s in segs if s.get("name")}
    assert "add_kernel" in names or "launch_add" in names


def test_lua_extension_maps_to_lua_grammar():
    from chonks.index.segment import _lang_for_path, CODE_LANGUAGES
    from pathlib import Path
    assert _lang_for_path(Path("init.lua")) == "lua"
    assert "lua" in CODE_LANGUAGES


def test_lua_module_chunks_functions_and_local_functions():
    """cocos2d-x-style module: a table of exported functions built via
    `function M.foo()`, a `local function` helper, and a bare top-level
    function, all real-world Lua idioms."""
    body = "\n".join(f"  print({i})" for i in range(60))
    src = f"""
local M = {{}}

function M.foo(x)
{body}
  return x + 1
end

local function bar(y)
{body}
  return y * 2
end

function baz()
{body}
end

return M
""".encode()

    counters: dict = {}
    segs = segment_file(src, "lua", path="init.lua", counters=counters)
    names = {s["name"] for s in segs if s.get("name")}
    assert {"M.foo", "bar", "baz"} <= names
    assert "parse_error" not in counters


def test_lua_table_of_methods_no_parse_error():
    """A table literal holding anonymous method functions (`Methods = { greet =
    function(...) end }`) is best-effort: no named chunk for the anonymous
    function, but it must not blow up the parse or explode into noise."""
    src = b"""
local Methods = {}

function Methods.greet(self, name)
  print("hi " .. name)
end

Methods.legacy = {
  ping = function(self)
    print("pong")
  end,
}

return Methods
"""
    counters: dict = {}
    segs = segment_file(src, "lua", path="methods.lua", counters=counters)
    names = {s["name"] for s in segs if s.get("name")}
    assert "Methods.greet" in names
    assert "parse_error" not in counters
    assert len(segs) < 10, f"table-of-methods chunking exploded: {len(segs)} chunks"


def test_chunks_tile_files_no_overlap_over_corpus():
    """Adjacent chunks of the same file must never share a line: consumers
    like eval/scip_precision.py's ChunkMapper assume chunks tile the file.
    Runs the real chunker over chonks/**/*.py as a live corpus sample."""
    root = Path(__file__).resolve().parent.parent / "chonks"
    py_files = sorted(root.glob("**/*.py"))
    assert len(py_files) >= 40, f"corpus sample too small: {len(py_files)} files"

    overlaps = []
    for path in py_files:
        src = path.read_bytes()
        segs = sorted(
            segment_file(src, "python", path=str(path)),
            key=lambda s: (s["start_line"], s["end_line"]),
        )
        for a, b in zip(segs, segs[1:]):
            if a["end_line"] >= b["start_line"]:
                overlaps.append((
                    path.name,
                    (a["name"], a["start_line"], a["end_line"]),
                    (b["name"], b["start_line"], b["end_line"]),
                ))
    assert not overlaps, f"overlapping adjacent chunks: {overlaps}"


def test_c_sharp_operators_are_named():
    from chonks.index.segment import _collect_symbols_from_root
    from tree_sitter_language_pack import get_parser
    src = b'''
struct V {
  public static V operator +(V a, V b) { return a; }
  public static bool operator ==(V a, V b) => true;
  public static V operator checked -(V a) => a;
  public static implicit operator int(V v) => 0;
  public static explicit operator Foo.Bar(V v) => null;
}
'''
    root = get_parser("csharp").parse(src).root_node
    names = {s["name"] for s in _collect_symbols_from_root(root, "c_sharp", src)}
    assert names == {"V", "operator+", "operator==", "operator-",
                     "operator int", "operator Foo.Bar"}, names


@pytest.mark.parametrize("src,name", [
    (b"namespace render {\nint draw_count();\nextern int frame_index;\n}\n", "render"),
    (b"namespace render::detail {\nconstexpr int kMax = 4;\n}\n", "render::detail"),
    (b"namespace a {\nnamespace b {\nint x;\n}\n}\n", "b"),
    (b"namespace {\nint hidden;\n}\n", None),
    (b'extern "C" {\nint c_api();\n}\n', None),
], ids=["plain", "qualified", "nested", "anonymous", "extern-c"])
def test_cpp_namespace_body_chunk_takes_the_namespace_name(src, name):
    chunks = segment_file(src, "cpp")
    assert [(c["chunk_type"], c["name"]) for c in chunks] == [("declaration_list", name)]


@pytest.mark.parametrize("src,symbols", [
    (b"class Node;\n", [("forward_declaration", "Node")]),
    (b"struct Opaque;\n", [("forward_declaration", "Opaque")]),
    (b"struct Foo *make_foo();\n", []),
    (b"struct stat st;\n", []),
    (b"typedef struct Foo Foo;\n", [("forward_declaration", "Foo")]),
    (b"typedef struct _FcConfig FcConfig;\n", [("forward_declaration", "FcConfig")]),
    (b"typedef int Id;\n", []),
    (b"template <class T> class Vec;\n", [("forward_declaration", "Vec")]),
    (b"class A {\n  friend class B;\n  class Inner;\n};\n",
     [("class_specifier", "A"), ("forward_declaration", "Inner")]),
    (b"class GODOT_API Node;\n", []),
    (b"class Node {\n  int x;\n};\n", [("class_specifier", "Node")]),
    (b"struct P { int x; } p;\n", [("struct_specifier", "P")]),
    (b"template <class T> class Vec {\n  T* data;\n};\n",
     [("template_declaration", "Vec"), ("class_specifier", "Vec")]),
], ids=["fwd-class", "fwd-struct", "elaborated-return", "elaborated-var", "opaque-typedef",
        "opaque-typedef-alias", "plain-typedef", "fwd-template", "nested-fwd-not-friend", "unhealed-macro-fwd", "class",
        "struct-with-declarator", "template-class"])
def test_cpp_only_a_class_with_a_body_is_a_definition(src, symbols):
    from chonks.index.segment import _collect_symbols_from_root
    from tree_sitter_language_pack import get_parser
    root = get_parser("cpp").parse(src).root_node
    got = [(s["kind"], s["name"]) for s in _collect_symbols_from_root(root, "cpp", src)]
    assert sorted(got) == sorted(symbols)


@pytest.mark.parametrize("src,symbols", [
    (b"struct wl_surface;\n", [("forward_declaration", "wl_surface")]),
    (b"struct stat st;\n", []),
    (b"void f(struct Foo *p);\n", []),
    (b"struct Point { int x; };\n", [("struct_specifier", "Point")]),
], ids=["fwd-struct", "elaborated-var", "elaborated-param", "struct"])
def test_c_forward_declaration_is_a_fallback_symbol(src, symbols):
    from chonks.index.segment import _collect_symbols_from_root
    from tree_sitter_language_pack import get_parser
    root = get_parser("c").parse(src).root_node
    assert [(s["kind"], s["name"]) for s in _collect_symbols_from_root(root, "c", src)] == symbols


@pytest.mark.parametrize("src,symbols", [
    (b"struct VSOut;\n", [("forward_declaration", "VSOut")]),
    (b"struct VSOut make();\n", []),
    (b"void f(struct VSOut v);\n", []),
    (b"struct VSOut { float4 pos : SV_Position; };\n", [("struct_specifier", "VSOut")]),
], ids=["fwd-struct", "elaborated-return", "elaborated-param", "struct"])
def test_hlsl_only_a_struct_with_a_body_is_a_definition(src, symbols):
    from chonks.index.segment import _collect_symbols_from_root
    from chonks.languages import REGISTRY
    from tree_sitter_language_pack import get_parser
    root = get_parser(REGISTRY.get("hlsl").grammar).parse(src).root_node
    assert [(s["kind"], s["name"]) for s in _collect_symbols_from_root(root, "hlsl", src)] == symbols


@pytest.mark.parametrize("lang,body", [
    ("cpp", b"    IF_FEATURE(Render) {\n        draw();\n    }\n"),
    ("cpp", b"    wl_array_for_each(state, states) {\n        use(state);\n    }\n"),
    ("cpp", b"    LOCK_SCOPE\n    if (ready) {\n        run();\n    }\n"),
    ("cpp", b"    LOCK_SCOPE\n    for (const auto &e : items) {\n        run(e);\n    }\n"),
    ("cpp", b"    LOCK_SCOPE\n    while (busy) {\n        wait();\n    }\n"),
    ("cpp", b"    LOCK_SCOPE\n    switch (mode) {\n    case 1: run(); break;\n    }\n"),
    ("c", b"    LOCK_SCOPE\n    if (ready) {\n        run();\n    }\n"),
], ids=["macro-block", "foreach-macro", "macro-before-if", "macro-before-range-for",
        "macro-before-while", "macro-before-switch", "c-macro-before-if"])
def test_c_family_statement_in_a_body_is_not_a_function(lang, body):
    from chonks.index.segment import _collect_symbols_from_root
    from tree_sitter_language_pack import get_parser
    src = b"void tick(int n) {\n" + body + b"}\n"
    root = get_parser(lang).parse(src).root_node
    got = [(s["kind"], s["name"]) for s in _collect_symbols_from_root(root, lang, src)]
    assert got == [("function_definition", "tick")]


@pytest.mark.parametrize("src,names", [
    (b"void outer() {\n  if (x) {\n}\nint helper(int a) { return a; }\nFoo::Foo() {}\nFoo::~Foo() {}\n",
     ["outer", "helper", "Foo::Foo", "Foo::~Foo"]),
    (b"void outer() {\n  struct Local {\n    Local() {}\n  };\n}\n", ["outer", "Local"]),
], ids=["unbalanced-braces", "local-class-constructor"])
def test_cpp_real_functions_inside_a_body_stay_functions(src, names):
    from chonks.index.segment import _collect_symbols_from_root
    from tree_sitter_language_pack import get_parser
    root = get_parser("cpp").parse(src).root_node
    got = [s["name"] for s in _collect_symbols_from_root(root, "cpp", src) if s["kind"] == "function_definition"]
    assert sorted(got) == sorted(names)


def test_cpp_macro_blocks_in_a_split_function_keep_the_function_name():
    blocks = "".join(
        f"    IF_FEATURE(Feature{i}) {{\n"
        + "".join(f"        step_{i}_{j}(state, {j});\n" for j in range(25))
        + "    }\n"
        for i in range(8))
    src = ("void Engine::UpdateAll(State &state) {\n" + blocks + "}\n").encode()
    chunks = segment_file(src, "cpp")
    assert len(chunks) > 1
    assert {c["name"] for c in chunks} == {"Engine::UpdateAll"}


def test_cpp_anonymous_struct_with_a_body_stays_a_boundary():
    chunks = segment_file(b"struct { int x; } s;\n", "cpp")
    assert [(c["chunk_type"], c["name"]) for c in chunks] == [("struct_specifier", None)]


def test_cpp_forward_declaration_does_not_name_the_chunk_around_it():
    src = b"namespace render {\nint draw_count();\nextern int frame_index;\nstruct Opaque;\n}\n"
    chunks = segment_file(src, "cpp")
    assert [(c["chunk_type"], c["name"]) for c in chunks] == [("declaration_list", "render")]


def test_c_sharp_namespace_chunk_is_named():
    src = b"namespace Game.Events\n{\n    public delegate void Hit(int damage);\n}\n"
    chunks = segment_file(src, "c_sharp")
    assert [(c["chunk_type"], c["name"]) for c in chunks] == [("namespace_declaration", "Game.Events")]


def test_cpp_names_look_through_pointer_reference_and_cast_declarators():
    from chonks.index.segment import _collect_symbols_from_root
    from tree_sitter_language_pack import get_parser
    src = b'''
struct V {
  operator int() const { return 0; }
  explicit operator bool() const noexcept { return true; }
  operator std::string() { return {}; }
  operator std::function<void(int)>() { return {}; }
  V& operator=(const V&) { return *this; }
  int* ptr() { return 0; }
  const V& ref() const { return *this; }
  V&& mv() { return static_cast<V&&>(*this); }
};
V::operator int*() const { return 0; }
int** V::pp() { return 0; }
int (*fnptr())(int) { return 0; }
int plain(int a) { return a; }
A::B::operator long() { return 0; }
'''
    root = get_parser("cpp").parse(src).root_node
    names = {s["name"] for s in _collect_symbols_from_root(root, "cpp", src)}
    assert names == {
        "V", "operator int", "operator bool", "operator std::string",
        "operator std::function<void(int)>", "operator=",
        "ptr", "ref", "mv", "V::operator int*", "V::pp", "fnptr", "plain",
        "A::B::operator long",
    }, names


def test_c_names_look_through_pointer_declarators():
    from chonks.index.segment import _collect_symbols_from_root
    from tree_sitter_language_pack import get_parser
    src = b"char *dup(const char *s) { return 0; }\nint plain(void) { return 0; }\n"
    root = get_parser("c").parse(src).root_node
    names = {s["name"] for s in _collect_symbols_from_root(root, "c", src)}
    assert names == {"dup", "plain"}, names
