from chonks.chunking import CHUNK_MAX, CHUNK_MIN, segment_file, segment_text_file

# Pushes each boundary comfortably over CHUNK_MIN so _merge_small doesn't fold
# adjacent tiny functions together, these tests are about boundary detection
# and naming, not merge behavior.
_PAD = "  // pad pad pad pad pad pad pad pad pad pad pad pad pad pad pad pad pad pad\n" * 6


def test_js_function_declaration_detected_and_named():
    src = f"export function add(a, b) {{\n{_PAD}return a + b;\n}}\n".encode()
    segs = segment_file(src, "javascript")
    assert len(segs) == 1
    assert segs[0]["chunk_type"] == "function_declaration"
    assert segs[0]["name"] == "add"


def test_js_const_arrow_function_named_after_variable():
    # No dedicated node type for `const NAME = (...) => ...`: the whole
    # lexical_declaration boundaries, name comes from the variable.
    src = f"export const double = (x) => {{\n{_PAD}return x * 2;\n}};\n".encode()
    segs = segment_file(src, "javascript")
    assert len(segs) == 1
    assert segs[0]["name"] == "double"


def test_js_var_function_expression_named_after_variable():
    src = f"var square = function(x) {{\n{_PAD}return x * x;\n}};\n".encode()
    segs = segment_file(src, "javascript")
    assert len(segs) == 1
    assert segs[0]["name"] == "square"


def test_js_exported_class_and_methods():
    src = (
        f"export class Widget {{\n"
        f"  constructor() {{\n{_PAD}  }}\n"
        f"  render() {{\n{_PAD}    return 1;\n  }}\n"
        f"}}\n"
    ).encode()
    segs = segment_file(src, "javascript")
    # Under CHUNK_MAX, so it folds into one chunk; methods only split out
    # individually when the class is oversized.
    assert len(segs) == 1
    assert segs[0]["chunk_type"] == "class_declaration"
    assert segs[0]["name"] == "Widget"


def test_js_generator_function_declaration_detected():
    src = f"export function* gen() {{\n{_PAD}yield 1;\n}}\n".encode()
    segs = segment_file(src, "javascript")
    assert len(segs) == 1
    assert segs[0]["chunk_type"] == "generator_function_declaration"
    assert segs[0]["name"] == "gen"


def test_ts_interface_declaration_detected_and_named():
    src = f"export interface Foo {{\n{_PAD}  bar: string;\n}}\n".encode()
    segs = segment_file(src, "typescript")
    assert len(segs) == 1
    assert segs[0]["chunk_type"] == "interface_declaration"
    assert segs[0]["name"] == "Foo"


def test_ts_enum_declaration_detected_and_named():
    src = f"export enum Color {{\n{_PAD}Red, Green, Blue\n}}\n".encode()
    segs = segment_file(src, "typescript")
    assert len(segs) == 1
    assert segs[0]["chunk_type"] == "enum_declaration"
    assert segs[0]["name"] == "Color"


def test_ts_type_alias_declaration_detected_and_named():
    # type_alias_declaration has no body to pad, so use a large union instead.
    variants = " | ".join(f'"variant_{i}"' for i in range(60))
    src = f"export type Alias = {variants};\n".encode()
    segs = segment_file(src, "typescript")
    assert len(segs) == 1
    assert segs[0]["chunk_type"] == "type_alias_declaration"
    assert segs[0]["name"] == "Alias"


def test_ts_abstract_class_declaration_detected_and_named():
    src = (
        f"export abstract class Base {{\n"
        f"  abstract doIt(): void;\n{_PAD}\n"
        f"}}\n"
    ).encode()
    segs = segment_file(src, "typescript")
    assert len(segs) == 1
    assert segs[0]["chunk_type"] == "abstract_class_declaration"
    assert segs[0]["name"] == "Base"


def test_ts_exported_class_still_detected():
    # export_statement is transparent (pass-through), not a boundary itself.
    src = f"export default class Other {{\n  m() {{\n{_PAD}  }}\n}}\n".encode()
    segs = segment_file(src, "typescript")
    assert len(segs) == 1
    assert segs[0]["chunk_type"] == "class_declaration"
    assert segs[0]["name"] == "Other"


def test_tsx_function_component_detected():
    src = (
        f"export function App() {{\n"
        f"{_PAD}return <div className=\"x\">Hi</div>;\n"
        f"}}\n"
    ).encode()
    segs = segment_file(src, "tsx")
    assert len(segs) == 1
    assert segs[0]["chunk_type"] == "function_declaration"
    assert segs[0]["name"] == "App"


def test_tsx_arrow_component_named_after_variable():
    src = (
        f"export const Comp = () => {{\n"
        f"{_PAD}return <span>hi</span>;\n"
        f"}};\n"
    ).encode()
    segs = segment_file(src, "tsx")
    assert len(segs) == 1
    assert segs[0]["name"] == "Comp"


# --------------------------------------------------------------------------
# Generic text fallback (segment_text_file)
# --------------------------------------------------------------------------

def test_fallback_html_produces_line_sliced_text_chunks():
    html = "<html>\n<body>\n" + ("  <p>hello</p>\n" * 5) + "</body>\n</html>\n"
    segs = segment_text_file(html.encode(), path="index.html")
    assert segs, "fallback produced no chunks"
    for s in segs:
        assert s["chunk_type"] == "text"
        assert s["name"] is None
    assert segs[0]["start_line"] == 1
    assert segs[-1]["end_line"] == len(html.splitlines())


def test_fallback_markdown_is_heading_aware():
    # See test_chunking_md.py for dedicated markdown coverage.
    md = "# Title\n\nSome body text.\n\n" + ("more text line\n" * 10)
    segs = segment_text_file(md.encode(), path="README.md")
    assert segs
    assert segs[0]["chunk_type"] == "text"
    assert segs[0]["name"] == "Title"
    assert segs[0]["start_line"] == 1


def test_fallback_enforces_chunk_max_ceiling():
    line = "x" * (CHUNK_MAX * 3)
    segs = segment_text_file(line.encode(), path="min.json")
    assert len(segs) > 1
    assert all(len(s["content"].encode("utf-8")) <= CHUNK_MAX for s in segs)


def test_fallback_splits_large_file_into_multiple_chunks():
    big = "line of filler content here\n" * 2000
    segs = segment_text_file(big.encode(), path="big.yaml")
    assert len(segs) > 1
    assert all(len(s["content"].encode("utf-8")) <= CHUNK_MAX for s in segs)
