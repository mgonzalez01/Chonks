import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tree_sitter_language_pack import get_parser

import chonks.languages as languages
from chonks.languages import _naming
from chonks.languages.spec import NOT_HANDLED

_GOLDEN_DIR = Path(__file__).parent / "data" / "golden"
_GRAMMAR = dict(languages.table("grammar"))
_LITERAL_QUOTES = ('"""', "'''", '"', "'", "`")


def _load_f1_chunking():
    # Same loader as tests/test_languages_registry.py, pinned to the F1 tag.
    proc = subprocess.run(
        ["git", "show", "F1:chonks/chunking.py"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        pytest.skip("F1 tag unavailable (detached/shallow checkout)")
    spec = importlib.util.spec_from_loader("chunking_f1_naming_reference", loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chunking_f1_naming_reference"] = mod
    exec(compile(proc.stdout, "chunking_f1_naming_reference.py", "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def f1():
    return _load_f1_chunking()


def _load_cases():
    cases = []
    for path in sorted(_GOLDEN_DIR.glob("*.json")):
        cases.extend(json.loads(path.read_text()))
    return cases


_SUPPLEMENT = {
    "cpp": (
        "template<typename T> T identity(T x) { return x; }\n"
        "template<typename T> class Box { T value; };\n"
        "template<typename T> struct Pair { T a; T b; };\n"
        "union U { int a; float b; };\n"
        "enum E { A, B };\n"
    ),
    "c": (
        "union U { int a; float b; };\n"
        "typedef int (*fn_t)(int);\n"
    ),
    "c_sharp": (
        "class Foo {\n"
        "    Foo() {}\n"
        "    ~Foo() {}\n"
        "    int Value { get; set; }\n"
        "    public static Foo operator +(Foo a, Foo b) { return a; }\n"
        "    public static explicit operator int(Foo a) { return a; }\n"
        "}\n"
        "enum Color { Red, Green }\n"
    ),
    "gdscript": "class Inner:\n    var x\n",
    "typescript": (
        "namespace NS { export const x = 1; }\n"
        "module M { export const y = 1; }\n"
        "var f = () => 1;\n"
    ),
    "tsx": (
        "class Base {}\n"
        "abstract class Shape {\n"
        "    area(): number { return 0; }\n"
        "}\n"
        "interface I {}\n"
        "enum E { A, B }\n"
        "type T = number;\n"
        "namespace NS { export const x = 1; }\n"
        "module M { export const y = 1; }\n"
        "var f = () => 1;\n"
        "function* gen() {}\n"
    ),
    "python": "def g(a, b=1):\n    return a + b\n\ng(1)\n",
}

_CASES_BY_LANG: dict[str, list[dict]] = {}
for _case in _load_cases():
    _CASES_BY_LANG.setdefault(_case["language"], []).append(_case)
for _lang, _source in _SUPPLEMENT.items():
    _CASES_BY_LANG.setdefault(_lang, []).append(
        {"id": "supplement", "language": _lang, "source": _source})

_LANGS = sorted(_CASES_BY_LANG)


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _iter_nodes(lang):
    parser = get_parser(_GRAMMAR[lang])
    for case in _CASES_BY_LANG.get(lang, []):
        src = case["source"].encode("utf-8")
        tree = parser.parse(src)
        for node in _walk(tree.root_node):
            yield case["id"], node, src


def _is_boundary_new(spec, node, src):
    if node.type in spec.boundary_nodes:
        if node.type in spec.boundary_filters:
            return spec.boundary_filters[node.type](node, src)
        return True
    if spec.extra_boundary is not None and spec.extra_boundary(node, src):
        return True
    return False


def _is_salvage_eligible_new(spec, node, src):
    if spec.salvage_extra is not None and spec.salvage_extra(node, src):
        return True
    return node.type in spec.salvage_nodes


def _classify_param_new(spec, part):
    if spec.classify_param is not None:
        result = spec.classify_param(part)
        if result is not NOT_HANDLED:
            return result
    return "default" if "=" in part else "required"


def _strip_implicit_params_new(spec, parts):
    if spec.strip_implicit_params is not None:
        return spec.strip_implicit_params(parts)
    return parts


def _is_empty_params_new(spec, parts):
    return len(parts) == 1 and parts[0] in spec.empty_param_spellings


def _param_corpus(f1, lang):
    corpus = []
    for case in _CASES_BY_LANG.get(lang, []):
        src = case["source"].encode("utf-8")
        segments = f1.segment_file(src, lang)
        names = sorted({
            (call.get("name") if isinstance(call, dict) else call)
            for seg in segments for call in seg["refs"].get("calls", [])
        } - {None})
        for seg in segments:
            content = seg["content"]
            for name in names:
                for params_text in f1._find_signature_param_texts(content, name, lang):
                    parts = f1._split_top_level_params(params_text)
                    if parts:
                        corpus.append((case["id"], parts))
    return corpus


def _has_recognised_quote_pair(rest):
    for q in _LITERAL_QUOTES:
        if rest.startswith(q) and rest.endswith(q) and len(rest) >= 2 * len(q):
            return True
    return False


@pytest.mark.parametrize("lang", _LANGS)
def test_extract_name(f1, lang):
    spec = languages.get(lang)
    count = 0
    mismatches = []
    for case_id, node, src in _iter_nodes(lang):
        count += 1
        old = f1._extract_name(node, lang, src)
        new = _naming.extract_name(spec, node, src)
        if old != new:
            mismatches.append((case_id, node.type, node.start_byte, old, new))
    assert count > 0
    assert not mismatches, mismatches


@pytest.mark.parametrize("lang", _LANGS)
def test_is_boundary(f1, lang):
    spec = languages.get(lang)
    count = 0
    mismatches = []
    for case_id, node, src in _iter_nodes(lang):
        count += 1
        old = f1._is_boundary(node, lang, src)
        new = _is_boundary_new(spec, node, src)
        if old != new:
            mismatches.append((case_id, node.type, node.start_byte, old, new))
    assert count > 0
    assert not mismatches, mismatches


@pytest.mark.parametrize("lang", _LANGS)
def test_salvage_eligible(f1, lang):
    spec = languages.get(lang)
    count = 0
    mismatches = []
    for case_id, node, src in _iter_nodes(lang):
        count += 1
        old = f1._is_salvage_eligible(node, lang, src)
        new = _is_salvage_eligible_new(spec, node, src)
        if old != new:
            mismatches.append((case_id, node.type, node.start_byte, old, new))
    assert count > 0
    assert not mismatches, mismatches


def test_synthetic_chunk_type(f1):
    spec = languages.get("hlsl")
    count = 0
    mismatches = []
    for case_id, node, src in _iter_nodes("hlsl"):
        count += 1
        old = "cbuffer" if f1._is_cbuffer(node, src) else node.type
        new = spec.synthetic_chunk_type(node, src) or node.type
        if old != new:
            mismatches.append((case_id, node.type, node.start_byte, old, new))
    assert count > 0
    assert not mismatches, mismatches


def test_module_residue_split(f1):
    spec = languages.get("python")
    count = 0
    mismatches = []
    for case_id, node, src in _iter_nodes("python"):
        count += 1
        old = f1._is_main_block(node, src)
        new = spec.module_residue_split(node, src)
        if old != new:
            mismatches.append((case_id, node.type, node.start_byte, old, new))
    assert count > 0
    assert not mismatches, mismatches


@pytest.mark.parametrize("lang", ("cpp", "gdscript"))
def test_call_receiver(f1, lang):
    spec = languages.get(lang)
    count = 0
    mismatches = []
    if lang == "cpp":
        for case_id, node, src in _iter_nodes(lang):
            if node.type != "call_expression":
                continue
            callee = node.child_by_field_name("function")
            if callee is None:
                continue
            count += 1
            new = spec.call_receiver(node, callee, src)
            if new is NOT_HANDLED:
                if callee.type == "qualified_identifier":
                    mismatches.append((case_id, node.type, node.start_byte,
                                        "should-be-handled", "NOT_HANDLED"))
                continue
            old = f1._call_receiver(callee, src)
            if old != new:
                mismatches.append((case_id, node.type, node.start_byte, old, new))
    else:
        for case_id, node, src in _iter_nodes(lang):
            if node.type not in ("call", "attribute_call"):
                continue
            count += 1
            old = f1._gdscript_call_receiver(node, src)
            new = spec.call_receiver(node, node, src)
            if old != new:
                mismatches.append((case_id, node.type, node.start_byte, old, new))
    assert count > 0
    assert not mismatches, mismatches


@pytest.mark.parametrize("lang", ("c", "c_sharp", "cpp", "python"))
def test_param_hooks(f1, lang):
    spec = languages.get(lang)
    corpus = _param_corpus(f1, lang)
    assert corpus
    mismatches = []
    for case_id, parts in corpus:
        for part in parts:
            old = f1._classify_param(part, lang)
            new = _classify_param_new(spec, part)
            if old != new:
                mismatches.append((case_id, "classify_param", 0, old, new))
        if lang == "python":
            old_parts = f1._strip_python_self(parts)
            new_parts = _strip_implicit_params_new(spec, parts)
            if old_parts != new_parts:
                mismatches.append((case_id, "strip_implicit_params", 0, old_parts, new_parts))
        if lang in ("cpp", "c"):
            old_empty = parts == ["void"]
            new_empty = _is_empty_params_new(spec, parts)
            if old_empty != new_empty:
                mismatches.append((case_id, "empty_param_spellings", 0, old_empty, new_empty))
    assert not mismatches, mismatches


@pytest.mark.parametrize("lang", _LANGS)
def test_literal_wrapper(f1, lang):
    spec = languages.get(lang)
    if lang != "lua":
        assert spec.literal_wrapper is None
    if lang != "python":
        assert spec.raw_string_prefix is None
    if lang not in ("lua", "python"):
        return
    leaf_types = spec.literals.leaf_types
    count = 0
    mismatches = []
    for case_id, node, src in _iter_nodes(lang):
        if not (node.is_named and node.type in leaf_types):
            continue
        raw = src[node.start_byte:node.end_byte].decode(errors="replace")
        if lang == "lua":
            count += 1
            old = f1._strip_literal_wrapper(raw, lang)
            new = spec.literal_wrapper(raw)
            if new is NOT_HANDLED:
                if f1._LUA_LONG_BRACKET_RE.match(raw):
                    mismatches.append((case_id, node.type, node.start_byte, old, "NOT_HANDLED"))
                continue
            if old != new:
                mismatches.append((case_id, node.type, node.start_byte, old, new))
        else:
            text, mode = f1._strip_literal_wrapper(raw, lang)
            if mode not in ("raw", "normal"):
                continue
            m = f1._LITERAL_PREFIX_RE.match(raw)
            idx = m.end() if m else 0
            prefix = raw[:idx]
            rest = raw[idx:]
            if not _has_recognised_quote_pair(rest):
                continue
            count += 1
            old = mode == "raw"
            new = spec.raw_string_prefix(prefix)
            if old != new:
                mismatches.append((case_id, node.type, node.start_byte, old, new))
    assert count > 0
    assert not mismatches, mismatches


def test_file_metadata():
    spec = languages.get("python")

    def old_rule(path):
        if path.endswith(".py"):
            return {"module": path[:-3].replace("/", ".")}
        return None

    for path in ("a/b.py", "a/b.pyi", "a/b.md"):
        assert spec.file_metadata(path) == old_rule(path)


@pytest.mark.parametrize("lang", _LANGS)
def test_block_comments(lang):
    spec = languages.get(lang)
    assert spec.block_comments == (spec.name != "python")
