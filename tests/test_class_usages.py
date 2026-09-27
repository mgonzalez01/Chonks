"""Graph queries on one class's method: `Owner::m` read through the class
model, callers checked against their recorded calls, bare names grouped by class."""
import hashlib
import json
from pathlib import Path

import chonks.serve.app as serve_app
import chonks.serve.models as serve_models
import chonks.serve.projects as serve_projects
from chonks.index.graph.refs import build_refs
from chonks.index.pipeline import index_paths
from chonks.retrieval.graph_queries import find_outgoing, find_symbol, find_usages, get_impact
from chonks.retrieval.trace import trace_path
from chonks.storage.store import Store


class _FakeEmbedder:
    model = "fake"
    url = "http://localhost:9999"

    def embed_documents(self, texts, client=None, **kw):
        return [[(b - 127.5) / 127.5 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]

    def embed_queries(self, texts, client=None, **kw):
        return self.embed_documents(texts)


def _index(tmp_path, files: dict[str, str]) -> Store:
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    store = Store(tmp_path / "test.db")
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    return store


def _names(result: dict) -> set[str]:
    return {r["name"] for r in result["results"]}


_VECTORS = {
    "vector.h": """template <class T>
class Vector {
    T *data = nullptr;
    int count = 0;

public:
    T *ptrw() { return data; }
    const T *ptr() const { return data; }
    int size() const { return count; }
    int capacity() const { return count; }
};
""",
    "string.h": "class String {\npublic:\n    int size() const;\n};\n",
    "string.cpp": '#include "string.h"\n\nint String::size() const { return 0; }\n',
    "fill.cpp": '#include "vector.h"\nvoid fill(Vector<int> &v) { v.ptrw(); }\n',
    "count.cpp": '#include "vector.h"\nint count(const Vector<int> &v) { return v.size(); }\n',
    "total.cpp": '#include "vector.h"\nint total(const Vector<int> &v) { return v.size() * 2; }\n',
    "length.cpp": '#include "string.h"\nint length(const String &s) { return s.size(); }\n',
    "peek.cpp": "void peek() { /* ptrw */ }\n",
}


def test_a_qualified_method_name_answers_for_that_class_alone(tmp_path):
    store = _index(tmp_path, _VECTORS)
    assert _names(find_usages(store, "Vector::size")) == {"count", "total"}
    assert _names(find_usages(store, "String::size")) == {"length"}
    assert get_impact(store, "Vector::size")["total_references"] == 2
    assert [(d["name"], d["start_line"]) for d in find_symbol(store, "Vector::size")["symbols"]] == [("size", 9)]
    assert trace_path(store, "count", "Vector::size")["found"]


def test_a_caller_of_another_method_in_the_same_chunk_is_not_a_user(tmp_path):
    store = _index(tmp_path, _VECTORS)
    chunks = {r["name"]: r["chunk_id"] for r in store.get_members_by_owners(["Vector"])}
    assert chunks["size"] == chunks["ptrw"]
    assert _names(find_usages(store, "Vector::ptrw")) == {"fill", "peek"}
    assert "fill" not in _names(find_usages(store, "Vector::size"))
    assert _names(find_usages(store, "size")) == {"count", "total", "length"}
    assert "ptrw" in find_outgoing(store, "Vector::size")["note"]


def test_a_method_a_base_defines_answers_through_the_derived_class(tmp_path):
    store = _index(tmp_path, {
        "shapes.h": "class Shape {\npublic:\n    void fill();\n};\nclass Circle : public Shape {};\n"
                    "class Label {\npublic:\n    void fill();\n};\n",
        "shape.cpp": '#include "shapes.h"\n\nvoid Shape::fill() {}\n',
        "label.cpp": '#include "shapes.h"\n\nvoid Label::fill() {}\n',
        "paint.cpp": '#include "shapes.h"\nvoid paint(Circle &c) { c.fill(); }\n',
        "print.cpp": '#include "shapes.h"\nvoid print(Label &l) { l.fill(); }\n',
    })
    result = find_usages(store, "Circle::fill")
    assert _names(result) == {"paint"}
    assert "Shape::fill" in result["note"]


def test_a_namespace_function_keeps_its_qualified_callers(tmp_path):
    store = _index(tmp_path, {
        "math.h": "namespace Math {\ndouble ease(double p_x);\n}\n",
        "math.cpp": '#include "math.h"\n\ndouble Math::ease(double p_x) { return p_x; }\n',
        "tween.cpp": '#include "math.h"\ndouble step(double t) { return Math::ease(t); }\n',
    })
    assert [r["name"] for r in find_usages(store, "Math::ease")["results"] if r["edge_type"] == "calls"] == ["step"]


def test_a_call_without_evidence_is_kept_and_marked_unverified(tmp_path):
    store = _index(tmp_path, _VECTORS)
    for cid, raw in store._conn.execute("SELECT id, metadata FROM chunks WHERE metadata IS NOT NULL").fetchall():
        md = json.loads(raw)
        md["calls"] = [{k: v for k, v in c.items() if k in ("name", "receiver", "arity")}
                       for c in md.get("calls") or ()]
        store._conn.execute("UPDATE chunks SET metadata = ? WHERE id = ?", (json.dumps(md), cid))
    store.commit()
    build_refs(store)
    rows = {r["name"]: r for r in find_usages(store, "Vector::size")["results"]}
    assert set(rows) == {"count", "total", "length"}
    assert all(r.get("unverified") for r in rows.values())


def test_a_bare_method_name_groups_its_callers_by_class(tmp_path):
    store = _index(tmp_path, _VECTORS)
    result = find_usages(store, "size")
    assert result["by_class"] == [{"class": "Vector", "callers": 2}, {"class": "String", "callers": 1}]
    assert {r["name"]: r.get("classes") for r in result["results"]} == {
        "count": ["Vector"], "total": ["Vector"], "length": ["String"]}
    assert "Vector::size" in result["note"]


def test_a_caller_whose_class_lacks_the_method_stays_a_mention(tmp_path):
    store = _index(tmp_path, {
        "node.h": "class Node {\npublic:\n    void ready();\n};\nclass Timer {\npublic:\n    void tick();\n};\n",
        "timer.cpp": '#include "node.h"\n\nvoid Timer::tick() {}\n',
        "poll.cpp": '#include "node.h"\nvoid poll(Node *n) { n->tick(); }\n',
    })
    assert {r["name"]: r["edge_type"] for r in find_usages(store, "Timer::tick")["results"]} == {"poll": "mentions"}


_SHAPES = {
    "shapes.h": "class Shape {\npublic:\n    void fill();\n};\nclass Circle : public Shape {};\n",
    "shape.cpp": '#include "shapes.h"\n\nvoid Shape::fill() {}\n',
    "timer.h": "template <class T>\nclass Timer {\npublic:\n    Timer() {}\n    void start() {}\n};\n",
    "label.h": "namespace ui {\nclass Label {\npublic:\n    int size() const { return 0; }\n};\n}\n",
    "math.h": "namespace Math {\ndouble ease(double p_x);\n}\n",
    "math.cpp": '#include "math.h"\n\ndouble Math::ease(double p_x) { return p_x; }\n',
}


def _sites(result: dict) -> list[tuple[str, str, int]]:
    return [(r["name"], Path(r["path"]).name, r["start_line"]) for r in result["symbols"]]


def test_find_symbol_answers_a_method_defined_inside_or_outside_its_class(tmp_path):
    store = _index(tmp_path, _VECTORS)
    assert _sites(find_symbol(store, "Vector::size")) == [("size", "vector.h", 9)]
    assert _sites(find_symbol(store, "String::size")) == [("String::size", "string.cpp", 3)]
    assert find_symbol(store, "Vector::size")["note"] is None


def test_find_symbol_answers_an_inherited_method_and_names_the_base(tmp_path):
    store = _index(tmp_path, _SHAPES)
    result = find_symbol(store, "Circle::fill")
    assert _sites(result) == [("Shape::fill", "shape.cpp", 3)]
    assert result["note"] == "Circle does not define 'fill'; it inherits Shape::fill, whose definitions these are"


def test_find_symbol_answers_a_method_of_a_class_in_a_namespace(tmp_path):
    store = _index(tmp_path, _SHAPES)
    assert _sites(find_symbol(store, "ui::Label::size")) == [("size", "label.h", 4)]
    result = find_symbol(store, "Label::size")
    assert _sites(result) == [("size", "label.h", 4)]
    assert result["note"] == "'Label' matched ui::Label"


def test_find_symbol_answers_a_namespace_function(tmp_path):
    store = _index(tmp_path, _SHAPES)
    assert _sites(find_symbol(store, "Math::ease")) == [("Math::ease", "math.cpp", 3)]


def test_find_symbol_leaves_the_class_and_its_template_out_of_a_constructor_lookup(tmp_path):
    store = _index(tmp_path, _SHAPES)
    assert {r["kind"] for r in store.find_symbols("Timer")} >= {"class_specifier", "template_declaration"}
    assert [(r["name"], r["kind"]) for r in find_symbol(store, "Timer::Timer")["symbols"]] == [
        ("Timer", "function_definition")]


def test_find_symbol_miss_names_the_class_that_lacks_the_method(tmp_path):
    store = _index(tmp_path, _SHAPES)
    result = find_symbol(store, "Timer::stop")
    assert result["symbols"] == []
    assert result["note"].startswith("Timer has no method 'stop' — ")
    assert "codebase_search mode=fts" in result["note"]


def test_find_symbol_scopes_a_class_method_to_path_prefix(tmp_path):
    store = _index(tmp_path, _VECTORS)
    assert _sites(find_symbol(store, "Vector::size", path_prefix="vec")) == [("size", "vector.h", 9)]
    assert find_symbol(store, "Vector::size", path_prefix="string")["symbols"] == []


def test_find_symbol_reads_bare_names_and_prefixes_from_the_symbol_index(tmp_path):
    store = _index(tmp_path, _VECTORS)
    assert _sites(find_symbol(store, "size")) == [("size", "vector.h", 9)]
    assert _sites(find_symbol(store, "String::", prefix=True)) == [("String::size", "string.cpp", 3)]
    assert find_symbol(store, "Vector::", prefix=True)["symbols"] == []


def test_symbol_and_investigate_routes_name_the_base_of_an_inherited_method(monkeypatch, tmp_path):
    store = _index(tmp_path, _SHAPES)
    monkeypatch.setattr(serve_projects, "_projects", {serve_projects.DEFAULT_PROJECT: {"store": store}})
    note = "Circle does not define 'fill'; it inherits Shape::fill, whose definitions these are"
    body = json.loads(serve_app.symbol(serve_models.SymbolRequest(name="Circle::fill")).body)
    assert (body["count"], body["note"]) == (1, note)
    body = json.loads(serve_app.investigate(serve_models.InvestigateRequest(name="Circle::fill")).body)
    assert [d["name"] for d in body["definitions"]] == ["Shape::fill"]
    assert body["notes"]["definitions"] == note
