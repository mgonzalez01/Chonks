"""Graph queries on one class's method: `Owner::m` read through the class
model, callers checked against their recorded calls, bare names grouped by class."""
import hashlib
import json

from chonks.index.graph.refs import build_refs
from chonks.index.pipeline import index_paths
from chonks.retrieval.graph_queries import find_definitions, find_outgoing, find_usages, get_impact
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
    assert [(d["name"], d["start_line"]) for d in find_definitions(store, "Vector::size")] == [("size", 9)]
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
