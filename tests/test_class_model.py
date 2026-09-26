"""The class model: members and bases of each C++ class, from the segmenter to the store."""
import hashlib

from chonks.index.pipeline import index_paths
from chonks.index.segment import CHUNK_MAX, segment_file
from chonks.storage.store import Store


def _model(source: str) -> tuple[list[dict], list[dict]]:
    counters: dict = {}
    segment_file(source.encode(), "cpp", counters=counters)
    return counters.get("members", []), counters.get("bases", [])


def _rows(source: str, *keys: str) -> list[tuple]:
    members, _ = _model(source)
    return [tuple(m[k] for k in keys) for m in members]


def test_out_of_class_definition_in_a_namespace_belongs_to_the_qualified_class():
    src = """
namespace core {
class Thread {
public:
    void start(int p_priority = 0);
};
}

namespace core {
void Thread::start(int p_priority) {}
}

void core::Thread::stop() {}
"""
    assert _rows(src, "owner", "name", "kind", "arity_min", "arity_max") == [
        ("core::Thread", "start", "method_decl", 0, 1),
        ("core::Thread", "start", "method_def", 1, 1),
        ("core::Thread", "stop", "method_def", 0, 0),
    ]


def test_nested_class_members_are_owned_by_the_outer_and_inner_names():
    src = """
namespace scene {
template <class T>
class List {
public:
    class Element {
        T value;
    public:
        Element *next();
    };
    Element *front();
};
}

template <class T>
void scene::List<T>::Element::erase() {}
"""
    assert _rows(src, "owner", "name", "kind", "type_text") == [
        ("scene::List::Element", "value", "field", "T"),
        ("scene::List::Element", "next", "method_decl", "Element *"),
        ("scene::List", "front", "method_decl", "Element *"),
        ("scene::List::Element", "erase", "method_def", "void"),
    ]


def test_template_class_and_specialization_owners_have_no_template_arguments():
    src = """
template <class T>
class Vector {
    T *data;
public:
    int size() const;
};

template <class T>
int Vector<T>::size() const { return 0; }

template <>
struct Hasher<int> {
    static uint32_t hash(int p_value);
};
"""
    assert _rows(src, "owner", "name", "kind") == [
        ("Vector", "data", "field"),
        ("Vector", "size", "method_decl"),
        ("Vector", "size", "method_def"),
        ("Hasher", "hash", "method_decl"),
    ]


def test_pure_virtual_static_const_and_override_are_flags():
    src = """
class Shape {
public:
    virtual ~Shape();
    virtual float area() const = 0;
    virtual void draw(Canvas *p_canvas, int p_layer = 0);
    static Shape *create(const String &p_type);
};

class Circle : public Shape {
public:
    float area() const override;
};
"""
    assert _rows(src, "owner", "name", "flags", "type_text") == [
        ("Shape", "~Shape", "virtual", None),
        ("Shape", "area", "virtual,pure,const", "float"),
        ("Shape", "draw", "virtual", "void"),
        ("Shape", "create", "static", "Shape *"),
        ("Circle", "area", "const,override", "float"),
    ]


def test_arity_counts_defaults_and_marks_packs_and_ellipses_variadic():
    src = """
class Printer {
    void none(void);
    void some(int p_a, const char *p_b = "", float p_c = 1.0);
    void format(const char *p_fmt, ...);
    template <class... Args>
    void print(const String &p_sep, Args &&...p_args);
};
"""
    assert _rows(src, "name", "arity_min", "arity_max", "variadic") == [
        ("none", 0, 0, 0),
        ("some", 1, 3, 0),
        ("format", 1, 1, 1),
        ("print", 1, 1, 1),
    ]


def test_macro_call_in_a_class_body_is_not_a_member_and_constructors_are():
    src = """
class Node : public Object {
    GDCLASS(Node, Object);
public:
    Node();
    explicit Node(const String &p_name);
    operator bool() const;
};
"""
    assert _rows(src, "name", "kind", "type_text", "arity_max") == [
        ("Node", "method_decl", None, 0),
        ("Node", "method_decl", None, 1),
        ("operator bool", "method_decl", "bool", 0),
    ]


def test_operator_names_drop_inner_spaces_before_a_symbol():
    src = """
struct Color {
    bool operator == (const Color &p_other) const;
    Color &operator[](int p_idx);
    void *operator new(size_t p_size);
};
"""
    assert _rows(src, "name", "type_text") == [
        ("operator==", "bool"), ("operator[]", "Color &"), ("operator new", "void *"),
    ]


def test_anonymous_union_fields_belong_to_the_enclosing_class():
    src = """
struct Vector2 {
    union {
        struct {
            real_t x;
            real_t y;
        };
        real_t coord[2];
    };
    void (*on_change)(int);
};
"""
    assert _rows(src, "owner", "name", "kind", "type_text") == [
        ("Vector2", "x", "field", "real_t"),
        ("Vector2", "y", "field", "real_t"),
        ("Vector2", "coord", "field", "real_t"),
        ("Vector2", "on_change", "field", "void (*)(int)"),
    ]


def test_members_inside_preprocessor_branches_are_read():
    src = """
class Viewport {
#ifdef TOOLS_ENABLED
    bool debug_draw;
#else
    int draw_mode;
#endif
};
"""
    assert _rows(src, "name") == [("debug_draw",), ("draw_mode",)]


def test_a_class_local_to_a_function_body_is_not_recorded():
    src = """
void RenderingServer::sync() {
    struct Pending { int count; };
    Pending p;
}
"""
    assert _rows(src, "owner", "name") == [("RenderingServer", "sync")]


def test_bases_keep_their_qualification_and_drop_template_arguments():
    src = """
namespace editor {
class Plugin : public Object, protected core::Named<Plugin> {};
}
"""
    _, bases = _model(src)
    assert [(b["owner"], b["base"]) for b in bases] == [
        ("editor::Plugin", "Object"), ("editor::Plugin", "core::Named"),
    ]


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


def _chunk(store: Store, chunk_id: str) -> dict:
    return dict(store._conn.execute(
        "SELECT name, start_line, end_line, content FROM chunks WHERE id=?", (chunk_id,)).fetchone())


def test_a_field_between_inline_methods_of_a_split_class_maps_to_the_chunk_holding_its_line(tmp_path):
    fields = "\n".join(f"    int light_mask_{i} = {i};" for i in range(20))
    decls = "\n".join(
        f"    void draw_shape_{i}(const Point2 &p_from, const Point2 &p_to, float p_width = -1.0);"
        for i in range(70))
    src = (
        "class Canvas {\n"
        "    bool is_dirty() const { return dirty; }\n"
        f"{fields}\n"
        "    int get_id() const { return id; }\n"
        f"{decls}\n"
        "    int get_layer() const { return layer; }\n"
        "};\n"
    )
    assert len(src.encode()) > CHUNK_MAX
    store = _index(tmp_path, {"canvas.h": src})
    lines = src.splitlines()
    members = store.get_members_by_owners(["Canvas"])
    assert len(members) == 20 + 70 + 3
    for m in members:
        chunk = _chunk(store, m["chunk_id"])
        assert chunk["start_line"] <= m["line"] <= chunk["end_line"]
        assert lines[m["line"] - 1].strip() in chunk["content"]


def test_a_method_in_a_merged_chunk_named_after_another_method_maps_to_that_chunk(tmp_path):
    src = (
        "String String::slice(int p_begin, int p_end) const { return substr(p_begin, p_end); }\n"
        "void String::drop_front(int p_count) { erase(0, p_count); }\n"
    )
    store = _index(tmp_path, {"ustring.cpp": src})
    [slice_row] = store.get_members("String", "slice")
    [drop_row] = store.get_members("String", "drop_front")
    assert drop_row["chunk_id"] == slice_row["chunk_id"]
    assert _chunk(store, drop_row["chunk_id"])["name"] == "String::slice"


def test_a_using_namespace_directive_is_recorded_for_the_file_that_defines_a_bare_owner(tmp_path):
    store = _index(tmp_path, {
        "foo.h": "namespace llvm {\nclass Foo {\n  void bar();\n};\n}\n",
        "foo.cpp": '#include "foo.h"\nusing namespace llvm;\n\nvoid Foo::bar() {}\n',
    })
    assert [(m["path"], m["kind"]) for m in store.get_members("llvm::Foo", "bar")] == [
        ("foo.h", "method_decl")]
    assert [(m["path"], m["kind"]) for m in store.get_members("Foo", "bar")] == [
        ("foo.cpp", "method_def")]
    assert store.get_using_namespaces(["foo.cpp", "foo.h"]) == [
        {"path": "foo.cpp", "scope": None, "namespace": "llvm", "line": 2}]


def test_using_directives_keep_their_enclosing_namespace_and_using_declarations_are_skipped():
    src = """
using namespace ::core::io;
namespace editor {
using namespace scene;
}
struct Derived : Base { using Base::update; };
using std::vector;
"""
    counters: dict = {}
    segment_file(src.encode(), "cpp", counters=counters)
    assert [(u["scope"], u["namespace"], u["line"]) for u in counters["using_namespaces"]] == [
        (None, "::core::io", 2), ("editor", "scene", 4)]


def test_reindexing_a_file_replaces_its_members_and_pruning_it_removes_them(tmp_path):
    store = _index(tmp_path, {"a.h": "class A : public Base { void run(); };\n"})
    assert [m["name"] for m in store.get_members_by_owners(["A"])] == ["run"]
    assert store.get_meta("class_model_version") == "1"

    (tmp_path / "a.h").write_text("class A : public Other { void stop(); int count; };\n")
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    assert [m["name"] for m in store.get_members_by_owners(["A"])] == ["count", "stop"]
    assert [b["base"] for b in store.get_bases(["A"])] == ["Other"]

    (tmp_path / "a.h").unlink()
    (tmp_path / "b.h").write_text("class B {};\n")
    index_paths([str(tmp_path)], store, _FakeEmbedder(), root=tmp_path)
    assert store.get_members_by_owners(["A"]) == []
    assert store.get_bases(["A"]) == []


def test_store_returns_member_and_base_rows_by_owner(tmp_path):
    store = Store(tmp_path / "test.db")
    row = {"language": "cpp", "kind": "method_decl", "type_text": "void", "arity_min": 0,
           "arity_max": 1, "variadic": 0, "flags": "virtual", "chunk_id": "c1"}
    store.replace_class_model("node.h", [
        {**row, "path": "node.h", "owner": "Node", "name": "ready", "line": 3},
        {**row, "path": "node.h", "owner": "Node2D", "name": "draw", "line": 9},
    ], [{"path": "node.h", "owner": "Node2D", "base": "Node", "line": 8}],
        [{"path": "node.h", "scope": None, "namespace": "core", "line": 1}])

    assert [(m["owner"], m["name"], m["flags"]) for m in store.get_members("Node", "ready")] == [
        ("Node", "ready", "virtual")]
    assert [m["name"] for m in store.get_members_by_owners(["Node", "Node2D"])] == ["ready", "draw"]
    assert [(b["owner"], b["base"]) for b in store.get_bases(["Node2D"])] == [("Node2D", "Node")]

    assert [u["namespace"] for u in store.get_using_namespaces(["node.h"])] == ["core"]

    store.replace_class_model("node.h", [], [], [])
    assert store.get_members_by_owners(["Node", "Node2D"]) == []
    assert store.get_bases(["Node2D"]) == []
    assert store.get_using_namespaces(["node.h"]) == []
