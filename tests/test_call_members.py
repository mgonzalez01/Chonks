"""C++ calls resolved through the class model: the receiver's class from
call-site evidence, then the method in that class or its bases."""
import hashlib
import json

import pytest

import chonks.resolve.members as members
import chonks.index.graph.refs as refs
from chonks.index.graph.refs import build_refs
from chonks.index.pipeline import index_paths
from chonks.languages._members import type_ref
from chonks.languages.spec import TypeRef
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


def _files(store: Store, caller: str, edge_type: str = "calls") -> set[str]:
    """The files of the chunks the chunk named `caller` links to by `edge_type`."""
    rows = store._conn.execute(
        "SELECT t.path FROM chunk_refs r JOIN chunks f ON f.id = r.from_id "
        "JOIN chunks t ON t.id = r.to_id WHERE r.edge_type = ? AND f.name = ?", (edge_type, caller))
    return {r[0].rsplit("/", 1)[-1] for r in rows}


_SHAPES = {
    "shapes.h": """namespace scene {
class Shape {
public:
    virtual float area() const;
    void fill();
};
class Circle : public Shape {
public:
    float area() const override;
};
}
class Label {
public:
    float area() const;
    void fill();
};
""",
    "shape.cpp": '#include "shapes.h"\nnamespace scene {\nfloat Shape::area() const { return 0.0f; }\n}\n',
    "shape_fill.cpp": '#include "shapes.h"\nnamespace scene {\nvoid Shape::fill() {}\n}\n',
    "circle.cpp": '#include "shapes.h"\nnamespace scene {\nfloat Circle::area() const { return 1.0f; }\n}\n',
    "label.cpp": '#include "shapes.h"\nfloat Label::area() const { return 2.0f; }\n',
    "label_fill.cpp": '#include "shapes.h"\nvoid Label::fill() {}\n',
}


def test_type_ref_reads_the_class_a_written_type_names():
    assert type_ref("const scene::Vector<Ref<Node>> *&") == TypeRef("scene::Vector", ("Ref<Node>",), 1)
    assert type_ref("struct Node **") == TypeRef("Node", (), 2)
    assert type_ref("List<T>::Element") == TypeRef("List::Element")
    assert type_ref("Node []") == TypeRef("Node", (), 1)
    assert type_ref("unsigned int") is None
    assert type_ref("void (*)(int)") is None
    assert type_ref("auto") is None


def test_a_typed_receiver_links_only_its_class_method(tmp_path):
    store = _index(tmp_path, {**_SHAPES, "use.cpp": (
        '#include "shapes.h"\n'
        "float measure(scene::Circle *c) { return c->area(); }\n")})
    assert _files(store, "measure") == {"circle.cpp"}


def test_a_method_only_a_base_defines_links_the_base_method(tmp_path):
    store = _index(tmp_path, {**_SHAPES, "use.cpp": (
        '#include "shapes.h"\n'
        "namespace scene {\nvoid paint(Circle &c) { c.fill(); }\n}\n")})
    assert _files(store, "paint") == {"shape_fill.cpp"}


def test_a_receiver_chain_walks_field_and_return_types(tmp_path):
    store = _index(tmp_path, {
        "mesh.h": """class Mesh {
public:
    int surface_count() const;
};
class MeshInstance {
public:
    Mesh *get_mesh() const;
    Mesh *mesh = nullptr;
};
class Font {
public:
    int surface_count() const;
};
""",
        "mesh.cpp": '#include "mesh.h"\nint Mesh::surface_count() const { return 1; }\n',
        "mesh_instance.cpp": '#include "mesh.h"\nMesh *MeshInstance::get_mesh() const { return mesh; }\n',
        "font.cpp": '#include "mesh.h"\nint Font::surface_count() const { return 0; }\n',
        "use.cpp": '#include "mesh.h"\nint count(MeshInstance *mi) {\n'
                   "    return mi->get_mesh()->surface_count() + mi->mesh->surface_count();\n}\n",
    })
    assert _files(store, "count") == {"mesh.cpp", "mesh_instance.cpp"}


def test_arrow_on_a_class_goes_through_its_operator_arrow(tmp_path):
    store = _index(tmp_path, {
        "ref.h": """template <class T>
class Ref {
    T *ptr = nullptr;
public:
    T *operator->() const { return ptr; }
};
class Texture {
public:
    int get_width() const;
};
class Image {
public:
    int get_width() const;
};
""",
        "texture.cpp": '#include "ref.h"\nint Texture::get_width() const { return 1; }\n',
        "image.cpp": '#include "ref.h"\nint Image::get_width() const { return 2; }\n',
        "use.cpp": '#include "ref.h"\nint width(const Ref<Texture> &t) { return t->get_width(); }\n',
    })
    assert _files(store, "width") == {"texture.cpp"}


def test_an_explicit_operator_call_links_the_operator_of_its_class(tmp_path):
    ops = "operator String() const;\n    bool operator==(const {0} &p_other) const;\n" \
          "    bool operator()(int p_x) const;\n    int operator[](int p_idx) const;\n"
    defs = "{0}::operator String() const {{ return String(); }}\n" \
           "bool {0}::operator==(const {0} &p_other) const {{ return true; }}\n" \
           "bool {0}::operator()(int p_x) const {{ return false; }}\n" \
           "int {0}::operator[](int p_idx) const {{ return 0; }}\n"
    store = _index(tmp_path, {
        "variant.h": "template <class T>\nclass Base {\npublic:\n    Base &operator=(const Base &p_other);\n};\n"
                     "class Variant {\npublic:\n    " + ops.format("Variant") + "    int first() const;\n};\n"
                     "class Color {\npublic:\n    " + ops.format("Color") + "};\n"
                     "template <class T>\nclass Typed : public Base<T> {\npublic:\n"
                     "    void assign(const Typed &p_other);\n};\n",
        "base.cpp": '#include "variant.h"\ntemplate <class T>\n'
                    "Base<T> &Base<T>::operator=(const Base &p_other) { return *this; }\n",
        "variant.cpp": '#include "variant.h"\n' + defs.format("Variant"),
        "color.cpp": '#include "variant.h"\n' + defs.format("Color"),
        "use.cpp": '#include "variant.h"\nbool use(const Variant &v, const Variant *p) {\n'
                   "    v.operator String();\n    return p->operator==(v);\n}\n",
        "first.cpp": '#include "variant.h"\n'
                     "int Variant::first() const { return this->operator[](0) + Variant::operator()(1); }\n",
        "assign.cpp": '#include "variant.h"\ntemplate <class T>\n'
                      "void Typed<T>::assign(const Typed &p_other) { Base<T>::operator=(p_other); }\n",
    })
    assert _files(store, "use") == {"variant.cpp"}
    assert _files(store, "Variant::first") == {"variant.cpp"}
    assert _files(store, "Typed<T>::assign") == {"base.cpp"}


def test_a_static_call_reaches_members_defined_under_a_using_directive(tmp_path):
    store = _index(tmp_path, {
        "server.h": """namespace ns {
class Server {
public:
    static Server *get_singleton();
    void sync();
};
}
class Queue {
public:
    void sync();
};
""",
        "server.cpp": '#include "server.h"\nusing namespace ns;\nServer *Server::get_singleton() { return nullptr; }\n',
        "server_sync.cpp": '#include "server.h"\nusing namespace ns;\nvoid Server::sync() {}\n',
        "queue.cpp": '#include "server.h"\nvoid Queue::sync() {}\n',
        "use.cpp": '#include "server.h"\nvoid flush() { ns::Server::get_singleton()->sync(); }\n',
    })
    assert _files(store, "flush") == {"server.cpp", "server_sync.cpp"}


def test_a_bare_call_in_a_method_links_the_calling_class_member(tmp_path):
    store = _index(tmp_path, {
        "node.h": """class Node {
public:
    void update();
    void queue_redraw();
};
class Control {
public:
    void update();
};
""",
        "node.cpp": '#include "node.h"\nvoid Node::update() {}\n',
        "node_redraw.cpp": '#include "node.h"\nvoid Node::queue_redraw() { update(); }\n',
        "control.cpp": '#include "node.h"\nvoid Control::update() {}\n',
    })
    assert _files(store, "Node::queue_redraw") == {"node.cpp"}


def test_a_global_call_links_only_free_functions(tmp_path):
    store = _index(tmp_path, {
        "file.h": "class File {\npublic:\n    void flush();\n    void close();\n    void sync();\n};\n"
                  "class Pipe {\npublic:\n    void flush();\n    void close();\n};\n"
                  "class Buffer {\npublic:\n    Buffer(int p_size);\n};\n",
        "file_flush.cpp": '#include "file.h"\nvoid File::flush() {}\n',
        "file_close.cpp": '#include "file.h"\nvoid File::close() {}\n',
        "pipe.cpp": '#include "file.h"\nvoid Pipe::flush() {}\nvoid Pipe::close() {}\n',
        "buffer.cpp": '#include "file.h"\nBuffer::Buffer(int p_size) {}\n',
        "global.cpp": "void flush() {}\n",
        "sync.cpp": '#include "file.h"\nvoid File::sync() { ::flush(); ::close(3); ::Buffer(4); }\n',
    })
    assert _files(store, "File::sync") == {"global.cpp", "buffer.cpp"}


def test_a_method_only_declared_links_the_chunk_holding_the_declaration(tmp_path):
    store = _index(tmp_path, {
        "shape.h": "class Shape {\npublic:\n    virtual void draw() = 0;\n};\n"
                   "class Canvas {\npublic:\n    void draw();\n};\n",
        "canvas.cpp": '#include "shape.h"\nvoid Canvas::draw() {}\n',
        "use.cpp": '#include "shape.h"\nvoid render(Shape *s) { s->draw(); }\n',
    })
    assert _files(store, "render") == {"shape.h"}


_THREAD = {
    "thread.h": """class Thread {
public:
    void start(int p_priority = 0);
    void start(int a, int b, int c);
};
""",
    "thread.cpp": '#include "thread.h"\nvoid Thread::start(int p_priority) {}\n',
    "thread_start3.cpp": '#include "thread.h"\nvoid Thread::start(int a, int b, int c) {}\n',
}


def test_a_call_using_a_default_links_the_overload_whose_declaration_has_it(tmp_path):
    store = _index(tmp_path, {**_THREAD, "use.cpp": '#include "thread.h"\nvoid run(Thread &t) { t.start(); }\n'})
    assert _files(store, "run") == {"thread.cpp"}


def test_a_call_no_overload_fits_links_every_overload_of_the_class(tmp_path):
    store = _index(tmp_path, {**_THREAD, "timer.cpp": "void Timer::start(int a, int b) {}\n",
                              "use.cpp": '#include "thread.h"\nvoid run(Thread &t) { t.start(1, 2); }\n'})
    assert _files(store, "run") == {"thread.cpp", "thread_start3.cpp"}


def test_an_auto_local_takes_the_return_type_of_the_function_it_is_set_from(tmp_path):
    store = _index(tmp_path, {
        "node.h": "class Node {\npublic:\n    static Node *create();\n    void ready();\n};\n"
                  "class Timer {\npublic:\n    void ready();\n};\n",
        "node.cpp": '#include "node.h"\nNode *Node::create() { return nullptr; }\n',
        "node_ready.cpp": '#include "node.h"\nvoid Node::ready() {}\n',
        "timer.cpp": '#include "node.h"\nvoid Timer::ready() {}\n',
        "use.cpp": '#include "node.h"\nvoid boot() { auto n = Node::create(); n->ready(); }\n',
    })
    assert _files(store, "boot") == {"node.cpp", "node_ready.cpp"}


_MISSING = {
    "node.h": "class Node {\npublic:\n    void ready();\n};\nclass Timer {\npublic:\n    void tick();\n};\n",
    "node.cpp": '#include "node.h"\nvoid Node::ready() {}\n',
    "timer.cpp": '#include "node.h"\nvoid Timer::tick() {}\n',
    "poll.cpp": '#include "node.h"\nvoid poll(Node *n) { n->tick(); }\n',
    "wait.cpp": '#include "node.h"\nvoid wait(Server *s) { s->tick(); }\n',
}


def test_calls_the_model_cannot_place_link_nothing(tmp_path):
    store = _index(tmp_path, _MISSING)
    assert _files(store, "poll") == set()
    assert _files(store, "wait") == set()


def test_an_operator_call_the_model_cannot_place_takes_no_name_index(tmp_path):
    store = _index(tmp_path, {
        "vec.h": "class Vec {\npublic:\n    Vec &operator=(const Vec &p_other);\n    void reset();\n};\n"
                 "class Mat {\npublic:\n    Mat &operator=(const Mat &p_other);\n};\nusing Alias = Vec;\n",
        "vec.cpp": '#include "vec.h"\nVec &Vec::operator=(const Vec &p_other) { return *this; }\n',
        "vec_reset.cpp": '#include "vec.h"\nvoid Vec::reset() {}\n',
        "mat.cpp": '#include "vec.h"\nMat &Mat::operator=(const Mat &p_other) { return *this; }\n',
        "use.cpp": '#include "vec.h"\nvoid assign(const Vec &b) { Alias::operator=(b); Alias::reset(); }\n',
    })
    assert _files(store, "assign") == {"vec_reset.cpp"}


@pytest.mark.parametrize("outcome, caller", [(members.NOT_IN_CLASS, "poll"), (members.NO_CLASS, "wait")])
def test_calls_the_model_cannot_place_take_the_name_index_under_the_fallback_policy(
        tmp_path, monkeypatch, outcome, caller):
    monkeypatch.setitem(members.UNRESOLVED_TARGETS, outcome, None)
    store = _index(tmp_path, _MISSING)
    assert _files(store, caller) == {"timer.cpp"}


def test_an_index_without_the_class_model_resolves_calls_by_name(tmp_path):
    store = _index(tmp_path, {**_SHAPES, "use.cpp": (
        '#include "shapes.h"\n'
        "float measure(scene::Circle *c) { return c->area(); }\n")})
    store._conn.execute("DELETE FROM meta WHERE key = 'class_model_version'")
    store.commit()
    build_refs(store)
    assert _files(store, "measure") == {"shape.cpp", "circle.cpp", "label.cpp"}


def test_calls_without_evidence_resolve_by_name_under_a_class_model(tmp_path, monkeypatch):
    store = _index(tmp_path, {**_SHAPES, "use.cpp": (
        '#include "shapes.h"\n'
        "float measure(scene::Circle *c) { return c->area(); }\n")})
    for cid, raw in store._conn.execute("SELECT id, metadata FROM chunks WHERE metadata IS NOT NULL").fetchall():
        md = json.loads(raw)
        md["calls"] = [{k: v for k, v in c.items() if k in ("name", "receiver", "arity")}
                       for c in md.get("calls") or ()]
        store._conn.execute("UPDATE chunks SET metadata = ? WHERE id = ?", (json.dumps(md), cid))
    store.commit()
    build_refs(store)
    stripped = set(store.get_all_refs_typed())
    monkeypatch.setattr(refs, "members_index", lambda _store: None)
    build_refs(store)
    assert stripped == set(store.get_all_refs_typed())
    assert _files(store, "measure") == {"shape.cpp", "circle.cpp", "label.cpp"}


def test_a_name_every_call_of_which_resolves_makes_no_mentions(tmp_path):
    store = _index(tmp_path, {**_SHAPES, "use.cpp": (
        '#include "shapes.h"\n'
        "float measure(scene::Circle *c) {\n    return c->area();\n}\n")})
    assert _files(store, "measure") == {"circle.cpp"}
    assert _files(store, "measure", "mentions") == {"shapes.h"}


def _use(text: str) -> dict[str, str]:
    return {**_SHAPES, "use.cpp": '#include "shapes.h"\n' + text}


@pytest.mark.parametrize("files, caller, mentioned", [
    (_MISSING, "poll", {"node.h", "timer.cpp"}),
    (_use("float wait(Server *s) {\n    return s->area();\n}\n"), "wait", {"shape.cpp", "circle.cpp", "label.cpp"}),
    (_use("void run() {\n    label.area();\n}\n"), "run", {"shape.cpp", "circle.cpp"}),
    (_use("float both(scene::Circle *c, Server *s) {\n    return c->area() + s->area();\n}\n"), "both",
     {"shape.cpp", "label.cpp", "shapes.h"}),
])
def test_a_name_with_a_call_the_class_model_leaves_open_keeps_its_mentions(tmp_path, files, caller, mentioned):
    store = _index(tmp_path, files)
    assert _files(store, caller, "mentions") == mentioned


def test_a_name_mentioned_without_a_call_keeps_its_mentions(tmp_path):
    store = _index(tmp_path, {**_SHAPES, "use.cpp": (
        '#include "shapes.h"\n'
        "float measure(scene::Circle *c) {\n    // fill is left to the caller\n    return c->area();\n}\n")})
    assert _files(store, "measure", "mentions") == {"shapes.h", "shape_fill.cpp", "label_fill.cpp"}


def test_a_language_without_a_class_model_keeps_the_mentions_of_names_it_calls(tmp_path, monkeypatch):
    store = _index(tmp_path, {
        **_SHAPES,
        "canvas.py": "class Canvas:\n    def fill(self):\n        pass\n",
        "sprite.py": "class Sprite:\n    def fill(self):\n        pass\n",
        "render.py": "def render(canvas):\n    canvas.fill()\n",
    })
    cid, raw = store._conn.execute("SELECT id, metadata FROM chunks WHERE name = 'render'").fetchone()
    md = json.loads(raw)
    for call in md["calls"]:
        call.update(racc=".", rhead={"name": "canvas", "via": "param", "type": "Label"})
    store._conn.execute("UPDATE chunks SET metadata = ? WHERE id = ?", (json.dumps(md), cid))
    store.commit()

    def python_edges() -> set[tuple[str, str, str]]:
        py = {r[0] for r in store._conn.execute("SELECT id FROM chunks WHERE language = 'python'")}
        return {e for e in store.get_all_refs_typed() if e[0] in py}

    build_refs(store)
    with_model = python_edges()
    monkeypatch.setattr(refs, "members_index", lambda _store: None)
    build_refs(store)
    assert with_model == python_edges()
    assert _files(store, "render") == {"canvas.py"}
    assert _files(store, "render", "mentions") == {"sprite.py", "shape_fill.cpp", "label_fill.cpp"}
