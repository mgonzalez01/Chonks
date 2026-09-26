"""C++ calls resolved through the class model: the receiver's class from
call-site evidence, then the method in that class or its bases."""
import hashlib
import json

import pytest

import chonks.index.graph.members as members
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


def _files(store: Store, caller: str) -> set[str]:
    """The files of the chunks the chunk named `caller` calls."""
    rows = store._conn.execute(
        "SELECT t.path FROM chunk_refs r JOIN chunks f ON f.id = r.from_id "
        "JOIN chunks t ON t.id = r.to_id WHERE r.edge_type = 'calls' AND f.name = ?", (caller,))
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
