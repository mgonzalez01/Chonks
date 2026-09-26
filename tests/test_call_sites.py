"""Call-site evidence on C++ `calls` entries: racc, rhead, rpath and cls."""
import tempfile

from chonks.index.graph.refs import build_refs
from chonks.index.refs_extract import _REFS_MAX_EVIDENCE_PER_CALL
from chonks.index.rows import _chunk_metadata
from chonks.index.segment import CHUNK_MAX, segment_file
from chonks.storage.store import Store

_EVIDENCE_KEYS = ("racc", "rhead", "rpath", "cls")


def _calls(src: bytes) -> list[dict]:
    return [c for seg in segment_file(src, "cpp", path="t.cpp") for c in seg["refs"]["calls"]]


def _entry(src: bytes, name: str) -> dict:
    found = [c for c in _calls(src) if c["name"] == name]
    assert len(found) == 1, found
    return found[0]


def _head(src: bytes, name: str) -> dict:
    return _entry(src, name)["rhead"]


def test_parameter_head_has_declared_type():
    src = b"void f(const Widget *w, Node &n) { w->draw(); n.show(); }"
    assert _head(src, "draw") == {"name": "w", "via": "param", "type": "const Widget *"}
    assert _head(src, "show") == {"name": "n", "via": "param", "type": "Node &"}


def test_local_head_has_declared_type():
    src = b"void f() { Widget *a = nullptr, b; a->draw(); b.show(); }"
    assert _head(src, "draw") == {"name": "a", "via": "local", "type": "Widget *"}
    assert _head(src, "show") == {"name": "b", "via": "local", "type": "Widget"}


def test_specifiers_are_not_part_of_the_type():
    src = b"void f() { constexpr Vec3 up{0, 1, 0}; static const Node *root = get(); up.length(); root->name(); }"
    assert _head(src, "length")["type"] == "Vec3"
    assert _head(src, "name")["type"] == "const Node *"


def test_innermost_declaration_wins():
    src = b"void f() { Foo x; { Bar x; x.inner(); } x.outer(); }"
    assert _head(src, "inner")["type"] == "Bar"
    assert _head(src, "outer")["type"] == "Foo"


def test_range_for_variable_is_typed_in_the_body_only():
    src = b"void f(List &items) { for (Node *n : items) n->process(); }"
    assert _head(src, "process") == {"name": "n", "via": "local", "type": "Node *"}


def test_condition_and_preprocessor_declarations_are_visible():
    src = b"""void f() {
#ifdef TOOLS
  Tool *t = get_tool();
#endif
  t->use();
  if (Foo *p = next()) p->step();
}"""
    assert _head(src, "use")["type"] == "Tool *"
    assert _head(src, "step")["type"] == "Foo *"


def test_auto_takes_the_type_its_initializer_states():
    src = (b"void f() { auto a = Vec(1); auto *b = new Node(); auto c = std::make_unique<Mesh>();"
           b" auto d = Color{1}; auto e = static_cast<Light *>(p); auto &g = *q; auto h = b;"
           b" a.m1(); b->m2(); c->m3(); d.m4(); e->m5(); g.m6(); h->m7(); }")
    assert _head(src, "m1")["type"] == "Vec"
    assert _head(src, "m2")["type"] == "Node *"
    assert _head(src, "m3")["type"] == "Mesh"
    assert _head(src, "m4")["type"] == "Color"
    assert _head(src, "m5")["type"] == "Light *"
    assert _head(src, "m6") == {"name": "g", "via": "local", "type": None}
    assert _head(src, "m7")["type"] == "Node *"


def test_cast_receiver_is_typed():
    src = b"void f(Object *o) { static_cast<Node *>(o)->a(); ((Node *)o)->b(); Object::cast_to<Node>(o)->c(); }"
    assert _head(src, "a") == {"name": "static_cast()", "via": "unknown", "type": "Node *"}
    assert _head(src, "b") == {"name": "(Node *)o", "via": "unknown", "type": "Node *"}
    assert _head(src, "c") == {"name": "cast_to()", "via": "unknown", "type": "Node"}


def test_this_member_head_and_path():
    e = _entry(b"struct A { void f() { this->items.clear(); } };", "clear")
    assert (e["racc"], e["rhead"], e["rpath"]) == (".", {"name": "this", "via": "this", "type": None}, ["items"])


def test_member_field_head_is_unknown_with_steps():
    e = _entry(b"struct A { void f() { m_items.get(i).size(); } };", "size")
    assert (e["racc"], e["rhead"], e["rpath"]) == (
        ".", {"name": "m_items", "via": "unknown", "type": None}, ["get()"])


def test_static_call_head_is_the_qualifier():
    e = _entry(b"void f() { ns::Server::get_singleton()->run(); X::g(); }", "run")
    assert (e["racc"], e["rhead"], e["rpath"]) == (
        "->", {"name": "ns::Server", "via": "type", "type": "ns::Server"}, ["get_singleton()"])
    e = _entry(b"void f() { X::g(); }", "g")
    assert (e["racc"], e["rhead"]) == ("::", {"name": "X", "via": "type", "type": "X"})
    assert "rpath" not in e


def test_chained_call_path_keeps_call_steps():
    e = _entry(b"void f(Tree &t) { t.root().child(0).name(); }", "name")
    assert (e["racc"], e["rhead"], e["rpath"]) == (
        ".", {"name": "t", "via": "param", "type": "Tree &"}, ["root()", "child()"])


def test_bare_call_has_no_receiver_evidence():
    e = _entry(b"void f() { helper(1); }", "helper")
    assert not set(e) & set(_EVIDENCE_KEYS)


def test_later_piece_of_split_function_types_receiver_from_first_piece():
    filler = "".join(f"    total += {i};\n" for i in range(CHUNK_MAX // 16))
    src = f"void f() {{\n    Widget *w = make();\n{filler}    w->draw();\n}}\n".encode()
    segs = segment_file(src, "cpp", path="t.cpp")
    assert len(segs) > 1
    piece = next(s for s in segs if any(c["name"] == "draw" for c in s["refs"]["calls"]))
    assert "Widget *w" not in piece["content"]
    draw = next(c for c in piece["refs"]["calls"] if c["name"] == "draw")
    assert draw["rhead"] == {"name": "w", "via": "local", "type": "Widget *"}


def test_calling_class_of_inline_and_out_of_class_methods():
    src = b"""namespace ns {
class Outer { class Inner { void f() { a(); } }; };
void Thread::start() { b(); }
}
void ns::Timer::tick() { c(); }
template <class T> void Vector<T>::push_back(T v) { d(); }
void free_function() { e(); }
"""
    assert _entry(src, "a")["cls"] == "ns::Outer::Inner"
    assert _entry(src, "b")["cls"] == "ns::Thread"
    assert _entry(src, "c")["cls"] == "ns::Timer"
    assert _entry(src, "d")["cls"] == "Vector"
    assert "cls" not in _entry(src, "e")


def test_evidence_variants_keep_the_fingerprints_the_caps_kept_before():
    heads = "".join(f"a{i}.x.size(); " for i in range(10))
    others = "".join(f"r{i}.size(); " for i in range(9))
    calls = _calls(f"void f() {{ {heads}{others}}}".encode())
    fingerprints = list(dict.fromkeys((c["name"], c["receiver"], c["arity"]) for c in calls))
    assert fingerprints == [("size", "x", 0)] + [("size", f"r{i}", 0) for i in range(7)]
    assert sum(c["receiver"] == "x" for c in calls) == _REFS_MAX_EVIDENCE_PER_CALL


def test_graph_edges_ignore_call_site_evidence():
    files = {
        "parser.cpp": b"void Parser::parse(int a) {}\n",
        "lexer.cpp": b"void Lexer::parse(int a) {}\n",
        "drive.cpp": b"void drive(Parser *parser, Lexer &lexer) { parser->parse(1); lexer.parse(2); }\n",
    }
    segs = [(path, seg) for path, src in files.items() for seg in segment_file(src, "cpp", path=path)]

    def edges(strip: bool) -> set:
        chunks = []
        for i, (path, seg) in enumerate(segs):
            refs = seg["refs"]
            if strip:
                refs = dict(refs, calls=[{k: v for k, v in c.items() if k not in _EVIDENCE_KEYS}
                                         for c in refs["calls"]])
            chunks.append({"id": f"c{i}", "path": path, "language": "cpp",
                           "chunk_type": seg["chunk_type"], "name": seg["name"],
                           "start_line": seg["start_line"], "end_line": seg["end_line"],
                           "content": seg["content"], "metadata": _chunk_metadata(None, refs)})
        store = Store(tempfile.mktemp(suffix=".db"))
        store.insert_chunks(chunks, [[0.1, 0.2, 0.3, 0.4] for _ in chunks])
        build_refs(store)
        got = set(store.get_all_refs_typed())
        store.close()
        return got

    with_evidence = edges(strip=False)
    assert any(t == "calls" for _f, _t, t in with_evidence)
    assert with_evidence == edges(strip=True)


def test_languages_without_call_site_facts_get_no_evidence():
    segs = segment_file(b"class A:\n    def f(self):\n        self.x.go()\n", "python", path="t.py")
    assert all(not set(c) & set(_EVIDENCE_KEYS) for s in segs for c in s["refs"]["calls"])
