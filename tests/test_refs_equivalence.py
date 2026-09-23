"""Refactor safety net: _extract_refs moved from a per-language if/elif walk
to declarative specs (LANG_REFS_SPECS + a generic walker). Proves the live
version still matches the golden recorded from main's per-language walk,
node by node over a fixture corpus gnarlier than test_chunking.py's."""

import json
from pathlib import Path

import pytest
from tree_sitter_language_pack import get_parser

import chonks.index.refs_extract as new_chunking

GOLDEN_PATH = Path(__file__).parent / "data" / "refs_equivalence_golden.json"


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fixture corpus: real snippets per language, exercising calls / imports /
# inherits, including the bespoke nested/aliased/qualified-name cases each
# language's spec has to reproduce exactly.
# ---------------------------------------------------------------------------

CPP_SRC = b'''
#include <vector>
#include <foo/bar.h>
#include "local.h"
#include "local.h"

namespace ns {

class Base {};
struct IOther {};

class Widget : public Base, protected ns::IOther {
public:
    void run() {
        helper();
        helper();
        this->other->call(x);
        ns::Static();
        Bar::Baz::Qux();
    }
};

struct Plain : Base {
    void go() { plain_call(); }
};

}
'''

CSHARP_SRC = b'''
using System;
using System.Collections.Generic;
using Godot;

namespace Foo {

class Base {}
interface IThing {}

class Bar : Base, IThing {
    void Run() {
        Helper();
        Helper();
        this.other.Call(x);
        new Widget();
        new List<int>();
        var q = new Some.Qualified.Type();
    }
}

struct SBase {}
struct SDerived : SBase {}

interface IExtra : IThing {}

}
'''

PYTHON_SRC = b'''
import os
import os.path
from collections import defaultdict
from typing import Any, Optional
import xml.etree.ElementTree as ET
from a.b import c as d, e as f

class Base:
    pass

class Foo(Base):
    def run(self):
        helper()
        helper()
        self.other.call(x)
        mod.sub.deep_call()

class Bar(Base, metaclass=type):
    def go(self):
        another()
'''

GDSCRIPT_SRC = b'''
extends Node2D

func _ready():
    var x = get_node("Bar")
    x.do_thing()
    x.do_thing()
    other.nested.chained_call()

func _process(delta):
    helper_fn()
'''


def _many_calls_src(n: int) -> bytes:
    calls = "\n".join(f"    fn_{i}();" for i in range(n))
    return f"void driver() {{\n{calls}\n}}\n".encode()


CASES = {
    "cpp": CPP_SRC,
    "c_sharp": CSHARP_SRC,
    "python": PYTHON_SRC,
    "gdscript": GDSCRIPT_SRC,
}


def _walk_all_nodes(node):
    yield node
    for child in node.children:
        yield from _walk_all_nodes(child)


def _normalize(refs: dict) -> dict:
    # Drops the "literals" bucket new_chunking added, and flattens each
    # 'calls' entry to its bare name (main's are plain strings, new_chunking's
    # are {'name','receiver','arity'} fingerprint dicts).
    out = {k: v for k, v in refs.items() if k != "literals"}
    out["calls"] = [e["name"] if isinstance(e, dict) else e for e in out.get("calls", [])]
    return out


@pytest.mark.parametrize("lang", sorted(CASES))
def test_extract_refs_matches_golden_node_by_node(golden, lang):
    src = CASES[lang]
    parser = get_parser(lang)
    tree = parser.parse(src)

    mismatches = []
    for node, (gtype, gstart, gend, gref) in zip(
            _walk_all_nodes(tree.root_node), golden["nodes"][lang], strict=True):
        new = _normalize(new_chunking._extract_refs(node, lang, src))
        if node.type != gtype or node.start_byte != gstart or node.end_byte != gend or gref != new:
            mismatches.append((node.type, node.start_byte, node.end_byte, gref, new))

    assert not mismatches, mismatches


@pytest.mark.parametrize("lang", sorted(CASES))
def test_extract_refs_matches_golden_at_root(golden, lang):
    src = CASES[lang]
    parser = get_parser(lang)
    tree = parser.parse(src)
    gref = golden["nodes"][lang][0][3]
    new = _normalize(new_chunking._extract_refs(tree.root_node, lang, src))
    assert gref == new


def test_extract_refs_matches_golden_cap_and_dedup_cpp(golden):
    src = _many_calls_src(300)
    parser = get_parser("cpp")
    tree = parser.parse(src)
    new = _normalize(new_chunking._extract_refs(tree.root_node, "cpp", src))
    assert golden["cap_dedup"] == new
    assert len(new["calls"]) == golden["max_names"]


def test_extract_refs_matches_golden_unsupported_language(golden):
    # Lua has no refs spec; must fall back to empty refs on both sides.
    src = b"local function helper() end\nlocal function main() helper() end\n"
    parser = get_parser("lua")
    tree = parser.parse(src)
    new = _normalize(new_chunking._extract_refs(tree.root_node, "lua", src))
    assert golden["unsupported"] == new == {"calls": [], "imports": [], "inherits": []}


def test_lang_refs_specs_covers_same_languages_as_before(golden):
    # Guards monotonic growth (new languages can be added, none can silently
    # lose a refs spec), not that the set is frozen.
    assert set(golden["refs_langs"]) <= set(new_chunking._REFS_LANGS)
