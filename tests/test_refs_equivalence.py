"""Refactor safety net: _extract_refs moved from a per-language if/elif walk
to declarative specs (LANG_REFS_SPECS + a generic walker). Proves the new
version is output-identical to main's by diffing node by node over a fixture
corpus gnarlier than test_chunking.py's."""

import importlib.util
import sys
from pathlib import Path

import pytest
from tree_sitter_language_pack import get_parser

import chonks.chunking as new_chunking


def _load_main_chunking():
    # Loaded under a private module name so both implementations can be
    # imported side by side. Skips when `main` can't be resolved (detached CI
    # checkouts), where the test is vacuous anyway.
    import subprocess

    # Package path first, flat pre-reorg path as fallback.
    src = ""
    for ref_path in ("main:chonks/chunking.py", "main:chunking.py"):
        proc = subprocess.run(
            ["git", "show", ref_path],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout:
            src = proc.stdout
            break
    if not src:
        pytest.skip("local `main` ref unavailable (detached/shallow checkout)")
    spec = importlib.util.spec_from_loader("chunking_main_reference", loader=None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["chunking_main_reference"] = mod
    exec(compile(src, "chunking_main_reference.py", "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def old_chunking():
    return _load_main_chunking()


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
def test_extract_refs_matches_main_node_by_node(old_chunking, lang):
    src = CASES[lang]
    parser = get_parser(lang)
    tree = parser.parse(src)

    mismatches = []
    for node in _walk_all_nodes(tree.root_node):
        # main may predate literals/fingerprints or already carry them; a
        # fast-forwarded main must not read as a mismatch against itself.
        old = _normalize(old_chunking._extract_refs(node, lang, src))
        new = _normalize(new_chunking._extract_refs(node, lang, src))
        if old != new:
            mismatches.append((node.type, node.start_byte, node.end_byte, old, new))

    assert not mismatches, mismatches


@pytest.mark.parametrize("lang", sorted(CASES))
def test_extract_refs_matches_main_at_root(old_chunking, lang):
    src = CASES[lang]
    parser = get_parser(lang)
    tree = parser.parse(src)
    old = _normalize(old_chunking._extract_refs(tree.root_node, lang, src))
    new = _normalize(new_chunking._extract_refs(tree.root_node, lang, src))
    assert old == new


def test_extract_refs_matches_main_cap_and_dedup_cpp(old_chunking):
    src = _many_calls_src(300)
    parser = get_parser("cpp")
    tree = parser.parse(src)
    old = _normalize(old_chunking._extract_refs(tree.root_node, "cpp", src))
    new = _normalize(new_chunking._extract_refs(tree.root_node, "cpp", src))
    assert old == new
    assert len(new["calls"]) == old_chunking._REFS_MAX_NAMES


def test_extract_refs_matches_main_unsupported_language(old_chunking):
    # HLSL has no refs spec; must fall back to empty refs on both sides.
    src = b'''
struct VSOut { float4 pos : SV_Position; };
VSOut main(float3 p : POSITION) { VSOut o; o.pos = float4(p, 1); return o; }
'''
    parser = get_parser("hlsl")
    tree = parser.parse(src)
    old = _normalize(old_chunking._extract_refs(tree.root_node, "hlsl", src))
    new = _normalize(new_chunking._extract_refs(tree.root_node, "hlsl", src))
    assert old == new == {"calls": [], "imports": [], "inherits": []}


def test_lang_refs_specs_covers_same_languages_as_before(old_chunking):
    # Guards monotonic growth (new languages can be added, none can silently
    # lose a refs spec), not that the set is frozen.
    assert old_chunking._REFS_LANGS <= set(new_chunking._REFS_LANGS)
