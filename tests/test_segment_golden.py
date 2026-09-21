"""Pins segment_file, symbol and definer-arity output over a frozen fixture corpus."""
import json
import sys
import warnings
from pathlib import Path

import pytest
from tree_sitter_language_pack import get_parser

from chonks.index.graph.call_resolve import _definer_param_arity
from chonks.index.segment import (
    PARSE_TIMEOUT_MICROS,
    _collect_symbols_from_root,
    segment_file,
)

# The corpus-arm baseline (Godot and one Loc-Bench repo, indexed at F1) is
# recorded outside git at ~/projects/chonks-plan/log/corpus-baseline-F1.md.
_DATA_DIR = Path(__file__).parent / "data" / "golden"
_LANG_PACK_NAME = {"c_sharp": "csharp"}


def _parse_root(src: bytes, lang: str):
    parser = get_parser(_LANG_PACK_NAME.get(lang, lang))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        parser.timeout_micros = PARSE_TIMEOUT_MICROS
    return parser.parse(src).root_node


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_to_jsonable(v) for v in obj)
    return obj


def _call_names(segments: list[dict]) -> list[str]:
    names: set[str] = set()
    for seg in segments:
        for call in seg["refs"].get("calls", []):
            name = call.get("name") if isinstance(call, dict) else call
            if name:
                names.add(name)
    return sorted(names)


def build_record(source: str, language: str) -> dict:
    src = source.encode("utf-8")
    counters: dict = {}
    segments = segment_file(src, language, counters=counters)
    root = _parse_root(src, language)
    symbols = _collect_symbols_from_root(root, language, src)
    names = _call_names(segments)
    arity = [
        {"chunk_index": i, "name": name, "arity": found}
        for i, seg in enumerate(segments)
        for name in names
        if (found := _definer_param_arity(seg["content"], name, language)) is not None
    ]
    return _to_jsonable({
        "segments": segments,
        "counters": counters,
        "symbols": symbols,
        "definer_param_arity": arity,
    })


def _case_files() -> list[Path]:
    return sorted(_DATA_DIR.glob("*.json"))


def _load_cases() -> list[dict]:
    cases = []
    for path in _case_files():
        cases.extend(json.loads(path.read_text()))
    return cases


def regen() -> None:
    for path in _case_files():
        cases = json.loads(path.read_text())
        for case in cases:
            case["expected"] = build_record(case["source"], case["language"])
        path.write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")


_CASES = _load_cases()
_IDS = [f"{case['language']}-{case['id']}" for case in _CASES]


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_golden_case(case):
    got = build_record(case["source"], case["language"])
    assert got == case["expected"]


if __name__ == "__main__":
    if "--regen" not in sys.argv:
        raise SystemExit("usage: test_segment_golden.py --regen")
    regen()
