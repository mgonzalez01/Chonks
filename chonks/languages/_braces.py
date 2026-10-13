"""Brace-language text scan: where in a chunk a `name(` can be a signature rather than a call."""
from __future__ import annotations

import bisect
import re
from functools import lru_cache
from typing import Callable

_TOKENS = re.compile(
    r"//[^\n]*|/\*.*?(?:\*/|\Z)|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n]){1,8}'|[{}();]", re.S)
_COMMENTS = re.compile(r"//[^\n]*|/\*.*?(?:\*/|\Z)", re.S)
_DIRECTIVES = re.compile(r"^[ \t]*#.*$", re.M)
_STRINGS = re.compile(r"\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n]){1,8}'")
_BRACKETS = re.compile(r"\[[^\[\]]*\]")
_TEMPLATE = re.compile(r"\btemplate\s*<")
_WORD = re.compile(r"[A-Za-z_]\w*")


def _without_template_head(header: str) -> str:
    m = _TEMPLATE.search(header)
    if m is None:
        return header
    depth = 0
    for i in range(m.end() - 1, len(header)):
        if header[i] == "<":
            depth += 1
        elif header[i] == ">":
            depth -= 1
            if depth == 0:
                return header[:m.start()] + header[i + 1:]
    return header


def _opens_body(header: str, keywords: frozenset[str]) -> bool:
    """Whether the `{` after `header` opens code rather than a class or
    namespace body: `void f() const {`, `[](int x) {`, `= {`, `get {`, but not
    `class API(x) Foo : Bar {` or `REFLECT() class Foo {`."""
    text = _COMMENTS.sub(" ", header)
    text = _BRACKETS.sub(" ", _STRINGS.sub(" s ", _DIRECTIVES.sub(" ", text)))
    text = _without_template_head(text).strip()
    if any(w in keywords for w in _WORD.findall(text.split("(", 1)[0] + " " + text.rsplit(")", 1)[-1])):
        return False
    return "(" in text or text.endswith(("=", "=>")) or len(text.split()) < 2


@lru_cache(maxsize=256)
def _hidden(content: str, keywords: frozenset[str]) -> tuple[list[int], list[int]]:
    """Sorted, disjoint spans of comments, strings, parenthesized text and
    function bodies, as (starts, ends)."""
    spans: list[tuple[int, int]] = []
    braces: list[bool] = []  # True: code
    parens = 0
    paren_from = 0
    header = 0
    code_from: int | None = None
    for m in _TOKENS.finditer(content):
        tok, at = m.group(), m.start()
        if len(tok) > 1:
            if code_from is None:
                spans.append((at, m.end()))
        elif code_from is not None:
            if tok == "{":
                braces.append(True)
            elif tok == "}" and braces:
                braces.pop()
                if not any(braces):
                    spans.append((code_from, m.end()))
                    code_from = None
                    header = m.end()
        elif tok == "(":
            if parens == 0:
                paren_from = m.end()
            parens += 1
        elif tok == ")":
            if parens == 1:
                spans.append((paren_from, at))
            parens = max(0, parens - 1)
        elif parens:
            continue
        elif tok == "{":
            code = _opens_body(content[header:at], keywords)
            braces.append(code)
            if code:
                code_from = at
            header = m.end()
        elif tok in "};":
            if tok == "}" and braces:
                braces.pop()
            header = m.end()
    if code_from is not None:
        spans.append((code_from, len(content)))
    if parens:
        spans.append((paren_from, len(content)))
    starts: list[int] = []
    ends: list[int] = []
    for a, b in sorted(spans):
        if ends and a <= ends[-1]:
            ends[-1] = max(ends[-1], b)
        else:
            starts.append(a)
            ends.append(b)
    return starts, ends


def outside_bodies(keywords: frozenset[str]) -> Callable[[str, int], bool]:
    """A `signature_at` for a brace language whose `keywords` open a body of
    declarations (`class`, `namespace`): the offset is outside comments,
    strings, parentheses and function bodies."""
    def signature_at(content: str, pos: int) -> bool:
        starts, ends = _hidden(content, keywords)
        i = bisect.bisect_right(starts, pos) - 1
        return i < 0 or ends[i] <= pos
    return signature_at
