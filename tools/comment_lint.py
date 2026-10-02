#!/usr/bin/env python3
"""Comment budget gate for every first-party source.

Two rules: a comment occupies exactly one line, and the line carrying it fits in 140 columns.
The rule is "single line only, no multi-line splitting", so a run of consecutive comment lines
is a violation however short. The 140 matches python/pyproject.toml's ruff line-length, which cites
the same rule; ruff enforces the column on Python alone and cannot see the one-line rule at all, so
this gate is what applies both to every language in the repo.

Two comment syntaxes, each lexed rather than pattern-matched. Slash comments cover C++,
Objective-C++ and GLSL; hash comments cover CMake, Python and the YAML-style tool configs. Lexing
is not optional: tools/albedo_table.cpp emits '//' inside string literals when it generates
albedo_table.inc, and a regex would score that generated text as over-length comments of this file.

Out of scope, deliberately: third_party/ and build/ are not ours, *.inc is generated (its comments
are albedo_table.cpp's output, fixed there), *.json admits no comments, and *.md is prose, which is
where a derivation too long for one line belongs.
"""

from __future__ import annotations

import argparse
import bisect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

MAX_COLS = 140
MAX_RUN = 1

# Slash: C++, Objective-C++, GLSL. Hash: CMake, Python, and the tool configs, which are named rather than suffixed.
SLASH_SUFFIXES = frozenset({".cpp", ".h", ".hpp", ".mm", ".frag", ".vert"})
HASH_SUFFIXES = frozenset({".cmake", ".py"})
HASH_NAMES = frozenset({"CMakeLists.txt", ".clang-tidy", ".cppcheck-suppressions"})
TREES = ("src", "include", "tools", "cmake", "python", "assets/shaders")
ROOT_FILES = ("CMakeLists.txt", ".clang-tidy", ".cppcheck-suppressions")
SKIP_PARTS = frozenset({"third_party", "build", ".venv", "__pycache__"})

# Quote delimiters per language, longest first so a triple quote wins. CMake has no single-quoted string.
PY_QUOTES = ('"""', "'''", '"', "'")
CMAKE_QUOTES = ('"',)
YAML_QUOTES = ('"', "'")

_CODE, _LINE, _BLOCK, _STRING, _CHAR, _RAW = range(6)

# A lexer: source text in, one (start, end, owns_its_first_line) triple per comment out.
SpansFn = Callable[[str], list[tuple[int, int, bool]]]


@dataclass(frozen=True)
class Violation:
    """One breach of the budget, addressed as path:line."""

    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


@dataclass(frozen=True)
class Stats:
    """What a scan measured, for the summary line."""

    files: int = 0
    lines: int = 0
    comment_lines: int = 0
    comment_bytes: int = 0
    over_cols: int = 0
    over_run: int = 0
    longest: int = 0
    longest_run: int = 0


def _is_digit_separator(text: str, i: int) -> bool:
    """True if text[i] is a C++14 digit separator (1'000'000) rather than a char literal quote."""
    j = i
    while j > 0 and (text[j - 1].isalnum() or text[j - 1] == "_"):
        j -= 1
    return j < i and text[j].isdigit()


def _continues(text: str, newline: int) -> bool:
    """True if the line ending at this newline is spliced onto the next by a trailing backslash."""
    j = newline - 1
    if j >= 0 and text[j] == "\r":
        j -= 1
    return j >= 0 and text[j] == "\\"


def comment_spans(text: str) -> list[tuple[int, int, bool]]:
    """Lex C++ and return each comment as (start, end, owns_its_first_line) in character offsets."""
    spans: list[tuple[int, int, bool]] = []
    start = 0
    state = _CODE
    raw_close = ""
    line = 0
    only_space = True
    span_owns = False
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\n":
            if state == _LINE and not _continues(text, i):
                spans.append((start, i, span_owns))
                state = _CODE
            elif state in (_STRING, _CHAR):
                state = _CODE
            line += 1
            i += 1
            only_space = True
            continue
        if state == _CODE:
            if c == "/" and i + 1 < n and text[i + 1] in "/*":
                state = _LINE if text[i + 1] == "/" else _BLOCK
                span_owns = only_space
                start = i
                i += 2
                continue
            if c == "R" and i + 1 < n and text[i + 1] == '"':
                open_paren = text.find("(", i + 2)
                if open_paren != -1:
                    raw_close = ")" + text[i + 2 : open_paren] + '"'
                    state = _RAW
                    i = open_paren + 1
                    only_space = False
                    continue
            if c == '"':
                state = _STRING
            elif c == "'" and not _is_digit_separator(text, i):
                state = _CHAR
            if not c.isspace():
                only_space = False
            i += 1
            continue
        if state == _BLOCK:
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                state = _CODE
                i += 2
                spans.append((start, i, span_owns))
                continue
            i += 1
            continue
        if state in (_STRING, _CHAR):
            if c == "\\":
                # A backslash-newline splices the literal on; count it or later comments report one line early.
                if i + 1 < n and text[i + 1] == "\n":
                    line += 1
                i += 2
                continue
            if (c == '"' and state == _STRING) or (c == "'" and state == _CHAR):
                state = _CODE
            i += 1
            continue
        if state == _RAW:
            if text.startswith(raw_close, i):
                state = _CODE
                i += len(raw_close)
                continue
            i += 1
            continue
        i += 1
    if state in (_LINE, _BLOCK):
        spans.append((start, n, span_owns))
    return spans


def hash_comment_spans(text: str, quotes: tuple[str, ...] = PY_QUOTES) -> list[tuple[int, int, bool]]:
    """Lex a '#'-comment language, returning each comment as (start, end, owns_its_first_line).

    A '#' inside a string is not a comment: CMakeLists.txt passes '#' in message() text and Python
    docstrings quote shell. `quotes` is longest-first, so a Python triple quote is matched before
    the single quote that opens it.
    """
    spans: list[tuple[int, int, bool]] = []
    i, n = 0, len(text)
    only_space = True
    closing = ""
    while i < n:
        c = text[i]
        if closing:
            if c == "\\":
                i += 2
                continue
            if text.startswith(closing, i):
                i += len(closing)
                closing = ""
                continue
            i += 1
            continue
        if c == "#":
            end = text.find("\n", i)
            end = n if end == -1 else end
            spans.append((i, end, only_space))
            i = end
            continue
        opener = next((q for q in quotes if text.startswith(q, i)), "")
        if opener:
            closing = opener
            i += len(opener)
            only_space = False
            continue
        if c == "\n":
            only_space = True
        elif not c.isspace():
            only_space = False
        i += 1
    return spans


def spans_for(path: Path) -> SpansFn | None:
    """The lexer for this file, or None if the repo does not lint its language."""
    if path.suffix in SLASH_SUFFIXES:
        return comment_spans
    if path.name in HASH_NAMES:
        quotes = CMAKE_QUOTES if path.name == "CMakeLists.txt" else YAML_QUOTES
        return lambda text: hash_comment_spans(text, quotes)
    if path.suffix in HASH_SUFFIXES:
        return lambda text: hash_comment_spans(text, CMAKE_QUOTES if path.suffix == ".cmake" else PY_QUOTES)
    return None


def scan(text: str, spans: SpansFn = comment_spans) -> tuple[set[int], set[int]]:
    """(0-based lines carrying a comment, those whose comment owns its first line), from the spans."""
    carries: set[int] = set()
    owns: set[int] = set()
    starts = [0]
    for i, c in enumerate(text):
        if c == "\n":
            starts.append(i + 1)
    for begin, end, span_owns in spans(text):
        first = bisect.bisect_right(starts, begin) - 1
        last = bisect.bisect_right(starts, max(end - 1, begin)) - 1
        # A comment with code after it is an inline annotation, not prose, so it cannot start or extend a block.
        trailing = text[end : text.find("\n", end) if "\n" in text[end:] else len(text)]
        for ln in range(first, last + 1):
            carries.add(ln)
            if span_owns and not trailing.strip():
                owns.add(ln)
    return carries, owns


def check(path: Path, name: Path | None = None, spans: SpansFn | None = None) -> tuple[list[Violation], Stats]:
    """Apply both rules to one file. `name` is how it is reported; `spans` defaults to the lexer for its language."""
    name = name if name is not None else path
    spans = spans or spans_for(path) or comment_spans
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    carries, owns = scan(text, spans)
    comment_bytes = sum(end - begin for begin, end, _ in spans(text))
    found: list[Violation] = []
    longest = 0
    over_cols = 0
    for index in sorted(carries):
        width = len(lines[index].rstrip("\r"))
        longest = max(longest, width)
        if width > MAX_COLS:
            over_cols += 1
            found.append(Violation(name, index + 1, f"comment line is {width} columns, budget is {MAX_COLS}"))
    over_run = 0
    longest_run = 0
    run_start = None
    ordered = sorted(owns)
    for position, index in enumerate(ordered):
        if run_start is None:
            run_start = index
        last = position + 1 == len(ordered) or ordered[position + 1] != index + 1
        if last:
            length = index - run_start + 1
            longest_run = max(longest_run, length)
            if length > MAX_RUN:
                over_run += 1
                found.append(Violation(name, run_start + 1, f"comment spans {length} lines, the budget allows one"))
            run_start = None
    stats = Stats(1, len(lines), len(carries), comment_bytes, over_cols, over_run, longest, longest_run)
    return found, stats


def sources(root: Path) -> list[Path]:
    """Every first-party source this gate knows how to lex, in a stable order."""
    found = [p for tree in TREES for p in (root / tree).rglob("*") if spans_for(p) is not None]
    found += [root / name for name in ROOT_FILES if (root / name).is_file()]
    return sorted(p for p in found if not SKIP_PARTS & set(p.relative_to(root).parts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()

    paths = sources(args.root)
    if not paths:
        # A gate matching nothing looks like one finding nothing: a867840's HeaderFilterRegex matched no file for the repo's life.
        print(f"comment_lint: no sources under {args.root} in {', '.join(TREES)}", file=sys.stderr)
        return 2

    violations: list[Violation] = []
    total = Stats()
    for path in paths:
        found, stats = check(path, path.relative_to(args.root))
        violations += found
        total = Stats(
            total.files + stats.files,
            total.lines + stats.lines,
            total.comment_lines + stats.comment_lines,
            total.comment_bytes + stats.comment_bytes,
            total.over_cols + stats.over_cols,
            total.over_run + stats.over_run,
            max(total.longest, stats.longest),
            max(total.longest_run, stats.longest_run),
        )

    density = 100.0 * total.comment_lines / total.lines if total.lines else 0.0
    print(
        f"comment_lint: {total.files} files, {total.lines} lines, {total.comment_lines} comment "
        f"({density:.1f}%, {total.comment_bytes} bytes), {total.over_cols} over {MAX_COLS} cols "
        f"(longest {total.longest}), {total.over_run} multi-line (longest {total.longest_run})"
    )
    for violation in violations:
        print(violation, file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
