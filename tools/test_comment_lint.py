#!/usr/bin/env python3
"""Lexer tests for comment_lint. Run standalone; ctest registers this as style.comment_lint_selftest."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comment_lint import (  # noqa: E402
    CMAKE_QUOTES,
    MAX_COLS,
    PY_QUOTES,
    check,
    comment_spans,
    hash_comment_spans,
    main,
    scan,
    spans_for,
)

FAILURES: list[str] = []


def expect(name: str, got: object, want: object) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")


def carries(text: str) -> set[int]:
    return scan(text)[0]


def owns(text: str) -> set[int]:
    return scan(text)[1]


# A '//' inside a string literal is code: albedo_table.cpp writes exactly this when it generates albedo_table.inc.
expect("string // not a comment", carries('out << "// generated";\n'), set())
expect("string with escaped quote", carries('s = "a\\"// no";\n'), set())
expect("raw string // not a comment", carries('s = R"(// no)";\n'), set())
expect("raw string with delimiter", carries('s = R"x(// no)x";\n'), set())
expect("char literal quote", carries("c = '\\'';\n"), set())

# 1'000'000 is a digit separator, not the start of a char literal that would swallow the //.
expect("digit separator", carries("n = 1'000'000; // real\n"), {0})
expect("hex digit separator", carries("n = 0xFF'FF; // real\n"), {0})

expect("trailing comment counts", carries("code(); // why\n"), {0})
expect("trailing comment not owned", owns("code(); // why\n"), set())
expect("standalone owned", owns("// why\n"), {0})
expect("indented standalone owned", owns("    // why\n"), {0})

expect("block spans lines", carries("/* a\nb\nc */\n"), {0, 1, 2})
expect("inline block not owned", owns("code(); /* a\nb */\n"), set())

# A /*name=*/ label has code after it on the same line, so it is an inline annotation and must not stack into a block.
expect("argument label not owned", owns("f(/*a=*/1,\n  /*b=*/2);\n"), set())
expect("standalone block owned", owns("/* a\nb */\n"), {0, 1})

# A backslash at end of line splices the next line into the same // comment.
expect("continued line comment", carries("// a\\\nb\nc();\n"), {0, 1})
expect("uncontinued line comment", carries("// a\nb();\n"), {0})

expect("comment after string", carries('s = "x"; // why\n'), {0})

# A backslash-newline splices a literal onto the next line; miss it and every later comment reports one line early.
expect("string splice keeps line count", carries('s = "a\\\nb";\n// why\n'), {2})
expect("char splice keeps line count", carries("c = 'a\\\n'; // why\n"), {1})
expect("two splices keep line count", carries('s = "a\\\nb\\\nc";\n// why\n'), {3})
expect("slash divide not comment", carries("a = b / c;\n"), set())


def check_text(text: str) -> tuple[int, int]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.cpp"
        path.write_text(text, encoding="utf-8")
        found, stats = check(path)
        return len(found), stats.comment_lines


expect("over-column flagged", check_text("// " + "x" * MAX_COLS + "\n")[0], 1)
expect("at-budget accepted", check_text("//" + "x" * (MAX_COLS - 2) + "\n")[0], 0)
expect("one line accepted", check_text("// a\n")[0], 0)
expect("two lines flagged", check_text("// a\n// b\n")[0], 1)
expect("run flagged once, not per line", check_text("// a\n// b\n// c\n")[0], 1)
expect("lines split by code accepted", check_text("// a\ncode();\n// b\n")[0], 0)
expect("trailing comment does not extend a run", check_text("// a\ncode(); // b\n")[0], 0)
expect("four argument labels accepted", check_text("f(/*a=*/1,\n/*b=*/2,\n/*c=*/3,\n/*d=*/4);\n")[0], 0)
expect("no trailing newline", check_text("// a")[1], 1)

# --- hash-comment languages: CMake, Python, and the YAML-style tool configs ---


def hash_carries(text: str, quotes: tuple[str, ...] = PY_QUOTES) -> set[int]:
    return scan(text, lambda t: hash_comment_spans(t, quotes))[0]


expect("hash comment found", hash_carries("# why\n"), {0})
expect("trailing hash comment found", hash_carries("x = 1  # why\n"), {0})
expect("hash in a double-quoted string is not a comment", hash_carries('s = "a # b"\n'), set())
expect("hash in a single-quoted string is not a comment", hash_carries("s = 'a # b'\n"), set())
expect("hash in a docstring is not a comment", hash_carries('"""a # b"""\n'), set())
expect("docstring spanning lines hides its hashes", hash_carries('"""\n# a\n# b\n"""\n# real\n'), {4})

# cmake/GitSha.cmake writes "#pragma once\n#define ..." into a generated header; a regex would score those as its own.
CMAKE_TRAP = 'file(WRITE "${OUT}" "#pragma once\\n#define X \\"${sha}\\"\\n")\n# real\n'
expect("cmake generated-string hashes are not comments", hash_carries(CMAKE_TRAP, CMAKE_QUOTES), {1})

# CMake has no single-quoted string, so an apostrophe must not swallow the rest of the file.
expect("cmake apostrophe opens no string", hash_carries("# it's fine\n# second\n", CMAKE_QUOTES), {0, 1})


def hash_runs(text: str, quotes: tuple[str, ...] = PY_QUOTES) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t"
        path.write_text(text, encoding="utf-8")
        return len(check(path, spans=lambda t: hash_comment_spans(t, quotes))[0])


expect("one hash line accepted", hash_runs("# a\n"), 0)
expect("two hash lines flagged", hash_runs("# a\n# b\n"), 1)
expect("hash lines split by code accepted", hash_runs("# a\ncode()\n# b\n"), 0)
expect("trailing hash does not extend a run", hash_runs("# a\ncode()  # b\n"), 0)
expect("over-column hash flagged", hash_runs("# " + "x" * MAX_COLS + "\n"), 1)

# Language dispatch: every linted family resolves, and the deliberate exclusions resolve to None.
expect("cpp dispatches to slash", spans_for(Path("a.cpp")) is comment_spans, True)
expect("objcpp dispatches to slash", spans_for(Path("a.mm")) is comment_spans, True)
expect("glsl frag dispatches to slash", spans_for(Path("a.frag")) is comment_spans, True)
expect("glsl vert dispatches to slash", spans_for(Path("a.vert")) is comment_spans, True)
expect("CMakeLists dispatches to hash", spans_for(Path("CMakeLists.txt")) is not None, True)
expect("cmake module dispatches to hash", spans_for(Path("a.cmake")) is not None, True)
expect("python dispatches to hash", spans_for(Path("a.py")) is not None, True)
expect("clang-tidy dispatches to hash", spans_for(Path(".clang-tidy")) is not None, True)
expect("cppcheck suppressions dispatch to hash", spans_for(Path(".cppcheck-suppressions")) is not None, True)
expect("markdown is not linted", spans_for(Path("a.md")), None)
expect("generated inc is not linted", spans_for(Path("albedo_table.inc")), None)
expect("json is not linted", spans_for(Path("a.json")), None)

# The dead-gate guard: a root with no sources must fail loudly, not report a clean tree.
with tempfile.TemporaryDirectory() as empty:
    argv = sys.argv
    sys.argv = ["comment_lint", "--root", empty]
    expect("empty root exits 2", main(), 2)
    sys.argv = argv

if FAILURES:
    for failure in FAILURES:
        print(f"FAIL {failure}", file=sys.stderr)
    sys.exit(1)
print("comment_lint selftest: all checks passed")
