#!/usr/bin/env python3
"""Error checks that a NaN slips through, in the tests and the checks (test/, bench/).

    tools/check_nan_blind.py

Every comparison with a NaN is false, so an error check written as the test
that fails passes a NaN: `if (fabs(a - b) > tol) fail` never fails on one, and
`worst = std::max(worst, fabs(a - b))` drops it (std::max(x, NaN) is x); nor does
`if (!(e <= worst)) worst = e` keep one, the next error replacing it. Write the
test that passes, `if (!(fabs(a - b) <= tol)) fail`, as tl::allclose does, and
keep a running worst with tl::max_err. This flags a |difference| (fabs / abs of
an expression with a minus in it) compared with `>` / `>=` or inside std::max /
std::fmax, and that last form. It cannot see std::max taken of an error computed
the statement before (`worst = std::max(worst, err)`): the name alone does not
say it is one. Exit 1 when it finds any.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ["test", "bench"]
ABS = re.compile(r"\b(?:std::)?f?abs\(")


def close_paren(s, i):
    """The index just past the ')' matching the '(' at s[i - 1]."""
    depth = 1
    while i < len(s) and depth:
        depth += {"(": 1, ")": -1}.get(s[i], 0)
        i += 1
    return i


def differences(s):
    """(start, end) of each fabs / abs call on an expression with a minus."""
    for m in ABS.finditer(s):
        end = close_paren(s, m.end())
        if re.search(r"[\w\])]\s*-\s*[\w(.]", s[m.end():end - 1]):
            yield m.start(), end


def code(text):
    """text with its comments and string / char literals blanked (newlines kept)."""
    blank = lambda m: re.sub(r"[^\n]", " ", m.group(0))
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'',
                  blank, text, flags=re.S)


def statements(text):
    """(offset, text) of each statement: split at ; { } outside parentheses, so
    that a call broken over lines, or holding a brace list, is seen whole."""
    depth, start = 0, 0
    for i, c in enumerate(text):
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c in ";{}" and depth == 0:
            yield start, text[start:i]
            start = i + 1
    yield start, text[start:]


def findings(text):
    for at, s in statements(text):
        line = text.count("\n", 0, at + len(s) - len(s.lstrip())) + 1
        for start, end in differences(s):
            if re.match(r"\s*>", s[end:]):
                yield line, "a |difference| tested with '>' (passes a NaN)"
            for mx in re.finditer(r"\bstd::f?max\s*(?:<[^<>()]*>)?\s*\(", s):
                if mx.start() < start and close_paren(s, mx.end()) >= end:
                    yield line, "a |difference| through std::max (drops a NaN)"
        if re.search(r"if\s*\(\s*!\s*\(\s*(\w+)\s*<=\s*(\w+)\s*\)\s*\)\s*\2\s*=\s*\1\b", s):
            yield line, "a running worst the next error replaces a NaN in (use tl::max_err)"


def main():
    bad = 0
    for d in DIRS:
        for base, _, files in os.walk(os.path.join(ROOT, d)):
            for f in sorted(files):
                if not f.endswith((".cpp", ".h", ".cu", ".cuh")):
                    continue
                path = os.path.join(base, f)
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                for line, what in sorted(set(findings(code(text)))):
                    print(f"{os.path.relpath(path, ROOT)}:{line}: {what}")
                    bad += 1
    print(f"{bad} NaN-blind error check(s)" if bad else "no NaN-blind error checks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
