#!/usr/bin/env python3
"""Backend parity checker: which GPU ops does each backend (CUDA/Metal/
WebGPU) actually implement, vs. which are still a `return false;` stub.

The op surface it checks is the exact "kernels" / "LLM path" lists in
include/gpu.h's own doc comment -- that comment is the contract array.h/
storage.h dispatch through, and every CUDA-first landing so far has already
had to update it, so it stays in sync without this script needing its own
copy of the list.

For each op name, every backend header (cuda.h / metal.h / webgpu.h)
follows the same convention: the *first* `inline bool <name>(...)` in the
file is that backend's real-platform implementation (Apple for metal.h,
TENSORLIB_CUDA for cuda.h, TENSORLIB_WEBGPU&&__EMSCRIPTEN__ for webgpu.h);
a disabled-platform build gets a second definition later in the file with
unnamed parameters, always `return false;`. So "is this backend's first
definition's body anything other than exactly `return false;`" is a
reliable REAL/STUB signal without needing a real C++ parser.

Usage: tools/check_backend_parity.py
Exit status: 0 if every op is REAL or STUB the same way on all three
backends (a uniform stub is fine -- that just means nobody has ported it
yet); 1 if any op's status differs across backends, printing what's
missing where.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKENDS = {
    "cuda": ROOT / "include/cuda.h",
    "metal": ROOT / "include/metal.h",
    "webgpu": ROOT / "include/webgpu.h",
}


def canonical_ops():
    """Pull the op-name list straight out of gpu.h's own contract comment."""
    text = (ROOT / "include/gpu.h").read_text()
    m = re.search(
        r"^//\s*kernels\s+(.*?)^// A backend with no kernel",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not m:
        sys.exit("check_backend_parity: couldn't find gpu.h's kernels/LLM "
                 "path comment -- has its wording changed?")
    body = m.group(1)
    # Drop the leading "//" (and any inherited indentation) from every
    # wrapped comment line before splitting on "/", so the LLM path label
    # itself doesn't get treated as an identifier.
    lines = [re.sub(r"^\s*//\s*", "", ln) for ln in body.splitlines()]
    lines = [ln for ln in lines if not ln.strip().startswith("LLM path")]
    # The "LLM path" line's own identifiers are on the same physical line as
    # the label in gpu.h, so re-extract them directly instead of dropping
    # the whole line blindly.
    llm_m = re.search(r"^//\s*LLM path\s+(.*)$", text, re.MULTILINE)
    ops = []
    for ln in lines:
        ops += [tok.strip() for tok in ln.split("/") if tok.strip()]
    if llm_m:
        ops += [tok.strip() for tok in llm_m.group(1).split("/") if tok.strip()]
    # Keep discovery order but drop duplicates.
    seen = set()
    ordered = []
    for op in ops:
        if op not in seen:
            seen.add(op)
            ordered.append(op)
    return ordered


def function_body(text, name):
    """Return the first `inline bool <name>(...) { ... }` body's source, or
    None if `name` isn't defined as an `inline bool` function in `text`."""
    m = re.search(rf"^inline bool {re.escape(name)}\s*\(", text, re.MULTILINE)
    if not m:
        return None
    i = text.index("(", m.end() - 1)
    depth = 1
    i += 1
    while depth:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    while text[i] != "{":
        i += 1
    body_start = i + 1
    depth = 1
    i = body_start
    while depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[body_start:i - 1]


def status(text, name):
    body = function_body(text, name)
    if body is None:
        return "MISSING"
    return "stub" if re.sub(r"\s+", " ", body).strip() == "return false;" else "REAL"


ALLOWLIST_PATH = ROOT / "tools/backend_parity_allowlist.txt"


def allowlist():
    if not ALLOWLIST_PATH.exists():
        return set()
    lines = ALLOWLIST_PATH.read_text().splitlines()
    return {ln.split("#", 1)[0].strip() for ln in lines if ln.split("#", 1)[0].strip()}


def main():
    ops = canonical_ops()
    sources = {b: p.read_text() for b, p in BACKENDS.items()}
    rows = []
    for op in ops:
        rows.append((op, {b: status(src, op) for b, src in sources.items()}))

    names = list(BACKENDS)
    width = max(len(op) for op, _ in rows) + 2
    print(f"{'op':<{width}}" + "".join(f"{b:<10}" for b in names))
    mismatches = []
    for op, st in rows:
        print(f"{op:<{width}}" + "".join(f"{st[b]:<10}" for b in names))
        if len(set(st.values())) > 1:
            mismatches.append((op, st))

    allowed = allowlist()
    unallowed = [(op, st) for op, st in mismatches if op not in allowed]
    stale = sorted(allowed - {op for op, _ in mismatches})

    if not mismatches:
        print(f"\nbackend-parity OK ({len(ops)} ops, all backends agree)")
    else:
        print(f"\nbackend-parity: {len(mismatches)}/{len(ops)} ops differ "
              f"across backends:")
        for op, st in mismatches:
            real = [b for b in names if st[b] == "REAL"]
            behind = [b for b in names if st[b] != "REAL"]
            tag = " (allowlisted)" if op in allowed else " (NOT allowlisted)"
            print(f"  {op}: real on {real or '(none)'}, "
                  f"missing/stub on {behind}{tag}")

    if stale:
        print(f"\n{ALLOWLIST_PATH.name} lists ops that are no longer "
              f"asymmetric -- remove these lines:")
        for op in stale:
            print(f"  {op}")

    if unallowed:
        print(f"\nFAIL: {len(unallowed)} unallowlisted mismatch(es). Either "
              f"port the missing backend(s), or add the op to "
              f"{ALLOWLIST_PATH.name} with a reason if the gap is "
              f"deliberate.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
