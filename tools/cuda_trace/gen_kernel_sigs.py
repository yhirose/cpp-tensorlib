#!/usr/bin/env python3
"""Every __global__ kernel's argument layout, read from the .cu itself.

cuLaunchKernel receives `void** argv` and nothing else, so the fake driver
cannot tell a pointer slot from a 4-byte one without knowing the kernel. This
runs the C preprocessor over kernels/tensorlib_cuda.cu (so macro-generated
kernels appear under their real names) and prints one table row a kernel:

    {"tl_add", "PPPuff"},

P = pointer, u = 4-byte integer, f = float. Anything else is an error: the
launch contract is "a pointer or a 4-byte scalar", and a kernel that breaks it
should fail here, loudly, not shift its arguments at run time.
"""
import re
import shutil
import subprocess
import sys

INT4 = {"unsigned", "int", "unsigned int", "uint32_t", "int32_t"}


def preprocess(path):
    src = open(path).read()
    # The only #include is a CUDA header this host does not have; nothing in a
    # signature needs it, and g++ stops preprocessing at a missing include.
    src = re.sub(r"^\s*#\s*include[^\n]*$", "", src, flags=re.M)
    cc = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
    if not cc:
        sys.exit("gen_kernel_sigs: no C++ compiler to preprocess with")
    out = subprocess.run([cc, "-E", "-P", "-x", "c++", "-"], input=src,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("gen_kernel_sigs: preprocessing failed:\n" + out.stderr)
    return out.stdout


def params_of(text, open_paren):
    depth = 0
    for j in range(open_paren, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1:j]
    sys.exit("gen_kernel_sigs: unbalanced parameter list")


def layout(name, params):
    sig = ""
    for p in params.split(","):
        p = " ".join(p.split())
        if not p:
            continue
        if "*" in p:
            sig += "P"
            continue
        ty = " ".join(w for w in p.split()[:-1] if w != "const")
        if ty in INT4:
            sig += "u"
        elif ty == "float":
            sig += "f"
        else:
            sys.exit(f"gen_kernel_sigs: {name}: parameter `{p}` is neither a "
                     "pointer nor a 4-byte scalar")
    return sig


def main():
    text = preprocess(sys.argv[1])
    rows = {}
    for m in re.finditer(r"__global__\s+void\s+(\w+)\s*\(", text):
        rows[m.group(1)] = layout(m.group(1), params_of(text, m.end() - 1))
    if not rows:
        sys.exit("gen_kernel_sigs: no kernels found")
    for name in sorted(rows):
        print(f'{{"{name}", "{rows[name]}"}},')


if __name__ == "__main__":
    main()
