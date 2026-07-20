#!/usr/bin/env bash
# Build the browser test harness for the WebGPU backend.
#
#   ./build.sh && python3 -m http.server 8731
#   # open http://localhost:8731/test/wasm/site/index.html
#
# A flat emcc line, not CMake — which is why the WGSL .inc is committed rather
# than generated at build time (see kernels/gen_wgsl_inc.sh).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/site"

source "${EMSDK_DIR:-$HOME/Projects/emsdk}/emsdk_env.sh" >/dev/null 2>&1

mkdir -p "$OUT"

# JSPI over Asyncify: smaller, and no conflict with -fwasm-exceptions (which
# culebra's interpreter needs). The cost is that this wasm *requires* JSPI, so
# it runs on Chrome only — acceptable for a test harness, and the shipping
# playground pairs it with a CPU build behind a feature check.
set -x
emcc -std=c++23 -O2 -fwasm-exceptions \
  --use-port=emdawnwebgpu \
  -DTENSORLIB_WEBGPU \
  -sJSPI=1 -sJSPI_EXPORTS=run_tests \
  -sSTACK_SIZE=16MB -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createTests \
  -sENVIRONMENT=web,worker \
  -sEXPORTED_FUNCTIONS=_run_tests,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,UTF8ToString \
  -I"$ROOT/include" -I"$ROOT/kernels" -I"$ROOT/test" \
  "$HERE/main_wasm.cpp" "$ROOT/test/test_array.cpp" \
  -o "$OUT/tests.js"
set +x

cp "$HERE/index.html" "$HERE/worker.js" "$OUT/"
echo "wasm size: $(stat -f%z "$OUT/tests.wasm") bytes"
