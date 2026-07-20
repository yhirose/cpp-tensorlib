#!/usr/bin/env bash
# Build the browser harnesses for the WebGPU backend.
#
#   ./build.sh && python3 -m http.server 8731
#   # tests:  http://localhost:8731/test/wasm/site/index.html
#   # census: http://localhost:8731/test/wasm/site/census.html
#
# Two separate wasms, not one with two entry points: the test suite has to stay
# a fast pass/fail and the census runs for minutes.
#
# A flat emcc line, not CMake — which is why the WGSL .inc is committed rather
# than generated at build time (see kernels/gen_wgsl_inc.sh).
#
# Pass `tests` or `census` to build just one.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/site"
WHAT="${1:-all}"

# Local dev sources emsdk from a checkout; CI (setup-emsdk) already has emcc on
# PATH and has no emsdk_env.sh to source, so this must not be fatal there.
EMSDK_ENV="${EMSDK_DIR:-$HOME/Projects/emsdk}/emsdk_env.sh"
if [[ -f "$EMSDK_ENV" ]]; then
  source "$EMSDK_ENV" >/dev/null 2>&1
fi
command -v emcc >/dev/null || { echo "emcc not found (set EMSDK_DIR or activate emsdk)" >&2; exit 1; }

mkdir -p "$OUT"

# JSPI over Asyncify: smaller, and no conflict with -fwasm-exceptions (which
# culebra's interpreter needs). The cost is that this wasm *requires* JSPI, so
# it runs on Chrome only — acceptable for a test harness, and the shipping
# playground pairs it with a CPU build behind a feature check.
common=(
  -std=c++23 -O2 -fwasm-exceptions
  --use-port=emdawnwebgpu
  -DTENSORLIB_WEBGPU
  -sSTACK_SIZE=16MB -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB
  -sMODULARIZE=1 -sEXPORT_ES6=1
  -sENVIRONMENT=web,worker
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,UTF8ToString
  -I"$ROOT/include" -I"$ROOT/kernels" -I"$ROOT/test"
)

if [[ "$WHAT" == all || "$WHAT" == tests ]]; then
  set -x
  emcc "${common[@]}" \
    -sJSPI=1 -sJSPI_EXPORTS=run_tests \
    -sEXPORT_NAME=createTests \
    -sEXPORTED_FUNCTIONS=_run_tests,_malloc,_free \
    "$HERE/main_wasm.cpp" "$ROOT/test/test_array.cpp" \
    -o "$OUT/tests.js"
  set +x
  cp "$HERE/index.html" "$HERE/worker.js" "$OUT/"
  echo "tests.wasm:  $(stat -f%z "$OUT/tests.wasm") bytes"
fi

if [[ "$WHAT" == all || "$WHAT" == census ]]; then
  set -x
  emcc "${common[@]}" \
    -sJSPI=1 -sJSPI_EXPORTS=run_census \
    -sEXPORT_NAME=createCensus \
    -sEXPORTED_FUNCTIONS=_run_census,_malloc,_free \
    "$HERE/census_wasm.cpp" \
    -o "$OUT/census.js"
  set +x
  cp "$HERE/census.html" "$HERE/census-worker.js" "$OUT/"
  echo "census.wasm: $(stat -f%z "$OUT/census.wasm") bytes"
fi
