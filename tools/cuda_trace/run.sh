#!/bin/bash
# Trace what a tensorlib tree's CUDA backend asks of the driver, with no GPU:
#
#   tools/cuda_trace/run.sh <out-dir> [tree]
#
# `tree` defaults to the checkout this script lives in. The test suite and the
# CUDA checkers are built on Linux in a container against a recording driver
# (fake_libcuda.cpp), and each writes <out-dir>/<program>.trace.
# TL_CUDA_TRACE_CHECK=1 stops after compiling: does the CUDA branch still build?
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: run.sh <out-dir> [tree]}
tree=$(cd "${2:-$here/../..}" && pwd)
mkdir -p "$out"
out=$(cd "$out" && pwd)

image=tensorlib-cuda-trace
if ! docker image inspect "$image" > /dev/null 2>&1; then
  docker build -q -t "$image" "$here" > /dev/null
fi
docker run --rm -e TL_CUDA_TRACE_CHECK="${TL_CUDA_TRACE_CHECK:-0}" -v "$here":/tools:ro -v "$tree":/tree:ro -v "$out":/out \
  "$image" bash /tools/in_container.sh
