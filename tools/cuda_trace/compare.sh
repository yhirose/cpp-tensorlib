#!/bin/bash
# Did a change to the backend's host side alter what reaches the driver?
#
#   tools/cuda_trace/compare.sh [base-ref]      (default: origin/master)
#
# Traces `base-ref` and the working tree with the same tooling (this
# directory's, so a base that predates it still traces) and diffs them. No
# output and exit 0 means every kernel still gets the same arguments in the
# same order. A diff is not a failure by itself — a fix is supposed to change
# the trace — but every line of it should be one you meant.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
base=${1:-origin/master}

work=$(mktemp -d "${TMPDIR:-/tmp}/cuda_trace.XXXXXX")
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/base_tree" "$work/base" "$work/head"
git -C "$repo" archive "$base" | tar -x -C "$work/base_tree"

"$here/run.sh" "$work/base" "$work/base_tree"
"$here/run.sh" "$work/head" "$repo"

status=0
for t in "$work"/head/*.trace; do
  name=$(basename "$t")
  if [ ! -f "$work/base/$name" ]; then
    echo "== $name: new in the working tree, nothing to compare"
    continue
  fi
  if ! diff -u "$work/base/$name" "$t" > "$work/$name.diff"; then
    status=1
    echo "== $name: $(grep -c '^[-+][^-+]' "$work/$name.diff") changed lines"
    head -40 "$work/$name.diff"
  fi
done
[ "$status" -eq 0 ] && echo "cuda_trace: identical to $base"
exit "$status"
