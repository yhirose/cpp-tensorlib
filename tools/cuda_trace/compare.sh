#!/bin/bash
# Did a change to the backend's host side alter what reaches the driver?
#
#   tools/cuda_trace/compare.sh [base-ref]      (default: origin/master)
#
# Traces `base-ref` and the working tree with the same tooling (this
# directory's, so a base that predates it still traces) and diffs them. No
# output and exit 0 means every kernel still gets the same arguments in the
# same order. A diff is not a failure by itself — a fix is supposed to change
# the trace, and a new test appends to it — but every line of it should be one
# you meant. Removed lines are the ones to read first: a refactor that only
# moves code removes none.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
base=${1:-origin/master}

work=$(mktemp -d "${TMPDIR:-/tmp}/cuda_trace.XXXXXX")
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
  # The closing line counts allocations, so it moves whenever anything is added.
  if ! diff <(grep -av '^end ' "$work/base/$name") <(grep -av '^end ' "$t") \
      > "$work/$name.diff"; then
    status=1
    echo "== $name: $(grep -c '^<' "$work/$name.diff") removed, $(grep -c '^>' "$work/$name.diff") added"
    grep -a '^<' "$work/$name.diff" | head -10 || true
  fi
done
if [ "$status" -eq 0 ]; then
  echo "cuda_trace: identical to $base"
  rm -rf "$work"
else
  echo "cuda_trace: diffs kept in $work"
fi
exit "$status"
