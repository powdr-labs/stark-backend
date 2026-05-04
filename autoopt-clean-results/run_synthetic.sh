#!/bin/bash
# Run the synthetic stark-backend benchmark for one commit.
# Usage: run_synthetic.sh <tag>
# Assumes:
# - stark-backend is already on the target commit
# Outputs:
# - Three measurement runs to <repo>/autoopt-clean-results/synthetic/$TAG/run-{1,2,3}.json
#
# For the actual measurement runs we wrote output to /tmp/autoopt-synthetic-results
# instead so that git checkouts between steps do not touch the output files;
# the results were copied into this tree after all runs completed.

set -euo pipefail
TAG="$1"
if [ -z "$TAG" ]; then
    echo "Usage: $0 <tag>"
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO/autoopt-clean-results/synthetic/$TAG"
mkdir -p "$OUT_DIR"

cd "$REPO"

# Use a 16 MiB VPMM page size (8x the RTX 4090's 2 MiB minimum granularity).
# Same setting as the pairing benchmark above.
export VPMM_PAGE_SIZE=16777216

echo "==== Building openvm-benchmark-proving for $TAG (VPMM_PAGE_SIZE=$VPMM_PAGE_SIZE) ===="
cargo build -p openvm-benchmark-proving --release --features cuda 2>&1 | tail -3
BIN="$(cargo metadata --format-version 1 --no-deps 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["target_directory"])')/release/openvm-benchmark-proving"
echo "Binary: $BIN"

ARGS=(--num-airs 311 --cols-per-air 171 --constraints-per-col 0.5 --interactions-per-col 0.6 --log-rows-per-air 12)

# Warmup (discarded)
echo "==== Warmup (discarded) ===="
"$BIN" "${ARGS[@]}" > /dev/null 2>&1

# Three measurement runs
for N in 1 2 3; do
    echo "==== Measurement run $N ===="
    METRICS_OUTPUT="$OUT_DIR/run-${N}.json" "$BIN" "${ARGS[@]}" 2>&1 | grep -E "Done proving|stark_prove|^Proving" | tail -3
    echo "  -> $OUT_DIR/run-${N}.json"
done

echo "==== Done $TAG ===="
