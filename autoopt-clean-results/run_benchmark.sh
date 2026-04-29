#!/bin/bash
# Run the pairing benchmark for one commit. Skips nsys profiling.
# Usage: run_benchmark.sh <tag_name>
# Assumes:
# - stark-backend is already on the target commit
# - powdr is checked out at /home/georg/powdr
# - guest artifacts already compiled at /home/georg/powdr/results/pairing/apc{000,100,300}.cbor
# Outputs:
# - Per-APC metrics to /home/georg/stark-backend/autoopt-clean-results/$TAG/apc{000,100,300}.json
#
# For the actual measurement runs we wrote output to /tmp/autoopt-measurements
# instead so that git checkouts between steps do not touch the output files;
# the results were copied into this tree after all runs completed.

set -e
TAG="$1"
if [ -z "$TAG" ]; then
    echo "Usage: $0 <tag>"
    exit 1
fi

OUT_DIR="/home/georg/stark-backend/autoopt-clean-results/$TAG"
mkdir -p "$OUT_DIR"

cd /home/georg/powdr

# Use a 16 MiB VPMM page size (8x the RTX 4090's 2 MiB minimum granularity).
# This replaces the former first commit of the chain (which changed the
# in-code default). With the env var set, the env-var path in
# VpmmConfig::from_env() supplies the same value and the built-in default
# is never reached.
export VPMM_PAGE_SIZE=16777216

echo "==== Building powdr_openvm_riscv for tag=$TAG (VPMM_PAGE_SIZE=$VPMM_PAGE_SIZE) ===="
cargo build --bin powdr_openvm_riscv -r --features "metrics,cuda" 2>&1 | tail -3
PROVE_BIN="$(cargo metadata --format-version 1 --no-deps 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["target_directory"])')/release/powdr_openvm_riscv"
echo "Binary: $PROVE_BIN"

for APC in 000 100 300; do
    CBOR="/home/georg/powdr/results/pairing/apc${APC}.cbor"
    OUT="$OUT_DIR/apc${APC}.json"
    if [ ! -f "$CBOR" ]; then
        echo "ERROR: $CBOR missing"
        exit 1
    fi
    echo "==== Running APC=$APC ===="
    "$PROVE_BIN" prove --artifact "$CBOR" --input "0" --metrics "$OUT" --recursion 2>&1 | tail -5
    echo "  -> $OUT"
done

echo "==== Done tag=$TAG ===="
