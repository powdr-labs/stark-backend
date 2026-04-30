#!/bin/bash
# Run pairing benchmark for one commit, against the post-rebase wiring:
# powdr=georgwiese/openvm-deps-update (with the --artifact CLI flag
# cherry-picked back in), openvm=georgwiese/columns-air-trait, and the
# local stark-backend (this branch). The [patch] sections in powdr's
# Cargo.toml must be uncommented so the workspace resolves both to the
# local checkouts.
#
# Usage: run_benchmark_post_rebase.sh <tag>
# Assumes:
# - stark-backend is already on the target commit
# - powdr is checked out at /home/georg/powdr on the deps-update branch
# - openvm is checked out at /home/georg/openvm on columns-air-trait
# - guest artifacts already compiled at /home/georg/powdr/results/pairing/apc{000,100,300}.cbor
# Outputs:
# - Per-APC metrics to /home/georg/stark-backend/autoopt-clean-results/pairing-post-rebase/$TAG/apc{000,100,300}.json
#
# For the actual measurement runs we wrote output to /tmp/autoopt-pairing-rebase
# instead so that git checkouts between steps do not touch the output files;
# the results were copied into this tree after all runs completed.

set -euo pipefail
TAG="$1"
if [ -z "$TAG" ]; then
    echo "Usage: $0 <tag>"
    exit 1
fi

OUT_DIR="/home/georg/stark-backend/autoopt-clean-results/pairing-post-rebase/$TAG"
mkdir -p "$OUT_DIR"

cd /home/georg/powdr

export VPMM_PAGE_SIZE=16777216

echo "==== Building powdr_openvm_riscv for $TAG (VPMM_PAGE_SIZE=$VPMM_PAGE_SIZE) ===="
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
    "$PROVE_BIN" prove --artifact "$CBOR" --input "0" --metrics "$OUT" --recursion 2>&1 | tail -3
    echo "  -> $OUT"
done

echo "==== Done $TAG ===="
