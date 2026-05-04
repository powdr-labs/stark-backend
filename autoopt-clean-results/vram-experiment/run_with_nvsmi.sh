#!/bin/bash
# Run pairing prove and capture nvidia-smi memory.used at 50ms intervals.
# Usage: run_with_nvsmi.sh <out_label>
#   <out_label> e.g. "step-00-apc100" -> writes vram-experiment/nvsmi-<label>.csv
# Assumes:
# - powdr_openvm_riscv already built (release, metrics+cuda features)
# - openvm checked out at /home/georg/openvm on georgwiese/columns-air-trait
# - powdr checked out at /home/georg/powdr with [patch] blocks active
# - stark-backend on the desired commit
# - guest artifact at /home/georg/powdr/results/pairing/apc100.cbor

set -e
LABEL="$1"
APC="${2:-100}"
if [ -z "$LABEL" ]; then
    echo "Usage: $0 <label> [apc=100]"
    exit 1
fi

OUT_DIR="/home/georg/stark-backend/autoopt-clean-results/vram-experiment"
NVSMI_OUT="$OUT_DIR/nvsmi-${LABEL}.csv"
METRICS_OUT="$OUT_DIR/metrics-${LABEL}.json"
CBOR="/home/georg/powdr/results/pairing/apc${APC}.cbor"

if [ ! -f "$CBOR" ]; then
    echo "ERROR: $CBOR missing"
    exit 1
fi

PROVE_BIN="/home/georg/powdr/target/release/powdr_openvm_riscv"
if [ ! -x "$PROVE_BIN" ]; then
    echo "ERROR: $PROVE_BIN missing — build powdr_openvm_riscv first"
    exit 1
fi

# Wait for GPU to fully drain before starting the trace, so the baseline is clean.
echo "Waiting for GPU to drain..."
for i in $(seq 1 30); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$USED" -lt 200 ]; then
        echo "  GPU at ${USED} MiB"
        break
    fi
    echo "  GPU at ${USED} MiB, waiting..."
    sleep 1
done

echo "==== Starting nvidia-smi polling (50ms) -> $NVSMI_OUT ===="
nvidia-smi --query-gpu=timestamp,memory.used --format=csv -lms 50 -i 0 > "$NVSMI_OUT" &
NVSMI_PID=$!

# Brief delay so nvidia-smi captures pre-prove baseline.
sleep 1

echo "==== Running prove on APC=$APC ===="
cd /home/georg/powdr
VPMM_PAGE_SIZE=16777216 "$PROVE_BIN" prove --artifact "$CBOR" --input "0" --metrics "$METRICS_OUT" --recursion 2>&1 | tail -5

# Brief delay so nvidia-smi captures post-prove fall-off.
sleep 1

echo "==== Stopping nvidia-smi ===="
kill $NVSMI_PID 2>/dev/null || true
wait $NVSMI_PID 2>/dev/null || true

echo "==== Wrote ===="
echo "  $NVSMI_OUT"
echo "  $METRICS_OUT"
wc -l "$NVSMI_OUT"

PEAK_MIB=$(awk -F',' 'NR>1 {gsub(/[^0-9]/,"",$2); if ($2+0 > m) m=$2+0} END {print m}' "$NVSMI_OUT")
echo "Peak (nvidia-smi): ${PEAK_MIB} MiB"
