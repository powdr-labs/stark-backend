#!/bin/bash
# Build combined_metrics.json in each step-* directory using powdr's
# basic_metrics.py `combine` subcommand. Every combined file includes the
# baseline APCs (labelled baseline_apc{000,100,300}); non-baseline steps
# additionally include their own three APCs (labelled <step>_apc{000,100,300}).
# For the baseline step, the combined file is just the three baseline entries.
#
# basic_metrics.py uses the filename stem as the experiment label, so we
# stage symlinks with unique names in a tmp directory per step.

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASIC_METRICS=/home/georg/powdr/openvm-riscv/scripts/basic_metrics.py
BASELINE="$HERE/step-00-baseline"

for step_dir in "$HERE"/step-*; do
    step=$(basename "$step_dir")
    stage=$(mktemp -d)
    args=()
    for apc in 000 100 300; do
        ln -s "$BASELINE/apc${apc}.json" "$stage/baseline_apc${apc}.json"
        args+=("$stage/baseline_apc${apc}.json")
    done
    if [ "$step" != "step-00-baseline" ]; then
        for apc in 000 100 300; do
            ln -s "$step_dir/apc${apc}.json" "$stage/${step}_apc${apc}.json"
            args+=("$stage/${step}_apc${apc}.json")
        done
    fi
    python3 "$BASIC_METRICS" combine "${args[@]}" > "$step_dir/combined_metrics.json"
    rm -rf "$stage"
    echo "$step_dir/combined_metrics.json"
done
