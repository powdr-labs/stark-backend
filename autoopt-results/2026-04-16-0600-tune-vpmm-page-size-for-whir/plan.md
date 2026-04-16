# Plan: Tune VPMM Page Size for WHIR

## Goal

Find the optimal VPMM page size that preserves the -186ms improvement from the 2→16 MiB page size change while recovering the 28ms WHIR regression. The 16 MiB threshold routes WHIR's 2-4 MiB allocations through cudaMallocAsync instead of VPMM. An intermediate page size (e.g., 4 MiB) would keep WHIR buffers in VPMM while still reducing cuMemCreate overhead for larger allocations.

## Current Code Path

### VPMM page size configuration (`crates/cuda-common/src/memory_manager/vm_pool.rs:55-65`)
```rust
None => {
    // Use 8x the minimum granularity (typically 16 MiB)
    8 * granularity
}
```

The page size defaults to 8× the CUDA minimum granularity (2 MiB × 8 = 16 MiB). Overridable via `VPMM_PAGE_SIZE` environment variable.

### Allocation routing (`crates/cuda-common/src/memory_manager/mod.rs:75-87`)
Allocations below `pool.page_size` (currently 16 MiB) → `cudaMallocAsync`.
Allocations at or above `pool.page_size` → VPMM (cuMemCreate + cuMemMap).

### Impact on WHIR (`crates/cuda-backend/src/whir.rs:100-175`)
Key WHIR buffers:
- `f_ple_evals`: height × D_EF × 4 bytes ≈ 2-4 MiB (goes through cudaMallocAsync at 16 MiB threshold)
- `f_coeffs`: height × 16 bytes ≈ 2-4 MiB
- `w_moments`: height × 16 bytes ≈ 2-4 MiB
- Per-round g_rs codewords, Merkle digest layers

### Previous measurements
- **2 MiB pages (baseline)**: STARK excl trace APC 300 = 2455ms, WHIR = 100ms
- **16 MiB pages (current)**: STARK excl trace APC 300 = 1110ms (-1345ms), WHIR = 128ms (+28ms)
- **decouple-vpmm-pool-threshold** (64 MiB threshold): regressed +20ms at APC 300, +51ms at APC 0
- **prealloc-whir-fold-buffers**: pre-allocated 256 MiB VPMM-routed scratch for fold, only saved 4ms

## Changes

### Step 1: Diagnostic — confirm VPMM page size as WHIR regression root cause
**No code changes.** Run benchmarks with `VPMM_PAGE_SIZE` env var:

```bash
cd /home/georg/powdr
# Test 1: Baseline page size (2 MiB) — should recover WHIR to ~100ms
VPMM_PAGE_SIZE=2097152 openvm-riscv/scripts/run_pairing.sh  # APC 0, 100, 300

# Test 2: 4 MiB pages — routes 2-4 MiB WHIR buffers through VPMM
VPMM_PAGE_SIZE=4194304 openvm-riscv/scripts/run_pairing.sh

# Test 3: 8 MiB pages — compromise
VPMM_PAGE_SIZE=8388608 openvm-riscv/scripts/run_pairing.sh
```

Record WHIR, STARK excl trace, and key component metrics for each page size at each APC config.

### Step 2: Analyze results and select optimal page size

**Decision matrix:**
| Page Size | Expected WHIR | Expected cuMemCreate | Expected Other Phases |
|-----------|---------------|---------------------|-----------------------|
| 2 MiB | ~100ms (baseline) | High (+50ms Trace Commit) | Baseline performance |
| 4 MiB | ~100-110ms? | Moderate (+25ms?) | Small improvement? |
| 8 MiB | ~110-120ms? | Low (+12ms?) | Close to current? |
| 16 MiB | 128ms (current) | Minimal (current) | Best (current) |

The optimal page size maximizes WHIR recovery while minimizing cuMemCreate regression.

**Acceptance criteria:**
- STARK excl trace at APC 300 improves by ≥ 10ms net (WHIR recovery minus any cuMemCreate regression)
- No APC 0 regression > 20ms
- No proof verification failures

### Step 3: Apply the optimal page size

**File**: `crates/cuda-common/src/memory_manager/vm_pool.rs`
**What**: Change the default page size multiplier from 8× to the optimal value found in Step 2:

```rust
None => {
    // Use Nx the minimum granularity
    N * granularity  // where N is the optimal multiplier (2, 4, or 8)
}
```

This is a ONE-LINE change (identical in nature to the original 2→16 MiB change).

## Invariants

1. **Correctness**: The VPMM page size affects only allocation routing, not computation. All proof outputs are identical regardless of page size.
2. **Backward compatibility**: The `VPMM_PAGE_SIZE` environment variable still overrides the default.
3. **Memory budget**: Smaller page sizes increase cuMemCreate calls but don't change peak memory usage.
4. **APC 0 safety**: The original 2 MiB→16 MiB change improved APC 0 by 328ms (mostly from cuMemCreate reduction). A partial revert (e.g., 4 MiB pages) may regress APC 0 by some fraction. This must be measured.

## Measurement Plan

1. **Step 1 measurements**: Run `openvm-riscv/scripts/run_pairing.sh` for APC 0, 100, 300 at each page size (2, 4, 8, current=16 MiB).
2. Use `spec.py` to analyze each metrics.json.
3. Compare WHIR and STARK excl trace across all configurations.
4. Record per-segment breakdown for detailed analysis.
5. Total: 4 page sizes × 3 APC configs = 12 benchmark runs.
6. **Optimization: skip full run_pairing.sh** — the script includes nsight profiling which is slow. For parameter sweeping, run only the `prove` step (skip nsight) to reduce measurement time.

## Rollback Criteria

- If no page size improves STARK excl trace at APC 300 by ≥ 10ms: no change (keep 16 MiB)
- If the best page size regresses APC 0 by > 20ms: no change
- If the improvement is < 10ms net but WHIR recovers significantly: document findings for future reference, still no code change
