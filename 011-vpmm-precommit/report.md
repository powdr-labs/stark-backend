# Report: VPMM Precommit (011-vpmm-precommit)

## Idea

Pre-commit 8192 VPMM pages (~16GB) at startup to avoid `cuMemSetAccess` overhead during the first large allocations. Configurable via the `VPMM_PAGES` environment variable.

## Implementation

Added a startup phase in the VPMM virtual memory pool that eagerly commits the configured number of pages. This moves the `cuMemSetAccess` cost from the STARK proving span to initialization, where it does not affect proving latency.

Files changed:
- `crates/cuda-common/src/memory_manager/vm_pool.rs`

## Results

| Phase  | APC 0 | APC 100 | APC 300 |
|--------|-------|---------|---------|
| **Before** STARK | -     | -       | ~1568ms |
| **After** STARK  | -     | -       | ~1558ms |

APC300 STARK improved by ~10ms. The `cuMemSetAccess` cost is moved to startup rather than eliminated, so total wall-clock time is unchanged but proving latency is reduced.

## Future Work

- The default page count (8192) could be auto-tuned based on available GPU memory.
- A more aggressive strategy could pre-commit all available GPU memory and release pages on demand.
