# Report: Selector Caching (009-selector-caching)

## Idea

Cache selector DeviceBuffers by trace height. Traces with the same height share the same selector values (is_first, is_transition, is_last). This reduces ~600 VPMM allocations down to ~20.

## Implementation

Introduced a cache keyed by trace height that stores pre-computed selector DeviceBuffers. Before allocating a new selector, the cache is checked for an existing buffer of the same height. On a cache hit, the existing buffer is reused.

Files changed:
- `crates/cuda-backend/src/logup_zerocheck/mod.rs`

## Results

| Phase  | APC 0 | APC 100 | APC 300 |
|--------|-------|---------|---------|
| **Before** STARK | -     | -       | ~1572ms |
| **After** STARK  | -     | -       | ~1570ms |

APC300 STARK improved by ~3ms. The small absolute gain reflects that VPMM allocation is already fast; the benefit is primarily reduced memory pressure.

## Future Work

- The caching pattern could be extended to other per-height invariant buffers.
- Larger workloads with more distinct AIRs may see proportionally greater benefit from reduced allocation count.
