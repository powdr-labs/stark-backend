# Report: LogUp Rules Caching (007-logup-rules-caching)

## Idea

Cache the Round 0 logup DAG (SymbolicRulesGpu + interaction mappings) at keygen time. This avoids rebuilding the DAG per-AIR during every proving call.

## Implementation

Moved the construction of `SymbolicRulesGpu` and associated interaction mappings into the keygen phase, storing them in the proving key. During proving, the cached structures are read directly instead of being rebuilt.

Files changed:
- `crates/cuda-backend/src/pkey.rs`
- `crates/cuda-backend/src/logup_zerocheck/round0.rs`
- `crates/cuda-backend/src/logup_zerocheck/mod.rs`

## Results

| Phase      | APC 0 | APC 100 | APC 300 |
|------------|-------|---------|---------|
| **Before** Round 0 | -     | -       | ~222ms  |
| **After** Round 0  | -     | -       | ~220ms  |

APC300 Round 0 improved by ~2ms. The improvement is hidden by parallel stream overlap.

## Future Work

- The caching pattern could be extended to other per-AIR structures that are invariant across proving calls.
- Benefit grows with repeated proving (e.g., recursive proof generation) where keygen is amortized.
