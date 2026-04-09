# Review Guideline

## Goal

Review an implementation plan against the current codebase and existing measurements.

The job is not to paraphrase the plan. The job is to find:

- correctness holes
- missing invariants
- interface mismatches with the current code
- places where estimated wins are not supported by measurements
- places where the plan is overcomplicated relative to expected proof-time benefit
- possible simplifications

## Core standard

A good review answers four questions:

1. Is the proposal correct against the code that exists today?
2. Is it specific enough to implement a prototype without inventing missing design later?
3. Is it likely to improve the measured bottleneck it claims to target?
4. Is the proposed complexity justified by expected wall-clock benefit?

## Review method

### 1. Read the plan for concrete claims

Extract:

- the claimed bottleneck
- the proposed mechanism
- what is said to be AIR-level vs trace-level
- what is said to be reused, cached, batched, or hoisted
- expected savings and what units they use

Do not accept vague claims like “should help” or “future cleanup will handle this.”

### 2. Check the current code path end to end

Find the actual call chain in the current implementation:

- caller
- helper
- FFI boundary
- CUDA/kernel side if relevant
- postprocess / D2H / synchronization points

Verify that the plan matches the real architecture, not an assumed one.

### 3. Check measurements before trusting priorities

Use existing profiling or instrumentation first.

For each claimed win, ask:

- is this work actually inside the measured bottleneck?
- is the cited span wall-clock or only a subspan?
- is the estimate per segment, per round, or per proof?

Reject priority arguments that contradict the data.

### 4. Separate design blockers from cleanup

A design blocker is something that makes the proposal:

- incorrect
- unimplementable as written
- likely to fail at runtime
- likely to miss the intended performance target

Cleanup issues are:

- wording imprecision
- stale checklist items
- missing explicit `Default` impl mention
- test/checklist polish

Be strict about the difference.

## What to verify every time

### Correctness / architecture

- Does the proposal preserve current invariants?
- Does it account for all inputs needed to reproduce current behavior?
- Does it match current buffer layout, indexing, launch geometry, and reduction shape?
- Does it accidentally rely on information only available on the CUDA side or only on the Rust side?
- Does it handle fallback paths, not just the new fast path?

### Memory / buffers

- Is capacity confused with logical length?
- Are reused buffers read back to host with the correct live length?
- Does the plan assume async frees happen immediately?
- Is the memory budget a real bound or just a heuristic?
- If sub-batching is proposed, is the scope of reuse actually large enough to matter?

### Performance realism

- Is the plan attacking the span that is actually large in Nsight/NVTX?
- Are claimed savings plausible relative to the cited measured span?
- Is the proposal removing work, or only moving it around?
- Is the proposal saving GPU time, CPU time, runtime API overhead, or sync time? Be explicit.

### Implementation readiness

- Are required new structs/helpers/FFI changes fully specified?
- Are raw pointers, lifetimes, or ownership changes spelled out?
- Does the checklist include all touched paths?
- Are there hidden “to be solved later” gaps in core behavior?

## Preferred review output structure

Use:

1. `Findings`
2. `Improvements I would consider`
3. `Assessment`

Under `Findings`:

- order by severity
- use precise statements
- include file references
- say exactly what is wrong and what must change

Under `Improvements I would consider`:

- only include real alternatives that could better improve proof time
- distinguish these from blockers

Under `Assessment`:

- state whether there is a remaining design-level blocker
- state whether the plan is prototype-ready
- say what the strongest part of the plan is

## Writing rules

- Be concrete, not rhetorical.
- Prefer “As written, X is wrong because Y in current code does Z.”
- Do not say “looks good overall” before findings.
- Do not invent performance numbers.
- Do not call something “overengineering” unless you explain what simpler path exists.
- Do not let “future work” hide a present correctness or implementation gap.
- If the proposal is good, say so plainly after checking it.

## Useful review heuristics

- Cached data must include every static input needed to rebuild current behavior.
- Reuse scoped inside a function call does not help if the costly repetition happens across calls.
- Exact-size host copies matter when buffer APIs do not track logical length separately.
- A plan that ignores fallback paths is usually incomplete.
- A measured `~100us` span is not a serious optimization target inside a `~10ms+` phase.
- The cleanest prototype path is usually the one that matches an existing pattern already used elsewhere in the codebase.

## Minimum bar before approving a prototype plan

Approve only if all of these are true:

- no remaining design-level correctness blocker
- no missing data needed to reproduce current behavior
- no hidden interface gap between Rust and CUDA
- the intended win targets a measured bottleneck
- the checklist is sufficient for someone else to implement the prototype without guessing