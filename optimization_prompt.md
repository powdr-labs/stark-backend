# Openvm v2 GPU prover optimizations for autoprecompiles

We’re analyzing OpenVM v2 with autoprecompiles, which got better for recursion but worse in app proofs. We expect app proof STARK time excl trace to roughly follow the savings in trace cells, bus interaction messages and constraint instances. This happens with the CPU prover, but not with the GPU prover. Our task here is to improve the GPU prover.

Relevant repos and branches:
stark-backend: https://github.com/powdr-labs/stark-backend, branch v2-powdr-07-04
openvm: https://github.com/powdr-labs/openvm, branch v2-powdr-07-04
powdr: https://github.com/powdr-labs/powdr, branch openvm-v2-integration-07-04. 
The powdr repo’s Cargo.toml uses local patches for openvm and stark-backend, so you need to check out all of them.

Previous experiments:
https://gist.github.com/leonardoalt/cb8fba32b08669fc63dea4d7ab0b67af. Make sure you read it in full and explore the linked resources (e.g. the paper).
nsight profiles from different runs are available for analysis here in results/pairing/apc*.
Previous results like metrics.json for apc 0, 300 are already collected here in results/pairing/apc*.

## How to run experiments

In the powdr repo given above, run script `openvm-riscv/scripts/run_pairing.sh` for apcs {0, 100, 300}. The results will be in `results/pairing`, and each config will have a `metrics.json` file. The script first compiles the guest, generates autoprecompiles, and saves them in a cbor artifact. We want to do this just once because apc gen is slow, and we want to be able to test many prover changes without having to re-compile autoprecompiles.

You should download and use https://raw.githubusercontent.com/powdr-labs/powdr/refs/heads/main/openvm/metrics-viewer/spec.py to analyze the metrics files, which gives you a fine-granular breakdown of the proof time.

An example run would be:
```
$ python spec.py https://gist.githubusercontent.com/leonardoalt/f2b810eaf2d5d37491da491f496639d1/raw/551f8590bd372dc909db3d3da2235b6a8aff70d0/metrics_pairing_v2_combined.json metrics_pairing_v2_apc300

Experiment: metrics_pairing_v2_apc300  (OpenVM 2)

  App Proof Basic Stats
  ──────────────────────────────────────────────────────────
  Segments                  2
  AIR Instances             623
  Columns                   106,842
  Cells                     811.18M (811,182,720)
  Constraints               54,961
  Constraint Instances      505.24M (505,240,712)
  Bus Interactions          67,357
  Bus Interaction Messages  448.46M (448,457,256)

  Proof Time
  ──────────────────────────────────────────────────────────
  Metered Execution        0.67s (669 ms)  (  8.6%)
  App Proof Time           5.90s (5896 ms)  ( 76.1%)
    STARK (excl. trace)    2.58s (2579 ms)  ( 33.3%)
      Constraints          1.63s (1628 ms)  ( 21.0%)
        LogUp GKR          0.78s (781 ms)  ( 10.1%)
        Round 0            0.66s (662 ms)  (  8.5%)
        MLE Rounds         0.18s (182 ms)  (  2.3%)
        Other              0.00s (3 ms)  (  0.0%) (residual)
      Openings             0.51s (514 ms)  (  6.6%)
        WHIR               0.19s (188 ms)  (  2.4%)
        Stacked Reduction  0.33s (325 ms)  (  4.2%)
        Other              0.00s (1 ms)  (  0.0%) (residual)
      Trace Commit         0.43s (434 ms)  (  5.6%)
      Other                0.00s (3 ms)  (  0.0%) (residual)
    Preflight Execution    1.83s (1830 ms)  ( 23.6%)
    Set Initial Memory     0.20s (203 ms)  (  2.6%)
    Trace Gen              1.28s (1282 ms)  ( 16.6%)
    Other                  0.00s (2 ms)  (  0.0%) (residual)
  Leaf Recursion           0.50s (502 ms)  (  6.5%)
  Inner Recursion          0.41s (409 ms)  (  5.3%)
  Compression              0.27s (270 ms)  (  3.5%)
  ──────────────────────────────────────────────────────────
  Total                    7.75s (7746 ms)  (100.0%)
```

We would expect “STARK (excl. trace)” to scale with a combination of cells, bus interaction messages and constraint instances, which are all going down by a factor of >2x when going from 0 to 300 APCs. This is not the case in practice on GPU though.

## Goal

Our only goal is to improve metric `STARK excl trace` over apc000 as much as the trace cells, bus interaction messages and constraint instances improve in either apc100 or apc300 vs apc000 (> 2x). Do NOT worry about other proving phases like trace gen or preflight execution.

## Profiling 

The tools NVIDIA nsight and nvidia-smi tools are available for profiling if needed.

## Tools

Aside from spec.py mentioned above, download https://github.com/powdr-labs/powdr/tree/main/openvm-riscv/scripts from the powdr **main** branch. Note that these files are out-of-date on your current branch. This includes the basic_metrics.py file.

## Workflow

Run in a loop, testing one idea at a time.

1. Read the ideas.md file if it exists. If it doesn’t exist, skip to step 8.
2. Pick the first idea and generate a task name (e.g. 001-air-batching). Create a branch (starting from the initial branches mentioned in the beginning) and directory (not committed!) with the same name. Make the the branch doesn't exist already.
3. Collect all relevant context for the task, including following links to previous reports, reading the relevant pieces of the code, etc.
4. Create a file called <task dir>/plan.md with a detailed plan.
5. Launch a subagent to review the plan, following the review_guideline.md. Address the review by updating the plan. Repeat this process until the reviewer approves the plan.
6. Read the final plan and do the task. Make sure to use the profiler to confirm your changes have the intended effect, and regenerate metrics files for the APC 0, 100 and 300 case. Use spec.py to see whether and by how much you sped up the metrics. DO NOT change the verifier, and test that your proof still verifies. Iterate until you are convinced that the idea was fully implemented and you can see the effect of your changes in the profiling data and metrics.
7. In <task dir>, add the following files:
  - Metrics files before the change: before_apc{000,100,300}.json
  - Metrics files after the change: after_apc{000,100,300}.json
  - A combined metrics file, created using the basic_metrics.py script: metrics_combined.json
  - A COPY of the results dir, generated by run_pairing.sh.
  - Relevant profiling files.
  - A report.md file, containing:
    - A description of the overall idea.
    - A description of the implementation
    - Results. Add a table of the relevant proving phases (as generated by spec.py) with comparison between baseline and your changes, for APC 0, 100, and 300. Show the factor of increase / decrease for each phase, relative to the same number of APCs before the optimization, and relative to APC=0 with the optimization.
    - Future work: Reflect on what worked well and how the idea could be improved, whether the idea could be combined with other ideas and whether working on this task gave you new ideas.
8. Iterate on the ideas.md file. Remove the idea that you implemented and move it to the “Done” section (with a mention of the task name). If you have new ideas, add them to the list. To generate ideas, explore what has been tried so far. Ideas should first be tested in isolation, but if you believe that two ideas could compound, add the combination of two ideas as a new ideas. Sort ideas by priority, by what you believe will yield the best results.

## DO NOT

- Do not change the underlying cryptographic protocol.
- In particular, the verifier should not be changed.

## Existing attempts

We have done previous iterations of this optimization. Before making a plan, make sure you read these reports, follow the relevant links and use the learnings to inform your decisions:
- https://gist.github.com/leonardoalt/836ab633e3bb005da8ef96e3b3c68b69
- https://gist.github.com/pacheco/57b8e21a57fe650e367c689225b6919d

## Ideas do try

The following are some vague ideas. Most of these have been attempted before, see the previous section for results. You can and should re-implement the most promising ideas though.

### AIR batching in GPU kernels

Kernel logup_batch_mle_kernel does batching of AIRs to optimize their GPU usage. There are other AIRs that could benefit from it, namely:
zero_check_ntt_evaluate_constraints_kernel and/or zero_check_ntt_evaluate_constraints_coset_parallel_kernel.
evaluate_interactions_gkr_kernel.
logup_r0_ntt_eval_interactions_coset_parallel_kernel.

Implement batching for the kernels above, so that small AIRs are combined to use the GPU optimally. Use logup_batch_mle_kernel as inspiration for the architecture and implementation.

### Run kernels in parallel

In the profile, it looks like kernels are typically run sequentially (often with a thread per row), because they are on the same CUDA stream. Investigate whether it would make sense to run several kernels in parallel in order to maximize GPU utilization.

### Play around with some protocol parameters

For example, you could see the effect of changing DEFAULT_APP_L_SKIP in crates/stark-sdk/src/config/mod.rs.

DO NOT change the parameters in a way that leads to a lower security.

### Draw inspiration from SP1

Check out the dev branch of SP1 (https://github.com/succinctlabs/sp1) and analyze their GPU prover. Compare it to OpenVM’s. Does it parallelize over the same things? Does it use different heuristics?

Write a report on your findings. If there are good ideas in SP1 that would be applicable here too, try them out.

### Break down DAGs

One thing that’s different with APCs is the increased number of constraints and bus interactions (not instances!). This leads to larger DAGs, which could be an issue. For example, check out evaluate_interactions_gkr_kernel. It can be started with GLOBAL=true or GLOBAL=false. As can be seen in the nsight systems profile, for most APCs it’s started with GLOBAL=true, which has worse performance.

Perhaps large DAGs can be broken down into smaller DAGs. Then, the GPU can parallelize over both the rows and DAG, and then combine results.

A similar idea is already implemented, where later sum-check rounds evaluate monomials in parallel, see zerocheck_monomial_kernel.

### APC-specific kernels

calculate_zero_hash only happens for APCs. Try to find out why that is and whether it can be fixed so that it costs zero during prover, like for RISC-V AIRs.

### Wider investigation

Investigate the benchmark results as well as nsight reports and check if there are new ideas that can be attempted besides the ones listed above.
