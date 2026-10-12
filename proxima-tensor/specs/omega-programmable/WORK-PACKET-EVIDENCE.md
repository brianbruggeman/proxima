# Work packet: integration and measurement

## Worker contract

The implementation packets are dependencies. Do not benchmark a config parser in
isolation or substitute an emitter string test for device execution. The root
integrator owns the combined tree; the evidence worker builds and runs it in an
isolated worktree with `CARGO_TARGET_DIR` supplied on every cargo invocation.
Use compiled commands only; no shell, Python, Perl, or other scripts. Do not
commit, delete artifacts, change model weights, or run competing GPU jobs.

## Inputs that must remain bound

- Program: the existing `Op` graph, symbolic extents, named `QuantizedBlock`
  bindings, and requested `NodeId` roots. No new graph representation.
- Performance oracle: llama.cpp Ouro revision
  `1550360e089185632c2036c908411b05fafa1119`, native harness
  `/private/tmp/proxima-ourollama-oracle-build/ouro_llama_oracle`.
- Model: `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/models/ouro-1.4b`, revision
  `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
- GGUF: `/private/tmp/ouro-llama-converted-v1/ouro-bf16.gguf`.
- Before-change Proxima evidence:
  `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/ouro-llama-oracle/results/proxima-profile-v3/independent-timing-deltas.jsonl`.
- Exact prefix IDs, run conditions, llama samples and logits:
  `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/ouro-llama-oracle/results/llama-v1/llama.jsonl`.

Open the actual records above before running. Copy their exact prefix arrays;
do not tokenize the display text again. Preserve the two prefix lengths and
four recurrent loops. Current Proxima is the before-change baseline, never
the performance oracle. llama is not automatically the numerical oracle.

## Integration edit

Own `examples/ouro_loopcd_lab/engine.rs`, `model.rs`, `performance.rs`, and the
argument loader in `examples/ouro_loopcd_lab.rs`. These files are not owned by
the config, execution, or tiled workers.

1. Add an explicit `--omega-config PATH` input to the existing argument loader.
   Read `omega::config::RuntimeConfig` once during model setup and pass it by
   reference to the existing executor preparation path. Do not read environment
   variables per layer or per token. Without a supplied config, retain the
   existing exact arm as a separately identified baseline.
2. Replace the hard-coded `plan_named_exact(..., bit_exact())` only in the
   configured arm with the execution packet's configured planner. Keep the same
   program, symbols, bindings, and roots. The plan remains the existing
   `omega::backend::Plan`; execution remains its existing function.
3. Record the resolved config and numeric permission bits in every run context.
   Replace the unconditional `numeric_policy: bit_exact ...` description when
   a different config is used. Also record the selected backend/driver, selected
   candidate, rejected candidates and reasons, source codecs, kernel identities,
   and resource reservation report. Missing facts are explicit nulls with reasons.
4. Preserve the existing statistics roots, recurrent state boundaries, and
   output readbacks. Changing those in the same comparison would conflate work
   elimination with tiled lowering. A distinct later experiment may change them.
5. The config-driven arm must be usable by other model callers through Omega's
   public planner; the Ouro caller is evidence of use, not a special dispatch.

## Experiments, in dependency order

### A. Literal matrix execution

Use the tiled packet's hand-derived integer-valued matrix fixture through the
configured planner on CPU and real Metal. Compare each output coordinate to
the literal fixture, with both the fixture itself and a one-coordinate
corruption passed through the same comparator. Save inputs, encoded BF16/F16
bytes, expected and actual outputs, selected kernel source and launch geometry.
The kernel source must contain the intended matrix operation; a feature name
alone is not evidence of dispatch.

### B. Microbench

Use the existing native `ouro_loopcd_lab --performance --microbench` branch,
sharing its resource sampler, checkpoint loader and raw statistics format.
No ignored benchmark test is added. Both arms explicitly set
`OMEGA_TILED_GEMM_MIN_TOKENS=1`; the production threshold remains 160. The
`OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE` value is the only build difference.

Run projection workloads with activation rows 1, 5, and 24 and checkpoint
projection extents taken from actual loaded metadata. Arms are existing scalar
lowering and configured tiled lowering, with identical source bytes, output
roots, and numeric permissions wherever comparing the kernel alone. Record
policy changes as their own arm. Keep host staging/compilation separate from
warm plan execution. Synchronize the GPU before stopping each timer.

Capture every duration, CPU time, peak RSS, allocation/reservation counters,
errors, output payload, and device timestamps available. Five measured samples
follow a separately recorded warmup. Report sample standard deviation
(`n-1` denominator), CoV% (`100 * SD / mean`), min/max and nearest-rank latency
percentiles. Never remove outliers. Explain unavailable counters instead of
inventing utilization percentages.

### C. Ouro end-to-end

Use the existing performance harness on both frozen prefixes. Keep baseline4,
logit contrast and hidden contrast arms and their full output payloads. Run the
new config with reassociation/contraction explicitly granted in two paired
build-gate arms: dense float storage enabled and disabled, with identical
config, threshold, inputs, and roots. This isolates float-storage lowering
from arithmetic-policy and token-threshold changes. Each arm uses the same
prefix/method matrix and records its output under distinct result directories.
Exact and relaxed arithmetic results must have different condition labels.
Compare fresh-prefix forward latency with the same pinned llama oracle, whose
arithmetic and last-layer work differences are already documented in the oracle
evidence.

Collect signed per-coordinate deltas in both directions; retain all top-token
changes, nonfinite results, failed samples and their causes. A matching argmax
does not discharge the vector comparison. A speed measurement does not
discharge the numerical gate. Never relabel relaxed output as bit exact.

### D. Negative and compatibility controls

- Denied reassociation or contraction cannot select the new matrix path.
- Lowering disabled at compile time retains the existing source path.
- An impossible resource cap fails before the allocation/dispatch it forbids.
- A missing driver returns a typed error and produces no CPU fallback sample.
- Reuse with different input bytes re-executes; a plan cache cannot return old
  output. Shape/codec/policy changes cannot reuse incompatible compiled kernels.
- Run the existing Apple LoopCD hand-derived controls through both accepted
  configured paths; preserve same-prefix/completed-loop semantics.

## Artifact schema and review

Every row includes source file/record ID, checkpoint/hash, prefix IDs, roots,
depth, arm, full config, backend/device/driver identity, build features and source
revision, setup/warmup/timed classification, raw duration, synchronization
boundary, output artifact path, error and resource fields. Derived rows point
to every raw input record and record their formula.

Write `results/` payloads plus `EVIDENCE.md` and a discipline log. Root must open
the output payloads, emitted kernels, launch records and actual diff before
reporting observations. Reports say what ran and what was observed; they do not
declare universal optimality, 100% utilization, numerical parity, or a speedup
without bound conditions. Retain every failed and rolled-back experiment.

## Failure handling

A zero-test filter, dropped telemetry event, missing synchronization, changed
prefix/model hash, fallback execution, mislabeled policy, or output file with
missing samples invalidates that measurement cohort. Preserve it with the exact
reason, repair the cause, then rerun the complete affected cohort. Do not fix
an error by loosening tolerances, omitting the slow arm, or averaging away the
payload that produced it.
