# packet: resumable streams, bounded batching and fairness

**Route:** GPT-6 Sol (preselected). Splitting the current decode loop changes
the state/ownership boundary for KV, sampler/RNG, cancellation and command
buffer lifetime. Use Luna only for mechanical fixture expansion once Sol fixes
the continuation contract.

**Owned paths:** `proxima-model-interop/src/serving.rs`,
`proxima-model-interop/src/generate/decode.rs`,
`proxima-model-interop/src/generate/residency_caches.rs`, new serving modules
only if the Pipe audit establishes no fitting primitive, tests in
`proxima-model-interop/src/generate/`, the package feature declaration for
`unified-scheduling`, and `proxima-model-interop/benches/unified_scheduler.rs`.
Do not edit Omega config/backend-owned files while packet CONFIG/EXECUTOR is
active; root integrates API seams.

**Current path:** `run_decode_loop_from_ids` owns the full generation loop,
token history, caches, `cached_len`, sampler/RNG and scratch state until it
returns `PrefixState`. `BackendRuntime` is created per request. Existing
`PrefixState` owns IDs and layer caches but is not a one-step continuation.
`parallel_sequences != 1` and phase interleaving are typed rejections. Existing
`ubatch_size` only splits one request's prefill and drains chunks in order.

**Existing substrate audit:** write and compile the proposed existing-Pipe
call site before creating a new type. `Filter` can perform typed admission;
`BoundedQueue` supplies explicit overflow behavior; `FanIn` can select ready
pollable sources but does not advance today's synchronous backend evaluations.
Extend the current serving/decode owner for owned continuation, and use existing
Pipe elements for admission/readiness when the actual signatures compose.

**Worked rows:** use the five literal cases in `SPEC.md`, expanding exact
stream IDs, token IDs, sequence positions, cache allocation generations,
expected row ordering, independent logits/token oracle, sampler/RNG before and
after, queue state, and completion/cancellation events. The checker must run
both independent oracle output and its own output through one comparator; a
one-field mutation must be rejected. No symbolic-only expected states.

**Invariants:** no position/KV cross-talk; preserve model and logical
recurrent-visit identity; each stream's EOS/sampler/RNG and cancellation is
independent; sequence boundaries reach attention; dependencies and device
lifetimes survive packing; in-flight and queued work are bounded; every
continuously waiting prefill receives service within the configured bound;
resident/cold scheduling never drops runnable streams; Bloom positives always
perform exact identity/validity checks; queue overflow is typed backpressure;
reset/cancel drops exactly one owner's state.

**Primitive gate:** the bounded stream scheduler is new performance-sensitive
serving machinery. Keep its implementation behind default-off
`unified-scheduling` until its serial-incumbent microbench and real-model
paired arm are recorded. Config fields remain readable in all builds; a
compiled-out build must return the fixture's typed unsupported-feature error.
The gate does not disable the existing serial decode implementation.

**Microbench:** add the `unified_scheduler` bench with the current serial
serving selector as the incumbent and the new selector as the candidate at 2,
8 and 32 ready streams. Retain 5 raw runs per arm/size, selection latency,
allocations per selection and N-1 CoV. Add a frequency-weighted table for the
per-token selection path; use real-model checkpoint arms for end-to-end cost.
No perf claim is permitted from the microbench alone.

**Isolation:** after admission, the assigned worker uses worktree
`/private/tmp/proxima-unified-serving` on branch `codex/unified-serving` and
build output `/private/tmp/proxima-target-unified-serving`. Its packet excludes
CONFIG and EXECUTOR owned files; the root integrates shared public signatures
and reviews the actual diff before applying it to the dirty mainline tree.

**Dependency order:** policy resolved in Omega first. Implement continuation
state and literal transition tests before scheduling; then compatible row
packing; then prefill service/fairness; then cached residency preference.
After every implementation checkpoint, run Ouro end-to-end and all available
dense/MoE artifact workloads before proceeding.

**Validation:** AC3, AC4 and AC16-AC17 in `SPEC.md`. Preserve the current nextest config
and use a unique `CARGO_TARGET_DIR` per delegated validation run. Include exact
selected/excluded counts and full raw continuation records.

**Negative controls:** swapped stream map, reused position, mismatched logical
layer, stale allocation generation, Bloom-only authorization, cancelled queued
request returned by selector, cancellation freeing a submitted buffer, full
queue accepting one more item, and a decode-only schedule starving prefill.

**Artifacts:** `results/serving/` retains each scheduler choice and competing
candidate, exact inputs/state/output rows, cancellations, pressure/rejection,
cache snapshot, per-token timings and errors. Do not discard a losing schedule.
