# unified scheduler discipline log

Component: bounded serving stream scheduler

Status: unsealed; no implementation or benchmark result is claimed in this
specification turn.

## Gate record

| Gate | Required evidence | State |
|---|---|---|
| 1. Compile gate | `unified-scheduling` default-off; compiled-out typed rejection | Specified; unmeasured |
| 2. Build tiers | Enabled std/Metal build and alloc-only Omega build identify modules compiled | Specified; unmeasured |
| 3. Tests | Exact selected counts for schedule, lifecycle, concurrency and compiled-out cases | Specified; unmeasured |
| 4. Lint | Workspace strict lint command, no new suppressions | Specified; unmeasured |
| 5. Microbench | `unified_scheduler` bench, serial incumbent and scheduler arm | Specified; unmeasured |
| 6. Compare bench | 2, 8 and 32 ready streams; 5 raw runs per arm/size | Specified; unmeasured |
| 7. E2E bench | Dense and MoE, single and mixed long conversation arms at each checkpoint | Specified; unmeasured |
| 8. Optimization sweep | Existing bounded queue/Pipe patterns, fixed capacity, stack/borrow representation and setup allocation budget reviewed | Specified; unmeasured |
| 9. Rebench | Every tuning change and rollback receives a retained paired row | Specified; unmeasured |
| 10. SIMD/state machine | Selector state machine inspected; SIMD applicability and generated code recorded | Specified; unmeasured |
| 11. O(1) | Steady-state selection dequeues from fixed-capacity ready rings by head index; admission preclassifies compatibility so selection performs no ready-row scan or growth | Specified; unmeasured |
| 12. Config/API parity | Settings/file/env/fluent fixture resolves identical effective scheduler state | Specified; unmeasured |
| 13. Home-turf arm | Per-token selection is the hot path; paired serving model run covers realistic request cadence | Specified; unmeasured |
| 14. Internal primitive audit | Existing bounded queue/Pipe call sites tested before any new queue module | Specified; unmeasured |
| 15. Numeric sizing | Queue, sequence and chunk capacities come from config/build-generated sizing values | Specified; unmeasured |
| 16. Reproducibility | Exact command, source digest, raw inputs and outputs retained under `results/` | Specified; unmeasured |

## Optimization sweep

| Alternative | Disposition | Evidence required |
|---|---|---|
| Existing serial serving selector | Named incumbent | Retain its raw microbench and real-model arm. |
| Existing `BoundedQueue` | Reuse candidate | Compile the admission expression and prove ownership/overflow semantics. |
| Existing Pipe `FanIn` | Reuse candidate for readiness only | Compile a call site; it does not itself advance today's synchronous backend evaluation. |
| Growable priority queue | Rejected by fixed-cap/O(1) constraint | No heap growth may occur during selection. |
| New generic executor | Rejected; existing graph/backend owners execute work | Would duplicate execution and resource ownership. |

## Numeric axes

Ready stream count, active sequence cap, prefill chunk width, decode rows per
batch, in-flight submissions, bytes resident, and decode batches before a
prefill sweep are numeric. Backend family, queue ownership, schedule
mode, cache residency policy and strategy family are structural. The service
bound is two decode batches in this admitted fixture; configurable values must
be generated/sized inputs and retain the same explicit finite-bound semantics.

## Result rows

No rows exist yet. Every future result row must include command, commit/tree
source digest, feature set, host/device/driver, input fixture digest, arm, size,
raw per-run payload path, count, p50/p95/p99 selection latency, allocations per
selection, N-1 CoV, mechanism trace, status (`proven`, `plausible`,
`unmeasured` or `retracted`) and rollback relation. Missing fields keep the row
unsealed.

## R16 load and arena findings

| Change | Retained evidence | Observation | Next experiment |
|---|---|---|---|
| Keep BF16/F16 embedding tensors packed at bind time | `results/checkpoints/e2e-checkpoint-17/proxima/physical_footprint.jsonl`; `results/checkpoints/e2e-checkpoint-17/qwen-command.log`; binder test output in the session command log | Qwen3 load reached 58,336,832 current bytes and 63,104,576 lifetime peak bytes, below the 500,000,000-byte process cap. Generation then failed before dispatch at the Metal arena gate. The BF16 binder test observed an empty owned-F32 list and exact borrowed bytes. | Route a real token embedding through the packed binding on CPU and Metal, compare selected row values with the independent BF16 decoder, then rerun paired generation. |
| Reject a plan before creating arena buffers | `omega/src/metal/arena_encode_dispatch_finish.rs:119-192`; `results/checkpoints/e2e-checkpoint-18-qwen/command.log`; `results/checkpoints/e2e-checkpoint-18-granite/command.log`; `results/checkpoints/e2e-checkpoint-9/proxima/plan_sources.jsonl:1` | The live-output peak is computed from bound output lengths and retirement lists; the `ArenaOverCap` branch now precedes `allocate_buffer`. Qwen reported 1,246,414,464 requested vs 864,060,625 allowed bytes at five rows. Granite reported 201,634,008 vs 172,812,125 at one row. Both failed before dispatch; no generation or llama.cpp rows exist for these attempts. C9's actual Qwen plan row has node3192 `elementwise` extents `[1,2048,151936]`, followed by node3193 `keep::reduce fold`; the full product is part of the live-output peak. | Keep this projection in the existing graph/Plan owners while bounding the intermediate by vocabulary/output tiles, preserve reduction order, and compare complete output IDs before widening the cohort. Trace Granite's active expert/intermediate output shapes independently. |

These are failed implementation checkpoints for the end-to-end acceptance
criteria. They record load and admission behavior only; neither is a model
quality result, throughput sample or Proxima/llama.cpp comparison.

## Checkpoint 24 cached-attention binding hypothesis

### Worked example before the correction

Inputs for one composed decode row:

- namespace: `stream2`
- cached K/V live rows before the step: `119`
- padded cached extent supplied by the graph: `128`
- one new query/K/V row
- input node name after composition: `stream2::cached_len`

The current candidate matcher asks for an input whose entire name is
`cached_len`. It therefore finds no cached-length node in this composed
program. The no-bound branch can emit an eight-operand attention operation
whose shader fixes `cached_key_rows = 128`. The key loop then covers cached
indices `0..127` plus the new row, so the nine padded cached rows participate
in the softmax. The expected operation binds this row's scalar value `119` as
the ninth operand; the key loop covers cached indices `0..118` plus the new
row. For the other concrete CP24 failure, stream 41 has cached length `134`
and padded extent `160`, so the same distinction is 26 padded rows.

### Candidate-specific lookup rule

For each cached-attention output, identify the composed invocation that owns
that output, then select the scalar `cached_len` and optional sliding-length
input from that same invocation. Preserve the literal unnamespaced lookup for
an original single program. Do not select a global first suffix match: that
would allow stream 0's scalar to bound stream 1's attention. If the owner or
its scalar input is ambiguous, decline the bounded rewrite and retain the
existing behavior until the graph provides an explicit owner relation.

The intended test concatenates two three-node invocations. The first has
`stream0::cached_len` at `NodeId(0)` and its root at `NodeId(2)`; the second
has `stream1::cached_len` at `NodeId(3)` and its root at `NodeId(5)`.
Expected bindings are `(root0, node0)` and `(root1, node3)`. The negative
control makes a candidate reach both namespaces and requires the resolver to
reject it. This rule and test are recorded before changing the matcher; the
captured-shader/source join and real-model toggle are still required to
establish whether this mechanism explains CP24's output payloads.

### composed attention ownership regression

The executable two-range Mistral regression composes two attention graphs and
shares only the embedding weight. The bound stream0 candidate at `NodeId(86)`
uses its own cached-length `NodeId(15)` as operand nine; stream1 output
`NodeId(207)` uses `NodeId(136)`. The test also confirms those length leaves
are disconnected from each candidate's arithmetic ancestry. Renaming only
stream1's length leaves stream0's fusion intact, declines stream1's fusion,
and does not bind the unnamespaced decoy at `NodeId(1)`. The retained focused
run reports one selected/passed test and 720 skipped at
`results/executor/composed-two-range-attention-regression.log:10-15`.
This demonstrates the matcher behavior on the synthetic composed graph; it
does not establish the CP24 real-model causal link. The CP25 paired run is
the next one-variable execution check.

### CP24/CP25 paired toggle

CP24 retained two mixed-workload samples. In both, unified stream41 diverged
at output indices `40..63` and stream89 at `20`, `53`, and `56`; stream73
matched. The six exact comparison rows are CP24 `token-comparison.jsonl`
lines 171, 182, 201, 262, 273, and 292. Each input pair matched; all outputs
had 64 tokens and no EOS. The serial and single-workload rows also matched
their paired llama.cpp outputs.

CP25 changed the cached-attention candidate to resolve `cached_len` within
the owning invocation. Its mixed workload retained 12 Proxima/llama.cpp
request pairs, 12 equal input-ID arrays, 12 equal output-ID arrays, and 768
tokens per runtime. The raw paired output IDs and empty per-request difference
arrays are in `results/checkpoints/e2e-checkpoint-25-cached-len-qwen/comparison.jsonl`;
the paired command is in its `command.log`. CP25 Proxima outputs for stream41
and stream89 differ from CP24's failing Proxima outputs and equal CP25's
llama.cpp outputs. Stream73 remains equal. This is the expected one-variable
toggle direction, but it does not yet establish the matcher mechanism on the
real dispatch: the dispatch/source recorder loses plan history before it
serializes source rows.

The source-join loss is concrete in the retained CP25 Qwen payload: dispatch
0's kernel identity digest is `36188a4a...`, while the only end-of-cohort
`plan_sources.jsonl` row has digest `d127c379...`; dispatch 195 joins that
last row. `unified_execution_lab.rs:679`, `:700`, and `:721` record dispatch
snapshots during execution, but `:760` gathers sources after the cohort.
`residency_caches.rs:2948-2949` clears an old cached plan on a cache miss,
and `:3315-3316` clears plans when a stream program changes. Therefore the
early plan/source association is absent from retained evidence.

Next experiment: capture the exact plan's ordered kernel identity and resolved
pipeline-source identity at the execution owner before a later evaluation can
replace it; attach those digests to that physical dispatch ID and retain its
sources. Then rerun the Qwen and Granite mixed long-conversation pairs. Keep
the source mechanism `plausible` until the mismatching CP24 and matching CP25
dispatches join to the relevant ninth-operand kernel source. Process footprint
remains open: the current runner writes only startup and post-load
`TASK_VM_INFO` snapshots, not a lifecycle allocation/fence ledger.

The one non-warmup Qwen mixed-workload sample reports `cohort_response_ms`
128161.204625 in CP24 Proxima unified (`proxima/cohorts.jsonl:8`) and
139808.109959 in CP25 Proxima unified (`proxima/cohorts.jsonl:4`). The same
model digest, input IDs, three streams, parallel row cap of 3, workload and
sample index are recorded across these two runs; their generated token arrays
differ on streams41 and89 because the CP24 outputs diverged. CP25's paired
llama.cpp unified sample reports 993.016834 ms (`llama/cohorts.jsonl:4`). These
are single-sample response measurements; CoV is unmeasured and no timing cause
is established. Keep the raw cohorts and token payloads with both checkpoints.

## 2026-10-04: dispatch/source capture boundary repair

Proposed and authorized scope: adapt the existing BackendRuntime execution
calls and unified_execution_lab evidence owner. No model process is part of
this source repair. The current runner records plan hashes after each advance
but retains sources only from the final cache; resolve_cached_plan clears a
superseded plan on a miss. The C25 mixed serial prefill row references
36188a4a443c706b71a8c696edb3440fa7d0229022a37b646b3d05d9289e5844,
while its sole plan-source row retains final-plan d127c3798d42a916f75226a1f2e32078727fd1d4bf6a70edaae1a77cdeda1c8e.
Sources: examples/unified_execution_lab.rs:679,760 and
src/generate/residency_caches.rs:2948,3301 under proxima-model-interop;
results/checkpoints/e2e-checkpoint-25-cached-len-qwen/proxima/dispatch.jsonl:1
and plan_sources.jsonl:1 under this spec.

Reuse and lifecycle constraints reject both the final-cache snapshot and
retaining live Plan objects until cohort end: the former loses evicted
execution identity, and the latter extends buffer ownership. Capture instead
from each exact executed plan's existing resolved pipeline-pointer source
records while its borrow is alive. An optional instrument-gated observer on
BackendRuntime records bounded owned evidence in the example; its compilation
boundary owns serde_json and SHA-256 dev dependencies. Dispatch references
preserve execution order and distinguish equal kernel keys with different
resolved source tuples. Source files are verified once per unique digest
outside cohort latency. Failed verification never enters the recorded set.

Acceptance: five literal checks cover two distinct plan identities, an evicted
plan, equal kernels with different sources, one execution shared by multiple
batch request rows, and failed verification followed by retry. Build/check and
literal tests use a separate target directory; no GPU E2E or commit. Runtime
latency/allocation effects remain unmeasured until a subsequent authorized
model checkpoint. Pending implementation/check artifacts will be appended.

### CP25 Granite MoE paired payload

The retained Granite 3.1 1B A400M mixed_three_stream run used model digest
`cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80`, model
metadata `models.json:56-65`, and one non-warmup sample per arm. All 12 paired
input-ID arrays match llama.cpp; all 12 output-ID arrays differ, with 64
tokens per request and no EOS. The complete Proxima and llama.cpp IDs and
position differences remain in `token-comparison.jsonl:1-12`. The serial arm
also differs, so these rows do not isolate unified scheduling as the cause.
The per-token cause is unexplained by these token payloads.

Measured cohort response rows are Proxima serial 135599.811833 ms and unified
151026.135959 ms (`proxima/cohorts.jsonl:2,4`), and llama.cpp serial
1387.416209 ms and unified 1163.885417 ms (`llama/cohorts.jsonl:2,4`). There is
one non-warmup sample per arm, so CoV is unmeasured. The runner command is
retained at `command.log:2`. The process-footprint snapshot contains only
startup and model-load rows; it does not measure the full-run 500 MB cap.
These outcomes are adverse raw results alongside the Qwen CP25 matched payload
and must remain in the evidence set.

### Granite first-token source investigation

Opened the complete request arrays for all 12 CP25 Granite pairs. For each
stream, its four Proxima output arrays (serial/unified, warmup/measured) are
identical to one another, and the four llama.cpp arrays are identical to one
another; every paired input array matches. First tokens are stream41 Proxima
26 versus llama.cpp 203, stream73 34 versus 48841, and stream89 2953 versus
14084 (`token-comparison.jsonl:1-12`, `proxima/requests.jsonl:1-12`,
`llama/requests.jsonl:1-12`). This places the observed difference in serial
execution before any generated-token history diverges; it does not explain
the mechanism.

The exact GGUF header bytes contain `granitemoe.attention.scale=0.015625`,
`embedding_scale=12`, `residual_scale` bits `0x3e6147ae` (approximately
0.22), and `logit_scale=6`, at offsets `0x536`, `0x560`, `0x589`, and
`0x5af`. Proxima's `architecture_from_metadata` at
`proxima-model-interop/src/bind.rs:590-662` does not read these values;
`architecture.rs:640-651,687-699` routes the architecture through the generic
DenseArch; `dense.rs:117-148` builds the generic Mistral descriptor. The
descriptor applies inverse-square-root attention scaling at
`proxima-tensor/src/spec/descriptor.rs:452,481,751-769`, and the captured
attention source has `scale=0.125` (`proxima/pipeline_sources/pipeline_2a261a1bc03a2508.metal:855`).
llama.cpp reads these metadata fields at `granite-moe.cpp:5-8`, uses them in
`llama-graph.cpp:2429-2433` and `granite.cpp:189,226-229,241-244,301-304`.

The router source also exposes a separate hypothesis: its maximum-score mask
excludes every expert tied for the maximum
(`proxima/pipeline_sources/pipeline_755a1a534b0ccfe0.metal:878-905`). No
retained CP25 router score payload shows that a tie occurred, so this remains
unmeasured. Head-logit division by positive six alone preserves finite logit
ordering and cannot by itself explain a changed greedy first token.

Next experiment: on stream41's exact first-prefix input, capture the embedding
boundary, layer-0 attention and residual boundaries, router scores/selected
experts/weights, final hidden row and full logits, plus effective oracle input
positions and exact plan/dispatch identity. Toggle only embedding scaling from
1 to the checkpoint value 12, then compare those payload boundaries with
llama.cpp. The missing scales and tied-route hypothesis remain plausible until
their one-variable effects are observed; no cause is accepted by source
inspection alone.

The final design also rejects the initial function-pointer observer proposal:
it could not carry per-run state without an external registry. BackendRuntime
instead owns a preallocated event vector. Capacity per model is
`max(LoadedModel::program_node_count(), 1) * MAX_STREAMS`, where the runner's
existing scheduler capacity is eight streams. The runner records the actual
node count, slot count and reserved event bytes in `capture_capacity.jsonl`.
The run-wide source-record limit is the sum of admitted model capacities;
source files and plan records are deduplicated across the entire run. Both
limits fail explicitly rather than dropping a record. Runtime slot admission
precedes execution. The event vector is drained after every advance/batch.

Omega computes the ordered kernel and resolved-source SHA-256 identities once
when `resolve_steps` constructs its existing pipeline array, the earliest point
at which actual compiled source pointers exist. It stores immutable Arc-owned
metadata containing strings/shapes only; those handles cannot keep device
buffers or raw pipeline pointers alive. A third digest binds kernel, source,
and shape records. Per-execution capture copies fixed digests, ordinal, resource
scalars and the shared metadata handle, without rebuilding keys or hashing
sources on a cache hit. The runner retains dynamic resource scalars on each
execution reference as well as the initial source record. No default-build
capture storage, observer registry, lock, or second execution path is added.

### Source repair and CPU-only checks retained

Implemented source paths: `omega/src/metal/device_buffers_arena_plan.rs:959`
constructs immutable capture metadata from existing resolved pipeline pointers;
`:2348` returns its shared handle. `omega/src/metal/arena_encode_dispatch_finish.rs:677`
invalidates capture on in-place refit, `:769` refreshes it before the next
execution, and `:830` constructs it with new resolved steps.
`proxima-model-interop/src/generate/residency_caches.rs:1830` owns the bounded
capture vector; `:1897` admits a slot before each existing execution call and
records the exact borrowed plan afterward; `:3412` enables the model-derived
capacity and `:3421` drains ordered events. `proxima-model-interop/src/error.rs:15`
returns typed overflow and `:18` reports unavailable execution identity.
`proxima-model-interop/examples/unified_execution_lab.rs:164` owns the run-wide
source ledger; `:253` derives capacity, `:335` records its actual reservation,
`:1045` drains per advance/batch, and `:780` verifies new source files after
cohort completion. Each dispatch carries execution ordinal, plan_source_id,
kernel digest, source digest and current resource scalars. Batch request rows
share the captured execution sequence. No cache-at-cohort-end source snapshot
remains in the runner.

The following commands ran after the refit invalidation change, all with
`CARGO_TARGET_DIR=/private/tmp/proxima-target-execution-capture`. The logs were
opened after execution; these are CPU-only compilation/literal observations,
not real-model or performance evidence.

| command | retained observation | artifact |
|---|---|---|
| `cargo test -p omega --features 'cpu metal instrument' --lib execution_capture_refit -- --nocapture` | one selected test passed; 338 filtered; one invalidation preserves one old event and yields one refreshed source with unchanged kernel identity | `results/executor/execution-capture-refit-test.log:56`, `:59` |
| `cargo check -p proxima-model-interop --features 'std metal metal-output-placement instrument unified-scheduling' --example unified_execution_lab` | compiler finished the requested example; command exit 0 | `results/executor/execution-capture-example-check.log:3` |
| `cargo test -p proxima-model-interop --features 'std metal metal-output-placement instrument unified-scheduling' --example unified_execution_lab execution_capture_tests -- --nocapture` | five selected tests passed; two post-eviction joins, two same-kernel/different-source rows, four batch references joining two source rows, failed verification records zero rows before retry records one | `results/executor/execution-capture-literal-tests.log:7`, `:15`, `:18` |
| `cargo test -p proxima-model-interop --features 'std metal metal-output-placement instrument unified-scheduling' --lib execution_capture_tests -- --nocapture` | two selected tests passed; 592 filtered; overflow admits zero subsequent backend calls; both evicted plan events survive without vector reallocation | `results/executor/execution-capture-owner-tests.log:6`, `:8`, `:11` |
| `cargo check -p proxima-model-interop --features 'std metal metal-output-placement unified-scheduling' --lib` | compiler finished the library with capture instrumentation disabled; command exit 0 | `results/executor/execution-capture-disabled-check.log:3` |

`git diff --check` emitted no findings after the final source edits. The source
implementation check itself made no model/GPU calls, no Cargo or
`omega/src/backend.rs` edits, and no commit. The literal vector-pointer
assertion observes no event-vector resize; it is not a whole-process allocation
count. The real-model CP26 runs and retained joins follow.

### CP26 execution identity capture on dense and MoE payloads

The runner was rebuilt with `instrument` enabled and run against the exact
Qwen3-1.7B BF16 digest `ef2864916b951ebd2f072c759d11a442ba31acc3bcb7e5db5ca1d6fcf78b93cb`
and Granite 3.1 1B A400M digest
`cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80`. Each
run used mixed_three_stream, streams 41/73/89, Metal, bit_exact, one warmup
and one measured sample per serial/unified arm, and 64 generated IDs/request.
The exact invocations and Qwen's initial missing-environment attempt are
retained in each checkpoint's `command.log`.

Qwen's 12 request pairs have equal input and output arrays: 768 generated IDs
per runtime and zero differing token positions
(`e2e-checkpoint-26-capture-qwen/token-comparison.jsonl:1-12`). Granite's 12
pairs have equal input arrays but no equal 64-token output array; 764 of the
768 compared positions differ. Its first output token remains stream41 26 vs
203, stream73 34 vs 48841, and stream89 2953 vs 14084. The complete arrays
and all position deltas are retained at
`e2e-checkpoint-26-capture-granite/token-comparison.jsonl:1-12`. All these
Granite differences appear in serial and unified request rows, so this data
does not isolate batching. The Granite first-prefix mechanism remains
unexplained by token IDs alone.

The source join is now present in the same execution payload: Qwen has 800
dispatch rows with 800 execution references, 17 distinct retained plan-source
rows and zero unjoined references; Granite has 804 rows/references, 16 source
rows and zero unjoined references. The independent recount used each
checkpoint's `proxima/dispatch.jsonl` and `proxima/plan_sources.jsonl`; these
files retain per-dispatch ordinals, kernel/source/plan digests, resource
scalars and verified source files. In particular, Qwen dispatch row 1's
kernel digest `36188a4a...` joins its own plan-source identity, rather than a
later cached plan. The raw source rows let a reviewer follow a dispatch to
its ordered kernel and resolved pipeline source.

For Qwen's one measured sample, Proxima serial/unified cohort response was
120100.004292/127159.779250 ms and llama.cpp serial/unified was
1003.814708/995.336500 ms (`proxima/cohorts.jsonl:2,4`,
`llama/cohorts.jsonl:2,4`). CP25 Qwen Proxima measured rows were
144063.390833/139808.109959 ms, under the same model/workload/sample shape but
without per-dispatch source capture; the one sample per arm and changed
instrumentation make timing attribution unmeasured. CoV is unmeasured for
both checkpoints. Granite CP26 measured Proxima serial/unified response was
104066.449667/113838.108083 ms and llama.cpp was 798.290625/910.667208 ms
(`proxima/cohorts.jsonl:2,4`, `llama/cohorts.jsonl:2,4`). One measured
sample/arm leaves CoV unmeasured.

The two `physical_footprint.jsonl` rows in each checkpoint are startup and
model_load_complete only. Model-load lifetime peaks were 66594368 bytes for
Qwen and 49565760 bytes for Granite against the 500000000-byte configured
cap; no during-dispatch or teardown samples exist, so the lifecycle cap
remains unmeasured. These native snapshots do not establish that the complete
run stayed below the cap.

The one-variable output result is mixed: Qwen's formerly divergent CP24
unified requests remain equal to llama.cpp after the scoped cached-length
change; Granite's serial output remains divergent. CP26 source evidence is
now available for its own successful and failing dispatches, but it does not
join the older CP24 failing dispatches. For Granite, continue with first-prefix
embedding/attention/residual/router/logit payload capture and a single
embedding-scale 1-to-12 toggle before attributing the missing checkpoint
scales as cause. No claim about production timing follows from these one-sample
instrumented cohorts.

### CP27/CP28 footprint observer failure and retained E2E payloads

CP27 ran the exact Qwen3-1.7B BF16 digest in `models.json`, with single and
mixed-three-stream workloads, serial/unified arms, five measured samples and
one warmup, and llama.cpp paired execution. All four workload/arm cells emitted
24 cohort rows per runtime in total, 48 request rows and 3,072 token rows per
runtime. The request-ID/position comparator joined 48 requests, matched all
3,072 output token positions, and found zero length mismatches; the complete
arrays and position deltas are retained in
`results/e2e-checkpoint-27-qwen/token-comparison.jsonl:1`.

The runner exited 1 after its final sampler check reported
`periodic_sample_deadline_missed`. Across 173,843 process observations it
recorded 177 deadline errors and 8,369 cases where current `phys_footprint`
exceeded `ledger_phys_footprint_peak`; no sampled current/peak crossed the
500,000,000-byte threshold. The largest current sample is 364,753,664 bytes
(`results/e2e-checkpoint-27-qwen/proxima/physical_footprint.jsonl:128084`),
while the largest reported native peak is 341,914,368 bytes
(`.../physical_footprint.jsonl:173843`). At sequence 1565, current was
93,612,032 bytes and reported peak 91,564,032 bytes. Their inconsistency means
the sampler output does not establish a lifecycle cap result.

The same checkpoint separately recorded a 5,582,176,256-byte
`getrusage(RUSAGE_SELF) ru_maxrss` sample and 4,079,220,784 Metal
`current_allocated_size_bytes` sample (`results/e2e-checkpoint-27-qwen/proxima/resources.jsonl:48`).
These are distinct counters; they are retained without summing or substituting
them for `TASK_VM_INFO phys_footprint`. `--allocation-ledger` explicitly
recorded incomplete owner/acquire/retire/alias/fence coverage. Thus CP27 is
payload-complete for generation and token pairing, but not a valid capped
memory checkpoint.

CP28 used the runner phase change that continues after sampling errors and
returns them at finalization, stopping a live cohort only on an observed
physical-footprint threshold crossing. It was interrupted before any llama.cpp
cohort after 15 Proxima cohorts, 21 requests and 1,344 token rows. The retained
maximum RSS sample was 4,920,770,560 bytes and maximum Metal allocation sample
was 4,079,220,784 bytes (`results/e2e-checkpoint-28-qwen/proxima/resources.jsonl:21`).
The highest native current sample was 280,177,728 bytes at sequence 43,503 and
reported peak 277,720,128 bytes, an inconsistent pair
(`results/e2e-checkpoint-28-qwen/proxima/physical_footprint.jsonl:43504`).
These observations do not settle whether the requested native physical cap was
crossed; the run is kept partial and no cap verdict is made.

The native query's task-info struct is 336 bytes (84 natural words), matching
the compiled local SDK `TASK_VM_INFO_REV3_COUNT`; the separate review claim of
85 words did not reproduce against the active SDK. Header and Rust field offsets
match, while CP27/CP28 payloads still show current above the returned peak. The
reason for that value mismatch is unresolved. Query-duration fields omit JSON
construction/write time, so sampler overhead is also unmeasured. Evidence and
exit/interruption status are in each checkpoint's `RUN-STATUS.json`.

Next experiment: complete AC18 owner admission and lifecycle joins in existing
mapping, binding, Metal Plan, KV, expert, cache and fence owners; correct any
actual native counter mismatch only when a source-level/SDK reproduction
isolates it; then rerun Qwen dense and Granite MoE with llama.cpp under the same
cap and retain raw sources, routes, resource rows, all samples and CoV. Keep
these failed observer and partial-run payloads immutable.
