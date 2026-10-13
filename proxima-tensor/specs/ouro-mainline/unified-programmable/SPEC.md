# unified programmable execution and serving

status: admitted
owner: brian
created: 2026-10-03

## problem

Proxima/Omega callers cannot express and observe one typed policy that constrains serving, graph scheduling, backend parallelism, batching and residency, so the shared execution path must consume a resolved `omega::RuntimeConfig` and produce per-request traces plus matched real-model wall-time records.

## refutation condition

The adaptation is refuted if any accepted control does not affect the execution record or return a typed unsupported-capability error, if interleaved multi-stream execution changes a stream's token/KV/sampler state or violates graph hazards/resource ceilings, or if the measured shared path is a separate Ouro implementation instead of the existing model descriptor, Omega plan and backend drivers.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Extend the existing `omega::RuntimeConfig` and conflaguration Settings/fluent surfaces with typed workload objective, hard resource, numeric permission, parallelism, schedule, residency and strategy inputs; config and builder serialize to the same state. | yes |
| R2 | Resolve generated defaults, file, environment and explicit fluent/legacy-serving overrides once with precedence `build floor < file < environment < explicit fluent/serving override`. | yes |
| R3 | Consume every accepted field at its named owner, reject mandatory unsupported capabilities with typed errors, keep performance choices separate from semantic/numeric permissions, and key plans/caches by effective policy/layout/backend. | yes |
| R4 | Bound CPU kernel workers, independent DAG operations, in-flight submissions, transfer/compute overlap and device placement while preserving dependency/hazard edges, buffer lifetimes and cancellation; absent drivers report unavailable. | yes |
| R5 | Represent generation as resumable owned stream state and schedule bounded prefill chunks/compatible decode rows on the shared `BackendRuntime`; preserve order, EOS, cancellation, sampler/RNG, positions/KV; admit with backpressure and begin an arrival-ordered prefill sweep after at most two decode batches. | yes |
| R6 | Use exact stream/cache identity and valid resident placement for scheduling/reuse. Bloom/prefix indexes screen only; cold/evicted streams remain runnable; reject stale model/config/context/position/logical-visit/allocation-generation state. | yes |
| R7 | Route decode-shaped kernels and numeric-permitted fusion through existing bind/classify/emit/cache identities; unsupported forced shapes reject. | yes |
| R8 | Apply plan/encoding reuse, intermediate/device residency, exact token-selection boundaries and transfer/compute overlap through existing plan/cache/placement owners; unsupported required strategies reject. | yes |
| R9 | Preserve target-verified speculative execution: each accepted draft advances only after target verification; rejected drafts fall back to the target token with exact per-stream state. | yes |
| R10 | Cache weight conversions by source bytes, target format, rounding mode and layout; perform lower-precision conversion only under explicit numeric permission. | yes |
| R11 | Group MoE rows by the already-selected expert for shared execution without changing token-to-expert routing or output order. | yes |
| R12 | Emit structured effective-plan and per-request evidence joining requested/effective policy, backend/kernel/schedule, rejected candidates, resource use, placement/cache decisions, transfers, dispatches and complete token outcomes. | yes |
| R13 | Exercise exact local dense and MoE models plus longer/mixed-arrival conversations against llama.cpp on matched artifact, input IDs, precision, context, output budget, arrival schedule and instrumentation; retain raw payloads, regressions and CoV%. | yes |
| R14 | Preserve the current no_std/alloc floor and generated build-time defaults for restricted tiers, with runtime defaults seeded from generated constants. | yes |
| R15 | Keep the newly introduced bounded stream scheduler behind a default-off `unified-scheduling` feature until its microbench and paired real-model arm are retained; the compiled-out configuration rejects requests for that scheduler with a typed error. | yes |
| R16 | Enforce the workspace 500,000,000-byte Proxima process physical-footprint ceiling before and throughout load, compile, execution, transitions and teardown; account allocations by owner/backing identity/lifetime/fence, retain unmatched ranges explicitly and refuse before an allocation would exceed the reserved ceiling. | yes |

## architecture

```text
generated omega sizing/profile defaults
               |
file -> env -> fluent/serving overrides -> validated RuntimeConfig
               |
ServingConfig compatibility adapter + model descriptor/cache identity
               |
bounded per-stream continuation + fair prefill/decode selection
               |
existing Proxima graph, numeric policy, bind/liveness/hazard DAG
               |
existing Omega configured planner -> concrete CPU/Metal/CUDA/WGPU plans
               |
existing placement/transfer/dispatch/execute/readback owners
               |
effective-plan and request/token payloads -> independent recount + llama.cpp
```

The common programmable surface extends `omega::RuntimeConfig`; the existing
`ServingConfig::omega_runtime` reference remains the compatibility route. Runtime
resolution occurs once at the cold request/admission boundary. Explicit serving
overrides win over `RuntimeConfig`, which wins over file/env values, which win
over generated defaults. Numeric permissions and hard resource ceilings remain
independent of objective/strategy preferences.

Serving owns request admission, resumable stream state, cancellation, phase
fairness and compatible row formation. Omega owns legal DAG/worker scheduling,
resource placement and configured backend lowering. Concrete drivers own device
allocation, synchronization, command submission and device-specific capabilities.
Existing graph dependencies and output lifetimes remain authoritative. One shared
runtime/weight owner is reused; no Ouro pipeline or second executor is added.

### worked scheduling cases

1. **Packed decode rows.** Fixture `fixtures.json#/packed_decode` binds row
   inputs `[17,29]`, stream IDs `[41,73]`, positions `[2,3]`, KV slots and
   generations `[(7,4),(9,8)]`, KV prefixes `[[11,12],[21,22,23]]`, and token
   vocabulary `[317,902,1001]`. Logits are `[[0,3,1],[4,1,2]]`; output token
   IDs must be `[902,317]`. Greedy samplers' eight-byte RNG states remain
   `0001020304050607` and `08090a0b0c0d0e0f`. After the step positions are
   `[3,4]`, KV lengths `[3,4]`, and slots/generations and sampler bytes remain
   attached to their original IDs. The oracle itself passes the same checker
   with zero differences; swapping stream ID, position, or generation is
   rejected exactly once each.
2. **Positions/KV isolation.** Continuation consumes `902` at position `3`
   and `317` at position `4`; attention spans are `[0,4)` and `[0,5)`. Reset
   stream 41 drops slot `(7,4)` and leaves stream 73's exact prefix
   `[21,22,23]`, slot `(9,8)`, token `317`, and sampler bytes
   `08090a0b0c0d0e0f` unchanged.
3. **Dependency hazards.** Fixture DAG node IDs `[0,1,2,3]`, edges
   `[(0,1),(0,2),(1,3),(2,3)]`, independent output vector `[11,18,33,36]`,
   and legal levels `[[0],[1,2],[3]]`. Oracle output passes the same checker
   with zero mismatches; bad levels `[[0],[1,3],[2]]` produce exactly one
   hazard violation.
4. **Resource pressure.** A 64-byte ceiling admits `[24,24,16]` with allocation
   count `3` and byte total `64`; a fourth `1` returns
   `ResourceLimitExceeded` with the counters unchanged. A 63-byte ceiling
   rejects the three allocations with allocation count `0` and dispatch count
   `0`.
5. **Cancellation.** Cancel stream 73 while stream 41's row is in flight.
   Stream 41 completes with token `902` and slot `(7,4)` still live; stream 73
   emits zero IDs and slot `(9,8)` is released only after command completion.
   Next selection is `[41]`, stale handle count is `0`, and submitted-buffer
   live count falls from `1` to `0` only on the completion event.
6. **Mixed arrivals and fairness.** Fixture `fixtures.json#/arrival_schedule`
   has arrivals at ticks `0` and `2`, decode rows A/B, prefill streams P0/P1,
   two-token prompts and chunk width `1`. Required trace is
   `[D(A,B),D(A,B),P0-0,P1-0,D(A,B),P0-1,P1-1]`. The oldest ready prefill
   waits at most two decode batches. Its independent oracle passes the same
   checker; an infinite-decode trace with P0 ready at tick `0` is rejected at
   decode batch `3`.

These fixtures use independent literal records/checkers, including the oracle
record itself as a positive input and a one-field mutation as a negative control.

### scheduler state transition

The selector stores `consecutive_decode_batches`, the arrival ordinal of each
ready prefill stream, and whether a bounded prefill sweep is active. A sweep
visits the ready-prefill snapshot in arrival order, one chunk per stream, then
returns to decode selection. Fixed-capacity ready rings use generated sizing
constants; admission places work into compatibility queues, so selection pops
by head index without scanning ready rows. Selection is deterministic:

1. If no prefill is ready, select the oldest compatible decode rows.
2. If a prefill sweep is active, select the next stream from its captured
   arrival-ordered ready set and one bounded chunk; when exhausted, end sweep
   and reset the counter.
3. If prefill is ready and `consecutive_decode_batches == 2`, begin a sweep
   from the ready-prefill snapshot and select its oldest stream's bounded
   chunk.
4. Otherwise, select compatible decode rows if any; increment the counter
   only when a ready prefill is waiting.
5. If no decode row is ready, begin the prefill sweep immediately.

For the fixture trace, A/B are decode-ready at ticks 0, 1 and 4; P0 arrives at
tick 0 and P1 at tick 2. Selections 1 and 2 are `D(A,B)`, taking the counter
from 0 to 2. At tick 2 the ready-prefill snapshot is `[P0,P1]`, so the sweep
emits `P0-0`, then `P1-0`. At tick 4 it emits `D(A,B)`; no decode is ready at
tick 5, so the selector starts a prefill sweep immediately and emits `P0-1`,
then `P1-1`. Thus the exact trace is
`[D(A,B),D(A,B),P0-0,P1-0,D(A,B),P0-1,P1-1]`. By induction, while prefill
remains ready, no interval contains more than two decode selections before a
sweep begins; each sweep services every stream in its captured ready set once.
New arrivals join the next sweep in arrival order.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| shared configuration home | Extend Omega `RuntimeConfig`, preserving `ServingConfig::omega_runtime` as its consumer | A second serving/runtime config would create competing precedence and policy identities. |
| scheduler work | Extend serving continuation and queue owners; use existing bounded/Pipe primitives where their call sites fit | `FanIn` chooses ready owned items but does not advance synchronous GPU evaluations; current full-call decode loop cannot be passed to it unchanged. |
| batch ownership | Row descriptors bind stream identity, position and cache-generation to each existing graph invocation | Concatenating token IDs alone loses attention history boundaries. |
| device placement | Expose portable required/allowed capability data; drivers report their actual execution facts | Selecting multiple names from backend candidates is not distributed execution. |
| unknown performance gain | Keep strategy requests measurable and individually switchable, and retain negative results | A setting name, utilization counter or aggregate delta does not prove lower wall time or causality. |
| restricted tier | Generated Omega floor constants seed runtime defaults | Environment/file access cannot exist at the no_std/no_alloc floor. |

### principles applied before implementation

- **Reuse first:** the proposed pipe expression is bounded admission into the
  serving owner, `Filter::new(admit).and_then(enqueue)`; `BoundedQueue` holds
  owned work and `FanIn` can poll ready sources. The caller capability newly
  enabled is a single stream continuation that yields between prefill/decode
  evaluations and can carry exact KV/position/sampler ownership into a compatible
  row batch. If an existing serving or Pipe primitive can express that by a
  compiling call site, extend it rather than creating a peer.
- **Abandoned design:** a new generic job/executor API was rejected. The existing
  graph DAG, `BackendRuntime`, serving caches and backend placement APIs already
  own the respective work; only a continuation/batch boundary is missing.
- **Allocation budget:** policy resolution and queue setup may allocate at cold
  setup; per-token selection and fixed-capacity scheduling target zero new heap
  allocation after admission; graph/backend allocations are separately charged
  to their concrete owners. Evidence must count allocations before any zero
  allocation claim.
- **Tunable axes:** queue/in-flight capacities, worker count, decode row cap,
  prefill chunk cap, max decode turns before prefill, resident-byte ceiling,
  submission chunk count, warmups/samples are numeric axes. Backend, executor,
  permitted device families, schedule phase, cache residency and strategy family
  are structural axes. Build constants come from Omega's sizing/profile source;
  runtime values overlay them.
- **Design pressure that changed the API:** the fixed service bound replaces a
  freely tunable “fairness” label. It is an explicit max-decode count with a
  literal state transition and bounded prefill chunk; an unbounded priority
  queue or best-effort scheduler would not satisfy the contract.
- **Skill routing:** `algorithm-development` covers the specified state
  transitions and worked examples; `proxima-pipe` audits admission and readiness
  composition; `instrument` and `bench-metrics` govern performance attribution;
  `disciplined-component` applies to each new performance primitive/strategy.

## acceptance criteria

Commands are run from the Proxima repository root. Every emitted JSONL payload is
retained beneath this directory's `results/` and recounted by a Rust example that
checks exact expected row counts and a one-record-corruption negative control.
Before the shared real-model runner is executable, each source slice carries its
owner tests and existing mainline dense and MoE compatibility checkpoints. Once the runner
is executable, every integrated source checkpoint runs its dense and MoE real
model controls before the next integration checkpoint; the final checkpoint
runs the full AC12/AC13 matrix.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1,R2,R3,R4 | `cargo nextest run -p proxima-omega --lib -E 'test(unified_runtime_config_)' --no-capture` | 16 named tests; `fixtures.json#/resolved_config` expects workers=4, independent-ops=3, transfer/compute overlap=true, sequences=3, in-flight=2, device-bytes=64, reassociation=false and engine=cpu. Independent fixture passes same comparator with 0 differences; one changed field rejects exactly once; 0 ignored. |
| AC2 | R3,R4 | `cargo nextest run -p proxima-omega --lib -E 'test(unified_execution_plan_)' --no-capture`<br>`cargo run -p proxima-model-interop --example unified_execution_recount --features std -- --policy-results proxima-tensor/specs/ouro-mainline/unified-programmable/results/executor/policy-consumption.jsonl --policy-fixture proxima-tensor/specs/ouro-mainline/unified-programmable/fixtures.json` | First command: 18 named tests, 0 ignored; worker cap, bounded submissions, exact 64-byte/one-byte-over controls, candidate report and unavailable-driver rejection. DAG oracle passes the same checker with 0 mismatches; bad level order yields exactly 1 hazard violation. Second command: exactly 26 distinct `execution_result` rows covering the 20 RuntimeConfig fields, 3 resource constraints and 3 selection fields in `fixtures.json#/policy_consumption`; each has non-null `field_path`, `requested_value`, `effective_value_or_error`, `owner`, `status`, `source`. Status is `consumed` with an actual effective value, or `typed_unsupported`/`rejected_by_validation` with a named typed error variant and field. Source resolves to a retained actual consumer event/result, including dispatch/allocation/scheduler result or pre-dispatch rejection; a declaration or owner inventory does not qualify. Required paths match exactly: 0 duplicate, missing, extra, owner_mapping or unaccounted rows. Retain the raw 26-row artifact at the command's input destination and the recount/control output under `results/recount/policy-consumption/`. The independent 26-row literal schema oracle passes the same row checker with 0 mismatches; changing its first row's `status` to `owner_mapping` yields exactly 1 rejection. Literal controls are marked and never counted as execution evidence. |
| AC3 | R5,R6 | `cargo nextest run -p proxima-model-interop --features 'std metal metal-output-placement' --lib -E 'test(unified_stream_state_)' --no-capture` | 20 named tests, 0 ignored; `packed_decode` checks IDs `[902,317]`, positions `[3,4]`, KV lengths `[3,4]`, slots `(7,4)/(9,8)`. `sampling_resume` checks seed 192, temperature 0.8, IDs `[234,246,187,224]`, and post-draw RNG bytes `[65796caacc8d352d,0af2d854991b6b5a,af6a45ff65a9a087,54e3b1a93237d6b4]`; pause/resume preserves state and cancellation draws 0 tokens. `residency_strategy` rejects each model/policy/context-allocation-pointer/context-length/position/logical-visit/allocation-generation mutation exactly once (7 controls). |
| AC4 | R5,R6 | `cargo nextest run -p proxima-model-interop --features 'std metal metal-output-placement unified-scheduling' --lib -E 'test(unified_fair_schedule_)' --no-capture` | 16 named tests, 0 ignored; arrival oracle trace exactly `[D(A,B),D(A,B),P0-0,P1-0,D(A,B),P0-1,P1-1]`; oldest prefill begins its sweep within 2 decode batches. Oracle passes same checker once; starvation control rejects at batch 3. |
| AC5 | R3,R4 | `cargo nextest run -p proxima-omega --features 'cpu metal wgpu-backend cuda-driver' --lib -E 'test(unified_backend_capability_)' --no-capture` | 12 named tests, 0 ignored; on recorded Apple M1 Max CPU=1, Metal=1, WGPU=1 actual device execution, CUDA=0 and NPU=0; all five requests record actual execution or a typed rejection with 0 CPU fallback dispatches. |
| AC6 | R7 | `cargo nextest run -p proxima-omega --features 'cpu metal' --lib -E 'test(unified_decode_strategy_)' --no-capture` | 12 named kernel-shape/fusion tests; independent literal output oracle passes same comparator, one-coordinate corruption rejects once. |
| AC7 | R8 | `cargo nextest run -p proxima-omega --features 'cpu metal' --lib -E 'test(unified_residency_strategy_)' --no-capture` | 12 named tests; fixture requires 1 exact-resident reuse, 1 stale-generation rejection, 1 Bloom-positive exact-mismatch rejection and 1 transfer/compute pair sharing command IDs. Independent literal oracle passes same comparator; invalid placement rejects exactly once. |
| AC8 | R9 | `cargo nextest run -p proxima-model-interop --features 'std metal metal-output-placement' --lib -E 'test(unified_speculation_verify_)' --no-capture` | 4 named tests: independent target logits oracle itself passes the verifier comparator; accepted/rejected draft token arrays and per-stream positions match literal records; one corrupted target token rejects exactly once. |
| AC9 | R10 | `cargo nextest run -p proxima-omega --features 'cpu metal' --lib -E 'test(unified_weight_conversion_)' --no-capture` | `fixtures.json#/weight_conversion` oracle passes the same byte comparator with 0 mismatches. Four named tests: identical source/format/rounding/layout hits once; each changed key field misses; denied precision conversion returns one typed rejection and emits zero converted bytes. |
| AC10 | R11 | `cargo nextest run -p proxima-model-interop --features 'std metal metal-output-placement' --lib -E 'test(unified_moe_grouping_)' --no-capture` | 4 named tests: literal route-control fixture groups rows only after route selection, preserves per-token expert IDs/output order and rejects one changed route exactly once; real Granite route IDs are emitted in AC12 payloads and recounted by model/token. |
| AC11 | R12 | `cargo run -p proxima-model-interop --example unified_execution_recount --features std -- proxima-tensor/specs/ouro-mainline/unified-programmable/results/checkpoint-1` | `fixtures.json#/recount` requires 4 model summaries, 2 workload summaries, 16 model×workload×arm cells, 96 cohort runs and 192 stream-request rows (16 cohort warmups + 80 measured cohort runs). Every token timing count equals output-ID length; every dispatch/resource/residency row joins one request; dropped/unjoined rows=0. Independent literal recount fixture passes; one changed model digest rejects once. |
| AC12 | R13,R15 | `cargo run -p proxima-model-interop --release --example unified_execution_lab --features 'std metal metal-tiled-gemm unified-scheduling' -- --manifest proxima-tensor/specs/ouro-mainline/unified-programmable/models.json --samples 5 --output proxima-tensor/specs/ouro-mainline/unified-programmable/results/checkpoint-1` | Exactly four identities: ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, Qwen3-1.7B local GGUF/config, `ibm-granite/granite-3.1-1b-a400m-instruct` at the blob in `decode-performance-lab/manifest.json`, and Gemma 4 E2B QAT GGUF at that manifest path. Ouro/Qwen/Gemma are dense; Granite is MoE with 32 experts/top-8. Exactly 2 arms, 2 workloads/model, 6 cohort runs/cell/arm: 96 cohort runs and 192 stream-request rows. All prompt/output IDs, text/EOS, token timing, TTFT/response, CPU/RSS, effective plan, cache/residency/dispatch events/errors retained; dropped telemetry=0. |
| AC13 | R13,R15 | `cargo run -p proxima-model-interop --release --example unified_execution_lab --features 'std metal metal-tiled-gemm unified-scheduling' -- --manifest proxima-tensor/specs/ouro-mainline/unified-programmable/models.json --compare llama.cpp --samples 5 --output proxima-tensor/specs/ouro-mainline/unified-programmable/results/llama-paired-1` | `fixtures.json#/paired_comparator` passes with 0 mismatches; five-sample sample standard deviation divided by mean, percent rounded to 2 decimals, yields Proxima 7.07% and llama.cpp 5.89%. Four model IDs × two workloads × two arms × six cohort runs/runtime yields 96 cohort rows and 192 stream-request rows per runtime. Each pair binds content digest, input IDs, precision, context, output budget, sequences and arrival trace; outputs and five-sample N-1 CoV% are recounted. Changed timing and changed input each reject once. |
| AC14 | R14 | `cargo check -p proxima-omega --no-default-features --features alloc --message-format=json` | 1 omega library artifact, 0 error diagnostics; alloc graph/binding/emission modules built and std runtime config excluded. |
| AC15 | R1,R2,R14,R15 | `cargo clippy -p proxima-omega -p proxima-tensor -p proxima-model-interop --features 'std metal metal-output-placement proxima-model-interop/metal-tiled-gemm proxima-model-interop/unified-scheduling' --all-targets -- -D warnings -A clippy::expect_used -A clippy::unwrap_used` | 0 compiler/clippy diagnostics across affected targets. |
| AC16 | R5,R15 | `cargo nextest run -p proxima-model-interop --features 'std' --lib -E 'test(unified_scheduler_feature_off_)' --no-capture` | `fixtures.json#/scheduler_feature_off`: 4 named requests return variant `UnifiedSchedulingUnavailable` with field `feature="unified-scheduling"`; 0 scheduler dispatches and 0 scheduler implementation tests selected. Enabled-feature scheduler tests are counted separately in AC4. |
| AC17 | R5,R15 | `cargo bench -p proxima-model-interop --features 'std metal metal-output-placement unified-scheduling' --bench unified_scheduler` | Named incumbent arm (serial serving selection) and scheduler arm each retain 5 raw runs at 2, 8 and 32 ready streams; N-1 CoV and frequency-weighted cost table present; bench harness records allocation/op and selection latency; no claimed win without paired real-model arm. |
| AC18 | R16 | `PROXIMA_PIPELINE_CAPTURE=proxima-tensor/specs/ouro-mainline/unified-programmable/results/memory-checkpoint-1/proxima/pipeline_sources cargo run -p proxima-model-interop --release --example unified_execution_lab --features 'std metal metal-tiled-gemm instrument unified-scheduling' -- --manifest proxima-tensor/specs/ouro-mainline/unified-programmable/models.json --compare llama.cpp --enforce-physical-footprint-cap-bytes 500000000 --allocation-ledger --samples 5 --output proxima-tensor/specs/ouro-mainline/unified-programmable/results/memory-checkpoint-1` | All four manifest model identities, both workload classes, serial and unified arms are retained; each process emits startup/load/compile/dispatch/completion/teardown footprint observations and allocation acquire/retire rows with owner, address/range, backing identity/alias, bytes, generation and fence. Query the OS physical-footprint lifetime high-water counter at each acquire/retire, command submit/complete, phase transition and teardown; retain the native high-water plus transition samples so brief peaks are not inferred from sparse RSS polling. A missing/reset/unavailable high-water field or missed lifecycle event invalidates the capped row. Every allocation is admitted before creation; `unjoined_allocation_rows=0`, `unowned_physical_ranges=0`, `cap_exceeded_events=0`, and native process physical-footprint high-water <=500,000,000 bytes for each Proxima run. Host RSS, mapped residency and Metal allocation remain separate measures; matched llama.cpp rows remain separately attributed under AC13. Exact-fit reservation is admitted and one-byte-over is rejected before allocator/dispatch counters change. Literal ledger oracle passes with 0 differences and its one-byte-over mutation rejects exactly once. |

## out of scope

- Publishing, committing, pushing or deploying changes.
- A new tensor IR, model-specific executor, duplicate Ouro transformer or new
  cache owner.
- Claiming unavailable hardware execution. Add a driver only if it can execute
  the existing plan boundary and report real allocator/dispatch/synchronization
  evidence.
- Global-optimum scheduling, full hardware occupancy, unsupported asynchronous
  multi-device partitioning, or silent relaxation of model numeric policy.
- Downloading models without a confirmed public source, sufficient local storage
  and exact artifact identity; record absent artifacts and continue all locally
  executable work.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| decode refactor loses state at a yield | high | token, RNG, position or KV corruption | literal per-stream traces and reset/cancel controls at every transition |
| row packing crosses attention/cache boundaries | high | semantic corruption | explicit row-to-stream/position/cache-generation map checked before bind and after output |
| serialization around one mutable backend runtime limits overlap | medium | false parallelism claim or unsafe concurrent plans | distinguish request batching from DAG/driver concurrency; instrument actual submission overlap |
| some named model artifacts or drivers are absent | medium | incomplete hardware/model coverage | inspect exact local paths/digests and report typed unavailable rows; never substitute simulation |
| strategy toggles share effects | medium | un-attributable result | one strategy toggle per paired experiment, retain complete dispatch/cache/copy records and regressions |

## context

- `ai_docs/AGENT.md`
- `ai_docs/index.jsonl`
- `ai_docs/task-routes.jsonl`
- `ai_docs/invariants.jsonl`
- `ai_docs/ouro-mainline.jsonl`
- `ai_docs/omega-programmable.jsonl`
- `omega/src/config.rs`
- `omega/src/backend.rs`
- `omega/build.rs`
- `omega/omega-runtime.toml`
- `proxima-model-interop/src/serving.rs`
- `proxima-model-interop/src/generate/decode.rs`
- `proxima-model-interop/src/generate/residency_caches.rs`
- `proxima-tensor/specs/ouro-mainline/SCHEDULING-INVENTORY.md`
- `proxima-tensor/specs/ouro-mainline/EVIDENCE.md`
- `proxima-tensor/specs/ouro-mainline/results/BROADCAST-FOLD-COMMANDS.md`
- `proxima-tensor/specs/ouro-mainline/results/broadcast-fold-final-audit.md`
- `proxima-tensor/specs/omega-programmable/SPEC.md`
- `proxima-tensor/specs/omega-programmable/TASKS.md`
- `proxima-tensor/specs/decode-performance-lab/manifest.json`
- `proxima-tensor/specs/decode-performance-lab/fixtures/granite.log`
- `proxima-tensor/specs/ouro-mainline/unified-programmable/fixtures.json`
- `proxima-tensor/specs/ouro-mainline/unified-programmable/models.json`
- `proxima-tensor/specs/ouro-mainline/unified-programmable/WORK-PACKET-CONFIG.md`
- `proxima-tensor/specs/ouro-mainline/unified-programmable/WORK-PACKET-EXECUTOR.md`
- `proxima-tensor/specs/ouro-mainline/unified-programmable/WORK-PACKET-SERVING.md`
- `proxima-tensor/specs/ouro-mainline/unified-programmable/WORK-PACKET-E2E.md`
- `proxima-tensor/specs/ouro-llama-oracle/EVIDENCE.md`
- `/Users/brianbruggeman/.cache/proxima/models/Qwen3-1.7B/config.json`
- `/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80`
- `/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`
- `/private/tmp/ouro-llama-converted-v1/ouro-bf16.gguf`
