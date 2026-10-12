# existing prefill and decode scheduling seams

Scope: read-only inspection of the current shared generation path in checkout
`39e1d3fb3d18d6650bd4391b05330898f440af2b`. The checkout is dirty from the
active shared implementation. No scheduler source was changed for this
inventory. These pointers describe the code read in this checkout.

## Current request path

`LoadedModel::generate_with_serving_config` applies defaults and admission,
constructs a fresh `BackendRuntime`, then enters one decode loop
(`proxima-model-interop/src/generate/decode.rs:2134-2153`). The loop tokenizes
one prompt and runs prefill followed by autoregressive steps. The public
streaming method changes the token callback, but invokes this same loop
(`decode.rs:2711-2749`). Its callback is synchronous and `ControlFlow::Break`
ends that generation call; it does not suspend a request and return a
continuation.

Two current settings record the missing scheduling capabilities. `parallel_sequences`
may only be `1`: `apply_serving_config` returns a typed `UnsupportedServingConfig`
for another value because evaluation currently runs one sequence per call and
there is no per-slot KV cache or cross-slot request scheduler
(`proxima-model-interop/src/serving.rs:811-812, 1345-1361`). `PhaseSchedule`
defaults to prefill-before-decode; the decode entry rejects `false` with a typed
error saying cross-sequence prefill/decode interleaving is not implemented
(`serving.rs:707-725`; `generate/decode.rs:3120-3135`). Thus there are
separate config names for request admission and phase order, but no multi-stream
execution path behind either setting.

The state needed to resume a stream currently lives on the stack of
`run_decode_loop_from_ids`: it takes a seed `PrefixState`, builds mutable token
history, sampling configuration and RNG, owns per-layer caches and
`cached_len`, and keeps `next_ids`, pending speculative IDs and scratch buffers
through the loop (`generate/decode.rs:3014-3067, 3174-3190, 3412-3423,
3551-3567`). It returns a new `PrefixState` only at the end of the loop
(`decode.rs:6071-6106`). `PrefixState` itself owns token IDs, layer-cache states
and the cached length (`generate/residency_caches.rs:903-925`). Public
`prefill_prefix` finishes its prefill before returning that state
(`decode.rs:2156-2209`); `generate_from_prefix` clones it into another complete
decode call (`decode.rs:2212-2278`). Neither exposes a one-chunk/one-token
continuation step.

`BackendRuntime` is also created per public request, with mutable plan caches,
backend selection and runtime numeric settings. Evaluation requires
`&mut BackendRuntime` (`generate/residency_caches.rs:1806-1885, 1918-1945,
1973-1983`). The decode loop calls it synchronously for each prefill batch or
decode step. A same-device scheduler could serialize these bounded evaluations
on one runtime; the source inspected here does not show an API that yields a
request continuation between them. Nothing in this inventory assumes separate
devices or simultaneous GPU execution.

## Prefill chunking, including Ouro

`ServingConfig::ubatch_size` is the per-evaluation prompt row limit for
architectures with `single_position_step == false`; `0` means whole-prompt
evaluation, and speculative verify batches are not chunked
(`proxima-model-interop/src/serving.rs:820-832`). In the decode loop,
single-position architectures, speculative steps, and cacheless architectures
are not sent through `ubatch_prefill_chunks`; otherwise the prompt is divided
into `ceil(prompt_rows / ubatch_size)` row slices
(`generate/decode.rs:3686-3712`). The loop evaluates those slices in order and
requests logits only from the final prefill batch before entering decode
(`decode.rs:3714-3729, 5039-5070, 5551-5563`). This is chunking within one
request, not a yield to another request between chunks.

The Ouro architecture currently binds `single_position_step: false`
(`proxima-model-interop/src/ouro.rs:193`). Therefore Ouro takes the ordinary
`ubatch_size` chunk path. It is not eligible for the separate
`prefill_one_evaluation` alternate program, whose predicate requires
`self.single_position_step` (`generate/decode.rs:3686-3703`); that opt-in exists
for a single-position architecture only (`serving.rs:1036-1046`). An Ouro
prompt split into ubatches is still completed before its decode stream starts.

## Reusable Pipe primitives read

The pipe source index lists 23 combinators in `ai_docs/examples-index.jsonl`.
The relevant API shapes and examples were read from
`proxima-primitives/src/pipe/primitives.rs`, `fan_in.rs`, `filter.rs`,
`demand.rs`, `bounded.rs`, and `examples/fan_in/main.rs`, `examples/filter/main.rs`,
`examples/gate/main.rs`, `examples/backpressure/main.rs`.

| Primitive | Scheduler relationship |
|---|---|
| `FanIn` | Candidate for fair selection of ready per-request step sources only after a request step is represented as a pollable source. Its sources are `UnpinPipe<In=(), Err=Exhausted>` and it returns the first ready owned item; it does not run sources concurrently (`fan_in.rs:1-15, 165-189`). The present decode loop is not such a source and does not yield during its inner evaluation. |
| `BoundedQueue` / `BatchSource` | Candidate for bounded admission/work buffering with an explicit overflow policy. It does not own KV state or schedule GPU evaluation. `BoundedQueue` is a heap-backed alias over `proxima_core::ring::HeapBoundedQueue` (`bounded.rs:1-15`); the backpressure example demonstrates FailClosed and drop policies (`examples/backpressure/main.rs:1-30`). |
| `Filter` / `AndThen` | Candidate for pure request admission before enqueue. `Filter` is a typed `In -> Result<In, Err>` decision composed before the inner pipe; it does not maintain stream progress (`filter.rs:1-26`; `primitives.rs:194-250`; `examples/filter/main.rs`). |
| `Demand` / `AtomicGate` | Candidate for readiness/dormancy control around a scheduler sink. A closed gate returns `Ok(())` without calling the inner sink (`demand.rs:1-15, 114-134`), so it is not itself a lossless wait-queue or fairness policy. |
| `Pipe` / `AndThen` | Base contract if a future step operation is made a typed transform. The trait is one `call(&self, In)` returning `Future<Result<Out, Err>>`; `AndThen` chains typed stages (`primitives.rs:38-70, 194-250`). Current evaluator entry is a synchronous `&mut BackendRuntime` call, so this is not a drop-in conversion by itself. |

The remaining indexed combinators were checked against their indexed
`use_when` and do not schedule per-request GPU steps in the current question:
`FanOut` duplicates one input to sinks; `best-effort` chooses lossy queue
overflow; `RateLimit` throttles per-key admission; `Retry` retries classified
failures; `Fallback` routes an error to an alternate pipe; `CircuitBreaker`
short-circuits a failing dependency; `codec` transforms wire frames; `Deadline`
tests expiry against an injected clock; `delivery` carries delivery guarantees;
`cache` does lookup/fallthrough/write-back; `record` and `replay` persist or
serve cassettes; `plugin-skeleton` packages a pipe; `load-balance` chooses one
backend from several; `multi_runtime` drives the same pipe under multiple
executors; `Signal` and `cancellation` coordinate lifecycle; `Chaos` injects
test faults. Their indexed locations and examples are in
`ai_docs/examples-index.jsonl`; no claim is made that these primitives cannot
serve adjacent request concerns such as cancellation, timeouts or multi-device
selection.

## Scheduling distinction

Three distinct things must remain distinct in any later design: (1) graph phase
inside one forward (attention/FFN/LoopCD), (2) per-request continuation state
(prefix IDs, KV, cached length, sampler/RNG and prompt/decode progress), and
(3) cross-request batching that combines compatible rows into one backend
evaluation. The current code exposes within-request ubatch splitting, but no
request continuation step and no cross-request packed decode API. A single
device can still be the serialized executor for bounded work from multiple
streams; claiming parallel GPU execution would require separate evidence.

## Proposed scheduling shape

```text
incoming prompts --> bounded prefill chunks --+
                                             |
ready decode streams --> compatible row batch +--> shared Omega executor
          ^                                  |       + shared weights
          |                                  |
          +------- emitted token / EOS <------+
                  per-stream KV + sampler
```

The separation is between queues and resumable state. It does not require a
second transformer implementation or duplicate model weights. The scheduler
would select bounded work; Omega would lower the selected batch for its backend
and resource constraints. Decode batching also requires explicit sequence
boundaries and per-stream positions/cache destinations, so packing token IDs
into today's single-sequence input would mix attention histories.

Prefill chunk size and decode service budget interact: a large prompt chunk
holds the executor longer, while a smaller matrix may miss a useful tile shape.
The existing measured tile floor is a property of the current experimental
kernel selection, not a universal scheduling constant. Admission and a finite
prefill budget must also prevent new prompts from waiting indefinitely under
continuous decode load.

The retained checkpoint cohorts execute single requests. They contain no mixed
arrival workload and cannot measure this proposed schedule's throughput,
first-token latency, inter-token latency, fairness or per-stream cache isolation.

## Cache-aware selection: existing substrate and proposed use

The owner asked whether guesses or Bloom membership could favor decodes whose
KV is already available. Two existing mechanisms answer different parts of that
question:

- `generate/block_bloom.rs:36` defines `BlockBloom` over token-block content
  hashes. `PromptCache::bloom_candidates` at `prompt_cache.rs:865` checks entries
  with an equal `CacheKey`, then reports candidate blocks beyond the reused
  prefix. `prompt_cache_lookup` at `:1095` consumes that result as telemetry;
  it does not select a ready decode stream.
- `generate/prefix_trie.rs:1` indexes shared prefixes. Its hash-table hits are
  verified against the stored token IDs (`:22`); token content hashes alone do
  not authorize reuse. `prompt_cache_key.rs:28` also requires matching execution
  configuration; a cache entry belongs to its loaded model.

For active streams, the proposed scheduler should use its exact KV ownership,
valid ranges, device placement and completion state first. Those records are
already the authority it needs to execute a step. A Bloom filter can screen
larger cache inventories or remote candidates when that saves lookup work;
every positive still requires exact identity, validity and residency checks.
A miss may change placement or order, but must not discard a runnable request.
An unpopulated or stale directory cannot be treated as proof of absence.

Content-only membership is insufficient for causal KV reuse: the preceding
context and position affect the states behind matching token blocks. A future
block directory must retain that provenance, model/config identity, logical
layer/recurrent visit and allocation generation. Predicting a future token is
not a substitute for the next forward pass.

Within compatible ready streams and explicit service deadlines, confirmed
resident KV can favor a decode batch that needs fewer transfers/restores.
Bounded aging and prefill service prevent repeated resident hits from starving
new or evicted streams. No priority weights or benefit are claimed here.
The comparison must hold arrivals, requests and policies fixed and retain each
selection's candidates, residency snapshot, Bloom answer, exact lookup result,
transfers, wait, emitted IDs and per-token times. Report false positives,
first-token/inter-token latency, throughput and repeated-run CoV separately.
The current `bloom_candidates` allocates and scans entries and prompt blocks;
its present implementation is not evidence of a zero-allocation scheduling path.

This is a proposed adaptation of the existing cache and shared pipeline, with
no cache-aware multi-stream scheduling measurement or implementation claimed.

## Additional performance levers

The owner's next question is what remains beyond avoiding repeated work. The
following are proposed experiments, not measured improvements. They extend the
same graph, planner, backend and serving owners.

| Lever | Mechanism to investigate | Current evidence and boundary |
|---|---|---|
| Decode-sized kernels | Select matrix-vector or small-row matrix kernels, layouts and workgroup sizes for the actual decode shape; inspect weight bytes read and occupancy before choosing | C7's captured BF16 matrix tile changes long prefill. Its decode rows do not show the same reduction; see EVIDENCE.md's C7 table. A prefill tile is not automatically useful for a single row. |
| Continuous decode batching with bounded prefill chunks | Amortize weight reads and launches over compatible streams while bounding each prompt chunk's executor occupancy | The continuation and cross-stream attention/cache boundaries described above are absent. This targets throughput and service latency under arrivals; it does not establish faster isolated decoding. |
| Reduce intermediate memory traffic | Fuse eligible attention, normalization, activation and projection epilogues; keep temporary values in registers/shared memory where the backend supports it | Cached attention fusion is already enabled in the fresh configured cohort. Audit actual emitted kernels and intermediate lifetimes for coverage gaps, not the setting name. Every rewrite remains subject to numeric-policy permissions; the broadcast-fold repair records why. |
| Reduce host execution overhead and boundaries | Reuse prepared execution state, reduce encode work, and evaluate device-side token selection for requests that do not need all logits on the host | The production placement path commits then waits at placements_execute_named.rs:1594-1607. finish at arena_encode_dispatch_finish.rs:2310 skips placed non-root outputs but still reads the root. decode.rs:1321 selects from a host logits slice. These source boundaries are candidates, not evidence they dominate elapsed time. |
| Overlap independent work | Prepare a following batch or move required data while an earlier batch executes, with explicit buffer lifetimes, completion and backpressure | The placement driver already contains configurable command-buffer chunking and overlap instrumentation. The retained cached cohort uses one chunk. Do not infer a need for another executor or claim overlap from an async API alone. |
| Verify drafted tokens in batches | Use the existing speculative verification seam to amortize full-model evaluations when accepted drafts outweigh drafting and verification costs | Existing select_decoded_token is shared by ordinary and speculative paths. A new workload comparison must retain every draft, acceptance/rejection, output token and cost; draft acceptance alone is not a latency result. |
| Narrow weight/KV storage where admitted | Reduce bytes transferred/read, using native backend kernels and formats; cache conversions outside the hot path | Precision changes require separate numerical and answer-quality evidence. An eight-bit format name does not establish native arithmetic support, nor can causal KV be dropped merely because it appears cold. |
| Group MoE work by selected expert | Pack rows already routed to the same expert, reuse resident expert weights and investigate measured transfer prefetch | This is a proposal for the MoE workloads. It preserves model routing; no expert frequency, transfer cost or gain has been measured by the Ouro dense cohorts. |

The proposed investigation order is decode dispatch/memory/host attribution for
single-conversation latency, and resumable streams plus compatible row packing
for serving throughput. Speculation and precision changes are separate paired
experiments. Bloom screening complements placement; it does not remove the
forward pass needed to compute a new token's KV.

The relevant primary algorithms are
[Sarathi-Serve](https://arxiv.org/abs/2403.02310) for chunked-prefill scheduling,
[FlashAttention](https://arxiv.org/abs/2205.14135) for tiled attention IO, and
[speculative decoding](https://arxiv.org/abs/2211.17192) for target-verified drafts.
Their reported hardware/model results are not Proxima measurements.

Measurement must retain dispatch shapes, source identity, byte movement,
allocation/plan events and host/GPU timestamps joined to each request and token.
GPU_EXEC_TICKS is host commit-through-wait duration, not device busy time
(placements_execute_named.rs:1634). The cached log's token_stages counters are
not an additive critical-path breakdown. Compare identical model, tokens,
precision, context, concurrency and instrumentation against the llama oracle;
retain both gains and losses, raw durations, sample CoV, first-token/inter-token
latencies, total wall time, throughput and resource records.

Resource utilization is diagnostic. A rewrite may improve wall time by moving
fewer bytes while lowering a utilization percentage. Acceptance therefore uses
elapsed work and output behavior under the supplied constraints, without a
promise that every hardware unit is busy.

## Parallelism and the unified configuration boundary

Owner clarification: parallelization and the performance levers above belong
inside Proxima's programmable interface. They are shared integrations and
configuration policies, not an Ouro-specific optimization collection.

The current source exposes only part of that contract:

| Current surface | Observed execution boundary |
|---|---|
| `omega::RuntimeConfig` | Settings derive, serde and fluent builder exist in `omega/src/config.rs:15`; fields cover backend, fusion, numeric permissions, memory constraints and candidate selection. There is no request scheduler or parallelism budget in this struct. |
| CPU workers | `proxima-tensor/src/cpu/gemm_q8k.rs:126` reads `PROXIMA_MATMUL_WORKERS` once through `OnceLock`; valid nonzero values override the core-count default. Row parallelism also requires the workload threshold and enough rows (`:144`). This control is outside the serving/Omega settings structures. |
| `ServingConfig::dispatch_type` | Metal-specific encoder policy at `serving.rs:950`; forwarded to plans by `generate/residency_caches.rs:2072`. `omega/src/backend.rs:1400` applies it to Metal and does nothing in CPU/WGPU/CUDA arms. Concurrent dispatch is not cross-request serving. |
| `ServingConfig::command_buffer_chunks` | Submission splitting reaches the Metal plan through `residency_caches.rs:2073` and `omega/src/backend.rs:1425`; this is a backend mechanism, not a portable count of parallel requests. |
| `parallel_sequences` and phase scheduling | Values requesting multiple slots are rejected at `serving.rs:1355`; cross-request prefill/decode interleaving is rejected at `generate/decode.rs:3126`. Their data fields do not imply an implemented scheduler. |
| `overlap_transfer_compute` | `serving.rs:1129` explicitly documents an unwired overlap arm. Searching that field in serving and decode finds declarations/defaults/fixtures, not a production consumer in decode. |
| KV precision and speculation | The settings represent more options than the current serving implementation accepts. KV admission at `serving.rs:1364` requires F32; speculative admission at `:1507` rejects unwired draft types. |
| Backend/device selection | `omega/src/backend.rs:452` plans one configured candidate or measures backend alternatives and selects a winner. It does not partition that plan across the candidates. The Metal device/queue owner at `omega/src/metal/device_buffers_arena_plan.rs:726` obtains the system default device, with no device-selection input at that boundary. |

The requested adaptation is one composed configuration entry point over these
existing owners. Serving owns bounded admission, resumable sequence state,
fairness and batch formation. Omega owns legal graph scheduling, resource
placement and backend lowering. The backend owns hardware-specific dispatch.
These responsibilities remain composable beneath the common entry point.

Parallelism must describe distinct opportunities: lanes/tiles within a kernel,
independent graph operations, independent requests, transfer/compute overlap,
and device partitioning where an implementation exists. CPU worker limits,
permitted devices/queues, in-flight work and memory limits are resource inputs.
Autoregressive token dependencies and successive Ouro refinement states remain
dependency edges; requesting parallelism cannot erase them.

Policy should express the objective and permitted strategies, with explicit
overrides for experiments. Backend-specific tile sizes, launch geometry and
queue schedules should be chosen from supported candidates using shape,
dependency/liveness information and measurements under those constraints.
Semantic permissions such as reassociation or reduced precision stay separate
from performance selection. Required unsupported behavior must be rejected;
automatic selection must report its effective choice and exclusions.

This rules out adding another collection of disconnected booleans or a second
executor. Config-file and fluent construction must resolve to the same effective
plan. Defaults at runtime derive from the build-time floor where applicable;
restricted tiers use generated constants. An execution record needs requested
policy, effective backend schedule, resource use and rejected alternatives so
the user can see whether a knob actually affected execution. This section
records the requested contract and source gaps; it does not claim they have
been implemented or benchmarked.
