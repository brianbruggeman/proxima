# mainline experiment caller and cached oracle

status: independently admitted; implementation validation in progress
owner: root source changes; delegated native build and execution

## problem

Replace the lab Model's private layer loop with LoadedModel, preserve all four
paper modes on two frozen prefixes, and measure cached production generation
against pinned llama on identical 24-token and 256-token input sequences with
up to 64 generated tokens, one cold/warmup record and five measured requests per
case. Record actual token counts when EOS ends a request earlier.

## refutation condition

A lab transformer body in the production call chain, relabeled uncached timings,
missing generated token IDs, dropped failed samples, or different input IDs
between runtimes refutes the corresponding requirement.

## requirements

| id | requirement | isolated |
|---|---|---|
| C1 | Lab Model loads and evaluates through shared model interop; no Executor or graph::layer invocation remains in Model. | yes |
| C2 | Baseline, fixed-logit, adaptive-logit and fixed-hidden modes consume admitted shared readout roots and preserve complete vector controls. | yes |
| C3 | Cached performance calls existing generate_from_ids; llama clears once per request then appends each token without clearing between steps. | yes |
| C4 | Two frozen prefix cases, five measured samples and one warmup per case emit token IDs, per-step times, total wall, resource samples, errors and CoV. | yes |
| C5 | Existing Omega runtime configuration reaches the production planner or produces an explicit unsupported-config error before execution; no ignored knobs. | yes |
| C6 | Exercise the configured production planner with the existing dense-float tile gate off and on, preserving default tile threshold and identical requests/numeric policy. Retain both token trajectories and full diagnostic vectors. | yes |
| C7 | Repeat the configured gate pair with 256-token production microbatches matched to the pinned llama context, and capture the actual step-zero Metal dispatches and source for each arm. | yes |

## architecture

Root owns examples/ouro_loopcd_lab/{model,checkpoint,performance}.rs,
examples/ouro_loopcd_lab.rs, examples/ouro_llama_oracle/main.cpp and common
planner forwarding in interop/src/generate/residency_caches.rs. Coordinate
ServingConfig/prefix-key additions with Luna; Sol owns tensor graph.
The example's direct native-parser dependency is added using
`cargo add proxima-safetensors --path proxima-safetensors --dev`; the first caller
build exposed its absence. No manifest is edited by hand.

- Keep Checkpoint as mmap/provenance/tokenizer fixture ownership, but use native
  proxima_safetensors::parse_complete and interop HfConfig/descriptor binding for
  model loading. No independent model weight binder executes the transformer.
  Existing packed projection microbench and tiny independent graph controls keep
  their direct inputs and are labeled reference-only.
- Model caches LoadedModel by resolved Arm outside timed calls. Map Guidance to
  RecurrentConfig. Request final logits always; request reference/raw/normalized
  roots for diagnostic records. Retain mode-neutral Readouts for current callers.
  Guided raw hidden state and adaptive strength are extracted from the existing
  graph nodes exposed in RecurrentReadoutRoots. Probability and margin metadata
  remains labeled as independently derived from the actual terminal logits.
  Completed-depth roots provide full
  boundary vectors; do not synthesize per-layer execution traces from shapes.
- Baseline diagnostic reference logits use a separately requested first-pass head
  diagnostic; normal baseline timing requests terminal head only. Hidden mode
  requests raw first/terminal and the guided normalized state; never substitute
  normalized terminal for raw terminal.
- Add --cached-performance to the existing lab runner and --cached-run to the
  existing llama native runner. Both accept the same frozen JSON panel path and
  max-new-tokens=64. Existing full-forward mode remains explicitly labeled.
  Freeze the long prefix by cycling the existing 24-token prefix and truncating
  to 256 IDs; identify this as a synthetic length-control, not a quality corpus.
  The short case uses those original 24 IDs exactly.
- Production uses greedy sampling, F32 KV, depth four, same source checkpoint,
  prompt cache off per measured request, speculative decoding off. Each sample
  starts empty KV, one prefill then autoregressive steps; output differences are
  retained. Timings include actual production plan work and readbacks, with setup
  separately timed. First-token latency, subsequent token latencies, total wall,
  generated count and input count are explicit. No throughput ratio across
  different output counts without reporting those counts.
- Extend native oracle's existing run_oracle body and reuse model/context setup;
  context/batch capacity derives from max panel length plus output budget. Keep
  full-forward behavior unchanged. Preserve actual per-step logits in a separate
  untimed diagnostic run; timing samples retain generated IDs and boundaries.
- Reuse existing runner::stats sample standard deviation (N-1 denominator) and
  CoV=100*sample_stddev/mean; label the convention in both runtimes. Mean zero yields
  null with reason. Include latency percentiles, CPU seconds/RSS, errors and
  one-request measurements from actual caller callbacks. Never drop slow samples.
- Omega config is borrowed in ServingConfig and cloned once by BackendRuntime.
  Ordinary plan misses call plan_named_with_config, retaining candidate records
  through structured telemetry. None preserves old planning. Model performance
  uses configured mode; measured selection with external mutable placements is
  rejected before work until scratch ownership exists. Constraints retain their
  current typed unsupported-operation behavior. Placed-only paths must reject
  unsupported config rather than silently switch planner. Prefix keys include
  numeric/fusion/backend identity; configured calls disable plan refit unless
  refit revalidates resource bounds. No new executor/plan-cache owner.

### decisions

Rejected retaining the lab's loop behind another wrapper: the shared loader and
serving loop must own the work. Rejected translating only numeric fields while
discarding configuration constraints. Rejected using a graph clone as independent
paper evidence: source literals and retained incumbent payloads remain references.

## acceptance criteria

| id | discharges | native command | expected |
|---|---|---|---|
| CAC1 | C1,C2 | `cargo nextest run --example ouro_loopcd_lab --features proxima-tensor -E 'test(mainline_caller_)' --no-capture` | Exactly 4 cases: guidance_mapping, root_extraction, cached_panel, sample_statistics; 0 skipped; hand literals and wrong-root/control assertions retained. |
| CAC2 | C1,C2 | `cargo check --example ouro_loopcd_lab --features proxima-tensor --message-format=json` | 1 ouro_loopcd_lab example artifact, 0 compiler errors. |
| CAC3 | C3,C4 | `cargo run --release --example ouro_loopcd_lab --features 'proxima-tensor omega/metal-tiled-gemm' -- --cached-performance --backend metal --samples 5 --max-new-tokens 64 --output proxima-tensor/specs/ouro-mainline/results/proxima-cached` | Exactly 2 cases, 12 request records including 2 warmups and 10 measured samples; every actual generated token has an ID and latency. |
| CAC4 | C3,C4 | `/private/tmp/proxima-ourollama-oracle-build/ouro_llama_oracle --cached-run --model /private/tmp/ouro-llama-converted-v1/ouro-bf16.gguf --panel proxima-tensor/specs/ouro-mainline/results/proxima-cached/cached-panel.json --max-new-tokens 64 --samples 5 --output-dir proxima-tensor/specs/ouro-mainline/results/llama-cached` | Exactly 2 cases, same prefix IDs, 12 request records including 2 warmups and 10 measured samples; complete actual generated ID/latency arrays. |
| CAC5 | C5 | `cargo nextest --config-file proxima-tensor/specs/ouro-mainline/results/shared-graph/validation/native-unit-nextest.toml run -p proxima-model-interop --features 'std metal' --lib -E 'test(mainline_omega_)' --no-capture` | Exactly 3 cases: configured_forwarding, unsupported_placement, cache_identity; 0 skipped; constraint refusal and wrong-policy controls retained. |
| CAC6 | C5,C6 | Two configured commands plus native cached recount below | Each arm: 12 requests, 2 diagnostics, 2 summaries; recount: 36 request rows, 6 timing cells, 4 signed diagnostic comparisons, 24 trajectory pairs, 1 equality/corruption control. |
| CAC7 | C5-C7 | Two matched-batch configured commands, matched llama command, and native cached recount below | Each arm: 12 requests, 2 diagnostics, 2 summaries, 12 begin/end profile brackets, zero profile drops, and step-zero dispatch/source evidence; recount: 36 request rows, 6 timing cells, 4 signed diagnostic comparisons, 24 trajectory pairs, 1 equality/corruption control. |

Native oracle build uses existing CMake target, delegated. Real timing runs never
overlap GPU work. Successful process exit alone does not satisfy the expected
record counts. Runner itself validates sample/array lengths and reports failures.

### configured end-to-end checkpoint

Preregistered before executing either configured arm: the existing dense-float
tile gate may change prefill work at the 256-token shape; the default 160-token
threshold keeps single-token decode outside that tile path. This is a hypothesis,
not a throughput result. Record emitted kernels when attributing a latency change.
No increase in default numeric permissions or threshold override is introduced.

CAC6 command, executed once for each gate value `0` and `1` with respective fresh
directories `proxima-configured-off` and `proxima-configured-on`:

```text
env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0 cargo run --release --example ouro_loopcd_lab --features 'proxima-tensor omega/metal-tiled-gemm' -- --cached-performance --backend metal --samples 5 --max-new-tokens 64 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/ouro-mainline/results/proxima-configured-off
env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 cargo run --release --example ouro_loopcd_lab --features 'proxima-tensor omega/metal-tiled-gemm' -- --cached-performance --backend metal --samples 5 --max-new-tokens 64 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/ouro-mainline/results/proxima-configured-on
```

Expected: 12 request records and 2 diagnostic records in each arm; 2 cases with
matching prefix IDs and budget, complete actual output IDs/latencies, 5 measured
samples per case, sample CoV and retained failures. Full diagnostic coordinate
differences and both output trajectories are evidence; matching argmax alone
does not establish numerical preservation. This discharges C5-C6 through the real
production caller. The default/no-config CAC3 run remains a separate condition.
These arms use baseline recurrence: terminal and generation logits are the same
root and no reference head is requested. The complete diagnostic vector is all
49,152 coordinates of that root at the initial prefix. C2's guided and hidden
readout checks remain separate; C6 does not relabel this baseline as LoopCD.

### matched-microbatch tiled checkpoint

CAC6 used `ubatch_size=32` from `ServingConfig::default()`. The production
decode loop splits a 256-token prefix into 32-row evaluations
(`proxima-model-interop/src/generate/decode.rs:3707,3733-3735`), so those timed
prefills do not reach the default tile floor of 160. CAC6 is retained as the
below-floor control. Its separate untimed full-prefix diagnostic does not count
as timed tile coverage.

CAC7 repeats the same gate-off/gate-on comparison with production
`batch_size=256`, `ubatch_size=256`, and `ContextLength::Within(320)`. The lab's
existing `--cached-performance` mode records these structured setup fields;
there are no batch/context CLI overrides. The pinned llama runner's existing
`--cached-run` mode sets `n_ctx=256+token_budget` (320 for this 64-token cell),
`n_batch=256`, and `n_ubatch=n_batch`, and emits the values in `run_context`.
No source or threshold override is part of this cell.

Each proxima arm uses the existing `--gpu-profile` recorder with identical
decode, arena-dispatch, pipeline-create and bracket-marker filters. The recorder
drains outside request timers. Its output is a new instrumentation-on timing
condition: compare gate arms within CAC7 and retain all latency records, but do
not compare CAC7 timings against CAC3/CAC6 as a one-variable speed change or
present them as an uninstrumented llama/proxima performance ratio. Both gate
arms carry the same profiling overhead. `gpu-events.jsonl` contains one
begin/end marker pair for each warmup and measured request; no dropped records
are admitted.

Create these fresh capture directories before running each proxima arm:

```text
mkdir -p proxima-tensor/specs/ouro-mainline/results/cac7-pipeline-off-r3
mkdir -p proxima-tensor/specs/ouro-mainline/results/cac7-pipeline-on-r3
```

Run each proxima command once, serially, with the same pinned panel, config,
sample count and budgets:

```text
env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_PIPELINE_CAPTURE=proxima-tensor/specs/ouro-mainline/results/cac7-pipeline-off-r3 cargo run --release --example ouro_loopcd_lab --features 'proxima-tensor omega/metal-tiled-gemm omega/instrument' -- --cached-performance --gpu-profile --backend metal --samples 5 --max-new-tokens 64 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-off-r3
env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 PROXIMA_PIPELINE_CAPTURE=proxima-tensor/specs/ouro-mainline/results/cac7-pipeline-on-r3 cargo run --release --example ouro_loopcd_lab --features 'proxima-tensor omega/metal-tiled-gemm omega/instrument' -- --cached-performance --gpu-profile --backend metal --samples 5 --max-new-tokens 64 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-on-r3
```

Run the native arm with matching batch dimensions and the frozen panel emitted
by the off arm:

```text
/private/tmp/proxima-ourollama-oracle-build/ouro_llama_oracle --cached-run --model /private/tmp/ouro-llama-converted-v1/ouro-bf16.gguf --panel proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-off-r3/cached-panel.json --max-new-tokens 64 --samples 5 --output-dir proxima-tensor/specs/ouro-mainline/results/llama-matched-batch-r3
```

The `--cached-matched-batch` recount mode consumes these three JSONL inputs and emits the same
36 request payload rows, 6 timing cells, 4 full signed diagnostic comparisons,
24 trajectory pairs and 1 equality/corruption comparator control as CAC6. In
addition it requires each proxima setup to report `batch_size=256`,
`ubatch_size=256`, and `context_size=320`, equal across gate arms, and the llama `run_context` to report
`context_size=320`, `batch_size=256`, `ubatch_size=256`. It pins both prefix
cases and the full token/diagnostic comparison to the frozen panel. The paired
GPU records must include at least one step-zero dense multi-token projection dispatch
per arm, joined by node to its bound-op extents and by MSL SHA to the exact
`pipeline_<first-16-sha-hex>.metal` source in that arm's capture directory;
verify the file's full SHA against the full dispatch digest. The dispatch event
must fall between the begin/end markers carrying the same sample ID. Preserve
the entry name, node, step, extents, grid and pipeline source hash; a build flag or
the untimed diagnostic alone is not dispatch evidence.

The first C7 attempts used the `packed-multi-token` capture selector. That
selector calls `classify_packed_row_block`, whose codec whitelist explicitly
rejects BF16/Float16 (`omega/src/msl/emit_and_classify.rs:2499`). It therefore
excluded these dense projections. The initial explanation that instrumentation
was compiled out is retracted: the root dependency already enabled that feature
transitively. Both replacement arms capture `all` nodes at step zero, using
fresh suffixed directories. Request markers and pipeline source files alone do
not satisfy the dispatch criterion. Partial attempts and actual commands remain
archived beside the results; no partial timings enter the completed cohort.

The old CAC6 off/on records remain unchanged and are labeled
`ubatch_size=32`, below the 160-token floor for timed production prefill. CAC7
is a separately named instrumentation-on condition with no substitution into
those records.

The existing native recount gains `--cached` input support:

```text
cargo run --example omega_programmable_recount -- --cached proxima-tensor/specs/ouro-mainline/results/proxima-configured-off/cached.jsonl proxima-tensor/specs/ouro-mainline/results/proxima-configured-on/cached.jsonl proxima-tensor/specs/ouro-mainline/results/llama-cached-r2/llama.jsonl proxima-tensor/specs/ouro-mainline/results/cached-recount
```

For CAC7, use the matched-work-shape guard and CAC7-specific outputs:

```text
cargo run --example omega_programmable_recount -- --cached-matched-batch proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-off-r3/cached.jsonl proxima-tensor/specs/ouro-mainline/results/proxima-matched-batch-on-r3/cached.jsonl proxima-tensor/specs/ouro-mainline/results/llama-matched-batch-r3/llama.jsonl proxima-tensor/specs/ouro-mainline/results/cached-matched-batch-recount-r3
```

It checks 12 requests (10 measured, 2 warmups), 2 diagnostics, 2 summaries and
one completion per input, unique sample repetitions 0 through 4, model and source
checkpoint SHA256, exact prefix IDs/budget, ID/latency counts and finite values.
Independent Welford wall/TTFT sample statistics carry source lines. Paired output
includes every generated ID from both runs and every changed diagnostic
coordinate with both values and signed difference. Identical vectors and a
single-coordinate corruption exercise the same comparator. Expected output
counts are 36 request payload rows and 6 timing cells, then 1 comparator-control
row, 4 full diagnostic comparisons and 24 request trajectory pairs. Native
runner output directories may gain retry suffixes; actual commands retain those
paths in the evidence log rather than replacing failed cohorts.

## slices

### Validation repairs discovered after C7

The existing full four-mode caller must execute all 48 warmup/sample requests
(four modes, two frozen prefixes, one warmup plus five samples). Its first
attempt was rejected because Model requested bit-exact binding but inherited
ServingConfig's Relaxed Metal math mode. Expose the existing Omega
`numeric_policy_as_metal_math_mode` projection through `MathMode::for_numeric_policy`
and select the caller's mode from its resolved policy. Reuse the existing subset
logic; no permission widening or new selection heuristic is admitted. Extend
the existing four caller cases with exact/relaxed/partial-permission assertions.

The offline C7 recount also exposed JSON float parsing and large-offset variance
rounding. The parsed value `37978.240083000004` became `37978.240083`; reported
SD 26.5043894672330396 differed from independent 26.5043894672311495. Enable the
already locked serde_json version's `float_roundtrip` feature using native
`cargo add`, retaining alloc/no-default settings. Translate Welford's inputs
by the first sample before accumulating moments; translation leaves variance
unchanged and avoids subtraction against the large wall-time offset. Keep the
existing comparison bound unchanged. Exactly two native recount tests must
cover serialized input bits, the actual C7 sample values, known sample variance,
and rejection of a deliberately changed summary. Preserve failed recounts.

Native validation command:
`cargo nextest --config-file proxima-tensor/specs/ouro-mainline/results/shared-graph/validation/native-unit-nextest.toml run --example omega_programmable_recount -E 'test(recount_)' --no-capture`.
Expected: exactly two cases, `recount_float_roundtrip` and
`recount_centered_variance`, with zero failures. The existing CAC1 command still
selects exactly four caller cases. The native `--performance --backend metal
--samples 5` invocation against the frozen default panel must emit 48 requests:
eight warmups and forty measured samples, with no suppressed arm failures.

These repairs do not relabel the completed C7 timing condition. A fresh real
caller run and independent method-vector comparisons validate the caller change;
the repaired offline recount processes the immutable original records.

Every implementation checkpoint includes a real-model end-to-end run through
loading, prefill, cached decode and emitted tokens. Unit and graph checks do not
replace this run. Record the exact source condition, command, model identity,
input/output IDs, first-token and decode latency, total wall, resources and errors.
Performance checkpoints run matched llama inputs/output budgets and repeated
samples with sample CoV. Retain failed runs before repairing and repeating them.
This checkpoint uses CAC3/CAC4; later checkpoint commands are recorded alongside
their payloads before another implementation change is measured.

| slice | requirement | validation | expected |
|---|---|---|---|
| migrate caller/config and instrument cached requests | C1-C6 | CAC1-CAC6 commands above | 4 caller cases, 3 planner cases, 1 example artifact, 12 requests per runtime/arm and the CAC6 recount counts |
| prove timed dense tiling at a matched 256-row microbatch | C5-C7 | CAC7 paired commands and recount above | 12 requests and 2 diagnostics per runtime/arm; batch/ubatch fields equal 256; matched llama context 320; paired profile brackets without drops; step-zero dispatch joined to emitted MSL source; exact recount counts |

## out of scope

- No new transformer, cache, backend, sampler, model training or publication.
- This length-control cohort does not claim downstream task quality; existing
  paper/vector controls remain separate from throughput evidence.

## risks

| risk | response |
|---|---|
| EOS causes unequal work | Preserve actual counts/tokens; compare time per observed work with explicit conditions. |
| Diagnostic roots change timed workload | Separate diagnostic and timed calls, record requested roots per condition. |
| Independent runtime numeric mismatch | Preserve full signed vectors under registered ORACLE-SPEC tolerance, no widening. |
| GPU contention corrupts variance | One resident runner; raw resource and sample records retained. |

## context

WORK-PACKET.md; INTEROP-SPEC.md; SHARED-GRAPH-SPEC.md;
`examples/ouro_loopcd_lab.rs`; `examples/ouro_loopcd_lab/{model,checkpoint,performance}.rs`;
`examples/ouro_llama_oracle/main.cpp`; `omega/src/{backend,config}.rs`;
`proxima-model-interop/src/generate/{decode,residency_caches,prompt_cache_key}.rs`.
