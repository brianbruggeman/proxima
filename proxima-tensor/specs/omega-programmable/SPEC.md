# Omega programmable configuration and lowering

Status: admitted by independent audit v2; measurement command amendments are
recorded separately before execution. Paths and commands below are relative to the isolated repo
root `/private/tmp/proxima-omega-unified-config` unless absolute.

## problem

Omega's existing programmable graph cannot currently be planned through its `Settings`-derived `RuntimeConfig`, and BF16/F16 projection bindings cannot enter its dense tiled Metal path; this adaptation must expose consumed per-plan policy and measured candidate selection while preserving literal output fixtures and rejecting resource violations before their prohibited allocations.

## refutation condition

The adaptation is refuted if a configured field leaves the corresponding bound
graph/allocation/dispatch unchanged when its test fixture requires a change,
if an unauthorized numeric rewrite is emitted, if a forbidden allocation occurs
before rejection, or if the selected candidate is not the least measured
admissible candidate in the retained selection record. A measured slower tiled
arm refutes that performance hypothesis for that workload; preserve the row
and keep its gate default off.

## intent and boundary

Adapt `RuntimeConfig`, `Op`, binding, the existing concrete `Plan` variants and
their prepare/execute paths. The external input remains a program, symbolic
extents, named blocks, requested outputs, and a typed config. No second IR,
trait-object runtime, model-specific fast-path API, global mutable config, or
environment mutation at execution time is introduced.

The latency objective is the least observed synchronized execution wall time
among measured feasible candidates for the same graph and inputs. It is not a
promise of a globally optimal schedule or universally full hardware occupancy.
Resource ceilings are hard requirements with precisely named accounting scopes;
hardware utilization is an observed diagnostic. Compilation and search cost are
reported separately and amortized only across an explicitly named reuse count.

llama.cpp is the **performance oracle**. Existing Proxima is the before-change
baseline. Literal arithmetic fixtures and the independent model evidence provide
separate numerical checks. A permission to reorder a reduction is not a claim
that resulting logits equal a different implementation's logits.

## requirements

| ID | Independently falsifiable requirement |
|---|---|
| R1 | Existing legacy RuntimeConfig fields remain loadable; file/env/fluent construction resolves the same explicit policy and generated defaults. |
| R2 | Fusion and numeric permissions reach actual CPU/GPU binding; a disabled fusion setting emits separate operations and retains requested roots. |
| R3 | One configured planning entry selects an existing concrete backend, reports actual selection and refuses unavailable drivers without silent fallback. |
| R4 | Every exposed resource ceiling is enforced at its named payload allocation owner, including repeated execution; overflow and unsupported bounds are typed failures. |
| R5 | Measured selection records each synchronized raw sample and rejection, picks the least observed admissible mean with stable ties, and does no measurement in configured mode. |
| R6 | BF16/F16 dense tiled lowering shares the existing graph/classifier/renderer/grid/cache machinery and requires explicit contraction plus reassociation permission. |
| R7 | Literal matrix fixtures traverse actual configured CPU and Metal execution and use a comparator that accepts its independent literal oracle and rejects a one-coordinate corruption. |
| R8 | Experimental float-storage lowering compiles out by default. |
| R9 | The existing Ouro caller consumes Omega config and records the actual policy, backend and output payloads in performance runs against the pinned llama oracle. |
| R10 | Micro and end-to-end measurement artifacts retain every sample, CoV%, errors, resource observations, signed output changes and failed/rolled-back arms. |
| R11 | Core binding/emission retains its alloc tier and runtime I/O stays std-only. |

## architecture and decisions

1. Config resolves once at the cold planning boundary. Existing plan execution
   remains reusable. A caller-owned selection report contains evidence, not a
   second execution object or global cache.
2. Backend-specific prepare/allocation functions own enforcement. A host tensor
   payload cap must not be described as a total RSS cap; driver/compiler metadata
   and borrowed mappings have separate accounting. Unsupported enforcement
   returns a named constraint error before execution.
3. Binding has an explicit unfused route built from the existing builder. The
   cached-attention capability boolean retains its existing separate meaning.
4. Dense float storage uses existing operand decoders and float tiles. New
   storage eligibility is compiled under `omega_dense_float_storage`, generated
   by the existing sizing loader from `[experimental] dense_float_storage=0`.
   The build override is `OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1`.
5. Every changed policy/codec/geometry affects the existing kernel identity.
   Declining a specialization must select compatible source and launch geometry.
6. CPU, Metal, wgpu and CUDA retain their existing driver boundaries and
   operation capabilities. NPU/ANE has no existing driver: it is not an alias
   for CPU or MLX. A future driver must implement actual prepare, resource facts,
   allocation enforcement, synchronization, execution and output extraction at
   this same boundary before becoming selectable.

The constraints ruled out a parallel scheduler/IR, a global settings singleton,
pretend NPU execution, treating a device working-set hint as an enforced cap,
and duplicating the tiled kernel for a particular model. Existing primitives
remain the implementation surface.

## acceptance criteria

Commands run from the isolated root; the implementing worker supplies a unique
`CARGO_TARGET_DIR` to each invocation. Test assertions and oracle values are
specified in the linked work packets. Expected counts are mandatory, including
for the comparator's own positive and negative controls.

| ID | R | Runnable command | Expected count / oracle |
|---|---|---|---|
| AC1 | R1 | `cargo nextest run -p proxima-omega --lib -E 'test(runtime_config_)'` | 12 tests; literal config values and source-order cases in config packet |
| AC2 | R2 | `cargo nextest run -p proxima-tensor --lib -E 'test(programmable_binding_)'` | 8 tests; independent `[12,16]` and `[5,11,17]`; positive comparator zero mismatches and corruption exactly one |
| AC3 | R3,R4 | `cargo nextest run -p proxima-omega --lib -E 'test(config_plan_)'` | 12 tests; Fixture A literal `[11,18,33,36]` accepts itself with 0 mismatches through the actual-output comparator; `[11,18,33,37]` produces exactly 1; resource/planning controls in execution packet |
| AC3b | R4 | `cargo nextest run -p proxima-omega --lib -E 'test(config_cpu_budget_)'` | 5 tests; literal 56-byte envelope, 55-byte refusal, pool trimming and unsupported paths |
| AC3c | R5 | `cargo nextest run -p proxima-omega --lib -E 'test(config_selection_)'` | 5 tests; literal timing arrays, mean comparison, stable ties and rejection retention |
| AC3d | R3,R4 | `cargo nextest run -p proxima-omega --lib -E 'test(config_metal_)'` | 4 real Metal tests; Fixture A literal accepts itself with 0 mismatches and one-coordinate corruption produces exactly 1 through the device-output comparator; physical buffer rows, zero silent CPU fallback |
| AC4 | R6 | `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 cargo nextest run -p proxima-omega --features metal-tiled-gemm --lib -E 'test(programmable_tiled_)' --no-capture` | 8 tests; both rewrite permissions, codec/grid/cache/source consistency and independent `[17,39,23,53]` comparator controls |
| AC5 | R6,R8 | `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0 cargo nextest run -p proxima-omega --features metal-tiled-gemm --lib -E 'test(programmable_tiled_)' --no-capture` | 1 test; both float-storage codecs decline when compiled off |
| AC6 | R7 | `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 cargo nextest run -p proxima-omega --features metal-tiled-gemm --test programmable_execution_vectors --no-capture` | 8 tests: 2 CPU + 4 Metal literal vectors, 1 oracle self-check, 1 corruption control; 0 skipped on this Metal host |
| AC7 | R11 | `cargo check -p proxima-omega --no-default-features --features alloc --message-format=json` | 1 compiler artifact for omega library, 0 error diagnostics; emitter modules built, runtime config excluded explicitly |
| AC8 | R11 | `cargo check -p proxima-omega --no-default-features --features std,cpu --message-format=json` | 1 compiler artifact for omega library, 0 error diagnostics; CPU/config modules included |
| AC9 | R9,R10 | `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 OMEGA_TILED_GEMM_MIN_TOKENS=1 cargo run --release --example ouro_loopcd_lab --features proxima-tensor,omega/metal-tiled-gemm -- --performance --backend metal --samples 5 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/omega-programmable/results/ouro-configured`; paired control: `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0 OMEGA_TILED_GEMM_MIN_TOKENS=1 cargo run --release --example ouro_loopcd_lab --features proxima-tensor,omega/metal-tiled-gemm -- --performance --backend metal --samples 5 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/omega-programmable/results/ouro-configured-control` | Each arm: 40 timed records over 8 prefix/method cells, 8 warmups, 8 CoV rows, full payloads, 0 silent fallback records; only float-storage gate differs; pinned llama comparison kept separately |
| AC10 | R10 | `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=1 OMEGA_TILED_GEMM_MIN_TOKENS=1 cargo run --release --example ouro_loopcd_lab --features proxima-tensor,omega/metal-tiled-gemm -- --performance --microbench --backend metal --samples 5 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/omega-programmable/results/micro-enabled` | 3 activation-row shapes × 5 samples = 15 raw timing records, 3 warmups, 3 CoV rows, full coordinate payloads; experimental lowering enabled |
| AC10b | R10 | `env OMEGA_EXPERIMENTAL_DENSE_FLOAT_STORAGE=0 OMEGA_TILED_GEMM_MIN_TOKENS=1 cargo run --release --example ouro_loopcd_lab --features proxima-tensor,omega/metal-tiled-gemm -- --performance --microbench --backend metal --samples 5 --omega-config proxima-tensor/specs/omega-programmable/relaxed-metal.toml --output proxima-tensor/specs/omega-programmable/results/micro-disabled` | Same 3 shapes × 5 samples = 15 raw timing records, 3 warmups, 3 CoV rows; experimental lowering disabled; same numeric permissions and tiled threshold |

AC6 contains eight regular tests and no ignored benchmark. AC10 reuses the
native Ouro performance runner's resource sampler, checkpoint loader and timer.
Both microbenchmark arms override the existing tiled minimum from 160 to 1;
the production default remains 160. Correctness
comparators consume the literal oracle itself and its single-coordinate
corruption; CPU/Metal mutual agreement alone is not an oracle.

AC9's eight CoV rows are the eight objects in each arm's `summary.json` array,
with `cost.cov_percent`, `cost.sample_standard_deviation_ms`, and the five raw
durations; they are not extra rows in `samples.jsonl`. Reconcile them using:

`cargo run --example omega_programmable_recount -- proxima-tensor/specs/omega-programmable/results/ouro-configured-control proxima-tensor/specs/omega-programmable/results/ouro-configured /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/ouro-llama-oracle/results/llama-v1/llama.jsonl proxima-tensor/specs/omega-programmable/results/recount`

Expected `timing-recount.jsonl` count: 25 records, comprising 16 paired model
cells, 6 paired projection cells, 2 pinned llama cells and 1 oracle identity
record. The compiled recount requires exactly eight distinct summaries per
model arm and reconciles their raw durations, mean, sample standard deviation,
CoV and percentiles. It also emits all signed coordinate differences for
18 paired projection executions and every non-null vector in all 48 paired
model records, plus the comparator's positive and corruption controls.

## out of scope

- New model-release research, training, and tokenizer changes.
- Per-operation heterogeneous scheduling across multiple physical devices.
- Implementing a new NPU/ANE driver or representing its simulation as hardware
  execution. The common boundary must remain extendable without changing model
  programs; actual absent-driver requests are rejected.
- Silent FP8 conversion: there is no current floating FP8 codec, and integer Q8
  is not a substitute. BF16/F16 native storage decoding is the concrete first
  lowering change. A converted-weight cache must key source bytes, format,
  rounding and layout before any separately requested conversion is enabled.
- Changing Ouro's recurrent semantics, eliminating its statistics readbacks, or
  relaxing numeric permissions without an explicit config in a measured arm.

## context

- `omega/src/config.rs`
- `omega/src/backend.rs`
- `omega/build.rs`
- `omega/omega-runtime.toml`
- `omega/src/msl/emit_and_classify.rs`
- `omega/src/msl/tiled_gemm_cooperative_scan.rs`
- `omega/src/identity.rs`
- `proxima-tensor/src/numeric.rs`
- `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs`
- `proxima-tensor/src/bind/builder_compose_window.rs`
- `proxima-tensor/src/cpu/quantized_eval.rs`
- `examples/ouro_loopcd_lab/engine.rs`
- `examples/ouro_loopcd_lab/performance.rs`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/ouro-llama-oracle/EVIDENCE.md`

## work packets

- [config](WORK-PACKET-CONFIG.md)
- [binding](WORK-PACKET-BINDING.md)
- [execution](WORK-PACKET-EXECUTION.md)
- [tiled lowering](WORK-PACKET-TILED.md)
- [integration and evidence](WORK-PACKET-EVIDENCE.md)
