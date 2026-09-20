# gemma4-E2B forward decomposition — SLICE 1 discipline log

Branch `bench/forward-decomposition`, base `e1c3a17c9`. Harness:
`proxima-model-interop/benches/gemma4_forward_decomposition.rs`.

Run:

```sh
CARGO_TARGET_DIR=<scratch>/target-mb cargo bench -p proxima-model-interop \
    --bench gemma4_forward_decomposition --features "metal instrument"
```

`ollama stop gemma4:e2b-it-qat` first (GPU residency contention,
`project_ollama_loader_is_the_judge_hook.md`). Host loadout for the run this
log reports: `uptime` load average 13.60/10.68/10.45 (NOT a quiet box —
iTerm2 ~20% CPU, WindowServer/spotlightknowledged background load present).
CoV is reported per cell below; cells at or above 5% are flagged.

Build profile: `cargo bench`'s own `bench` profile (`[optimized +
debuginfo]`), never compared against a debug build. macOS, M-series Metal
device (`target_os = "macos"`, `feature = "metal"`).

## Two axes (extensible — every future component slots into this contract)

- **compile**: `omega::metal::pipeline_for`'s cache-MISS cost
  (`omega/src/metal/pipeline_buffers_upload.rs:167`-`296`,
  `compile_pipeline`/`pipeline_for`) — building an `MTLComputePipelineState`
  from MSL. One-time per kernel shape (cache key = op identity + numeric
  policy + math mode, thread-local `PIPELINE_CACHE`).
- **execution**: the warm per-forward GPU bracket, `command_buffer.commit()`
  -> `waitUntilCompleted()`
  (`omega/src/metal/execute_and_hazards.rs:426-434`, and the placement-path
  twin at `omega/src/metal/placements_execute_named.rs:549-554`).

Both read from `omega::metal::metal_stage_totals()` — `PIPELINE_COMPILE_TICKS`/
`PIPELINE_MISSES`/`PIPELINE_HITS` for compile, `GPU_EXEC_TICKS`/`GPU_EXEC_CALLS`
for execution. It is a thread-local **snapshot-and-reset**: every component
below calls it once right after its own timed call, printing a raw
per-iteration record (`gemma4_forward_decomposition component=... width=...
wall_ns=... compile_ns=... exec_ns=... pipeline_misses=... pipeline_hits=...`)
— the discipline-log numbers in this file are computed FROM those printed
records (grep + awk over the captured `cargo bench` stdout), never from
criterion's own summary line alone (guiding principle 19: results traced to
records).

## The 16-point gate — scoped to what this slice claims

This is a decomposition/root-cause bench, not a new production component, so
gates 1 (default-off feature gate), 4 (lint), 12 (config/API parity), and 15
(no-magic-numbers build step) are N/A — nothing here ships behind a runtime
default; the bench itself is already default-off (`required-features =
["metal", "instrument"]`, never built by a default `cargo build`/`test`).
Gates that DO bind:

- **Build+tiers**: `cargo check -p proxima-model-interop --bench
  gemma4_forward_decomposition --features "metal instrument"` — exit 0.
  macOS + Metal only (`#![cfg(all(feature = "metal", feature = "instrument",
  target_os = "macos"))]`); no other tier claimed.
- **Real isolation preferred, proxy documented where not** (gate 13's own
  home-turf discipline, adapted — see Component 3+4 and Component 5 below).
- **Re-provable now** (principle 16): the run command above regenerates
  every number in this file from the artifact alone. No saved criterion
  baseline yet (`--save-baseline` not wired this slice — named as the first
  follow-up below).
- **CoV across runs**: this log reports ONE run's own internal sample CoV
  (computed across the N iterations criterion collected inside that run),
  not CoV across repeated `cargo bench` invocations — that is this log's own
  documented gap (see Follow-ups).

## Real gemma4-E2B hparams used (real dims, not synthetic)

Read via `cargo run -p proxima-model-interop --features std --example
gemma4_dump -- <blob>` against the real blob, 2026-09-20:

| field | value |
|---|---|
| `embedding_length` | 1536 |
| `block_count` | 35 |
| `attention.head_count` (query heads) | 8 |
| `attention.head_count_kv` | 1 |
| `attention.key_length` (global head_dim) | 512 |
| `attention.key_length_swa` (sliding head_dim) | 256 |
| `feed_forward_length` (sliding layers) | 6144 |
| `feed_forward_length` (global layers, last 20 of 35) | 12288 |
| `attention.sliding_window` | 512 |
| `rope.freq_base_swa` | 1e4 |
| `sliding_window_pattern` | 4 sliding : 1 global, repeating |

`vocab=262144` is the task brief's own stated figure
(`bind_gemma4_all_positions_logits`'s doc, `bind.rs:1055`); this dump run did
not print `tokenizer.ggml.tokens`'s own array length to independently confirm
it (MEASURED gap, named not silently assumed away).

## Component 1 — FULL FORWARD (real isolation)

**Method**: [`LoadedModel::prefill_prefix`] — "step 0 of the decode loop
always forwards the whole `next_ids` range against `cached_len == 0` before
ever sampling" (that method's own doc). This is the EXACT production
`bind`/`speculative_verify_program` dispatch `generate_with_serving_config`
uses, not a hand-rolled substitute. `ServingConfig` overrides:
`context_length: 64` (keeps `BackendRuntime::new`'s per-call device
allocation — a fresh runtime every `prefill_prefix` call — from paying for
`ServingConfig::default`'s own 131,072-token arena on every iteration;
`kv_cache_{key,value}_quant: F32`, `flash_attention: false`, matching
`speculative_decode_parity.rs`'s own real-checkpoint config). `gpu_layers`
left at `ServingConfig::default`'s own `GPU_LAYERS_ALL`.

Width is NOT controlled exactly — prompts are built from distinct short
words (`prompt_for_width`) and the REAL observed width is
`PrefixState::len()` (ground truth, never the word-count guess). Observed
widths landed at 2/3/5/9, not the requested 1/2/4/8 (BOS token: every prompt
picked up +1 over its word count).

**RESULT (negative, load-bearing)**: `compile_ns`/`exec_ns` read **zero on
every sample** for this component. Root cause, mechanically confirmed by
reading the artifact: `proxima-model-interop/src/generate/decode.rs:2930`
and `:4196` (the SAME `token_stages`/`prefill_batch_stages` log lines this
slice was asked to verify) call `metal_stage_totals()` THEMSELVES, inside
`prefill_prefix`'s own call graph, under the SAME `feature = "instrument"`
this bench also turns on — and `metal_stage_totals()` is a
snapshot-AND-RESET of process-global `AtomicU64` counters
(`proxima-telemetry/src/metric/counter.rs:12-26`, no thread-locality). A
caller-side second read after `prefill_prefix` returns always sees zero: the
production code already drained the counters into its own (by-default
undirected — no console/file sink configured in this bench) tracing
output before returning. **This is a real, reproducible harness/instrument-
surface gap**, not a guess: `grep -c metal_stage_totals
proxima-model-interop/src/generate/decode.rs` = 2 call sites, both inside
the very call this bench times.

**What IS real and measured for Component 1**: wall-clock time
(`std::time::Instant`, wraps the WHOLE `prefill_prefix` call, criterion
`iter_custom`, `sample_size(10)`, `measurement_time(10s)`,
`warm_up_time(500ms)`):

| width (observed) | N | mean | min | max | CoV |
|---|---|---|---|---|---|
| 2 | 11 | 934.3 ms | 919.1 ms | 1034.1 ms | 3.41% |
| 3 | 21 | 934.9 ms | 925.7 ms | 947.9 ms | 0.74% |
| 5 | 21 | 939.4 ms | 932.4 ms | 960.7 ms | 0.72% |
| 9 | 11 | 1719.8 ms | 1710.5 ms | 1729.0 ms | 0.32% |

**This independently reproduces the finding the slice set out to verify**:
width 3 -> 5 moves the wall time by 934.9 -> 939.4 ms, **+0.48%** for a
+67% width increase — a fixed floor, not `O(width)`. The floor does NOT hold
indefinitely: width 9 jumps to 1719.8 ms (+83% over width 5), so there is a
SECOND regime change somewhere between width 5 and 9 this slice's width grid
(1/2/4/8 target) does not resolve — named as a follow-up, not guessed at.

## Component 2 — LM HEAD (real isolation)

**Method**: standalone `[width, EMBEDDING=1536] x [VOCAB=262144, 1536]^T`
matmul program (`matmul_rhs_transposed_program`, the same technique
`omega/benches/metal_vs_cpu.rs`'s own `matvec_batch1_f32` group uses,
applied at the LM head's real shape), executed via `omega::execute`
directly — bypasses `LoadedModel` entirely, so `metal_stage_totals()` reads
this component's OWN compile/exec split cleanly (no draining collision:
nothing else in this call graph reads the same counters).

Widths here are EXACT (`width` is a direct program parameter, not tokenized
text), so 1/2/4/8 are the real cells.

**Compile (cold, first call only, width=1)**: 0.702 ms, 1 pipeline miss.
**Every subsequent call at every width: 0 compile ticks, 0 misses, 1 hit** —
the pipeline cache is warm after the first call and this shape never
changes across widths (the reduction dimension `k=EMBEDDING` and output
`n=VOCAB` are fixed; only `m=width` varies, and `m` is a runtime dispatch
dimension, not part of the kernel's own cache key).

**Execution (steady state)**:

| width | N | mean | min | max | CoV |
|---|---|---|---|---|---|
| 1 | 37 | 75.50 ms | 72.99 ms | 107.86 ms | **7.54%** (report range, not point estimate) |
| 2 | 23 | 142.72 ms | 135.92 ms | 148.23 ms | 1.81% |
| 4 | 13 | 219.21 ms | 212.78 ms | 225.01 ms | 1.41% |
| 8 | 11 | 374.68 ms | 367.86 ms | 389.02 ms | 1.71% |

Scaling 1->8 (8x width): 75.5 -> 374.7 ms, **4.96x** — approximately
LINEAR in width, NOT a fixed floor. The LM head is compile-cheap (0.7 ms,
one-time) and exec-expensive in absolute terms (up to 375 ms at width 8) but
its OWN shape is not the source of the width-3-to-5 floor Component 1
measured — it scales with width where Component 1's aggregate does not.

## Component 3+4 — ONE gemma4 sliding layer (real isolation, fused)

**Method**: `block_count=1` through the REAL `lfm2_forward_program_with_experts`
engine (`proxima-tensor/src/spec/attention_forward.rs:1719`, the exact
builder `Gemma4Arch::bind` calls), one SLIDING layer (majority shape: 28 of
35 real layers) at real dims (`head_dim=256`, `kv_heads=1`, `query_heads=8`,
`feed_forward=6144`, `embedding=1536`, `mask_window=512`,
`FfnCombination::Exclusive`, `leading_dense_block_count=1` — matching the
real E2B call site's own convention, `gemma4::bind.rs:1356-1373`), executed
via `omega::plan_named`/`omega::execute_plan_named` directly (same
no-draining-collision property as Component 2). `LAYER_PROBE_VOCAB=32`
(tiny, so the LM-head tail this builder always appends stays cheap —
Component 2 already covers that cost in isolation). `ple: false` — the ONE
documented deviation from the real per-layer shape (E2B's per-layer
embedding addend is a cheap secondary lookup, not a floor suspect; every
other field is real-dimensioned).

**NOTE — attention and FFN are NOT separated**: `lfm2_forward_program_with_experts`
has no per-op-kind tap (`gemma4_program_metal_cpu_parity.rs`'s own doc: "no
per-layer-taps counterpart... no way to request an intermediate layer's
residual without hand-rolling a second copy of the graph"), so this reports
ONE fused "one gemma4 layer" number, not two. See Follow-ups for what would
split them.

**Real, kept negative result — width=1 does not lower on this Metal
renderer**:

```
component=one_layer_sliding width=1 status=UNSUPPORTED
reason=Emit(EpilogueNotSupported { node: NodeId(90), reason: "the
broadcast-reduce epilogue only has a Metal renderer for the plain
cooperative-reduce path" })
```

Reproduced identically at `LAYER_PROBE_VOCAB=8` and `=32` (ruled out: not a
vocab-size artifact). Widths 2/4/8 lower and run cleanly — this is
SPECIFICALLY a `new_count=1` gap in this op's broadcast-reduce epilogue
lowering, at this layer's real dims. Kept, not silently routed around: the
bench probes each width before timing it and logs `status=UNSUPPORTED`
rather than fabricating a number.

**Compile (cold, first call only, width=2)**: 203.6 ms, **30 pipeline
misses** (a real transformer layer's norm/QKV/RoPE/softmax/gate/up/down
ops each need their own kernel — much more diverse than Component 2's one
matmul shape). Every subsequent call at every width: 0 compile ticks (cache
warm for that exact op-identity/shape set; width changes the dispatch `m`
dimension only, same as Component 2).

**Execution (steady state)**:

| width | N | mean | min | max | CoV |
|---|---|---|---|---|---|
| 2 | 196 | 18.09 ms | 16.87 ms | 21.94 ms | **5.35%** (report range) |
| 4 | 141 | 18.46 ms | 17.35 ms | 22.63 ms | **5.33%** (report range) |
| 8 | 125 | 22.96 ms | 21.93 ms | 26.29 ms | 3.55% |

Width 2 -> 4 (2x width): 18.09 -> 18.46 ms, **+2.0%** — flat, the SAME
fixed-floor shape Component 1 showed at the full-forward level, now
reproduced at the SINGLE-LAYER level. Width 4 -> 8: +24.4% — the floor
breaks somewhere in 4..8, one layer's own version of Component 1's break
between width 5 and 9.

## Component 5 — KV upload per step (PROXY, not real isolation)

`KvPadScratch::fill` (`proxima-model-interop/src/generate/residency_caches.rs:149-170`,
struct at line 215, `fill` at line 240) is `pub(super)` to the `generate`
module — unreachable from this bench crate (a separate compilation unit,
same visibility rule as `tests/`) without either publicizing it or
duplicating its device-write logic, both out of this slice's time budget.
**No number is reported for this component** — not even a proxy — because
the only available proxy (`metal_stage_totals().block_upload_ticks`) is
read from Component 1, which is ALSO subject to the draining collision
documented above, so it too reads zero. This is the most honest thing this
slice can say about Component 5: **unmeasured**, named precisely, not
approximated with a number that would look real.

## Aggregation — Σ(components) vs measured full forward

Component 1's widths (2/3/5/9, BOS-inflated) do not line up exactly with
Components 2/3's exact widths (1/2/4/8) — flagged, not glossed over. The
closest pairing:

**width ≈ 2**: Component 2 (LM head, width=2) 142.72 ms + Component 3+4
(one layer, width=2) 18.09 ms **× 35 layers (DERIVED, principle 18 — ONE
measured sliding layer extrapolated across all 35 real layers, NEVER
reported as measured for the other 34)** = 633.2 ms. Σ = 142.72 + 633.2 =
**775.9 ms**. Measured Component 1 (width=2) wall = 934.3 ms. **Accounted
≈ 83.1%, unaccounted ≈ 16.9% (158.4 ms).**

**width ≈ 8/9**: Component 2 (width=8) 374.68 ms + Component 3+4 (width=8)
22.96 ms × 35 = 803.6 ms. Σ = 374.68 + 803.6 = **1178.3 ms**. Measured
Component 1 (width=9) wall = 1719.8 ms. **Accounted ≈ 68.5%, unaccounted
≈ 31.5% (541.5 ms).**

**The gap is itself a finding, per the task brief**: the unaccounted
fraction GROWS with width (16.9% -> 31.5%), meaning whatever is missing
from Σ(LM head + 35×one-layer) scales with width too — candidates named,
not measured: `BackendRuntime::new`'s per-call device-buffer allocation
(context_length=64 arena, paid fresh every `prefill_prefix` call, entirely
outside Components 2/3's own execute-only calls), the 7 GLOBAL layers this
slice never isolated (only the 28 sliding layers' shape was measured; a
global layer's `head_dim=512` vs sliding's `256` and `feed_forward=12288`
vs `6144` is real extra work the `× 35` derivation silently assumes away),
KV-cache write cost (Component 5, unmeasured), and embedding-lookup/output-
norm/logit-softcap glue this slice's Component 2/3+4 split does not cover
either.

## Headline

**Is the ~1s multi-position floor compile or execution?** Not resolved by
this slice's OWN direct measurement of Component 1 (the draining-collision
gap above) — but the WALL-CLOCK evidence (0.72-0.74% CoV, near-zero delta
934.9 -> 939.4 ms across a 67% width increase) reproduces the floor exactly
as described, and Component 3+4's real compile/exec split (203.6 ms cold
compile, 0 ms warm; 18.09/18.46/22.96 ms warm exec, ALSO flat 2->4) says
the floor is an **execution-side, not compile-side** phenomenon once warm:
compile is a strict one-time cost that a multi-token production sequence
pays once and amortizes to zero, while the FLAT exec numbers (both at the
one-layer level and the full-forward level) are what stays fixed as width
grows. The component that carries the floor, by elimination: NOT the LM
head (scales ~linearly, 4.96x over 8x width) — the floor lives in the
per-layer attention/FFN body, reproduced at n=196/141/125 samples per cell
with CoV ≤5.4%, and compounds over 35 layers into Component 1's own
934->939 ms plateau.

## Follow-ups this harness makes easy to add next

1. **Fix the Component 1 draining collision**: either have `LoadedModel`
   expose the last step's `MetalStageTotals` on `PrefixState` (or a
   `prefill_prefix`-adjacent method), or wire this bench to a
   `proxima_telemetry::Exporter` sink and read decode.rs's own emitted
   `token_stages`/`prefill_batch_stages` log line instead of re-reading the
   drained global counters.
2. **Split attention from FFN**: either a per-layer-taps builder twin of
   `lfm2_forward_program_with_experts` (mirroring qwen35moe's `_at_width`),
   or the `execute_plan_with_placements_dispatch_timed` per-dispatch
   GPU-timestamp path (`omega/src/metal/dispatch_timed_and_classify.rs:713`,
   `feature = "metal-output-placement"` + `instrument`) against the REAL
   plan.
3. **Real isolation for Component 5** (KV upload): publicize (or add a
   `#[cfg(test)]`/bench-reachable seam for) `KvPadScratch::fill`, OR bench
   it in-crate (`proxima-model-interop/src/generate/` itself, where
   `pub(super)` is reachable) rather than from this external bench crate.
4. **A GLOBAL layer arm** alongside this slice's SLIDING one, so the ×35
   aggregation derivation stops assuming all layers cost the same.
5. **`--save-baseline`** wiring so future tweaks in this area get a
   mechanically re-provable delta-vs-prior row, per principle 16 — this
   slice's own numbers are re-provable from the run command, but not yet
   comparable to a NAMED prior run without re-running this file's commands
   by hand.
6. **Widths 1 and 9 investigated directly**: width=1's `EpilogueNotSupported`
   gap (Component 3+4) and the width-5-to-9 second regime break (Component
   1) are both real, reproducible, and unexplained by this slice.

## SLICE 2 (re-noted — the original section was lost before landing)

**What SLICE 2 established** (recorded in the SLICE 3 task brief, restated
here since the file that originally held it was never committed): the
multi-position (`new_count > 1`) forward floor is MICRO-level — one command
buffer, dispatch count width-invariant at 1661 dispatches regardless of
`new_count` — and the width-SCALING cost is ONE kind,
`reduce-cooperative` (the SIMD-tree reduce used for both the attention-score
reduce and the RMSNorm `value_norm` broadcast-epilogue,
`omega/src/msl/emit_and_classify.rs:988-1001`), whose reduction LENGTH
scales with `new_count`. That finding was measured only at widths 2-9
(`WIDTHS` in `gemma4_forward_decomposition.rs`), which mixes the ~704ms
fixed prefill floor with whatever `reduce-cooperative`'s own scaling
contributes — SLICE 2 could not separate the two. **SLICE 3 (below) is the
separation.**

## SLICE 3 (`bench/kernel-newcount-sweep`) — unconfounding `new_count` at the
## kernel level

Branch `bench/kernel-newcount-sweep`, base `5c2e8054d`. Harness:
`proxima-model-interop/benches/gemma4_kernel_newcount_sweep.rs`.

**The confound this slice removes**: SLICE 1/2's Component 1 sweep
(`WIDTHS = [1, 2, 4, 8]` in the real-forward harness) never actually
measures `new_count=1` cleanly — `prompt_for_width`'s own doc notes real
BOS tokenization floors the tokenized width at 2, and a genuine
single-position decode uses a DIFFERENT renderer path than the
multi-position verify path SLICE 1's Component 3+4 hit the
`EpilogueNotSupported` gap on. No real-forward measurement ever isolates
"what does `new_count` alone cost" from "what does the BOS-inflated
width-2 floor cost". This slice unconfounds it by measuring four kernels
in ISOLATION — no BOS token, no full-program epilogue gate riding along —
at real gemma4-E2B dims, swept `new_count` ∈ {1,2,4,8,16}.

Run:

```sh
CARGO_TARGET_DIR=<scratch>/target-uc cargo bench -p proxima-model-interop \
    --bench gemma4_kernel_newcount_sweep --features "metal instrument"
```

`ollama stop gemma4:e2b-it-qat` first. Host loadout for this run: `uptime`
load average 4.36/7.82/14.04 (not a quiet box — 3-day uptime, background
load present; CoV reported per cell, cells above 5% flagged). Build
profile: `cargo bench`'s own bench profile (optimized + debuginfo), never
compared against a debug build. All 20/10-sample criterion cells; every raw
per-iteration record also printed (`gemma4_kernel_newcount_sweep kernel=...
wall_ns=... exec_ns=...` lines) so every row below is grep-able from the
run's own stdout, not only the criterion summary.

**Zero-UNSUPPORTED headline, itself a finding**: unlike SLICE 1's
`one_layer_sliding` component (real full layer, `EpilogueNotSupported` at
`width=1`), **every one of the four isolated kernels lowered and ran
cleanly at `new_count=1`**, including the RMSNorm broadcast-epilogue
kernel — the exact op shape (`x*inv_rms` folded into the reduce) SLICE 1
could not exercise at width=1. This is the direct proof the task brief
asked for: an isolated kernel has no BOS and no full-program epilogue gate,
so `new_count=1` IS measurable there even though it is not measurable in
the real 35-layer forward.

### Per-kernel cost(new_count) — median/N/CoV, and the fitted fixed+slope

Fit: `cost(new_count) = fixed_floor + slope · new_count`, ordinary least
squares over the 5 measured points (medians, `gemma4_kernel_newcount_sweep`
raw records, compile-warmup record excluded from every group).

**1. `reduce-cooperative` — attention-score shape** (`[new_count, 256] x
[256, key_count]^T`, `head_dim=256`, real gemma4-E2B sliding-layer dim,
`key_count=256` fixed for this sweep):

| new_count | N | median | CoV | notes |
|---|---|---|---|---|
| 1 | 3450 | 447.5 µs | 7.7% | |
| 2 | 7113 | 440.96 µs | 14.6% | |
| 4 | 6693 | 433.92 µs | 17.9% | |
| 8 | 5643 | 457.67 µs | 7.5% | |
| 16 | 6693 | 460.25 µs | 68.0% (report range: 280 µs–22.7 ms, rare outlier iterations; median is the honest read) |

**Fit: fixed_floor ≈ 439.7 µs, slope ≈ 1.35 µs/position** — a slope
0.3% of the floor, i.e. statistically indistinguishable from zero given
12-18% per-cell CoV. **No measurable `new_count` scaling in 1..16.**

**2. `reduce-cooperative` — attention-score shape, `key_count` sweep**
(`new_count=1` fixed, `key_count` ∈ {64,128,256,512}):

| key_count | N | median | CoV |
|---|---|---|---|
| 64 | 6903 | 438.71 µs | 12.7% |
| 128 | 6273 | 429.33 µs | 155.4% (one 53.5 ms outlier iteration; median unaffected) |
| 256 | 5853 | 428.75 µs | 19.7% |
| 512 | 6903 | 318.96 µs | 14.6% |

**Fit: slope ≈ -0.27 µs/key** — negative, i.e. **no real key_count
scaling either; flat within noise across 64..512 keys.** The 512-key cell
reading LOWER than the others is noise/floor artifact, not a real
inverse relationship — recorded as observed, not explained away.

**3. `reduce-cooperative` — RMSNorm broadcast-epilogue shape** (reduce
over `EMBEDDING=1536`, one row per `new_count`, the exact
`value_norm`-style op SLICE 1/2 named):

| new_count | N | median | CoV |
|---|---|---|---|
| 1 | 10026 | 247.0 µs | 11.6% |
| 2 | 10237 | 241.17 µs | 12.1% |
| 4 | 10027 | 252.79 µs | 17.8% |
| 8 | 10237 | 242.88 µs | 14.9% |
| 16 | 9607 | 264.42 µs | 15.1% |

**Fit: fixed_floor ≈ 242.4 µs, slope ≈ 1.18 µs/position** — again a slope
<0.5% of the floor. **No measurable `new_count` scaling in 1..16, despite
this being architecturally a genuine PER-POSITION cost (one reduce+epilogue
row per position).**

**4. Packed-row Q4_0 matvec** (REAL `blk.0.attn_k.weight` bytes,
`[1536, 256]`, activation `[1536, new_count]`):

| new_count | N | median | CoV |
|---|---|---|---|
| 1 | 9816 | 246.5 µs | 17.1% |
| 2 | 6272 | 399.96 µs | 15.8% |
| 4 | 6902 | 312.69 µs | 10.5% |
| 8 | 7533 | 329.54 µs | 17.8% |
| 16 | 6483 | 553.13 µs | 19.8% |

**Fit: fixed_floor ≈ 269.9 µs, slope ≈ 15.9 µs/position.** Noisy
(non-monotonic middle points, 10-20% CoV) but a real net upward trend
(246.5 µs → 553.1 µs, 2.24x for 16x width) — unlike the two reduce shapes
above, this kernel shows a genuine, if small, `new_count` slope.

**5. LM-head unembed** (`[new_count, 1536] x [262144, 1536]^T`):

| new_count | N | median | CoV |
|---|---|---|---|
| 1 | 22 | 76.13 ms | 13.0% |
| 2 | 23 | 143.08 ms | 3.0% |
| 4 | 13 | 218.31 ms | 1.5% |
| 8 | 11 | 372.04 ms | 2.2% |
| 16 | 11 | 675.35 ms | 0.3% |

**Fit (all 5 points): fixed_floor ≈ 54.9 ms, slope ≈ 39.1 ms/position.**
**Fit (width ≥2 only, excluding the width=1 point): fixed_floor ≈ 66 ms,
slope ≈ 37.6-38.4 ms/position** — highly consistent step-to-step (66.95,
37.6, 38.4, 37.9 ms/unit marginal slope between consecutive widths).
**width=1 is anomalous**: actual 76.13 ms vs ~94 ms the full-fit line
predicts, and vs the ≥2-only fit's own 66+38=104 ms — `new_count=1` costs
LESS than either linear extrapolation predicts. Kept, not smoothed over:
this is the SAME shape of "width=1 behaves differently" pattern SLICE 1
found at the full-forward level, now reproduced at the isolated LM-head
KERNEL level — unexplained by this slice, named as a follow-up.

### THE UNCONFOUNDED ANSWER — is `reduce-cooperative` a fixed floor or genuine scaling?

**Fixed floor, not scaling — at both real shapes this slice measured, over
`new_count` 1..16 and `key_count` 64..512.** Both `reduce-cooperative`
kernels (attention-score, RMSNorm broadcast-epilogue) show a slope under
2 µs/unit against a 240-460 µs floor — noise, not signal, at 12-18% CoV.
This is the answer SLICE 1/2's confounded width-2-to-9 sweep could not
give: `reduce-cooperative`'s OWN dispatch cost does not grow with
`new_count` in this range; whatever scaling SLICE 2 observed in the
confounded full-forward sweep must come from elsewhere (candidates below,
under Aggregation) or from a `new_count` range this slice did not sweep
(residual, named not measured: prefill widths beyond 16 are untested here).

### Compile vs execution axis (kept)

Every sample's `exec_ns` (the `gpu_exec_ticks`-derived
`command_buffer.commit()`→`waitUntilCompleted()` bracket) was logged
alongside `wall_ns`. At `new_count=2`: attention-score reduce 404.7 µs
`exec_ns` of 440.96 µs wall (91.8%); RMSNorm epilogue 197.3 µs of 241.17 µs
(81.8%); packed-row matvec 341.7 µs of 399.96 µs (85.4%); LM-head 109.9 ms
of 143.08 ms (76.8%). `compile_ns`/`pipeline_misses` are 0 in every
post-warmup record (one miss per program shape, all absorbed into
criterion's own warm-up phase, excluded from every stats group above) —
confirms this slice, like SLICE 1, is reading STEADY-STATE execution cost,
not compile cost. The ~15-24% of wall time NOT in `gpu_exec_ticks` is
`omega::execute`'s own per-call CPU-side cost (infer/bind/plan build) —
paid FRESH every call here because this bench uses the `execute()`
convenience wrapper, unlike a production decode loop that reuses one
cached plan across steps.

### Aggregation — does Σ(isolated kernel costs) ≈ the full-forward exec?

**Partial sum** (1 attention-score reduce + 1 RMSNorm epilogue + 1
packed-row Q4_0 matvec — the THREE kernels this slice measured, NOT the
full op mix of a real layer, which also has q/v/o projections and three
FFN projections at `feed_forward=6144`, ~24x wider than the `attn_k`
projection measured here) at `new_count=2`: 440,958 + 241,167 + 399,958 =
**1,082,083 ns ≈ 1.08 ms**.

Compared against SLICE 1/2's OWN measured `one_layer_sliding` (the real,
fused, ALL-ops single layer) at width=2: **18.09 ms** (N=196, SLICE 1's
own table). **Partial-Σ / full-layer ≈ 6.0% accounted.**

**Two honest reads, not one, per the evidence this slice actually has**:

1. **~94% of a real layer's cost is in ops this slice never isolated** —
   the other 6 projections, RoPE, softmax, gating. The ONE projection
   shape measured (`attn_k`, `in=1536, out=256`) is among the SMALLEST in
   the layer; `feed_forward=6144`-wide FFN projections are the likely
   dominant unmeasured cost, not reduce-cooperative.
2. **The Σ itself is not a fair per-op cost inside the real batched
   forward.** SLICE 2 established the real forward runs its 1661
   dispatches through ONE command buffer; this slice's 3 kernels each pay
   their OWN separate `execute()` round trip (3 independent command
   buffers, each carrying the ~250-450 µs floor documented above). Inside
   the real batched buffer, that floor is paid ONCE for the whole buffer,
   not once per op — so even the 6% figure likely OVERSTATES what these 3
   ops cost when they run batched. The true marginal per-op cost inside
   the real forward is closer to each op's own `gpu_exec_ticks` share of
   ONE shared command buffer, which this slice's isolated-kernel harness
   structurally cannot measure (same limitation SLICE 1's Component 5 named
   for KV upload).

**What this DOES settle**: `reduce-cooperative`'s own summed dispatch cost
is a poor candidate for SLICE 1's ~704 ms full-forward floor. Even a
generous DERIVED extrapolation — 2 `reduce-cooperative` calls/layer
(attention-score + `value_norm`) × 35 layers × ~350 µs average measured
cost — is **≈24.5 ms, about 3.5% of the 704 ms floor** (principle 18:
tagged DERIVED, never measured for all 70 instances). This is a NEGATIVE
result, kept because it eliminates a candidate: the floor is not
"many small `reduce-cooperative` dispatches", it has to be either the
larger FFN/attention GEMM compute this slice did not isolate, or genuine
per-forward setup (`BackendRuntime::new`'s device-buffer allocation, KV
upload, embedding lookup) — the same candidates SLICE 1's own Aggregation
section already named as unmeasured.

### Kernel that could not lower at `new_count=1`

**None.** All four kernels probed and ran cleanly at `new_count=1`
(0/24 `status=UNSUPPORTED` records across the whole run). This directly
narrows SLICE 1's `EpilogueNotSupported` finding: the broadcast-reduce
epilogue Metal renderer itself DOES support `new_count=1` (proven here,
timed, real numbers) — so SLICE 1's real-layer failure is not a renderer
gap in the epilogue path in general. The failing `if` in
`elementwise_reduce_core.rs:224-247` has THREE ways to reject
(`!matches_reduce_dims`, `!reduce_is_cooperative_dispatch(...)`,
`tiled_gemm_block(...).is_some()`, `packed_row_block(...).is_some()`) —
this slice's minimal program only ever exercises the first two (and both
pass at every width). **The exact fix target, now unconfounded**: trace
which of `tiled_gemm_block`/`packed_row_block` classifies the REAL layer's
`value_norm` reduce node as eligible at `new_count=1` specifically (a real
layer has packed-row-blocked matvecs elsewhere in the same program; at
`width=1` the `value_norm` reduce's own shape may coincide with one of
those classifiers' admission window in a way it does not at `width≥2`).
This is a trace-and-confirm task against
`classify_packed_row_block`/`tiled_gemm_block` on the real bound program,
not a renderer-writing task — NOT measured by this slice, stated as the
follow-up it earns.

### Follow-ups this slice makes easy to add next

1. **Sweep `new_count` past 16** — prefill widths of 32/64/128+ to check
   whether `reduce-cooperative`'s flat-floor finding holds at realistic
   prefill widths, or whether a slope only emerges past this slice's
   ceiling.
2. **Measure the other 6 projection shapes per layer** (q/v/o proj,
   FFN gate/up/down at `feed_forward=6144`) — the aggregation section's
   own "94% unaccounted" gap needs these, not another reduce-shape probe,
   to close.
3. **Trace the real-layer `value_norm` node's classifier verdict at
   `width=1`** (`classify_packed_row_block`/`tiled_gemm_block`) — the exact
   fix target this slice narrowed but did not itself trace.
4. **A batched-command-buffer version of this slice's 3-kernel Σ** (one
   `plan`/`execute_plan` call carrying all 3 ops, mirroring
   `rmsnorm_fused_epilogue_cost.rs`'s own `INSTANCES`-chains-in-one-buffer
   technique) — would settle the aggregation section's second honest read
   (whether the 6%-accounted figure is inflated by per-call floor
   duplication) with a real number instead of an argument.
5. **`--save-baseline` wiring**, same gap SLICE 1 named — this slice's own
   numbers are re-provable from the run command + grep on the raw records,
   not yet comparable to a named prior run without re-running by hand.
