# GPU parity through omega — ONE RISC
## An independent, phased implementation plan (cards for Luna)

Baseline: **main `4be2f3a`**, sealed cell **R13** (2026-09-03). Every number below cites its ledger section. Numbers I compute from ledger numbers are tagged **DERIVED** and are never used as a mechanism claim (§18).

---

# I. Diagnosis

## I.1 What the sealed cell says (R13)

| cell | value | tag |
|---|---|---|
| llama.cpp-Metal `llama-bench -n 32 -p 0 -r 5 -t 8 -ngl 99` | 57.08 t/s = **17.52 ms/tok**, CoV 0.89% | MEASURED R13 |
| ours `step_wall_ms` | **67.92**, CoV 0.5% → **3.88x** | MEASURED R13 |
| ours `gpu_exec_ms` | **56.93**, CoV 0.7% → **3.25x** kernel-only | MEASURED R13 |
| op_count | **1196** | MEASURED R13 |
| incumbent dispatches | **~740** (23 real ops/layer × 32 + 4) | READ R8 |

Per-phase (ms/token, steps 1..7): prepare 1.97 | emit 0.81 | block_upload 2.00 | op_setup 3.90 | pipeline_lookup 0.04 | encode_dispatch 0.47 | readback 0.22 | Σ ≈ 9.4; `wall − gpu_exec = 11.0`.

Per-op profile (R13): reduce-packed-row-blocked **225 ops / 44.450 ms**; reduce-cooperative **385 / 9.113**; elementwise **547 / 7.350**; constant+iota 39 / 0.169.

## I.2 The mass, re-derived on today's main

Weight traffic per token from R13's own per-family true bytes: 32×(33.05+33.05+34.00+9.45+9.45+2.40+2.40) MB + 107.5 MB = **4.069 GB/token** (DERIVED) — within 1.7% of R1's 3.9996 GB/token, so the two independent paths agree. At the incumbent's achieved 228.9 GB/s (R1) that is **17.78 ms** (DERIVED). Measured 44.450 → the Q4_K matvec runs at **2.50x** the incumbent's achieved streaming time (DERIVED), i.e. **~26.7 ms of pure ALU excess** (R13 says 26.9 by a different arithmetic; the two agree within 1%).

| bucket | ms | class | ledger |
|---|---|---|---|
| irreducible weight stream | 17.5 | physics | R13 |
| Q4_K matvec above that rate | **26.9** | real work badly done | R13 |
| non-matmul GPU (9.113 coop + 7.350 ew + 0.169) | **16.6** | mostly should not exist | R13 |
| CPU orchestration (`wall − gpu_exec`) | **11.0** | should not exist; shape is identical token to token | R13 |

## I.3 The four structural facts that make the above true, each read on main this session

**D1 — the plan is not one plan (R16, re-verified).** `omega/src/metal.rs:1003` binds, then `:1013` calls `correct_packed_matmul_layouts(&mut resolved, …)` (`proxima-tensor/src/bind.rs:1648`). `proxima-tensor/src/cpu.rs:358` calls `bind::bind` and never applies it. **The bound plan Metal executes is not the bound plan CPU executes, today.** Brief one-RISC item 1 is FALSE on main. Any "one RISC" claim that does not capture the plan *after* each driver's own rewrite is unfalsifiable.

**D2 — the route is not a value (R5/M10, R12 ROW 263).** `classify_kind` (`metal.rs:785-826`, called `:709`) buckets by substring of emitted MSL; its own doc (`:777-783`) admits the routing decision "is not exposed as its own accessor". The parallel branch already ate this defect once (R12 ROW 263: 9/601 → 225/385 after a marker string was added). `Q4K_UNPACK_MSL`/`Q5K`/`Q6K` are concatenated with no delimiter at `msl.rs:1978-1982`, so a "grep the Q4_K region" tie-break is undecidable on emitted source (R16).

**D3 — the plan cache cannot hit, by construction (R11 M6', confirmed by R13).** Key is `(symbols[0], symbols[1])` = `(new_count, cached_len)` (`generate.rs:966`); `cached_len` is `Extent::Symbolic(1)` on every KV leaf (verified this session: `spec.rs:6216-6245`, shapes `[Symbolic(1), kv_heads, pairs|head_dim]`). `plan_hits=0 plan_misses=8` every run (R13). The harness *asserts* this at `proxima-model-interop/src/bind.rs:3052-3055`. So `prepare` 1.97 + `op_setup` 3.90 = **5.87 ms/token** is paid to rebuild a program whose shape never changes.

**D4 — the KV cache is host-resident and re-wired every token (R11 M2').** `LayerCache {k_even, k_odd, v: Vec<f32>}` (`generate.rs:621-625`), `append` = 3× `extend_from_slice` (`:636-640`) on `Vec::new()` — reallocating, so the base pointer moves. `NOCOPY_BUFFERS` is keyed `(pointer, byte_length)` (`metal.rs:1848`); `create_no_copy_buffer` (`:1914`) requires a page-aligned pointer **and** a page-aligned length, which a growing slice cannot satisfy; non-resident blocks route to `upload_block_no_copy_uncached` (`:1903`) which creates a fresh `MTLBuffer` **every token**. `kv_cache_upload_bytes` grows **+262,144 B/token** (R13). `AlignedBuffer` (`proxima-tensor/src/align.rs:69`) exists with **zero production callers**.

## I.4 The diagnosis this plan commits to

The GPU lane loses 3.88x for four reasons in this order of removable mass, and **none of them is a Metal API problem** — the incumbent uses the same API, one command buffer, `n_cb=1`, and **no fusion at all** (`grep -rln fuse ggml/src` is empty at b25346221, R8):

1. **26.9 ms** — the Q4_K matvec body does ~6x the ALU work per 8 weights (R3 M3). Two independent fixes are already MEASURED (mask-fma −17.2% `gpu_exec`, R7; pair-dot −29% on the Q4_K family, R12 ROW 257) and **neither is in git on a branch off today's main**.
2. **16.6 ms** — non-matmul ops run 32 lanes wide (`SIMD_WIDTH`, `sized.rs:45`) where the incumbent's `rms_norm` uses up to 1024 with float4 loads (R8, `ggml-metal.m:3797-3804`).
3. **11.0 ms** — orchestration that exists only because D3 and D4 hold.
4. **1196 vs ~740 dispatches** — the attention duplication (R11 M1'). R12's own control proves this is the **smallest** lever: 616 vs 1194 dispatches moved wall from 51.571 to 51.535 ms, i.e. **nothing**. It is scheduled last, not first, and the plan says so out loud.

D1 and D2 are not on that list because they remove no milliseconds. They are scheduled **first** because every claim in 1–4 is unprovable while they stand.

---

# II. One-RISC binding — the eight design pressures resolved

## II.1 Item 1 — GPU scatter and structural injectivity

**Where it stands.** `IndexMap::scatter` (`map.rs:175`) and `scatter_extent` (`:209`) exist; `cpu.rs:6911` `run_reduce_scatter` implements it; `msl.rs:933`, `wgsl.rs:364`, `cuda.rs:241` all raise `EmitError::ScatterNotSupported` (`error.rs:53`). `map.rs:110-131` states the reason in its own words: *"this crate runs the CPU interpreter's reduce loop strictly sequentially, so a scatter never needs atomics"*. A GPU has no such loop.

**The rule this plan adopts — injectivity is proved by SHAPE, never by a leaf's name.** At bind time, for a `Reduce` whose `out_map` is `IndexMap::Computed{indices, …}`, walk `indices` backwards through the program:

- ACCEPT iff the chain reduces to `index(i) = coeff·i + base` where
  - the terminal node is `Op::Iota` over an iteration axis, and
  - every intervening node is `Op::Elementwise` with body in `{Identity, Add, Multiply}` (all present in the 17 `ScalarOp` bodies, verified `op.rs:60-78`), and
  - the non-Iota operand of each is either rank-0 `Op::Constant` or a rank-0 `Op::Input` broadcast over the whole iteration space, and
  - `coeff != 0`.
- On ACCEPT, bind **emits no scatter at all**: `coeff` folds into `out_strides`, `base` folds into `out_layout.base`, and `out_scatter` is set to `None`. This is exactly the fold `layout_of` already performs on the read side (`bind.rs:1594-1606`: `base += i64::from(axis.offset) * stride`). The write then lowers as an ordinary strided store that every backend already covers.
- On REJECT, `ScatterNotSupported` stays raised — but the decline becomes a first-class census value `Route::Declined(ScatterNotProvenInjective, node)` (§II.4), never a silent CPU fallback and never an atomics-based GPU scatter.

**Why this is the right shape and not a dodge.** The only scatter this model actually needs is the KV append: `dest[cached_len + i] = src[i]`. That is affine with a loop-invariant base. It is *not* data-dependent addressing; it only looks like one because `Reduce.out_map` had no other way to say "write at an offset". `Layout {base: i64, strides}` already exists on every bound reduce (`bind.rs:95-98`) and `out_base` is already a **per-dispatch uniform field**, not baked MSL text (verified this session: `msl.rs:2216` declares `long out_base;` inside `struct Uniforms`, consumed at `:2361` `long out_offset = u.out_base;`, uploaded by `upload_uniforms` `metal.rs:2070`). So the machinery is there.

**The one thing that must change, and only if measured.** `Layout.base` is a baked `i64`. A base that varies per token (the KV write offset) would defeat plan reuse. The minimal extension is a runtime source for that single field. **This plan does not mint it up front.** It is gated behind P8.2's op-count cell and carries a named un-park condition (§VII). Everything earlier is achieved without it.

## II.2 Item 2 — where `cached_len` goes when the leaf extent is bucketed

Today `cached_len` is one value doing two jobs: the `Symbolic(1)` extent on every KV leaf (`spec.rs:6216-6245`) and the RoPE position base (`generate.rs:1304-1309`). Bucketing splits it into **three** carriers:

1. **RoPE position base — unchanged.** `build_position_inputs(&next_ids, cached_len, head_dim, rope_freq_base)` keeps the true `cached_len`. Its output shapes depend on `next_ids.len()`, never on `cached_len`. No plan impact.
2. **The KV leaf extent — becomes `ceil(cached_len / bucket) * bucket`.** This, and only this, is what the plan key sees.
3. **A NEW rank-0 `Op::Input` leaf `kv_valid_len`** carrying the true `cached_len` as f32. Exactness to 2^24 is the same bound `map.rs:105` already documents for gather indices — the same ceiling, not a new one.

**The tail mask, spelled with ops that exist.** `ScalarOp` has exactly **17** bodies (read in full this session, `op.rs:60-78`): `Identity, Add, Subtract, Multiply, Divide, Maximum, Minimum, Negate, Reciprocal, Exponential, Logarithm, SquareRoot, Tanh, Erf, Greater, Equal, Select`. **There is no `Less`, no `LessEqual`, no `GreaterEqual`.** The mask must therefore be spelled with `Greater` in this argument order:

```
kv_valid_len : Input, rank-0, f32          # the TRUE cached_len
cache_slot   : Iota{ extent: Symbolic(1) } # over the BUCKETED cache axis
is_valid     : Greater(kv_valid_len -> "->t", cache_slot -> "t->t")   # 1.0 where t < cached_len
score_masked : Select(is_valid, score_cached_scaled, neg_infinity)
```

This mirrors `causal_mask` node for node: `spec.rs:823-845` builds two `Iota{Symbolic(0)}` + `Greater` and a `scalar_constant(f32::NEG_INFINITY)`, consumed at `spec.rs:2604-2615` as `Select(is_future, neg_infinity, score_new_scaled)` — read verbatim this session. The tail mask is the same construction with the arguments swapped (valid, not future).

**Node cost, pre-registered.** `cache_slot` and `is_valid` depend only on the bucketed axis and the shared scalar, so they are built **once for the whole program**, not per layer. The per-layer cost is one `Select`, which bind is expected to fuse into `score_cached_scaled`'s existing `ComposedBody` (`BoundOpKind::Elementwise{body: ComposedBody, …}`, `bind.rs:221-264`). **Prediction: `op_count` rises by exactly 2, from 1196 to 1198. If it rises by 34, the fusion assumption is refuted and that is the finding, not a failure.**

## II.3 Item 3 — ordering device-residency against bucketing

**Device-residency comes FIRST. Bucketing second.** Three reasons, each mechanical:

1. **Residency changes no graph.** The block handed to omega stays `&cache[..cached_len*row]`, so `element_count(shapes.of(node))` still equals `block_element_count(block)` and the strict check at `metal.rs:991-1000` / `cpu.rs:346-356` is **preserved, not relaxed**. A card that relaxes a correctness check to make a perf number move is the punt §15 forbids; this plan does not contain one.
2. **The mechanism already exists and needs one caller.** `register_checkpoint_mapping` (`metal.rs:1744-1773`) registers a page-aligned, process-lifetime host span and `checkpoint_mapping_offset` (`:1786-1815`) hands back `(whole-span buffer, byte offset)`. Its own doc says: *"the scratch and KV-cache buffers … never live inside the checkpoint's own mmap, so they fall through unchanged."* That is an invitation. Generalizing the single `CHECKPOINT_MAPPING: Option<(usize,usize)>` slot to N slots makes the KV arena addressable by offset by the **same primitive** (§1 reuse: extend, do not add a peer). `AlignedBuffer` (`align.rs:69`, zero production callers) supplies the page-aligned base, gaining its first production caller.
3. **Doing it the other way confounds two effects.** Bucketing inflates the cached-range reduces (§II.7) while residency deflates `block_upload`. Measured together, neither is attributable. Measured residency-first, each has its own cell.

The board never carries a pre-registered regression in `default` because every step is behind a default-off compile-time feature (gate point 1) and `omega`'s `default = ["std","metal","cpu"]` is not touched by any card.

## II.4 Item 4 — the route census, lock-free and free per dispatch

**Not copied:** `WIDTH_TILE_DECLINE: Mutex<BTreeMap<(u32, WidthDeclineReason), WidthDeclineTotals>>` (`instrument.rs:842`, `.lock()` at `:857`, read verbatim this session). Its neighbour is already the right shape: `ENCODE_DISPATCH_CALLS` is an atomic `proxima_telemetry::Counter` (`metal.rs:2243`, defined `:1484`).

**The design.** The route is a property of the **plan**, not of the dispatch.

```rust
// proxima-tensor or omega, one place, one decision function
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Route {
    Elementwise, Iota, Constant, Scan,
    ReduceSerial, ReduceCooperativeGeneric,
    ReduceCooperativePackedRow, ReduceCooperativeTiledGemm,
    Declined,                       // carries a reason alongside
}
```

Exactly **8 live variants** — one per the 8 Metal kernel-body shapes R5 counts, no invention. `fn route_of(&BoundOp, &PackedOperands) -> (Route, DeclineReason)` becomes the function `emit` itself branches on, so census and emission **cannot disagree** (this is the defect R12 ROW 263 hit and patched with a second marker string).

**Storage — three properties, zero locks.**
- `Plan.routes: Vec<Route>` — one `u8` per `BoundOp`, filled **once** at plan build, single-owner, never shared. No synchronisation of any kind.
- A fixed-size atomic histogram `static ROUTE_TOTALS: [Counter; ROUTE_COUNT]`, incremented **once per plan build**, not per dispatch.
- The `(NodeId, reason)` detail the census needs is `plan.routes` zipped with `plan.prepared.resolved` at snapshot time — a read of data the plan already owns.

**Cost bound.** Per-dispatch recording cost is **zero instructions** because nothing is recorded during encode. Pre-registered against R13's `encode_dispatch` 0.47 ms / 1196 dispatches = 393 ns/dispatch (DERIVED): **predicted Δ = 0; kill if `encode_dispatch` mean exceeds 0.4935 ms** (0.47 × 1.05, R13 CoV band).

`classify_kind` (`metal.rs:785-826`) and `diagnose_kind` (`:835-854`) are **deleted**, not kept alongside. Their call sites (`:709-710`) read `plan.routes[position]`.

## II.5 Item 5 — closing the false one-RISC claim early

The order is: **test RED, then fix, in Phase 2 — before any route work, any kernel work, and any board prediction.**

- **P2.1** adds a test that captures `&[BoundOp]` from **each backend's plan after that backend's own rewrites** and asserts equality. It is **pre-registered RED on main** because of `metal.rs:1013`.
- **P2.2** moves ownership into `bind`: `packed_operands_of` (currently `metal.rs:375`) descends into proxima-tensor next to `QuantizedBlock` (`cpu.rs:3084-3110`, where the type already lives); `bind` gains the packed set; `correct_packed_matmul_layouts` becomes an internal step of `bind`; the call at `metal.rs:1013` is **deleted**. P2.1 turns GREEN.
- CPU passes its own packed set too. Whether that changes any CPU layout is **tested, never assumed** — the CPU quantized path is claimed not to read packed bytes through `layout_of` (R16), and P2.3 proves it with the existing greedy oracle (`bind.rs:2797-2803`: token id `2651`, text `"known"`, llama.cpp's own captured answer, §14).

## II.6 Item 6 — the two Q4_K bodies, recovered, made exclusive, baked off, one landed

**Where they are (R7, R12).** Body A = `metal-q4k-mask-fma`, an **uncommitted diff** in `proxima-wt-all` on branch `perf/gpu-all-wins`, based on `2b95210`, which main has moved **9 commits past** including `spec.rs +8735/-2836`. Body B = `q4k_pair_dot`, a **commit** on `perf/q4k-independent-accumulators`, off today's main `4be2f3a`. They are independent re-derivations of the same ggml mechanism (mask-without-shift + fold 1/16, 1/256 into the scale at combine, `ggml-metal.metal:5147-5175`, R8).

**Mutual exclusion, and why it cannot be two cargo features.** `scripts/omega-gate.sh` step **[2/6]** runs `cargo build -p omega --all-targets --all-features` (read verbatim this session). Two mutually-exclusive features guarded by `compile_error!` would make the crate's own gate red. So the two bodies are **not two features**. They are **one build-time selector**, which is exactly guiding-principles §8's *profile input* half:

```toml
# omega/omega-runtime.toml
[q4k]
# which Q4_K matvec body the packed row-blocked route emits.
# "mask_fma"  -- mask without shift, 1/16 and 1/256 folded into the scale at combine
# "pair_dot"  -- paired-nibble dot, independent accumulators
body = "pair_dot"
```

`omega/build.rs` gains `emit_profile_cfg()` alongside `emit_sizing_consts()` (`build.rs:105`), emitting `cargo:rustc-check-cfg=cfg(omega_q4k_body, values("mask_fma","pair_dot"))` + `cargo:rustc-cfg=omega_q4k_body="…"` + `cargo:rerun-if-env-changed=OMEGA_Q4K_BODY` (the same env-override + rerun contract `resolve_int` `build.rs:79-85` already implements). `--all-features` compiles exactly one body, always. The bake-off is a rebuild with `OMEGA_Q4K_BODY=…`.

**The terminal tie-break, decidable, in this order** — and it never greps emitted source (R16 proved that undecidable at `msl.rs:1978-1982`):

1. `gpu_exec_ms` mean over interleaved A B A B A B, 3 runs per arm. Winner iff `|mean_A − mean_B| > max(CoV_A, CoV_B) × max(mean_A, mean_B)`.
2. Else: max absolute parity error vs an f32 reference on the real `blk.0.attn_q.weight` (§9 real data; B has a recorded 3.1e-6, R12 ROW 257). Lower wins iff the difference is nonzero.
3. Else: total emitted MSL byte length for the full decode program. Smaller wins. Deterministic by `emit_is_deterministic_byte_equal` (`msl.rs:4656`, R15).
4. Else: the body that is already a git commit (**pair_dot**) — the incumbent-holds rule.

The loser is a **recorded negative row**, not a deletion (disciplined-component: *especially rollbacks*).

## II.7 Item 7 — memory is a kill on every card

Owner rule 2026-09-03 (R13): *"if you've fucked up memory again, you lose"*. Prior failures: the **34 GB** KV allocation from the `context_length` default, and the plan-cache heap growth (bounded by `ff749a0`).

**The trap, closed at build time.** `ServingConfig::context_length` defaults to **131_072** (verified this session, `proxima-model-interop/src/serving.rs:161`). 131_072 × 262_144 B/token (R13's measured KV slope) = **34,359,738,368 B = 34.36 GB** (DERIVED — the trap reproduced exactly from source). A KV arena sized from `context_length` is therefore a 34 GB allocation. P6.1 makes the arena size a **build-time** key with a build-time assertion:

```
kv_capacity_tokens × 262,144 + 4,140,417,024 + 41,943,040  ≤  device_cap_bytes
```

violation = `panic!` in `build.rs`, i.e. a **compile error**, following `require_nonzero` (`build.rs:16`) / `require_divides_q4k_block` (`:43`). The default `kv_capacity_tokens = 512` yields **4,316,577,792 B ≈ 4.317 GB** (DERIVED), against R13's measured steady `device_allocated_bytes` of 4.152–4.163 GB and prefill 4.299–4.305 GB.

**Three named memory gates**, referenced by every card:

- **MG-1 (build/lint only).** No process runs the model. Gate: build exit 0; no new heap-holding `static`/`thread_local` introduced without a bound stated at the site.
- **MG-2 (probe or bench, no checkpoint).** Peak task RSS ≤ **400 MB** (R13 prefill peak 310–357 MB + headroom). `omega::metal::current_allocated_size()` returns to its pre-probe value ± 2 MB after the last run.
- **MG-3 (decode harness).** All four, any one failing is a KILL:
  1. **RSS slope** — steady-state task RSS over steps 3..S must have **no monotonic trend** (R13: 48–66 MB steady, no trend). Slope > 1 MB/step = KILL.
  2. **Device slope** — `device_allocated_bytes` rise per steady step ≤ **262,144 + 2,097,152 B** (R13: +1–2 MB/token).
  3. **Absolute cap** — peak `device_allocated_bytes` ≤ `4,140,417,024 + (prompt_tokens + PROXIMA_MAX_TOKENS) × 262,144 + 41,943,040`.
  4. **`plan_cache_len` ≤ 1** on every step (R13; the `ff749a0` bound must not regress).

## II.8 Item 8 — the measurement mutex, the `-fa 1` arm, the roofline

**The lock does not exist yet, and neither does `flock`.** Verified this session: `ls /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock` → No such file; `which flock` → not found; `/opt/homebrew/bin/flock` → absent. macOS ships no `flock(1)`. **P0.1 is therefore a real card, not boilerplate**, and it is the first card in the plan. Every GPU-measuring or building command in every later card is welded through it.

**The `-fa 1` arm.** Flash attention is OFF by default at the incumbent's checkout (`common/common.h:328`, R8), so R13's 57.08 t/s is the incumbent's *default*, not its *best*. Home-turf discipline requires the arm where their machinery is fully engaged. If `-fa 1` is faster for them, the gap is **larger** than 3.88x and the board must say so before any board-level prediction is made (P0.5).

**The roofline.** `omega/examples/membw_probe.rs` times the whole `execute_plan` call (read verbatim: `Instant::now()` at `:165`, `execute_plan` at `:166`) — block upload, command-buffer creation, `waitUntilCompleted` **and** readback are all inside the window, and the denominator is read-bytes only. Two defects, both fixed in P0.6: (a) time the device window with `command_buffer.GPUEndTime() - GPUStartTime()`, which the codebase already uses in the op-timed path (`metal.rs:734`) but **not** in the batched path (`:546-554` uses host ticks around `commit()`/`waitUntilCompleted()`); (b) add a **streaming-copy** arm (read N, write N) and report **both** denominators, N and 2N. `rooflines.md:411` records the GPU bandwidth ceiling as **DEBT**; this card pays it.

---

# III. Global protocol — binding on every card

**G1 — Tiers.** `hands` = runs commands, applies a diff written by a worker card, cherry-picks, records numbers, appends rows. **Hands never designs and never writes new code.** `worker` = writes source, tests, build steps. `judge` = adjudicates and rules; produces no code. No hybrids.

**G2 — Isolation.** Every card that writes or builds gets its own worktree, branch, and `CARGO_TARGET_DIR`. Never enter another card's worktree. Never enter `/Users/brianbruggeman/repos/slot-0/proxima-wt-*` from R7/R15's set.

**G3 — The measurement mutex.** Every command that builds, probes, benches, or runs the decode harness is welded:
```
flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock <command>
```

**G4 — Pinned environment for every decode-harness invocation.**
```
PROXIMA_MAX_TOKENS=8 CARGO_TERM_COLOR=never CARGO_TARGET_DIR=<card's dir>
```
`PROXIMA_MAX_TOKENS` is read at `proxima-model-interop/src/bind.rs:2719` and defaults to 24; it is pinned to 8 on every card so the KV growth, the RSS window, and the device cap are the same shape in every cell.

**G5 — Counting, EOS-invariant.** Never a literal token count.
```
F = tokens_generated + (stopped_by_eos ? 1 : 0)     # forward calls taken; bind.rs:3057's own quantity
S = F - 1                                            # steady decode steps (prefill excluded)
```
Every `expect` is a formula over `F`, `S`, `prompt_tokens`, and constants read from source. **N == 0 is RED** on every count.

**G6 — The bench ladder.** `nano` (unit test / emitted-source assertion, no device) → `micro` (an `omega/examples/*` probe, one kernel on device) → `milli` (`profiles_one_real_decode_step_by_per_op_gpu_time`, one decode step, per-op) → `bench` (`runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache` + `runs_a_cached_greedy_decode_loop_and_reports_per_token_wall_clock`, full cell vs incumbent). Each card predicts **exactly one rung ahead**. A miss kills the climb and is decomposed into *inconsistency* vs *understanding-gap* with a named work item.

**G7 — Interleaving.** Every compare cell runs arms **A B A B A B**, 3 runs per arm, one measurer on the box, load reported.

**G8 — Correctness oracle.** `generated_text` must stay identical to R13's `"Here is a simple Python function that returns"` and the single-token oracle must stay `2651` / `"known"` (`bind.rs:2797-2803`, llama.cpp's captured answer, §14). Any drift is a KILL regardless of the perf number.

**G9 — The crate gate.** Any card touching omega ends with `bash scripts/omega-gate.sh`, recording `ran_count` from step [3/6] and `passed_count` from step [6/6]. **Either == 0 is RED.** `ran_count` must not decrease card to card; if a card deletes tests it names them.

**G10 — Log rows.** Main's last row is **ROW 233** (`proxima-tensor/docs/discipline.md:18736`). Branches carry the placeholder heading `## ROW <NEXT> -- <title>`; the number is assigned **at land time from main**. Never reuse R12's 234-267 (they collide with three unlanded "ROW 234"s and with physical order).

**G11 — Commits.** Conventional, lowercase, imperative, < 72 chars, one logical change, every commit a green bisect point. **No commit is made without the owner's authorization**; cards prepare the commit and stop.

**G12 — No verdicts, no time estimates.** Cards produce evidence rows.

---

# IV. The cards

## Phase 0 — Measurement truth
*Nothing downstream may be measured until this phase closes. Every GB/s number on main today is wrong (R13's two instrument defects), the roofline is DEBT, and there is no mutex on the box.*

---

### P0.1 — Establish the GPU measurement mutex
- **tier:** hands
- **depends_on:** —
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 -b risc/measure-truth 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target`
- **opens:** nothing in-repo; `/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock` (absent, verified)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && command -v flock; echo "flock_exit=$?"
  cd /Users/brianbruggeman/repos/slot-0/proxima && brew install flock
  cd /Users/brianbruggeman/repos/slot-0/proxima && command -v flock
  cd /Users/brianbruggeman/repos/slot-0/proxima && touch /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
  cd /Users/brianbruggeman/repos/slot-0/proxima && flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock echo lock-ok
  cd /Users/brianbruggeman/repos/slot-0/proxima && uptime && pgrep -l -x cargo; pgrep -l -x llama-bench; system_profiler SPDisplaysDataType | head -30
  ```
- **expect:** first `command -v flock` reports absent (that is the finding, it is why this card exists); after install, `flock … echo lock-ok` prints `lock-ok` — **N==0 lines is RED**. `system_profiler` reports Apple M1 Max, 32-core GPU, Metal 3 (R0).
- **predict:** *nano* — no measurement. Next rung: P0.5's *micro/bench* re-seal reproduces R13's llama arm within its own CoV band (57.08 t/s ± 0.89%).
- **kill:** `brew install flock` unavailable → fall back to a repo-local `scripts/gpu-measure-lock.sh` using `python3 -c 'import fcntl,os,sys; fcntl.flock(os.open(sys.argv[1], os.O_CREAT|os.O_RDWR), fcntl.LOCK_EX); os.execvp(sys.argv[2], sys.argv[2:])'`, and every later card's weld becomes that script. **The plan does not proceed without a working mutex** — an unserialised GPU box makes every CoV in this document a lie.
- **memory gate:** MG-1.
- **rollback:** `rm /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`; `brew uninstall flock`.
- **blast:** the developer host only. No repo file changes.
- **observe:** the lock file's existence; `pgrep` output recorded as the host loadout for every later cell.
- **reprove:** `flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock echo lock-ok`
- **log row:** `## ROW <NEXT> -- the GPU box had no measurement mutex and macOS ships no flock(1)`

---

### P0.2 — Fix `operand_bytes`: report the tensor, not the bound buffer
- **tier:** worker
- **depends_on:** P0.1
- **worktree:** `proxima-wt-risc00` / `risc/measure-truth` (as P0.1)
- **opens:** `omega/src/metal.rs:694-703` (the defect: `.map(|(buffer, _offset)| buffer.length() as u64)`), `omega/src/metal.rs:1786-1815` (why: `checkpoint_mapping_offset` returns the whole-mapping buffer), `proxima-model-interop/src/generate.rs:109-212` (the consumer: total, per-bucket, per-op top, per-family)
- **the defect, verbatim from main:**
  ```rust
  let operand_bytes: u64 = bound.operands().iter()
      .map(|(source, _, _)| device_buffers.get(source)
          .map(|(buffer, _offset)| buffer.length() as u64).unwrap_or(0)).sum();
  ```
  Since `7d09145` addresses weights by offset into ONE buffer, `buffer.length()` is **4,140,417,024** for every matvec (R13) and `total_operand_bytes` reads **1.2 TB**.
- **the fix:** compute from the operand's declared shape and its on-device element encoding — `element_count(prepared.shapes.of(*source))` × (4 for f32/`Float32`; for a node in `packed_operands`, the codec's bytes-per-element: Q4_K = 144/256 = 0.5625, and the sibling constants at `msl.rs:294-556`). This reproduces R13's hand-derived "true bytes/op" column (`rows*k*0.5625`) mechanically instead of in a spreadsheet.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo build -p omega --all-features
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument --release -- --ignored --nocapture profiles_one_real_decode_step_by_per_op_gpu_time 2>&1 | tee /tmp/p02-profile.log
  ```
- **expect:** the `ffn_up` family reports **33.05 MB/op ± 1%** and `attn_v`/`attn_k` report **2.4 MB/op ± 1%**, matching R13's derived column. `total_operand_bytes` for one decode step ≈ **4.07 GB**, not 1.2 TB. N of families reported = 8 (`ffn_up, ffn_gate, ffn_down, attn_q, attn_output, attn_v, attn_k, output.weight`); **N==0 is RED**.
- **predict:** *milli* → next rung *bench*: with correct bytes, `ffn_up` GB/s reads **97.4 ± 3** (R13's derived value), and the whole-decode achieved rate is **4.07 GB / 44.450 ms = 91.6 GB/s** (DERIVED), i.e. **0.40x** the incumbent's achieved 228.9 GB/s.
- **kill:** if the corrected per-family bytes disagree with R13's derived column by > 5%, the shape-derivation is wrong, not the instrument — stop and re-derive from `prepared.shapes` before any GB/s row is written anywhere.
- **memory gate:** MG-3.
- **rollback:** `git revert` the single commit; the old field is restored and every GB/s row on the branch is struck.
- **blast:** instrument-gated reporting only. `#[cfg(feature = "instrument")]` paths in `metal.rs` and the printer in `generate.rs`. Zero effect on `default`.
- **observe:** `operand_bytes` per op; `total_operand_bytes` per step; the per-family table at `generate.rs:109-212`.
- **reprove:** the second command above.
- **log row:** `## ROW <NEXT> -- operand_bytes reported the checkpoint mapping, so every GPU GB/s row was 300x wrong`

---

### P0.3 — Split `block_upload_bytes` into copied bytes and bound bytes
- **tier:** worker
- **depends_on:** P0.2
- **worktree:** `proxima-wt-risc00` / `risc/measure-truth`
- **opens:** `omega/src/metal.rs:465-468` (the defect: `counter!(BLOCK_UPLOAD_BYTES, block_byte_len(block))` fires **before** the upload path is chosen), `:1786-1815` (`checkpoint_mapping_offset`, the no-copy path), `:1879-1892` (`upload_block_no_copy`), `:1903-1910` (`upload_block_no_copy_uncached`), `proxima-model-interop/src/generate.rs:1723-1742` (the printer)
- **the defect:** the counter is incremented unconditionally at the top of the loop, so it counts *bytes considered*, not *bytes copied*. R13 measured **4,147,777,096 B/token** with `mapping_offset_uploads=291` and `copying_uploads=4` — i.e. 291 of 295 blocks were bound by offset and copied nothing.
- **the fix:** move the byte counting **into** the three terminal paths — `BLOCK_COPIED_BYTES` (real `newBufferWithBytes`), `BLOCK_NOCOPY_BOUND_BYTES` (fresh no-copy wrapper), `BLOCK_OFFSET_BOUND_BYTES` (mapping/span offset). Keep `BLOCK_UPLOAD_BYTES` as the sum so no existing row silently changes meaning; add the three as new fields on the printed line.
- **commands:** as P0.2's two commands, plus
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument --release -- --ignored --nocapture runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache 2>&1 | tee /tmp/p03-decode.log
  ```
- **expect:** on a steady step, `block_copied_bytes` ≈ **786,432 B** (3 KV blocks × 262,144, DERIVED from R13's KV slope) and `block_offset_bound_bytes` ≈ 4.14 GB. `block_copied_bytes + block_nocopy_bound_bytes + block_offset_bound_bytes == block_upload_bytes` on **every** step — an identity assertion in the test; **any step where it fails is RED**. Steps counted = `S`; **S==0 is RED**.
- **predict:** *milli* → next rung *bench*: `block_upload` wall stays **2.0 ms ± CoV** (this card changes accounting only, not work). If the wall moves, the counter placement changed a code path and that is a defect in this card.
- **kill:** wall `block_upload` outside 2.0 ± 0.2 ms.
- **memory gate:** MG-3.
- **rollback:** revert; the single summed counter returns.
- **blast:** instrument-only.
- **observe:** the three new counters, `mapping_offset_uploads`, `copying_uploads`, `nocopy_reuses`.
- **reprove:** the third command.
- **log row:** `## ROW <NEXT> -- block_upload_bytes counted bytes considered, not bytes copied; 291 of 295 blocks copy nothing`

---

### P0.4 — Give the batched path a device-side `gpu_exec` window
- **tier:** worker
- **depends_on:** P0.3
- **worktree:** `proxima-wt-risc00` / `risc/measure-truth`
- **opens:** `omega/src/metal.rs:545-555` (host ticks around `commit()`/`waitUntilCompleted()`), `omega/src/metal.rs:734` (the device window the op-timed path already uses: `(GPUEndTime() - GPUStartTime()) * 1e9`)
- **the fix:** add `GPU_DEVICE_NS` alongside `GPU_EXEC_TICKS`, taken from `GPUEndTime()-GPUStartTime()` on the one batched command buffer. Both are reported; neither replaces the other. The difference is the commit + wakeup cost, which is currently silently inside "kernel time".
- **commands:** P0.3's decode command.
- **expect:** `gpu_device_ms ≤ gpu_exec_ms` on every step (identity assertion; a violation is RED). `gpu_exec_ms` mean reproduces R13's **56.93 ± 0.7%**.
- **predict:** *milli* → next rung *bench*: `gpu_exec_ms − gpu_device_ms` is **≤ 1.0 ms/token**, i.e. the 56.93 is real kernel time and not wakeup latency. A gap > 1.0 ms would relocate mass from bucket 2 to bucket 4 of §I.2 and rewrites the diagnosis.
- **kill:** `gpu_device_ms` returns 0 or negative on any step (`GPUStartTime` unpopulated) → the counter is unusable, revert and record the negative.
- **memory gate:** MG-3.
- **rollback:** revert; only `GPU_EXEC_TICKS` remains.
- **blast:** instrument-only.
- **observe:** `gpu_exec_ms`, new `gpu_device_ms`.
- **reprove:** P0.3's decode command.
- **log row:** `## ROW <NEXT> -- the batched gpu_exec window is host ticks; the device window is <N> ms narrower`

---

### P0.5 — Re-seal the baseline with fixed instruments, plus the `-fa 1` incumbent arm
- **tier:** hands
- **depends_on:** P0.4
- **worktree:** `proxima-wt-risc00` / `risc/measure-truth`
- **opens:** `proxima-model-interop/src/bind.rs:3002` (metal decode harness), `:2818` (wall-clock decode harness), `:2719` (`PROXIMA_MAX_TOKENS`)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && uptime | tee /tmp/p05-load-before.log
  cd /Users/brianbruggeman/repos/slot-0/proxima && flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock /Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench -m /Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf -n 32 -p 0 -r 5 -t 8 -ngl 99 2>&1 | tee /tmp/p05-llama-A1.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument --release -- --ignored --nocapture runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache 2>&1 | tee /tmp/p05-ours-B1.log
  cd /Users/brianbruggeman/repos/slot-0/proxima && flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock /Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench -m /Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf -n 32 -p 0 -r 5 -t 8 -ngl 99 -fa 1 2>&1 | tee /tmp/p05-llama-fa-A2.log
  ```
  then repeat the three arms **A B A B A B** for 3 runs each, and `uptime | tee /tmp/p05-load-after.log`.
- **expect:** three arms × 3 runs = **9 result lines; N==0 is RED**. Arm `llama -fa 0` reproduces **57.08 t/s within its own 0.89% CoV**. Arm `ours` reproduces **step_wall 67.92 ± 0.5%**, **gpu_exec 56.93 ± 0.7%**, **op_count == 1196**, `plan_hits=0`, `plan_misses == F`, `generated_text` identical to R13.
- **predict:** *bench* — this is the top rung and it predicts nothing further; it is the anchor. Pre-registered: **`-fa 1` is faster than `-fa 0`**, because `soft_max_ext` + two `mul_mat`s (R8) is three dispatches per layer versus flash attention's one, and the incumbent's default-off is a compatibility choice, not a perf one. If `-fa 1` wins, **the standing gap is worse than 3.88x and every later ratio is quoted against `-fa 1`**.
- **kill:** any arm's CoV > 5% → the box is not quiet, re-run; three consecutive noisy attempts → the box is the finding and no phase past 0 may proceed.
- **memory gate:** MG-3, all four clauses, per run.
- **rollback:** none — a measurement card. If it disagrees with R13 beyond CoV, R13 is superseded and every DERIVED number in this plan is recomputed against the new anchor before Phase 1 starts.
- **blast:** none (read-only measurement).
- **observe:** `step_wall_ms`, `gpu_exec_ms`, `gpu_device_ms` (new, P0.4), all 7 phase counters, `op_count`, `plan_hits`, `plan_misses`, `plan_cache_len`, `kv_cache_upload_bytes`, `device_allocated_bytes`, task RSS, `generated_text`.
- **reprove:** the four commands above.
- **log row:** `## ROW <NEXT> -- 2026-09-03 re-seal on fixed instruments; the -fa 1 incumbent arm added`

---

### P0.6 — Pay the GPU roofline debt: streaming-copy, readback outside the window, both denominators
- **tier:** worker
- **depends_on:** P0.4
- **worktree:** `proxima-wt-risc00` / `risc/measure-truth`
- **opens:** `omega/examples/membw_probe.rs:152-200` (the whole `execute_plan` inside `Instant::now()`; single denominator `elements*4`), `proxima-tensor/docs/rooflines.md:411` ("GPU bandwidth ceiling = DEBT"), `:766-773` (the doc's own note that the GPU ratio "is not a gap-to-machine at all")
- **the fix, three parts:** (a) a **copy** arm — an `Elementwise` `Identity` over N f32 into an N f32 output, so the kernel both reads and writes; (b) the timing window becomes `GPUEndTime−GPUStartTime` (P0.4's counter), with block upload and readback **outside** it; (c) report **both** denominators: `read_bytes = 4N` and `read_plus_write_bytes = 8N`, each with its own GB/s, and never silently pick one.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo run --release -p omega --example membw_probe --features metal,cpu,instrument 2>&1 | tee /tmp/p06-membw.log
  ```
- **expect:** 4 GB/s figures (reduce arm × 2 denominators, copy arm × 2 denominators) at each of 2 buffer sizes = **8 rows; N==0 is RED**. The copy arm's `read_plus_write` GB/s is the machine's streaming ceiling.
- **predict:** *micro* → next rung *milli*: the copy arm's `read_plus_write` GB/s **exceeds 228.9** (the incumbent's *achieved* decode rate, R1), which would establish for the first time that 228.9 is not the machine ceiling and that the incumbent itself has headroom. If it lands **below** 228.9, the reduce-based probe was never measuring bandwidth and R1's 228.9 becomes the ceiling estimate by default — that is the more interesting outcome and must be reported first (§19: report the number that hurts).
- **kill:** copy-arm CoV > 5% across 21 runs.
- **memory gate:** MG-2. Buffers are 64 MB and 256 MB; peak RSS ≤ 400 MB is **not** satisfiable at 256 MB × 2 arms — so this card's MG-2 cap is raised to **1.2 GB** with the arithmetic stated: 256 MB source + 256 MB destination + 256 MB host mirror + 400 MB baseline. Exceeding it is a KILL.
- **rollback:** revert the example; `rooflines.md` keeps the DEBT marker.
- **blast:** one example file plus one doc section. No library change.
- **observe:** `gpu_device_ms` per arm; both GB/s columns; CoV over 21 runs.
- **reprove:** the command above.
- **log row:** `## ROW <NEXT> -- the GPU streaming ceiling, measured: copy arm, device window, both denominators`

---

### P0.7 — ai_docs records for the GPU lane
- **tier:** worker
- **depends_on:** P0.5, P0.6
- **worktree:** `proxima-wt-risc00` / `risc/measure-truth`
- **opens:** `ai_docs/AGENT.md:15-17` ("If the index is missing required structure or evidence, **add records** to `ai_docs` instead of bypassing the structure"), `ai_docs/index.jsonl:1` (schema), `ai_docs/task-routes.jsonl:1` (schema), `ai_docs/invariants.jsonl:1` (schema). R0: these files have **zero** tensor/omega/GPU records.
- **records to add** (schemas copied exactly from the first line of each file):
  - `index.jsonl`: `{"id":"proxima.omega.gpu_lane","kind":3,"summary":"The Metal decode lane: bound plan, route, kernel bodies, and the measurement protocol.","path":"proxima-tensor/docs/discipline.md","read_when":["gpu-perf","omega","metal","decode"],"source_paths":["omega/src/metal.rs","omega/src/msl.rs","proxima-tensor/src/bind.rs","proxima-model-interop/src/generate.rs"],"relations":[]}`
  - `task-routes.jsonl`: task `gpu-decode-perf`, `must_read` = AGENT.md + invariants.jsonl + `proxima-tensor/docs/rooflines.md`, `done_when` = ["every GB/s row cites a measured denominator","every route cites a Route value, never a source substring","memory gate MG-3 recorded on every decode cell"].
  - `invariants.jsonl` × 4: **(i)** one bound plan — `&[BoundOp]` after every driver's own rewrites must be equal across backends (evidence: P2.1's test). **(ii)** route is a value — no route may be recovered by substring of emitted source (evidence: `Route` enum + P3.2's deletion of `classify_kind`). **(iii)** no bare mutex on a per-dispatch path (evidence: `Plan.routes` + atomic totals; the anti-pattern is `instrument.rs:842`). **(iv)** KV arena bytes are a build-time bound (evidence: P6.1's `build.rs` assertion; the failure mode is the 34.36 GB `context_length` default).
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && jq -c . ai_docs/index.jsonl > /dev/null && echo index-ok
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && jq -c . ai_docs/task-routes.jsonl > /dev/null && echo routes-ok
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && jq -c . ai_docs/invariants.jsonl > /dev/null && echo invariants-ok
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc00 && bash ai_docs/query.sh gpu-decode-perf
  ```
- **expect:** 1 index record, 1 task-route record, **4** invariant records = **6 new lines; N==0 is RED**. All three `jq -c` parses clean. `query.sh gpu-decode-perf` returns the new route.
- **predict:** *nano*. Next rung *micro*: no bench effect; this is structure.
- **kill:** any `jq` parse failure.
- **memory gate:** MG-1.
- **rollback:** revert the three appended blocks.
- **blast:** three JSONL files. No source.
- **observe:** line counts before/after each file.
- **reprove:** the four commands.
- **log row:** `## ROW <NEXT> -- the GPU lane enters ai_docs: one index record, one route, four invariants`

---

## Phase 1 — Recover what exists into git; adjudicate the parallel branch
*R7: the measured −17.2% and −4.9% wins live as uncommitted diffs on a 9-commits-stale base. R12: a 42-commit branch on today's main carries a fifth bound kind. Neither may be resolved by momentum.*

---

### P1.1 — Adjudicate `BoundOpKind::CachedAttention`
- **tier:** judge
- **depends_on:** P0.7
- **worktree:** none (read-only adjudication; **do not enter** `proxima-wt-cattn` or any `proxima-wt-*`)
- **opens:** R12 in full; and read-only via git: `git show perf/cached-attention-streaming --stat`, `git show perf/cached-attention-streaming:proxima-tensor/src/physical.rs | head -80`, `git log --oneline main..perf/cached-attention-streaming`
- **the ruling (pre-written; the card's job is to confirm each premise against `git show`, not to re-litigate):**
  - **REJECT** `BoundOpKind::CachedAttention` (5th bound kind; `cpu.rs:19141-19190` CPU arm; `render_cached_attention` `msl.rs:104`; the post-bind structural matcher `cached_attention_candidates` in bind.rs). Three ruling constraints, each independently sufficient: **(a)** AGENTS.md "problem solving" — *"we should not be adding arbitrary rules/code for specific instances"*; a macro-op that pattern-matches one model's attention cluster is exactly that. **(b)** guiding-principles §1 — the expression already exists: two `Reduce`s with an online-softmax combine (`spec.rs:2596-2720`); the type buys no new caller capability. **(c)** Their own measurement refutes the premise: feature-on 616 dispatches at 51.535 ms wall vs feature-off 1194 at 51.571 ms (R12 ROW 262/267) — **halving the dispatch count moved nothing**, so the macro-op's stated purpose is unachieved on the box it was measured on.
  - **KEEP** `prune_dead` / `dead_resolved_nodes` (commit `216d925`) — generic, RISC-conformant, no new kind.
  - **KEEP** the **consumer index** (their ROW 248) — it is the mechanism behind their ROW 247's `prepare` 150.7 → 11.6 ms/token; generic bind work.
  - **KEEP** the **paired Q4_K body** — it enters this plan's bake-off as arm B (P4.x).
  - **KEEP as recorded negatives, never re-proposed:** float4 accumulation, explicit unroll, width-64 dispatch, row-batch 8, tiled-gemm for decode, ggml nsg=2 geometry (ROWs 249, 251-254, 259-260, 265-267). nsg=2 is now a **fourth-time** negative across both lanes (R4 + R12).
  - **KEEP as a defect finding:** their ROW 263 classifier mislabel (9/601 → 225/385) — cited as the motivating evidence for §II.4's `Route` value; their fix (a second marker string) is **not** adopted.
- **expect:** 6 rulings, each with its ruling constraint named and its premise confirmed by a `git show` line. **N rulings == 0 is RED.**
- **predict:** *nano*. Next rung *micro*: `prune_dead` alone lowers `op_count` below 1196 on the P0.5 anchor — the cell that proves it is P1.3.
- **kill:** if `git show` contradicts any premise above (e.g. `prune_dead` turns out to be entangled with the macro-op), the ruling for that item is re-derived from the diff and recorded as a correction, not carried forward.
- **memory gate:** MG-1 (no build).
- **rollback:** n/a — a ruling. Reversal requires new evidence recorded as its own row.
- **blast:** none.
- **observe:** the commit list `main..perf/cached-attention-streaming`; the `--stat` per adopted commit.
- **reprove:** `cd /Users/brianbruggeman/repos/slot-0/proxima && git log --oneline main..perf/cached-attention-streaming`
- **log row:** `## ROW <NEXT> -- adjudicating a fifth bound kind: rejected on three constraints, four pieces kept`

---

### P1.2 — Recover the mask-fma Q4_K body into git on today's main
- **tier:** hands
- **depends_on:** P1.1
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 -b risc/q4k-mask-fma 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/target`
- **opens:** R7 (the diff lives uncommitted in `proxima-wt-all`, `perf/gpu-all-wins`, 13 files +3859/-197, based on `2b95210`); `omega/src/msl.rs:294-556` (codec constants), `omega/src/msl.rs:2526-2528` (packed loop `ib += SIMD_WIDTH/lanes_per_block`), `ggml/src/ggml-metal/ggml-metal.metal:5147-5175` at b25346221 (the upstream mechanism: `kmask1/2/3` branch-free extraction; 1/16 and 1/256 folded into the scale at combine)
- **the extraction, read-only against the other worktree — never enter it, never build in it:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-all diff -- omega/src/msl.rs > /tmp/p12-maskfma.patch
  cd /Users/brianbruggeman/repos/slot-0/proxima && wc -l /tmp/p12-maskfma.patch
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 && git apply --3way /tmp/p12-maskfma.patch || echo "CONFLICT: expected; resolve against main's msl.rs"
  ```
  **The 9 intervening commits (R0) include `spec.rs +8735/-2836` and two `metal.rs` changes; conflicts are the work, not a blocker** (AGENTS.md directive compliance). Only the Q4_K body hunks are taken — the other 12 files in that worktree belong to other cards or to nothing.
- **commands (after resolution):**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock bash scripts/omega-gate.sh 2>&1 | tee /tmp/p12-gate.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument --release -- --ignored --nocapture runs_one_real_forward_pass_and_greedy_picks_a_real_token 2>&1 | tee /tmp/p12-oracle.log
  ```
- **expect:** `omega-gate.sh` prints `== omega gate: PASS ==`; record `ran_count` (**==0 is RED**) and `passed_count` (**==0 is RED**). The greedy oracle asserts token `2651` / `"known"` (`bind.rs:2797-2803`) — **any drift is a KILL under G8**.
- **predict:** *nano* (the gate + oracle) → next rung *micro*: `cargo run --release -p omega --example q4k_matvec_probe` shows the mask-fma body at **≥ 20% lower ns/op** than main's body at the `ffn_up` shape, consistent with R7's −36% on ffn families measured at the milli rung.
- **kill:** oracle drift; or `ran_count` lower than P0.5's recorded value.
- **memory gate:** MG-1 for the gate; MG-3 for the oracle run.
- **rollback:** `git worktree remove /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 --force; git branch -D risc/q4k-mask-fma`. The branch is a recovery vehicle; nothing depends on it until P4.
- **blast:** `omega/src/msl.rs` Q4_K body only, behind the P4.2 selector once that lands; until then, on this branch only.
- **observe:** `ran_count`, `passed_count`, greedy token id and text.
- **reprove:** the two commands.
- **log row:** `## ROW <NEXT> -- the measured -17.2% mask-fma Q4_K body, rebased 9 commits forward and committed`

---

### P1.3 — Recover the paired-nibble Q4_K body, `prune_dead`, and the consumer index
- **tier:** hands
- **depends_on:** P1.1
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 -b risc/q4k-pair-dot 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/target`
- **opens:** R12 (ROW 257 paired body, ROW 248 consumer index, `216d925` prune_dead); `git log --oneline main..perf/q4k-independent-accumulators`
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git log --oneline --reverse main..perf/q4k-independent-accumulators | tee /tmp/p13-commits.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git cherry-pick <sha of the q4k_pair_dot commit>
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git cherry-pick 216d925
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && git cherry-pick <sha of the consumer-index commit>
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock bash scripts/omega-gate.sh 2>&1 | tee /tmp/p13-gate.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc02 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc02/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument --release -- --ignored --nocapture runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache 2>&1 | tee /tmp/p13-decode.log
  ```
  **Each cherry-pick is its own commit and each must be green on its own** (AGENTS.md pr sequencing: every commit a green bisect point). If a cherry-pick drags in `CachedAttention`, it is the wrong commit — split it.
- **expect:** exactly **3 commits** on the branch (**N==0 is RED**); gate PASS with `ran_count` ≥ P0.5's value; decode `op_count` **strictly less than 1196** (that is `prune_dead`'s witness); `generated_text` identical to R13.
- **predict:** *milli* → next rung *bench*: `prepare` drops from **1.97 ms** toward R12 ROW 248's consumer-index result; pre-registered target **≤ 1.4 ms/token** (−30%). The paired body drops the packed-row-blocked bucket from **44.450 ms** toward **31.6 ms** (R12's −29% applied to R13's mass, DERIVED).
- **kill:** `op_count` unchanged at 1196 (then `prune_dead` prunes nothing on this graph, which is itself the finding and the commit is dropped); or `generated_text` drift.
- **memory gate:** MG-3.
- **rollback:** per-commit `git revert`; the three are independent by construction.
- **blast:** `bind.rs` (prune_dead, consumer index) is **cross-backend** — it changes the plan CPU executes too. The CPU oracle (`runs_one_real_forward_pass_and_greedy_picks_a_real_token`) must be run on this branch, not only the Metal path.
- **observe:** `op_count`, `prepare` ms, packed-row-blocked bucket ms, `generated_text`.
- **reprove:** the last two commands.
- **log row:** `## ROW <NEXT> -- three generic pieces recovered from the parallel branch; the fifth bound kind left behind`

---

## Phase 2 — One bound plan (brief item 5)
*The false claim closes here, second in the whole plan, before any route or kernel work rests on it.*

---

### P2.1 — The plan-identity test, pre-registered RED on main
- **tier:** worker
- **depends_on:** P0.7
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 -b risc/one-bound-plan 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target`
- **opens:** `omega/src/metal.rs:1003` (`bind(...)`), `omega/src/metal.rs:1013` (`correct_packed_matmul_layouts(&mut resolved, ...)`), `proxima-tensor/src/bind.rs:1618-1707` (the rewrite and its doc), `proxima-tensor/src/cpu.rs:358` (`bind::bind`, no rewrite), `proxima-tensor/src/bind.rs:200-215` (`BoundOp`), `:95-98` (`Layout`)
- **what the test does:** builds the real cached-forward program for one layer with a real Q4_K weight block, plans it through **both** backends, and captures each backend's `&[BoundOp]` **after that backend's own rewrites** — a `#[cfg(feature = "instrument")]` accessor `Plan::bound_ops(&self) -> &[BoundOp]` on the Metal plan and the CPU equivalent — then asserts full equality.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p omega --features metal,cpu,instrument -- --nocapture the_bound_plan_is_identical_across_backends_after_every_driver_rewrite 2>&1 | tee /tmp/p21-red.log
  ```
- **expect:** the test **FAILS**, and its failure message names the first differing `BoundOp` index, the node id, and both `Layout{base, strides}` values. **A PASS here is RED** — it would mean the test does not capture the post-rewrite plan and the test is wrong, not the code. Differing ops expected = **one per packed matmul weight operand in the fixture; N==0 differing is RED.**
- **predict:** *nano* → next rung *micro*: after P2.2 the same test passes and `q4k_matvec_probe` output is byte-identical before and after, because the rewrite moved, not changed.
- **kill:** the test passes on main → rewrite the test to capture later in the pipeline and re-register.
- **memory gate:** MG-1.
- **rollback:** delete the test.
- **blast:** one test + one `#[cfg(feature="instrument")]` accessor per backend. No production path.
- **observe:** the differing-op count and the two `Layout` values.
- **reprove:** the command above (RED before P2.2, GREEN after).
- **log row:** `## ROW <NEXT> -- one bound plan was false: Metal rewrites the plan after bind, CPU does not`

---

### P2.2 — Bind owns the packed layout; delete the post-bind rewrite
- **tier:** worker
- **depends_on:** P2.1
- **worktree:** `proxima-wt-risc03` / `risc/one-bound-plan`
- **opens:** `omega/src/metal.rs:375` (`packed_operands_of`), `:412` (its call site, currently **after** `prepare`), `:1003`, `:1013`; `proxima-tensor/src/bind.rs:1718` (`pub fn bind`), `:1648` (`correct_packed_matmul_layouts`), `proxima-tensor/src/cpu.rs:3084-3110` (`QuantizedBlock` — the type `packed_operands_of` needs already lives here), `proxima-tensor/src/cpu.rs:358`
- **the change, in three moves:**
  1. `packed_operands_of` descends from omega into proxima-tensor next to `QuantizedBlock`. Nothing is minted: it is a `&[NodeId] × &[QuantizedBlock] -> BTreeSet<NodeId>` function moving to the crate that owns both argument types (§1 — relocation, not a new type).
  2. `bind` gains the packed set: `pub fn bind(program, shapes, outputs, packed: &BTreeSet<NodeId>)`, and `correct_packed_matmul_layouts` becomes its private final step. The public `correct_packed_matmul_layouts` is **removed**, not deprecated (§15).
  3. `omega/src/metal.rs:1013` is **deleted**; `packed_operands` is computed before `prepare` and threaded in. `cpu.rs:358` passes its own set.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p omega --features metal,cpu,instrument -- --nocapture the_bound_plan_is_identical_across_backends_after_every_driver_rewrite 2>&1 | tee /tmp/p22-green.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock bash scripts/omega-gate.sh 2>&1 | tee /tmp/p22-gate.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc03/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-tensor --all-features 2>&1 | tee /tmp/p22-tensor.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc03 && grep -rn "correct_packed_matmul_layouts" --include="*.rs" . | tee /tmp/p22-callsites.log
  ```
- **expect:** P2.1 GREEN. `omega-gate.sh` PASS, `ran_count` ≥ P0.5's, `passed_count` ≥ P0.5's. `grep` for `correct_packed_matmul_layouts` returns **exactly 1 hit** (the private definition inside `bind`) — **more than 1 is RED**, it means a driver still rewrites.
- **predict:** *nano* → next rung *micro*: `q4k_matvec_probe` output is **byte-identical** to the pre-change run; the layout is the same layout, computed one step earlier.
- **kill:** the CPU greedy oracle drifts off `2651`/`"known"` — that would mean the CPU path *does* read packed bytes through `layout_of`, contradicting R16, and the finding supersedes the design (the correction must then live behind a per-backend physical-layout query, not in `bind`).
- **memory gate:** MG-3 (the oracle run).
- **rollback:** `git revert`; `metal.rs:1013` returns and P2.1 goes RED again — which is exactly the signal the revert is meant to produce.
- **blast:** **cross-crate and cross-backend.** `bind`'s signature changes; every caller in proxima-tensor, omega, and proxima-model-interop is touched. This is the widest-signature card in the plan and it is deliberately early, when nothing else is in flight.
- **observe:** P2.1's assertion; `ran_count`; the grep hit count; the greedy oracle.
- **reprove:** the four commands.
- **log row:** `## ROW <NEXT> -- bind owns the packed layout; no backend rewrites the plan after bind`

---

### P2.3 — Re-anchor: same plan, same numbers
- **tier:** hands
- **depends_on:** P2.2, P0.5
- **worktree:** `proxima-wt-risc03` / `risc/one-bound-plan`
- **opens:** `proxima-model-interop/src/bind.rs:3002`, `:2765`
- **commands:** P0.5's `ours` arm and the greedy-oracle command, both against this branch, interleaved A B A B A B with the P0.5 anchor binary.
- **expect:** `step_wall_ms`, `gpu_exec_ms`, `op_count`, and `generated_text` all **within the P0.5 CoV band**. This card is a no-op cell by construction; its value is proving that a signature change across three crates moved zero milliseconds.
- **predict:** *bench* — anchor card, predicts nothing further.
- **kill:** any metric outside the CoV band → the move was not behaviour-preserving; bisect the three sub-moves of P2.2.
- **memory gate:** MG-3.
- **rollback:** see P2.2.
- **blast:** none (measurement).
- **observe:** all P0.5 counters.
- **reprove:** the P0.5 command set on this branch.
- **log row:** `## ROW <NEXT> -- one bound plan lands with zero measured cost`

---

## Phase 3 — One route, as a value, censused lock-free (brief item 4)

### P3.1 — `Route` as the emitter's own decision
- **tier:** worker
- **depends_on:** P2.3
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 -b risc/route-value 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target`
  (rebase onto `risc/one-bound-plan` before building)
- **opens:** `omega/src/msl.rs:673-697` (`emit` routes to render_*), `:824` (`reduce_is_cooperative`), `:751-754` (the load-bearing ordering comment), `:3140-3194` (`push_cooperative_reduce_body`'s three-way split), `:1235` (`packed_row_block`), `:1450` (`tiled_gemm_block`), `:1487` (`diagnose_packed_row_block`), `omega/src/metal.rs:785-826` (`classify_kind`), `:835-854` (`diagnose_kind`)
- **the change:** one `fn route_of(&BoundOp, &BTreeSet<NodeId>) -> (Route, DeclineReason)` in `msl.rs`, called by `emit` **as its dispatch**, so the census and the emitted kernel cannot disagree. `Route` has the 8 live variants of §II.4 — exactly R5's 8 Metal body shapes, no invention. `DeclineReason` reuses `diagnose_packed_row_block`'s existing reason set plus `ScatterNotProvenInjective` (P8.1 fills it).
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p omega --all-features -- --nocapture route 2>&1 | tee /tmp/p31-route.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p omega --all-features -- --nocapture emit_is_deterministic_byte_equal 2>&1 | tee /tmp/p31-det.log
  ```
- **expect:** a table-driven test asserting `route_of` returns each of the **8** variants for a hand-built `BoundOp` at that shape — **8 cases; N==0 is RED**, and a variant with no case is RED (an unroutable variant is an invented variant). `emit_is_deterministic_byte_equal` (`msl.rs:4656`) still passes — the emitted MSL is **byte-identical** to pre-change for every case, proving `route_of` re-expresses the existing gate ordering rather than changing it.
- **predict:** *nano* → next rung *micro*: `q4k_matvec_probe` ns/op unchanged within CoV, because no kernel text changed.
- **kill:** `emit_is_deterministic_byte_equal` fails, or any emitted kernel differs by one byte — the refactor changed routing, which is a different card.
- **memory gate:** MG-1.
- **rollback:** revert; the `if let Some(..)` gates return.
- **blast:** `msl.rs` emit path. `wgsl.rs`/`cuda.rs` are untouched by this card and keep their own gates — they gain `Route` in P3.3 (they must, or "one route" is a Metal-only claim).
- **observe:** the 8-case table; the determinism test.
- **reprove:** both commands.
- **log row:** `## ROW <NEXT> -- the route becomes a value; emitted MSL byte-identical`

---

### P3.2 — Delete `classify_kind`; census from the plan, lock-free
- **tier:** worker
- **depends_on:** P3.1
- **worktree:** `proxima-wt-risc04` / `risc/route-value`
- **opens:** `omega/src/metal.rs:709-710` (call sites), `:785-826`, `:835-854`, `:2243` (`ENCODE_DISPATCH_CALLS`, the atomic pattern to follow), `:1484` (`Counter::new`), `proxima-tensor/src/instrument.rs:842-864` (**the pattern NOT to copy**), `omega/src/msl.rs:1978-1982` (why substring tie-breaks are undecidable)
- **the change:** `Plan.routes: Vec<Route>` filled once at plan build (single-owner, no synchronisation); `static ROUTE_TOTALS: [Counter; ROUTE_COUNT]` bumped once per plan build; `classify_kind` and `diagnose_kind` **deleted**; their call sites read `plan.routes[position]`.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && grep -rn "classify_kind\|diagnose_kind" --include="*.rs" . | tee /tmp/p32-grep.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && grep -rn "Mutex" --include="*.rs" omega/src | tee /tmp/p32-mutex.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock bash scripts/omega-gate.sh 2>&1 | tee /tmp/p32-gate.log
  ```
- **expect:** `classify_kind`/`diagnose_kind` grep = **0 hits** (**any hit is RED**). `Mutex` grep in `omega/src` = **0 hits** (**any hit is RED**; §21 and AGENTS.md hot-path lock-free). Gate PASS, `ran_count` ≥ prior.
- **predict:** *nano* → next rung *milli*: `encode_dispatch` mean is **unchanged at 0.47 ms ± CoV**, because per-dispatch census cost is zero instructions.
- **kill:** `encode_dispatch` > **0.4935 ms** (0.47 × 1.05, R13 CoV band) — the census leaked into the encode loop; move it back to plan build.
- **memory gate:** MG-3. Additional clause: `Plan.routes` adds `1 byte × op_count` = **1196 B/plan** (DERIVED), and `plan_cache_len ≤ 1` bounds it. Anything that makes `routes` grow per token is a KILL.
- **rollback:** revert; the substring classifier returns and every route row on the branch is struck.
- **blast:** instrument reporting + `Plan` struct. `Plan` is constructed in one place (`metal.rs:412`).
- **observe:** the two grep counts; `encode_dispatch` ms; `ROUTE_TOTALS` per variant.
- **reprove:** the three commands.
- **log row:** `## ROW <NEXT> -- the route census is a plan property: zero locks, zero per-dispatch cost`

---

### P3.3 — Route the other two backends; the coverage census
- **tier:** worker
- **depends_on:** P3.2
- **worktree:** `proxima-wt-risc04` / `risc/route-value`
- **opens:** `omega/src/wgsl.rs:105` (the 3 shared types + 2 shared fns), `omega/src/wgsl.rs:364` (`ScatterNotSupported`), `omega/src/cuda.rs:66`, `cuda.rs:146-183` (`emit_cuda` **rejects Iota and Constant** via `CudaUnsupportedOpKind`), `cuda.rs:241`
- **the change:** `route_of` moves to the shared surface; `wgsl` and `cuda` each return a `Route` (including `Declined`) for every `BoundOpKind`. **CUDA's rejection of Iota and Constant becomes `Route::Declined(BackendLacksKind)` — a recorded, censused coverage hole, not a silent `Err`.** This is the honest form of "every backend covers every kind": the plan names the holes rather than claiming they are closed.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc04/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p omega --all-features -- --nocapture backend_route_coverage 2>&1 | tee /tmp/p33-cov.log
  ```
- **expect:** a coverage matrix test, 3 backends × 4 `BoundOpKind` = **12 cells; N==0 is RED**. Pre-registered content: Metal 4/4 covered; WGSL 4/4 covered but with no tiled-GEMM and no packed-row-block route; CUDA **2/4** (Iota and Constant declined). The test asserts the matrix **equals** that pre-registration — a cell changing without a card is RED.
- **predict:** *nano* → next rung *micro*: no device effect; WGSL and CUDA have no decode driver (CUDA is `NotImplemented` in `backend.rs`, verified in `omega/Cargo.toml`'s own feature comment).
- **kill:** the matrix disagrees with the pre-registration → R5's coverage claim is stale and must be re-read before the matrix is written.
- **memory gate:** MG-1.
- **rollback:** revert.
- **blast:** `wgsl.rs`, `cuda.rs` emit entry points.
- **observe:** the 12-cell matrix.
- **reprove:** the command.
- **log row:** `## ROW <NEXT> -- backend route coverage, censused: CUDA is 2 of 4 kinds and now says so`

---

### P3.4 — The full-decode route census cell
- **tier:** hands
- **depends_on:** P3.3
- **worktree:** `proxima-wt-risc04` / `risc/route-value`
- **opens:** `proxima-model-interop/src/generate.rs:1721-1750` (`token_breakdown_metal`)
- **commands:** P0.5's `ours` arm with an added `--nocapture` route dump.
- **expect:** every one of `op_count` ops carries a `(NodeId, Route, DeclineReason)` triple; sum over routes == `op_count` (**identity assertion; failure is RED**). Pre-registered against R13's buckets: `ReduceCooperativePackedRow` = **225**, `ReduceCooperativeGeneric` = **385**, `Elementwise` = **547**, `Constant` = **37**, `Iota` = **2**, `ReduceSerial` = 0, `ReduceCooperativeTiledGemm` = 0 (feature off), `Scan` = 0.
- **predict:** *bench* — anchor. Pre-registered: the route counts **match R13's substring-derived buckets exactly**. If they do not, the substring classifier was mislabelling on main too (as it did on the parallel branch, R12 ROW 263), and **every bucket number in R13 must be restated against routes**.
- **kill:** route sum ≠ `op_count`.
- **memory gate:** MG-3.
- **rollback:** none (measurement).
- **blast:** none.
- **observe:** the route histogram; the per-node census.
- **reprove:** the command.
- **log row:** `## ROW <NEXT> -- 1196 ops, censused by route with a reason, not by grepping MSL`

---

## Phase 4 — The Q4_K body: 26.9 ms, the largest single mass (brief item 6)

### P4.1 — The build-time body selector
- **tier:** worker
- **depends_on:** P3.4, P1.2, P1.3
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 -b risc/q4k-bakeoff 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target`
  (rebase onto `risc/route-value`, then merge in `risc/q4k-mask-fma` and `risc/q4k-pair-dot` as the two bodies)
- **opens:** `omega/omega-runtime.toml` (the sizing file, read in full this session), `omega/build.rs:105` (`emit_sizing_consts`), `:79-85` (`resolve_int` + `rerun-if-env-changed`), `scripts/omega-gate.sh` step **[2/6]** (`--all-targets --all-features` — the constraint that forbids two exclusive features)
- **the change:** the `[q4k] body` key of §II.6, plus `emit_profile_cfg()` in `build.rs` emitting `rustc-check-cfg` + `rustc-cfg` + `rerun-if-env-changed=OMEGA_Q4K_BODY`. `msl.rs` selects the body under `#[cfg(omega_q4k_body = "…")]`. **No new cargo feature.**
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target CARGO_TERM_COLOR=never OMEGA_Q4K_BODY=mask_fma flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock bash scripts/omega-gate.sh 2>&1 | tee /tmp/p41-gate-maskfma.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target CARGO_TERM_COLOR=never OMEGA_Q4K_BODY=pair_dot flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock bash scripts/omega-gate.sh 2>&1 | tee /tmp/p41-gate-pairdot.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target CARGO_TERM_COLOR=never OMEGA_Q4K_BODY=nonsense flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo build -p omega --all-features 2>&1 | tee /tmp/p41-bad.log
  ```
- **expect:** both valid values PASS the **full** gate including `--all-features` (**this is the card's whole point**); record `ran_count`/`passed_count` for each. The nonsense value **fails the build with a named `build.rs` panic** (**a successful build on a nonsense value is RED**).
- **predict:** *nano* → next rung *micro*: `q4k_matvec_probe` under `OMEGA_Q4K_BODY=pair_dot` shows **lower ns/op** than under `mask_fma` at the `attn_q` shape (R12 measured pair-dot on `blk.0.attn_q.weight`), and the reverse or a tie at the `ffn_up` shape (R7 measured mask-fma on ffn). If one body wins **both** shapes at the micro rung, the bake-off's milli rung is expected to agree and a disagreement is the finding.
- **kill:** `--all-features` fails under either value.
- **memory gate:** MG-1.
- **rollback:** revert the `build.rs` and toml hunks; both bodies remain in source, unselectable, and the branch is parked.
- **blast:** `omega/build.rs`, `omega/omega-runtime.toml`, `omega/src/msl.rs` body selection. Nothing outside omega.
- **observe:** two gate runs' counts; the negative-path panic message.
- **reprove:** the three commands.
- **log row:** `## ROW <NEXT> -- two Q4_K bodies, one build-time selector: exclusive without breaking --all-features`

---

### P4.2 — The interleaved bake-off and the terminal tie-break
- **tier:** hands
- **depends_on:** P4.1
- **worktree:** `proxima-wt-risc05` / `risc/q4k-bakeoff`
- **opens:** `omega/examples/q4k_matvec_probe.rs`, `proxima-model-interop/src/bind.rs:3084` (per-op profile), `:3002` (decode cell)
- **commands (one full sweep; run the sweep 3 times, arms interleaved A B A B A B):**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target CARGO_TERM_COLOR=never OMEGA_Q4K_BODY=mask_fma PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument --release -- --ignored --nocapture profiles_one_real_decode_step_by_per_op_gpu_time 2>&1 | tee /tmp/p42-A1.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/target CARGO_TERM_COLOR=never OMEGA_Q4K_BODY=pair_dot PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument --release -- --ignored --nocapture profiles_one_real_decode_step_by_per_op_gpu_time 2>&1 | tee /tmp/p42-B1.log
  ```
  plus the P0.5 `ours` decode arm under each value, and the incumbent arm at both `-fa 0` and `-fa 1`, all inside the same interleaved sweep.
- **expect:** 2 bodies × 3 runs × 2 rungs (milli, bench) = **12 cells; N==0 is RED**. Every cell carries `generated_text` == R13's string (G8) and the parity number vs f32 on `blk.0.attn_q.weight`.
- **predict:** *bench* — the winning body drops `ReduceCooperativePackedRow` from **44.450 ms** to **≤ 33.0 ms** (R12's −29%, DERIVED against R13's mass) and `gpu_exec` from **56.93** to **≤ 45.5 ms**, moving the kernel-only ratio from 3.25x to **≤ 2.60x**.
- **tie-break:** §II.6's four-step chain, applied in order, terminating at step 4 by construction.
- **kill:** neither body beats main's body outside the CoV band → **both** are recorded as negatives and the 26.9 ms bucket returns to the open-questions table with a new work item (the incumbent's `<4,2,32>` template geometry, `ggml-metal.metal:5086`, becomes the next hypothesis). Also KILL: parity error vs f32 > **1e-4** on any real weight tensor (R12's recorded 3.1e-6 is the standard).
- **memory gate:** MG-3, both arms, every run.
- **rollback:** flip `[q4k] body` back in the toml — a one-line, no-rebuild-of-logic revert. This is why the selector is a config key and not a code fork.
- **blast:** the packed-row-blocked route only. 225 of 1196 ops (R13).
- **observe:** `ReduceCooperativePackedRow` bucket ms; per-family ms; `gpu_exec_ms`; `gpu_device_ms`; parity max-abs error; `generated_text`.
- **reprove:** the sweep commands.
- **log row:** `## ROW <NEXT> -- Q4_K bake-off: two independent re-derivations of ggml's mask-without-shift, one landed`

---

### P4.3 — Land one, record the other as a negative
- **tier:** judge
- **depends_on:** P4.2
- **worktree:** `proxima-wt-risc05` / `risc/q4k-bakeoff`
- **opens:** P4.2's 12 cells; `proxima-tensor/docs/discipline.md:18736`
- **expect:** one row for the winner with its delta vs the P0.5 anchor; one row for the loser with its measured number and the tie-break step that decided it. **A loser row with a blank number is RED.** The losing body's source is **kept**, selectable by `OMEGA_Q4K_BODY`, because a body that lost by 3% on one machine is the first thing to try on the next one.
- **predict:** none (a ruling).
- **kill:** n/a.
- **memory gate:** MG-1.
- **rollback:** the toml key.
- **blast:** `omega-runtime.toml` default value.
- **observe:** the two rows.
- **reprove:** P4.2's sweep.
- **log row:** `## ROW <NEXT> -- <winner> lands as the default Q4_K body; <loser> recorded at <delta>`

---

## Phase 5 — KV device residency (brief item 3, first half)

### P5.1 — Generalize the registered host span
- **tier:** worker
- **depends_on:** P4.3
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 -b risc/kv-span 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target`
  (rebase onto `risc/q4k-bakeoff`)
- **opens:** `omega/src/metal.rs:1744-1752` (`CHECKPOINT_MAPPING: RefCell<Option<(usize,usize)>>`), `:1768-1773` (`register_checkpoint_mapping`), `:1786-1815` (`checkpoint_mapping_offset`, whose own doc says the KV buffers "fall through unchanged"), `omega/src/backend.rs:402-414`, `omega/src/metal.rs:1616` (`is_page_aligned`), `:1914` (`create_no_copy_buffer` — page-aligned pointer **and** length required)
- **the change:** the single `Option` slot becomes a fixed-size array of `OMEGA_RESIDENT_SPAN_SLOTS` entries, that constant coming from a new `[spans] slots` key in `omega-runtime.toml` via `emit_sizing_consts` (§12 — no bare `const` in source). `register_checkpoint_mapping` becomes `register_host_span(name, bytes) -> SpanSlot`; the checkpoint is slot 0 and its existing call site is updated. `checkpoint_mapping_offset` becomes `host_span_offset`, scanning the slots. `MAPPING_OFFSET_UPLOADS` gains a per-slot breakdown.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && grep -rn "register_checkpoint_mapping\|CHECKPOINT_MAPPING" --include="*.rs" . | tee /tmp/p51-grep.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock bash scripts/omega-gate.sh 2>&1 | tee /tmp/p51-gate.log
  ```
- **expect:** `register_checkpoint_mapping` grep = 0 hits; `register_host_span` present at exactly the loader call site + the new KV call site (P5.2). Slot-exhaustion test: registering `slots + 1` spans returns a named error, never silently overwrites — **1 negative-path test; N==0 is RED**. Gate PASS.
- **predict:** *nano* → next rung *micro*: `mapping_offset_uploads` on a decode step is **unchanged at 291** (R13), because slot 0 behaves exactly as the old singleton did.
- **kill:** `mapping_offset_uploads` ≠ 291 ± 0 before P5.2 lands.
- **memory gate:** MG-3. The span table is `slots × 16 B`, fixed at build time — state the number in the row.
- **rollback:** revert; the singleton returns.
- **blast:** `omega/src/metal.rs` upload path + `backend.rs` re-export + the loader call site in proxima-model-interop.
- **observe:** the two greps; `mapping_offset_uploads` per slot.
- **reprove:** the two commands.
- **log row:** `## ROW <NEXT> -- one registered-span primitive, N slots, checkpoint is slot 0`

---

### P5.2 — A capacity-reserved, page-aligned KV arena
- **tier:** worker
- **depends_on:** P5.1
- **worktree:** `proxima-wt-risc06` / `risc/kv-span`
- **opens:** `proxima-model-interop/src/generate.rs:621-654` (`LayerCache`, `Vec::new()`, three `extend_from_slice`), `proxima-tensor/src/align.rs:42-46` and `:69` (`AlignedBuffer::new` — **zero production callers**), `omega/src/metal.rs:1903` (`upload_block_no_copy_uncached` — the fresh-buffer-every-token path), `:991-1000` (`InputSizeMismatch`, **preserved**), `proxima-model-interop/src/serving.rs:161` (`context_length: 131_072` — the 34.36 GB trap)
- **the change:** `LayerCache` is backed by one `AlignedBuffer` per component, sized `kv_capacity_tokens × row_elements` at load, registered once as a host span. `append` writes into the reserved region; **the base pointer never moves.** `named_blocks` still hands `&arena[..cached_len*row]`, so the element count still equals the declared `Symbolic(1)` extent and the strict check at `metal.rs:991-1000` is untouched. The KV blocks now resolve through `host_span_offset` — one no-copy buffer for the whole arena, addressed by offset, exactly as the checkpoint is.
- **`kv_capacity_tokens` is a build-time key** (`proxima-model-interop-runtime.toml` + a new `build.rs`), with the §II.7 assertion failing the build on violation. **`context_length` never sizes an allocation.**
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target CARGO_TERM_COLOR=never PROXIMA_KV_CAPACITY_TOKENS=1000000 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo build -p proxima-model-interop --features metal,instrument 2>&1 | tee /tmp/p52-trap.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument --release -- --ignored --nocapture runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache 2>&1 | tee /tmp/p52-decode.log
  ```
- **expect:** the 1,000,000-token build **fails** with the byte arithmetic in the panic message (**a successful build is RED — that is the 34 GB trap re-opened**). On the decode run: `nocopy_reuses` ≥ `3 × layers × S` = `96 × S` (**S==0 is RED**), `copying_uploads` for KV nodes == **0**, `block_copied_bytes` (P0.3) for KV == **0** on every steady step, `generated_text` == R13's.
- **predict:** *milli* → next rung *bench*: `block_upload` drops from **2.00 ms** to **≤ 0.5 ms/token**, since 291 of 295 blocks were already offset-bound (R13) and the remaining 3–4 KV blocks become offset-bound too. `step_wall` drops by the same **~1.5 ms**; `gpu_exec` is **unchanged** (this card moves no GPU work — if `gpu_exec` moves, something else changed).
- **kill:** any of — `gpu_exec` outside its CoV band; `device_allocated_bytes` peak above §II.7's cap; RSS slope > 1 MB/step; `generated_text` drift.
- **memory gate:** MG-3, with clause 3's cap computed at `kv_capacity_tokens = 512` = **4,316,577,792 B**. Note the arena is now allocated **up front** (512 × 262,144 = 134,217,728 B), so prefill RSS rises by ~134 MB; the MG-2/MG-3 RSS ceiling for this card and every later one is therefore **540 MB**, stated here once with its arithmetic.
- **rollback:** revert; `LayerCache` returns to `Vec::new()`. The build-time trap assertion is kept regardless — it is a correctness guard, and §15 forbids reverting a repair to restore a perf number.
- **blast:** `generate.rs` KV path; a new `build.rs` on proxima-model-interop; the arena is allocated per `generate` call.
- **observe:** `block_upload_ms`, `block_copied_bytes`, `nocopy_reuses`, `nocopy_cache_len`, `kv_cache_upload_bytes`, `device_allocated_bytes`, RSS.
- **reprove:** the two commands.
- **log row:** `## ROW <NEXT> -- the KV cache becomes device-resident by offset; AlignedBuffer gets its first production caller`

---

## Phase 6 — Plan stability (brief items 2 and 3, second half)

### P6.1 — Bucket the KV leaf extent; mask the tail with `Greater` + `Select`
- **tier:** worker
- **depends_on:** P5.2
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 -b risc/plan-stable 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target`
  (rebase onto `risc/kv-span`)
- **opens:** `proxima-tensor/src/spec.rs:6216-6245` (the three `Extent::Symbolic(1)` KV leaves — read verbatim this session), `:823-845` (`causal_mask`: two `Iota` + `Greater` + `scalar_constant(NEG_INFINITY)`), `:2604-2615` (`Select(is_future, neg_infinity, score_new_scaled)` — the exact consumption pattern to mirror), `proxima-tensor/src/op.rs:60-78` (**17 bodies; no `Less`, no `GreaterEqual`**), `proxima-model-interop/src/generate.rs:966` (the plan key), `:1304-1309` (`build_position_inputs` — keeps the TRUE `cached_len`)
- **the change:** §II.2, behind a default-off feature `kv-capacity-bucket` (proxima-tensor), with the passthrough chain `proxima-model-interop/kv-capacity-bucket = ["proxima-tensor/kv-capacity-bucket"]` following the pattern R15 records for every existing omega feature. `[kv] bucket_tokens` joins `[kv] capacity_tokens` in the sizing toml.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-tensor --features config,kv-capacity-bucket -- --nocapture the_cache_tail_mask_zeroes_every_slot_at_or_past_the_valid_length 2>&1 | tee /tmp/p61-mask.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument --release -- --ignored --nocapture runs_one_real_forward_pass_and_greedy_picks_a_real_token 2>&1 | tee /tmp/p61-oracle.log
  ```
- **expect:** the CPU mask test asserts, for `bucket ∈ {8, 32, 256}` and `cached_len` spanning a bucket boundary in each, that the attention output is **bit-identical** to the unbucketed output — **3 × 3 = 9 cases; N==0 is RED**. The greedy oracle holds `2651`/`"known"`. `op_count` rises by **exactly 2** (§II.2's pre-registration).
- **predict:** *nano* → next rung *micro*: `op_count` == 1198 (or `prune_dead`'s value + 2). If it is +34, the `Select` did not fuse into `score_cached_scaled`'s `ComposedBody` and that is the finding — the fix is then a bind-level composition question, not a graph question.
- **kill:** any of — the CPU mask test not bit-identical; greedy oracle drift; `op_count` delta ∉ {2, 34} (an unexplained delta means the graph changed in a way this card did not design).
- **memory gate:** MG-3; RSS ceiling 540 MB per P5.2.
- **rollback:** feature off — one flag, zero source revert. This is why it is a feature and not an edit.
- **blast:** `spec.rs` cached-layer builder (`append_mistral_cached_layer:2336`, sole caller `:6282`) and the three KV leaves. **Under the feature only**; `default` is untouched.
- **observe:** `op_count`, the CPU parity cases, greedy oracle.
- **reprove:** the two commands.
- **log row:** `## ROW <NEXT> -- the KV extent buckets to a capacity; the tail masks with Greater + Select (there is no Less)`

---

### P6.2 — Invert the `plan_hits == 0` assertion, `#[cfg]`-paired, with a formula
- **tier:** worker
- **depends_on:** P6.1
- **worktree:** `proxima-wt-risc07` / `risc/plan-stable`
- **opens:** `proxima-model-interop/src/bind.rs:2996-2998` (the doc that states the finding), `:3052-3055` (`assert_eq!(runtime.plan_hits, 0, "cached_len grows every decode step, so no (new_count, cached_len) shape can repeat")`), `:3056-3059` (`plan_misses == forward_calls_taken`), `generate.rs:966` (the key), `:973` (`self.plans.clear()` on miss — the cache holds **exactly one** entry, so a hit requires the key to equal the **immediately preceding** key)
- **the change:** the existing pair of assertions is `#[cfg(not(feature = "kv-capacity-bucket"))]`; a new pair is `#[cfg(feature = "kv-capacity-bucket")]`:
  ```rust
  let bucket = |value: usize| value.div_ceil(KV_BUCKET_TOKENS) * KV_BUCKET_TOKENS;
  let mut expected_misses = 1usize;                       // the prefill plan
  let mut previous_key = (prompt_tokens, 0usize);         // (new_count, cached_len)
  for step in 1..forward_calls_taken {
      let key = (1usize, bucket(prompt_tokens + step - 1));
      if key != previous_key { expected_misses += 1; }
      previous_key = key;
  }
  assert_eq!(runtime.plan_misses, expected_misses, "...");
  assert_eq!(runtime.plan_hits, forward_calls_taken - expected_misses, "...");
  assert!(runtime.plan_hits > 0, "a bucketed cache extent that never hits is the null result, not a pass");
  ```
  `forward_calls_taken = generated.0.len() + usize::from(generated.2)` — the file's own existing EOS-invariant quantity (`bind.rs:3051`). **No literal token count anywhere.**
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument --release -- --ignored --nocapture runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache 2>&1 | tee /tmp/p62-off.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc07 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc07/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument,kv-capacity-bucket --release -- --ignored --nocapture runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache 2>&1 | tee /tmp/p62-on.log
  ```
- **expect:** feature **off** → the original assertions hold, `plan_hits == 0`, `plan_misses == F` (unchanged behaviour on `default`). Feature **on** → `plan_hits > 0` (**`plan_hits == 0` is RED**) and both counts equal the formula exactly. With `PROXIMA_MAX_TOKENS=8` and any bucket ≥ 8 that the prompt does not straddle, the formula yields `expected_misses = 2` and `plan_hits = F − 2`.
- **predict:** *bench* — `prepare` + `op_setup` = **5.87 ms/token** on `F − 2` of `F` steps becomes near-zero; pre-registered `step_wall` reduction **≥ 4.5 ms/token** (5.87 × (F−2)/F at F ≥ 8, DERIVED). Note: `op_setup` is per-op buffer and uniform allocation (R11 M6''), which a plan hit alone does **not** remove — so the honest pre-registration is **`prepare` → ≤ 0.2 ms** and **`op_setup` unchanged at 3.9 ms**, a `step_wall` reduction of **~1.7 ms**. `op_setup` is P7's target, not this card's. Reporting the smaller number here is the point.
- **kill:** feature-off behaviour changes at all.
- **memory gate:** MG-3; additionally `plan_cache_len ≤ 1` on every step (a bucketed key that fills the map is a leak re-opened).
- **rollback:** feature off.
- **blast:** one test in `bind.rs`, `#[cfg]`-paired so both worlds are asserted.
- **observe:** `plan_hits`, `plan_misses`, `plan_cache_len`, `prepare` ms, `op_setup` ms.
- **reprove:** the two commands.
- **log row:** `## ROW <NEXT> -- the harness asserted plan_hits==0; now it asserts the formula, both ways`

---

### P6.3 — The bucketing trade cell: orchestration saved vs cached-range work added
- **tier:** hands
- **depends_on:** P6.2
- **worktree:** `proxima-wt-risc07` / `risc/plan-stable`
- **opens:** R13's per-family table (`kv_cache.v` 1.559, `kv_cache.k_odd` 0.783, `kv_cache.k_even` 0.774 ms, Σ **3.116 ms**)
- **commands:** the P6.2 pair, run at `OMEGA_KV_BUCKET_TOKENS ∈ {8, 32, 256}`, interleaved with the feature-off control, 3 runs each.
- **expect:** 4 configurations × 3 runs × 2 rungs = **24 cells; N==0 is RED**.
- **predict:** *bench* — the reduce over the cache axis now runs over `bucket` slots instead of `cached_len`, so the three `kv_cache.*` families inflate by a factor of `bucket / mean(cached_len)`. At `PROXIMA_MAX_TOKENS=8` and bucket 256 the inflation factor is large and the pre-registered outcome is **a net LOSS at short context**: added GPU work ≫ 1.7 ms saved. At bucket 8 the inflation is small and the pre-registered outcome is a **net win of ~1.5 ms**. **The prediction is that bucket size has an optimum at roughly `bucket ≈ prompt_tokens + PROXIMA_MAX_TOKENS`, i.e. that bucketing buys plan stability only when the padding is small relative to the real context — which is precisely why the incumbent, running at real context lengths, can pad to 256 and we cannot at 8.**
- **kill (the decisive one):** if `Δ(kv_cache.* families gpu_ms) > Δ(prepare + op_setup ms)` at **every** bucket value, bucketing is **dead as a landing route** and the fork resolves to the shape-invariant plan (P6.4). Record the loss; do not soften it.
- **memory gate:** MG-3 at each bucket; the device cap uses `kv_capacity_tokens`, not `bucket`.
- **rollback:** feature off.
- **blast:** none beyond P6.1/P6.2.
- **observe:** the three `kv_cache.*` families' gpu_ms; `prepare`; `op_setup`; `plan_hits`; `step_wall_ms`; `gpu_exec_ms`.
- **reprove:** the sweep.
- **log row:** `## ROW <NEXT> -- bucketing the KV extent: the orchestration saving against the padding cost, by bucket size`

---

### P6.4 — *(contingent on P6.3's kill)* The shape-invariant plan
- **tier:** worker
- **depends_on:** P6.3 **and only if P6.3 killed bucketing at every bucket size**
- **worktree:** `proxima-wt-risc07` / branch `risc/plan-stable-invariant`
- **opens:** `omega/src/msl.rs:2207-2218` (the `Uniforms` struct: `output_total`, `reduction_total`, `output_extents[]`, `reduction_extents[]`, `operand_base[]`, `operand_strides[][]`, `out_base`, `out_strides[]` — **every one already a per-dispatch uniform, verified this session**), `omega/src/metal.rs:2070` (`upload_uniforms`), `:2069` (`UNIFORM_BUFFER_REUSES`), `msl.rs:1517-1560` (`grid_threads`), `:731` (`kernel_cache_key`), `proxima-tensor/src/bind.rs:200-215` (`BoundOp.extents: Vec<u64>` — the one baked thing)
- **the design, stated now so the fork is decidable, built only if reached:** split `Plan` into a **shape-invariant** part (kernel source, pipelines, routes, retirement, packed operands, resident marking) and a **per-token** part (uniform bytes, grid dims, output buffer sizes). The evidence this is cheap: `pipeline_lookup` is already **0.04 ms** (R13), so pipelines already survive `cached_len` changes — the MSL source is already extent-independent. What is rebuilt per token is bind + retirement + packed-operand resolution, none of which depends on the *value* of `cached_len`, only on its position in `symbols`. Output buffers for KV-dependent intermediates are allocated at `kv_capacity_tokens`; the **grid is dispatched at the true extent**, so there is no padding cost at all and no tail mask is needed.
- **expect:** `plan_hits == F − 1` (every step after the prefill), `op_count` unchanged, `generated_text` unchanged.
- **predict:** *bench* — `prepare` → ≤ 0.2 ms with **zero** GPU inflation, i.e. strictly better than bucketing's best cell.
- **kill:** `gpu_exec` rises at all (it must not; the grid is unchanged).
- **memory gate:** MG-3; capacity-sized intermediates add `kv_capacity_tokens × head_dim × kv_heads × 4 B` per KV-dependent intermediate — state the count and the total in the row before running.
- **rollback:** the split is behind the same feature flag.
- **blast:** `Plan` construction in omega; the widest omega-internal change in the plan. **This is why it is contingent** — it is only paid for if P6.3 proves the cheap route cannot work.
- **observe:** `plan_hits`, `prepare`, `gpu_exec`, `op_count`, `UNIFORM_BUFFER_REUSES`.
- **reprove:** the P6.2 command pair.
- **log row:** `## ROW <NEXT> -- the plan splits shape-invariant from per-token; extents were already uniforms`

---

## Phase 7 — The non-matmul 16.6 ms

### P7.1 — Cooperative-reduce width as a sizing axis
- **tier:** worker
- **depends_on:** P6.3 (or P6.4 if reached)
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 -b risc/wide-reduce 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target`
- **opens:** `omega/src/sized.rs:45` (`SIMD_WIDTH=32`, a hardware fact — **not** the tunable), `omega/src/msl.rs:3140-3194` (`push_cooperative_reduce_body`; `output_index = gid/32`, `lane = gid%32` at `:3190-3194`), `:1517-1560` (`grid_threads`: cooperative = `output_total*32`), `ggml/src/ggml-metal/ggml-metal.m:3797-3804` at b25346221 (**nth doubles from 32 up to `min(ne00/4, maxTotalThreadsPerThreadgroup)`**), `ggml-metal.metal:1679-1721` (float4 loads, `simd_sum` → threadgroup shmem → `simd_sum`), R7 (`metal-wide-cooperative-reduce`, uncommitted, measured **−20%** on this bucket / **−2.8 ms**)
- **the change:** a `[cooperative_reduce] max_threads` key in `omega-runtime.toml` (§12 — the width is a tunable, `SIMD_WIDTH` is not) and a two-level tree body (`simd_sum` → threadgroup memory → `simd_sum`) with float4 loads when the reduction extent is a multiple of 4. Behind `Route::ReduceCooperativeGeneric`, so the route census immediately shows how many of R13's **385** ops take the wide path.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p omega --all-features -- --nocapture cooperative 2>&1 | tee /tmp/p71-unit.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc08/target CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-model-interop --features metal,instrument --release -- --ignored --nocapture profiles_one_real_decode_step_by_per_op_gpu_time 2>&1 | tee /tmp/p71-profile.log
  ```
- **expect:** CPU-oracle parity cases across reduction extents `{31, 32, 33, 127, 128, 4096}` — **6 cases; N==0 is RED** — proving the tree is correct at non-multiples of the width, which is where a two-level reduction breaks.
- **predict:** *milli* → next rung *bench*: `ReduceCooperativeGeneric` drops from **9.113 ms** to **≤ 7.3 ms** (R3 M4's −20%, DERIVED), i.e. **−1.8 ms** on `gpu_exec`.
- **kill:** parity failure at any non-multiple extent; or the bucket does not drop by ≥ 10% (below that, R7's −20% did not survive the rebase and the row records the negative).
- **memory gate:** MG-3. Threadgroup memory is device-side and does not touch `device_allocated_bytes`; state the per-threadgroup bytes (`max_threads/32 × 4 B`) in the row.
- **rollback:** set `max_threads = 32` in the toml — the old behaviour exactly, no source revert.
- **blast:** `Route::ReduceCooperativeGeneric` only — 385 of 1196 ops (R13). Does **not** touch the packed-row or tiled-GEMM routes.
- **observe:** the route histogram; the cooperative bucket ms; per-op ns.
- **reprove:** the two commands.
- **log row:** `## ROW <NEXT> -- the cooperative reduce stops being 32 lanes wide for every size`

---

## Phase 8 — Write placement and the op count (brief item 1)
*Scheduled last on purpose. R12's own control (616 vs 1194 dispatches, 51.535 vs 51.571 ms wall) says this lever moves the least. It is here for the RISC, not for the milliseconds, and the plan says so.*

### P8.1 — Structural injectivity at bind; the affine scatter degenerates
- **tier:** worker
- **depends_on:** P7.1
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 -b risc/write-placement 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target`
- **opens:** `proxima-tensor/src/map.rs:110-131` (the write-direction convention, read verbatim: `offset` at `gathered_dim` carries the destination extent; the CPU loop is sequential so scatter needs no atomics), `:175-201` (`IndexMap::scatter`), `:209` (`scatter_extent`), `:238` (`as_gather_from_output`), `proxima-tensor/src/shape.rs:469-485` (`project_output_shape`: `[term] if term.coeff == 1 => Ok(...)` else `NotLowerable{reason: "reduce output maps must be pure projections in v1"}` — **THE line**), `:441-467` (`bounds_check` already handles `axis.offset` on the read side), `proxima-tensor/src/bind.rs:1594-1606` (`layout_of`: `base += offset * stride`), `:1011` (`build_scatter_out_layout`), `:95-98` (`Layout`), `omega/src/msl.rs:933`, `wgsl.rs:364`, `cuda.rs:241` (the three rejections), `proxima-tensor/src/cpu.rs:6911` (`run_reduce_scatter`)
- **the change:** §II.1's structural prover, run inside `bind`. On ACCEPT the scatter folds into `out_layout.base`/`out_strides` and `out_scatter` becomes `None`; on REJECT the emitters' `ScatterNotSupported` stays and the decline is recorded as `Route::Declined(ScatterNotProvenInjective)`. `project_output_shape` accepts a pure projection **plus an offset** on the write side, mirroring what `bounds_check` already does on the read side.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p proxima-tensor --all-features -- --nocapture injectiv 2>&1 | tee /tmp/p81-inj.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc09/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo test -p omega --all-features -- --nocapture scatter 2>&1 | tee /tmp/p81-scatter.log
  ```
- **expect:** a case table with **both** directions — ACCEPT: `Iota`; `Add(Iota, rank-0 Constant)`; `Add(Iota, rank-0 Input)`; `Multiply(Iota, rank-0 Constant≠0)`. REJECT: `Multiply(Iota, Constant(0))` (coeff 0, not injective); a gather-fed index (data-dependent); an index through `Maximum` (not affine); an index through a non-rank-0 operand. **8 cases minimum; N==0 is RED, and a table with only ACCEPT cases is RED** — the fallback is the load-bearing half.
- **predict:** *nano* → next rung *micro*: on ACCEPT, the emitted MSL contains **no gather-index read** for that operand and `u.out_base` carries the offset. Assert on the emitted text of the **synthetic** fixture (not the decode program — R16's undecidability applies only to grepping the real concatenated codec region).
- **kill:** any REJECT case that the prover ACCEPTs. A false ACCEPT is a **race on the GPU**, i.e. a correctness defect, and kills the card outright regardless of the perf story.
- **memory gate:** MG-1.
- **rollback:** revert; `project_output_shape` returns to `coeff == 1, offset ignored`, and every scatter is rejected as today.
- **blast:** `shape.rs` + `bind.rs`, **cross-backend**. The CPU scatter path (`cpu.rs:6911`) must be re-tested: an ACCEPT that used to go through `run_reduce_scatter` now goes through the ordinary strided store, and the two must agree bit-for-bit.
- **observe:** the ACCEPT/REJECT table; `Route::Declined` counts on the real decode program.
- **reprove:** the two commands.
- **log row:** `## ROW <NEXT> -- injectivity proved by shape, never by a leaf name; the affine scatter stops being a scatter`

---

### P8.2 — Single-range attention; the op-count cell
- **tier:** worker
- **depends_on:** P8.1
- **worktree:** `proxima-wt-risc09` / `risc/write-placement`
- **opens:** `proxima-tensor/src/spec.rs:2303-2319` (the doc that names the IR constraint: *"`Reduce::out_map` must stay a pure projection … so nothing upstream of a reduce can splice two tensors into one axis"*), `:2336-2865` (`append_mistral_cached_layer`, 530 lines, 25 args), `:2616-2617`, `:2596-2720` (the two-range online-softmax combine), R3 M11 (single-range proven 488/488, 1196 → 939 BoundOps, driver unbuilt), R8 (`ggml_cpy(k_cur, ggml_view_1d(k, ...))` — the incumbent's in-place write at a byte offset, `llama-kv-cache-unified.cpp:749-788`)
- **the change:** with P8.1's placement available, the new token's K/V is written into the KV arena **in the graph** at `out_layout.base = cached_len * row`, and attention reduces over ONE range. Behind a default-off feature.
- **commands:** the CPU parity harness, the greedy oracle, then P0.5's decode arm.
- **expect:** CPU parity **488/488** (R3 M11's own count; a lower number is RED); `op_count` **939 or lower** (R3 M11); `generated_text` == R13's.
- **predict:** *bench* — **this card predicts a wall-clock change of approximately ZERO.** R12 measured 616 vs 1194 dispatches at 51.535 vs 51.571 ms (R12 ROW 262/267). Pre-registered: `step_wall` moves by **less than 1 ms**, and `gpu_exec` may **rise** slightly (one wide reduce over `cached_len + 1` slots instead of two narrow ones). **The value of this card is the RISC — 1196 → 939 dispatches, `Concat` still nonexistent, no new `Op` — not the milliseconds.** Predicting the win here would be the dishonest move.
- **kill:** `step_wall` **rises** by more than the CoV band; or CPU parity below 488/488; or the `Layout.base` runtime-source extension proves necessary and its blast radius exceeds one file — in which case the card parks with the named un-park condition "when a real-context (≥1024 token) decode cell exists", because at that context the KV write dominates differently.
- **memory gate:** MG-3.
- **rollback:** feature off.
- **blast:** `spec.rs` cached-layer builder; possibly `Layout.base`'s representation (the ONE new thing this plan may mint, §II.1).
- **observe:** `op_count`, per-route histogram, `gpu_exec`, `step_wall`, CPU parity count.
- **reprove:** the three commands.
- **log row:** `## ROW <NEXT> -- single-range attention at 939 ops: the RISC wins, the clock does not`

---

## Phase 9 — The two runtimes with no cell on either side

*R9 is blunt: "llama, ggml, ort and torch can beat us on gpu" is **MEASURED only for llama.cpp-Metal**. For ORT-CoreML and torch-MPS there is **no cell on either side**. These two cards create the first ones, and their scope is stated honestly rather than stretched to look like a decode comparison.*

### P9.1 — torch-MPS, honest scope: MNIST batch-1 inference
- **tier:** worker
- **depends_on:** P0.6
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 -b risc/cross-runtime 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target`
- **opens:** `proxima-onnx/scripts/torch_reference/inference_bench.py:29-32` (**only `--threads` and `--runs`; no device flag** — read verbatim this session), `model.py:1-30` (the model is **mnist.onnx**: 3× Conv+Relu, BatchNorm, Flatten, Gemm+Relu, Gemm, BatchNorm, LogSoftmax — **not** a transformer, **not** Q4_K), R0 (torch 2.13.0 with MPS available in that venv)
- **the honest scope, stated in the row before any number:** torch has no Q4_K kernel and no GGUF loader. **A torch-MPS arm on the openchat decode does not exist and cannot be built without changing what is being compared.** The comparable surface is the one the repo already models: mnist.onnx, batch 1, f32. That is a **cold-path** arm by the frequency bands (it is not the product's 80% case), and it is labelled `design-favors: incumbent` because torch-MPS is exactly what torch was built for.
- **the change:** `--device {cpu,mps}` added to `inference_bench.py`; a matching omega arm (mnist f32 through the Metal backend) added to `omega/benches/metal_vs_cpu.rs`, which R9 records as registered at `omega/Cargo.toml:207-210` and **never run**.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock proxima-onnx/scripts/torch_reference/venv/bin/python proxima-onnx/scripts/torch_reference/inference_bench.py --device mps --runs 200 2>&1 | tee /tmp/p91-torch-mps.log
  cd /Users/brianbruggeman/repos/slot-0/proxima && flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock proxima-onnx/scripts/torch_reference/venv/bin/python proxima-onnx/scripts/torch_reference/inference_bench.py --device cpu --threads 1 --runs 200 2>&1 | tee /tmp/p91-torch-cpu.log
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc10/target CARGO_TERM_COLOR=never flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock cargo bench -p omega --features metal,cpu --bench metal_vs_cpu 2>&1 | tee /tmp/p91-omega.log
  ```
- **expect:** 3 arms (torch-MPS, torch-CPU, ours-Metal) × p50/p95/p99/mean/CoV = **15 numbers; N==0 is RED**. Every row carries `design-favors:` and a frequency band.
- **predict:** *micro* → next rung *milli*: **torch-MPS is SLOWER than torch-CPU at batch 1 on mnist**, because the model is ~14 nodes of a few hundred KB and MPS dispatch overhead dominates. If that holds, the honest read is *"torch-MPS is not an incumbent on this shape; the arm exists to establish the cell, and the claim 'torch beats us on GPU' has no evidence at this shape in either direction."*
- **kill:** MPS unavailable in the venv (`torch.backends.mps.is_available()` false) → the cell cannot be built; record the feature gap, do not fabricate a number.
- **memory gate:** MG-2 (RSS ≤ 540 MB per P5.2's revised ceiling; the model is tiny, so a breach is a leak).
- **rollback:** revert the `--device` argument.
- **blast:** one python script + one existing, unrun bench file.
- **observe:** p50/p95/p99/mean/CoV per arm; `torch.backends.mps.is_available()`.
- **reprove:** the three commands.
- **log row:** `## ROW <NEXT> -- the first torch-MPS cell: mnist batch-1, and what it does not tell us about Q4_K decode`

---

### P9.2 — ORT-CoreML, honest scope: BGE-small embedding
- **tier:** worker
- **depends_on:** P9.1
- **worktree:** `proxima-wt-risc10` / `risc/cross-runtime`
- **opens:** `scripts/onnx_reference/bench.py:96` (**`providers=["CPUExecutionProvider"]` hardcoded** — R14), `scripts/onnx_reference/export_model.py:1-12` (the model is **BAAI/bge-small-en-v1.5**, exported from cached safetensors, f32), R0 (onnxruntime **not installed** in the torch venv; ORT source checkout exists at `~/repos/others/onnxruntime`)
- **the honest scope:** ORT has no Q4_K GGUF path and CoreML EP has no int4 matvec. **There is no ORT arm on the openchat decode.** The comparable surface is BGE-small f32 embedding, which the repo already exports and benches on CPU. Frequency band: this is the **80% case for the BGE product**, and near-zero for the decode product — the bill-mover filter says a loss here gates the embedding claim, not the decode claim, and the row must say which.
- **the change:** `--provider {cpu,coreml}` in `bench.py`; install `onnxruntime` (which ships the CoreML EP on macOS) into a dedicated venv, **not** into `torch_reference/venv` (that venv's `requirements.txt` is a recorded artifact; polluting it breaks P9.1's re-prove).
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && python3 -m venv scripts/onnx_reference/venv && scripts/onnx_reference/venv/bin/pip install onnxruntime
  cd /Users/brianbruggeman/repos/slot-0/proxima && scripts/onnx_reference/venv/bin/python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
  cd /Users/brianbruggeman/repos/slot-0/proxima && flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock scripts/onnx_reference/venv/bin/python scripts/onnx_reference/bench.py --provider coreml 2>&1 | tee /tmp/p92-ort-coreml.log
  cd /Users/brianbruggeman/repos/slot-0/proxima && flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock scripts/onnx_reference/venv/bin/python scripts/onnx_reference/bench.py --provider cpu 2>&1 | tee /tmp/p92-ort-cpu.log
  ```
- **expect:** `get_available_providers()` contains `CoreMLExecutionProvider` (**absent is RED — the cell cannot be built and that is the finding**). 2 arms × p50/p95/p99/mean/CoV = **10 numbers; N==0 is RED**.
- **predict:** *micro* → next rung *milli*: CoreML EP **partitions** the BGE graph and falls back to CPU for unsupported nodes; the log will show the partition count. Pre-registered: **more than one partition**, i.e. the "GPU" arm is partly CPU, and any speedup must be attributed per-partition before it is called a GPU win.
- **kill:** CoreML EP unavailable, or the partition log unobtainable → record the feature gap, no number.
- **memory gate:** MG-2.
- **rollback:** remove the venv and the `--provider` argument.
- **blast:** one python script + one new venv (gitignored).
- **observe:** provider list; partition count; p50/p95/p99/mean/CoV.
- **reprove:** the four commands.
- **log row:** `## ROW <NEXT> -- the first ORT-CoreML cell: BGE-small, partitioned, and what "GPU" means when half the graph runs on CPU`

---

## Phase 10 — Close the board

### P10.1 — The scoreboard, the roofline row, the row numbers, the ai_docs closure
- **tier:** hands
- **depends_on:** P4.3, P5.2, P6.3 (or P6.4), P7.1, P8.2, P9.2
- **worktree:**
  ```
  git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11 -b risc/board 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc11/target`
- **opens:** `proxima-tensor/docs/discipline.md:18736` (ROW 233, the last row on main), `proxima-tensor/docs/rooflines.md:396-479` (GPU lane; ceiling marked DEBT), `:751` (summary table row), `:766-773` (the doc's own closing note that the GPU ratio "is not a gap-to-machine at all")
- **commands:** one final interleaved sweep of every landed arm against `-fa 0` and `-fa 1`, 3 runs each; then row-number assignment from main.
- **expect:** one board with, per cell: value, CoV, `design-favors`, frequency band, provenance tag (MEASURED/DERIVED/ASSUMED), and a roofline column that is a **measured** ceiling (P0.6) rather than DEBT. Row numbers assigned sequentially from **234** in land order (G10). Every ai_docs invariant from P0.7 has at least one evidence pointer.
- **predict:** *bench* — the composed stack's `gpu_exec` is **≤ 45.5 − 1.8 = 43.7 ms** (P4.2 + P7.1, DERIVED) and `step_wall` is **≤ 43.7 + 11.0 − 1.5 − 1.7 = 51.5 ms** (P5.2 + P6.2, DERIVED), giving **≤ 2.94x** against 17.52. **This is a composition prediction and therefore the weakest number in the document** — each component was measured alone and their sum is DERIVED, not MEASURED. It is stated as a prediction to be falsified, and it is notably the same 2.95x the parallel branch reached by a different route (R12), which is a convergence worth noting and not a confirmation.
- **kill:** the composed number is worse than the best single-card number → the cards interact, and the interaction is the next work item.
- **memory gate:** MG-3 on the composed build, all four clauses, with the cap at the landed `kv_capacity_tokens`.
- **rollback:** the board is a document; a wrong row is corrected in place with a dated note, never silently.
- **blast:** `discipline.md`, `rooflines.md`, `ai_docs/*.jsonl`.
- **observe:** every counter named in P0.5, plus the route histogram, plus the two cross-runtime cells.
- **reprove:** the sweep + `bash ai_docs/query.sh gpu-decode-perf`.
- **log row:** `## ROW <NEXT> -- the GPU board, composed: <ratio>x against -fa 0 and <ratio>x against -fa 1`

---

# V. Dependency graph

**Prose.** Phase 0 gates everything: no card measures anything until the mutex exists (P0.1), the two byte counters tell the truth (P0.2, P0.3), the device window exists (P0.4), the anchor is re-sealed with the `-fa 1` arm (P0.5), the roofline debt is paid (P0.6), and the lane is in ai_docs (P0.7). Phase 1 forks off P0.7 into three independent strands: the adjudication (P1.1) rules, and its ruling releases the two recovery cards (P1.2 mask-fma, P1.3 pair-dot + prune_dead + consumer index), which are independent of each other and both feed only Phase 4. Phase 2 also hangs off P0.7 and is independent of Phase 1: P2.1 goes RED, P2.2 makes it GREEN, P2.3 re-anchors. Phase 3 requires P2.3, because a route census over a plan that two backends disagree about is a census of nothing. Phase 4 requires P3.4 (so the bake-off is scored by route, not by substring) **and** both recovery cards. Phase 5 requires P4.3, so the device-residency cell is measured on top of the landed kernel body rather than under it. Phase 6 requires P5.2, per §II.3. P6.4 exists only if P6.3 kills bucketing. Phase 7 requires P6.3 or P6.4. Phase 8 requires P7.1 and is last by design. Phase 9 hangs off P0.6 alone and runs in parallel with everything from Phase 1 onward — it shares no source with the decode lane. Phase 10 requires the terminal card of every landing phase.

**Picture.**

```
P0.1 flock
  └─ P0.2 operand_bytes
       └─ P0.3 block_upload_bytes
            └─ P0.4 device gpu_exec window
                 ├─ P0.5 re-seal + -fa 1 ──┐
                 └─ P0.6 roofline ─────────┤
                                           ├─ P0.7 ai_docs
                                           │      ├──────────────────────┐
                                           │      │                      │
                                           │   P1.1 adjudicate        P2.1 plan-identity RED
                                           │      ├─ P1.2 mask-fma       └─ P2.2 bind owns layout
                                           │      └─ P1.3 pair-dot          └─ P2.3 re-anchor
                                           │           +prune_dead               └─ P3.1 Route value
                                           │           +consumer index                └─ P3.2 census, lock-free
                                           │              │                                └─ P3.3 wgsl/cuda coverage
                                           │              │                                     └─ P3.4 census cell
                                           │              └──────────────┬──────────────────────────┘
                                           │                             ▼
                                           │                        P4.1 selector
                                           │                          └─ P4.2 bake-off
                                           │                               └─ P4.3 land one
                                           │                                    └─ P5.1 host spans
                                           │                                         └─ P5.2 KV arena
                                           │                                              └─ P6.1 bucket+mask
                                           │                                                   └─ P6.2 invert assertion
                                           │                                                        └─ P6.3 trade cell
                                           │                                                             ├─(kill)─ P6.4 invariant plan
                                           │                                                             └─ P7.1 wide reduce
                                           │                                                                  └─ P8.1 injectivity
                                           │                                                                       └─ P8.2 single-range
                                           └─ P9.1 torch-MPS ─ P9.2 ORT-CoreML                                          │
                                                    │                  │                                                │
                                                    └──────────────────┴────────────────────────────────────────────────┴─ P10.1 board
```

Every id in a `depends_on` appears as a node above. No cycles: the graph is a DAG rooted at P0.1 with a single sink at P10.1.

---

# VI. Rollback map

| card | rollback mechanism | cost | what goes RED on rollback | what must NOT be rolled back |
|---|---|---|---|---|
| P0.1 | `rm` the lock file | none | nothing | — |
| P0.2 | `git revert` | 1 commit | every GB/s row on the branch is struck | — |
| P0.3 | `git revert` | 1 commit | the copied-vs-bound split | — |
| P0.4 | `git revert` | 1 commit | `gpu_device_ms` | — |
| P0.5 | none (measurement) | — | — | — |
| P0.6 | `git revert` | 1 commit | `rooflines.md` returns to DEBT | — |
| P0.7 | revert 3 JSONL blocks | none | `query.sh gpu-decode-perf` | — |
| P1.1 | ruling; reversal needs new evidence | — | — | — |
| P1.2 / P1.3 | `worktree remove` + `branch -D` / per-commit revert | 1–3 commits | nothing downstream until P4 | — |
| P2.1 | delete the test | none | the falsifier for D1 | — |
| P2.2 | `git revert` | 1 commit, **3 crates** | P2.1 returns to RED — the intended signal | — |
| P3.1–P3.3 | `git revert` | 1 commit each | route coverage matrix | — |
| P3.4 | none (measurement) | — | — | — |
| P4.1 | revert build.rs + toml hunks | 1 commit | both bodies unselectable | — |
| P4.2 / P4.3 | flip `[q4k] body` | **one toml line** | the winner's delta | the loser's recorded row |
| P5.1 | `git revert` | 1 commit | multi-slot spans | — |
| P5.2 | `git revert` | 1 commit | KV residency | **the build-time 34 GB assertion — a correctness guard, §15** |
| P6.1 / P6.2 | feature off | **one flag** | bucketing + the inverted assertion | the `#[cfg(not(...))]` arm keeps asserting `plan_hits == 0` |
| P6.3 | none (measurement) | — | — | the recorded loss, if it lost |
| P6.4 | feature off | one flag | the invariant plan | — |
| P7.1 | `[cooperative_reduce] max_threads = 32` | **one toml line** | the wide tree | — |
| P8.1 | `git revert` | 1 commit, **cross-backend** | injectivity; every scatter rejected as today | — |
| P8.2 | feature off | one flag | single-range attention | — |
| P9.1 / P9.2 | revert the script arg; remove the venv | none | the two cross-runtime cells | the recorded feature gaps |
| P10.1 | dated correction in place | none | — | **no row is ever silently deleted** |

**Rollback ordering rule.** Rolling back a card requires rolling back everything downstream of it in the §V graph first, in reverse topological order. The one exception is P4.2/P4.3, whose rollback is a config value and therefore leaves the graph intact.

---

# VII. Abandoned designs, each with the constraint that ruled it out

1. **`BoundOpKind::CachedAttention` — a fifth bound kind for one model's attention shape.** *Ruled out by:* AGENTS.md "problem solving" (*"we should not be adding arbitrary rules/code for specific instances"*) **and** guiding-principles §1 (the expression exists: two `Reduce`s with an online-softmax combine, `spec.rs:2596-2720`) **and** its own measurement (R12: 616 vs 1194 dispatches, 51.535 vs 51.571 ms wall — the premise is unachieved). Their own `failure-cached-attention-matcher.md` records the first matcher abandoned as "a heuristic" that "cannot prove the semantic roles"; the second is a structural matcher for the same non-generic macro-op. Adjudicated in P1.1.

2. **An atomics-based GPU scatter, to make `run_reduce_scatter` portable.** *Ruled out by:* §21 / AGENTS.md hot-path lock-free, **and** by `map.rs:110-131`'s own convention — the CPU needs no atomics only because its loop is sequential; a GPU version would change the semantics of colliding writes. §II.1's structural proof gets the one case that matters (the affine KV append) without any atomic, and names the fallback for every other case.

3. **A `Mutex<BTreeMap>` route census mirroring `WIDTH_TILE_DECLINE` (`instrument.rs:842`).** *Ruled out by:* §21 lock-free-first **and** the per-dispatch cost bound (R13: `encode_dispatch` 0.47 ms / 1196 = 393 ns/dispatch; a lock acquisition per dispatch is a material fraction of that). Replaced by a plan-owned `Vec<Route>` and atomic totals — the census is free because the route is a property of the plan.

4. **Two mutually-exclusive cargo features for the two Q4_K bodies, guarded by `compile_error!`.** *Ruled out by:* `scripts/omega-gate.sh` step **[2/6]** `cargo build -p omega --all-targets --all-features` — the crate's own gate would go red. Replaced by a build-time profile axis (§8's *profile input* half), which is exclusive **and** `--all-features`-clean.

5. **Relaxing the strict `InputSizeMismatch` check (`metal.rs:991-1000`) so an over-allocated KV buffer passes.** *Ruled out by:* §15 — a workaround left in place of a repair. The capacity buffer is registered as a **host span** instead, and the block handed over is still exactly `cached_len` elements, so the check is preserved untouched. This is the constraint that produced §II.3's ordering.

6. **A fusion engine (rms_norm + mul, scale + mask + softmax) as the route to parity.** *Ruled out by:* R8 — `grep -rln fuse ggml/src` is **empty** at the incumbent's own checkout b25346221, and `rms_norm` + `mul` dispatch as two kernels there. Parity is reachable without fusion; fusion is upside **past** parity and does not belong in a parity plan.

7. **Threading `op_setup`; nsg=2 threadgroup regrouping; rematerializing all ≤2-consumer nodes; blaming per-dispatch fixed cost.** *Ruled out by:* R4's dead-lever list (nsg=2 is now a **fourth**-time negative across both lanes, R4 + R12 ROW 267) and by the type system in the threading case (non-`Send` `MTLBuffer`, R3 M9). Not re-proposed anywhere in this plan.

8. **Making `BoundOp.extents` symbolic up front (`Vec<Extent>` instead of `Vec<u64>`).** *Ruled out for now by:* blast radius — `grid_threads` (`msl.rs:1517`), `kernel_cache_key` (`:731`), and `kernel_dispatch_shape` (`:797`) all read extents, and R11 M6' names it "a bigger change". **Parked with a named un-park condition:** P6.3 killing bucketing at every bucket size. Parked as a measured trade-off, in the log with the number (disciplined-component's parking limits), never deleted.

9. **Grepping the emitted Q4_K region to break a bake-off tie.** *Ruled out by:* R16 — `Q4K_UNPACK_MSL`/`Q5K`/`Q6K` are concatenated with no delimiter at `msl.rs:1978-1982`, so "the Q4_K region" has no decidable boundary. Replaced by the four-step chain in §II.6, whose only source-based step is **total** emitted byte length, which needs no region parsing and is deterministic by `emit_is_deterministic_byte_equal` (`msl.rs:4656`).

10. **Sizing the KV arena from `ServingConfig::context_length`.** *Ruled out by:* the arithmetic — 131_072 × 262_144 = **34.36 GB** (`serving.rs:161` × R13's measured slope). Replaced by a build-time `kv_capacity_tokens` with a build-time byte assertion, making the trap a compile error.

---

# VIII. Open questions, each resolved by a named measurement

| # | Open question | Resolved by | The measurement that decides it | Pre-registered answer | What a miss means |
|---|---|---|---|---|---|
| 1 | Is `-fa 0` the incumbent's real home turf, or is 3.88x an understatement? | **P0.5** | `llama-bench … -fa 1` mean t/s vs `-fa 0`, interleaved, 3 runs | `-fa 1` is faster; the true gap is **worse** than 3.88x and every later ratio is quoted against it | If `-fa 1` is slower, `-fa 0` is genuinely their design point and R13's ratio stands unamended |
| 2 | Is 228.9 GB/s the machine's ceiling or merely the incumbent's achieved rate? | **P0.6** | copy-arm `read_plus_write` GB/s, device window, 21 runs | copy arm **exceeds 228.9** → the incumbent has headroom too, and our roofline is a real ceiling for the first time | Copy arm below 228.9 → the reduce probe never measured bandwidth; 228.9 becomes the ceiling by default and `rooflines.md:766-773`'s own caveat stands |
| 3 | Do R13's substring-derived buckets (225/385/547) survive a real route census? | **P3.4** | route histogram vs R13's `classify_kind` buckets | exact match | A mismatch means `classify_kind` mislabelled on main as it did on the parallel branch (R12 ROW 263), and **every bucket number in R13 is restated against routes** |
| 4 | Which Q4_K body wins, and is it the same body at every shape? | **P4.2** | interleaved A/B at micro and milli rungs; §II.6's 4-step chain | one body wins **both** the `ffn_up` and `attn_q` shapes | A shape split (one wins ffn, the other wins attn) makes the body a per-route selection, not a global one — a new card, not a tie-break |
| 5 | Does the census cost anything per dispatch? | **P3.2** | `encode_dispatch` mean vs R13's 0.47 ms | Δ = 0 | > 0.4935 ms → the census leaked into the encode loop |
| 6 | Does KV device-residency move `gpu_exec` at all? | **P5.2** | `gpu_exec_ms` before/after, CoV band | unchanged; only `block_upload` moves (2.0 → ≤ 0.5 ms) | A `gpu_exec` move means the card changed GPU work, which it was not designed to do |
| 7 | Does the tail-mask `Select` fuse into the existing `ComposedBody`? | **P6.1** | `op_count` delta | **+2** | **+34** → the fusion assumption is refuted; the fix is a bind-level composition question |
| 8 | Does bucketing pay at short context, and where is the optimum bucket? | **P6.3** | `Δ(kv_cache.* families gpu_ms)` vs `Δ(prepare + op_setup ms)` at buckets 8/32/256 | net **loss** at 256, net **win ~1.5 ms** at 8; optimum near `prompt_tokens + PROXIMA_MAX_TOKENS` | Loss at **every** bucket → bucketing is dead as a landing route and **P6.4 (shape-invariant plan) is unparked** |
| 9 | Does R7's uncommitted −20% wide-reduce survive a 9-commit rebase? | **P7.1** | cooperative bucket 9.113 ms → ? | ≤ 7.3 ms (−20%) | < 10% improvement → R7's number did not survive; record the negative and the bucket returns to open |
| 10 | Does halving the dispatch count buy wall-clock on **our** stack, as opposed to theirs? | **P8.2** | `step_wall` at 939 ops vs at 1196 | **< 1 ms change**, matching R12's independent refutation | A large win would contradict R12's control and both cells must be re-run before either is believed |
| 11 | Is the affine-scatter prover sound? | **P8.1** | the REJECT half of the case table | every REJECT case rejected | A false ACCEPT is a GPU race — a correctness defect, not a perf miss, and kills the card |
| 12 | Is torch-MPS an incumbent at any shape this repo measures? | **P9.1** | torch-MPS vs torch-CPU, mnist batch 1, p50/p95/p99 | **MPS is slower at batch 1**; the "torch beats us on GPU" claim has no evidence at this shape either way | MPS faster → a real incumbent arm exists and the omega mnist Metal arm must be compared against it |
| 13 | Does ORT-CoreML actually run BGE-small on the GPU? | **P9.2** | CoreML EP partition count from the ORT log | **more than one partition** — the "GPU" arm is partly CPU | One partition → a genuine whole-graph GPU arm and a real incumbent for the embedding lane |
| 14 | Does the composed stack equal the sum of its measured parts? | **P10.1** | composed `step_wall`/`gpu_exec` vs the DERIVED sum ≤ 51.5 ms / ≤ 43.7 ms | within CoV of the sum | Worse than the best single card → the cards interact, and the interaction is the next work item, decomposed into inconsistency vs understanding-gap |

---

### Critical Files for Implementation
- /Users/brianbruggeman/repos/slot-0/proxima/omega/src/metal.rs
- /Users/brianbruggeman/repos/slot-0/proxima/omega/src/msl.rs
- /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/bind.rs
- /Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/generate.rs
- /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/spec.rs