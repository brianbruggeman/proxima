# P2 v3: fused attention under the llama.cpp oracle, gated by decode, inside a whole-layer dispatch budget

status: DESIGN, not audited. Supersedes `design-p2-fused-attention-v2.md` (NOT ADMITTED, critic 2026-09-29). Owner decision of 2026-09-29
(TASKS.md "P2 decision", ANSWERED) binds: the llama.cpp oracle governs fused attention; bit-equality to unfused Metal is dropped; speculative
R1 (speculative output byte-identical to non-speculative) still binds; llama.cpp is an external oracle, never a runtime or dependency.
author: proxima-architect pass, 2026-09-29. Read-only design. Writes: this file, and `E/` (C and awk; two programs link the external f1ea20621 ggml CPU build on synthetic bytes for well under a second; no cargo, no GPU, no Ollama).

## 0. Conventions and what was run

- `T:` = `/private/tmp/long_ctx_main2` (non-git export: main 73fd1cbd plus this spec's work). `S:` = `/Users/brianbruggeman/repos/slot-0/proxima` (spec files only). `L:` = `/Users/brianbruggeman/repos/others/llama.cpp` at f1ea20621 (`git log -1`: f1ea20621, 2026-09-28 19:52 +0200). `M:` = a memory note under `~/.claude/projects/-Users-brianbruggeman-repos-slot-0/memory/`. `E/` = `<session>/long_ctx_backups/p2_experiments/v3/`. `PEER` = `/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/52ba967c-be7e-4dee-829f-ab7524193cb3/scratchpad/llama-server-build/bin` (the f1ea20621 CMake build).
- Tags: [R] read from source this session; [M] measured this session (file in `E/`); [D] derived from other numbers, never a mechanism basis; [A] assumed, must be measured before it is acted on; [P] recorded by another session or by the coordinator, read from a note or message, not re-measured here.
- Every count in a gate below is a count the command prints; N == 0 is RED.
- Run this session (all under 1.5 s, CPU): `E/flash_sim` (0.84 s), `E/support_exact` (1.53 s, over the sub-second limit; disclosed, a re-run of v2's program whose four rows equal v2's), `E/fill_rule.awk`, `E/partition_rule.awk`, `E/chunk_cap.awk`, `E/exact.awk`, `E/llama_worked_example_peer`, `E/llama_oracle_demo_peer` (0.44 s), and `llama-tokenize-f1ea20621` on 3 fixtures. Not run: cargo, Metal, Ollama, llama on a model (a peer holds the GPU window; a model decode on CPU is minutes).

## 1. Answer map (every v2 blocker and major, and the four owner constraints)

| v2 finding | v3 answer | section |
|---|---|---|
| B1 irreversible steps before evidence; regression 23.57 to 33.17 ms/tok; Hkv=1 fill undefined | Candidate B and every legacy arm stay. Flash is a reversible, all-or-nothing route behind the existing `flash_attention` bit. Deletion of Candidate B is the last slice, conditional on a THRESHOLD decode gate (fused ms/token <= B ms/token at 2K and 32K, plus the recorded-regression context 128). Split-K rule defined; fill threshold derived from GPU cores (IORegistry) and occupancy (pipeline footprint) | 4, 7, 8 |
| B2 oracle does not see what the change can break; unrun G2; prebuilt tools predate gemma4 | Gemma4-capable f1ea20621 build located and exercised (`llama-tokenize-f1ea20621`, 3 of 3 fixtures byte-equal [M]); capture tool is slice 1; tolerance comes from llama's own CPU-vs-Metal spread, kernel-level on llama's own q/k/v bytes; what it cannot see is stated | 6 |
| B3 R1 evidence tautological; key at `T:omega/src/identity.rs:524-595` bakes `new_upper`/`context_chunks`; T2 with zero accepted drafts; C=32 missing | Row-count-free key specified in that function. R1 tests require accepted > 0 and rejected > 0, compare every verify row's logits to the decode row, include C=32 and the qwen3.6 pass-plane shape. The simulation takes rows, Br, Sk and P as inputs and fails four controls on purpose | 5 |
| M1 "28/49 rows differ" used chunk cap 4; real full-layer cap is 1 | Corrected: cap(G=8, hd=512)=1; the 28/49 is the qwen3-8b (cap 4) case. Control N1 measured 0 / 12 / 28 for caps 1 / 3 / 4 | 5.3 |
| M2 owner bar retired unannounced | Surfaced: TASKS.md P2 decision names it; v3 states the retired bar and what still binds | 7.4 |
| M3 qwen3.6, lfm2, dense ungated; parity tests not ported; WGSL/CUDA matches | Per-family table: each family stays on its current path or has a named gate; parity tests are kept (legacy arms stay) and listed with their fixtures; WGSL/CUDA/wgpu matches enumerated at file:line | 7.2, 7.5 |
| M4 grid2d branch returns before any width check | Typed width check on both branches of `dispatch`, plus at pipeline creation | 3.5 |
| M5 Ollama may not be llama.cpp | Ollama is the performance incumbent only; no correctness gate uses it except where it is the positive control of a harness (needles), labelled | 6.1 |
| M6 SPEC drift: R12L/AC12L, AC23 vs ABSENT list | Reconciled; AC12L-a/b as written point at a binary that predates gemma4 (found); R18 is not retired, the only deletions are 3 named tests behind an owner-signed amendment | 10 |
| M7 NPV/Ls register feasibility, merge cost, Safe cost (matvec 179 vs 241 GB/s) | Math mode is a measured rule, not a pin; pinned arithmetic lives in noinline functions; merge cost and register footprint are read from the compiled pipeline and measured | 3.3, 8 |
| M8 slices not minimal | 11 attention slices (v2 had 18), each one behaviour, one command, one count | 11 |
| owner: whole layer | Dispatch budget per layer and per token, per glue class, primitive named, tiers, order by dispatches removed per unit of work | 9 |
| coordinator: 2 s GPU recovery | Per-dispatch duration bound derived and enforced by the partition rule; measured in slice 8 | 4.3 |

## 2. Decision, and the one contested call

Shape. One op, `BoundOpKind::CachedAttention`, gains one field, `fold: AttentionFold { Legacy, Flash }`. `Legacy` is every arm that exists today (sequential, chunked, block-staged, split-plus-merge) and Candidate B (`CachedSoftmaxWeights`); `Flash` is one new MSL lowering: FlashAttention-2 online-softmax tiles, Q-tiled through threadgroup memory, arithmetic defined over absolute key index only (section 3). The recognizer emits `Flash` only when the plan's `flash_attention` bit is set and the anchor is a full-attention gemma4 layer; otherwise it emits what it emits today, byte for byte. The bit is all-or-nothing per model instance: decode, verify and prefill of a given model all take the same route, so speculative R1 never compares two arithmetics.

The contested decision: v2's "one lowering replaces the three legacy arms and Candidate B" against "a second lowering that coexists behind a demand bit until a measured gate deletes the loser". v3 takes the second. Forced by invariant 1 (nothing irreversible before evidence) and by the recorded regression of the previous kernel of this family, whose cause is unattributed (M: `project_gemma4_attn_fusion_recognizer_gap`: "Attribution NOT decomposed"). The price is two implementations for the length of the gate; the abandoned alternative is in section 12.

What the constraints changed. no_std / alloc-free: the partition rule is a pure integer function with no allocation, so it is testable on CPU and portable to a poll-mode source; it replaced an idea to pick partitions inside the encoder from the compiled pipeline (abandoned: the pipeline does not exist when `emit` runs). Reuse first: no new op kind (the bool would have been a `NumericPolicy` field, which changes every family); `AttentionFold` is the two-state datum that must survive from recognizer to emitter. Teaching surface: every constant that changes bits is named "arithmetic" and appears in the pipeline key; everything else is "non-arithmetic" and must not.

## 3. The arithmetic contract (kept from v2, with changes marked)

Notation as v2 section 3.2: query vector `q`, scale `s`, absolute key index `j` (cached live rows first, then new rows), `visible(row, j)`, tile `Bk = 32`, block `Ls` keys (multiple of 32), `NPV` interleaved value chains. Every multiply-add is an explicit `fma`; a masked key is selected away, never multiplied by zero.

1. Score: lane partials `fma` over `d = l, l+32, ...` ascending on the row `[even | odd | pass]`, xor butterfly 16, 8, 4, 2, 1, then `* s`, else `-inf` by select.
2. Tile `t = floor(j/32)`: `m' = max(m, m_t)`; `alpha = (m == -inf) ? 0 : exp(m - m')`; `p_j = (score_j == -inf) ? 0 : exp(score_j - m')`; `l = fma(l, alpha, butterfly(p))`.
3. Value fold per output dim: `NPV` chains over the tile's visible keys in ascending `j` (chain `j mod NPV`), pairwise combine `w = NPV/2 .. 1`, `o = fma(o, alpha, part[0])`.
4. Block `b = floor(j/Ls)` folds its tiles from the identity `(m=-inf, l=0, o=0)`.
5. Blocks merge strictly left to right from the identity: `m = max`, `wA`, `wB` from `exp`, `l = fma(lA, wA, lB*wB)`, `o = fma(oA, wA, oB*wB)`. The identity is an exact two-sided neutral.
6. Output `o / l`, `0` when `l == 0`.

Arithmetic (changes bits, appears in the key and the fixtures): `Bk`, `Ls`, `NPV`, the lane-to-dim map, the butterfly order, the merge expression, the math mode. Non-arithmetic (free to vary, output bits unchanged): the row tile `Br`, the staging width `Sk`, simdgroups, the number of threadgroups per block run `P`, the number of chained dispatches `Dk` (section 4), and resident versus split mode.

### 3.1 What v3 verified about that contract (sim `E/flash_sim.c`; scalar f32, `-ffp-contract=off`; not a Metal measurement)

The simulated kernel takes `(cached, rows, query_offset, Br, Sk, P, NPV, Ls)` as inputs and loops over simulated threadgroups (row tile x kv head x partition); K/V are read through an accessor that splits cached and new buffers at `cached`, returns NaN rows for padding, and stages `Sk` keys at a time. Reference for row `i`: the 1-row dispatch with `cached = base + i`. [M] `E/flash_sim.out`:

| check | shapes | result |
|---|---|---|
| `TOTAL dispatch_configs=60 rows_checked=1224` | gemma4 full hd512 (G=2 stand-in), qwen3-8b hd128 G4 Hkv8, qwen3.6 pass-plane hd256 G8 Hkv2; rows {1,3,17,32,49}; (Br,Sk,P) in {(1,32,1),(2,16,2),(4,8,3),(2,32,5)}; base 100 (cached/new boundary mid-tile, block boundary inside the verify rows) | `mismatching_rows=0` |
| `CHAIN links=1,2,3,7` (duration chain, section 4.3) | gemma4 full hd512, rows 49 | `mismatching_rows=0` each |
| control N1 legacy extent-keyed chunks, cap 1 / 3 / 4 | hd128 G4, base 20, rows 49 | `0 / 12 / 28` |
| control N2 block size differs between dispatches (Ls 32 vs 64) | base 100, rows 49 | `49` |
| control N3 each threadgroup merges its own blocks first (P=3 vs P=1) | base 100, rows 49 | `49` |
| control N4 tile grid anchored at `cached_len`, not absolute index | base 100, rows 49 | `47` |

The four controls fail on purpose. If the harness were insensitive they would print 0. What the sim cannot show: that the Metal compiler preserves this order (section 3.3), the qwen3.6 `G` (8, from `omega/src/msl/tests.rs:4771-4778` which asserts `effective_context_chunk_cap(8, 256) == 3` for the "qwen35 shape" [R]), and any timing.

### 3.2 Choosing `Ls` and `NPV`

`Ls` starting value 128, derived: M2 partials cost `2 * G * (hd + 2) * 4` bytes of scratch write plus read per block per kv head, against `Ls * 2 * hd * 4` bytes of K/V streamed. For hd 512, G 8 that is 32,896 B against `Ls * 4096` B: 6.3 percent at `Ls = 128`, 12.5 percent at 64, 25 percent at 32 [D, `E/fill_rule.out`]. A 10 percent scratch overhead needs `Ls >= 80`, so 128 is the smallest power-of-two block that meets it. `NPV = 4` is a starting value. Both are swept on the real payload in slice 5 over `NPV in {2,4,8}`, `Ls in {64,128,256}` (9 cells) and the decode gate may move `Ls` only through the follow-up rule of section 8, which re-runs slices 4, 5 and 6.

The oracle cannot choose `NPV`: v2 measured (against the Sep-6 llama libraries) that `NMSE(llama, x)` moves 5 percent across `NPV` 1..8 while the distance to f64 moves 43 percent [P: v2 4.4]. The selection therefore stays with the internal f64 check, LABELLED INTERNAL: among (`NPV`, `Ls`) that pass the llama gate, take the smallest `NPV` whose f64 rms error lies inside the K-order control region against f64. Acceptance stays with the oracle; selection is internal.

Register footprint [D, unmeasured]: per lane at hd 512, `q` 16, `o_run` 16, `o_blk` 16, value partials `NPV * S_d` with the dim-slab width `S_d = 4` (16 at NPV 4), tile scores and weights about 4: about 72 plus temporaries. The kernel carries `[[max_total_threads_per_threadgroup(512)]]`; `pipeline_for` compares the compiled `maxTotalThreadsPerThreadgroup()` with the declared width and returns a typed error (3.5). The number of registers is read as that pipeline value in slice 4, not assumed.

### 3.3 Math mode is a measured rule, and the pinned arithmetic is function-level

v2 pinned `MathMode::Safe` for the whole flash pipeline. The recorded cost of Safe on the matvec is 179 GB/s against 241 to 247 GB/s Relaxed, and 33.82 against 28.19 ms/token on a whole decode program [P: M `project_gpu_one_risc_plan_2026_09_03`, ROW 296/297]; the shipped default is Relaxed (`T:omega/src/metal/pipeline_buffers_upload.rs:44-45`, `T:proxima-model-interop/src/serving.rs:991`). v3 therefore does not pin.

Pinning is done where it can be enforced: the score, the tile fold, the block fold and the merge are four `noinline` device functions that both the resident kernel and the split/merge kernels call (the method that reproduced the unfused kernels bit-exactly in the earlier staged-replay work [P: M `project_gemma4_attn_fusion_recognizer_gap`, "verbatim per-node transcription in noinline functions is the method that works"]). Decode, verify and prefill share the same compiled binary (section 5), so R1 does not depend on which math mode is chosen; resident versus split equality does depend on the compiler not re-treeing across the shared functions, and that is tested, not assumed (slice 4: `flash_split_equals_resident`, `flash_chain_equals_single`).

Rule, stated before measurement: compile Safe until slice 8 records both modes. Adopt Relaxed for the flash pipelines iff (a) every slice-4 equality test and the slice-6 R1 test pass under Relaxed and (b) Relaxed decode ms/token beats Safe by more than 2 sigma of the per-pair ratio. Otherwise Safe. The mode is already part of every pipeline cache key (`MathMode::cache_token`, `T:omega/src/metal/pipeline_buffers_upload.rs:75-81`).

### 3.4 Tile derivation (v2 3.4, unchanged)

Budget `T = 32768` B (`T:omega/omega-runtime.toml`, `[cached_attention] threadgroup_memory_bytes`), `Sk = min(32, pow2_floor(T / (s * hd * e)))` with `s` threadgroups per core targeted. gemma4 full (hd 512, f32): `Sk = 16` at `s = 1`, 8 at `s = 2`, stage 16,384 B at `s = 2` [D]. `Br = simdgroups / G`: decode `Br = 1` (8 simdgroups, 256 threads), prefill and verify `Br = 2` (16 simdgroups, 512 threads); `Rq = Br * G` query vectors share every staged K/V tile.

### 3.5 Threadgroup width: both branches of `dispatch` get the typed check

`T:omega/src/metal/resident_nocopy_cache.rs:1318-1330` (the `grid2d` branch) issues `dispatchThreadgroups` and returns before any width check; `:1334-1337` clamps a REQUIRED width with `.min(max_threadgroup)` while the doc at `:1305-1309` says a required width is honoured exactly [R]. The flash launch uses `grid2d` (x = row tiles, y = kv head, z = `P` via `grid.depth`), so the check that v2 put only on the clamp would not have covered it. v3:

```rust
pub(super) fn dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    entry: &str,
    grid: GridSpec,
) -> Result<(), MetalError>;
// MetalError::ThreadgroupWidthExceedsPipelineLimit { entry: String, required: u64, limit: u64 }
```

`required` is `threads_per_threadgroup_x * threads_per_threadgroup_y` on the `grid2d` branch and `threadgroup_width` on the width branch; `None` (grid-derived width, an occupancy hint) keeps its `min`. `pipeline_for` runs the same comparison right after `compile_pipeline`, so the error names the entry once at creation, and the footprint that `T:omega/src/metal/pipeline_buffers_upload.rs:452-455` reads only under `instrument` becomes an unconditional cold-path read at that point. Call sites: `T:omega/src/metal/device_buffers_arena_plan.rs:1667`, `T:omega/src/metal/arena_encode_dispatch_finish.rs:1566,1622` (three). The capture copy at `arena_encode_dispatch_finish.rs:751-755` is a dump path and keeps its own clamp. This repairs every existing cooperative kernel with a required width, not only the flash kernel.

## 4. The partition rule: fill for Hkv=1 decode, and a per-dispatch duration bound

gemma4 has `Hkv = 1`. A decode dispatch (`rows = 1`) has `ceil(1/Br) * Hkv = 1` threadgroup unless keys are split. The recorded regression of the previous fused kernel (wall 23.57 to 33.17 ms/tok, gpu_exec 17.68 to 27.24) shipped exactly that shape: one threadgroup per layer, 256 threads, looping keys and dims, against 2048 threadgroups in the unfused AV [P: M `project_gemma4_attn_fusion_recognizer_gap`]. The note records lost parallelism as "the candidate mechanism ... unmeasured shares for barriers, scratch traffic, compiler effects"; it is not an established cause, so v3 does not claim it is one. The rule below removes the shape; the gate in section 8 measures whether removing it was enough.

### 4.1 Where the inputs come from

- GPU core count. Metal exposes none: `grep -il 'corecount|core_count|gpucore|numberOfCores'` over the 97 headers of `Metal.framework/Headers` in the active SDK matches 0 files [M], and `grep -rn 'gpu-core-count'` over `T:omega` and `T:proxima-model-interop` matches nothing [M]. The value is in the IORegistry: `ioreg -c AGXAccelerator -r -d1` prints `"gpu-core-count" = 32` and class `AGXAcceleratorG13X` on this M1 Max [M]; `system_profiler SPDisplaysDataType` prints `Total Number of Cores: 32` [M]. The read is `IOServiceMatching("AGXAccelerator")` then `IORegistryEntryCreateCFProperty("gpu-core-count")`, three `extern "C"` declarations, once at `Plan` construction, kept as a `u32` field of `Plan` and passed to `flash_split` as a plain argument (a `DeviceFacts` wrapper around one integer was considered and dropped: the two call sites are identical lines). If the read fails the emitter falls back to the build-time constant `sized::FLASH_GPU_CORES_ASSUMED` (a failed read is a typed `debug!`, not a silent default).
- Occupancy. Per pipeline: `maxTotalThreadsPerThreadgroup`, `threadExecutionWidth`, `staticThreadgroupMemoryLength` (`MTLComputePipeline.h`), and per device `maxThreadgroupMemoryLength` (`MTLDevice.h`); all four are already read at `T:omega/src/metal/pipeline_buffers_upload.rs:452-455` under `instrument` and become unconditional. Resident threadgroups per core is `R = min(floor(pool / tg_mem), floor(slots / tg_threads))`; the per-core pool and slot count are not exposed by Metal, so `R` is a build-time constant `sized::FLASH_RESIDENT_TGS_PER_CORE`, default 2 [A], calibrated by a probe in slice 8 (a memory-bound kernel with the flash kernel's threadgroup memory and width, threadgroup count swept from 1 to 512; `R` = the smallest count reaching 95 percent of the plateau, divided by cores).
- Fill threshold `F = gpu_cores * R` = 64 on this machine at `R = 2` [D].

### 4.2 The rule (pure integer function, no allocation, `omega::msl`, testable on CPU)

```rust
pub struct FlashSplit { pub tgs_per_block_run: u32, pub links: u32, pub keys_per_link: u64, pub partial_scratch_bytes: u64 }

pub const fn flash_split(shape: FlashShape, gpu_cores: u32, caps: FlashCaps) -> FlashSplit;
// FlashShape { rows, kv_heads, groups, head_dim, live_key_capacity, elem_bytes, br }
// FlashCaps  { block_keys, resident_tgs_per_core, scratch_budget_bytes, dispatch_budget_ns, bandwidth_floor_bytes_per_s }; `FlashCaps::SIZED` is built from the `sized::FLASH_*` constants, and the record exists so a test can sweep the constants that differ per target (principle 12): a test call site passes other caps, which a bare constant read would not allow
```

1. `base = ceil(rows / br) * kv_heads`; `nblk = ceil(live_key_capacity / block_keys)`.
2. Fill: `p_fill = clamp(ceil(F / base), 1, nblk)`. Partial scratch for `p_fill > 1` is `rows * groups * kv_heads * nblk * (head_dim + 2) * 4` bytes. `P = p_fill` if that fits `scratch_budget_bytes`, else `P = 1` (resident mode). The budget is `[arena] transient_cap / 8` = 21,601,515 B (`T:omega/omega-runtime.toml:176`, divisor 8 a build constant).
3. Duration: `stream_bytes = live_key_capacity * ceil(rows / br) * 2 * head_dim * elem_bytes` (bytes all threadgroups read for one dispatch). `links = ceil(stream_bytes / (dispatch_budget * bandwidth_floor))`; `keys_per_link = ceil_to_block(ceil(cap / links))`. Defaults: budget 0.5 s, floor 50 GB/s [A: a third of the 150 GB/s the in-situ matvec reaches, `M: project_gemma4_decode_dispatch_type`], so a link may stream at most 25 GB.
4. `links > 1` is a chain of dispatches; link `k` folds its run of blocks into the running state `(m, l, o)` saved in scratch by link `k-1` (`rows * groups * (head_dim + 2) * 4` bytes, 8.4 MB at C=512). Restoring a saved f32 state is exact, and the fold order is the resident order, so the chain is arithmetic-neutral; [M] `CHAIN links=1,2,3,7 mismatching_rows=0`. Only `P = 1` chains (a large-`rows` dispatch); `P > 1` dispatches stream at most a few GB.

Everything in `FlashSplit` is non-arithmetic: `P` and `links` change the grid and the scratch, never a bit (controls N2 and N3 show what an arithmetic partition would do). The grid is built after `emit` by patching `grid.depth` (z = `P`) and issuing `links` dispatches; the pipeline key never contains `P` or `links`.

### 4.3 What the rule produces (gemma4 full layer, hd 512, G 8, Hkv 1, f32, `Br = 2` for rows > 1) [D from `E/partition_rule.out`]

| rows | live keys | base | `P` | `links` | stream per dispatch | time per dispatch at 400 / 150 / 50 GB/s | bounded by |
|---|---|---|---|---|---|---|---|
| 1 (decode) | 2048 | 1 | 16 | 1 | 0.01 GB | 0.02 / 0.06 / 0.17 ms | nblk = 16 < F: under-filled, see below |
| 1 | 32768 | 1 | 64 | 1 | 0.13 GB | 0.3 / 0.9 / 2.7 ms | fill |
| 1 | 131072 | 1 | 64 | 1 | 0.54 GB | 1.3 / 3.6 / 10.7 ms | fill |
| 32 (default ubatch, `T:proxima-model-interop/src/serving.rs:970`) | 2048 | 16 | 4 | 1 | 0.13 GB | 0.3 / 0.9 / 2.7 ms | fill |
| 32 | 32768 | 16 | 1 | 1 | 2.15 GB | 5.4 / 14 / 43 ms | scratch: 135 MB > budget, resident, 16 of 64 |
| 32 | 131072 | 16 | 1 | 1 | 8.59 GB | 21 / 57 / 172 ms | scratch, resident |
| 49 (verify) | 32768 | 25 | 1 | 1 | 3.36 GB | 8 / 22 / 67 ms | scratch: 206 MB, resident |
| 512 | 32768 | 256 | 1 | 2 | 34.4 GB | 43 / 115 / 344 ms per link | duration |
| 512 | 131072 | 256 | 1 | 6 | 137.4 GB | 57 / 153 / 459 ms per link | duration |

Reading it. (1) Decode at 2K has only 16 blocks, so `P = 16 < F = 64`: the dispatch fills a quarter of one wave. Each threadgroup streams 128 keys x 4 KB = 524 KB; the whole layer streams 8.4 MB, which is 56 us at 150 GB/s [D]. The regression shape (one threadgroup) is gone, but the decode gate, not this table, decides whether 16 is enough; if the 2K cell fails, the pre-stated follow-up is `Ls in {64, 32}` (P = 32, 64), an arithmetic change that re-runs slices 4 to 6. (2) The default-ubatch prefill (C = 32) at 32K or more runs resident on 16 threadgroups because M2 partials at that shape exceed the scratch budget (135 MB at 32K, 539 MB at 131K with `Ls = 128` [D]); a larger ubatch (512: 256 threadgroups, no scratch) is the fix, and it is a `ServingConfig.ubatch_size` choice, not a kernel one. (3) At C = 512 and 131072 keys one un-chained dispatch would stream 137.4 GB: 0.92 s at 150 GB/s, 2.75 s at the 50 GB/s floor [D]. The FLOP side is 1.10 TFLOP, 0.10 s at 10.6 TFLOPs f32 peak (32 cores x 128 lanes x 2 x 1.296 GHz [A]) [D], so the bytes bound. The first is 2.2 times under, and the second 1.4 times over, the roughly 2 s at which a peer measured macOS GPU recovery (n = 2, plausible [P]); the true bandwidth of this kernel is unmeasured, which is why the chain exists: 6 links of at most 21,888 keys, 153 ms each at 150 GB/s and 459 ms at the floor.

Unfused control (the legacy path with the flag off): the score reduce touches `C * L * Hq * hd * 4` bytes through the cache: 68.7 GB at C = 32 (0.46 s at 150 GB/s), 274.9 GB at C = 128 (1.83 s), 1099.5 GB at C = 512 (7.3 s) at L = 131072 [D, upper bound: it assumes no cache reuse]. The C at which it crosses 0.5 s is 34.9 rows [D]: the default ubatch of 32 is the largest power of two under that bound, so a long-context prompt on the legacy path must not run above `ubatch_size = 32`; slice 8 records the actual time.

Per-command-buffer time is a separate quantity. Prefill-shaped plans encode one command buffer per forward (`command_buffer_chunks = 1`, `T:omega/src/metal/device_buffers_arena_plan.rs:788-808`, `T:proxima-model-interop/src/serving.rs:1023`), so all 35 layers of a C = 512, L = 131072 chunk share it. The peer's measurement was of a dispatch; whether the recovery is triggered per dispatch or per buffer is not established [A]. Slice 8 records both; `Plan::set_command_buffer_chunks` (`PROXIMA_COMMAND_BUFFER_CHUNKS`) already exists if the buffer is the unit.

### 4.4 Sized constants (principle 12) and config (principle 4)

`omega/omega-runtime.toml [flash_attention]`, each key documented with the cost of raising or lowering it, `OMEGA_FLASH_ATTENTION_<KEY>` overrides, `rerun-if-env-changed`: `block_keys = 128` and `value_chains = 4` (arithmetic; both in the pipeline key), `resident_tgs_per_core = 2`, `gpu_cores_assumed = 8`, `scratch_budget_divisor = 8`, `dispatch_budget_ms = 500`, `dispatch_bandwidth_floor_gbps = 50` (non-arithmetic). They sit beside, and replace nothing in, `[attention_context_chunks]`, `[attention_block]`, `[attention_splits]` (`T:omega/omega-runtime.toml:188-269`), which the legacy arms keep. The one runtime knob is the existing `ServingConfig.flash_attention: bool` (`T:proxima-model-interop/src/serving.rs:588`, default false at `:968`); its rejection at `:1113-1121` ("requires a new fused Op variant") is stale and slice 3 replaces it with the demand check of 7.1. Builder and config parity: one test, `flash_attention_agrees_across_literal_and_default_override`, in the pattern of the existing `serving.rs` fixtures. No new config type.

## 5. R1 on the real key

### 5.1 The key today, and what changes

The kernel identity for `CachedAttention` is built at `T:omega/src/identity.rs:524-606`. It bakes `q{query_rows}`, `c{cached_key_rows}`, `n{new_key_rows}`, `u{new_upper | dyn}`, `x{context_chunks}` (itself `context_chunks_for(cached + new, ..)`, a function of the dispatch extent) and `_cb`. The entry name is a second string, built in `T:omega/src/msl/signature_tokens_prelude.rs:325-350`, with the same `q/c/n` tokens. Consequence: a 1-row decode BoundOp, a 32-row prefill BoundOp and a 49-row verify BoundOp of the same layer are three pipelines, and `x` differs with the extent.

Flash key (a new arm of the `match` at `T:omega/src/identity.rs:530`, taken when `fold == Flash`):

```
{prefix}_flash_attention_h{kv_heads}_g{query_groups}_d{head_dim}{_r{rotary_dim}}_s{scale_bits:08x}_l{cached_lower}_u{new_upper}_e{kv_dtype}_a{block_keys}x{value_chains}_o{operand_count}
```

`q`, `c`, `n` and `x` are gone. `l` and `u` stay because they are properties of the layer's mask, constant for a full layer (`MIN` and `0` for every row count), not of the dispatch; `_cb` is subsumed by `_o` (operand count). Row count, live cached rows, new rows, `P`, `links` and the scratch stride are uniforms. `entry` is derived from the identity in one function (the two strings must not drift; slice 4 test `flash_entry_and_identity_agree`). Two entries exist per shape class: the block/partials kernel and the merge kernel, both row-count-free.

### 5.2 The tests (each prints a count, and the count must be non-zero)

- `flash_pipeline_key_shared_by_decode_and_verify` (omega, CPU): `kernel_identity` for gemma4 full-layer BoundOps at `new_count in {1, 2, 3, 8, 17, 32, 49, 512}` x `kv bucket in {32, 2048, 32768}` (24 shapes) prints `distinct_keys=1 shapes=24`; a control with `fold = Legacy` prints `distinct_keys > 1`.
- `flash_rows_independent_of_row_count` (omega, Metal): shapes gemma4 full 512/8/1, qwen3-8b 128/4/8, qwen3.6 pass plane 256/8/2 (64 rotary + 192 pass), rows `{1, 2, 3, 8, 17, 32, 49, 64, 512}`, every row's bits equal the 1-row dispatch of that row; prints `dispatch_shapes=27 mismatching_rows=0`. `flash_split_equals_resident` (`P in {1, 2, 7, 64}`) and `flash_chain_equals_single` (`links in {1, 2, 7}`) on the same three shapes. C = 32 is in the row list because it is the default ubatch (`T:proxima-model-interop/src/serving.rs:970`); v2's list stopped at 17 and 49.
- `flash_verify_rows_equal_decode_rows` (interop, real gemma4 E2B; the blob is absent means RED with `rows_compared=0`, not skip). Procedure: prefill a fixed prefix to position `p0`; run the verify program over rows `[t_0 .. t_{C-1}]` with `C in {2, 3, 17, 32, 49}` (the tokens are the plain greedy continuation, so the drafts are the true ones); record each row's logits bits through `LogitsSink` (`T:proxima-model-interop/src/generate/pregather.rs:3074`); `LayerCache::truncate` back to `p0`; run the same tokens as `C` plain decode steps and record each step's logits bits (`T:proxima-model-interop/src/generate/decode.rs:5687`). Assert every verify row equals the decode row at that position, bit for bit. Today the verify branch (`T:proxima-model-interop/src/generate/decode.rs:5394-5470`) never calls `logits_sink.observe`, so the rows are consumed by sampling and dropped: recording them is part of slice 6. Prints `rows_compared=N` with `N = 2+3+17+32+49 = 103` and `mismatching_rows=0`.
- `flash_speculative_matches_plain` (interop, real gemma4): the AC10 prompt of the speculative spec (`Repeat exactly five times: the quick brown fox ...`), greedy and sampled configs, speculation on versus off. Prints `accepted_total=A rejected_total=B` and asserts `A > 0 and B > 0` (v2 asserted neither; a run with zero accepted drafts exercised only the row-0 path) and `identical=true`, `verify_steps >= 1`. Control `flash_vs_legacy_bits_differ`: flash and legacy logits differ in bits on the same context, so the equality asserts are not vacuous.
- No synthetic random-weight gemma4 model: it accepts no drafts, which is the v2 defect.

### 5.3 The correction to v2 B1

v2 said the legacy chunk count changes between decode and verify and "28 of 49 rows differ at short context". The count uses `effective_context_chunk_cap(G, hd) = min(4, max(1, floor(32768 / (4 * G * (hd + 2)))))` (`T:omega/src/msl/signature_tokens_prelude.rs:1378-1382`). [M] `E/chunk_cap.out`: gemma4 full (G 8, hd 512) 16,448 B per chunk, cap **1**; gemma4 sliding (8, 256) cap 3; qwen3-8b (4, 128) cap 4; qwen3.6 (8, 256) cap 3 (asserted at `T:omega/src/msl/tests.rs:4771-4778`). For a gemma4 full layer the legacy chunk count is constant 1, so that arm cannot break decode-versus-verify there ([M] control N1, cap 1: 0 of 49). The 28 of 49 is the cap-4 case: dense and qwen3-8b. The hazard v3 removes by construction is therefore real for the dense family (where no verify program exists yet, `T:proxima-model-interop/src/architecture.rs:318-332`) and for sliding and qwen3.6 shapes at cap 3, not for gemma4 full layers. The other legacy split, `splits_for` (`T:omega/omega-runtime.toml:222-269`), is keyed on live length, not simulated here.

## 6. The oracle: feasibility first, then tolerance, then what it cannot see

### 6.1 Roles

llama.cpp at f1ea20621 is the correctness oracle, invoked as an external build; fixtures are vendored with the commit and command (precedent: `T:proxima-tokenizer/tests/fixtures/llama-gemma4-tokenize/README.md`). proxima-internal references (CPU path, f64 definition, fused-versus-legacy, decode-versus-verify) are labelled internal wherever they appear. Ollama is the performance incumbent; the harness's needle arm uses it as a positive control (`AC14/AC15` of the long-context SPEC), and that use is labelled, not an oracle claim: the repo cites Ollama's own Go engine for gemma4 (`T:proxima-model-interop/src/gemma4/bind.rs:690-694`, "gemma4.go") [R], so an Ollama run is not a llama.cpp run.

### 6.2 What exists (located, not assumed)

- `L:build/bin` holds the prebuilt tools (dated Jun 2025) and libraries dated Sep 6; `strings -a build/bin/libllama.dylib | grep -ci gemma4` prints 0 [M]. It predates gemma4, as the tokenizer fixture README records (`unknown model architecture: 'gemma4'`).
- `PEER` is an f1ea20621 CMake build (Release, `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON`, `LLAMA_BUILD_EXAMPLES/TOOLS=ON`, Unix Makefiles, `CMAKE_HOME_DIRECTORY = L`; `CMakeCache.txt` [R]); `strings -a PEER/libllama.0.5.0.dylib | grep -ci gemma4` prints 10 [M]; only `llama-server` was built as a target (`ls PEER` lists `llama-server` and libraries).
- `<session>/long_ctx_backups/tools/llama-tokenize-f1ea20621` (42,304 B), linked against `PEER` (its `LC_RPATH` is `PEER`'s directory [M]). [M] On the gemma4 E2B blob it reproduces the vendored ids byte for byte on 3 of 3 fixtures tried (`newline_single`, `war_and_peace_2000`, `ascii_single_spaces`; 17 s cold, 0.56 s warm).
- `llama.h` at f1ea20621 carries every knob the capture tool needs: `cb_eval` and `cb_eval_user_data` (`L:include/llama.h:392-393`), `type_k`/`type_v` (`:395-396`), `flash_attn_type` (`:380`), `n_ubatch` (`:368`), `swa_full` (`:409`) [R]. `llama-eval-callback` cannot be the capture tool: its printer shows 3 leading and trailing elements per dimension (`L:common/debug.cpp:75-118`) [R].
- The external v2 programs printed "commit f1ea20621" but linked `L:build/bin` (their `LC_RPATH`), whose libraries are dated Sep 6, before the commit (Sep 28). [M] re-linked against `PEER` (libggml 0.25.3): the worked-example error is unchanged (7.493e-07, `E/llama_worked_example_peer.out`); llama's own flash-versus-unfused NMSE moved from 9.781e-06 to 2.312e-07 (peaked scores) and from 3.209e-05 to 3.901e-08 (flat) (`E/llama_oracle_demo_peer.out`, `E/llama_oracle_demo_peer_flat.out`); the f32 chain numbers (7.003e-13, 5.557e-13, 1.154e-13) are unchanged. v2's flash-versus-unfused figures were a different llama.

### 6.3 Configuration that must match on both sides

| axis | proxima | llama.cpp |
|---|---|---|
| weights | `$GEMMA4` blob (Q4_0 QAT E2B) | the same blob |
| token ids | the fixed tokenizer (fixtures `llama-gemma4-tokenize`); slice 1 asserts `proxima_ids == llama-tokenize ids`, n = 2048 | `llama-tokenize-f1ea20621` |
| KV dtype | f32 (`LayerCache` host cache; KV is f32-only today, SPEC "today's blockers") | `-ctk f32 -ctv f32` |
| flash attention | n/a; the reference structure is the unfused f32 chain | `-fa off`. With `-fa on`, `build_attn_mha` casts an f32 K and V to f16 before `ggml_flash_attn_ext` (`L:src/llama-graph.cpp:2644-2650`) [R], and the default is `auto` (`L:common/arg.cpp:1751`) [R]; neither can be an f32 reference |
| ubatch | `ServingConfig.ubatch_size = 512` (default 32, `T:proxima-model-interop/src/serving.rs:970`), so the captured chunk has cached 1536, new 512 | `-ub 512` (default 512, `L:common/common.h:454`), `-b 2048` |
| window | ring, W = 512 | `swa_full = false` (default), W = 512 |
| backend | Metal, `metal` feature | CPU `-ngl 0` for the reference, `-ngl 99` for the spread |

An f16 pair (proxima F16 KV after long-context R13/R14, against `-ctk f16 -ctv f16 -fa on`) needs its own tolerance measured on that pair; it is not part of v3 and no f32 tolerance transfers to it.

### 6.4 Two levels, because the model level cannot see attention arithmetic

llama's CPU backend quantizes activations to Q8_0 for a Q4_0 weight matmul (`L:ggml/src/ggml-cpu/ggml-cpu.c:240-243`: `.vec_dot_type = GGML_TYPE_Q8_0`) [R]; its Metal matvec takes f32 activations. So `kqv_out-il` differs between llama's own CPU and Metal runs by the projection noise of layers 0 to `il`, which has nothing to do with attention. A model-level tolerance measured that way is dominated by non-attention noise and cannot fail on a reassociation. v3 therefore uses both levels for different questions.

- G-K (kernel level, binding for the attention arithmetic). Capture, from the CPU llama run of slice 1, `Qcur_pos-4`, `Kcur_pos-4`, `Vcur_normed-4` (`L:src/models/gemma4.cpp:229,259,264`) and `kqv_out-4` (`L:src/llama-graph.cpp:2810`) for the last ubatch, layer 4 (the first full layer, hd 512, own KV). Feed llama's own bytes to proxima's flash op standalone: `q` and `k` are permuted identically into the `[even | odd | pass]` plane layout (a bijection on every dot product), `v` and the output are not permuted; a full layer's mask is plain causal and is rebuilt exactly. This is the inverse of v2's G1 (llama's graph on proxima's bytes), which could not see a proxima layout, mask or RoPE-plane defect.
  Tolerance: run llama's unfused f32 chain (`mul_mat` with f32 accumulate, `soft_max_ext`, `mul_mat`; the `-fa off` structure of `build_attn_mha`, `L:src/llama-graph.cpp:2679-2736`) on the same bytes on its CPU backend and on its Metal backend. `T_kernel = NMSE(llama_metal, llama_cpu)` with llama's own metric (`L:tests/test-backend-ops.cpp:302-313`); `T_max` is the same for max-abs. Reproduction control: the CPU graph reproduces the captured `kqv_out-4` bit for bit (`elements_equal == total`). Cross-check: `T_kernel` must be under llama's own default cross-backend bar for f32 graph ops, `1e-7` (`L:tests/test-backend-ops.cpp:1214-1216`) [R]; if not, llama's Metal is outside its own bar on this data and slice 2 is RED. The `5e-4` bar (`:7915-7917`) belongs to `FLASH_ATTN_EXT` with f16 K/V and is not used: it is 9 orders above the f32 reorder spread (7.0e-13 [M, `E/llama_oracle_demo_peer.out`]).
  Fixture size [D]: G-K, layer 4, cached 1024 + new 32 rows at the default ubatch: K and V `1056 * 512 * 4 * 2` = 4.3 MB, `q` `32 * 8 * 512 * 4` = 0.5 MB, `kqv_out` 0.5 MB, about 5.3 MB. G-M, layer 4, every 8th row of the last 512-row ubatch (64 rows, indices recorded): `64 * 4096 * 4` = 1.0 MB. Both are vendored with a sha256 manifest; the full dumps stay outside the repo and are named by hash.
- G-M (model level, end to end). Teacher-forced top-1 agreement over the fixed 2,304 ids (long-context AC12L-a/b/c): `A_flash >= A_ll`, where `A_ll` is llama's Metal-versus-CPU agreement and `A_flash` is proxima with flash on versus llama CPU. Plus, at layer 4 of the same runs, `NMSE(flash, llama_cpu) <= T_model` with `T_model = NMSE(llama_metal, llama_cpu)` on `kqv_out-4`: the number the owner's phrase names literally ("its CPU vs its Metal output on the same prompt"). The layer-4 inputs of the flash and legacy proxima runs are identical (layers 0 to 3 are sliding, `Kept`), so the upstream floor cancels between those two arms; the acceptance is stated against llama, not against the legacy arm.

### 6.5 Pre-registration and controls

`tolerance.toml` is committed by slice 2, from vendored llama outputs, before any kernel: `git rev-list --count <tolerance commit>..<kernel commit> >= 1` is an AC. Controls that must FAIL, each measured against `T_kernel`: one masked key leaked into the fold; the last key dropped; the mask shifted by one row. A leaked key of weight `w` adds about `(w |v|)^2 / mean(y^2)` to the NMSE, `w` near 1e-3 to 1e-4 at these lengths, so 1e-7 to 1e-8 [D], against an expected `T_kernel` near 1e-12 [A]; if a control does not exceed `T_kernel`, the gate cannot see that defect class and slice 5 is RED. Internal, exact, oracle-independent (kept from v2): the exact-support test over a `{0,1}` residue signature (three coprime moduli, CRT bound past 131072 keys); [M] `E/support_exact.out`: `worst_rel_err_vs_exact_count` 3.6e-8, 3.6e-8, 3.8e-8, 1.5e-8 against defect deviations 3.4e-2, 3.4e-2, 1.3e-3, 2.1e-2 (over-include, over-mask, swap pair), `nan_in_output=0` in all four rows (NaN in masked K/V rows never reached the output).

### 6.6 What this oracle cannot see

It cannot see accumulation choices below its noise: `NPV`, `Ls`, and any reorder whose NMSE is under `T_kernel`. [M] `E/llama_oracle_demo_peer.out`: `NMSE(llama unfused, contract fold) = 7.003e-13` while `NMSE(contract fold, f64) = 1.154e-13` and `NMSE(llama unfused, f64) = 5.557e-13`: the distance to llama is dominated by llama's own rounding. Those constants are chosen by the internal f64 check (3.2), labelled internal. It also cannot see anything about the sliding layers until they are routed, nor the f16 KV path, nor a defect only visible past position 2048 (the captured chunk sits at 1536 to 2047; the needle gate at 8192 in slice 7 covers the long-range behaviour).

## 7. Scope: routing, reversibility, families, deletion

### 7.1 Routing (`dead_code_cached_attention.rs`)

`cached_attention_candidates` (`T:proxima-tensor/src/bind/dead_code_cached_attention.rs:645`). For a gemma4 full-layer anchor the branches up to `local_window_not_vacuous` pass; `softmax_weights_decode_only` (`:1221-1225`, `via_gemma_template && new_key_rows != 1`) is where prefill and verify stop today, and `softmax_weights_eligible = via_gemma_template` (`:1241`) routes decode into Candidate B (arm `:1366-1580`). v3's edit, gated by the plan bit:

- With `flash_attention = false` (default): no line of this function changes behaviour; a test asserts the bound program is identical to today's (`plan_hash_equal=true` over the census shapes).
- With `flash_attention = true`: an anchor with no window (`cached_lower == MIN`, `local_row_bound == MAX`) and `via_gemma_template` skips both the decode-only decline and the Candidate B arm and builds `CachedAttention { fold: Flash }` for any `new_key_rows`, at the existing build site (`:1712-1742`). A windowed anchor keeps its legacy outcome (unfused at C > 1, Candidate B at decode when its feature is on). The new branch returns `Result<FlashRoute, DeclineReason>`, so a decline in NEW code carries its reason; the 64 legacy `continue;` sites (`:645-1745`) are not refactored (no gate here needs their reasons; v2's outcome-values rewrite is dropped from this spec, see 12).
- The plan bit rides `PlanNumerics` (`T:proxima-model-interop/src/generate/residency_caches.rs:1573-1576`) beside `fuse_cached_attention`, which `BackendRuntime` copies at construction, and reaches `bind_with_fusion` as a parameter next to `fuse_cached_attention` (`T:proxima-tensor/src/bind/gdn_moe_fusion_apply.rs:62-68`). The plan cache key is `(program_key, new_count, kv_bound_extent, outputs)` (`T:proxima-model-interop/src/generate/residency_caches.rs:1874-1882`), so decode, verify and each prefill chunk are separate plans that share one compiled pipeline (section 5).
- `flash_attention = true` with `cached_attention_fusion = false` is rejected in `apply_serving_config` (one typed check). Under demand, `BackendRuntime` counts `Flash` ops in the bound program against the architecture's declared full-attention layer count (7 for E2B, from the period-5 pattern, `T:proxima-model-interop/src/gemma4/bind.rs:686-696`) and returns `InteropError::FlashAttentionDeclined { expected: u32, fused: u32 }`. Call sites both ways: `ServingConfig { flash_attention: true, ..Default::default() }` versus the same with `flash_attention: false` selects the route; there is no `FlashAttention::{Auto, On, Off}` enum, because `(false, true)` is the only state it would remove and one admission check removes it.

`AttentionFold` is the one new type. Second question (what can a caller do that they could not before): route full layers to a different arithmetic without changing qwen3.6, dense or lfm2. The alternative without a type is the `PlanNumerics` bit alone read at emit, which flips every `CachedAttention` in the plan; that is the family-wide switch v3 refuses. It is data (two states, `#[non_exhaustive]`), not behaviour, and it is the information the recognizer holds and the emitter needs.

### 7.2 Per family (evidence, gate, and what stays)

| family | attention program | today | v3 | named gate to change that |
|---|---|---|---|---|
| gemma4 E2B full layers (7 of 35) | two-range, scale 1.0, hd 512 | decode: Candidate B only if feature `metal-fuse-attn-decode` is on (default-off: `T:proxima-model-interop/Cargo.toml:177-186` [R]); C > 1: unfused | flash route behind the bit; default flips only via slice 9 | slices 4 to 8 |
| gemma4 sliding layers (28 of 35) | ring, W = 512, hd 256 | unfused (Candidate B at decode with the feature) | stay legacy until slice 10 | slice 10: bands + ring phase alignment (v2 5.1, retained by reference), llama window-facts gate with answers from llama, 2-sigma decode rule |
| qwen3.6 (`QWEN35MOE`) | two-range with pass plane, `single_position_step: true` (C = 1) | legacy fused arms | stays legacy; the kernel is T1-tested on its shape (hd 256 = 64 rotary + 192 pass, G 8, Hkv 2) but not routed | G-Q36: `cached_attention_partial_rotary_parity` (3 tests) and `qwen35_partial_rotary_cached_attention_fuses_and_matches_the_unfused_layer`, ported to the flash arm with a derived tolerance; llama `qwen35moe` capture (`L:src/models/qwen35moe.cpp` exists [R]); qwen3.6 decode ms/token pair vs the current arm at 2K and 32K |
| qwen3-8b, mistral, qwen2 (`DENSE`) | prefill two-range; decode single-range placed KV, runtime `new_upper` (`T:proxima-model-interop/src/generate/load_model.rs:1295`) | legacy arms (cap 4) | stays legacy; the 131072 YaRN path is unchanged | G-DENSE: `cached_attention_coop_load_parity` (`m_greater_than_one_prefill_holds_parity_past_the_split_knee`, `the_single_range_fused_kernel_holds_parity_at_every_kv_capacity_bucket_padding`) and the qwen3 GQA two-range fixture (`cached_attention_rewrite_accepts_the_qwen3_gqa_qk_norm_fixture`) ported; llama `qwen3` capture (`L:src/models/qwen3.cpp` [R]); decode pair with YaRN 4/32768 at 131072 |
| LFM2 | cacheless per-token program (`T:proxima-model-interop/src/lfm2.rs:1-12`) | no cached anchor | unchanged; the census asserts `flash_ops=0` | a cached LFM2 program is an LFM2 port, not designed here |

Nothing in this spec routes qwen3.6, dense or lfm2. The parity tests above stay where they are, because the legacy arms they exercise stay (they are in `S:before.txt`; long-context R18 keeps binding). The dense speculative verify program (spec R2) is not built; when it lands it inherits decode-versus-verify identity from the flash arm only if that family is routed by its own gate.

### 7.3 Backends other than Metal

`fuse_cached_attention: false` is what wgpu and CUDA receive, so no `CachedAttention` reaches them; the existing arms return typed unsupported errors: `CachedAttention` at `T:omega/src/wgsl.rs:230`, `T:omega/src/cuda.rs:183`, `T:omega/src/wgpu_driver.rs:440`; `CachedSoftmaxWeights` at `T:omega/src/wgsl.rs:254`, `T:omega/src/cuda.rs:207,435`, `T:omega/src/wgpu_driver.rs:743-755`, `T:omega/src/error.rs:80` (`CachedSoftmaxWeightsNotSupported`). Adding a field to `CachedAttention` touches none of them (they match `{ .. }`). The CPU executor (`T:proxima-tensor/src/cpu/run_node.rs`) runs `Flash` with the sequential per-key arm, which is row-count-free by construction; it is not an oracle and is labelled internal wherever it is compared.

### 7.4 The retired bar, surfaced

The owner's bar of 2026-09-21 ("byte equivalence stays binding; fusion NOT promoted until it matches unfused Metal bytes", `M: project_gemma4_attn_fusion_recognizer_gap`) is retired for fused attention by the owner's answer of 2026-09-29 (TASKS.md "P2 decision"). What still binds: byte identity where the arithmetic is unchanged (the legacy arms, and Candidate B against unfused, 105 of 105 layer lines [P]); decode versus verify under speculative R1; the `flash_attention = false` default plan being identical to today's.

### 7.5 Deletion is the last slice and is conditional

Slice 11 deletes Candidate B (`BoundOpKind::CachedSoftmaxWeights`: 96 references in 34 files [M, `grep -rn` on `T:`]; the four backend arms above; `EmitError::CachedSoftmaxWeightsNotSupported`; the recognizer arm `:1366-1580`; the feature `metal-fuse-attn-decode`). It runs only if slice 8 recorded the decode gate as passed at every context, slice 9 shipped the default, and the R18 amendment of section 10 is signed by the owner. It does not delete the legacy chunk/block/split arms or `context_chunks_for`: qwen3.6 and dense still use them (7.2).

## 8. The decode gate (invariant 1), the recorded regression, and what slice 8 records

### 8.1 Arms, contexts, protocol

Three builds per iteration, run back to back in an order that rotates each iteration:

- A0 OFF: `metal-fuse-attn-decode` off, `flash_attention = false` (the shipped default; Candidate B is compiled out, `T:proxima-model-interop/Cargo.toml:177-186` [R]).
- A1 B: the feature on, `flash_attention = false`. This is the threshold arm: "Candidate B" means the byte-identical arm that measured 21.6 to 22.0 against 22.8 to 23.0 ms/token OFF (wall) [P: `M: project_gemma4_attn_fusion_recognizer_gap`, validated 2026-09-22].
- A2 FLASH: the feature on, `flash_attention = true`.

Contexts: 128 (the short-context shape at which the previous kernel family regressed), 2048, 32768. All arms use `ubatch_size = 32` (the default, `T:proxima-model-interop/src/serving.rs:970`) so the legacy prefill stays under the 0.5 s bound of 4.3 and under 2^32 threads (`32 * 32768 * 8 * 64 = 5.4e8`). Decode window: 64 tokens, `SpeculativeConfig::none()`, ms/token = `(elapsed_last - elapsed_first) / 63` from `TokenEvent.elapsed_ms` (a u64 millisecond clock, so a window statistic; `T:proxima-model-interop/src/generate/residency_caches.rs:2975-2999`, the same computation as `T:proxima-model-interop/examples/speculative_bench.rs:1290-1308`). Prompt: the R16c1 haystack reduced to ASCII with single spaces until the tokenizer fix lands.

Protocol (each item is a recorded failure of an earlier measurement, `M: feedback_interleave_gpu_arms_per_iteration`, `feedback_check_box_load_before_every_timed_run`, `feedback_quit_ollama_for_timing_runs`, `feedback_foreground_the_race_turn_boundaries_reload_ollama`, `feedback_bench_alone_no_fanout`): arms alternate per iteration, never as blocks (a block measured a 2.7x that was 1.1x); the box is idle (no agents, no compiles, no other Metal process); load, GPU utilization and `ollama ps` are printed before every arm; Ollama is quit for the run and reopened after; the run is foreground in the main shell; a warm-up iteration is discarded. `PROXIMA_METAL_ENCODER_ERROR_STATUS=1` is not relied on (it panics on main until the peer's fix lands): the harness reads `MTLCommandBuffer.status` and `.error` itself after every command buffer and prints `command_buffer_errors=N`; any N > 0 is RED.

### 8.2 The threshold, pre-stated

Per cell (context) at least 3 valid iterations, target 5; `r_i = ms_per_token(A2)_i / ms_per_token(A1)_i`.

- PASS iff `median(r) <= 1.000` at 2048 and at 32768, and the same at 128; pair-ratio CoV at most 5 percent; `contaminated = 0`; `command_buffer_errors = 0`.
- A cell with CoV above 5 percent or a contaminated pair is invalid: repeat up to 7 iterations; invalid twice is RED and no result is stated.
- Sanity row every time: `ms(A1) / ms(A0)` must sit near the recorded 0.95 [P]; if B is not faster than OFF on this box, the protocol is suspect and the cell is RED.
- Always recorded whether or not it passes: per arm the ioreg "Device Utilization %" samples (about 25 Hz, `feedback_gpu_saturation_on_every_perf_arm`), the launch-shape census (threadgroup count per flash dispatch, `base * P`), and, in a separate untimed `instrument` run, `PHYSICAL_DISPATCH_CALLS` per warm token: A1 1154 [P], A2 1112 (7 layers x (8 -> 2) fewer) or 1105 if resident [D].

### 8.3 Failure rule (no verdict, and the arm is not deleted)

If any cell fails: (1) run the captured-input single-layer harness (the owner's disposition of 2026-09-21) at that context, sweeping `P` from 1 to 128 at fixed shape: a knee near `F = 64` means fill was the cost and `R` or `Ls` is the lever; a flat curve means parallelism was not the cause and the recorded mechanism is refuted for this kernel, and the next measurement is the merge dispatch alone and the barrier count; (2) sweep `Ls in {64, 32}` (arithmetic; re-run slices 4 to 6), `Sk`, simdgroups, Safe versus Relaxed; (3) if no cell passes, flash is not promoted: it stays as the opt-in demand bit for memory-bound long prefill (R1 still holds, because the bit is all-or-nothing), the default stays false, and slices 9 to 11 do not run. B, OFF and every legacy arm are untouched in every branch.

### 8.4 Also recorded by slice 8

(a) the fill probe (4.1) writing `resident_tgs_per_core`; (b) Safe versus Relaxed for the flash pipelines (3.3); (c) max single-dispatch GPU time and total command-buffer time at 32K and 131K for `C in {32, 512}`, from a harness that owns its command buffers and reads `GPUStartTime` and `GPUEndTime` (per-dispatch counter sampling inflates 40x on this device [P: `M: project_gemma4_attn_fusion_recognizer_gap`], so it is not used). Threshold: max single dispatch at most 1.0 s (half the roughly 2 s recovery trigger [P]) and `command_buffer_errors = 0`; otherwise lower `dispatch_budget_ms` and re-run. The legacy unfused C = 32 dispatch at 32K and 131K is timed in the same harness to test the 0.46 s bound of 4.3 [D].

## 9. The whole-layer dispatch budget (owner: "~250 is not enough")

### 9.1 What is dispatched today, from data

Sources: the per-op census log `<session>/scratchpad/census/main_with_fusion.log` (996 KB; 1661 lines `boundop[unfused_production] ...` with `kind`, `extents`, `reads`), which I classified by `extents` with `E/census_classes.awk` [M]; the summary `long_ctx_backups/decode_op_census.md` [P]; the resident-skip code [R].

- BoundOps per decode token (s = 1, KV bucket 32): unfused 1661 = constant 262 + elementwise 488 + keep::reduce 903 + iota 8 (recomputed from the log: 903 and 488 [M]); with Candidate B 1416, having removed 245 (reduce 175, elementwise 105, minus 35 new fused ops) [P].
- Are the 262 constants dispatched? On the first call only. `ServingConfig::default().plan_time_constants = true` (`T:proxima-model-interop/src/serving.rs:1022`) makes `BackendRuntime` call `Plan::mark_plan_time_constants_resident` (`T:proxima-model-interop/src/generate/residency_caches.rs:1813-1814`; `T:omega/src/metal/device_buffers_arena_plan.rs:1784-1795`, which adds every `Op::Constant` to `resident_nodes`, iota excluded), and the encode loop skips a resident `Constant` or `Iota` whose buffer exists (`resident_skip`, `T:omega/src/metal/placements_execute_named.rs:1212-1216`). So the warm dispatched count is `BoundOps - 262`: **1399 unfused** [D] and **1154 with Candidate B**, which equals the 1416 -> 1154 a prior session measured when that default landed [P: M `project_gemma4_attn_fusion_recognizer_gap`, commit d7072f0b6]. The counter that proves it is `PHYSICAL_DISPATCH_CALLS` (`T:omega/src/metal/resident_nocopy_cache.rs:1330,1345`, `instrument`) on a warm step (step >= 2). `ENCODE_DISPATCH_CALLS` (`T:omega/src/metal/arena_encode_dispatch_finish.rs:1626`, `device_buffers_arena_plan.rs:1669`) also sits after the skip but counts `encode_op` calls, and one `encode_op` can issue split-plus-merge dispatches, so the physical counter is the one. Every count below is a DISPATCHED count; P3-3's census prints the BoundOp class matrix on CPU, and the untimed `instrument` run of P3-8 prints `PHYSICAL_DISPATCH_CALLS` at steps 1, 2 and 3 (steps 2 and 3 must be equal).
- The reduce split is exact from shapes [M `E/census_classes.out`, inputs `E/census_reduce_shapes.out`]: 903 = matvec **277** (q 35, k 15, v 15, wo 35, FFN 105 = 15 layers x 3 at width 6144 plus 20 x 3 at 12288, PLE 70, PLE projection 1, LM head 1) + RMSNorm hidden `[1,1536]` **176** (5 per layer x 35 + the output norm) + PLE input norm `[1,256]` **35** + q/k/v norm reduces **65** + softmax max/sum **140** + attention dot/AV **210**. 277 agrees with the "about 275" of the coordinator and with `M: project_gemma4_decode_dispatch_type` (275 dispatches, 39.6 percent of GPU time).
- The elementwise 488 by shape [M `E/census_elementwise_shapes.out`]: hidden `[1,1536]` 107; PLE `[1,256]` 35 plus `[1,8960]` 1; rope / norm-apply on q and k `[1,8,x]` 105 and `[1,1,x]` 60; K/V group broadcast `[1,1,8,x]` 105; softmax `[1,32,1,8]` 35 and `[1,1,1,8]` 35; misc 5. (No `[1,6144]` or `[1,12288]` elementwise: GeGLU is already composed into the down matvec's element body.) The assignment of the 105 broadcast ops to the attention chain is derived from the fact that the attention class then equals 350 + 70 + 105 = 525 = 35 x 15, the recognizer's absorbed-plus-anchor count.
- By class, unfused dispatched (sums to 1399): matvec 277, hidden norm 283, PLE 71, qkv-post 230 (reduce 65 + elementwise 165), attention chain 525, misc 13 (iota 8, five small elementwise). With Candidate B the attention class is 280 = 35 x 8 and the total 1154 [D, reconciles exactly].
- The census summary's sentence "gemma4 decode has no matmul/matvec ops" is wrong (the matvecs are `keep::reduce` with a `multiply_add` body; 277 of them); the numbers here come from the log lines, not from that sentence.

### 9.2 Structure of a layer, and the target

From the program builder (`T:proxima-tensor/src/spec/lfm2_single_range_cached.rs:1266`, the two-range function gemma4 binds through) and the shapes above, an own-KV layer is: pre-attn norm; q, k, v matvecs; q/k/v norm and RoPE; attention; wo matvec; post-attn norm, residual, ffn norm; gate and up matvecs, GeGLU; down matvec; post-ffw norm, residual; PLE gate matvec, gelu times the per-layer input, PLE projection matvec, post-norm, residual, layer scale; the per-layer PLE input norm. Nine matvecs. A shared-KV layer has no k, v matvecs and no k/v norm: seven.

Per-layer dispatch counts by tier (each cell is one dispatch):

| step | T1: existing primitives only | T2: + row chain | T3: + horizontal merge |
|---|---|---|---|
| pre-attn RMSNorm | 1 | 0 (folded into the previous layer's chain) | 0 |
| q, k, v matvec (own) / q (shared) | 3 / 1 | 3 / 1 | 1 / 1 |
| q, k, v norm and RoPE (own) / q (shared) | 3 / 1 | 1 / 0 (q prologue in the attention kernel) | 1 / 0 |
| attention (flash) | 1 | 1 | 1 |
| wo matvec | 1 | 1 | 1 |
| post-attn norm + residual + ffn norm | 2 | 1 | 1 |
| gate, up matvec (GeGLU in body) | 2 | 2 | 1 |
| down matvec | 1 | 1 | 1 |
| post-ffw norm + residual | 1 | 1 | 1 |
| PLE input norm | 1 | 0 (one batched dispatch per token) | 0 |
| PLE gate and projection matvecs | 2 | 2 | 2 |
| PLE post-norm + residual + layer scale (+ next pre-attn norm) | 1 | 1 | 1 |
| own-KV layer / shared-KV layer | **19 / 15** | **14 / 11** | **11 / 10** |

Per token, by class (dispatched; B-state 1154 is the baseline; the misc 13 is unchanged): 

| class | unfused | with B | T1 | T2 | T3 |
|---|---|---|---|---|---|
| matvec | 277 | 277 | 277 | 277 | 212 |
| hidden norm (reduce + elementwise) | 283 | 283 | 176 | 106 | 106 |
| PLE | 71 | 71 | 36 | 1 | 1 |
| q/k/v post (norm, RoPE) | 230 | 230 | 65 | 15 | 15 |
| attention chain | 525 | 280 | 35 | 35 | 35 |
| misc | 13 | 13 | 13 | 13 | 13 |
| **total** | **1399** | **1154** | **602** | **447** | **382** |

The per-layer construction gives 15 x 19 + 20 x 15 = 585 (T1), 15 x 14 + 20 x 11 = 430 (T2), 15 x 11 + 20 x 10 = 365 (T3), plus about 13 once-per-token dispatches: 598, 443, 378, within 4 of the class table (602, 447, 382) [D]; the two derivations use different bookkeeping (per layer versus per class) and agree to that tolerance, which is the cross-check that neither dropped a class.

Committed budget: the T2 numbers, 14 dispatches per own-KV layer, 11 per shared-KV layer, about 450 per token: 3.1 times fewer than 1399 and 2.6 times fewer than 1154. T1 (about 600, 1.9 times fewer than 1154) needs no new primitive. T3 (about 380, 3.0 times fewer than 1154) is the stretch and depends on horizontal merge admitting Q4_0/Q8_0 (9.3). All are derived by construction; P3-3's census and each D slice's census print the exact count.

What this does not claim: that ms/token moves in proportion. The recorded evidence is that one predicted magnitude was wrong by 10x (predicted +93 percent from dispatch-type, measured +8.6 percent GPU and +4.6 percent wall, `M: project_gemma4_decode_dispatch_type`, "Calibration"), and the non-matvec GPU time was about 60 percent (9.7 of 16.1 ms) [P] over the roughly 1386 non-matvec BoundOps of that measurement, about 7 us each [D] (877 are dispatched today); the fit in 9.5 decides whether dispatch count or bytes is the binding term.

### 9.3 Per class: the primitive that fuses it, and the type questions

None of these is a `Pipe`. They are `BoundOp`s of the tensor compiler; a `Pipe` is the runtime request/response face, and `BoundOpKind` is the compiler's algebra (12 explains why that is a structural exception).

| class | existing primitive | what must be added, and the two questions |
|---|---|---|
| attention chain (525 -> 35) | `CachedAttention`, one new field `fold` (7.1) | nothing else. Pipe question: an op, not a pipe. Caller question: routes a different arithmetic per family |
| RMSNorm tails: post-attn, post-ffw, PLE post-norm, with residual and layer scale (hidden norm 283) | `BoundOpKind::Reduce` with `epilogue_body`, `epilogue_operands`, `epilogue_broadcast_axes` (`T:omega/src/identity.rs:614-650` keys them); `NumericRewrite::ReduceEpilogueFusion` (`T:proxima-tensor/src/numeric.rs:158-161`) | extend the recognizer to the gemma shapes. The doc at `T:proxima-tensor/src/numeric.rs:158-161` says "bit-exact by construction"; a recorded GPU check found logits bits differ under that absorption (`M: project_gemma4_attn_fusion_recognizer_gap`, "the generic reduce-epilogue absorption is NOT bit-exact; REVERTED"), so the label is false as written. Under the owner decision the gate is the llama oracle plus R1, not bits, and the slice re-labels the permission (it needs at least `contraction`). No new type: the epilogue already carries a residual operand |
| q/k/v post: norm plus RoPE (qkv-post 230) | `ReduceEpilogueFusion` for the norm; `ChainFusion` for RoPE (`T:proxima-tensor/src/numeric.rs:154-157`) | recognizer extension; RoPE reads two planes, so the norm reduce takes both planes as operands. No new type |
| GeGLU, PLE gelu-times-input | composed into the following matvec's element body at bind (no `[1,6144]`/`[1,12288]`/`[1,256]`-gelu ops appear in the 488) | nothing; the P3-3 census asserts it stays composed |
| residual adds, layer scale | the norm-tail epilogue operands | nothing |
| identity copies (105 in the B-state census [P]) | feature `identity-copy-alias` (the recorded census 1663 -> 1593 with it [P: `52ba967c` `census_results.txt`]) | promote it; no new type |
| row chain: post-attn norm -> residual -> ffn norm, and PLE norm -> residual -> scale -> next pre-attn norm (two reductions on one hidden row of 1536 f32 = 6 KB, one threadgroup) | none: an epilogue feeds its own reduce's output, not a second reduce | ONE primitive, only if the fit in 9.5 says dispatch count binds: a `Reduce` chain whose second stage reads the first stage's epilogue output inside one threadgroup. Pipe question answered by writing it: with the existing primitive G_a is two `Reduce` BoundOps, two dispatches; the algebra has no single BoundOp holding two reduce stages. Caller question answered by the call site both ways: before, two dispatches per chain and a `[1,1536]` round-trip through DRAM; after, one. That changes the dispatch count, so it is not a relocation. Shape: a `BoundOpKind::RowChain { stages }` built by a recognizer, not a wrapper; `stages` is bounded (2), inline, box-free |
| horizontal merge (qkv 3 -> 1, gate/up 2 -> 1) | feature `metal-horizontal-merge`, `build_merged_dispatch` (`T:omega/src/metal/device_buffers_arena_plan.rs:1135`): N independent packed-row matvecs into one `grid.z` dispatch; the oracle test is Q4_K (`T:omega/tests/horizontal_merge_dispatch.rs`) | extend admission to the Q4_0/Q8_0 shapes the gemma4 checkpoint uses; today's refusal reads "not a packed two-operand matvec shape" (`:1105`) [A: whether that refusal fires for Q4_0 is unmeasured]. No new type |
| embedding, PLE shared projection, output norm, LM head (about 13) | matvec + gather | nothing |

### 9.4 Slices ordered by dispatches removed per unit of work

Removal is measured against the B-state 1154, an upper bound (constants' class membership is not logged, 9.1 residual). Work is counted in slices.

| rank | slice | removes (upper bound) | work | gate |
|---|---|---|---|---|
| 1 | D1: qkv-post via `ReduceEpilogueFusion` + `ChainFusion` on gemma shapes | 165 (230 -> 65) | 1 slice, existing primitives | gate F |
| 2 | D2: hidden-norm tails and residuals via the same recognizer, permission re-labelled | 107 (283 -> 176) | 1 slice, same machinery | gate F |
| 3 | P3-3..P3-8 then P3-10: flash on all 35 layers | 245 (280 -> 35); 49 from the 7 full layers alone, 196 more from the 28 sliding layers | 8 slices for the full layers, +1 for sliding | slices 4 to 8, 10 |
| 4 | D3: PLE fold (norm + gelu-times-input into the matvec body, batched input norm) | 35 to 70 (71 -> 36 -> 1) | 1 slice | gate F |
| 5 | D4: promote `identity-copy-alias`; extend `metal-horizontal-merge` to Q4_0/Q8_0 | 65 (matvec 277 -> 212), identity copies inside the classes above | 1 slice each | gate F |
| 6 | D5: `RowChain` (the one new primitive) | 70 (hidden 176 -> 106) + 35 (PLE) + 50 (qkv-post 65 -> 15) = 155 | 1 slice, a new BoundOp kind | gate F, and the 9.5 fit |

Per slice, D1 removes 165 and D2 107, while the eight attention slices up to P3-9 together remove 49 (the 7 full layers alone), about 6 per slice; P3-10 by itself removes 196. That is why the T1 budget is 602 and not 1154 - 49. The build order still starts with P3-1 to P3-8, because the oracle tool, the R1 test and the decode gate are the substrate that gate F reuses for every D slice; D1 and D2 can start once P3-6 and P3-8 exist.

### 9.5 Gate F (every fusion slice, attention or glue) and the fit

(1) Count: the census prints `PHYSICAL_DISPATCH_CALLS` per warm token at steps 2 and 3, equal to each other and to the tier's expected total; a different count is itself the finding. (2) R1: the fused op's pipeline key is row-count-free (same shape as 5.1), and `flash_verify_rows_equal_decode_rows` (5.2) passes with the fused glue on. (3) Oracle: llama tensors at the fused boundaries, captured by the slice-1 tool with the same filter mechanism: `attn_norm`, `Qcur_pos`, `Kcur_pos`, `attn_post_norm`, `attn_out`, `ffn_norm`, `ffn_post_norm`, `pe_in`, `per_layer_embd_out`, `out_scaled`, `l_out` (`L:src/models/gemma4.cpp:199,229,264,285,288,349,362,370,385,394,398`); kernel-level tolerance from llama's own same-bytes CPU-versus-Metal spread of the corresponding `rms_norm`/`mul`/`add` chain (procedure of 6.4); end-to-end `A >= A_ll`. (4) Decode pair: the fused arm's ms/token at most the previous arm's, protocol and threshold of section 8. (5) The fit: over the arms measured so far, `ms_per_token = a + c * N_dispatched`, with residual and degrees of freedom printed. D5 (`RowChain`) proceeds iff the predicted saving `c * 155` is at least 3 sigma of the pair-ratio noise of section 8; otherwise the bandwidth term binds and the next slice is the matvec, not fusion.

### 9.6 What is left for tok/s beyond attention (one paragraph)

Attention flash on all 35 layers removes at most 245 of the 1154 dispatches (21 percent) and, from the 7 full layers alone, 49 (4 percent); the recorded decode gap is about 60 percent unfused non-matvec GPU time (9.7 of 16.1 ms gpu_exec over about 1386 non-matvec BoundOps then, 877 dispatched now) and about 25 percent host overhead (5.3 of 21.4 ms wall, of which encoding 1661 dispatches was 1.75 ms, about 1.05 us each [D]) [P: `M: project_gemma4_decode_dispatch_type`], and both scale with the dispatch count that the glue classes, not attention, dominate: 283 hidden-norm ops, 230 q/k/v-post ops, 71 PLE ops. The next primitive after flash is not new: it is the existing `ReduceEpilogueFusion` (a `Reduce` with epilogue body and broadcast operands, so an RMSNorm is a sum-of-squares reduce plus a broadcast epilogue that also carries the residual and the layer scale) and `ChainFusion` (elementwise chains such as RoPE) extended to the gemma4 shapes, worth 165 + 107 + 35 = 307 dispatches (T1, 602 total), followed by horizontal merge for q/k/v and gate/up (65) and, only if the 9.5 fit says dispatch count binds, the one new `RowChain` kind (155) for T2 (about 450). The bytes side is separate: the Q4_0/Q8_0 matvecs are about 40 percent of GPU time at about 150 GB/s in situ against Ollama's 158 GB/s [P], so after T2 the remaining lever is the matvec, and no claim is made here about the ms/token any tier reaches.

## 10. SPEC reconciliation (`S:SPEC.md`), no self-excusing lists

### 10.1 R12L / AC12L (llama.cpp ring-window oracle)

- AC12L-a/b as written (`S:SPEC.md:220-221`) name `others/llama.cpp/build/bin`. That build's `libllama.dylib` has 0 gemma4 strings [M], so the command cannot run on gemma4. Amend both rows to the f1ea20621 build of slice P3-1 (its target list adds `llama-perplexity`, the tool `S:TASKS.md:410` names for `--kl-divergence-base`), and add the flags that must match: `-ctk f32 -ctv f32 -fa off -ub 512` (6.3). AC12L-c/d are unchanged.
- One llama run pair yields three artifacts: AC12L-b's `A_ll` (Metal versus CPU top-1 agreement over the fixed 2,304 ids), G-M's `T_model` (`kqv_out-4` CPU versus Metal), and the input bytes of G-K. Slice P3-2 consumes SPEC slice 6L's run if it has landed and otherwise performs 6L's commands; it never runs llama twice on the same configuration.
- Add AC12L-e: proxima with `flash_attention = true`, ring on, teacher-forced over the same ids: agreement `>= A_ll`, first-disagreement index printed. Add AC12L-f (control): the same with `--ring-offset 1` must be below `A_ll`, so the check still sees the ring with flash on. Both reuse AC12L-a's fixture.
- R12 and R12L stay as written; the "amendment 2026-09-29" section is edited in place, not duplicated.

### 10.2 R18 / AC23 (regression baseline)

R18 ("every test that passed before slice 1 still passes after slice 13", `S:SPEC.md:93`) and AC23 (`S:SPEC.md:211`, final compare `comm -23 before.txt after.txt` printing `0` lines) bind unchanged through slices P3-1 to P3-10: those slices add tests, change no default (`flash_attention = false`, 7.1) and delete nothing, so the compare must print 0 after each slice that builds. Slice P3-9 changes the default for gemma4; its exposure list (tests in `S:before.txt` that can see gemma4 full-layer arithmetic) is `proxima-model-interop/tests/gemma4_program_metal_cpu_parity.rs`, `gemma4_tiled_gemm_defaults_full_logit_vector_diff.rs`, `gemma4_correctness_gate.rs`, and the ring-parity example (AC12/AC13, an example, not a test binary). None is edited to pass: a test whose expectation the legacy arm produced sets `flash_attention = false` explicitly (it names the arm it pins, and that arm remains supported), and a test that then fails with the default on is a finding for the slice, not for the test.

The only deletion the spec proposes is Candidate B, slice P3-11, and it is the only place R18 changes. In `S:before.txt` the tests whose subject is `BoundOpKind::CachedSoftmaxWeights` are exactly three [M, `grep -n softmax_weights before.txt`]: `omega msl::tests::cached_softmax_weights_render_is_deterministic_and_width_dependent` (`before.txt:97`), `proxima-tensor cpu::tests::cached_softmax_weights_bound_step_matches_hand_computed_softmax` (`:837`), `proxima-tensor cpu::tests::cached_softmax_weights_group_max_folds_in_new_scores_when_larger` (`:838`). Their contracts are re-asserted for the flash arm before deletion: `flash_render_is_deterministic_and_shape_dependent`; the existing sequential-arm hand-computed test `cpu::tests::cached_attention_bound_step_runs_online_softmax` (passes today) at the gemma4 shape; and `flash_max_folds_new_and_cached_when_each_wins` (the recorded defect where the group max ignored the new token's own score, `M: project_gemma4_attn_fusion_recognizer_gap`: gate a fused reduce on a fixture where EACH operand wins). The amendment text, applied only if P3-11 runs and only after the owner signs it: "R18 (amended for P2 v3): every test that passed before slice 1 still passes after slice 13, except the 3 tests named in `p2_deleted_tests.txt`, whose subject is deleted by slice P3-11 and whose contract is asserted by the named flash tests." If P3-11 does not run, R18 stands as written. Tests behind `metal-fuse-attn-decode` never entered `before.txt` (the feature is default-off), so P3-11 first generates `before_fuse.txt` at the pre-deletion commit with the feature on; its gate is that every name in it either passes afterwards or is one of the 3 above; `passed_tests.sh` records FAIL and SKIP as well as PASS for that run (v2's P2-0 change, folded in rather than a slice of its own).

### 10.3 New requirements

| id | requirement | testable in isolation |
|---|---|---|
| R19 | No dispatch of the flash lowering has threads or scratch that grow with context beyond `nblk` partials inside the scratch budget; `flash_ops = 7` on gemma4 E2B | yes |
| R20 | One pipeline and one association order serve decode, verify and prefill: a verify row's logits equal the decode row at that position, bit for bit, with accepted drafts > 0 and rejected > 0 | yes |
| R21 | With `flash_attention = false` the bound program is identical to today's; with it true the 7 full layers are `Flash`, and a shortfall is a typed error carrying counts | yes |
| R22 | The flash op on llama's own q/k/v bytes is within `T_kernel` of llama's CPU output; three controls exceed it; end-to-end top-1 agreement is at least `A_ll`; needle recall holds | yes |
| R23 | The fold uses exactly the visible keys (exact-support test, both directions, NaN in masked rows) and the worked example holds | yes |
| R24 | A required threadgroup width above the compiled pipeline's limit is a typed error on both dispatch branches | yes |
| R25 | Flash decode ms/token is at most Candidate B's at 2048 and 32768 (and 128), interleaved, 3+ iterations, CoV recorded; Candidate B is deleted only if this passed | yes |
| R26 | No single flash dispatch runs longer than the duration bound; the measured max single dispatch at 32K and 131K is at most 1.0 s and `command_buffer_errors = 0` | yes |
| R27 | The dispatched-op count per warm decode token is printed per class and per tier and equals the tier's expected total | yes |

## 11. Slices

Every slice is one commit, one behaviour, one command, one count (N == 0 is RED). "Slice N" anywhere in this file means P3-N. Build and Metal slices wait for the GPU window as in `S:TASKS.md`. Writing code is sonnet, running and measuring is haiku, judgment stays here. Skills (principle 13): the arithmetic contract engaged `/algorithm-development` (worked example, sim, this file); the routing and partition decisions were the contested ones and are what this synthesis resolved; no `/security-review` or `/discovery-loop` (an oracle exists).

### 11.1 Attention

| # | behaviour | command | expected count and rule |
|---|---|---|---|
| P3-1 | a gemma4-capable llama.cpp build exists and captures per-layer attention tensors on the fixed prompt (CPU only) | cmake configure of `$L` into `<tools>/llama-f1ea20621-build` (Release, `GGML_METAL=ON`, `LLAMA_CURL=OFF`), cmake build of `llama-perplexity llama-tokenize llama-attn-dump`, then `llama-attn-dump -m $GEMMA4 --ids <2048 ids> -ngl 0 -ctk f32 -ctv f32 -fa off -ub 512 -o <dir>` (`llama-attn-dump` is a C++ generator of about 60 lines beside the fixtures, `cb_eval` asking only for the names in 6.4) | `gemma4_strings_in_libllama >= 1`; `files_written=152` (35 layers x 4 ubatches of `kqv_out` = 140, plus 4 x {`Qcur_pos-4`, `Kcur_pos-4`, `Vcur_normed-4`} = 12); `token_ids_equal=true n=2048` (llama-tokenize versus proxima's fixed tokenizer) |
| P3-2 | the tolerance is derived from llama's own CPU-versus-Metal spread and committed before any kernel | `llama-attn-dump ... -ngl 99`; `llama-attn-spread <layer-4 bytes>` on the CPU and Metal ggml backends; `llama-perplexity` for AC12L-a/b (Metal window) | prints `T_kernel_full`, `T_max_full`, `cpu_reproduction elements_equal == total`, `T_model_layer4`, `A_ll=<n>/2304`; RED unless `0 < T_kernel_full < 1e-7`; writes `tolerance.toml` |
| P3-3 | the route exists, is off by default, and the census prints the dispatched-op classes | cargo nextest run -p proxima-tensor --features std,cached-attention-streaming flash_route; cargo nextest run -p proxima-model-interop --features std,metal attention_dispatch_census | 8 passed: plan hash equal with the bit off at `new_count` 1, 32, 49 (3); `flash_ops=7` with it on at the same 3 (3); typed `FlashAttentionDeclined` (1); rejected without fusion (1). Census prints `boundops=1661 constants=262 matvec=277 hidden_norm=283 ple=71 qkv_post=230 attention=525 misc=13` and asserts the class sum equals 1399 |
| P3-4 | the flash lowering exists: row-count-free key, both dispatch branches checked, T1 | cargo nextest run -p omega --features metal flash_ dispatch_rejects_width | 13 passed: `flash_rows_independent_of_row_count` x3 shapes printing `dispatch_shapes=27 mismatching_rows=0`; `flash_split_equals_resident` x3; `flash_chain_equals_single` x3; `flash_pipeline_key_shared_by_decode_and_verify` printing `distinct_keys=1 shapes=24` (its `Legacy` control prints `> 1`); `flash_entry_and_identity_agree`; `dispatch_rejects_width_over_pipeline_limit` on the `grid2d` branch and on the width branch (2); `flash_pipeline_supports_declared_width` printing `required=512 limit>=512` |
| P3-5 | flash matches llama within the derived tolerance, and the controls that must fail do | cargo nextest run -p omega flash_kernel_matches_llama; cargo nextest run -p proxima-model-interop --features std,metal flash_layer_matches_llama | 2 + 2 passed: `nmse <= T_kernel`, `max_abs <= T_max`, controls (leak, drop, mask shift) each `> T_kernel` (3 must fail), `fixture_commit=f1ea20621`, the 9-cell `NPV x Ls` table with the internal f64 line labelled internal; `git rev-list --count <tolerance commit>..<kernel commit> >= 1` |
| P3-6 | decode and verify are bit-identical on the real model, with accepted drafts | cargo nextest run -p proxima-model-interop --features std,metal flash_verify_rows_equal_decode_rows flash_speculative_matches_plain flash_vs_legacy_bits_differ | 3 passed: `rows_compared=103 mismatching_rows=0`; `accepted_total>0 rejected_total>0 identical=true verify_steps>=1`; control `differing_elements>0` |
| P3-7 | with flash on, real gemma4 still answers, finds needles and agrees with llama | `gemma4_e2b_answers`; `niah --model "$GEMMA4" --ctx 8192 --needles 10`; AC12L-e and AC12L-f | `answered_correctly=true` x4; `proxima found=X/10 ollama found=Y/10` with Y >= 9 and X >= Y (Ollama as the labelled positive control, 6.1); `agreement=A_flash/2304 >= A_ll` and the control below `A_ll` |
| P3-8 | the decode gate and the duration bound are measured under the pre-stated rules | section 8 protocol, per arm and context | 3 contexts x 3 arms x >= 3 valid iterations printed; per cell `median_ratio`, `pair_cov`, `contaminated=0`, `command_buffer_errors=0`; the fill probe row; the Safe-versus-Relaxed row; max single dispatch at 32K and 131K for C in {32, 512} (4 rows); PASS or RED by 8.2 and 8.4, no verdict beyond the rule |
| P3-9 | (conditional on P3-8 PASS at every cell) the gemma4 default becomes `flash_attention = true` | P3-3 to P3-7 re-run at the default; then `comm -23 before.txt after.txt` against the `passed_tests.sh` output | the same counts as those slices; `0` lines |
| P3-10 | (conditional on P3-9 and its own 2-sigma decode rule) the 28 sliding layers route to flash | bands replace the two scalars (72 references in 12 files [P: v2 5.4]); ring phase alignment (v2 5.1); `flash_rows_independent_of_row_count` gains ring-offset cases; window-facts gate with llama.cpp answers as the oracle (positive control: llama finds >= 4 of 5) | `mismatching_rows=0` on the ring cases; window gate `X >= Y >= 4` with the ring-offset control lowering X; ring parity numbers of AC12/AC13 at their recorded values; flip iff ms/token improves by more than 2 sigma and nothing regresses by more than 2 sigma |
| P3-11 | (conditional on P3-8, P3-9 and the signed R18 amendment) Candidate B is deleted | `git grep -c CachedSoftmaxWeights -- proxima-tensor omega proxima-model-interop`; `passed_tests.sh` with statuses against `before.txt` and `before_fuse.txt` | `0`; `PASS -> ABSENT` equals the 3 names of 10.2, every other transition 0 |

### 11.2 Dispatch budget (gate F of 9.5 for each; order of 9.4)

Expected dispatched count per warm token, starting from the B-state 1154 with attention unchanged, D slices applied in order (each prints its own class delta):

| # | behaviour | class delta | running total |
|---|---|---|---|
| D1 | q/k/v post (norm, RoPE) fused by the existing recognizers | 230 -> 65 | 989 |
| D2 | hidden-norm tails, residuals, layer scale fused; `numeric.rs` permission re-labelled | 283 -> 176 | 882 |
| D3 | PLE input norm and gelu-times-input folded | 71 -> 36 | 847 |
| D4 | `identity-copy-alias` promoted; `metal-horizontal-merge` extended to Q4_0/Q8_0 | matvec 277 -> 212 | 782 |
| D5 | (conditional on the 9.5 fit) `RowChain` | 176 -> 106, 36 -> 1, 65 -> 15 | 627 |

Attention's 245 (280 -> 35: 49 at P3-9 for the 7 full layers, 196 more at P3-10) is applied on top of these. End states: T1 = 1154 - 165 - 107 - 35 - 245 = 602; T3 = 1154 - 165 - 107 - 35 - 65 - 155 - 245 = 382; T2 = T3 + 65 = 447 (T2 leaves the 65 matvecs unmerged). Each D slice's own count is the class delta in the table; the running total is what its census must print if the earlier D slices have landed and attention is still at B (280).

## 12. Structural checks

Central claim as a lint ("everything is a pipe"). Not pipes, and why each is justified: `BoundOpKind::CachedAttention`, `AttentionFold`, `RowChain`, the flash lowering, `FlashSplit`, `FlashShape`, `FlashCaps` are tensor-compiler algebra (`BoundOp` is data the emitter lowers to one dispatch); a `Pipe` is the runtime request/response face, and the tensor compiler sits below it exactly as the codecs do. `flash_split` is a pure function over integers, no allocation, tier-1 shaped. The consumer-facing edge (`LoadedModel`, `ServingConfig`) is unchanged and stays the existing surface; no new `Pipe`, no blanket impl, no wrapper type, no `Box`.

Second question on each type. `AttentionFold`: routes one arithmetic per family (7.1). `FlashSplit`: returns four related numbers (`P`, `links`, `keys_per_link`, scratch) that would otherwise be a 4-tuple. `FlashShape` and `FlashCaps`: argument records; `FlashCaps` is what lets a test sweep the per-target constants (4.2). `DeviceFacts` was written, then dropped: its two call sites were identical lines. `RowChain`: changes the dispatch count (9.3), conditional on the fit. `FlashAttention::{Auto, On, Off}`: dropped (7.1).

Information destroyed, found, and what closes it: (1) verify-row logits are consumed by sampling and dropped at `T:proxima-model-interop/src/generate/decode.rs:5394-5470` (closed by `LogitsSink::observe` per verify row, slice P3-6); (2) the recognizer's route decision does not reach the emitter (closed by `fold`); (3) a required threadgroup width is lost to `min` and unchecked on `grid2d` (closed, 3.5); (4) the kernel identity mixes dispatch data (`q`, `c`, `n`, `x`) with arithmetic (closed, 5.1); (5) the baseline records PASS only, so a FAIL or SKIP never enters it (closed for the one deletion, 10.2); (6) the census does not log which class the 262 constants belong to (OPEN: the class counts are BoundOps, and the dispatched split by class needs the class-by-kind matrix that P3-3's census prints); (7) the reason at 53 declined recognizer sites (unchanged: no gate here needs it; reasons exist only in the new branch).

Abandoned designs (constraints applied before, not after):

1. v2's one lowering that replaces Candidate B and the three legacy arms, deleted before any decode measurement. Abandoned: irreversible before evidence (invariant 1), and it would have removed arms that qwen3.6 and dense still run, with no gate for them.
2. Choosing `P` inside the encoder from the compiled pipeline's footprint. Abandoned by the no-ambient-state constraint: the pipeline does not exist when `emit` runs; `P` comes from the core count and build constants, and the footprint only validates.
3. Making `P`, `Ls`, the merge tree or the tile grid depend on rows or on the dispatch. Abandoned by measurement: block size from rows N2 (49 of 49 rows differ), each threadgroup merging its own blocks first N3 (49 of 49), tile grid anchored at `cached_len` N4 (47 of 49), legacy extent-keyed chunks N1 (0 / 12 / 28 at caps 1 / 3 / 4).
4. G1 as llama's graph on proxima's bytes plus the whole-model layer-4 comparison as the attention tolerance (v2). Abandoned: it cannot see a proxima layout, mask or RoPE-plane defect, and the model-level CPU-versus-Metal spread is dominated by llama's Q8_0 activation quantization on CPU (`ggml-cpu.c:240-243`), which is not attention.
5. Pinning `MathMode::Safe` for the whole flash pipeline (v2). Abandoned: recorded 1.35x on the matvec and 1.20x on a whole decode program [P]; pinning moved to noinline functions plus a measured rule.
6. The `FlashAttention` enum, `DeviceFacts`, and v2's outcome-values rewrite of 64 recognizer sites. Abandoned: identical call sites, or no gate that needs them.
7. `bands: [CausalBand; 2]` in this spec's main path. Moved to slice P3-10: gemma4 full layers are the constants (MIN, 0) for every row count, so only the sliding route needs the new-range lower bound.

Corrections to v2 (cites and arithmetic):

1. B1 chunk cap: 1 for gemma4 full layers, not 4 (5.3).
2. "commit f1ea20621" on the v2 ggml runs: they linked the Sep-6 libraries. Flash-versus-unfused NMSE was 9.781e-06 and 3.209e-05 there and is 2.312e-07 and 3.901e-08 on f1ea20621 (6.2). The worked-example error 7.493e-07 and the f32-chain numbers are unchanged.
3. v2's cite `ggml-cpu/ops.cpp:7101` ("fattn: unsupported K-type"): at f1ea20621 the assertion is at `ops.cpp:8693`; irrelevant to v3 because `build_attn_mha` casts an f32 K to f16 before `ggml_flash_attn_ext` (`llama-graph.cpp:2644-2650`).
4. The identity cite `T:omega/src/identity.rs:524-595` ends at 606 (the `format!`); the entry name is a second string in `signature_tokens_prelude.rs:325-350`, which v2 did not name.
5. v2's dispatch percentages used 1661, which includes 262 constants that are not dispatched on warm steps (`placements_execute_named.rs:1212-1216`, `T:proxima-model-interop/src/serving.rs:1022`): the baselines are 1399 and 1154, and "98 fewer, 5.9 percent" (7 x 14, against unfused) becomes 49 of 1154, 4.2 percent, against the B-state (7 x (8 - 1)). v2's 15 dispatches per attention layer (525 per token) is reproduced exactly from shapes (9.1).
6. v2's decode pairs had no Candidate B arm; the threshold arm is B.
7. v2's P2-14 deleted the legacy chunk, block and split arms and `context_chunks_for`; qwen3.6 and dense still use them.
8. `T:proxima-model-interop/src/serving.rs:992` is the `numeric_policy` line and `:991` the `math_mode` line; v2 cited 992 for both.

Unmeasured or assumed, any of which can flip a decision: `T_kernel` and `T_model` (procedures only; the llama CPU-versus-Metal spread is a Metal-window measurement); whether `T_kernel < 1e-7` on this data; the fill constant `R = 2`, the 50 GB/s duration floor and the 0.5 s budget (all [A] until slice 8); the per-core pool and slot counts (not exposed by Metal); the register footprint and whether 512 threads fit (read from the pipeline in slice 4); whether Relaxed preserves resident-versus-split equality; the flash kernel's true bandwidth; whether GPU recovery is per dispatch or per command buffer (a peer's n = 2 measurement); the class membership of the 262 constants; the identification of the 105 broadcast ops as part of the attention chain (only the sum 525 supports it); whether `metal-horizontal-merge` refuses Q4_0/Q8_0; the qwen3.6 `G = 8` (supported only by the `(8, 256) == 3` assertion); the linearity of decode time in dispatch count; the gemma4 tokenizer fix, on which every real-text fixture depends; the stability of the IORegistry `gpu-core-count` key across macOS versions. Nothing about MLA or architectures beyond those served is claimed.

## 13. Worked example (doubles as the test, P17)

Setup: `Hkv = 1`, `G = 2`, `hd = 2`, scale `ln 2`; two cached keys and five new keys at positions 2 to 6; `k_even = 1 0 1 0 1 0 1`, `k_odd = 0 1 2 0 1 2 0`; `V(k) = (k, 6 - k)`; head 0 query `(1, 1)`, head 1 query `(0, -1)` on even rows and `(1, -1)` on odd rows. Full layer, bands `cached{MIN, MAX} new{MIN, 0}`. Re-derived this session with exact integer arithmetic (`E/exact.awk`, `E/exact.out`; the weights are powers of two) and identical to v2's table:

| row | head 0 | head 1 |
|---|---|---|
| 0 | (3/2, 9/2) | (4/7, 38/7) |
| 1 | (21/13, 57/13) | (9/8, 39/8) |
| 2 | (37/17, 65/17) | (24/13, 54/13) |
| 3 | (19/7, 23/7) | (13/7, 29/7) |
| 4 | (3, 3) | (53/18, 55/18) |

`flash_worked_example` runs the CPU sequential arm and the Metal flash lowering on this and asserts every cell within 1e-5. The tolerance is derived: llama.cpp's unfused f32 chain on its CPU backend (f1ea20621, `E/llama_worked_example_peer`) reproduces all 40 cells (this table and the windowed one) with a maximum absolute error of 7.493e-07 [M], so 1e-5 is 13 times the oracle's own deviation; `worked_example_matches_llama_fixture` vendors that check. The windowed table (bands `cached{-2, MAX} new{-2, 0}`, `W = 3`) is in `E/exact.out` and belongs to slice P3-10.
