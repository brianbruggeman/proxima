# GPU-lane evidence ledger (2026-09-03) — inputs for the plan tournament

Status tags: MEASURED (a record this session or a discipline row read this session),
READ (source opened this session by an agent with file:line), MEMORY (from the
2026-09-02 memory files; must be re-verified against main before any task relies on it),
DERIVED (computed from other rows; never a mechanism claim).

## R0. Repo state at plan time

- main == origin/main == HEAD 4be2f3a (READ, git rev-parse). Working tree clean.
- 9 commits since 2b95210 (READ, git log): qwen3.5 hybrid attention/ssm (0c3bd4f, spec.rs
  +8735/-2836), `perf(metal): bind checkpoint mapping once, address weights by offset`
  (7d09145, omega/src/metal.rs +177), `perf(interop): gather embedding rows from packed bytes`
  (e64992b), `fix(metal): gate no-copy buffer caching on resident classification` (23e2e5e),
  `fix(interop): bound the plan cache and add memory instrumentation` (ff749a0).
- omega/Cargo.toml [features] on main (READ, full file): default/std/alloc/cpu/metal/vulkan/
  cuda/npu/ane/instrument/metal-tiled-gemm/wgpu-backend. **None of the 2026-09-02 features
  (`metal-q4k-mask-fma`, `metal-wide-cooperative-reduce`, `metal-buffer-pool`,
  `metal-q4k-single-fetch`, `metal-output-placement`, `metal-packed-row-nsg2`,
  `metal-q4k-split-k`) exist on main.**
- Worktrees perf/gpu-all-wins, perf/kv-device-resident, perf/op-rule-census, perf/q4k-split-k,
  perf/output-placement, perf/attention-single-range, perf/gpu-kernel-combo,
  perf/gpu-dispatch-count are ALL at 2b95210 with 0 commits ahead of main (READ, git log
  main..branch = 0). The sealed 3.98x->3.54x headline win (MEMORY) therefore lives as
  uncommitted diffs in those worktrees or is lost. Dirty-state audit in flight.
- perf/metal-simdgroup-geometry: 2 commits (nsg=2 feature, measured LOSS -2.36%). Dead lever.
- perf/decode-orchestration-2: 1 commit (PROXIMA_ORCH_THREADS knob), timing unmeasured.
- perf/gpu-decode-ladder: 2 commits off 3076d81 (membw_probe fix + llama reference harness).
- bench/sealed-pass: 2 commits (scripts/sealed-pass.sh + quiet-gate fix). NOT on main.
- **perf/cached-attention-streaming: 42 commits ahead, based ON main 4be2f3a (current),
  authored today by another agent**: stream grouped-query attention, parallelize metal
  attention lanes, pair q4k nibbles, vectorize q4k accumulation, match ggml packed simdgroup
  geometry (then "reject packed simdgroup grouping"), drop dead resolved nodes before GPU
  dispatch, instrument packed kernel variants/codec costs, many docs rows. Diff-stat in flight.
  perf/q4k-independent-accumulators = first 12 of those same commits.
- Machine: Apple M1 Max, 32-core GPU, Metal 3 (READ, system_profiler via agent).
- Incumbent binary: /Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench,
  checkout b25346221. Model: ~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/
  openchat-3.5-1210.Q4_K_S.gguf (3.9 GB) present.
- torch 2.13.0 with MPS available in proxima-onnx/scripts/torch_reference/venv (READ).
  onnxruntime NOT installed in that venv (READ). ORT source checkout exists at
  ~/repos/others/onnxruntime.
- ai_docs/task-routes.jsonl and invariants.jsonl have ZERO tensor/omega/GPU records (READ,
  grep). Per ai_docs/AGENT.md the plan must ADD records, not bypass.

## R1. Standing board (MEMORY 2026-09-02, quiet box; re-seal in flight)

| cell | value | provenance |
|---|---|---|
| llama.cpp-Metal openchat 7B Q4_K_S decode | 17.470 ms/token = 228.9 GB/s = 407.0 GMAC/s (t=1..8 identical within 0.39%) | MEMORY, ROW 234 addendum |
| ours, main 2b95210, step_wall_ms | 67.917 (board) / 69.60 (paired OFF) -> 3.89x-3.98x | MEMORY |
| ours, three features stacked (uncommitted) | 61.85 step_wall, gpu_exec 44.8 -> 3.54x | MEMORY, worktree proxima-wt-all |
| kernel-only gpu_exec_ms, main | 56.5-57.0 | MEMORY |
| CPU orchestration per token | 11.39 ms (op_setup 4.394, block_upload 2.201, prepare 2.087, emit 0.765, encode 0.517, readback 0.297) | MEMORY ROW 234 |
| dispatches per token | ours 1196 BoundOps; llama.cpp ~483-643 | MEMORY |
| rate constants | MACs/token 7,110,402,048; weight bytes/token 3.9996 GB; neither computed in code | MEMORY |

## R2. Gap decomposition (MEMORY, 50.45 ms of 67.917 vs 17.470)

| bucket | ms | class |
|---|---|---|
| weight streaming at 228.9 GB/s | 17.5 | irreducible |
| Q4_K matmul above that rate (43.4 measured vs 17.8) | 25.6-25.9 | real work badly done |
| non-matmul GPU ops (reduce-cooperative 8.6 + elementwise 6.65) | 15.4 | mostly should not exist |
| CPU orchestration | 11.4 | should not exist (shape is identical token to token) |
| KV cache re-upload (grows with context) | ~3.3% | should not exist |

## R3. Mechanisms already traced (MEMORY unless noted; each must be re-read on main before a task cites it)

M1. Attention emits ~26 ops/layer vs ggml's 3 because the graph cannot append the new
    token's K/V into the cache tensor: online-softmax combine over two ranges
    (`spec.rs:2596-2720`, "no literal concatenation anywhere") doubled again by even/odd
    RoPE split. 37 ops/layer vs 15. -> 1196 vs ~483 dispatches. Root: no write-placement
    (`out_map`) in the RISC; `Concat` does not exist.
M2. KV cache re-uploaded every token: `LayerCache` grows by `extend_from_slice`, no-copy
    cache keyed `(pointer, byte_length)` misses on growth; `InputSizeMismatch` strict
    equality at `omega/src/metal.rs:974-976` blocks over-allocated buffers.
    `AlignedBuffer` in `proxima-tensor/src/align.rs:69` exists with zero production callers.
M3. Q4_K matvec body: explicit shift+mask+cast per element (~48 ops/8 weights) vs ggml's
    mask-without-shift + fold 1/16,1/256 into scale (~8 ops); divergent `q4k_scale_min`
    branch on `sub_block < 4`; 2x redundant load of the same 32-byte chunk per lane pair.
    mask-fma feature measured -36% on ffn_gate/up, -17.2% gpu_exec. (uncommitted)
M4. Cooperative reduce pinned to 32 threads (`SIMD_WIDTH`) for every reduction size;
    ggml rms_norm uses up to 1024 threads, float4 loads, two-level tree. Wide-reduce feature
    measured -20% on that bucket, -2.8 ms. (uncommitted)
M5. Rising marginal GB/s with simdgroup count (52 -> 147 GB/s from 256 -> 8001 simdgroups):
    low-row shapes (attn_q/k/v/o, 1024-4096 rows) starve; split-K adds simdgroups. Unbuilt.
M6. `op_setup` 4.4 ms + `prepare` 2.1 ms per token with `plan_hits=0` every token: the
    program is re-planned and re-set-up per token although its shape is identical.
    main now has `ff749a0 bound the plan cache` and `7d09145 bind checkpoint mapping once` —
    these may have moved M6; re-measure.
M7. Overlap CPU/GPU impossible: `greedy_pick` argmax depends on `waitUntilCompleted`.
    (True data dependency; the fix is to move argmax on-device, not to thread.)
M8. Rematerialization: only the `elements < 247` subset (96 nodes) is a certain win
    (1196 -> 1100); aggregate set is a loss. Unbuilt. Re-measure after M1 lands.
M9. Non-Send `MTLBuffer` blocks threaded op_setup at the type level; threading the
    plain-data half was landed (`PROXIMA_ORCH_THREADS`), timing unmeasured.
M10. `classify_kind` buckets by kernel SOURCE TEXT (`kernel.source.contains("q4k_run8(blk")`)
    — a labeling instrument that silently relabels when a kernel body changes. Compare by
    family, never by bucket.
M11. Single-range attention graph proven (488/488, 1196 -> 939 BoundOps) but requires
    in-graph write-placement, which `cpu::evaluate` lacks; driver unbuilt; KV driver
    allocation bug (34 GB from context_length default) found and fixed in worktree.

## R4. Dead levers (do not re-propose)

- nsg=2 threadgroup regrouping: -2.36% on the real graph, third time tried.
- Encoder churn: one encoder, one command buffer already.
- Per-dispatch fixed cost as the gap: Σper-op == batched within 3.6%; floor ~2.4 us.
- Threading `-t`: incumbent is GPU-bound; thread count explains zero of the gap.
- Rematerialize all <=2-consumer nodes: 6x downside on the slow ALU arm.
- Quarantine heuristic as a defect: it is cost-optimal by 5 orders of magnitude.

## R5. The "one RISC" — what exists (READ, census agent, main 4be2f3a)

### The instruction set IS one RISC (closed, small)
- `Op` (`proxima-tensor/src/op.rs:175-266`): 5 variants — Input, Elementwise, Reduce(Reduce),
  Iota, Constant. Doc header at `op.rs:166` still says "The four generators" (stale).
- `ScalarOp` (`op.rs:60-78`): 17 bodies. Associative: Add/Multiply/Maximum/Minimum (`op.rs:112-117`).
- `Reduce` (`op.rs:153-164`): `{dtype, body, init, operand, in_map, out_map, keep, name}`.
  **`out_map: IndexMap` addresses the RESULT** — data-dependent out_map = scatter. Write
  placement therefore exists on Reduce at the IR level; Elementwise has no out_map.
- `IndexMap` (`map.rs:134-152`): Affine(IndexPattern) | Computed{indices, index_map, base,
  gathered_dim}. Doc table `map.rs:8-15`: transpose/broadcast/slice/stride/conv/gather are
  patterns, not variants. `Keep` (`op.rs:142`): Reduce | Scan. `ReduceInit` (`op.rs:124`): 5.
- Zero hits for `Concat`, `Op::Pad`, `Op::Tile`, `write_placement`, `PlacedBuffer` on main.
- `BoundOp` (`bind.rs:200-215`) `{node, dtype, extents, kind}`; `BoundOpKind` (`bind.rs:221-264`)
  4 variants: Elementwise{body: ComposedBody, operands}, Reduce{element_body, reduce_op, init,
  keep, operands, output_axes, out_layout: Layout, out_scatter: Option<Lookup>}, Iota,
  Constant{value}. `Layout {base: i64, strides}` (`bind.rs:95-98`) — **an output base offset
  exists on every bound Reduce.** `MAX_INLINE_RANK=4` (`bind.rs:82`).

### The LOWERING is three unequal non-RISC emitters
- omega/src: msl.rs 4712, wgsl.rs 1929, cuda.rs 1838, metal.rs 2385, wgpu_driver.rs 872,
  backend.rs 615, sized.rs 45, error.rs 84, lib.rs 73 = 12,553 lines.
- Metal `emit` (`msl.rs:673-697`) routes BoundOpKind -> render_elementwise / render_reduce /
  render_scan / render_iota / render_constant. `render_reduce` (`msl.rs:2178-2257`) splits
  serial vs cooperative via `reduce_is_cooperative` (`msl.rs:824`, associative op + no gather).
  `push_cooperative_reduce_body` (`msl.rs:3140-3194`) splits tiled-GEMM (`tiled_gemm_block`)
  -> packed-row-blocked (`packed_row_block`) -> generic SIMD fold. Ordering load-bearing
  (`msl.rs:751-754`). **8 kernel-body shapes total on Metal.**
- `classify_kind` (`metal.rs:785-826`, instrument-only) buckets by SUBSTRING OF EMITTED MSL
  (`simdgroup_multiply_accumulate`, `q4k_run8(blk`, `simd_sum(`...). Its own doc
  (`metal.rs:777-783`) admits the routing decision "is not exposed as its own accessor".
  `diagnose_kind` (`metal.rs:835-854`) is structural but only answers packed-row-block PASS/reason.
- Shared across msl/wgsl/cuda: exactly 3 types + 2 fns (`Binding`, `PackedCodec`,
  `PackedOperands`, `gather_count`, `gather_slots`; `wgsl.rs:105`, `cuda.rs:66`). ~26 functions
  reimplemented per backend (validate, reduction_dims, bindings, grid_threads, entry_name,
  scalar_op_expr, fold_init_tokens, push_body_steps, preamble, kernel_signature, gather
  helpers, operand_read, render_*, reduce_is_cooperative, ...) ≈ 78 near-duplicates.
- Coverage asymmetry: CUDA `emit_cuda` (`cuda.rs:146-183`) REJECTS Iota and Constant
  (`CudaUnsupportedOpKind`), has serial + cooperative reduce only (no tiled-GEMM, no packed
  row-block). WGSL covers all 5 kinds but has no tiled-GEMM / packed row-block path. Only Metal
  has all 8 shapes.
- Consts: `SIMD_WIDTH=32` (`sized.rs:45`, hardware fact); msl.rs hardcodes
  `PACKED_ROWS_PER_GROUP=4` (`msl.rs:1017`), `TILE_DIM=8` (`:1030`), `TILED_GEMM_NSG=4`
  (`:1046`), codec block consts (`:294-556`). Build-time sizing exists ONLY for tiled_gemm
  (`omega-runtime.toml`: min_tokens=8, block_m=64, block_n=32, block_k=32) — the packed
  row-block geometry (rows/group, lanes/block=8 hardcoded at `msl.rs:2732` per memory) and the
  cooperative-reduce width are bare source consts -> §12 violations.
- Feature cfg sites: instrument 55, metal 19, metal-tiled-gemm 18, wgpu-backend 17, cpu 13,
  cuda 5. `vulkan`/`npu`/`ane` = name-only stubs in backend.rs.
- Metal phases (all instrument-gated): prepare `metal.rs:405-410`; block_upload `:457-491`;
  emit `:2187-2199`; pipeline_lookup `:2201-2207` (pipeline_for `:1402-1426`); op_setup
  `:2209-2220`; encode_dispatch `:2230-2248`; gpu_exec `:547-554`; readback `:2358-2378`.
  Encode loop `execute_plan` `:449-568`: one encoder, one commit, one wait; per-op
  `encode_op(&device,&encoder,&mut device_buffers,bound,packed_operands)`; retires per position.
  Diagnostic twin `execute_plan_op_timed` `:654-761` (one command buffer per op).
- Decode driver: exactly one, `run_decode_loop` (`generate.rs:1175-1181`); per-token sequence
  apply_serving_config `:1298` -> build_position_inputs `:1304` -> named_blocks assembly
  `:1313-1391` (weights + ids/eps/rope + per-layer KV blocks) -> `runtime.evaluate`
  `:1443-1449` (resolve_plan + execute_plan_named, `:927-941`) -> cache.append/advance
  `:1477-1556` -> cached_len += new_count `:1559` -> sample_next_token `:1643` -> next_ids `:1649`.
  `LayerCache {k_even, k_odd, v: Vec<f32>}` `generate.rs:621-625`.
- backend.rs doc (`backend.rs:1-52`): six Backend variants, two implemented; plan/execute
  adjudicated NOT a pipe on 2026-08-30 (relocation question failed).
- CPU sizing config exists (`proxima-tensor-runtime.toml`: parallel/cohort/quantize/transpose/
  neon/rope/staged_batch sections) — the pattern the omega config must mirror.

### What "one RISC" must therefore mean for this plan
1. ONE bound plan (`&[BoundOp]`) produced by ONE rewrite engine, identical for every backend.
2. ONE route decision, a first-class value (enum), decided BEFORE emission, recorded by a
   census keyed `(NodeId, reason)` — not recovered by grepping emitted source.
3. ONE emitter core over the 4 BoundOpKinds with backend-specific TEXT only (intrinsics,
   signature syntax); every backend covers every kind; the 8 Metal body shapes become
   instantiations of one route enum, not five hand-ordered `if let Some(..)` gates.
4. ONE sizing config (`omega-runtime.toml`) that owns every geometry constant.
5. The Llama-arch graph the model emits is the MINIMAL graph (ggml: 15 ops/layer) — the
   attention duplication is a graph defect upstream of any backend.

## R6. Today's parallel work on the same lane (READ, git)
`perf/cached-attention-streaming` (42 commits on main 4be2f3a, another agent, today): adds
`proxima-tensor/src/physical.rs` (+576, NEW MODULE), bind.rs +666, cpu.rs +244, msl.rs +202,
metal.rs +65, `benches/bench_cached_attention.rs` +212, discipline.md +938 (ROWs 234-267,
their own numbering — COLLIDES with main's ROW 234+), a `failure-cached-attention-matcher.md`.
Row titles of note: 235 "RISC bound-step seam and no-alloc floor check", 243 "GPU home-turf
comparison", 247 "acceleration attribution correction: plan preparation dominates wall gap",
248 "consumer index reduces repeated bind work", 259 "tiled GEMM feature does not reduce this
decode cell", 267 "ggml's two-SIMD-group Q4_K geometry regresses the cached-feature wall
cell" (agrees with our nsg=2 negative). Verbatim extraction in flight.

## R8. The incumbent, read at the exact checkout llama-bench runs (b25346221, READ)

- Files: `ggml/src/ggml-metal/ggml-metal.m` (6105 lines), `ggml-metal.metal` (7193), `ggml-metal-impl.h`
  (622). 106 `kernel void` functions; 26 are mul_mv variants.
- **NO FUSION at this checkout**: `grep -rln fuse ggml/src` = empty. rms_norm + mul dispatch as two
  kernels. Parity is therefore reachable WITHOUT a fusion engine; fusion is upside past parity.
- **Real ops per Llama layer at decode = 23** (`llama-model.cpp:4691-4845` + `llama-graph.cpp:464-497,
  514-616, 1071-1121, 1205-1253`): rms_norm, mul, wq, wk, wv, rope(Q), rope(K), cpy_k, cpy_v,
  mul_mat(KQ), soft_max_ext, mul_mat(V), cont, wo, add, rms_norm, mul, ffn_up, ffn_gate, silu, mul,
  ffn_down, add. Views/reshape/permute are no-ops (`ggml-metal.m:1835-1847`). Per token:
  32x23 + get_rows + rms_norm + mul + output mul_mat = **~740 dispatches**. The memory figure
  "15/layer, ~483" was WRONG; ours 1196 vs 740 = 1.62x, and single-range 939 vs 740 = 1.27x.
- KV cache: allocated ONCE on the device (`llama-kv-cache-unified.cpp:74-118`, size kv_size);
  written per token by `ggml_cpy(k_cur, ggml_view_1d(k, n_tokens*n_embd_k_gqa, row_size*head_cur))`
  (`:749-788`) — an in-place write at a byte offset into a persistent buffer; read back as
  `ggml_view_3d` (no-op). V is stored TRANSPOSED (`v_trans = !flash_attn = true`).
- Flash attention OFF by default at this checkout (`common/common.h:328`); decode attention =
  mul_mat(K,Q) -> soft_max_ext (fused scale+mask+max+exp+sum+normalize in ONE kernel,
  `ggml-metal.metal:1051-1145`, nth up to 256/ne00 via `ggml-metal.m:2501-2525`) -> mul_mat(V).
- RoPE: Llama/Mistral arch uses `kernel_rope_norm` (`llama-model.cpp:14250-14273`, NORM not NEOX),
  nth=min(1024, ne00), grid (ne01, ne02, ne03) (`ggml-metal.m:3941,4046`) — ONE dispatch per tensor.
- rms_norm: nth doubles from 32 up to min(ne00/4, maxTotalThreadsPerThreadgroup)
  (`ggml-metal.m:3797-3804`), float4 loads, simd_sum -> threadgroup shmem -> simd_sum
  (`ggml-metal.metal:1679-1721`). 4096-wide row -> 1024 threads.
- Q4_K matvec `kernel_mul_mv_q4_K_f32_impl<4,2,32>` (`ggml-metal.metal:5086-5193`): mask without
  shift (`& 0x000F/0x0F00/0x00F0/0xF000`), fold 1/256 and 1/16 into scale at combine
  (`:5171-5175`), `kmask1/2/3` branch-free scale/min extraction (`:5147-5150`), 4 rows/simdgroup,
  2 simdgroups/threadgroup, `dispatchThreadgroups((ne01+7)/8, 1, ne12*ne13)` x `(32,2,1)`
  (`ggml-metal.m:3330`, `:3215-3220`).
- Command buffers: `n_cb=1` default (`ggml-metal.m:5783`); first 128 nodes on the main thread, rest
  in one more command buffer (`:5222-5289`); no barrier/concurrency API at this checkout.

## R14. Citations verified on main after Plan A (READ, haiku pass)
- `shape.rs:469-485` `project_output_shape`: `[term] if term.coeff == 1 => Ok(iter_extents[..])`,
  else `NotLowerable { reason: "reduce output maps must be pure projections in v1" }`. **This is
  THE line placement changes** (accept `offset` on the write side). `bounds_check` `:441-467`
  already handles `axis.offset` for READ-side maps.
- `causal_mask` `spec.rs:823-845`: two `Iota{extent: Symbolic(0)}` + `Greater` ("t->st","s->st")
  + `scalar_constant(-inf)`; consumed as `(is_future, "sw->swug")` at `:2610`. A cache-tail mask
  needs an `Iota` over the bucketed cache axis vs a `cached_len` scalar — same construction.
- `grid_threads` `msl.rs:1517-1560` (tiled -> packed `output_total.div_ceil(4)*32` -> cooperative
  `output_total*32` -> serial `output_total`); `kernel_cache_key` `:731`, `kernel_dispatch_shape`
  `:797`, `packed_row_block` `:1235`, `tiled_gemm_block` `:1450`, `diagnose_packed_row_block`
  `:1487`, `tiled_gemm_threadgroup_width` `:3118`; cooperative body `output_index = gid/32`,
  `lane = gid%32` at `:3190-3194`; packed loop `ib += SIMD_WIDTH/lanes_per_block` at `:2526-2528`.
- `generate.rs:1655` `token_breakdown` println (fields incl. `kv_cache_upload_bytes`,
  `greedy_pick_ms`); `:1721-1750` `token_breakdown_metal` println (fields incl. `plan_hits`,
  `plan_misses`, `nocopy_reuses`, `mapping_offset_uploads`, `device_allocated_bytes`).
- `bind.rs:2719` `PROXIMA_MAX_TOKENS`; `:2797-2803` greedy oracle `2651`/`"known"` (llama.cpp's
  captured answer, §14); `:3002` metal decode harness; `:3084` per-op profile harness.
- `metal.rs:1616` `is_page_aligned`, `:1879` `upload_block_no_copy`, `:1903`
  `upload_block_no_copy_uncached`, `:1914` `create_no_copy_buffer`.
- `instrument.rs:842` `WIDTH_TILE_DECLINE: Mutex<BTreeMap<(u32, WidthDeclineReason), ..>>`.
- `QuantizedBlock` `cpu.rs:3084-3110`: Float32 | Q4K | Q5K | Q6K | Q8_0 | Q4_0 | (F16/BF16).
- omega examples: attention_tiled_gemm_probe, membw_probe, q4k_matvec_probe,
  real_forward_emit_probe, real_forward_packed_probe, resident_gemv_topk (+baseline.json).
- scripts: `omega-gate.sh`, `proxima-tensor-gate.sh`, `onnx_reference/{bench.py,run.sh,...}`
  (`bench.py:96` hardcodes `providers=["CPUExecutionProvider"]`); NO `sealed-pass.sh`, NO
  `llama_reference/` on main. `torch_reference/inference_bench.py:29-32` has only `--threads`
  and `--runs`; `train_bench.py` exists.

## R18. Host facts verified by Plan B3 and re-checked (READ)
- **`flock(1)` does not exist on this Mac** (`which flock` → not found; `/opt/homebrew/bin/flock`
  absent). Every `flock ... -c '...'` weld in synthesis_2 would fail. Options: `brew install flock`
  (the `discoteq/flock` port) or a repo-local shim `scripts/gpu-measure-lock.sh` using python's
  `fcntl.flock` then `os.execvp`. The mutex card is real work, and it is card ONE.
- `ServingConfig::context_length` default = **131,072** (`serving.rs:161`); × 262,144 B/token =
  **34,359,738,368 B** — the 34 GB trap reproduced exactly from source. Capacity must be a
  build-time key with a build-time byte assertion, never derived from `context_length`.
- `Uniforms` (`msl.rs:2207-2218`) already carries per-dispatch `output_total`, `reduction_total`,
  `output_extents[]`, `reduction_extents[]`, `operand_base[]`, `operand_strides[][]`, `out_base`,
  `out_strides[]` — extents are ALREADY per-dispatch uniforms; `pipeline_lookup` is 0.04 ms
  (R13) so pipelines already survive cached_len changes. What is rebuilt per token is bind +
  retirement + packed-operand resolution. A shape-invariant plan (split Plan into invariant +
  per-token parts) is therefore cheaper than it looks and is the fallback if bucketing loses.
- `membw_probe.rs` times the whole `execute_plan` with `Instant::now()` (`:165-166`) — upload,
  commit, wait AND readback inside the window; the batched path `metal.rs:546-554` uses host
  ticks; only the op-timed path (`:734`) uses `GPUStartTime/GPUEndTime`.
- `checkpoint_mapping_offset` (`metal.rs:1786-1815`) doc: "the scratch and KV-cache buffers ...
  never live inside the checkpoint's own mmap, so they fall through unchanged" — generalising the
  single `CHECKPOINT_MAPPING` slot to N registered host spans makes a page-aligned KV arena
  addressable by offset through the SAME primitive (§1: extend, don't add a peer).
- `packed_operands_of` lives in omega (`metal.rs:375`) while `QuantizedBlock` lives in
  proxima-tensor (`cpu.rs:3084`) — moving the packed-set computation into proxima-tensor next to
  the type is how `bind` can own the packed layout.
- Two mutually-exclusive cargo features + `compile_error!` would turn `omega-gate.sh [2/6]`
  (`--all-features`) RED; a build-time PROFILE axis (`[q4k] body = "..."` → `cargo:rustc-cfg`)
  selects exactly one body under any feature set (guiding-principles §8 profile input).

## R17. Facts verified by the round-3 critique (READ)
- `ScalarOp` (`op.rs:60-78`) has `Greater` and `Equal` but NO `GreaterEqual`; its doc `op.rs:51-53`
  says it is "the one closed set in this crate that stays closed". A tail mask must compose
  `Greater(cached_len_leaf, iota)` (keep where iota < cached_len) with the existing Select/-inf.
- `finish`'s readback (`metal.rs:2355-2378`) carries the invariant at `:2360-2364`: "an output
  node's buffer is always freshly allocated by encode_op at offset 0 -- only a weight INPUT can
  carry a nonzero offset ... reading from the buffer's own start is always correct here." Any
  arena that sub-allocates within one buffer breaks it; whole-buffer sharing is safe; outputs
  are excluded from retirement by `bound_op_retirement` (`metal.rs:1128-1147`, `!outputs.contains`).
- `execute_plan`'s retire loop `metal.rs:541-543` removes from `BTreeMap<NodeId, DeviceBuffer>`;
  a retire no-op makes every operand lookup walk ~1196 entries.
- `struct Prepared` is private (`metal.rs:859`, `resolved` `:864`; `cpu.rs:280`); `Plan` fields
  private (`:313-325`) — a fingerprint test needs pub accessors, cannot live in `omega/tests/` alone.
- `execute_plan_op_timed` has its OWN block-upload loop (`metal.rs:664-690`) separate from
  `execute_plan`'s `:457-491`; an instrument fix must cover both. `op_profile_family`
  (`generate.rs:184`) exists ONLY in per-op mode (`Vec<OpGpuTiming>` from `:654-690`).
- `align.rs:56-58`: page_size must be "a real host page size the caller queried itself (e.g.
  omega::metal::page_size), never hard-coded"; `page_size()` exists only at `metal.rs:1606`.
  proxima-tensor already has `dep:libc` under `std` — `libc::sysconf(_SC_PAGESIZE)` is the
  non-Metal source.
- KV row bytes per token: k_even 8x64x4 = 2048 B, k_odd 2048 B, v 8x128x4 = 4096 B per layer; x32
  = 262,144 B/token (matches R13). `capacity x 2048` is a 16 KiB multiple iff capacity % 8 == 0.
- `Route::Declined(reason)` is data-carrying: `route as usize` is E0605; needs `fn slot(&self)`.
- `Counter` is non-Copy (`counter.rs:12-17`): `[Counter; N]` needs N explicit const initializers.
- `build_position_inputs` defined `generate.rs:799-827`; uses `start_position` only at `:813`
  for cos/sin (extent = symbol 0); called with the TRUE `cached_len` at `:1304-1309`;
  `apply_serving_config(.., cached_len + new_count)` `:1298` — bucketing symbol 1 does NOT touch
  RoPE positions or `is_future`.
- The KV roots are program OUTPUTS today (`generate.rs:1393-1400`, 97 effective outputs), so
  `bound_op_retirement` never retires them; turning K/V writes into in-place scatters shrinks the
  output set to ~1 and changes every liveness partition.
- `grep -rn write_row` = 0 relevant hits: an injectivity-by-name convention has no enforcement;
  a scatter whose `indices` is a host-fed `Op::Input` is UNPROVABLE at bind; `Iota(coeff 1) +
  loop-invariant scalar` IS provable (the `causal_mask` construction, `spec.rs:823-845`).
- `bash scripts/omega-gate.sh` steps [2/6]/[3/6] build and run `--all-features` (Metal tests
  included) — GPU work that must also take the measurer lock.

## R16. A SECOND REWRITE OF THE BOUND PLAN, Metal-only (READ, round-2 critique, verified)
`omega/src/metal.rs:1003` `let mut resolved = bind(program, &shapes, &effective_outputs)?;` then
`:1013` `correct_packed_matmul_layouts(&mut resolved, &packed_operands.keys()...)` — imported from
proxima-tensor (`metal.rs:191`), defined `bind.rs:1618-1647+`: "Rewrites a packed matmul weight
operand's Layout from layout_of's default -- row-major over the operand's DECLARED axis order --
to the layout its packed bytes actually have on disk ... `layout_of` has no way to get this right
on its own." The CPU path (`cpu.rs:358` calls `bind::bind`) does not apply it because the CPU
quantized path never reads packed bytes through `layout_of`. **So the bound plan Metal executes
is NOT the plan CPU executes, on main, today** — brief one-RISC item 1 is false as-is, and the
fix is to make `bind` produce the correct layout for packed operands (bind takes the packed set,
or `layout_of` reads a per-operand physical layout) so no backend rewrites the plan after bind.
Card: "one bound plan" test must capture the plan AFTER the driver's own rewrite, per backend.
Also from the round-2 critique, verified: the census pattern `WIDTH_TILE_DECLINE` is a
`Mutex<BTreeMap>` (`instrument.rs:842`, `.lock()` at `:857`) while its proposed neighbour
`ENCODE_DISPATCH_CALLS` (`metal.rs:2243`) is an atomic `proxima_telemetry::Counter` (`:1484`) —
a per-dispatch census must be lock-free (fixed-size atomic array indexed by route, or a
per-plan preallocated Vec<u8> of routes filled once at plan time since the route is a property
of the plan, not of the dispatch). `block_node_ids` (`metal.rs:1111-1118`) is program-order;
`resolve_named_blocks` (`:584`) reorders name→node; `InputCountMismatch` at `:984-990` /
`cpu.rs:337-343`. `build_position_inputs` `generate.rs:1305-1310`; `named_blocks`
`Vec::with_capacity(.. + 3 + layer_caches.len()*3)` `:1313-1318`; three named entry-point call
sites `:1436, :1446, :1455`. `bound_op_retirement` called at `metal.rs:1013±`; `plan_named`
`:578-585`, `execute_plan_named` `:593`, `execute_plan_named_op_timed` `:769`.
`Q4K_UNPACK_MSL`/`Q5K`/`Q6K` concatenated at `msl.rs:1978-1982` with no delimiter (a
"grep the Q4_K region" tie-break is undecidable on emitted source).

## R15. Plan-B / synthesis citations verified on main (READ, haiku pass 2)
- `u.out_base` emitted at `msl.rs:2361, 2734, 3079, 3460, 3541`; `long out_base` uniform field `:2216, 3502`.
- `EmitError::ScatterNotSupported` raised at `msl.rs:933`, `wgsl.rs:364`, `cuda.rs:241`; defined `error.rs:53`.
- CPU implements scatter: `run_reduce_scatter` `cpu.rs:6911`; `run_reduce` `:7638`.
- `IndexMap::scatter` `map.rs:175`, `scatter_extent` `:209`, `as_gather_from_output` `:238`;
  write-direction convention doc `map.rs:110-131` ("`offset` carries the destination axis's
  static extent"; "CPU interpreter runs the reduce loop strictly sequentially, so a scatter
  never needs atomics").
- `UNIFORM_BUFFER_REUSES` `metal.rs:2069`, `upload_uniforms` `:2070` (reuse path `:2075`).
- `layout_of` `bind.rs:1594-1606`: `base += i64::from(axis.offset) * stride` — the read-side
  offset ALREADY folds into `Layout.base`; `build_scatter_out_layout` `:1011`.
- `emit_is_deterministic_byte_equal` test `msl.rs:4656` (seed for a golden-source test).
- `omega/build.rs`: `require_nonzero :16`, `require_multiple_of_sixteen :35`,
  `require_divides_q4k_block :43`, `require_multiple_of_eight :59`, `get_int :67`, `resolve_int
  :79` (env `OMEGA_{SECTION}_{KEY}` + `rerun-if-env-changed` `:85`), `emit_sizing_consts :105`.
- `scripts/omega-gate.sh`: [1/6] `--no-default-features --features alloc` build, [2/6]
  `--all-targets --all-features` build, [3/6] nextest `--all-features` with `ran_count` asserted
  nonzero (not a specific N), [4/6] clippy pedantic.
- `GPU_LAYERS_ALL = -1` `serving.rs:55`; default `gpu_layers: GPU_LAYERS_ALL` `:168`;
  `select_backend` `generate.rs:855-861`.
- proxima-model-interop features: default=[] / std / interop-bgpool / instrument
  (passthrough `omega?/instrument`) / metal (`dep:omega`,`std`) / metal-tiled-gemm
  (`omega?/metal-tiled-gemm`) — the passthrough PATTERN every new omega feature must follow.
  proxima-tensor features: default=[std,config,q4k-int8-dot,q5k-int8-dot,q6k-int8-dot] plus
  alloc/config/ggml-bench/dynamic-elision-probe/epilogue-profile-probe/instrument/tensor-bgpool/...
- `classify_kind` called at `metal.rs:709` (op-timed path) with `diagnose_kind` `:710`.
- `operand_bytes` computed `metal.rs:694` (sums bound BUFFER lengths), consumed
  `generate.rs:109-212` (total, per-bucket, per-op top, per-family, pass/reject).
- `greedy_pick_started/ticks` `generate.rs:1640-1670`.
- `sealed-pass.sh` (branch only): `REPO_ROOT` hardcoded to `proxima-wt-seal` (:4), four sibling
  worktrees hardcoded (:25-28), quiet gate uses `pgrep -x` on exact names, constants
  `MACS_PER_TOKEN=7110402048`, `WEIGHT_BYTES_PER_TOKEN_GB=3.9996`.
- Branches that EXIST (do not reuse): bench/{bge-real-traffic,sealed-pass,torch-arms},
  feat/{backend-selectable,qwen3-model-interop,tensor-consolidated}, docs/cached-attention-rereview,
  perf/{amx-width-tile,arena-hit-cost,attention-single-range,attention-tile,batch-regression,
  batch-seal,bge-integration,bind-weights-once,cached-attention-streaming,composition-split,
  const-from-program,decode-orchestration,decode-orchestration-2,gpu-all-wins,gpu-decode-ladder,
  gpu-dispatch-count,gpu-kernel-combo,kernel-latency,kv-device-resident,matmul-geometry,
  metal-simdgroup-geometry,narrow-tile,one-path,op-count,op-rule-census,output-placement,
  plan-cache,q4k-independent-accumulators,q4k-orchestration,q4k-split-k,rebind-identity,
  resolve-once,route-census,seal-fast-path,three-axis-merge,train-parity,transposed-a-gemm,
  unify-arena-fusion,width-gate-decline,width-tile-accs,zero-copy-weights}.
- spec.rs layer builders: `append_mistral_layer :856`, `append_mistral_moe_layer :1663`,
  `append_mistral_cached_layer :2336` (sole caller `:6282`), `append_qwen35_dense_attention_layer
  :2912`, `append_mistral_cached_moe_layer :3455`, `append_qwen35_delta_net_step :4347`,
  `append_qwen35_conv_branch :4778`, `append_qwen35_ssm_mixer :4924`.

## R13. TODAY'S SEALED CELL on main 4be2f3a (MEASURED 2026-09-03, box LOADED: load 4.7-5.7, cdb-daemon resident; arms interleaved A B A B A B; raw logs scratchpad/baseline-2026-09-03/{llama_A1..3,ours_B1..3,profile}.log)

| arm | per-run | mean | CoV (across runs) | ms/token | ratio |
|---|---|---|---|---|---|
| llama.cpp-Metal `llama-bench -n 32 -p 0 -r 5 -t 8 -ngl 99` tg32 t/s | 56.70±0.35 / 56.89±0.54 / 57.66±0.14 | **57.08 t/s** | 0.89% | **17.52** | 1x |
| ours `step_wall_ms` steps 1-7 (7 per run) | 67.55 / 67.96 / 68.25 | **67.92** | 0.5% | 67.92 | **3.88x** |
| ours `gpu_exec_ms` steps 1-7 | 57.07 / 56.49 / 57.22 | **56.93** | 0.7% | | 3.25x kernel-only |
generated_text identical all three: "Here is a simple Python function that returns". op_count 1196.
Per-phase means (ms/token, steps 1-7): prepare **1.97** | emit 0.81 | block_upload **2.0** (0.4 on
step 2, 1.7-3.5 otherwise) | op_setup **3.9** | pipeline_lookup 0.04 | encode_dispatch 0.47 |
readback 0.22 | Σ ≈ 9.4; wall − gpu_exec = 11.0 → ~1.6 ms residual (sampling, cache append).
**`plan_hits=0 plan_misses=8` in `metal_decode_summary` every run — P4 CONFIRMED on today's
main.** CORRECTION: an earlier draft of this row claimed a per-step `plan_hits=10,20..70`
field; that was a sed backreference artifact (`\10` = group 1 + literal "0") in the extraction
script, not data. There is exactly ONE `plan_hits` field (`generate.rs:905`, incremented
`:968`, printed at `:1764` and `bind.rs:3046` from the same `runtime.plan_hits`); it is 0 on
every step. The harness ASSERTS `plan_hits == 0` at `bind.rs:3052-3055` ("cached_len grows
every decode step, so no (new_count, cached_len) shape can repeat") — any card that makes the
plan hit must invert that assertion in the same change.
Per-op profile step 3 (diagnostic, one command buffer per op): Σ 61.082 ms over 1196 ops vs
batched 56.93 → 7.3% excess → admissible.
| bucket | ops | gpu_ms | ns/op |
| reduce-packed-row-blocked (Q4_K/Q6_K matvec) | 225 | **44.450** | 197,556 |
| reduce-cooperative | 385 | 9.113 | 23,670 |
| elementwise | 547 | 7.350 | 13,437 |
| constant / iota (degenerate control, 0 bytes) | 37 / 2 | 0.161 / 0.008 | 4,352 / 4,208 |
| family | ops | gpu_ms | true bytes/op | GB/s |
| ffn_up | 32 | 10.862 | 33.05 MB | 97.4 |
| ffn_gate | 32 | 10.843 | 33.05 MB | 97.5 |
| ffn_down | 32 | 10.005 | 34.00 MB | 108.7 |
| attn_q | 32 | 5.172 | 9.45 MB | 58.5 |
| attn_output | 32 | 3.846 | 9.45 MB | 78.6 |
| attn_v / attn_k | 32 / 32 | 1.523 / 1.462 | 2.4 MB | 50-53 |
| output.weight (Q6_K) | 1 | 0.737 | 107.5 MB | 145.9 |
| "(no named operand)" (attention/norm/elementwise) | 681 | 11.382 | — | — |
| kv_cache.v / k_odd / k_even (cached-range attention reduces) | 32 each | 1.559 / 0.783 / 0.774 | | |
| rope_cos / rope_sin | 64 each | 0.779 / 0.762 | | |
| eps (rms-norm) | 65 | 0.571 | | |
Gap decomposition, today (67.92 − 17.52 = 50.40 ms): weight stream at 228 GB/s ≈ 17.5
irreducible; Q4_K matvec above that rate ≈ **26.9**; non-matmul GPU ≈ **16.6** (9.1 coop + 7.35
elementwise + 0.17); orchestration ≈ **11.0**. Sums to 54.5 vs 50.4 (per-op mode inflation).
**INSTRUMENT DEFECT FOUND:** `operand_bytes` per matvec op reports 4,140,417,024 = the whole
checkpoint mapping buffer (7d09145 addresses weights by offset into ONE buffer), so
`gpu_ns_per_byte` and `total_operand_bytes` (1.2 TB!) are wrong; the profile must report the
tensor's byte length, not the bound buffer's. True bytes above are derived from shapes
(rows*k*0.5625). Card: fix `op_profile` byte accounting before any GB/s row is written.
The MEMORY numbers (R1-R3) are CONFIRMED within 1-3% by this cell; the diagnosis stands on
today's main.
MEMORY per step (3 runs): task RSS 310-357 MB at prefill, 48-66 MB steady, no monotonic trend;
`device_allocated_bytes` 4.299-4.305 GB at prefill, 4.152-4.163 GB steady (+1-2 MB/token);
`plan_cache_len=1` every step; `kv_cache_upload_bytes` 8.13 -> 9.70 MB over steps 1-7
(+262,144 B/token, linear in context = M2'). No leak in this cell. SECOND INSTRUMENT DEFECT:
`block_upload_bytes` = 4,147,777,096 per steady token = the mapping-offset BINDING of the whole
checkpoint (`mapping_offset_uploads=291`, `copying_uploads=4`), not copied bytes. Owner rule
2026-09-03: memory is a kill criterion on every card ("if you've fucked up memory again, you
lose"); prior failures = 34 GB KV allocation from context_length default (worktree only) and the
plan-cache heap growth (bounded by ff749a0 on main).
CPU vs Metal in OUR stack (READ, discipline.md:9013, ROW 100): decode ties — ours-cpu-w8 66.647
vs ours-metal 68.976 ms/token; llama.cpp CPU -ngl 0 t=8 = 39.25 (MEMORY thread sweep). Metal
should be ~3x CPU on a bandwidth-bound decode and is not — because the Q4_K body is ALU-bound
at 97 GB/s and the non-matmul ops run 32-wide.

## R12. The parallel branch `perf/cached-attention-streaming`, read via git show (READ)

What it built (default-off feature `cached-attention-streaming` in proxima-tensor, omega,
proxima-model-interop; `libm` dep added; `physical.rs` +576):
- **A NEW `BoundOpKind::CachedAttention` variant** (ROW 235, CPU arm `cpu.rs:19141-19190`;
  `render_cached_attention` in `msl.rs:104` emits an "eight-input online-softmax kernel").
  A post-bind STRUCTURAL matcher in bind.rs (`cached_attention_candidates` `bind.rs:+268`,
  `attention_score_sources`, `is_exact_causal_mask`, `removable_attention_dependencies`) that
  pattern-matches the two-range online-softmax cluster and replaces it with the macro-op.
  Their own failure record (`failure-cached-attention-matcher.md`) says the BoundOp-only matcher
  was abandoned as "a heuristic" that "cannot prove the semantic roles", then superseded by a
  structural matcher that "passes positive synthetic and two-layer fixture tests".
  **This is a fifth bound kind for one model's attention shape** — the exact "arbitrary rule
  for a specific instance" the workspace AGENTS.md forbids, and NOT the generic band-streaming
  law the rewrite-algebra spec (§8-9) names. It must be adjudicated, not merged by momentum.
- `prune_dead` / `dead_resolved_nodes` in bind.rs (216d925 "drop dead resolved nodes before
  GPU dispatch") — generic, RISC-conformant, keep.
- A paired-nibble Q4_K body (`q4k_pair_dot`, ROW 257): GPU family 47.8 -> 33.9 ms (-29%), parity
  3.1e-6 vs f32 on real `blk.0.attn_q.weight`. **Independent re-derivation of the same ggml
  mechanism as the uncommitted `metal-q4k-mask-fma` (-36% on ffn).** Two bodies, one mechanism,
  two worktrees. ONE must be chosen.
- float4 accumulation, explicit unroll, width-64 dispatch, row-batch 8, tiled-gemm for decode,
  ggml nsg=2 geometry: ALL measured negatives (ROWs 249, 251-254, 259-260, 265-267). nsg=2 is
  now a FOURTH-time negative across both lanes.
- Classifier defect found and fixed on the branch (ROW 263): `classify_kind` mislabeled the
  paired body as `reduce-cooperative` (9/601 -> 225/385 after fix) — the same substring trap
  M10 names; their fix adds a second marker string rather than a first-class route value.

Their numbers (feature on, 24-token protocol, quiet-ish, within-process CoV):
| cell | wall ms/tok | GPU ms/tok | dispatches | source |
|---|---|---|---|---|
| main-equivalent feature-off control | 51.571 (CoV 1.75%) | 35.117 (2.00%) | 1194 | ROW 267 |
| feature-on, consumer index, paired Q4_K | 51.535 (2.16%) | 39.841 (0.98%) | 616 | ROW 262 |
| llama.cpp exact-prompt llama-cli, 8 tokens | **17.467** (0.363%) | unreported | ~740 | ROW 250 |
| ratio | **2.95x** | — | | |
Note the feature-off control at 51.6 wall / 35.1 GPU ALREADY carries the paired Q4_K body (it is
in the same tree) — so Q4_K body alone took main's ~57 GPU -> ~35 GPU. And feature ON is NOT
faster on wall (51.5 vs 51.6) despite 616 vs 1194 dispatches: **halving the dispatch count moved
nothing**, consistent with H2's refutation (the GPU is busy, not gapped). ROW 247: `prepare`
150.7 ms/token with the matcher before indexing, 11.6 after — the matcher itself is a per-token
CPU cost because plan_hits=0 (M6').
Their ROW 264 (physically at line 5891, mixed with an OLD block_upload analysis at 5910-5988
that predates 7d09145 — "381 of 391 blocks copy 5.84 GB per token" is a STALE finding; main now
addresses weights by mapping offset): Q4_K = 217 ops 42.6 ms, Q5_K 8 ops 1.6, Q6_K 1 op 2.4,
other 969 ops 27.3 ms in diagnostic mode.
Row numbering: their ROW 234-267 collide with the three unlanded "ROW 234"s and with the
physical order (263/264 at line 5870). Renumber at land.

## R11. Mechanisms bound to CURRENT main (READ, 4be2f3a) — supersedes the MEMORY rows in R3

M1' (attention duplication, graph-level): `append_mistral_cached_layer` `spec.rs:2336-2865`
  (530 lines, 25 args). Doc `spec.rs:2303-2319`: "Two Reduce blocks — one per source — combine
  through online-softmax arithmetic ... rather than a literal concatenation: **`Reduce::out_map`
  must stay a pure projection (`shape::project_output_shape`'s own doc), so nothing upstream of a
  reduce can splice two tensors into one axis.**" Comment `spec.rs:2616-2617`. THE IR CONSTRAINT IS
  NAMED: out_map is a pure projection (no offset). `IndexPattern` already carries `offset` on the
  READ side (`map.rs:12`: slice = non-zero offset). Allowing a non-zero base on the WRITE side
  (out_layout.base already exists in BoundOpKind::Reduce, `bind.rs:95-98`) plus multiple
  producers writing disjoint ranges of one caller-owned buffer IS concat/placement — no new Op.
M2' (KV re-upload): `LayerCache` `generate.rs:621-654`, `append` = 3x `extend_from_slice`
  (`:636-640`), `named_blocks` hands the whole Vec as `QuantizedBlock::Float32` (`:642-653`).
  `NOCOPY_BUFFERS: BTreeMap<(usize,usize), MetalBuffer>` keyed (pointer, byte_length)
  (`metal.rs:1848-1849`), lookup `upload_block_no_copy` `:1879-1892`. 23e2e5e now routes
  NON-resident blocks to `upload_block_no_copy_uncached` — so KV blocks (not in resident_names)
  create a fresh no-copy buffer EVERY token (no cache growth, but no reuse either).
  `mark_resident` `metal.rs:350-362` classifies by NAME. Strict `found != expected` at
  `metal.rs:991-1000` and `cpu.rs:346-356`. `AlignedBuffer` `align.rs:42-46`, `new` at `:69`,
  ZERO production callers (example `omega/examples/resident_gemv_topk.rs:275`, test only).
M6' (per-token re-plan, ROOT CAUSE FOUND): plan cache key = `(symbols[0], symbols[1])` =
  `(new_count, cached_len)` (`generate.rs:966`); `cached_len` is `Extent::Symbolic(1)` on every
  KV input leaf (`spec.rs:6216-6245`). **cached_len changes every token, so the key misses every
  token by construction** — `plan_hits=0` is not a bug in the cache, it is the shape symbol.
  ff749a0 added `self.plans.clear()` on every miss (`:973`) to stop the leak, so now EVERY token
  pays `plan_named` (= infer + bind + packed_operands + retirement, `metal.rs:945`) = the 2.087 ms
  `prepare` slice, AND `mark_resident`, AND fresh `Plan`. The incumbent avoids this by padding
  n_kv to a 256 multiple and masking (graph shape changes once per 256 tokens). Two RISC-conformant
  fixes: (a) bucket cached_len to a capacity multiple + mask the tail (zero IR change; hits
  255/256); (b) make the reduce extent over the cache a runtime uniform (BoundOp extents are
  baked `Vec<u64>` today, `bind.rs:200-215` — bigger change).
M6'' (per-op setup): `encode_op` `metal.rs:2179-2252` per op per token: `kernel_cache_key` +
  `kernel_dispatch_shape` (`:2193-2194`), `pipeline_for` (cached, `:1402`), `allocate_buffer`
  for the OUTPUT (`:2210`), `upload_uniforms` (`:2211`), fault buffer, bind, dispatch. Output
  buffers and uniform buffers are allocated fresh per op per token = 1196 `newBufferWithLength`
  + 1196 uniform uploads per token. That is the 4.4 ms `op_setup`. A plan-stable program can
  preallocate every output buffer and uniform once (the uncommitted `metal-buffer-pool`).
M7' 7d09145 landed `register_checkpoint_mapping` (`backend.rs:402-414`, `metal.rs:1744-1815`):
  ONE no-copy buffer for the whole mmap'd checkpoint, weights addressed by `(buffer, offset)`.
  So block_upload for WEIGHTS is now offset arithmetic; only KV/activation inputs still upload.
Census pattern to mirror: `WidthDeclineReason` 8 variants `instrument.rs:809-828`, `Path`
  `:1430-1435`, `record_width_tile_decline(node, reason, m,k,n,stride_a,stride_b)` `:848-864`
  keyed `(node.0, reason)`. `FuseSite`/`FuseDeclineReason` DO NOT EXIST on main (memory was
  from the uncommitted census worktree). Module gated at `lib.rs:213-214`; call sites gated.
`select_backend` picks Metal only at `gpu_layers == GPU_LAYERS_ALL` (ROW 223).

## R10. The discipline log on main (READ, discipline.md 18766 lines, rooflines.md 776)
- Last row on main is **ROW 233** (line 18736). ROWs 234-237 from the 2026-09-02 GPU/train session
  exist only on branches/worktrees; the strings `3.54x`, `17.470`, `228.9`, `metal-q4k-*`,
  `kv-device-resident`, `single-range`, `PlacedBuffer`, `RISC` have ZERO hits in main's log.
  **Main's log does not know the 2026-09-02 GPU session happened.**
- GPU rows on main: ROW 193 (lines 17258-17338: first per-shape GPU attribution, llama.cpp 58.53 t/s
  = 17.086 ms/tok, ours 69.86 full / 57.032 gpu_exec, 4.09x / 3.34x, loaded box), ROW 221
  (18390-18430: encoder churn REFUTED, kernel 83.2% of step, op_setup 4.46 ms, 1196 dispatches,
  llama.cpp 20.589 ms/tok on a loaded box), ROW 223 (18460-18487: `width_tile_plan` per-head
  weight collapse; Metal was correct, CPU fused path was wrong; `select_backend` picks Metal only
  at `gpu_layers == GPU_LAYERS_ALL`).
- rooflines.md GPU lane (lines 396-479): candidate ceiling = **DEBT, not measured**; only ratio
  available is vs incumbent achieved (416.1 GMAC/s). Summary table row line 751. The doc's own
  closing note (766-773) says the GPU lane's ratio "is not a gap-to-machine at all".
- Machine constants that exist: CPU streaming triad 69.95 / 81.21 GB/s (ROW 176). No GPU constant.
- Row numbering already non-monotonic (ROW 205 at line 17864 precedes ROW 204 at 17950) from
  concurrent worktrees; three unlanded branches each carry their own "ROW 234"; the
  cached-attention branch carries ROW 234-267. Assign row numbers at LAND time, from main.

## R9. Non-decode GPU arms (READ)
- The ONLY GPU bench outside decode: `omega/benches/metal_vs_cpu.rs` (gemm_square_f32 512/1024/2048;
  matvec_batch1_f32 at Mistral f32 shapes), registered `omega/Cargo.toml:207-210`, doc says UNRUN.
- mnist f32 inference, BGE-small, MLP train step: ZERO Metal/wgpu bench or timed-test arms. GPU
  train step exists only as parity tests in `omega/tests/training_step_parity.rs:400-607` (untimed).
- `rooflines.md:29`: the only GPU lane tracked is q4_K decode, marked "stale — not re-measured".
- `membw_probe.rs` GPU bandwidth ceiling = DEBT (`rooflines.md:411`).
- So "llama, ggml, ort and torch can beat us on gpu" is MEASURED only for llama.cpp-Metal; for
  ORT (CoreML EP) and torch (MPS) there is NO CELL on either side. Those are plan tasks.

## R7. Uncommitted 2026-09-02 work, by worktree (READ, git status/diff --stat)
| worktree | branch | dirty | features present in omega/Cargo.toml |
|---|---|---|---|
| proxima-wt-all | perf/gpu-all-wins | 13 files +3859/-197, 6 untracked | tiled-gemm, packed-row-nsg2, q4k-mask-fma, q4k-single-fetch, wide-cooperative-reduce, buffer-pool, output-placement |
| proxima-wt-drive | perf/kv-device-resident | 7 files +2375/-96 | tiled-gemm, output-placement |
| proxima-wt-rules | perf/op-rule-census | 3 files +842/-9 | (none new) |
| proxima-wt-splitk | perf/q4k-split-k | 5 files +352/-48 | metal-q4k-split-k |
| proxima-wt-place | perf/output-placement | 3 files +322/-14 | output-placement |
| proxima-wt-merge | perf/attention-single-range | 1 file +1181/-82 (spec.rs) | (none) |
| proxima-wt-gpudisp | perf/gpu-dispatch-count | 4 files +244/-28 | wide-cooperative-reduce |
| proxima-wt-gpuker | perf/metal-simdgroup-geometry @bfc150d | 2 files +237/-41 | packed-row-nsg2, q4k-mask-fma |
| proxima-wt-q4k | perf/q4k-orchestration @a2175c2 | 3 files +589/-182 | q4k-single-fetch |
| proxima-wt-lat | perf/kernel-latency @14f1304 | 5 files +352/-48 | q4k-split-k |
All based on 2b95210; main has since moved 9 commits including spec.rs +8735/-2836 (qwen3.5)
and metal.rs changes (7d09145, 23e2e5e) — every one of these diffs will conflict on rebase.
**The measured -17.2% (mask-fma), -4.9% (wide-reduce), -11.1% stacked headline are
UNCOMMITTED and UNREBASED.**
