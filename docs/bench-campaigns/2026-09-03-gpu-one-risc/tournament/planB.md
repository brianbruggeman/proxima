I have verified the ledger against main at `4be2f3a` and found several claims that sharpen or reorder it. Plan follows.

---

# GPU parity plan — proxima-tensor through omega, ONE RISC

## Diagnosis (one paragraph, ledger-cited)

The GPU lane's mass is **not** where the brief's ordering puts it, and the repo says so. R12's own control cell shows the parallel branch halved dispatches (1194 → 616) and wall moved 51.571 → 51.535 ms/token while `gpu_exec` got *worse* (35.117 → 39.841) — dispatch count is not the denominator. What did move: that branch's feature-OFF control already carries a paired-nibble Q4_K body and reads ~35 ms GPU against main's 56.5–57.0 (R1, R12 note) — i.e. **the Q4_K kernel body is the single largest removable block** (R2: 25.6–25.9 ms of the 50.45 ms gap), and it has been independently re-derived twice (R3-M3 `metal-q4k-mask-fma` −17.2% `gpu_exec`; R12 ROW 257 `q4k_pair_dot` −29% on the GPU family), both uncommitted (R7). Second is the cooperative reduce pinned to 32 threads for every reduction size against the incumbent's up-to-1024 (R3-M4, R8 `ggml-metal.m:3797-3804`), −20% of a 15.4 ms bucket (R2). Third is CPU orchestration at 11.4 ms (R1), whose root cause is now *proven* rather than suspected: I re-read `resolve_plan` at `proxima-model-interop/src/generate.rs:958-975` and the cache key is `(symbols[0], symbols[1])` = `(new_count, cached_len)` with `cached_len` an `Extent::Symbolic(1)` on every KV leaf (R11-M6′), so `plan_hits=0` is structural, and `ff749a0`'s `self.plans.clear()` at `:973` makes every token pay `plan_named`. Fourth — and I am *demoting* it against the brief's ordering, on R12's null result — is graph minimality/write placement; it is still required, but as a correctness-and-KV-residency change (R11-M2′), not as a dispatch-count change. Underneath all four sits an instrumentation defect that makes every attribution above unfalsifiable: `classify_kind` (`omega/src/metal.rs:785-826`) buckets by substring of emitted MSL (R5-M10), and R12 ROW 263 records it mislabelling 9/601 → 225/385 when a body changed — so the census must be fixed **before** any body is swapped or the numbers cannot be attributed. Two verifications against main that the ledger did not carry: `layout_of` at `proxima-tensor/src/bind.rs:1594-1605` *already* folds `axis.offset * stride` into `Layout.base` and `msl.rs:2735` already emits `long out_offset = u.out_base;` — write placement is emitted today; what blocks it is `shape::project_output_shape` (`proxima-tensor/src/shape.rs:469-486`) forcing the output extent to equal the iteration extent, and all three GPU emitters rejecting the scatter form outright (`omega/src/msl.rs:932`, `omega/src/wgsl.rs:363`, `omega/src/cuda.rs:240`, `EmitError::ScatterNotSupported`) while `proxima-tensor/src/cpu.rs:6892-6927` implements it. And `scripts/llama_reference/` plus the streaming-copy `membw_probe` fix are **not on main** (commit `0fecd5a` on `perf/gpu-decode-ladder`, `git merge-base --is-ancestor 0fecd5a HEAD` → not-in-main), so main today has no committed incumbent harness and its `membw_probe` still uses `Op::Reduce{ScalarOp::Add}` (`omega/examples/membw_probe.rs:139-146`) — the reduce-to-scalar that measures dispatch, not bandwidth.

---

## One-RISC binding

### What changes in `proxima-tensor`

**B1. `Reduce.out_map` write placement — offset and destination extent.**
- Today: `bind::bind_reduce` (`proxima-tensor/src/bind.rs:955-996`) calls `layout_of(pattern, shapes.of(node))` (`:1594-1605`), which already computes `base += i64::from(axis.offset) * stride`. The write offset therefore already reaches `BoundOpKind::Reduce.out_layout.base` (`bind.rs:243`, `Layout` at `:95-98`) and already reaches emitted MSL (`omega/src/msl.rs:2735`: `long out_offset = u.out_base;`).
- The single blocker: `shape::project_output_shape` (`proxima-tensor/src/shape.rs:469-486`) returns `iter_extents[term.axis]` for every out_map axis, so the destination tensor is exactly the size of the iteration — an offset would address past its own shape. The change is **one match arm**: an out_map axis may declare a destination extent, using the convention `IndexMap` already owns for the write direction (`map.rs:110-131`, `IndexMap::scatter_extent` at `:209-222` — "that otherwise-always-`0` `offset` carries the destination axis's static extent"). No new `Op`, no new `IndexMap` variant, no new `Reduce` field.
- **Tried first, because §1 says write the expression before minting anything**: the placement is expressible on main *today* as a scatter — `out_map = IndexMap::scatter(...)` with `indices = Elementwise(Add, [Iota(w), Constant(cached_len)])`. Zero IR change; `cpu::run_reduce_scatter` (`cpu.rs:6892-6927`) runs it. If that expression evaluates correctly, the entire remaining work is **GPU coverage of a field that already exists**, not an IR feature. P5.1 writes the expression; P5.2 measures both forms.

**B2. Write placement into a caller-owned persistent buffer.**
- `omega::metal::encode_op` (`omega/src/metal.rs:2179-2252`) calls `allocate_buffer(device, bound_output_len(bound), bound.dtype)` at `:2210` for every op every token, then `device_buffers.insert(bound.node, (output, 0))` at `:2249`. That map is already `BTreeMap<NodeId, DeviceBuffer>` where `DeviceBuffer` is `(MetalBuffer, usize)` — **a buffer plus an offset**. Aliasing a bound op's output onto a persistent, caller-owned buffer at a byte offset is `device_buffers.insert(node, (persistent, offset))`. It is an insert, not a type.
- This is exactly the incumbent's shape: KV allocated once (`llama-kv-cache-unified.cpp:74-118`), written per token by `ggml_cpy` into `ggml_view_1d(k, …, row_size*head_cur)` (`:749-788`) — an in-place write at a byte offset into a persistent buffer (R8).

**B3. `cached_len` as a plan-stable capacity bucket, not a runtime uniform.**
- `resolve_plan` (`generate.rs:958-975`) keys on `(new_count, cached_len)`; `cached_len` is `Extent::Symbolic(1)` on every KV input leaf (`spec.rs:6216-6245`, R11-M6′). Change lands in the **caller**: round `cached_len` up to a capacity multiple and mask the tail with the machinery that already exists (`causal_mask`, `Op::Iota`, `ScalarOp::Select` — `op.rs:207-231` documents exactly this composition). Incumbent precedent: n_kv padded to 256 and masked (R11-M6′).
- **Not** a runtime uniform: `BoundOp.extents` is a baked `Vec<u64>` (`bind.rs:200-215`); making it dynamic touches every backend's uniform packing. Parked with a named un-park condition (see Abandoned designs #6).

### What changes in `omega`

**B4. One first-class route enum, decided before emission, censused `(NodeId, reason)`.**
- Today the route is re-derived independently in three places: `emit` (`msl.rs:673-697`), `kernel_cache_key` (`msl.rs:751-772`, running `tiled_gemm_block(..).is_some()` then `packed_row_block(..).is_some()`), and `push_cooperative_reduce_body` (`msl.rs:3140-3194`, running the same two gates again). Ordering is load-bearing and commented as such (`msl.rs:751-754`).
- Bind: `pub enum KernelRoute { Elementwise, SerialReduce, CooperativeReduce, PackedRowBlock, TiledGemm, Scan, Iota, Constant }` — 8 variants = the 8 Metal body shapes R5 counts. Decided **once** by `fn route_of(&BoundOp, &[Option<PackedCodec>]) -> KernelRoute`; `emit`, `kernel_cache_key`, `push_cooperative_reduce_body` all consume the value.
- Census keyed `(NodeId, KernelRoute)`, mirroring the shipped pattern verbatim: `WidthDeclineReason` (`proxima-tensor/src/instrument.rs:809-828`), `WidthDeclineRow` (`:840`), `record_width_tile_decline(node, reason, …)` (`:848-864`), module gated at `proxima-tensor/src/lib.rs:213-214`.
- `classify_kind` (`omega/src/metal.rs:785-826`) is **deleted**, not extended. Its own doc (`:777-783`) admits the routing decision "is not exposed as its own accessor"; R12 ROW 263 shows the parallel branch's fix added a second marker string instead. Adding a marker string reconstructs information that was never destroyed once `KernelRoute` is a value.

**B5. One emitter core over 4 `BoundOpKind`s; backend text tables only.**
- `emit` already matches exactly 4 kinds + the `Keep` split (`msl.rs:673-697`). The core is that match, parameterized over a backend text table (intrinsic names, signature syntax, type tokens). Backend-specific text only — no backend-specific *structure*.
- **Every backend covers every kind and every field.** Current holes, all verified: `cuda.rs:146-183` rejects `Iota` and `Constant` (`CudaUnsupportedOpKind`); `msl.rs:932`, `wgsl.rs:363`, `cuda.rs:240` all reject `out_scatter` with `EmitError::ScatterNotSupported` while `cpu.rs:6892` implements it. That asymmetry *is* the "one RISC" violation — the IR has a field two of five executors honour.

**B6. One sizing config owns every geometry constant.**
- Move to `omega/omega-runtime.toml` via `omega/build.rs`'s `emit_sizing_consts` (`:105`) / `resolve_int` (`:79`) / `require_*` validators (`:16-66`): `PACKED_ROWS_PER_GROUP` (`msl.rs:1017`), `TILE_DIM` (`msl.rs:1030`), `TILED_GEMM_NSG` (`msl.rs:1046`), and the cooperative-reduce thread width (today the bare `SIMD_WIDTH` literal at `msl.rs:3141`, `:3172`, `:3190`).
- `SIMD_WIDTH` (`omega/src/sized.rs:45`) **stays a hardware fact**, not a knob — its own doc gives the reason and §12's exemption for hardware facts applies.

### What does NOT change

- **No new `Op` variant.** 5 stay (`op.rs:175-266`). No `Concat`, no `Pad`, no `Tile` — zero hits on main and none added.
- **No new `BoundOpKind`.** 4 stay (`bind.rs:221-264`). This is the explicit adjudication against `BoundOpKind::CachedAttention` (R12) — see P0.6.
- **No new trait, no trait object, no `Box<dyn>`.** Enum + match (§20, §11).
- **No new driver type.** No `PlacedBuffer`, no `write_placement` — `(MetalBuffer, usize)` already exists at `metal.rs:2249`.
- `Op::Reduce`'s `out_map` stays the *only* write-placement mechanism; `Elementwise` gains no `out_map`.

---

## Bench ladder (defined once; every step's prediction is exactly one rung ahead)

| rung | instrument | unit |
|---|---|---|
| **nano** | `omega::metal::execute_plan_op_timed` (`metal.rs:654-761`), one command buffer per op | µs/dispatch, ALU ops/8 weights |
| **micro** | `omega/examples/q4k_matvec_probe.rs`, `omega/examples/membw_probe.rs`, `cargo bench -p omega --bench metal_vs_cpu --features metal` (`omega/Cargo.toml:207-210`) | GB/s, ms/family |
| **milli** | `token_breakdown_metal` line (`generate.rs:1721-1753`): `prepare_ms`, `block_upload_ms`, `op_setup_ms`, `pipeline_lookup_ms`, `encode_dispatch_ms`, `gpu_exec_ms`, `readback_ms` | ms/token per stage |
| **bench** | board cell: `step_wall_ms` + `gpu_exec_ms` + ratio vs llama.cpp-Metal + CoV over ≥5 runs | ms/token, ratio, CoV% |

A miss at rung k+1 **kills the climb** and is decomposed in the log row into *inconsistency* (the two rungs measure different things) vs *understanding-gap* (the mechanism is wrong), with a named work item. No step predicts two rungs ahead.

**Standing gate commands** (used by every step; `N==0 is RED` everywhere):

```
cd $WT && CARGO_TERM_COLOR=never cargo nextest run -p omega --features metal,cpu,instrument --no-fail-fast     # N=97 on main (discipline.md:18486)
cd $WT && CARGO_TERM_COLOR=never cargo nextest run -p proxima-tensor --features std,instrument --no-fail-fast   # N=478 on main (discipline.md:18486)
cd $WT && bash scripts/omega-gate.sh
cd $WT && bash scripts/proxima-tensor-gate.sh
cd $WT && cargo clippy -p omega -p proxima-tensor -p proxima-model-interop --all-targets --all-features -- -D warnings
```

**Standing decode cell** (the milli/bench measurement, run min-of-N interleaved):

```
cd $WT && PROXIMA_MAX_TOKENS=24 CARGO_TARGET_DIR=$TD \
  cargo nextest run -p proxima-model-interop --features std,metal,instrument --release \
  --lib --run-ignored all -E 'test(runs_a_cached_greedy_decode_loop)' --no-capture
```
(`ServingConfig::default().gpu_layers == GPU_LAYERS_ALL` at `proxima-model-interop/src/serving.rs:168`, and `select_backend` picks Metal only at that value — `generate.rs:856`. Test at `proxima-model-interop/src/bind.rs:2818`, `#[ignore]`, so `--run-ignored all` is mandatory or N==0.)

**Standing incumbent arms:**
```
/Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench \
  -m /Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf \
  -n 32 -r 5 -t 8 -ngl 99            # arm A: -fa 0 (checkout default, R8 common/common.h:328)
  ... -fa 1                          # arm B: flash attention, second incumbent arm
```

---

# Phase 0 — seal, land, reconcile what already exists

**Why first:** every number in R1/R2/R3 is tagged MEMORY, main's discipline log does not know the 2026-09-02 GPU session happened (R10: last row is ROW 233 at `discipline.md:18736`; `3.54x`, `17.470`, `228.9`, `metal-q4k-*` have zero hits), and the measured wins live as unrebased diffs against a commit main has moved 9 commits past (R7). Nothing downstream is attributable until this closes.

### P0.1 — quiet-box re-seal of the board on `4be2f3a` (the baseline every later Δ is measured against)

- **WT** `proxima-wt-riscseal` · **BR** `bench/gpu-reseal-4be2f3a` · **TD** `/Users/brianbruggeman/repos/slot-0/proxima-wt-riscseal/target`
- **Commands**
  ```
  git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add \
    /Users/brianbruggeman/repos/slot-0/proxima-wt-riscseal -b bench/gpu-reseal-4be2f3a 4be2f3a
  git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-riscseal cherry-pick b437f49 fb61d04   # scripts/sealed-pass.sh from bench/sealed-pass
  ```
  Then fix the two defects the cherry-pick carries in, in the same commit that lands it: `scripts/sealed-pass.sh:4` hardcodes `REPO_ROOT="/Users/brianbruggeman/repos/slot-0/proxima-wt-seal"` and `:25-28` hardcodes four sibling worktrees. Replace with `REPO_ROOT="$(git rev-parse --show-toplevel)"` and drop the sibling-worktree arms (they are re-added as real cells in Phase 8).
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-riscseal && SEALED_PASS_LOAD_THRESHOLD=2.0 bash scripts/sealed-pass.sh
  ```
- **OPEN** `scripts/sealed-pass.sh:1-40` (from `git show bench/sealed-pass:scripts/sealed-pass.sh`), `proxima-model-interop/src/bind.rs:2818`
- **N** decode cell must emit ≥5 `token_breakdown_metal` lines and ≥5 `decode_summary` lines; incumbent arm must emit 5 `llama-bench` rows. **N==0 is RED** — the decode test is `#[ignore]` and silently `return`s if the gguf is absent (`bind.rs:2823-2830`), which exits 0.
- **PRED (bench rung, the only rung this step has)** `step_wall_ms` on `4be2f3a` lands in **62–70 ms/token** (R1 board 67.917 at `2b95210`; main has since added `ff749a0` plan-cache bound and `7d09145` checkpoint-mapping-once, which R11-M6′/M7′ predict move `prepare` and `block_upload` down but not `gpu_exec`). `gpu_exec_ms` lands in **54–58** (R1: 56.5–57.0). Incumbent lands at **17.3–17.6** (R1: 17.470; R12 ROW 250: 17.467, CoV 0.363%).
- **KILL** CoV > 5% on `step_wall_ms` over 5 runs after two quiet-box attempts → the box is not sealable; the entire plan's Δ claims are unbacked and every later step reverts to min-of-9 interleaved pairs instead of medians (the ROW 143 precedent, `discipline.md:13353`).
- **ROLLBACK** none (measurement only); `git worktree remove`.
- **BLAST** zero source. One new script at a project-level path.
- **COUNTER** `plan_hits` / `plan_misses` / `plan_cache_len` (`generate.rs:1732`, `:1764`) — records whether `ff749a0` changed R11-M6′; `gpu_exec_calls` (dispatches/token) — records whether main is still at 1196.
- **REPROVE** `bash scripts/sealed-pass.sh` (self-contained; pins llama-bench binary, model path, MAC constant `7110402048`, weight-byte constant `3.9996` — R1 rate constants, which R1 notes are *not computed in code*).
- **LOG** ROW 234.

### P0.2 — land the committed incumbent harness and the streaming-copy `membw_probe`

- **WT** `proxima-wt-riscbw` · **BR** `perf/gpu-incumbent-harness` · **TD** `.../proxima-wt-riscbw/target`
- **Commands** `git cherry-pick 0fecd5a` onto `4be2f3a`; resolve against main's `membw_probe`.
- **OPEN** `omega/examples/membw_probe.rs:123-200` (main's `Op::Reduce{ScalarOp::Add}` at `:139-146` is the defect), `git show 0fecd5a`
- **N** `scripts/llama_reference/run.sh` + `README.md` present after land (`git ls-tree` count == 2, **0 is RED**); `cargo run -p omega --example membw_probe --features metal --release` emits ≥4 size rows.
- **PRED (micro)** with `ScalarOp::Negate` + identity map, marginal GB/s at 64→256 MiB lands at **9.0 ± 2** (ROW 221, `discipline.md:18402`).
- **KILL** the cherry-pick's `membw_probe` reads < 5 GB/s marginal → the Negate reshape is not the fix either; escalate to P1.1 immediately rather than landing a second wrong probe.
- **ROLLBACK** `git reset --hard 4be2f3a`.
- **BLAST** one example + two new script files. No library code.
- **COUNTER** none (probe is the counter).
- **REPROVE** `bash scripts/llama_reference/run.sh`; `cargo run -p omega --example membw_probe --features metal --release`.
- **LOG** ROW 235.

### P0.3 — recover, rebase, and land the Q4_K body candidate A (`metal-q4k-mask-fma`)

- **WT** `proxima-wt-riscq4ka` · **BR** `perf/metal-q4k-mask-fma` · **TD** `.../proxima-wt-riscq4ka/target`
- **Commands**
  ```
  git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-all diff -- omega/ > /tmp/q4k-mask-fma.patch   # R7: 13 files +3859/-197, uncommitted
  git -C .../proxima-wt-riscq4ka apply --3way /tmp/q4k-mask-fma.patch
  ```
  Then strip everything that is not the Q4_K body: the worktree carries seven features (R7). Keep **only** `metal-q4k-mask-fma`. One green commit, one feature, default-off (`omega/Cargo.toml [features]`, gate 1).
- **OPEN** `omega/src/msl.rs:190-311` (`Q4K_UNPACK_MSL`), `omega/src/msl.rs:2425-2824` (`push_packed_row_blocked_body`); incumbent reference `ggml/src/ggml-metal/ggml-metal.metal:5086-5193`, mask-without-shift at `:5147-5150`, scale fold at `:5171-5175` (R8)
- **N** `cargo nextest run -p omega --features metal,cpu,instrument,metal-q4k-mask-fma` ≥ 97 (main's asserted count, `discipline.md:18486`). Q4_K real-checkpoint parity: `omega/tests/q4k_real_checkpoint_parity.rs` must run ≥1 case. **N==0 is RED** — a default-off feature is exactly the condition gate 3 warns hides tests.
- **PRED (micro)** `q4k_matvec_probe` Q4_K family time drops ≥ 15% vs the P0.1 baseline (R3-M3: −36% on ffn_gate/up, −17.2% `gpu_exec`; the conservative floor is the aggregate figure).
- **KILL** parity vs `cpu::evaluate` exceeds the tolerance `q4k_real_checkpoint_parity.rs` already pins → §14 (the incumbent wins on correctness); the body does not land at any speed.
- **ROLLBACK** `git reset --hard`; the feature is default-off so main's default build is untouched regardless.
- **BLAST** `omega/src/msl.rs` MSL text under one cfg. Zero callers outside omega.
- **COUNTER** `(NodeId, KernelRoute::PackedRowBlock)` census count (from P1.2) — proves the swapped body is the one that ran. Until P1.2 lands, `diagnose_kind` (`metal.rs:835-854`, structural) is the only trustworthy witness; `classify_kind` is NOT (R5-M10, R12 ROW 263).
- **REPROVE** `cargo nextest run -p omega --features metal,cpu,instrument,metal-q4k-mask-fma` + `cargo run -p omega --example q4k_matvec_probe --features metal --release`.
- **LOG** ROW 239 (shared with P0.4/P2.1 — one row, two candidates, one chosen).

### P0.4 — recover and land Q4_K body candidate B (`q4k_pair_dot`, R12 ROW 257)

- **WT** `proxima-wt-riscq4kb` · **BR** `perf/metal-q4k-pair-dot` · **TD** `.../proxima-wt-riscq4kb/target`
- **Commands** extract only the paired-nibble commit from `perf/cached-attention-streaming` (R12 names it in the first 12 commits, also on `perf/q4k-independent-accumulators`):
  ```
  git log --oneline main..perf/q4k-independent-accumulators
  git cherry-pick <the q4k pair-nibble commit>            # one commit, not the 42
  ```
  Do **not** bring `physical.rs`, `BoundOpKind::CachedAttention`, the bind.rs matcher, or the `libm` dep (R12) — those are adjudicated separately in P0.6.
- **OPEN** same as P0.3, plus the cherry-picked diff.
- **N** same gate as P0.3, feature `metal-q4k-pair-dot`.
- **PRED (micro)** Q4_K family time drops ≥ 20% (R12 ROW 257: 47.8 → 33.9 ms, −29%; parity 3.1e-6 vs f32 on real `blk.0.attn_q.weight`).
- **KILL** same parity kill as P0.3.
- **ROLLBACK / BLAST / COUNTER / REPROVE** as P0.3.
- **Note** P0.3 and P0.4 both land as **default-off features on the same branch point**, so P2.1 can run them interleaved at one commit. Only one survives into `default`.

### P0.5 — rebase and land the wide cooperative reduce (`metal-wide-cooperative-reduce`)

- **WT** `proxima-wt-riscwide` · **BR** `perf/metal-wide-cooperative-reduce` · **TD** `.../proxima-wt-riscwide/target`
- **Commands** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-gpudisp diff > /tmp/wide-reduce.patch` (R7: 4 files +244/−28, feature already present), `git apply --3way`, strip to the one feature.
- **OPEN** `omega/src/msl.rs:3140-3194` (`push_cooperative_reduce_body`; the three `SIMD_WIDTH` literals at `:3141`, `:3172`, `:3190` are the pin). Incumbent reference: `ggml-metal.m:3797-3804` (nth doubles 32 → `min(ne00/4, maxTotalThreadsPerThreadgroup)`), `ggml-metal.metal:1679-1721` (float4 loads, `simd_sum` → threadgroup shmem → `simd_sum`) — R8.
- **N** omega gate ≥ 97 with the feature on; `omega/tests/metal_parity.rs` must run ≥ 13 cases (the count `omega/Cargo.toml`'s own feature comment names).
- **PRED (milli)** the `reduce-cooperative` family's share of `gpu_exec_ms` falls ≥ 15% (R3-M4: −20% on that bucket, −2.8 ms; R2 sizes the bucket at 8.6 ms).
- **KILL** `metal_parity` or `backend_parity` regresses → roll back; a wider tree changes float summation order and §14 binds on the oracle, not on speed.
- **ROLLBACK** feature default-off; `git revert`.
- **BLAST** `msl.rs` cooperative-reduce body only; the tiled-GEMM and packed-row-block paths return before it (`msl.rs:3163-3187`) and are untouched.
- **COUNTER** `(NodeId, KernelRoute::CooperativeReduce)` count and per-route `gpu_exec` ticks (P1.2).
- **REPROVE** `cargo nextest run -p omega --features metal,cpu,instrument,metal-wide-cooperative-reduce` + the decode cell.
- **LOG** ROW 240.

### P0.6 — adjudicate `BoundOpKind::CachedAttention` against the one-RISC rule (a NEGATIVE row, landed)

- **WT** `proxima-wt-riscadj` · **BR** `docs/adjudicate-cached-attention` · **TD** `.../proxima-wt-riscadj/target`
- **Commands** (read-only adjudication + one docs commit)
  ```
  git show perf/cached-attention-streaming --stat
  git diff main..perf/cached-attention-streaming -- proxima-tensor/src/bind.rs proxima-tensor/src/physical.rs
  git show perf/cached-attention-streaming:failure-cached-attention-matcher.md
  ```
- **OPEN** `proxima-tensor/src/bind.rs:221-264` (the 4 kinds), `proxima-tensor/src/op.rs:175-266` (the 5 ops), workspace `AGENTS.md` "problem solving" ("we should not be adding arbitrary rules/code for specific instances"), the branch's own `failure-cached-attention-matcher.md`.
- **The adjudication, stated as the row's content, not as a verdict:** a fifth `BoundOpKind` matched by a post-bind structural matcher over one model's attention shape is an instance rule, not an instruction. The branch's own failure record already abandoned the BoundOp-only matcher as "a heuristic" that "cannot prove the semantic roles" (R12). R12's own numbers show the macro-op did not move wall (51.535 vs 51.571 control) and made `gpu_exec` worse (39.841 vs 35.117), while `prepare` was 150.7 ms/token with the matcher before indexing (ROW 247) — the matcher is a per-token CPU cost *because* `plan_hits=0` (R11-M6′), which P4.1 removes at the root.
- **What is adopted from that branch instead:** `prune_dead` / `dead_resolved_nodes` (commit `216d925`, R12) — generic, RISC-conformant, no new kind. And the paired Q4_K body (P0.4).
- **N** the docs commit adds exactly 1 discipline row + 1 `ai_docs/invariants.jsonl` record (**0 is RED**).
- **PRED** none — this step produces no measurement; it produces a recorded decision boundary and a re-usable invariant.
- **KILL** if P6.2's minimal-graph measurement later shows the online-softmax cluster cannot be expressed at ≤ 23 ops/layer through `out_map` placement alone, this adjudication re-opens with that number attached.
- **ROLLBACK** n/a (docs only).
- **BLAST** `proxima-tensor/docs/discipline.md`, `ai_docs/invariants.jsonl`.
- **COUNTER** ops/layer census from P1.2 is what would overturn it.
- **REPROVE** `jq -c 'select(.id=="proxima.omega.one_risc_bound_kinds")' ai_docs/invariants.jsonl`
- **LOG** ROW 242.

### P0.7 — adopt `prune_dead` from the parallel branch

- **WT** `proxima-wt-riscprune` · **BR** `perf/bind-prune-dead` · **TD** `.../proxima-wt-riscprune/target`
- **Commands** `git cherry-pick 216d925` (R12: "drop dead resolved nodes before GPU dispatch").
- **OPEN** `proxima-tensor/src/bind.rs` (the `prune_dead` / `dead_resolved_nodes` addition in that diff)
- **N** proxima-tensor gate ≥ 478; a new test asserting the pruned count on the real Mistral program is ≥ 1.
- **PRED (milli)** dispatches/token falls below 1196 by the pruned-node count; `gpu_exec_ms` falls by **≤ 1 ms** — deliberately a *small* prediction, because R12 ROW 262/267 shows halving dispatches moved wall by 0.036 ms.
- **KILL** any parity test fails → revert (a dead-node pruner that changes output is not a pruner).
- **ROLLBACK** `git revert`.
- **BLAST** `bind.rs` only; both backends consume the pruned plan.
- **COUNTER** `encode_dispatch_calls` (`metal.rs` `ENCODE_DISPATCH_CALLS`).
- **REPROVE** decode cell; read `encode_dispatch_calls` from `token_breakdown_metal`.
- **LOG** ROW 241.

### P0.8 — land the log rows and renumber the parallel branch

- **WT** `proxima-wt-riscrows` · **BR** `docs/gpu-lane-rows-234-plus` · **TD** n/a (docs)
- **Commands**
  ```
  grep -n "^## ROW" proxima-tensor/docs/discipline.md | tail -1     # must print ROW 233 at line 18736
  ```
  Assign row numbers **at land time, from main**: ROW 234 (P0.1), 235 (P0.2), 239 (P0.3/P0.4/P2.1), 240 (P0.5), 241 (P0.7), 242 (P0.6). The parallel branch's ROW 234–267 are renumbered into the tail of the reservation as they land; their physical-order defect (263/264 at line 5870, R12) is corrected on the way in.
- **N** every landed row has **zero blank cells** across the 16-gate table; `grep -c "^## ROW 2[3-5][0-9]"` must equal the number of steps landed. **0 is RED.**
- **KILL** a row whose re-prove command does not run today is unsealed and does not land (§16).
- **BLAST** `proxima-tensor/docs/discipline.md`, `proxima-tensor/docs/rooflines.md`.
- **REPROVE** run each row's own re-prove command in sequence.

### P0.9 — clear the dirty tree

- `proxima-onnx/scripts/torch_reference/venv/` is untracked in `git status` on main. It is a venv, not source. Add it to the ignore file the way `scripts/onnx_reference/.gitignore` already handles its own `.venv`. One `chore:` commit. **N** = `git status --porcelain` returns 0 lines (non-zero is RED for every later interleaved measurement, because a dirty tree makes `git stash`-based A/B unsafe).

---

# Phase 1 — measurement substrate and observability (zero mass removed, near-zero risk, unblocks all attribution)

**Ordering rationale:** mass-per-risk is undefined while the denominator is unmeasured and the classifier lies. R5-M10 + R12 ROW 263 are the proof that route attribution is currently reconstructed from emitted source text; ROW 221 (`discipline.md:18402`) is the proof the bandwidth ceiling is DEBT.

### P1.1 — GPU streaming-copy bandwidth probe with readback outside the timed window (the roofline debt)

- **WT** `proxima-wt-riscroof` · **BR** `perf/gpu-roofline-streaming` · **TD** `.../proxima-wt-riscroof/target`
- **Commands**
  ```
  cd $WT && cargo run -p omega --example membw_probe --features metal --release
  ```
- **OPEN** `omega/examples/membw_probe.rs:139-200` (after P0.2's Negate reshape lands). The named residual in ROW 221 is explicit: *"readback of the output buffer (same size as the input) happens INSIDE the timed window and is not in the GB/s numerator, which deflates every row."* The fix: time only the `commit` → `waitUntilCompleted` window; perform the host readback after `read_ticks()` stops. `execute_plan` (`omega/src/metal.rs:449-568`) does one commit / one wait / then readback at `:2358-2378` — split the timer boundary there, or add a probe-only path that skips readback entirely and validates on a separate untimed run.
- **N** ≥ 4 size rows (256 KiB / 8 MiB / 64 MiB / 256 MiB) plus a two-size marginal row. **N==0 is RED.**
- **PRED (micro)** removing readback from the timed window moves the 64→256 MiB marginal figure from 9.0 GB/s (ROW 221) to **≥ 18 GB/s** — because readback is byte-for-byte the same size as the copy and is currently timed but not counted, so the correction is bounded below by 2×. Stated as a floor, not a point.
- **KILL** the corrected figure still lands under the incumbent's *achieved* 228.9 GB/s (R1) — a "ceiling" below a measured achievement is not a ceiling. Then the probe shape is still wrong (candidate: the one-thread-per-element grid, `membw_probe.rs`'s identity map, is dispatch-bound not bandwidth-bound) and the debt row stays open with that named next experiment. **No spec-sheet figure is ever substituted** (ROW 221's own standing rule).
- **ROLLBACK** example-only; `git revert`.
- **BLAST** `omega/examples/membw_probe.rs`, `proxima-tensor/docs/rooflines.md:396-479` + summary row at `:751`.
- **COUNTER** `GPU_EXEC_TICKS` vs `READBACK_TICKS` (`metal.rs`) — the split that proves readback left the window.
- **REPROVE** `cargo run -p omega --example membw_probe --features metal --release`
- **LOG** ROW 236; rewrite `rooflines.md:411` from DEBT to a measured cell, and rewrite the closing note at `:766-773` ("is not a gap-to-machine at all").

### P1.2 — `KernelRoute` as a first-class value + `(NodeId, KernelRoute)` census; delete `classify_kind`

- **WT** `proxima-wt-riscroute` · **BR** `refactor/omega-kernel-route-enum` · **TD** `.../proxima-wt-riscroute/target`
- **Commands** standard omega gate + decode cell.
- **OPEN, in this order**
  1. `omega/src/msl.rs:673-697` — `emit`'s 4-kind match (the core that survives).
  2. `omega/src/msl.rs:751-772` — `kernel_cache_key`'s `tiled_gemm_block(..).is_some()` / `packed_row_block(..).is_some()` re-derivation.
  3. `omega/src/msl.rs:3140-3194` — `push_cooperative_reduce_body`'s *third* re-derivation of the same two gates.
  4. `omega/src/msl.rs:1145-1240` — `classify_packed_row_block` / `PackedRowBlockRejection` (already a first-class reason enum; `KernelRoute` is its sibling).
  5. `omega/src/metal.rs:785-826` — `classify_kind` (delete) and `:835-854` — `diagnose_kind` (fold into the census).
  6. `proxima-tensor/src/instrument.rs:809-828`, `:840`, `:848-864` — the shipped census pattern to mirror exactly.
- **N** omega gate ≥ 97 **plus** ≥ 3 new tests: (a) `route_of` is total over all 4 `BoundOpKind`s; (b) `route_of`'s answer equals the branch `emit` actually takes, asserted per route on a real bound program; (c) the census sums to the dispatch count. **N==0 is RED.**
- **PRED (milli)** behaviour-neutral: `gpu_exec_ms` and `step_wall_ms` change by **< 1%** (inside the P0.1 CoV band). The census emits **1196 ± the P0.7 pruned count** `(NodeId, KernelRoute)` rows per token, summing exactly to `encode_dispatch_calls`.
- **KILL** the census sum ≠ `encode_dispatch_calls` → the route is still being decided in more than one place; do not proceed to Phase 2 (every Phase-2 attribution would be unfalsifiable).
- **ROLLBACK** `git revert`; instrument-gated so the default build carries none of it.
- **BLAST** `omega/src/msl.rs` (three call sites collapse to one), `omega/src/metal.rs` (`classify_kind` deleted — check every caller; it is instrument-only per its own doc), `proxima-tensor/src/instrument.rs` (+1 enum, +1 recorder, mirroring `WidthDeclineReason`). `omega/src/wgsl.rs` and `omega/src/cuda.rs` untouched in this step.
- **COUNTER** the census itself: `record_kernel_route(node, route)` keyed `(node.0, route)`, read out per token next to `token_breakdown_metal` (`generate.rs:1721-1753`).
- **REPROVE** decode cell with `--features instrument`; assert the per-route table sums to `encode_dispatch_calls`.
- **LOG** ROW 237.

### P1.3 — geometry constants into `omega-runtime.toml` (§12 debt)

- **WT** `proxima-wt-riscgeom` · **BR** `refactor/omega-geometry-sizing-config` · **TD** `.../proxima-wt-riscgeom/target`
- **OPEN** `omega/src/msl.rs:1017` (`PACKED_ROWS_PER_GROUP=4`), `:1030` (`TILE_DIM=8`), `:1046` (`TILED_GEMM_NSG=4`), `:3141`/`:3172`/`:3190` (cooperative-reduce width, bare `SIMD_WIDTH`); pattern at `omega/build.rs:16-66` (validators), `:79` (`resolve_int`), `:105` (`emit_sizing_consts`); config at `omega/omega-runtime.toml` (`[tiled_gemm]` section is the existing instance); doc contract at `omega/src/sized.rs:1-45`.
- **New sections:** `[packed_row_block] rows_per_group`, `lanes_per_block`; `[tile] dim`; `[tiled_gemm] nsg`; `[cooperative_reduce] max_threads`, `vector_width`.
- **N** ≥ 4 new build-time validator tests + 1 env-override test per key (`OMEGA_<SECTION>_<KEY>`), each with its own `cargo:rerun-if-env-changed` (gate 15 point 5). Assert `grep -cE "^const (PACKED_ROWS_PER_GROUP|TILE_DIM|TILED_GEMM_NSG)" omega/src/msl.rs` **== 0** after the move.
- **PRED (milli)** behaviour-neutral at the default values: `gpu_exec_ms` unchanged within the P0.1 CoV band. This step's product is a *knob*, not a delta.
- **KILL** a value cannot be made a build-time const without becoming a runtime read → §12's interaction note binds (a value the optimiser must see as constant can only come from this mechanism); that key stays a source const with a one-line why at the site, recorded as a named exception.
- **ROLLBACK** `git revert`.
- **BLAST** `omega/build.rs`, `omega/omega-runtime.toml`, `omega/src/sized.rs`, `omega/src/msl.rs` const sites.
- **COUNTER** none; the gate is `grep -c` == 0 and the env-override tests.
- **REPROVE** `OMEGA_COOPERATIVE_REDUCE_MAX_THREADS=256 cargo build -p omega --features metal` then assert the generated `omega_sized.rs` carries 256.
- **LOG** ROW 238.

### P1.4 — the missing incumbent arm: llama.cpp with `-fa 1`

- **WT** `proxima-wt-riscfa` · **BR** `bench/llama-flash-attention-arm` · **TD** n/a
- **Commands** extend `scripts/llama_reference/run.sh` (landed in P0.2) with a second arm.
- **OPEN** `common/common.h:328` at checkout `b25346221` (flash attention OFF by default, R8) — the reason arm A is the checkout default and arm B is a *second* incumbent, not a replacement.
- **N** run.sh emits exactly 2 arm tables × 5 runs = 10 rows. **N==0 is RED.**
- **PRED (bench)** arm B (`-fa 1`) lands within **±10%** of arm A's 17.470 ms/token (R1) at a 24-token decode — decode attention at n_kv small is not where flash attention pays, and R8 records that the decode path at this checkout is `mul_mat(K,Q)` → `soft_max_ext` (already one fused kernel, `ggml-metal.metal:1051-1145`) → `mul_mat(V)`.
- **KILL** arm B is > 20% faster than arm A → the home-turf incumbent for every later row is arm B, and every ratio in R1/R2/R10 is re-based against it. Say so loudly rather than keeping the flattering arm.
- **BLAST** one script.
- **REPROVE** `bash scripts/llama_reference/run.sh`
- **LOG** ROW 235 (with P0.2).

---

# Phase 2 — the Q4_K body (largest measured mass, lowest risk: one kernel body, parity-gated, feature-flagged)

### P2.1 — head-to-head at one commit; choose ONE body

- **WT** `proxima-wt-riscq4k2` · **BR** `perf/metal-q4k-body-selection` · **TD** `.../proxima-wt-riscq4k2/target`
- **Commands** branch from the merge of P0.3 + P0.4 + P1.2, so both features and the route census are compiled in at one commit. Run **interleaved** (A, B, A, B × 5) — never a sequential before-block/after-block (`discipline.md:13385` records exactly that contamination).
  ```
  for i in 1 2 3 4 5; do
    <decode cell> --features std,metal,instrument,metal-q4k-mask-fma
    <decode cell> --features std,metal,instrument,metal-q4k-pair-dot
  done
  ```
- **OPEN** `omega/src/msl.rs:190-311`, `:2425-2824`; incumbent `ggml-metal.metal:5086-5193` (R8: mask without shift `& 0x000F/0x0F00/0x00F0/0xF000`, fold 1/256 and 1/16 into the scale at combine `:5171-5175`, branch-free `kmask1/2/3` scale/min at `:5147-5150`, `<4,2,32>` = 4 rows/simdgroup, 2 simdgroups/threadgroup, `dispatchThreadgroups((ne01+7)/8,1,ne12*ne13)` × `(32,2,1)` at `ggml-metal.m:3330`, `:3215-3220`).
- **N** both features' full parity suites green: `q4k_real_checkpoint_parity`, `q4k_matmul_layout`, `q4k_unpack`, `metal_parity`, `backend_parity`, `metal_real_forward` — omega gate ≥ 97 under **each** feature separately. **N==0 is RED under either.**
- **PRED (milli, one rung past P0.3/P0.4's micro evidence)** the surviving body takes `gpu_exec_ms` from the P0.1 baseline (predicted 54–58) to **34–42 ms/token**. Anchor: R12's feature-off control already carries the paired body and reads `gpu_exec` 35.117 (CoV 2.00%) where main reads 56.5–57.0 (R1).
- **KILL** neither body clears −10% `gpu_exec_ms` beyond both arms' CoV bands → the −17.2%/−29% micro figures (R3-M3, R12 ROW 257) do not transfer to the real graph; decompose into inconsistency vs understanding-gap and stop the climb here. Do **not** proceed to P2.2.
- **Selection rule, pre-registered:** the body with the lower median `gpu_exec_ms` wins **only if** its Q4_K-route census count equals the other's (same route, same op set — R5-M10's trap). If counts differ, the comparison is between different op sets and is void; re-run with the route census pinned.
- **ROLLBACK** the loser's feature is deleted, not left dark — two bodies for one mechanism is the §1 violation this step exists to close (R12 names it: "Two bodies, one mechanism, two worktrees. ONE must be chosen").
- **BLAST** `omega/src/msl.rs`. After selection, the winner's feature is **promoted to default** in `omega/Cargo.toml` in a separate commit, gated on the full parity suite.
- **COUNTER** per-route `gpu_exec` ticks for `KernelRoute::PackedRowBlock` (P1.2); `q4k_macs` execution witness (`bind.rs:2856-2860`, the counter that proves the packed-int8 Q4_K branch ran at all).
- **REPROVE** the interleaved loop above + `cargo run -p omega --example q4k_matvec_probe --features metal --release`.
- **LOG** ROW 239.

### P2.2 — split-K for the starved low-row shapes (conditional on P2.1 clearing)

- **WT** `proxima-wt-riscsplitk2` · **BR** `perf/metal-q4k-split-k` · **TD** `.../proxima-wt-riscsplitk2/target`
- **Commands** recover `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-splitk diff` (R7: 5 files +352/−48, feature `metal-q4k-split-k` present), rebase onto P2.1's tip.
- **OPEN** `omega/src/msl.rs:2425-2824` (`push_packed_row_blocked_body`); the shapes are `attn_q/k/v/o` at 1024–4096 rows (R3-M5: rising marginal GB/s 52 → 147 as simdgroups go 256 → 8001).
- **N** omega gate ≥ 97 with the feature; ≥ 1 test asserting split-K's partial-sum combine is bit-reproducible run to run (a non-deterministic reduction order is a §14 hazard).
- **PRED (milli)** the `attn_q/k/v/o` share of `gpu_exec_ms` falls ≥ 10%; total `gpu_exec_ms` falls ≥ 2 ms. Anchor: R3-M5's 52 → 147 GB/s marginal curve; only the *low-row* families are addressed.
- **KILL** total `gpu_exec_ms` does not fall beyond CoV, or the combine is non-deterministic → roll back. R4 already records that adding simdgroups by regrouping (nsg=2) is a four-time negative; split-K is a *different* mechanism (more simdgroups over K, not regrouped threads), and this step is the one that tests whether the distinction is real.
- **ROLLBACK** feature default-off; delete.
- **BLAST** one MSL body under one cfg.
- **COUNTER** per-route census counts split by output-row extent bucket.
- **REPROVE** decode cell + per-route table.
- **LOG** ROW 244 (reserve).

---

# Phase 3 — the non-matmul GPU bucket

### P3.1 — promote the wide cooperative reduce to default (conditional on P0.5 clearing)

- **WT** `proxima-wt-riscwide2` · **BR** `perf/metal-wide-reduce-default` · **TD** `.../proxima-wt-riscwide2/target`
- **OPEN** `omega/src/msl.rs:3140-3194`; the width now reads from `[cooperative_reduce] max_threads` (P1.3), not `SIMD_WIDTH`.
- **N** omega gate ≥ 97 with the feature **in `default`**; `cargo build -p omega --no-default-features --features alloc` exit 0 and reports which modules it built (gate 2's N==0 form).
- **PRED (bench, one rung past P0.5's milli evidence)** with P2.1 + P3.1 both in `default`, `step_wall_ms` lands in **44–52** and the ratio vs arm A lands in **2.5×–3.0×**. Anchor: R12's paired-body control at 51.571 wall (CoV 1.75%) / 2.95× (ROW 250), before any reduce work.
- **KILL** `step_wall_ms` does not fall below the P0.1 baseline by more than both CoV bands → the two GPU-side wins are being eaten by CPU orchestration, which is Phase 4's premise; record that and go to Phase 4 without promoting to default.
- **ROLLBACK** demote from `default`; the feature stays.
- **BLAST** `omega/Cargo.toml [features] default`. Every downstream crate that turns on `omega/metal` inherits it — check `proxima-model-interop/Cargo.toml`'s `metal` feature passthrough.
- **COUNTER** per-route `KernelRoute::CooperativeReduce` ticks.
- **REPROVE** `bash scripts/sealed-pass.sh`
- **LOG** ROW 240 (extended).

### P3.2 — the elementwise bucket census before any elementwise work

- **WT** `proxima-wt-riscelem` · **BR** `perf/gpu-elementwise-census` · **TD** `.../proxima-wt-riscelem/target`
- **OPEN** `omega/src/msl.rs:2104-2177` (`render_elementwise`); R2 sizes the elementwise bucket at 6.65 ms.
- **N** the census emits ≥ 1 row per elementwise node with `(NodeId, KernelRoute::Elementwise, extents, operand_count)`; the count equals the `Elementwise` share of `encode_dispatch_calls`. **N==0 is RED.**
- **PRED (nano)** the top-5 elementwise nodes by tick share account for ≥ 50% of the 6.65 ms bucket — i.e. the bucket is concentrated, not uniform. If it is uniform, there is no lever here and the phase ends with that recorded.
- **KILL** the bucket is uniform across > 200 nodes → no single-node lever exists; the only remaining lever is fewer nodes, which is Phase 6, and this step closes with that pointer. (R8 is the standing reason not to reach for fusion: `grep -rln fuse ggml/src` is **empty** at `b25346221` — parity is reachable without a fusion engine.)
- **BLAST** instrument-gated only.
- **COUNTER** itself.
- **REPROVE** decode cell + per-route table.

---

# Phase 4 — CPU orchestration (11.4 ms, R1; root cause proven at `generate.rs:966`)

### P4.1 — capacity-bucketed `cached_len` so the plan cache can hit

- **WT** `proxima-wt-riscplan` · **BR** `perf/plan-cache-capacity-bucket` · **TD** `.../proxima-wt-riscplan/target`
- **OPEN, in this order**
  1. `proxima-model-interop/src/generate.rs:958-975` — `resolve_plan`; key `(symbols[0], symbols[1])` at `:966`; `self.plans.clear()` at `:973`.
  2. `proxima-tensor/src/spec.rs:6216-6245` — `cached_len` as `Extent::Symbolic(1)` on every KV input leaf.
  3. `proxima-tensor/src/spec.rs:2336-2865` — `append_mistral_cached_layer`; masking is already `[s,w]`-shaped via `causal_mask` (doc at `:2303-2319`), which is the machinery the tail mask reuses.
  4. `proxima-model-interop/src/generate.rs:1443-1449`, `:1477-1559` — the per-token sequence and `cached_len += new_count`.
  5. Incumbent precedent: n_kv padded to a 256 multiple and masked (R11-M6′); `llama-kv-cache-unified.cpp:74-118`.
- **The change:** `cached_len` presented to the program is `round_up(actual, CAPACITY_BUCKET)`; the `[bucket - actual]` tail is masked to `ReduceInit::NegativeInfinity` through the existing `Iota`/`Constant`/`Select` composition (`op.rs:207-231`). `CAPACITY_BUCKET` traces to `proxima-tensor-runtime.toml` (the CPU sizing config that already exists — R5), **not** a source const.
- **N** proxima-tensor gate ≥ 478 + ≥ 3 new tests: (a) bucketed and unbucketed decode produce **identical token ids** over 24 tokens (§14 — the incumbent here is our own unbucketed path); (b) `plan_hits` ≥ 23 of 24 tokens at bucket 256; (c) the mask covers exactly `bucket - actual` positions. **N==0 is RED.**
- **PRED (milli)** `plan_hits` goes 0 → ≥ 23/24; `prepare_ms` falls from ~2.087 (R1) to **< 0.2 ms** on hit tokens; `block_upload_ms` falls because `mark_resident` (`metal.rs:350-362`) stops re-running per token.
- **KILL** token ids drift at any bucket size → the tail mask is wrong; revert (a decode loop that generates different text is not faster, it is broken). Or: `plan_hits` rises but `prepare_ms` does not fall → the cost was never in `plan_named`; decompose and re-instrument before touching anything else.
- **ROLLBACK** `git revert`; the bucket size 1 is the identity, so a config value of 1 is an instant runtime-level rollback without a rebuild only if the bucket is a runtime read — state which it is in the row.
- **BLAST** `proxima-model-interop/src/generate.rs` (caller), `proxima-tensor/src/spec.rs` (mask composition). Every model spec with a KV cache is affected — `append_mistral_cached_layer` and the qwen3.5 hybrid path (`0c3bd4f`, R0). Test both.
- **COUNTER** `plan_hits` / `plan_misses` / `plan_cache_len` (`generate.rs:1732`, `:1764`); `PREPARE_CALLS`/`PREPARE_TICKS` and `RESIDENT_BUFFER_REUSES` (`metal.rs:1983`).
- **REPROVE** decode cell; grep `plan_hits=` from the `token_breakdown_metal` lines.
- **LOG** ROW 243.

### P4.2 — preallocate per-op output and uniform buffers once per plan

- **WT** `proxima-wt-riscpool` · **BR** `perf/metal-plan-buffer-pool` · **TD** `.../proxima-wt-riscpool/target`
- **Commands** recover `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-all diff -- omega/src/metal.rs` and strip to the `metal-buffer-pool` feature (R7).
- **OPEN** `omega/src/metal.rs:2179-2252` — `encode_op`; `allocate_buffer(device, bound_output_len(bound), bound.dtype)` at `:2210` and `upload_uniforms(device, &pack_uniforms(bound))` at `:2211`, both **per op per token** = 1196 `newBufferWithLength` + 1196 uniform uploads (R11-M6″, = the 4.4 ms `op_setup`, R1). Note `UNIFORM_BUFFER_REUSES` already exists at `:2069` with a reuse path at `:2075` — read it first and report its current hit rate before assuming uniforms are the cost.
- **Precondition:** P4.1 must land first. A plan-stable program is what makes "allocate every output buffer once" *possible*; without `plan_hits` the pool is refilled every token.
- **N** omega gate ≥ 97; ≥ 2 new tests: (a) pooled and unpooled `execute_plan_named` produce identical outputs on the real forward; (b) the pool's live buffer count is bounded (strict O(1) in steady state, gate 11).
- **PRED (milli)** `op_setup_ms` falls from 4.394 (R1) to **< 1.0 ms**; `newBufferWithLength` calls/token fall from ~1196 to ~0 in steady state.
- **KILL** `op_setup_ms` falls but `step_wall_ms` does not, beyond CoV → the 11.4 ms orchestration slice overlaps GPU execution and removing it does not shorten the token; record that (it would be the same shape as R12's dispatch-count null) and stop Phase 4.
- **ROLLBACK** feature default-off; `git revert`.
- **BLAST** `omega/src/metal.rs` driver only. `device_buffers` lifetime and the retirement logic at `execute_plan:449-568` — a pooled buffer must not be retired mid-plan. This is the step most likely to produce a use-after-retire; the parity tests are the gate.
- **COUNTER** `OP_SETUP_CALLS`/`OP_SETUP_TICKS`, `UNIFORM_BUFFER_REUSES` (`metal.rs:2069`), plus a new pooled-output reuse counter mirroring it.
- **REPROVE** decode cell; read `op_setup_ms` and the reuse counters.
- **LOG** ROW 244.

### P4.3 — on-device argmax to remove the `waitUntilCompleted` dependency

- **WT** `proxima-wt-riscargmax` · **BR** `perf/metal-on-device-argmax` · **TD** `.../proxima-wt-riscargmax/target`
- **OPEN** `proxima-model-interop/src/generate.rs:1643` (`sample_next_token`), `:1649` (`next_ids`); R3-M7 states the true data dependency: `greedy_pick` argmax depends on `waitUntilCompleted`, so the fix is to move argmax on-device, not to thread.
- **Expressible today?** Yes — argmax is `Op::Reduce` with `Keep::Reduce` and a data-dependent `out_map`, which `op.rs:203-205` names explicitly ("Reduce, scan, scatter, contraction and **argmax**, distinguished by `Reduce::keep` and by whether `Reduce::out_map` is data-dependent"). That is a **scatter**, which every GPU emitter rejects (`msl.rs:932`) — so this step **depends on P5.3**, not the other way round. Sequence it after Phase 5.
- **N** ≥ 2 tests: on-device argmax matches `greedy_pick` bit-for-bit over 24 tokens on the real checkpoint.
- **PRED (milli)** `readback_ms` (R1: 0.297) falls to near-zero and the readback payload falls from a vocab-sized f32 vector to one `u32`; `readback_bytes` (`metal.rs` `READBACK_BYTES`) falls by ≥ 3 orders of magnitude.
- **KILL** token ids drift → §14, revert.
- **BLAST** `generate.rs` sampling path + one new emitted kernel route.
- **COUNTER** `READBACK_CALLS`/`READBACK_BYTES`.
- **REPROVE** decode cell; assert generated text unchanged.

---

# Phase 5 — write placement into a caller-owned persistent buffer (highest risk; IR + driver + all three emitters)

### P5.1 — write the expression before changing anything (§1's binary question)

- **WT** `proxima-wt-riscexpr` · **BR** `test/kv-placement-expression` · **TD** `.../proxima-wt-riscexpr/target`
- **Commands** a single new test in `proxima-tensor` that builds a KV-append as `Reduce { out_map: IndexMap::scatter(destination_extent = capacity, indices = Add(Iota(w), Constant(cached_len)), …) }` and runs it through `cpu::evaluate`.
- **OPEN** `proxima-tensor/src/map.rs:110-175` (the `Computed` write-direction convention), `:172-208` (`IndexMap::scatter` constructor), `:209-222` (`scatter_extent`); `proxima-tensor/src/bind.rs:961-996` (`bind_reduce`'s scatter arm), `:1011-1035` (`build_scatter_out_layout`); `proxima-tensor/src/cpu.rs:6892-6927` (`run_reduce_scatter`); `proxima-tensor/src/shape.rs:495-540` (`scatter_output_shape`).
- **N** ≥ 3 tests: (a) the scatter expression evaluates on CPU and places `w` rows at offset `cached_len` in a `capacity`-sized buffer; (b) it round-trips two successive appends; (c) it is rejected by `omega::msl::emit` with exactly `EmitError::ScatterNotSupported` (`msl.rs:932`) — asserting the *known* hole, so P5.3's fix has a red test to turn green. **N==0 is RED.**
- **PRED** none — this is a compile-and-evaluate proof, not a measurement. Its output decides P5.2 vs P5.4.
- **KILL** the expression does not compile or does not evaluate → the scatter form cannot express placement; **only then** does P5.4 (the affine-offset change to `project_output_shape`) become the route, and the row records exactly which clause blocked it.
- **BLAST** one test file. Zero production code.
- **REPROVE** `cargo nextest run -p proxima-tensor --features std -E 'test(kv_placement)'`
- **LOG** feeds ROW 246.

### P5.2 — measure the two placement forms head to head (nano)

- **WT** `proxima-wt-riscplace2` · **BR** `perf/placement-form-selection` · **TD** `.../proxima-wt-riscplace2/target`
- Form **S** (scatter): index tensor + indirection per write; costs an extra `Iota` + `Constant` + `Elementwise` node and a per-element index fetch.
- Form **A** (affine offset): `out_map` axis carries `offset = cached_len` and a declared destination extent; `layout_of` (`bind.rs:1594-1605`) already folds it to `Layout.base`; `msl.rs:2735` already emits `u.out_base`. Zero indirection, zero extra nodes.
- **PRED (nano)** form A costs **0 extra dispatches** and form S costs **3 extra nodes per KV write per layer** (= 96 extra dispatches at 32 layers × K/V). Given R12's dispatch-count null result, dispatch count is *not* the deciding metric; the deciding metric is per-write `execute_plan_op_timed` µs.
- **Selection rule, pre-registered:** form A unless its `project_output_shape` change (P5.4) fails a bounds test that form S passes. Form S remains implemented regardless, because P4.3 (argmax) needs scatter on GPU independently.
- **KILL** both forms exceed the unplaced baseline's per-write time → placement is not the lever and Phase 5 reduces to P5.3 (coverage) alone.

### P5.3 — scatter coverage in all three GPU emitters (removes a one-RISC coverage asymmetry)

- **WT** `proxima-wt-riscscat` · **BR** `feat/omega-scatter-coverage` · **TD** `.../proxima-wt-riscscat/target`
- **OPEN** `omega/src/msl.rs:923-945` (`validate`; the reject at `:932`), `omega/src/wgsl.rs:355-370` (reject at `:363`), `omega/src/cuda.rs:232-245` (reject at `:240`), `omega/src/cuda.rs:146-183` (`emit_cuda` also rejects `Iota` and `Constant` — fix in the same phase, P7.1); reference implementation `proxima-tensor/src/cpu.rs:6892-6927`; the field's own doc at `proxima-tensor/src/bind.rs:245-256` (destination axis fetched from `indices` at `index_layout`, bounds-checked against `extent`, scaled by `element_stride`, added to `out_layout`'s offset — "the exact mirror of how a gathered *operand*'s `Lookup` already contributes to a *read* offset").
- **Colliding writes:** `map.rs:110-115` records that the CPU interpreter runs the reduce loop strictly sequentially so "a scatter never needs atomics". A GPU emitter has no such guarantee. The KV-append case writes **disjoint** ranges by construction; the emitter must therefore either (a) prove disjointness from the index expression, or (b) emit the reduce body as an atomic for `Add`/`Maximum`/`Minimum` (associative — `op.rs:112-117`) and reject the rest. Decide by writing the disjointness check as a route (`KernelRoute::ScatterDisjoint` vs `KernelRoute::ScatterAtomic`) — a route value, not a new kind.
- **N** the red test from P5.1(c) turns green; omega gate ≥ 97 + ≥ 6 new tests (per backend: disjoint scatter parity vs CPU, colliding scatter parity vs CPU). `omega/tests/wgpu_parity.rs` and the CUDA emit tests must each gain ≥ 1. **N==0 is RED.**
- **PRED (nano)** a disjoint scatter dispatch costs within **±20%** of the equivalent affine-offset dispatch at the same extents; the gather-fault buffer machinery (`metal.rs:2214-2216`, `check_gather_fault` at `:2260`) extends to the write side with no additional dispatch.
- **KILL** parity fails on colliding writes on any backend → restrict the route to `ScatterDisjoint` and emit `EmitError` for the colliding case, recording the restriction as a named coverage gap with the reason (not as "unsupported").
- **ROLLBACK** `git revert`; the emitters return to rejecting, which is main's behaviour.
- **BLAST** all three emitters + the Metal driver's fault-buffer path. Highest blast radius in the plan. Land msl first, green, then wgsl, then cuda — three commits, each a green bisect point (AGENTS.md "git": primitives land before their callers; every commit a green bisect point).
- **COUNTER** `(NodeId, KernelRoute::ScatterDisjoint|ScatterAtomic)` census counts.
- **REPROVE** `cargo nextest run -p omega --features metal,cpu,instrument -E 'test(scatter)'` + `cargo nextest run -p omega --features wgpu-backend -E 'test(scatter)'`.
- **LOG** ROW 245.

### P5.4 — `project_output_shape` accepts a declared destination extent (form A, conditional on P5.2)

- **WT** `proxima-wt-riscwrite` · **BR** `feat/tensor-write-placement-offset` · **TD** `.../proxima-wt-riscwrite/target`
- **OPEN** `proxima-tensor/src/shape.rs:469-486` (`project_output_shape` — the one match arm), `:440-467` (`bounds_check`, which must gain the write-side check), `:495-540` (`scatter_output_shape`, the sibling convention to mirror); `proxima-tensor/src/bind.rs:1594-1605` (`layout_of`, already correct); doc that must be corrected in the same commit: `proxima-tensor/src/spec.rs:2303-2319` ("`Reduce::out_map` must stay a pure projection … so nothing upstream of a reduce can splice two tensors into one axis") and the comment at `:2616-2617`.
- **N** proxima-tensor gate ≥ 478 + ≥ 4 new tests: (a) a placed write lands at the declared offset; (b) an offset that would exceed the declared extent is rejected by `bounds_check` with `IndexOutOfBounds`; (c) two producers writing disjoint ranges of one destination compose; (d) `Op::Reduce` variant count is still 5 and `BoundOpKind` variant count is still 4 (an assertion test, so the one-RISC rule is mechanically re-proved — §16). **N==0 is RED.**
- **PRED (nano)** a placed reduce emits **one** dispatch with a non-zero `u.out_base`, identical thread count to the unplaced form; `execute_plan_op_timed` within ±5%.
- **KILL** the change makes any existing `out_map` ambiguous (a pure projection whose offset was previously ignored now changes meaning) → the convention must be explicit at the constructor, mirroring `IndexMap::scatter`; add `IndexMap::placed(...)` as a **constructor**, not a variant, exactly as `map.rs:172` does for scatter.
- **ROLLBACK** `git revert`. This one is hard to roll back after callers adopt it — land P5.4 and its first caller (P5.5) as separate commits so the caller can revert alone.
- **BLAST** `shape.rs`, `spec.rs` docs, and every existing `Reduce` — the bounds check changes for all of them. Run the full proxima-tensor and proxima-autograd suites (the adjoint path reads `as_gather_from_output`, `map.rs:238`).
- **COUNTER** the census count of reduces with `out_layout.base != 0`.
- **REPROVE** `bash scripts/proxima-tensor-gate.sh`
- **LOG** ROW 246.

### P5.5 — the driver-level persistent-buffer alias; KV stops round-tripping the host

- **WT** `proxima-wt-riscdrive2` · **BR** `perf/kv-device-resident` · **TD** `.../proxima-wt-riscdrive2/target`
- **Commands** recover `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-drive diff` (R7: 7 files +2375/−96) as *reference for the shape only*; rebase onto P5.4's tip and rewrite against the aliasing insert rather than any new type.
- **OPEN, in this order**
  1. `omega/src/metal.rs:2249` — `device_buffers.insert(bound.node, (output, 0))`; the `(buffer, offset)` pair that is the whole mechanism.
  2. `omega/src/metal.rs:2210` — `allocate_buffer` for the output (the call that must be skipped for a placed node).
  3. `proxima-model-interop/src/generate.rs:621-654` — `LayerCache { k_even, k_odd, v: Vec<f32> }`, `append` = 3× `extend_from_slice` at `:636-640`, `named_blocks` handing the whole `Vec` as `QuantizedBlock::Float32` at `:642-653`.
  4. `omega/src/metal.rs:1848-1892` — `NOCOPY_BUFFERS: BTreeMap<(usize,usize), MetalBuffer>` keyed `(pointer, byte_length)`; `upload_block_no_copy` at `:1879-1892`; `upload_block_no_copy_uncached` at `:1903` (23e2e5e routes non-resident blocks here, so KV creates a fresh no-copy buffer every token — R11-M2′).
  5. `omega/src/metal.rs:350-362` — `mark_resident` classifies by NAME.
  6. `omega/src/metal.rs:991-1000` and `proxima-tensor/src/cpu.rs:346-356` — the strict `found != expected` `InputSizeMismatch` that blocks over-allocated buffers.
  7. `proxima-tensor/src/align.rs:42-46`, `:69` — `AlignedBuffer`, **zero production callers**; this is its first (§1: extend the existing primitive rather than mint a peer). The only current use is `omega/examples/resident_gemv_topk.rs:275`.
  8. Incumbent: `llama-kv-cache-unified.cpp:74-118` (allocate once), `:749-788` (`ggml_cpy` into `ggml_view_1d` at a byte offset), V stored transposed since `v_trans = !flash_attn` (R8).
- **N** proxima-model-interop tests green + ≥ 4 new: (a) 24-token decode produces identical text with and without device-resident KV; (b) `block_upload_bytes` for KV blocks is 0 after the first token; (c) `InputSizeMismatch` accepts an over-allocated destination when the declared extent is the capacity; (d) the KV allocation is bounded (R3-M11 found a 34 GB allocation from a `context_length` default — assert the allocation size explicitly).
- **PRED (milli)** `block_upload_ms` falls from 2.201 (R1) toward the weight-only residual; `block_upload_bytes` per token falls by the KV cache size and stops growing with context (R2: KV re-upload ~3.3%, growing).
- **KILL** generated text drifts → §14, revert. Or: `block_upload_ms` falls but `step_wall_ms` does not → record and stop; the KV slice was 3.3%, and a 3.3% claim that does not show at the wall is inside the CoV band, which is the honest read.
- **ROLLBACK** `git revert` P5.5 alone (P5.4 stays; it has its own tests).
- **BLAST** the Metal driver's buffer lifetime, `LayerCache`, `named_blocks`, `mark_resident`, and the CPU backend's matching size check. Widest correctness surface in the plan after P5.3.
- **COUNTER** `BLOCK_UPLOAD_BYTES`, `NOCOPY_BUFFER_REUSES` vs `NOCOPY_BUFFER_UPLOADS` (`metal.rs:1985`, `:1983`), `COPYING_BUFFER_UPLOADS`, `MAPPING_OFFSET_UPLOADS` (`:1779`), and `nocopy` cache length (`:1858`).
- **REPROVE** decode cell; assert `block_upload_bytes` is flat across tokens 2..24.
- **LOG** ROW 246.

---

# Phase 6 — the minimal graph (depends on Phase 5; demoted from the brief's position by R12)

### P6.1 — ops-per-layer census against the incumbent's 23

- **WT** `proxima-wt-riscmin` · **BR** `perf/ops-per-layer-census` · **TD** `.../proxima-wt-riscmin/target`
- **OPEN** `proxima-tensor/src/spec.rs:2336-2865` (`append_mistral_cached_layer`, 530 lines, 25 args); incumbent's 23 real ops/layer enumerated at `llama-model.cpp:4691-4845` + `llama-graph.cpp:464-497, 514-616, 1071-1121, 1205-1253` (R8), views/reshape/permute no-ops at `ggml-metal.m:1835-1847`, total ~740 dispatches/token.
- **N** the census prints ops/layer for our graph and the count is asserted in a test. **N==0 is RED.** Main's number to confirm: 37 ops/layer, 1196 dispatches (R3-M1, R1) — note R8 corrects the "15/layer, ~483" memory figure to **23/layer, ~740**, so the ratio is 1.62×, not 2.5×.
- **PRED** none (census).
- **KILL** n/a.
- **REPROVE** decode cell + the census assertion.

### P6.2 — collapse the two-range online softmax to one range using placement

- **WT** `proxima-wt-riscmerge2` · **BR** `perf/attention-single-range` · **TD** `.../proxima-wt-riscmerge2/target`
- **Commands** recover `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-merge diff` (R7: 1 file +1181/−82, spec.rs) as reference; rebase onto P5.5. Main has moved spec.rs +8735/−2836 since (`0c3bd4f`, R0) — expect a full conflict; rewrite rather than merge.
- **OPEN** `proxima-tensor/src/spec.rs:2596-2720` (the online-softmax combine, "no literal concatenation anywhere" at `:2616-2617`), `:2303-2319` (the doc that P5.4 already corrected), plus the even/odd RoPE split that doubles it again (R3-M1). Incumbent's decode attention: `mul_mat(K,Q)` → `soft_max_ext` (fused scale+mask+max+exp+sum+normalize in ONE kernel, `ggml-metal.metal:1051-1145`, nth up to 256/ne00 via `ggml-metal.m:2501-2525`) → `mul_mat(V)` (R8).
- **N** ≥ 488 parity assertions green (R3-M11 records 488/488 proven for the single-range graph); ops/layer census falls; 24-token text identical.
- **PRED (milli, one rung past P6.1's census)** ops/layer falls from 37 to **≤ 25**; dispatches/token falls from ~1196 to **≤ 950** (R3-M11: 939). `gpu_exec_ms` falls by **≤ 2 ms** — deliberately small, because R12 ROW 262/267 measured 1194 → 616 dispatches moving wall by 0.036 ms.
- **KILL** `gpu_exec_ms` does not fall beyond CoV → the graph is not the mass, and this is the *second* independent measurement saying so (R12 is the first). Record the conclusion (two agreeing results, §19 rung 3) and stop Phase 6. The single-range graph still lands if it is correctness- or maintenance-positive, but not as a perf row.
- **ROLLBACK** `git revert`.
- **BLAST** `spec.rs`'s attention builder — every model that uses it, including the qwen3.5 hybrid attention/ssm path from `0c3bd4f`.
- **COUNTER** ops/layer census + `encode_dispatch_calls`.
- **REPROVE** decode cell + P6.1's assertion.
- **LOG** ROW 247.

---

# Phase 7 — backend coverage parity and the emitter core

### P7.1 — CUDA covers `Iota` and `Constant`

- **WT** `proxima-wt-risccuda` · **BR** `feat/cuda-iota-constant` · **TD** `.../proxima-wt-risccuda/target`
- **OPEN** `omega/src/cuda.rs:146-183` (`emit_cuda`, `CudaUnsupportedOpKind`); reference `omega/src/msl.rs:2044-2103` (`render_iota`, `render_constant`).
- **N** ≥ 2 new emit tests per kind; `cargo nextest run -p omega --features cuda` ≥ 1. **N==0 is RED** — `cuda` is not in `default` (`omega/Cargo.toml`), which is exactly gate 3's hiding condition.
- **PRED** none (coverage, not perf). The one-RISC claim "every backend covers every kind" is falsifiable by a `grep -c UnsupportedOpKind` == 0 assertion.
- **KILL** n/a.
- **BLAST** `omega/src/cuda.rs`.
- **REPROVE** `cargo nextest run -p omega --features cuda,cpu` and `grep -c "CudaUnsupportedOpKind" omega/src/cuda.rs` == 0 (excluding the error-type definition if the variant is retained for a genuinely unreachable case, with a one-line why).
- **LOG** ROW 248.

### P7.2 — one emitter core; backend-specific text tables

- **WT** `proxima-wt-risccore` · **BR** `refactor/omega-one-emitter-core` · **TD** `.../proxima-wt-risccore/target`
- **OPEN** the ~26 functions R5 lists as reimplemented per backend (`validate`, `reduction_dims`, `bindings`, `grid_threads`, `entry_name`, `scalar_op_expr`, `fold_init_tokens`, `push_body_steps`, `preamble`, `kernel_signature`, gather helpers, `operand_read`, `render_*`, `reduce_is_cooperative`) ≈ 78 near-duplicates across `msl.rs` (4712 lines), `wgsl.rs` (1929), `cuda.rs` (1838); the 3 types + 2 fns already shared (`Binding`, `PackedCodec`, `PackedOperands`, `gather_count`, `gather_slots` — `wgsl.rs:105`, `cuda.rs:66`).
- **The shape:** one generic core over the 4 kinds parameterized by a `struct BackendText { … }` value table (intrinsic names, type tokens, signature syntax). **A value table, not a trait** — §20 box-free, §11 no dynamic dispatch, and a table is data so §4's config-as-composition question has an answer.
- **N** omega gate ≥ 97 unchanged; wgpu parity tests unchanged; a line-count assertion recorded in the row (the 12,553-line total, R5, is the before).
- **PRED (milli)** behaviour-neutral: `gpu_exec_ms` and emitted MSL byte-identical for every route on the real forward. Assert byte-identity of emitted source before/after as a test — that is the strongest possible neutrality gate and it is cheap.
- **KILL** emitted source is not byte-identical for any route → the refactor changed behaviour; bisect by route.
- **ROLLBACK** `git revert`; large but mechanically clean if the byte-identity test exists.
- **BLAST** all three emitters. Land per-kind, one commit per kind, each green.
- **REPROVE** the byte-identity test + omega gate.
- **LOG** ROW 248.

---

# Phase 8 — the incumbent cells that do not exist yet

R9 is explicit: "llama, ggml, ort and torch can beat us on gpu" is MEASURED only for llama.cpp-Metal. For ORT (CoreML EP) and torch (MPS) **there is no cell on either side**. These are not verification tasks; they are the tasks that determine whether the claim is true.

### P8.1 — torch-MPS arms for mnist inference, MLP train step, and BGE

- **WT** `proxima-wt-risctorch2` · **BR** `bench/torch-mps-arms` · **TD** n/a (python fixture)
- **Commands**
  ```
  proxima-onnx/scripts/torch_reference/venv/bin/python -c "import torch;print(torch.__version__, torch.backends.mps.is_available())"   # verified: 2.13.0 True
  ```
  Add a `--device {cpu,mps}` argument to `proxima-onnx/scripts/torch_reference/inference_bench.py` and `train_bench.py`. **Verified gap:** `grep -n "mps\|device" ` over both files returns **nothing** — both are CPU-only today, so this is new arms, not a flag flip.
- **OPEN** `proxima-onnx/scripts/torch_reference/README.md` (the named, scoped python exception and what each script re-proves), `inference_bench.py`, `train_bench.py`, `diagnostics.py`; ours: `omega/tests/training_step_parity.rs:400-607` (GPU train step exists only as **untimed** parity tests, R9).
- **N** each arm emits ≥ 5 runs with CoV. **N==0 is RED** — an MPS arm that silently falls back to CPU exits 0; assert `next(model.parameters()).device.type == "mps"` in the harness.
- **PRED (bench)** torch-MPS on the mnist 14-node graph is **slower than torch-CPU** on this box, because the graph is tiny and per-op MPS dispatch dominates. Pre-registered so the opposite result is informative.
- **KILL** MPS silently falls back → the arm is void; report it as a feature gap in torch's own harness, not as our win.
- **BLAST** two python fixture files.
- **COUNTER** the device assertion + `torch.mps.current_allocated_memory()`.
- **REPROVE** `.../venv/bin/python inference_bench.py --device mps` and `train_bench.py --device mps`.
- **LOG** ROW 249.

### P8.2 — ORT-CoreML arms

- **WT** `proxima-wt-riscort` · **BR** `bench/ort-coreml-arms` · **TD** n/a
- **Commands** verified: `onnxruntime` is **not installed** in `proxima-onnx/scripts/torch_reference/venv` (`ModuleNotFoundError`). `scripts/onnx_reference/` already exists on main with a pinned-venv `run.sh` (`scripts/onnx_reference/run.sh:1-40`) — extend that harness with a `CoreMLExecutionProvider` arm rather than creating a second venv (§1).
- **OPEN** `scripts/onnx_reference/run.sh`, `scripts/onnx_reference/bench.py`, `scripts/onnx_reference/README.md`; source checkout at `~/repos/others/onnxruntime` (R0) if a wheel is unavailable.
- **N** `providers=['CoreMLExecutionProvider','CPUExecutionProvider']` and the harness asserts `session.get_providers()[0] == 'CoreMLExecutionProvider'`. **N==0 is RED** — ORT silently falls back to CPU when CoreML rejects a node, and that fallback is exactly the failure this assertion exists to catch. Also assert the **partition count** (how many subgraphs CoreML actually took) — a 1-node CoreML partition with 200 CPU nodes is a CPU cell wearing a CoreML label.
- **PRED (bench)** ORT-CoreML on BGE-small lands **within 2× of ORT-CPU** on this box, and the CoreML partition covers < 100% of nodes.
- **KILL** the wheel does not install for the pinned interpreter (`ONNX_REF_PYTHON` defaults to `python3.12`, `run.sh:19-21`) → build from `~/repos/others/onnxruntime`; if that also fails, the cell is a documented **feature gap** ("cannot run the arm"), never omitted (§19: losses omitted are a verdict; and gate 13's "cannot run the arm" is a legitimate honest verdict).
- **BLAST** `scripts/onnx_reference/` only.
- **REPROVE** `BGE_MODEL_PATH=... ONNX_REF_PROVIDER=coreml bash scripts/onnx_reference/run.sh`
- **LOG** ROW 250.

### P8.3 — our own GPU arms for mnist / train / BGE (there are none)

- **WT** `proxima-wt-riscarms` · **BR** `bench/omega-nondecode-gpu-arms` · **TD** `.../proxima-wt-riscarms/target`
- **OPEN** `omega/benches/metal_vs_cpu.rs` (the **only** GPU bench outside decode: gemm_square_f32 512/1024/2048, matvec_batch1_f32 at Mistral f32 shapes), registered at `omega/Cargo.toml:207-210`, **doc says UNRUN** (R9); `omega/tests/training_step_parity.rs:400-607` (GPU train step, untimed).
- **Commands** `cargo bench -p omega --bench metal_vs_cpu --features metal` — run it, for the first time. Then add arms: mnist f32 inference, MLP train step, BGE-small — each with `design-favors: incumbent` labels against P8.1/P8.2's cells.
- **N** the bench emits ≥ 5 arm results. **N==0 is RED** — a `required-features`-gated bench that is never invoked compiles to nothing.
- **PRED (micro)** `gemm_square_f32` at 2048 is the arm where Metal beats CPU by the widest margin; `matvec_batch1_f32` at Mistral shapes is where it is narrowest — the same low-row starvation R3-M5 names (52 GB/s at 256 simdgroups).
- **KILL** the bench does not build under `--features metal` → fix the registration; a bench that does not build is gate-5 red.
- **BLAST** `omega/benches/metal_vs_cpu.rs`.
- **REPROVE** `cargo bench -p omega --bench metal_vs_cpu --features metal`
- **LOG** ROW 251.

---

# Phase 9 — ai_docs records and the board

### P9.1 — `ai_docs` JSONL records for the GPU lane

- **WT** `proxima-wt-riscdocs` · **BR** `docs/ai-docs-gpu-lane` · **TD** n/a
- **Verified gap:** `ai_docs/task-routes.jsonl` (18 records) and `ai_docs/invariants.jsonl` (31 records) have **zero** tensor/omega/GPU records (R0). `ai_docs/AGENT.md` step 5: "If the index is missing required structure or evidence, **add records** to `ai_docs` instead of bypassing the structure."
- **`index.jsonl`** — schema `{id, kind, summary, path, read_when[], source_paths[], relations[]}` (verified against record 1):
  - `proxima.omega.gpu_decode_lane` → `path: proxima-tensor/docs/discipline.md`, `read_when: ["gpu-decode","metal","q4k","omega"]`, `source_paths: ["omega/src/msl.rs","omega/src/metal.rs","proxima-model-interop/src/generate.rs","proxima-tensor/docs/rooflines.md"]`
  - `proxima.omega.kernel_route_census` → `source_paths: ["omega/src/msl.rs","proxima-tensor/src/instrument.rs"]`
  - `proxima.tensor.write_placement` → `source_paths: ["proxima-tensor/src/shape.rs","proxima-tensor/src/map.rs","proxima-tensor/src/bind.rs"]`
- **`task-routes.jsonl`** — schema `{task, purpose, must_read[], then_read_if_relevant[], queries[], done_when[]}`:
  - `{"task":"gpu-decode-perf", "must_read":["ai_docs/AGENT.md","ai_docs/invariants.jsonl","proxima-tensor/docs/rooflines.md"], "queries":["jq -c 'select(any(.applies_to[]?; .==\"gpu-decode\"))' ai_docs/invariants.jsonl"], "done_when":["board cell carries CoV and a roofline cell","incumbent arm present","route census sums to encode_dispatch_calls"]}`
  - `{"task":"omega-kernel-emission", …}`
- **`invariants.jsonl`** — schema `{id, kind, summary, rule, applies_to[], evidence_required[], relations[]}`:
  - `proxima.omega.one_risc_bound_kinds` — rule: "`BoundOpKind` has exactly 4 variants and `Op` exactly 5; a new variant for one model's operator shape is an instance rule, not an instruction. Express it through `Reduce.out_map` placement or a `KernelRoute`." evidence_required: the variant-count assertion test from P5.4(d), plus `git show perf/cached-attention-streaming:failure-cached-attention-matcher.md`.
  - `proxima.omega.route_is_a_value_not_a_substring` — rule: "kernel route is a first-class enum decided before emission and censused `(NodeId, KernelRoute)`; never recovered by substring of emitted source." evidence_required: the census-sum test from P1.2; the defect record `omega/src/metal.rs:785-826` and R12 ROW 263's 9/601 → 225/385 relabelling.
  - `proxima.omega.backend_covers_every_kind` — rule: "every backend emits every `BoundOpKind` and honours every field including `out_scatter`." evidence_required: `grep -c ScatterNotSupported` == 0 and `grep -c CudaUnsupportedOpKind` == 0.
  - `proxima.omega.geometry_traces_to_sizing_config` — rule: §12 for omega geometry. evidence_required: `grep -cE "^const (PACKED_ROWS_PER_GROUP|TILE_DIM|TILED_GEMM_NSG)" omega/src/msl.rs` == 0.
  - `proxima.gpu.dispatch_count_is_not_the_denominator` — rule: "do not schedule work on dispatch count alone." evidence_required: R12 ROW 262/267 (1194 → 616 dispatches, wall 51.571 → 51.535).
- **N** `wc -l` deltas: index +3 (23→26), task-routes +2 (18→20), invariants +5 (31→36). Each file must remain valid JSONL: `jq -c . ai_docs/*.jsonl > /dev/null` exit 0. **N==0 on any file is RED.**
- **REPROVE** `bash ai_docs/query.sh gpu-decode-perf`; `jq -c 'select(any(.applies_to[]?; .=="gpu-decode"))' ai_docs/invariants.jsonl`

### P9.2 — final board re-seal and the roofline lane

- **WT** `proxima-wt-riscboard` · **BR** `bench/gpu-board-reseal-final` · **TD** `.../proxima-wt-riscboard/target`
- **Commands** `bash scripts/sealed-pass.sh` on a quiet box, with every landed feature in `default`.
- **N** every board cell filled: ours (`step_wall_ms`, `gpu_exec_ms`, CoV, n), llama.cpp arm A, llama.cpp arm B (`-fa 1`), torch-MPS, ORT-CoreML, roofline fraction. **A blank cell is RED** (gate: "don't move on with blank cells").
- **PRED (bench)** the plan makes exactly one board-level prediction, and it is stated once here: with P2.1 + P3.1 + P4.1 + P4.2 landed, `step_wall_ms` lands in **30–40** and the ratio vs arm A in **1.7×–2.3×**. Derivation, all cited: P2.1 predicts `gpu_exec` 34–42 (anchored on R12's 35.117 control); P4.1+P4.2 predict the 11.4 ms orchestration slice (R1) falls to ≤ 3 ms; 35 + 3 ≈ 38, against arm A's 17.470 (R1) = 2.2×. Phases 5–6 are **not** in this prediction, because R12's null result gives no basis to predict a wall movement from them.
- **KILL** the ratio does not fall below 3.0× → the decomposition in R2 is wrong somewhere, and the row names which bucket did not move.
- **REPROVE** `bash scripts/sealed-pass.sh`
- **LOG** ROW 252; update `proxima-tensor/docs/rooflines.md:396-479`, the summary row at `:751`, and rewrite the closing note at `:766-773`.

---

## Dependency graph

```
P0.1 (re-seal baseline) ──┬─> everything (no Δ is claimable without it)
P0.2 (llama harness + membw) ──> P1.1 (roofline), P1.4 (-fa 1 arm)
P0.9 (clean tree) ──> every interleaved measurement

P0.3 (mask-fma) ─┐
P0.4 (pair-dot) ─┼─> P2.1 (choose ONE) ──> P2.2 (split-K) ──┐
P1.2 (KernelRoute census) ─┘  ^                              │
                              └── P1.2 is a HARD precondition │
                                  of P2.1 (R5-M10, R12 ROW 263)│
P0.5 (wide reduce) ──> P3.1 (promote to default) ─────────────┤
P1.3 (geometry consts) ──> P3.1 (the width is a config key)   │
P3.2 (elementwise census) ──> (no lever OR Phase 6)           │
                                                              ├─> P9.2 (final board)
P4.1 (cached_len bucket) ──> P4.2 (buffer pool) ──────────────┤
     ^ P4.2 is impossible before P4.1 (a pool refilled per     │
       token is not a pool)                                    │
                                                              │
P5.1 (write the expression) ──> P5.2 (form selection) ──> P5.4 (affine offset)
P5.1 ──> P5.3 (scatter coverage, all 3 emitters) ──> P4.3 (on-device argmax)
P5.3 + P5.4 ──> P5.5 (persistent buffer alias / KV residency) ──> P6.2 (single-range graph)
P6.1 (ops/layer census) ──> P6.2
P5.3 ──> P7.1 (CUDA kinds) ──> P7.2 (one emitter core)
P0.6 (adjudication) ──> P0.7 (adopt prune_dead only) ; blocks any CachedAttention merge
P8.1/P8.2/P8.3 (incumbent + our non-decode arms) ──> P9.2
P9.1 (ai_docs) ── independent, land any time after P1.2 names KernelRoute
P0.8 (row numbers from 233) ── land-time gate on every row-producing step
```

**Critical path to the board prediction:** P0.1 → P1.2 → P2.1 → P3.1 → P4.1 → P4.2 → P9.2.
**Everything in Phase 5–7 is off that path** and is sequenced for correctness/one-RISC conformance, not for the predicted wall movement.

---

## Rollback map

| step | rollback | is main's default affected before rollback? |
|---|---|---|
| P0.1, P0.2, P1.1, P1.4, P6.1, P3.2, P8.* | `git revert`; measurement/example/script only | no |
| P0.3, P0.4, P0.5, P2.2 | feature is default-off in `omega/Cargo.toml`; delete the feature | no |
| P0.6, P0.8, P9.1 | docs/JSONL revert | no |
| P0.7 (`prune_dead`) | `git revert` one commit | yes — lands in `default` |
| P1.2 (`KernelRoute`) | `git revert`; instrument-gated recorder, but `route_of` itself is on the default path | partially — `classify_kind`'s deletion is default-path |
| P1.3 (geometry consts) | `git revert` build.rs + toml + msl const sites together | yes, but value-identical by construction |
| P2.1 promotion | demote the winning feature out of `default`, one line | yes |
| P3.1 promotion | demote out of `default`, one line | yes |
| P4.1 (`cached_len` bucket) | set the bucket to 1 (identity) via the sizing config, then `git revert` | yes |
| P4.2 (buffer pool) | feature default-off; `git revert` | no while gated |
| P4.3 (argmax) | `git revert` the sampling-path commit alone | yes |
| P5.3 (scatter coverage) | `git revert` per backend, 3 independent commits | no (adds capability; nothing on the default path emits scatter until P5.5) |
| P5.4 (`project_output_shape`) | `git revert`; **land P5.4 and P5.5 as separate commits** so the caller reverts alone | yes — bounds-check semantics change for every existing `Reduce` |
| P5.5 (KV residency) | `git revert` | yes |
| P6.2 (single-range graph) | `git revert` `spec.rs` | yes — affects every model using `append_mistral_cached_layer`, including qwen3.5 |
| P7.2 (emitter core) | `git revert` per kind, 4 independent commits, each gated by the byte-identity test | yes, but byte-identical by construction |

**Blast-radius ranking (widest first):** P5.4 → P5.5 → P6.2 → P7.2 → P5.3 → P4.1 → P4.2 → P1.2 → P3.1/P2.1 promotion → everything else.

---

## Abandoned designs (and the constraint that ruled each out)

1. **`BoundOpKind::CachedAttention` — a fifth bound kind matched by a post-bind structural matcher over one model's attention cluster** (R12, `perf/cached-attention-streaming`, `physical.rs` +576, `bind.rs` +666).
   **Ruled out by:** workspace `AGENTS.md` "problem solving" — *"we should not be adding arbitrary rules/code for specific instances"* — and guiding-principles §1's binary question (an existing primitive, `Reduce.out_map` placement, can express it; write the expression, which is P5.1). Corroborating evidence, not the reason: the branch's own `failure-cached-attention-matcher.md` abandoned the first matcher as a heuristic that "cannot prove the semantic roles", and its numbers show wall unmoved (51.535 vs 51.571) with `gpu_exec` worse (39.841 vs 35.117). **What survives from that branch:** `prune_dead` (P0.7) and the paired Q4_K body (P0.4).

2. **A new `Op::Concat` (or `Op::Pad`/`Op::Tile`) variant to splice the KV cache and the new token into one axis.**
   **Ruled out by:** §1 reuse-first plus the read of the code that §6 demands. `bind::layout_of` (`bind.rs:1594-1605`) already folds `axis.offset * stride` into `Layout.base`; `msl.rs:2735` already emits `long out_offset = u.out_base;`; `IndexMap::Computed` already carries the write-direction destination-extent convention (`map.rs:110-131`). The only thing missing is one match arm in `shape::project_output_shape` (`shape.rs:469-486`). A new generator would have added a sixth `Op` variant to work around a five-line shape rule. `spec.rs:2303-2319`'s claim that "nothing upstream of a reduce can splice two tensors into one axis" is a statement about `project_output_shape`, not about the algebra — and it is corrected in the same commit.

3. **A `PlacedBuffer` / `write_placement` type in the Metal driver to own the persistent KV allocation.**
   **Ruled out by:** §1's relocation question — *write the call site both ways.* `omega/src/metal.rs:2249` is already `device_buffers.insert(bound.node, (output, 0))` over a `BTreeMap<NodeId, (MetalBuffer, usize)>`. Aliasing is `insert(node, (persistent, offset))`. Identical lines at the call site ⇒ the type is a relocation. Additionally §1's "extend an existing primitive": `AlignedBuffer` (`proxima-tensor/src/align.rs:42-46`, `:69`) already exists with **zero production callers**; a new peer next to an unused primitive is debt twice over.

4. **A `trait KernelBackend` / trait-object emitter registry to unify msl/wgsl/cuda's ~78 near-duplicate functions (R5).**
   **Ruled out by:** §20 (box-free by default; "a discriminated enum + match, typestate, a generic parameter") and §11 ("Forbidden: trait objects"). The replacement is a generic core parameterized by a **value table** of backend text — which additionally satisfies §4 config-as-composition, because a table is data and a trait impl is a recompile. P7.2.

5. **Threading `op_setup` / the encode loop to hide the 4.4 ms orchestration slice.**
   **Ruled out by:** §21 (a lock is a missing owner — the owner here is the plan, and the fix is to stop re-deriving it) plus two measured negatives: R3-M9 (non-`Send` `MTLBuffer` blocks it at the type level; the plain-data half was landed as `PROXIMA_ORCH_THREADS` and remains unmeasured) and R4 ("Threading `-t`: incumbent is GPU-bound; thread count explains zero of the gap"). R11-M6″ shows the cost is 1196 `newBufferWithLength` calls, i.e. allocation, not serialism — which is P4.2, a pool, not threads.

6. **`cached_len` as a runtime uniform so the bound plan is shape-independent.**
   **Ruled out (parked, not resolved) by:** blast radius against §15's legitimate-deferral surface. `BoundOp.extents` is a baked `Vec<u64>` (`bind.rs:200-215`); making it dynamic changes every backend's uniform packing, every `entry_name`/`kernel_cache_key` axis, and every grid computation. **Named claim it gates:** a plan cache with a 100% hit rate at every context length. **Named un-park condition:** P4.1's capacity bucket measures a hit rate below 255/256, or the mask tail is measured to cost more than the re-plan it replaces. The measured cost stays a row (§ parking hard limits), not a deletion.

7. **Rematerializing the low-element node set to cut dispatches (R3-M8: 1196 → 1100 on the `elements < 247` subset).**
   **Ruled out by:** R4's dead-lever list ("Rematerialize all ≤2-consumer nodes: 6× downside on the slow ALU arm") *combined with* R12's measurement that dispatch count is not the denominator (1194 → 616 moved wall by 0.036 ms). A 96-dispatch reduction cannot matter when a 578-dispatch reduction did not.

8. **nsg=2 / ggml packed-simdgroup threadgroup regrouping.**
   **Ruled out by:** four independent measured negatives — R4 (−2.36% on the real graph, "third time tried"), R0 (`perf/metal-simdgroup-geometry`, measured LOSS), R12 ROW 267 ("ggml's two-SIMD-group Q4_K geometry regresses the cached-feature wall cell"), R12 ROW 259/260. §7 (negative results recorded) makes re-proposing this a discipline failure, not an experiment.

9. **A kernel-fusion engine (rms_norm + mul, softmax chains) as the route to parity.**
   **Ruled out by:** §14's framing of what parity requires plus R8's read of the incumbent at the exact checkout llama-bench runs: `grep -rln fuse ggml/src` is **empty** at `b25346221`; rms_norm and mul dispatch as two kernels. Parity is reachable without a fusion engine. Fusion is upside past parity, and scheduling it now would spend the budget on the thing the incumbent does not do.

---

## Open questions the plan resolves by measurement (never by asking)

| # | question | the step that answers it | the measurement, not the opinion |
|---|---|---|---|
| Q1 | Which Q4_K body — `metal-q4k-mask-fma` (R3-M3) or `q4k_pair_dot` (R12 ROW 257)? | **P2.1** | Interleaved A/B at one commit with the route census pinned; lower median `gpu_exec_ms` **only if** the `PackedRowBlock` route counts match. Unequal counts void the comparison. |
| Q2 | Does the 2026-09-02 `-17.2%` / `-4.9%` / `-11.1%` set survive a rebase onto 9 commits of main including `spec.rs` +8735/−2836? | **P0.3, P0.4, P0.5, P2.1** | Re-measured on the P0.1 baseline. R7 says every diff conflicts; the numbers are re-earned, not carried. |
| Q3 | Did `ff749a0` (plan-cache bound) and `7d09145` (checkpoint mapping once) already move the 11.4 ms orchestration slice? | **P0.1** | `prepare_ms` / `block_upload_ms` / `op_setup_ms` on `4be2f3a` vs R1's `2b95210` figures. R3-M6 flags this explicitly as needing re-measurement. |
| Q4 | Does bucketing `cached_len` to a capacity multiple take `plan_hits` from 0 to ≥23/24, and does that actually remove `prepare_ms`? | **P4.1** | `plan_hits=` in `token_breakdown_metal`; `prepare_ms` on hit vs miss tokens. Two separate numbers — the second does not follow from the first. |
| Q5 | Is write placement cheaper as an affine `out_map` offset or as a scatter? | **P5.2** | `execute_plan_op_timed` µs per placed write, both forms, same extents. Form A costs 0 extra nodes; form S costs 3 per write per layer — but R12 says node count is not the denominator, so the timer decides. |
| Q6 | Does removing 250+ dispatches via the single-range graph move the wall at all? | **P6.2** | `gpu_exec_ms` and `step_wall_ms` deltas. R12 already measured 1194→616 moving 0.036 ms; P6.2 is the second, independent test of the same proposition. Two agreeing negatives would be a §19 rung-3 conclusion. |
| Q7 | Do torch-MPS and ORT-CoreML actually beat us on GPU, on any lane? | **P8.1, P8.2, P8.3** | R9: there is **no cell on either side** today. The owner's premise is measured for llama.cpp-Metal only. |
| Q8 | What is the GPU streaming-bandwidth ceiling, so a fraction-of-ceiling exists at all? | **P1.1** | Marginal GB/s with readback outside the timed window. A "ceiling" below the incumbent's achieved 228.9 GB/s (R1) is not a ceiling and the debt row stays open. |
| Q9 | Does `-fa 1` change which llama.cpp arm is the home-turf incumbent? | **P1.4** | Two arms, 5 runs each. If arm B is >20% faster, every ratio in R1/R2/R10 re-bases to it. |
| Q10 | Is the elementwise bucket (6.65 ms, R2) concentrated or uniform? | **P3.2** | Top-5 nodes' share of the bucket. Uniform ⇒ no single-node lever exists and the phase closes with that pointer. |
| Q11 | Can a GPU scatter be emitted without atomics for the disjoint case? | **P5.3** | Parity vs `cpu::run_reduce_scatter` on disjoint and colliding fixtures, per backend. `map.rs:110-115` guarantees sequential order on CPU only. |
| Q12 | Does the 3.3% KV re-upload slice (R2) show at the wall once KV is device-resident? | **P5.5** | `block_upload_bytes` flat across tokens 2..24, then `step_wall_ms` delta against both CoV bands. A 3.3% claim inside the noise band is reported as inside the noise band. |

---

### Critical files for implementation

- `/Users/brianbruggeman/repos/slot-0/proxima/omega/src/msl.rs` (route decision at `:673-697`, `:751-772`, `:3140-3194`; scatter reject at `:932`; geometry consts at `:1017`, `:1030`, `:1046`; `u.out_base` emission at `:2735`)
- `/Users/brianbruggeman/repos/slot-0/proxima/omega/src/metal.rs` (`classify_kind` `:785-826`; `encode_op` `:2179-2252` with `allocate_buffer` `:2210` and `device_buffers.insert(node,(output,0))` `:2249`; `NOCOPY_BUFFERS` `:1848-1892`; `InputSizeMismatch` `:991-1000`)
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/generate.rs` (`resolve_plan` `:958-975`, key at `:966`, `plans.clear()` at `:973`; `LayerCache` `:621-654`; `token_breakdown_metal` `:1721-1753`)
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/shape.rs` (`project_output_shape` `:469-486` — the one match arm that blocks write placement; `bounds_check` `:440-467`; `scatter_output_shape` `:495-540`)
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/bind.rs` (`Layout` `:95-98`; `BoundOpKind` 4 variants `:221-264`; `bind_reduce` `:955-996`; `layout_of` `:1594-1605`)