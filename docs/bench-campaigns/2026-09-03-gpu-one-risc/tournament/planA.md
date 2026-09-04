I verified the ledger against the repo read-only (HEAD `4be2f3a`, 68 worktrees, every file:line below opened this session). No `proxima-wt-*` directory was entered.

---

# Plan A — GPU parity for proxima-tensor through omega, ONE RISC

## Diagnosis (one paragraph, ledger-cited)

The instruction set is already one RISC and is not the defect: `Op` has 5 variants (`proxima-tensor/src/op.rs:175-266`, doc header at `:166` stale — says "four generators"), `BoundOpKind` has 4 (`bind.rs:221-264`), and write placement already exists at the IR level as `Reduce.out_map` addressing the result (`op.rs:158-162`) and `Layout {base, strides}` on every bound Reduce (`bind.rs:95-98`) — R5. What is broken is three layers above and one layer below that: (a) the decode driver re-plans every token by construction, because the plan-cache key is `(new_count, cached_len)` (`generate.rs:966`) and `cached_len` is `Extent::Symbolic(1)` on every KV leaf (`spec.rs:6220,6230,6240`), so `plan_hits=0` is the shape symbol, not a cache bug (R11/M6'), and `ff749a0`'s `self.plans.clear()` (`generate.rs:973`) now makes every token pay `plan_named` + `mark_resident` + fresh output/uniform buffers per op (`metal.rs:2210-2211`, 1196 `newBufferWithLength` + 1196 uniform uploads/token = the 4.4 ms `op_setup`, R11/M6''); (b) the Q4_K matvec body does ~48 ops/8 weights against ggml's ~8 (R3/M3, incumbent form read at `ggml-metal.metal:5147-5175`, R8) and the cooperative reduce is pinned to `SIMD_WIDTH=32` (`omega/src/sized.rs:45`, `msl.rs:3195-3197`) where the incumbent's rms_norm reaches 1024 (R8); (c) the lowering is 5 hand-ordered `if let Some(..)` gates inside `push_cooperative_reduce_body` (`msl.rs:3162-3190`) whose decision is recovered by grepping emitted MSL substrings (`classify_kind`, `metal.rs:794-825`, whose own doc at `:777-783` admits the route "is not exposed as its own accessor"), with `PACKED_ROWS_PER_GROUP=4` (`msl.rs:1017`), `TILE_DIM=8` (`:1030`), `TILED_GEMM_NSG=4` (`:1046`) as bare source consts while only `[tiled_gemm]` lives in `omega/omega-runtime.toml` — a §12 violation. **The ledger contradicts the diagnosis's own mass ordering at the bench rung and the plan is reordered because of it**: R12 records feature-off 51.571 wall / 35.117 GPU ms at 1194 dispatches vs feature-on 51.535 wall / 39.841 GPU ms at 616 dispatches — halving the dispatch count moved wall by 0.07% and moved GPU time *up* 13.5%; and that feature-off control already carries the paired Q4_K body, i.e. main's ~56.5-57.0 gpu_exec (R1, MEMORY) → ~35.1 came from the kernel body alone. So the largest *measured* mover in the entire ledger is the Q4_K body, which exists twice, uncommitted, in two independent worktrees (R7 `metal-q4k-mask-fma` −17.2% gpu_exec / −36% on ffn; R12 `q4k_pair_dot` −29% GPU family, parity 3.1e-6); the second largest is CPU orchestration at 11.4 ms/token (R1) with a zero-IR-change fix; and the attention graph collapse — the diagnosis's headline — is measured-neutral on wall today and is therefore sequenced *after* the two, where its win becomes visible.

---

## One-RISC binding

### What changes in `proxima-tensor`

1. **`Reduce.out_map` gains a non-zero write offset.** The IR constraint is named verbatim at `spec.rs:2310-2313`: *"`Reduce::out_map` must stay a pure projection (`shape::project_output_shape`'s own doc), so nothing upstream of a reduce can splice two tensors into one axis."* That is the sole reason attention emits two online-softmax Reduce blocks instead of one. `IndexPattern` already carries `offset` on the **read** side (`map.rs:12`, "slice = non-zero offset"), and `Layout::base: i64` already exists on the **write** side of every bound Reduce (`bind.rs:95-98`, consumed by `Layout::offset_of` at `:100-108`). The change is: `project_output_shape` accepts a constant base term, `bind` folds it into `out_layout.base`, and `cpu::run_reduce` + `omega::msl::render_reduce` honour it. **Two producers writing disjoint ranges of one caller-owned buffer IS concat.** No `Concat`, no `Op::Pad`, no `Op::Tile`, no `PlacedBuffer` (all zero hits on main, R5).
2. **Write placement into a caller-owned persistent buffer** is a *driver* fact, not an IR fact: `AlignedBuffer` (`align.rs:42`, `new` at `:69`, **zero production callers**) becomes the KV allocation; the Metal driver aliases it once through `register_checkpoint_mapping`'s existing mechanism (`backend.rs:409`, `metal.rs:1768`), and per-token writes land at `out_layout.base = row_stride * cached_len`. This is exactly the incumbent's shape (`llama-kv-cache-unified.cpp:749-788`, R8: `ggml_cpy` into a `ggml_view_1d` at a byte offset in a once-allocated buffer).
3. **`cached_len` becomes a plan-stable capacity bucket, not a per-token symbol.** Bucket to a multiple of `KV_CAPACITY_BUCKET` (from `proxima-tensor-runtime.toml`) and mask the tail with the causal-mask machinery that already exists (`Iota` + `Greater` + `Select`, `op.rs:207-220`). The incumbent does the same (pads `n_kv` to 256 and masks, R11/M6'). This is **zero IR change** and makes the key `(new_count, bucket)` hit 255/256. The alternative — making the reduce extent a runtime uniform — requires `BoundOp::extents: Vec<u64>` (`bind.rs:210`) to stop being baked and is held as a **contingency only** (Step 7.3), not the primary.

### What changes in `omega`

4. **A first-class route enum decided before emission.** `enum KernelRoute { Elementwise, Scan, Iota, Constant, ReduceSerial, ReduceCooperative, ReduceRowBlockedPacked, ReduceTiledGemm }` returned by one `fn route_of(&BoundOp, &PackedOperands) -> KernelRoute` that `emit` (`msl.rs:673-697`) and `push_cooperative_reduce_body` (`msl.rs:3140-3194`) both consume, replacing the 5 hand-ordered `if let Some(..)` gates whose ordering is load-bearing (`msl.rs:745-754` comment says so). Censused `(NodeId, KernelRoute, reason)` on the **exact** shape already shipped for `WidthDeclineReason` (`instrument.rs:809-828`, `WIDTH_TILE_DECLINE: Mutex<BTreeMap<(u32, WidthDeclineReason), ..>>` at `:842`, `record_width_tile_decline` at `:848-864`). `classify_kind` (`metal.rs:785-826`) is then **deleted**, not fixed — it reads `route_of` instead of grepping `kernel.source`.
5. **One emitter core over the 4 `BoundOpKind`s, backend text in tables.** ~26 functions are reimplemented per backend across `msl.rs` 4712 / `wgsl.rs` 1929 / `cuda.rs` 1838 lines with exactly 3 shared types + 2 shared fns (`Binding`, `PackedCodec`, `PackedOperands`, `gather_count`, `gather_slots` — `wgsl.rs:105`, `cuda.rs:66`), R5. The core walks `(BoundOpKind, KernelRoute)` and asks a `BackendText` value (a **struct of `&'static str` fields + small fns**, not a trait object — §20 box-free, AGENTS.md `### source`) for intrinsic spellings, signature syntax, and address-space keywords.
6. **Every backend covers all 4 `BoundOpKind`s.** `cuda::emit_cuda` (`cuda.rs:146-183`) currently rejects `Iota` and `Constant` with `CudaUnsupportedOpKind`; WGSL covers all 4 but has no tiled-GEMM or packed-row-block route (R5). Coverage becomes a compile-time exhaustive match, not a runtime rejection.
7. **Every geometry const moves into `omega/omega-runtime.toml`.** `PACKED_ROWS_PER_GROUP` (`msl.rs:1017`), `TILE_DIM` (`:1030`), `TILED_GEMM_NSG` (`:1046`), the packed-row lanes/block (`msl.rs:2527`), and the cooperative-reduce width all become new sections beside `[tiled_gemm]`, emitted through the same `build.rs` → `OUT_DIR/omega_sized.rs` path the file's own header documents. `SIMD_WIDTH` (`sized.rs:45`) **stays** a source const — its doc at `sized.rs:9` classifies it as a hardware-family fact, not a policy knob, and that classification holds.

### What does NOT change

- **No new `Op` variant.** 5 stays 5.
- **No new `BoundOpKind`.** 4 stays 4. This is what rules out `BoundOpKind::CachedAttention` (R12) — a fifth bound kind matched by a post-bind structural matcher over one model's attention shape is the "arbitrary rule for a specific instance" AGENTS.md `### constraint tiers` / hard invariants forbids, and the branch's own `failure-cached-attention-matcher.md` already records the first matcher being abandoned as a heuristic.
- **No new trait.** `BackendText` is a struct of data + free fns; routing is `enum + match` (§20, §11).
- **No new module in `proxima-tensor`.** `physical.rs` (+576 on the parallel branch) does not land; its content is either `out_layout.base` (already exists) or a driver fact (belongs in `omega/src/metal.rs`).
- `plan`/`execute` stays not-a-pipe (adjudicated 2026-08-30, `backend.rs:1-52`).

---

## The bench ladder used by every step

| rung | unit of measure | harness |
|---|---|---|
| **nano** | counts, no device: ops/layer, `BoundOp` count, route census rows, emitted-MSL assertions | `cargo test -p proxima-tensor` / `-p omega` |
| **micro** | one kernel on device: ms/dispatch, GB/s, GMAC/s | `omega/examples/q4k_matvec_probe.rs`, `membw_probe.rs`, `omega/benches/metal_vs_cpu.rs` (`omega/Cargo.toml:207-210`) |
| **milli** | one decode step's stage split | `token_breakdown_metal` (`generate.rs:1721-1747`) + `token_breakdown` (`generate.rs:1655-1671`) |
| **bench** | 24-token interleaved paired cell vs llama.cpp-Metal | `bind.rs:3002` metal test + `llama-bench` |

**Rule:** a step measured at rung R pre-registers its prediction at rung R+1 **only**. A miss kills the climb and is decomposed into *inconsistency* (the two rungs disagree about the same quantity) vs *understanding-gap* (the mechanism is not what we named), each with its own work item.

**Canonical commands** (weld `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-<name> &&` and `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-<name>/target` into each):

```
# BENCH rung, ours
cargo test -p proxima-model-interop --release --features std,metal,instrument \
  runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache \
  -- --ignored --nocapture 2>&1 | tee /tmp/cell-<step>.txt

# BENCH rung, incumbent (home turf)
/Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench \
  -m ~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf \
  -p 0 -n 24 -ngl 99 -r 5

# gates
bash scripts/omega-gate.sh
bash scripts/proxima-tensor-gate.sh
```

`PROXIMA_MAX_TOKENS` (default 24, `bind.rs:2719`) sets token count. Arms are interleaved ours/theirs/ours/theirs, one measurer on the box, 5 runs, CoV reported; CoV > 5% means sweep, never a point estimate.

---

# Phase 0 — Seal, land, reconcile what exists

**Goal:** every number the rest of the plan cites is on main, re-provable, and numbered from ROW 233 (`proxima-tensor/docs/discipline.md:18736`). Row numbers are `ROW <PH0-k>` placeholders on branches; the landing step rewrites them.

### 0.1 — Fresh quiet-box seal of main
- worktree `proxima-wt-qbox` · branch `bench/quiet-seal-main` · `CARGO_TARGET_DIR=.../proxima-wt-qbox/target`
- open: `proxima-model-interop/src/bind.rs:3002-3040` (the metal decode test), `generate.rs:1721-1747` (the counter print)
- commands: quiet the box (`sudo pmset -a disablesleep 1` not required; close other agents, AC power, no other cargo build). Then 5 interleaved pairs of the two BENCH-rung commands above.
- expected N: 5 ours-runs × 24 `token_breakdown_metal` lines each = **120 lines** parsed; `plan_hits` field present on every line. **N==0 is RED.**
- prediction (nano → **micro**): `encode_dispatch_calls == 1196` per steady-state token and `plan_hits == 0` on every step, reproducing R1/R11 on today's main.
- kill: if `encode_dispatch_calls` differs from 1196 by >2%, the 9 commits since `2b95210` moved the graph and every R1/R2/R3 MEMORY row is void — stop and re-derive the gap decomposition before any other step.
- rollback: none (read-only measurement).
- blast radius: none.
- counter: `gpu_exec_ticks`, `op_setup_ticks`, `encode_dispatch_calls`, `plan_hits`, `plan_misses`, `block_upload_bytes`, `kv_cache_upload_bytes`.
- re-prove: the BENCH-rung command above.
- log row: `ROW 234 — quiet-box re-seal of the GPU decode cell on 4be2f3a: the board main actually stands on.`

### 0.2 — Export the uncommitted worktree diffs as patches without entering them
- worktree `proxima-wt-qbox` (same) · read-only
- commands: `git -C /Users/brianbruggeman/repos/slot-0/proxima --work-tree=<n/a>` is not usable; use `git diff` piped through the *branch pointer*, not the directory: `git log --format=%H -1 perf/gpu-all-wins`, then for the dirty content use `git --git-dir=$(git rev-parse --git-common-dir) diff` from each worktree's own shell. **Luna executes this inside each worktree; the plan author did not enter them.**
- expected N: **10 patch files** in `/tmp/gpu-wins/` matching R7's table (`gpu-all-wins`, `kv-device-resident`, `op-rule-census`, `q4k-split-k`, `output-placement`, `attention-single-range`, `gpu-dispatch-count`, `metal-simdgroup-geometry`, `q4k-orchestration`, `kernel-latency`). **N==0 is RED.**
- prediction (nano → **nano**, no rung climb): each patch applies to `2b95210` cleanly and to `4be2f3a` with conflicts in `spec.rs` and `metal.rs` (R7: main moved `spec.rs +8735/-2836` and `metal.rs` twice).
- kill: a patch that does not apply to `2b95210` means the worktree was mutated since the R7 audit; re-audit before landing anything from it.
- rollback: delete `/tmp/gpu-wins/`.
- blast radius: none.
- counter: patch count + `git apply --check` exit codes.
- re-prove: `for p in /tmp/gpu-wins/*.patch; do git apply --check --3way "$p"; echo "$p $?"; done`

### 0.3 — Land the wide cooperative reduce (independent of the Q4_K adjudication)
- worktree `proxima-wt-landwide` · branch `perf/metal-wide-cooperative-reduce` · own target dir
- open: `omega/src/msl.rs:3195-3197` (`output_index = gid / SIMD_WIDTH`, `lane = gid % SIMD_WIDTH`), `msl.rs:1555-1557` (`grid_threads`: `output_total * SIMD_WIDTH`), `omega/src/sized.rs:45`
- commands: rebase the `wide-cooperative-reduce` hunk of `perf/gpu-dispatch-count` onto `4be2f3a`; add feature `metal-wide-cooperative-reduce` to `omega/Cargo.toml [features]` (default-off, gate 1); `bash scripts/omega-gate.sh`.
- expected N: `cargo nextest run -p omega --features metal,metal-wide-cooperative-reduce` reports **the same test count as `--all-features` on main, plus ≥3 new** (narrow-row, wide-row, non-power-of-two-row). **N==0 is RED.**
- prediction (micro → **milli**): reduce-family GPU time falls ≥15% (R3/M4 measured −20%, −2.8 ms); whole-token `gpu_exec_ms` falls 4-6% (R7 measured −4.9%).
- kill: `gpu_exec_ms` delta < +0% or any parity test regresses → roll back, record the negative.
- rollback: `git reset --hard` the branch; the feature is default-off so main is unaffected either way.
- blast radius: `omega/src/msl.rs` cooperative path only; `render_reduce`'s serial path, tiled-GEMM and packed-row-block untouched (they return before the generic fold, `msl.rs:3162-3190`).
- counter: `gpu_exec_ticks` + the route census once 3.1 lands; until then, `classify_kind`'s `reduce-cooperative` bucket **by family, never by bucket count** (M10: it relabels when a body changes).
- re-prove: `cargo test -p proxima-model-interop --release --features std,metal,instrument,omega/metal-wide-cooperative-reduce runs_the_cached_decode_loop... -- --ignored --nocapture`
- log row: `ROW 235 — wide cooperative reduce: SIMD_WIDTH is a lane count, not a thread budget.`

### 0.4 — Head-to-head adjudication of the two independently-derived Q4_K bodies
- worktree `proxima-wt-onebody` · branch `perf/q4k-body-adjudication` · own target dir
- open: `omega/src/msl.rs:2452-2530` (`push_packed_row_blocked_body`, `Q4K_BLOCK_ELEMENTS / SIMD_WIDTH`, `lanes_per_block` at `:2527`); incumbent form at `/Users/brianbruggeman/repos/others/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5147-5175`
- commands: apply BOTH bodies to one tree behind two default-off features (`metal-q4k-mask-fma` from 0.2's patch; `metal-q4k-pair-dot` cherry-picked from `perf/cached-attention-streaming`'s ROW 257 commit). Run three arms interleaved: off / mask-fma / pair-dot, 5 runs each.
- expected N: parity assertion on the real `blk.0.attn_q.weight` for **both** bodies, max abs error vs f32 ≤ 1e-5 (R12 reports 3.1e-6 for pair-dot). **N==0 is RED** — a filtered run matching zero parity tests exits 0.
- prediction (micro → **milli**): both bodies land the Q4_K family in the same band (mask-fma −36% on ffn / −17.2% whole-token gpu_exec, R7; pair-dot −29% GPU family, R12); the spread between them is **< 5%**, i.e. within CoV, and the choice is therefore made on §14 + surface, not on ms.
- decision rule, pre-registered (not a verdict — a rule executed on numbers): (1) any body failing parity is out; (2) if the milli-rung spread exceeds 2× the pooled CoV, the faster body wins; (3) otherwise the body textually closer to `ggml-metal.metal:5147-5175` wins (§14 — the incumbent's formulation is the oracle), and (4) the loser's feature flag is **deleted, not left dormant** (§15 no punt).
- kill: if both bodies underperform main's current body at the milli rung, the R7 and R12 numbers were measured on a different tree than main and the whole Q4_K premise is re-opened.
- rollback: both features default-off; drop the branch.
- blast radius: `omega/src/msl.rs` packed-row-block body; `kernel_cache_key` (`msl.rs:745-760`) gains no new key char if the body is behind a feature.
- counter: `gpu_exec_ticks` split by route once 3.1 lands; interim `token_breakdown_metal gpu_exec_ms` + the `q4k_matvec_probe` micro arm.
- re-prove: `cargo run --release -p omega --features metal --example q4k_matvec_probe` interleaved across the three feature sets.
- log row: `ROW 236 — two independent re-derivations of ggml's Q4_K body, one landed: the adjudication and the deleted loser.`

### 0.5 — Land the chosen Q4_K body
- worktree `proxima-wt-onebody` (same) · branch `perf/q4k-body-land`
- expected N: `bash scripts/omega-gate.sh` green with its asserted test count; `bash scripts/proxima-tensor-gate.sh` green.
- prediction (milli → **bench**): whole-cell `step_wall_ms` falls from 0.1's sealed number by 15-25%; the ratio vs llama.cpp-Metal moves from ~3.9x toward **~2.9-3.1x** (R12 measured 2.95x with this body plus their consumer index).
- kill: bench-rung move < half the milli-rung prediction → decompose (inconsistency: GPU time fell but wall did not, meaning CPU orchestration is now the binding term — which is exactly Phase 2's premise and would *promote* Phase 2, not kill it).
- rollback: revert one commit; feature default-off.
- blast radius: `omega/src/msl.rs`, `omega/Cargo.toml`.
- counter: `gpu_exec_ticks`, `encode_dispatch_calls` (must be **unchanged at 1196** — a body change that moves the dispatch count means the route changed too and the arms are not comparable).
- re-prove: BENCH-rung command.

### 0.6 — Land `prune_dead` / `dead_resolved_nodes` from the parallel branch
- worktree `proxima-wt-prune` · branch `perf/bind-prune-dead` · own target dir
- open: `proxima-tensor/src/bind.rs:200-215` (`BoundOp`), the `216d925` commit on `perf/cached-attention-streaming`
- rationale: R12 marks this generic and RISC-conformant; it is the one piece of that 42-commit branch that lands unmodified.
- expected N: ≥2 new tests (a program with a dead resolved node; a program with none) plus the unchanged `proxima-tensor` gate count. **N==0 is RED.**
- prediction (nano → **micro**): `encode_dispatch_calls` falls by the dead-node count, which is **0 on the openchat decode graph** unless the census says otherwise — this is a correctness/generality landing, not a perf landing, and the row says so.
- kill: any change to `encode_dispatch_calls` on the decode cell that the census cannot name node-by-node.
- rollback: revert one commit.
- blast radius: `proxima-tensor/src/bind.rs` only; CPU and every backend consume the same shortened `&[BoundOp]`.
- counter: `encode_dispatch_calls`, plus a nano assertion on `BoundOp` count for a fixture program.
- re-prove: `cargo nextest run -p proxima-tensor prune_dead`
- log row: `ROW 237 — dead resolved nodes never reach a backend.`

### 0.7 — Adjudicate `BoundOpKind::CachedAttention` against the one-RISC binding
- worktree `proxima-wt-adjcat` · branch `docs/cached-attention-adjudication` · own target dir
- open: the branch's `render_cached_attention` (`msl.rs:104` on that branch), `cpu.rs:19141-19190` on that branch, their `failure-cached-attention-matcher.md`, and against them: `bind.rs:221-264` (4 kinds) + AGENTS.md `### constraint tiers` hard invariant "do not add arbitrary rules for specific instances".
- commands: read the diff hunks (`git show`), not the commit list (AGENTS.md `### the index is not the thing`). Extract their ROW 234-267 verbatim into `/tmp/parallel-rows.md`.
- expected N: **34 rows** extracted (234-267), each tagged keep / re-number / supersede. **N==0 is RED.**
- prediction (nano → **nano**): the binding above admits **0** of the 42 commits as a fifth `BoundOpKind`; it admits `prune_dead` (0.6), the Q4_K body (0.4/0.5 if it wins), the classifier defect finding (ROW 263, superseded by Step 3.1), and every measured negative (ROWs 249, 251-254, 259-260, 265-267) as log rows.
- what happens to their win: the macro-op's dispatch reduction (1194 → 616) is **re-expressed** through Step 4.2's write placement, where the same collapse comes from `out_layout.base` and costs no new kind. Their measurement (`51.535` wall / `39.841` GPU at 616 dispatches) is the pre-registered *control* for Step 4.3: it says a dispatch collapse that raises GPU ms is not a win, and Step 4.3's kill criterion is written against it.
- kill: none — this step produces rows, not code.
- rollback: n/a.
- blast radius: `proxima-tensor/docs/discipline.md` only.
- counter: row count landed; `git log --oneline main..perf/cached-attention-streaming | wc -l == 42` accounted for member-by-member (AGENTS.md `scope is literal`).
- re-prove: `grep -c '^## ROW' proxima-tensor/docs/discipline.md`
- log rows: `ROW 238 — the fifth bound kind the binding does not admit, and what its measurement is kept for.` `ROW 239-247 — the parallel lane's measured negatives, renumbered onto main (nsg=2 fourth negative, float4 accumulation, explicit unroll, width-64 dispatch, row-batch 8, tiled-gemm at decode).`

### 0.8 — Land the seal harness
- worktree `proxima-wt-qbox` · branch `bench/sealed-pass`
- commands: rebase `bench/sealed-pass`'s 2 commits (R0: `scripts/sealed-pass.sh` + quiet-gate fix; confirmed absent from main — `ls scripts/sealed-pass.sh` → No such file) onto `4be2f3a`.
- expected N: `bash scripts/sealed-pass.sh --dry-run` enumerates ≥2 arms (ours, llama.cpp) and exits 0. **N==0 is RED.**
- prediction (nano → **micro**): the script reproduces 0.1's numbers within CoV on a re-run.
- kill: script's own numbers disagree with 0.1 by >CoV → the harness is measuring something else; fix before any later step uses it.
- rollback: revert 2 commits.
- blast radius: `scripts/` only.
- counter: arm count, CoV per arm.
- re-prove: `bash scripts/sealed-pass.sh`
- log row: `ROW 248 — the seal is a script, not a memory (§16).`

---

# Phase 1 — Measurement substrate (zero mass, zero risk, unblocks every scoreboard)

Runs **in parallel** with Phase 2. Nothing here touches a hot path.

### 1.1 — GPU roofline debt: a correct streaming-copy bandwidth probe
- worktree `proxima-wt-bwprobe` · branch `bench/gpu-streaming-bandwidth` · own target dir
- open: `omega/examples/membw_probe.rs:139-146` — the Metal arm builds `Op::Reduce(Reduce { body: ScalarOp::Add, init: ReduceInit::Zero, keep: Keep::Reduce, .. })`, i.e. a **reduce-to-scalar**, which is why `rooflines.md:411` records 0.3-0.4 GB/s and tags it **DEBT — not measured**.
- design: add a streaming-copy arm — `Op::Elementwise { body: ScalarOp::Identity, operands: [(src, affine identity map)] }` over N f32, output a full-size buffer, sweep N ∈ {16 MiB, 64 MiB, 256 MiB, 1 GiB}. **Readback is outside the timed window**: time only `gpu_exec` (`metal.rs:547-554`), never `readback` (`metal.rs:2358-2378`); the row states which counter bounded the window.
- expected N: **4 sweep points × 5 runs = 20 samples**, each with GB/s and CoV. **N==0 is RED.**
- prediction (micro → **micro**, sweep is the rung): the largest N reaches ≥ 200 GB/s on M1 Max, i.e. at or above the incumbent's *achieved* 228.9-234.1 GB/s (R1, `rooflines.md` GPU table) — because the incumbent's achieved figure is a lower bound on the machine's streaming ceiling and a copy must beat a Q4_K matvec.
- kill: if the copy probe reads **below** llama.cpp's achieved 228.9 GB/s, the probe is still not measuring a ceiling; do not substitute a spec-sheet figure (that is what `rooflines.md:411` explicitly refused). Escalate to a 2-buffer ping-pong with `MTLBlitCommandEncoder` before writing a ceiling.
- rollback: example-only; delete the arm.
- blast radius: `omega/examples/membw_probe.rs` + `proxima-tensor/docs/rooflines.md:396-479`.
- counter: `gpu_exec_ticks`, `block_upload_bytes`, `readback_bytes` (readback_bytes must be **0** inside the timed window — that is the assertion that proves the window is clean).
- re-prove: `cargo run --release -p omega --features metal --example membw_probe -- --arm streaming-copy`
- log row: `ROW 249 — the GPU streaming ceiling exists: the roofline debt at rooflines.md:411 is paid.` Also rewrite `rooflines.md:411` from DEBT to a measured ceiling with its CoV, and update the summary row at `rooflines.md:751`.

### 1.2 — torch-MPS incumbent cell (mnist / train / BGE lanes)
- worktree `proxima-wt-mps` · branch `bench/torch-mps-arms` · own target dir
- open: `proxima-onnx/scripts/torch_reference/inference_bench.py` (no `--device` flag today; `parse_args` has only `--threads` and `--runs`), `train_bench.py`, `model.py`, `data.py`
- commands: add `--device {cpu,mps}`; `torch.device("mps")`, `torch.mps.synchronize()` around the timed window (the MPS analogue of the readback rule in 1.1), warmup ≥50 images as the file already does (`WARMUP_IMAGES = 50`). Use the existing venv: `proxima-onnx/scripts/torch_reference/venv/bin/python` (torch 2.13.0 present, R0).
- expected N: 3 lanes × 2 devices × 5 runs; mnist inference reports **p50/p95/p99/mean/CoV over 200 runs** (the file's own `--runs` default). **N==0 is RED** — `torch.backends.mps.is_available()` must be asserted True and printed, or the "mps" arm silently ran on CPU.
- prediction (micro → **micro**): mnist batch=1 MPS is **slower** than CPU (dispatch latency dominates a 4-layer MNIST at batch 1) and the MLP train step at batch ≥64 is faster on MPS. Both directions are the result; the loss is reported first (§19).
- kill: `is_available()` False → install path is wrong, not a hardware fact; fix the venv, do not report a CPU number as an MPS number.
- rollback: python-only; revert the flag.
- blast radius: `proxima-onnx/scripts/torch_reference/*.py`; zero Rust.
- counter: wall-clock p50/p95/p99 + CoV per arm; `torch.mps.current_allocated_memory()` for the memory column.
- re-prove: `proxima-onnx/scripts/torch_reference/venv/bin/python proxima-onnx/scripts/torch_reference/inference_bench.py --device mps --runs 200`
- log row: `ROW 250 — the torch-MPS cell that did not exist: mnist, MLP train step, and what batch=1 costs on a GPU.`

### 1.3 — ORT-CoreML incumbent cell
- worktree `proxima-wt-coreml` · branch `bench/ort-coreml-arms` · own target dir
- open: `scripts/onnx_reference/bench.py:96` — `providers=["CPUExecutionProvider"]` is hardcoded; `scripts/onnx_reference/run.sh` (pinned venv at `$HERE/.venv`, `ONNX_REF_PYTHON=python3.12`)
- commands: `scripts/onnx_reference/.venv/bin/pip install onnxruntime` (the macOS wheel ships `CoreMLExecutionProvider`). If no network: build from the existing source checkout at `~/repos/others/onnxruntime` with `./build.sh --config Release --use_coreml --build_wheel --parallel`. **Neither path is a blocker** (AGENTS.md `### stopping`); pick pip first and fall through.
- then: add `--provider {cpu,coreml}` to `bench.py`, threading it into the `InferenceSession` call at `:96`; assert the resolved provider with `session.get_providers()` and print it.
- expected N: BGE-small, 2 providers × 5 runs, with the existing `cosine_similar` / `cosine_dissimilar_a/b` fidelity fields (`bench.py:82-85`) reported per provider. **N==0 is RED**; additionally `get_providers()[0] == "CoreMLExecutionProvider"` or the arm did not run on CoreML.
- prediction (micro → **micro**): CoreML EP falls back to CPU for a subset of BGE's ops and reports a **mixed** partition; the ms/sentence lands within 2x of the CPU EP, and the fidelity fields are unchanged (fp32 path).
- kill: fidelity drift (cosine similar/dissimilar move) means CoreML chose fp16 — that is a different arm and must be labelled as one (§14: differing output is our problem to explain, not theirs).
- rollback: revert the flag; the venv is gitignored (`scripts/onnx_reference/.gitignore`).
- blast radius: `scripts/onnx_reference/bench.py`, `run.sh`; zero Rust.
- counter: ms/sentence, CoV, provider partition node counts from `sess_options.log_severity_level=0`.
- re-prove: `BGE_MODEL_PATH=... bash scripts/onnx_reference/run.sh --provider coreml`
- log row: `ROW 251 — the ORT-CoreML cell that did not exist, and the partition that explains it.`

### 1.4 — llama.cpp `-fa 1` as a second incumbent arm
- worktree `proxima-wt-fa1` · branch `bench/llama-flash-attention-arm` · own target dir
- rationale: flash attention is **OFF by default** at checkout `b25346221` (`common/common.h:328`, R8), so every existing scoreboard row compares against the non-FA incumbent. `-fa 1` changes the attention path *and* the KV layout (`v_trans = !flash_attn`, R8) — a second incumbent arm, not a variant of the first.
- commands: `llama-bench -m <openchat Q4_K_S> -p 0 -n 24 -ngl 99 -fa 1 -r 5`, interleaved with the `-fa 0` arm.
- expected N: **2 arms × 5 runs = 10 samples**, each with ms/token and CoV. **N==0 is RED.**
- prediction (bench → **bench**, incumbent-only ladder): `-fa 1` is within 5% of `-fa 0` at batch 1 decode (flash attention's win is a prefill/long-context effect; at n_kv≈24 the KV read is trivial) — so the scoreboard's denominator does not move and every existing ratio stands.
- kill: if `-fa 1` is >10% faster, every ratio in the log is against the weaker incumbent arm and must be restated; that restatement is the row.
- rollback: n/a, measurement only.
- blast radius: `proxima-tensor/docs/discipline.md` + `rooflines.md` denominators.
- counter: ms/token per arm; `design-favors: incumbent` on both.
- re-prove: the two `llama-bench` invocations.
- log row: `ROW 252 — the second incumbent arm: flash attention is off by default at b25346221 and here is what turning it on costs.`

### 1.5 — GPU arms for the mnist / train / BGE lanes on our side
- worktree `proxima-wt-lanes` · branch `bench/omega-nondecode-arms` · own target dir
- open: `omega/benches/metal_vs_cpu.rs` (the ONLY GPU bench outside decode; registered `omega/Cargo.toml:207-210`; its doc says UNRUN — R9), `omega/tests/training_step_parity.rs:400-607` (GPU train step exists as **untimed** parity tests only)
- commands: run `metal_vs_cpu` for the first time (`cargo bench -p omega --features metal --bench metal_vs_cpu`); add a timed arm around the existing training-step parity fixture rather than writing a new one (§1 reuse-first — the fixture is the workload, it just has no timer).
- expected N: `metal_vs_cpu` reports **4 arms** (gemm_square 512/1024/2048, matvec_batch1) × 5 runs; the train-step arm reports ≥1 timed cell. **N==0 is RED** — a bench that has never run may not compile under the feature set.
- prediction (nano → **micro**): `matvec_batch1_f32` at Mistral shapes lands under 25% of 1.1's streaming ceiling — the same low-simdgroup starvation R3/M5 names (52 GB/s at 256 simdgroups rising to 147 GB/s at 8001).
- kill: the bench does not compile → that is the finding, and fixing it is the step (AGENTS.md: a repo that does not build its own benches is our bug).
- rollback: revert the timed arm.
- blast radius: `omega/benches/metal_vs_cpu.rs`, `omega/tests/training_step_parity.rs`.
- counter: GB/s and GMAC/s per arm against 1.1's ceiling.
- re-prove: `cargo bench -p omega --features metal --bench metal_vs_cpu`
- log row: `ROW 253 — the non-decode GPU lanes get their first numbers; matvec at batch 1 against the measured ceiling.`

### 1.6 — ai_docs JSONL records for the GPU lane
- worktree `proxima-wt-aidocs` · branch `docs/ai-docs-gpu-lane` · own target dir
- open: `ai_docs/AGENT.md` (Update Rule: kind=5 decisions, kind=7 failures, `relations.idx=7` for grounding), `ai_docs/index.jsonl` (record shape), `ai_docs/task-routes.jsonl` (task/purpose/must_read/then_read_if_relevant/queries/done_when), `ai_docs/invariants.jsonl` (id/kind/summary/rule/applies_to/evidence_required)
- rationale: R0 — `task-routes.jsonl` has **0** tensor/omega/GPU records; per `ai_docs/AGENT.md` we add records, never bypass.
- records to add (exact ids):
  - `index.jsonl`: `proxima.gpu.decode_lane` (kind=3, path `proxima-tensor/docs/discipline.md`, source_paths the 8 Metal phase sites at `metal.rs:405,457,547,2187,2201,2209,2230,2358`), `proxima.gpu.rooflines` (kind=3, path `proxima-tensor/docs/rooflines.md`).
  - `task-routes.jsonl`: `{"task":"gpu-decode","must_read":["ai_docs/AGENT.md","proxima-tensor/docs/rooflines.md"],"then_read_if_relevant":["proxima-tensor/docs/discipline.md"],"done_when":["every route decision is a KernelRoute census row, not a source grep","every geometry const traces to omega/omega-runtime.toml","the cell carries a llama.cpp-Metal arm and a CoV"]}`; `{"task":"omega-emitter", ...}`.
  - `invariants.jsonl`: `proxima.invariant.one_risc_four_bound_kinds` (kind=5, rule: "BoundOpKind has exactly 4 variants; a model-specific macro-op is not a fifth", evidence_required: `bind.rs:221-264`); `proxima.invariant.route_is_a_value` (kind=5, rule: "a kernel route is an enum decided before emission and censused (NodeId, route); never recovered by grepping emitted source", grounded_in `metal.rs:777-783`); `proxima.invariant.gpu_geometry_from_config` (kind=5, §12, grounded_in `omega/omega-runtime.toml`); `proxima.failure.nsg2_regrouping` (kind=7, four measured negatives across two lanes — R4 + R12 ROW 267); `proxima.failure.dispatch_count_is_not_the_mass` (kind=7, grounded_in R12's 616-vs-1194 cell).
- expected N: **2 index + 2 task-route + 5 invariant = 9 records**; `jq -c . ai_docs/*.jsonl` exits 0 on all three files; `ai_docs/query.sh gpu-decode` returns ≥1 row. **N==0 is RED.**
- prediction (nano → **nano**): `grep -ci "omega\|metal\|gpu" ai_docs/task-routes.jsonl` moves from **0** (verified this session) to ≥2.
- kill: malformed JSONL (jq non-zero) → fix before landing; a broken index is worse than none.
- rollback: revert one commit.
- blast radius: `ai_docs/` only.
- counter: record counts per file; `ai_docs/query.sh` hit count.
- re-prove: `jq -c 'select(.task=="gpu-decode")' ai_docs/task-routes.jsonl && jq -c 'select(any(.applies_to[]?; .=="gpu-decode"))' ai_docs/invariants.jsonl`
- log row: `ROW 254 — the GPU lane enters ai_docs.`

---

# Phase 2 — Plan stability: kill the per-token re-plan

Highest mass-per-risk in the ledger: 11.4 ms/token CPU orchestration (R1), of which `op_setup` 4.394 + `prepare` 2.087 are directly addressed, with **zero IR change**.

### 2.1 — `cached_len` as a plan-stable capacity bucket
- worktree `proxima-wt-bucket` · branch `perf/kv-capacity-bucket` · own target dir
- open: `proxima-model-interop/src/generate.rs:966` (`let shape = (symbols[0] as usize, symbols[1] as usize)`), `:973` (`self.plans.clear()`), `proxima-tensor/src/spec.rs:6216-6245` (the three `Extent::Symbolic(1)` KV leaves at `:6220`, `:6230`, `:6240`), `spec.rs:2314-2319` (the mask asymmetry doc: "the cached block never needs masking at all" — **this is the sentence the bucket changes**, because a padded cache does need tail masking)
- design: `cached_len_bucket = round_up(cached_len, KV_CAPACITY_BUCKET)`; the KV `Op::Input` leaves are sized by the bucket; the tail `[cached_len, bucket)` is masked with the causal-mask composition already in the graph (`causal_mask`, `Iota`+`Greater`+`Select`, `op.rs:207-220`). `KV_CAPACITY_BUCKET` is a **new key in `proxima-tensor/proxima-tensor-runtime.toml`** under a `[kv_cache]` section (§12; the file already has parallel/cohort/quantize/transpose/neon/rope/staged_batch sections, R5) — default 256, matching the incumbent (R11/M6').
- expected N: parity — the CPU decode test `bind.rs:2818` must still assert `generated.0[0] == 2651` / `"known"` (`bind.rs:2797-2804`, llama.cpp's captured greedy answer, §14). Plus ≥3 new tests: bucket boundary (cached_len == bucket), bucket interior, bucket+1. **N==0 is RED.**
- prediction (nano → **milli**): `plan_hits` goes from 0 to **23 of 24 steps** (one miss per bucket crossing; at 24 tokens and bucket 256 there is exactly one miss, the first); `prepare_ms` per token falls from ~2.087 toward ~0.09 (2.087/23).
- kill: greedy token drifts off 2651/"known" → roll back immediately; a masking bug is a correctness defect and §14 says the incumbent is right.
- rollback: revert; the bucket is behind feature `kv-capacity-bucket` (default-off) so main's path is bit-identical when off.
- blast radius: `proxima-tensor/src/spec.rs` KV leaf extents + mask construction; `proxima-model-interop/src/generate.rs` plan key + `LayerCache` sizing (`:621-654`); **CPU and Metal both**, because the graph changes — so the CPU decode test at `bind.rs:2818` is the parity gate, run first.
- counter: `plan_hits`, `plan_misses`, `plan_cache_len`, `prepare_ms` (all in `token_breakdown_metal`, `generate.rs:1721-1747`).
- re-prove: `cargo test -p proxima-model-interop --release --features std,metal,instrument,kv-capacity-bucket runs_the_cached_decode_loop... -- --ignored --nocapture | grep -o 'plan_hits=[0-9]*'`
- log row: `ROW 255 — plan_hits=0 was the shape symbol, not the cache: cached_len becomes a capacity bucket and the tail gets masked.`

### 2.2 — Preallocate output and uniform buffers once per plan-stable program
- worktree `proxima-wt-pool` · branch `perf/metal-buffer-pool` · own target dir · **depends on 2.1**
- open: `omega/src/metal.rs:2210` (`allocate_buffer(device, bound_output_len(bound), bound.dtype)` — per op per token), `:2211` (`upload_uniforms(device, &pack_uniforms(bound))` — per op per token), `:2249` (`device_buffers.insert(bound.node, (output, 0))`), and R7's `metal-buffer-pool` patch from 0.2
- design: once a `Plan` is stable (2.1), every `BoundOp`'s output extent and uniform block are fixed. Allocate both **at plan time**, store them on the `Plan`, and have `encode_op` bind rather than allocate. Buffers are caller-owned and fixed-capacity (minimal-runtime tier discipline, guiding-principles "Minimal-runtime tier"), not a growable pool — strict O(1) per op (gate 11).
- expected N: `bash scripts/omega-gate.sh` count unchanged + ≥3 new (reuse across steps, extent change forces realloc, uniform contents change per step but the buffer does not). **N==0 is RED.**
- prediction (milli → **bench**): `op_setup_ms` falls from ~4.4 to **< 0.5**; `device_allocated_bytes` becomes flat across steps instead of churning; whole-cell `step_wall_ms` falls by ≥4 ms/token from 0.5's post-Q4K number.
- kill: `op_setup_ms` falls but `step_wall_ms` does not → decompose as *inconsistency* (a stage counter moved and the wall did not: either another stage absorbed it or the counters do not sum) and open a work item to re-check the residual, exactly as `generate.rs:1671`'s residual field already does for the CPU split.
- rollback: revert; feature `metal-buffer-pool` default-off.
- blast radius: `omega/src/metal.rs` `encode_op` / `execute_plan` / `Plan`; no `proxima-tensor` change; no other backend.
- counter: `op_setup_calls`, `op_setup_ms`, `device_allocated_bytes`, `phys_footprint_bytes` (all in `token_breakdown_metal`).
- re-prove: BENCH-rung command with `--features ...,omega/metal-buffer-pool`, grep `op_setup_ms`.
- log row: `ROW 256 — 1196 buffer allocations and 1196 uniform uploads per token become zero.`

### 2.3 — Re-seal the cell and restate the scoreboard
- worktree `proxima-wt-qbox` · branch `bench/quiet-seal-phase2`
- commands: `bash scripts/sealed-pass.sh` (landed 0.8), 5 interleaved pairs, plus the `-fa 1` arm from 1.4.
- expected N: 5 × 24 = 120 `token_breakdown_metal` lines; all three incumbent arms present (llama.cpp `-fa 0`, `-fa 1`, and torch-MPS where the lane applies). **N==0 is RED.**
- prediction (bench → **bench**, no rung above): ratio vs llama.cpp-Metal in **2.2-2.6x**, down from 0.1's sealed ~3.9x.
- kill: ratio outside 2.0-3.0x → the sum of 0.5 + 0.3 + 2.1 + 2.2 does not compose; bisect by toggling one feature at a time (all four are default-off gates, which is what makes this bisect mechanical — gate 1).
- rollback: n/a.
- blast radius: docs.
- counter: `step_wall_ms`, `gpu_exec_ms`, `op_setup_ms`, `prepare_ms`, `encode_dispatch_calls`, `plan_hits`, CoV per arm, plus fraction-of-ceiling against 1.1.
- re-prove: `bash scripts/sealed-pass.sh`
- log row: `ROW 257 — the board after the body and the plan: what is left is the graph and the emitter.`

---

# Phase 3 — First-class route and its census

Zero mass removed. It is sequenced here because **every Phase 4/5 claim is unprovable without it** (M10: `classify_kind` buckets by kernel source text and silently relabels when a body changes — which Steps 0.4/0.5 just did).

### 3.1 — `KernelRoute` decided before emission
- worktree `proxima-wt-route` · branch `refactor/omega-kernel-route` · own target dir
- open: `omega/src/msl.rs:673-697` (`emit`'s match), `:745-760` (`kernel_cache_key` and its "checked FIRST" ordering comment), `:824-838` (`reduce_is_cooperative`), `:3162-3190` (the two `if let Some(..)` gates: `tiled_gemm_block` then `packed_row_block`), `omega/src/metal.rs:777-826` (`classify_kind` and its confession), `:835-854` (`diagnose_kind`)
- design: `pub enum KernelRoute` (8 variants, listed in the binding above) + `pub fn route_of(&BoundOp, &PackedOperands) -> (KernelRoute, RouteReason)`. `emit`, `kernel_cache_key`, `grid_threads` (`msl.rs:1513-1557` — the three places that today re-derive the same decision) all call `route_of`. `classify_kind` is **deleted**; `diagnose_kind` becomes `route_of`'s reason.
- expected N: an exhaustive-match test asserting all 8 routes are reachable + one test per route on a real fixture shape (8 tests), plus the unchanged omega gate count. **N==0 is RED.**
- prediction (nano → **nano**): `route_of`'s census over the openchat decode plan reproduces `classify_kind`'s current bucket counts **exactly** on the pre-0.5 body, and **disagrees** on the post-0.5 body — the disagreement is the proof that the substring instrument was lying (R12 ROW 263 measured exactly this: 9/601 → 225/385 after their fix).
- kill: `route_of` and `classify_kind` disagree on the *pre-0.5* body → `route_of` has a bug; fix before deleting anything.
- rollback: revert; `route_of` is not feature-gated (it is a pure refactor with no behaviour change) so the rollback is a single revert with no flag surface.
- blast radius: `omega/src/msl.rs`, `omega/src/metal.rs` instrument block. `wgsl.rs`/`cuda.rs` untouched in this step (they get `route_of` in Phase 5).
- counter: **new** — `omega.route.decisions` keyed `(NodeId, KernelRoute)` mirroring `WIDTH_TILE_DECLINE` at `instrument.rs:842`, with `record_route(node, route, reason)` mirroring `record_width_tile_decline` at `:848-864`, gated behind `feature = "instrument"` at the module and every call site (`lib.rs:213-214` pattern).
- re-prove: BENCH-rung command, then `grep 'route=' /tmp/cell-3.1.txt | sort | uniq -c` → a per-route dispatch census summing to `encode_dispatch_calls`.
- log row: `ROW 258 — the route becomes a value: 8 kernel shapes, one enum, a (NodeId, route) census, and the substring instrument deleted.`

### 3.2 — Route census as a gate assertion
- worktree `proxima-wt-route` (same) · branch `test/route-census-gate`
- commands: add to `scripts/omega-gate.sh` a step asserting the census sums to the dispatch count.
- expected N: census rows sum **== `encode_dispatch_calls`** exactly. **N==0 is RED**, and a mismatch is RED too (an op that dispatched with no route row means a path bypassed `route_of`).
- prediction (nano → **milli**): on the openchat decode plan the census reports `ReduceRowBlockedPacked` ≈ 226 (the Q4_K/Q5_K/Q6_K ops R12 ROW 264 counts as 217+8+1) and `ReduceCooperative` + `Elementwise` covering the remaining ~970.
- kill: any op landing on `ReduceSerial` that R12 ROW 264 attributes to the Q4_K family — a quantized matvec on the serial path is a route bug worth more than any kernel tweak.
- rollback: revert the gate step.
- blast radius: `scripts/omega-gate.sh`.
- counter: `omega.route.decisions` vs `encode_dispatch_calls`.
- re-prove: `bash scripts/omega-gate.sh`
- log row: `ROW 259 — the census is a gate: an op with no route row is RED.`

---

# Phase 4 — Write placement: persistent KV and the minimal attention graph

Sequenced **after** Phases 2-3 because R12 proves the dispatch collapse alone is wall-neutral at today's kernel speed (51.535 vs 51.571 wall, 616 vs 1194 dispatches), and because Phase 3's census is what will attribute the change.

### 4.1 — Allow a constant write offset on `Reduce.out_map`
- worktree `proxima-wt-place2` · branch `feat/tensor-write-offset` · own target dir
- open: `proxima-tensor/src/op.rs:158-162` (`out_map` doc: "Addresses the result. Data-dependent here is what makes a scatter"), `proxima-tensor/src/map.rs:8-15` (the pattern table: transpose/broadcast/slice/stride/conv/gather; `:12` slice = non-zero offset **on the read side**), `map.rs:134-152` (`IndexMap::Affine(IndexPattern) | Computed{..}`), `proxima-tensor/src/bind.rs:95-108` (`Layout {base, strides}` + `offset_of`), `bind.rs:243-256` (`out_layout` / `out_scatter` on `BoundOpKind::Reduce`), `proxima-tensor/src/spec.rs:2310-2313` (the doc naming the constraint)
- design: `project_output_shape` accepts a constant base term in the projection; `bind` folds it into `out_layout.base` (the field already exists and is already `i64`); `cpu::run_reduce` adds it (it already calls `Layout::offset_of`); `msl::render_reduce` (`msl.rs:2178-2257`) adds it to the output index expression. **`Elementwise` gets no `out_map`** — it does not need one, because a placement is definitionally a reduce's write and an elementwise that needs placement is a `Keep::Scan` reduce with an identity body.
- expected N: ≥6 tests — zero offset (bit-identical to today), non-zero offset, two producers writing disjoint ranges of one buffer, overlapping ranges rejected at bind, offset + `out_scatter` together, offset on CPU and Metal agreeing bit-exactly. **N==0 is RED.**
- prediction (nano → **nano**): the two-producer-disjoint-range test produces the same buffer contents as a concatenation of two separately-evaluated reduces — this is the **worked example that IS the spec** for the whole phase.
- kill: overlapping ranges are not rejected at bind → a silent data race on GPU; do not proceed.
- rollback: revert; behind feature `tensor-write-offset` (default-off) in `proxima-tensor/Cargo.toml`.
- blast radius: `proxima-tensor/src/{op.rs,map.rs,shape.rs,bind.rs,cpu.rs}` + `omega/src/msl.rs` render path. Every backend, because it is IR — which is why every backend's parity test is the gate.
- counter: nano — a fixture asserting `out_layout.base != 0` reaches the emitted kernel; `omega.route.decisions` must be **unchanged** (a write offset must not change which route an op takes).
- re-prove: `cargo nextest run -p proxima-tensor --features tensor-write-offset write_offset` and `cargo nextest run -p omega --features metal,tensor-write-offset`
- log row: `ROW 260 — out_map stops being a pure projection: a constant write base is concat, and it costs no new Op.`

### 4.2 — KV cache as a caller-owned persistent device buffer
- worktree `proxima-wt-persist` · branch `perf/kv-device-resident` · own target dir · **depends on 4.1, 2.1**
- open: `proxima-model-interop/src/generate.rs:621-625` (`LayerCache {k_even, k_odd, v: Vec<f32>}`), `:636-640` (`append` = 3× `extend_from_slice`), `:642-653` (`named_blocks` hands the whole `Vec` as `QuantizedBlock::Float32`), `omega/src/metal.rs:1848-1849` (`NOCOPY_BUFFERS: BTreeMap<(usize,usize), MetalBuffer>` keyed pointer+len), `:1879-1892` (`upload_block_no_copy`), `:1903` (`upload_block_no_copy_uncached` — where `23e2e5e` now routes non-resident blocks, so **KV creates a fresh no-copy wrapper every token**), `:350-362` (`mark_resident`, classifies by NAME), `:991-1000` (strict `found != expected` → `InputSizeMismatch`), `proxima-tensor/src/align.rs:42-46,69` (`AlignedBuffer`, **zero production callers**)
- design: `LayerCache` allocates one `AlignedBuffer` per tensor at `bucket_capacity` (2.1) instead of growing a `Vec`; the buffer's pointer+length are stable, so `NOCOPY_BUFFERS`'s `(pointer, byte_length)` key **hits** from token 2 onward; `mark_resident` adds the KV names; the strict size check at `metal.rs:995` is relaxed to `found >= expected` **only for a block explicitly declared over-allocated** (a new field on `QuantizedBlock`, not a global relaxation — a global relaxation would hide real shape bugs). §1: this extends `AlignedBuffer`, it does not mint a peer type.
- expected N: ≥5 tests (pointer stability across appends, no-copy cache hit from step 2, over-allocated block accepted, under-allocated block still rejected, CPU/Metal parity on the greedy token). Plus `bind.rs:2818`'s `2651`/`"known"` assertion. **N==0 is RED.**
- prediction (milli → **bench**): `kv_cache_upload_bytes` per token falls to **0** after step 1; `nocopy_reuses` rises to 3×32=96/token; `block_upload_ms` falls; whole cell moves ≥3% (R2 attributes ~3.3% to KV re-upload) and, because the cache no longer reallocates, `phys_footprint_bytes` flattens.
- kill: a 34 GB allocation (the KV driver bug R3/M11 found and fixed in a worktree — `context_length` default used as capacity). Assert `device_allocated_bytes < 8 GB` in the test or it is RED.
- rollback: revert; feature `kv-device-resident` default-off.
- blast radius: `proxima-model-interop/src/generate.rs`, `omega/src/metal.rs` upload path, `proxima-tensor/src/align.rs`. The size-check relaxation is the highest-risk hunk in the plan — it is scoped to a declared-over-allocated block and has its own sad-path test.
- counter: `kv_cache_upload_bytes`, `nocopy_uploads`, `nocopy_reuses`, `nocopy_cache_len`, `resident_reuses`, `device_allocated_bytes`, `phys_footprint_bytes` (all in `token_breakdown_metal`, `generate.rs:1734-1745`).
- re-prove: BENCH-rung command, grep `kv_cache_upload_bytes=`.
- log row: `ROW 261 — AlignedBuffer gets its first production caller and the KV cache stops crossing the bus.`

### 4.3 — Single-range attention: 26 ops/layer → the minimal graph
- worktree `proxima-wt-onerange` · branch `perf/attention-single-range` · own target dir · **depends on 4.1, 4.2**
- open: `proxima-tensor/src/spec.rs:2303-2319` (the doc that must be rewritten — it currently *states* the constraint 4.1 removes), `:2336` (`append_mistral_cached_layer`, 530 lines, 25 args), `:2616-2617` (the two-range comment)
- design: with 4.1's write offset and 4.2's persistent buffer, `k_new`/`v_new` are written **into** the cache buffer at `base = row_stride * cached_len`, and attention becomes ONE Reduce over `[0, cached_len + new_count)` — the incumbent's exact shape (R8: `ggml_cpy` into a `ggml_view_1d` at a byte offset, then `mul_mat(KQ) → soft_max_ext → mul_mat(V)`). The even/odd RoPE split collapses in the same move because there is one range to rotate, not two.
- expected N: full CPU parity — `cpu::evaluate` must produce bit-identical logits to the two-range graph on the openchat fixture (R3/M11 proved 488/488 for the graph but noted `cpu::evaluate` lacked write placement; 4.1 supplies it). Plus the `2651`/`"known"` assertion. **N==0 is RED.**
- prediction (nano → **milli**): ops/layer falls from 37 to **≤23** (the incumbent's count at its checkout, R8: rms_norm, mul, wq, wk, wv, rope(Q), rope(K), cpy_k, cpy_v, mul_mat(KQ), soft_max_ext, mul_mat(V), cont, wo, add, rms_norm, mul, ffn_up, ffn_gate, silu, mul, ffn_down, add); `encode_dispatch_calls` falls from 1196 to **≤ 940** (R8 recomputes the incumbent at ~740, and R3/M11 measured 939 for single-range alone).
- **kill criterion, written against R12's control:** R12 measured that a dispatch collapse from 1194 → 616 moved wall by 0.07% and moved GPU time **up** 13.5% (35.117 → 39.841 ms). Therefore: this step is killed if `gpu_exec_ms` **rises** at all. A dispatch reduction that costs GPU time is the parallel branch's outcome and is not a win. The success shape is: dispatches down **and** `gpu_exec_ms` flat-or-down.
- rollback: revert; feature `attention-single-range` default-off. `spec.rs` is the file main moved most (`+8735/-2836` in `0c3bd4f`), so this is the highest-conflict hunk — rebase early and often against main.
- blast radius: `proxima-tensor/src/spec.rs` (`append_mistral_cached_layer` and every arch that calls it — Mistral, Qwen3, Qwen3.5 hybrid). Qwen3.5's hybrid attention/ssm path from `0c3bd4f` must be re-parity-tested, not assumed.
- counter: `encode_dispatch_calls`, `gpu_exec_ticks`, `omega.route.decisions` per-route split (Phase 3) — the census is what says whether the removed ops were elementwise or reduces.
- re-prove: BENCH-rung command, grep `encode_dispatch_calls=`; plus `cargo nextest run -p proxima-tensor --features attention-single-range` for the ops/layer nano assertion.
- log rows: `ROW 262 — one range, not two: the attention graph reaches the incumbent's op count.` `ROW 263 — the control that says a dispatch collapse is not automatically a win (R12's 616-dispatch cell restated on our tree).`

### 4.4 — Re-seal and restate
- worktree `proxima-wt-qbox` · branch `bench/quiet-seal-phase4` · same protocol as 2.3
- prediction (bench → **bench**): ratio vs llama.cpp-Metal in **1.6-2.1x**; `encode_dispatch_calls` ≤ 940 against the incumbent's ~740.
- kill: ratio does not improve over 2.3's → Phase 4's mass was already absorbed by Phase 2, which is itself the result and reorders Phase 5 ahead of any further graph work.
- log row: `ROW 264 — the board after write placement.`

---

# Phase 5 — One emitter core, backend coverage, geometry in config

Structural one-RISC debt. Removes ~78 near-duplicate functions (R5) and closes two §12 violations. Zero expected perf mass — and the row must say so up front, because a refactor headlined as a perf win is exactly the anti-headline the discipline forbids.

### 5.1 — Geometry constants into `omega-runtime.toml`
- worktree `proxima-wt-geoconf` · branch `refactor/omega-geometry-config` · own target dir
- open: `omega/omega-runtime.toml` (has only `[tiled_gemm]` with min_tokens/block_m/block_n/block_k), `omega/src/msl.rs:1017` (`PACKED_ROWS_PER_GROUP = 4`), `:1030` (`TILE_DIM = 8`), `:1046` (`TILED_GEMM_NSG = 4`), `:2527` (`SIMD_WIDTH as usize / lanes_per_block`), `omega/src/sized.rs:9,45` (`SIMD_WIDTH`'s "hardware-family fact, never a policy knob" classification — **stays**)
- design: new `[packed_row_block]` section (`rows_per_group`, `lanes_per_block`) and `[cooperative_reduce]` section (`max_threads`, the knob 0.3 introduced) beside `[tiled_gemm]`; emitted through the existing `build.rs` → `OUT_DIR/omega_sized.rs` path with per-key `OMEGA_<SECTION>_<KEY>` env overrides and one `cargo:rerun-if-env-changed` per override consulted (the file's own header documents this contract).
- expected N: a test per new const asserting the source value equals the TOML value; a test asserting an env override changes the emitted const (via `temp_env::with_vars`, AGENTS.md `### testing`). ≥4 tests. **N==0 is RED.**
- prediction (nano → **nano**): `grep -nE 'const (PACKED_ROWS_PER_GROUP|TILE_DIM|TILED_GEMM_NSG)' omega/src/msl.rs` returns **0 hits** afterwards; all references resolve through `crate::sized`.
- kill: a cached build ignoring an env override (the failure mode the rerun-if-env-changed directive exists to prevent) → the directive is missing; fix before landing.
- rollback: revert one commit.
- blast radius: `omega/omega-runtime.toml`, `omega/build.rs`, `omega/src/sized.rs`, `omega/src/msl.rs` const sites.
- counter: nano grep count == 0; the emitted `omega_sized.rs` contents.
- re-prove: `grep -cE 'const (PACKED_ROWS_PER_GROUP|TILE_DIM|TILED_GEMM_NSG): ' omega/src/msl.rs` → `0`, and `OMEGA_PACKED_ROW_BLOCK_ROWS_PER_GROUP=8 cargo build -p omega --features metal && grep ROWS_PER_GROUP target/.../omega_sized.rs`
- log row: `ROW 265 — every GPU geometry constant traces to omega-runtime.toml (§12); SIMD_WIDTH stays a hardware fact and the row says why.`

### 5.2 — One emitter core over the 4 bound kinds
- worktree `proxima-wt-emitcore` · branch `refactor/omega-one-emitter` · own target dir · **depends on 3.1**
- open: `omega/src/msl.rs:673-697`, `omega/src/wgsl.rs:105` (the 3 shared types + 2 shared fns), `omega/src/cuda.rs:66`, `omega/src/cuda.rs:146-183` (`emit_cuda`'s `CudaUnsupportedOpKind` rejections)
- design: `omega/src/emit.rs` — one core walking `(BoundOpKind, KernelRoute)` and consulting a `BackendText` **struct of `&'static str` + small fns**, never a trait object (§20; AGENTS.md `### source` box-free). The ~26 per-backend functions (validate, reduction_dims, bindings, grid_threads, entry_name, scalar_op_expr, fold_init_tokens, push_body_steps, preamble, kernel_signature, gather helpers, operand_read, render_*) collapse to one implementation each with a text table per backend.
- expected N: the **existing** msl/wgsl/cuda emission tests must pass unchanged (this is a behaviour-preserving refactor), plus a golden-source test per backend per route asserting byte-identical emitted source vs the pre-refactor emitter for the openchat decode plan. **N==0 is RED**, and a single byte of drift in the Metal golden is RED (the kernel is the product).
- prediction (nano → **micro**): `wc -l omega/src/{msl,wgsl,cuda}.rs` sum falls from 8479 by ≥2500; emitted Metal source for every one of the 1196 (or post-4.3 ≤940) ops is **byte-identical**; `gpu_exec_ms` unchanged within CoV.
- kill: any byte of Metal source drift not explained by a named route → revert; a refactor that changes the kernel is not a refactor.
- rollback: revert; **not feature-gated** (a default-off gate on a refactor would mean two emitters, which is the opposite of the goal) — so this lands as one atomic commit with the golden test as its bisect proof.
- blast radius: all of `omega/src/{msl,wgsl,cuda}.rs`. Largest blast radius in the plan; sequenced last for exactly that reason, and gated on a byte-exact golden.
- counter: line count, golden-source hash per route per backend, `gpu_exec_ticks` (must be flat).
- re-prove: `cargo nextest run -p omega --all-features golden_source` and `bash scripts/omega-gate.sh`
- log row: `ROW 266 — three emitters become one core plus three text tables; the kernel bytes did not move and here is the hash that proves it.`

### 5.3 — Every backend covers all four bound kinds
- worktree `proxima-wt-cover` · branch `feat/omega-backend-coverage` · own target dir · **depends on 5.2**
- open: `omega/src/cuda.rs:146-183` (rejects `Iota` and `Constant`), `omega/src/wgsl.rs` (covers 4 kinds, no tiled-GEMM / packed-row-block route), `omega/src/backend.rs:1-52` (six `Backend` variants, two implemented; `vulkan`/`npu`/`ane` name-only stubs)
- design: `CudaUnsupportedOpKind` is deleted; the core's exhaustive match makes coverage a compile-time property. WGSL gains `ReduceRowBlockedPacked` and `ReduceTiledGemm` through the core (the routes are shared; only the intrinsic text differs).
- expected N: a coverage test iterating `[Elementwise, Reduce(Reduce), Reduce(Scan), Iota, Constant] × [msl, wgsl, cuda]` = **15 emissions, all Ok**. **N==0 is RED**, and any `Err` is RED.
- prediction (nano → **micro**): the `wgpu-backend` arm (17 cfg sites, R5) runs the openchat decode plan end-to-end for the first time; correctness is the claim, not speed.
- kill: CUDA cannot be compiled on this host (no toolchain) → emission is still testable as **text**, which is the whole point of a sans-IO emitter (§11: "no I/O traits in the signature"). Assert the emitted CUDA parses structurally; do not claim it runs.
- rollback: revert; `CudaUnsupportedOpKind` restoration is mechanical.
- blast radius: `omega/src/{cuda.rs,wgsl.rs,error.rs}`.
- counter: coverage matrix 15/15; per-backend route census.
- re-prove: `cargo nextest run -p omega --all-features backend_coverage_matrix`
- log row: `ROW 267 — one RISC means every backend covers every kind: the 15-cell matrix and the deleted rejection.`

---

# Phase 6 — Residual levers, each gated on a re-measure

Every item here was measured or reasoned about **before** Phases 2-5 changed the tree. None may be built until re-measured. §18: a claim from a stale tree is not a claim.

### 6.1 — Re-measure the rematerialization subset
- worktree `proxima-wt-remat2` · branch `perf/remat-small-nodes` · own target dir
- open: R3/M8 — only the `elements < 247` subset (96 nodes) was a certain win (1196 → 1100); the aggregate set was a loss; "rematerialize all ≤2-consumer nodes" was a 6x downside on the slow ALU arm (R4, a **dead lever**).
- gate: after 4.3 the node population changed. Re-run the subset census **before** building anything.
- expected N: the census reports the post-4.3 count of `elements < 247` nodes. **N==0 is RED** — and N==0 would mean 4.3 already removed them, which is itself the finding and closes this item.
- prediction (nano → **milli**): the subset shrinks by more than half after 4.3 (most were attention-graph intermediates), and the remaining win is under 1% — below the noise floor at typical CoV, which retires the lever.
- kill: predicted win < 2× measured CoV → row it as "no signal, kept simpler form" and do not build.
- rollback: n/a if not built.
- blast radius: `proxima-tensor/src/bind.rs`.
- counter: node census by element count; `encode_dispatch_calls`.
- re-prove: `cargo run --release -p omega --features metal,instrument --example real_forward_emit_probe`
- log row: `ROW 268 — rematerialization re-measured on the post-write-placement graph.`

### 6.2 — On-device argmax
- worktree `proxima-wt-argmax` · branch `perf/on-device-argmax` · own target dir
- open: `proxima-model-interop/src/generate.rs:1640-1649` (`greedy_pick_started` … `sample_next_token`), R3/M7 — `greedy_pick` argmax depends on `waitUntilCompleted`, a **true data dependency**; the fix is to move argmax on-device, not to thread it.
- design: argmax is already expressible as `Reduce { body: Maximum, keep: Reduce }` plus an `out_scatter` index — `Reduce`'s doc at `op.rs:174-176` names argmax explicitly as one of the things `keep` + data-dependent `out_map` distinguishes. **No new Op.** This is a `spec.rs` change: append the argmax reduce to the program and read back 4 bytes instead of the vocab-sized logits.
- expected N: greedy token still `2651`/`"known"` (`bind.rs:2797-2804`); ≥3 tests (argmax over a fixture, ties broken like the CPU path, sampling path unaffected). **N==0 is RED.**
- prediction (milli → **bench**): `readback_bytes` falls from vocab×4 (≈128 KB) to 4; `readback_ms` falls from ~0.297 (R1) toward 0; `greedy_pick_ms` falls to ~0. Combined < 0.5 ms/token — small, and the row leads with that.
- kill: any drift in generated text; §14.
- rollback: revert; feature `on-device-argmax` default-off.
- blast radius: `proxima-tensor/src/spec.rs` output roots, `generate.rs` sampling. Note the **sampling** path (non-greedy) still needs full logits — the feature must not silently degrade `sample_config`; a test asserts that.
- counter: `readback_bytes`, `readback_calls`, `readback_ms`, `greedy_pick_ms`.
- re-prove: BENCH-rung command, grep `readback_bytes=`.
- log row: `ROW 269 — argmax moves on-device with no new Op; readback falls from 128 KB to 4 bytes and the win is under half a millisecond.`

### 6.3 — Split-K re-measure (simdgroup starvation)
- worktree `proxima-wt-skwide` · branch `perf/q4k-split-k-remeasure` · own target dir
- open: R3/M5 — rising marginal GB/s with simdgroup count (52 → 147 GB/s from 256 → 8001 simdgroups); low-row shapes (attn_q/k/v/o at 1024-4096 rows) starve. R7's `metal-q4k-split-k` patch from 0.2 is unbuilt/unmeasured.
- gate: 5.1 made `rows_per_group` a config knob; sweep it **first** — a config sweep is cheaper than a new kernel route and may retire the lever (§1 reuse-first, applied to geometry).
- expected N: sweep `OMEGA_PACKED_ROW_BLOCK_ROWS_PER_GROUP ∈ {1,2,4,8}` × 3 runs = **12 cells**. **N==0 is RED.**
- prediction (micro → **milli**): a knob value other than 4 wins on the low-row shapes by ≥5%; if none does, split-K's premise (more simdgroups is the fix) is refuted by a cheaper experiment and the kernel is not written.
- kill: no knob value beats 4 by more than CoV → do not build split-K; row the negative.
- rollback: knob is build-time config; revert to 4.
- blast radius: build config only, until/unless the kernel is built.
- counter: per-route GB/s from the Phase 3 census; `gpu_exec_ticks`.
- re-prove: the 12-cell sweep script.
- log row: `ROW 270 — the simdgroup-starvation hypothesis, tested with a config knob before a kernel.`

---

# Phase 7 — Contingencies (built only if a pre-registered kill fires)

### 7.1 — If 2.1's bucket masking breaks parity
Fall back to bucketing **without** graph-level masking by zero-filling the padded KV tail in `LayerCache` (the mask becomes a data property, not a graph property). Costs a memset per bucket crossing (once per 256 tokens), preserves `plan_hits`. Worktree `proxima-wt-zerotail`, branch `perf/kv-zero-tail`.

### 7.2 — If 4.2's size-check relaxation lets a real shape bug through
Replace the `found >= expected` relaxation with an explicit `QuantizedBlock::Float32Capacity { data, used_len, capacity }` variant so the over-allocation is in the type, not in a comparison. This is a new *data* variant on an existing enum, not a new type — the relocation question passes only if the call site changes (`generate.rs:642-653` does). Worktree `proxima-wt-capblock`, branch `feat/capacity-block`.

### 7.3 — If bucketing cannot reach `plan_hits > 0` at all
Make the reduce extent over the cache a **runtime uniform**: `BoundOp::extents: Vec<u64>` (`bind.rs:210`) gains a symbolic form and the Metal uniform block (`pack_uniforms`, `metal.rs:2211`) carries it. This is the larger change M6' names as option (b) and is held here, not in the primary path, because it touches every backend's grid computation (`msl.rs:1513-1557`). Worktree `proxima-wt-uniform`, branch `feat/runtime-extent-uniform`.

---

# Dependency graph

```
0.1 seal ──┬─> 0.2 patches ──┬─> 0.3 wide-reduce ─────────┐
           │                 ├─> 0.4 Q4K adjudication ──> 0.5 Q4K land ─┐
           │                 └─> 0.8 seal harness                        │
           ├─> 0.6 prune_dead ───────────────────────────────────────────┤
           └─> 0.7 CachedAttention adjudication ─────────────────────────┤
                                                                          v
                              2.1 kv-capacity-bucket ──> 2.2 buffer-pool ──> 2.3 re-seal
                                        │                       │
                                        │                       v
   [PARALLEL, no deps on 2.x]           │            3.1 KernelRoute ──> 3.2 census gate
   1.1 bandwidth probe ─┐               │                       │
   1.2 torch-MPS ───────┤               │                       v
   1.3 ORT-CoreML ──────┼─> scoreboard  └──────> 4.1 write-offset ──> 4.2 persistent KV
   1.4 llama -fa 1 ─────┤                                │                    │
   1.5 non-decode arms ─┤                                └────────────────────┴──> 4.3 single-range
   1.6 ai_docs ─────────┘                                                             │
                                                                                      v
                                                                                  4.4 re-seal
                                                                                      │
                                              5.1 geometry-config ──> 5.2 emitter core ──> 5.3 coverage
                                                        │                                       │
                                                        └──> 6.3 split-K sweep                  │
                                                4.3 ──> 6.1 remat re-measure                    │
                                                4.4 ──> 6.2 on-device argmax ───────────────────┘
```

Hard edges: 2.2 requires 2.1 (a buffer pool over an unstable plan reallocates anyway). 4.2 requires 2.1 (the bucket is what makes the buffer's length stable, which is what makes the `(pointer, byte_length)` key hit). 4.3 requires 4.1 **and** 4.2 (the write needs both an offset and a persistent destination). 5.2 requires 3.1 (the core dispatches on `KernelRoute`). 5.3 requires 5.2. 6.3 requires 5.1 (the knob must exist before it can be swept). Phase 1 has **no** edges into Phases 2-6 and is fully parallel; it only feeds the scoreboard columns.

Soft edge worth stating: 3.1 has no *build* dependency on Phase 2, but every Phase 4/5 **proof** depends on it, so it must land before 4.3.

---

# Rollback map

| step | rollback | main affected while branch is live | firewall |
|---|---|---|---|
| 0.3 | `git reset --hard`; drop branch | no | `metal-wide-cooperative-reduce` default-off |
| 0.4 | drop both features | no | two default-off features, one deleted at land |
| 0.5 | revert 1 commit | yes (it lands) | feature default-off; toggle to bisect |
| 0.6 | revert 1 commit | yes | no flag — pure generic pass; bisect by revert |
| 0.8 | revert 2 commits | scripts only | n/a |
| 1.1-1.5 | delete arm / revert flag | no Rust hot path | python + example + bench only |
| 1.6 | revert 1 commit | `ai_docs/` only | n/a |
| 2.1 | revert; feature off ⇒ bit-identical graph | yes | `kv-capacity-bucket` default-off |
| 2.2 | revert | yes | `metal-buffer-pool` default-off |
| 3.1 | revert (single commit, pure refactor) | yes | **no flag** — golden-source test is the firewall |
| 4.1 | revert | yes | `tensor-write-offset` default-off |
| 4.2 | revert | yes | `kv-device-resident` default-off; **highest-risk hunk is the `metal.rs:995` size check** — its sad-path test is the guard |
| 4.3 | revert | yes | `attention-single-range` default-off; highest rebase conflict (spec.rs) |
| 5.1 | revert | yes | consts move, values identical; nano test proves equality |
| 5.2 | revert (atomic commit) | yes | **no flag**; byte-exact golden-source hash is the firewall |
| 5.3 | restore `CudaUnsupportedOpKind` | yes | coverage matrix test |
| 6.1-6.3 | not built unless the kill passes | no | measured-first |

Every landing commit is a green bisect point (AGENTS.md `#### pr sequencing`): primitives before callers — 4.1 before 4.2 before 4.3; 3.1 before 5.2; 5.1 before 6.3. **No commit lands without owner authorization** (AGENTS.md `### git`).

---

# Abandoned designs

1. **`BoundOpKind::CachedAttention` — a fifth bound kind carrying an eight-input fused online-softmax attention macro-op**, together with the post-bind structural matcher (`cached_attention_candidates`, `attention_score_sources`, `is_exact_causal_mask`, `removable_attention_dependencies`) and the `physical.rs` module (+576) that supports it. **Ruled out by:** AGENTS.md `### constraint tiers` / hard invariants — "do not add arbitrary rules for specific instances; solve with sparse network and matrix structure" — reinforced by guiding-principles §1 (`Reduce.out_map` + `Layout.base` already express the write, so the expression exists and no type is earned) and §11 (a route decided by structural pattern-matching after bind is a runtime "is this the right shape" check where an exhaustive match would do). **What it changed:** the plan reaches the same 616-dispatch collapse through 4.1's write offset with **zero** new kinds, and keeps R12's measurement as the pre-registered control for 4.3's kill criterion instead of as a win.

2. **A `Backend` trait with per-backend `impl Emitter`** — the obvious way to unify `msl.rs` / `wgsl.rs` / `cuda.rs`. **Ruled out by:** guiding-principles §20 and AGENTS.md `### source` box-free-by-default. A trait over three backends is dynamic dispatch on a path that emits kernel text per op per token, and it would host the exact blanket-impl-under-a-new-name shape §20's last clause names. **What it changed:** 5.2's `BackendText` is a plain struct of `&'static str` fields consumed by one concrete core function with an exhaustive `match (BoundOpKind, KernelRoute)` — which is also why 5.3's coverage becomes a *compile-time* property instead of a runtime `CudaUnsupportedOpKind`.

3. **Threading `op_setup` across worker threads to hide the 4.4 ms** — the reflexive fix for a CPU-side stage cost. **Ruled out by:** R3/M9 (non-`Send` `MTLBuffer` blocks it at the type level) and, more bindingly, guiding-principles §21 lock-free-first: "a lock is usually a missing owner." The missing owner here is the `Plan` — once it is stable (2.1) the buffers belong to it and are allocated once (2.2), so there is no per-token work left to parallelize. **What it changed:** `PROXIMA_ORCH_THREADS` (landed on `perf/decode-orchestration-2`, timing unmeasured, R0) is **not** in this plan; 2.2 removes the work rather than distributing it.

4. **A `Concat` / `Op::Pad` / `PlacedBuffer` addition to express KV append** — the shape a reader of `spec.rs:2310-2313` would reach for first. **Ruled out by:** guiding-principles §1's two binary questions, answered by writing the expression: `Layout { base, strides }` already exists on every bound `Reduce` (`bind.rs:95-98`) and `Layout::offset_of` already adds it (`:100-108`); the call site with and without a `Concat` op is identical lines, which makes it a relocation. **What it changed:** the entire Phase 4 became a *constraint removal* in `project_output_shape` plus a driver alias, instead of a new variant every backend would have to learn.

5. **A spec-sheet GPU bandwidth figure to close the roofline debt** at `rooflines.md:411`. **Ruled out by:** guiding-principles §18 — a spec sheet is ASSUMED provenance and "a DERIVED number may never be the basis of a mechanism claim"; `rooflines.md:411` already explicitly refused this substitution once. **What it changed:** 1.1 is a real streaming-copy probe with the readback outside the timed window and a `readback_bytes == 0` assertion proving the window is clean, and its kill criterion refuses to write a ceiling that reads below the incumbent's achieved rate.

6. **Headlining the dispatch-count reduction** (1196 → 616 → 740-parity) as the plan's spine, which is what the one-line diagnosis proposed. **Ruled out by:** disciplined-component's frequency-weighted-scorecard rule plus R12's own control: 616 dispatches produced 51.535 ms wall against 1194's 51.571, with GPU time *up* 13.5%. **What it changed:** the entire phase ordering. Dispatch count moved from spine to counter, and the spine became (kernel body, plan stability, route census, then graph).

---

# Open questions the plan resolves by measurement (never by asking)

| # | question | resolved by | the number that answers it |
|---|---|---|---|
| Q1 | Do the 9 commits since `2b95210` already move `op_setup`/`prepare`, making R1's 11.4 ms stale? (`ff749a0` bounded the plan cache; `7d09145` bound checkpoint mapping once) | 0.1 | `op_setup_ms`, `prepare_ms`, `block_upload_bytes` on 4be2f3a |
| Q2 | Are `metal-q4k-mask-fma` and `q4k_pair_dot` the same mechanism at the same speed, or does one dominate? | 0.4 | milli-rung spread vs pooled CoV; parity max-abs-error vs f32 on `blk.0.attn_q.weight` |
| Q3 | Does bucketing `cached_len` actually reach `plan_hits > 0`, or is there a second per-token symbol? | 2.1 | `plan_hits` / `plan_misses` over 24 steps |
| Q4 | Does removing 1196 buffer allocations remove 4.4 ms, or does the cost reappear in `encode_dispatch`? | 2.2 | `op_setup_ms` and `encode_dispatch_ms` together, plus `step_wall_ms` |
| Q5 | Was `classify_kind` lying about the post-Q4K route distribution? | 3.1 | `route_of` census vs `classify_kind` buckets on the same plan, pre- and post-0.5 |
| Q6 | Does the write offset change any route decision? (it must not) | 4.1 | `omega.route.decisions` before/after, must be identical |
| Q7 | Does the persistent KV buffer actually hit `NOCOPY_BUFFERS`, or does something else invalidate the `(pointer, byte_length)` key? | 4.2 | `nocopy_reuses` (predicted 96/token), `kv_cache_upload_bytes` (predicted 0 after step 1) |
| Q8 | Does the single-range graph reduce GPU time, or only dispatch count — R12's control says the two can move opposite ways | 4.3 | `encode_dispatch_calls` **and** `gpu_exec_ms`, jointly |
| Q9 | Does the emitter unification change a single byte of Metal source? | 5.2 | golden-source hash per route, per backend |
| Q10 | Is the machine's streaming ceiling above the incumbent's achieved 228.9-234.1 GB/s, i.e. is there headroom at all? | 1.1 | GB/s at N=1 GiB with CoV |
| Q11 | Is llama.cpp with `-fa 1` a stronger incumbent at batch-1 decode, invalidating every ratio in the log? | 1.4 | ms/token, `-fa 0` vs `-fa 1`, interleaved |
| Q12 | Do torch-MPS and ORT-CoreML beat us on the mnist / train / BGE lanes, where no cell exists on either side? | 1.2, 1.3, 1.5 | ms/sentence and p50/p95/p99 per provider, with fidelity fields |
| Q13 | Does the rematerialization subset survive the post-4.3 graph? | 6.1 | count of `elements < 247` nodes after 4.3 |
| Q14 | Is simdgroup starvation fixable with a config knob, before writing a split-K kernel? | 6.3 | 12-cell `rows_per_group` sweep vs CoV |

---

### Critical Files for Implementation

- /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/bind.rs
- /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/spec.rs
- /Users/brianbruggeman/repos/slot-0/proxima/omega/src/msl.rs
- /Users/brianbruggeman/repos/slot-0/proxima/omega/src/metal.rs
- /Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/generate.rs