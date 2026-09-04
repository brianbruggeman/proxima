I have verified the ledger, both plans, and every critique finding against main `4be2f3a` read-only. No `proxima-wt-*` was entered.

---

# synthesis_AB — GPU parity for proxima-tensor through omega, ONE RISC

## Diagnosis (one paragraph, R13-cited)

R13 (MEASURED 2026-09-03, main `4be2f3a`, loaded box, arms interleaved) is the baseline and every delta below is a delta against one of its fields: `step_wall_ms` **67.92** (CoV 0.5%), `gpu_exec_ms` **56.93** (CoV 0.7%), incumbent `llama-bench -n 32 -p 0 -r 5 -t 8 -ngl 99` **57.08 t/s = 17.52 ms/token** (CoV 0.89%), ratio **3.88x** wall / **3.25x** kernel-only, `op_count` **1196**, `plan_hits=0 plan_misses=8`. Its gap decomposition (67.92 − 17.52 = 50.40) puts **26.9 ms** in Q4_K matvec above the incumbent's own achieved streaming rate, **16.6 ms** in non-matmul GPU (`reduce-cooperative` 385 ops / 9.113 ms + `elementwise` 547 ops / 7.350 ms + 0.17 degenerate), **11.0 ms** in orchestration (`op_setup` 3.9, `block_upload` 2.0, `prepare` 1.97, `emit` 0.81, `encode_dispatch` 0.47, `readback` 0.22, `pipeline_lookup` 0.04, ~1.6 residual), and 17.5 irreducible. The mass ordering the brief proposes is refuted at the bench rung by two independent cells: R12's control (1194 → 616 dispatches moved wall 51.571 → 51.535 and moved `gpu_exec` **up** 35.117 → 39.841) and R13's own per-op profile (`reduce-packed-row-blocked` = 225 ops carrying **44.450** of 61.082 diagnostic ms while 547 elementwise ops carry 7.350) — dispatch count is not the denominator; the Q4_K body is. The orchestration slice has a *proven* root cause on today's main: `resolve_plan`'s key is `(symbols[0], symbols[1]) = (new_count, cached_len)` (`generate.rs:966`) and `cached_len` is `Extent::Symbolic(1)` on every KV leaf (`spec.rs:6216-6245`), so `plan_hits=0` is the shape symbol, and `ff749a0`'s `self.plans.clear()` (`:973`) makes every token pay `plan_named` plus 1196 `newBufferWithLength` + 1196 `upload_uniforms` inside `encode_op` (`metal.rs:2210-2211`). Underneath all of it sit two instrument defects that make every attribution above unfalsifiable until they are fixed first: `classify_kind` buckets by substring of emitted MSL (`metal.rs:785-826`, its own doc at `:777-783` admits the route "is not exposed as its own accessor"; R12 ROW 263 measured it relabelling 9/601 → 225/385 when a body changed), and R13's carded finding that `operand_bytes` (`metal.rs:612`, computed at `:694` as `buffer.length()`) reports **4,140,417,024** — the whole checkpoint mapping buffer that `7d09145` introduced — so `gpu_ns_per_byte` and `total_operand_bytes` (`generate.rs:109-212`) are wrong and no GB/s row may be written until it is fixed. Verified on main this session and carried from Plan B: `layout_of` (`bind.rs:1594-1605`) already folds `axis.offset * stride` into `Layout.base`, `msl.rs:2733` already emits `long out_offset = u.out_base;`, `out_scatter` is rejected at `msl.rs:932`/`wgsl.rs:363`/`cuda.rs:240` but implemented at `cpu.rs:6892-6927`, `IndexMap::scatter` exists at `map.rs:172`, `UNIFORM_BUFFER_REUSES` already exists at `metal.rs:2069` with a live reuse path at `:2074`, and `scripts/sealed-pass.sh` is absent from main and hardcodes `REPO_ROOT=".../proxima-wt-seal"` on its branch.

---

## One-RISC binding (brief items 1–8, each bound to a step)

| # | brief clause | bound to | how it is proved, not asserted |
|---|---|---|---|
| 1 | ONE bound plan (`&[BoundOp]`, 4 kinds) from ONE rewrite engine, identical for every backend | **0.10** | a cross-backend test that binds the real openchat program once and asserts the `&[BoundOp]` fingerprint is byte-identical for the msl / wgsl / cuda / cpu emission paths; re-run as a gate step by 1.2 [crit M-7] |
| 2 | ONE first-class route enum decided before emission, censused `(NodeId, reason)` | **1.1, 1.2** | `KernelRoute` + `route_of`, recorded per **DISPATCH** inside `encode_op` beside `ENCODE_DISPATCH_CALLS` (`metal.rs:2244`), census-sum gate [crit V-5] |
| 3 | ONE emitter core over the 4 kinds, backend-specific TEXT only | **7.2** | one core + `BackendText` value table; byte-identity golden per route per backend, extending `emit_is_deterministic_byte_equal` (`msl.rs:4656`) |
| 4 | Every backend covers every kind | **5.3, 7.1** | `grep -c CudaUnsupportedOpKind == 0`, `grep -c ScatterNotSupported == 0`, 15-cell coverage matrix |
| 5 | ONE sizing config owning every geometry constant | **1.3** | `grep -cE '^const (PACKED_ROWS_PER_GROUP\|TILE_DIM\|TILED_GEMM_NSG)' omega/src/msl.rs == 0`; `SIMD_WIDTH` (`sized.rs:45`) stays a hardware fact with the reason in the row |
| 6 | Write placement expressed with existing `Reduce.out_map` / `out_layout.base`, NOT a new Op | **5.1, 5.2, 5.4** | scatter expression written first (§1), then the affine form; `Op` stays 5, `BoundOpKind` stays 4, asserted mechanically by 0.9's variant-count test |
| 7 | Driver-level persistent-buffer alias, NOT a new type | **5.5, 5.6** | `device_buffers.insert(node, (persistent, offset))` over the existing `BTreeMap<NodeId,(MetalBuffer,usize)>` (`metal.rs:2249`); `AlignedBuffer` (`align.rs:69`, zero production callers) gets its first caller |
| 8 | Llama-arch graph at ≤23 real ops/layer | **6.1, 6.2** | ops/layer census asserted in a test against R8's enumerated 23 |

**Non-negotiables**: `Op` stays 5 variants (`op.rs:175-266`); `BoundOpKind` stays 4 (`bind.rs:221-264`); no trait object, no `Box<dyn>`, no `PlacedBuffer`, no `Concat`/`Pad`/`Tile`; `plan`/`execute` stays not-a-pipe (adjudicated 2026-08-30, `backend.rs:1-52`).

---

## Global protocol

**G1. Naming (one worktree = one branch = one card).** Card `N.M` → worktree `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>` where `<NN>` is the card's sequential id in the table below; branch `gpu-risc/<NN>-<slug>`; target dir `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>/target`. Verified this session: **zero** existing worktree dirs match `risc`, and `git branch --list 'gpu-risc/*'` returns **0**. The three branch names Plan A and Plan B reused are already checked out and `git worktree add` would refuse them: `perf/kv-device-resident` (at `proxima-wt-drive`), `perf/attention-single-range` (at `proxima-wt-merge`), `bench/sealed-pass` (at `proxima-wt-seal`), plus `perf/route-census`, `perf/op-rule-census`, `perf/q4k-split-k`, `perf/q4k-orchestration`, `perf/q4k-independent-accumulators`, `perf/gpu-all-wins`, `perf/gpu-dispatch-count`, `perf/output-placement`, `perf/metal-simdgroup-geometry` — **none of these names appears in this plan** [crit g].

**G2. Creation commands, welded into every card (Luna runs these verbatim):**
```
git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add \
  /Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN> -b gpu-risc/<NN>-<slug> <base-ref>
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>/target
export CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>/target
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>
```
Teardown at rollback: `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree remove --force /Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>` [crit M-3].

**G3. The harness command and its real token count.** The three harnesses, verified by name on main:
- `proxima-model-interop/src/bind.rs:2765` `runs_one_real_forward_pass_and_greedy_picks_a_real_token` — the §14 oracle, asserts `generated.0[0] == 2651` and `generated.1 == "known"` (`:2797-2803`), CPU path.
- `bind.rs:2818` `runs_a_cached_greedy_decode_loop_and_reports_per_token_wall_clock` — multi-token, prints `token_breakdown` (`generate.rs:1655`, fields incl. `kv_cache_upload_bytes`, `greedy_pick_ms`) [crit H-5].
- `bind.rs:3002` `runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache` — the BENCH-rung cell, prints `token_breakdown_metal` (`generate.rs:1721-1750`, fields incl. `plan_hits`, `plan_misses`, `nocopy_reuses`, `mapping_offset_uploads`, `device_allocated_bytes`) and `metal_decode_summary` (`bind.rs:3041`).
- `bind.rs:3084` `profiles_one_real_decode_step_by_per_op_gpu_time` — the per-op diagnostic (one command buffer per op; R13 records this mode inflating Σ to 61.082 vs batched 56.93 = **+7.3%**; every per-op number must carry that inflation note) [crit V-3].

```
PROXIMA_MAX_TOKENS=8 CARGO_TARGET_DIR=$TD cargo nextest run -p proxima-model-interop \
  --release --features std,metal,instrument --lib --run-ignored all \
  -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' \
  --no-capture 2>&1 | tee runs/cell-<NN>-<arm>-<run>.txt
```
`--run-ignored all` is **mandatory** (`#[ignore]` at `bind.rs:3001`); the test also `return`s silently when the gguf is absent (`bind.rs:3004-3011`), so a missing fixture exits 0 — **N==0 is RED everywhere.**

**Token-count contract (R13: 8 tokens, 7 steady steps at `PROXIMA_MAX_TOKENS=8`).** `decode_loop_max_tokens()` defaults to 24 (`bind.rs:2718-2723`); R13 ran at 8 and reports `plan_misses=8` with `step_wall_ms` means over **steps 1-7 (7 rows per run)** — step 0 carries prefill and is excluded from every mean. eos behaviour: `run_decode_loop` may stop early on the model's own eos and `forward_calls_taken = generated.0.len() + usize::from(generated.2)` (`bind.rs:3051`), so **every N assertion is written eos-invariantly** as
`token_breakdown_metal lines == metal_decode_summary plan_misses == tokens_generated + stopped_by_eos`, with the steady-state row count = that number − 1. Do not assert a literal 24 or 120 [crit M-9].

**G4. Gate N contract.** `scripts/omega-gate.sh:37-47` asserts a **nonzero** count, not a specific N (verified this session) — so "gate green" means "nonzero and green", and any card claiming a specific N states it as its own explicit `nextest` count under a *named* feature set. Never compare a single-feature count to `--all-features`: `--all-features` additionally enables `cuda`, `wgpu-backend`, `metal-tiled-gemm`, `vulkan`, `npu`, `ane`, all of which contribute tests [crit M-10]. Note `omega-gate.sh [2/6]` builds `--all-targets --all-features`, so a `wgpu_driver.rs` compile break lands in the gate for every subsequent card [crit H-6].

**G5. Feature declaration rule.** Every feature a card's re-prove command names must be declared in **the manifest that the `-p` names** and forwarded downward. Verified today: `omega/Cargo.toml [features]` = `default(std,metal,cpu)/std/alloc/cpu/metal/vulkan/cuda/npu/ane/instrument/metal-tiled-gemm/wgpu-backend`; `proxima-model-interop/Cargo.toml [features]` = `default/std/interop-bgpool/instrument/metal/metal-tiled-gemm`; `proxima-tensor/Cargo.toml [features]` = `default/std/alloc/config/q4k-int8-dot/q5k-int8-dot/q6k-int8-dot/...`. So e.g. card 4.1's `kv-capacity-bucket` is declared in **`proxima-tensor/Cargo.toml`** and forwarded as `proxima-model-interop/kv-capacity-bucket = ["proxima-tensor/kv-capacity-bucket"]`; card 5.4's `tensor-write-offset` is declared in `proxima-tensor` and forwarded through **both** `omega/Cargo.toml` and `proxima-model-interop/Cargo.toml` [crit M-4]. Each card names its manifest edits explicitly.

**G6. Measurement serialization — one measurer on the box.** GPU-measuring cards hold the box exclusively; there is no parallel Phase. The measurer queue order is fixed: **0.3 → 0.4 → 2.3 → 3.1 → 3.3 → 4.1 → 4.2 → 4.3 → 5.2 → 5.6 → 6.2 → 8.1 → 8.2 → 8.3 → 8.4 → 8.5 → 10.1 → 9.2.** Protocol per measuring card: (i) build every arm to completion first, (ii) `scripts/sealed-pass.sh`'s quiet gate must pass (`SEALED_PASS_LOAD_THRESHOLD`, builder-name check `cargo|rustc|cc|clang|ld` minus `cdb-daemon|sccache|rust-analyzer`), (iii) create `/Users/brianbruggeman/repos/slot-0/.gpu-measurer.lock` containing the card id and remove it at the end; a card that finds the lock held **waits, it does not measure**. Non-measuring cards (edits, docs, nano tests) may proceed concurrently only if they run no `cargo build` while the lock is held [crit O-1].

**G7. Arms and CoV.** Every measuring card runs arms **interleaved** A B A B (never before-block/after-block), ≥5 runs, reports mean + CoV, and carries the llama.cpp-Metal home-turf arm on the same pass. CoV > 5% ⇒ sweep, never a point estimate. R13's CoV bands are the noise floor: `step_wall_ms` **0.5%**, `gpu_exec_ms` **0.7%**, incumbent **0.89%**. No kill criterion may be set inside a CoV band [crit b].

**G8. Row numbers.** Branches carry `ROW <PH-NN>` **placeholders only**; the literal number is assigned at land time from main's last row, **233** (`proxima-tensor/docs/discipline.md:18736`). Card 0.11 owns the assignment protocol. No card in this plan states a literal ROW number [crit O-3].

**G9. Provenance.** Every number carries its ledger section. MEMORY figures may appear only as "MEMORY, superseded by R13" — R1's `op_setup 4.394` → R13 **3.9**; R1's `prepare 2.087` → R13 **1.97**; R1's `readback 0.297` → R13 **0.22**; R2's "KV re-upload ~3.3%" has **no counterpart term** in R13's decomposition and may not anchor a prediction [crit a].

**G10. Bench ladder.** nano (counts, no device: op/dispatch/route-census/variant counts, emitted-source hashes) → micro (one kernel on device: `q4k_matvec_probe`, `membw_probe`, `metal_vs_cpu`) → milli (one decode step's stage split from `token_breakdown_metal`) → bench (the interleaved paired board cell vs llama.cpp-Metal). Every prediction is **exactly one rung ahead**; a card with nothing to measure states `predict: none — this card produces records, not a measurement`. **No same-rung predictions** [crit M-11]. A miss kills the climb and is decomposed in the row into *inconsistency* (two rungs disagree about the same quantity) vs *understanding-gap* (the mechanism is not what we named), each with a named work item.

**G11. Tiers.** `hands` = Luna (tool use, bounded edits, runs the given commands, no design judgment). `worker` = writes non-trivial code against a named design. `judge` = adjudicates a pre-registered decision rule against numbers; never invents the rule.

---

# Phase 0 — instrument truth, the baseline, and what already exists

*Nothing downstream is attributable until 0.1 and 1.1 land. Phase 0 fixes the instruments that lie, replicates R13 on a quiet box, and lands the work that already exists without carrying it in a `/tmp` directory.*

### 0.1 — Fix `op_profile` byte accounting BEFORE any GB/s row `[crit M-2]`
- **tier** worker · **depends_on** — (first card)
- **worktree/branch/target** `proxima-wt-risc01` · `gpu-risc/01-op-profile-byte-accounting` · own target
  ```
  git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add \
    /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01 -b gpu-risc/01-op-profile-byte-accounting 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc01/target
  ```
- **opens** `omega/src/metal.rs:612` (`pub operand_bytes: u64`), `:694-700` (the defect: `device_buffers.get(source).map(|(buffer,_offset)| buffer.length())` — since `7d09145` every weight resolves to the ONE checkpoint mapping buffer, so this reports 4,140,417,024 per matvec op, R13); `proxima-model-interop/src/generate.rs:109,113,123,128,139-150,171-193,211-212` (every `gpu_ns_per_byte` / `total_operand_bytes` / `passed_operand_bytes` consumer); `proxima-tensor/src/bind.rs:200-215` (`BoundOp {extents}`) and the `(buffer, offset)` `DeviceBuffer` pair.
- **commands** replace `buffer.length()` with the **tensor's** byte length derived from the operand's own bound extents and dtype (rows·k·bytes-per-element, packed codec accounted); keep the buffer length as a separate `bound_buffer_bytes` field so the mapping-offset behaviour stays observable. `bash scripts/omega-gate.sh`; then the G3 per-op harness at `bind.rs:3084`.
- **expect** N = the per-op profile emits one row per bound op (R13: **1196**) and `total_operand_bytes` for one step is **< 6 GB** (R13's wrong value is 1.2 TB); ≥3 new tests: Q4_K operand bytes match `rows*k*0.5625`, an f32 operand matches `elements*4`, an operand bound at a non-zero mapping offset reports the tensor length not the buffer length. **N==0 is RED**; a `total_operand_bytes` still above 1 TB is RED.
- **predict (nano → micro)** with corrected bytes, the `ffn_up` family's derived rate lands at **97.4 GB/s ± 5** and `output.weight` (Q6_K) at **145.9 ± 8** — R13's shape-derived true-bytes column, reproduced by the instrument instead of by hand.
- **kill** the corrected per-family GB/s disagrees with R13's hand-derived column by >10% ⇒ the byte model is wrong in a second place; do not write any GB/s row (8.2 is blocked) until it agrees.
- **rollback** `git revert`; instrument-only surface, no default-path behaviour.
- **blast** `omega/src/metal.rs` `OpGpuTiming` struct + producer, `proxima-model-interop/src/generate.rs` profile printers. Instrument-gated; zero kernel change.
- **observe** `operand_bytes`, `total_operand_bytes`, `gpu_ns_per_byte`, plus the new `bound_buffer_bytes`.
- **reprove** `PROXIMA_MAX_TOKENS=8 cargo nextest run -p proxima-model-interop --release --features std,metal,instrument --lib --run-ignored all -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' --no-capture`
- **log-row title** `the profiler was reporting the checkpoint buffer, not the tensor: every GB/s row before this one is void`

### 0.2 — The harness N-contract and the `plan_hits` coupling, recorded `[crit R-1, f, M-9]`
- **tier** worker · **depends_on** 0.1
- **worktree/branch/target** `proxima-wt-risc02` · `gpu-risc/02-harness-n-contract` · own target (creation per G2, base `gpu-risc/01-...`)
- **opens** `proxima-model-interop/src/bind.rs:3041-3060` — `metal_decode_summary` print, then `let forward_calls_taken = generated.0.len() + usize::from(generated.2);` and `assert_eq!(runtime.plan_hits, 0, "cached_len grows every decode step, so no (new_count, cached_len) shape can repeat within one call")`; the test's own doc at `:2994-2998` which states the zero-hit finding as the test's purpose; `generate.rs:905` (`pub(crate) plan_hits: usize`, the **single** definition), `:968` (increment inside `resolve_plan`), `:1764` and `bind.rs:3046` (the two prints of that same field — R13's parenthetical that these are different counters does not hold on main; `grep -rn plan_hits` returns only these sites).
- **commands** do **not** weaken the assertion. Restate it as a contract keyed to the feature that will change it: `assert_eq!(runtime.plan_hits, expected_plan_hits())` where `expected_plan_hits()` is `0` on main's shape and is changed **by card 4.1 in the same commit that changes the shape**. Add the eos-invariant N assertions from G3: `token_breakdown_metal` line count == `plan_misses` == `tokens_generated + stopped_by_eos`.
- **expect** N = the harness prints exactly `tokens_generated + stopped_by_eos` breakdown lines (R13 at `PROXIMA_MAX_TOKENS=8`: **8**, `stopped_by_eos=false`, `plan_misses=8`); the assertion still fires on an artificially injected hit. **N==0 is RED.**
- **predict (nano → micro)** on today's main the contract is satisfied with `expected_plan_hits() == 0` and `metal_decode_summary` reads `plan_hits=0 plan_misses=8` — reproducing R13's field exactly.
- **kill** the harness reads a `plan_hits` other than 0 on unmodified main ⇒ the counter's semantics moved since R13; stop and re-derive before 4.1 is designed.
- **rollback** `git revert` one commit; test-only.
- **blast** `proxima-model-interop/src/bind.rs` test module only. Every card from 2.3 onward re-proves through this harness, so it is edited once, here.
- **observe** `plan_hits`, `plan_misses`, `plan_cache_len`, `tokens_generated`, `stopped_by_eos`.
- **reprove** the G3 command; `grep -o 'plan_hits=[0-9]*\|plan_misses=[0-9]*' runs/cell-02-*.txt`
- **log-row title** `the bench harness asserts the defect: what it takes to move plan_hits without a red gate`

### 0.3 — Quiet-box replicate of R13 (delta and CoV band, not a new baseline) `[crit S-1, A 0.1, B P0.1]`
- **tier** hands · **depends_on** 0.2 · **measurer queue position 1**
- **worktree/branch/target** `proxima-wt-risc03` · `gpu-risc/03-quiet-box-replicate` · own target (base `gpu-risc/02-...`)
- **opens** `scratchpad/baseline-2026-09-03/{llama_A1..3,ours_B1..3,profile}.log` (R13's raw logs), `bind.rs:3002`, `generate.rs:1721-1750`
- **commands** close every other agent; AC power; `uptime` load recorded before and after each run. 5 interleaved pairs of (G3 decode cell at `PROXIMA_MAX_TOKENS=8`) and (`llama-bench -m /Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf -n 32 -p 0 -r 5 -t 8 -ngl 99`).
- **expect** N = 5 ours-runs × (`tokens_generated + stopped_by_eos`) breakdown lines each, all parsed; 5 llama-bench tables. **N==0 is RED.**
- **predict (milli → bench)** on a quiet box (load < 2.0) `step_wall_ms` lands within **±3%** of R13's **67.92** and `gpu_exec_ms` within ±3% of **56.93**; `op_count` is **1196**; CoV tightens from R13's 0.5%/0.7% (R13 was taken on a LOADED box, load 4.7-5.7, cdb-daemon resident) rather than the means moving.
- **kill** `op_count != 1196` or `step_wall_ms` outside ±5% of 67.92 ⇒ the box or the tree is not what R13 measured; stop, and do not proceed until the difference is named. This is **not** a re-baseline: R13 stays THE baseline; this card produces the quiet-box CoV band that every later kill criterion is set against.
- **rollback** none (measurement only); `git worktree remove`.
- **blast** none. Zero source change.
- **observe** `step_wall_ms`, `gpu_exec_ms`, `op_setup_ms`, `prepare_ms`, `block_upload_ms`, `encode_dispatch_ms`, `readback_ms`, `plan_hits`, `plan_misses`, `encode_dispatch_calls`, `uptime` load before/after.
- **reprove** the G3 command + the llama-bench command, interleaved.
- **log-row title** `R13 replicated on a quiet box: the CoV band every later kill criterion is set inside`

### 0.4 — Land `sealed-pass.sh` with its hardcoded paths removed `[B P0.1, A 0.8]`
- **tier** worker · **depends_on** 0.3 · **measurer queue position 2**
- **worktree/branch/target** `proxima-wt-risc04` · `gpu-risc/04-sealed-pass-harness` · own target (base `gpu-risc/03-...`). **Branch `bench/sealed-pass` is checked out at `proxima-wt-seal` and may not be reused** [crit g].
- **opens** `git show bench/sealed-pass:scripts/sealed-pass.sh` — verified this session: `:4` `REPO_ROOT="/Users/brianbruggeman/repos/slot-0/proxima-wt-seal"`, `:26-29` four hardcoded sibling worktrees (`proxima-wt-orch2`, `-nanofix`, `-transa`, `-train`), plus the quiet-gate constants at `:8-18` and the MAC/weight-byte constants at `:31+`.
- **commands** `git cherry-pick b437f49 fb61d04`; in the **same** commit replace `REPO_ROOT` with `"$(git rev-parse --show-toplevel)"` and delete the four sibling-worktree arms (they are re-added as real cells in Phase 8). `bash scripts/sealed-pass.sh --dry-run` then a full pass.
- **expect** N = the script enumerates ≥2 arms (ours, llama.cpp) and exits 0; `grep -c 'proxima-wt-' scripts/sealed-pass.sh == 0`. **N==0 is RED.**
- **predict (milli → bench)** the script's own ours/llama cells reproduce 0.3's numbers within 0.3's measured CoV band.
- **kill** the script's numbers disagree with 0.3 beyond the CoV band ⇒ it is measuring something else (different token budget, different arm order); fix before any later card uses it.
- **rollback** `git revert` 2 commits; `scripts/` only.
- **blast** `scripts/sealed-pass.sh` (new file at a project-level path). No library code.
- **observe** arm count, per-arm CoV, the quiet-gate load reading it recorded.
- **reprove** `bash scripts/sealed-pass.sh`
- **log-row title** `the seal is a script, not a memory, and it no longer points at one worktree`

### 0.5 — Quarantine the uncommitted worktree diffs INSIDE the repo `[A 0.2, crit R-2, M-5]`
- **tier** hands · **depends_on** 0.2
- **worktree/branch/target** `proxima-wt-risc05` · `gpu-risc/05-quarantine-uncommitted-wins` · own target
- **opens** R7's table (10 worktrees, `perf/gpu-all-wins` 13 files +3859/-197 among them)
- **commands** — real commands, no prose [crit M-5]. Luna runs these **from inside each named worktree** (the plan author did not enter them):
  ```
  mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered
  for wt in all drive rules splitk place merge gpudisp gpuker q4k lat; do
    ( cd /Users/brianbruggeman/repos/slot-0/proxima-wt-$wt && \
      git diff  > /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered/$wt-tracked.patch && \
      git status --porcelain > /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05/docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered/$wt-status.txt )
  done
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc05 && git add docs/bench-campaigns && git status --porcelain
  ```
  The patches are **committed to this branch**, not left in `/tmp` — `/tmp` is OS-cleared and Plan A's stated rollback was to delete it, which would destroy the only copy of the −17.2% / −4.9% / −11.1% results [crit R-2, B-2].
- **expect** N = **10** `*-tracked.patch` files present and non-empty; each `git apply --check` against `2b95210` exits 0. **N==0 is RED**, and an empty patch for a worktree R7 lists as dirty is RED.
- **predict (nano → micro)** each patch applies cleanly at `2b95210` and conflicts at `4be2f3a` in `spec.rs` and/or `omega/src/metal.rs` (main moved `spec.rs +8735/-2836` in `0c3bd4f` and `metal.rs` in `7d09145`/`23e2e5e`).
- **kill** a patch that fails `git apply --check` at `2b95210` ⇒ that worktree mutated since the R7 audit; re-audit it before anything is landed from it.
- **rollback** `git revert` one commit — the patches remain in git history, which is the point.
- **blast** `docs/bench-campaigns/` only. Nothing on any build path.
- **observe** patch count, per-patch line counts vs R7's table, `git apply --check` exit codes at both refs.
- **reprove** `for p in docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered/*.patch; do git apply --check --3way "$p"; echo "$p $?"; done`
- **log-row title** `the measured-but-uncommitted wins enter git before anything else touches them`

### 0.6 — Adopt `prune_dead` / `dead_resolved_nodes` `[A 0.6, B P0.7]`
- **tier** worker · **depends_on** 0.5
- **worktree/branch/target** `proxima-wt-risc06` · `gpu-risc/06-bind-prune-dead` · own target
- **opens** `proxima-tensor/src/bind.rs:200-215` (`BoundOp`); `git show 216d925` on `perf/cached-attention-streaming` ("drop dead resolved nodes before GPU dispatch") — R12 marks it generic and RISC-conformant, the one piece of that 42-commit branch that lands unmodified.
- **commands** `git cherry-pick 216d925`; `bash scripts/proxima-tensor-gate.sh`; `bash scripts/omega-gate.sh`.
- **expect** N ≥ 2 new tests (a program with a dead resolved node; a program with none) plus the tensor gate's nonzero count green. **N==0 is RED.**
- **predict (nano → micro)** `encode_dispatch_calls` falls from R13's **1196** by exactly the pruned-node count on the openchat graph, and the census (1.1, once landed) names every removed node; if the count is 0 this is a correctness/generality landing and the row leads with that.
- **kill** any change to `encode_dispatch_calls` the pruner cannot name node-by-node, or any parity regression (a pruner that changes output is not a pruner).
- **rollback** `git revert` one commit. No feature flag — it is a generic pass and lands in `default`; bisect by revert.
- **blast** `proxima-tensor/src/bind.rs`. Every backend consumes the same shortened `&[BoundOp]`, which is why 0.10's cross-backend identity test runs after it.
- **observe** `encode_dispatch_calls`, plus a nano assertion on `BoundOp` count for a fixture program.
- **reprove** `cargo nextest run -p proxima-tensor --features std,instrument -E 'test(prune_dead)'`
- **log-row title** `dead resolved nodes never reach a backend`

### 0.7 — Adjudicate `BoundOpKind::CachedAttention`; extract the parallel branch's rows `[A 0.7, B P0.6]`
- **tier** judge · **depends_on** 0.5
- **worktree/branch/target** `proxima-wt-risc07` · `gpu-risc/07-adjudicate-cached-attention` · own target
- **opens** `git diff main..perf/cached-attention-streaming -- proxima-tensor/src/bind.rs proxima-tensor/src/physical.rs`; `git show perf/cached-attention-streaming:failure-cached-attention-matcher.md`; against them `proxima-tensor/src/bind.rs:221-264` (4 kinds), `op.rs:175-266` (5 ops), workspace `AGENTS.md` hard invariant "do not add arbitrary rules/code for specific instances". Read the **diff hunks**, not the commit list.
- **commands** the three `git show`/`git diff` above; extract ROWs 234-267 verbatim into `docs/bench-campaigns/2026-09-03-gpu-one-risc/parallel-rows.md`; one docs commit.
- **expect** N = **34** rows extracted (234-267), each tagged keep / renumber / supersede; +1 `ai_docs/invariants.jsonl` record `proxima.omega.one_risc_bound_kinds`. **N==0 is RED.**
- **predict** none — this card produces a recorded decision boundary and a reusable invariant, not a measurement.
- **kill (the re-open condition, pre-registered)** if 6.2 later measures that the online-softmax cluster cannot reach ≤23 ops/layer through `out_map` placement alone, this adjudication re-opens **with that number attached**.
- **rollback** docs revert.
- **blast** `proxima-tensor/docs/discipline.md`, `ai_docs/invariants.jsonl`. No source.
- **observe** row count landed; `git log --oneline main..perf/cached-attention-streaming | wc -l == 42` accounted for member-by-member.
- **reprove** `jq -c 'select(.id=="proxima.omega.one_risc_bound_kinds")' ai_docs/invariants.jsonl`
- **log-row titles** `the fifth bound kind the binding does not admit, and what its measurement is kept for` · `the parallel lane's measured negatives, renumbered onto main (nsg=2 fourth negative, float4 accumulation, explicit unroll, width-64 dispatch, row-batch 8, tiled-gemm at decode)`

### 0.8 — Clean the tree so interleaved A/B is safe `[B P0.9]`
- **tier** hands · **depends_on** —
- **worktree/branch/target** `proxima-wt-risc08` · `gpu-risc/08-clean-tree` · own target
- **opens** `git status --porcelain` on main — verified this session: `?? docs/bench-campaigns/2026-09-03-gpu-one-risc/` and `?? proxima-onnx/scripts/torch_reference/venv/`; the pattern to mirror is `scripts/onnx_reference/.gitignore`.
- **commands** add `venv/` to `proxima-onnx/scripts/torch_reference/.gitignore`; commit the campaign directory as real content (it is this session's evidence). One `chore:` commit.
- **expect** N = `git status --porcelain | wc -l` is **0**. **Non-zero is RED** for every later interleaved measurement (a dirty tree makes stash-based A/B unsafe).
- **predict** none — a state assertion, not a measurement.
- **kill** n/a. **rollback** `git revert`. **blast** ignore file + one directory of evidence.
- **observe** `git status --porcelain` line count.
- **reprove** `git status --porcelain`
- **log-row title** `a dirty tree is not a measurable tree`

### 0.9 — Fix `op.rs:166` and make the RISC's cardinality mechanically re-provable `[crit M-8]`
- **tier** hands · **depends_on** 0.6
- **worktree/branch/target** `proxima-wt-risc09` · `gpu-risc/09-risc-cardinality` · own target
- **opens** `proxima-tensor/src/op.rs:166` — verified this session, the doc header reads **"The four generators."** directly above `pub enum Op` at `:175`, which has **5** variants (Input, Elementwise, Reduce, Iota, Constant); `bind.rs:221-264` (`BoundOpKind`, 4 variants).
- **commands** correct the header to name five generators and say which is which; add two assertion tests: `Op` has exactly 5 variants and `BoundOpKind` exactly 4, each written as an exhaustive `match` over a constructed value so adding a variant breaks compilation rather than a count.
- **expect** N = 2 new tests; `grep -c "The four generators" proxima-tensor/src/op.rs == 0`. **N==0 is RED.**
- **predict (nano → micro)** the exhaustive matches compile on today's tree with no `_ =>` arm — i.e. the RISC's cardinality is 5/4 as R5 reads it, and any future fifth bound kind fails the build before it reaches a review.
- **kill** the exhaustive match needs a wildcard arm ⇒ the variant count is not what R5 read; stop and re-derive the one-RISC binding.
- **rollback** `git revert`. **blast** `proxima-tensor/src/op.rs` doc + test module.
- **observe** variant counts; the doc grep.
- **reprove** `cargo nextest run -p proxima-tensor --features std -E 'test(risc_cardinality)'`
- **log-row title** `the RISC's own doc said four over five variants; the cardinality is now a compile error to change`

### 0.10 — One bound plan, identical for every backend (brief one-RISC item 1) `[crit M-7]`
- **tier** worker · **depends_on** 0.9
- **worktree/branch/target** `proxima-wt-risc10` · `gpu-risc/10-one-bound-plan-identity` · own target
- **opens** `proxima-tensor/src/bind.rs:200-215` (`BoundOp`), `:221-264` (`BoundOpKind`), `:95-98` (`Layout`), `:82` (`MAX_INLINE_RANK=4`); `omega/src/msl.rs:673-697` (`emit`'s entry), `omega/src/wgsl.rs:105`, `omega/src/cuda.rs:66,146-183`; `proxima-tensor/src/cpu.rs` evaluate entry.
- **commands** add one test that binds the real openchat decode program **once** and asserts the resulting `&[BoundOp]` is bit-identical whichever backend consumes it — i.e. that `bind` is called once and no backend re-binds or rewrites. Serialize the plan to a stable fingerprint (node id, dtype, extents, kind discriminant, layout base/strides, operand ids) and assert equality across the four consumption paths.
- **expect** N = 1 fingerprint, 4 consumers, **4/4 equal**; ≥1 negative test (a deliberately mutated plan is detected). **N==0 is RED**, any inequality is RED.
- **predict (nano → micro)** the fingerprint is identical across all four and the plan length equals R13's **1196** minus 0.6's pruned count.
- **kill** any backend produces a different `&[BoundOp]` ⇒ there is more than one rewrite engine and the one-RISC claim is false today; that finding outranks every performance card and Phase 7 is promoted to name the second engine.
- **rollback** `git revert`; test-only.
- **blast** test module in `omega` (it is the crate that sees all emitters); no production code.
- **observe** the plan fingerprint, plan length, per-backend equality matrix.
- **reprove** `cargo nextest run -p omega --features metal,cpu,instrument -E 'test(one_bound_plan_identity)'`
- **log-row title** `one bound plan, four consumers, one fingerprint: brief item 1 becomes a test`

### 0.11 — Row-number placeholder protocol `[crit O-3, A 0.7, B P0.8]`
- **tier** hands · **depends_on** 0.7
- **worktree/branch/target** `proxima-wt-risc11` · `gpu-risc/11-row-number-protocol` · own target
- **opens** `proxima-tensor/docs/discipline.md:18736` (last row on main is **ROW 233**); R10's record of ROW 205 at line 17864 preceding ROW 204 at 17950 — non-monotonic numbering already produced by concurrent worktrees; the three unlanded "ROW 234"s and the parallel branch's ROW 234-267.
- **commands** every branch writes `## ROW <PH-NN>` where `NN` is this plan's card id. At land time: `grep -n "^## ROW" proxima-tensor/docs/discipline.md | tail -1` gives the current maximum; the landing commit rewrites the placeholder to `max+1` and appends. A row whose re-prove command does not run **today** does not land (§16).
- **expect** N = `grep -c "^## ROW <PH-" proxima-tensor/docs/discipline.md == 0` on main after every land; every landed row has zero blank cells across the 16-gate table. **N==0 is RED.**
- **predict** none — a protocol, not a measurement.
- **kill** two branches carrying the same literal row number reach main ⇒ the protocol was bypassed; renumber before the next land.
- **rollback** docs revert. **blast** `proxima-tensor/docs/discipline.md`.
- **observe** placeholder count; monotonicity check `grep -oE '^## ROW [0-9]+' | sort -c`.
- **reprove** `grep -c "^## ROW <PH-" proxima-tensor/docs/discipline.md`
- **log-row title** `row numbers are assigned at land time from main, and here is the check that proves it`

---

# Phase 1 — The route becomes a value (hard precondition for every body swap)

*Conflict 1 resolved for B: the census lands BEFORE the Q4_K body swap. R13's `reduce-packed-row-blocked 225 / reduce-cooperative 385` split is produced by `classify_kind`'s substring match, and R12 ROW 263 measured that same instrument relabelling 9/601 → 225/385 when a body changed — swapping a body first would make every Phase-2 attribution unfalsifiable* [crit O-2].

### 1.1 — `KernelRoute` decided before emission; census per DISPATCH; `classify_kind` deleted `[A 3.1, B P1.2, crit R-5, e, H-3, V-5]`
- **tier** worker · **depends_on** 0.10
- **worktree/branch/target** `proxima-wt-risc12` · `gpu-risc/12-kernel-route-enum` · own target. **Not** `perf/route-census` or `perf/op-rule-census` — both are checked out elsewhere.
- **opens, in this order**
  1. `omega/src/msl.rs:673-697` — `emit`'s 4-kind + `Keep` match (the core that survives).
  2. `omega/src/msl.rs:731-775` — `kernel_cache_key`: pushes exactly **three** route characters, `'G'` if `tiled_gemm_block(..).is_some()`, else `'B'` if `packed_row_block(..).is_some()`, else `'S'` — and `'S'` also covers Elementwise, Scan, Iota, Constant, serial and cooperative reduce. The comment at `:751-758` says the ordering is load-bearing because `tiled_gemm_block` only returns `Some` where `packed_row_block` also would.
  3. `omega/src/msl.rs:3164-3187` — verified this session: `push_cooperative_reduce_body`'s **third** independent re-derivation of the same two gates (`if let Some(block) = tiled_gemm_block(...) { ...; return; }` then `if packed_row_block(...).is_some() { ...; return; }`).
  4. `omega/src/msl.rs:797` `kernel_dispatch_shape`, `:1517-1560` `grid_threads` (tiled → packed `output_total.div_ceil(4)*32` → cooperative `output_total*32` → serial `output_total`), `:824` `reduce_is_cooperative`, `:1235` `packed_row_block`, `:1450` `tiled_gemm_block`, `:1487` `diagnose_packed_row_block`.
  5. `omega/src/metal.rs:777-826` — `classify_kind` and its confession; `:835-854` `diagnose_kind`; **`:709-710`** where both are called inside the per-op timed path; and `omega/examples/real_forward_packed_probe.rs`, which `classify_kind`'s own doc ties it to (ROW 85) [crit H-3].
  6. `proxima-tensor/src/instrument.rs:809-828` (`WidthDeclineReason`, 8 variants), `:842` (`WIDTH_TILE_DECLINE: Mutex<BTreeMap<(u32, WidthDeclineReason), ..>>`), `:848-864` (`record_width_tile_decline`), `lib.rs:213-214` (module gate) — the shipped census pattern to mirror verbatim.
  7. `omega/src/metal.rs:2192-2194` and `:2244` — `kernel_cache_key` + `kernel_dispatch_shape` are called **per op per token inside `encode_op` even on a pipeline-cache hit**, and `ENCODE_DISPATCH_CALLS` is incremented at `:2244`. **This is the record site**: the census is recorded per DISPATCH here, not per emission — emission is cached (that is what `pipeline_hits`/`pipeline_misses` exist for), so a per-emission census could never sum to a per-dispatch count [crit V-5].
- **commands**
  ```
  # design
  # pub enum KernelRoute { Elementwise, Scan, Iota, Constant,
  #                        ReduceSerial, ReduceCooperative, ReduceRowBlockedPacked, ReduceTiledGemm }
  # pub fn route_of(&BoundOp, &PackedOperands) -> Result<(KernelRoute, RouteReason), EmitError>
  bash scripts/omega-gate.sh
  ```
  `emit`, `kernel_cache_key`, `kernel_dispatch_shape`/`grid_threads` and `push_cooperative_reduce_body` all consume `route_of`'s value. `classify_kind` is **deleted**; `real_forward_packed_probe.rs` is updated to read the census in the same commit; `diagnose_kind` folds into `RouteReason`.
  **The 9th label** `classify_kind` emits — `reduce-unclassified`, produced by `Err(_)` from `emit()` — is accounted for by `route_of` returning `Result`: an op whose route cannot be decided is an `EmitError`, not a census label, and the census-sum gate (1.2) is what makes an unrouted dispatch impossible rather than silently bucketed [crit e]. (`classify_kind`'s other 8: elementwise, iota, constant, scan, reduce-tiled-gemm, reduce-packed-row-blocked, reduce-cooperative, reduce-generic-scalar → `ReduceSerial`.)
- **expect** N = omega gate green (nonzero, under a named feature set) **plus ≥6 new tests**: (a) `route_of` is total over all 4 `BoundOpKind`s and both `Keep`s; (b) `route_of`'s answer equals the branch `emit` actually takes, asserted per route on the real bound openchat program; (c) **`kernel_cache_key` stability**: for every op in the real program the key produced through `route_of` is byte-identical to the key produced by main's three-character logic — the 8→3 collapse (`ReduceTiledGemm`→`'G'`, `ReduceRowBlockedPacked`→`'B'`, everything else→`'S'`) preserved exactly, G-before-B preserved [crit R-5, e]; (d) **golden emitted source**: the emitted MSL for every op in the real program hashes identically before and after, extending `emit_is_deterministic_byte_equal` (`msl.rs:4656`) to a cross-commit golden [crit B-1]; (e) the census sums to `encode_dispatch_calls`; (f) a deliberately unroutable op returns `EmitError`, not a label. **N==0 is RED.**
- **predict (nano → micro)** behaviour-neutral: the golden hashes match for **1196 minus 0.6's pruned count** ops; the census emits one `(NodeId, KernelRoute)` row per dispatch summing exactly to `encode_dispatch_calls`; `gpu_exec_ms` moves **< 0.7%** (inside R13's own `gpu_exec_ms` CoV).
- **kill** the census sum ≠ `encode_dispatch_calls`, or one golden hash drifts, or one `kernel_cache_key` differs ⇒ the route is still decided in more than one place; do **not** proceed to Phase 2 (every Phase-2 attribution would be unfalsifiable).
- **rollback** `git revert`. `route_of` is **not** feature-gated (a gated route means two routers); the recorder is `instrument`-gated at the module and every call site (`lib.rs:213-214` pattern). The firewall is (c) + (d), which live **here**, not in 7.2. Note for the rollback map: once 7.2 lands on top, reverting 1.1 is a multi-commit unwind [crit B-1].
- **blast** `omega/src/msl.rs` (three decision sites collapse to one), `omega/src/metal.rs` (`classify_kind` deleted — every caller, including `:709-710` and the example), `omega/examples/real_forward_packed_probe.rs`, `proxima-tensor/src/instrument.rs` (+1 enum, +1 recorder). `wgsl.rs` / `cuda.rs` untouched here; they get `route_of` in Phase 7.
- **observe** the new `(NodeId, KernelRoute)` census keyed `(node.0, route)`; `encode_dispatch_calls`; the per-route `gpu_exec` tick split; the golden hash per route.
- **reprove** G3 decode cell, then `grep -o 'route=[A-Za-z]*' runs/cell-12-*.txt | sort | uniq -c` and assert the sum equals `encode_dispatch_calls`.
- **log-row title** `the route becomes a value: eight kernel shapes, one enum, a per-dispatch census, and the substring instrument deleted`

### 1.2 — The census-sum and bound-plan-identity gate `[A 3.2, crit M-7]`
- **tier** hands · **depends_on** 1.1
- **worktree/branch/target** `proxima-wt-risc13` · `gpu-risc/13-census-gate` · own target
- **opens** `scripts/omega-gate.sh` (6 steps today; `[3/6]` asserts a nonzero nextest count, `[2/6]` builds `--all-targets --all-features`)
- **commands** add a step asserting (i) the route census sums to `encode_dispatch_calls`, and (ii) 0.10's cross-backend bound-plan fingerprint is equal.
- **expect** N: census rows sum **==** `encode_dispatch_calls` exactly. **N==0 is RED, and a mismatch is RED** — a dispatch with no route row means a path bypassed `route_of`.
- **predict (nano → micro)** on the openchat decode plan the census reports `ReduceRowBlockedPacked` ≈ **225** and `ReduceCooperative` ≈ **385** and `Elementwise` ≈ **547** and `Constant`/`Iota` = **37**/**2** — R13's per-op bucket table, now produced by a route value rather than by grepping emitted MSL.
- **kill** any op landing on `ReduceSerial` that R13 attributes to the Q4_K/Q5_K/Q6_K families — a quantized matvec on the serial path is a route bug worth more than any kernel tweak.
- **rollback** revert the gate step. **blast** `scripts/omega-gate.sh`.
- **observe** the census table vs `encode_dispatch_calls`; the plan fingerprint equality matrix.
- **reprove** `bash scripts/omega-gate.sh`
- **log-row title** `the census is a gate: a dispatch with no route row is RED`

### 1.3 — Every geometry constant into `omega-runtime.toml` `[A 5.1, B P1.3]`
- **tier** worker · **depends_on** 1.1
- **worktree/branch/target** `proxima-wt-risc14` · `gpu-risc/14-omega-geometry-config` · own target
- **opens** `omega/src/msl.rs:1017` (`PACKED_ROWS_PER_GROUP = 4`), `:1030` (`TILE_DIM = 8`), `:1046` (`TILED_GEMM_NSG = 4`), `:2526-2528` (packed loop `ib += SIMD_WIDTH/lanes_per_block`), `:3172`/`:3190`/`:3193` (the bare `SIMD_WIDTH` literals that pin the cooperative reduce width, verified this session); `omega/build.rs:16-66` (`require_nonzero`, `require_multiple_of_sixteen`, `require_divides_q4k_block`, `require_multiple_of_eight`), `:79` (`resolve_int`), `:85` (`cargo:rerun-if-env-changed`), `:105` (`emit_sizing_consts`); `omega/omega-runtime.toml` (today only `[tiled_gemm]` min_tokens/block_m/block_n/block_k, with the `OMEGA_<SECTION>_<KEY>` override contract in its own header); `omega/src/sized.rs:9,45` — `SIMD_WIDTH` **stays** a source const, its doc classifies it a hardware-family fact, not a policy knob.
- **commands** add `[packed_row_block] rows_per_group, lanes_per_block`, `[tile] dim`, `[tiled_gemm] nsg`, `[cooperative_reduce] max_threads, vector_width`; route each through `resolve_int` + a validator + `emit_sizing_consts`. `bash scripts/omega-gate.sh`.
- **expect** N ≥ 6 new tests: one per key asserting the source value equals the TOML value, plus one env-override test per key using `temp_env::with_vars`. Assert `grep -cE '^const (PACKED_ROWS_PER_GROUP|TILE_DIM|TILED_GEMM_NSG)' omega/src/msl.rs == 0`. **N==0 is RED.**
- **predict (nano → micro)** behaviour-neutral at default values: 1.1's golden emitted-source hashes are unchanged for every route, and `gpu_exec_ms` moves **< 0.7%** (inside R13's CoV).
- **kill** a cached build ignores an env override ⇒ a `rerun-if-env-changed` line is missing; fix before landing. If a value cannot become a build-time const without becoming a runtime read, §12's interaction note binds: it stays a source const with a one-line why at the site, recorded as a **named exception** in the row, not silently.
- **rollback** `git revert` build.rs + toml + msl const sites together; values identical by construction.
- **blast** `omega/build.rs`, `omega/omega-runtime.toml`, `omega/src/sized.rs`, `omega/src/msl.rs` const sites.
- **observe** the grep count == 0; the generated `OUT_DIR/omega_sized.rs` contents; 1.1's golden hashes.
- **reprove** `OMEGA_COOPERATIVE_REDUCE_MAX_THREADS=256 cargo build -p omega --features metal && grep MAX_THREADS $CARGO_TARGET_DIR/*/build/omega-*/out/omega_sized.rs`
- **log-row title** `every GPU geometry constant traces to omega-runtime.toml; SIMD_WIDTH stays a hardware fact and the row says why`

---

# Phase 2 — The Q4_K body (largest measured mass in R13)

*R13: `reduce-packed-row-blocked` = 225 ops / **44.450** of 61.082 diagnostic ms; ffn_up 10.862 + ffn_gate 10.843 + ffn_down 10.005 = 31.7 ms in three families. The mechanism exists twice, uncommitted, independently re-derived. ONE must be chosen.*

### 2.1 — Recover and rebase Q4_K body candidate A (`metal-q4k-mask-fma`) `[A 0.4, B P0.3]`
- **tier** worker · **depends_on** 1.2, 1.3, 0.5
- **worktree/branch/target** `proxima-wt-risc15` · `gpu-risc/15-q4k-mask-fma` · own target (base `gpu-risc/14-...`)
- **opens** `omega/src/msl.rs:190-311` (`Q4K_UNPACK_MSL`), `:2452-2530` (`push_packed_row_blocked_body`, `lanes_per_block` at `:2527`); incumbent form `/Users/brianbruggeman/repos/others/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5086-5193` — mask **without** shift (`& 0x000F/0x0F00/0x00F0/0xF000`, `:5147-5150` branch-free `kmask1/2/3`), 1/256 and 1/16 folded into the scale at combine (`:5171-5175`) (R8).
- **commands** `git apply --3way docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered/all-tracked.patch` (from 0.5, in git, not `/tmp`), then **strip to the one feature**: `proxima-wt-all` carries seven (R7). Declare `metal-q4k-mask-fma` in `omega/Cargo.toml [features]` default-off and forward it from `proxima-model-interop/Cargo.toml` as `metal-q4k-mask-fma = ["omega/metal-q4k-mask-fma"]` [G5]. One green commit, one feature.
- **expect** N = `cargo nextest run -p omega --features metal,cpu,instrument,metal-q4k-mask-fma` green with an explicit non-zero count recorded in the row; `omega/tests/q4k_real_checkpoint_parity.rs` runs ≥1 case with the feature **on** (a default-off feature is exactly the condition that hides tests). **N==0 is RED.**
- **predict (nano → micro)** `q4k_matvec_probe` Q4_K family time drops **≥15%** against the same probe at 0.3's tree (R3/M3 MEMORY, superseded by R13 for the whole-cell figure: −36% on ffn_gate/up, −17.2% `gpu_exec`; the conservative floor is the aggregate).
- **kill** parity vs `cpu::evaluate` exceeds the tolerance `q4k_real_checkpoint_parity.rs` already pins ⇒ §14, the body does not land at any speed.
- **rollback** `git reset --hard`; default-off so main's default build is untouched either way.
- **blast** `omega/src/msl.rs` MSL text under one cfg + two manifest lines. Zero callers outside omega. `kernel_cache_key` gains no new character (the feature does not change the route).
- **observe** the `(NodeId, KernelRoute::ReduceRowBlockedPacked)` census count from 1.1 — this is what proves the swapped body is the one that ran; `q4k_macs` execution witness (`bind.rs:2856-2860`).
- **reprove** `cargo nextest run -p omega --features metal,cpu,instrument,metal-q4k-mask-fma` + `cargo run --release -p omega --features metal --example q4k_matvec_probe`
- **log-row title** *(shared with 2.2/2.3)* `two independent re-derivations of ggml's Q4_K body: candidate A recovered and rebased`

### 2.2 — Recover Q4_K body candidate B (`metal-q4k-pair-dot`) `[A 0.4, B P0.4]`
- **tier** worker · **depends_on** 2.1
- **worktree/branch/target** `proxima-wt-risc16` · `gpu-risc/16-q4k-pair-dot` · own target (base `gpu-risc/15-...`, so both features exist at one commit)
- **opens** same as 2.1, plus `git log --oneline main..perf/q4k-independent-accumulators` to identify the paired-nibble commit (R12 ROW 257: GPU family 47.8 → 33.9 ms, −29%, parity 3.1e-6 vs f32 on real `blk.0.attn_q.weight`).
- **commands** `git cherry-pick <the q4k pair-nibble commit>` — **one** commit, not the 42. Do **not** bring `physical.rs`, `BoundOpKind::CachedAttention`, the bind.rs matcher, or the `libm` dep (adjudicated in 0.7). Declare `metal-q4k-pair-dot` default-off in `omega/Cargo.toml` and forward from `proxima-model-interop`.
- **expect** N as 2.1 under `metal-q4k-pair-dot`; both features present in the manifest at this commit and each buildable alone and neither in `default`. **N==0 is RED.**
- **predict (nano → micro)** `q4k_matvec_probe` Q4_K family time drops **≥20%** (R12 ROW 257's −29% is the anchor; the floor is conservative).
- **kill** same parity kill as 2.1.
- **rollback / blast / observe / reprove** as 2.1 with the other feature name.
- **log-row title** *(shared)* `candidate B recovered: the paired-nibble body, cherry-picked without the macro-op it shipped beside`

### 2.3 — Head-to-head at one commit; choose ONE body `[A 0.4/0.5, B P2.1, crit M-6, O-2]`
- **tier** judge · **depends_on** 2.2 · **measurer queue position 3**
- **worktree/branch/target** `proxima-wt-risc17` · `gpu-risc/17-q4k-body-selection` · own target (base `gpu-risc/16-...`: both bodies and the route census compiled in at one commit)
- **opens** `omega/src/msl.rs:190-311`, `:2452-2530`; incumbent `ggml-metal.metal:5086-5193`, dispatch geometry `<4,2,32>` at `ggml-metal.m:3330`, `:3215-3220` (R8)
- **commands** interleaved, never before-block/after-block:
  ```
  for i in 1 2 3 4 5; do
    PROXIMA_MAX_TOKENS=8 <G3 cell> --features std,metal,instrument,metal-q4k-mask-fma
    PROXIMA_MAX_TOKENS=8 <G3 cell> --features std,metal,instrument,metal-q4k-pair-dot
    PROXIMA_MAX_TOKENS=8 <G3 cell> --features std,metal,instrument           # the R13-shape control arm
  done
  ```
- **expect** N = 3 arms × 5 runs, each emitting `tokens_generated + stopped_by_eos` breakdown lines; parity suites green under **each** feature separately (`q4k_real_checkpoint_parity`, `q4k_matmul_layout`, `q4k_unpack`, `metal_parity`, `backend_parity`, `metal_real_forward`); max abs error vs f32 on real `blk.0.attn_q.weight` ≤ 1e-5 for both. **N==0 is RED under either arm.**
- **predict (micro → milli)** the surviving body takes `gpu_exec_ms` from R13's **56.93** to **34–42 ms/token**. Anchor: R12's feature-off control already carried the paired body and read `gpu_exec` 35.117 (CoV 2.00%) where R13 reads 56.93.
- **decision rule, pre-registered and fully executable** [crit M-6]:
  1. Any body failing parity is out.
  2. **Route-count pin**: the comparison is void unless both arms report the **same** `KernelRoute::ReduceRowBlockedPacked` census count (R13's 225). Unequal counts mean different op sets, not different bodies.
  3. If the arms' median `gpu_exec_ms` differ by more than **2× the pooled CoV** (R13 `gpu_exec_ms` CoV 0.7%; use the measured pooled value), the faster body wins.
  4. Otherwise decide by an **executable** incumbent-proximity metric, not by a reader's judgment: for each candidate body count (a) the number of arithmetic shift operations per 8 weights in the emitted MSL (`grep -c '>>' ` over the emitted kernel's Q4_K unpack region) and (b) whether the 1/16 and 1/256 folds appear at the scale-combine site rather than per element. The body with the **lower shift count** and the folds at combine wins — this is `ggml-metal.metal:5147-5175`'s form reduced to two integers, and §14 makes the incumbent's formulation the oracle. A tie on both integers goes to the body with fewer total emitted MSL lines in that region.
  5. The loser's feature is **deleted** in the landing commit — not left dormant. Its patch survives in git from 0.5, so the deletion is reversible [crit B-2].
- **kill** neither body clears −10% `gpu_exec_ms` beyond both arms' CoV bands ⇒ the −17.2% / −29% micro figures do not transfer to the real graph; decompose into inconsistency vs understanding-gap, stop the climb, and do not proceed to 2.4.
- **rollback** both features default-off; drop the branch.
- **blast** `omega/src/msl.rs`; two manifest entries.
- **observe** per-route `gpu_exec` ticks for `ReduceRowBlockedPacked` (1.1); the per-family split (R13: ffn_up 10.862 / ffn_gate 10.843 / ffn_down 10.005 / attn_q 5.172 / attn_output 3.846 / attn_v 1.523 / attn_k 1.462 / output.weight 0.737) with 0.1's corrected byte accounting; `q4k_macs`.
- **reprove** the interleaved loop above.
- **log-row title** *(shared)* `two bodies, one mechanism, one chosen: the adjudication, the pinned route counts, and the deleted loser`

### 2.4 — Promote the winning body to `default` `[A 0.5, B P2.1]`
- **tier** worker · **depends_on** 2.3
- **worktree/branch/target** `proxima-wt-risc18` · `gpu-risc/18-q4k-body-default` · own target
- **opens** `omega/Cargo.toml [features] default = ["std","metal","cpu"]`; `proxima-model-interop/Cargo.toml [features]`
- **commands** move the winner into `default` in one commit gated on the full parity suite; delete the loser's feature and body.
- **expect** N = `bash scripts/omega-gate.sh` green (all six steps, including `[1/6]` `--no-default-features --features alloc` and `[2/6]` `--all-targets --all-features`); `bash scripts/proxima-tensor-gate.sh` green. **N==0 is RED.**
- **predict (milli → bench)** whole-cell `step_wall_ms` falls from R13's **67.92** to **44–52**, and the ratio against R13's incumbent arm (**17.52 ms/token**) moves from **3.88x** to **2.5x–3.0x**. Anchor: R12's paired-body control at 51.571 wall / 2.95x, before any reduce or orchestration work.
- **kill** the bench move is less than half the milli-rung prediction ⇒ decompose. The inconsistency branch (GPU time fell, wall did not) **promotes** Phase 4 rather than killing it, since R13 puts 11.0 ms in orchestration; say so in the row.
- **rollback** demote out of `default`, one line; then `git revert`.
- **blast** `omega/Cargo.toml`, `omega/src/msl.rs`. Every downstream crate turning on `omega/metal` inherits it — check `proxima-model-interop/Cargo.toml`'s metal passthrough.
- **observe** `gpu_exec_ticks`, `step_wall_ms`, and **`encode_dispatch_calls` must be unchanged** at R13's 1196 minus 0.6's pruned count — a body change that moves the dispatch count means the route changed and the arms are not comparable.
- **reprove** `bash scripts/sealed-pass.sh`
- **log-row title** `the Q4_K body lands in default: R13's 44.450 ms of packed-row-blocked time, re-measured`

---

# Phase 3 — The non-matmul GPU bucket (R13: 16.6 ms)

### 3.1 — Wide cooperative reduce, width from config `[A 0.3, B P0.5/P3.1, crit O-5, V-3]`
- **tier** worker · **depends_on** 2.4, 1.3 · **measurer queue position 4**
- **worktree/branch/target** `proxima-wt-risc19` · `gpu-risc/19-wide-cooperative-reduce` · own target. **Not** `perf/gpu-dispatch-count` (checked out at `proxima-wt-gpudisp`).
- **opens** `omega/src/msl.rs:3188-3196` — verified this session: `long output_index = (long)gid / SIMD_WIDTH; if (output_index >= u.output_total) { return; } uint lane = gid % SIMD_WIDTHu;` — the pin; `msl.rs:1517-1560` `grid_threads` (cooperative arm = `output_total * 32`); `omega/src/sized.rs:45`; incumbent `ggml-metal.m:3797-3804` (nth doubles from 32 up to `min(ne00/4, maxTotalThreadsPerThreadgroup)`), `ggml-metal.metal:1679-1721` (float4 loads, `simd_sum` → threadgroup shmem → `simd_sum`; a 4096-wide row gets 1024 threads) (R8).
- **commands** **first, the mechanism re-read that Plan A skipped** [crit O-5]: before predicting from R3/M4 (MEMORY), run the per-op diagnostic (`bind.rs:3084`) on 2.4's tree and record the *current* `reduce-cooperative` route time — R13's 385 ops / 9.113 ms is the pre-Phase-2 figure and the body swap may have moved which ops route there. Then apply `docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered/gpudisp-tracked.patch` from 0.5, strip to one feature, and read the width from `[cooperative_reduce] max_threads` (1.3) rather than `SIMD_WIDTH`. Declare `metal-wide-cooperative-reduce` default-off in `omega/Cargo.toml` + forward from `proxima-model-interop`.
- **expect** N = omega gate green with the feature on; `omega/tests/metal_parity.rs` runs its full case set with the feature on (the count is recorded explicitly, not compared to `--all-features`); ≥3 new tests (narrow row, wide row, non-power-of-two row). **N==0 is RED.**
- **predict (micro → milli)** the `KernelRoute::ReduceCooperative` share of `gpu_exec_ms` falls **≥15%** against the figure this card just re-measured (R13's pre-Phase-2 anchor: 9.113 ms over 385 ops, per-op mode, **+7.3% inflation** noted).
- **kill** `metal_parity` or `backend_parity` regresses ⇒ a wider tree changes float summation order and §14 binds on the oracle, not on speed. Or: the cooperative share does not fall beyond the measured CoV ⇒ record the negative and do not promote.
- **rollback** feature default-off; `git revert`.
- **blast** `omega/src/msl.rs` cooperative body only — the tiled-GEMM and packed-row-block paths `return` before it (`msl.rs:3164-3187`, verified) and are untouched.
- **observe** per-route `KernelRoute::ReduceCooperative` count and tick share (1.1); `gpu_exec_ticks`; the per-op mode's inflation factor stated beside every per-family number [crit V-3].
- **reprove** `cargo nextest run -p omega --features metal,cpu,instrument,metal-wide-cooperative-reduce` + G3 cell
- **log-row title** `SIMD_WIDTH is a lane count, not a thread budget: the cooperative reduce gets its width from the sizing config`

### 3.2 — Elementwise bucket census before any elementwise work `[B P3.2]`
- **tier** hands · **depends_on** 3.1
- **worktree/branch/target** `proxima-wt-risc20` · `gpu-risc/20-elementwise-census` · own target
- **opens** `omega/src/msl.rs:2104-2177` (`render_elementwise`); R13's bucket: **547 elementwise ops / 7.350 ms** (13,437 ns/op) plus 37 constant / 2 iota at 0.161 / 0.008 ms — i.e. the degenerate control ops are already free.
- **commands** extend 1.1's census rows with `(NodeId, KernelRoute::Elementwise, extents, operand_count)` and rank by tick share.
- **expect** N ≥ 1 row per elementwise node; the count equals the `Elementwise` share of `encode_dispatch_calls` (R13: 547). **N==0 is RED.**
- **predict (nano → micro)** the top-5 elementwise nodes by tick share account for **≥50%** of the 7.350 ms bucket — i.e. the bucket is concentrated, not uniform.
- **kill** the bucket is uniform across >200 nodes ⇒ no single-node lever exists; the only remaining lever is *fewer nodes*, which is Phase 6, and this card closes with that pointer. (Standing reason not to reach for fusion: `grep -rln fuse ggml/src` is **empty** at `b25346221` — parity is reachable without a fusion engine, R8.)
- **rollback** `git revert`; instrument-gated only. **blast** instrument only.
- **observe** the census itself; `gpu_exec` ticks per elementwise node.
- **reprove** G3 cell + the per-route table.
- **log-row title** `is the 7.35 ms elementwise bucket concentrated or uniform, and what that decides`

### 3.3 — Re-seal after the GPU-side work `[A 2.3, B P3.1]`
- **tier** hands · **depends_on** 3.1 · **measurer queue position 5**
- **worktree/branch/target** `proxima-wt-risc21` · `gpu-risc/21-reseal-after-kernels` · own target
- **opens** `scripts/sealed-pass.sh` (0.4)
- **commands** `bash scripts/sealed-pass.sh` with the winning Q4_K body and (if 3.1 cleared) the wide reduce in `default`; 5 interleaved pairs; the `-fa 1` incumbent arm from 8.1 if it has landed.
- **expect** N = 5 ours-runs × (`tokens_generated + stopped_by_eos`) lines; both incumbent arms present. **N==0 is RED.**
- **predict (milli → bench)** `step_wall_ms` lands **44–52** and the ratio vs R13's incumbent arm (17.52) lands **2.5x–3.0x**; `gpu_exec_ms` lands **32–40**. Derivation: R13 `gpu_exec` 56.93 minus 2.3's measured Q4_K delta minus 3.1's measured cooperative delta; orchestration (R13: 11.0) is untouched and is Phase 4's premise.
- **kill** `step_wall_ms` does not fall below R13's 67.92 by more than both CoV bands ⇒ the two GPU-side wins are being eaten by CPU orchestration; record that, do not promote further, and go straight to Phase 4.
- **rollback** demote features out of `default`, one line each.
- **blast** docs; `omega/Cargo.toml [features] default`.
- **observe** `step_wall_ms`, `gpu_exec_ms`, `op_setup_ms`, `prepare_ms`, `block_upload_ms`, `encode_dispatch_calls`, `plan_hits`, per-route census, CoV per arm.
- **reprove** `bash scripts/sealed-pass.sh`
- **log-row title** `the board after the kernel bodies: what is left is the orchestration`

---

# Phase 4 — Plan stability and orchestration (R13: 11.0 ms; root cause proven at `generate.rs:966`)

### 4.1 — `cached_len` as a plan-stable capacity bucket, with the cache-tail mask it actually needs `[A 2.1, B P4.1, crit c, R-1, B-3]`
- **tier** worker · **depends_on** 3.3, 0.2 · **measurer queue position 6**
- **worktree/branch/target** `proxima-wt-risc22` · `gpu-risc/22-kv-capacity-bucket` · own target
- **opens, in this order**
  1. `proxima-model-interop/src/generate.rs:958-975` — `resolve_plan`; key `(symbols[0], symbols[1])` at `:966`; `self.plans.clear()` at `:973`.
  2. `proxima-tensor/src/spec.rs:6216-6245` — `cached_len` as `Extent::Symbolic(1)` on all three KV input leaves (`:6220`, `:6230`, `:6240`).
  3. `proxima-tensor/src/spec.rs:823-845` `causal_mask` — **verified this session**: both `Op::Iota` are `Extent::Symbolic(0)`, combined by `ScalarOp::Greater` as `("t->st","s->st")` with `scalar_constant(f32::NEG_INFINITY)`; consumed as `(is_future, "sw->swug")` at `:2610`.
  4. `proxima-tensor/src/spec.rs:2303-2319` — the doc that names the premise: `is_future` is sized `[s,w]` "since `w` and `s` share symbol 0's extent … so the cached block never needs masking at all", and "the masking-only-within-`s,w` asymmetry is what makes this correct **without a `cached_len` scalar**".
  5. `proxima-tensor/proxima-tensor-runtime.toml` (the CPU sizing config with parallel/cohort/quantize/transpose/neon/rope/staged_batch sections — the pattern `[kv_cache] capacity_bucket` mirrors).
  6. `proxima-model-interop/src/bind.rs:3051-3058` — the `plan_hits == 0` assertion **this card changes in the same commit** [crit R-1].
- **the change, stated honestly** — this is **NOT** "zero IR change" [crit c]. Bucketing makes the cached extent `bucket ≥ cached_len`, so the cached block **does** now need masking, over **symbol 1**, against a runtime `cached_len` the graph deliberately does not carry. The card therefore adds, in the graph: **one `Op::Iota { extent: Extent::Symbolic(1) }`**, **one `Op::Input` scalar leaf carrying `cached_len`**, and a `Greater` + `Select` against `NEG_INFINITY` — exactly the `causal_mask` construction at `spec.rs:823-845` reused over the other symbol. **Zero new `Op` variants, zero new `BoundOpKind` variants** (0.9's exhaustive matches prove it mechanically); it is a graph change, not a driver fix, and the row leads with that. `capacity_bucket` traces to `proxima-tensor-runtime.toml` (§12), default 256, matching the incumbent (n_kv padded to a 256 multiple and masked, R11/M6′). `spec.rs:2314-2319`'s doc is corrected in the same commit.
- **commands** declare `kv-capacity-bucket` in **`proxima-tensor/Cargo.toml`**, forward as `proxima-model-interop/kv-capacity-bucket = ["proxima-tensor/kv-capacity-bucket"]` and `omega/kv-capacity-bucket = ["proxima-tensor/kv-capacity-bucket"]` [crit M-4]. In the same commit set `expected_plan_hits()` (0.2) to the bucketed value. Run the CPU oracle **first**: `cargo nextest run -p proxima-model-interop --release --features std --lib --run-ignored all -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)'`.
- **expect** N: `generated.0[0] == 2651` and `generated.1 == "known"` (`bind.rs:2797-2803`) still hold; proxima-tensor gate green; **≥6 new tests**: (a) bucketed and unbucketed decode produce identical token ids over the full budget; (b) `plan_hits == tokens − 1` at bucket 256 and budget 8 (one miss, the first); (c) the tail mask covers exactly `bucket − actual` positions; (d) bucket boundary (`cached_len == bucket`), (e) bucket+1 forces exactly one new plan; (f) **feature-OFF program identity**: the bound-plan fingerprint from 0.10 with the feature off is byte-identical to main's [crit B-3] — the "feature off ⇒ bit-identical graph" claim is gated by a hash, not asserted. **N==0 is RED.**
- **predict (nano → milli)** `plan_hits` goes from R13's **0** to `tokens_generated + stopped_by_eos − 1` (at `PROXIMA_MAX_TOKENS=8`, bucket 256: **7 of 8**); `prepare_ms` falls from R13's **1.97** toward **< 0.3** on hit tokens; `block_upload_ms` falls from R13's **2.0** because `mark_resident` (`metal.rs:350-362`) stops re-running per token.
- **kill** token ids drift at any bucket size ⇒ the tail mask is wrong; revert immediately (§14 — a decode loop that generates different text is not faster, it is broken). Second kill: `plan_hits` rises but `prepare_ms` does not fall beyond CoV ⇒ the cost was never in `plan_named`; decompose (inconsistency vs understanding-gap) and re-instrument before touching 4.2.
- **rollback** `git revert`; feature default-off, and test (f) is what makes "off is main" checkable rather than asserted. Note the bucket value 1 is the identity only if the bucket is a **runtime** read — the row states explicitly which it is (build-time const from the sizing config ⇒ rollback needs a rebuild).
- **blast** `proxima-tensor/src/spec.rs` (KV leaf extents + the new mask composition — **every** model spec with a KV cache: `append_mistral_cached_layer` at `:2336-2865` and the qwen3.5 hybrid attention/ssm path from `0c3bd4f`; test both), `proxima-model-interop/src/generate.rs` (plan key, `LayerCache` sizing `:621-654`), `proxima-model-interop/src/bind.rs` (the assertion). **CPU and Metal both**, because the graph changes — the CPU oracle runs first.
- **observe** `plan_hits`, `plan_misses`, `plan_cache_len` (`generate.rs:1732`, `:1764`), `prepare_ms`, `PREPARE_CALLS`/`PREPARE_TICKS`, `RESIDENT_BUFFER_REUSES` (`metal.rs:1983`), and the new symbol-1 mask node in the census.
- **reprove** G3 cell; `grep -o 'plan_hits=[0-9]*' runs/cell-22-*.txt`
- **log-row title** `plan_hits=0 was the shape symbol, not the cache: cached_len becomes a capacity bucket, and the cache tail gets the symbol-1 mask the design was built to avoid`

### 4.2 — Preallocate output and uniform buffers once per plan-stable program `[A 2.2, B P4.2]`
- **tier** worker · **depends_on** 4.1 · **measurer queue position 7**
- **worktree/branch/target** `proxima-wt-risc23` · `gpu-risc/23-metal-plan-buffer-pool` · own target
- **opens** `omega/src/metal.rs:2179-2252` `encode_op` — verified this session: `:2210` `allocate_buffer(device, bound_output_len(bound), bound.dtype)` and `:2211` `upload_uniforms(device, &pack_uniforms(bound))`, both **per op per token** = R13's 1196 `newBufferWithLength` + 1196 uniform uploads = the **3.9 ms** `op_setup` (R13; MEMORY said 4.394, superseded); `:2249` `device_buffers.insert(bound.node, (output, 0))`; **`:2068-2075` `UNIFORM_BUFFER_REUSES` already exists with a live cache path** — read its current hit rate and report it **before** assuming uniforms are the cost [B, verified]; `:449-568` `execute_plan`'s retirement logic.
- **commands** recover `docs/bench-campaigns/.../recovered/all-tracked.patch`'s `metal-buffer-pool` hunk (0.5); rewrite against 4.1's plan-stable `Plan`. Buffers are **caller-owned, fixed-capacity**, allocated at plan time, bound (not allocated) in `encode_op` — strict O(1) per op, not a growable pool. Declare `metal-buffer-pool` default-off in `omega/Cargo.toml`, forward from `proxima-model-interop`.
- **expect** N = omega gate green; ≥4 new tests: (a) pooled and unpooled `execute_plan_named` produce identical outputs on the real forward; (b) the pool's live buffer count is bounded in steady state; (c) an extent change forces a documented realloc; (d) uniform contents change per step while the buffer does not. **N==0 is RED.**
- **predict (milli → bench)** `op_setup_ms` falls from R13's **3.9** to **< 1.0**; `newBufferWithLength` calls/token fall from ~1196 to ~0 in steady state; `device_allocated_bytes` flattens across steps.
- **kill** `op_setup_ms` falls but `step_wall_ms` does not, beyond both CoV bands ⇒ the orchestration slice overlaps GPU execution and removing it does not shorten the token. That is the same shape as R12's dispatch-count null; record it and **stop Phase 4** rather than continuing to 4.3's promotion.
- **rollback** feature default-off; `git revert`.
- **blast** `omega/src/metal.rs` `encode_op` / `execute_plan` / `Plan` — `device_buffers` lifetime and retirement are the hazard (a pooled buffer must not be retired mid-plan); this is the card most likely to produce a use-after-retire, and the parity tests are the gate. No `proxima-tensor` change; no other backend.
- **observe** `OP_SETUP_CALLS`/`OP_SETUP_TICKS`, `UNIFORM_BUFFER_REUSES` (`metal.rs:2069`) before and after, a new pooled-output reuse counter mirroring it, `device_allocated_bytes`, `phys_footprint_bytes`.
- **reprove** G3 cell with `--features std,metal,instrument,metal-buffer-pool`; grep `op_setup_ms=`
- **log-row title** `1196 buffer allocations and 1196 uniform uploads per token become zero, and here is what UNIFORM_BUFFER_REUSES was already doing`

### 4.3 — Re-seal: the plan's one board-level prediction `[A 2.3, B P9.2]`
- **tier** hands · **depends_on** 4.2 · **measurer queue position 8**
- **worktree/branch/target** `proxima-wt-risc24` · `gpu-risc/24-reseal-after-orchestration` · own target
- **commands** `bash scripts/sealed-pass.sh`, 5 interleaved pairs, both incumbent arms.
- **expect** N = 5 ours-runs × (`tokens_generated + stopped_by_eos`) lines; llama arm A and arm B present. **N==0 is RED.**
- **predict (milli → bench)** — **the plan's single board-level prediction, made once, here.** `step_wall_ms` lands **33–43 ms/token** and the ratio against R13's incumbent arm (**17.52 ms/token**) lands **1.9x–2.5x**. Full derivation, every term cited: R13 `step_wall_ms` **67.92** = `gpu_exec_ms` **56.93** + **11.0** non-GPU. 2.3 predicts `gpu_exec` **34–42** (anchored on R12's paired-body control 35.117). 4.1 + 4.2 predict the 11.0 slice (R13: op_setup 3.9 + block_upload 2.0 + prepare 1.97 + emit 0.81 + encode_dispatch 0.47 + readback 0.22 + pipeline_lookup 0.04 + ~1.6 residual) falls to **≤ 4** (the residual, encode_dispatch and readback are untouched by these two cards). 38 + 4 ≈ 42 at the pessimistic end, 34 + 3 ≈ 37 at the optimistic; band widened to 33–43 for CoV. **Phases 5 and 6 are NOT in this prediction** — R12's null result gives no basis to predict a wall movement from a dispatch collapse.
- **kill** the ratio does not fall below **3.0x** ⇒ R13's decomposition is wrong somewhere, and the row names **which bucket did not move**, by counter, before any further card is scheduled.
- **rollback** n/a (measurement); the underlying features demote one line each.
- **blast** docs.
- **observe** every counter in the board cell + the per-route census + fraction-of-ceiling against 8.2 once it exists.
- **reprove** `bash scripts/sealed-pass.sh`
- **log-row title** `the board after the body and the plan: the one prediction this plan made, against the number it made it from`

---

# Phase 5 — Write placement (correctness and one-RISC conformance; OFF the critical path)

*Conflict 2 resolved for B's ordering: the scatter expression is written FIRST (§1 forbids minting before the expression is written), then the affine form is chosen on a measured nano number — Plan A's affine-offset design survives as the expected winner but is no longer assumed. Conflict 3 resolved for B: this phase is not on the critical path to 4.3's board prediction.*

### 5.1 — Write the expression before changing anything `[B P5.1]`
- **tier** worker · **depends_on** 4.3
- **worktree/branch/target** `proxima-wt-risc25` · `gpu-risc/25-kv-placement-expression` · own target
- **opens** `proxima-tensor/src/map.rs:110-131` (the `Computed` write-direction convention), **`:172`** `IndexMap::scatter` constructor (verified this session, doc at `:168-173` names the `destination_extent` at `gathered_dim` convention and `pure_projection_axes`), `:209-222` `scatter_extent`; `proxima-tensor/src/bind.rs:955-996` `bind_reduce`'s scatter arm, `:1011-1035` `build_scatter_out_layout`, `:243-256` `out_scatter` on `BoundOpKind::Reduce`; `proxima-tensor/src/cpu.rs:6892-6927` `run_reduce_scatter`; `proxima-tensor/src/shape.rs:495-540` `scatter_output_shape`.
- **commands** one new test in `proxima-tensor` building a KV-append as `Reduce { out_map: IndexMap::scatter(indices = Elementwise(Add,[Iota(w), Constant(cached_len)]), destination_extent = capacity, …) }` and running it through `cpu::evaluate`. **No production code.**
- **expect** N ≥ 3 tests: (a) the scatter expression evaluates on CPU and places `w` rows at offset `cached_len` in a `capacity`-sized buffer; (b) two successive appends round-trip; (c) `omega::msl::emit` rejects it with exactly `EmitError::ScatterNotSupported` (`msl.rs:932`, verified) — a **red test asserting the known hole**, so 5.3 has something to turn green. **N==0 is RED.**
- **predict** none — this is a compile-and-evaluate proof, not a measurement; its output decides 5.2's arms.
- **kill** the expression does not compile or does not evaluate ⇒ the scatter form cannot express placement; **only then** does the affine form (5.4) become the sole route, and the row records exactly which clause blocked it.
- **rollback** `git revert`; one test file, zero production code.
- **blast** one test file.
- **observe** CPU buffer contents vs a hand-computed placement; the emitter's error variant.
- **reprove** `cargo nextest run -p proxima-tensor --features std -E 'test(kv_placement)'`
- **log-row title** `the expression before the type: KV placement, written as a scatter, on main, today`

### 5.2 — Measure the two placement forms head to head `[B P5.2, A 4.1]`
- **tier** judge · **depends_on** 5.1 · **measurer queue position 9**
- **worktree/branch/target** `proxima-wt-risc26` · `gpu-risc/26-placement-form-selection` · own target
- **opens** form **S** (scatter): index tensor + per-element index fetch; costs an extra `Iota` + `Constant` + `Elementwise` node per KV write per layer. Form **A** (affine): the out_map axis carries `offset` and a declared destination extent; `layout_of` (`bind.rs:1594-1605`) **already** folds `axis.offset * stride` into `Layout.base` and `msl.rs:2733` **already** emits `long out_offset = u.out_base;` (both verified this session) — zero indirection, zero extra nodes.
- **commands** `bind.rs:3084`'s per-op timer (`execute_plan_op_timed`, `metal.rs:654-761`) on both forms at identical extents, 5 interleaved runs; per-op mode's **+7.3%** inflation (R13) stated beside every number.
- **expect** N = 2 forms × 5 runs × per-write µs; both forms produce identical CPU output. **N==0 is RED.**
- **predict (nano → micro)** form A costs **0** extra dispatches and form S costs **3 extra nodes per KV write per layer** (= 96 extra dispatches at 32 layers × K/V, against R13's 1196). Given R12's dispatch-count null, dispatch count is explicitly **not** the deciding metric — the deciding metric is per-write µs.
- **selection rule, pre-registered** form **A** unless its `project_output_shape` change (5.4) fails a bounds test that form S passes. Form **S** is implemented **regardless**, because 5.7 (on-device argmax) needs GPU scatter independently — so 5.3 is not conditional on this choice.
- **kill** both forms exceed the unplaced baseline's per-write time ⇒ placement is not a performance lever and Phase 5 reduces to 5.3 (coverage) alone, which still lands on one-RISC grounds.
- **rollback** measurement + throwaway fixtures; `git revert`.
- **blast** test fixtures only.
- **observe** per-write µs both forms; dispatch delta; the route census (a placed write must not change any route).
- **reprove** the per-op harness on both fixtures.
- **log-row title** `two placement forms, one timer: what an index fetch costs against an out_base`

### 5.3 — Scatter coverage in all three GPU emitters and the wgpu driver `[B P5.3, crit H-6]`
- **tier** worker · **depends_on** 5.1
- **worktree/branch/target** `proxima-wt-risc27` · `gpu-risc/27-omega-scatter-coverage` · own target
- **opens** `omega/src/msl.rs:923-936` (verified: `if out_scatter.is_some() { return Err(EmitError::ScatterNotSupported { node }) }` at `:932`), `omega/src/wgsl.rs:355-370` (reject at `:363`), `omega/src/cuda.rs:232-245` (reject at `:240`); reference implementation `proxima-tensor/src/cpu.rs:6892-6927`; the field's own doc `proxima-tensor/src/bind.rs:245-256`; **`omega/src/wgpu_driver.rs` (872 lines; `execute_plan` at `:599`, `execute_plan_named` at `:866`)** — named here explicitly because `omega-gate.sh [2/6]` builds `--all-targets --all-features`, so a wgpu compile break lands in the gate for every subsequent card [crit H-6]; `omega/src/metal.rs:2214-2216` (fault buffer) and `:2260` (`check_gather_fault`).
- **colliding writes** `map.rs:110-115` records that the CPU interpreter runs the reduce loop strictly sequentially, so "a scatter never needs atomics". A GPU emitter has no such guarantee. KV-append writes **disjoint** ranges by construction. The emitter therefore either (a) proves disjointness from the index expression, or (b) emits an atomic body for `Add`/`Maximum`/`Minimum` (associative, `op.rs:112-117`) and rejects the rest. Decide by making it a **route value**: `KernelRoute::ScatterDisjoint` vs `KernelRoute::ScatterAtomic` — two more variants on the 1.1 enum, **not** a new `BoundOpKind` (0.9's exhaustive match still holds).
- **commands** land **msl first, green; then wgsl; then cuda** — three commits, each a green bisect point. `bash scripts/omega-gate.sh` after each.
- **expect** N: 5.1(c)'s red test turns green; omega gate green; ≥6 new tests (per backend: disjoint scatter parity vs CPU, colliding scatter parity vs CPU); `omega/tests/wgpu_parity.rs` and the CUDA emit tests each gain ≥1; `grep -c ScatterNotSupported omega/src/{msl,wgsl,cuda}.rs == 0` (excluding the error-type definition if retained for a genuinely unreachable case, with a one-line why). **N==0 is RED.**
- **predict (nano → micro)** a disjoint scatter dispatch costs within **±20%** of the equivalent affine-offset dispatch at the same extents (5.2's number); the fault-buffer machinery extends to the write side with **no additional dispatch**.
- **kill** parity fails on colliding writes on any backend ⇒ restrict to `ScatterDisjoint`, emit `EmitError` for the colliding case, and record the restriction as a **named coverage gap with its reason** — not as "unsupported".
- **rollback** `git revert` per backend, 3 independent commits; the emitters return to rejecting, which is main's behaviour.
- **blast** all three emitters + `wgpu_driver.rs` + the Metal fault-buffer path. Widest emitter blast radius in the plan.
- **observe** `(NodeId, KernelRoute::ScatterDisjoint|ScatterAtomic)` census counts; gather/scatter fault counters.
- **reprove** `cargo nextest run -p omega --features metal,cpu,instrument -E 'test(scatter)'` and `cargo nextest run -p omega --features wgpu-backend -E 'test(scatter)'`
- **log-row title** `the IR had a field two of five executors honoured: scatter reaches every backend`

### 5.4 — `project_output_shape` accepts a declared destination extent (form A) `[A 4.1, B P5.4, crit R14]`
- **tier** worker · **depends_on** 5.2 (conditional on its selection rule)
- **worktree/branch/target** `proxima-wt-risc28` · `gpu-risc/28-tensor-write-placement-offset` · own target
- **opens** `proxima-tensor/src/shape.rs:469-485` — **the line placement changes** (R14, verified): `[term] if term.coeff == 1 => Ok(iter_extents[..])`, else `NotLowerable { reason: "reduce output maps must be pure projections in v1" }`; `:441-467` `bounds_check`, which already handles `axis.offset` for READ-side maps and must gain the write-side check; `:495-540` `scatter_output_shape` (the sibling convention to mirror); `proxima-tensor/src/bind.rs:1594-1605` `layout_of` (already correct); `map.rs:12` (slice = non-zero offset on the read side); docs corrected in the same commit: `spec.rs:2303-2319` ("`Reduce::out_map` must stay a pure projection … so nothing upstream of a reduce can splice two tensors into one axis") and `:2616-2617`.
- **commands** declare `tensor-write-offset` in **`proxima-tensor/Cargo.toml`**, forward from **both** `omega/Cargo.toml` and `proxima-model-interop/Cargo.toml` [crit M-4]. Add `IndexMap::placed(...)` as a **constructor**, not a variant, exactly as `map.rs:172` does for `scatter`, so an existing pure projection whose offset was previously ignored cannot silently change meaning. `bash scripts/proxima-tensor-gate.sh`.
- **expect** N ≥ 6 new tests: (a) zero offset is bit-identical to today; (b) a placed write lands at the declared offset; (c) an offset exceeding the declared extent is rejected by `bounds_check` with `IndexOutOfBounds`; (d) two producers writing disjoint ranges of one destination compose to the same bytes as a concatenation of two separately-evaluated reduces — **the worked example that IS the spec for this phase**; (e) overlapping ranges are rejected at bind; (f) CPU and Metal agree bit-exactly on a placed write. Plus 0.9's variant-count assertions still compile with no wildcard arm. **N==0 is RED.**
- **predict (nano → micro)** a placed reduce emits **one** dispatch with a non-zero `u.out_base` and an identical thread count to the unplaced form; per-op time within **±5%** of the unplaced form (5.2's timer); the 1.1 route census is **unchanged** — a write offset must not change which route an op takes.
- **kill** overlapping ranges are not rejected at bind ⇒ a silent data race on GPU; do not proceed. Or: an existing `out_map` becomes ambiguous ⇒ the convention must be explicit at the constructor; fix before landing.
- **rollback** `git revert`. **Land 5.4 and its first caller (5.6) as separate commits** so the caller can revert alone.
- **blast** `proxima-tensor/src/{shape.rs,map.rs,bind.rs,cpu.rs}` + `omega/src/msl.rs` render path + `spec.rs` docs — the bounds check changes for **every existing `Reduce`**. Run the full proxima-tensor **and** proxima-autograd suites (the adjoint path reads `as_gather_from_output`, `map.rs:238`).
- **observe** the census count of reduces with `out_layout.base != 0`; route census unchanged; `bounds_check` rejection counts.
- **reprove** `bash scripts/proxima-tensor-gate.sh` and `cargo nextest run -p omega --features metal,cpu,tensor-write-offset -E 'test(write_offset)'`
- **log-row title** `out_map stops being a pure projection: a declared destination extent is concat, and it costs no new Op`

### 5.5 — The over-allocated-destination contract: pipe and relocation questions answered in-line `[crit R-3, d, H-1, H-2]`
- **tier** judge+worker · **depends_on** 5.4
- **worktree/branch/target** `proxima-wt-risc29` · `gpu-risc/29-declared-capacity-block` · own target
- **opens** `omega/src/metal.rs:991-1000` — verified this session: `for (node, block) in block_nodes.iter().zip(blocks.iter()) { let expected = element_count(shapes.of(*node)); let found = block_element_count(block)?; if found != expected { return Err(InputSizeMismatch) } }` — this loop runs over **every named input, weights included**, so any relaxation is in the weight path by construction [crit R-3]; **`proxima-tensor/src/cpu.rs:346-356`** — verified: the twin strict check `if data.len() != expected` over `&[&[f32]]`, which is the path the §14 oracle (`bind.rs:2797-2803`, token 2651/"known") runs through and which a `QuantizedBlock` field **cannot reach** [crit H-1]; `proxima-tensor/src/cpu.rs:3084-3110` (`QuantizedBlock`: Float32 | Q4K | Q5K | Q6K | Q8_0 | Q4_0 | F16/BF16 — public proxima-tensor surface); `proxima-tensor/src/align.rs:38-41,42-46,69` (`AlignedBuffer::new(min_elements, page_size)` rounds to `next_multiple_of(page_size).max(page_size)`; its doc says the caller "must size its tensor input to `buffer.len()`, not the value it originally asked for", and that `page_size` must come from e.g. `omega::metal::page_size`) [crit H-2].
- **the two questions, answered here rather than skipped** [crit d]:
  - **Pipe question**: is an over-allocated destination a stage in a dataflow? No — it is a *declaration about one input's extent*, consumed at validation, with no upstream/downstream, no backpressure, no cancellation. It is data, not a pipe.
  - **Relocation question**: write the call site both ways. With a new field on `QuantizedBlock`, `named_blocks` (`generate.rs:642-653`) constructs `QuantizedBlock::Float32 { data, declared_capacity: Some(cap) }` and both checks read `block.declared_capacity()`. Without it, `named_blocks` is unchanged and the checks read a **separate `&[DeclaredCapacity]` argument** threaded alongside `block_nodes`. The call sites are **not** identical lines — the second form changes two function signatures and both call sites — so this is **not** a relocation and the field is earned. **But** the field cannot reach `cpu.rs:346-356`, whose arm takes `&[&[f32]]` and not `QuantizedBlock` at all [crit H-1]. **Resolution**: the declaration is carried as a **parallel `&[Option<u64>] declared_capacity` slice** passed beside `block_nodes` into **both** `omega::metal`'s validation loop (`metal.rs:991-1000`) and `proxima_tensor::cpu`'s (`cpu.rs:346-356`) — one shape, both executors, **no new type in either crate**, and no `QuantizedBlock` change at all. That is the form the two questions select.
- **commands** thread the slice; both loops become `if found != expected && Some(found as u64) != declared_capacity[i] { Err(InputSizeMismatch) }`. `bash scripts/proxima-tensor-gate.sh`; `bash scripts/omega-gate.sh`.
- **expect** N ≥ 5 tests: (a) an over-allocated **KV** block with a declaration is accepted on Metal; (b) the same on CPU; (c) an **under**-allocated block is still rejected on both; (d) **an over-allocated WEIGHT block with NO declaration is still rejected on both** — the sad-path test that matters, which Plan A's "under-allocated still rejected" did not exercise [crit R-3, d]; (e) a declaration that does not match the actual buffer length is rejected. **N==0 is RED.**
- **predict (nano → micro)** the §14 oracle still returns 2651/"known" through `cpu::evaluate` with a declared over-allocated KV input, and the Metal arm's `InputSizeMismatch` count for weights is **0** across a full decode.
- **kill** any weight block passes validation without a declaration ⇒ the firewall leaked; revert.
- **rollback** `git revert`; the strict checks return.
- **blast** `omega/src/metal.rs` validation loop, `proxima-tensor/src/cpu.rs` validation loop, the two call sites in `proxima-model-interop`. Public signature change in both crates — sequenced before its only caller (5.6).
- **observe** `InputSizeMismatch` occurrences by node name; declared-capacity acceptance count.
- **reprove** `cargo nextest run -p proxima-tensor --features std -E 'test(declared_capacity)'` and the omega twin
- **log-row title** `an over-allocated destination is a declaration, not a type: the pipe and relocation questions, run, and where they landed`

### 5.6 — KV as a caller-owned persistent device buffer `[A 4.2, B P5.5, crit R-4, H-2]`
- **tier** worker · **depends_on** 5.4, 5.5, 4.1 · **measurer queue position 10**
- **worktree/branch/target** `proxima-wt-risc30` · `gpu-risc/30-kv-device-resident-buffer` · own target. **Not** `perf/kv-device-resident` (checked out at `proxima-wt-drive`) [crit g].
- **opens, in this order**
  1. `omega/src/metal.rs:2249` `device_buffers.insert(bound.node, (output, 0))` over `BTreeMap<NodeId,(MetalBuffer,usize)>` — **the whole mechanism is an insert, not a type**; `:2210` `allocate_buffer` (the call skipped for a placed node).
  2. `proxima-model-interop/src/generate.rs:621-654` — `LayerCache { k_even, k_odd, v: Vec<f32> }`, `append` = 3× `extend_from_slice` (`:636-640`), `named_blocks` handing the whole `Vec` as `QuantizedBlock::Float32` (`:642-653`).
  3. `omega/src/metal.rs:1848-1892` `NOCOPY_BUFFERS: BTreeMap<(usize,usize), MetalBuffer>` keyed `(pointer, byte_length)`; `:1879` `upload_block_no_copy`; `:1903` `upload_block_no_copy_uncached` (where `23e2e5e` routes non-resident blocks, so KV creates a **fresh** no-copy wrapper every token); `:1914` `create_no_copy_buffer`; `:1616` `is_page_aligned`.
  4. `omega/src/metal.rs:350-362` `mark_resident` (classifies by **name**).
  5. `proxima-tensor/src/align.rs:42-46,69` — `AlignedBuffer`, **zero production callers**; this is its first (§1: extend the existing primitive). **Two named consequences** [crit H-2]: (i) its rounding makes the over-allocation *structural*, which is exactly what 5.5's declaration covers; (ii) `proxima-model-interop`'s `metal` feature is optional (`metal = ["dep:omega","std"]`), so `LayerCache` must obtain a page size on a **non-Metal** build — the card supplies it from the sizing config with the Metal value as the override, and a test builds `proxima-model-interop --features std` (no metal) to prove it.
  6. Incumbent: `llama-kv-cache-unified.cpp:74-118` (allocated once), `:749-788` (`ggml_cpy` into `ggml_view_1d` at a byte offset), V stored transposed since `v_trans = !flash_attn` (R8).
- **commands** declare `kv-device-resident` default-off in `proxima-model-interop/Cargo.toml` forwarding `omega/kv-device-resident`. **Assert the capacity BEFORE allocating** — R3/M11 found a 34 GB allocation from a `context_length` default; a detector that fires after the allocation poisons every concurrent cell on the box, so the guard is a pre-allocation `debug_assert` + a hard runtime check with the computed size in the error [crit R-4]. Hold the G6 measurer lock for the measuring half.
- **expect** N ≥ 6 tests: (a) full-budget decode produces identical text with and without device-resident KV; (b) `block_upload_bytes` for KV blocks is 0 after the first token; (c) pointer stability across appends; (d) the no-copy cache hits from step 2; (e) the KV allocation size is asserted **before** allocation and is < 8 GB; (f) `proxima-model-interop --features std` (no metal) builds and gets a page size. Plus the §14 oracle 2651/"known". **N==0 is RED.**
- **predict (milli → bench)** `block_upload_ms` falls from R13's **2.0** toward the weight-only residual (R13 notes it is 0.4 on step 2 and 1.7–3.5 otherwise — the "otherwise" is the KV term); `kv_cache_upload_bytes` (`token_breakdown`, **`generate.rs:1657`** — not `token_breakdown_metal`) [crit H-5] falls to **0** after step 1; `nocopy_reuses` (`token_breakdown_metal`, `generate.rs:1721-1750`) rises to 3×32 = **96**/token.
- **kill** generated text drifts ⇒ §14, revert. Or: `block_upload_ms` falls but `step_wall_ms` does not beyond both CoV bands ⇒ record it as inside the noise band and stop; R2's "~3.3%" is MEMORY with **no counterpart term in R13's decomposition** and may not be claimed as a win it did not produce [crit a].
- **rollback** `git revert` 5.6 alone (5.4 and 5.5 stay; each has its own tests).
- **blast** the Metal driver's buffer lifetime, `LayerCache`, `named_blocks`, `mark_resident`, `align.rs`. Widest correctness surface after 5.3.
- **observe** `BLOCK_UPLOAD_BYTES`, `NOCOPY_BUFFER_REUSES` vs `NOCOPY_BUFFER_UPLOADS` (`metal.rs:1983`, `:1985`), `COPYING_BUFFER_UPLOADS`, `MAPPING_OFFSET_UPLOADS` (`:1779`), nocopy cache length (`:1858`), `device_allocated_bytes`, `phys_footprint_bytes`.
- **reprove** G3 cell; assert `block_upload_bytes` is flat across steps 2..N.
- **log-row title** `AlignedBuffer gets its first production caller and the KV cache stops crossing the bus`

### 5.7 — On-device argmax `[A 6.2, B P4.3]`
- **tier** worker · **depends_on** 5.3
- **worktree/branch/target** `proxima-wt-risc31` · `gpu-risc/31-on-device-argmax` · own target
- **opens** `proxima-model-interop/src/generate.rs:1643` `sample_next_token`, `:1649` `next_ids`, `:1659` `greedy_pick_ms` (in `token_breakdown`, not `token_breakdown_metal`) [crit H-5]; `proxima-tensor/src/op.rs:203-205` — `Reduce`'s own doc names **argmax** explicitly as distinguished by `keep` and by whether `out_map` is data-dependent, i.e. a **scatter**, which is why this card depends on 5.3 and not the reverse. R3/M7: `greedy_pick` depends on `waitUntilCompleted` — a true data dependency; the fix is to move argmax on-device, not to thread it.
- **commands** append the argmax reduce to the program in `spec.rs`; read back 4 bytes instead of vocab-sized logits. Declare `on-device-argmax` default-off in `proxima-tensor/Cargo.toml`, forwarded.
- **expect** N ≥ 3 tests: on-device argmax matches `greedy_pick` bit-for-bit over the full budget on the real checkpoint; ties break as the CPU path does; **the non-greedy sampling path still receives full logits** (the feature must not silently degrade `sample_config`). Plus 2651/"known". **N==0 is RED.**
- **predict (nano → micro)** `readback_bytes` falls from vocab×4 (≈128 KB) to **4**; `readback_ms` falls from R13's **0.22** toward 0; `greedy_pick_ms` falls toward 0. Combined **< 0.5 ms/token** against R13's 67.92 — small, and the row leads with that.
- **kill** any drift in generated text ⇒ §14, revert.
- **rollback** `git revert`; feature default-off.
- **blast** `proxima-tensor/src/spec.rs` output roots, `generate.rs` sampling path, one new emitted scatter route.
- **observe** `READBACK_CALLS`, `READBACK_BYTES`, `readback_ms`, `greedy_pick_ms`.
- **reprove** G3 cell; grep `readback_bytes=`; assert generated text unchanged.
- **log-row title** `argmax moves on-device with no new Op; readback falls from 128 KB to 4 bytes and the win is under half a millisecond`

---

# Phase 6 — The minimal graph (demoted off the critical path by R12's null; conflict 3 resolved for B)

### 6.1 — Ops-per-layer census against the incumbent's 23 `[B P6.1, brief item 8]`
- **tier** hands · **depends_on** 5.6
- **worktree/branch/target** `proxima-wt-risc32` · `gpu-risc/32-ops-per-layer-census` · own target
- **opens** `proxima-tensor/src/spec.rs:2336-2865` `append_mistral_cached_layer` (530 lines, 25 args); incumbent's 23 real ops/layer enumerated at `llama-model.cpp:4691-4845` + `llama-graph.cpp:464-497,514-616,1071-1121,1205-1253`; views/reshape/permute are no-ops at `ggml-metal.m:1835-1847`; ~740 dispatches/token (R8 — which **corrects** the MEMORY figure "15/layer, ~483": ours 1196 vs 740 = **1.62x**, not 2.5x).
- **commands** print ops/layer for our graph from the 1.1 census and assert it in a test.
- **expect** N = the census prints per-layer op counts and the assertion holds; main's number to confirm is **37 ops/layer** at R13's **1196** dispatches. **N==0 is RED.**
- **predict** none — a census, not a measurement.
- **kill** n/a. **rollback** `git revert`; instrument + test only. **blast** instrument.
- **observe** ops/layer; `encode_dispatch_calls`.
- **reprove** G3 cell + the census assertion.
- **log-row title** `37 against 23: the ops-per-layer census, and the incumbent count it is measured against`

### 6.2 — Collapse the two-range online softmax to one range `[A 4.3, B P6.2, crit b, H-4]`
- **tier** worker · **depends_on** 6.1, 5.6, 5.4 · **measurer queue position 11**
- **worktree/branch/target** `proxima-wt-risc33` · `gpu-risc/33-attention-single-range` · own target. **Not** `perf/attention-single-range` (checked out at `proxima-wt-merge`) [crit g].
- **opens** `proxima-tensor/src/spec.rs:2596-2720` (the online-softmax combine, "no literal concatenation anywhere" at `:2616-2617`), `:2303-2319` (the doc 5.4 already corrected); the even/odd RoPE split that doubles it again; **`spec.rs:6282`** (the sole call site of `append_mistral_cached_layer`), and the 16 doc-level fan-out sites — specifically **`:2867-2891`** (Qwen3.5 dense-attention counterpart), **`:6465`** (the split-half RoPE path, documented as **NOT** going through this function — so the "even/odd RoPE split collapses in the same move" claim does **not** cover it and the row must say which checkpoints it does cover), and **`:3445`** (the MoE counterpart) [crit H-4]. Incumbent decode attention: `mul_mat(K,Q)` → `soft_max_ext` (scale+mask+max+exp+sum+normalize fused in ONE kernel, `ggml-metal.metal:1051-1145`, nth up to 256/ne00 via `ggml-metal.m:2501-2525`) → `mul_mat(V)` (R8).
- **commands** recover `docs/bench-campaigns/.../recovered/merge-tracked.patch` (0.5; 1 file +1181/−82, `spec.rs`) as **reference only** — main moved `spec.rs +8735/−2836` since (`0c3bd4f`), so expect a full conflict and **rewrite rather than merge**. Declare `attention-single-range` default-off in `proxima-tensor/Cargo.toml`, forwarded to `omega` and `proxima-model-interop`. Run the CPU oracle first.
- **expect** N: full CPU parity (R3/M11 recorded 488/488 for the single-range graph, gated on `cpu::evaluate` having write placement — 5.4 supplies it); ops/layer census falls; full-budget generated text identical; 2651/"known". Qwen3.5's hybrid attention/ssm path **re-parity-tested, not assumed**. **N==0 is RED.**
- **predict (nano → milli)** ops/layer falls from **37** to **≤25**; `encode_dispatch_calls` falls from R13's **1196** to **≤950** (R3/M11 measured 939 for single-range alone); `gpu_exec_ms` falls by **≤2 ms** — deliberately small, because R12 measured 1194 → 616 dispatches moving wall by 0.036 ms.
- **kill, written against BOTH the control and R13's noise floor** [crit b]: the success shape is **dispatches down AND `gpu_exec_ms` not risen beyond 2× the measured CoV AND `step_wall_ms` down beyond both CoV bands.** R13's `gpu_exec_ms` CoV is **0.7%**, so "killed if gpu_exec rises at all" would kill on noise; the criterion is a rise **> 1.4%**. And **wall is in the criterion**, not only counts — R12's finding was about wall (0.07%), and a 6.2 that reduces dispatches to 940 with flat GPU and flat wall reproduces the parallel branch's outcome exactly and must not pass. If wall does not move: that is the **second independent** measurement saying the graph is not the mass (R12 is the first); record the two-agreeing-results conclusion and stop Phase 6. The single-range graph still lands if it is correctness- or maintenance-positive, but **not as a perf row**, and 0.7's adjudication re-opens only if ≤23 ops/layer proves unreachable.
- **rollback** `git revert`; feature default-off. Highest rebase-conflict surface in the plan (`spec.rs`); rebase against main early and often.
- **blast** `spec.rs`'s attention builder and every model that uses it — Mistral, Qwen3, Qwen3.5 hybrid, and the MoE counterpart; the split-half RoPE path at `:6465` is explicitly **out of scope** and the row says so.
- **observe** ops/layer census, `encode_dispatch_calls`, `gpu_exec_ticks`, `step_wall_ms`, and the per-route split (the census is what says whether the removed ops were elementwise or reduces).
- **reprove** G3 cell + 6.1's assertion.
- **log-row titles** `one range, not two: the attention graph against the incumbent's op count` · `the control that says a dispatch collapse is not automatically a win, restated on our tree with wall in the criterion`

---

# Phase 7 — Backend coverage and the one emitter core

### 7.1 — CUDA covers `Iota` and `Constant` `[A 5.3, B P7.1, crit B-4]`
- **tier** worker · **depends_on** 5.3
- **worktree/branch/target** `proxima-wt-risc34` · `gpu-risc/34-cuda-iota-constant` · own target
- **opens** `omega/src/cuda.rs:146-183` (`emit_cuda`, `CudaUnsupportedOpKind`); reference `omega/src/msl.rs:2044-2103` (`render_iota`, `render_constant`); `omega/src/error.rs` (the variant's definition).
- **commands** implement both kinds; **do not delete the error variant in the same commit as the implementation** — land the implementation first (green), then remove the now-unreachable variant in a second commit, so the rollback is one revert of a small commit rather than an unwind through downstream exhaustive matches [crit B-4].
- **expect** N ≥ 2 new emit tests per kind; `cargo nextest run -p omega --features cuda,cpu` runs a non-zero count (`cuda` is not in `default`, which is exactly the condition that hides tests); `grep -c CudaUnsupportedOpKind omega/src/cuda.rs == 0` after the second commit. **N==0 is RED.**
- **predict (nano → micro)** the 15-cell coverage matrix `[Elementwise, Reduce(Reduce), Reduce(Scan), Iota, Constant] × [msl, wgsl, cuda]` is **15/15 Ok**; any `Err` is RED.
- **kill** CUDA cannot be compiled on this host (no toolchain) ⇒ emission is still testable **as text**, which is the point of a sans-IO emitter (§11); assert the emitted CUDA parses structurally and do **not** claim it runs.
- **rollback** `git revert` the second commit (restores the variant), then the first.
- **blast** `omega/src/cuda.rs`, `omega/src/error.rs`.
- **observe** the coverage matrix; per-backend route census.
- **reprove** `cargo nextest run -p omega --all-features -E 'test(backend_coverage_matrix)'`
- **log-row title** `one RISC means every backend covers every kind: the 15-cell matrix and the deleted rejection`

### 7.2 — One emitter core; backend text tables `[A 5.2, B P7.2]`
- **tier** worker · **depends_on** 7.1, 1.1
- **worktree/branch/target** `proxima-wt-risc35` · `gpu-risc/35-one-emitter-core` · own target
- **opens** the ~26 functions R5 lists as reimplemented per backend (`validate`, `reduction_dims`, `bindings`, `grid_threads`, `entry_name`, `scalar_op_expr`, `fold_init_tokens`, `push_body_steps`, `preamble`, `kernel_signature`, gather helpers, `operand_read`, `render_*`, `reduce_is_cooperative`) ≈ 78 near-duplicates across `msl.rs` 4712 / `wgsl.rs` 1929 / `cuda.rs` 1838; the 3 types + 2 fns already shared (`Binding`, `PackedCodec`, `PackedOperands`, `gather_count`, `gather_slots` — `wgsl.rs:105`, `cuda.rs:66`); `omega/src/msl.rs:673-697` (the surviving match); `msl.rs:4656` `emit_is_deterministic_byte_equal` (the determinism test the golden extends).
- **the shape** one generic core over `(BoundOpKind, KernelRoute)` consulting a `struct BackendText { … }` of `&'static str` fields + small fns. **A value table, not a trait** — §20 box-free, §11 no trait objects; a table is data, so §4's config-as-composition question has an answer.
- **commands** land **one kind per commit**, each green, each gated by the byte-identity golden. `bash scripts/omega-gate.sh` after each.
- **expect** N = omega gate green; wgpu parity tests unchanged; **byte-identity of emitted source for every op in the real openchat program, per route, per backend** — one byte of Metal drift is RED. The before line count (R5's 12,553 total across omega/src) is recorded in the row. **N==0 is RED.**
- **predict (nano → micro)** `wc -l omega/src/{msl,wgsl,cuda}.rs` falls from **8479** by **≥2500**; emitted source is byte-identical for all ~1196 (or post-6.2 ≤950) ops; `gpu_exec_ms` unchanged within R13's 0.7% CoV.
- **kill** any byte of source drift not explained by a named route ⇒ revert; a refactor that changes the kernel is not a refactor. Bisect by route.
- **rollback** `git revert` per kind, 4 independent commits. **Not feature-gated** (a gate on a refactor would mean two emitters, which is the opposite of the goal); the golden is the firewall — and unlike Plan A, the golden already exists from 1.1, so 1.1 is not left exposed [crit B-1].
- **blast** all three emitters. Largest blast radius in the plan; sequenced last for exactly that reason.
- **observe** line counts, golden hash per route per backend, `gpu_exec_ticks` flat.
- **reprove** `cargo nextest run -p omega --all-features -E 'test(golden_source)'` + `bash scripts/omega-gate.sh`
- **log-row title** `three emitters become one core plus three text tables; the kernel bytes did not move and here is the hash that proves it`

---

# Phase 8 — The incumbent cells that do not exist (R9: measured only for llama.cpp-Metal)

### 8.1 — llama.cpp `-fa 1` as a second incumbent arm `[A 1.4, B P1.4]`
- **tier** hands · **depends_on** 0.4 · **measurer queue position 12**
- **worktree/branch/target** `proxima-wt-risc36` · `gpu-risc/36-llama-flash-attention-arm` · own target
- **opens** `common/common.h:328` at checkout `b25346221` — flash attention is **OFF by default**, so every scoreboard row including R13's compares against the non-FA incumbent; `-fa 1` changes the attention path **and** the KV layout (`v_trans = !flash_attn`), so it is a second incumbent, not a variant.
- **commands** add a second arm to `scripts/sealed-pass.sh`: `llama-bench … -fa 1 -r 5`, interleaved with the `-fa 0` arm.
- **expect** N = 2 arms × 5 runs = 10 rows with ms/token and CoV each. **N==0 is RED.**
- **predict (micro → bench)** arm B lands within **±10%** of R13's arm A (**17.52 ms/token**, CoV 0.89%) at batch-1 decode — flash attention's win is a prefill/long-context effect, and R8 records the decode path at this checkout already fusing scale+mask+max+exp+sum+normalize into one `soft_max_ext` kernel.
- **kill** arm B is >20% faster ⇒ the home-turf incumbent for every later row is arm B, and **every ratio in R13/R1/R2/R10 re-bases against it**. Say so loudly rather than keeping the flattering arm.
- **rollback** `git revert`; one script.
- **blast** `scripts/sealed-pass.sh`, and the denominator of every ratio in `discipline.md`/`rooflines.md`.
- **observe** ms/token per arm; `design-favors: incumbent` on both.
- **reprove** `bash scripts/sealed-pass.sh`
- **log-row title** `the second incumbent arm: flash attention is off by default at b25346221 and here is what turning it on costs`

### 8.2 — The GPU streaming ceiling (roofline debt) `[A 1.1, B P1.1/P0.2, crit V-4, M-2]`
- **tier** worker · **depends_on** 0.1 · **measurer queue position 13**
- **worktree/branch/target** `proxima-wt-risc37` · `gpu-risc/37-gpu-streaming-roofline` · own target
- **opens** `omega/examples/membw_probe.rs:139-146` — the Metal arm builds `Op::Reduce(Reduce { body: ScalarOp::Add, init: ReduceInit::Zero, keep: Keep::Reduce, … })`, i.e. a **reduce-to-scalar**, which is why `rooflines.md:411` records the ceiling as **DEBT — not measured**; `omega/src/metal.rs:547-554` (`gpu_exec` window) vs `:2358-2378` (`readback`) — ROW 221's named residual is that readback of an equally-sized output sits **inside** the timed window and outside the numerator; `rooflines.md:396-479` and the summary row at `:751` and the closing note at `:766-773`.
- **commands** add a streaming-copy arm: `Op::Elementwise { body: ScalarOp::Identity, operands: [(src, affine identity)] }` over N f32 with a full-size output; sweep N ∈ {16 MiB, 64 MiB, 256 MiB, 1 GiB} × 5 runs. **Time only the commit → `waitUntilCompleted` window**; do the host readback after `read_ticks()` stops, and validate correctness on a separate untimed run.
- **the denominator, stated** [crit V-4]: the probe reports **two** columns — `read_bytes/s` and `(read+write)_bytes/s` — because a streaming copy touches bytes twice while the incumbent's 228.9 GB/s is a weights-**read**-only figure derived from 3.9996 GB / 17.470 ms (R1, MEMORY). Every kill and every fraction-of-ceiling names which column it uses. Depends on 0.1 so the bytes in the numerator are the tensor's, not the checkpoint buffer's [crit M-2].
- **expect** N = 4 sweep points × 5 runs = **20 samples**, each with both GB/s columns and CoV; `readback_bytes == 0` inside the timed window — that assertion is what proves the window is clean. **N==0 is RED.**
- **predict (micro → milli)** at N = 1 GiB the **read-bytes** column reaches **≥200 GB/s** on M1 Max — at or near the incumbent's achieved 228.9–234.1 GB/s, since a copy must beat a Q4_K matvec and the incumbent's achievement is a lower bound on the machine.
- **kill** the corrected **read-bytes** figure still lands below the incumbent's achieved 228.9 GB/s ⇒ the probe shape is still wrong (candidate: the one-thread-per-element grid is dispatch-bound, not bandwidth-bound); the debt row stays open with that named next experiment. **No spec-sheet figure is ever substituted** — `rooflines.md:411` already refused that once, and §18 makes an ASSUMED number ineligible to anchor a mechanism claim.
- **rollback** example-only; `git revert`.
- **blast** `omega/examples/membw_probe.rs`, `proxima-tensor/docs/rooflines.md:396-479`, `:751`, `:766-773`.
- **observe** `GPU_EXEC_TICKS` vs `READBACK_TICKS` (the split that proves readback left the window), `block_upload_bytes`, `readback_bytes == 0`.
- **reprove** `cargo run --release -p omega --features metal --example membw_probe -- --arm streaming-copy`
- **log-row title** `the GPU streaming ceiling exists: the roofline debt at rooflines.md:411 is paid, in two denominators`

### 8.3 — torch-MPS arms `[A 1.2, B P8.1]`
- **tier** worker · **depends_on** 0.8 · **measurer queue position 14**
- **worktree/branch/target** `proxima-wt-risc38` · `gpu-risc/38-torch-mps-arms` · own target
- **opens** `proxima-onnx/scripts/torch_reference/inference_bench.py:29-32` — `parse_args` has only `--threads` and `--runs` (R14); `train_bench.py`; `diagnostics.py`; `README.md` (the named, scoped python exception); ours: `omega/tests/training_step_parity.rs:400-607` (GPU train step exists as **untimed** parity tests only, R9).
- **commands** add `--device {cpu,mps}`; `torch.device("mps")`; `torch.mps.synchronize()` around the timed window (the MPS analogue of 8.2's readback rule); keep the file's existing `WARMUP_IMAGES = 50`. Use the existing venv: `proxima-onnx/scripts/torch_reference/venv/bin/python` (torch 2.13.0, MPS available — verified R0).
- **expect** N = 3 lanes × 2 devices × 5 runs, mnist reporting p50/p95/p99/mean/CoV. **N==0 is RED**, and the harness must assert `torch.backends.mps.is_available()` **and** `next(model.parameters()).device.type == "mps"` or the arm silently ran on CPU and exits 0.
- **predict (nano → micro)** mnist batch=1 on MPS is **slower** than CPU on this box (per-op MPS dispatch dominates a 14-node graph at batch 1), and the MLP train step at batch ≥64 is faster on MPS. Both directions are the result; the loss is reported first (§19).
- **kill** MPS silently falls back ⇒ the arm is void; report it as a gap in torch's own harness, never as our win.
- **rollback** python-only; revert the flag.
- **blast** two python fixture files; zero Rust.
- **observe** p50/p95/p99 + CoV per arm; `torch.mps.current_allocated_memory()`; the device assertion.
- **reprove** `proxima-onnx/scripts/torch_reference/venv/bin/python inference_bench.py --device mps --runs 200`
- **log-row title** `the torch-MPS cell that did not exist: mnist, MLP train step, and what batch=1 costs on a GPU`

### 8.4 — ORT-CoreML arms `[A 1.3, B P8.2]`
- **tier** worker · **depends_on** 0.8 · **measurer queue position 15**
- **worktree/branch/target** `proxima-wt-risc39` · `gpu-risc/39-ort-coreml-arms` · own target
- **opens** `scripts/onnx_reference/bench.py:96` — `providers=["CPUExecutionProvider"]` hardcoded (R14); `scripts/onnx_reference/run.sh` (pinned venv at `$HERE/.venv`, `ONNX_REF_PYTHON=python3.12`); the fidelity fields `cosine_similar` / `cosine_dissimilar_a/b` at `bench.py:82-85`. `onnxruntime` is **not installed** (R0) — extend this existing harness rather than creating a second venv (§1).
- **commands** `scripts/onnx_reference/.venv/bin/pip install onnxruntime`; if no network, build from `~/repos/others/onnxruntime` with `./build.sh --config Release --use_coreml --build_wheel --parallel`. Add `--provider {cpu,coreml}` threaded into the `InferenceSession` call at `:96`.
- **expect** N = BGE-small × 2 providers × 5 runs with the fidelity fields per provider; assert `session.get_providers()[0] == "CoreMLExecutionProvider"` **and** the **partition count** (a 1-node CoreML partition beside 200 CPU nodes is a CPU cell wearing a CoreML label). **N==0 is RED.**
- **predict (nano → micro)** the CoreML EP takes a **partial** partition (< 100% of nodes) and ms/sentence lands within **2x** of the CPU EP, with the fidelity fields unchanged (fp32 path).
- **kill** fidelity drift (cosine similar/dissimilar move) ⇒ CoreML chose fp16; that is a **different arm** and must be labelled as one (§14: differing output is ours to explain). Wheel unavailable for the pinned interpreter and the source build also fails ⇒ the cell is a documented **feature gap** ("cannot run the arm"), never omitted (§19: an omitted loss is a verdict).
- **rollback** revert the flag; the venv is gitignored.
- **blast** `scripts/onnx_reference/bench.py`, `run.sh`; zero Rust.
- **observe** ms/sentence, CoV, provider partition node counts (`sess_options.log_severity_level=0`), fidelity fields.
- **reprove** `BGE_MODEL_PATH=... ONNX_REF_PROVIDER=coreml bash scripts/onnx_reference/run.sh`
- **log-row title** `the ORT-CoreML cell that did not exist, and the partition that explains it`

### 8.5 — Our own GPU arms for the non-decode lanes `[A 1.5, B P8.3]`
- **tier** hands · **depends_on** 8.2 · **measurer queue position 16**
- **worktree/branch/target** `proxima-wt-risc40` · `gpu-risc/40-omega-nondecode-gpu-arms` · own target
- **opens** `omega/benches/metal_vs_cpu.rs` — the **only** GPU bench outside decode (gemm_square_f32 512/1024/2048, matvec_batch1_f32 at Mistral f32 shapes), registered at `omega/Cargo.toml:207-210`, doc says **UNRUN** (R9); `omega/tests/training_step_parity.rs:400-607` (GPU train step, untimed).
- **commands** `cargo bench -p omega --bench metal_vs_cpu --features metal` — run it for the first time; then add a timed arm **around the existing parity fixture** rather than writing a new workload (§1 reuse-first — the fixture is the workload, it just has no timer).
- **expect** N = 4 bench arms × 5 runs + ≥1 timed train-step cell. **N==0 is RED** — a `required-features`-gated bench that is never invoked compiles to nothing, and a bench that has never run may not compile.
- **predict (nano → micro)** `matvec_batch1_f32` at Mistral shapes lands under **25%** of 8.2's measured streaming ceiling — the low-simdgroup starvation R3/M5 names (52 GB/s at 256 simdgroups rising to 147 at 8001); `gemm_square_f32` at 2048 is the widest Metal-over-CPU margin.
- **kill** the bench does not build under `--features metal` ⇒ that is the finding, and fixing the registration is the card; a repo that does not build its own benches is our bug.
- **rollback** revert the timed arm.
- **blast** `omega/benches/metal_vs_cpu.rs`, `omega/tests/training_step_parity.rs`.
- **observe** GB/s and GMAC/s per arm against 8.2's ceiling (naming the denominator column), with 0.1's corrected byte accounting.
- **reprove** `cargo bench -p omega --bench metal_vs_cpu --features metal`
- **log-row title** `the non-decode GPU lanes get their first numbers; matvec at batch 1 against the measured ceiling`

---

# Phase 9 — ai_docs and the final board

### 9.1 — ai_docs JSONL records for the GPU lane `[A 1.6, B P9.1]`
- **tier** hands · **depends_on** 1.1, 0.7
- **worktree/branch/target** `proxima-wt-risc41` · `gpu-risc/41-ai-docs-gpu-lane` · own target
- **opens** `ai_docs/AGENT.md` (Update Rule: kind=5 decisions, kind=7 failures, `relations.idx=7` for grounding; step 5 says **add records**, never bypass); `ai_docs/index.jsonl` (schema `{id,kind,summary,path,read_when[],source_paths[],relations[]}`), `task-routes.jsonl` (`{task,purpose,must_read[],then_read_if_relevant[],queries[],done_when[]}`), `invariants.jsonl` (`{id,kind,summary,rule,applies_to[],evidence_required[],relations[]}`). R0: **zero** tensor/omega/GPU records in task-routes or invariants.
- **records**
  - index: `proxima.omega.gpu_decode_lane`, `proxima.omega.kernel_route_census`, `proxima.tensor.write_placement`.
  - task-routes: `gpu-decode-perf` (done_when: "board cell carries CoV and a roofline cell", "incumbent arm present", "route census sums to encode_dispatch_calls"), `omega-kernel-emission`.
  - invariants: `proxima.omega.one_risc_bound_kinds` (evidence: 0.9's variant-count tests + the parallel branch's own failure record); `proxima.omega.route_is_a_value_not_a_substring` (evidence: 1.2's census-sum gate; the defect at `metal.rs:785-826`; R12 ROW 263's 9/601 → 225/385); `proxima.omega.backend_covers_every_kind` (evidence: `grep -c ScatterNotSupported == 0`, `grep -c CudaUnsupportedOpKind == 0`); `proxima.omega.geometry_traces_to_sizing_config` (evidence: 1.3's grep == 0); `proxima.gpu.dispatch_count_is_not_the_denominator` (evidence: R12 1194→616 wall 51.571→51.535, and 6.2's second test); `proxima.gpu.profile_bytes_are_tensor_bytes` (evidence: 0.1, against R13's 4,140,417,024 defect).
- **expect** N = index +3, task-routes +2, invariants +6; `jq -c . ai_docs/*.jsonl > /dev/null` exits 0 on all three; `bash ai_docs/query.sh gpu-decode-perf` returns ≥1 row. **N==0 on any file is RED.**
- **predict (nano → micro)** `grep -ci "omega\|metal\|gpu" ai_docs/task-routes.jsonl` moves from **0** to ≥2, and the route query returns the invariants a future agent must read before touching the lane.
- **kill** malformed JSONL (jq non-zero) ⇒ fix before landing; a broken index is worse than none.
- **rollback** `git revert`; `ai_docs/` only. **blast** `ai_docs/`.
- **observe** record counts per file; query hit count.
- **reprove** `jq -c 'select(.task=="gpu-decode-perf")' ai_docs/task-routes.jsonl && jq -c 'select(any(.applies_to[]?; .=="gpu-decode"))' ai_docs/invariants.jsonl`
- **log-row title** `the GPU lane enters ai_docs`

### 9.2 — Final board re-seal `[A 4.4, B P9.2]`
- **tier** hands · **depends_on** 6.2, 5.6, 8.1, 8.2, 8.3, 8.4, 8.5 · **measurer queue position 18 (last)**
- **worktree/branch/target** `proxima-wt-risc42` · `gpu-risc/42-final-board-reseal` · own target
- **commands** `bash scripts/sealed-pass.sh` on a quiet box with every landed feature in `default`.
- **expect** N = every board cell filled: ours (`step_wall_ms`, `gpu_exec_ms`, CoV, n), llama arm A, llama arm B (`-fa 1`), torch-MPS, ORT-CoreML, roofline fraction (naming 8.2's denominator column). **A blank cell is RED.**
- **predict (milli → bench)** the band from 4.3, adjusted **only** by the milli-rung deltas 5.6 and 6.2 actually measured — this card does **not** re-predict from theory; it carries 4.3's band plus the two measured deltas, each with its CoV. Phases 5–6 contributed nothing to 4.3's original derivation by design (R12's null).
- **kill** the ratio does not fall below **3.0x** ⇒ R13's decomposition is wrong somewhere, and the row names which bucket did not move, by counter.
- **rollback** n/a (measurement). **blast** docs.
- **observe** every counter in the board; `rooflines.md:396-479`, the summary row at `:751`, and the closing note at `:766-773` are rewritten from DEBT to measured.
- **reprove** `bash scripts/sealed-pass.sh`
- **log-row title** `the board, every cell filled, against R13`

---

# Phase 10 — Residual levers, each gated on a re-measure

*Every item here was reasoned about before Phases 1–6 changed the tree. §18: a claim from a stale tree is not a claim.*

### 10.1 — Config sweep before a split-K kernel `[A 6.3, B P2.2]`
- **tier** hands · **depends_on** 1.3, 2.4 · **measurer queue position 17**
- **worktree/branch/target** `proxima-wt-risc43` · `gpu-risc/43-packed-row-geometry-sweep` · own target
- **opens** R3/M5 (marginal GB/s rising 52 → 147 from 256 → 8001 simdgroups; low-row shapes starve — R13's `attn_q` 5.172 ms at 58.5 GB/s and `attn_k`/`attn_v` at 50–53 GB/s against `ffn_down` at 108.7 are exactly that curve); 1.3's `[packed_row_block] rows_per_group` knob.
- **commands** sweep `OMEGA_PACKED_ROW_BLOCK_ROWS_PER_GROUP ∈ {1,2,4,8}` × 3 runs = **12 cells**, interleaved. A config sweep is cheaper than a new kernel route and may retire the lever (§1 reuse-first applied to geometry).
- **expect** N = 12 cells with per-family GB/s (0.1's corrected bytes). **N==0 is RED.**
- **predict (micro → milli)** a knob value other than 4 wins on the low-row families (`attn_q/k/v/o`) by **≥5%**; if none does, split-K's premise (more simdgroups is the fix) is refuted by a cheaper experiment and the kernel is not written.
- **kill** no knob value beats 4 by more than the measured CoV ⇒ do not build split-K; row the negative. If a knob does win, split-K is a **separate** card with its own bit-reproducibility test for the partial-sum combine (a non-deterministic reduction order is a §14 hazard) — and note nsg=2 regrouping is a **four-time** negative (R4, `perf/metal-simdgroup-geometry`, R12 ROW 267, R12 ROW 259/260) and is a different mechanism from split-K; re-proposing nsg=2 is a discipline failure, not an experiment.
- **rollback** build-time config; revert to 4.
- **blast** build config only, unless/until a kernel is built.
- **observe** per-route GB/s from the 1.1 census; `gpu_exec_ticks`.
- **reprove** the 12-cell sweep script.
- **log-row title** `the simdgroup-starvation hypothesis, tested with a config knob before a kernel`

### 10.2 — Re-measure the rematerialization subset `[A 6.1]`
- **tier** hands · **depends_on** 6.2
- **worktree/branch/target** `proxima-wt-risc44` · `gpu-risc/44-remat-remeasure` · own target
- **opens** R3/M8: only the `elements < 247` subset (96 nodes) was a certain win (1196 → 1100); the aggregate set was a loss; "rematerialize all ≤2-consumer nodes" is a **dead lever** (R4, 6x downside on the slow ALU arm) and is not re-proposed.
- **commands** re-run the subset census on the post-6.2 graph **before** building anything.
- **expect** N = the post-6.2 count of `elements < 247` nodes. **N==0 is RED** — and N==0 would mean 6.2 already removed them, which is itself the finding and closes the item.
- **predict (nano → micro)** the subset shrinks by more than half after 6.2 (most were attention-graph intermediates), and the remaining win is under 1% of R13's 67.92 — below the noise floor at R13's 0.5% wall CoV, which retires the lever.
- **kill** the predicted win is < 2× the measured CoV ⇒ row it as "no signal, kept the simpler form" and do not build. R12's dispatch-count null is the standing reason: a 578-dispatch reduction moved 0.036 ms, so a 96-dispatch reduction cannot matter.
- **rollback** n/a if not built. **blast** `proxima-tensor/src/bind.rs` if built.
- **observe** node census by element count; `encode_dispatch_calls`.
- **reprove** `cargo run --release -p omega --features metal,instrument --example real_forward_emit_probe`
- **log-row title** `rematerialization re-measured on the post-placement graph`

---

# Dependency graph

```
0.1 op_profile bytes ──> 0.2 harness N-contract ──> 0.3 quiet-box replicate ──> 0.4 sealed-pass.sh
0.1 ──────────────────────────────────────────────────────────────────────────> 8.2 roofline probe
0.2 ──> 0.5 quarantine patches ──> 0.6 prune_dead ──> 0.9 op.rs doc + cardinality ──> 0.10 one bound plan
0.5 ──> 0.7 CachedAttention adjudication ──> 0.11 row-number protocol
0.8 clean tree ──> (every interleaved measurement) ; ──> 8.3, 8.4

0.10 ──> 1.1 KernelRoute + per-dispatch census ──┬─> 1.2 census+identity gate ─┐
                                                 └─> 1.3 geometry config ──────┤
                                                                               v
                                    2.1 mask-fma ──> 2.2 pair-dot ──> 2.3 adjudicate ──> 2.4 default
                                                                                              │
                          1.3 ──────────────> 3.1 wide cooperative reduce <───────────────────┤
                                                    │                                         │
                                                    ├─> 3.2 elementwise census                │
                                                    └─> 3.3 re-seal <─────────────────────────┘
                                                             │
                                    0.2 ──────> 4.1 kv-capacity-bucket (flips the harness assertion)
                                                             │
                                                4.2 buffer pool ──> 4.3 RE-SEAL (the one board prediction)
                                                             │
                                                5.1 write the expression ──┬─> 5.2 form selection ──> 5.4 offset
                                                                           └─> 5.3 scatter coverage ──> 5.7 argmax
                                                5.4 + 5.5 capacity contract ──> 5.6 KV device-resident
                                                                                       │
                                                             6.1 ops/layer census ──> 6.2 single-range
                                                5.3 ──> 7.1 CUDA kinds ──> 7.2 one emitter core
1.3 + 2.4 ──> 10.1 geometry sweep                6.2 ──> 10.2 remat re-measure
0.4 ──> 8.1 -fa 1 arm ; 8.2 ──> 8.5 non-decode arms ; 0.8 ──> 8.3 torch-MPS, 8.4 ORT-CoreML
1.1 + 0.7 ──> 9.1 ai_docs
6.2 + 5.6 + 8.1..8.5 ──> 9.2 final board
```

**Hard edges, stated in prose so the picture and the text agree** [crit O-4]: 1.1 requires 0.10 (the identity test is what makes "one bound plan" checkable before the router changes). **2.1 requires 1.2** — the census is a hard precondition of any body swap (conflict 1). 3.1 requires 1.3 (the width must be a config key before it is swept). 4.2 requires 4.1 (a pool refilled every token is not a pool). 4.1 requires 0.2 (the harness assertion is flipped in 4.1's own commit). 5.4 requires 5.2's selection rule; 5.3 requires only 5.1 and is **not** conditional on 5.2, because 5.7 needs GPU scatter independently. 5.6 requires 5.4 **and** 5.5 (an offset and a destination the validators accept). 6.2 requires 5.6, 5.4 and 6.1. 7.2 requires 1.1 (the core dispatches on `KernelRoute`) and 7.1. 10.1 requires 1.3.

**Critical path to the board prediction:** 0.1 → 0.2 → 0.3 → 0.10 → 1.1 → 1.2 → 2.1 → 2.2 → 2.3 → 2.4 → 3.1 → 3.3 → 4.1 → 4.2 → **4.3**. Everything in Phases 5–7 is **off** that path and is sequenced for correctness and one-RISC conformance, not for predicted wall movement (conflict 3).

**There is no "parallel phase."** Every GPU-measuring card is serialized by G6's queue; non-measuring cards may proceed concurrently only while no build runs on the box [crit O-1].

---

# Rollback map

| card | rollback | main's default affected before rollback | firewall |
|---|---|---|---|
| 0.1 | `git revert` | yes (instrument only) | new tests assert byte model vs R13's derived column |
| 0.2 | `git revert` | test-only | the assertion still fires on an injected hit |
| 0.3, 0.4 | `worktree remove` / revert 2 | no / scripts only | quiet-gate + CoV band |
| 0.5 | `git revert` — **patches remain in git history**, which is the point | docs only | never `/tmp` [crit R-2] |
| 0.6 | `git revert` 1 commit | yes | no flag; bisect by revert; census names every removed node |
| 0.7, 0.11, 9.1 | docs/JSONL revert | no | n/a |
| 0.8 | `git revert` | ignore file | `git status --porcelain` == 0 |
| 0.9, 0.10 | `git revert` | doc + tests | exhaustive matches fail to compile on a variant change |
| 1.1 | `git revert` — **multi-commit unwind once 7.2 lands on top; revert 7.2 first** [crit B-1] | yes (`route_of` is on the default path; recorder is instrument-gated) | **kernel_cache_key stability + golden emitted source, both in THIS card** |
| 1.2 | revert the gate step | gate only | n/a |
| 1.3 | revert build.rs + toml + const sites together | yes, value-identical by construction | per-key equality + env-override tests |
| 2.1, 2.2 | `git reset --hard`; delete the feature | no | default-off |
| 2.3 | drop both features | no | route-count pin voids incomparable arms |
| 2.4 | demote out of `default` (one line), then revert | yes | full parity suite; `encode_dispatch_calls` unchanged |
| 3.1 | feature default-off; revert | no while gated | `metal_parity` / `backend_parity` |
| 3.2 | `git revert` | instrument only | n/a |
| 4.1 | `git revert`; **bucket=1 is the identity only if the bucket is a runtime read — the row states which** | yes | feature-off **program-hash** test, not an assertion [crit B-3]; CPU oracle 2651/"known" |
| 4.2 | feature default-off; revert | no while gated | pooled-vs-unpooled output identity; retirement tests |
| 5.1 | `git revert` | test-only | n/a |
| 5.3 | `git revert` per backend, 3 independent commits | no (nothing on the default path emits scatter until 5.6) | per-backend CPU parity on disjoint and colliding fixtures |
| 5.4 | `git revert`; **land 5.4 and 5.6 as separate commits** | yes — bounds-check semantics change for every existing `Reduce` | `IndexMap::placed` is a constructor, so no existing map changes meaning; autograd suite |
| 5.5 | `git revert`; the strict checks return | yes — public signature in two crates | the **over-declared weight** sad-path test on BOTH `metal.rs:991-1000` and `cpu.rs:346-356` |
| 5.6 | `git revert` 5.6 alone | yes | pre-allocation size assert (not a post-hoc detector); text identity |
| 5.7 | `git revert` the sampling commit alone | yes | sampling path still receives full logits (asserted) |
| 6.2 | `git revert` `spec.rs` | yes — every model using `append_mistral_cached_layer`, incl. qwen3.5 | feature default-off; text identity; split-half RoPE explicitly out of scope |
| 7.1 | revert the variant-deletion commit, then the implementation | yes | two commits, not one [crit B-4] |
| 7.2 | `git revert` per kind, 4 commits | yes, byte-identical by construction | the golden from 1.1 |
| 8.x, 9.2, 10.x | revert script/example/flag; not built unless the kill passes | no Rust hot path | measured-first |

Every landing commit is a green bisect point; primitives land before callers (5.4 before 5.6; 1.1 before 7.2; 1.3 before 3.1 and 10.1). **No commit lands without owner authorization.**

---

# Abandoned designs (traced)

1. **`BoundOpKind::CachedAttention`** — a fifth bound kind carrying an eight-input fused online-softmax macro-op, with the post-bind structural matcher (`cached_attention_candidates`, `attention_score_sources`, `is_exact_causal_mask`, `removable_attention_dependencies`) and `physical.rs` (+576). **Ruled out by** workspace AGENTS.md's hard invariant against arbitrary rules for specific instances, and by §1's binary question — `Reduce.out_map` placement can express it, so the expression is written (5.1) rather than a kind minted. Corroborating, not the reason: the branch's own `failure-cached-attention-matcher.md` abandoned the first matcher as a heuristic that "cannot prove the semantic roles", and its numbers show wall unmoved (51.535 vs 51.571) with `gpu_exec` worse (39.841 vs 35.117). **What changed:** the plan reaches the same collapse through 5.4 + 5.6 with zero new kinds, and keeps R12's cell as 6.2's pre-registered control instead of as a win. **What survives from that branch:** `prune_dead` (0.6) and the paired Q4_K body (2.2). **Re-open condition:** 6.2 measuring that ≤23 ops/layer is unreachable through placement.
2. **`Op::Concat` / `Op::Pad` / `Op::Tile` / `PlacedBuffer` / `write_placement`** — zero hits on main, and none added. **Ruled out by** §1 plus §6's read-the-code: `layout_of` (`bind.rs:1594-1605`) already folds `axis.offset * stride` into `Layout.base`, `msl.rs:2733` already emits `u.out_base`, and `IndexMap` already carries the write-direction destination-extent convention (`map.rs:110-131`, constructor at `:172`). The only blocker is one match arm at `shape.rs:469-485`. **What changed:** Phase 5 became a constraint removal plus a driver insert instead of a variant every backend must learn.
3. **A `PlacedBuffer` type in the Metal driver.** **Ruled out by** the relocation question, run: `metal.rs:2249` is already `device_buffers.insert(bound.node, (output, 0))` over `BTreeMap<NodeId,(MetalBuffer,usize)>`; aliasing is `insert(node, (persistent, offset))` — identical lines at the call site ⇒ a relocation. Plus `AlignedBuffer` (`align.rs:69`) exists with zero production callers; a peer beside an unused primitive is debt twice over.
4. **A new field on `QuantizedBlock` to declare over-allocation.** **Ruled out by** the pipe and relocation questions run in-line at 5.5 — the field cannot reach `cpu.rs:346-356`, whose arm takes `&[&[f32]]`, and that path is where the §14 oracle runs. **What changed:** the declaration became a parallel `&[Option<u64>]` slice threaded into **both** validators, with no new type in either crate.
5. **A `trait KernelBackend` / trait-object emitter registry** to unify the ~78 near-duplicates. **Ruled out by** §20 box-free and §11 (forbidden: trait objects) — and a table is data, so §4's config-as-composition question has an answer where a trait impl is a recompile. **What changed:** 7.2's `BackendText` is a plain value table over an exhaustive `match (BoundOpKind, KernelRoute)`, which is also why 7.1's coverage is a compile-time property rather than a runtime rejection.
6. **Threading `op_setup` / the encode loop** to hide the orchestration slice. **Ruled out by** §21 (a lock is a missing owner — the owner is the `Plan`), R3/M9 (non-`Send` `MTLBuffer` blocks it at the type level), and R4 ("thread count explains zero of the gap"). R13 shows the cost is 1196 `newBufferWithLength` calls, i.e. allocation, not serialism. **What changed:** 4.2 removes the work rather than distributing it; `PROXIMA_ORCH_THREADS` is **not** in this plan.
7. **`cached_len` as a runtime uniform** so the bound plan is shape-independent. **Parked, not deleted**, by blast radius: `BoundOp.extents` is a baked `Vec<u64>` (`bind.rs:200-215`) and making it dynamic touches every backend's uniform packing, `entry_name`, `kernel_cache_key` and grid computation. **Claim it gates:** a 100% plan-cache hit rate at every context length. **Un-park condition:** 4.1 measures a hit rate below (bucket−1)/bucket, or the tail mask costs more than the re-plan it replaces.
8. **Headlining the dispatch-count reduction** as the plan's spine, which is what the brief's one-line diagnosis proposed. **Ruled out by** R12's control (1194 → 616 moved wall 0.07% and moved GPU **up** 13.5%) plus R13's own per-op table (225 packed-row-blocked ops carry 44.450 ms while 547 elementwise carry 7.350). **What changed:** the entire phase ordering — dispatch count moved from spine to counter, and the spine became (route census, kernel body, plan stability, then graph).
9. **Rematerializing the low-element node set.** **Ruled out by** R4's dead-lever record combined with R12's null: a 578-dispatch reduction moved 0.036 ms, so a 96-dispatch reduction cannot matter. Retained only as 10.2, a re-measure with a kill at < 2× CoV.
10. **nsg=2 / ggml packed-simdgroup regrouping.** **Ruled out by** four independent measured negatives (R4, `perf/metal-simdgroup-geometry`, R12 ROW 267, R12 ROW 259/260). Re-proposing it is a discipline failure, not an experiment.
11. **A kernel-fusion engine** as the route to parity. **Ruled out by** R8: `grep -rln fuse ggml/src` is **empty** at `b25346221`; rms_norm and mul dispatch as two kernels. Parity is reachable without fusion; fusion is upside past parity.
12. **A spec-sheet GPU bandwidth figure** to close the roofline debt. **Ruled out by** §18 (ASSUMED provenance may never anchor a mechanism claim) and by `rooflines.md:411`, which already refused this substitution once. **What changed:** 8.2 is a real streaming-copy probe with readback outside the timed window, a `readback_bytes == 0` assertion, and **two** denominator columns.

---

# Open questions resolved by measurement (never by asking)

| # | question | card | the number that answers it |
|---|---|---|---|
| Q1 | Is the profiler's byte accounting wrong in a second place besides the checkpoint buffer? | 0.1 | per-family GB/s vs R13's shape-derived column (ffn_up 97.4, output.weight 145.9) |
| Q2 | Does the quiet box move R13's means or only its CoV? | 0.3 | `step_wall_ms` vs 67.92, `gpu_exec_ms` vs 56.93, CoV vs 0.5%/0.7% |
| Q3 | Is there really ONE bound plan across all four executors? | 0.10 | the plan fingerprint, 4/4 equal or not |
| Q4 | Was `classify_kind` lying about the route distribution, and does a route value reproduce R13's buckets? | 1.1, 1.2 | census counts vs R13's 225/385/547/37/2, and the census-sum vs `encode_dispatch_calls` |
| Q5 | Are `metal-q4k-mask-fma` and `q4k_pair_dot` the same mechanism at the same speed? | 2.3 | median `gpu_exec_ms` spread vs 2× pooled CoV, with the `ReduceRowBlockedPacked` route count pinned equal; then the shift-count/fold-site integers |
| Q6 | Do the 2026-09-02 numbers survive a rebase onto 9 commits of main? | 2.1, 2.2, 2.3, 3.1 | re-earned against 0.3's cell, not carried |
| Q7 | Does bucketing `cached_len` reach `plan_hits > 0`, and does that actually remove `prepare_ms`? | 4.1 | `plan_hits` vs R13's 0; `prepare_ms` on hit tokens vs R13's 1.97 — two separate numbers, the second does not follow from the first |
| Q8 | Does removing 1196 allocations remove the 3.9 ms, or does the cost reappear in `encode_dispatch`? | 4.2 | `op_setup_ms` vs R13's 3.9 **and** `encode_dispatch_ms` vs 0.47 **and** `step_wall_ms` |
| Q9 | What was `UNIFORM_BUFFER_REUSES` already doing before we assumed uniforms were the cost? | 4.2 | its hit rate at `metal.rs:2069`, read first |
| Q10 | Is write placement cheaper as an affine offset or as a scatter? | 5.2 | per-write µs both forms at the same extents, per-op mode's +7.3% inflation noted |
| Q11 | Can a GPU scatter be emitted without atomics for the disjoint case? | 5.3 | parity vs `cpu::run_reduce_scatter` on disjoint and colliding fixtures, per backend |
| Q12 | Does an over-declared **weight** block get through the relaxed validators? | 5.5 | the sad-path test on both `metal.rs:991-1000` and `cpu.rs:346-356` |
| Q13 | Does the persistent KV buffer actually hit `NOCOPY_BUFFERS`? | 5.6 | `nocopy_reuses` (predicted 96/token), `kv_cache_upload_bytes` (predicted 0 after step 1) |
| Q14 | Does removing ~250 dispatches move the wall at all? | 6.2 | `encode_dispatch_calls`, `gpu_exec_ms` and **`step_wall_ms` jointly** — R12 is the first test, this is the second |
| Q15 | Does the emitter unification change a single byte of Metal source? | 1.1, 7.2 | the golden hash per route per backend |
| Q16 | Is the machine's streaming ceiling above the incumbent's achieved 228.9 GB/s — is there headroom at all? | 8.2 | GB/s at 1 GiB, **read-bytes column**, with CoV |
| Q17 | Is llama.cpp `-fa 1` a stronger incumbent, invalidating every ratio in the log incl. R13's 3.88x? | 8.1 | ms/token `-fa 0` vs `-fa 1`, interleaved |
| Q18 | Do torch-MPS and ORT-CoreML actually beat us on any lane? | 8.3, 8.4, 8.5 | ms/sentence and p50/p95/p99 per provider with fidelity fields — R9: today there is **no cell on either side** |
| Q19 | Is the 7.35 ms elementwise bucket concentrated or uniform? | 3.2 | top-5 nodes' share |
| Q20 | Is simdgroup starvation fixable with a config knob before a split-K kernel is written? | 10.1 | the 12-cell `rows_per_group` sweep vs CoV |

---

# Conflict resolutions

1. **Route census before or after the Q4_K body swap** — **B wins (hard precondition, card 1.1 → 1.2 before 2.1).** R12 ROW 263 measured `classify_kind` relabelling 9/601 → 225/385 the moment a body changed, and R13's own 225/385 split is that same substring instrument's output; swapping a body first makes every Phase-2 attribution unfalsifiable, which Plan A diagnosed at its own line 312 and then kept the order anyway.
2. **Scatter-expression-first (B P5.1/P5.2) vs affine `out_map` offset (A 4.1)** — **B's ordering wins, A's design survives as the expected winner.** §1 binds: the expression is written and evaluated on main today (5.1) before `project_output_shape` is touched, and 5.2's timer — not a preference — selects the affine form; scatter coverage (5.3) lands regardless because on-device argmax needs it independently.
3. **Is the single-range graph on the critical path to the board prediction** — **B wins (no; demoted to Phase 6, off the path).** Two independent cells say a dispatch collapse is not a wall movement (R12's 1194→616 for 0.036 ms; R13's 547 elementwise ops for 7.350 ms against 225 matvec ops for 44.450 ms), so 4.3's board prediction is derived only from the body and the orchestration, and 6.2 is a correctness/one-RISC card carrying the second independent test of the same proposition.
4. **Worktree naming scheme** — **neither; a new scheme (G1).** Plan A's `perf/kv-device-resident`, `perf/attention-single-range` and `bench/sealed-pass` are all checked out elsewhere and `git worktree add` would refuse them, and Plan A gave one worktree four branches; the merge uses `proxima-wt-risc<NN>` / `gpu-risc/<NN>-<slug>`, one worktree per branch per card, verified collision-free (`gpu-risc/*` = 0 branches, no `risc` worktree dirs).
5. **Re-seal on a quiet box in Phase 0 given R13 already exists** — **yes, but re-scoped (card 0.3): R13 stays THE baseline and the quiet-box run produces the CoV band, not a new baseline.** R13 was taken on a loaded box (load 4.7–5.7, cdb-daemon resident) so its 0.5%/0.7% CoV is an upper bound on the noise; every kill criterion in this plan is set outside a measured band, and setting bands from a loaded-box cell would kill cards on noise — which is precisely how Plan A's 4.3 kill ("killed if `gpu_exec_ms` rises at all") landed inside R13's own 0.7% CoV.

---

### Critical Files for Implementation

- `/Users/brianbruggeman/repos/slot-0/proxima/omega/src/metal.rs` — `operand_bytes` defect at `:612`/`:694`; validation loop `:991-1000`; `classify_kind` `:785-826` and its call sites `:709-710`; `encode_op` `:2179-2252` with `allocate_buffer` `:2210`, `upload_uniforms` `:2211`, `ENCODE_DISPATCH_CALLS` `:2244` (the per-dispatch census site), `device_buffers.insert(node,(output,0))` `:2249`; `UNIFORM_BUFFER_REUSES` `:2069`; `NOCOPY_BUFFERS` `:1848-1892`; `mark_resident` `:350-362`
- `/Users/brianbruggeman/repos/slot-0/pro