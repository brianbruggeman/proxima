I have verified enough against main. Here is the critique.

---

# CRITIQUE — Plan A, GPU parity through omega

Verified read-only at `/Users/brianbruggeman/repos/slot-0/proxima`, HEAD `4be2f3ac297261592a0f827422bb80a0c4393d9f`. No `proxima-wt-*` entered.

## 1. Risk surface

**R-1. Step 2.1's success breaks the plan's own BENCH-rung harness, and every step after it inherits a red test.** The test the plan names as the BENCH rung in its "Canonical commands" block (planA:53-55) is `runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache`, which hard-asserts:

```rust
assert_eq!(runtime.plan_hits, 0,
  "cached_len grows every decode step, so no (new_count, cached_len) shape can repeat within one call");
assert_eq!(runtime.plan_misses, forward_calls_taken, ...);
```
`proxima-model-interop/src/bind.rs:3052-3058`. Step 2.1's pre-registered prediction is `plan_hits` 0 → 23 (planA:275). The step's own re-prove command (planA:280) greps that test's stdout. Neither `bind.rs:3052-3058` nor the test's doc comment (`bind.rs:2996-2998`, which states the zero-hit finding as the test's purpose) appears in 2.1's `open` list or blast radius. Blast radius: 2.1, 2.2, 2.3, 4.2, 4.3, 4.4, 6.2 all use this command as their gate.

**R-2. Step 0.2's rollback destroys the only copy of the measured wins.** R7 records the −17.2% / −4.9% / −11.1% results as uncommitted diffs in ten worktrees. 0.2 exports them to `/tmp/gpu-wins/` and states "rollback: delete `/tmp/gpu-wins/`" (planA:94). `/tmp` is not a repo path and is OS-cleared; 0.4 then "deletes, not leaves dormant" the losing body (planA:118). After 0.4 the loser exists only in a dirty worktree the plan never re-audits and a `/tmp` dir whose documented rollback is deletion.

**R-3. Step 4.2's `found >= expected` relaxation sits in the shared input-validation loop, not a KV-specific one.** `omega/src/metal.rs:991-1001` iterates `block_nodes.iter().zip(blocks.iter())` over *every* named input — weights included — and raises `InputSizeMismatch` on `found != expected`. The relaxation is therefore in the weight path by construction; only the proposed `QuantizedBlock` declaration field keeps it out. The plan calls this "the highest-risk hunk" (planA:366) but its sad-path test is described as "under-allocated block still rejected" — which does not test the case that matters (an over-declared *weight* block).

**R-4. 4.2's 34 GB kill is a detector, not a guard.** planA:364 asserts `device_allocated_bytes < 8 GB` *after* the allocation. On a box the plan also schedules Phase 1 GPU measurers on (see O-1), a 34 GB Metal allocation poisons every concurrent cell before the assert executes.

**R-5. 3.1 is unflagged and touches kernel identity.** `kernel_cache_key` at `omega/src/msl.rs:754-771` derives the route char by calling `tiled_gemm_block(...)` then `packed_row_block(...)` and pushing `'G'`/`'B'`/`'S'`; `'S'` is shared by serial reduce, cooperative reduce, elementwise, iota and constant. A `KernelRoute` with 8 variants that also feeds `kernel_cache_key` (planA:317 says `kernel_cache_key` calls `route_of`) changes key cardinality unless the 8→3 collapse is preserved exactly. 3.1 is explicitly "not feature-gated … pure refactor" (planA:321) and carries no golden-source test — the golden appears only at 5.2 (planA:413).

## 2. Ordering

**O-1. Phase 1 is declared parallel with Phase 2 (planA:179) and contains four GPU measurers.** 1.1 (streaming-copy probe), 1.4 (`llama-bench -fa 1`), 1.5 (`cargo bench metal_vs_cpu`) and 2.3's re-seal all contend for the same GPU. The plan's own protocol line says "one measurer on the box" (planA:67). The parallelism claim is justified as "nothing here touches a hot path" (planA:179) — hot path is not the contended resource.

**O-2. The observability fix (3.1) is sequenced after the three steps whose attribution needs it.** 0.3, 0.4 and 0.5 change kernel bodies; `classify_kind` (`omega/src/metal.rs:785-826`) classifies by `kernel.source.contains("q4k_run8(blk")` / `"simd_sum("` etc. after calling `emit()`. The plan states this itself at planA:312 ("`classify_kind` … silently relabels when a body changes — which Steps 0.4/0.5 just did") and at planA:108 instructs Luna to use the lying instrument "by family, never by bucket" in the interim. R12 ROW 263 already measured this exact relabel (9/601 → 225/385). The plan diagnoses the ordering defect and keeps the order.

**O-3. Phase 0 pre-assigns ROW 234-248, Phase 1 ROW 249-254, Phase 2 ROW 255-257 — while declaring Phase 1 and Phase 2 concurrent.** Two concurrently-landing branches cannot both own ROW 255. R10 records that this exact mechanism already produced non-monotonic numbering (ROW 205 at discipline.md:17864 preceding ROW 204 at 17950). The brief's rule is placeholders assigned at land time; planA:73 states the rule and then every step violates it with a literal number.

**O-4. The dependency ASCII contradicts the prose.** planA:504-513 draws an edge descending from 2.2 into `3.1 KernelRoute`; planA:526 says "3.1 has no *build* dependency on Phase 2." One of the two is wrong, and Luna executes the picture.

**O-5. No step re-verifies the MEMORY mechanisms that Phase 0 depends on.** The brief requires MEMORY rows be re-verified before a task relies on them. 0.3's prediction rests on R3/M4 (wide reduce −20%, −2.8 ms) and 0.4's on R3/M3; 0.1 re-seals only the aggregate cell, not those mechanisms. There is no step that re-reads `msl.rs:3195-3197` behaviour against a measurement before 0.3 predicts from it.

## 3. Rollback

**B-1. 3.1's rollback is stated as "a single revert" (planA:321, planA:543) but 5.2 and 3.2 build on it.** Once 5.2 ("depends on 3.1", planA:410) lands, reverting 3.1 is a multi-commit unwind. The rollback map row for 3.1 (planA:543) says "revert (single commit, pure refactor)" with "**no flag** — golden-source test is the firewall" — but 3.1 has no golden-source test; that test is introduced by 5.2.

**B-2. 0.4's "delete the loser's feature flag, not leave it dormant" is irreversible against a `/tmp` backing store.** See R-2.

**B-3. Untestable rollback: 2.1's "feature off ⇒ bit-identical graph" (planA:541) is asserted, not gated.** No step asserts the feature-off program is node-for-node identical to main's (e.g. a `BoundOp`-count or program-hash nano test on both arms). The claim is exactly the kind the plan elsewhere insists be proven by a hash (5.2).

**B-4. 5.3's rollback "restore `CudaUnsupportedOpKind`" (planA:429) is mechanical only if 5.2 has not already deleted the runtime rejection path.** 5.3 deletes the error variant from `omega/src/error.rs`; reverting it after downstream code compiles against the exhaustive match is not "mechanical."

## 4. Missing steps

**M-1. The plan never cites R13. Zero occurrences.** Citation counts across planA: R0=4, R1=8, R2=2, R3=11, R4=2, R5=7, R7=9, R8=8, R9=1, R11=5, R12=20, **R13=0**, R6=0, R10=0. Every per-stage baseline is taken from the superseded MEMORY rows (see hazard (a)).

**M-2. No step fixes the instrument defect R13 cards.** R13 (ledger:270-274) states `operand_bytes` reports 4,140,417,024 (the whole checkpoint mapping buffer) and closes with "Card: fix `op_profile` byte accounting before any GB/s row is written." The producer is `omega/src/metal.rs:694` (`pub operand_bytes: u64` at `:612`); the consumers are `proxima-model-interop/src/generate.rs:109,128,139-150,171-193,211-212` — every `gpu_ns_per_byte` and `total_operand_bytes` row. Plan A writes GB/s rows in 1.1, 1.5, 3.2 and 6.3 and never mentions `operand_bytes`, `op_profile`, or byte accounting.

**M-3. No worktree-creation commands.** The brief requires "worktree + branch + target dir, the exact commands." 26 worktrees are named; no `git worktree add` appears anywhere. Luna cannot `cd` into a directory that does not exist.

**M-4. Three features the plan's own re-prove commands pass do not exist and no step creates them in the right manifest.** `proxima-model-interop/Cargo.toml [features]` = `default, std, interop-bgpool, instrument, metal, metal-tiled-gemm`. Step 2.1's re-prove is `cargo test -p proxima-model-interop --features std,metal,instrument,kv-capacity-bucket` (planA:280) — unqualified, so the feature must exist in *that* manifest and forward to proxima-tensor; the plan says only that it is "behind feature `kv-capacity-bucket`". Same for 4.1's `cargo nextest run -p omega --features metal,tensor-write-offset` (planA:355) — `omega/Cargo.toml [features]` has no such key and does not forward proxima-tensor features. `proxima-tensor/Cargo.toml` has no `tensor-write-offset` / `attention-single-range` either.

**M-5. 0.2's "commands" field contains prose where a command must be.** planA:90 reads: "`git -C /Users/brianbruggeman/repos/slot-0/proxima --work-tree=<n/a>` is not usable; use `git diff` piped through the *branch pointer*…". A hands model cannot execute a sentence explaining why a command does not work.

**M-6. 0.4's tie-break rule is not executable.** Rule (3): "the body textually closer to `ggml-metal.metal:5147-5175` wins" (planA:118). No metric, no diff procedure, no threshold. Rules (1),(2),(4) are executable; (3) is the one that decides the likely case, since (2) predicts the spread is inside CoV (planA:117).

**M-7. Brief one-RISC item 1 has no step.** "ONE bound plan (`&[BoundOp]`) produced by ONE rewrite engine, identical for every backend" (brief:181 / ledger R5). The plan's binding section covers items 2 (3.1), 3 (5.2/5.3), 4 (5.1), 5 (4.1/4.2) and the ≤23-ops graph (4.3). Nothing verifies or asserts a single rewrite engine or backend-identical bound plan — no cross-backend `&[BoundOp]` equality test appears.

**M-8. `op.rs:166` stale doc is diagnosed and never fixed.** planA:9 notes the header says "The four generators" over a 5-variant enum; verified at `proxima-tensor/src/op.rs:166` above `pub enum Op` at `:175`. The one-RISC claim's own doc contradicts the RISC's cardinality; no step touches it.

**M-9. 0.1's expected N contradicts today's sealed cell.** planA:79 asserts "5 ours-runs × 24 `token_breakdown_metal` lines each = **120 lines**". R13's sealed cell reports "steps 1-7 (7 per run)" with generated text "Here is a simple Python function that returns". `decode_loop_max_tokens()` defaults to 24 (`bind.rs:2719-2723`) but the loop stops on eos. An N assertion of 120 will read RED on a correct run. The same 24-step assumption propagates to 2.1 ("23 of 24 steps", planA:275) and 2.3 ("5 × 24 = 120", planA:299).

**M-10. 0.3's N assertion compares incomparable feature sets.** planA:103: "`cargo nextest run -p omega --features metal,metal-wide-cooperative-reduce` reports **the same test count as `--all-features` on main**". `--all-features` additionally enables `cuda`, `wgpu-backend`, `metal-tiled-gemm`, `vulkan`, `npu`, `ane` (`omega/Cargo.toml [features]`), all of which contribute tests. The two counts cannot be equal.

**M-11. Four steps pre-register a same-rung prediction, which is not a prediction.** 0.2 "nano → nano, no rung climb" (planA:92), 1.1 "micro → micro, sweep is the rung" (planA:186), 3.1 "nano → nano" (planA:319), 5.1 "nano → nano" (planA:401), 1.6 "nano → nano" (planA:256). The ladder rule is a prediction one rung *ahead*; a same-rung prediction cannot be missed by the climb it is supposed to gate.

## 5. Hidden coupling

**H-1. `proxima-tensor/src/cpu.rs:346-354` carries the twin strict check and step 4.2 never names it.**
```rust
let expected = element_count(shapes.of(*node));
if data.len() != expected { return Err(TensorError::InputSizeMismatch { ... }); }
```
The CPU arm takes `&[&[f32]]`, not `QuantizedBlock`, so the plan's declared-over-allocated field cannot reach it. The §14 oracle for 4.2 and 4.3 is the CPU decode test asserting token 2651 / `"known"` (`bind.rs:2797-2804`), which runs `cpu::evaluate`. An over-allocated KV buffer is rejected there. Step 4.2's `open` list (planA:360) cites `metal.rs:991-1000` and omits `cpu.rs:346-356`, which ledger R11 names.

**H-2. `AlignedBuffer` forces over-allocation and needs a Metal-only page size.** `proxima-tensor/src/align.rs:38-41` doc: "must size its tensor input to `buffer.len()`, not the value it originally asked for"; `new(min_elements, page_size)` (`:69`) rounds to `next_multiple_of(page_size).max(page_size)` and the doc says page_size must come from "e.g. `omega::metal::page_size`". So (a) the over-allocation in H-1/R-3 is structural, not a bucket choice, and (b) `proxima-model-interop`'s `LayerCache` (a crate whose `metal` feature is optional, `Cargo.toml`: `metal = ["dep:omega", "std"]`) must obtain a page size on the non-Metal build. Neither is named.

**H-3. `classify_kind` is not free-standing.** It is called at `omega/src/metal.rs:709` alongside `diagnose_kind` at `:710`, inside the per-op timed path, and its doc at `:777-783` ties it to `omega/examples/real_forward_packed_probe.rs` (ROW 85). Step 3.1 says "`classify_kind` is **deleted**" (planA:23, :317) and lists blast radius as "`omega/src/msl.rs`, `omega/src/metal.rs` instrument block" (planA:322). The example is not named.

**H-4. `append_mistral_cached_layer` has one call site but a wide doc-level fan-out.** `proxima-tensor/src/spec.rs:6282` is the sole caller; 16 other sites reference it by doc link, including the Qwen3.5 dense-attention counterpart at `:2867-2891` and the split-half RoPE path at `:6465`, plus the MoE counterpart at `:3445`. 4.3 names Qwen3.5 re-parity (planA:379) but the "even/odd RoPE split collapses in the same move" claim (planA:374) is asserted against a checkpoint whose split-half RoPE path (`:6465`) is documented as *not* going through this function.

**H-5. 4.2's counters are misfiled.** planA:367 says `kv_cache_upload_bytes` is "in `token_breakdown_metal`, `generate.rs:1734-1745`". It is in `token_breakdown` at `generate.rs:1657`, and `greedy_pick_ms` (6.2's counter) at `:1659`. `token_breakdown_metal`'s field list (`generate.rs:1728-1732`) contains neither.

**H-6. wgpu driver.** 5.3 gives WGSL the packed-row and tiled-GEMM routes and predicts "the `wgpu-backend` arm runs the openchat decode plan end-to-end for the first time" (planA:427). `omega/src/wgpu_driver.rs` (872 lines, R5) is never opened, never named in blast radius (planA:430 lists `cuda.rs, wgsl.rs, error.rs`), and `omega-gate.sh [2/6]` builds `--all-targets --all-features`, so a wgpu compile break lands in the gate for every subsequent step.

## 6. Observability

**V-1. Every named counter exists on main.** Verified present in `omega/src/metal.rs` and/or `proxima-model-interop/src/generate.rs`: `kv_cache_upload_bytes`, `greedy_pick_ms`, `greedy_pick_started`, `op_setup_ticks`, `gpu_exec_ticks`, `readback_calls`, `encode_dispatch_ms`, `device_allocated_bytes`, `phys_footprint_bytes`, `nocopy_reuses`, `plan_hits`, `plan_misses`, `plan_cache_len`, `block_upload_bytes`, `readback_bytes`. `omega.route.decisions` is correctly declared new (planA:323).

**V-2. But `plan_hits` does not mean what the plan's counter list implies.** There is exactly one such field: `pub(crate) plan_hits: usize` at `generate.rs:905`, incremented at `:968`, printed at `:1764` (`token_breakdown_metal`) and `bind.rs:3046` (`metal_decode_summary`). Both prints read the same field. See hazard (f).

**V-3. 0.3's counter cannot prove 0.3's mechanism.** planA:108 concedes it: the only available split is `classify_kind`'s `reduce-cooperative` bucket, which the same document says relabels when a body changes. The step's prediction is "reduce-family GPU time falls ≥15%" — a per-family number that only the per-op diagnostic profile (`execute_plan_op_timed`, `metal.rs:654-761`) produces, and R13 records that mode inflating Σ by 7.3% over batched. Neither the mode nor the inflation correction is stated.

**V-4. 1.1's GB/s is undefined.** The kill is "if the copy probe reads below llama.cpp's achieved 228.9 GB/s" (planA:187). A streaming copy touches bytes twice (read + write); 228.9 GB/s is a weights-read-only figure derived from 3.9996 GB / 17.470 ms (R1, MEMORY). Whether the probe's denominator is read bytes or read+write bytes decides the kill by a factor of two, and the plan does not say.

**V-5. 3.2's gate assertion has no counterpart on main.** "census rows sum == `encode_dispatch_calls`" (planA:330) requires the census to be emitted from the same code path that increments `encode_dispatch_calls` (`metal.rs`, encode loop `execute_plan` `:449-568`). 3.1 places `record_route` in emission (`msl.rs`), which runs once per *distinct kernel* through the pipeline cache — `pipeline_hits`/`pipeline_misses` exist precisely because emission is cached. A per-emission census cannot sum to a per-dispatch count without an explicit per-op record site, which is not specified.

## 7. Scope discipline

**S-1. Step 0.1 re-measures what R13 already sealed on the same commit.** R13 is MEASURED 2026-09-03 on `4be2f3a` with interleaved arms and raw logs at `scratchpad/baseline-2026-09-03/`, reporting op_count 1196, `plan_hits=0 plan_misses=8`, per-phase means. Q1 in the plan's open-questions table (planA:576) asks whether the 9 commits made R1's 11.4 ms stale — R13 answers it (prepare 1.97, op_setup 3.9, wall − gpu_exec = 11.0). The step is not wrong to re-seal on a quiet box (R13's was taken on a loaded box, load 4.7-5.7), but the plan does not say that is why, because it never cites R13.

**S-2. No dead lever is re-proposed.** Checked against R4 and R12: nsg=2 does not appear as a step; encoder churn does not; per-dispatch fixed cost does not; `-t` threading does not; rematerialize-all does not (6.1 is explicitly a *re-measure* of the `elements < 247` subset with a kill at <2× CoV, planA:447); the quarantine heuristic does not. `PROXIMA_ORCH_THREADS` is explicitly excluded (planA:562).

**S-3. Step 1.4 and 1.5 are outside the ask's mass but inside the brief.** 1.4 (`-fa 1`) is a denominator-integrity step; 1.5 (non-decode arms) answers the brief's "ort and torch can beat us" clause. Both justified.

**S-4. 5.2 is the largest-blast-radius step and removes zero mass by its own statement** (planA:394, planA:414 predicts `gpu_exec_ms` unchanged). It is required by brief item 3, so it is in scope — but it is sequenced after 4.4's re-seal and before nothing, so its byte-exact golden is the only thing standing between an 8479-line rewrite and the kernel. That is stated (planA:417) and is the correct firewall; the finding is that 3.1, which changes route classification with the same kind of exposure, does not get one.

---

## Hazards

**(a) MEMORY baselines used as delta anchors instead of the R13 sealed cell — FOUND, four instances.**
- planA:288 `op_setup_ms` "falls from ~4.4" → R1 MEMORY `op_setup 4.394`; R13 sealed = **3.9**.
- planA:275 `prepare_ms` "falls from ~2.087 toward ~0.09 (2.087/23)" → R1 MEMORY `prepare 2.087`; R13 sealed = **1.97**. The arithmetic `2.087/23` is built on the superseded number.
- planA:459 `readback_ms` "falls from ~0.297 (R1)" → R13 sealed = **0.22**.
- planA:363 "whole cell moves ≥3% (R2 attributes ~3.3% to KV re-upload)" → R2 is MEMORY; R13's gap decomposition (ledger:267-269) has no KV re-upload term at all.
- Root: R13 is cited zero times in the plan (counts in M-1).

**(b) 4.3's kill vs the ledger's own dispatch-halving finding — FOUND, inconsistent in two ways.** The step does quote the control correctly (planA:377: 1194→616, wall 51.571→51.535, GPU 35.117→39.841). But the operational criterion it derives — "killed if `gpu_exec_ms` **rises** at all" — is set below the measurement's own noise: R13's `gpu_exec_ms` CoV is 0.7% across runs, so a 0.5% rise kills a step that moved nothing. Second, the ledger's finding was about **wall** (0.07%), and the success shape at planA:377 is "dispatches down **and** `gpu_exec_ms` flat-or-down" — wall is absent from both the prediction (planA:376, counts only) and the success shape. A 4.3 that reduces dispatches to 940 with flat gpu_exec and flat wall passes every stated criterion while reproducing exactly the parallel branch's outcome.

**(c) 2.1's mask composition does not cover the cached-range tail — FOUND.** `causal_mask` (`spec.rs:823-845`) builds two `Op::Iota { extent: Extent::Symbolic(0) }` and one `Greater`, producing `is_future` shaped `[s,t]` — both axes are symbol 0, the *new*-token count. The cache axis is symbol 1 (`spec.rs:6216-6245`, three `Extent::Symbolic(1)` leaves at `:6218`, `:6228`, `:6238`). The doc at `spec.rs:2314-2319` states the design's premise verbatim: "`is_future` … sized `[s,w]` since `w` and `s` share symbol 0's extent … so the cached block never needs masking at all", and — the sentence the plan does not quote — "The masking-only-within-`s,w` asymmetry is what makes this correct **without a `cached_len` scalar**". Bucketing makes the cached extent `bucket ≥ cached_len`, so the cached block now *does* need masking, over symbol 1, against a runtime `cached_len` the graph deliberately does not carry. That requires a new Iota over symbol 1, a new scalar input leaf, and a new `Select` — none of which is the "causal-mask machinery that already exists" the design claims (planA:273), and none of which is "zero IR change" (planA:19, planA:268).

**(d) 4.2's relaxation leaks into weights; the declared field is unadjudicated — FOUND, both halves.** The check at `omega/src/metal.rs:991-1001` runs over every `(node, block)` pair in `block_nodes.zip(blocks)`, weights included, so `found >= expected` is in the weight path unless the declaration field excludes it — the field is the only firewall and its sad-path test as written (planA:362, "under-allocated block still rejected") does not exercise an over-declared weight. On the type question: `QuantizedBlock` is a public enum at `proxima-tensor/src/cpu.rs:3084`, i.e. proxima-tensor library surface. The brief binds the pipe question and the relocation question before ANY new type in omega or proxima-tensor. Step 4.2 adds "a new field on `QuantizedBlock`" (planA:361) and runs neither question on it. Step 7.2 runs the relocation question on the *contingency* variant (planA:488) — so the plan demonstrates it knows the gate applies and skips it on the primary path.

**(e) 3.1 is not behaviour-preserving by inspection — FOUND (unproven, not disproven).** `kernel_cache_key` (`omega/src/msl.rs:735-775`) pushes exactly three route characters, `'G'` if `tiled_gemm_block(...).is_some()`, else `'B'` if `packed_row_block(...).is_some()`, else `'S'` — with `'S'` also covering every non-Reduce kind. The comment at `:755-758` states the ordering is safe because `tiled_gemm_block` "only ever returns `Some` when `packed_row_block` also would (it is built ON TOP of that same gate)". So an 8-variant `KernelRoute` feeding `kernel_cache_key` must collapse `{Elementwise, Scan, Iota, Constant, ReduceSerial, ReduceCooperative}` → `'S'` and preserve G-before-B. Nothing in 3.1 asserts key stability: its prediction is about census counts vs `classify_kind` (planA:319), its counter is the new census (planA:323), and the golden-source test arrives only at 5.2. The step is also unflagged (planA:321). Additionally the prediction "reproduces `classify_kind`'s current bucket counts **exactly**" is malformed at the codomain: `classify_kind` emits 9 labels including `reduce-generic-scalar` and `reduce-unclassified` (`metal.rs:786-826`), the latter produced by `Err(_)` from `emit()`, for which no `KernelRoute` variant exists in the 8 listed.

**(f) The interop plan-cache counter vs the per-step field — FOUND, and the ledger's own distinction does not hold on main.** There is exactly one counter: `generate.rs:905` `pub(crate) plan_hits: usize`, incremented at `:968` inside `resolve_plan`, printed at `generate.rs:1764` (`token_breakdown_metal`) and at `bind.rs:3046` (`metal_decode_summary`) — the same field, read twice. R13's parenthetical ("the per-step `plan_hits=10,20..70` field in `token_breakdown_metal` is a different counter … it is not the interop plan cache") is not supported by any second definition on main; `grep -rn plan_hits` over all crates returns only these sites. The plan never asserts the distinction either way, never names `metal_decode_summary` (0 occurrences), and — the consequence — never notices that the counter it plans to move from 0 to 23 is the subject of `assert_eq!(runtime.plan_hits, 0)` at `bind.rs:3052-3055` (see R-1).

**(g) Worktree names collision-free; branch names are NOT — FOUND.** Against the live list (`git worktree list`, 68 entries: `proxima-wt-3ax … proxima-wt-verify`), none of the plan's 26 directory names collides. But three branches the plan assigns are already checked out, and `git worktree add` refuses a branch checked out elsewhere:
- `perf/kv-device-resident` → in use at `/Users/brianbruggeman/repos/slot-0/proxima-wt-drive` (2b95210). Plan 4.2 assigns it to `proxima-wt-persist` (planA:359).
- `perf/attention-single-range` → in use at `proxima-wt-merge` (2b95210). Plan 4.3 assigns it to `proxima-wt-onerange` (planA:372).
- `bench/sealed-pass` → in use at `proxima-wt-seal` (fb61d04). Plan 0.8 assigns it to `proxima-wt-qbox` (planA:164).
Separately, `proxima-wt-qbox` is assigned four different branches across three phases (0.1 `bench/quiet-seal-main`, 0.8 `bench/sealed-pass`, 2.3 `bench/quiet-seal-phase2`, 4.4 `bench/quiet-seal-phase4`) — one worktree holds one branch, which serializes steps the plan schedules in parallel (see O-1).

**(h) Time estimates: NOT FOUND — clean.** `grep -niE "\b(hours?|days?|weeks?|minutes?|sprint|timeline|eta|effort|man-|person-)\b"` over planA.md returns zero hits. Verdict language: essentially clean. The residual instances are (i) "green with its asserted test count" (planA:128) naming a script's own output — and `scripts/omega-gate.sh:37-47` does assert a nonzero count, though it asserts *nonzero*, not a specific N, so "its asserted test count" overstates what the gate provides; (ii) "is not a win" / "the success shape is" (planA:377), which is a pre-registered rule, not a verdict; (iii) "proves the window is clean" (planA:190, :566), a claim attached to a stated assertion (`readback_bytes == 0`).

---

## Summary — the three most consequential findings

The plan is anchored on the wrong evidence and does not know it: **R13, today's sealed cell on the exact commit the plan targets, is cited zero times**, while R1/R2/R3 MEMORY rows are cited twenty-one times and supply four of the plan's per-stage baselines (`op_setup` 4.4 vs sealed 3.9, `prepare` 2.087 vs 1.97, `readback` 0.297 vs 0.22, KV 3.3% vs a decomposition that has no KV term) — and R13's carded prerequisite, "fix `op_profile` byte accounting before any GB/s row is written" (`operand_bytes` at `omega/src/metal.rs:694` reporting the whole 4.14 GB checkpoint buffer, consumed at `generate.rs:109-212`), is absent from the plan entirely while three steps write GB/s rows. Second, **the plan's own bench harness contradicts its Phase 2 goal**: `bind.rs:3052-3055` asserts `plan_hits == 0` with the comment "no `(new_count, cached_len)` shape can repeat within one call", and step 2.1's pre-registered success is `plan_hits == 23` — the test the plan greps for that number is the test that fails when it appears, and steps 2.2 through 6.2 all re-prove through it. Third, **step 2.1's "zero IR change" is not zero**: `causal_mask` (`spec.rs:823-845`) builds `is_future` over symbol 0 on both axes, and `spec.rs:2314-2319` says in the source that the design is correct precisely *because* it carries no `cached_len` scalar and never masks the cached block — bucketing requires a symbol-1 Iota and a runtime `cached_len` leaf that do not exist, which relocates 2.1 from "graph-shape-preserving driver fix" to a graph change whose CPU parity oracle (`bind.rs:2797-2804`, token 2651/"known") runs through a second strict size check at `cpu.rs:346-354` that step 4.2 never names.