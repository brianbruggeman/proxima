# GPU parity plan — proxima-tensor through omega, ONE RISC

Everything below is verified on `/Users/brianbruggeman/repos/slot-0/proxima` at `4be2f3a` (`git rev-parse --short HEAD` → `4be2f3a`, working tree clean but for two untracked dirs: `docs/bench-campaigns/2026-09-03-gpu-one-risc/`, `proxima-onnx/scripts/torch_reference/venv/`). `pwd` = the main checkout; no `proxima-wt-*` path was entered.

---

## 0. Three ledger corrections found by re-reading, before the plan

These change what Luna must type. They are not stylistic.

**C-A. `bind.rs` is ambiguous and the ledger's cites split across two files.** There are two: `proxima-tensor/src/bind.rs` (3089 lines) and `proxima-model-interop/src/bind.rs` (4257 lines). R5/R15/R16's `bind.rs:200-215`, `:95-98`, `:1594-1606`, `:1618-1647`, `:1011` are **proxima-tensor**. R13/R14's `bind.rs:2719`, `:3002`, `:3046`, `:3052-3055`, `:3084` are **proxima-model-interop**. `grep -rn "plan_hits" proxima-tensor/ omega/` returns **zero** hits in `proxima-tensor/src/bind.rs` — the only hits are `proxima-model-interop/src/{generate,bind}.rs`, `omega/src/metal.rs:346,2191`, and discipline prose. A hands model opening `proxima-tensor/src/bind.rs:3052` finds `a_masked_window_reduce_with_a_non_windowed_axis_matches_a_direct_window_read`, an unrelated test. **Every card below spells the crate path.**

**C-B. The `plan_hits` harness has TWO assertions, not one, and it never runs in any gate.** At `proxima-model-interop/src/bind.rs:3052-3054` `assert_eq!(runtime.plan_hits, 0, "cached_len grows every decode step, so no (new_count, cached_len) shape can repeat within one call")` and at `:3056-3058` `assert_eq!(runtime.plan_misses, forward_calls_taken, "every forward step builds exactly one new plan when none can ever be reused")`. R13 named only the first. The test is `#[cfg(feature = "metal")]` **and** `#[ignore = "depends on a host-local openchat gguf checkout outside this repo, and a real Metal device"]` (`:3001`), and lives in **proxima-model-interop**, so `scripts/omega-gate.sh` never executes it. Any card that makes the plan hit must invert **both** assertions in the same change, and must be re-proven with `--run-ignored all`.

**C-C. `omega-gate.sh` has six steps, not four, and two of them are count-asserted.** Read in full: `[1/6]` `--no-default-features --features alloc` build; `[2/6]` `--all-targets --all-features` build; `[3/6]` nextest `--all-features` with `ran_count` asserted nonzero; `[4/6]` clippy pedantic on **two** arms (`--lib --no-default-features --features alloc`, then `--all-targets --all-features`); `[5/6]` `cargo doc` on two arms; `[6/6]` `cargo test --doc --all-features` with `passed_count` asserted nonzero. R15 stopped at `[4/6]`. This confirms R18's warning with a second consequence: mutually-exclusive features + `compile_error!` break `[2/6]`, `[3/6]`, `[4/6]`-arm-2, `[5/6]`-arm-2 **and** `[6/6]` — five of six steps, not one.

Two further host facts, re-verified: `which flock` → `flock not found`, `/opt/homebrew/bin/flock` absent (R18 reproduces); `python3 -c "import fcntl; print(hasattr(fcntl,'flock'))"` → `True`, so the shim is viable. And `git worktree list | wc -l` → **68** (not ~70), `git branch --list | wc -l` → **72**; a nested checkout `.claude/worktrees/agent-a6600bd71b04b05fe` lives **inside** the repo, so every repo-wide `grep -rn` must be scoped to crate directories or it double-counts.

---

## 1. Diagnosis, bound to file:line — one class confirmed, one ordering overturned

### 1.1 What is confirmed

**Class 2 (kernel body) is confirmed and is the largest movable mass.** `R13 MEASURED`: reduce-packed-row-blocked = 225 ops, **44.450 ms**, 197,556 ns/op, against reduce-cooperative 385 ops / 9.113 ms and elementwise 547 ops / 7.350 ms. Gap decomposition today (67.92 − 17.52 = **50.40 ms** `MEASURED`): Q4_K matvec above stream rate ≈ **26.9 ms**, non-matmul GPU ≈ **16.6**, orchestration ≈ **11.0**, irreducible stream 17.5. The mechanism is read on main: `classify_kind` (`omega/src/metal.rs:785-826`) and the packed body reached through `push_cooperative_reduce_body` → `packed_row_block` (`omega/src/msl.rs:1235`); the incumbent's contrasting body is `kernel_mul_mv_q4_K_f32_impl<4,2,32>` (`ggml-metal.metal:5086-5193`, mask-without-shift, 1/16 and 1/256 folded into scale at `:5171-5175`) `R8 READ`.

**Class 3 (unauditable lowering) is confirmed, and has already failed twice.** `classify_kind`'s own doc at `omega/src/metal.rs:777-783` admits the routing decision "is not exposed as its own accessor," and its body carries a comment recording a prior silent relabeling: ROW 113's weight-staging fix made `push_tiled_gemm_body` emit `q4k_run8` too, so the row-blocked arm's `"q4k_run8(blk"` match "now fires on BOTH kernel bodies." `R12 READ` records a second: the paired body was mislabeled `reduce-cooperative`, 9/601 → 225/385 after fix. **A labeling instrument with two proven silent relabelings cannot carry a scorecard.** Geometry constants confirmed bare: `PACKED_ROWS_PER_GROUP = 4` (`omega/src/msl.rs:1017`), `TILE_DIM = 8` (`:1030`), `TILED_GEMM_NSG = 4` (`:1046`); `omega/omega-runtime.toml` read in full contains exactly one section, `[tiled_gemm]` (min_tokens/block_m/block_n/block_k), and `omega/build.rs`'s `emit_sizing_consts` emits those consts **only** when `CARGO_FEATURE_METAL_TILED_GEMM` is set. §12 violation confirmed.

**One-RISC item 1 is false on main today.** `omega/src/metal.rs:1013` runs `correct_packed_matmul_layouts(&mut resolved, &packed_operands.keys().copied().collect());` **after** `bind` at `:1003`. `grep -rn "correct_packed_matmul_layouts"` returns hits only in `omega/src/metal.rs`, `proxima-tensor/src/{bind,lib}.rs` — **`proxima-tensor/src/cpu.rs` has zero hits**. The Metal plan is not the CPU plan. `R16 READ` confirmed verbatim.

**Write placement is one line away, and the source says so.** `project_output_shape` (`proxima-tensor/src/shape.rs:469-485`) rejects everything but `[term] if term.coeff == 1` with `NotLowerable { reason: "reduce output maps must be pure projections in v1" }`. `append_mistral_cached_layer`'s doc (`proxima-tensor/src/spec.rs:2306-2317`) names this as **the** cause of the duplication, verbatim: "Two `Op::Reduce` blocks — one per source — combine through online-softmax arithmetic … rather than a literal concatenation: `Reduce::out_map` must stay a pure projection (`shape::project_output_shape`'s own doc), so nothing upstream of a reduce can splice two tensors into one axis." The write-side destination already exists: `Layout { base: i64, strides }` (`proxima-tensor/src/bind.rs:95-98`) on every bound `Reduce`, emitted as `long out_base` and consumed at `omega/src/msl.rs:2216, 2361, 2734, 3079, 3460, 3541`.

### 1.2 What is overturned — the brief's ordering of mass

The brief orders the three classes "in this order of mass," putting graph duplication first. **The evidence refutes that ordering, by a clean one-variable toggle in a single tree.** `R12 MEASURED`, same worktree, feature off vs on:

| arm | wall ms/tok | GPU ms/tok | dispatches |
|---|---|---|---|
| feature-off control (ROW 267) | 51.571 (CoV 1.75%) | 35.117 (2.00%) | 1194 |
| feature-on, consumer index, paired Q4_K (ROW 262) | 51.535 (2.16%) | 39.841 (0.98%) | 616 |

Halving the dispatch count moved wall by **0.036 ms (0.07%)**, inside both arms' CoV, and made **GPU 4.7 ms worse**. Dispatch count is not the mechanism. Meanwhile that same tree's feature-off control already carried the paired Q4_K body and sits at 35.1 GPU against main's 56.93 `MEASURED R13` — **the kernel body alone is worth ~21.8 ms `DERIVED` (R12 35.117 vs R13 56.93; cross-tree, so it is a hypothesis to re-measure, not a finding).**

Therefore: **the graph is a correctness-and-auditability problem and an enabler, not the leading performance lever.** Its value is (a) making in-place KV expressible, which collapses the 97-output liveness partition (`proxima-model-interop/src/generate.rs:1393-1400` pushes `logits_root` + 3 roots × 32 layers; `bound_op_retirement` at `omega/src/metal.rs:1128-1147` never retires an output, `!outputs.contains(&node)`), and (b) removing coop/elementwise ops, whose 16.6 ms is real. It is sequenced **after** the body and the orchestration, and its prediction is written against the coop+elementwise buckets, never against dispatch count.

Two consequences the plan must carry: the brief's diagnosis (1) is **demoted**, and the parallel branch's `BoundOpKind::CachedAttention` is **rejected on its own numbers** — it is a fifth bound kind that costs 4.7 ms of GPU to buy a dispatch reduction worth zero.

### 1.3 A measurement-provenance defect nobody has flagged

`gpu_exec_ms` on the batched path is **not** a GPU timestamp. `omega/src/metal.rs:545-554`: `let gpu_exec_started = read_ticks(); command_buffer.commit(); command_buffer.waitUntilCompleted();` then `counter!(GPU_EXEC_TICKS, elapsed_ticks(gpu_exec_started))` — host ticks wrapping commit **and** wait. Only the op-timed twin uses real GPU time: `omega/src/metal.rs:733-734`, `((command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1e9)`. So R13's "kernel-only 56.93, 3.25x" is a host-side wall figure that includes commit and wait overhead, and the 7.3% per-op-mode excess is measured against it. Every card that predicts on `gpu_exec` must say which clock it means.

---

## 2. One-RISC binding — the brief's eight items, one card each

| # | Item (brief §"What one RISC means") | Card | Status of the item on main |
|---|---|---|---|
| 1 | ONE bound plan from ONE rewrite engine, identical for every backend | **C5.1** | FALSE — `omega/src/metal.rs:1013` rewrites post-bind; `proxima-tensor/src/cpu.rs` does not |
| 2 | ONE route decision, first-class enum, decided BEFORE emission | **C1.1** | FALSE — decided inside `emit`, recovered by substring (`omega/src/metal.rs:785-826`) |
| 3 | …censused `(NodeId, reason)` | **C1.2** | ABSENT — `WIDTH_TILE_DECLINE` (`proxima-tensor/src/instrument.rs:842`) is the only census pattern and it is `Mutex`-based |
| 4 | ONE emitter core over the 4 kinds, backend text only | **C6.1** | FALSE — ~78 near-duplicates across msl/wgsl/cuda; 3 shared types + 2 fns (`R5 READ`) |
| 5 | Every backend covers every kind | **C6.2** | FALSE — `EmitError::ScatterNotSupported` raised at `omega/src/{msl.rs:933, wgsl.rs:364, cuda.rs:241}`; CUDA also rejects Iota/Constant |
| 6 | ONE sizing config owning every geometry constant | **C2.1** | FALSE — three bare consts in `omega/src/msl.rs`; toml has only `[tiled_gemm]` |
| 7 | Write placement via existing `out_map`/`out_layout.base` + driver alias, NOT a new Op | **C5.2 + C5.3 + C5.4** | BLOCKED at `proxima-tensor/src/shape.rs:469-485`; destination `Layout.base` already exists |
| 8 | Llama graph at ≤23 real ops/layer | **C5.5** | 37/layer, 1196 dispatches; incumbent 23/layer ≈ 740 (`R8 READ`) |

---

## 3. Global protocol — binding on every card

**Tiers, no hybrids.** Exactly one tier per card, declared in the card header.
- **hands (Luna)** — types verbatim commands, opens named `file:line`, records the printed N into the row placeholder. Makes no design judgment, chooses nothing, writes no prose beyond the row template. If a command errors or an N is absent, hands STOPS and reports; it never improvises a substitute command.
- **worker** — writes code inside one named file set, may choose among pre-named alternatives, may not add a type, feature, or file not named on the card.
- **judge** — adjudicates a contested design (which of two bodies; accept/reject a type), reads both arms' evidence, writes the decision and its constraint trace. Produces no code.

A card is never two tiers. A card that would need hands to decide is split.

**Worktree + branch + target per card.** Every card names its own `proxima-wt-<name>` (none of the 68 existing), its own branch (none of the 72 existing), and `CARGO_TARGET_DIR=<worktree>/target`. Worktree creation is an owner-authorized action, not a Luna action; the card names it, the owner creates it.

**Measurement mutex.** `flock(1)` does not exist on this host (`which flock` → not found) `R18 READ`, so the mutex is **card one** and nothing that runs a process may precede it. Every measuring command in every later card is welded as `<worktree>/scripts/gpu-measure-lock.sh -- <command>`. One measurer on the box, always. `scripts/omega-gate.sh` steps `[2/6]`/`[3/6]` build and run `--all-features`, which includes Metal tests, so **the gate itself takes the lock**.

**N==0 is RED everywhere.** Every card states an expected N. A run that prints no N, or prints zero, is RED — never "passed, nothing to do." This mirrors the gate's own two count assertions (`ran_count`, `passed_count`) and extends them to every card.

**One-rung predictions.** Ladder: **nano** (one kernel body in isolation, `omega/examples/q4k_matvec_probe.rs`) → **micro** (one family through a probe, `omega/examples/real_forward_packed_probe.rs`) → **milli** (one real decode step per-op, `profiles_one_real_decode_step_by_per_op_gpu_time`, `proxima-model-interop/src/bind.rs:3084`) → **bench** (full interleaved decode cell vs llama.cpp-Metal, `runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache`, `:3002`). A card measuring at rung k pre-registers a number for rung k+1 **only**. A miss is decomposed on the row into *inconsistency* (the two rungs disagree about the same quantity) vs *understanding-gap* (both are right; the model of the mechanism is wrong), and each spawns a named work item. A miss kills the climb; the card does not proceed to k+2.

**Interleaved arms.** Never A A A B B B. Always `A B A B A B`, ≥3 pairs, CoV reported per arm. The incumbent arm is llama.cpp-Metal at its own shape set (`/Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench`, checkout `b25346221`) on every compare row.

**Memory is a KILL, with a byte formula, on every card that runs a process.** Owner rule 2026-09-03. The formula:

```
expected_device_bytes = CHECKPOINT_BYTES + KV_BYTES + SCRATCH_BYTES
  CHECKPOINT_BYTES = 4,140,417,024        [MEASURED R13, the one no-copy mapping buffer]
  KV_BYTES         = 262,144 * capacity_tokens
                     [DERIVED R17: 32 layers * (k_even 8*64*4 + k_odd 8*64*4 + v 8*128*4) = 32 * 8192]
  SCRATCH_BYTES    = per-op output + uniform buffers, steady-state observed
                     total device_allocated_bytes 4.152-4.163 GB   [MEASURED R13]
```

KILL if **any** holds: `device_allocated_bytes > 4.40e9` at any steady step (ceiling = MEASURED 4.163e9 + 5.7%); OR `device_allocated_bytes` rises **> 2 MB/token** across steps 3..7 (MEASURED +1–2 MB/token); OR task RSS at steady state **> 120 MB** (MEASURED 48–66 MB); OR any KV capacity is derived from `ServingConfig::context_length` (`proxima-model-interop/src/serving.rs:161` = `131_072`; × 262,144 = **34,359,738,368 B**, the reproduced 34 GB trap `R18 READ`). Capacity is a build-time key with a build-time byte assertion, never a runtime derivation.

**Row placeholders.** Main's last row is **ROW 233** at `proxima-tensor/docs/discipline.md:18736` (file is 18766 lines) `R10 READ`, and `grep -c "3.54x\|17.470\|228.9"` over that file returns **0** — main's log does not know the 2026-09-02 session happened. Branches therefore carry `## ROW <NEXT> -- <title>` literally. Numbers are assigned at land time from main. Row format follows ROW 233: `## ROW <N> -- <title>`, bolded findings, tables, a mechanism paragraph, and a closing `**Gates:**` line naming gate results, branch, and commit.

**No time estimates.** Anywhere. Not "quick," not "a day," not ordering language that implies duration.

**No verdicts.** Cards produce evidence rows. "Faster" is not an output; a number with a clock, a CoV, an arm and a provenance tag is.

**No commit without owner authorization.** Every commit is a green bisect point; conventional, one lowercase line under 72 chars, no trailing period. Cards prepare commits and stop.

**Type discipline.** No `Box<dyn>`, §20. No new `Op`, `BoundOpKind`, `ScalarOp`, or `IndexMap` variant — the closed sets stay closed (`proxima-tensor/src/op.rs:51-53`: `ScalarOp` "is the one closed set in this crate that stays closed"; verified 17 variants, `Greater` and `Equal` present, **no `GreaterEqual`**). Any new type answers both binary questions **on the card, in writing**, or it is not minted (§4).

---

## 4. The cards

### PHASE 0 — seal, and land what is already measured

---

#### C0.1 — the measurement mutex (CARD ONE; nothing that runs a process precedes it)

- **tier:** worker
- **depends_on:** —
- **worktree/branch/target:** `proxima-wt-lock` / `perf/gpu-measure-lock` / `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-lock/target`
- **opens:** `scripts/omega-gate.sh:36-46` (the `[2/6]`/`[3/6]` all-features steps that must take the lock); `scripts/proxima-tensor-gate.sh` (same, tensor side)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-lock && which flock; echo "flock-absent-exit=$?"
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-lock && python3 -c "import fcntl,os; print('fcntl.flock', hasattr(fcntl,'flock'))"
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-lock && bash scripts/gpu-measure-lock.sh -- echo lock-acquired
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-lock && bash scripts/gpu-measure-lock.sh -- sleep 5 & sleep 1; cd /Users/brianbruggeman/repos/slot-0/proxima-wt-lock && bash scripts/gpu-measure-lock.sh --timeout 2 -- echo second-should-not-print; echo "contended-exit=$?"
  ```
  The shim: `python3` acquiring `fcntl.flock(fd, LOCK_EX)` on a fixed host path (`/tmp/proxima-gpu-measure.lock`), then `os.execvp` into the wrapped command so the lock is held by the measuring process itself and released by kernel on exit — no cleanup path to leak. `--timeout` uses `LOCK_NB` in a bounded retry.
- **expect (N):** `lock-acquired` printed exactly **1** time; contended run prints `second-should-not-print` exactly **0** times and exits nonzero. N==0 on the first (no acquisition) is RED.
- **predict (one rung, nano→micro):** the shim adds **< 40 ms** to a wrapped `echo`, so wrapping the micro-rung probe (`real_forward_packed_probe`) changes its reported family total by **< 0.1%**.
- **kill:** the contended run prints `second-should-not-print` — the mutex does not exclude, and every later cell is unsound. Stop the whole plan here.
- **memory gate:** n/a (no model process). The shim must not read the checkpoint.
- **rollback:** delete `scripts/gpu-measure-lock.sh`; no other file is touched.
- **blast:** one new file under `scripts/`; two gate scripts gain a wrapper line. Nothing in any crate. No feature, no type.
- **observe:** the lock file's presence and the exit codes; `ls -l /tmp/proxima-gpu-measure.lock`.
- **reprove:** `cd /Users/brianbruggeman/repos/slot-0/proxima-wt-lock && bash scripts/gpu-measure-lock.sh -- echo lock-acquired`
- **log-row title:** `## ROW <NEXT> -- the box had no mutex: flock(1) is absent on this host, and every "quiet box" cell before this one was unprotected`

---

#### C0.2 — re-seal the anchor cell on a quiet box

- **tier:** hands
- **depends_on:** C0.1
- **worktree/branch/target:** `proxima-wt-baseline` / `bench/gpu-anchor-2026-09-03` / `.../proxima-wt-baseline/target`
- **opens:** `proxima-model-interop/src/bind.rs:3002` (bench-rung harness), `:3052-3058` (both assertions), `:2719` (`PROXIMA_MAX_TOKENS`, default 24)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-baseline && pgrep -x cdb-daemon; uptime
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-baseline && bash scripts/gpu-measure-lock.sh -- /Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench -m ~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf -n 32 -p 0 -r 5 -t 8 -ngl 99
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-baseline && PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh -- cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' --no-capture
  ```
  Repeat the two measuring commands **interleaved A B A B A B**, three pairs.
- **expect (N):** `op_count` = **1196**; `plan_hits=0 plan_misses=8`; `generated_text` identical across all three ours-runs; llama-bench tg32 three means within CoV **< 1.5%**. Any N absent → RED.
- **predict (one rung, bench):** llama.cpp mean lands within **±1.0%** of `57.08 t/s` (`MEASURED R13`, CoV 0.89%) and ours `step_wall_ms` within **±1.5%** of `67.92` (`MEASURED R13`, CoV 0.5%). A miss ≥ that is an **inconsistency** (box state differs) and blocks every downstream prediction until the box is re-quieted.
- **kill:** the box is not quiet (`uptime` load > 2.0, or `cdb-daemon` resident) and the numbers land outside the band — the anchor does not reproduce, so no later delta is attributable.
- **memory gate:** formula §3. Record `device_allocated_bytes` per step and task RSS; KILL on any of the four conditions. Expect steady `4.152-4.163e9` and RSS `48-66 MB` (`MEASURED R13`).
- **rollback:** none — read-only measurement, no source change.
- **blast:** zero source files. One raw-log directory under `docs/bench-campaigns/`.
- **observe:** `metal_decode_summary` (`proxima-model-interop/src/bind.rs:3042`) and `token_breakdown_metal` (`proxima-model-interop/src/generate.rs:1721-1750`, fields `plan_hits`, `plan_misses`, `nocopy_reuses`, `mapping_offset_uploads`, `device_allocated_bytes`, `plan_cache_len`).
- **reprove:** the same two commands, interleaved.
- **log-row title:** `## ROW <NEXT> -- the 2026-09-03 anchor, re-sealed under a real mutex: llama.cpp-Metal vs ours at 4be2f3a, arms interleaved`

---

#### C0.3 — fix the two instrument defects before any GB/s row is written

- **tier:** worker
- **depends_on:** C0.2
- **worktree/branch/target:** `proxima-wt-bytes` / `fix/op-profile-byte-accounting` / `.../proxima-wt-bytes/target`
- **opens:** `omega/src/metal.rs:694` (`operand_bytes`, sums bound **buffer** lengths); `omega/src/metal.rs:664-690` (the op-timed path's **own** block-upload loop); `omega/src/metal.rs:457-491` (`execute_plan`'s separate loop — both must be fixed, `R17 READ`); `proxima-model-interop/src/generate.rs:109-212` (consumer: total, per-bucket, per-op top, per-family, pass/reject); `omega/src/metal.rs:1786-1815` (`checkpoint_mapping_offset`, the cause)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-bytes && grep -n "operand_bytes" omega/src/metal.rs proxima-model-interop/src/generate.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-bytes && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-bytes && PROXIMA_MAX_TOKENS=4 bash scripts/gpu-measure-lock.sh -- cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' --no-capture
  ```
- **expect (N):** per-matvec `operand_bytes` for `blk.*.ffn_up.weight` = **34,668,544 B** (`DERIVED`: rows×k×0.5625 ≈ 33.05 MB `MEASURED R13`), **not** `4,140,417,024`; `total_operand_bytes` per token drops from **1.2 TB** to the low-GB range; `block_upload_bytes` per steady token drops from `4,147,777,096` to the actually-copied bytes with `copying_uploads=4`. Gate `ran_count` > 0 and `passed_count` > 0.
- **predict (one rung, milli→bench):** with byte accounting correct, the ffn_up family GB/s at the **bench** rung lands within ±10% of the **97.4 GB/s** shape-derived figure in `R13 DERIVED`, confirming the derivation rather than the instrument.
- **kill:** the corrected `operand_bytes` still equals the mapping buffer length for any op — the fix did not reach the mapping-offset path, and `gpu_ns_per_byte` stays unusable.
- **memory gate:** formula §3, full KILL set. Instrument-only change; any device-byte movement at all is itself a defect.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-bytes checkout -- omega/src/metal.rs proxima-model-interop/src/generate.rs`
- **blast:** two files, both instrument-gated paths. No public API. Both the batched and op-timed upload loops (`R17`), or the fix is half-applied.
- **observe:** `token_breakdown_metal`'s `kv_cache_upload_bytes` (must stay `+262,144 B/token`, MEASURED) as the control that the fix changed accounting, not behavior.
- **reprove:** the third command above.
- **log-row title:** `## ROW <NEXT> -- every GB/s row before this one was wrong: op_profile charged each matvec the whole 4.14 GB checkpoint buffer`

---

#### C0.4 — capture the uncommitted 2026-09-02 work as patches before it is lost

- **tier:** hands
- **depends_on:** C0.1
- **worktree/branch/target:** read-only across the named worktrees; artifacts land in `proxima-wt-rescue` / `rescue/gpu-uncommitted-2026-09-02` / `.../proxima-wt-rescue/target`
- **opens:** `R7` table (10 worktrees, all based on `2b95210`, all listed with dirty file counts)
- **commands:** for each of the ten worktrees in `R7` (`proxima-wt-{all,drive,rules,splitk,place,merge,gpudisp,gpuker,q4k,lat}`):
  ```
  git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-all status --porcelain
  git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-all diff --stat
  git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-all diff > /Users/brianbruggeman/repos/slot-0/proxima-wt-rescue/docs/bench-campaigns/2026-09-03-gpu-one-risc/patches/gpu-all-wins.patch
  ```
  (`git -C … diff` is read-only on the source tree; only the rescue worktree is written. Luna never `cd`s into a `proxima-wt-*` sibling.)
- **expect (N):** exactly **10** patch files; `proxima-wt-all` diff stat = **13 files, +3859/-197** and **6 untracked** (`R7 READ`) — a mismatch means the tree moved since the ledger and the patch is of something else.
- **predict (one rung, n/a — this card does not measure):** no bench prediction. This card's output is provenance, not a number. It therefore carries **no** rung and is explicitly excluded from the ladder.
- **kill:** any of the ten diff stats disagrees with `R7` — the uncommitted state is not what the ledger recorded, and C0.5/C0.6 cannot adjudicate patches of unknown origin.
- **memory gate:** n/a (no process runs the model).
- **rollback:** delete the patches directory. Source worktrees are never written.
- **blast:** zero source files anywhere; one untracked directory in the rescue worktree.
- **observe:** the `--stat` lines themselves, recorded verbatim on the row against `R7`'s.
- **reprove:** re-run the ten `git -C … diff --stat` commands and diff against the recorded stats.
- **log-row title:** `## ROW <NEXT> -- ten worktrees, 9 commits behind main, carrying the only copy of a -17.2% and a -4.9%: the uncommitted inventory`

---

#### C0.5 — adjudicate the two Q4_K bodies; exactly one survives

- **tier:** judge
- **depends_on:** C0.3, C0.4
- **worktree/branch/target:** `proxima-wt-q4kbody` / `perf/q4k-body-adjudication` / `.../proxima-wt-q4kbody/target`
- **opens:** `omega/src/msl.rs:1235` (`packed_row_block`), `:2526-2528` (packed loop `ib += SIMD_WIDTH/lanes_per_block`), `:1978-1982` (`Q4K_UNPACK_MSL`/`Q5K`/`Q6K` concatenated **with no delimiter** — so "grep the Q4_K region" of emitted source is undecidable, `R16 READ`); the two candidate bodies: `metal-q4k-mask-fma` (patch from C0.4, `-36%` on ffn_gate/up, `-17.2%` gpu_exec, `MEMORY R3/M3`) and `q4k_pair_dot` (`perf/cached-attention-streaming` ROW 257, family `47.8 → 33.9 ms`, `-29%`, parity `3.1e-6` vs f32 on real `blk.0.attn_q.weight`, `MEASURED R12`)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-q4kbody && git log --oneline main..perf/cached-attention-streaming -- omega/src/msl.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-q4kbody && git show perf/cached-attention-streaming -- omega/src/msl.rs | head -400
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-q4kbody && bash scripts/gpu-measure-lock.sh -- cargo run --release -p omega --example q4k_matvec_probe --features metal
  ```
- **expect (N):** both bodies re-derive the **same** ggml mechanism (`R12`: "Independent re-derivation of the same ggml mechanism"), so the census must show **1** surviving body, not 2. Parity for the survivor ≤ **3.1e-6** vs f32 on `blk.0.attn_q.weight` (`MEASURED R12`). N==0 surviving bodies is RED; N==2 is RED.
- **predict (one rung, nano→micro):** the surviving body at the **micro** rung (`real_forward_packed_probe`) moves the ffn_up+ffn_gate+ffn_down family from **31.71 ms** (`DERIVED R13`: 10.862+10.843+10.005) to **≤ 22.5 ms**, i.e. ≥ 29% (the smaller of the two measured family deltas, `−29%` R12, chosen deliberately over `−36%`).
- **kill:** the two bodies disagree on parity by more than 1e-5 on the same weight — they are not the same mechanism and the adjudication premise is false; both go back to measurement.
- **memory gate:** formula §3. A body change must move `device_allocated_bytes` by **0**; any movement means it changed allocation, not arithmetic.
- **rollback:** the judge writes a decision, not code. Rollback = withdraw the decision row.
- **blast:** zero source files (judge tier). The decision binds C0.6's blast radius.
- **observe:** the route census from C1.1/C1.2 once it exists; until then, the parity number and the nano-rung probe timing — **not** `classify_kind`, which has relabeled twice (§1.1).
- **reprove:** re-run the nano probe on the survivor and confirm the parity figure.
- **log-row title:** `## ROW <NEXT> -- two worktrees independently re-derived ggml's Q4_K mask-without-shift; the adjudication, and the one body that lands`

---

#### C0.6 — rebase and land the surviving Q4_K body, one green commit

- **tier:** worker
- **depends_on:** C0.5
- **worktree/branch/target:** `proxima-wt-q4kbody` / `perf/q4k-body-adjudication` / same target
- **opens:** the survivor's diff hunks in `omega/src/msl.rs`; the conflict surface named by `R7`: main has moved 9 commits past `2b95210` including `spec.rs +8735/-2836` (qwen3.5) and `omega/src/metal.rs` changes (`7d09145`, `23e2e5e`) — **every** patch from C0.4 conflicts on rebase.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-q4kbody && git rev-parse --short HEAD
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-q4kbody && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-q4kbody && PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh -- cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' --no-capture
  ```
  Interleaved against the C0.2 anchor arm, three pairs.
- **expect (N):** gate all six steps PASS with `ran_count` > 0 **and** `passed_count` > 0; `op_count` still **1196** (a body change must not change the graph); `generated_text` byte-identical to C0.2's.
- **predict (one rung, micro→milli):** at the **milli** rung, `reduce-packed-row-blocked` falls from **44.450 ms / 225 ops** (`MEASURED R13`) to **≤ 31.6 ms** (−29%), and the per-op figure from 197,556 ns to ≤ 140,000 ns.
- **kill:** `generated_text` differs by one byte, or gate `[3/6]`/`[6/6]` reports zero — correctness before speed, §14 incumbent wins on correctness.
- **memory gate:** formula §3, full KILL set. Expect `device_allocated_bytes` unchanged from C0.2 within 2 MB.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-q4kbody reset --hard origin/main`; the branch is discarded, main is untouched (no commit without owner authorization).
- **blast:** `omega/src/msl.rs` only, inside the packed body emitter. No feature added (the body is unconditional once adjudicated — a mutually-exclusive feature pair would break five of six gate steps, C-C). If the owner wants it gated, it must be a **build-time profile axis** (`[q4k] body = "..."` → `cargo:rustc-cfg` via `omega/build.rs`'s `resolve_int`/`emit_sizing_consts` pattern, §8 profile input), never two cargo features.
- **observe:** milli-rung per-bucket table; ffn_up/ffn_gate/ffn_down family rows.
- **reprove:** the third command, interleaved with llama-bench.
- **log-row title:** `## ROW <NEXT> -- the Q4_K body lands on main: ggml's mask-without-shift, one green commit, the family delta re-measured at the milli rung`

---

#### C0.7 — land the wide cooperative reduce

- **tier:** worker
- **depends_on:** C0.6
- **worktree/branch/target:** `proxima-wt-coop` / `perf/wide-cooperative-reduce` / `.../proxima-wt-coop/target`
- **opens:** `omega/src/msl.rs:3140-3194` (`push_cooperative_reduce_body`; body computes `output_index = gid/32`, `lane = gid%32` at `:3190-3194`); `omega/src/msl.rs:1517-1560` (`grid_threads`; cooperative arm is `output_total * SIMD_WIDTH` at `:1557`); `omega/src/sized.rs:45` (`SIMD_WIDTH = 32`, declared a **hardware fact, never a policy knob** — so the fix is not to change it but to add a separate policy-tier width); incumbent contrast `ggml-metal.m:3797-3804` (nth doubles from 32 up to `min(ne00/4, maxTotalThreadsPerThreadgroup)`, float4 loads, `simd_sum` → threadgroup shmem → `simd_sum`, `ggml-metal.metal:1679-1721`)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-coop && sed -n '3140,3194p' omega/src/msl.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-coop && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-coop && PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh -- cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' --no-capture
  ```
- **expect (N):** `reduce-cooperative` op count stays **385** (geometry change, not graph change); `eps` (rms-norm) family stays **65** ops. A changed op count means the graph moved and the arm is confounded.
- **predict (one rung, milli→bench):** `reduce-cooperative` falls from **9.113 ms** (`MEASURED R13`) to **≤ 7.3 ms** (−20%, `MEASURED R3/M4`), and at the **bench** rung `step_wall_ms` falls by **≥ 1.5 ms** against the C0.6 arm.
- **kill:** the wide reduce is a loss on the real graph, as nsg=2 was **four** times (`R4`, `R12` ROWs 249/251-254/259-260/265-267). One measured loss on the real graph retires the lever; it does not get a fifth attempt.
- **memory gate:** formula §3. A wider threadgroup raises threadgroup memory, not device buffers; `device_allocated_bytes` must be unchanged within 2 MB.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-coop reset --hard origin/main`
- **blast:** `omega/src/msl.rs` cooperative body + `grid_threads` cooperative arm. The width becomes a config key in C2.1, not a new bare const — until C2.1 lands, the card carries a `TODO(<NEXT> C2.1)` naming it, and the row records the §12 debt explicitly.
- **observe:** milli-rung `reduce-cooperative` bucket; per-family `eps` row.
- **reprove:** the third command.
- **log-row title:** `## ROW <NEXT> -- the cooperative reduce ran 32 lanes wide for every reduction size; ggml runs up to 1024`

---

#### C0.8 — adjudicate the parallel branch: reject the fifth bound kind, keep the generic half

- **tier:** judge
- **depends_on:** C0.2, C0.5
- **worktree/branch/target:** `proxima-wt-reconcile` / `docs/cached-attention-adjudication` / `.../proxima-wt-reconcile/target`
- **opens:** `perf/cached-attention-streaming` (42 commits **on main 4be2f3a**, another agent, today, `R6/R12 READ`): `BoundOpKind::CachedAttention` (CPU arm `proxima-tensor/src/cpu.rs:19141-19190`, `render_cached_attention` `omega/src/msl.rs:104`), the post-bind structural matcher (`cached_attention_candidates`, `attention_score_sources`, `is_exact_causal_mask`, `removable_attention_dependencies`), their own `failure-cached-attention-matcher.md`, and `prune_dead`/`dead_resolved_nodes` (commit `216d925`)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-reconcile && git log --oneline main..perf/cached-attention-streaming | wc -l
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-reconcile && git diff --stat main..perf/cached-attention-streaming
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-reconcile && git show 216d925 --stat
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-reconcile && grep -n "CachedAttention" proxima-tensor/src/bind.rs omega/src/msl.rs
  ```
- **expect (N):** 42 commits; `physical.rs` **+576** (new module); `bind.rs` **+666**; `discipline.md` **+938** carrying **ROW 234-267** (their numbering, which collides with main's ROW 233+1 and with three unlanded "ROW 234"s, `R10/R12`). A count mismatch means the branch moved and the adjudication is of something else.
- **predict:** none — judge tier, no rung.
- **kill:** n/a. The decision itself is the output.
- **memory gate:** n/a (no process).
- **rollback:** withdraw the decision row.
- **blast:** zero source files.
- **The decision, with its two binary questions answered:**
  - **`BoundOpKind::CachedAttention` — REJECT.**
    - *Q1, can an existing primitive express it? Write the expression.* Yes: two `Op::Reduce` writing **disjoint ranges of one caller-owned buffer** via `out_layout.base`, which already exists on every bound `Reduce` (`proxima-tensor/src/bind.rs:95-98`) and is already emitted as `long out_base` (`omega/src/msl.rs:2216`, consumed `:2361, 2734, 3079, 3460, 3541`). The only blocker is `project_output_shape` (`proxima-tensor/src/shape.rs:469-485`) rejecting a non-unit term, while `bounds_check` (`:441-467`) already handles `axis.offset` on the read side. Card C5.2 writes exactly that expression. **Q1 fails for the new type → do not mint.**
    - *Q2, what can a caller DO that it could not before?* Nothing. It computes the same attention. Call site before: `evaluate(&program, …)` over the two-range online-softmax cluster. Call site after: `evaluate(&program, …)` over the same cluster, now matched into one macro-op. The caller's vocabulary is unchanged.
    - *And it costs.* `R12 MEASURED`: feature-on GPU **39.841** vs feature-off **35.117** — the macro-op is **+4.7 ms of GPU**; wall is flat (51.535 vs 51.571). It is a fifth kind that CUDA and WGSL would also owe, when they already fail coverage on Iota/Constant (`omega/src/cuda.rs:146-183`). Their own `failure-cached-attention-matcher.md` records the BoundOp-only matcher abandoned as "a heuristic" that "cannot prove the semantic roles." Workspace AGENTS.md forbids an arbitrary rule for a specific instance.
  - **`prune_dead`/`dead_resolved_nodes` (216d925) — KEEP.** Generic, RISC-conformant, no new kind, applies to every graph. Cherry-picked in C3.x.
  - **The paired Q4_K body — already adjudicated in C0.5** (same mechanism as mask-fma; one survives).
  - **Their classifier fix (ROW 263) — SUPERSEDE, do not merge.** It adds a *second marker string* to `classify_kind`, which is the same substring trap one layer deeper. C1.1 replaces the mechanism with a first-class route.
  - **Row numbering — renumber at land** from main's ROW 233; their 234-267 collide with three unlanded "ROW 234"s and are non-monotonic in physical order (263/264 at line 5870).
- **observe:** the two GPU-ms figures above, side by side, as the refutation.
- **reprove:** re-run the four commands; the diff stats are the reprovable artifact.
- **log-row title:** `## ROW <NEXT> -- a fifth bound kind for one model's attention: rejected on its own numbers (+4.7 ms GPU, flat wall), with prune_dead kept`

---

#### C0.9 — land the discipline rows

- **tier:** hands
- **depends_on:** C0.2, C0.3, C0.5, C0.6, C0.7, C0.8
- **worktree/branch/target:** `proxima-wt-rows` / `docs/gpu-lane-discipline-rows` / `.../proxima-wt-rows/target`
- **opens:** `proxima-tensor/docs/discipline.md:18736` (ROW 233, the last on main); `proxima-tensor/docs/rooflines.md:396-479` (GPU lane; candidate ceiling = **DEBT, not measured**) and `:766-773` (the doc's own closing note that the GPU ratio "is not a gap-to-machine at all")
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-rows && grep -n "^## ROW" proxima-tensor/docs/discipline.md | tail -3
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-rows && grep -c "ROW <NEXT>" proxima-tensor/docs/discipline.md
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-rows && grep -n "3.54x\|17.470\|228.9" proxima-tensor/docs/discipline.md
  ```
- **expect (N):** before renumbering, `grep -c "ROW <NEXT>"` equals the number of rows staged (one per landed card); after, **0**. The last `## ROW` line is `233` before, and monotonically increasing after. The third command returns **0** before the rows land and **≥ 3** after — main's log learning what it did not know (`R10 READ`, verified: currently 0).
- **predict:** none — documentation card, no rung.
- **kill:** any placeholder `ROW <NEXT>` survives into a staged commit, or a number is assigned that already exists on main.
- **memory gate:** n/a.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-rows checkout -- proxima-tensor/docs/discipline.md`
- **blast:** two docs files. No source.
- **observe:** the row count and the monotonicity check.
- **reprove:** the three commands.
- **log-row title:** `## ROW <NEXT> -- main's log did not know the GPU session happened: the 2026-09-02 and 2026-09-03 rows, renumbered from 233 at land`

---

#### C0.10 — ai_docs records for the GPU lane

- **tier:** hands
- **depends_on:** C0.9
- **worktree/branch/target:** `proxima-wt-aidocs` / `docs/ai-docs-gpu-lane` / `.../proxima-wt-aidocs/target`
- **opens:** `ai_docs/AGENT.md` (the record contract — the plan must ADD records, not bypass), `ai_docs/index.jsonl` (23 lines), `ai_docs/task-routes.jsonl` (18), `ai_docs/invariants.jsonl` (31)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-aidocs && grep -c "omega\|tensor\|gpu" ai_docs/task-routes.jsonl ai_docs/invariants.jsonl
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-aidocs && bash ai_docs/query.sh gpu
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-aidocs && python3 -c "import json,sys; [json.loads(l) for l in open('ai_docs/invariants.jsonl')]; print('jsonl-parses')"
  ```
- **expect (N):** first command returns **0 and 0** before (verified on main: `ai_docs/task-routes.jsonl:0`, `ai_docs/invariants.jsonl:0`) and **> 0** after. Every line must parse as JSON — N==0 parsed lines is RED.
- **predict:** none.
- **kill:** a malformed line breaks `query.sh` for every other lane.
- **memory gate:** n/a.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-aidocs checkout -- ai_docs/`
- **blast:** three JSONL files, append-only.
- **Invariants to record** (each traceable to a file:line above): `out_map` is a pure projection until C5.2; `ScalarOp` is closed (no `GreaterEqual`); the Metal plan is post-bind-rewritten until C5.1; KV capacity is never derived from `context_length`; the route is a property of the plan, not of the dispatch.
- **observe:** `query.sh gpu` returning records.
- **reprove:** the three commands.
- **log-row title:** `## ROW <NEXT> -- the GPU lane had zero ai_docs records; the index, routes and invariants that make it findable`

---

### PHASE 1 — make the lowering auditable (gates everything after it)

---

#### C1.1 — `Route`: one first-class decision, made before emission

- **tier:** worker
- **depends_on:** C0.6, C0.8
- **worktree/branch/target:** `proxima-wt-routeenum` / `feat/omega-route-enum` / `.../proxima-wt-routeenum/target`
- **opens:** `omega/src/msl.rs:673-697` (`emit`, the five-arm match on `BoundOpKind`); `:751-754` (the load-bearing gate ordering); `:824` (`reduce_is_cooperative` = associative op **and** `gather_count == 0`); `:3140-3194` (`push_cooperative_reduce_body` splitting tiled-GEMM → packed-row-blocked → generic SIMD fold); `:731` (`kernel_cache_key`, "cheap structural identity … built without ever rendering the MSL body text"); `:1487` (`diagnose_packed_row_block`); `omega/src/metal.rs:785-826` (`classify_kind`, to be retired) and `:835-854` (`diagnose_kind`)
- **The two binary questions, answered on the card:**
  - **Q1 — can a pipe or existing primitive express it? Write the expression.** The existing expression is `classify_kind(bound, packed_operands) -> &'static str`, which computes `emit(bound, packed)` and greps `kernel.source` for `"simdgroup_multiply_accumulate"`, `"q4k_run8(blk"`, `"simd_sum("`. It **is** an expression, and it is measurably wrong: its own doc at `omega/src/metal.rs:790-800` records ROW 113 silently relabeling tiled-GEMM as row-blocked, and `R12` ROW 263 records the paired body relabeled as `reduce-cooperative` (9/601 → 225/385). The nearest structural primitive, `kernel_cache_key` (`omega/src/msl.rs:731`), returns an identity **for caching** and carries no reason, so it cannot express a decline. There is no existing expression for "why did this op not take the packed path."
  - **Q2 — what can a caller DO that it could not before? Both call sites.**
    - *before:* `let bucket: &'static str = classify_kind(bound, packed);` — the caller can print a string. It cannot ask why an op declined a route, cannot count declines by reason, and the string silently changes when a kernel body changes.
    - *after:* `let route: Route = msl::route_of(resolved, &quantized);` — the caller can now write `assert_eq!(census.count(Route::PackedRowBlocked), 225);` in a test, and `assert_eq!(census.count(Route::Declined(WidthDeclineReason::…)), N);`. **An asserted N that a substring cannot produce.** That is a capability, not a rename. **Q2 passes → mint, in omega only.**
  - **Constraint check:** `Route` lives in `omega`, over the existing 4 `BoundOpKind`s. It is **not** a new `Op`, `BoundOpKind`, `ScalarOp`, or `IndexMap` variant. No `Box<dyn>`. §1 satisfied (the existing expression was written first and shown insufficient).
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-routeenum && sed -n '673,700p;745,760p;820,835p' omega/src/msl.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-routeenum && grep -n "classify_kind\|diagnose_kind" omega/src/metal.rs proxima-model-interop/src/generate.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-routeenum && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  ```
- **expect (N):** `route_of` is total over the 4 kinds — a match with **no** `_` arm, so adding a kind is a compile error, and **8** Metal body shapes map to **8** route slots (`R5 READ`: tiled-GEMM, packed-row-blocked, cooperative, generic-scalar, scan, elementwise, iota, constant). Gate `ran_count` > 0, `passed_count` > 0.
- **predict (one rung, nano→micro):** `route_of` is text-free (it must not call `emit`), so at the **micro** rung the `emit` phase (`0.81 ms/token MEASURED R13`) is unchanged within ±5%; if it moves more, `route_of` is rendering source and has reproduced the defect it replaces.
- **kill:** `route_of` cannot decide a route without calling `emit` — then the decision genuinely is downstream of text and the whole one-RISC item 2 needs redesign, not a wrapper.
- **memory gate:** formula §3. `Route` is `Copy` where it carries no payload; `Route::Declined(WidthDeclineReason)` is data-carrying, so `route as usize` is **E0605** (`R17`) and it needs `fn slot(&self) -> usize`. Zero heap allocation on the route path (§11).
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-routeenum reset --hard origin/main`
- **blast:** `omega/src/msl.rs` (new `route_of` + `Route`), `omega/src/metal.rs` (`classify_kind` becomes a thin `Route → &'static str` for row compatibility, then is deleted in C1.3). `omega/src/{wgsl,cuda}.rs` untouched this card.
- **observe:** a unit test asserting the route of a synthetic packed reduce, and the emit-phase timing as the no-regression control.
- **reprove:** the third command.
- **log-row title:** `## ROW <NEXT> -- the routing decision was recovered by grepping emitted MSL, and it had silently relabeled twice; Route becomes a value`

---

#### C1.2 — the census, lock-free, keyed `(NodeId, Route)`

- **tier:** worker
- **depends_on:** C1.1
- **worktree/branch/target:** `proxima-wt-routeenum` / `feat/omega-route-enum` / same
- **opens:** `proxima-tensor/src/instrument.rs:842` (`static WIDTH_TILE_DECLINE: Mutex<BTreeMap<(u32, WidthDeclineReason), WidthDeclineTotals>>`, `.lock()` at `:857-859` — read in full, confirmed `Mutex`); `:809-828` (`WidthDeclineReason`, 8 variants — the census **pattern** to mirror); `omega/src/metal.rs:2243` (`ENCODE_DISPATCH_CALLS`, an atomic `proxima_telemetry::Counter`, `:1484`); `proxima-telemetry/src/metric/counter.rs:12-17` (`Counter { name, unit, description, value: AtomicU64 }` — **non-`Copy`**, with `const fn new` at `:19`)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-routeenum && sed -n '840,865p' proxima-tensor/src/instrument.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-routeenum && sed -n '1,25p' proxima-telemetry/src/metric/counter.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-routeenum && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-routeenum && PROXIMA_MAX_TOKENS=4 bash scripts/gpu-measure-lock.sh -- cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' --no-capture
  ```
- **expect (N):** the census sums to **1196** routed ops per token (`MEASURED R13`), partitioned as `reduce-packed-row-blocked` **225**, `reduce-cooperative` **385**, `elementwise` **547**, `constant` **37**, `iota` **2** (`MEASURED R13`; 225+385+547+37+2 = 1196 `DERIVED`). A partition that does not sum to `op_count` is RED. N==0 in any expected-nonzero slot is RED.
- **predict (one rung, micro→milli):** the census is filled **once at plan time** (the route is a property of the plan, not of the dispatch, `R16`), so at the **milli** rung `encode_dispatch` (`0.47 ms/token MEASURED R13`) is unchanged within ±5%.
- **kill:** the census needs a lock on the per-dispatch path — §21 lock-free is violated and the instrument perturbs what it measures. `WIDTH_TILE_DECLINE`'s `Mutex` is the pattern to **learn from**, not to copy.
- **memory gate:** formula §3. Preferred shape: a per-plan preallocated `Vec<u8>` of route slots filled once at plan time (bounded by `op_count`, ≈1196 B/plan) **or** a fixed-size `[AtomicU64; N_ROUTES]` via `[const { AtomicU64::new(0) }; N]`. `[Counter; N]` needs N explicit const initializers (`Counter` is non-`Copy`, verified) — prefer the atomic array. Zero per-dispatch allocation.
- **rollback:** revert the census commit only; `Route` from C1.1 survives.
- **blast:** `omega/src/metal.rs` census statics + one call at plan time; `proxima-model-interop/src/generate.rs:109-212` consumer gains a route column. Instrument-gated throughout, following the `omega?/instrument` passthrough pattern (`proxima-model-interop/Cargo.toml`, read: `instrument = ["proxima-tensor/instrument", "omega?/instrument", …]`).
- **observe:** the census table itself; cross-checked against `op_count` from `metal_decode_summary`.
- **reprove:** the fourth command.
- **log-row title:** `## ROW <NEXT> -- a route census keyed (NodeId, Route), lock-free and filled at plan time; the partition sums to op_count or it is wrong`

---

#### C1.3 — retire `classify_kind`; compare by family, never by bucket

- **tier:** worker
- **depends_on:** C1.2
- **worktree/branch/target:** `proxima-wt-routeenum` / `feat/omega-route-enum` / same
- **opens:** `omega/src/metal.rs:709-710` (`classify_kind` + `diagnose_kind` call sites, op-timed path only); `proxima-model-interop/src/generate.rs:184` (`op_profile_family`, exists **only** in per-op mode)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-routeenum && grep -rn "classify_kind" omega/src proxima-model-interop/src
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-routeenum && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  ```
- **expect (N):** after the change, `grep -rn "classify_kind" omega/src proxima-model-interop/src` returns **0** hits. Before, it returns hits at `omega/src/metal.rs:709,785` (verified on main). N != 0 after is RED.
- **predict (one rung, micro→milli):** deleting a substring classifier from the op-timed path changes the per-op-mode total (**61.082 ms over 1196 ops MEASURED R13**, 7.3% over batched) by **< 0.5%** — the classifier called `emit` per op, so if anything the total falls.
- **kill:** a family row cannot be reproduced without `classify_kind` — then the family attribution was never structural and the scorecard needs rebuilding from `op_profile_family` first.
- **memory gate:** formula §3.
- **rollback:** revert this commit; C1.1/C1.2 survive.
- **blast:** `omega/src/metal.rs` (delete), `proxima-model-interop/src/generate.rs` (consumer switches to the route column).
- **observe:** the grep returning 0.
- **reprove:** both commands.
- **log-row title:** `## ROW <NEXT> -- classify_kind deleted: an instrument that relabels itself when a kernel body changes cannot carry a scorecard`

---

### PHASE 2 — one sizing config

#### C2.1 — every geometry constant traces to `omega/omega-runtime.toml`

- **tier:** worker
- **depends_on:** C0.7, C1.3
- **worktree/branch/target:** `proxima-wt-geom` / `feat/omega-geometry-config` / `.../proxima-wt-geom/target`
- **opens:** `omega/src/msl.rs:1017` (`PACKED_ROWS_PER_GROUP = 4`), `:1030` (`TILE_DIM = 8`), `:1046` (`TILED_GEMM_NSG = 4`), `:2732` (lanes/block, `MEMORY R5` — **re-read before citing**); `omega/src/sized.rs:45` (`SIMD_WIDTH = 32`, hardware fact, **stays a const**, per its own doc: "Cannot be runtime config at any tier"); `omega/build.rs:16` (`require_nonzero`), `:35` (`require_multiple_of_sixteen`), `:43` (`require_divides_q4k_block`), `:59` (`require_multiple_of_eight`), `:67` (`get_int`), `:79` (`resolve_int`, env `OMEGA_{SECTION}_{KEY}` + `rerun-if-env-changed` at `:85`), `:105` (`emit_sizing_consts`); `omega/omega-runtime.toml` (read in full — one section, `[tiled_gemm]`); `proxima-tensor/proxima-tensor-runtime.toml` (the CPU pattern to mirror)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-geom && grep -n "^const \|^pub const " omega/src/msl.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-geom && cat omega/omega-runtime.toml
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-geom && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-geom && OMEGA_PACKED_ROW_ROWS_PER_GROUP=8 cargo build -p omega --features metal 2>&1 | tail -20
  ```
- **expect (N):** after the change, `grep -n "^const " omega/src/msl.rs` returns **0** geometry consts (codec block consts at `:294-556` are codec facts, not policy — they stay, and the row says so). `omega/omega-runtime.toml` gains `[packed_row]` and `[cooperative_reduce]` sections. The env override rebuilds (proving `rerun-if-env-changed` fired) — a build that does **not** rebuild is RED.
- **predict (one rung, nano→micro):** moving a const from source to build.rs is value-preserving, so the **micro** rung family total is unchanged within **±0.5%** at the default values. Any larger move means a value changed, not a location.
- **kill:** a constraint cannot be expressed as a build-time assertion (e.g. rows-per-group must divide `SIMD_WIDTH`) — then the value is not a policy knob and belongs in `sized.rs` with `SIMD_WIDTH`, and the row says which.
- **memory gate:** formula §3. Threadgroup geometry does not touch device buffers; `device_allocated_bytes` unchanged within 2 MB.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-geom reset --hard origin/main`
- **blast:** `omega/omega-runtime.toml`, `omega/build.rs`, `omega/src/sized.rs` (docs), `omega/src/msl.rs` (const → `sized::` import). **Care:** `emit_sizing_consts` currently emits the tiled-GEMM block only under `CARGO_FEATURE_METAL_TILED_GEMM` (read at `omega/build.rs:120`); the new sections must be emitted **unconditionally** (they feed the default packed path) or `[1/6]`'s bare-alloc build breaks on a missing const.
- **observe:** the generated `OUT_DIR/omega_sized.rs` contents; the rebuild triggered by the env var.
- **reprove:** commands three and four.
- **log-row title:** `## ROW <NEXT> -- three bare geometry consts in msl.rs were the last §12 violations in the GPU lane; every one now traces to omega-runtime.toml`

---

### PHASE 3 — orchestration, 11.0 ms `MEASURED R13`

#### C3.1 — bucket `cached_len`; invert **both** harness assertions

- **tier:** worker
- **depends_on:** C1.2, C2.1
- **worktree/branch/target:** `proxima-wt-bucket` / `perf/kv-bucketed-plan-cache` / `.../proxima-wt-bucket/target`
- **opens:** `proxima-model-interop/src/generate.rs:966` (`let shape = (symbols[0] as usize, symbols[1] as usize);` — **the root cause**, read in full at `:958-977`, with `self.plans.clear()` at `:973`); `proxima-tensor/src/spec.rs:6216-6245` (KV leaves at `Extent::Symbolic(1)`, verified: `Symbolic(1)` at `:6220, 6230, 6240`); `proxima-model-interop/src/bind.rs:3052-3058` (**both** assertions, C-B); `proxima-tensor/src/spec.rs:823-845` (`causal_mask`: two `Iota{extent: Symbolic(0)}` + `Greater` + `scalar_constant(-inf)` — the construction a tail mask mirrors); `proxima-model-interop/src/generate.rs:799-827` (`build_position_inputs`, uses `start_position` only for cos/sin) called at `:1304-1309` with the **true** `cached_len`, and `apply_serving_config(config, cached_len + new_count)` at `:1298` — **so bucketing symbol 1 does not touch RoPE positions or `is_future`** (`R17`, verified by reading all three)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-bucket && sed -n '958,977p' proxima-model-interop/src/generate.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-bucket && sed -n '3040,3060p' proxima-model-interop/src/bind.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-bucket && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-bucket && PROXIMA_MAX_TOKENS=32 bash scripts/gpu-measure-lock.sh -- cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' --no-capture
  ```
- **expect (N):** with a bucket of 256 (the incumbent's own quantum — `n_kv` padded to a 256 multiple and masked, `R11/M6'`), over 32 tokens: `plan_hits` = **31**, `plan_misses` = **1**. Both current assertions must be **inverted in the same change** (C-B): `plan_hits == forward_calls_taken - ceil(tokens/256)` and `plan_misses == ceil(tokens/256)`. `generated_text` byte-identical to C0.2's. N: `plan_hits == 0` after the change is RED — it means the bucketing did not take.
- **predict (one rung, milli→bench):** at the **bench** rung, `prepare` falls from **1.97 ms** and `op_setup` from **3.9 ms** (`MEASURED R13`) to a combined **≤ 2.0 ms** on a hitting step, and `step_wall_ms` falls by **≥ 3.5 ms** against the C0.7 arm. (`op_setup` only partly falls here — the per-op buffer allocation is C3.2's.)
- **kill:** `generated_text` changes by one byte. The tail mask must compose `Greater(cached_len_leaf, iota)` with the existing `Select`/`-inf` — `ScalarOp` has **no `GreaterEqual`** (verified: 17 variants, `Greater` and `Equal` only), and §"the one closed set that stays closed" forbids adding one. If correctness needs `GreaterEqual`, the card is dead and the fallback (C3.1-alt) runs.
- **C3.1-alt fallback** (named now, not improvised): make the reduce extent over the cache a runtime uniform. `R18` shows this is cheaper than it looks — `Uniforms` (`omega/src/msl.rs:2207-2218`, read in full) **already** carries per-dispatch `output_total`, `reduction_total`, `output_extents[]`, `reduction_extents[]`, `operand_base[]`, `operand_strides[][]`, `out_base`, `out_strides[]`, and `pipeline_lookup` is **0.04 ms** (`MEASURED R13`), so pipelines already survive `cached_len` changes. What is rebuilt per token is bind + retirement + packed-operand resolution. Splitting `Plan` into invariant + per-token parts is the fallback. Blocker: `BoundOp::extents` is a baked `Vec<u64>` (`proxima-tensor/src/bind.rs:200-215`, verified).
- **memory gate:** formula §3, **strictest application on this card**. Bucketing implies a KV capacity. `capacity_tokens` must be a **build-time key** with a build-time byte assertion; it must **never** come from `ServingConfig::context_length` (`proxima-model-interop/src/serving.rs:161` = `131_072` → 34,359,738,368 B). At capacity 2048: `262,144 × 2048 = 536,870,912 B` — assert that literal in `build.rs`. `capacity × 2048` is a 16 KiB multiple iff `capacity % 8 == 0` (`R17`), so `require_multiple_of_eight` (`omega/build.rs:59`) already exists to enforce it.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-bucket reset --hard origin/main`. The two assertions revert with it — this is why they must move in the same commit.
- **blast:** `proxima-model-interop/src/generate.rs` (cache key), `proxima-tensor/src/spec.rs` (tail mask in the cached layer), `proxima-model-interop/src/bind.rs` (both assertions). Touches `spec.rs`, which main moved `+8735/-2836` nine commits ago — expect conflict with any un-rebased branch.
- **observe:** `plan_hits`/`plan_misses`/`plan_cache_len` in `token_breakdown_metal` (`proxima-model-interop/src/generate.rs:1721-1750`); `prepare` and `op_setup` phase ticks.
- **reprove:** the fourth command.
- **log-row title:** `## ROW <NEXT> -- plan_hits=0 was never a cache bug: cached_len IS the shape symbol, so the key missed by construction`

---

#### C3.2 — preallocate output and uniform buffers on a now-stable plan

- **tier:** worker
- **depends_on:** C3.1
- **worktree/branch/target:** `proxima-wt-pool` / `perf/metal-buffer-pool` / `.../proxima-wt-pool/target`
- **opens:** `omega/src/metal.rs:2179-2252` (`encode_op`: `kernel_cache_key` + `kernel_dispatch_shape` at `:2193-2194`, `pipeline_for` cached at `:1402`, `allocate_buffer` for the **output** at `:2210`, `upload_uniforms` at `:2211` — **1196 `newBufferWithLength` + 1196 uniform uploads per token**, `R11/M6''`); `omega/src/metal.rs:2069-2075` (`UNIFORM_BUFFER_REUSES`, `upload_uniforms`, reuse path); `omega/src/metal.rs:541-543` (the retire loop removing from `BTreeMap<NodeId, DeviceBuffer>` — read in full; **a retire no-op makes every operand lookup walk ~1196 entries**, `R17`)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-pool && sed -n '2179,2252p' omega/src/metal.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-pool && sed -n '536,556p' omega/src/metal.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-pool && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-pool && PROXIMA_MAX_TOKENS=32 bash scripts/gpu-measure-lock.sh -- cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' --no-capture
  ```
- **expect (N):** `newBufferWithLength` calls per steady token fall from **1196** to **0**; `UNIFORM_BUFFER_REUSES` rises to **≥ 1196/token**. N==0 reuses is RED.
- **predict (one rung, milli→bench):** at the **bench** rung, `op_setup` falls from **3.9 ms** (`MEASURED R13`) to **≤ 1.0 ms**, and `step_wall_ms` falls by **≥ 2.5 ms** against the C3.1 arm.
- **kill:** the pool must sub-allocate **within** one buffer to pay off. That breaks the readback invariant at `omega/src/metal.rs:2360-2364`, read verbatim: "an output node's buffer is always freshly allocated by `encode_op` at offset 0 -- only a weight INPUT can carry a nonzero offset ... reading from the buffer's own start is always correct here." **Whole-buffer reuse per node is safe; sub-allocation within one buffer is not.** If the win needs sub-allocation, this card stops and the invariant is renegotiated on its own card.
- **memory gate:** formula §3, **and a new steady-state term**. Preallocating 1196 output buffers holds them all live simultaneously where retirement previously freed them (`omega/src/metal.rs:541-543`). Compute the pool's peak from `bound_op_retirement`'s liveness partition **before** building, and assert it against the 4.40e9 KILL ceiling. If peak-live × output sizes exceeds it, pool only the retired-and-reused classes.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-pool reset --hard origin/main`
- **blast:** `omega/src/metal.rs` only. Do **not** make retirement a no-op — `R17`'s warning is confirmed by the loop at `:541-543`; a no-op turns every operand lookup into a walk of ~1196 `BTreeMap` entries.
- **observe:** `op_setup` ticks (`omega/src/metal.rs:2209-2220`), `UNIFORM_BUFFER_REUSES`, `device_allocated_bytes`.
- **reprove:** the fourth command.
- **log-row title:** `## ROW <NEXT> -- 1196 newBufferWithLength and 1196 uniform uploads per token were op_setup; a stable plan preallocates both once`

---

### PHASE 4 — the non-matmul GPU mass, 16.6 ms `MEASURED R13`

#### C4.1 — cooperative-reduce geometry from config, swept

- **tier:** worker
- **depends_on:** C0.7, C2.1, C1.2
- **worktree/branch/target:** `proxima-wt-coop` / `perf/wide-cooperative-reduce` / same as C0.7
- **opens:** `omega/src/msl.rs:1557` (cooperative `grid_threads` arm), `:3190-3194` (body indexing); `omega/omega-runtime.toml` `[cooperative_reduce]` from C2.1
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-coop && for w in 64 128 256 512 1024; do OMEGA_COOPERATIVE_REDUCE_MAX_THREADS=$w bash scripts/gpu-measure-lock.sh -- cargo run --release -p omega --example membw_probe --features metal,instrument; done
  ```
- **expect (N):** **5** cells, one per width, each with CoV reported. N < 5 is RED.
- **predict (one rung, micro→milli):** the width minimizing the micro cell also minimizes `reduce-cooperative` at the **milli** rung, within **±10%** of the micro-predicted delta. A rank inversion between rungs is an **inconsistency** and spawns a work item.
- **kill:** no width beats 32 on the real graph — retire the lever, as nsg=2 was retired after four attempts (`R4`, `R12`).
- **memory gate:** formula §3. Note `membw_probe.rs:165-166` times the whole `execute_plan` with `Instant::now()` — upload, commit, wait **and** readback inside the window (`R18`). This card's micro numbers are therefore host-wall, not GPU time; the row must say so.
- **rollback:** revert to the C2.1 default value; no code change needed (that is the point of C2.1).
- **blast:** one config value. Zero source edits — the sweep is an env override (`omega/build.rs:79-85`).
- **observe:** the route census `Route::Cooperative` count (must stay 385) and the milli bucket.
- **reprove:** re-run the sweep at the chosen width plus 32 as control, interleaved.
- **log-row title:** `## ROW <NEXT> -- the cooperative-reduce width, swept 32 through 1024 from the config, with the real-graph cell as the arbiter`

---

#### C4.2 — the 547 elementwise ops, 7.35 ms

- **tier:** worker
- **depends_on:** C1.2, C3.2, C0.8 (inherits `prune_dead`)
- **worktree/branch/target:** `proxima-wt-elem` / `perf/elementwise-mass` / `.../proxima-wt-elem/target`
- **opens:** the route census from C1.2 (`Route::Elementwise` = **547** ops, **7.350 ms**, 13,437 ns/op `MEASURED R13`); `prune_dead`/`dead_resolved_nodes` cherry-picked from `216d925` per C0.8; `R3/M8` (rematerialization: only the `elements < 247` subset, **96 nodes**, is a certain win, 1196 → 1100; the aggregate set is a **loss**, and "rematerialize all ≤2-consumer nodes" is a **6x downside** on the slow ALU arm, `R4`)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-elem && git cherry-pick 216d925
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-elem && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-elem && PROXIMA_MAX_TOKENS=8 bash scripts/gpu-measure-lock.sh -- cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all -E 'test(profiles_one_real_decode_step_by_per_op_gpu_time)' --no-capture
  ```
- **expect (N):** `prune_dead` removes **N > 0** dead resolved nodes; `Route::Elementwise` count falls below 547. N==0 removed is RED (the pick did not take, or main already prunes).
- **predict (one rung, milli→bench):** at the **bench** rung, `step_wall_ms` falls by **≥ 0.8 ms**. Deliberately modest: `R12` proves dispatch-count reduction alone buys ~0, so this card's prediction is written against **GPU ms in the elementwise bucket**, not against op count.
- **kill:** elementwise ops fall but `step_wall_ms` does not move outside CoV — the same refutation `R12` already produced for dispatch count, now reproduced in our tree. That result is itself the row; the lever is retired, not retried.
- **memory gate:** formula §3.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-elem reset --hard origin/main`
- **blast:** `proxima-tensor/src/bind.rs` (`prune_dead`). Rematerialization is **not** attempted here — `R8` says only the 96-node subset is a certain win and the aggregate is a loss; it gets its own card after C5.5 re-partitions liveness (`R3/M8`: "Re-measure after M1 lands").
- **observe:** route census; milli bucket table.
- **reprove:** the third command.
- **log-row title:** `## ROW <NEXT> -- 547 elementwise ops for 7.35 ms; prune_dead removes the provably dead, and the wall does or does not follow`

---

### PHASE 5 — one bound plan, and write placement without a new Op

#### C5.1 — `bind` owns the packed layout; delete the Metal-only rewrite (one-RISC item 1)

- **tier:** worker
- **depends_on:** C1.2
- **worktree/branch/target:** `proxima-wt-onebind` / `fix/bind-owns-packed-layout` / `.../proxima-wt-onebind/target`
- **opens:** `omega/src/metal.rs:1003` (`let mut resolved = bind(program, &shapes, &effective_outputs)?;`) and `:1013` (`correct_packed_matmul_layouts(&mut resolved, …)`) — read in full, with the 8-line comment at `:1004-1012` explaining why; `proxima-tensor/src/bind.rs:1618-1647` (the function's doc: "`layout_of` has no way to get this right on its own"), `:1648` (`pub fn`); `proxima-tensor/src/bind.rs:1594-1606` (`layout_of`, `base += i64::from(axis.offset) * stride`); `proxima-tensor/src/cpu.rs:358` (calls `bind::bind` and **does not** apply the correction — verified: `grep -rn "correct_packed_matmul_layouts"` has **zero** hits in `cpu.rs`); `omega/src/metal.rs:375` (`packed_operands_of` lives in omega while `QuantizedBlock` lives in `proxima-tensor/src/cpu.rs:3084-3110` — the relocation `R18` names)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-onebind && grep -rn "correct_packed_matmul_layouts" omega/src proxima-tensor/src
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-onebind && sed -n '1000,1020p' omega/src/metal.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-onebind && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-onebind && bash scripts/gpu-measure-lock.sh -- bash scripts/proxima-tensor-gate.sh
  ```
- **The relocation question:** `packed_operands_of` (`omega/src/metal.rs:375`) computes the packed set in omega, while the type it is computed from (`QuantizedBlock`, `proxima-tensor/src/cpu.rs:3084-3110`) lives in proxima-tensor. Moving the packed-set computation next to its type lets `bind` take the packed set as an argument and produce the correct `Layout` directly — **no backend rewrites the plan after bind**. This is a relocation, not a new type: no `Op`, no `BoundOpKind`, no `Box<dyn>`.
- **expect (N):** after, `grep -rn "correct_packed_matmul_layouts" omega/src` returns **0**. A new test captures the plan **after each driver's own preparation** and asserts CPU and Metal plans are equal — currently they are **not** (that is the finding). Both gates green with nonzero counts.
- **predict (one rung, nano→micro):** a layout computed earlier, not differently, so the **micro** rung packed-family total is unchanged within **±0.5%**. Any movement means the layout changed and parity is at risk.
- **kill:** the plans still differ after the change — something else rewrites post-bind and the census must find it before item 1 can be claimed.
- **memory gate:** formula §3.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-onebind reset --hard origin/main`
- **blast:** `proxima-tensor/src/bind.rs`, `proxima-tensor/src/cpu.rs`, `omega/src/metal.rs`. **Test placement blocker (`R17`, to re-read before writing):** `struct Prepared` is private (`omega/src/metal.rs:859`, `resolved` at `:864`) and `Plan` fields are private (`:313-325`), so a plan-fingerprint test **cannot** live in `omega/tests/` alone — it needs `pub` accessors or an in-crate test module. The card must choose one and say which.
- **observe:** the plan-equality test's own N (nodes compared); `emit_is_deterministic_byte_equal` (`omega/src/msl.rs:4656`) as the seed for a golden-source companion.
- **reprove:** commands three and four.
- **log-row title:** `## ROW <NEXT> -- the plan Metal executes was never the plan CPU executes: metal.rs:1013 rewrote it after bind, and one-RISC item 1 was false`

---

#### C5.2 — accept a constant offset on the write side (one-RISC item 7, part 1)

- **tier:** worker
- **depends_on:** C5.1
- **worktree/branch/target:** `proxima-wt-offset` / `feat/write-side-offset` / `.../proxima-wt-offset/target`
- **opens:** `proxima-tensor/src/shape.rs:469-485` (`project_output_shape` — **THE line**, read verbatim: `[term] if term.coeff == 1 => Ok(iter_extents[term.axis as usize])`, else `NotLowerable { reason: "reduce output maps must be pure projections in v1" }`); `:441-467` (`bounds_check`, which **already** handles `axis.offset` on the read side); `proxima-tensor/src/map.rs:110-131` (write-direction convention doc: "`offset` carries the destination axis's static extent"; "CPU interpreter runs the reduce loop strictly sequentially, so a scatter never needs atomics"); `proxima-tensor/src/bind.rs:1011` (`build_scatter_out_layout`); `proxima-tensor/src/spec.rs:2306-2317` (the doc that names this constraint as the cause)
- **The two binary questions (no new type is minted here — this is the check that none is needed):**
  - *Q1, expression:* `Reduce { out_map: IndexMap::Affine(pattern_with_offset), … }` where the offset is a constant, folding into the existing `Layout.base` exactly as `layout_of` already folds the read side (`proxima-tensor/src/bind.rs:1594-1606`: `base += i64::from(axis.offset) * stride`). The destination field already exists; the emitter already reads it (`omega/src/msl.rs:2216, 2361, …`). **An existing primitive expresses it. No new type.**
  - *Q2, call site both ways:* before — `append_mistral_cached_layer` emits two `Reduce` blocks combining by online softmax over two ranges (`proxima-tensor/src/spec.rs:2336-2865`, 530 lines, 25 args). After — one `Reduce` writes `k_new` into the cache buffer at `base = cached_len * row_stride`, and attention reads one contiguous range. The caller can now express concatenation, which the source doc says it cannot. **New capability, achieved without a new type.**
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-offset && sed -n '441,490p' proxima-tensor/src/shape.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-offset && sed -n '110,131p' proxima-tensor/src/map.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-offset && bash scripts/gpu-measure-lock.sh -- bash scripts/proxima-tensor-gate.sh
  ```
- **expect (N):** a CPU-arm test writing two disjoint ranges into one buffer and reading back a contiguous concatenation passes; `NotLowerable { reason: "reduce output maps must be pure projections in v1" }` is **no longer** raised for a constant offset, and **is still** raised for a non-unit coefficient (the v1 restriction narrows, it does not vanish). N==0 new tests is RED.
- **predict (one rung, nano→micro):** CPU-only this card; no GPU rung is predicted. The **nano** claim is that the offset folds into `Layout.base` with zero added arithmetic in the inner loop; the **micro** prediction is that the CPU reduce path's throughput is unchanged within ±1%.
- **kill:** the offset cannot be proven loop-invariant at bind. **This is the sharp constraint** (`R17`, verified: `grep -rn write_row` = 0 relevant hits, so an injectivity-by-name convention has no enforcement): a scatter whose `indices` is a host-fed `Op::Input` is **UNPROVABLE at bind**; `Iota(coeff 1) + loop-invariant scalar` **IS** provable — that is exactly the `causal_mask` construction at `proxima-tensor/src/spec.rs:823-845`. If the write offset cannot be expressed in that provable form, the card stops; it does **not** fall back to a name convention.
- **memory gate:** formula §3. CPU-side; the KILL that matters is the 34 GB trap — no buffer sized from `context_length`.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-offset reset --hard origin/main`
- **blast:** `proxima-tensor/src/shape.rs`, `proxima-tensor/src/bind.rs`, `proxima-tensor/src/map.rs` (docs). GPU untouched — C5.3 is the GPU half. **No `IndexMap` variant is added**; `Affine(IndexPattern)` already carries `offset` (`proxima-tensor/src/map.rs:12`: slice = non-zero offset).
- **observe:** the new CPU test's assertions; the preserved `NotLowerable` for non-unit coefficients.
- **reprove:** the third command.
- **log-row title:** `## ROW <NEXT> -- "reduce output maps must be pure projections in v1" was the whole attention duplication; a constant write-side offset lifts it without a new Op`

---

#### C5.3 — GPU scatter: close the coverage gap the offset needs

- **tier:** worker
- **depends_on:** C5.2, C1.2
- **worktree/branch/target:** `proxima-wt-gpuscatter` / `feat/gpu-scatter` / `.../proxima-wt-gpuscatter/target`
- **opens:** `omega/src/msl.rs:933`, `omega/src/wgsl.rs:364`, `omega/src/cuda.rs:241` (all three raise `EmitError::ScatterNotSupported`, verified); `omega/src/error.rs:17,53` (the error's own doc calls it "a genuine, reachable gate"); `proxima-tensor/src/cpu.rs:6911` (`run_reduce_scatter` — **the CPU already implements it**) and `:7638` (`run_reduce`); `proxima-tensor/src/map.rs:175` (`IndexMap::scatter`), `:209` (`scatter_extent`), `:238` (`as_gather_from_output`); `omega/src/msl.rs:2216` (`long out_base` uniform)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-gpuscatter && grep -rn "ScatterNotSupported" omega/src
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-gpuscatter && sed -n '6911,6960p' proxima-tensor/src/cpu.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-gpuscatter && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  ```
- **expect (N):** a Metal↔CPU parity test over a scatter reduce passes with max abs delta ≤ **3.1e-6** (the parity bar `R12` set for the Q4_K body on real weights). `grep -rn "ScatterNotSupported" omega/src/msl.rs` returns **0** after; WGSL and CUDA keep theirs until C6.2. N==0 parity tests is RED.
- **predict (one rung, nano→micro):** a scatter reduce's **micro** cell lands within **±15%** of the equivalent affine reduce at the same shape — a scatter adds an index fetch and a bounds check, not a different memory pattern.
- **kill:** the scatter needs atomics on the GPU. The CPU convention (`proxima-tensor/src/map.rs:110-131`, read verbatim) is that "the CPU interpreter runs the reduce loop strictly sequentially, so a scatter never needs atomics" — a GPU has no such guarantee. If the write ranges are **not** provably disjoint, this needs atomics or a serialization, and the card stops. For the KV case they **are** disjoint by construction (each token writes its own row), which is the only case C5.5 needs — the card scopes to provably-disjoint scatters and says so.
- **memory gate:** formula §3.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-gpuscatter reset --hard origin/main`
- **blast:** `omega/src/msl.rs` only (Metal). `omega/src/{wgsl,cuda}.rs` keep the gate — closing them is C6.2, and doing it here would widen the blast radius across three emitters at once.
- **observe:** the route census gains a scatter route; the parity delta.
- **reprove:** the third command.
- **log-row title:** `## ROW <NEXT> -- the CPU has implemented scatter since run_reduce_scatter; all three GPU emitters refused it. Metal now does not.`

---

#### C5.4 — a persistent device-resident KV arena, by extending the checkpoint mapping

- **tier:** worker
- **depends_on:** C5.3, C3.1
- **worktree/branch/target:** `proxima-wt-kvarena` / `feat/kv-device-arena` / `.../proxima-wt-kvarena/target`
- **opens:** `omega/src/metal.rs:1786-1815` (`checkpoint_mapping_offset`, read in full — the doc states the KV buffers "never live inside the checkpoint's own mmap, so they fall through unchanged"; the body reads a **single** slot, `CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow())?`); `omega/src/backend.rs:402-414` (`register_checkpoint_mapping`); `omega/src/metal.rs:1848-1849` (`NOCOPY_BUFFERS: BTreeMap<(usize,usize), MetalBuffer>` keyed **(pointer, byte_length)**), `:1879` (`upload_block_no_copy`), `:1903` (`upload_block_no_copy_uncached`), `:1914` (`create_no_copy_buffer`), `:1616` (`is_page_aligned`), `:1606` (`page_size`); `omega/src/metal.rs:350-362` (`mark_resident`, classifies **by NAME** — read in full); `omega/src/metal.rs:991-1000` (strict `found != expected` → `InputSizeMismatch`, read in full; blocks an over-allocated buffer) and the CPU twin `proxima-tensor/src/cpu.rs:346-356`; `proxima-tensor/src/align.rs:42-46, 56-58, 69` (`AlignedBuffer`, page_size "must be a real host page size the caller queried itself … never hard-coded"); `proxima-model-interop/src/generate.rs:621-654` (`LayerCache`, three `extend_from_slice` at `:636-640`, handed whole as `QuantizedBlock::Float32` at `:642-653`)
- **The relocation question, answered:** generalizing the single `CHECKPOINT_MAPPING` slot to **N registered host spans** makes a page-aligned KV arena addressable by offset **through the same primitive** — §1 extend, don't add a peer. No `PlacedBuffer` type is minted (`grep -rn "PlacedBuffer" proxima-tensor/src omega/src` → **0 hits**, verified). The expression: `register_host_span(ptr, len)` → `(buffer, offset)` via the existing `upload_block_no_copy`.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-kvarena && sed -n '1786,1816p' omega/src/metal.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-kvarena && sed -n '984,1002p' omega/src/metal.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-kvarena && grep -rn "AlignedBuffer" proxima-tensor/src omega/src proxima-model-interop/src
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-kvarena && PROXIMA_MAX_TOKENS=32 bash scripts/gpu-measure-lock.sh -- cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' --no-capture
  ```
- **expect (N):** `kv_cache_upload_bytes` per steady token falls from **+262,144 B/token** (`MEASURED R13`, linear in context) to **0**; `nocopy_reuses` for KV blocks rises from **0** to **≥ 96/token** (3 blocks × 32 layers). `AlignedBuffer` gains its **first production caller** (verified today: `grep -rn "AlignedBuffer"` outside `align.rs` hits only `error.rs` doc, `lib.rs:240` re-export, `omega/examples/resident_gemv_topk.rs`, `omega/tests/metal_parity.rs` — **zero production callers**). N==0 nocopy reuses is RED.
- **predict (one rung, milli→bench):** at the **bench** rung, `block_upload` falls from **2.0 ms/token** (`MEASURED R13`) to **≤ 0.6 ms**, and the KV re-upload's context-linear growth (**~3.3%** `MEMORY R2`, re-measured as `+262,144 B/token` `MEASURED R13`) goes flat across steps 1-7.
- **kill:** `InputSizeMismatch`'s strict `found != expected` (`omega/src/metal.rs:991-1000`, read verbatim) rejects an over-allocated arena. Either the check learns a capacity-vs-length distinction **in the same change**, or the arena cannot be over-allocated and the card stops. Both the Metal and CPU twins (`proxima-tensor/src/cpu.rs:346-356`) must move together or the backends diverge again — the exact defect C5.1 just fixed.
- **memory gate:** formula §3, **the highest-risk card in the plan for memory**. The prior failure was exactly here: 34 GB from `context_length` (worktree only). Requirements: `capacity_tokens` is a **build-time key**; `build.rs` asserts `262,144 × capacity` against a literal ceiling; `capacity % 8 == 0` so `capacity × 2048` is a 16 KiB multiple (`R17`) — enforced by the existing `require_multiple_of_eight` (`omega/build.rs:59`); `page_size` comes from `omega::metal::page_size()` (`omega/src/metal.rs:1606`) or `libc::sysconf(_SC_PAGESIZE)` (`proxima-tensor` already has `dep:libc` under `std`, `R17`), **never** hard-coded, per `align.rs:56-58`. KILL if `device_allocated_bytes` at prefill exceeds `4.40e9`.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-kvarena reset --hard origin/main`. Because this card changes an allocation ceiling, rollback must be verified by re-running C0.2's memory observation, not just by the git command.
- **blast:** `omega/src/metal.rs` (N-span registry), `omega/src/backend.rs` (the register entry point), `proxima-model-interop/src/generate.rs` (`LayerCache` → `AlignedBuffer`), `proxima-tensor/src/cpu.rs` (the twin size check). **Widest blast radius in the plan.** Note `23e2e5e` on main now routes non-resident blocks to `upload_block_no_copy_uncached`, so KV blocks (not in `resident_names`, classified by name at `omega/src/metal.rs:350-362`) create a fresh no-copy buffer **every token** — no cache growth, but no reuse either (`R11/M2'`). The arena makes them resident.
- **observe:** `kv_cache_upload_bytes`, `nocopy_reuses`, `mapping_offset_uploads`, `device_allocated_bytes` — all in `token_breakdown_metal`.
- **reprove:** the fourth command, plus C0.2's memory observation.
- **log-row title:** `## ROW <NEXT> -- the KV cache round-tripped through the host every token; one registry of N host spans makes it device-resident through the primitive that already existed`

---

#### C5.5 — the minimal attention graph (one-RISC item 8: ≤23 real ops/layer)

- **tier:** worker
- **depends_on:** C5.4
- **worktree/branch/target:** `proxima-wt-minattn` / `perf/minimal-attention-graph` / `.../proxima-wt-minattn/target`
- **opens:** `proxima-tensor/src/spec.rs:2336-2865` (`append_mistral_cached_layer`, 530 lines, 25 args), sole caller `:6282`; `:2616-2617` (the comment); `:2610` (`is_future` consumed as `(is_future, "sw->swug")`); the incumbent's 23 ops (`llama-model.cpp:4691-4845` + `llama-graph.cpp:464-497, 514-616, 1071-1121, 1205-1253`, `R8 READ`), and its in-place write `ggml_cpy(k_cur, ggml_view_1d(k, n_tokens*n_embd_k_gqa, row_size*head_cur))` (`llama-kv-cache-unified.cpp:749-788`); `proxima-model-interop/src/generate.rs:1393-1400` (the 97 effective outputs)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-minattn && sed -n '2300,2340p' proxima-tensor/src/spec.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-minattn && bash scripts/gpu-measure-lock.sh -- bash scripts/proxima-tensor-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-minattn && PROXIMA_MAX_TOKENS=32 bash scripts/gpu-measure-lock.sh -- cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all -E 'test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)' --no-capture
  ```
- **expect (N):** real ops/layer ≤ **23** (`R8 READ`, the incumbent's count at `b25346221`); `op_count` ≤ **740** (`R8 DERIVED`: 32×23 + get_rows + rms_norm + mul + output mul_mat); effective outputs fall from **97** to **~1** (`R17`: turning K/V writes into in-place scatters shrinks the output set and changes every liveness partition); `generated_text` byte-identical to C0.2's. The single-range graph was already proven **488/488, 1196 → 939 BoundOps** (`R3/M11`) but required in-graph write placement, which C5.2/C5.3 now supply.
- **predict (one rung, milli→bench):** at the **bench** rung, `Route::Cooperative` + `Route::Elementwise` GPU ms fall by **≥ 4.0 ms** combined. **Explicitly NOT predicted: any wall win from the dispatch count itself** — `R12 MEASURED` refuted that (1194 → 616 moved wall by 0.036 ms). If the wall moves only via the two buckets, the prediction holds; if it moves via dispatch count, that is an **understanding-gap** and spawns a work item.
- **kill:** `generated_text` changes by one byte (§14). Or: the 97→1 output collapse changes `bound_op_retirement`'s partition (`omega/src/metal.rs:1128-1147`) such that peak-live device bytes exceed the 4.40e9 ceiling — the liveness change is the point, and it is also the risk.
- **memory gate:** formula §3, **with the liveness term recomputed**. The output set collapsing from 97 to ~1 means 96 formerly-unretirable buffers now retire — memory should **fall**. If `device_allocated_bytes` **rises**, the scatter is allocating where it should be aliasing. That inversion is itself the kill signal.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-minattn reset --hard origin/main`
- **blast:** `proxima-tensor/src/spec.rs` (the cached layer builder), `proxima-model-interop/src/generate.rs` (roots). `spec.rs` moved `+8735/-2836` on main nine commits ago and is the file the parallel branch also touches — the largest conflict surface in the plan.
- **observe:** `op_count` from `metal_decode_summary`; the route census partition; effective-output count.
- **reprove:** the third command.
- **log-row title:** `## ROW <NEXT> -- 37 ops per layer became 23: the duplication was one rejected write-side offset, not an attention algorithm`

---

### PHASE 6 — one emitter core, every backend every kind

#### C6.1 — one emitter core over the 4 kinds (one-RISC item 4)

- **tier:** worker
- **depends_on:** C1.3, C2.1, C5.3
- **worktree/branch/target:** `proxima-wt-emitcore` / `refactor/omega-emitter-core` / `.../proxima-wt-emitcore/target`
- **opens:** `omega/src/msl.rs:673-697` (`emit`), `omega/src/wgsl.rs:105`, `omega/src/cuda.rs:66` (the **3 shared types + 2 fns**: `Binding`, `PackedCodec`, `PackedOperands`, `gather_count`, `gather_slots`); the ~26 per-backend reimplementations `R5 READ` (validate, reduction_dims, bindings, grid_threads, entry_name, scalar_op_expr, fold_init_tokens, push_body_steps, preamble, kernel_signature, gather helpers, operand_read, render_*, reduce_is_cooperative) ≈ **78 near-duplicates**
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-emitcore && grep -c "^fn \|^pub fn " omega/src/msl.rs omega/src/wgsl.rs omega/src/cuda.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-emitcore && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  ```
- **expect (N):** the per-backend function count falls; the shared surface rises from **5** items to the count the card commits to. The route enum from C1.1 becomes the core's dispatch, so the **8 Metal body shapes are instantiations of one route**, not five hand-ordered `if let Some(..)` gates (`omega/src/msl.rs:751-754`, whose ordering is load-bearing today).
- **predict (one rung, nano→micro):** emission is text generation, off the measured path; `emit` at the **micro** rung is unchanged within **±5%** of `0.81 ms/token` (`MEASURED R13`). A refactor that moves it more has changed behavior.
- **kill:** the core needs `Box<dyn>` to abstract over backends — §20 forbids it. The backend-specific part is **text** (intrinsics, signature syntax); if it cannot be expressed as a trait with static dispatch or a generic parameter, the card stops rather than boxing.
- **memory gate:** formula §3. Emission is alloc-tier and touches no device.
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-emitcore reset --hard origin/main`
- **blast:** all three emitters at once — **the largest source-line blast radius** (msl 4712 + wgsl 1929 + cuda 1838 = 8479 lines `R5 READ`). This is why it is sequenced last among the structural cards: every earlier card's numbers must be banked first.
- **observe:** `emit_is_deterministic_byte_equal` (`omega/src/msl.rs:4656`) extended to a **golden-source** test — the emitted MSL must be byte-identical before and after the refactor, which is the only honest proof a refactor changed nothing.
- **reprove:** the second command plus the golden-source test.
- **log-row title:** `## ROW <NEXT> -- three unequal emitters shared 3 types and 2 functions and reimplemented 26 apiece; one core over the four kinds, backend text only`

---

#### C6.2 — every backend covers every kind (one-RISC item 5)

- **tier:** worker
- **depends_on:** C6.1
- **worktree/branch/target:** `proxima-wt-cover` / `feat/backend-kind-coverage` / `.../proxima-wt-cover/target`
- **opens:** `omega/src/cuda.rs:146-183` (`emit_cuda` **rejects** Iota and Constant with `CudaUnsupportedOpKind`; serial + cooperative reduce only — no tiled-GEMM, no packed row-block); `omega/src/wgsl.rs` (covers all 5 kinds but no tiled-GEMM / packed row-block); `omega/src/{wgsl.rs:364, cuda.rs:241}` (`ScatterNotSupported`, still raised after C5.3); `omega/src/backend.rs:1-52` (six `Backend` variants, two implemented; `vulkan`/`npu`/`ane` are name-only stubs)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-cover && grep -rn "CudaUnsupportedOpKind\|ScatterNotSupported" omega/src
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-cover && bash scripts/gpu-measure-lock.sh -- bash scripts/omega-gate.sh
  ```
- **expect (N):** a coverage matrix test asserting **4 kinds × 3 backends = 12** cells, all emitting. Today `emit_cuda` fails **2** of its 4 (Iota, Constant) and scatter fails **2** of 3 backends. N < 12 covered is RED, and the row states which cells remain.
- **predict:** none on the decode ladder — CUDA and WGSL are not on this host's measured path. This card's output is a **coverage N**, not a timing. It is explicitly off the rung ladder and says so.
- **kill:** a kind genuinely cannot be expressed in WGSL (e.g. no 64-bit integer for `out_base`) — then item 5 is false for that backend and the row records the constraint rather than pretending coverage.
- **memory gate:** formula §3 (no model process runs; the gate applies to `omega-gate.sh` itself).
- **rollback:** `git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-cover reset --hard origin/main`
- **blast:** `omega/src/{cuda,wgsl}.rs`. Metal untouched — no risk to the measured lane.
- **observe:** the 12-cell matrix test.
- **reprove:** both commands.
- **log-row title:** `## ROW <NEXT> -- CUDA rejected Iota and Constant and two of three backends rejected scatter; the 4x3 coverage matrix, asserted`

---

### PHASE 7 — the cells that do not exist on either side

`R9 READ` is unambiguous: "llama, ggml, ort and torch can beat us on gpu" is **MEASURED only for llama.cpp-Metal**. For ORT (CoreML EP) and torch (MPS) there is **no cell on either side**. The owner's premise is one-third measured. These cards make it falsifiable.

#### C7.1 — a torch-MPS decode cell

- **tier:** worker
- **depends_on:** C0.1, C0.2
- **worktree/branch/target:** `proxima-wt-torchmps` / `bench/torch-mps-decode` / `.../proxima-wt-torchmps/target`
- **opens:** `proxima-onnx/scripts/torch_reference/inference_bench.py:28-33` — read in full: `parse_args` defines **only** `--threads` and `--runs`. There is **no `--device`**, and the harness is image-shaped (`WARMUP_IMAGES = 50`). torch 2.13.0 with MPS is available in `proxima-onnx/scripts/torch_reference/venv` (`READ R0`).
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-torchmps && sed -n '25,40p' proxima-onnx/scripts/torch_reference/inference_bench.py
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-torchmps && ls proxima-onnx/scripts/torch_reference/
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-torchmps && bash scripts/gpu-measure-lock.sh -- proxima-onnx/scripts/torch_reference/venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
  ```
- **expect (N):** `torch.backends.mps.is_available()` → `True`. The existing harness supports **0** GPU arms — a `--device mps` arm is new work, and the harness measures an **image model**, not a 7B decode. **The honest finding this card produces first: there is no same-workload torch arm, and creating one is a separate build, not a flag.**
- **predict (one rung, micro→milli):** none until the arm exists. The card's first output is a scoping row, not a number.
- **kill:** torch-MPS cannot load the openchat Q4_K_S GGUF (it cannot — GGUF Q4_K is not a torch format), so a same-workload comparison requires either a different quantization or an f16 arm at a different byte rate. **The comparison is not apples-to-apples and the row must say so, or the scoreboard lies.**
- **memory gate:** formula §3, adapted: torch-MPS memory is measured by `torch.mps.current_allocated_memory()`, and an f16 7B model is ~14 GB against the Q4_K_S 3.9 GB — a KILL on this box's unified memory unless the arm is scoped smaller.
- **rollback:** delete the new arm; `inference_bench.py` is otherwise untouched.
- **blast:** one Python file under `proxima-onnx/scripts/torch_reference/`. Zero Rust.
- **observe:** the device string and allocated-memory readout.
- **reprove:** the third command.
- **log-row title:** `## ROW <NEXT> -- "torch beats us on GPU" had no cell on either side: the torch harness has no --device flag and measures an image model`

---

#### C7.2 — an ORT-CoreML cell

- **tier:** worker
- **depends_on:** C0.1
- **worktree/branch/target:** `proxima-wt-ortcoreml` / `bench/ort-coreml-decode` / `.../proxima-wt-ortcoreml/target`
- **opens:** `scripts/onnx_reference/bench.py:96` — read verbatim: `session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])`. **Hardcoded to CPU.** Note the path: the ledger's "onnx_reference" is at **repo-root `scripts/onnx_reference/`** (contents verified: `bench.py`, `diff_embeddings.py`, `export_model.py`, `README.md`, `run.sh`, `traffic_batch_bench.py`, `traffic_bench.py`), **not** under `proxima-onnx/scripts/`. onnxruntime is **not installed** in the torch venv (`READ R0`); an ORT source checkout exists at `~/repos/others/onnxruntime`.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-ortcoreml && grep -n "providers" scripts/onnx_reference/bench.py
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-ortcoreml && proxima-onnx/scripts/torch_reference/venv/bin/python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
  ```
- **expect (N):** the second command currently **fails** (onnxruntime absent) — that failure **is** the N, and it is recorded, not worked around. If installed, `CoreMLExecutionProvider` must appear in the provider list; **0** occurrences is RED for the CoreML arm.
- **predict:** none until a provider exists.
- **kill:** CoreML EP does not support the ops the model needs and silently falls back to CPU — then the "ORT beats us on GPU" claim is unfalsifiable as posed, and the row says which ops fell back (ORT's own profiling gives the per-node EP assignment).
- **memory gate:** formula §3, adapted to ORT's arena.
- **rollback:** revert `bench.py`'s provider list to `["CPUExecutionProvider"]`.
- **blast:** one Python file under `scripts/onnx_reference/`. Zero Rust. Installing onnxruntime into the venv is an **owner-authorized** action, not a Luna action.
- **observe:** the provider list; the per-node EP assignment from ORT profiling.
- **reprove:** both commands.
- **log-row title:** `## ROW <NEXT> -- "ORT beats us on GPU" had no cell either: bench.py:96 hardcodes CPUExecutionProvider and onnxruntime is not installed`

---

#### C7.3 — the GPU roofline constant, currently DEBT

- **tier:** worker
- **depends_on:** C0.1, C0.3
- **worktree/branch/target:** `proxima-wt-roofline` / `docs/gpu-roofline-constant` / `.../proxima-wt-roofline/target`
- **opens:** `proxima-tensor/docs/rooflines.md:396-479` (GPU lane; candidate ceiling = **DEBT, not measured**; only ratio available is vs incumbent achieved, 416.1 GMAC/s), `:751` (summary row), `:766-773` (the doc's own closing note that the GPU ratio "is not a gap-to-machine at all"), `:29` (the only GPU lane tracked is q4_K decode, marked "stale — not re-measured"), `:411` (`membw_probe` GPU bandwidth ceiling = DEBT); `omega/examples/membw_probe.rs:165-166` (times the whole `execute_plan` with `Instant::now()` — upload, commit, wait **and** readback inside the window, `R18`)
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-roofline && sed -n '396,420p' proxima-tensor/docs/rooflines.md
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-roofline && sed -n '160,170p' omega/examples/membw_probe.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-roofline && bash scripts/gpu-measure-lock.sh -- cargo run --release -p omega --example membw_probe --features metal,instrument
  ```
- **expect (N):** one measured GB/s ceiling for the M1 Max 32-core GPU, replacing the DEBT marker. The CPU lane has its constant (streaming triad **69.95 / 81.21 GB/s**, ROW 176 `R10`); the GPU lane has **none**. N==0 measured constants is the current state and is RED.
- **predict (one rung, nano→micro):** the measured ceiling is **≥ 228.9 GB/s** — the incumbent **achieves** that (`MEMORY R1`), so the machine's ceiling cannot be below it. A measured ceiling below 228.9 falsifies the probe, not the machine.
- **kill:** `membw_probe`'s window includes readback (`omega/examples/membw_probe.rs:165-166`), so it cannot measure a pure bandwidth ceiling as written. Fix the window first (use `GPUEndTime - GPUStartTime` as the op-timed path does at `omega/src/metal.rs:733-734`) or the constant is not a constant.
- **memory gate:** formula §3.
- **rollback:** revert the rooflines edit and the probe's timing window.
- **blast:** `proxima-tensor/docs/rooflines.md`, `omega/examples/membw_probe.rs`.
- **observe:** the probe's reported GB/s at increasing simdgroup counts — `R3/M5` measured rising marginal GB/s (**52 → 147 GB/s from 256 → 8001 simdgroups**), so the ceiling must be taken at saturation, not at one shape.
- **reprove:** the third command.
- **log-row title:** `## ROW <NEXT> -- the GPU lane had no machine constant, so every "3.88x" was a ratio to a competitor, never a gap to the silicon`

---

## 5. Dependency graph

```
C0.1 mutex ──┬─> C0.2 anchor ──> C0.3 byte accounting ──┐
             │                                          │
             └─> C0.4 patch capture ────────────────────┤
                                                        v
                                             C0.5 adjudicate Q4_K bodies (judge)
                                                        │
                                                        v
                                             C0.6 land Q4_K body
                                                        │
                          C0.2 ──> C0.8 adjudicate       v
                                   parallel branch ──> C0.7 wide coop reduce
                                   (judge)                │
                                        │                 │
   C0.6,C0.7,C0.8,C0.3,C0.5,C0.2 ──> C0.9 rows ──> C0.10 ai_docs
                                        │
                          C0.6,C0.8 ──> C1.1 Route ──> C1.2 census ──> C1.3 retire classify_kind
                                                            │                    │
                                                  C0.7 ─────┴──────> C2.1 sizing config
                                                                          │
                            C1.2,C2.1 ──> C3.1 bucket cached_len ──> C3.2 buffer pool
                                                                          │
                     C0.7,C2.1,C1.2 ──> C4.1 coop width sweep            │
                     C1.2,C3.2,C0.8 ──> C4.2 elementwise mass <──────────┘
                                                                          
                            C1.2 ──> C5.1 bind owns packed layout
                                          │
                                          v
                                     C5.2 write-side offset (CPU)
                                          │
                                          v
                              C1.2 ──> C5.3 GPU scatter (Metal only)
                                          │
                              C3.1 ──────>┤
                                          v
                                     C5.4 KV device arena
                                          │
                                          v
                                     C5.5 minimal attention graph (<=23/layer)

              C1.3,C2.1,C5.3 ──> C6.1 emitter core ──> C6.2 backend coverage

              C0.1,C0.2 ──> C7.1 torch-MPS      (independent lane)
              C0.1      ──> C7.2 ORT-CoreML     (independent lane)
              C0.1,C0.3 ──> C7.3 roofline       (independent lane)
```

**Critical path:** C0.1 → C0.2 → C0.3 → C0.5 → C0.6 → C1.1 → C1.2 → C5.1 → C5.2 → C5.3 → C5.4 → C5.5.
**Hard serialization:** every card that runs a process holds the C0.1 mutex, so no two measuring cards run concurrently regardless of graph independence. The Phase 7 lane is graph-independent but **mutex-dependent**, so it interleaves rather than parallelizes.

---

## 6. Rollback map

| Card | Rollback | Verified by | Irreversible? |
|---|---|---|---|
| C0.1 | delete `scripts/gpu-measure-lock.sh` | contended-run test | no |
| C0.2 | none (read-only) | — | no |
| C0.3 | `git checkout -- omega/src/metal.rs proxima-model-interop/src/generate.rs` | `kv_cache_upload_bytes` still `+262,144 B/tok` | no |
| C0.4 | delete patch dir; source worktrees never written | ten `--stat` re-runs | no |
| C0.5 | withdraw decision row | — | no |
| C0.6 | `reset --hard origin/main` | `generated_text` + `op_count` 1196 | no |
| C0.7 | `reset --hard origin/main` | coop count still 385 | no |
| C0.8 | withdraw decision row | — | no |
| C0.9 | `git checkout -- proxima-tensor/docs/` | `grep -c "ROW <NEXT>"` | no |
| C0.10 | `git checkout -- ai_docs/` | JSONL parse test | no |
| C1.1–C1.3 | revert each commit; `Route` survives a C1.3 revert | route census sums to `op_count` | no |
| C2.1 | `reset --hard origin/main` | generated `omega_sized.rs` | no |
| C3.1 | `reset --hard origin/main` — **both assertions revert with it** | `plan_hits == 0` restored | no |
| C3.2 | `reset --hard origin/main` | `newBufferWithLength` back to 1196/tok | no |
| C4.1 | revert config value only (no code change) | env-override rebuild | no |
| C4.2 | `reset --hard origin/main` | elementwise count back to 547 | no |
| C5.1 | `reset --hard origin/main` | `grep correct_packed_matmul_layouts` = 1 in metal.rs | no |
| C5.2 | `reset --hard origin/main` | `NotLowerable` raised again for offset | no |
| C5.3 | `reset --hard origin/main` | `ScatterNotSupported` at msl.rs:933 again | no |
| **C5.4** | `reset --hard origin/main` **plus re-run C0.2's memory observation** | `device_allocated_bytes` back to 4.152-4.163e9 | **allocation-ceiling change — git revert alone is not proof** |
| C5.5 | `reset --hard origin/main` | `op_count` back to 1196; outputs back to 97 | no |
| C6.1 | `reset --hard origin/main` | golden-source byte-equality test | no |
| C6.2 | `reset --hard origin/main` | 12-cell matrix back to 8 | no |
| C7.1–C7.3 | revert the single script/doc file | provider list / DEBT marker restored | no |

No card commits to main. Every rollback is worktree-local. The one card whose rollback needs a **measurement** to confirm, not just a git command, is C5.4.

---

## 7. Abandoned designs, each traced to the constraint that killed it

1. **`BoundOpKind::CachedAttention` (the parallel branch's fifth bound kind).** Killed by **Q1 of the type gate** — `out_layout.base` (`proxima-tensor/src/bind.rs:95-98`) plus a constant write-side offset expresses it, and C5.2 writes that expression. Also killed independently by its **own measurement**: +4.7 ms GPU for flat wall (`R12`). Also killed by **brief constraint "4 kinds"** and by the coverage burden it would place on CUDA/WGSL, which already fail coverage.
2. **`Op::Concat`.** Killed by **Q1** — two `Reduce` writing disjoint ranges of one buffer via `out_layout.base` is the expression. `grep -rn "Concat" proxima-tensor/src omega/src` returns **0**, and it stays 0.
3. **A `PlacedBuffer` type.** Killed by **§1 extend-don't-add-a-peer** — `checkpoint_mapping_offset` (`omega/src/metal.rs:1786-1815`) generalized from one slot to N registered host spans reaches the same capability through the primitive that exists. `grep -rn "PlacedBuffer"` returns **0**.
4. **`ScalarOp::GreaterEqual` for the tail mask.** Killed by `proxima-tensor/src/op.rs:51-53`, read verbatim: `ScalarOp` "is the one closed set in this crate that stays closed." The mask composes `Greater` with `Select`/`-inf`, mirroring `causal_mask` (`spec.rs:823-845`).
5. **Two mutually-exclusive cargo features + `compile_error!` to select a Q4_K body.** Killed by `scripts/omega-gate.sh`, read in full: it breaks **five of six** steps (`[2/6]`, `[3/6]`, `[4/6]`-arm-2, `[5/6]`-arm-2, `[6/6]`), not one as R18 estimated. Replaced by a build-time PROFILE axis via `omega/build.rs`'s `resolve_int`/`emit_sizing_consts`.
6. **`flock(1)`-based measurement mutex** (the synthesis_2 weld). Killed by the host: `which flock` → not found, `/opt/homebrew/bin/flock` absent. Replaced by a `python3` `fcntl.flock` + `os.execvp` shim (verified: `fcntl.flock` present).
7. **Leading with graph minimality / dispatch-count reduction.** Killed by `R12`'s one-variable toggle: 1194 → 616 dispatches moved wall 0.036 ms and made GPU 4.7 ms worse. Demoted to Phase 5 and re-justified as an enabler for in-place KV and bucket reduction, never as a dispatch-count win.
8. **nsg=2 / ggml two-simdgroup threadgroup regrouping.** Killed by **four** independent measured negatives (`R4`: −2.36%; `R12` ROW 267 and ROWs 249/251-254/259-260/265). Not re-proposed.
9. **Threading `op_setup` across the `MTLBuffer` boundary.** Killed at the type level: non-`Send` `MTLBuffer` (`R3/M9`). The plain-data half was already landed (`PROXIMA_ORCH_THREADS`, `perf/decode-orchestration-2`, timing unmeasured). Superseded anyway by C3.1+C3.2, which remove the work rather than parallelizing it.
10. **Threading the CPU/GPU overlap around `greedy_pick`.** Killed by a **true data dependency**: argmax depends on `waitUntilCompleted` (`R3/M7`). The only fix is moving argmax on-device; not threading. Deferred, not abandoned — it needs C5.3's scatter first.
11. **Rematerializing all ≤2-consumer nodes.** Killed by a measured **6x downside** on the slow ALU arm (`R4`). Only the `elements < 247` subset (96 nodes, 1196 → 1100) is a certain win, and it is deferred until after C5.5 re-partitions liveness (`R3/M8`).
12. **A retire no-op to simplify the buffer pool.** Killed by `omega/src/metal.rs:541-543`, read in full: the retire loop removes from a `BTreeMap<NodeId, DeviceBuffer>`; a no-op makes every operand lookup walk ~1196 entries.
13. **Arena sub-allocation within one device buffer.** Killed by the readback invariant at `omega/src/metal.rs:2360-2364`, read verbatim ("reading from the buffer's own start is always correct here"). Whole-buffer sharing is safe; sub-allocation is not.
14. **A `Mutex<BTreeMap>` route census, copying the `WIDTH_TILE_DECLINE` pattern.** Killed by **§21 lock-free** on a per-dispatch path. Replaced by a per-plan preallocated route vector filled once at plan time (the route is a property of the plan).
15. **`[Counter; N_ROUTES]` for the census.** Killed by `Counter` being non-`Copy` (`proxima-telemetry/src/metric/counter.rs:12-17`, contains `AtomicU64`) — N explicit const initializers with N distinct names. Replaced by `[AtomicU64; N]`.
16. **`route as usize` for census indexing.** Killed by `Route::Declined(reason)` being data-carrying — **E0605**. Replaced by `fn slot(&self) -> usize`.
17. **Deriving KV capacity from `ServingConfig::context_length`.** Killed by the reproduced 34 GB trap: `serving.rs:161` = `131_072` × 262,144 = 34,359,738,368 B. Replaced by a build-time key with a build-time byte assertion.
18. **Reusing `sealed-pass.sh` from `bench/sealed-pass`.** Killed by hardcoding: `REPO_ROOT` pinned to `proxima-wt-seal` (`:4`), four sibling worktrees hardcoded (`:25-28`), and `MACS_PER_TOKEN`/`WEIGHT_BYTES_PER_TOKEN_GB` as script constants (`R15`) — magic numbers outside any sizing config, §12. Confirmed absent from main (`ls scripts/` shows no `sealed-pass.sh`). C0.1's lock shim replaces its mutex role; the rate constants belong in a config, not a script.
19. **Comparing arms by `classify_kind` bucket.** Killed by two proven silent relabelings (its own doc at `omega/src/metal.rs:790-800`; `R12` ROW 263). Replaced by family comparison and, after C1.2, by route.
20. **Trusting `gpu_exec_ms` as kernel time.** Killed by reading `omega/src/metal.rs:545-554`: host ticks around `commit()` + `waitUntilCompleted()`. Every prediction now names its clock.

---

## 8. Open questions, each resolved by a named measurement

| # | Question | Resolved by | The N that answers it |
|---|---|---|---|
| Q1 | Is the paired-Q4_K family win (`R12`: 57 → 35 GPU) reproducible on **main**, or is it confounded by the rest of that 42-commit tree? | **C0.6**, milli rung | `reduce-packed-row-blocked` ms on main+body-only vs C0.2's 44.450 |
| Q2 | Which of the two Q4_K bodies is faster **at equal parity**? | **C0.5** nano probe + **C0.6** milli | one surviving body; parity ≤ 3.1e-6 |
| Q3 | Does bucketing `cached_len` change generated text? | **C3.1** | byte-identical `generated_text` over 32 tokens, `plan_hits` 31/32 |
| Q4 | Does the coop-reduce width win survive on the **real graph** (nsg=2 did not, four times)? | **C4.1** sweep + **C0.7** milli | `reduce-cooperative` ms at 5 widths, real-graph arm as arbiter |
| Q5 | Can a write-side offset be proven loop-invariant at bind, or does it need an unenforceable name convention? | **C5.2** | the `Iota(coeff 1) + loop-invariant scalar` construction lowers; a host-fed `Op::Input` index does not |
| Q6 | Does the 97→1 output collapse **reduce** peak device bytes (as predicted) or raise them? | **C5.5** memory gate | `device_allocated_bytes` direction of change |
| Q7 | Is `InputSizeMismatch`'s strict equality (`omega/src/metal.rs:991-1000`) removable without breaking the CPU twin (`cpu.rs:346-356`)? | **C5.4** | both gates green with a capacity ≠ length arena |
| Q8 | After C5.5, is the `elements < 247` rematerialization subset still a win? | deferred card, milli rung | `op_count` 1196 → 1100 with GPU ms direction |
| Q9 | What **is** the machine's GPU bandwidth ceiling? (DEBT since the lane began) | **C7.3** | one GB/s number at simdgroup saturation, ≥ 228.9 |
| Q10 | Can torch-MPS run the **same** workload, or is the comparison structurally apples-to-oranges? | **C7.1** | whether a Q4_K GGUF loads; if not, the row says so |
| Q11 | Does ORT's CoreML EP actually execute the model's ops, or silently fall back to CPU? | **C7.2** | per-node EP assignment from ORT profiling |
| Q12 | Does the emitter-core refactor change emitted text at all? | **C6.1** | golden-source byte-equality on the seed test (`omega/src/msl.rs:4656`) |
| Q13 | Has `perf/cached-attention-streaming` moved since `R12` was read? | **C0.8** | 42 commits; `physical.rs` +576; `bind.rs` +666 |
| Q14 | Do the ten dirty worktrees still hold what `R7` recorded? | **C0.4** | `proxima-wt-all` = 13 files +3859/-197, 6 untracked |
| Q15 | Which clock does each banked number use — host ticks or GPU timestamps? | **C0.3** + every card's row | every row names its clock; `gpu_exec` is host-wall around commit+wait |

Every MEMORY-tagged claim this plan leans on is re-verified before a later card depends on it: M1'/M2'/M6'/M6''/M7' were re-read on main during this pass (§1); M3 and M4's percentages are re-measured by C0.6 and C0.7 respectively; M5's rising-marginal-GB/s is re-measured by C7.3; M8 and M11 are deferred behind C5.5.

---

### Critical Files for Implementation

- `/Users/brianbruggeman/repos/slot-0/proxima/omega/src/metal.rs` — the driver: post-bind plan rewrite `:1013`, `classify_kind` `:785-826`, `encode_op` `:2179-2252`, retire loop `:541-543`, readback invariant `:2360-2364`, `checkpoint_mapping_offset` `:1786-1815`, `gpu_exec` host-tick window `:545-554`
- `/Users/brianbruggeman/repos/slot-0/proxima/omega/src/msl.rs` — the emitter: `emit` `:673-697`, gate ordering `:751-754`, bare geometry consts `:1017/:1030/:1046`, `grid_threads` `:1517-1560`, `Uniforms` `:2207-2218`, cooperative body `:3140-3194`, `ScatterNotSupported` `:933`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/shape.rs` — `project_output_shape` `:469-485`, the single rejection that causes the attention duplication, and `bounds_check` `:441-467` which already accepts the mirror-image offset on the read side
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/generate.rs` — the decode driver: plan cache key `:966`, `LayerCache` `:621-654`, `build_position_inputs` `:799-827`, 97 roots `:1393-1400`, `token_breakdown_metal` `:1721-1750`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/bind.rs` — the harnesses and **both** assertions Luna must invert: `:3002` bench rung, `:3052-3058`, `:3084` milli rung, `PROXIMA_MAX_TOKENS` `:2719` (**not** `proxima-tensor/src/bind.rs` — see correction C-A)
