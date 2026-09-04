I have read all six inputs in full and re-verified the load-bearing facts against main `4be2f3a` read-only (no `proxima-wt-*` entered): `flock` absent (`command -v flock` → exit 1, `/opt/homebrew/bin/flock` absent, `.gpu-measure.lock` absent); `ScalarOp` = 17 variants with **no** `GreaterEqual` and no `Less` (`op.rs:60-78`); the readback invariant verbatim at `metal.rs:2360-2364`; `struct Prepared` private with `resolved` private (`metal.rs:859-864`); `execute_plan_op_timed`'s own upload loop (`metal.rs:663-690`) separate from `execute_plan`'s; `named_blocks` `Vec::with_capacity(… + 3 + layer_caches.len()*3)` (`generate.rs:1313-1318`) and `symbols = [new_count, cached_len]` (`:1393`); `context_length: 131_072` (`serving.rs:161`); `align.rs`'s page-size contract ("never hard-coded here"); `map.rs:109-131` rejecting a **`Reduce`-wide destination-extent field**, not a write offset; `project_output_shape` `[term] if term.coeff == 1` (`shape.rs:469-485`) beside `bounds_check`'s existing `axis.offset` fold (`:441-467`); `bound_op_retirement`'s `if !outputs.contains(&node)` (`metal.rs:1128-1147`); the retire loop `device_buffers.remove` (`:541-543`); `Counter` holding a non-`Copy` `AtomicU64`; `omega-gate.sh` `[2/6] --all-targets --all-features` / `[3/6] nextest --all-features`; `packed_operands_of` at `metal.rs:375`; last row on main = **ROW 233** (`discipline.md:18736`); `git branch --list 'risc/*' 'gpu-risc/*'` = **0** and none of the 68 worktrees matches `risc`.

---

# synthesis_3 — GPU parity for proxima-tensor through omega, ONE RISC

---

# I. Diagnosis

## I.1 The baseline (R13, MEASURED 2026-09-03, main `4be2f3a`, loaded box, arms interleaved A B A B)

| cell | value | tag |
|---|---|---|
| llama.cpp-Metal `llama-bench -n 32 -p 0 -r 5 -t 8 -ngl 99` | 57.08 t/s = **17.52 ms/tok**, CoV 0.89% | MEASURED R13 |
| ours `step_wall_ms` | **67.92**, CoV 0.5% → **3.88x** | MEASURED R13 |
| ours `gpu_exec_ms` | **56.93**, CoV 0.7% → **3.25x** kernel-only | MEASURED R13 |
| `op_count` / incumbent dispatches | **1196** / ~740 (23 real ops/layer, R8) | MEASURED R13 / READ R8 |
| phases (ms/tok) | prepare 1.97, emit 0.81, block_upload 2.00, op_setup 3.90, pipeline_lookup 0.04, encode_dispatch 0.47, readback 0.22; `wall − gpu_exec` = 11.0 | MEASURED R13 |
| per-op buckets | packed-row-blocked **225 / 44.450 ms**; cooperative **385 / 9.113**; elementwise **547 / 7.350**; constant+iota 39 / 0.169 | MEASURED R13 (per-op mode, +7.3% inflation) |

Removable mass, in order (R13): **26.9 ms** Q4_K matvec above the incumbent's achieved streaming rate; **16.6 ms** non-matmul GPU; **11.0 ms** orchestration; **17.5 ms** irreducible weight streaming.

## I.2 The five structural facts that make every claim above unfalsifiable until they are fixed

**D1 — two instrument defects (R13).** `operand_bytes` sums `buffer.length()` (`metal.rs:694-700`), so since `7d09145` every weight operand reports **4,140,417,024** and `total_operand_bytes` reads 1.2 TB. `BLOCK_UPLOAD_BYTES` fires at `metal.rs:467` **before** the path match at `:469-486`, reporting **4,147,777,096 B/token** while `mapping_offset_uploads=291` / `copying_uploads=4`. **No GB/s row on main is valid.** R18 adds: the batched `gpu_exec` window is host ticks around `commit()`/`waitUntilCompleted()` (`metal.rs:546-554`) while only the op-timed path uses `GPUStartTime/GPUEndTime` (`:734`); and `membw_probe` times readback inside its window (`:165-166`). R17 adds the trap S2 missed: **`execute_plan_op_timed` has its own upload loop** (`metal.rs:663-690`), and every `op_profile_family` number comes from that path — so a fix applied only to `execute_plan` fixes the numbers nobody reads.

**D2 — the route is not a value (R5/M10, R12 ROW 263, R16).** `classify_kind` (`metal.rs:785-826`) buckets by substring of emitted MSL; its own doc (`:777-783`) admits the routing decision "is not exposed as its own accessor". R12 ROW 263 MEASURED that instrument relabelling **9/601 → 225/385** when a body changed. R13's own 225/385/547 split **is that instrument's output**. `Q4K_UNPACK_MSL`/`Q5K`/`Q6K` are concatenated with no delimiter (`msl.rs:1978-1982`), so "grep the Q4_K region" is undecidable (R16).

**D3 — brief item 1 is FALSE on main today (R16).** `metal.rs:1003` binds; `:1013` calls `correct_packed_matmul_layouts(&mut resolved, …)`; `cpu.rs:358` calls `bind::bind` and does not. **The bound plan Metal executes is not the bound plan CPU executes.** R17 adds: `Prepared` and its `resolved` are private (`metal.rs:859-864`), so this cannot be tested from `omega/tests/` without new `pub` surface, and a single `u64` fingerprint cannot answer "which nodes differ" — the comparison must be a per-op `Vec<u64>`.

**D4 — the plan cache cannot hit, by construction (R11 M6′, R13).** Key `(symbols[0], symbols[1]) = (new_count, cached_len)` (`generate.rs:1393`, `:966`); `cached_len` is `Extent::Symbolic(1)` on every KV leaf (`spec.rs:6216-6245`). `plan_hits=0 plan_misses=F` every run, and the harness **asserts** it (`bind.rs:3053-3055`). `ff749a0`'s `self.plans.clear()` on miss (`:973`) means the cache holds exactly one entry, so a hit requires the key to equal the **immediately preceding** key — a fact that changes the hit formula (§G5).

**D5 — the KV cache is host-resident and re-wired every token (R11 M2′).** `LayerCache {k_even,k_odd,v: Vec<f32>}` grows by `extend_from_slice`, so the base pointer moves; `NOCOPY_BUFFERS` is keyed `(pointer, byte_length)` (`metal.rs:1848`) and misses; `23e2e5e` routes non-resident blocks to `upload_block_no_copy_uncached` (`:1903`), creating a fresh `MTLBuffer` every token; `kv_cache_upload_bytes` grows **+262,144 B/token** (R13; = 32 layers × (2048 + 2048 + 4096) B, R17). `AlignedBuffer` (`align.rs:69`) exists with **zero production callers**.

## I.3 What this plan commits to, and what it refuses to claim

R12 and R13 jointly refute the brief's proposed mass ordering: **1194 → 616 dispatches moved wall 51.571 → 51.535 and moved `gpu_exec` UP 35.117 → 39.841.** Dispatch count is not the denominator. The spine is therefore: instruments → one plan → one route → the Q4_K body → KV residency → plan stability → the non-matmul bucket → write placement and the op count **last**, for the RISC and not for the milliseconds, and the plan says so out loud.

---

# II. One-RISC binding (brief items 1–8 → cards)

| # | brief clause | bound to | how it is proved |
|---|---|---|---|
| 1 | ONE bound plan (`&[BoundOp]`, 4 kinds) from ONE rewrite engine | **2.1** (RED) → **2.2** (fix) → **2.3** (re-anchor) → **3.4** (gate) | per-op `Vec<u64>` fingerprint captured after **each** driver's own rewrite through new `pub` accessors; pre-registered RED on main [R16, R17] |
| 2 | ONE route enum decided before emission, censused `(NodeId, reason)` | **3.1, 3.2, 3.3, 3.4** | unit-only `#[repr(u8)] Route` with `fn slot(&self) -> usize`; cold per-plan table + hot `[Counter; 8]`; `Σ ROUTE_DISPATCHES == ENCODE_DISPATCH_CALLS` |
| 3 | ONE emitter core over the 4 kinds, backend TEXT only | **8.1** (classification) → **8.3** (one kind, byte-identical) + continuation gate | golden emitted source byte-identical per route per backend |
| 4 | Every backend covers every kind | **3.3, 8.2** | `grep -c CudaUnsupportedOpKind == 0`; every remaining hole is a censused `Route::Declined(reason)`, never a silent `Err` |
| 5 | ONE sizing config owning every geometry constant | **7.1** (+ 4.1 `[q4k]`, 5.1 `[spans]`, 6.1 `[kv]`, 7.1 `[cooperative_reduce]`, `[packed_row_block]`) | a grep for policy consts in `omega/src` returns only GGUF wire-format facts |
| 6 | Write placement via existing `out_map`/`out_layout.base`, NOT a new Op | **9.1, 9.2** | structural injectivity at bind; the affine scatter **degenerates to a strided store**; no GPU scatter emitter, no atomics |
| 7 | Driver-level persistent-buffer alias, NOT a new type | **5.1, 5.2** | `register_checkpoint_mapping` generalised to N registered host spans; `AlignedBuffer` gets its first production caller |
| 8 | Llama graph at ≤23 real ops/layer | **9.3** | ops/layer census against R8's enumerated 23; `op_count` cell with wall in the criterion |

**Non-negotiables.** `Op` stays 5 variants (`op.rs:175-266`); `BoundOpKind` stays 4 (`bind.rs:221-264`); **`ScalarOp` stays 17** (`op.rs:60-78`, whose own doc at `:51-53` calls it "the one closed set in this crate that stays closed") — the tripwire in 0.9 covers **all four** sets, `Op`, `BoundOpKind`, `ScalarOp`, `IndexMap` [crit RS-4, SD-3]. No `Box<dyn>`, no `Concat`/`Pad`/`Tile`, no `PlacedBuffer`. `plan`/`execute` stays not-a-pipe (adjudicated 2026-08-30, `backend.rs:1-52`).

---

# III. Global protocol (binding on every card)

**G1 — tiers, no hybrids.** `hands` = runs the given commands verbatim, applies a diff a worker card wrote, records numbers, appends rows. **Hands never designs and never writes new code.** `worker` = writes source, tests, scripts, build steps against a fixed design. `judge` = adjudicates a pre-registered rule; produces no code. A card needing two tiers is split [crit j, round-2 j].

**G2 — worktrees: one per phase-branch, cards on it strictly sequential.** Verified collision-free: `git branch --list 'risc/*' 'gpu-risc/*'` = 0; none of the 68 worktrees matches `risc`. **13 worktrees, not 49** [conflict 7]. Every card's first commands are its worktree block, so a fresh Luna invocation inherits nothing and needs nothing:
```
WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>
TD=$WT/target
LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock
git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add $WT -b risc/<PH>-<slug> <base-ref>
mkdir -p $TD
```
Teardown: `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree remove --force $WT && git -C /Users/brianbruggeman/repos/slot-0/proxima branch -D risc/<PH>-<slug>`. **Never enter another card's worktree; never enter an R7/R15 `proxima-wt-*`** — cross-worktree reads use `git -C <path>`.

**G3 — the measurement mutex, which does not exist and must be built [conflict 1].** `flock(1)` is **absent on this Mac** (verified: `command -v flock` → exit 1; `/opt/homebrew/bin/flock` absent) — every `flock … -c '…'` weld in synthesis_2 would have failed. Card **0.1** writes a repo-local, versioned shim `scripts/gpu-measure-lock.sh` (python `fcntl.flock` on a file descriptor, then `os.execvp`), because a repo-local script lands in git, is re-provable by `§16`, and mutates no host state; `brew install flock` is recorded as the alternative on the row and is **not** the mechanism. The shim takes `--wait <seconds>` and exits **75** on timeout so a card fails loudly instead of hanging [crit SD-1]. Every command that **builds, probes, benches, runs the decode harness, or runs `scripts/omega-gate.sh`** is welded through it — `omega-gate.sh [2/6]/[3/6]` build and run `--all-features`, which includes the Metal suites, i.e. real GPU work [crit MS-5, k]:
```
cd $WT && CARGO_TARGET_DIR=$TD CARGO_TERM_COLOR=never PROXIMA_MAX_TOKENS=8 \
  bash scripts/gpu-measure-lock.sh $LOCK --wait 5400 -- <command>
```
**Any command in any card not written in this welded form is RED** [crit l]. The one card that runs an unbounded C++ build (10.4) caps `--parallel 4`, records peak RSS via `/usr/bin/time -l`, and is scheduled terminal [crit SD-2].

**G4 — the harness commands, budget pinned.** `decode_loop_max_tokens()` defaults to 24 when unset (`bind.rs:2719`); **every** command sets `PROXIMA_MAX_TOKENS=8` explicitly, including the CPU-oracle runs [crit l]. The cells:
```
# BENCH rung   -E "test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)"
# MILLI rung   PROXIMA_METAL_OP_PROFILE_STEP=3  -E "test(profiles_one_real_decode_step_by_per_op_gpu_time)"
# ORACLE       -E "test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)"
```
`--run-ignored all` is mandatory (`#[ignore]` at `bind.rs:3001`) and the test `return`s silently if the gguf is absent (`:3004-3011`) — a missing fixture exits 0, so **N==0 is RED everywhere**.

**G5 — the N contract, EOS-invariant, as a formula; never a literal token count** [round-2 j6]. From the harness's own quantities (`bind.rs:3051`):
```
F := generated.0.len() + usize::from(generated.2)      # forward calls taken
S := F - 1                                             # steady rows; step 0 is prefill, excluded from every mean
plan_hits + plan_misses == F
plan_misses == 1 + #{ steps k in 1..F : key(k) != key(k-1) }   # the cache is CLEARED on miss (generate.rs:973),
plan_hits   == F - plan_misses                                  # so a hit needs the IMMEDIATELY PRECEDING key
```
On main `key = (new_count, cached_len)` changes every step ⇒ `plan_hits == 0 && plan_misses == F` — R13 confirmed, and **written that way, never as literal `8`**. `F >= 2`, `S >= 1`, `op_count > 0`, `ENCODE_DISPATCH_CALLS / F == op_count`, else RED. No card assumes the prompt length; **0.8's `symbols` dump observes it.**

**G6 — arms, interleaving, CoV.** A B A B A B, never before-block/after-block; ≥3 runs at milli, ≥5 at bench; mean + CoV; the llama.cpp-Metal home-turf arm on every compare row, at **both** `-fa 0` and `-fa 1` (0.5). Every measuring card prints the loadout first (`pgrep -x` per named process + `sysctl -n vm.loadavg`). A loaded box is admissible (R13 returned CoV 0.5–0.9% at load 4.7–5.7) **provided the loadout is on the row**. **No kill criterion may be set inside a CoV band**; R13's bands (wall 0.5%, gpu 0.7%, incumbent 0.89%) are the floor until 0.5 re-measures them.

**G7 — the bench ladder.** nano (counts, hashes, emitted text, no device) → micro (one kernel: `q4k_matvec_probe`, `membw_probe`, `metal_vs_cpu`) → milli (`profiles_one_real_decode_step_by_per_op_gpu_time`, +7.3% per-op inflation quoted beside every per-op number, R13) → bench (the interleaved cell vs `llama-bench`). Every prediction is **exactly one rung ahead**; a card with nothing to measure writes `predict: none — this card produces records, not a measurement`. A miss kills the climb and is decomposed into *inconsistency* vs *understanding-gap* with a named work item.

**G8 — memory is a KILL on every card that runs a process, with the byte formula** (owner rule 2026-09-03). Observables that exist: `phys_footprint_bytes()` (`generate.rs:248`), `omega::metal::current_allocated_size()` (`metal.rs:270`), `kv_cache_upload_bytes` (`:1657`), `device_allocated_bytes` / `plan_cache_len` (`:1731`). Three named gates:
- **MG-1 (build/lint only, no process runs the model):** build exit 0; no new heap-holding `static`/`thread_local` without a bound stated at the site.
- **MG-2 (probe/bench, no checkpoint):** peak task RSS ≤ **400 MB** (R13 prefill 310–357 MB), raised **only** with the arithmetic stated on the card; `current_allocated_size()` returns to its pre-probe value ±2 MB.
- **MG-3 (decode harness), all four, any one failing is a KILL:**
```
(1) phys_footprint slope over steps 3..S <= 1_000_000 B/step        (R13: "no monotonic trend")
(2) device_allocated_bytes slope <= 262_144 + 2_097_152 B/step      (R13: +1-2 MB/token; the 262_144 term
                                                                     goes to 0 after 5.2 and stays there)
(3) peak device_allocated_bytes <= DEVICE_CAP_BYTES
    DEVICE_CAP_BYTES = 4_140_417_024                   # the ONE checkpoint mapping buffer (R13)
                     + kv_capacity_tokens * 262_144    # 32 x (2048 k_even + 2048 k_odd + 4096 v) (R17)
                     + 41_943_040                      # activations+uniforms: R13 steady 4.163e9-4.1404e9 = 22.6 MB x1.85
    at kv_capacity_tokens = 512 => 4_316_577_792 B (DERIVED)
(4) plan_cache_len <= 1 on every step                                (R13; the ff749a0 bound)
```
**The 34 GB trap, closed at build time:** `ServingConfig::context_length` defaults to **131_072** (`serving.rs:161`, verified) × 262_144 = **34,359,738,368 B** (DERIVED — the trap reproduced exactly from source). `kv_capacity_tokens` is a **build-time key** with a `build.rs` byte assertion in the style of `require_nonzero` (`build.rs:16`); **`context_length` never sizes an allocation**, and a runaway is a compile error. RSS ceiling rises to **540 MB** from 5.2 onward (400 + the 512×262,144 = 134,217,728 B arena), stated once with its arithmetic and inherited by every later card. Non-measuring cards record `memory gate: MG-1 — <reason>`, never blank.

**G9 — the crate gate.** Any card touching omega ends with `bash scripts/omega-gate.sh` **under the mutex**, recording `ran_count` (step [3/6]) and `passed_count` (step [6/6]). **Either == 0 is RED**; `ran_count` must not decrease card to card, and a card that deletes tests names them.

**G10 — correctness oracle (§14).** `generated_text` stays `"Here is a simple Python function that returns"` and the single-token oracle stays `2651` / `"known"` (`bind.rs:2797-2803`, llama.cpp's captured answer). Any drift is a KILL regardless of the number.

**G11 — row placeholders.** Main's last row is **ROW 233** (`discipline.md:18736`). Branches carry `## ROW <NEXT> -- <title>` only; the number is assigned at land time from `grep -n '^## ROW' proxima-tensor/docs/discipline.md | tail -1`. **Every cherry-pick uses `--no-commit` and drops the `discipline.md` hunk** so no literal ROW from R12's 234-267 numbering lands beside main's own [crit RB-1]:
```
git cherry-pick -x --no-commit <sha>
git checkout HEAD -- proxima-tensor/docs/discipline.md
git diff --cached --name-only | grep -c discipline.md    # must be 0, else RED
```

**G12 — provenance.** Every number cites its ledger section; computed numbers are tagged **DERIVED** and never anchor a mechanism claim (§18). MEMORY figures appear only as "MEMORY, superseded by R13" (R1 `op_setup 4.394` → R13 **3.90**; `prepare 2.087` → **1.97**; `readback 0.297` → **0.22**). R2's "KV re-upload ~3.3%" has no counterpart term in R13 and may not anchor a prediction. R1's `228.9 GB/s` is MEMORY and may not be a kill threshold until 0.6 measures a ceiling.

**G13 — commits.** Conventional, lowercase, imperative, <72 chars, one logical change, **every commit a green bisect point**. Behaviour-changing cards ship a default-off feature (or a build-time profile key) and `#[cfg]`-pair any assertion the change invalidates, so both arms compile and both pass. **No commit without owner authorization**; cards prepare the commit and stop.

**G14 — no time estimates, no verdicts.** Cards produce evidence rows.

---

# IV. The band ladder (every band DERIVED from its predecessor's own predicted delta)

Round 3's arithmetic audit [crit O-1, O-2, O-3] found S2's board band unreachable from its own terms. Every band below is re-derived here, once, and each card's `predict` is the row for that card and nothing else.

| after | `gpu_exec_ms` | `step_wall_ms` | ratio vs 17.52 | the delta and its source |
|---|---|---|---|---|
| R13 baseline | **56.93** | **67.92** | 3.88x | MEASURED R13 |
| **4.3** Q4_K body | [44.0, 48.0] | [55.0, 59.0] | 3.14–3.37x | −29% (R12 ROW 257, MEASURED on the family) to a −20% floor, applied to R13's 44.450 bucket — **DERIVED** |
| **5.2** KV residency | unchanged | [53.5, 57.5] | 3.05–3.28x | `block_upload` 2.00 → ≤0.5 (R13: 0.4 on step 2, where only weights uploaded) — **DERIVED** |
| **6.3** bucketing at the swept optimum | +δ_b | [51.8, 55.8] + δ_b | — | `prepare` 1.97 → ≤0.2 on hit steps (−1.7); δ_b = the trade cell's **MEASURED** GPU inflation on the three `kv_cache.*` families (R13: 3.116 ms total) |
| **6.5** device arena | +δ_b | [48.3, 52.7] + δ_b | — | `op_setup` 3.90 → [0.4, 0.8] (`pipeline_lookup` is already 0.04, R13) — **DERIVED** |
| **7.2** wide cooperative reduce | [42.2, 47.1] + δ_b | [46.5, 51.8] + δ_b | — | −20% of 9.113 (R3/M4, **MEMORY**, flagged; floor −10%) — **DERIVED** |
| **9.3** single-range attention | ±1.0 | ±1.0 | — | pre-registered ≈ **zero** wall movement (R12's control: 1194→616 moved 0.036 ms) |
| **11.2** the board, one prediction | **[42.2, 48.1]** | **[45.5, 53.8]** | **2.60–3.07x** | the sum of the above with δ_b ≤ 1.0 at the swept optimum — **DERIVED, the weakest number in this document** |

**The board's kill:** `step_wall_ms` **> 59.0** (band top + one CoV band + the single-range widening) ⇒ R13's decomposition is wrong somewhere and the row names **which bucket did not move, by counter**, before any further card is scheduled. The kill is set outside the band the cards' own terms produce, so it cannot fire on a tree behaving exactly as designed [crit O-2].

---

# V. The cards

## PHASE 0 — measurement truth
*Worktree `proxima-wt-risc00`, branch `risc/0-measure-truth`, base `4be2f3a`. Cards 0.1–0.9 are strictly sequential on it. Nothing downstream is attributable until this phase closes.*

### 0.1 — The measurement mutex, which does not exist `[B3 P0.1, crit SD-1, R18]`
- **tier** worker · **depends_on** —
- **worktree** `WT=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc00`; `TD=$WT/target`; `LOCK=/Users/brianbruggeman/repos/slot-0/.gpu-measure.lock`; create per G2 with base `4be2f3a`.
- **opens** nothing in-repo; `scripts/omega-gate.sh` (the weld target); `scripts/proxima-tensor-gate.sh`.
- **commands**
```
cd /Users/brianbruggeman/repos/slot-0/proxima && command -v flock; echo "flock_present=$?"
cd $WT && cat > scripts/gpu-measure-lock.sh   # python fcntl.flock(LOCK_EX) + os.execvp, --wait N -> exit 75
cd $WT && bash -n scripts/gpu-measure-lock.sh && shellcheck scripts/gpu-measure-lock.sh
cd $WT && touch $LOCK && bash scripts/gpu-measure-lock.sh $LOCK --wait 10 -- echo lock-ok
cd $WT && ( bash scripts/gpu-measure-lock.sh $LOCK --wait 60 -- sleep 5 & sleep 1; \
            bash scripts/gpu-measure-lock.sh $LOCK --wait 2 -- echo second; echo "exit=$?" )
```
- **expect** `command -v flock` reports **absent** — that is the finding and the reason this card is card one. The shim prints `lock-ok`. The contention test: the second invocation **blocks then runs**, and with `--wait 2` against a 5 s holder it **exits 75** and prints nothing. N = 3 assertions (present, blocks, times out); **N==0 is RED**.
- **predict (nano → micro)** the shim adds < 20 ms to a welded command (one `open` + one `flock` + `execvp`), i.e. below one part in 10³ of any measuring cell.
- **kill** the shim cannot obtain an exclusive lock from two processes ⇒ **the plan does not proceed**; an unserialised GPU box makes every CoV in this document a lie. Fallback recorded on the row: `brew install flock` (host mutation, not preferred — it is not in git and cannot be re-proved by §16).
- **memory gate** MG-1.
- **rollback** `git revert`; `rm $LOCK`. **blast** one new script; zero library code.
- **observe** the shim's exit codes; the lock file; the loadout capture that every later card reuses.
- **reprove** `cd $WT && bash scripts/gpu-measure-lock.sh $LOCK --wait 10 -- echo lock-ok`
- **log-row title** `the GPU box had no measurement mutex and macOS ships no flock(1): the lock is a repo script, not a homebrew formula`

### 0.2 — Both byte counters, fixed in BOTH upload loops `[S2 0.1, B3 P0.2+P0.3, crit HC-3, l]`
- **tier** worker · **depends_on** 0.1
- **worktree** `risc/0-measure-truth` (as 0.1)
- **opens** `omega/src/metal.rs:612` (`pub operand_bytes: u64`), **`:694-700`** (defect 1, verified verbatim: `.map(|(buffer,_offset)| buffer.length() as u64)`); **`:465-468`** (defect 2, verified: `counter!(BLOCK_UPLOAD_BYTES, …)` fires **before** the path match at `:469-486`); **`:663-690`** — `execute_plan_op_timed`'s **own** block-upload loop, verified this session, which is where every `op_profile_family` number comes from [crit HC-3]; `:1786-1815` `checkpoint_mapping_offset`; `:1879`/`:1903`/`:1914`; `msl.rs:294-556` (codec block constants); `proxima-model-interop/src/generate.rs:109-212` (every consumer), `:1723` (the printed field list).
- **commands** Defect 1: compute the operand's **tensor** bytes from `element_count(prepared.shapes.of(*source))` × dtype/codec bytes-per-element (f32 = 4; Q4_K = 144/256 = 0.5625, siblings at `msl.rs:294-556`); keep `bound_buffer_bytes` as a separate field so mapping-offset behaviour stays observable. Defect 2: delete the unconditional counter; record `BLOCK_COPIED_BYTES` / `BLOCK_NOCOPY_BOUND_BYTES` / `BLOCK_OFFSET_BOUND_BYTES` **inside** each terminal path, keeping `BLOCK_UPLOAD_BYTES` as their sum. **Apply the same three counters to `execute_plan_op_timed`'s loop at `:663-690` in this same commit.** Then G9's gate and the G4 MILLI and BENCH cells, welded per G3.
- **expect** N1 `COPIED + NOCOPY_BOUND + OFFSET_BOUND == BLOCK_UPLOAD_BYTES` on **every** step in **both** paths (identity assertion; a step where it fails is RED), and the steady-token sum equals R13's **4,147,777,096**. N2 `OFFSET_BOUND / COPIED > 100` (R13: 291 vs 4). N3 all **8** `op_profile_family` rows within **1%** of R13's shape-derived column (`ffn_up` 33.05 MB, `ffn_down` 34.00, `attn_q` 9.45, `attn_v`/`attn_k` 2.40, `output.weight` 107.5 MB); N < 8 is RED. N4 `total_operand_bytes` falls from ~1.2 TB to **≈ 4.07 GB/step** (DERIVED from R13's per-family column). N5 ≥3 unit tests (Q4_K operand == `rows*k*0.5625`; f32 == `elements*4`; an operand bound at a nonzero mapping offset reports the **tensor** length). **N==0 is RED.**
- **predict (milli → bench)** `step_wall_ms` unchanged within R13's CoV, mean in **[67.6, 68.3]** — this card changes accounting, not work. A timing move means the byte computation is on the hot path and is itself the finding.
- **kill** `step_wall_ms` moves > 2× R13's CoV (>1.0%) ⇒ hoist the computation to once-per-plan over `prepared.resolved` and re-measure; still moving ⇒ revert. Corrected per-family bytes disagreeing with R13's derived column by >5% ⇒ the shape derivation is wrong, **no GB/s row may be written anywhere and 0.6 is blocked**.
- **memory gate** MG-3. Six `AtomicU64` statics (+336 B static, 56 B each per `counter.rs:98`); zero heap.
- **rollback** `git revert`; instrument-gated only; reverting restores R13's (wrong) numbers exactly.
- **blast** `omega/src/metal.rs` (2 defects × 2 loops, 4 counter decls, 2 struct fields), `generate.rs` printers. Zero kernel, zero IR, zero feature.
- **observe** `operand_bytes`, `bound_buffer_bytes` (**NEW**, `metal.rs:694` and `:686`), `total_operand_bytes`, `gpu_ns_per_byte`, the three block counters (**NEW**, inside each terminal path in **both** loops), `mapping_offset_uploads`, `copying_uploads`.
- **reprove** the welded MILLI cell; the row's claim is N1's identity in both paths and N3's 8-family table.
- **log-row title** `two byte counters measured the wrong thing in two loops: operand_bytes was the checkpoint mapping, block_upload_bytes was the binding, and the per-op path had its own copy of the defect`

### 0.3 — A device-side `gpu_exec` window for the batched path `[B3 P0.4, R18]`
- **tier** worker · **depends_on** 0.2
- **opens** `omega/src/metal.rs:545-555` (host ticks around `commit()`/`waitUntilCompleted()`), `:734` (the device window the op-timed path already uses).
- **commands** add `GPU_DEVICE_NS` from `GPUEndTime()-GPUStartTime()` on the one batched command buffer; report **both**, neither replacing the other. The difference is commit + wakeup cost, today silently inside "kernel time". Run G9 + the G4 BENCH cell.
- **expect** `gpu_device_ms <= gpu_exec_ms` on every step (identity assertion; a violation is RED); `gpu_exec_ms` mean reproduces R13's **56.93 ± 0.7%**. **N==0 (no steps parsed) is RED.**
- **predict (milli → bench)** `gpu_exec_ms − gpu_device_ms` ≤ **1.0 ms/token**. A larger gap relocates mass from the GPU bucket to orchestration and **rewrites §I.1's decomposition** — which is the finding, reported first.
- **kill** `gpu_device_ms` returns 0 or negative on any step (`GPUStartTime` unpopulated) ⇒ unusable; revert and record the negative.
- **memory gate** MG-3. **rollback** `git revert`; instrument-only. **blast** instrument-only.
- **observe** `gpu_exec_ms`, `gpu_device_ms` (**NEW**, `metal.rs:546-554`).
- **reprove** the welded BENCH cell.
- **log-row title** `the batched gpu_exec window was host ticks; the device window is <N> ms narrower`

### 0.4 — `scripts/gpu-cell.sh`: parameterized, budget-pinned, N-asserting, memory-gated `[S2 0.2, B3 P0.5, crit MS-4]`
- **tier** worker · **depends_on** 0.3
- **opens** `git show bench/sealed-pass:scripts/sealed-pass.sh` — verified hardcoded (`REPO_ROOT` = `proxima-wt-seal` at `:4`, four sibling worktrees `:25-28`, `MACS_PER_TOKEN=7110402048`, `WEIGHT_BYTES_PER_TOKEN_GB=3.9996`); `scripts/omega-gate.sh:37-47` (the assert-nonzero pattern to mirror); `bind.rs:2719`, `:3002`, `:3084`; `generate.rs:107, 248, 1657, 1723-1764`.
- **commands** write `scripts/gpu-cell.sh <worktree-abs> <target-dir> <feature-list> <runs>` **fresh** (do not port the hardcoded script). It: (1) prints the G6 loadout; (2) takes the 0.1 mutex; (3) exports `PROXIMA_MAX_TOKENS=8` **inside the script**; (4) interleaves incumbent and ours A B A B; (5) extracts named `key=value` fields with `grep -oE`, **never a `sed` backreference** (R13 records `plan_hits=10,20..70` as a `\10` artifact — a fabricated finding); (6) asserts G5's contract and exits nonzero on `F<2 || S<1 || op_count==0`; (7) computes MG-3's four clauses and exits nonzero on a breach; (8) emits one machine-readable line per arm. Then `bash -n`, `shellcheck`, and one real pass.
- **expect** `grep -c 'proxima-wt-' scripts/gpu-cell.sh == 0`; ≥2 arms enumerated; `runs × S` rows per arm; a run breaching a memory clause exits nonzero. **N==0 is RED.**
- **predict (milli → bench)** the script's own ours/llama cells reproduce 0.5's numbers inside 0.5's measured CoV band.
- **kill** the script's numbers disagree with 0.5 beyond that band ⇒ it is measuring something else (budget, arm order); fix before any later card uses it.
- **memory gate** MG-3 — the script *implements* it; its own run asserts clause 3 at `kv_capacity_tokens = 0` (= 4,182,360,064) against R13's steady 4.152–4.163 GB.
- **rollback** `git revert`; `scripts/` only. **blast** one new script.
- **observe** `plan_hits`/`plan_misses` (`generate.rs:1764`, `bind.rs:3046`), `op_count` (`:107`), `phys_footprint_bytes`, `device_allocated_bytes`, `kv_cache_upload_bytes`, per-arm CoV, loadout.
- **reprove** `cd $WT && bash scripts/gpu-cell.sh $WT $TD std,metal,instrument 5`
- **log-row title** `the GPU cell is a script, not a memory: parameterized, budget-pinned, N-asserting, memory-gated`

### 0.5 — Re-seal R13 on fixed instruments, and add the `-fa 1` incumbent arm `[S2 0.3+0.4, B3 P0.5, crit b, O-2]`
- **tier** hands · **depends_on** 0.4
- **opens** `scratchpad/baseline-2026-09-03/{llama_A1..3,ours_B1..3,profile}.log`; `common/common.h:328` at checkout `b25346221` — flash attention **OFF** by default, so R13's 17.52 is the incumbent's *default*, not its *best*; `-fa 1` also changes the KV layout (`v_trans = !flash_attn`), so it is a **second incumbent**, not a variant (R8).
- **commands** add the `-fa 1` arm to `scripts/gpu-cell.sh`; run three arms `llama -fa 0` / ours / `llama -fa 1` interleaved A B C A B C, 5 rounds, every command welded per G3, loadout before and after.
- **expect** 3 arms × 5 runs, all parsed; `op_count == 1196`; `plan_hits == 0 && plan_misses == F` **written as the formula** [round-2 j6]; `generated_text` identical. **N==0 is RED.**
- **predict (bench, the anchor rung)** `step_wall_ms` within ±3% of **67.92**, `gpu_exec_ms` within ±3% of **56.93**, and the CoV **tightens** if the box is quieter rather than the means moving. Pre-registered on the second incumbent: **`-fa 1` is faster** — their decode path at this checkout is `mul_mat(K,Q)` → `soft_max_ext` → `mul_mat(V)`, three dispatches per layer against flash attention's one (R8).
- **kill** `op_count != 1196` or `step_wall_ms` outside ±5% of 67.92 ⇒ the box or the tree is not what R13 measured; STOP and name the difference. **This is not a re-baseline** — R13 stays THE baseline; this card produces the band. If `-fa 1` wins, **it becomes the home-turf incumbent for every later ratio, the standing gap is worse than 3.88x, and R13/R1/R10's ratios re-base against it, loudly.** If the build rejects `-fa 1`, record the exact stderr and proceed with `-fa 0` as sole incumbent with the reason on the row.
- **memory gate** MG-3 per run, all four clauses; a breach on **unmodified main** is RED and stops the plan. The incumbent arms' peak RSS via `/usr/bin/time -l` (an incumbent that swaps invalidates interleaved cells sharing the box).
- **rollback** none (measurement); one script line for the arm. **blast** `scripts/gpu-cell.sh` and the denominator of every ratio in `discipline.md`/`rooflines.md`.
- **observe** all seven phase counters, `gpu_device_ms` (0.3), `op_count`, `plan_hits`/`plan_misses`/`plan_cache_len`, `kv_cache_upload_bytes`, `device_allocated_bytes`, RSS, CoV, loadout.
- **reprove** the three-arm interleaved sweep.
- **log-row title** `R13 replicated on fixed instruments, and the second incumbent arm lands before any board prediction: flash attention is off by default at b25346221`

### 0.6 — The streaming ceiling: a copy arm, device-timed, readback outside the window `[S2 0.5, B3 P0.6, crit V-4, R18]`
- **tier** worker · **depends_on** 0.3
- **opens** `omega/examples/membw_probe.rs:139-146` — the Metal arm is a **reduce-to-scalar**, which is why `rooflines.md:411` records the ceiling as **DEBT**; `:165-166` — `Instant::now()` wraps the whole `execute_plan`, so upload, commit, wait **and readback** are inside the window (R18); `rooflines.md:396-479`, `:751`, `:766-773`.
- **commands** add a copy arm — `Op::Elementwise { body: ScalarOp::Identity, operands: [(src, affine identity)] }` over N f32 into an N f32 output — at two sizes (64 MiB, 256 MiB), 21 runs, min reported, two-size marginal. Time with 0.3's `GPUStartTime/GPUEndTime` so readback is **outside** the window; validate correctness on a separate untimed run. Print **both** denominators on every line: `read_only_gbs = 4N/gpu_s` and `traffic_gbs = 8N/gpu_s`.
- **expect** N = 3 existing arms + 2 copy arms × 2 denominators + 1 marginal = **8 rows**; `readback_bytes == 0` inside the timed window (the assertion that proves the window is clean); a marginal row with `delta_ms <= 0` is RED; `read_only_gbs` below the CPU multi-thread triad (69.95/81.21 GB/s, ROW 176) is RED. **N==0 is RED.**
- **predict (micro → milli)** the copy arm's `traffic_gbs` **exceeds 228.9** (R1's MEMORY figure for the incumbent's *achieved* decode rate), establishing for the first time that 228.9 is not the machine ceiling. If it lands **below** 228.9, the reduce probe was never measuring bandwidth and 228.9 becomes the ceiling estimate by default — the more interesting outcome, reported first (§19).
- **kill** copy-arm CoV > 5% over 21 runs ⇒ raise both sizes 4× and re-run once; still >5% ⇒ report single-size numbers with the contamination stated and leave the ceiling DEBT. **No spec-sheet figure is ever substituted** (§18; `rooflines.md:411` refused that once already).
- **memory gate** MG-2, **raised to 1.2 GB with the arithmetic on the row**: 256 MB source + 256 MB destination + 256 MB host mirror + 400 MB baseline; asserted via `current_allocated_size()` before and after. The decode caps are deliberately exempt and the exemption is stated.
- **rollback** `git revert`; example + one doc section. **blast** `omega/examples/membw_probe.rs`, `rooflines.md:396-479, :751, :766-773`.
- **observe** `gpu_device_ms` per arm, `readback_bytes == 0`, both GB/s columns with CoV.
- **reprove** `cd $WT && CARGO_TARGET_DIR=$TD bash scripts/gpu-measure-lock.sh $LOCK --wait 5400 -- cargo run --release -p omega --features metal,cpu,instrument --example membw_probe`
- **log-row title** `the GPU streaming ceiling stops being DEBT: a copy, device-timestamped, readback outside the window, both denominators printed`

### 0.7 — Quarantine the uncommitted worktree diffs INSIDE git `[S2 0.6, crit RS-6, round-2 j6]`
- **tier** hands · **depends_on** 0.1
- **opens** R7's table (10 worktrees; `proxima-wt-all` 13 files +3859/−197 **and 6 untracked**); the three based on other HEADs: `gpuker@bfc150d`, `q4k@a2175c2`, `lat@14f1304`.
- **commands** run verbatim from `$WT`; **no `sh -c`, so `$wt` is expanded by the shell that owns it** [crit RS-6]; **each apply-check runs against that worktree's own recorded HEAD** [round-2 j6]:
```
R=$WT/docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered
mkdir -p $R
for wt in all drive rules splitk place merge gpudisp gpuker q4k lat; do
  W=/Users/brianbruggeman/repos/slot-0/proxima-wt-$wt
  git -C $W rev-parse HEAD                       > $R/$wt-head.txt
  git -C $W diff                                 > $R/$wt-tracked.patch
  git -C $W status --porcelain                   > $R/$wt-status.txt
  mkdir -p $R/$wt-untracked
  while IFS= read -r -d '' f; do
    mkdir -p "$R/$wt-untracked/$(dirname "$f")"
    cp "$W/$f" "$R/$wt-untracked/$f"
  done < <(git -C $W ls-files --others --exclude-standard -z)
done
cd $WT && git add docs/bench-campaigns && git status --porcelain
for wt in all drive rules splitk place merge gpudisp gpuker q4k lat; do
  git -C /Users/brianbruggeman/repos/slot-0/proxima-wt-$wt apply --check \
      $R/$wt-tracked.patch 2>&1 | sed "s/^/$wt /"
done
```
- **expect** N = **10** non-empty `*-tracked.patch`, **10** `*-head.txt`, and a distinct `$wt-untracked/` tree for **every** worktree whose `git status` shows `??` (R7 records 6 files under `all`). **N==0 is RED**; an empty patch for a worktree R7 lists as dirty is RED; a shared or missing untracked tree is RED. Patches and untracked **contents** are committed to this branch — never `/tmp`, which is OS-cleared.
- **predict** none — this card produces records, not a measurement.
- **kill** a patch fails `git apply --check` **at its own recorded HEAD** ⇒ that worktree mutated since the R7 audit; re-audit before anything is landed from it.
- **memory gate** MG-1 — no build, no process. **rollback** `git revert` one commit; the patches remain in history, which is the point. **blast** `docs/bench-campaigns/` only.
- **observe** patch count, per-patch line counts vs R7's table, per-worktree HEAD, ten `apply --check` exit codes.
- **reprove** the apply-check loop above.
- **log-row title** `the measured-but-uncommitted wins enter git from each worktree's own HEAD, untracked files included and not colliding`

### 0.8 — The harness N-contract, the hit formula, and the `symbols` dump `[S2 0.12, B3 G5, crit k, round-2 j6]`
- **tier** worker · **depends_on** 0.5
- **opens** `proxima-model-interop/src/bind.rs:3040-3059` — `let forward_calls_taken = generated.0.len() + usize::from(generated.2);` `:3051`, `assert_eq!(runtime.plan_hits, 0, "cached_len grows every decode step, …")` `:3053-3055`, `assert_eq!(runtime.plan_misses, forward_calls_taken, …)` `:3057-3059`; `generate.rs:905` (the **single** `plan_hits` definition), `:968` (increment), **`:973` `self.plans.clear()`** — the fact that makes the hit formula *consecutive-key*, not *distinct-key* [correction to S2 G4]; `:966` (the key), `:1393` (`symbols`).
- **commands** do **not** weaken the assertions. Restate them as G5's formula (`plan_hits + plan_misses == F`; `plan_misses == 1 + #{consecutive key changes}`; `plan_misses >= 1`). Add a `symbols`-per-step dump behind `instrument` so the key sequence is **observed, not assumed**. On main the formula reduces to `plan_hits == 0`, so today's assertion is preserved exactly; **6.2 changes it in 6.2's own commit, `#[cfg]`-paired.**
- **expect** the harness prints exactly `F` breakdown lines; on main `plan_hits == 0`, `plan_misses == F`, `stopped_by_eos` recorded; the assertion still fires on an artificially injected hit; the dump prints one `(new_count, symbol1)` pair per step. **No card anywhere assumes the prompt length** — it is read from the dump. **N==0 is RED.**
- **predict (nano → micro)** the dumped pairs on main are `(prompt_tokens, 0)` then `(1, cached_len)` with `cached_len` strictly increasing, so every consecutive key differs — which *is* the mechanism `plan_hits=0` is, and the sequence 6.1 changes.
- **kill** `plan_hits != 0` on unmodified main ⇒ the counter's semantics moved since R13; stop and re-derive before 6.1 is designed.
- **memory gate** MG-3 (it runs the harness). **rollback** `git revert`; test-only. **blast** the `bind.rs` test module only, edited once, here.
- **observe** `plan_hits`, `plan_misses`, `plan_cache_len`, `tokens_generated`, `stopped_by_eos`, the `symbols` dump (**NEW**, `generate.rs:966` region under `instrument`).
- **reprove** the welded BENCH cell + `grep -o 'symbols=([0-9]*,[0-9]*)' runs/cell-08-*.txt`
- **log-row title** `the harness asserts the defect as a formula: the cache is cleared on miss, so a hit needs the immediately preceding key`

### 0.9 — The RISC's cardinality, the row protocol, ai_docs, and a clean tree `[S2 0.9+0.10+0.13, B3 P0.7, crit RS-4, SD-3]`
- **tier** worker · **depends_on** 0.5, 0.6
- **opens** `proxima-tensor/src/op.rs:166` — the doc header reads **"The four generators"** directly above `pub enum Op` at `:175`, which has **5** variants; `:60-78` (`ScalarOp`, **17**, verified this session); `bind.rs:221-264` (`BoundOpKind`, 4); `map.rs:134-152` (`IndexMap`, 2); `discipline.md:18736` (**ROW 233**); `ai_docs/AGENT.md` ("add records … instead of bypassing"); `ai_docs/{index,task-routes,invariants}.jsonl` (R0: zero tensor/omega/GPU records); `git status --porcelain` on main (`?? proxima-onnx/scripts/torch_reference/venv/`).
- **commands** (a) correct the `op.rs:166` header and add **four** tests written as exhaustive `match`es over constructed values with **no `_` arm** — over `Op`, `BoundOpKind`, **`ScalarOp`**, and `IndexMap` — so adding a variant to any of the four **breaks the build** [crit RS-4, SD-3]; (b) write `docs/bench-campaigns/2026-09-03-gpu-one-risc/row-protocol.md` per G11; (c) append the ai_docs records: **index** ×1 (`proxima.omega.gpu_lane`), **task-routes** ×1 (`gpu-decode-perf`, `done_when` = ["every GB/s row cites a measured denominator", "every route cites a `Route` value, never a source substring", "MG-3 recorded on every decode cell", "every geometry constant traces to omega-runtime.toml", "the census sum equals ENCODE_DISPATCH_CALLS"]), **invariants** ×5 (one bound plan; route-is-a-value; no lock on a per-dispatch path; KV arena bytes are a build-time bound; profile bytes are tensor bytes) — each with `evidence_required`; (d) add `venv/` to `proxima-onnx/scripts/torch_reference/.gitignore`.
- **expect** N1 four exhaustive-match tests compile with no wildcard; `grep -c "The four generators" proxima-tensor/src/op.rs == 0`. N2 `jq -c .` parses all three JSONL files; **7 new records**; `bash ai_docs/query.sh gpu-decode-perf` returns ≥1 row. N3 `git status --porcelain | wc -l == 0`. **N==0 on any of the three is RED.**
- **predict (nano → micro)** the four matches compile today, i.e. the cardinalities are 5/4/17/2 as R5 and this session read them, and `grep -ci "omega\|metal\|gpu" ai_docs/task-routes.jsonl` moves from **0** to ≥1.
- **kill** a match needs a wildcard ⇒ a cardinality is not what was read; stop and re-derive the one-RISC binding. Malformed JSONL or an empty query ⇒ read `query.sh` and fix the record shape rather than bypassing the index.
- **memory gate** MG-1 — doc, tests and JSONL. **rollback** `git revert`. **blast** `op.rs` doc + tests, three JSONL files, one ignore file, one docs file.
- **observe** the four variant counts; record counts per file; query hit count; the porcelain line count.
- **reprove** `cd $WT && CARGO_TARGET_DIR=$TD bash scripts/gpu-measure-lock.sh $LOCK --wait 900 -- cargo nextest run -p proxima-tensor --features std -E 'test(risc_cardinality)'` + the three `jq` commands.
- **log-row title** `the RISC's doc said four over five variants and its tripwire watched two of four closed sets; now all four are compile errors to change, and the GPU lane enters ai_docs`

---

## PHASE 1 — recover what exists into git; adjudicate the parallel branch

### 1.1 — Adjudicate `BoundOpKind::CachedAttention` `[S2 0.7, B3 P1.1]`
- **tier** judge · **depends_on** 0.7
- **worktree** none — read-only adjudication via `git -C /Users/brianbruggeman/repos/slot-0/proxima show …`; **never enter a `proxima-wt-*`**.
- **opens** `git diff main..perf/cached-attention-streaming -- proxima-tensor/src/bind.rs proxima-tensor/src/physical.rs`; `git show perf/cached-attention-streaming:failure-cached-attention-matcher.md`; against them `bind.rs:221-264`, `op.rs:175-266`, workspace `AGENTS.md` ("we should not be adding arbitrary rules/code for specific instances"). Read the **diff hunks**, never the commit list.
- **the ruling, four independently sufficient grounds** (1) a fifth variant of a closed 4-variant set minted for one model's attention shape (AGENTS.md); (2) §1 — the expression already exists (two `Reduce`s with an online-softmax combine, `spec.rs:2596-2720`), so the type buys no caller capability; (3) MEASURED not to buy wall time — R12: 51.535 ON vs 51.571 OFF (CoV 1.75–2.16%) with dispatches 1194 → 616 and GPU **worse** (39.841 vs 35.117); (4) the matcher is itself a per-token CPU cost — their ROW 247: `prepare` 150.7 ms/token before indexing, 11.6 after, because `plan_hits=0`. **REJECT** `CachedAttention`, `physical.rs`, `render_cached_attention`, the bind.rs matcher, the `libm` dep. **KEEP** as separate cards: `prune_dead` (1.3), the consumer index (1.3), the paired Q4_K body (1.3 → 4.2), and their ROW 263 classifier-mislabel **finding** as the third witness on 3.2's row (their fix — a second marker string — is **not** ported). **KEEP as recorded negatives, never re-proposed:** float4 accumulation, explicit unroll, width-64 dispatch, row-batch 8, tiled-gemm at decode, nsg=2 (now a **fourth**-time negative across both lanes, R4 + R12 ROW 267).
- **commands** the `git show`/`git diff` above; extract ROWs 234-267 verbatim into `docs/bench-campaigns/2026-09-03-gpu-one-risc/parallel-rows.md` on the Phase-0 branch.
- **expect** N = **34** rows extracted, each tagged keep / renumber / supersede; six rulings each with its constraint named and its premise confirmed by a `git show` line; `git log --oneline main..perf/cached-attention-streaming | wc -l == 42` accounted member-by-member. **N==0 is RED.**
- **predict** none — a recorded decision boundary.
- **kill (pre-registered re-open condition)** if 9.3 measures that ≤23 ops/layer is unreachable through affine write placement, this adjudication re-opens **with that number attached**. If `git show` contradicts a premise (e.g. `prune_dead` is entangled with the macro-op), that item's ruling is re-derived from the diff and recorded as a correction.
- **memory gate** MG-1 — no build. **rollback** docs revert; a ruling is reversed only by new evidence in its own row. **blast** docs only.
- **observe** the 42-commit accounting; the row count.
- **reprove** `cd /Users/brianbruggeman/repos/slot-0/proxima && git log --oneline main..perf/cached-attention-streaming | wc -l`
- **log-row titles** `the fifth bound kind the binding does not admit, rejected on four grounds, four pieces kept` · `the parallel lane's measured negatives, renumbered onto main`

### 1.2 — Recover the mask-fma Q4_K body onto today's main `[S2 2.1, B3 P1.2]`
- **tier** worker · **depends_on** 1.1, 0.7 · **worktree** `proxima-wt-risc01` / `risc/1-q4k-mask-fma`, base `risc/0-measure-truth`
- **opens** `omega/src/msl.rs:190-311` (`Q4K_UNPACK_MSL`), `:2452-2530` (`push_packed_row_blocked_body`, `lanes_per_block` `:2516`, loop step `:2527`); incumbent `ggml-metal.metal:5086-5193` — mask **without** shift, branch-free `kmask1/2/3` `:5147-5150`, 1/16 and 1/256 folded into the scale at combine `:5171-5175` (R8); 0.7's `all-tracked.patch` and `all-head.txt`.
- **commands** `git apply --3way $R/all-tracked.patch` restricted to the `msl.rs` Q4_K hunks, then **strip to the one body** — `proxima-wt-all` carries seven features (R7). The 9 intervening commits include `spec.rs +8735/−2836`; **conflicts are the work, not a blocker**, which is why this is a worker card. Then G9's gate and the ORACLE cell, both welded.
- **expect** gate PASS with `ran_count`/`passed_count` recorded (either ==0 is RED); the oracle asserts `2651`/`"known"`. `omega/tests/q4k_real_checkpoint_parity.rs` runs ≥1 case with the body selected. **N==0 is RED.**
- **predict (nano → micro)** `q4k_matvec_probe` at the `ffn_up` shape shows **≥20% lower ns/op** than main's body (R7's −36% on ffn families is MEMORY and the floor is conservative; the row says the anchor is MEMORY).
- **kill** oracle drift (G10); or parity max-abs error vs `cpu::evaluate` on real `blk.0.attn_q.weight` > **1e-4** (§14 — the body does not land at any speed); or `ran_count` below 0.5's recorded value.
- **memory gate** MG-1 for the gate; MG-3 for the oracle. A kernel-body change allocates nothing; any device increase is a NEGATIVE.
- **rollback** `worktree remove --force` + `branch -D`; nothing depends on this branch until 4.1.
- **blast** `omega/src/msl.rs` Q4_K body only.
- **observe** `ran_count`, `passed_count`, oracle token/text, `q4k_macs` (`bind.rs:2856-2860`).
- **reprove** the gate + oracle commands, welded.
- **log-row title** `the measured -17.2% mask-fma Q4_K body, rebased nine commits forward and committed`

### 1.3 — Cherry-pick the paired body, `prune_dead`, and the consumer index — without their row numbers `[S2 0.8+0.9+2.2, B3 P1.3, crit RB-1]`
- **tier** worker · **depends_on** 1.1 · **worktree** `proxima-wt-risc02` / `risc/1-recovered-picks`, base `risc/0-measure-truth`
- **opens** `git log --oneline --reverse main..perf/q4k-independent-accumulators`; `git show 216d925` ("drop dead resolved nodes before GPU dispatch"); R12 ROW 257 (paired body: family 47.8 → 33.9 ms, −29%, parity 3.1e-6), ROW 248 (consumer index behind ROW 247's `prepare` 150.7 → 11.6); `bind.rs:200-215`.
- **commands** three picks, **each its own commit, each green on its own**, each through G11's `--no-commit` + `git checkout HEAD -- proxima-tensor/docs/discipline.md` so **no literal ROW 234-267 lands** [crit RB-1]; assert `git show --stat HEAD | grep -c discipline.md == 0` after each. Do **not** bring `physical.rs`, `CachedAttention`, the matcher, or `libm` (1.1). Then both crate gates and the welded BENCH + ORACLE cells.
- **expect** exactly **3** commits, zero `discipline.md` hunks; `op_count` **strictly less than 1196** — this card publishes **`OPS_AFTER_PRUNE`** as a MEASURED number that every downstream card quotes, never the expression "1196 minus the pruned count" [crit O-5]; `prepare_ms` **below** R13's 1.97; `generated_text` unchanged; ≥2 new `prune_dead` tests. **`op_count == 1196` is RED for this card specifically.** **N==0 is RED.**
- **predict (milli → bench)** `encode_dispatch_calls` falls from 1196 by exactly the pruned-node count and every removed node is nameable; `gpu_exec_ms` moves **< 0.2 ms** (R13: the 39 degenerate constant/iota control ops total 0.169 ms); `prepare_ms` **≤ 1.4** (−30%, R12 ROW 248 anchored).
- **kill** `gpu_exec_ms` moves > 0.2 ms ⇒ live nodes were removed — a correctness event; diff `generated_text` and stop. `prepare_ms` unchanged ⇒ the index does not bind here; record the negative and drop that commit. Oracle drift ⇒ revert (§14).
- **memory gate** MG-3; `prune_dead` **removes** buffers, so `device_allocated_bytes` must be `<=` 0.5's steady value — an increase is a NEGATIVE and rolls back regardless of the dispatch win.
- **rollback** per-commit `git revert`; the three are independent by construction.
- **blast** `proxima-tensor/src/bind.rs` — **cross-backend**, so the CPU oracle runs on this branch, not only Metal.
- **observe** `OPS_AFTER_PRUNE` (**NEW**, printed via `op_count` at `generate.rs:107`), `encode_dispatch_calls`, `prepare_ms`, `device_allocated_bytes`, `generated_text`.
- **reprove** the welded BENCH + ORACLE cells.
- **log-row title** `three generic pieces recovered from the parallel branch, without its row numbers; the post-prune op count becomes the number every later card quotes`

---

## PHASE 2 — one bound plan (brief item 1), closed EARLY
*Conflict 8: this lands in Phase 2, not behind the emitter reorganisation. Every claim about "the plan" downstream is unfalsifiable while D3 stands, and this is the widest-signature change in the plan — cheapest when nothing else is in flight. Worktree `proxima-wt-risc03` / `risc/2-one-bound-plan`, base `risc/1-recovered-picks`.*

### 2.1 — The plan-identity test: a per-op fingerprint vector through `pub` accessors, pre-registered RED `[B3 P2.1, S2 0.14, crit RS-5, MS-6, e]`
- **tier** worker · **depends_on** 0.9, 1.3
- **opens** `omega/src/metal.rs:1003` (`bind(...)`), **`:1013`** (`correct_packed_matmul_layouts(&mut resolved, …)`), `:1005-1012` (the comment: `layout_of` "assumes every operand is stored row-major in its DECLARED axis order … never true for a packed Q4_K/Q5_K/Q6_K weight"); **`:859-864` — `struct Prepared` and its `resolved` are PRIVATE, verified this session**; `:313-325` (`Plan`'s fields private); `proxima-tensor/src/cpu.rs:358` (`bind::bind`, no rewrite), `:280` (`Prepared<'block>`); `bind.rs:1718-1722` (`bind` has **no** backend parameter), `:200-215`, `:95-98`, `:1594-1606`; `msl.rs:4656` (`emit_is_deterministic_byte_equal`).
- **commands** add `pub fn fingerprint_ops(plan: &[BoundOp]) -> Vec<u64>` in `proxima-tensor` — **one FNV-1a-64 per op, not one per plan** [crit e], over a canonical serialization (node id, dtype, extents, kind discriminant; for `Elementwise` every `ComposedBody` step's `ScalarOp` + operand indices; for `Reduce` `element_body`, `reduce_op`, `init`, `keep`, `output_axes`, `out_layout.base`/`strides`, `out_scatter` fields; per operand `(source, Layout, Option<Lookup>)`; for `Constant` `value.to_bits()`). Add the **`pub` surface the capture needs** [crit RS-5]: `#[cfg(feature="instrument")] pub fn bound_ops(&self) -> &[BoundOp]` on omega's `Plan` and the CPU-side equivalent. The test builds the real cached-forward program with real symbols and real blocks, captures each backend's ops **after that backend's own rewrites** (Metal after `:1013`, CPU after `cpu.rs:358`), and compares **vector to vector**, reporting the differing **node ids**.
- **expect** the test **FAILS**, naming every differing op's node id and both `Layout{base,strides}` values. **A PASS is RED** — it would mean the fingerprint does not cover `Layout.strides`. Differing ops expected = **exactly the packed matmul weight operands' consumers**; `N differing == 0` is RED. Lands `#[ignore]`d with an `// EXPECTED RED, see R16 / metal.rs:1013` marker plus a companion always-green test asserting the differing **node set == `packed_operands.keys()`**. **N==0 (test not run) is RED** — the test is gated on `metal` **and** `cpu` together, the likeliest trap.
- **predict (nano → micro)** `fingerprint_ops` over `OPS_AFTER_PRUNE` ops costs **< 100 µs**, i.e. < 0.01% of R13's `prepare_ms` 1.97.
- **kill** the differing set includes a node **not** in `packed_operands` ⇒ there is a **third** rewrite; that finding outranks every performance card and 2.2 is re-scoped to name it. A card that makes the test green by loosening the field list has inverted it; **the field list is fixed here and is not negotiable downstream.**
- **memory gate** MG-1; an allocation-counter assertion over 1000 calls asserts zero heap in `fingerprint_ops`.
- **rollback** `git revert`; one pure function, two `#[cfg]` accessors, one test file. **blast** `proxima-tensor/src/bind.rs` +1 function; `omega/src/metal.rs` +1 accessor; `cpu.rs` +1 accessor. Zero hot path.
- **observe** the two vectors and the differing node set (**NEW**, the test's own output).
- **reprove** `cd $WT && CARGO_TARGET_DIR=$TD bash scripts/gpu-measure-lock.sh $LOCK --wait 1800 -- cargo nextest run -p omega --features metal,cpu,instrument --run-ignored all -E 'test(plan_fingerprint)'`
- **log-row title** `brief item 1 is false on main today: the Metal driver rewrites the bound plan after bind, and here are the node ids it rewrites`

### 2.2 — `bind` owns the packed layout; the post-bind rewrite is deleted `[B3 P2.2, S2 7.6, R16, R18]`
- **tier** worker · **depends_on** 2.1
- **opens** `omega/src/metal.rs:375` (`packed_operands_of`, verified this session — it lives in omega while `QuantizedBlock` lives in `proxima-tensor/src/cpu.rs:3084-3110`), `:412` (its call site), `:1003`, `:1013`, `:191` (the import); `proxima-tensor/src/bind.rs:1618-1707` (the rewrite and its doc), `:1718` (`pub fn bind`), `cpu.rs:358`.
- **commands** three moves: (1) `packed_operands_of` **descends** into proxima-tensor beside `QuantizedBlock` — a `&[NodeId] × &[QuantizedBlock] → PackedOperands` function moving to the crate that owns both argument types (§1 relocation, nothing minted) [R18]; (2) `bind` gains the packed set and `correct_packed_matmul_layouts` becomes its private final step, the public function **removed**, not deprecated (§15); (3) `metal.rs:1013` is **deleted** and `cpu.rs:358` passes its own set. Then G9 + `cargo test -p proxima-tensor --all-features` + the ORACLE cell, all welded.
- **expect** N1 2.1 turns **GREEN** and its `#[ignore]`/`EXPECTED RED` marker is removed in this commit. N2 `grep -rn correct_packed_matmul_layouts --include='*.rs' .` returns **exactly 1** hit (the private definition inside `bind`) — more than 1 is RED, a driver still rewrites. N3 every Q4_K/Q5_K/Q6_K parity suite green (`q4k_real_checkpoint_parity`, `q4k_matmul_layout`, `q4k_unpack`, `metal_parity`, `backend_parity`, `metal_real_forward`) with counts recorded. N4 `generated_text` and `2651`/`"known"` unchanged. **N==0 is RED.**
- **predict (nano → micro)** emitted MSL is **byte-identical** (`emit_is_deterministic_byte_equal`, `msl.rs:4656`) and `q4k_matvec_probe` output is byte-identical — the corrected layout **is** the layout Metal already executed; only *where* the correction happens changes.
- **kill** any byte drift, any parity regression, or a still-unequal fingerprint ⇒ a **third** rewrite; name it from the node set and stop. If the CPU oracle drifts, the CPU path **does** read packed bytes through `layout_of`, contradicting R16 — the finding supersedes the design and the correction must live behind a per-backend physical-layout query instead.
- **memory gate** MG-3 (the oracle run); no allocation change.
- **rollback** `git revert` the deletion commit (restores `:1013`) then the bind commit; 2.1 returns to its **documented** RED — the intended signal, not a break.
- **blast** **cross-crate and cross-backend**: `bind`'s signature, every caller in proxima-tensor / omega / proxima-model-interop, `metal.rs:1003-1013`. Deliberately early, when nothing else is in flight.
- **observe** the fingerprint vectors, the grep count, `ran_count`, the six parity suites, the oracle.
- **reprove** the four commands, welded.
- **log-row title** `bind owns the packed layout; no backend rewrites the plan after bind, and metal.rs:1013 is gone`

### 2.3 — Re-anchor: same plan, same numbers `[B3 P2.3]`
- **tier** hands · **depends_on** 2.2, 0.5
- **commands** the 0.5 three-arm sweep on this branch, interleaved against the 0.5 anchor binary.
- **expect** `step_wall_ms`, `gpu_exec_ms`, `op_count` (== `OPS_AFTER_PRUNE`) and `generated_text` all **within 0.5's CoV band**. A no-op cell by construction; its value is proving a three-crate signature change moved zero milliseconds. **N==0 is RED.**
- **predict (bench, anchor)** predicts nothing further; it is the anchor for Phase 3 onward.
- **kill** any metric outside the band ⇒ the move was not behaviour-preserving; bisect 2.2's three sub-moves.
- **memory gate** MG-3. **rollback** see 2.2. **blast** none (measurement).
- **observe** every 0.5 counter.
- **reprove** the 0.5 sweep on this branch.
- **log-row title** `one bound plan lands with zero measured cost`

---

## PHASE 3 — the route becomes a value
*Worktree `proxima-wt-risc04` / `risc/3-route-value`, base `risc/2-one-bound-plan`. R13's 225/385/547 split is `classify_kind`'s substring output and R12 ROW 263 measured that instrument relabelling 9/601 → 225/385 when a body changed — so the census lands **before** any body swap.*

### 3.1 — `Route`, unit-only, with `fn slot(&self) -> usize`, decided by `emit` itself `[S2 1.1, B3 P3.1, crit RS-2, f, OB-2]`
- **tier** worker · **depends_on** 2.3
- **opens, in this order** (1) `omega/src/msl.rs:673-697` `emit`'s 4-kind + `Keep` match; (2) `:731-775` `kernel_cache_key` — three route characters, `'G'` if `tiled_gemm_block(..).is_some()` else `'B'` if `packed_row_block(..).is_some()` else `'S'`, with `:751-758` recording the ordering as load-bearing; (3) `:3164-3187` `push_cooperative_reduce_body`'s **third** re-derivation of the same gates; (4) `:797`, `:1517-1560`, `:824`, `:1235`, `:1450`, `:1487`; (5) `omega/src/metal.rs:777-826` and `:835-854` and `:709-710`; (6) `proxima-tensor/src/instrument.rs:809-828`/`:842`/`:848-864` — the `(NodeId, reason)` **shape** to mirror, **explicitly not its `Mutex<BTreeMap>`**; (7) `proxima-telemetry/src/metric/counter.rs:12-17` — **`Counter` holds an `AtomicU64` and is not `Copy`**, verified.
- **the design, both questions answered** *Pipe question:* `Route` is a decision value computed once per `BoundOp` and consumed once by `emit` — no stages, no backpressure; `backend.rs:1-52` already adjudicated this boundary not-a-pipe. *Relocation question, call site both ways:* Way A `let (route, reason) = route::of(bound, packed); match route { … }` makes `assert_eq!(route::of(bound).0, recorded(node))` and `assert_eq!(Σ ROUTE_DISPATCHES, ENCODE_DISPATCH_CALLS)` **possible**; Way B (today) is four hand-ordered `if let` gates in three places plus a substring recovery MEASURED to mislabel. New caller capability ⇒ **the type is earned.**
- **the enum, and why `Declined` carries no data** [conflict 9, crit RS-2/f/OB-2]: `#[derive(Clone,Copy,PartialEq,Eq)] #[repr(u8)] pub enum Route { Elementwise, Scan, Iota, Constant, ReduceSerial, ReduceCooperative, ReduceRowBlockedPacked, ReduceTiledGemm, Declined }` — **unit-only**, so no `route as usize` (E0605 on a data-carrying variant) ever appears; the mapping is an explicit `pub fn slot(&self) -> usize { match self { Elementwise => 0, … Declined => 8 } }`. The **reason** travels beside the route as a separate `DeclineReason` (reusing `diagnose_packed_row_block`'s existing reason set), never inside the variant. `route::of` returns `(Route, DeclineReason)` and is **the function `emit` branches on**, so census and emission cannot disagree.
- **commands** `emit`, `kernel_cache_key`, `kernel_dispatch_shape`/`grid_threads` and `push_cooperative_reduce_body` all consume `route::of`. Then:
```
cd $WT && CARGO_TARGET_DIR=$TD bash scripts/gpu-measure-lock.sh $LOCK --wait 1800 -- cargo build -p omega --no-default-features --features alloc
cd $WT && CARGO_TARGET_DIR=$TD bash scripts/gpu-measure-lock.sh $LOCK --wait 5400 -- bash scripts/omega-gate.sh
cd $WT && CARGO_TARGET_DIR=$TD bash scripts/gpu-measure-lock.sh $LOCK --wait 1800 -- cargo nextest run -p omega --all-features -E 'test(route) + test(emit_is_deterministic_byte_equal)'
```
- **expect** N1 a table-driven test returning each of the **8 dispatching variants** for a hand-built `BoundOp` at that shape — a variant with no case is RED (an unroutable variant is an invented variant). N2 **`kernel_cache_key` byte-stability**: for every op the key through `route::of` is byte-identical to main's three-character logic, G-before-B preserved. N3 **golden emitted source**: the emitted MSL for every op in the real program hashes identically before and after, extending `emit_is_deterministic_byte_equal` to a cross-commit golden. N4 the alloc-tier build compiles `route.rs` and **states which modules it built** (§3's N==0 warning). **N==0 is RED.**
- **predict (nano → micro)** behaviour-neutral: golden hashes match for all `OPS_AFTER_PRUNE` ops; `q4k_matvec_probe` ns/op unchanged within CoV, because no kernel text changed.
- **kill** one golden hash drifts or one `kernel_cache_key` differs ⇒ the refactor changed routing, which is a different card; **do not proceed to 3.2.**
- **memory gate** MG-1. **rollback** `git revert`; `route::of` is **not** feature-gated (a gated route means two routers). **blast** new `omega/src/route.rs`; `msl.rs` three decision sites collapse to one. `wgsl.rs`/`cuda.rs` untouched here (3.3 gives them the route).
- **observe** the 8-case table; the golden hashes; `kernel_cache_key` equality.
- **reprove** the three commands above.
- **log-row title** `the route stops being a substring of the kernel it selected: one unit-only enum with an explicit slot map, and emitted MSL byte-identical`

### 3.2 — Delete `classify_kind`; the census, lock-free, cost-bounded `[S2 1.1 census half, B3 P3.2, crit RS-1, OB-1, OB-2]`
- **tier** worker · **depends_on** 3.1
- **opens** `metal.rs:709-710`, `:785-826`, `:835-854`, **`:2243`** (`counter!(ENCODE_DISPATCH_CALLS, 1)` inside `#[cfg(feature="instrument")]`, with `ENCODE_DISPATCH_TICKS` `:2244-2246`), `:1484` (`Counter::new`), `:578-585` (`plan_named`), `:1577` (`snapshot_and_reset`), `instrument.rs:842-864` (**the pattern NOT to copy**).
- **the census, hot and cold** *Cold, once per plan* at `plan_named`: `Plan.routes: Vec<(Route, DeclineReason)>` filled at plan build, single-owner, no synchronisation — **the census of record**, keyed `(NodeId, Route, reason)`. *Hot, per dispatch* at `:2243`: `ROUTE_DISPATCHES[route.slot()].add(1)` on a **`[Counter; 8]`** written as **eight explicit `Counter::new("omega.metal.route_dispatches.<variant>")` const initializers** (`Counter` is non-`Copy`, verified) with eight matching `snapshot_and_reset` fields [crit OB-1]. **`Declined` gets NO hot slot** [conflict 9, crit OB-2]: a declined op never dispatches, so its counter would be structurally 0 and would make the sum identity vacuous — declines are observable only in the cold table, and 3.4's gate asserts **both** the hot sum identity and the cold decline count. `classify_kind` and `diagnose_kind` are **deleted**, not kept alongside; call sites read `plan.routes[position]`.
- **the cost bound, MEASURED not assumed** R13: `encode_dispatch_ms = 0.47` over 1196 dispatches = **393 ns/dispatch** (DERIVED). Budget **≤ 5% = 19.6 ns/dispatch = 0.0235 ms/token**. The card runs census-ON vs census-OFF interleaved and asserts the `encode_dispatch_ms`, `op_setup_ms` and `prepare_ms` deltas are each ≤ 0.0235. The guard is **`ENCODE_DISPATCH_TICKS`**, not `gpu_exec_ms` — `gpu_exec` is the device window (`metal.rs:546-554`) and cannot see a CPU-side cost [crit OB-1].
- **commands** the two greps below plus G9 and the welded BENCH cell in both arms, interleaved:
```
cd $WT && grep -rn "classify_kind\|diagnose_kind" --include='*.rs' .        # must be 0
cd $WT && grep -rn "Mutex" --include='*.rs' omega/src                       # must be 0
```
- **expect** both greps **0 hits** (any hit is RED; §21). `Σ ROUTE_DISPATCHES == ENCODE_DISPATCH_CALLS` exactly, every step. `ENCODE_DISPATCH_CALLS / F == op_count == OPS_AFTER_PRUNE`. The per-route counts against R13's buckets (225/385/547/37/2): **a difference is not automatically RED, it is the finding** — R12 ROW 263 proves the substring classifier mislabels; any difference is reported as "the census disagrees with `classify_kind` at N ops, here are their nodes", and **every bucket number in R13 is then restated against routes**. **N==0 is RED.**
- **predict (milli → bench)** `encode_dispatch_ms` delta ON−OFF ≤ **0.0235 ms**; `step_wall_ms` ON within R13's 0.5% CoV of OFF.
- **kill** the budget is exceeded ⇒ collapse to the cold table only and re-measure; still over ⇒ the census does not go on the dispatch path at all and the row says so. Sum ≠ `ENCODE_DISPATCH_CALLS` ⇒ a path bypassed `route::of`; **do not proceed to Phase 4.**
- **memory gate** MG-3. 8 `Counter`s = 448 B static; the cold vector is `2 B × op_count` ≈ 2.4 KB/plan, bounded by `plan_cache_len == 1`; anything that makes it grow per token is a KILL.
- **rollback** `git revert`; the counters are behind `instrument`, the route is not.
- **blast** `omega/src/metal.rs` (`classify_kind` deleted incl. `:709-710`; +8 counters + one line at `:2243`; the cold table in `plan_named`), `omega/examples/real_forward_packed_probe.rs`, `generate.rs` (`op_profile_bucket` sources `kind` from `Route`).
- **observe** `ROUTE_DISPATCHES[8]` (**NEW**, `metal.rs:2243`), the cold table (**NEW**, `plan_named` `:578-585`), `ENCODE_DISPATCH_CALLS`, `ENCODE_DISPATCH_TICKS`, `op_setup_ms`, `prepare_ms`.
- **reprove** the ON/OFF interleaved pair + both greps.
- **log-row title** `the route census is a plan property: zero locks, one atomic add on a line that already had one, and a measured 5% budget`

### 3.3 — Route the other two backends; the coverage matrix `[B3 P3.3, S2 7.2 partial]`
- **tier** worker · **depends_on** 3.2
- **opens** `omega/src/wgsl.rs:105`, `:364`; `omega/src/cuda.rs:66`, `:146-183` (`emit_cuda` **rejects Iota and Constant** via `CudaUnsupportedOpKind`), `:241`; `omega/src/error.rs:53`.
- **commands** `route::of` moves to the shared surface; `wgsl` and `cuda` each return a `Route` for every `BoundOpKind`. **CUDA's rejection of Iota/Constant becomes `Route::Declined(BackendLacksKind)` — a recorded, censused coverage hole, not a silent `Err`.** This is the honest form of "every backend covers every kind": the plan **names** the holes; 8.2 closes the CUDA ones.
- **expect** a coverage-matrix test, 3 backends × 4 `BoundOpKind` (+ `Keep::Scan`) = **15 cells; N==0 is RED**. Pre-registered: Metal 4/4; WGSL 4/4 with no tiled-GEMM and no packed-row-block route; CUDA **2/4** with two named declines. The test asserts the matrix **equals** the pre-registration — a cell changing without a card is RED.
- **predict (nano → micro)** no device effect; WGSL and CUDA have no decode driver.
- **kill** the matrix disagrees with the pre-registration ⇒ R5's coverage claim is stale; re-read before the matrix is written.
- **memory gate** MG-1. **rollback** `git revert`. **blast** `wgsl.rs`, `cuda.rs` emit entry points.
- **observe** the 15-cell matrix; per-backend decline reasons.
- **reprove** `cd $WT && CARGO_TARGET_DIR=$TD bash scripts/gpu-measure-lock.sh $LOCK --wait 1800 -- cargo nextest run -p omega --all-features -E 'test(backend_route_coverage)'`
- **log-row title** `backend route coverage, censused: CUDA is two of four kinds and now says so with a reason`

### 3.4 — The census cell, the census gate, the fingerprint gate, and the elementwise concentration `[S2 1.2+3.2+7.7, B3 P3.4]`
- **tier** worker · **depends_on** 3.3, 0.8, 2.2
- **opens** `scripts/omega-gate.sh`; `generate.rs:1721-1750`; R13's elementwise bucket (**547 ops / 7.350 ms**, 13,437 ns/op).
- **commands** (a) run the welded BENCH cell with the route dump; (b) add gate steps under a **named** feature set (never `--all-features`, which also enables `cuda`/`wgpu-backend`/`vulkan`/`npu`/`ane`): `Σ ROUTE_DISPATCHES == ENCODE_DISPATCH_CALLS`; cold-table decline count == the census's; **2.1's fingerprint equality**; `grep -c correct_packed_matmul_layouts == 1`; (c) extend the **cold** table with `(extents, operand_count)` per elementwise node — cold, so 3.2's budget is not re-spent — and rank by tick share.
- **expect** every op carries a `(NodeId, Route, DeclineReason)` triple; sum == `op_count`; the gate is RED on any mismatch and prints both sides; ≥1 cold row per elementwise node with the count equal to the `Elementwise` share of `ENCODE_DISPATCH_CALLS`. **N==0 is RED; a mismatch is RED.**
- **predict (nano → micro)** the route histogram **matches R13's substring buckets exactly** (225/385/547/37/2 scaled to `OPS_AFTER_PRUNE`); a mismatch means `classify_kind` mislabelled on main too and every R13 bucket is restated. Separately: the **top-5 elementwise nodes carry ≥50%** of the 7.350 ms bucket — the bucket is concentrated, not uniform.
- **kill** any op landing on `ReduceSerial` that R13 attributes to a Q4_K/Q5_K/Q6_K family — a quantized matvec on the serial path is a route bug worth more than any kernel tweak. If the elementwise bucket is uniform across >200 nodes, no single-node lever exists and the only remaining lever is *fewer nodes* (Phase 9); this card closes with that pointer. Standing reason not to reach for fusion: `grep -rln fuse ggml/src` is **empty** at `b25346221` (R8).
- **memory gate** MG-3; the cold table grows by ~3 fields × plan length ≈ 15 KB, bounded by `plan_cache_len == 1`.
- **rollback** `git revert` the gate steps and the census extension. **blast** `scripts/omega-gate.sh`, the cold recorder.
- **observe** the route histogram; the gate output; per-node elementwise ticks.
- **reprove** the welded BENCH cell + `bash scripts/gpu-measure-lock.sh $LOCK --wait 5400 -- bash scripts/omega-gate.sh`
- **log-row titles** `1196 ops censused by route with a reason, not by grepping MSL` · `one bound plan and one route are gates, not claims`

---

## PHASE 4 — the Q4_K body (R13's largest measured mass: 225 ops / 44.450 ms)
*Worktree `proxima-wt-risc05` / `risc/4-q4k-bakeoff`, base `risc/3-route-value`, with `risc/1-q4k-mask-fma` and `risc/1-recovered-picks` merged in as the two bodies.*

### 4.1 — The build-time body selector, not two cargo features `[B3 P4.1, R18, conflict 4]`
- **tier** worker · **depends_on** 3.4, 1.2, 1.3
- **opens** `scripts/omega-gate.sh` step **[2/6]** `cargo build -p omega --all-targets --all-features` and **[3/6]** `nextest --all-features` (verified verbatim this session) — the constraint that forbids two exclusive features; `omega/omega-runtime.toml`; `omega/build.rs:16` `require_nonzero`, `:35`, `:43`, `:59`, `:79-85` `resolve_int` + `rerun-if-env-changed`, `:105` `emit_sizing_consts`; `msl.rs:190-311`, `:2452-2530`.
- **commands** add `[q4k] body = "pair_dot"` to `omega-runtime.toml` with both values documented at the key, and `emit_profile_cfg()` in `build.rs` emitting `cargo:rustc-check-cfg=cfg(omega_q4k_body, values("mask_fma","pair_dot"))` + `cargo:rustc-cfg=omega_q4k_body="…"` + `cargo:rerun-if-env-changed=OMEGA_Q4K_BODY`. `msl.rs` selects under `#[cfg(omega_q4k_body="…")]`. **No new cargo feature.** Run the full gate under **each** value, and once under a nonsense value.
- **expect** both valid values PASS the **full** gate including `--all-features` — **that is the card's whole point** — with `ran_count`/`passed_count` recorded per value (either ==0 is RED). The nonsense value **fails the build** with a named `build.rs` panic; **a successful build on a nonsense value is RED.** **N==0 is RED.**
- **predict (nano → micro)** `q4k_matvec_probe` under `pair_dot` shows lower ns/op than under `mask_fma` at the `attn_q` shape (R12 measured pair-dot there) and the reverse or a tie at `ffn_up` (R7 measured mask-fma there). If one body wins **both** micro shapes, the milli rung is expected to agree; a disagreement is the finding.
- **kill** `--all-features` fails under either value.
- **memory gate** MG-1. **rollback** revert the `build.rs` + toml hunks; both bodies remain in source, unselectable. **blast** `omega/build.rs`, `omega-runtime.toml`, `msl.rs` body selection. Nothing outside omega.
- **observe** two gate runs' counts; the negative-path panic message.
- **reprove** the three welded gate commands with `OMEGA_Q4K_BODY` set.
- **log-row title** `two Q4_K bodies, one build-time selector: exclusive without breaking --all-features`

### 4.2 — The bake-off: batched AND per-op arms, terminal tie-break, nothing deleted `[S2 2.3, B3 P4.2, crit HC-4, O-6, g, d]`
- **tier** hands · **depends_on** 4.1
- **opens** `msl.rs:1978-1982` — `Q4K_UNPACK_MSL`/`Q5K`/`Q6K` concatenated with **no delimiter**, which is why a "grep the Q4_K region" tie-break is undecidable [R16]; `bind.rs:3084` (per-op), `:3002` (decode cell); incumbent geometry `<4,2,32>` (`ggml-metal.m:3330`, `:3215-3220`).
- **commands** one sweep per round, three rounds, arms interleaved A B C A B C where C is the control body: for each arm, **both** the welded MILLI cell (per-family `gpu_ms`) **and** the welded BENCH cell (`gpu_exec_ms`, `gpu_device_ms`), plus both incumbent arms. `OMEGA_Q4K_BODY` selects the arm; features are identical across arms.
- **the tie-break, pre-registered, terminal, every rung decidable without reading emitted text** [crit d]
  1. **Parity gate.** Max-abs error vs `cpu::evaluate` on real `blk.0.attn_q.weight` > **1e-4** ⇒ out (§14; R12's recorded 3.1e-6 is the standard). All six parity suites green under each value.
  2. **Route-count pin.** The comparison is **void** unless both arms report the same `Route::ReduceRowBlockedPacked` census count. Unequal counts mean different op sets; a body change that moves the route is RED.
  3. **Primary metric — batched `gpu_exec_ms`**, mean over interleaved runs, winner iff the difference exceeds `max(CoV_A, CoV_B) × max(mean_A, mean_B)`. **This rung is batched precisely because `op_profile_family` exists only in per-op mode** (`generate.rs:184` from `Vec<OpGpuTiming>`, produced only by `execute_plan_op_timed`) and per-op mode inserts a commit/wait between every op, removing all inter-op overlap [crit HC-4, g].
  4. **Tie → summed per-op family `gpu_ms`** over the seven weight families, with 0.2's true bytes and the +7.3% inflation quoted.
  5. **Tie → lower max-abs parity error** on the same real tensor.
  6. **Tie → smaller total emitted MSL byte length** for the full decode program (deterministic by `emit_is_deterministic_byte_equal`, `msl.rs:4656`; needs no region parsing).
  7. **Terminal: `pair_dot` wins** — it is a commit on `4be2f3a` while mask-fma was an unrebased diff off `2b95210` (R7); lower landing risk breaks the last tie. **No rung can fail to decide.**
- **expect** 2 bodies × 3 rounds × 2 rungs + 2 incumbent arms = **≥16 cells**; every cell carries `generated_text` == R13's string and the parity number. **N==0 is RED.**
- **predict (milli → bench)** the winner drops `ReduceRowBlockedPacked` from **44.450** to **≤33.0 ms** (−29%, R12 ROW 257 applied to R13's mass, **DERIVED**) ⇒ `gpu_exec_ms` **56.93 → [44.0, 48.0]** and `step_wall_ms` **67.92 → [55.0, 59.0]** = **3.14–3.37x** against 0.5's chosen incumbent arm.
- **kill** neither body clears **−10%** on the packed-row-blocked bucket beyond both CoV bands ⇒ the −17.2%/−29% micro figures did not transfer; decompose — *inconsistency* if the bucket moved but wall did not (orchestration is absorbing it; Phase 6 owns that), *understanding-gap* if the bucket itself missed (the work item is "what else was in R12's feature-off control", which R12 records **already carried the paired body**). Stop the climb. Also KILL: parity > 1e-4, or `generated_text` drift.
- **memory gate** MG-3, both arms, every run; any device increase is a NEGATIVE that rolls back the winning arm.
- **rollback** flip `[q4k] body` — one toml line, no code fork. **blast** the packed-row-blocked route only: 225 of 1196 ops (R13).
- **observe** `ReduceRowBlockedPacked` census count and bucket ms, per-family ms with 0.2's bytes, `gpu_exec_ms`, `gpu_device_ms`, parity error, `generated_text`.
- **reprove** the interleaved sweep.
- **log-row title** `Q4_K bake-off: two independent re-derivations of ggml's mask-without-shift, decided on a batched arm before anything was deleted`

### 4.3 — Land one; the loser is a recorded negative, kept selectable `[S2 2.4, B3 P4.3, crit O-6]`
- **tier** judge · **depends_on** 4.2
- **commands** set `[q4k] body` to the winner named by the **mechanism**; write two rows.
- **expect** one row for the winner with its delta vs the 2.3 anchor; one row for the loser with its measured number and the rung that decided it. **A loser row with a blank number is RED.** The losing body's source is **kept and selectable** — a body that lost by 3% on one machine is the first thing to try on the next one, and nothing is deleted before a batched confirmation exists [crit O-6].
- **predict** none — a ruling over 4.2's cells.
- **kill** n/a. **memory gate** MG-1. **rollback** the toml key. **blast** one toml default.
- **observe** the two rows; `encode_dispatch_calls` unchanged at `OPS_AFTER_PRUNE` (a body change that moves the dispatch count changed the route).
- **reprove** 4.2's sweep at the landed value.
- **log-row title** `<winner> lands as the default Q4_K body; <loser> recorded at <delta> and still selectable`

---

## PHASE 5 — KV device residency, BEFORE bucketing
*Conflict 2: **residency first.** Worktree `proxima-wt-risc06` / `risc/5-kv-residency`, base `risc/4-q4k-bakeoff`. Residency changes no graph — the block handed to omega stays a `cached_len`-sized slice, so `element_count(shapes.of(node)) == block_element_count(block)` and the strict checks at `metal.rs:991-1000` / `cpu.rs:346-356` are **preserved, not relaxed**. Doing it the other way round is what produced the 4.1↔5.3 code-level cycle [crit RS-1, j].*

### 5.1 — Generalize the registered host span to N slots `[B3 P5.1, R18]`
- **tier** worker · **depends_on** 4.3
- **opens** `omega/src/metal.rs:1744-1752` (`CHECKPOINT_MAPPING`), `:1768-1773` (`register_checkpoint_mapping`), **`:1786-1815` `checkpoint_mapping_offset`, whose own doc says the scratch and KV-cache buffers "never live inside the checkpoint's own mmap, so they fall through unchanged"** — the invitation [R18]; `backend.rs:402-414`; `:1616` `is_page_aligned`; `:1914` `create_no_copy_buffer` (page-aligned pointer **and** length required).
- **commands** the single `Option` slot becomes a fixed-size array of `OMEGA_RESIDENT_SPAN_SLOTS`, that constant from a new `[spans] slots` key in `omega-runtime.toml` via `emit_sizing_consts` (§12 — no bare source const). `register_checkpoint_mapping` becomes `register_host_span(name, ptr, bytes) -> SpanSlot`; **the checkpoint is slot 0** and its call site is updated. `checkpoint_mapping_offset` becomes `host_span_offset`, scanning slots. `MAPPING_OFFSET_UPLOADS` gains a per-slot breakdown. Extend, do not add a peer (§1).
- **expect** `grep -rn 'register_checkpoint_mapping\|CHECKPOINT_MAPPING' --include='*.rs' .` = **0**; `register_host_span` present at exactly the loader call site (and the KV site after 5.2); a slot-exhaustion negative-path test returns a **named error**, never silently overwriting — **N==0 is RED**; gate PASS with `ran_count` ≥ prior.
- **predict (nano → micro)** `mapping_offset_uploads` on a decode step is **unchanged at 291** (R13), because slot 0 behaves exactly as the old singleton did.
- **kill** `mapping_offset_uploads != 291` before 5.2 lands ⇒ the generalisation changed the checkpoint path.
- **memory gate** MG-3; the span table is `slots × 24 B`, fixed at build time — the number goes on the row.
- **rollback** `git revert`; the singleton returns. **blast** `metal.rs` upload path, `backend.rs` re-export, the loader call site.
- **observe** the two greps; `mapping_offset_uploads` per slot; `nocopy_cache_len`.
- **reprove** the grep + the welded gate.
- **log-row title** `one registered-span primitive, N slots, the checkpoint is slot 0`

### 5.2 — A capacity-reserved, page-aligned KV arena; `found == expected` preserved `[B3 P5.2, S2 5.3 driver half, crit RS-1, MS-4, d]`
- **tier** worker · **depends_on** 5.1
- **opens** `proxima-model-interop/src/generate.rs:621-654` (`LayerCache`, `Vec::new()`, three `extend_from_slice` `:636-640`), `:1364-1389` (the per-layer KV push loop and its "full `cached_len`-sized array re-bound every step" comment), `:1559` (`cached_len += new_count`); `omega/src/metal.rs:1903` (the fresh-buffer-every-token path), `:991-1000` (`InputSizeMismatch`, **preserved untouched**), `cpu.rs:346-356` (its twin), `:350-362` `mark_resident` (classifies by **name**), `:1606` `page_size()`; `proxima-tensor/src/align.rs:42-46`, **`:56-58`** ("a real host page size the caller queried itself … never hard-coded here", verified) and `:69`, `:78-79` (`next_multiple_of(page_size).max(page_size)`); incumbent `llama-kv-cache-unified.cpp:74-118` (allocated once), `:749-788`.
- **the page-size source, which S2 had no answer for** [crit MS-4]: `omega::metal::page_size()` exists only inside the metal+macOS module, and `metal` is optional in `proxima-model-interop`. `proxima-tensor` already depends on `libc` under `std`, so the non-Metal source is **`libc::sysconf(_SC_PAGESIZE)`**, exposed as `proxima_tensor::align::host_page_size()`; the Metal value is used when the `metal` feature is on and a test asserts the two agree on this host. A test builds `-p proxima-model-interop --features std` (no metal) to prove the arena obtains a page size on a non-Metal build.
- **commands** `LayerCache`'s three `Vec<f32>` become three `AlignedBuffer`s sized `kv_capacity_tokens × row_elements` at load (`AlignedBuffer` gets its **first production caller**), each registered **once** through 5.1's `register_host_span`; `append` writes into the reserved region and **the base pointer never moves**; `named_blocks` still hands `&arena[..cached_len*row]`, so **the element count still equals the declared `Symbolic(1)` extent and neither validator is touched**. KV blocks resolve through `host_span_offset` — one no-copy buffer per arena, addressed by offset, exactly as the checkpoint is; `mark_resident` gains the KV names so `23e2e5e`'s non-resident routing no longer applies. `kv_capacity_tokens` is a **build-time key** (`proxima-model-interop-runtime.toml` + a new `build.rs`) with G8's byte assertion; **`context_length` never sizes an allocation.** Feature `metal-kv-resident`, default-off, forwarded.
- **expect** N1 a build with `PROXIMA_KV_CAPACITY_TOKENS=1000000` **fails** with the byte arithmetic in the panic — **a successful build is RED, that is the 34 GB trap re-opened.** N2 `nocopy_reuses >= 3 × layers × S` per run; `copying_uploads` for KV nodes == **0**; `BLOCK_COPIED_BYTES` for KV == 0 on every steady step (0.2's counter). N3 `kv_cache_upload_bytes` stops growing (+262,144 B/token → 0 after step 1). N4 `element_count(shapes.of(kv_node)) == block_element_count(block)` asserted on the real program **with no validator edit**. N5 `generated_text` and `2651`/`"known"` unchanged. N6 the no-metal build test; the page-size agreement test; pointer stability across appends; the qwen3.5 `DenseAttention`/`Ssm` cache states (`generate.rs:687`, `:738`) untouched with their `unreachable!` at `:1386-1388` intact. **N==0 is RED.**
- **predict (milli → bench)** `block_upload_ms` **2.00 → ≤0.5** (R13 records 0.4 on step 2, the weights-only step, so the "otherwise" 1.7–3.5 is the KV term) ⇒ `step_wall_ms` **[53.5, 57.5]**; `gpu_exec_ms` **unchanged** within CoV — this card moves no GPU work, and if `gpu_exec` moves, something else changed.
- **kill** `gpu_exec_ms` outside its CoV band; `device_allocated_bytes` peak above G8 clause 3; RSS slope > 1 MB/step; `generated_text` drift. `block_upload_ms` falls but wall does not, beyond both CoV bands ⇒ record it inside the noise band and stop (R2's "~3.3%" is MEMORY with no counterpart term in R13 and may not be claimed).
- **memory gate — the site of a prior failure.** MG-3 with clause 3 at `kv_capacity_tokens = 512` = **4,316,577,792 B** (DERIVED). The arena is allocated **up front** (512 × 262,144 = 134,217,728 B), so prefill RSS rises ~134 MB and **the MG-2/MG-3 RSS ceiling becomes 540 MB for this card and every later one**, stated here once with its arithmetic. The card **prints the computed byte total before allocating**. **Both slopes must go to 0 on the KV term** — a resident cache that still grows per token has not become resident, and that is RED **on the slope**, not on the timing.
- **rollback** feature default-off; `git revert` restores `Vec::new()`. **The build-time byte assertion is kept regardless** — it is a correctness guard and §15 forbids reverting a repair to restore a number.
- **blast** `generate.rs` KV path (`:621-654`, `:1364-1389`), a new `build.rs` on proxima-model-interop, `proxima-tensor/src/align.rs` (+`host_page_size`, first production caller), `metal.rs` `mark_resident` name set.
- **observe** `block_upload_ms`, `BLOCK_COPIED_BYTES`/`_NOCOPY_BOUND`/`_OFFSET_BOUND` (0.2), `nocopy_reuses`, `nocopy_cache_len`, `kv_cache_upload_bytes`, `device_allocated_bytes`, `phys_footprint_bytes`.
- **reprove** the trap-build command + the welded BENCH cell in both arms, interleaved 3×.
- **log-row title** `the KV cache stops round-tripping through the host: one registered span per layer at capacity, addressed by offset, and the validators never moved because the block is still cached_len elements`

---

## PHASE 6 — plan stability
*Worktree `proxima-wt-risc07` / `risc/6-plan-stable`, base `risc/5-kv-residency`.*

### 6.1 — Bucket the KV leaf extent; the tail mask spelled with the ops that exist `[S2 4.1, B3 P6.1, crit RS-1, RS-4, MS-1, MS-2, b, c, HC-1]`
- **tier** worker · **depends_on** 5.2, 0.8
- **opens** `proxima-tensor/src/spec.rs:6216-6245` (the three KV leaves at `[Symbolic(1), kv_heads, pairs|head_dim]`), `:823-845` `causal_mask` (two `Iota{Symbolic(0)}` + `Greater` + `scalar_constant(NEG_INFINITY)`), `:2604-2615` (`Select(is_future, neg_infinity, score_new_scaled)` — the consumption pattern to mirror **argument-for-argument**), `:2303-2319` (the doc naming the premise, corrected in this commit), `:2336-2865`, sole caller `:6282`; **`proxima-tensor/src/op.rs:60-78` — 17 bodies, no `Less`, no `GreaterEqual`** (verified this session); `generate.rs:1304-1309` `build_position_inputs` (keeps the TRUE `cached_len`; `start_position` is used only at `:813` for cos/sin, extent = symbol 0 — **bucketing symbol 1 does not touch RoPE or `is_future`**, R17), `:1313-1318` (`Vec::with_capacity(… + 3 + …)`), `:1332-1334`, `:1393` (`symbols`), `:1364-1389`; `metal.rs:1111-1118` `block_node_ids` (program order), `:984-990` `InputCountMismatch`.
- **the mask, with the arithmetic the closed set actually permits** [crit MS-1, RS-4]: there is **no `GreaterEqual`**, so the tail mask is
```
kv_valid_len : Op::Input, rank-0, f32      # the TRUE cached_len, ONE leaf for the whole program
cache_slot   : Op::Iota { extent: Extent::Symbolic(1) }      # over the BUCKETED cache axis
is_valid     : Elementwise{ Greater }( kv_valid_len -> "->t", cache_slot -> "t->t" )   # 1.0 where slot < cached_len
score_masked : Elementwise{ Select }( is_valid, score_cached_scaled, neg_infinity )
```
`kv_valid_len` and `cache_slot` depend only on the bucketed axis and the shared scalar, so they are built **once for the program**, not per layer; the per-layer cost is one `Select`, expected to fuse into `score_cached_scaled`'s existing `ComposedBody`. **Zero new `Op`, `BoundOpKind`, `ScalarOp` or `IndexMap` variants** — mechanically proved by 0.9's four exhaustive matches, which now include `ScalarOp` [crit SD-3].
- **the scalar leaf is wired, not assumed** [crit MS-1, MS-2]: an `Op::Input` **is** a block node, and both drivers hard-fail on count mismatch. In this same commit `build_position_inputs` gains a fifth output field owning the backing storage; `named_blocks.push(("kv_valid_len", …))` lands beside `"eps"`; and **the `+ 3` capacity literal at `:1316` becomes `+ 4` — exactly one, because the leaf is layer-invariant** (there are no 32 per-layer index leaves; there are no index leaves at all in this design [conflict 3]). A test asserts `named_blocks.len() == block_node_ids(program).len()` on the real program.
- **`found == expected` holds at every stage** [conflict 2, crit RS-1, d]: after bucketing, the leaf's symbol-1 extent **is** `bucket`, and `named_blocks` hands **`&arena[..bucket*row]`** — a `bucket`-sized slice of 5.2's `kv_capacity_tokens`-sized arena, with `bucket <= kv_capacity_tokens` asserted. So `element_count == block_element_count`, **neither validator is edited, and the arena exists one card earlier so there is no code-level cycle.** Rows in `[cached_len, bucket)` were never written (the arena is zero-initialised) and are masked to `-inf` by the `Select`, so they contribute `exp(-inf) = 0` exactly.
- **the rollback fork is decided here** [crit RB-3]: `KV_BUCKET_TOKENS` is a build-time const from `[kv] bucket_tokens` beside 5.2's `capacity_tokens`, so **feature-off rollback requires a rebuild** — stated on the card, not left to the row. 256 is not arbitrary: it ports the incumbent's own `n_kv` padding (R8/M6′); 6.3 sweeps it.
- **the baselines this card invalidates, re-captured here** [crit HC-1]: adding a leaf renumbers `NodeId`s, so 3.1's golden emitted source and 2.1's fingerprint vectors are **re-captured for the feature-ON tree in this same commit and both hashes recorded**; the feature-OFF hashes must equal Phase 3's recorded values, which is what makes "off is main" a hash rather than a claim.
- **commands** declare `kv-capacity-bucket` in `proxima-tensor/Cargo.toml`, forwarded from `omega` and `proxima-model-interop` (the verified passthrough pattern). Set `let bucket = (cached_len + new_count).div_ceil(KV_BUCKET_TOKENS) * KV_BUCKET_TOKENS; let symbols = [new_count as u64, bucket as u64];` at `generate.rs:1393`. Run the CPU mask test, then the ORACLE cell, then both gates, then the welded BENCH cell — every command welded per G3 **including the oracle** [crit l].
- **expect** N1 the CPU mask test: for `bucket ∈ {8, 32, 256}` × `cached_len` spanning a boundary in each = **9 cases**, attention output **bit-identical** to unbucketed (masked lanes contribute exactly 0 — this is 0-ULP, not approximate). N2 `2651`/`"known"` and `generated_text` unchanged. N3 `named_blocks.len() == block_node_ids(program).len()`; `InputCountMismatch` never fires. N4 `op_count` rises by **exactly 2**. N5 the route census shows **no new route**. N6 feature-OFF golden and fingerprint hashes equal Phase 3's. **N==0 is RED.**
- **predict (nano → micro)** `op_count == OPS_AFTER_PRUNE + 2`. **If it is +34, the `Select` did not fuse into the existing `ComposedBody` and that is the finding** — a bind-level composition question, not a graph question.
- **kill** any ULP difference on the CPU mask test, or token drift at any bucket size ⇒ the mask is wrong; revert immediately (§14). `op_count` delta ∉ {2, 34} ⇒ the graph changed in a way this card did not design.
- **memory gate** MG-3, RSS ceiling 540 MB (5.2). `kv_cache_upload_bytes` becomes **constant at `bucket × 262,144 / capacity`-shaped** rather than linear — flat is the point, and the row headlines the trade. Device cap uses `kv_capacity_tokens`, never `bucket`. **KILL.**
- **rollback** feature off **plus a rebuild** (the bucket is a build-time const); the two-arm `#[cfg]` pairing in 6.2 keeps both worlds green.
- **blast** `spec.rs` cached-layer builder and the three KV leaves **under the feature only**; `proxima-tensor-runtime.toml` +1 key; `generate.rs` (`build_position_inputs`, `named_blocks` incl. the `+3`→`+4` literal, `symbols`). **`append_qwen35_*` builders are NOT touched** and a test asserts it.
- **observe** `op_count`, the 9 parity cases, the oracle, the two hash sets, `kv_cache_upload_bytes`.
- **reprove** the mask test + the ORACLE cell + the BENCH cell, all welded.
- **log-row title** `the KV extent buckets to a capacity and the tail masks with Greater + Select, because there is no Less and no GreaterEqual`

### 6.2 — Invert the `plan_hits == 0` assertion, `#[cfg]`-paired, as a formula `[S2 4.1 assertion half, B3 P6.2, crit k]`
- **tier** worker · **depends_on** 6.1
- **opens** `bind.rs:2994-2998` (the doc stating the finding), `:3051`, `:3053-3055`, `:3057-3059`; `generate.rs:966` (the key), **`:973` `self.plans.clear()`** — the cache holds exactly one entry, so a hit requires the **immediately preceding** key.
- **commands** the existing pair becomes `#[cfg(not(feature="kv-capacity-bucket"))]`; a new pair is `#[cfg(feature="kv-capacity-bucket")]` computing `expected_misses = 1 + #{consecutive key changes}` over `forward_calls_taken` from the file's own `generated.0.len() + usize::from(generated.2)`, then `assert_eq!(plan_misses, expected_misses)`, `assert_eq!(plan_hits, F - expected_misses)`, `assert!(plan_hits > 0, "a bucketed extent that never hits is the null result, not a pass")`. **No literal token count anywhere** [round-2 j6].
- **expect** feature **off** → the original assertions hold unchanged (`default` behaviour untouched). Feature **on** → `plan_hits > 0` (`plan_hits == 0` is RED) and both counts equal the formula exactly, with the key sequence taken from 0.8's `symbols` dump, never assumed. **N==0 is RED.**
- **predict (milli → bench)** `prepare_ms` **1.97 → ≤0.2** on hit steps. **`op_setup_ms` is unchanged at 3.90** — a plan hit alone does not remove per-op buffer and uniform allocation (R11 M6′′); that is 6.5's target. The honest `step_wall` reduction from this card alone is **~1.7 ms**, and reporting the smaller number here is the point.
- **kill** feature-off behaviour changes at all; or `plan_hits` stays 0 with the feature on ⇒ the key is still moving; dump `symbols` and name the other varying symbol; or `plan_hits` rises while `prepare_ms` does not fall beyond CoV ⇒ the cost was never in `plan_named`; re-instrument before 6.5.
- **memory gate** MG-3, plus `plan_cache_len <= 1` every step (a bucketed key that fills the map is the `ff749a0` leak re-opened).
- **rollback** feature off. **blast** one test module, `#[cfg]`-paired so both worlds are asserted.
- **observe** `plan_hits`, `plan_misses`, `plan_cache_len`, `prepare_ms`, `op_setup_ms`, the `symbols` dump.
- **reprove** the welded BENCH cell in both arms.
- **log-row title** `the harness asserted plan_hits==0; now it asserts the formula, both ways, and the formula is consecutive-key because the cache clears on miss`

### 6.3 — The bucketing trade cell: orchestration saved against padding added `[B3 P6.3, conflict 2]`
- **tier** hands · **depends_on** 6.2
- **opens** R13's per-family table: `kv_cache.v` 1.559, `k_odd` 0.783, `k_even` 0.774 ms, **Σ 3.116 ms** at the R13 context.
- **commands** the 6.2 pair at `OMEGA_KV_BUCKET_TOKENS ∈ {8, 32, 64, 256}`, interleaved with the feature-off control, 3 runs each, both rungs.
- **expect** 5 configurations × 3 runs × 2 rungs = **30 cells; N==0 is RED**. Each cell carries the three `kv_cache.*` family times, `prepare_ms`, `op_setup_ms`, `plan_hits`, `step_wall_ms`, `gpu_exec_ms`.
- **predict (milli → bench)** the cached-range reduce now runs over `bucket` slots instead of `cached_len`, so the three `kv_cache.*` families inflate by `bucket / mean(cached_len)`. At the R13 context (mean cached_len ≈ 34.5, observed not assumed) bucket 256 gives ×7.4 ⇒ **+19.9 ms** (DERIVED) — a **net LOSS**; bucket 32/64 gives ×1.1–1.9 ⇒ **δ_b ∈ [0.3, 2.8] ms** against 1.7 ms saved. **The pre-registered claim is that the optimum sits near `bucket ≈ prompt_tokens + PROXIMA_MAX_TOKENS`** — which is exactly why the incumbent, running at real context lengths, pads to 256 and we cannot at this budget.
- **kill (the decisive one)** if `Δ(kv_cache.* gpu_ms) > Δ(prepare + op_setup ms)` at **every** bucket value, **bucketing is dead as a landing route**, the fork resolves to 6.4, and the loss is recorded without softening.
- **memory gate** MG-3 at each bucket; the device cap uses `kv_capacity_tokens`, not `bucket`.
- **rollback** feature off. **blast** none beyond 6.1/6.2.
- **observe** the three `kv_cache.*` families, `prepare_ms`, `op_setup_ms`, `plan_hits`, wall, gpu, per-bucket.
- **reprove** the sweep.
- **log-row title** `bucketing the KV extent: the orchestration saving against the padding cost, by bucket size, with the optimum measured`

### 6.4 — *(contingent on 6.3's kill)* The shape-invariant plan `[B3 P6.4, S2 abandoned-9, R18]`
- **tier** worker · **depends_on** 6.3 **and only if 6.3 killed bucketing at every bucket size**
- **opens** **`omega/src/msl.rs:2207-2218` — the `Uniforms` struct already carries `output_total`, `reduction_total`, `output_extents[]`, `redu
ction_extents[]`, `operand_base[]`, `operand_strides[][]`, `out_base`, `out_strides[]` — every one already a per-dispatch uniform** (R18); `metal.rs:2070` `upload_uniforms`, `:2069` `UNIFORM_BUFFER_REUSES` (a live reuse path at `:2075`); `msl.rs:1517-1560` `grid_threads`, `:731` `kernel_cache_key`, `:797` `kernel_dispatch_shape`; `bind.rs:200-215` (`BoundOp.extents: Vec<u64>` — the one baked thing).
- **the design, stated now so the fork is decidable, built only if reached** split `Plan` into a **shape-invariant** part (kernel source, pipelines, routes, retirement, packed operands, resident marking) and a **per-token** part (uniform bytes, grid dims, output buffer sizes). The evidence it is cheaper than it looks: `pipeline_lookup` is already **0.04 ms** (R13), so pipelines already survive `cached_len` changes and the MSL source is already extent-independent; what is rebuilt per token is bind + retirement + packed-operand resolution, none of which depends on the *value* of `cached_len`. KV-dependent intermediates are allocated at `kv_capacity_tokens` while **the grid is dispatched at the true extent** — so there is no padding cost at all and **no tail mask is needed**, which is the whole reason this fork can beat bucketing's best cell.
- **commands** the 6.2 command pair on branch `risc/6-plan-stable-invariant`, behind the same feature flag.
- **expect** `plan_hits == F − 1` (every step after prefill); `op_count` unchanged (the mask nodes 6.1 added are removed with the bucket); `generated_text` unchanged. **N==0 is RED.**
- **predict (milli → bench)** `prepare_ms` → ≤0.2 with **zero** GPU inflation, i.e. strictly better than bucketing's best cell; `gpu_exec_ms` unchanged within CoV.
- **kill** `gpu_exec_ms` rises at all — it must not, the grid is unchanged. If the split cannot be made without `BoundOp.extents` becoming symbolic (`Vec<Extent>`), stop: that touches `grid_threads`, `kernel_cache_key` and `kernel_dispatch_shape` (R11 M6′ calls it "a bigger change") and the card parks with the un-park condition "a real-context (≥1024-token) decode cell exists".
- **memory gate** MG-3; capacity-sized intermediates add `kv_capacity_tokens × kv_heads × head_dim × 4 B` per KV-dependent intermediate — **the count and the total are printed before running**, and the total is inside G8 clause 3 or the card does not run.
- **rollback** feature off. **blast** `Plan` construction in omega — the widest omega-internal change in the plan, which is why it is contingent and paid for only if the cheap route is proven dead.
- **observe** `plan_hits`, `prepare_ms`, `gpu_exec_ms`, `op_count`, `UNIFORM_BUFFER_REUSES`.
- **reprove** the 6.2 command pair.
- **log-row title** `the plan splits shape-invariant from per-token: the extents were already uniforms`

### 6.5 — The device output/uniform arena: whole-buffer sharing only `[S2 4.2, crit RS-3, RB-3, h, HC-2, conflict 5]`
- **tier** worker · **depends_on** 6.3 (or 6.4 if reached)
- **opens** `omega/src/metal.rs:2179-2252` `encode_op` — `:2193-2194` `kernel_cache_key` + `kernel_dispatch_shape` **per op per token even on a pipeline hit**, `:2210` `allocate_buffer`, `:2211` `upload_uniforms`, `:2249` `device_buffers.insert(bound.node, (output, 0))`; **`:2068-2078` `UNIFORM_BUFFER_REUSES` already exists with a live reuse path at `:2075` — read its current hit rate BEFORE assuming uniforms are the cost** [Q11]; **`:541-543` the retire loop `device_buffers.remove(retired)`** [crit RB-3]; **`:2355-2378` `finish`'s readback with the invariant spelled out at `:2360-2364`: "an output node's buffer is always freshly allocated by `encode_op` at offset 0 — only a weight INPUT can carry a nonzero offset … so reading from the buffer's own start is always correct here"** (verified verbatim this session) [crit RS-3]; **`:1128-1147` `bound_op_retirement`'s `if !outputs.contains(&node)`** (verified) — the constraint that pins outputs, not merely a test oracle [crit h]; `generate.rs:1393-1400` (the KV roots are program **outputs** today — 97 effective outputs) [crit HC-2].
- **the two soundness constraints, stated as constraints and not as tests**
  1. **Whole-buffer sharing only.** The arena is a free list of whole `MetalBuffer`s by size class, reused across positions whose live ranges (from `bound_op_retirement`) do not overlap. **It never sub-allocates within one buffer**, because the moment it does, `device_buffers.insert(node, (buffer, 0))` at `:2249` is a lie and `finish` reads another op's data from the buffer's start, against the invariant at `:2360-2364` [crit RS-3, h].
  2. **Outputs are pinned by construction.** `bound_op_retirement` excludes `effective_outputs`, so an output's buffer is never returned to the free list — that is what makes readback sound, and it is written on the card as the constraint the partition rests on.
  3. **The retire loop is NOT made a no-op** [crit RB-3]. `device_buffers.remove(retired)` still fires, so operand lookups keep walking the **live** set instead of ~1196 entries; the arena keeps the `MetalBuffer` alive for reuse behind the map. Strict O(1) per op in steady state: index the free list by size class, no allocation, no hashing on the dispatch path.
- **commands** hang a `BufferArena` off the cached `Plan`, built once when the plan is built; `encode_op` **binds** rather than allocates; one uniform buffer per position written in place (the `:2069-2078` mechanism already proves the shape is legal). Recover 0.7's `all-tracked.patch` `metal-buffer-pool` hunk as **reference only** and rewrite against the stable `Plan`. Feature `metal-plan-stable-buffers`, default-off, forwarded.
- **expect** N1 `OUTPUT_BUFFER_ALLOCATIONS == op_count` on the first step and **0 on every steady step**; a nonzero steady value is RED **and names the position that reallocated**. N2 `UNIFORM_BUFFER_REUSES` rises to `op_count` per steady step, reported **against its pre-card rate**. N3 `plan_hits` still satisfies 6.2's formula (this card is a no-op without plan stability). N4 `generated_text` unchanged. N5 ≥6 tests: pooled == unpooled on the real forward; **a pooled buffer is never rebound before its last consumer, asserted against `bound_op_retirement`'s own order on the real program**; **a program output's buffer is never in the free list** (the readback-soundness test); an extent change forces a documented realloc; uniform contents change per step while the buffer does not; the live buffer count is bounded in steady state. **N==0 is RED.**
- **predict (milli → bench)** `op_setup_ms` **3.90 → [0.4, 0.8]** (what remains is `kernel_cache_key` + `pipeline_for` + bind; `pipeline_lookup` is already 0.04) ⇒ `step_wall_ms` improves by **[3.1, 3.5] ms** from wherever 6.3 left it, landing **[48.3, 52.7] + δ_b**.
- **kill** `OUTPUT_BUFFER_ALLOCATIONS` nonzero in steady state ⇒ the plan is not stable; the work item goes **back to 6.3/6.4**, not forward. `op_setup_ms` falls but wall does not, beyond both CoV bands ⇒ the orchestration slice overlaps GPU execution — the same shape as R12's dispatch-count null; record it and **stop Phase 6** rather than continuing.
- **memory gate — the highest-risk memory card in the plan.** The arena holds intermediates alive for the plan's lifetime. **The card's FIRST action, before allocating, is to compute and print `Σ over resolved ops of product(extents) × dtype_bytes`** and compare it against G8's 41,943,040 B activation term; if the naive arena exceeds it, it **must** be liveness-partitioned (whole buffers only) and the row records the arena's peak bytes and reuse factor. **`ARENA_PEAK_BYTES` above the term rolls the card back regardless of the `op_setup` win.** MG-3. **KILL.**
- **re-derivation hook** [crit HC-2]: the KV roots are outputs today, so the partition is computed against a 97-output set. **9.2 shrinks that set to ~1**, which changes every liveness range — 9.2 therefore re-runs this card's peak assertion and its two soundness tests in its own commit, and that obligation is written on both cards.
- **rollback** feature default-off (`get_or_allocate` is `allocate_buffer` with the feature off); `git revert`.
- **blast** `omega/src/metal.rs` `encode_op` (two call sites), `Plan` (+1 field), the arena in `plan_named`. No IR, no graph, no emitter, no other backend.
- **observe** `OP_SETUP_CALLS`/`_TICKS`, `UNIFORM_BUFFER_REUSES` **before and after**, `OUTPUT_BUFFER_ALLOCATIONS` (**NEW**, `metal.rs:2210`), `ARENA_PEAK_BYTES` (**NEW**, the arena constructor), `device_allocated_bytes`, `phys_footprint_bytes`.
- **reprove** the welded BENCH cell ON/OFF interleaved 3×; the row's claim is N1 plus the arena peak.
- **log-row title** `1196 device buffers and 1196 uniform uploads per token become zero: whole buffers only, outputs pinned by retirement, and the retire loop still fires`

### 6.6 — Orchestration re-seal `[S2 3.3 analogue]`
- **tier** hands · **depends_on** 6.5
- **commands** `cd $WT && bash scripts/gpu-cell.sh $WT $TD std,metal,instrument 5` with the landed body, the resident KV, the bucket at 6.3's measured optimum and the arena on; 5 interleaved rounds; **both** incumbent arms, which exist since Phase 0 — this card carries **no conditional** [crit O-1 of round 2].
- **expect** 5 runs × S rows; both incumbent arms present; every board cell filled or explicitly `FEATURE GAP` with its reason. **N==0 is RED; a blank cell is RED.**
- **predict (milli → bench)** `step_wall_ms` **[48.3, 52.7] + δ_b**, `gpu_exec_ms` **[44.0, 48.0] + δ_b**, ratio **2.76–3.06x** against 0.5's chosen incumbent arm, plus the fraction-of-ceiling against 0.6's `traffic_gbs` column, named. Every term is carried from the measured deltas of 4.2, 5.2, 6.3 and 6.5 — nothing is re-derived from theory.
- **kill** `step_wall_ms` does not fall below **57.5** (5.2's own band top) beyond both CoV bands ⇒ the GPU-side and orchestration wins are cancelling somewhere; name **which counter did not move** before Phase 7 is scheduled.
- **memory gate** MG-3, all four clauses, at the landed `kv_capacity_tokens`; a breach is a board-level NEGATIVE and demotes the offending feature.
- **rollback** demote features one line each. **blast** docs.
- **observe** all seven phase counters, the route census, `gpu_device_ms`, both memory slopes, CoV per arm, loadout.
- **reprove** the seal command.
- **log-row title** `the board after the body, the residency and the plan: what is left is the non-matmul bucket and the graph`

---

## PHASE 7 — the sizing config and the non-matmul 16.6 ms
*Worktree `proxima-wt-risc08` / `risc/7-geometry`, base `risc/6-plan-stable`.*

### 7.1 — Every geometry constant into `omega-runtime.toml` `[S2 1.3, B3 P7.1 half]`
- **tier** worker · **depends_on** 3.4
- **opens (each a verified §12 violation)** `msl.rs:1017` `const PACKED_ROWS_PER_GROUP: usize = 4`, `:1030` `TILE_DIM = 8`, `:1046` `TILED_GEMM_NSG = 4`, `:2516` `let lanes_per_block = 8;` (a **local** — not overridable at all), `:2527` the loop step `SIMD_WIDTH/lanes_per_block`, `:3172/:3190/:3193` the bare `SIMD_WIDTH` literals pinning the cooperative width; the compliant surface `omega/build.rs:16/:35/:43/:59/:79/:85/:105`; `omega/src/sized.rs:45` — **`SIMD_WIDTH` stays a source const, documented as a hardware-family fact, never a policy knob.**
- **commands** add `[packed_row_block] rows_per_group, lanes_per_block, tile_dim`, `[tiled_gemm] nsg`, `[cooperative_reduce] max_threads, vec_width` (7.2 consumes the last two); route each through `resolve_int` + a cross-axis validator + `emit_sizing_consts`. Validators: `lanes_per_block` must divide `SIMD_WIDTH` (else `:2527`'s step is wrong); `rows_per_group` nonzero (a `div_ceil` denominator at `:1552`); `tile_dim` a multiple of 8. Each key's measurement record lives on the consuming const's doc in `sized.rs`, per that file's convention.
- **expect** N1 `grep -rnE '^ *(pub )?const [A-Z_]+ *: *(usize|u64|u32) *= *[0-9]' omega/src/ | grep -v sized.rs` returns **only** GGUF wire-format constants (`Q4K_BLOCK_BYTES=144` `:294`, `Q4K_BLOCK_ELEMENTS=256` `:299`, and the Q5K/Q6K/Q8_0/Q4_0/F16/BF16 siblings), each gaining a one-line doc saying it is a wire-format fact; any surviving **policy** const is RED. N2 ≥8 tests: source == TOML per key, an env override per key, and the override **visibly changes the emitted MSL** against 3.1's golden (proving `rerun-if-env-changed` works and a cached build did not silently ignore it). N3 an invalid value fails at **build time** with the validator's message. **N==0 is RED.**
- **predict (nano → micro)** behaviour-neutral at default values: 3.1's golden hashes unchanged for every route and `gpu_exec_ms` moves < 0.7% (inside R13's CoV).
- **kill** moving a const changes emitted MSL at the **default** value ⇒ an off-by-one; the golden catches it and the card stops. A value that cannot become a build-time const without becoming a runtime read stays a source const with a one-line why at the site, recorded as a **named exception** on the row, never silently.
- **memory gate** MG-1 — compile-time only. **rollback** revert build.rs + toml + const sites together; values identical by construction.
- **blast** `omega/build.rs`, `omega-runtime.toml`, `omega/src/sized.rs`, `msl.rs` const sites.
- **observe** the N1 grep (a source-level assertion runnable in the gate); `OUT_DIR/omega_sized.rs`; 3.1's golden hashes.
- **reprove** the grep + the two override commands, welded.
- **log-row title** `every GPU geometry constant traces to omega-runtime.toml with its cross-axis validator; SIMD_WIDTH stays a hardware fact and the row says why`

### 7.2 — The cooperative reduce stops being 32 lanes wide for every size `[S2 3.1, B3 P7.1]`
- **tier** worker · **depends_on** 7.1, 6.6
- **opens** `msl.rs:1517-1560` `grid_threads` — the cooperative arm is `output_total * SIMD_WIDTH`, i.e. 32 threads per output element for **every** reduction length; `:3188-3196` (`output_index = gid/SIMD_WIDTH`, `lane = gid % SIMD_WIDTHu` — the pin); `:824` `reduce_is_cooperative`; `sized.rs:45`; incumbent `ggml-metal.m:3797-3804` (nth doubles from 32 to `min(ne00/4, maxTotalThreadsPerThreadgroup)`) and `ggml-metal.metal:1679-1721` (float4 loads, `simd_sum` → threadgroup shmem → `simd_sum`; a 4096-wide row gets 1024 threads) (R8); `metal.rs:1402-1426` `pipeline_for` (the device's own `maxTotalThreadsPerThreadgroup`).
- **commands** **first, re-measure the mechanism**: run the welded MILLI cell on 6.6's tree and record the **current** `Route::ReduceCooperative` time — R13's 385 ops / 9.113 ms is a pre-Phase-4 figure and the body swap may have moved which ops route there [crit m]. Then implement a two-level tree: threads-per-output = `min(next_pow2(reduction_len / vec_width), COOPERATIVE_REDUCE_MAX_THREADS)`, floored at `SIMD_WIDTH`, **clamped against the pipeline's own `maxTotalThreadsPerThreadgroup`, never against a constant**; float4 loads when the extent is a multiple of 4. Both consts from 7.1's `[cooperative_reduce]`. Feature `metal-wide-reduce`, default-off, forwarded. Recover `gpudisp-tracked.patch` (0.7) as reference only.
- **expect** N1 the `reduce-cooperative` **op count must not move** across arms — only the time; a moved count is RED (the route changed, not the geometry). N2 CPU-oracle parity across reduction extents `{31, 32, 33, 127, 128, 4096}` — **6 cases**, proving the tree is correct at non-multiples of the width, where two-level reductions break. N3 `metal_parity` runs its full case set with the feature on, count recorded. N4 the `OMEGA_COOPERATIVE_REDUCE_MAX_THREADS` override visibly changes emitted MSL at 32 vs 1024 against 3.1's golden. **N==0 is RED.**
- **predict (milli → bench)** the cooperative bucket falls **≥10%** from the figure this card just re-measured, with **−20%** as the target (R3/M4, **MEMORY**, flagged; band [7.0, 7.6] against R13's 9.113) ⇒ `gpu_exec_ms` **[42.2, 47.1] + δ_b** and `step_wall_ms` **[46.5, 51.8] + δ_b**.
- **kill** `metal_parity` or `backend_parity` regresses ⇒ a wider tree changes float summation order and §14 binds on the oracle, not on speed. The bucket does not fall ≥10% beyond the measured CoV ⇒ R7's −20% did not survive the rebase; record the negative and do not promote. The device clamp makes the width identical to 32 for our shapes ⇒ the lever is dead; record it.
- **memory gate** MG-3. Threadgroup memory is on-chip and does not touch `device_allocated_bytes`; the per-threadgroup bytes (`max_threads/32 × 4`) go on the row; any device increase is a NEGATIVE.
- **rollback** `[cooperative_reduce] max_threads = 32` — the old behaviour exactly, no source revert.
- **blast** the cooperative body + `grid_threads`'s cooperative arm only — tiled-GEMM and packed-row-block `return` before it (`:3164-3187`) and are untouched.
- **observe** `Route::ReduceCooperative` count and tick share, `op_profile_bucket … gpu_ms` and `gpu_ns_per_op` (R13: 9.113 ms / 23,670 ns), `gpu_exec_ticks`, the +7.3% inflation quoted.
- **reprove** the ON/OFF interleaved MILLI pair, welded.
- **log-row title** `SIMD_WIDTH is a lane count, not a thread budget: the width comes from the sizing config and the clamp from the device`

---

## PHASE 8 — one emitter core, scoped honestly
*Worktree `proxima-wt-risc09` / `risc/8-emitter-core`, base `risc/3-route-value`. Scope-down stated up front: the full three-backend reorganisation of ~78 near-duplicates has **no measured payoff** and the widest blast radius in the plan. This phase lands the classification, closes CUDA's two coverage holes, and proves the core's shape on **one** kind with byte-identical emission; continuing to the remaining kinds is gated on that proof and on a pre-registered line-count delta. S2's "≤5 unclassifiable functions" threshold had no evidence behind it [round-2 j4] and is not used.*

### 8.1 — The `Dialect` classification: which of the ~26 functions are TEXT and which are STRUCTURE `[S2 7.1, B3 II, crit MS-5, g, round-2 j4]`
- **tier** judge · **depends_on** 3.4
- **opens** R5's census re-verified: `msl.rs` 4712 + `wgsl.rs` 1929 + `cuda.rs` 1838 + `metal.rs` 2385 + `wgpu_driver.rs` 872 + `backend.rs` 615 + `sized.rs` 45 + `error.rs` 84 + `lib.rs` 73 = **12,553 lines**; exactly **3 types + 2 fns** shared (`Binding`, `PackedCodec`, `PackedOperands`, `gather_count`, `gather_slots`; `wgsl.rs:105`, `cuda.rs:66`); `msl.rs:673-697`, `:4656`; `cuda.rs:146-183`; `omega/src/wgpu_driver.rs` (872 lines) — named because `omega-gate.sh [2/6]` builds `--all-targets --all-features`, so a wgpu break lands in the gate for every later card.
- **the deliverable — the seven text methods enumerated here, not elided** [crit MS-5, g]. `trait Dialect`, consumed through a **generic parameter** (§20: no `Box<dyn>`; the three backends are a closed compile-time set and each dialect monomorphises into a hot string builder):
  1. `scalar_op_expr(ScalarOp, &[&str]) -> String`  2. `preamble() -> &'static str`  3. `kernel_signature(&Bindings) -> String`  4. `entry_name(&BoundOp, Route) -> String`  5. `simd_reduce_intrinsic(ScalarOp) -> &'static str`  6. `threadgroup_barrier() -> &'static str`  7. `atomic_fold(ScalarOp) -> Option<&'static str>` — `None` is a legitimate answer producing `Route::Declined(NoDialectIntrinsic)`.
  Everything structural stays in **one generic core**: `validate`, `reduction_dims`, `bindings`, `push_body_steps`, `operand_read`, the four `render_*`, and — the classification crit-g said was missing — **`grid_threads` (`msl.rs:1517-1560`) is arithmetic and `reduce_is_cooperative` (`:824`) is a predicate, so both go to the core, not to a dialect.**
- **commands** produce `docs/bench-campaigns/2026-09-03-gpu-one-risc/dialect-map.md`: one row per one of the ~26 functions × 3 backends, each classified **TEXT (which method) / STRUCTURE (core) / DELETE (duplicate) / BEHAVIOUR (neither)**, plus the commit order. One docs commit, **no source change** — this card decides, 8.3 types.
- **expect** N = 26 × 3 = **78 rows**, every one classified with no blanks; the seven signatures written out; the `(Backend, Route)` exhaustiveness plan stated; the **BEHAVIOUR** column counted and each member named. **N==0 is RED; a blank classification is RED.**
- **predict** none — this card produces a design record.
- **kill** the BEHAVIOUR count is nonzero ⇒ the one-core claim is **scoped to the classified subset in this row, with the members named**, before 8.3 starts. No numeric threshold is asserted, because none has evidence.
- **memory gate** MG-1 — docs only. **rollback** docs revert. **blast** `docs/bench-campaigns/`; no source.
- **observe** the 78-row table; the per-method fan-in; the BEHAVIOUR list.
- **reprove** `cd $WT && awk -F'|' 'NF>3' docs/bench-campaigns/2026-09-03-gpu-one-risc/dialect-map.md | wc -l`
- **log-row title** `78 near-duplicates classified before one line moves: seven text methods, one structural core, and the functions that are neither`

### 8.2 — CUDA covers `Iota` and `Constant`, in two commits `[S2 7.2, crit B-4]`
- **tier** worker · **depends_on** 8.1
- **opens** `cuda.rs:146-183` `emit_cuda` + `CudaUnsupportedOpKind`; reference `msl.rs:2044-2103` `render_iota`/`render_constant`; `error.rs`.
- **commands** implement both kinds; **do not delete the error variant in the same commit** — land the implementation first (green), then remove the now-unreachable variant in a second commit, so rollback is one small revert rather than an unwind through downstream exhaustive matches. Update 3.3's coverage matrix pre-registration in the same commit as the change that moves it.
- **expect** ≥2 emit tests per kind; `cargo nextest run -p omega --features cuda,cpu` runs a **non-zero** count (`cuda` is not in `default` — the N==0 trap); `grep -c CudaUnsupportedOpKind omega/src/cuda.rs == 0` after the second commit; 3.3's matrix moves from CUDA 2/4 to **4/4** and the test asserts the new pre-registration. **N==0 is RED.**
- **predict (nano → micro)** the 15-cell matrix is **15/15** either `Ok` or an explicitly named `Route::Declined(reason)`; a silent absence is RED.
- **kill** CUDA cannot be compiled on this host (no toolchain) ⇒ emission is still testable **as text**, which is the point of a sans-IO emitter (§11); assert the emitted CUDA parses structurally and do **not** claim it runs.
- **memory gate** MG-1 — emission only, no device. **rollback** revert the second commit (restores the variant), then the first. **blast** `cuda.rs`, `error.rs`.
- **observe** the coverage matrix; per-backend route census.
- **reprove** `cd $WT && CARGO_TARGET_DIR=$TD bash scripts/gpu-measure-lock.sh $LOCK --wait 1800 -- cargo nextest run -p omega --all-features -E 'test(backend_route_coverage)'`
- **log-row title** `one RISC means every backend covers every kind: the 15-cell matrix and the deleted rejection`

### 8.3 — The core, one kind (`Elementwise`), with the continuation gate `[S2 7.3, B3 —, round-2 j4]`
- **tier** worker · **depends_on** 8.2
- **opens** 8.1's rows for `render_elementwise`, `scalar_op_expr`, `operand_read`, `push_body_steps`, `bindings`, `preamble`, `kernel_signature`, `entry_name`; `msl.rs:2104-2177` and its wgsl/cuda twins; `msl.rs:4656`.
- **commands** move the structural half to the core, implement the touched `Dialect` methods three times, delete the duplicates. G9's gate. One commit, green. **Then the continuation gate, decided in this row:** the remaining kinds (`Reduce` incl. every route, `Iota`/`Constant`, `Keep::Scan`) are scheduled **only if** N1 holds byte-for-byte **and** the measured line-count delta on this one kind is ≥ 600; otherwise the phase closes here with the classification and the coverage landed, and the row records the scope with its number.
- **expect** N1 **byte-identical emitted source** for every `Elementwise` op in the real program, before and after, per backend — one byte of drift is RED. N2 the gate's `ran_count` **≥** prior. N3 the alloc-tier build (`--no-default-features --features alloc`) compiles the core and all three dialects and **states which modules it built** (§3's N==0 warning). **N==0 is RED.**
- **predict (nano → micro)** `wc -l omega/src/{msl,wgsl,cuda}.rs` falls by **≥600** on this kind alone; `gpu_exec_ms` unchanged within R13's 0.7% CoV; per-op `gpu_ns` unchanged within 1% (the more sensitive of the two tests).
- **kill** N1 fails ⇒ the refactor changed emission; bisect the dialect method that moved and either restore its text or record the change as its own row with its own parity evidence. **Never accept "the new text is equivalent" without the byte comparison.**
- **memory gate** MG-1 — compile-time; runtime allocation identical by N1's construction.
- **rollback** `git revert` one commit. **Not feature-gated** (a gate on a refactor means two emitters); 3.1's golden is the firewall.
- **blast** the `Elementwise` paths in all three emitters + the new core module. Deliberately sequenced **after** every perf card so no number is entangled with a 12,553-line reorganisation.
- **observe** line counts, golden hash per route per backend, per-op `gpu_ns`.
- **reprove** `cd $WT && CARGO_TARGET_DIR=$TD bash scripts/gpu-measure-lock.sh $LOCK --wait 5400 -- cargo nextest run -p omega --all-features -E 'test(golden_source)'` + the welded gate.
- **log-row title** `kind one of four through the core: the elementwise bytes did not move, here is the hash, and here is the delta that decides whether the rest follows`

---

## PHASE 9 — write placement and the op count
*Worktree `proxima-wt-risc10` / `risc/9-write-placement`, base `risc/7-geometry`. Scheduled last on purpose: R12's own control (1194 → 616 dispatches moved wall 51.571 → 51.535) says this lever moves the least. **It is here for the RISC, not for the milliseconds, and the plan says so.***

### 9.1 — Structural injectivity at bind; the affine scatter degenerates to a strided store `[B3 P8.1, crit a, OB-3, conflict 3]`
- **tier** worker · **depends_on** 7.2
- **opens** `proxima-tensor/src/map.rs:105-131` — read verbatim this session: the write-direction convention, "this crate runs the CPU interpreter's reduce loop strictly sequentially, so a scatter never needs atomics", and **"see `shape.rs`'s `infer_reduce` doc for why a `Reduce`-wide field was rejected on blast-radius grounds"** — which rejects a **destination-extent field on `Reduce`**, not a write offset; `:175-201` `IndexMap::scatter`, `:209` `scatter_extent`, `:238` `as_gather_from_output`; **`shape.rs:469-485` `project_output_shape`: `[term] if term.coeff == 1 => Ok(iter_extents[..])` else `NotLowerable{ "reduce output maps must be pure projections in v1" }`** — THE line; **`shape.rs:441-467` `bounds_check`, which already folds `axis.offset` on the READ side** (verified); `bind.rs:1594-1606` `layout_of` (`base += i64::from(axis.offset) * stride` — the read-side offset already folds into `Layout.base`), `:1011` `build_scatter_out_layout`, `:95-98` `Layout`; `msl.rs:933`, `wgsl.rs:364`, `cuda.rs:241`, `error.rs:53`; `cpu.rs:6911` `run_reduce_scatter` with its doc's worked example (`src=[10,20,30,40]`, `idx=[2,0,2,1]`, dest extent 3, `Add`/`Zero` → `[20,40,40]`).
- **the rule: injectivity is proved by SHAPE, never by a leaf's name** [crit a, OB-3]. At bind, for a `Reduce` whose `out_map` is `Computed{indices, …}`, walk `indices` backwards: **ACCEPT** iff the chain reduces to `index(i) = coeff·i + base` where the terminal node is `Op::Iota` over an iteration axis, every intervening node is `Op::Elementwise` with body in `{Identity, Add, Multiply}` (all present in the 17 verified bodies), the non-`Iota` operand of each is a rank-0 `Op::Constant` or a rank-0 `Op::Input` broadcast over the whole iteration space, and `coeff != 0`. **On ACCEPT bind emits no scatter at all**: `coeff` folds into `out_strides`, `base` folds into `out_layout.base`, `out_scatter` becomes `None`, and the write lowers as an **ordinary strided store every backend already covers** — so there is **no GPU scatter emitter, no atomics question, and no `ScatterMayCollide` decline on the hot path.** On **REJECT** the three emitters' `ScatterNotSupported` stays and the decline is recorded as `Route::Declined(ScatterNotProvenInjective)`.
- **the honest reconciliation with round 2's ruling** [conflict 3]. Round 2 ruled "`shape.rs` untouched is the proof". That over-read `map.rs:118-124`, which — verified verbatim above — rejects a **`Reduce`-wide destination-extent field**, i.e. *where a scatter's static output extent lives*. Accepting `coeff == 1` **plus a nonzero `axis.offset`** in `project_output_shape` is a **different** change, and it is the write-side mirror of what `bounds_check` (`:441-467`) and `layout_of` (`bind.rs:1594-1606`) already do on the read side. It is adjudicated here, on its own evidence, with three guardrails: the offset must be loop-invariant; the inferred output extent adds the offset so bounds stay checkable; and **every existing `Reduce` with `offset == 0` must produce a byte-identical fingerprint and byte-identical emitted source** (2.1's vectors + 3.1's golden), which is the mechanism that keeps the blast radius provable rather than argued.
- **commands** implement the prover inside `bind`; extend `project_output_shape`; the CPU scatter path stays for REJECT. Then `cargo nextest -p proxima-tensor --all-features -E 'test(injectiv)'`, `-p omega -E 'test(scatter)'`, G9, the ORACLE cell — all welded.
- **expect** a case table with **both** directions — **ACCEPT:** `Iota`; `Add(Iota, rank-0 Constant)`; `Add(Iota, rank-0 Input)`; `Multiply(Iota, rank-0 Constant≠0)`. **REJECT:** `Multiply(Iota, Constant(0))` (coeff 0, not injective); a gather-fed index (data-dependent, **unprovable in principle** — R17); an index through `Maximum` (not affine); an index through a non-rank-0 operand. **8 cases minimum; N==0 is RED, and a table with only ACCEPT cases is RED** — the fallback is the load-bearing half. Plus: `run_reduce_scatter`'s own doc worked example reproduced exactly (the example is the spec and the test); every existing `Reduce`'s fingerprint and golden byte-identical; the six parity suites green.
- **predict (nano → micro)** on ACCEPT the emitted MSL contains **no gather-index read** for that operand and `u.out_base` carries the offset — asserted on the emitted text of a **synthetic** fixture, never on the real concatenated codec region (R16's undecidability applies there).
- **kill** any REJECT case the prover ACCEPTs. **A false ACCEPT is a race on the GPU** — a correctness defect that kills the card outright regardless of the perf story. Any fingerprint or golden drift on an existing `offset == 0` `Reduce` ⇒ the extension is not behaviour-preserving; stop.
- **memory gate** MG-1 for the prover; MG-3 for the oracle run.
- **rollback** `git revert`; `project_output_shape` returns to `coeff == 1, offset ignored` and every scatter is rejected exactly as today.
- **blast** `shape.rs` + `bind.rs`, **cross-backend**. The CPU scatter path must be re-tested: an ACCEPT that used to run through `run_reduce_scatter` now runs as an ordinary strided store, and the two must agree bit-for-bit.
- **observe** the ACCEPT/REJECT table; `Route::Declined(ScatterNotProvenInjective)` counts on the real program; the fingerprint vectors; the goldens.
- **reprove** the four commands, welded.
- **log-row title** `injectivity proved by shape, never by a leaf name: the affine scatter stops being a scatter and becomes out_layout.base`

### 9.2 — The per-token write base, and the KV write moves into the graph `[B3 P8.2 first half, S2 5.3 graph half, crit HC-2, MS-3]`
- **tier** worker · **depends_on** 9.1, 6.5
- **opens** `msl.rs:2216` (`long out_base;` inside `struct Uniforms`) and `:2361` (`long out_offset = u.out_base;`) — **the write offset is ALREADY a per-dispatch uniform, not baked MSL text** (R18, verified in R15's citation set); `metal.rs:2070` `upload_uniforms`, `:2075` the reuse path; `bind.rs:200-215` (`Layout.base` is a baked `i64` on the `BoundOp` — the one place a per-token value cannot live today); `generate.rs:1393-1400` (the KV roots are program **outputs**, 97 effective outputs), `:1559` `cached_len += new_count`; `metal.rs:1128-1147` `bound_op_retirement`.
- **the one thing this plan may mint, bounded here** [crit MS-3]. The KV write's base is `cached_len × row`, which changes every token; a baked `Layout.base` would defeat plan reuse and re-open D4. The minimal source is a **plan-level `dynamic_bases: Vec<(position, i64)>`** filled per token and patched into the uniform bytes at `upload_uniforms` — **no new `Op`, no new `BoundOpKind`, no re-bind, and it rides the uniform mechanism 6.5 already writes in place.** The alternative (making `BoundOp.extents`/`Layout.base` symbolic) is parked with its blast radius named (`grid_threads`, `kernel_cache_key`, `kernel_dispatch_shape`).
- **the output-set change, re-derived here and not left to Phase 6** [crit HC-2]. Turning the K/V writes into in-place placements shrinks `effective_outputs` from ~97 to ~1, so every liveness range from `bound_op_retirement` changes. **This card re-runs 6.5's two soundness tests and its `ARENA_PEAK_BYTES` assertion in its own commit**, and the row carries the before/after output count and the new arena peak.
- **commands** the KV component writes become placed `Reduce`s with `out_layout.base` from `dynamic_bases`; the KV roots leave the output set; feature `kv-scatter-write` in `proxima-tensor`, default-off, forwarded. Run the CPU oracle first, then both gates, then the welded BENCH cell.
- **expect** N1 `generated_text` and `2651`/`"known"` unchanged. N2 the effective-output count falls from 97 to ~1, **printed**, and 6.5's arena peak re-asserted under G8's activation term. N3 a scatter write and a read of the same buffer within one command buffer produce the written data — we have one encoder and one command buffer (`metal.rs:449-568`) and the incumbent has no barrier API at `n_cb=1` (R8), so intra-encoder ordering is what is relied on, and the test asserts it. N4 `plan_hits` still satisfies 6.2's formula (a per-token base must not re-key the plan) — a drop is RED. N5 `kv_cache_upload_bytes == 0` on every steady step. **N==0 is RED.**
- **predict (milli → bench)** `gpu_exec_ms` and `step_wall_ms` unchanged within CoV — this card moves the *write*, not the work; the payoff is 9.3's op count. `UNIFORM_BUFFER_REUSES` unchanged (the base is patched into an existing uniform, not a new buffer).
- **kill** the write-then-read within one encoder returns stale data ⇒ the write moves to its own dispatch ordered before the read; if that also fails, the card dies and the row records the ordering finding. `plan_hits` falls ⇒ the dynamic base leaked into the plan key; stop. Any `generated_text` drift ⇒ revert (§14).
- **memory gate** MG-3 plus the re-asserted `ARENA_PEAK_BYTES`; both KV slopes must stay at 0.
- **rollback** feature off; `git revert` restores the host-side append path and the 97-output set.
- **blast** `spec.rs` cached-layer builder, `omega/src/metal.rs` (`Plan` +1 field, `upload_uniforms` patch), `generate.rs` roots.
- **observe** effective-output count, `ARENA_PEAK_BYTES`, `plan_hits`, `kv_cache_upload_bytes`, `UNIFORM_BUFFER_REUSES`, the oracle.
- **reprove** the ORACLE + BENCH cells, welded, in both arms.
- **log-row title** `the KV write moves into the graph at out_layout.base, the output set collapses from 97 to one, and the arena's peak is re-derived on the graph that actually exists`

### 9.3 — Single-range attention; the op-count cell, with wall in the criterion `[S2 6.1+6.2, B3 P8.2, crit b, H-4]`
- **tier** worker · **depends_on** 9.2
- **opens** `spec.rs:2303-2319` (the doc naming the IR constraint: "`Reduce::out_map` must stay a pure projection … so nothing upstream of a reduce can splice two tensors into one axis" — **rewritten in this commit to record how the constraint was satisfied**), `:2596-2720` (the two-range combine), `:2616-2617`, `:2610`; the fan-out sites the row must scope: **`:2867-2891`** (Qwen3.5 dense attention), **`:3455`** (MoE), **`:6465`** (the split-half RoPE path, documented as **NOT** going through this function — so "the even/odd RoPE split collapses in the same move" does **not** cover it and the row says which checkpoints it does cover) [crit H-4]; `:6282` the sole caller; R3/M11 (single-range proven 488/488, 1196 → 939 BoundOps); the incumbent's 23 real ops/layer (`llama-model.cpp:4691-4845` + `llama-graph.cpp:464-497, 514-616, 1071-1121, 1205-1253`; views/reshape/permute are no-ops, `ggml-metal.m:1835-1847`; ~740 dispatches/token, R8).
- **the mechanism** with 9.2 landed the new token's K/V are **already in the cache buffer before the attention reduce runs**, so there is only ever ONE source range and the two-range combine is dead code: one `Reduce` over `[0, bucket)` with 6.1's tail mask and the existing causal mask.
- **commands** recover `merge-tracked.patch` (0.7; 1 file +1181/−82 on `spec.rs`) as **reference only** — main moved `spec.rs +8735/−2836` since (`0c3bd4f`), so expect a full conflict and **rewrite rather than merge**. Feature `attention-single-range`, default-off; the two-range body stays under `#[cfg(not(...))]` so both arms bisect green. Add the ops/layer census by grouping `NodeId` ranges to `layer_roots` (fixing the definition of "a layer" in code, which is why this is a worker card). CPU oracle first, then the welded BENCH cell.
- **expect** N1 CPU parity **488/488** (R3 M11's own count; lower is RED) and single-range == two-range to **1e-6** with the **token bit-identical** (summation order genuinely changes; the tolerance is stated and justified). N2 real ops/layer **≤ 23**; derived total `32×23 + get_rows + rms_norm + mul + output mul_mat = 740`, so **`op_count <= 780`** (a 5% allowance for our `Iota`/`Constant` control nodes, R13: 39 ops / 0.169 ms); `op_count > 780` fails brief item 8. N3 `generated_text` and `2651`/`"known"` unchanged. N4 the route census shows the elementwise count collapsing toward the incumbent's shape. N5 Qwen3.5's hybrid path **re-parity-tested, not assumed**. **N==0 is RED.**
- **predict (milli → bench)** ops/layer **37 → ≤23**; `encode_dispatch_calls → ≤780`; **`step_wall_ms` moves by less than 1 ms and `gpu_exec_ms` may rise slightly** (one wide reduce over `bucket` slots instead of two narrower ones). **The value of this card is the RISC — the dispatch count, `Concat` still nonexistent, no new `Op` — not the milliseconds. Predicting a win here would be the dishonest move** (R12: 1194 → 616 moved wall 0.036 ms and moved GPU **up** 13.5%).
- **kill, written against BOTH the control and the noise floor** [crit b]: the success shape is **dispatches down AND `gpu_exec_ms` not risen beyond 2× the measured CoV (>1.4%) AND `step_wall_ms` not risen beyond both CoV bands**. **Wall is in the criterion**, not only counts. If wall does not move, that is the **second independent** measurement saying the graph is not the mass; record the two-agreeing-results conclusion, land the graph for the RISC and maintenance, and **not as a perf row**. 1.1's adjudication re-opens only if ≤23 ops/layer proves unreachable.
- **memory gate** MG-3; fewer nodes ⇒ fewer arena buffers ⇒ `device_allocated_bytes` must **decrease** vs 9.2; an increase is a NEGATIVE.
- **rollback** feature off; the two-range body stays under `#[cfg(not(...))]`. Highest rebase-conflict surface in the plan (`spec.rs`); rebase against main early and often.
- **blast** `proxima-tensor/src/spec.rs` only — Mistral, Qwen3, Qwen3.5 hybrid and the MoE counterpart; **the split-half RoPE path at `:6465` is explicitly out of scope and the row says so.** Zero emitter change, zero driver change — the duplication was a graph defect upstream of any backend.
- **observe** ops/layer census, `encode_dispatch_calls`, `op_profile_bucket kind=elementwise op_count` (R13: 547 / 7.350 ms), `gpu_exec_ticks`, `step_wall_ms`, the per-route split, `device_allocated_bytes`.
- **reprove** the ORACLE + BENCH cells ON/OFF interleaved 3×, welded, + the census assertion.
- **log-row titles** `attention was duplicated because the graph could not write in place: with affine placement the two ranges become one, at or under the incumbent's 23 ops per layer` · `the control that says a dispatch collapse is not automatically a win, restated on our tree with wall in the criterion`

---

## PHASE 10 — the sweeps and the cells that do not exist
*Sweeps run **before** the final board so nothing changes the default tree after the board is written [crit O-5]. Worktrees `proxima-wt-risc11` / `risc/10-sweeps` (10.1, 10.2, base `risc/9-write-placement`) and `proxima-wt-risc12` / `risc/10-cross-runtime` (10.3, 10.4, base `risc/0-measure-truth`).*

### 10.1 — The geometry sweeps, re-run against the graph that now exists `[S2 10.1+10.3, B3 —]`
- **tier** hands · **depends_on** 9.3, 7.1, 7.2
- **opens** `msl.rs:1550-1552` (the packed arm's `div_ceil(PACKED_ROWS_PER_GROUP) * SIMD_WIDTH` simdgroup count); 7.1's `[packed_row_block]` and `[cooperative_reduce]` sections; R13's family table (`attn_q` 5.172 ms at 58.5 GB/s, `attn_output` 78.6, `attn_v`/`attn_k` 50–53 vs `ffn_*` 97–109) — the curve R3/M5 names (MEMORY: 52 → 147 GB/s from 256 → 8001 simdgroups; **the anchor here is R13's MEASURED family table, not the MEMORY curve**).
- **commands** two sweeps, config-only, interleaved round-robin (never blocked by value), every command welded: `OMEGA_PACKED_ROW_BLOCK_ROWS_PER_GROUP ∈ {1,2,4,8}` × 3 runs = 12 cells; `OMEGA_COOPERATIVE_REDUCE_MAX_THREADS ∈ {32,64,128,256,512,1024}` × 3 runs = 18 cells. **A config sweep is cheaper than a new kernel and may retire the lever** (§1 applied to geometry). **The reduction lengths changed when attention collapsed, so 7.2's tuning was against a graph that no longer exists.**
- **expect** **30 cells; N==0 is RED**; per-family GB/s with 0.2's corrected bytes; `attn_*` and `reduce-cooperative` op counts **constant across arms** — a moved count is RED (the route changed, not the geometry). Monotonic-then-flat is expected on the width axis; a non-monotonic curve is the finding and gets its own row.
- **predict (nano → micro)** `rows_per_group = 8` halves the simdgroup count and is **10–30% slower**; a value below 4 is faster on the low-row families by **≥5%**. The width optimum is `min(reduction_len/4, 1024)` per the incumbent's own rule (R8) and `reduce-cooperative` lands **≤60%** of its post-9.3 value.
- **kill** no `rows_per_group` value beats 4 by more than the measured CoV ⇒ **do not build split-K**; row the negative with all 12 numbers. No width value beats 32 by more than 2× CoV ⇒ the lever is dead on the new graph; record the negative with all 18 numbers and demote 7.2's feature permanently. **nsg=2 regrouping is a four-time negative (R4, `perf/metal-simdgroup-geometry`, R12 ROW 267, R12 ROW 259/260) and is a different mechanism from split-K; re-proposing it is a discipline failure, not an experiment.**
- **memory gate** MG-3; build-time config only, `device_allocated_bytes` unchanged in slope and absolute; any increase is a NEGATIVE.
- **rollback** two TOML integers. **blast** build config only.
- **observe** per-route and per-family GB/s; `gpu_exec_ticks`; `gpu_ns_per_op`.
- **reprove** the two sweeps at the landed values.
- **log-row titles** `the simdgroup-starvation hypothesis, tested with a config knob before a kernel` · `the cooperative-reduce width, re-swept against the collapsed graph`

### 10.2 — Split-K for the starving low-row shapes — with a delete-the-card entry gate `[S2 10.2, crit SD-3, h]`
- **tier** worker · **depends_on** 10.1
- **opens** `msl.rs:1550-1552`, `:2452-2530`; 0.7's `splitk-tracked.patch` and `lat-tracked.patch` (R7: `perf/q4k-split-k`, 5 files +352/−48 — the only cards that open them); 7.1's `[packed_row_block]`.
- **entry gate — this card is DELETED if it does not fire.** After 9.3 and 10.1, the route census + family table must **still** show `attn_q`/`attn_k`/`attn_v`/`attn_output` achieving **under 70%** of the `ffn_*` families' GB/s on the same body. If 4.3's winning body or 10.1's knob already closed it, **the card is deleted and the row records why, with the numbers** — a lever no longer needed is a negative worth writing down, not silent scope.
- **commands** split the reduction axis K into `split_k` partitions with a cheap combine; `split_k` traces to `[packed_row_block].split_k` (§12). Feature `metal-packed-split-k`, default-off. Sweep `OMEGA_PACKED_ROW_BLOCK_SPLIT_K ∈ {1,2,4,8}` × 3 runs, interleaved, welded.
- **expect** **12 rows**; `attn_*` op counts unchanged (a split that changes the op count changed the route — RED); **bit-reproducibility**: parity vs `cpu::evaluate` on real `blk.0.attn_q.weight` at every split value at 1e-6 (a split changes summation order; the tolerance is stated) **and the generated token bit-identical**; the same fixture run **100×** byte-identical. **N==0 is RED.**
- **predict (nano → micro)** on `metal_vs_cpu.rs`'s `matvec_batch1_f32` Mistral arm, `split_k = 4` raises achieved GB/s at the 4096-row shape from ~58 toward the ffn families' ~97, i.e. **[85, 105] GB/s**.
- **kill** the combine's cost exceeds the split's win (the elementwise count rises and the net is flat) ⇒ dead lever, negative row, all four numbers recorded. Any non-determinism across the 100 runs.
- **memory gate** MG-3. Partials are `split_k` extra intermediates per matvec **through 6.5's arena**, so the peak grows by `split_k × attn_output_bytes` — **computed and printed before enabling**; a peak above G8's activation term is a NEGATIVE and rolls back.
- **rollback** default-off feature. **blast** `msl.rs` packed body + `grid_threads`'s packed arm, behind a feature. No graph, no driver change.
- **observe** the four `attn_*` families' GB/s (true bytes from 0.2), `ARENA_PEAK_BYTES`, the route census.
- **reprove** the sweep at the landed `split_k`.
- **log-row title** `split-K for the starving low-row attention shapes — or: the census says the body already closed it, and here is the number that deleted this card`

### 10.3 — torch-MPS, honest scope `[S2 8.1, B3 P9.1]`
- **tier** worker · **depends_on** 0.6
- **opens** `proxima-onnx/scripts/torch_reference/inference_bench.py:29-32` — verified: only `--threads` and `--runs`, **no device flag**; `model.py` (the model is **mnist.onnx**: 3× Conv+Relu, BatchNorm, Flatten, Gemm+Relu, Gemm, BatchNorm, LogSoftmax — not a transformer, not Q4_K); `omega/benches/metal_vs_cpu.rs` (the **only** GPU bench outside decode, registered `omega/Cargo.toml:207-210`, doc says **UNRUN**, R9); `omega/tests/training_step_parity.rs:400-607` (GPU train step, untimed). torch 2.13.0 with MPS verified present (R0).
- **the honest scope, on the row before any number** torch has no Q4_K kernel and no GGUF loader: **a torch-MPS arm on the openchat decode does not exist and cannot be built without changing what is compared.** Two comparable surfaces exist and both are labelled: (a) **mnist batch-1 f32**, the surface the repo already models — a cold-path arm by the frequency bands, `design-favors: incumbent`; (b) the **matvec shapes** (`[1,4096]×[4096,4096]`, `[1,4096]×[4096,14336]`, `[1,14336]×[14336,4096]`) as a **roofline companion** — "what a tuned framework achieves at these shapes on this silicon", **not** a decode competitor.
- **commands** add `--device {cpu,mps}` (default `cpu`, so the existing arm stays byte-identical) with `torch.mps.synchronize()` **before** each timer stop (the MPS analogue of 0.6's readback rule); run the omega `metal_vs_cpu` bench **for the first time**; all runs welded through the mutex.
- **expect** 3 torch arms + 4 omega bench arms × 5 runs, reporting p50/p95/p99/mean/CoV. The harness asserts `torch.backends.mps.is_available()` **and** `next(model.parameters()).device.type == "mps"` or the arm silently ran on CPU and exits 0. **N==0 is RED** — a `required-features`-gated bench never invoked compiles to nothing and may not even compile.
- **predict (micro → milli)** **torch-MPS is SLOWER than torch-CPU at mnist batch 1** (a ~14-node graph where per-op MPS dispatch dominates); `matvec_batch1_f32` at Mistral shapes lands under **25%** of 0.6's measured `traffic_gbs` ceiling (the low-simdgroup starvation R3/M5 names, anchored on 0.6's MEASURED ceiling and not on the MEMORY curve). Both directions are the result; the loss is reported first (§19).
- **kill** MPS silently falls back ⇒ the arm is void; report it as a gap in torch's own harness, never as our win. The omega bench does not build under `--features metal` ⇒ that is the finding and fixing the registration is the card; a repo that does not build its own benches is our bug.
- **memory gate** MG-2 at the 540 MB ceiling (5.2); the row records peak RSS via `/usr/bin/time -l` and `torch.mps.current_allocated_memory()`, because an arm that swaps invalidates every interleaved cell sharing the box.
- **rollback** revert the flag and the timed arm; the venv is gitignored by 0.9. **blast** one python file + one existing unrun bench; zero Rust hot path.
- **observe** p50/p95/p99/mean/CoV per arm; the device assertion; peak RSS; GB/s against 0.6's named denominator column.
- **reprove** the three welded commands.
- **log-row title** `the first torch-MPS cell: mnist batch-1 and our matvec shapes, and what neither tells us about Q4_K decode`

### 10.4 — ORT-CoreML, honest scope, and the one unbounded build `[S2 8.2, B3 P9.2, crit SD-2, k]`
- **tier** worker · **depends_on** 0.9 · **scheduled terminal within Phase 10**
- **opens** `scripts/onnx_reference/bench.py:96` — verified: `providers=["CPUExecutionProvider"]` hardcoded; `export_model.py` (the model is **BAAI/bge-small-en-v1.5**, f32); the fidelity fields `cosine_similar` / `cosine_dissimilar_a/b` at `:82-85`; `run.sh` (pinned venv, `ONNX_REF_PYTHON=python3.12`). `onnxruntime` is **not installed** (R0); an ORT source checkout exists at `~/repos/others/onnxruntime`.
- **the honest scope** ORT has no Q4_K GGUF path and the CoreML EP has no int4 matvec: **there is no ORT arm on the openchat decode.** The comparable surface is BGE-small f32 embedding, which the repo already exports and benches on CPU. Frequency band: the **80% case for the BGE product**, near-zero for the decode product — a loss here gates the embedding claim, not the decode claim, and the row says which.
- **commands** add `--provider {cpu,coreml}` (default `cpu`, preserving today's arm byte-for-byte) into the `InferenceSession` call at `:96`; install `onnxruntime` into a **dedicated** venv, not `torch_reference/venv` (whose `requirements.txt` is a recorded artifact and whose pollution would break 10.3's re-prove). **The wheel path is tried first.** If no wheel: `./build.sh --config Release --use_coreml --build_wheel --parallel 4` — **capped, peak RSS recorded via `/usr/bin/time -l`, and this card is scheduled terminal within its phase so a box-saturating build never parks the single mutex while 40 other cards wait** [crit SD-2]. The measurement runs are welded; the build runs with the mutex held and a `--wait 5400` bound, so a stall fails loudly instead of hanging.
- **expect** `get_available_providers()` contains `CoreMLExecutionProvider`; 2 providers × 5 runs with the fidelity fields per provider **or** one explicit `FEATURE GAP: CoreMLExecutionProvider unavailable/rejected the graph` row carrying the ORT error text. Assert `session.get_providers()[0] == "CoreMLExecutionProvider"` **and** the **partition node count** — a 1-node CoreML partition beside 200 CPU nodes is a CPU cell wearing a CoreML label. **A missing arm is RED; a documented gap is green.**
- **predict (micro → milli)** the CoreML EP takes a **partial** partition (>1 partition, <100% of nodes) and ms/sentence lands within **2×** of the CPU EP with the fidelity fields unchanged (fp32 path).
- **kill** fidelity drift (cosine similar/dissimilar move) ⇒ CoreML chose fp16; that is a **different arm** and must be labelled as one (§14). Wheel unavailable **and** the source build fails ⇒ a documented **feature gap**, never omitted (§19: an omitted loss is a verdict).
- **memory gate** MG-2; the build's peak RSS recorded; **KILL** if the build's peak forces swap, and the box is re-sealed before the next measuring card.
- **rollback** revert the flag; remove the venv (gitignored). **blast** two python files; zero Rust.
- **observe** ms/sentence, CoV, provider partition counts (`sess_options.log_severity_level=0`), fidelity fields, peak RSS.
- **reprove** the two welded measurement commands.
- **log-row title** `the first ORT-CoreML cell: BGE-small, partitioned, and what "GPU" means when half the graph runs on CPU`

---

## PHASE 11 — the log, the roofline, and the board
*Worktree `proxima-wt-risc13` / `risc/11-board`, base `risc/10-sweeps`.*

### 11.1 — Discipline rows, rooflines, ai_docs closure `[S2 9.1, B3 P10.1 half]`
- **tier** hands · **depends_on** 0.9, 1.1, 3.4, 0.6, 10.1
- **opens** `discipline.md:18736` (ROW 233 is main's last); `rooflines.md:396-479`, `:411` (GPU ceiling = DEBT), `:751` (summary row), `:766-773` (the doc's own note that the GPU ratio "is not a gap-to-machine at all" — now answerable); `ai_docs/AGENT.md`; the three JSONL files.
- **commands** (a) renumber every `<NEXT>` placeholder sequentially from `grep -n '^## ROW' proxima-tensor/docs/discipline.md | tail -1` per G11, in land order; (b) replace `rooflines.md:411`'s DEBT with 0.6's measured ceiling in **both** denominators, update `:751`, answer `:766-773`; (c) attach at least one evidence pointer to each of 0.9's five invariants and add two more (`proxima.gpu.dispatch_count_is_not_the_denominator`, evidence: R12's 1194→616 and 9.3's second test; `proxima.gpu.memory_is_a_kill_criterion`, evidence: G8's formula and both slopes per steady step).
- **expect** `grep -c '^## ROW <NEXT>' discipline.md == 0` after landing; the last row number monotonically greater than 233; `jq -c .` parses all three JSONL files (a malformed line is RED); `bash ai_docs/query.sh gpu-decode-perf` returns ≥1 row; every landed row has zero blank cells across the 16-gate table; every negative row carries its number. **N==0 on any file is RED.**
- **predict** none — a protocol and a records card.
- **kill** two branches carrying the same literal row number reach main ⇒ the protocol was bypassed; renumber before the next land. Malformed JSONL or an empty query ⇒ fix the record shape, never bypass the index (AGENT.md is explicit).
- **memory gate** MG-1 — docs and JSONL. **rollback** `git revert`; docs only. **blast** `discipline.md`, `rooflines.md`, three JSONL files. Zero source.
- **observe** row monotonicity; record counts per file; query hit count.
- **reprove** the grep + the three `jq` commands + `bash ai_docs/query.sh gpu-decode-perf`.
- **log-row title** `main's log learns the GPU session happened, the roofline DEBT is paid, and ai_docs carries the lane's invariants with evidence`

### 11.2 — The final board, sealed AFTER every sweep `[S2 9.2, B3 P10.1, crit O-5]`
- **tier** hands · **depends_on** 11.1, 9.3, 8.3, 10.1, 10.2, 10.3, 10.4, 6.6
- **commands** one final interleaved sweep of every landed feature at the sweeps' landed values, 5 rounds, both incumbent arms, on a box whose loadout is recorded — **scheduled after 10.1/10.2 so no sweep can change the default tree behind the board** [crit O-5].
- **expect** every board cell filled: ours (`step_wall_ms`, `gpu_exec_ms`, `gpu_device_ms`, CoV, n), llama `-fa 0`, llama `-fa 1`, torch-MPS, ORT-CoreML (or its documented gap), the roofline fraction **naming 0.6's denominator column**, both memory slopes with both caps, `design-favors` and a frequency band per cell, and a provenance tag (MEASURED / DERIVED) per number. **A blank cell is RED.**
- **predict (bench) — the plan's single composed prediction, made once, here.** `step_wall_ms` **[45.5, 53.8]** and `gpu_exec_ms` **[42.2, 48.1]**, ratio **2.60–3.07x** against 0.5's chosen incumbent arm. **Derivation: §IV's band ladder, every term carried from a MEASURED card delta, none re-derived from theory. This is a composition prediction and therefore the weakest number in this document** — each component was measured alone and their sum is DERIVED. Noted without being treated as confirmation: the parallel lane reached 2.95x by a different route (R12).
- **kill** `step_wall_ms` **> 59.0** ⇒ R13's decomposition is wrong somewhere and the row names **which bucket did not move, by counter**, before any further card is scheduled. If the composed number is worse than the best single-card number, **the cards interact** and the interaction is the next work item, decomposed into inconsistency vs understanding-gap.
- **memory gate** MG-3, all four clauses, at the landed `kv_capacity_tokens`; a breach is a board-level NEGATIVE and demotes the offending feature **before** the board is written.
- **rollback** n/a (measurement); the underlying features demote one line each. A wrong row is corrected in place with a dated note, **never silently deleted**.
- **blast** docs.
- **observe** every counter in the board, the route census, the fraction-of-ceiling, both memory slopes, CoV per arm, loadout.
- **reprove** the seal command.
- **log-row title** `the GPU board, composed and re-sealed after the sweeps: <ratio>x against -fa 0 and <ratio>x against -fa 1`

---

# VI. Dependency graph

**Prose, agreeing with the picture.** Phase 0 gates everything: nothing is measured before the mutex exists (0.1), both byte counters tell the truth in **both** upload loops (0.2), the batched device window exists (0.3), the cell script exists (0.4), the anchor is re-sealed with the second incumbent arm (0.5), the roofline debt is paid (0.6), the uncommitted work is inside git (0.7), the harness asserts its own contract (0.8), and the four closed sets are compile-error-to-change (0.9). **0.5 is an ancestor of every card whose kill quotes "the CoV band"; 0.5 and 0.6 are ancestors of 6.6 and 11.2**, so the plan's only composed prediction divides by a settled denominator and reports a fraction of an existing ceiling. Phase 1 forks off 0.7: the adjudication (1.1) releases two independent recovery strands (1.2, 1.3) that feed only Phase 4; **1.3 publishes `OPS_AFTER_PRUNE`**, so no downstream card carries the undecidable phrase "1196 minus the pruned count". Phase 2 hangs off 0.9 and 1.3 and closes brief item 1 **early**: 2.1 RED → 2.2 fix → 2.3 re-anchor. Phase 3 requires 2.3, because a route census over a plan two backends disagree about is a census of nothing; **3.4 is where the census, the fingerprint and the census-sum all become gates.** Phase 4 requires 3.4 (the bake-off is scored by route, never by substring) **and** both recovery strands. Phase 5 requires 4.3 so residency is measured on top of the landed body. **Phase 6 requires 5.2 — this is the round-3 reordering: the capacity arena exists before anything buckets the leaf extent, so `found == expected` holds at every step and the 4.1↔5.3 code-level cycle does not exist** [crit RS-1, j]. Within Phase 6, 6.1 → 6.2 → 6.3, with 6.4 reached **only** if 6.3's decisive kill fires; 6.5 (the device arena) requires plan stability from 6.3 or 6.4, because a pool refilled every token is not a pool; 6.6 re-seals. Phase 7 needs 3.4 for the config card and 6.6 for the width card. Phase 8 needs only 3.4 and is off every perf path. Phase 9 requires 7.2, and **9.2 re-derives 6.5's arena partition** because it collapses the output set from ~97 to ~1 [crit HC-2]. Phase 10's sweeps require 9.3 and precede the board; 10.3/10.4 hang off Phase 0 alone and share no source with the decode lane. Phase 11 requires the terminal card of every landing phase.

```
0.1 mutex ─ 0.2 byte counters (both loops) ─ 0.3 device window ─┬─ 0.4 gpu-cell.sh ─ 0.5 re-seal + -fa 1 ─┬─ 0.8 N-contract
                                                                └─ 0.6 roofline ────────────────────────┴─ 0.9 cardinality+ai_docs
0.1 ─ 0.7 quarantine ─ 1.1 adjudicate ─┬─ 1.2 mask-fma ──────────┐
                                        └─ 1.3 picks (OPS_AFTER_PRUNE) ─┐
0.9 + 1.3 ─ 2.1 fingerprint RED ─ 2.2 bind owns layout ─ 2.3 re-anchor  │
                                        └─ 3.1 Route ─ 3.2 census ─ 3.3 backends ─ 3.4 gates ─┤
                                                                                              ├─ 4.1 selector ─ 4.2 bake-off ─ 4.3 land
                                                                                              │        │
3.4 ─ 7.1 geometry config ────────────────────────────────────────────────────────────────────┤        │
3.4 ─ 8.1 dialect map ─ 8.2 CUDA kinds ─ 8.3 core:Elementwise (continuation gate)              │        │
                                                                                    5.1 spans ─┴─ 5.2 KV arena
                                                                                              6.1 bucket+mask
                                                                                              6.2 invert assertion
                                                                                              6.3 trade cell ─┬─(kill)─ 6.4 invariant plan
                                                                                                              └─ 6.5 device arena ─ 6.6 re-seal
                                                                                    7.1 + 6.6 ─ 7.2 wide reduce
                                                                                              9.1 injectivity ─ 9.2 in-place KV write ─ 9.3 single-range
                                                                                              10.1 sweeps ─ 10.2 split-K (entry-gated, deletable)
0.6 ─ 10.3 torch-MPS ;  0.9 ─ 10.4 ORT-CoreML (terminal in its phase)
0.9 + 1.1 + 3.4 + 0.6 + 10.1 ─ 11.1 docs ─ 11.2 FINAL BOARD ← 9.3, 8.3, 10.1, 10.2, 10.3, 10.4, 6.6
```

**The true longest chain** [crit O-4]: `0.1 → 0.2 → 0.3 → 0.4 → 0.5 → 0.9 → 2.1 → 2.2 → 2.3 → 3.1 → 3.2 → 3.3 → 3.4 → 4.1 → 4.2 → 4.3 → 5.1 → 5.2 → 6.1 → 6.2 → 6.3 → 6.5 → 6.6 → 7.2 → 9.1 → 9.2 → 9.3 → 10.1 → 10.2 → 11.2` = **30 cards**. Every edge above is a real `depends_on`; S2's "21-card critical path" listed non-edges (`0.3→0.4`, `0.4→0.5`, `0.5→0.6`) and is not repeated. **And the honest caveat the path length hides:** G3 serialises every measuring card on one mutex, so the schedule length is the count of measuring cards (**33 of 44**), not the path length. There is no parallel phase; non-measuring cards (0.7, 1.1, 8.1, 11.1) may proceed only while no build runs on the box.

---

# VII. Rollback map

| card | rollback | main's default affected | firewall |
|---|---|---|---|
| 0.1 | `git revert`; `rm` the lock | no | the two-process contention test |
| 0.2 | `git revert` | instrument only | the three-way identity **in both loops** + the 8-family table vs R13's derived column |
| 0.3, 0.4 | `git revert` | instrument / scripts only | `gpu_device_ms <= gpu_exec_ms`; the reproduction band |
| 0.5 | none (measurement) | no | the memory band on unmodified main is RED and stops the plan |
| 0.6 | `git revert` | example + one doc | `readback_bytes == 0` in the window; two denominators |
| 0.7 | `git revert` — **patches and untracked contents remain in git history** | docs only | per-worktree HEAD; ten apply-checks |
| 0.8, 0.9 | `git revert` | test-only / doc+tests+JSONL | the assertion still fires on an injected hit; **four** exhaustive matches fail to compile on any variant change |
| 1.1 | a ruling; reversal requires new evidence in its own row | no | the 42-commit accounting |
| 1.2 | `worktree remove --force` + `branch -D` | no (recovery vehicle) | oracle + parity; nothing depends on it until 4.1 |
| 1.3 | per-commit `git revert`; the three are independent | yes (generic passes land in `default`) | **zero `discipline.md` hunks** asserted per pick; `gpu_exec` move < 0.2 ms; `device_allocated_bytes` must not rise |
| 2.1 | `git revert` | test + one pure fn + two `#[cfg]` accessors | **the RED is the expected state**; the field list is fixed here and not negotiable downstream |
| 2.2 | revert the deletion commit (restores `:1013`), then the bind commit | yes — `bind`'s signature, three crates | 2.1 returns to its **documented** RED; golden byte-identity; six parity suites |
| 2.3, 6.6, 11.2 | n/a (measurement); features demote one line each | yes | the memory gate is a board-level kill |
| 3.1 | `git revert` — multi-commit unwind once 8.3 lands; revert 8.3 first | yes (`route::of` is not gated) | `kernel_cache_key` byte-stability + golden emitted source, **both in this card** |
| 3.2, 3.3, 3.4 | `git revert` | instrument + emit entry points / gates | the sum identity; the two greps at zero; the measured ≤5% budget |
| 4.1 | revert the `build.rs`+toml hunks | no | both values pass the **full** `--all-features` gate |
| 4.2, 4.3 | **flip one toml line** | the default body | parity gate first; the route-count pin voids incomparable arms; the tie-break is terminal; **the loser is kept selectable** |
| 5.1 | `git revert` | yes (slot 0 is the checkpoint) | `mapping_offset_uploads` unchanged at 291 |
| 5.2 | feature off; `git revert` | no while gated | **`found == expected` by construction — no validator edited**; the pre-allocation print; both KV slopes → 0. **The build-time byte assertion is NOT rolled back** (§15) |
| 6.1, 6.2 | feature off **plus a rebuild** (the bucket is a build-time const) | no while gated | `#[cfg]`-paired assertions keep both arms green; feature-OFF **hash identity** for golden and fingerprint; 0-ULP CPU parity; `2651`/`"known"` |
| 6.3 | none (measurement) | no | **the recorded loss, if it lost, is not softened** |
| 6.4 | feature off | no | reached only if 6.3's decisive kill fired |
| 6.5 | feature default-off; `git revert` | no while gated | whole-buffer sharing only; **outputs pinned by `bound_op_retirement`**; the retire loop still fires; the peak printed before allocating |
| 7.1 | revert build.rs + toml + const sites together | yes, value-identical by construction | per-key equality + the env-override-changes-MSL test |
| 7.2 | **one toml integer** (`max_threads = 32`) | no while gated | `metal_parity`/`backend_parity`; the op count must not move |
| 8.1 | docs revert | no | the 78-row classification is the gate on 8.3 starting |
| 8.2 | revert the variant-deletion commit, then the implementation | yes | two commits, not one |
| 8.3 | `git revert` one commit | yes, byte-identical by construction | 3.1's golden; the continuation gate is decided in the row |
| 9.1 | `git revert` | yes — `shape.rs`+`bind.rs`, cross-backend | the REJECT half of the case table; fingerprint + golden byte-identity for every existing `offset == 0` `Reduce`; CPU/GPU bit-agreement |
| 9.2 | feature off | no while gated | intra-encoder ordering test; `plan_hits` must not fall; the arena peak re-asserted |
| 9.3 | feature off; the two-range body stays under `#[cfg(not(...))]` | no while gated | wall is in the kill; split-half RoPE explicitly out of scope; Qwen3.5 re-parity-tested |
| 10.1, 10.2 | one/two toml integers; default-off feature | build config only | negatives rowed with all cells; 100× determinism for split-K |
| 10.3, 10.4 | revert the flag / the timed arm; venvs gitignored | no Rust hot path | device and provider assertions; partition counts; fidelity fields |
| 11.1 | docs revert; **a wrong row is corrected in place with a dated note, never deleted** | no | row monotonicity; `jq` parses |

**Ordering rule.** Rolling back a card requires rolling back everything downstream of it in §VI first, in reverse topological order. Two exceptions are designed in: 4.2/4.3 (a config value, so the graph stays intact) and 7.2/10.1 (toml integers). **The 6.1↔5.2 pair has an explicit order** [crit RB-2]: 6.1 may be turned off while 5.2 stays on (the block reverts to a `cached_len`-sized slice of the same arena and `found == expected` still holds), but **5.2 may not be turned off while 6.1 is on** — that inverts the mismatch, and the rollback map says so on both rows. Every landing commit is a green bisect point; primitives land before callers (2.2 before 3.1; 3.1 before 8.3; 7.1 before 7.2 and 10.1; 5.2 before 6.1; 6.3/6.4 before 6.5; 9.1 before 9.2 before 9.3). **No commit lands without owner authorization.**

---

# VIII. Abandoned designs (each traced to the constraint that ruled it out)

1. **`BoundOpKind::CachedAttention`** — a fifth bound kind carrying an eight-input fused online-softmax macro-op with a post-bind structural matcher and `physical.rs` (+576). *Ruled out by* AGENTS.md's hard invariant against arbitrary rules for specific instances against a closed 4-variant set (`bind.rs:221-264`); §1's binary question (the expression exists at `spec.rs:2596-2720`); its own measurement (R12: 51.535 ON vs 51.571 OFF with `gpu_exec` **worse**, 39.841 vs 35.117); and the matcher's own per-token cost (ROW 247: `prepare` 150.7 → 11.6 ms/token). *What changed:* the plan attacks the graph (6.1 + 9.1 + 9.2 + 9.3) instead of pattern-matching the graph's defect after bind. *What survives:* `prune_dead`, the consumer index, the paired Q4_K body, and ROW 263 as 3.2's third witness. *Re-open condition:* 9.3 measuring that ≤23 ops/layer is unreachable through affine write placement.

2. **`Op::Concat` / `Op::Pad` / `Op::Tile` / `PlacedBuffer` / `write_placement`** — zero hits on main, none added. *Ruled out by* §1 + §6: `IndexMap::scatter` (`map.rs:175`), `out_scatter` (`bind.rs:253`) and `run_reduce_scatter` (`cpu.rs:6911`) already express and execute write placement on CPU today, with the worked example in the function's own doc. *What changed:* 9.1 makes the affine case **degenerate**, so not even the scatter machinery is used for the KV write.

3. **A GPU scatter emitter with injectivity proved by a leaf-name convention (`"*.write_row"`).** *Ruled out by* R17/crit (a): `grep -rn write_row` returns no relevant hits, nothing enforces the convention, and — decisively — a scatter whose `indices` is a host-fed `Op::Input` leaf is **unprovable at bind in principle**, so the census would record the name-check's answer rather than the property (the same class of defect as `classify_kind`'s substring bucket that 3.1 exists to delete). *What changed:* 9.1 proves injectivity **structurally** from `Iota(coeff 1) + loop-invariant scalar` — the construction `causal_mask` already uses — and folds the write into `out_layout.base`, so **no GPU scatter emitter is written at all** and the atomics question never arises.

4. **An atomics-based GPU scatter, to make `run_reduce_scatter` portable.** *Ruled out by* §21 and by `map.rs:110-131`'s own reasoning: the CPU needs no atomics only because its loop is sequential; a GPU version would change the semantics of colliding writes. The one case that matters is affine and needs none; every other case is a named `Route::Declined(ScatterNotProvenInjective)`.

5. **Treating "`shape.rs` untouched" as the proof that the constraint was routed around** (round 2's ruling). *Ruled out by* reading `map.rs:105-131` verbatim: what was rejected on blast-radius grounds is a **`Reduce`-wide destination-extent field**, not a write-side offset. *What changed:* 9.1 adjudicates the `project_output_shape` offset **on its own evidence**, mirrored on `bounds_check`'s existing read-side `axis.offset` fold, with byte-identity of every existing `offset == 0` `Reduce` as the guardrail — an honest adjudication instead of an over-read one.

6. **A parallel `&[Option<u64>] declared_capacity` slice threaded into both validators, and a new `QuantizedBlock` field.** *Ruled out by* three findings at once: the predicate degenerates to "accept any length"; the slice must be built in **node** order while `named_blocks` is built in **name** order (`resolve_named_blocks`, `metal.rs:584`); and its unit was never stated against `AlignedBuffer`'s element-in/byte-rounded contract. *What changed:* residency-before-bucketing makes the handed block **always** exactly the declared extent (a `cached_len`-sized slice before 6.1, a `bucket`-sized slice after), so **neither validator is edited at all**.

7. **Bucketing before residency (synthesis_2's 4.1 → 5.3 order).** *Ruled out by* crit RS-1: setting `symbols[1]` to the bucket while `LayerCache` still hands a `cached_len`-sized `Vec` returns `InputSizeMismatch` on the first token on both backends, and the card that pads the buffer depended on the card that breaks — a two-card cycle in the **code**, invisible in the `depends_on` graph. *What changed:* the whole Phase 5/6 ordering.

8. **Two mutually-exclusive cargo features for the two Q4_K bodies with a precedence rule.** *Ruled out by* `scripts/omega-gate.sh` step [2/6] `--all-targets --all-features` and [3/6] `nextest --all-features`, verified verbatim: `--all-features` would exercise only the precedence winner and the bake-off arms could never be distinguished under the crate's own gate. *What changed:* a build-time profile axis (`[q4k] body` → `rustc-cfg`), exclusive under any feature set, with the loser kept selectable.

9. **A `Mutex<BTreeMap>` route census mirroring `WIDTH_TILE_DECLINE` (`instrument.rs:842`).** *Ruled out by* §21 and by arithmetic: 1196 lock/unlock + BTreeMap lookups per token sit **inside** the very slice Phase 6 measures, guarded only by `gpu_exec_ms`, a device window that cannot see a CPU lock. *What changed:* a per-plan route table (the route is a property of the plan) plus a fixed-size atomic array on the dispatch path, with a **measured** ≤5% budget and an ON/OFF arm.

10. **`route as usize` over a data-carrying `Declined(reason)` variant, and a hot counter slot for `Declined`.** *Ruled out by* E0605 (Rust permits `enum as usize` only for unit-only enums) and by OB-2 (a declined op never dispatches, so its slot is structurally 0 and makes the sum identity vacuous). *What changed:* a unit-only `Route` with an explicit `fn slot(&self)`, the reason carried beside it, `[Counter; 8]` with eight explicit non-`Copy` initializers, and declines observable only in the cold table.

11. **A single `u64` plan fingerprint.** *Ruled out by* crit (e): two `u64`s admit exactly one predicate, yet three separate requirements (the always-green companion test, the kill on an unexpected node, and the gate's node-set diff) need node identities. *What changed:* a per-op `Vec<u64>` plus the `pub` accessors the private `Prepared`/`Plan` require [crit RS-5].

12. **Landing the second-rewrite fix late, behind the emitter reorganisation** (S2's 7.6). *Ruled out by* the fact that brief item 1 would stay known-false through thirty cards while every downstream claim about "the plan" rested on it, and by blast radius: `bind`'s signature is the widest change in the plan and is cheapest when nothing else is in flight. *What changed:* it is Phase 2.

13. **Threading `op_setup` / the encode loop.** *Ruled out by* §21 (the owner is the `Plan`), R3/M9 (non-`Send` `MTLBuffer` blocks it at the type level) and R4 ("thread count explains zero of the gap"). R13 shows the cost is 1196 `newBufferWithLength` calls — allocation, not serialism. *What changed:* 6.5 removes the work instead of distributing it; `PROXIMA_ORCH_THREADS` is not in this plan.

14. **Headlining the dispatch-count reduction as the spine** (the brief's own one-line diagnosis). *Ruled out by* R12's control (1194 → 616 moved wall 0.07% and moved GPU **up** 13.5%) plus R13's per-op table (225 packed-row-blocked ops carry 44.450 ms while 547 elementwise carry 7.350). *What changed:* the entire phase ordering — dispatch count moved from spine to counter, and Phase 9 is scheduled last, for the RISC.

15. **A kernel-fusion engine as the route to parity.** *Ruled out by* R8: `grep -rln fuse ggml/src` is **empty** at `b25346221`; rms_norm and mul dispatch as two kernels there. Parity is reachable without fusion; fusion is upside **past** parity.

16. **A spec-sheet GPU bandwidth figure to close the roofline debt.** *Ruled out by* §18 and by `rooflines.md:411`, which refused it once already. *What changed:* 0.6 is a real streaming-copy probe with a device-timed window, a `readback_bytes == 0` assertion, and two denominator columns.

17. **A second marker string to fix the classifier mislabel** (the parallel branch's ROW 263 fix). *Ruled out by* "find where information is destroyed": another substring is more of the mechanism that caused the mislabel. *What changed:* 3.1/3.2 make the route a value and delete the substring buckets; ROW 263 becomes a witness, not a patch we inherit.

18. **`BoundOp.extents` symbolic up front (`Vec<Extent>`).** *Parked, not deleted*, by blast radius: `grid_threads`, `kernel_cache_key` and `kernel_dispatch_shape` all read extents (R11 M6′ calls it "a bigger change"). *Claim it gates:* a 100% plan-hit rate at any context length with zero padding cost. *Un-park condition:* 6.3 killing bucketing at every bucket size (→ 6.4), or 9.2's dynamic-base mechanism proving insufficient.

19. **The full three-backend emitter reorganisation as a mandatory phase** (S2's 7.1–7.5, and its "≤5 unclassifiable functions" threshold). *Ruled out by* the absence of any measured payoff, the widest blast radius in the plan, and — for the threshold specifically — the absence of any evidence behind the number [round-2 j4]. *What changed:* Phase 8 lands the classification, closes CUDA's coverage holes, proves the core on one kind with byte-identical emission, and **gates continuation on that proof plus a measured line-count delta**, with the scope-down and its number on the row.

---

# IX. Open questions, each resolved by measurement (never by asking)

| # | question | card | the number that answers it | pre-registered answer | what a miss means |
|---|---|---|---|---|---|
| Q1 | Are the profiler's bytes wrong in a second place, and does the fix reach the path that produces every family number? | 0.2 | the three-way identity in **both** upload loops; the 8-family table within 1% of R13's derived column | identity holds; `total_operand_bytes` ≈ 4.07 GB/step | >5% family disagreement ⇒ the shape derivation is wrong and **no GB/s row may be written anywhere** |
| Q2 | Is `gpu_exec` real kernel time or partly wakeup latency? | 0.3 | `gpu_exec_ms − gpu_device_ms` | ≤ 1.0 ms/token | a larger gap relocates mass from the GPU bucket to orchestration and **rewrites §I.1** |
| Q3 | Does the box move R13's means or only its CoV — and does memory move? | 0.5 | wall vs 67.92, gpu vs 56.93, CoV vs 0.5%/0.7%, both memory slopes | means reproduce, CoV tightens | a memory breach on unmodified main is RED and stops the plan |
| Q4 | Is `-fa 1` a stronger incumbent, making 3.88x an understatement? | 0.5 | ms/token `-fa 0` vs `-fa 1`, interleaved | **`-fa 1` is faster**; every later ratio re-bases against it | `-fa 1` slower ⇒ `-fa 0` is genuinely their design point and R13's ratio stands unamended |
| Q5 | Is 228.9 GB/s the machine's ceiling or the incumbent's achieved rate? | 0.6 | copy-arm `traffic_gbs`, device window, 21 runs | **exceeds 228.9** ⇒ the incumbent has headroom too | below 228.9 ⇒ the reduce probe never measured bandwidth; 228.9 is the ceiling by default and `rooflines.md:766-773`'s caveat stands |
| Q6 | Is there really ONE bound plan across the executors? | 2.1, 2.2, 3.4 | the two per-op fingerprint vectors and the differing node set | differs by **exactly** `packed_operands`; green after 2.2 | a node outside that set ⇒ a **third** rewrite, which outranks every performance card |
| Q7 | Was `classify_kind` lying about the route distribution, and what does the census cost? | 3.2, 3.4 | census counts vs R13's 225/385/547/37/2; the sum vs `ENCODE_DISPATCH_CALLS`; `encode_dispatch_ms` ON−OFF | exact match; Δ ≤ 0.0235 ms | a mismatch ⇒ the substring classifier mislabelled on main too, and **every R13 bucket is restated against routes** |
| Q8 | Are mask-fma and pair-dot the same mechanism at the same speed, and is the −36%/−29% one win double-counted? | 4.2 | batched `gpu_exec_ms` vs pooled CoV with the route count pinned; then per-op family ms; then parity; then emitted bytes | one body wins at both shapes | a shape split makes the body a per-route selection, not a global one — a new card, not a tie-break |
| Q9 | Do the 2026-09-02 numbers survive a rebase onto nine commits? | 1.2, 1.3, 4.2, 7.2 | re-earned against 0.5's cell, never carried | mask-fma ≥20% at ffn; wide-reduce ≥10% | R7's numbers did not survive the rebase; record the negatives |
| Q10 | Does KV residency move `gpu_exec` at all? | 5.2 | `gpu_exec_ms` before/after, CoV band | unchanged; only `block_upload` moves 2.0 → ≤0.5 | a `gpu_exec` move means the card changed GPU work, which it was not designed to do |
| Q11 | Does the tail-mask `Select` fuse into the existing `ComposedBody`? | 6.1 | `op_count` delta | **+2** | **+34** ⇒ the fusion assumption is refuted; a bind-level composition question, not a graph question |
| Q12 | Does bucketing pay at this budget, and where is the optimum bucket? | 6.3 | `Δ(kv_cache.* gpu_ms)` vs `Δ(prepare + op_setup)` at {8,32,64,256} | net loss at 256, net win at 32–64; optimum near `prompt_tokens + PROXIMA_MAX_TOKENS` | a loss at **every** bucket ⇒ bucketing is dead as a landing route and **6.4 is unparked** |
| Q13 | What was `UNIFORM_BUFFER_REUSES` already doing before we assumed uniforms were the cost? | 6.5 | its hit rate at `metal.rs:2069`, read **first** | — | a high pre-existing rate re-scopes 6.5 to output buffers alone |
| Q14 | Does removing 1196 allocations remove the 3.9 ms, or does it reappear in `encode_dispatch` — and what does the arena cost in bytes? | 6.5 | `op_setup_ms` vs 3.90 **and** `encode_dispatch_ms` vs 0.47 **and** wall **and** `ARENA_PEAK_BYTES` vs 41,943,040 | op_setup → [0.4,0.8]; peak inside the term | op_setup falls but wall does not ⇒ orchestration overlaps GPU execution; **stop Phase 6** |
| Q15 | Is the affine-write prover sound? | 9.1 | the REJECT half of the case table | every REJECT case rejected | a false ACCEPT is a **GPU race** — a correctness defect that kills the card outright |
| Q16 | Does the per-token write base defeat plan reuse? | 9.2 | `plan_hits` against 6.2's formula with the base patched per token | unchanged | a drop ⇒ the dynamic base leaked into the plan key; stop |
| Q17 | Does halving the dispatch count buy wall on **our** stack, as opposed to theirs? | 9.3 | `encode_dispatch_calls`, `gpu_exec_ms` and **`step_wall_ms` jointly** | **< 1 ms change**, matching R12's refutation | a large win contradicts R12's control and **both** cells are re-run before either is believed |
| Q18 | Is the elementwise bucket concentrated or uniform? | 3.4 | the top-5 nodes' share of 7.350 ms | ≥50% | uniform ⇒ no single-node lever; the only lever is fewer nodes (Phase 9) |
| Q19 | Is simdgroup starvation fixable with a config knob before a kernel — and if not, does split-K fix it? | 10.1, 10.2 | the 12-cell `rows_per_group` sweep; then 10.2's entry gate (attn GB/s < 70% of ffn) and its 12-cell `split_k` sweep | knob helps below 4; the gate may **delete** 10.2 | no knob beats 4 ⇒ do not build split-K, row the negative with all 12 numbers |
| Q20 | Was the reduce width tuned for a graph that no longer exists? | 10.1 | the 18-row width sweep on the post-9.3 graph | optimum at `min(reduction_len/4, 1024)` | no value beats 32 ⇒ demote 7.2's feature permanently, with all six numbers |
| Q21 | Do torch-MPS and ORT-CoreML beat us on any lane this repo measures? | 10.3, 10.4 | p50/p95/p99 per provider with fidelity fields and partition counts | MPS slower at batch 1; CoreML takes a **partial** partition | MPS faster ⇒ a real incumbent exists at that shape; one CoreML partition ⇒ a genuine whole-graph GPU arm for the embedding lane |
| Q22 | Does the composed stack equal the sum of its measured parts? | 11.2 | composed wall/gpu vs §IV's DERIVED sum | within CoV of the sum | worse than the best single card ⇒ **the cards interact**, and the interaction is the next work item |
| Q23 | Did the memory rule hold on every card that ran a process? | every measuring card | MG-3's four clauses per steady step against G8's byte formula | held | any breach is a NEGATIVE that rolls the card back **even when it wins on time** |

---

# X. Conflict resolutions

1. **The mutex, given `flock` is absent (B3 P0.1).** **B3 wins, with the mechanism changed.** `flock(1)` does not exist on this Mac (verified three ways this session), so every `flock … -c '…'` weld in synthesis_2 would have failed silently-then-loudly on card one. The resolution is **not** `brew install flock` but a repo-local `scripts/gpu-measure-lock.sh` (python `fcntl.flock` + `os.execvp`), because it lands in git, is re-provable under §16, mutates no host state, and can carry a `--wait` bound that exits 75 — which is also the fix for SD-1's "no timeout anywhere". `brew` is recorded as the alternative on the row. The mutex is **card 0.1**, and the weld covers `scripts/omega-gate.sh` too, whose `[2/6]`/`[3/6]` run `--all-features` Metal work [crit MS-5].

2. **Residency-before-bucketing (B3) vs bucketing-first (S2).** **B3 wins; the 4.1↔5.3 cycle does not survive.** S2's 4.1 set `symbols[1] = bucket` while `LayerCache` still handed a `cached_len`-sized `Vec`, so `found != expected` at `metal.rs:991-1000` and `cpu.rs:346-356` on the first token, on both backends — and the card that padded the buffer *depended on* the card that broke [crit RS-1, j]. Here the capacity arena lands first (5.2) handing a `cached_len`-sized slice, and bucketing (6.1) hands a **`bucket`-sized slice of that same capacity arena**, so `element_count(shapes.of(node)) == block_element_count(block)` holds at **every** stage and **neither validator is ever edited**. This also disposes of crit (d): S2's "the leaf extent IS the capacity" was false for the bucket it actually computed (256, 512, 768 … is never `capacity_tokens`); the leaf extent is the **bucket**, and the slice is sized to match it.

3. **Placement: S2's GPU scatter emitter + name-convention injectivity vs B3's bind-time structural proof folding into `out_layout.base`.** **B3 wins.** Name-convention injectivity is unenforceable (`grep -rn write_row` finds nothing relevant) and, for a host-fed `Op::Input` indices leaf, **unprovable in principle** — the census would record the name-check's answer rather than the property, reproducing the exact defect `classify_kind` embodies [crit a, OB-3]. The `Iota(coeff 1) + loop-invariant scalar` form **is** provable at bind, it is the construction `causal_mask` already builds, and on ACCEPT the write **degenerates to an ordinary strided store with `out_layout.base`** — no GPU scatter emitter, no atomics question, no `ScatterMayCollide` decline on the hot path. **On `shape.rs`, honestly:** round 2 ruled "`shape.rs` untouched is the proof", citing `map.rs:118-124`. Read verbatim, that passage rejects a **`Reduce`-wide destination-extent field** — where a scatter's static output extent lives — **not** a write-side offset in `project_output_shape`. The two are different changes, and treating the first as having settled the second was an over-read. So 9.1 touches `shape.rs:469-485` **on its own evidence**: it is the write-side mirror of the `axis.offset` fold `bounds_check` (`:441-467`) and `layout_of` (`bind.rs:1594-1606`) already perform on the read side, guarded by loop-invariance, offset-aware extent inference, and **byte-identical fingerprints and goldens for every existing `offset == 0` `Reduce`** — a provable blast radius rather than an argued one.

4. **Two Q4_K features + a precedence rule (S2) vs a build-time profile axis (B3).** **B3 wins.** `omega-gate.sh [2/6]` builds `--all-targets --all-features` and `[3/6]` runs `nextest --all-features` (verified verbatim): under a precedence rule the gate would exercise only the precedence winner, so the two arms could never be distinguished by the crate's own gate, and `compile_error!`-guarded exclusivity would turn it red outright. `[q4k] body` → `cargo:rustc-cfg` (guiding-principles §8's profile input) selects exactly one body under **any** feature set, makes the bake-off a rebuild, and makes the rollback one toml line. Two of S2's rungs are kept and two are added: the parity gate and the route-count pin stay, a **batched `gpu_exec_ms`** arm becomes the primary metric (per-family numbers exist only in per-op mode, where a commit/wait sits between every op and a separate upload path runs [crit HC-4, g]), and the terminal rung is incumbent-holds. **Nothing is deleted:** the loser stays selectable, which also disposes of O-6.

5. **S2's 4.2 arena vs B3's registered-span KV arena.** **Both, because they are different objects — and both are constrained.** B3's registered host span (5.1/5.2) is the **KV** arena: page-aligned, capacity-reserved, addressed by offset through the *same* primitive the checkpoint uses (§1: extend, don't add a peer), giving `AlignedBuffer` its first production caller. S2's device arena (6.5) is the **output/uniform** arena that removes 1196 `newBufferWithLength` + 1196 uniform uploads per token. The constraints round 3 adds: **whole-buffer sharing only** — sub-allocating within one buffer makes `device_buffers.insert(node, (buffer, 0))` a lie and breaks the readback invariant spelled out at `metal.rs:2360-2364` [crit RS-3, h]; **outputs are pinned by `bound_op_retirement`'s `!outputs.contains`**, stated as the constraint the partition rests on rather than as a test oracle; and **the retire loop is not made a no-op** — it still removes from the map so lookups stay over the live set, while the arena keeps buffers alive behind it [crit RB-3]. 9.2's collapse of the output set from ~97 to ~1 obliges a re-derivation of the partition and the peak, written on both cards [crit HC-2].

6. **The board prediction's arithmetic.** **S2's 4.3 was unreachable from its own terms and is replaced by §IV's band ladder.** S2 predicted `[33,43]` from components that floor at 49.0 before its own pre-registered `+[15,25]`, and set a 3.0x kill at the exact point where its own loss was unpaid [crit O-1, O-2, O-3]. Here **every band is DERIVED from its predecessor's own predicted delta and shown in one table**: 4.3 `[55.0, 59.0]`; 5.2 `[53.5, 57.5]`; 6.3 `[51.8, 55.8] + δ_b`; 6.5 `[48.3, 52.7] + δ_b`; 7.2 `[46.5, 51.8] + δ_b`; the board **`[45.5, 53.8]` = 2.60–3.07x**, with `δ_b` a **MEASURED** quantity from 6.3 rather than a linear extrapolation dressed as a prediction. Each band is reachable from the one above it; the board's kill (wall > 59.0) sits outside the band the cards' own terms produce, so it cannot fire on a tree behaving as designed. **The board is stated as the weakest number in the document** — a composition of separately measured parts.

7. **The number of worktrees and cards (S2 49 vs B3 11).** **One worktree per phase-branch, cards on it strictly sequential: 13 worktrees, 44 cards.** A new worktree is minted only when cards can proceed independently or must be revertible from a different base — which is true of the two recovery strands (1.2, 1.3), the cross-runtime lane (10.3/10.4, which shares no source with the decode lane), and the sweeps. This removes the surface the round-2 judges actually objected to (49 worktree lines with a literal ellipsis, and a duplicated heading) while keeping every card's isolation guarantee. Verified collision-free: `git branch --list 'risc/*' 'gpu-risc/*'` = 0; no worktree matches `risc`.

8. **Where the second-rewrite fix lands: early (B3 Phase 2) vs late (S2 7.6).** **Early.** S2 left brief item 1 known-false for thirty cards behind the entire emitter reorganisation [round-2 j4], while every downstream claim about "the plan" — the route census above all — rests on there being one. It is also the widest-signature change in the plan (`bind`'s signature, three crates) and is cheapest when nothing else is in flight, and it is behaviour-neutral by prediction (emitted MSL byte-identical, the layout is the layout Metal already executed). Round 3 adds the two things that made S2's version unrunnable: the `pub` accessors the private `Prepared`/`Plan` require [crit RS-5], and a **per-op `Vec<u64>`** so the three requirements that need node identities can be met [crit e, MS-6]. `packed_operands_of` descends to proxima-tensor beside `QuantizedBlock`, which is what lets `bind` own the packed layout without minting anything [R18].

9. **The census index mapping, the counter array, and whether `Declined` gets a hot slot.** **`Route` is unit-only with an explicit `fn slot(&self) -> usize`; the hot array is `[Counter; 8]` with eight named const initializers; `Declined` gets NO hot slot.** `route as usize` is E0605 on a data-carrying variant [crit RS-2, f], so the reason travels beside the route as a separate `DeclineReason` and the mapping is an explicit `match`, never a cast. `Counter` holds an `AtomicU64` and is not `Copy` (verified), so the array cannot be written as a repeat expression and needs eight named initializers plus eight snapshot fields [crit OB-1]. And a declined op **never dispatches**, so a hot slot for it would be structurally 0 and would make `Σ ROUTE_DISPATCHES == ENCODE_DISPATCH_CALLS` hold whether declines exist or not — a vacuous gate [crit OB-2]. Declines are therefore observable **only** in the cold per-plan table, and 3.4's gate asserts **both** the hot sum identity and the cold decline count, so neither half can go unwatched.

---

### Critical Files for Implementation

- `/Users/brianbruggeman/repos/slot-0/proxima/omega/src/metal.rs` — the two byte defects `:694-700` / `:465-468` **and the second upload loop `:663-690`**; the batched window `:545-555` vs the device window `:734`; validators `:984-1000`; `bind` `:1003` then `correct_packed_matmul_layouts` `:1013`; `packed_operands_of` `:375`; private `Prepared` `:859-864`; `classify_kind` `:785-826` + call sites `:709-710`; `plan_named` `:578-585`; `encode_op` `:2179-2252` with `allocate_buffer` `:2210`, `upload_uniforms` `:2211`, `ENCODE_DISPATCH_CALLS` `:2243`, `insert` `:2249`; the retire loop `:541-543`; `bound_op_retirement` `:1128-1147`; **the readback invariant `:2355-2378`, stated at `:2360-2364`**; `register_checkpoint_mapping` `:1744-1815`; `NOCOPY_BUFFERS` `:1848-1892`; `mark_resident` `:350-362`; `page_size` `:1606`
- `/Users/brianbruggeman/repos/slot-0/proxima/omega/src/msl.rs` — `emit` `:673-697`; `kernel_cache_key` `:731-775`; `grid_threads` `:1517-1560`; `ScatterNotSupported` `:933`; **`Uniforms` `:2207-2218` with `long out_base` `:2216`, consumed `:2361`**; `push_packed_row_blocked_body` `:2452-2530` (`lanes_per_block` `:2516`, step `:2527`); `push_cooperative_reduce_body` `:3140-3194` (`:3190-3194` the 32-lane pin); the geometry consts `:1017`/`:1030`/`:1046`; the delimiter-free unpack concatenation `:1978-1982`; `emit_is_deterministic_byte_equal` `:4656`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/generate.rs` — `resolve_plan` key `:966` and **`plans.clear()` `:973`** (the reason the hit formula is consecutive-key); `build_position_inputs` `:799-827`, called `:1304-1309`; `named_blocks` `Vec::with_capacity(… + 3 + …)` **`:1313-1318`** and `"eps"`/rope `:1332-1334`; the KV loop `:1364-1389`; `symbols` `:1393`; the roots `:1393-1400`; `LayerCache` `:621-656`; `cached_len +=` `:1559`; `phys_footprint_bytes` `:248`; `token_breakdown` `:1657` / `token_breakdown_metal` `:1723-1764`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/shape.rs` and `map.rs` — **`project_output_shape` `:469-485`** (the write-side line 9.1 adjudicates) beside **`bounds_check` `:441-467`** (the read-side `axis.offset` fold it mirrors); `map.rs:105-131`, whose rejection is a **`Reduce`-wide destination-extent field**, not a write offset; `IndexMap::scatter` `:175`, `scatter_extent` `:209`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/spec.rs` — `causal_mask` `:823-845` and its consumption `:2604-2615` (the construction 6.1 mirrors with the arguments swapped); `append_mistral_cached_layer` `:2336-2865` with its constraint doc `:2303-2319`, the two-range combine `:2596-2720`, sole caller `:6282`; the KV leaves `:6216-6245`; the fan-out sites `:2867-2891`, `:3455`, `:6465`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/op.rs` — `Op` `:175-266` (5 variants under a doc header still saying four at `:166`) and **`ScalarOp` `:60-78` (17 bodies, no `GreaterEqual`, no `Less`), whose own doc `:51-53` calls it the one closed set that stays closed** — the set 0.9's tripwire now guards and 6.1 must spell its mask within