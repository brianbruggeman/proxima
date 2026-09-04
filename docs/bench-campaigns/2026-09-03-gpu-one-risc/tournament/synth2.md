I have read all six inputs in full and re-verified the load-bearing citations against main `4be2f3a` read-only (no `proxima-wt-*` entered): `ScatterNotSupported` raised at `msl.rs:933` / `wgsl.rs:364` / `cuda.rs:241`; `IndexMap::scatter` `map.rs:175`; `u.out_base` emitted `msl.rs:2734`; `counter!(ENCODE_DISPATCH_CALLS, 1)` `metal.rs:2243`; `correct_packed_matmul_layouts(&mut resolved, …)` `metal.rs:1013` after `bind(...)` `metal.rs:1004`, absent from `cpu.rs:358`; `BLOCK_UPLOAD_BYTES` counted unconditionally `metal.rs:467` before the path match `:469`; `operand_bytes = buffer.length()` `metal.rs:694-700`; `build_position_inputs` `generate.rs:1304-1309`; `named_blocks` `Vec::with_capacity(… + 3 + layer_caches.len()*3)` `:1313-1319`; `assert_eq!(runtime.plan_hits, 0, …)` `bind.rs:3053`, `plan_misses == forward_calls_taken` `:3057`; `PROXIMA_MAX_TOKENS` `bind.rs:2719` (default 24); `map.rs:118-124` states verbatim that a **`Reduce`-wide destination-extent field was rejected on blast-radius grounds**; `git branch --list 'gpu-risc/*'` = 0, no worktree matches `risc` (68 worktrees).

---

# synthesis_2 — GPU parity for proxima-tensor through omega, ONE RISC

## Diagnosis (R13 is the baseline; R16 is the new fact)

R13 (MEASURED 2026-09-03, main `4be2f3a`, loaded box, arms interleaved A B A B) is the baseline for every delta in this plan: `step_wall_ms` **67.92** (CoV 0.5%), `gpu_exec_ms` **56.93** (CoV 0.7%), incumbent `llama-bench -n 32 -p 0 -r 5 -t 8 -ngl 99` **57.08 t/s = 17.52 ms/token** (CoV 0.89%), ratio **3.88x** wall / **3.25x** kernel-only, `op_count` **1196**, `plan_hits=0 plan_misses=8`, `generated_text` identical across three runs. The 50.40 ms gap decomposes to: **26.9 ms** Q4_K matvec above the incumbent's achieved streaming rate (225 `reduce-packed-row-blocked` ops carrying **44.450** of 61.082 diagnostic ms at 97–108 GB/s on ffn and 50–58 GB/s on attn_q/k/v/o — ALU-bound, not bandwidth-bound); **16.6 ms** non-matmul GPU (385 cooperative reduces / 9.113 ms, every one dispatched at `output_total * SIMD_WIDTH` = 32 threads for every reduction length, `msl.rs:1517-1560`; 547 elementwise / 7.350 ms); **11.0 ms** orchestration with a proven root cause — the plan key is `(symbols[0], symbols[1]) = (new_count, cached_len)` (`generate.rs:966`) and `cached_len` is `Extent::Symbolic(1)` on every KV leaf (`spec.rs:6216-6245`), so the key misses **by construction** and `ff749a0`'s `self.plans.clear()` (`:973`) makes every token pay `plan_named` plus 1196 `newBufferWithLength` (`metal.rs:2210`) and 1196 `upload_uniforms` (`:2211`); **17.5 ms** irreducible weight streaming.

Three facts make the above unfalsifiable until they are fixed first. (1) **Two live instrument defects, both carded by R13**: `operand_bytes` (`metal.rs:694-700`) sums `buffer.length()`, so since `7d09145` every weight operand reports **4,140,417,024** (the whole checkpoint mapping) and `total_operand_bytes` reads 1.2 TB; `BLOCK_UPLOAD_BYTES` is counted at `metal.rs:467` **before** the path match at `:469-486`, so it reports **4,147,777,096 B/token** of "upload" while `mapping_offset_uploads=291` / `copying_uploads=4` say almost nothing moved. No GB/s row may exist until both are fixed. (2) **The route is recovered by substring** — `classify_kind` (`metal.rs:785-826`, its own doc at `:777-783` admitting the decision "is not exposed as its own accessor"), MEASURED by R12 ROW 263 to relabel 9/601 → 225/385 when a body changed; R13's own 225/385/547 split is that instrument's output. (3) **R16: brief one-RISC item 1 is false on main today.** `metal.rs:1004` binds, then `metal.rs:1013` calls `correct_packed_matmul_layouts(&mut resolved, &packed_operands…)` — a Metal-only post-bind rewrite of the bound plan, whose in-source comment at `:1005-1012` explains `layout_of` gets every packed Q4_K/Q5_K/Q6_K stride wrong; `cpu.rs:358` calls `bind::bind` and does not. **The plan Metal executes is not the plan CPU executes.** Any "one bound plan" test that fingerprints `bind`'s return value is vacuous (`bind.rs:1718-1722` has no backend parameter); the test must capture the plan **after** each driver's own rewrite, and is pre-registered to fail on main.

R12 and R13 jointly refute the brief's proposed mass ordering: 1194 → 616 dispatches moved wall **51.571 → 51.535** and moved `gpu_exec` **up** 35.117 → 39.841. Dispatch count is not the denominator; the Q4_K body and the per-token re-plan are.

---

## One-RISC binding (brief items 1–8 → cards)

| # | brief clause | bound to | how it is proved |
|---|---|---|---|
| 1 | ONE bound plan (`&[BoundOp]`, 4 kinds) from ONE rewrite engine, identical for every backend | **0.14** (RED witness) → **7.6** (the fix) → **7.7** (green + gate) | FNV-1a-64 fingerprint captured **after** `metal.rs:1013`'s rewrite and after `cpu.rs:358`, compared; pre-registered to FAIL on main [R16] |
| 2 | ONE first-class route enum decided before emission, censused `(NodeId, reason)` | **1.1, 1.2** | `omega::route::Route` + `route::of`; **lock-free**: per-plan `Vec<Route>` filled at plan time + a `[Counter; N]` atomic array at `metal.rs:2243`; census sum == `ENCODE_DISPATCH_CALLS` |
| 3 | ONE emitter core over the 4 kinds, backend TEXT only | **7.1** (design, judge) → **7.3–7.5** (one kind per worker card) | `Dialect` with 7 enumerated text methods; byte-identical golden per route per backend |
| 4 | Every backend covers every kind | **7.2, 5.2, 7.3–7.5** | `grep -c CudaUnsupportedOpKind == 0`; every route answered or `Route::Declined(reason)`, exhaustive `match (Backend, Route)` with no wildcard |
| 5 | ONE sizing config owning every geometry constant | **1.3** (+ 3.1's `[cooperative_reduce]`, 4.1's `[kv_cache]`) | `grep` for policy consts in `omega/src` returns only GGUF codec wire-format facts |
| 6 | Write placement via existing `Reduce.out_map` / `out_layout.base`, NOT a new Op | **5.1, 5.2, 5.3** | **scatter only**; `shape.rs:469-485` is NOT touched — the proof the constraint was routed around, not weakened |
| 7 | Driver-level persistent-buffer alias, NOT a new type | **5.3** | KV registered once through `register_checkpoint_mapping` (`backend.rs:402-414`, `metal.rs:1744-1815`); `AlignedBuffer` (`align.rs:69`, zero production callers) gets its first caller |
| 8 | Llama-arch graph at ≤23 real ops/layer | **6.1, 6.2** | ops/layer census asserted against R8's enumerated 23 (`op_count <= 780`) |

**Non-negotiables:** `Op` stays 5 variants (`op.rs:175-266`); `BoundOpKind` stays 4 (`bind.rs:221-264`); no `Box<dyn>`, no `PlacedBuffer`, no `Concat`/`Pad`/`Tile`, no new `IndexMap` variant; `plan`/`execute` stays not-a-pipe (adjudicated 2026-08-30, `backend.rs:1-52`).

---

## Global protocol

**G1 — naming.** Card `N.M` → worktree `/Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>`, branch `gpu-risc/<NN>-<slug>`, `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>/target`. Verified: `git branch --list 'gpu-risc/*'` = **0**, no worktree dir matches `risc`. None of R15's ~40 existing `perf/*`, `bench/*`, `feat/*`, `docs/*` names is reused; `perf/kv-device-resident`, `perf/attention-single-range`, `bench/sealed-pass`, `perf/route-census`, `perf/q4k-split-k` are all checked out elsewhere and `git worktree add` would refuse them.

**G2 — creation, welded into every card:**
```
git -C /Users/brianbruggeman/repos/slot-0/proxima worktree add \
  /Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN> -b gpu-risc/<NN>-<slug> <base-ref>
mkdir -p /Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>/target
export CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>/target
```
Teardown: `git -C /Users/brianbruggeman/repos/slot-0/proxima worktree remove --force /Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN> && git -C /Users/brianbruggeman/repos/slot-0/proxima branch -D gpu-risc/<NN>-<slug>`.

**G3 — the harness commands, token budget PINNED.** `decode_loop_max_tokens()` defaults to **24** when `PROXIMA_MAX_TOKENS` is unset (`bind.rs:2719`); **every** command in this plan, including `scripts/gpu-seal.sh`, sets it explicitly [crit MS-4].
```
# BENCH rung
cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN> && \
CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc<NN>/target \
PROXIMA_MAX_TOKENS=8 flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock -c \
'cargo nextest run -p proxima-model-interop --release --features std,metal,instrument,<card features> \
  --lib --run-ignored all --no-capture \
  -E "test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)"' \
  2>&1 | tee runs/cell-<NN>-<arm>-<run>.txt

# MILLI rung (per-op; one command buffer per op — R13 records +7.3% inflation, quote it beside every per-op number)
… PROXIMA_MAX_TOKENS=8 PROXIMA_METAL_OP_PROFILE_STEP=3 … -E "test(profiles_one_real_decode_step_by_per_op_gpu_time)"
```
`--run-ignored all` is mandatory (`#[ignore]` at `bind.rs:3001`); the test `return`s silently if the gguf is absent (`:3004-3011`), so a missing fixture exits 0 — **N==0 is RED everywhere.**

**G4 — the N contract, eos-invariant, as a formula.** From the harness's own quantities (`bind.rs:3051-3059`):
```
F := runtime.plan_misses + runtime.plan_hits == generated.0.len() + usize::from(generated.2)
S := F - 1                                  # steady rows; step 0 is prefill and is excluded from every mean
plan_hits  == F - |{ distinct (new_count, symbol1) shapes over the run }|
```
On main today symbol1 == `cached_len`, which changes every step, so distinct == F and `plan_hits == 0` — R13 confirmed. After 4.1, the shapes are prefill `(31, 0→256)` and decode `(1, 256)`, distinct == **2**, so at `PROXIMA_MAX_TOKENS=8`: **`plan_hits == 8 − 2 == 6`, `plan_misses == 2`** [crit k, j3]. **`F >= 2` or RED; `S >= 1` or RED; `op_count > 0` or RED; `ENCODE_DISPATCH_CALLS / F == op_count`.** No card asserts a literal 8, 24 or 120. The card that verifies the prompt's token count does so by dumping `symbols` per step, never by assuming 31.

**G5 — feature declaration.** Every feature a card's re-prove names is declared in the manifest that its `-p` names and forwarded down, mirroring the verified pattern `metal-tiled-gemm = ["omega?/metal-tiled-gemm"]` in `proxima-model-interop/Cargo.toml`. A `proxima-tensor` feature `Y` is forwarded as `Y = ["proxima-tensor/Y"]` in **both** `omega` and `proxima-model-interop`. **The two Q4_K bodies must never both compile:** on branch `gpu-risc/20-q4k-body-bakeoff` (a) a documented precedence rule — `#[cfg(all(feature="metal-q4k-pair-dot", …))]` selects pair-dot and `#[cfg(all(feature="metal-q4k-mask-fma", not(feature="metal-q4k-pair-dot")))]` selects mask-fma, so `--all-features` compiles exactly one body and `omega-gate.sh [2/6]` stays green; **and** (b) the bake-off's own gate runs **explicit feature sets** (`--features metal,cpu,instrument,metal-q4k-mask-fma` and `…,metal-q4k-pair-dot`), never `--all-features`, so the arms are never conflated [crit j3]. After 2.4 the loser's feature is deleted and the precedence rule is deleted with it.

**G6 — one measurer on the box.** Every command whose execution runs the decode harness, an `omega/examples/*` probe, a `cargo bench`, **or a C++/python build** (onnxruntime `./build.sh`, pip install) is welded through `flock /Users/brianbruggeman/repos/slot-0/.gpu-measure.lock -c '…'`. `flock` with no timeout **queues at the process level**, which is where the contended resource is — there is no hand-maintained queue list to fall out of date [crit j3, SD-4]. Every measuring card additionally prints and records the loadout before measuring:
```
for p in cargo rustc llama-bench llama-cli python3.11 python3.12 cdb-daemon; do \
  printf '%s=%s\n' "$p" "$(pgrep -x "$p" | wc -l | tr -d ' ')"; done; \
  printf 'load=%s\n' "$(sysctl -n vm.loadavg)"
```
A loaded box is admissible (R13 returned CoV 0.5–0.9% at load 4.7–5.7) **provided the loadout is on the row and arms are interleaved**.

**G7 — arms and CoV.** Interleaved A B A B, never before-block/after-block; ≥3 runs for milli, ≥5 for bench; mean + CoV; the llama.cpp-Metal home-turf arm on every compare row. **No kill criterion may be set inside a CoV band.** R13's bands are the floor: `step_wall_ms` 0.5%, `gpu_exec_ms` 0.7%, incumbent 0.89%. 0.3 re-measures them and **is a dependency ancestor of every card whose kill quotes "the CoV band"** [crit O-4].

**G8 — the memory gate (owner rule 2026-09-03: a memory regression is a NEGATIVE regardless of timing).** Observables that exist on main: `phys_footprint_bytes()` (`generate.rs:248`, macOS `TASK_VM_INFO.phys_footprint`) and `omega::metal::current_allocated_size()` (`metal.rs:270`), both printed per step in `token_breakdown_metal` (`generate.rs:1731`), plus `kv_cache_upload_bytes` (`generate.rs:1657`). Byte formula, every term cited to R13:
```
DEVICE_CAP_BYTES = 4_140_417_024                # the ONE checkpoint mapping buffer
                 + capacity_tokens * 262_144    # 32 layers x (64 k_even + 64 k_odd + 128 v) x 8 kv_heads x 4 B
                                                #   reproduces R13's MEASURED +262,144 B/token exactly
                 + 40_000_000                   # activations+uniforms: R13 steady 4.163e9 - 4.1404e9 = 22.6 MB, x1.75
RSS_CAP_BYTES    = 400_000_000 at prefill (R13: 310-357 MB) ; 100_000_000 steady (R13: 48-66 MB)
slope(device_allocated_bytes) <= 2_000_000 B/step   (R13 observed +1-2 MB/token; target 0 after 4.2 and 5.3)
slope(phys_footprint_bytes)   <= 1_000_000 B/step   (R13: "no monotonic trend")
plan_cache_len == 1 at every step                   (R13; the ff749a0 clear-on-miss bound)
```
At `capacity_tokens = 256`: `DEVICE_CAP_BYTES = 4_247_525_888`. **This is a KILL, not an observation, on every card whose commands run the decode harness, a probe or a bench** — exceeding any cap or slope rolls the card back even when it wins on time, and the row headlines the memory number [crit RS-5, l]. **The 34 GB trap** (R3/M11: a KV allocation from the `context_length` default, worktree only): capacity comes from `[kv_cache].capacity_tokens` in `proxima-tensor-runtime.toml`, **never** from `context_length`; `build.rs` gains `require_at_most(65536)` so a runaway fails at **build time**; and every card that allocates prints the computed byte total **before** allocating. Non-measuring cards record "memory gate: N/A — <reason>", never blank.

**G9 — row numbers.** Branches carry `## ROW <PH-NN>` placeholders only; assigned at land time from `grep -n '^## ROW' proxima-tensor/docs/discipline.md | tail -1` (main's last is **233**, `:18736`). No literal row number is written on any branch.

**G10 — provenance.** Every number carries its ledger section. MEMORY figures appear only as "MEMORY, superseded by R13": R1 `op_setup 4.394` → R13 **3.9**; `prepare 2.087` → **1.97**; `readback 0.297` → **0.22**; R2's "KV re-upload ~3.3%" has **no counterpart term in R13** and may not anchor a prediction. R1's `228.9 GB/s` is MEMORY and may not be a kill threshold — 0.5 replaces it with a MEASURED ceiling before any fraction-of-ceiling is written [crit i].

**G11 — bench ladder.** nano (counts, hashes, no device) → micro (one kernel: `q4k_matvec_probe`, `membw_probe`, `metal_vs_cpu`) → milli (`profiles_one_real_decode_step_by_per_op_gpu_time`) → bench (the interleaved board cell vs `llama-bench`). Every prediction is **exactly one rung ahead**; a card with nothing to measure writes `predict: none — this card produces records, not a measurement`. A miss kills the climb and is decomposed in the row into *inconsistency* vs *understanding-gap*, each with a named work item.

**G12 — tiers.** `hands` = Luna: tool use, bounded edits, runs the given commands verbatim, **no design judgment and no new code**. `worker` = writes code against a fixed design. `judge` = adjudicates a pre-registered rule against numbers; does not type. **There is no `judge+worker` tier** — a card needing both is split [crit j].

---

# PHASE 0 — the instruments, the seal, the denominators, and what already exists

*Nothing downstream is attributable until 0.1 lands. Both incumbent denominators (0.4, 0.5) land in this phase, before any board-level prediction is made [crit b].*

### 0.1 — Both R13 instrument defects, fixed in the first card `[S 0.1, B2 R01, crit l]`
- **tier** worker · **depends_on** —
- **worktree** `git -C …/proxima worktree add …/proxima-wt-risc01 -b gpu-risc/01-instrument-byte-accounting 4be2f3a` ; `mkdir -p …/proxima-wt-risc01/target` ; `CARGO_TARGET_DIR=…/proxima-wt-risc01/target`
- **opens** `omega/src/metal.rs:612` (`pub operand_bytes: u64`), `:692-700` (**defect 1**, verified: `device_buffers.get(source).map(|(buffer,_)| buffer.length())`); `omega/src/metal.rs:464-468` (**defect 2**, verified: `counter!(BLOCK_UPLOAD_BYTES, block_byte_len(block))` fires unconditionally **before** the path match at `:469-486`); `:1467` decl, `:1572` snapshot, `:1744-1815` `register_checkpoint_mapping`, `:1879` `upload_block_no_copy`, `:1903` `upload_block_no_copy_uncached`, `:1914` `create_no_copy_buffer`; `omega/src/msl.rs:294,299,362,451,486,491,522,527,540,544,553,556` (the codec block constants already public); `proxima-model-interop/src/generate.rs:109-212` (every `gpu_ns_per_byte` / `total_operand_bytes` / `passed_operand_bytes` consumer), `:1723` (the `token_breakdown_metal` field list).
- **commands** Defect 1: compute the operand's **tensor** byte length from `bound.operands()` + `plan.program` extents + dtype/codec (`elements*4` for f32, else `elements / block_elements * block_bytes`); keep the buffer length as a separate `bound_buffer_bytes` field so mapping-offset behaviour stays observable. Defect 2: delete the unconditional counter at `:467`; record `BLOCK_UPLOAD_BYTES_COPIED` / `_NOCOPY_WRAPPED` / `_MAPPING_OFFSET` **inside** each chosen path; keep `BLOCK_UPLOAD_CALLS`. Then:
  ```
  bash scripts/omega-gate.sh
  <G3 MILLI command, features std,metal,instrument>
  ```
- **expect** N1 `BYTES_COPIED + BYTES_NOCOPY_WRAPPED + BYTES_MAPPING_OFFSET == 4_147_777_096` per steady token — exactly today's `block_upload_bytes` (R13); a sum that does not match is RED. N2 `BYTES_MAPPING_OFFSET / BYTES_COPIED > 100` (R13: 291 vs 4 uploads); `BYTES_COPIED == 0` is RED. N3 all **9** `op_profile_family` rows match R13's shape-derived true bytes within 1% (`ffn_up` 32×33.05 MB, `ffn_down` 32×34.00, `attn_q` 32×9.45, `output.weight` 107.5 MB); N < 9 is RED. N4 `total_operand_bytes` falls from ~1.2 TB to < 6 GB/step. N5 `op_count == 1196`, one row per bound op. ≥3 new unit tests (Q4_K operand == `rows*k*0.5625`; f32 == `elements*4`; an operand bound at a non-zero mapping offset reports the tensor length). **N==0 is RED.**
- **predict (milli → bench)** `step_wall_ms` unchanged within R13's CoV, mean in **[67.6, 68.3]**. A timing move here means the byte computation is on the hot path and is itself the finding.
- **kill** `step_wall_ms` moves > 2× R13's CoV (>1.0%) ⇒ move the byte computation to once-per-plan over `prepared.resolved` and re-measure; if it still moves, roll back. Corrected per-family GB/s disagreeing with R13's hand-derived column by >10% ⇒ the byte model is wrong in a second place; **no GB/s row may be written and 0.5 is blocked**.
- **memory gate** three `AtomicU64` statics (+168 B static, `proxima-telemetry/src/metric/counter.rs:98` documents the 56 B layout); zero heap; slopes and caps must equal 0.3's sealed values exactly. **KILL** per G8.
- **rollback** `git revert`; instrument-gated only; reverting restores R13's (wrong) numbers exactly.
- **blast** `omega/src/metal.rs` (2 sites, 3 counter decls, 3 struct fields), `proxima-model-interop/src/generate.rs` printers. Zero kernel change, zero IR change, zero feature added.
- **observe** `operand_bytes`, `bound_buffer_bytes` (**NEW**, record site `metal.rs:694`), `total_operand_bytes`, `gpu_ns_per_byte`, `BLOCK_UPLOAD_BYTES_COPIED`/`_NOCOPY_WRAPPED`/`_MAPPING_OFFSET` (**NEW**, record sites `metal.rs` inside `upload_block`, `upload_packed_bytes`, `create_no_copy_buffer` `:1914`, and the mapping-offset resolution in `register_checkpoint_mapping` `:1744-1815`).
- **reprove** the G3 milli command; the row's claim is N1's sum identity and N3's 9-family table.
- **log-row title** `two byte counters were measuring the wrong thing: operand_bytes was the checkpoint mapping, block_upload_bytes was the binding — every GB/s row before this one is void`

### 0.2 — `scripts/gpu-seal.sh`: worktree-parameterized, budget-pinned, N-asserting, memory-gated `[B2 R00, S 0.4, crit MS-4]`
- **tier** worker · **depends_on** 0.1
- **worktree** `…/proxima-wt-risc02` · `gpu-risc/02-gpu-seal-harness` (base `gpu-risc/01-…`) · own target
- **opens** `git show bench/sealed-pass:scripts/sealed-pass.sh` — verified: `:4` `REPO_ROOT=".../proxima-wt-seal"`, `:25-28` four hardcoded sibling worktrees, quiet-gate `:8-18`, `MACS_PER_TOKEN=7110402048` / `WEIGHT_BYTES_PER_TOKEN_GB=3.9996`; `scripts/omega-gate.sh:37-47` (the assert-nonzero pattern to mirror); `proxima-model-interop/src/bind.rs:2719` (`PROXIMA_MAX_TOKENS`, default 24), `:3002`, `:3084`; `omega/src/metal.rs:270`; `proxima-model-interop/src/generate.rs:107, 248, 1657, 1723-1764`.
- **commands** write `scripts/gpu-seal.sh` fresh (do **not** port the hardcoded script). Signature `gpu-seal.sh <worktree-abs-path> <target-dir> <feature-list> <runs>`. It: (1) prints G6's loadout; (2) takes the `flock`; (3) **exports `PROXIMA_MAX_TOKENS=8`** — the budget is pinned in the script, not left to the environment; (4) interleaves incumbent and ours A B A B; (5) extracts named `key=value` fields with `grep -oE`, **never a `sed` backreference** (R13 records `plan_hits=10,20..70` as a `\10` artifact — a fabricated finding); (6) asserts G4's N contract and exits nonzero on `F<2 || S<1 || op_count==0`; (7) computes both memory slopes and both caps per G8 and exits nonzero on a breach; (8) emits one machine-readable line per arm. Then `bash -n scripts/gpu-seal.sh && shellcheck scripts/gpu-seal.sh` and one real pass.
- **expect** `grep -c 'proxima-wt-' scripts/gpu-seal.sh == 0`; ≥2 arms enumerated; N = runs × S rows extracted per arm; **N==0 is RED**; a run that breaches a memory cap exits nonzero.
- **predict (milli → bench)** the script's own ours/llama cells reproduce 0.3's numbers inside 0.3's measured CoV band.
- **kill** the script's numbers disagree with 0.3 beyond that band ⇒ it is measuring something else (budget, arm order); fix before any later card uses it.
- **memory gate** the script *implements* the gate; its own run asserts `DEVICE_CAP_BYTES` at `capacity_tokens=0` (= 4_180_417_024) against R13's steady 4.152–4.163 GB, plus both slopes. **KILL.**
- **rollback** `git revert`; `scripts/` only. **blast** one new script; zero library code.
- **observe** `plan_hits`/`plan_misses` (`generate.rs:1764`, `bind.rs:3046`), `op_count` (`generate.rs:107`), `phys_footprint_bytes`, `device_allocated_bytes`, `kv_cache_upload_bytes`, per-arm CoV, loadout.
- **reprove** `bash scripts/gpu-seal.sh $PWD $CARGO_TARGET_DIR std,metal,instrument 5`
- **log-row title** `the GPU seal is a script, not a memory: worktree-parameterized, budget-pinned, N-asserting, memory-gated`

### 0.3 — Replicate R13 and produce the CoV band every later kill is set inside `[S 0.3, B2 R00, crit O-4, OB-2]`
- **tier** hands · **depends_on** 0.2
- **worktree** `…/proxima-wt-risc03` · `gpu-risc/03-baseline-replicate` (base `gpu-risc/02-…`) · own target
- **opens** `scratchpad/baseline-2026-09-03/{llama_A1..3,ours_B1..3,profile}.log`; `bind.rs:3002`; `generate.rs:1721-1750`
- **commands** record loadout before and after each run; 5 interleaved pairs of `bash scripts/gpu-seal.sh $PWD $CARGO_TARGET_DIR std,metal,instrument 5` and `/Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench -m /Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf -n 32 -p 0 -r 5 -t 8 -ngl 99`, both under the `flock`.
- **expect** 5 ours-runs × S rows each, all parsed; 5 llama tables; `op_count == 1196`; `plan_hits=0 plan_misses=8` at budget 8. **N==0 is RED.**
- **predict (milli → bench)** `step_wall_ms` within ±3% of R13's **67.92** and `gpu_exec_ms` within ±3% of **56.93**; the CoV **tightens** relative to R13's 0.5%/0.7% if the box is quieter, rather than the means moving.
- **kill** `op_count != 1196` or `step_wall_ms` outside ±5% of 67.92 ⇒ the box or the tree is not what R13 measured; STOP and name the difference. **This is not a re-baseline** — R13 stays THE baseline; this card produces the band.
- **memory gate** records `phys_footprint_bytes` and `device_allocated_bytes` per step and both slopes — **this card establishes the memory floor as well as the timing floor**, and a breach of G8's caps on unmodified main is RED and stops the plan [crit OB-2]. **KILL.**
- **rollback** none (measurement); `git worktree remove`. **blast** none, zero source change.
- **observe** `step_wall_ms`, `gpu_exec_ms`, `op_setup_ms`, `prepare_ms`, `block_upload_ms`, `encode_dispatch_ms`, `readback_ms`, `plan_hits`, `plan_misses`, `plan_cache_len`, `encode_dispatch_calls`, `phys_footprint_bytes`, `device_allocated_bytes`, `kv_cache_upload_bytes`, loadout.
- **reprove** the interleaved pair above.
- **log-row title** `R13 replicated: the timing band and the memory band every later kill criterion is set inside`

### 0.4 — llama.cpp `-fa 1` as the second incumbent arm, BEFORE any board prediction `[S 8.1, B2 R02-A, crit b, O-2]`
- **tier** hands · **depends_on** 0.2
- **worktree** `…/proxima-wt-risc04` · `gpu-risc/04-llama-fa-arm` · own target
- **opens** `common/common.h:328` at checkout `b25346221` — flash attention **OFF** by default, so R13's 17.52 ms/token is the non-FA incumbent; `-fa 1` changes the attention path **and** the KV layout (`v_trans = !flash_attn`), so it is a second incumbent, not a variant (R8).
- **commands** add a second arm to `scripts/gpu-seal.sh`: the same `llama-bench` line plus `-fa 1`; run interleaved A(no-fa) B(fa) A B A B, 5 pairs, under the `flock`.
- **expect** 2 arms × 5 runs = 10 rows with ms/token and CoV each. **N==0 is RED.**
- **predict (micro → bench)** arm B lands within **±5%** of R13's arm A (17.52 ms/token, CoV 0.89%) at batch-1 decode with `n_kv ≈ 40` — FA's win is a long-context effect, and R8 records the decode path at this checkout already fusing scale+mask+max+exp+sum+normalize into one `soft_max_ext` kernel.
- **kill** arm B is >5% faster ⇒ **arm B becomes the home-turf incumbent for every later row and every ratio in R13/R1/R2/R10 re-bases against it.** Say so loudly rather than keeping the flattering arm. If this build rejects `-fa 1`, record the exact stderr and use the no-FA arm as sole incumbent with the reason on the row — a recorded scope, not a blocker.
- **memory gate** N/A for our process; the row records the arm's peak RSS via `/usr/bin/time -l`, because an incumbent arm that swaps invalidates interleaved cells sharing the box. **KILL** on our own process is not applicable and the reason is stated.
- **rollback** `git revert`; one script. **blast** `scripts/gpu-seal.sh` and the denominator of every ratio in `discipline.md`/`rooflines.md`.
- **observe** ms/token per arm, CoV, `design-favors: incumbent` on both.
- **reprove** `bash scripts/gpu-seal.sh …` with both llama arms.
- **log-row title** `the second incumbent arm lands before the board prediction: flash attention is off by default at b25346221 and here is what turning it on costs`

### 0.5 — The streaming-copy roofline probe: two denominators, readback outside the window `[S 8.2, B2 R03, crit V-4]`
- **tier** worker · **depends_on** 0.1, 0.2
- **worktree** `…/proxima-wt-risc05` · `gpu-risc/05-streaming-roofline` · own target
- **opens** `omega/examples/membw_probe.rs:139-146` — the Metal arm is a **reduce-to-scalar** (`Op::Reduce{body: Add, init: Zero, keep: Reduce}`), which is why `rooflines.md:411` records the GPU ceiling as **DEBT — not measured**; `omega/src/metal.rs:654-761` `execute_plan_op_timed` (`GPUStartTime`/`GPUEndTime`); `:547-554` (`gpu_exec` window) vs `:2358-2378` (`readback`); `rooflines.md:396-479`, `:751`, `:766-773`.
- **commands** add a third arm: `Op::Elementwise { body: ScalarOp::Identity, operands: [(src, affine identity)] }` over N f32 with a full-size output; two sizes (256 MiB, 1 GiB), 21 runs, min reported, two-size marginal. Time with `GPUStartTime`/`GPUEndTime` so the N-byte readback is **outside** the window entirely; validate correctness on a separate untimed run. Print **both** denominators on every line: `read_only_gbs = N/gpu_s` (comparable to R13's family column and to llama.cpp's weight-sweep figure) and `traffic_gbs = 2N/gpu_s` (what the memory system moved).
  ```
  flock …/.gpu-measure.lock -c 'cargo run --release -p omega --features metal,cpu,instrument --example membw_probe'
  bash scripts/omega-gate.sh
  ```
- **expect** N = 3 existing arms + 2 new copy rows + 1 marginal row; `readback_bytes == 0` inside the timed window (the assertion that proves the window is clean); a marginal row with `delta_ms <= 0` is RED; `read_only_gbs` below the CPU multi-thread triad (69.95/81.21 GB/s, ROW 176) is RED. **N==0 is RED.**
- **predict (nano → micro)** on `omega/benches/metal_vs_cpu.rs`'s `matvec_batch1_f32` Mistral arm, achieved read-only GB/s lands within **±20%** of the copy probe's `read_only_gbs`, because an f32 batch-1 matvec is a pure weight sweep. A larger miss means the f32 matvec is also ALU-bound (the Q4_K finding generalising) — a named work item, not a noise excuse.
- **kill** the two-size marginal's CoV exceeds 5% over 21 runs ⇒ raise both sizes 4× and re-run once; still >5% ⇒ report single-size numbers with the contamination stated and leave the ceiling DEBT. **No spec-sheet figure is ever substituted** (§18; `rooflines.md:411` already refused that once).
- **memory gate** the 1 GiB arm allocates 1 GiB host + 1 GiB in + 1 GiB out = **3 GiB peak**; explicit per-card caps `device <= 3_000_000_000`, `RSS <= 2_500_000_000`, asserted by the probe via `current_allocated_size()` before and after. This is a probe, not the decode path, so the decode caps are **deliberately** exempted and the exemption is on the row. **KILL** at the probe caps.
- **rollback** `git revert`; example only. **blast** `omega/examples/membw_probe.rs`, `rooflines.md:396-479, :751, :766-773`.
- **observe** `GPU_EXEC_TICKS` vs `READBACK_TICKS`, `readback_bytes == 0`, both GB/s columns with CoV.
- **reprove** the `cargo run --example membw_probe` command above.
- **log-row title** `the GPU streaming ceiling stops being DEBT: a copy, GPU-timestamped, readback outside the window, both denominators printed`

### 0.6 — Quarantine the uncommitted worktree diffs INSIDE git, from each worktree's own HEAD, with untracked content `[S 0.5, crit RS-4, j3]`
- **tier** hands · **depends_on** 0.2
- **worktree** `…/proxima-wt-risc06` · `gpu-risc/06-quarantine-uncommitted` · own target
- **opens** R7's table (10 worktrees; `proxima-wt-all` 13 files +3859/-197 **and 6 untracked**); the three worktrees based on other HEADs: `gpuker@bfc150d`, `q4k@a2175c2`, `lat@14f1304` [crit j3].
- **commands** run verbatim; Luna runs these **from inside each named worktree**:
  ```
  R=/Users/brianbruggeman/repos/slot-0/proxima-wt-risc06/docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered
  mkdir -p $R
  for wt in all drive rules splitk place merge gpudisp gpuker q4k lat; do
    W=/Users/brianbruggeman/repos/slot-0/proxima-wt-$wt
    ( cd $W && git rev-parse HEAD > $R/$wt-head.txt \
      && git diff > $R/$wt-tracked.patch \
      && git status --porcelain > $R/$wt-status.txt \
      && git ls-files --others --exclude-standard -z \
         | xargs -0 -I{} sh -c 'mkdir -p '"$R"'/$wt-untracked/$(dirname {}); cp {} '"$R"'/$wt-untracked/{}' )
  done
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-risc06 && git add docs/bench-campaigns && git status --porcelain
  for wt in all drive rules splitk place merge gpudisp gpuker q4k lat; do
    git -C /Users/brianbruggeman/repos/slot-0/proxima apply --check \
      --directory=. $R/$wt-tracked.patch 2>&1 | sed "s/^/$wt /"
  done
  ```
  The patches and the untracked **contents** are committed to this branch — not left in `/tmp`, which is OS-cleared and whose deletion was Plan A's stated rollback [crit RS-4].
- **expect** N = **10** `*-tracked.patch` files present and non-empty, **10** `*-head.txt`, and an `*-untracked/` tree for every worktree R7 records untracked files in (`proxima-wt-all`: 6). Each `git apply --check` runs **against that worktree's own recorded HEAD**, not against `2b95210` — `gpuker@bfc150d`, `q4k@a2175c2`, `lat@14f1304` are not at `2b95210` and would fail spuriously [crit j3]. **N==0 is RED**; an empty patch for a worktree R7 lists as dirty is RED; a missing untracked capture where `git status` shows `??` is RED.
- **predict** none — this card produces records, not a measurement.
- **kill** a patch fails `git apply --check` at its own recorded HEAD ⇒ that worktree mutated since the R7 audit; re-audit before anything is landed from it.
- **memory gate** N/A — no code path, no allocation; stated rather than blank.
- **rollback** `git revert` one commit; the patches **remain in git history**, which is the point. **blast** `docs/bench-campaigns/` only.
- **observe** patch count, per-patch line counts vs R7's table, per-worktree HEAD, `git apply --check` exit codes.
- **reprove** the apply-check loop above.
- **log-row title** `the measured-but-uncommitted wins enter git before anything touches them — from each worktree's own HEAD, untracked files included`

### 0.7 — Adjudicate `BoundOpKind::CachedAttention`; extract the parallel branch's rows `[S 0.7, B2 R04-judge]`
- **tier** judge · **depends_on** 0.6
- **worktree** `…/proxima-wt-risc07` · `gpu-risc/07-adjudicate-cached-attention` · own target
- **opens** `git diff main..perf/cached-attention-streaming -- proxima-tensor/src/bind.rs proxima-tensor/src/physical.rs`; `git show perf/cached-attention-streaming:failure-cached-attention-matcher.md`; against them `proxima-tensor/src/bind.rs:221-264` (4 kinds), `op.rs:175-266` (5 ops), workspace `AGENTS.md` ("we should not be adding arbitrary rules/code for specific instances"). Read the **diff hunks**, never the commit list, and never enter the worktree.
- **the ruling, four citable grounds** (1) a fifth variant of a closed set minted for one model's attention shape; (2) it reconstructs information the graph destroyed — their own failure record abandons the BoundOp-only matcher as "a heuristic" that "cannot prove the semantic roles", so the defect is upstream in `append_mistral_cached_layer` (`spec.rs:2336-2865`); (3) MEASURED not to buy wall time — R12: 51.535 ON vs 51.571 OFF (CoV 1.75–2.16%) with dispatches 1194 → 616 and GPU **worse** (39.841 vs 35.117); (4) the matcher is itself a per-token CPU cost — their ROW 247: `prepare` 150.7 ms/token before indexing, 11.6 after, because `plan_hits=0`. **REJECT** `CachedAttention`, `physical.rs`, `render_cached_attention`. **KEEP** as separate cards: `prune_dead` (0.8), the consumer index (0.9), the paired Q4_K body (2.2), and their ROW 263 classifier-mislabel **finding** (into 1.1's row as the third witness — their fix adds a second marker string, i.e. more of M10, and is not ported).
- **commands** the `git show`/`git diff` above; extract ROWs 234-267 verbatim into `docs/bench-campaigns/2026-09-03-gpu-one-risc/parallel-rows.md`; one docs commit.
- **expect** N = **34** rows extracted, each tagged keep / renumber / supersede; the four measured negatives preserved as negative rows (float4 accumulation, explicit unroll, width-64 dispatch, row-batch 8, tiled-gemm at decode, nsg=2 — now a **fourth** negative across two lanes); `git log --oneline main..perf/cached-attention-streaming | wc -l == 42` accounted member-by-member. **N==0 is RED.**
- **predict** none — a recorded decision boundary.
- **kill (pre-registered re-open condition)** if 6.2 measures that ≤23 ops/layer is unreachable through scatter placement, this adjudication re-opens **with that number attached**.
- **memory gate** N/A — docs only; stated. **rollback** docs revert. **blast** `discipline.md`, `docs/bench-campaigns/`. No source.
- **observe** row count landed; the 42-commit accounting.
- **reprove** `grep -c '^## ROW' docs/bench-campaigns/2026-09-03-gpu-one-risc/parallel-rows.md`
- **log-row titles** `the fifth bound kind the binding does not admit, and what its measurement is kept for` · `the parallel lane's measured negatives, renumbered onto main`

### 0.8 — `prune_dead`, and the pruned count becomes a MEASURED published number `[S 0.6, B2 R04-C1, crit O-5]`
- **tier** worker · **depends_on** 0.7, 0.3
- **worktree** `…/proxima-wt-risc08` · `gpu-risc/08-prune-dead` · own target
- **opens** `proxima-tensor/src/bind.rs:200-215` (`BoundOp`); `git show 216d925` ("drop dead resolved nodes before GPU dispatch") — R12 marks it generic and RISC-conformant.
- **commands** `git cherry-pick -x 216d925`; `bash scripts/proxima-tensor-gate.sh`; `bash scripts/omega-gate.sh`; then the G3 bench cell. **This card closes 0.6's free variable** [crit O-5]: it prints and records `OPS_AFTER_PRUNE` — the openchat graph's `op_count` after pruning, measured on this tree — and **every downstream card quotes that measured number, never the expression "1196 minus the pruned count"**.
- **expect** N ≥ 2 new tests (a program with a dead resolved node; one with none) plus both gates' nonzero counts; `OPS_AFTER_PRUNE` printed and recorded; `op_count == 1196` **is RED for this card specifically** (a pruner that removes nothing did not land). **N==0 is RED.**
- **predict (nano → micro)** `encode_dispatch_calls` falls from R13's **1196** by exactly the pruned-node count, and every removed node is nameable; `gpu_exec_ms` moves **< 0.2 ms** (R13: the 39 degenerate `constant`/`iota` control ops total 0.169 ms).
- **kill** `gpu_exec_ms` moves > 0.2 ms ⇒ it removed live nodes; that is a correctness event — stop and diff `generated_text`. Any change to `generated_text` from `"Here is a simple Python function that returns"` ⇒ revert (§14; the `2651`/`"known"` oracle at `bind.rs:2797-2803`).
- **memory gate** `prune_dead` **removes** device buffers, so `device_allocated_bytes` must be `<=` 0.3's sealed steady value; an **increase** is a NEGATIVE and rolls back regardless of the dispatch win. **KILL.**
- **rollback** `git revert` one commit; no feature flag (a generic pass lands in `default`); bisect by revert.
- **blast** `proxima-tensor/src/bind.rs` — every backend consumes the shortened `&[BoundOp]`, which is why 0.14 runs after it.
- **observe** `OPS_AFTER_PRUNE` (**NEW**, record site: the `prune_dead` return in `bind.rs`, printed via `op_count` at `generate.rs:107`), `encode_dispatch_calls`, `device_allocated_bytes`.
- **reprove** `cargo nextest run -p proxima-tensor --features std,instrument -E 'test(prune_dead)'` + the G3 bench cell.
- **log-row title** `dead resolved nodes never reach a backend, and the post-prune op count becomes a measured number every later card quotes`

### 0.9 — The consumer index `[B2 R04-C2]`
- **tier** worker · **depends_on** 0.8
- **worktree** `…/proxima-wt-risc09` · `gpu-risc/09-consumer-index` (base `gpu-risc/08-…`) · own target
- **opens** the parallel branch's ROW 248 commit (`prepare` 150.7 → 11.6 ms/token with the matcher, i.e. a generic bind speedup independent of the rejected matcher); `proxima-tensor/src/bind.rs` consumer scan.
- **commands** `git cherry-pick -x <the consumer-index commit>`; both crate gates; the G3 bench cell.
- **expect** both gates nonzero and green; `prepare_ms` **decreases** from R13's **1.97**; `prepare_ms >= 1.97` is RED for this card. **N==0 is RED.**
- **predict (milli → bench)** `gpu_exec_ms` unchanged within R13's 0.7% CoV; `step_wall_ms` improves by exactly the `prepare_ms` delta.
- **kill** `generated_text` drift ⇒ revert. `prepare_ms` unchanged ⇒ the index does not bind here; record the negative and drop the commit.
- **memory gate** an index is a `Vec` sized by program length; `device_allocated_bytes` unchanged, RSS slope within G8. **KILL.**
- **rollback** `git revert` one commit. **blast** `proxima-tensor/src/bind.rs` only.
- **observe** `prepare_ms`, `PREPARE_CALLS`/`PREPARE_TICKS`, `phys_footprint_bytes`.
- **reprove** the G3 bench cell; `grep -o 'prepare_ms=[0-9.]*' runs/cell-09-*.txt`
- **log-row title** `a consumer index removes repeated bind work (generic, kept from the parallel lane)`

### 0.10 — Clean the tree so interleaved A/B is safe `[S 0.8]`
- **tier** hands · **depends_on** —
- **worktree** `…/proxima-wt-risc10` · `gpu-risc/10-clean-tree` · own target
- **opens** `git status --porcelain` on main — verified: `?? docs/bench-campaigns/2026-09-03-gpu-one-risc/` and `?? proxima-onnx/scripts/torch_reference/venv/`; mirror `scripts/onnx_reference/.gitignore`.
- **commands** add `venv/` to `proxima-onnx/scripts/torch_reference/.gitignore`; commit the campaign directory as real content. One `chore:` commit.
- **expect** `git status --porcelain | wc -l == 0`. **Non-zero is RED** for every later interleaved measurement.
- **predict** none — a state assertion.
- **kill** n/a. **memory gate** N/A — no code path. **rollback** `git revert`. **blast** one ignore file + one evidence directory.
- **observe** `git status --porcelain` line count.
- **reprove** `git status --porcelain`
- **log-row title** `a dirty tree is not a measurable tree`

### 0.11 — The RISC's cardinality becomes a compile error to change `[S 0.9, crit M-8]`
- **tier** worker · **depends_on** 0.8
- **worktree** `…/proxima-wt-risc11` · `gpu-risc/11-risc-cardinality` · own target
- **opens** `proxima-tensor/src/op.rs:166` — the doc header reads **"The four generators."** directly above `pub enum Op` at `:175`, which has **5** variants; `bind.rs:221-264` (`BoundOpKind`, 4).
- **commands** correct the header to name five and say which is which; add two tests written as **exhaustive `match`es over a constructed value with no `_` arm**, so adding a variant breaks the build rather than a count.
- **expect** N = 2 new tests; `grep -c "The four generators" proxima-tensor/src/op.rs == 0`. **N==0 is RED.**
- **predict (nano → micro)** the exhaustive matches compile with no wildcard on today's tree — the cardinality is 5/4 as R5 reads it, and any future fifth bound kind fails the build before review.
- **kill** the match needs a wildcard ⇒ the variant count is not what R5 read; stop and re-derive the one-RISC binding.
- **memory gate** N/A — doc + tests. **rollback** `git revert`. **blast** `op.rs` doc + test module.
- **observe** variant counts; the doc grep.
- **reprove** `cargo nextest run -p proxima-tensor --features std -E 'test(risc_cardinality)'`
- **log-row title** `the RISC's own doc said four over five variants; the cardinality is now a compile error to change`

### 0.12 — The harness N-contract and the `plan_hits` formula `[S 0.2, B2 §N-contract, crit k, j3]`
- **tier** worker · **depends_on** 0.3
- **worktree** `…/proxima-wt-risc12` · `gpu-risc/12-harness-n-contract` · own target
- **opens** `proxima-model-interop/src/bind.rs:3040-3059` — verified: `metal_decode_summary` println `:3042-3049`, `let forward_calls_taken = generated.0.len() + usize::from(generated.2);` `:3051`, `assert_eq!(runtime.plan_hits, 0, "cached_len grows every decode step, so no (new_count, cached_len) shape can repeat within one call")` `:3053-3055`, `assert_eq!(runtime.plan_misses, forward_calls_taken, …)` `:3057-3059`; the test's doc `:2994-2998`; `generate.rs:905` (the **single** `plan_hits` definition), `:968` (increment), `:1764` and `bind.rs:3046` (the two prints of that same field).
- **commands** do **not** weaken the assertions. Restate them as G4's formula: `assert_eq!(runtime.plan_hits, expected_plan_hits())` where `expected_plan_hits()` returns `F - distinct_shape_count()`, and add `assert_eq!(plan_hits + plan_misses, F)` and `assert!(plan_misses >= 1)`. Add a `symbols`-per-step dump behind `instrument` so `distinct_shape_count()` is **observed, not assumed**. On main `distinct_shape_count() == F` ⇒ `expected_plan_hits() == 0`, so today's assertion is preserved exactly; **4.1 changes it in 4.1's own commit, `#[cfg]`-paired** so both arms stay green.
- **expect** the harness prints exactly `F` breakdown lines; at budget 8 on main: F=8, S=7, `plan_hits=0 plan_misses=8`, `stopped_by_eos=false`; the assertion still fires on an artificially injected hit; the `symbols` dump prints one `(new_count, symbol1)` pair per step. **N==0 is RED.**
- **predict (nano → micro)** the dumped symbol pairs on main are `(prompt_len, 0)` then `(1, cached_len)` with `cached_len` strictly increasing — 8 distinct shapes, which is the mechanism `plan_hits=0` is, and the number 4.1 changes to **2**.
- **kill** the harness reads `plan_hits != 0` on unmodified main ⇒ the counter's semantics moved since R13; stop and re-derive before 4.1 is designed.
- **memory gate** test-only; caps and slopes must equal 0.3's. **KILL** (it runs the harness).
- **rollback** `git revert` one commit; test-only. **blast** `proxima-model-interop/src/bind.rs` test module only — edited once, here, because every card from 2.3 on re-proves through it.
- **observe** `plan_hits`, `plan_misses`, `plan_cache_len`, `tokens_generated`, `stopped_by_eos`, the `symbols` dump (**NEW**, record site `generate.rs:966` region under `instrument`).
- **reprove** the G3 bench cell; `grep -o 'plan_hits=[0-9]*\|plan_misses=[0-9]*\|symbols=([0-9]*,[0-9]*)' runs/cell-12-*.txt`
- **log-row title** `the bench harness asserts the defect: plan_hits as a formula over distinct shapes, and the symbol dump that proves the count`

### 0.13 — Row-number placeholder protocol `[S 0.11, crit j — retiered hands → judge]`
- **tier** judge · **depends_on** 0.7
- **worktree** `…/proxima-wt-risc13` · `gpu-risc/13-row-number-protocol` · own target
- **opens** `proxima-tensor/docs/discipline.md:18736` (last row on main = **ROW 233**); R10's record of ROW 205 at line 17864 preceding ROW 204 at 17950 (non-monotonic from concurrent worktrees); the three unlanded "ROW 234"s and the parallel branch's 234-267.
- **commands** author the protocol: every branch writes `## ROW <PH-NN>`; at land time `grep -n "^## ROW" proxima-tensor/docs/discipline.md | tail -1` gives the maximum and the landing commit rewrites the placeholder to `max+1`; a row whose re-prove does not run **today** does not land (§16); a collision rule for two branches landing in the same window.
- **expect** `grep -c "^## ROW <PH-" proxima-tensor/docs/discipline.md == 0` on main after every land; every landed row has zero blank cells across the 16-gate table. **N==0 is RED.**
- **predict** none — a protocol.
- **kill** two branches carrying the same literal row number reach main ⇒ the protocol was bypassed; renumber before the next land.
- **memory gate** N/A — docs. **rollback** docs revert. **blast** `discipline.md`.
- **observe** placeholder count; `grep -oE '^## ROW [0-9]+' | sort -c` monotonicity.
- **reprove** `grep -c "^## ROW <PH-" proxima-tensor/docs/discipline.md`
- **log-row title** `row numbers are assigned at land time from main, and here is the check that proves it`

### 0.14 — One bound plan: the fingerprint captured AFTER each driver's own rewrite, pre-registered to FAIL `[R16, crit MS-3, HC-2, f; B2 R15]`
- **tier** worker · **depends_on** 0.11, 0.8
- **worktree** `…/proxima-wt-risc14` · `gpu-risc/14-plan-fingerprint-red` · own target
- **opens** `omega/src/metal.rs:1004` `let mut resolved = bind(program, &shapes, &effective_outputs)?;`, `:1005-1012` (the comment explaining `layout_of` gets every packed stride wrong), **`:1013` `correct_packed_matmul_layouts(&mut resolved, &packed_operands.keys().copied().collect());`** — the Metal-only post-bind rewrite, imported at `metal.rs:191`, defined `proxima-tensor/src/bind.rs:1618-1647+`; `:1014` `bound_op_retirement`; `proxima-tensor/src/cpu.rs:358` `let resolved = bind::bind(program, &shapes, &effective_outputs)?;` — **no rewrite**; `proxima-tensor/src/bind.rs:1718-1722` (`bind` has **no backend parameter**), `:200-215` `BoundOp`, `:221-264` `BoundOpKind`, `:95-98` `Layout`, `:82` `MAX_INLINE_RANK`, `:377-446` `BoundOp::split` (its `out_scatter` arm is already backend-conditional), `:1594-1606` `layout_of`; `omega/src/msl.rs:4656` `emit_is_deterministic_byte_equal` (the emit-side twin this is the bind-side of); `generate.rs:855-861` `select_backend`.
- **commands** add `pub fn fingerprint(plan: &[BoundOp]) -> u64` (FNV-1a-64) over a canonical serialization: per op in order `node.0`, dtype discriminant, `extents`, kind discriminant; for `Elementwise` every `ComposedBody` step's `ScalarOp` discriminant + operand indices; for `Reduce` `element_body` steps, `reduce_op`, `init`, `keep`, `output_axes`, `out_layout.base` and `strides`, `out_scatter.is_some()` plus its `extent`/`element_stride`/`index_layout`; per operand `(source.0, Layout{base,strides}, Option<Lookup>{…})`; for `Constant` `value.to_bits()`. Then the test: bind the **real openchat cached-forward program** with real symbols and real blocks and capture the fingerprint **at the last point before execution on each path** — on Metal **after `metal.rs:1013`**, on CPU after `cpu.rs:358` — and `assert_eq!` them. Second case: fingerprint invariant under `metal-tiled-gemm` on/off. Third: determinism across two calls.
- **expect** N = 3 fingerprint tests, each printing both values so a mismatch names both sides; `ran_count == 0` is RED (the test is gated on `metal` **and** `cpu` together — the likeliest N==0 trap). **This card is PRE-REGISTERED TO FAIL on main:** the Metal fingerprint differs from the CPU fingerprint because `correct_packed_matmul_layouts` rewrites every packed Q4_K/Q5_K/Q6_K operand's `Layout.strides` and Metal alone applies it. **A GREEN result on main is the RED outcome here** — it would mean the fingerprint does not cover `Layout.strides` and the field list is wrong. The card lands the test `#[ignore]`d with an `// EXPECTED RED, see R16 / metal.rs:1013` marker and a companion always-green test asserting the **difference is exactly the packed operands' strides**.
- **predict (nano → micro)** `fingerprint` over a plan of `OPS_AFTER_PRUNE` ops costs **< 100 µs**, i.e. < 0.01% of R13's `prepare_ms` 1.97, so it can run unconditionally in debug builds. The diff between the two fingerprints is **exactly** the set of nodes in `packed_operands.keys()` and nothing else.
- **kill** the diff includes a node **not** in `packed_operands` ⇒ there is a **third** rewrite the plan has not found; that finding outranks every performance card and 7.6 is promoted to name it. A card that "fixes" the RED by loosening the fingerprint's field list has inverted the test; the field list is fixed here and is not negotiable downstream.
- **memory gate** the function folds over borrowed slices into a `u64`; an allocation-counter assertion over 1000 calls asserts zero. **KILL** at G8 for the harness run.
- **rollback** `git revert`; one pure function + one test file. **blast** `proxima-tensor/src/bind.rs` gains one function (no type, no field); `omega/tests/` gains one file. Zero hot path.
- **observe** the two printed fingerprints and the node-set diff (**NEW**, record site: the test itself, a compile-and-run observable, stated as such).
- **reprove** `cargo nextest run -p omega --features metal,cpu -E 'test(plan_fingerprint)'` and the same with `metal-tiled-gemm`.
- **log-row title** `brief item 1 is false on main today: the Metal driver rewrites the bound plan after bind, and here is the fingerprint that says by exactly how much`

---

# PHASE 1 — the route becomes a value

*Conflict 1 (round 1) stands: the census lands BEFORE any body swap. R13's 225/385/547 split is `classify_kind`'s substring output, and R12 ROW 263 measured that same instrument relabelling 9/601 → 225/385 when a body changed.*

### 1.1 — `omega::route::Route`, decided before emission, censused LOCK-FREE, cost-bounded `[S 1.1, B2 R08, crit RS-1, OB-1, a, R16]`
- **tier** worker · **depends_on** 0.14, 0.3
- **worktree** `…/proxima-wt-risc15` · `gpu-risc/15-kernel-route-value` · own target. Not `perf/route-census` or `perf/op-rule-census` (both checked out).
- **opens, in this order** (1) `omega/src/msl.rs:673-697` `emit`'s 4-kind + `Keep` match; (2) `:731-775` `kernel_cache_key` — pushes exactly **three** route characters, `'G'` if `tiled_gemm_block(..).is_some()`, else `'B'` if `packed_row_block(..).is_some()`, else `'S'`, with the comment at `:751-758` recording the ordering as load-bearing; (3) `:3164-3187` `push_cooperative_reduce_body`'s **third** independent re-derivation of the same two gates; (4) `:797` `kernel_dispatch_shape`, `:1517-1560` `grid_threads`, `:824` `reduce_is_cooperative`, `:1235` `packed_row_block`, `:1450` `tiled_gemm_block`, `:1487` `diagnose_packed_row_block`; (5) `omega/src/metal.rs:777-826` `classify_kind` + its confession, `:835-854` `diagnose_kind`, `:709-710` (both called in the op-timed path), `omega/examples/real_forward_packed_probe.rs`; (6) `proxima-tensor/src/instrument.rs:809-828` `WidthDeclineReason`, `:842` `WIDTH_TILE_DECLINE: Mutex<BTreeMap<…>>`, `:848-864` `record_width_tile_decline` — the `(NodeId, reason)` **shape** to mirror, **explicitly not its Mutex**; (7) `proxima-telemetry/src/metric/counter.rs:12-16,:46,:98` (`Counter` = one `AtomicU64`, 56 B); (8) `omega/src/metal.rs:2225-2248` — verified `counter!(ENCODE_DISPATCH_CALLS, 1)` at **`:2243`** inside `#[cfg(feature="instrument")]`, with `ENCODE_DISPATCH_TICKS` at `:2244-2246`.
- **the design, both questions answered** *Pipe question:* `Route` is a decision value computed once per `BoundOp` and consumed once by `emit` — no stages, no backpressure, no fan-in/out; `backend.rs:1-52` already adjudicated this boundary not-a-pipe on 2026-08-30. *Relocation question, call site both ways:* Way A `let route = route::of(bound, packed)?; match route { … }` — exhaustive, compiler-proved coverage, and `assert_eq!(route::of(bound), route_recorded(bound.node))` plus `assert_eq!(sum(ROUTE_DISPATCHES), ENCODE_DISPATCH_CALLS)` become **possible**. Way B (today) is four hand-ordered `if let` gates in three places plus a substring recovery that produces a label, not a count, and is MEASURED to mislabel. Those two assertions are new caller capability, not identical lines — **the type is earned.** *Why not extend `WidthDeclineReason`:* its eight variants are CPU width-tile stride/shape conditions naming nothing a GPU emitter decides; `Route` is its GPU-side sibling in `omega`, not an extension of it in `proxima-tensor`.
- **the census is LOCK-FREE and hot/cold split** [crit RS-1, R16]. The route is a **property of the plan, not of the dispatch**. **Cold, once per plan** at `plan_named` (`metal.rs:578-585`): a `Vec<Route>` of length `prepared.resolved.len()`, filled at plan time — this is the census of record, keyed `(NodeId, Route)`. **Hot, per dispatch** at `metal.rs:2243`: one `counter!(ROUTE_DISPATCHES[route as usize], 1)` on a fixed-size `[Counter; 10]` array — a single relaxed `AtomicU64::fetch_add`, the *same* operation already executing on that line, so the sum identity holds without a lock. **No `Mutex`, no `BTreeMap`, no allocation on the dispatch path.**
- **the cost bound, MEASURED not assumed** R13: `encode_dispatch_ms = 0.47` over 1196 dispatches = **393 ns/dispatch**. Budget: **≤ 5% = 19.6 ns/dispatch = 0.0235 ms/token**. The card runs census-ON vs census-OFF interleaved and asserts `encode_dispatch_ms_on − encode_dispatch_ms_off <= 0.0235` **and** `op_setup_ms` and `prepare_ms` deltas each `<= 0.0235`. **A census that costs more than 5% of the slice it measures rolls back** — which is why `ENCODE_DISPATCH_TICKS` (`metal.rs:2244`), not `gpu_exec_ms`, is the guard: `gpu_exec` is the device window (`metal.rs:547-554`) and cannot see a CPU-side cost [crit OB-1].
- **commands**
  ```
  # pub enum Route { Elementwise, Scan, Iota, Constant, ReduceSerial, ReduceCooperative,
  #                  ReduceRowBlockedPacked, ReduceTiledGemm, ScatterReduce, Declined(DeclineReason) }
  # pub fn of(&BoundOp, &PackedOperands) -> Result<Route, EmitError>
  cargo build -p omega --no-default-features --features alloc     # route.rs is alloc-tier, unconditional
  bash scripts/omega-gate.sh
  <G3 bench cell, features std,metal,instrument,metal-route-census>    # ON arm
  <G3 bench cell, features std,metal,instrument>                       # OFF arm, interleaved
  ```
  `emit`, `kernel_cache_key`, `kernel_dispatch_shape`/`grid_threads` and `push_cooperative_reduce_body` all consume `route::of`'s value; `classify_kind`'s substring buckets are **deleted** and `op_profile_bucket`'s `kind` is re-sourced from `Route`; `diagnose_kind` folds into `Route::Declined(reason)`; `real_forward_packed_probe.rs` is updated in the same commit. `classify_kind`'s 9th label (`reduce-unclassified`, from `Err(_)`) is accounted for by `route::of` returning `Result`: an undecidable route is an `EmitError`, not a census bucket.
- **expect** N1 `sum(ROUTE_DISPATCHES) == ENCODE_DISPATCH_CALLS` exactly, every step; inequality is RED. N2 `ENCODE_DISPATCH_CALLS / F == op_count == OPS_AFTER_PRUNE` (0.8's measured number). N3 the per-route counts against R13's buckets (225/385/547/37/2) — **a difference is not automatically RED, it is the finding**, because R12 ROW 263 proves the substring classifier mislabels; any difference is reported as "the census disagrees with `classify_kind` at N ops, here are their nodes". N4 ≥6 new tests: `route::of` total over all 4 kinds and both `Keep`s; its answer equals the branch `emit` takes, per route, on the real bound program; **`kernel_cache_key` byte-stability** — for every op the key through `route::of` is byte-identical to main's three-character logic (`ReduceTiledGemm`→`'G'`, `ReduceRowBlockedPacked`→`'B'`, all else→`'S'`, G-before-B preserved); **golden emitted source** — the emitted MSL for every op hashes identically before and after, extending `emit_is_deterministic_byte_equal` (`msl.rs:4656`) to a cross-commit golden; an unroutable op returns `EmitError`. N5 the cost bound above. N6 the alloc-tier build compiles `route.rs` and **states which modules it built** (§3's N==0 warning). **N==0 is RED.**
- **predict (milli → bench)** behaviour-neutral: golden hashes match for all `OPS_AFTER_PRUNE` ops; `step_wall_ms` with the census ON is within R13's 0.5% CoV of the OFF arm; `encode_dispatch_ms` delta ≤ 0.0235 ms.
- **kill** N5 exceeded ⇒ collapse to the cold per-plan vector only and re-measure; still over ⇒ the census does not go on the dispatch path at all and the row says so. Census sum ≠ `ENCODE_DISPATCH_CALLS`, or one golden hash drifts, or one `kernel_cache_key` differs ⇒ the route is still decided in more than one place; **do not proceed to Phase 2.**
- **memory gate** 10 `Counter`s = 560 B static; the cold vector is `prepared.resolved.len() × 1 B` ≈ 1.2 KB per plan, bounded by `plan_cache_len == 1` (R13). Slopes and caps identical to 0.3's seal; any increase is a NEGATIVE. **KILL.**
- **rollback** `git revert`. `route::of` is **not** feature-gated (a gated route means two routers); only the counters are, behind `metal-route-census`. The firewall is the `kernel_cache_key` stability test plus the golden, both **in this card**. Note: once 7.3–7.5 land on top, reverting 1.1 is a multi-commit unwind — revert those first.
- **blast** new `omega/src/route.rs`; `omega/src/msl.rs` (three decision sites collapse to one); `omega/src/metal.rs` (`classify_kind` deleted — every caller including `:709-710`; +10 counters + one line at `:2243`; the cold vector in `plan_named`); `omega/examples/real_forward_packed_probe.rs`; `proxima-model-interop/src/generate.rs` (`op_profile_bucket` sources `kind` from `Route`). `wgsl.rs`/`cuda.rs` untouched here.
- **observe** `ROUTE_DISPATCHES[10]` (**NEW**, record site `omega/src/metal.rs:2243`), the cold `(NodeId, Route)` vector (**NEW**, record site `plan_named` `metal.rs:578-585`, printed as `route_census step=N node=… route=…`), `ENCODE_DISPATCH_CALLS`, `ENCODE_DISPATCH_TICKS` (`metal.rs:2244`), `op_setup_ms`, `prepare_ms`.
- **reprove** the ON/OFF interleaved pair; `grep -o 'route=[A-Za-z]*' runs/cell-15-*.txt | sort | uniq -c` and assert the sum equals `encode_dispatch_calls`.
- **log-row title** `the route stops being a substring of the kernel it selected: one enum decided before emission, a lock-free census under a measured 5% budget, and the substring instrument deleted`

### 1.2 — The census-sum gate `[S 1.2]`
- **tier** hands · **depends_on** 1.1
- **worktree** `…/proxima-wt-risc16` · `gpu-risc/16-census-gate` · own target
- **opens** `scripts/omega-gate.sh` (6 steps; `[3/6]` asserts `ran_count` nonzero, `[2/6]` builds `--all-targets --all-features`)
- **commands** add a step asserting `sum(ROUTE_DISPATCHES) == ENCODE_DISPATCH_CALLS`, run under a **named** feature set (never `--all-features`, which additionally enables `cuda`, `wgpu-backend`, `metal-tiled-gemm`, `vulkan`, `npu`, `ane`).
- **expect** the sum matches exactly. **N==0 is RED and a mismatch is RED** — a dispatch with no route row means a path bypassed `route::of`.
- **predict (nano → micro)** on the openchat plan the census reports the R13 bucket shape (`ReduceRowBlockedPacked` ≈ 225, `ReduceCooperative` ≈ 385, `Elementwise` ≈ 547, `Constant`/`Iota` 37/2), now produced by a value rather than by grepping MSL.
- **kill** any op landing on `ReduceSerial` that R13 attributes to a Q4_K/Q5_K/Q6_K family — a quantized matvec on the serial path is a route bug worth more than any kernel tweak.
- **memory gate** N/A — a gate step; the underlying run is 1.1's. **rollback** revert the gate step. **blast** `scripts/omega-gate.sh`.
- **observe** the census table vs `ENCODE_DISPATCH_CALLS`.
- **reprove** `bash scripts/omega-gate.sh`
- **log-row title** `the census is a gate: a dispatch with no route row is RED`

### 1.3 — Every geometry constant into `omega-runtime.toml` `[S 1.3, B2 R16]`
- **tier** worker · **depends_on** 1.1
- **worktree** `…/proxima-wt-risc17` · `gpu-risc/17-geometry-config` · own target
- **opens (each a verified §12 violation)** `omega/src/msl.rs:1017` `const PACKED_ROWS_PER_GROUP: usize = 4`, `:1030` `const TILE_DIM: usize = 8`, `:1046` `const TILED_GEMM_NSG: usize = 4`, `:2516` `let lanes_per_block = 8;` (a **local** — cannot be overridden at all), `:2527` the loop step `SIMD_WIDTH/lanes_per_block`, `:3172/:3190/:3193` the bare `SIMD_WIDTH` literals pinning the cooperative width; the compliant surface `omega/build.rs:16` `require_nonzero`, `:35` `require_multiple_of_sixteen`, `:43` `require_divides_q4k_block`, `:59` `require_multiple_of_eight`, `:79` `resolve_int`, `:85` `rerun-if-env-changed`, `:105` `emit_sizing_consts`; `omega/omega-runtime.toml` (today only `[tiled_gemm]`); `omega/src/sized.rs:45` — `SIMD_WIDTH` **stays** a source const, documented as a hardware-family fact, never a policy knob.
- **commands** add `[packed_row_block] rows_per_group=4, lanes_per_block=8, tile_dim=8`, `[tiled_gemm] nsg=4`, `[cooperative_reduce] max_threads, vec_width` (3.1 consumes the last); route each through `resolve_int` + a cross-axis validator + `emit_sizing_consts`. Validators: `lanes_per_block` must divide `SIMD_WIDTH` (else `:2527`'s step is wrong); `rows_per_group` nonzero (a `div_ceil` denominator at `:1552`); `tile_dim` a multiple of 8. Each key's measurement record lives on the consuming const's doc in `sized.rs`, per that file's convention.
  ```
  grep -rnE '^ *(pub )?const [A-Z_]+ *: *(usize|u64|u32) *= *[0-9]' omega/src/ | grep -v sized.rs
  OMEGA_PACKED_ROW_BLOCK_ROWS_PER_GROUP=8 bash scripts/omega-gate.sh
  OMEGA_PACKED_ROW_BLOCK_LANES_PER_BLOCK=7 cargo build -p omega --features metal 2>&1 | grep -q 'must divide' && echo VALIDATOR_OK
  ```
- **expect** N1 the grep returns **only** GGUF codec wire-format constants (`Q4K_BLOCK_BYTES=144` `:294`, `Q4K_BLOCK_ELEMENTS=256` `:299`, Q5K/Q6K/Q8_0/Q4_0/F16/BF16), each gaining a one-line doc saying it is a wire-format fact; any surviving **policy** const is RED. N2 ≥8 tests: one per key asserting source == TOML, one env-override per key via `temp_env::with_vars`, and the override **visibly changes the emitted MSL** (compared against 1.1's golden) proving `rerun-if-env-changed` works and a cached build did not silently ignore it. N3 an invalid value fails at **build time** with the validator's message. **N==0 is RED.**
- **predict (nano → micro)** behaviour-neutral at default values: 1.1's golden hashes are unchanged for every route and `gpu_exec_ms` moves < 0.7% (inside R13's CoV).
- **kill** moving a const changes emitted MSL at the **default** value ⇒ an off-by-one; the golden catches it and the card stops. A value that cannot become a build-time const without becoming a runtime read stays a source const with a one-line why at the site, recorded as a **named exception** on the row, never silently.
- **memory gate** compile-time only; unchanged, stated rather than blank.
- **rollback** `git revert` build.rs + toml + const sites together; values identical by construction.
- **blast** `omega/build.rs`, `omega/omega-runtime.toml`, `omega/src/sized.rs`, `omega/src/msl.rs` const sites.
- **observe** the N1 grep (a source-level assertion runnable in CI); the generated `OUT_DIR/omega_sized.rs`; 1.1's golden hashes.
- **reprove** the three commands above.
- **log-row title** `every GPU geometry constant traces to omega-runtime.toml with its cross-axis validator; SIMD_WIDTH stays a hardware fact and the row says why`

---

# PHASE 2 — the Q4_K body (R13's largest measured mass: 225 ops / 44.450 ms)

### 2.1 — Recover candidate A, `metal-q4k-mask-fma` `[S 2.1, B2 R05-A]`
- **tier** worker · **depends_on** 1.2, 1.3, 0.6
- **worktree** `…/proxima-wt-risc18` · `gpu-risc/18-q4k-mask-fma` (base `gpu-risc/17-…`) · own target
- **opens** `omega/src/msl.rs:190-311` (`Q4K_UNPACK_MSL`), `:2452-2530` (`push_packed_row_blocked_body`, `lanes_per_block` at `:2516`, loop step `:2527`); incumbent `ggml-metal.metal:5086-5193` — mask **without** shift (`& 0x000F/0x0F00/0x00F0/0xF000`), branch-free `kmask1/2/3` `:5147-5150`, 1/16 and 1/256 folded into the scale at combine `:5171-5175` (R8).
- **commands** `git apply --3way docs/bench-campaigns/2026-09-03-gpu-one-risc/recovered/all-tracked.patch` (from 0.6, in git), then **strip to the one feature** — `proxima-wt-all` carries seven (R7). Declare `metal-q4k-mask-fma` default-off in `omega/Cargo.toml`, forward as `metal-q4k-mask-fma = ["omega?/metal-q4k-mask-fma"]` from `proxima-model-interop` [G5]. One green commit, one feature.
- **expect** `cargo nextest run -p omega --features metal,cpu,instrument,metal-q4k-mask-fma` green with an **explicit non-zero count on the row**; `omega/tests/q4k_real_checkpoint_parity.rs` runs ≥1 case with the feature **on** (a default-off feature is exactly the condition that hides tests — `omega/Cargo.toml`'s own `default` comment records 13 `metal_parity` cases compiling to zero). **N==0 is RED.**
- **predict (nano → micro)** `q4k_matvec_probe`'s Q4_K family time drops **≥15%** against 0.3's tree (R3/M3 is MEMORY, superseded by R13 for the whole-cell figure; the conservative floor is the aggregate, and the row says the anchor is MEMORY).
- **kill** parity vs `cpu::evaluate` exceeds the tolerance `q4k_real_checkpoint_parity.rs` already pins, or max-abs error > 1e-4 on a real weight tensor ⇒ §14, the body does not land at any speed.
- **memory gate** a kernel-body change allocates nothing; `device_allocated_bytes` and `phys_footprint_bytes` slopes must be identical to 0.3's seal; any device increase is a NEGATIVE and rolls back the arm even if it wins. **KILL.**
- **rollback** `git reset --hard`; default-off, so main's default build is untouched either way.
- **blast** `omega/src/msl.rs` MSL text under one cfg + two manifest lines. `kernel_cache_key` gains no new character (the feature does not change the route).
- **observe** the `Route::ReduceRowBlockedPacked` census count from 1.1 (this is what proves the swapped body ran); `q4k_macs` witness (`bind.rs:2856-2860`); the 9 `op_profile_family` rows with 0.1's true bytes.
- **reprove** the nextest command + `flock … cargo run --release -p omega --features metal,metal-q4k-mask-fma --example q4k_matvec_probe`
- **log-row title** *(shared with 2.2/2.3)* `two independent re-derivations of ggml's Q4_K body: candidate A recovered and rebased`

### 2.2 — Recover candidate B, `metal-q4k-pair-dot` `[S 2.2, B2 R05-B, crit RB-2]`
- **tier** worker · **depends_on** 2.1
- **worktree** `…/proxima-wt-risc19` · `gpu-risc/19-q4k-pair-dot` (base `gpu-risc/18-…`, so both features exist at one commit) · own target
- **opens** same as 2.1, plus `git log --oneline main..perf/q4k-independent-accumulators` to identify the paired-nibble commit (R12 ROW 257: GPU family 47.8 → 33.9 ms, −29%, parity 3.1e-6 vs f32 on real `blk.0.attn_q.weight`).
- **commands** `git cherry-pick -x <the q4k pair-nibble commit>` — **one** commit, not the 42. Do **not** bring `physical.rs`, `BoundOpKind::CachedAttention`, the bind.rs matcher, or the `libm` dep (0.7). Declare `metal-q4k-pair-dot` default-off + forward. **Reversibility note for the rollback map** [crit RB-2]: candidate B is a **commit on a branch**, so it is already in git and its deletion in 2.4 is reversible by cherry-pick; candidate A is an uncommitted diff, which is why 0.6 exists. The two are **not** symmetric and the rollback map says so.
- **expect** as 2.1 under `metal-q4k-pair-dot`; both features present in the manifest at this commit, each buildable alone, neither in `default`, and **the G5 precedence rule compiles exactly one body under `--all-features`** so `omega-gate.sh [2/6]` stays green. **N==0 is RED.**
- **predict (nano → micro)** `q4k_matvec_probe` Q4_K family time drops **≥20%** (R12 ROW 257's −29% is the anchor; the floor is conservative and is MEASURED, not MEMORY).
- **kill** same parity kill as 2.1. **memory gate** as 2.1, **KILL**.
- **rollback / blast / observe / reprove** as 2.1 with the other feature name.
- **log-row title** *(shared)* `candidate B recovered: the paired-nibble body, cherry-picked without the macro-op it shipped beside`

### 2.3 — The bake-off: ONE body, by a tie-break written before the numbers and terminal `[S 2.3, B2 R05, crit d, HC-3]`
- **tier** judge · **depends_on** 2.2, 0.3
- **worktree** `…/proxima-wt-risc20` · `gpu-risc/20-q4k-body-bakeoff` (base `gpu-risc/19-…`) · own target
- **opens** `omega/src/msl.rs:190-311`, `:2452-2530`; `:1978-1982` — **verified**: `Q4K_UNPACK_MSL`, `Q5K_UNPACK_MSL`, `Q6K_UNPACK_MSL` are pushed into one source string with **no delimiter**, which is why a "grep the emitted Q4_K region" tie-break is undecidable [crit HC-3, d]; incumbent geometry `<4,2,32>` at `ggml-metal.m:3330`, `:3215-3220` (R8).
- **commands** interleaved, never before-block/after-block, **explicit feature sets, never `--all-features`** [G5]:
  ```
  for i in 1 2 3 4 5; do
    <G3 MILLI> --features std,metal,instrument,metal-q4k-mask-fma
    <G3 MILLI> --features std,metal,instrument,metal-q4k-pair-dot
    <G3 MILLI> --features std,metal,instrument                     # the R13-shape control arm
  done
  ```
- **the decision rule, pre-registered, four rungs, terminal — every rung decidable without reading emitted text** [crit d]
  1. **Parity first.** Any body whose max-abs error vs `cpu::evaluate` on real `blk.0.attn_q.weight` bytes exceeds **1e-4** is out (§14). Both parity suites green under **each** feature separately: `q4k_real_checkpoint_parity`, `q4k_matmul_layout`, `q4k_unpack`, `metal_parity`, `backend_parity`, `metal_real_forward`.
  2. **Route-count pin.** The comparison is **void** unless both arms report the same `Route::ReduceRowBlockedPacked` census count (R13: 225). Unequal counts mean different op sets, not different bodies — and a body change that moves the op count changed the **route**, which is RED.
  3. **Primary metric.** Lower summed `op_profile_family` `gpu_ms` over the seven weight families `{ffn_up, ffn_gate, ffn_down, attn_q, attn_k, attn_v, attn_output}`, with 0.1's true bytes. If the medians differ by more than **2× the pooled CoV** (R13 `gpu_exec_ms` CoV 0.7% ⇒ threshold 1.4%; use the measured pooled value), the faster body wins.
  4. **Tie → lower max-abs parity error** on the same real tensor (Arm B has a recorded 3.1e-6; Arm A must produce its own number on the same tensor). **Still tied → fewer lines in the Rust source region `push_packed_row_blocked_body` plus the feature's own `const` block**, measured with `awk` over rustfmt function boundaries in `omega/src/msl.rs` — a file region with defined start and end, **not** the concatenated emitted source, which has no Q4_K boundary [crit HC-3]. **Still tied → terminal: Arm B wins**, because B is a commit on `4be2f3a` and A is an unrebased diff off `2b95210` conflicting with nine commits (R7); lower landing risk breaks the last tie. **No rung can fail to decide.**
  5. **The loser is not silently dropped:** it lands as a **negative row** with its measured number and the rung that decided it, and its feature is deleted in 2.4's commit. Arm A's patch survives in git from 0.6; Arm B's commit survives on its branch.
- **expect** 3 arms × 5 runs, each emitting S steady rows; 9 `op_profile_family` rows per run (N < 9 is RED); parity suites green per feature with explicit counts. **N==0 is RED under any arm.**
- **predict (milli → bench)** the winner's `ReduceRowBlockedPacked` bucket falls from **44.450** to **≤ 33.0 ms** (band [30, 33], from B's −29% conservatively applied); therefore `gpu_exec_ms` **56.93 → [45.0, 47.0]** and `step_wall_ms` **67.92 → [56.0, 58.0]** = **3.20–3.31x** against 17.52 (or against 0.4's arm B if 0.4's kill fired).
- **kill** neither body clears −10% on the packed-row-blocked bucket beyond both CoV bands ⇒ the −17.2%/−29% micro figures do not transfer to the real graph; decompose — **inconsistency** if the family bucket moved but wall did not (orchestration is absorbing it; 4.1/4.2 own that), **understanding-gap** if the family bucket itself missed (the work item is "what else was in R12's feature-off control", which R12 records **already carried the paired body**). Stop the climb; do not proceed to 2.4.
- **memory gate** both arms' device and RSS slopes identical to 0.3's; any increase is a NEGATIVE and rolls back the winning arm. **KILL.**
- **rollback** both features default-off; drop the branch. **blast** `omega/src/msl.rs`; two manifest entries; the G5 precedence rule.
- **observe** per-route `gpu_exec` ticks for `ReduceRowBlockedPacked`; the per-family split (R13: ffn_up 10.862 / ffn_gate 10.843 / ffn_down 10.005 / attn_q 5.172 / attn_output 3.846 / attn_v 1.523 / attn_k 1.462 / output.weight 0.737) with 0.1's corrected bytes; `q4k_macs`; per-op mode's **+7.3%** inflation quoted beside every per-op number.
- **reprove** the interleaved loop above.
- **log-row titles** `one Q4_K body, chosen by a tie-break written before the numbers` · `NEGATIVE: the losing Q4_K body, its number, and the rung that decided it`

### 2.4 — Promote the winner to `default`, delete the loser `[S 2.4]`
- **tier** worker · **depends_on** 2.3
- **worktree** `…/proxima-wt-risc21` · `gpu-risc/21-q4k-body-default` · own target
- **opens** `omega/Cargo.toml [features] default = ["std","metal","cpu"]`; `proxima-model-interop/Cargo.toml [features]`
- **commands** rename the winner's feature to the **mechanism** (`metal-q4k-fold-scale`, not the author or worktree), move it into `default` in one commit gated on the full parity suite; delete the loser's feature, body and the G5 precedence rule in the same commit.
- **expect** `bash scripts/omega-gate.sh` green all six steps including `[1/6]` `--no-default-features --features alloc` and `[2/6]` `--all-targets --all-features`; `bash scripts/proxima-tensor-gate.sh` green. **N==0 is RED.**
- **predict (milli → bench)** whole-cell `step_wall_ms` **67.92 → [56.0, 58.0]**, ratio **3.88x → 3.20–3.31x**, carrying 2.3's measured delta forward, not re-derived from theory.
- **kill** the bench move is less than half 2.3's milli prediction ⇒ decompose. The inconsistency branch (GPU fell, wall did not) **promotes** Phase 4 rather than killing it, since R13 puts 11.0 ms in orchestration; the row says so.
- **memory gate** unchanged from 2.3's winning arm; slopes and caps re-asserted at `default`. **KILL.**
- **rollback** demote out of `default` (one line), then `git revert`.
- **blast** `omega/Cargo.toml`, `omega/src/msl.rs`; every downstream crate turning on `omega/metal` inherits it.
- **observe** `gpu_exec_ticks`, `step_wall_ms`, and **`encode_dispatch_calls` unchanged at `OPS_AFTER_PRUNE`** (0.8's measured number) — a body change that moves the dispatch count means the route changed and the arms are not comparable [crit OB-3].
- **reprove** `bash scripts/gpu-seal.sh $PWD $CARGO_TARGET_DIR std,metal,instrument 5`
- **log-row title** `the Q4_K body lands in default: R13's 44.450 ms of packed-row-blocked time, re-measured`

---

# PHASE 3 — the non-matmul GPU bucket (R13: 16.6 ms)

### 3.1 — Wide cooperative reduce, width from the sizing config `[S 3.1, B2 R06]`
- **tier** worker · **depends_on** 2.4, 1.3
- **worktree** `…/proxima-wt-risc22` · `gpu-risc/22-wide-cooperative-reduce` · own target. Not `perf/gpu-dispatch-count` (checked out).
- **opens** `omega/src/msl.rs:1517-1560` `grid_threads` — verified: the cooperative arm is `output_total * SIMD_WIDTH`, i.e. 32 threads per output element for **every** reduction length; `:3188-3196` (`output_index = gid/SIMD_WIDTH`, `lane = gid % SIMD_WIDTHu` — the pin); `:824` `reduce_is_cooperative`; `omega/src/sized.rs:45` (`SIMD_WIDTH` stays a hardware fact — the reduce width is a **new policy const**, not an override of it); incumbent `ggml-metal.m:3797-3804` (nth doubles from 32 to `min(ne00/4, maxTotalThreadsPerThreadgroup)`), `ggml-metal.metal:1679-1721` (float4 loads, `simd_sum` → threadgroup shmem → `simd_sum`; a 4096-wide row gets 1024 threads) (R8); `metal.rs:1402-1426` `pipeline_for` (the device's own `maxTotalThreadsPerThreadgroup`).
- **commands** **first, the mechanism re-read** [crit O-5 analogue]: run the G3 milli cell on 2.4's tree and record the **current** `Route::ReduceCooperative` time — R13's 385 ops / 9.113 ms is a pre-Phase-2 figure and the body swap may have moved which ops route there. Then implement a two-level tree: threads-per-output = `min(next_pow2(reduction_len / vec_width), COOPERATIVE_REDUCE_MAX_THREADS)`, floored at `SIMD_WIDTH`, **clamped against the pipeline's own `maxTotalThreadsPerThreadgroup`, never against a constant**. Both consts come from 1.3's `[cooperative_reduce]`. Declare `metal-wide-reduce` default-off + forward. Recover `docs/bench-campaigns/.../recovered/gpudisp-tracked.patch` as reference only.
- **expect** `op_profile_bucket kind=reduce-cooperative op_count == 385` in both arms — **the count must not move, only the time**; a moved count is RED (the route changed, not the geometry). `omega/tests/metal_parity.rs` runs its full case set with the feature on, count recorded explicitly. ≥3 new tests (narrow row, wide row, non-power-of-two row). The `OMEGA_COOPERATIVE_REDUCE_MAX_THREADS` override must **visibly change the emitted MSL** at 32 vs 1024 (compared against 1.1's golden). **N==0 is RED.**
- **predict (milli → bench)** `reduce-cooperative` falls from the figure this card just re-measured toward **−20%** (R3/M4, MEMORY, superseded by R13 for the bucket total; band [7.0, 7.6] ms against R13's 9.113); combined with 2.4, `gpu_exec_ms` lands **[43.5, 45.5]** and `step_wall_ms` **[54.5, 56.5]** = **3.11–3.22x**.
- **kill** `metal_parity` or `backend_parity` regresses ⇒ a wider tree changes float summation order and §14 binds on the oracle, not on speed. Or the cooperative share does not fall beyond the measured CoV ⇒ record the negative and do not promote. Or the device clamp makes the width identical to 32 for our shapes ⇒ the lever is dead; record it.
- **memory gate** a wider threadgroup uses **threadgroup** (on-chip) memory, not device memory; `device_allocated_bytes` unchanged in slope and absolute vs 0.3's seal; an increase is a NEGATIVE. **KILL.**
- **rollback** feature default-off; `git revert`.
- **blast** `omega/src/msl.rs` cooperative body + `grid_threads` cooperative arm only — the tiled-GEMM and packed-row-block paths `return` before it (`:3164-3187`) and are untouched; `omega/src/sized.rs` +2 consts; `omega-runtime.toml` +1 section; `omega/build.rs` +2 keys.
- **observe** `Route::ReduceCooperative` count and tick share (1.1); `op_profile_bucket … gpu_ms` and `gpu_ns_per_op` (R13: 9.113 ms / 23,670 ns per op); `gpu_exec_ticks`; the +7.3% per-op inflation quoted.
- **reprove** the ON/OFF interleaved milli pair.
- **log-row title** `SIMD_WIDTH is a lane count, not a thread budget: the cooperative reduce gets its width from the sizing config and its clamp from the device`

### 3.2 — Elementwise bucket census `[S 3.2, crit OB-4, j — retiered hands → worker]`
- **tier** worker · **depends_on** 3.1
- **worktree** `…/proxima-wt-risc23` · `gpu-risc/23-elementwise-census` · own target
- **opens** the **recorder**, not the emitter [crit OB-4]: `omega/src/metal.rs:578-585` `plan_named` (1.1's cold census site — this is where the extra fields are recorded) and `:2243` (the hot counter); `omega/src/msl.rs:2104-2177` `render_elementwise` for context only; R13's bucket: **547 elementwise / 7.350 ms** (13,437 ns/op) plus 37 constant / 2 iota at 0.161 / 0.008 ms.
- **commands** extend 1.1's **cold per-plan** census rows with `(NodeId, Route::Elementwise, extents, operand_count)` — cold, once per plan, so the hot path is untouched and 1.1's 5% budget is not re-spent — and rank by tick share.
- **expect** ≥1 row per elementwise node; the count equals the `Elementwise` share of `ENCODE_DISPATCH_CALLS` (R13: 547). **N==0 is RED.**
- **predict (nano → micro)** the top-5 elementwise nodes by tick share account for **≥50%** of the 7.350 ms bucket — the bucket is concentrated, not uniform.
- **kill** the bucket is uniform across >200 nodes ⇒ no single-node lever exists; the only remaining lever is *fewer nodes*, which is Phase 6, and this card closes with that pointer. (Standing reason not to reach for fusion: `grep -rln fuse ggml/src` is **empty** at `b25346221` — parity is reachable without a fusion engine, R8.)
- **memory gate** the cold vector grows by 3 fields × plan length ≈ 15 KB, bounded by `plan_cache_len == 1`; slopes unchanged. **KILL.**
- **rollback** `git revert`; instrument-gated only. **blast** the cold census recorder only.
- **observe** the census itself; `gpu_exec` ticks per elementwise node.
- **reprove** the G3 milli cell + the per-route table.
- **log-row title** `is the 7.35 ms elementwise bucket concentrated or uniform, and what that decides`

### 3.3 — Re-seal after the GPU-side work `[S 3.3, crit O-1]`
- **tier** hands · **depends_on** 3.1, 0.4, 0.5
- **worktree** `…/proxima-wt-risc24` · `gpu-risc/24-reseal-kernels` · own target
- **opens** `scripts/gpu-seal.sh` (0.2)
- **commands** `bash scripts/gpu-seal.sh $PWD $CARGO_TARGET_DIR std,metal,instrument 5` with the winning Q4_K body and (if 3.1 cleared) the wide reduce in `default`; 5 interleaved pairs; **both** incumbent arms, which exist since Phase 0 — this card carries **no conditional** [crit O-1].
- **expect** 5 ours-runs × S rows; both incumbent arms present; every board cell filled or explicitly `FEATURE GAP` with its reason. **N==0 is RED; a blank cell is RED.**
- **predict (milli → bench)** `step_wall_ms` **[54.5, 56.5]**, `gpu_exec_ms` **[43.5, 45.5]**, ratio **3.11–3.22x** against 0.4's chosen incumbent arm, plus the fraction-of-ceiling against 0.5's measured `read_only_gbs` column, named. Derivation: R13 `gpu_exec` 56.93 minus 2.3's measured Q4_K delta minus 3.1's measured cooperative delta; orchestration (R13: 11.0) untouched and is Phase 4's premise.
- **kill** `step_wall_ms` does not fall below R13's 67.92 by more than both CoV bands ⇒ the two GPU-side wins are being eaten by CPU orchestration; record that, do not promote further, go straight to Phase 4.
- **memory gate** full G8 assertion at `capacity_tokens = 0`; a breach rolls back the promoted features regardless of the timing. **KILL.**
- **rollback** demote features out of `default`, one line each. **blast** docs + `omega/Cargo.toml [features] default`.
- **observe** `step_wall_ms`, `gpu_exec_ms`, `op_setup_ms`, `prepare_ms`, `block_upload_ms` (0.1's three split counters), `encode_dispatch_calls`, `plan_hits`, per-route census, `phys_footprint_bytes`, `device_allocated_bytes`, CoV per arm.
- **reprove** the seal command above.
- **log-row title** `the board after the kernel bodies, against both incumbent arms and the measured ceiling: what is left is the orchestration`

---

# PHASE 4 — plan stability and orchestration (R13: 11.0 ms; root cause at `generate.rs:966`)

### 4.1 — `cached_len` as a capacity bucket, with the symbol-1 tail mask AND its scalar leaf fully wired `[S 4.1, B2 R09, crit MS-1, e, RB-3]`
- **tier** worker · **depends_on** 3.3, 0.12
- **worktree** `…/proxima-wt-risc25` · `gpu-risc/25-kv-capacity-bucket` · own target
- **opens, in this order** (1) `proxima-model-interop/src/generate.rs:958-977` `resolve_plan`, key `(symbols[0], symbols[1])` at `:966`, `plan_hits += 1` `:968`, `self.plans.clear()` `:973`; (2) `:1391` `let symbols = [new_count as u64, cached_len as u64];`; (3) `proxima-tensor/src/spec.rs:6216-6245` — `k_even_cache`/`k_odd_cache`/`v_cache` each `input_leaf([Extent::Symbolic(1), Static(kv_heads), Static(pairs|head_dim)])`; **symbol 1 is `cached_len`**; (4) `spec.rs:823-845` `causal_mask` — verified: two `Op::Iota{extent: Symbolic(0)}`, `ScalarOp::Greater` with `"t->st"`/`"s->st"`, `scalar_constant(NEG_INFINITY)` broadcast `"->stug"`, consumed as `(is_future, "sw->swug")` at `:2610` — **the construction to copy, verbatim in shape**; (5) `spec.rs:2303-2319` — the doc naming the premise ("the masking-only-within-`s,w` asymmetry is what makes this correct **without a `cached_len` scalar**"), corrected in this commit; (6) **`generate.rs:1304-1309` `build_position_inputs(&next_ids, cached_len, head_dim, rope_freq_base)`** and **`:1313-1319` `Vec::with_capacity(owned + packed + packed_owned + 3 + layer_caches.len()*3)`** with `"ids"` `:1322`, `"eps"`/`"rope_cos"`/`"rope_sin"` `:1332-1334` — verified; (7) `omega/src/metal.rs:1111-1118` `block_node_ids` (filters `Op::Input` by **program order**), `:984-990` and `proxima-tensor/src/cpu.rs:337-343` `InputCountMismatch`; (8) `proxima-model-interop/src/bind.rs:3053-3059` — the two assertions this card inverts.
- **the change, stated honestly** This is **not** "zero IR change". Bucketing makes the cached extent `bucket >= cached_len`, so the cached block **does** now need masking over **symbol 1**, against a runtime `cached_len` the graph deliberately does not carry. The card adds, in the graph, **four nodes per attention block**: one `Op::Iota { extent: Extent::Symbolic(1) }`, one `Op::Input` scalar leaf `"cached_len"` (`[Extent::Static(1)]`), a `GreaterEqual` elementwise, and the `Select` against the existing `NEG_INFINITY` `scalar_constant`. **Zero new `Op` variants, zero new `BoundOpKind` variants, zero new `IndexMap` variants** (0.11's exhaustive matches prove it mechanically). It is a graph change, not a driver fix, and the row leads with that.
- **the scalar leaf is wired, not assumed** [crit MS-1, e]: an `Op::Input` **is** a block node (`block_node_ids` filters exactly `matches!(expr, Op::Input{..})`) and both drivers hard-fail on count mismatch. Therefore, in this same commit: `build_position_inputs` (`generate.rs:1304-1309`) gains a **fifth output field** owning the backing storage beside `ids_f32`/`epsilon`/`cos`/`sin`; `named_blocks` gains `named_blocks.push(("cached_len", QuantizedBlock::Float32(inputs.cached_len.as_slice())))` beside `"eps"` at `:1332`; and the **`+ 3` capacity literal at `:1316` becomes `+ 4`**. A test asserts `named_blocks.len() == block_node_ids(program).len()` on the real program.
- **the rollback fork is decided here, not deferred** [crit RB-3]: `KV_BUCKET_TOKENS` is a **build-time const** from a new `[kv_cache] bucket_tokens = 256, capacity_tokens = 4096` section in `proxima-tensor-runtime.toml` via that crate's existing `build.rs` `resolve_int`/`emit_sizing_consts` (§12), plus a new `require_at_most(65536)` [G8's 34 GB trap]. **Therefore bucket=1 is *not* a runtime identity and rollback is `--features` omission plus a rebuild** — stated on the card, not left to the row. 256 is not arbitrary: it ports the incumbent's own `n_kv` padding (R8/M6′).
- **the harness assertion is inverted in this same commit, `#[cfg]`-paired** so both arms compile and both pass and every commit is a green bisect point:
  ```rust
  #[cfg(feature = "kv-bucketed-cache")]
  { assert_eq!(runtime.plan_hits + runtime.plan_misses, forward_calls_taken);
    assert!(runtime.plan_misses >= 1);
    assert_eq!(runtime.plan_hits, forward_calls_taken - distinct_shape_count()); }
  #[cfg(not(feature = "kv-bucketed-cache"))]
  { assert_eq!(runtime.plan_hits, 0, "cached_len grows every decode step, …"); }
  ```
- **commands** declare `kv-bucketed-cache` in **`proxima-tensor/Cargo.toml`**, forward as `kv-bucketed-cache = ["proxima-tensor/kv-bucketed-cache"]` in **both** `omega` and `proxima-model-interop`. Set `let bucket = (cached_len + new_count).div_ceil(KV_BUCKET_TOKENS) * KV_BUCKET_TOKENS; let symbols = [new_count as u64, bucket as u64];` at `:1391`. Run the CPU oracle **first**: `cargo nextest run -p proxima-model-interop --release --features std --lib --run-ignored all -E 'test(runs_one_real_forward_pass_and_greedy_picks_a_real_token)'`. Then both crate gates, then the G3 bench cell.
- **expect** N1 `generated.0[0] == 2651` and `generated.1 == "known"` (`bind.rs:2797-2803`) still hold; `generated_text == "Here is a simple Python function that returns"`. N2 **`plan_hits == F − distinct_shape_count()`; at budget 8 with prefill `(31, 256)` and decode `(1, 256)` the distinct count is 2, so `plan_hits == 6` and `plan_misses == 2`** — and the `symbols` dump from 0.12 **observes** the distinct count rather than assuming the prompt is 31 tokens. `plan_hits == 0` is now RED. N3 `plan_cache_len == 1` still. N4 `named_blocks.len() == block_node_ids(program).len()` on the real program, and `InputCountMismatch` never fires. N5 the route census shows **no new route** — the four mask nodes are `Iota`, `Constant`, `Elementwise`, `Elementwise`; a new route is RED. N6 ≥6 more tests: bucketed and unbucketed produce identical token ids over the full budget; the tail mask covers exactly `bucket − actual` positions; bucket boundary (`cached_len == bucket`); bucket+1 forces exactly one new plan; **feature-OFF program identity** — 0.14's fingerprint with the feature off is byte-identical to main's (the "off is main" claim is gated by a hash, not asserted); CPU parity bucketed == unbucketed to **0 ULP** (masked lanes contribute `exp(-inf) = 0`, so this is exact, not approximate). **N==0 is RED.**
- **predict (milli → bench)** `prepare_ms` falls from R13's **1.97** to **≤ 0.30** on hit tokens and orchestration 11.0 → **[9.0, 9.3]**; **simultaneously the cache axis grows from ~37 to 256, so R13's `kv_cache.v`/`k_odd`/`k_even` reduces (1.559 / 0.783 / 0.774 ms) grow up to ~7×, adding [15, 25] ms of GPU work. Net: `step_wall_ms` gets WORSE, into [63, 72].** This card is **pre-registered as a timing loss** whose payoff arrives only when 5.3 makes the cache device-resident and 6.2 collapses the two ranges. Saying so before measuring is the point: a card that predicts a win and delivers a loss kills the climb; this one predicts the loss, and the kill below is written against the *prediction*, not against R13.
- **kill** token ids drift at any bucket size ⇒ the tail mask is wrong; revert immediately (§14). Any ULP difference on the CPU parity test ⇒ the card dies on correctness before any timing is read. `plan_hits` does not reach `F − distinct_shape_count()` ⇒ the key is still moving; dump `symbols` per step and name the other varying symbol. `plan_hits` rises but `prepare_ms` does not fall beyond CoV ⇒ the cost was never in `plan_named`; decompose and re-instrument before touching 4.2. `step_wall_ms` lands outside **[63, 72]** ⇒ the climb is killed and decomposed even though the card predicted a loss.
- **memory gate — the most memory-relevant change in the plan** [crit l]. Bucketing grows the KV upload from `cached_len × 262_144` to `256 × 262_144 = 67_108_864 B` — a **known, bounded, pre-registered increase**: `kv_cache_upload_bytes` becomes **constant at 67.1 MB/token** instead of R13's linear 8.13 → 9.70 MB. Flat is the point (slope → 0); the absolute is higher and the row **headlines that trade**. `DEVICE_CAP_BYTES` at `capacity_tokens = 256` = **4_247_525_888**; **exceeding it rolls the card back regardless of `plan_hits`.** RSS caps unchanged. `require_at_most(65536)` in `build.rs` makes a `context_length`-style runaway a **build-time** failure — where the 34 GB allocation should have been caught. **KILL.**
- **rollback** `git revert`; feature default-off **plus a rebuild** (the bucket is a build-time const, stated above); the `#[cfg]`-paired assertion keeps the OFF arm green; test (feature-OFF fingerprint identity) is what makes "off is main" checkable.
- **blast** `proxima-tensor/src/spec.rs` (`append_mistral_cached_layer` `:2336-2865` gains 4 nodes; the mask helper sits beside `causal_mask` `:823`; the `:2303-2319` doc corrected), `proxima-tensor-runtime.toml` +1 section, `proxima-tensor/build.rs` +2 keys + `require_at_most`, `proxima-model-interop/src/generate.rs` (`build_position_inputs` `:1304-1309`, `named_blocks` `:1313-1334` incl. the `+3`→`+4` literal, symbols `:1391`), `proxima-model-interop/src/bind.rs:3053-3059`. **`append_qwen35_*` builders are NOT touched** — the feature scopes to the mistral cached layer, so the qwen3.5 hybrid path (`0c3bd4f`) is unaffected, and a test asserts it. **CPU and Metal both**, because the graph changes; the CPU oracle runs first.
- **observe** `plan_hits`, `plan_misses`, `plan_cache_len` (`generate.rs:1732`, `:1764`), the `symbols` dump (0.12), `prepare_ms`, `PREPARE_CALLS`/`PREPARE_TICKS`, `RESIDENT_BUFFER_REUSES` (`metal.rs:1983`), `kv_cache_upload_bytes` (`generate.rs:1657`), `device_allocated_bytes`, `phys_footprint_bytes`, the new symbol-1 mask nodes in the census.
- **reprove** the G3 bench cell ON/OFF interleaved 3×, plus the CPU 0-ULP parity test.
- **log-row title** `plan_hits=0 was the shape symbol, not the cache: cached_len becomes a capacity bucket, the tail gets the symbol-1 mask the design was built to avoid, and the card is pre-registered as a timing loss`

### 4.2 — The plan-stable device arena, with its peak-memory sum computed BEFORE it allocates `[B2 R10, S 4.2, crit RS-3, l]`
- **tier** worker · **depends_on** 4.1
- **worktree** `…/proxima-wt-risc26` · `gpu-risc/26-plan-stable-arena` · own target
- **opens** `omega/src/metal.rs:2179-2252` `encode_op` — verified: `:2193-2194` `kernel_cache_key` + `kernel_dispatch_shape` **per op per token even on a pipeline-cache hit**, `:2210` `allocate_buffer(device, bound_output_len(bound), bound.dtype)`, `:2211` `upload_uniforms(device, &pack_uniforms(bound))`, `:2249` `device_buffers.insert(bound.node, (output, 0))`; **`:2068-2078` `UNIFORM_BUFFER_REUSES` already exists with a live reuse path at `:2075`** — read its current hit rate and report it **before** assuming uniforms are the cost; `:449-568` `execute_plan` (one encoder, one commit, one wait; **retires per position**); **`:1014` `bound_op_retirement(&resolved, &effective_outputs)`** — the owner of retirement order, named here because a pooled output that is still an operand of a later op is this card's hazard [crit RS-3]; `:1402-1426` `pipeline_for`.
- **commands** hang a `BufferArena` off the cached `Plan`: a `Vec<MetalBuffer>` indexed by **resolved position**, allocated once when the plan is built, bound (not allocated) in `encode_op`; one uniform buffer per position written in place (the `:2069-2078` mechanism already proves the shape is legal); retirement becomes a no-op on arena-owned buffers. **Strict O(1) per op in steady state: index into a `Vec`, no allocation, no hashing.** Recover `all-tracked.patch`'s `metal-buffer-pool` hunk (0.6) as reference and rewrite against 4.1's plan-stable `Plan`. Declare `metal-plan-stable-buffers` default-off + forward.
- **expect** N1 a new counter `OUTPUT_BUFFER_ALLOCATIONS` equals `op_count` on the first step and **0 on every steady step**; a nonzero steady value is RED **and names the position that reallocated**. N2 `UNIFORM_BUFFER_REUSES` rises to `op_count` per steady step. N3 `plan_hits == F − distinct_shape_count()` still (this card is a no-op without 4.1). N4 `generated_text` unchanged. N5 ≥5 tests: pooled and unpooled `execute_plan_named` produce identical outputs on the real forward; **a pooled output that is still an operand of a later op is not reused before its last consumer** — asserted against `bound_op_retirement`'s own order, on the real program, which is the use-after-retire hazard test [crit RS-3]; an extent change forces a documented realloc; uniform contents change per step while the buffer does not; the arena's live buffer count is bounded in steady state. **N==0 is RED.**
- **predict (milli → bench)** `op_setup_ms` **3.9 → [0.4, 0.8]** (what remains is `kernel_cache_key` + `pipeline_for` + bind; `pipeline_lookup` is already 0.04); `newBufferWithLength` calls/token ~1196 → ~0 in steady state; orchestration 11.0 → **[5.5, 6.2]**; therefore `step_wall_ms` improves by **[3.1, 3.5] ms** from wherever 4.1 left it.
- **kill** `OUTPUT_BUFFER_ALLOCATIONS` nonzero in steady state ⇒ the plan is not stable; the work item goes **back to 4.1**, not forward. `op_setup_ms` falls but `step_wall_ms` does not, beyond both CoV bands ⇒ the orchestration slice overlaps GPU execution and removing it does not shorten the token — the same shape as R12's dispatch-count null; record it and **stop Phase 4** rather than continuing to 4.3's promotion.
- **memory gate — the highest-risk memory card in the plan** [crit l]. The arena holds every intermediate output alive for the plan's lifetime instead of retiring per position. **The card's FIRST action, before allocating anything, is to compute and print `sum over all resolved ops of product(extents) * dtype_bytes`** and compare it against G8's activation term, `DEVICE_CAP_BYTES − 4_140_417_024 − 256*262_144 = 40_000_000`. If the naive arena exceeds 40 MB it **must** be liveness-partitioned — buffers shared across positions whose live ranges (from `bound_op_retirement`) do not overlap, computed once per plan, cold. The row records the arena's peak bytes and its reuse factor. **`device_allocated_bytes` above `DEVICE_CAP_BYTES` rolls the card back regardless of the `op_setup` win.** **KILL.**
- **rollback** feature default-off (`get_or_allocate` is `allocate_buffer` with the feature off); `git revert`.
- **blast** `omega/src/metal.rs` `encode_op` (two call sites), `Plan` (+1 field), `execute_plan`'s retire loop. No IR change, no graph change, no emitter change, no other backend.
- **observe** `OP_SETUP_CALLS`/`OP_SETUP_TICKS`, `UNIFORM_BUFFER_REUSES` (`metal.rs:2069`) **before and after**, `OUTPUT_BUFFER_ALLOCATIONS` (**NEW**, record site `metal.rs:2210`), `ARENA_PEAK_BYTES` (**NEW**, record site: the arena constructor in `plan_named`), `device_allocated_bytes`, `phys_footprint_bytes`.
- **reprove** the G3 bench cell ON/OFF interleaved 3×; the row's claim is N1 plus the arena peak-bytes number.
- **log-row title** `1196 device buffers and 1196 uniform uploads per token become zero: the arena hangs off the plan, liveness-partitioned, with its peak bytes computed before it allocated`

### 4.3 — The board re-seal: the plan's one board-level prediction `[S 4.3, crit b, O-2]`
- **tier** hands · **depends_on** 4.2, 0.4, 0.5, 3.3
- **worktree** `…/proxima-wt-risc27` · `gpu-risc/27-board-reseal` · own target
- **commands** `bash scripts/gpu-seal.sh $PWD $CARGO_TARGET_DIR std,metal,instrument 5`, 5 interleaved pairs, **both** incumbent arms (0.4) and the measured ceiling (0.5) — both landed in Phase 0, so this prediction is made against a settled denominator and an existing ceiling [crit b, O-2].
- **expect** 5 ours-runs × S rows; llama arm A and arm B present; the fraction-of-ceiling cell filled with 0.5's named denominator column. **N==0 is RED; a blank cell is RED.**
- **predict (milli → bench) — the plan's single board-level prediction, made once, here.** `step_wall_ms` lands **[33, 43]** and the ratio against 0.4's **chosen** incumbent arm lands **1.9x–2.5x**. Derivation, every term cited: R13 `step_wall_ms` 67.92 = `gpu_exec_ms` 56.93 + 11.0 non-GPU. 2.3+3.1 measured `gpu_exec` into [43.5, 45.5]; 4.1 added GPU work (its own measured KV-reduce delta) and 4.2 removed the allocation slice, leaving orchestration at [5.5, 6.2] with `emit` 0.81, `encode_dispatch` 0.47, `readback` 0.22 and the ~1.6 residual untouched. The band is widened for CoV and carries 4.1's measured (not predicted) net. **Phases 5 and 6 are NOT in this prediction** — R12's null gives no basis to predict wall movement from a dispatch collapse.
- **kill** the ratio does not fall below **3.0x** against the chosen incumbent arm ⇒ R13's decomposition is wrong somewhere, and the row names **which bucket did not move, by counter**, before any further card is scheduled.
- **memory gate** full G8 assertion at `capacity_tokens = 256`, both slopes and both caps, on the board tree. A breach here is a board-level NEGATIVE and demotes the offending feature. **KILL.**
- **rollback** n/a (measurement); the underlying features demote one line each.
- **blast** docs.
- **observe** every counter in the board cell + the per-route census + fraction-of-ceiling against 0.5 + `phys_footprint_bytes` / `device_allocated_bytes` slopes.
- **reprove** the seal command above.
- **log-row title** `the board after the body and the plan: the one prediction this plan made, against a denominator and a ceiling that were both measured first`

---

# PHASE 5 — write placement (scatter only; `shape.rs` is not touched)

*Conflict 1 (round 2) resolved for B2: `map.rs:118-124` states verbatim that a `Reduce`-wide destination-extent field "was rejected on blast-radius grounds" and the scatter convention chosen instead. Relaxing `project_output_shape` re-litigates a closed adjudication; the scatter expression compiles and evaluates today on CPU with its worked example in its own doc. **Scatter only. `shape.rs:469-485` is never edited — that is the proof the constraint was routed around, not weakened.***

### 5.1 — Write the expression before changing anything `[S 5.1, B2 R11(ii)]`
- **tier** worker · **depends_on** 4.3
- **worktree** `…/proxima-wt-risc28` · `gpu-risc/28-kv-scatter-expression` · own target
- **opens** `proxima-tensor/src/map.rs:109-131` — verified verbatim: the write-direction convention, "the CPU interpreter runs the reduce loop strictly sequentially, so a scatter never needs atomics", and "see `shape.rs`'s `infer_reduce` doc for why a `Reduce`-wide field was rejected on blast-radius grounds"; **`:175` `IndexMap::scatter`**, `:209` `scatter_extent`, `:238` `as_gather_from_output`; `proxima-tensor/src/bind.rs:245-262` (`out_layout: Layout`, `out_scatter: Option<Lookup>` and the field doc already specifying the semantics), `:955-996` `bind_reduce`'s scatter arm, `:1011` `build_scatter_out_layout`; **`proxima-tensor/src/cpu.rs:6911` `run_reduce_scatter`** with its doc's worked example (`src=[10,20,30,40]`, `idx=[2,0,2,1]`, dest extent 3, `Add`/`Zero` → `[20,40,40]`); `proxima-tensor/src/shape.rs:493-511` `scatter_output_shape`; the hole: `omega/src/msl.rs:933`.
- **commands** one new test in `proxima-tensor` building a KV-append as `Reduce { out_map: IndexMap::scatter(indices = "kv_cache.{l}.write_row", index_map, iter_rank, non_scattered = [(1, head_axis), (2, dim_axis)], gathered_dim = 0, destination_extent = capacity_tokens) }` and running it through `cpu::evaluate`. **No production code.**
- **expect** ≥4 tests: (a) `run_reduce_scatter`'s own doc worked example reproduced exactly — **the example is the spec and the test**; (b) the KV expression places `new_count` rows at offset `cached_len` in a `capacity`-sized buffer on CPU; (c) two successive appends round-trip; (d) `omega::msl::emit` rejects it with exactly `EmitError::ScatterNotSupported` at **`msl.rs:933`** — a **red test asserting the known hole**, so 5.2 has something to turn green. **N==0 is RED.**
- **predict** none — a compile-and-evaluate proof, not a measurement.
- **kill** the expression does not compile or does not evaluate ⇒ the scatter form cannot express placement; **only then** does the affine form become a candidate, and the row records exactly which clause blocked it (and that re-opening `shape.rs:469-485` means re-opening `map.rs:118-124`'s adjudication, with that quote attached).
- **memory gate** N/A — one test file, CPU only. **rollback** `git revert`; zero production code. **blast** one test file.
- **observe** CPU buffer contents vs a hand-computed placement; the emitter's error variant.
- **reprove** `cargo nextest run -p proxima-tensor --features std -E 'test(kv_placement)'`
- **log-row title** `the expression before the type: KV placement written as a scatter, on main, today, with the CPU doc's own worked example as the test`

### 5.2 — The MSL scatter emitter: injective only, collisions declined by name `[B2 R11, S 5.3, crit R15]`
- **tier** worker · **depends_on** 5.1, 1.1
- **worktree** `…/proxima-wt-risc29` · `gpu-risc/29-msl-scatter-emitter` · own target
- **opens** `omega/src/msl.rs:923-946` `validate` — verified, the reject at **`:933`**; siblings **`wgsl.rs:364`**, **`cuda.rs:241`**, error defined `error.rs:53`; `msl.rs:2361, 2734, 3079, 3460, 3541` (`u.out_base` already emitted — **`:2734` verified**) and `:2216, 3502` (`long out_base` uniform field), so the write-side offset plumbing already exists; `msl.rs:1517-1560` `grid_threads`; `metal.rs:2214-2216` (fault buffer), `:2260` `check_gather_fault`; reference implementation `cpu.rs:6911`.
- **the one field addition, both questions answered** *Pipe question:* the uniform struct is a POD ABI record read by a kernel — no stages, no dataflow; `backend.rs:1-52`'s adjudication covers this boundary. *Relocation question, call site both ways:* **Way A** — three `long`s beside the `out_base` the struct already carries (`out_index_offset`, `out_element_stride`, `out_extent`), bind unchanged, one uniform buffer, one bind slot, the `UNIFORM_BUFFER_REUSES` path at `:2075` intact. **Way B** — a new `ScatterUniforms` type in a second buffer: one extra allocation **and** one extra bind slot **per dispatch on the 100% path**, and `UNIFORM_BUFFER_REUSES` duplicated. The lines are not identical and Way B costs a bind slot on every dispatch ⇒ **Way A; no new type.**
- **the atomics question, which the CPU doc explicitly does not answer for GPU** `map.rs:113-115` says a scatter never needs atomics *because the CPU interpreter is sequential*. A GPU emitter has no such guarantee. This card supports the **injective** case only and proves injectivity **structurally**, not by assumption: `Route::ScatterReduce` is selected only when `keep == Keep::Reduce`, `init != FirstElement`, and the indices leaf is declared strictly monotonic by name convention (`"*.write_row"`); every other scatter is **`Route::Declined(ScatterMayCollide)`** falling back to `EmitError::ScatterNotSupported` **with a reason** — declined honestly, never silently racing. A colliding-scatter GPU path is a named future work item.
- **commands**
  ```
  cargo build -p omega --no-default-features --features alloc
  bash scripts/omega-gate.sh
  flock …/.gpu-measure.lock -c 'cargo nextest run -p omega --features metal,cpu,instrument,metal-scatter -E "test(scatter)"'
  ```
  Declare `metal-scatter` default-off + forward.
- **expect** N1 Metal-vs-CPU parity on `run_reduce_scatter`'s doc worked example. N2 parity at the real KV shape — `[new_count=1, kv_heads=8, head_dim=128]` scattered into `[capacity=256, 8, 128]` at row `cached_len`, Metal vs `cpu::evaluate`, **0 ULP**. N3 `ran_count >= 6`; **`ran_count == 0` is RED** (the default-off N==0 trap). N4 with the feature OFF, `ScatterNotSupported` still fires — 5.1(d) stays green. N5 a colliding fixture returns `Route::Declined(ScatterMayCollide)` and the named error, never a wrong answer. N6 the parity test run **100×** is byte-identical every time (the race check). **N==0 is RED.**
- **predict (nano → micro)** a scatter-reduce writing `[1,8,128]` into `[256,8,128]` costs within **10%** of the equivalent affine reduce writing `[1,8,128]` into `[1,8,128]` on `metal_vs_cpu.rs` — the only difference is one indexed load and one add on the output address, and the traffic is identical. A larger cost means the index load is not coalescing, and the work item is the index layout.
- **kill** any ULP difference vs `cpu::evaluate` (§14). Any non-determinism across the 100 runs.
- **memory gate** the destination is a caller-owned buffer that already exists; the emitter allocates nothing beyond three uniform `long`s (+24 B per uniform buffer). Caps and slopes identical to 4.2's; any increase is a NEGATIVE. **KILL.**
- **rollback** default-off; with it off `validate` returns `ScatterNotSupported` exactly as today. WGSL and CUDA are untouched and keep declining — 7.3–7.5 give them the route.
- **blast** `omega/src/msl.rs` (`validate` `:933`, a new `push_scatter_reduce_body`, the uniform struct `:2216`/`:3502`, `grid_threads`'s reduce arm, `route.rs` gains `ScatterReduce` + `Declined(ScatterMayCollide)`), `omega/src/error.rs` (a reason on `ScatterNotSupported`). **`proxima-tensor` is not touched at all — that is the headline.**
- **observe** `Route::ScatterReduce` census count; `ScatterNotSupported` count going to zero for the KV writes; gather/scatter fault counters.
- **reprove** the three commands above.
- **log-row title** `write placement needed no new Op: IndexMap::scatter already existed, shipped on CPU, and was rejected by three emitters — Metal now emits it, injective only, declining collisions by name`

### 5.3 — KV becomes a device-resident buffer written in place; `found == expected` holds by construction `[B2 R12, S 5.6, crit RS-2, MS-2, c, HC-4]`
- **tier** worker · **depends_on** 5.2, 4.1, 4.2
- **worktree** `…/proxima-wt-risc30` · `gpu-risc/30-kv-device-resident-write` · own target. Not `perf/kv-device-resident` (checked out).
- **opens** (1) `proxima-model-interop/src/generate.rs:621-656` — `LayerCache { k_even, k_odd, v: Vec<f32> }` `:621-625`, `append` = 3× `extend_from_slice` `:636-640`, `named_blocks` handing the whole `Vec` as `QuantizedBlock::Float32` `:642-656`; (2) `:1313-1391` the assembly, the per-layer KV loop `:1364-1389`, `kv_cache_upload_elements` `:1346-1361` whose comment states "this is the full `cached_len`-sized array re-bound as a model input every single step"; (3) `:1559` `cached_len += new_count`; (4) `omega/src/metal.rs:1744-1815` `register_checkpoint_mapping` + `backend.rs:402-414` — the shipped "one device buffer addressed by offset" mechanism landed by `7d09145`, **reused here for KV**; (5) `metal.rs:1848-1892` `NOCOPY_BUFFERS: BTreeMap<(usize,usize), MetalBuffer>` keyed `(pointer, byte_length)`, `:1879` `upload_block_no_copy`, `:1903` `upload_block_no_copy_uncached` (where `23e2e5e` routes non-resident blocks, so KV creates a fresh wrapper every token), `:1914` `create_no_copy_buffer`, `:1616` `is_page_aligned`, `:350-362` `mark_resident` (classifies **by name**); (6) `metal.rs:984-1000` and `cpu.rs:337-356` — the two strict validators; (7) `proxima-tensor/src/align.rs:42-46, :69` `AlignedBuffer::new(min_elements, page_size)`, **zero production callers**, doc requiring the caller to "size its tensor input to `buffer.len()`, not the value it originally asked for"; (8) incumbent `llama-kv-cache-unified.cpp:74-118` (allocated once), `:749-788` (`ggml_cpy` into `ggml_view_1d` at a byte offset) (R8).
- **the over-allocation contract needs NO validator change** [conflict 6; crit RS-2, MS-2, c]. Both validators compare **element counts**: `metal.rs:991-1000` `let expected = element_count(shapes.of(*node)); let found = block_element_count(block)?; if found != expected` and `cpu.rs:346-356` `if data.len() != expected`. After 4.1 the KV leaf's symbol-1 extent **is** the bucketed capacity, and this card hands the **capacity-sized** buffer — so **`found == expected` holds and neither loop is touched**. No parallel `&[Option<u64>]` slice, no `QuantizedBlock` field, no positional desync across `resolve_named_blocks` (`metal.rs:584`) / `block_node_ids` (`:1111-1118`), no unsound `declared == found` predicate, no unit ambiguity. **`AlignedBuffer`'s page rounding is handled at the source**: `capacity_elements` is read back from `buffer.len()` **after** `AlignedBuffer::new` rounds, and the symbol fed to the plan is computed from that same number — which is exactly what `align.rs:36-41`'s doc requires of every caller [crit HC-4]. `capacity_tokens` therefore lands as an element count derived from the buffer, never from the value asked for.
- **commands** `LayerCache`'s three `Vec<f32>` become three `AlignedBuffer`s sized from `[kv_cache].capacity_tokens` (4.1's section) with `page_size` from the sizing config and the Metal value as the override; register the three **once** through `register_checkpoint_mapping` so the KV read leaf and the write destination are the **same device buffer** (item 7's driver-level alias — a driver fact, not an IR concept); delete `LayerCache::append`; the write happens on-device via 5.2's scatter with `indices` leaf `"kv_cache.{l}.write_row"` pushed into `named_blocks` beside `"eps"` (the `+4` capacity literal from 4.1 becomes `+5`, asserted by 4.1's `named_blocks.len() == block_node_ids.len()` test); `named_blocks` hands the **capacity-sized** buffer unchanged every token so `NOCOPY_BUFFERS`' `(pointer, byte_length)` key **hits** instead of missing on growth, and `mark_resident` adds the KV names so `23e2e5e`'s non-resident routing no longer sends them to `upload_block_no_copy_uncached`. Two features: `kv-scatter-write` in `proxima-tensor` (the graph half) and `metal-kv-resident` in `omega` (the driver half), both default-off, both forwarded — two features because they are two independently revertable concerns. A test builds `proxima-model-interop --features std` (no metal) to prove `LayerCache` obtains a page size on a **non-Metal** build (`metal = ["dep:omega","std"]` is optional).
- **expect** N1 `kv_cache_upload_bytes == 0` on every steady step (R13: 8.13 → 9.70 MB, +262,144 B/token); nonzero is RED. N2 `nocopy_reuses` rises to 32 × 3 = **96**/steady step; `copying_uploads` for KV names == 0. N3 `resident_uploads` counts the KV names once, at the first step only. N4 `generated_text` unchanged and `2651`/`"known"` green. N5 `plan_hits == F − distinct_shape_count()` still. N6 `BLOCK_UPLOAD_BYTES_COPIED` (0.1) drops by the KV share. N7 ≥7 tests: full-budget decode identical with and without; pointer stability across appends; the no-copy cache hits from step 2; **the KV allocation size is printed and asserted BEFORE allocation** and is < 8 GB; `found == expected` on both validators with **no validator edit** (a test asserting `element_count(shapes.of(kv_node)) == buffer.len()`); the no-metal build; the qwen3.5 `DenseAttention`/`Ssm` cache states (`generate.rs:687`, `:738`) untouched and their `unreachable!` at `:1386-1388` intact. **N==0 is RED.**
- **predict (milli → bench)** `block_upload_ms` falls from R13's **2.0** to the weight-only residual (R13 notes it is 0.4 on step 2 and 1.7–3.5 otherwise — the "otherwise" is the KV term); `kv_cache_upload_bytes` → 0 after step 1; combined with 4.1's now-paid-off bucketing, `step_wall_ms` lands **[48, 53]** = **2.74–3.02x** and `gpu_exec_ms` **[42, 45]**.
- **kill** `generated_text` drift ⇒ §14, revert. A scatter write and a read of the same buffer within one command buffer producing stale data ⇒ a missing barrier; we have one encoder / one command buffer (`metal.rs:449-568`) and the incumbent has no barrier API at `n_cb=1` (R8), so ordering within the encoder is what we rely on; if it does not hold, the write moves to its own dispatch ordered before the read; if **that** fails, the card dies and the row records the ordering finding. `block_upload_ms` falls but `step_wall_ms` does not beyond both CoV bands ⇒ record it inside the noise band and stop; R2's "~3.3%" is MEMORY with **no counterpart term in R13** and may not be claimed as a win it did not produce.
- **memory gate — the site of a prior failure** [G8's 34 GB trap]. Capacity comes from `[kv_cache].capacity_tokens`, **never** `context_length`; `build.rs`'s `require_at_most(65536)` (4.1) rejects a runaway at **build time**; the card **prints `capacity_tokens * 262_144` before allocating** and runs at 256 first (`DEVICE_CAP_BYTES = 4_247_525_888`) before raising it (at 4096: 5_254_158_848). **Both slopes must go to 0** — a device-resident cache that still grows per token has not become resident, and that is RED **on the slope**, not on the timing. **KILL.**
- **rollback** two independent default-off features; `kv-scatter-write` off restores the host `append`, `metal-kv-resident` off restores per-token `named_blocks`; either reverts alone.
- **blast** `proxima-model-interop/src/generate.rs` (`LayerCache` `:621-656`, the KV loop `:1364-1389`, `cached_len` `:1559`), `proxima-tensor/src/spec.rs` (`append_mistral_cached_layer` gains a scatter-`Reduce` per KV component), `omega/src/metal.rs` (`mark_resident` name set, KV registration), `proxima-tensor/src/align.rs` (first production caller). Widest correctness surface after 5.2.
- **observe** `kv_cache_upload_bytes` (`generate.rs:1657`), `nocopy_reuses` / `resident_uploads` / `resident_reuses` / `copying_uploads` / `mapping_offset_uploads` (`generate.rs:1729-1730`), `nocopy_cache_len` (`metal.rs:1852-1858`), `BLOCK_UPLOAD_BYTES_*` (0.1), `device_allocated_bytes`, `phys_footprint_bytes`.
- **reprove** the G3 bench cell in three arms (both off / driver only / both on), interleaved 3×.
- **log-row title** `the KV cache stops round-tripping through the host: one device buffer per layer at capacity, written in place by a scatter, read as an alias — and the validators never changed, because the leaf extent IS the capacity`

### 5.4 — On-device argmax `[S 5.7]`
- **tier** worker · **depends_on** 5.2
- **worktree** `…/proxima-wt-risc31` · `gpu-risc/31-on-device-argmax` · own target
- **opens** `proxima-model-interop/src/generate.rs:1643` `sample_next_token`, `:1649` `next_ids`, `:1659` `greedy_pick_ms` (in `token_breakdown`, **not** `token_breakdown_metal`), `:1640-1670` `greedy_pick_started/ticks`; `proxima-tensor/src/op.rs:203-205` — `Reduce`'s own doc names **argmax**, distinguished by `keep` and by whether `out_map` is data-dependent, i.e. a **scatter**, which is why this depends on 5.2 and not the reverse. R3/M7: `greedy_pick` depends on `waitUntilCompleted` — a true data dependency; the fix is on-device, not threading.
- **commands** append the argmax reduce to the program in `spec.rs`; read back 4 bytes instead of vocab-sized logits. Declare `on-device-argmax` default-off in `proxima-tensor/Cargo.toml`, forwarded.
- **expect** ≥3 tests: on-device argmax matches `greedy_pick` bit-for-bit over the full budget on the real checkpoint; ties break as the CPU path does; **the non-greedy sampling path still receives full logits** (the feature must not silently degrade `sample_config`). Plus `2651`/`"known"`. **N==0 is RED.**
- **predict (nano → micro)** `readback_bytes` falls from vocab×4 (≈128 KB) to **4**; `readback_ms` falls from R13's **0.22** toward 0; `greedy_pick_ms` toward 0. Combined **< 0.5 ms/token** against R13's 67.92 — small, and the row leads with that.
- **kill** any drift in generated text ⇒ §14, revert.
- **memory gate** removes a vocab-sized host readback buffer; RSS should fall; an increase is a NEGATIVE. **KILL.**
- **rollback** `git revert`; feature default-off.
- **blast** `proxima-tensor/src/spec.rs` output roots, `generate.rs` sampling path, one emitted scatter route.
- **observe** `READBACK_CALLS`, `READBACK_BYTES`, `readback_ms`, `greedy_pick_ms`, `phys_footprint_bytes`.
- **reprove** the G3 bench cell; `grep -o 'readback_bytes=[0-9]*'`; assert generated text unchanged.
- **log-row title** `argmax moves on-device with no new Op; readback falls from 128 KB to 4 bytes and the win is under half a millisecond`

---

# PHASE 6 — the minimal graph

*Off the critical path to 4.3's board prediction by construction (conflict 3, round 1): two independent cells say a dispatch collapse is not a wall movement. This phase carries the second test of that proposition, on our tree, with wall in the criterion.*

### 6.1 — Ops-per-layer census against the incumbent's 23 `[S 6.1, B2 R13-N1, crit j — retiered hands → worker]`
- **tier** worker · **depends_on** 5.3
- **worktree** `…/proxima-wt-risc32` · `gpu-risc/32-ops-per-layer-census` · own target
- **opens** `proxima-tensor/src/spec.rs:2336-2865` `append_mistral_cached_layer` (530 lines, 25 args), sole caller `:6282`; the incumbent's 23 real ops/layer enumerated at `llama-model.cpp:4691-4845` + `llama-graph.cpp:464-497, 514-616, 1071-1121, 1205-1253`, views/reshape/permute no-ops at `ggml-metal.m:1835-1847`, ~740 dispatches/token — which **corrects** the MEMORY figure "15/layer, ~483": ours 1196 vs 740 = **1.62x** (R8).
- **commands** derive ops/layer from 1.1's cold census by grouping `NodeId` ranges to layer roots (the definition of "a layer" is fixed here, in code, from `layer_roots` — which is why this is a worker card, not hands [crit j]) and assert it in a test.
- **expect** the census prints per-layer op counts and the assertion holds; main's number to confirm is **37 ops/layer** at `OPS_AFTER_PRUNE` dispatches. **N==0 is RED.**
- **predict** none — a census.
- **kill** n/a. **memory gate** instrument + test only; slopes unchanged; **KILL** at G8 for the harness run.
- **rollback** `git revert`. **blast** the census + one test.
- **observe** ops/layer; `encode_dispatch_calls`.
- **reprove** the G3 bench cell + the census assertion.
- **log-row title** `37 against 23: the ops-per-layer census, and the incumbent count it is measured against`

### 6.2 — Collapse the two-range online softmax to one range `[S 6.2, B2 R13, crit b, H-4]`
- **tier** worker · **depends_on** 6.1, 5.3, 5.2
- **worktree** `…/proxima-wt-risc33` · `gpu-risc/33-attention-single-range` · own target. Not `perf/attention-single-range` (checked out).
- **opens** `proxima-tensor/src/spec.rs:2303-2319` — the doc naming the constraint being routed around; `:2596-2720` the two-range combine, `:2616-2617` the comment, `:2610` the mask consumption; the even/odd RoPE split; `:6282` the sole call site; and the fan-out sites the row must scope: **`:2867-2891`** (Qwen3.5 dense-attention counterpart), **`:6465`** (the split-half RoPE path, documented as **NOT** going through this function — so "the even/odd RoPE split collapses in the same move" does **not** cover it and the row says which checkpoints it does cover), **`:3455`** (the MoE counterpart) [crit H-4]. Incumbent decode attention: `mul_mat(K,Q)` → `soft_max_ext` (scale+mask+max+exp+sum+normalize fused in ONE kernel, `ggml-metal.metal:1051-1145`, nth up to 256/ne00 via `ggml-metal.m:2501-2525`) → `mul_mat(V)` (R8).
- **the mechanism** with 5.3 landed, the new token's K/V are **already in the cache buffer before the attention reduce runs**, so there is only ever ONE source range and the two-range combine is dead code. One `Reduce` over `[0, bucket_capacity)` with 4.1's tail mask and the existing causal mask. `spec.rs:2303-2319`'s doc is rewritten to record that the constraint it names was satisfied **by scatter**, not by relaxing `project_output_shape` — **`shape.rs:469-485` is not touched**, which is the proof.
- **commands** recover `merge-tracked.patch` (0.6; 1 file +1181/−82 on `spec.rs`) as **reference only** — main moved `spec.rs +8735/−2836` since (`0c3bd4f`), so expect a full conflict and **rewrite rather than merge**. Declare `attention-single-range` default-off in `proxima-tensor/Cargo.toml`, forwarded. Keep the two-range body under `#[cfg(not(feature="attention-single-range"))]` so both arms stay green and every commit bisects. Run the CPU oracle first.
- **expect** N1 real ops per layer **≤ 23**; derived total `32*23 + get_rows + rms_norm + mul + output mul_mat = 740`, so **`op_count <= 780`** (a 5% allowance for our `Iota`/`Constant` control nodes, R13: 39 ops / 0.169 ms); `op_count > 780` fails brief item 8. N2 `generated_text` unchanged, `2651`/`"known"` green — §14 binds hardest here because this changes softmax arithmetic order. N3 a CPU parity test single-range == two-range to **1e-6** (the summation order genuinely changes; the tolerance is stated and justified) with the **token bit-identical**. N4 the route census shows the elementwise count collapsing from 547 toward the incumbent's shape. N5 Qwen3.5's hybrid attention/ssm path **re-parity-tested, not assumed**. **N==0 is RED.**
- **predict (milli → bench)** ops/layer **37 → ≤23**; `encode_dispatch_calls` → **≤780**; `gpu_exec_ms` → **[32, 38]** and `step_wall_ms` → **[40, 46]** = **2.28–2.63x**, on the strength of removing most of R13's elementwise 7.350 ms and a large share of the `"(no named operand)"` 681 ops / 11.382 ms, with R12's feature-off control (35.117 ms GPU carrying the paired body) as independent support.
- **kill, written against BOTH the control and the noise floor** [crit b]: the success shape is **dispatches down AND `gpu_exec_ms` not risen beyond 2× the measured CoV AND `step_wall_ms` down beyond both CoV bands**. R13's `gpu_exec_ms` CoV is 0.7%, so "killed if gpu_exec rises at all" would kill on noise; the criterion is a rise **> 1.4%**. **Wall is in the criterion**, not only counts — R12's finding was about wall (0.07%), and a 6.2 that reduces dispatches with flat GPU and flat wall reproduces the parallel branch's outcome exactly and **must not pass**. If wall does not move, that is the **second independent** measurement saying the graph is not the mass; record the two-agreeing-results conclusion and stop Phase 6. The single-range graph still lands if it is correctness- or maintenance-positive, but **not as a perf row**, and 0.7's adjudication re-opens only if ≤23 ops/layer proves unreachable.
- **memory gate** fewer nodes ⇒ fewer arena buffers ⇒ `device_allocated_bytes` must **decrease** vs 5.3; an increase is a NEGATIVE. **KILL.**
- **rollback** `git revert`; feature default-off; the two-range body stays under `#[cfg(not(...))]`. Highest rebase-conflict surface in the plan (`spec.rs`); rebase against main early and often.
- **blast** `proxima-tensor/src/spec.rs` only — Mistral, Qwen3, Qwen3.5 hybrid and the MoE counterpart; the split-half RoPE path at `:6465` is explicitly **out of scope** and the row says so. Zero emitter change, zero driver change — the duplication was a graph defect upstream of any backend.
- **observe** ops/layer census, `encode_dispatch_calls`, `op_profile_bucket kind=elementwise op_count` (R13: 547 / 7.350 ms), `gpu_exec_ticks`, `step_wall_ms`, the per-route split.
- **reprove** the G3 bench cell ON/OFF interleaved 3× + 6.1's assertion + the CPU parity test.
- **log-row titles** `attention was duplicated because the graph could not write in place: with scatter placement the two ranges become one, at or under the incumbent's 23 ops per layer` · `the control that says a dispatch collapse is not automatically a win, restated on our tree with wall in the criterion`

---

# PHASE 7 — one emitter core, every backend covers every kind, one plan

### 7.1 — The `Dialect` design: which of the ~26 functions are TEXT and which are STRUCTURE `[S 7.2, B2 R14, crit MS-5, g
### 7.1 — The `Dialect` design: which of the ~26 functions are TEXT and which are STRUCTURE `[S 7.2, B2 R14, crit MS-5, g]`
- **tier** judge · **depends_on** 1.1, 5.2
- **worktree** `…/proxima-wt-risc34` · `gpu-risc/34-dialect-design` · own target
- **opens** R5's census re-verified: `msl.rs` 4712 + `wgsl.rs` 1929 + `cuda.rs` 1838 + `metal.rs` 2385 + `wgpu_driver.rs` 872 + `backend.rs` 615 + `sized.rs` 45 + `error.rs` 84 + `lib.rs` 73 = **12,553 lines**; exactly **3 types + 2 fns** shared (`Binding`, `PackedCodec`, `PackedOperands`, `gather_count`, `gather_slots`; `wgsl.rs:105`, `cuda.rs:66`); ~26 functions × 3 backends ≈ 78 near-duplicates; `msl.rs:673-697` (the surviving match), `:4656` `emit_is_deterministic_byte_equal`; `cuda.rs:146-183` `emit_cuda` rejecting Iota/Constant; `omega/src/wgpu_driver.rs` (872 lines, `execute_plan` `:599`, `execute_plan_named` `:866`) — named because `omega-gate.sh [2/6]` builds `--all-targets --all-features`, so a wgpu compile break lands in the gate for every subsequent card.
- **the deliverable — the field list is enumerated here, not elided** [crit MS-5, g]. `trait Dialect` carries **exactly seven text-producing methods**, each returning `&'static str` or writing into a `&mut String`:
  1. `scalar_op_expr(ScalarOp, &[&str]) -> String` — the per-op infix/intrinsic text.
  2. `preamble() -> &'static str` — includes/using/namespace header.
  3. `kernel_signature(&Bindings) -> String` — entry-point signature syntax.
  4. `entry_name(&BoundOp, Route) -> String` — symbol naming convention.
  5. `simd_reduce_intrinsic(ScalarOp) -> &'static str` — `simd_sum` / `subgroupAdd` / `__shfl_down_sync` family.
  6. `threadgroup_barrier() -> &'static str`.
  7. `atomic_fold(ScalarOp) -> Option<&'static str>` — `None` is a legitimate answer and produces `Route::Declined(NoDialectIntrinsic)`.
  Everything structural stays in **one generic core**: `validate`, `reduction_dims`, `bindings`, `grid_threads`, `push_body_steps`, `operand_read`, the four `render_*`, `reduce_is_cooperative`, and `route::of`. **`grid_threads` (`msl.rs:1517-1560`) is arithmetic and `reduce_is_cooperative` (`:824`) is a predicate — neither is text, and both go to the core**, which is the classification crit-g says was missing. §20: **static dispatch through a generic parameter, no `Box<dyn Dialect>`** — the three backends are a closed set known at compile time and each dialect's methods monomorphise into a hot string builder.
- **commands** produce `docs/bench-campaigns/2026-09-03-gpu-one-risc/dialect-map.md`: a table with one row per one of the ~26 functions × 3 backends, each classified **TEXT (which of the 7 methods) / STRUCTURE (core) / DELETE (duplicate)**, and the per-kind commit order for 7.3–7.5. One docs commit. **No source change** — this card decides, 7.3–7.5 type.
- **expect** N = 26 × 3 = **78 rows**, every one classified with no blanks; the 7 method signatures written out; the `(Backend, Route)` exhaustiveness plan stated. **N==0 is RED; a blank classification is RED.**
- **predict** none — this card produces a design record, not a measurement.
- **kill** a function cannot be classified TEXT or STRUCTURE without reading the other backends' bodies ⇒ it is a fourth category (backend-specific *behaviour*); name it, count it, and if the count exceeds 5 the one-core claim is scoped down in this row before 7.3 starts.
- **memory gate** N/A — docs only; stated.
- **rollback** docs revert. **blast** `docs/bench-campaigns/`. No source.
- **observe** the 78-row table; the per-method fan-in count.
- **reprove** `awk -F'|' 'NF>3' docs/bench-campaigns/2026-09-03-gpu-one-risc/dialect-map.md | wc -l`
- **log-row title** `78 near-duplicates classified before one line moves: seven text methods, one structural core, and the four functions that are neither`

### 7.2 — CUDA covers `Iota` and `Constant`, in two commits `[S 7.1, crit B-4]`
- **tier** worker · **depends_on** 7.1
- **worktree** `…/proxima-wt-risc35` · `gpu-risc/35-cuda-iota-constant` · own target
- **opens** `omega/src/cuda.rs:146-183` `emit_cuda` + `CudaUnsupportedOpKind`; reference `omega/src/msl.rs:2044-2103` `render_iota` / `render_constant`; `omega/src/error.rs` (the variant's definition).
- **commands** implement both kinds; **do not delete the error variant in the same commit** — land the implementation first (green), then remove the now-unreachable variant in a second commit, so rollback is one revert of a small commit rather than an unwind through downstream exhaustive matches.
- **expect** ≥2 new emit tests per kind; `cargo nextest run -p omega --features cuda,cpu` runs a **non-zero** count (`cuda` is not in `default` — the N==0 trap); `grep -c CudaUnsupportedOpKind omega/src/cuda.rs == 0` after the second commit. **N==0 is RED.**
- **predict (nano → micro)** the 15-cell coverage matrix `[Elementwise, Reduce(Reduce), Reduce(Scan), Iota, Constant] × [msl, wgsl, cuda]` is **15/15** either `Ok` or an explicitly named `Route::Declined(reason)`; a silent absence is RED.
- **kill** CUDA cannot be compiled on this host (no toolchain) ⇒ emission is still testable **as text**, which is the point of a sans-IO emitter (§11); assert the emitted CUDA parses structurally and do **not** claim it runs.
- **memory gate** N/A — emission only, no device. **rollback** revert the second commit (restores the variant), then the first. **blast** `omega/src/cuda.rs`, `omega/src/error.rs`.
- **observe** the coverage matrix; per-backend route census.
- **reprove** `cargo nextest run -p omega --all-features -E 'test(backend_coverage_matrix)'`
- **log-row title** `one RISC means every backend covers every kind: the 15-cell matrix and the deleted rejection`

### 7.3 — The core, kind 1: `Elementwise` `[S 7.2 split, B2 R14, crit g]`
- **tier** worker · **depends_on** 7.1, 7.2
- **worktree** `…/proxima-wt-risc36` · `gpu-risc/36-core-elementwise` · own target
- **opens** 7.1's `dialect-map.md` rows for `render_elementwise`, `scalar_op_expr`, `operand_read`, `push_body_steps`, `bindings`, `preamble`, `kernel_signature`, `entry_name`; `msl.rs:2104-2177`, the wgsl and cuda twins; `msl.rs:4656`.
- **commands** move the structural half to the core, implement the touched `Dialect` methods three times, delete the duplicates. `bash scripts/omega-gate.sh`. One commit, green.
- **expect** N1 **byte-identical emitted source** for every `Elementwise` op in the real openchat program, before and after, per backend — one byte of drift is RED. N2 the gate's `ran_count` is **≥** main's. N3 the alloc-tier build (`--no-default-features --features alloc`) compiles the core and all three dialects and **states which modules it built** (§3's N==0 warning). **N==0 is RED.**
- **predict (nano → micro)** `wc -l omega/src/{msl,wgsl,cuda}.rs` falls by **≥600** on this kind alone; `gpu_exec_ms` unchanged within R13's 0.7% CoV.
- **kill** N1 fails ⇒ the refactor changed emission; bisect the dialect method that moved and either restore its text or record the change as its own row with its own parity evidence. Never accept "the new text is equivalent" without the byte comparison.
- **memory gate** compile-time only; runtime allocation identical by N1; stated rather than blank.
- **rollback** `git revert` one commit. **Not feature-gated** (a gate on a refactor means two emitters); the golden from 1.1 is the firewall.
- **blast** the `Elementwise` paths in all three emitters + the new core module.
- **observe** line counts, golden hash per route per backend, `gpu_exec_ticks` flat.
- **reprove** `cargo nextest run -p omega --all-features -E 'test(golden_source)'` + `bash scripts/omega-gate.sh`
- **log-row title** `kind one of four through the core: the elementwise bytes did not move and here is the hash`

### 7.4 — The core, kind 2: `Reduce` (all routes, including `ScatterReduce`) `[S 7.2 split, B2 R14]`
- **tier** worker · **depends_on** 7.3
- **worktree** `…/proxima-wt-risc37` · `gpu-risc/37-core-reduce` · own target
- **opens** 7.1's rows for `render_reduce`, `reduce_is_cooperative`, `reduction_dims`, `grid_threads`, `fold_init_tokens`, `simd_reduce_intrinsic`, `atomic_fold`, `push_scatter_reduce_body` (5.2); `msl.rs:2178-2257`, `:3140-3194`, `:1235`, `:1450`, `:1517-1560`; the wgsl/cuda twins (neither has tiled-GEMM or packed row-block).
- **commands** move the structural half; where a route genuinely has no dialect expression (no `simdgroup_matrix` outside Metal), the dialect returns `Route::Declined(NoDialectIntrinsic)` — **a named decline, exhaustively matched, never a silent absence**. One commit, green.
- **expect** N1 byte-identical emitted source for every `Reduce` op per route per backend. N2 a compile-time exhaustiveness proof: `match (Backend, Route)` with **no wildcard arm**, so adding a route without answering it in every dialect **fails to compile**. N3 `omega/tests/wgpu_parity.rs` unchanged and green; the scatter tests from 5.2 unchanged and green. **N==0 is RED.**
- **predict (nano → micro)** line count falls by a further **≥1200**; per-op `gpu_ns` in the milli harness is **unchanged within 1%** for every op — **any per-op movement falsifies N1** and is the more sensitive of the two tests.
- **kill** as 7.3. Additionally: a route that can only be expressed by adding an eighth `Dialect` method ⇒ stop and amend 7.1's design record with the reason before adding it.
- **memory gate** compile-time only; unchanged, proven by byte-identical emission.
- **rollback** `git revert` one commit; revert 7.4 before 7.3.
- **blast** the widest structural blast in the plan — `msl.rs`, `wgsl.rs`, `cuda.rs` reduce paths + the core. **Zero behaviour change by N1's construction**, and deliberately sequenced **after** every perf card so no perf number is entangled with a 12,553-line reorganisation.
- **observe** line counts, golden hashes, the `(Backend, Route)` exhaustive match (a compile-time observable — the strongest kind).
- **reprove** the golden test + `bash scripts/omega-gate.sh` + the G3 milli cell.
- **log-row title** `kind two of four: every reduce route through one core, and every backend answers — with a reason where it declines`

### 7.5 — The core, kinds 3 and 4: `Iota`, `Constant`, and `Keep::Scan` `[S 7.2 split, B2 R14]`
- **tier** worker · **depends_on** 7.4
- **worktree** `…/proxima-wt-risc38` · `gpu-risc/38-core-iota-constant-scan` · own target
- **opens** 7.1's rows for `render_iota`, `render_constant`, `render_scan`; `msl.rs:2044-2103`, `:673-697` (the `Keep` arm); `cuda.rs` post-7.2.
- **commands** move the structural half; delete the duplicates. One commit, green.
- **expect** N1 byte-identical emitted source for every `Iota`/`Constant`/`Scan` op. N2 duplicated-function count falls from ~78 to **≈21** (7 dialect methods × 3), recorded on the row against R5's before-count. N3 the gate's `ran_count` ≥ main's. **N==0 is RED.**
- **predict (nano → micro)** total `wc -l omega/src/{msl,wgsl,cuda}.rs` falls from **8479** by **≥2500** across 7.3–7.5; emitted source byte-identical for all ops; `gpu_exec_ms` unchanged within CoV.
- **kill** as 7.3.
- **memory gate** compile-time only; unchanged, stated.
- **rollback** `git revert`; revert 7.5 → 7.4 → 7.3 in that order.
- **blast** all three emitters + the core.
- **observe** the before/after line counts, the golden hashes, the duplicate-function count.
- **reprove** the golden test + both gates.
- **log-row title** `78 near-duplicate functions across three emitters become one core and 21 dialect methods`

### 7.6 — `bind` owns the packed layout; the Metal-only post-bind rewrite is deleted `[R16, crit MS-3, HC-2, f]`
- **tier** worker · **depends_on** 7.5, 0.14
- **worktree** `…/proxima-wt-risc39` · `gpu-risc/39-bind-packed-layout` · own target
- **opens** `omega/src/metal.rs:1004` (the `bind` call), **`:1005-1012`** (the comment stating `layout_of` "assumes every operand is stored row-major in its DECLARED axis order … never true for a packed Q4_K/Q5_K/Q6_K weight, whose bytes are GGUF's native `[out,in]`"), **`:1013` `correct_packed_matmul_layouts(&mut resolved, &packed_operands.keys().copied().collect())`**, `:191` (the import), `proxima-tensor/src/bind.rs:1618-1647+` (its definition and doc), `:1594-1606` `layout_of` (`base += i64::from(axis.offset) * stride`), `:1718-1722` `bind`'s signature (**no backend parameter**); `proxima-tensor/src/cpu.rs:358` (the path that does **not** apply it and does not need to, because the CPU quantized path never reads packed bytes through `layout_of`); 0.14's fingerprint test and its recorded node-set diff.
- **commands** make `bind` produce the correct layout for packed operands so **no backend rewrites the plan after bind**. Two admissible forms, decided by the relocation question written out in the row: (A) `bind` takes the packed-operand node set as a parameter and applies the correction inside `layout_of`; (B) `layout_of` reads a per-operand **physical layout** carried on the `Op::Input` leaf. Prefer (B) if the leaf already carries the codec (it does — `QuantizedBlock` names it), because (A) changes one public signature and every caller. **Delete `correct_packed_matmul_layouts`'s call site at `metal.rs:1013` and the function**, in a second commit, so the rollback is one small revert.
- **expect** N1 0.14's fingerprint test **turns green**: Metal and CPU fingerprints equal, invariant under `metal-tiled-gemm`, deterministic across calls — and the `#[ignore]`/`EXPECTED RED` marker is removed in this commit. N2 `grep -c correct_packed_matmul_layouts omega/src/ proxima-tensor/src/ == 0` after the second commit. N3 every Q4_K/Q5_K/Q6_K parity suite green (`q4k_real_checkpoint_parity`, `q4k_matmul_layout`, `q4k_unpack`, `metal_parity`, `backend_parity`, `metal_real_forward`) — the strides this card moves are the strides those tests exist to protect. N4 `generated_text` unchanged and `2651`/`"known"` green. **N==0 is RED.**
- **predict (nano → micro)** 1.1's golden emitted source is **byte-identical** — the corrected layout is the layout Metal already executed, so the emitted MSL cannot move; only *where* the correction happens changes. Any golden drift means the pre-existing correction and the new one disagree, which is a correctness finding, not a refactor.
- **kill** any golden drift, any parity regression, or a fingerprint that is still unequal ⇒ there is a **third** rewrite; name it from the node-set diff and stop. A card that makes the fingerprint equal by weakening the fingerprint's field list has inverted the test.
- **memory gate** no allocation change; slopes and caps identical to 6.2's; **KILL** at G8 for the harness run.
- **rollback** `git revert` the deletion commit (restores `correct_packed_matmul_layouts`), then the bind commit; 0.14's test returns to its pre-registered RED state, which is a documented state, not a broken one.
- **blast** `proxima-tensor/src/bind.rs` (`layout_of` and/or `bind`'s signature — **every backend's plan**), `omega/src/metal.rs:1004-1013`. Highest-consequence correctness surface in Phase 7, which is why it lands after the emitter core and after every perf card.
- **observe** the two fingerprints and the node-set diff (0.14), the golden hashes, the parity suites' counts.
- **reprove** `cargo nextest run -p omega --features metal,cpu -E 'test(plan_fingerprint)'` + `bash scripts/omega-gate.sh` + the G3 bench cell.
- **log-row title** `the Metal driver stops rewriting the bound plan: bind produces the packed layout, metal.rs:1013 is deleted, and brief item 1 becomes true`

### 7.7 — The fingerprint becomes a gate `[B2 R15, crit MS-3]`
- **tier** hands · **depends_on** 7.6
- **worktree** `…/proxima-wt-risc40` · `gpu-risc/40-plan-fingerprint-gate` · own target
- **opens** `scripts/omega-gate.sh`; 0.14's test file.
- **commands** add a gate step running the three fingerprint tests under the named feature set `metal,cpu,instrument` and asserting equality; add a second step asserting `grep -c correct_packed_matmul_layouts == 0`.
- **expect** the gate step passes; a fingerprint inequality is RED and prints both values and the node-set diff. **N==0 is RED.**
- **predict** none — a gate.
- **kill** the gate is green while 7.6's grep is non-zero ⇒ the gate is running the wrong feature set; fix before landing.
- **memory gate** N/A — a gate step. **rollback** revert the gate step. **blast** `scripts/omega-gate.sh`.
- **observe** the gate's own output; both fingerprints.
- **reprove** `bash scripts/omega-gate.sh`
- **log-row title** `one bound plan is a gate, not a claim: the same fingerprint through the CPU entry and the Metal entry, invariant under metal-tiled-gemm`

---

# PHASE 8 — the incumbent cells that do not exist (R9: measured only for llama.cpp-Metal)

### 8.1 — torch-MPS arms `[S 8.3, B2 R02-B]`
- **tier** worker · **depends_on** 0.10, 0.5
- **worktree** `…/proxima-wt-risc41` · `gpu-risc/41-torch-mps-arms` · own target
- **opens** `proxima-onnx/scripts/torch_reference/inference_bench.py:29-32` — verified: `parse_args` has only `--threads` and `--runs`, no device arg; `model.py`; `train_bench.py`; `diagnostics.py`; ours: `omega/tests/training_step_parity.rs:400-607` (GPU train step exists as **untimed** parity tests only, R9). venv verified present with torch 2.13.0 + MPS (R0).
- **commands** add `--device {cpu,mps}` (default `cpu`, so the existing arm is byte-identical) and `torch.mps.synchronize()` **before** each timer stop — the MPS analogue of 0.5's readback rule. Keep `WARMUP_IMAGES = 50`. **Honest scope, on the row:** torch cannot run this Q4_K_S checkpoint, so the comparable surface is the **matvec shape** — `torch.mm` at f16 on `[1,4096]×[4096,4096]` (attn_q/o), `[1,4096]×[4096,14336]` (ffn_up/gate), `[1,14336]×[14336,4096]` (ffn_down) on MPS. This is a **roofline companion** ("what a tuned framework achieves at these shapes on this silicon"), **not** a decode competitor, and the row says so. All runs welded through the `flock` [G6].
- **expect** N = 3 shapes × 5 runs = **15 rows**, plus the mnist lane at 2 devices × 5 runs, reporting p50/p95/p99/mean/CoV. The harness must assert `torch.backends.mps.is_available()` **and** `next(model.parameters()).device.type == "mps"` or the arm silently ran on CPU and exits 0. **N==0 is RED.**
- **predict (nano → micro)** mnist batch=1 on MPS is **slower** than CPU on this box (per-op MPS dispatch dominates a 14-node graph at batch 1), and the MLP train step at batch ≥64 is faster on MPS. Both directions are the result; the loss is reported first (§19).
- **kill** MPS silently falls back ⇒ the arm is void; report it as a gap in torch's own harness, never as our win.
- **memory gate** the row records each arm's peak RSS via `/usr/bin/time -l` and `torch.mps.current_allocated_memory()`, because a torch arm that swaps invalidates every interleaved cell sharing the box. **KILL** at 2.5 GB RSS for the probe process (the 0.5 probe-cap precedent), stated as a deliberate exemption from the decode caps.
- **rollback** python-only; revert the flag; the venv is gitignored by 0.10.
- **blast** two python fixture files; zero Rust.
- **observe** p50/p95/p99 + CoV per arm; the device assertion; peak RSS.
- **reprove** `flock … -c 'proxima-onnx/scripts/torch_reference/venv/bin/python inference_bench.py --device mps --runs 200'`
- **log-row title** `the torch-MPS cell that did not exist: our matvec shapes as a roofline companion, and what batch=1 costs on a GPU`

### 8.2 — ORT-CoreML arms `[S 8.4, B2 R02-C, crit SD-4]`
- **tier** worker · **depends_on** 0.10
- **worktree** `…/proxima-wt-risc42` · `gpu-risc/42-ort-coreml-arms` · own target
- **opens** `scripts/onnx_reference/bench.py:96` — verified: `providers=["CPUExecutionProvider"]` hardcoded; `run.sh` (pinned venv `$HERE/.venv`, `ONNX_REF_PYTHON=python3.12`); the fidelity fields `cosine_similar` / `cosine_dissimilar_a/b` at `bench.py:82-85`; `export_model.py`. `onnxruntime` is **not installed** (R0) — extend this harness rather than creating a second venv (§1).
- **commands** **every command here, including the builds, is welded through the `flock`** [G6, crit SD-4] — a full onnxruntime C++ build saturates the box and must not run beside a measurement:
  ```
  flock …/.gpu-measure.lock -c 'scripts/onnx_reference/.venv/bin/pip install onnxruntime'
  # if no network:
  flock …/.gpu-measure.lock -c 'cd ~/repos/others/onnxruntime && ./build.sh --config Release --use_coreml --build_wheel --parallel'
  flock …/.gpu-measure.lock -c 'BGE_MODEL_PATH=… ONNX_REF_PROVIDER=coreml bash scripts/onnx_reference/run.sh'
  ```
  Add `--providers` (default `CPUExecutionProvider`, preserving today's arm byte-for-byte) threaded into the `InferenceSession` call at `:96`. **Honest scope, on the row:** the only ONNX model exported here is BGE-small, not a 7B decode — "ORT-CoreML cell exists at the BGE-small embedding shape; **no comparable micro surface** for 7B Q4_K decode; e2e compare REQUIRED before any ORT verdict."
- **expect** N = BGE-small × 2 providers × 5 runs with the fidelity fields per provider **or** one explicit `FEATURE GAP: CoreMLExecutionProvider unavailable/rejected the graph` row carrying the ORT error text. Assert `session.get_providers()[0] == "CoreMLExecutionProvider"` **and** the **partition node count** (a 1-node CoreML partition beside 200 CPU nodes is a CPU cell wearing a CoreML label). **A missing arm is RED; a documented gap is green.**
- **predict (nano → micro)** the CoreML EP takes a **partial** partition (<100% of nodes) and ms/sentence lands within **2×** of the CPU EP with the fidelity fields unchanged (fp32 path).
- **kill** fidelity drift (cosine similar/dissimilar move) ⇒ CoreML chose fp16; that is a **different arm** and must be labelled as one (§14). Wheel unavailable for the pinned interpreter **and** the source build also fails ⇒ the cell is a documented **feature gap**, never omitted (§19: an omitted loss is a verdict).
- **memory gate** the build's peak RSS is recorded via `/usr/bin/time -l`; the lock is what protects other cells. **KILL** if the build's peak forces swap (recorded, and the box is re-sealed before the next measuring card).
- **rollback** revert the flag; the venv is gitignored.
- **blast** `scripts/onnx_reference/bench.py`, `run.sh`; zero Rust.
- **observe** ms/sentence, CoV, provider partition node counts (`sess_options.log_severity_level=0`), fidelity fields, peak RSS.
- **reprove** the third command above.
- **log-row title** `the ORT-CoreML cell that did not exist, the partition that explains it, and the surface we cannot compare on`

### 8.3 — Our own GPU arms for the non-decode lanes `[S 8.5, crit j — retiered hands → worker]`
- **tier** worker · **depends_on** 0.5, 7.5
- **worktree** `…/proxima-wt-risc43` · `gpu-risc/43-nondecode-gpu-arms` · own target
- **opens** `omega/benches/metal_vs_cpu.rs` — the **only** GPU bench outside decode (gemm_square_f32 512/1024/2048; matvec_batch1_f32 at Mistral f32 shapes), registered `omega/Cargo.toml:207-210` with `required-features = ["metal"]`, doc says **UNRUN** (R9); `omega/tests/training_step_parity.rs:400-607` (GPU train step, untimed); 0.5's timing-window rule (`GPUStartTime`/`GPUEndTime`, readback outside).
- **commands** `flock … -c 'cargo bench -p omega --bench metal_vs_cpu --features metal'` — run it for the first time; then add a timed arm **around the existing parity fixture** rather than writing a new workload (§1: the fixture is the workload, it just has no timer). **Choosing the timing window is a design decision, which is why this is a worker card** [crit j]: the window is `GPUStartTime`→`GPUEndTime` with readback outside, identical to 0.5's rule, and the card states that explicitly rather than leaving it to Luna.
- **expect** N = 4 bench arms × 5 runs + ≥1 timed train-step cell. **N==0 is RED** — a `required-features`-gated bench that is never invoked compiles to nothing, and a bench that has never run may not compile.
- **predict (nano → micro)** `matvec_batch1_f32` at Mistral shapes lands under **25%** of 0.5's measured streaming ceiling (naming the `read_only_gbs` column) — the low-simdgroup starvation R3/M5 names (MEMORY: 52 GB/s at 256 simdgroups rising to 147 at 8001; the prediction is anchored on 0.5's MEASURED ceiling, not on the MEMORY curve); `gemm_square_f32` at 2048 is the widest Metal-over-CPU margin.
- **kill** the bench does not build under `--features metal` ⇒ that is the finding, and fixing the registration is the card; a repo that does not build its own benches is our bug.
- **memory gate** bench arms allocate GEMM operands up to 2048² f32 ×3 = 50 MB — well under the probe caps; asserted before allocating, and the decode caps are explicitly not applied. **KILL** at the probe caps.
- **rollback** revert the timed arm. **blast** `omega/benches/metal_vs_cpu.rs`, `omega/tests/training_step_parity.rs`.
- **observe** GB/s and GMAC/s per arm against 0.5's ceiling (naming the denominator column), with 0.1's corrected byte accounting.
- **reprove** the `cargo bench` command above.
- **log-row title** `the non-decode GPU lanes get their first numbers; matvec at batch 1 against the measured ceiling`

---

# PHASE 9 — the log, ai_docs, and the final board

### 9.1 — Discipline rows, rooflines, and the ai_docs JSONL records `[S 9.1, B2 R07]`
- **tier** hands · **depends_on** 0.13, 0.7, 1.1, 0.5
- **worktree** `…/proxima-wt-risc44` · `gpu-risc/44-docs-and-aidocs` · own target
- **opens** `proxima-tensor/docs/discipline.md:18736` (**ROW 233** is main's last); `rooflines.md:396-479`, `:411` (GPU ceiling = DEBT), `:751` (summary row), `:766-773` (the closing note "the GPU lane's ratio is not a gap-to-machine at all" — now answerable); `ai_docs/AGENT.md` (Update Rule: kind=5 decisions, kind=7 failures, `relations.idx=7` for grounding; step 5 says **add records**, never bypass); `ai_docs/{index,task-routes,invariants}.jsonl` (R0: **zero** tensor/omega/GPU records).
- **commands** (a) renumber every `<PH-NN>` placeholder sequentially from `grep -n '^## ROW' … | tail -1` per 0.13's protocol; (b) replace `rooflines.md:411`'s DEBT with 0.5's measured ceiling in **both** denominators, update `:751`, answer `:766-773`; (c) land the records:
  - **index**: `proxima.omega.gpu_decode_lane`, `proxima.omega.kernel_route_census`, `proxima.tensor.write_placement`.
  - **task-routes**: `gpu-lane` with `done_when` = ["a home-turf llama.cpp-Metal arm is on the row", "the N contract (F>=2, S>=1, op_count>0) is asserted", "both memory slopes and both caps are recorded", "every geometry constant traces to omega-runtime.toml", "the route census sum equals ENCODE_DISPATCH_CALLS"]; `omega-kernel-emission`.
  - **invariants** (kind 7, each with `evidence_required`): `proxima.gpu.one_bound_plan` (evidence: 7.7's gate, equal fingerprints, `grep -c correct_packed_matmul_layouts == 0`); `proxima.gpu.route_is_a_value` (evidence: the census sum identity **and** the measured ≤5% dispatch-slice cost; the defect at `metal.rs:785-826`; R12 ROW 263's 9/601 → 225/385); `proxima.omega.backend_covers_every_kind` (evidence: `grep -c ScatterNotSupported == 0` in the reachable paths, `grep -c CudaUnsupportedOpKind == 0`, the 15-cell matrix); `proxima.omega.geometry_traces_to_sizing_config` (evidence: 1.3's grep); `proxima.gpu.dispatch_count_is_not_the_denominator` (evidence: R12 1194→616 wall 51.571→51.535, and 6.2's second test); `proxima.gpu.profile_bytes_are_tensor_bytes` (evidence: 0.1, against R13's 4,140,417,024 and 4,147,777,096); `proxima.gpu.memory_is_a_kill_criterion` (evidence: G8's formula, both slopes, both caps, per steady step).
  ```
  for f in ai_docs/index.jsonl ai_docs/task-routes.jsonl ai_docs/invariants.jsonl; do jq -c . "$f" >/dev/null && echo "$f OK $(wc -l < $f)"; done
  bash ai_docs/query.sh gpu-lane
  ```
- **expect** index +3, task-routes +2, invariants +7 = **N == 12 new records**; `jq` parses all three (a malformed line is RED); `grep -c "^## ROW <PH-" discipline.md == 0`; the last row number monotonically greater than 233; `bash ai_docs/query.sh gpu-lane` returns ≥1 row. **N==0 on any file is RED.**
- **predict (nano → micro)** `grep -ci "omega\|metal\|gpu" ai_docs/task-routes.jsonl` moves from **0** to ≥2, and the route query returns the invariants a future agent must read before touching the lane.
- **kill** malformed JSONL, or `query.sh gpu-lane` returning nothing ⇒ the schema assumption is wrong; read `query.sh` and fix the record shape rather than bypassing the index (AGENT.md is explicit).
- **memory gate** N/A — docs and JSONL only; stated with this rationale.
- **rollback** `git revert`; docs and `ai_docs/` only. **blast** `discipline.md`, `rooflines.md`, three JSONL files. Zero source.
- **observe** record counts per file; query hit count; row monotonicity.
- **reprove** the two commands above.
- **log-row title** `main's log learns the GPU session happened, the roofline DEBT is paid, and ai_docs gains its first tensor/omega records`

### 9.2 — The final board `[S 9.2, B2 R07]`
- **tier** hands · **depends_on** 6.2, 5.3, 7.7, 8.1, 8.2, 8.3, 9.1
- **worktree** `…/proxima-wt-risc45` · `gpu-risc/45-final-board` · own target
- **commands** `flock … -c 'bash scripts/gpu-seal.sh $PWD $CARGO_TARGET_DIR std,metal,instrument 5'` with every landed feature in `default`, on a box whose loadout is recorded.
- **expect** every board cell filled: ours (`step_wall_ms`, `gpu_exec_ms`, CoV, n), llama arm A, llama arm B (`-fa 1`), torch-MPS, ORT-CoreML (or its documented gap), roofline fraction naming 0.5's denominator column, and both memory slopes with both caps. **A blank cell is RED.**
- **predict (milli → bench)** 4.3's band **adjusted only by the milli-rung deltas 5.3, 5.4 and 6.2 actually measured**, each with its CoV. This card does **not** re-predict from theory; Phases 5–6 contributed nothing to 4.3's original derivation by design (R12's null).
- **kill** the ratio does not fall below **3.0x** against 0.4's chosen incumbent arm ⇒ R13's decomposition is wrong somewhere, and the row names **which bucket did not move, by counter**.
- **memory gate** full G8 at the landed `capacity_tokens`; a breach is a board-level NEGATIVE and demotes the offending feature before the board is written. **KILL.**
- **rollback** n/a (measurement). **blast** docs.
- **observe** every counter in the board + the per-route census + fraction-of-ceiling + `phys_footprint_bytes` / `device_allocated_bytes` slopes.
- **reprove** the seal command above.
- **log-row title** `the board, every cell filled, against R13`

---

# PHASE 10 — residual levers, each gated on a re-measure

*§18: a claim from a stale tree is not a claim. Every item here was reasoned about before Phases 1–7 changed the tree.*

### 10.1 — A config sweep before a split-K kernel `[S 10.1, B2 R16-predict]`
- **tier** hands · **depends_on** 1.3, 2.4, 6.2
- **worktree** `…/proxima-wt-risc46` · `gpu-risc/46-packed-geometry-sweep` · own target
- **opens** `omega/src/msl.rs:1550-1552` (the packed arm's `div_ceil(PACKED_ROWS_PER_GROUP)*SIMD_WIDTH` simdgroup count); 1.3's `[packed_row_block] rows_per_group`; R13's family table (`attn_q` 5.172 ms at 58.5 GB/s, `attn_output` 78.6, `attn_v`/`attn_k` 50–53, vs `ffn_*` 97–109) — the curve R3/M5 names (MEMORY: 52 → 147 GB/s from 256 → 8001 simdgroups; the anchor here is R13's MEASURED family table, not the MEMORY curve).
- **commands** sweep `OMEGA_PACKED_ROW_BLOCK_ROWS_PER_GROUP ∈ {1,2,4,8}` × 3 runs = **12 cells**, interleaved round-robin (never blocked by value), each welded with `cd` and `flock`. A config sweep is cheaper than a new kernel route and may retire the lever (§1 applied to geometry).
- **expect** 12 cells with per-family GB/s (0.1's corrected bytes) and `attn_*` op counts unchanged. **N==0 is RED; a moved op count is RED** (the route changed, not the geometry).
- **predict (nano → micro)** on `metal_vs_cpu.rs`'s `matvec_batch1_f32` arm, `rows_per_group = 8` **halves** the simdgroup count and is **10–30% slower**; a value below 4 is faster on the low-row families by **≥5%**. If 8 is *faster*, M5's mechanism is wrong and 10.2 must not be built.
- **kill** no knob value beats 4 by more than the measured CoV ⇒ **do not build split-K**; row the negative with all 12 numbers. Note nsg=2 regrouping is a **four-time** negative (R4, `perf/metal-simdgroup-geometry`, R12 ROW 267, R12 ROW 259/260) and is a **different mechanism** from split-K; re-proposing nsg=2 is a discipline failure, not an experiment.
- **memory gate** build-time config only; `device_allocated_bytes` unchanged in slope and absolute; any increase is a NEGATIVE. **KILL.**
- **rollback** build-time config; revert to 4. **blast** build config only.
- **observe** per-route and per-family GB/s from the census; `gpu_exec_ticks`.
- **reprove** the 12-cell sweep.
- **log-row title** `the simdgroup-starvation hypothesis, tested with a config knob before a kernel`

### 10.2 — Split-K for the starving low-row shapes — the successor card, with a delete-the-card gate `[B2 R18, crit SD-3, MS-6, h]`
- **tier** worker · **depends_on** 10.1, 1.1, 6.2, 10.3
- **worktree** `…/proxima-wt-risc47` · `gpu-risc/47-packed-split-k` · own target
- **opens** `omega/src/msl.rs:1550-1552`, `:2452-2530`; R13's family table; 0.6's `splitk-tracked.patch` and `lat-tracked.patch` (R7: `perf/q4k-split-k` 5 files +352/−48 with a `metal-q4k-split-k` feature — measured-adjacent work that 0.6 quarantined and this card is the only one that opens); 1.3's `[packed_row_block]` section.
- **entry gate — this card is DELETED if it does not fire** [crit SD-3, h]: after 6.2 and 10.3, the route census + family table must **still** show `attn_q`/`attn_k`/`attn_v`/`attn_output` achieving **under 70%** of the `ffn_*` families' GB/s on the same body. If 2.4's winning body already closed the gap, **this card is deleted and the row records why, with the numbers** — a lever that is no longer needed is a negative result worth writing down, not silent scope. A measured **win** at 10.1 and a measured **loss** at 10.1 now both have a successor.
- **commands** split the reduction axis K into `split_k` partitions, each producing a partial, followed by a cheap combine; `split_k` traces to `[packed_row_block].split_k` (1.3's section, §12). Feature `metal-packed-split-k` default-off + forwarded. Sweep `OMEGA_PACKED_ROW_BLOCK_SPLIT_K ∈ {1,2,4,8}` × 3 runs, interleaved.
- **expect** 4 arms × 3 runs = **12 rows**; `attn_*` op counts unchanged (a split that changes the op count changed the route — RED); **bit-reproducibility**: parity vs `cpu::evaluate` on real `blk.0.attn_q.weight` at every split value, tolerance stated at 1e-6 because a split changes summation order, **and the generated token bit-identical**; the same fixture run 100× byte-identical (a non-deterministic partial-sum combine is a §14 hazard). **N==0 is RED.**
- **predict (nano → micro)** on `metal_vs_cpu.rs`'s `matvec_batch1_f32` Mistral arm, `split_k = 4` raises achieved GB/s at the 4096-row shape from ~58 toward the ffn families' ~97, i.e. **[85, 105] GB/s**.
- **kill** the combine's cost exceeds the split's win (visible as the elementwise op count rising and the net flat) ⇒ dead lever, negative row, all four numbers recorded. Any non-determinism across the 100 runs.
- **memory gate** partials are `split_k` extra intermediate buffers per matvec, allocated **through 4.2's arena**, so the arena's peak grows by `split_k × attn_output_bytes`. **Compute and print it before enabling**; an arena peak above G8's 40 MB activation term is a NEGATIVE and rolls back. **KILL.**
- **rollback** default-off feature; omit `--features metal-packed-split-k`.
- **blast** `omega/src/msl.rs` packed body + `grid_threads` packed arm, behind a feature. No graph change, no driver change.
- **observe** `op_profile_family` GB/s for the four `attn_*` families (true bytes from 0.1); `ARENA_PEAK_BYTES` (4.2); the route census.
- **reprove** the sweep at the landed `split_k`.
- **log-row title** `split-K for the starving low-row attention shapes — or: the census says the body already closed it, and here is the number that deleted this card`

### 10.3 — The cooperative-reduce width, re-swept against the collapsed graph `[B2 R17]`
- **tier** hands · **depends_on** 3.1, 6.2, 1.3
- **worktree** `…/proxima-wt-risc48` · `gpu-risc/48-reduce-width-resweep` · own target
- **opens** `omega/src/msl.rs:1517-1560` `grid_threads`; 1.3's `[cooperative_reduce]`; `ggml-metal.m:3797-3804` (nth doubles 32 → `min(ne00/4, maxTotalThreads)`, R8).
- **commands** a sweep, not a change: `OMEGA_COOPERATIVE_REDUCE_MAX_THREADS ∈ {32,64,128,256,512,1024}`, six arms × 3 runs, interleaved round-robin, each welded with `cd` and `flock`. **The reduction lengths changed when attention collapsed, so 3.1's tuning was against a graph that no longer exists.**
- **expect** 6 arms × 3 runs = **18 rows**; `reduce-cooperative` op count constant across arms. **N==0 is RED.** Monotonic-then-flat is expected; a non-monotonic curve is the finding and gets its own row.
- **predict (milli → bench)** the optimum is `min(reduction_len/4, 1024)` per the incumbent's rule (R8) — 1024 for the 4096-wide rms_norm rows; `reduce-cooperative` lands at **≤60%** of its post-6.2 value and `step_wall_ms` improves by **[1.5, 3.0] ms**.
- **kill** no value beats 32 by more than 2× CoV ⇒ the lever is dead on the new graph; record the negative with all six numbers and demote 3.1's feature permanently.
- **memory gate** threadgroup memory scales with width; `device_allocated_bytes` unchanged; an increase is a NEGATIVE. **KILL.**
- **rollback** one integer in `omega-runtime.toml`. **blast** one TOML integer.
- **observe** `op_profile_bucket kind=reduce-cooperative gpu_ms` and `gpu_ns_per_op`.
- **reprove** the sweep at the landed value.
- **log-row title** `the cooperative-reduce width, re-swept against the collapsed graph (the old tuning was for a graph that no longer exists)`

### 10.4 — Re-measure the rematerialization subset `[S 10.2]`
- **tier** hands · **depends_on** 6.2
- **worktree** `…/proxima-wt-risc49` · `gpu-risc/49-remat-remeasure` · own target
- **opens** R3/M8 (MEMORY): only the `elements < 247` subset (96 nodes) was a certain win (1196 → 1100); the aggregate set was a loss; "rematerialize all ≤2-consumer nodes" is a **dead lever** (R4, 6× downside on the slow ALU arm) and is not re-proposed.
- **commands** re-run the subset census on the post-6.2 graph **before** building anything.
- **expect** N = the post-6.2 count of `elements < 247` nodes. **N==0 is RED** — and N==0 would mean 6.2 already removed them, which is itself the finding and closes the item.
- **predict (nano → micro)** the subset shrinks by more than half after 6.2 (most were attention-graph intermediates), and the remaining win is under 1% of R13's 67.92 — below the noise floor at 0.3's measured wall CoV, which retires the lever.
- **kill** the predicted win is < 2× the measured CoV ⇒ row it as "no signal, kept the simpler form" and do not build. R12's null is the standing reason: a 578-dispatch reduction moved 0.036 ms, so a 96-dispatch reduction cannot matter.
- **memory gate** census only; slopes unchanged. **KILL** at G8 for the harness run.
- **rollback** n/a if not built. **blast** `proxima-tensor/src/bind.rs` if built.
- **observe** node census by element count; `encode_dispatch_calls`.
- **reprove** `flock … -c 'cargo run --release -p omega --features metal,instrument --example real_forward_emit_probe'`
- **log-row title** `rematerialization re-measured on the post-placement graph`

---

# Dependency graph

```
0.1 instrument bytes ─> 0.2 gpu-seal.sh ─┬─> 0.3 baseline + CoV/memory band ──────────────────┐
                                          ├─> 0.4 llama -fa 1 arm  (DENOMINATOR)              │
                                          ├─> 0.5 streaming roofline (CEILING)                │
                                          └─> 0.6 quarantine ─> 0.7 adjudicate ─> 0.8 prune_dead ─> 0.9 consumer index
0.3 ──────────────────────────────────────────> 0.12 harness N-contract + plan_hits formula   │
0.8 ─> 0.11 RISC cardinality ─> 0.14 fingerprint (PRE-REGISTERED RED, R16 / metal.rs:1013)    │
0.10 clean tree ─> (every interleaved measurement) ; ─> 8.1, 8.2                              │
0.13 row-number protocol                                                                       │
                                                                                               │
0.14 + 0.3 ─> 1.1 Route (lock-free census, 5% budget) ─┬─> 1.2 census gate ────────┐          │
                                                        └─> 1.3 geometry config ────┤          │
                                                                                    v          │
                              1.2 + 1.3 + 0.6 ─> 2.1 mask-fma ─> 2.2 pair-dot ─> 2.3 bake-off ─> 2.4 default
                                                                                          │
                          1.3 ───────────────> 3.1 wide cooperative reduce <──────────────┤
                                                     ├─> 3.2 elementwise census           │
                                                     └─> 3.3 re-seal <────────────────────┴── 0.4 + 0.5
                                                              │
                                          0.12 ─> 4.1 kv bucket + tail mask (inverts the assertion)
                                                              │
                                                    4.2 arena (peak sum first) ─> 4.3 BOARD PREDICTION
                                                              │                     ^        ^
                                                              │                     └── 0.4 ──┘ 0.5
                                        4.3 ─> 5.1 scatter expression ─> 5.2 MSL scatter emitter ─┬─> 5.4 argmax
                                                              4.1 + 4.2 + 5.2 ─> 5.3 KV device-resident write
                                                                                       │
                                                              6.1 ops/layer census ─> 6.2 single-range
                                                                                       │
        1.1 + 5.2 ─> 7.1 Dialect design ─> 7.2 CUDA kinds ─> 7.3 core:Elementwise ─> 7.4 core:Reduce ─> 7.5 core:Iota/Const/Scan
                                                                                       │
                                                              7.5 + 0.14 ─> 7.6 bind owns packed layout ─> 7.7 fingerprint gate
0.5 + 7.5 ─> 8.3 non-decode arms ; 0.10 ─> 8.1 torch-MPS, 8.2 ORT-CoreML
0.13 + 0.7 + 1.1 + 0.5 ─> 9.1 docs/ai_docs ─> 9.2 FINAL BOARD <── 6.2, 5.3, 7.7, 8.1, 8.2, 8.3
1.3 + 2.4 + 6.2 ─> 10.1 geometry sweep ─> 10.2 split-K (entry-gated, deletable) <── 10.3
3.1 + 6.2 + 1.3 ─> 10.3 width re-sweep ;  6.2 ─> 10.4 remat re-measure
```

**Hard edges, in prose, agreeing with the picture** [crit O-4]. **0.3 is an ancestor of every card whose kill quotes "the CoV band"** — 1.1, 2.3, 3.1, 4.1, 4.2 all depend on it directly or through 0.12/2.4, and no kill in this plan names a band no edge guarantees exists. **0.4 and 0.5 are ancestors of 3.3, 4.3 and 9.2** — the plan's only board-level prediction (4.3) is made against a settled denominator and an existing ceiling, never against one the plan schedules for possible invalidation later [crit b, O-2]. **0.8 is an ancestor of 0.14, 1.1, 2.4, 6.1 and 7.5** and publishes `OPS_AFTER_PRUNE` as a measured number, so no downstream card carries the undecidable expression "1196 minus the pruned count" [crit O-5]. **1.2 precedes 2.1** — the census is a hard precondition of any body swap. **1.3 precedes 3.1, 10.1 and 10.3** — the width and the geometry must be config keys before they are swept. **4.1 precedes 4.2** (a pool refilled every token is not a pool) and **4.1 precedes 5.3** (the bucketed leaf extent is what makes `found == expected` hold with no validator change). **5.2 precedes 5.3 and 5.4** and depends on 5.1 only; there is **no conditional dependency anywhere in Phase 5** — the affine form was abandoned, so 5.3 and 6.2 have an unconditional path [crit O-3]. **6.2 requires 5.3, 5.2 and 6.1.** **7.3–7.5 require 7.1's classification and 1.1's route.** **7.6 requires 7.5 and 0.14**, and **7.7 requires 7.6.** **10.2 requires 10.1 and 10.3 and is deleted if its entry gate does not fire.**

**Critical path to the board prediction:** 0.1 → 0.2 → 0.3 → 0.4 → 0.5 → 0.6 → 0.7 → 0.8 → 0.11 → 0.14 → 1.1 → 1.2 → 2.1 → 2.2 → 2.3 → 2.4 → 3.1 → 3.3 → 4.1 → 4.2 → **4.3**. Everything in Phases 5–8 and 10 is **off** that path and is sequenced for correctness, one-RISC conformance, and cells that do not exist — not for predicted wall movement.

**There is no parallel phase.** Every card whose commands run the decode harness, a probe, a bench, or a C++/python build is serialized by G6's `flock`. Non-measuring cards (0.6, 0.7, 0.10, 0.13, 7.1, 9.1) may proceed concurrently only while no build runs on the box.

---

# Rollback map

| card | rollback | main's default affected before rollback | firewall |
|---|---|---|---|
| 0.1 | `git revert` | yes (instrument only) | the sum identity N1 + the 9-family table vs R13's derived column |
| 0.2, 0.3, 0.4 | revert script / `worktree remove` | scripts only / no | the reproduction band + the memory band |
| 0.5 | `git revert` | example only | `readback_bytes == 0` inside the window; two denominators |
| 0.6 | `git revert` — **patches and untracked contents remain in git history**, which is the point | docs only | never `/tmp`; per-worktree HEAD recorded |
| 0.7, 0.13, 9.1 | docs/JSONL revert | no | n/a |
| 0.8 | `git revert` 1 commit | yes (no flag; bisect by revert) | census names every removed node; `gpu_exec` move < 0.2 ms; `device_allocated_bytes` must not rise |
| 0.9 | `git revert` 1 commit | yes | `generated_text` identity |
| 0.10, 0.11, 0.12 | `git revert` | ignore file / doc+tests / test-only | exhaustive matches fail to compile on a variant change; the assertion still fires on an injected hit |
| 0.14 | `git revert` | test + one pure fn | **the RED is the expected state**; the field list is fixed here and not negotiable downstream |
| 1.1 | `git revert` — **multi-commit unwind once 7.3–7.5 land; revert those first** | yes (`route::of` on the default path; counters gated) | `kernel_cache_key` byte-stability + golden emitted source + the measured ≤5% dispatch cost, all in THIS card |
| 1.2, 7.7 | revert the gate step | gate only | n/a |
| 1.3 | revert build.rs + toml + const sites together | yes, value-identical by construction | per-key equality + env-override-changes-MSL tests |
| 2.1, 2.2 | `git reset --hard`; delete the feature | no | default-off. **Not symmetric:** A is a patch in git from 0.6; **B is a commit on `perf/q4k-independent-accumulators` and is restored by cherry-pick, not by 0.6** [crit RB-2] |
| 2.3 | drop both features | no | parity first; the route-count pin voids incomparable arms; the tie-break is terminal |
| 2.4 | demote out of `default` (one line), then revert | yes | full parity suite; `encode_dispatch_calls == OPS_AFTER_PRUNE` |
| 3.1 | feature default-off; revert | no while gated | `metal_parity` / `backend_parity`; the count must not move |
| 3.2 | `git revert` | instrument only | cold-path only, so 1.1's budget is not re-spent |
| 3.3, 4.3, 9.2 | n/a (measurement); features demote one line each | yes | the memory gate is a board-level kill |
| 4.1 | `git revert`; **feature-off PLUS A REBUILD — the bucket is a build-time const, decided on the card** [crit RB-3] | yes | `#[cfg]`-paired assertions keep both arms green; feature-off **fingerprint identity**, not an assertion; CPU 0-ULP parity; `2651`/`"known"` |
| 4.2 | feature default-off; revert | no while gated | pooled-vs-unpooled identity; **the retirement-order test against `bound_op_retirement`** [crit RS-3]; the peak sum printed before allocating |
| 5.1 | `git revert` | test-only | n/a |
| 5.2 | `git revert`; feature off ⇒ `ScatterNotSupported` fires exactly as today | no | the CPU doc's worked example + 0-ULP KV-shape parity + 100× determinism; collisions declined by name |
| 5.3 | two independent default-off features; either reverts alone | yes | **`found == expected` by construction — no validator was edited**; pre-allocation size print; text identity; both slopes → 0 |
| 5.4 | `git revert` the sampling commit alone | yes | sampling path still receives full logits (asserted) |
| 6.2 | `git revert`; two-range body stays under `#[cfg(not(...))]` | yes — every model using `append_mistral_cached_layer` incl. qwen3.5 | feature default-off; text identity; wall in the kill; split-half RoPE explicitly out of scope |
| 7.1 | docs revert | no | the 78-row classification is the gate on 7.3 starting |
| 7.2 | revert the variant-deletion commit, then the implementation | yes | two commits, not one |
| 7.3, 7.4, 7.5 | `git revert` per kind, in reverse order 7.5 → 7.4 → 7.3 | yes, byte-identical by construction | the golden from 1.1 + the `(Backend, Route)` exhaustive match |
| 7.6 | revert the deletion commit, then the bind commit; 0.14 returns to its documented RED | yes — `layout_of` touches every backend's plan | golden byte-identity + all six Q4_K/Q5_K/Q6_K parity suites |
| 8.1, 8.2, 8.3 | revert the flag / the timed arm; venvs gitignored | no Rust hot path | device assertions; partition counts; fidelity fields |
| 10.1, 10.3 | one TOML integer | build config | negatives rowed with all cells |
| 10.2 | default-off feature | no | the entry gate deletes the card; bit-reproducibility over 100 runs; arena peak printed |

Every landing commit is a green bisect point; primitives land before callers (1.1 before 7.3–7.5; 1.3 before 3.1, 10.1 and 10.3; 4.1 before 4.2 and 5.3; 5.2 before 5.3 and 5.4; 0.14 before 7.6). The mechanism that makes this true for behaviour-changing cards is that each ships a default-off feature **and** any assertion the feature invalidates is `#[cfg]`-paired, so both arms compile and both pass — 4.1 is where this matters most and says so on the card. **No commit lands without owner authorization.**

---

# Abandoned designs (traced)

1. **`BoundOpKind::CachedAttention`** — a fifth bound kind carrying an eight-input fused online-softmax macro-op, with the post-bind structural matcher (`cached_attention_candidates`, `attention_score_sources`, `is_exact_causal_mask`, `removable_attention_dependencies`) and `physical.rs` (+576). **Ruled out by** workspace AGENTS.md's hard invariant against arbitrary rules for specific instances against a closed 4-variant set (`bind.rs:221-264`), and by §1's binary question — `IndexMap::scatter` expresses it, so the expression is written (5.1) rather than a kind minted. Corroborating: the branch's own `failure-cached-attention-matcher.md` abandoned the first matcher as a heuristic that "cannot prove the semantic roles", and its numbers show wall unmoved (51.535 vs 51.571) with `gpu_exec` **worse** (39.841 vs 35.117), while the matcher itself cost `prepare` 150.7 ms/token before indexing. **What changed:** the plan attacks the graph (4.1 + 5.2 + 5.3 + 6.2) instead of pattern-matching the graph's defect after bind — the single largest way the constraints reshaped the design. **What survives:** `prune_dead` (0.8), the consumer index (0.9), the paired Q4_K body (2.2), and their ROW 263 as 1.1's third witness. **Re-open condition:** 6.2 measuring that ≤23 ops/layer is unreachable through scatter placement.
2. **`Op::Concat` / `Op::Pad` / `Op::Tile` / `PlacedBuffer` / `write_placement`** — zero hits on main, and none added. **Ruled out by** §1 plus §6: `IndexMap::scatter` (`map.rs:175`), `out_scatter` (`bind.rs:253`) and `run_reduce_scatter` (`cpu.rs:6911`) already express **and execute** write placement today on CPU, with the worked example in the function's own doc. **What changed:** 5.2 became an *emitter* card — Metal implements what the IR already models — so `proxima-tensor` is untouched by it and the blast radius of write placement is one file.
3. **Relaxing `project_output_shape` to accept an affine write offset (the round-1 synthesis's 5.4).** **Ruled out by** a closed prior adjudication read verbatim on main: `map.rs:118-124` states "see `shape.rs`'s `infer_reduce` doc for why a `Reduce`-wide field was rejected on blast-radius grounds" and records that the scatter `offset`-at-`gathered_dim` convention was chosen instead. Re-opening `shape.rs:469-485` re-litigates that decision and changes bounds semantics for **every existing `Reduce`** plus the autograd adjoint path (`map.rs:238`) — the one place round 1's judges named where wrong-but-green was possible. **What changed:** `shape.rs` is never edited, and that fact is itself the proof on 6.2's row that the constraint was routed around rather than weakened.
4. **A `PlacedBuffer` type in the Metal driver.** **Ruled out by** the relocation question, run: `metal.rs:2249` is already `device_buffers.insert(bound.node, (output, 0))` over `BTreeMap<NodeId,(MetalBuffer,usize)>`; aliasing is `insert(node, (persistent, offset))` — identical lines ⇒ a relocation. Plus `AlignedBuffer` (`align.rs:69`) exists with zero production callers; a peer beside an unused primitive is debt twice over, so 5.3 gives it its first caller instead.
5. **A parallel `&[Option<u64>] declared_capacity` slice threaded into both validators (the round-1 synthesis's 5.5), and a new field on `QuantizedBlock`.** **Ruled out by** three findings at once: the predicate `found != expected && Some(found) != declared[i]` is **unsound** — `declared` comes from the same caller as the block, so it degenerates to "accept any length, including a truncated one" [crit RS-2]; the slice must be built in **node** order while `named_blocks` is built in **name** order and `resolve_named_blocks` (`metal.rs:584`) does the reorder, so a positional desync produces an out-of-bounds index or an accepted wrong-size weight [crit MS-2, c]; and its unit was never stated against `AlignedBuffer`'s element-in/byte-rounded contract [crit HC-4]. **What changed:** 4.1's bucketing makes the KV leaf extent **equal** the buffer capacity, so `element_count(shapes.of(node)) == block_element_count(block)` holds and **neither validator is edited at all** — the contract needs no code, only an ordering. The page rounding is absorbed by deriving the capacity symbol from `buffer.len()` after `AlignedBuffer::new` rounds, which is what `align.rs:36-41` already requires of every caller.
6. **A `trait KernelBackend` / trait-object emitter registry** to unify the ~78 near-duplicates. **Ruled out by** §20 box-free and §11 (no trait objects). **What changed:** 7.1 enumerates a **seven-method `Dialect` consumed through a generic parameter**, monomorphised into a hot string builder, with everything structural — including `grid_threads` (arithmetic) and `reduce_is_cooperative` (a predicate) — in the core, and coverage as a **compile-time** exhaustive `match (Backend, Route)` rather than a runtime rejection.
7. **Threading `op_setup` / the encode loop.** **Ruled out by** §21 (a lock is a missing owner — the owner is the `Plan`), R3/M9 (non-`Send` `MTLBuffer` blocks it at the type level), and R4 ("thread count explains zero of the gap"). R13 shows the cost is 1196 `newBufferWithLength` calls, i.e. allocation, not serialism. **What changed:** 4.2 removes the work rather than distributing it; `PROXIMA_ORCH_THREADS` is **not** in this plan.
8. **A `Mutex<BTreeMap>` route census mirroring `WIDTH_TILE_DECLINE` verbatim.** **Ruled out by** §21 lock-free and by arithmetic: `instrument.rs:842` is `Mutex<BTreeMap<…>>` with `.lock()` at `:857`, and the proposed record site's own neighbour at `metal.rs:2243` is an atomic `Counter` (`:1484`); 1196 lock/unlock + BTreeMap lookups per token would sit **inside the very slice** 4.1–4.3 measure, guarded only by `gpu_exec_ms` (`metal.rs:547-554`), a device window that cannot see a CPU lock [crit RS-1, OB-1, a]. **What changed:** the census became a **per-plan route table filled once at plan time** (the route is a property of the plan, not of the dispatch — R16) plus a fixed-size atomic counter array on the dispatch path, with a **measured** ≤5%-of-`encode_dispatch_ms` budget and an ON/OFF arm.
9. **`cached_len` as a runtime uniform** so the bound plan is shape-independent. **Parked, not deleted**, by blast radius: `BoundOp.extents` is a baked `Vec<u64>` (`bind.rs:200-215`) and `grid_threads` (`msl.rs:1517-1560`) computes the dispatch grid **from** those extents, so a runtime extent makes the dispatch geometry runtime too. **Claim it gates:** a 100% plan-cache hit rate at every context length. **Un-park condition:** 4.1 measures a hit rate below `F − distinct_shape_count()`, or the tail mask costs more than the re-plan it replaces.
10. **Headlining the dispatch-count reduction** as the plan's spine, which the brief's one-line diagnosis proposed. **Ruled out by** R12's control (1194 → 616 moved wall 0.07% and moved GPU **up** 13.5%) plus R13's own per-op table (225 packed-row-blocked ops carry 44.450 ms while 547 elementwise carry 7.350). **What changed:** the entire phase ordering — dispatch count moved from spine to counter, and the spine became (instruments, route census, kernel body, plan stability, then graph).
11. **Rematerializing the low-element node set.** **Ruled out by** R4's dead-lever record combined with R12's null: a 578-dispatch reduction moved 0.036 ms, so a 96-dispatch reduction cannot matter. Retained only as 10.4, a re-measure with a kill at < 2× CoV.
12. **nsg=2 / ggml packed-simdgroup regrouping.** **Ruled out by** four independent measured negatives (R4, `perf/metal-simdgroup-geometry`, R12 ROW 267, R12 ROW 259/260). Re-proposing it is a discipline failure, not an experiment; 10.2 raises simdgroup count by **split-K** instead, and only if its entry gate fires.
13. **A kernel-fusion engine** as the route to parity. **Ruled out by** R8: `grep -rln fuse ggml/src` is **empty** at `b25346221`; rms_norm and mul dispatch as two kernels. Parity is reachable without fusion; fusion is upside past parity.
14. **A spec-sheet GPU bandwidth figure** to close the roofline debt. **Ruled out by** §18 (ASSUMED provenance may never anchor a mechanism claim) and by `rooflines.md:411`, which refused this once already. **What changed:** 0.5 is a real streaming-copy probe with readback outside the timed window, a `readback_bytes == 0` assertion, and **two** denominator columns.
15. **A second marker string to fix the classifier mislabel** (the parallel branch's ROW 263 fix). **Ruled out by** "find where information is destroyed": another substring is more of M10, the mechanism that caused the mislabel. **What changed:** 1.1 makes the route a value and deletes the substring buckets; their ROW 263 becomes the third witness on 1.1's row rather than a patch we inherit.

---

# Open questions resolved by measurement (never by asking)

| # | question | card | the number that answers it |
|---|---|---|---|
| Q1 | Are the profiler's bytes wrong in a second place besides the checkpoint buffer, and where did `block_upload_bytes` come from? | 0.1 | the three-way sum == 4,147,777,096; the 9-family table within 1% of R13's shape-derived column |
| Q2 | Does the box move R13's means, or only its CoV — and does memory move? | 0.3 | `step_wall_ms` vs 67.92, `gpu_exec_ms` vs 56.93, CoV vs 0.5%/0.7%, plus both memory slopes |
| Q3 | Is `-fa 1` a stronger incumbent, invalidating every ratio in the log including R13's 3.88x? | 0.4 | ms/token `-fa 0` vs `-fa 1`, interleaved, **before** any board prediction |
| Q4 | Is the machine's streaming ceiling above the incumbent's achieved rate — is there headroom at all, and in which denominator? | 0.5 | GB/s at 1 GiB, read-only column and traffic column, with CoV |
| Q5 | Is there really ONE bound plan across the executors? | 0.14, 7.6, 7.7 | the two fingerprints and the node-set diff — pre-registered to differ by exactly `packed_operands` |
| Q6 | Was `classify_kind` lying about the route distribution, and what does the census itself cost? | 1.1, 1.2 | census counts vs R13's 225/385/547/37/2; the sum vs `ENCODE_DISPATCH_CALLS`; `encode_dispatch_ms` ON−OFF ≤ 0.0235 ms |
| Q7 | Are `metal-q4k-mask-fma` and `q4k_pair_dot` the same mechanism at the same speed, and is the −36%/−29% one win double-counted? | 2.3 | summed family `gpu_ms` vs 2× pooled CoV with the route count pinned equal; then parity error; then source lines; then the terminal rung |
| Q8 | Do the 2026-09-02 numbers survive a rebase onto nine commits of main? | 2.1, 2.2, 2.3, 3.1 | re-earned against 0.3's cell, not carried |
| Q9 | Does bucketing `cached_len` reach `plan_hits > 0`, and does it actually remove `prepare_ms` — and what does the 256-wide cache axis cost? | 4.1 | `plan_hits` vs the formula's 6; `prepare_ms` on hit tokens vs 1.97; the KV-reduce delta — three separate numbers, none implying the others |
| Q10 | Does removing 1196 allocations remove the 3.9 ms, or does the cost reappear in `encode_dispatch` — and what does the arena cost in bytes? | 4.2 | `op_setup_ms` vs 3.9 **and** `encode_dispatch_ms` vs 0.47 **and** `step_wall_ms` **and** `ARENA_PEAK_BYTES` vs 40 MB |
| Q11 | What was `UNIFORM_BUFFER_REUSES` already doing before we assumed uniforms were the cost? | 4.2 | its hit rate at `metal.rs:2069`, read first |
| Q12 | Can a GPU scatter be emitted without atomics for the injective case, and is it deterministic? | 5.2 | parity vs `run_reduce_scatter` on the doc's worked example and at the KV shape (0 ULP), 100× byte-identical |
| Q13 | Does the persistent KV buffer actually hit `NOCOPY_BUFFERS`, and does the validator really not need changing? | 5.3 | `nocopy_reuses` (predicted 96/token), `kv_cache_upload_bytes` (predicted 0 after step 1), and `element_count(shapes.of(kv_node)) == buffer.len()` |
| Q14 | Does removing ~400 dispatches move the wall at all? | 6.2 | `encode_dispatch_calls`, `gpu_exec_ms` and **`step_wall_ms` jointly** — R12 is the first test, this is the second |
| Q15 | Does the emitter unification change a single byte of emitted source? | 1.1, 7.3, 7.4, 7.5 | the golden hash per route per backend, plus per-op `gpu_ns` unchanged within 1% |
| Q16 | Is the 7.35 ms elementwise bucket concentrated or uniform? | 3.2 | the top-5 nodes' share |
| Q17 | Do torch-MPS and ORT-CoreML actually beat us on any lane? | 8.1, 8.2, 8.3 | ms/sentence and p50/p95/p99 per provider with fidelity fields and partition counts — R9: today there is **no cell on either side** |
| Q18 | Is simdgroup starvation fixable with a config knob before a split-K kernel is written — and if not, does split-K fix it? | 10.1, 10.2 | the 12-cell `rows_per_group` sweep vs CoV; then 10.2's entry gate (attn GB/s < 70% of ffn) and its 12-cell `split_k` sweep |
| Q19 | Was 3.1's reduce width tuned for a graph that no longer exists? | 10.3 | the 18-row width sweep on the post-6.2 graph |
| Q20 | Did the memory rule hold on every card that touched the device? | every measuring card | both slopes and both caps per steady step, against G8's byte formula |

---

# Conflict resolutions

1. **Placement: scatter-only (B2, `shape.rs` untouched) vs scatter-first-then-affine (S 5.1–5.4)** — **B2 wins, scatter only**: `map.rs:118-124` states on main, verbatim, that a `Reduce`-wide destination-extent field "was rejected on blast-radius grounds" and the scatter convention chosen instead, so relaxing `project_output_shape` re-litigates a closed adjudication and changes bounds semantics for every existing `Reduce` plus the autograd adjoint — and leaving `shape.rs:469-485` untouched is itself the evidence on 6.2's row that the constraint was routed around, not weakened.
2. **Route census: lock-free per-plan route table (R16) vs per-dispatch recording** — **per-plan table wins as the census of record** (the route is a property of the plan, not of the dispatch), with a fixed-size `[Counter; N]` atomic array at `metal.rs:2243` preserving the sum identity; the `Mutex<BTreeMap>` pattern is abandoned outright and the census carries a **measured** ≤5%-of-`encode_dispatch_ms` budget with an ON/OFF arm.
3. **Must R13's board prediction move after `-fa 1` and the roofline (crit b)?** — **Yes**: 0.4 and 0.5 land in Phase 0 as dependency ancestors of 3.3, 4.3 and 9.2, so the plan's single board prediction divides by a settled denominator and reports a fraction of an existing ceiling, and 3.3 carries no "if it has landed" conditional.
4. **B2's device arena + pre-allocation peak sum vs S 4.2's per-op buffer pool** — **B2's arena wins**, because its first action is to compute and print the summed peak against G8's 40 MB activation term **before** allocating and to liveness-partition if it exceeds it, which makes the memory kill decidable in advance rather than detected after; S's retirement hazard is kept as an explicit test against `bound_op_retirement` (`metal.rs:1014`).
5. **The Q4_K tie-break (crit d)** — **S 2.3 rule 4 is replaced by B2's four-rung terminal ladder**: parity gate, route-count pin, summed family `gpu_ms` vs 2× pooled CoV, then max-abs parity error, then lines in the **Rust source region** `push_packed_row_blocked_body` (a file region with defined boundaries, not the delimiter-free concatenated MSL at `msl.rs:1978-1982`), then a terminal rung (Arm B, a commit on `4be2f3a`, beats an unrebased diff off `2b95210`) — so no rung can fail to decide.
6. **The over-allocation contract (crit RS-2, c)** — **S 5.5 is deleted entirely**: 4.1's bucketing makes the KV leaf's symbol-1 extent equal the buffer capacity, so `found == expected` holds at both `metal.rs:991-1000` and `cpu.rs:346-356` and **no validator, no `QuantizedBlock` field, and no parallel slice is needed**; `AlignedBuffer`'s page rounding is absorbed by deriving the capacity symbol from `buffer.len()`, as `align.rs:36-41` already requires.
7. **The one-bound-plan card (R16)** — **split in two and re-anchored**: 0.14 captures the fingerprint **after** each driver's own post-bind rewrite and is **pre-registered to FAIL on main** because of `metal.rs:1013`'s `correct_packed_matmul_layouts`, with a companion always-green test asserting the diff is exactly the packed operands; 7.6 makes `bind` produce the packed layout and deletes the rewrite; 7.7 turns the equality into a gate.
8. **0.6 `prune_dead`'s free variable (crit O-5)** — **it lands first and publishes a measured number**: 0.8 sits before 0.14, 1.1, 2.4, 6.1 and 7.5 on the dependency graph and prints `OPS_AFTER_PRUNE` on the openchat graph, so no downstream card carries the undecidable phrase "1196 minus the pruned count"; `op_count == 1196` is RED for 0.8 specifically.

*(Carried forward from round 1, unchanged: the census lands before any body swap; the worktree scheme is `proxima-wt-risc<NN>` / `gpu-risc/<NN>-<slug>`, verified collision-free; R13 stays THE baseline and 0.3 produces the band, not a new baseline; Phase 6 stays off the critical path.)*

---

### Critical Files for Implementation

- `/Users/brianbruggeman/repos/slot-0/proxima/omega/src/metal.rs` — `operand_bytes` defect `:612`/`:692-700`; `BLOCK_UPLOAD_BYTES` defect `:467`; validators `:984-1000`; **`bind` `:1004` then `correct_packed_matmul_layouts` `:1013`** (R16); `classify_kind` `:785-826` + call sites `:709-710`; `plan_named` `:578-585` (the cold census site); `encode_op` `:2179-2252` with `allocate_buffer` `:2210`, `upload_uniforms` `:2211`, **`ENCODE_DISPATCH_CALLS` `:2243`**, `device_buffers.insert` `:2249`; `UNIFORM_BUFFER_REUSES` `:2069`; `NOCOPY_BUFFERS` `:1848-1892`; `mark_resident` `:350-362`; `register_checkpoint_mapping` `:1744-1815`
- `/Users/brianbruggeman/repos/slot-0/proxima/omega/src/msl.rs` — `emit` `:673-697`; `kernel_cache_key` `:731-775`; `grid_threads` `:1517-1560`; `validate`'s **`ScatterNotSupported` at `:933`**; `u.out_base` **`:2734`**; `push_packed_row_blocked_body` `:2452-2530` (`lanes_per_block` `:2516`); `push_cooperative_reduce_body` `:3140-3194`; the geometry consts `:1017/:1030/:1046`; the delimiter-free unpack concatenation `:1978-1982`; `emit_is_deterministic_byte_equal` `:4656`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/generate.rs` — `resolve_plan` key `:966` and `plans.clear()` `:973`; `build_position_inputs` `:1304-1309`; `named_blocks` `Vec::with_capacity(… + 3 + …)` **`:1313-1319`** and `"eps"`/rope `:1332-1334`; the KV loop `:1364-1389`; `symbols` `:1391`; `LayerCache` `:621-656`; `cached_len +=` `:1559`; `phys_footprint_bytes` `:248`; `token_breakdown` `:1657` / `token_breakdown_metal` `:1723-1764`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/map.rs` — the write-direction convention and the **closed adjudication at `:109-131`**; **`IndexMap::scatter` `:175`**, `scatter_extent` `:209`, `as_gather_from_output` `:238`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/src/spec.rs` — `causal_mask` `:823-845` (the construction 4.1 copies); `append_mistral_cached_layer` `:2336-2865` with its constraint doc `:2303-2319`, the two-range combine `:2596-2720`, the mask consumption `:2610`, sole caller `:6282`; the KV leaves `:6216-6245`; the fan-out sites `:2867-2891`, `:3455`, `:6465`
- `/Users/brianbruggeman/repos/slot-0/proxima/proxima-model-interop/src/bind.rs` — `PROXIMA_MAX_TOKENS` `:2719`; the §14 oracle `:2797-2803`; the harness `:3002`, `:3084`; `metal_decode_summary` `:3042-3049`; **`forward_calls_taken` `:3051` and the two assertions `:3053-3059`** that 4.1 inverts `#[cfg]`-paired