I have verified the load-bearing citations. Here is the plan.

---

# GPU parity through ONE RISC — phased implementation plan

**Baseline: main `4be2f3a`, R13 (today's sealed cell). Every number below cites its ledger section. No time estimates. No verdicts — cards produce evidence rows.**

---

## Diagnosis

On today's main at `4be2f3a`, measured interleaved on a loaded box (R13), we run 67.92 ms/token against llama.cpp-Metal's 17.52 (3.88x wall, 3.25x kernel-only at gpu_exec 56.93), and the 50.40 ms gap decomposes into four buckets whose mechanisms are each traced to a line: **26.9 ms is Q4_K matvec above the weight-stream rate** — 225 packed-row-blocked ops burn 44.450 ms at 97-108 GB/s on ffn and 50-58 GB/s on attn_q/k/v/o (R13 family table), i.e. the body is ALU-bound, not bandwidth-bound, and the low-row attn shapes additionally starve for simdgroups (R3 M5); **16.6 ms is non-matmul GPU** — 385 cooperative reduces at 9.113 ms, every one dispatched at exactly `output_total * SIMD_WIDTH` = 32 threads regardless of reduction length (`msl.rs:1517-1560`, verified), where the incumbent's rms_norm doubles `nth` up to 1024 (R8), plus 547 elementwise at 7.350 ms that exist only because attention is emitted as a two-range online-softmax combine (`spec.rs:2336-2865`, doc at `:2303-2319`) rather than one range; **11.0 ms is orchestration that should not exist** — `plan_hits=0 plan_misses=8` in `metal_decode_summary` on every run, root-caused to the plan-cache key `(symbols[0], symbols[1]) = (new_count, cached_len)` at `generate.rs:966` where `cached_len` is `Extent::Symbolic(1)` on every KV leaf (`spec.rs:6216-6245`, verified), so the key misses **by construction** and `op_setup` 3.9 + `prepare` 1.97 + `block_upload` 2.0 are paid per token for a shape that never changes; and **17.5 ms is irreducible** weight streaming. Underneath all four sits the reason none of it can be attributed cleanly: the route decision is recovered by substring-matching emitted MSL (`classify_kind` `metal.rs:785-826`, whose own doc admits the decision "is not exposed as its own accessor"), and R13 found two live instrument defects that make any GB/s row a lie today — `operand_bytes` at `metal.rs:694` sums bound *buffer* lengths, so every matvec reports 4,140,417,024 bytes (the whole checkpoint mapping, since `7d09145` addresses weights by offset into ONE buffer), and `BLOCK_UPLOAD_BYTES` is counted unconditionally at `metal.rs:467` **before** the upload path is chosen, so it reports 4,147,777,096 B/token of "upload" when `mapping_offset_uploads=291` means almost none of it moved.

---

## One-RISC binding (brief §"What one RISC means", items 1-8 → cards)

| # | Item (brief lines 69-76) | Card(s) |
|---|---|---|
| 1 | ONE bound plan (`&[BoundOp]`, 4 kinds) from one rewrite engine, identical for every backend | **R15** (fingerprint), enforced by R14 |
| 2 | ONE first-class route enum decided before emission, censused `(NodeId, reason)` | **R08** |
| 3 | ONE emitter core over the 4 kinds, backend-specific TEXT only | **R14** |
| 4 | Every backend covers every kind | **R14** (WGSL/CUDA gain Iota+Constant+tiled+packed; `cuda.rs:146-183` rejects Iota/Constant today) |
| 5 | ONE sizing config owning every geometry constant | **R16** (+ R09 lands `[kv_cache]`, R17 lands `[cooperative_reduce]`) |
| 6 | Write placement via existing `Reduce.out_map` / `out_layout.base`, NOT a new Op | **R11** (MSL scatter emitter) + **R12** (KV write expressed as `IndexMap::scatter`) |
| 7 | A driver-level persistent-buffer alias | **R12** (KV buffer registered once, the `register_checkpoint_mapping` mechanism at `backend.rs:402-414` / `metal.rs:1744-1815`) |
| 8 | The Llama graph emitted at ≤23 real ops/layer (the incumbent's count, R8) | **R13** (single-range collapse), gated on R09+R11+R12 |

### Item 1 in detail — what is fingerprinted, and why it is not trivially true

**Fingerprinted:** a stable FNV-1a-64 over a canonical byte serialization of `&[BoundOp]` covering exactly what the emitters read and nothing about the device — for each op in order: `node.0`, `dtype` discriminant, `extents`, `BoundOpKind` discriminant; for `Elementwise`: every `ComposedBody` step's `ScalarOp` discriminant and operand indices; for `Reduce`: `element_body` steps, `reduce_op`, `init`, `keep`, `output_axes`, `out_layout.base` and `strides`, `out_scatter.is_some()` plus its `extent`/`element_stride`/`index_layout`; for every operand: `(source.0, Layout{base,strides}, Option<Lookup>{indices, index_layout, element_stride, extent})`; for `Constant`: `value.to_bits()`.

**Why it is not trivially true.** Four independent ways the plans can differ today, each verified: (a) the backend is selected *before* binding — `select_backend` at `generate.rs:855-861` returns `Backend::Metal` only when `gpu_layers == GPU_LAYERS_ALL` (`serving.rs:55`), and the CPU arm reaches `proxima_tensor::cpu` through a different entry, where `cpu.rs` runs folds (`run_reduce_scatter` `:6911`, the width tiles, the quantized fast paths) the GPU emitters reject outright at `msl.rs:933`; (b) `BoundOp::split` (`bind.rs:377-446`) treats `out_scatter: Some(_)` differently from `None` for CPU chunk rebasing, so a scatter's bound form is already backend-conditional in one direction; (c) `metal-tiled-gemm` is a Cargo feature that changes emission but **must not** change binding — nothing asserts that today; (d) the parallel branch's `prune_dead` changes the plan and the branch has no cross-backend assertion. The fingerprint test therefore binds the *same* program, symbols, and blocks twice — once through the CPU entry, once through the Metal entry — and asserts equal fingerprints, then asserts the fingerprint is invariant under `--features metal-tiled-gemm`. `emit_is_deterministic_byte_equal` (`msl.rs:4656`) is the existing emit-side twin; this is the bind-side one it never had.

---

## Global protocol

### Naming scheme (collision-checked against R0/R7/R15 and `git worktree list`)

- Branches: `risc/<slug>`. The prefix `risc/` appears nowhere in R15's branch list (which holds only `bench/`, `feat/`, `docs/`, `perf/`, `probe/`, `fix/`, `audit/`, `release/`, `verify/`, `qwen*`).
- Worktrees: `/Users/brianbruggeman/repos/slot-0/proxima-wt-r00` … `-r18`. Verified against the live `git worktree list`: no `proxima-wt-r<NN>` exists.
- Target dirs: `/Users/brianbruggeman/repos/slot-0/.cargo-target/r<NN>`, one per worktree (disciplined-component "Isolation for parallel code-landing work": a shared output dir produces lock contention *and* false greens).
- Features: named for the **mechanism**, never the author or the worktree. Every new `omega` feature `X` gets a forwarding line `X = ["omega?/X"]` in `proxima-model-interop/Cargo.toml`, exactly mirroring the verified pattern `metal-tiled-gemm = ["omega?/metal-tiled-gemm"]`; every new `proxima-tensor` feature `Y` gets `Y = ["proxima-tensor/Y"]` in **both** `omega` and `proxima-model-interop`. A feature the re-prove command's `-p` crate does not declare cannot be turned on from that command line — that is the failure this rule exists to stop.
- Row titles on a branch: `ROW ZZZ -- <title>` with the literal three characters `ZZZ`. Row numbers are assigned at land time from main's then-current last row, which is **233** (`proxima-tensor/docs/discipline.md:18736`, verified). Never write a number on a branch: R10 records that three unlanded branches each carry their own "ROW 234" and `perf/cached-attention-streaming` carries 234-267.

### The harness command (the one every bench-rung card runs)

```
cd /Users/brianbruggeman/repos/slot-0/proxima && \
CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r<NN> \
PROXIMA_MAX_TOKENS=8 \
flock /tmp/proxima-gpu-measure.lock -c \
'cargo nextest run -p proxima-model-interop --features metal,instrument,<card features> \
  --run-ignored all --no-capture \
  -E "test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)"'
```

Verified: the test is `#[ignore]`d and `#[cfg(feature = "metal")]` at `proxima-model-interop/src/bind.rs:3002`; `PROXIMA_MAX_TOKENS` is read at `bind.rs:2710-2715` defaulting to 24. Per-op (milli rung) swaps the filter to `test(profiles_one_real_decode_step_by_per_op_gpu_time)` (`bind.rs:3084`) and adds `PROXIMA_METAL_OP_PROFILE_STEP=3`.

### The N contract (eos-invariant)

The openchat prompt (`bind.rs:2724-2731`) stops on eos, so the token count is **not** a constant and no contract may name one. Define from the harness's own quantities (`bind.rs:3051`):

```
F  := runtime.plan_misses                     # from metal_decode_summary
F  == generated.0.len() + usize::from(generated.2)   # the harness's forward_calls_taken
S  := F - 1                                   # steady step rows (step 0 is prefill)
```

Every card asserts: **`F >= 2` — F == 0 or F == 1 is RED** (a run with no steady step measured nothing); **`S >= 1` — S == 0 is RED**; **`op_count > 0` — op_count == 0 is RED** (`generate.rs:107`, printed as `op_profile ... op_count=`); **`ENCODE_DISPATCH_CALLS / F == op_count`**. At `PROXIMA_MAX_TOKENS=8` on today's main this evaluates to F=8, S=7, op_count=1196 (R13) — but the *contract* is the formula, so it survives an eos at a different position.

### The measurer queue (one measurer on the box)

Every GPU-measuring command is welded through `flock /tmp/proxima-gpu-measure.lock -c '...'`. `flock` with no timeout **queues** — that is the explicit queue, at the process level, where the contended resource actually is. Additionally, each measuring card runs the quiet gate first and records its result in the row (never skips on failure — records the loadout, per disciplined-component "Pin the host loadout in the log"):

```
cd /Users/brianbruggeman/repos/slot-0/proxima && \
  for p in cargo rustc llama-bench llama-cli python3.11 cdb-daemon; do \
    printf '%s=%s\n' "$p" "$(pgrep -x "$p" | wc -l | tr -d ' ')"; done; \
  printf 'load=%s\n' "$(sysctl -n vm.loadavg)"
```

R13's cell was taken on a **LOADED** box (load 4.7-5.7, cdb-daemon resident) and still returned CoV 0.5-0.9% — so a loaded box is admissible provided the loadout is recorded and arms are interleaved A B A B A B. `sealed-pass.sh` exists only on `bench/sealed-pass` and hardcodes `REPO_ROOT` to `proxima-wt-seal` (:4) with four sibling worktrees hardcoded (:25-28) — R00 reworks it to take the worktree as `$1` before it lands.

### The memory gate (owner rule 2026-09-03: a memory regression is a NEGATIVE regardless of timing)

Observables that exist on main today, both printed per step in `token_breakdown_metal` (`generate.rs:1731`, verified): `phys_footprint_bytes()` (`generate.rs:248-292`, macOS `TASK_VM_INFO.phys_footprint` — the RSS observable) and `omega::metal::current_allocated_size()` (`metal.rs:270` — the device-allocation observable). Third: `kv_cache_upload_bytes` in `token_breakdown` (`generate.rs:1657`).

**The byte formula** (every term cites R13):

```
DEVICE_CAP_BYTES = 4_140_417_024                  # the ONE checkpoint mapping buffer (R13, = operand_bytes' bogus value, which IS that buffer's length)
                 + capacity_tokens * 262_144      # KV at capacity. 262_144 = 32 layers x (64 k_even + 64 k_odd + 128 v) x 8 kv_heads x 4 B
                                                  #   -- reproduces R13's MEASURED +262,144 B/token exactly
                 + 40_000_000                     # activations+uniforms headroom: R13 steady 4.163e9 - 4.1404e9 = 22.6 MB, x1.75
RSS_CAP_BYTES    = 400_000_000 at the prefill step   (R13: 310-357 MB)
                 = 100_000_000 at every steady step  (R13: 48-66 MB)
```

At `capacity_tokens = 256` (R09's bucket): `DEVICE_CAP_BYTES = 4_247_525_888`.

**Slopes**, computed by least squares over the S steady steps:
- `slope(device_allocated_bytes)` ≤ **2_000_000 B/step** (R13 observed +1-2 MB/token). After R10 and R12 the pre-registered target is **0**; exceeding the prior card's measured slope is a NEGATIVE.
- `slope(phys_footprint_bytes)` ≤ **1_000_000 B/step** (R13: "no monotonic trend" across steady steps).
- `plan_cache_len == 1` at every step (R13; the `self.plans.clear()`-on-miss bound landed by `ff749a0` at `generate.rs:973`).

**Enforcement:** a card that exceeds any cap or slope **rolls back**, even if it wins on time, and the row records the memory number as the headline. Two prior memory failures are the reason (R13): the 34 GB KV allocation from the `context_length` default (worktree only) and the plan-cache heap growth (bounded by `ff749a0`).

### Bench ladder (prediction is exactly one rung ahead — never same-rung)

| rung | artifact (verified to exist) |
|---|---|
| **nano** | `omega/examples/q4k_matvec_probe.rs`, `membw_probe.rs`, `attention_tiled_gemm_probe.rs` — one kernel, two-size marginal, GPU timestamps |
| **micro** | `omega/benches/metal_vs_cpu.rs` (registered `omega/Cargo.toml:207-210`, `required-features = ["metal"]`, doc says UNRUN) |
| **milli** | `profiles_one_real_decode_step_by_per_op_gpu_time` (`bind.rs:3084`) — one decode step, per-op, one command buffer per op |
| **bench** | `runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache` (`bind.rs:3002`) vs `llama-bench` |

A card that measures at rung R pre-registers a number at rung R+1 **only**. A miss kills the climb and is decomposed into inconsistency vs understanding-gap with a named work item.

### Tiers

`hands` = Luna (tool use, bounded edits, runs commands, no design judgment). `worker` = writes code against a fixed design. `judge` = adjudicates a fork; does not type.

---

# PHASE 0 — seal, land, reconcile, and repair the instruments

**Gate on the whole phase: no card in Phase 1+ may write a GB/s number until R01 lands.** R13's two instrument defects make every byte-derived rate wrong today.

---

### R00 — the sealed-pass harness lands on main, worktree-parameterized, with the measurer queue

- **tier:** hands
- **depends_on:** —
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r00 -b risc/seal-harness 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r00
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r00`
- **opens:** `scripts/omega-gate.sh:1-77` (the gate pattern to mirror: `[3/6]` asserts `ran_count` nonzero, `[6/6]` asserts `passed_count` nonzero); `proxima-model-interop/src/bind.rs:3002` and `:3084` (the two harness tests); `proxima-model-interop/src/bind.rs:2710-2715` (`PROXIMA_MAX_TOKENS`); `omega/src/metal.rs:270` (`current_allocated_size`); `proxima-model-interop/src/generate.rs:248` (`phys_footprint_bytes`).
- **work:** write `scripts/gpu-seal.sh` fresh on main (do **not** port `sealed-pass.sh` verbatim — R15 records it hardcodes `REPO_ROOT` to `proxima-wt-seal` at `:4` and four sibling worktrees at `:25-28`). Signature `gpu-seal.sh <worktree-abs-path> <target-dir> <feature-list> <runs>`. It (1) prints the quiet gate; (2) takes `flock /tmp/proxima-gpu-measure.lock`; (3) interleaves incumbent and ours A B A B A B for `<runs>`; (4) extracts `step_wall_ms`, `gpu_exec_ms`, `op_count`, `plan_hits`, `plan_misses`, `phys_footprint_bytes`, `device_allocated_bytes`, `kv_cache_upload_bytes` **with `grep -oE` on named `key=value` fields, never a `sed` backreference** — R13 records that an earlier draft's `plan_hits=10,20..70` was a `sed` artifact (`\10` = group 1 + literal `0`), a fabricated finding; (5) asserts the N contract and exits nonzero on `F<2 || S<1 || op_count==0`; (6) computes both memory slopes and both caps; (7) emits one machine-readable line per arm.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r00 && bash scripts/gpu-seal.sh /Users/brianbruggeman/repos/slot-0/proxima-wt-r00 /Users/brianbruggeman/repos/slot-0/.cargo-target/r00 metal,instrument 3
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r00 && bash -n scripts/gpu-seal.sh && shellcheck scripts/gpu-seal.sh
  ```
- **expect:** 3 ours runs x S steady rows each, S ≥ 1 per run, F ≥ 2 per run, `F == plan_misses`, `op_count > 0`; 3 llama arms. **Total extracted arm-rows N == 6; N == 0 is RED.** Reproduces R13 within its CoV: ours `step_wall_ms` mean in [67.0, 69.0] (R13 67.92, CoV 0.5%), `gpu_exec_ms` in [56.0, 58.0] (R13 56.93, CoV 0.7%), incumbent `tg32` in [56.0, 58.5] t/s (R13 57.08, CoV 0.89%).
- **predict (one rung ahead — this card measures at bench, so it predicts nothing higher; it instead pre-registers the *reproduction band* above, which is the seal's own contract).** Formally: this is the rung-0 anchor card and is exempt from the one-rung rule because it climbs nothing; it establishes the denominators every later prediction divides by. Stated explicitly so no later card silently inherits an unstated exemption.
- **kill:** if the reproduction band is missed on 2 of 3 runs, the box is not the box R13 measured; STOP the phase, record the loadout diff, and re-seal. Do not proceed on a moved baseline.
- **rollback:** `git worktree remove --force /Users/brianbruggeman/repos/slot-0/proxima-wt-r00 && git branch -D risc/seal-harness`. Nothing on main changes until the owner authorizes the commit.
- **blast:** one new file under `scripts/`. Zero source files touched. Zero features added.
- **observe:** `plan_misses` (`generate.rs:1764`), `op_count` (`generate.rs:107`) — both exist on main.
- **memory gate:** the script *implements* the gate; its own run asserts `RSS_CAP_BYTES` and `DEVICE_CAP_BYTES` at `capacity_tokens=0` (i.e. `4_180_417_024`) and both slopes, against R13's steady 4.152-4.163 GB.
- **reprove:** `cd /Users/brianbruggeman/repos/slot-0/proxima && bash scripts/gpu-seal.sh $PWD /Users/brianbruggeman/repos/slot-0/.cargo-target/r00 metal,instrument 3`
- **log row:** `ROW ZZZ -- the GPU seal is a script, not a memory: worktree-parameterized, N-asserting, memory-gated`

---

### R01 — the two R13 instrument defects, fixed before any GB/s row exists

- **tier:** worker
- **depends_on:** R00
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r01 -b risc/instrument-bytes 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r01
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r01`
- **opens:**
  - `omega/src/metal.rs:693-700` — `operand_bytes` sums `buffer.length()`. Verified: since `7d09145` binds ONE checkpoint mapping buffer, every weight operand reports 4,140,417,024.
  - `omega/src/metal.rs:466-468` — `counter!(BLOCK_UPLOAD_BYTES, block_byte_len(block))` fires **unconditionally, before** the match at `:469-486` chooses `upload_block` / `upload_packed_bytes`, and therefore before the mapping-offset / no-copy / copying path is known.
  - `omega/src/metal.rs:1467` (`BLOCK_UPLOAD_BYTES` decl), `:1572` (snapshot), `:1879` `upload_block_no_copy`, `:1903` `upload_block_no_copy_uncached`, `:1914` `create_no_copy_buffer`, `:1744-1815` `register_checkpoint_mapping`.
  - `proxima-model-interop/src/generate.rs:109-212` — every consumer of `operand_bytes`: `total_operand_bytes`, `op_profile_bucket`, `op_profile_top`'s `gpu_ns_per_byte`, `op_profile_family`'s `gpu_ns_per_byte`, `op_profile_family_split`'s `passed_/rejected_gpu_ns_per_byte`.
- **work, defect 1 (`operand_bytes`):** compute the operand's **tensor** byte length, not the bound buffer's. At `metal.rs:694` the loop already has `bound.operands()`, `plan.program`, and `packed_operands`. For each operand: `elements = product(program[source].shape resolved against bound.extents)`; bytes = `elements * 4` for `Float32`, else `elements / codec_block_elements * codec_block_bytes` using the constants already public in `msl.rs` (`Q4K_BLOCK_BYTES=144` `:294`, `Q4K_BLOCK_ELEMENTS=256` `:299`, `Q6K_BLOCK_BYTES=210` `:362`, `Q5K_BLOCK_BYTES=176` `:451`, `Q8_0_*` `:486/:491`, `Q4_0_*` `:522/:527`, `FLOAT16_*` `:540/:544`, `BFLOAT16_*` `:553/:556`). No new type; no new constant.
- **work, defect 2 (`block_upload_bytes`):** delete the unconditional counter at `:467`; add three counters recorded **inside** each path — `BLOCK_UPLOAD_BYTES_COPIED` (the real `memcpy` sites in `upload_block`/`upload_packed_bytes`), `BLOCK_UPLOAD_BYTES_NOCOPY_WRAPPED` (`create_no_copy_buffer` `:1914`), `BLOCK_UPLOAD_BYTES_MAPPING_OFFSET` (the `register_checkpoint_mapping` offset-resolution path). Keep `BLOCK_UPLOAD_CALLS` unchanged. Add all three to `MetalStageTotals` (`metal.rs:1514` region) and to the `token_breakdown_metal` println (`generate.rs:1723`).
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r01 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r01 bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r01 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r01 PROXIMA_MAX_TOKENS=8 PROXIMA_METAL_OP_PROFILE_STEP=3 flock /tmp/proxima-gpu-measure.lock -c 'cargo nextest run -p proxima-model-interop --features metal,instrument --run-ignored all --no-capture -E "test(profiles_one_real_decode_step_by_per_op_gpu_time)"'
  ```
- **expect:**
  - **N1 (sum invariant):** `BYTES_COPIED + BYTES_NOCOPY_WRAPPED + BYTES_MAPPING_OFFSET == 4_147_777_096` per steady token — exactly today's `block_upload_bytes` (R13). **A sum that does not match is RED; N == 0 is RED.**
  - **N2:** `BYTES_MAPPING_OFFSET / BYTES_COPIED > 100` — R13 measured `mapping_offset_uploads=291` vs `copying_uploads=4`, so the copied share must be tiny. `BYTES_COPIED == 0` is RED (4 copying uploads exist).
  - **N3 (the real bytes):** `op_profile_family` `operand_bytes` for `ffn_up` == `32 * 33.05 MB` within 1%, `ffn_down` == `32 * 34.00 MB`, `attn_q` == `32 * 9.45 MB`, `output.weight` == `107.5 MB` — the shape-derived true bytes R13 had to compute by hand (`rows*k*0.5625`). All 9 families in R13's table must match within 1%. **N == 9 rows; N < 9 is RED.**
  - **N4:** `total_operand_bytes` drops from ~1.2 TB (R13, absurd) to ~1.7 GB/step.
  - **N5:** `op_count == 1196`, `gpu_ns` totals unchanged within CoV vs R13's 61.082 ms per-op sum — this card must not move time.
- **predict (one rung ahead: measured at milli → predict at bench):** `step_wall_ms` unchanged within R13's CoV, i.e. mean in [67.6, 68.3]. This is an instrument card; a *timing* move here would mean the counters are on the hot path and is itself the finding.
- **kill:** if `step_wall_ms` moves by more than 2x the R13 CoV (>1.0%), the byte computation is on the hot path — move it behind `#[cfg(feature = "instrument")]` at the op-profile site only (it already is; verify `generate.rs:95` gating) and re-measure. If it still moves, roll back and compute bytes from `prepared.resolved` once per plan instead of per op.
- **rollback:** `git worktree remove --force ...proxima-wt-r01 && git branch -D risc/instrument-bytes`. Both defects are additive counters + one arithmetic change; reverting restores R13's (wrong) numbers exactly.
- **blast:** `omega/src/metal.rs` (2 sites + 3 counter decls + 3 struct fields), `proxima-model-interop/src/generate.rs` (println field list). Zero IR change. Zero feature added — both live under the existing `instrument` feature.
- **observe:** `BLOCK_UPLOAD_BYTES_COPIED` / `_NOCOPY_WRAPPED` / `_MAPPING_OFFSET` — **declared NEW by this card**; record sites: `omega/src/metal.rs` inside `upload_block`, `upload_packed_bytes`, `create_no_copy_buffer` `:1914`, and the mapping-offset resolution in `register_checkpoint_mapping` `:1744-1815`. Existing observable unchanged: `op_profile_family ... operand_bytes` (`generate.rs:184`).
- **memory gate:** three `AtomicU64` statics (`proxima-telemetry/src/metric/counter.rs:12-16`, 56 B each per its own doc `:98`) = +168 B static. No heap. Slopes and caps must be **identical to R00's sealed values** — this card allocates nothing per token.
- **reprove:** the R00 seal command with `--features metal,instrument` plus the milli command above; the row's claim is N1's sum identity and N3's 9-family table.
- **log row:** `ROW ZZZ -- two byte counters were measuring the wrong thing: operand_bytes was the checkpoint mapping, block_upload_bytes was the binding`

---

### R02 — the three incumbent arms: llama.cpp `-fa 1`, torch-MPS, ORT-CoreML

- **tier:** hands (llama arm) + worker (torch/ORT arms)
- **depends_on:** R00
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r02 -b risc/incumbent-arms 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r02
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r02`
- **opens:** `proxima-onnx/scripts/torch_reference/inference_bench.py` (verified: only `--threads` and `--runs`; no device arg); `proxima-onnx/scripts/torch_reference/model.py`; `scripts/onnx_reference/bench.py:96` (verified: `providers=["CPUExecutionProvider"]` hardcoded); `scripts/onnx_reference/run.sh`; `scripts/onnx_reference/export_model.py`.
- **work, arm A (`-fa 1`, hands):** R8 records flash attention is **OFF** by default at checkout `b25346221` (`common/common.h:328`), so R13's 57.08 t/s is the no-FA incumbent. Add the FA arm as a *second* incumbent row, interleaved with the no-FA arm:
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && flock /tmp/proxima-gpu-measure.lock -c '/Users/brianbruggeman/repos/others/llama.cpp/build/bin/llama-bench -m ~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf -n 32 -p 0 -r 5 -t 8 -ngl 99 -fa 1'
  ```
  Run 3x, interleaved A(no-fa) B(fa) A B A B.
- **work, arm B (torch-MPS, worker):** add `--device {cpu,mps}` to `inference_bench.py` (default `cpu`, so the existing arm is byte-identical) and a `torch.mps.synchronize()` **before** each timer stop. **Honest scope, stated in the row:** torch cannot run this Q4_K_S checkpoint; the comparable surface is the **matvec shape**, so the arm is `torch.mm` at f16 on `[1,4096]x[4096,4096]` (attn_q/o), `[1,4096]x[4096,14336]` (ffn_up/gate) and `[1,14336]x[14336,4096]` (ffn_down) on MPS. This is a *roofline companion* — "what a tuned framework achieves at these shapes on this silicon" — not a decode competitor. venv is verified present with torch 2.13.0 + MPS (R0).
- **work, arm C (ORT-CoreML, worker):** onnxruntime is NOT installed in that venv (R0), and `bench.py:96` hardcodes the CPU EP. Install `onnxruntime` into the existing venv; add `--providers` (default `CPUExecutionProvider`, preserving the current arm byte-for-byte) and add `CoreMLExecutionProvider`. **Honest scope:** the only ONNX model exported here is BGE-small (`scripts/onnx_reference/export_model.py`), not a 7B decode. The row therefore reads "ORT-CoreML cell exists at the BGE-small embedding shape; **no comparable micro surface** for 7B Q4_K decode — e2e compare REQUIRED before any ORT verdict" (disciplined-component, "When the incumbent's 80% surface is structurally inaccessible", option 2). Recording the gap is mandatory; omitting the arm is not permitted.
- **commands:** (each welded with `cd` and `flock`; the three arms are serialized against each other by the same lock)
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && flock /tmp/proxima-gpu-measure.lock -c 'proxima-onnx/scripts/torch_reference/venv/bin/python proxima-onnx/scripts/torch_reference/inference_bench.py --device mps --runs 5'
  cd /Users/brianbruggeman/repos/slot-0/proxima && flock /tmp/proxima-gpu-measure.lock -c 'proxima-onnx/scripts/torch_reference/venv/bin/python scripts/onnx_reference/bench.py --providers CoreMLExecutionProvider'
  ```
- **expect:** llama `-fa 1`: 3 runs, CoV < 2%, **N == 3 rows; N == 0 is RED**. torch-MPS: 3 shapes x 5 runs = **N == 15 rows**. ORT: **N == 1 row** (BGE-small, CoreML EP) **or** one explicit `FEATURE GAP: CoreMLExecutionProvider unavailable/rejected the graph` row with the ORT error text — a missing arm is RED, a documented gap is green.
- **predict (one rung ahead — these are bench-rung arms; they predict nothing, they *supply denominators*):** pre-registered directional expectation, falsifiable: `-fa 1` at decode with `n_kv` ≈ 40 changes tg32 by less than 5% (FA's win is at long context; at 8 tokens the KV is tiny). If `-fa 1` moves tg32 by more than 5%, R13's 3.88x headline divides by the wrong denominator and **every board-level prediction in this plan is re-derived against the FA arm**.
- **kill:** if `-fa 1` is not accepted by this llama.cpp build, record the exact stderr and use the no-FA arm as the sole incumbent, with the reason in the row. Not a blocker — a recorded scope.
- **rollback:** `git worktree remove --force ...proxima-wt-r02 && git branch -D risc/incumbent-arms`; the venv `pip install onnxruntime` is additive to a venv that is already untracked (`?? proxima-onnx/scripts/torch_reference/venv/`, verified in `git status`).
- **blast:** two Python scripts gain one optional flag each, both defaulting to today's behavior. Zero Rust. Zero features.
- **observe:** the arms' own stdout; recorded as named baseline files under `docs/bench-campaigns/2026-09-03-gpu-one-risc/` (verified untracked and present).
- **memory gate:** N/A for the external arms (they are other processes) — but the row records each arm's peak RSS via `/usr/bin/time -l`, because a torch-MPS arm that swaps invalidates the interleaved cell for *our* arms sharing the box.
- **reprove:** the three commands above, verbatim.
- **log row:** `ROW ZZZ -- three incumbent arms the board never had: llama.cpp -fa 1, torch-MPS at our matvec shapes, ORT-CoreML at the only ONNX shape we export`

---

### R03 — the streaming-copy roofline probe, both denominators, readback outside the timed window

- **tier:** worker
- **depends_on:** R00
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r03 -b risc/roofline-copy 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r03
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r03`
- **opens:** `omega/examples/membw_probe.rs:1-211` (verified: the existing Metal arm is a **full reduce-to-scalar**, one add per element, timed around `omega::execute_plan` which includes command-buffer creation, `waitUntilCompleted`, and readback — a 4-byte readback, cancelled by the two-size marginal at `:180-183`); `omega/src/metal.rs:654-761` `execute_plan_op_timed` (verified: one command buffer per op, reads `GPUStartTime`/`GPUEndTime`); `proxima-tensor/docs/rooflines.md:396-479` (GPU candidate ceiling = **DEBT, not measured**) and `:411` (membw_probe GPU ceiling = DEBT).
- **work:** add a THIRD arm to `membw_probe.rs`: a **streaming copy** — `Elementwise{Identity}` over an N-element f32 buffer producing an N-element output. This is the traffic shape a matvec actually has (read weights, write activations), which the reduce-to-scalar arm does not measure. Time it with `execute_plan_op_timed`'s `GPUStartTime`/`GPUEndTime` (`metal.rs:654-761`) so the **readback of N bytes is outside the timed window entirely** — the reason the reduce arm could get away with `Instant::now()` (4-byte readback) and this one cannot. Two sizes (256 MiB, 1 GiB), marginal difference, 21 runs, min reported.
  **BOTH denominators are printed, side by side, on every line:**
  - `read_only_gbs = N_bytes / gpu_seconds` — the denominator R13's family table uses (`ffn_up` 33.05 MB → 97.4 GB/s) and the denominator llama.cpp's 228.9 GB/s uses (3.9996 GB weight sweep / 17.47 ms). Comparable to both.
  - `traffic_gbs = 2 * N_bytes / gpu_seconds` — the bytes the memory system actually moved. A copy that hits 200 GB/s read-only is moving 400 GB/s of traffic; conflating the two is exactly the DERIVED-as-MEASURED error `rooflines.md:766-773` already closes on.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r03 flock /tmp/proxima-gpu-measure.lock -c 'cargo run --release -p omega --example membw_probe --features metal,cpu,instrument'
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r03 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r03 bash scripts/omega-gate.sh
  ```
- **expect:** **N == 3 arms** (CPU single, CPU multi, Metal reduce) **+ 2 new rows** (Metal copy at each size) **+ 1 marginal row**. `N == 0 is RED`; a marginal row with `delta_ms <= 0` is RED (the larger buffer must take longer). Sanity floor: `read_only_gbs` for the copy arm must exceed the CPU multi-thread triad (69.95/81.21 GB/s, ROW 176) — a GPU copy slower than the CPU triad means the probe is measuring the driver, not the memory system, and is RED.
- **predict (one rung ahead: measured at nano → predict at micro):** on `omega/benches/metal_vs_cpu.rs`'s `matvec_batch1_f32` arm at the Mistral 4096x4096 shape, achieved read-only GB/s will land within **±20%** of the copy probe's `read_only_gbs` — because an f32 matvec at batch 1 is a pure weight sweep with 2 flops/4 bytes. A miss larger than 20% means the f32 matvec kernel is *also* ALU-bound (the Q4_K finding generalizing), which is a named work item, not a noise excuse.
- **kill:** if the two-size marginal's CoV exceeds 5% over 21 runs, raise both sizes by 4x and re-run once. If it still exceeds 5%, the probe is measuring the driver; report the single-size numbers with the contamination stated and mark the ceiling still DEBT. Never report a marginal above 5% CoV as a ceiling.
- **rollback:** `git worktree remove --force ...proxima-wt-r03 && git branch -D risc/roofline-copy`. The arm is additive to one example file; the existing two arms are untouched.
- **blast:** `omega/examples/membw_probe.rs` only. No library code. No feature.
- **observe:** `GPUStartTime`/`GPUEndTime` via `execute_plan_op_timed` (`metal.rs:654-761`, exists on main, `instrument`-gated).
- **memory gate:** the 1 GiB arm allocates 1 GiB host + 1 GiB device in **and** 1 GiB out = **3 GiB peak**. Explicit cap for this card: device `<= 3_000_000_000` and RSS `<= 2_500_000_000`, both asserted by the probe itself via `current_allocated_size()` before and after. This is a probe, not the decode path, so the decode caps do not apply — stated so the exemption is deliberate, not silent.
- **reprove:** the `cargo run --example membw_probe` command above; the row's claim is the marginal `read_only_gbs` / `traffic_gbs` pair with its CoV.
- **log row:** `ROW ZZZ -- the GPU memory ceiling stops being DEBT: a streaming copy, GPU-timestamped, readback outside the window, both denominators printed`

---

### R04 — adjudicate `perf/cached-attention-streaming`: reject the fifth bound kind, land three generic commits

- **tier:** judge (the adjudication) → hands (the three cherry-picks)
- **depends_on:** R00, R01
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r04 -b risc/reconcile-parallel 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r04
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r04`
- **opens:** `proxima-tensor/src/bind.rs:221-264` (the closed 4-variant `BoundOpKind`, verified); the branch's `physical.rs` (+576, NEW MODULE), `cached_attention_candidates`, `render_cached_attention` (`msl.rs:104` on that branch), `failure-cached-attention-matcher.md` — read via `git show`, **never** by entering `proxima-wt-cattn`.

**The ruling (with its four grounds, each citable):**

1. **It is a fifth variant of a closed set minted for one model's attention shape.** `BoundOpKind` has exactly 4 variants (`bind.rs:221-264`, verified) and the whole "one RISC" claim is that 4 is enough. AGENTS.md §problem-solving is literal: *"we should not be adding arbitrary rules/code for specific instances."* A `CachedAttention` kind is that rule.
2. **It reconstructs information the graph destroyed.** Their own `failure-cached-attention-matcher.md` records the BoundOp-only matcher was abandoned as "a heuristic" that "cannot prove the semantic roles". guiding-principles §"Find where information is destroyed": the graph *knew* it was attention; bind threw the structure away; the matcher rebuilds it by pattern-matching. **The defect is upstream, in `append_mistral_cached_layer` (`spec.rs:2336-2865`), not in bind.** Fixing it downstream is the compensator.
3. **It is MEASURED not to buy wall time.** R12: feature-on 51.535 ms/tok wall vs feature-off control 51.571 (CoV 1.75-2.16%) — inside noise — while dispatches went 1194 → 616. **Halving the dispatch count moved nothing**, and GPU time got *worse* (39.841 vs 35.117). The macro-op's own number refutes its own thesis.
4. **The matcher is itself a per-token CPU cost.** Their ROW 247: `prepare` 150.7 ms/token with the matcher before indexing, 11.6 after — because `plan_hits=0` (M6') means it re-runs every token.

**Therefore: REJECT `BoundOpKind::CachedAttention`, `physical.rs`, and `render_cached_attention`.** The graph fix (R09+R11+R12+R13) removes the duplication at its source, which is what the macro-op was pattern-matching around.

**KEEP, as three independent, standalone, individually-green commits (AGENTS.md §pr-sequencing):**
- **C1** `prune_dead` / `dead_resolved_nodes` (branch commit `216d925`) — generic, RISC-conformant, no new kind.
- **C2** the consumer index that took `prepare` 150.7 → 11.6 (their ROW 248) — generic bind speedup, useful even with the matcher deleted.
- **C3** the classifier-mislabel *finding* (their ROW 263: `classify_kind` labeled the paired body `reduce-cooperative`, 9/601 → 225/385 after their fix). **Do not port their fix** — it adds a second marker string, i.e. more of M10. Port the finding into R08's row as the third independent witness that substring classification is broken, and let R08's route enum be the fix.
- **Their four measured negatives are preserved as negative rows** (ROWs 249, 251-254, 259-260, 265-267: float4 accumulation, explicit unroll, width-64 dispatch, row-batch 8, tiled-gemm for decode, ggml nsg=2). nsg=2 is now a **fourth** independent negative across two lanes (R4 + R12) — the dead-lever list gains its citation.

- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git log --oneline main..perf/cached-attention-streaming | cat
  cd /Users/brianbruggeman/repos/slot-0/proxima && git show --stat 216d925 | cat
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r04 && git cherry-pick -x 216d925
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r04 bash scripts/proxima-tensor-gate.sh && bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r04 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r04 bash scripts/gpu-seal.sh $PWD /Users/brianbruggeman/repos/slot-0/.cargo-target/r04 metal,instrument 3
  ```
  (repeat cherry-pick + full gate + seal per commit; **every commit is a green bisect point**)
- **expect:** per commit, `omega-gate.sh` `[3/6] ran_count > 0` and `proxima-tensor-gate.sh` test count > 0 — **N == 0 is RED** (`omega-gate.sh:41-45` already asserts this; verified). After C1: `op_count` **decreases** from 1196 by the dead-node count and `op_count == 1196` is RED for C1 specifically (a `prune_dead` that removes nothing did not land). After C2: `prepare_ms` decreases from R13's 1.97; `prepare_ms >= 1.97` is RED for C2.
- **predict (one rung ahead: C1/C2 measured at bench → there is no rung above bench, so predict at bench from the milli evidence they came with):** C1 removes dead nodes with zero gpu_ms (R13: `constant`/`iota` degenerate control ops total 0.169 ms over 39 ops), so **`gpu_exec_ms` moves by less than 0.2 ms** and the win is dispatch-count and `op_setup`, not GPU. C2 moves `prepare_ms` and nothing else; **`gpu_exec_ms` unchanged within CoV**. If C1 moves `gpu_exec_ms` by more than 0.2 ms it removed live nodes — that is a **correctness** event, and §14 says the assumption is ours is wrong: stop and diff the generated text.
- **kill:** any cherry-pick that changes `generated_text` from `"Here is a simple Python function that returns"` (R13, identical across all three runs) is reverted immediately — §14, the incumbent-captured greedy answer at `bind.rs:2797-2803` (`2651` / `"known"`) is the oracle and it is asserted by the harness.
- **rollback:** per-commit `git revert`; the branch is a stack of three independent commits by construction, so any one reverts alone.
- **blast:** C1 touches `proxima-tensor/src/bind.rs` (plan construction — every backend). C2 touches `bind.rs` only. Neither touches `spec.rs`, `msl.rs`, or `metal.rs`. The rejected material touches `physical.rs` (+576) and `msl.rs` (+202) and is **not** brought over.
- **observe:** `op_count` (`generate.rs:107`) for C1; `prepare_ms` in `token_breakdown_metal` (`generate.rs:1723`) for C2.
- **memory gate:** `prune_dead` **removes** device buffers, so `device_allocated_bytes` must be `<=` R00's sealed steady value; an **increase** is a NEGATIVE and rolls back regardless of the dispatch win. Caps and slopes per the global formula at `capacity_tokens=0`.
- **reprove:** `bash scripts/gpu-seal.sh` at this worktree, plus both crate gates.
- **log rows:** three — `ROW ZZZ -- BoundOpKind::CachedAttention is adjudicated OUT: a fifth kind for one model's attention, whose own wall-clock cell is null`; `ROW ZZZ -- dead resolved nodes are dropped before dispatch (generic, kept from the parallel lane)`; `ROW ZZZ -- a consumer index removes repeated bind work (generic, kept from the parallel lane)`

---

### R05 — ONE Q4_K body: a bake-off with a tie-break decidable before the numbers exist

- **tier:** worker (build both arms) → judge (apply the tie-break)
- **depends_on:** R00, R01 (the GB/s rows are meaningless before R01)
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r05 -b risc/q4k-body-bakeoff 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r05
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r05`
- **opens:** `omega/src/msl.rs:2515-2535` (verified: `let lanes_per_block = 8;` at `:2516`, the loop step `ib += SIMD_WIDTH/lanes_per_block` at `:2527` — the packed-row-blocked Q4_K body); `omega/src/msl.rs:190-311` (`Q4K_UNPACK_MSL`, `Q4K_BLOCK_BYTES=144` `:294`, `Q4K_BLOCK_ELEMENTS=256` `:299`); the ggml original at `ggml-metal.metal:5086-5193` per R8 (mask without shift `& 0x000F/0x0F00/0x00F0/0xF000`, fold 1/256 and 1/16 into scale at combine `:5171-5175`, branch-free `kmask1/2/3` at `:5147-5150`).
- **the two candidates, same underlying ggml mechanism, independently re-derived:**
  - **Arm A** — `metal-q4k-mask-fma`, uncommitted in `proxima-wt-gpuker` / `proxima-wt-all` off `2b95210`. MEMORY-tagged: -36% on ffn_gate/up, -17.2% gpu_exec (R3 M3). Nine commits behind main including `spec.rs +8735/-2836` and two `metal.rs` changes — **will conflict on rebase** (R7).
  - **Arm B** — `q4k_pair_dot`, a real commit on today's main in `perf/cached-attention-streaming` (their ROW 257). MEASURED: GPU family 47.8 → 33.9 ms (-29%), parity 3.1e-6 vs f32 on real `blk.0.attn_q.weight`.
- **the tie-break, written down BEFORE either arm is measured (four rungs, terminal):**
  1. **Primary:** lower summed `op_profile_family` `gpu_ms` over the seven weight families `{ffn_up, ffn_gate, ffn_down, attn_q, attn_k, attn_v, attn_output}`, same box, same seal, arms interleaved A B A B A B, 3 runs each. Denominator: R13's `reduce-packed-row-blocked` bucket, **44.450 ms over 225 ops**.
  2. **If within 2x pooled CoV** (R13 `gpu_exec` CoV 0.7% → threshold 1.4%): tie. Break on **parity error** vs `proxima_tensor::cpu::evaluate` on real `blk.0.attn_q.weight` bytes from the actual checkpoint (§9: real data, byte level). Arm B has a recorded 3.1e-6; Arm A must produce its own number on the same tensor. **Lower max-abs-error wins** (§14).
  3. **Still tied:** fewer emitted MSL lines in the packed-row-blocked body (§1 minimality / §2 teaching surface — the body a reader can follow to the ggml source wins).
  4. **Still tied — terminal, always decidable:** **Arm B wins**, because B is a commit on `4be2f3a` and A is an unrebased diff off `2b95210` that conflicts with nine commits (R7). Lower landing risk breaks the last tie. No rung can fail to decide.
- **the loser is not deleted.** It lands as a negative row with its measured number and the tie-break rung that decided it (disciplined-component: "Negative deltas are knowledge"). The winner's feature is named for the **mechanism** — `metal-q4k-fold-scale` — not the author, so the row does not encode which worktree won.
- **feature wiring:** `omega/Cargo.toml` gains `metal-q4k-fold-scale = ["metal"]`; `proxima-model-interop/Cargo.toml` gains `metal-q4k-fold-scale = ["omega?/metal-q4k-fold-scale"]` — the verified forwarding pattern.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r05 flock /tmp/proxima-gpu-measure.lock -c 'cargo run --release -p omega --example q4k_matvec_probe --features metal,metal-q4k-fold-scale'
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r05 PROXIMA_MAX_TOKENS=8 PROXIMA_METAL_OP_PROFILE_STEP=3 flock /tmp/proxima-gpu-measure.lock -c 'cargo nextest run -p proxima-model-interop --features metal,instrument,metal-q4k-fold-scale --run-ignored all --no-capture -E "test(profiles_one_real_decode_step_by_per_op_gpu_time)"'
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r05 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r05 bash scripts/omega-gate.sh
  ```
  (the same three, with the feature omitted, are the OFF control; interleave)
- **expect:** `op_profile_family` emits **N == 9 family rows** (R13's table) per run; `N < 9 is RED`. `reduce-packed-row-blocked op_count == 225` in both arms — a body change that moves the op count changed the *route*, not the body, and is RED. Parity: `omega/tests/metal_parity` `ran_count > 0` under the feature; **a feature whose tests compile to zero is exactly the N==0 trap `omega/Cargo.toml`'s own `default` comment records** (`metal` was default-off and 13 `metal_parity` cases compiled to zero).
- **predict (one rung ahead: measured at milli → predict at bench):** the winner's packed-row-blocked bucket drops from 44.450 ms to **≤ 33.0 ms** (B's -29% on 47.8→33.9, conservatively applied to 44.450 = 31.6; band [30, 33]); therefore `gpu_exec_ms` drops from 56.93 to **[45.0, 47.0]** and `step_wall_ms` from 67.92 to **[56.0, 58.0]**, i.e. **3.20-3.31x** vs the 17.52 ms incumbent. A miss outside that band kills the climb and is decomposed: **inconsistency** if the family bucket moved as predicted but wall did not (→ orchestration is absorbing it, which R09/R10 own), **understanding-gap** if the family bucket itself missed (→ the -29% did not transfer off their tree, and the work item is "what else is in their feature-off control", since R12 records that control **already carried the paired body**).
- **kill:** parity error above 1e-4 max-abs on any real weight tensor → the arm is dead on correctness (§14), regardless of speed. `generated_text` drift from R13's string → dead.
- **rollback:** the feature is default-off; `--features` omission is the rollback. Branch deletion: `git worktree remove --force ...proxima-wt-r05 && git branch -D risc/q4k-body-bakeoff`.
- **blast:** `omega/src/msl.rs` packed-row-blocked body only, behind a default-off feature. The generic SIMD-fold and tiled-GEMM bodies are untouched. Two `Cargo.toml` feature lines.
- **observe:** `op_profile_bucket kind=reduce-packed-row-blocked gpu_ms` (`generate.rs:127`) and the 9 `op_profile_family` rows (`generate.rs:184`) — **both now carry true bytes because R01 landed**, which is why this card depends on R01.
- **memory gate:** a kernel-body change allocates nothing new; `device_allocated_bytes` and `phys_footprint_bytes` must be **bit-identical in slope** to R00's seal. Any device increase is a NEGATIVE and rolls back the arm even if it wins the bake-off.
- **reprove:** the three commands above, ON and OFF, interleaved; the row's claim is the 9-family table with CoV over 3 runs.
- **log rows:** two — `ROW ZZZ -- one Q4_K body, chosen by a tie-break written before the numbers` and `ROW ZZZ -- NEGATIVE: the losing Q4_K body, its number, and the rung that decided it`

---

### R06 — the wide cooperative reduce (M4), rebased onto main and re-measured

- **tier:** worker
- **depends_on:** R00, R01, R05 (serialized on the measurer; and the Q4_K body must be fixed first so the reduce bucket's share is measured against the final matvec cost)
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r06 -b risc/wide-cooperative-reduce 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r06
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r06`
- **opens:** `omega/src/msl.rs:1517-1560` `grid_threads` — **verified**, the cooperative arm is `output_total * SIMD_WIDTH`, i.e. exactly 32 threads per output element for **every** reduction length; `omega/src/msl.rs:3190-3194` (`output_index = gid/32`, `lane = gid%32`); `omega/src/msl.rs:824` `reduce_is_cooperative`; `omega/src/sized.rs` (`SIMD_WIDTH: u64 = 32`, documented as a **hardware fact, not a policy knob** — so the reduce *width* must be a **new** policy const, not an override of this one); the incumbent at `ggml-metal.m:3797-3804` (nth doubles from 32 up to `min(ne00/4, maxTotalThreadsPerThreadgroup)`, float4 loads, `simd_sum` → threadgroup shmem → `simd_sum`, `ggml-metal.metal:1679-1721`; a 4096-wide row gets 1024 threads — R8).
- **work:** two-level tree reduce. Threads-per-output = `min(next_pow2(reduction_len / vec_width), COOPERATIVE_REDUCE_MAX_THREADS)`, floored at `SIMD_WIDTH`. `COOPERATIVE_REDUCE_MAX_THREADS` and `COOPERATIVE_REDUCE_VEC_WIDTH` come from a **new `[cooperative_reduce]` section in `omega-runtime.toml`** through `build.rs`'s existing `resolve_int` + `require_nonzero` + `emit_sizing_consts` (verified at `omega/build.rs:67-120`, with per-key `cargo:rerun-if-env-changed` at `:85`) — §12, no bare const. Feature `metal-wide-reduce` in `omega`, forwarded as `metal-wide-reduce = ["omega?/metal-wide-reduce"]`.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r06 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r06 bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r06 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r06 OMEGA_COOPERATIVE_REDUCE_MAX_THREADS=1024 PROXIMA_MAX_TOKENS=8 PROXIMA_METAL_OP_PROFILE_STEP=3 flock /tmp/proxima-gpu-measure.lock -c 'cargo nextest run -p proxima-model-interop --features metal,instrument,metal-q4k-fold-scale,metal-wide-reduce --run-ignored all --no-capture -E "test(profiles_one_real_decode_step_by_per_op_gpu_time)"'
  ```
- **expect:** `op_profile_bucket kind=reduce-cooperative op_count == 385` (R13) in both arms — the count must not move, only the time. `op_count != 385` is RED (the route changed, not the geometry). `ENCODE_DISPATCH_CALLS/F == op_count` still holds. The `OMEGA_COOPERATIVE_REDUCE_MAX_THREADS` env override must actually take effect — prove it by building at 32 and at 1024 and asserting the emitted MSL differs (`emit_is_deterministic_byte_equal` at `msl.rs:4656` is the determinism twin that makes this a valid assertion).
- **predict (one rung ahead: measured at milli → predict at bench):** `reduce-cooperative` drops from 9.113 ms toward the -20% M4 measured (R3) — band **[7.0, 7.6] ms**; combined with R05's winner, `gpu_exec_ms` lands in **[43.5, 45.5]** and `step_wall_ms` in **[54.5, 56.5]** = **3.11-3.22x**.
- **kill:** a threadgroup wider than the pipeline's `maxTotalThreadsPerThreadgroup` fails at dispatch — clamp against the device's own value at `pipeline_for` (`metal.rs:1402-1426`), never against a constant. If the clamp makes the width identical to 32 for our shapes, the lever is dead: record it as a negative and move on.
- **rollback:** default-off feature; omit `--features metal-wide-reduce`.
- **blast:** `omega/src/msl.rs` (`grid_threads` cooperative arm + `push_cooperative_reduce_body` `:3140-3194`), `omega/src/sized.rs` (+2 consts), `omega/omega-runtime.toml` (+1 section), `omega/build.rs` (+2 `resolve_int` calls). Elementwise, packed-row-blocked, tiled-GEMM and serial paths untouched.
- **observe:** `op_profile_bucket kind=reduce-cooperative gpu_ms` and `gpu_ns_per_op` (`generate.rs:127`; R13 baseline 9.113 ms / 23,670 ns per op).
- **memory gate:** a wider threadgroup uses more **threadgroup** memory (on-chip), not device memory. `device_allocated_bytes` must be unchanged in slope and absolute vs R00's seal; an increase is a NEGATIVE.
- **reprove:** the two commands above, ON and OFF, interleaved 3x.
- **log row:** `ROW ZZZ -- the cooperative reduce stops running 32-wide for every length: threads scale with the reduction, from the sizing config`

---

### R07 — the discipline rows and the ai_docs JSONL records land

- **tier:** hands
- **depends_on:** R00-R06
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r07 -b risc/log-and-aidocs 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r07
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r07`
- **opens:** `proxima-tensor/docs/discipline.md:18736` (**ROW 233 is main's last**, verified; 18,766 lines total); `proxima-tensor/docs/rooflines.md:396-479, :751, :766-773`; `ai_docs/AGENT.md:1-40` (verified: *"If the index is missing required structure or evidence, add records to `ai_docs` instead of bypassing the structure"*); `ai_docs/index.jsonl`, `task-routes.jsonl`, `invariants.jsonl` (verified: **ZERO tensor/omega/GPU records**, R0).
- **work:** (a) renumber every `ROW ZZZ` from Phase 0 sequentially starting at **234**, in the order the commits land, and append them to `discipline.md`. R10 warns main's numbering is already non-monotonic (ROW 205 at line 17864 precedes ROW 204 at 17950) from concurrent worktrees — so renumbering happens once, at land, from `grep -n '^## ROW' | tail -1`. (b) `rooflines.md`: replace the GPU-lane DEBT entry (`:411`) with R03's measured ceiling and **both** denominators; update the summary row at `:751`; the closing note at `:766-773` ("the GPU lane's ratio is not a gap-to-machine at all") is now answerable and gets its answer. (c) three JSONL records, matching the schemas verified in the files:

```jsonl
# ai_docs/index.jsonl
{"id":"proxima.gpu.one_risc.discipline","kind":3,"summary":"The GPU lane's discipline log: bound-plan RISC, route census, write placement, Q4_K body, cooperative reduce geometry.","path":"proxima-tensor/docs/discipline.md","read_when":["gpu-lane","metal","omega","q4k","decode-perf"],"source_paths":["proxima-tensor/docs/discipline.md","proxima-tensor/docs/rooflines.md","omega/src/msl.rs","omega/src/metal.rs","proxima-tensor/src/bind.rs"],"relations":[]}

# ai_docs/task-routes.jsonl
{"task":"gpu-lane","purpose":"Change or measure anything on the proxima-tensor -> omega GPU path (graph, bind, route, emit, drive).","must_read":["ai_docs/AGENT.md","ai_docs/invariants.jsonl","proxima-tensor/docs/discipline.md","proxima-tensor/docs/rooflines.md"],"then_read_if_relevant":["proxima-tensor/src/op.rs","proxima-tensor/src/map.rs","proxima-tensor/src/bind.rs","proxima-tensor/src/shape.rs","omega/src/route.rs","omega/src/msl.rs","omega/src/metal.rs","omega/omega-runtime.toml","proxima-model-interop/src/generate.rs","proxima-model-interop/src/bind.rs"],"queries":["jq -c 'select(any(.applies_to[]?; . == \"gpu-lane\"))' ai_docs/invariants.jsonl"],"done_when":["a home-turf llama.cpp-Metal arm is on the row","the N contract (F>=2, S>=1, op_count>0) is asserted","both memory slopes and both caps are recorded","every geometry constant traces to omega-runtime.toml","the route census sum equals ENCODE_DISPATCH_CALLS"]}

# ai_docs/invariants.jsonl  (three records)
{"id":"proxima.gpu.one_bound_plan","kind":7,"summary":"One bound plan (&[BoundOp], 4 kinds) is produced by one rewrite engine and is identical for every backend.","rule":"Never add a BoundOpKind variant for one model's shape. A backend-specific plan difference is a defect; the plan fingerprint test is the witness.","applies_to":["gpu-lane","hot-path","disciplined-component"],"evidence_required":["The bind-side plan fingerprint is equal across the CPU and Metal entries and invariant under metal-tiled-gemm."],"relations":[{"idx":7,"target":"proxima.gpu.one_risc.discipline"}]}
{"id":"proxima.gpu.route_is_a_value","kind":7,"summary":"The kernel route is a first-class value decided before emission, not a substring of emitted MSL.","rule":"Never classify a kernel by matching its source text. classify_kind's substring buckets silently relabel when a body changes; the route enum and its per-dispatch census are the only admissible attribution.","applies_to":["gpu-lane","instrumentation"],"evidence_required":["sum(route dispatch counters) == ENCODE_DISPATCH_CALLS, and the census's per-dispatch cost is under 5% of encode_dispatch_ms."],"relations":[{"idx":7,"target":"proxima.gpu.one_risc.discipline"}]}
{"id":"proxima.gpu.memory_is_a_kill_criterion","kind":7,"summary":"A memory regression on the GPU decode path is a NEGATIVE regardless of the timing result.","rule":"Every GPU card records RSS slope, device-allocation slope, and both absolute caps from the byte formula; exceeding any one rolls the card back even when it wins on time.","applies_to":["gpu-lane","hot-path"],"evidence_required":["phys_footprint_bytes and current_allocated_size per steady step, least-squares slope over S steps, against DEVICE_CAP_BYTES = 4_140_417_024 + capacity_tokens*262_144 + 40_000_000."],"relations":[{"idx":7,"target":"proxima.gpu.one_risc.discipline"}]}
```

- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r07 && grep -n '^## ROW' proxima-tensor/docs/discipline.md | tail -1
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r07 && for f in ai_docs/index.jsonl ai_docs/task-routes.jsonl ai_docs/invariants.jsonl; do jq -c . "$f" > /dev/null && echo "$f OK $(wc -l < $f)"; done
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r07 && bash ai_docs/query.sh gpu-lane
  ```
- **expect:** `jq` parses all three files (a malformed JSONL line is RED); `index.jsonl` gains exactly 1 record, `task-routes.jsonl` 1, `invariants.jsonl` 3 — **N == 5 new records; N == 0 is RED**. `grep -c '^## ROW' discipline.md` increases by exactly the number of Phase-0 rows and the last row number is monotonically greater than 233.
- **predict:** N/A — this card measures nothing and therefore predicts nothing. Stated so it is not mistaken for an omission.
- **kill:** if `ai_docs/query.sh gpu-lane` returns nothing after the records land, the schema assumption is wrong; read `query.sh` and fix the record shape rather than bypassing the index (AGENT.md is explicit).
- **rollback:** documentation-only; `git revert` the single commit.
- **blast:** `proxima-tensor/docs/discipline.md`, `rooflines.md`, three JSONL files. Zero source.
- **observe:** the row count itself; `bash ai_docs/query.sh gpu-lane` returning ≥ 1 record.
- **memory gate:** N/A (no code path changes). Recorded as N/A with this rationale rather than left blank.
- **reprove:** the three commands above.
- **log row:** `ROW ZZZ -- main's log learns the GPU session happened, and ai_docs gains its first tensor/omega records`

---

# PHASE 1 — the route becomes a value

### R08 — `omega::route::Route`, decided before emission, censused per dispatch, cost-bounded

- **tier:** worker
- **depends_on:** R01 (bytes), R04-C3 (the third witness), R05 (the body is settled, so the routes stop moving under us)
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r08 -b risc/route-enum 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r08
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r08`
- **opens:** `omega/src/msl.rs:673-697` (`emit` routes the 4 kinds), `:751-754` (**"Ordering load-bearing"** — the hand-ordered gates), `:824` `reduce_is_cooperative`, `:1235` `packed_row_block`, `:1450` `tiled_gemm_block`, `:1487` `diagnose_packed_row_block`, `:2178-2257` `render_reduce`, `:3140-3194` `push_cooperative_reduce_body`; `omega/src/metal.rs:777-783` (the doc that **admits** the routing decision "is not exposed as its own accessor"), `:785-826` `classify_kind`, `:835-854` `diagnose_kind`, `:709-710` (the call sites); `omega/src/metal.rs:2225-2248` (the encode-dispatch block; **`counter!(ENCODE_DISPATCH_CALLS, 1)` at `:2243` — verified**); `proxima-tensor/src/instrument.rs:809-828` `WidthDeclineReason` + `:842` `WIDTH_TILE_DECLINE: Mutex<BTreeMap<(u32, Reason), Totals>>` + `:848` `record_width_tile_decline` (the shipped `(NodeId, reason)` census pattern to mirror); `proxima-telemetry/src/metric/counter.rs:12-16, :46` (`Counter` = one `AtomicU64`, `add(delta, tags)`).

**The new type, and both questions answered in-line.**

*The pipe question:* `Route` is not a pipe. It is a decision value computed once per `BoundOp` and consumed once by `emit` — no stages, no backpressure, no cancellation, no fan-in/out. `backend.rs:1-52` records that plan/execute was already adjudicated **not** a pipe on 2026-08-30 because the relocation question failed; `Route` sits strictly inside that already-settled boundary.

*The relocation question — the call site, written both ways:*

```rust
// WAY A (Route enum):
let route = route::of(bound, quantized);          // one decision, before emission
match route {                                      // exhaustive; the compiler proves coverage
    Route::TiledGemm(b)      => push_tiled_gemm_body(&mut src, bound, b)?,
    Route::PackedRowBlock(b) => push_packed_row_blocked_body(&mut src, bound, b)?,
    Route::CooperativeSimd   => push_cooperative_reduce_body(&mut src, bound)?,
    Route::SerialReduce      => push_serial_reduce_body(&mut src, bound)?,
    Route::Scan | Route::Elementwise | Route::Iota | Route::Constant => { /* ... */ }
}
// and, at metal.rs:2243, beside the existing counter:
counter!(ROUTE_DISPATCHES[route as usize], 1);

// WAY B (today: hand-ordered gates + substring recovery):
if let Some(b) = tiled_gemm_block(resolved, quantized, op, init, axes) { push_tiled_gemm_body(..)? }
else if let Some(b) = packed_row_block(resolved, quantized)            { push_packed_row_blocked_body(..)? }
else if reduce_is_cooperative(resolved)                                { push_cooperative_reduce_body(..)? }
else                                                                   { push_serial_reduce_body(..)? }
// ... and then, separately, in metal.rs:785-826:
let kind = if src.contains("simdgroup_multiply_accumulate") { "tiled-gemm" }
           else if src.contains("q4k_run8(blk")            { "reduce-packed-row-blocked" }
           else if src.contains("simd_sum(")               { "reduce-cooperative" } else { .. };
```

**What a caller can DO with Way A that Way B cannot:** assert `route::of(bound) == route_recorded_at_dispatch(bound.node)` (Way B has no `route::of` to compare against — the decision is not a value), and assert `sum(ROUTE_DISPATCHES) == ENCODE_DISPATCH_CALLS` (Way B's classifier produces a label, not a count, and is MEASURED to mislabel: R12 ROW 263, 9/601 → 225/385 after a fix; R13's own bucket table depends on it; M10 names it a "labeling instrument that silently relabels when a kernel body changes"). Those two assertions are new caller capability, not identical lines. **The type is earned.**

*Why no existing primitive could be extended (the one sentence gate 14 requires):* `WidthDeclineReason` (`instrument.rs:809-828`) is exactly this pattern and is the model, but its eight variants are CPU width-tile stride/shape conditions (`NoFusedMultiplyAdd`, `NarrowWidth`, `StrideLayout`, …) that name nothing a GPU emitter decides, so `Route` is its GPU-side sibling in `omega`, not an extension of it in `proxima-tensor`.

- **the cost bound (the instrument must not sit uncosted inside the slice it measures).** R13: `encode_dispatch_ms = 0.47` over 1196 dispatches = **393 ns/dispatch**; `op_setup = 3.9 ms` = 3,261 ns/op. Budget: **≤ 5% of encode_dispatch = 19.6 ns/dispatch**. A `Mutex<BTreeMap>` insert per dispatch (the `WIDTH_TILE_DECLINE` shape) blows that budget by orders of magnitude. So the census is **hot/cold split**:
  - **Hot (per dispatch, at `metal.rs:2243` beside `ENCODE_DISPATCH_CALLS`):** one `counter!` on a `[Counter; 8]` array indexed by the route discriminant — a single relaxed `AtomicU64::fetch_add`, uncontended, single-threaded, the *same* operation already executing on that line. Marginal cost ≈ one atomic add.
  - **Cold (once per plan, at `plan_named`/`prepare`):** the `(NodeId, Route)` pairs, into a `Vec<(u32, Route)>` sized `prepared.resolved.len()`. Once per plan is once per token today and once per 256 tokens after R09.
  - **The bound is MEASURED, not assumed:** the card runs census-ON vs census-OFF interleaved and asserts `encode_dispatch_ms_on - encode_dispatch_ms_off <= 0.0235 ms` (5% of 0.47). **A census that costs more than 5% of the slice it measures rolls back.**
- **delete `classify_kind`'s substring buckets** (`metal.rs:785-826`) and re-source `op_profile_bucket`'s `kind` from `Route`. `diagnose_kind` (`:835-854`) becomes `Route::Declined(reason)` payload data.
- **feature:** `metal-route-census = ["metal", "instrument"]` in `omega`; `metal-route-census = ["omega?/metal-route-census"]` in `proxima-model-interop`. `route.rs` itself is **unconditional and alloc-tier** (§3: the route decision is emission logic and emission never touches a device); only the census counters are feature-gated.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r08 cargo build -p omega --no-default-features --features alloc
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r08 bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r08 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r08 PROXIMA_MAX_TOKENS=8 flock /tmp/proxima-gpu-measure.lock -c 'cargo nextest run -p proxima-model-interop --features metal,instrument,metal-q4k-fold-scale,metal-route-census --run-ignored all --no-capture -E "test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)"'
  ```
- **expect:**
  - **N1:** `sum(ROUTE_DISPATCHES) == ENCODE_DISPATCH_CALLS` exactly, every step. Inequality is RED.
  - **N2:** `ENCODE_DISPATCH_CALLS / F == op_count == 1196` (R13). `op_count == 0` is RED.
  - **N3:** the per-route counts reproduce R13's bucket table **within the classifier's own known error**: packed-row-blocked 225, cooperative 385, elementwise 547, constant 37, iota 2 (sum 1196). **A route whose count differs from R13's bucket is not automatically RED — it is the finding**, because R12 ROW 263 proves the substring classifier mislabels; any difference is reported as "the census disagrees with `classify_kind` at N ops, here are their nodes."
  - **N4:** `route::of` is total — `Route::Declined` must never be reachable for a plan that emits successfully; the alloc-tier build (`--no-default-features --features alloc`) must compile `route.rs` (state which modules the tier build built, per §3's N==0 warning).
  - **N5 (the cost bound):** `encode_dispatch_ms` delta ON−OFF ≤ 0.0235 ms.
- **predict (one rung ahead: measured at bench → the census is a bench-rung instrument, so it predicts at bench for the *next* card):** `step_wall_ms` with the census on is within R05+R06's band, unmoved beyond CoV. And the load-bearing prediction the census exists to make testable: **the 225 packed-row-blocked ops are all Q4_K/Q6_K weight matvecs and the 385 cooperative reduces are all norms and attention folds** — if the census shows attention folds routing to *packed-row-blocked*, R13's family attribution ("(no named operand)" 681 ops / 11.382 ms) is mis-bucketed and the whole gap decomposition is re-derived.
- **kill:** N5 exceeded → collapse the 8 counters to one and record the route only in the cold per-plan vector, then re-measure. If still over, the census does not go on the dispatch path at all and the row says so.
- **rollback:** `route.rs` and the `match` are behaviour-preserving (the same four gates, same order, now named); the census is default-off. Revert = one commit.
- **blast:** new `omega/src/route.rs`; `omega/src/msl.rs` `render_reduce` + `emit` become a match; `omega/src/metal.rs` loses `classify_kind`'s buckets, gains 8 counters + one line at `:2243`; `proxima-model-interop/src/generate.rs` `op_profile_bucket` sources `kind` from `Route`. **Behaviour-preserving by construction: the emitted MSL must be byte-identical** — proven by `emit_is_deterministic_byte_equal` (`msl.rs:4656`) extended to compare pre- and post-refactor emission for the real openchat program. That golden-source test is the card's safety net.
- **observe:** `ROUTE_DISPATCHES[8]` — **declared NEW**; record site `omega/src/metal.rs:2243`, beside `counter!(ENCODE_DISPATCH_CALLS, 1)`. Cold site: `(NodeId, Route)` vector built in `plan_named` (`metal.rs:945` region), printed as `route_census step=N node=… route=… count=…`.
- **memory gate:** 8 `Counter`s = 8 x 56 B = 448 B static (`counter.rs:98` documents the 56 B layout). The cold vector is `prepared.resolved.len()` x 8 B = 9.6 KB per plan, and `plan_cache_len == 1` (R13) bounds it at one. Slopes and caps unchanged from R00's seal; a device or RSS slope increase is a NEGATIVE.
- **reprove:** the three commands above, ON and OFF, interleaved 3x; the row's claim is N1's identity and N5's bound.
- **log row:** `ROW ZZZ -- the route stops being a substring of the kernel it selected: one enum, decided before emission, censused per dispatch under a measured 5% budget`

---

# PHASE 2 — the plan stops being rebuilt every token

### R09 — the cache-tail mask: a symbol-1 Iota + a `cached_len` scalar leaf, and the inverted harness assertion

- **tier:** worker
- **depends_on:** R00, R08 (so the route census can prove the mask added no new kernel shape)
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r09 -b risc/kv-bucketed-cache 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r09
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r09`
- **opens:**
  - `proxima-tensor/src/spec.rs:6216-6245` — **verified**: `k_even_cache`, `k_odd_cache`, `v_cache` each `input_leaf(... [Extent::Symbolic(1), Static(kv_heads), Static(pairs|head_dim)] ...)`. Symbol 1 **is** `cached_len`.
  - `proxima-model-interop/src/generate.rs:958-977` — **verified**: `resolve_plan`, key `let shape = (symbols[0] as usize, symbols[1] as usize)` at `:966`, `plan_hits += 1` at `:968`, `self.plans.clear()` at `:973` (the `ff749a0` leak bound).
  - `proxima-model-interop/src/generate.rs:1391` — `let symbols = [new_count as u64, cached_len as u64];`
  - `proxima-tensor/src/spec.rs:823-845` — **verified** `causal_mask`: two `Op::Iota{extent: Symbolic(0)}`, `ScalarOp::Greater` with maps `"t->st"` / `"s->st"`, `scalar_constant(f32::NEG_INFINITY)` broadcast `"->stug"`, consumed as `(is_future, "sw->swug")` at `:2610`. **This is the construction to copy, verbatim in shape.**
  - `proxima-model-interop/src/bind.rs:3040-3060` — **verified**: `metal_decode_summary` println at `:3040-3049`, `forward_calls_taken` at `:3051`, `assert_eq!(runtime.plan_hits, 0, "cached_len grows every decode step, so no (new_count, cached_len) shape can repeat within one call")` at `:3052-3055`, `assert_eq!(runtime.plan_misses, forward_calls_taken, ...)` at `:3056-3059`.
  - `proxima-model-interop/src/generate.rs:1313-1391` — **the `named_blocks` assembly**, named: inline in `run_decode_loop`, `Vec::with_capacity(owned + packed + packed_owned + 3 + layer_caches.len()*3)` at `:1313`, `"ids"` at `:1322`, weights `:1323-1331`, `"eps"`/`"rope_cos"`/`"rope_sin"` at `:1332-1334`, and the per-layer KV extension loop at `:1364-1389` calling `LayerCache::named_blocks` (`generate.rs:642-656`).
- **work (a graph change; zero new `Op` variants):**
  1. Symbol 1 stops meaning `cached_len` and starts meaning `bucket_capacity`. In `run_decode_loop`: `let bucket_capacity = (cached_len + new_count).div_ceil(KV_BUCKET_TOKENS) * KV_BUCKET_TOKENS; let symbols = [new_count as u64, bucket_capacity as u64];`
  2. A **`cached_len` scalar leaf** — `Op::Input`, `[Extent::Static(1)]`, name `"cached_len"`, pushed into `named_blocks` at `:1332` beside `"eps"` as `QuantizedBlock::Float32(&[cached_len as f32])`. Nothing new: `"eps"` is exactly this shape today.
  3. A **symbol-1 `Iota`** over the bucketed cache axis — `Op::Iota { dtype: Float32, extent: Extent::Symbolic(1) }`, the same variant `causal_mask` uses at `spec.rs:825-837` with `Symbolic(0)`.
  4. The mask, node-for-node the `causal_mask` shape: `elementwise(GreaterEqual, [(cache_pos, "u->su"), (cached_len_leaf, "->su")])` → `is_unwritten`, combined with the existing `neg_infinity` `scalar_constant` through the same `Select` the causal mask already feeds at `spec.rs:2610`. **Four nodes per attention block. Zero new `Op` variants, zero new `BoundOpKind` variants, zero new `IndexMap` variants.**
  5. `KV_BUCKET_TOKENS` traces to a **new `[kv_cache]` section in `proxima-tensor-runtime.toml`** (`bucket_tokens = 256`, `capacity_tokens = 4096`) through that crate's existing `build.rs` `emit_sizing_consts`/`resolve_int` (the pattern `omega/build.rs:67-120` mirrors) — §12. 256 is not arbitrary: it ports the incumbent's own `n_kv` padding (R8/M6': "The incumbent avoids this by padding n_kv to a 256 multiple and masking").
  6. **Invert the harness assertion**, in this same change (mandatory — the assertion is currently a *correct* statement about a graph this card is changing):
     ```rust
     // was: assert_eq!(runtime.plan_hits, 0, "cached_len grows every decode step, ...")
     assert!(runtime.plan_hits >= forward_calls_taken - 2,
        "cached_len is bucketed to KV_BUCKET_TOKENS, so every step but the first (and any bucket rollover) must reuse its plan");
     assert_eq!(runtime.plan_hits + runtime.plan_misses, forward_calls_taken,
        "every forward step either reuses a plan or builds exactly one");
     assert!(runtime.plan_misses >= 1, "the first step must build a plan; zero misses means no forward ran");
     ```
     Note the `- 2`: one miss for the first step, one for a possible bucket rollover — **eos-invariant and rollover-invariant**, no literal step count anywhere.
- **feature:** `kv-bucketed-cache` in `proxima-tensor`; forwarded as `kv-bucketed-cache = ["proxima-tensor/kv-bucketed-cache"]` in **both** `omega` and `proxima-model-interop`.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r09 bash scripts/proxima-tensor-gate.sh && bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r09 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r09 PROXIMA_MAX_TOKENS=8 flock /tmp/proxima-gpu-measure.lock -c 'cargo nextest run -p proxima-model-interop --features metal,instrument,metal-q4k-fold-scale,metal-route-census,kv-bucketed-cache --run-ignored all --no-capture -E "test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)"'
  ```
- **expect:**
  - **N1:** `plan_hits >= F - 2` and `plan_hits + plan_misses == F` and `plan_misses >= 1`. **`plan_hits == 0` is now RED** — the exact inversion of R13's confirmed `plan_hits=0 plan_misses=8`.
  - **N2:** `plan_cache_len == 1` still (the `:973` clear-on-miss bound holds; a rollover drops the old plan).
  - **N3:** `generated_text == "Here is a simple Python function that returns"` and the greedy oracle at `bind.rs:2797-2803` (`2651` / `"known"`) still passes — **the mask must be mathematically exact**, not approximate. §14: masked-out lanes contribute `exp(-inf) = 0` to the softmax denominator, so the result is bit-comparable to the unbucketed graph. A CPU parity test asserts bucketed == unbucketed on `cpu::evaluate` to 0 ULP.
  - **N4:** `op_count` **rises** by 4 nodes per attention block (128 for 32 layers) → ~1324, and `reduce-*` op counts rise because the masked cache axis is now `bucket_capacity` (256) instead of `cached_len` (~37). **This card trades GPU work for CPU work and says so up front.**
  - **N5:** `route_census` shows **no new route** — the four mask nodes are `Iota`, `Constant`, `Elementwise`, `Elementwise`. A new route appearing is RED (the mask must not create a kernel shape).
- **predict (one rung ahead: measured at bench → predicted at bench is forbidden; this card measures at bench, so it pre-registers the *milli* consequence it will then verify, and the bench number is the outcome, not the prediction).** Precisely: measured at **milli** (per-op, which the card runs first), predicted at **bench**: `prepare_ms` 1.97 → **≤ 0.30** and `op_setup_ms` 3.9 → unchanged (R10 owns that one), so orchestration 11.0 → **[9.0, 9.3]**; simultaneously the cache axis grows from ~37 to 256, adding GPU work to the attention reduces (R13: `kv_cache.v`/`k_odd`/`k_even` = 1.559/0.783/0.774 ms at ~37 wide → **≤ 7x** those at 256 wide, band **[15, 25] ms added**). **Net prediction: `step_wall_ms` gets WORSE, into [63, 72].** This card is pre-registered as a **timing loss** whose payoff is only realized when R12 makes the cache device-resident and R13 collapses the two ranges to one. Saying so before measuring is the point: a card that predicts a win and delivers a loss kills the climb; this one predicts the loss.
- **kill:** if `plan_hits` does not reach `F - 2`, the key is still moving — dump `symbols` per step and find the other varying symbol. If the CPU parity test shows any ULP difference, the mask is wrong and the card dies on correctness before any timing is read.
- **rollback:** default-off feature `kv-bucketed-cache`; omitting it restores `symbols[1] = cached_len` and the original assertion. **The assertion inversion must be feature-gated too** — `#[cfg(feature = "kv-bucketed-cache")]` on the new asserts and `#[cfg(not(...))]` on the old ones — so the OFF build still asserts `plan_hits == 0` and both arms stay green. That is what makes every commit a green bisect point.
- **blast:** `proxima-tensor/src/spec.rs` (`append_mistral_cached_layer` `:2336-2865` gains 4 nodes; the mask helper sits beside `causal_mask` `:823`), `proxima-tensor-runtime.toml` (+1 section), `proxima-tensor/build.rs` (+2 keys), `proxima-model-interop/src/generate.rs` (`:1332` named_blocks, `:1391` symbols), `proxima-model-interop/src/bind.rs:3052-3059` (the assertion). **`append_qwen35_*` layer builders are NOT touched** — the feature scopes to the mistral cached layer only, so the qwen3.5 hybrid path (`0c3bd4f`, spec.rs +8735) is unaffected.
- **observe:** `plan_hits` / `plan_misses` / `plan_cache_len` in `metal_decode_summary` (`bind.rs:3040-3049`) and `token_breakdown_metal` (`generate.rs:1732`, `:1764`) — both exist on main. `prepare_ms` (`generate.rs:1723`).
- **memory gate:** bucketing to 256 grows the KV upload from `cached_len * 262_144` to `256 * 262_144 = 67_108_864 B` — a **known, bounded, pre-registered increase**. `DEVICE_CAP_BYTES` at `capacity_tokens = 256` = **4_247_525_888**. `kv_cache_upload_bytes` becomes constant at 67.1 MB/token instead of R13's linear 8.13 → 9.70 MB — flat is the point (slope → 0), but the absolute is higher, and the row must headline that trade. RSS caps unchanged. **If `device_allocated_bytes` exceeds 4_247_525_888 the card rolls back regardless of `plan_hits`.**
- **reprove:** the two commands above, ON and OFF, interleaved 3x, plus the CPU 0-ULP parity test.
- **log row:** `ROW ZZZ -- plan_hits was 0 by construction, not by bug: bucketing cached_len and masking the tail inverts the harness assertion, and costs GPU time until the cache goes device-resident`

---

### R10 — plan-stable preallocation: output buffers and uniforms allocated once, not 1196 times per token

- **tier:** worker
- **depends_on:** R09 (there is nothing to preallocate *across* until the plan is stable)
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r10 -b risc/plan-stable-buffers 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r10
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r10`
- **opens:** `omega/src/metal.rs:2179-2252` `encode_op` — per op per token: `kernel_cache_key` `:2193`, `kernel_dispatch_shape` `:2194`, `pipeline_for` (cached, `:1402-1426`), **`allocate_buffer` for the output `:2210`**, **`upload_uniforms` `:2211`** (reuse path at `:2075`, `UNIFORM_BUFFER_REUSES` at `:2069`), fault buffer, bind, dispatch `:2225-2248`; `omega/src/metal.rs:449-568` `execute_plan` (one encoder, one commit, one wait; retires per position). M6'': **1196 `newBufferWithLength` + 1196 uniform uploads per token = the 3.9 ms `op_setup`** (R13).
- **work:** hang a `BufferArena` off the cached `Plan` — a `Vec<MetalBuffer>` indexed by resolved position, allocated once when the plan is built, reused on every subsequent execute. Same for uniforms: one uniform buffer per position, written in place (the `UNIFORM_BUFFER_REUSES` mechanism at `:2069-2078` already exists and proves the shape is legal). Retirement (`execute_plan` retires per position) becomes a no-op on arena-owned buffers. **Strict O(1) per op in steady state**: index into a `Vec`, no allocation, no hashing (AGENTS.md §hot-path: "do not heap allocate in query, scoring, or inner-loop traversal paths"; disciplined-component gate 11).
- **feature:** `metal-plan-stable-buffers = ["metal"]`; forwarded `metal-plan-stable-buffers = ["omega?/metal-plan-stable-buffers"]`.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r10 bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r10 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r10 PROXIMA_MAX_TOKENS=8 flock /tmp/proxima-gpu-measure.lock -c 'cargo nextest run -p proxima-model-interop --features metal,instrument,metal-q4k-fold-scale,metal-route-census,kv-bucketed-cache,metal-plan-stable-buffers --run-ignored all --no-capture -E "test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)"'
  ```
- **expect:** **N1:** a new counter `OUTPUT_BUFFER_ALLOCATIONS` equals `op_count` on the first step and **0 on every steady step**; a nonzero steady value is RED and names the position that reallocated. **N2:** `UNIFORM_BUFFER_REUSES` (`metal.rs:2069`) rises to `op_count` per steady step. **N3:** `plan_hits >= F - 2` still (this card is a no-op without R09). **N4:** `generated_text` unchanged.
- **predict (one rung ahead: measured at milli → predicted at bench):** `op_setup_ms` 3.9 → **[0.4, 0.8]** (what remains is `kernel_cache_key` + `pipeline_for` + bind, R13's `pipeline_lookup` is already only 0.04); orchestration 11.0 → **[5.5, 6.2]**; therefore `step_wall_ms` improves by **[3.1, 3.5] ms** from whatever R09 left it at.
- **kill:** if `OUTPUT_BUFFER_ALLOCATIONS` is nonzero in steady state, the plan is not stable — that is an R09 defect surfacing here, and the work item goes back to R09, not forward.
- **rollback:** default-off feature. The arena is additive: `allocate_buffer` at `:2210` becomes `arena.get_or_allocate(position)`, and with the feature off `get_or_allocate` is `allocate_buffer`.
- **blast:** `omega/src/metal.rs` (`encode_op` two call sites, `Plan` gains one field, `execute_plan`'s retire loop). No IR change. No graph change. No emitter change.
- **observe:** `op_setup_ms` and `op_setup_calls` (`generate.rs:1723`), `UNIFORM_BUFFER_REUSES` (`metal.rs:2069`, exists), `OUTPUT_BUFFER_ALLOCATIONS` — **declared NEW**, record site: `omega/src/metal.rs:2210`, the `allocate_buffer` call in `encode_op`.
- **memory gate:** **this is the highest-risk memory card in the plan.** The arena holds every intermediate output buffer alive for the whole plan lifetime instead of retiring them per position. Compute the cap explicitly: sum over all 1196 resolved ops of `product(extents) * 4 B`. **The card's FIRST action is to print that sum, before allocating anything**, and compare it against `DEVICE_CAP_BYTES - 4_140_417_024 - 256*262_144 = 40_000_000`. If the naive arena exceeds 40 MB, the arena must be **liveness-partitioned** (buffers reused across positions whose live ranges do not overlap — a graph-coloring over the resolved order, computed once per plan, cold). The row records the arena's peak bytes and its reuse factor. **`device_allocated_bytes` above `DEVICE_CAP_BYTES` rolls the card back regardless of the `op_setup` win** — owner rule, and this is exactly the shape of the two prior memory failures (R13).
- **reprove:** the two commands above, ON and OFF, interleaved 3x; the row's claim is N1 (zero steady allocations) plus the arena peak-bytes number.
- **log row:** `ROW ZZZ -- 1196 device buffers and 1196 uniform uploads per token become zero: the arena hangs off the plan, liveness-partitioned, with its peak bytes on the row`

---

# PHASE 3 — write placement, and the graph collapses

### R11 — the MSL scatter emitter: `EmitError::ScatterNotSupported` stops firing on Metal

- **tier:** worker
- **depends_on:** R08 (scatter is a route), R01
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r11 -b risc/metal-scatter 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r11
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r11`
- **opens:**
  - `omega/src/msl.rs:923-946` `validate` — **verified**: `if out_scatter.is_some() { return Err(EmitError::ScatterNotSupported { node }) }` at `:932-936`. Siblings: `wgsl.rs:364`, `cuda.rs:241`; the error is defined at `error.rs:53`.
  - `proxima-tensor/src/bind.rs:245-262` — **verified**: `out_layout: Layout` and `out_scatter: Option<Lookup>` with a field doc that already specifies the semantics: *"the destination axis is fetched from `indices` at `index_layout`, bounds-checked against `extent`, then scaled by `element_stride` and added to `out_layout`'s own offset — the exact mirror of how a gathered operand's `Lookup` already contributes to a read offset."*
  - `proxima-tensor/src/cpu.rs:6911-6969` `run_reduce_scatter` — **verified**: the reference implementation, with the worked example in its doc (`src=[10,20,30,40]`, `idx=[2,0,2,1]`, dest extent 3, `Add`/`Zero` → `out=[20,40,40]`).
  - `proxima-tensor/src/map.rs:109-131` — **verified**: the write-direction convention (`offset` at `gathered_dim` carries the destination axis's static extent) and the note *"the CPU interpreter runs the reduce loop strictly sequentially, so a scatter never needs atomics"* — **which is exactly the clause that does NOT hold on a GPU** and is therefore the design constraint of this card.
  - `omega/src/msl.rs:2361, 2734, 3079, 3460, 3541` (`u.out_base` already emitted) and `:2216, 3502` (the `long out_base` uniform field) — **verified**: the write-side offset plumbing already exists.
  - `omega/src/msl.rs:1517-1560` `grid_threads`; `proxima-tensor/src/shape.rs:493-511` `scatter_output_shape`.

**Placement — the form that needs no new type, and why.** Two candidate expressions:

- **(i) static write offset:** allow `Reduce.out_map`'s affine axes a nonzero `offset` and relax `project_output_shape` (`shape.rs:468-484`, verified: `[term] if term.coeff == 1 => Ok(iter_extents[term.axis])`, else `NotLowerable{"reduce output maps must be pure projections in v1"}`). This *needs a destination extent from somewhere*, and `map.rs:109-131` records that a `Reduce`-wide field for exactly that was **already rejected on blast-radius grounds** and the scatter convention chosen instead. Re-opening it re-litigates a closed adjudication.
- **(ii) data-dependent scatter:** use `IndexMap::scatter` (`map.rs:175`, verified, with `scatter_extent` `:209` and `as_gather_from_output` `:238`). It is **already** in the IR, **already** in shape inference (`scatter_output_shape` `shape.rs:493`), **already** in bind (`build_scatter_out_layout` `bind.rs:1011`, `out_scatter` `bind.rs:253`), and **already** implemented on CPU (`cpu.rs:6911`). The only missing piece is the GPU emitter.

**(ii) wins on §1: write the expression — it compiles today, on CPU, with the answer in its own doctest.** This card implements the missing emitter. **Zero new `Op` variants, zero new `BoundOpKind` variants, zero new `IndexMap` variants, zero new types in `proxima-tensor`.**

**The one field addition, with both questions answered in-line.** The MSL uniform struct gains three `long`s beside the `out_base` it already has. *Pipe question:* the uniform struct is a POD ABI record read by a kernel — no stages, no dataflow; `backend.rs:1-52`'s 2026-08-30 adjudication already covers this boundary. *Relocation question — the call site both ways:*

```rust
// WAY A — three longs on the struct that already carries out_base:
uniforms.out_base          = out_layout.base;                 // exists today
uniforms.out_index_offset  = target.index_layout.base;        // new
uniforms.out_element_stride= target.element_stride;           // new
uniforms.out_extent        = target.extent;                   // new
// (bind unchanged: one uniform buffer, one bind slot, UNIFORM_BUFFER_REUSES path at metal.rs:2075 intact)

// WAY B — a new ScatterUniforms type in a second buffer:
uniforms.out_base = out_layout.base;
let scatter = ScatterUniforms { index_offset: target.index_layout.base,
                                element_stride: target.element_stride, extent: target.extent };
let scatter_buf = upload_uniforms(&device, &scatter)?;         // a SECOND allocation per dispatch
encoder.setBuffer(Some(&scatter_buf), 0, next_slot);           // a SECOND bind slot per dispatch
```

Way B costs one buffer allocation and one bind slot **on the 100% path** and forces `UNIFORM_BUFFER_REUSES` (`metal.rs:2069`) to be duplicated for the second buffer. Way A is three `long`s in a struct that already carries `out_base`. **The new type is a relocation that costs a bind slot — Way A.**

**The atomics question, which the CPU doc explicitly does not answer for GPU.** A scatter whose destinations collide needs an atomic fold on GPU. This card supports the **injective** case only, and proves injectivity structurally rather than assuming it: `Route::ScatterReduce` is selected only when `keep == Keep::Reduce`, `init != FirstElement` (already rejected at shape inference per `cpu.rs:6903-6909`), and the scatter's iteration space maps one-to-one onto destinations. The emitter asserts this at emit time by checking the indices leaf is declared strictly-monotonic (a new `Op::Input` name-convention, `"*.write_row"`), and **`Route::Declined(ScatterMayCollide)`** falls back to `EmitError::ScatterNotSupported` for every other scatter — so the non-injective case is *still* declined, honestly, with a reason, instead of silently racing. A colliding-scatter GPU path is a named future work item, not this card.

- **feature:** `metal-scatter = ["metal"]`; forwarded `metal-scatter = ["omega?/metal-scatter"]`.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r11 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r11 cargo build -p omega --no-default-features --features alloc
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r11 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r11 bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r11 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r11 flock /tmp/proxima-gpu-measure.lock -c 'cargo nextest run -p omega --features metal,metal-scatter,cpu -E "test(scatter)"'
  ```
- **expect:** **N1:** a Metal-vs-CPU parity test over `run_reduce_scatter`'s own doc worked example (`src=[10,20,30,40]`, `idx=[2,0,2,1]`, dest extent 3, `Add`/`Zero` → `[20,40,40]`) — that example is the spec **and** the test (§17, §9). **N2:** a parity test at the real KV shape: `[new_count=1, kv_heads=8, head_dim=128]` scattered into `[capacity=256, 8, 128]` at row `cached_len`, Metal vs `cpu::evaluate`, **0 ULP**. **N3:** `cargo nextest -E "test(scatter)"` reports **`ran_count >= 6`; `ran_count == 0` is RED** — this is precisely the trap `omega/Cargo.toml`'s `default` comment records (13 `metal_parity` cases compiled to zero behind a default-off feature). **N4:** with the feature OFF, `EmitError::ScatterNotSupported` still fires — the existing negative test must stay green.
- **predict (one rung ahead: measured at nano → predicted at micro):** a scatter-reduce writing `[1,8,128]` into `[256,8,128]` costs the same GPU time as the equivalent affine reduce writing `[1,8,128]` into `[1,8,128]`, **within 10%** on `omega/benches/metal_vs_cpu.rs` — because the only difference is one extra indexed load and one add on the output address, and the traffic is identical. A larger cost means the index load is not coalescing and the work item is the index layout.
- **kill:** any ULP difference vs `cpu::evaluate` kills the card on correctness (§14). A scatter that emits but produces a race under repeated execution (run the parity test 100x, assert byte-identical every time) kills it.
- **rollback:** default-off feature; with it off, `validate` returns `ScatterNotSupported` exactly as today. WGSL and CUDA are untouched and keep declining (R14 gives them the route).
- **blast:** `omega/src/msl.rs` (`validate` `:932`, a new `push_scatter_reduce_body`, the uniform struct at `:2216`/`:3502`, `grid_threads`'s reduce arm, `route.rs` gains `ScatterReduce` + `Declined(ScatterMayCollide)`), `omega/src/error.rs` (a reason on `ScatterNotSupported`). **`proxima-tensor` is not touched at all** — that is the headline of this card.
- **observe:** `route_census` (from R08) showing `Route::ScatterReduce` with a nonzero count; `EmitError::ScatterNotSupported` count going to zero for the KV writes.
- **memory gate:** the scatter's destination is a caller-owned buffer that already exists (the KV cache); the emitter allocates nothing beyond three uniform `long`s (+24 B per uniform buffer). Caps and slopes identical to R10's; any increase is a NEGATIVE.
- **reprove:** the three commands above; the row's claim is N1's worked example and N2's 0-ULP KV-shape parity.
- **log row:** `ROW ZZZ -- write placement needed no new Op: IndexMap::scatter already existed, shipped on CPU, and was rejected by three emitters — Metal now emits it, injective only, declining collisions by name`

---

### R12 — the KV cache becomes device-resident and is written in place, as the incumbent does

- **tier:** worker
- **depends_on:** R09 (bucketed capacity), R10 (arena), R11 (the scatter emitter)
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r12 -b risc/kv-device-resident-write 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r12
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r12`
- **opens:**
  - `proxima-model-interop/src/generate.rs:621-656` — **verified**: `struct LayerCache { k_even, k_odd, v: Vec<f32> }` at `:621-625`, `fn append` = three `extend_from_slice` at `:636-640`, `fn named_blocks` handing the whole `Vec` as `QuantizedBlock::Float32` at `:642-656`.
  - `proxima-model-interop/src/generate.rs:1313-1391` — **the `named_blocks` assembly**, and specifically the per-layer KV extension loop at `:1364-1389`; `kv_cache_upload_elements` at `:1346-1361` whose own comment states *"this is the full `cached_len`-sized array re-bound as a model input every single step -- not the `new_count`-sized increment"*.
  - `proxima-model-interop/src/generate.rs:1559` — `cached_len += new_count`.
  - `omega/src/metal.rs:1744-1815` `register_checkpoint_mapping` + `omega/src/backend.rs:402-414` — the shipped mechanism for "one device buffer, addressed by offset", landed by `7d09145`. **This card reuses it for KV.**
  - `omega/src/metal.rs:1848-1892` `NOCOPY_BUFFERS: BTreeMap<(usize, usize), MetalBuffer>` keyed `(pointer, byte_length)`; `:1903` `upload_block_no_copy_uncached` (M2': `23e2e5e` routes non-resident blocks here, so KV creates a **fresh** no-copy buffer every token — no cache growth, but no reuse either); `:350-362` `mark_resident` (classifies **by name**); `:991-1000` the strict `found != expected` extent check.
  - `proxima-tensor/src/align.rs:42-46, :69` `AlignedBuffer` — **zero production callers** (R11/M2); `omega/src/metal.rs:1616` `is_page_aligned`.
  - The incumbent, R8: KV allocated **once** on the device (`llama-kv-cache-unified.cpp:74-118`), written per token by `ggml_cpy(k_cur, ggml_view_1d(k, n_tokens*n_embd_k_gqa, row_size*head_cur))` (`:749-788`) — **an in-place write at a byte offset into a persistent buffer** — and read back as a `ggml_view_3d` no-op.
- **work:**
  1. `LayerCache`'s three `Vec<f32>` become three `AlignedBuffer`s sized at `capacity_tokens` (`proxima-tensor-runtime.toml` `[kv_cache].capacity_tokens`, landed by R09). `AlignedBuffer` (`align.rs:69`) gets its **first production caller** — §1 in action: the primitive existed and was unused, so nothing new is minted. Page-aligned (`is_page_aligned` `metal.rs:1616`) is what makes the no-copy wrap legal.
  2. Register the three buffers **once** through the `register_checkpoint_mapping` mechanism, so the KV read leaf and the KV write destination are the **same device buffer**. That is item 7's "driver-level persistent-buffer alias" — a driver fact, not an IR concept.
  3. `LayerCache::append` (`generate.rs:636-640`) is **deleted**. The write happens on-device via R11's scatter, whose `indices` leaf is `"kv_cache.{l}.write_row"` = `[cached_len, cached_len+1, …]`, pushed into `named_blocks` beside `"eps"`.
  4. `LayerCache::named_blocks` (`:642-656`) hands the **capacity-sized** buffer, unchanged every token — so `NOCOPY_BUFFERS`' `(pointer, byte_length)` key (`metal.rs:1848`) **hits** instead of missing on growth (M2), and `mark_resident` (`:350-362`) adds the KV names to `resident_names` so `23e2e5e`'s non-resident routing no longer sends them to `upload_block_no_copy_uncached` (`:1903`).
  5. The graph: `append_mistral_cached_layer` (`spec.rs:2336-2865`) gains a scatter-`Reduce` per KV component whose `out_map` is `IndexMap::scatter(write_row, index_map, iter_rank, non_scattered=[(1, head_axis), (2, dim_axis)], gathered_dim=0, destination_extent=capacity_tokens)` — the exact constructor at `map.rs:175-199`.
- **feature:** `kv-scatter-write` in `proxima-tensor` (the graph half) + `metal-kv-resident` in `omega` (the driver half); both forwarded into `proxima-model-interop`. Two features because they are two independently-revertable concerns (AGENTS.md §pr-sequencing: independent + orthogonal).
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r12 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r12 bash scripts/proxima-tensor-gate.sh && bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r12 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r12 PROXIMA_MAX_TOKENS=8 flock /tmp/proxima-gpu-measure.lock -c 'cargo nextest run -p proxima-model-interop --features metal,instrument,metal-q4k-fold-scale,metal-route-census,kv-bucketed-cache,metal-plan-stable-buffers,metal-scatter,kv-scatter-write,metal-kv-resident --run-ignored all --no-capture -E "test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)"'
  ```
- **expect:** **N1:** `kv_cache_upload_bytes == 0` on every steady step (R13 baseline: 8.13 → 9.70 MB, +262,144 B/token). **Nonzero is RED.** **N2:** `nocopy_reuses` rises to `32 layers x 3 = 96` per steady step; `copying_uploads` for KV names == 0. **N3:** `resident_uploads` counts the KV names once, at the first step only. **N4:** `generated_text` unchanged and the `2651`/`"known"` oracle green — **§14: the incumbent's captured answer is the correctness gate for an in-place-write graph.** **N5:** `plan_hits >= F - 2` still holds. **N6:** `block_upload_bytes_copied` (from R01) drops by the KV share.
- **predict (one rung ahead: measured at milli → predicted at bench):** removing the KV re-upload removes R13's `block_upload` share attributable to KV plus the host-side `extend_from_slice` and the `named_blocks_kv_ms` slice; combined with R09's now-paid-off bucketing, `step_wall_ms` lands in **[48, 53]** = **2.74-3.02x**, and `gpu_exec_ms` in **[42, 45]**. The KV attention reduces (R13: 1.559 + 0.783 + 0.774 = 3.1 ms at ~37-wide) grow to 256-wide but now read a **device-resident** buffer, so the band already includes that.
- **kill:** if the scatter write and the read of the same buffer within one command buffer produce stale data, there is a missing memory barrier — R8 records the incumbent has *no* barrier/concurrency API at this checkout with `n_cb=1`, and we have one encoder / one command buffer (`metal.rs:449-568`), so ordering within an encoder is the guarantee we rely on. If it does not hold, the write moves to its **own** dispatch ordered before the read, which the single encoder already sequences. If *that* fails, the card dies and the row records the ordering finding.
- **rollback:** two independent default-off features. `kv-scatter-write` off restores the host `append`; `metal-kv-resident` off restores per-token `named_blocks` of a growing `Vec`. Either reverts alone.
- **blast:** `proxima-model-interop/src/generate.rs` (`LayerCache` `:621-656`, the `named_blocks` KV loop `:1364-1389`, `cached_len` `:1559`), `proxima-tensor/src/spec.rs` (`append_mistral_cached_layer`), `omega/src/metal.rs` (`mark_resident` name set, KV registration). **The qwen3.5 `DenseAttention` and `Ssm` cache states (`generate.rs:687`, `:738` `named_blocks`) are NOT touched** — the feature scopes to `LayerCacheState::Attention`, and the `unreachable!("cache_names/layer_caches built from the same layer_roots, in lockstep")` at `:1386-1388` stays intact.
- **observe:** `kv_cache_upload_bytes` (`generate.rs:1657`), `nocopy_reuses` / `resident_uploads` / `resident_reuses` / `copying_uploads` / `mapping_offset_uploads` (`generate.rs:1729-1730`), `nocopy_cache_len` (`metal.rs:1852-1858`) — all exist on main.
- **memory gate:** **the second-highest-risk memory card, and the site of a prior failure.** R13 records a 34 GB KV allocation from the `context_length` default (worktree only). The capacity is now `[kv_cache].capacity_tokens` from `proxima-tensor-runtime.toml`, **not** `context_length` — and `build.rs` `require_nonzero` plus a new `require_at_most(65536)` reject a runaway at **build time**, which is where a 34 GB allocation should have been caught. Bytes: `capacity_tokens * 262_144`; at 4096 = **1_073_741_824 B**, so `DEVICE_CAP_BYTES = 4_140_417_024 + 1_073_741_824 + 40_000_000 = 5_254_158_848`. At 256 = 4_247_525_888. **The card runs at `capacity_tokens = 256` first and prints `device_allocated_bytes` before raising it.** Slopes must go to **0** — a device-resident cache that still grows per token has not become resident, and that is RED on the slope, not on the timing.
- **reprove:** the two commands above, ON and OFF for each of the two features independently (three arms: both off, driver only, both on), interleaved 3x.
- **log row:** `ROW ZZZ -- the KV cache stops round-tripping through the host: one device buffer per layer at capacity, written in place by a scatter, read as an alias — the incumbent's own mechanism`

---

### R13 — the attention graph collapses to one range: ≤23 real ops/layer

- **tier:** worker
- **depends_on:** R09, R11, R12
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r13 -b risc/single-range-attention 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r13
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r13`
- **opens:** `proxima-tensor/src/spec.rs:2303-2319` — the doc that **names the constraint being removed**: *"Two Reduce blocks — one per source — combine through online-softmax arithmetic … rather than a literal concatenation: `Reduce::out_map` must stay a pure projection (`shape::project_output_shape`'s own doc), so nothing upstream of a reduce can splice two tensors into one axis."*; `spec.rs:2336-2865` `append_mistral_cached_layer` (530 lines, 25 args), sole caller `:6282`; `spec.rs:2596-2720` (the two-range combine), `:2616-2617` (the comment), `:2610` (the mask consumption `(is_future, "sw->swug")`); `proxima-tensor/src/shape.rs:468-484` `project_output_shape`.
- **work:** with R12 landed, the new token's K/V **are already in the cache buffer before the attention reduce runs** — so there is only ever ONE source range, and the online-softmax combine over two ranges (and its even/odd RoPE doubling) is dead code. Delete it: one `Reduce` over `[0, bucket_capacity)` with the R09 tail mask and the existing causal mask. The `spec.rs:2303-2319` doc is rewritten to record that the constraint it names was satisfied by scatter, not by relaxing `project_output_shape` — **`shape.rs:468-484` is not touched**, which is the proof the constraint was routed around rather than weakened.
- **feature:** `attention-single-range` in `proxima-tensor`, forwarded to `omega` and `proxima-model-interop`.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r13 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r13 bash scripts/proxima-tensor-gate.sh && bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r13 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r13 PROXIMA_MAX_TOKENS=8 flock /tmp/proxima-gpu-measure.lock -c 'cargo nextest run -p proxima-model-interop --features metal,instrument,metal-q4k-fold-scale,metal-route-census,kv-bucketed-cache,metal-plan-stable-buffers,metal-scatter,kv-scatter-write,metal-kv-resident,attention-single-range --run-ignored all --no-capture -E "test(runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache)"'
  ```
- **expect:** **N1:** real ops per Llama layer **≤ 23** — the incumbent's count at its own checkout (R8, enumerated from `llama-model.cpp:4691-4845` + `llama-graph.cpp:464-497, 514-616, 1071-1121, 1205-1253`). Derived total: `32 * 23 + get_rows + rms_norm + mul + output mul_mat = 740`. So **`op_count <= 780`** (a 5% allowance for our `Iota`/`Constant` control nodes, which R13 measures at 39 ops / 0.169 ms). **`op_count > 780` fails item 8. `op_count == 0` is RED.** Note R8's correction: the earlier "15/layer, ~483" memory figure was **wrong**; 23/layer and ~740 is the real bar.
  - **N2:** `generated_text` unchanged, `2651`/`"known"` green. §14 binds hardest here: this card changes the arithmetic order of a softmax, and the incumbent's captured answer is the only admissible oracle.
  - **N3:** a CPU parity test asserting single-range == two-range on `cpu::evaluate` to within 1e-6 (not 0 ULP — the summation order genuinely changes; the tolerance is stated and justified, and the *token* must still be bit-identical).
  - **N4:** `route_census` shows the elementwise count collapsing from 547 toward the incumbent's shape.
- **predict (one rung ahead: measured at milli → predicted at bench):** removing the duplicated range removes most of R13's `elementwise` 7.350 ms and a large share of the `"(no named operand)"` 681 ops / 11.382 ms. Band: `gpu_exec_ms` → **[32, 38]**, `step_wall_ms` → **[40, 46]** = **2.28-2.63x**. R12's own control gives independent support: their feature-off arm ran 35.117 ms GPU with the paired Q4_K body already in the tree.
- **kill:** any change to `generated_text` or the greedy oracle. A single-range graph that is faster and wrong is a loss (§14); revert and report the numerical difference, do not argue the incumbent is wrong.
- **rollback:** default-off feature; `append_mistral_cached_layer`'s two-range body stays in the file under `#[cfg(not(feature = "attention-single-range"))]` until a later row deletes it, so both arms stay green and every commit bisects.
- **blast:** `proxima-tensor/src/spec.rs` only (`append_mistral_cached_layer` and its doc). Zero emitter change, zero driver change — **the whole point: the duplication was a graph defect upstream of any backend** (R5's item 5).
- **observe:** `op_count` (`generate.rs:107`), `op_profile_bucket kind=elementwise op_count` (R13 baseline 547 / 7.350 ms), `ENCODE_DISPATCH_CALLS`.
- **memory gate:** fewer nodes → fewer arena buffers → `device_allocated_bytes` must **decrease** vs R12. An increase is a NEGATIVE. Caps and slopes per the global formula at the card's `capacity_tokens`.
- **reprove:** the two commands above, ON and OFF, interleaved 3x, plus the CPU parity test.
- **log row:** `ROW ZZZ -- attention was duplicated because the graph could not write in place: with scatter placement the two ranges become one, at or under the incumbent's 23 ops per layer`

---

# PHASE 4 — one emitter core, every backend covers every kind, one plan

### R14 — one emitter core over the 4 `BoundOpKind`s; backend-specific TEXT only

- **tier:** worker
- **depends_on:** R08 (the route enum is the core's dispatch), R11 (scatter is a route every backend must answer)
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r14 -b risc/emitter-core 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r14
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r14`
- **opens:** R5's census, re-verified: `omega/src/` = `msl.rs` 4712 + `wgsl.rs` 1929 + `cuda.rs` 1838 + `metal.rs` 2385 + `wgpu_driver.rs` 872 + `backend.rs` 615 + `sized.rs` 45 + `error.rs` 84 + `lib.rs` 73 = **12,553 lines**, of which exactly **3 types + 2 fns** are shared (`Binding`, `PackedCodec`, `PackedOperands`, `gather_count`, `gather_slots`; `wgsl.rs:105`, `cuda.rs:66`) and **~26 functions x 3 backends ≈ 78 near-duplicates** (validate, reduction_dims, bindings, grid_threads, entry_name, scalar_op_expr, fold_init_tokens, push_body_steps, preamble, kernel_signature, gather helpers, operand_read, render_*, reduce_is_cooperative). Coverage asymmetry, verified: `cuda.rs:146-183` `emit_cuda` **rejects Iota and Constant** (`CudaUnsupportedOpKind`) and has serial + cooperative reduce only; `wgsl.rs` covers all 5 kinds but has no tiled-GEMM and no packed row-block; **only Metal has all 8 shapes**.
- **work:** a `Dialect` trait with **only** the text-producing methods (`scalar_op_expr`, `preamble`, `kernel_signature`, `entry_name`, `simd_reduce_intrinsic`, `threadgroup_barrier`, `atomic_or`), implemented three times. Everything structural — `validate`, `reduction_dims`, `bindings`, `grid_threads`, `push_body_steps`, `operand_read`, the four `render_*`, and `route::of` — moves to a generic core over `BoundOpKind` and `Route`. **§20: static dispatch, generic parameter, no `Box<dyn Dialect>`** — the three backends are a closed set known at compile time, so a generic parameter is the box-free answer and the enum-vs-generic call is made in favor of the generic because each backend's methods are monomorphized into a hot string builder.
  Then close the coverage gap: CUDA gains `Iota` and `Constant` (both are one line of generated text each — `CudaUnsupportedOpKind` at `cuda.rs:146-183` is a *gap*, not a limitation); WGSL and CUDA gain the packed-row-block and tiled-GEMM routes, or, where a route genuinely has no dialect expression (no `simdgroup_matrix` outside Metal), they return `Route::Declined(NoDialectIntrinsic)` — **a named decline, exhaustively matched, never a silent absence**. That is item 4: every backend answers for every kind, and "answers" includes a reason.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r14 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r14 cargo build -p omega --no-default-features --features alloc
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r14 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r14 bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r14 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r14 cargo nextest run -p omega --all-features -E "test(wgpu_parity) or test(cuda) or test(metal_parity)"
  ```
- **expect:** **N1 (the golden-source test):** the emitted MSL for the real openchat program is **byte-identical** before and after the refactor — extend `emit_is_deterministic_byte_equal` (`msl.rs:4656`, verified) to compare against a checked-in golden. A refactor that changes one byte of emitted text is not behaviour-preserving and is RED. **N2:** `omega-gate.sh [3/6] ran_count` **strictly greater** than main's, since CUDA and WGSL gain kinds and therefore cases. **N3:** the alloc-tier build (`--no-default-features --features alloc`) compiles the core, `route.rs`, and all three dialects — **state which modules it built** (§3's N==0 warning: a tier build whose modules are all gated off compiles zero lines of the claim). **N4:** a compile-time exhaustiveness proof: a `match` over `(Backend, Route)` with no wildcard arm, so adding a route without answering it in every dialect **fails to compile**. **N5:** duplicated-function count drops from ~78 toward ~21 (7 dialect methods x 3).
- **predict (one rung ahead: this card measures at micro — `metal_vs_cpu.rs`, the registered-but-UNRUN bench, finally run — and predicts at milli):** per-op `gpu_ns` in `profiles_one_real_decode_step_by_per_op_gpu_time` is **unchanged within 1%** for every one of the 1196 (post-R13: ≤780) ops, because N1 says the emitted text is byte-identical. **Any per-op movement falsifies N1** and is the more sensitive test of the two.
- **kill:** N1 fails → the refactor changed emission; bisect the dialect method that moved and either restore its text or record the change as its own row with its own parity evidence. Never accept "the new text is equivalent" without the byte comparison (§6: an unverified negative is a guess).
- **rollback:** a pure refactor, one commit, `git revert`. Nothing is feature-gated because nothing changes behavior — which is exactly why N1 must be airtight before it lands.
- **blast:** the widest structural blast in the plan: `msl.rs`, `wgsl.rs`, `cuda.rs`, plus a new core module. **Zero behaviour change** by N1's construction, and zero `proxima-tensor` change. It is deliberately sequenced **after** every perf card so that no perf number is entangled with a 12,553-line reorganization.
- **observe:** `omega-gate.sh`'s `ran_count`; the golden-source byte comparison; the `(Backend, Route)` exhaustive match (a compile-time observable — the strongest kind).
- **memory gate:** compile-time only; runtime allocation identical by N1. Recorded as "unchanged, proven by byte-identical emission" rather than left blank.
- **reprove:** the three commands above; the row's claim is N1 plus the module list from N3.
- **log row:** `ROW ZZZ -- 78 near-duplicate functions across three emitters become one core and 21 dialect methods; every backend now answers for every kind, declines by name`

---

### R15 — the plan fingerprint: one bound plan, identical for every backend (item 1)

- **tier:** worker
- **depends_on:** R14 (there is one core to be identical *through*), R04-C1 (`prune_dead` changes the plan and must be inside the fingerprint's scope)
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r15 -b risc/plan-fingerprint 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r15
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r15`
- **opens:** `proxima-tensor/src/bind.rs:200-215` (`BoundOp {node, dtype, extents, kind}`), `:221-264` (`BoundOpKind`, 4 variants), `:95-98` (`Layout {base, strides}`), `:82` (`MAX_INLINE_RANK = 4`), `:377-446` (`BoundOp::split`, whose scatter arm is backend-conditional today), `:961-1010` (the `out_map` → `(out_layout, out_scatter)` construction), `:1011` `build_scatter_out_layout`, `:1594-1606` `layout_of` (**verified**: `base += i64::from(axis.offset) * stride` — the read-side offset already folds into `Layout.base`); `proxima-model-interop/src/generate.rs:855-861` `select_backend`; `omega/src/backend.rs:1-52` (six `Backend` variants, two implemented).
- **work:** `pub fn fingerprint(plan: &[BoundOp]) -> u64` in `proxima-tensor::bind` — FNV-1a-64 over the canonical serialization enumerated in the One-RISC binding section above. Then the test, in `omega/tests/`: bind the real openchat cached-forward program with real symbols and real blocks through the `Backend::Cpu` entry and the `Backend::Metal` entry (`backend::plan_named`, both compiled in — `omega/Cargo.toml`'s own comment records backends are **mix-and-match**, "both may be compiled in at once and the choice made at runtime"), and `assert_eq!` the fingerprints. Second case: same program, fingerprint under `metal-tiled-gemm` on and off — **equal**, proving the feature changes emission and not binding. Third case: the fingerprint is stable across two calls (determinism), the bind-side twin of `emit_is_deterministic_byte_equal`.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r15 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r15 cargo nextest run -p omega --features metal,cpu -E "test(plan_fingerprint)"
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r15 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r15 cargo nextest run -p omega --features metal,cpu,metal-tiled-gemm -E "test(plan_fingerprint)"
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r15 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r15 bash scripts/proxima-tensor-gate.sh && bash scripts/omega-gate.sh
  ```
- **expect:** **N == 3 fingerprint tests, all green; `ran_count == 0` is RED** (and this is the single most likely place for the N==0 trap, because the test is feature-gated on `metal` and `cpu` together). Each test prints the fingerprint value so a mismatch names both sides.
- **predict (one rung ahead: measured at nano — the fingerprint is a pure function over a bound plan — predicted at micro):** `fingerprint` over a 1196-op plan costs **< 100 µs**, i.e. below 0.01% of `prepare_ms` even at R13's 1.97 ms, so it can be called unconditionally in debug builds. If it costs more, it goes behind `debug_assertions` and the row says so.
- **kill:** the fingerprints differ. That is **the finding this card exists to produce** — a difference means the plan is not one plan, and the divergence is diffed op by op and becomes its own work item. A card that "fixes" the difference by loosening the fingerprint has inverted the test; the fingerprint's field list is fixed by this card's design and is not negotiable downstream.
- **rollback:** test-only plus one pure function. `git revert`.
- **blast:** `proxima-tensor/src/bind.rs` gains one function (no type, no field); `omega/tests/` gains one file. Zero hot-path change.
- **observe:** the printed fingerprint values themselves — a compile-and-run observable, not a counter, and stated as such.
- **memory gate:** the function allocates nothing (folds over borrowed slices into a `u64`). An allocation-counter assertion over 1000 fingerprint calls asserts zero (§11's mechanical form).
- **reprove:** the three commands above.
- **log row:** `ROW ZZZ -- one bound plan is now an assertion, not a claim: the same fingerprint through the CPU entry and the Metal entry, and invariant under metal-tiled-gemm`

---

# PHASE 5 — one sizing config owns every geometry constant

### R16 — every geometry constant into `omega-runtime.toml` (item 5)

- **tier:** hands
- **depends_on:** R14 (the constants live in the core after the refactor, so moving them once is cheaper and the diff is legible)
- **worktree:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima && git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r16 -b risc/geometry-config 4be2f3a
  mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r16
  ```
  `CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r16`
- **opens (each a verified §12 violation on main):** `omega/src/msl.rs:1017` `const PACKED_ROWS_PER_GROUP: usize = 4;`, `:1030` `const TILE_DIM: usize = 8;`, `:1046` `const TILED_GEMM_NSG: usize = 4;`, `:2516` `let lanes_per_block = 8;` (a **local**, even worse than a const — it cannot be overridden at all), and the packed-loop step at `:2527` `SIMD_WIDTH as usize / lanes_per_block`. Contrast the compliant surface: `omega/src/sized.rs` (`SIMD_WIDTH` documented as a **hardware fact, never a policy knob** — correctly *not* configurable) and `omega/omega-runtime.toml`'s `[tiled_gemm]` section with `min_tokens`/`block_m`/`block_n`/`block_k`, each with its cross-axis rule enforced in `omega/build.rs` (`require_nonzero :16`, `require_multiple_of_sixteen :35`, `require_divides_q4k_block :43`, `require_multiple_of_eight :59`, `resolve_int :79` with `rerun-if-env-changed :85`, `emit_sizing_consts :105`).
- **work:** new `[packed_row_block]` section (`rows_per_group = 4`, `lanes_per_block = 8`, `tile_dim = 8`) and `[tiled_gemm].nsg = 4` moved in; `[cooperative_reduce]` already landed by R06. Each key gets its cross-axis validator in `build.rs` following the four existing ones: `lanes_per_block` must divide `SIMD_WIDTH` (else the `ib += SIMD_WIDTH/lanes_per_block` step at `:2527` is wrong), `rows_per_group` must be nonzero (it is a `div_ceil` denominator in `grid_threads` `:1552`), `tile_dim` must be a multiple of 8. Each key's **measurement record lives on the consuming const's doc comment in `sized.rs`**, per that file's own stated convention, not in the TOML.
- **commands:**
  ```
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r16 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r16 grep -rn '^ *\(pub \)\?const [A-Z_]* *: *\(usize\|u64\|u32\) *= *[0-9]' omega/src/ | grep -v sized.rs
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r16 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r16 OMEGA_PACKED_ROW_BLOCK_ROWS_PER_GROUP=8 bash scripts/omega-gate.sh
  cd /Users/brianbruggeman/repos/slot-0/proxima-wt-r16 && CARGO_TARGET_DIR=/Users/brianbruggeman/repos/slot-0/.cargo-target/r16 OMEGA_PACKED_ROW_BLOCK_LANES_PER_BLOCK=7 cargo build -p omega --features metal 2>&1 | grep -q 'must divide' && echo VALIDATOR_OK
  ```
- **expect:** **N1:** the first grep returns **only** codec block constants (`Q4K_BLOCK_BYTES=144`, `Q4K_BLOCK_ELEMENTS=256`, `Q6K/Q5K/Q8_0/Q4_0/FLOAT16/BFLOAT16_*`) — those are **wire-format facts of the GGUF codec, not tunables**, and each gains a one-line doc saying so. Any *policy* const surviving the grep is RED. **N2:** the env override at `OMEGA_PACKED_ROW_BLOCK_ROWS_PER_GROUP=8` visibly changes the emitted MSL (compare golden sources) **and** the gate stays green — proving the `rerun-if-env-changed` line works and a cached build did not silently ignore it, which is the specific failure §12 clause 5 exists to prevent. **N3:** an invalid value fails at **build time** with the validator's message, not at runtime.
- **predict (one rung ahead: measured at nano — `q4k_matvec_probe` at `rows_per_group` 4 vs 8 — predicted at micro):** `rows_per_group = 8` changes `grid_threads`'s packed arm from `output_total.div_ceil(4)*32` to `div_ceil(8)*32`, **halving** the simdgroup count. R3/M5 measured rising marginal GB/s with simdgroup count (52 → 147 GB/s from 256 → 8001 simdgroups), so on `metal_vs_cpu.rs`'s `matvec_batch1_f32` arm the 8-row variant will be **slower**, by 10-30%. If it is *faster*, M5's mechanism is wrong and R18 must not be built.
- **kill:** if moving a const changes emitted MSL at the *default* value, the move introduced an off-by-one — the golden-source test (R14 N1) catches it and the card stops.
- **rollback:** `git revert`; the generated module is regenerated from the reverted TOML.
- **blast:** `omega/src/msl.rs` (5 sites), `omega/src/sized.rs` (+4 consts with their measurement docs), `omega/omega-runtime.toml` (+1 section, +1 key), `omega/build.rs` (+4 `resolve_int` + 2 validators). Zero behaviour change at default values, proven by the golden source.
- **observe:** the grep in N1 is the observable — a source-level assertion that no policy magic number survives, runnable in CI.
- **memory gate:** compile-time only; unchanged, and stated as such rather than blank.
- **reprove:** the three commands above.
- **log row:** `ROW ZZZ -- the last four geometry magic numbers leave the source: packed-row-block and tiled-GEMM geometry now trace to omega-runtime.toml with their cross-axis validators`

---

# PHASE 6 — geometry, only after the graph is minimal

### R17 — cooperative-reduce width re-tuned against the *final* graph

- **tier:** hands
- **depends_on:** R06 (the mechanism), R13 (the graph), R16 (the knob)
- **worktree:** `git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r17 -b risc/reduce-width-sweep 4be2f3a` + `mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r17`
- **opens:** `omega/src/msl.rs:1517-1560` `grid_threads`; `omega/src/sized.rs` `[cooperative_reduce]` (R06); `ggml-metal.m:3797-3804` per R8 (nth doubles 32 → `min(ne00/4, maxTotalThreads)`).
- **work:** a sweep, not a change — `OMEGA_COOPERATIVE_REDUCE_MAX_THREADS` ∈ {32, 64, 128, 256, 512, 1024}, six arms, interleaved, on the post-R13 graph. The reduction *lengths* changed when attention collapsed, so R06's tuning was against a graph that no longer exists.
- **commands:** the R06 milli command, once per value, each welded with `cd` and `flock`, interleaved round-robin rather than blocked by value.
- **expect:** 6 arms x 3 runs = **N == 18 rows; N == 0 is RED**. Monotonic-then-flat is expected; a non-monotonic curve is the finding and gets a row.
- **predict (one rung ahead: measured at milli → predicted at bench):** the optimum is `min(reduction_len/4, 1024)` per the incumbent's rule (R8), so 1024 for the 4096-wide rms_norm rows; `reduce-cooperative` lands at **≤ 60%** of its post-R13 value and `step_wall_ms` improves by **[1.5, 3.0] ms**.
- **kill:** if no value beats 32 by more than 2x CoV, the lever is dead on the new graph — record the negative with all six numbers and revert R06's feature to default-off permanently.
- **rollback:** it is an env-var sweep; the landed value is one integer in `omega-runtime.toml`.
- **blast:** one TOML integer.
- **observe:** `op_profile_bucket kind=reduce-cooperative gpu_ms` and `gpu_ns_per_op`.
- **memory gate:** threadgroup memory scales with width; `device_allocated_bytes` unchanged. Any device increase is a NEGATIVE.
- **reprove:** the sweep command with the landed value.
- **log row:** `ROW ZZZ -- the cooperative-reduce width, re-swept against the collapsed graph (the old tuning was for a graph that no longer exists)`

---

### R18 — split-K for the starving low-row shapes — **conditional; built only if the census still says so**

- **tier:** worker
- **depends_on:** R08 (the census), R13 (the graph), R17 (the width)
- **worktree:** `git worktree add /Users/brianbruggeman/repos/slot-0/proxima-wt-r18 -b risc/packed-split-k 4be2f3a` + `mkdir -p /Users/brianbruggeman/repos/slot-0/.cargo-target/r18`
- **opens:** `omega/src/msl.rs:1550-1552` (the packed arm's `div_ceil(PACKED_ROWS_PER_GROUP)*SIMD_WIDTH` simdgroup count); R13's family table (`attn_q` 5.172 ms at 9.45 MB = 58.5 GB/s, `attn_output` 78.6, `attn_v`/`attn_k` 50-53 GB/s — versus `ffn_*` at 97-109); M5 (rising marginal GB/s 52 → 147 from 256 → 8001 simdgroups).
- **entry gate — this card is NOT built unless it fires:** after R13 and R17, the route census must still show `attn_q`/`attn_k`/`attn_v`/`attn_output` achieving **under 70% of the `ffn_*` families' GB/s** on the same body. If R05's winning body already closed the gap, **this card is deleted and the row records why** — a lever that is no longer needed is a negative result worth writing down, not silent scope.
- **work:** split the reduction axis K into `split_k` partitions, each producing a partial, followed by a cheap combine. `split_k` traces to `[packed_row_block].split_k` in `omega-runtime.toml` (§12, via R16's section). This raises simdgroup count for the 1024-4096-row shapes without touching the ffn shapes.
- **commands:** the R06/R17 milli command with `--features …,metal-packed-split-k` and `OMEGA_PACKED_ROW_BLOCK_SPLIT_K` ∈ {1,2,4,8}.
- **expect:** 4 arms x 3 runs = **N == 12 rows; N == 0 is RED**. `attn_*` op counts must not change (a split that changes the op count changed the route). Parity vs `cpu::evaluate` on real `blk.0.attn_q.weight` at every split value — a split-K reduction changes summation order, so the tolerance is stated (1e-6) and the *token* must stay bit-identical.
- **predict (one rung ahead: measured at nano — `q4k_matvec_probe` at the four attn shapes — predicted at micro):** on `metal_vs_cpu.rs`'s `matvec_batch1_f32` Mistral arm, `split_k = 4` raises achieved GB/s at the 4096-row shape from ~58 toward the ffn families' ~97, i.e. **[85, 105] GB/s**, because M5 says the deficit is simdgroup starvation and 4x the partitions is 4x the simdgroups.
- **kill:** the combine step's cost exceeds the split's win (visible as `elementwise` op count rising and the net being flat) → dead lever, negative row, all four numbers recorded.
- **rollback:** default-off feature `metal-packed-split-k`, forwarded `metal-packed-split-k = ["omega?/metal-packed-split-k"]`.
- **blast:** `omega/src/msl.rs` packed-row-blocked body + `grid_threads` packed arm, behind a feature. No graph change, no driver change.
- **observe:** `op_profile_family` GB/s for the four `attn_*` families (true bytes, from R01).
- **memory gate:** partials are `split_k` extra intermediate buffers per matvec — through R10's arena, so the arena's peak grows by `split_k * attn_output_bytes`. Compute and print it before enabling; `DEVICE_CAP_BYTES` binds unchanged. An arena peak above the 40 MB activation term is a NEGATIVE and rolls back.
- **reprove:** the sweep command at the landed `split_k`.
- **log row:** `ROW ZZZ -- split-K for the starving low-row attention shapes (or: the census says the body already closed it, and here is the number)`

---

## Dependency graph

```
                        R00 (seal harness + measurer queue)
                         |
      +---------+--------+--------+---------+
      |         |                 |         |
     R01       R02               R03       R04-judge
  (bytes)  (incumbent arms)   (roofline)  (adjudicate CachedAttention)
      |         |                 |         |
      |         +--------+--------+         R04-C1 (prune_dead), R04-C2 (consumer index)
      |                  |                        |
      +---------> R05 (Q4_K body bake-off) <------+
                         |
                        R06 (wide cooperative reduce)
                         |
                        R07 (log rows + ai_docs JSONL)     [Phase 0 closes]
                         |
                        R08 (route enum + per-dispatch census, 5% budget)   [Phase 1]
                         |
                        R09 (bucketed cached_len + tail mask + INVERT bind.rs:3051-3058)  [Phase 2]
                         |
                        R10 (plan-stable arena)
                         |
             +-----------+-----------+
             |                       |
            R11 (MSL scatter)        |
             |                       |
             +-----------> R12 (device-resident KV, in-place write)          [Phase 3]
                                     |
                                    R13 (single-range attention, <=23 ops/layer)
                                     |
                        +------------+------------+
                        |                         |
                       R14 (emitter core)        R17 (reduce width re-sweep)   [Phase 6]
                        |                         |
             +----------+----------+             R18 (split-K, CONDITIONAL on the census)
             |                     |
            R15 (fingerprint)     R16 (geometry -> config)   [Phases 4, 5]
```

**Serialization constraint that cuts across the graph:** every card whose commands include `flock /tmp/proxima-gpu-measure.lock` is serialized against every other such card. R01, R02, R03, R05, R06, R08-R13, R17, R18 all measure on the GPU. Only R04's judge half, R07, R14's build half, R15, and R16 can run truly concurrently with a measurement. **R02 (the `-fa 1` arm) and R03 (the roofline) are both in Phase 0 and both precede every board-level prediction** — because R05 onward divide by 17.52 ms/token and by the measured ceiling, and a prediction against an unmeasured denominator is a DERIVED number carrying a mechanism claim (§18).

---

## Rollback map

| Card | Rollback | Blast on revert | Leaves behind |
|---|---|---|---|
| R00 | delete branch + worktree | none (new script) | nothing |
| R01 | `git revert` | counters return to R13's wrong values | nothing |
| R02 | delete branch; venv is untracked | Python flags default to today | the recorded arms |
| R03 | `git revert` | one example loses one arm | the ceiling number |
| R04 | per-commit `git revert` (three independent commits) | `prune_dead` / consumer index removed | the adjudication row |
| R05 | omit `--features metal-q4k-fold-scale` | Q4_K body returns to main's | the loser's negative row |
| R06 | omit `--features metal-wide-reduce` | reduce returns to 32-wide | `[cooperative_reduce]` section (inert) |
| R07 | `git revert` | docs only | nothing |
| R08 | `git revert` (behaviour-preserving, golden-proven) | gates return to hand-ordered `if let` | nothing |
| R09 | omit `--features kv-bucketed-cache` — **and the assertion inversion is `#[cfg]`-paired**, so the OFF build re-asserts `plan_hits == 0` | graph returns to `Symbolic(1) = cached_len` | `[kv_cache]` section (inert) |
| R10 | omit `--features metal-plan-stable-buffers` | per-op allocation returns | nothing |
| R11 | omit `--features metal-scatter` | `ScatterNotSupported` fires again | nothing |
| R12 | omit `metal-kv-resident` and/or `kv-scatter-write` (independent) | host `append` and/or per-token upload returns | nothing |
| R13 | omit `--features attention-single-range`; the two-range body stays under `#[cfg(not(...))]` | two-range graph returns | nothing |
| R14 | `git revert` (pure refactor, golden-proven) | three emitters diverge again | nothing |
| R15 | `git revert` | one function + one test file | nothing |
| R16 | `git revert` | consts return to source | nothing |
| R17 | change one TOML integer | width returns to prior | nothing |
| R18 | omit `--features metal-packed-split-k` | packed body unsplit | nothing |

**Every commit is a green bisect point** (AGENTS.md §pr-sequencing). The mechanism that makes this true for the behaviour-changing cards is that each ships its default-off feature **and** any assertion the feature invalidates is `#[cfg]`-paired, so both arms compile and both arms pass. R09 is the card where this matters most and is called out explicitly there.

---

## Abandoned designs (each with the constraint that ruled it out)

1. **`BoundOpKind::CachedAttention`, a fifth bound kind for one model's attention shape** (`perf/cached-attention-streaming`, `physical.rs` +576, `render_cached_attention` at their `msl.rs:104`). **Ruled out by** AGENTS.md §problem-solving — *"we should not be adding arbitrary rules/code for specific instances"* — against a closed 4-variant set (`bind.rs:221-264`), reinforced by its own measured null (R12: 51.535 ON vs 51.571 OFF, dispatches 1194 → 616, GPU *worse* at 39.841 vs 35.117) and by their own failure record admitting the matcher "cannot prove the semantic roles". **What it changed:** the plan attacks the *graph* (R09+R11+R12+R13) instead of pattern-matching the graph's defect after bind. This is the single largest way the constraints reshaped the design.

2. **A new `Op::Concat` / `Op::Pad` / `Op::Tile` variant to splice the new token's K/V onto the cache.** **Ruled out by** guiding-principles §1's binary question — *write the expression* — which succeeds: `IndexMap::scatter` (`map.rs:175`) plus `BoundOpKind::Reduce.out_scatter` (`bind.rs:253`) plus `run_reduce_scatter` (`cpu.rs:6911`) already express and *execute* write placement, today, on CPU, with the worked example in the function's own doc. R5's "zero hits for `Concat` on main" is a fact about main, not a licence. **What it changed:** R11 became an *emitter* card (Metal implements what the IR already models) rather than an *IR* card, which is why `proxima-tensor` is untouched by R11 and why the blast radius of write placement is one file.

3. **A `PlacedBuffer` type or a `write_placement` field on `Reduce`.** **Ruled out by** the relocation question plus a closed prior adjudication: `map.rs:109-131` records that a `Reduce`-wide destination-extent field was **already rejected on blast-radius grounds** and the scatter `offset`-at-`gathered_dim` convention chosen instead. Adding the field now re-litigates that decision, and `out_layout.base` (`bind.rs:95-98`) already carries a destination offset on every bound reduce. **What it changed:** the only new state in the entire placement path is three `long`s on a uniform struct that already carries `out_base` — and even that had its call site written both ways (R11), showing the alternative costs a bind slot on the 100% path.

4. **Making the reduce extent over the cache a runtime uniform** (M6' fix (b), the "bigger change" the ledger names). **Ruled out by** §12's interaction clause: `BoundOp.extents` is a baked `Vec<u64>` (`bind.rs:200-215`) and `grid_threads` (`msl.rs:1517-1560`) computes the dispatch grid **from** those extents, so a runtime extent makes the dispatch geometry runtime too — pushing a value into the exact place the optimiser and the dispatcher both need a constant. **What it changed:** R09 took fix (a), bucket-and-mask, which is the incumbent's own mechanism (R8: pad `n_kv` to a 256 multiple and mask) and needs **zero IR change** — four graph nodes built from `causal_mask`'s existing shape.

5. **Threading `op_setup` across cores** (M9, the landed `PROXIMA_ORCH_THREADS` knob on `perf/decode-orchestration-2`, timing unmeasured). **Ruled out by** the type system and by R4: `MTLBuffer` is non-`Send`, so the buffer half cannot be threaded at all, and R4 records that thread count explains **zero** of the gap because the incumbent is GPU-bound. §21's framing is the tell — *a lock is usually a missing owner*, and here the missing owner is a plan that should own its buffers. **What it changed:** R10 removes the 3.9 ms of `op_setup` by **not doing the work** (allocate once per plan) rather than by doing it faster on more threads.

6. **nsg=2 threadgroup regrouping to match ggml's packed simdgroup geometry.** **Ruled out by** four independent measured negatives across two lanes: R4 (-2.36% on the real graph, "third time tried"), `perf/metal-simdgroup-geometry`'s 2 commits, and the parallel branch's ROW 267 ("ggml's two-SIMD-group Q4_K geometry regresses the cached-feature wall cell") after it had already been tried and rejected on that same branch. **What it changed:** R18 raises simdgroup count by **split-K** (partitioning the reduction, which M5's 52 → 147 GB/s curve supports) rather than by regrouping threads, and R18 is gated on the census still showing starvation at all.

7. **A second marker string to fix the classifier mislabel** (their ROW 263's fix). **Ruled out by** §"Find where information is destroyed": another substring is more of M10, the mechanism that caused the mislabel. **What it changed:** R08 makes the route a value and *deletes* the substring buckets, and their ROW 263 becomes the third independent witness on R08's row rather than a patch we inherit.

---

## Open questions resolved by measurement (not by argument)

| Question | The card that answers it | The measurement that decides |
|---|---|---|
| Does the `-fa 1` incumbent arm move the denominator every ratio in this plan divides by? | **R02** | tg32 with `-fa 1` vs without, 3 runs interleaved. >5% movement re-derives every board prediction. |
| What is this GPU's actual streaming ceiling, and in which denominator? | **R03** | two-size marginal, GPU-timestamped, readback outside the window, `read_only_gbs` and `traffic_gbs` both printed. Ends `rooflines.md:411`'s DEBT. |
| Which Q4_K body — and is the -36% (Arm A) and -29% (Arm B) the *same* win double-counted? | **R05** | interlea