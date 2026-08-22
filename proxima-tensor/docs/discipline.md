# proxima-tensor executor perf gap — discipline log

Repo: /Users/brianbruggeman/repos/slot-0/proxima-fuse-wt, branch feat/tensor-fuse, HEAD c891caeb.
CARGO_TARGET_DIR pinned to scratchpad/opt-target for every command below (never `export`).
Host: Apple M1 Max, macOS 15.7.8, arm64.

The 13-point gate (see SKILL): build / tests / clippy / micro-bench /
compare-bench / E2E / opt-sweep / SIMD-SM-noBox / O(1) / Cfg-API /
home-turf / delta. This log follows the task's Phase1-instrument /
Phase2-one-tweak-at-a-time structure layered on the same discipline
(row = one tweak, before/after, CoV, kept/rolled back).

## C1 — cpu.rs executor (run_reduce / run_elementwise / apply_body)

**Incumbent:** ggml (commit 2d191b5d), Accelerate-backed, on this same M1 Max.
**Bench arms:** `proxima-tensor/benches/bench_vs_ggml.rs` row F (bare f32 GEMM,
512/1024/2048) and row C (gather -> scale -> reduce, table[50000x512], 4096
indices) — copied from the sibling proxima-wire-wt worktree (not modified
there), registered behind the `ggml-bench` feature added to this worktree's
`proxima-tensor/Cargo.toml`.

## ROW 0 — baseline instrumentation (before any code change)

Harness: `proxima-tensor/examples/profile_hot.rs` (throwaway, not part of the
crate's public surface, deleted at the end of this task) — runs `evaluate()`
directly, no criterion overhead, so profiler symbolication maps 1:1 to
`cpu.rs` functions.

### 1. Baseline timing (direct `Instant`, release build, 5 runs averaged)

- GEMM 1024x1024x1024 (`Reduce(Elementwise(Multiply))`, fused, no
  materialized product tensor): **8.3s - 10.7s per call** (two independent
  runs, `evaluate()` single-threaded). ~1.073e9 total (m,n,k) inner steps
  -> **~8-10 ns per scalar multiply-accumulate**. On a 3.2GHz core that is
  ~26-32 cycles for what should be a 1-cycle-throughput FMA.
- gather -> scale -> reduce (table [50000x512], 4096 indices), 200 repeats:
  measured separately below (ROW 0 section 5).

### 2. Does the innermost loop vectorize? NO — confirmed by disassembly, not inference.

`objdump -d` on the release `profile_hot` binary
(`scratchpad/opt/full_disasm.txt`), symbol
`_RNvNtCskHGgWzEiGui_14proxima_tensor3cpu10apply_body` (cpu.rs:941): every
arithmetic instruction in the op-dispatch jump table is **scalar single-lane**
— `fmul s0, s0, s1`, `fsub s0, s0, s1`, `fdiv s0, s0, s1`, `fsqrt s0, s0`,
`fneg s0, s0`, `fcmp s0, #0.0`. Zero NEON vector opcodes (`.4s`/`.2s`, e.g.
`fmla.4s`/`fmul.4s`) anywhere in the function. Dispatch is a computed jump
table (`adr x9` + `ldrb`+`add`+`br x9`) selecting among the `ScalarOp` arms,
executed **once per element** — not hoisted per node.

`apply_body` is a real out-of-line subroutine (confirmed by its own
prologue: `stp d9,d8,[sp,#-0x70]!` + 5 more `stp` pairs = 10 GPRs + 2 FP regs
saved/restored on every call), called via `bl` from the reduce/elementwise
loops for every one of the ~1.07e9 GEMM inner steps. It is not inlined
despite `lto = "fat"` + `codegen-units = 1` — LLVM's inliner does not inline
a function whose body contains an unbounded runtime loop
(`body.steps.iter()`), since inlining a loop doesn't shrink the call-site
cost the way inlining a straight-line body does.

### 3. Is `-C target-cpu=native` in effect? NO, and on this target it would not matter.

`Cargo.toml` `[profile.release]` (repo root, line 222): `lto = "fat"`,
`codegen-units = 1`, `panic = "abort"`, `opt-level = 3` — no
`target-cpu`/`target-feature` anywhere, no `.cargo/config.toml`
`[build] rustflags` for this target either (checked
`/Users/brianbruggeman/repos/slot-0/proxima-fuse-wt/.cargo/config.toml` —
only a wasm32-scoped `getrandom_backend` flag).

`rustc --print cfg` (default `aarch64-apple-darwin` host target, no flags)
vs `rustc -C target-cpu=native --print cfg` (native = `apple-m1` on this
host) report **the identical `target_feature` set**: `neon`, `dotprod`,
`fp16`, `fcma`, `lse`, `rdm`, `sha3`, etc. all present in BOTH. The
`aarch64-apple-darwin` target already bakes in the Apple M-series feature
baseline; `target-cpu=native` adds nothing measurable here. This is a
**refuted hypothesis, not an unmeasured one** — checked before being written
into the sweep.

### 4. Cycle attribution — `sample` (macOS built-in profiler), 15s window on the live GEMM-1024 run.

`scratchpad/opt/gemm_sample2.txt`, "Sort by top of stack" (leaf self-time,
12,589 total samples at 1ms/sample):

| symbol | self-time samples | % |
|---|---|---|
| `apply_body` (cpu.rs:941) | 6536 | 51.9% |
| `evaluate`-inlined body (cpu.rs `run_node`/`run_node_into`/`run_reduce` all inlined into the `pub fn evaluate` symbol) | 5847 | 46.4% |
| malloc-family (`_nanov2_free`+`nanov2_calloc_type`+`_free`+`nanov2_malloc_type`+`_malloc_zone_calloc`+`_malloc_zone_malloc`) | 169 | 1.3% |
| `build_gather_cursors` + its iterator `.next()` | 32 | 0.25% |
| `merge_coordinates` | 5 | 0.04% |

**This refutes the code-reading hypothesis formed before profiling.**
Reading `run_reduce` first suggested the `Vec::collect()` calls building
`running`/`strides`/`gather_cursors`/`full_coordinate` fresh on every
reduction step (`cpu.rs:823-834`) — 3-4 heap allocations per (m,k) pair,
~4.2M total for the 1024 GEMM — would dominate. Measured: **malloc-family
leaf time is 1.3% of the run.** The dominant cost (98.3%) is scalar
per-element dispatch and bookkeeping, not allocation. The call-graph view in
the same `sample` output confirms the malloc calls are a real but small
branch under the `evaluate` symbol (`+7420`/`+8020`/`+9324`/`+9344`/`+9352`/
`+9368`/`+9384` offsets each carrying 1-30 samples), not the trunk.

### 5. Per-element overhead count (from source + disassembly, cross-checked against the profile)

Per **inner GEMM step** (`run_reduce`'s `for slot in &mut accumulator` body,
cpu.rs:836-851, executed M*N*K = ~1.073e9 times for the 1024 case):
- 2x bounds-checked slice index (`data[offset as usize]`, one per operand) —
  visible in `apply_body`'s disasm as `cmp`/`b.ls` pairs guarding every
  `ldr s0, [x9, x0, lsl #2]`.
- 1x `Option::as_mut()` check per operand for `gather_cursors[index]` (always
  `None` for GEMM — dead weight paid every element regardless).
- 1x out-of-line call (`bl`) into `apply_body`, with its 10-GPR+2-FP-reg
  save/restore prologue/epilogue, for a body that is a **single** `ScalarOp`
  step (`Multiply`) in this program.
- 1x computed-jump-table dispatch inside `apply_body` to pick the `Multiply`
  arm out of 16 possible `ScalarOp` variants — re-decided every element even
  though the op is loop-invariant for the whole node.
- 1x `apply_scalar_op` call (may or may not be inlined into `apply_body`;
  not separately visible in `sample`'s symbol table, consistent with it
  being inlined — the jump table lives directly in `apply_body`'s disasm).

Per **reduction step** (once per (m,k) pair, K=1024 times per row, M=1024
rows -> ~1.05M times for the 1024 GEMM, cpu.rs:816-834): 4 short `Vec`
allocations (`full_coordinate`, `running`, `strides`, `gather_cursors`) via
`.collect()`/`vec![]` macro. Real, but only 1.3% of measured self-time (see
#4) — noted as a **secondary** target, not the primary one.

### Baseline vs ggml (context from the pre-fusion bench cited in the task; not
re-run here to conserve budget — the GEMM row's absolute proxima number
above, ~8-10s for 1024^3, is the load-bearing baseline this task closes
against):
- bare f32 GEMM 1024: ggml was reported 63-199x faster pre-fusion.
- gather->scale->reduce: proxima wrote 1,025x fewer bytes, ran 10-22x
  slower pre-fusion.

**Implication for Phase 2 ordering:** the sweep item "lift the ScalarOp/body
dispatch out of the inner loop — specialize the loop per body shape" is the
justified first move (targets the measured 51.9%+46.4% = 98.3% of
self-time: scalar dispatch + call overhead + bounds-checked indexing).
"Blocking/tiling for cache" and "std::simd/NEON intrinsics" are **not**
justified yet — the loop isn't even scalar-optimal, let alone
memory-bound; profiling after the dispatch fix decides whether either is
still worth trying. `target-cpu=native` is refuted (see #3). The
Vec-allocation-per-reduction-step pattern (#5, second table) is real
representation debt worth fixing but is not expected to move the top-line
number given its 1.3% measured share — it will get a row, but after the
dispatch fix, and its delta will be judged against noise floor honestly.

## Phase 2 — one tweak at a time

### ROW 1 — hoist ScalarOp/body dispatch out of the per-element loop

**Change** (`proxima-tensor/src/cpu.rs`): added `BodyShape` (`Unary`/`Binary`/
`Generic`) classified once per node via `body_shape(body)`, computed
alongside the existing `let body = resolved.element_body();` hoist point in
`run_elementwise`, `run_reduce`, and `run_scan` (all three already computed
`body` once outside their loop nests — the classification piggybacks on
that, no new hoist point needed). Added `#[inline(always)] fn
eval_body_shape` which matches on the pre-classified shape and, for the
`Unary`/`Binary` arms, calls `apply_scalar_op` directly with a 1- or
2-element array — skipping `apply_body`'s per-element step loop, its
dynamic `StepArg` resolution, and the out-of-line `bl` call entirely for
the (overwhelmingly common, post-fusion) single-step case. `Generic` falls
through to the original `apply_body` unchanged — no behavior change for
real multi-step fused chains. Also added `#[inline(always)]` to
`apply_scalar_op` (was already effectively inlined per the ROW 0
disassembly — not itself a distinct measured lever, kept for
correctness-neutral cleanliness). All 3 call sites
(`cpu.rs:770`/`845`(now via shape)/`920`) now call `eval_body_shape(&shape,
...)` instead of `apply_body(body, ...)`.

**Why this order:** ROW 0 measured 51.9% + 46.4% = 98.3% of GEMM self-time
in `apply_body` + the surrounding scalar/dispatch bookkeeping, and 1.3% in
allocation. This tweak targets the 98.3%, not the 1.3% — the allocation
fix (Vec::collect() per reduction step) is deliberately NOT bundled in here
(one tweak at a time, per the skill) and is documented as a candidate for a
future row rather than attempted this session (see "Not attempted" below).

**Bench (row F, GEMM 1024x1024x1024, `examples/profile_hot.rs` — direct
`Instant`, no criterion overhead, so the number is 1:1 comparable to ROW
0's baseline measured the same way):**

| | before | after |
|---|---|---|
| run 1 | 8.339s | 5.022s |
| run 2 | 10.724s | 5.017s |

CoV after: **<0.1%** (5.022s / 5.017s, 2 runs) — a genuinely tight signal,
not noise. CoV before was itself high (8.3s vs 10.7s, ~13% swing) — host
loadout was not quiet for the ROW 0 baseline captures; the "after" numbers
happened to land in a quieter window. Taking the more conservative
(smaller) before number: **8.34s -> 5.02s = 1.66x**. Taking the larger:
**10.72s -> 5.02s = 2.14x**. Reporting the range, not a single ratio, per
the CoV discipline.

Host loadout: quiet except for this run itself (no other cargo/criterion
processes active); `sample`'s own 1ms-interval sampling ran concurrently
with the ROW 0 captures only, not these timing-only runs.

**Kept.** Correctness confirmed on the exact post-tweak binary:
`cargo nextest run -p proxima-tensor` → **128/128** (matches pre-tweak
baseline exactly), `--features config` → **144/144**, `cargo test -p
proxima-tensor --doc` → **1/1**, `cargo nextest run -p omega --features
metal` (real Metal device) → **25/25**. Numerical GEMM output is byte-for-
byte reused by the same `evaluate()` call the pre-existing
`fused_matmul_matches_a_naive_triple_loop` / `matmul_binds_a_symbolic_
sequence_length_at_eval_time` / `fused_contraction_skips_the_product_
tensor` tests already assert against a hand-written triple loop — those
are 3 of the 128 and passed.

**Bench (row C, gather -> scale -> reduce, table[50000x512], 4096 indices):**

| | before | after |
|---|---|---|
| run 1 | 27.42ms | 27.11ms |
| run 2 | 46.20ms | 22.70ms |
| run 3 | — | 27.57ms |

This row's body IS also a single-step `Multiply` (qualifies for the new
`Binary` fast path), but the delta is **noisy and much smaller** than
GEMM's: before ranges 27.4-46.2ms (CoV ~30%+ across just 2 runs — host was
not quiet for this shorter, ~5s-total workload), after ranges 22.7-27.6ms
(CoV ~9%). Point estimates overlap between before and after. **Honest
read: directionally positive (means ~36.8ms before vs ~25.8ms after, a
~1.4x if the means are trusted) but NOT a clean signal at this CoV — this
row does not meet the "trust a single delta" bar the skill sets.** The
mechanism explains why the gap between rows is real: GEMM's inner loop
runs the classified-body path 1.07e9 times (dispatch overhead dominates,
matches ROW 0's profile); row C's inner loop runs it only ~2.1M times
(seq=4096 x dim=512) and is additionally paying for a genuinely
random-access gather fetch (`table[ids[i], :]` into a 50000-row table —
cache-hostile by construction), which ROW 0 did not separately profile.
Dispatch-hoisting cannot fix a latency-bound random-access read; that is a
**different, unaddressed** bottleneck for row C, named here rather than
implied away.

**Implication:** the dispatch-hoist fix is a real, clean win on the
dispatch-bound row (GEMM) and an unproven-but-plausible smaller win on the
gather row, which is bound by something else this session did not
instrument (memory latency on the gather fetch, not scalar dispatch). A
correct next ROW 0-style profiling pass on row C specifically (not
reused from the GEMM profile) would be required before spending more time
optimizing row C's dispatch path further — it is very likely dispatch is
already not row C's bottleneck.

### Not attempted this session (named, not silently dropped)

- **Vec-allocation-per-reduction-step** (`running`/`strides`/`gather_cursors`/
  `full_coordinate` rebuilt via `.collect()` every `(m,k)` pair in
  `run_reduce`, cpu.rs ~816-834): real, measured at 1.3% of GEMM self-time
  in ROW 0 — a legitimate representation-shortcut regression per the rules
  (heap alloc instead of stack/reused buffer) but not the load-bearing
  cost. Candidate for a future row; not attempted here to stay inside the
  90-minute budget after ROW 1's build+bench+gate cycles.
- **Blocking/tiling for cache locality** in `run_reduce`: not attempted.
  ROW 0's profile showed no evidence of memory-bandwidth-bound behavior
  (the dominant costs were dispatch and bookkeeping, not cache misses) —
  premature without re-profiling after ROW 1 to see whether a new
  bottleneck emerged.
- **`std::simd` / explicit NEON intrinsics**: not attempted. ROW 0
  established the loop wasn't even scalar-optimal (dispatch overhead
  dominated); per the task's own instruction, SIMD is only justified "if
  autovectorization provably fails after the loop is shaped correctly" —
  the loop was not yet shaped correctly (fixed in ROW 1) and this session's
  budget ran out before a re-profile could show whether the post-ROW-1
  scalar loop is now vectorizable or still dispatch/bookkeeping-bound in a
  way that blocks autovectorization (the `Vec<&[f32]>` operand indirection
  and gather-cursor `Option` check per operand, both still present, are
  likely autovectorization blockers even post-ROW-1 — named, not fixed).
- **`-C target-cpu=native`**: investigated and REFUTED in ROW 0 (#3) — the
  `aarch64-apple-darwin` target already ships the full Apple M-series
  feature baseline (neon/dotprod/fp16/lse/...) by default; native adds
  nothing measurable on this host/target pair. Not re-tried.
- **Re-profiling with `sample`/`objdump` after ROW 1**: not done — budget
  spent on getting ROW 1 measured, gated, and logged honestly instead. The
  post-ROW-1 profile (does GEMM's inner loop now vectorize? is the
  jump-table gone from the hot path? what's the new #1 symbol?) is the
  natural next ROW 0-shaped step for a follow-on session.

## Final gap vs ggml (context, not re-measured this session)

The ggml comparison numbers cited in the task (bare f32 GEMM 63-199x,
GEMV 650-1100x, gather-reduce 10-22x-slower-while-1025x-fewer-bytes) are
from a PRE-fusion baseline per the task's own framing. This session did
not re-run `bench_vs_ggml.rs`'s ggml arms (`row_f_ggml_gemm_1024`,
`row_c_ggml_gather_scale_sumrows`) to get a same-host, same-commit,
apples-to-apples ratio against the ROW 1 numbers above — the bench file is
copied in and building/linking against the prebuilt ggml at
`scratchpad/ggml` (commit 2d191b5d) succeeded (`cargo build --release -p
proxima-tensor --bench bench_vs_ggml --features ggml-bench` → EXIT=0), but
running the criterion harness end-to-end (`sample_size=10`,
`measurement_time=500ms`, and GEMM-1024 alone costing ~5s/sample at
post-ROW-1 speed = ~50s minimum for just that one arm, before rows A/B/D/E/
G/H's own setup cost) did not fit in the remaining budget after ROW 0's
instrumentation and ROW 1's gate cycles. **This is a real gap in this
session's deliverable, named rather than glossed: "1.66-2.14x faster than
pre-tweak" is measured; "Nx closer to ggml" is not, this session.**


### ROW 2 — hoist every loop-invariant allocation

**Change 1** (`proxima-tensor/src/cpu.rs`, `run_elementwise`/`run_reduce`/`run_scan`):
`strides` — depends only on the operand `Layout`s and a fixed dim, never on
the current coordinate — moved from inside the per-position loop to once
per bound-op call (was rebuilt via `.collect()` every outer position in
elementwise/scan, and every `(leading, reduction)` coordinate pair in
reduce). `running` and `gather_cursors` are now allocated once (`vec![0;
raw.len()]` / `(0..raw.len()).map(|_| None).collect()`) before the loop
nest and refilled in place per position via two new helpers,
`fill_running_offsets` and `fill_gather_cursors` (replaces the old
`Vec`-returning `build_gather_cursors`), both writing into the caller's
buffer instead of collecting a new one.

**Change 2** (found via measurement, not in the task's original list —
named and fixed under the same row rather than silently folded in):
`odometer`/`unflatten` returned `impl Iterator<Item = Vec<u64>>`, heap-
allocating one `Vec<u64>` per coordinate on every `.next()`, and
`merge_coordinates` allocated a fresh `Vec<u64>` on every call. Both are on
`run_reduce`'s inner (reduction-coordinate) loop — up to ~1.05M calls each
for the 1024^3 GEMM — and were still firing after Change 1's hoists, which
only addressed `strides`/`running`/`gather_cursors`. Replaced with
`odometer_len` (a plain product, no allocation) plus `unflatten_into` and
`merge_coordinates_into`, which write into caller-owned buffers
(`leading_coordinate`/`reduction_coordinate`/`full_coordinate`/
`outer_coordinate`, one `Vec<u64>` each, allocated once per bound-op call
and reused for every position) instead of returning owned `Vec`s. All 6
call sites (2x `odometer` in `run_elementwise`+`run_scan`'s outer loop, 2x
in `run_reduce`'s leading+reduction loops, 2x `merge_coordinates` in
`run_reduce`) converted to the `_into` form.

**Target:** allocations per bound-op call O(operand count + rank), not
O(output elements) or O(reduction steps). Confirmed below.

**Measurement method:** `proxima-tensor/examples/profile_hot.rs` (already
present from ROW 0/1) modified to install a `#[global_allocator]`
`CountingAllocator` wrapping `std::alloc::System`, incrementing a static
`AtomicU64` on every `alloc()` call, snapshotted immediately before/after
the single `evaluate()` call for the 1024x1024x1024 GEMM. This counts every
allocation the process makes during that call, not just ones the crate
author remembered to instrument — MEASURED, not derived from reading the
loop nest. To get a same-binary "before" number, `git stash push --keep-
index -- proxima-tensor/src/cpu.rs` reverted cpu.rs to the ROW 1 (pre-ROW-2)
state, rebuilt, measured, then `git stash pop` restored ROW 2's changes.

**Allocation count (MEASURED, GEMM 1024x1024x1024, identical binary+harness
modulo the `git stash` swap of cpu.rs only):**

| | before (ROW 1 state) | after (ROW 2, both changes) |
|---|---|---|
| allocations during `evaluate()` | 5,246,029 (3 runs, identical) | 1,107 (5 runs, identical) |

4,738x fewer allocations. `root[0]=18370`, `root_len=1,048,576` identical
before/after — same numerical output. (Intermediate point, not separately
tabled: after Change 1 alone, before Change 2, allocations were 2,100,304 —
Change 2's `odometer`/`merge_coordinates` fix accounted for the remaining
~2.1M of the ~5.2M baseline, roughly matching 2 allocations x ~1.05M
reduction steps, DERIVED from that arithmetic, not separately re-measured
after reverting only Change 2.)

**Timing (direct `Instant`, `examples/profile_hot.rs`, release build, 1024^3
GEMM, 5 runs each unless noted):**

| | ROW 1 (before ROW 2) | ROW 2 (after both changes) |
|---|---|---|
| runs | 4.709s, 4.732s, 4.741s (3 runs) | 4.626s, 4.656s, 4.659s, 4.663s, 4.669s |
| range | 4.709s - 4.741s | 4.626s - 4.669s |
| CoV | ~0.3% | ~0.4% |

Both tight (low CoV, quiet host). Point-estimate delta: ~4.73s -> ~4.65s,
roughly **1.02x** — a small, real-but-marginal timing win, consistent with
ROW 0's profile finding that malloc-family leaf time was only 1.3% of
self-time before ROW 1's dispatch fix (and, per this task's own framing,
that percentage under-charges allocation's true cost — cache/TLB damage
attributed to whatever runs next — so the 4,738x allocation-count drop is
the load-bearing number for this row, not the ~1.02x wall-clock delta).
Note this ROW 2 "before" (4.71-4.74s) sits noticeably below ROW 1's own
row-F numbers logged in ROW 1 (5.017s/5.022s) despite being the same
cpu.rs state — different host loadout between sessions, not a code
difference (confirmed identical via `git stash`, same commit content).

**Disassembly:** not re-pulled this row — ROW 2 does not touch the
innermost per-element arithmetic (`eval_body_shape`/`apply_scalar_op`
unchanged), only the per-position bookkeeping around it, so ROW 1's
disassembly finding (scalar `fmul s0`/`fmadd s0`, no NEON) is expected to
still hold and is re-checked in ROW 3, which does touch the innermost loop
body.

**KEPT.** Correctness: `cargo nextest run -p proxima-tensor` → **128/128**
(post-ROW-2 binary, matches ROW 1's 128/128 exactly, includes the
hand-written triple-loop GEMM parity tests). Remaining gates (`--features
config`, `cargo nextest run -p omega --features metal`, `cargo test -p
proxima-tensor --doc`) run once at the end of ROW 3 rather than duplicated
after every row, per budget — flagged here so it isn't silently skipped.

### ROW 3 — kill the per-element copy and the per-element Option (run_reduce only)

**Scope note:** implemented for `run_reduce`'s width loop only — the exact
block the task's defect diagnosis names (`for slot in &mut accumulator {
for (index, data) in raw.iter().enumerate() { ... } }`, cpu.rs, formerly
~838-853, the measured 1.073e9-iteration hot loop for the 1024^3 GEMM).
`run_elementwise` and `run_scan` have the identical per-element shape but
were **not** touched this row — named here rather than silently folded in,
budget ran out after getting `run_reduce`'s change measured, disassembled,
and gated.

**Change:** added `body_shape_is_affine_fast_path` (checked ONCE per bound
op, right after `strides` is computed, never per element or per position):
true when every physical operand the pre-classified `BodyShape`
(`Unary`/`Binary`, from ROW 1) actually reads is gather-free AND has a
width-dim stride of 0 (broadcast) or 1 (contiguous) — the two cases the
task named, nothing else. When true, the width loop calls
`reduce_width_fast`, which dispatches to `reduce_width_unary` or
`reduce_width_binary`: straight-line code operating on real `&[f32]`
subslices (`&data[base..base+width]`) via `.iter_mut().zip()`, or a single
hoisted scalar read for a stride-0 operand — no `operand_values` scratch
copy, no `gather_cursors[index].as_mut()` `Option` check, no
`running[index] += strides[index]` per-element increment (unnecessary once
the whole width span is addressed as one contiguous read). `fill_gather_cursors`
is skipped entirely on the fast path (nothing to build — eligibility already
proved no operand gathers). Any operand with a non-0/1 stride, or a
`Generic` (multi-step fused) body, falls through unchanged to the original
per-element loop with its `Option` check and `operand_values` copy intact —
named explicitly, not silently narrowed.

**Runtime confirmation the fast path is actually taken for GEMM (not just
eligible in theory):** temporarily added `eprintln!("fast_path={fast_path}
raw.len()={} strides={strides:?}", raw.len())` immediately after computing
`fast_path`, rebuilt, ran the GEMM harness once, captured the output, then
reverted the line and rebuilt clean before any gate/bench run counted below.
Output: `ROW3-VERIFY fast_path=true raw.len()=2 strides=[0, 1]` — `lhs`
(stride 0, broadcast across `n`) hits `reduce_width_binary`'s scalar-hoist
arm, `rhs` (stride 1, contiguous across `n`) hits its slice arm. This is a
MEASURED runtime fact, not an assumption from reading the eligibility
predicate.

**Timing (direct `Instant`, `examples/profile_hot.rs`, release build, 1024^3
GEMM, 5 runs):**

| | ROW 2 (before ROW 3) | ROW 3 (after) |
|---|---|---|
| runs | 4.626s, 4.656s, 4.659s, 4.663s, 4.669s | 1.441s, 1.446s, 1.446s, 1.447s, 1.449s |
| range | 4.626s - 4.669s | 1.441s - 1.449s |
| CoV | ~0.4% | ~0.2% |

Both tight, quiet host. **3.20x - 3.24x faster** (4.626/1.449 = 3.19x low
end, 4.669/1.441 = 3.24x high end). ns-per-MAC at the end of ROW 3: 1.073e9
MACs / 1.441-1.449s = **1.34 - 1.35 ns/MAC** (down from ROW 0's baseline
~8-10 ns/MAC — roughly 6-7x across all three rows combined).

**Allocation count:** unchanged at **1,107 - 1,108** across runs (ROW 3
doesn't touch allocation — it removes per-element work inside a loop nest
ROW 2 already made allocation-free). Consistent, confirms ROW 3 is
orthogonal to ROW 2's fix.

**Correctness:** `cargo nextest run -p proxima-tensor` → **128/128**
(includes the hand-written triple-loop GEMM parity tests — `root[0]=18370`,
`root_len=1048576`, byte-identical to ROW 0/1/2's output on every run).
`--features config` → **144/144**. `cargo nextest run -p omega --features
metal` (real Metal device) → **25/25**. `cargo test -p proxima-tensor
--doc` → **1/1**. All four run again, clean, after reverting the temporary
verification `eprintln!` — the numbers above are the post-revert binary's.

**Disassembly — do NEON opcodes appear?** NO, checked binary-wide, not just
near a guessed hot address. `objdump -d` on the release `profile_hot`
binary (`scratchpad/opt/row3_full_disasm.txt`, 74,546 lines): `grep -c
'fmla'` → **0** anywhere in the whole binary. `grep -c 'fmul\.4s\|fmul\.2s'`
→ **0**. Every occurrence of a `.4s`/`.2s`/`.2d`-suffixed instruction in the
entire binary (4 total) is a `movi.2d vN, #0` zero-init or one `dup.4s`
broadcast setup, not a float multiply/add. `grep -cE 'fmul\s+s[0-9]+,
s[0-9]+, s[0-9]+'` → 16 scalar single-lane multiplies exist in the binary
(across all compiled op arms, not just the GEMM path) — e.g.
`10000f8f4: fmul s8, s8, s0`. **Still scalar, confirmed absent NEON, not
inferred.**

**Mechanism — this is NOT a vectorization win.** `sample` (macOS profiler,
1ms interval, 3s window covering the full ~1.4s run,
`scratchpad/opt/row3_sample.txt`) shows 834/839 total samples (99.4%) with
top-of-stack in `run_node_into` (the function `run_reduce` — a private,
non-`#[inline(never)]` function — got fully inlined into, same as ROW 0's
finding for the pre-fusion build). `run_node_into` is one large function
LTO produced by inlining `run_reduce`/`run_elementwise`/`run_scan` and
their generic (`apply_body`) fallback arms together for every possible
`ScalarOp`/`BodyShape` combination the crate supports, not just GEMM's —
manually isolating the exact basic block `reduce_width_binary`'s
Scalar+Contiguous arm compiles to, inside that one large function, was not
reliably achievable by address arithmetic in the remaining budget (tried;
the addresses `sample` attributed samples to sit adjacent to unrelated
compiled-but-cold arms for `Reciprocal`/`SquareRoot`/`Divide`/`Logarithm`,
part of the same giant function, and distinguishing "hot but unlabeled" from
"adjacent cold code sample landed near" from raw addresses alone was not
achieved this row — named as a residual, not glossed over). What IS
established: (1) the runtime verification above proves
`reduce_width_binary`'s fast arm executes for GEMM; (2) the binary contains
zero NEON float instructions anywhere; (3) the measured 3.2x speedup is
therefore a **non-SIMD mechanism** — consistent with eliminating, on every
one of the 1.073e9 element visits, a `Vec` index read (`running[index]`), an
`Option::as_mut()` branch (`gather_cursors[index]`), a scratch-buffer write
(`operand_values[index] = ...`), and a`running[index] += strides[index]`
update, in favor of two direct slice reads and a scalar broadcast, per the
task's own framing that this bookkeeping is roughly "ten memory operations
to produce one multiply-accumulate" before ROW 3.

**Residual (V6, named not glossed):** whether the now-eliminated
per-element bookkeeping was also blocking autovectorization (i.e., would a
further pass — hoisting `apply_scalar_op`'s `op`/`reduce_op` match itself
out of the width loop via an explicit `match op { Multiply => ..., Add =>
... }` specialization, rather than relying on LLVM to do it — unlock NEON)
is unmeasured this session. The `apply_scalar_op` call inside
`reduce_width_binary`'s hot arm still receives `op`/`reduce_op` as runtime
parameters (loop-invariant within one `run_reduce` call, but not a
compile-time constant), so LLVM must still select among `ScalarOp`'s 15
arms via *some* mechanism per element unless it proved loop-unswitching was
profitable and did it silently — not confirmed either way this row. This is
the natural next-row candidate, not attempted here.

**KEPT** — both changes (ROW 2 and ROW 3), all four correctness gates green
on the exact binary the timing/allocation numbers above came from.

### ROW 3 addendum — clippy pedantic gate

`cargo clippy -p proxima-tensor --all-targets -- -D warnings` flagged
`reduce_width_binary`'s original 10-argument signature
(`clippy::too_many_arguments`, limit 7). Fixed by introducing `OperandSpan<'a>
{ data: &'a [f32], base: usize, contiguous: bool }` — a plain data bundle,
not a new algebra type — cutting `reduce_width_binary` to 6 args and
`reduce_width_unary` to 5. `cargo clippy -p proxima-tensor --lib --features
config -- -D warnings` → **clean, 0 warnings** on cpu.rs after the fix (the
only remaining `--all-targets` failure is `clippy::expect_used` in
`examples/profile_hot.rs:89`, pre-existing from ROW 0/prior sessions, not
touched by this task's edits, out of scope per the environment note to
ignore already-present `examples/`). Re-ran all four correctness gates
after the refactor: **128/128, 144/144, 25/25, 1/1** — unchanged. Re-timed:
1.391s/1.406s/1.421s (3 runs) — matches ROW 3's original 1.44-1.45s range
within noise, confirms the `OperandSpan` bundling didn't cost performance.
Allocations still 1,107 (0 delta). `root[0]=18370` unchanged.

### ROW 4 — hoist ScalarOp/reduce_op dispatch out of the width loop (monomorphized generics)

**Named hypothesis under test (ROW 3 residual):** `reduce_width_binary`'s /
`reduce_width_unary`'s width loop still called `apply_scalar_op(op, ...)`
and `combine_reduction(reduce_op, ...)` — both runtime matches over
`ScalarOp` — once per element, even though `op`/`reduce_op` are loop-
invariant for the whole call. Untested whether that residual branch/call
was blocking autovectorization.

**Change** (`proxima-tensor/src/cpu.rs`): `reduce_width_unary` and
`reduce_width_binary` are now thin dispatchers. Each matches on `op` ONCE
(7 arms for unary/arity-1, 8 arms for binary/arity-2 — the full domain
`ScalarOp::arity()` structurally admits for those `BodyShape` variants),
and inside each arm matches on `reduce_op` ONCE more, but only specializes
the four ops a fold realistically combines with — `Add`/`Multiply`/
`Maximum`/`Minimum` (sum/product/max-pool/min-pool). Each of the resulting
28 (unary) + 32 (binary) leaf combinations calls a new generic function —
`reduce_width_unary_monomorphic<F: Fn(f32) -> f32, R: Fn(f32,f32) -> f32>`
/ `reduce_width_binary_monomorphic<F: Fn(f32,f32) -> f32, R: Fn(f32,f32)
-> f32>` — with two concrete, non-capturing closure literals written
directly at the call site (e.g. `|x, y| x * y`, `|acc, v| acc + v`).
Passing closures as generic params (not `fn` pointers, not `dyn`) gives
each (op, reduce_op) pair its own monomorphized instantiation, so the
arithmetic is inlined as a literal operation, not a runtime-selected
value. `seeded` is also branched on ONCE per call, outside both the
contiguous and broadcast loop bodies (previously `combine_reduction`
re-checked it every element) — the two functions each contain 8 tight
loops total (4 stride-shape arms x 2 seeded arms), every one containing
exactly one call to the inlined `op` closure and, in the seeded arms, one
call to the inlined `reduce` closure — no branch, no indirect call, per
the task's requirement.

A `reduce_op` outside the accelerated four (`Subtract`/`Divide`/
`Greater`/`Equal` used as a fold combiner — legal by the type system,
arity 2, but not a reduction any current caller in this crate
constructs — verified by grep: `rg reduce_op proxima-tensor/src` shows
the only production call sites are `bind.rs:602`
(`reduce_op: reduce.body`, taking whatever `ScalarOp` the `Reduce` op
node carries, unconstrained by the type at that point) — falls back to
`reduce_width_unary_scalar_dispatch` / `reduce_width_binary_scalar_dispatch`,
the unmodified ROW 3 implementation (`apply_scalar_op` + `combine_reduction`,
per-element match, no vectorization). Named explicitly, not silently
narrowed — a fast-but-wrong result for an off-menu reduce_op would be a
correctness bug; this instead is a documented, correct, unaccelerated
fallback.

**Timing (direct `Instant`, `examples/profile_hot.rs`, release build, 1024^3
GEMM, 5 runs):**

| | ROW 3 (before ROW 4) | ROW 4 (after) |
|---|---|---|
| runs | 1.441s, 1.446s, 1.446s, 1.447s, 1.449s | 0.104s, 0.100s, 0.101s, 0.101s, 0.101s |
| range | 1.441s - 1.449s | 0.100s - 0.104s |
| CoV | ~0.2% | ~1.3% (mean 101.4ms, std ~1.36ms) |

**14.1x - 14.5x faster** (1.441/0.104 = 13.86x low end using the worst
after-run vs best before-run; 1.449/0.100 = 14.49x high end using the best
after-run vs worst before-run; point-estimate on means: 1.446s/0.101s =
14.32x). ns/MAC at the end of ROW 4: 1.073741824e9 MACs / 0.100-0.104s =
**0.0931 - 0.0969 ns/MAC** — this is AT or slightly BELOW the task's own
~0.1 ns/MAC hardware-ceiling estimate (M1 Max, ~3.2GHz, 4-wide NEON FMA);
the ceiling estimate was approximate (the M1's dual FMA-capable NEON
pipes issuing more than one 4-wide op per cycle would explain landing
under it). Combined across ROW 1-4 from the ROW 0 baseline (~8-10 ns/MAC):
**~83x - 108x**.

**Allocation count:** unchanged at **1,107** (5 runs, identical) — ROW 4
doesn't touch allocation, confirms orthogonality to ROW 2's fix.
`root[0]=18370`, `root_len=1048576` — byte-identical to every prior row.

**Disassembly — do NEON opcodes appear? YES**, confirmed by opening the
disassembly, not inferred from the speedup. `objdump -d` on the release
`profile_hot` binary (`scratchpad/opt/row4_full_disasm.txt`, 103,787
lines): `grep -c 'fmul\.4s\|fmul\.2s'` -> **337**; `grep -c 'fadd\.4s\|fadd\.2s'`
-> **327**; `grep -c 'ldp.*q[0-9]'` (128-bit paired vector loads) -> **1,310**.
`grep -c 'fmla'` -> **0** — LLVM chose separate multiply+add (not the
fused single-instruction form), still genuine 4-wide SIMD, not a fused
FMA. All of this sits inside the single symbol
`__RNvNtCskHGgWzEiGui_14proxima_tensor3cpu19reduce_width_binary` (disasm
line 24710 through the next symbol at line 45930 — LLVM inlined every one
of the 32 monomorphic instantiations directly into the dispatcher, same
pattern ROW 3 saw for `run_node_into`).

**Excerpt proving the exact GEMM arm vectorized** (lhs broadcast, `stride
0`, scalar-hoisted into `v0[0]`; rhs contiguous, `stride 1`, read via
`ldp`; accumulator read-modify-write via `ldp`/`stp` at `x11`; this is the
`(false, true)`-contiguous, `seeded = true` arm of
`reduce_width_binary_monomorphic`, i.e. `*slot = reduce(*slot, op(value_a,
value_b))` = `accumulator[i] = accumulator[i] + lhs_broadcast *
rhs[i]`, unrolled 4x, 4 lanes per instruction = 16 elements/iteration):

```
1000240d8: ad7f0981  ldp q1, q2, [x12, #-0x20]      ; rhs contiguous data
1000240dc: acc21183  ldp q3, q4, [x12], #0x40
1000240e0: ad7f1965  ldp q5, q6, [x11, #-0x20]       ; accumulator (old value)
1000240e4: ad404167  ldp q7, q16, [x11]
1000240e8: 4f809021  fmul.4s v1, v1, v0[0]           ; rhs * lhs_broadcast
1000240ec: 4f809042  fmul.4s v2, v2, v0[0]
1000240f0: 4f809063  fmul.4s v3, v3, v0[0]
1000240f4: 4f809084  fmul.4s v4, v4, v0[0]
1000240f8: 4e21d4a1  fadd.4s v1, v5, v1              ; accumulator + product
1000240fc: 4e22d4c2  fadd.4s v2, v6, v2
100024100: 4e23d4e3  fadd.4s v3, v7, v3
100024104: 4e24d604  fadd.4s v4, v16, v4
100024108: ad3f0961  stp q1, q2, [x11, #-0x20]       ; write accumulator
10002410c: ac821163  stp q3, q4, [x11], #0x40
100024110: f10041ad  subs x13, x13, #0x10
100024114: 54fffe21  b.ne 0x1000240d8
```

**Mechanism.** This IS a vectorization win, not a repeat of ROW 3's
non-SIMD bookkeeping-elimination mechanism — the disassembly excerpt above
is unambiguous: 128-bit paired loads, 4-lane `fmul.4s`/`fadd.4s`, 128-bit
paired stores, all inside one basic block with no branch except the
trip-count decrement. The residual named in ROW 3 (whether the
per-element `apply_scalar_op`/`combine_reduction` match was blocking
autovectorization) is **confirmed true**: removing it (replacing the
runtime match with a monomorphized closure call, resolved once per call
outside the loop) was sufficient for LLVM's autovectorizer to vectorize
the loop, with no `std::simd`/intrinsics needed.

**KEPT.** Correctness: `cargo nextest run -p proxima-tensor` -> **128/128**,
`--features config` -> **144/144**, `cargo nextest run -p omega --features
metal` (real Metal device) -> **25/25**, `cargo test -p proxima-tensor
--doc` -> **1/1**. All four green on the exact binary the timing numbers
above came from. `cargo clippy -p proxima-tensor --lib --features config
-- -D warnings` -> clean, 0 warnings (the nested nested nested macro +
32-arm dispatch did not trip `too_many_lines`/`cognitive_complexity` at
the configured pedantic level).

**Residual (V6, named not glossed):** the 32/28-combo bound is a
deliberate scope cut (reduce_op restricted to the 4 real fold ops), not
the full 8x8/7x8 cross product — an off-menu `reduce_op` still hits the
unaccelerated ROW-3 path. No production call site is known to construct
one (checked, see above), but this is a real, named narrowing of ROW 4's
acceleration, not of correctness. `run_elementwise`/`run_scan` still use
the un-monomorphized `eval_body_shape` -> `apply_scalar_op` path entirely
(no width-loop fast path at all yet, per-element like pre-ROW-3
`run_reduce`) — that is ROW 5's scope, not touched here.

### ROW 5 — extend the fast path to run_elementwise and run_scan

**Change** (`proxima-tensor/src/cpu.rs`): `run_elementwise` and `run_scan`
now compute `body_shape_is_affine_fast_path` once per bound op (the exact
function `run_reduce` already uses — reused verbatim, not reimplemented)
and, when it holds, skip the per-element gather-`Option`-check /
`operand_values` scratch copy / per-element `op` dispatch entirely:

- `run_elementwise` -> new `elementwise_width_fast` ->
  `elementwise_width_unary`/`elementwise_width_binary` (op matched once,
  32+28-combo-style monomorphized closure dispatch, same technique as
  ROW 4 but with no `reduce_op` axis — a plain map, no accumulator).
- `run_scan` -> new `scan_width_fast` -> `scan_width_unary`/
  `scan_width_binary` (op AND `reduce_op` both matched once, same
  4-reduce-op-accelerated / fallback-to-`_scalar_dispatch` split as
  ROW 4). Scan additionally requires the **output's** own width-dim
  stride to be 1 (`out_stride == 1`), checked once per bound op — the
  fast path writes into a contiguous `&mut [f32]` slice, so a strided
  output falls back to the unchanged per-element loop. Both operand-side
  and output-side eligibility are named explicitly; neither is silently
  narrowed.
- `reduce_width_binary`/`reduce_width_unary`'s ROW-4 dispatch pattern is
  reused conceptually (same macro shape, same 4-reduce-op restriction,
  same `OperandSpan` reads) but NOT literally shared as one function,
  since elementwise has no accumulator/`reduce_op` and scan's accumulate
  is a genuine sequential fold across the width dim (not a batched
  vector-accumulate like `run_reduce`'s) — `scan_width_binary_monomorphic`
  resolves the `!seeded`-first-element special case ONCE before its loop
  (not per element), then runs an unconditional fold loop.

**Two corrections made mid-implementation, before any gate ran (not
shipped as bugs):** (1) the scalar-dispatch fallbacks for scan
(`scan_width_unary_scalar_dispatch`/`scan_width_binary_scalar_dispatch`)
initially read `span.data[span.base]` unconditionally instead of
advancing by `index` for a contiguous span — caught by re-reading the
diff before building, fixed to branch on `span.contiguous` per read. (2)
`elementwise_width_unary`/`_binary`'s defensive `_` match arm (for an
op outside the 7/8 specialized ones) had the same bug and was additionally
provably unreachable (`BodyShape::Unary`/`Binary` only ever carry an
arity-matching `ScalarOp`) — replaced with `unreachable!()` instead of
leaving live-but-wrong dead code as a landmine.

**Clippy caught one real issue**: `scan_width_fast` at 8 arguments
tripped `too_many_arguments` (limit 7). Fixed by bundling `seeded`/
`accumulator` into a 2-field `ScanState` struct (same pattern as
`OperandSpan`, ROW 3 addendum) — cut it to 7 args. `cargo clippy -p
proxima-tensor --lib --features config -- -D warnings` -> clean, 0
warnings, after the fix.

**Measurement harness** (`proxima-tensor/examples/profile_hot.rs`,
extended this row — 3 new programs alongside the existing GEMM one):
- `elementwise_binary_program`: one `Multiply` over two 64M-element
  1-D inputs — a genuine `BodyShape::Binary`, qualifies for the ROW 5
  fast path.
- `elementwise_chain_program`: 7 single-use unary ops
  (`Negate`/`Reciprocal`/`Negate`/`Reciprocal`/`Negate`/`Negate`/`Tanh`)
  chained over 64M elements — the workload the task named ("64M-element
  7-op elementwise chain").
- `scan_program`: a plain cumulative sum (`Reduce` with `Keep::Scan`,
  `body: Add`, `element_body` = `Identity`) over 64M elements.

**Finding named before any number is reported: the 7-op chain does NOT
exercise ROW 5's fast path.** Read `bind.rs::compose_fused_operands`
(lines 637-661) — a chain of single-use elementwise nodes fuses into ONE
multi-step `ComposedBody` at bind time (the mechanism the c891caeb commit
"fuse elementwise chains into one bound op" added, predating this
session). `body_shape()` classifies any multi-step body as `Generic`
(`cpu.rs`, ROW 1), and `body_shape_is_affine_fast_path` returns `false`
for every `Generic` body by construction (`match *shape { ...
BodyShape::Generic(_) => false }`) — this was true before ROW 5 and is
unchanged by it. So the 7-op chain still runs entirely through
`apply_body`'s original per-element path, exactly as before ROW 1-5 ever
touched anything. This is a **scope finding, not a regression**: ROW 5's
fast path targets single-step (`Unary`/`Binary`) bound ops, and the
binder had already fused the named workload out of that shape before ROW 5
runs at all. Fusing a multi-step `Generic` body into its own vectorized
width loop is a materially larger change (would need to generate, per
step count and per op combination, its own monomorphized straight-line
sequence) and is out of this row's scope — named as the natural next
target, not attempted.

**Timing (direct `Instant`, `examples/profile_hot.rs`, release build, 5
runs each). "Before" measured on the exact ROW-3-committed `cpu.rs`
(`git stash push -- proxima-tensor/src/cpu.rs`, rebuild, measure, `git
stash pop`, rebuild — `examples/profile_hot.rs` is untracked so its new
programs survive the stash unchanged; this is the same A/B technique
ROW 2 used):**

| workload | before (ROW 3 state) | after (ROW 5) | ratio |
|---|---|---|---|
| gemm 1024^3 (context check the stash round-tripped correctly) | 1.379s - 1.397s | 0.100s - 0.104s | ~13.3x - 14.0x (matches ROW 4's own number, confirms the stash was clean) |
| elementwise_binary, 64M (Multiply) | 0.3395s - 0.3482s | 0.1093s - 0.1193s | **2.85x - 3.19x** |
| elementwise_chain, 64M (7-op, Generic) | 2.8188s - 2.8545s | 2.8090s - 2.8789s | **~1.0x — no measurable change**, as predicted above |
| scan, 64M (cumsum) | 0.3029s - 0.3138s | 0.1304s - 0.1431s | **2.11x - 2.41x** |

`root[0]`/`root_len` identical before/after for all four workloads
(`elementwise_binary root[0]=1`, `elementwise_chain root[0]=0.7615942`,
`scan root[0]=1`, all `root_len=67108864`) — same numerical output,
confirming the fast path (and the scalar-dispatch fallback bug fixes
made before this run) did not change results.

**Disassembly — elementwise_binary: vectorized, confirmed by opening the
binary.** `objdump -d` (`scratchpad/opt/row5_full_disasm.txt`, 117,330
lines): global `fmul.4s`/`fmul.2s` count rose from ROW 4's 337 to
**352** (+15), `fadd.4s`/`fadd.2s` from 327 to **342** (+15), paired
128-bit loads (`ldp.*q[0-9]`) from 1,310 to **1,387** (+77). The delta
sits inside the `evaluate` symbol (`run_elementwise` was fully inlined
into it, same LTO pattern ROW 0/3 saw for `run_reduce`/`run_node_into`),
excerpt at `scratchpad/opt/row5_full_disasm.txt:100043484`-`1000434b0`:

```
100043484: ad7f0500  ldp q0, q1, [x8, #-0x20]     ; operand a, 4x unrolled
100043488: acc20d02  ldp q2, q3, [x8], #0x40
10004348c: ad7f1544  ldp q4, q5, [x10, #-0x20]    ; operand b
100043490: acc21d46  ldp q6, q7, [x10], #0x40
100043494: 6e24dc00  fmul.4s v0, v0, v4
100043498: 6e25dc21  fmul.4s v1, v1, v5
10004349c: 6e26dc42  fmul.4s v2, v2, v6
1000434a0: 6e27dc63  fmul.4s v3, v3, v7
1000434a4: ad3f0560  stp q0, q1, [x11, #-0x20]    ; write out
1000434a8: ac820d62  stp q2, q3, [x11], #0x40
1000434ac: f100418c  subs x12, x12, #0x10
1000434b0: 54fffea1  b.ne 0x100043484
```

Pure elementwise `a * b`, both operands contiguous, no accumulator — the
`(true, true)` arm of `elementwise_width_binary_monomorphic`. **This IS
vectorization** (same mechanism as ROW 4). The elementwise ratio (~3x)
is smaller than GEMM's (~14x) because this workload reads 2 x 64M x 4
bytes and writes 1 x 64M x 4 bytes (768MB total traffic) with only one
multiply per element — almost certainly memory-bandwidth-bound rather
than compute-bound, so SIMD narrows the compute side without touching the
bandwidth ceiling; not separately re-measured this row (named, not
glossed).

**Disassembly — scan: confirmed NOT vectorized, matching the mechanism
predicted in the code comment before the binary was ever built.**
`objdump -d` symbol `scan_width_binary` (line 49187) through
`scan_width_binary_monomorphic` (line 54199) to the next symbol (line
54297): `grep` for `fmul.4s`/`fadd.4s` inside that whole range -> **0**.
The excerpt at that address is scalar control flow (arity/contiguity
dispatch, `bl` calls into the monomorphic instantiations) with a single
`fsub s4, s8, s0` (a scalar float subtract computing the seed value from
two accumulator states, not a lane operation). **Scan's ~2.2x speedup is
the same non-SIMD mechanism ROW 3 established for `run_reduce`**:
eliminating the per-element `Option` check, scratch copy, and per-element
op dispatch — not vectorization, exactly as predicted, and confirmed
rather than assumed.

**KEPT** — `run_elementwise`'s fast path (genuine ~3x + vectorization,
confirmed) and `run_scan`'s fast path (genuine ~2.2x, confirmed non-SIMD,
correctly labeled as such). Correctness on the exact binary the numbers
above came from: `cargo nextest run -p proxima-tensor` -> **128/128**,
`--features config` -> **144/144** (re-run a second time after the
stash-pop round-trip to confirm the restore was clean -> **144/144**
again), `cargo nextest run -p omega --features metal` (real Metal
device) -> **25/25**, `cargo test -p proxima-tensor --doc` -> **1/1**.
`cargo clippy -p proxima-tensor --lib --features config -- -D warnings`
-> clean, 0 warnings.

**Allocation count:** not separately re-measured this row (ROW 5 doesn't
touch allocation — same discipline as ROW 3/4, which only re-checked
allocation when they were the row under test). `evaluate()`'s GEMM path
still shows **1,107** in every timing run above (printed alongside every
GEMM invocation), unchanged.

**Residual (V6, named not glossed):** the 7-op elementwise chain (the
literal workload the task named) is untouched by this row's fast path,
for the structural reason given above — a real gap between "what the
task described" and "what the binder's existing fusion makes reachable."
`run_elementwise`'s fast path is real and measured on a single-step body;
extending acceleration to fused multi-step (`Generic`) bodies is not
attempted this session. Scan's fast path only fires when the output
layout itself has width-dim stride 1 — a scan writing into a
non-contiguous output view falls back unaccelerated, unmeasured this row
(no test in the 128/144-test suite specifically constructs one; the
existing scan-shape tests that passed do not by themselves prove that
fallback path is exercised, only that it is *correct* by inspection and
by the shared code path with the pre-ROW-5 behavior it falls back to).

### ROW 6 — align Interpreter's In so the full three-stage chain composes

**Not a perf row** (per the owner's framing) — a shape fix, requested via
the coordinator mid-session, in a file this session already owns
(`cpu.rs`). A prior session's `Interpreter` `Pipe` impl had `In = BoundOp`
against `BoundOpBuilder::Out = Vec<BoundOp>`, and rather than fix the
mismatch, kept a hand-rolled per-node driving loop at the one call site
(`for computed in ready_nodes { block_on(Pipe::call(&executor, computed))
}`, in the crate's own composition-proof test) and documented the E0271
rejection of `shapes.and_then(builder).and_then(executor)` as "a genuine
multiplicity boundary of push-based fusion, not a container" — prose
defending the mismatch rather than fixing it.

**Change** (`proxima-tensor/src/cpu.rs`):
- `impl Pipe for Interpreter<'_>`: `type In` changed from `BoundOp` to
  `Vec<BoundOp>`. `call` now loops over the batch internally, folding
  each ready node into the buffer table in order — the same fold the
  buffer table already did one write at a time, now driven for every
  element of one `Vec<BoundOp>` per call instead of one call per element.
  An empty batch is a no-op loop body, not a special-cased branch.
- Module header doc (lines 30-56) and the `Interpreter` struct doc
  rewritten: no more claim of "a genuine multiplicity boundary... not a
  container" or "does NOT typecheck" — replaced with the actual fact,
  that `Second::In = First::Out` now holds by construction
  (`BoundOpBuilder::Out = Vec<BoundOp> = Interpreter::In`), so the
  three-stage chain composes with `AndThen` directly.
- `execute_composes_through_pipe_ext_matching_the_free_function` (the
  crate's composition-proof test) rebuilt around
  `shapes.and_then(builder).and_then(Interpreter::new(&mut buffers))`,
  called once per `Op` record in the program (`for expr in &program {
  block_on(Pipe::call(&chain, expr.clone()))... }`) — the hand-rolled
  inner loop over `ready_nodes` is gone; the chain's own sink absorbs the
  batch. Reading the result back required one adaptation: `Interpreter`
  is moved into the `AndThen`, so its `get()` is unreachable after
  `chain` is built — the test `drop(chain)` to release the buffer table's
  mutable borrow, then reads `buffers[sum.0 as usize].clone()` directly,
  which is byte-for-byte the same read `Interpreter::get` performs (both
  read the identical backing `Vec<Option<Vec<f32>>>`). `Interpreter::get`
  itself is untouched, per the instruction to keep it as-is — the test
  just cannot call it once ownership moved, and says so in a comment
  rather than silently working around it.

**Only call site of `Interpreter`/`Pipe::call(&interpreter, ...)` in the
crate is this one test** (checked: `rg "Interpreter::new|Interpreter<"
proxima-tensor/src` and `rg "Pipe::call(&(interpreter|executor)"`) —
`evaluate`/`evaluate_parallel` call `run_node_into` directly, never
through the `Pipe` abstraction, so this change has no other call sites to
update.

**The chain typechecks by the existing `AndThen` bound**
(`proxima-primitives/src/pipe/primitives.rs:203-211`,
`Second: Pipe<In = First::Out>`, `Second::Err: From<First::Err>`):
`ShapeTable::Out = (Op, Shapes) = BoundOpBuilder::In` at the first join
(already true before this row), and now `BoundOpBuilder::Out =
Vec<BoundOp> = Interpreter::In` at the second. All three stages already
share `Err = TensorError`, so the `From` bound is the reflexive blanket
impl. Confirmed by the compiler, not asserted: the test above compiles
and passes.

**KEPT.** `cargo nextest run -p proxima-tensor -E
'test(execute_composes_through_pipe_ext)'` -> **1/1**. Full gates on the
same binary: `cargo nextest run -p proxima-tensor --features config` ->
**144/144**, `cargo nextest run -p omega --features metal` (real Metal
device) -> **25/25**, `cargo test -p proxima-tensor --doc` -> **1/1**.
`cargo clippy -p proxima-tensor --lib --tests --features config -- -D
warnings` -> clean, 0 warnings.

### ROW 7 — vocabulary sweep: no pipe kinds, name the instantiation

**Not a perf row.** The owner's correction: "sink"/"source"/"observe"/
"transform" are not categories a type belongs to — each is the
instantiation of `Pipe`'s two associated types (`Out = ()` for what this
session had been calling "sink", `In = ()` for "source", `Out = In` for
"observe"). The concrete failure the owner named: reading "sink" as a
type-level kind produced a wrong claim about `FanOut<S, Policy>`'s bound
(`S: Pipe` with `S::Out` unconstrained — `Out = ()` belongs to `FanOut`'s
own impl, not to a constraint on its arms). This row's scope: find every
doc comment in the files this session owns (`proxima-tensor/src/*`,
`omega/src/*`) using a pipe-kind category noun where an instantiation was
meant, and rewrite to instantiation language or neutral prose. No public
type renamed; comments/docs only.

**Search** (`rg -rniE 'sink' --include="*.rs" proxima-tensor/src
omega/src`, then a second pass for `a source`/`the transform`/`an
observe`/`pipe kind` as doc-comment nouns): found 10 occurrences, all in
`proxima-tensor/src/cpu.rs` and `proxima-tensor/src/lib.rs` — 8 of them
introduced by this session's own ROW 3/4/5/6 doc comments (naming
`Interpreter` "a SINK"/"the sink" while explaining its `Pipe` impl), 1
pre-existing in `proxima-tensor/src/shape.rs:278` ("the observe form"),
`omega/src` had zero hits (its doc comments never used pipe-kind nouns to
begin with).

**Fixes:**
- `cpu.rs` module header, `Interpreter` struct doc, `Interpreter::new`/
  `get` doc comments, and the ROW-6 test's doc comment: every "a SINK"/
  "the sink('s)"/"a real sink"/"this sink"/"what a sink produced" rewritten
  to name `Interpreter` directly or state the instantiation
  (`` `In = Vec<BoundOp>`, `Out = ()` `` / "this stage" / "`Interpreter`
  never allocates..."). One phrase ("not a transform wearing a mutation")
  also swapped "a transform's result" for "a nonempty `Out`" — same fix,
  the contrast no longer names a category on either side.
- `lib.rs`'s crate-level doc (`# Stream stance` section) had both a
  vocabulary violation ("a SINK") AND a **stale factual claim** left over
  from before ROW 6: it still said `Interpreter`'s `In = BoundOp` and that
  the three-stage chain "does not typecheck," describing the pre-ROW-6
  state as current. Rewritten in one pass: `In = Vec<BoundOp>` (matching
  ROW 6's actual signature), and the chain now correctly stated as
  composing (`Second::In = First::Out` holds at both joins), with no
  category noun.
- `shape.rs:278` — found a **second, independent defect** while sweeping
  vocabulary, not itself a vocabulary issue: the doc claimed `` `In = Out
  = Op` `` (implying the observe instantiation, `Out = In`) but the actual
  impl three lines below is `type Out = (Op, Shapes)` — not equal to
  `In = Op`. The doc was simply wrong, independent of the "observe form"
  phrasing. Fixed both in one edit: `` `In = Op`, `Out = (Op, Shapes)` ``
  (the true, current types) with the category noun dropped.

**KEPT.** `cargo nextest run -p proxima-tensor --features config` ->
**144/144**, `cargo nextest run -p omega --features metal` (real Metal
device) -> **25/25**, `cargo test -p proxima-tensor --doc` -> **1/1**,
`cargo clippy -p proxima-tensor --lib --tests --features config -- -D
warnings` -> clean, 0 warnings (doc-only edits, no intra-doc link
breakage, no behavior change — expected and confirmed).

**Residual (V6):** this sweep covered `proxima-tensor/src` and
`omega/src` only, per the owner's named scope for this row — ROW 8
covers `proxima-primitives/src/pipe/fanout.rs` separately, and no other
crate in the workspace was swept (out of scope, not silently declared
clean).

### ROW 8 — fix the misleading FanOut doc at its source (proxima-primitives)

**Not a perf row.** `proxima-primitives/src/pipe/fanout.rs` is in this
worktree (a full-repo checkout, not just `proxima-tensor`) and the owner
named its doc as the source of the vocabulary bug ROW 7 was reacting to:
`FanOut<S, Policy>`'s bound is `S: Pipe` (or `S: SendPipe`) with `S::In:
Clone` and **no constraint on `S::Out`** — verified by opening the actual
impls, `proxima-primitives/src/pipe/fanout.rs:136-144` (`SendPipe` arm)
and `:185-193` (`Pipe` arm): both list `S::In: Clone [+ Send]` and
`Policy: FanPolicy`, nothing on `S::Out`. Each arm's `.call()` result is
pattern-matched only for `Err` (`if let Err(err) = sink.call(...).await`,
lines 156/165/203/212) — any `Ok(value)` an arm produces is silently
discarded. `FanOut`'s OWN impl sets `type Out = ();` (line 143/192). The
doc at line 69 (module doc at line 1, and the composing sentence at
lines 5-6) said "N sink `SendPipe`s" / "the sinks are ordinary pipes, so
'a sink' needs no bespoke trait" — read as a type constraint, this claims
every arm's `Out` is `()`, which the bound does not say and the `call`
body does not require.

**Change** (`proxima-primitives/src/pipe/fanout.rs`, doc comments only,
3 sites — the module header and the `FanOut` struct doc, the two places
that made the type-narrowing claim; the ~35 other informal uses of
"sink"/"sinks" elsewhere in the file — describing runtime role in
`FanPolicy` variant docs, constructor docs, the hand-written poll state
machine's docs, and test comments — do not claim `S::Out = ()` and were
left as informal English, not touched, to keep this row terterse and
scoped to the actual defect rather than a blanket rename):
- Module doc: `"broadcast one input to N sink SendPipes"` -> `"broadcast
  one input to N SendPipe arms"`; `"the sinks are ordinary pipes, so 'a
  sink' needs no bespoke trait"` -> `"each arm is an ordinary pipe with
  no bespoke trait of its own; S::Out is unconstrained (an arm can be any
  pipe, including one with a real, non-() output — FanOut discards
  whatever each arm produces)"`.
- `FanOut` struct doc: `"Broadcast composition over N sink SendPipes."`
  -> `"Broadcast composition over N SendPipe arms — each arm any pipe
  (S::Out unconstrained; FanOut's own call discards whatever each arm
  produces, matching its own Out = ())."`

No public type renamed (`sinks` field, `sink_count()` method, `SinkFut`
poll-state type, all untouched), per the instruction.

**KEPT.** `cargo nextest run -p proxima-primitives` -> **410/410** (N
asserted, nonzero). `cargo clippy -p proxima-primitives --lib --tests --
-D warnings` -> clean, 0 warnings. Re-ran the dependent crates' gates
since `proxima-primitives` is a shared dependency: `cargo nextest run -p
proxima-tensor --features config` -> **144/144**, `cargo nextest run -p
omega --features metal` (real Metal device) -> **25/25**, `cargo test -p
proxima-tensor --doc` -> **1/1** — unaffected, as expected for a
doc-only change in a dependency, confirmed rather than assumed.

## ROW 10 — layout sensitivity of the fast path (coordinator, MEASURED)

`body_shape_is_affine_fast_path` requires EVERY operand's stride along the
innermost iteration dim to be 0 or 1. The innermost dim for a `Keep::Reduce`
matmul is `j` (the last OUTPUT dim), not the reduction dim `k`.

Same harness (`examples/profile_hot.rs`), same release build, same machine,
1024^3 GEMM. Only the RHS layout changed:

| RHS shape | RHS map      | innermost strides | fast path | time    | ns/MAC |
|-----------|--------------|-------------------|-----------|---------|--------|
| `[k,n]`   | `[2,1]`      | `[0, 1]`          | YES       | 0.103 s | 0.096  |
| `[n,k]`   | `[1,2]`      | `[0, 1024]`       | NO        | 5.342 s | 4.98   |

**52x, from layout alone.** MEASURED, both runs, EXIT=0.

Consequence: ROW 1-4's headline (0.093-0.097 ns/MAC, "at the hardware
ceiling") holds ONLY for the `[k,n]` layout. `benches/bench_vs_ggml.rs`
uses `[n,k]` because that is ggml's own `mul_mat` convention (B transposed),
so every ggml comparison lands on the uncovered path. The ~86x deficit seen
in the partial re-bench at 512^3 is this, not a regression.

Open, not fixed: the fast path needs to cover a non-unit innermost stride on
one operand — loop-order choice (make `k` innermost when the output stride
is bad), or a strided variant of `reduce_width_binary`. Neither attempted.

## ROW 11 — cover the transposed-B layout: loop-order choice (mechanism 1), landed

**Scope:** `run_reduce` only (the `Keep::Reduce` matmul path ROW 10 measured).
`run_elementwise`/`run_scan` don't have a contraction dim at all, so ROW 10's
defect doesn't apply to them — not touched, not silently narrowed.

**Change** (`proxima-tensor/src/cpu.rs`): when the width dim `n`'s own stride
disqualifies the existing `fast_path` (`body_shape_is_affine_fast_path` on
`strides`) but the bound op has exactly one contraction dim and THAT dim is
affine on every operand the body reads, `run_reduce` folds along `k` for one
output position at a time instead of accumulating across `n` once per `k`
step. Concretely:

- `reduction_strides` — `strides`'s sibling table, computed once per bound
  op, holding each operand's stride along the single contraction dim instead
  of the width dim.
- `reduction_fast_path = !fast_path && reduction_dims.len() == 1 &&
  body_shape_is_affine_fast_path(resolved, &shape, &reduction_strides)` —
  **the existing eligibility predicate reused verbatim**, just handed a
  different dim's stride table. No change to
  `body_shape_is_affine_fast_path` itself.
- A new `if reduction_fast_path { ... } else { <the untouched ROW 3/4
  reduction_flat loop> }` branch inside `run_reduce`'s `leading_flat` loop:
  for each output position (the `n in 0..width` walk, advanced via the
  EXISTING `strides`/`running` increment idiom the generic path already
  uses), computes the whole `k`-length contraction in one call to a new
  `reduce_dot_fast`, instead of one `reduce_width_fast` call per `k` with an
  N-wide accumulator.
- `reduce_dot_fast` / `reduce_dot_unary` / `reduce_dot_binary` /
  `reduce_dot_{unary,binary}_monomorphic` / `reduce_dot_{unary,binary}_scalar_dispatch`
  — the contraction-dim counterpart of `reduce_width_fast` and friends, same
  op/reduce_op monomorphized-closure dispatch technique as ROW 4, same
  `OperandSpan` (data/base/contiguous) reused unchanged, same 4-reduce-op
  acceleration + scalar-dispatch-fallback split. `DotFold { len, init,
  seeded }` bundles the fold's three scalar parameters (mirrors
  `OperandSpan`/`ScanState`, keeps `reduce_dot_binary` at 5 arguments instead
  of tripping `clippy::too_many_arguments`).

**Correctness argument (not just tested — reasoned, then confirmed):** the
original per-step loop accumulates `accumulator[n] = combine(accumulator[n],
value_k)` for `k = 0, 1, ..., K-1` in that exact sequential order, with
`accumulator[n]` pre-seeded to `initial_value(init)` and `seeded` starting
`false` only for `ReduceInit::FirstElement`. `reduce_dot_*_monomorphic`
performs the identical sequence of `combine`/`reduce` calls, in the same
`k`-ascending order, for the same starting `(init, seeded)` pair — just
inside one function call instead of across `K` separate calls into
`reduce_width_fast`. No floating-point operation is reordered, so the result
is bit-for-bit the same regardless of which loop drives it.

**New test** (`proxima-tensor/src/cpu.rs` test module):
`fused_matmul_with_transposed_rhs_matches_a_naive_triple_loop` — RHS stored
`[n, k]`, `k = 7` (not a multiple of the NEON lane width, to exercise the
fast path's scalar remainder), asserted `assert_eq!` (exact, not
approximate) against the same `naive_matmul` triple loop the existing
`fused_matmul_matches_a_naive_triple_loop` test already uses for the `[k,
n]` layout. **Passed on the first run** — direct confirmation of the
correctness argument above, not just a hope.

**Runtime confirmation the two fast paths are disjoint and each fires for its
own layout (not just eligible in theory):** temporary
`eprintln!("fast_path={fast_path} reduction_fast_path={reduction_fast_path}
strides={strides:?} reduction_strides={reduction_strides:?}")` immediately
after computing `reduction_fast_path`, rebuilt, ran both GEMM programs once,
reverted the line, rebuilt clean before any gate/timing run counted below.
Output:

```
ROW11-VERIFY fast_path=true  reduction_fast_path=false strides=[0, 1]    reduction_strides=[1, 1024]   <- [k,n]
ROW11-VERIFY fast_path=false reduction_fast_path=true  strides=[0, 1024] reduction_strides=[1, 1]      <- [n,k]
```

`[k,n]` still takes the ROW 3/4 width-based `fast_path` unchanged;
`[n,k]` now takes the new `reduction_fast_path`, with `reduction_strides =
[1, 1]` — both operands contiguous along `k`, exactly the case ROW 10 named.

**Measurement harness** (`proxima-tensor/examples/profile_hot.rs`): added
`matmul_program_rhs_transposed` (RHS shape `[n, k]`, map
`projection(3, &[1, 2])`) and a second timed block in `main` building the RHS
buffer as the literal transpose of the `[k, n]` buffer, so `root[0]` is
directly comparable between the two layouts in one run.

**Timing (direct `Instant`, `examples/profile_hot.rs`, release build, 1024^3
GEMM, this session's host, 3-5 runs per layout, on the exact binary the gates
below ran against):**

| RHS layout | before (ROW 10) | after (ROW 11) | ns/MAC before | ns/MAC after |
|---|---|---|---|---|
| `[k, n]` (unchanged, `fast_path`) | 0.103s | 0.101s - 0.110s | 0.096 | 0.094 - 0.102 |
| `[n, k]` (`reduction_fast_path`, this row) | 5.342s | 0.882s - 0.933s (8 runs across two build cycles) | 4.977 | 0.822 - 0.869 |

**5.72x - 6.06x faster** on the transposed layout (5.342 / 0.933 = 5.73x low
end, 5.342 / 0.882 = 6.06x high end). The layout gap (`[n,k]` time / `[k,n]`
time) shrinks from ROW 10's **52x** to **8.0x - 9.2x** (0.882/0.110 = 8.02x
best case for the after numbers, 0.933/0.101 = 9.24x worst case) — a real,
large reduction in the gap, not closure of it. `root[0]=18370` identical to
the `[k,n]` layout's own `root[0]` in every run (5+ repeats) — same numbers,
different layout, matching the byte-exact test above.

**Disassembly — what actually got vectorized, and what didn't (checked, not
assumed).** `objdump -d` on the release `profile_hot` binary
(`scratchpad/opt/row11_full_disasm.txt`, 141,075 lines). `reduce_dot_binary`
(all 32 op x reduce_op combinations LTO-inlined into one symbol, same
pattern ROW 4/5 saw) spans lines 54813-72572 of that file
(`scratchpad/opt/reduce_dot_binary_slice.txt`, 17,760 lines): **90
`fmul.4s`/`fmul.2s`, 90 `fadd.4s`/`fadd.2s`, 384 paired 128-bit loads
(`ldp q`), 0 `fmla`, 0 `faddv`/horizontal-add-vector-reduce instructions
anywhere in the function.**

The `Maximum`/`Minimum` reduce_op arms DO fully vectorize end-to-end
(`fminnm.4s`/`fmaxnm.4s` accumulate, closed by a single `fminnmv.4s`/
`fmaxnmv.4s` horizontal reduce — confirmed by opening one such arm at
`scratchpad/opt/reduce_dot_binary_slice.txt:8515`-`8571`). **The GEMM
combination — `op = Multiply`, `reduce_op = Add` — does NOT**, and this is
mechanistic, not a missed optimization: floating-point addition is not
associative, so reordering a `+=` chain changes the bit pattern of the
result. LLVM will not do that without fast-math, and this session's
correctness bar (byte-identical to the naive triple loop, enforced by
`assert_eq!` in the tests above) requires that it doesn't. Confirmed by
directly opening the exact `Multiply`/`Add` arm inside `reduce_dot_binary`,
found via its unique fingerprint (`fmul.4s` immediately followed by 12 `mov
sN, vM[lane]` lane-extract instructions and 15 scalar `fadd s0, s0, sN`
additions in strict lane order), at
`scratchpad/opt/reduce_dot_binary_slice.txt:8869`-`8908`:

```
10003e7c8:  ldp   q5, q6, [x15, #-0x20]     ; rhs, contiguous along k
10003e7cc:  ldp   q7, q16, [x15], #0x40
10003e7d0:  fmul.4s v0, v0, v5              ; lhs * rhs, 4 lanes at once
10003e7d4:  mov   s5, v0[3]                 ; extract each lane...
10003e7d8:  mov   s17, v0[2]
10003e7dc:  mov   s18, v0[1]
...
10003e810:  fadd  s0, s1, s0                ; ...and add them back scalar,
10003e814:  fadd  s0, s0, s18               ; in strict element order 0,1,2,3
10003e818:  fadd  s0, s0, s17               ; per group of 4, group-by-group
10003e81c:  fadd  s0, s0, s5                ; -- this IS the naive loop's
10003e820:  fadd  s0, s0, s2                ; own left-to-right summation
10003e824:  fadd  s0, s0, s20               ; order, never reassociated
```

Cross-checked against an isolated, minimal probe outside the crate
(`scratchpad/opt/dotprobe.rs`, same `-C opt-level=3 -C lto=fat -C
codegen-units=1`, no `target-cpu=native`/fast-math — same flags this
worktree's release profile uses): a bare `for i in 0..n { acc = acc + a[i] *
b[i] }` compiles to the **identical** vectorized-multiply /
scalar-lane-extract / sequential-scalar-add shape
(`scratchpad/opt/dotprobe_disasm.txt:6`-`98`), while the `f32::min` version
of the same loop compiles to the full `fminnm.4s` + `fminnmv.4s`
horizontal-reduce form (`scratchpad/opt/dotprobe_disasm.txt:101`+). This is
a general LLVM/AArch64 fact about float-add reductions under strict IEEE
semantics, not something specific to this codebase — checked directly, not
inferred from the speedup number.

**Mechanism, stated plainly:** ROW 11's ~5.7-6.1x win is **partial-SIMD +
non-SIMD bookkeeping removal**, the same mechanism family ROW 3 established
(straight-line `OperandSpan` reads replacing the per-element `Option`-checked
scratch copy) PLUS a genuine but partial vectorization win on the multiply
(4 lanes at once via `fmul.4s`) that the horizontal add cannot share, because
sequential float addition order is load-bearing for this session's
correctness bar. It is not, and structurally cannot be, ROW 4's ~14x
full-SIMD-throughput win — that combination (both operands independent
per-lane, no serial accumulation dependency) is exactly what `[k,n]`'s width-
based `fast_path` already has and this transposed case does not.

**Mechanism 2 — strided variant of `reduce_width_binary`, evaluated, NOT
landed (the loser, per the task's own instruction to record it):**

Rather than swap loop order, mechanism 2 keeps `n` as the parallel/lane
dimension (the *existing* `fast_path` shape, which ROW 4 already proved
vectorizes fully end-to-end for the untransposed layout) and instead lets
one operand's `OperandSpan` carry a non-unit stride, reading `rhs[n *
stride]` per lane instead of a contiguous slice.

Evaluated via an isolated, same-flags probe rather than fully wired into
`cpu.rs` (`scratchpad/opt/strideprobe.rs`) — reusing the exact `dotprobe.rs`
methodology above, because wiring mechanism 2 for real requires widening
`OperandSpan` to a third (strided) read shape and re-deriving
`body_shape_is_affine_fast_path`'s semantics from "0 or 1" to "any stride,"
a materially bigger, riskier change than mechanism 1's, and the probe
answers the "which mechanism wins" question this task asks for without that
risk. `width_accumulate_strided` (the `n`-parallel, `rhs`-strided-by-`k`
shape) vs `dot_contiguous` (mechanism 1's shape) on the same 1024^3-scale
synthetic data, 3 runs each:

| | mechanism 2 (strided width-accumulate) | mechanism 1 (contiguous dot fold) |
|---|---|---|
| runs | 0.858s, 0.870s, 0.875s | 0.876s, 0.878s, 0.882s |
| range | 0.858s - 0.875s | 0.876s - 0.882s |

**Statistically indistinguishable** (ranges overlap by 0.876-0.875s) — no
measured advantage to mechanism 2 over mechanism 1 at this scale, on this
host. The task's own prediction ("a strided load will not vectorize as well
— measure rather than assume") is **not falsified but also not confirmed as
a large effect** here: mechanism 2 is not measurably worse than mechanism 1
in this probe, just not measurably better, while requiring a strictly larger
and riskier production change (new `OperandSpan` read shape threaded through
every one of its callers: `reduce_width_fast`, `elementwise_width_fast`,
`scan_width_fast`, not just `run_reduce`). **Decision: keep mechanism 1
(landed, tested, gated); mechanism 2 not implemented in `cpu.rs` — recorded
here as evaluated-and-not-pursued, not silently dropped.** If a future
session wants the last ~8x of the remaining gap, re-measuring mechanism 2
properly wired (not probed) on real GEMM data, and/or a hand-vectorized
partial-sums variant with an explicit "this changes summation order, needs a
tolerance-based test instead of `assert_eq!`" carve-out, are the two next
candidates — neither attempted here.

**Shapes still falling back to the fully generic (unaccelerated) loop,
named explicitly:** (1) any `Keep::Reduce` fold with more than one
contraction dim (`reduction_dims.len() != 1`) — `reduction_fast_path` is
gated off entirely, regardless of stride shape; (2) a gathered operand on
either the width or the contraction dim; (3) a `Generic` (multi-step fused)
body, same as every prior row; (4) a layout where NEITHER the width dim NOR
the single contraction dim is affine (stride 0/1) on every operand — e.g. a
genuinely strided view on both axes. None of these are exercised by
`bench_vs_ggml.rs`'s GEMM rows; named as residual, not measured this
session.

**KEPT.** Correctness on the exact binary the timing numbers above came
from: `cargo nextest run -p proxima-tensor` -> **129/129** (128 prior +
this row's new transposed-RHS parity test), `--features config` -> **145/145**,
`cargo nextest run -p omega --features metal` (real Metal device) ->
**25/25**, `cargo test -p proxima-tensor --doc` -> **1/1**. `cargo clippy -p
proxima-tensor --lib --tests --features config -- -D warnings` -> clean, 0
warnings (the `DotFold` bundle was needed specifically to stay under
`clippy::too_many_arguments`, same discipline as `OperandSpan`/`ScanState`).

**Allocation count:** not separately re-measured this row — `reduce_dot_*`
and the `reduction_fast_path` branch allocate nothing (`reduction_strides`
is the only new heap allocation, and it is loop-invariant, built once per
bound op, matching ROW 2's discipline); the GEMM path's own allocation count
(1,107-1,108) printed alongside every timing run above is unchanged.

layout-target removed at the end of this session (see final cleanup below).

## New session — repo re-checked at HEAD e4629496 (matches the ROW 11 commit exactly:
`perf(tensor): fold the contraction dim when the width dim is strided`).
CARGO_TARGET_DIR repinned to `scratchpad/acc-target` for this session's
commands per this task's environment note (never `export`). Host: same M1
Max, this run.

## ROW 12 — vectorize the reduce_dot fold with multiple accumulators

**Target, as diagnosed by the task:** `reduce_dot_binary_monomorphic`'s
`(true, true)` arm (the transposed-RHS/`reduction_fast_path` GEMM
combination, `op = Multiply`, `reduce_op = Add`) folds via one strict
left-to-right scalar chain — ROW 11 showed this compiles to `fmul.4s`
(4-wide multiply, real SIMD) immediately followed by 12 `mov sN, vM[lane]`
extracts and 15 scalar `fadd s0, s0, sN` in strict lane order, because
float `+` is not associative and the naive loop's exact left-to-right
order was this session's correctness bar. Fix, as instructed: N
independent partial accumulators, summed once at the end, so the loop's
accumulate step is no longer a single serial dependency chain.

**Scope check — does `reduce_width_*` need the same fix? NO, confirmed by
re-reading ROW 4's own disassembly finding, not re-measured this row.**
`reduce_width_binary_monomorphic`'s accumulator is the whole `n`-wide
output span (`accumulator: &mut [f32]`, one slot per **independent**
output element) — the width loop's `k` steps write
`accumulator[i] = accumulator[i] + lhs*rhs[i]` for every `i` in the same
call, so the "N accumulators" ROW 4 already has are `n` of them (up to
1024, not 4 or 8), one per real output position — not a single scalar
being serially folded. That is exactly why ROW 4 measured full
`fmla`-shaped SIMD (`fmul.4s` + `fadd.4s`, `ldp`/`stp` q-registers, zero
lane-extraction, `scratchpad/opt/discipline.md` ROW 4) with no
associativity problem at all: each lane IS a different output element,
never combined with another lane. `reduce_dot_*` is the one family that
folds many terms into ONE scalar per call — the associativity bottleneck
is specific to it. Scope: `reduce_width_*` untouched this row, correctly.

**Change** (`proxima-tensor/src/cpu.rs`):
- `DOT_LANES: usize = 8` (module-level const, tunable — see the 4-vs-8
  measurement below).
- `dot_fold_multi_accumulator_binary<F, R>` / `dot_fold_multi_accumulator_unary<F, R>`
  replace the old single-accumulator fold in `reduce_dot_binary_monomorphic`'s
  `(true, true)` arm and `reduce_dot_unary_monomorphic`'s contiguous
  branch. Both operate on `chunks_exact(DOT_LANES)` over real `&[f32]`
  slices (not manual `slice[index]` indexing behind an opaque closure —
  first version of this row tried that, an index-closure abstraction, and
  it measurably vectorized WORSE, see "one wrong turn" below) — the same
  slice-based technique `reduce_width_binary_monomorphic` already uses
  (ROW 4), which is what lets LLVM see the fixed relationship between
  loop bound and slice length needed to elide bounds checks and pack the
  chunk into vector ops. Each `DOT_LANES`-wide chunk updates all
  `DOT_LANES` lanes of a `[f32; DOT_LANES]` array via `reduce(lane,
  op(a, b))`; only after every chunk is consumed does a single
  `DOT_LANES`-wide horizontal combine (`lanes[0]` folded against
  `lanes[1..]`) produce the scalar. `fold.len < DOT_LANES` falls back to
  `dot_fold_scalar_binary`/`dot_fold_scalar_unary` — the untouched
  pre-ROW-12 strict fold, byte-for-byte, so a tiny `k` never reassociates
  at all (not enough terms for lanes to pay for themselves).
- `seeded == false` (`ReduceInit::FirstElement`) seeds each lane from its
  own first chunk instead of `fold.init`, so no lane ever combines with a
  non-identity `fold.init` — same discipline the old code used for its
  single accumulator.
- Scope cut, named not silently narrowed: only the `(true, true)` binary
  arm and the unary contiguous branch were converted. `(true, false)`/
  `(false, true)`/`(false, false)` (one or both operands a stride-0
  broadcast) and the unary non-contiguous branch are untouched — those
  arms either have no real memory traffic to vectorize (broadcast case)
  or aren't exercised by any GEMM shape this session measures; converting
  them is mechanical repetition of the same pattern, not attempted here.

**One wrong turn, recorded not hidden:** the first implementation used a
single generic `dot_fold_multi_accumulator<R>(len, fold, term: impl
FnMut(usize) -> f32, reduce)` shared by both the unary and binary call
sites, with `term` closing over `slice[index]` reads. Built, gated green
(130/130, 146/146, 25/25, 1/1, clippy clean), and measured a real win
(0.882-0.933s -> 0.352-0.354s at `DOT_LANES=4`, 0.337-0.349s at
`DOT_LANES=8` — a genuine ~2.5x). Disassembly of an isolated,
same-flags probe of that exact shape
(`scratchpad/opt/lanesprobe.rs`/`lanesprobe_bin_disasm.txt`) showed the
main 8-wide loop was **not vectorized at all** — 8 independent *scalar*
`s`-register accumulators (ILP via superscalar issue, no NEON `.4s`
anywhere in that loop) — because `slice[index]` behind an opaque
`FnMut(usize) -> f32` hides the length relationship between `len` and
the slice from LLVM, so it could not elide the bounds check or prove the
chunk-of-`DOT_LANES` shape needed to vectorize. Rewritten to the
`chunks_exact`-based, slice-native form above (still one wrong turn's
worth of the 90-minute budget spent, named here rather than deleted from
the record). The rewrite measured a further ~2.5x on top of the first
version's win (below) — kept.

**Timing (direct `Instant`, `examples/profile_hot.rs`, release build,
1024^3 GEMM, this session's host, `[n, k]` transposed-RHS layout, the
`reduction_fast_path`/`reduce_dot_binary` arm ROW 10/11 established):**

| version | runs | range | vs ROW 11 baseline (0.886-0.891s, re-measured this session via `git stash`) |
|---|---|---|---|
| ROW 11 (before, re-measured this session) | 0.890s, 0.891s, 0.886s | 0.886-0.891s | 1.0x |
| index-closure, `DOT_LANES=4` | 0.353s, 0.354s, 0.353s, 0.354s, 0.354s | 0.353-0.354s | 2.50x-2.52x |
| index-closure, `DOT_LANES=8` | 0.338s, 0.337s, 0.349s, 0.340s, 0.337s | 0.337-0.349s | 2.54x-2.64x |
| **chunks_exact, `DOT_LANES=8` (KEPT)** | 0.133s, 0.131s, 0.134s, 0.131s, 0.130s | 0.130-0.134s | **6.61x-6.85x** |

`[k, n]` (untransposed, `reduce_width_fast` path, untouched by this row):
0.099-0.116s across every run above — unchanged from ROW 11 within noise,
confirming this row is orthogonal to the width-based fast path. ns/MAC
at the end of this row (`[n,k]` layout): 1.073741824e9 / 0.130-0.134s =
**0.121-0.125 ns/MAC** (down from ROW 11's 0.822-0.869 ns/MAC — the
combined ROW 10-12 improvement on this layout is 52x (ROW 10 baseline,
5.342s) / 0.130-0.134s = **39.9x-41.1x**; ROW 10's original layout-alone
gap (`[n,k]`/`[k,n]`) of 52x is now 0.134/0.116 = **1.16x** at worst
case, effectively closed for this shape/size).

**4-vs-8 lanes decision:** measured head-to-head on the index-closure
version (both scalar-ILP, no SIMD, so the comparison isolates the
lane-count effect from the vectorization-shape effect): 8 lanes
consistently faster (0.337-0.349s vs 0.352-0.354s, ranges barely
overlap, ~4% mean improvement) — kept 8. Not independently re-swept on
the final `chunks_exact` version (budget); the mechanism (more
independent partial sums hide more of the reduce chain's latency, up to
the point pressure on physical vector registers reverses the trend) is
expected to still favor 8 over 4 for the same reason, named as an
assumption carried from the index-closure measurement, not re-verified
on the final shape.

**Correctness — max relative error, MEASURED (`cargo test --release -p
proxima-tensor --lib -- fused_matmul --nocapture`, on the exact
`chunks_exact`/`DOT_LANES=8` binary):**

| k | data | max relative error | tolerance |
|---|---|---|---|
| 7 | sequential small integers (0..35) | **0** | 1e-5 |
| 1024 | `sin(i*0.0137)` / `cos(i*0.0271)`, m=8,n=8 | **3.8956164e-6** | 1e-5 |

k=7's zero error is explained, not just observed: small integers this
size sum exactly in f32 regardless of grouping (no rounding occurs at
all below 2^24), so reassociation is a no-op numerically at this size —
matches the task's own prediction to measure rather than assume. k=1024
shows real, nonzero rounding difference from reassociation, comfortably
inside the 1e-5 relative tolerance the task specified.

**Assertions weakened, recorded explicitly (the task's own requirement):**
- `fused_matmul_with_transposed_rhs_matches_a_naive_triple_loop`
  (`proxima-tensor/src/cpu.rs`): `assert_eq!(evaluated.root(),
  naive_matmul(...))` (bit-exact) -> `assert_all_close(evaluated.root(),
  &reference, 1e-5)` (1e-5 relative tolerance). Reason, in the test's own
  comment and here: `reduce_dot_binary_monomorphic`'s `(true, true)` arm
  now reassociates the sum via `DOT_LANES` partial accumulators, matching
  Accelerate/OpenBLAS/ggml practice — bit-exactness against the naive
  triple loop is no longer the correctness bar for this path. Measured
  max relative error at this test's k=7 size: 0 (see table above).
- New test added (not a weakening, a new tolerance-based case covering
  the size that actually exercises rounding):
  `fused_matmul_with_transposed_rhs_k1024_within_tolerance_of_a_naive_triple_loop`,
  1e-5 relative tolerance, measured 3.8956164e-6.
- `fused_matmul_matches_a_naive_triple_loop` (untransposed `[k,n]`
  layout, uses `reduce_width_fast`, untouched by this row): left as
  `assert_eq!` — correct, that path was not changed.

**Disassembly evidence.** Full binary: `objdump -d` on the release
`profile_hot` binary (`scratchpad/opt/row12_v2_full_disasm.txt`, 151,332
lines). The `reduce_dot_binary` symbol (all 32 op x reduce_op combos
LTO-inlined into one symbol, same pattern every prior row saw) spans
19,329 lines (`scratchpad/opt/row12_v2_reduce_dot_binary_slice.txt`):
**148 `fmul.4s`/`fmul.2s`, 132 `fadd.4s`/`fadd.2s`, 0 `fmla`, 326 paired
128-bit loads (`ldp q`)** — up from ROW 11's 90/90/0/384 in the same
symbol (fewer paired loads because `chunks_exact` unrolling changed the
load shape, more vector arithmetic ops).

Isolating the exact `(op=Multiply, reduce_op=Add)` instantiation (the
GEMM-relevant one) required a same-source, same-flags standalone probe
(`scratchpad/opt/lanesprobe3.rs`, `-C opt-level=3 -C lto=fat -C
codegen-units=1`, `#[inline(never)]` to keep it a distinct symbol) rather
than guessing from address proximity in the 32-combo dump — confirmed
numerically first (`dot_multiply_add(a, b) = 714.7797`, matching a
plain naive-loop probe's `714.7797` at this test input's display
precision) before trusting its disassembly as representative. Its hot
loop (`scratchpad/opt/lanesprobe3_fn.txt:170-207`, addresses
`0x100000ae8`-`0x100000b7c`) processes 16 elements/iteration (LLVM
unrolled the `DOT_LANES=8` chunk loop by 2x):

```
100000ae8: ad7f0961    ldp q1, q2, [x11, #-0x20]      ; a, 4x unrolled loads
100000aec: acc21163    ldp q3, q4, [x11], #0x40
100000af0: ad7f1985    ldp q5, q6, [x12, #-0x20]      ; b
100000af4: acc24187    ldp q7, q16, [x12], #0x40
100000af8: 6e25dc21    fmul.4s v1, v1, v5             ; 4-wide multiply x4
100000b08: 6e26dc42    fmul.4s v2, v2, v6
100000b18: 6e27dc63    fmul.4s v3, v3, v7
100000b28: 6e30dc84    fmul.4s v4, v4, v16
                                                        ; then 16 scalar fadd s0,s0,sN
                                                        ; combining all 16 lanes into one
                                                        ; running s0 EVERY iteration
100000b78: f10041ad    subs x13, x13, #0x10
100000b7c: 54fffb61    b.ne 0x100000ae8
```

**Mechanism, stated plainly, including the part that did NOT go as
designed (V6, named not glossed).** The multiply side IS genuine 4-wide
SIMD (4x `fmul.4s`, processing 16 elements' worth of products per
iteration in parallel). The accumulate side is **not** what the source
describes: the source carries 8 independent lanes across iterations,
combined once at the very end; the compiled code instead re-collapses
all 16 of an iteration's products into the single running scalar `s0`
**every iteration**, via 16 scalar `fadd`s in a lane order that is NOT
sequential array order (interleaved: `s1, s18, s17, s5, s2, s20, s19,
s6, ...`) — LLVM is legally reassociating these particular 16 adds
because, within one iteration, they are 16 independent SSA values (not
one serial chain), so combining them in any order is sound regardless of
float non-associativity; it simply chose to do that combining every
iteration rather than deferring it to the end as the source's structure
suggested it might. Net effect: the horizontal-reduce **operation count**
per element is unchanged from ROW 11 (summing N values always costs
N-1 adds, however grouped) — the measured 2.5x-6.7x speedup is
therefore attributable to (1) the genuine 4-wide `fmul.4s` on the
multiply side (real, ROW-4-style SIMD), (2) 4x fewer loop iterations
(16-wide unroll vs ROW 11's 4-wide), cutting loop-control/bounds-check
overhead per element, and (3) better instruction-level parallelism in
the 16-add combine (the 16 lane-extracts are mutually independent and
can issue back-to-back, unlike ROW 11's tighter one-fmul-then-immediate-
horizontal-reduce-then-next-fmul shape) — not the "defer horizontal
combine to once per whole call" mechanism this row's design intended.
Confirmed by opening the exact arm's disassembly, not inferred from the
timing win. A DIFFERENT reduce_op combination (`op=Add`, `reduce_op=Minimum`,
found while searching `scratchpad/opt/row12_v2_reduce_dot_binary_slice.txt:7794-7801`)
DOES show the intended shape — 8 consecutive `fadd.4s` (one per lane,
no interleaved scalar extraction) feeding a `fminnmv`-style tree combine
only at loop exit — so the "defer to one combine" outcome the design
intended IS achievable by this source pattern, just not the specific
compiler decision LLVM made for the `Multiply`/`Add` combo on this
build. Not re-attempted with a restructuring (e.g. two separate
4-lane `[f32; 4]` accumulator arrays instead of one `[f32; 8]`, to see if
that changes LLVM's unroll/schedule choice) — named as the natural next
lever, not attempted this row given the budget and the win already
measured is large and real regardless of the exact instruction shape.

**KEPT.** Correctness, exact binary the timing/error numbers above came
from: `cargo nextest run -p proxima-tensor` -> **130/130** (129 + this
row's new k=1024 tolerance test), `--features config` -> **146/146**
(145 + 1), `cargo nextest run -p omega --features metal` (real Metal
device) -> **25/25**, `cargo test -p proxima-tensor --doc` -> **1/1**.
`cargo clippy -p proxima-tensor --lib --tests --features config -- -D
warnings` -> clean, 0 warnings.

**Allocation count:** not measured this row — `dot_fold_multi_accumulator_binary`/
`_unary` allocate nothing (`lanes: [f32; DOT_LANES]` is a stack array,
`chunks_exact` allocates nothing), consistent with every prior
reduce_dot row.

---

## ROW 13 — home-turf arm vs ggml, first run against the optimized executor

Gate point 13. Every prior ggml ratio in this log predates ROWs 1-12 and
was stale by ~100x. This is the first `bench_vs_ggml` row_f run against
the current binary. MEASURED, criterion, aarch64 M-series, contended box
(see the control note).

| size | ggml t1 | ours 1-thread | ggml t8 | ours w8 |
|---|---|---|---|---|
| 512^3  | 8.09 ms | 17.09 ms (2.11x) | 1.27 ms | 2.84 ms (2.23x) |
| 1024^3 | 71.3 ms | 134.5 ms (1.89x) | 9.39 ms | 20.29 ms (2.16x) |
| 2048^3 | 803.6 ms | arm absent | 154.1 ms | 284.4 ms (1.85x) |

design-favors: **incumbent** — ggml's `mul_mat` takes a `[n,k]` right-hand
operand and row_f builds exactly that, so this engages their layout, not
ours.

30.1 vs 15.97 GFLOPS at 1024^3. Reconciles with `profile_hot`'s 0.124
ns/MAC on the `[n,k]` layout (0.133s ~ 134 ms), so the two harnesses
agree. **Corrects an earlier claim in this session of a ~1.5x gap** — that
figure came from the `[k,n]` layout, which ggml's arm does not use.

**Host was not quiet.** ggml's own arms, which no change of ours touches,
drifted between runs: 2048 t8 154.05 -> 89.62 ms (1.72x), 1024 t1 71.32 ->
68.34 ms (4.2%). Only the 512 t1 arm held still (8.0936 -> 8.0929 ms,
0.01%), so it is the only arm that can carry a cross-run claim.

## ROW 14 — explicit fused multiply-add at both MAC sites, + accumulator hoist

**Mechanism first.** ROW 12 recorded `fmla.4s` = 0 with `fmul.4s` = 467
and `fadd.4s` = 458 and read it as a vectorization shortfall. It is not:
**Rust never contracts `a * b + c` into an FMA**, because contraction
rounds once instead of twice and so changes the result. It is a semantics
guarantee, not a missed optimization. `f32::mul_add` is the explicit
request and lowers to `fmla.4s` on aarch64.

Three changes, in one row because two are the same mechanism at two call
sites:

1. `dot_fold_fused_multiply_add` — `DOT_LANES` lanes via `mul_add`, taken
   in `reduce_dot_binary` for `(op=Multiply, reduce_op=Add)` with both
   operands contiguous along k. The `[n,k]` layout's path.
2. The same specialization in `reduce_width_binary` for all three
   contiguity combos. The `[k,n]` layout's axpy inner loop.
3. `accumulator` hoisted out of the `leading_flat` loop (`vec!` per output
   row -> one per bound op, refilled with `fill`).

Guarded by `FUSED_MULTIPLY_ADD` = `cfg!(target_arch = "aarch64") ||
cfg!(target_feature = "fma")`. Without hardware FMA `mul_add` becomes a
libm call and is far slower than the two-op form. **This is a structural
axis (principle 8) sitting in a `cfg` where it does not belong** — it
should resolve in the build-time profile alongside lane width and unroll
factor. Recorded as gate-15 debt, not fixed here.

**Disassembly, our symbols only** (`awk` scoped to `proxima_tensor`, since
the bench binary statically links ggml which is full of `fmla`):
`reduce_dot_binary` now contains two consecutive `fmla.4s` at +0x2d8/+0x2dc
(`v0`,`v1` accumulators against `v2/v3`,`v4/v5`) — 8 lanes as 2x 4-wide
FMA, one loop body. Read the arm, did not infer it from the timing.

**Numbers.** `profile_hot`, 1024^3, MEASURED:

| layout | before | after | delta |
|---|---|---|---|
| `[k,n]` (width path) | 0.108s | 0.102s | 1.06x |
| `[n,k]` (dot path)   | 0.133s | 0.127s | 1.05x |

Allocations during `evaluate()`: 1107 -> **85**.

row_f 512 t1, against a control arm that drifted 0.01%: 17.094 -> 15.249 ms,
**1.12x**. The 1024 and 2048 arms drifted 4.2% and 72% on the incumbent
side and carry no claim.

**KEPT.** `cargo nextest run -p proxima-tensor` -> **130/130**, 0 skipped.

**The result is the size of the win, not the win.** Halving the
multiply-accumulate instruction count bought 1.05-1.06x. If the kernel
were issue-bound it would approach 2x. It is **load-bound**: each 8-lane
block issues 2 `fmla.4s` against 2x32 B of loads, and the single-column
fold re-streams the invariant operand's entire k-slice once per output
column, so an `m x n` output reads it `n` times. That is the register-
blocking case, now measured rather than asserted, and it is the next
lever.

**Method error worth recording:** the width-site specialization (change 2)
was benched against row_f, which builds the `[n,k]` layout and therefore
never executes the width path. Both arms moved ~3-4% together and the row
initially read "no signal" — the arm did not exercise the code under
change. The N==0 failure in a different costume. Attribution above comes
from `profile_hot`, which builds both layouts.

## ROW 15 — the `instrument` feature gated zero lines (defect, found 2026-08-17)

`proxima-tensor/Cargo.toml:30` declares

```
instrument = ["std", "dep:proxima-telemetry", "proxima-telemetry/macros"]
```

with a six-line docstring describing "real execution-witness counters
(elements/bytes/gather-vs-affine) plus per-bound-op spans", and asserting
that "counters are incremented in a local accumulator inside the loop and
committed ONCE per bound-op call (never per element), so the feature does
not perturb the thing it measures."

`grep -rn instrument src/` returns **0 lines**. None of it exists.
`cargo build --features instrument` succeeds and counts nothing.

This is the N==0 defect inside the feature whose entire purpose is to
prevent it, and it is why ROWs 10-14 are all timers and opcodes: twelve
rows of optimization work carrying zero execution witness. Every
"mechanism" in this log up to here was read off a disassembly or inferred
from a delta, never counted.

The manifest docstring was written as if the code existed. Treat a
feature's prose as a claim requiring the same evidence as any other —
`grep` the gate before believing it.

Counters now being built (loads-per-MAC and the re-read factor are the two
quantities that decide whether the GEMM kernel is load-bound). Until they
land, the load-bound reading of ROW 14 is a **hypothesis supported by the
size of the FMA win**, not a measurement.

## ROW 16 — degenerate control: the working-set sweep refutes DRAM-bandwidth

Run on data already collected in ROW 13 plus machine facts — no new bench.

Machine (MEASURED, `sysctl`): Apple M1 Max, 8 performance cores, L1d
131072 B per core, L2 **12582912 B (12 MB)** shared across the perf
cluster. Clock 3.228 GHz is **ASSUMED** — `sysctl` exposes only the 24 MHz
timebase, not the core clock.

Working set, f32, A+B+C (DERIVED from the shapes):
512^3 = 3.1 MB (L2-resident) · 1024^3 = 12.6 MB (spills L2) · 2048^3 = 50 MB.

ns/MAC, single thread (MEASURED, ROW 13 timings / DERIVED MAC counts):

| size | ours | ggml |
|---|---|---|
| 512^3  | 0.1274 | 0.0603 |
| 1024^3 | 0.1253 | 0.0664 |

**Our cost per MAC is flat across a 4x working-set change that crosses the
L2 boundary** — 1.7% *lower* at the larger size. A DRAM-bandwidth-bound
kernel cannot do that. The "load-bound" reading of ROW 14 is refuted **at
the DRAM level**; whether it holds at L1 is still open and is what the
counters must decide.

**The trend has the opposite cause to the one it looks like.** The ratio
narrows 2.11x -> 1.89x from 512^3 to 1024^3 not because we improve but
because **ggml degrades 10% per MAC** when its working set leaves L2. We
are flat. Reading the narrowing as our gap closing with size is wrong.

Cycles/MAC at the assumed 3.228 GHz: ggml **0.214**, ours **0.404**. One
`fmla.4s` retires 4 MACs, so one FMA per cycle is 0.25 cycles/MAC — ggml
is at or under the vector-FMA issue ceiling and is issue-saturated; we
issue roughly one 4-wide FMA every 1.6 cycles.

**So the 2x is an issue-rate gap, not a memory wall.** What remains
unexplained, and what the counters must separate: loads per MAC in L1
versus per-iteration overhead (address arithmetic, loop trip counts,
accumulator dependency). Those two produce the same wall-clock and this
row cannot tell them apart.

## ROW 17 — the mechanism: ggml's f32 GEMM is tinyBLAS, and it is 0.42 loads/FMA

The explanation ROWs 13-16 lacked. Found by opening the incumbent's
kernel instead of ours — every prior row read our own disassembly.

**ggml's f32 `mul_mat` never calls `ggml_vec_dot_f32`.** `ggml-cpu.c:1240`
dispatches contiguous src1 to `llamafile_sgemm` — tinyBLAS,
`llamafile/sgemm.cpp`, 3594 lines. The f32 microkernel is `gemm_bloc<RM,RN>`
(sgemm.cpp:396), instantiated at RM=4, RN=6 (`mnpack<4, 6, 4>`,
sgemm.cpp:349):

```c
D Cv[RN][RM] = {};                                 // RM*RN vector accumulators
for (l = 0; l < k; l += KN) {
    V Av[RM];
    for (i) Av[i] = load(A + lda*(ii+i) + l);      // RM loads
    for (j) { V Bv = load(B + ldb*(jj+j) + l);     // 1 load
        for (i) Cv[j][i] = madd(Av[i], Bv, Cv[j][i]); }
}
```

A **rectangular outer-product tile**: RM+RN loads buy RM*RN fused
multiply-adds, and the RM*RN accumulators stay register-resident across
the entire k loop.

| | loads per FMA | live accumulator registers |
|---|---|---|
| tinyBLAS RM=4 RN=6 | **0.42** | 24 |
| ours | **2.0** | 2 |

Clock-independent, and it predicts everything measured:

- The 1.89x gap. With ~2-3 load slots and 4 FMA pipes per cycle, 2.0
  loads/FMA caps throughput near 1 FMA/cycle regardless of FMA capacity.
  0.404 vs 0.214 cycles/MAC (ROW 16) is that cap.
- **Why ROW 14's FMA bought only 1.06x.** We were never short of
  arithmetic. We are short of load slots. Halving instruction count on the
  non-binding resource cannot help.
- **Why the first blocking attempt lost 57%** (agent report, `[n,k]`
  0.133s -> 0.209-0.210s, 3 runs each, ~1ms spread). That design kept the
  dot-product form and ran four of them sharing ONE operand: ~1.25
  loads/FMA, barely moved, and the per-column strided reindexing
  (`base_b + column * column_stride + offset`) destroyed the contiguous
  `chunks_exact` read the single-column fold depends on. **Reuse has to
  come from a rectangular tile reusing BOTH operands, not from sharing
  one.** Correctness held throughout (`root[0]=18370` identical), so the
  gate and block math were right and only the shape was wrong.

Also register-resident depth: we hold 8 floats (2 vector registers) of 32
available. tinyBLAS holds 24. A tile that spills is worthless, so the
next attempt must count `str q` spills, not just `fmla`.

**Dispatch error on my side, recorded:** both agent worktrees were created
`-b <branch>` off HEAD while ROW 14's FMA work was **uncommitted**, so the
first agent baselined and built pre-FMA code. Its report correctly stated
`FUSED_MULTIPLY_ADD` does not exist and `fmla.4s` = 0 — true of its base,
not of the working tree. Its negative stands on its own terms but on the
wrong base. Subsequent dispatches carry the diff as an explicit
`git apply` step.

## ROW 18 — counters land; 2.000000 loads/MAC MEASURED (closes ROW 15)

`instrument` feature now gates real code (`proxima-tensor/src/instrument.rs`,
new; call-site accumulation in `run_reduce` and `run_elementwise`,
committed once per bound op via `KernelCounters::commit`). Worktree
`scratchpad/inst-wt`, branch `perf/execution-counters`, uncommitted.

1024^3, single thread, **N = 1 bound op per program** (asserted, not assumed):

| | `[k,n]` width_fast | `[n,k]` dot_fast (ggml's layout) |
|---|---|---|
| mac_ops | 1,073,741,824 | 1,073,741,824 |
| operand_loads | 1,074,790,400 | 2,147,483,648 |
| **loads / MAC** | **1.000977** | **2.000000** |
| distinct_operand_elements | 2,097,152 | 2,097,152 |
| **re-read factor** | 512.5 | **1024.0 = K exactly** |
| output_writes | 1,048,576 | 1,048,576 |
| leading_iters / kernel_calls | 1024 / 1,048,576 | 1024 / 1,048,576 |

**2.000000 loads per MAC, every operand element reloaded exactly K times.**
MEASURED at the load site. Confirms ROW 17's 2.0 figure, which was read off
the disassembly, and confirms the re-streaming mechanism by counting it
rather than inferring it. tinyBLAS is at 0.42.

Perturbation check: without `instrument` 0.106/0.106/0.106 and
0.138/0.139/0.139; with it 0.107/0.107/0.107 and 0.139/0.139/0.140. <1%,
inside run-to-run spread — the commit-once-per-bound-op granularity held.
`nextest -p proxima-tensor` **130/130** both with and without the feature;
`--no-default-features --features alloc` still clean (counters are
`std`+`instrument` gated, no leak into the alloc tier); clippy
`--all-targets --features instrument` clean after fixing 5 pre-existing
`expect_used` errors in `examples/profile_hot.rs` in place.

**Limit on the claim, and it matters.** Our own two layouts differ 2.0x in
loads/MAC but only **1.31x in wall time** (139 vs 106 ms). Loads/MAC is a
real limiter and not the only one — a per-call cost (accumulator dependency
chain, `DotFold` lane structure, loop overhead) is unapportioned, and
`leading_iters`/`kernel_calls` are identical across the two layouts here
(m=n=k makes the run symmetric) so this benchmark cannot separate
per-call-count from per-call-work.

Following it through: our better layout at 1.0 loads/MAC still runs
0.0987 ns/MAC vs ggml's 0.0664. So driving the dot path to 1.0 loads/MAC
predicts ~1.5x remaining, **not parity**. The rectangular tile (ROW 17)
goes further, to 0.42, and is the right lever — but the counters do not
support a claim that it closes the gap, and none is made.

## ROW 19 — outer-product tile in plain arrays: 3x SLOWER, 737 spills. ROLLED BACK

Second blocking negative. Worktree `scratchpad/tile-wt`, branch
`perf/outer-product-tile`, tile implemented then reverted.

Design: `gemm_tile_fma`, TILE_ROWS=4 x TILE_COLS=4 x width 4, accumulators
declared `[[[f32; 4]; 4]; 4]`, applicability gate resolved once per bound
op (all six conditions + `leading_extents.len() == 1`), `leading_flat`
stepped by TILE_ROWS with row/column remainders falling back to the
existing per-slot path.

| | gemm `[k,n]` 1024^3 | gemm_rhs_transposed `[n,k]` 1024^3 |
|---|---|---|
| before (patch only) | 0.103-0.111s | 0.128-0.141s |
| tile wired in | 0.099-0.111s | **0.394-0.407s** |
| after rollback | 0.100s | 0.124s |

**~3x regression** on the exact path it targets — worse than ROW 17's
57%. `[k,n]` unaffected, as expected, since it never enters
`reduction_fast_path`.

Correctness held: `root[0]=18370` unchanged, 130/130, clippy clean. The
applicability gate and the tile arithmetic were right.

**Disassembly says why**, and it is not the loop shape:

| | fmla.4s | fmul.4s | fadd.4s | `str q` spills |
|---|---|---|---|---|
| pre-FMA baseline | 0 | 467 | 458 | — |
| tile | 17 | 467 | 458 | **737** |

`fmul.4s`/`fadd.4s` sat at the pre-FMA baseline and 737 register-spill
stores appeared. The tile's arithmetic never vectorized and the
accumulator never reached registers.

**Root cause, and it is a coordinator design error, not an execution
error.** In Rust `[[[f32; 4]; 4]; 4]` indexed by loop variables is
**memory**; LLVM did not scalarize it into registers. tinyBLAS declares
`D Cv[RN][RM]` where `D` is `float32x4_t` — a *native vector type* that
maps onto a NEON register directly (ROW 17, sgemm.cpp:396). I ported the
loop structure and not the type that makes it work, so the tile paid
stack spill traffic on top of un-vectorized scalar arithmetic instead of
the intended reuse.

**The lesson generalizes past this kernel:** reading an incumbent's loop
nest is not reading its kernel. The register allocation IS the design, and
it lives in the accumulator's *type*, not in the loop structure. Two
attempts were spent on shapes derived from the loop nest alone.

Next attempt uses `core::arch::aarch64::{float32x4_t, vfmaq_f32,
vld1q_f32, vaddvq_f32}` directly. `unsafe` is sanctioned here by principle
11's SIMD axis ("hand-rolled SIMD is the next step ONLY when the mature
library's shape doesn't fit" — nothing fits this) and has precedent in
`proxima-core` (`arch.rs`, `ring/mpsc.rs`, `sync/blocking/futex.rs`);
`proxima-tensor` currently carries zero and there is no `forbid(unsafe)`
anywhere in the crate or workspace manifests.

## ROW 20 — NEON tile lands: 0.122s -> 0.028s. The mechanism was never loads

Worktree `scratchpad/neon-wt`, branch `perf/neon-tile`, uncommitted.

`gemm_tile_neon` — tinyBLAS's `gemm_bloc` ported with the accumulators as
a genuine 2D array of `float32x4_t` (`core::arch::aarch64`), 4x4 tile,
aarch64-gated, wired into `run_reduce`'s `reduction_fast_path` with the
same six-condition gate, `leading_flat` stepped by 4, remainders falling
back unchanged.

| | before | after |
|---|---|---|
| `gemm_rhs_transposed` `[n,k]` 1024^3 | 0.123/0.123/0.122 s | **0.028/0.028/0.029 s** |
| `gemm` `[k,n]` 1024^3 (untouched path) | 0.099-0.108 s | 0.099-0.108 s |

**4.33x.** 0.1141 -> **0.0264 ns/MAC**.

**Execution asserted, not assumed** — gate passes **1**, tile invocations
**65536** (256 row-tiles x 256 col-tiles), fallback elements **0**. Both
earlier negatives were accepted without this; had either fallen back
silently we would have measured the old kernel and learned nothing.

Correctness: `root[0]=18370` bit-identical, **130/130**, clippy clean.
Spills: **0 `str q`** in the kernel body, against attempt 2's 737. The
tile now sits with the dot kernel (0 spills) rather than the width kernel
(261 `str q` + 520 `stp q`).

### The mechanism, and both prior models are refuted

Measured pure-register FMA throughput on this box (no loads or stores in
the loop, N independent `float32x4_t` accumulators):

| accumulators | G vector-FMA/s | ns/MAC ceiling |
|---|---|---|
| 1 | 0.769 | 0.325 |
| 2 | 1.533 | 0.163 |
| 4 | 3.070 | 0.081 |
| 8 | 6.113 | 0.041 |
| 16 | 11.92 | **0.021** |
| 24 | 12.15 | 0.021 |

One accumulator measures FMA *latency* (~4.2 cycles); throughput scales
linearly to 16 chains and saturates there.

| model | predicted at the tile point | measured | error |
|---|---|---|---|
| A loads, power law `t = 0.0987 r^0.391` | 0.0753 | 0.0264 | **+186%** |
| B loads, linear `t = 0.0680 + 0.0307 r` | 0.0834 | 0.0264 | **+216%** |
| C accumulator chains | 16 chains -> 0.0206 ceiling | 0.0264 | **1.28x ceiling** |

**It was FMA latency hiding, not load count.** The old dot kernel had ~2
independent accumulator chains and ran at 0.129 against a 2-chain ceiling
of 0.163 — it was pinned there. The tile supplies 16 chains and lands at
1.28x the 16-chain ceiling. The load reduction 2.0 -> ~0.5 rode along as a
minor term.

Model C was measured **before** the confirming experiment, so this is a
prediction, not a story fitted afterwards. A and B were both exact fits to
our two data points and both wrong by ~3x out of sample — two points
cannot choose between models, and the out-of-sample hit that made A look
good (5.6% on ggml) was coincidence.

**Retracted with it:** ROW 18's framing that loads/MAC "decides" the gap,
and the claim that ggml sits at the hardware FMA ceiling. ggml runs at
0.0664 against a measured ceiling of 0.0206 — it is **3.2x off peak**, so
parity was never the maximum.

### Not claimed

Our 0.028 s and ggml's 0.071 s are from **different runs**, and ggml's own
arms drifted 72% between two earlier runs on this box (ROW 13). No
head-to-head claim until one harness produces both in one run; that is
running. Also untested: the parallel arm, and whether the bench's GEMM
program even routes through the tile (its operand layout may differ from
`profile_hot`'s).

The `[k,n]` width path is untouched at ~0.10 s and now has ~4x the cost
per MAC of the tiled path. Same treatment applies to it.

## ROW 21 — RETRACTION: every ggml number was against the WRONG KERNEL

`scratchpad/ggml/build/CMakeCache.txt:466` — **`GGML_LLAMAFILE:BOOL=OFF`**.

`llamafile/sgemm.cpp` (tinyBLAS) was never compiled. `nm -gU
build/src/libggml-cpu.a | grep llamafile_sgemm` returns nothing. No
`*sgemm*` object exists anywhere in the build tree.

ggml's f32 `mul_mat` reaches tinyBLAS only inside `#if GGML_USE_LLAMAFILE`
(`ggml-cpu.c:1240`). With the flag off, control falls through to
`ggml_vec_dot_f32`. **Every ggml measurement in this log — ROW 13's "we
are 1.89x slower", today's "we are 2.29x faster" — was taken against
ggml's naive fallback, not against the kernel ggml actually ships.**

### What this retracts

- **ROW 17 in full.** Its mechanism — tinyBLAS's 0.42 loads/FMA, the 4x6
  tile, 24 register-resident accumulators — describes code **that was
  never in the binary being benchmarked**. I read `sgemm.cpp`, verified
  the dispatch site, and never checked whether the `#if` around it was
  satisfied. Reading code inside a disabled preprocessor branch as if it
  were live is the same class of error as reading a commit list instead of
  a diff.
- **The premise the entire optimization effort ran on.** "We are 2x behind
  ggml" was 2x behind a fallback.
- **Every ratio in ROWs 13, 16, 18 and 20.** The proxima-side numbers in
  those rows stand — they are ours, measured in isolation. Only the
  comparisons die.

### What survives

Our kernel work is independent of ggml and is unaffected: 0.122s ->
0.028s at 1024^3, the FMA-accumulator-chain mechanism (ROW 20), and the
pure-register FMA sweep, which was measured on this machine with no ggml
involvement at all. The size sweep and the counters likewise stand.

### The tell I ignored for hours

ggml's 1024^3 t1 = 68.86 ms is **31 GFLOPS**. One M1 Max performance core
has a NEON ceiling near 100 GFLOPS and the pure-FMA sweep in ROW 20
measured 97 GFLOPS achievable. A mature tuned library at 31% of peak
should have been the first thing questioned. I only questioned it once the
numbers started favouring us — which is the bias the whole discipline
exists to prevent, and it cost the session its baseline.

### Gate 13 amendment this earns

"Home-turf arm" is not satisfied by naming the incumbent and calling their
API. **Verify the incumbent's optimized path is compiled in and taken at
runtime**, by symbol (`nm`) and by build flag, before recording a single
comparative number. An incumbent built with its fast kernel disabled is a
straw man, and it produces a favourable number that looks exactly like a
real one.

Rebuild with `-DGGML_LLAMAFILE=ON` and full re-bench in flight. The
expected outcome is that we are behind again.

## ROW 22 — width path tiled too; both kernels confirm the chain mechanism

Worktree `scratchpad/width-wt`, branch `perf/width-tile`, uncommitted.

`gemm_width_tile_neon`, `WIDTH_TILE_ROWS=4` x `WIDTH_TILE_VECS=4` = 16
`float32x4_t` accumulators. For `[k,n]`, B is contiguous along n and A is
invariant across the width dim, so it vectorizes along n and broadcasts A
via `vfmaq_n_f32` (scalar multiplier) — a different intrinsic from the dot
tile's `vfmaq_f32`, same accumulator count.

| path | before | after | ns/MAC | vs 16-chain ceiling (0.0206) |
|---|---|---|---|---|
| `[n,k]` dot   | 0.122 s | **0.028 s** | 0.0264 | 1.28x |
| `[k,n]` width | 0.101 s | **0.032 s** | 0.0298 | 1.45x |

Two kernels, two intrinsics, one mechanism, both landing inside 1.5x of
the measured chain-count ceiling. Neither confirmation depends on ggml —
which matters, because ROW 21 retracted every ggml comparison.

Loop body: 16 `fmla.4s`, **0 `str`/`stp q`**, accumulators resident in
v2-v7/v16-v25 across the whole k reduction. Counters: 1 gate pass, 16384
invocations, 0 fallback. 130/130, `root[0]=18370` unchanged, clippy clean.
(The brief said 4096 invocations; correct is 256 row-tiles x 64 col-tiles
= 16384. My arithmetic, agent's catch.)

## ROW 23 — the tile computes a correct full output; my test oracle did not

Adversarial full-output check at m=n=k in {64, 257, 260, 1024}, every
element compared.

**Coverage identity `invocations*16 + fallback == m*n` holds EXACTLY at
all four sizes**, including 257 where the remainder path fires
(`fallback_elements=513`). No output position is skipped. This is the
check neither earlier negative had, and it is the one that would have
caught a tile that "won" by not computing.

`operand_loads / mac_ops = 0.5000` exactly with the tile active — matches
the design's 8 loads per 16 FMAs and confirms the tile is on the measured
path, not bypassed.

**Two coordinator errors, both caught by the agent:**

1. **260 does not force the remainder path.** 260 = 4 x 65. All three
   sizes I chose were multiples of the tile dimension, so all three
   reported `fallback_elements=0` and none exercised the path I
   specifically wanted tested. The agent added 257.
2. **My oracle was less accurate than the thing under test.** The tests
   compared the tile against a naive f32 triple loop at 1e-4 relative
   tolerance and 4 "failed". Against an f64-accumulated ground truth the
   tile's RMS error is **0.25-0.37x the naive f32 loop's own RMS error** —
   the tile's 4-wide FMA lanes do pairwise summation, which has lower
   error growth than a sequential sum. The failures were two
   differently-rounded f32 answers disagreeing near zero crossings, and
   the reference was the worse of the two.

Max absolute error 1e-4 to 2e-4 across all sizes, consistent with f32
accumulation over k up to 1024. Mismatches scattered, not clustered at
edges — a broken remainder path would cluster.

**Generalizes:** a naive implementation is not automatically a valid
oracle. When the optimized form changes summation order it can be
*more* accurate, and then equality-within-tolerance against the naive form
tests nothing but agreement between two roundings. The invariant that is
actually true — and falsifiable in the right direction — is *error against
a higher-precision ground truth must be no worse than the naive form's*.
Test rewrite in flight.

`--features instrument` compiles and runs for the first time (ROW 15/21
defect closed): 134 tests, alloc tier clean, no telemetry leak.

## ROW 24 — the REAL home-turf number: ggml rebuilt with tinyBLAS

Rebuild proof (not inferred from the symbol alone): `nm -gU
ggml-tinyblas/build/src/libggml-cpu.a | grep llamafile_sgemm` ->
`T _llamafile_sgemm`; `sgemm.cpp.o` present in the build tree; and every
ggml arm moved 2.7-3.2x faster than the llamafile-off build. A runtime
decline (`src1_cont` false) would have left the numbers unchanged — they
did not, so tinyBLAS both compiled AND was taken.

The llama.cpp checkout is a vendored subtree, not a standalone project;
configuring it top-level forces `GGML_STANDALONE=ON` unconditionally. The
agent built it through a 5-line wrapper `add_subdirectory()` project, which
is how llama.cpp itself consumes it. Only intended flag diff vs the old
cache: `GGML_LLAMAFILE=ON`.

| size | ggml OLD (naive) t1 | ggml NEW (tinyBLAS) t1 | speedup | new GFLOPS |
|---|---|---|---|---|
| 512  | 8.1523 ms | 2.9822 ms | 2.73x | 90.0 |
| 1024 | 68.859 ms | 23.655 ms | 2.91x | 90.8 |
| 2048 | 661.02 ms | 209.68 ms | 3.15x | 81.9 |

### Where we actually stand

| size | ours ST | ggml ST | | ours w8 | ggml t8 | |
|---|---|---|---|---|---|---|
| 512  | 3.870 ms | 2.982 ms | **1.30x slower** | 1.691 ms | 0.718 ms | **2.36x slower** |
| 1024 | 29.734 ms | 23.655 ms | **1.26x slower** | 7.601 ms | 3.762 ms | **2.02x slower** |
| 2048 | harness-skipped | 209.68 ms | — | 63.837 ms | 35.177 ms | **1.81x slower** |

Against the 97 GFLOPS this box actually sustains (ROW 20's pure-register
FMA sweep): **ggml 90.8 = 94% of achievable; ours 72.2 = 74%.**

### What today's work was worth, measurable only now

Against tinyBLAS, this morning's kernel (134.5 ms at 1024^3) was **5.69x
behind**. It is now **1.26x behind**. The tiles closed ~4.4x of a 5.7x gap.
Every intermediate claim in ROWs 13-20 about our position was void; the
kernel work behind it was not.

### The remaining single-thread 1.26x is NOT accumulator count

tinyBLAS runs 4x6 = 24 accumulators to our 4x4 = 16, but ROW 20's sweep
measured 11.92 vs 12.15 G vector-FMA/s between 16 and 24 chains — **2%**.
Chain count is saturated at 16 and cannot explain 26%.

The untested difference is **cache blocking**: `mnpack` adapts tile shape
(4x6 / 4x3 / smaller) and blocks over BM/BN, while our tile streams B in
full for every row-tile with no blocking at all. Named as the hypothesis,
not measured.

### The parallel arm is now the larger gap

2.02x at 1024^3. ggml scales 23.655 -> 3.762 = **6.29x** on 8 threads; we
scale 29.734 -> 7.601 = **3.91x**. Under investigation.

## ROW 25 — parallel scaling: only M is split, so every worker re-reads all of B

Worker sweep, transposed-RHS GEMM, median of 3 (worktree `scale-wt`):

| workers | 1024^3 | speedup | 2048^3 | speedup |
|---|---|---|---|---|
| `evaluate` | 29.116 ms | — | 284.297 ms | — |
| 1 | 29.152 | 1.00x | 285.204 | 1.00x |
| 2 | 16.572 | 1.76x | 167.135 | 1.71x |
| 4 | 12.436 | 2.34x | 85.370 | 3.34x |
| 8 | 7.306 | **3.99x** | 46.705 | **6.11x** |

Ruled out by measurement, not argument:
- **load imbalance** — `BoundOp::split(8)` yields 8 chunks of exactly 131072 elements, spread 0
- **tile fallback at chunk edges** — `fallback_elements = 0` at both 1 and 8 workers (1024/8 = 128 rows/chunk is itself a multiple of `TILE_ROWS`)
- **fixed parallel overhead** — `evaluate` vs `evaluate_parallel(1)` differ by 0.13% at 1024^3

**Best supported: only the leading (M) axis is ever split.** `split_axis()`
returns `output_axes.first()` (`bind.rs:282-292`) and `rebase_operands`
(`bind.rs:333-350`) shifts only the split axis's base, so the RHS layout is
untouched and **every worker streams the entire K x N operand**. Redundant
RHS bytes scale as `workers x N x K`, independent of M, while compute
scales as `M x N x K` — so the ratio improves with problem size, which is
exactly the 3.99x -> 6.11x the sweep shows.

**The decomposition this gives.** At ggml's 6.29x scaling we would be at
29.734/6.29 = **4.73 ms vs their 3.762 = 1.26x** — identical to the
single-thread gap. So the parallel deficit is *entirely* scaling, and
fixing it collapses two problems into one. The fix is 2D partitioning
(split N as well as M): with Wm x Wn workers, traffic falls from
`W*K*N + M*K` to `Wn*M*K + Wm*K*N`.

Two residuals, named not buried:
- `NEON_TILE_FALLBACK_ELEMENTS` (`cpu.rs:1023`) counts only the **column**
  tail; a row tail falls through an untracked scalar path, so the coverage
  identity is silently wrong for m not a multiple of 4. Being fixed in the
  integration.
- workers=1 never enters `thread::scope` (`split` returns `None` below 2
  parts), so **per-call OS thread spawn — fresh threads every call, no
  pool — is untested**, not ruled out.

## ROW 26 — test oracle corrected; 134/134

`tests/neon_tile_full_output.rs` now asserts what is actually true:
coverage identity; tile RMS error vs f64 ground truth **<=** naive f32's
own RMS error; max absolute error under `1e-6 * k`.

| size | coverage | tile RMS | naive RMS | ratio | max abs | bound |
|---|---|---|---|---|---|---|
| 64 | 4096 == 4096 | 1.307e-6 | 3.506e-6 | 0.373 | 8.03e-6 | 6.4e-5 |
| 257 (fallback=513) | 66049 == 66049 | 4.771e-6 | 1.654e-5 | 0.288 | 2.78e-5 | 2.57e-4 |
| 260 | 67600 == 67600 | 4.626e-6 | 1.667e-5 | 0.278 | 2.79e-5 | 2.6e-4 |
| 1024 | 1048576 == 1048576 | 7.783e-6 | 3.089e-5 | 0.252 | 4.90e-5 | 1.024e-3 |

134 passed / 0 failed, with and without `--features instrument`. No bound
was loosened; headroom 9-21x at every size.

## ROW 27 — the remaining 1.26x is NOT cache blocking (hypothesis withdrawn)

ROW 24 named cache blocking as the likely cause. Checked against numbers
already in hand, it does not carry:

- traffic: ours 0.5 loads/MAC = 2.15 GB at 1024^3; tinyBLAS 0.42 = 1.80 GB.
  **19% more**, and the traffic-to-time relationship is sublinear.
- instruction density: ours 64 MACs / 24 instructions = 2.67; theirs 96/34
  = 2.82. **5.6% apart.**

Neither is 26%. Against the pure-arithmetic floor (22.1 ms for 1024^3 at
the measured 12.15 G vector-FMA/s), tinyBLAS carries **7% overhead** and we
carry **35%**. The question is not traffic volume; it is why our kernel
carries 5x their non-arithmetic overhead, and **no mechanism is currently
supported**. Isolation microbenchmark in flight — the same standalone-probe
method that settled ROW 20, which is the only thing that has actually
worked when a model was in doubt.

## ROW 28 — the kernel is clean; 77% of the overhead is TRAVERSAL

Standalone probe (`scratchpad/tileprobe/main.rs`), `gemm_tile_neon` copied
verbatim, rustc `-O -C target-cpu=native`, no cargo, QoS-pinned, 3 trials
per arm with matched sustained-load windows.

| arm | G vector-FMA/s | ns/MAC |
|---|---|---|
| 1 — one L1-resident tile, k=1024 | 11.66-11.70 | 0.0214 |
| 2 — same, k=16 | 4.82-4.85 | 0.0517 |
| 3 — full traversal 512^3 | 8.34-8.62 | 0.0290-0.0300 |
| 4 — full traversal 1024^3 | 9.29-9.31 | 0.0268-0.0269 |

**Decomposition of the 34.5% total overhead** (29.73 ms measured vs the
22.1 ms pure-arithmetic floor):

| component | percentage points | share |
|---|---|---|
| kernel-intrinsic (arm1 vs 12.15 ceiling) | 3.96 | 11.5% |
| **traversal (arm4 vs arm1)** | **26.6** | **77%** |
| production vs standalone harness | 3.95 | 11.5% |
| sum | 34.5 | matches measured |

**The kernel is clean.** On one resident tile it runs within 3.7% of the
machine's register-only ceiling, and that shortfall is itself explained:
arm 2 isolates a fixed **7.7-7.8 ns per tile call** (accumulator zeroing,
16x `vaddvq_f32`, output read-modify-write), which at k=1024 is ~2.2% of
call time. Three quarters of the overhead appears only when the tile is
driven across a real M x N grid.

**This partly reverses ROW 27.** I withdrew the blocking hypothesis on
*traffic-volume* grounds — 19% more bytes cannot explain 26% more time.
That reasoning was sound and the conclusion was still wrong, because
traversal cost is not volume: it is reuse distance and access order.
tinyBLAS blocks over BM/BN (`sgemm.cpp` `gemm<RM,RN,BM>`); we do not block
at all. Withdrawing a hypothesis for a correct reason that does not
address the actual mechanism is its own failure mode.

**Anomaly kept as an anomaly.** Arm 3 (512^3, ~3 MB, L2-resident) is
**slower** per FMA than arm 4 (1024^3, ~12.6 MB, L2-overflowing) —
8.48 vs 9.30, reproducible across runs with matched load windows, checked
against and not explained by a DVFS ramp. A smaller working set measuring
slower contradicts simple L2-capacity reasoning. Nothing in the harness
isolates prefetcher warm-up, reuse distance, or traversal-order effects, so
"traversal" stays a bucket the data cannot split further. Not attributed.

**Residual, named not counted:** the 3.95-point production-vs-harness gap.
`run_reduce` does per-tile bookkeeping the harness omits — atomic
`fetch_add` on the invocation counters, `unflatten_into`,
`fill_running_offsets`, output-stride computation, tail-column fallback.
Plausible, uninstrumented, and deliberately excluded from the attributed
total.

Blocked-traversal sweep in flight — the lever gets proven in isolation
before production is touched, which is the ordering that has worked every
time today and whose absence produced two 3x regressions.

## ROW 29 — blocked traversal recovers NOTHING. Hypothesis dead.

Same standalone probe, kernel held constant, only tile-traversal loop
order varied. 1024^3, 3 trials per arm.

| arm | GFMA/s |
|---|---|
| control (current prod order) | 9.22-9.26 |
| N-blocked BN=64 / 128 / 256 / 512 | 9.13-9.18 / 9.23-9.29 / 9.26-9.28 / 9.27-9.30 |
| MN-blocked 64x64 / 128x128 / 256x256 | 9.05-9.18 / 9.18-9.28 / 9.06-9.31 |
| K-split BK=128 / 256 / 512 | 6.71-6.77 / 7.85-7.94 / 8.70-8.71 |

Every N- and MN-blocked arm is **statistically tied** with the unblocked
baseline. Nominal best (BN=512, mean 9.285) beats control (9.249) by 0.4%
— inside control's own 0.45% trial spread. Fraction of the 2.43 GFMA/s
kernel-ceiling gap recovered: **~1.5%, i.e. zero.**

K-split is reliably **worse** and worsens as BK shrinks (6% / 15% / 27%
below baseline) — the expected direction for adding an output
read-modify-write pass per chunk and re-deriving tile addresses.

**ROW 28's redirect is itself refuted.** I said traversal cost is reuse
distance and named ggml's BM/BN blocking as the fix. Blocking, at every
size swept, in both one and two dimensions, does nothing. That is now two
consecutive wrong hypotheses about the same 26 percentage points —
first traffic volume, then reuse distance — and both were argued from
structure rather than measured before being asserted.

**The 512^3 inversion survives blocking**: best config gives 8.43-8.50 at
512^3 vs 9.27-9.30 at 1024^3, the same ~9% inversion the unblocked
baseline showed. Whatever causes a smaller working set to run slower, it
is not traversal order.

**Methodology finding worth keeping.** The first combined run — all arms
back-to-back in one process — showed monotonic within-config degradation
from `k_split BK=256` onward (6.42 -> 4.73 -> 3.07 GFMA/s across three
trials, thermal throttling) which then corrupted the following control
arm (6.66 / 4.77 / 6.52). Re-running each arm as its own process with
cooldowns produced stable trials. **A long sweep in one process silently
manufactures a winner**: whichever arm runs while the box is cool. Every
future sweep on this box runs one arm per process invocation.

Next: pure load-bandwidth control with no arithmetic. At 9.25 GFMA/s and
0.5 loads/MAC we pull ~74 GB/s. If sustained bandwidth at 4-16 MB working
sets lands near that, we are bandwidth-bound and tile *width* is the only
lever (4x4 = 0.5 loads/MAC; tinyBLAS's 4x6 = 0.4167, 17% fewer). If
bandwidth is far above 74 GB/s the wider-tile hypothesis dies too, and
that refutation is the more useful result. Control first, sweep only if
the control supports it — the ordering these two dead hypotheses earned.

## ROW 30 — integration: both tiles on one branch, no regression

`scratchpad/merge-wt`, branch `perf/tensor-neon-integrated`, uncommitted.
Base from `verify-wt` (dot tile + counters + tests); width tile hand-ported
from `width-wt`; `examples/scaling.rs` from `scale-wt`; the
`neon_tile_counters()` bench delta from `neon-wt`. All five source
worktrees left intact as the record.

The two `run_reduce` branches are mutually exclusive by construction
(`reduction_fast_path = !fast_path && ...`), so `try_run_width_tile` is an
early-exit before the dot tile's plan resolution — no shared mutable state,
no ordering dependency.

| gate | result |
|---|---|
| `nextest -p proxima-tensor` | **134/134** |
| `nextest --features instrument` | **134/134** |
| `clippy --all-targets --features instrument -- -D warnings` | clean |
| `check --no-default-features --features alloc` | clean |
| `profile_hot` | `gemm` 0.032 s, `gemm_rhs_transposed` 0.029 s |

Both within noise of their source branches (0.032 / 0.028). No regression.

Coverage identity exact at 1023 and 1025 — sizes forcing row AND column
tails — for **both** tiles independently:
dot 1023: 65025x16 + 6129 = 1046529 = m*n · width 1023: 16065x64 + 18369 = 1046529
dot 1025: 65536x16 + 2049 = 1050625 = m*n · width 1025: 16384x64 + 2049 = 1050625

### A bug I invented and a bug that was real

**Invented:** I briefed this agent to fix a row-tail counting gap in
`NEON_TILE_FALLBACK_ELEMENTS`. It checked both source worktrees and the
row-tail `fetch_add` was already present in both. ROW 25 reported the gap;
I repeated it as a premise without opening the file. **A subagent's report
is not a read** — the rule I have applied to our code all day, violated on
a report about our code.

**Real, and found instead:** `try_run_width_tile`'s early `return Ok(())`
skipped `run_reduce`'s entire `KernelCounters::commit`, so
`--features instrument` reported `operand_loads=0 mac_ops=0` for the width
path while the tile ran correctly. A counter reading zero and a code path
not executing are indistinguishable — the N==0 defect, now inside the
instrumentation built to prevent it.

### Counter units are inconsistent between the two paths

Settled by arithmetic, per k-step per tile (64 MACs each):

| path | geometry | scalar-element units | instruction units | reported |
|---|---|---|---|---|
| dot | 4x4, 8 `vld1q` = 32 floats | **0.5000** | 0.1250 | 0.5000 |
| width | 4x16, 4 `vld1q` + 4 scalars = 20 floats | **0.3125** | 0.1250 | 0.1250 |

**The dot counter counts scalar elements; the width counter counts load
instructions.** The width path's true loads/MAC is **0.3125**, not 0.125.
Any cross-path comparison drawn from the raw counters is wrong by 2.5x
until the units are unified.

And the corrected figure is itself evidence: the width tile has **better**
operand reuse than the dot tile (0.3125 vs 0.5) — better even than
tinyBLAS's 0.4167 — while running **slower** (0.032 vs 0.029 s). A third
independent observation that loads/MAC does not drive time on this box,
after the traffic-volume and reuse-distance hypotheses both died.

## ROW 31 — bandwidth-bound, CONFIRMED; and loads/MAC does not pick a tile shape

### Arm A — pure load bandwidth, no arithmetic

8 independent `float32x4_t` accumulators fed by unrolled `vld1q_f32`
(`ldp q,q` in the disassembly, verified not elided), one process per size,
5 s cooldowns.

| working set | GB/s |
|---|---|
| 256 KB | 81.1-81.2 |
| 4 MB | 80.9-81.1 |
| **12 MB (the real 1024^3 A+B+C)** | **66.1-69.4** |
| 16 MB | 58.7-60.0 |
| 64 MB | 53.5-57.0 |

The brief named 4 MB and 16 MB; the agent noticed they straddle rather
than bracket the 74 GB/s implied rate and **added the 12 MB point that
matches the real working set** on its own initiative. That is the number
that decides it.

Our traversal implies ~74 GB/s at 9.25 GFMA/s and 0.5 loads/MAC; sustained
bandwidth there is 66-69. **We are at the machine's sustained load
bandwidth.** That is the 26-point traversal bucket from ROW 28, and it
explains ROW 29: you cannot reorder your way out of a bandwidth wall, only
move fewer bytes.

### Arm B — tile shape sweep, 1024^3, one process per shape

| shape | accumulators | loads/MAC (DERIVED) | GFMA/s | `str q` | `stp q` |
|---|---|---|---|---|---|
| 4x4 (current) | 16 | 0.5000 | 9.41-9.45 | 0 | 18 |
| **6x4** | 24 | 0.4167 | **10.73-10.75** | **0** | 27 |
| 5x5 | 25 | 0.4000 | 10.36-10.53 | 2 | 27 |
| 4x6 (ggml's shape) | 24 | 0.4167 | 6.19-6.26 | 1 | 27 |
| 8x4 | 32 | 0.3750 | 6.22-6.27 | 7 | 38 |
| 4x8 | 32 | 0.3750 | 2.26-2.29 | 7 | 55 |
| 6x6 | 36 | 0.3333 | 2.15-2.18 | 22 | 45 |

**loads/MAC does not predict throughput.** 4x6 and 6x4 have the *same*
accumulator count and the *same* derived loads/MAC and differ by **74%**.
ggml's own orientation, ported verbatim into our kernel and traversal, is
**42% worse than the 4x4 we currently ship**. Copying the incumbent's
constants is not the same as copying its performance — the third time
today that reading their source produced a wrong expectation about our
binary (after ROW 17's disabled `#if` and ROW 29's blocking).

Spilling, not the load ratio, drives the collapse: `objdump` on 6x6 shows
`stp q`/`ldr q` round-tripping accumulators through the stack **inside**
the per-k-step loop. Spills rise 16->18, 24-25->27, 32->38-55, 36->45.

**Unexplained and flagged as such:** the 4x6 vs 6x4 asymmetry at equal
accumulator count, equal loads/MAC and near-equal spills. The agent named
tile-orientation-vs-row-major-stride as a candidate and explicitly declined
to assert it. It stands unexplained.

**Prediction under test:** 6x4 in production, 9.43 -> 10.74 GFMA/s, i.e.
29.73 -> ~26.1 ms at 1024^3, single-thread gap 1.26x -> ~1.10x. Recorded
before the result, and 6x4's zero spills are the gate — if production
spills, the shape did not fail, the implementation did.

## ROW 32 — 2D partition REGRESSED, rolled back. The plumbing, not the theory.

Worktree `scratchpad/split2d-wt`, branch `perf/split-2d`, implemented,
measured, reverted per brief.

Design: `BoundOp::split_grid(parts)` — for a `Keep::Reduce` with two output
axes, choose `Wm x Wn == parts` minimising `Wn*M*K + Wm*K*N` by exhaustive
divisor-pair search, tie-broken toward chunk dims divisible by 4.
Factorisations chosen: 2->1x2, 3->1x3, 4->2x2, 6->2x3, **8->2x4**.

| workers | 1024^3 before | after | 2048^3 before | after |
|---|---|---|---|---|
| 1 | 29.228 ms (1.00x) | 29.157 (1.00x) | 277.996 (1.00x) | 281.050 (1.00x) |
| 2 | 18.365 (1.59x) | 16.717 (1.74x) | 158.824 (1.75x) | 170.028 (1.65x) |
| 4 | 11.544 (2.53x) | 11.785 (2.47x) | 83.876 (3.31x) | 92.675 (3.03x) |
| **8** | **7.893 (3.70x)** | **12.195 (2.39x)** | **47.012 (5.91x)** | **64.461 (4.36x)** |

Rerun of the 8-worker after-case: **17.138 ms (1.83x)** — worse still, so
the regression is reproducible, against a before-sweep with 7.87-8.22 ms
spread.

**The theory held; the plumbing killed it.** Operand traffic did fall as
predicted (9MK -> 6MK in units of M=N=K). But a 2D sub-rectangle is not
contiguous in a row-major buffer, so the implementation gave each cell a
fresh zero-based `out_layout` plus its own `vec![0.0f32; row_len*col_len]`
**allocated serially before the threads spawned**, then a
**single-threaded row-by-row copy-back** after the join. Two extra
full-output serial passes, both outside the parallel region — Amdahl,
added in order to fix a scaling problem.

**What is NOT refuted:** 2D partitioning itself. What is refuted is
materialising cells into private buffers. Real BLAS writes in place with
strided addressing precisely to avoid this.

**The strongest result in the attempt, and why a retry is warranted:**
output was **bit-identical to `evaluate` at workers 2/3/4/6/8 at both
256^3 and 1024^3, zero mismatches**. The partitioning arithmetic is
correct and is not what failed. Tile counters were also unchanged
(`gate_passes=8 invocations=65536 fallback=0` both before and after), so
no chunk was pushed off the tile path.

Retry in flight: cells write **directly into the parent output** via a raw
pointer per worker, disjointness by construction — the `Wm x Wn` cells
partition the output exactly, so no two workers address the same element,
which the bit-identical check above already demonstrated empirically. No
per-cell allocation, no copy-back, nothing serial added. If that regresses
too, the operand-traffic theory is wrong rather than the plumbing, and
that is worth knowing.

## ROW 33 — 6x4 tile lands: +8.7% (predicted +13.9%), shortfall explained

Worktree `scratchpad/tile6x4-wt`, detached at `847f20c5` with `merge-wt`'s
files copied in. One-const change: `TILE_ROWS 4 -> 6`. `gemm_tile_neon`,
the tiled loop, `out_prefixes` and the row-remainder loop were already
generic over the consts.

| path | before | after |
|---|---|---|
| `gemm_rhs_transposed` (dot, changed) | 0.029/0.029/0.030 s | **0.027/0.027/0.027 s** |
| `gemm` (width, untouched) | 0.030/0.032/0.031 s | 0.031/0.030/0.030 s |

9.15 -> **9.94 G vector-FMA/s, +8.7%.** Probe predicted 9.43 -> 10.74,
+13.9%. Direction and rough magnitude held; 5.2 points short.

Gates: 134/134 with and without `--features instrument`; clippy
`--all-targets --features instrument` clean; alloc tier clean. Coverage
identity exact at 1023 (43350x24 + 6129 = 1046529) and 1025 (43520x24 +
6145 = 1050625).

**Spills: 0 `str q` / 0 `stp q` inside the k-loop.** The agent noted 24
`str q` immediately *after* the loop back-branch and correctly identified
them as the one-time final-accumulator store, not spilling — the exact
misreading that would have triggered a false rollback.

**The shortfall is structural and quantified: `1024 mod 6 = 4`.** Four
leftover rows per node drop to the scalar remainder path —
`fallback_elements = 4096`, and `operand_loads/mac_ops` measured **0.4229**
against the pure-tile **0.4167**. The probe chose sizes that tile exactly
and never paid it.

So 6x4 buys better loads/MAC at the cost of not dividing power-of-two
problems, which is the shape every benchmark uses. **Four leftover rows is
exactly one 4x4 tile block, not a scalar case** — being recovered with a
second, 4-row tile instantiation rather than a wider scalar path.

Single-thread standing after this: ~29.73 -> ~27.5 ms at 1024^3 against
ggml's 23.66 = **~1.16x behind**, from 1.26x.

## ROW 34 — "bandwidth-bound" is a property of ONE kernel, not the workload

Arithmetic on figures already in hand, 1024^3:

| path | loads/MAC | bytes | time | achieved bandwidth |
|---|---|---|---|---|
| dot (4x4, `vfmaq_f32`) | 0.5000 | 2.15 GB | 0.029 s | **74.1 GB/s** |
| width (4x16, `vfmaq_n_f32`) | 0.3125 | 1.34 GB | 0.032 s | **41.9 GB/s** |

Sustained bandwidth at the 12 MB working set is 66-69 GB/s (ROW 31). The
dot path is **at the wall**; the width path is at **61% of it**, moving
**38% fewer bytes** and running **10% slower**.

**I had been generalising the dot path's limiter to both kernels.** The
width path is not bandwidth-bound and has ~1.6x of headroom — at the wall
it would run 19.4 ms rather than 32.

Three candidate causes, and they are separable: `vfmaq_n_f32`
(vector x scalar broadcast) versus `vfmaq_f32` (vector x vector); 4 scalar
loads + 4 vector loads per k-step versus 8 vector loads; and 4x16 versus
4x4 geometry. Probe in flight measuring each in isolation, same
pure-register method that settled ROW 20.

## ROW 35 — the width kernel is EXONERATED; the deficit is entirely traversal

Three isolation arms, standalone, one process each with cooldowns.

**Arm 1 — `vfmaq_n_f32` pure-register sweep**, verbatim structural port of
the `vfmaq_f32` sweep that produced ROW 20's ceiling:

| accumulators | `vfmaq_n_f32` | `vfmaq_f32` reference |
|---|---|---|
| 1 | 0.751-0.771 | 0.769 |
| 2 | 1.517-1.523 | 1.533 |
| 4 | 3.033-3.067 | 3.070 |
| 8 | 6.027-6.065 | 6.113 |
| 16 | 11.92-12.03 | 11.92 |
| 24 | 12.14-12.61 | 12.15 |

Curve for curve. **The intrinsic is not the cap.**

**Arm 2 — mixed scalar/vector load pressure**, equal FMA count per call,
L1-resident: dot shape (8 `vld1q` + 16 `vfmaq_f32`) **10.68-11.78**; width
shape (4 scalar + 4 `vld1q` + 16 `vfmaq_n_f32`) **11.55-11.80**. Overlapping,
width marginally higher. **The load mix is not the cap.**

**Arm 3 — the width kernel body itself**, L1-resident, k=1024, same
single-tile protocol the dot kernel scored 11.68 on: **11.89-11.93**.
**It exceeds the dot kernel.** The width kernel is the fastest thing in the
crate when its data is in L1.

**So the entire 32 ms vs ~19.4 ms deficit is traversal.** The agent
declined to name a mechanism its arms could not reach, and explicitly said
the report template's anticipated answer ("arm 3 falls short") did not fit
because arm 3 did not fall short. Refusing the offered conclusion when the
data contradicts it is the behaviour that has caught the most errors today.

### The structural difference, stated as a hypothesis under test

`[k,n]`: B is contiguous along **n**, the tile iterates **k**, so
consecutive k-steps read B addresses `n*4` = 4096 bytes apart — 64 useful
bytes per 4 KB page, ~1024 pages walked per column-tile pass.
`[n,k]`: both operands contiguous along k, each tile reading ~10 contiguous
4 KB runs.

A genuine tension, not a bug: **contiguous B access requires a wide
accumulator; a register-resident accumulator requires strided B access.**
Which implies the fix is not in the kernel — transpose B once (O(k*n))
and use the dot kernel (O(m*n*k) of compute), which is precisely why ggml
requires `src1` contiguous and works in the `[n,k]` convention.

Under test now, transpose cost included and broken out separately, at
1024^3 and 2048^3. A refutation is worth more than a confirmation here:
eight hypotheses have been tested today and five have died.

## ROW 36 — row-remainder tile lands; single-thread 1.105x; parallel MEASURED UNDER CONTENTION (invalid)

**Row remainder** (`rowrem-wt`): `gemm_tile_neon` made const-generic over
ROWS; after the 6-row pass, `>= 4` leftover rows run one 4x4 tile pass, 0..=3
stay scalar. Separate counter so the identity stays checkable.

- `operand_loads/mac_ops` **0.4229 -> 0.4170** (pure-tile target 0.4167) —
  ~95% of that specific gap closed
- `fallback_elements` at 1024^3 -> **0**
- coverage identity `main*24 + rem*16 + fallback == m*n` exact at 1023 /
  1024 / 1025
- 0 `str`/`stp q` inside BOTH loop bodies
- 134/134 with and without `--features instrument`, clippy + alloc tier clean

**Head-to-head, ggml built WITH tinyBLAS (`_llamafile_sgemm` confirmed by `nm`):**

| 1024^3 | ours | ggml | ratio |
|---|---|---|---|
| single-thread | 26.97 ms | 24.41 ms | **1.105x slower** (was 1.26x) |
| 8-wide | 18.50 ms | 4.34 ms | 4.26x (was 2.02x) |

**The parallel row is inadmissible and the cause is mine.** I dispatched an
8-thread benchmark while **five agents** were building and benching on the
same box. Evidence it is contention, not the kernel: our parallel interval
spans 26% ([16.568, 18.498, 21.461]); our own 8-worker speedup measured
1.46x against 3.99x earlier; the 2048^3 ggml control — code nobody touched —
moved **+20.19%** with a 13% CoV. A single-thread arm needs one core and
survives load; an 8-thread arm needs the machine. The 1024^3 single-thread
control drifted only +3.18%, so that row stands.

`/disciplined-component` says pin the host loadout and isolate when noisy. I
enforced that in every agent brief today and did not apply it to my own
dispatch scheduling. **Parallel numbers require a quiet box; that is a
scheduling constraint on the coordinator, not a note in a log.**

**Kernel identity verified, not assumed.** The agent decomposed the
cumulative counters into 147 single-threaded call-units (43,520 inv / 4,096
fallback) plus 566 parallel ones (43,008 / 16,384, 8 gate-passes each) and
checked `147 + 566*8 = 4,675` against measured `gate_passes` — exact. Fallback
totals at 512/1024/2048 are integer multiples of the 6-row remainder unit,
impossible under a 4x4 kernel where all three sizes divide evenly and
fallback would be identically 0.

## ROW 37 — CORRECTNESS: the NEON tile is wrong on non-divisible chunk extents

Mine, written today. Not "pre-existing" — the word is banned in AGENTS.md
for exactly this reason and I used it because I repeated a subagent's
framing (it meant "predates *my* change") without reading what it said.

`evaluate_parallel` output diverges from `evaluate` at **workers 3 and 6**:
624/65536 and 652/65536 at 256^3; 3073/1048576 and 3309/1048576 at 1024^3.
1024/3 and 1024/6 do not divide by 4; /2, /4, /8 do.

The reporting agent did the isolation properly rather than blaming its own
change: same mismatches against the untouched 1D baseline, and a
contiguous-RHS GEMM that never engages the NEON tile is bit-identical at the
same worker counts. That localises it to **tile row/column-tail handling
when a chunk's extent is not a multiple of the tile dimension.**

**It survived because every correctness check I specified ran
single-threaded.** ROW 23's full-output verification covered sizes 64/257/
260/1024 on one thread. Nothing ever compared the chunked path against the
serial one until an agent doing unrelated perf work happened to check.

**Consequence: every parallel measurement at workers 3 and 6 was timing a
kernel producing wrong output.**

A second agent reported bit-identical at 2/3/4/6/8 on what should be the
same baseline — a direct contradiction. Neither is accepted. Diagnosis in
flight across all three tile generations at sizes 256, 1024 and **1026**
(divides by 6 and 2, not 4 — a case neither prior agent tried), with
first-mismatch (row, col) required so tail-clustering is visible rather
than inferred.

## ROW 38 — the bench was configured for SPEED, not accuracy. Every number today inherits it.

`benches/bench_vs_ggml.rs:1104-1105`:

```rust
.sample_size(10)                                 // criterion's MINIMUM; default 100
.measurement_time(Duration::from_millis(500))    // default 5s
```

Iterations actually obtained per arm:

| size | per-op | iters in the 500 ms window |
|---|---|---|
| 512^3 | 3.4 ms | ~147 |
| 1024^3 | 27 ms | ~18 |
| 2048^3 | 252 ms | **~2** |

Ten samples, two iterations. That is why ggml's 2048^3 t1 came back with
**26.68% CI width** and why the whole row_f suite finishes in ten minutes.
**Every comparative number in ROWs 13, 24, 36 was measured this way.**

Raised to `sample_size(50)` / `measurement_time(10 s)` for the current run:
~1470 iterations at 512^3, ~370 at 1024^3, ~40 at 2048^3.

**A tight CI on four samples is not tight, it is unsampled**, and the two
are indistinguishable in criterion's output unless the iteration count is
reported alongside. Now required in the brief.

## ROW 39 — CoV discipline applied retroactively; 4 of 11 arms are UNUSABLE

CI width = `100*(upper-lower)/estimate` on the ROW 36 head-to-head:

| arm | CI width | verdict |
|---|---|---|
| ggml 1024 t1 | 0.29% | tight |
| ours 1024 single | 1.41% | tight |
| ggml 512 t1 | 0.16% | tight |
| ours 512 single | 0.58% | tight |
| ggml 1024 t8 | 3.51% | ok |
| ours 2048 w8 | 4.20% | ok |
| ours 512 w8 | 5.35% | noisy |
| **ggml 512 t8** | **10.13%** | **UNUSABLE** |
| **ggml 2048 t8** | **10.46%** | **UNUSABLE** |
| **ours 1024 w8** | **26.45%** | **UNUSABLE** |
| **ggml 2048 t1** | **26.68%** | **UNUSABLE** |

Ratios with propagated uncertainty (`dr = r*sqrt((da/a)^2+(db/b)^2)`):

- **1024^3 single-thread: 1.105 +/- 0.008 -> 9.7% to 11.3% behind.** Holds.
- **512^3 single-thread: 1.110 +/- 0.003 -> 10.7% to 11.3% behind.** Holds.
- 1024^3 parallel: 4.26 +/- 0.57 — inadmissible.

**Every parallel arm but one fails the 5% bar.** So every parallel claim
made today rests on measurements that do not support it — including the
"+20% ggml control drift" I flagged in ROW 36, which was itself computed
against a 26.68% arm.

## ROW 40 — the residual gap is SIZE-INVARIANT, which rules out bandwidth

Owner's observation, and it reframes the residual: 1.110x at 512^3 and
1.105x at 1024^3 — flat across a 4x working-set change. 512^3 is L2-resident
(81 GB/s available), 1024^3 spills L2 (66-69 GB/s). **A 20% swing in
available bandwidth with no movement in the gap means bandwidth is not what
sets it.** ROWs 31 and 34 established the dot kernel sits at the bandwidth
wall; that is true and is not the same thing as it explaining the residual.

Size-invariance is the signature of a fixed *fraction* of work on a slow
path. The arithmetic points at one candidate:

- 512 = 85x6 + **2** leftover rows -> 2/512 = **0.39%**
- 1024 = 170x6 + **4** leftover rows -> 4/1024 = **0.39%**

Identical fractions, which is exactly the invariance observed. The benched
branch (`tile6x4-wt`) has **no** row-remainder tile, so both push 0.39% of
rows through the scalar path. At ~28x slower per row that is ~11% — the
whole gap.

Corroboration already measured: `rowrem-wt` took 1024^3 from 0.030 -> 0.027 s
in its own before/after (**-10%**), which applied to the head-to-head's
26.97 ms lands at ~24.3 ms against ggml's 24.41 — parity.

**Discriminating prediction, recorded before the result:** 512 mod 6 = 2 is
*below* the 4-row remainder tile's threshold, so those rows stay scalar.
If the explanation is right, **1024^3 closes to ~parity while 512^3 stays
~1.11x behind.** A uniform improvement at both sizes would mean something
else changed and the explanation is wrong.

## ROW 41 — where we allocate that we should not: 12 per bound-op in run_reduce

Counted in `rowrem-wt/proxima-tensor/src/cpu.rs`, per **call**, not per program:

| function | allocations per call |
|---|---|
| `run_reduce` | **12** |
| `run_elementwise` | 5 |
| `run_scan` | 5 |
| `prepare` | 5 |
| `run_chunks_threaded` | 2 |
| `evaluate_parallel` | 0 |

The 12 in `run_reduce`, by what sizes them:

| site | sized by |
|---|---|
| `operand_values`, `step_values` | operand count / fused body depth |
| `reduction_dims`, `leading_extents`, `reduction_extents` | **rank** |
| `strides`, `running`, `gather_cursors` | operand count |
| `leading_coordinate`, `reduction_coordinate`, `full_coordinate` | **rank** |
| `accumulator` | **data width** |

**Eleven of twelve are sized by rank or operand count** — small, bounded,
known at bind time. They are stack arrays wearing a `Vec`. Only
`accumulator` scales with data, and that one belongs in caller-provided
scratch, which is the discipline `Interpreter` already applies to its
borrowed buffer table.

**Why the 85-allocations-per-GEMM figure hid this.** One GEMM is one bound
op: 12 allocations against 10^9 MACs, invisible. **A real model is thousands
of bound ops per forward pass** — 12 x N per inference, plus 5 per
elementwise and per scan. The metric was measured on the one workload that
cannot show the defect.

### Four principles converge on one change

- **§11 sans-IO** — hot path must be zero-alloc per operation, enforced by
  an alloc-counter test over a 100k-iteration loop. We have 12 and no test.
- **§3 tiers** — `cpu` is `#[cfg(feature = "std")]` (`lib.rs:174`); the
  module doc blames float transcendentals, but these 12 are the other
  blocker and nobody had counted them.
- **§12 no magic numbers** — fixed-capacity arrays *need* `MAX_RANK` /
  `MAX_OPERANDS`, and §12 says caps come from a per-crate sizing TOML +
  build.rs + `pub mod sized`. That is the same build-time axis logged as
  debt in ROW 14 and still unbuilt.
- **§11 FSM** — path selection is a cascade of `if fast_path / if
  reduction_fast_path / if tile / if row-remainder / if column-remainder /
  else scalar`, which is exactly the "deep call tree" §11 says to replace
  with one discriminated enum matched once.

no_std forces fixed caps; fixed caps force the build-time config; the config
is §12's contract. **Two constraints converging on one design is the signal
AGENTS.md names as evidence it is right** — here it is four.

Remaining std coupling: `std::thread::scope` in `evaluate_parallel`. The
core owns its threads, which sans-IO forbids; `proxima-runtime` is
no_std-capable and is the seam that should own them.

Audit in flight for the exact ledger, the true rank/operand bounds reachable
through `bind`, whether `libm` covers each transcendental, the terminal-path
count in the cascade, and the sizing axes with their current hardcoded
values.

## ROW 42 — tier audit: three of my claims corrected, and the doc overstates the code

### Corrections to ROW 41

**The operand bound is NOT arity.** `ScalarOp::arity()` (`op.rs:84-103`) tops
at 3, but `operand_values`/`strides`/`running`/`gather_cursors`/
`reduction_strides` are sized by `resolved.operands().len()` — the physical
leaf-operand count of a **fused** `ComposedBody`. `compose_operand`
(`bind.rs:727`) recurses through held elementwise ops with **no depth cap**,
so `a=x0+x1; b=a+x2; …` collapses to one body with N steps and N+1 operands.
True bound: **fusion-chain depth / program length**, not 3.

**There is no `MAX_RANK` anywhere.** Grepped `shape.rs`/`op.rs`/`bind.rs` —
nothing. `BoundOp::extents` is an unbounded `Vec<u64>`. So the "rank-bounded"
allocations are not fixed-cap *today* either; they become so only once a cap
exists and is enforced at `shape::infer`.

**`Interpreter::call` allocates per node** — `cpu.rs:663`,
`vec![0.0f32; node_output_len(&resolved)]`, data-sized, every call. Meanwhile
`cpu.rs:607-623` claims the buffer table is "caller-provided scratch… what
lets a caller run this against its own no-alloc scratch" and `cpu.rs:48-51`
claims `run_node_into` reaches "a no-alloc-at-the-write-site tier." The
*table* is borrowed; each slot's *contents* are allocated by `Interpreter`.
**I had been citing that doc as evidence of no-alloc readiness.** A
doc-comment is a claim and gets the same evidence bar as any other.

### The std blocker is smaller than the module doc says

Verified against the installed toolchain and the cached `libm` 0.2.16, with
file:line for each:

| fn | `core`? | `libm` |
|---|---|---|
| `exp` / `ln` / `tanh` | no | `expf` / `logf` / `tanhf` |
| `sqrt` | unstable only (`core_float_math`, `f32.rs:2101`) | `sqrtf`, plus an aarch64-NEON path |
| `abs` | **core-native `const fn`** (`f32.rs:1630`) | n/a |

Transcendentals are **"needs a crate not yet added,"** not "needs std." The
only genuine std dependencies in 4862 lines are `std::thread::scope`
(`cpu.rs:378`) and `std::panic::resume_unwind` (`cpu.rs:391`), both in the
parallel path.

### FSM: 6 terminal paths, gates resolved once, dispatch re-evaluated per row

Ten decision points (`cpu.rs:923, 934, 940, 981, 1014, 1045, 1108/1211,
1154, 1261, 1307`). Terminal kernels on aarch64: width-tile, 6x4 tile, 4x4
remainder tile, `reduce_dot_fast`, `reduce_width_fast`, generic gather —
**6**; on other targets 3.

The four gating booleans are computed **once per bound op**, as their docs
claim. But the *branches* on them re-execute finer: `cpu.rs:1261` per leading
row, `cpu.rs:1307` per reduction element. So the cascade is a shape problem
(one enum picked once) with a residual per-row/per-element dispatch cost.

### N=0 confirmed mechanically

`cargo check --no-default-features --features alloc` → exit **0**, and the
log shows only `proxima-build`/`proxima-core`/`proxima-primitives` plus the
non-executor modules. `lib.rs:174` gates `pub mod cpu;` behind
`feature = "std"`, so **0 of 4862 executor lines were handed to rustc.**
The green proves the plumbing, not the executor — exactly the trap ROW 15
named, still live.

### Sizing axes that would populate a `proxima-tensor.toml` (none exists)

`TILE_ROWS=6` (`cpu.rs:2358`) · `TILE_COLS=4` (`:2360`) ·
`ROW_REMAINDER_TILE_ROWS=4` (`:2369`) · `WIDTH_TILE_ROWS=4` (`:1933`) ·
`WIDTH_TILE_VECS=4` (`:1940`, doc calls it "the measured saturation point
for this core") · `PARALLEL_THRESHOLD=4096` (`:297`) · **MAX_RANK — absent**
· **MAX_OPERANDS / MAX_FUSION_DEPTH — absent**.

### Four decisions, named not designed

1. How to bound fusion depth / operand count — cap and reject, or cap
   operands and bail with a `TensorError`. Both are behaviour changes.
2. What `MAX_RANK` is. Picking a number constrains every program the crate
   can express; it is an API constraint, not a default.
3. Whether `Interpreter::call`'s per-node allocation moves to the caller
   (a real `Interpreter::new` API change) or the type stays alloc-tier-only.
4. What replaces `evaluate_parallel` below alloc. `std::thread::scope` has no
   no_std substitute; prime's reactor is the workspace-internal candidate.

**Smallest change for the alloc tier** (in flight): swap the four math calls
to optional `libm`, and move the std gate from the whole module inward to
`evaluate_parallel`/`run_chunks_threaded` alone. Nothing else in the file
carries a std-only symbol. The brief requires proving `cpu` actually
compiles this time by injecting a deliberate syntax error and confirming the
alloc-tier check fails — because a green check here has already lied once.

## ROW 43 — executor compiles at the alloc tier, PROVEN by deliberate failure

Worktree `scratchpad/alloctier-wt`, uncommitted.

**Change 1 — math.** `libm` added optional via `cargo add`, enabled by the
`alloc` feature. One `#[inline]` shim per op with two `cfg` bodies
(`cpu.rs:82-129`): std calls `f32::{exp,ln,sqrt,tanh}`, alloc calls
`libm::{expf,logf,sqrtf,tanhf}`.

**Change 2 — gate.** `pub mod cpu;` moved from `feature = "std"` to
`feature = "alloc"` (`lib.rs:174`). `evaluate_parallel`,
`evaluate_node_parallel`, `run_chunks_threaded`, `PARALLEL_THRESHOLD`, the
`std::{panic,thread}` imports and `NonZeroUsize` each gated
`#[cfg(feature = "std")]` individually. Re-exports split accordingly.

**Proof the tier check is real this time.** Injected
`DELIBERATE_SYNTAX_ERROR_PROOF !!!` into `evaluate` -> alloc-tier check
**exit 101**, `error: expected one of '(', '[', or '{'` at `cpu.rs:337`.
Reverted -> **exit 0**. `cpu.rs` now genuinely reaches rustc at the alloc
tier. ROW 42 recorded the same command returning 0 while compiling **0 of
4862** executor lines; that is now closed.

Gates: **136/136** with and without `--features instrument` (136 not 134 —
inherited row-remainder boundary cases), clippy `--all-targets --features
instrument` clean, default std build clean with `evaluate_parallel` present.

### The audit missed a whole symbol category, and it was the important one

Beyond ROW 42's four transcendentals, two more were required:

- a **fifth** transcendental site, `apply_scalar_op` (`cpu.rs:3398-3401`),
  reachable only via `BodyShape::Generic`
- **`f32::mul_add` at 7 sites** (`cpu.rs:1818, 1826, 1834, 2218, 2409, 2419,
  2609`) — **not in `core` on stable**, and a category ROW 42 never
  mentioned. Fixed with `libm::fmaf` and the same shim shape.

`mul_add` is the instruction at the centre of everything landed today — the
FMA fold (ROW 14), both tile kernels (ROWs 20, 22), the remainder tile
(ROW 36). **The audit enumerated exactly what I asked it to look for** —
"transcendentals, threads" — and the most load-bearing symbol in the file
fell outside the question. A checklist audit finds what the checklist names;
the framing is the coverage.

### Honest limitation, reported not glossed

`cargo tree -i libm` shows libm in the **std** graph too. `std = ["alloc",
...]` is pre-existing, so std transitively enables alloc, which enables
`dep:libm`. Cargo features are additive and cannot express "alloc but not
std", so a std build resolves a crate its `cfg` bodies never reference.
Unavoidable without restructuring the feature graph; recorded rather than
claimed as isolation.

## ROW 44 — prediction CONFIRMED: remainder explains half the gap. 1024^3 now 1.059x

Quiet box, `sample_size(50)`, `measurement_time(10 s)`, tinyBLAS confirmed by
`nm`. **All 8 arms usable** — CI widths 0.24%-3.02%, 450-17000 iterations
each. Contrast ROW 38: the old config gave 2 iterations at 2048^3.

| arm | estimate | iters/samples | CI width |
|---|---|---|---|
| ggml t1 @512 | 2.9473 ms | 3825 / 50 | 0.24% |
| ours ST @512 | 3.2717 ms | 3825 / 50 | 0.51% |
| ggml t1 @1024 | 23.483 ms | 450 / 50 | 0.42% |
| ours ST @1024 | 24.877 ms | 450 / 50 | 0.54% |
| ggml t8 @1024 | 3.8562 ms | 3825 / 50 | 1.07% |
| ours w8 @1024 | 7.4935 ms | 2550 / 50 | 2.24% |

**Single-thread, propagated uncertainty:**
- **1024^3: 1.0594 +/- 0.0036 -> 5.6% to 6.3% behind** (was 1.105)
- **512^3: 1.1101 +/- 0.0031 -> 10.7% to 11.3% behind** (unchanged)

ggml control drift -0.7% to -3.8% across both prior runs. No signal.

**The asymmetry recorded in ROW 40 before the result appeared exactly.**
1024 mod 6 = 4, absorbed by the 4-row tile, `fallback_elements = 0`, ratio
improved. 512 mod 6 = 2, below the tile's `>= 4` threshold, still scalar,
ratio did not move at all. A uniform improvement would have refuted the
explanation; the split confirms it.

**It bought half.** 1.105 -> 1.059 closes ~4.6 of ~11 points. **~5.6%
remains unexplained at 1024^3**, and the full ~11% remains at 512^3.

### The same defect is what pins the parallel arm at 1.94x

`evaluate_parallel` splits M=1024 across 8 workers = **128 rows each, and
128 mod 6 = 2** — below the threshold. Measured in that arm alone:
`row_remainder_invocations = 0`, `fallback_elements = 25,067,520`. The tile
that fixed the serial case never fires once rows are chunked.

Parallel ratios this run: 512^3 **1.918 +/- 0.035**, 1024^3 **1.943 +/-
0.024**.

### Two measurement saves worth keeping

**The counters lumped two arms together.** The ported snippet summed
`evaluate` and `evaluate_parallel`, printing
`fallback_elements = 50,151,424` — which reads as "the remainder tile
failed." Separated: serial **0**, parallel **25,067,520**. An aggregate over
two populations that disagree, again (ROW 30's mean(1,1024) was the same
shape).

**The agent discarded its own re-run's timings** — 16.2% CI width and
criterion itself flagged "+22.373% regression" from insufficient cooldown
between consecutive builds — while keeping that run's counters, which are
discrete and thermally invariant. Knowing which measurements a contaminated
run still supports is the distinction; throwing out the whole run would have
lost the diagnostic that found the parallel gap.

**Fix dispatched:** dispatch every leftover count 1..=5 to a tile of that
exact width rather than one `>= 4` threshold. Register budget at ROWS=5 is
20 accumulators + 9 staging = 29 of 32, the tightest case and the one to
check for spills. Covers 512^3 and the parallel path with one change.

## ROW 45 — READ THE INCUMBENT'S SOURCE: tinyBLAS declines shapes it cannot tile

`sgemm.cpp:343-379`, `tinyBLAS::matmul`:

```cpp
if (k % KN != 0) return false;                              // KN = 4 for f32
if (m % 16 == 0 && (m/16 >= nth)) { mnpack<4,6,4>(); return true; }
if (m % 8  == 0)                  { mnpack<4,6,2>(); return true; }
if (m % 4  == 0)                  { mnpack<4,6,1>(); return true; }
return false;                                               // -> naive ggml_vec_dot_f32
```

**tinyBLAS has no remainder handling at all — it refuses the job.** M not
divisible by 4, or K not divisible by the vector width, and ggml drops to
`UseGgmlGemm1` (`ggml-cpu.c:1240-1252`), the naive path.

**Every comparison in this log used m = 512 / 1024 / 2048** — all divisible
by 16, so ggml took `mnpack<4,6,4>`, its best path, every single time. Our
kernel handles arbitrary M through remainder tiles. **The regime where ggml
falls back has never been benchmarked**, and it is a regime we should win
outright. Under test now, with the source reading itself treated as
falsifiable: if ggml's ns/MAC shows no cliff at 1022/1023/1025/1026, the
reading is wrong and that is the finding.

### Three structural differences the source states outright

**1. Their RM is always 4.** Only RN varies (6 -> 1) via `mnpack` recursing
on `BLOCK_SIZE<6>(n)`. They tile **4 rows x 6 cols**; we tile 6 x 4.

**2. They pick loop order by comparing the tile dimensions**, and say so:

```cpp
// help compiler for op order.
if constexpr (RM <= RN) { V Av[RM]; /* load all A, stream B */ }
else                    { V Bv[RN]; /* load all B, stream A */ }
```

The **smaller** operand set stays resident; the larger is streamed. We
hardcode one order.

**This is almost certainly ROW 31's unexplained asymmetry** — 4x6 measured
6.22 GFMA/s and 6x4 measured 10.74 in our kernel, 74% apart at identical
accumulator count and identical loads/MAC, and the agent flagged it
unexplained. Their 4x6 is fast because `RM <= RN` selects the A-resident
order; our 4x6 got whichever order we wrote. **The shape was never the
variable — the loop order was**, and it is one `if constexpr` in their
source that I read past twice.

**3. They block over N** — `BLOCK_SIZE<6>(n)` with `BN = 12`, and `BM` as a
row-block multiplier (`gemm<RM,RN,BM>` asserts `m % (RM*BM) == 0`, loops
`bi` over `BM*RM` rows). ROW 29 tested blocking on *our* tile and found
nothing; theirs is entangled with the mnpack shape selection and is not the
same experiment.

### The process failure

I read `gemm_bloc` twice — ROW 17 and again for ROW 24 — and both times took
the inner loop and stopped. The dispatch above it says tinyBLAS declines
whole shape classes, and the `if constexpr` inside it says loop order is
chosen per shape. Two facts that reframe the comparison and the shape sweep,
both in a file already open, missed because I was reading for the mechanism
I had already hypothesised. **Owner: "you have the code from ggml, it's not
like you don't have code you can look at."**

## ROW 46 — principle 12 satisfied: build-time sizing through conflaguration

Worktree `scratchpad/sizing-wt`, uncommitted.

`proxima-tensor.toml` (5 sections, every key documented with meaning and
raise/lower cost) -> `build.rs` via
`conflaguration::builder().file().env().validate().build()` ->
`pub mod sized { include!(..) }` in `lib.rs`. **All 7 consts deleted, 84
usage sites rewritten to `crate::sized::*`, zero bare literals left.**

New caps enforced with the two deliberately different behaviours:
- `PROGRAM_MAX_RANK = 8` — hard `TensorError::RankExceedsMax`, enforced in
  `ShapeTable::push`
- `PROGRAM_MAX_OPERANDS = 8` / `PROGRAM_MAX_FUSION_STEPS = 16` — **fusion
  cutoff, never an error**; `compose_body`/`compose_operand` stop absorbing
  and materialize. `FUSION_CUTOFFS` counter behind `instrument`.

Gates: **138/138** with and without `--features instrument` (136 base + 2
new), clippy `--all-targets --features instrument` clean, alloc tier clean
with `sized` reachable.

**The override was demonstrated, not asserted.** `PROXIMA_TENSOR_TILE_ROWS=4`
-> emitted `pub const TILE_ROWS: usize = 4;` (was 6), with
`cargo:rerun-if-env-changed=PROXIMA_TENSOR_TILE_ROWS` present in `-vv`
output. And the cross-axis `Validate` rule fired for real: setting rows=4
while `row_remainder_rows` was still 4 violated `row_remainder_rows < rows`
and **failed the build first** — the cascade works through the env path,
proven by tripping it.

### The required test caught a cap that never fired

First cutoff implementation checked `steps.len()` directly. Steps are pushed
**postorder** — later steps must reference already-computed earlier ones
(`cpu.rs:3311` documents the invariant) — so `steps.len()` reads **0**
throughout the downward walk, and the cutoff never engaged for the exact
long-chain shape the cap exists to bound. An interim fix reserving preorder
broke 4 existing tests by reversing `StepArg::Step` ordering against that
same invariant. Final: a separate `step_budget` in `ComposeSink`,
incremented on entry, with `steps` still pushed postorder.

**Without the brief's "construct a chain that would fuse past the cap"
requirement this ships as a cap that silently never engages** — the same
class as ROW 15's feature gating zero lines. Verifying that a limit *fires*
is a distinct test from verifying the code compiles with the limit present.

### A gap in the workspace, not the crate

`proxima_build`'s helpers (`resolve_profile`, `emit_generated_module`,
`emit_cfg_directives`, `emit_rerun_directives`) are **hardcoded to the
`Profile` struct** and cannot be parameterized over another schema. AGENTS.md
§8 lists exactly those as "domain-agnostic (the pattern itself) … **All
reusable**". They are not. `prime/build.rs` hand-rolls its own sizing axes
around them, and proxima-tensor now does too.

§8 already names the fix — "extract a `build-support` crate from
`proxima-build` that exposes generic helpers parameterized over a
domain-supplied `Profile` struct and sizing schema" — and there are now
**two consumers** demonstrating the need rather than one predicting it.

## ROW 47 — integrated2-wt: three branches consolidated, nothing dropped

`scratchpad/integrated2-wt`, detached at `847f20c5`, uncommitted. Carries
`rowrem-wt` (6x4 tile + 4-row remainder + width tile + counters + tests),
`alloctier-wt` (libm shims incl. `fma`, module gate moved to `alloc`), and
`sizing-wt` (principle-12 build-time config, enforced caps).

| gate | result |
|---|---|
| `nextest -p proxima-tensor` | **138/138** |
| `nextest --features instrument` | **138/138** |
| `clippy --all-targets --features instrument` | clean |
| alloc tier | exit 0, **proven** (see below) |
| default std | exit 0, `evaluate_parallel` present |
| `profile_hot` | `gemm` 0.030 s, `gemm_rhs_transposed` 0.027 s — no regression |

**Both critical properties demonstrated by making them fail.** Alloc tier:
injected `DELIBERATE_SYNTAX_ERROR_FOR_ALLOC_TIER_PROOF!!!` into `evaluate`
-> exit 101 at `cpu.rs:337`, reverted -> exit 0. Sizing: override emitted
`TILE_ROWS = 4`; the same override *alone* failed the build with
`tile.row_remainder_rows: must be < tile.rows (4), got 4`.

**A conflict resolved rather than picked.** `alloctier` gated
`PARALLEL_THRESHOLD` behind `std`; `sizing` deleted the const outright. The
answer was neither side — the generated `crate::sized::PARALLEL_THRESHOLD`
is reachable at every tier and only the std-gated call site references it,
so the const-level cfg is correctly gone. Taking either diff wholesale
would have been wrong.

## ROW 48 — WHAT WE ARE MISSING: the tile is over the register budget

Register arithmetic, 32 NEON registers, from reading both kernels:

| form | acc | staging | total |
|---|---|---|---|
| **ours 6x4** (both staging arrays live) | 24 | 6 + 4 | **34 — over by 2** |
| **ours 4x6** (both arrays) | 24 | 4 + 6 | **34 — over by 2** |
| tinyBLAS 4x6 (streams the larger side) | 24 | 4 + 1 | **29** |
| 6x4 rewritten in streaming form | 24 | 4 + 1 | **29 — 5 spare** |

Our `gemm_tile_neon` materialises **both** `av[ROWS]` and `bv[TILE_COLS]`
each k-step. tinyBLAS holds only the smaller side as an array and streams
the larger through **one** local, chosen by `if constexpr (RM <= RN)`.

**This invalidates how I read ROW 31's shape sweep.** It measured 4x6 at
6.22 GFMA/s against 6x4 at 10.74 and I concluded 4x6 was a bad shape — but
under our form 4x6 costs 34 registers, *identical* to 6x4. The sweep was
ranking over-budget arrangements by how gracefully LLVM degraded them, not
ranking shapes. tinyBLAS runs 4x6 as its primary path precisely because
their form fits it in 29.

**Zero spills was never evidence of comfort.** ROWs 20/22/33 all reported
`str q = 0` inside the k-loop and I read that as headroom. Rematerialising a
load is what LLVM does *before* it resorts to spilling, so the decisive
static number is **`ldr q` inside the loop versus the intended ROWS+COLS
per k-step** — a count nobody has taken. Under test now, with the refutation
condition stated in the brief: if actual loads equal intended, the
over-budget theory is dead and the timing half is not worth running.

## ROW 49 — every row remainder 1..=5 tiled; parallel fallback 25M -> 0

Worktree `scratchpad/rowfull-wt`. `macro_rules! row_remainder_tile!` over
the identical kernel body, dispatched by
`match rows_remaining { 0 => {}, 5|4|3|2|1 => ... }`. `ROW_REMAINDER_TILE_ROWS`
deleted. New `NEON_TILE_ROW_REMAINDER_ELEMENTS` counter, since the coverage
identity can no longer assume 16 outputs per remainder call.

**Spills: 0 `str q` / 0 `stp q` inside the k-loop at ROWS = 6, 5, 4, 3, 2, 1**
— including ROWS=5, the 29-of-32 tightest case.

Coverage identity `covered == m*n` **exact at 1021, 1022, 1023, 1024, 1025,
1026**, so every match arm (1,2,3,4,5,0) is exercised. 139/139 tests, clippy
clean.

**The result that matters — `evaluate_parallel`, 1024^3, workers=8:**

| | before | after |
|---|---|---|
| `row_remainder_invocations` | 0 | 2048 |
| `row_remainder_elements` | 0 | 16384 |
| **`fallback_elements`** | **25,067,520** | **0** |

16384 = 8 workers x (128 mod 6 = 2 rows) x 1024 cols, exact. That was the
defect pinning the parallel arm at 1.94x (ROW 44).

### Three things the agent did that I would have got wrong

1. **It labelled the `profile_hot` run "a no-regression control, not
   evidence of the win."** 1024 mod 6 = 4 was already covered before this
   change, so that size structurally cannot exercise the new arms.
   Presenting an unchanged number as validation would have been easy.
2. **It found in-process counter contamination** — `cargo test`'s default
   parallelism gave `gate_passes` of 4-6 instead of 1. Isolated per-process
   via nextest. Aggregating counters across concurrent tests would have
   silently corrupted the coverage identity.
3. **It explained the residual instead of claiming zero.**
   `fallback_elements` is still nonzero at 1021/1022/1023/1025/1026 because
   that counter also tallies the **column** tail and n is not a multiple of
   `TILE_COLS = 4` at those sizes. The row axis is structurally 0; the column
   axis is untouched.

**Next, dispatched:** the same fix on the column axis, generic over both
dimensions, with the four regions — main, row-tail, column-tail, **corner**
— each tiled. Rectangular cases (1021,1022,1024) and (1022,1021,1024)
exercise the corner block where both tails meet, which neither axis alone
would catch.

## ROW 50 — we BEAT ggml 28-30% at every shape tinyBLAS declines

Row I added to `bench_vs_ggml.rs`. tinyBLAS confirmed by `nm`. All arms
<=5.08% CI. Correctness max abs diff **5.4e-7** at every shape.

| shape | ggml ns/MAC | ours ns/MAC | ratio |
|---|---|---|---|
| 1024^3 (their `mnpack<4,6,4>`) | 0.0037 | 0.0086 | 2.320 +/- 0.044 |
| 1020^3 (their narrow `mnpack<4,6,1>`) | 0.0047 | 0.0071 | 1.523 +/- 0.044 |
| **1022^3** | **0.0118** | 0.0085 | **0.718 +/- 0.012** |
| **1023^3** | **0.0117** | 0.0085 | **0.722 +/- 0.013** |
| **1025^3** | **0.0104** | 0.0080 | **0.764 +/- 0.022** |
| **1026^3** | **0.0098** | 0.0073 | **0.745 +/- 0.007** |
| **k=1022, m=n=1024** | **0.0104** | 0.0075 | **0.722 +/- 0.008** |

**ggml's per-MAC cost jumps 2.6-3.2x exactly where ROW 45's source reading
predicted.** Ours is flat across every shape (0.0071-0.0086, no correlation
with alignment). The `tiled_1020` control — divisible by 4 but not 8, so
tinyBLAS takes its *narrow* path — sits only 27% above their control, which
is what "narrower NEON kernel" looks like versus a cliff.

`k_decline_1022` isolates the `k % KN != 0` gate with m and n both aligned:
still 0.722. The k precondition alone triggers it.

Agent's own caveat, correctly raised: for the **square** decline shapes k = m,
so `k % KN != 0` trips independently of the m-branch — two sufficient
reasons, not a demonstration of the m-logic alone.

**Caveat that is mine:** this ran `evaluate_parallel_w8` vs ggml t8 on
`rowrem-wt` — *before* ROW 49 took parallel `fallback_elements` from
25,067,520 to 0. So the 2.320 control carries a defect since fixed, and it
is a parallel number, not the 1.059x single-thread figure. Needs re-measuring
on the current branch.

## ROW 51 — ROW 48's register arithmetic was too naive; refuted in part

| shape | form | acc | intended loads | actual | spills |
|---|---|---|---|---|---|
| 6x4 | A (ours) | 24 | 10 | **10** | **0** |
| 5x5 | A | 25 | 10 | **10** | **0** |
| 4x6 | A | 24 | 10 | **11** | **1** |
| 4x6 | **B** (streaming) | 24 | 10 | **10** | **0** |
| 8x4 | A / B | 32 | 12 | 19 / 18 | 7 / 6 |
| 4x8 | A / B | 32 | 12 | 41 / 21 | 61 / 41 |
| 6x6 | A / B | 36 | 12 | 52 / 29 | 71 / 53 |

**ROW 48 claimed 6x4 needs 34 registers and is over budget. It is not** —
actual loads equal intended, zero spills. LLVM does not hold all staging
vectors live simultaneously; it interleaves loads with FMAs so lifetimes
never overlap. **A static sum of declared arrays is not a register-pressure
measurement, and I reported it as one.**

What survives is narrower and real: **4x6 in our form spills** (1 extra
load, 1 store per k-step) and the streaming form removes it **exactly**
(11/1 -> 10/0). That is the only shape where *staging* drives pressure. At
8x4/4x8/6x6 the accumulator count alone (>=32) exhausts the file before
staging is counted; streaming reduces the damage but cannot fix it.

Whether one spill explains 4x6's 6.22 vs 6x4's 10.74 GFMA/s is **unknown**.
It is ~8% more memory ops against a 73% gap, so not by volume; a
store-then-reload round trip inside the loop could serialise far worse than
its count implies, but that is a mechanism I would be inventing without
timing.

### I starved my own measurement

PART 2 never ran. The probe checked load average every 60 s for 15 minutes —
**17.21, 11.32, 18.14, 14.70, 16.13, 12.46, 8.56, 7.75, 6.43, 6.44, 7.95,
12.08** — never below 2.0, and correctly refused to time on a contended box
per its own brief.

The contention was **me**, running six agents concurrently. I have been
treating dispatch as free. It is not: build work and benchmark work compete
for the same cores, and this is the **second** timing run lost to it today
(ROW 36 was the first). **Timing work must be serialised against everything
else, and that is a scheduling constraint on the coordinator.**

## ROW 52 — DEFINITIVE, quiet box: 6.5% behind single-thread, 67% behind parallel

23 minutes of load polling before benching (3.95 … 1.93, 1.50, 1.57 —
three consecutive readings under 2.0). Every arm **under 2% CI**. ggml
control drift **-0.12%** at 512^3 and **-0.46%** at 1024^3.
`fallback_elements = 0` in **every** arm that ran, `evaluate` and
`evaluate_parallel` separated.

| size | single-thread | parallel (8w) |
|---|---|---|
| 512^3 | **1.1006 +/- 0.0033** | 2.0446 +/- 0.0233 |
| 1024^3 | **1.0648 +/- 0.0038** | **1.6661 +/- 0.0202** |
| 2048^3 | harness skips our t1 arm | 1.3017 +/- 0.0105 |

**Both predictions resolved:**
- **512^3 improved, barely** — 1.1101 -> 1.1006, a 2.1σ shift.
  `row_remainder_invocations = 620,544` proves the new tiling fires. Real
  but incremental; I had implied a step-change.
- **Parallel improved substantially** — 1.943 -> **1.6661**, an 8.8σ shift,
  matching `fallback_elements` 25,067,520 -> 0 exactly.
- **1024^3 single-thread did not move** (1.0594 -> 1.0648, ~1σ). Correct:
  1024 mod 6 = 4 was already covered by the old 4-row tile.

## ROW 53 — the parallel gap is 1x decomposition vs their 16-32x

`sgemm.cpp:429-478`. Their `gemm<RM,RN,BM>`:

```cpp
const int64_t ytiles = m / (RM * BM);
const int64_t nb_job = ytiles * NB_BN;          // 2D grid: row-block x column-block
if (params->ith == 0) ggml_threadpool_chunk_set(params->threadpool, params->nth);
ggml_barrier(params->threadpool);
int64_t job = params->ith;                       // start on own index
while (job < nb_job) {
    const int64_t ii = (job % ytiles) * RM * BM;
    const int64_t jb =  job / ytiles;
    ... work the block ...
    job = ggml_threadpool_chunk_add(params->threadpool, 1);   // ATOMIC claim
}
ggml_barrier(params->threadpool);
```

| | ggml | ours |
|---|---|---|
| jobs per worker | **16-32x** (128-256 jobs / 8 threads at 1024^3) | **1x** (8 chunks / 8 threads) |
| assignment | dynamic, atomic claim | static, pre-assigned |
| threads | persistent pool, barrier in/out | **fresh spawn per call**, join |
| grid | 2D (row-block x column-block) | 1D (M only) |

Measured scaling: ggml 5.08x / 6.08x / 6.38x at 512/1024/2048; ours 2.74x /
3.88x. With static 1x partitioning the join waits on the slowest worker and
any OS scheduling hiccup costs the whole GEMM.

**This also explains why ROW 32's 2D partition failed to help.** I split into
8 static cells and kept the same **1x** decomposition — the missing
ingredient was never the 2D split, it was the **atomic job counter**. I had
read `gemm_bloc` three times and never read the 40 lines above it.

Their `chunk_set(nth)` detail: each thread starts on its own `ith`, so the
first unclaimed job is `nth` and round one needs no coordination at all.

Dispatched: keep `thread::scope`, spawn once, each worker runs a claim loop
over an `AtomicUsize` against a 2D job grid sized for **>= 8x** jobs per
worker, blocks aligned to `TILE_ROWS`/`TILE_COLS` so they stay on the tiled
path, writing in place via the `SharedOutput` pattern that already exists in
`split2d-b-wt`. Target 6.08x.

## ROW 54 — work-stealing lands correct but does NOT move scaling; my target was 14x too coarse

`scratchpad/steal-wt`. `BoundOp::job_grid` + `run_jobs_threaded`:
`SharedOutput` raw pointer, `AtomicUsize` claim loop mirroring ggml's —
each thread starts on job `ith`, counter seeded at `active_workers`.

**Correct**: 0/65536 and 0/1048576 **bitwise** mismatches at workers
2/3/4/6/8, max relative error **0.000e0**. 139/139 tests. clippy clean.
`fallback_elements = 0`, coverage identity holds
(`43520*24 + 4096 = 1048576`).

**And it did not help.** 1024^3 8-worker: **3.88x**, versus 3.88x cited and
**4.08x** in the agent's own reproduced static baseline. ggml: 6.08x.

| workers | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|
| speedup | 3.88x | 4.06x | 4.09x | 3.91x | 4.03x |

Flat 4.0-4.1x from 16 to 64 with **no collapse** — the claim loop does
tolerate heavy over-subscription, which is what it was for. It just is not
the binding constraint.

### My brief set the target 14x too low

Computed from `sgemm.cpp:349` + `429-441` at m=n=1024, RM=4 BM=4 RN=6 BN=12:

```
ytiles = m/(RM*BM)        = 64 row-blocks   (16 rows each)
xtiles = ceil(n/RN)       = 171
NB_BN  = (xtiles+BN/2)/BN = 14 col-blocks   (~73 cols each)
nb_job = 64 * 14          = 896             -> 112 jobs/worker at 8 threads
```

Ours: **64 jobs, 8 per worker, 128x128 blocks.** I specified
`nb_jobs >= 8 * workers`. Theirs is **112x**, and is a function of the
problem shape alone — **not coupled to worker count at all**, which is
precisely what lets over-subscription be absorbed.

### The agent's traffic hypothesis is refuted by ggml's own grid

It proposed that a square grid re-reads operands more and that this caused
the plateau. Their 64x14 grid re-reads A **14** times and B **64** times;
our square 8x8 re-reads each **8** times. **ggml moves strictly more
operand traffic than we do and still scales 6.08x to our 3.88x.** Traffic is
not the limiter. The agent labelled its own reasoning inference-not-
measurement, and their numbers falsify it.

Dispatched: ggml's formula with our tile — `row_block_rows = TILE_ROWS*BM`
(24), `col_blocks = (xtiles + BN/2)/BN` — giving 42 x 21 = **882 jobs** at
1024^3 against their 896. `BM`/`BN` go in the sizing TOML per §12, not as
literals. If this still does not move, granularity is not the limiter
either, and the remaining suspects are **per-call thread spawn** (we spawn
via `thread::scope` every call; they use a persistent pool) and the
**barrier structure**.

## ROW 55 — granularity REFUTED; and prime dispatch is 53x CHEAPER than thread::scope

### Granularity was not the limiter

Matched ggml's formula exactly: 42 row-blocks x 21 col-blocks = **882 jobs**
at 1024^3 against their 896, `nb_jobs` decoupled from `workers` as theirs is.
Correct throughout — 0 bitwise mismatches at every worker count,
`fallback_elements = 0`, 139/139.

**8-worker speedup went 3.88x -> 3.27x. Worse.** The 16-64 plateau
(3.9-4.1x) did not move at all. A 14x finer grid widened the gap at the
worker count under test. **Refuted.**

### The absolute times say it is a wall, not a curve

| workers | 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| ms | 25.80 | 7.90 | **6.29** | 6.41 | 6.62 |

**No worker count breaks ~6.3 ms.** ggml is at 3.85 ms. A 4.1x cap on 8
cores implies ~14% serial by Amdahl.

### Measured: prime dispatch vs thread::scope

| | prime `spawn_on_core` | `thread::scope` spawn+join |
|---|---|---|
| round trip | **308.5 ns** | **16,471 ns** |
| throughput, 8 cores | 2.41 M/s | 62.6 k/s |

**53x cheaper per round trip, 39x the throughput.** Prime's workers are
already-running per-core executors; a spawn pushes a message into a lane.
`thread::scope` pays OS thread create/teardown **per call**.

`evaluate_parallel` spawns `workers` threads per call. Spawn as a fraction
of measured runtime:

| workers | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|
| thread::scope | 1.7% | 4.2% | 8.2% | **11.9%** | **15.9%** |
| prime | 0.03% | 0.08% | 0.15% | 0.22% | 0.30% |

**That is the plateau** — past 16 workers, spawn cost outruns the gain,
which is why 6.29 ms degrades to 6.62 ms. It does not explain the 8-worker
deficit (1.7% there).

Per-job dispatch is affordable on prime: 882 spawns = 0.272 ms = **1.09%**
of a 25 ms GEMM. On `thread::scope` it would be 14.5 ms = 58%.

### Two more findings from source

**Core placement.** ggml sets thread priority and affinity
(`ggml-cpu.c:2360-2420`, `set_numa_thread_affinity` at `:2015`). We set
**nothing** — bare `std::thread::spawn` takes default QoS, and on Apple
silicon that permits placement on the 2 efficiency cores instead of the 8
performance ones. Every clean standalone probe this session pinned
`QOS_CLASS_USER_INTERACTIVE`; production does not.

**I had the ownership argument backwards.** I wrote that prime's
shared-nothing model *conflicts* with a shared mutable output. The opposite:
`spawn_on_core` requires `'static` and prime has **no scoped-thread
equivalent**, so a borrowed `&mut [f32]` cannot be expressed at all. The
raw-pointer + disjointness-by-construction wrapper is the **only** shape
prime admits — required, not a workaround. `InboxFull` is real but distant:
1024 slots per lane against ~110 jobs per core.

Everything converges on one change — dispatch through `Runtime` fixes spawn
cost, core placement, and the `'static`/ownership question at once, and is
what principle 11 requires anyway.

## ROW 56 — versailles: our dispatch scales 7.11x. The Mac is the outlier.

Intel i7-9700K, 8 homogeneous cores, no HT, `taskset` works, governor set to
`performance`, load 0.17-0.33. **x86-64, so the NEON tiles do not compile —
this measures the parallel machinery on the scalar path, not kernel speed.**

1024^3, medians of 3, speedup vs workers=1:

| workers | A scope unpinned | B scope pinned | C prime unpinned | D prime pinned | Mac |
|---|---|---|---|---|---|
| 8 | 6.48x | **7.11x** | 4.82x | 4.91x | 3.88x |
| 16 | 6.62x | 7.11x | 5.86x | 5.46x | 4.06x |
| 32 | **7.09x** | 6.91x | 6.19x | 5.75x | 4.03x |

**7.11x on 8 cores.** The work distribution I have suspected for hours is
not the problem. **The 3.88x is specific to the M1 Max**, and since the NEON
kernel there runs ~6x faster per core than this scalar path, the likely
cause is bandwidth saturation arriving long before core saturation — a
different problem from the one I was chasing.

### Three of my positions die

**1. Pinning does not help, on either platform.** D vs C: no benefit,
*worse* at 16 and 32. B vs A: ~10% only exactly at workers=8, gone by 16.
The QoS/affinity theory is dead. On the Mac it was untestable
(`THREAD_AFFINITY_POLICY` is a no-op on Apple silicon); on Linux it is
testable and the answer is no.

**2. My static-ranges correction made it worse.** I told the agent to drop
the shared counter because prime is shared-nothing and "uniform work on
dedicated cores balances exactly." Measured: prime static ranges **30%
slower** than `thread::scope` dynamic claiming at 8 workers (30.2 vs
23.2 ms). Stealing is not a workaround for a problem prime lacks — prime
has it too.

**3. Cheap dispatch does not buy speed.** Prime dispatches **43x cheaper**
(612 ns vs 26,267 ns per round trip on this box) and is **slower at every
worker count from 4 up**. Dispatch cost was never the binding constraint;
claiming discipline was. Agent's read of the two implementations
(`cpu.rs:518-556` vs `:703+`) is flagged as unverified mechanism, not
measured.

### A real cross-arch defect

`evaluate_node_parallel` / `evaluate_node_parallel_on_runtime`
(`cpu.rs:430-434, 467-471`) reference `TILE_ROWS`/`TILE_COLS`/
`JOB_GRID_ROW_BLOCK_TILES`/`JOB_GRID_COL_BLOCK_DIVISOR` unconditionally,
but those are defined only under `#[cfg(target_arch = "aarch64")]`
(`cpu.rs:2784-2804`). **The `runtime` feature does not compile on x86-64
at all.** Job-grid partitioning is pure arithmetic; only `gemm_tile_neon`
needs the arch gate. Also `WidthPathContext` (`cpu.rs:2402`) trips
`deny(dead_code)` off-aarch64. Fixed on the remote copy only; our worktrees
untouched. Worth landing.

### What this reframes

The parallel question is now: **why does the Mac cap at ~4x when the same
code caps at 7.11x on Linux?** Candidates, none yet measured: heterogeneous
P/E cores with no pinning available; or the tiled kernel being fast enough
that 8 cores saturate memory bandwidth. The second is testable — run the
Mac sweep against a deliberately slowed kernel and see whether scaling
recovers.

## ROW 57 — the chain, from machine ceiling to the gap. METRICS ARE NOT RESULTS.

Owner: *"metrics are not results. understanding is not metrics."* Correct —
I had accumulated ratios and called the pile a finding. The chain:

| rung | G vFMA/s | % of machine peak |
|---|---|---|
| machine pure-FMA ceiling (no memory) | 12.15 | 100% |
| **our kernel, one L1-resident tile** | 11.68 | **96.1%** |
| **ggml, full 1024^3** | 11.48 | **94.5%** |
| **ours, full 1024^3** | 10.79 | **88.8%** |

**ggml runs a full 1024^3 GEMM at 94.5% of the machine's pure-register FMA
rate**, on a 12 MB working set that overflows L2. Their memory system is
effectively free — fully overlapped with arithmetic.

**That kills the framing I used all day.** I repeatedly said "we are
bandwidth-bound." If this machine were bandwidth-bound at these sizes,
**ggml could not be at 94.5% either.** It is not a wall. Their traversal
hides memory; ours does not.

And our kernel is not the problem — 96.1% in isolation, *better* than
ggml's achieved full-grid figure. We lose **7.6%** from isolated tile to
full grid; they lose **1.7%**. The entire 6.4% is that difference.

Untested item, read in their source and skipped twice:
`for (bi = 0; bi < BM*RM; bi += RM)` runs **BM=4 consecutive row-tiles
against the same column block** before advancing — B stays hot across four
row-tiles. We do one tile and move on. ROW 29 tested blocking the *outer*
traversal and found nothing; this is a different granularity.

## ROW 58 — the benchmark shape hid the cost that matters

Owner: *"can't we reuse threads? why build new ones — that's expensive."*

`evaluate_parallel` spawns fresh OS threads **per call**. Measured
16,471 ns per spawn+join on this box, 26,267 ns on versailles.

| | per call, 8 workers | x 500 ops/token |
|---|---|---|
| `thread::scope` | 131.8 us | **65.9 ms/token** |
| prime | 2.47 us | 1.23 ms/token |

One 1024^3 GEMM is ~6 ms, so spawn is **2.2%** — invisible. A transformer
forward pass is ~15 ops/layer x 32 layers, and at 30 tok/s the budget is
33.3 ms/token. **Thread creation alone would consume 198% of it.**

**Every scaling measurement today used the one workload shape that hides
this**, which is why prime looked equal-or-worse in all of them, and why I
went hunting the mechanism in granularity, pinning, counter contention and
dispatch cost — all measured, all null or backwards.

It also means my argument *for* prime was wrong even when the conclusion
was right. I justified it on per-call dispatch being 43x cheaper, then
measured a single GEMM where that bought nothing. The real argument is
**reuse**: prime's executors already exist, so there is no creation cost to
pay 500 times per token.

Chain benchmark dispatched at real Llama-3-8B dimensions (d_model 4096,
d_ff 14336, seq 128), 1 and 32 layers, reporting **time per bound op** and
the spawn-cost fraction so the arithmetic is falsifiable rather than
confirmed.

## ROW 59 — cross-arch defect fixed, proven by exit code

`cargo check --target x86_64-unknown-linux-gnu`: **exit 101 -> exit 0.**

Before: `E0425` on `TILE_ROWS`/`TILE_COLS`/`JOB_GRID_ROW_BLOCK_TILES`/
`JOB_GRID_COL_BLOCK_DIVISOR` at `cpu.rs:373-376` — referenced
unconditionally, defined only under `#[cfg(target_arch = "aarch64")]`.
The constants are pure arithmetic and needed no gate.

Fixing that exposed a second failure — `WidthPathContext`'s fields going
dead off-aarch64. Fixed by **genuine absence**: the struct is now
`#[cfg(target_arch = "aarch64")]` and its sole construction site gated with
it, rather than a blanket `allow(dead_code)`. `gemm_tile_neon` and
`gemm_width_tile_neon` remain arch-gated. 139/139, clippy clean.

**Only found because we finally built on a second architecture.** One host
cannot detect a cfg error that a second host makes immediate.

## ROW 60 — q4_k dot: independent accumulators, 5.7x measured (prediction was 3.2x, LOW)

A root-cause pass decomposed the 48-55x q4_K deficit vs ggml into two
multiplicative factors, both DERIVED from assembly/ISA reading, neither
measured in isolation: (a) 16x scalar-fmadd-vs-`vdotq_s32` SIMD width, (b)
3.2x one-accumulator-serial-chain vs ggml's four independent partial sums.
This row isolates (b) alone.

`dot_q4k_f32` (`cpu.rs:3804`) ran one scalar `mul_add` per weight/activation
pair, threaded through a single `acc` across the whole 256-wide block —
the serial FMA chain `dot_fold_fused_multiply_add`'s own doc comment
already names as LLVM-unwidenable. Changed ONLY the accumulator shape:
each block now folds through the file's existing `dot_fold_fused_multiply_add`
(`DOT_LANES=8` independent partial sums, the same primitive
`reduce_dot_binary` already uses for every f32 GEMM contraction, ROW 12) —
reuse, not a hand-rolled second implementation. `Q4K_BLOCK_ELEMENTS` (256)
is a whole multiple of `DOT_LANES` (8), so every block folds with zero
remainder; no new constant, no new type.

**Correctness (MEASURED):** `bench_q4k_matmul`'s own inline
`assert!(diff < 0.5)` against ggml's `ggml_mul_mat` on the same packed
Q4_K bytes from the real `openchat-3.5-1210.Q4_K_S.gguf`, ran on every one
of 3 shapes x 4 runs (12 checks), all passed, exit 0. Diff magnitude
unchanged by the reassociation (e.g. attn_q: `2.738297e-3` before vs
`2.738595e-3` after — last-ULP drift from reassociating the sum, expected
and within the pre-existing 0.5 tolerance this bench already uses for
Q4_K's own lossy codec).

**Bench: `proxima-tensor/benches/bench_q4k_matmul.rs`, release, single
thread (`_t1` arm), `sample_size(30)`, `measurement_time(5s)`.**

| shape (macs/call) | before ns/mac (3 runs) | CoV | after ns/mac (3 runs) | CoV | speedup |
|---|---|---|---|---|---|
| attn_q 4096x4096 (16,777,216) | 1.3175 / 1.3103 / — (n=2) | 0.27% | 0.2307 / 0.2317 / 0.2333 | 0.47% | **5.68x** |
| attn_k 4096x1024 (4,194,304) | 1.3207 / 1.3128 / — (n=2) | 0.30% | 0.2321 / 0.2315 / 0.2322 | 0.15% | **5.68x** |
| ffn_gate 4096x14336 (58,720,256) | 1.3267 / 1.3117 / — (n=2) | 0.57% | 0.2321 / 0.2316 / 0.2325 | 0.20% | **5.69x** |

Before-run3 timed out (host contention, see loadout below) mid-collection;
n=2 for the before arm, n=3 for after. All CoV well under the 5% bar.

**Host loadout:** shared Mac host, NOT quiet — `uptime` load average
ranged 1.25 (start) to 5.60 (during the after-runs), with other local
tenant processes visible in `ps` throughout alongside this worktree's own
gate/build/nextest runs. Despite the load, per-shape CoV stayed under 0.6%
both before and after — the wall-clock bench proved robust to the
contention, but the load is recorded per the loadout-disclosure rule
regardless.

**Prediction FAILED — in the good direction.** Predicted ~1.315 -> ~0.41
ns/mac (3.2x) if factor (b) alone were real. Measured ~1.32 -> ~0.232
ns/mac, a **5.7x** improvement, not 3.2x. Mechanism, read from the
disassembled release binary (`objdump --macho -d` on the built bench,
`matmul_q4k_f32`'s inlined closure): the compiler did NOT keep 8 scalar
accumulators. It auto-vectorized the `[f32; 8]` lane array into **two**
`float32x4_t` NEON registers (`v6`, `v7`) and the inner loop is:

```asm
1000b0d90: add    x12, x21, x11
1000b0d94: add    x13, x19, x11
1000b0d98: ldp    q16, q17, [x12]     ; 8 scratch (weight) floats
1000b0d9c: ldp    q18, q19, [x13]     ; 8 activation floats
1000b0da0: fmla.4s v6, v19, v17       ; 4 macs, accumulator 1
1000b0da4: fmla.4s v7, v18, v16       ; 4 macs, accumulator 2
1000b0da8: add    x11, x11, #0x20
1000b0dac: cmp    x11, #0x400
1000b0db0: b.ne   0x1000b0d90
```

8 macs/iteration via 2 vector `fmla.4s` instead of the original 1
mac/iteration via 1 scalar `fmadd s8, s0, s1, s8`. The isolated-(b)
experiment did not isolate (b) — breaking the serial dependency chain
*also* unlocked LLVM's auto-vectorizer, which packed the now-independent
lanes into SIMD registers. Factors (a) and (b) are not separable in this
codepath: the derivation's own premise (scalar fmadd persists, only the
chain breaks) was falsified by the compiler's actual behavior. The
combined win (independent chains x this SIMD packing) landed at 5.7x
rather than the modeled 3.2x, still short of ggml's full 16x (int8
`vdotq_s32`, which this f32-intermediate path does not attempt).

**Remaining gap to ggml (MEASURED, same `openchat-3.5-1210` shapes,
`ggml_mul_mat` on identical packed bytes):** before, ours/ggml-t1 =
22.104ms/414.24us = **53.35x** (matches the task's cited 48-55x). After,
ours/ggml-t1 = 3.8701ms/414.24us = **9.34x**. Against ggml's 8-thread arm
(232-244us) — not an apples-to-apples thread count, noted as such — the
remaining gap is **~16.3x**. The 4.24 cycles/mac the original decomposition
assumed no longer describes this path; the dominant remaining cost is the
per-block dequantize-into-`[f32;256]`-scratch step (never eliminated by
this change) plus the still-f32, still-not-int4-native dot.

**Re-prove:** `CARGO_TARGET_DIR=<scratch> GGML_BUILD_DIR=<a built ggml
checkout's dir containing build/src/{libggml,libggml-cpu,libggml-base}.a>
cargo build --release -p proxima-tensor --bench bench_q4k_matmul --features
ggml-bench`, then run the produced binary with `--bench`; the `ns/mac`
values are `time / macs/call` printed per shape, and correctness is the
bench's own inline `assert!(diff < 0.5)`. Disassembly re-derivable via
`objdump --macho -d` on the same binary, `matmul_q4k_f32`'s inlined
`ChunksExact::next` closure.

**Gates:** `cargo nextest run -p proxima-tensor` — 293 passed, 0 failed, 2
skipped. `bash scripts/proxima-tensor-gate.sh` — `passed: 18, failed: 0`.
GEMM checksums (`busy_per_mac` example, `--features instrument`)
unchanged: `512 4 1` -> `135.87619`, `1024 4 1` -> `260.24106`, `2048 4 1`
-> `513.10425` (expected — this path already reused `DOT_LANES=8` via
`dot_fold_fused_multiply_add`; the q4_k change touches a different
function entirely, and the checksums confirm no cross-contamination).

**Landed.** `dot_q4k_f32` now calls the file's existing
`dot_fold_fused_multiply_add`/`DotFold` instead of hand-rolling a serial
loop — reuse-first (principle 1), no new magic number (principle 12/§15,
`DOT_LANES` already existed and was already the measured-best value from
ROW 12).

## ROW 61 — int8 dot on packed q4_K/Q8_K blocks: 6.8x over the f32-dequant path, 1.29-1.40x behind ggml t1 (was 9.34x)

ROW 60 closed the "own gap" between two variants of the SAME mechanism (f32
multiply-add over a dequantized `[f32;256]` scratch). This row replaces the
mechanism: `dot_q4k_q8k` (`cpu.rs`, feature `q4k-int8-dot`, default-off)
reads `Q4_K`'s packed nibbles and a `Q8_K`-quantized activation directly,
doing an INTEGER dot (`i32` partial sums, two `f32` ops per 256-element
super-block: `d*sumi - dmin*mins_correction`) — no dequantize pass, no
`f32` intermediate at all. Two implementations of that ONE mechanism, not
two mechanisms: `dot_q4k_q8k_block_scalar` (portable, no arch intrinsics,
what every non-aarch64 target compiles) and `dot_q4k_q8k_block_neon_dotprod`
(aarch64, `sdot`-accelerated). `matmul_q4k_q8k_f32`/`_portable_f32`
quantize the activation to `Q8_K` ONCE per call (`quantize_row_q8k`,
hoisted out of the row loop — paying it per row would cost 4096x at this
crate's shapes) and share it across every row.

**`vdotq_s32` is unstable on this toolchain (MEASURED, not assumed):**
`rustc 1.97.1` rejects `core::arch::aarch64::vdotq_s32` with
`unstable library feature 'stdarch_neon_dotprod'` (probed directly,
`rustc --edition 2024 -O --target aarch64-apple-darwin`). Issued the `sdot`
instruction via `core::arch::asm!` instead (`sdot_s32`, `#[target_feature(enable
= "dotprod")]`) — ggml's own `ggml_vdotq_s32` C wrapper is the exact
analogue. `FEAT_DotProd` is a build-time decision, not runtime-detected:
`build.rs::emit_dotprod_cfg` emits `cargo:rustc-cfg=q4k_dotprod` whenever
`CARGO_CFG_TARGET_ARCH == "aarch64"` (every aarch64 target this workspace
builds for has the feature; matches the existing NEON tiles' own
"`neon` is unconditional on aarch64 baseline ISA" assumption, `cpu.rs:3877`).
`dot_q4k_q8k` picks the `q4k_dotprod`-cfg'd arm at compile time; no runtime
`is_aarch64_feature_detected!`, no second implementation shipped as a dead
fallback. `dot_q4k_q8k_portable`/`matmul_q4k_q8k_portable_f32` additionally
expose the scalar arm directly (bypassing `q4k_dotprod` dispatch) so it
stays reachable and separately benchable from an aarch64 host, since this
session had no non-aarch64 hardware to bench the portable arm on natively.

**Correctness (MEASURED, ggml as oracle, real `openchat-3.5-1210.Q4_K_S.gguf`
bytes, `bench_q4k_matmul.rs`):** packed-int8 `max_abs_diff` vs `ggml_mul_mat`
on the SAME packed weight bytes:

| shape | old f32-dequant diff vs ggml | packed-int8 diff vs ggml |
|---|---|---|
| attn_q 4096x4096 | 2.7386e-3 | **8.643e-7** |
| attn_output 4096x4096 | 6.2376e-4 | **1.490e-7** |
| attn_k 4096x1024 | 3.0455e-3 | **7.004e-7** |
| ffn_gate 4096x14336 | 1.5026e-3 | **5.066e-7** |
| ffn_up 4096x14336 | 9.1866e-4 | **2.459e-7** |

The packed-int8 diff is 3-4 ORDERS OF MAGNITUDE tighter than the existing
f32-dequant path's diff against the same oracle. Mechanism: ggml's own
`ggml_mul_mat` quantizes the `f32` activation to `Q8_K` internally before
calling this exact `q4_K x q8_K` int8 dot when multiplying against a `Q4_K`
weight — `dot_q4k_q8k` is not merely *like* what ggml does, it is computing
the SAME quantized-activation path ggml runs, so the two agree to within
float32 rounding (~1e-7) rather than differing by two independently-lossy
codecs. Additionally: `matmul_q4k_q8k_f32` (`q4k_dotprod` dispatch) and
`matmul_q4k_q8k_portable_f32` (forced scalar) produce **bit-exact**-equal
output on identical input (`matmul_q4k_q8k_f32_agrees_bit_exact_with_the_
portable_arm`, `cpu.rs` test) — every intermediate value both arms compute
is `i32` until the final two `f32` ops, and integer addition has no
rounding, so `sdot`'s 16-lane hardware reduction and the scalar's 32-wide
serial loop are provably the same mechanism, not two that happen to agree
by luck.

**Bench: `bench_q4k_matmul.rs`, release, single thread, real GGUF weight
bytes, `sample_size(30)`, `measurement_time(5s)`. `attn_q` run 3x for CoV;
`attn_k`/`ffn_gate` run 1x each (time budget) — reported as single-run,
not averaged into a false-precision CoV.**

| shape (macs/call) | old f32 (ROW 60) ns/mac | packed-int8 **portable** ns/mac | packed-int8 **dispatched** (`sdot`) ns/mac, CoV (n=3) | ggml t1 ns/mac, CoV (n=3) | ggml t8 ns/mac |
|---|---|---|---|---|---|
| attn_q 4096x4096 | 0.2288 | 0.1659 | 0.03378, **0.28%** | 0.02477, **0.65%** | 0.01483 |
| attn_k 4096x1024 | 0.2258 | 0.1665 | 0.03414 (n=1) | 0.02644 (n=1) | 0.04452 |
| ffn_gate 4096x14336 | 0.2291 | 0.1654 | 0.03343 (n=1) | 0.02381 (n=1) | 0.01089 |

**Frequency-weighted scorecard (design-favors labels):**

| arm | design-favors | verdict vs ggml t1 | verdict vs prior (ROW 60) f32 |
|---|---|---|---|
| packed-int8 dispatched (`sdot`) | **incumbent** (this IS ggml's own mechanism) | **LOSE 1.29-1.40x** (was 9.34x — 6.7-7.2x closer) | **WIN 6.6-6.9x** |
| packed-int8 portable (scalar) | neutral | LOSE 6.5-7.0x | **WIN 1.36-1.39x** |
| ggml t8 vs ggml t1 | incumbent, thread scaling | 1.7-2.2x (attn_q/ffn_gate); **REGRESSES** 1.68x at attn_k (thread overhead exceeds the work at that shape — a real ggml finding, not ours) | n/a |

Honest read: the `sdot`-accelerated arm is the load-bearing number and it
is a genuine, large step toward the incumbent (9.34x -> 1.29-1.40x gap),
but it is still a LOSS on ggml's own single-thread home turf — this row
does not claim parity. The portable arm, with zero architecture-specific
code, already beats this crate's own prior f32-dequant path by ~1.36-1.39x
at every shape, which is the "portable packing alone" number the owner
asked to see reported standalone, separate from what the intrinsic adds.

**Emitted assembly (MEASURED, not claimed):** `objdump --macho -d` (Apple
LLVM 17 bundled `objdump`) does not have a mnemonic table entry for `sdot`
on this host and prints raw `.long 0x4e949513` words instead; manually
decoding the opcode (`bits[15:10] = 100101`, `U`-bit clear) confirms
ARMv8.2 `SDOT (vector)`, signed. `llvm-objdump` (same Xcode toolchain,
`xcrun --find llvm-objdump`) DOES carry the mnemonic and confirms it
directly — `dot_q4k_q8k`'s compiled body (symbol
`_RNvNtCsdesHdT7369h_14proxima_tensor3cpu11dot_q4k_q8k`) contains 16
`sdot.4s` instructions interleaved with `and.16b` (low-nibble mask) and
`ushr.16b` (high-nibble shift) — the exact three-instruction shape
`vandq_u8`/`vshrq_n_u8`/`sdot` was written to produce, confirmed in machine
code:

```asm
1000b4330: ushr.16b v8, v21, #0x4
1000b4334: movi.2d  v19, #0
1000b4338: sdot.4s  v19, v8, v20
1000b433c: ushr.16b v20, v22, #0x4
1000b4340: sdot.4s  v19, v20, v31
1000b434c: and.16b  v21, v21, v18
1000b4350: sdot.4s  v20, v21, v31
```

**Host loadout:** shared Mac host, moderately loaded throughout this row's
bench runs — `uptime` load average 4.86/6.20/4.69 (1/5/15 min) before, rose
to 6.56/6.20/5.40 after; `ps -eo pcpu,comm` topped by `iTerm2` (46-84%),
`mediaanalysisd` (51-57%), `mds_stores`/`mdworker_shared` (Spotlight
indexing, 7-61%), and one other local CLI process (32-37%). Despite
the load, dispatched-arm CoV stayed at 0.28% (n=3) — well under the 5% bar
— so the load did not visibly contaminate the headline number, but per-run
values ranged 565.31/566.30/568.36 µs, a real (if small) spread worth
recording rather than a single point estimate.

**Types minted: none beyond what the wire format requires.**
`activation_q8k: &[u8]` mirrors ggml's own `block_q8_K` byte layout exactly
(`f32` scale + 256 `i8` quants + 16 `i16` bsums = 292 bytes/block) —
guiding-principles §1: a byte buffer already in the incumbent's own wire
shape needs no host struct type any more than `dot_q4k_f32`'s existing
`weight_row: &[u8]` does. `get_scale_min_k4`/`nearest_int`
(`proxima-gguf::quant::q4_k`) made `pub` (were crate-private) and reused
rather than re-derived — one ggml `nearest_int`/`get_scale_min_k4` per the
upstream source, not two.

**Allocation budget:** hot path (`dot_q4k_q8k*`, `dot_q4k_q8k_block_*`) —
**zero**, matches the stated budget; every buffer is caller-provided or
stack (`[u8; K_SCALE_SIZE]`, `[u8; 4]`). Setup path
(`matmul_q4k_q8k_f32`/`_portable_f32`) — one `Vec<u8>` allocation for the
shared `Q8_K` activation buffer, sized once per call, not per row (the
whole point of hoisting `quantize_row_q8k` out of the row loop).

**Feature gate:** `q4k-int8-dot`, default-off (`proxima-tensor/Cargo.toml`).
`dot_q4k_f32`/`matmul_q4k_f32` (ROW 60's path) are UNTOUCHED and remain the
production default — this row adds a sibling arm behind its own gate, per
guiding-principles §3/§11, until an e2e bench justifies the switch.

**Gates:** `cargo nextest run -p proxima-tensor` (default features) — 293
passed, 0 failed, 2 skipped (unchanged N). With `--features
q4k-int8-dot,test-support` — 298 passed, 0 failed, 2 skipped (+6 new tests:
bit-exact dispatched-vs-portable cross-check, dequantize-oracle tolerance,
zero-vector `Q8_K` exactness, three shape-mismatch guards).
`bash scripts/proxima-tensor-gate.sh` (with `GGML_BUILD_DIR` set) —
`passed: 18, failed: 0`. GEMM checksums (`busy_per_mac --features
instrument`) unchanged: `512 4 1` -> `135.87619`, `1024 4 1` -> `260.24106`,
`2048 4 1` -> `513.10425` — confirms zero cross-contamination with ROW 60's
path. `cargo check -p proxima-tensor --target x86_64-unknown-linux-gnu
--features q4k-int8-dot` — EXIT 0 (the portable arm is what that target
actually compiles; `q4k_dotprod` never fires off-aarch64).

**Re-prove:** `CARGO_TARGET_DIR=<scratch> GGML_BUILD_DIR=<built ggml>/
cargo build --release -p proxima-tensor --bench bench_q4k_matmul --features
ggml-bench,q4k-int8-dot`, then run the produced binary with `--bench
<shape-name>` (the raw binary defaults to criterion's `--test` mode and
prints no timing without `--bench` — verified by reading
`criterion-0.5.1/src/lib.rs:960-964`, `(bench, test) = (false, _) => true`
i.e. test-mode unless `--bench` is passed). `ns/mac` is `time / macs/call`
printed per shape; correctness is the bench's own inline `assert!(diff <
0.5)` plus the bit-exact dispatched-vs-portable assertion. Disassembly:
`xcrun --find llvm-objdump` then `llvm-objdump -d --symbolize-operands
<binary>`, find the `dot_q4k_q8k` symbol.

**Not landed as a default; two negatives kept, not buried:**
1. The `sdot`-accelerated arm still LOSES to ggml single-thread by
   1.29-1.40x — this row closes most, not all, of the gap. The remaining
   difference is plausibly ggml's tinyBLAS-style block/tile scheduling
   around the same `sdot` primitive (unmeasured; a follow-up row's job, not
   asserted here per principle 18).
2. `attn_k`/`ffn_gate` shapes were benched n=1 (time budget), not the
   3-5-run minimum this skill calls for — CoV is UNKNOWN for those two
   rows' numbers, reported as single-run rather than dressed up with a
   borrowed CoV from the attn_q shape.

## ROW 62 — AVX2 int8 dot on the same packed q4_K/Q8_K blocks: compiles, disassembles, UNVERIFIED-BY-EXECUTION

ROW 61 landed `dot_q4k_q8k`'s aarch64 arm. This row adds the x86 sibling:
`dot_q4k_q8k_block_avx2` (`cpu.rs`, feature `q4k-int8-dot`, still
default-off), selected by a new `q4k_avx2` cfg (`build.rs::emit_avx2_cfg`)
the same way `q4k_dotprod` selects the NEON arm — same mechanism (per
sub-block: unpack `(scale, min)` via the already-`pub`
`get_scale_min_k4`, dot 32 packed nibbles against 32 `Q8_K` `i8`
activations, scale, accumulate; mins correction identical), a third
acceleration of it, not a different one. Ports the scalar body of
`ggml_vec_dot_q4_K_q8_K`'s `__AVX2__` arm
(`ggml-cpu/arch/x86/quants.c`, read before writing anything, per the
brief): `_mm256_and_si256`/`_mm256_srli_epi16` split low/high nibbles
(the same 16-bit-lane-shift-then-per-byte-mask trick ggml's own kernel
uses, since x86 has no per-byte 8-bit shift instruction), then
`_mm256_maddubs_epi16` (32-lane unsigned-nibble x signed-`i8` multiply,
pairwise-summed to 16 `i16`) + `_mm256_madd_epi16` against an all-ones
vector (pairwise-summed to 8 `i32`) + a `hsum_epi32_avx2` horizontal
fold — **without** ggml's `_mm256_shuffle_epi8` scale-broadcast table:
this kernel multiplies the horizontally-summed 32-lane partial dot by
its scalar `i32` scale code AFTER the sum, matching
`dot_q4k_q8k_block_scalar`'s own order rather than folding the scale
into the SIMD `madd` itself. Integer multiplication distributes over
integer addition exactly, so this is the identical mechanism at the
identical resulting value, minting no scale-shuffle table this component
does not otherwise need.

**AVX2 is NOT the x86-64 baseline (MEASURED, not assumed):** `rustc
--print cfg --target x86_64-unknown-linux-gnu` lists `target_feature`
values `fxsr,sse,sse2` only — no `avx2`. Unlike `q4k_dotprod` (every
aarch64 target this workspace builds for carries `FEAT_DotProd`, so
`build.rs` can key that cfg on `target_arch` alone), `emit_avx2_cfg`
additionally requires `CARGO_CFG_TARGET_FEATURE` to list `avx2` — i.e.
the build itself must opt in via `-C target-feature=+avx2` / `-C
target-cpu=<v3 or newer, or native on an AVX2 host>`. An unqualified
`x86_64-unknown-linux-gnu` build (this crate's own `cargo check --target
x86_64-unknown-linux-gnu --features q4k-int8-dot` gate cell, no
RUSTFLAGS) compiles the portable scalar arm only — `q4k_avx2` never
fires there, confirmed by grepping the resulting disassembly for
`dot_q4k_q8k_block_scalar`'s callq (found; no `vpmaddubsw`).

**This host cannot execute an x86_64 binary (aarch64 Apple Silicon,
no x86 emulator/rosetta-for-linux-elf available) — every claim below
about the AVX2 arm is COMPILE-TIME and DISASSEMBLY evidence only. No
throughput number, no correctness-by-running number, exists for this
arm. Explicitly UNVERIFIED-BY-EXECUTION:**

1. **Compiles (MEASURED, exit code):**
   `cargo check -p proxima-tensor --target x86_64-unknown-linux-gnu
   --features q4k-int8-dot` — exit 0, portable arm only (no RUSTFLAGS,
   `q4k_avx2` off by design). `RUSTFLAGS="-C target-feature=+avx2" cargo
   check -p proxima-tensor --target x86_64-unknown-linux-gnu --features
   q4k-int8-dot` — exit 0, `q4k_avx2` cfg active. Same command with
   `,test-support` added — exit 0.
2. **The intrinsic path is what actually compiles, not merely what the
   cfg claims (MEASURED, disassembly, not trusted from the cfg name):**
   `cargo rustc -p proxima-tensor --lib --target x86_64-unknown-linux-gnu
   --features q4k-int8-dot --release -- --emit=asm` with
   `RUSTFLAGS="-C target-feature=+avx2"` emits a `.s` file whose
   `dot_q4k_q8k` symbol (`_RNvNtCs82pdBFFVUbe_14proxima_tensor3cpu11dot_
   q4k_q8k`) contains 8 `vpmaddubsw`/8 `vpmaddwd` instances (16 total,
   `grep -c`), inlined directly — no call out to
   `dot_q4k_q8k_block_scalar` (confirmed absent from that symbol's call
   sites, present instead when the SAME command is run WITHOUT the
   `+avx2` RUSTFLAGS). Inner loop (one of four `j` iterations, low-nibble
   half):
   ```asm
   vpbroadcastb .LCPI275_3(%rip), %ymm6
   vpand   %ymm6, %ymm0, %ymm2
   vpmaddubsw 196(%r14,%r13), %ymm2, %ymm2
   vpbroadcastw .LCPI275_4(%rip), %ymm7
   vpmaddwd %ymm7, %ymm2, %ymm2
   vextracti128 $1, %ymm2, %xmm3
   vpaddd  %xmm2, %xmm3, %xmm2
   vpshufd $238, %xmm2, %xmm3
   vpaddd  %xmm3, %xmm2, %xmm2
   vpshufd $85, %xmm2, %xmm3
   vpaddd  %xmm2, %xmm3, %xmm2
   vmovd   %xmm2, %r11d
   ```
   `vpand`+`vpmaddubsw`+`vpmaddwd` is the nibble-mask/multiply/pairwise-sum
   sequence `dot_q4k_q8k_block_avx2` was written to produce; the
   `vextracti128`/`vpaddd`/`vpshufd` chain is `hsum_epi32_avx2` inlined.
   VEX-encoded 256-bit (`ymm`) forms confirm AVX2, not merely SSE.
3. **Bit-exactness is asserted by construction, not by a dedicated new
   test (mirrors ROW 61's own test rather than duplicating it):**
   `matmul_q4k_q8k_f32_agrees_bit_exact_with_the_portable_arm`
   (`cpu.rs`, unchanged by this row) compares `matmul_q4k_q8k_f32`
   (whichever arm `dot_q4k_q8k`'s cfg selects) against
   `matmul_q4k_q8k_portable_f32` (always scalar) and asserts
   `assert_eq!`. Because `dot_q4k_q8k`'s three arms are selected by
   mutually-exclusive cfg (`q4k_dotprod` / `q4k_avx2 and not
   q4k_dotprod` / neither), this ONE test exercises whichever arm the
   build under test actually compiled — NEON dotprod on this host's
   native aarch64 runs (293/298 nextest totals above, unchanged), and
   would exercise AVX2 on an x86_64 build with `q4k_avx2` active, but
   **that build cannot be executed from this host** (point above) — so
   the AVX2 arm's bit-exactness is argued by the SAME reasoning ROW 61
   used for `sdot` (every intermediate is `i32` until the final two
   `f32` ops; integer addition is exact and associative regardless of
   SIMD-vs-scalar summation order, so `hsum_epi32_avx2`'s
   extract-and-fold reduction must equal `dot_q4k_q8k_block_scalar`'s
   32-iteration serial sum bit-for-bit) — NOT proven by a passing
   assertion on this arm, because no assertion has run on it anywhere.
4. **No throughput number exists for this arm at all.** No `ns/mac`, no
   CoV, no `design-favors` scorecard row against ggml's AVX2 kernel —
   the frequency-weighted-scorecard requirement (guiding-principles
   §13/`disciplined-component` gate 13) is UNSATISFIED for this row.
   Producing one requires either (a) a real x86_64 host with the
   toolchain to link a Rust binary (this box has `rustc`/`cargo` targets
   but no `x86_64-linux-gnu-gcc` cross-linker — confirmed: `cargo test
   --no-run --target x86_64-unknown-linux-gnu ...` fails at `alloca`'s
   build script with `ToolNotFound: x86_64-linux-gnu-gcc`, a linking
   step `cargo check`/`cargo rustc --emit=asm` never reach), or (b) CI
   running on an actual x86_64 runner. Scheduled, not done here.

**Types minted: none.** Reuses `get_scale_min_k4` (already `pub`,
ROW 61), the existing `Q4K_*`/`Q8K_*` byte-offset constants, and
`f16_le_at` — the only new items are the two functions
(`dot_q4k_q8k_block_avx2`, `hsum_epi32_avx2`) and their intrinsic
imports, cfg-gated identically to the NEON arm's own `use` block.

**Allocation budget:** hot path (`dot_q4k_q8k_block_avx2`,
`hsum_epi32_avx2`) — zero, matches ROW 61's stated budget; every value
is a register or stack scalar, no new buffers. Not independently
measured on this row (no executable x86 build) — inherits the
COMPILE-TIME guarantee that the function contains no `alloc`/`Vec`/`Box`
call (grepped the source; none present), not a runtime allocation-counter
result.

**Feature gate:** `q4k-int8-dot`, unchanged, still default-off. No new
feature added for the x86 arm — it rides the same gate ROW 61 opened,
selected purely by `build.rs`'s cfg logic at compile time, per the
brief's instruction not to add a second x86 tier (no AVX-without-AVX2
arm, no VNNI/AVX-512 — this workspace's actual targets are aarch64 and
AVX2-or-later x86_64, and the brief's own framing that "if you believe a
second x86 tier is needed, STOP and report why" did not surface a case
for one: `ggml_vec_dot_q4_K_q8_K` itself branches only `__AVX2__` /
`__AVX__` / scalar, and this row deliberately stops at the first,
matching what the incumbent treats as its primary x86 tier).

**Gates:** `cargo nextest run -p proxima-tensor` (default features,
native aarch64) — 293 passed, 0 failed, 2 skipped (unchanged N; AVX2
code is cfg'd off on this host by construction). `--features
q4k-int8-dot,test-support` — 298 passed, 0 failed, 2 skipped (unchanged
N — no new test was added; ROW 61's bit-exact test already covers
whichever arm is active, per point 3 above). `cargo check -p
proxima-tensor --target x86_64-unknown-linux-gnu --features
q4k-int8-dot` — exit 0. `bash scripts/proxima-tensor-gate.sh` (with
`GGML_BUILD_DIR` pointed at a freshly-built static-lib ggml checkout —
`/Users/brianbruggeman/repos/others/llama.cpp/ggml`, standalone
`cmake -S ggml -B ggml/build -DBUILD_SHARED_LIBS=OFF
-DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF`, `ggml.pc.in` was
missing from this vendored checkout and was recreated as a minimal
pkg-config stub so `configure_file` would not abort the build — the
static libs themselves are the standard cmake output, untouched) —
`passed: 18, failed: 0`. GEMM checksums unchanged (native aarch64,
unaffected by an x86-only cfg): `512 4 1` -> `135.87619`, `1024 4 1` ->
`260.24106`, `2048 4 1` -> `513.10425`.

**Re-prove:** the two x86 cells above
(`cargo check --target x86_64-unknown-linux-gnu --features
q4k-int8-dot`, and the same command prefixed with `RUSTFLAGS="-C
target-feature=+avx2"` plus `cargo rustc ... --emit=asm` to regrep for
`vpmaddubsw`/`vpmaddwd`) are exit-code and grep-count re-provable from
this repo alone, no external asset needed. `bash
scripts/proxima-tensor-gate.sh` additionally needs a built ggml
checkout at `GGML_BUILD_DIR` (documented in ROW 61's own re-prove line;
unchanged by this row).

**Not landed as measured, one gap kept, not buried:** this row is
compile-time and disassembly evidence ONLY. It does not claim the AVX2
arm is fast, correct-by-execution, or even that it LINKS on a real
x86_64 host — only that it compiles, that the compiled code contains
the intended AVX2 instructions, and that its bit-exactness rests on the
same integer-associativity argument ROW 61 used for `sdot`, unconfirmed
by an actual run. A future row on x86_64 hardware (or x86_64 CI) owes:
the `q4k-int8-dot,test-support` nextest run showing
`matmul_q4k_q8k_f32_agrees_bit_exact_with_the_portable_arm` PASS under
`q4k_avx2`, and a `bench_q4k_matmul` `ns/mac` table against ggml's own
AVX2 arm (`design-favors: incumbent`) to fill in point 4 above.

## ROW 63 — int8 dot on packed Q5_K/Q8_K blocks: CORRECTNESS ONLY, no timing taken

**This row deliberately carries zero throughput numbers.** A second agent
was mid-edit on `run_reduce`'s parallel dispatch in this same file while
this row landed, and a third was preparing to bench that change plus
Metal together; running a bench under that contention would have produced
a CoV-contaminated number (this repo's own prior incident: CoV 0.3% ->
53% from concurrent agents on one host, `feedback_own_agents_contaminate_the_bench.md`).
Landing scope for this row is explicitly narrowed to: land the kernel,
prove it correct against the already-trusted dequantize-then-fold
reference on REAL packed bytes, prove the two arms bit-exact against each
other, and stop. The `ns/mac` table, CoV, and ggml-parity max_abs_diff
this repo's discipline normally requires (guiding-principles §18: no
throughput claim without a bench number) are explicitly DEFERRED to a
follow-up row, to be taken by whichever agent benches the landed kernel,
the parallel-dispatch change, and Metal together on a quiet tree.

Mirrors ROW 61's mechanism one format over: `Q5_K` shares `Q4_K`'s exact
super-block/sub-block shape (256 elements, 8 sub-blocks of 32, the same
6-bit bit-interleaved scale/min packing — `get_scale_min_k4` is reused
UNCHANGED from `q4_k`, not re-derived) plus one addition, a `qh` high-bit
plane supplying each weight's 5th bit. `dot_q5k_q8k_block_scalar` is
`dot_q4k_q8k_block_scalar` with one surgical addition: `qh_mask = 1u8 <<
sub_block` extracts exactly the bit `proxima_gguf::quant::q5_k::dequantize_block`
reads for that sub-block (derived from, and cross-checked against, that
function's `mask_lo`/`mask_hi` cycling — the derivation is in
`dot_q5k_q8k_block_scalar`'s own doc comment, `cpu.rs`). The `dmin`/mins
correction is untouched from `Q4_K`'s kernel — identical bsums pairing,
identical sub-block width.

`dot_q5k_q8k_block_neon_dotprod` ports `ggml_vec_dot_q5_K_q8_K`'s
`__ARM_NEON` arm (`arch/arm/quants.c:2492-2579`) directly: `mone`/`mtwo`
masks extract the current chunk's two high-bit planes from a
persistently-right-shifted `qh` register pair, OR'd into the nibble
before two `sdot_s32` pairs per 64-element chunk (`Q4K_SUB_BLOCKS/2 = 4`
iterations). No AVX2 arm — per this landing's task ordering ("portable
arm first, measured on its own, then the aarch64 arm... a landed
portable Q5_K with numbers is worth more than two half-finished
intrinsic arms"), and per the coordinator's mid-task redirect, this row
stops at portable+aarch64.

**Feature gate:** `q5k-int8-dot`, new, compile-time, **default-off**
(`proxima-tensor/Cargo.toml`) — unlike `q4k-int8-dot` (default-on since
ROW 61's e2e bench), this has not yet earned the switch; no e2e bench has
run. `QuantizedBlock::Q5K(&[u8])` (new enum variant) always exists
regardless of the feature; `matmul_q5k_f32`/`dot_q5k_f32` (always
compiled, dequantize-then-fold via `proxima_gguf::quant::q5_k::dequantize_block`)
is the codec path when the feature is off, exactly mirroring
`matmul_q4k_f32`'s pre-ROW-61 role. `run_reduce_quantized` dispatches on
the `QuantizedBlock` variant (`Q4K`/`Q5K`/`Q6K`), generalized from ROW
61's Q4K-only body — the byte-offset consts, `f16_le_at`,
`quantize_row_q8k`/`quantize_q8k_block` (`Q8_K` activation quantization,
shared by every K-quant weight codec), `sdot_s32`, and the aarch64
`vorrq_u8`/`vshlq_n_u8` intrinsic imports were broadened from
`q4k-int8-dot`-only gates to `any(q4k-int8-dot, q5k-int8-dot,
q6k-int8-dot)` so `Q5_K`/`Q6_K` reuse them rather than duplicating —
`git diff proxima-tensor/src/cpu.rs` shows every one of those gate
broadenings as a one-line cfg edit, no logic change.

**Types minted: none, per the task's explicit instruction.** Only
`QuantizedBlock::Q5K`/`Q6K` (new enum VARIANTS on an existing type, not a
new type) plus functions and byte-offset consts, mirroring ROW 61/62's
own pattern exactly.

**Correctness — the only claim this row makes:**

1. **Bit-exact, dispatched arm vs portable arm** (synthetic data, 4 rows x
   5 super-blocks): `matmul_q5k_q8k_f32_agrees_bit_exact_with_the_portable_arm`
   — PASS. Every intermediate is integer until the final `f32` multiply
   (same argument ROW 61 makes for `Q4_K`), so `q4k_dotprod`'s
   `vdotq_s32`-accelerated arm and the portable scalar arm must match
   EXACTLY on this aarch64 host, not merely closely, and do.
2. **Packed int8 kernel vs dequantize-then-fold reference, on REAL packed
   `Q5_K` bytes** read directly out of
   `openchat-3.5-1210.Q4_K_S.gguf` (guiding-principles §9: real-world
   data, not synthetic): `blk.0.attn_v.weight` (4096x1024) and
   `blk.0.ffn_down.weight` (14336x4096), this landing's two named `Q5_K`
   shapes —
   `matmul_q5k_q8k_f32_agrees_with_dequantize_then_fold_on_real_gguf_bytes`
   and its `_ffn_down` sibling — both PASS, relative_max_error < 0.01
   (same loose sanity bound ROW 61's own real-weight relative-error test
   uses — Q8_K activation quantization is a second lossy step neither
   arm shares with the other). The dispatched (NEON) and portable arms
   are ALSO asserted bit-exact against each other inside these same two
   tests, on the real bytes, not just the synthetic fixture above.
3. **`evaluate_quantized` end-to-end, `QuantizedBlock::Q5K` variant**:
   `evaluate_quantized_routes_q5k_block_and_matches_dequantize_then_evaluate`
   — PASS, relative_max_diff < 0.01. Proves the dispatch chain
   `evaluate_quantized` -> `run_node_into` -> `run_reduce_with_quantized_weights`
   -> `quantized_operand` -> `run_reduce_quantized` -> `matmul_q5k_q8k_f32`
   is reachable from the program-level entry point for the NEW variant,
   the same proof ROW 61's own e2e test gives `Q4K`.

**No ggml FFI parity number in this row.** The task brief asked for
`max_abs_diff` against ggml directly (mirroring `bench_q4k_matmul.rs`'s
fail-fast-before-timing assert); this row's correctness claim instead
rests on agreement with `matmul_q5k_f32`, the dequantize-then-fold
reference path already ROW-56-through-62-proven against ggml for `Q4_K`'s
own dequantize path and unchanged in mechanism for `Q5_K` (same
`proxima_gguf::quant::q5_k::dequantize_block`, itself bit-exact-tested
against a hand-packed fixture in `proxima-gguf/src/quant/q5_k.rs`). A
direct ggml-FFI `max_abs_diff` number is scheduled for the follow-up
bench row (`bench_q5k_matmul.rs`, written and registered behind
`ggml-bench,q5k-int8-dot` in this same commit, NOT run — see the top of
this row).

**Gates (all re-run on this host, this commit; commands are the
re-prove):**
- `cargo nextest run -p proxima-tensor --features q5k-int8-dot,q6k-int8-dot -E 'test(q5k) + test(q6k)'` — 6 passed, 0 failed (the 3 Q5_K tests above plus ROW 64's 3 Q6_K tests).
- `cargo nextest run -p proxima-tensor --features q5k-int8-dot,q6k-int8-dot --no-fail-fast` — 312 passed, 0 failed, 2 skipped (full crate suite, both new features on, run AFTER a concurrent agent's `run_reduce` parallel-dispatch change landed in this same `cpu.rs` — N grew from the 300/2 baseline at `3f4e4b9` because of this row's + ROW 64's + that concurrent agent's new tests on the same shared tree; the number is reported, not reconciled to a stale baseline). `cargo nextest run -p proxima-tensor --no-fail-fast` (default features, no q5k/q6k) — 306 passed, 0 failed, 2 skipped, same post-merge state.
- `cargo clippy -p proxima-tensor --features q5k-int8-dot,q6k-int8-dot,ggml-bench --all-targets -- ` (workspace lints, `-D warnings` implied) — exit 0, zero warnings, including the new `benches/bench_q5k_matmul.rs`/`bench_q6k_matmul.rs` files (registered, not run).
- `cargo check -p proxima-tensor --target x86_64-unknown-linux-gnu --features q5k-int8-dot,q6k-int8-dot` — exit 0 (the `dot_q5k_q8k_block_neon_dotprod`/AVX2 arms are cfg'd off on this target; the portable arm is what compiles and is what an unqualified x86-64 build runs).
- `bash scripts/proxima-tensor-gate.sh` (unmodified — does not yet carry a `q5k-int8-dot`/`q6k-int8-dot` cell, a gap named here, not silently left) — re-run clean on this commit after a `dead_code` fix (below).
- GEMM checksums, unaffected by this row (no shared code path changed except the `QuantizedBlock` enum and `run_reduce_quantized`'s dispatch, both proven equivalent for the `Q4K` arm by ROW 61's own unchanged tests): `512 4 1` -> `135.87619`, `1024 4 1` -> `260.24106`, `2048 4 1` -> `513.10425`.

**One real bug this row's own gate caught and fixed:** the first draft
left `REAL_OPENCHAT_GGUF_PATH` (a test-module constant) ungated, so a
default build (neither `q5k-int8-dot` nor `q6k-int8-dot` on) failed
`-D dead-code` — `cpu.rs`'s `default check`/`default tests`/`default
clippy` cells caught it immediately. Fixed by gating the constant
itself behind `any(feature = "q5k-int8-dot", feature = "q6k-int8-dot")`,
matching every usage site. Recorded, not silently squashed into the
first commit, per this skill's own "record the negative result" rule —
this was a real red gate on this tree, not a hypothetical one.

**A second, larger collision, also caught by re-running the gate, not by
inspection:** a concurrent agent's `run_reduce` parallel-dispatch change
landed in this SAME `cpu.rs` while this row's gate script was mid-run
(confirmed by `shasum` on the file diverging across three consecutive
60-120s checks before finally settling). A gate run captured DURING that
window showed `sync_channel`/`nest_pool` "not found in this scope" —
transient breakage from an in-progress edit, not this row's defect; the
file was re-checked once its hash stopped changing and came back clean
(`cargo check --all-targets --features q5k-int8-dot,q6k-int8-dot` exit
0, full nextest 312/2, default nextest 306/2, GEMM checksums unchanged —
all re-run post-merge, all numbers in this Gates section are the
POST-merge ones). No edit in this row touched `run_reduce`'s dispatch
loop itself; the two changes are disjoint regions of the same file.
Once the file stabilized, `cargo doc -p proxima-tensor --no-deps`
surfaced two intra-doc-link errors on `evaluate_parallel`'s doc comment
(`[`run_chunks_threaded`]`/`[`nest_pool`]` linking to private items) —
part of the OTHER agent's parallel-dispatch change, not this row's own
code, but blocking the shared `default rustdoc` gate cell for everyone
on this tree. Fixed with the identical one-line pattern this row's own
four analogous `dot_q5k_f32`/`dot_q6k_f32`/`dot_q5k_q8k`/`dot_q6k_q8k`
doc-link errors needed (bracket-link syntax on a private target ->
plain-backtick code span, no semantic change) — `cargo doc --no-deps`
(default features) and with `q4k-int8-dot,q5k-int8-dot,q6k-int8-dot`
both exit 0 after.

**Re-prove:** every command in the Gates section above runs from this
repo alone, no external asset beyond the real GGUF file at the fixed
path already required by `bench_q4k_matmul.rs` (`REAL_OPENCHAT_GGUF_PATH`
in `cpu.rs`'s test module, same path). The `evaluate_quantized_named`
binding path and `bind.rs` loader wiring for these 9 tensors are
NOT covered by this row — `proxima-model-interop/src/bind.rs` was
dirty (another agent's in-flight work) for this entire session; see the
task report for the exact patch description handed back instead of
applied.

**Not landed, named not buried:** AVX2 arm (deferred, same rationale ROW
62 gives); `ns/mac`/CoV/ggml-FFI-parity bench numbers (deferred to the
coordinator's planned joint bench pass); `bind.rs` rewiring (deferred,
file was contended).

## ROW 64 — int8 dot on packed Q6_K/Q8_K blocks: CORRECTNESS ONLY, no timing taken

Same landing, same session, same "no timing" scope as ROW 63 — read that
row's opening paragraph for why. `Q6_K` is a DIFFERENT super-block shape
from `Q4_K`/`Q5_K`, not a small variation: 16 sub-blocks of 16 (not 8 of
32), one signed 8-bit scale per sub-block, no `dmin` term at all
(`x = d*sc*(q-32)`, `proxima_gguf::quant::q6_k`'s own module doc — a
level bias, not a min-value subtraction).

`dot_q6k_q8k_block_scalar` does NOT reuse `Q4_K`/`Q5_K`'s
`get_scale_min_k4` unpack or `byte_base = (sub_block/2)*32` addressing —
neither applies to `Q6_K`'s genuinely different byte layout. Its
addressing (`half = sub_block/8`, `local_sub = sub_block%8`, `lane =
local_sub/2`, `subhalf = local_sub%2`) is derived from, and kept
consistent with,
`proxima_gguf::quant::q6_k::unpack_levels` — the already-tested reference
this crate ships (bit-exact-tested against a hand-packed fixture,
`proxima-gguf/src/quant/q6_k.rs`) — not re-derived from ggml's C
independently. The scalar kernel's per-element formula (`level =
nibble | (high << 4)`, `quant = level - 32`, dot against `Q8_K` `i8`
activations, scale by the sub-block's own signed `i8` code, sum) is the
exact same value `unpack_levels` plus a dot product would compute,
verified not by proof-reading alone but by the real-bytes test below.

`dot_q6k_q8k_block_neon_dotprod` ports `ggml_vec_dot_q6_K_q8_K`'s plain
`__ARM_NEON` arm (`arch/arm/quants.c:3001-3090`, the non-`__ARM_FEATURE_MATMUL_INT8`,
non-SVE path) with ONE deliberate simplification, recorded as a
mechanism change, not a silent deviation: ggml keeps levels unbiased
(`0..63`) through the dot and corrects for the `-32` bias afterward via
`bsums`/`isum_mins` (an optimization that avoids a per-lane subtract, at
the cost of decoding `y[i].bsums` and a second correction term); this
port applies the `-32` bias directly in-register via `vsubq_s8`
immediately after assembling each of the 8 `q6bytes` lanes per
super-block, then dots with no separate correction term at all — the
SAME value by a simpler, easier-to-verify derivation, one extra `vsubq_s8`
per lane (8 total) traded for never touching `bsums` in this kernel.
Chosen under this row's time budget: correct-and-simple over
matching ggml's exact instruction sequence.

**Feature gate:** `q6k-int8-dot`, new, compile-time, default-off — same
posture as ROW 63's `q5k-int8-dot`. `QuantizedBlock::Q6K(&[u8])` always
exists; `matmul_q6k_f32`/`dot_q6k_f32` (dequantize-then-fold via
`proxima_gguf::quant::q6_k::dequantize_block`) is the always-compiled
codec path. `Q6K_D_OFFSET` deliberately trails the block (`proxima_gguf::quant::q6_k`'s
own module doc: unlike `Q4_K`/`Q5_K`, `d` sits LAST in `Q6_K`'s on-disk
layout) — read from that module's doc, not re-derived by guessing a
layout that matched the other two codecs.

**Types minted: none.** Same posture as ROW 63.

**Correctness — the only claim this row makes:**

1. **Bit-exact, dispatched arm vs portable arm** (synthetic, 4 rows x 5
   super-blocks): `matmul_q6k_q8k_f32_agrees_bit_exact_with_the_portable_arm`
   — PASS.
2. **Packed int8 kernel vs dequantize-then-fold reference, on REAL packed
   `Q6_K` bytes** read directly out of `openchat-3.5-1210.Q4_K_S.gguf`:
   `output.weight` (4096x32002), this landing's named `Q6_K` shape —
   `matmul_q6k_q8k_f32_agrees_with_dequantize_then_fold_on_real_gguf_bytes`
   — PASS, relative_max_error < 0.01, dispatched/portable arms bit-exact
   against each other on the real bytes (same test).

No `evaluate_quantized` end-to-end test for `Q6K` in this row (ROW 63's
own e2e test already proves the enum-dispatch MACHINERY generically —
`run_reduce_quantized`'s `match weight_block` arm for `Q6K` is the same
shape as its `Q5K` arm, one match arm apart in `cpu.rs`; a dedicated
`Q6K` e2e test is a real gap, named here, not silently assumed covered).

**No ggml FFI parity number in this row** — same reason and same
follow-up plan as ROW 63 (`bench_q6k_matmul.rs` written and registered
behind `ggml-bench,q6k-int8-dot`, not run).

**Gates:** covered jointly with ROW 63 above (both features were always
built and tested together in this session) — see ROW 63's Gates section
for the exact commands and counts; nothing in this row's own gate run
diverged from that section.

**Re-prove:** same command set as ROW 63.

**Not landed, named not buried:** AVX2 arm; `ns/mac`/CoV/ggml-FFI-parity
bench numbers; `bind.rs` rewiring; a dedicated `Q6K` `evaluate_quantized`
e2e test (`Q5K`'s covers the dispatch machinery, not a `Q6K`-specific
regression).

## ROW 65 — cache `available_parallelism`; sweep workers 6/8/10 on the P+E hypothesis: 8 beats 10, MEASURED

**Repo:** this worktree, HEAD `388d93a`, tree clean before this row's edit.
**Host:** Apple M1 Max, macOS, 8 P-cores + 2 E-cores (`sysctl -n
hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu` -> `8` / `2`).
`std::thread::available_parallelism()` reports `10` (P+E summed, no tier
distinction).

### A. Cache `available_parallelism` — landed

Prior code called `thread::available_parallelism()` on every
`quantized_matmul_workers` invocation — 1350 calls per real forward pass,
each a `sysctl`. Added `matmul_worker_count()` (`cpu.rs:4319-4326`,
immediately above `quantized_matmul_workers`), a `OnceLock<usize>` resolved
once for the process lifetime. `quantized_matmul_workers` now calls it
instead of the raw `available_parallelism()`.

**Measured, before/after, same instrumented counter
(`MATMUL_AVAILABLE_PARALLELISM_NANOS`), same 1350-call forward pass:**

| | total over 1350 calls | per-call |
|---|---|---|
| before (uncached) | 4.768 ms | 3.53 us |
| after (cached, this row, mean of 9 runs) | 0.052 ms | 0.038 ns effective |

A ~98.9% reduction in that specific accounted cost, MEASURED via the
existing `instrument` counter, not estimated.

### B. Worker-count override — landed as override only, default unchanged

Added `PROXIMA_MATMUL_WORKERS` env override inside the same `OnceLock`
closure — read exactly once (never per-call: a per-call `std::env::var`
allocates a `String` 1350 times and would contaminate the very cost A
removes). Unset -> unchanged behavior (`available_parallelism()`).

### Hypothesis under test

`recv_wait_ms` (42.9 ms) + unattributed residual (59.85 ms) from the prior
session's decomposition is straggler-shaped; `available_parallelism()`
returning 10 (8P+2E) means every dispatch waits on the 2 slow E-cores.
Corroborating: llama.cpp on this same box measures FASTER at `-t 8` (150.1
ms) than `-t 10` (205.7 ms).

### Sweep: real forward, `PROXIMA_PREFAULT=1`, `--features std`, release,
`test-threads=1`, order alternated 10/8/6 x3 (9 runs total, driver:
`bind::real_openchat_file::runs_one_real_forward_pass_and_greedy_picks_a_real_token`)

**Token every run, every arm: `2651` / `"known"`. 9/9 `test result: ok. 1
passed`. EXIT=0 every run.**

**Host loadout (pasted next to every number, not summarized away):**
`uptime` load average 2.7-5.4 across the 9 runs (not a quiet box);
`mediaanalysisd` (a macOS system indexing daemon, single-process) at
52-95% CPU throughout, plus a local background service at 17-43%
intermittently.
Present identically across all three arms since order was alternated, so
drift is spread rather than confounding one arm.

| arm (workers) | forward wall (mean, 3 runs) | range | CoV | spawn_ms | recv_wait_ms | own_chunk_ms |
|---|---|---|---|---|---|---|
| **10 (default)** | 356.40 ms | 353.19-359.12 ms | 0.69% | 19.96 | 41.03 | 236.78 |
| **8 (override)** | **346.55 ms** | 341.13-350.65 ms | 1.15% | **16.42** | **36.57** | 236.01 |
| 6 (override) | 397.84 ms | 392.33-407.93 ms | 1.79% | 10.62 | 36.46 | 292.11 |

Delta, 8 vs 10: **-9.86 ms, -2.77%**, forward wall. Delta, 6 vs 10: +41.44
ms, +11.63% (slower — too few cores, `own_chunk_ms` rises 23% because each
worker does more per-row work).

**Per-shape `ns_per_mac`, mean of 3 runs each, from `DIAG q4k_shape_table`:**

| shape (rows x k) | w=10 | w=8 | w=6 |
|---|---|---|---|
| 1024x4096 | 0.018141 | **0.017952** | 0.019449 |
| 4096x4096 | 0.009654 | **0.008929** | 0.010061 |
| 4096x14336 | 0.007084 | **0.006849** | 0.007894 |
| 14336x4096 | 0.006154 | **0.006087** | 0.007286 |

`w=8` wins on **every one of the 4 shapes**, not just the aggregate —
12/12 individual runs (3 runs x 4 shapes) show `w=8 < w=10` on
`ns_per_mac`. This is the strongest evidence in this row: the aggregate
forward-wall delta (2.77%, ~2x the combined CoV) is corroborated by a
shape-level signal an order of magnitude more consistent than the noise
floor.

### Mechanism (why 8 wins, not just that it does)

`spawn_ms` drops 17.7% (19.96 -> 16.42 ms) and `recv_wait_ms` drops 10.9%
(41.03 -> 36.57 ms) going from 10 to 8 workers — fewer threads to spawn
per dispatch and fewer stragglers to wait on. `own_chunk_ms` (actual
compute) is FLAT between 10 and 8 (236.78 vs 236.01 ms) — the 2 extra
"workers" at `w=10` are the 2 E-cores, doing E-core-speed work that does
not move the compute-time needle but does cost coordination on every one
of 1350 dispatches. At `w=6`, coordination drops further (spawn 10.62,
recv_wait 36.46) but compute rises sharply (292.11 ms, +23.5% vs w=8)
because now only 6 P-cores worth of parallelism is available and the
per-worker row count grows — net loss.

**Result: the hypothesis holds. `available_parallelism()`'s 10 (8P+2E) is
measurably worse than the P-core count (8) on this SoC — not because 8
cores are individually faster, but because dispatching to the 2 E-cores
adds coordination overhead (spawn+recv_wait) without adding usable
compute.**

### Selection rule — data only, no rule landed this row per the task's
instruction

The right rule is P-core count, not a hardcoded 8. `hw.perflevel0.logicalcpu`
is available via `sysctlbyname` and IS what this data says to use.
Checked both crates already in the dependency graph:

- **`rustix` 1.1.x (workspace dep, `Cargo.toml:196`):** grepped the vendored
  source (`~/.cargo/registry/.../rustix-1.1.4/`) for `perflevel` and
  `sysctlbyname` — zero matches. Rustix does not expose Apple's
  `sysctlbyname`-by-name surface; it is not the mechanism.
- **`libc` 0.2 (already a `proxima-tensor` dependency, `Cargo.toml:87,109`,
  optional/std-gated):** `libc::sysctlbyname` exists on the Apple target
  (`unix/bsd/apple/mod.rs:4409`, `extern "C"`). This is the mechanism: an
  unsafe `sysctlbyname(c"hw.perflevel0.logicalcpu", ...)` call, parsed as
  `u32`, would need to live behind the same `std`-gated surface
  `available_parallelism()` already requires (this file's `use std::thread`
  at `cpu.rs:105` is already std-only). Not implemented this row — the task
  asked for data, not a landed rule.

### Gates

- `cargo nextest run -p proxima-tensor --no-fail-fast`: **328 tests run,
  328 passed, 3 skipped.** Command re-proves as-is.
- `cargo clippy -p proxima-tensor --all-targets -- -D warnings`: clean,
  exit 0.
- `bash scripts/proxima-tensor-gate.sh`: **19 passed, 0 failed.**

**Re-prove command (this row's forward-wall and per-shape numbers):**
```
PROXIMA_PREFAULT=1 PROXIMA_MATMUL_WORKERS=8 cargo test -p proxima-model-interop \
  --release --lib --features std -- --ignored --exact \
  bind::real_openchat_file::runs_one_real_forward_pass_and_greedy_picks_a_real_token \
  --nocapture --test-threads=1
```
Swap `PROXIMA_MATMUL_WORKERS` for `10`/`6`/unset to reproduce the other
arms. Token (`2651`/`"known"`), `forward_wall_clock`, `matmul_dispatch`
line, and `q4k_shape_table` all print to stdout — same artifact this row's
numbers were read from, nothing paraphrased.

### Bar

llama.cpp forward on this box: **205.7 ms** (`-t 8`, prior session,
external instrument, not re-measured this row — flagged DERIVED, not
re-verified here). Best arm this row (`w=8`, 346.55 ms mean): **1.69x
slower than llama.cpp.** `w=10` (prior default): 1.73x slower. The
2.77% win narrows the gap but does not close it — the residual 33.5%
"unattributed straggler tail" from the prior session's decomposition is
partially explained (E-core dispatch cost) but not eliminated; `own_chunk_ms`
at `w=8` (236.01 ms) is still the dominant single line and unmoved by this
row's change.

**Not landed, named not buried:** the P-core-count selection rule itself
(data says use it; owner decides); an `sysctlbyname`-based
`hw.perflevel0.logicalcpu` read (would need its own log row: allocation
budget, error path for non-Apple targets, and a compile-time gate per
platform); a rebench of the llama.cpp `-t 8` bar in this same session
(carried forward from the prior session as DERIVED).

### Changelog
| Date | Change | Δ vs prior | CoV / runs | Host loadout |
|---|---|---|---|---|
| 2026-08-20 | cache `available_parallelism` in `OnceLock` | -98.9% on `MATMUL_AVAILABLE_PARALLELISM_NANOS` (4.768ms -> 0.052ms over 1350 calls) | measured every run, 9/9 | load avg 2.7-5.4, mediaanalysisd 52-95% |
| 2026-08-20 | worker override, swept 6/8/10 | **8: -2.77% forward wall vs 10 (KEPT as override, default unchanged). 6: +11.63% vs 10 (documented, not landed as default)** | 0.69-1.79% CoV, 3 runs/arm | same as above |

## ROW 66 — `OperandSpan` carries a stride, not a `contiguous` flag; the elementwise `Generic` gate widens, the reduce/scan gate does NOT

**Repo:** worktree `agent-a585f8281e4851e98`, branch `feat/tensor-consolidated`,
tree dirty across nine unrelated efforts before this row's edit (nothing was
reverted or stashed).
**Host:** Apple M1 Max, macOS, 10 logical cores, 64 GiB.
**Driver:** `bind::real_openchat_file::runs_a_cached_greedy_decode_loop_and_reports_per_token_wall_clock`,
`--release --features std,instrument`, `PROXIMA_PREFAULT=1`,
`PROXIMA_MAX_TOKENS=24`, openchat-3.5-1210 Q4_K_S. Generated text held the
required prefix ("Here is a simple Python function that returns the nth
Fibonacci number using recursion:") on every run of every arm.

### The defect

`OperandSpan` stored `contiguous: bool`, computed as `stride == 1`. That is a
rich-to-poor boundary: an operand's real width-dim stride was thrown away at
span construction, so `operand_is_affine` could only admit the two strides the
flag could still represent (`0 | 1`). RoPE binds stride 2
(`specs/rope.toml:42,50,67`, `s,2*i->si` / `s,2*i+1->si`), so every fused RoPE
body failed `generic_body_is_affine_fast_path` and fell to the per-element
interpreter.

**Measured before, per decode step, `instrument` counters:**

| path | elements | ms | ns/element |
|---|---|---|---|
| `elementwise` Generic fast | 944,128 | 2.18 | 2.31 |
| `elementwise` Generic slow | 163,840 | 2.65 | 16.18 |

14.8% of Generic elements were taking 55% of its wall time.

### The change

`OperandSpan { data, base, stride: usize }`. `1` reads a real subslice, `0`
reads `data[base]` once and broadcasts, anything else walks
`base + position * stride`. Every kernel that reads a span takes a
once-per-call `is_strided()` early exit into a `*_strided` sibling; the
existing stride-0/1 arms are byte-for-byte unchanged, and no cartesian match
was widened. The six `*_scalar_dispatch` fallbacks got SHORTER — their
per-element `if contiguous { .. } else { .. }` collapsed into `span.at(step)`.
Strided arms fold strictly left-to-right, never through
`dot_fold_multi_accumulator_*`, so a body that moves off the interpreter keeps
the interpreter's exact arithmetic order.

### The two gates DISAGREE, and the disagreement is a measurement, not an oversight

The first attempt widened the shared `operand_is_affine` to
`stride >= 0`, which widened BOTH gates. `n=3`, A/B/A/B interleaved, quiet box
(load 0.8-2.6):

| | narrow both | wide both |
|---|---|---|
| prefill `reduce_f32_dense` | 81.0 ms | **180.6 ms** |
| decode `reduce_f32_dense` /step | 3.99 ms | **7.93 ms** |
| decode `elementwise` /step | 7.88 ms | 5.37 ms |

The elementwise win was real and the reduce regression was larger. Mechanism:
`run_elementwise`'s alternative to its fast path is the per-element
interpreter at 16.2 ns/element, so a strided width walk at ~2.3 ns/element
wins. `run_reduce`'s alternative is NOT an interpreter — it is
`reduce_dot_fast`'s contraction-dim path and its NEON tile, a FASTER kernel.
Admitting stride > 1 stole those nodes into a scalar width walk. So
`body_shape_is_affine_fast_path` keeps `operand_is_unit_or_broadcast`
(`stride <= 1`) and only `generic_body_is_affine_fast_path` uses the widened
`operand_is_affine`. The strided arms in the reduce/scan kernels stay: they
are correct, unit-tested for bit-identity, and are what a future widening of
that gate would need — but nothing in production reaches them today.

### Result, n=6 per arm, A/B/A/B interleaved in one window

A = narrow both gates. B = widened elementwise `Generic` gate only. Box under
concurrent load (1-min avg 4.1 rising to 14.8) — both arms saw the same load,
and the `elementwise` effect is far outside the spread.

| metric | A median | A min | A range | B median | B min | B range |
|---|---|---|---|---|---|---|
| decode `elementwise` ms/step | 8.200 | 7.995 | 7.995-9.365 | **5.594** | **5.407** | 5.407-6.688 |
| decode `reduce_f32_dense` ms/step | 3.994 | 3.967 | 3.967-4.198 | 4.032 | 3.935 | 3.935-4.113 |
| prefill `elementwise` ms | 47.60 | 39.2 | 39.2-56.4 | **32.35** | **29.2** | 29.2-73.8 |
| prefill `reduce_f32_dense` ms | 79.10 | 78.3 | 78.3-80.3 | 79.20 | 78.3 | 78.3-80.3 |
| 24-token wall ms | 2665.4 | 2607.2 | 2607-3266 | 2591.7 | 2451.8 | 2452-2906 |

Deterministic counter, every run, no spread at all:

| per decode step | A | B |
|---|---|---|
| `ELEMENTWISE_ELEMENTS_GENERIC_SLOW` | 163,840 | **0** |
| `ELEMENTWISE_ELEMENTS_GENERIC_FAST` | 944,128 | **1,107,968** |

`944,128 + 163,840 == 1,107,968` exactly: every element that was on the
interpreter moved to the fast path, and no element was created or lost.

`elementwise` decode is -2.606 ms/step at the median, -2.588 ms/step
min-vs-min. Predicted from the counters was 163,840 x (16.18 - 2.31) ns =
2.27 ms/step; measured is slightly better because the whole node also stops
paying `operand_values`/`gather_cursors` bookkeeping.

### What this row does NOT claim

The 24-token wall delta (-73.7 ms median, -155.4 ms min-vs-min) is 2.8% on a
number whose A-arm range spans 25%; `reduce_matmul_quantized` dominates the
step and moves by more than this between runs. The `elementwise` node total
and the `GENERIC_SLOW` counter are the two numbers this row stands on. No
claim is made about the llama.cpp ratio from these runs.

### Gates

`proxima-tensor --features std` 350/350 - `--features std,instrument` 354/354
(345/349 before this row; +5 new bit-identity tests) - `prime --features
runtime-prime-cohort` 156/156 - `proxima-primitives` 413/413 -
`proxima-model-interop --features std` 24/24 - `clippy -p proxima-tensor
--features std,instrument --all-targets` 0 warnings.

## ROW 67 — head-to-head vs llama.cpp, re-measured; and the matmul ceiling is 26.5 ms, not 39.6

**Repo:** worktree `agent-a585f8281e4851e98`, branch `feat/tensor-consolidated`,
HEAD `65fbb8e` for the head-to-head, `8ea6015` for the bucket split.
**Host:** Apple M1 Max, 10 logical cores, 64 GiB. Background load is the
desktop (WindowServer + iTerm ~0.8 core), not killable; noted per run.
**Incumbent:** llama.cpp b2534622 (build 5761),
`/Users/brianbruggeman/repos/others/llama.cpp/bin/llama-cli`.
Its fast CPU path is verified ON, not assumed:
`NEON=1 ARM_FMA=1 FP16_VA=1 DOTPROD=1 LLAMAFILE=1 ACCELERATE=1 REPACK=1`,
and `load_tensors: offloaded 0/33 layers to GPU` — CPU-to-CPU, no Metal.

**Both sides, same everything:** same
`openchat-3.5-1210.Q4_K_S.gguf`, same prompt (31 tokens on both — our
prefill elementwise call sizes are exact multiples of 31: 961, 992, 1984,
3844, 7936, 15872), 24 tokens generated, greedy. **Both emit byte-identical
text** ("Here is a simple Python function that returns the nth Fibonacci
number using recursion:"), so the incumbent is doing our work, not a
cheaper one.

### A to B, n=18 per arm, three arms interleaved in ONE window

| decode, ms/token | min | median | max | spread |
|---|---|---|---|---|
| llama.cpp `-t 8` | **39.60** | **49.00** | 106.74 | **170%** |
| proxima `w=8` | 60.57 | 66.29 | 77.97 | 29% |
| proxima `w=10` | 61.11 | 66.32 | 72.00 | 18% |

| prefill, ms | min | median | max | spread |
|---|---|---|---|---|
| llama.cpp `-t 8` | **725.72** | **806.17** | 983.37 | 36% |
| proxima `w=8` | 935.10 | 1021.35 | 1305.16 | 40% |
| proxima `w=10` | 940.58 | 1070.12 | 1393.06 | 48% |

**The two estimators disagree and the row does not pick the flattering one.**
decode min-vs-min **1.529x behind**; decode median-vs-median **1.353x**.
prefill min-vs-min **1.289x**; median-vs-median **1.267x**. llama's decode
spread is 170% against our 18-29%: it is far more sensitive to the desktop
interference, so its median is inflated by its own tail and min-vs-min is
the better estimate of a quiet-box gap. Against the prior standing figures
(prefill 1.233x, decode 1.404x) the median says decode improved and the min
says it regressed. **This box cannot resolve which.** The prior log's
llama bar (205.7 ms, flagged DERIVED, never re-verified) is superseded.

### Is the matmul bucket kernel, or orchestration? MEASURED: kernel.

`matmul_split` counters, `PROXIMA_MAX_TOKENS=24` minus `=1`, divided by the
23 decode steps that separates them (quiet box, load 1.01, n=3):

| per decode step | ms | share of bucket |
|---|---|---|
| bucket (`reduce_quantized`) | 46.16 | 100% |
| packed int8 kernel (q4k+q5k+q6k calls) | **45.06** | **97.6%** |
| quantize activation to Q8_K | 1.82 | 3.9% |
| q4k transpose | 0.98 | 2.1% |
| dispatch setup | 0.23 | 0.5% |
| spawn / recv_wait | 0.00 / 0.00 | 0% |
| caller's own chunk | 42.52 | — (2.54 ms, 5.6%, waiting on cohort stragglers) |

So the bucket is not hiding dispatch. It is the kernel. Separately, the arm
is proven by execution witness, not by reading the feature flags:
`q4k_macs=363,293,835,264` (incremented only inside the packed branch),
`q5k_f32_calls=0`, `q6k_f32_calls=0` — the dequantize-then-fold codec never
ran. `q4k_ns_per_mac=0.00500` against the crate's own 0.2534 for
dequantize-then-fold.

### The correction: our own overhead sets the matmul ceiling

Full decode step, quiet box, `w=8`, n=3, median run — this accounts for
100% of the step, nothing residual:

| | ms/step | % of our step | % of llama's 39.60 ms token |
|---|---|---|---|
| matmul bucket | 49.881 | 79.2% | 126% |
| elementwise | 5.494 | 8.7% | 13.9% |
| `reduce_f32_dense` | 3.972 | 6.3% | 10.0% |
| setup + loop overhead | 3.614 | 5.7% | 9.1% |
| **total** | **62.961** | 100% | 159% |

**Non-matmul = 13.08 ms/step = 33% of the incumbent's ENTIRE token budget.**

The previous framing ("our matmul alone, 47.78 ms, is 1.21x llama's whole
39.60 ms token") was the wrong comparison: it silently gave our own 13.08 ms
of overhead a free pass. Corrected —

- matmul budget to reach llama's **min**: 39.60 - 13.08 = **26.52 ms**.
  We are at 49.88. Required speedup on the matmul: **1.88x**, not 1.21x.
- matmul budget to reach llama's **median**: 49.00 - 13.08 = 35.92 ms.
  Required: **1.39x**.

And neither term alone reaches it: zeroing ALL non-matmul work still leaves
49.88 ms against a 39.60 ms bar (1.26x behind). Both must move.

### Where the non-matmul 13.08 ms actually is — measured, not derived

The elementwise kernel time is instrumented directly:
`fast_e=1,107,968 fast_ms=2.443`. Against a 5.494 ms elementwise node total,
**3.05 ms/step of elementwise cost is OUTSIDE the kernel** — per-node setup,
`step_values` allocation, dispatch. `reduce_f32_dense` by contrast is
essentially all arithmetic (11.6M elements x 0.34 ns = 3.95 ms vs a 3.972 ms
node total). Adding the 3.614 ms of graph setup + loop overhead:

**~6.7 ms/step is neither matmul nor arithmetic. That is 17% of the
incumbent's entire token budget, spent on graph execution.**

A forward resolves 1196 nodes (225 `reduce_matmul_quantized`, 385
`reduce_f32_dense`, 547 `elementwise`, 37 `constant`, 2 `iota`). 971 of them
are non-matmul.

### What this says about ROW 66

ROW 66's 2.6 ms/step elementwise win is real and independently confirmed
here (the whole elementwise node total is 5.494 ms with the slow-path
element count at 0). It attacked the right region at the wrong level: the
remaining elementwise cost is 2.44 ms of kernel against 3.05 ms of per-node
overhead. Fusing the step chain (the ~1.86 ms lever) targets the 2.44; the
3.05 needs the node count or the per-node cost to fall, which is a graph
question, not a kernel one.

### DERIVED, labelled as such, not measured here

ROW 61 measured our packed kernel at 1.29-1.40x behind ggml t1 on isolated
shapes. If that ratio transfers to this forward, llama's matmul is
49.88/1.40 to 49.88/1.29 = 35.6-38.7 ms, leaving it 0.9-4.0 ms of
non-matmul against our 13.08 — a 3-14x overhead gap. This rests on ROW 61's
ratio transferring from isolated shapes to a real forward, which has not
been shown. It is the hypothesis the next row should test directly, by
instrumenting llama.cpp's own op-level split rather than inferring it.

### Re-prove

```
# incumbent
llama-cli -m openchat-3.5-1210.Q4_K_S.gguf -ngl 0 -t 8 -n 24 \
  --temp 0 --top-k 1 -no-cnv --seed 1 -p "<the 31-token prompt>"
# ours
PROXIMA_PREFAULT=1 PROXIMA_MAX_TOKENS=24 PROXIMA_MATMUL_WORKERS=8 \
  <test-bin> --exact --nocapture --ignored \
  bind::real_openchat_file::runs_a_cached_greedy_decode_loop_and_reports_per_token_wall_clock
```
`matmul_split` and `quant_arm` print to stdout; subtract a
`PROXIMA_MAX_TOKENS=1` run and divide by 23 to isolate decode.

## ROW 68 — the kernel is NOT the gap: at 1 thread we equal llama.cpp exactly. Two fixes tried, both REFUTED, root cause is per-round dispatch cost

**Host:** Apple M1 Max, 10 cores. **Incumbent:** llama.cpp b2534622, `-ngl 0`,
same fixture/prompt/24 tokens, byte-identical output.
**Repo:** `feat/tensor-consolidated` @ `34c7a40`.

### Read the incumbent's source before optimizing against it — two hypotheses died

- **`REPACK=1` does not repack Q4_K on this box.** `ggml/src/ggml-cpu/repack.cpp:1444`
  gates Q4_K repack on `ggml_cpu_has_avx2()`; aarch64 is false. It applies to
  Q4_0/IQ4_NL only.
- **ggml's 2-row Q4_K dot never runs here.** `arch/arm/quants.c:2149` puts the
  `nrc == 2` path behind `__ARM_FEATURE_MATMUL_INT8` (i8mm). M1 has dotprod,
  not i8mm. And `ggml-cpu.c:1372` sets `num_rows_per_vec_dot = 1` for decode.

So llama runs the SAME one-row `vdotq_s32` kernel we do. Reading both bodies
(`arch/arm/quants.c:2370-2431` vs `cpu.rs:6332-6391`) they are the same
16 `sdot` + 8 `vaddvq` + `bsums` mins-correction, hand-unrolled the same way.

### The scaling curves settle it — decode ms/token

| threads | llama.cpp **whole token** | ours **matmul only** | ours whole step |
|---|---|---|---|
| 1 | 177.70 | **178.1** | 187.6 |
| 2 | 98.15 | **96.4** | 107.0 |
| 4 | 56.85 | 64.96 | 76.06 |
| 8 | 47.99 | 49.52 | 61.37 |

**At 1 thread our matmul and llama's ENTIRE token are the same number.** At 2
threads we are faster. The kernel is not the problem and no amount of kernel
work will close this. The gap opens only as threads are added, and it lands
almost entirely outside the matmul:

at t=8, step 61.37 vs 47.99 = 13.38 ms gap — matmul 49.52 vs their ~46
(3.5 ms), non-matmul 11.85 vs their ~2 (**9.85 ms, 74% of the gap**).

**Our non-matmul does not scale at all**, measured across the same sweep:
`reduce_f32_dense` 3.859 ms at 1 worker, 3.952 at 8. `elementwise` 2.644 at 1
worker, **4.144 at 8** — it gets 1.5 ms WORSE, the parked matmul workers
interfering with a phase that is entirely serial. Cause: `cpu.rs:2225`
returns to the sequential path when `outer_len < 2`, and decode has exactly
one outer position, so every one of the 547 elementwise and 385 f32 nodes
runs single-threaded.

### FIX 1 — stop the workers parking. REFUTED, 20-33% WORSE.

`PROXIMA_COHORT_SPIN_POLLS`, n=3 per arm, interleaved:

| spin_polls | parks/round | matmul ms/step | step ms |
|---|---|---|---|
| 2000 (default) | 6.5 | **48.7 / 48.3 / 49.6** | **60.5 / 60.1 / 61.5** |
| 200,000 | 0.10 | 69.0 / 64.3 / 61.2 | 84.3 / 79.4 / 75.4 |
| 5,000,000 | 0.00 | 65.6 / 59.8 / 58.4 | 80.5 / 74.8 / 71.6 |

Decode is bandwidth-bound; idle members spinning contend for the bus and cost
more than the park they avoid. Parking is the CHEAP option. The wake ramp
visible in `cohort_slot` (first claim 9.6 us at slot 0 rising to 22.6 us at
slot 7, ~34 us tail per slot, slot 7 present in only 5701 of 6972 rounds) is
therefore NOT recoverable by spinning. Do not re-run this sweep.

### FIX 2 — split the width axis so decode nodes parallelize. REFUTED, elementwise 2x WORSE.

Implemented and gated green (354/354, clippy clean): `ElementwiseRowRound`
gained a second chunking axis, splitting a single row's width when the outer
axis has nothing to give. n=4 per arm, interleaved:

| | step ms | elementwise ms | matmul ms |
|---|---|---|---|
| before | 57.21 / 56.75 / 57.22 / 57.10 | **3.99 / 3.98 / 4.00 / 4.03** | 45.61 / 45.09 / 45.56 / 45.48 |
| width-split | 60.04 / 58.87 / 60.95 / 60.75 | **7.87 / 7.10 / 8.59 / 8.45** | 44.50 / 44.00 / 44.67 / 44.63 |

Reverted. A decode elementwise node holds ~33 us of work (14336 elements at
2.3 ns); opening a cohort round costs ~25 us. Splitting cannot pay at this
granularity no matter where the threshold is set.

### Root cause, stated once

Two different execution models. ggml runs the WHOLE graph on a persistent
team with a cheap spin barrier between nodes — every node, including every
elementwise op, is split across all threads, and there is no per-node
open/close. We run the graph on the main thread and open one cohort round per
large node, so (a) small nodes cannot be parallelized because the round costs
more than the work, and (b) between rounds the workers have nothing to do,
park, and pay a 9.6-22.6 us wake ramp on the next round.

Both refutations above are consequences of that shape, not independent bugs.
The fix is the execution model, not a knob and not a threshold: threads must
own the whole graph. Nothing smaller was found that moves it, and two things
that looked like they would, measurably do not.

### Standing, n=12 one window, `34c7a40`

decode 59.83 vs 44.09 ms/token (**1.357x**); prefill 976.81 vs 749.81 ms
(**1.303x**); 24-token wall 2385.33 vs 1767.74 ms (**1.349x**). Our spread
2-6%, llama's 13-29%.

## ROW 69 — the Metal gap is ARCHITECTURE, not tuning: 143x, and ROW 13's lesson recurred on the other backend

**Host:** Apple M1 Max. **Incumbent:** llama.cpp b2534622, `-ngl 99`.
**Repo:** `feat/tensor-consolidated` @ `c5aac28`.

### Measured first

| | bytes read | median | elem/s | GB/s |
|---|---|---|---|---|
| omega, **f32** weight, 4096x4096 | 67.1 MB | 2.570 ms | 6.53 G | **26.1** |
| omega, **packed Q4_K**, same space | 9.44 MB | 6.273 ms | 2.67 G | **1.5** |
| llama.cpp Metal, 7B Q4_K_S decode | 3.784 GB/token | 17.62 ms | — | **214.7** |

Two separate facts. Packed is **2.4x SLOWER than f32 while reading 7x fewer
bytes** — the unpack costs more than the traffic it saves. And the plain f32
kernel is already **8x** off the incumbent with no quantization involved,
so Q4_K is not the problem.

Correctness is not in question: the packed path matches a dequantized-f32
CPU oracle to **9.4e-7** relative (`metal_parity.rs`).

### Four peephole hypotheses, all measured, all essentially zero

| change | before -> after | verdict |
|---|---|---|
| MSL recompiled per `execute` call (was true) | 8.18 -> 6.82 ms | not the cost |
| per-element 64-bit `%` and `/` on a runtime extent (was true) | 6.39 -> 6.20 ms | not the cost |
| 64-bit walk offsets in the inner loop (was true) | 6.20 -> ~6.2 ms | not the cost |
| weights re-uploaded per call | counters: `nocopy=66 copying=66` over 66 calls — the 9.44 MB weight is no-copy | never true |

Every one of those was a real defect and every one was irrelevant. That is
the signature of an architecture gap, and it is what finally forced the
right question.

### What the incumbent's kernel actually is (`ggml-metal.metal:5087`)

`kernel_mul_mv_q4_K_f32_impl`, with `N_R0_Q4_K 4` / `N_SG_Q4_K 2`
(`ggml-metal-impl.h:32-33`):

| | ggml | omega |
|---|---|---|
| output rows per thread | **4** (`float sumf[nr0]`); 2 simdgroups/tg = 8 rows/threadgroup | **1** |
| activation | `float yl[16]; yh[16]` in REGISTERS, loaded once per super-block, reused across all 4 rows | re-read from device per (row, element) |
| nibble extract | **4 nibbles per `uint16_t` load**, shift folded into `1/256` and `1/16` constants at the end | one nibble/element, explicit shift+mask |
| scale/min decode | once per super-block per row (3 `uint16_t` loads -> `sc16[0..3]`) | **per element** |
| `d`/`dmin` | once per super-block per row | 2 f16 loads + converts **per element** |
| `dmin` correction | via `sumy`, accumulated during the register load: 4 MACs per super-block | applied **per element** |
| inner-loop indexing | none; pointers step `q1 += args.nb01/2` per row | offset add per element |

Roughly **1.6 ops per element against ~40**. That is the 143x, derived from
the source, not fitted to the measurement.

### The architecture, stated plainly

`ggml-metal` is a **hand-written kernel library with a dispatcher**: 121
`kernel void` entry points in the `.metal`, 1134 kernel-type references in
`ggml-metal.m`, selected per (op x dtype), each carrying its own register
blocking constants, its own memory staging, and its own dispatch geometry.

`omega` is a **generic code generator**: one `emit(BoundOp)`, one thread per
output element, every operand read an affine index map evaluated per
element. `BoundOp` carries extents and layouts — it describes WHAT, and each
backend improvises HOW.

Those are not two points on one axis. Register blocking, activation reuse,
and per-super-block amortization are all statements about a SCHEDULE, and
there is no schedule anywhere in this IR to state them in.

### This is the same finding as the CPU one

ROW 68: at ONE thread our CPU matmul equals llama exactly (178.1 vs 177.70
ms/token) and the entire gap opens with threading — per-node cohort rounds,
no persistent team. Same shape here: the arithmetic is right (bit-exact
unpack, 9.4e-7 parity) and the structure around it is wrong. Both backends
are missing the same thing, and it is not a kernel.

### Recurrence against a lesson this log already carries

ROW 13 recorded: *"reading an incumbent's loop nest is not reading its
kernel. The register allocation IS the design, and it lives in the
accumulator's TYPE, not in the loop structure."* ggml's design is literally
`float sumf[nr0]`.

That lesson was learned on the aarch64 CPU kernel and not applied to the
Metal one. Four peephole rows were spent before reading
`ggml-metal.metal`. The generalization the log should have carried, and now
does: **read the incumbent's kernel STRUCTURE before writing any
optimization against it, on every backend separately — a lesson recorded
for one target is not transferred to another for free.**

## ROW 70 — RETRACTION of ROW 69's conclusion. The Metal gap is 2.0x, not 143x; the cost was our own per-call buffer wiring

**Host:** Apple M1 Max. **Repo:** `feat/tensor-consolidated`.

### What ROW 69 got wrong, and how

ROW 69 concluded "the Metal gap is ARCHITECTURE, not tuning: 143x". That
conclusion is **retracted**. It was reached by a tautology: four peephole
fixes each measured ~zero, therefore the cause must be architectural. That
is fitting a conclusion to the shape of my own failures, not deriving it
from evidence. A fix measuring zero is evidence about that fix and nothing
else.

Worse, ROW 69's own probe printed `every execute re-uploads every block`,
and I dismissed it after reading `nocopy=66 copying=66` — reasoning that a
"no-copy" wrapper is free. It is not.
`newBufferWithBytesNoCopy` creates a fresh `MTLBuffer` and Metal must wire
those pages for GPU access on every call, a cost that scales with BYTES.
That is exactly why it hid inside a bytes-normalized metric, and why BOTH
arms of the f32-vs-packed control paid it. **The control was confounded and
the kernel was never measured.**

Second confound: the probe reported a MEDIAN while a sibling process on
this box ran at 171.9% CPU. Under interference the median tracks the
interferer. The probe now reports min-of-N.

### Corrected measurement — same probe, buffers reused, min of 21

| | ROW 69 (median, per-call wiring) | corrected (min, wiring removed) | speedup |
|---|---|---|---|
| packed Q4_K 4096x4096 | 6.273 ms | **1.380 ms** | 4.5x |
| f32 control, same space | 2.570 ms | **0.623 ms** | 4.1x |
| f32 achieved bandwidth | 26.1 GB/s | **107.7 GB/s** | 4.1x |

Reuse witness: `nocopy_attempts=66 of which REUSED=63 (so 3 real wires)` —
three distinct weight buffers wired once each instead of 66 times.

### Where it actually stands

| | achieved | bar (llama.cpp Metal, 214.7 GB/s) |
|---|---|---|
| omega f32 kernel | **107.7 GB/s** | **2.0x off** |
| omega packed Q4_K | 12.16 G elem/s vs f32's 26.93 | **2.2x slower per element than f32** |

Two live, ordinary defects — not an architecture wall:

1. **2.0x on the f32 kernel.** Real, and the register-blocking comparison in
   ROW 69's table is still the best lead: ggml carries 4 output rows per
   thread (`float sumf[nr0]`, `N_R0_Q4_K 4`) and holds the activation in
   registers across those rows; omega does one row per simdgroup and
   re-reads the activation per element.
2. **Packed is 2.2x slower per element than f32.** `q4k_element` decodes
   `d`, `dmin`, and the 6-bit scale/min per ELEMENT where ggml decodes them
   once per super-block. Still true, still worth fixing, and now correctly
   sized as ~2x rather than as the whole gap.

### The unsound edge this fix currently carries

The buffer cache keys on `(pointer, len)` and `newBufferWithBytesNoCopy`
does not own the memory. If a caller drops the backing allocation between
calls, the cached wrapper aliases freed pages. That holds fine for mmap'd
GGUF weights (the case this exists for) and NOT in general. The sound
version is a resident-blocks handle whose lifetime borrows the caller's
data, which is also the API a serving loop wants; until then the precondition
is documented on the cache itself and is a caller obligation.

### Method note, which is the actual lesson

ROW 69 spent four rows peepholing before reading `ggml-metal.metal`, then
concluded "architecture" from the peepholes' failure. Both halves were
wrong. Read the incumbent's kernel structure FIRST — and when a fix measures
zero, that is a fact about the fix, never a syllogism about the cause.

## ROW 71 — dissection: THREE defects, correctly sized. f32 kernel 1.42x, packed kernel 3.9x, and 0.19-0.40 ms of fixed cost PER `execute`

**Host:** Apple M1 Max, load 3.98. **Method:** two problem sizes per arm,
min of 21, so the per-call fixed cost cancels in the difference and the
slope is the kernel. Single-size numbers cannot separate the two, which is
what produced ROW 69's and ROW 70's wrong sizings.

### Measured

| arm | small | large | MARGINAL bandwidth | fixed-cost intercept |
|---|---|---|---|---|
| f32 | 0.302 ms (16.8 MB) | 0.635 ms (67.1 MB) | **151.2 GB/s** | **0.191 ms** |
| packed Q4_K | 0.444 ms (2.36 MB) | 0.574 ms (9.44 MB) | **54.3 GB/s** | **0.400 ms** |

Per ELEMENT on the margin (12.58M elements either way):

| arm | marginal element rate |
|---|---|
| f32 | 37.8 G elem/s |
| packed Q4_K | **96.8 G elem/s** |
| llama.cpp Metal (3.784 GB / 17.62 ms at 0.5625 B/weight) | **381 G elem/s** |

### The three defects, in the order their size says to fix them

1. **0.19-0.40 ms of fixed cost per `execute` call.** A real forward is 1196
   nodes; at one `execute` per node that is 228-478 ms per forward of pure
   overhead, against llama.cpp Metal's 17.62 ms for the WHOLE token. This is
   the serving-path killer and nothing else comes close. It is
   `prepare` (infer + bind) re-run per call, an output buffer and a uniforms
   buffer allocated per op per call, a command buffer per call, a
   `waitUntilCompleted` full GPU sync per call, and a readback per call.
2. **Packed kernel 3.9x off on element rate** (96.8 vs 381 G elem/s). THIS is
   where ggml's register blocking actually pays: `float sumf[nr0]` with
   `N_R0_Q4_K 4`, activation held in registers across those 4 rows, and
   `d`/`dmin`/scale-min decoded once per super-block instead of per element.
   The ROW 69 structural comparison was right about the mechanism and wrong
   about the magnitude.
3. **f32 kernel 1.42x off on bandwidth** (151.2 vs 214.7 GB/s). Smallest of
   the three, and plausibly the same register-blocking lead.

### Corrections to ROW 70

ROW 70 reported "packed is 2.2x slower per element than f32". That was also
a fixed-cost artifact: packed carries the LARGER intercept (0.400 vs 0.191
ms) because its per-call work is the same while its bytes are 4x fewer, so a
single-size comparison charged the intercept to the kernel. On the margin
packed is **2.6x FASTER per element** than f32, exactly as it should be for
a bandwidth-bound sweep reading 4x fewer bytes. The packed read path is
doing the right thing; it is simply not yet bandwidth-saturated.

### Method, now three rows deep

ROW 69 concluded from failed peepholes. ROW 70 corrected the magnitude but
still compared single-size numbers and mis-sized defect 2 in the opposite
direction. Only two sizes per arm separated kernel from overhead. **A
performance number taken at one problem size is not a kernel measurement —
it is a kernel measurement plus an unknown intercept, and the intercept has
now been the dominant term twice.**

## ROW 72 — Q4 was COMPUTE-bound, not bandwidth-bound. Super-block tiling: 5x on the packed kernel, 17x gap -> 3.5x

**Host:** Apple M1 Max. **Method:** two problem sizes per arm so the per-call
intercept cancels; sizes raised to 9.44/37.75 MB (packed) and 67.1/268.4 MB
(f32) so the KERNEL dominates that intercept — at the previous sizes both
arms ran in ~0.3 ms against a ~0.25 ms intercept and the marginal figure
swung 3x between runs.

### The question that found it

Q4_K reads 0.5625 bytes/weight against f32's 4.0 — 7.1x less traffic. On a
bandwidth-bound sweep it should have been several times FASTER per element.
It was 2.8x slower:

| | bytes/element | marginal | elements/s |
|---|---|---|---|
| f32 | 4.0 | 242 GB/s | 60.5 G |
| packed Q4_K (before) | 0.5625 | 12.3 GB/s | **21.9 G** |
| llama.cpp Metal Q4_K | 0.5625 | 214.7 GB/s | **381 G** |

Reading less and running slower is not a memory problem. `q4k_element`
derived `d`, `dmin` and the 6-bit scale/min PER ELEMENT — two f16 loads plus
bit-assembly and converts, a branch and several byte loads for the scale/min,
group/sub-block/byte-index arithmetic, a `/256` and `%256` — roughly 40
instructions to produce one weight, and every one of those values is
constant across the whole 256-element super-block. We turned a bandwidth win
into a compute loss.

### The fix, which is ggml's shape

Split `q4k_element` into `q4k_header_for` (decode once) and `q4k_value`
(one byte load, one mask-or-shift, one fma). Then give each lane a
CONTIGUOUS run of `Q4K_BLOCK_ELEMENTS / SIMD_WIDTH` = 8 elements instead of a
32-strided walk. `lane*8 .. lane*8+7` never crosses a 32-element sub-block
boundary, so one header serves the whole run — the same amortization as
ggml's `for (short i = 0; i < 8; ++i)`.

Gated at EMIT time, not runtime, from the bound layout: exactly one packed
operand, contiguous along the reduction dim, reduction extent a whole number
of super-blocks.

### Measured, 3 runs, 51 samples each, min

| | before | after |
|---|---|---|
| packed 4096x4096 wall | 1.152 ms | **0.353 ms** (3.3x) |
| packed MARGINAL bandwidth | 12.3 GB/s | **~61 GB/s** (95.6 / 58.6 / 61.2) |
| packed element rate | 21.9 G elem/s | **~109 G elem/s** (5x) |
| gap to llama.cpp's 381 G elem/s | 17x | **3.5x** |

Parity unchanged: 38/38, and the packed-vs-dequantized-f32 device parity
test still holds. The tiling changes lane->element assignment, so the
per-lane partial sums are reassociated; the fold was already reassociated by
`simd_sum`, and the parity bound is relative.

### And the f32 kernel is not slow at all

f32 MARGINAL measured **355.6 / 366.4 / 355.9 GB/s** — stable, and 1.66x
ABOVE the 214.7 GB/s llama.cpp achieves on packed bytes. ROW 71's "f32
kernel 1.42x off" is retracted; that was a 21-sample artifact. The machine
delivers 356 GB/s to this kernel, so llama's 214.7 GB/s of PACKED bytes is
not a bandwidth ceiling either — it is a compute rate (381 G elem/s), which
is what the remaining 3.5x is against.

### Still open, correctly sized

Per-call fixed cost of **0.23-0.45 ms**, unmoved by taking `infer`/`bind` out
of the timed region (ROW 71 predicted that would be the cost; it was not).
What remains in it: command buffer creation, submit, `waitUntilCompleted`
round-trip, readback, and a per-op output and uniforms buffer allocation. At
1196 nodes per forward this is still the dominant serving-path term.

## ROW 73 — CORRECTION: the per-call fixed cost is per-FORWARD, not per-node. It is 1.6% of the budget, not the dominant term

**Repo:** `feat/tensor-consolidated`. **Host:** Apple M1 Max.

### The claim being corrected

ROW 71 and ROW 72 both stated that the 0.23-0.45 ms per-call fixed cost is
"the dominant serving-path term", reasoning: a forward is 1196 nodes, so at
one `execute` per node that is 228-478 ms per forward.

**The premise is false and I asserted it twice without opening the file.**
`execute_plan` encodes EVERY bound op of the program into ONE command
buffer and calls `commit()` + `waitUntilCompleted()` exactly once
(`omega/src/metal.rs:382-383`, and the loop above it at `for (position,
bound) in prepared.resolved.iter()`). The intercept is paid once per
`execute`, which for a serving loop is once per FORWARD.

0.28 ms per forward against llama.cpp Metal's 17.62 ms per token is **1.6%**.
It is not the dominant term and it was never worth the two rows spent on it.

### What the intercept actually is, and why the probe cannot see the rest

Measured across every configuration tried, the f32 intercept sat at
0.266-0.321 ms and did not move for:

| removed from the timed region | intercept before -> after |
|---|---|
| MSL compile (persistent pipeline cache) | unchanged |
| `infer` + `bind` (the `Plan` API, ROW 71's prediction) | 0.319 -> 0.287 ms |
| uniform buffer allocation (this row) | 0.287 -> 0.291 ms |

Three predicted causes, three misses. A floor that survives all of them and
sits at ~0.28 ms regardless of graph content is the CPU-GPU command buffer
round trip — submit, GPU wake, `waitUntilCompleted`, readback.

**And the probe is single-op, so it cannot measure per-OP cost at all.** The
uniform-buffer cache landed here allocates one fewer `MTLBuffer` per op per
call; on a 1196-node forward that is 1196 fewer allocations, and this probe
has exactly one op, so it correctly measured nothing. The same is true of
the per-op output buffer, which is still allocated fresh. Both are real for
a real graph and invisible here. **A single-op probe is the wrong instrument
for a per-op cost, and this row is the third time this file has recorded a
number the instrument could not have produced.**

### Where the GPU gap actually stands

| | measured | vs llama.cpp Metal |
|---|---|---|
| f32 kernel marginal | 336-385 GB/s | 1.6x FASTER than its 214.7 GB/s |
| packed Q4_K element rate | ~109 G elem/s | **3.5x** off its 381 G elem/s |
| per-forward intercept | ~0.28 ms | 1.6% of a 17.62 ms token |

One live defect: the packed kernel's remaining 3.5x. Nothing else measured
above noise.

## ROW 74 — row-blocking the packed kernel: 17x -> 3.5x -> 2.0x off llama.cpp Metal

**Host:** Apple M1 Max. **Method:** ROW 71's, two sizes per arm at
9.44/37.75 MB so the kernel dominates the ~0.28 ms per-forward intercept.

### The arc, one defect at a time

| packed Q4_K kernel | marginal bandwidth | element rate | vs llama.cpp's 381 G elem/s |
|---|---|---|---|
| per-element header decode (ROW 72 start) | 12.3 GB/s | 21.9 G | **17x** |
| + super-block tiling (ROW 72) | ~61 GB/s | ~109 G | **3.5x** |
| + 4-row blocking (this row) | **~108 GB/s** (90.4/108.5/132.1) | **~193 G** | **2.0x** |

Wall on the 37.75 MB arm: 0.845/0.871/0.864 ms -> **0.634/0.650/0.654 ms**.

### What row-blocking is

One SIMD group now folds `PACKED_ROWS_PER_GROUP` = 4 output rows at once
(`float sumf[4]`, ggml's `N_R0_Q4_K`). The activation's 8-value run is loaded
into registers ONCE per super-block and reused across all four rows, so
activation traffic falls 4x and each row costs one header decode plus eight
nibble extracts.

### The seam that made it safe

`grid_threads` and the kernel body must agree on the blocking factor or the
dispatch silently folds the wrong rows — it would not fail to compile, it
would produce wrong numbers. So `packed_row_block(resolved, quantized)` is
the single predicate both call: cooperative reduce, exactly two operands,
exactly one packed, packed operand contiguous along the reduction dim,
reduction extent a whole number of super-blocks. All decided at EMIT time
from the bound layout, none at runtime.

The device parity test caught the first attempt (a clobbered `sumf`
declaration emitted two init loops and no declaration) as an MSL compile
failure inside `execute`, which is exactly the failure mode a
"looks-like-MSL" gate would have missed.

### Where the GPU now stands

| | measured | vs llama.cpp Metal |
|---|---|---|
| f32 kernel marginal | 336-385 GB/s | 1.6x FASTER than its 214.7 GB/s |
| packed Q4_K element rate | ~193 G elem/s | **2.0x** off its 381 G elem/s |
| per-forward intercept | ~0.28 ms | 1.6% of a 17.62 ms token |

Remaining known amortization ggml has and this does not: the `dmin`
correction. ggml accumulates `sumy` (the activation sums) during the
register load and applies the min term as 4 MACs per super-block; this
kernel still subtracts `header.minimum` once per element inside
`q4k_value`.

## ROW 75 — NEGATIVE: factoring the `dmin` correction out of the dot measured ZERO. Reverted.

**Host:** Apple M1 Max. **Method:** ROW 71's two-size dissection, 3 runs.

### The change

ROW 74 closed with the one amortization ggml has that this kernel did not:
the min term. For a `weight * other` body folded with `Add`, the per-sub-block
scale and min factor straight out of the dot product —

```
sum_i (scale*n_i - min) * a_i  ==  scale * sum_i(n_i * a_i) - min * sum_i(a_i)
```

— turning one subtract PER ELEMENT into one multiply-subtract per
sub-block run, with `sum_i(a_i)` (`sumy`) computed once and shared across all
4 rows in the group. Implemented: a `q4k_nibble` MSL helper returning the raw
unscaled level, an `is_scaled_dot` recognizer for the exact
`Multiply`-body/`Add`-reduce shape, and a second emitter branch.

### Measured

| packed MARGINAL, 3 runs | median |
|---|---|
| ROW 74 (min per element) | 90.4 / 108.5 / 132.1 GB/s | ~108.5 |
| with `sumy` factoring | 101.3 / 115.8 / 107.5 GB/s | ~107.5 |

**Zero, inside the spread.** Parity 38/38 on both. Reverted: 80 lines and a
second emitter branch that buys nothing is complexity, not a fix, whatever
the incumbent does with it.

### What the negative result tells us

The subtract was not a binding instruction, so the packed kernel is no
longer instruction-bound on that axis. It is also not bandwidth-bound: at
~108 GB/s of packed bytes the 37.75 MB arm takes 0.63 ms, where the SAME
loop shape on f32 sustains 356 GB/s and would take 0.106 ms. So the kernel
sits between the two, and the next candidate has to be argued from what is
left per element rather than from ggml's feature list.

What is left per element in `q4k_value`: one byte load, a `/64` and `%64` and
`%32` (all powers of two), a select between mask and shift, one fma. ggml
reads FOUR nibbles per `uint16_t` load and folds the shift into `1/256` and
`1/16` scale constants applied once at the end, so its per-element cost is
closer to one load per four values plus an fma.

**Do not port the incumbent's optimizations by inventory.** Two of ggml's
three amortizations paid here (super-block header ROW 72, row blocking
ROW 74) and the third measured zero. The list is not the argument; the
per-element instruction count is.

## ROW 76 — eight levels from two 32-bit loads: 2.0x -> 1.64x off llama.cpp Metal

**Host:** Apple M1 Max. **Method:** ROW 71's two-size dissection at
9.44/37.75 MB, 3 runs. Run 2 was an interference outlier (large arm 1.382 ms
against 0.605/0.598) and is reported, not dropped silently.

### The change

ROW 75 said the next candidate had to be argued from what is left per
element, not from the incumbent's feature list. What was left in `q4k_value`:
ONE BYTE LOAD per element, plus a select between mask and shift.

A lane's run is `slot .. slot+7`, never crosses a 32-element sub-block
boundary, so all eight levels share a group and a nibble half and their bytes
are eight CONSECUTIVE bytes of `qs`. `slot % 32` is one of {0,8,16,24} and a
super-block is 144 bytes, so the address is 4-byte aligned: two `uint` loads
cover the run, and the eight nibbles fall out as shifts of those two words.
`q4k_run8` does that, hoisted out of the element loop.

ggml does the same thing one width down (`q1[i] & 0x000F / 0x0F00 / 0x00F0 /
0xF000` off a `uint16_t`) for the same reason — the extract is cheap and the
LOAD is what costs.

### Measured

| | large arm (37.75 MB) | MARGINAL | element rate | vs 381 G elem/s |
|---|---|---|---|---|
| ROW 74, byte loads | 0.634/0.650/0.654 ms | ~108 GB/s | ~193 G | 2.0x |
| this row, 32-bit loads | 0.605/**1.382**/0.598 ms | **~131 GB/s** | **~233 G** | **1.64x** |

1.21x, parity 38/38 unchanged.

### The whole packed arc, one defect at a time

| | marginal | element rate | gap |
|---|---|---|---|
| per-element header decode | 12.3 GB/s | 21.9 G | 17x |
| + super-block tiling (ROW 72) | ~61 | ~109 G | 3.5x |
| + 4-row blocking (ROW 74) | ~108 | ~193 G | 2.0x |
| + wide level loads (this row) | ~131 | ~233 G | **1.64x** |
| `dmin` factoring (ROW 75) | ZERO, reverted | | |

Four of ggml's amortizations tried, three paid, one measured nothing. The
ordering was found by measuring what was left, never by working down a list.

### Corroboration from a sibling session, same GPU

A concurrent large-GEMV bench on this same box independently measured a
**1.9x win from access pattern alone** on a 1M-row f32 GEMV — dim-major so
adjacent threads read adjacent words — taking 68.5 -> 129.2 GB/s. Same
machine, same lesson: on this GPU the shape of the load dominates the
arithmetic around it. That session also flagged `omega::execute` for creating
a device, queue and pipeline cache per call with no handle to hold across
calls; that is fixed (ROW 70/71: thread-local device/queue/pipelines,
`(pointer,len)`-keyed no-copy buffers, and `omega::plan`/`execute_plan`), so
a resident corpus now uploads once.
