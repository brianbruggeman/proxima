# Machine rooflines, per scoreboard lane

Derived from measurements already recorded in `docs/discipline.md`. Every
constant below cites its ROW and, where the row gives one, a `file:line`. No
benchmarking ran to produce this document — it is a re-derivable read of the
existing log, on Apple M1 Max / macOS / arm64, single core unless stated.
Numbers tagged **MEASURED** were read off an instrument in the cited row;
**DERIVED** means computed from other cited numbers in this doc;
**ASSUMED/DEBT** means no citable measurement exists and the number is either
a standard-formula estimate (flagged) or explicitly not produced here.

Re-derive this doc by re-reading the cited ROWs; nothing here should be
trusted without opening `discipline.md` at the cited line.

## Two machine-constants tables

### Bandwidth (single core, this M1 Max)

| ceiling | value | shape it was measured at | ROW / cite |
|---|---|---|---|
| same-shape streaming triad (2 read + 1 write, f32) | **69.95 GB/s** (CoV 0.09%, n=5) | 3,211,264 elements, 36.75 MiB traffic | ROW 176, `discipline.md:15776` |
| DRAM-bound streaming triad | **81.21 GB/s** (CoV 0.12%, n=5) | 16,777,216 elements, 192.00 MiB traffic | ROW 176, `discipline.md:15777` |
| implied per-element floor (4 B f32 read, `BANDWIDTH_BYTES_PER_NS`) | **0.0572 ns/element** (= 4 B / 69.95 GB/s) | -- | cited at ROW 181, `discipline.md:16460` |

### Per-unit compute (single core)

| ceiling | value | derivation | ROW / cite |
|---|---|---|---|
| scalar FMA (no vectorization) | **5.95-6.07 GMAC/s** (11.90-12.14 GFLOP/s), CoV 0.35-1.66% across sessions | `fma_roofline_macs_per_sec()`, 8 independent `mul_add` chains | ROW 145, `discipline.md:13433`; re-confirmed ROWs 148/149/150/154/156 |
| NEON register-blocked FMA (hardware ceiling, this core) | **48.5 GMAC/s** (97 GFLOP/s) | 24-accumulator saturation point: 12.15 G vector-FMA/s x 4 MACs/`float32x4_t` FMA = 48.6 GMAC/s, rounds to the cited 48.5 | ROW 20 sweep table, `discipline.md:1903-1910`; cited as 97 GFLOPS/48.5 GMAC/s at ROW 145, `discipline.md:13433-13434` |
| Accelerate/AMX `cblas_sgemm`, real mnist conv shapes (M=8/16/24) | **7.84 / 64.55 / 40.87 GFLOP/s** (layer1/layer2/layer3) | measured, `bench_conv_gemm_tile` micro-cell | ROW 189, `discipline.md:16920-16922` |
| Accelerate/AMX `cblas_sgemm`, larger synthetic shape (M=64) | **467.0 GFLOP/s** | same bench, `larger_square 64,32,32,32,3,3` | ROW 189, `discipline.md:16923` |
| this crate's own interpreted per-element dispatch floor | **0.2612 ns/element** (hand-rolled 8-op chain, n=7, CoV 0.21-0.37%) | standalone `rustc -O -C target-cpu=native` microbench of the identical Adam-update math | ROW 176, `discipline.md:15807` |
| this crate's own dedicated-kernel dispatch rate (best landed case) | **0.2865 ns/element** (9.7% over the 0.2612 floor) | `BodyShape::FusedAdamUpdate`, register-resident kernel, N=4 nodes matched | ROW 179, `discipline.md:16307` |
| this crate's own large-block interpreted-execution asymptote | **0.2016-0.2058 ns/element** | cross-regime two-point fit, streaming vs cache-resident | ROW 181, `discipline.md:16460` |

**Machine matrix-ceiling note.** For matrix-shaped lanes the box's real ceiling
is the max over its compute units, not NEON alone: Accelerate/AMX (467
GFLOP/s at the larger synthetic shape, ROW 189) is 4.8x the NEON per-core
ceiling (97 GFLOP/s, ROW 20) at that shape, and still 1.97-5.51x NEON even at
mnist's tiny real conv shapes (ROW 189). Every matrix lane below states BOTH
ceilings: the machine-wide (AMX) number for "what this box can actually do,"
and the NEON-only number for portability to non-Apple targets.

---

## Lane: mnist f32 inference (tile pipeline, per image)

**Shape (verified against `scripts/torch_reference/model.py` counts already
cited in the log, not re-read this session):** conv1 48,672 / conv2 663,552 /
conv3 1,672,704 / fc1 371,712 / fc2 320 MACs = **2,756,960 MACs/image**,
exactly the total the log's own roofline-denominator note independently
re-derives (`discipline.md:15220`, "Roofline-implied ns/MAC ... 2,756,960
total MACs").

### Candidate ceilings

| ceiling | formula | value | cite |
|---|---|---|---|
| scalar-FMA compute | 2,756,960 / 5.949e9 | **0.463 ms** | ROW 145, `discipline.md:13437` |
| NEON compute (per-core) | 2,756,960 / 48.5e9 | **0.057 ms** | ROW 145, `discipline.md:13438` |
| AMX/Accelerate compute (machine-wide, mnist conv shapes only) | conv-layer FLOPs / measured GFLOP/s per layer (7.84/64.55/40.87), summed | conv alone: **~0.130 ms** (97,344+1,327,104+3,344,384 FLOPs at the 3 measured rates: 24.4+20.6+81.8us NEON-tile-comparable Accelerate medians = 114.8us conv-only, ROW 189 micro-cell, `discipline.md:16920-16922`) plus untouched fc/glue -- **DERIVED**, not a full-net Accelerate roofline (fc path not re-measured under Accelerate this session) | ROW 189 |

### Binding constraint

NEON compute (0.057 ms) is the tightest per-core ceiling by 8.1x over scalar;
Accelerate/AMX beats NEON on conv specifically (1.97-5.51x per layer, ROW 189)
but was not landed default-on and does not have a measured full-net number
(fc1/fc2/BatchNorm/LogSoftmax untouched by that row) -- so the full-net
roofline below is NEON-denominated, with the AMX conv-only figure carried as
context, not substituted in.

### Roofline vs current best

| milestone | ms/image | x scalar roofline | x NEON roofline | cite |
|---|---|---|---|---|
| NEON roofline (hardware floor) | 0.057 | 1x | 1x | ROW 145 |
| scalar-FMA roofline | 0.463 | 1x | 8.1x | ROW 145 |
| pytorch/Accelerate incumbent bar | 0.119-0.134 | 3.5-3.9x under scalar | 2.09-2.35x over NEON | ROW 158, `discipline.md:14274`; ROW 189 Phase A, `discipline.md:16910` |
| **current best measured (this crate)** | **0.393907 ms** (band_kh, mean of 3 clean runs, CoV 0.19%) | **0.85x of scalar roofline** (beats it) | **6.91x over NEON roofline** | ROW 190, `discipline.md:17051,17059` |

Gap multiple, current best vs NEON roofline: **0.393907 / 0.057 = 6.91x**. Gap
vs the pytorch/Accelerate incumbent bar: **0.393907 / 0.119 = 3.31x** (using
the tighter end of the incumbent window).

Per-stage breakdown (conv3 dominates): conv3 GMAC/s achieved **8.527**
(1,672,704 MACs / 196.149us) vs 48 GMAC/s NEON roofline = **5.63x over**;
whole-net achieved **6.998 GMAC/s** vs 48 GMAC/s roofline = **6.86x over**.
ROW 190, `discipline.md:17067-17070`.

### Debts

- No default-on, full-net (conv+fc+glue) Accelerate/AMX roofline number
  exists yet — ROW 189 measured conv only; fc's Accelerate route was ROW 188
  (a documented tie at fc's GEMV shapes, not re-cited in full here).
- ROW 190's own residual gap (5.63-6.86x over NEON) is itself only partially
  attributed: instruction-count analysis (`discipline.md:17080`) names a
  26-instruction-for-16-useful-FMA loop body and an un-costed horizontal-sum
  stack round-trip, neither independently timed.

---

## Lane: train-step (MLP 784-128-10, batch 32, Adam)

**Shape.** `proxima-autograd`'s train fixture: fc1 784->128, fc2 128->10,
batch 32, Adam optimizer. Per-step element counts below are MEASURED counters
from the crate's own instrumentation, not derived from a MAC formula:
`generic_fast` (fused multi-op `BodyShape::Generic`/`FusedAdamUpdate` nodes)
**367,140 elements/step** and `monomorphic_fast` (single-op `Unary`/`Binary`
nodes, e.g. node 87's gradient-accumulate Add) **3,919,759 elements/step**
(both = the ROW 176 100-step counter totals, 36,714,000 and 391,975,920,
divided by 100; `discipline.md:15815-15818`). `*_slow` counters are exactly
zero at every measured row — no element in this lane ever falls off the fast
dispatch path.

### Candidate ceilings

| component | ceiling | formula | value | cite |
|---|---|---|---|---|
| node 87 (bandwidth-bound Add, [32,784,128], 12 B/element) | same-shape streaming ceiling | 38,535,168 B / 69.95 GB/s | **550.42-551.67 us** (hand-rolled triad, this shape) | ROW 176, `discipline.md:15764-15781` |
| `generic_fast` mass, at the crate's own dedicated-kernel floor | dispatch floor | 367,140 x 0.2865 ns | **105.2 us** | ROW 179 rate, `discipline.md:16307` |
| `generic_fast` mass, at the pure hand-rolled floor (upper-bound optimism) | dispatch floor | 367,140 x 0.2612 ns | **95.9 us** | ROW 176 rate, `discipline.md:15807` |
| `monomorphic_fast` mass, at this-shape's measured bandwidth ceiling | bandwidth floor | 3,919,759 x 0.1715 ns (node-87-matched hand-rolled rate) | **672.3 us** | ROW 176, `discipline.md:15781` |
| whole-step compute-only floor (architecture-derived MAC count, **DERIVED, flagged**) | NEON compute | see below | **~135 us** | this doc, formula below |

**DERIVED MAC count (flagged, not independently cited):** forward fc1 (b=32,
784x128) + backward grad-weight fc1 (same shape) + forward fc2 (b=32, 128x10)
+ backward grad-input fc2 + backward grad-weight fc2 = 2x(32x784x128) +
3x(32x128x10) = 6,422,528 + 122,880 = **6,545,408 MACs**, standard backprop
MAC-counting convention (grad-input for the input layer omitted — no
upstream layer consumes it). At 48.5 GMAC/s NEON: 6,545,408 / 48.5e9 =
**134.96 us**. This MAC count is a formula-based estimate from the stated
784-128-10/batch-32 architecture, not read off a per-step MAC counter in the
log — carried as a DEBT (see below), not trusted as tightly as the two
element-count-based floors above.

### Binding constraint

**Dispatch floor, not bandwidth, not compute** — this is the log's own
explicit, cross-validated conclusion (ROW 181, `discipline.md:16460`): the
crate's per-element interpreted-dispatch cost (0.2612-0.2865 ns/element,
measured at 3 independent points: hand-rolled floor, `FusedAdamUpdate`
kernel, and the large-block streaming asymptote) is **3.5-5x slower per
element than the DRAM-bandwidth-implied rate** (0.0572 ns/element) and
**~12.7x slower per element than the NEON-compute-implied rate** (0.0206
ns/element = 1/48.5 GMAC/s). Node 87 itself is the one component that IS at
the bandwidth wall (82.6% of its own same-shape ceiling, ROW 176) — but it is
a minority of the step's element mass (3.9M of ~4.3M total fast-path
elements) and even its own achieved rate (0.2077-0.2209 ns/element) sits
above the pure 0.0572 ns/element bandwidth floor because of dispatch
overhead riding on top.

### Roofline vs current best

Composed floor (elementwise mass only, DERIVED as a sum of the two
element-count x measured-rate rows above, NOT a single sealed benchmark):
95.9-105.2 us (`generic_fast`) + 672.3 us (`monomorphic_fast` at its
shape-specific bandwidth ceiling) = **~768-778 us**. Adding the DERIVED
whole-step compute-only floor (135 us) as a rough upper-bound composition
(the elementwise floor already includes most matmul-adjacent work per ROW
174's own node attribution, so this is not a clean sum — flagged) gives an
outer bound of **~900 us-1.0 ms**.

| milestone | value | cite |
|---|---|---|
| current best sealed (mean-of-p50, 3 runs, CoV 0.29%) | **1.3699 ms** | ROW 179, `discipline.md:16307` |
| pre-`FusedAdamUpdate` baseline | 1.8831 ms | ROW 177, `discipline.md:15853` |
| composed dispatch-floor estimate (elementwise mass only) | ~0.768-0.778 ms | this doc, DERIVED above |

Gap multiple, current best vs the composed elementwise-mass floor: **1.3699 /
0.773 = 1.77x** (midpoint of the 0.768-0.778 ms band). This is a DERIVED
composition, not a single measured target — reported as a band, not a point.

### Debts

- The 6,545,408-MAC whole-step compute count is a standard-formula DERIVATION
  from the stated architecture, not a counter read from an instrument. Named
  as DEBT: no per-step MAC counter exists in this crate's instrumentation
  (only element counters for the fast-dispatch paths).
- The "~900 us-1.0 ms" outer-bound composition double-counts an unknown
  fraction of node 87's own Add against the DERIVED MAC-based compute floor
  (ROW 176 already notes node 87 is part of the grad_w1 reduce accumulation,
  `discipline.md:15783`) — flagged, not resolved, this session.
- 66.04%/22.72%/9.17% (elementwise/reduce/dead-constant) phase shares are
  ROW 174's own (`discipline.md:15609`), measured against a since-superseded
  2.0935 ms sealed step, not re-normalized to the current 1.3699 ms step.

---

## Lane: q4_K int8 dot (LLM decode, single weight-matrix GEMV)

**Constants.** Q4_K: 144 bytes / 256 elements = **0.5625 B/weight** (4.5
bits/weight), `Q4K_BLOCK_BYTES`/`Q4K_BLOCK_ELEMENTS`, cited at
`discipline.md:12386-12392` (source: `omega/src/msl.rs:291,296`). Activation
side (Q8_K, quantized once per call and reused across all output rows/tokens
sharing that activation): `f32` scale + 256 `i8` quants + 16 `i16` bsums =
292 B/256 elements = **1.140625 B/element**, `discipline.md:3906-3907`.

**MAC-per-weight-byte relationship (DERIVED, exact for GEMV, w=1):** a GEMV
`y[i] = sum_j W[i,j]*x[j]` reads each weight element exactly once and
performs exactly one MAC with it — bytes/MAC = bytes/weight = **0.5625 B**
for the w=1 (single-token) case. This is a clean, no-reuse bandwidth
relationship (activation bytes amortize across the whole row and do not
scale with weight-matrix size, so they are excluded from the weight-side
bytes/MAC figure).

### Candidate ceilings

| ceiling | formula | value | cite |
|---|---|---|---|
| bandwidth ceiling (weight-stream, same-shape triad rate) | 69.95 GB/s / 0.5625 B/MAC | **124.4 GMAC/s** | this doc, DERIVED from ROW 176's `discipline.md:15776` |
| bandwidth ceiling (weight-stream, DRAM-bound triad rate) | 81.21 GB/s / 0.5625 B/MAC | **144.4 GMAC/s** | this doc, DERIVED from ROW 176's `discipline.md:15777` |
| this crate's own kernel, single-thread, kernel-call-only | measured directly | **147.72-150.60 GMAC/s** | ROW 116, `discipline.md:11985` |
| ggml, single-thread, kernel-call-only, identical shapes/bytes | measured directly | **35.49-35.60 GMAC/s** | ROW 116, `discipline.md:11986` |
| production (this crate), achieved, w=1 | measured directly | **39.54 GMAC/s** | ROW 116/99, `discipline.md:11965` |
| production (this crate), achieved, w=8 | measured directly | **146.92 GMAC/s** | ROW 116/99, `discipline.md:11966` |

### Binding constraint

The crate's own kernel (147.72-150.60 GMAC/s) sits **above** the naive
weight-byte bandwidth ceiling computed from the read+write triad rate
(124.4-144.4 GMAC/s) — a DERIVED, not directly cited, effective-bandwidth
recomputation: total weight traffic for the summed decode-step shape set
(7,110,402,048 MACs x 0.5625 B/MAC = 3.9996 GB) over the kernel's own
47.214-48.132 ms gives **83.10-84.72 GB/s effective**, 2-4% above the 81.21
GB/s read+write DRAM-bound ceiling. This is plausible (a pure sequential
READ stream, as the weight side is, is not bound by the same write-port
contention a read+write triad pays) but is not independently verified against
a read-only bandwidth ceiling — **flagged as DEBT**, not asserted as a clean
"beats the wall" finding. Read at face value: this kernel is **at, or just
over, the single-core DRAM streaming ceiling** — it is compute/bandwidth
co-bound, not dispatch-limited the way the train-step lane is.

ggml's own kernel, by the same computation, achieves only **19.96-20.02
GB/s effective** (25% of the DRAM ceiling) — ggml is compute/dispatch-bound
at this shape, not bandwidth-bound, which is why it trails by 4.15-4.24x
(ROW 116, `discipline.md:11988`) despite reading identical bytes.

### Roofline vs current best

| milestone | GMAC/s | vs 124.4-144.4 GMAC/s bandwidth ceiling | cite |
|---|---|---|---|
| bandwidth ceiling (weight-stream) | 124.4-144.4 | 1x | this doc |
| **current best (this crate's kernel)** | **147.72-150.60** | **1.02-1.04x** (at or just over) | ROW 116 |
| ggml (incumbent) | 35.49-35.60 | 3.5-4.1x under | ROW 116 |
| production achieved, w=1 (orchestration overhead, not kernel) | 39.54 | 3.1-3.7x under | ROW 116/99 |
| production achieved, w=8 | 146.92 | ~1x (matches kernel) | ROW 116/99 |

Gap multiple, kernel vs bandwidth ceiling: **effectively closed (0.98-1.04x
of the DRAM-bound figure)** — the remaining gap in this lane is entirely in
production's w=1 orchestration path (3.74-3.81x slower than its own kernel,
ROW 116, `discipline.md:11992`), not in the kernel or the hardware ceiling.

### Debts

- The "kernel exceeds the read+write triad's DRAM-bound rate" finding is a
  DERIVED recomputation from ROW 116's MAC/time figures against ROW 176's
  triad bandwidth, never independently cross-checked with a read-only
  bandwidth microbench. No read-only streaming ceiling has been measured on
  this box — named as DEBT, not measured this session.
- w=8's 146.92 GMAC/s is a production-orchestration number, not a raw-kernel
  number at batch width 8 — no batched kernel-call-only figure exists to
  compare it against directly.

---

## Lane: BGE-small embedding (per sentence, 7-9 tokens)

**Shape, as measured (not assumed):** 3 real sentences, 8/9/7 tokens,
hidden=384, 25 LayerNorm sites/sentence (ROW 191, `discipline.md:17121`).
**33M params** and full layer/head/intermediate-size architecture are **not
recorded anywhere in `discipline.md`** — the log measures per-op costs
(epilogue kernel, LayerNorm) and e2e sentence latency, never a full op
histogram with total MAC count. A standard-formula MAC estimate (MACs/token
~= param count, the usual transformer approximation) would require assuming
architecture facts (layer count, intermediate size, attention head count)
this repo's own log does not cite — **flagged as DEBT below, not computed**,
per this session's own instruction that unmeasured constants are named, not
guessed.

### What IS measured, and grounds a partial roofline

| quantity | value | cite |
|---|---|---|
| interpreted `apply_body` evaluator, LayerNorm tail (pre-fusion) | **32.90 ns/element** | ROW 183, cited at ROW 191 `discipline.md:17123` |
| fused LayerNorm epilogue kernel | **5.25 ns/element** (1,208,877 ns / 230,400 elements) | ROW 191, `discipline.md:17123` |
| LayerNorm-tail speedup (fused vs interpreted) | **6.3x** | ROW 191, `discipline.md:17123` |
| engagement | 75 hits/run (25/sentence x 3), 230,400 elements/run | ROW 191, `discipline.md:17121` |
| e2e per-sentence, fused (5 paired runs) | **26.68 ms** (CoV 4.68%) | ROW 191, `discipline.md:17123` |
| e2e per-sentence, unfused | **27.49 ms** (CoV 12.63%, one run to 34.26ms) | ROW 191, `discipline.md:17123` |
| e2e delta | **-0.81 ms (~2.9%)**, inside the unfused arm's own noise band | ROW 191, `discipline.md:17123` |
| LayerNorm epilogue's own share of sentence time | ~1.2 ms of ~26 ms, **~4.5%** | ROW 191, `discipline.md:17123` |

### Binding constraint

Cannot be stated with a whole-net roofline number — no total-MAC-per-sentence
figure exists to divide by a compute ceiling, and no full-forward
bytes-moved figure exists to divide by a bandwidth ceiling. What IS
established: the fused LayerNorm kernel (5.25 ns/element) sits between the
crate's own general-purpose interpreted-dispatch floor family (0.26-0.29
ns/element, train-step lane) and the pre-fusion interpreted evaluator (32.90
ns/element) — i.e. LayerNorm's fused kernel is real and large (6.3x) at the
op level but the op itself is a small fraction (~4.5%) of sentence time, so
the e2e win is real but modest (~2.9%, inside noise). The dominant ~95.5%
of sentence time (matmul/attention/dispatch mass) has no per-element rate
cited in the log at all.

### Roofline vs current best

Not computable as a single number this session — see debts. The only
grounded ratio available: **26.68 ms measured / roofline unknown**.

### Debts

- **Full architecture (layer count, intermediate size, head count) is not
  recorded in discipline.md — MAC/token cannot be derived without assuming
  facts outside the cited material. Named as DEBT, not guessed, per this
  session's own constraint.**
- 133 MB f32 weight-stream-vs-cache framing (from the task brief) has no
  corresponding measurement in the log — no bytes-moved figure for BGE's
  full forward pass exists to check against the 69.95/81.21 GB/s ceilings.
  DEBT.
- The ~95.5% of sentence time outside the LayerNorm epilogue (attention,
  matmul, dispatch/glue) has no per-op profiler breakdown in the log the way
  mnist's forward pass has (ROW 189 Phase A, torch per-op attribution) — no
  BGE-side equivalent of that table exists. DEBT.

---

## Summary table

| lane | binding constraint | roofline | current best | gap | debts |
|---|---|---|---|---|---|
| mnist f32 inference | NEON per-core compute (0.057 ms/image) | 0.057 ms/image | 0.393907 ms/image (ROW 190) | **6.91x** | full-net AMX roofline unmeasured |
| train-step (MLP 784-128-10, b32, Adam) | this crate's own interpreted per-element dispatch floor (0.26-0.29 ns/element) | ~0.768-0.778 ms/step (composed, DERIVED) | 1.3699 ms/step (ROW 179) | **~1.77x** (DERIVED band) | whole-step MAC count is formula-derived, not counter-measured |
| q4_K int8 dot (decode GEMV) | co-bound: kernel is at/just-over the DRAM streaming ceiling | 124.4-144.4 GMAC/s (bandwidth) | 147.72-150.60 GMAC/s (kernel, ROW 116) | **effectively closed, ~1.0x** (kernel already at ceiling; gap lives in production orchestration, 3.74-3.81x on w=1) | read-only bandwidth ceiling never independently measured |
| BGE-small embedding (7-9 tok, LayerNorm epilogue) | unknown for the whole net; LayerNorm epilogue op itself is dispatch-floor-shaped (5.25 ns/element fused vs 32.90 ns/element interpreted) | not computable (no whole-net MAC/byte count) | 26.68 ms/sentence e2e (ROW 191) | **not computable** | architecture (layers/intermediate size), full-net MAC count, full-net bytes-moved all unrecorded |

All citations above point at `proxima-tensor/docs/discipline.md`; re-derive
by opening the cited `ROW`/`file:line` pairs directly.
