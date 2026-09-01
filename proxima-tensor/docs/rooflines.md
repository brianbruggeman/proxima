# Machine rooflines, per scoreboard lane

Apple M1 Max / macOS / arm64, single core unless stated. Every constant
cites its ROW or its branch/commit. Numbers tagged **MEASURED** were read
off an instrument; **DERIVED** means computed from other cited numbers in
this doc; **ASSUMED/DEBT** means no citable measurement exists and the
number is either a standard-formula estimate (flagged) or explicitly not
produced here.

**Provenance changed 2026-09-01.** The original edition of this doc was a
pure re-read of `discipline.md` with no benchmarking of its own. It is no
longer that: the BGE lane now carries measurements taken directly, on
branches not yet merged to `main` (`perf/amx-width-tile` `a03b96d`,
`perf/width-tile-accs` `15b02cf`, `perf/plan-cache` `e5bdb8e`), plus a
re-run of the ROW 195 profile at `1f56f95`. Those cells cite the branch,
not a ROW, and are marked as such. Do not treat this doc as re-derivable
from the log alone until those branches land and their rows are written.

**As-of dates, per lane** — a lane with no fresh measurement keeps its last
recorded figure and says so, rather than being silently carried forward as
current:

| lane | current-best figure last MEASURED | freshness |
|---|---|---|
| BGE-small embedding | 2026-09-01, this session | **CURRENT** |
| mnist f32 inference | ROW 190/192 | stale — not re-measured since |
| train-step MLP b32 | ROW 179 | stale — not re-measured since |
| q4_K CPU kernel | ROW 116 | stale — not re-measured since |
| q4_K GPU decode (Metal) | ROW 193 | stale — not re-measured since |

Re-derive the ROW-cited figures by opening `discipline.md` at the cited
line; re-derive the branch-cited ones with the re-prove commands in the
matching lane section.

## Roofline discipline: physics-only ceilings, self-limits kept separate

**Owner ruling, 2026-09-01.** A roofline derives from silicon physics ONLY —
bandwidth (bytes/s), unit compute rates (MACs/s or FLOP/s), bytes moved, MAC
counts. A ceiling derived from this crate's OWN code costs (interpreted
per-element dispatch, an achieved-not-peak rate, a specific kernel's own
measured ns/element) is a **SELF-LIMIT**, not a roofline — it describes what
this architecture currently costs, not what the hardware allows. Every lane
below states BOTH, separately labeled, never blended into one number:

- **MACHINE ROOFLINE** — physics-only: bandwidth ceiling, NEON/AMX compute
  ceiling, or the max/min composition of the two, cited at a hardware
  constant, never at this crate's own achieved rate.
- **CURRENT-ARCHITECTURE FLOOR** — a self-limit: the best this specific
  dispatch/execution shape can do, cited at this crate's own measured rate.
  Kept as an indictment of the architecture, not mistaken for the ceiling —
  **the current-architecture floor is the next deletion target, not the
  ceiling.**

**Trigger.** The train-step lane below presented a composed
"0.768-0.778ms roofline" built entirely from this crate's own
0.2612-0.2865 ns/element interpreted-dispatch rate (ROW 176/179,
`discipline.md:15807`/`16307`) — a self-limit dressed as a hardware ceiling.
PyTorch measures **0.380ms** on the identical shape (ROW 159,
`discipline.md:14448`; corroborated at ROW 157's own 0.3406ms,
`discipline.md:14260`) — beneath the claimed "roofline" by 2x, which is
sufficient on its own to prove the composed number was never a ceiling. The
corrected derivation is below.

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

All three rows below are **MACHINE ROOFLINE** (physics-only: a hardware
compute rate divided into this doc's own MAC count) — no self-limit
conflation in this lane; no "our own achieved dispatch rate" figure is used
as a ceiling anywhere in this section.

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

### CURRENT-ARCHITECTURE FLOOR (self-limit — this crate's own dispatch cost, not physics)

This composition was previously mislabeled a "roofline." It is not: every
input is this crate's own achieved interpreted-dispatch rate, not a hardware
constant. Kept intact below as the indictment of the current dispatch
architecture, not the ceiling to chase.

| component | floor | formula | value | cite |
|---|---|---|---|---|
| node 87 (bandwidth-bound Add, [32,784,128], 12 B/element), this crate's own achieved rate | same-shape streaming, hand-rolled triad at this shape | 38,535,168 B / 69.95 GB/s | **550.42-551.67 us** | ROW 176, `discipline.md:15764-15781` |
| `generic_fast` mass, at the crate's own dedicated-kernel floor | dispatch floor | 367,140 x 0.2865 ns | **105.2 us** | ROW 179 rate, `discipline.md:16307` |
| `generic_fast` mass, at the pure hand-rolled floor (upper-bound optimism) | dispatch floor | 367,140 x 0.2612 ns | **95.9 us** | ROW 176 rate, `discipline.md:15807` |
| `monomorphic_fast` mass, at this-shape's measured bandwidth ceiling | bandwidth floor | 3,919,759 x 0.1715 ns (node-87-matched hand-rolled rate) | **672.3 us** | ROW 176, `discipline.md:15781` |

Composed self-limit (elementwise mass only, DERIVED as a sum of the two
element-count x measured-rate rows above, NOT a single sealed benchmark):
95.9-105.2 us (`generic_fast`) + 672.3 us (`monomorphic_fast` at its
shape-specific bandwidth ceiling) = **~768-778 us CURRENT-ARCHITECTURE
FLOOR**, not a roofline: it is built entirely from this crate's own
interpreted-dispatch rates (ROW 181, `discipline.md:16460`), and the
architecture producing it is itself the next deletion target.

### MACHINE ROOFLINE (physics-only, two candidates)

**Compute-bound candidate — NEON per-core, whole-step MAC count (DERIVED MAC
count, flagged, not independently cited):** forward fc1 (b=32, 784x128) +
backward grad-weight fc1 (same shape) + forward fc2 (b=32, 128x10) +
backward grad-input fc2 + backward grad-weight fc2 = 2x(32x784x128) +
3x(32x128x10) = 6,422,528 + 122,880 = **6,545,408 MACs**, standard backprop
MAC-counting convention (grad-input for the input layer omitted — no
upstream layer consumes it). At the NEON register-blocked FMA hardware
ceiling this doc's own bandwidth table cites for the mnist lane (**48.5
GMAC/s**, 24-accumulator saturation point, ROW 20 sweep table,
`discipline.md:1903-1910`, cited at ROW 145 `discipline.md:13433-13434`):
6,545,408 / 48.5e9 = **134.96 us**. This MAC count is a formula-based
estimate from the stated 784-128-10/batch-32 architecture, not read off a
per-step MAC counter in the log — carried as a DEBT (see below).

**Bandwidth-bound candidate — real per-step byte traffic, enumerated by
stream, MLP 784-128-10 batch=32 f32 (4 B/element), trainable-param count
`P` = W1(784x128=100,352) + b1(128) + W2(128x10=1,280) + b2(10) = **101,770
elements**:**

| stream | elements | formula | bytes |
|---|---|---|---:|
| params read, forward pass (W1,b1,W2,b2) | 101,770 | `P` | 407,080 |
| params read, backward (W2 only — propagates grad into layer1) | 1,280 | `W2` | 5,120 |
| activations written, forward (h1 [32,128], h2 [32,10]) | 4,416 | 32x128 + 32x10 | 17,664 |
| activations read, backward (h1, h2, input x [32,784]) | 29,504 | h1+h2+x | 118,016 |
| grads written (dW1,db1,dW2,db2) | 101,770 | `P` | 407,080 |
| grads read by Adam | 101,770 | `P` | 407,080 |
| Adam m,v read (old) | 203,540 | `2P` | 814,160 |
| Adam m,v written (new) | 203,540 | `2P` | 814,160 |
| params written (post-update W,b) | 101,770 | `P` | 407,080 |
| **total** | | | **3,397,440 B (3.24 MiB)** |

This is a DERIVED, formula-based traffic enumeration (not a byte counter read
off an instrument) — carried as DEBT alongside the MAC count. It omits any
loss-scalar/logit-softmax traffic (negligible, batch=32 x 10 elements) and
assumes no operand is re-read from a hot cache for free (worst-case DRAM
assumption, consistent with this doc's own triad-ceiling convention). At the
same-shape streaming ceiling (**69.95 GB/s**) and the DRAM-bound ceiling
(**81.21 GB/s**, both ROW 176, `discipline.md:15776-15777`):
3,397,440 / 69.95e9 = **48.58 us**; 3,397,440 / 81.21e9 = **41.84 us** ->
**41.8-48.6 us bandwidth-bound candidate**.

**AMX candidate (second unit column, measured-at-conv-shapes caveat).** FLOPs
= 2 x 6,545,408 MACs = 13,090,816 FLOPs. Applying the measured Accelerate/AMX
`cblas_sgemm` rates this doc's own bandwidth table cites at ROW 189
(`discipline.md:16920-16922`, mnist's real conv shapes, 7.84-64.55 GFLOP/s;
`discipline.md:16923`, the M=64 synthetic shape, 467.0 GFLOP/s) gives a wide,
**unmeasured-at-this-shape** band: 13,090,816 / 64.55e9 = 202.8 us down to
13,090,816 / 467.0e9 = 28.0 us, or up to 1670 us at the smallest cited rate
(7.84 GFLOP/s, mnist's M=8 conv1 layer). **This is not a tight candidate**:
train-step's own GEMM shapes (M=batch=32 for both fc1/fc2, K=784/128, N=128/10)
have never been measured under Accelerate. ROW 188 found Accelerate TIES NEON
at fc's own `M=1` GEVM shapes (`discipline.md:16912`, batch-1 inference,
not this batch-32 train shape); ROW 189 found Accelerate WINS 1.97-5.51x at
conv's `M=8..24` shapes (closer in magnitude to this lane's own `M=32`). Train
step's own AMX ceiling sits somewhere in this band, unresolved — carried as
DEBT, not asserted.

### Binding constraint

**Compute-bound**, machine roofline = **~135 us** (NEON candidate), since
134.96 us (compute) > 41.8-48.6 us (bandwidth) — this workload's own
arithmetic intensity (6,545,408 MACs / 3,397,440 bytes = 1.93 MACs/byte)
sits on the compute-bound side of the classic roofline ridge point at these
shapes. This is the OPPOSITE conclusion from the current-architecture floor's
own binding constraint below: the machine has slack in both bandwidth AND
compute relative to what this crate currently spends dispatching.

The current-architecture floor's own binding constraint, unchanged from the
log's own explicit, cross-validated conclusion (ROW 181,
`discipline.md:16460`): **dispatch cost, not bandwidth, not compute** — the
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

| milestone | value | tag | cite |
|---|---|---|---|
| **machine roofline (compute-bound, NEON)** | **~135 us/step** | MACHINE ROOFLINE | this doc, DERIVED above |
| machine roofline, bandwidth-bound candidate (non-binding) | ~41.8-48.6 us/step | MACHINE ROOFLINE, non-binding | this doc, DERIVED above |
| machine roofline, AMX candidate (unmeasured at this shape) | ~28-1670 us/step | MACHINE ROOFLINE, wide DEBT band | this doc, DERIVED above |
| pytorch incumbent (single-thread, quiet-host p50 range) | 0.317-0.471 ms (mean 0.380 ms) | reference point, between floor and ceiling | ROW 159, `discipline.md:14448`; corroborated ROW 157 0.3406ms, `discipline.md:14260` |
| current-architecture floor (self-limit, composed) | ~0.768-0.778 ms | CURRENT-ARCHITECTURE FLOOR | this doc, DERIVED above |
| current best sealed (mean-of-p50, 3 runs, CoV 0.29%) | **1.3699 ms** | measured | ROW 179, `discipline.md:16307` |
| pre-`FusedAdamUpdate` baseline | 1.8831 ms | measured | ROW 177, `discipline.md:15853` |

Gap multiples, all vs the ~135 us MACHINE ROOFLINE: current best **1.3699 ms
/ 0.135 ms = 10.1x**; pytorch **0.380 ms / 0.135 ms = 2.8x** (torch itself
sits 2.8x off the physics ceiling — the incumbent has its own dispatch tax,
just a far smaller one); current-architecture floor **0.773 ms / 0.135 ms =
5.7x** (midpoint of the 0.768-0.778 ms band) — i.e. even if this crate hit
its OWN best-case dispatch floor exactly, it would still be 5.7x off the
machine, because the floor is architecture-limited, not physics-limited.
Gap, current best vs pytorch: **1.3699 / 0.380 = 3.6x**.

### Debts

- The 6,545,408-MAC whole-step compute count and the 3,397,440-byte traffic
  enumeration are both standard-formula DERIVATIONS from the stated
  architecture, not counters read from an instrument. Named as DEBT: no
  per-step MAC counter or byte counter exists in this crate's
  instrumentation (only element counters for the fast-dispatch paths).
- The AMX candidate band (28-1670 us) is DEBT-heavy: no Accelerate/AMX
  measurement exists at train-step's own M=32 GEMM shapes; the two anchors
  cited (conv M=8-24, synthetic M=64) bracket but do not pin this lane's own
  rate.
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

The first two rows are **MACHINE ROOFLINE** (bandwidth is a hardware
constant divided by a bytes/MAC relationship, no achieved-rate substitution).
The remaining rows are measured kernels/production paths, not rooflines.

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

## Lane: q4_K GPU decode (Metal, LLM decode, real weight-matrix shapes)

Derived from ROW 193 (`discipline.md`, ROW 193 heading), a same-day interleaved
paired cell against llama.cpp Metal on the identical `openchat-3.5-1210.Q4_K_S.gguf`
checkpoint (7.24B params, Mistral architecture, 32 layers) both arms read.

**MAC-per-token constant (reused from the CPU lane above for consistency):**
7,110,402,048 MACs/token (ROW 116's own summed decode-step shape-set figure);
weight-byte sweep 3.9996 GB/token at Q4_K's 0.5625 B/MAC (same figure the CPU
lane's Binding Constraint section already cites).

### Candidate ceilings

| ceiling | value | cite |
|---|---|---|
| GPU streaming bandwidth (trivial-copy or read-only probe) | **DEBT -- not measured**. `membw_probe`'s Metal arm uses a reduce-to-scalar (`Add`) pattern, not a streaming copy, and returned 0.3-0.4 GB/s on a loaded host -- tagged not-a-ceiling, not substituted with a spec-sheet figure. | ROW 193 |
| llama.cpp Metal, achieved (incumbent's own kernel+driver, today) | **234.1 GB/s** (17.086 ms/token, 2.403e-3 ns/mac, 416.1 GMAC/s) | ROW 193, fresh; corroborates the older 214.7-230.99 GB/s citations (`q4k_matvec_probe.rs` docstring; `discipline.md:7882`) within the same order |
| proxima Metal, achieved, kernel-only (batched, steady-state `gpu_exec`) | **70.14 GB/s** (57.032 ms/token GPU-exec, 8.02e-3 ns/mac, 124.7 GMAC/s) | ROW 193 |
| proxima Metal, achieved, full decode loop (incl. CPU-side orchestration) | **57.24 GB/s** (69.86 ms/token, 9.65e-3 ns/mac, 103.6 GMAC/s) | ROW 193 |
| proxima CPU, achieved, kernel-only (for reference, same MAC constant) | 147.72-150.60 GMAC/s | ROW 116, cited in the CPU lane above |

### Binding constraint

**GPU-side, unresolved this session (DEBT):** no independent GPU streaming-bandwidth
ceiling exists to compute a fraction-of-ceiling against, the same gap the
`membw_probe` landing row (`discipline.md:12461` vicinity) named and deferred.
Read at face value against the incumbent's OWN achieved rate rather than a
hardware ceiling: proxima's Metal kernel (124.7 GMAC/s) sits at **30.0%** of
llama.cpp Metal's achieved rate (416.1 GMAC/s) -- notably BELOW proxima's own
CPU kernel (147.72-150.60 GMAC/s, ROW 116), i.e. **this GPU port does not yet
beat this crate's own CPU kernel**, let alone the incumbent. The full-decode
wall-clock gap (4.09x) is wider than the kernel-only gap (3.34x); the
difference (~12.8 ms/token) is CPU-side dispatch/orchestration around the GPU
kernel (`prepare`/`emit`/`pipeline_lookup`/`op_setup`/`block_upload`,
`token_breakdown_metal` fields, ROW 193), the same shape ROW 100/116 already
named for the CPU arm's own w=1 orchestration tax.

### Per-shape attribution (new this row, diagnostic-only -- see caveat)

`profiles_one_real_decode_step_by_per_op_gpu_time`'s op-isolated timing (one
command buffer per op, ~1.7x inflated vs the batched production figure above,
`total_gpu_ms=99.055` op-isolated vs `gpu_exec_ms=57.032` batched at the same
step) is not production-representative in absolute terms, but its measured
`operand_bytes` correctly DERIVES each tensor's real bytes/weight and
therefore its quant type without trusting file metadata: `ffn_gate`/`ffn_up`/
`attn_q`/`attn_output`/`attn_k` measure exactly 0.5625 B/weight (Q4_K);
`output.weight` measures exactly 0.8203125 B/weight (Q6_K's 210B/256elem,
exact match) -- confirming `bench_q4k_matmul.rs`'s own comment that Q4_K_S
bumps precision-sensitive tensors; `ffn_down`/`attn_v` measure 0.5781 B/weight,
+2.8% over Q4_K nominal, mechanism unresolved (DEBT, named not guessed).

### Roofline vs current best

| milestone | GMAC/s | vs llama.cpp Metal achieved (416.1 GMAC/s) | cite |
|---|---|---|---|
| llama.cpp Metal (incumbent, achieved) | 416.1 | 1x | ROW 193 |
| proxima CPU kernel (for reference) | 147.72-150.60 | 2.76-2.82x under | ROW 116 |
| **proxima Metal kernel-only (batched, steady-state)** | **124.7** | **3.34x under** | ROW 193 |
| proxima Metal full decode loop | 103.6 | 4.02x under (4.09x on ms/token, same ratio within rounding) | ROW 193 |

No GPU streaming-bandwidth ceiling exists to state a roofline-vs-hardware gap
the way the CPU lane does (124.4-144.4 GMAC/s bandwidth ceiling, effectively
closed) -- this lane's only honest ratio is against the incumbent's own
achieved rate, not against this box's hardware limit.

### Debts

- **GPU streaming bandwidth ceiling never cleanly measured** -- `membw_probe`'s
  Metal arm is the wrong probe shape (reduce-to-scalar, not streaming copy)
  and this session's host was never quiet enough to trust even that shape.
  Re-prove command in ROW 193; needs both a corrected probe shape AND a quiet
  host.
- **`ffn_down`/`attn_v`'s 0.5781 B/weight (+2.8% over Q4_K's 0.5625) is
  measured, not explained** -- named, not guessed at, per principle 6.
- **No standalone criterion bench exists for the GPU per-shape arm** (the
  orchestration gap ROW 193 names) -- today's per-shape numbers come from an
  `instrument`-gated diagnostic test, not a criterion-tracked, baseline-saved
  artifact.
- Per-shape numbers are op-isolated (command-buffer-per-op), not batched --
  the ~1.7x inflation factor between op-isolated and batched totals at the
  same step is itself only a two-point comparison, not independently
  verified per-shape.

---

## Lane: BGE-small embedding (per sentence, 7-9 tokens)

**Shape, as measured (not assumed):** 3 real sentences, 8/9/7 tokens,
hidden=384, 25 LayerNorm sites/sentence (ROW 191, `discipline.md:17121`).

**Debt history:** ROW 191 flagged full architecture (layer count,
intermediate size, head count), the 133 MB weight-stream-vs-cache framing,
and a per-op profiler breakdown as three DEBTs — "not computable" pending
those facts. ROW 195 closes all three: architecture facts read directly off
the real model's own `config.json` (cached on-host,
`BAAI/bge-small-en-v1.5`, not assumed) and cross-checked against the ONNX
graph's own op histogram (`onnx.load` on the real `model.onnx`, 96 `MatMul`
nodes counted directly — matches the task brief's own citation); MAC/byte
counts derived from those facts; and a per-node-class profile run via the
landed `epilogue-profile-probe` feature (`proxima-onnx/examples/
bge_epilogue_profile.rs`).

### Architecture facts (MEASURED, not assumed)

| quantity | value | source |
|---|---|---|
| hidden size | 384 | `config.json` (cached HF snapshot) + ONNX graph output shape |
| encoder layers | 12 | `config.json`; cross-checked via `encoder.layer.{0..11}` node-name prefixes in the ONNX graph, 12 distinct indices found |
| attention heads | 12 (head_dim=32) | `config.json` |
| intermediate (FFN) size | 1536 | `config.json`; cross-checked via ONNX initializer shapes `(384,1536)`/`(1536,384)`, 12 of each |
| vocab size | 30522 | `config.json`; cross-checked via ONNX initializer shape `(30522,384)` |
| MatMul node count | **96** | direct ONNX graph node count (`onnx.load`, `op_type == "MatMul"`) -- matches the task brief's own citation |
| MatMul shape breakdown | 48x `(384,384)` (Q/K/V/O per 12 layers), 12x `(384,1536)` + 12x `(1536,384)` (FFN per 12 layers), 24x no-initializer (`Q@K^T`, `softmax@V`, 2/layer x 12) | direct node/initializer inspection; 48+12+12+24=96 |
| total initializer elements | 33,212,160 | direct ONNX initializer sum (`sum(prod(dims))`) |
| total initializer bytes (f32) | **132,848,640 bytes (126.7 MiB)** | 33,212,160 x 4; closes the 133 MB weight-stream DEBT -- the on-disk `model.onnx` is 133,093,490 bytes total (protobuf node/string overhead accounts for the ~245 KB difference from the tensor-payload-only figure) |

Re-prove command (no tracked path to the model — pass via `BGE_MODEL_PATH`,
same convention as `bge_eval.rs`):
```
cd <worktree> && BGE_MODEL_PATH=<local BGE checkout>/model.onnx python3 inspect_graph.py
```
(`inspect_graph.py` is a throwaway `onnx`-package inspection script, not
committed -- the MatMul-count/shape/initializer-byte facts it printed are
transcribed above and independently re-derivable from
`proxima_onnx::pipe::parse_complete` + `GraphProto` against the same
on-disk file, or from the cached HF `config.json` directly.)

### MAC count per sentence (DERIVED from the architecture facts above)

Per encoder layer, sequence length S, hidden H=384, intermediate I=1536,
12 heads (head_dim H/12=32):

- QKVO linear projections: `4 x S x H^2`
- `Q@K^T` + `softmax@V`: `2 x S^2 x H`
- FFN up + down: `2 x S x H x I`

| sentence | tokens (S) | MACs/sentence (12 layers) |
|---|---|---|
| "the cat sat on the mat" | 7 | 149,087,232 |
| "a cat is sitting on a mat" | 9 | 191,849,472 |
| "quantum physics explains atomic energy" | 8 | 170,459,136 |
| **mean across the 3 real sentences** | 8 (avg) | **170,465,280 (~0.1705 GMAC)** |

Cross-checked against a MEASURED counter (not just the formula above):
`proxima-tensor`'s own `instrument` feature reports `mac_ops=170,631,168`
averaged over 60 `evaluate_named` calls in the ROW 195 profile run (`DIAG
nsper reduce_f32_dense`, `bge_epilogue_profile.rs` run log) — **0.1% off**
the hand-derived 170,465,280, i.e. the formula and the counter agree.

Activation bytes/sentence (hidden-state read/write across 12 layers,
S~8, H=384, I=1536) are ~3.7 MB — DERIVED, not counter-measured, and
**not** the bottleneck below; weight bytes (132.85 MB, read once per
sentence at batch=1) dominate total bytes moved by ~36x.

### Binding constraint

Arithmetic intensity = 170,465,280 MACs / 132,848,640 bytes = **1.283
MACs/byte**. Ridge point at the two named candidate ceilings (NEON compute
48.5 GMAC/s, ROW 20; DRAM bandwidth 69.95-81.21 GB/s, ROW 176, midpoint
75.58 GB/s) = 48.5 / 75.58 = **0.642 MACs/byte**. The workload's AI (1.283)
sits **above** the ridge point, so **compute binds, not bandwidth** — the
same conclusion holds at all three real sentence lengths (weight bytes are
S-independent; compute time scales with S):

| sentence (S) | compute time (MACs / 48.5 GMAC/s) | bandwidth time (132.85 MB / 75.58 GB/s) | binding |
|---|---|---|---|
| 7 | 3.074 ms | 1.758 ms | compute |
| 8 (avg) | 3.515 ms | 1.758 ms | compute |
| 9 | 3.955 ms | 1.758 ms | compute |

**AMX DEBT CLOSED, 2026-09-01 — and it moves this lane's roofline.** ROW 189's
AMX band was flagged at conv shapes and had never been measured at BERT-style
batch=1 GEMM shapes. It has now been measured at exactly this lane's own
shapes, via `cblas_sgemm` on the width-tile route (branch
`perf/amx-width-tile`, `a03b96d`, quiet host, 5 repeats, paired against the
packed-NEON arm):

| shape | M=7 | M=8 | M=9 |
|---|---|---|---|
| QKVO `(M,384)x(384,384)` | 67.16 | 64.38 | 86.64 |
| FFN-up `(M,384)x(384,1536)` | 62.18 | 73.62 | 81.77 |
| FFN-down `(M,1536)x(1536,384)` | 67.55 | 79.12 | 87.88 |

All figures GMAC/s. **Every cell exceeds the 48.5 GMAC/s NEON per-core
ceiling** this lane's roofline was derived from, by 1.28x-1.84x — so the
3.515 ms/sentence figure below is a NEON-only ceiling, not this machine's
ceiling, exactly the substitution `rooflines.md`'s own matrix-ceiling note
(above, "the box's real ceiling is the max over its compute units") warns
against. These are Apple's `cblas_sgemm` rates, not this crate's own achieved
dispatch rate, so they are admissible as a machine constant under the
physics-only rule — but they are measured AT THESE SHAPES, and M=7-9 is the
binding limit: ROW 189's 467 GFLOP/s (233 GMAC/s) at an M=64 synthetic shape
shows the unit itself has far more headroom than this lane's tiny M can
extract. Carried as the shape-bound AMX ceiling, not as AMX's peak.

### Roofline vs current best

**Machine roofline (compute-bound), two units, both stated:**

- **AMX/Accelerate, measured at this lane's own shapes: 1.94-2.75
  ms/sentence** (170,465,280 MACs / 88.0 GMAC/s best cell = 1.94 ms;
  / 62.2 GMAC/s worst cell = 2.75 ms). THIS IS THE BINDING MACHINE
  CEILING for this box — it is the max over compute units, per the
  matrix-ceiling note above.
- **NEON per-core, 48.5 GMAC/s: 3.515 ms/sentence** (mean across the 3
  real sentences; range 3.074-3.955 ms). Retained as the PORTABLE ceiling
  — the number that binds on non-Apple aarch64 targets with no matrix
  coprocessor — never as this machine's ceiling.

The 3.515 ms figure was reported as "the" machine roofline for this lane
through ROWs 195-208. It was a NEON-only ceiling and understated the
machine by 1.28-1.81x. Corrected here at the first measurement that could
falsify it.

**Current best, 2026-09-01, by arm — no single path yet carries every
landed lever, so this is a set of measurements, not one number:**

| arm | M=7 | M=8 | M=9 | what it carries |
|---|---|---|---|---|
| `evaluate_named`, sealed `bge_eval` path | 18.44 | 18.01 | 19.98 | fusion only |
| arena + packing (ROW 206) | 12.13 | 11.47 | 13.42 | packing + arena, NO fusion |
| arena + packing + **Accelerate** | 13.14 | **11.52** | 13.50 | + AMX, still no fusion |
| per-sentence COLD, uncached lowering | 44.08 | 42.26 | 48.70 | what a production caller actually pays |
| per-sentence, lowering cached | 18.15 | 17.62 | 20.03 | lowering cache only |

All ms/sentence. CoV per cell is in the discipline rows; cells above the 5%
trust line are flagged there and not point-quoted here.

Two facts this table makes visible that the prior single-number form hid:
(1) `build_static_arena*` never calls `run_rewrite_worklist`, so every
packed/arena number above was measured WITHOUT fusion, and every fused
number WITHOUT packing — no measurement of the two together has ever
existed; (2) the sealed harness times evaluation only, excluding a
26-30 ms/sentence `lower_graph_pinned` cost (`proxima-onnx/src/lower.rs:
246-262` decodes and then `.clone()`s the full 133 MB weight set on every
call), so the shipping per-sentence cost was ~2.4x the headline figure.

**Gap to machine roofline:** against the AMX ceiling (1.94-2.75 ms), the
best measured arm (11.47 ms) is **4.2x-5.9x**; against the portable NEON
ceiling (3.515 ms), **3.26x**. The incumbent `onnxruntime` (5.6682
ms/sentence, CoV 1.76%, `intra_op=1 inter_op=1`, `scripts/onnx_reference/`)
is itself **2.1x-2.9x off the AMX ceiling** — the incumbent is a rung, not
the target.

### Per-node-class profile (closes the "no BGE-side profiler breakdown" DEBT)

`proxima-onnx/examples/bge_epilogue_profile.rs`, `epilogue-profile-diag`
feature (reuses the landed `epilogue-profile-probe` probe unchanged — no
new instrumentation), 20 iterations x 3 sentences = 60 `evaluate_named`
calls, 3-call warm-up excluded from the counters:

**SUPERSEDED — the ROW 195 profile below was taken at 25.62 ms/sentence,
before ROWs 198-208 landed. Re-run 2026-09-01 at `1f56f95`, same harness,
same 60 calls x 3 sentences (M=8 shown; M=9 and M=7 in the row):**

| node class | calls | ns total | % of attributed step time | ns/call |
|---|---|---|---|---|
| (a) reduce-fold | 7,200 | 788,519,278 | **83.70%** | 109,516.6 |
| — (a.1) gemm-shaped (96 `MatMul`s) | 5,760 | 786,407,606 | **83.47%** | 136,529.1 |
| — (a.2) small non-gemm (LayerNorm/pool) | 1,440 | 2,111,672 | 0.22% | 1,466.4 |
| (b) post-reduce epilogue | 4,320 | 42,120,755 | 4.47% | 9,750.2 |
| (c) everything else (softmax/glue/gather) | 34,620 | 111,476,293 | 11.83% | 3,220.0 |
| **total attributed** | 46,140 | 942,116,326 | 100% | 15.70 ms/sentence attributed vs **18.34 ms/sentence wall-clock** |

Prior (ROW 195, superseded): 88.65% / 3.15% / 8.20%, 23.09 attributed vs
25.62 wall-clock.

**The next lever, re-derived from the current profile — the ROW 195 reading
below it is wrong and is corrected here.** ROW 195 named "the
interpreted/generic dispatch path (not a tiled NEON kernel)" as the lever,
inferring a slow kernel from an 8.31 GMAC/s aggregate. That inference is
now falsified: the same width-tile kernel, measured in isolation against
packed panels at exactly these shapes (branch `perf/width-tile-accs`,
`15b02cf`, quiet host, CoV <1%), runs at **48.0-48.8 GMAC/s** — at the NEON
ceiling, on every BGE shape at every M. A 16-vs-24 accumulator sweep across
the same shapes moved it 2-4%, so register pressure is measured out too.

In-graph, the same kernel delivers **13.0 GMAC/s** (5,760 calls,
136,529 ns/call, 1.776 MMAC/call). **The 3.7x lives between "the graph asks
for a matmul" and "the kernel runs" — not in the kernel and not in register
blocking.** That is the open diagnosis, not a kernel-quality problem.

Two mechanisms already found inside that gap, both structural:
- `build_static_arena*` (`cpu.rs:598-719`) calls `bind::bind` directly and
  never `run_rewrite_worklist`, so the packed/arena path has NO fusion —
  every arena number was measured without it and every fused number
  without packing.
- Class (b) at 4.47% is confirmed near its floor, so `LayerNorm` epilogue
  work is NOT where remaining time lives — and ORT's own optimized graph
  (351 nodes: `LayerNormalization` x25, `BiasGelu` x12, `FusedMatMul` x12,
  `MatMul` x84) fires no matmul-count-reducing fusion either. ORT runs the
  same 96 matmuls we do at 5.6682 ms/sentence. The gap is throughput per
  matmul, not graph structure.

Re-prove command:
```
cd <worktree> && CARGO_TARGET_DIR=<scratch> BGE_MODEL_PATH=<local BGE checkout>/model.onnx \
  cargo run --release -p proxima-onnx --example bge_epilogue_profile --features epilogue-profile-diag
```

### Debts (remaining)

- The reduce-fold class (a) is not further split between the 48 `(384,384)`
  QKVO matmuls, the 24 `(384,1536)`/`(1536,384)` FFN matmuls, and the 24
  no-initializer `Q@K^T`/`softmax@V` matmuls — the probe attributes by node
  *kind* (`Keep::Reduce`), not by shape. A per-shape split (the FFN matmuls
  carry 4x the MACs/call of the QKVO matmuls) would sharpen the lever
  further; not built this session.
- Activation-bytes-moved (~3.7 MB/sentence) is a DERIVED estimate from
  hidden-state tensor sizes, not a counter-measured figure — it does not
  change the binding-constraint conclusion (weight bytes dominate by ~36x)
  but is flagged as unmeasured, not guessed-and-hidden.
- ~~AMX candidate rate remains unmeasured at BGE's own GEMM shapes.~~
  **CLOSED 2026-09-01**, `perf/amx-width-tile` `a03b96d` — 62.2-88.0 GMAC/s
  measured at this lane's own shapes; it moved this lane's roofline from
  3.515 to 1.94-2.75 ms/sentence. See the AMX table above.
- **THE OPEN GAP: 3.7x in-graph vs isolated, unattributed.** The width-tile
  kernel runs at 48.0-48.8 GMAC/s isolated and 13.0 GMAC/s in-graph on the
  same shapes. Kernel quality and register blocking are both measured out.
  Candidate mechanisms not yet discriminated: route miss (some of the 96
  MatMuls never reaching the width tile — the N==0 class, prior in ROWs
  197/199), in-route overhead outside the microkernel, residual cold-cache
  after packing, and per-call `shape::infer` + `bind::bind` + per-node alloc
  (`cpu.rs:464-479`). Diagnosis in flight on `perf/route-census`.
- **No arm carries every landed lever.** Fusion, packing+arena, Accelerate,
  and the lowering cache have each been measured alone or in pairs; the
  combined path does not exist yet (`perf/unify-arena-fusion` in flight).
  Every ratio in this lane is therefore a lower bound on what the landed
  work can do, and none of them may be summed — composition is measured,
  never derived.
- **`lower_graph_pinned` costs 26-30 ms/sentence** and is excluded from the
  sealed harness's timed window (`bge_eval.rs:96` vs `:98`). Root cause:
  `lower.rs:246-262` decodes and then `.clone()`s the full 133 MB weight
  set on every call. Cached: -58.3% to -58.9% per sentence
  (`perf/plan-cache` `e5bdb8e`). Until that lands, every published
  ms/sentence in this lane understates the shipping cost by ~2.4x.

---

## Summary table

Three columns where a self-limit exists: **machine roofline** (physics-only,
the ceiling), **current-architecture floor** (self-limit, this crate's own
best-case dispatch cost — N/A where no such composition exists), and
**current best** (measured). Gap is always reported **to the MACHINE
roofline** — the chase number, never to the self-limit. Where the machine
roofline itself is unmeasured (named DEBT), the gap column falls back to a
reference-only ratio against an incumbent's own achieved rate, flagged as
such — never silently promoted to a machine-roofline gap.

| lane | binding constraint (machine) | machine roofline | current-architecture floor (self-limit) | current best | gap-to-MACHINE | debts |
|---|---|---|---|---|---|---|
| mnist f32 inference | NEON per-core compute | 0.057 ms/image | N/A (no composed self-limit derived this lane) | 0.393907 ms/image (ROW 190) | **6.91x** | full-net AMX roofline unmeasured |
| train-step (MLP 784-128-10, b32, Adam) | NEON per-core compute (6,545,408 MACs @ 48.5 GMAC/s; bandwidth candidate 41.8-48.6us is non-binding, AMX candidate unmeasured at this shape) | **~135 us/step** | ~0.768-0.778 ms/step, composed from this crate's own 0.26-0.29 ns/element interpreted-dispatch rate (DERIVED) | 1.3699 ms/step (ROW 179) | **~10.1x** (pytorch itself is 2.8x off this ceiling, ROW 159) | whole-step MAC count and byte-traffic enumeration are both formula-derived, not counter-measured; AMX candidate unmeasured at M=32 |
| q4_K int8 dot (decode GEMV) | co-bound: kernel is at/just-over the DRAM streaming ceiling | 124.4-144.4 GMAC/s (bandwidth) | N/A (kernel already at the physics ceiling — no distinct self-limit below it) | 147.72-150.60 GMAC/s (kernel, ROW 116) | **effectively closed, ~1.0x** (kernel already at ceiling; gap lives in production orchestration, 3.74-3.81x on w=1) | read-only bandwidth ceiling never independently measured |
| q4_K GPU decode (Metal, same MAC constant) | **GPU-BW DEBT** — no independent GPU streaming-bandwidth ceiling measured (`membw_probe`'s Metal arm is the wrong probe shape, reduce-to-scalar not streaming copy) | **DEBT — not measured** (GPU-BW DEBT; not substituted with a spec-sheet figure) | N/A (no composed self-limit derived this lane; kernel-only 124.7 GMAC/s, 57.032 ms/token, ROW 193, is a measured point, not a derived floor) | **69.9 ms/token** full decode loop (103.6 GMAC/s, ROW 193) | **not computable to machine** (roofline unmeasured); reference-only vs incumbent llama.cpp Metal achieved **17.1 ms/token** (416.1 GMAC/s, ROW 193) — **4.02x under incumbent** (3.34x on kernel-only) | GPU streaming bandwidth ceiling never cleanly measured; per-shape numbers op-isolated, not batched |
| BGE-small embedding (7-9 tok, 96 MatMuls, hidden=384/12 layers/12 heads/intermediate=1536) | AMX/Accelerate compute, measured at this lane's own shapes (62.2-88.0 GMAC/s, `perf/amx-width-tile` `a03b96d`); NEON 48.5 GMAC/s is the PORTABLE ceiling, not this box's; AI=1.283 MACs/byte is above the 0.642 ridge point vs the 69.95-81.21 GB/s candidate, so bandwidth is non-binding | **1.94-2.75 ms/sentence (AMX, binding)**; 3.515 ms/sentence (NEON, portable) | N/A (no composed self-limit derived this lane) | 11.47 ms/sentence best arm (arena+packing+Accelerate, no fusion); 18.01 on the sealed fusion-only path; 42.26 per-sentence cold | **4.2x-5.9x to AMX** (3.26x to the portable NEON ceiling) | no arm carries fusion AND packing together (`build_static_arena*` never calls `run_rewrite_worklist`); activation-bytes DERIVED not counter-measured; reduce class not split by MatMul shape; in-graph GEMM runs 13.0 GMAC/s vs 48.0-48.8 for the same kernel in isolation, 3.7x unattributed |

**A roofline derived from one compute unit is not a machine roofline.** The
BGE lane carried "3.515 ms/sentence" as its machine ceiling through ROWs
195-208. It was NEON-only, on a box whose matrix unit measures 62.2-88.0
GMAC/s at that lane's own shapes — so the real ceiling was 1.94-2.75 ms and
every gap-to-roofline figure in that lane understated the target by
1.28-1.81x. This is the same failure mode as the train-step self-limit
below, one level up: not "our own cost mistaken for physics," but "one
unit's physics mistaken for the machine's." The matrix-ceiling note at the
top of this doc existed precisely to prevent it and was not applied. Any
lane whose work is matrix-shaped must state BOTH units or it is
understating its own target.

**The chase number is the machine roofline gap, not the self-limit gap, and
never a bare incumbent-reference ratio.** Train-step's self-limit gap alone
(1.3699 / 0.773 = 1.77x) understated the real target by 5.7x — closing
dispatch cost to its own best-case floor still leaves 5.7x on the table
against what the hardware allows. The GPU decode lane's 4.02x-under-incumbent
figure is the same trap in the other direction: it is not a gap-to-machine
at all (no GPU bandwidth ceiling exists yet), so it is reported as a
reference ratio, not conflated with the physics gap the other lanes carry.

All citations above point at `proxima-tensor/docs/discipline.md`; re-derive
by opening the cited `ROW`/`file:line` pairs directly.
