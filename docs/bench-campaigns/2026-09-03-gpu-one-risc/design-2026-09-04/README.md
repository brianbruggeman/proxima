# 2026-09-04 design tournament — RISC/pipe shape for omega dispatch

Design-only session, no cargo, no measurement run. `design-final.md` is the design of record; every
other file here is the trail that produced it.

## Round 1: design-A vs design-B vs design-AB

Three independent designs answering the same task (`design-task.md`) against the same evidence
(`dispatch-census.md`, `byte-levers-probe.md`). Three judges scored all three, each under a
different randomized Plan-1/2/3 anonymization so the mapping differs per judge file:

- judge-1: Plan-1=design-A, Plan-2=design-B, Plan-3=design-AB — totals 37/46/**69**
- judge-2: Plan-1=design-B, Plan-2=design-AB, Plan-3=design-A — totals 45/**62**/41
- judge-3: Plan-1=design-AB, Plan-2=design-B, Plan-3=design-A — totals **68**/39/50

design-AB is highest under all three anonymizations. Borda (0/3/6 per judge, worst/middle/best):
design-AB 6+6+6=18, design-B 3+0+3=6, design-A 0+3+0=3. **design-AB first, unanimous.**

## Round 2: design-AB vs design-B2 vs design-AB2

design-AB (round-1 winner) was critiqued (`critique-AB.md`, alongside `critique-A.md` and
`critique-B.md` on the round-1 losers), and two follow-on designs were produced against the
critique: design-B2 (from design-B's shape) and design-AB2 (from design-AB's shape, closing the
holes `critique-AB.md` named). Three judges scored the round-2 field:

| judge | AB | B2 | AB2 |
|---|---:|---:|---:|
| judge-r2-1 | 42 | 55 | **59** |
| judge-r2-2 | 48 | 58 | **60** |
| judge-r2-3 | 48 | 59 | **63** |

design-AB2 highest for all three judges. Borda: design-AB2 18, design-B2 9, design-AB 0.
**design-AB2 first, unanimous.**

## The stop decision

The incumbent changed between rounds (design-AB won round 1, design-AB2 — a different design, not
a strict refinement carrying the same shape forward untouched — won round 2), so there is no strict
two-round convergence on one candidate. The panel was unanimous in both rounds and the cost of a
third round (another two designs, three more full judge passes) was weighed against the marginal
gap already closing (round-1 spread 69-37=32 points; round-2 spread 63-42=21 points, and the
winning margin over second place narrowed from 7 to 3-4). Round 2 stopped the tournament and
design-AB2 was taken forward for direct repair rather than a round 3.

## design-final: design-AB2 with the round-2 panel's holes closed in place

`design-final.md` is not a round-3 candidate — it is design-AB2 with the specific holes the round-2
judges named (`judge-r2-1.md`, `judge-r2-2.md`, `judge-r2-3.md`, all three scored "missing steps 2"
on the same gap) fixed against the shipped source, re-opened at `HEAD ce05362`:

- **section C written in full** — the round-2 panel's largest hole: design-AB2 asserted the generic
  model-program story rather than writing it. design-final derives it from `proxima-tensor/src/lib.rs`
  and `spec.rs` directly: the `config` feature gate, `LayerBindings` replacing 21 swappable `NodeId`
  positional arguments, the `cached_len` leaf deletion, and the new-architecture capability table.
- **decision 1's rationale corrected** — design-AB2's original reason for choosing `Loop` over
  `ReduceChain` (that `Band` cannot state causal and sliding-window jointly) was false and is
  withdrawn; the decision itself (`Loop`) stands, re-argued on declared-vs-derived semantics instead.
- **twelve support types resolved** — the round-2 panel counted twelve types appearing in
  signatures with no field list. Six are now defined with fields (`CommandRange`, `ExtentDigest`,
  `CodecMask`, `KernelKey`, `DraftSpan`, `TokenWindow`, `ResidencyPlan`/`SlotBinding` — collapsed
  from AB's `ResidentSet`), seven are deleted with the call site shown unchanged (`Wiring`,
  `Bindings`, `Grid`, `UniformPatch`, `Completion`, `ResidentSet`, `Budget`).
- **mixed engines, §B.4** — the owner's questions answered directly: `Backend`'s seven variants
  (three executable, four `NotImplemented`) collapse to `Engine::{Cpu, Gpu}` plus a `GpuDriver`
  resolved once per target; "mixed" is not a third engine, it is a placement field on
  `ScheduledOp`, with a hybrid CPU+GPU worked example and an open card (D0c) to measure whether
  concurrent streaming beats the GPU-alone ceiling at all before any placement rule ships.

## Still open (quoted verbatim from design-final.md, closing block)

> Still open: (1) `ScatterBounds::{Fault, Drop}` and its interaction with the binding `Domain` (a
> dropped scatter is an out-of-domain write) — the semantics of the elision lever, unworked; (2) the
> `sized::` const table with per-const overflow policy is not re-tabulated, and `crit #6`'s
> reachability gate (`PACKED_ROW_BLOCK_SIMDGROUPS`'s only consumer is unreachable — a swept constant
> that never reached a dispatch) needs one cell per new const, including the three added here
> (`PACKED_ROW_ACTIVATION_GROUP`, `MAX_INLINE_CONSTRAINTS`, `MAX_PLAN_SLOTS`); (3) the CPU sink's own
> allocation budget under `matmul_worker_count()` threads is not stated, and B8's **A** gate needs
> it; (4) `Q1-Q5` are one row for five levers — each needs its own row with its own MB/pass before D3
> reports.

## Headline arithmetic (design-final.md §D.4)

D0 (device streaming ceiling) is measured, not assumed: best cell `nocopy_resident`, wide grid,
4 GB, run 1 = 264.29 GB/s (CoV 26.84%), run 2 = 237.79 GB/s (CoV 29.40%), on a box carrying 30-47
concurrent builder processes; 35 of 36 measured cells exceed the 5% trust line.

At A = 2.18, the elision stack clearing its quality gate at favourable densities, and the attention
arm fixed at its own byte floor, the favourable stack reaches **4.46 ms/token at the ceiling high
(264.29 GB/s)**, 4.82 ms at the ceiling low (237.79 GB/s) — both miss the 3.5 ms/token target.

Meeting 3.5 ms/token requires bandwidth **B_w >= 379.5 GB/s**, which is **1.44-1.60x above the
measured ceiling of 237.79-264.29 GB/s** — not a tuning target, unreachable on this host under the
measurement recorded here.

Today's matvec stream runs at **173.6 GB/s**, which is **66-73% of the measured ceiling** (65.7-73.0%
precisely) — the largest single remaining term, worth more than every lossy byte lever combined if
closed.
