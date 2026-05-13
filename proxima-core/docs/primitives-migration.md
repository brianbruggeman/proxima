# core primitives migration — telemetry → proxima-core

Moving general concurrency/measurement primitives out of `proxima-telemetry`
(std, recorder-domain) into `proxima-core` (no_std) so prime/pipe, downstream
consumer crates, and the DPDK/SPDK bare-metal paths can use them. Discipline reminder: **no_std AND
no-alloc** (Principle 3/11) — the bare-metal tier forbids heap.

## Landed (on `main`)
- **`Ring<T>` (lock-free MPMC) → `proxima-core::ring`** (`fb70adc2`). Builds
  std / no_std+alloc / loom. Telemetry-error coupling → ring-local `CapacityError`;
  `proxima_core::arch` prefetch → `crate::arch`. Telemetry re-exports
  `proxima_core::ring` (zero consumer churn). Unit + loom tests moved with it.
  - **Tier caveat:** Ring is no_std **+ alloc** — the buffer is `Box<[Cell<T>]>`.
    A no-alloc tier (const-cap `[Cell<T>; N]`) is the follow-up for bare-metal.
- **Ring benches → `proxima-core/benches`** (`a17ef512`). `bench_ring` /
  `bench_ring_decompose` (Ring vs flume/crossbeam/tokio) follow the primitive.
- **`StaticRing<T, const N>` — the no-alloc ring tier.** Inline `[Cell<T>; N]`
  buffer (no heap); compiles + runs bare-metal (`proxima-core --no-default-features`,
  no alloc). The alloc `Ring` (Box) and `StaticRing` now share ONE lock-free
  algorithm (`cells_push`/`cells_dequeue` free-fns over `&[Cell<T>]`) — loom
  model-checks it once and covers both. `Ring`/`Drainer` gated on `feature="alloc"`;
  `StaticRing` always on. 5 tests + loom green. Bench: StaticRing 6.87µs/1024 vs
  Ring 6.26µs/1000 — the ~7% is per-iter construction (inline init vs Box); the
  push/pop hot path is identical, so steady-state matches at no-alloc. `N` power of
  two >= 2. TODO: `const fn new` (blocked on per-index seq init).

## Landed (cont.)
- **`Histogram<V>` → `proxima-core::histogram`.** No_std + **no-alloc** primitive
  (`[AtomicU64; 32]`), builds bare-metal. The `Tag`/attrs decision resolved
  WITHOUT an API reduction: core `record(value)` is **tagless** (attrs are a
  telemetry export-layer concern — C9 shards by attr-set at the registry, not in
  the bucket-counter primitive); telemetry's `histogram!` tagged form stays,
  validating the tag syntax and recording tagless in v1. Telemetry re-exports
  `proxima_core::histogram`; 20 histogram tests moved to core; all suites green.
- **`PerCore<T>` → `proxima-core::per_core`.** Both design axes resolved: (1) the
  **routing primitive** is pure `slot(index)` + `count()` (no_std, no ambient
  state — the caller names its core), and `local()` is a **std-gated convenience**
  (`feature="std"`, TLS sticky ticket); a bare-metal caller passes prime's per-core
  id to `slot`. (2) **Two storage tiers:** heap `PerCore<T>` (`Vec`, `feature="alloc"`)
  and inline no-alloc `StaticPerCore<T, const N>` (`[T; N]`, builds bare-metal). The
  ticket is a per-thread monotonic value modded by `count` at access (NOT a cached
  slot index — caching would index a smaller `PerCore` out of bounds). Telemetry
  re-exports `PerCore`; the originally-untested move gained 3 tests (static routing,
  alloc new/from_vec, sticky-local). All suites green.

## Next candidates — each needs a DESIGN DECISION

### ~~`Histogram<V>` → core~~ — DONE (see above). The Tag question resolved: tagless core, tagged macro stays at the telemetry layer.
<details><summary>original decision (kept for the record)</summary>

The param was `_tags` (ignored in v1) BUT the `histogram!` macro exposes a
tagged form reserved for **C9** attr-set sharding. Chosen: core is tagless (attrs
are an export-layer concern), so no API reduction and no core↔Tag coupling.
</details>
leave Histogram in telemetry until C9 lands.

### ~~`PerCore<T>` → core~~ — DONE (see above). Both axes resolved: pure `slot`/`count` primitive + std-gated `local()`; `PerCore` (Vec/alloc) and `StaticPerCore` (`[T;N]`/no-alloc) tiers.

### `Ring<T>` no-alloc tier
Give the just-moved Ring a bare-metal tier: const-cap `[Cell<T>; N]` buffer (or a
caller-provided-storage design, Principle 11) instead of `Box`. Redesign of a
lock-free primitive — bench it (disciplined-component) against the alloc tier.

### lossless-backpressure (`deliver` / `block_until_pushed` / `SlotGate`) — std today, futex-swappable (NOT impossible)
**CORRECTION (2026-07-01):** an earlier seal here claimed the sync park is
"irreducibly std." That was WRONG — it conflated "no primitive exists in-workspace
today" with "impossible without std." A sync park is an **OS syscall** (Linux
`futex`, Windows `WaitOnAddress`, macOS `__ulock_wait`), reachable from `no_std`
via `libc`; `std::sync::Condvar` is itself futex-backed on Linux. The clean no_std
primitive is Mara Bos's **`atomic-wait`** crate (`#![no_std]` + `libc`,
`wait`/`wake_one`/`wake_all`).

**Accurate constraint** — `SlotGate` has two waits and only one is on the no_std path:
- `block_until_pushed` → `freed.wait()` — the **producer** parking for a freed
  slot — is **untimed** → maps directly to an `atomic-wait` futex, NO per-platform
  timeout FFI. This is the piece a no_std/DPDK consumer needs.
- `pump_park` → `data_ready.wait_timeout(flush_interval)` — the **sync-thread**
  pump — is the only **timed** wait. A no_std deployment doesn't use it: it runs
  the **async prime pump** instead, already no_std-clean (`PumpWait` future +
  prime timer supplies `flush_interval`). Timed futex-with-timeout FFI is the one
  fiddly bit, and it's OFF the no_std path.

So `SlotGate` CAN be no_std with a bounded change (swap `freed` Condvar →
`atomic-wait` futex; keep the timed std-thread `pump_park` as the std fallback).
It stays std **today** on YAGNI (Principle 1): no no_std/DPDK consumer of
lossless-backpressure exists yet, and the futex swap is a known move to make when
one appears — not a wall.

Prime today has no no_std sync-park to reuse (only `crossbeam_utils::sync::Parker`,
std, gated `runtime-prime-bgpool`→std) — so the futex path means `atomic-wait` (or
a small `libc` futex wrapper), not reuse.

**RISC win that DID land** (`f756b9cc`): the async size-trigger half hand-rolled
`std::sync::Mutex<Option<Waker>>` + lost-wakeup logic — exactly
`atomic_waker::AtomicWaker`, already a workspace dep and already what prime's inbox
uses. Swapped in → async-wake half now lock-free + no_std-clean.

## Telemetry → pipe-forms collapse — 3-tier design of record

Directive: **everything no-alloc/alloc + no_std/std tiered where possible; the
recorder collapse must be 3-tier by shape, not alloc-only.** The telemetry
emit→ring→drain→export path is a parallel re-implementation of the pipe-forms
algebra (`SinkFront`≈`RingSink`emit, `Drainer`≈`DrainSource`, exporters ARE
`SendPipe`); collapse it onto those traits, tiered. Dep chain (no cycle):
`telemetry → pipe → pipe-forms → core`.

Tier capability split (the load-bearing constraint):
- **Drop policies** (`FailMode::{DropOldest,DropNewest,FailClosed}`) — all tiers.
- **Lossless via producer-assist** (on full, the producer drains+exports itself —
  no park) — all tiers incl. no_std. Uses `try_enqueue` (hand-back).
- **Lossless via park** (wait for a pump to free a slot) — **std only** (Condvar;
  no no_std sync-park in-workspace, futex-via-`atomic-wait` is the documented
  later option). This is `SlotGate` → promote to `ParkGate`.

Shape: one generic engine per layer, tier aliases on top. `RingSink`
(pipe-forms) is NOT reusable — it's `&[u8]`-arena + single-writer; `StaticRing`
(MPMC, generic) is the no-alloc storage.

### Landed
- **`BoundedQueue<R: RingStorage>` → proxima-core** (`326f0200`) — generic over
  `RingStorage` (assoc `Item`); `StaticBoundedQueue` (no-alloc) + `HeapBoundedQueue`
  (alloc) share ONE overflow algorithm. `try_enqueue` added. proxima-pipe re-exports
  (zero churn). Both tiers proven via one assertion body; builds no-alloc.
- **`SinkFront<R: RingStorage>` → proxima-pipe-forms** (`4c3a9d1c`, steps 4-6) — one
  generic engine (emit / emit_lossless / drain_one / lifecycle FSM / demand flag);
  `StaticSinkFront` (no-alloc, bare-metal) + `HeapSinkFront` (alloc). Demand flag is an
  inline `AtomicBool` — dropped the Arc `AtomicGate`/controller split (single `new()`,
  `arm()`/`disarm()` on the sink). `emit_lossless` = producer-assist (no park, T0-legal).
  proxima-pipe `SinkFront` is now a thin `Arc`+`Deref` handle. Enums + engine proven on
  both tiers.

**FOUNDATION COMPLETE:** the 3-tier bounded-queue + sink primitives exist and are proven.

### Telemetry collapse LANDED (`de463fc4`) — onto BoundedQueue, not SinkFront
The full-map surfaced the load-bearing finding: **`SinkFront::emit` does a per-emit `record_append`
(SeqCst) for its quiescence FSM — that would tax the emit hot path the deferred-metric-fold work
tuned mutex-free/zero-alloc.** `BoundedQueue::try_enqueue`/`enqueue` on the not-full path is
byte-identical to `ring.push`. So the recorder collapsed onto **`BoundedQueue`** (the layer it needs),
NOT `SinkFront` (the demand-FSM layer, which serves standalone consumers). Hot-path-neutral by
construction (`try_enqueue` is an `#[inline]` forwarder to `Ring::push`).
- `RingSet`'s 7 streams: `Ring<T>` → `HeapBoundedQueue<T>`; deleted the per-core `dropped` atomic +
  `note_drop` (folded into the streams' own counters; `Recorder::dropped` sums them). `deliver` keeps
  the recorder's dynamic per-emit Block/Drop routing on top via `try_enqueue` + `BoundedQueue::note_drop`;
  the `on_drop` closure dropped out of all 8 emit sites. Drain routes through `BoundedQueue::drain_into`.
- 336 incumbent + 343 deferred green; all feature combos build; clippy clean; no external `RingSet` users.

**Two follow-up corrections (both were sloppy first-pass claims):**
- **Park is futex now, not std** (`fb0fc7c6`). I wrote "std lossless park" — wrong, we'd already established a
  sync park is an OS syscall, not std. The producer's wait-for-a-freed-slot (the 0%-drop guarantee) is UNTIMED,
  so it maps onto an `atomic-wait` futex: `wait(&freed_epoch, snapshot)` + pump `wake_all`. Mutex-free (honours
  the original directive) + no_std-capable. Only remaining Condvar = the std-thread pump's TIMED `data_ready`
  wait (futex has no timeout; a no_std pump uses the async prime path).
- **Drain drives the algebra now** (`ecb4a6df`). I said the drain "doesn't fit DrainSource" — the REAL reason it
  didn't fit: `DrainSource`/`PollSource` are `&mut self` single-consumer; the recorder's drain is `&self`
  MULTI-consumer (drainers partition cores). The algebra lacked a multi-consumer source. Added **`BatchSource`**
  (`&self` owned-batch, MPMC) impl'd by `BoundedQueue`; `drain_owned` is now generic over it. Both drain ends are
  algebra traits (BatchSource source + SendPipe sink); the batch-encode between is telemetry wire format (domain).
- `assisted` counter stays recorder-specific (producer-assist observability, no queue-primitive analogue).

### Loose-end sweep (`9a1c2a74`, `3112a8f7`)
- **Producer-assist unified** — the assist loop (try_enqueue; make room; retry) was hand-rolled twice (recorder
  no-pump branch + `SinkFront::emit_lossless`). Extracted as `BoundedQueue::enqueue_assisting(item, on_full)`:
  `on_full` returns true=retry / false=give-up (item back via `Err`). Recorder never gives up (yields on
  no-progress); SinkFront gives up → FailMode. One loop, two edge policies. no_std (yield/park live in `on_full`).
- **`atomic-wait` → workspace dependency** (was a direct version dep).

### Park EXTRACTED (`3dde59e3`) — it was a punt, now a primitive
`SlotPark` → `proxima_core::park` (behind the `park` feature): a mutex-free futex (`atomic-wait`) over one
`AtomicU32` epoch, no_std (OS-backed; absent on bare-metal no-OS), **const-constructible** (backs a `static`).
A parker isn't telemetry-specific — a queue/arena/pool/rate-limiter all want block-until-room — so it's a
first-class primitive now; telemetry's `SlotGate` holds one instead of hand-rolled `freed_epoch`/`waiters`/`parked`.
Two threaded unit tests prove the handshake. `data_ready` Condvar stays as the std-thread pump's wake — that's
FINE because there are two pump impls (std-thread Condvar + async prime waker); a timed Condvar has no no_std
equivalent (futex has no timeout), so the no_std pump is the async one.

### Genuinely remaining — one item, real reason
- **`StaticRing::const fn new` — blocked by stable Rust**, not by design. Each cell's `sequence` seeds to its
  index; stable const array init can't express per-index values (no const `from_fn`; `array_assume_init` is
  unstable; a size-generic `transmute` is `E0512`). So the no-alloc tier can't be a plain `const static` until
  that stabilises. Heavily documented at `StaticRing` with a `TODO(const-new)` tied to whichever feature lands.

### Remaining sequence (steps 7-9 = one validated telemetry-integration pass)
7. **proxima-pipe `park_gate.rs`** — promote `SlotGate` (crate-private today) →
   `ParkGate` + `emit_parked`, `#[cfg(feature="std")]`.
8. **proxima-telemetry `RingSet`** — 7 streams become `SinkFront<T, Ring<T>>`; delete
   the bespoke `dropped`/`assisted` atomics (fold into `SinkCounters`). `OverflowPolicy::
   {Block,Drop}` maps to `FailMode::{FailClosed,DropNewest}`. **RISKY: 343-test emit
   path — its own validated pass.** Heterogeneous 5-stream `RingSet` stays typed (no
   `DrainFanIn` — that's homogeneous-only).
9. **proxima-telemetry** — drain loops → `SinkFront::drain_one`/`drain_batch`; rerun
   drain bench within CoV of baseline.
10. **workspace sweep** — grep confirms no external reach into `RingSet` fields or
    `proxima_pipe::{bounded,sink}` internals.

## Session context (what produced this)
Extracted `Ring` while landing **`deferred-metric-fold`** in telemetry: moving the
span-duration histogram fold off the emit hot path via the per-core `Ring` +
`deliver`/Block/producer-assist path (lossless, mutex-free/zero-alloc emit, ~1.9x
faster @ 8 threads, 0% drop). See `proxima-telemetry/docs/deferred-metric-fold/`.
That work reused `Ring`/`PerCore`/`Histogram` "differently" — which surfaced that
they're general primitives mis-located in telemetry, hence this migration.
