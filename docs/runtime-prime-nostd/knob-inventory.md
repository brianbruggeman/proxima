# Primitives for no_std/alloc — knob inventory (rev 2)

**Goal (user, 2026-05-21):** replicate the proxima-telemetry c1-c15 pattern
for the no_std/alloc cliff. The user composes a precise system shape via
Cargo features + Profile TOML.

**Correction from rev 1**: the c1-c15 pattern is NOT "add many small feature
flags to existing code". Each c* is a **greenfield primitive** built from
scratch, competing against **named incumbents** on the incumbent's **home
turf** (per disciplined-component gate point 13). The Cargo features compose
the primitives; the primitives are the work.

Reference: `rust/docs/telemetry/discipline.md`.

  - C1 ring: SPSC ring vs flume::bounded / crossbeam_queue::ArrayQueue /
    tokio::sync::mpsc / CausalIndex Vec<Mutex<Vec<_>>>. Measured **11.6× faster
    than crossbeam at 1k cap, 4.4× faster at 1M**. ~3.3 ns/item at 1M cap.
  - C2 id: W3C traceparent parser vs faster-hex (SIMD) / hex (scalar) /
    opentelemetry::trace::TraceContextExt. Measured **5.4× faster than
    opentelemetry parse**; matches NEON faster-hex on 16-byte payload.
  - C3-C15: each carries its own discipline-log row, bench numbers, opt-sweep
    table, and design-favors labels. `legacy` module preserves the
    pre-component surface for compatibility.

## What this means for no_std/alloc

The work is **primitives**, not feature flags. Each primitive:

1. Has a clear public API (a struct + a small set of methods).
2. Has a named incumbent on the bare-metal/no_std ecosystem (heapless,
   embassy, riscv-rt, etc.).
3. Has a home-turf bench arm where the incumbent's design point is engaged.
4. Lands behind its own Cargo feature flag (default off).
5. Composes via explicit `feature = ["..."]` dependencies.

When the user picks `PROXIMA_PROFILE=embedded-mqtt-gateway`, the Profile
selects a specific composition of primitives (e.g. inbox-const + executor-const
+ timer-const + embedded-tls + no cancellation + heapless headers). The user
gets exactly that shape, no more.

## Primitives to build (named incumbents + design-favors)

### A1. `prime-inbox-const` — stack-backed SPSC ring

**In flight as DCa.** Greenfield variant of `prime/src/core/inbox.rs`
using `[Lane; N]` + `[Slot; CAP]` const-generic storage instead of
`Box<[Lane]>` + `Box<[Slot]>`. Same SPSC algorithm (Acquire/Release atomic
pair + CLOSED_BIT quiesce marker); different backing.

| incumbent | design-favors | workload |
|---|---|---|
| `heapless::spsc::Queue<T, CAP>` | incumbent | SPSC, stack-backed — heapless's home turf |
| `embassy_sync::channel::Channel<NoopRawMutex, T, CAP>` | incumbent | bare-metal Channel — embassy's home turf |
| `prime::core::inbox` (inbox-alloc) | proxima | the same algorithm on heap |

Must meet or beat heapless::spsc on CAP=16, 256, 4096. Equal-ish on embassy
(different cap-vs-channel-Cap convention; document the scope). Strictly equal
to inbox-alloc (same algorithm, different backing).

### A2. `prime-executor-const` — stack-backed slab executor (DCb)

Greenfield variant of `prime/src/core/local_executor.rs` with const-generic
`[Slot; N]` slab. No `Vec<Slot>`, no `Box<Slot>` runtime growth. Per-slot
waker is built into a fixed-size waker pool. Static `RawWakerVTable` as today.

| incumbent | design-favors | workload |
|---|---|---|
| `embassy::executor::Spawner` (with static task pool) | incumbent | bare-metal static-pool executor — embassy's home turf |
| `tokio::runtime::Builder::new_current_thread` | incumbent | std current-thread runtime (close shape) |
| `prime::os::runtime::PrimeRuntime` (alloc) | proxima | the same executor on heap |

Must meet or beat embassy on a fixed N-task throughput bench (each task does
fixed work, N=64). Document if we lose under task spawn/despawn churn (we
will — embassy's static task pool is specifically tuned for that).

### A3. `prime-timer-const` — const-generic hierarchical timer wheel (DCc)

Greenfield variant of `prime/src/core/timer.rs` with const-generic
`[Vec<u32>; LEVELS]` slot heads + fixed-cap entry slab. No `Vec` growth
on the entry list; cap is reachable via const-generic.

| incumbent | design-favors | workload |
|---|---|---|
| `embassy_time::Timer::after` + scheduler | incumbent | bare-metal Timer — embassy's home turf |
| `tokio::time::sleep_until` + `tokio::time::driver` | incumbent | tokio's time driver (close shape) |
| `prime::core::timer::TimerWheel` (alloc) | proxima | the same wheel on heap |

Must meet or beat embassy timer arming + firing on a 100k-timer scheduling
bench. Document the per-tick granularity difference (tokio is 1ms, embassy
is per-MONOTONIC_COUNTER_RESOLUTION).

### A4. `prime-inline-task-const` — typed inline task with explicit CAP (DCd)

`inline_task.rs` already has 56-byte `MaybeUninit` inline storage. DCd
exposes the buffer size as a const generic (`InlineTask<F, const N: usize>`)
so users can pick a tighter or looser inline budget for their workload.
Currently the constant `INLINE_TASK_BYTES = 56` is hardcoded.

| incumbent | design-favors | workload |
|---|---|---|
| embassy's `#[embassy_executor::task]` macro (static futures) | incumbent | bare-metal static-pool task storage — embassy's home turf |
| `Box::pin(async move { ... })` | incumbent | the standard allocation path |
| `InlineTask` (alloc-backed today) | proxima | typed-spawn on heap-fallback |

Must meet or beat `Box::pin` on small futures (≤56 bytes); equal-or-better on
oversize futures (where Box-spill kicks in).

### A5. `core-error-typed-const` — no_std error type variants

DC2 added typed sub-enum variants (`DecodeError`, `EncodeError`, etc.) but
they still use `Box<dyn core::error::Error + Send + Sync>` for sources, which
requires `alloc`. The no-alloc variant uses `&'static str` chains or stack-
allocated error structs (a la `embedded-hal`-style).

| incumbent | design-favors | workload |
|---|---|---|
| `snafu` (no_std + alloc) | incumbent | typed error chains with source preservation |
| `anyhow::Error` | incumbent | type-erased boxed errors (std-bound) |
| `embedded_hal::Error` | incumbent | bare-metal error trait |
| `ProximaError` typed variants (alloc) | proxima | our current typed surface |

Construction cost + size_of comparison. Trade-off: no `source()` chain
without alloc, but `&'static str` context is cheaper.

### A6. `pipe-body-const` — stack-backed Body without `Box<dyn Stream>`

`proxima_pipe::body::BodyStream = Pin<Box<dyn Stream<Item = ...> + Send>>`
is fundamentally alloc-bound. A const-variant could carry a small enum of
known stream shapes (single Bytes, iter of N chunks, polling closure) without
heap.

| incumbent | design-favors | workload |
|---|---|---|
| `bytes::Bytes` (single-shot, alloc) | incumbent | one-shot body, heap-backed |
| `heapless::Vec<u8, N>` carried inline | incumbent | bounded-capacity body |
| `Body::from_bytes` (alloc) | proxima | our existing API |

Throughput on streaming small bodies (≤256 B); construction cost.

### A7. `pipe-cancellation-const` — atomic-bool cancellation

`tokio_util::sync::CancellationToken` is `Arc`-based; std-bound. A no-alloc
variant uses a single `AtomicBool` shared between producer + consumer (stack
or static storage).

| incumbent | design-favors | workload |
|---|---|---|
| `tokio_util::sync::CancellationToken` | incumbent | tokio's cancel signal |
| `futures::future::AbortHandle` | incumbent | abort + future composition |
| `core::sync::atomic::AtomicBool` (the rolling-our-own substitute) | proxima | minimal no-alloc cancel |

Cancellation-fire latency; race-free cancel-during-drop semantics.

### A8. `pipe-headers-const` — const-generic header list

`HeaderList: Vec<(Bytes, Bytes)>` is alloc-bound. A const variant uses
`heapless::Vec<(BytesRef, BytesRef), N>` where `BytesRef` is either
`&'static [u8]` or an inline `[u8; M]` (depending on the use case).

| incumbent | design-favors | workload |
|---|---|---|
| `heapless::Vec<_, N>` | incumbent | bare-metal vector |
| `arrayvec::ArrayVec<_, N>` | incumbent | stack-allocated vector |
| `HeaderList` (alloc) | proxima | our current API |

Append/lookup throughput; cache behavior under typical 5-15 header workload.

## Composition + Profile

Each primitive lands behind its own sub-flag:

```toml
[features]
# new primitives
prime-inbox-const     = []           # A1 — DCa
prime-executor-const  = []           # A2 — DCb
prime-timer-const     = []           # A3 — DCc
prime-inline-task-const = []         # A4 — DCd
core-error-typed-const = []          # A5 — DCe
pipe-body-const       = []           # A6 — DCf
pipe-cancellation-const = []         # A7 — DCg
pipe-headers-const    = []           # A8 — DCh

# aggregator: "no-alloc-bare-metal"
no-alloc-bare-metal = [
    "prime-inbox-const",
    "prime-executor-const",
    "prime-timer-const",
    "prime-inline-task-const",
    "core-error-typed-const",
    "pipe-body-const",
    "pipe-cancellation-const",
    "pipe-headers-const",
]
```

The Profile TOML maps high-level deployment shape → feature set:

```toml
# profiles/bare-metal.toml
schema = 1
alloc = false
std = false
executor = "prime-const"
reactor = "none"
tls = "none"
quic_enabled = false

# new sections drive const-generic capacity caps via build.rs codegen:
[caps.prime]
inbox_lanes = 4
inbox_lane_cap = 256
executor_slab = 64
timer_slots = 256
timer_levels = 4

[caps.pipe]
headers_inline = 16
body_inline_bytes = 256
```

`proxima-build` extends to emit `pub const INBOX_LANES: usize = N;` consts
from the profile, which the const-generic primitives instantiate against.

## Sequencing

| order | component | incumbent | scope |
|---|---|---|---|
| DCa | prime-inbox-const | heapless::spsc + embassy::channel | SPSC ring on `[Lane; N]` (IN FLIGHT) |
| DCb | prime-executor-const | embassy::executor + tokio current-thread | const-generic slab executor |
| DCc | prime-timer-const | embassy::time + tokio::time | const-generic timer wheel |
| DCd | prime-inline-task-const | embassy static futures + Box::pin | typed inline task with explicit CAP |
| DCe | core-error-typed-const | snafu + anyhow + embedded_hal::Error | stack-backed error chains |
| DCf | pipe-body-const | bytes::Bytes + heapless::Vec | const-generic Body without Box |
| DCg | pipe-cancellation-const | tokio CancellationToken + AbortHandle | AtomicBool-based cancel |
| DCh | pipe-headers-const | heapless::Vec + arrayvec::ArrayVec | const-generic header list |
| DCi | Profile + caps expansion | (no incumbent) | TOML → const-generic codegen |
| DCj | xtask + variant-bench updates | (no incumbent) | profile-driven build matrix |

Each DC follows the disciplined-component 13-point gate. The discipline-log
row for each carries: gate state, opt-sweep table, home-turf bench arms with
`design-favors` labels, named-incumbent results (meet/beat/lose with
specific deltas), honest read, implication.

The umbrella `no-alloc-bare-metal` feature does NOT auto-flip the production
default. It flips on only when the e2e bench (composition of all primitives)
demonstrates the no-alloc-bare-metal shape wins or meets the corresponding
alloc-backed configuration on the workload it was designed for.

## Runtime configurability (deferred)

The user's "either build.rs or runtime" hint:

- **Build.rs path**: PROXIMA_PROFILE + proxima-build today + the expansion
  in DCi. This is the well-supported track.
- **Runtime path**: compile-time-included with runtime-selected dispatch.
  Requires runtime polymorphism (trait objects / enum dispatch) which is
  the OPPOSITE of the no-alloc primitives we're building. Tension between
  the two goals.

Recommendation: keep build.rs as the primary track. Runtime configurability
on top of compile-time-selected primitives is a thin layer that can land
later via a `pub enum BackendChoice { Heap, Inline }` style switch in user
code.

## Honest scope

Building these 8 primitives + the Profile/caps expansion is **substantial
research-grade work**. Each primitive needs:

- Design pass against the named incumbents
- Implementation with const-generic constraints
- ≥6 tests covering happy/sad/edge/drop/concurrency
- Bench harness with home-turf arms
- Numbers recorded in the discipline log
- Opt-sweep pass per gate point 8 (state machine, zero-copy, SIMD, stack-
  over-heap, branchless, no-Box, O(1))

Realistic time per primitive: 1-3 days of focused work + sonnet delegation
for the mechanical pieces (tests, bench scaffolding).

Total for the 8 primitives + caps expansion: ~2-3 weeks at the disciplined-
component pace the telemetry c1-c15 set.

This is the real ambition. DCa is the first concrete step.
