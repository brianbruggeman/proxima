# prime-tokio-compat — what you actually get

`PrimeRuntime::builder().tokio_compat()` lets user code keep its
`tokio::*` imports while running on prime. This doc was originally
titled "what you give up" — bench data on 2026-05-19 changed the
framing.

> For the full engineering history, bench logs, and root-cause
> investigation, see
> [discipline-prime-tokio-compat.md](discipline-prime-tokio-compat.md).

**Audit retraction (2026-05-19).** An earlier draft of this doc
claimed prime+compat was "structurally faster than default tokio
multi-thread" for h2 servers, with 44%/41% wins for hyper/pingora.
A code audit by the user caught that the bench rigged the comparison:
the multi-thread baseline ran a 4-worker tokio runtime serving a
**single TCP connection** (single h2 muxer), so 3 workers were idle
paying scheduler tax. Sister tokio is single-threaded
(`new_current_thread()`), so it doesn't pay that tax. The "win"
was current-thread-beats-multi-thread on single-connection workloads,
NOT a compat-mode win.

Re-bench with an added `*_current_thread` control arm shows
compat ≈ current_thread (within 0–6% across libraries and workloads):

| library | h2 fan-in: compat / current_thread | h2 fan-in: compat / multi_thread (was claimed) |
| ------- | ---------------------------------- | ---------------------------------------------- |
| hyper   | 1.00× (parity)                     | 2.03× (rigged)                                 |
| pingora | 1.03× (within noise)               | 1.97× (rigged)                                 |
| proxima | 1.06× (small real signal)          | 1.11× (mostly rigged)                          |

**Honest framing.** Compat lets your tokio-using library run on prime
at parity with `tokio::Builder::new_current_thread()` on **single-stream**
workloads (0-6% per Bench B). On **multi-conn h2**, compat costs 15-25%
vs `TokioPerCoreRuntime` due to thread-budget doubling (Cost D below).
The mechanism — sister tokio per core + EnterGuard on prime worker —
is correctness-correct, but the 2N thread architecture is not free
under multi-connection load. See
[discipline-prime-tokio-compat.md](discipline-prime-tokio-compat.md)
Bench B section for the full numbers and engineering history.

That said, compat is NOT a free abstraction. The costs below are
empirically validated where measured; predicted-but-not-measured costs
are flagged. Read this before flipping the feature on.

> The recommended default is still native prime + `proxima::sync::*` /
> `proxima::time::*` (P1.3 plumbing landing separately). Compat is the
> bridge for consumers who cannot or will not migrate tokio imports
> today — and on single-stream workloads it's a remarkably good bridge.
> For multi-conn h2 where tokio imports are required, prefer
> `TokioPerCoreRuntime` over compat (N threads vs 2N, no dual-reactor
> overhead).

## How it's wired

Per prime worker core, compat mode spawns:

1. One **sister tokio current-thread runtime** (`tokio::runtime::Builder
   ::new_current_thread().enable_all()`), running on its own OS thread
   (`tokio-compat-<core>-driver`). Best-effort pinned to the same
   physical core as its prime worker.
2. A static `tokio::runtime::EnterGuard` parked on the prime worker
   thread (one `tokio::runtime::Handle` leaked per worker — bounded
   by `num_cores`).

Inside any prime task:

- `tokio::spawn(future)` dispatches `future` to the **sister** task
  queue. The future runs on the sister OS thread, not the prime
  worker's thread.
- `tokio::sync::*` lock / wait / wake mechanics work without runtime
  coupling — these primitives are runtime-agnostic in tokio 1.x — but
  any cross-task wake hopping through `tokio::sync::Notify` /
  `tokio::sync::watch` enters the sister scheduler if the waker fires
  from sister-resident code.
- `tokio::time::{sleep, timeout, interval}` register on the **sister's
  timer wheel**, not prime's. Wake fires from the sister thread back
  to the awaiting prime task — cross-thread waker hop.
- `tokio::net::{TcpStream, TcpListener, UdpSocket}` register on the
  **sister's mio reactor**. Read / write completion fires from the
  sister thread.

## What this costs

### Cost A — dual reactors per core (predicted, NOT measured to matter)

Both reactors run when compat is on:

| Reactor       | Owns                                           |
| ------------- | ---------------------------------------------- |
| prime's epoll/kqueue | proxima's accept loops, prime's native `TcpListener` / `TcpStream`, prime worker wakeup |
| tokio's mio   | `tokio::net::*`, `tokio::io::unix::AsyncFd`, anything driven by `tokio::time::*` |

Each reactor performs its own syscall budget. On idle, both park. On
active I/O, both wake. Linux `epoll_wait` / macOS `kqueue` cost adds
up under heavy fan-in **in theory**.

**Empirical status (2026-05-19, with audit retraction):** Bench A W3
(streaming response) and W2 (h2 fan-in) showed compat ≈ prime in
ratio terms, but those workloads were single-connection — they do
not stress reactor-level I/O parallelism. The dual-reactor cost
remains **unmeasured under a workload that would surface it**
(multi-connection, ≥CORES inbound + outbound sockets, SO_REUSEPORT).
Earlier framing of this cost as "empirically dominated" was wrong;
the right statement is "empirically untested at the workloads that
would surface it." A proper multi-conn rebench is queued.

> **Why two reactors?** tokio 1.x (1.52 at pin) exposes **no public hook**
> to swap in a custom I/O driver. `tokio::runtime::Builder` has
> `enable_io()`, `max_io_events_per_tick()`, `enable_io_uring()`, none
> of which accept a foreign `Driver`. The `Park`/`Driver`/`IoHandle`
> types are `pub(crate)`; `mio::Poll::new()` is unconditional inside
> `tokio::runtime::io::Driver`. Plugging prime's reactor *as* tokio's
> driver requires forking tokio. See
> [discipline-prime-tokio-compat.md](discipline-prime-tokio-compat.md)
> §C3 for the full investigation.

### Cost B — cross-thread waker hops dominate ONLY on external-thread spawn-heavy producers (measured: W1 is 5.9× slower)

`tokio::spawn`, `tokio::time::sleep`, `tokio::net::*` all run on the
sister thread. When user code dispatches at high rate, every dispatch
crosses a thread boundary to the sister.

**Measured cost (Bench A W1, 2026-05-19):** an external thread
dispatching 4000 spawns/iter via per-core sister `Handle::spawn` runs
at **0.43 Melem/s** vs prime's native `runtime.spawn_on_core` at
**2.55 Melem/s** — compat is **5.9× slower** on this workload. This
is the cost of compat for code shaped like "external CLI/sidecar
process dispatches many small jobs into the runtime."

This cost **does not appear** in workloads where:

- spawn rate is amortized over per-task work (h2 handlers, request
  pipelines) — see Bench A W2/W3
- producers run inside the runtime, not as external threads — the
  spawn path then takes prime's native fast path, not the sister
  handle path

If your application is shaped like "external producer thread pushes
many tasks/sec to compat," budget 5-6× the spawn latency vs native
prime. For the more common "tasks running in the runtime spawn
sub-tasks," cost is near-zero.

### Cost C — locality between prime tasks and tokio tasks is broken

Native prime guarantees that a task spawned on core K runs on core K
for its entire life — no implicit migration, predictable cache
locality. `tokio::spawn` from a prime task on core K dispatches the
future to the **sister's** runtime, which runs on its own thread
core-pinned to core K. The task body executes on a different OS
thread; any per-task working set is now on a different stack and
different L1 / L2 footprint than the parent.

For a request handler that spawns sub-work via `tokio::spawn`, this
means the sub-task's TLB / cache state has no relationship to the
parent's. Native prime users `proxima::runtime::prime::os::core_shard::
spawn_on_current_core` to preserve locality; compat does not surface
that knob through `tokio::spawn`.

### Cost D — per-core thread budget doubling (the multi-conn perf story)

Prime+compat uses **2N OS threads on N cores**: one prime worker thread
per core, plus one sister tokio driver thread per core. `TokioPerCoreRuntime`
uses N threads. That 2× thread budget is the root cause of the 15-25%
multi-conn h2 gap.

**Cross-thread hop chain per inbound connection accept:**

```
sister mio reactor fires (inbound socket ready)
  → sister task queue push (cross-thread hop 1: sister → prime worker)
  → prime worker receives waker, parks task
  → prime accept handler completes
  → response path crosses back to sister for tokio::net::TcpStream write
  → cross-thread hop 2: prime worker → sister
```

Two hops per accept-to-response cycle, plus dual reactors both polling
under load. On single-stream workloads the hops amortize to noise;
on multi-conn workloads (≥N inbound connections distributed across all
cores, each with its own muxer), the hop overhead compounds and the
15-25% gap surfaces.

`TokioPerCoreRuntime` IS the tokio-import-preserving path — it is N
pinned current-thread tokio runtimes, one per core, preserving
`tokio::*` imports with N threads (not 2N). When the workload is
multi-conn h2 and you need tokio imports, `TokioPerCoreRuntime` is the
right choice, not compat.

**RSS and OS thread count** (compat vs pure prime):

`tokio::runtime::Builder::new_current_thread().enable_all().build()`
creates a runtime with its scheduler + mio reactor + timer driver
state — ~1 MB RSS per runtime, plus the driver thread's stack (~2 MB
default on macOS, configurable). Multiplied by `num_cores`, the
overhead is real on small-RAM deployments.

| `num_cores` | extra RSS vs pure prime | extra OS threads |
| ----------- | ----------------------- | ---------------- |
| 1           | ~3 MB                   | 1                |
| 4           | ~12 MB                  | 4                |
| 8           | ~24 MB                  | 8                |
| 16          | ~48 MB                  | 16               |

Negligible on bare-metal hosts, non-negligible on cgroup-tightened
containers.

### Cost E — tokio's known-cost primitives are not free in compat

`tokio::sync::Mutex` uses an internal `tokio::sync::Semaphore` —
contention costs the same in compat as in native tokio. Same for
`tokio::sync::Notify`, `watch`, `broadcast`. Compat doesn't make
these worse, but it doesn't make them better either. If
`proxima::sync::*` (P1.3) is on the table for the consumer, the
runtime-agnostic primitives from `event-listener` / `async-broadcast`
/ `async-lock` tend to win on prime-native paths.

### Cost F — heavy `tokio::sync::Mutex` contention is ~3× slower on compat

`tokio::sync::Mutex` shared across 4 prime cores at ~4000 lock
cycles each: compat measured **99.7 K elem/s** vs native tokio
multi-thread at **329 K elem/s** — compat is 30% of tokio. The cost
is real: prime workers waking each other via `tokio::sync::Mutex`'s
internal semaphore go through prime's cross-core wake path on every
unlock, vs tokio's intra-runtime wake which is a direct local-thread
push.

**Earlier reports** of this workload **hanging** were a prime
cross-core wake bug — the race-close pattern around `arm_wakeup`
in `worker_main` didn't re-check the inbox after arming. Fixed
2026-05-19 (see `discipline-prime-tokio-compat.md` §C4). Post-fix
the workload completes cleanly with the slow-but-honest numbers
above.

Guidance: for workloads dominated by heavy contended async mutex,
prefer either native tokio (if you can't migrate) or `futures::lock::Mutex`
/ `async-lock::Mutex` (which prime handles much faster — 8.98 M
elem/s on the same workload). The prime arm of the same bench using
`futures::lock::Mutex` is **~27× faster than tokio_per_core**, so
the cost is specifically in compat's tokio-API bridging, not in
prime itself.

## What it gets you (measured, 2026-05-19)

- **Drop-in compat for tokio-API user code.** `use tokio::sync::Mutex;`,
  `tokio::spawn`, `tokio::time::sleep` — all keep working without a
  source rewrite. Cargo flag flip from `runtime-tokio` to
  `runtime-prime-full,prime-tokio-compat` ships the migration in a
  single PR.
- **Performance at parity with `tokio::Builder::new_current_thread()`
  on single-stream h2 workloads** (Bench B 2026-05-19, 0-6%);
  **15-25% slower than `TokioPerCoreRuntime` on multi-conn h2** (Cost D).
  Numbers for single-stream:

  | library | compat / current_thread (single-stream) | compat / current_thread (h2 fan-in) |
  | ------- | --------------------------------------- | ------------------------------------ |
  | hyper   | 1.00× (parity)                          | 1.00× (parity)                      |
  | pingora | 1.03×                                   | 1.03×                               |
  | proxima | 1.21× (real, modest)                    | 1.06× (real, modest)                |

  An earlier draft claimed compat was 44%/41% faster than default
  multi-thread tokio. That was a rigged comparison (multi-thread
  baseline serving single-connection — 3 of 4 workers idle paying
  scheduler tax). Fixed.
- **Best-effort core pinning across the sister.** When prime's worker
  is pinned to physical core K, the sister is pinned to K too. NUMA
  locality is preserved at the OS level.

## When to use compat (decision matrix)

| Consumer shape | Recommendation |
|---|---|
| H2 server, can use TokioPerCoreRuntime | **Use per_core. Not compat.** Compat costs 15-25% on this workload. |
| H2 server, must keep tokio imports AND TokioPerCoreRuntime is acceptable | **Use per_core.** It IS the tokio-import-preserving path. |
| Tokio library that can't be rewritten AND prime's scheduling semantics are wanted for OTHER (non-h2) parts of the workload | **Use compat.** The 15-25% perf cost on the h2 portion is the price; prime native paths still work for everything else. |
| H2 server that wants prime's per-core sharded scheduling without tokio imports | Use native prime (no compat). |
| Tokio library where only this thin layer needs to keep tokio imports, no other prime affordances are wanted | Use TokioPerCoreRuntime. |
| Can migrate `use tokio::sync::*` → `use proxima::sync::*` (P1.3) | native prime (`PrimeRuntime::builder().build()`) |
| Cannot migrate, dispatches many tasks/sec from an external (non-runtime) thread | tokio per-core (`TokioPerCoreRuntime`) — Bench A W1 shows compat is 5.9× slower on this pattern |
| Heavy contended async mutex workload | NOT compat — W4 is a 0.30× perf-fail. Use native tokio or migrate to `futures::lock::Mutex`. |

**Honest framing for multi-conn h2:** compat costs 15-25% vs
`TokioPerCoreRuntime` on multi-connection h2 workloads. The cause is
the per-core thread budget doubling (see Cost D below). On single-stream
workloads, compat ties or beats `tokio::Builder::new_current_thread()`
by 0-6% (Bench B, `0cc99c9`). These two regimes have opposite
recommendations — choose based on your actual workload shape.

## Future direction

A clean fix for Cost A requires either (a) a tokio fork that exposes
a `ParkDriver` injection seam, or (b) waiting for tokio upstream to
land such a hook. As of the investigation in §C3, no upstream PR is
on the path. Until that lands, dual reactors per core is the price
of API parity.

Cost B (cross-thread waker hops) is fundamentally driven by the
"sister thread per core" architecture. An alternative architecture
would drive the tokio current-thread runtime *cooperatively* from
inside the prime executor's poll loop — challenging because the
tokio runtime expects to own its thread. Out of P2 scope.

Cost C (locality) is intrinsic to having two runtimes on one core.
Mitigations are policy-level (don't `tokio::spawn` from prime tasks
unless you mean to leave the prime side; use `prime::os::core_shard::
spawn_on_current_core` for locality-sensitive sub-work).
