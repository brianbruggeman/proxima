# Exploration prompt — prime multi-core / multi-app saturation

Paste the section below into a fresh agent/session. It is self-contained.
Everything above this line is just a pointer.

---

## Mission

Empirically characterize how prime's runtime behaves under multi-core and
multi-app load on one machine, and turn that understanding into a design:
**how many co-located prime `App`s can run before performance breaks, and
what the right core-budget / compat model is.** Measure first; theorize
second. Produce a saturation curve, not a guess.

This is a follow-on to the prime-default serve flip (landed on `main` at
commit `4cbd273`, repo `github.com:brianbruggeman/proxima`, in the workspace
checkout at `proxima`). The flip made
`PrimeRuntime` the default serve+chain runtime. See
`docs/prime-serve/discipline.md` for the full log.

## Verified facts (don't re-derive — but DO re-verify with file:line if you build on them)

**Prime worker idle behavior — workers PARK, they don't spin forever.**
`prime/src/os/core_shard.rs`:
- A recently-busy worker spins ~`SPIN_BEFORE_PARK_BUSY = 256` iterations
  (~100–300 ns) before parking, to dodge the ~1 µs park syscall on bursty
  load.
- After `IDLE_PARK_THRESHOLD = 4` consecutive empty parks it sets
  `SPIN_BEFORE_PARK_IDLE = 0` — a truly idle worker parks immediately,
  blocking on `kevent`/`epoll_wait` (`prime/src/os/reactor.rs` `turn`),
  costing ≈0 CPU.
- Implication: **idle apps are cheap (parked threads burn no CPU). The
  limit is *simultaneously-busy* apps, not idle ones.**

**tokio-compat DOUBLES the thread count.** `prime/src/os/tokio_compat.rs`:
each prime core gets a *sister* `tokio::runtime::Builder::new_current_thread`
runtime on its **own dedicated thread** (`tokio-compat-<core>-driver`,
`block_on(future::pending())`). The prime worker `.enter()`s that sister's
`Handle` for life so `tokio::{spawn,time,sync}` / hyper from a prime task
resolve against it. So **with compat: `num_cores` prime workers +
`num_cores` sister tokio driver threads + a small `BackgroundPool` ≈
2×cores + a few threads per App.** Without compat: ≈ cores + a few.

**The default App uses compat.** `src/app.rs` `default_runtime()` builds
`PrimeRuntime::new_with_tokio_compat(runtime_cores())` (prime arm) so user
pipes that call tokio primitives keep working. `runtime_cores()` reads
`PROXIMA_RUNTIME_CORES` (env), default `num_cpus::get()`, min 1. The
non-compat constructor is `PrimeRuntime::new(cores)`; `serve_parity`'s
reactor-absence proof uses the non-compat one to show the prime *transport*
needs no tokio reactor.

**nextest is process-per-test.** A per-process shared runtime helps
many-Apps-in-ONE-process; it does NOT bound oversubscription across the
concurrent test *processes* nextest spawns. Keep this distinction — it
bit the last session.

## The user's design decisions (already made — implement toward these)

1. **Shared global core budget.** Multiple `App`s in one process should
   draw from a *process-wide* core budget rather than each grabbing
   `num_cpus`. The natural shape: a process-global lazily-built default
   runtime (e.g. `OnceLock`) that every `App::new()` shares unless it calls
   `.with_runtime(...)`. This makes K Apps-in-a-process = 1 runtime, not
   K×cores. (Cross-*process* co-location is handled by per-process
   `PROXIMA_RUNTIME_CORES`.)
2. **Compat = Auto by default, with On/Off optional.** A `CompatMode {
   Auto, On, Off }`. `Auto` enables the sister-tokio compat only when the
   loaded pipe graph references tokio-backed pipes (e.g. the hyper
   `HttpUpstream` under `http-hyper`). Caveat you MUST handle: Auto cannot
   detect a *user* closure that calls `tokio::time` internally (opaque), so
   Auto is best-effort — a pure-prime default app should pay no compat tax,
   but a user with tokio-using custom pipes sets `On`. For the *shared*
   runtime this is in tension (it's built once, before all apps' pipes are
   known) — resolve it (conservative-on for the shared one? per-app
   dedicated runtimes when Auto detects tokio? your call, justify it).
3. **Build the breaking-point experiment** (below).

## Questions to answer with NUMBERS

- For a given per-app core budget (sweep 1, 2, 4, num_cpus), how many
  **simultaneously-busy** prime serve `App`s can run before p50/p99 latency
  or error-rate falls off a cliff? Produce a curve: x = #busy apps, y =
  p99 (and error-rate), one line per core budget.
- Where exactly does "break" happen — latency degradation (graceful) or
  hard failures (connection resets / accept-backlog overflow / handshake
  timeouts)? Capture the failure MODE, not just the threshold.
- What's the compat thread tax in practice — measure thread count + RSS for
  N idle apps vs N busy apps, compat on vs off.
- Does the shared global budget actually fix many-apps-in-one-process?
  Measure K Apps sharing one runtime vs K Apps each with their own.

## Suggested harness (adapt freely)

A `benches/` or a standalone `tests/`-style binary that, in ONE process:
- spins K prime serve `App`s (a trivial echo/fixed-body pipe), each on a
  chosen core budget / shared-vs-dedicated runtime;
- drives closed-loop or open-loop load at all K simultaneously (reuse the
  client shape in `tests/serve_parity.rs` / `benches/bench_serve_prime_vs_tokio.rs`);
- ramps K and records p50/p99/error-rate + process thread count + RSS;
- emits a table/curve.
Reuse `tests/serve_parity.rs`'s `spawn_prime_serve` (prime serve on a
CoreShard worker via `spawn_factory_on_core` + `Box::leak` for the
`'static` serve future) and its raw client as the building blocks.

## Constraints / discipline

- Honor `/guiding-principles` and `/disciplined-component`. P14 (parity),
  P16 (proof keeps up — numbers, not vibes; saved baselines).
- Edition 2024; clippy pedantic deny-warnings; no `unwrap`/`expect` outside
  tests; `cargo nextest`; benches via criterion.
- workspace git convention: conventional commits, NO `Co-Authored-By`,
  `--no-gpg-sign` only if 1Password blocks. Branch off `main`; don't push
  without explicit authorization.
- `proxima-prime-http` is a WORKTREE pattern — if you create one, remember
  `main` may be checked out elsewhere (`git worktree list`), so update
  `main` in its own worktree, not via a failed `git checkout main`.

## What NOT to do (lessons from the flip session)

- **Don't conflate idle and busy.** Parked workers are ≈free; the cost is
  simultaneously-runnable threads ≫ physical cores. Frame everything in
  those terms.
- **Don't mask with `cores=1`.** That hides behavior; the goal is to
  understand multi-core, so test 2/4/num_cpus and measure.
- **Capture the actual failure before theorizing.** Last session a serve
  test flake was misdiagnosed as oversubscription for a long time; it was
  actually a *test* bug (a client discarding a pipelined response). Get the
  real panic/error/latency number first, every time.
- **Measure, don't reason from the code alone.** The code tells you the
  mechanism (park/spin, sister threads); only the harness tells you the
  threshold.

## Deliverables

1. The saturation harness + a results table/curve (#busy apps vs p99/error,
   per core budget; compat on/off; shared vs dedicated runtime).
2. A short findings doc (the real "how many apps" answer + the failure
   mode) in `docs/prime-serve/` or the workspace journal.
3. Implement the shared global core budget + `CompatMode{Auto,On,Off}` per
   the decisions above, gated/configurable, with tests.
4. Update `docs/prime-serve/discipline.md` (the "MULTI-APP SCALING" entry)
   with the measured numbers and the landed design.
