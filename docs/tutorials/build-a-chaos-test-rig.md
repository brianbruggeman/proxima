# Build a chaos test rig

**Prerequisites:** [Foundations](./00-foundations.md) — the `Pipe` and its `Err` type.
**You will:** inject seeded, reproducible faults in front of a system under test, and prove two resilience shapes absorb them — **retry** (re-run the same pipe) and **fallback** (route to a different pipe). Fault injection is not a framework; it is a `Pipe` you compose in front.
**New concepts (in order):** `Chaos<Inner>` (seeded fault-injection decorator) · retry (`RetryController`) · fallback (`Fallback`).
**Answer key:** [`examples/chaos/main.rs`](../../examples/chaos/main.rs) — `cargo run --example chaos`.

The example frames it: *"Chaos testing in proxima is not a framework bolted on from outside; it is a `Pipe` you compose IN FRONT of the system under test."*

## 1. `Chaos<Inner>`: fault injection as a decorator

`Chaos<Inner>` wraps any `Pipe`. On every call it rolls a seeded, deterministic PRNG against a `ChaosPolicy` — plain data: a percentage for each fault kind (error, drop, delay) plus how long a `Delay` fault should pretend to wait (`chaos/main.rs:96-102`) — and injects one of three faults, or lets `inner` run clean (`chaos/main.rs:193-246`):

```rust
impl<Inner: Pipe> Pipe for Chaos<Inner> {
    type In = Inner::In;
    type Out = Inner::Out;
    type Err = ChaosFault<Inner::Err>;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        let fault = self.policy.classify(self.rng.borrow_mut().next_u64());
        self.stats.record(fault);
        if fault == FaultKind::Delay { self.clock.advance(self.policy.delay); }
        async move {
            match fault {
                FaultKind::Error   => Err(ChaosFault::Injected),   // inner never runs
                FaultKind::Dropped => Err(ChaosFault::Dropped),    // blackholed
                FaultKind::Delay | FaultKind::Clean =>
                    self.inner.call(input).await.map_err(ChaosFault::Inner),
            }
        }
    }
}
```

`self.rng.borrow_mut()` in the snippet above is Rust's `RefCell` — it is how a method that only borrows `&self` is still allowed to mutate something inside itself (here, the PRNG's state); not important to follow further here.

Two things make the assertions provable, not eyeballed: the PRNG is **seeded** (same seed → same fault sequence every run — `chaos/main.rs:55-81`), and `Delay` advances a **fake clock**, never a real sleep (`chaos/main.rs:162-180`). `Chaos`'s `Err` is distinct (`ChaosFault`) so "chaos struck" is never confused with "the system failed on its own" (`chaos/main.rs:182-191`).

## 2. Retry absorbs faults by re-running the same pipe

Stack a `RetryController` in front of `Chaos(50% fault)`: a failed attempt re-runs the **same** pipe, so every request in the batch still resolves `Ok` (`chaos/main.rs:338-399`):

```rust
let chaos = Chaos::new(UpstreamService, policy_50pct, seed, stats);
let controller = RetryController { max_attempts: 4, backoff: Backoff::Exponential { .. }, .. };
// per attempt: controller.on_outcome(...) -> Retry { after } | Done | Exhausted
```

`stats` above is the shared `ChaosStats` counter from `Chaos::new` — the same one the wrap-up print reads to report what was actually injected. `Backoff::Exponential` grows the wait between retries with each attempt, so a flaky call backs off instead of hammering the system immediately (`chaos/main.rs:357-361`). The bare `{ .. }` is this tutorial's shorthand for "other fields omitted for brevity" — not runnable on its own; see `chaos/main.rs:355-365` for the real, complete values.

`RetryController::on_outcome` decides Retry/Done/Exhausted from the outcome + rules; the loop re-calls the pipe on `Retry` (`chaos/main.rs:318-336`). 16/16 requests recover despite a 50% per-attempt fault rate.

## 3. Fallback absorbs faults by routing to a different pipe

Where retry re-runs the *same* pipe, `Fallback` routes to a **different** one on any failure. `Chaos(80% fault)` as the primary, a reliable `Cache` as the secondary — every request resolves `Ok` regardless of how hostile the policy is (`chaos/main.rs:429-486`):

```rust
let composite = Fallback { primary: chaos_80pct, secondary: Cache { .. } };
let response = composite.call(request).await.unwrap();  // always resolves
```

`Fallback`'s guarantee does not depend on tuning luck: any primary failure → the secondary answers.

## What you built

- **`Chaos<Inner>`** — seeded, deterministic fault injection (error / drop / delay) as a `Pipe` in front of the system under test; no real randomness, no real sleeps.
- **retry** — `RetryController` re-runs the same pipe on a retryable outcome.
- **fallback** — `Fallback` routes to a different pipe on any failure.

Chaos is injected at the seam, not baked into the service — and the two resilience shapes that absorb it are ordinary `Pipe`s wrapped around it. Same seed, same faults, provable assertions. (Both retry and fallback have standalone examples: [`examples/retry`](../../examples/retry), [`examples/fallback`](../../examples/fallback).)
