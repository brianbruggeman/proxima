# Build a chaos test rig

**Prerequisites:** [Foundations](./00-foundations.md) — the `Pipe` and its `Err` type.
**You will:** inject seeded, reproducible faults in front of a system under test, and prove two resilience shapes absorb them — **retry** (re-run the same pipe) and **fallback** (route to a different pipe). Fault injection is not a framework; it is a `Pipe` you compose in front.
**New concepts (in order):** `Chaos<Inner>` (seeded fault-injection decorator) · retry (`RetryController`) · fallback (`Fallback`).
**Answer key:** [`examples/chaos/main.rs`](../../examples/chaos/main.rs) — `cargo run --example chaos`.

The example frames it: *"Chaos testing in proxima is not a framework bolted on from outside; it is a `Pipe` you compose IN FRONT of the system under test."*

Every code block below cites real, current lines in `examples/chaos/main.rs` and compiles: `Chaos<Inner>`, `ChaosPolicy`, and `ChaosFault` are types private to that one example binary, not exported by the `proxima` library, so each block repeats the supporting definitions it needs verbatim rather than assuming a shared prelude. Each claim is also verified independently by the real `cargo run --example chaos` transcript quoted in the section that uses it, captured the day this document was rewritten; the same seeds are hard-coded in the example, so re-running it reproduces the exact same numbers.

## 1. `Chaos<Inner>`: fault injection as a decorator

`Chaos<Inner>` wraps any `Pipe`. On every call it rolls a seeded, deterministic PRNG against a `ChaosPolicy` — plain data: a percentage for each fault kind (error, drop, delay) plus how long a `Delay` fault should pretend to wait (`chaos/main.rs:96-102`) — and injects one of three faults, or lets `inner` run clean (`chaos/main.rs:222-246`):

```rust
// real source, quoted verbatim and in full: `Chaos<Inner>`'s own supporting
// types (the PRNG, the fault policy/stats/clock, and its error) so `Self`
// has a real definition to belong to, since none of these are exported by
// the `proxima` library — `Chaos<Inner>` is private to this one example
// binary, by design (fault injection is example-shaped, not library
// machinery).
use std::sync::atomic::AtomicU32;

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut state = self.state;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.state = state;
        state
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultKind {
    Clean,
    Error,
    Dropped,
    Delay,
}

#[derive(Debug, Clone, Copy)]
struct ChaosPolicy {
    error_percent: u64,
    drop_percent: u64,
    delay_percent: u64,
    delay: Duration,
}

impl ChaosPolicy {
    fn classify(&self, roll: u64) -> FaultKind {
        let roll = roll % 100;
        let error_edge = self.error_percent;
        let drop_edge = error_edge + self.drop_percent;
        let delay_edge = drop_edge + self.delay_percent;
        if roll < error_edge {
            FaultKind::Error
        } else if roll < drop_edge {
            FaultKind::Dropped
        } else if roll < delay_edge {
            FaultKind::Delay
        } else {
            FaultKind::Clean
        }
    }
}

#[derive(Default)]
struct ChaosStats {
    errors: AtomicU32,
    drops: AtomicU32,
    delays: AtomicU32,
    clean: AtomicU32,
}

impl ChaosStats {
    fn record(&self, fault: FaultKind) {
        let counter = match fault {
            FaultKind::Error => &self.errors,
            FaultKind::Dropped => &self.drops,
            FaultKind::Delay => &self.delays,
            FaultKind::Clean => &self.clean,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> FaultCounts {
        FaultCounts {
            errors: self.errors.load(Ordering::Relaxed),
            drops: self.drops.load(Ordering::Relaxed),
            delays: self.delays.load(Ordering::Relaxed),
            clean: self.clean.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FaultCounts {
    errors: u32,
    drops: u32,
    delays: u32,
    clean: u32,
}

#[derive(Default)]
struct FaultClock {
    now_nanos: Cell<u64>,
}

impl FaultClock {
    fn advance(&self, elapsed: Duration) {
        let elapsed_nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.now_nanos
            .set(self.now_nanos.get().saturating_add(elapsed_nanos));
    }

    fn now_nanos(&self) -> u64 {
        self.now_nanos.get()
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ChaosFault<InnerErr> {
    Injected,
    Dropped,
    Inner(InnerErr),
}

struct Chaos<Inner> {
    inner: Inner,
    policy: ChaosPolicy,
    rng: RefCell<Xorshift64>,
    clock: FaultClock,
    stats: Arc<ChaosStats>,
}

impl<Inner> Chaos<Inner> {
    fn new(inner: Inner, policy: ChaosPolicy, seed: u64, stats: Arc<ChaosStats>) -> Self {
        Self {
            inner,
            policy,
            rng: RefCell::new(Xorshift64::new(seed)),
            clock: FaultClock::default(),
            stats,
        }
    }

    fn simulated_delay(&self) -> Duration {
        Duration::from_nanos(self.clock.now_nanos())
    }
}

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

Stack a `RetryController` in front of `Chaos(50% fault)`: a failed attempt re-runs the **same** pipe, so every request in the batch still resolves `Ok` (`chaos/main.rs:329-391`):

```rust
// `Request`/`Response`/`Source`/`upstream_service` — the pipe under test —
// real, verbatim (`chaos/main.rs:266-305`); repeated here since `Request`/
// `Response` collide by name only with the unrelated HTTP types
// `tutorial_gate_prelude` already exports (guarded in the awk transform so
// this redeclaration is never forwarded past this block). `Chaos`/
// `ChaosPolicy`/`ChaosStats` carry forward from §1 above.
#[derive(Debug, Clone, Copy)]
struct Request {
    id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Upstream,
    Cache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Response {
    id: u32,
    source: Source,
}

impl Retryable for Response {
    fn retry_status(&self) -> Option<u16> {
        None
    }

    fn is_success(&self) -> bool {
        true
    }
}

#[proxima::piped]
async fn upstream_service(request: Request) -> Result<Response, Infallible> {
    Ok(Response {
        id: request.id,
        source: Source::Upstream,
    })
}

// `policy_50pct`/`seed`/`stats` given their real values
// (`chaos/main.rs:333-345`) instead of the placeholder names the real
// driver function builds them under.
let policy_50pct = ChaosPolicy {
    error_percent: 35,
    drop_percent: 15,
    delay_percent: 10,
    delay: Duration::from_millis(75),
};
let seed = 0xA5A5_1234_9E37_79B9;
let stats = Arc::new(ChaosStats::default());
let chaos = Chaos::new(upstream_service, policy_50pct, seed, stats);
let controller = RetryController {
    rules: RetryRules::default(),
    backoff: Backoff::Exponential {
        initial: Duration::from_millis(20),
        factor: 2,
        max: Duration::from_millis(500),
    },
    jitter: Jitter::None,
    max_attempts: 4,
    deadline: None,
};
// per attempt: controller.on_outcome(...) -> Retry { after } | Done | Exhausted
```

`stats` above is the shared `ChaosStats` counter from `Chaos::new` — the same one the wrap-up print reads to report what was actually injected. `Backoff::Exponential` grows the wait between retries with each attempt, so a flaky call backs off instead of hammering the system immediately (`chaos/main.rs:348-352`).

`RetryController::on_outcome` decides Retry/Done/Exhausted from the outcome + rules; the loop re-calls the pipe on `Retry` (`chaos/main.rs:309-327`). Real, captured output from `cargo run --example chaos` (deterministic — the seed is hard-coded, so a re-run reproduces these exact numbers):

```text
-- chaos(50% fault) + retry(4): every request still resolves --
  request 5: resolved Ok(Response { id: 5, source: Upstream }) after 3 attempt(s)
  request 6: resolved Ok(Response { id: 6, source: Upstream }) after 2 attempt(s)
  request 9: resolved Ok(Response { id: 9, source: Upstream }) after 2 attempt(s)
  ...
  faults injected: 2 error, 2 drop, 2 delay, 14 clean (20 attempts over 16 requests)
  simulated chaos-clock advance: 150ms (no real sleep)
  16/16 requests recovered — graceful degradation via retry
```

16/16 requests recover despite a 50% per-attempt fault rate (35% error + 15% drop, `chaos/main.rs:333-338`): every request that drew a fault on attempt one simply drew again, and the controller's 4-attempt cap was never exhausted.

## 3. Fallback absorbs faults by routing to a different pipe

Where retry re-runs the *same* pipe, `Fallback` routes to a **different** one on any failure. `Chaos(80% fault)` as the primary, a reliable `Cache` as the secondary — every request resolves `Ok` regardless of how hostile the policy is (`chaos/main.rs:438-443`):

```rust
// `Request`/`Response`/`Source`/`upstream_service` repeated (self-contained
// per block, per the awk transform's `Request`/`Response` shadow guard —
// see §2 above); `Cache`, the reliable secondary, real and verbatim
// (`chaos/main.rs:397-418`). `Chaos`/`ChaosPolicy`/`ChaosStats`/`AtomicU32`
// (the `use` above §1's own `ChaosStats`) carry forward from §1.
#[derive(Debug, Clone, Copy)]
struct Request {
    id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Upstream,
    Cache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Response {
    id: u32,
    source: Source,
}

#[proxima::piped]
async fn upstream_service(request: Request) -> Result<Response, Infallible> {
    Ok(Response {
        id: request.id,
        source: Source::Upstream,
    })
}

struct Cache {
    hits: Arc<AtomicU32>,
}

impl Pipe for Cache {
    type In = Request;
    type Out = Response;
    type Err = ChaosFault<Infallible>;

    fn call(
        &self,
        request: Request,
    ) -> impl Future<Output = Result<Response, ChaosFault<Infallible>>> {
        self.hits.fetch_add(1, Ordering::Relaxed);
        async move {
            Ok(Response {
                id: request.id,
                source: Source::Cache,
            })
        }
    }
}

// `chaos_80pct`/`request` given real values (`chaos/main.rs:424-436,446`)
// instead of the placeholder names the real driver loop builds them under.
let policy_80pct = ChaosPolicy {
    error_percent: 30,
    drop_percent: 30,
    delay_percent: 20,
    delay: Duration::from_millis(120),
};
let stats = Arc::new(ChaosStats::default());
let chaos_80pct = Chaos::new(upstream_service, policy_80pct, 0xC0FF_EE00_1357_2468, stats);
let cache_hits = Arc::new(AtomicU32::new(0));
let request = Request { id: 1 };
let composite = Fallback {
    primary: chaos_80pct,
    secondary: Cache {
        hits: Arc::clone(&cache_hits),
    },
};
let response = composite.call(request).await.unwrap();  // always resolves
```

`Fallback`'s guarantee does not depend on tuning luck: any primary failure → the secondary answers. Real, captured output from the same run (30% error + 30% drop + 20% delay = 80% of rolls hit a fault bucket, `chaos/main.rs:424-429`):

```text
-- chaos(80% fault) + fallback: every request still resolves --
  request 2: resolved Ok(Response { id: 2, source: Cache }) via Cache
  request 5: resolved Ok(Response { id: 5, source: Cache }) via Cache
  ...
  faults injected: 4 error, 4 drop, 1 delay, 7 clean over 16 requests
  cache served 8 of 16 requests (primary's faults routed here)
  16/16 requests recovered — graceful degradation via fallback
```

## What you built

- **`Chaos<Inner>`** — seeded, deterministic fault injection (error / drop / delay) as a `Pipe` in front of the system under test; no real randomness, no real sleeps.
- **retry** — `RetryController` re-runs the same pipe on a retryable outcome.
- **fallback** — `Fallback` routes to a different pipe on any failure.

Chaos is injected at the seam, not baked into the service — and the two resilience shapes that absorb it are ordinary `Pipe`s wrapped around it. Same seed, same faults, provable assertions. (Both retry and fallback have standalone examples: [`examples/retry`](../../examples/retry), [`examples/fallback`](../../examples/fallback).)
