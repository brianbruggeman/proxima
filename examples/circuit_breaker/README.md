# circuit-breaker — a gate that opens itself on failures

A breaker over a failing dependency: closed calls pass through, enough consecutive failures trips it open (short-circuit, no dependency call), a cooldown lets it probe again, and enough successful probes closes it.

## Builds on

[gate](../gate/README.md) — a breaker is a gate that opens on a failure threshold. `AtomicGate` is armed/disarmed by an external controller; `CircuitBreaker` is the same shed-vs-pass decision, but the pipe arms and disarms itself from the outcomes it observes.

## What it demonstrates

`CircuitBreaker` (`proxima_primitives::pipe::resilience::circuit_breaker`) is sans-IO: three methods, no wall-clock read inside any of them.

- `allow(now_nanos) -> bool` — may this call proceed right now?
- `on_success()` — record a success.
- `on_failure(now_nanos)` — record a failure.

`CircuitState` is the observable three-state machine:

1. **Closed** — calls pass through; consecutive failures are counted, reset by any success.
2. **Open** — tripped after `failure_threshold` consecutive failures; every call is refused before it reaches the dependency, until `cooldown` elapses.
3. **HalfOpen** — cooldown elapsed; a bounded number of probe calls (`half_open_max_probes`) are let through to test recovery. Enough successes closes the circuit; any failure during a probe re-opens it immediately.

The example wires a `Breaker<Inner>` `Pipe` around a `FlakyDependency`: `allow` is checked synchronously at `call` time, before the inner pipe's future is even constructed, so an open circuit never touches the dependency — proved with an `AtomicUsize` call counter on the dependency that stops moving the instant the circuit opens. Because the cooldown is `now_nanos`-driven rather than a real timer, the example advances a `ManualClock` by hand instead of sleeping — the same injected-clock idiom as [clock](../clock/README.md) and [backoff](../backoff/README.md).

## Run

```
cargo run --example circuit_breaker
```

## What you'll see

```
circuit breaker: closed -> open (short-circuit) -> half-open (probe) -> closed
-- closed: dependency failing, calls pass through until the threshold --
  call 1: Err(Inner("dependency unavailable")) (state=Closed)
  call 2: Err(Inner("dependency unavailable")) (state=Closed)
  call 3: Err(Inner("dependency unavailable")) (state=Open)
-- open: cooldown not elapsed, calls are refused before the dependency --
  call 4: Err(Open) (state=Open)
  call 5: Err(Open) (state=Open)
-- cooldown elapses: next call probes in half-open --
  call 6 (probe 1): Ok("ok") (state=HalfOpen)
  call 7 (probe 2): Ok("ok") (state=Closed)
-- closed again: dependency reached normally --
  call 8: Ok("ok") (state=Closed)

closed -> open -> half-open -> closed, proved by state and by a call count the open circuit never moved.
```

- **Closed → Open**: calls 1-2 fail but stay under the 3-failure threshold, state stays `Closed`. Call 3 is the third consecutive failure — `on_failure` trips the breaker to `Open`. All three calls reach the dependency (`calls == 3`): `Closed` never short-circuits.
- **Open short-circuits**: calls 4-5 land inside the 1s cooldown. `allow` returns `false` before the dependency's future is built, so the error is `Err(Open)` — not the dependency's own `"dependency unavailable"` error — and the dependency's call counter stays at 3. This is the proof: an open circuit does not call the inner pipe, it refuses before it.
- **Open → HalfOpen → Closed**: advancing the `ManualClock` by exactly the 1s cooldown and re-marking the dependency healthy, call 6's `allow` crosses the cooldown deadline and transitions to `HalfOpen`, admitting one probe. That probe succeeds, but this breaker requires 2 successful probes to close, so the state after call 6 is still `HalfOpen`. Call 7 is the second successful probe — `on_success` reaches the probe quota and closes the circuit.
- **Closed again**: call 8 runs normally through the now-healthy dependency, call count reaches 6 (3 failures + 2 probes + 1 normal call) — every call that should have reached the dependency did, and no call that shouldn't have did.

The example asserts every transition and the call count inline (`assert_eq!`), so a regression in `CircuitBreaker`'s state machine fails the run, not just the eyeball check. No real sleeps: the cooldown is crossed by `ManualClock::advance`, never `std::thread::sleep`.
