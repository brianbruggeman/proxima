# Atomic permit pool discipline

`AtomicPermitPool` is a default-off, synchronous counted gate for callers that
only need bounded `try_acquire` and RAII release. It is not a `Pipe`, an async
semaphore, or a publication cell. The existing `sync::Semaphore` remains the
incumbent when callers need waiting, close notification, or async acquisition.

| Row | Change | Verification | Decision |
| --- | --- | --- | --- |
| 1 | New `AtomicUsize` CAS pool with `AtomicPermit` drop release. | `cargo test -p proxima-primitives --features atomic-permits atomic_permits`: 4 passed; `cargo check -p proxima-primitives --no-default-features --features atomic-permits`: passed. | Keep as opt-in until the compare bench and Blackhole end-to-end arm establish a win. |
| 2 | First compare measurement; no code change. | Linux host, release Criterion run, 20 samples, 3 s warmup, 1 s measurement: atomic `10.491–10.503 ns` with 3 outliers; `Semaphore::try_acquire` `32.214–32.314 ns` with 1 outlier. | Micro-level signal favors the atomic pool; end-to-end allocation, latency, and contention evidence is still required. |

The required compare command is:

```text
cargo bench -p proxima-primitives --features atomic-permits --bench atomic_permits
```

The benchmark must be run on the same host with the Blackhole end-to-end
admission workload before any production-performance claim is made. No result
is recorded here until that command has produced provenance-tagged output.

Design constraints applied: fixed capacity and O(1) steady-state operations;
no allocation, lock, queue, wall-clock read, or dynamic dispatch; explicit
RAII lifecycle; default-off feature firewall. `Semaphore` was retained rather
than changed because its async wait/close contract is a distinct capability.
