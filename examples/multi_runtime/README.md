# multi_runtime

Prime and tokio serve HTTP concurrently, in one process, on the same sans-IO pipe.

## Demonstrates

proxima's `Runtime` trait is an interface, not a process-wide singleton. `App`
holds an `Option<Arc<dyn Runtime>>`; nothing in the type stops two `App`
instances in the same process from installing two different `Runtime` impls.
Each `App` still owns exactly one runtime — the composite property comes from
running two `App` instances side by side, not from one `App` juggling two.
This example builds a `PrimeRuntime` and a `TokioPerCoreRuntime` that way,
binds one HTTP/1 listener on each, and mounts the SAME `Pipe` instance (one
`Arc<AtomicU64>` counter, shared) on both. `TokioPerCoreRuntime` spawns its
own OS threads rather than attaching to an ambient executor, so it needs no
`#[tokio::main]` around it — that's why this example's `fn main` is plain,
synchronous, and runtime-agnostic even with tokio in the mix.

tokio, glommio, and monoio are all process-singletons — exactly one runtime
instance may drive a process. Two processes could each run one, but they
cannot share memory without IPC. Here, one process, two live executors, one
`Arc` that both mutate directly — no IPC, no serialization, no second
process. That composability is only possible because the request/response
core (`SendPipe`) is sans-IO: it does not know or care which runtime called
it, so the same instance is reachable from either. The sharing itself is
built once: the `Arc<AtomicU64>` moves into one `SharedCounterPipe`, that pipe
is wrapped once into a `PipeHandle`, and the SAME handle is `.clone()`d onto
both `App`s' routers — both listeners dispatch into one pipe object, mutated
from whichever runtime's worker thread happens to be running the request.

## Run

```sh
cargo run --example multi_runtime --features "runtime-tokio tokio"
```

(`runtime-tokio` is required — `TokioPerCoreRuntime` is opt-in; `tokio` is
also required — the tokio-backed `AcceptorFactory` lives behind the full
`tokio` feature, not the narrower marker. The prime side ships in
`serve-prime`, which is on by default.)

## Expected output

```
prime listener on 127.0.0.1:8081 (prime runtime, 2 cores)
tokio listener on 127.0.0.1:8082 (tokio runtime, 2 cores)
GET http://127.0.0.1:8081/ (prime) -> shared_total=2
GET http://127.0.0.1:8081/ (prime) -> shared_total=4
GET http://127.0.0.1:8081/ (prime) -> shared_total=6
GET http://127.0.0.1:8082/ (tokio) -> shared_total=1
GET http://127.0.0.1:8082/ (tokio) -> shared_total=3
GET http://127.0.0.1:8082/ (tokio) -> shared_total=5
observed totals across both listeners (sorted): [1, 2, 3, 4, 5, 6]
prime drained: cores_acked=2 hooks_drained=0
tokio drained: cores_acked=2 hooks_drained=0
both runtimes shut down cleanly; final shared total = 6
```

The exact interleaving of `shared_total` values between the two ports varies
run to run (two client threads race two independently-scheduled runtimes) —
that non-determinism IS the proof. What's invariant: the six totals, pooled
across both ports, are always the contiguous set `{1..6}` with no gaps and no
repeats. That means every increment from both runtimes landed on the one
counter with no lost updates and no double counts — real concurrent mutation
of shared state across the prime/tokio boundary, not two independent tallies
that happen to print next to each other.
