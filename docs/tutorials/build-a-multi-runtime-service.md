# Build a multi-runtime service

**Prerequisites:** [Foundations](./00-foundations.md) — serving a pipe. Requires `--features "runtime-tokio,tokio,http1"`.
**You will:** serve the *same* sans-IO pipe on prime **and** tokio concurrently in one process, sharing state across the runtime boundary. proxima's `Runtime` is an interface, not a process-singleton.
**New concepts (in order):** the `Runtime` trait (prime vs tokio impls) · two runtimes in one process · shared state across the boundary.
**Answer key:** [`examples/multi_runtime/main.rs`](../../examples/multi_runtime/main.rs) — `cargo run --example multi_runtime --features "runtime-tokio,tokio,http1"`.

`http1` registers the "http" listen protocol both `App`s bind (`RunConfig::http`/`ListenerSpec::http` resolve to it) — without it this fails at runtime with `Registry("no listen protocol named 'http'")`, not a compile error; `Cargo.toml`'s own `required-features` for this example already lists it.

The example frames it: *"tokio and glommio and monoio are process-singletons — one runtime per process. proxima's `Runtime` trait is just an interface: any number of implementations can live in the same process side by side."*

## 1. One pipe, runtime-neutral

The pipe is a plain sans-IO `SendPipe` with shared state; nothing in it names a runtime (`multi_runtime/main.rs:40-59`):

```rust
struct SharedCounterPipe { total: Arc<AtomicU64> }

impl SendPipe for SharedCounterPipe {
    type In = Request<Bytes>; type Out = Response<Bytes>; type Err = ProximaError;
    fn call(&self, _request: Request<Bytes>)
        -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let total = self.total.clone();
        async move {
            let observed = total.fetch_add(1, Ordering::AcqRel) + 1;
            Ok(Response::ok(format!("shared_total={observed}\n")))
        }
    }
}
let pipe: PipeHandle = into_handle(SharedCounterPipe { total: shared_total.clone() });
```

`shared_total` is an `Arc<AtomicU64>` — `let shared_total = Arc::new(AtomicU64::new(0));` — created once, earlier in `main` (`multi_runtime/main.rs:68`). Read it as "a shared counter that's safe to bump from many threads at once"; `.clone()` makes another pointer to that *same* counter, not a copy of its value. Inside `call`, `total.fetch_add(1, Ordering::AcqRel)` bumps the counter by one and returns the value it had *before* the bump (hence the `+ 1`) — the `Ordering::AcqRel` detail is about cross-thread memory visibility and isn't important here.

## 2. Two runtimes, two listeners, one pipe

Build two `App`s — one on prime, one on tokio — each with its own runtime and acceptor factory, both mounting the **same** handle (`multi_runtime/main.rs:77-93`):

```rust
let prime_runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(2)?);
let prime_app = App::builder()
    .with_defaults()?
    .build()?
    .with_runtime(prime_runtime.clone())
    .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
prime_app.mount("/", pipe.clone())?;

let tokio_runtime: Arc<dyn Runtime> = Arc::new(TokioPerCoreRuntime::new(2)?);
let tokio_app = App::builder()
    .with_defaults()?
    .build()?
    .with_runtime(tokio_runtime.clone())
    .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
tokio_app.mount("/", pipe.clone())?;
```

`mount("/", pipe.clone())` is the same terse mount from Foundations Section 12 — `pipe` is already a `PipeHandle`, so it is passed straight in, no wrapper. The `2` passed to `PrimeRuntime::new` and `TokioPerCoreRuntime::new` is the number of worker threads (cores) that runtime gets to run on. `with_acceptor_factory` is the other half of "run on this runtime": an acceptor factory is the piece that accepts incoming network connections for a given runtime, so `proxima_net::prime::PrimeAcceptorFactory` accepts connections on prime's reactor (its event loop) and `proxima_net::tokio::TokioAcceptorFactory` accepts them on tokio's — every runtime needs its own. Note both live as submodules of the one `proxima_net` crate (`proxima_net::prime`, `proxima_net::tokio`), not as separate `proxima_net_prime`/`proxima_net_tokio` crates.

`App::builder().with_defaults()?.build()?` builds the app the same way Foundations' `App::new()` does, just spelled out through the builder so `.with_runtime(...)` and `.with_acceptor_factory(...)` can immediately override the defaults it installs. `with_runtime` + `with_acceptor_factory` is the *only* difference between serving on prime vs tokio — the pipe is identical.

Each app then builds its own listener directly, no `block_on` needed — `build_listener` is a plain synchronous call that blocks the calling thread only until its own accept lane has acked ready, no polling, no sleeping (`multi_runtime/main.rs:98-99`):

```rust
let prime_listener = prime_app.build_listener(ListenerSpec::http(prime_bind))?;
let tokio_listener = tokio_app.build_listener(ListenerSpec::http(tokio_bind))?;
```

Because each `build_listener` call returns as soon as *that* listener is accepting — and the serving itself then runs on the runtime's own worker threads, not on the thread that called `build_listener` — both listeners end up running concurrently on their separate runtimes by the time the second call returns (`multi_runtime/main.rs:100-107`).

## 3. Shared state across the boundary

The two runtimes race requests from separate OS threads against the one `Arc<AtomicU64>`. The example asserts the shared counter is contiguous and lock-free across both — no lost updates, no double counts: it sorts the totals observed across both listeners and asserts they equal `1..=(REQUESTS_PER_LISTENER * 2)` (`multi_runtime/main.rs:109-148`).

## What you built

- **the `Runtime` trait** — an interface, not a process-singleton; prime and tokio are two impls.
- **runtime-agnostic serving** — the same sans-IO pipe served by two runtimes; only `with_runtime` / `with_acceptor_factory` differ.
- **shared state** — one `Arc<AtomicU64>` neither runtime owns, correct under concurrent access from both.

Because a pipe is sans-IO, it does not care which executor drives it — so you can run prime for its per-core kernel-bypass path and tokio for compatibility, in the same process, over the same logic. (The [`examples/runtime_select`](../../examples/runtime_select) example shows the simpler "same pipe, pick one runtime" case.)
