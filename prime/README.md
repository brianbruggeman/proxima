# prime

A from-scratch per-core async runtime: a bounded SPSC inbox, a typed-future fast path, and the reactor, executor, timer and background-pool primitives a runtime is made of.

The crate is published as `proxima-prime`; add it with `cargo add proxima-prime` and import it as `prime`.

## What is in it

Two layers:

- `prime::core` (no_std + alloc; `core::*` and `alloc::*` only, never `std::*`): `inbox`, `timer`, `local_executor` (a per-thread `?Send` future executor), `inline_task`, `sized`.
- `prime::os` (std plus libc): `reactor`, `child` (async child-process wait), `background` and `par` (background pool), `cohort`, `core_shard`, `signal`, `net`, `io_uring`, `primitives`, and `runtime`, which holds `PrimeRuntime`. Most of these are behind `runtime-prime-*` features.

`PrimeRuntime` implements `proxima_runtime::Runtime`. `prime::config::PrimeConfig` is its typed configuration: it loads from `PRIME_*` env vars or a TOML file through `conflaguration`, and composes into `PrimeRuntime::builder()`.

## What it composes

From `Cargo.toml`:

- `proxima-runtime`: the `Runtime` trait it implements.
- `proxima-core`: `ProximaError`, rings, time.
- `proxima-primitives` (with `alloc` and `async-mutex`): the pipe algebra and the waker-based async mutex.
- `proxima-clock`: `TickCell`, for the virtual-clock timer wheel.
- `proxima-telemetry` (with `emit`): instrumentation.

`proxima-build` is its build-dependency (sizing constants from `prime-runtime.toml`). Optional: `tokio` (compat mode only), `rayon`, `io-uring`, `core_affinity`, `socket2`.

## Usage

`PrimeRuntime` needs the `std`, `runtime-prime-executor`, `runtime-prime-inbox-alloc`, `runtime-prime-reactor` and `runtime-prime-bgpool` features together; the default features alone do not include it. From the `PrimeRuntime::builder` docs:

```rust
use prime::os::runtime::PrimeRuntime;
let runtime = PrimeRuntime::builder()
    .cores(4)
    .background_inline()  // tests / small workloads
    .build()?;
```

The snippet sits inside a function that returns a `Result`.

## Features and tiers

Default features: `std` and `runtime-prime`.

- no_std + alloc: `alloc`, with the `runtime-prime-*` features that name it. `runtime-prime-inbox-const` is the only runtime feature that compiles no heap type, so it is the floor.
- `wasm`: the `prime::core` executor (run queue, timer wheel, inbox) with no reactor, no tokio and no threads; builds on `wasm32-unknown-unknown`: `cargo build -p prime --target wasm32-unknown-unknown --no-default-features --features wasm`.
- `std` (default): implies `alloc`; adds `prime::os` and the std-side dependencies.

Component features, all additive: `runtime-prime-inbox-alloc`, `runtime-prime-inbox-const`, `runtime-prime-inbox-dynamic`, `runtime-prime-timer`, `runtime-prime-executor`, `runtime-prime-reactor` (plus `-harvest-io` and `-trace`), `runtime-prime-bgpool` (plus `-rayon`, `-par`, `-async`), `runtime-prime-cohort`, `runtime-prime-config` (a TOML runtime-override file), `runtime-prime-virtual-clock`, `runtime-prime-thread-identity`, `prime-tokio-compat` and `prime-tokio-compat-inverted` (run alongside tokio), `io-uring`, `rayon`.

## License

MIT OR Apache-2.0.

Repository: <https://github.com/brianbruggeman/proxima>
