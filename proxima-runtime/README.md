# proxima-runtime

The runtime trait surface for proxima: per-core spawn, cross-core dispatch, and a background pool, defined independent of any concrete executor.

Add it with `cargo add proxima-runtime`.

## What it provides

- `Runtime`: spawn `!Send` futures on the current core (`spawn_on_current_core`), `Send` futures on a designated core (`spawn_on_core`, which returns `SpawnError::InboxFull` or `SpawnError::Disconnected` rather than blocking), and a `Send` factory that builds a `?Send` future on a target core.
- `BackgroundPool`: cross-thread CPU-bound work, awaited from a per-core caller through a `Send` handle.
- `SpawnRequest`, `CoreId`, `SpawnError`: the cross-core dispatch message and its identifiers.
- `primitives`: the `RuntimeFactory` sibling trait and the `*Like` shapes (`MutexLike`, `NotifyLike`, `JoinSetLike`, `LocalMutexLike`, `LocalNotifyLike`, `JoinError`) so code can be generic over the runtime that supplies its lock.
- `ext`: a fluent spawn surface over `Runtime`.

The trait and value types live here, outside the `proxima` umbrella, so that `prime` can be a separate crate without depending on it.

## What it composes

It depends on `proxima-core` (for `ProximaError`). Implementations of `Runtime`:

- `prime` (the `proxima-prime` package): `PrimeRuntime`, a from-scratch per-core runtime.
- `proxima_runtime::tokio::TokioPerCoreRuntime` (feature `tokio`): N tokio current-thread runtimes pinned one per CPU core.
- `RayonBackgroundPool` (feature `rayon`): a `BackgroundPool` over rayon's work-stealing pool.

`proxima-primitives` depends on this crate (optionally, under its `alloc` feature) for the runtime types.

## Features and tiers

Default feature: `std`.

| feature | effect |
|---|---|
| `alloc` | the `Runtime`, `BackgroundPool`, `ext` and `primitives` surface (no_std + alloc) |
| `std` (default) | implies `alloc`; adds `RuntimeFactory` / `LocalRuntimeFactory` |
| `tokio` | `TokioPerCoreRuntime` and the `Tokio*` primitive adapters; implies `std` |
| `io-uring` | on Linux, swaps epoll for io_uring under the tokio backend; implies `tokio` |
| `rayon` | `RayonBackgroundPool`; implies `std` |
| `concurrency` | adaptive per-core concurrency control (a controller with preset strategies, tunable from TOML/env); implies `std` |
| `runtime-prime-full` | empty; kept so dependents' feature forwarding still resolves |

With no features the crate is `no_std` and exposes only the error and identifier types.

## License

MIT OR Apache-2.0.

Repository: <https://github.com/brianbruggeman/proxima>
