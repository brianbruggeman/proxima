# proxima-core

Foundation primitives for the proxima workspace: the shared error type, lock-free rings, arenas, runtime-agnostic time, and marker traits. Everything else in proxima is built out of this crate.

Add it with `cargo add proxima-core`.

## What it composes

Nothing from proxima: it is the bottom layer. Its lib docs call it the one crate that cannot itself be expressed as a pipe, because `proxima_primitives::Pipe` is defined one level up, on top of it. (`proxima-build` is its build-dependency, used by `build.rs` to bake sizing constants.) Layers above it:

- `proxima-primitives` defines the `Pipe` algebra over these types.
- `proxima-runtime`, `proxima-codec`, `proxima-config` and `proxima-telemetry` use `ProximaError` from here; `proxima-telemetry` also re-exports `ring`.

Main surface: `ProximaError` (typed `*Kind` variants are the convention; the `String` variants remain for crates not yet migrated), `ring` (`Ring<T>`, `StaticRing<T, N>`), `arena`, `batch`, `datagram_batch`, `per_core`, `time`, `markers`, `factory`, `arch`, `histogram`, and the build-time `sized` constants generated from `proxima-core.toml`.

## Features and tiers

Default feature: `std`. Each module carries the finest tier it can reach, so a bare-metal consumer takes a subset:

| tier | enable | modules |
|---|---|---|
| no_std, no alloc | `--no-default-features` | `arch`, `datagram_batch`, `factory::Named`, `markers`, `per_core`, `ring`, `time` (inline fixed-capacity forms sized by `build.rs`) |
| no_std + alloc | `alloc` | adds `arena`, `batch`, `accept_rule`, `serving_state`, and the heap forms of the buffered modules |
| std | `std` (default) | adds `buffer`, `signal` |

Opt-in features (all default-off):

- `registry`: `FactoryRegistry`, a copy-on-write name-to-factory registry (std).
- `config`: config-as-composition for the registry (`FactorySpec` with a fluent builder and a conflaguration loader); implies `registry`.
- `live`: `Live<T>`, a lock-free live-swappable value (std).
- `io-async`: poll-based `AsyncRead` / `AsyncWrite` traits over an associated error type; no_std, no alloc. `io-async-compat` and `io-async-compat-tokio` bridge `futures-io` and `tokio::io` to it (std).
- `park`: `SlotPark`, a futex wait-for-a-freed-slot primitive; no_std but needs an OS.
- `time-driver-mock`, `time-driver-wasm`, `time-driver-prime-wheel`: timer drivers for `time`. The `time-driver-std-thread` driver comes with `std`.
- `loom`: swaps atomics for loom's, for model-checking the ring.

`histogram` is absent on targets without 64-bit atomics (for example cortex-m).

## License

MIT OR Apache-2.0.

Repository: <https://github.com/brianbruggeman/proxima>
