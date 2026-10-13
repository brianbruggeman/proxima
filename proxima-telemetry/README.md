# proxima-telemetry

Core-native telemetry for proxima: traces, metrics, logs and events, recorded through per-core rings and exported through pluggable exporters.

Add it with `cargo add proxima-telemetry`.

## What it provides

- Identifiers and records: `id` (W3C trace and span ids), `trace` (span, event and link records), `metric` (counter, gauge, up/down counter instruments), `log`, `tag`, `level`, `sampler`, `propagation` (W3C `traceparent` and baggage codec over `HeaderList`).
- `recorder` (std): per-core thread-local ring buffers drained by a background thread, with `Recorder` and `RecorderBuilder`.
- `export` (std): `Exporter` and `set_default_recorder`; native and OTLP exporters; `config::TelemetryConfig` is the layered config surface.
- `emit` (feature `emit`): the `error!`, `warn!`, `info!`, `debug!` and `trace!` macros plus a hierarchical filter. `emit_statically_enabled` folds a build-time severity floor into a `const`, so a statically disabled level emits no runtime code.
- `Metrics`, `Labels`, `Telemetry` and `TelemetryHandle` (std, re-exported from a private `legacy` module): the older surface that existing proxima code still uses. The crate docs direct new code to the `emit` macros and the `recorder`, `trace`, `metric` and `log` primitives.

From the crate docs:

```rust
use proxima_telemetry::emit_statically_enabled;
// folds at compile time — usable in `const` context, so a guarded emit below
// the floor is dead code the optimizer deletes (zero runtime cost).
const KEEPS_ERROR: bool = emit_statically_enabled(17); // error severity
assert!(KEEPS_ERROR); // default floor is trace (1), so error is enabled
```

## What it composes

- `proxima-core`: the lock-free `ring` (re-exported as `proxima_telemetry::ring`), `arch` CPU hints, `live` swappable cells, and `park` for lossless backpressure.
- `proxima-primitives` (from `alloc`): `HeaderList` for propagation, and the pipes in `out` and `pipes`.
- `proxima-clock` (from `std`): the monotonic-to-wall-clock bridge behind `SystemClock`.
- `proxima-macros` (feature `macros`): re-exports `span`, `instrument` and `SpanCarrier` from the crate root.
- `proxima-build` at build time: bakes `proxima-telemetry.toml` sizing constants, overridable with `PROXIMA_TELEMETRY_*` env vars.

## Features and tiers

Default feature: `std`.

- no_std, no alloc (no features): `sized`, `id`, `error` and `trace::{status, kind}`.
- `alloc`: no_std + alloc data and primitive layer: tag, trace records, log, metric instruments, sampler, ring, level, clock, propagation. Checked on thumbv7m and thumbv7em per the `Cargo.toml` comments.
- `std` (default): implies `alloc`; adds the recorder, config loading, exporters and pipes.

Opt-in features: `emit`, `histogram`, `instrument-metrics` (a span's close also records its duration into a per-name histogram), `deferred-metric-fold`, `lossless-backpressure` (a full ring makes the producer drain a batch itself or park, instead of dropping), `elevation` (tail-sampled trace buffering with error-triggered replay), `otlp-http`, `otlp-grpc`, `macros`, `tee-generic`, `tracing-init`, `loom` (test only).

## License

MIT OR Apache-2.0.

Repository: <https://github.com/brianbruggeman/proxima>
