# proxima-build

A `build.rs` library: it resolves a typed build profile and per-crate sizing constants, then emits a generated Rust module and `cargo:` directives for the crate being built.

Add it as a build dependency: `cargo add --build proxima-build`.

## What it composes

No other proxima crates. It sits below them: `proxima-core`, `proxima-primitives`, `proxima-telemetry` and `prime` list it under `[build-dependencies]`. Externally it uses `conflaguration` (layered config loading), `serde`, `thiserror` and `toml`.

- `Profile` (re-exported from `profile`): the structural axes (executor, reactor, timer, TLS backend, QUIC and HTTP/3 implementation, and so on). It resolves in conflaguration's standard layering: type defaults, then a `profiles/<name>.toml` file (the name comes from the `PROXIMA_PROFILE` env var), then `PROXIMA_*` env overrides, then a `Validate` pass that rejects impossible combinations.
- `sizing`: reads a per-crate sizing TOML, resolves `[section].key` integers with a `<ENV_PREFIX>_<SECTION>_<KEY>` env override, and emits the matching `cargo:rerun-if-changed` and `cargo:rerun-if-env-changed` lines.

`resolve_profile()` returns `Error::ProfileEnvUnset` when `PROXIMA_PROFILE` is not set.

## Usage

From the crate docs, in a downstream `build.rs`:

```rust
fn main() -> proxima_build::Result<()> {
    let resolved = proxima_build::resolve_profile()?;
    proxima_build::emit_generated_module(&resolved)?;
    proxima_build::emit_cfg_directives(&resolved);
    proxima_build::emit_rerun_directives(&resolved);
    Ok(())
}
```

The generated module lands at `$OUT_DIR/proxima_profile.rs`; include it with `include!(concat!(env!("OUT_DIR"), "/proxima_profile.rs"));`.

The crate also ships a `dump-profile` binary that resolves the active profile and prints it: `PROXIMA_PROFILE=linux-daemon cargo run -p proxima-build --bin dump-profile` (run from the proxima workspace, where `profiles/` lives).

## Features and tiers

No cargo features. Std only: it reads env vars and writes files, and is meant to run inside `build.rs`, never in the compiled artifact.

## License

MIT OR Apache-2.0.

Repository: <https://github.com/brianbruggeman/proxima>
