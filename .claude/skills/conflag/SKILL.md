---
name: conflag
description: How to use the `conflaguration` crate for typed configuration, across the whole tier matrix. The std/alloc tier is RUNTIME config — `#[derive(Settings)]`, env vars, config files, the layered fluent builder, `Validate`/`ConfigDisplay`. The no_std/no_alloc tier has NO runtime config — constants ARE the config, baked at BUILD time by a `build.rs` that reads a TOML (or a `#[derive(ConfigCodegen)]` struct) and emits a per-platform/os/arch set of `pub const`s. Covers proxima's house pattern (`#[derive(Builder, Deserialize, Serialize, Settings)]` + `Validate` + a `layered()`/`from_path`/`from_env`/`with_*` builder) and the build-time `sized` module that seeds the runtime defaults so there's no double source of truth. Use when adding or wiring a config struct, when a crate must build no_std/no_alloc but still be configurable, when designing a build.rs that generates constants, or when the user says "conflag", "conflaguration", "how do we configure X", "Settings derive", "build-time constants", "sized config", "noalloc config".
---

# conflag (conflaguration)

`conflaguration` is the workspace config crate (`~/repos/mine/conflaguration`,
path-dep `version = "2.0.0"`). Typed config from env vars, files, and fluent
builders — with a build-time codegen path for the tiers that have no runtime.

The governing fact: **config availability is a function of the tier.**

| tier | runtime config? | mechanism |
|---|---|---|
| std + alloc | yes | `#[derive(Settings)]`, env, files, layered builder |
| no_std + alloc | limited | builder/serde possible; no fs/env — feed it bytes |
| no_std + no_alloc | NO | constants baked at build time; `build.rs` IS the config surface |

So a configurable primitive that must reach the bare-metal floor needs BOTH: a
build-time constant (the floor's only knob) AND a std runtime override on top —
and the runtime default must be *seeded from* the build-time constant, never
duplicated. That bridge is the whole discipline.

## Tier 1 — std/alloc: runtime config

### The derive

```rust
use conflaguration::{Settings, Validate, init};

#[derive(Settings, Validate)]
#[settings(prefix = "APP")]
struct Config {
    #[setting(default = 8080)]
    port: u16,

    #[setting(default = "localhost")]
    host: String,

    #[setting(nested)]                 // sub-struct: APP_PRIMARY_HOST
    primary: Database,

    #[setting(flatten)]                // merged, no segment: APP_HOST
    extra: Database,

    #[setting(sensitive, default = "")] // masked in ConfigDisplay
    token: String,
}

let config: Config = init()?;          // defaults <- APP_* env, validated
```

Field attributes worth knowing: `default` / `default = v` / `default_str`,
`envs = "KEY"` or `envs = ["K1","K2"]` (cascade, first set wins), `override`
(ignore prefix), `resolve_with = "fn"` (custom `fn(&str) -> Result<T, E>`),
`nested` / `nested, prefix = "RO"` / `nested, override_prefix = "SHARED"`,
`flatten`, `skip`, `sensitive`. Conflicting combinations are rejected at compile
time. Loaders: `init()`, `Config::from_env()`, `from_file("c.toml")`,
`from_file_then_env("c.toml")` (file format needs the `toml`/`json`/`yaml`
feature; detected by lowercase extension).

### The layered builder

Stack sources; first source builds in full, later sources either **override**
(`.file()`, `.mapping()`, `.env()` — replace keys, last wins) or **overlay**
(`.overlay_file()`, `.overlay_mapping()` — fill only unset keys, never clobber):

```rust
let config: Config = conflaguration::builder()
    .file("base.toml")          // base layer (full)
    .file("prod.toml")          // override: prod's keys win
    .overlay_mapping(fallbacks) // overlay: fill only still-unset keys
    .env()                      // override: env wins
    .validate()
    .build()?;

// struct <-> fluent flop: seed from an owned value, patch, hand back
let patched: Config = conflaguration::builder()
    .value(existing)
    .overlay_file("local.toml")
    .override_with(|cfg| /* compute from running state */ ())
    .build()?;
```

File/mapping sources require `serde::Serialize` (the merge reads the value back).

### proxima's house pattern (copy this for any new config struct)

Every proxima config surface (`TelemetryConfig`, `EmitConfig`,
`InstrumentConfig`) is the SAME shape, so it is first-class as BOTH a
conflaguration target AND a fluent builder (principle: both, not one):

```rust
use bon::Builder;
use conflaguration::{Settings, Validate};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "INSTRUMENT")]
#[builder(derive(Clone, Debug))]
pub struct InstrumentConfig {
    #[setting(default = false)]
    #[serde(default = "default_metrics")]      // defaults pulled FROM the sized floor
    #[builder(default = default_metrics())]    // (see the bridge, below)
    pub metrics: bool,
}

fn default_metrics() -> bool { crate::sized::INSTRUMENT_METRICS_DEFAULT }
```

Plus a hand-written `XxxLayerBuilder` with **call-order precedence** — `.with_*`
BEFORE `.from_path`/`.from_env` means operator config wins; AFTER means code
overrides win:

```rust
impl InstrumentConfig {
    pub fn layered() -> InstrumentLayerBuilder { /* seeded from Default */ }
}
impl InstrumentLayerBuilder {
    pub fn from_path<P: AsRef<Path>>(mut self, p: P) -> Result<Self, conflaguration::Error> {
        self.inner = conflaguration::from_file(p.as_ref())?; Ok(self)
    }
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        self.inner = InstrumentConfig::from_env()?; Ok(self)
    }
    pub fn with_metrics(mut self, on: bool) -> Self { self.inner.metrics = on; self }
    pub fn build(self) -> InstrumentConfig { self.inner }
}
```

The invariant this buys: **one built config == one serialisable wire form == one
applied policy.** A built struct round-trips through serde unchanged, and a
single `.apply(&recorder)` seam drives the running system — annotations and call
sites never change when config does.

## Tier 2 — no_std/no_alloc: constants ARE the config

Below std there is no fs, no env at runtime, no serde-from-file. The only place
to make a decision is the build. So a `build.rs` resolves the config ONCE at
compile time and bakes it into `pub const`s. Two ways:

### A. conflaguration-native codegen (`feature = "codegen"`)

```rust
// build.rs
use conflaguration::{Settings, ConfigCodegen, codegen};

#[derive(Settings, ConfigCodegen)]
#[settings(prefix = "APP")]
struct Build {
    #[setting(default = 1024)]
    ring_capacity: usize,
}

fn main() -> conflaguration::Result<()> {
    let build = Build::from_env()?;             // defaults + APP_* at BUILD time
    codegen::write_consts(&build, "build.rs")?; // -> pub const RING_CAPACITY: usize = 1024;
    build.emit_cfg("app");                      // -> cargo:rustc-cfg=app_* directives
    codegen::rerun_for::<Build>();              // rerun-if-env-changed from the struct (no drift)
    Ok(())
}
```

```rust
mod generated { include!(concat!(env!("OUT_DIR"), "/build.rs")); }
type RingBuffer = [u8; generated::RING_CAPACITY];  // build-time const sizes the TYPE
```

`ConfigCodegen` covers flat structs of scalar fields; the rerun list derives
from the struct's own attributes, so there is no hand-kept env-var list to drift.

### B. proxima's hand-rolled `sized` TOML pattern (what `proxima-telemetry` does)

A crate-root TOML holds the defaults; `build.rs` reads it, applies optional
`PROXIMA_<CRATE>_<SECTION>_<KEY>` env overrides, and writes a const module:

```rust
// build.rs — resolve(table, "instrument", "metrics") reads the TOML then the
// PROXIMA_TELEMETRY_INSTRUMENT_METRICS env override, emitting:
//   pub const INSTRUMENT_METRICS_DEFAULT: bool = false;
//   pub const EMIT_COMPILE_FLOOR: u8 = 1;            // [emit] max_level = "trace"
// into OUT_DIR/proxima_telemetry_sized.rs
```

```rust
// lib.rs
pub mod sized {
    include!(concat!(env!("OUT_DIR"), "/proxima_telemetry_sized.rs"));
}

// the compile-time half of the emit filter: a const fn over a const folds to a
// bool at the call site, so a below-floor emit is dead code the optimizer
// deletes — zero runtime cost, works even at no_std+no_alloc.
#[must_use]
pub const fn emit_statically_enabled(severity: u8) -> bool {
    severity >= sized::EMIT_COMPILE_FLOOR
}
```

Pick B over A when the config is sectioned, carries rich doc comments, needs
non-scalar resolution (a level NAME → severity u8), or must drive both a const
AND a `cargo:rustc-cfg`. Pick A for a flat scalar struct you want with no
hand-written reader.

### Per-platform / per-os / per-arch constants

This is the point the no_alloc tier exists for: the constant set can DIFFER by
target, chosen in `build.rs` from the cargo-provided target env vars
(`CARGO_CFG_TARGET_OS`, `CARGO_CFG_TARGET_ARCH`, `TARGET`):

```rust
// build.rs — select the backend/const set per (os, arch) combo
let target_os   = env::var("CARGO_CFG_TARGET_OS")?;
let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;
match (target_os.as_str(), target_arch.as_str()) {
    ("linux", "x86_64") => { /* emit linux consts / link linux backend */ }
    ("macos", "aarch64") => { /* emit macos consts / link Hypervisor.framework */ }
    _ => {}
}
```

(Live examples: `proxima-vm/build.rs` selects the C backend + links
`Hypervisor` on macOS aarch64; `proxima-process/build.rs` picks the libc-shim
artifact by `target_os`.) The same `build.rs` can therefore bake a different
`RING_CAPACITY`, page size, or feature default per platform — the bare-metal
floor's only configuration mechanism, resolved before a single instruction runs.

## The bridge — one source of truth across tiers

The std runtime default must be SEEDED from the build-time constant, not
re-declared. In `proxima-telemetry`: `InstrumentConfig::metrics` defaults via
`default_metrics() -> sized::INSTRUMENT_METRICS_DEFAULT`. So:

- at no_std+no_alloc, `sized::INSTRUMENT_METRICS_DEFAULT` IS the config;
- at std, that same const seeds the runtime `InstrumentConfig` default, which a
  config file / env / `.with_metrics(true)` then overrides.

A test pins the invariant (`defaults_track_the_sized_floor`): the fluent default
equals the `sized` const. If you add a knob, add it to the TOML/`sized` const
AND seed the runtime default from it — never type the default twice.

## Features

`derive` (the `Settings`/`Validate`/`ConfigDisplay`/`ConfigCodegen` macros),
`codegen` (the build-support module), `toml`/`json`/`yaml` (file parsing).
proxima's workspace dep enables `["derive", "toml"]`.

## Source pointers

- the crate + README (full attribute table, examples): `~/repos/mine/conflaguration`,
  `examples/{codegen,sizing,database,logging,http}`.
- proxima house pattern (std runtime): `proxima/proxima-telemetry/src/config.rs`,
  `src/emit/config.rs`, `src/metric/instrument_config.rs` (`layered`/`from_path`/
  `from_env`/`with_*` + `apply`).
- the `sized` build pattern: `proxima/proxima-telemetry/build.rs` +
  `proxima-telemetry.toml`; the `sized` module + `emit_statically_enabled` in
  `src/lib.rs`.
- per-platform build selection: `proxima/proxima-vm/build.rs`,
  `proxima-process/build.rs`.
