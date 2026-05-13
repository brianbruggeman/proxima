# new-platform — bring up a new target

The porting workflow: point `PROXIMA_PROFILE` at a profile file and a
`build.rs` bakes that profile's axes into `pub const`s. No source changes
between platforms — only the resolved constants do.

## Builds on

[config](config/README.md) — same `conflaguration` machinery, moved from
runtime to build time.

[no-std](no-std/README.md) — this is the config story for the tier that
example lives on: no allocator, no OS, so no runtime config either.

## What it demonstrates

`config` showed `conflaguration::Settings` resolving a struct at runtime —
defaults, then a file, then env, then `Validate`. On a `no_std`/`no_alloc`
target there is no runtime to resolve anything in: no heap for a `String`
env var, often no filesystem, sometimes no environment at all. `conflaguration`
answers that tier differently — the same layering happens once, at build
time, and the result is baked in as `pub const`s the compiler can fold and
dead-code-eliminate.

The mechanism, exactly as it runs in this repo (`.github/workflows/no-std.yml`,
`proxima-build/src/lib.rs`, `prime/build.rs`, `proxima-core/build.rs`):

1. Set `PROXIMA_PROFILE=<name>` as an env var for the build.
2. `build.rs` calls `proxima_build::resolve_profile()`, which:
   - reads `PROXIMA_PROFILE` from the environment,
   - loads `<workspace-root>/profiles/<name>.toml` (a typed `Profile` struct —
     `alloc`, `std`, `executor`, `reactor`, `tls`, `timer`, `quic_impl`,
     `h3_impl`),
   - layers `PROXIMA_*` env-var overrides on top,
   - runs `Validate` for cross-axis sanity (e.g. `tls = "rustls"` requires
     `std = true` and `alloc = true`; `executor = "embassy"` requires
     `std = false`).
3. `proxima_build::emit_generated_module(&resolved)` writes
   `$OUT_DIR/proxima_profile.rs` — one `pub const` per resolved field.
4. The crate `include!`s that file into a `profile` module and uses the
   constants like any other compile-time constant.

Porting to a new target is adding one file — a new `profiles/<name>.toml` —
not touching any crate's source. This example doesn't add a new profile; it
runs the identical `build.rs` logic that `prime` and `proxima-core` (which
folded in the former `proxima-time` crate's timer axis) run in production
against two profiles that already ship in this repo, so the before/after is
real, not staged:

- `linux-daemon` — std + alloc + tokio + rustls (today's default; what
  `proximad` ships with on Linux).
- `bare-metal` — no_std + no_alloc + the static-pool `prime` executor + no
  TLS (Cortex-M/RISC-V class MCU with no allocator at all). Its timer axis
  resolves to `crate::time::drivers::unbound::DRIVER`, a stub that compiles
  everywhere and panics only if actually used — proxima ships no
  board-specific systick driver, since that's a platform concern; a real
  board overrides just this one axis with its own driver path, same as any
  other new-target port.

Same `src/main.rs`, same `build.rs`. Only the env var driving the build
changes, and the constants compiled into the binary follow.

## Run

```
PROXIMA_PROFILE=linux-daemon cargo run -p proxima-example-new-platform
PROXIMA_PROFILE=bare-metal   cargo run -p proxima-example-new-platform
```

`cargo build -p proxima-example-new-platform` with `PROXIMA_PROFILE` unset
also works — `build.rs` falls back to `Profile::default()` (the same
convention `proxima-core/build.rs` uses for the timer axis it folded in from
the former `proxima-time` crate) so the crate still compiles without the env
var.

## What you'll see

```
$ PROXIMA_PROFILE=linux-daemon cargo run -p proxima-example-new-platform -q
// source: /path/to/proxima/profiles/linux-daemon.toml

schema        = 1
alloc         = true
std           = true
executor      = tokio
reactor       = tokio-epoll
tls           = rustls
timer         = std-thread
quic_enabled  = false
quic_impl     = none
h3_impl       = none

$ PROXIMA_PROFILE=bare-metal cargo run -p proxima-example-new-platform -q
// source: /path/to/proxima/profiles/bare-metal.toml

schema        = 1
alloc         = false
std           = false
executor      = static-prime
reactor       = none
tls           = none
timer         = crate::time::drivers::unbound::DRIVER
quic_enabled  = false
quic_impl     = none
h3_impl       = none
```

Same binary source, two different builds, two different constant sets. There
is no `if profile == "bare-metal"` anywhere in `src/main.rs` — the branching
already happened, once, in `build.rs`, before a single line of the crate's
own logic ran. A real no_std crate reads `alloc`/`executor`/`timer` off this
same generated module to pick which code paths even get compiled in
(`#[cfg(proxima_alloc)]`, `#[cfg(proxima_executor = "...")]`, emitted by
`proxima_build::emit_cfg_directives`) — this example emits those cfgs too,
matching the real crates, even though `src/main.rs` doesn't branch on them.
