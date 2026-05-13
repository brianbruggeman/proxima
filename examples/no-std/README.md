# no-std — the sans-IO core on bare metal

## Builds on

[transform](../transform/README.md) — the `Pipe` you write here is the same trait, on a smaller planet.
[config](../config/README.md) — typed config, but resolved once, before the program exists.

## What it demonstrates

Every other rung in this curriculum runs on a host with an OS, a heap, and a
runtime under it. This one asks what's left of proxima once all three are
gone: `FrameStore` in `src/lib.rs` is a plain
[`Pipe`](../../proxima-primitives/src/pipe/primitives.rs) — the exact trait `transform`
teaches — that writes a borrowed frame into a
[`RingSink`](../../proxima-primitives/src/pipe/drain_sink.rs), a fixed-capacity array
sized by two `const` generics. No `Box`, no `Vec`, no allocator, no executor.
`block_on` is `proxima_primitives::block_on` — the workspace's `core`-only
`Waker::noop()` poll loop (stable since Rust 1.85) that drives the pipe's future
to completion. Polling in a loop is the whole "runtime."

The `config` rung's `conflaguration::Settings` resolves fields at **runtime**
from env vars and files. That machinery needs `std` (`std::env`, a filesystem)
and doesn't exist here. So the two knobs this pipe needs — `RING_SLOTS`,
`RING_SLOT_BYTES` — are read from `no-std.toml` by `build.rs`, once, on the
host, before the crate is even compiled, and baked into `pub const`s
(`OUT_DIR/no_std_config.rs`, included via `include!`). There is no code path
in the compiled binary that re-reads `no-std.toml`, checks an env var, or
opens a file — the constants ARE the config. This is the same recipe
`proxima-primitives/build.rs` uses to bake `RETRY_STATUS_CAP` for
`RetryRules`'s no-alloc `StatusSet`; this example isolates it to one file.

The crate is `#![no_std]` **by default** — `cargo build` with no flags proves
it. A `std` feature exists for exactly two things libtest and a demo binary
need: running `cargo test`, and `println!`-ing the pipe's output from
`no-std-demo`. Turning `std` on doesn't change `FrameStore` or `block_on` at
all — same source, same behavior, just a different attribute on the crate.

`ring_capacity` is the same claim made about `#[proxima::pipe]`
(`00-foundations.md` section 7): the macro's generated struct is always a
fieldless ZST and always derives `Clone` unconditionally, so it has to hold
even where there is no allocator to fall back on. `#[proxima_macros::pipe]`
on a plain `fn` (no `send`, no `async`) expands to a `struct ring_capacity;`
with `#[derive(::core::clone::Clone)]` plus an `impl UnpinPipe for
ring_capacity`, wrapping the call in `core::future::ready` — the whole
expansion is `core`-only. Because this fn sits in the crate's default
(zero-feature) module, the bare `#![no_std]` build below is *also* the
compile-time proof that the macro's auto-`Clone` costs nothing at the floor;
`tests::macro_generated_pipe_clones_and_calls_at_the_bare_floor` (`--features
std`) then proves it isn't just syntax — cloning the pipe and calling the
clone through `UnpinPipe::call` returns the real value.

## Build (the no_std proof)

```
cargo build -p proxima-example-no-std
```

No features, no `--no-default-features` needed — `std` defaults off. This is
the cliff: if `FrameStore`, `block_on`, or anything they touch reached for
`std`, this line would fail to compile.

## Test and run (opts into `std`)

```
cargo test -p proxima-example-no-std --features std
cargo run  -p proxima-example-no-std --bin no-std-demo --features std
```

## What you'll see

```
stored "hello" (ring occupancy = 1)
rejected "this-frame-is-way-too-long-for-one-slot": TooLarge
stored "world" (ring occupancy = 2)
```

## A note on "bare metal"

The `cargo build` above proves `#![no_std]` + no-alloc on the **host**
target — no OS services used, but still linked as a library, not flashed to a
board. Cross-compiling to a real Cortex-M target also works:

```
cargo build -p proxima-example-no-std --target thumbv7em-none-eabihf
```

32-bit ARM targets have no native 64-bit atomics, which used to block any
crate that (transitively) touched `core::sync::atomic::AtomicU64` — including
`proxima-primitives/src/pipe/sink_front.rs`, compiled into every build of this
crate regardless of whether `FrameStore`'s own path (`primitives` +
`drain_sink`) needs it. `portable_atomic` closed that gap, so the whole crate
now builds clean on the embedded cliff, not just the modules `FrameStore`
touches. `scripts/thumbv7m-cliff-gate.sh` is the CI gate that proves this
holds for the full no_std + alloc floor tier, not just this one crate.
