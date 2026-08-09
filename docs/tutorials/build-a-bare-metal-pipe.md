# Build a bare-metal (no_std) pipe

**Prerequisites:** [Foundations](./00-foundations.md) — the base sans-IO `Pipe` in its **transform** form, and section 7's `#[proxima::piped]` (a plain fn becomes a fieldless, unconditionally-`Clone` `UnpinPipe`). Nothing here is served behind a listener, so you never reach `Handler`.
**You will:** run the same `Pipe` trait on bare metal — no heap, no executor, no OS — turn config into build-time constants baked before the program exists, and prove the macro-generated pipe from Foundations §7 holds at that same floor.
**New concepts (in order):** the sans-IO `Pipe` under `#![no_std]` (a fixed-capacity `RingSink`, no `Box`/`Vec`/alloc) · `block_on` via `Waker::noop` (a polling loop is the whole runtime) · build-time config constants (`build.rs` bakes a TOML into `pub const`s) · `#[proxima_macros::piped]`'s auto-`Clone` costing zero bytes with no allocator to fall back on.
**Answer key:** [`examples/no-std/src/lib.rs`](../../examples/no-std/src/lib.rs) — `cargo build -p proxima-example-no-std` (no flags = the `no_std` proof).

The example frames it, verbatim from its own module doc-comment (`no-std/src/lib.rs:1-2`): *"proxima's sans-IO core — a `Pipe` — compiling with no std and no runtime."* Its README puts the same point another way (`no-std/README.md:10-12`): *"Every other rung in this curriculum runs on a host with an OS, a heap, and a runtime under it. This one asks what's left of proxima once all three are gone."*

## 1. The same Pipe, on a smaller planet

The crate root carries `#![cfg_attr(not(feature = "std"), no_std)]` (`no-std/src/lib.rs:19`) — genuinely `#![no_std]` unless the `std` feature is turned on. `FrameStore` is a plain `Pipe` — the exact trait Foundations taught — writing a borrowed frame into a `RingSink`, a fixed-capacity array sized by two const generics. No `Box`, no `Vec`, no allocator. Its error type is a two-variant enum, `TooLarge | Full` (`no-std/src/lib.rs:33-41`) — no partial writes, no silent drops. Copied from `no-std/src/lib.rs:21-26` (imports) and `:47-76` (the pipe itself); the sizing constants are shown here as their literal values — section 3 covers where they really come from:

```rust
use core::cell::RefCell;
use core::future::Future;
use core::ops::ControlFlow;

use proxima_primitives::pipe::{DrainSink, Pipe, RingSink};

const RING_SLOTS: usize = 4;
const RING_SLOT_BYTES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    TooLarge,
    Full,
}

pub struct FrameStore {
    ring: RefCell<RingSink<RING_SLOTS, RING_SLOT_BYTES>>,
}

impl Default for FrameStore {
    fn default() -> Self {
        Self {
            ring: RefCell::new(RingSink::new()),
        }
    }
}

impl Pipe for FrameStore {
    type In = &'static [u8];
    type Out = usize;
    type Err = StoreError;

    fn call(&self, frame: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            if frame.len() > RING_SLOT_BYTES {
                return Err(StoreError::TooLarge);
            }
            let mut ring = self.ring.borrow_mut();
            match ring.accept(frame) {
                ControlFlow::Continue(()) => Ok(ring.len()),
                ControlFlow::Break(()) => Err(StoreError::Full),
            }
        }
    }
}
```

`In = &'static [u8]` is a borrowed frame — a real caller on bare metal borrows a DMA buffer (memory shared directly with hardware) or a stack arena rather than owning a `'static` slice; the demo uses `'static` byte-string literals to keep the example self-contained. `Out = usize` is the ring's occupancy after the write.

`self.ring.borrow_mut()` reaches through a `RefCell` — a container that lets you mutate what's inside through a shared `&self` reference, checking at runtime (instead of compile time) that nothing else is touching the ring at the same moment. `ring.accept(frame)` — a method from the `DrainSink` trait, which is why that trait is imported alongside `RingSink` above — then returns a `ControlFlow`, Rust's two-way "keep going or stop" signal: `Continue(())` means the frame was stored and there's still room, so `call` reports the new occupancy; `Break(())` means the ring is already full, so `call` maps it to `StoreError::Full`.

The same `trait Pipe { type In; type Out; type Err; fn call(...) -> impl Future<...>; }` as Foundations — only the tier is smaller. The hot path allocates nothing; state is a fixed array behind a `RefCell`. `RingSink` itself is a public primitive, `proxima_primitives::pipe::RingSink` — defined at `proxima-primitives/src/pipe/drain_sink.rs:47`, re-exported at `proxima-primitives/src/pipe/mod.rs:200`.

## 2. `block_on` is a polling loop

With no executor, `block_on` drives the pipe's future to completion with a `Waker::noop()` (stable since Rust 1.85) — polling in a loop **is** the runtime. This crate no longer hand-rolls its own copy: it re-exports the workspace's own no-runtime floor primitive, `pub use proxima_primitives::block_on;` (`no-std/src/lib.rs:25`), so every `no_std` caller in the workspace shares the identical loop rather than each crate writing its own. Copied verbatim from where it actually lives, `proxima-primitives/src/driver.rs:13-21`:

```rust
pub fn block_on<Fut: core::future::Future>(future: Fut) -> Fut::Output {
    let mut future = core::pin::pin!(future);
    let mut context = core::task::Context::from_waker(core::task::Waker::noop());
    loop {
        if let core::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
    }
}
```

You don't need to trace every token in that loop — `pin!`, `Context::from_waker`, `.poll(...)`, and `Poll::Ready` are just the mechanics of asking a future "are you done yet?". The loop asks, over and over, until the answer is yes; that polling loop **is** the runtime — the same black box Foundations waved off, just cracked open once so you can see there's no magic inside, only asking-and-checking. Its own doc comment names the reason it lives in `proxima-primitives` rather than in this example crate: "the floor every other `block_on` in the workspace points down to" — `proxima_runtime::block_on(&dyn Runtime, ..)` and the edge `run*` drivers add a real runtime on top of this same verb; this is what is left once there is nothing left to add one to.

If you've read the [chaos](./build-a-chaos-test-rig.md) or [delivery](./build-delivery-guarantees.md) tutorials, this is the same one-shot poll shape they used (`block_on_ready`) — not required reading, just a familiar face if you have. Here it's the *entire* runtime, no reactor, no allocator.

## 3. Config becomes build-time constants

The runtime config machinery (`conflaguration::Settings` reading env/files) needs `std` and doesn't exist here. So the two knobs — `RING_SLOTS`, `RING_SLOT_BYTES` — are read from `no-std.toml` by `build.rs`, once, on the host, before the crate compiles, and baked into `pub const`s (`no-std/src/lib.rs:28-31`):

```text
mod config {
    include!(concat!(env!("OUT_DIR"), "/no_std_config.rs"));
}
pub use config::{RING_SLOT_BYTES, RING_SLOTS};
```

`no_std_config.rs` is generated at build time (by `examples/no-std/build.rs`) and contains exactly `pub const RING_SLOTS: usize = 4;` and `pub const RING_SLOT_BYTES: usize = 16;`, baked from `no-std.toml`'s `[ring]` table (`no-std/no-std.toml:10-11`) — the values section 1's snippet used as literals. There is no code path in the compiled binary that re-reads the TOML, checks an env var, or opens a file — **the constants ARE the config**, the no-runtime tier of `conflaguration`. Same recipe `proxima-primitives/build.rs` uses to bake `RETRY_STATUS_CAP` from `proxima-primitives.toml` for `RetryRules`'s no-alloc backing store (`proxima-primitives/build.rs:84`) — `examples/no-std/build.rs`'s own doc comment says so directly: "This mirrors `proxima-primitives/build.rs` ... the same recipe, minimal enough to read end-to-end in one file" (`examples/no-std/build.rs:6-8`).

(That snippet is `text`, not a compiled example: `include!` resolves a path relative to *this* crate's own `OUT_DIR` at compile time, which only exists inside `proxima-example-no-std`'s own build — there is nothing this tutorial's harness could point that `include!` at from outside that crate.)

## 4. The macro-generated pipe holds at the same floor

Foundations §7 taught `#[proxima::piped]`: a plain (non-`async`) function becomes a fieldless struct implementing `UnpinPipe`, with `Clone` derived unconditionally — free, because a fieldless struct has nothing to copy. This crate is the proof that promise holds even here, with zero allocator linked. `ring_capacity` is the identical macro, spelled out by its raw crate path, `#[proxima_macros::piped]`, rather than through the `proxima::piped` re-export — this crate sits too low in the dependency graph to pull in the full `proxima` crate, which pulls in `std` (`no-std/src/lib.rs:89-92`):

```rust
const RING_SLOTS: usize = 4;

#[proxima_macros::piped]
fn ring_capacity() -> Result<usize, core::convert::Infallible> {
    Ok(RING_SLOTS)
}
```

The expansion is a fieldless `struct ring_capacity;` carrying `#[derive(::core::clone::Clone)]` plus an `impl UnpinPipe for ring_capacity`, wrapping the return value in `core::future::ready` — itself `Unpin` unconditionally, so reaching this tier costs no heap allocation either. Cloning the generated struct moves zero bytes, and the clone still runs the real body when called:

```rust
let first = ring_capacity;
let second = first.clone();
let capacity = UnpinPipe::call(&second, ()).await.expect("infallible");
assert_eq!(capacity, RING_SLOTS);
```

Because `ring_capacity` sits in the crate's default (zero-feature) module, section 5's bare `cargo build` is *also* the compile-time proof that this auto-`Clone` costs nothing at the floor — the whole expansion has to compile with no allocator linked, same as `FrameStore`. `no-std/src/lib.rs`'s own `macro_generated_pipe_clones_and_calls_at_the_bare_floor` test (`--features std`, `no-std/src/lib.rs:152-158`) proves it isn't just syntax: it clones the pipe and calls the clone through `UnpinPipe::call`, and gets the real value back.

## 5. The `no_std` proof

The crate is `#![no_std]` by default; `cargo build` with no flags is the proof — if `FrameStore`, `block_on`, `ring_capacity`, or anything they touch reached for `std`, it would not compile (`no-std/README.md:32-36`):

```text
cargo build -p proxima-example-no-std
```

This compiles clean with no flags — verified this session; a `std`-only symbol anywhere on `FrameStore`'s, `block_on`'s, or `ring_capacity`'s path would fail this build. The `std` feature exists only to give `cargo test` a harness and the demo a `println!` (`no-std/README.md:64-66`):

```text
cargo test -p proxima-example-no-std --features std
cargo run  -p proxima-example-no-std --bin no-std-demo --features std
```

The real, unedited transcript from that `cargo run`:

```text
stored "hello" (ring occupancy = 1)
rejected "this-frame-is-way-too-long-for-one-slot": TooLarge
stored "world" (ring occupancy = 2)
```

## What you built

- **the same Pipe** — under `#![no_std]`, writing into a fixed-capacity `RingSink`; no heap, no executor.
- **`block_on` via `Waker::noop`** — a polling loop is the whole runtime.
- **build-time config** — a `build.rs` bakes a TOML into `pub const`s; the constants are the config, resolved before the program exists.
- **the macro tier holds too** — `#[proxima_macros::piped]`'s auto-`Clone`, unconditional since Foundations §7, costs zero bytes even with no allocator to fall back on.

This is the frontier: sans-IO + `no_std` is the price of admission to kernel-bypass (DPDK/SPDK — talking to network or disk hardware directly, skipping the OS) and bare metal — and the `Pipe` you wrote in Foundations is the same one that runs here. `cargo build -p proxima-example-no-std` above proves `#![no_std]` on the host target only (linked as a library, not flashed to a board); this crate also cross-compiles clean to a real Cortex-M target — `cargo build -p proxima-example-no-std --target thumbv7em-none-eabihf` — with the target installed via `rustup target add thumbv7em-none-eabihf`. That claim isn't just asserted: 32-bit ARM targets have no native 64-bit atomics, which used to block any crate that transitively touched `core::sync::atomic::AtomicU64` — including `proxima-primitives/src/pipe/sink_front.rs`, compiled into every build of this crate regardless of whether `FrameStore`'s own path needs it. `portable_atomic` closed that gap (`proxima-primitives/src/pipe/sink_front.rs:30`), and `scripts/thumbv7m-cliff-gate.sh` is the CI gate that proves `proxima-primitives` — this crate's own no_std dependency — still compiles on a real embedded target, not just under a host build (`no-std/README.md:87-94`).
