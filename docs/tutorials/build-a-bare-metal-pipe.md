# Build a bare-metal (no_std) pipe

**Prerequisites:** [Foundations](./00-foundations.md) — the base sans-IO `Pipe` (the **transform** role, not the served one).
**You will:** run the same `Pipe` trait on bare metal — no heap, no executor, no OS — and turn config into build-time constants baked before the program exists.
**New concepts (in order):** the sans-IO `Pipe` under `#![no_std]` (a fixed-capacity `RingSink`, no `Box`/`Vec`/alloc) · `block_on` via `Waker::noop` (a polling loop is the whole runtime) · build-time config constants (`build.rs` bakes a TOML into `pub const`s).
**Answer key:** [`examples/no-std/src/lib.rs`](../../examples/no-std/src/lib.rs) — `cargo build -p proxima-example-no-std` (no flags = the `no_std` proof).

The example frames it, verbatim from its own module doc-comment (`no-std/src/lib.rs:1-2`): *"proxima's sans-IO core — a `Pipe` — compiling with no std and no runtime."* Its README puts the same point another way (`no-std/README.md:10-12`): *"Every other rung in this curriculum runs on a host with an OS, a heap, and a runtime under it. This one asks what's left of proxima once all three are gone."*

## 1. The same Pipe, on a smaller planet

The crate root carries `#![cfg_attr(not(feature = "std"), no_std)]` (`no-std/src/lib.rs:15`) — genuinely `#![no_std]` unless the `std` feature is turned on. `FrameStore` is a plain `Pipe` — the exact trait Foundations taught — writing a borrowed frame into a `RingSink`, a fixed-capacity array sized by two const generics. No `Box`, no `Vec`, no allocator. Copied verbatim from `no-std/src/lib.rs:44-73`:

```rust
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

`In = &'static [u8]` is a borrowed frame — a real caller on bare metal borrows a DMA buffer (memory shared directly with hardware) or a stack arena rather than owning a `'static` slice; the demo uses `'static` byte-string literals to keep the example self-contained. `Out = usize` is the ring's occupancy after the write. `Err = StoreError` is a two-variant enum, `TooLarge | Full` (`no-std/src/lib.rs:30-38`) — no partial writes, no silent drops.

`self.ring.borrow_mut()` reaches through a `RefCell` — a container that lets you mutate what's inside through a shared `&self` reference, checking at runtime (instead of compile time) that nothing else is touching the ring at the same moment. `ring.accept(frame)` then returns a `ControlFlow`, Rust's two-way "keep going or stop" signal: `Continue(())` means the frame was stored and there's still room, so `call` reports the new occupancy; `Break(())` means the ring is already full, so `call` maps it to `StoreError::Full`.

The same `trait Pipe { type In; type Out; type Err; fn call(...) -> impl Future<...>; }` as Foundations — only the tier is smaller. The hot path allocates nothing; state is a fixed array behind a `RefCell`. `RingSink` itself is a public primitive, `proxima_primitives::pipe::RingSink` — re-exported from `proxima-primitives/src/pipe/drain_sink.rs:47` at `proxima-primitives/src/pipe/mod.rs:184`.

## 2. `block_on` is a polling loop

With no executor, `block_on` drives the pipe's future to completion with a `Waker::noop()` (stable since Rust 1.85) — polling in a loop **is** the runtime (`no-std/src/lib.rs:80-88`):

```rust
pub fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
    }
}
```

You don't need to trace every token in that loop — `pin!`, `Context::from_waker`, `.poll(...)`, and `Poll::Ready` are just the mechanics of asking a future "are you done yet?". The loop asks, over and over, until the answer is yes; that polling loop **is** the runtime — the same black box Foundations waved off, just cracked open once so you can see there's no magic inside, only asking-and-checking.

If you've read the [chaos](./build-a-chaos-test-rig.md) or [delivery](./build-delivery-guarantees.md) tutorials, this is the same one-shot poll shape they used (`block_on_ready`) — not required reading, just a familiar face if you have. Here it's the *entire* runtime, no reactor, no allocator.

## 3. Config becomes build-time constants

The runtime config machinery (`conflaguration::Settings` reading env/files) needs `std` and doesn't exist here. So the two knobs — `RING_SLOTS`, `RING_SLOT_BYTES` — are read from `no-std.toml` by `build.rs`, once, on the host, before the crate compiles, and baked into `pub const`s (`no-std/src/lib.rs:25-28`):

```rust
mod config {
    include!(concat!(env!("OUT_DIR"), "/no_std_config.rs"));
}
pub use config::{RING_SLOT_BYTES, RING_SLOTS};
```

`no_std_config.rs` is generated at build time (by `examples/no-std/build.rs`) and contains exactly `pub const RING_SLOTS: usize = ...;` and `pub const RING_SLOT_BYTES: usize = ...;`, baked from `no-std.toml`. There is no code path in the compiled binary that re-reads the TOML, checks an env var, or opens a file — **the constants ARE the config**, the no-runtime tier of `conflaguration`. Same recipe `proxima-primitives/build.rs` uses to bake `RETRY_STATUS_CAP` from `proxima-primitives.toml` for `RetryRules`'s no-alloc backing store — `examples/no-std/build.rs`'s own doc comment says so directly: "This mirrors `proxima-primitives/build.rs` ... the same recipe, minimal enough to read end-to-end in one file."

## 4. The `no_std` proof

The crate is `#![no_std]` by default; `cargo build` with no flags is the proof — if `FrameStore`, `block_on`, or anything they touch reached for `std`, it would not compile (`no-std/README.md:37-45`):

```
cargo build -p proxima-example-no-std
```

This compiles clean with no flags — verified this session; a `std`-only symbol anywhere on `FrameStore`'s or `block_on`'s path would fail this build. The `std` feature exists only to give `cargo test` a harness and the demo a `println!` (`no-std/README.md:47-52`):

```
cargo test -p proxima-example-no-std --features std
cargo run  -p proxima-example-no-std --bin no-std-demo --features std
```

The real, unedited transcript from that `cargo run`:

```
stored "hello" (ring occupancy = 1)
rejected "this-frame-is-way-too-long-for-one-slot": TooLarge
stored "world" (ring occupancy = 2)
```

## What you built

- **the same Pipe** — under `#![no_std]`, writing into a fixed-capacity `RingSink`; no heap, no executor.
- **`block_on` via `Waker::noop`** — a polling loop is the whole runtime.
- **build-time config** — a `build.rs` bakes a TOML into `pub const`s; the constants are the config, resolved before the program exists.

This is the frontier: sans-IO + `no_std` is the price of admission to kernel-bypass (DPDK/SPDK — talking to network or disk hardware directly, skipping the OS) and bare metal — and the `Pipe` you wrote in Foundations is the same one that runs here. `cargo build -p proxima-example-no-std` above proves `#![no_std]` on the host target only (linked as a library, not flashed to a board); this crate also cross-compiles clean to a real Cortex-M target — `cargo build -p proxima-example-no-std --target thumbv7em-none-eabihf` — with the target installed via `rustup target add thumbv7em-none-eabihf`.
