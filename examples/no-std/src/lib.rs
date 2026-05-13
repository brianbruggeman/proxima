//! proxima's sans-IO core — a [`Pipe`](proxima_primitives::pipe::Pipe) — compiling
//! with **no std and no runtime**. `RING_SLOTS` / `RING_SLOT_BYTES` below are
//! not read at runtime: `build.rs` bakes them from `no-std.toml` into
//! `pub const`s once, at compile time. This is the no-runtime tier of
//! `conflaguration` — there is no config system to run here, because there is
//! no runtime to run it in.
//!
//! The default build (`cargo build`, no features) compiles this crate
//! genuinely `#![no_std]` with no allocator: no heap, no OS, no executor.
//! `FrameStore` is a plain [`Pipe`]; the `RingSink` it writes into is a fixed
//! array sized by the two baked constants. `ring_capacity` is the same
//! floor claim for `#[proxima::pipe]`'s codegen instead of a hand-written
//! `impl Pipe`: the generated struct + its unconditional `Clone` derive +
//! its `UnpinPipe` impl all have to compile with zero allocator too. The
//! `std` feature exists only to give `cargo test` a libtest harness and the
//! `no-std-demo` binary a `println!` — the pipe logic in this file is
//! identical either way.

#![cfg_attr(not(feature = "std"), no_std)]

use core::cell::RefCell;
use core::future::Future;
use core::ops::ControlFlow;

pub use proxima_primitives::block_on;
use proxima_primitives::pipe::{DrainSink, Pipe, RingSink};

mod config {
    include!(concat!(env!("OUT_DIR"), "/no_std_config.rs"));
}
pub use config::{RING_SLOT_BYTES, RING_SLOTS};

/// The only two ways [`FrameStore::call`] can fail — no partial writes, no
/// silent drops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    /// `frame.len()` exceeds `RING_SLOT_BYTES` — the arena slot can't hold it.
    TooLarge,
    /// The ring is at `RING_SLOTS` capacity — the caller must drain first.
    Full,
}

/// The one pipe: writes a borrowed frame into a fixed-capacity ring sized by
/// the build-time constants. `In = &'static [u8]` keeps the demo to literal
/// byte strings; a real bare-metal caller borrows from a DMA buffer or a
/// stack arena instead. `Out` is the ring's occupancy after the write.
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

/// `#[proxima::pipe]`'s auto-`Clone`, proven at the floor: this fn expands to
/// a fieldless `struct ring_capacity;` carrying `#[derive(::core::clone::Clone)]`
/// plus an `impl UnpinPipe for ring_capacity`, and the whole thing has to
/// compile under this crate's default (zero-feature, genuinely `#![no_std]`,
/// no allocator linked) build — see the crate README's "Build (the no_std
/// proof)" section. Cloning the generated struct copies zero bytes: no
/// heap, no `alloc` crate, nothing to move because the type has no fields.
/// A plain `fn` (not `async fn`)
/// lands on `UnpinPipe`, the tier `#[proxima::pipe]` reaches by wrapping the
/// call in `core::future::ready` — itself `Unpin` unconditionally, so no
/// heap allocation is needed to reach that tier either.
#[proxima_macros::pipe]
fn ring_capacity() -> Result<usize, core::convert::Infallible> {
    Ok(RING_SLOTS)
}

/// std-only demo wrapper: drives the same `FrameStore` pipe the `#![no_std]`
/// build compiles and prints each outcome. A bare-metal caller would swap
/// `println!` for a UART write; the pipe call underneath is untouched.
#[cfg(feature = "std")]
pub fn run_demo() {
    let store = FrameStore::default();
    let too_long: &'static [u8] = b"this-frame-is-way-too-long-for-one-slot";
    let frames: [&'static [u8]; 3] = [b"hello", too_long, b"world"];

    for frame in frames {
        let text = core::str::from_utf8(frame).unwrap_or("<binary frame>");
        match block_on(Pipe::call(&store, frame)) {
            Ok(occupancy) => std::println!("stored {text:?} (ring occupancy = {occupancy})"),
            Err(err) => std::println!("rejected {text:?}: {err:?}"),
        }
    }
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proxima_primitives::pipe::UnpinPipe;

    use super::{FrameStore, Pipe, RING_SLOT_BYTES, RING_SLOTS, StoreError, block_on, ring_capacity};

    #[test]
    fn stores_a_frame_that_fits() {
        let store = FrameStore::default();
        let occupancy = block_on(Pipe::call(&store, b"hello".as_slice())).expect("fits a slot");
        assert_eq!(occupancy, 1);
    }

    #[test]
    fn rejects_a_frame_larger_than_a_slot() {
        let store = FrameStore::default();
        let oversized: &'static [u8] = b"this-frame-is-way-too-long-for-one-slot";
        assert!(
            oversized.len() > RING_SLOT_BYTES,
            "fixture must exceed the baked cap"
        );

        let err = block_on(Pipe::call(&store, oversized)).expect_err("oversized frame rejected");
        assert_eq!(err, StoreError::TooLarge);
    }

    #[test]
    fn rejects_once_the_ring_is_full() {
        let store = FrameStore::default();
        for _ in 0..RING_SLOTS {
            block_on(Pipe::call(&store, b"x".as_slice())).expect("room while under capacity");
        }

        let err = block_on(Pipe::call(&store, b"y".as_slice())).expect_err("ring is at capacity");
        assert_eq!(err, StoreError::Full);
    }

    #[test]
    fn macro_generated_pipe_clones_and_calls_at_the_bare_floor() {
        let first = ring_capacity;
        let second = first.clone();
        let capacity = block_on(UnpinPipe::call(&second, ())).expect("infallible");
        assert_eq!(capacity, RING_SLOTS);
    }

    #[test]
    fn baked_constants_come_from_no_std_toml() {
        assert_eq!(RING_SLOTS, 4);
        assert_eq!(RING_SLOT_BYTES, 16);
    }
}
