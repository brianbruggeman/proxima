//! proxima at the edge: one sans-IO `Pipe`, compiled to `wasm32-unknown-unknown`.
//!
//! `primitives::Pipe` is no_std, no-alloc, no `Send` bound — nothing about it
//! assumes an OS, a socket, or a thread. That absence is the whole point: this
//! crate never links tokio, never touches `std::net`, never spawns anything,
//! so there is nothing in it that would refuse to target wasm32.
//!
//! `#![no_std]` unconditionally, no `alloc` either — `primitives::Pipe` never
//! needs either, so nothing here does. `cargo test` still runs on a normal
//! host: `#![no_std]` only opts the crate itself out of the std prelude, the
//! test harness binary links std regardless. The same source, unmodified,
//! is what `cargo build --target wasm32-unknown-unknown` compiles.

#![no_std]

use core::future::Future;

use proxima_primitives::block_on;
use proxima_primitives::pipe::Pipe;

/// `Double` overflowed: doubling `input` would have wrapped past `u64::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overflow;

/// The whole example: one stage, doubling its input. No socket, no clock, no
/// allocator — the smallest `Pipe` that still proves the point.
pub struct Double;

impl Pipe for Double {
    type In = u64;
    type Out = u64;
    type Err = Overflow;

    fn call(&self, input: u64) -> impl Future<Output = Result<u64, Overflow>> {
        async move { input.checked_mul(2).ok_or(Overflow) }
    }
}

/// Run the pipe on a single input. Host-agnostic entry point: native tests
/// call it directly, the wasm export below wraps it.
pub fn run(input: u64) -> Result<u64, Overflow> {
    block_on(Double.call(input))
}

/// wasm host entry point: no wasm-bindgen glue, no linker script — an
/// exported C-ABI function a browser or wasi host calls directly, the same
/// hand-rolled-import shape `proxima-time`'s `driver-wasm` and
/// `proxima-net-wasm` use. A bare `u64` return has no `Result` slot, so
/// overflow is signaled with a sentinel rather than by panicking.
#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn double_at_the_edge(input: u64) -> u64 {
    run(input).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{Overflow, run};

    #[test]
    fn doubles_a_regular_input() {
        assert_eq!(run(21), Ok(42));
    }

    #[test]
    fn overflow_is_reported_not_wrapped() {
        assert_eq!(run(u64::MAX), Err(Overflow));
    }
}
