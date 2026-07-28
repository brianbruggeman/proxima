//! Sans-IO, no-alloc cryptographic substrate — the core of the Centauri
//! family.
//!
//! Everything here compiles with no operating system, no allocator, and no
//! runtime. That is not a portability flourish; it is the security property
//! the crate exists for:
//!
//! - **A secret has exactly one address for its whole life.** A `Vec`
//!   holding key material leaves a copy in every freed page it outgrew, and
//!   zeroizing the live allocation does not reach them. Fixed inline storage
//!   makes zeroization total instead of best-effort.
//! - **An attacker-controlled length cannot allocate.** `with_capacity(n)`
//!   from a wire field is the canonical parser-exhaustion CVE; with no
//!   allocator it is a bounds check that returns an error.
//! - **There is no OOM abort path.** A panic in a handshake is a remote
//!   denial of service. Every bound here is known at compile time, so the
//!   failure mode is [`CentauriError`], not `abort`.
//!
//! None of that closes state-confusion, nonce reuse, or a missing
//! constant-time comparison. Those are the state machines' job and human
//! review's job respectively; no-alloc converts a class of *exploitable*
//! failures into a class of *loud* ones, and claims nothing more.
//!
//! # Capabilities are inputs, not fields
//!
//! A sans-IO state machine may not reach for the world. It cannot read a
//! clock, draw entropy, fetch a key, or touch a socket — each of those is a
//! capability, and a state machine holding one is no longer a function of its
//! inputs. So the two things a handshake genuinely needs from outside are
//! passed in per step:
//!
//! ```text
//! fn step(&mut self, input: &[u8], entropy: Entropy32,
//!         now: Ticks, out: &mut [u8]) -> Result<Poll, CentauriError>
//! ```
//!
//! Time comes from `proxima_clock`, whose `TickCell` lets a caller say what
//! time it is. Entropy comes from [`entropy`], whose [`FixedSequence`] lets a
//! caller say what the next draw is. Both are
//! `Pipe<In = (), Out = _, Err = _>` sources, so neither is a trait, and the
//! composition root — never this crate — decides whether the real one is a
//! hardware register or a syscall.
//!
//! The output buffer is the caller's for the same reason: `out: &mut [u8]`
//! returning a written length, rather than a returned `Vec`. Sizing is the
//! caller's decision because only the caller knows where the bytes are going.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

pub mod entropy;
pub mod error;
pub mod esp;
pub mod handshake;
pub mod hash;

pub use entropy::{CounterDrbg, Entropy32, EntropyCell, FixedSequence};
pub use error::CentauriError;
pub use esp::ChildSa;
pub use handshake::{Handshake, Progress, Role, SessionKeys};
pub use hash::{derive_key, derive_key_into, hash, keyed_hash};

/// A fixed-size `core::fmt::Write` sink, so tests that inspect rendered output
/// run at the no-alloc tier too rather than only where `format!` exists.
#[cfg(test)]
mod test_support {
    use core::fmt::{self, Write};

    pub struct Buffer {
        bytes: [u8; 512],
        len: usize,
    }

    impl Buffer {
        pub const fn new() -> Self {
            Self {
                bytes: [0; 512],
                len: 0,
            }
        }

        pub fn as_str(&self) -> &str {
            core::str::from_utf8(&self.bytes[..self.len]).unwrap_or("<invalid utf-8>")
        }
    }

    impl Write for Buffer {
        fn write_str(&mut self, text: &str) -> fmt::Result {
            let end = self.len + text.len();
            if end > self.bytes.len() {
                return Err(fmt::Error);
            }
            self.bytes[self.len..end].copy_from_slice(text.as_bytes());
            self.len = end;
            Ok(())
        }
    }
}
