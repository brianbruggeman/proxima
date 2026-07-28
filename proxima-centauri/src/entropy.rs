//! Entropy source polymorphism, expressed as [`proxima_primitives`] pipes
//! instead of a bespoke RNG trait — the same principle
//! [`proxima_clock`](https://docs.rs/proxima-clock) applies to time, applied
//! to randomness.
//!
//! # Why this is a pipe, not a trait
//!
//! An entropy source is a source: it takes nothing and yields bytes. That is
//! `Pipe<In = (), Out = Entropy32, Err = CentauriError>`, the source form
//! from `Pipe`'s own "four forms" table. There is therefore no
//! `EntropySource` / `CryptoRng` / `Rng` capability trait in this crate —
//! the shape was checked against the binary rule ("can this be expressed
//! with a pipe? If yes, it does not get a type") by writing the composition,
//! not by arguing about it. Anything with that shape is an entropy source;
//! there is nothing else to implement.
//!
//! This is what makes entropy **fakeable**, which is the whole point of
//! giving it time's shape. A state machine that draws from an injected
//! source cannot tell a hardware TRNG from [`CounterDrbg`] from
//! [`FixedSequence`], so:
//!
//! - a spec worked-example test injects the RFC's ephemeral key and nonce
//!   and asserts the wire bytes match — impossible if the state machine
//!   calls `getrandom` internally;
//! - a regression run seeds [`CounterDrbg`] and replays byte-for-byte;
//! - a nonce-reuse test scripts the *same* value twice on purpose and
//!   asserts the state machine rejects it.
//!
//! No source in this crate reads the operating system or the hardware.
//! `getrandom`, `RDRAND`, and a TRNG register are all edge concerns: the
//! core declares the shape it needs and the composition root supplies it.
//!
//! ```
//! use proxima_centauri::{CentauriError, CounterDrbg, Entropy32, FixedSequence};
//! use proxima_primitives::block_on;
//! use proxima_primitives::pipe::Pipe;
//!
//! // a state machine draws from whatever source it was handed, and cannot
//! // tell which one it is.
//! async fn draw_two<S>(source: S) -> Result<([u8; 32], [u8; 32]), CentauriError>
//! where
//!     S: Pipe<In = (), Out = Entropy32, Err = CentauriError> + Copy,
//! {
//!     let first = source.call(()).await?;
//!     let second = source.call(()).await?;
//!     Ok((*first.expose(), *second.expose()))
//! }
//!
//! // scripted: the test states what the draws are, the way it states the time.
//! let script = [[1u8; 32], [2u8; 32]];
//! let scripted = FixedSequence::new(&script);
//! assert_eq!(block_on(draw_two(&scripted))?, ([1u8; 32], [2u8; 32]));
//!
//! // seeded: reproducible across instances, and never repeats a draw.
//! let seeded = CounterDrbg::new([7u8; 32]);
//! let (first, second) = block_on(draw_two(&seeded))?;
//! assert_ne!(first, second);
//! assert_eq!(block_on(draw_two(&CounterDrbg::new([7u8; 32])))?, (first, second));
//! # Ok::<(), CentauriError>(())
//! ```
//!
//! # The cell form, with one property inverted
//!
//! [`EntropyCell`] is the direct analogue of
//! [`proxima_clock`'s `TickCell`](https://docs.rs/proxima-clock): an owner
//! pushes a value in, holders read it out as a `Pipe`, and how the cell is
//! reached is the caller's wiring decision rather than a `static` this crate
//! declares. That form is what neither other source provides —
//! [`FixedSequence`] fixes its script at construction and [`CounterDrbg`]
//! generates its own, so neither lets a producer running at a *different
//! time* than the draw supply the bytes. A TRNG that takes microseconds to
//! produce 32 bytes cannot be read inline from a state-machine step; an
//! interrupt or a dedicated task reads the peripheral and fills the cell,
//! and the handshake draws without ever touching hardware.
//!
//! Exactly one property inverts, and it is the one that matters.
//! `TickCell` is a *broadcast* cell: many readers get the same answer, which
//! is correct because time is shared and re-reading it is meaningful. Two
//! entropy draws yielding the same bytes is nonce reuse — keystream reuse in
//! a stream AEAD, with Poly1305 key recovery behind it. So [`EntropyCell`] is
//! **take-once**: a draw claims the value and empties the cell, and a second
//! draw without an intervening [`EntropyCell::set`] fails rather than
//! repeating. That is enforced by a compare-exchange, not documented as a
//! contract — only one caller can win the claim.

use core::future::Future;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::sized::{EntropyCounter, EntropyCounterValue};

use proxima_primitives::pipe::Pipe;

use crate::error::CentauriError;
use crate::hash::keyed_hash;

/// 32 bytes of entropy, moved rather than copied.
///
/// Deliberately neither `Copy` nor `Clone`. A state machine that takes
/// `Entropy32` by value consumes it, so passing the same draw to two
/// transitions does not compile — the nonce-reuse footgun that injecting
/// entropy would otherwise introduce becomes a type error instead of a
/// review finding.
///
/// The bytes never leave the token: [`Entropy32::expose`] hands out a
/// reference, and [`Drop`] zeroes the array. This is the property no-alloc
/// buys — one address for the secret's whole life, so zeroization is total
/// rather than best-effort.
///
/// Also deliberately not `PartialEq`: a derived comparison on secret material
/// short-circuits on the first differing byte, which is a timing oracle. Code
/// that must compare secrets needs a constant-time comparison chosen on
/// purpose, not `==` reached for by habit.
#[derive(Debug)]
pub struct Entropy32([u8; 32]);

impl Entropy32 {
    /// Wrap raw bytes. Callers inside this crate construct from a source;
    /// tests construct scripted values directly.
    #[must_use]
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Borrow the bytes. Named to make the read visible at the call site.
    #[must_use]
    pub const fn expose(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Drop for Entropy32 {
    fn drop(&mut self) {
        self.0 = [0u8; 32];
        // the write above is dead-store-eliminable: nothing reads self.0
        // afterwards. black_box forces the optimiser to treat it as observed,
        // which is the zeroize crate's job done with core and no unsafe.
        let _ = core::hint::black_box(&self.0);
    }
}

/// A counter-mode DRBG over BLAKE3's keyed hash.
///
/// Seed once from an expensive true source, then draw cheaply: each call
/// derives `keyed_hash(seed, counter)` and advances the counter, so every
/// draw is distinct by construction rather than by hoping the caller
/// remembered not to reuse one. `keyed_hash` is a PRF, which makes this the
/// standard PRF-in-counter-mode construction.
///
/// Lock-free and `&self`: the counter is a single atomic, so the source
/// composes as a `Pipe` from any number of holders without a mutex. Its width
/// is [`crate::sized::ENTROPY_COUNTER_BITS`], resolved at build time against
/// the target's advertised atomics rather than assumed in source — Cortex-M3
/// has no 64-bit atomic instructions, so `AtomicU64` does not compile there.
///
/// # Forward secrecy
///
/// Not provided: the seed is fixed for the DRBG's life, so disclosing it
/// discloses every past and future draw. That is the right trade for a
/// reproducible test source and for a session-scoped draw pool seeded from a
/// protected source; it is not a substitute for a ratcheting DRBG in a
/// long-lived process holding one instance across sessions.
#[derive(Debug)]
pub struct CounterDrbg {
    seed: [u8; 32],
    counter: EntropyCounter,
}

impl CounterDrbg {
    /// Seed the DRBG. Two instances with the same seed produce the same
    /// sequence — that is the property regression and parity tests want, and
    /// the reason a production seed must come from a real source.
    #[must_use]
    pub const fn new(seed: [u8; 32]) -> Self {
        Self {
            seed,
            counter: EntropyCounter::new(0),
        }
    }

    /// Draw the next 32 bytes, advancing the counter.
    ///
    /// # Errors
    ///
    /// [`CentauriError::EntropyExhausted`] once the counter cannot advance.
    pub fn draw(&self) -> Result<Entropy32, CentauriError> {
        let counter = self
            .counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(
                |current: EntropyCounterValue| CentauriError::EntropyExhausted {
                    drawn: current as usize,
                    available: current as usize,
                },
            )?;

        Ok(Entropy32::new(keyed_hash(
            &self.seed,
            &counter.to_le_bytes(),
        )))
    }
}

impl Pipe for &CounterDrbg {
    type In = ();
    type Out = Entropy32;
    type Err = CentauriError;

    fn call(&self, (): ()) -> impl Future<Output = Result<Entropy32, CentauriError>> {
        let drawn = self.draw();
        async move { drawn }
    }
}

/// A finite, scripted entropy source — the test seam, and the direct
/// analogue of setting the clock.
///
/// Where a test tells `TickCell` what time it is, this tells a state machine
/// what its next draw is: RFC test vectors, an all-zero ephemeral key, or
/// the same value twice to prove reuse is rejected. Borrows its values, so
/// it stays no-alloc.
#[derive(Debug)]
pub struct FixedSequence<'values> {
    values: &'values [[u8; 32]],
    index: AtomicUsize,
}

impl<'values> FixedSequence<'values> {
    /// Script a sequence of draws, consumed in order.
    #[must_use]
    pub const fn new(values: &'values [[u8; 32]]) -> Self {
        Self {
            values,
            index: AtomicUsize::new(0),
        }
    }

    /// How many draws remain.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.values
            .len()
            .saturating_sub(self.index.load(Ordering::Relaxed))
    }

    /// Take the next scripted value.
    ///
    /// # Errors
    ///
    /// [`CentauriError::EntropyExhausted`] when the state machine drew more
    /// times than the test scripted — which names the mismatch rather than
    /// silently repeating the last value.
    pub fn draw(&self) -> Result<Entropy32, CentauriError> {
        let index = self.index.fetch_add(1, Ordering::Relaxed);

        self.values
            .get(index)
            .map(|bytes| Entropy32::new(*bytes))
            .ok_or(CentauriError::EntropyExhausted {
                drawn: index.saturating_add(1),
                available: self.values.len(),
            })
    }
}

impl Pipe for &FixedSequence<'_> {
    type In = ();
    type Out = Entropy32;
    type Err = CentauriError;

    fn call(&self, (): ()) -> impl Future<Output = Result<Entropy32, CentauriError>> {
        let drawn = self.draw();
        async move { drawn }
    }
}

/// Cell is empty and may be filled.
const CELL_EMPTY: u32 = 0;
/// A producer is mid-write; the words must not be read.
const CELL_FILLING: u32 = 1;
/// A value is present and may be claimed.
const CELL_FULL: u32 = 2;
/// A consumer won the claim and is mid-read; the words must not be written.
const CELL_CLAIMING: u32 = 3;

/// A settable, **take-once** entropy slot — the cell form, and the seam for
/// a source that produces on a different schedule than the draw.
///
/// [`set`](EntropyCell::set) fills it, [`draw`](EntropyCell::draw) claims and
/// empties it. In production that is a TRNG interrupt or a dedicated task
/// filling the cell while a handshake draws from it without touching the
/// peripheral. In a test it is stronger than [`FixedSequence`]: the next draw
/// can be decided *after* observing what the state machine did with the last
/// one, rather than scripted up front.
///
/// # Take-once, not broadcast
///
/// This is the one place the analogy to `TickCell` breaks, deliberately.
/// A second [`draw`](EntropyCell::draw) with no intervening
/// [`set`](EntropyCell::set) returns
/// [`CentauriError::EntropyUnavailable`] — it does not repeat the value,
/// because a repeated draw is nonce reuse. The claim is a
/// `compare_exchange`, so exactly one caller can win it even with several
/// drawing concurrently; the property is enforced rather than promised.
///
/// # Portability
///
/// The claim needs 32-bit compare-exchange, which is the repo's tier-3 floor
/// (`thumbv7m-none-eabi`, Cortex-M3 and up). A CAS-free target such as
/// Cortex-M0 would need a different construction — not a
/// `critical-section`-style global assumption, which `proxima_clock` already
/// declined to make on a caller's behalf.
#[derive(Debug)]
pub struct EntropyCell {
    words: [AtomicU32; 8],
    state: AtomicU32,
}

impl Default for EntropyCell {
    fn default() -> Self {
        Self::new()
    }
}

impl EntropyCell {
    /// An empty cell.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            words: [const { AtomicU32::new(0) }; 8],
            state: AtomicU32::new(CELL_EMPTY),
        }
    }

    /// Whether a draw would currently succeed.
    ///
    /// A producer polls this to decide whether to generate; it is advisory,
    /// since a concurrent draw or fill can change the answer immediately
    /// after it returns.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.state.load(Ordering::Acquire) == CELL_FULL
    }

    /// Fill the cell with the next draw.
    ///
    /// # Errors
    ///
    /// [`CentauriError::EntropyUnavailable`] if the cell already holds an
    /// unclaimed value or another caller is mid-operation. Filling does not
    /// overwrite: silently replacing an unclaimed value would hide a
    /// producer/consumer rate mismatch, and the caller is better told.
    pub fn set(&self, bytes: [u8; 32]) -> Result<(), CentauriError> {
        self.state
            .compare_exchange(
                CELL_EMPTY,
                CELL_FILLING,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .map_err(|_| CentauriError::EntropyUnavailable("cell not empty"))?;

        for (word, chunk) in self.words.iter().zip(bytes.chunks_exact(4)) {
            let mut quad = [0u8; 4];
            quad.copy_from_slice(chunk);
            word.store(u32::from_le_bytes(quad), Ordering::Relaxed);
        }

        self.state.store(CELL_FULL, Ordering::Release);

        Ok(())
    }

    /// Claim the cell's value, emptying it.
    ///
    /// # Errors
    ///
    /// [`CentauriError::EntropyUnavailable`] if the cell is empty or another
    /// caller won the claim. This is the take-once property: it never repeats
    /// a value that has already been drawn.
    pub fn draw(&self) -> Result<Entropy32, CentauriError> {
        self.state
            .compare_exchange(
                CELL_FULL,
                CELL_CLAIMING,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .map_err(|_| CentauriError::EntropyUnavailable("cell empty"))?;

        let mut bytes = [0u8; 32];
        for (word, chunk) in self.words.iter().zip(bytes.chunks_exact_mut(4)) {
            chunk.copy_from_slice(&word.load(Ordering::Relaxed).to_le_bytes());
        }

        self.state.store(CELL_EMPTY, Ordering::Release);

        Ok(Entropy32::new(bytes))
    }
}

impl Pipe for &EntropyCell {
    type In = ();
    type Out = Entropy32;
    type Err = CentauriError;

    fn call(&self, (): ()) -> impl Future<Output = Result<Entropy32, CentauriError>> {
        let drawn = self.draw();
        async move { drawn }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proxima_primitives::pipe::Pipe;

    use super::{CounterDrbg, Entropy32, EntropyCell, FixedSequence};
    use crate::error::CentauriError;

    fn block_on<Fut: core::future::Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    #[test]
    fn drbg_draws_differ() {
        let source = CounterDrbg::new([0u8; 32]);

        let first = source.draw().expect("first draw within counter range");
        let second = source.draw().expect("second draw within counter range");

        assert_ne!(
            first.expose(),
            second.expose(),
            "a repeated draw is nonce reuse"
        );
    }

    #[test]
    fn drbg_is_reproducible_across_instances() {
        let seed = [42u8; 32];
        let recorded = CounterDrbg::new(seed);
        let replayed = CounterDrbg::new(seed);

        let recorded_draws = [
            recorded.draw().unwrap(),
            recorded.draw().unwrap(),
            recorded.draw().unwrap(),
        ];
        let replayed_draws = [
            replayed.draw().unwrap(),
            replayed.draw().unwrap(),
            replayed.draw().unwrap(),
        ];

        for (left, right) in recorded_draws.iter().zip(replayed_draws.iter()) {
            assert_eq!(
                left.expose(),
                right.expose(),
                "same seed must replay byte-for-byte"
            );
        }
    }

    #[test]
    fn drbg_separates_seeds() {
        let first = CounterDrbg::new([1u8; 32]);
        let second = CounterDrbg::new([2u8; 32]);

        assert_ne!(
            first.draw().unwrap().expose(),
            second.draw().unwrap().expose()
        );
    }

    #[test]
    fn drbg_drives_as_a_pipe() {
        let source = CounterDrbg::new([7u8; 32]);

        let piped = block_on((&source).call(())).expect("pipe draw succeeds");
        let direct = CounterDrbg::new([7u8; 32])
            .draw()
            .expect("direct draw succeeds");

        assert_eq!(piped.expose(), direct.expose(), "the pipe form is the draw");
    }

    #[test]
    fn fixed_sequence_replays_the_script_in_order() {
        let scripted = [[1u8; 32], [2u8; 32]];
        let source = FixedSequence::new(&scripted);

        assert_eq!(source.draw().unwrap().expose(), &[1u8; 32]);
        assert_eq!(source.draw().unwrap().expose(), &[2u8; 32]);
    }

    #[test]
    fn fixed_sequence_can_script_deliberate_reuse() {
        let repeated = [[9u8; 32], [9u8; 32]];
        let source = FixedSequence::new(&repeated);

        assert_eq!(
            source.draw().unwrap().expose(),
            source.draw().unwrap().expose(),
            "a test must be able to force reuse in order to assert it is rejected"
        );
    }

    #[test]
    fn fixed_sequence_names_the_shortfall_when_overdrawn() {
        let scripted = [[1u8; 32]];
        let source = FixedSequence::new(&scripted);

        let _ = source.draw().expect("first draw is scripted");

        assert_eq!(
            source.draw().err(),
            Some(CentauriError::EntropyExhausted {
                drawn: 2,
                available: 1
            })
        );
    }

    #[test]
    fn fixed_sequence_reports_remaining() {
        let scripted = [[1u8; 32], [2u8; 32]];
        let source = FixedSequence::new(&scripted);

        assert_eq!(source.remaining(), 2);
        let _ = source.draw().unwrap();
        assert_eq!(source.remaining(), 1);
    }

    #[test]
    fn fixed_sequence_drives_as_a_pipe() {
        let scripted = [[5u8; 32]];
        let source = FixedSequence::new(&scripted);

        let drawn = block_on((&source).call(())).expect("pipe draw succeeds");

        assert_eq!(drawn.expose(), &[5u8; 32]);
    }

    #[test]
    fn cell_round_trips_a_pushed_value() {
        let cell = EntropyCell::new();

        cell.set([3u8; 32]).expect("empty cell accepts a fill");

        assert_eq!(cell.draw().unwrap().expose(), &[3u8; 32]);
    }

    #[test]
    fn cell_draw_is_take_once() {
        let cell = EntropyCell::new();
        cell.set([4u8; 32]).expect("empty cell accepts a fill");

        let _claimed = cell.draw().expect("first draw claims the value");

        assert_eq!(
            cell.draw().err(),
            Some(CentauriError::EntropyUnavailable("cell empty")),
            "a second draw must not repeat the value: that is nonce reuse"
        );
    }

    #[test]
    fn cell_draw_on_empty_reports_unavailable() {
        let cell = EntropyCell::new();

        assert_eq!(
            cell.draw().err(),
            Some(CentauriError::EntropyUnavailable("cell empty"))
        );
    }

    #[test]
    fn cell_refuses_to_overwrite_an_unclaimed_value() {
        let cell = EntropyCell::new();
        cell.set([5u8; 32]).expect("empty cell accepts a fill");

        assert_eq!(
            cell.set([6u8; 32]).err(),
            Some(CentauriError::EntropyUnavailable("cell not empty")),
            "overwriting would hide a producer/consumer rate mismatch"
        );
        assert_eq!(
            cell.draw().unwrap().expose(),
            &[5u8; 32],
            "the first value survives"
        );
    }

    #[test]
    fn cell_cycles_between_fills_and_draws() {
        let cell = EntropyCell::new();

        for marker in 0..4u8 {
            cell.set([marker; 32])
                .expect("cell is empty after each draw");
            assert_eq!(cell.draw().unwrap().expose(), &[marker; 32]);
        }
    }

    #[test]
    fn cell_reports_fullness() {
        let cell = EntropyCell::new();
        assert!(!cell.is_full());

        cell.set([1u8; 32]).unwrap();
        assert!(cell.is_full());

        let _ = cell.draw().unwrap();
        assert!(!cell.is_full());
    }

    #[test]
    fn cell_drives_as_a_pipe() {
        let cell = EntropyCell::new();
        cell.set([8u8; 32]).unwrap();

        let drawn = block_on((&cell).call(())).expect("pipe draw claims the value");

        assert_eq!(drawn.expose(), &[8u8; 32]);
        assert!(
            block_on((&cell).call(())).is_err(),
            "the pipe form is take-once too"
        );
    }

    #[test]
    fn entropy_is_not_copy() {
        // a compile-time property, asserted by construction: this function
        // would not compile if Entropy32 were Copy, because `expose` would
        // still be callable after the move below.
        let token = Entropy32::new([1u8; 32]);
        let moved = token;

        assert_eq!(moved.expose(), &[1u8; 32]);
    }
}
