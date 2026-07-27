use core::convert::Infallible;
use core::future::Future;

use proxima_primitives::pipe::primitives::Pipe;

use crate::seq_words::SeqU64;
use crate::ticks::Ticks;

/// A shared, coarsely-updated tick count — the common per-core/per-shard
/// "what time is it, roughly" cell, distinct from a per-read hardware
/// register.
///
/// Some deployments do not want every caller reading a hardware counter
/// directly (a syscall-backed clock, a register behind a slow bus, a
/// counter shared across cores where every core hitting the same
/// cache line would thrash it). The usual answer is one owner reads the
/// real source occasionally and writes the result here; every other
/// reader reads this cell instead — cheap, lock-free, and no less
/// accurate than the owner's update cadence.
///
/// `TickCell` reads exactly like a hardware counter: `impl Pipe for
/// &TickCell` gives it the same `In = (), Out = Ticks, Err = Infallible`
/// source shape as [`crate::coarse`]'s hardware-mmap counterpart (see
/// `tests/hardware_mmap_seam.rs`) — a caller composing a pipeline does not
/// know or care whether a given tick source is a raw register or this
/// cell.
///
/// Deliberately NOT a process-wide `static` here: how a cell is reached
/// (a field on a per-core context, threaded through a runtime handle, a
/// `static` a caller chooses to declare) is a runtime-wiring decision
/// this `no_std` leaf crate does not make on the caller's behalf.
///
/// # Contract
///
/// Single writer: [`TickCell::set`] is for the one owner that reads the
/// real source and republishes it. Concurrent `set` calls from more than
/// one caller are not synchronized against each other (see
/// [`crate::seq_words`]'s seqlock contract). Any number of readers may
/// call [`TickCell::get`] (or drive it as a `Pipe`) concurrently,
/// lock-free.
pub struct TickCell {
    ticks: SeqU64,
}

impl TickCell {
    /// Construct a cell starting at `initial`.
    #[must_use]
    pub fn new(initial: Ticks) -> Self {
        Self {
            ticks: SeqU64::new(initial.as_raw()),
        }
    }

    /// Republish the current tick count. See the struct doc's
    /// single-writer contract.
    pub fn set(&self, ticks: Ticks) {
        self.ticks.store(ticks.as_raw());
    }

    /// Read the current tick count.
    #[must_use]
    pub fn get(&self) -> Ticks {
        Ticks::from_raw(self.ticks.load())
    }
}

impl Pipe for &TickCell {
    type In = ();
    type Out = Ticks;
    type Err = Infallible;

    fn call(&self, (): ()) -> impl Future<Output = Result<Ticks, Infallible>> {
        let ticks = self.get();
        async move { Ok(ticks) }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::TickCell;
    use crate::ticks::Ticks;
    use proxima_primitives::pipe::primitives::Pipe;

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
    fn set_then_get_round_trips() {
        let cell = TickCell::new(Ticks::from_raw(1_000));

        cell.set(Ticks::from_raw(2_000));

        assert_eq!(cell.get(), Ticks::from_raw(2_000));
    }

    #[test]
    fn reads_as_a_source_pipe() {
        let cell = TickCell::new(Ticks::from_raw(24_000_000));

        let read = block_on(Pipe::call(&&cell, ())).expect("cell reads never fail");

        assert_eq!(read, Ticks::from_raw(24_000_000));
    }
}
