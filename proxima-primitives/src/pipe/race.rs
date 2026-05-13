//! `Race<Sink, Policy>` — concurrent first-wins dispatch over N [`SendPipe`]s.
//!
//! Where [`crate::pipe::FanOut`] drives sinks *sequentially* and awaits *all* of them,
//! `Race` drives N sinks *concurrently* and returns the first `Ok` response,
//! dropping the remaining in-flight futures. The dispatch is `futures::future::
//! select_all` — a safe, no_std+alloc combinator (no hand-rolled pin state
//! machine, no `unsafe`).
//!
//! # Cancellation contract
//!
//! Dropping a losing future IS the cancellation. `Race` therefore requires
//! `Sink: DropSafe` — a sink whose in-flight future is sound to drop at any
//! await point (detached blocking work, pure computation, datagram-atomic
//! transport). A streaming sink that would leave a torn message on drop must
//! NOT be raced; the marker is the compile-time gate. The sequential
//! [`crate::pipe::FanOut`] awaits every branch and needs no such bound.
//!
//! # Cost
//!
//! All N inputs are cloned (every future is live at once; for Arc-backed
//! payloads that is N refcount bumps), and one `Vec` of boxed futures is
//! allocated per call. This is the concurrent path, NOT the sequential hot path
//! — recording/telemetry fan-all use `FanOut`. The hedging latency win pays for
//! the one alloc.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::marker::PhantomData;

use futures::future::select_all;
use proxima_core::markers::DropSafe;
use crate::pipe::SendPipe;

use crate::pipe::fanout::{AllOrNothing, FanPolicy};

/// Construction error for [`Race`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaceBuildError {
    /// A race needs at least one sink to produce a winner.
    EmptySinks,
}

impl core::fmt::Display for RaceBuildError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptySinks => formatter.write_str("race requires at least one sink"),
        }
    }
}

impl core::error::Error for RaceBuildError {}

/// Concurrent first-`Ok`-wins dispatcher over N drop-safe sink [`SendPipe`]s.
///
/// `Policy` governs pre-winner errors: [`AllOrNothing`] returns the first
/// completion's error; [`crate::pipe::BestEffort`] ignores errors and waits for an
/// `Ok` (or all-failed).
pub struct Race<Sink, Policy = AllOrNothing> {
    sinks: Arc<Vec<Sink>>,
    policy: PhantomData<fn() -> Policy>,
}

impl<Sink, Policy> Clone for Race<Sink, Policy> {
    fn clone(&self) -> Self {
        Self {
            sinks: Arc::clone(&self.sinks),
            policy: PhantomData,
        }
    }
}

impl<Sink, Policy> Race<Sink, Policy>
where
    Sink: SendPipe + DropSafe,
    Policy: FanPolicy,
{
    /// Race over at least one sink. `Err(EmptySinks)` for empty input, so the
    /// `call` path is guaranteed a non-empty set (`select_all` would panic on
    /// empty).
    pub fn build(sinks: Vec<Sink>) -> Result<Self, RaceBuildError> {
        if sinks.is_empty() {
            return Err(RaceBuildError::EmptySinks);
        }
        Ok(Self {
            sinks: Arc::new(sinks),
            policy: PhantomData,
        })
    }

    #[must_use]
    pub fn sink_count(&self) -> usize {
        self.sinks.len()
    }
}

impl<Sink, Policy> SendPipe for Race<Sink, Policy>
where
    Sink: SendPipe + DropSafe,
    Sink::In: Clone + Send,
    Sink::Out: Send,
    Policy: FanPolicy,
{
    type In = Sink::In;
    type Out = Sink::Out;
    type Err = Sink::Err;

    fn call(&self, item: Sink::In) -> impl Future<Output = Result<Sink::Out, Sink::Err>> + Send {
        let sinks = Arc::clone(&self.sinks);
        async move {
            // homogeneous Pin<Box<F>> (same Sink => same future type) — no dyn,
            // no unsafe; Pin<Box<_>> is Unpin as select_all requires.
            let mut live: Vec<_> = sinks
                .iter()
                .map(|sink| Box::pin(sink.call(item.clone())))
                .collect();
            loop {
                // non-empty: build() guarantees >=1, and we return below before
                // re-entering with an empty set.
                let (result, _winner, rest) = select_all(live).await;
                match result {
                    Ok(output) => return Ok(output), // `rest` dropped here = cancel losers
                    Err(err) => {
                        if Policy::SHORT_CIRCUIT || rest.is_empty() {
                            return Err(err);
                        }
                        live = rest; // best-effort: keep waiting for an Ok
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::pipe::BestEffort;
    use alloc::sync::Arc;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use futures::executor::block_on;

    #[derive(Debug, PartialEq)]
    struct SinkErr(u32);

    struct Sink {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    impl SendPipe for Sink {
        type In = u32;
        type Out = u32;
        type Err = SinkErr;

        fn call(&self, input: u32) -> impl Future<Output = Result<u32, SinkErr>> + Send {
            let calls = Arc::clone(&self.calls);
            let fail = self.fail;
            async move {
                calls.fetch_add(1, Ordering::Relaxed);
                if fail { Err(SinkErr(input)) } else { Ok(input) }
            }
        }
    }

    impl DropSafe for Sink {}

    fn ok(calls: &Arc<AtomicUsize>) -> Sink {
        Sink {
            calls: Arc::clone(calls),
            fail: false,
        }
    }
    fn bad(calls: &Arc<AtomicUsize>) -> Sink {
        Sink {
            calls: Arc::clone(calls),
            fail: true,
        }
    }

    #[test]
    fn empty_sinks_is_a_build_error() {
        let built = Race::<Sink, AllOrNothing>::build(vec![]);
        assert_eq!(built.err(), Some(RaceBuildError::EmptySinks));
    }

    #[test]
    fn race_returns_an_ok_winner() {
        let calls = Arc::new(AtomicUsize::new(0));
        let race =
            Race::<_, AllOrNothing>::build(vec![ok(&calls), ok(&calls), ok(&calls)]).unwrap();
        assert_eq!(block_on(race.call(99)), Ok(99));
    }

    #[test]
    fn best_effort_skips_failures_and_finds_the_ok() {
        let calls = Arc::new(AtomicUsize::new(0));
        // two failing sinks + one ok; best-effort must surface the Ok, not an error.
        let race =
            Race::<_, BestEffort>::build(vec![bad(&calls), bad(&calls), ok(&calls)]).unwrap();
        assert_eq!(block_on(race.call(7)), Ok(7));
    }

    #[test]
    fn all_failing_surfaces_an_error() {
        let calls = Arc::new(AtomicUsize::new(0));
        let race = Race::<_, BestEffort>::build(vec![bad(&calls), bad(&calls)]).unwrap();
        assert!(block_on(race.call(3)).is_err(), "no Ok anywhere -> Err");
    }
}
