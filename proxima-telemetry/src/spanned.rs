use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use pin_project_lite::pin_project;

use crate::id::{SpanId, TraceId};

/// A guard that can name the span it holds open, so a [`Spanned`] built with
/// [`Spanned::scoped`] can make that span the thread's current one FOR THE
/// DURATION OF EACH POLL. Implemented for `trace::SpanGuard` and the recorder's
/// `RecorderSpanGuard`; a sampled-out (noop) guard returns `None` and is never
/// scoped. Kept minimal so `Spanned`'s generic `new` path (the runtime spawn
/// carrier) stays unbounded — only the per-poll scoping opts into this bound.
pub trait SpanContext {
    fn span_context(&self) -> Option<(TraceId, SpanId)>;
}

pin_project! {
    /// A future paired with an owned trace-span guard: the span context
    /// travels WITH the future itself, the same shape as tracing's
    /// `Instrumented` (returned by `.instrument()`) or tokio-tracing's
    /// idiom — wrap, don't embed.
    ///
    /// Generic over the guard type `G` rather than the concrete
    /// `trace::SpanGuard<S, C>`: the recorder's real span-open path
    /// (`Recorder::span(..).start()`) returns `RecorderSpanGuard` — a
    /// 3-way enum over `Active` / `MetricOnly` / `Noop` guards, picked by
    /// the sampler — not a bare `SpanGuard`. `Spanned` only needs "a value
    /// whose `Drop` finishes whatever it represents"; it doesn't need to
    /// know which guard shape that is.
    ///
    /// Wave D Phase 1 de-embeds trace-span context from cross-cutting
    /// carriers (`proxima_runtime::SpawnRequest` no longer has a
    /// `SpanId` field, prime's `LocalExecutor` no longer threads one
    /// through its slab). A caller that wants a spawned task to carry
    /// its parent's span wraps the future in `Spanned` before boxing
    /// it — `Spanned<F, G>` still satisfies `Future<Output = F::Output>`,
    /// so it slots into the exact same `Pin<Box<dyn Future<Output = ()>
    /// + Send>>` the `Runtime` trait already spawns. `#[proxima::instrument]`
    /// on an `async fn` uses this same primitive for its own span.
    ///
    /// The ambient "current span" is task-safe BY CONSTRUCTION: [`scoped`]
    /// brackets the current-span stack per poll (push before polling `inner`,
    /// pop after), so it is set only across THIS task's synchronous poll and
    /// never leaks across an `.await` to whatever else the executor runs next.
    /// The `new` constructor opts out entirely — the runtime spawn carrier that
    /// only needs the span RECORD to travel with the future, not the ambient.
    ///
    /// [`scoped`]: Spanned::scoped
    pub struct Spanned<T, G> {
        #[pin]
        inner: T,
        guard: Option<G>,
        // Some => this task's span is made current per-poll; None => carry-only.
        // Captured once at construction (a guard's ids never change), so the poll
        // path is a stack push/pop, no guard re-inspection.
        context: Option<(TraceId, SpanId)>,
    }
}

impl<T, G> Spanned<T, G> {
    /// Wrap `inner` so `guard` covers its whole lifetime: already
    /// entered when this returns, finished the moment `inner` resolves
    /// or this `Spanned` is dropped early (a cancelled task closes its
    /// span at the point of cancellation, same as an early-dropped
    /// `tracing::Instrumented`).
    ///
    /// Carry-only: the span record travels with the future, but the ambient
    /// current-span is NOT set. Used by the runtime spawn path, whose `G` is
    /// left unbounded. Use [`scoped`](Spanned::scoped) to also correlate
    /// logs/metrics emitted inside the future to this span.
    pub fn new(inner: T, guard: G) -> Self {
        Self {
            inner,
            guard: Some(guard),
            context: None,
        }
    }
}

impl<T, G: SpanContext> Spanned<T, G> {
    /// Like [`new`](Spanned::new), but also makes `guard`'s span the thread's
    /// current one for the duration of each poll — so a log or metric emitted
    /// while this future is being polled correlates to it, and correctly does
    /// NOT correlate to it between polls (across an `.await`). This is the
    /// primitive `#[proxima::instrument]` on an `async fn` uses.
    pub fn scoped(inner: T, guard: G) -> Self {
        let context = guard.span_context();
        Self {
            inner,
            guard: Some(guard),
            context,
        }
    }
}

/// Restores the displaced parent on drop, so [`Spanned::poll`]'s per-poll enter is
/// balanced even if the inner future's poll unwinds — the same panic-safety the
/// sync `SpanGuard` gets from its own `Drop`.
struct RestoreOnDrop(Option<(TraceId, SpanId)>);

impl Drop for RestoreOnDrop {
    fn drop(&mut self) {
        crate::current::restore(self.0);
    }
}

impl<T: Future, G> Future for Spanned<T, G> {
    type Output = T::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        // bracket the current-span over THIS poll only: enter before polling the
        // inner future, restore the displaced parent (via the guard's drop) after
        // — so a task parked at an `.await` does not hold the executor thread's
        // current span while a sibling task runs, even if the inner poll unwinds.
        let _scope = this
            .context
            .map(|(trace_id, span_id)| RestoreOnDrop(crate::current::enter(trace_id, span_id)));
        match this.inner.poll(context) {
            Poll::Ready(value) => {
                // finish the span at completion time, not at Spanned's own
                // drop — a `Pending` future may be polled many times before
                // it resolves, and the record's duration must reflect
                // "start to resolve", not "start to some later poll".
                this.guard.take();
                Poll::Ready(value)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::rc::Rc;
    use alloc::vec::Vec;
    use core::cell::{Cell, RefCell};

    use super::*;
    use crate::clock::MonotonicCounter;
    use crate::id::{SpanId, TraceId};
    use crate::trace::span::{SpanBuilder, SpanGuard, SpanSink};
    use crate::trace::tracestate::TraceState;
    #[cfg(feature = "std")]
    use crate::current::current;
    #[cfg(feature = "std")]
    use crate::level::Level;
    #[cfg(feature = "std")]
    use crate::log::{LogBuilder, LogRecord};

    struct RecordingSink {
        records: Rc<RefCell<Vec<SpanId>>>,
    }

    impl SpanSink for RecordingSink {
        fn emit(&mut self, record: crate::trace::span::SpanRecord) {
            self.records.borrow_mut().push(record.span_id);
        }
    }

    fn guard_for(
        records: Rc<RefCell<Vec<SpanId>>>,
        span_id: SpanId,
    ) -> SpanGuard<RecordingSink, MonotonicCounter> {
        let builder = SpanBuilder::new("test-span", TraceId::from_bytes([1; 16]), span_id)
            .with_tracestate(TraceState::empty());
        builder
            .start(&MonotonicCounter::new(0), RecordingSink { records })
            .enter(MonotonicCounter::new(0))
    }

    #[test]
    fn spanned_future_emits_its_record_when_the_future_resolves() {
        let records: Rc<RefCell<Vec<SpanId>>> = Rc::new(RefCell::new(Vec::new()));
        let span_id = SpanId::from_bytes([9; 8]);
        let guard = guard_for(records.clone(), span_id);
        let ran = Rc::new(Cell::new(false));
        let ran_for_future = ran.clone();

        let spanned = Spanned::new(
            async move {
                ran_for_future.set(true);
                42_u32
            },
            guard,
        );

        let output = poll_to_completion(spanned);

        assert_eq!(output, 42, "wrapped future's output passes through Spanned");
        assert!(ran.get(), "inner future ran exactly once");
        assert_eq!(
            records.borrow().as_slice(),
            [span_id],
            "the span record must be emitted with the wrapped span's id once the future resolves"
        );
    }

    #[test]
    fn spanned_future_dropped_before_resolving_still_emits_its_record() {
        let records: Rc<RefCell<Vec<SpanId>>> = Rc::new(RefCell::new(Vec::new()));
        let span_id = SpanId::from_bytes([7; 8]);
        let guard = guard_for(records.clone(), span_id);

        let pending_forever = core::future::pending::<()>();
        let spanned = Spanned::new(pending_forever, guard);
        drop(spanned);

        assert_eq!(
            records.borrow().as_slice(),
            [span_id],
            "a cancelled (dropped-before-ready) Spanned future must still close its span, \
             mirroring tracing::Instrumented's drop behavior"
        );
    }

    /// A future that, on every poll, emits a span-less log (so the emit path
    /// stamps whatever span is CURRENT for that poll) and records the span id it
    /// was stamped with. Pending `polls` times, then Ready.
    #[cfg(feature = "std")]
    struct SpanEcho {
        polls: usize,
        seen: Rc<RefCell<Vec<Option<SpanId>>>>,
    }

    #[cfg(feature = "std")]
    impl Future for SpanEcho {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<()> {
            let this = self.get_mut();
            let seen = this.seen.clone();
            LogBuilder::new(
                Level::INFO,
                move |record: LogRecord| seen.borrow_mut().push(record.span_id),
                MonotonicCounter::new(0),
            )
            .message("tick")
            .emit();
            if this.polls == 0 {
                Poll::Ready(())
            } else {
                this.polls -= 1;
                Poll::Pending
            }
        }
    }

    #[cfg(feature = "std")]
    fn deferred_guard(
        span_id: SpanId,
    ) -> SpanGuard<impl SpanSink, MonotonicCounter> {
        SpanBuilder::new("task", TraceId::from_bytes([0; 16]), span_id)
            .with_tracestate(TraceState::empty())
            .start(&MonotonicCounter::new(0), |_record: crate::trace::span::SpanRecord| {})
            .enter_deferred(MonotonicCounter::new(0))
    }

    // The whole point: two instrumented futures polled INTERLEAVED on one thread.
    // A log emitted while polling A must carry A's span, while polling B must carry
    // B's — and between polls the thread must hold NO current span (nothing leaks
    // across the `.await`).
    #[cfg(feature = "std")]
    #[test]
    fn interleaved_polls_carry_the_polling_tasks_span() {
        let a_span = SpanId::from_bytes([0xa1; 8]);
        let b_span = SpanId::from_bytes([0xb2; 8]);
        let a_seen: Rc<RefCell<Vec<Option<SpanId>>>> = Rc::new(RefCell::new(Vec::new()));
        let b_seen: Rc<RefCell<Vec<Option<SpanId>>>> = Rc::new(RefCell::new(Vec::new()));

        let mut task_a = core::pin::pin!(Spanned::scoped(
            SpanEcho {
                polls: 2,
                seen: a_seen.clone(),
            },
            deferred_guard(a_span),
        ));
        let mut task_b = core::pin::pin!(Spanned::scoped(
            SpanEcho {
                polls: 2,
                seen: b_seen.clone(),
            },
            deferred_guard(b_span),
        ));

        let waker = core::task::Waker::noop();
        let mut context = Context::from_waker(waker);

        // A, B, A, B, A, B — each pending twice then ready on the third poll.
        for _ in 0..3 {
            let _ = task_a.as_mut().poll(&mut context);
            assert_eq!(current(), None, "no current span leaks between A's and B's polls");
            let _ = task_b.as_mut().poll(&mut context);
            assert_eq!(current(), None, "no current span leaks between B's and A's polls");
        }

        assert_eq!(a_seen.borrow().len(), 3, "A polled three times");
        assert_eq!(b_seen.borrow().len(), 3, "B polled three times");
        assert!(
            a_seen.borrow().iter().all(|id| *id == Some(a_span)),
            "every log emitted while polling A carries A's span, never B's: {:?}",
            a_seen.borrow()
        );
        assert!(
            b_seen.borrow().iter().all(|id| *id == Some(b_span)),
            "every log emitted while polling B carries B's span, never A's: {:?}",
            b_seen.borrow()
        );
    }

    // Carry-only `new` (the spawn path) must NOT set the current span — only
    // `scoped` does. A log emitted inside a `new`-wrapped future's poll sees no
    // current span.
    #[cfg(feature = "std")]
    #[test]
    fn carry_only_new_does_not_scope_current() {
        let seen: Rc<RefCell<Vec<Option<SpanId>>>> = Rc::new(RefCell::new(Vec::new()));
        let mut task = core::pin::pin!(Spanned::new(
            SpanEcho {
                polls: 0,
                seen: seen.clone(),
            },
            deferred_guard(SpanId::from_bytes([0xcc; 8])),
        ));

        let waker = core::task::Waker::noop();
        let mut context = Context::from_waker(waker);
        let _ = task.as_mut().poll(&mut context);

        assert_eq!(
            seen.borrow().as_slice(),
            [None],
            "carry-only new leaves the current span unset"
        );
    }

    /// Drive a future to completion with a no-op waker. Every future under
    /// test here resolves synchronously on first poll (no real I/O or timer
    /// wait), so this is a plain single-poll executor stand-in — no runtime
    /// dependency needed for a unit test this small.
    fn poll_to_completion<F: Future>(future: F) -> F::Output {
        let mut future = core::pin::pin!(future);
        let waker = core::task::Waker::noop();
        let mut context = Context::from_waker(waker);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
        }
    }
}
