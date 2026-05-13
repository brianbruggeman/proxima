//! `Demand<S, G>` — a dormancy gate over any [`SendPipe`].
//!
//! Generalises recording's `AccumulatingSink::is_armed()` spigot into a
//! composable wrapper: when the [`DemandGate`] is closed (no downstream
//! consumer) the wrapped pipe is a no-op — `call` returns `Ok` immediately with
//! no await, no inner future, nothing consumed. "No downstream → the pipe is
//! dormant."
//!
//! Dormancy (is there a consumer at all?) is orthogonal to backpressure (is
//! there room right now? — see [`crate::pipe::BoundedQueue`]). Compose them in series:
//! `Demand::new(BoundedPipe(inner), gate)`.
//!
//! This is a wrapper, NOT a parameter on [`crate::pipe::FanOut`]: a pipe with no gate
//! pays nothing because it is simply not wrapped, and the gate composes over any
//! `SendPipe`, not only fans.

use alloc::sync::Arc;
use core::future::Future;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::pipe::SendPipe;

/// Whether a [`Demand`]-wrapped pipe currently has downstream demand.
///
/// `is_armed` is checked synchronously at `call` entry — no await on the gate.
/// `AlwaysArmed` folds to a constant `true` the optimiser deletes.
pub trait DemandGate: Send + Sync + 'static {
    /// `true` when at least one downstream consumer is active.
    fn is_armed(&self) -> bool;

    /// Called when the gate transitions armed→disarmed. Default no-op; a gate
    /// that fronts a [`crate::pipe::BoundedQueue`] overrides this to drain the queue
    /// and avoid losing items buffered during the coming dormancy.
    fn on_close(&self) {}
}

/// The always-on gate: the wrapped pipe always dispatches. Zero-cost — the
/// `true` check is eliminated by the optimiser.
pub struct AlwaysArmed;

impl DemandGate for AlwaysArmed {
    fn is_armed(&self) -> bool {
        true
    }
}

/// A shared atomic gate. Construct via [`AtomicGate::pair`] to get the gate
/// (handed to the pipe) plus an [`AtomicGateController`] (kept by the caller to
/// arm/disarm after the gate has been moved into the pipe).
pub struct AtomicGate(Arc<AtomicBool>);

impl AtomicGate {
    /// A gate and its controller, sharing one atomic. The gate moves into the
    /// [`Demand`] wrapper; the controller stays with the caller to signal demand.
    #[must_use]
    pub fn pair(initial_armed: bool) -> (Self, AtomicGateController) {
        let shared = Arc::new(AtomicBool::new(initial_armed));
        (Self(Arc::clone(&shared)), AtomicGateController(shared))
    }
}

impl DemandGate for AtomicGate {
    fn is_armed(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// The write-half of an [`AtomicGate`] — arms/disarms demand from outside the
/// pipe that holds the gate. A distinct type from the gate so the pipe side
/// cannot accidentally flip its own demand.
pub struct AtomicGateController(Arc<AtomicBool>);

impl AtomicGateController {
    /// Signal that downstream demand exists — the pipe will dispatch.
    pub fn arm(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Signal that downstream demand is gone — the pipe goes dormant.
    pub fn disarm(&self) {
        self.0.store(false, Ordering::Release);
    }

    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// A [`SendPipe`] gated by downstream [`DemandGate`]: dormant (no-op `Ok`) when
/// the gate is closed, the inner pipe when open.
pub struct Demand<S, G = AlwaysArmed> {
    inner: S,
    gate: G,
}

impl<S, G> Demand<S, G> {
    /// Gate `inner` behind `gate`.
    #[must_use]
    pub fn new(inner: S, gate: G) -> Self {
        Self { inner, gate }
    }

    /// Whether the gate is currently open.
    #[must_use]
    pub fn is_armed(&self) -> bool
    where
        G: DemandGate,
    {
        self.gate.is_armed()
    }
}

impl<S, G> SendPipe for Demand<S, G>
where
    S: SendPipe,
    S::In: Send,
    G: DemandGate,
{
    type In = S::In;
    type Out = ();
    type Err = S::Err;

    fn call(&self, item: S::In) -> impl Future<Output = Result<(), S::Err>> + Send {
        // synchronous gate check — a dormant gate consumes nothing and never awaits.
        let armed = self.gate.is_armed();
        async move {
            if armed {
                self.inner.call(item).await.map(|_| ())
            } else {
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use alloc::sync::Arc;
    use core::sync::atomic::AtomicUsize;
    use futures::executor::block_on;

    #[derive(Debug)]
    struct NoErr;

    struct CountingSink(Arc<AtomicUsize>);

    impl SendPipe for CountingSink {
        type In = u32;
        type Out = ();
        type Err = NoErr;

        fn call(&self, _input: u32) -> impl Future<Output = Result<(), NoErr>> + Send {
            let calls = Arc::clone(&self.0);
            async move {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        }
    }

    #[test]
    fn always_armed_always_dispatches() {
        let calls = Arc::new(AtomicUsize::new(0));
        let gated = Demand::new(CountingSink(Arc::clone(&calls)), AlwaysArmed);
        block_on(gated.call(1)).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn closed_gate_is_dormant_no_dispatch() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (gate, _controller) = AtomicGate::pair(false);
        let gated = Demand::new(CountingSink(Arc::clone(&calls)), gate);
        for _ in 0..100 {
            block_on(gated.call(1)).expect("dormant returns Ok");
        }
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "no dispatch while dormant"
        );
    }

    #[test]
    fn controller_arms_after_gate_moved_into_pipe() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (gate, controller) = AtomicGate::pair(false);
        let gated = Demand::new(CountingSink(Arc::clone(&calls)), gate);

        block_on(gated.call(1)).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 0, "dormant before arm");

        controller.arm();
        block_on(gated.call(2)).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1, "dispatches once armed");

        controller.disarm();
        block_on(gated.call(3)).unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "dormant again after disarm"
        );
    }
}
