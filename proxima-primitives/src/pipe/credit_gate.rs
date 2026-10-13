//! A weighted, awaitable admission stage for bounded Pipe work.

use alloc::sync::Arc;
use core::fmt::{Debug, Formatter};
use core::future::Future;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::pipe::primitives::Pipe;
use crate::sync::Notify;

const CLOSED_BIT: usize = 1 << (usize::BITS - 1);
const CREDIT_MASK: usize = CLOSED_BIT - 1;

/// An input paired with the amount of capacity it needs while in flight.
pub struct CreditInput<Input> {
    /// The value passed to the next Pipe stage.
    pub value: Input,
    /// The amount of bounded capacity retained until that stage completes.
    pub credits: usize,
}

impl<Input> CreditInput<Input> {
    /// Pair an input with its positive or zero credit weight.
    #[must_use]
    pub const fn new(value: Input, credits: usize) -> Self {
        Self { value, credits }
    }
}

/// Pipe stage that awaits weighted capacity and holds it through its inner call.
pub struct CreditGate<Inner> {
    inner: Inner,
    state: Arc<GateState>,
}

impl<Inner> CreditGate<Inner> {
    /// Build a gate with a fixed maximum credit capacity.
    ///
    /// Capacities must fit in the low bits reserved for the credit counter.
    pub fn new(inner: Inner, capacity: usize) -> Result<Self, CreditCapacityError> {
        if capacity > CREDIT_MASK {
            return Err(CreditCapacityError);
        }
        Ok(Self {
            inner,
            state: Arc::new(GateState {
                state: AtomicUsize::new(capacity),
                capacity,
                changed: Notify::new(),
            }),
        })
    }

    /// Close admission and wake every caller waiting for capacity.
    pub fn close(&self) {
        self.state.state.fetch_or(CLOSED_BIT, Ordering::AcqRel);
        self.state.changed.notify_waiters();
    }

    /// Return a snapshot of currently available credits.
    #[must_use]
    pub fn available(&self) -> usize {
        self.state.state.load(Ordering::Acquire) & CREDIT_MASK
    }

    /// Reserve credits directly for an owned resource that outlives one Pipe call.
    pub async fn acquire(&self, credits: usize) -> Result<CreditLease, CreditAcquireError> {
        self.state.acquire(credits).await
    }
}

impl<Inner> Pipe for CreditGate<Inner>
where
    Inner: Pipe,
    Inner::In: 'static,
{
    type In = CreditInput<Inner::In>;
    type Out = Inner::Out;
    type Err = CreditGateError<Self::In, Inner::Err>;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        let state = Arc::clone(&self.state);
        async move {
            let permit = match state.acquire(input.credits).await {
                Ok(permit) => permit,
                Err(CreditAcquireError::Closed) => return Err(CreditGateError::Closed(input)),
                Err(CreditAcquireError::OverCapacity) => {
                    return Err(CreditGateError::OverCapacity(input));
                }
            };
            let result = self.inner.call(input.value).await;
            drop(permit);
            result.map_err(CreditGateError::Inner)
        }
    }
}

struct GateState {
    state: AtomicUsize,
    capacity: usize,
    changed: Notify,
}

impl GateState {
    async fn acquire(self: &Arc<Self>, credits: usize) -> Result<CreditLease, CreditAcquireError> {
        if credits > self.capacity {
            return Err(CreditAcquireError::OverCapacity);
        }

        loop {
            let notified = self.changed.notified();
            futures::pin_mut!(notified);
            let _ = notified.as_mut().enable();

            let current = self.state.load(Ordering::Acquire);
            if current & CLOSED_BIT != 0 {
                return Err(CreditAcquireError::Closed);
            }
            let available = current & CREDIT_MASK;
            if available >= credits {
                match self.state.compare_exchange_weak(
                    current,
                    current - credits,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        return Ok(CreditLease {
                            state: Arc::clone(self),
                            credits,
                        });
                    }
                    Err(_) => continue,
                }
            }

            notified.await;
        }
    }
}

/// Owned reservation that returns its credits when dropped.
pub struct CreditLease {
    state: Arc<GateState>,
    credits: usize,
}

impl Drop for CreditLease {
    fn drop(&mut self) {
        self.state.state.fetch_add(self.credits, Ordering::Release);
        self.state.changed.notify_waiters();
    }
}

/// Admission failure from a direct [`CreditGate::acquire`] call.
#[derive(Debug, thiserror::Error)]
pub enum CreditAcquireError {
    /// The gate closed before the requested credits were available.
    #[error("credit gate is closed")]
    Closed,
    /// The request exceeds the gate's total capacity.
    #[error("credit request exceeds gate capacity")]
    OverCapacity,
}

/// Error from gate admission or its inner Pipe; admission errors retain input.
pub enum CreditGateError<Input, InnerError> {
    /// The gate closed before this input acquired capacity.
    Closed(Input),
    /// The input requested more credits than the gate's total capacity.
    OverCapacity(Input),
    /// The inner Pipe returned this error after admission.
    Inner(InnerError),
}

impl<Input, InnerError: Debug> Debug for CreditGateError<Input, InnerError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Closed(_) => formatter.write_str("Closed(..)"),
            Self::OverCapacity(_) => formatter.write_str("OverCapacity(..)"),
            Self::Inner(error) => formatter.debug_tuple("Inner").field(error).finish(),
        }
    }
}

/// Invalid capacity supplied when constructing a [`CreditGate`].
#[derive(Debug, thiserror::Error)]
#[error("credit capacity exceeds the supported counter range")]
pub struct CreditCapacityError;

#[cfg(test)]
mod tests {
    use super::{CreditGate, CreditGateError, CreditInput};
    use crate::pipe::primitives::Pipe;
    use core::convert::Infallible;
    use core::future::Future;

    struct Echo;

    impl Pipe for Echo {
        type In = usize;
        type Out = usize;
        type Err = Infallible;

        fn call(&self, input: usize) -> impl Future<Output = Result<usize, Infallible>> {
            async move { Ok(input) }
        }
    }

    #[proxima::test]
    async fn test_omega_async_credit_gate_weighted_wait_releases_capacity() {
        let gate = CreditGate::new(Echo, 3).expect("valid capacity");
        let held = gate.state.acquire(2).await.expect("first credits");
        assert_eq!(gate.available(), 1);

        let waiting = gate.call(CreditInput::new(41, 2));
        futures::pin_mut!(waiting);
        assert!(matches!(
            futures::poll!(waiting.as_mut()),
            core::task::Poll::Pending
        ));
        drop(held);

        assert!(matches!(
            futures::poll!(waiting.as_mut()),
            core::task::Poll::Ready(Ok(41))
        ));
        assert_eq!(gate.available(), 3);
    }

    #[proxima::test]
    async fn test_omega_async_credit_gate_never_over_admits() {
        let gate = CreditGate::new(Echo, 3).expect("valid capacity");
        let first = gate.state.acquire(2).await.expect("two credits");
        let second = gate.state.acquire(1).await.expect("last credit");
        assert_eq!(gate.available(), 0);
        drop(first);
        assert_eq!(gate.available(), 2);
        drop(second);
        assert_eq!(gate.available(), 3);
    }

    #[proxima::test]
    async fn test_omega_async_credit_gate_close_returns_input() {
        let gate = CreditGate::new(Echo, 1).expect("valid capacity");
        let held = gate.state.acquire(1).await.expect("held credit");
        let waiting = gate.call(CreditInput::new(73, 1));
        futures::pin_mut!(waiting);
        assert!(matches!(
            futures::poll!(waiting.as_mut()),
            core::task::Poll::Pending
        ));
        gate.close();
        assert!(matches!(
            futures::poll!(waiting.as_mut()),
            core::task::Poll::Ready(Err(CreditGateError::Closed(CreditInput {
                value: 73,
                credits: 1
            })))
        ));
        drop(held);
    }
}
