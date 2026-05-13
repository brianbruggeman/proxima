//! `proxima::sync::mpsc` — multi-producer single-consumer channel,
//! shape-compatible with `tokio::sync::mpsc`. Backed by `async-channel`
//! because tokio's `Sender::send(&self, T)` shape — async send without
//! a `&mut` borrow — requires interior synchronization that
//! `async-channel` provides directly. `futures::channel::mpsc`'s
//! capacity-per-clone semantics inflate buffers unexpectedly when used
//! the same way, so we route through `async-channel` instead.
//!
//! Two flavors:
//! - [`channel`]/[`Sender`]/[`Receiver`] — bounded.
//! - [`unbounded_channel`]/[`UnboundedSender`]/[`UnboundedReceiver`] — unbounded.
//!
//! Differences from tokio:
//! - [`TrySendError`] is split into `Full(T)` / `Closed(T)` rather than
//!   tokio's enum with the same discriminants; existing callers that
//!   match on variants port verbatim.
//!
//! # Non-coverage
//!
//! The following tokio APIs are intentionally NOT shimmed today. If a
//! caller appears, file a request — most are mechanical to add, just
//! unused so far.
//!
//! **Bounded sender side**
//! - `Sender::reserve() -> Permit` / `Sender::reserve_owned()` — slot
//!   reservation without send. No internal caller uses the
//!   reserve-then-send-later pattern; we expose `try_send` instead.
//! - `Sender::send_timeout(value, duration)` — timeout-bounded send.
//!   Compose with [`crate::time::timeout`] at the call site.
//! - `Sender::closed().await` — wait-until-receiver-dropped sender
//!   primitive. Compose via [`Sender::is_closed`] polling or
//!   subscribe to a separate shutdown signal.
//! - `Sender::downgrade() -> WeakSender` and weak-sender plumbing.
//!
//! **Bounded receiver side**
//! - `Receiver::try_recv() -> Result<T, TryRecvError>` — non-blocking
//!   poll without await. Callers needing this pattern typically want
//!   a [`crate::sync::watch`] instead.
//! - `Receiver::poll_recv(&mut Context)` — manual poll for callers
//!   embedding the receiver in a custom future. Wrap in a state
//!   machine if needed.
//! - `Receiver::recv_many(buf, limit)` — batch drain. Compose via a
//!   `while let Some(value) = rx.recv().await` loop with a counter.
//!
//! **Introspection**
//! - `Sender::capacity()` / `Sender::max_capacity()` — runtime
//!   capacity introspection. `async-channel` exposes equivalents but
//!   we have no caller plumbing them through.
//! - `Sender::same_channel(&other)` — identity check across clones.
//! - `Sender::weak_count()` / `Sender::strong_count()` — refcount
//!   diagnostics.
//!
//! **Unbounded variant**
//! - Mirrors the bounded gaps. `UnboundedReceiver::try_recv` and
//!   `recv_many` likewise absent.

/// Returned by [`Sender::send`] / [`UnboundedSender::send`] when the
/// receiver has been dropped. The unsent value is returned so callers
/// can recover or retry on a fresh channel.
#[derive(Debug, thiserror::Error)]
#[error("send on closed channel")]
pub struct SendError<T>(pub T);

impl<T> SendError<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

/// Returned by [`Sender::try_send`] when the channel cannot accept the
/// value synchronously.
#[derive(Debug, thiserror::Error)]
pub enum TrySendError<T> {
    /// Channel is at capacity; caller should retry or use `send`.
    #[error("channel full")]
    Full(T),
    /// Receiver has been dropped; the channel is closed for good.
    #[error("channel closed")]
    Closed(T),
}

impl<T> TrySendError<T> {
    pub fn into_inner(self) -> T {
        match self {
            Self::Full(value) | Self::Closed(value) => value,
        }
    }
}

/// Bounded sender. Clone-cheap (atomic refcount); multiple producers
/// share the same channel buffer (no per-clone capacity inflation).
#[derive(Debug)]
pub struct Sender<T> {
    inner: async_channel::Sender<T>,
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Sender<T> {
    /// Send a value, awaiting channel capacity if full. Returns the
    /// value back in [`SendError`] if the receiver has been dropped.
    pub async fn send(&self, value: T) -> Result<(), SendError<T>> {
        self.inner
            .send(value)
            .await
            .map_err(|error| SendError(error.0))
    }

    /// Synchronous send attempt.
    pub fn try_send(&self, value: T) -> Result<(), TrySendError<T>> {
        match self.inner.try_send(value) {
            Ok(()) => Ok(()),
            Err(async_channel::TrySendError::Full(value)) => Err(TrySendError::Full(value)),
            Err(async_channel::TrySendError::Closed(value)) => Err(TrySendError::Closed(value)),
        }
    }

    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

/// Bounded receiver. Drop it to close the channel for all senders.
#[derive(Debug)]
pub struct Receiver<T> {
    inner: async_channel::Receiver<T>,
}

impl<T> Receiver<T> {
    /// Await the next value, or `None` if all senders have been
    /// dropped and the buffer is drained.
    pub async fn recv(&mut self) -> Option<T> {
        self.inner.recv().await.ok()
    }

    pub fn close(&mut self) -> bool {
        self.inner.close()
    }
}

/// Build a bounded channel with the given buffer capacity.
pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let (sender, receiver) = async_channel::bounded(capacity.max(1));
    (Sender { inner: sender }, Receiver { inner: receiver })
}

/// Unbounded sender. Send is sync because the channel never blocks on
/// capacity.
#[derive(Debug)]
pub struct UnboundedSender<T> {
    inner: async_channel::Sender<T>,
}

impl<T> Clone for UnboundedSender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> UnboundedSender<T> {
    /// Send a value. Always succeeds unless the receiver has been
    /// dropped, in which case the value is returned via [`SendError`].
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        match self.inner.try_send(value) {
            Ok(()) => Ok(()),
            Err(async_channel::TrySendError::Closed(value)) => Err(SendError(value)),
            Err(async_channel::TrySendError::Full(_)) => {
                unreachable!("unbounded channel reported full")
            }
        }
    }

    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

/// Unbounded receiver. `recv` resolves to `None` once all senders are
/// dropped and the buffer is drained.
#[derive(Debug)]
pub struct UnboundedReceiver<T> {
    inner: async_channel::Receiver<T>,
}

impl<T> UnboundedReceiver<T> {
    pub async fn recv(&mut self) -> Option<T> {
        self.inner.recv().await.ok()
    }

    pub fn close(&mut self) -> bool {
        self.inner.close()
    }
}

/// Build an unbounded channel.
pub fn unbounded_channel<T>() -> (UnboundedSender<T>, UnboundedReceiver<T>) {
    let (sender, receiver) = async_channel::unbounded();
    (
        UnboundedSender { inner: sender },
        UnboundedReceiver { inner: receiver },
    )
}

// under `--cfg loom`, `async_channel` becomes loom-instrumented too (it
// depends on `event_listener` / `concurrent-queue`) and requires every
// call inside a `loom::model()`; these plain tests are unrelated to the
// Notify/watch loom protocol this crate loom-tests, so they're normal
// (non-loom) build only.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn bounded_send_recv_roundtrips() {
        let (sender, mut receiver) = channel::<u32>(4);
        block_on(async move {
            sender.send(7).await.expect("send");
            assert_eq!(receiver.recv().await, Some(7));
        });
    }

    #[test]
    fn bounded_try_send_returns_full_when_at_capacity() {
        let (sender, _receiver) = channel::<u32>(1);
        sender.try_send(1).expect("first slot");
        match sender.try_send(2) {
            Err(TrySendError::Full(value)) => assert_eq!(value, 2),
            other => panic!("expected Full, got {other:?}"),
        }
    }

    #[test]
    fn bounded_send_returns_value_after_receiver_dropped() {
        let (sender, receiver) = channel::<u32>(1);
        drop(receiver);
        block_on(async move {
            match sender.send(9).await {
                Err(error) => assert_eq!(error.into_inner(), 9),
                Ok(()) => panic!("send to dropped receiver should error"),
            }
        });
    }

    #[test]
    fn unbounded_send_recv_roundtrips() {
        let (sender, mut receiver) = unbounded_channel::<u32>();
        sender.send(42).expect("send");
        block_on(async move {
            assert_eq!(receiver.recv().await, Some(42));
        });
    }

    #[test]
    fn unbounded_send_returns_value_after_receiver_dropped() {
        let (sender, receiver) = unbounded_channel::<u32>();
        drop(receiver);
        match sender.send(3) {
            Err(error) => assert_eq!(error.into_inner(), 3),
            Ok(()) => panic!("send to dropped receiver should error"),
        }
    }
}
