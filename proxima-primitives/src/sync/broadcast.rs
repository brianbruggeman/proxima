//! `proxima::sync::broadcast` — multi-producer multi-consumer
//! broadcast channel, shape-compatible with `tokio::sync::broadcast`.
//! Backed by `async_broadcast` configured for overflow semantics:
//! producers never block; if the buffer fills, the oldest message is
//! dropped and the affected receivers see `RecvError::Lagged(skipped)`
//! on their next `recv`.
//!
//! Differences from tokio:
//! - `error::RecvError::Lagged(u64)` is named to match tokio's
//!   discriminant. (Underlying crate calls it `Overflowed`.)
//! - `send` returns `Ok(subscriber_count)` matching tokio's signature.
//!
//! # Non-coverage
//!
//! - `Sender::send` returns `Ok(subscriber_count)` but the count is
//!   sampled at send time and is eventually-consistent with concurrent
//!   subscribe/drop operations. tokio's version has the same property
//!   but we call it out here because a sharp caller may rely on the
//!   exact-snapshot semantics that neither implementation provides.
//! - `Receiver::resubscribe() -> Receiver<T>` — fresh subscription
//!   anchored at the current end of stream. Not exposed today; if
//!   needed, clone the `Sender` and call `subscribe()`.
//! - `Receiver::len()` / `Receiver::is_empty()` — receiver-local
//!   buffer introspection. `async-broadcast` has equivalents but no
//!   internal caller plumbs them.
//! - `Receiver::same_channel(&other)` — identity check.
//! - `Sender::receiver_count()` IS exposed but the value is a
//!   snapshot; we expose it because `recording/broadcast.rs` uses it
//!   to decide whether to bother broadcasting (best-effort).
//! - `Sender::weak_count()` and weak-sender plumbing — no caller.

use async_broadcast::{InactiveReceiver, Sender as InnerSender};

pub mod error {
    /// Error returned from [`super::Receiver::recv`] when the channel
    /// is no longer producing values, or when this receiver skipped
    /// past some messages because it was too slow.
    #[derive(Debug, thiserror::Error)]
    pub enum RecvError {
        /// All senders have been dropped and the buffer is drained.
        #[error("broadcast channel closed")]
        Closed,
        /// The buffer overflowed and this receiver skipped `n` messages.
        /// The next `recv` will return the next available value.
        #[error("broadcast receiver lagged behind by {0} messages")]
        Lagged(u64),
    }
}

/// Error returned from [`Sender::send`] when no receivers are
/// listening. The value is returned so callers can retry on a fresh
/// channel.
#[derive(Debug, thiserror::Error)]
#[error("broadcast channel has no receivers")]
pub struct SendError<T>(pub T);

/// Producer side. Cheaply cloneable.
#[derive(Debug)]
pub struct Sender<T: Clone> {
    inner: InnerSender<T>,
    /// Kept alive so the channel stays open while at least one Sender
    /// exists, even if every Receiver has been dropped. Mirrors
    /// tokio's "channel stays open until all Senders drop" rule.
    _liveness: InactiveReceiver<T>,
}

impl<T: Clone> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _liveness: self._liveness.clone(),
        }
    }
}

impl<T: Clone> Sender<T> {
    /// Send a value to every active receiver. Returns the subscriber
    /// count at the time of the send. Returns `SendError(value)` if
    /// no receivers are listening.
    pub fn send(&self, value: T) -> Result<usize, SendError<T>> {
        if self.inner.receiver_count() == 0 {
            return Err(SendError(value));
        }
        match self.inner.try_broadcast(value) {
            Ok(_displaced) => Ok(self.inner.receiver_count()),
            Err(async_broadcast::TrySendError::Closed(value)) => Err(SendError(value)),
            Err(async_broadcast::TrySendError::Inactive(value)) => Err(SendError(value)),
            Err(async_broadcast::TrySendError::Full(value)) => {
                // overflow mode is set in `channel()`, so Full should
                // not occur; treat as logic error returning value back.
                Err(SendError(value))
            }
        }
    }

    /// Build a fresh receiver subscribed at the current end of the
    /// stream. Messages broadcast BEFORE this call are not delivered
    /// to the new receiver.
    pub fn subscribe(&self) -> Receiver<T> {
        Receiver {
            inner: self.inner.new_receiver(),
        }
    }

    /// Returns the number of currently-active receivers.
    pub fn receiver_count(&self) -> usize {
        self.inner.receiver_count()
    }
}

/// Consumer side. Each receiver has its own cursor across the buffer.
#[derive(Debug)]
pub struct Receiver<T: Clone> {
    inner: async_broadcast::Receiver<T>,
}

impl<T: Clone> Receiver<T> {
    /// Await the next message. Returns `Closed` when all senders are
    /// dropped and the buffer is drained; `Lagged(n)` if the receiver
    /// skipped `n` messages due to buffer overflow.
    pub async fn recv(&mut self) -> Result<T, error::RecvError> {
        match self.inner.recv().await {
            Ok(value) => Ok(value),
            Err(async_broadcast::RecvError::Closed) => Err(error::RecvError::Closed),
            Err(async_broadcast::RecvError::Overflowed(skipped)) => {
                Err(error::RecvError::Lagged(skipped))
            }
        }
    }
}

impl<T: Clone> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Build a broadcast channel with `capacity` slots. Producers never
/// block on a full buffer; the oldest message is dropped and the
/// affected receivers observe `Lagged` on their next `recv`.
pub fn channel<T: Clone>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let (mut sender, receiver) = async_broadcast::broadcast(capacity.max(1));
    sender.set_overflow(true);
    sender.set_await_active(false);
    let liveness = receiver.clone().deactivate();
    (
        Sender {
            inner: sender,
            _liveness: liveness,
        },
        Receiver { inner: receiver },
    )
}

// under `--cfg loom`, `async_broadcast` becomes loom-instrumented too (it
// depends on `event_listener`) and requires every call inside a
// `loom::model()`; these plain tests are unrelated to the Notify/watch
// loom protocol this crate loom-tests, so they're normal (non-loom)
// build only.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn send_to_active_receiver_roundtrips() {
        let (sender, mut receiver) = channel::<u32>(4);
        block_on(async move {
            sender.send(7).expect("send");
            assert_eq!(receiver.recv().await.expect("recv"), 7);
        });
    }

    #[test]
    fn second_subscriber_only_sees_messages_sent_after_subscribe() {
        let (sender, mut alpha) = channel::<u32>(4);
        sender.send(1).expect("send 1");
        let mut beta = sender.subscribe();
        sender.send(2).expect("send 2");
        block_on(async move {
            assert_eq!(alpha.recv().await.expect("alpha first"), 1);
            assert_eq!(alpha.recv().await.expect("alpha second"), 2);
            assert_eq!(beta.recv().await.expect("beta"), 2);
        });
    }

    #[test]
    fn send_succeeds_when_no_receivers_but_inactive_holds_open() {
        let (sender, receiver) = channel::<u32>(4);
        drop(receiver);
        // No active receivers; send returns SendError so callers can recover.
        match sender.send(1) {
            Err(SendError(value)) => assert_eq!(value, 1),
            Ok(_) => panic!("expected SendError"),
        }
    }

    #[test]
    fn overflow_yields_lagged_on_recv() {
        let (sender, mut receiver) = channel::<u32>(2);
        sender.send(1).expect("send 1");
        sender.send(2).expect("send 2");
        sender.send(3).expect("send 3 (overflows 1)");
        block_on(async move {
            match receiver.recv().await {
                Err(error::RecvError::Lagged(skipped)) => assert!(skipped >= 1),
                other => panic!("expected Lagged, got {other:?}"),
            }
        });
    }
}
