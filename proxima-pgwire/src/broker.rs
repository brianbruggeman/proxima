//! `NotifyBroker` — the LISTEN/NOTIFY pub/sub fabric shared across every
//! connection a [`crate::pipe::PgWireConnectionPipe`] upgrades.
//!
//! The engine never touches this broker: on LISTEN/UNLISTEN/NOTIFY SQL it
//! returns the matching [`crate::pipe_contract::PgReply`] variant, and the
//! driver — which already owns the wire — subscribes, unsubscribes, or
//! publishes. Delivery to a listening connection rides an unbounded
//! `futures::channel::mpsc`; the driver drains it only at a safe idle point
//! (after ReadyForQuery), so a NotificationResponse never interleaves a
//! message sequence.
//!
//! Lock-free on the hot path: subscriptions are a [`DashMap`] keyed by
//! channel, so a publish on one channel never blocks a subscribe on
//! another. Publish prunes senders whose receiver has dropped, so a gone
//! connection cannot leak a slot.

use dashmap::DashMap;
use futures::channel::mpsc::UnboundedSender;

/// One async notification delivered to a listening connection.
/// `process_id` is the *notifying* connection's backend pid (its
/// BackendKeyData process id), per the PostgreSQL protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notification {
    pub process_id: i32,
    pub channel: String,
    pub payload: String,
}

/// A live subscriber: the connection that issued LISTEN and the sender half
/// of its notification channel. `connection_id` keys removal on UNLISTEN /
/// connection close.
struct Subscriber {
    connection_id: u64,
    sender: UnboundedSender<Notification>,
}

/// Process-wide LISTEN/NOTIFY fabric. Construct once and share via `Arc`.
#[derive(Default)]
pub struct NotifyBroker {
    subscriptions: DashMap<String, Vec<Subscriber>>,
}

impl NotifyBroker {
    #[must_use]
    pub fn new() -> Self {
        Self {
            subscriptions: DashMap::new(),
        }
    }

    /// Subscribes a connection to a channel. A repeat LISTEN on the same
    /// channel from the same connection is idempotent (PostgreSQL collapses
    /// duplicate listens), so the sender is replaced rather than appended.
    pub fn subscribe(
        &self,
        channel: &str,
        connection_id: u64,
        sender: UnboundedSender<Notification>,
    ) {
        let mut subscribers = self.subscriptions.entry(channel.to_owned()).or_default();
        if let Some(existing) = subscribers
            .iter_mut()
            .find(|subscriber| subscriber.connection_id == connection_id)
        {
            existing.sender = sender;
            return;
        }
        subscribers.push(Subscriber {
            connection_id,
            sender,
        });
    }

    /// Removes a connection's subscription to one channel.
    pub fn unsubscribe(&self, channel: &str, connection_id: u64) {
        if let Some(mut subscribers) = self.subscriptions.get_mut(channel) {
            subscribers.retain(|subscriber| subscriber.connection_id != connection_id);
        }
    }

    /// Removes every subscription a connection holds — the connection-close
    /// path (UNLISTEN *).
    pub fn unsubscribe_all(&self, connection_id: u64) {
        for mut subscribers in self.subscriptions.iter_mut() {
            subscribers.retain(|subscriber| subscriber.connection_id != connection_id);
        }
    }

    /// Publishes to every live subscriber of a channel, pruning any whose
    /// receiver has dropped. Self-notify is delivered: a connection that
    /// LISTENs and NOTIFYs the same channel gets its own message, matching
    /// PostgreSQL.
    pub fn publish(&self, channel: &str, notification: &Notification) {
        let Some(mut subscribers) = self.subscriptions.get_mut(channel) else {
            return;
        };
        subscribers.retain(|subscriber| {
            subscriber
                .sender
                .unbounded_send(notification.clone())
                .is_ok()
        });
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use futures::StreamExt;
    use futures::channel::mpsc;

    use super::*;

    fn note(channel: &str, payload: &str) -> Notification {
        Notification {
            process_id: 42,
            channel: channel.into(),
            payload: payload.into(),
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_delivers_to_every_live_subscriber() {
        let broker = NotifyBroker::new();
        let (tx_a, mut rx_a) = mpsc::unbounded();
        let (tx_b, mut rx_b) = mpsc::unbounded();
        broker.subscribe("chan", 1, tx_a);
        broker.subscribe("chan", 2, tx_b);

        broker.publish("chan", &note("chan", "hi"));

        assert_eq!(rx_a.next().await.unwrap().payload, "hi");
        assert_eq!(rx_b.next().await.unwrap().payload, "hi");
    }

    #[proxima::test(runtime = "tokio")]
    async fn unsubscribe_stops_delivery() {
        let broker = NotifyBroker::new();
        let (tx, mut rx) = mpsc::unbounded();
        broker.subscribe("chan", 1, tx);

        broker.unsubscribe("chan", 1);
        broker.publish("chan", &note("chan", "hi"));

        assert_eq!(
            rx.try_recv().ok(),
            None,
            "no notification after unsubscribe"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn unsubscribe_all_stops_every_channel() {
        let broker = NotifyBroker::new();
        let (tx_one, mut rx_one) = mpsc::unbounded();
        let (tx_two, mut rx_two) = mpsc::unbounded();
        broker.subscribe("one", 1, tx_one);
        broker.subscribe("two", 1, tx_two);

        broker.unsubscribe_all(1);
        broker.publish("one", &note("one", "x"));
        broker.publish("two", &note("two", "y"));

        assert_eq!(
            rx_one.try_recv().ok(),
            None,
            "channel one drained after unsubscribe_all"
        );
        assert_eq!(
            rx_two.try_recv().ok(),
            None,
            "channel two drained after unsubscribe_all"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_prunes_closed_senders() {
        let broker = NotifyBroker::new();
        let (tx_live, mut rx_live) = mpsc::unbounded();
        let (tx_dead, rx_dead) = mpsc::unbounded::<Notification>();
        broker.subscribe("chan", 1, tx_dead);
        broker.subscribe("chan", 2, tx_live);
        drop(rx_dead);

        broker.publish("chan", &note("chan", "first"));
        // the dead subscriber is now pruned; the live one still receives
        broker.publish("chan", &note("chan", "second"));

        assert_eq!(rx_live.next().await.unwrap().payload, "first");
        assert_eq!(rx_live.next().await.unwrap().payload, "second");
        assert_eq!(
            broker
                .subscriptions
                .get("chan")
                .map(|subscribers| subscribers.len()),
            Some(1),
            "the closed sender must be pruned"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn self_subscribe_receives_its_own_notify() {
        let broker = NotifyBroker::new();
        let (tx, mut rx) = mpsc::unbounded();
        broker.subscribe("chan", 7, tx);

        broker.publish("chan", &note("chan", "self"));

        assert_eq!(rx.next().await.unwrap().payload, "self");
    }

    #[proxima::test(runtime = "tokio")]
    async fn repeat_listen_is_idempotent() {
        let broker = NotifyBroker::new();
        let (tx_first, _rx_first) = mpsc::unbounded();
        let (tx_second, mut rx_second) = mpsc::unbounded();
        broker.subscribe("chan", 1, tx_first);
        broker.subscribe("chan", 1, tx_second);

        broker.publish("chan", &note("chan", "once"));

        assert_eq!(rx_second.next().await.unwrap().payload, "once");
        assert_eq!(
            broker
                .subscriptions
                .get("chan")
                .map(|subscribers| subscribers.len()),
            Some(1),
            "a repeat listen from the same connection must not add a second slot"
        );
    }
}
