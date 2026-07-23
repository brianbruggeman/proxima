//! `MqttBroker` — the PUBLISH/SUBSCRIBE pub/sub fabric shared across every
//! connection a [`crate::pipe::MqttConnectionPipe`] upgrades.
//!
//! Mirrors `proxima_redis::broker::RedisBroker`'s role, built on
//! [`proxima_primitives::pipe::KeyedFanOut`] instead of a hand-rolled
//! `DashMap<Vec<u8>, Vec<Subscriber>>` — but with only ONE fan-out group,
//! not redis's exact/pattern pair: MQTT has a single `SUBSCRIBE` command
//! whose one topic filter MAY or MAY NOT carry a `+`/`#` wildcard (unlike
//! redis's separate SUBSCRIBE-exact vs PSUBSCRIBE-pattern commands), so
//! every registered filter — wildcard or not — lives in the same
//! [`KeyedFanOut`] keyed by the raw filter bytes, with a
//! [`LiveFilter<TopicFilterSet>`] snapshot of "every filter currently
//! registered" kept in sync so [`MqttBroker::publish`] finds the matching
//! filter keys with one wait-free read instead of a wildcard-unaware scan
//! of the registry's own key list — exactly the role
//! `RedisBroker`'s `pattern_set` plays for PSUBSCRIBE.
//!
//! Delivery is always QoS 0, retain-flag cleared: this broker does not
//! implement QoS 1/2 redelivery bookkeeping or retained-message storage
//! for FANNED-OUT messages (a publisher's own QoS 1/2 handshake with the
//! broker — `PUBACK`/`PUBREC`/`PUBREL`/`PUBCOMP` — is answered by
//! [`crate::connection`] regardless; only the onward fan-out to
//! subscribers is downgraded). Scope-matches
//! `RedisBroker::publish`'s at-most-once pub/sub semantics.

use core::future::Future;

use bytes::Bytes;
use futures::channel::mpsc::UnboundedSender;

use proxima_core::ProximaError;
use proxima_primitives::pipe::{
    BestEffort, FilterControl, KeyedFanOut, LiveFilter, SendPipe, SubscriptionId, live_filter,
};
use proxima_protocols::mqtt::encode::encode_publish;

use crate::topic_filter::TopicFilterSet;

/// A connection's push-sink: the sender half of its outbound-frame channel,
/// adapted to [`SendPipe`] so it composes as an ordinary
/// [`KeyedFanOut`]/[`FanOut`](proxima_primitives::pipe::FanOut) sink
/// (workspace principle 1 — "a sink is an ordinary pipe"). The connection
/// driver's `select!` races the paired receiver against the socket read.
#[derive(Clone)]
pub struct PushSink(UnboundedSender<Bytes>);

impl PushSink {
    #[must_use]
    pub fn new(sender: UnboundedSender<Bytes>) -> Self {
        Self(sender)
    }
}

impl SendPipe for PushSink {
    type In = Bytes;
    type Out = ();
    type Err = ProximaError;

    fn call(&self, item: Bytes) -> impl Future<Output = Result<(), ProximaError>> + Send {
        let result = self.0.unbounded_send(item);
        async move {
            result.map_err(|error| ProximaError::Upstream(format!("push sink closed: {error}")))
        }
    }
}

/// Process-wide PUBLISH/SUBSCRIBE fabric. Construct once and share via
/// `Arc`.
pub struct MqttBroker {
    subscriptions: KeyedFanOut<PushSink, BestEffort>,
    /// Live mirror of `subscriptions`'s registered keys, kept in sync by
    /// `subscribe`/`unsubscribe` — lets `publish` find the matching filter
    /// keys with one wait-free read instead of iterating
    /// `subscriptions.keys()` and wildcard-testing each on every PUBLISH.
    filter_set: LiveFilter<TopicFilterSet>,
    filter_control: FilterControl<TopicFilterSet>,
}

impl Default for MqttBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl MqttBroker {
    #[must_use]
    pub fn new() -> Self {
        let (filter_set, filter_control) = live_filter(TopicFilterSet::new());
        Self {
            subscriptions: KeyedFanOut::new(),
            filter_set,
            filter_control,
        }
    }

    /// SUBSCRIBE: register `sink` under `filter` (wildcard or exact).
    pub fn subscribe(&self, filter: &[u8], sink: PushSink) -> SubscriptionId {
        let id = self.subscriptions.subscribe(filter.to_vec(), sink);
        self.filter_control.update(|set| set.with(filter.to_vec()));
        id
    }

    /// UNSUBSCRIBE: remove one filter subscription, dropping the filter
    /// from the live match set once its last subscriber is gone.
    pub fn unsubscribe(&self, filter: &[u8], id: SubscriptionId) -> bool {
        let existed = self.subscriptions.unsubscribe(filter, id);
        if existed && self.subscriptions.subscription_count(filter) == 0 {
            self.filter_control.update(|set| set.without(filter));
        }
        existed
    }

    /// The number of connections currently subscribed to `filter`.
    #[must_use]
    pub fn subscription_count(&self, filter: &[u8]) -> usize {
        self.subscriptions.subscription_count(filter)
    }

    /// PUBLISH: deliver to every subscriber whose filter matches `topic`,
    /// as a QoS 0, retain-cleared `PUBLISH` frame (see module docs for the
    /// delivery-QoS scope note). Returns the total number of connections
    /// reached — the real MQTT broker's fan-out count (not part of the
    /// wire reply, but useful for logging/metrics call sites).
    pub async fn publish(&self, topic: &[u8], payload: &[u8]) -> Result<usize, ProximaError> {
        let matched_filters: Vec<Vec<u8>> = {
            let set = self.filter_set.snapshot();
            set.matching(topic).map(<[u8]>::to_vec).collect()
        };

        let mut reached = 0;
        for filter in &matched_filters {
            let count = self.subscriptions.subscription_count(filter);
            if count == 0 {
                continue;
            }
            let mut frame = Vec::new();
            encode_publish(topic, None, payload, 0, false, false, &mut frame);
            self.subscriptions
                .publish(filter, Bytes::from(frame))
                .await?;
            reached += count;
        }
        Ok(reached)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use futures::channel::mpsc;
    use proxima_protocols::mqtt::{Packet, parse_packet};

    fn sink() -> (PushSink, mpsc::UnboundedReceiver<Bytes>) {
        let (tx, rx) = mpsc::unbounded();
        (PushSink::new(tx), rx)
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_reaches_an_exact_filter_subscriber_as_a_publish_frame() {
        let broker = MqttBroker::new();
        let (push, mut rx) = sink();
        broker.subscribe(b"news", push);

        let reached = broker.publish(b"news", b"hello").await.expect("publish");

        assert_eq!(reached, 1);
        let bytes = rx.next().await.expect("push delivered");
        let (packet, used) = parse_packet(&bytes).expect("valid MQTT packet");
        assert_eq!(used, bytes.len());
        match packet {
            Packet::Publish { flags, topic, packet_id, payload } => {
                assert_eq!(flags.qos, 0);
                assert!(!flags.retain);
                assert_eq!(topic, b"news");
                assert!(packet_id.is_none());
                assert_eq!(payload, b"hello");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_reaches_a_matching_wildcard_subscriber() {
        let broker = MqttBroker::new();
        let (push, mut rx) = sink();
        broker.subscribe(b"news/#", push);

        let reached = broker
            .publish(b"news/tech", b"hi")
            .await
            .expect("publish");

        assert_eq!(reached, 1);
        let bytes = rx.next().await.expect("push delivered");
        let (packet, _) = parse_packet(&bytes).expect("valid MQTT packet");
        assert!(matches!(packet, Packet::Publish { topic: b"news/tech", .. }));
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_reaches_both_an_exact_and_a_wildcard_subscriber() {
        let broker = MqttBroker::new();
        let (exact_push, mut exact_rx) = sink();
        let (wildcard_push, mut wildcard_rx) = sink();
        broker.subscribe(b"news/tech", exact_push);
        broker.subscribe(b"news/#", wildcard_push);

        let reached = broker
            .publish(b"news/tech", b"both")
            .await
            .expect("publish");

        assert_eq!(reached, 2);
        assert!(exact_rx.next().await.is_some());
        assert!(wildcard_rx.next().await.is_some());
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_to_an_unsubscribed_topic_reaches_nobody() {
        let broker = MqttBroker::new();
        let reached = broker
            .publish(b"quiet", b"anyone?")
            .await
            .expect("publish is Ok even with no subscribers");
        assert_eq!(reached, 0);
    }

    #[proxima::test(runtime = "tokio")]
    async fn unsubscribe_stops_delivery_and_drops_the_filter_from_the_live_set() {
        let broker = MqttBroker::new();
        let (push, mut rx) = sink();
        let id = broker.subscribe(b"news/#", push);
        assert_eq!(broker.subscription_count(b"news/#"), 1);

        assert!(broker.unsubscribe(b"news/#", id));
        assert_eq!(broker.subscription_count(b"news/#"), 0);

        broker.publish(b"news/tech", b"gone").await.expect("publish");
        assert_eq!(rx.try_recv().ok(), None);
    }

    #[proxima::test(runtime = "tokio")]
    async fn two_subscribers_to_the_same_filter_both_receive_and_one_unsub_leaves_the_filter_live() {
        let broker = MqttBroker::new();
        let (first_push, mut first_rx) = sink();
        let (second_push, mut second_rx) = sink();
        let first_id = broker.subscribe(b"news/#", first_push);
        broker.subscribe(b"news/#", second_push);

        assert!(broker.unsubscribe(b"news/#", first_id));

        let reached = broker
            .publish(b"news/tech", b"still-live")
            .await
            .expect("publish");
        assert_eq!(reached, 1, "one subscriber remains on the shared filter");
        assert_eq!(first_rx.try_recv().ok(), None);
        assert!(second_rx.next().await.is_some());
    }
}
