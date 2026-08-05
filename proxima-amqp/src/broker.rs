//! `AmqpBroker` — the exchange -> queue routing fabric shared across every
//! connection a [`crate::pipe::AmqpConnectionPipe`] upgrades.
//!
//! Mirrors `proxima_redis::broker::RedisBroker`'s role and shape: one
//! broker, shared by `Arc`, driven by the connection layer (the business
//! handler pipe never touches it directly). The consumer-delivery fabric
//! IS a [`KeyedFanOut`] — exactly redis's own primitive — keyed by *queue
//! name*: `basic.consume` subscribes a [`ConsumerSink`] under the queue
//! name (redis's `SUBSCRIBE channel`), and [`AmqpBroker::publish`]
//! broadcasts to every sink registered under a queue (redis's `PUBLISH`).
//!
//! What redis doesn't need and AMQP does: a message doesn't name its queue
//! directly — `basic.publish` names an *exchange* + *routing key*, and a
//! `queue.bind` earlier registered which queues that (exchange, key) pair
//! reaches. That indirection is plain bookkeeping (a small live map, not
//! sink/fan-out machinery), kept in [`AmqpBroker`]'s `exchanges` field:
//! [`AmqpBroker::publish`] resolves (exchange, routing_key) -> a set of
//! queue names first ([`ExchangeKind::routes`] is the whole difference
//! between direct exact-match, fanout-all-bound, and
//! [`topic_match`]-matched, mirroring redis's channel vs. pattern split),
//! then hands each matched queue name to the SAME `queues.publish` a bare
//! `basic.consume` on the default exchange would use directly.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use futures::channel::mpsc::UnboundedSender;

use proxima_core::ProximaError;
use proxima_core::live::{Live, LiveControl, live};
use proxima_primitives::pipe::{BestEffort, KeyedFanOut, SendPipe, SubscriptionId};

use crate::frame::{encode_body_frames, encode_header_frame, encode_method_frame};
use crate::method::{Method, id};
use crate::topic::topic_match;

/// The exchange kind a `exchange.declare` registered. The default exchange
/// (name `""`) is implicit `Direct` routing where the routing key IS the
/// queue name — it needs no `exchange.declare` and no `queue.bind` (AMQP
/// 0-9-1 §3.1.3.4), handled as a special case in [`AmqpBroker::publish`]
/// before consulting `exchanges` at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeKind {
    Direct,
    Fanout,
    Topic,
}

impl ExchangeKind {
    /// Parses the `exchange.declare` `type` field. `None` for a kind this
    /// broker does not implement (`headers` — see the crate-level gap
    /// notes).
    #[must_use]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        match bytes {
            b"direct" => Some(Self::Direct),
            b"fanout" => Some(Self::Fanout),
            b"topic" => Some(Self::Topic),
            _ => None,
        }
    }

    /// Does a binding registered under `binding_key` carry a message
    /// published with `routing_key`? This IS the difference between the
    /// three kinds — everything else about routing is identical.
    fn routes(self, binding_key: &[u8], routing_key: &[u8]) -> bool {
        match self {
            Self::Direct => binding_key == routing_key,
            Self::Fanout => true,
            Self::Topic => topic_match(binding_key, routing_key),
        }
    }
}

#[derive(Debug, Clone)]
struct Binding {
    routing_key: Vec<u8>,
    queue: Vec<u8>,
}

#[derive(Debug, Clone)]
struct Exchange {
    kind: ExchangeKind,
    bindings: Vec<Binding>,
}

type ExchangeMap = BTreeMap<Vec<u8>, Exchange>;

/// One reassembled `basic.publish` message handed to every matched queue's
/// [`ConsumerSink`]s — the fan-out item type, mirroring redis's `Bytes`
/// (pre-encoded wire frame) except AMQP needs the RAW fields here: each
/// sink encodes its OWN `basic.deliver` (consumer-tag + delivery-tag are
/// per-consumer, unlike redis where the channel name IS the addressing —
/// see [`ConsumerSink::call`]).
#[derive(Debug, Clone)]
pub struct Delivery {
    pub exchange: Vec<u8>,
    pub routing_key: Vec<u8>,
    pub properties: Vec<u8>,
    pub body: Vec<u8>,
}

/// A connection's consumer-side push-sink: encodes `basic.deliver` +
/// content-header + content-body frames addressed to ITS OWN channel and
/// consumer-tag, then pushes the encoded bytes onto the connection's
/// outbound-frame channel. Adapted to [`SendPipe`] so it composes as an
/// ordinary [`KeyedFanOut`] sink (workspace principle 1).
#[derive(Clone)]
pub struct ConsumerSink {
    channel: u16,
    consumer_tag: Vec<u8>,
    delivery_tag: Arc<AtomicU64>,
    sender: UnboundedSender<Bytes>,
    /// the `frame-max` this connection advertised in `connection.tune` —
    /// the whole-frame ceiling, envelope included, that
    /// [`encode_body_frames`] chunks the delivery body under.
    frame_max: usize,
}

impl ConsumerSink {
    #[must_use]
    pub fn new(
        channel: u16,
        consumer_tag: Vec<u8>,
        sender: UnboundedSender<Bytes>,
        frame_max: usize,
    ) -> Self {
        Self {
            channel,
            consumer_tag,
            delivery_tag: Arc::new(AtomicU64::new(1)),
            sender,
            frame_max,
        }
    }
}

impl SendPipe for ConsumerSink {
    type In = Delivery;
    type Out = ();
    type Err = ProximaError;

    fn call(&self, delivery: Delivery) -> impl Future<Output = Result<(), ProximaError>> + Send {
        let delivery_tag = self.delivery_tag.fetch_add(1, Ordering::Relaxed);
        let mut out = Vec::new();
        encode_method_frame(
            &mut out,
            self.channel,
            &Method::BasicDeliver {
                consumer_tag: self.consumer_tag.clone(),
                delivery_tag,
                redelivered: false,
                exchange: delivery.exchange,
                routing_key: delivery.routing_key,
            },
        );
        encode_header_frame(
            &mut out,
            self.channel,
            id::BASIC,
            delivery.body.len() as u64,
            &delivery.properties,
        );
        encode_body_frames(&mut out, self.channel, &delivery.body, self.frame_max);

        let result = self.sender.unbounded_send(Bytes::from(out));
        async move {
            result.map_err(|error| ProximaError::Upstream(format!("consumer sink closed: {error}")))
        }
    }
}

/// Process-wide exchange -> queue routing fabric. Construct once and share
/// via `Arc`.
pub struct AmqpBroker {
    queues: KeyedFanOut<ConsumerSink, BestEffort>,
    exchanges: Live<ExchangeMap>,
    exchanges_control: LiveControl<ExchangeMap>,
}

impl Default for AmqpBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl AmqpBroker {
    #[must_use]
    pub fn new() -> Self {
        let (exchanges, exchanges_control) = live(ExchangeMap::new());
        Self {
            queues: KeyedFanOut::new(),
            exchanges,
            exchanges_control,
        }
    }

    /// `exchange.declare`: registers `name` with `kind`. Declaring an
    /// already-known exchange with the SAME kind is idempotent (real
    /// clients redeclare on every connect); a kind mismatch is the
    /// `PRECONDITION_FAILED` case a caller renders as `channel.close`.
    pub fn declare_exchange(&self, name: Vec<u8>, kind: ExchangeKind) -> Result<(), ExchangeKind> {
        let existing = self
            .exchanges
            .read(|exchanges| exchanges.get(&name).map(|exchange| exchange.kind));
        if let Some(existing_kind) = existing {
            if existing_kind != kind {
                return Err(existing_kind);
            }
            return Ok(());
        }
        self.exchanges_control.update(|current| {
            let mut next = current.clone();
            next.insert(
                name.clone(),
                Exchange {
                    kind,
                    bindings: Vec::new(),
                },
            );
            next
        });
        Ok(())
    }

    /// `queue.bind`: binds `queue` under `routing_key` on `exchange`.
    /// Rebinding an identical `(queue, routing_key)` pair is a no-op —
    /// bindings are a set (AMQP 0-9-1 §1.4.2.3), and real clients redeclare
    /// and rebind on every reconnect, so appending unconditionally would
    /// both duplicate every delivery and grow the binding list forever.
    ///
    /// An unknown exchange is reported rather than leniently accepted, so
    /// the connection driver can render `channel.close` (`NOT_FOUND`) the
    /// way a real broker does.
    pub fn bind_queue(&self, exchange: &[u8], queue: Vec<u8>, routing_key: Vec<u8>) -> bool {
        let exists = self
            .exchanges
            .read(|exchanges| exchanges.contains_key(exchange));
        if !exists {
            return false;
        }
        self.exchanges_control.update(|current| {
            let mut next = current.clone();
            if let Some(entry) = next.get_mut(exchange) {
                let already_bound = entry
                    .bindings
                    .iter()
                    .any(|binding| binding.queue == queue && binding.routing_key == routing_key);
                if !already_bound {
                    entry.bindings.push(Binding {
                        routing_key: routing_key.clone(),
                        queue: queue.clone(),
                    });
                }
            }
            next
        });
        true
    }

    /// `basic.consume`: register `sink` under `queue` (a bare queue name —
    /// the SAME key the default exchange's implicit direct routing and a
    /// declared exchange's bound queues both publish to).
    #[must_use = "dropping the id leaks the subscription — only it can unsubscribe"]
    pub fn subscribe_queue(&self, queue: &[u8], sink: ConsumerSink) -> SubscriptionId {
        self.queues.subscribe(queue.to_vec(), sink)
    }

    /// Cancels one consumer registration (`basic.cancel`, channel close,
    /// or connection close).
    pub fn unsubscribe_queue(&self, queue: &[u8], id: SubscriptionId) -> bool {
        self.queues.unsubscribe(queue, id)
    }

    /// The number of consumers currently registered on `queue`.
    #[must_use]
    pub fn queue_consumer_count(&self, queue: &[u8]) -> usize {
        self.queues.subscription_count(queue)
    }

    /// `basic.publish`: routes `(exchange, routing_key)` to every matched
    /// queue's consumers. The default exchange (`""`) routes directly to
    /// the queue named `routing_key` — no `queue.bind` required (AMQP
    /// 0-9-1 §3.1.3.4). A named exchange resolves its bound queues by
    /// kind: `Direct` exact-matches the binding key, `Fanout` reaches
    /// every bound queue regardless of routing key, `Topic` glob-matches
    /// via [`TopicSet`]. Returns the number of consumers reached (a
    /// message with zero matched consumers, or matched queues with zero
    /// consumers, is a harmless no-op — mirrors redis `PUBLISH` to an
    /// unsubscribed channel).
    pub async fn publish(
        &self,
        exchange: &[u8],
        routing_key: &[u8],
        properties: Vec<u8>,
        body: Vec<u8>,
    ) -> Result<usize, ProximaError> {
        let queues = self.matched_queues(exchange, routing_key);
        let mut reached = 0;
        for queue in queues {
            let delivery = Delivery {
                exchange: exchange.to_vec(),
                routing_key: routing_key.to_vec(),
                properties: properties.clone(),
                body: body.clone(),
            };
            reached += self.queues.subscription_count(&queue);
            self.queues.publish(&queue, delivery).await?;
        }
        Ok(reached)
    }

    /// The DISTINCT queues `(exchange, routing_key)` reaches. Distinct is
    /// the point: a queue bound twice (two patterns that both match, or a
    /// fanout binding under two keys) is one destination, not two — AMQP
    /// bindings are a set, and a publisher expects one copy per queue.
    fn matched_queues(&self, exchange: &[u8], routing_key: &[u8]) -> Vec<Vec<u8>> {
        if exchange.is_empty() {
            return vec![routing_key.to_vec()];
        }
        self.exchanges.read(|exchanges| {
            let Some(entry) = exchanges.get(exchange) else {
                return Vec::new();
            };
            entry
                .bindings
                .iter()
                .filter(|binding| entry.kind.routes(&binding.routing_key, routing_key))
                .map(|binding| binding.queue.clone())
                .collect::<BTreeSet<Vec<u8>>>()
                .into_iter()
                .collect()
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use futures::channel::mpsc;

    fn sink(channel: u16, consumer_tag: &[u8]) -> (ConsumerSink, mpsc::UnboundedReceiver<Bytes>) {
        let (tx, rx) = mpsc::unbounded();
        (
            ConsumerSink::new(channel, consumer_tag.to_vec(), tx, 128 * 1024),
            rx,
        )
    }

    #[proxima::test(runtime = "tokio")]
    async fn default_exchange_routes_directly_to_the_queue_named_by_routing_key() {
        let broker = AmqpBroker::new();
        let (push, mut rx) = sink(1, b"ctag-1");
        let _ = broker.subscribe_queue(b"orders", push);

        let reached = broker
            .publish(b"", b"orders", Vec::new(), b"hello".to_vec())
            .await
            .expect("publish");

        assert_eq!(reached, 1);
        assert!(rx.next().await.is_some());
    }

    #[proxima::test(runtime = "tokio")]
    async fn direct_exchange_requires_an_exact_binding_key_match() {
        let broker = AmqpBroker::new();
        broker
            .declare_exchange(b"orders-x".to_vec(), ExchangeKind::Direct)
            .expect("declare");
        assert!(broker.bind_queue(b"orders-x", b"orders.eu".to_vec(), b"eu".to_vec()));

        let (push, mut rx) = sink(1, b"ctag-1");
        let _ = broker.subscribe_queue(b"orders.eu", push);

        let reached = broker
            .publish(b"orders-x", b"eu", Vec::new(), b"body".to_vec())
            .await
            .expect("publish");
        assert_eq!(reached, 1);
        assert!(rx.next().await.is_some());

        let missed = broker
            .publish(b"orders-x", b"us", Vec::new(), b"body".to_vec())
            .await
            .expect("publish");
        assert_eq!(missed, 0);
    }

    #[proxima::test(runtime = "tokio")]
    async fn fanout_exchange_reaches_every_bound_queue_regardless_of_routing_key() {
        let broker = AmqpBroker::new();
        broker
            .declare_exchange(b"broadcast".to_vec(), ExchangeKind::Fanout)
            .expect("declare");
        broker.bind_queue(b"broadcast", b"q1".to_vec(), Vec::new());
        broker.bind_queue(b"broadcast", b"q2".to_vec(), Vec::new());

        let (push1, mut rx1) = sink(1, b"c1");
        let (push2, mut rx2) = sink(2, b"c2");
        let _ = broker.subscribe_queue(b"q1", push1);
        let _ = broker.subscribe_queue(b"q2", push2);

        let reached = broker
            .publish(b"broadcast", b"ignored", Vec::new(), b"body".to_vec())
            .await
            .expect("publish");
        assert_eq!(reached, 2);
        assert!(rx1.next().await.is_some());
        assert!(rx2.next().await.is_some());
    }

    #[proxima::test(runtime = "tokio")]
    async fn topic_exchange_matches_wildcard_binding_keys() {
        let broker = AmqpBroker::new();
        broker
            .declare_exchange(b"events".to_vec(), ExchangeKind::Topic)
            .expect("declare");
        broker.bind_queue(b"events", b"eu-orders".to_vec(), b"orders.eu.*".to_vec());

        let (push, mut rx) = sink(1, b"c1");
        let _ = broker.subscribe_queue(b"eu-orders", push);

        let reached = broker
            .publish(
                b"events",
                b"orders.eu.created",
                Vec::new(),
                b"body".to_vec(),
            )
            .await
            .expect("publish");
        assert_eq!(reached, 1);
        assert!(rx.next().await.is_some());

        let missed = broker
            .publish(
                b"events",
                b"orders.us.created",
                Vec::new(),
                b"body".to_vec(),
            )
            .await
            .expect("publish");
        assert_eq!(missed, 0);
    }

    // real clients redeclare and rebind on every reconnect, so an
    // unconditional append grows the binding list — cloned in full on every
    // later declare/bind — without bound for the broker's whole lifetime.
    // Asserted on the list itself: nothing else observes it, because
    // routing already collapses duplicates.
    #[test]
    fn rebinding_an_identical_queue_and_key_registers_one_binding() {
        let broker = AmqpBroker::new();
        broker
            .declare_exchange(b"orders-x".to_vec(), ExchangeKind::Direct)
            .expect("declare");
        for _reconnect in 0..16 {
            assert!(broker.bind_queue(b"orders-x", b"orders.eu".to_vec(), b"eu".to_vec()));
        }
        assert!(broker.bind_queue(b"orders-x", b"orders.eu".to_vec(), b"us".to_vec()));

        let bindings = broker.exchanges.read(|exchanges| {
            exchanges
                .get(b"orders-x".as_slice())
                .map(|exchange| exchange.bindings.len())
        });
        assert_eq!(bindings, Some(2), "a distinct key still binds separately");
    }

    // two DISTINCT bindings may both name the same queue (two topic patterns
    // that both match, or a fanout binding under two keys) — the queue is
    // still one destination, so the publisher's one message stays one
    // message.
    #[proxima::test(runtime = "tokio")]
    async fn a_queue_two_matching_patterns_reach_still_receives_one_copy() {
        let broker = AmqpBroker::new();
        broker
            .declare_exchange(b"events".to_vec(), ExchangeKind::Topic)
            .expect("declare");
        broker.bind_queue(b"events", b"audit".to_vec(), b"orders.#".to_vec());
        broker.bind_queue(b"events", b"audit".to_vec(), b"orders.eu.*".to_vec());

        let (push, mut rx) = sink(1, b"ctag-1");
        let _ = broker.subscribe_queue(b"audit", push);

        let reached = broker
            .publish(
                b"events",
                b"orders.eu.created",
                Vec::new(),
                b"body".to_vec(),
            )
            .await
            .expect("publish");

        assert_eq!(reached, 1);
        assert!(rx.next().await.is_some());
        assert_eq!(rx.try_recv().ok(), None, "the message was delivered twice");
    }

    #[proxima::test(runtime = "tokio")]
    async fn declaring_the_same_exchange_with_a_mismatched_kind_is_rejected() {
        let broker = AmqpBroker::new();
        broker
            .declare_exchange(b"orders-x".to_vec(), ExchangeKind::Direct)
            .expect("first declare");
        let outcome = broker.declare_exchange(b"orders-x".to_vec(), ExchangeKind::Topic);
        assert_eq!(outcome, Err(ExchangeKind::Direct));
    }

    #[proxima::test(runtime = "tokio")]
    async fn unsubscribe_stops_delivery() {
        let broker = AmqpBroker::new();
        let (push, mut rx) = sink(1, b"c1");
        let id = broker.subscribe_queue(b"orders", push);

        assert!(broker.unsubscribe_queue(b"orders", id));
        broker
            .publish(b"", b"orders", Vec::new(), b"gone".to_vec())
            .await
            .expect("publish");
        assert_eq!(rx.try_recv().ok(), None);
    }

    // the delivery a consumer receives must fit the `frame-max` the broker
    // advertised in `connection.tune`; a body chunk sized at the raw
    // `frame-max` overshoots it by the 8-byte envelope and a conforming
    // client closes the connection with a frame error.
    #[proxima::test(runtime = "tokio")]
    async fn a_delivery_is_chunked_under_the_advertised_frame_max() {
        let frame_max = 4096;
        let (tx, mut rx) = mpsc::unbounded();
        let broker = AmqpBroker::new();
        let _ = broker.subscribe_queue(
            b"orders",
            ConsumerSink::new(1, b"ctag-1".to_vec(), tx, frame_max),
        );

        broker
            .publish(b"", b"orders", Vec::new(), vec![b'x'; 40_000])
            .await
            .expect("publish");

        let pushed = rx.next().await.expect("delivery");
        let mut cursor = &pushed[..];
        while !cursor.is_empty() {
            let (_frame, consumed) =
                proxima_protocols::amqp::parse_frame(cursor).expect("delivery frame");
            assert!(
                consumed <= frame_max,
                "pushed a {consumed}-byte frame past the {frame_max}-byte frame-max"
            );
            cursor = &cursor[consumed..];
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_to_an_unknown_exchange_reaches_nobody() {
        let broker = AmqpBroker::new();
        let reached = broker
            .publish(b"missing", b"anything", Vec::new(), b"body".to_vec())
            .await
            .expect("publish is Ok even against an unknown exchange");
        assert_eq!(reached, 0);
    }
}
