//! `RedisBroker` — the PUBLISH/SUBSCRIBE/PSUBSCRIBE pub/sub fabric shared
//! across every connection a [`crate::pipe::RedisConnectionPipe`] upgrades.
//!
//! Mirrors `proxima_pgwire::broker::NotifyBroker`'s role (one broker, shared
//! by `Arc`, subscribed/unsubscribed/published by the driver — the business
//! handler never touches it), built on
//! [`proxima_primitives::pipe::KeyedFanOut`] instead of a hand-rolled
//! `DashMap<String, Vec<Subscriber>>`: exact-channel delivery IS a
//! `KeyedFanOut` keyed by the channel name (SUBSCRIBE); pattern delivery is
//! a second `KeyedFanOut` keyed by the raw pattern string (PSUBSCRIBE), with
//! a [`LiveFilter<GlobSet>`] snapshot of "every pattern currently
//! registered" kept in sync so [`RedisBroker::publish`] finds the matching
//! pattern keys with one wait-free read instead of a glob-unaware scan of
//! the pattern registry's own key list.
//!
//! [`RedisBroker::publish`] frames the wire message itself (`message` for
//! exact-channel delivery, `pmessage` — carrying the matched pattern too —
//! for pattern delivery), so every sink only ever pushes already-framed
//! wire bytes; the connection driver's `select!` writes them straight
//! through with no further encoding.

use core::future::Future;

use bytes::Bytes;
use futures::channel::mpsc::UnboundedSender;

use proxima_core::ProximaError;
use proxima_primitives::pipe::{
    BestEffort, FilterControl, KeyedFanOut, LiveFilter, SendPipe, SubscriptionId, live_filter,
};
use proxima_protocols::redis::{Frame, encode};

use crate::glob::GlobSet;

/// A connection's push-sink: the sender half of its outbound-frame channel,
/// adapted to [`SendPipe`] so it composes as an ordinary
/// [`KeyedFanOut`]/[`FanOut`](proxima_primitives::pipe::FanOut) sink
/// (workspace principle 1 — "a sink is an ordinary pipe", no bespoke sink
/// trait). The connection driver's `select!` races the paired receiver
/// against the socket read.
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
pub struct RedisBroker {
    channels: KeyedFanOut<PushSink, BestEffort>,
    patterns: KeyedFanOut<PushSink, BestEffort>,
    /// Sharded pub/sub (`SSUBSCRIBE`/`SPUBLISH`) exact-channel fabric — a
    /// `KeyedFanOut` entirely separate from `channels`: real Redis (7.0+)
    /// keeps the two namespaces distinct, so an `SPUBLISH` never reaches a
    /// `SUBSCRIBE`'d connection and vice versa, even though proxima is
    /// single-node (the shard IS the node; only the namespace is what real
    /// Redis requires to stay distinct).
    shard_channels: KeyedFanOut<PushSink, BestEffort>,
    /// Live glob-set mirror of `patterns`'s registered keys, kept in sync by
    /// `subscribe_pattern`/`unsubscribe_pattern` — lets `publish` find the
    /// matching pattern keys with one wait-free read instead of iterating
    /// `patterns.keys()` and glob-testing each on every PUBLISH.
    pattern_set: LiveFilter<GlobSet>,
    pattern_control: FilterControl<GlobSet>,
}

impl Default for RedisBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl RedisBroker {
    #[must_use]
    pub fn new() -> Self {
        let (pattern_set, pattern_control) = live_filter(GlobSet::new());
        Self {
            channels: KeyedFanOut::new(),
            patterns: KeyedFanOut::new(),
            shard_channels: KeyedFanOut::new(),
            pattern_set,
            pattern_control,
        }
    }

    /// SUBSCRIBE: register `sink` under the exact channel name.
    pub fn subscribe_channel(&self, channel: &[u8], sink: PushSink) -> SubscriptionId {
        self.channels.subscribe(channel.to_vec(), sink)
    }

    /// UNSUBSCRIBE: remove one channel subscription.
    pub fn unsubscribe_channel(&self, channel: &[u8], id: SubscriptionId) -> bool {
        self.channels.unsubscribe(channel, id)
    }

    /// The number of connections currently subscribed to `channel`.
    #[must_use]
    pub fn channel_subscriber_count(&self, channel: &[u8]) -> usize {
        self.channels.subscription_count(channel)
    }

    /// PSUBSCRIBE: register `sink` under `pattern`, and — if this is the
    /// first subscriber for that exact pattern string — add it to the live
    /// glob set `publish` scans.
    pub fn subscribe_pattern(&self, pattern: &[u8], sink: PushSink) -> SubscriptionId {
        let id = self.patterns.subscribe(pattern.to_vec(), sink);
        self.pattern_control.update(|set| set.with(pattern.to_vec()));
        id
    }

    /// PUNSUBSCRIBE: remove one pattern subscription, dropping the pattern
    /// from the live glob set once its last subscriber is gone.
    pub fn unsubscribe_pattern(&self, pattern: &[u8], id: SubscriptionId) -> bool {
        let existed = self.patterns.unsubscribe(pattern, id);
        if existed && self.patterns.subscription_count(pattern) == 0 {
            self.pattern_control.update(|set| set.without(pattern));
        }
        existed
    }

    /// The number of connections currently subscribed to `pattern`.
    #[must_use]
    pub fn pattern_subscriber_count(&self, pattern: &[u8]) -> usize {
        self.patterns.subscription_count(pattern)
    }

    /// SSUBSCRIBE: register `sink` under the exact shard-channel name — a
    /// namespace distinct from [`Self::subscribe_channel`].
    pub fn subscribe_shard_channel(&self, channel: &[u8], sink: PushSink) -> SubscriptionId {
        self.shard_channels.subscribe(channel.to_vec(), sink)
    }

    /// SUNSUBSCRIBE: remove one shard-channel subscription.
    pub fn unsubscribe_shard_channel(&self, channel: &[u8], id: SubscriptionId) -> bool {
        self.shard_channels.unsubscribe(channel, id)
    }

    /// The number of connections currently `SSUBSCRIBE`d to `channel`.
    #[must_use]
    pub fn shard_channel_subscriber_count(&self, channel: &[u8]) -> usize {
        self.shard_channels.subscription_count(channel)
    }

    /// SPUBLISH: deliver to every shard-channel subscriber of `channel` (an
    /// `smessage` frame) — never to a regular `channels`/`patterns`
    /// subscriber, and never reached by a regular [`Self::publish`]. Returns
    /// the number of connections the message reached, the real Redis
    /// SPUBLISH reply.
    pub async fn publish_shard(&self, channel: &[u8], payload: &[u8]) -> Result<usize, ProximaError> {
        let count = self.shard_channels.subscription_count(channel);
        if count > 0 {
            let frame = Frame::Array(vec![
                Frame::BlobString(b"smessage"),
                Frame::BlobString(channel),
                Frame::BlobString(payload),
            ]);
            self.shard_channels
                .publish(channel, Bytes::from(encode(&frame)))
                .await?;
        }
        Ok(count)
    }

    /// PUBLISH: deliver to every exact-channel subscriber (a `message`
    /// frame) via [`KeyedFanOut::publish`] — a
    /// `FanOut::best_effort(...).call(...)` broadcast — then to every
    /// pattern subscriber whose pattern glob-matches `channel` (a
    /// `pmessage` frame carrying the matched pattern). Returns the total
    /// number of connections the message reached — the real Redis PUBLISH
    /// reply.
    pub async fn publish(&self, channel: &[u8], payload: &[u8]) -> Result<usize, ProximaError> {
        let exact_count = self.channels.subscription_count(channel);
        if exact_count > 0 {
            let frame = Frame::Array(vec![
                Frame::BlobString(b"message"),
                Frame::BlobString(channel),
                Frame::BlobString(payload),
            ]);
            self.channels
                .publish(channel, Bytes::from(encode(&frame)))
                .await?;
        }

        let matched_patterns: Vec<Vec<u8>> = {
            let set = self.pattern_set.snapshot();
            set.matching(channel).map(<[u8]>::to_vec).collect()
        };
        let mut pattern_count = 0;
        for pattern in &matched_patterns {
            let count = self.patterns.subscription_count(pattern);
            if count == 0 {
                continue;
            }
            pattern_count += count;
            let frame = Frame::Array(vec![
                Frame::BlobString(b"pmessage"),
                Frame::BlobString(pattern),
                Frame::BlobString(channel),
                Frame::BlobString(payload),
            ]);
            self.patterns
                .publish(pattern, Bytes::from(encode(&frame)))
                .await?;
        }
        Ok(exact_count + pattern_count)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use futures::channel::mpsc;
    use proxima_protocols::redis::parse;

    fn sink() -> (PushSink, mpsc::UnboundedReceiver<Bytes>) {
        let (tx, rx) = mpsc::unbounded();
        (PushSink::new(tx), rx)
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_reaches_an_exact_channel_subscriber_as_a_message_frame() {
        let broker = RedisBroker::new();
        let (push, mut rx) = sink();
        broker.subscribe_channel(b"news", push);

        let reached = broker.publish(b"news", b"hello").await.expect("publish");

        assert_eq!(reached, 1);
        let bytes = rx.next().await.expect("push delivered");
        let (frame, _) = parse(&bytes).expect("valid RESP frame");
        assert_eq!(
            frame,
            Frame::Array(vec![
                Frame::BlobString(b"message"),
                Frame::BlobString(b"news"),
                Frame::BlobString(b"hello"),
            ])
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_reaches_a_matching_pattern_subscriber_as_a_pmessage_frame() {
        let broker = RedisBroker::new();
        let (push, mut rx) = sink();
        broker.subscribe_pattern(b"news.*", push);

        let reached = broker
            .publish(b"news.tech", b"hi")
            .await
            .expect("publish");

        assert_eq!(reached, 1);
        let bytes = rx.next().await.expect("push delivered");
        let (frame, _) = parse(&bytes).expect("valid RESP frame");
        assert_eq!(
            frame,
            Frame::Array(vec![
                Frame::BlobString(b"pmessage"),
                Frame::BlobString(b"news.*"),
                Frame::BlobString(b"news.tech"),
                Frame::BlobString(b"hi"),
            ])
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_reaches_both_exact_and_pattern_subscribers() {
        let broker = RedisBroker::new();
        let (exact_push, mut exact_rx) = sink();
        let (pattern_push, mut pattern_rx) = sink();
        broker.subscribe_channel(b"news.tech", exact_push);
        broker.subscribe_pattern(b"news.*", pattern_push);

        let reached = broker
            .publish(b"news.tech", b"both")
            .await
            .expect("publish");

        assert_eq!(reached, 2);
        assert!(exact_rx.next().await.is_some());
        assert!(pattern_rx.next().await.is_some());
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_to_an_unsubscribed_channel_reaches_nobody() {
        let broker = RedisBroker::new();
        let reached = broker
            .publish(b"quiet", b"anyone?")
            .await
            .expect("publish is Ok even with no subscribers");
        assert_eq!(reached, 0);
    }

    #[proxima::test(runtime = "tokio")]
    async fn unsubscribe_channel_stops_delivery() {
        let broker = RedisBroker::new();
        let (push, mut rx) = sink();
        let id = broker.subscribe_channel(b"news", push);

        assert!(broker.unsubscribe_channel(b"news", id));
        broker.publish(b"news", b"gone").await.expect("publish");

        assert_eq!(rx.try_recv().ok(), None);
    }

    #[proxima::test(runtime = "tokio")]
    async fn unsubscribe_pattern_removes_it_from_the_live_glob_set() {
        let broker = RedisBroker::new();
        let (push, mut rx) = sink();
        let id = broker.subscribe_pattern(b"news.*", push);
        assert_eq!(broker.pattern_subscriber_count(b"news.*"), 1);

        assert!(broker.unsubscribe_pattern(b"news.*", id));
        assert_eq!(broker.pattern_subscriber_count(b"news.*"), 0);

        broker
            .publish(b"news.tech", b"gone")
            .await
            .expect("publish");
        assert_eq!(rx.try_recv().ok(), None);
    }

    #[proxima::test(runtime = "tokio")]
    async fn publish_shard_reaches_an_ssubscribe_subscriber_as_an_smessage_frame() {
        let broker = RedisBroker::new();
        let (push, mut rx) = sink();
        broker.subscribe_shard_channel(b"orders", push);

        let reached = broker
            .publish_shard(b"orders", b"shipped")
            .await
            .expect("publish_shard");

        assert_eq!(reached, 1);
        let bytes = rx.next().await.expect("push delivered");
        let (frame, _) = parse(&bytes).expect("valid RESP frame");
        assert_eq!(
            frame,
            Frame::Array(vec![
                Frame::BlobString(b"smessage"),
                Frame::BlobString(b"orders"),
                Frame::BlobString(b"shipped"),
            ])
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn shard_and_regular_channel_namespaces_never_cross() {
        let broker = RedisBroker::new();
        let (regular_push, mut regular_rx) = sink();
        let (shard_push, mut shard_rx) = sink();
        broker.subscribe_channel(b"orders", regular_push);
        broker.subscribe_shard_channel(b"orders", shard_push);

        let shard_reached = broker
            .publish_shard(b"orders", b"only-shard")
            .await
            .expect("publish_shard");
        assert_eq!(shard_reached, 1, "SPUBLISH reaches only the SSUBSCRIBE side");
        assert!(shard_rx.next().await.is_some());
        assert_eq!(
            regular_rx.try_recv().ok(),
            None,
            "SPUBLISH must not reach a SUBSCRIBE subscriber on the same channel name"
        );

        let regular_reached = broker.publish(b"orders", b"only-regular").await.expect("publish");
        assert_eq!(regular_reached, 1, "PUBLISH reaches only the SUBSCRIBE side");
        assert!(regular_rx.next().await.is_some());
        assert_eq!(
            shard_rx.try_recv().ok(),
            None,
            "PUBLISH must not reach an SSUBSCRIBE subscriber on the same channel name"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn unsubscribe_shard_channel_stops_delivery() {
        let broker = RedisBroker::new();
        let (push, mut rx) = sink();
        let id = broker.subscribe_shard_channel(b"orders", push);

        assert!(broker.unsubscribe_shard_channel(b"orders", id));
        broker.publish_shard(b"orders", b"gone").await.expect("publish_shard");

        assert_eq!(rx.try_recv().ok(), None);
    }

    #[proxima::test(runtime = "tokio")]
    async fn two_subscribers_to_the_same_pattern_both_receive_and_one_unsub_leaves_the_pattern_live() {
        let broker = RedisBroker::new();
        let (first_push, mut first_rx) = sink();
        let (second_push, mut second_rx) = sink();
        let first_id = broker.subscribe_pattern(b"news.*", first_push);
        broker.subscribe_pattern(b"news.*", second_push);

        assert!(broker.unsubscribe_pattern(b"news.*", first_id));

        let reached = broker
            .publish(b"news.tech", b"still-live")
            .await
            .expect("publish");
        assert_eq!(reached, 1, "one subscriber remains on the shared pattern");
        assert_eq!(first_rx.try_recv().ok(), None);
        assert!(second_rx.next().await.is_some());
    }
}
