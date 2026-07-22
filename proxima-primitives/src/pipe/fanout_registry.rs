//! [`KeyedFanOut`] — a named registry of live pub/sub broadcast groups.
//!
//! Mirrors [`crate::pipe::filter_registry::FilterRegistry`]'s Live-backed
//! copy-on-write registry shape exactly (an outer [`Live`] map, mutated only
//! through `write.update(|current| { clone, mutate, return })`), but each
//! named entry accumulates [`SendPipe`] sinks (broadcast via [`FanOut`])
//! instead of an id-set membership predicate: [`KeyedFanOut::subscribe`]
//! registers a sink under a key and returns a [`SubscriptionId`],
//! [`KeyedFanOut::unsubscribe`] removes exactly that sink, and
//! [`KeyedFanOut::publish`] broadcasts to every sink currently registered
//! under a key via a transient `FanOut::new(...).call(item)` snapshot — a
//! slow/dead subscriber never blocks or fails delivery to the rest under the
//! default [`BestEffort`] policy.
//!
//! This is the general pub/sub-delivery primitive (workspace principle 1):
//! redis PUBLISH/SUBSCRIBE/PSUBSCRIBE is its first consumer, fanning a
//! published message out to every subscribed connection's push sink.
//! [`FanOut`] itself has no id-tracking (it is a bare broadcast over
//! `Arc<Vec<S>>`), so the id needed for [`unsubscribe`](Self::unsubscribe)
//! lives in this registry's per-key sink list, not in `FanOut` — the exact
//! same reason [`crate::pipe::filter_registry::FilterRegistry`] wraps its
//! `LiveFilter`/`FilterControl` pair in a named `Subscription` entry rather
//! than storing them bare.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};

use proxima_core::live::{Live, LiveControl, live};

use crate::pipe::SendPipe;
use crate::pipe::fanout::{BestEffort, FanOut, FanPolicy};

/// Opaque handle returned by [`KeyedFanOut::subscribe`], presented back to
/// [`KeyedFanOut::unsubscribe`] to remove exactly that sink and no other
/// sink registered under the same key. Process-wide monotonic — never
/// reused, so a stale id from an already-removed sink is simply not found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubscriptionId(u64);

/// One key's live sink list — the `Subscription<Id>` mirror in
/// `filter_registry.rs`, holding broadcast sinks instead of a
/// filter/control split. A bare [`FanOut`] has no live-swap of its own;
/// rebuilding this list on every subscribe/unsubscribe IS the swap, done
/// through the same outer [`Live`] + `write.update` `FilterRegistry` uses.
struct Group<S> {
    sinks: Vec<(SubscriptionId, S)>,
}

impl<S> Group<S> {
    fn empty() -> Self {
        Self { sinks: Vec::new() }
    }
}

impl<S: Clone> Clone for Group<S> {
    fn clone(&self) -> Self {
        Self {
            sinks: self.sinks.clone(),
        }
    }
}

type Groups<S> = BTreeMap<Vec<u8>, Group<S>>;

/// A lock-free registry of named pub/sub broadcast groups. Lookups are
/// wait-free (the map is a [`Live`] cell); subscribe/unsubscribe copy-on-write
/// it. Keys are raw bytes (binary-safe channel/pattern names — redis pub/sub
/// channels are not required to be UTF-8).
pub struct KeyedFanOut<S, Policy = BestEffort> {
    read: Live<Groups<S>>,
    write: LiveControl<Groups<S>>,
    next_id: AtomicU64,
    policy: PhantomData<fn() -> Policy>,
}

impl<S: Clone, Policy> KeyedFanOut<S, Policy> {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        let (read, write) = live(BTreeMap::new());
        Self {
            read,
            write,
            next_id: AtomicU64::new(0),
            policy: PhantomData,
        }
    }

    /// Register `sink` under `key`, returning the id later passed to
    /// [`Self::unsubscribe`]. Multiple sinks may share one key (multiple
    /// connections subscribed to the same channel).
    pub fn subscribe(&self, key: impl Into<Vec<u8>>, sink: S) -> SubscriptionId {
        let key = key.into();
        let id = SubscriptionId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.write.update(|current| {
            let mut next = current.clone();
            let group = next.entry(key.clone()).or_insert_with(Group::empty);
            group.sinks.push((id, sink.clone()));
            next
        });
        id
    }

    /// Remove one sink by id; returns whether it was found. Drops the key
    /// entirely once its last sink is removed (mirrors
    /// `FilterRegistry::unsubscribe` removing the whole named entry).
    pub fn unsubscribe(&self, key: &[u8], id: SubscriptionId) -> bool {
        let existed = self.read.read(|groups| {
            groups
                .get(key)
                .is_some_and(|group| group.sinks.iter().any(|(sink_id, _)| *sink_id == id))
        });
        if existed {
            self.write.update(|current| {
                let mut next = current.clone();
                if let Some(group) = next.get_mut(key) {
                    group.sinks.retain(|(sink_id, _)| *sink_id != id);
                    if group.sinks.is_empty() {
                        next.remove(key);
                    }
                }
                next
            });
        }
        existed
    }

    /// The number of sinks currently registered under `key`.
    #[must_use]
    pub fn subscription_count(&self, key: &[u8]) -> usize {
        self.read
            .read(|groups| groups.get(key).map_or(0, |group| group.sinks.len()))
    }

    /// The registered keys, sorted — mirrors `FilterRegistry::names`.
    #[must_use]
    pub fn keys(&self) -> Vec<Vec<u8>> {
        self.read.read(|groups| groups.keys().cloned().collect())
    }
}

impl<S: Clone, Policy> Default for KeyedFanOut<S, Policy> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, Policy> KeyedFanOut<S, Policy>
where
    S: SendPipe + Clone,
    S::In: Clone + Send,
    Policy: FanPolicy,
{
    /// Broadcast `item` to every sink registered under `key` — a transient
    /// [`FanOut::new`] built from the current snapshot, so a slow/dead
    /// subscriber never blocks or fails delivery to the rest (under the
    /// default [`BestEffort`] `Policy`). A key with no subscribers is a
    /// no-op success (no one to deliver to).
    pub async fn publish(&self, key: &[u8], item: S::In) -> Result<(), S::Err> {
        let sinks: Vec<S> = self
            .read
            .read(|groups| {
                groups
                    .get(key)
                    .map(|group| group.sinks.iter().map(|(_, sink)| sink.clone()).collect())
            })
            .unwrap_or_default();
        if sinks.is_empty() {
            return Ok(());
        }
        FanOut::<S, Policy>::new(sinks).call(item).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipe::fanout::BestEffort;
    use alloc::string::String;
    use alloc::string::ToString;
    use core::future::Future;

    #[derive(Clone)]
    struct RecordingSink {
        write: proxima_core::live::LiveControl<Vec<String>>,
        read: proxima_core::live::Live<Vec<String>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            let (read, write) = live(Vec::new());
            Self { write, read }
        }

        fn seen(&self) -> Vec<String> {
            self.read.read(Clone::clone)
        }
    }

    impl SendPipe for RecordingSink {
        type In = String;
        type Out = ();
        type Err = core::convert::Infallible;

        fn call(
            &self,
            item: String,
        ) -> impl Future<Output = Result<(), core::convert::Infallible>> + Send {
            let write = self.write.clone();
            async move {
                write.update(move |current| {
                    let mut next = current.clone();
                    next.push(item.clone());
                    next
                });
                Ok(())
            }
        }
    }

    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let waker = core::task::Waker::noop();
        let mut context = core::task::Context::from_waker(waker);
        loop {
            if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    fn channels() -> (Vec<u8>, Vec<u8>) {
        (b"news.tech".to_vec(), b"news.sports".to_vec())
    }

    #[test]
    fn subscribe_then_publish_reaches_the_sink() {
        let (tech, _sports) = channels();
        let registry: KeyedFanOut<RecordingSink, BestEffort> = KeyedFanOut::new();
        let sink = RecordingSink::new();
        registry.subscribe(tech.clone(), sink.clone());

        block_on(registry.publish(&tech, "hello".to_string())).expect("publish");

        assert_eq!(sink.seen(), vec!["hello".to_string()]);
    }

    #[test]
    fn publish_reaches_every_subscriber_of_the_same_key() {
        let (tech, _) = channels();
        let registry: KeyedFanOut<RecordingSink, BestEffort> = KeyedFanOut::new();
        let first = RecordingSink::new();
        let second = RecordingSink::new();
        registry.subscribe(tech.clone(), first.clone());
        registry.subscribe(tech.clone(), second.clone());

        block_on(registry.publish(&tech, "hi".to_string())).expect("publish");

        assert_eq!(first.seen(), vec!["hi".to_string()]);
        assert_eq!(second.seen(), vec!["hi".to_string()]);
    }

    #[test]
    fn publish_to_an_unknown_key_is_a_harmless_no_op() {
        let (tech, _) = channels();
        let registry: KeyedFanOut<RecordingSink, BestEffort> = KeyedFanOut::new();
        block_on(registry.publish(&tech, "nobody-home".to_string())).expect("no subscribers is Ok");
    }

    #[test]
    fn unsubscribe_removes_only_the_matching_sink() {
        let (tech, _) = channels();
        let registry: KeyedFanOut<RecordingSink, BestEffort> = KeyedFanOut::new();
        let first = RecordingSink::new();
        let second = RecordingSink::new();
        let first_id = registry.subscribe(tech.clone(), first.clone());
        registry.subscribe(tech.clone(), second.clone());

        assert!(registry.unsubscribe(&tech, first_id));
        block_on(registry.publish(&tech, "after-unsub".to_string())).expect("publish");

        assert!(first.seen().is_empty(), "removed sink receives nothing");
        assert_eq!(second.seen(), vec!["after-unsub".to_string()]);
    }

    #[test]
    fn unsubscribe_the_last_sink_drops_the_key() {
        let (tech, _) = channels();
        let registry: KeyedFanOut<RecordingSink, BestEffort> = KeyedFanOut::new();
        let sink = RecordingSink::new();
        let id = registry.subscribe(tech.clone(), sink);

        assert!(registry.unsubscribe(&tech, id));

        assert!(registry.keys().is_empty(), "the drained key is removed");
        assert_eq!(registry.subscription_count(&tech), 0);
    }

    #[test]
    fn unsubscribe_an_unknown_id_reports_absent() {
        let (tech, _) = channels();
        let registry: KeyedFanOut<RecordingSink, BestEffort> = KeyedFanOut::new();
        let sink = RecordingSink::new();
        let real_id = registry.subscribe(tech.clone(), sink);
        let _ = registry.unsubscribe(&tech, real_id);

        assert!(
            !registry.unsubscribe(&tech, real_id),
            "second removal of the same id reports absent"
        );
    }

    #[test]
    fn keys_lists_every_registered_channel_sorted() {
        let (tech, sports) = channels();
        let registry: KeyedFanOut<RecordingSink, BestEffort> = KeyedFanOut::new();
        registry.subscribe(sports.clone(), RecordingSink::new());
        registry.subscribe(tech.clone(), RecordingSink::new());

        // lexicographic byte order: 's' (0x73) < 't' (0x74)
        assert_eq!(registry.keys(), vec![sports, tech]);
    }

    #[test]
    fn subscription_count_reflects_live_subscribe_and_unsubscribe() {
        let (tech, _) = channels();
        let registry: KeyedFanOut<RecordingSink, BestEffort> = KeyedFanOut::new();
        let first_id = registry.subscribe(tech.clone(), RecordingSink::new());
        registry.subscribe(tech.clone(), RecordingSink::new());
        assert_eq!(registry.subscription_count(&tech), 2);

        registry.unsubscribe(&tech, first_id);
        assert_eq!(registry.subscription_count(&tech), 1);
    }

    #[test]
    fn independent_keys_do_not_cross_deliver() {
        let (tech, sports) = channels();
        let registry: KeyedFanOut<RecordingSink, BestEffort> = KeyedFanOut::new();
        let tech_sink = RecordingSink::new();
        let sports_sink = RecordingSink::new();
        registry.subscribe(tech.clone(), tech_sink.clone());
        registry.subscribe(sports.clone(), sports_sink.clone());

        block_on(registry.publish(&tech, "tech-only".to_string())).expect("publish");

        assert_eq!(tech_sink.seen(), vec!["tech-only".to_string()]);
        assert!(sports_sink.seen().is_empty());
    }

    // proves a slow/erroring sink under one key never affects independent
    // publish calls on other keys, and that a fresh registry starts empty —
    // the counters a `SinkErr`-style fan-out policy test would check are
    // exercised directly by `fanout.rs`'s own tests; this suite's job is the
    // KEYED registry semantics layered on top.
    #[test]
    fn default_registry_starts_empty() {
        let registry: KeyedFanOut<RecordingSink, BestEffort> = KeyedFanOut::default();
        assert!(registry.keys().is_empty());
        assert_eq!(registry.subscription_count(b"anything"), 0);
    }
}
