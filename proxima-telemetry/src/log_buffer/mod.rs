//! Ring-buffered log capture with live-tail fanout. Folded in from the
//! former `proxima-log-buffer` satellite crate.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arc_swap::ArcSwap;
use crossbeam_queue::ArrayQueue;
use dashmap::DashMap;
use proxima_primitives::sync::Notify;

mod config;
pub(crate) mod ring;

pub use config::{LogBufferConfig, LogBufferLayerBuilder};
use ring::LogRing;

/// Build-time sizing constants generated from `proxima-log-buffer.toml`.
/// They seed [`LogBufferConfig`]'s runtime defaults — never duplicated —
/// and are the same values [`crate::log_buffer::ring::LogRing`] would use
/// as its no_std + alloc floor when constructed directly, without this
/// crate's runtime config layer.
pub mod sized {
    include!(concat!(env!("OUT_DIR"), "/proxima_log_buffer_sized.rs"));
}

/// Retained ring-buffer capacity (lines) for a [`LogBuffer`] created with no
/// explicit config. An alias for [`sized::LOG_BUFFER_CAPACITY_DEFAULT`] kept
/// for callers that construct `LogBuffer::new(DEFAULT_LOG_BUFFER_CAPACITY)`
/// directly; [`LogBufferConfig`] is the first-class, overridable surface.
pub const DEFAULT_LOG_BUFFER_CAPACITY: usize = sized::LOG_BUFFER_CAPACITY_DEFAULT;

/// Per-subscriber lock-free FIFO for the live-tail fanout. The buffer
/// pushes to every subscriber's queue via `try_push`; full queues drop
/// the new line (slow-subscriber backpressure). Receivers poll with
/// `try_recv` or block on a [`proxima_primitives::sync::Notify`] surfaced via the
/// `LiveTailReceiver` wrapper.
struct Subscriber {
    id: u64,
    queue: Arc<ArrayQueue<String>>,
    notify: Arc<Notify>,
}

/// Receiver handle for a live-tail subscription. `recv().await` parks
/// on a `Notify` until the next push, then drains via `try_recv`.
pub struct LiveTailReceiver {
    id: u64,
    queue: Arc<ArrayQueue<String>>,
    notify: Arc<Notify>,
    parent: Arc<ArcSwap<Vec<Subscriber>>>,
}

impl LiveTailReceiver {
    /// Pop the next live-tail line, or return `None` if the queue is
    /// currently empty. Does not block.
    pub fn try_recv(&self) -> Option<String> {
        self.queue.pop()
    }

    /// Block until a line arrives, then return it. If the parent
    /// `LogBuffer` is dropped, returns `None`.
    pub async fn recv(&self) -> Option<String> {
        loop {
            if let Some(line) = self.queue.pop() {
                return Some(line);
            }
            self.notify.notified().await;
        }
    }
}

impl Drop for LiveTailReceiver {
    fn drop(&mut self) {
        // unregister from the parent subscriber list
        let current = self.parent.load_full();
        let next: Vec<Subscriber> = current
            .iter()
            .filter(|entry| entry.id != self.id)
            .map(|entry| Subscriber {
                id: entry.id,
                queue: entry.queue.clone(),
                notify: entry.notify.clone(),
            })
            .collect();
        self.parent.store(Arc::new(next));
    }
}

/// Bounded ring buffer of stdout/stderr lines per supervised pipe,
/// plus a lock-free fanout to live-tail subscribers.
///
/// Implementation: the retained ring buffer is [`crate::ring::LogRing`]
/// (sans-IO, `push` evicts oldest when full — see that crate for the
/// no_std + alloc cliff this was carved for). Subscribers ride
/// `ArcSwap<Vec<Subscriber>>` with per-subscriber `ArrayQueue<String>` +
/// `Notify`. Read fanout is lock-free; subscription churn (register /
/// drop) is copy-on-write, fine because subscribe is rare relative to
/// push. Slow subscribers drop messages silently — log fanout is
/// best-effort (the retained ring buffer is the system of record for
/// late readers).
pub struct LogBuffer {
    ring: LogRing,
    subscribers: Arc<ArcSwap<Vec<Subscriber>>>,
    next_id: AtomicU64,
    live_tail_channel_capacity: usize,
}

impl LogBuffer {
    /// Construct with an explicit ring-buffer capacity; the live-tail channel
    /// capacity comes from [`LogBufferConfig`]'s default. Equivalent to
    /// `LogBuffer::from_config(&LogBufferConfig::builder().capacity(capacity).build())`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self::from_config(&LogBufferConfig::builder().capacity(capacity).build())
    }

    /// Construct from a [`LogBufferConfig`] — the config-driven constructor.
    /// Sizes both the retained ring buffer and every future subscriber's
    /// live-tail queue from the config, so a deployment tunes both via one
    /// config surface (file / env / fluent builder) rather than a bare arg.
    #[must_use]
    pub fn from_config(config: &LogBufferConfig) -> Self {
        Self {
            ring: LogRing::new(config.capacity),
            subscribers: Arc::new(ArcSwap::from_pointee(Vec::new())),
            next_id: AtomicU64::new(1),
            live_tail_channel_capacity: config.live_tail_channel_capacity.max(1),
        }
    }

    pub fn push(&self, line: String) {
        // one clone kept for the fanout loop below; the ring takes the rest.
        let fanout_line = line.clone();
        self.ring.push(line);
        // Lock-free fanout: snapshot the subscriber list, try_push to
        // each queue. Full queue = drop the new line for that subscriber
        // (slow-consumer policy; the retained ring buffer remains
        // authoritative).
        let snapshot = self.subscribers.load();
        for entry in snapshot.iter() {
            let _ = entry.queue.push(fanout_line.clone());
            entry.notify.notify_one();
        }
    }

    /// Oldest-first; `max_lines = None` returns everything retained.
    #[must_use]
    pub fn snapshot(&self, max_lines: Option<usize>) -> Vec<String> {
        self.ring.snapshot(max_lines)
    }

    /// Register a live-tail subscriber. Returns a `LiveTailReceiver`
    /// that yields lines pushed AFTER subscription; the retained ring
    /// buffer's existing contents are NOT replayed. Drop the receiver
    /// to unregister.
    #[must_use]
    pub fn subscribe(&self) -> LiveTailReceiver {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let queue = Arc::new(ArrayQueue::new(self.live_tail_channel_capacity));
        let notify = Arc::new(Notify::new());
        // copy-on-write append
        let current = self.subscribers.load_full();
        let mut next: Vec<Subscriber> = current
            .iter()
            .map(|entry| Subscriber {
                id: entry.id,
                queue: entry.queue.clone(),
                notify: entry.notify.clone(),
            })
            .collect();
        next.push(Subscriber {
            id,
            queue: queue.clone(),
            notify: notify.clone(),
        });
        self.subscribers.store(Arc::new(next));
        LiveTailReceiver {
            id,
            queue,
            notify,
            parent: self.subscribers.clone(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

/// Process-wide registry mapping pipe-name → LogBuffer. The
/// `process` upstream registers its buffer here on spawn so the
/// daemon control plane can serve `logs(name)` requests.
///
/// Sticks with `DashMap` (sharded RwLock — lock-free reads in the
/// common case; per-shard locks on write) rather than ArcSwap.
/// `per_core_vs_arcswap.rs` shows ArcSwap reads are 12-41ns and
/// writes do a full-map clone; DashMap has equivalent uncontended
/// read perf without the clone-per-write cost. This registry sees
/// rare register/deregister (subprocess spawn/exit) and rare reads
/// (`logs(name)` CLI calls), so the choice is moot at this scale —
/// but where the choice DOES matter, the substrate's per-core
/// thread-local pattern (10× faster than either) is the answer,
/// not ArcSwap-by-default.
#[derive(Default)]
pub struct LogBufferRegistry {
    buffers: DashMap<String, Arc<LogBuffer>>,
}

impl LogBufferRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffers: DashMap::new(),
        }
    }

    pub fn register(&self, name: impl Into<String>, buffer: Arc<LogBuffer>) {
        self.buffers.insert(name.into(), buffer);
    }

    pub fn deregister(&self, name: &str) -> Option<Arc<LogBuffer>> {
        self.buffers.remove(name).map(|(_, buffer)| buffer)
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<LogBuffer>> {
        self.buffers.get(name).map(|entry| entry.value().clone())
    }

    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.buffers
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn push_appends_until_capacity_then_evicts_oldest() {
        let buffer = LogBuffer::new(3);
        buffer.push("one".into());
        buffer.push("two".into());
        buffer.push("three".into());
        assert_eq!(buffer.len(), 3);
        buffer.push("four".into());
        assert_eq!(buffer.len(), 3, "capacity is 3");
        let lines = buffer.snapshot(None);
        assert_eq!(lines, vec!["two", "three", "four"]);
    }

    #[test]
    fn snapshot_with_max_returns_recent_n_in_order() {
        let buffer = LogBuffer::new(10);
        for index in 0..5 {
            buffer.push(format!("line-{index}"));
        }
        let recent = buffer.snapshot(Some(2));
        assert_eq!(recent, vec!["line-3", "line-4"]);
    }

    #[proxima::test]
    async fn subscribe_receives_new_lines() {
        let buffer = LogBuffer::new(8);
        let receiver = buffer.subscribe();
        buffer.push("hello".into());
        let received = receiver.recv().await.expect("recv");
        assert_eq!(received, "hello");
    }

    #[proxima::test]
    async fn dropping_receiver_unregisters_it() {
        let buffer = LogBuffer::new(8);
        let r1 = buffer.subscribe();
        let r2 = buffer.subscribe();
        assert_eq!(buffer.subscribers.load().len(), 2);
        drop(r2);
        // give the drop a moment (ArcSwap store is sync but the Vec
        // rebuild happens inside drop)
        tokio::task::yield_now().await;
        assert_eq!(buffer.subscribers.load().len(), 1);
        drop(r1);
        tokio::task::yield_now().await;
        assert_eq!(buffer.subscribers.load().len(), 0);
    }

    #[test]
    fn registry_deregister_handles_concurrent_writers() {
        // Tight loop to exercise the compare_and_swap retry path.
        let registry = std::sync::Arc::new(LogBufferRegistry::new());
        let mut handles = Vec::new();
        for index in 0..8 {
            let registry = registry.clone();
            handles.push(std::thread::spawn(move || {
                let name = format!("svc-{index}");
                registry.register(name.clone(), Arc::new(LogBuffer::new(4)));
                let _ = registry.deregister(&name);
            }));
        }
        for handle in handles {
            handle.join().expect("thread");
        }
        assert!(registry.names().is_empty());
    }

    #[test]
    fn registry_register_lookup_round_trip() {
        let registry = LogBufferRegistry::new();
        let buffer = Arc::new(LogBuffer::new(4));
        registry.register("svc", buffer.clone());
        let fetched = registry.get("svc").expect("registered");
        fetched.push("logged".into());
        assert_eq!(buffer.snapshot(None), vec!["logged"]);
    }

    #[test]
    fn registry_deregister_removes_entry() {
        let registry = LogBufferRegistry::new();
        registry.register("svc", Arc::new(LogBuffer::new(4)));
        let removed = registry.deregister("svc");
        assert!(removed.is_some());
        assert!(registry.get("svc").is_none());
    }
}
