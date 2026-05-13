//! Sans-IO bounded log-line ring — capacity-bounded push with
//! oldest-line eviction, oldest-first snapshot draining.
//!
//! Folded in from the former `proxima-log-buffer-core` satellite crate
//! (single consumer: this crate). Pure MPMC queue logic over
//! `crossbeam_queue::ArrayQueue` — no async, no OS, no subscriber fanout.
//! Live-tail subscriptions and the process-wide registry stay in the rest
//! of this crate, which layers them on top of this ring.
//!
//! This module is alloc-only by construction (no `std::`-specific APIs),
//! matching the no_std + alloc floor the former satellite crate targeted —
//! this crate itself stays unconditionally std by design (see
//! `config.rs`'s tier note: `dashmap` and `arc-swap`'s no_std paths aren't
//! viable for the subscriber fanout this crate also carries).

use crossbeam_queue::ArrayQueue;

/// Capacity-bounded ring of retained items; the oldest item is evicted
/// once the ring is full. Concurrency-safe (MPMC) via
/// [`crossbeam_queue::ArrayQueue`] — no locking, no async, no OS calls.
///
/// Generic over the element `T`, defaulting to `String` so the live-tail
/// log path (`LogBuffer`) is unchanged. The elevation path reuses it as
/// `LogRing<LogRecord>` — a per-trace replay ring — rather than minting a
/// second bounded-ring primitive.
pub struct LogRing<T = String> {
    lines: ArrayQueue<T>,
}

impl<T> LogRing<T> {
    /// Construct with the given retained-item capacity, clamped to at
    /// least 1 (a zero-capacity ring could never retain anything).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            lines: ArrayQueue::new(capacity.max(1)),
        }
    }

    /// Append an item, evicting the oldest retained item first if the
    /// ring is already at capacity.
    pub fn push(&self, item: T) {
        if let Err(rejected) = self.lines.push(item) {
            let _ = self.lines.pop();
            let _ = self.lines.push(rejected);
        }
    }

    /// Number of items currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// True when no items are currently retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

impl<T: Clone> LogRing<T> {
    /// Oldest-first; `max_lines = None` returns everything retained.
    /// Drains the queue and re-pushes — under concurrent `push` this
    /// briefly empties the ring, then refills; acceptable for log
    /// inspection (rare, not on the hot path).
    #[must_use]
    pub fn snapshot(&self, max_lines: Option<usize>) -> Vec<T> {
        let mut drained: Vec<T> = Vec::with_capacity(self.lines.len());
        while let Some(item) = self.lines.pop() {
            drained.push(item);
        }
        for item in &drained {
            let _ = self.lines.push(item.clone());
        }
        let count = max_lines.unwrap_or(drained.len()).min(drained.len());
        let start = drained.len().saturating_sub(count);
        drained[start..].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_appends_until_capacity_then_evicts_oldest() {
        let ring = LogRing::new(3);
        ring.push("one".to_string());
        ring.push("two".to_string());
        ring.push("three".to_string());
        assert_eq!(ring.len(), 3);
        ring.push("four".to_string());
        assert_eq!(ring.len(), 3, "capacity is 3");
        let lines = ring.snapshot(None);
        assert_eq!(lines, vec!["two", "three", "four"]);
    }

    #[test]
    fn snapshot_with_max_returns_recent_n_in_order() {
        let ring = LogRing::new(10);
        for index in 0..5 {
            ring.push(format!("line-{index}"));
        }
        let recent = ring.snapshot(Some(2));
        assert_eq!(recent, vec!["line-3", "line-4"]);
    }

    #[test]
    fn snapshot_none_returns_everything_retained() {
        let ring = LogRing::new(10);
        ring.push("a".to_string());
        ring.push("b".to_string());
        assert_eq!(ring.snapshot(None), vec!["a", "b"]);
    }

    #[test]
    fn zero_capacity_clamps_to_one() {
        let ring = LogRing::new(0);
        ring.push("only".to_string());
        ring.push("replaced".to_string());
        assert_eq!(ring.len(), 1);
        assert_eq!(ring.snapshot(None), vec!["replaced"]);
    }

    #[test]
    fn empty_ring_reports_empty() {
        let ring: LogRing<String> = LogRing::new(4);
        assert!(ring.is_empty());
        assert_eq!(ring.snapshot(None), Vec::<String>::new());
    }

    #[test]
    fn snapshot_leaves_ring_intact_for_repeat_reads() {
        let ring = LogRing::new(4);
        ring.push("kept".to_string());
        assert_eq!(ring.snapshot(None), vec!["kept"]);
        assert_eq!(
            ring.snapshot(None),
            vec!["kept"],
            "snapshot is non-destructive"
        );
    }
}
