//! `Batch<T>` — accumulate items, emit a full batch when a count threshold is
//! hit.
//!
//! The coalescing half of a buffered sink, factored out of the sink so any
//! caller composes it with its own terminal: recording's `AccumulatingSink`
//! buffers `RecordingEvent`s before a durable write; a telemetry exporter
//! buffers records before an OTLP flush. `Batch` owns only the buffer — it never
//! touches a sink, so it carries no `SendPipe` bound and no error type. `push`
//! hands back the full batch for the caller to flush; `drain` takes the partial
//! remainder on an explicit flush.
//!
//! Lock is poison-tolerant: a poisoned buffer only means a prior panic, and the
//! `Vec` is still sound to take, so a batcher never wedges on one bad appender.

use alloc::vec::Vec;
use std::sync::{Mutex, MutexGuard, PoisonError};

/// A count-thresholded accumulator. `Clone`-free, shareable behind an `Arc`.
pub struct Batch<T> {
    buffer: Mutex<Vec<T>>,
    batch_size: usize,
}

impl<T> Batch<T> {
    /// Coalesce into `batch_size`-item blocks (clamped to at least 1).
    #[must_use]
    pub fn new(batch_size: usize) -> Self {
        let batch_size = batch_size.max(1);
        Self {
            buffer: Mutex::new(Vec::with_capacity(batch_size)),
            batch_size,
        }
    }

    /// Push one item. Returns the full batch to flush once the buffer reaches
    /// the threshold (leaving the buffer empty), otherwise `None`.
    #[must_use = "the returned batch must be flushed to its sink"]
    pub fn push(&self, item: T) -> Option<Vec<T>> {
        let mut guard = self.lock();
        guard.push(item);
        if guard.len() >= self.batch_size {
            Some(core::mem::take(&mut *guard))
        } else {
            None
        }
    }

    /// Take whatever is buffered (an explicit flush); empties the buffer.
    #[must_use = "the drained batch must be flushed to its sink"]
    pub fn drain(&self) -> Vec<T> {
        core::mem::take(&mut *self.lock())
    }

    /// Items currently buffered (not yet flushed).
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// The flush threshold.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn lock(&self) -> MutexGuard<'_, Vec<T>> {
        self.buffer.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn push_below_threshold_buffers_and_returns_none() {
        let batch = Batch::new(4);
        assert!(batch.push(1).is_none());
        assert!(batch.push(2).is_none());
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn push_at_threshold_emits_full_batch_and_empties() {
        let batch = Batch::new(3);
        assert!(batch.push(10).is_none());
        assert!(batch.push(20).is_none());
        let full = batch.push(30).expect("threshold reached emits the batch");
        assert_eq!(full, alloc::vec![10, 20, 30]);
        assert!(batch.is_empty(), "buffer emptied after emitting");
    }

    #[test]
    fn drain_takes_the_partial_remainder() {
        let batch = Batch::new(8);
        let _ = batch.push(1);
        let _ = batch.push(2);
        assert_eq!(batch.drain(), alloc::vec![1, 2]);
        assert!(batch.is_empty());
        assert_eq!(
            batch.drain(),
            Vec::<i32>::new(),
            "draining empty yields nothing"
        );
    }

    #[test]
    fn zero_batch_size_clamps_to_one() {
        let batch = Batch::new(0);
        assert_eq!(batch.batch_size(), 1);
        assert_eq!(
            batch.push(7),
            Some(alloc::vec![7]),
            "every push flushes at size 1"
        );
    }
}
