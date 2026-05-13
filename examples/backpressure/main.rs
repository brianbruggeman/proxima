//! Backpressure — the strategy space for a producer that outruns its consumer,
//! built entirely from pipe-algebra primitives already in the tree. There is no
//! `Backpressure` type here: each strategy below is a few lines composing
//! [`BoundedQueue`], a plain sampling predicate, [`Demand`]/[`AtomicGate`],
//! [`BatchSource`], [`Batch`], or [`Live`] — with an assertion proving its
//! distinct behavior.
//!
//! Builds on: gate (`Demand`/`AtomicGate` dormancy), fan-in (`BatchSource`
//! pull-mode draining of the same `BoundedQueue`).
//!
//! Run:
//!     cargo run --example backpressure

use core::convert::Infallible;
use core::future::Future;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

use proxima_core::live::{Live, LiveControl, live};
use proxima_primitives::pipe::{BatchSource, SendPipe};
use proxima_primitives::pipe::{AtomicGate, Batch, BoundedQueue, Demand, EnqueueOutcome, FailMode};

#[proxima::main(cores = 1)]
async fn main() {
    block_strategy();
    drop_newest_strategy();
    drop_oldest_strategy();
    sample_strategy();
    credit_strategy();
    demand_strategy().await;
    batch_strategy();
    coalesce_strategy();
    println!("all backpressure strategies verified");
}

// BLOCK: the producer retries against a full queue until the consumer makes
// room. Nothing is lost; the producer pays the wait instead.
fn block_strategy() {
    let queue: BoundedQueue<u32> = BoundedQueue::new(2, FailMode::FailClosed);
    assert_eq!(queue.enqueue(1), EnqueueOutcome::Enqueued);
    assert_eq!(queue.enqueue(2), EnqueueOutcome::Enqueued);

    let mut retries = 0;
    let landed = queue.enqueue_assisting(3, || {
        retries += 1;
        queue.dequeue().is_some()
    });

    assert!(landed.is_ok(), "3 lands once the consumer frees a slot");
    assert_eq!(retries, 1, "one make-room round before landing");
    assert_eq!(
        queue.dequeue(),
        Some(2),
        "1 was drained to make room; 2 still precedes 3"
    );
    assert_eq!(queue.dequeue(), Some(3));
    assert_eq!(queue.dropped(), 0, "block never drops, it waits");

    println!("block:       producer retried {retries} time(s) for room, 0 dropped");
}

// DROP-NEWEST: a full queue simply refuses the arriving item.
fn drop_newest_strategy() {
    let queue: BoundedQueue<u32> = BoundedQueue::new(2, FailMode::DropNewest);
    assert_eq!(queue.enqueue(1), EnqueueOutcome::Enqueued);
    assert_eq!(queue.enqueue(2), EnqueueOutcome::Enqueued);
    assert_eq!(
        queue.enqueue(3),
        EnqueueOutcome::DroppedNewest,
        "3 arrives after the queue is full; it is the one discarded"
    );
    assert_eq!(queue.dropped(), 1);
    assert_eq!(
        queue.dequeue(),
        Some(1),
        "the two oldest items are untouched"
    );
    assert_eq!(queue.dequeue(), Some(2));

    println!("drop-newest: kept [1, 2], dropped 3 (arrival order preserved)");
}

// DROP-OLDEST: a full queue evicts what it already holds to make room.
fn drop_oldest_strategy() {
    let queue: BoundedQueue<u32> = BoundedQueue::new(2, FailMode::DropOldest);
    assert_eq!(queue.enqueue(1), EnqueueOutcome::Enqueued);
    assert_eq!(queue.enqueue(2), EnqueueOutcome::Enqueued);
    assert_eq!(
        queue.enqueue(3),
        EnqueueOutcome::DroppedOldest,
        "queue is full; 1 is evicted to make room for 3"
    );
    assert_eq!(queue.dropped(), 1);
    assert_eq!(
        queue.dequeue(),
        Some(2),
        "1 was evicted; the freshest pair remains"
    );
    assert_eq!(queue.dequeue(), Some(3));

    println!("drop-oldest: kept [2, 3], dropped 1 (freshest pair survives)");
}

// SAMPLE: keep 1-in-N, drop the rest. This is a plain query, not a filter
// stage: the item never flows through `admits` (a plain `std::iter::Filter`
// closure asks it and keeps the item itself), so it stays a bare predicate
// method rather than a decision pipe (see `filter`, which admits an item
// THROUGH a pipe stage instead of just asking about it).
#[derive(Clone)]
struct EveryNth {
    period: u32,
    calls: Arc<AtomicU32>,
}

impl EveryNth {
    fn admits(&self, _item: &u32) -> bool {
        let index = self.calls.fetch_add(1, Ordering::Relaxed);
        index.is_multiple_of(self.period)
    }
}

fn sample_strategy() {
    let sampler = EveryNth {
        period: 3,
        calls: Arc::new(AtomicU32::new(0)),
    };
    let produced: Vec<u32> = (0..9).collect();
    let kept: Vec<u32> = produced
        .iter()
        .copied()
        .filter(|item| sampler.admits(item))
        .collect();

    assert_eq!(
        kept,
        vec![0, 3, 6],
        "keeps exactly 1-in-3, drops the rest under load"
    );

    println!(
        "sample:      kept {kept:?} of {} produced (1-in-3)",
        produced.len()
    );
}

// CREDIT: the consumer pulls only as many items as it has room for; the
// backlog stays queued until more credit is granted.
fn credit_strategy() {
    let queue: BoundedQueue<u32> = BoundedQueue::new(8, FailMode::DropNewest);
    for value in 0..5u32 {
        assert_eq!(queue.enqueue(value), EnqueueOutcome::Enqueued);
    }

    let credit = 2;
    let mut out = vec![0u32; credit];
    let pulled = BatchSource::drain_batch(&queue, &mut out);

    assert_eq!(
        pulled, credit,
        "consumer pulls exactly its granted credit, never the backlog"
    );
    assert_eq!(&out[..pulled], &[0, 1]);
    assert_eq!(
        queue.len(),
        3,
        "ungranted items stay queued until more credit is issued"
    );

    println!(
        "credit:      pulled {pulled} of {} queued (credit={credit})",
        pulled + queue.len()
    );
}

// DEMAND: while no consumer has signaled readiness, the producer's sends are
// absorbed as a no-op — not queued, not dropped, never dispatched.
struct RecordSink(Arc<AtomicUsize>);

impl SendPipe for RecordSink {
    type In = u32;
    type Out = ();
    type Err = Infallible;

    fn call(&self, _item: u32) -> impl Future<Output = Result<(), Infallible>> + Send {
        let delivered = Arc::clone(&self.0);
        async move {
            delivered.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }
}

async fn demand_strategy() {
    let delivered = Arc::new(AtomicUsize::new(0));
    let (gate, controller) = AtomicGate::pair(false);
    let demand_pipe = Demand::new(RecordSink(Arc::clone(&delivered)), gate);

    for value in 0..5u32 {
        SendPipe::call(&demand_pipe, value)
            .await
            .unwrap_or_else(|never: Infallible| match never {});
    }
    assert_eq!(
        delivered.load(Ordering::Relaxed),
        0,
        "dormant: sends are absorbed, not queued"
    );

    controller.arm();
    for value in 5..8u32 {
        SendPipe::call(&demand_pipe, value)
            .await
            .unwrap_or_else(|never: Infallible| match never {});
    }
    assert_eq!(
        delivered.load(Ordering::Relaxed),
        3,
        "delivered only once demand is armed"
    );

    println!("demand:      0 delivered while dormant, 3 delivered after controller.arm()");
}

// BATCH: accumulate items, flush once as a block instead of one write per item.
fn batch_strategy() {
    let batch: Batch<u32> = Batch::new(3);
    assert!(batch.push(1).is_none());
    assert!(batch.push(2).is_none());
    let flushed = batch.push(3);

    assert_eq!(
        flushed,
        Some(vec![1, 2, 3]),
        "threshold reached: one flush, not three"
    );
    assert!(batch.is_empty(), "buffer emptied after emitting");

    println!("batch:       3 pushes coalesced into 1 flush of [1, 2, 3]");
}

// COALESCE: repeated writes overwrite in place; the consumer reads whatever is
// latest, and the superseded values leave no trace.
fn coalesce_strategy() {
    let (reader, control): (Live<u32>, LiveControl<u32>) = live(0u32);
    for value in 1..=5u32 {
        control.replace(value);
    }
    let latest = reader.snapshot();

    assert_eq!(
        *latest, 5,
        "only the newest value survives; four superseded writes vanish"
    );

    println!("coalesce:    5 writes collapsed to the latest value ({latest})");
}
