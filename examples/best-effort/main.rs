//! `best-effort` — the composite guarantee: drop locally so a *presence*
//! guarantee holds globally. There is no `BestEffort` type in proxima; it is
//! `backpressure`'s lossy `BoundedQueue`/`FailMode` stage, pointed at a
//! producer that must never stall. `delivery`'s at-least-once/exactly-once
//! guarantees pay for zero loss with a retry loop the producer can block on
//! (`enqueue_assisting`, `send_until_acked`). Best-effort refuses that trade:
//! every producer call returns immediately — accepted or dropped, never
//! retried, never blocked — and the pipeline as a whole always makes
//! progress. This is `tracing`'s own model: a full telemetry buffer drops
//! events rather than block the request path. You lose completeness, you
//! never lose availability.
//!
//! Builds on: delivery.
//!
//! Run: `cargo run --example best-effort`

use proxima_primitives::pipe::{BoundedQueue, FailMode};

/// How many items the fast producer emits.
const PRODUCED: u32 = 20;

/// The lossy stage's capacity — deliberately small so a producer this much
/// faster than the consumer forces real drops, not a lucky near-miss.
const CAPACITY: usize = 4;

/// The slow consumer drains one item for every this-many produced — the
/// speed mismatch that makes the lossy stage necessary at all.
const DRAIN_EVERY: u32 = 5;

fn main() {
    println!("best-effort: drop locally, presence holds globally\n");
    run_best_effort();
}

fn run_best_effort() {
    let queue: BoundedQueue<u32> = BoundedQueue::new(CAPACITY, FailMode::DropOldest);
    let mut delivered = Vec::new();
    let mut enqueue_calls = 0_u32;

    for item in 0..PRODUCED {
        // availability: enqueue is O(1) and always returns — Enqueued or
        // DroppedOldest — never a retry loop, never a wait. Contrast with
        // delivery's send_until_acked, which loops until an ack lands.
        queue.enqueue(item);
        enqueue_calls += 1;

        let produced_so_far = item + 1;
        if produced_so_far.is_multiple_of(DRAIN_EVERY)
            && let Some(value) = queue.dequeue()
        {
            delivered.push(value);
        }
    }

    // the slow consumer catches up once the burst ends — the stream
    // completes, it does not stall waiting on a producer that already moved
    // on to the next item.
    while let Some(value) = queue.dequeue() {
        delivered.push(value);
    }

    assert_eq!(
        enqueue_calls, PRODUCED,
        "availability: one enqueue call per item, zero retries — the producer never blocked"
    );
    assert_eq!(
        queue.len(),
        0,
        "presence: the consumer drains to completion, the stream never stalls"
    );
    assert!(
        queue.dropped() > 0,
        "degradation: the slow consumer must force the lossy stage to actually drop something"
    );
    assert!(
        !delivered.is_empty(),
        "presence: something always survives to the far end"
    );
    assert_eq!(
        delivered.len() as u64 + queue.dropped(),
        u64::from(PRODUCED),
        "accounting closes: every produced item is either delivered or dropped, none vanish silently"
    );

    println!(
        "produced {PRODUCED}  delivered {}  dropped {}  (delivered + dropped = produced)",
        delivered.len(),
        queue.dropped()
    );
    println!("delivered, in order: {delivered:?}");

    println!(
        "\ncontrast with delivery: at-least-once/exactly-once retry until every message is \
         acked — zero loss, unbounded wait. best-effort refuses that trade: {enqueue_calls} \
         enqueue calls for {enqueue_calls} items, 0 retries, {} dropped locally — so the \
         pipeline as a whole is never not making progress.",
        queue.dropped()
    );
}
