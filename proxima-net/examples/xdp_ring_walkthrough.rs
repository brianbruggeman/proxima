//! A runnable walkthrough of the AF_XDP SPSC ring index protocol —
//! [`ProducerIndex`] / [`ConsumerIndex`], the pure arithmetic behind the
//! FILL/RX/TX/COMPLETION rings — with no socket and no `xdp` feature, so it
//! runs on any target. Read it (and run it) to learn:
//!   - producer: `reserve(n)` hands out the first of `n` contiguous slots,
//!     `commit()` returns the counter to publish to the kernel;
//!   - consumer: `peek(n)` returns the first slot + how many are ready,
//!     `release(n)` returns the counter to publish (peek is non-destructive);
//!   - `slot()` masks the free-running `u32` counter into the ring array index;
//!   - the counters keep climbing past the ring size — the slots wrap.
//!
//! `cargo run -p proxima-net-xdp --example xdp_ring_walkthrough`

use proxima_net::xdp::{ConsumerIndex, ProducerIndex};

fn main() {
    const SIZE: u32 = 8; // a small ring so the wrap is visible in a few steps

    // ── arrange ──────────────────────────────────────────────────────────────
    // `producer` is our side of a ring we FILL (fill/tx); `consumer` models the
    // side that DRAINS it (rx/completion). Both track only cached counters.
    let Ok(mut producer) = ProducerIndex::new(SIZE) else {
        eprintln!("ring size must be a non-zero power of two");
        return;
    };
    let Ok(mut consumer) = ConsumerIndex::new(SIZE) else {
        eprintln!("ring size must be a non-zero power of two");
        return;
    };
    println!("ring size = {SIZE} (a power of two: the index math masks with size-1)\n");

    // ── act 1: the producer reserves 5 slots on an empty ring ────────────────
    // reserve(want, live_consumer): live_consumer is the peer's shared counter,
    // consulted only if the cached view looks full. On an empty ring it grants.
    let Some(start) = producer.reserve(5, 0) else {
        eprintln!("unexpected: an empty ring rejected 5 slots");
        return;
    };
    println!("act 1  producer.reserve(5) -> first slot index {start}");
    assert_eq!(
        start, 0,
        "the first reservation on a fresh ring starts at index 0"
    );
    assert_eq!(producer.slot(start), 0);
    assert_eq!(
        producer.slot(start + 4),
        4,
        "5 contiguous slots occupy 0..=4"
    );
    let published = producer.commit();
    println!("       producer.commit() -> publish producer counter = {published}");
    assert_eq!(published, 5, "the counter advanced by the 5 we reserved");

    // ── act 2: the consumer drains exactly what was published ────────────────
    let (peeked, ready) = consumer.peek(8, published);
    println!("act 2  consumer.peek(8, live_producer={published}) -> start {peeked}, ready {ready}");
    assert_eq!(
        (peeked, ready),
        (0, 5),
        "it sees exactly the 5 the producer published"
    );
    let released = consumer.release(ready);
    println!("       consumer.release({ready}) -> publish consumer counter = {released}");
    assert_eq!(released, 5);
    assert_eq!(
        consumer.peek(8, published),
        (5, 0),
        "peek was non-destructive; now drained"
    );

    // ── act 3: fill past the ring size — counters climb, slots WRAP ──────────
    let Some(start) = producer.reserve(6, released) else {
        eprintln!("unexpected: a drained ring rejected 6 slots");
        return;
    };
    println!("act 3  producer.reserve(6, live_consumer={released}) -> first slot index {start}");
    assert_eq!(start, 5, "the free-running counter keeps climbing to 5");
    assert_eq!(
        producer.slot(start),
        5,
        "index 5 -> array slot 5 (still in range)"
    );
    assert_eq!(
        producer.slot(start + 2),
        7,
        "index 7 -> array slot 7 (last before the wrap)"
    );
    assert_eq!(
        producer.slot(start + 3),
        0,
        "index 8 -> array slot 0 (WRAPS to the head)"
    );
    assert_eq!(
        producer.slot(start + 5),
        2,
        "index 10 -> array slot 2 (continues from the head)"
    );
    println!(
        "       counter index {} masks to array slot {} — the wrap",
        start + 3,
        producer.slot(start + 3)
    );
    assert_eq!(
        producer.commit(),
        11,
        "the producer counter is now 11, past the ring size 8"
    );

    // ── act 4: a full ring applies backpressure until the peer drains ────────
    let (start, ready) = consumer.peek(SIZE, producer.commit());
    assert_eq!(
        (start, ready),
        (5, 6),
        "the 6 frames from act 3 are ready to drain"
    );
    let drained = consumer.release(ready);
    assert_eq!(drained, 11);
    let Some(full_start) = producer.reserve(SIZE, drained) else {
        eprintln!("unexpected: a drained ring rejected a full-ring reservation");
        return;
    };
    println!("act 4  producer.reserve({SIZE}) after drain -> first slot index {full_start}");
    assert_eq!(full_start, 11, "the counter keeps climbing to 11");
    assert!(
        producer.reserve(1, drained).is_none(),
        "a FULL ring rejects further reservations until the consumer counter advances"
    );
    println!("       a full ring rejects one more reservation — SPSC backpressure holds");

    println!("\nOK: reserve/commit + peek/release + slot-wrap + backpressure all hold.");
}
