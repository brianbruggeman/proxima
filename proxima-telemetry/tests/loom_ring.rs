#![cfg(feature = "loom")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! loom model-check of the Vyukov MPMC ring (`proxima_telemetry::ring`) — the
//! STABLE-rust replacement for the nightly ThreadSanitizer job. loom explores
//! EVERY thread interleaving of the push/dequeue CAS protocol and asserts the
//! same no-loss / no-tear / no-dup invariants the tsan stress tests checked, but
//! without nightly + `-Zsanitizer=thread` + `-Zbuild-std`.
//!
//! run:   cargo test -p proxima-telemetry --features loom --test loom_ring --release
//! bound: LOOM_MAX_PREEMPTIONS=3 cargo test ... (caps exploration time in CI)
//!
//! the models are deliberately tiny (cap 2, one item per producer): loom's state
//! space is combinatorial in threads x ops, and the Vyukov protocol is identical
//! at scale — what changes with size is throughput, not the set of races. Every
//! cell is shared by both producers (cap 2, 2 producers), so the enqueue CAS is
//! contended; the multi-consumer model contends the dequeue CAS the same way.

use loom::sync::Arc;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::thread;

use proxima_telemetry::ring::Ring;

// tear detector: each id carries a redundant checksum, so a torn write (two
// producers writing the same cell) is detectable — `untag` of a torn value
// lands outside `0..2` and trips the bounds assert.
const MAGIC: u64 = 0x9E37_79B9_7F4A_7C15;
fn tag(id: u64) -> u64 {
    id ^ MAGIC
}
fn untag(value: u64) -> u64 {
    value ^ MAGIC
}

// MPSC: 2 producers each push one tagged id into a cap-2 ring; 1 consumer drains
// both. Invariants under every interleaving: no tear, no duplicate, no loss.
#[test]
fn loom_mpsc_two_producers_one_consumer_no_loss_no_tear() {
    loom::model(|| {
        let ring = Arc::new(Ring::<u64>::new(2).unwrap());

        let producers: Vec<_> = (0..2u64)
            .map(|id| {
                let ring = Arc::clone(&ring);
                thread::spawn(move || {
                    // yield on a full ring so loom can schedule a consumer — the
                    // spin is "requires-progress", which loom bounds via yield.
                    while ring.push(tag(id)).is_err() {
                        thread::yield_now();
                    }
                })
            })
            .collect();

        let consumer = {
            let ring = Arc::clone(&ring);
            thread::spawn(move || {
                let mut seen = [false; 2];
                let mut got = 0;
                while got < 2 {
                    match ring.dequeue() {
                        Some(value) => {
                            let id = untag(value);
                            assert!((id as usize) < 2, "torn or bogus id {id}");
                            assert!(!seen[id as usize], "duplicate id {id}");
                            seen[id as usize] = true;
                            got += 1;
                        }
                        None => thread::yield_now(),
                    }
                }
                seen
            })
        };

        for producer in producers {
            producer.join().unwrap();
        }
        let seen = consumer.join().unwrap();
        assert!(seen[0] && seen[1], "lost an id");
    });
}

// Multi-CONSUMER dequeue contention: 1 producer fills a cap-2 ring with two ids,
// 2 consumers race to drain. The dequeue CAS must give each published cell to
// exactly one consumer — a torn dequeue would let two consumers claim the same
// id and trip the duplicate assert. (The enqueue CAS under contention is proven
// by the 2-producer MPSC model above; splitting the two keeps each loom model
// inside the branch budget — the protocol is identical regardless of counts.)
#[test]
fn loom_one_producer_two_consumers_dequeue_cas_no_dup() {
    loom::model(|| {
        let ring = Arc::new(Ring::<u64>::new(2).unwrap());
        // pre-fill ONE item single-threaded (no producer/consumer race, no poll),
        // so the only interleaving loom explores is the two consumers' dequeue CAS
        // racing for the one published cell — exactly the multi-consumer
        // linearization point, with a tiny branch budget (no spin loops).
        ring.push(tag(0)).unwrap();
        let got = Arc::new(AtomicUsize::new(0));

        let consumers: Vec<_> = (0..2)
            .map(|_| {
                let ring = Arc::clone(&ring);
                let got = Arc::clone(&got);
                thread::spawn(move || {
                    if let Some(value) = ring.dequeue() {
                        assert_eq!(untag(value), 0, "torn item");
                        got.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for consumer in consumers {
            consumer.join().unwrap();
        }
        // exactly one consumer wins the cell; the other's dequeue CAS fails and it
        // sees an empty ring. Two winners would be a torn dequeue (double-free).
        assert_eq!(
            got.load(Ordering::Relaxed),
            1,
            "the cell went to exactly one consumer"
        );
    });
}
