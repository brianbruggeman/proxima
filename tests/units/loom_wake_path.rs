#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
// loom model tests for the Dekker-pattern wake-path fence pairs introduced
// in commit fe4fe6d ("fix(runtime-prime): SeqCst fence pairs close Dekker-
// pattern wake races").
//
// run with:
//   cargo test --features loom --test loom_wake_path --release
//
// to bound exploration time during CI:
//   LOOM_MAX_PREEMPTIONS=2 cargo test --features loom --test loom_wake_path --release
//
// note: do NOT use RUSTFLAGS="--cfg loom" — that applies the loom cfg to
// every crate in the dependency tree including tokio (which gates its net
// module out under loom) and will fail to compile. the `--features loom`
// approach activates loom only as a dev-dependency of this crate.
#![cfg(feature = "loom")]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicBool, Ordering, fence};
use loom::thread;

// model 1: reactor Wakeup::fire  ↔  core_shard worker_main arm_wakeup path.
//
// production sequence (reactor.rs + core_shard.rs):
//
//   producer (Wakeup::fire):
//     [prior] inbox tail.store(Release)      — push happened
//     fence(SeqCst)                          — the fix
//     needs_wake.load(Acquire)               — observe if worker parked
//     if true: needs_wake.swap(false, AcqRel) + fire_syscall()
//
//   worker (worker_main park section):
//     reactor.arm_wakeup()
//       → needs_wake.store(true, Release)    — arm
//     fence(SeqCst)                          — the fix
//     consumer.try_recv()                    — re-drain inbox
//     if empty: reactor.turn(timeout)        — park
//
// the Dekker invariant: after both threads pass the fence, at least one of
// the following holds:
//   (a) producer observed needs_wake = true  → fires syscall → worker wakes
//   (b) worker observed inbox non-empty      → worker drains → task runs
//
// we model the two stores + two fences + two loads. `work_seen` stands in for
// "worker successfully drained the inbox (case b)"; `fire_called` for "producer
// detected worker parked and fired wakeup (case a)". the assertion is: at least
// one of those outcomes is true for every interleaving — no lost-wake schedule.
#[test]
fn reactor_wake_no_lost_wake() {
    loom::model(|| {
        let needs_wake = Arc::new(AtomicBool::new(false));
        let wakeup_pending = Arc::new(AtomicBool::new(false));
        let fire_called = Arc::new(AtomicBool::new(false));
        let work_seen = Arc::new(AtomicBool::new(false));

        let needs_wake_producer = needs_wake.clone();
        let wakeup_pending_producer = wakeup_pending.clone();
        let fire_called_producer = fire_called.clone();

        let producer = thread::spawn(move || {
            // "push task into inbox" — modelled as arming the pending flag.
            wakeup_pending_producer.store(true, Ordering::Release);
            // the fix: SeqCst fence before loading needs_wake.
            fence(Ordering::SeqCst);
            // observe whether worker has armed for wakeup.
            if needs_wake_producer.load(Ordering::Acquire) {
                // worker was parked; fire the syscall (model: set flag).
                fire_called_producer.store(true, Ordering::Release);
            }
        });

        let needs_wake_worker = needs_wake.clone();
        let wakeup_pending_worker = wakeup_pending.clone();
        let work_seen_worker = work_seen.clone();

        let worker = thread::spawn(move || {
            // arm_wakeup: worker is about to park.
            needs_wake_worker.store(true, Ordering::Release);
            // the fix: SeqCst fence before re-draining inbox.
            fence(Ordering::SeqCst);
            // re-drain: load the inbox "tail" — modelled as loading the
            // wakeup_pending flag that the producer set on push.
            if wakeup_pending_worker.load(Ordering::Acquire) {
                // inbox had work; worker drains instead of parking.
                work_seen_worker.store(true, Ordering::Release);
            }
        });

        producer.join().unwrap();
        worker.join().unwrap();

        let fired = fire_called.load(Ordering::Acquire);
        let drained = work_seen.load(Ordering::Acquire);

        // Dekker guarantee: at least one side must have observed the other's
        // store. if both flags are false we have a lost-wake schedule.
        assert!(
            fired || drained,
            "lost-wake: producer did not fire and worker did not drain"
        );
    });
}

// model 2: inbox try_send_on  ↔  Recv::poll Dekker fence pair.
//
// production sequence (inbox.rs):
//
//   producer (try_send_on):
//     lane.tail.store(next, Release)          — publish item (via CAS)
//     fence(SeqCst)                           — the fix
//     consumer_parked.load(Acquire)           — observe if consumer parked
//     if true: consumer_parked.swap(false, AcqRel) + waker.wake()
//
//   consumer (Recv::poll):
//     waker.register(cx.waker())
//     consumer_parked.store(true, Release)    — mark as parked
//     fence(SeqCst)                           — the fix
//     try_recv()                              — re-drain lanes
//       → lane.tail.load(Acquire)            — observe published item
//     if non-empty: consumer_parked = false; return Ready
//     else: return Pending (will be woken by producer's waker.wake())
//
// the Dekker invariant: for every interleaving in which the producer completes
// its push and the consumer registers as parked, at least one of the following
// holds:
//   (a) producer observed consumer_parked = true → fires waker.wake()
//   (b) consumer observed lane non-empty         → returns Ready (no park)
//
// `waker_fired` stands in for (a); `item_seen` for (b).
#[test]
fn inbox_send_poll_no_lost_wake() {
    loom::model(|| {
        let consumer_parked = Arc::new(AtomicBool::new(false));
        let item_queued = Arc::new(AtomicBool::new(false));
        let waker_fired = Arc::new(AtomicBool::new(false));
        let item_seen = Arc::new(AtomicBool::new(false));

        let consumer_parked_producer = consumer_parked.clone();
        let item_queued_producer = item_queued.clone();
        let waker_fired_producer = waker_fired.clone();

        let producer = thread::spawn(move || {
            // "publish item to lane" — modelled as setting item_queued.
            item_queued_producer.store(true, Ordering::Release);
            // the fix: SeqCst fence before loading consumer_parked.
            fence(Ordering::SeqCst);
            // observe whether the consumer is parked.
            if consumer_parked_producer.load(Ordering::Acquire) {
                // consumer was parked; fire the waker (model: set flag).
                waker_fired_producer.store(true, Ordering::Release);
            }
        });

        let consumer_parked_consumer = consumer_parked.clone();
        let item_queued_consumer = item_queued.clone();
        let item_seen_consumer = item_seen.clone();

        let consumer = thread::spawn(move || {
            // register waker + mark parked.
            consumer_parked_consumer.store(true, Ordering::Release);
            // the fix: SeqCst fence before the re-drain try_recv.
            fence(Ordering::SeqCst);
            // re-drain: load the lane tail — modelled as loading item_queued.
            if item_queued_consumer.load(Ordering::Acquire) {
                // item is visible; consumer returns Ready and clears flag.
                consumer_parked_consumer.store(false, Ordering::Release);
                item_seen_consumer.store(true, Ordering::Release);
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();

        let fired = waker_fired.load(Ordering::Acquire);
        let drained = item_seen.load(Ordering::Acquire);

        // Dekker guarantee: at least one side observed the other's store.
        assert!(
            fired || drained,
            "lost-wake: producer did not fire waker and consumer did not see item"
        );
    });
}
