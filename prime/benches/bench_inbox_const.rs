//! home-turf bench for DCa — `runtime-prime-inbox-const` (SPSC ring).
//!
//! **Named incumbents and design points:**
//!
//! 1. `heapless::spsc::Queue<T, CAP>` — `design-favors: incumbent`
//!    Stack-backed SPSC queue; this is the canonical no_alloc SPSC incumbent.
//!    Design point: single-producer / single-consumer, stack-allocated ring.
//!    Their bench workload: produce N items, consume N items. CAP = 16/256/4096.
//!
//! 2. `embassy_sync::channel::Channel<NoopRawMutex, T, CAP>` — `design-favors: incumbent`
//!    Bare-metal no_alloc async channel. Design point: embedded bare-metal
//!    targets with async wake semantics. Same SPSC workload as heapless.
//!
//! 3. inbox-alloc SPSC arm — `design-favors: proxima`
//!    Our own heap-backed `channel::<T>(1, CAP)`. Documents the overhead
//!    of the const-generic constraint vs the heap-backed version.
//!
//! **Workload:** synchronous round-trip (no async runtime). Each bench arm
//! pushes `ITEMS` values through the ring, consuming each immediately after.
//! Thread-safe only for the SPSC contract — producer and consumer are on the
//! same thread (no cross-thread sends). This is the design point all three
//! incumbents were optimized for.
//!
//! **Multiple input sizes:** CAP = 16 (small embedded), 256 (mid), 4096 (large).
//! The bench sweeps all three to expose where fixed overhead (setup amortized
//! over `ITEMS`) vs per-item cost dominates.

#[allow(unused_extern_crates)]
extern crate alloc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::channel::Channel;
use heapless::spsc::Queue as HeaplessQueue;
use prime::core::inbox::{channel, inbox_const::Inbox as ConstInbox};

const ITEMS: usize = 100_000;

fn bench_const_inbox<const CAP: usize>(criterion: &mut Criterion, label: &str) {
    criterion.bench_with_input(
        BenchmarkId::new("inbox_const_spsc", label),
        &CAP,
        |bench, _cap| {
            bench.iter(|| {
                let mut inbox: ConstInbox<u64, CAP> = ConstInbox::new();
                let (producer, consumer) = inbox.split();
                let mut count = 0usize;
                let mut index = 0u64;
                while count < ITEMS {
                    let batch = CAP.min(ITEMS - count);
                    let mut pushed = 0;
                    while pushed < batch {
                        if producer.try_send(index).is_ok() {
                            index += 1;
                            pushed += 1;
                        } else {
                            break;
                        }
                    }
                    let mut drained = 0;
                    while drained < pushed {
                        if consumer.try_recv().is_ok() {
                            drained += 1;
                            count += 1;
                        } else {
                            break;
                        }
                    }
                }
                count
            });
        },
    );
}

fn bench_heapless_spsc<const CAP: usize>(criterion: &mut Criterion, label: &str) {
    criterion.bench_with_input(
        BenchmarkId::new("heapless_spsc", label),
        &CAP,
        |bench, _cap| {
            bench.iter(|| {
                let mut queue: HeaplessQueue<u64, CAP> = HeaplessQueue::new();
                let (mut producer, mut consumer) = queue.split();
                let mut count = 0usize;
                let mut index = 0u64;
                while count < ITEMS {
                    let batch = CAP.min(ITEMS - count);
                    let mut pushed = 0;
                    while pushed < batch {
                        if producer.enqueue(index).is_ok() {
                            index += 1;
                            pushed += 1;
                        } else {
                            break;
                        }
                    }
                    let mut drained = 0;
                    while drained < pushed {
                        if consumer.dequeue().is_some() {
                            drained += 1;
                            count += 1;
                        } else {
                            break;
                        }
                    }
                }
                count
            });
        },
    );
}

fn bench_embassy_sync<const CAP: usize>(criterion: &mut Criterion, label: &str) {
    criterion.bench_with_input(
        BenchmarkId::new("embassy_sync_channel", label),
        &CAP,
        |bench, _cap| {
            bench.iter(|| {
                let channel: Channel<NoopRawMutex, u64, CAP> = Channel::new();
                let sender = channel.sender();
                let receiver = channel.receiver();
                let mut count = 0usize;
                let mut index = 0u64;
                while count < ITEMS {
                    let batch = CAP.min(ITEMS - count);
                    let mut pushed = 0;
                    while pushed < batch {
                        if sender.try_send(index).is_ok() {
                            index += 1;
                            pushed += 1;
                        } else {
                            break;
                        }
                    }
                    let mut drained = 0;
                    while drained < pushed {
                        if receiver.try_receive().is_ok() {
                            drained += 1;
                            count += 1;
                        } else {
                            break;
                        }
                    }
                }
                count
            });
        },
    );
}

fn bench_inbox_alloc_spsc<const CAP: usize>(criterion: &mut Criterion, label: &str) {
    criterion.bench_with_input(
        BenchmarkId::new("inbox_alloc_spsc", label),
        &CAP,
        |bench, _cap| {
            bench.iter(|| {
                let (producer, consumer) = channel::<u64>(1, CAP);
                let mut count = 0usize;
                let mut index = 0u64;
                while count < ITEMS {
                    let batch = CAP.min(ITEMS - count);
                    let mut pushed = 0;
                    while pushed < batch {
                        if producer.try_send(index).is_ok() {
                            index += 1;
                            pushed += 1;
                        } else {
                            break;
                        }
                    }
                    let mut drained = 0;
                    while drained < pushed {
                        if consumer.try_recv().is_ok() {
                            drained += 1;
                            count += 1;
                        } else {
                            break;
                        }
                    }
                }
                count
            });
        },
    );
}

fn bench_all(criterion: &mut Criterion) {
    bench_const_inbox::<16>(criterion, "cap16");
    bench_const_inbox::<256>(criterion, "cap256");
    bench_const_inbox::<4096>(criterion, "cap4096");

    bench_heapless_spsc::<16>(criterion, "cap16");
    bench_heapless_spsc::<256>(criterion, "cap256");
    bench_heapless_spsc::<4096>(criterion, "cap4096");

    bench_embassy_sync::<16>(criterion, "cap16");
    bench_embassy_sync::<256>(criterion, "cap256");
    bench_embassy_sync::<4096>(criterion, "cap4096");

    bench_inbox_alloc_spsc::<16>(criterion, "cap16");
    bench_inbox_alloc_spsc::<256>(criterion, "cap256");
    bench_inbox_alloc_spsc::<4096>(criterion, "cap4096");
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
