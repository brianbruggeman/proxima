#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Per-sink channel primitive choice for `Tee` fan-out.
//!
//! `src/tee.rs` currently uses `tokio::sync::mpsc::channel(N)` +
//! `PollSender` for each sink's outbox. The original Stage 3 plan
//! prescribed converting this to `crossbeam_queue::ArrayQueue` for
//! Runtime portability (a future DPDK Runtime impl wouldn't need
//! tokio-specific channels). The plan didn't claim ArrayQueue was
//! faster — but we should know before converting.
//!
//! This bench compares three primitives under the actual tee
//! access pattern: 1 producer (the primary stream) pushes
//! `TeeEvent`-shaped values into a per-sink queue; 1 consumer
//! pulls from the receiver side. Same-task fan-out under per-core
//! dispatch is the common case.
//!
//! Three primitives:
//! 1. `tokio::sync::mpsc::channel(16)` — the current shape
//! 2. `crossbeam_queue::ArrayQueue<T>::new(16)` — bounded MPMC,
//!    lock-free; backpressure via try_push returning Err
//! 3. `crossbeam_queue::SegQueue<T>` — unbounded MPMC; no
//!    backpressure (the cost we'd pay to drop bounded semantics)
//!
//! Workloads:
//! - one_shot_roundtrip: push one event, pop one event (the
//!   uncontested-fan-out cost)
//! - producer_outpaces: push 16 events back-to-back (fill the
//!   queue), then drain — measures the per-push cost when the
//!   queue is hot

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use crossbeam_queue::{ArrayQueue, SegQueue};

#[derive(Clone)]
#[allow(dead_code)]
enum TeeEvent {
    Chunk(Bytes),
    End,
    Error(String),
}

fn fixture_event() -> TeeEvent {
    TeeEvent::Chunk(Bytes::from_static(b"chunk payload of moderate size"))
}

const QUEUE_SIZE: usize = 16;

fn one_shot_roundtrip(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tee_sink_one_shot_roundtrip");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // tokio::sync::mpsc
    group.bench_function("tokio_mpsc", |bencher| {
        bencher.to_async(&runtime).iter(|| async {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<TeeEvent>(QUEUE_SIZE);
            tx.send(fixture_event()).await.expect("send");
            std::hint::black_box(rx.recv().await);
        });
    });

    // crossbeam ArrayQueue
    group.bench_function("crossbeam_arrayqueue", |bencher| {
        bencher.iter(|| {
            let queue: ArrayQueue<TeeEvent> = ArrayQueue::new(QUEUE_SIZE);
            let _ = queue.push(fixture_event());
            std::hint::black_box(queue.pop());
        });
    });

    // crossbeam SegQueue (unbounded)
    group.bench_function("crossbeam_segqueue", |bencher| {
        bencher.iter(|| {
            let queue: SegQueue<TeeEvent> = SegQueue::new();
            queue.push(fixture_event());
            std::hint::black_box(queue.pop());
        });
    });

    group.finish();
}

fn producer_outpaces_consumer(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tee_sink_fill_then_drain");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(QUEUE_SIZE as u64));

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // tokio::sync::mpsc — fill to capacity then drain
    group.bench_function("tokio_mpsc", |bencher| {
        bencher.to_async(&runtime).iter(|| async {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<TeeEvent>(QUEUE_SIZE);
            for _ in 0..QUEUE_SIZE {
                tx.send(fixture_event()).await.expect("send");
            }
            for _ in 0..QUEUE_SIZE {
                std::hint::black_box(rx.recv().await);
            }
        });
    });

    // ArrayQueue — fill then drain (try_push always succeeds at capacity)
    group.bench_function("crossbeam_arrayqueue", |bencher| {
        bencher.iter(|| {
            let queue: ArrayQueue<TeeEvent> = ArrayQueue::new(QUEUE_SIZE);
            for _ in 0..QUEUE_SIZE {
                let _ = queue.push(fixture_event());
            }
            for _ in 0..QUEUE_SIZE {
                std::hint::black_box(queue.pop());
            }
        });
    });

    // SegQueue — same but unbounded
    group.bench_function("crossbeam_segqueue", |bencher| {
        bencher.iter(|| {
            let queue: SegQueue<TeeEvent> = SegQueue::new();
            for _ in 0..QUEUE_SIZE {
                queue.push(fixture_event());
            }
            for _ in 0..QUEUE_SIZE {
                std::hint::black_box(queue.pop());
            }
        });
    });

    group.finish();
}

fn shared_arc_long_lived(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tee_sink_long_lived_arc");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    // The realistic shape: an `Arc<Queue>` lives for the duration of a
    // connection; sinks share it. This bench avoids the per-iter
    // construct cost so we see steady-state push+pop only.

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // tokio::sync::mpsc — channel built once, push+recv steady-state
    group.bench_function("tokio_mpsc", |bencher| {
        let (tx, rx) = tokio::sync::mpsc::channel::<TeeEvent>(QUEUE_SIZE);
        let tx = Arc::new(tokio::sync::Mutex::new(tx));
        let rx = Arc::new(tokio::sync::Mutex::new(rx));
        bencher.to_async(&runtime).iter(|| {
            let tx = tx.clone();
            let rx = rx.clone();
            async move {
                tx.lock().await.send(fixture_event()).await.expect("send");
                std::hint::black_box(rx.lock().await.recv().await);
            }
        });
    });

    // ArrayQueue — Arc<ArrayQueue> lives once
    let arrayqueue: Arc<ArrayQueue<TeeEvent>> = Arc::new(ArrayQueue::new(QUEUE_SIZE));
    group.bench_function("crossbeam_arrayqueue", |bencher| {
        let queue = arrayqueue.clone();
        bencher.iter(|| {
            let _ = queue.push(fixture_event());
            std::hint::black_box(queue.pop());
        });
    });

    let segqueue: Arc<SegQueue<TeeEvent>> = Arc::new(SegQueue::new());
    group.bench_function("crossbeam_segqueue", |bencher| {
        let queue = segqueue.clone();
        bencher.iter(|| {
            queue.push(fixture_event());
            std::hint::black_box(queue.pop());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    one_shot_roundtrip,
    producer_outpaces_consumer,
    shared_arc_long_lived,
);
criterion_main!(benches);
