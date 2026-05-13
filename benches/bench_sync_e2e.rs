#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! E2E composition bench for `proxima::sync::mpsc` — closes gate-7 of
//! the disciplined-component skill: "when downstream components
//! compose this one, the e2e bench gains an arm with this swap."
//!
//! Workload mirrors the body-pump pattern in `listeners/http.rs`:
//! - 1 producer task pumps chunks into bounded(32) mpsc with a
//!   cancellation token racing the send (via select_biased!)
//! - 1 consumer task drains via `futures::stream::unfold` (same shape
//!   as `listeners/http.rs:1465`)
//! - When the consumer has received N chunks, the iter completes
//!
//! Two arms compared:
//! - `tokio_mpsc` — `tokio::sync::mpsc` + `tokio_util::sync::CancellationToken`
//! - `proxima_mpsc` — `proxima::sync::mpsc` + same cancel token
//!
//! Two host-runtime matrices per arm:
//! - `current_thread` — single tokio current-thread runtime
//! - `multi_thread_4` — 4-thread tokio multi-thread runtime (forces
//!   cross-thread send-recv handoff)
//!
//! The matrix matters because the criterion compat bench
//! (`bench_sync_compat.rs`) only exercises current_thread. If proxima
//! mpsc's interior-mutex cost shows up anywhere, multi_thread is
//! where.
//!
//! Run as: `cargo bench --bench bench_sync_e2e`

use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::FutureExt;
use futures::stream::StreamExt;
use tokio::runtime::{Builder as TokioBuilder, Runtime};
use tokio_util::sync::CancellationToken;

const BODY_CHANNEL_DEPTH: usize = 32;
const CHUNK_COUNT: usize = 1024;
const CHUNK_SIZE: usize = 8 * 1024;

fn current_thread_runtime() -> Runtime {
    TokioBuilder::new_current_thread()
        .enable_all()
        .build()
        .expect("current thread runtime")
}

fn multi_thread_runtime() -> Runtime {
    TokioBuilder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("multi thread runtime")
}

fn payload(seed: u8) -> Bytes {
    let mut buf = Vec::with_capacity(CHUNK_SIZE);
    buf.resize(CHUNK_SIZE, seed);
    Bytes::from(buf)
}

// ---------- tokio mpsc body-pump shape ----------

async fn pump_tokio(body_tx: tokio::sync::mpsc::Sender<Bytes>, cancel: CancellationToken) -> usize {
    let mut sent = 0;
    for index in 0..CHUNK_COUNT {
        let chunk = payload((index & 0xFF) as u8);
        let cancel_fut = cancel.cancelled().fuse();
        let send_fut = body_tx.send(chunk).fuse();
        futures::pin_mut!(cancel_fut, send_fut);
        let send_won = futures::select_biased! {
            _ = cancel_fut => false,
            outcome = send_fut => outcome.is_ok(),
        };
        if !send_won {
            break;
        }
        sent += 1;
    }
    sent
}

async fn drain_tokio(body_rx: tokio::sync::mpsc::Receiver<Bytes>) -> usize {
    // mirrors listeners/http.rs:1465 — receiver is the unfold state,
    // threaded through each yield so the closure stays FnMut.
    let body_stream = futures::stream::unfold(body_rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (chunk, rx))
    });
    futures::pin_mut!(body_stream);
    let mut received = 0;
    while body_stream.next().await.is_some() {
        received += 1;
    }
    received
}

async fn one_iter_tokio() {
    let (body_tx, body_rx) = tokio::sync::mpsc::channel::<Bytes>(BODY_CHANNEL_DEPTH);
    let cancel = CancellationToken::new();
    let pump = tokio::spawn(pump_tokio(body_tx, cancel.clone()));
    let drain = tokio::spawn(drain_tokio(body_rx));
    let sent = pump.await.expect("pump join");
    let received = drain.await.expect("drain join");
    assert_eq!(sent, CHUNK_COUNT, "pump must send all chunks");
    assert_eq!(received, CHUNK_COUNT, "drain must receive all chunks");
}

// ---------- proxima mpsc body-pump shape ----------
//
// the body_rx receiver type is different but the listener's
// `futures::stream::unfold(body_rx, ...)` pattern is the same.

async fn pump_proxima(
    body_tx: proxima::sync::mpsc::Sender<Bytes>,
    cancel: CancellationToken,
) -> usize {
    let mut sent = 0;
    for index in 0..CHUNK_COUNT {
        let chunk = payload((index & 0xFF) as u8);
        let cancel_fut = cancel.cancelled().fuse();
        let send_fut = body_tx.send(chunk).fuse();
        futures::pin_mut!(cancel_fut, send_fut);
        let send_won = futures::select_biased! {
            _ = cancel_fut => false,
            outcome = send_fut => outcome.is_ok(),
        };
        if !send_won {
            break;
        }
        sent += 1;
    }
    sent
}

async fn drain_proxima(body_rx: proxima::sync::mpsc::Receiver<Bytes>) -> usize {
    let body_stream = futures::stream::unfold(body_rx, |mut rx| async move {
        rx.recv().await.map(|chunk| (chunk, rx))
    });
    futures::pin_mut!(body_stream);
    let mut received = 0;
    while body_stream.next().await.is_some() {
        received += 1;
    }
    received
}

async fn one_iter_proxima() {
    let (body_tx, body_rx) = proxima::sync::mpsc::channel::<Bytes>(BODY_CHANNEL_DEPTH);
    let cancel = CancellationToken::new();
    let pump = tokio::spawn(pump_proxima(body_tx, cancel.clone()));
    let drain = tokio::spawn(drain_proxima(body_rx));
    let sent = pump.await.expect("pump join");
    let received = drain.await.expect("drain join");
    assert_eq!(sent, CHUNK_COUNT, "pump must send all chunks");
    assert_eq!(received, CHUNK_COUNT, "drain must receive all chunks");
}

fn bench_body_pump_current_thread(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("body_pump_current_thread");
    // design-favors: tokio (partial) — body-pump shape but SPSC
    group.throughput(Throughput::Bytes((CHUNK_COUNT * CHUNK_SIZE) as u64));
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("tokio_mpsc", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(one_iter_tokio()));
    });

    group.bench_function("proxima_mpsc", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| runtime.block_on(one_iter_proxima()));
    });

    group.finish();
}

fn bench_body_pump_multi_thread(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("body_pump_multi_thread_4");
    // design-favors: tokio (partial) — MT4 host but still SPSC
    group.throughput(Throughput::Bytes((CHUNK_COUNT * CHUNK_SIZE) as u64));
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("tokio_mpsc", |bench| {
        let runtime = multi_thread_runtime();
        bench.iter(|| runtime.block_on(one_iter_tokio()));
    });

    group.bench_function("proxima_mpsc", |bench| {
        let runtime = multi_thread_runtime();
        bench.iter(|| runtime.block_on(one_iter_proxima()));
    });

    group.finish();
}

// ---------- MPSC under contention (8p/1c, cap=32, MT4) ----------
//
// tokio's home turf: many concurrent producers hammer a bounded channel
// forcing real multi-producer arbitration (segmented queue + permit
// reservation). MT4 scheduler gets to work-steal across 8 producer tasks.
// This is the workload tokio::sync::mpsc was designed for.

const MPSC_CONTENTION_PRODUCERS: usize = 8;
const MPSC_CONTENTION_CHUNKS_PER_PRODUCER: usize = CHUNK_COUNT / MPSC_CONTENTION_PRODUCERS;

async fn contention_iter_tokio() {
    let (body_tx, body_rx) = tokio::sync::mpsc::channel::<Bytes>(BODY_CHANNEL_DEPTH);
    let cancel = CancellationToken::new();

    let mut producers = Vec::with_capacity(MPSC_CONTENTION_PRODUCERS);
    for producer_index in 0..MPSC_CONTENTION_PRODUCERS {
        let sender = body_tx.clone();
        let token = cancel.clone();
        producers.push(tokio::spawn(async move {
            let mut sent = 0usize;
            for chunk_index in 0..MPSC_CONTENTION_CHUNKS_PER_PRODUCER {
                let seed = ((producer_index * MPSC_CONTENTION_CHUNKS_PER_PRODUCER + chunk_index)
                    & 0xFF) as u8;
                let chunk = payload(seed);
                let cancel_fut = token.cancelled().fuse();
                let send_fut = sender.send(chunk).fuse();
                futures::pin_mut!(cancel_fut, send_fut);
                let send_won = futures::select_biased! {
                    _ = cancel_fut => false,
                    outcome = send_fut => outcome.is_ok(),
                };
                if !send_won {
                    break;
                }
                sent += 1;
            }
            sent
        }));
    }
    drop(body_tx);

    let drain = tokio::spawn(drain_tokio(body_rx));

    let mut total_sent = 0usize;
    for producer in producers {
        total_sent += producer.await.expect("producer join");
    }
    let received = drain.await.expect("drain join");
    assert_eq!(
        total_sent, CHUNK_COUNT,
        "all producers must deliver all chunks"
    );
    assert_eq!(received, CHUNK_COUNT, "drain must receive all chunks");
}

async fn contention_iter_proxima() {
    let (body_tx, body_rx) = proxima::sync::mpsc::channel::<Bytes>(BODY_CHANNEL_DEPTH);
    let cancel = CancellationToken::new();

    let mut producers = Vec::with_capacity(MPSC_CONTENTION_PRODUCERS);
    for producer_index in 0..MPSC_CONTENTION_PRODUCERS {
        let sender = body_tx.clone();
        let token = cancel.clone();
        producers.push(tokio::spawn(async move {
            let mut sent = 0usize;
            for chunk_index in 0..MPSC_CONTENTION_CHUNKS_PER_PRODUCER {
                let seed = ((producer_index * MPSC_CONTENTION_CHUNKS_PER_PRODUCER + chunk_index)
                    & 0xFF) as u8;
                let chunk = payload(seed);
                let cancel_fut = token.cancelled().fuse();
                let send_fut = sender.send(chunk).fuse();
                futures::pin_mut!(cancel_fut, send_fut);
                let send_won = futures::select_biased! {
                    _ = cancel_fut => false,
                    outcome = send_fut => outcome.is_ok(),
                };
                if !send_won {
                    break;
                }
                sent += 1;
            }
            sent
        }));
    }
    drop(body_tx);

    let drain = tokio::spawn(drain_proxima(body_rx));

    let mut total_sent = 0usize;
    for producer in producers {
        total_sent += producer.await.expect("producer join");
    }
    let received = drain.await.expect("drain join");
    assert_eq!(
        total_sent, CHUNK_COUNT,
        "all producers must deliver all chunks"
    );
    assert_eq!(received, CHUNK_COUNT, "drain must receive all chunks");
}

fn bench_mpsc_mpsc_under_contention_8p1c_mt4(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mpsc_mpsc_under_contention_8p1c_mt4");
    // design-favors: incumbent (tokio::mpsc home turf — MPSC under contention on MT4)
    group.throughput(Throughput::Bytes((CHUNK_COUNT * CHUNK_SIZE) as u64));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("tokio_mpsc", |bench| {
        let runtime = multi_thread_runtime();
        bench.iter(|| runtime.block_on(contention_iter_tokio()));
    });

    group.bench_function("proxima_mpsc", |bench| {
        let runtime = multi_thread_runtime();
        bench.iter(|| runtime.block_on(contention_iter_proxima()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_body_pump_current_thread,
    bench_body_pump_multi_thread,
    bench_mpsc_mpsc_under_contention_8p1c_mt4,
);
criterion_main!(benches);
