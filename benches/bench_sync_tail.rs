#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Tail-latency sweep for `proxima::sync` primitives vs. their
//! `tokio::sync` analogues. Records per-roundtrip latency into an HDR
//! histogram and reports p50/p90/p99/p999/max — not just means.
//!
//! Companion to `bench_sync_compat.rs` (criterion-driven throughput
//! median). The two cover different questions: compat reports
//! steady-state throughput; tail reports outlier latency under the
//! same workloads.
//!
//! Not a criterion bench — drives its own loop because criterion
//! reports mean/median, not tail percentiles. Single binary, runs
//! as `cargo bench --bench bench_sync_tail`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use hdrhistogram::Histogram;

/// How long each (primitive × arm) combo runs.
const SAMPLE_WINDOW: Duration = Duration::from_secs(3);

const MPSC_CAPACITY: usize = 32;
const MPSC_PAYLOAD: usize = 8 * 1024;
const NOTIFY_BATCH: usize = 64;
const WATCH_BATCH: usize = 64;
const BROADCAST_BATCH: usize = 64;
const BROADCAST_CAPACITY: usize = 1024;
const BROADCAST_SUBSCRIBERS: usize = 4;

/// Histogram range: 1ns to 60s, 3 sig figs. Auto-resizes if needed.
fn fresh_histogram() -> Histogram<u64> {
    Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).expect("hdr bounds")
}

fn current_thread_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current thread runtime")
}

fn payload(seed: u8) -> Bytes {
    let mut buf = Vec::with_capacity(MPSC_PAYLOAD);
    buf.resize(MPSC_PAYLOAD, seed);
    Bytes::from(buf)
}

fn format_ns(value: u64) -> String {
    if value < 1_000 {
        format!("{value} ns")
    } else if value < 1_000_000 {
        format!("{:.2} us", value as f64 / 1_000.0)
    } else if value < 1_000_000_000 {
        format!("{:.2} ms", value as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", value as f64 / 1_000_000_000.0)
    }
}

fn report(label: &str, histogram: &Histogram<u64>) {
    let p50 = histogram.value_at_quantile(0.50);
    let p90 = histogram.value_at_quantile(0.90);
    let p99 = histogram.value_at_quantile(0.99);
    let p999 = histogram.value_at_quantile(0.999);
    let max = histogram.max();
    let count = histogram.len();
    let mean = histogram.mean();
    println!(
        "{label:<40} count={count:<8} mean={:<10} p50={:<10} p90={:<10} p99={:<10} p999={:<10} max={}",
        format_ns(mean as u64),
        format_ns(p50),
        format_ns(p90),
        format_ns(p99),
        format_ns(p999),
        format_ns(max),
    );
}

// ---------- mpsc tail: per-chunk send→recv latency ----------

async fn mpsc_tokio_one_chunk(
    sender: &tokio::sync::mpsc::Sender<Bytes>,
    receiver: &mut tokio::sync::mpsc::Receiver<Bytes>,
    seed: u8,
) -> Duration {
    let chunk = payload(seed);
    let start = Instant::now();
    sender.send(chunk).await.expect("send");
    receiver.recv().await.expect("recv");
    start.elapsed()
}

async fn mpsc_proxima_one_chunk(
    sender: &proxima::sync::mpsc::Sender<Bytes>,
    receiver: &mut proxima::sync::mpsc::Receiver<Bytes>,
    seed: u8,
) -> Duration {
    let chunk = payload(seed);
    let start = Instant::now();
    sender.send(chunk).await.expect("send");
    receiver.recv().await.expect("recv");
    start.elapsed()
}

fn sweep_mpsc_tail(runtime: &tokio::runtime::Runtime) {
    let mut tokio_histo = fresh_histogram();
    runtime.block_on(async {
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<Bytes>(MPSC_CAPACITY);
        let deadline = Instant::now() + SAMPLE_WINDOW;
        let mut seed = 0_u8;
        while Instant::now() < deadline {
            let elapsed = mpsc_tokio_one_chunk(&sender, &mut receiver, seed).await;
            let _ = tokio_histo.record((elapsed.as_nanos() as u64).max(1));
            seed = seed.wrapping_add(1);
        }
    });
    report("mpsc_tail/tokio", &tokio_histo);

    let mut proxima_histo = fresh_histogram();
    runtime.block_on(async {
        let (sender, mut receiver) = proxima::sync::mpsc::channel::<Bytes>(MPSC_CAPACITY);
        let deadline = Instant::now() + SAMPLE_WINDOW;
        let mut seed = 0_u8;
        while Instant::now() < deadline {
            let elapsed = mpsc_proxima_one_chunk(&sender, &mut receiver, seed).await;
            let _ = proxima_histo.record((elapsed.as_nanos() as u64).max(1));
            seed = seed.wrapping_add(1);
        }
    });
    report("mpsc_tail/proxima", &proxima_histo);
}

// ---------- notify tail: per-roundtrip notify_one → notified() latency ----------

fn sweep_notify_tail(runtime: &tokio::runtime::Runtime) {
    let mut tokio_histo = fresh_histogram();
    runtime.block_on(async {
        let notify = Arc::new(tokio::sync::Notify::new());
        let consumer_notify = notify.clone();
        let consumer = tokio::spawn(async move {
            loop {
                consumer_notify.notified().await;
            }
        });
        let deadline = Instant::now() + SAMPLE_WINDOW;
        while Instant::now() < deadline {
            // batch up roundtrips so we measure the per-pair latency,
            // not just the yield overhead between batches
            for _ in 0..NOTIFY_BATCH {
                let start = Instant::now();
                notify.notify_one();
                tokio::task::yield_now().await;
                let elapsed = start.elapsed();
                let _ = tokio_histo.record((elapsed.as_nanos() as u64).max(1));
            }
        }
        consumer.abort();
    });
    report("notify_tail/tokio", &tokio_histo);

    let mut proxima_histo = fresh_histogram();
    runtime.block_on(async {
        let notify = Arc::new(proxima::sync::Notify::new());
        let consumer_notify = notify.clone();
        let consumer = tokio::spawn(async move {
            loop {
                consumer_notify.notified().await;
            }
        });
        let deadline = Instant::now() + SAMPLE_WINDOW;
        while Instant::now() < deadline {
            for _ in 0..NOTIFY_BATCH {
                let start = Instant::now();
                notify.notify_one();
                tokio::task::yield_now().await;
                let elapsed = start.elapsed();
                let _ = proxima_histo.record((elapsed.as_nanos() as u64).max(1));
            }
        }
        consumer.abort();
    });
    report("notify_tail/proxima", &proxima_histo);
}

// ---------- watch tail: per send→observe latency ----------

fn sweep_watch_tail(runtime: &tokio::runtime::Runtime) {
    let mut tokio_histo = fresh_histogram();
    runtime.block_on(async {
        let (sender, mut receiver) = tokio::sync::watch::channel(0_u64);
        let deadline = Instant::now() + SAMPLE_WINDOW;
        let mut value = 0_u64;
        while Instant::now() < deadline {
            for _ in 0..WATCH_BATCH {
                value = value.wrapping_add(1);
                let start = Instant::now();
                sender.send(value).expect("send");
                tokio::task::yield_now().await;
                receiver.changed().await.expect("changed");
                let _ = *receiver.borrow_and_update();
                let elapsed = start.elapsed();
                let _ = tokio_histo.record((elapsed.as_nanos() as u64).max(1));
            }
        }
    });
    report("watch_tail/tokio", &tokio_histo);

    let mut proxima_histo = fresh_histogram();
    runtime.block_on(async {
        let (sender, mut receiver) = proxima::sync::watch::channel(0_u64);
        let deadline = Instant::now() + SAMPLE_WINDOW;
        let mut value = 0_u64;
        while Instant::now() < deadline {
            for _ in 0..WATCH_BATCH {
                value = value.wrapping_add(1);
                let start = Instant::now();
                sender.send(value).expect("send");
                tokio::task::yield_now().await;
                receiver.changed().await.expect("changed");
                let _ = *receiver.borrow_and_update();
                let elapsed = start.elapsed();
                let _ = proxima_histo.record((elapsed.as_nanos() as u64).max(1));
            }
        }
    });
    report("watch_tail/proxima", &proxima_histo);
}

// ---------- broadcast tail: per-send → last-subscriber-recv latency ----------

fn sweep_broadcast_tail(runtime: &tokio::runtime::Runtime) {
    let mut tokio_histo = fresh_histogram();
    runtime.block_on(async {
        let (sender, _seed) = tokio::sync::broadcast::channel::<u64>(BROADCAST_CAPACITY);
        let mut consumers = Vec::with_capacity(BROADCAST_SUBSCRIBERS);
        for _ in 0..BROADCAST_SUBSCRIBERS {
            let mut receiver = sender.subscribe();
            consumers.push(tokio::spawn(async move {
                loop {
                    match receiver.recv().await {
                        Ok(_value) => {}
                        Err(_) => break,
                    }
                }
            }));
        }
        let deadline = Instant::now() + SAMPLE_WINDOW;
        let mut value = 0_u64;
        while Instant::now() < deadline {
            for _ in 0..BROADCAST_BATCH {
                value = value.wrapping_add(1);
                let start = Instant::now();
                let _ = sender.send(value);
                tokio::task::yield_now().await;
                let elapsed = start.elapsed();
                let _ = tokio_histo.record((elapsed.as_nanos() as u64).max(1));
            }
        }
        for consumer in consumers {
            consumer.abort();
        }
    });
    report("broadcast_tail/tokio", &tokio_histo);

    let mut proxima_histo = fresh_histogram();
    runtime.block_on(async {
        let (sender, _seed) = proxima::sync::broadcast::channel::<u64>(BROADCAST_CAPACITY);
        let mut consumers = Vec::with_capacity(BROADCAST_SUBSCRIBERS);
        for _ in 0..BROADCAST_SUBSCRIBERS {
            let mut receiver = sender.subscribe();
            consumers.push(tokio::spawn(async move {
                loop {
                    match receiver.recv().await {
                        Ok(_value) => {}
                        Err(proxima::sync::broadcast::error::RecvError::Lagged(_)) => {}
                        Err(proxima::sync::broadcast::error::RecvError::Closed) => break,
                    }
                }
            }));
        }
        let deadline = Instant::now() + SAMPLE_WINDOW;
        let mut value = 0_u64;
        while Instant::now() < deadline {
            for _ in 0..BROADCAST_BATCH {
                value = value.wrapping_add(1);
                let start = Instant::now();
                let _ = sender.send(value);
                tokio::task::yield_now().await;
                let elapsed = start.elapsed();
                let _ = proxima_histo.record((elapsed.as_nanos() as u64).max(1));
            }
        }
        for consumer in consumers {
            consumer.abort();
        }
    });
    report("broadcast_tail/proxima", &proxima_histo);
}

fn main() {
    let runtime = current_thread_runtime();

    println!(
        "proxima::sync vs tokio::sync — tail latency sweep, {}s per arm",
        SAMPLE_WINDOW.as_secs(),
    );
    println!();

    sweep_mpsc_tail(&runtime);
    sweep_notify_tail(&runtime);
    sweep_watch_tail(&runtime);
    sweep_broadcast_tail(&runtime);
}
