#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Recording sink write-path primitives.
//!
//! `JsonlSink` and `BinSink` use `Arc<Mutex<File>>` to serialize
//! concurrent recorder writes. This bench validates that decision
//! against the realistic alternatives:
//!
//! 1. **`Mutex<File>`** — current shape. Single-writer queue across
//!    callers; one write_all per record.
//! 2. **`AtomicFile<O_APPEND>`** — relies on kernel atomicity for
//!    writes ≤ PIPE_BUF (4 KiB on Linux). No userspace lock, but
//!    breaks for larger frames.
//! 3. **`SegQueue + single writer task`** — lock-free MPSC queue
//!    with a dedicated writer task draining to the file. Adds task
//!    scheduling latency per record.
//!
//! Pattern: 4 recorder threads each push a small record. We measure
//! per-record latency from the recorder's perspective.

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use crossbeam_queue::SegQueue;
use tempfile::tempdir;

const RECORD_BYTES: &[u8] = b"{\"id\":1234,\"kind\":\"chunk\",\"len\":256,\"ts\":1700000000000}\n";

fn mutex_file_write(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("recording_mutex_file_4_writers");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mutex.jsonl");
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .expect("open");
    let file = Arc::new(std::sync::Mutex::new(file));

    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    for _ in 0..4 {
        let file = file.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let _ = file.lock().map(|mut f| f.write_all(RECORD_BYTES));
            }
        }));
    }

    group.bench_function("write", |bencher| {
        bencher.iter(|| {
            let _ = file.lock().map(|mut f| f.write_all(RECORD_BYTES));
        });
    });

    stop.store(true, Ordering::Relaxed);
    for handle in handles {
        handle.join().expect("noise writer joined");
    }
    group.finish();
}

fn append_only_file_write(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("recording_o_append_4_writers");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("oappend.jsonl");

    // Each writer holds its OWN handle to the same O_APPEND file;
    // kernel atomically serializes appends for writes ≤ PIPE_BUF.
    // RECORD_BYTES is 56 bytes, well under the 4 KiB Linux limit.
    let open_handle = || {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .expect("open")
    };

    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    for _ in 0..4 {
        let stop = stop.clone();
        let path = path.clone();
        handles.push(thread::spawn(move || {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .expect("open");
            while !stop.load(Ordering::Relaxed) {
                let _ = file.write_all(RECORD_BYTES);
            }
        }));
    }

    let mut file = open_handle();
    group.bench_function("write", |bencher| {
        bencher.iter(|| {
            let _ = file.write_all(RECORD_BYTES);
        });
    });

    stop.store(true, Ordering::Relaxed);
    for handle in handles {
        handle.join().expect("noise writer joined");
    }
    group.finish();
}

fn segqueue_single_writer(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("recording_segqueue_single_writer_4_writers");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("segqueue.jsonl");

    let queue: Arc<SegQueue<Vec<u8>>> = Arc::new(SegQueue::new());
    let stop = Arc::new(AtomicBool::new(false));

    // Single dedicated writer task drains the queue.
    let writer_handle = {
        let queue = queue.clone();
        let stop = stop.clone();
        let path = path.clone();
        thread::spawn(move || {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .expect("open");
            while !stop.load(Ordering::Relaxed) {
                if let Some(record) = queue.pop() {
                    let _ = file.write_all(&record);
                } else {
                    // empty queue — yield rather than spin hard
                    std::thread::yield_now();
                }
            }
            // drain remaining
            while let Some(record) = queue.pop() {
                let _ = file.write_all(&record);
            }
        })
    };

    // 4 noise recorders pushing into the queue
    let mut noise = Vec::new();
    for _ in 0..4 {
        let queue = queue.clone();
        let stop = stop.clone();
        noise.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                queue.push(RECORD_BYTES.to_vec());
            }
        }));
    }

    group.bench_function("push", |bencher| {
        bencher.iter(|| {
            queue.push(RECORD_BYTES.to_vec());
        });
    });

    stop.store(true, Ordering::Relaxed);
    for handle in noise {
        handle.join().expect("noise recorder joined");
    }
    writer_handle.join().expect("writer joined");
    group.finish();
}

criterion_group!(
    benches,
    mutex_file_write,
    append_only_file_write,
    segqueue_single_writer
);
criterion_main!(benches);
