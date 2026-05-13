//! Compare-bench for proxima-storage's pmem module against the std durable path.
//!
//! Pure Rust, zero C: the incumbent is `std::fs::File::sync_data` (fsync), the
//! canonical std "make this write survive a crash" primitive — no `libc`, no
//! `mmap`, no unsafe anywhere in this file.
//!
//! HONEST READ (see docs/pmem/discipline.md): on a host WITHOUT real pmem and
//! without the x86 cache-flush path (e.g. aarch64 macOS, the dev box), `persist`
//! is a documented no-op, so `cow_commit_real_persist` measures the FSM's
//! compute, NOT a durability barrier — comparing it to fsync is no-syscall vs
//! syscall. The real durable comparison (x86 clwb+sfence per cache line vs
//! fsync) is meaningful on an x86_64-linux host, where `persist` emits real
//! instructions; that arm is run on host-b and recorded in the discipline
//! log. The `cow_recover` arm (the O(1) crux read) is meaningful anywhere.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_storage::pmem::cow::CowRoot;
use proxima_storage::pmem::persist;
use std::fs::OpenOptions;
use std::hint::black_box;
use std::os::unix::fs::FileExt;

fn bench(criterion: &mut Criterion) {
    let layout = CowRoot::new(8).unwrap();
    let old = [0xAAu8; 8];
    let new = [0xBBu8; 8];
    let noop = |_: &[u8]| {};

    let mut group = criterion.benchmark_group("pmem_commit");
    group.throughput(Throughput::Elements(1));

    // design-favors: proxima — the FSM's pure compute (slot copy + LE root store)
    group.bench_function("cow_commit_noop_persist", |bencher| {
        let mut region = vec![0u8; layout.region_len()];
        layout.init(&mut region, &old, &noop).unwrap();
        let toggle = [new, old];
        let mut turn = 0usize;
        bencher.iter(|| {
            layout
                .commit(&mut region, &toggle[turn & 1], &noop)
                .unwrap();
            turn += 1;
            black_box(&region);
        });
    });

    // design-favors: neutral on dev (no-op persist), incumbent-favored on x86
    // (real clwb+sfence) — the per-commit durability barrier proxima-pmem ships.
    group.bench_function("cow_commit_real_persist", |bencher| {
        let mut region = vec![0u8; layout.region_len()];
        layout.init(&mut region, &old, &persist::persist).unwrap();
        let toggle = [new, old];
        let mut turn = 0usize;
        bencher.iter(|| {
            layout
                .commit(&mut region, &toggle[turn & 1], &persist::persist)
                .unwrap();
            turn += 1;
            black_box(&region);
        });
    });

    // design-favors: proxima — recovery is a single atomic root read, O(1) in payload
    group.bench_function("cow_recover", |bencher| {
        let mut region = vec![0u8; layout.region_len()];
        layout.init(&mut region, &old, &noop).unwrap();
        layout.commit(&mut region, &new, &noop).unwrap();
        bencher.iter(|| {
            black_box(layout.recover(black_box(&region)).unwrap());
        });
    });

    // design-favors: incumbent — the std durable path's design point is the
    // per-commit durability barrier: write the cell, fsync. Pure Rust std.
    group.bench_function("incumbent_file_sync_data", |bencher| {
        let path = std::env::temp_dir().join("proxima_pmem_bench.dat");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        file.set_len(4096).unwrap();
        file.sync_all().unwrap();
        let mut counter: u64 = 0;
        bencher.iter(|| {
            counter = counter.wrapping_add(1);
            file.write_at(&counter.to_le_bytes(), 0).unwrap();
            file.sync_data().unwrap();
            black_box(counter);
        });
        let _ = std::fs::remove_file(&path);
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
