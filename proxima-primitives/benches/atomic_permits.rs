//! Micro-bench for the synchronous counted permit primitive.
//!
//! The named incumbent is `sync::Semaphore::try_acquire`; both arms keep a
//! single permit live for exactly one acquisition/release operation.
#![allow(clippy::expect_used)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_primitives::sync::{AtomicPermitPool, Semaphore};

fn atomic_permit(c: &mut Criterion) {
    let pool = AtomicPermitPool::new(1);
    c.bench_function("atomic_permit_pool_try_acquire", |b| {
        b.iter(|| {
            let permit = pool.try_acquire().expect("capacity is one");
            black_box(permit);
        });
    });
}

fn semaphore_permit(c: &mut Criterion) {
    let pool = Semaphore::new(1);
    c.bench_function("semaphore_try_acquire", |b| {
        b.iter(|| {
            let permit = pool.try_acquire().expect("capacity is one");
            black_box(permit);
        });
    });
}

criterion_group!(benches, atomic_permit, semaphore_permit);
criterion_main!(benches);
