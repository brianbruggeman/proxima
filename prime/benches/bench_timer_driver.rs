//! Gate rows 5, 6, 13 — per-arm `schedule_wake` cost: prime-wheel vs std-thread.
//!
//! Home-turf 80% case: a server arms one timer per recv-loop iteration (the
//! native h3 listener's 1ms tick fires on every datagram recv with no data).
//! std-thread spawns an OS thread per arm; prime-wheel inserts into an O(1)
//! hashed timer wheel on the calling core's slab.
//!
//! Arms:
//!   - `std_thread`  (incumbent): `StdThreadDriver::schedule_wake` — one
//!     `std::thread::spawn` per arm. Measured N=1k; N=100k exceeds practical OS
//!     limits on macOS (thread ceiling ~2k) so is not included for arm A.
//!   - `prime_wheel` (component): `TimerWheel::register` directly — the O(1)
//!     slab insert without a running worker. Measured at N=1k and N=100k to
//!     confirm flat per-arm cost. This is the steady-state hot path cost;
//!     setup (wheel creation) is excluded from the measured loop.
//!
//! Note on context asymmetry: both arms run on the bench's calling thread.
//! StdThreadDriver spawns real OS threads per call. TimerWheel::register
//! is the primitive that `core_shard::schedule_wake` calls when running on
//! a prime worker; the overhead of routing through the thread-local is ~1ns
//! and is not measured here. The gap between the two arms is dominated by
//! thread-spawn vs slab-insert and is honest on its own terms.
//!
//! Run:
//!   cargo bench -p prime \
//!     --features runtime-prime-timer,runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor \
//!     --bench bench_timer_driver \
//!     -- --save-baseline timer-driver-$(date +%Y%m%d)

#![cfg(all(
    feature = "runtime-prime-timer",
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::hint::black_box;
use std::task::{RawWaker, RawWakerVTable, Waker};
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use prime::core::timer::{Clock, Tick, TimerWheel};
use proxima_core::time::drivers::std_thread::DRIVER as STD_THREAD_DRIVER;
use proxima_core::time::{Driver, Instant};

const SIZES: &[u64] = &[1_000, 100_000];

/// A clock that always returns 0 — the wheel registers all entries relative
/// to tick 0, which is representative of real use (every fresh timer call
/// starts from the current tick). The zero value means all deadlines > 0
/// land in L0 or L1 without cascading, matching the h3 short-timeout pattern.
struct ZeroClock;

impl Clock for ZeroClock {
    fn now(&self) -> Tick {
        0
    }
}

/// Noop waker: no allocation, no signalling. Using it in the bench avoids
/// measuring Arc/vtable overhead that production wakers carry; the insert
/// cost is what matters, not waker-wake dispatch.
fn noop_waker() -> Waker {
    unsafe fn clone_noop(data: *const ()) -> RawWaker {
        RawWaker::new(data, &NOOP_VTABLE)
    }
    unsafe fn noop(_: *const ()) {}
    static NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(clone_noop, noop, noop, noop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &NOOP_VTABLE)) }
}

/// Arm A (incumbent): `StdThreadDriver::schedule_wake` — one thread spawn per arm.
///
/// Limited to N=1_000 because macOS limits ~2k threads per process; 100k
/// spawns per criterion iteration would exhaust the limit. The per-arm cost
/// at N=1k is representative.
///
/// Deadline: `Instant::from_monotonic(Duration::ZERO)` → `target = epoch`
/// (already past) → spawned threads don't sleep, just call `waker.wake()`
/// and exit. Measures pure thread-spawn cost.
fn bench_std_thread(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("schedule_wake/std_thread");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));

    let arms: u64 = 1_000;
    group.throughput(Throughput::Elements(arms));
    let past_deadline = Instant::from_monotonic(Duration::ZERO);

    group.bench_with_input(
        BenchmarkId::from_parameter("1k"),
        &arms,
        |bencher, &arm_count| {
            bencher.iter(|| {
                for _ in 0..arm_count {
                    let waker = black_box(noop_waker());
                    STD_THREAD_DRIVER.schedule_wake(black_box(past_deadline), waker);
                }
            });
        },
    );

    group.finish();
}

/// Arm B (component): `TimerWheel::register` — O(1) slab insert.
///
/// Measured at N=1k and N=100k to confirm flat per-arm cost. A fresh wheel
/// is created per iteration (setup cost excluded by `iter_batched`). For
/// N=100k the slab grows beyond its initial 256-entry pre-allocation; the
/// measured cost includes amortized Vec growth, making this a conservative
/// (slightly pessimistic) per-arm estimate vs a fully-warm slab.
fn bench_prime_wheel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("schedule_wake/prime_wheel");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    for &arms in SIZES {
        group.throughput(Throughput::Elements(arms));
        let label = if arms >= 1_000 {
            format!("{}k", arms / 1_000)
        } else {
            arms.to_string()
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(&label),
            &arms,
            |bencher, &arm_count| {
                bencher.iter_batched(
                    || TimerWheel::new(ZeroClock),
                    |mut wheel| {
                        for index in 0..arm_count {
                            let waker = noop_waker();
                            let deadline = black_box(index + 1);
                            black_box(wheel.register(deadline, waker));
                        }
                        wheel
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_std_thread, bench_prime_wheel);
criterion_main!(benches);
