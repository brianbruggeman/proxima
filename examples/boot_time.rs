//! micro test: how we determine prime "boot" time.
//!
//! "boot" is a decomposition, not one number. per fresh process this binary
//! reports two of the pieces:
//!   boot_ns       main-entry -> prime runtime ready (first instant a task
//!                 actually runs on the per-core runtime). this is the
//!                 proxima-specific component a microVM slice pays.
//!   first_task_ns the first task's own execution (proves ready-for-compute).
//!
//! run it MANY times (fresh process each) for a cold-start distribution.
//! the harness also measures outer wall (spawn -> exit); outer_wall - boot_ns
//! is the OS process/loader tax that a VM slice AVOIDS (it jumps to a
//! pre-loaded image, no dyld). and this host number includes one
//! pthread_create for the worker that a slice skips (the vCPU IS the worker),
//! so it is an UPPER BOUND on the slice's prime-init. the KVM VMM-create cost
//! is Linux-only and not measured here.
#![allow(clippy::all, clippy::pedantic)]
use std::hint::black_box;
use std::time::Instant;

use proxima::runtime::run_prime;

fn main() -> Result<(), proxima::ProximaError> {
    let before = Instant::now();
    // this bench measures prime boot specifically, so it names the prime
    // backend directly rather than going through the adaptive `run`.
    #[allow(clippy::disallowed_methods)]
    let (ready, first_task_ns) = run_prime(async move {
        let ready = Instant::now();
        let out = black_box(black_box(21u32) * 2);
        assert_eq!(out, 42);
        (ready, ready.elapsed().as_nanos())
    })?;
    let boot_ns = ready.saturating_duration_since(before).as_nanos();
    println!("{boot_ns} {first_task_ns}");
    Ok(())
}
