//! From-scratch per-core async runtime. Umbrella-free: depends only
//! on `proxima_core` and `proxima_runtime` (runtime ends at
//! deps-only-`proxima-core` — see Wave D Phase 1). The
//! umbrella's serve adapters (`PrimeServeExt::serve_http`,
//! `serve_https_with_tls`) live in the umbrella because they pull
//! `crate::listener` and `crate::listeners`, which are still
//! umbrella-only today.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod core;

#[cfg(feature = "runtime-prime-reactor-trace")]
pub mod trace;

#[cfg(feature = "std")]
pub mod os;

#[cfg(all(
    feature = "std",
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
))]
pub mod config;
// the prime-wheel timer driver needs the per-core wheel (core_shard), which is
// only present with the executor/reactor/inbox stack.
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc",
))]
pub mod timer_driver;

#[cfg(all(
    feature = "std",
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
))]
pub use config::{CoreSelection, PoolKind, PrimeConfig};

#[cfg(all(
    feature = "std",
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
))]
pub use os::runtime::PrimeRuntime;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    /// DC5 test 1: alloc-only build succeeds and core inbox + executor are usable.
    /// Verifies the no_std cliff compiles and the critical primitives work under
    /// alloc-only (no TLS, no std). `try_send` (SPSC) is available; `try_send_mpsc`
    /// (TLS-backed) is only available under `std`.
    #[cfg(all(
        feature = "alloc",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-executor"
    ))]
    #[test]
    fn alloc_only_inbox_and_executor_are_usable() {
        use crate::core::inbox::channel;
        use crate::core::local_executor::LocalExecutor;

        let (producer, consumer) = channel::<u64>(2, 16);
        producer.try_send(42).expect("spsc send under alloc-only");
        assert_eq!(consumer.try_recv().expect("spsc recv under alloc-only"), 42);

        let executor = LocalExecutor::new();
        let counter = alloc::sync::Arc::new(core::sync::atomic::AtomicU32::new(0));
        let counter_clone = counter.clone();
        executor.spawn_local(async move {
            counter_clone.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        });
        executor.tick();
        assert_eq!(counter.load(core::sync::atomic::Ordering::Relaxed), 1);
    }

    /// DC5 test 2: std build produces a functional PrimeRuntime equivalent to
    /// the pre-DC5 baseline. `PrimeRuntime::new(1)` should succeed and the
    /// core count should be accessible.
    #[cfg(all(
        feature = "std",
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool",
    ))]
    #[test]
    fn std_build_prime_runtime_is_constructible() {
        use crate::os::runtime::PrimeRuntime;
        let runtime = PrimeRuntime::new(1).expect("build 1-core runtime");
        drop(runtime);
    }

    /// DC5 test 3: alloc-only build with timer module compiles. Verifies the
    /// `TimerWheel` type is accessible and constructible under alloc-only, with
    /// no std required. The clock trait is the only dependency.
    #[cfg(all(feature = "alloc", feature = "runtime-prime-timer"))]
    #[test]
    fn alloc_only_timer_wheel_is_constructible() {
        use crate::core::timer::{Clock, Tick, TimerWheel};

        struct TestClock(u64);
        impl Clock for TestClock {
            fn now(&self) -> Tick {
                self.0
            }
        }

        let wheel = TimerWheel::new(TestClock(0));
        assert_eq!(wheel.now(), 0);
    }
}
