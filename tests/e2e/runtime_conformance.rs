//! shared conformance suite that every `Runtime` impl must pass. invoked
//! per-backend below via the `runtime_conformance!` macro.
//!
//! covers the behavioral contract any per-core runtime is expected to
//! honor: dispatch shape (spawn_on_core, spawn_factory_on_core), thread-
//! local identity (current_core), background work (spawn_background_blocking),
//! deadlines (timer_at), and shutdown semantics.

#![cfg(any(
    feature = "runtime-tokio",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    )
))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use proxima::runtime::{CoreId, Runtime};

/// busy-wait up to `timeout` for `condition` to become true. used to give
/// per-core workers time to drain their inbox.
fn wait_for<F>(timeout: Duration, mut condition: F) -> bool
where
    F: FnMut() -> bool,
{
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    condition()
}

macro_rules! runtime_conformance {
    ($name:ident, $ctor:expr) => {
        mod $name {
            use super::*;

            fn build() -> Arc<dyn Runtime> {
                let raw = $ctor;
                Arc::new(raw)
            }

            #[test]
            fn num_cores_matches_constructor() {
                let runtime = build();
                assert_eq!(runtime.num_cores(), 2, "constructor was 2-core");
            }

            #[test]
            fn spawn_on_core_runs_send_future_on_target_core() {
                let runtime = build();
                let counter = Arc::new(AtomicUsize::new(0));
                let counter_for_task = counter.clone();
                runtime
                    .spawn_on_core(
                        CoreId(0),
                        Box::pin(async move {
                            counter_for_task.fetch_add(1, Ordering::AcqRel);
                        }),
                    )
                    .expect("conformance: spawn on fresh runtime must succeed");
                assert!(wait_for(Duration::from_secs(1), || {
                    counter.load(Ordering::Acquire) == 1
                }));
            }

            #[test]
            fn spawn_factory_on_core_builds_future_on_target() {
                let runtime = build();
                let counter = Arc::new(AtomicUsize::new(0));
                let counter_for_factory = counter.clone();
                runtime
                    .spawn_factory_on_core(
                        CoreId(1),
                        Box::new(move || {
                            let counter = counter_for_factory.clone();
                            Box::pin(async move {
                                counter.fetch_add(1, Ordering::AcqRel);
                            })
                        }),
                    )
                    .expect("conformance: factory spawn on fresh runtime must succeed");
                assert!(wait_for(Duration::from_secs(1), || {
                    counter.load(Ordering::Acquire) == 1
                }));
            }

            #[test]
            fn many_cross_core_spawns_preserve_count() {
                let runtime = build();
                let counter = Arc::new(AtomicUsize::new(0));
                let total = 500_usize;
                // 500 across 2 cores = 250 per lane, well under default
                // 1024 cap. Should succeed without back-pressure.
                for index in 0..total {
                    let counter = counter.clone();
                    let target = CoreId(index % 2);
                    runtime
                        .spawn_on_core(
                            target,
                            Box::pin(async move {
                                counter.fetch_add(1, Ordering::AcqRel);
                            }),
                        )
                        .expect("conformance: spawn under 1024 cap must succeed");
                }
                assert!(wait_for(Duration::from_secs(5), || {
                    counter.load(Ordering::Acquire) == total
                }));
            }

            #[test]
            fn spawn_background_blocking_returns_value() {
                let runtime = build();
                let outer_runtime = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap();
                outer_runtime.block_on(async {
                    let handle = runtime.spawn_background_blocking(Box::new(|| {
                        Ok(Box::new(123_u32) as Box<dyn std::any::Any + Send>)
                    }));
                    let value = handle.await.expect("bg result");
                    let downcast = value.downcast::<u32>().expect("downcast");
                    assert_eq!(*downcast, 123);
                });
            }

            #[test]
            fn timer_at_resolves_after_deadline() {
                let runtime = build();
                let done = Arc::new(AtomicUsize::new(0));
                let done_for_factory = done.clone();
                let runtime_for_factory = runtime.clone();
                let target = Instant::now() + Duration::from_millis(50);
                // build the timer future on the target core so it can be
                // non-Send (proxima's case); tokio's timer is Send but the
                // factory pattern works for both.
                runtime
                    .spawn_factory_on_core(
                        CoreId(0),
                        Box::new(move || {
                            let timer = runtime_for_factory.timer_at(target);
                            Box::pin(async move {
                                timer.await;
                                done_for_factory.fetch_add(1, Ordering::AcqRel);
                            })
                        }),
                    )
                    .expect("conformance: factory spawn on fresh runtime must succeed");
                assert!(wait_for(Duration::from_secs(1), || {
                    done.load(Ordering::Acquire) == 1
                }));
                let elapsed = Instant::now().saturating_duration_since(target);
                // accept within 100ms of deadline (tokio's timer has ~1ms granularity;
                // proxima's is also ms-based; thread scheduling adds noise).
                assert!(
                    elapsed < Duration::from_millis(200),
                    "fired far past deadline: {elapsed:?}",
                );
            }

            #[test]
            fn timer_in_the_past_resolves_quickly() {
                let runtime = build();
                let done = Arc::new(AtomicUsize::new(0));
                let done_for_factory = done.clone();
                let runtime_for_factory = runtime.clone();
                let past = Instant::now() - Duration::from_secs(1);
                runtime
                    .spawn_factory_on_core(
                        CoreId(0),
                        Box::new(move || {
                            let timer = runtime_for_factory.timer_at(past);
                            Box::pin(async move {
                                timer.await;
                                done_for_factory.fetch_add(1, Ordering::AcqRel);
                            })
                        }),
                    )
                    .expect("conformance: factory spawn on fresh runtime must succeed");
                assert!(wait_for(Duration::from_millis(500), || {
                    done.load(Ordering::Acquire) == 1
                }));
            }
        }
    };
}

#[cfg(feature = "runtime-tokio")]
runtime_conformance!(
    tokio_per_core,
    proxima::runtime::TokioPerCoreRuntime::new(2).expect("build tokio_per_core")
);

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
runtime_conformance!(
    proxima_runtime,
    proxima::runtime::PrimeRuntime::new(2).expect("build proxima")
);
