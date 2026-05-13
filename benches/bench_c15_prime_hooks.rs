#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! C15 bench: prime::spawn span-carry overhead vs baselines.
//!
//! Named baselines per disciplined-component skill §6 and §13:
//!
//! 1. prime_spawn_no_span_carry   — control floor (no span set, no carry)
//! 2. prime_spawn_with_span_carry — parent has a span; child inherits it
//!    via `telemetry::Spanned<T>` wrapping the future (Wave D Phase 1 —
//!    the span no longer rides as a side-channel field on `SpawnRequest`
//!    / the executor's task slab; overhead vs baseline is what we measure)
//! 3. tracing_instrument_spawn_via_tokio — incumbent home-turf:
//!    `tokio::spawn(future.instrument(Span::current()))`, the canonical
//!    idiom in the ecosystem for carrying spans across spawns
//! 4. prime_spawn_cross_core_with_carry — cross-core SpawnRequest with
//!    span carry (no equivalent in tokio world — proxima-only design point)
//!
//! Target delta for arm 2 vs arm 1: < 5%.
//! Arm 3 sets the incumbent home-turf that C15 must meet-or-beat on the
//! spawn+carry operation.

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "c15-prime-hooks")]
struct NoopSink;

#[cfg(feature = "c15-prime-hooks")]
impl proxima_telemetry::trace::SpanSink for NoopSink {
    fn emit(&mut self, _record: proxima_telemetry::trace::SpanRecord) {}
}

#[cfg(feature = "c15-prime-hooks")]
fn spanned_guard()
-> proxima_telemetry::trace::SpanGuard<NoopSink, proxima_telemetry::clock::MonotonicCounter> {
    use proxima_telemetry::id::{SpanId, TraceId};
    use proxima_telemetry::trace::SpanBuilder;

    let trace_id = TraceId::from_bytes([0xab; 16]);
    let span_id = SpanId::from_bytes([0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xf0, 0x0d]);
    SpanBuilder::new("bench_parent", trace_id, span_id)
        .start(
            &proxima_telemetry::clock::MonotonicCounter::new(0),
            NoopSink,
        )
        .enter(proxima_telemetry::clock::MonotonicCounter::new(0))
}

fn bench_c15(criterion: &mut Criterion) {
    #[cfg(not(feature = "c15-prime-hooks"))]
    let _ = criterion;
    #[cfg(feature = "c15-prime-hooks")]
    {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        use proxima::runtime::prime::core::local_executor::LocalExecutor;
        use proxima_telemetry::spanned::Spanned;

        let mut group = criterion.benchmark_group("c15_prime_hooks");
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(3));

        // arm 1 — control floor: prime::spawn with no span carry
        {
            let executor = LocalExecutor::new();
            executor.arm();
            let counter = Arc::new(AtomicUsize::new(0));
            group.bench_function("prime_spawn_no_span_carry", |bencher| {
                bencher.iter(|| {
                    let counter = counter.clone();
                    executor.spawn_local(async move {
                        counter.fetch_add(1, Ordering::Relaxed);
                    });
                    executor.tick();
                });
            });
            executor.disarm();
        }

        // arm 2 — C15 design point: prime::spawn with span carry, now via
        // `Spanned` wrapping the future instead of a side-channel field.
        {
            let executor = LocalExecutor::new();
            executor.arm();
            let counter = Arc::new(AtomicUsize::new(0));
            group.bench_function("prime_spawn_with_span_carry", |bencher| {
                bencher.iter(|| {
                    let counter = counter.clone();
                    let spanned = Spanned::new(
                        async move {
                            counter.fetch_add(1, Ordering::Relaxed);
                        },
                        spanned_guard(),
                    );
                    executor.spawn_local_pin(Box::pin(spanned));
                    executor.tick();
                });
            });
            executor.disarm();
        }

        // arm 3 — incumbent home-turf: tokio + tracing Instrument
        // This is the canonical idiom to carry a span across a spawn boundary.
        // Note: spawn_local requires a LocalSet; we wrap the bench in one so
        // the comparison is apples-to-apples with the actual usage pattern.
        #[cfg(feature = "runtime-tokio")]
        {
            use tracing::Instrument;

            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            let local_set = tokio::task::LocalSet::new();
            let counter = Arc::new(AtomicUsize::new(0));
            rt.block_on(local_set.run_until(async {
                group.bench_function("tracing_instrument_spawn_via_tokio", |bencher| {
                    bencher.iter(|| {
                        let counter = counter.clone();
                        let span = tracing::info_span!("bench_parent");
                        let _enter = span.enter();
                        let _handle = tokio::task::spawn_local(
                            async move {
                                counter.fetch_add(1, Ordering::Relaxed);
                            }
                            .instrument(tracing::Span::current()),
                        );
                    });
                });
            }));
        }

        // arm 4 — cross-core spawn with carry: the span rides inside the
        // dispatched future itself (`Spanned` wraps it before it's boxed),
        // not as a second `SpawnRequest` field the receiving core has to
        // unpack.
        #[cfg(all(
            feature = "runtime-prime-executor",
            feature = "runtime-prime-reactor",
            feature = "runtime-prime-inbox-alloc",
        ))]
        {
            use proxima::runtime::prime::os::core_shard;
            use proxima::runtime::CoreId;

            let handle =
                core_shard::launch_with_lanes(CoreId(0), None, 8, 256).expect("launch core");
            let counter = Arc::new(AtomicUsize::new(0));
            group.bench_function("prime_spawn_cross_core_with_carry", |bencher| {
                bencher.iter(|| {
                    let counter = counter.clone();
                    let spanned = Spanned::new(
                        async move {
                            counter.fetch_add(1, Ordering::Relaxed);
                        },
                        spanned_guard(),
                    );
                    let _ = handle.dispatch_send(Box::pin(spanned));
                });
            });
            handle.shutdown_and_join().expect("shutdown");
        }

        group.finish();
    }
}

criterion_group!(c15_benches, bench_c15);
criterion_main!(c15_benches);
