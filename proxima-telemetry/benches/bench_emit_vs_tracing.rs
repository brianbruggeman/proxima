#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! Home-turf incumbent arm (disciplined-component gate 13): proxima's emit
//! filter decision vs `tracing-subscriber`'s `EnvFilter` on the SAME directive
//! string and the SAME (target, level) callsites.
//!
//! HONEST SCOPE NOTE — the two architectures filter at different points:
//! - proxima filters at DRAIN: `CompiledEmit::decide(target, coord)` is paid per
//!   record, every record. That is what the `proxima_*` arms measure (a bare
//!   function call, no dispatch).
//! - tracing filters at the CALLSITE: `EnvFilter` decides via per-callsite
//!   cached `Interest`, so a statically-disabled site is near-free on repeat
//!   (the `tracing_event_dropped` arm) while an enabled site pays build +
//!   dispatch (`tracing_event_kept`). That is the incumbent's real per-event
//!   cost, measured exactly the way tracing's own `benches/filter.rs` does it.
//!
//! So `proxima_decide_*` vs `tracing_event_*` is decision-vs-(dispatch+decision);
//! the takeaway is each side's real per-record filter cost in its own model, not
//! a single "X faster" number.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_telemetry::emit::{CallsiteGate, Coord, EnvFilter as ProximaEnvFilter};
use proxima_telemetry::level::Level;
use tracing::{Event, Id, Metadata, span};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

const FILTER: &str = "proxima::h2=debug,proxima::h2::hpack=trace,proxima=info,downstream::store=warn";

/// A subscriber that is enabled but does nothing — copied from tracing's own
/// `benches/filter.rs` so the EnvFilter arm is measured on its home turf.
struct EnabledSubscriber;

impl tracing::Subscriber for EnabledSubscriber {
    fn new_span(&self, _: &span::Attributes<'_>) -> Id {
        Id::from_u64(0xDEAD_FACE)
    }
    fn event(&self, _: &Event<'_>) {}
    fn record(&self, _: &Id, _: &span::Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }
    fn enter(&self, _: &Id) {}
    fn exit(&self, _: &Id) {}
}

fn bench(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("emit_vs_tracing");

    // proxima: the per-record decision (paid at drain, every record).
    let compiled = ProximaEnvFilter::parse(FILTER);
    let info = Coord::from(Level::INFO);
    let trace = Coord::from(Level::TRACE);
    group.bench_function("proxima_decide_kept", |bencher| {
        bencher
            .iter(|| black_box(compiled.decide(black_box("proxima::h2::frame"), black_box(info))));
    });
    group.bench_function("proxima_decide_dropped", |bencher| {
        bencher
            .iter(|| black_box(compiled.decide(black_box("proxima::h2::frame"), black_box(trace))));
    });
    group.bench_function("proxima_decide_default", |bencher| {
        bencher.iter(|| black_box(compiled.decide(black_box("other::crate"), black_box(info))));
    });

    // the callsite-cached gate: proxima's disabled-fast-path equivalent of
    // tracing's cached Interest. Prime it once (computes + caches Drop), then
    // measure the steady-state cached hit — no `decide` scan, no record built.
    let gate = CallsiteGate::new();
    let _ = gate.decide(1, || compiled.decide("proxima::h2::frame", trace));
    group.bench_function("proxima_gate_cached_drop", |bencher| {
        bencher.iter(|| {
            black_box(gate.decide(black_box(1), || {
                compiled.decide("proxima::h2::frame", trace)
            }))
        });
    });

    // tracing: the per-event cost through dispatch + EnvFilter (callsite-cached).
    let filter: EnvFilter = FILTER.parse().expect("filter parses");
    tracing::subscriber::with_default(EnabledSubscriber.with(filter), || {
        // kept: proxima::h2::frame at info (>= debug floor) -> builds + dispatches
        group.bench_function("tracing_event_kept", |bencher| {
            bencher.iter(|| {
                tracing::event!(target: "proxima::h2::frame", tracing::Level::INFO, value = black_box(1u64));
            });
        });
        // dropped: proxima::h2::frame at trace (< debug floor) -> cached reject
        group.bench_function("tracing_event_dropped", |bencher| {
            bencher.iter(|| {
                tracing::event!(target: "proxima::h2::frame", tracing::Level::TRACE, value = black_box(1u64));
            });
        });
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
