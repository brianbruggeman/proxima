#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! C13 — config + verb-fluent typestate builder.
//!
//! Headline arms:
//!   - `proxima_config_default`            (design-favors: proxima — our shape)
//!   - `proxima_config_validate`           (design-favors: proxima)
//!   - `proxima_config_from_env`           (design-favors: neutral)
//!   - `proxima_builder_typestate_start`   (design-favors: proxima — typestate)
//!   - `proxima_config_round_trip`         (design-favors: proxima)
//!   - `opentelemetry_sdk_tracerprovider`  (design-favors: incumbent home turf —
//!     full TracerProvider + processor +
//!     InMemorySpanExporter pipeline)
//!   - `tracing_subscriber_fmt_finish`     (design-favors: incumbent home turf —
//!     layered subscriber composition)
//!   - `raw_conflaguration_no_typestate`   (design-favors: neutral — bare
//!     conflaguration::Settings load)

extern crate alloc;

use std::hint::black_box;

use conflaguration::{Settings, Validate};
use criterion::{Criterion, criterion_group, criterion_main};
use proxima_telemetry::config::{ResourceTag, TelemetryConfig};
use proxima_telemetry::pipes::NullPipe;
use proxima_telemetry::recorder::Recorder;

fn bench_proxima_config_default(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c13_config");
    group.bench_function("proxima_config_default", |bencher| {
        bencher.iter(|| black_box(TelemetryConfig::default()));
    });
    group.finish();
}

fn bench_proxima_config_validate(criterion: &mut Criterion) {
    let cfg = TelemetryConfig::default();
    let mut group = criterion.benchmark_group("c13_config");
    group.bench_function("proxima_config_validate", |bencher| {
        bencher.iter(|| black_box(black_box(&cfg).validate().is_ok()));
    });
    group.finish();
}

fn bench_proxima_config_from_env(criterion: &mut Criterion) {
    // hard-set the env once outside the loop so per-iteration cost is the
    // env scan + parse, not the std::env::set_var path.
    unsafe {
        std::env::set_var("PROXIMA_TELEMETRY_RING_SPANS", "8192");
        std::env::set_var("PROXIMA_TELEMETRY_RING_EVENTS", "8192");
        std::env::set_var("PROXIMA_TELEMETRY_RING_LOGS", "8192");
        std::env::set_var("PROXIMA_TELEMETRY_RING_METRICS", "16384");
        std::env::set_var("PROXIMA_TELEMETRY_CORE_COUNT", "4");
    }
    let mut group = criterion.benchmark_group("c13_config");
    group.bench_function("proxima_config_from_env", |bencher| {
        bencher.iter(|| black_box(TelemetryConfig::from_env().expect("from_env")));
    });
    group.finish();
}

fn bench_proxima_builder_typestate_start(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c13_config");
    group.bench_function("proxima_builder_typestate_start", |bencher| {
        bencher.iter(|| {
            let recorder = Recorder::builder()
                .core_count(1)
                .pipe(NullPipe::new())
                .start()
                .expect("start");
            black_box(recorder)
        });
    });
    group.finish();
}

fn bench_proxima_config_round_trip(criterion: &mut Criterion) {
    let cfg = TelemetryConfig::builder()
        .resource(alloc::vec![
            ResourceTag {
                key: "service.name".to_string(),
                value: "judi-api".to_string()
            },
            ResourceTag {
                key: "service.version".to_string(),
                value: "1.2.0".to_string()
            },
        ])
        .build();
    let mut group = criterion.benchmark_group("c13_config");
    group.bench_function("proxima_config_round_trip", |bencher| {
        bencher.iter(|| {
            let recorder = Recorder::from_config(black_box(&cfg))
                .start()
                .expect("start");
            black_box(recorder.to_config())
        });
    });
    group.finish();
}

// Home-turf incumbent arm: OTel SDK's TracerProvider builder pipeline with
// a SimpleSpanProcessor + InMemorySpanExporter. This is the workload their
// builder machinery was designed around — pipeline composition with
// processor selection, exporter injection, and resource attribution.
fn bench_opentelemetry_sdk_tracerprovider(criterion: &mut Criterion) {
    use opentelemetry_sdk::trace::SdkTracerProvider;

    let mut group = criterion.benchmark_group("c13_config");
    group.bench_function("opentelemetry_sdk_tracerprovider", |bencher| {
        bencher.iter(|| {
            let provider = SdkTracerProvider::builder()
                .with_simple_exporter(
                    opentelemetry_sdk::trace::InMemorySpanExporterBuilder::new().build(),
                )
                .build();
            black_box(provider)
        });
    });
    group.finish();
}

// Home-turf incumbent arm: tracing_subscriber::fmt builder construction.
// Their design point is layered composition (EnvFilter + fmt + custom
// layers); the fmt builder is the canonical entry point.
fn bench_tracing_subscriber_fmt_finish(criterion: &mut Criterion) {
    use tracing_subscriber::fmt;

    let mut group = criterion.benchmark_group("c13_config");
    group.bench_function("tracing_subscriber_fmt_finish", |bencher| {
        bencher.iter(|| {
            let subscriber = fmt()
                .with_max_level(tracing::Level::INFO)
                .with_target(false)
                .finish();
            black_box(subscriber)
        });
    });
    group.finish();
}

// Neutral arm: raw conflaguration::Settings load without proxima's typestate
// builder on top. Exposes the floor cost of the Settings derive itself.
fn bench_raw_conflaguration_no_typestate(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c13_config");
    group.bench_function("raw_conflaguration_no_typestate", |bencher| {
        bencher.iter(|| {
            let cfg: TelemetryConfig = TelemetryConfig::from_env().expect("from_env");
            black_box(cfg)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_proxima_config_default,
    bench_proxima_config_validate,
    bench_proxima_config_from_env,
    bench_proxima_builder_typestate_start,
    bench_proxima_config_round_trip,
    bench_opentelemetry_sdk_tracerprovider,
    bench_tracing_subscriber_fmt_finish,
    bench_raw_conflaguration_no_typestate,
);
criterion_main!(benches);
