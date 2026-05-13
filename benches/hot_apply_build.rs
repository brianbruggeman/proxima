#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Hot-apply build cost. Measures the hidden half of a config-driven
//! pipe swap: `crate::load::load(spec)` walking a chain spec into
//! a built `PipeHandle`. The other half — the `ArcSwap` install
//! itself — lives in `swap_under_load` and is already nanoseconds.
//!
//! The build path allocates one pipe per middleware layer, looks
//! each up in the factory registry, and chains the resulting handles.
//! When `proxima apply` fires, this is the part that runs BEFORE the
//! atomic swap can happen — so deep chains add to apply latency.
//!
//! Three shapes:
//! - `bare_synth` — leaf only (synth upstream), no middleware
//! - `chain_5_layer` — synth + retry + transform + rate_limit
//!   + middleware[auth, isolate] (matches the depth common in real configs)
//! - `chain_15_layer` — same factories repeated to stress allocator
//!   + factory-registry lookups

use std::sync::Arc;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::load::{LoadContext, Spec, load};
use serde_json::{Value, json};

fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
}

fn bare_synth_spec() -> Value {
    json!({
        "type": "synth",
        "name": "leaf",
        "status": 200,
        "body": "ok",
    })
}

fn chain_5_layer_spec() -> Value {
    json!({
        "type": "synth",
        "name": "leaf",
        "status": 200,
        "body": "ok",
        "retry": {
            "max_attempts": 2,
            "on_status": [502, 503]
        },
        "transform": {
            "request": [{"set_header": {"name": "x-trace", "value": "bench"}}]
        },
        "rate_limit": {
            "capacity": 100,
            "refill_per_sec": 50,
            "key": "path_and_method"
        },
        "middleware": [
            {"type": "auth", "allow": ["t-1"]},
            {"type": "isolate"}
        ]
    })
}

fn chain_15_layer_spec() -> Value {
    let middleware: Vec<Value> = (0..15)
        .map(|index| match index % 4 {
            0 => json!({"type": "auth", "allow": [format!("token-{index}")]}),
            1 => json!({"type": "isolate"}),
            2 => json!({"type": "transform", "request": [{"set_header": {"name": format!("x-layer-{index}"), "value": "v"}}]}),
            _ => json!({"type": "rate_limit", "capacity": 100, "refill_per_sec": 50, "key": "path_and_method"}),
        })
        .collect();
    json!({
        "type": "synth",
        "name": "leaf",
        "status": 200,
        "body": "ok",
        "middleware": middleware,
    })
}

fn build_chain(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hot_apply_build");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    let runtime = build_runtime();
    let context = Arc::new(
        runtime
            .block_on(async { LoadContext::with_noop_telemetry() })
            .expect("LoadContext"),
    );

    let cases: &[(&str, Value)] = &[
        ("bare_synth", bare_synth_spec()),
        ("chain_5_layer", chain_5_layer_spec()),
        ("chain_15_layer", chain_15_layer_spec()),
    ];
    for (label, spec) in cases {
        let context = context.clone();
        let spec = spec.clone();
        group.bench_function(*label, |bencher| {
            bencher.to_async(&runtime).iter(|| {
                let context = context.clone();
                let spec = spec.clone();
                async move {
                    let handle = load(Spec::Inline(spec), &context).await.expect("load");
                    std::hint::black_box(handle);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, build_chain);
criterion_main!(benches);
