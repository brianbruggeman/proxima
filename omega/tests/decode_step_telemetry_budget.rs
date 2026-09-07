//! Regression guard for the per-op telemetry level fix: `metal::pipeline_for`
//! used to fire a `debug!("pipeline cache lookup")` on EVERY op, every call
//! (`metal.rs`'s own history — the cache-key fix that added the debug-level
//! version overflowed the 4096-slot telemetry ring at 616 events per decode
//! step in the steady state, 888 dropped / 512 flushed over 8 steps). That
//! event is `trace!` now (per-op, per-step cadence — see this module's own
//! level table), so a caller who never raises the filter above `debug` sees
//! none of it.
//!
//! `SHAPE` below stands in for one decode step's op count: a chain of eight
//! distinct unary [`ScalarOp`]s, each with its own `kernel_cache_key`, so
//! [`omega::metal::pipeline_for`] is called once per op, exactly the cadence
//! `execute_plan`'s dispatch loop drives on a real forward pass. The first
//! [`omega::execute`] call is a cold cache (all misses, one compile per op);
//! the second is the steady-state case (all hits) this campaign's own oracle
//! test observed emitting 616 events on the real decode loop -- here it must
//! emit fewer than 64 events at `debug` level or above.
//!
//! Runs on the fake (synthetic-program) path rather than the real openchat
//! checkpoint the oracle test in `proxima-model-interop` depends on, so this
//! is a cheap, always-on regression test with no host-local GGUF fixture.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::sync::atomic::Ordering;

use proxima_telemetry::emit::EnvFilter;
use proxima_telemetry::emit::global;
use proxima_telemetry::pipes::CountingPipe;
use proxima_telemetry::recorder::Recorder;
use proxima_tensor::{
    DType, Extent, IndexMap, NumericPolicy, Op, QuantizedBlock, ScalarOp, append, projection,
};

const EXTENT: u32 = 8;

/// One `Input -> body_0(x) -> body_1(...) -> ... -> body_n(...)` chain,
/// each stage a distinct [`ScalarOp`] so each stage compiles to its own
/// `kernel_cache_key` -- eight ops, eight independent pipeline-cache slots,
/// standing in for one decode step's per-op dispatch count.
fn chained_unary_program(bodies: &[ScalarOp]) -> Vec<Op> {
    let mut program = Vec::new();
    let mut current = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(EXTENT)],
            name: None,
        },
    );
    for body in bodies {
        current = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: *body,
                operands: vec![(current, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
    }
    program
}

#[test]
fn one_steady_state_decode_step_stays_under_the_debug_level_event_budget() {
    let bodies = [
        ScalarOp::Identity,
        ScalarOp::Negate,
        ScalarOp::Reciprocal,
        ScalarOp::Exponential,
        ScalarOp::Logarithm,
        ScalarOp::SquareRoot,
        ScalarOp::Tanh,
        ScalarOp::Erf,
    ];
    let program = chained_unary_program(&bodies);
    let input = vec![0.25f32; EXTENT as usize];
    let blocks = [QuantizedBlock::Float32(&input)];

    let counting_pipe = CountingPipe::new();
    let logs = Arc::clone(&counting_pipe.logs);
    let recorder = Recorder::builder()
        .pipe(counting_pipe)
        .core_count(1)
        .install()
        .expect("telemetry recorder installs as process default");
    global::install(EnvFilter::parse("debug"));

    // cold cache: every op is a genuine miss/compile, warming
    // `metal::PIPELINE_CACHE` for the steady-state run below.
    omega::execute(&program, &[], &blocks, &[], NumericPolicy::default()).expect("cold-cache run executes on a real Metal device");
    recorder.drain();
    logs.store(0, Ordering::Relaxed);

    // steady state: every op hits the pipeline cache the cold run warmed --
    // the exact cadence a real decode step's Nth token repeats.
    omega::execute(&program, &[], &blocks, &[], NumericPolicy::default()).expect("steady-state run executes on a real Metal device");
    recorder.drain();

    let steady_state_debug_or_above_events = logs.load(Ordering::Relaxed);
    assert!(
        steady_state_debug_or_above_events < 64,
        "one steady-state decode step emitted {steady_state_debug_or_above_events} telemetry \
         events at debug level or above -- per-op pipeline-cache-lookup events must be trace, \
         not debug, or this regresses the 4096-slot ring overflow this test guards against"
    );
}
