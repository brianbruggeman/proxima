//! Regression guard for `msl::kernel_cache_key` missing the reduce-extent
//! axis `cooperative_reduce_width` derives from -- two `Reduce`s sharing
//! every field `kernel_cache_key` DID key on (rank, output rank, operand
//! count, body, reduce op, init, dtype, codecs, output axes) but picking a
//! DIFFERENT lane width purely from concrete reduce extents used to collide
//! on one cache entry (`metal::pipeline_for`, `metal::PIPELINE_CACHE`),
//! silently reusing a stale compiled kernel whose baked-in width no longer
//! matches the dispatch. Both cases below share every field `entry_name`
//! and the pre-fix `kernel_cache_key` covered; only the concrete extent
//! (17 vs 385) differs.
//!
//! Every test here requires a real Metal device and this feature on --
//! neither skips.

#![cfg(all(
    feature = "metal",
    feature = "metal-wide-cooperative-reduce",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_telemetry::emit::EnvFilter;
use proxima_telemetry::emit::global;
use proxima_telemetry::pipes::InMemoryPipe;
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::tag::Tag;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
    append, evaluate, infer, projection,
};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// One `Reduce` over a `(1, cols)` input's last axis, output `(1,)` --
/// `cols` alone controls the reduction extent
/// `msl::cooperative_reduce_width` sizes the dispatch from, matching every
/// other field (rank, output rank, operand count, body, init) across calls.
fn single_row_reduce_program(cols: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(cols)],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: input,
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    program
}

/// `cooperative_reduce_width(reduction_total=385) = next_mul_32(ceil(385/4))
/// = next_mul_32(97) = 128` -- four simdgroups, so a cache hit compiled here
/// bakes a multi-simdgroup `threadgroup partials[4]` tail fold into source
/// (`msl::push_cooperative_reduce_tail`).
const WIDE_COLS: u32 = 385;

/// `cooperative_reduce_width(reduction_total=17) = next_mul_32(ceil(17/4))
/// = next_mul_32(5) = 32` -- one simdgroup, the plain `simd_sum` tail. Below
/// `WIDE_COLS`'s width of 128, so a dispatch sized for THIS extent launches
/// fewer threads (and fewer simdgroups) than `WIDE_COLS`'s compiled kernel
/// expects if that stale kernel is reused.
const NARROW_COLS: u32 = 17;

/// Runs `WIDE_COLS` then `NARROW_COLS` through `omega::execute` in the SAME
/// thread (so both share one thread-local `metal::PIPELINE_CACHE`), captures
/// the `pipeline cache lookup` trace log `metal::pipeline_for` emits on
/// every hit/miss, and asserts: (1) the two calls' `kernel_cache_key`
/// strings differ (the fix -- pre-fix they collided), and (2) `NARROW_COLS`'s
/// metal result still agrees with `cpu::evaluate` (pre-fix, the second call
/// silently hit the WIDE-compiled kernel's stale multi-simdgroup tail fold
/// and read uninitialized `threadgroup` memory for its missing simdgroups).
#[test]
fn cooperative_reduce_extents_never_share_a_pipeline_cache_entry() {
    let pipe = InMemoryPipe::new();
    let recorder = Recorder::builder()
        .pipe(pipe.clone())
        .core_count(1)
        .install()
        .expect("telemetry recorder installs as process default");
    global::install(EnvFilter::parse("trace"));

    let wide_program = single_row_reduce_program(WIDE_COLS);
    let wide_input = random_vec(0x5000 + u64::from(WIDE_COLS), WIDE_COLS as usize);
    infer(&wide_program, &[]).expect("wide program infers");
    let wide_metal = omega::execute(
        &wide_program,
        &[],
        &[QuantizedBlock::Float32(&wide_input)],
        &[],
        NumericPolicy::default(),
    )
    .expect("wide program executes on a real Metal device");
    let wide_cpu =
        evaluate(&wide_program, &[], &[&wide_input], &[]).expect("wide program cpu-evaluates");
    assert!(
        (wide_cpu.root()[0] - wide_metal.root()[0]).abs() <= 1e-4,
        "wide (cols={WIDE_COLS}) itself must already agree with cpu: cpu={}, metal={}",
        wide_cpu.root()[0],
        wide_metal.root()[0]
    );

    let narrow_program = single_row_reduce_program(NARROW_COLS);
    let narrow_input = random_vec(0x6000 + u64::from(NARROW_COLS), NARROW_COLS as usize);
    infer(&narrow_program, &[]).expect("narrow program infers");
    let narrow_metal = omega::execute(
        &narrow_program,
        &[],
        &[QuantizedBlock::Float32(&narrow_input)],
        &[],
        NumericPolicy::default(),
    )
    .expect("narrow program executes on a real Metal device, reusing the shared pipeline cache");
    let narrow_cpu = evaluate(&narrow_program, &[], &[&narrow_input], &[])
        .expect("narrow program cpu-evaluates");

    recorder.drain();
    let logs = pipe.logs();
    let cache_log_keys: Vec<(String, bool)> = logs
        .iter()
        .filter_map(|record| {
            let mut key = None;
            let mut hit = None;
            for tag in &record.attrs {
                let Tag::Scalar {
                    key: tag_key,
                    value,
                } = tag
                else {
                    continue;
                };
                match *tag_key {
                    "cache_key" => key = Some(value.to_string()),
                    "hit" => hit = Some(value.to_string() == "true"),
                    _ => {}
                }
            }
            key.zip(hit)
        })
        .collect();
    assert_eq!(
        cache_log_keys.len(),
        2,
        "expected one pipeline-cache-lookup log per omega::execute call, got {}: {:#?}",
        cache_log_keys.len(),
        cache_log_keys
    );
    let (wide_key, wide_hit) = &cache_log_keys[0];
    let (narrow_key, narrow_hit) = &cache_log_keys[1];
    println!(
        "wide (cols={WIDE_COLS}) cache_key={wide_key} hit={wide_hit}\n\
         narrow (cols={NARROW_COLS}) cache_key={narrow_key} hit={narrow_hit}"
    );
    assert!(
        !wide_hit,
        "the first call of a fresh process must be a compile, not a hit"
    );
    assert_ne!(
        wide_key, narrow_key,
        "wide and narrow reduces pick different cooperative_reduce_width values \
         and MUST NOT share a kernel_cache_key -- a shared key means the narrow \
         dispatch would silently reuse the wide kernel's stale lane width"
    );
    assert!(
        !narrow_hit,
        "narrow_key differs from wide_key, so this must be a genuine miss/compile, not a cache hit"
    );

    let max_abs_diff = (narrow_cpu.root()[0] - narrow_metal.root()[0]).abs();
    assert!(
        max_abs_diff <= 1e-4,
        "narrow (cols={NARROW_COLS}) must agree with cpu after the wide (cols={WIDE_COLS}) \
         reduce already populated the shared pipeline cache: cpu={}, metal={}, abs diff={max_abs_diff:e}",
        narrow_cpu.root()[0],
        narrow_metal.root()[0]
    );
}
