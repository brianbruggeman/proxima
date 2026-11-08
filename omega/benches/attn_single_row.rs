//! K = 1 cached attention: the row-tiled shared-K lowering against the decode
//! split, on the same program, data and numeric policy. Only the attention
//! form differs, so the delta is the form's own cost on a real decode step.
//!
//! Swept over cache capacity because the two forms do not have the same scaling
//! in keys, and every prior tuning cell in `omega-runtime.toml` was measured at
//! 512 keys -- the capacity a single dispatch sees at 32k context is 64x that.
//!
//! Run on a quiet machine (no other cargo/rustc process above ~20% CPU):
//! `CARGO_TARGET_DIR=<scratch>/target cargo bench -p proxima-omega \
//!   --bench attn_single_row --features metal-attn-variants`
//!
//! One number per arm is not a result: criterion reports the distribution and
//! the comparison against the prior run; read the delta against the spread.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
use std::hint::black_box;
#[cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
use std::time::Duration;

#[cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
#[path = "../tests/gemma4_rows_support/mod.rs"]
mod gemma4_rows_support;

#[cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
#[path = "../tests/support/mod.rs"]
mod support;

/// Live cached rows the sweep binds, spanning the decode window through the
/// largest context the checkpoint serves (131072 trained, 262144 on the
/// incumbent's `-d`). Each capacity is its own bucket, so each is a distinct
/// compiled plan rather than the same plan re-run at a different length.
#[cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
const CACHED_CAPACITIES: [usize; 6] = [512, 4096, 16384, 32768, 65536, 131072];

#[cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
fn bench_single_row_attention(criterion: &mut Criterion) {
    use gemma4_rows_support::{Geometry, fixture_with};
    use support::{as_named_blocks, production_numeric_policy};

    let policy = production_numeric_policy();
    for cached_len in CACHED_CAPACITIES {
        let fixture = fixture_with(1, cached_len, Geometry::ONE_KV_HEAD);
        let named = as_named_blocks(&fixture.named);
        let roots = [fixture.logits];

        let decode_split =
            omega::plan_named(&fixture.program, &fixture.symbols, &named, &roots, policy)
                .expect("the single-query decode-split plan resolves");
        let mut row_tiled =
            omega::plan_named(&fixture.program, &fixture.symbols, &named, &roots, policy)
                .expect("the single-query row-tiled candidate plan resolves");
        row_tiled
            .set_attention_variant(omega::AttentionVariant {
                kv_reuse: omega::AttentionKvReuse::SharedK,
                ..omega::AttentionVariant::default()
            })
            .expect("the single-query shared-K variant is admitted");
        assert!(
            row_tiled
                .kernel_keys()
                .expect("row-tiled identities are collected")
                .iter()
                .any(|key| key.contains("_kv_shared_k")),
            "the measured plan must be the shared-K form"
        );

        let mut group = criterion.benchmark_group(format!("attn_single_row_cached_{cached_len}"));
        group.warm_up_time(Duration::from_millis(300));
        group.measurement_time(Duration::from_millis(1500));
        group.bench_function("decode_split", |bencher| {
            bencher.iter(|| {
                black_box(
                    omega::execute_plan_named(&decode_split, &named)
                        .expect("the decode-split plan executes"),
                )
            });
        });
        group.bench_function("row_tiled_shared_k", |bencher| {
            bencher.iter(|| {
                black_box(
                    omega::execute_plan_named(&row_tiled, &named)
                        .expect("the row-tiled plan executes"),
                )
            });
        });
        group.finish();
    }
}

#[cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
criterion_group!(benches, bench_single_row_attention);
#[cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
))]
criterion_main!(benches);

#[cfg(not(all(
    feature = "metal",
    feature = "metal-attn-variants",
    target_os = "macos"
)))]
fn main() {
    eprintln!("attn_single_row needs metal + metal-attn-variants on macOS");
    std::process::exit(1);
}
