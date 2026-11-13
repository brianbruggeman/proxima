//! Per-op GPU time for the K = 1 cached-attention arms: localizes the
//! decode-split vs row-tiled delta to the attention and merge kernels instead
//! of attributing the whole-program difference to them.

#![cfg(all(
    feature = "metal",
    feature = "metal-attn-variants",
    feature = "metal-output-placement",
    feature = "instrument",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod gemma4_rows_support;
mod support;

use gemma4_rows_support::{Geometry, fixture_with};
use support::{as_named_blocks, production_numeric_policy};

fn attention_totals(timings: &[omega::metal::OpGpuTiming]) -> (u64, usize, u64) {
    let mut attention_ns = 0_u64;
    let mut attention_ops = 0_usize;
    let mut total_ns = 0_u64;
    for timing in timings {
        total_ns += timing.gpu_ns;
        if timing.kind.contains("attention") {
            attention_ns += timing.gpu_ns;
            attention_ops += 1;
        }
    }
    (attention_ns, attention_ops, total_ns)
}

fn largest_ops(timings: &[omega::metal::OpGpuTiming], count: usize) -> Vec<(String, u64)> {
    let mut sorted: Vec<(String, u64)> = timings
        .iter()
        .map(|timing| (timing.kind.to_string(), timing.gpu_ns))
        .collect();
    sorted.sort_by_key(|entry| core::cmp::Reverse(entry.1));
    sorted.truncate(count);
    sorted
}

#[test]
fn single_query_attention_arms_report_per_op_gpu_time() {
    let policy = production_numeric_policy();
    let fixture = fixture_with(1, 512, Geometry::ONE_KV_HEAD);
    let named = as_named_blocks(&fixture.named);
    let roots = [fixture.logits];

    let legacy = omega::plan_named(&fixture.program, &fixture.symbols, &named, &roots, policy)
        .expect("the decode-split plan resolves");
    let mut row_tiled =
        omega::plan_named(&fixture.program, &fixture.symbols, &named, &roots, policy)
            .expect("the row-tiled candidate plan resolves");
    row_tiled
        .set_attention_variant(omega::AttentionVariant {
            kv_reuse: omega::AttentionKvReuse::SharedK,
            ..omega::AttentionVariant::default()
        })
        .expect("the single-query shared-K variant is admitted");

    let (_, legacy_timings) =
        omega::execute_plan_named_with_placements_op_timed(&legacy, &named, &[], &[])
            .expect("the decode-split plan times");
    let (_, tiled_timings) =
        omega::execute_plan_named_with_placements_op_timed(&row_tiled, &named, &[], &[])
            .expect("the row-tiled plan times");

    let (legacy_q, legacy_ops, legacy_total) = attention_totals(&legacy_timings);
    let (tiled_q, tiled_ops, tiled_total) = attention_totals(&tiled_timings);
    let legacy_largest = largest_ops(&legacy_timings, 6);
    let tiled_largest = largest_ops(&tiled_timings, 6);

    eprintln!(
        "decode split: attention {legacy_q} ns over {legacy_ops} ops, total {legacy_total} ns, \
         largest {legacy_largest:?}"
    );
    eprintln!(
        "row tiled: attention {tiled_q} ns over {tiled_ops} ops, total {tiled_total} ns, \
         largest {tiled_largest:?}"
    );

    assert_eq!(legacy_ops, 2, "one decode-split attention op per layer");
    assert_eq!(tiled_ops, 2, "one row-tiled attention op per layer");
    assert!(
        legacy_q > 0 && tiled_q > 0,
        "attention kernels must be timed"
    );
    assert!(
        legacy_q <= legacy_total && tiled_q <= tiled_total,
        "attention time is a slice of the plan, not more than it"
    );
}
