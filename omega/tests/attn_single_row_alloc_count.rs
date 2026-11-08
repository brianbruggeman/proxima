//! The single-query shared-K dispatch path must not add a per-call allocation
//! over the decode split, and a warm plan must allocate the same amount call to
//! call. `#[global_allocator]` is process-wide, so this file is the only
//! consumer of `proxima_test::alloc_count` in this test binary; `cargo val`
//! gives each test its own process.

#![cfg(all(feature = "alloc-count", feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::QuantizedBlock;
use proxima_test::alloc_count::{CountingAllocator, allocations};

mod gemma4_rows_support;
mod support;

use gemma4_rows_support::{Geometry, fixture_with};
use support::{as_named_blocks, production_numeric_policy};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// The first call warms whatever the plan resolves lazily; the second and third
/// are the steady-state pair this file compares.
fn warm_call_allocations(
    plan: &omega::Plan,
    blocks: &[(&str, QuantizedBlock<'_>)],
) -> (usize, usize) {
    omega::execute_plan_named(plan, blocks).expect("the warm-up call executes");

    let before_second = allocations();
    omega::execute_plan_named(plan, blocks).expect("the second warm call executes");
    let second = allocations() - before_second;

    let before_third = allocations();
    omega::execute_plan_named(plan, blocks).expect("the third warm call executes");
    let third = allocations() - before_third;

    (second, third)
}

#[test]
fn a_warm_single_query_shared_k_plan_allocates_like_the_decode_split() {
    let policy = production_numeric_policy();
    let fixture = fixture_with(1, 512, Geometry::ONE_KV_HEAD);
    let named = as_named_blocks(&fixture.named);
    let roots = [fixture.logits];

    let legacy = omega::plan_named(&fixture.program, &fixture.symbols, &named, &roots, policy)
        .expect("the single-query decode plan resolves");
    let mut shared_k =
        omega::plan_named(&fixture.program, &fixture.symbols, &named, &roots, policy)
            .expect("the single-query shared-K candidate plan resolves");
    shared_k
        .set_attention_variant(omega::AttentionVariant {
            kv_reuse: omega::AttentionKvReuse::SharedK,
            ..omega::AttentionVariant::default()
        })
        .expect("the single-query shared-K variant is admitted");
    assert!(
        shared_k
            .kernel_keys()
            .expect("shared-K identities are collected")
            .iter()
            .any(|key| key.contains("_kv_shared_k")),
        "the measured plan must be the shared-K form"
    );

    let (legacy_second, legacy_third) = warm_call_allocations(&legacy, &named);
    let (shared_second, shared_third) = warm_call_allocations(&shared_k, &named);

    eprintln!(
        "warm allocations: decode split {legacy_second}/{legacy_third}, \
         shared-K {shared_second}/{shared_third}"
    );
    assert_eq!(
        legacy_second, legacy_third,
        "the decode-split plan must allocate identically on consecutive warm calls"
    );
    assert_eq!(
        shared_second, shared_third,
        "the shared-K plan must allocate identically on consecutive warm calls"
    );
    assert_eq!(
        shared_second, legacy_second,
        "the single-query shared-K form must not add a per-call allocation over the decode split"
    );
}
