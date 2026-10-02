//! A plan refitted in place across a KV bucket crossing must execute exactly
//! as a plan built from scratch at the new extent. `Plan::refit_symbols` patches
//! the fused cached-attention ops, re-resolves only their steps and uniform
//! buffers, and keeps everything else; this file runs both plans on a real
//! device over the same named blocks and compares every requested output bit
//! for bit (not within a tolerance: the refit changes no arithmetic, so any
//! difference is a stale piece of per-position state).
//!
//! Both attention kinds a decode plan carries are covered: the two-range
//! kind (`cached_key_rows != 0`, the gemma4 / mistral two-range shape) and the
//! single-range kind (`cached_key_rows == 0`, the padded merged-KV shape).
//! Execution goes through the placement executor with no placements -- the
//! only executor that reads a plan's resolved steps and plan-owned uniform
//! buffers, which are exactly the state a refit has to keep correct.
//! Each crossing runs twice -- once with the plan executed BEFORE the refit
//! (resolved steps and plan-owned uniforms already built, so the in-place
//! re-resolve path runs) and once refitted cold.

#![cfg(all(
    feature = "metal",
    feature = "metal-output-placement",
    feature = "cached-attention-streaming",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::{NodeId, NumericPolicy};

mod support;
use support::{
    RealForwardFixture, as_named_blocks, production_numeric_policy,
    real_forward_fixture_with_cached_len, real_single_range_forward_fixture_with_padding,
};

fn outputs_bits(
    plan: &omega::Plan,
    owned: &[(String, Vec<f32>)],
    roots: &[NodeId],
) -> Vec<Vec<u32>> {
    let named = as_named_blocks(owned);
    let evaluated = omega::execute_plan_named_with_placements(plan, &named, &[], &[])
        .expect("metal executes the plan through the placement executor");
    roots
        .iter()
        .map(|root| {
            let (data, _) = evaluated
                .get(*root)
                .expect("every requested root is returned");
            data.iter().map(|value| value.to_bits()).collect()
        })
        .collect()
}

fn fresh_plan(fixture: &RealForwardFixture, policy: NumericPolicy) -> omega::Plan {
    let (program, symbols, roots, owned) = fixture;
    let named = as_named_blocks(owned);
    omega::plan_named(program, symbols, &named, roots, policy).expect("metal plans the fixture")
}

fn assert_refit_matches_fresh(
    low: &RealForwardFixture,
    high: &RealForwardFixture,
    run_before_refit: bool,
    policy: NumericPolicy,
    label: &str,
) {
    let (_, _, roots, _) = high;
    let mut refit = fresh_plan(low, policy);
    if run_before_refit {
        let (_, _, low_roots, low_owned) = low;
        outputs_bits(&refit, low_owned, low_roots);
    }

    let refitted = refit
        .refit_symbols(&high.1)
        .expect("refit resolves at the new symbols");

    assert!(
        refitted,
        "{label}: the crossing is local to the fused attention ops, refit must apply"
    );
    let fresh = fresh_plan(high, policy);
    let after_refit = outputs_bits(&refit, &high.3, roots);
    let from_scratch = outputs_bits(&fresh, &high.3, roots);
    assert_eq!(
        after_refit.len(),
        roots.len(),
        "{label}: every root requested"
    );
    for (index, (got, want)) in after_refit.iter().zip(from_scratch.iter()).enumerate() {
        assert!(!want.is_empty(), "{label}: root {index} must carry data");
        assert_eq!(
            got, want,
            "{label}: root {index} differs between the refitted and the fresh plan"
        );
    }
}

fn single_range(cached_len: u64, padding: u64) -> RealForwardFixture {
    real_single_range_forward_fixture_with_padding(cached_len, 1, padding)
}

#[test]
fn single_range_refit_matches_a_fresh_plan_across_bucket_crossings() {
    for policy in [NumericPolicy::default(), production_numeric_policy()] {
        for (low_padding, high_padding, run_before) in
            [(5, 37, true), (5, 37, false), (37, 69, true)]
        {
            assert_refit_matches_fresh(
                &single_range(5, low_padding),
                &single_range(5, high_padding),
                run_before,
                policy,
                &format!(
                    "single-range padding {low_padding}->{high_padding} run_before={run_before}"
                ),
            );
        }
    }
}

#[test]
fn two_range_refit_matches_a_fresh_plan_across_bucket_crossings() {
    for policy in [NumericPolicy::default(), production_numeric_policy()] {
        for (low, high, run_before) in [
            (3u64, 35u64, true),
            (3, 35, false),
            (35, 67, true),
            (130, 162, true),
        ] {
            assert_refit_matches_fresh(
                &real_forward_fixture_with_cached_len(low),
                &real_forward_fixture_with_cached_len(high),
                run_before,
                policy,
                &format!("two-range cached_len {low}->{high} run_before={run_before}"),
            );
        }
    }
}

/// Verify and prefill steps carry several query rows per attention op, and
/// those ops size their scratch by the key range -- so a refit across a
/// crossing there must re-size it, most visibly once the context passes the
/// split-at-scale knee (`cached_len` 140 + 4 new rows, past 128 keys).
#[test]
fn multi_row_refit_matches_a_fresh_plan_across_bucket_crossings() {
    let multi_row = |cached_len: u64, padding: u64| {
        real_single_range_forward_fixture_with_padding(cached_len, 4, padding)
    };
    for policy in [NumericPolicy::default(), production_numeric_policy()] {
        for (cached_len, low_padding, high_padding, run_before) in [
            (5u64, 5u64, 37u64, true),
            (140, 4, 36, true),
            (140, 4, 36, false),
            (140, 36, 68, true),
            (140, 4, 1000, true),
            (140, 4, 1000, false),
        ] {
            assert_refit_matches_fresh(
                &multi_row(cached_len, low_padding),
                &multi_row(cached_len, high_padding),
                run_before,
                policy,
                &format!(
                    "4-row cached_len {cached_len} padding {low_padding}->{high_padding} run_before={run_before}"
                ),
            );
        }
    }
}

#[test]
fn refit_declines_and_leaves_the_plan_executable_when_the_new_position_count_moves() {
    let policy = production_numeric_policy();
    let decode = real_single_range_forward_fixture_with_padding(5, 1, 5);
    let verify = real_single_range_forward_fixture_with_padding(5, 4, 5);
    let (_, _, roots, owned) = &decode;
    let mut plan = fresh_plan(&decode, policy);
    let before = outputs_bits(&plan, owned, roots);

    let refitted = plan
        .refit_symbols(&verify.1)
        .expect("refit evaluates the new symbols");

    assert!(
        !refitted,
        "a different new_count reaches far more than the attention key range"
    );
    assert_eq!(
        outputs_bits(&plan, owned, roots),
        before,
        "a declined refit leaves the plan untouched"
    );
}
