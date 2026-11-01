use super::*;

use crate::spec::{CachedLayerRoots, DuplicateHeadPosition};

struct Fixture {
    program: Vec<Op>,
    outputs: Vec<NodeId>,
    layers: usize,
}

fn outputs_for(logits: NodeId, cache_roots: &[CachedLayerRoots]) -> Vec<NodeId> {
    let mut outputs = alloc::vec![logits];
    for (even, odd, value) in cache_roots {
        outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    outputs
}

fn single_range_fixture(layers: u32) -> Fixture {
    let (program, logits, cache_roots, _) = crate::spec::gqa_single_range_cached_forward_program(
        32,
        16,
        24,
        4,
        2,
        4,
        layers,
        false,
        DuplicateHeadPosition::None,
        false,
    )
    .expect("single-range fixture builds");
    let outputs = outputs_for(logits, &cache_roots);
    Fixture {
        program,
        outputs,
        layers: layers as usize,
    }
}

fn two_range_fixture(layers: u32) -> Fixture {
    let (program, logits, cache_roots) =
        crate::spec::gqa_cached_forward_program(32, 16, 24, 4, 2, 4, layers)
            .expect("two-range fixture builds");
    let outputs = outputs_for(logits, &cache_roots);
    Fixture {
        program,
        outputs,
        layers: layers as usize,
    }
}

fn bound_at(fixture: &Fixture, symbols: &[u64], fuse: bool) -> (Shapes, Vec<BoundOp>) {
    let shapes = crate::shape::infer(&fixture.program, symbols).expect("fixture infers");
    let bound = bind_with_fusion(
        &fixture.program,
        &shapes,
        &fixture.outputs,
        fuse,
        NumericPolicy::llama_relaxed(),
    )
    .expect("fixture binds");
    (shapes, prune_dead(bound, &fixture.outputs))
}

fn assert_refit_equals_fresh(fixture: &Fixture, crossings: &[(u64, u64)]) {
    for &(low, high) in crossings {
        let (low_shapes, low_bound) = bound_at(fixture, &[1, low], true);
        let (high_shapes, high_bound) = bound_at(fixture, &[1, high], true);
        let patches =
            refit_cached_attention_rows(&low_bound, &fixture.program, &low_shapes, &high_shapes)
                .unwrap_or_else(|| {
                    panic!("crossing {low}->{high} is local to the fused attention ops")
                });
        assert_eq!(
            patches.len(),
            fixture.layers,
            "crossing {low}->{high}: one patch per layer's fused attention op"
        );
        let mut refit = low_bound.clone();
        for (position, patched) in patches {
            refit[position] = patched;
        }
        assert_eq!(
            refit, high_bound,
            "crossing {low}->{high}: refit ops differ from a fresh bind"
        );
        assert_ne!(
            low_bound, high_bound,
            "crossing {low}->{high}: the fixture must actually change across it"
        );
    }
}

#[test]
fn refit_matches_a_fresh_bind_across_single_range_bucket_crossings() {
    assert_refit_equals_fresh(
        &single_range_fixture(2),
        &[
            (5, 13),
            (13, 37),
            (37, 64),
            (64, 65),
            (64, 96),
            (1024, 1056),
        ],
    );
}

#[test]
fn refit_matches_a_fresh_bind_across_two_range_bucket_crossings() {
    assert_refit_equals_fresh(
        &two_range_fixture(2),
        &[(3, 9), (9, 33), (33, 65), (64, 96), (1024, 1056)],
    );
}

#[test]
fn refit_has_nothing_to_patch_when_the_symbols_did_not_move() {
    let fixture = single_range_fixture(2);
    let (shapes, bound) = bound_at(&fixture, &[1, 37], true);

    let patches = refit_cached_attention_rows(&bound, &fixture.program, &shapes, &shapes);

    assert_eq!(patches, Some(Vec::new()));
}

#[test]
fn refit_declines_when_the_new_position_count_moves() {
    let fixture = single_range_fixture(2);
    let (decode_shapes, decode_bound) = bound_at(&fixture, &[1, 33], true);
    let (verify_shapes, _) = bound_at(&fixture, &[4, 33], true);

    let patches = refit_cached_attention_rows(
        &decode_bound,
        &fixture.program,
        &decode_shapes,
        &verify_shapes,
    );

    assert_eq!(
        patches, None,
        "a different new_count changes shapes far beyond the attention row count"
    );
}

#[test]
fn refit_declines_when_an_unfused_op_reads_the_moving_symbol() {
    let fixture = single_range_fixture(2);
    let (low_shapes, low_bound) = bound_at(&fixture, &[1, 33], false);
    let (high_shapes, _) = bound_at(&fixture, &[1, 65], false);

    let patches =
        refit_cached_attention_rows(&low_bound, &fixture.program, &low_shapes, &high_shapes);

    assert_eq!(
        patches, None,
        "without fusion the attention chain is plain reduces and elementwise ops, which this refit never recomposes"
    );
}

fn partial_rotary_bound_at(cached_rows: u64) -> (Shapes, Vec<BoundOp>, Vec<Op>, Vec<NodeId>) {
    let (program, residual1, taps, _, _) = super::tests::qwen35_partial_rotary_attention_fixture();
    let outputs = alloc::vec![
        residual1,
        taps.attended,
        taps.rotated_k_new_first,
        taps.rotated_k_new_second,
        taps.k_pass,
        taps.v_new,
    ];
    let shapes = crate::shape::infer(&program, &[1, cached_rows]).expect("fixture infers");
    let bound = bind_with_fusion(
        &program,
        &shapes,
        &outputs,
        true,
        NumericPolicy::llama_relaxed(),
    )
    .expect("fixture binds");
    (shapes, prune_dead(bound, &outputs), program, outputs)
}

#[proxima::test]
#[case::grow_by_one(40, 41)]
#[case::cross_bucket(40, 64)]
#[case::shrink(64, 9)]
#[case::from_one_row(1, 40)]
async fn refit_matches_a_fresh_bind_for_a_partial_rotary_attention_op(
    #[case] low: u64,
    #[case] high: u64,
) {
    let (low_shapes, low_bound, program, _) = partial_rotary_bound_at(low);
    let (high_shapes, high_bound, _, _) = partial_rotary_bound_at(high);
    assert!(
        low_bound.iter().any(|bound| matches!(
            &bound.kind,
            BoundOpKind::CachedAttention { rotary_dim, head_dim, .. } if rotary_dim != head_dim
        )),
        "the fixture must fuse a partial-rotary cached-attention op"
    );

    let patches = refit_cached_attention_rows(&low_bound, &program, &low_shapes, &high_shapes)
        .expect("a pass-plane row move is local to the fused attention op");

    let mut refit = low_bound.clone();
    for (position, patched) in patches {
        refit[position] = patched;
    }
    assert_eq!(refit, high_bound, "refit ops differ from a fresh bind");
}

#[proxima::test]
async fn refit_declines_a_partial_rotary_op_whose_pass_plane_rows_disagree() {
    let (low_shapes, low_bound, program, _) = partial_rotary_bound_at(40);
    let (high_shapes, _, _, _) = partial_rotary_bound_at(64);
    let mut broken = low_bound.clone();
    for bound in &mut broken {
        if let BoundOpKind::CachedAttention { operands, .. } = &mut bound.kind {
            let last = operands.len() - 1;
            operands[last].0 = operands[last - 1].0;
        }
    }

    let patches = refit_cached_attention_rows(&broken, &program, &low_shapes, &high_shapes);

    assert_eq!(
        patches, None,
        "a new-key pass plane that reads the cached rows cannot be refit"
    );
}
