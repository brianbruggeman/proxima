use super::*;

use crate::op::{Extent, ReduceInit, append};
use crate::spec::reduce;

fn input(program: &mut Vec<Op>, name: &str, shape: &[u32]) -> NodeId {
    append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: shape.iter().map(|extent| Extent::Static(*extent)).collect(),
            name: Some(String::from(name)),
        },
    )
}

/// `kept` and `orphaned` are each a two-fold row-sum chain over their own
/// input; the second fold of each chain is the program's output.
fn two_independent_chains() -> (Vec<Op>, [NodeId; 2], [NodeId; 2]) {
    let mut program = Vec::new();
    let kept_input = input(&mut program, "kept", &[3, 4]);
    let orphaned_input = input(&mut program, "orphaned", &[3, 4]);
    let kept_rows = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        kept_input,
        "rc->rc",
        "r->rc",
    )
    .expect("kept row fold builds");
    let kept_total = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        kept_rows,
        "r->r",
        "->r",
    )
    .expect("kept total fold builds");
    let orphaned_rows = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        orphaned_input,
        "rc->rc",
        "r->rc",
    )
    .expect("orphaned row fold builds");
    let orphaned_total = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        orphaned_rows,
        "r->r",
        "->r",
    )
    .expect("orphaned total fold builds");
    (
        program,
        [kept_rows, kept_total],
        [orphaned_rows, orphaned_total],
    )
}

fn bind_both_chains(program: &[Op], outputs: &[NodeId]) -> Vec<BoundOp> {
    let shapes = crate::shape::infer(program, &[]).expect("static program infers");
    bind_plain(program, &shapes, outputs, NumericPolicy::bit_exact()).expect("both chains bind")
}

#[test]
fn a_fold_read_only_by_an_unread_fold_is_dead_with_it() {
    let (program, [kept_rows, kept_total], [orphaned_rows, orphaned_total]) =
        two_independent_chains();
    let resolved = bind_both_chains(&program, &[kept_total, orphaned_total]);

    let dead = dead_resolved_nodes(&resolved, &[kept_total]);

    assert_eq!(
        dead,
        BTreeSet::from([orphaned_rows, orphaned_total]),
        "the orphaned total is unread, and its row fold is read only by that total"
    );
    assert!(!dead.contains(&kept_rows), "the kept chain stays live");
}

#[test]
fn prune_dead_removes_the_whole_orphaned_chain_in_one_call() {
    let (program, [kept_rows, kept_total], [orphaned_rows, orphaned_total]) =
        two_independent_chains();
    let resolved = bind_both_chains(&program, &[kept_total, orphaned_total]);

    let pruned = prune_dead(resolved, &[kept_total]);

    let survivors: Vec<NodeId> = pruned.iter().map(|bound| bound.node).collect();
    assert_eq!(survivors, vec![kept_rows, kept_total]);
    assert!(!survivors.contains(&orphaned_rows) && !survivors.contains(&orphaned_total));
    assert!(
        dead_resolved_nodes(&pruned, &[kept_total]).is_empty(),
        "pruning is a fixpoint"
    );
}

#[test]
fn a_requested_output_is_never_dead_even_when_unread() {
    let (program, [_, kept_total], [orphaned_rows, orphaned_total]) = two_independent_chains();
    let resolved = bind_both_chains(&program, &[kept_total, orphaned_total]);

    let dead = dead_resolved_nodes(&resolved, &[kept_total, orphaned_total]);

    assert!(
        dead.is_empty(),
        "both totals are requested, and each row fold is read: {dead:?}"
    );
    assert!(!dead.contains(&orphaned_rows));
}

#[test]
fn pruning_leaves_a_requested_output_bit_identical() {
    let (program, [_, kept_total], [_, orphaned_total]) = two_independent_chains();
    let resolved = bind_both_chains(&program, &[kept_total, orphaned_total]);
    let pruned = prune_dead(resolved.clone(), &[kept_total]);
    let kept_data: Vec<f32> = (0..12).map(|index| index as f32 * 0.37 - 1.1).collect();
    let orphaned_data: Vec<f32> = (0..12).map(|index| index as f32 * 1.9 + 4.2).collect();
    let inputs = vec![(NodeId(0), kept_data), (NodeId(1), orphaned_data)];

    let before = super::tests::run_resolved(program.len(), &resolved, inputs.clone());
    let after = super::tests::run_resolved(program.len(), &pruned, inputs);

    assert_eq!(before[kept_total.0 as usize], after[kept_total.0 as usize]);
    assert!(
        before[kept_total.0 as usize].is_some(),
        "the kept total computes"
    );
}

#[test]
#[cfg(feature = "cached-attention-streaming")]
fn a_fused_single_range_decode_plan_keeps_no_orphaned_mask_node() {
    let (program, logits, cache_roots, _) =
        crate::spec::mistral_single_range_cached_forward_program(
            32,
            16,
            24,
            4,
            2,
            4,
            2,
            false,
            crate::spec::DuplicateHeadPosition::None,
            false,
        )
        .expect("single-range fixture builds");
    let mut outputs = alloc::vec![logits];
    for (even, odd, value) in &cache_roots {
        outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let shapes = crate::shape::infer(&program, &[1, 5]).expect("single-range fixture infers");
    let fused = bind_with_fusion(
        &program,
        &shapes,
        &outputs,
        true,
        NumericPolicy::bit_exact(),
    )
    .expect("fused bind succeeds");
    let fused_len = fused.len();

    let pruned = prune_dead(fused, &outputs);

    assert!(
        pruned.len() < fused_len,
        "the fused mask machinery leaves nodes to drop"
    );
    assert!(
        dead_resolved_nodes(&pruned, &outputs).is_empty(),
        "nothing unread survives one prune"
    );
    assert!(
        pruned
            .iter()
            .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. })),
        "the fused attention op itself is live"
    );
}
