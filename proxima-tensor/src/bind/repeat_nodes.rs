use super::*;

/// [`apply_repeat_nodes`]'s own return shape: the extended program, one
/// [`RepeatNodeRefusal`] per declined target, and the `(original, copies)`
/// pairing table `PROXIMA_REPEAT_VERIFY`'s byte-compare harness reads back
/// (`arena_encode_dispatch_finish::finish`'s doc) — named here so the
/// function signature states a type instead of a three-tuple literal.
pub type RepeatNodesOutcome = (Vec<BoundOp>, Vec<RepeatNodeRefusal>, Vec<(NodeId, Vec<NodeId>)>);

/// One [`BoundOp`] the duplicate-dispatch attribution harness could not copy
/// byte-identically — the caller's typed refusal instead of a silent partial
/// duplication. `kind_name` is [`BoundOpKind::render_kind_name`]'s own label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepeatNodeRefusal {
    pub node: NodeId,
    pub kind_name: &'static str,
}

/// The highest [`NodeId`] anywhere in `built` — every primary `node` plus
/// every named extra output ([`BoundOpKind::GatedDeltaNet::state_out`],
/// [`BoundOpKind::MoeTopK::routes`]/`weights`/`weight_total`,
/// [`BoundOpKind::CachedSoftmaxWeights`]'s three named fields,
/// [`BoundOpKind::RoundBatchedReduce::round_outputs`]) — so a fresh id never
/// collides with an id this program already uses, whether or not this pass
/// duplicates that particular kind.
fn max_node_id(built: &[BoundOp]) -> u32 {
    let mut highest = 0u32;
    for bound in built {
        highest = highest.max(bound.node.0);
        match &bound.kind {
            BoundOpKind::GatedDeltaNet { state_out, .. } => highest = highest.max(state_out.0),
            BoundOpKind::CachedSoftmaxWeights {
                cached_weight_sum,
                new_weight_sum,
                new_attended,
                ..
            } => {
                highest = highest
                    .max(cached_weight_sum.0)
                    .max(new_weight_sum.0)
                    .max(new_attended.0);
            }
            BoundOpKind::MoeTopK {
                routes,
                weights,
                weight_total,
                ..
            } => {
                highest = highest.max(weight_total.0);
                for extra in routes.iter().chain(weights.iter()) {
                    highest = highest.max(extra.0);
                }
            }
            BoundOpKind::RoundBatchedReduce { round_outputs, .. } => {
                for extra in round_outputs {
                    highest = highest.max(extra.0);
                }
            }
            BoundOpKind::CachedAttention { .. }
            | BoundOpKind::Elementwise { .. }
            | BoundOpKind::Reduce { .. }
            | BoundOpKind::Iota
            | BoundOpKind::Constant { .. } => {}
        }
    }
    highest
}

/// Duplicates `original` as a plain, independent [`BoundOp`]: same `dtype`,
/// `extents`, and `kind` (operands unchanged — the copy reads the exact same
/// upstream sources the original does, never the original's own output), but
/// a fresh primary `node` from `fresh` and, for a kind carrying named extra
/// outputs, fresh ids for those too so the copy never aliases the original's
/// buffers. Returns `None` (a typed refusal, not a panic) for
/// [`BoundOpKind::CachedAttention`] and [`BoundOpKind::GatedDeltaNet`] —
/// [`BoundOpKind::GatedDeltaNet::state_out`] is cross-decode-step recurrent
/// state a duplicate dispatch must never alias or advance twice, and
/// [`BoundOpKind::CachedAttention`]'s own multi-range operand shape
/// (`BoundOpKind::CachedAttention`'s own doc) has no extra-output field this
/// pass could safely rename — both need a bespoke copy rule this generic
/// pass does not attempt.
fn duplicate_bound_op(original: &BoundOp, fresh: &mut u32) -> Option<BoundOp> {
    let mut next_id = || {
        *fresh += 1;
        NodeId(*fresh)
    };
    let kind = match &original.kind {
        BoundOpKind::Elementwise { body, operands } => BoundOpKind::Elementwise {
            body: body.clone(),
            operands: operands.clone(),
        },
        BoundOpKind::Reduce {
            element_body,
            reduce_op,
            init,
            keep,
            operands,
            output_axes,
            out_layout,
            out_scatter,
            epilogue_body,
            epilogue_operands,
            epilogue_broadcast_axes,
        } => BoundOpKind::Reduce {
            element_body: element_body.clone(),
            reduce_op: *reduce_op,
            init: *init,
            keep: *keep,
            operands: operands.clone(),
            output_axes: output_axes.clone(),
            out_layout: out_layout.clone(),
            out_scatter: out_scatter.clone(),
            epilogue_body: epilogue_body.clone(),
            epilogue_operands: epilogue_operands.clone(),
            epilogue_broadcast_axes: epilogue_broadcast_axes.clone(),
        },
        BoundOpKind::Iota => BoundOpKind::Iota,
        BoundOpKind::Constant { value } => BoundOpKind::Constant { value: *value },
        // `CachedSoftmaxWeights`'s three named extra outputs
        // (`cached_weight_sum`/`new_weight_sum`/`new_attended`) are
        // themselves bind-time-invented ids with no row in the `Shapes`
        // table `infer` built from the pre-bind spec program (only the
        // primary `node` traces back to a real spec `Op`) -- `Shapes::of`
        // on one of a duplicate's extra outputs would need a THIRD aliasing
        // source this pass does not have proof for within this slice, so
        // this kind declines rather than risk an unverified byte match.
        BoundOpKind::CachedAttention { .. }
        | BoundOpKind::CachedSoftmaxWeights { .. }
        | BoundOpKind::GatedDeltaNet { .. }
        | BoundOpKind::MoeTopK { .. }
        | BoundOpKind::RoundBatchedReduce { .. } => return None,
    };
    Some(BoundOp {
        node: next_id(),
        dtype: original.dtype,
        extents: original.extents.clone(),
        kind,
    })
}

/// The duplicate-dispatch attribution harness (`PROXIMA_REPEAT_NODES`/
/// `PROXIMA_REPEAT_COUNT`, `instrument`-gated): for each `NodeId` in
/// `targets`, appends `count` independent copies of that node's own
/// [`BoundOp`] immediately after the original in `built`'s program order —
/// each copy reads the SAME upstream operands the original does (never the
/// original's own output), so its result is a byte-identical, independently
/// dispatched recomputation, exactly the method
/// `spec::lfm2_single_range_cached`'s own `PROXIMA_HEAD_REPEATS` established
/// at the spec-graph level, generalized to any already-bound node instead of
/// only the LM head. Every copy's primary `node` (and, for
/// [`BoundOpKind::CachedSoftmaxWeights`], its three named extra outputs) is
/// appended to `outputs` so the caller's own [`prune_dead`] keeps it live —
/// this pass never mutates `built`'s existing nodes or their readers, so a
/// node not named in `targets` is byte-for-byte unchanged.
///
/// Returns the extended program plus one [`RepeatNodeRefusal`] per requested
/// target this pass declined (an unmatched `NodeId`, or a kind
/// [`duplicate_bound_op`] does not support) — the caller decides whether an
/// unmet target is fatal; this pass never panics or silently drops one.
#[must_use]
pub fn apply_repeat_nodes(
    mut built: Vec<BoundOp>,
    targets: &[NodeId],
    count: u32,
    outputs: &mut Vec<NodeId>,
    shapes: &mut Shapes,
) -> RepeatNodesOutcome {
    if targets.is_empty() || count == 0 {
        return (built, Vec::new(), Vec::new());
    }
    let mut fresh = max_node_id(&built);
    let mut refusals = Vec::new();
    // one entry per target that produced at least one copy -- the
    // `PROXIMA_REPEAT_VERIFY` byte-compare harness's own pairing table
    // (`arena_encode_dispatch_finish::finish`'s doc), since a copy's fresh
    // id is only known here, at the moment it is minted.
    let mut pairs: Vec<(NodeId, Vec<NodeId>)> = Vec::new();
    for target in targets {
        let Some(original_index) = built.iter().position(|bound| bound.node == *target) else {
            refusals.push(RepeatNodeRefusal {
                node: *target,
                kind_name: "unknown",
            });
            continue;
        };
        let mut copies_of_target = Vec::new();
        for step in 0..count {
            let insert_at = original_index + 1 + step as usize;
            let Some(copy) = duplicate_bound_op(&built[original_index], &mut fresh) else {
                refusals.push(RepeatNodeRefusal {
                    node: *target,
                    kind_name: built[original_index].kind.name(),
                });
                break;
            };
            outputs.push(copy.node);
            // the copy's resolved output shape is byte-identical to the
            // original's own -- see `Shapes::push_alias`'s own doc for why
            // this is exact rather than an approximation.
            shapes.push_alias(*target);
            copies_of_target.push(copy.node);
            built.insert(insert_at, copy);
        }
        if !copies_of_target.is_empty() {
            pairs.push((*target, copies_of_target));
        }
    }
    (built, refusals, pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elementwise_op(node: u32, source: u32) -> BoundOp {
        BoundOp {
            node: NodeId(node),
            dtype: DType::Float32,
            extents: alloc::vec![4],
            kind: BoundOpKind::Elementwise {
                body: ComposedBody::leaf(ScalarOp::Identity),
                operands: alloc::vec![(
                    NodeId(source),
                    Layout { base: 0, strides: smallvec::smallvec![1] },
                    None
                )],
            },
        }
    }

    #[test]
    fn duplicates_land_immediately_after_the_original_and_become_outputs() {
        let built = alloc::vec![elementwise_op(1, 0), elementwise_op(2, 1)];
        let mut outputs = alloc::vec![NodeId(2)];

        let mut shapes = Shapes::from_rows(alloc::vec![alloc::vec![4], alloc::vec![4], alloc::vec![4]]);

        let (rewritten, refusals, pairs) =
            apply_repeat_nodes(built, &[NodeId(1)], 2, &mut outputs, &mut shapes);

        assert!(refusals.is_empty(), "an elementwise target must never be refused");
        assert_eq!(rewritten.len(), 4, "two copies appended to the original two-op program");
        assert_eq!(rewritten[0].node, NodeId(1), "the original stays in place");
        assert_eq!(rewritten[1].node, NodeId(3), "first copy lands immediately after node 1");
        assert_eq!(rewritten[2].node, NodeId(4), "second copy lands immediately after the first copy");
        assert_eq!(rewritten[3].node, NodeId(2), "the untouched original consumer stays last");
        assert_eq!(
            outputs,
            alloc::vec![NodeId(2), NodeId(3), NodeId(4)],
            "both copies are appended so prune_dead keeps them"
        );
        assert_eq!(
            pairs,
            alloc::vec![(NodeId(1), alloc::vec![NodeId(3), NodeId(4)])],
            "the verify harness's own pairing table names both copies against their original"
        );
    }

    #[test]
    fn an_unknown_target_is_a_typed_refusal_not_a_panic() {
        let built = alloc::vec![elementwise_op(1, 0)];
        let mut outputs = alloc::vec![NodeId(1)];

        let mut shapes = Shapes::from_rows(alloc::vec![alloc::vec![4]]);
        let (rewritten, refusals, pairs) =
            apply_repeat_nodes(built, &[NodeId(99)], 1, &mut outputs, &mut shapes);

        assert_eq!(rewritten.len(), 1, "no copy is inserted for an unmatched node");
        assert_eq!(refusals, alloc::vec![RepeatNodeRefusal { node: NodeId(99), kind_name: "unknown" }]);
        assert!(pairs.is_empty(), "an unmatched target contributes no pairing entry");
    }
}
