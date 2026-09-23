use super::*;

/// `true` when `layout` is the canonical, contiguous, base-0 read an
/// executor's fresh output buffer always has for `extents` -- the shape a
/// pure-copy identity's own single-operand read must carry for
/// [`apply_identity_copy_alias`] to fold it: the identity's own output
/// buffer is then a byte-for-byte replica of the source buffer (same
/// extents, same iteration order), so any reader's own layout/lookup keeps
/// working unchanged once redirected at the source.
fn is_canonical_layout(layout: &Layout, extents: &[u64]) -> bool {
    layout.base == 0 && layout.strides.as_slice() == row_major_strides(extents).as_slice()
}

/// `true` when `bound` is a pure-copy identity: one [`BoundOpKind::Elementwise`]
/// operand, a single-step [`ScalarOp::Identity`] body reading that operand
/// unchanged, no [`Lookup`] on the read, and the read is canonical
/// ([`is_canonical_layout`]) — the exact shape `omega_elementwise_r4_n1_identity`/
/// `omega_elementwise_r2_n1_identity` (the decode census's 105 per-token
/// dispatches) both have.
fn identity_copy_source(bound: &BoundOp) -> Option<(NodeId, i64)> {
    let BoundOpKind::Elementwise { body, operands } = &bound.kind else {
        return None;
    };
    let [(source, source_layout, source_lookup)] = operands.as_slice() else {
        return None;
    };
    let is_identity_body = matches!(
        body.steps.as_slice(),
        [BodyStep { op: ScalarOp::Identity, args }] if args.as_slice() == [StepArg::Operand(0)]
    );
    if !is_identity_body || source_lookup.is_some() || !is_canonical_layout(source_layout, &bound.extents) {
        return None;
    }
    Some((*source, source_layout.base))
}

fn rewrite_read_sources(operands: &mut [(NodeId, Layout, Option<Lookup>)], alias_of: &BTreeMap<NodeId, (NodeId, i64)>) {
    for entry in operands.iter_mut() {
        if let Some((source, source_base)) = alias_of.get(&entry.0) {
            entry.0 = *source;
            entry.1.base += *source_base;
        }
    }
}

/// Folds away every pure-copy identity [`BoundOp`] ([`identity_copy_source`])
/// by redirecting every reader to the identity's own source instead — no
/// dispatch, no bytes moved, bit-exact by construction (a verbatim buffer
/// replica read the same way). Declines per-identity (leaves it materialized)
/// when its own output is a requested output (the caller needs that exact
/// buffer to exist) or when any reader addresses it through a [`Lookup`] (a
/// gather whose indices are not proven safe to redirect); chained identities
/// resolve to their ultimate non-identity source before any reader is
/// rewritten, so no reader is ever left pointing at a node this pass removed.
///
/// Runs inside [`bind_with_fusion`] only, mirroring [`apply_gated_delta_net_fusion`]'s
/// own placement — the general fusion pipeline every backend already shares,
/// not a new one.
pub(super) fn apply_identity_copy_alias(built: Vec<BoundOp>, outputs: &[NodeId]) -> Vec<BoundOp> {
    let mut immediate: BTreeMap<NodeId, (NodeId, i64)> = BTreeMap::new();
    for bound in &built {
        if outputs.contains(&bound.node) {
            continue;
        }
        if let Some(source) = identity_copy_source(bound) {
            immediate.insert(bound.node, source);
        }
    }
    if immediate.is_empty() {
        return built;
    }

    let mut lookup_blocked: BTreeSet<NodeId> = BTreeSet::new();
    for bound in &built {
        for (node, _, lookup) in bound.all_read_sources() {
            if lookup.is_some() && immediate.contains_key(node) {
                lookup_blocked.insert(*node);
            }
            // this pass only rewrites the `(NodeId, Layout, Option<Lookup>)`
            // triples themselves, never a `Lookup`'s own `indices` field --
            // an identity a gather elsewhere reads its INDEX values from
            // must keep existing as its own `BoundOp` or that gather's
            // `indices` reference dangles.
            if let Some(lookup) = lookup
                && immediate.contains_key(&lookup.indices)
            {
                lookup_blocked.insert(lookup.indices);
            }
        }
    }

    let resolve = |mut node: NodeId| -> Option<(NodeId, i64)> {
        if lookup_blocked.contains(&node) {
            return None;
        }
        let mut base_offset = 0i64;
        let mut hops = 0usize;
        while let Some((source, source_base)) = immediate.get(&node) {
            if lookup_blocked.contains(source) {
                return None;
            }
            base_offset += *source_base;
            node = *source;
            hops += 1;
            if hops > built.len() {
                return None;
            }
        }
        Some((node, base_offset))
    };

    let alias_of: BTreeMap<NodeId, (NodeId, i64)> = immediate
        .keys()
        .filter_map(|node| resolve(*node).map(|resolved| (*node, resolved)))
        .collect();
    if alias_of.is_empty() {
        return built;
    }

    let mut rewritten = Vec::with_capacity(built.len());
    for mut bound in built {
        if alias_of.contains_key(&bound.node) {
            continue;
        }
        match &mut bound.kind {
            BoundOpKind::CachedAttention { operands, .. }
            | BoundOpKind::CachedSoftmaxWeights { operands, .. }
            | BoundOpKind::GatedDeltaNet { operands, .. }
            | BoundOpKind::MoeTopK { operands, .. }
            | BoundOpKind::Elementwise { operands, .. } => {
                rewrite_read_sources(operands, &alias_of);
            }
            BoundOpKind::Reduce {
                operands,
                epilogue_operands,
                ..
            }
            | BoundOpKind::RoundBatchedReduce {
                operands,
                epilogue_operands,
                ..
            } => {
                rewrite_read_sources(operands, &alias_of);
                rewrite_read_sources(epilogue_operands, &alias_of);
            }
            BoundOpKind::Iota | BoundOpKind::Constant { .. } => {}
        }
        rewritten.push(bound);
    }
    rewritten
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_op(node: u32, source: u32) -> BoundOp {
        BoundOp {
            node: NodeId(node),
            dtype: DType::Float32,
            extents: alloc::vec![4],
            kind: BoundOpKind::Elementwise {
                body: ComposedBody::leaf(ScalarOp::Identity),
                operands: alloc::vec![(NodeId(source), Layout { base: 0, strides: smallvec::smallvec![1] }, None)],
            },
        }
    }

    fn consumer_op(node: u32, reads: u32, other: u32) -> BoundOp {
        BoundOp {
            node: NodeId(node),
            dtype: DType::Float32,
            extents: alloc::vec![4],
            kind: BoundOpKind::Elementwise {
                body: ComposedBody::leaf(ScalarOp::Add),
                operands: alloc::vec![
                    (NodeId(reads), Layout { base: 0, strides: smallvec::smallvec![1] }, None),
                    (NodeId(other), Layout { base: 0, strides: smallvec::smallvec![1] }, None),
                ],
            },
        }
    }

    #[test]
    fn pure_copy_identity_is_folded_and_readers_redirected_to_the_source() {
        let built = alloc::vec![identity_op(10, 1), consumer_op(20, 10, 99)];
        let outputs = [NodeId(20)];

        let rewritten = apply_identity_copy_alias(built, &outputs);

        assert_eq!(rewritten.len(), 1, "the identity dispatch is dropped entirely");
        assert_eq!(rewritten[0].node, NodeId(20));
        let BoundOpKind::Elementwise { operands, .. } = &rewritten[0].kind else {
            panic!("consumer stays elementwise");
        };
        assert_eq!(
            operands[0].0,
            NodeId(1),
            "the consumer now reads the identity's own source directly"
        );
        assert_eq!(
            operands[0].1,
            Layout { base: 0, strides: smallvec::smallvec![1] },
            "the identity's own canonical read composes to the consumer's unchanged layout"
        );
    }

    #[test]
    fn identity_requested_as_an_output_is_never_folded() {
        let built = alloc::vec![identity_op(10, 1), consumer_op(20, 10, 99)];
        let outputs = [NodeId(10), NodeId(20)];

        let rewritten = apply_identity_copy_alias(built, &outputs);

        assert_eq!(
            rewritten.len(),
            2,
            "a requested-output identity must still materialize its own buffer"
        );
        let BoundOpKind::Elementwise { operands, .. } = &rewritten[1].kind else {
            panic!("consumer stays elementwise");
        };
        assert_eq!(
            operands[0].0,
            NodeId(10),
            "the consumer keeps reading the identity node, not its source"
        );
    }
}
