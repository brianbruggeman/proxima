use super::*;

type OperandKey = (NodeId, i64, Vec<i64>);

type SiblingKey = (Vec<u64>, Vec<OperandKey>);

fn operand_key(operand: &(NodeId, Layout, Option<Lookup>)) -> OperandKey {
    (operand.0, operand.1.base, operand.1.strides.to_vec())
}

/// The grouping key under which two [`BoundOpKind::Elementwise`] ops are siblings: the same
/// iteration space and the same multiset of (source, layout) reads; element type is checked at
/// merge time. `None` for an op
/// that cannot be a sibling: a gathered read carries a runtime index, and an index node is read by
/// value through a [`Lookup`] elsewhere, so it keeps its own buffer.
fn sibling_key(bound: &BoundOp, index_nodes: &BTreeSet<NodeId>) -> Option<SiblingKey> {
    let BoundOpKind::Elementwise { operands, .. } = &bound.kind else {
        return None;
    };
    if operands.len() < 2
        || index_nodes.contains(&bound.node)
        || operands.iter().any(|(_, _, lookup)| lookup.is_some())
    {
        return None;
    }
    let mut keys: Vec<OperandKey> = operands.iter().map(operand_key).collect();
    keys.sort();
    Some((bound.extents.clone(), keys))
}

/// Re-indexes `body` from `from`'s operand order into `onto`'s: operand `j` of `from` becomes the
/// first still-unclaimed entry of `onto` with the same source and layout. The multisets are equal
/// by the sibling key, so every operand finds a home.
fn remap_body_onto(
    body: &ComposedBody,
    from: &BoundOperands,
    onto: &BoundOperands,
) -> Option<ComposedBody> {
    let mut claimed = alloc::vec![false; onto.len()];
    let mut index_of = Vec::with_capacity(from.len());
    for operand in from {
        let key = operand_key(operand);
        let slot = onto
            .iter()
            .enumerate()
            .position(|(index, candidate)| !claimed[index] && operand_key(candidate) == key)?;
        claimed[slot] = true;
        index_of.push(slot as u16);
    }
    let steps = body
        .steps
        .iter()
        .map(|step| BodyStep {
            op: step.op,
            args: step
                .args
                .iter()
                .map(|arg| match arg {
                    StepArg::Operand(index) => StepArg::Operand(index_of[*index as usize]),
                    StepArg::Step(index) => StepArg::Step(*index),
                })
                .collect(),
        })
        .collect();
    Some(ComposedBody { steps })
}

fn merge_siblings(first: &BoundOp, second: &BoundOp) -> Option<BoundOp> {
    let (
        BoundOpKind::Elementwise { body, operands },
        BoundOpKind::Elementwise {
            body: second_body,
            operands: second_operands,
        },
    ) = (&first.kind, &second.kind)
    else {
        return None;
    };
    if first.dtype != second.dtype {
        return None;
    }
    let twin_body = remap_body_onto(second_body, second_operands, operands)?;
    Some(BoundOp {
        node: first.node,
        dtype: first.dtype,
        extents: first.extents.clone(),
        kind: BoundOpKind::ElementwiseTwin {
            body: body.clone(),
            operands: operands.clone(),
            twin_node: second.node,
            twin_body,
        },
    })
}

/// Collapses every pair of sibling [`BoundOpKind::Elementwise`] ops into one
/// [`BoundOpKind::ElementwiseTwin`]: two ops that walk the same iteration space and read the same
/// (source, layout) operands, differing only in body, become one dispatch with two outputs. The
/// merged op takes the earlier op's position, so the earlier op's consumers still follow it and
/// the later op's operands (identical to the earlier's) are already produced.
///
/// RoPE is the instance this exists for: `fused_rope_pair` emits `x_same * cos - x_partner * sin`
/// and `x_partner * cos + x_same * sin`, two ops over the same four reads, for split-half and
/// adjacent pairing alike (only the layouts differ, and the key compares layouts). Nothing here
/// names RoPE; any two sibling bodies merge. Bit-exact by construction: each body keeps its own
/// steps and reads the same operand values it read before.
///
/// This is bind-time fusion for a backend that renders the twin kind (`omega`'s Metal driver);
/// [`bind_with_fusion`] never runs it, so the CPU, wgpu and cuda paths keep two plain ops.
/// [`BoundOp::twin_halves`] is the inverse. The pass has the shape of `apply_identity_copy_alias`,
/// a `Vec<BoundOp> -> Vec<BoundOp>` rewrite, and runs after `prune_dead` so both siblings are live.
#[must_use]
pub fn fuse_twin_elementwise(built: Vec<BoundOp>, program: &[Op]) -> Vec<BoundOp> {
    let index_nodes = index_node_ids(program);
    let mut unpaired: BTreeMap<SiblingKey, usize> = BTreeMap::new();
    let mut merged: BTreeMap<usize, BoundOp> = BTreeMap::new();
    let mut absorbed: BTreeSet<usize> = BTreeSet::new();
    for (position, bound) in built.iter().enumerate() {
        let Some(key) = sibling_key(bound, &index_nodes) else {
            continue;
        };
        let Some(first) = unpaired.remove(&key) else {
            unpaired.insert(key, position);
            continue;
        };
        match merge_siblings(&built[first], bound) {
            Some(twin) => {
                merged.insert(first, twin);
                absorbed.insert(position);
            }
            None => {
                unpaired.insert(key, position);
            }
        }
    }
    if merged.is_empty() {
        return built;
    }
    built
        .into_iter()
        .enumerate()
        .filter(|(position, _)| !absorbed.contains(position))
        .map(|(position, bound)| merged.remove(&position).unwrap_or(bound))
        .collect()
}

impl BoundOp {
    /// The second output node of an [`BoundOpKind::ElementwiseTwin`], `None` for every other kind.
    #[must_use]
    pub fn twin_node(&self) -> Option<NodeId> {
        match &self.kind {
            BoundOpKind::ElementwiseTwin { twin_node, .. } => Some(*twin_node),
            _ => None,
        }
    }

    /// Splits an [`BoundOpKind::ElementwiseTwin`] back into the two plain
    /// [`BoundOpKind::Elementwise`] ops it fused, primary first. This is the two-dispatch form the
    /// fused kernel must reproduce, and the reference a backend without a twin renderer runs.
    #[must_use]
    pub fn twin_halves(&self) -> Option<(BoundOp, BoundOp)> {
        let BoundOpKind::ElementwiseTwin {
            body,
            operands,
            twin_node,
            twin_body,
        } = &self.kind
        else {
            return None;
        };
        let half = |node: NodeId, body: &ComposedBody| BoundOp {
            node,
            dtype: self.dtype,
            extents: self.extents.clone(),
            kind: BoundOpKind::Elementwise {
                body: body.clone(),
                operands: operands.clone(),
            },
        };
        Some((half(self.node, body), half(*twin_node, twin_body)))
    }
}
