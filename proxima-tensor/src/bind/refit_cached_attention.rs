use super::*;

/// The f32-exact integer bound the cached-attention matcher enforces on
/// `new_key_rows + cached_key_rows` (`cached_attention_candidates`'s
/// `precision_ok`): the kernel carries row indices as f32.
const F32_EXACT_INTEGER_BOUND: u64 = 1 << 24;

/// The replacement [`BoundOp`]s a symbol crossing needs, as `(position,
/// op)` pairs over `resolved`, or `None` when the crossing is not provably
/// local to fused cached-attention ops.
///
/// A fresh [`bind_with_fusion`] at `next` would produce `resolved` again with
/// every op unchanged except the cached-attention ops whose key range moved
/// with the symbol: their `cached_key_rows` / `new_key_rows` are read off the
/// shapes of their key operands, and nothing else about them is shape-
/// dependent. This function is that claim made checkable instead of assumed:
///
/// - an op is untouched only if neither its own output, any node it reads,
///   nor any node fused beneath it (its cone, down to the nodes it reads)
///   changed shape between `previous` and `next`;
/// - a touched op must be a cached-attention op whose accept conditions
///   still hold at `next` (the same conditions the matcher applies:
///   consistent key/value rows, a non-empty cached range for the two-range
///   form, the f32-exact row bound) and whose query rows and output extents
///   did not move;
/// - anything else touched, any condition failing, or a partial-rotary op
///   (whose pass-plane operands this function does not re-check), returns
///   `None` and the caller binds from scratch.
///
/// Applying the returned patches to `resolved` yields exactly the ops a
/// fresh bind at `next` returns, which `refit_matches_a_fresh_bind_*` in this
/// crate's tests assert position by position. Teaching pointer: this is the
/// `previous plan -> next plan` step of [`refresh_bound_ops`]'s own idea,
/// narrowed to the one fused kind whose only shape input is a row count; it
/// reads [`BoundOp`]s the matchers in `dead_code_cached_attention` built and
/// the [`Shapes`] [`crate::shape::infer`] returns, and decides nothing the
/// matchers did not already decide.
#[must_use]
pub fn refit_cached_attention_rows(
    resolved: &[BoundOp],
    program: &[Op],
    previous: &Shapes,
    next: &Shapes,
) -> Option<Vec<(usize, BoundOp)>> {
    let changed: BTreeSet<NodeId> = (0..program.len())
        .map(|position| NodeId(position as u32))
        .filter(|node| previous.of(*node) != next.of(*node))
        .collect();
    if changed.is_empty() {
        return Some(Vec::new());
    }
    let mut patches = Vec::new();
    for (position, bound) in resolved.iter().enumerate() {
        if touches_changed(bound, program, &changed) {
            patches.push((position, refit_one(bound, next)?));
        }
    }
    Some(patches)
}

fn touches_changed(bound: &BoundOp, program: &[Op], changed: &BTreeSet<NodeId>) -> bool {
    let reads = read_nodes(bound);
    changed.contains(&bound.node)
        || reads.iter().any(|node| changed.contains(node))
        || fused_cone(program, bound.node, &reads)
            .iter()
            .any(|node| changed.contains(node))
}

/// The program nodes fused beneath `root`: everything reachable backwards
/// from it that is not itself one of the nodes `root`'s bound op reads.
fn fused_cone(program: &[Op], root: NodeId, boundary: &BTreeSet<NodeId>) -> BTreeSet<NodeId> {
    let mut cone = BTreeSet::new();
    let mut pending = vec![root];
    while let Some(node) = pending.pop() {
        if boundary.contains(&node) || !cone.insert(node) {
            continue;
        }
        match program.get(node.0 as usize) {
            Some(Op::Elementwise { operands, .. }) => {
                for (operand, map) in operands {
                    pending.push(*operand);
                    push_indices_node_into(map, &mut pending);
                }
            }
            Some(Op::Reduce(reduce)) => {
                pending.push(reduce.operand);
                push_indices_node_into(&reduce.in_map, &mut pending);
                push_indices_node_into(&reduce.out_map, &mut pending);
            }
            Some(Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. }) | None => {}
        }
    }
    cone
}

fn push_indices_node_into(map: &IndexMap, pending: &mut Vec<NodeId>) {
    if let IndexMap::Computed { indices, .. } = map {
        pending.push(*indices);
    }
}

fn refit_one(bound: &BoundOp, next: &Shapes) -> Option<BoundOp> {
    let BoundOpKind::CachedAttention {
        operands,
        query_rows,
        cached_key_rows,
        new_key_rows,
        head_dim,
        rotary_dim,
        ..
    } = &bound.kind
    else {
        return None;
    };
    if rotary_dim != head_dim {
        return None;
    }
    let rows = |index: usize| next.of(operands.get(index)?.0).first().copied();
    if next.of(bound.node) != bound.extents.as_slice() || rows(0)? != *query_rows {
        return None;
    }
    let (refit_cached, refit_new) = if *cached_key_rows == 0 {
        let merged = rows(2)?;
        (merged == rows(6)? && merged >= *query_rows).then_some((0, merged))?
    } else {
        let (cached, new) = (rows(2)?, rows(4)?);
        let consistent = rows(6)? == cached && rows(7)? == new && new == *new_key_rows;
        let in_range = new.checked_sub(1)?.checked_add(cached)? < F32_EXACT_INTEGER_BOUND;
        (consistent && cached > 0 && in_range).then_some((cached, new))?
    };
    let mut patched = bound.clone();
    if let BoundOpKind::CachedAttention {
        cached_key_rows,
        new_key_rows,
        ..
    } = &mut patched.kind
    {
        *cached_key_rows = refit_cached;
        *new_key_rows = refit_new;
    }
    Some(patched)
}
