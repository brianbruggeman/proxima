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
/// - a partial-rotary op additionally needs its pass-plane operands (query,
///   cached key, new key) to agree with the same query, cached and new row
///   counts, and a non-empty cached range;
/// - anything else touched, or any condition failing, returns `None` and the caller binds from scratch.
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

/// The nodes whose resolved shape differs between `previous` and `next`.
#[must_use]
pub fn changed_nodes(program: &[Op], previous: &Shapes, next: &Shapes) -> Vec<NodeId> {
    (0..program.len())
        .map(|position| NodeId(position as u32))
        .filter(|node| previous.of(*node) != next.of(*node))
        .collect()
}

/// The positions in `resolved` a shape diff over `changed` touches; the result
/// depends only on `changed` and the bound ops' structure, so a caller that
/// sees the same `changed` set again can reuse it.
#[must_use]
pub fn touched_positions(
    resolved: &[BoundOp],
    program: &[Op],
    changed: &[NodeId],
) -> Vec<usize> {
    let changed: BTreeSet<NodeId> = changed.iter().copied().collect();
    resolved
        .iter()
        .enumerate()
        .filter(|(_, bound)| touches_changed(bound, program, &changed))
        .map(|(position, _)| position)
        .collect()
}

/// [`refit_cached_attention_rows`] restricted to `positions`: the replacement
/// ops for exactly those positions, or `None` when any of them is not a
/// refittable cached-attention op at `next`.
#[must_use]
pub fn refit_cached_attention_at(
    resolved: &[BoundOp],
    positions: &[usize],
    next: &Shapes,
) -> Option<Vec<(usize, BoundOp)>> {
    positions
        .iter()
        .map(|position| Some((*position, refit_one(resolved.get(*position)?, next)?)))
        .collect()
}

/// exposed so a caller can measure which ops a shape diff touches before refreshing them.
pub fn touches_changed(bound: &BoundOp, program: &[Op], changed: &BTreeSet<NodeId>) -> bool {
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
    refit_attention(bound, next, false)
}

/// the refresh form: the query row count and output extents may follow the new shapes.
pub(super) fn refresh_attention(bound: &BoundOp, next: &Shapes) -> Option<BoundOp> {
    refit_attention(bound, next, true)
}

fn refit_attention(bound: &BoundOp, next: &Shapes, query_may_move: bool) -> Option<BoundOp> {
    let BoundOpKind::CachedAttention {
        operands,
        query_rows,
        cached_key_rows,
        new_key_rows,
        head_dim,
        rotary_dim,
        attention_sink,
        ..
    } = &bound.kind
    else {
        return None;
    };
    let has_pass_plane = rotary_dim != head_dim;
    if has_pass_plane && *cached_key_rows == 0 {
        return None;
    }
    let rows = |index: usize| next.of(operands.get(index)?.0).first().copied();
    let next_query_rows = rows(0)?;
    let query_held = next.of(bound.node) == bound.extents.as_slice() && next_query_rows == *query_rows;
    if !query_held && !query_may_move {
        return None;
    }
    let (refit_cached, refit_new) = if *cached_key_rows == 0 {
        let merged = rows(2)?;
        (merged == rows(6)? && merged >= next_query_rows).then_some((0, merged))?
    } else {
        let (cached, new) = (rows(2)?, rows(4)?);
        let consistent = rows(6)? == cached && rows(7)? == new && new_rows_follow(new, next_query_rows, *query_rows, *new_key_rows);
        let in_range = new.checked_sub(1)?.checked_add(cached)? < F32_EXACT_INTEGER_BOUND;
        (consistent && cached > 0 && in_range).then_some((cached, new))?
    };
    // the sink, when bound, is the one operand after the pass triple
    let before_sink = &operands[..operands.len() - usize::from(*attention_sink && !operands.is_empty())];
    if has_pass_plane && !pass_planes_agree(before_sink, next_query_rows, refit_cached, refit_new, next) {
        return None;
    }
    let mut patched = bound.clone();
    if let BoundOpKind::CachedAttention {
        query_rows,
        cached_key_rows,
        new_key_rows,
        ..
    } = &mut patched.kind
    {
        *query_rows = next_query_rows;
        *cached_key_rows = refit_cached;
        *new_key_rows = refit_new;
    }
    patched.extents = next.of(bound.node).to_vec();
    Some(patched)
}

fn new_rows_follow(new: u64, next_query_rows: u64, query_rows: u64, new_key_rows: u64) -> bool {
    if new_key_rows == query_rows {
        new == next_query_rows
    } else {
        new == new_key_rows
    }
}

/// The partial-rotary pass triple is always the last three operands
/// (`cached_attention_candidates` re-appends it after the optional
/// `cached_len`): query pass, cached key pass, new key pass.
fn pass_planes_agree(
    operands: &[(NodeId, Layout, Option<Lookup>)],
    query_rows: u64,
    cached_rows: u64,
    new_rows: u64,
    next: &Shapes,
) -> bool {
    let Some(start) = operands.len().checked_sub(3) else {
        return false;
    };
    let rows = |index: usize| next.of(operands[index].0).first().copied();
    rows(start) == Some(query_rows)
        && rows(start + 1) == Some(cached_rows)
        && rows(start + 2) == Some(new_rows)
}
