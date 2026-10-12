//! The arithmetic a recurrent program adds around the layer loop: the
//! LoopCD contrast between a reference and a terminal state, the adaptive
//! per-row strength, and the pass that makes every logical visit of one
//! physical layer read the same weight leaf.
//!
//! All of it is built from [`elementwise`] and [`reduce`] nodes the algebra
//! already has, so no `Op` or `ScalarOp` variant exists for recurrence. The
//! layer loop that calls these is [`scheduled_two_range_cached_forward_program_with_layer_roots`].

use alloc::collections::BTreeMap;

use super::*;
use crate::partition::remap;

/// `terminal + strength * (terminal - reference)` over axes `axes`, with a
/// scalar `strength`.
pub(super) fn append_loopcd_contrast(
    program: &mut Vec<Op>,
    reference: NodeId,
    terminal: NodeId,
    strength: f32,
    axes: &str,
) -> Result<NodeId, TensorError> {
    let strength = scalar_constant(program, strength);
    append_loopcd_contrast_node(
        program,
        reference,
        terminal,
        strength,
        axes,
        &alloc::format!("->{axes}"),
    )
}

/// [`append_loopcd_contrast`] with `strength` already a node, read through
/// `strength_pattern` (`"s->sv"` for a per-row strength over `sv` logits).
pub(super) fn append_loopcd_contrast_node(
    program: &mut Vec<Op>,
    reference: NodeId,
    terminal: NodeId,
    strength: NodeId,
    axes: &str,
    strength_pattern: &str,
) -> Result<NodeId, TensorError> {
    let pattern = alloc::format!("{axes}->{axes}");
    let delta = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(terminal, &pattern), (reference, &pattern)],
    )?;
    let scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(delta, &pattern), (strength, strength_pattern)],
    )?;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(terminal, &pattern), (scaled, &pattern)],
    )
}

/// `max_strength * (1 - (p_top1 - p_top2))` per row of `terminal` logits, from
/// a max-stabilized softmax. The second probability is the largest value
/// strictly below the first, except when the first is attained more than once:
/// then the margin is zero. A plain `p < max` filter would drop a tied maximum
/// and report a positive margin for a flat distribution.
pub(super) fn append_adaptive_loopcd_strength(
    program: &mut Vec<Op>,
    terminal: NodeId,
    max_strength: f32,
) -> Result<NodeId, TensorError> {
    let maximum = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        terminal,
        "sv->sv",
        "s->sv",
    )?;
    let shifted = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(terminal, "sv->sv"), (maximum, "s->sv")],
    )?;
    let exponentials = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted, "sv->sv")],
    )?;
    let denominator = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        exponentials,
        "sv->sv",
        "s->sv",
    )?;
    let probabilities = elementwise(
        program,
        DType::Float32,
        ScalarOp::Divide,
        &[(exponentials, "sv->sv"), (denominator, "s->sv")],
    )?;
    let first = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        probabilities,
        "sv->sv",
        "s->sv",
    )?;
    let is_first = elementwise(
        program,
        DType::Float32,
        ScalarOp::Equal,
        &[(probabilities, "sv->sv"), (first, "s->sv")],
    )?;
    let first_count = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        is_first,
        "sv->sv",
        "s->sv",
    )?;
    let zero = scalar_constant(program, 0.0);
    let below_first = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_first, "sv->sv"),
            (zero, "->sv"),
            (probabilities, "sv->sv"),
        ],
    )?;
    let distinct_second = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::Zero,
        below_first,
        "sv->sv",
        "s->sv",
    )?;
    let one = scalar_constant(program, 1.0);
    let tied = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(first_count, "s->s"), (one, "->s")],
    )?;
    let second = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[(tied, "s->s"), (first, "s->s"), (distinct_second, "s->s")],
    )?;
    let margin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(first, "s->s"), (second, "s->s")],
    )?;
    let uncertainty = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(one, "->s"), (margin, "s->s")],
    )?;
    let ceiling = scalar_constant(program, max_strength);
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(uncertainty, "s->s"), (ceiling, "->s")],
    )
}

/// Every `Op` of `program` with each later `blk.*`, `output_norm.weight` or
/// `lm_head_row` leaf that repeats an earlier leaf's name folded onto that first
/// leaf, plus the table taking each original node to its node in the compacted
/// program. A recurrent schedule declares each logical visit's weights under
/// the physical layer's name and the readout heads re-gather the last row, so
/// the repeats are the same bytes; one leaf each keeps a plan from uploading
/// them once per visit.
///
/// # Errors
///
/// [`TensorError::UnsupportedInBuilder`] when two leaves share a name but not a
/// dtype and shape, which would make the fold change the program.
pub(super) fn share_physical_weight_leaves(
    program: &[Op],
) -> Result<(Vec<Op>, BTreeMap<NodeId, NodeId>), TensorError> {
    let overflow = || TensorError::UnsupportedInBuilder {
        builder: "share_physical_weight_leaves",
        feature: "Op index overflow",
    };
    let mut translations = BTreeMap::new();
    let mut first_leaf: BTreeMap<&str, NodeId> = BTreeMap::new();
    let mut compact: Vec<Op> = Vec::with_capacity(program.len());
    for (position, operation) in program.iter().enumerate() {
        let original = NodeId(u32::try_from(position).map_err(|_| overflow())?);
        let replacement = NodeId(u32::try_from(compact.len()).map_err(|_| overflow())?);
        if let Op::Input {
            dtype,
            shape,
            name: Some(name),
        } = operation
            && (name.starts_with("blk.") || name == "output_norm.weight" || name == "lm_head_row")
        {
            if let Some(existing) = first_leaf.get(name.as_str()) {
                let agrees = matches!(
                    &compact[existing.0 as usize],
                    Op::Input { dtype: prior_dtype, shape: prior_shape, .. }
                        if prior_dtype == dtype && prior_shape == shape
                );
                if !agrees {
                    return Err(TensorError::UnsupportedInBuilder {
                        builder: "share_physical_weight_leaves",
                        feature: "a repeated weight leaf differs in dtype or shape",
                    });
                }
                translations.insert(original, *existing);
                continue;
            }
            first_leaf.insert(name.as_str(), replacement);
        }
        compact.push(remap(operation, &translations, original)?);
        translations.insert(original, replacement);
    }
    Ok((compact, translations))
}

/// `node` through `table`, as [`share_physical_weight_leaves`] returned it.
pub(super) fn translated(
    table: &BTreeMap<NodeId, NodeId>,
    node: NodeId,
) -> Result<NodeId, TensorError> {
    table
        .get(&node)
        .copied()
        .ok_or(TensorError::NodeOutOfRange(node, node))
}

/// [`share_physical_weight_leaves`] over a finished recurrent forward program,
/// with every root it names carried through the same node table.
///
/// # Errors
///
/// A weight leaf shared by name disagrees in dtype or shape, or the program
/// holds a cache root kind a recurrent schedule never produces.
pub(super) fn share_recurrent_weights(
    forward: TwoRangeLayerRootForwardProgram,
) -> Result<TwoRangeLayerRootForwardProgram, TensorError> {
    let (program, logits, cache_roots, moe_sites, duplicate_head_roots, readouts) = forward;
    let (program, table) = share_physical_weight_leaves(&program)?;
    let map = |node: NodeId| translated(&table, node);
    let map_all = |nodes: Vec<NodeId>| nodes.into_iter().map(map).collect::<Result<Vec<_>, _>>();
    let cache_roots = cache_roots
        .into_iter()
        .map(|roots| match roots {
            LayerCacheRoots::Attention((even, odd, value)) => Ok(LayerCacheRoots::Attention((
                map(even)?,
                map(odd)?,
                map(value)?,
            ))),
            shared @ LayerCacheRoots::SharedFromLayer(_) => Ok(shared),
            _ => Err(TensorError::UnsupportedInBuilder {
                builder: "share_recurrent_weights",
                feature: "a cache root kind a recurrent schedule never produces",
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let moe_sites = moe_sites
        .0
        .into_iter()
        .map(|site| {
            Ok(MoeSite {
                layer: site.layer,
                selected: map_all(site.selected)?,
                weights: map_all(site.weights)?,
            })
        })
        .collect::<Result<Vec<_>, TensorError>>()?;
    let optional = |node: Option<NodeId>| node.map(map).transpose();
    let readouts = readouts
        .map(|roots| {
            Ok::<_, TensorError>(RecurrentReadoutRoots {
                reference_raw: optional(roots.reference_raw)?,
                terminal_raw: optional(roots.terminal_raw)?,
                reference_normalized: optional(roots.reference_normalized)?,
                terminal_normalized: optional(roots.terminal_normalized)?,
                reference_logits: optional(roots.reference_logits)?,
                terminal_logits: optional(roots.terminal_logits)?,
                guided_raw: optional(roots.guided_raw)?,
                adaptive_strength: optional(roots.adaptive_strength)?,
            })
        })
        .transpose()?;
    Ok((
        program,
        map(logits)?,
        cache_roots,
        MoeSites(moe_sites),
        map_all(duplicate_head_roots)?,
        readouts,
    ))
}
