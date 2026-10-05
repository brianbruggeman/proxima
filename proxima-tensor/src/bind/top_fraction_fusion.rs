use super::*;

const EXPRESSION_LEN: usize = 8;

fn rebuilt_expression(
    program: &[Op],
    root: usize,
    keep_rows: Option<NodeId>,
) -> Option<(usize, NodeId, NodeId)> {
    let selected = root.checked_sub(usize::from(keep_rows.is_some()))?;
    let first = selected.checked_sub(EXPRESSION_LEN - 1)?;
    let keep_count = *program.get(selected)?.dependencies().first()?;
    let scores = *program.get(first + 1)?.dependencies().first()?;
    let mut reference = alloc::vec![
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 0.0,
        };
        first
    ];
    crate::spec::top_fraction_mask(&mut reference, scores, keep_count, keep_rows).ok()?;
    (reference.get(first..)? == program.get(first..=root)?).then_some((first, scores, keep_count))
}

fn expression_leaks(program: &[Op], inside: &[usize], absorbed: &BTreeSet<NodeId>) -> bool {
    program.iter().enumerate().any(|(position, operation)| {
        !inside.contains(&position)
            && operation
                .dependencies()
                .iter()
                .any(|dependency| absorbed.contains(dependency))
    })
}

fn operand_layout(resolved: &[BoundOp], wanted: NodeId) -> Option<Layout> {
    resolved
        .iter()
        .flat_map(BoundOp::all_read_sources)
        .find(|(node, _, _)| *node == wanted)
        .filter(|(_, layout, lookup)| {
            lookup.is_none() && layout.strides.iter().all(|stride| *stride >= 0)
        })
        .map(|(_, layout, _)| layout.clone())
}

/// Every `spec::top_fraction_mask` expression in `program` whose score row count is at least
/// `min_rows`, as the `BoundOpKind::TopFractionSelect` that could replace it and the node ids
/// it would absorb. Skipped: an expression with an absorbed node that is a requested output or
/// is read outside it, and one whose operands `resolved` does not read as plain layouts, so
/// `resolved` must be bound with every operand requested as an output.
pub fn top_fraction_candidates(
    program: &[Op],
    shapes: &Shapes,
    resolved: &[BoundOp],
    effective_outputs: &[NodeId],
    min_rows: u64,
) -> Vec<(BoundOp, BTreeSet<NodeId>)> {
    let mut candidates = Vec::new();
    for (root, operation) in program.iter().enumerate() {
        let union_keep = match operation {
            Op::Elementwise {
                body: ScalarOp::Maximum,
                operands,
                ..
            } => operands.get(1).map(|(node, _)| *node),
            _ => None,
        };
        let Some((first, scores, keep_count)) = rebuilt_expression(program, root, union_keep)
        else {
            continue;
        };
        let rows = shapes.of(scores).first().copied().unwrap_or(0);
        let absorbed: BTreeSet<NodeId> = (first..root)
            .map(|position| NodeId(position as u32))
            .collect();
        let inside: Vec<usize> = (first..=root).collect();
        let sources = [Some(scores), Some(keep_count), union_keep];
        let operands = sources
            .into_iter()
            .flatten()
            .map(|node| operand_layout(resolved, node).map(|layout| (node, layout, None)))
            .collect::<Option<Vec<_>>>();
        let Some(operands) = operands.filter(|_| {
            shapes.of(scores).len() == 1
                && rows >= min_rows
                && !expression_leaks(program, &inside, &absorbed)
                && !absorbed.iter().any(|node| effective_outputs.contains(node))
        }) else {
            continue;
        };
        let kind = BoundOpKind::TopFractionSelect {
            operands,
            rows,
            has_keep_rows: union_keep.is_some(),
        };
        let fused = BoundOp {
            node: NodeId(root as u32),
            dtype: DType::Float32,
            extents: alloc::vec![rows],
            kind,
        };
        candidates.push((fused, absorbed));
    }
    let covered: BTreeSet<NodeId> = candidates
        .iter()
        .flat_map(|(_, absorbed)| absorbed.iter().copied())
        .collect();
    candidates.retain(|(fused, _)| !covered.contains(&fused.node));
    candidates
}

fn apply_top_fraction_fusion(
    built: Vec<BoundOp>,
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    min_rows: u64,
) -> Result<Vec<BoundOp>, TensorError> {
    let initial = top_fraction_candidates(program, shapes, &built, outputs, min_rows);
    if initial.is_empty() {
        return Ok(built);
    }
    let mut planning_outputs = outputs.to_vec();
    if planning_outputs.is_empty() {
        let root = program
            .len()
            .checked_sub(1)
            .map(|position| NodeId(position as u32))
            .ok_or(TensorError::Empty)?;
        planning_outputs.push(root);
    }
    for (fused, _) in &initial {
        for (source, _, _) in fused.operands() {
            if !planning_outputs.contains(source) {
                planning_outputs.push(*source);
            }
        }
    }
    let rebuilt = bind_plain(program, shapes, &planning_outputs, NumericPolicy::bit_exact())?;
    let candidates = top_fraction_candidates(program, shapes, &rebuilt, outputs, min_rows);
    if candidates.is_empty() {
        return Ok(built);
    }
    let fused_by_node = candidates
        .iter()
        .map(|(fused, _)| (fused.node, fused))
        .collect::<BTreeMap<_, _>>();
    let absorbed = candidates
        .iter()
        .flat_map(|(_, absorbed)| absorbed.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut rewritten = Vec::with_capacity(rebuilt.len());
    for bound in rebuilt {
        if let Some(fused) = fused_by_node.get(&bound.node) {
            rewritten.push((*fused).clone());
        } else if !absorbed.contains(&bound.node) {
            rewritten.push(bound);
        }
    }
    Ok(rewritten)
}

/// [`bind_with_fusion`], then collapses every `spec::top_fraction_mask` expression whose score
/// row count is at least `min_rows` into one `BoundOpKind::TopFractionSelect`. The threshold is
/// an argument because it lives in `omega`'s sizing file, which this crate cannot see. Expressions
/// below the threshold, and expressions whose intermediate nodes are requested outputs or are read
/// outside the expression, are left exactly as `bind_with_fusion` binds them.
pub fn bind_with_top_fraction(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
    fuse_cached_attention: bool,
    numeric_policy: NumericPolicy,
    min_rows: u64,
) -> Result<Vec<BoundOp>, TensorError> {
    let built = bind_with_fusion(
        program,
        shapes,
        outputs,
        fuse_cached_attention,
        numeric_policy,
    )?;
    apply_top_fraction_fusion(built, program, shapes, outputs, min_rows)
}
