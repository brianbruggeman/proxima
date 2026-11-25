use super::*;

/// What every backend needs to know about a `weight * activation` sum before
/// choosing a kernel: which output positions the weight owns, which the
/// activation owns, and how much work each side carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductContraction {
    pub feature_positions: Vec<usize>,
    /// Every output position the activation owns or the expert route moves along.
    pub token_positions: Vec<usize>,
    /// The subset of `token_positions` the gathered weight's expert route moves
    /// along, whether or not the activation strides them too: the routed token
    /// axes. Empty for a dense weight.
    pub routed_positions: Vec<usize>,
    /// Output positions both operands stride, or neither strides over an extent of one:
    /// the attention heads of `non_causal_attention_heads`. Always empty for a
    /// [`product_contraction`]; filled by [`dense_batched_contraction`].
    pub batch_positions: Vec<usize>,
    pub token_total: u64,
    pub reduction_extent: u64,
    pub gathered_operands: usize,
    /// Whether the weight operand is the one that reads through a gather.
    pub weight_gathered: bool,
}

/// Whether `body` is exactly one multiply of operands `first` and `second`, in either order.
#[must_use]
pub fn is_product_of(body: &ComposedBody, first: u16, second: u16) -> bool {
    let [only] = body.steps.as_slice() else {
        return false;
    };
    let (first, second) = (StepArg::Operand(first), StepArg::Operand(second));
    only.op == ScalarOp::Multiply && (only.args == [first, second] || only.args == [second, first])
}

/// How many of the op's operands read through a gather.
#[must_use]
pub fn gathered_operand_count(bound: &BoundOp) -> usize {
    bound.operands().iter().filter(|(_, _, gather)| gather.is_some()).count()
}

fn is_plain_product(body: &ComposedBody) -> bool {
    is_product_of(body, 0, 1)
}

struct AxisSplit {
    feature_positions: Vec<usize>,
    token_positions: Vec<usize>,
    routed_positions: Vec<usize>,
    batch_positions: Vec<usize>,
    token_total: u64,
    reduction_extent: u64,
}

fn route_stride(route: Option<&Layout>, axis: u16) -> i64 {
    route.map_or(0, |layout| layout.stride(axis))
}

/// A position the route moves along is a routed token position; an axis neither operand strides but the route does is a token position too.
/// A route that moves along any axis outside the token positions would name
/// several experts for one token, so that shape is refused.
fn split_axes(
    extents: &[u64],
    output_axes: &[u16],
    weight_layout: &Layout,
    activation_layout: &Layout,
    route: Option<&Layout>,
) -> Option<AxisSplit> {
    let mut feature_positions = Vec::new();
    let mut token_positions = Vec::new();
    let mut routed_positions = Vec::new();
    let mut batch_positions = Vec::new();
    for (position, &axis) in output_axes.iter().enumerate() {
        let routed = route_stride(route, axis) != 0;
        match (weight_layout.stride(axis) != 0, activation_layout.stride(axis) != 0, routed) {
            (true, false, false) => feature_positions.push(position),
            (false, true, _) => {
                token_positions.push(position);
                if routed {
                    routed_positions.push(position);
                }
            }
            (false, false, true) => {
                token_positions.push(position);
                routed_positions.push(position);
            }
            (true, true, false) => batch_positions.push(position),
            (false, false, false) if extents[axis as usize] == 1 => batch_positions.push(position),
            _ => return None,
        }
    }
    let token_axes: Vec<u16> = token_positions.iter().map(|&position| output_axes[position]).collect();
    let follows_tokens = (0..extents.len() as u16).all(|axis| token_axes.contains(&axis) || route_stride(route, axis) == 0);
    if !follows_tokens {
        return None;
    }
    let token_total = token_axes.iter().map(|&axis| extents[axis as usize]).product();
    let reduction_extent = (0..extents.len() as u16)
        .filter(|dim| !output_axes.contains(dim))
        .map(|dim| extents[dim as usize])
        .product();
    Some(AxisSplit {
        feature_positions,
        token_positions,
        routed_positions,
        batch_positions,
        token_total,
        reduction_extent,
    })
}

/// `Some` for a `Keep::Reduce` add-from-zero whose body is one multiply of
/// operands 0 and 1, with every output position owned by exactly one of the
/// two operands: the weight (`weight` is its operand index) makes it a
/// feature position, the other operand makes it a token position. A batch
/// position refuses it; [`dense_batched_contraction`] is the batch-aware reading.
#[must_use]
pub fn product_contraction(bound: &BoundOp, weight: usize) -> Option<ProductContraction> {
    product_facts(bound, weight).filter(|facts| facts.batch_positions.is_empty())
}

/// The batch-aware reading of a dense two-operand product: the weight operand
/// index, and the facts narrowed to one matrix. Roles come from stride
/// ownership, never from packed-ness: the weight owns the last exclusively
/// owned output position, whose position is the one feature position; the first
/// position the other operand exclusively owns is the one token position, with
/// `token_total` its extent; every other position (both-strided, broadcast over
/// an extent of one, or a further exclusive position) is a batch position, in
/// output-axis order. `None` when no position is exclusively owned by each
/// operand, or an axis neither operand strides has an extent above one.
///
/// This is the shape a backend's dense batched matmul takes, for example the
/// output of `non_causal_attention_heads`; a backend asks [`contraction_form`]
/// of the result rather than deriving ownership itself.
#[must_use]
pub fn dense_batched_contraction(bound: &BoundOp) -> Option<(usize, ProductContraction)> {
    let BoundOpKind::Reduce {
        output_axes,
        operands,
        ..
    } = &bound.kind
    else {
        return None;
    };
    let [first, second] = operands.as_slice() else {
        return None;
    };
    let weight = output_axes.iter().rev().find_map(|&axis| {
        match (first.1.stride(axis) != 0, second.1.stride(axis) != 0) {
            (true, false) => Some(0),
            (false, true) => Some(1),
            _ => None,
        }
    })?;
    let facts = product_facts(bound, weight)?;
    let feature = *facts.feature_positions.last()?;
    let token = *facts.token_positions.first()?;
    let mut batch_positions = facts.batch_positions;
    batch_positions.extend(
        facts
            .feature_positions
            .iter()
            .copied()
            .filter(|&position| position != feature),
    );
    batch_positions.extend(
        facts
            .token_positions
            .iter()
            .copied()
            .filter(|&position| position != token),
    );
    batch_positions.sort_unstable_by_key(|&position| output_axes[position]);
    let narrowed = ProductContraction {
        feature_positions: alloc::vec![feature],
        token_positions: alloc::vec![token],
        routed_positions: Vec::new(),
        batch_positions,
        token_total: bound.extents[output_axes[token] as usize],
        ..facts
    };
    Some((weight, narrowed))
}

fn product_facts(bound: &BoundOp, weight: usize) -> Option<ProductContraction> {
    let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        reduce_op: ScalarOp::Add,
        init: ReduceInit::Zero,
        output_axes,
        operands,
        ..
    } = &bound.kind
    else {
        return None;
    };
    let [first, second] = operands.as_slice() else {
        return None;
    };
    if !is_plain_product(bound.element_body()) || weight > 1 {
        return None;
    }
    let (weight_operand, activation_operand) = if weight == 0 { (first, second) } else { (second, first) };
    let route = weight_operand.2.as_ref().map(|lookup| &lookup.index_layout);
    let split = split_axes(&bound.extents, output_axes, &weight_operand.1, &activation_operand.1, route)?;
    Some(ProductContraction {
        feature_positions: split.feature_positions,
        token_positions: split.token_positions,
        routed_positions: split.routed_positions,
        batch_positions: split.batch_positions,
        token_total: split.token_total,
        reduction_extent: split.reduction_extent,
        gathered_operands: gathered_operand_count(bound),
        weight_gathered: weight_operand.2.is_some(),
    })
}

/// How a backend executes one product contraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractionForm {
    /// One tiled kernel over a dense weight shared by several token rows.
    Tiled,
    /// One tiled kernel per expert over the tokens routed to it.
    Grouped,
    /// Neither: each output row is reduced on its own.
    Rowwise,
}

/// The one choice between tiled, grouped and row-wise execution: a gathered
/// weight, the only gathered operand, whose route follows the token axes takes the grouped form, a dense weight the
/// tiled one, each only when `executes` grants the form and the token rows
/// reach its minimum.
#[must_use]
pub fn contraction_form(facts: &ProductContraction, executes: &Executes) -> ContractionForm {
    let Some(limits) = executes.contraction else {
        return ContractionForm::Rowwise;
    };
    if facts.gathered_operands > 0 {
        let grouped = facts.gathered_operands == 1
            && facts.weight_gathered
            && executes.forms.contains(Form::GroupedContraction)
            && facts.token_total >= limits.grouped_min_tokens;
        return if grouped { ContractionForm::Grouped } else { ContractionForm::Rowwise };
    }
    if executes.forms.contains(Form::TiledContraction) && facts.token_total >= limits.tiled_min_tokens {
        ContractionForm::Tiled
    } else {
        ContractionForm::Rowwise
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{Extent, append};
    use proxima_numeric::{DType, Numerics};

    fn layout(strides: &[i64]) -> Layout {
        Layout {
            base: 0,
            strides: strides.iter().copied().collect(),
        }
    }

    fn dense_split(extents: &[u64], weight_strides: &[i64], activation_strides: &[i64]) -> Option<(Vec<usize>, Vec<usize>, u64, u64)> {
        split_axes(extents, &[0, 1], &layout(weight_strides), &layout(activation_strides), None)
            .map(|split| (split.feature_positions, split.token_positions, split.token_total, split.reduction_extent))
    }

    #[test]
    fn a_route_moving_along_a_token_axis_reports_it_routed() {
        let route = layout(&[0, 1, 0]);
        let split = split_axes(&[8, 4, 32], &[0, 1], &layout(&[32, 0, 1]), &layout(&[0, 0, 1]), Some(&route)).expect("routed");
        assert_eq!(split.feature_positions, vec![0]);
        assert_eq!(split.token_positions, vec![1]);
        assert_eq!(split.routed_positions, vec![1]);
        assert_eq!(split.token_total, 4);
    }

    #[test]
    fn a_route_moving_along_a_feature_axis_is_refused() {
        let route = layout(&[1, 1, 0]);
        assert!(split_axes(&[8, 4, 32], &[0, 1], &layout(&[32, 0, 1]), &layout(&[0, 0, 1]), Some(&route)).is_none());
    }

    #[test]
    fn a_route_moving_along_the_reduction_axis_is_refused() {
        let route = layout(&[0, 1, 1]);
        assert!(split_axes(&[8, 4, 32], &[0, 1], &layout(&[32, 0, 1]), &layout(&[0, 0, 1]), Some(&route)).is_none());
    }

    #[test]
    fn a_dense_product_splits_feature_and_token_axes_and_totals_both_extents() {
        let (features, tokens, token_total, reduction) = dense_split(&[8, 4, 32], &[32, 0, 1], &[0, 32, 1]).expect("dense");
        assert_eq!(features, vec![0]);
        assert_eq!(tokens, vec![1]);
        assert_eq!(token_total, 4);
        assert_eq!(reduction, 32);
    }

    #[test]
    fn a_single_token_product_has_a_token_total_of_one() {
        let (_, tokens, token_total, _) = dense_split(&[8, 1, 32], &[32, 0, 1], &[0, 32, 1]).expect("single token");
        assert_eq!(tokens, vec![1]);
        assert_eq!(token_total, 1);
    }

    #[test]
    fn swapping_which_operand_is_the_weight_swaps_feature_and_token_positions() {
        let (features, tokens, token_total, _) = dense_split(&[8, 4, 32], &[0, 32, 1], &[32, 0, 1]).expect("swapped");
        assert_eq!(features, vec![1]);
        assert_eq!(tokens, vec![0]);
        assert_eq!(token_total, 8);
    }

    #[test]
    fn an_output_axis_both_operands_stride_is_a_batch_position() {
        let split = split_axes(
            &[8, 4, 32],
            &[0, 1],
            &layout(&[32, 32, 1]),
            &layout(&[0, 32, 1]),
            None,
        )
        .expect("batched");
        assert_eq!(split.feature_positions, vec![0]);
        assert!(split.token_positions.is_empty());
        assert_eq!(split.batch_positions, vec![1]);
    }

    #[test]
    fn an_axis_neither_operand_strides_is_a_batch_position_only_at_extent_one() {
        let weight = layout(&[32, 0, 1]);
        let activation = layout(&[0, 0, 1]);
        let broadcast = split_axes(&[8, 1, 32], &[0, 1], &weight, &activation, None)
            .expect("an extent of one broadcasts");
        assert_eq!(broadcast.batch_positions, vec![1]);
        assert!(split_axes(&[8, 4, 32], &[0, 1], &weight, &activation, None).is_none());
    }

    #[test]
    fn a_dense_product_has_no_batch_positions_and_a_batched_one_is_not_a_product_contraction() {
        let bound = bound_contraction(&contraction_program(4));
        assert!(
            product_contraction(&bound, 1)
                .expect("a dense product")
                .batch_positions
                .is_empty()
        );
        let batched = batched_contractions(&attention_program(3, 5, 2));
        assert!(!batched.is_empty());
        for (bound, weight, _) in &batched {
            assert_eq!(product_contraction(bound, *weight), None);
        }
    }

    #[test]
    fn a_multiply_of_either_operand_order_is_a_plain_product() {
        let swapped = ComposedBody {
            steps: vec![BodyStep {
                op: ScalarOp::Multiply,
                args: vec![StepArg::Operand(1), StepArg::Operand(0)],
            }],
        };
        assert!(is_plain_product(&ComposedBody::leaf(ScalarOp::Multiply)));
        assert!(is_plain_product(&swapped));
    }

    #[test]
    fn a_non_product_body_is_refused() {
        assert!(!is_plain_product(&ComposedBody::leaf(ScalarOp::Add)));
    }
    fn activation_dtype() -> DType {
        Numerics::default()
            .activation
            .to_dtype()
            .expect("the reference activation is a scalar format")
    }

    fn contraction_program(tokens: u32) -> Vec<Op> {
        let dtype = activation_dtype();
        let mut program = Vec::new();
        let activation = append(
            &mut program,
            Op::Input {
                dtype,
                shape: alloc::vec![Extent::Static(tokens), Extent::Static(16)],
                name: None,
            },
        );
        let weight = append(
            &mut program,
            Op::Input {
                dtype,
                shape: alloc::vec![Extent::Static(16), Extent::Static(8)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (weight, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        program
    }

    fn bound_contraction(program: &[Op]) -> BoundOp {
        let shapes = shape::infer(program, &[]).expect("the program infers");
        let output = NodeId((program.len() - 1) as u32);
        let mut built = bind(program, &shapes, &[output], NumericPolicy::bit_exact()).expect("the program binds");
        assert_eq!(built.len(), 1, "the product fuses into its reduce");
        built.remove(0)
    }

    #[test]
    fn a_bound_dense_product_reports_its_feature_and_token_split() {
        let bound = bound_contraction(&contraction_program(4));
        let contraction = product_contraction(&bound, 1).expect("a dense product");
        assert_eq!(contraction.feature_positions, vec![1]);
        assert_eq!(contraction.token_positions, vec![0]);
        assert_eq!(contraction.token_total, 4);
        assert_eq!(contraction.reduction_extent, 16);
        assert_eq!(contraction.gathered_operands, 0);
    }

    #[test]
    fn a_projection_reads_features_from_the_weight_and_tokens_from_the_activation_with_no_batch() {
        let bound = bound_contraction(&contraction_program(4));
        let (weight, facts) = dense_batched_contraction(&bound).expect("a projection");
        assert!(
            bound.operands()[weight].1.stride(1) != 0,
            "the weight strides the out axis"
        );
        assert_eq!(facts.feature_positions, vec![1]);
        assert_eq!(facts.token_positions, vec![0]);
        assert!(facts.batch_positions.is_empty());
        assert_eq!(facts.token_total, 4);
        assert_eq!(facts.reduction_extent, 16);
    }

    const ATTENTION_HEADS: u32 = 3;

    fn attention_program(queries: u32, keys: u32, head_dim: u32) -> Vec<Op> {
        let dtype = activation_dtype();
        let leaf = |program: &mut Vec<Op>, name: &str, shape: &[u32]| {
            let extents = shape.iter().map(|&size| Extent::Static(size)).collect();
            crate::spec::input_leaf(program, dtype, extents, name)
        };
        let mut program = Vec::new();
        let query = leaf(&mut program, "q", &[queries, ATTENTION_HEADS, head_dim]);
        let key = leaf(&mut program, "k", &[keys, ATTENTION_HEADS, head_dim]);
        let value = leaf(&mut program, "v", &[keys, ATTENTION_HEADS, head_dim]);
        let inv_sqrt =
            crate::spec::scalar_constant_like(&mut program, 0.5, query).expect("scale builds");
        crate::spec::non_causal_attention_heads(
            &mut program,
            &[(query, key)],
            value,
            inv_sqrt,
            "u",
            "i",
            "d",
        )
        .expect("attention builds");
        program
    }

    fn batched_contractions(program: &[Op]) -> Vec<(BoundOp, usize, ProductContraction)> {
        let shapes = shape::infer(program, &[]).expect("the attention infers");
        let output = NodeId((program.len() - 1) as u32);
        bind(program, &shapes, &[output], NumericPolicy::bit_exact())
            .expect("the attention binds")
            .into_iter()
            .filter_map(|bound| {
                dense_batched_contraction(&bound).map(|(weight, facts)| (bound, weight, facts))
            })
            .collect()
    }

    #[test]
    fn attention_scores_and_values_are_batched_over_heads_with_query_rows_as_tokens() {
        let (queries, keys, head_dim) = (3, 5, 2);
        let batched = batched_contractions(&attention_program(queries, keys, head_dim));
        assert_eq!(
            batched.len(),
            2,
            "the score product and the value product are the dense batched contractions"
        );
        for (_, _, facts) in &batched {
            assert_eq!(
                facts.batch_positions,
                vec![0],
                "the head axis leads the output"
            );
            assert_eq!(
                facts.token_positions,
                vec![1],
                "the query axis is the token position"
            );
            assert_eq!(facts.feature_positions, vec![2]);
            assert_eq!(facts.token_total, u64::from(queries));
        }
        let mut reductions: Vec<u64> = batched
            .iter()
            .map(|(_, _, facts)| facts.reduction_extent)
            .collect();
        reductions.sort_unstable();
        assert_eq!(reductions, vec![u64::from(head_dim), u64::from(keys)]);
    }

    #[test]
    fn a_decode_shaped_attention_keeps_the_head_axis_batched_not_a_token() {
        let batched = batched_contractions(&attention_program(1, 5, 2));
        assert_eq!(batched.len(), 2);
        for (_, _, facts) in &batched {
            assert_eq!(facts.batch_positions, vec![0]);
            assert_eq!(facts.token_total, 1);
        }
    }

    #[test]
    fn a_bound_single_token_product_has_a_token_total_of_one() {
        let bound = bound_contraction(&contraction_program(1));
        assert_eq!(product_contraction(&bound, 1).expect("a decode product").token_total, 1);
    }

    fn gathered_program() -> Vec<Op> {
        let dtype = activation_dtype();
        let mut program = Vec::new();
        let stack = append(
            &mut program,
            Op::Input {
                dtype,
                shape: alloc::vec![Extent::Static(3), Extent::Static(16), Extent::Static(8)],
                name: None,
            },
        );
        let route = append(
            &mut program,
            Op::Input {
                dtype,
                shape: alloc::vec![Extent::Static(2)],
                name: None,
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype,
                shape: alloc::vec![Extent::Static(2), Extent::Static(16)],
                name: None,
            },
        );
        let product = crate::spec::gathered_expert_product(&mut program, stack, route, activation);
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        program
    }

    #[test]
    fn a_bound_gathered_weight_counts_one_gathered_operand() {
        let bound = bound_contraction(&gathered_program());
        assert_eq!(gathered_operand_count(&bound), 1);
        assert_eq!(product_contraction(&bound, 0).expect("a gathered product").gathered_operands, 1);
    }

    fn contraction_executes(min_tokens: u64) -> Executes {
        Executes::BASE
            .with(Form::TiledContraction)
            .with(Form::GroupedContraction)
            .with_contraction(ContractionLimits {
                tiled_min_tokens: min_tokens,
                grouped_min_tokens: min_tokens,
            })
    }

    fn facts_of(program: &[Op], weight: usize) -> ProductContraction {
        product_contraction(&bound_contraction(program), weight).expect("a product contraction")
    }

    #[test]
    fn a_gathered_weight_with_routed_token_axes_is_grouped_under_a_granting_executor() {
        let facts = facts_of(&gathered_program(), 0);
        assert_eq!(facts.routed_positions, vec![0]);
        assert_eq!(contraction_form(&facts, &contraction_executes(2)), ContractionForm::Grouped);
    }

    #[test]
    fn a_gathered_weight_is_rowwise_under_an_executor_that_grants_neither_form() {
        let facts = facts_of(&gathered_program(), 0);
        assert_eq!(contraction_form(&facts, &Executes::BASE), ContractionForm::Rowwise);
        let tiled_only = Executes::BASE
            .with(Form::TiledContraction)
            .with_contraction(ContractionLimits::tiled_only(1));
        assert_eq!(contraction_form(&facts, &tiled_only), ContractionForm::Rowwise);
    }

    #[test]
    fn a_gathered_weight_is_flagged_and_a_second_gathered_operand_is_rowwise() {
        let facts = facts_of(&gathered_program(), 0);
        assert!(facts.weight_gathered);
        let two = ProductContraction { gathered_operands: 2, ..facts };
        assert_eq!(contraction_form(&two, &contraction_executes(2)), ContractionForm::Rowwise);
    }

    #[test]
    fn a_gathered_activation_with_a_dense_weight_is_rowwise() {
        let facts = facts_of(&gathered_program(), 0);
        let activation_gathered = ProductContraction { weight_gathered: false, ..facts };
        assert_eq!(contraction_form(&activation_gathered, &contraction_executes(2)), ContractionForm::Rowwise);
    }

    #[test]
    fn a_gathered_weight_below_the_grouped_minimum_is_rowwise() {
        let facts = facts_of(&gathered_program(), 0);
        assert_eq!(contraction_form(&facts, &contraction_executes(3)), ContractionForm::Rowwise);
    }

    #[test]
    fn a_dense_product_below_the_tiled_minimum_is_rowwise_and_at_it_is_tiled() {
        let executes = contraction_executes(4);
        let below = facts_of(&contraction_program(3), 1);
        let at = facts_of(&contraction_program(4), 1);
        assert_eq!(contraction_form(&below, &executes), ContractionForm::Rowwise);
        assert_eq!(contraction_form(&at, &executes), ContractionForm::Tiled);
    }

    #[test]
    fn a_dense_product_is_rowwise_without_the_tiled_grant() {
        let facts = facts_of(&contraction_program(4), 1);
        assert_eq!(contraction_form(&facts, &Executes::BASE), ContractionForm::Rowwise);
        let limits_only = Executes::BASE.with_contraction(ContractionLimits::tiled_only(1));
        assert_eq!(contraction_form(&facts, &limits_only), ContractionForm::Rowwise);
    }

    #[test]
    fn a_bound_max_reduce_is_not_a_product_contraction() {
        let mut program = contraction_program(4);
        let Some(Op::Reduce(reduce)) = program.last_mut() else {
            panic!("the fixture ends in a reduce");
        };
        reduce.body = ScalarOp::Maximum;
        assert_eq!(product_contraction(&bound_contraction(&program), 1), None);
    }
}
