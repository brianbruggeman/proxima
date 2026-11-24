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
    pub token_total: u64,
    pub reduction_extent: u64,
    pub gathered_operands: usize,
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
        token_total,
        reduction_extent,
    })
}

/// `Some` for a `Keep::Reduce` add-from-zero whose body is one multiply of
/// operands 0 and 1, with every output position owned by exactly one of the
/// two operands: the weight (`weight` is its operand index) makes it a
/// feature position, the other operand makes it a token position.
#[must_use]
pub fn product_contraction(bound: &BoundOp, weight: usize) -> Option<ProductContraction> {
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
        token_total: split.token_total,
        reduction_extent: split.reduction_extent,
        gathered_operands: gathered_operand_count(bound),
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
/// weight whose route follows the token axes takes the grouped form, a dense weight the
/// tiled one, each only when `executes` grants the form and the token rows
/// reach its minimum.
#[must_use]
pub fn contraction_form(facts: &ProductContraction, executes: &Executes) -> ContractionForm {
    let Some(limits) = executes.contraction else {
        return ContractionForm::Rowwise;
    };
    if facts.gathered_operands > 0 {
        let grouped = executes.forms.contains(Form::GroupedContraction)
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
    fn an_output_axis_owned_by_both_operands_is_refused() {
        assert_eq!(dense_split(&[8, 4, 32], &[32, 32, 1], &[0, 32, 1]), None);
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
