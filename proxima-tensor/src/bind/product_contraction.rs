use super::*;

/// What every backend needs to know about a `weight * activation` sum before
/// choosing a kernel: which output positions the weight owns, which the
/// activation owns, and how much work each side carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductContraction {
    pub feature_positions: Vec<usize>,
    pub token_positions: Vec<usize>,
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

fn split_axes(
    extents: &[u64],
    output_axes: &[u16],
    weight_layout: &Layout,
    activation_layout: &Layout,
) -> Option<(Vec<usize>, Vec<usize>, u64, u64)> {
    let mut feature_positions = Vec::new();
    let mut token_positions = Vec::new();
    for (position, &axis) in output_axes.iter().enumerate() {
        match (weight_layout.stride(axis) != 0, activation_layout.stride(axis) != 0) {
            (true, false) => feature_positions.push(position),
            (false, true) => token_positions.push(position),
            _ => return None,
        }
    }
    let token_total = token_positions
        .iter()
        .map(|&position| extents[output_axes[position] as usize])
        .product();
    let reduction_extent = (0..extents.len() as u16)
        .filter(|dim| !output_axes.contains(dim))
        .map(|dim| extents[dim as usize])
        .product();
    Some((feature_positions, token_positions, token_total, reduction_extent))
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
    let (weight_layout, activation_layout) = if weight == 0 {
        (&first.1, &second.1)
    } else {
        (&second.1, &first.1)
    };
    let (feature_positions, token_positions, token_total, reduction_extent) =
        split_axes(&bound.extents, output_axes, weight_layout, activation_layout)?;
    Some(ProductContraction {
        feature_positions,
        token_positions,
        token_total,
        reduction_extent,
        gathered_operands: gathered_operand_count(bound),
    })
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
        split_axes(extents, &[0, 1], &layout(weight_strides), &layout(activation_strides))
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

    #[test]
    fn a_bound_gathered_weight_counts_one_gathered_operand() {
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
        let bound = bound_contraction(&program);
        assert_eq!(gathered_operand_count(&bound), 1);
        assert_eq!(product_contraction(&bound, 0).expect("a gathered product").gathered_operands, 1);
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
