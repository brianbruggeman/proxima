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

fn is_plain_product(body: &ComposedBody) -> bool {
    let [only] = body.steps.as_slice() else {
        return false;
    };
    only.op == ScalarOp::Multiply
        && (only.args == [StepArg::Operand(0), StepArg::Operand(1)]
            || only.args == [StepArg::Operand(1), StepArg::Operand(0)])
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
        gathered_operands: operands.iter().filter(|(_, _, gather)| gather.is_some()).count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
