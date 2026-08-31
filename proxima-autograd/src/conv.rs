//! `conv2d`, spelled as a graph-building function over
//! [`proxima_tensor::op::Op`] — never a new `Op`/`ScalarOp`/[`IndexMap`]
//! variant, and never the two-term-affine window
//! `proxima-onnx/src/lower.rs`'s `window_materialize` builds for forward-only
//! inference.
//!
//! [`crate::adjoint::differentiate`] rejects a convolution-style window
//! operand map outright: `crate::expr::is_pure_projection` requires every
//! axis to be exactly one unit-coefficient term, and `window_axis`'s
//! `stride*out + dilation*kernel` axis is two terms, so routing a gradient
//! through it raises [`crate::error::AutogradError::NonProjectionOperandMap`]
//! (`adjoint.rs:454-457`) or, on a `Reduce`'s own `in_map`, the same check at
//! `adjoint.rs:502-505`.
//!
//! # Why a mask-composed window, not a `IndexMap::Computed` gather
//!
//! An earlier version of this module built the window as a chain of two
//! [`IndexMap::Computed`] gathers (graph-time-known `out*stride + kernel`
//! indices, the same [`Op::Iota`]-built-index mechanism
//! `proxima-onnx/src/lower.rs`'s `pad_axis`/`slice_axis_range` use). That
//! composes correctly for a *single* conv layer's weight gradient, but
//! `crate::adjoint`'s private `route_contribution`'s `IndexMap::Computed` arm
//! deliberately stops at recording a [`crate::adjoint::GatheredContribution`]
//! rather than scattering it back into `grad_of` (that function's own doc:
//! "the step this crate stops short of"). So the backward walk never resumes
//! past a gathered operand — fine for an embedding table (a leaf `Input`,
//! nothing upstream needs its own gradient), fatal for a *stacked* conv net,
//! where the second layer's "image" is the first layer's own output and
//! *does* have an upstream (the first layer's weight) that needs the chain
//! rule to keep going.
//!
//! This module instead builds the window as `proxima-onnx`'s own
//! `convtranspose2d_scatter` (`proxima-onnx/src/lower.rs:2171`)
//! **mask-composition** idiom, read in
//! the gather direction instead of the scatter direction: widen the
//! iteration space by two axes (`out_position`, `kernel_position`), multiply
//! the source by a `0.0`/`1.0` [`ScalarOp::Equal`] mask built from
//! `source_position == out_position*stride + kernel_position` (three
//! [`Op::Iota`]s plus `Multiply`/`Add`/`Equal`, mirroring
//! `proxima-onnx/src/lower.rs`'s `scatter_mask_axis`), then `Reduce(Add)`
//! away the original source axis. Every operand read in this composition —
//! `source` included — is a pure single-term `IndexMap::Affine` projection,
//! so [`crate::adjoint`]'s ordinary `Elementwise`/`Reduce` rules apply with
//! no gather-specific carve-out, and gradient keeps flowing back through
//! `source` exactly as far as the rest of the program needs it to. The extra
//! cost versus a true gather is `O(source_extent)` per spatial axis (a
//! multiply-and-reduce over every source position, only one of which is ever
//! nonzero) instead of `O(1)`, at the size this module targets (28x28
//! MNIST-scale images) — cheap enough measured against a from-scratch
//! backward derivation, and zero new `Op`/`ScalarOp`/`IndexMap` either way.
//!
//! `masked_window_axis` (private) does the whole trick, once per spatial axis:
//! given a source tensor and one of its axes, it returns a tensor with that
//! axis reduced away and replaced by two new trailing axes `(out_position,
//! kernel_position)`. [`conv2d`] calls it twice (height, then width) to build
//! the classic im2col window `[n, c, out_h, kernel_h, out_w, kernel_w]`, then
//! an ordinary `Elementwise(Multiply)` against the weight and `Reduce(Add)`
//! over `(ci, kh, kw)` — exactly `proxima-onnx`'s own `conv2d_core`
//! composition, with the window built by mask-composition instead of a
//! window-shaped `IndexMap`.
//!
//! No pooling here: the architecture this module was built for (see
//! `tests/real_mnist_conv_training.rs`) uses stride to downsample instead of
//! a separate pool op, staying entirely inside the primitives this module
//! already proves out rather than adding `Reduce(Maximum)`-over-a-window (a
//! second window-shaped composition with its own argmax-routing gradient to
//! re-derive).

use alloc::vec::Vec;

use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};

use crate::expr;

/// `image`'s `[n, c, h, w]` extents, as a graph builder needs them stated
/// (this crate never infers shape mid-build; see `crate::expr`'s own doc:
/// "whether the result actually lowers is judged once, by
/// `proxima_tensor::shape::infer`").
pub type ImageShape = (u64, u64, u64, u64);
/// `weight`'s `[out_channels, in_channels, kernel_h, kernel_w]` extents.
pub type WeightShape = (u64, u64, u64, u64);

/// `[n, out_channels, out_h, out_w]` for a valid (no padding) convolution, or
/// `None` when the kernel does not fit the input at all — the same shape
/// `proxima-onnx/src/lower.rs`'s `conv_output_extent` returns for one axis,
/// generalized to the whole output tensor and specialized to `pads = 0`
/// (this module builds no padding machinery; see this module's own doc for
/// why the architecture it targets does not need one).
#[must_use]
pub fn conv2d_output_shape(image_shape: ImageShape, weight_shape: WeightShape, stride_h: u64, stride_w: u64) -> Option<ImageShape> {
    let (batch, in_channels, height, width) = image_shape;
    let (out_channels, weight_in_channels, kernel_h, kernel_w) = weight_shape;
    if weight_in_channels != in_channels || stride_h == 0 || stride_w == 0 {
        return None;
    }
    if kernel_h > height || kernel_w > width {
        return None;
    }
    let out_h = (height - kernel_h) / stride_h + 1;
    let out_w = (width - kernel_w) / stride_w + 1;
    Some((batch, out_channels, out_h, out_w))
}

/// `Conv2d`, `group = 1`, no padding: `[n, ci, h, w]` image, `[co, ci, kh,
/// kw]` weight, optional `[co]` bias, `[n, co, out_h, out_w]` result — the
/// same forward relation `proxima-onnx/src/lower.rs`'s `conv2d_core` builds,
/// differentiable through [`crate::adjoint::differentiate`] (image, weight,
/// and bias all included, and the image gradient keeps propagating into
/// whatever built `image` — see this module's own doc for why that matters
/// for a stacked conv net) because the window is built by mask-composition
/// rather than a two-term affine `window_axis` or an unresumable
/// `IndexMap::Computed` gather.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    program: &mut Vec<Op>,
    dtype: DType,
    image: NodeId,
    image_shape: ImageShape,
    weight: NodeId,
    weight_shape: WeightShape,
    bias: Option<NodeId>,
    stride_h: u64,
    stride_w: u64,
) -> Option<NodeId> {
    let (_, _, height, width) = image_shape;
    let (_, _, kernel_h, kernel_w) = weight_shape;
    let (_, _, out_h, out_w) = conv2d_output_shape(image_shape, weight_shape, stride_h, stride_w)?;

    // image axes: n=0, c=1, h=2, w=3 -- window h first (axis 2), reducing it
    // away and landing the two new axes (oy, ky) at the end: [n, c, w, oy, ky].
    let windowed_h = masked_window_axis(program, dtype, image, 4, 2, height, out_h, kernel_h, stride_h);
    // windowed_h axes: n=0, c=1, w=2, oy=3, ky=4 -- window w next (now axis
    // 2), landing [n, c, oy, ky, ox, kx].
    let windowed = masked_window_axis(program, dtype, windowed_h, 5, 2, width, out_w, kernel_w, stride_w);

    // shared iteration space: 0=n 1=co 2=ci 3=oy 4=ky 5=ox 6=kx
    let windowed_pattern = IndexMap::Affine(map::projection(7, &[0, 2, 3, 4, 5, 6]));
    let weight_pattern = IndexMap::Affine(map::projection(7, &[1, 2, 4, 6]));
    let product = expr::binary(program, dtype, ScalarOp::Multiply, (windowed, windowed_pattern), (weight, weight_pattern));

    let out_map = IndexMap::Affine(map::projection(7, &[0, 1, 3, 5]));
    let reduced = expr::reduce(program, dtype, ScalarOp::Add, ReduceInit::Zero, product, expr::identity(7), out_map);

    let result = match bias {
        Some(bias) => {
            let bias_pattern = IndexMap::Affine(map::projection(4, &[1]));
            expr::binary(program, dtype, ScalarOp::Add, (reduced, expr::identity(4)), (bias, bias_pattern))
        }
        None => reduced,
    };
    Some(result)
}

/// Replaces `source`'s axis `windowed_axis` (out of `source_rank` total, with
/// static extent `source_extent`) with two new trailing axes `(out_position,
/// kernel_position)`, `Reduce(Add)`-ing the original axis away under a
/// `source_position == out_position*stride + kernel_position` mask so only
/// the one matching term ever survives. Every other axis is carried straight
/// through in its original relative order. See this module's own doc for why
/// this reads `source` through a plain pure-projection [`IndexMap::Affine`]
/// (not a gather) -- gradient keeps flowing into `source` exactly the way an
/// ordinary `Multiply` + `Reduce(Add)` composition always does.
#[allow(clippy::too_many_arguments)]
fn masked_window_axis(
    program: &mut Vec<Op>,
    dtype: DType,
    source: NodeId,
    source_rank: usize,
    windowed_axis: usize,
    source_extent: u64,
    out_extent: u64,
    kernel_extent: u64,
    stride: u64,
) -> NodeId {
    let widened_rank = source_rank + 2;
    let widened_rank_u16 = widened_rank as u16;
    let out_position_axis = source_rank as u16;
    let kernel_position_axis = out_position_axis + 1;

    let mask = window_mask(program, source_extent, out_extent, kernel_extent, stride);
    let source_axes: Vec<u16> = (0..source_rank as u16).collect();
    let source_pattern = IndexMap::Affine(map::projection(widened_rank_u16, &source_axes));
    let mask_pattern = IndexMap::Affine(map::projection(widened_rank_u16, &[windowed_axis as u16, out_position_axis, kernel_position_axis]));

    let masked = expr::binary(program, dtype, ScalarOp::Multiply, (source, source_pattern), (mask, mask_pattern));

    let keep_axes: Vec<u16> = (0..widened_rank_u16).filter(|&axis| axis != windowed_axis as u16).collect();
    let out_map = IndexMap::Affine(map::projection(widened_rank_u16, &keep_axes));
    expr::reduce(program, dtype, ScalarOp::Add, ReduceInit::Zero, masked, expr::identity(widened_rank_u16), out_map)
}

/// `mask[source_position, out_position, kernel_position] = (source_position
/// == out_position*stride + kernel_position)` -- the exact
/// `Iota`/`Multiply`/`Add`/`Equal` composition
/// `proxima-onnx/src/lower.rs`'s `scatter_mask_axis` builds for
/// `ConvTranspose`'s general case, without that function's `pad` term (this
/// module builds no padding; see this module's own doc).
fn window_mask(program: &mut Vec<Op>, source_extent: u64, out_extent: u64, kernel_extent: u64, stride: u64) -> NodeId {
    let source_position = op::append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(source_extent as u32) });
    let out_position = op::append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(out_extent as u32) });
    let kernel_position = op::append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(kernel_extent as u32) });

    let stride_const = expr::constant(program, DType::Float32, stride as f32);
    let scaled_out = expr::binary(program, DType::Float32, ScalarOp::Multiply, (out_position, expr::identity(1)), (stride_const, expr::broadcast(1)));
    let combined = expr::binary(
        program,
        DType::Float32,
        ScalarOp::Add,
        (scaled_out, IndexMap::Affine(map::projection(2, &[0]))),
        (kernel_position, IndexMap::Affine(map::projection(2, &[1]))),
    );

    expr::binary(
        program,
        DType::Float32,
        ScalarOp::Equal,
        (source_position, IndexMap::Affine(map::projection(3, &[0]))),
        (combined, IndexMap::Affine(map::projection(3, &[1, 2]))),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use proxima_tensor::op::Extent;

    use super::*;

    fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
        op::append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
    }

    /// Hand-worked example: a `4x4` single-channel image, a `2x2` kernel,
    /// stride 1, no padding -- `out_h = out_w = 3`. `image` is `0..15` in
    /// row-major order:
    /// ```text
    ///  0  1  2  3
    ///  4  5  6  7
    ///  8  9 10 11
    /// 12 13 14 15
    /// ```
    /// `weight = [[1, 0], [0, -1]]` (top-left minus bottom-right of each
    /// window), no bias. `out[oy, ox] = image[oy, ox] - image[oy+1, ox+1]`.
    /// For `(oy, ox) = (0, 0)`: `0 - 5 = -5`. Every one of the 9 output
    /// positions below was hand-computed the same way, not generated from
    /// the code under test.
    #[proxima::test]
    async fn worked_example_2x2_kernel_over_4x4_image_matches_hand_computed_values() {
        let image_values: [f32; 16] = core::array::from_fn(|index| index as f32);
        let weight_values: [f32; 4] = [1.0, 0.0, 0.0, -1.0];

        let mut program = Vec::new();
        let image = leaf(&mut program, "image", vec![Extent::Static(1), Extent::Static(1), Extent::Static(4), Extent::Static(4)]);
        let weight = leaf(&mut program, "weight", vec![Extent::Static(1), Extent::Static(1), Extent::Static(2), Extent::Static(2)]);
        let out = conv2d(&mut program, DType::Float32, image, (1, 1, 4, 4), weight, (1, 1, 2, 2), None, 1, 1).expect("2x2 kernel fits a 4x4 image at stride 1");

        let evaluated = proxima_tensor::cpu::evaluate_named(&program, &[], &[("image", &image_values), ("weight", &weight_values)], &[out])
            .expect("conv2d program lowers and evaluates");
        let (result, shape) = evaluated.get(out).expect("conv output requested");
        assert_eq!(shape, &vec![1, 1, 3, 3], "a 2x2 kernel over a 4x4 image at stride 1 produces a 3x3 output");

        let expected = [-5.0f32, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0];
        for (position, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "position {position}: got {got}, want {want}, full result {result:?}");
        }
    }

    fn relative_error(analytic: f32, numeric: f32) -> f32 {
        (analytic - numeric).abs() / (analytic.abs().max(numeric.abs()) + 1e-6)
    }

    /// Deterministic xorshift pseudo-random pattern -- the same generator
    /// `proxima-autograd/tests/real_mnist_training.rs`'s own `he_init` uses,
    /// scaled to `[-0.5, 0.5)` rather than fan-in. A rational, small-integer
    /// pattern (`training_loop.rs`'s own `counter_pattern`, a plain linear
    /// ramp) leaves enough exact algebraic structure in a small convolution
    /// that some window's channel-summed gradient contribution lands at
    /// *exactly* zero by coincidence, where the central-difference numeric
    /// estimate is still off by its own float32 noise floor (`~1e-4` at
    /// `step = 1e-3`) and the *relative* error against a true zero blows up
    /// to nearly 1.0 even though the absolute disagreement is negligible.
    /// Xorshift output has no such shared rational structure across
    /// positions, so no analytic entry lands on an exact zero.
    fn pseudo_random(seed: u64, count: usize) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Central-difference gradient check over every element of a small
    /// `image`/`weight`/`bias`, the same oracle
    /// `proxima-autograd/tests/training_loop.rs`'s
    /// `central_difference_matches_the_analytic_gradient_on_every_parameter`
    /// uses: it has no notion of "convolution" or "mask-composed window", it
    /// simply differentiates the actual forward program numerically, so it
    /// is a correct check for this composition regardless of how the window
    /// was built. Checks `image` too, not only `weight`/`bias`: this
    /// module's whole reason for choosing mask-composition over a gather
    /// chain is that `image`'s own gradient must keep flowing (see this
    /// module's own doc), so a passing check here is the proof, not just a
    /// nice-to-have.
    #[proxima::test]
    async fn conv2d_gradient_matches_central_difference_on_every_parameter() {
        let mut program = Vec::new();
        let image = leaf(&mut program, "image", vec![Extent::Static(1), Extent::Static(2), Extent::Static(5), Extent::Static(5)]);
        let weight = leaf(&mut program, "weight", vec![Extent::Static(3), Extent::Static(2), Extent::Static(3), Extent::Static(3)]);
        let bias = leaf(&mut program, "bias", vec![Extent::Static(3)]);
        let out = conv2d(&mut program, DType::Float32, image, (1, 2, 5, 5), weight, (3, 2, 3, 3), Some(bias), 2, 2).expect("3x3 kernel fits a 5x5 image at stride 2");
        let loss = crate::expr::reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, out, crate::expr::identity(4), crate::expr::broadcast(4));

        let differentiated = crate::adjoint::differentiate(&program, loss).expect("conv2d program differentiates");
        let grad_image = differentiated.gradient_of_named("image").expect("image feeds the loss");
        let grad_weight = differentiated.gradient_of_named("weight").expect("weight feeds the loss");
        let grad_bias = differentiated.gradient_of_named("bias").expect("bias feeds the loss");

        let image_values = pseudo_random(0x9E37_79B9, 50);
        let weight_values = pseudo_random(0x8542_D2C3, 54);
        let bias_values: [f32; 3] = [0.1, -0.2, 0.3];

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &differentiated.program,
            &[],
            &[("image", &image_values), ("weight", &weight_values), ("bias", &bias_values)],
            &[grad_image, grad_weight, grad_bias],
        )
        .expect("adjoint program lowers and evaluates");
        let analytic_image = evaluated.get(grad_image).expect("grad_image requested").0.to_vec();
        let analytic_weight = evaluated.get(grad_weight).expect("grad_weight requested").0.to_vec();
        let analytic_bias = evaluated.get(grad_bias).expect("grad_bias requested").0.to_vec();

        let loss_at = |image: &[f32], weight: &[f32], bias: &[f32]| {
            proxima_tensor::cpu::evaluate_named(&program, &[], &[("image", image), ("weight", weight), ("bias", bias)], &[loss])
                .expect("forward program lowers and evaluates")
                .get(loss)
                .expect("loss requested")
                .0[0]
        };

        let step = 1e-3f32;
        let mut worst = (0.0f32, "", 0usize);

        let mut perturbed = image_values.clone();
        for index in 0..perturbed.len() {
            let original = perturbed[index];
            perturbed[index] = original + step;
            let plus = loss_at(&perturbed, &weight_values, &bias_values);
            perturbed[index] = original - step;
            let minus = loss_at(&perturbed, &weight_values, &bias_values);
            perturbed[index] = original;
            let numeric = (plus - minus) / (2.0 * step);
            let relative = relative_error(analytic_image[index], numeric);
            if relative > worst.0 {
                worst = (relative, "image", index);
            }
        }

        let mut perturbed = weight_values.clone();
        for index in 0..perturbed.len() {
            let original = perturbed[index];
            perturbed[index] = original + step;
            let plus = loss_at(&image_values, &perturbed, &bias_values);
            perturbed[index] = original - step;
            let minus = loss_at(&image_values, &perturbed, &bias_values);
            perturbed[index] = original;
            let numeric = (plus - minus) / (2.0 * step);
            let relative = relative_error(analytic_weight[index], numeric);
            if relative > worst.0 {
                worst = (relative, "weight", index);
            }
        }

        let mut perturbed = bias_values;
        for index in 0..perturbed.len() {
            let original = perturbed[index];
            perturbed[index] = original + step;
            let plus = loss_at(&image_values, &weight_values, &perturbed);
            perturbed[index] = original - step;
            let minus = loss_at(&image_values, &weight_values, &perturbed);
            perturbed[index] = original;
            let numeric = (plus - minus) / (2.0 * step);
            let relative = relative_error(analytic_bias[index], numeric);
            if relative > worst.0 {
                worst = (relative, "bias", index);
            }
        }

        std::eprintln!("conv2d max relative gradient-check error: {} at {}[{}]", worst.0, worst.1, worst.2);
        assert!(worst.0 < 5e-3, "conv2d adjoint disagreed with central difference: {worst:?}");
    }

    /// Two stacked `conv2d` layers, no activation between them to keep the
    /// check purely about gradient flow: proof that a second layer's
    /// gradient keeps propagating through its own `image` operand (the first
    /// layer's output) into the first layer's weight -- the exact case a
    /// gather-chain window (see this module's own doc) cannot support, since
    /// `route_contribution`'s `IndexMap::Computed` arm never resumes the
    /// backward walk past a gathered operand.
    #[proxima::test]
    async fn stacked_conv2d_gradient_reaches_the_first_layers_weight() {
        let mut program = Vec::new();
        let image = leaf(&mut program, "image", vec![Extent::Static(1), Extent::Static(1), Extent::Static(6), Extent::Static(6)]);
        let weight1 = leaf(&mut program, "weight1", vec![Extent::Static(2), Extent::Static(1), Extent::Static(3), Extent::Static(3)]);
        let weight2 = leaf(&mut program, "weight2", vec![Extent::Static(1), Extent::Static(2), Extent::Static(2), Extent::Static(2)]);

        let hidden = conv2d(&mut program, DType::Float32, image, (1, 1, 6, 6), weight1, (2, 1, 3, 3), None, 1, 1).expect("3x3 kernel fits a 6x6 image at stride 1");
        let out = conv2d(&mut program, DType::Float32, hidden, (1, 2, 4, 4), weight2, (1, 2, 2, 2), None, 1, 1).expect("2x2 kernel fits a 4x4 image at stride 1");
        let loss = crate::expr::reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, out, crate::expr::identity(4), crate::expr::broadcast(4));

        let differentiated = crate::adjoint::differentiate(&program, loss).expect("stacked conv2d program differentiates");
        let grad_weight1 = differentiated.gradient_of_named("weight1").expect("weight1 must receive a gradient through the second layer's own image operand");

        let image_values: Vec<f32> = (0..36).map(|index| (index as f32 - 18.0) / 6.0).collect();
        let weight1_values: Vec<f32> = (0..18).map(|index| (index as f32 - 9.0) / 6.0).collect();
        let weight2_values: Vec<f32> = (0..8).map(|index| (index as f32 - 4.0) / 4.0).collect();

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &differentiated.program,
            &[],
            &[("image", &image_values), ("weight1", &weight1_values), ("weight2", &weight2_values)],
            &[grad_weight1],
        )
        .expect("adjoint program lowers and evaluates");
        let analytic_weight1 = evaluated.get(grad_weight1).expect("grad_weight1 requested").0.to_vec();

        assert!(analytic_weight1.iter().any(|&value| value != 0.0), "weight1's gradient must be nonzero somewhere, got {analytic_weight1:?}");

        let loss_at = |weight1: &[f32]| {
            proxima_tensor::cpu::evaluate_named(&program, &[], &[("image", &image_values), ("weight1", weight1), ("weight2", &weight2_values)], &[loss])
                .expect("forward program lowers and evaluates")
                .get(loss)
                .expect("loss requested")
                .0[0]
        };

        let step = 1e-3f32;
        let mut worst = (0.0f32, 0usize);
        let mut perturbed = weight1_values.clone();
        for index in 0..perturbed.len() {
            let original = perturbed[index];
            perturbed[index] = original + step;
            let plus = loss_at(&perturbed);
            perturbed[index] = original - step;
            let minus = loss_at(&perturbed);
            perturbed[index] = original;
            let numeric = (plus - minus) / (2.0 * step);
            let relative = relative_error(analytic_weight1[index], numeric);
            if relative > worst.0 {
                worst = (relative, index);
            }
        }

        std::eprintln!("stacked conv2d weight1 max relative gradient-check error: {} at index {}", worst.0, worst.1);
        assert!(worst.0 < 5e-3, "stacked conv2d's first-layer weight gradient disagreed with central difference: {worst:?}");
    }

    #[test]
    fn conv2d_output_shape_matches_the_standard_valid_convolution_formula() {
        assert_eq!(conv2d_output_shape((1, 1, 4, 4), (1, 1, 2, 2), 1, 1), Some((1, 1, 3, 3)));
        assert_eq!(conv2d_output_shape((32, 1, 28, 28), (8, 1, 3, 3), 2, 2), Some((32, 8, 13, 13)));
        assert_eq!(conv2d_output_shape((32, 8, 13, 13), (16, 8, 3, 3), 2, 2), Some((32, 16, 6, 6)));
        assert_eq!(conv2d_output_shape((1, 1, 2, 2), (1, 1, 3, 3), 1, 1), None, "kernel larger than the image cannot fit");
        assert_eq!(conv2d_output_shape((1, 2, 4, 4), (1, 3, 2, 2), 1, 1), None, "weight in-channels must match the image");
    }
}
