//! `conv2d` and `conv1d`, spelled as a graph-building function over
//! [`proxima_tensor::op::Op`] — never a new `Op`/`ScalarOp`/[`IndexMap`]
//! variant.
//!
//! # The window is one two-term affine read
//!
//! The windowed source is a single `Elementwise` read of the (padded) source
//! through the operand map `stride*out + dilation*kernel` (two terms in one
//! axis), the same read `proxima-onnx/src/lower.rs`'s `window_materialize1d`
//! and `stft_log_mel`'s `"{hop}*f+t->ft"` framing build. A stamp
//! (`Constant(1.0)` over `[out, kernel]`) pins the two window axes' extents.
//! The window is laid out `[n, out, c, kernel]` and the weight is `[co, c,
//! kernel]`, so the reduce axes `(c, kernel)` are the trailing axes of both
//! operands and fold as one contiguous `K = c*kernel` dot product.
//!
//! [`crate::adjoint::differentiate`] differentiates an `Elementwise` read
//! through a window operand map by scatter-adding the gradient along the same
//! map (`crate::affine_scatter`, reached from `route_contribution`), so
//! gradient keeps flowing through a stacked pair of layers. That backward
//! composition is a mask over the source axis; the forward program has no
//! operand wider than `n*out*c*kernel`.
//!
//! # Padding
//!
//! A read past the source extent is a shape error, so zero padding is a
//! materialised `[n, c, length + 2*padding]` tensor: a `Reduce(Add)` scatter
//! at `position + padding` (`pad_signal`), whose adjoint
//! (`differentiate_scatter`) is a gather of the upstream gradient and resumes
//! the backward walk. `proxima-onnx`'s `pad_axis` is a clamped
//! [`IndexMap::Computed`] gather: its backward stops at a recorded
//! [`crate::adjoint::GatheredContribution`] and never reaches a stacked
//! layer's weight, so it is not reused here.
//!
//! No pooling here: the architecture this module was built for (see
//! `tests/real_mnist_conv_training.rs`) uses stride to downsample instead of
//! a separate pool op, staying entirely inside the primitives this module
//! already proves out rather than adding `Reduce(Maximum)`-over-a-window (a
//! second window-shaped composition with its own argmax-routing gradient to
//! re-derive).

use alloc::vec::Vec;

use proxima_numeric::DType;
use proxima_tensor::map::{self, AxisIndex, AxisTerm, IndexMap};
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
pub fn conv2d_output_shape(
    image_shape: ImageShape,
    weight_shape: WeightShape,
    stride_h: u64,
    stride_w: u64,
) -> Option<ImageShape> {
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
/// for a stacked conv net) because the window is one two-term affine read
/// rather than an unresumable `IndexMap::Computed` gather.
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
    let (_, _, kernel_h, kernel_w) = weight_shape;
    let (_, _, out_h, out_w) = conv2d_output_shape(image_shape, weight_shape, stride_h, stride_w)?;

    // window iteration space: 0=n 1=oy 2=ox 3=c 4=ky 5=kx, image read
    // [n, c, stride_h*oy + ky, stride_w*ox + kx]
    let window_pattern = map::IndexPattern {
        iter_rank: 6,
        axes: alloc::vec![
            unit_axis(0),
            unit_axis(3),
            window_axis(1, 4, stride_h, 1)?,
            window_axis(2, 5, stride_w, 1)?,
        ],
    };
    let stamp = window_stamp(program, dtype, &[out_h, out_w, kernel_h, kernel_w]);
    let windowed = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (image, IndexMap::Affine(window_pattern)),
        (stamp, IndexMap::Affine(map::projection(6, &[1, 2, 4, 5]))),
    );

    // shared iteration space: 0=n 1=co 2=oy 3=ox 4=ci 5=ky 6=kx; the reduce
    // axes (ci, ky, kx) trail both the window [n, oy, ox, ci, ky, kx] and the
    // weight [co, ci, ky, kx]
    let windowed_pattern = IndexMap::Affine(map::projection(7, &[0, 2, 3, 4, 5, 6]));
    let weight_pattern = IndexMap::Affine(map::projection(7, &[1, 4, 5, 6]));
    let product = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (windowed, windowed_pattern),
        (weight, weight_pattern),
    );

    let out_map = IndexMap::Affine(map::projection(7, &[0, 1, 2, 3]));
    let reduced = expr::reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        expr::identity(7),
        out_map,
    );

    let result = match bias {
        Some(bias) => {
            let bias_pattern = IndexMap::Affine(map::projection(4, &[1]));
            expr::binary(
                program,
                dtype,
                ScalarOp::Add,
                (reduced, expr::identity(4)),
                (bias, bias_pattern),
            )
        }
        None => reduced,
    };
    Some(result)
}

/// `[n, c, length]` signal extents.
pub type SignalShape = (u64, u64, u64);
/// `[out_channels, in_channels, kernel]` extents.
pub type Kernel1dShape = (u64, u64, u64);

/// `[n, out_channels, out_length]` for a 1-D convolution with `stride`,
/// symmetric zero `padding` and `dilation`, or `None` when the dilated kernel
/// does not fit the padded signal, a channel count disagrees, or `stride` or
/// `dilation` is zero. `out_length = (length + 2*padding - dilation*(kernel-1)
/// - 1) / stride + 1`, the ONNX `Conv` rule.
#[must_use]
pub fn conv1d_output_shape(
    signal_shape: SignalShape,
    weight_shape: Kernel1dShape,
    stride: u64,
    padding: u64,
    dilation: u64,
) -> Option<SignalShape> {
    let (batch, in_channels, length) = signal_shape;
    let (out_channels, weight_in_channels, kernel) = weight_shape;
    if weight_in_channels != in_channels || stride == 0 || dilation == 0 || kernel == 0 {
        return None;
    }
    let reach = dilation * (kernel - 1) + 1;
    let padded = length + 2 * padding;
    (padded >= reach).then(|| (batch, out_channels, (padded - reach) / stride + 1))
}

/// `Conv1d`, `group = 1`: `[n, ci, l]` signal, `[co, ci, k]` weight, optional
/// `[co]` bias, `[n, co, out_l]` result. The one-spatial-axis case of
/// [`conv2d`]: the same two-term affine window (an `Elementwise` read, never a
/// window on a `Reduce`'s `in_map`), then `Multiply` against the weight and
/// `Reduce(Add)` over `(ci, k)`. Dilation is the kernel-axis coefficient and
/// padding is `pad_signal`'s scatter, so signal, weight and bias all
/// differentiate through [`crate::adjoint::differentiate`], and a stacked pair
/// of layers carries the gradient back to the first layer's weight.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    program: &mut Vec<Op>,
    dtype: DType,
    signal: NodeId,
    signal_shape: SignalShape,
    weight: NodeId,
    weight_shape: Kernel1dShape,
    bias: Option<NodeId>,
    stride: u64,
    padding: u64,
    dilation: u64,
) -> Option<NodeId> {
    let (_, _, length) = signal_shape;
    let (_, _, kernel) = weight_shape;
    let (_, _, out_length) =
        conv1d_output_shape(signal_shape, weight_shape, stride, padding, dilation)?;

    let padded = pad_signal(program, dtype, signal, padding, length)?;

    // window iteration space: 0=n 1=o 2=c 3=k, signal read [n, c, stride*o + dilation*k]
    let window_pattern = map::IndexPattern {
        iter_rank: 4,
        axes: alloc::vec![
            unit_axis(0),
            unit_axis(2),
            window_axis(1, 3, stride, dilation)?
        ],
    };
    let stamp = window_stamp(program, dtype, &[out_length, kernel]);
    let windowed = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (padded, IndexMap::Affine(window_pattern)),
        (stamp, IndexMap::Affine(map::projection(4, &[1, 3]))),
    );

    // shared iteration space: 0=n 1=co 2=o 3=ci 4=k; the reduce axes (ci, k)
    // are the trailing axes of both the window [n, o, ci, k] and the weight
    // [co, ci, k], so the sum folds as one contiguous K = ci*k dot product
    let windowed_pattern = IndexMap::Affine(map::projection(5, &[0, 2, 3, 4]));
    let weight_pattern = IndexMap::Affine(map::projection(5, &[1, 3, 4]));
    let product = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (windowed, windowed_pattern),
        (weight, weight_pattern),
    );
    let reduced = expr::reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        expr::identity(5),
        IndexMap::Affine(map::projection(5, &[0, 1, 2])),
    );

    let result = match bias {
        Some(bias) => expr::binary(
            program,
            dtype,
            ScalarOp::Add,
            (reduced, expr::identity(3)),
            (bias, IndexMap::Affine(map::projection(3, &[1]))),
        ),
        None => reduced,
    };
    Some(result)
}

fn unit_axis(axis: u16) -> AxisIndex {
    AxisIndex {
        terms: core::iter::once(AxisTerm::projection(axis)).collect(),
        offset: 0,
        len: None,
    }
}

fn count_constant(program: &mut Vec<Op>, dtype: DType, count: u64) -> NodeId {
    expr::constant(program, dtype, count as f32)
}

/// `[n, c, length + 2*padding]`: the clamped-gather-times-mask pad
/// `proxima-onnx`'s `pad_axis` builds, so every backend already runs it. The
/// gather reads `Identity(signal)` rather than `signal` so the operand is a
/// non-leaf node: `crate::adjoint` scatter-adds the gradient of a gather of a
/// non-leaf node into it, while a gathered leaf only records a sparse
/// contribution. Returns `signal` untouched when `padding` is zero.
fn pad_signal(
    program: &mut Vec<Op>,
    dtype: DType,
    signal: NodeId,
    padding: u64,
    length: u64,
) -> Option<NodeId> {
    if padding == 0 {
        return Some(signal);
    }
    let padded = u32::try_from(length + 2 * padding).ok()?;
    let position = op::append(
        program,
        Op::Iota {
            dtype,
            extent: Extent::Static(padded),
        },
    );
    let before = count_constant(program, dtype, padding);
    let shifted = expr::binary(
        program,
        dtype,
        ScalarOp::Subtract,
        (position, expr::identity(1)),
        (before, expr::broadcast(1)),
    );
    let zero = expr::constant(program, dtype, 0.0);
    let clamped_low = expr::binary(
        program,
        dtype,
        ScalarOp::Maximum,
        (shifted, expr::identity(1)),
        (zero, expr::broadcast(1)),
    );
    let last = count_constant(program, dtype, length.saturating_sub(1));
    let clamped_index = expr::binary(
        program,
        DType::INT32,
        ScalarOp::Minimum,
        (clamped_low, expr::identity(1)),
        (last, expr::broadcast(1)),
    );
    let below = expr::constant(program, dtype, -1.0);
    let valid_low = expr::binary(
        program,
        dtype,
        ScalarOp::Greater,
        (shifted, expr::identity(1)),
        (below, expr::broadcast(1)),
    );
    let extent = count_constant(program, dtype, length);
    let valid_high = expr::binary(
        program,
        dtype,
        ScalarOp::Greater,
        (extent, expr::broadcast(1)),
        (shifted, expr::identity(1)),
    );
    let mask = expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (valid_low, expr::identity(1)),
        (valid_high, expr::identity(1)),
    );

    let operand = expr::unary(
        program,
        dtype,
        ScalarOp::Identity,
        (signal, expr::identity(3)),
    );
    let base = map::IndexPattern {
        iter_rank: 3,
        axes: alloc::vec![unit_axis(0), unit_axis(1), AxisIndex::default()],
    };
    let gather_map = IndexMap::Computed {
        indices: clamped_index,
        index_map: map::projection(3, &[2]),
        base,
        gathered_dim: 2,
    };
    let gathered = expr::unary(program, dtype, ScalarOp::Identity, (operand, gather_map));
    Some(expr::binary(
        program,
        dtype,
        ScalarOp::Multiply,
        (gathered, expr::identity(3)),
        (mask, IndexMap::Affine(map::projection(3, &[2]))),
    ))
}

/// A window term `stride*out_axis + dilation*kernel_axis` on one operand axis.
fn window_axis(out_axis: u16, kernel_axis: u16, stride: u64, dilation: u64) -> Option<AxisIndex> {
    Some(AxisIndex {
        terms: [
            AxisTerm::scaled(out_axis, i32::try_from(stride).ok()?),
            AxisTerm::scaled(kernel_axis, i32::try_from(dilation).ok()?),
        ]
        .into_iter()
        .collect(),
        offset: 0,
        len: None,
    })
}

/// `Constant(1.0)` of the given extents. The two-term window read names no
/// iteration axis of its own extent, so this stamp (the same device
/// `stft_log_mel`'s framing and `proxima-onnx`'s `window_materialize1d` use)
/// is the operand that pins `out` and `kernel` through plain projections.
fn window_stamp(program: &mut Vec<Op>, dtype: DType, extents: &[u64]) -> NodeId {
    expr::broadcast_anchor(program, dtype, extents, 1.0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use proxima_tensor::op::Extent;

    use super::*;

    fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
        op::append(
            program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape,
                name: Some(name.into()),
            },
        )
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
        let image = leaf(
            &mut program,
            "image",
            vec![
                Extent::Static(1),
                Extent::Static(1),
                Extent::Static(4),
                Extent::Static(4),
            ],
        );
        let weight = leaf(
            &mut program,
            "weight",
            vec![
                Extent::Static(1),
                Extent::Static(1),
                Extent::Static(2),
                Extent::Static(2),
            ],
        );
        let out = conv2d(
            &mut program,
            DType::FLOAT32,
            image,
            (1, 1, 4, 4),
            weight,
            (1, 1, 2, 2),
            None,
            1,
            1,
        )
        .expect("2x2 kernel fits a 4x4 image at stride 1");

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[("image", &image_values), ("weight", &weight_values)],
            &[out],
        )
        .expect("conv2d program lowers and evaluates");
        let (result, shape) = evaluated.get(out).expect("conv output requested");
        assert_eq!(
            shape,
            &vec![1, 1, 3, 3],
            "a 2x2 kernel over a 4x4 image at stride 1 produces a 3x3 output"
        );

        let expected = [-5.0f32, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0];
        for (position, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "position {position}: got {got}, want {want}, full result {result:?}"
            );
        }
    }

    /// Forward-only (no `differentiate`) at stride 2 -- isolates the
    /// two-term window read from any backward machinery and checks it
    /// against a hand-computed stride-2 window.
    /// `image` is `0..24` row-major over a `1x1x5x5` tensor:
    /// ```text
    ///  0  1  2  3  4
    ///  5  6  7  8  9
    /// 10 11 12 13 14
    /// 15 16 17 18 19
    /// 20 21 22 23 24
    /// ```
    /// `weight = [[1, 0], [0, -1]]`, no bias, stride 2 -> `out_h = out_w = 2`,
    /// windows anchored at `(0, 0)`, `(0, 2)`, `(2, 0)`, `(2, 2)`.
    /// `out[oy, ox] = image[2oy, 2ox] - image[2oy+1, 2ox+1]`.
    #[proxima::test]
    async fn forward_only_stride_two_matches_hand_computed_values() {
        let image_values: [f32; 25] = core::array::from_fn(|index| index as f32);
        let weight_values: [f32; 4] = [1.0, 0.0, 0.0, -1.0];

        let mut program = Vec::new();
        let image = leaf(
            &mut program,
            "image",
            vec![
                Extent::Static(1),
                Extent::Static(1),
                Extent::Static(5),
                Extent::Static(5),
            ],
        );
        let weight = leaf(
            &mut program,
            "weight",
            vec![
                Extent::Static(1),
                Extent::Static(1),
                Extent::Static(2),
                Extent::Static(2),
            ],
        );
        let out = conv2d(
            &mut program,
            DType::FLOAT32,
            image,
            (1, 1, 5, 5),
            weight,
            (1, 1, 2, 2),
            None,
            2,
            2,
        )
        .expect("2x2 kernel fits a 5x5 image at stride 2");

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[("image", &image_values), ("weight", &weight_values)],
            &[out],
        )
        .expect("conv2d program lowers and evaluates");
        let (result, shape) = evaluated.get(out).expect("conv output requested");
        assert_eq!(
            shape,
            &vec![1, 1, 2, 2],
            "a 2x2 kernel over a 5x5 image at stride 2 produces a 2x2 output"
        );

        // out[0,0]=image[0,0]-image[1,1]=0-6=-6; out[0,1]=image[0,2]-image[1,3]=2-8=-6;
        // out[1,0]=image[2,0]-image[3,1]=10-16=-6; out[1,1]=image[2,2]-image[3,3]=12-18=-6.
        let expected = [-6.0f32, -6.0, -6.0, -6.0];
        for (position, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "position {position}: got {got}, want {want}, full result {result:?}"
            );
        }
    }

    /// Plain nested-loop reference convolution -- `[n, ci, h, w]` image,
    /// `[co, ci, kh, kw]` weight, valid (no padding) -- used only to check
    /// `conv2d`'s own output against an implementation that shares none of
    /// its machinery (no window map, no `Iota`, no `IndexMap`).
    #[allow(clippy::too_many_arguments)]
    fn brute_force_conv2d(
        image: &[f32],
        batch: usize,
        in_channels: usize,
        height: usize,
        width: usize,
        weight: &[f32],
        out_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> (Vec<f32>, usize, usize) {
        let out_h = (height - kernel) / stride + 1;
        let out_w = (width - kernel) / stride + 1;
        let mut out = vec![0.0f32; batch * out_channels * out_h * out_w];
        for n in 0..batch {
            for co in 0..out_channels {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut accumulator = 0.0f32;
                        for ci in 0..in_channels {
                            for ky in 0..kernel {
                                for kx in 0..kernel {
                                    let iy = oy * stride + ky;
                                    let ix = ox * stride + kx;
                                    let image_index =
                                        ((n * in_channels + ci) * height + iy) * width + ix;
                                    let weight_index =
                                        ((co * in_channels + ci) * kernel + ky) * kernel + kx;
                                    accumulator += image[image_index] * weight[weight_index];
                                }
                            }
                        }
                        out[((n * out_channels + co) * out_h + oy) * out_w + ox] = accumulator;
                    }
                }
            }
        }
        (out, out_h, out_w)
    }

    /// Two stacked `conv2d` layers, forward-only (no `differentiate`; see
    /// `forward_only_stride_two_matches_hand_computed_values`'s own doc),
    /// multi-channel, stride 2 on both layers, against
    /// `brute_force_conv2d` chained twice by hand -- the exact shape
    /// `real_mnist_conv_training.rs`'s own `build_eval_network` uses.
    #[proxima::test]
    async fn stacked_forward_only_multichannel_stride_two_matches_a_brute_force_reference() {
        let (batch, in_channels, height, width) = (2usize, 1usize, 9usize, 9usize);
        let (mid_channels, kernel, stride) = (3usize, 3usize, 2usize);
        let out_channels = 2usize;

        let image_values = pseudo_random(0x51EE_D0A5, batch * in_channels * height * width);
        let weight1_values =
            pseudo_random(0xACE1_5EED, mid_channels * in_channels * kernel * kernel);
        let weight2_values =
            pseudo_random(0xC0FF_EE11, out_channels * mid_channels * kernel * kernel);

        let mut program = Vec::new();
        let image = leaf(
            &mut program,
            "image",
            vec![
                Extent::Static(batch as u32),
                Extent::Static(in_channels as u32),
                Extent::Static(height as u32),
                Extent::Static(width as u32),
            ],
        );
        let weight1 = leaf(
            &mut program,
            "weight1",
            vec![
                Extent::Static(mid_channels as u32),
                Extent::Static(in_channels as u32),
                Extent::Static(kernel as u32),
                Extent::Static(kernel as u32),
            ],
        );
        let weight2 = leaf(
            &mut program,
            "weight2",
            vec![
                Extent::Static(out_channels as u32),
                Extent::Static(mid_channels as u32),
                Extent::Static(kernel as u32),
                Extent::Static(kernel as u32),
            ],
        );

        let hidden = conv2d(
            &mut program,
            DType::FLOAT32,
            image,
            (
                batch as u64,
                in_channels as u64,
                height as u64,
                width as u64,
            ),
            weight1,
            (
                mid_channels as u64,
                in_channels as u64,
                kernel as u64,
                kernel as u64,
            ),
            None,
            stride as u64,
            stride as u64,
        )
        .expect("3x3 kernel fits a 9x9 image at stride 2");
        let (_, _, mid_h, mid_w) = conv2d_output_shape(
            (
                batch as u64,
                in_channels as u64,
                height as u64,
                width as u64,
            ),
            (
                mid_channels as u64,
                in_channels as u64,
                kernel as u64,
                kernel as u64,
            ),
            stride as u64,
            stride as u64,
        )
        .expect("mid shape");
        let out = conv2d(
            &mut program,
            DType::FLOAT32,
            hidden,
            (batch as u64, mid_channels as u64, mid_h, mid_w),
            weight2,
            (
                out_channels as u64,
                mid_channels as u64,
                kernel as u64,
                kernel as u64,
            ),
            None,
            stride as u64,
            stride as u64,
        )
        .expect("3x3 kernel fits the first layer's output at stride 2");

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[
                ("image", &image_values),
                ("weight1", &weight1_values),
                ("weight2", &weight2_values),
            ],
            &[out],
        )
        .expect("stacked conv2d program lowers and evaluates");
        let (result, _shape) = evaluated.get(out).expect("stacked conv2d output requested");

        let (hidden_reference, hidden_h, hidden_w) = brute_force_conv2d(
            &image_values,
            batch,
            in_channels,
            height,
            width,
            &weight1_values,
            mid_channels,
            kernel,
            stride,
        );
        let (expected, _out_h, _out_w) = brute_force_conv2d(
            &hidden_reference,
            batch,
            mid_channels,
            hidden_h,
            hidden_w,
            &weight2_values,
            out_channels,
            kernel,
            stride,
        );

        for (position, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "position {position}: got {got}, want {want}"
            );
        }
    }

    /// Bisection step: same small shape as the no-bias stacked test above,
    /// but with bias on the FIRST layer only -- isolates whether "stacked +
    /// bias" disagrees regardless of scale, or only at real MNIST dimensions.
    #[proxima::test]
    async fn stacked_forward_only_small_with_bias_on_first_layer_matches_a_brute_force_reference() {
        let (batch, in_channels, height, width) = (2usize, 1usize, 9usize, 9usize);
        let (mid_channels, kernel, stride) = (3usize, 3usize, 2usize);
        let out_channels = 2usize;

        let image_values = pseudo_random(0x51EE_D0A5, batch * in_channels * height * width);
        let weight1_values =
            pseudo_random(0xACE1_5EED, mid_channels * in_channels * kernel * kernel);
        let bias1_values = pseudo_random(0xB1A5_0001, mid_channels);
        let weight2_values =
            pseudo_random(0xC0FF_EE11, out_channels * mid_channels * kernel * kernel);

        let mut program = Vec::new();
        let image = leaf(
            &mut program,
            "image",
            vec![
                Extent::Static(batch as u32),
                Extent::Static(in_channels as u32),
                Extent::Static(height as u32),
                Extent::Static(width as u32),
            ],
        );
        let weight1 = leaf(
            &mut program,
            "weight1",
            vec![
                Extent::Static(mid_channels as u32),
                Extent::Static(in_channels as u32),
                Extent::Static(kernel as u32),
                Extent::Static(kernel as u32),
            ],
        );
        let bias1 = leaf(
            &mut program,
            "bias1",
            vec![Extent::Static(mid_channels as u32)],
        );
        let weight2 = leaf(
            &mut program,
            "weight2",
            vec![
                Extent::Static(out_channels as u32),
                Extent::Static(mid_channels as u32),
                Extent::Static(kernel as u32),
                Extent::Static(kernel as u32),
            ],
        );

        let hidden = conv2d(
            &mut program,
            DType::FLOAT32,
            image,
            (
                batch as u64,
                in_channels as u64,
                height as u64,
                width as u64,
            ),
            weight1,
            (
                mid_channels as u64,
                in_channels as u64,
                kernel as u64,
                kernel as u64,
            ),
            Some(bias1),
            stride as u64,
            stride as u64,
        )
        .expect("3x3 kernel fits a 9x9 image at stride 2");
        let (_, _, mid_h, mid_w) = conv2d_output_shape(
            (
                batch as u64,
                in_channels as u64,
                height as u64,
                width as u64,
            ),
            (
                mid_channels as u64,
                in_channels as u64,
                kernel as u64,
                kernel as u64,
            ),
            stride as u64,
            stride as u64,
        )
        .expect("mid shape");
        let out = conv2d(
            &mut program,
            DType::FLOAT32,
            hidden,
            (batch as u64, mid_channels as u64, mid_h, mid_w),
            weight2,
            (
                out_channels as u64,
                mid_channels as u64,
                kernel as u64,
                kernel as u64,
            ),
            None,
            stride as u64,
            stride as u64,
        )
        .expect("3x3 kernel fits the first layer's output at stride 2");

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[
                ("image", &image_values),
                ("weight1", &weight1_values),
                ("bias1", &bias1_values),
                ("weight2", &weight2_values),
            ],
            &[out],
        )
        .expect("stacked conv2d program lowers and evaluates");
        let (result, _shape) = evaluated.get(out).expect("stacked conv2d output requested");

        let (mut hidden_reference, hidden_h, hidden_w) = brute_force_conv2d(
            &image_values,
            batch,
            in_channels,
            height,
            width,
            &weight1_values,
            mid_channels,
            kernel,
            stride,
        );
        add_bias(
            &mut hidden_reference,
            batch,
            mid_channels,
            hidden_h,
            hidden_w,
            &bias1_values,
        );
        let (expected, _out_h, _out_w) = brute_force_conv2d(
            &hidden_reference,
            batch,
            mid_channels,
            hidden_h,
            hidden_w,
            &weight2_values,
            out_channels,
            kernel,
            stride,
        );

        let mut worst = 0.0f32;
        for (position, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            let difference = (got - want).abs();
            if difference > worst {
                worst = difference;
            }
            assert!(
                difference < 1e-3,
                "position {position}: got {got}, want {want}, worst-so-far {worst}"
            );
        }
    }

    /// Same shape as `stacked_forward_only_multichannel_stride_two_matches_a_brute_force_reference`,
    /// but at `real_mnist_conv_training.rs`'s own real dimensions (28x28
    /// image, 1->8->16 channels, 3x3 kernel, stride 2, bias on both layers,
    /// `relu` between them) -- isolates whether the bug (if any) is
    /// dimension-dependent rather than shape-dependent.
    #[proxima::test]
    async fn stacked_forward_only_at_real_mnist_dimensions_matches_a_brute_force_reference() {
        let (batch, in_channels, height, width) = (2usize, 1usize, 28usize, 28usize);
        let (mid_channels, kernel, stride) = (8usize, 3usize, 2usize);
        let out_channels = 16usize;

        let image_values = pseudo_random(0x51EE_D0A5, batch * in_channels * height * width);
        let weight1_values =
            pseudo_random(0xACE1_5EED, mid_channels * in_channels * kernel * kernel);
        let bias1_values = pseudo_random(0xB1A5_0001, mid_channels);
        let weight2_values =
            pseudo_random(0xC0FF_EE11, out_channels * mid_channels * kernel * kernel);
        let bias2_values = pseudo_random(0xB1A5_0002, out_channels);

        let mut program = Vec::new();
        let image = leaf(
            &mut program,
            "image",
            vec![
                Extent::Static(batch as u32),
                Extent::Static(in_channels as u32),
                Extent::Static(height as u32),
                Extent::Static(width as u32),
            ],
        );
        let weight1 = leaf(
            &mut program,
            "weight1",
            vec![
                Extent::Static(mid_channels as u32),
                Extent::Static(in_channels as u32),
                Extent::Static(kernel as u32),
                Extent::Static(kernel as u32),
            ],
        );
        let bias1 = leaf(
            &mut program,
            "bias1",
            vec![Extent::Static(mid_channels as u32)],
        );
        let weight2 = leaf(
            &mut program,
            "weight2",
            vec![
                Extent::Static(out_channels as u32),
                Extent::Static(mid_channels as u32),
                Extent::Static(kernel as u32),
                Extent::Static(kernel as u32),
            ],
        );
        let bias2 = leaf(
            &mut program,
            "bias2",
            vec![Extent::Static(out_channels as u32)],
        );

        let hidden_pre = conv2d(
            &mut program,
            DType::FLOAT32,
            image,
            (
                batch as u64,
                in_channels as u64,
                height as u64,
                width as u64,
            ),
            weight1,
            (
                mid_channels as u64,
                in_channels as u64,
                kernel as u64,
                kernel as u64,
            ),
            Some(bias1),
            stride as u64,
            stride as u64,
        )
        .expect("3x3 kernel fits a 28x28 image at stride 2");
        let hidden = crate::activation::relu(&mut program, DType::FLOAT32, hidden_pre, 4);
        let (_, _, mid_h, mid_w) = conv2d_output_shape(
            (
                batch as u64,
                in_channels as u64,
                height as u64,
                width as u64,
            ),
            (
                mid_channels as u64,
                in_channels as u64,
                kernel as u64,
                kernel as u64,
            ),
            stride as u64,
            stride as u64,
        )
        .expect("mid shape");
        let out_pre = conv2d(
            &mut program,
            DType::FLOAT32,
            hidden,
            (batch as u64, mid_channels as u64, mid_h, mid_w),
            weight2,
            (
                out_channels as u64,
                mid_channels as u64,
                kernel as u64,
                kernel as u64,
            ),
            Some(bias2),
            stride as u64,
            stride as u64,
        )
        .expect("3x3 kernel fits the first layer's output at stride 2");
        let out = crate::activation::relu(&mut program, DType::FLOAT32, out_pre, 4);

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[
                ("image", &image_values),
                ("weight1", &weight1_values),
                ("bias1", &bias1_values),
                ("weight2", &weight2_values),
                ("bias2", &bias2_values),
            ],
            &[out],
        )
        .expect("stacked conv2d program lowers and evaluates");
        let (result, _shape) = evaluated.get(out).expect("stacked conv2d output requested");

        let (mut hidden_reference, hidden_h, hidden_w) = brute_force_conv2d(
            &image_values,
            batch,
            in_channels,
            height,
            width,
            &weight1_values,
            mid_channels,
            kernel,
            stride,
        );
        add_bias_and_relu(
            &mut hidden_reference,
            batch,
            mid_channels,
            hidden_h,
            hidden_w,
            &bias1_values,
        );
        let (mut expected, out_h, out_w) = brute_force_conv2d(
            &hidden_reference,
            batch,
            mid_channels,
            hidden_h,
            hidden_w,
            &weight2_values,
            out_channels,
            kernel,
            stride,
        );
        add_bias_and_relu(
            &mut expected,
            batch,
            out_channels,
            out_h,
            out_w,
            &bias2_values,
        );

        let mut worst = 0.0f32;
        for (position, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            let difference = (got - want).abs();
            if difference > worst {
                worst = difference;
            }
            assert!(
                difference < 1e-3,
                "position {position}: got {got}, want {want}, worst-so-far {worst}"
            );
        }
    }

    fn add_bias(
        values: &mut [f32],
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        bias: &[f32],
    ) {
        for n in 0..batch {
            for (channel, &channel_bias) in bias.iter().enumerate().take(channels) {
                for position in 0..height * width {
                    let index = (n * channels + channel) * height * width + position;
                    values[index] += channel_bias;
                }
            }
        }
    }

    fn add_bias_and_relu(
        values: &mut [f32],
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        bias: &[f32],
    ) {
        add_bias(values, batch, channels, height, width, bias);
        for value in values.iter_mut() {
            *value = value.max(0.0);
        }
    }

    /// Bisection step: single conv2d layer (no stacking), forward-only,
    /// with bias, at real MNIST layer-1 dimensions -- isolates whether bias
    /// alone at this scale disagrees, independent of any second layer.
    #[proxima::test]
    async fn single_layer_forward_only_with_bias_at_real_mnist_dimensions_matches_a_brute_force_reference()
     {
        let (batch, in_channels, height, width) = (2usize, 1usize, 28usize, 28usize);
        let (mid_channels, kernel, stride) = (8usize, 3usize, 2usize);

        let image_values = pseudo_random(0x51EE_D0A5, batch * in_channels * height * width);
        let weight1_values =
            pseudo_random(0xACE1_5EED, mid_channels * in_channels * kernel * kernel);
        let bias1_values = pseudo_random(0xB1A5_0001, mid_channels);

        let mut program = Vec::new();
        let image = leaf(
            &mut program,
            "image",
            vec![
                Extent::Static(batch as u32),
                Extent::Static(in_channels as u32),
                Extent::Static(height as u32),
                Extent::Static(width as u32),
            ],
        );
        let weight1 = leaf(
            &mut program,
            "weight1",
            vec![
                Extent::Static(mid_channels as u32),
                Extent::Static(in_channels as u32),
                Extent::Static(kernel as u32),
                Extent::Static(kernel as u32),
            ],
        );
        let bias1 = leaf(
            &mut program,
            "bias1",
            vec![Extent::Static(mid_channels as u32)],
        );

        let out = conv2d(
            &mut program,
            DType::FLOAT32,
            image,
            (
                batch as u64,
                in_channels as u64,
                height as u64,
                width as u64,
            ),
            weight1,
            (
                mid_channels as u64,
                in_channels as u64,
                kernel as u64,
                kernel as u64,
            ),
            Some(bias1),
            stride as u64,
            stride as u64,
        )
        .expect("3x3 kernel fits a 28x28 image at stride 2");

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &program,
            &[],
            &[
                ("image", &image_values),
                ("weight1", &weight1_values),
                ("bias1", &bias1_values),
            ],
            &[out],
        )
        .expect("conv2d program lowers and evaluates");
        let (result, _shape) = evaluated.get(out).expect("conv2d output requested");

        let (mut expected, _out_h, _out_w) = brute_force_conv2d(
            &image_values,
            batch,
            in_channels,
            height,
            width,
            &weight1_values,
            mid_channels,
            kernel,
            stride,
        );
        add_bias(
            &mut expected,
            batch,
            mid_channels,
            _out_h,
            _out_w,
            &bias1_values,
        );

        let mut worst = 0.0f32;
        for (position, (&got, &want)) in result.iter().zip(expected.iter()).enumerate() {
            let difference = (got - want).abs();
            if difference > worst {
                worst = difference;
            }
            assert!(
                difference < 1e-3,
                "position {position}: got {got}, want {want}, worst-so-far {worst}"
            );
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
    /// uses: it has no notion of "convolution" or "window", it simply
    /// differentiates the actual forward program numerically, so it is a
    /// correct check for this composition regardless of how the window was
    /// built. Checks `image` too, not only `weight`/`bias`: `image`'s own
    /// gradient must keep flowing through the window read (see this module's
    /// own doc), so a passing check here is the proof, not just a
    /// nice-to-have.
    #[proxima::test]
    async fn conv2d_gradient_matches_central_difference_on_every_parameter() {
        let mut program = Vec::new();
        let image = leaf(
            &mut program,
            "image",
            vec![
                Extent::Static(1),
                Extent::Static(2),
                Extent::Static(5),
                Extent::Static(5),
            ],
        );
        let weight = leaf(
            &mut program,
            "weight",
            vec![
                Extent::Static(3),
                Extent::Static(2),
                Extent::Static(3),
                Extent::Static(3),
            ],
        );
        let bias = leaf(&mut program, "bias", vec![Extent::Static(3)]);
        let out = conv2d(
            &mut program,
            DType::FLOAT32,
            image,
            (1, 2, 5, 5),
            weight,
            (3, 2, 3, 3),
            Some(bias),
            2,
            2,
        )
        .expect("3x3 kernel fits a 5x5 image at stride 2");
        let loss = crate::expr::reduce(
            &mut program,
            DType::FLOAT32,
            ScalarOp::Add,
            ReduceInit::Zero,
            out,
            crate::expr::identity(4),
            crate::expr::broadcast(4),
        );

        let differentiated =
            crate::adjoint::differentiate(&program, loss).expect("conv2d program differentiates");
        let grad_image = differentiated
            .gradient_of_named("image")
            .expect("image feeds the loss");
        let grad_weight = differentiated
            .gradient_of_named("weight")
            .expect("weight feeds the loss");
        let grad_bias = differentiated
            .gradient_of_named("bias")
            .expect("bias feeds the loss");

        let image_values = pseudo_random(0x9E37_79B9, 50);
        let weight_values = pseudo_random(0x8542_D2C3, 54);
        let bias_values: [f32; 3] = [0.1, -0.2, 0.3];

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &differentiated.program,
            &[],
            &[
                ("image", &image_values),
                ("weight", &weight_values),
                ("bias", &bias_values),
            ],
            &[grad_image, grad_weight, grad_bias],
        )
        .expect("adjoint program lowers and evaluates");
        let analytic_image = evaluated
            .get(grad_image)
            .expect("grad_image requested")
            .0
            .to_vec();
        let analytic_weight = evaluated
            .get(grad_weight)
            .expect("grad_weight requested")
            .0
            .to_vec();
        let analytic_bias = evaluated
            .get(grad_bias)
            .expect("grad_bias requested")
            .0
            .to_vec();

        let loss_at = |image: &[f32], weight: &[f32], bias: &[f32]| {
            proxima_tensor::cpu::evaluate_named(
                &program,
                &[],
                &[("image", image), ("weight", weight), ("bias", bias)],
                &[loss],
            )
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

        std::eprintln!(
            "conv2d max relative gradient-check error: {} at {}[{}]",
            worst.0,
            worst.1,
            worst.2
        );
        assert!(
            worst.0 < 5e-3,
            "conv2d adjoint disagreed with central difference: {worst:?}"
        );
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
        let image = leaf(
            &mut program,
            "image",
            vec![
                Extent::Static(1),
                Extent::Static(1),
                Extent::Static(6),
                Extent::Static(6),
            ],
        );
        let weight1 = leaf(
            &mut program,
            "weight1",
            vec![
                Extent::Static(2),
                Extent::Static(1),
                Extent::Static(3),
                Extent::Static(3),
            ],
        );
        let weight2 = leaf(
            &mut program,
            "weight2",
            vec![
                Extent::Static(1),
                Extent::Static(2),
                Extent::Static(2),
                Extent::Static(2),
            ],
        );

        let hidden = conv2d(
            &mut program,
            DType::FLOAT32,
            image,
            (1, 1, 6, 6),
            weight1,
            (2, 1, 3, 3),
            None,
            1,
            1,
        )
        .expect("3x3 kernel fits a 6x6 image at stride 1");
        let out = conv2d(
            &mut program,
            DType::FLOAT32,
            hidden,
            (1, 2, 4, 4),
            weight2,
            (1, 2, 2, 2),
            None,
            1,
            1,
        )
        .expect("2x2 kernel fits a 4x4 image at stride 1");
        let loss = crate::expr::reduce(
            &mut program,
            DType::FLOAT32,
            ScalarOp::Add,
            ReduceInit::Zero,
            out,
            crate::expr::identity(4),
            crate::expr::broadcast(4),
        );

        let differentiated = crate::adjoint::differentiate(&program, loss)
            .expect("stacked conv2d program differentiates");
        let grad_weight1 = differentiated
            .gradient_of_named("weight1")
            .expect("weight1 must receive a gradient through the second layer's own image operand");

        let image_values: Vec<f32> = (0..36).map(|index| (index as f32 - 18.0) / 6.0).collect();
        let weight1_values: Vec<f32> = (0..18).map(|index| (index as f32 - 9.0) / 6.0).collect();
        let weight2_values: Vec<f32> = (0..8).map(|index| (index as f32 - 4.0) / 4.0).collect();

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &differentiated.program,
            &[],
            &[
                ("image", &image_values),
                ("weight1", &weight1_values),
                ("weight2", &weight2_values),
            ],
            &[grad_weight1],
        )
        .expect("adjoint program lowers and evaluates");
        let analytic_weight1 = evaluated
            .get(grad_weight1)
            .expect("grad_weight1 requested")
            .0
            .to_vec();

        assert!(
            analytic_weight1.iter().any(|&value| value != 0.0),
            "weight1's gradient must be nonzero somewhere, got {analytic_weight1:?}"
        );

        let loss_at = |weight1: &[f32]| {
            proxima_tensor::cpu::evaluate_named(
                &program,
                &[],
                &[
                    ("image", &image_values),
                    ("weight1", weight1),
                    ("weight2", &weight2_values),
                ],
                &[loss],
            )
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

        std::eprintln!(
            "stacked conv2d weight1 max relative gradient-check error: {} at index {}",
            worst.0,
            worst.1
        );
        assert!(
            worst.0 < 5e-3,
            "stacked conv2d's first-layer weight gradient disagreed with central difference: {worst:?}"
        );
    }

    /// Isolates whether a stride greater than 1 on BOTH stacked layers
    /// disagrees with central difference -- `stacked_conv2d_gradient_reaches_the_first_layers_weight`
    /// above only exercises stride 1, and `conv2d_gradient_matches_central_difference_on_every_parameter`
    /// only exercises stride 2 on a single (unstacked) layer, so neither
    /// alone would catch a bug specific to the stride-2-through-two-layers
    /// composition.
    #[proxima::test]
    async fn stacked_conv2d_with_stride_two_on_both_layers_matches_central_difference() {
        let mut program = Vec::new();
        let image = leaf(
            &mut program,
            "image",
            vec![
                Extent::Static(1),
                Extent::Static(1),
                Extent::Static(9),
                Extent::Static(9),
            ],
        );
        let weight1 = leaf(
            &mut program,
            "weight1",
            vec![
                Extent::Static(2),
                Extent::Static(1),
                Extent::Static(3),
                Extent::Static(3),
            ],
        );
        let weight2 = leaf(
            &mut program,
            "weight2",
            vec![
                Extent::Static(1),
                Extent::Static(2),
                Extent::Static(2),
                Extent::Static(2),
            ],
        );

        let hidden = conv2d(
            &mut program,
            DType::FLOAT32,
            image,
            (1, 1, 9, 9),
            weight1,
            (2, 1, 3, 3),
            None,
            2,
            2,
        )
        .expect("3x3 kernel fits a 9x9 image at stride 2");
        let out = conv2d(
            &mut program,
            DType::FLOAT32,
            hidden,
            (1, 2, 4, 4),
            weight2,
            (1, 2, 2, 2),
            None,
            2,
            2,
        )
        .expect("2x2 kernel fits a 4x4 image at stride 2");
        let loss = crate::expr::reduce(
            &mut program,
            DType::FLOAT32,
            ScalarOp::Add,
            ReduceInit::Zero,
            out,
            crate::expr::identity(4),
            crate::expr::broadcast(4),
        );

        let differentiated = crate::adjoint::differentiate(&program, loss)
            .expect("stacked conv2d program differentiates");
        let grad_weight1 = differentiated
            .gradient_of_named("weight1")
            .expect("weight1 must receive a gradient through the second layer's own image operand");

        let image_values = pseudo_random(0x1234_5678, 81);
        let weight1_values = pseudo_random(0x9ABC_DEF0, 18);
        let weight2_values = pseudo_random(0x0F0F_0F0F, 8);

        let evaluated = proxima_tensor::cpu::evaluate_named(
            &differentiated.program,
            &[],
            &[
                ("image", &image_values),
                ("weight1", &weight1_values),
                ("weight2", &weight2_values),
            ],
            &[grad_weight1],
        )
        .expect("adjoint program lowers and evaluates");
        let analytic_weight1 = evaluated
            .get(grad_weight1)
            .expect("grad_weight1 requested")
            .0
            .to_vec();

        let loss_at = |weight1: &[f32]| {
            proxima_tensor::cpu::evaluate_named(
                &program,
                &[],
                &[
                    ("image", &image_values),
                    ("weight1", weight1),
                    ("weight2", &weight2_values),
                ],
                &[loss],
            )
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

        std::eprintln!(
            "stacked stride-2 conv2d weight1 max relative gradient-check error: {} at index {}",
            worst.0,
            worst.1
        );
        assert!(
            worst.0 < 5e-3,
            "stacked stride-2 conv2d's first-layer weight gradient disagreed with central difference: {worst:?}"
        );
    }

    #[test]
    fn conv2d_output_shape_matches_the_standard_valid_convolution_formula() {
        assert_eq!(
            conv2d_output_shape((1, 1, 4, 4), (1, 1, 2, 2), 1, 1),
            Some((1, 1, 3, 3))
        );
        assert_eq!(
            conv2d_output_shape((32, 1, 28, 28), (8, 1, 3, 3), 2, 2),
            Some((32, 8, 13, 13))
        );
        assert_eq!(
            conv2d_output_shape((32, 8, 13, 13), (16, 8, 3, 3), 2, 2),
            Some((32, 16, 6, 6))
        );
        assert_eq!(
            conv2d_output_shape((1, 1, 2, 2), (1, 1, 3, 3), 1, 1),
            None,
            "kernel larger than the image cannot fit"
        );
        assert_eq!(
            conv2d_output_shape((1, 2, 4, 4), (1, 3, 2, 2), 1, 1),
            None,
            "weight in-channels must match the image"
        );
    }

    /// Reference `Conv1d` on `[n, ci, l]` / `[co, ci, k]` with ONNX
    /// stride/pad/dilation semantics, written as the definition (a sum over
    /// `ci` and `k` of `signal[o*stride + k*dilation - pad] * weight`).
    #[allow(clippy::too_many_arguments)]
    fn brute_force_conv1d(
        signal: &[f32],
        shape: SignalShape,
        weight: &[f32],
        weight_shape: Kernel1dShape,
        stride: u64,
        padding: u64,
        dilation: u64,
    ) -> Vec<f32> {
        let (batch, in_channels, length) = (shape.0 as i64, shape.1 as usize, shape.2 as i64);
        let (_, out_channels, kernel) = (0, weight_shape.0 as usize, weight_shape.2 as i64);
        let (_, _, out_length) =
            conv1d_output_shape(shape, weight_shape, stride, padding, dilation).unwrap();
        let mut out = Vec::new();
        for sample in 0..batch {
            for co in 0..out_channels {
                for position in 0..out_length as i64 {
                    let mut total = 0.0f32;
                    for ci in 0..in_channels {
                        for tap in 0..kernel {
                            let at =
                                position * stride as i64 + tap * dilation as i64 - padding as i64;
                            if at < 0 || at >= length {
                                continue;
                            }
                            let signal_index = (sample as usize * in_channels + ci)
                                * length as usize
                                + at as usize;
                            let weight_index =
                                (co * in_channels + ci) * kernel as usize + tap as usize;
                            total += signal[signal_index] * weight[weight_index];
                        }
                    }
                    out.push(total);
                }
            }
        }
        out
    }

    #[test]
    fn conv1d_output_shape_follows_the_onnx_rule() {
        assert_eq!(
            conv1d_output_shape((1, 80, 3000), (384, 80, 3), 1, 1, 1),
            Some((1, 384, 3000))
        );
        assert_eq!(
            conv1d_output_shape((1, 384, 3000), (384, 384, 3), 2, 1, 1),
            Some((1, 384, 1500))
        );
        assert_eq!(
            conv1d_output_shape((1, 1, 7), (1, 1, 3), 1, 0, 2),
            Some((1, 1, 3))
        );
        assert_eq!(conv1d_output_shape((1, 1, 2), (1, 1, 3), 1, 0, 1), None);
        assert_eq!(conv1d_output_shape((1, 2, 4), (1, 3, 2), 1, 0, 1), None);
    }

    #[test]
    fn conv1d_forward_matches_brute_force_across_stride_padding_and_dilation() {
        let cases: [(u64, u64, u64); 5] = [(1, 0, 1), (1, 1, 1), (2, 1, 1), (1, 2, 2), (3, 2, 2)];
        for (stride, padding, dilation) in cases {
            let shape: SignalShape = (2, 2, 9);
            let weight_shape: Kernel1dShape = (3, 2, 3);
            let signal_values = pseudo_random(0xA5A5_1234, 2 * 2 * 9);
            let weight_values = pseudo_random(0x1357_9BDF, 3 * 2 * 3);
            let bias_values = [0.25f32, -0.5, 0.75];
            let mut program = Vec::new();
            let signal = leaf(&mut program, "signal", static_extents(&[2, 2, 9]));
            let weight = leaf(&mut program, "weight", static_extents(&[3, 2, 3]));
            let bias = leaf(&mut program, "bias", static_extents(&[3]));
            let out = conv1d(
                &mut program,
                DType::FLOAT32,
                signal,
                shape,
                weight,
                weight_shape,
                Some(bias),
                stride,
                padding,
                dilation,
            )
            .expect("kernel fits");
            let evaluated = proxima_tensor::cpu::evaluate_named(
                &program,
                &[],
                &[
                    ("signal", &signal_values),
                    ("weight", &weight_values),
                    ("bias", &bias_values),
                ],
                &[out],
            )
            .expect("conv1d program evaluates");
            let (got, _) = evaluated.get(out).expect("output requested");
            let mut want = brute_force_conv1d(
                &signal_values,
                shape,
                &weight_values,
                weight_shape,
                stride,
                padding,
                dilation,
            );
            let out_length = want.len() / (2 * 3);
            for (index, value) in want.iter_mut().enumerate() {
                *value += bias_values[(index / out_length) % 3];
            }
            assert_eq!(
                got.len(),
                want.len(),
                "stride {stride} pad {padding} dil {dilation}"
            );
            for (index, (&have, &expect)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (have - expect).abs() < 1e-4,
                    "stride {stride} pad {padding} dil {dilation} element {index}: {have} vs {expect}"
                );
            }
        }
    }

    /// Whisper's encoder stem (80 mel channels in, 384 out, kernel 3, padding
    /// 1, length 3000): the program must contain no node with two axes as
    /// long as the output, i.e. no `source x out` mask or window tensor.
    #[test]
    fn conv1d_at_whisper_stem_shape_builds_no_source_by_output_tensor() {
        for stride in [1u64, 2] {
            let mut program = Vec::new();
            let signal = leaf(&mut program, "signal", static_extents(&[1, 80, 3000]));
            let weight = leaf(&mut program, "weight", static_extents(&[384, 80, 3]));
            let dtype = program[signal.0 as usize].dtype();
            conv1d(
                &mut program,
                dtype,
                signal,
                (1, 80, 3000),
                weight,
                (384, 80, 3),
                None,
                stride,
                1,
                1,
            )
            .expect("kernel 3 fits a 3000-long signal at padding 1");

            let shapes = proxima_tensor::shape::infer(&program, &[])
                .expect("whisper-stem conv1d program infers");
            let out_length = 3000 / stride;
            let widest = (0..program.len())
                .map(|index| {
                    let shape = shapes.of(NodeId(index as u32));
                    let long_axes = shape.iter().filter(|&&extent| extent >= out_length).count();
                    (long_axes, shape.to_vec())
                })
                .max_by_key(|(long_axes, _)| *long_axes)
                .expect("program has nodes");
            assert!(
                widest.0 <= 1,
                "stride {stride}: node of shape {:?} has {} axes of extent >= {out_length}",
                widest.1,
                widest.0
            );
        }
    }

    fn static_extents(dims: &[u32]) -> Vec<Extent> {
        dims.iter().map(|&dim| Extent::Static(dim)).collect()
    }

    /// `sum(out * out)` over a rank-`rank` result: the spec's gradient-check
    /// loss, `|f| <= 10 * max|g|` by construction on small shapes.
    fn sum_of_squares(program: &mut Vec<Op>, out: NodeId, rank: u16) -> NodeId {
        let squared = crate::expr::binary(
            program,
            DType::FLOAT32,
            ScalarOp::Multiply,
            (out, crate::expr::identity(rank)),
            (out, crate::expr::identity(rank)),
        );
        crate::expr::reduce(
            program,
            DType::FLOAT32,
            ScalarOp::Add,
            ReduceInit::Zero,
            squared,
            crate::expr::identity(rank),
            crate::expr::broadcast(rank),
        )
    }

    /// Central-difference check on every element of every named input, the
    /// spec's metric `max|g_ad - g_fd| / max|g_fd|` per input at `h = 5e-3`
    /// (SPEC "Gradient tolerance (R10)", DERIVED ceiling 5e-4). Returns the
    /// `(name, elements compared, error)` rows.
    fn central_difference_errors(
        program: &[Op],
        loss: NodeId,
        inputs: &[(&str, Vec<f32>)],
    ) -> Vec<(String, usize, f32)> {
        const STEP: f32 = 5e-3;
        let differentiated =
            crate::adjoint::differentiate(program, loss).expect("conv1d program differentiates");
        let borrowed = |values: &[(&str, Vec<f32>)]| -> Vec<(String, Vec<f32>)> {
            values
                .iter()
                .map(|(name, v)| ((*name).into(), v.clone()))
                .collect()
        };
        let loss_at = |values: &[(&str, Vec<f32>)]| {
            let owned = borrowed(values);
            let refs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(name, v)| (name.as_str(), v.as_slice()))
                .collect();
            proxima_tensor::cpu::evaluate_named(program, &[], &refs, &[loss])
                .expect("forward evaluates")
                .get(loss)
                .expect("loss requested")
                .0[0]
        };
        let mut rows = Vec::new();
        for (slot, (name, values)) in inputs.iter().enumerate() {
            let grad = differentiated
                .gradient_of_named(name)
                .unwrap_or_else(|| panic!("{name} feeds the loss and has a gradient"));
            let owned = borrowed(inputs);
            let refs: Vec<(&str, &[f32])> = owned
                .iter()
                .map(|(label, v)| (label.as_str(), v.as_slice()))
                .collect();
            let evaluated =
                proxima_tensor::cpu::evaluate_named(&differentiated.program, &[], &refs, &[grad])
                    .expect("adjoint evaluates");
            let analytic = evaluated.get(grad).expect("gradient requested").0.to_vec();
            assert_eq!(analytic.len(), values.len(), "{name} gradient size");
            let mut probe: Vec<(&str, Vec<f32>)> = inputs.to_vec();
            let (mut worst_difference, mut largest) = (0.0f32, 0.0f32);
            for index in 0..values.len() {
                probe[slot].1[index] = values[index] + STEP;
                let plus = loss_at(&probe);
                probe[slot].1[index] = values[index] - STEP;
                let minus = loss_at(&probe);
                probe[slot].1[index] = values[index];
                let numeric = (plus - minus) / (2.0 * STEP);
                worst_difference = worst_difference.max((analytic[index] - numeric).abs());
                largest = largest.max(numeric.abs());
            }
            assert!(
                largest > 0.0,
                "{name}: central-difference gradient is identically zero"
            );
            rows.push(((*name).into(), values.len(), worst_difference / largest));
        }
        rows
    }

    fn assert_within_spec_bound(label: &str, rows: &[(String, usize, f32)]) {
        for (name, count, error) in rows {
            std::eprintln!("{label} wrt={name} n={count} max|g_ad-g_fd|/max|g_fd|={error:e}");
            assert!(*count > 0, "{label} {name}: compared zero elements");
            assert!(
                *error <= 5e-4,
                "{label} {name}: error {error} exceeds the derived 5e-4"
            );
        }
    }

    #[proxima::test]
    async fn conv1d_gradient_matches_central_difference_stride_1() {
        let mut program = Vec::new();
        let signal = leaf(&mut program, "signal", static_extents(&[1, 2, 7]));
        let weight = leaf(&mut program, "weight", static_extents(&[3, 2, 3]));
        let bias = leaf(&mut program, "bias", static_extents(&[3]));
        let out = conv1d(
            &mut program,
            DType::FLOAT32,
            signal,
            (1, 2, 7),
            weight,
            (3, 2, 3),
            Some(bias),
            1,
            1,
            1,
        )
        .expect("kernel fits");
        let loss = sum_of_squares(&mut program, out, 3);
        let inputs = [
            ("signal", pseudo_random(0x9E37_79B9, 14)),
            ("weight", pseudo_random(0x8542_D2C3, 18)),
            ("bias", vec![0.1f32, -0.2, 0.3]),
        ];
        let rows = central_difference_errors(&program, loss, &inputs);
        assert_eq!(rows.len(), 3);
        assert_within_spec_bound("conv1d stride 1", &rows);
    }

    /// A stride-1 padded layer feeds a stride-2
    /// layer, so the second layer's input is the first layer's output and the
    /// gradient must keep flowing into `weight1` through the affine window
    /// read and the scatter pad (a gather window would stop there).
    #[proxima::test]
    async fn conv1d_gradient_matches_central_difference_stride_2_stacked() {
        let mut program = Vec::new();
        let signal = leaf(&mut program, "signal", static_extents(&[1, 2, 8]));
        let weight1 = leaf(&mut program, "weight1", static_extents(&[3, 2, 3]));
        let bias1 = leaf(&mut program, "bias1", static_extents(&[3]));
        let weight2 = leaf(&mut program, "weight2", static_extents(&[2, 3, 3]));
        let bias2 = leaf(&mut program, "bias2", static_extents(&[2]));
        let hidden = conv1d(
            &mut program,
            DType::FLOAT32,
            signal,
            (1, 2, 8),
            weight1,
            (3, 2, 3),
            Some(bias1),
            1,
            1,
            1,
        )
        .expect("first layer fits");
        let out = conv1d(
            &mut program,
            DType::FLOAT32,
            hidden,
            (1, 3, 8),
            weight2,
            (2, 3, 3),
            Some(bias2),
            2,
            1,
            1,
        )
        .expect("second layer fits");
        let loss = sum_of_squares(&mut program, out, 3);
        let inputs = [
            ("signal", pseudo_random(0x1111_2222, 16)),
            ("weight1", pseudo_random(0x3333_4444, 18)),
            ("bias1", vec![0.05f32, -0.1, 0.15]),
            ("weight2", pseudo_random(0x5555_6666, 18)),
            ("bias2", vec![-0.2f32, 0.25]),
        ];
        let rows = central_difference_errors(&program, loss, &inputs);
        assert_eq!(rows.len(), 5);
        assert_within_spec_bound("conv1d stride 2 stacked", &rows);
    }

    #[proxima::test]
    async fn conv1d_padded_dilated_gradient_agrees_with_central_difference() {
        let mut program = Vec::new();
        let signal = leaf(&mut program, "signal", static_extents(&[1, 2, 9]));
        let weight = leaf(&mut program, "weight", static_extents(&[2, 2, 3]));
        let out = conv1d(
            &mut program,
            DType::FLOAT32,
            signal,
            (1, 2, 9),
            weight,
            (2, 2, 3),
            None,
            2,
            2,
            2,
        )
        .expect("kernel fits");
        let loss = sum_of_squares(&mut program, out, 3);
        let inputs = [
            ("signal", pseudo_random(0x7777_8888, 18)),
            ("weight", pseudo_random(0x9999_AAAA, 12)),
        ];
        let rows = central_difference_errors(&program, loss, &inputs);
        assert_within_spec_bound("conv1d padded dilated", &rows);
    }
}
