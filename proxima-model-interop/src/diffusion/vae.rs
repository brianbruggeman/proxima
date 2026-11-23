//! A convolutional autoencoder decoder as one [`Op`] program: 3x3 "same"
//! convolutions, 32-group normalisation, nearest-x2 upsampling and one
//! single-head non-causal attention at the latent resolution.

use proxima_tensor::TensorError;
use proxima_tensor::op::NodeId;
use proxima_tensor::spec::{
    conv3x3_crop_index, conv3x3_pad_index, conv3x3_pad_mask, conv3x3_pad_rows, conv3x3_same,
    conv3x3_same_three, conv3x3_taps_index, group_norm, nearest_upsample_x2,
    non_causal_attention_heads, scalar_constant_like, silu, unit_constant, upsample_x2_index,
};

use super::Graph;

/// How a 3x3 convolution's nine kernel positions reach the reduce.
///
/// [`Self::Nine`] is the incumbent: nine `[out, in]` leaves, one reduce each,
/// summed by eight adds. [`Self::Three`] is one `[out_g, out_c, 9, in_g, in_c]`
/// leaf and one reduce contracting `(t, in_g, in_c)` over one gathered operand
/// of nine stencil blocks, which gives the planner nothing to fuse into an
/// eight-operand epilogue.
///
/// The two contract the same nine products: they agree to 1e-5 against a
/// hand-written direct loop on the CPU (`spec::tests::conv_direct_loop`) and
/// decode to the same image on Metal. `Three` is the default because it stages
/// the activation with the tap axis *inner*, which makes the contraction
/// contiguous and keeps the reduce on a GEMM class kernel; `Nine` stays
/// reachable for A/B.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvTaps {
    /// Nine per-tap leaves and nine reduces (the incumbent, kept reachable).
    Nine,
    /// One kernel leaf and one reduce over a tap-inner staged activation.
    #[default]
    Three,
}

impl ConvTaps {
    /// The setting's own spelling; `None` for anything else, so a caller can
    /// fall back to the default rather than guess.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "nine" => Some(Self::Nine),
            "three" => Some(Self::Three),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nine => "nine",
            Self::Three => "three",
        }
    }
}

/// The decoder's dimensions, read from the checkpoint's `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VaeConfig {
    /// Latent channels after the 2x2 patch merge (`latent_channels / patch^2`).
    pub channels: u32,
    /// Channel widths of the four decoder resolutions, coarsest last.
    pub block_channels: [u32; 4],
    pub layers_per_block: u32,
    pub groups: u32,
    pub out_channels: u32,
    /// The latent grid the decoder starts from (already unpatchified).
    pub height: u32,
    pub width: u32,
    /// Which 3x3 lowering to emit ([`ConvTaps`]).
    pub conv_taps: ConvTaps,
}

const GROUP_EPS: f32 = 1e-6;

fn norm(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    channels: u32,
    groups: u32,
    rows: u32,
) -> Result<NodeId, TensorError> {
    let start = graph.program.len();
    let group_width = channels / groups;
    let weight = graph.weight(&format!("{tensor}.weight"), &[groups, group_width], None);
    let bias = graph.weight(&format!("{tensor}.bias"), &[groups, group_width], None);
    let normed = group_norm(
        &mut graph.program,
        x,
        weight,
        bias,
        rows * group_width,
        GROUP_EPS,
    )?;
    graph.record_step(format!("group_norm:{tensor}"), start);
    Ok(normed)
}

fn activated(graph: &mut Graph, x: NodeId) -> Result<NodeId, TensorError> {
    let start = graph.program.len();
    let one = scalar_constant_like(&mut graph.program, 1.0, x)?;
    let activated = silu(&mut graph.program, x, one, "rgc->rgc")?;
    graph.record_step(String::from("silu"), start);
    Ok(activated)
}

/// One 3x3 "same" convolution: the weight is the checkpoint's
/// `[out, kernel, kernel, in]` permutation (the loader reorders it), re-viewed
/// as `[out_group, out_width, 3, 3, in_group, in_width]`.
fn conv(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    height: u32,
    width: u32,
    in_channels: u32,
    out_channels: u32,
    in_groups: u32,
    out_groups: u32,
    tap_form: ConvTaps,
) -> Result<NodeId, TensorError> {
    let start = graph.program.len();
    let weight_shape = [
        out_groups,
        out_channels / out_groups,
        in_groups,
        in_channels / in_groups,
    ];
    let bias = graph.weight(
        &format!("{tensor}.bias"),
        &[out_groups, out_channels / out_groups],
        None,
    );
    let pad_index = graph.index_leaf(
        &format!("pad.{height}x{width}.index"),
        conv3x3_pad_index(height, width),
    );
    let pad_mask = graph.host(
        &format!("pad.{height}x{width}.mask"),
        &[conv3x3_pad_rows(height, width)],
        conv3x3_pad_mask(height, width)
            .into_iter()
            .map(|value| value as f32)
            .collect(),
    );
    let crop_index = graph.index_leaf(
        &format!("crop.{height}x{width}.index"),
        conv3x3_crop_index(height, width),
    );
    let convolved = match tap_form {
        ConvTaps::Nine => {
            let mut weights = [NodeId(0); 9];
            for tap_row in 0..3u32 {
                for tap_column in 0..3u32 {
                    let tap = (tap_row * 3 + tap_column) as usize;
                    weights[tap] = graph.weight_tap(
                        &format!("{tensor}.weight"),
                        &weight_shape,
                        (tap_row, tap_column),
                        in_groups,
                        in_channels / in_groups,
                    );
                }
            }
            conv3x3_same(
                &mut graph.program,
                x,
                &weights,
                pad_index,
                pad_mask,
                crop_index,
                height,
                width,
            )?
        }
        ConvTaps::Three => {
            // The checkpoint's `[out, 3, 3, in]` viewed as
            // `[out_g, out_c, 9, in_g, in_c]`: the tap pair flattened to one
            // block axis, which is the same element order (a reshape).
            let taps_shape = [
                out_groups,
                out_channels / out_groups,
                9,
                in_groups,
                in_channels / in_groups,
            ];
            let weight = graph.weight_taps(&format!("{tensor}.weight"), &taps_shape);
            let taps_index = graph.index_leaf(
                &format!("taps.{height}x{width}.index"),
                conv3x3_taps_index(height, width),
            );
            conv3x3_same_three(
                &mut graph.program,
                x,
                weight,
                pad_index,
                pad_mask,
                crop_index,
                taps_index,
            )?
        }
    };
    let program = &mut graph.program;
    let biased = proxima_tensor::spec::elementwise_like(
        program,
        convolved,
        proxima_tensor::ScalarOp::Add,
        &[(convolved, "rgc->rgc"), (bias, "gc->rgc")],
    )?;
    graph.record_step(
        format!("conv:{tensor}:{height}x{width}:{in_channels}->{out_channels}"),
        start,
    );
    Ok(biased)
}

/// One `1x1` convolution: a channel linear over the token axis.
fn pointwise(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    in_channels: u32,
    out_channels: u32,
    groups: u32,
) -> Result<NodeId, TensorError> {
    let start = graph.program.len();
    let weight = graph.weight(
        &format!("{tensor}.weight"),
        &[groups, out_channels / groups, groups, in_channels / groups],
        None,
    );
    let bias = graph.weight(
        &format!("{tensor}.bias"),
        &[groups, out_channels / groups],
        None,
    );
    let program = &mut graph.program;
    let product = proxima_tensor::spec::elementwise_like(
        program,
        x,
        proxima_tensor::ScalarOp::Multiply,
        &[(x, "r,i,j->rgcij"), (weight, "g,c,i,j->rgcij")],
    )?;
    let projected = proxima_tensor::spec::reduce_like(
        program,
        proxima_tensor::ScalarOp::Add,
        proxima_tensor::ReduceInit::Zero,
        product,
        "rgcij->rgcij",
        "rgc->rgcij",
    )?;
    let biased = proxima_tensor::spec::elementwise_like(
        program,
        projected,
        proxima_tensor::ScalarOp::Add,
        &[(projected, "rgc->rgc"), (bias, "gc->rgc")],
    )?;
    graph.record_step(
        format!("pointwise:{tensor}:{in_channels}->{out_channels}"),
        start,
    );
    Ok(biased)
}

/// One 3x3 "same" convolution as a program fragment, for a caller that composes
/// a decoder by hand (the per-shape bench) rather than through
/// [`build_vae_decode`].
///
/// `groups` is the channel-group count the decoder's checkpoint was trained
/// with; `out_groups` is `1` for the final `3`-channel carve and `groups`
/// otherwise.
///
/// # Errors
///
/// [`TensorError`] when a notation or extent fails to build.
#[allow(clippy::too_many_arguments)]
pub fn build_conv3x3(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    height: u32,
    width: u32,
    in_channels: u32,
    out_channels: u32,
    groups: u32,
    out_groups: u32,
    tap_form: ConvTaps,
) -> Result<NodeId, TensorError> {
    conv(
        graph,
        x,
        tensor,
        height,
        width,
        in_channels,
        out_channels,
        groups,
        out_groups,
        tap_form,
    )
}

fn resnet(
    graph: &mut Graph,
    x: NodeId,
    prefix: &str,
    in_channels: u32,
    out_channels: u32,
    height: u32,
    width: u32,
    config: &VaeConfig,
) -> Result<NodeId, TensorError> {
    let rows = height * width;
    let first = norm(
        graph,
        x,
        &format!("{prefix}.norm1"),
        in_channels,
        config.groups,
        rows,
    )?;
    let first = activated(graph, first)?;
    let first = conv(
        graph,
        first,
        &format!("{prefix}.conv1"),
        height,
        width,
        in_channels,
        out_channels,
        config.groups,
        config.groups,
        config.conv_taps,
    )?;
    let second = norm(
        graph,
        first,
        &format!("{prefix}.norm2"),
        out_channels,
        config.groups,
        rows,
    )?;
    let second = activated(graph, second)?;
    let second = conv(
        graph,
        second,
        &format!("{prefix}.conv2"),
        height,
        width,
        out_channels,
        out_channels,
        config.groups,
        config.groups,
        config.conv_taps,
    )?;
    let residual = if in_channels == out_channels {
        x
    } else {
        pointwise(
            graph,
            x,
            &format!("{prefix}.nin_shortcut"),
            in_channels,
            out_channels,
            config.groups,
        )?
    };
    let start = graph.program.len();
    let program = &mut graph.program;
    let summed = proxima_tensor::spec::elementwise_like(
        program,
        second,
        proxima_tensor::ScalarOp::Add,
        &[(second, "rgc->rgc"), (residual, "rgc->rgc")],
    )?;
    graph.record_step(format!("resnet_add:{prefix}"), start);
    Ok(summed)
}

fn upsample(
    graph: &mut Graph,
    x: NodeId,
    prefix: &str,
    channels: u32,
    height: u32,
    width: u32,
    config: &VaeConfig,
    trace: &mut Vec<(&'static str, NodeId)>,
) -> Result<NodeId, TensorError> {
    let start = graph.program.len();
    let index = graph.index_leaf(
        &format!("upsample.{height}x{width}.index"),
        upsample_x2_index(height, width),
    );
    let program = &mut graph.program;
    let grown = nearest_upsample_x2(program, x, index)?;
    graph.record_step(format!("upsample:{prefix}"), start);
    trace.push(("upsample", grown));
    conv(
        graph,
        grown,
        &format!("{prefix}.conv"),
        height * 2,
        width * 2,
        channels,
        channels,
        config.groups,
        config.groups,
        config.conv_taps,
    )
}

/// The mid-block attention: one head of width `channels`, no mask, no rope.
fn attention(
    graph: &mut Graph,
    x: NodeId,
    prefix: &str,
    channels: u32,
    rows: u32,
    config: &VaeConfig,
) -> Result<NodeId, TensorError> {
    let normed = norm(
        graph,
        x,
        &format!("{prefix}.norm"),
        channels,
        config.groups,
        rows,
    )?;
    let q = pointwise(
        graph,
        normed,
        &format!("{prefix}.q"),
        channels,
        channels,
        config.groups,
    )?;
    let k = pointwise(
        graph,
        normed,
        &format!("{prefix}.k"),
        channels,
        channels,
        config.groups,
    )?;
    let v = pointwise(
        graph,
        normed,
        &format!("{prefix}.v"),
        channels,
        channels,
        config.groups,
    )?;
    let start = graph.program.len();
    let activation = graph.activation();
    let inv_sqrt = unit_constant(
        &mut graph.program,
        1.0 / (channels as f32).sqrt(),
        activation,
    );
    let half = config.groups / 2;
    let q_even = group_range(graph, q, 0, half)?;
    let q_odd = group_range(graph, q, half, half)?;
    let k_even = group_range(graph, k, 0, half)?;
    let k_odd = group_range(graph, k, half, half)?;
    let attended = non_causal_attention_heads(
        &mut graph.program,
        q_even,
        q_odd,
        k_even,
        k_odd,
        v,
        inv_sqrt,
        "",
        "gc",
        "gc",
    )?;
    graph.record_step(format!("attention:{prefix}:core"), start);
    let projected = pointwise(
        graph,
        attended,
        &format!("{prefix}.proj_out"),
        channels,
        channels,
        config.groups,
    )?;
    let start = graph.program.len();
    let summed = proxima_tensor::spec::elementwise_like(
        &mut graph.program,
        projected,
        proxima_tensor::ScalarOp::Add,
        &[(projected, "rgc->rgc"), (x, "rgc->rgc")],
    )?;
    graph.record_step(format!("attention:{prefix}:residual"), start);
    Ok(summed)
}

/// The decoder's output plus the node of every stage a parity check reads.
#[derive(Debug, Clone)]
pub struct VaeOutput {
    pub image: NodeId,
    pub trace: Vec<(&'static str, NodeId)>,
}

/// Builds the decoder from an unpatchified, de-normalised latent
/// `[height * width, groups, channels / groups]` to the image
/// `[height_out * width_out, groups_out, out / groups_out]`
/// (image channels last, token-major), recording the stage nodes a parity
/// check compares against the reference's own captures.
///
/// # Errors
///
/// [`TensorError`] when a notation or extent fails to build.
pub fn build_vae_decode(
    graph: &mut Graph,
    config: &VaeConfig,
    latent: NodeId,
) -> Result<VaeOutput, TensorError> {
    let in_channels = config.channels;
    let first = config.block_channels[3];
    let mut height = config.height;
    let mut width = config.width;
    let mut channels = first;
    let mut trace = Vec::new();
    let quantised = pointwise(
        graph,
        latent,
        "decoder.post_quant_conv",
        in_channels,
        in_channels,
        config.groups,
    )?;
    trace.push(("z", quantised));
    let mut hidden = conv(
        graph,
        quantised,
        "decoder.conv_in",
        height,
        width,
        in_channels,
        first,
        config.groups,
        config.groups,
        config.conv_taps,
    )?;
    trace.push(("conv_in", hidden));
    let rows = height * width;
    hidden = resnet(
        graph,
        hidden,
        "decoder.mid.block_1",
        first,
        first,
        height,
        width,
        config,
    )?;
    trace.push(("mid1", hidden));
    hidden = attention(graph, hidden, "decoder.mid.attn_1", first, rows, config)?;
    trace.push(("attn", hidden));
    hidden = resnet(
        graph,
        hidden,
        "decoder.mid.block_2",
        first,
        first,
        height,
        width,
        config,
    )?;
    trace.push(("mid2", hidden));
    for level in (0..4).rev() {
        let out_channels = config.block_channels[level];
        for layer in 0..=config.layers_per_block {
            let prefix = format!("decoder.up.{level}.block.{layer}");
            hidden = resnet(
                graph,
                hidden,
                &prefix,
                channels,
                out_channels,
                height,
                width,
                config,
            )?;
            channels = out_channels;
        }
        let name = if level == 3 {
            "up3"
        } else if level == 2 {
            "up2"
        } else if level == 1 {
            "up1"
        } else {
            "up0"
        };
        trace.push((name, hidden));
        if level != 0 {
            hidden = upsample(
                graph,
                hidden,
                &format!("decoder.up.{level}.upsample"),
                channels,
                height,
                width,
                config,
                &mut trace,
            )?;
            height *= 2;
            width *= 2;
            let name = if level == 3 {
                "up3.up"
            } else if level == 2 {
                "up2.up"
            } else {
                "up1.up"
            };
            trace.push((name, hidden));
        }
    }
    let rows = height * width;
    let normed = norm(
        graph,
        hidden,
        "decoder.norm_out",
        channels,
        config.groups,
        rows,
    )?;
    trace.push(("norm_out", normed));
    let normed = activated(graph, normed)?;
    let image = conv(
        graph,
        normed,
        "decoder.conv_out",
        height,
        width,
        channels,
        config.out_channels,
        config.groups,
        1,
        config.conv_taps,
    )?;
    trace.push(("image", image));
    Ok(VaeOutput { image, trace })
}

/// The checkpoint's latent statistics: `mean` and the standard deviation
/// `sqrt(running_var + eps)` of each latent channel, both from the VAE's own
/// batch-norm buffers.
#[must_use]
pub fn latent_statistics(
    running_mean: &[f32],
    running_var: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let deviation = running_var
        .iter()
        .map(|variance| (variance + eps).sqrt())
        .collect();
    (running_mean.to_vec(), deviation)
}

/// `latent * deviation + mean`: the de-normalisation the decoder's input
/// expects, per channel and broadcast over tokens.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn latent_unnormalise(
    graph: &mut Graph,
    latent: NodeId,
    mean: NodeId,
    deviation: NodeId,
) -> Result<NodeId, TensorError> {
    let program = &mut graph.program;
    let scaled = proxima_tensor::spec::elementwise_like(
        program,
        latent,
        proxima_tensor::ScalarOp::Multiply,
        &[(latent, "rgc->rgc"), (deviation, "gc->rgc")],
    )?;
    proxima_tensor::spec::elementwise_like(
        program,
        scaled,
        proxima_tensor::ScalarOp::Add,
        &[(scaled, "rgc->rgc"), (mean, "gc->rgc")],
    )
}

/// The latent viewed as the decoder's `[rows, groups, channels / groups]`:
/// the model's `[rows, channels]` with the channel axis split at `groups`.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn latent_grouped(
    graph: &mut Graph,
    latent: NodeId,
    _groups: u32,
    group_width: u32,
) -> Result<NodeId, TensorError> {
    proxima_tensor::spec::elementwise_like(
        &mut graph.program,
        latent,
        proxima_tensor::ScalarOp::Identity,
        &[(latent, format!("{group_width}*g+c->rgc").as_str())],
    )
}

/// Group axis `[start, start + count)` of a `[rows, groups, width]` node.
fn group_range(
    graph: &mut Graph,
    x: NodeId,
    start: u32,
    count: u32,
) -> Result<NodeId, TensorError> {
    proxima_tensor::spec::elementwise_like(
        &mut graph.program,
        x,
        proxima_tensor::ScalarOp::Identity,
        &[(x, format!("r,g+{start}@{count},c->rgc").as_str())],
    )
}
