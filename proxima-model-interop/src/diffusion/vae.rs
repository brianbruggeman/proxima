//! A convolutional autoencoder decoder as one [`Op`] program: 3x3 "same"
//! convolutions, 32-group normalisation, nearest-x2 upsampling and one
//! single-head non-causal attention at the latent resolution.

use proxima_numeric::PackedRows;
use proxima_tensor::TensorError;
use proxima_tensor::op::NodeId;
use proxima_tensor::spec::{
    group_norm, nearest_upsample_x2, non_causal_attention_heads, scalar_constant_like, silu,
    upsample_x2_index, window_crop_index, window_pad_index, window_pad_mask, window_pad_rows,
    window_same, window_same_one_reduce, window_taps_index,
};

use super::Graph;
use super::dit::inverse_root_constant;

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
    /// The convolution kernel's square extent, read from a conv weight's
    /// `[out, in, kh, kw]` shape ([`super::Checkpoint::window_extent`]).
    pub window: u32,
}

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
    let eps_value = graph
        .settings()
        .group_eps()
        .map_err(|_| TensorError::NotLowerable {
            node: x,
            reason: "group_eps is not a number",
        })?;
    let eps = scalar_constant_like(&mut graph.program, eps_value, x)?;
    let normed = group_norm(&mut graph.program, x, weight, bias, rows * group_width, eps)?;
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
    grid: (u32, u32),
    channels: (u32, u32),
    groups: (u32, u32),
    config: &VaeConfig,
) -> Result<NodeId, TensorError> {
    let window = config.window;
    let (height, width) = grid;
    let (in_channels, out_channels) = channels;
    let (in_groups, out_groups) = groups;
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
        window_pad_index(height, width, window),
    );
    let mask: Vec<_> = window_pad_mask(height, width, window)
        .into_iter()
        .map(|value| value as _)
        .collect();
    let packed_mask = PackedRows::from_values(graph.activation_format(), &mask);
    let pad_mask = graph.host(
        &format!("pad.{height}x{width}.mask"),
        &[window_pad_rows(height, width, window)],
        packed_mask,
    );
    let crop_index = graph.index_leaf(
        &format!("crop.{height}x{width}.index"),
        window_crop_index(height, width, window),
    );
    let convolved = match config.conv_taps {
        ConvTaps::Nine => {
            let mut weights = Vec::with_capacity((window * window) as usize);
            for tap_row in 0..window {
                for tap_column in 0..window {
                    weights.push(graph.weight_tap(
                        &format!("{tensor}.weight"),
                        &weight_shape,
                        (tap_row, tap_column),
                        in_groups,
                        in_channels / in_groups,
                    ));
                }
            }
            window_same(
                &mut graph.program,
                x,
                &weights,
                pad_index,
                pad_mask,
                crop_index,
                height,
                width,
                window,
            )?
        }
        ConvTaps::Three => {
            // The checkpoint's `[out, window, window, in]` viewed as
            // `[out_g, out_c, taps, in_g, in_c]`: the tap pair flattened to one
            // block axis, which is the same element order (a reshape).
            let taps_shape = [
                out_groups,
                out_channels / out_groups,
                window * window,
                in_groups,
                in_channels / in_groups,
            ];
            let weight = graph.weight_taps(&format!("{tensor}.weight"), &taps_shape);
            let taps_index = graph.index_leaf(
                &format!("taps.{height}x{width}.index"),
                window_taps_index(height, width, window),
            );
            window_same_one_reduce(
                &mut graph.program,
                x,
                weight,
                pad_index,
                pad_mask,
                crop_index,
                taps_index,
                window,
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

/// One "same" windowed convolution as a program fragment, for a caller that
/// composes a decoder by hand (the per-shape bench) rather than through
/// [`build_vae_decode`]; the window extent and tap form come from `config`.
///
/// `grid` is `(height, width)`, `channels` is `(in, out)`, and `groups` is
/// `(in, out)` channel-group counts: the decoder's checkpoint was trained with
/// `groups` on the input side, and the output side is `1` for the final
/// carve and `groups` otherwise.
///
/// # Errors
///
/// [`TensorError`] when a notation or extent fails to build.
pub fn build_window_same(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    grid: (u32, u32),
    channels: (u32, u32),
    groups: (u32, u32),
    config: &VaeConfig,
) -> Result<NodeId, TensorError> {
    conv(graph, x, tensor, grid, channels, groups, config)
}

fn resnet(
    graph: &mut Graph,
    x: NodeId,
    prefix: &str,
    in_channels: u32,
    out_channels: u32,
    grid: (u32, u32),
    config: &VaeConfig,
) -> Result<NodeId, TensorError> {
    let rows = grid.0 * grid.1;
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
        grid,
        (in_channels, out_channels),
        (config.groups, config.groups),
        config,
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
        grid,
        (out_channels, out_channels),
        (config.groups, config.groups),
        config,
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
    grid: (u32, u32),
    config: &VaeConfig,
    trace: &mut Vec<(&'static str, NodeId)>,
) -> Result<NodeId, TensorError> {
    let (height, width) = grid;
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
        (height * 2, width * 2),
        (channels, channels),
        (config.groups, config.groups),
        config,
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
    let inv_sqrt = inverse_root_constant(graph, channels)?;
    let half = config.groups / 2;
    let q_even = group_range(graph, q, 0, half)?;
    let q_odd = group_range(graph, q, half, half)?;
    let k_even = group_range(graph, k, 0, half)?;
    let k_odd = group_range(graph, k, half, half)?;
    let attended = non_causal_attention_heads(
        &mut graph.program,
        &[(q_even, k_even), (q_odd, k_odd)],
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
        (height, width),
        (in_channels, first),
        (config.groups, config.groups),
        config,
    )?;
    trace.push(("conv_in", hidden));
    let rows = height * width;
    hidden = resnet(
        graph,
        hidden,
        "decoder.mid.block_1",
        first,
        first,
        (height, width),
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
        (height, width),
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
                (height, width),
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
                (height, width),
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
        (height, width),
        (channels, config.out_channels),
        (config.groups, 1),
        config,
    )?;
    trace.push(("image", image));
    Ok(VaeOutput { image, trace })
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
