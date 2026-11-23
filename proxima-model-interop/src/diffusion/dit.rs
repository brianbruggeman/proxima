//! A flow-matching transformer backbone as one [`Op`] program: linear prelude,
//! N double-stream blocks with joint attention, M single-stream blocks, and the
//! final adaLN layer. Every dimension is a config value and every tensor name
//! is the checkpoint's own (the diffusers layout its safetensors header
//! declares).

use proxima_tensor::op::{NodeId, ReduceInit, ScalarOp};
use proxima_tensor::spec::{
    RopePairing, scalar_constant_like, concat_rows, count_constant, elementwise_like, fused_rope_pair,
    layer_norm_no_affine, linear, matvec, modulate, non_causal_attention_heads, reduce_like,
    rmsnorm_per_head, silu, slice_columns, slice_range, unit_constant,
};
use proxima_tensor::TensorError;

use super::Graph;

/// The backbone's dimensions, read from the checkpoint's `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DitConfig {
    pub in_channels: u32,
    pub hidden: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub joint_attention_dim: u32,
    pub double_blocks: u32,
    pub single_blocks: u32,
    pub timestep_channels: u32,
    pub rope_theta: u32,
}

impl DitConfig {
    /// The SwiGLU hidden width (`mlp_ratio 3`).
    #[must_use]
    pub const fn mlp_hidden(&self) -> u32 {
        self.hidden * 3
    }

    /// `3 * hidden + 2 * mlp_hidden`, a single block's fused q/k/v/mlp width.
    #[must_use]
    pub const fn single_linear1_width(&self) -> u32 {
        self.hidden * 3 + self.mlp_hidden() * 2
    }

    const fn attention_width(&self) -> u32 {
        self.heads * self.head_dim
    }
}

/// The activations one forward call binds.
#[derive(Debug, Clone, Copy)]
pub struct DitInputs {
    /// `[image_tokens, in_channels]`, token-major.
    pub latent: NodeId,
    /// `[text_tokens, joint_attention_dim]`.
    pub text: NodeId,
    /// `[timestep_channels]`, the sine/cosine embedding of `t * 1000`.
    pub timestep: NodeId,
    /// `[image_tokens + text_tokens, head_dim / 2]`.
    pub rope_cos: NodeId,
    pub rope_sin: NodeId,
    pub image_tokens: u32,
    pub text_tokens: u32,
}

const NORM_EPS: f32 = 1e-6;

fn linear_of(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    shape: &[u32],
) -> Result<NodeId, TensorError> {
    let weight = graph.weight(tensor, shape, None);
    linear(&mut graph.program, x, weight, None)
}

fn linear_rows(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    in_width: u32,
    rows: (u32, u32),
) -> Result<NodeId, TensorError> {
    let weight = graph.weight(tensor, &[rows.1 - rows.0, in_width], Some(rows));
    linear(&mut graph.program, x, weight, None)
}

fn dense(
    graph: &mut Graph,
    like: NodeId,
    body: ScalarOp,
    inputs: &[(NodeId, &str)],
) -> Result<NodeId, TensorError> {
    elementwise_like(&mut graph.program, like, body, inputs)
}

fn reduce(
    graph: &mut Graph,
    like: NodeId,
    body: ScalarOp,
    init: ReduceInit,
    operand: NodeId,
    in_map: &str,
    out_map: &str,
) -> Result<NodeId, TensorError> {
    let _ = like;
    reduce_like(&mut graph.program, body, init, operand, in_map, out_map)
}

fn scaled_sum(graph: &mut Graph, left: NodeId, right: NodeId, map: &str) -> Result<NodeId, TensorError> {
    dense(graph, left, ScalarOp::Add, &[(left, map), (right, map)])
}

fn scaled_mul(graph: &mut Graph, left: NodeId, right: NodeId, map: &str) -> Result<NodeId, TensorError> {
    dense(graph, left, ScalarOp::Multiply, &[(left, map), (right, map)])
}

fn multiply_gate(graph: &mut Graph, value: NodeId, gate: NodeId) -> Result<NodeId, TensorError> {
    dense(graph, value, ScalarOp::Multiply, &[(value, "td->td"), (gate, "d->td")])
}

fn norm(graph: &mut Graph, x: NodeId, hidden: u32) -> Result<NodeId, TensorError> {
    layer_norm_no_affine(&mut graph.program, x, hidden, NORM_EPS)
}

fn activated(graph: &mut Graph, x: NodeId, map: &str) -> Result<NodeId, TensorError> {
    let one = scalar_constant_like(&mut graph.program, 1.0, x)?;
    silu(&mut graph.program, x, one, map)
}

/// `[t, heads * head_dim]`'s feature slice `[offset, offset + width)`, viewed
/// as `[t, width / head_dim, head_dim]`.
fn slice_heads(
    graph: &mut Graph,
    x: NodeId,
    offset: u32,
    width: u32,
    head_dim: u32,
) -> Result<NodeId, TensorError> {
    let program = &mut graph.program;
    let part = slice_columns(program, x, offset, width)?;
    head_split(graph, part, width, head_dim)
}

/// `[t, heads * head_dim]` read as `[t, heads, head_dim]`. The `ones`
/// operand only pins both split axes (a two-term address defines neither) and
/// disappears when the consumer composes it.
fn head_split(
    graph: &mut Graph,
    x: NodeId,
    width: u32,
    head_dim: u32,
) -> Result<NodeId, TensorError> {
    let heads = width / head_dim;
    let dtype = graph.activation();
    let ones = proxima_tensor::op::Op::Constant {
        dtype,
        shape: vec![
            proxima_tensor::op::Extent::Static(heads),
            proxima_tensor::op::Extent::Static(head_dim),
        ],
        value: 1.0,
    };
    let ones = proxima_tensor::op::append(&mut graph.program, ones);
    dense(
        graph,
        x,
        ScalarOp::Multiply,
        &
        [
            (x, format!("t,{head_dim}*h+d->thd").as_str()),
            (ones, "hd->thd"),
        ],
    )
}

/// `[t, heads]`'s rows `[start, start + rows)`.
fn slice_tokens(
    graph: &mut Graph,
    x: NodeId,
    start: u32,
    rows: u32,
    trailing: &str,
) -> Result<NodeId, TensorError> {
    let axes: Vec<String> = trailing.chars().map(|letter| letter.to_string()).collect();
    let map = format!(
        "t+{start}@{rows},{}->t{trailing}",
        axes.join(",")
    );
    dense(graph, x, ScalarOp::Identity, &[(x, map.as_str())])
}

/// `[t, heads, head_dim]` projected by a checkpoint's `[out, feature]` weight
/// into `[t, out]`: one reduce whose weight feature axis is the head split.
fn project_heads(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    out_width: u32,
    head_dim: u32,
    total_features: u32,
    used_features: u32,
    feature_offset: Option<(u32, u32)>,
) -> Result<NodeId, TensorError> {
    let declared_heads = total_features / head_dim;
    let used_heads = used_features / head_dim;
    let weight = graph.weight_slice(
        tensor,
        &[out_width, declared_heads, head_dim],
        None,
        feature_offset,
    );
    let map = format!("o,u@{used_heads},d->soud");
    let product = dense(
        graph,
        x,
        ScalarOp::Multiply,
        &[(x, "sud->soud"), (weight, map.as_str())],
    )?;
    reduce(graph, x, ScalarOp::Add, ReduceInit::Zero, product, "soud->soud", "so->soud")
}

/// `[t, width]` scaled by a rank-1 gate and added to the residual.
fn gated_residual(
    graph: &mut Graph,
    residual: NodeId,
    value: NodeId,
    gate: NodeId,
) -> Result<NodeId, TensorError> {
    let scaled = multiply_gate(graph, value, gate)?;
    scaled_sum(graph, residual, scaled, "td->td")
}

/// The `silu`-gated MLP of one stream: `down(silu(gate) * up)`.
fn swiglu(graph: &mut Graph, x: NodeId, prefix: &str, config: &DitConfig) -> Result<NodeId, TensorError> {
    let mlp = config.mlp_hidden();
    let gate = linear_rows(graph, x, &format!("{prefix}.0.weight"), config.hidden, (0, mlp))?;
    let gate = activated(graph, gate, "td->td")?;
    let up = linear_rows(graph, x, &format!("{prefix}.0.weight"), config.hidden, (mlp, 2 * mlp))?;
    let gained = scaled_mul(graph, gate, up, "td->td")?;
    linear_of(graph, gained, &format!("{prefix}.2.weight"), &[config.hidden, mlp])
}

/// Per-head RMSNorm of a `[t, heads, head_dim]` query or key.
fn head_norm(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    config: &DitConfig,
    count: NodeId,
    rows: u32,
) -> Result<NodeId, TensorError> {
    let gamma = graph.weight(tensor, &[config.head_dim], None);
    let eps = graph.host(
        &format!("norm_eps.{rows}"),
        &[rows],
        vec![NORM_EPS; rows as usize],
    );
    rmsnorm_per_head(&mut graph.program, x, gamma, count, eps, "h")
}

fn rope(
    graph: &mut Graph,
    x: NodeId,
    inputs: &DitInputs,
) -> Result<(NodeId, NodeId), TensorError> {
    fused_rope_pair(
        &mut graph.program,
        x,
        'h',
        inputs.rope_cos,
        inputs.rope_sin,
        RopePairing::Interleaved,
    )
}

/// One double-stream block.
#[allow(clippy::too_many_arguments)]
fn double_block(
    graph: &mut Graph,
    config: &DitConfig,
    inputs: &DitInputs,
    prefix: &str,
    image: NodeId,
    text: NodeId,
    image_mod: [NodeId; 6],
    text_mod: [NodeId; 6],
    count: NodeId,
    inv_sqrt: NodeId,
    trace: &mut Vec<(&'static str, NodeId)>,
    record: bool,
) -> Result<(NodeId, NodeId), TensorError> {
    let program = &mut graph.program;
    let image_normed = layer_norm_no_affine(program, image, config.hidden, NORM_EPS)?;
    let image_normed = modulate(program, image_normed, image_mod[0], image_mod[1])?;
    let text_normed = layer_norm_no_affine(program, text, config.hidden, NORM_EPS)?;
    let text_normed = modulate(program, text_normed, text_mod[0], text_mod[1])?;
    if record {
        trace.push(("b_imgmod", image_normed));
    }

    let attention_in = config.attention_width();
    let image_qkv = linear_of(
        graph,
        image_normed,
        &format!("{prefix}.img_attn.qkv.weight"),
        &[attention_in * 3, config.hidden],
    )?;
    let text_qkv = linear_of(
        graph,
        text_normed,
        &format!("{prefix}.txt_attn.qkv.weight"),
        &[attention_in * 3, config.hidden],
    )?;

    let mut parts = Vec::with_capacity(6);
    for (stream, qkv, rows) in [
        ("txt", text_qkv, inputs.text_tokens),
        ("img", image_qkv, inputs.image_tokens),
    ] {
        let attention = format!("{prefix}.{stream}_attn");
        for (offset, norm_name) in [
            (0, Some("query_norm.scale")),
            (attention_in, Some("key_norm.scale")),
            (attention_in * 2, None),
        ] {
            let part = slice_heads(graph, qkv, offset, attention_in, config.head_dim)?;
            parts.push(match norm_name {
                Some(name) => head_norm(
                    graph,
                    part,
                    &format!("{attention}.norm.{name}"),
                    config,
                    count,
                    rows,
                )?,
                None => part,
            });
        }
    }
    let (text_q, text_k, text_v, image_q, image_k, image_v) = (
        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
    );
    let index = graph.concat_indices(inputs.text_tokens, inputs.image_tokens, &format!("{prefix}.qkv"));
    let program = &mut graph.program;
    if record {
        trace.push(("b_imgq", image_q));
        trace.push(("b_imgk", image_k));
        trace.push(("b_imgv", image_v));
    }
    let q = concat_rows(program, text_q, image_q, index[0], index[1], index[2], "hd")?;
    let k = concat_rows(program, text_k, image_k, index[0], index[1], index[2], "hd")?;
    let v = concat_rows(program, text_v, image_v, index[0], index[1], index[2], "hd")?;
    let (q_even, q_odd) = rope(graph, q, inputs)?;
    let (k_even, k_odd) = rope(graph, k, inputs)?;
    let program = &mut graph.program;
    let attended = non_causal_attention_heads(
        program, q_even, q_odd, k_even, k_odd, v, inv_sqrt, "u", "p", "d",
    )?;
    if record {
        trace.push(("b_attn", attended));
    }
    let image_attended = slice_tokens(graph, attended, inputs.text_tokens, inputs.image_tokens, "ud")?;
    let image_attended = project_heads(
        graph,
        image_attended,
        &format!("{prefix}.img_attn.proj.weight"),
        config.hidden,
        config.head_dim,
        config.hidden,
        config.hidden,
        None,
    )?;
    let text_attended = slice_tokens(graph, attended, 0, inputs.text_tokens, "ud")?;
    let text_attended = project_heads(
        graph,
        text_attended,
        &format!("{prefix}.txt_attn.proj.weight"),
        config.hidden,
        config.head_dim,
        config.hidden,
        config.hidden,
        None,
    )?;
    if record {
        trace.push(("b_imgattnproj", image_attended));
    }
    let image = gated_residual(graph, image, image_attended, image_mod[2])?;
    let text = gated_residual(graph, text, text_attended, text_mod[2])?;

    let image_mlp_in = norm(graph, image, config.hidden)?;
    let image_mlp_in = {
        let program = &mut graph.program;
        modulate(program, image_mlp_in, image_mod[3], image_mod[4])?
    };
    let image_mlp = swiglu(graph, image_mlp_in, &format!("{prefix}.img_mlp"), config)?;
    if record {
        trace.push(("b_imgmlp", image_mlp));
    }
    let image = gated_residual(graph, image, image_mlp, image_mod[5])?;
    let text_mlp_in = norm(graph, text, config.hidden)?;
    let text_mlp_in = {
        let program = &mut graph.program;
        modulate(program, text_mlp_in, text_mod[3], text_mod[4])?
    };
    let text_mlp = swiglu(graph, text_mlp_in, &format!("{prefix}.txt_mlp"), config)?;
    let text = gated_residual(graph, text, text_mlp, text_mod[5])?;
    Ok((image, text))
}

/// One single-stream block.
fn single_block(
    graph: &mut Graph,
    config: &DitConfig,
    inputs: &DitInputs,
    prefix: &str,
    x: NodeId,
    modulation: [NodeId; 3],
    count: NodeId,
    inv_sqrt: NodeId,
) -> Result<NodeId, TensorError> {
    let program = &mut graph.program;
    let normed = layer_norm_no_affine(program, x, config.hidden, NORM_EPS)?;
    let normed = modulate(program, normed, modulation[0], modulation[1])?;
    let fused = linear_of(
        graph,
        normed,
        &format!("{prefix}.linear1.weight"),
        &[config.single_linear1_width(), config.hidden],
    )?;
    let attention_in = config.attention_width();
    let rows = inputs.text_tokens + inputs.image_tokens;
    let q = slice_heads(graph, fused, 0, attention_in, config.head_dim)?;
    let q = head_norm(graph, q, &format!("{prefix}.norm.query_norm.scale"), config, count, rows)?;
    let k = slice_heads(graph, fused, attention_in, attention_in, config.head_dim)?;
    let k = head_norm(graph, k, &format!("{prefix}.norm.key_norm.scale"), config, count, rows)?;
    let v = slice_heads(graph, fused, attention_in * 2, attention_in, config.head_dim)?;
    let program = &mut graph.program;
    let mlp = slice_columns(program, fused, attention_in * 3, config.mlp_hidden() * 2)?;
    let program = &mut graph.program;
    let gate = slice_columns(program, mlp, 0, config.mlp_hidden())?;
    let gate = activated(graph, gate, "td->td")?;
    let program = &mut graph.program;
    let up = slice_columns(program, mlp, config.mlp_hidden(), config.mlp_hidden())?;
    let gained = scaled_mul(graph, gate, up, "td->td")?;
    let mlp_out = {
        let weight = graph.weight_slice(
            &format!("{prefix}.linear2.weight"),
            &[config.hidden, config.mlp_hidden()],
            None,
            Some((attention_in, attention_in + config.mlp_hidden())),
        );
        linear(&mut graph.program, gained, weight, None)?
    };
    let (q_even, q_odd) = rope(graph, q, inputs)?;
    let (k_even, k_odd) = rope(graph, k, inputs)?;
    let program = &mut graph.program;
    let attended = non_causal_attention_heads(
        program, q_even, q_odd, k_even, k_odd, v, inv_sqrt, "u", "p", "d",
    )?;
    let attention_out = project_heads(
        graph,
        attended,
        &format!("{prefix}.linear2.weight"),
        config.hidden,
        config.head_dim,
        attention_in,
        attention_in,
        Some((0, attention_in)),
    )?;
    let output = scaled_sum(graph, attention_out, mlp_out, "td->td")?;
    gated_residual(graph, x, output, modulation[2])
}

/// One double-stream block as its own program fragment: the per-shape bench
/// drives this without the prelude, the other blocks or the final layer.
///
/// # Errors
///
/// [`TensorError`] when a notation, extent, or tensor name fails to build.
#[allow(clippy::too_many_arguments)]
pub fn build_double_block(
    graph: &mut Graph,
    config: &DitConfig,
    inputs: &DitInputs,
    prefix: &str,
    image: NodeId,
    text: NodeId,
    image_mod: [NodeId; 6],
    text_mod: [NodeId; 6],
) -> Result<(NodeId, NodeId), TensorError> {
    let activation = graph.activation();
    let count = count_constant(&mut graph.program, config.head_dim);
    let inv_sqrt = unit_constant(
        &mut graph.program,
        1.0 / (config.head_dim as f32).sqrt(),
        activation,
    );
    let mut trace = Vec::new();
    double_block(
        graph,
        config,
        inputs,
        prefix,
        image,
        text,
        image_mod,
        text_mod,
        count,
        inv_sqrt,
        &mut trace,
        false,
    )
}

/// One single-stream block as its own program fragment.
///
/// # Errors
///
/// [`TensorError`] when a notation, extent, or tensor name fails to build.
pub fn build_single_block(
    graph: &mut Graph,
    config: &DitConfig,
    inputs: &DitInputs,
    prefix: &str,
    x: NodeId,
    modulation: [NodeId; 3],
) -> Result<NodeId, TensorError> {
    let activation = graph.activation();
    let count = count_constant(&mut graph.program, config.head_dim);
    let inv_sqrt = unit_constant(
        &mut graph.program,
        1.0 / (config.head_dim as f32).sqrt(),
        activation,
    );
    single_block(graph, config, inputs, prefix, x, modulation, count, inv_sqrt)
}

fn modulation_chunks(
    graph: &mut Graph,
    vector: NodeId,
    tensor: &str,
    width: u32,
    chunks: u32,
) -> Result<Vec<NodeId>, TensorError> {
    let weight = graph.weight(tensor, &[width * chunks, width], None);
    let projected = matvec(&mut graph.program, vector, weight, None)?;
    (0..chunks)
        .map(|chunk| slice_range(&mut graph.program, projected, chunk * width, width))
        .collect()
}

/// The backbone's output plus the intermediate nodes a parity check reads.
#[derive(Debug, Clone)]
pub struct DitOutput {
    pub velocity: NodeId,
    pub trace: Vec<(&'static str, NodeId)>,
}

/// Builds the whole backbone as one program, returning the velocity node
/// (`[image_tokens, in_channels]`, token-major) and the node of every stage a
/// parity check compares against the reference's own capture.
///
/// # Errors
///
/// [`TensorError`] when a notation, extent, or tensor name fails to build.
pub fn build_dit(
    graph: &mut Graph,
    config: &DitConfig,
    inputs: &DitInputs,
) -> Result<DitOutput, TensorError> {
    let mut trace = Vec::new();
    let image_in = linear_of(
        graph,
        inputs.latent,
        "img_in.weight",
        &[config.hidden, config.in_channels],
    )?;
    let text_in = linear_of(
        graph,
        inputs.text,
        "txt_in.weight",
        &[config.hidden, config.joint_attention_dim],
    )?;
    let weight = graph.weight(
        "time_in.in_layer.weight",
        &[config.hidden, config.timestep_channels],
        None,
    );
    let time = matvec(&mut graph.program, inputs.timestep, weight, None)?;
    let time = activated(graph, time, "d->d")?;
    let weight = graph.weight(
        "time_in.out_layer.weight",
        &[config.hidden, config.hidden],
        None,
    );
    let vector = matvec(&mut graph.program, time, weight, None)?;
    trace.push(("img_in", image_in));
    trace.push(("txt_in", text_in));
    trace.push(("vec", vector));
    let shared = activated(graph, vector, "d->d")?;
    let image_mod = modulation_chunks(
        graph,
        shared,
        "double_stream_modulation_img.lin.weight",
        config.hidden,
        6,
    )?;
    let text_mod = modulation_chunks(
        graph,
        shared,
        "double_stream_modulation_txt.lin.weight",
        config.hidden,
        6,
    )?;
    let single_mod = modulation_chunks(
        graph,
        shared,
        "single_stream_modulation.lin.weight",
        config.hidden,
        3,
    )?;
    let final_mod = modulation_chunks(
        graph,
        shared,
        "final_layer.adaLN_modulation.1.weight",
        config.hidden,
        2,
    )?;
    let activation = graph.activation();
    let count = count_constant(&mut graph.program, config.head_dim);
    let inv_sqrt = unit_constant(
        &mut graph.program,
        1.0 / (config.head_dim as f32).sqrt(),
        activation,
    );

    let mut image = image_in;
    let mut text = text_in;
    for block in 0..config.double_blocks {
        let prefix = format!("double_blocks.{block}");
        let image_mod: [NodeId; 6] = image_mod[..6].try_into().expect("six image modulations");
        let text_mod: [NodeId; 6] = text_mod[..6].try_into().expect("six text modulations");
        let (next_image, next_text) = double_block(
            graph,
            config,
            inputs,
            &prefix,
            image,
            text,
            image_mod,
            text_mod,
            count,
            inv_sqrt,
            &mut trace,
            block == 0,
        )?;
        image = next_image;
        text = next_text;
        if block == 0 {
            trace.push(("d0_img", image));
            trace.push(("d0_txt", text));
        }
    }

    let index = graph.concat_indices(inputs.text_tokens, inputs.image_tokens, "single.joint");
    let program = &mut graph.program;
    let mut stream = concat_rows(program, text, image, index[0], index[1], index[2], "c")?;
    let single_mod: [NodeId; 3] = single_mod[..3].try_into().expect("three single modulations");
    for block in 0..config.single_blocks {
        let prefix = format!("single_blocks.{block}");
        stream = single_block(
            graph, config, inputs, &prefix, stream, single_mod, count, inv_sqrt,
        )?;
        if block == 0 {
            trace.push(("s0", stream));
        }
    }

    let final_mod: [NodeId; 2] = final_mod[..2].try_into().expect("two final modulations");
    let program = &mut graph.program;
    let normed = layer_norm_no_affine(program, stream, config.hidden, NORM_EPS)?;
    let normed = modulate(program, normed, final_mod[0], final_mod[1])?;
    let image_tokens = proxima_tensor::spec::slice_rows(
        &mut graph.program,
        normed,
        inputs.text_tokens,
        inputs.image_tokens,
    )?;
    let velocity = linear_of(
        graph,
        image_tokens,
        "final_layer.linear.weight",
        &[config.in_channels, config.hidden],
    )?;
    trace.push(("final", velocity));
    Ok(DitOutput { velocity, trace })
}
