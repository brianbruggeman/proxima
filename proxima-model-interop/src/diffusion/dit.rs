//! A flow-matching transformer backbone as one [`Op`] program: linear prelude,
//! N double-stream blocks with joint attention, M single-stream blocks, and the
//! final adaLN layer. Every dimension is a config value and every tensor name
//! is the checkpoint's own (the diffusers layout its safetensors header
//! declares).

use std::str::FromStr;

use proxima_numeric::{NumericFormat, PackedRows};
use proxima_tensor::TensorError;
use proxima_tensor::op::{NodeId, ReduceInit, ScalarOp};
use proxima_tensor::spec::{
    RopePairing, concat_rows, count_constant, elementwise_like, fused_rope_pair,
    layer_norm_no_affine, modulate, non_causal_attention_heads, project_rows, project_vector,
    reduce_like, rmsnorm_per_head, scalar_constant_as, scalar_constant_like, silu, slice_columns,
    slice_range,
};

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
    pub const fn single_stage1_width(&self) -> u32 {
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
    /// The per-head norm count constant, from [`head_norm_count`].
    pub norm_count: NodeId,
    /// The attention scale `1 / sqrt(head_dim)`, from [`attention_scale`].
    pub attention_scale: NodeId,
    pub image_tokens: u32,
    pub text_tokens: u32,
}

fn norm_eps<T: FromStr>(graph: &Graph, like: NodeId) -> Result<T, TensorError> {
    graph
        .settings()
        .norm_eps()
        .map_err(|_| TensorError::NotLowerable {
            node: like,
            reason: "norm_eps is not a number",
        })
}

fn project_leaf(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    shape: &[u32],
) -> Result<NodeId, TensorError> {
    let weight = graph.weight(tensor, shape, None);
    project_rows(&mut graph.program, x, weight, None)
}

fn project_row_slice(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    in_width: u32,
    rows: (u32, u32),
) -> Result<NodeId, TensorError> {
    let weight = graph.weight(tensor, &[rows.1 - rows.0, in_width], Some(rows));
    project_rows(&mut graph.program, x, weight, None)
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

fn scaled_sum(
    graph: &mut Graph,
    left: NodeId,
    right: NodeId,
    map: &str,
) -> Result<NodeId, TensorError> {
    dense(graph, left, ScalarOp::Add, &[(left, map), (right, map)])
}

fn scaled_mul(
    graph: &mut Graph,
    left: NodeId,
    right: NodeId,
    map: &str,
) -> Result<NodeId, TensorError> {
    dense(
        graph,
        left,
        ScalarOp::Multiply,
        &[(left, map), (right, map)],
    )
}

fn multiply_gate(graph: &mut Graph, value: NodeId, gate: NodeId) -> Result<NodeId, TensorError> {
    dense(
        graph,
        value,
        ScalarOp::Multiply,
        &[(value, "td->td"), (gate, "d->td")],
    )
}

fn norm(graph: &mut Graph, x: NodeId, hidden: u32) -> Result<NodeId, TensorError> {
    let eps = norm_eps(graph, x)?;
    layer_norm_no_affine(&mut graph.program, x, hidden, eps)
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
        &[
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
    let map = format!("t+{start}@{rows},{}->t{trailing}", axes.join(","));
    dense(graph, x, ScalarOp::Identity, &[(x, map.as_str())])
}

/// `[t, heads, head_dim]` projected by a checkpoint's `[out, feature]` weight
/// into `[t, hidden]`: one reduce whose weight feature axis is the head split.
fn project_heads(
    graph: &mut Graph,
    x: NodeId,
    tensor: &str,
    config: &DitConfig,
    total_features: u32,
    used_features: u32,
    feature_offset: Option<(u32, u32)>,
) -> Result<NodeId, TensorError> {
    let declared_heads = total_features / config.head_dim;
    let used_heads = used_features / config.head_dim;
    let weight = graph.weight_slice(
        tensor,
        &[config.hidden, declared_heads, config.head_dim],
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
    reduce(
        graph,
        x,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "soud->soud",
        "so->soud",
    )
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
fn swiglu(
    graph: &mut Graph,
    x: NodeId,
    prefix: &str,
    config: &DitConfig,
) -> Result<NodeId, TensorError> {
    let mlp = config.mlp_hidden();
    let gate = project_row_slice(
        graph,
        x,
        &format!("{prefix}.0.weight"),
        config.hidden,
        (0, mlp),
    )?;
    let gate = activated(graph, gate, "td->td")?;
    let up = project_row_slice(
        graph,
        x,
        &format!("{prefix}.0.weight"),
        config.hidden,
        (mlp, 2 * mlp),
    )?;
    let gained = scaled_mul(graph, gate, up, "td->td")?;
    project_leaf(
        graph,
        gained,
        &format!("{prefix}.2.weight"),
        &[config.hidden, mlp],
    )
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
    let eps_value = norm_eps(graph, x)?;
    let packed =
        PackedRows::from_values(graph.activation_format(), &vec![eps_value; rows as usize]);
    let eps = graph.host(&format!("norm_eps.{rows}"), &[rows], packed);
    rmsnorm_per_head(&mut graph.program, x, gamma, count, eps, "h")
}

fn rope(graph: &mut Graph, x: NodeId, inputs: &DitInputs) -> Result<(NodeId, NodeId), TensorError> {
    fused_rope_pair(
        &mut graph.program,
        x,
        'h',
        inputs.rope_cos,
        inputs.rope_sin,
        RopePairing::Interleaved,
    )
}

/// One double-stream block; `streams` and `modulation` are ordered `[image, text]`.
fn double_block(
    graph: &mut Graph,
    config: &DitConfig,
    inputs: &DitInputs,
    prefix: &str,
    streams: [NodeId; 2],
    modulation: [[NodeId; 6]; 2],
    mut trace: Option<&mut Vec<(&'static str, NodeId)>>,
) -> Result<(NodeId, NodeId), TensorError> {
    let [image, text] = streams;
    let [image_mod, text_mod] = modulation;
    let count = inputs.norm_count;
    let inv_sqrt = inputs.attention_scale;
    let eps = norm_eps(graph, image)?;
    let program = &mut graph.program;
    let image_normed = layer_norm_no_affine(program, image, config.hidden, eps)?;
    let image_normed = modulate(program, image_normed, image_mod[0], image_mod[1])?;
    let text_normed = layer_norm_no_affine(program, text, config.hidden, eps)?;
    let text_normed = modulate(program, text_normed, text_mod[0], text_mod[1])?;
    if let Some(trace) = trace.as_mut() {
        trace.push(("b_imgmod", image_normed));
    }

    let attention_in = config.attention_width();
    let image_qkv = project_leaf(
        graph,
        image_normed,
        &format!("{prefix}.img_attn.qkv.weight"),
        &[attention_in * 3, config.hidden],
    )?;
    let text_qkv = project_leaf(
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
    let (text_q, text_k, text_v, image_q, image_k, image_v) =
        (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]);
    let index = graph.concat_indices(
        inputs.text_tokens,
        inputs.image_tokens,
        &format!("{prefix}.qkv"),
    );
    let program = &mut graph.program;
    if let Some(trace) = trace.as_mut() {
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
        program,
        &[(q_even, k_even), (q_odd, k_odd)],
        v,
        inv_sqrt,
        "u",
        "p",
        "d",
    )?;
    if let Some(trace) = trace.as_mut() {
        trace.push(("b_attn", attended));
    }
    let image_attended = slice_tokens(
        graph,
        attended,
        inputs.text_tokens,
        inputs.image_tokens,
        "ud",
    )?;
    let image_attended = project_heads(
        graph,
        image_attended,
        &format!("{prefix}.img_attn.proj.weight"),
        config,
        config.hidden,
        config.hidden,
        None,
    )?;
    let text_attended = slice_tokens(graph, attended, 0, inputs.text_tokens, "ud")?;
    let text_attended = project_heads(
        graph,
        text_attended,
        &format!("{prefix}.txt_attn.proj.weight"),
        config,
        config.hidden,
        config.hidden,
        None,
    )?;
    if let Some(trace) = trace.as_mut() {
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
    if let Some(trace) = trace.as_mut() {
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
) -> Result<NodeId, TensorError> {
    let count = inputs.norm_count;
    let inv_sqrt = inputs.attention_scale;
    let eps = norm_eps(graph, x)?;
    let program = &mut graph.program;
    let normed = layer_norm_no_affine(program, x, config.hidden, eps)?;
    let normed = modulate(program, normed, modulation[0], modulation[1])?;
    let fused = project_leaf(
        graph,
        normed,
        &format!("{prefix}.linear1.weight"),
        &[config.single_stage1_width(), config.hidden],
    )?;
    let attention_in = config.attention_width();
    let rows = inputs.text_tokens + inputs.image_tokens;
    let q = slice_heads(graph, fused, 0, attention_in, config.head_dim)?;
    let q = head_norm(
        graph,
        q,
        &format!("{prefix}.norm.query_norm.scale"),
        config,
        count,
        rows,
    )?;
    let k = slice_heads(graph, fused, attention_in, attention_in, config.head_dim)?;
    let k = head_norm(
        graph,
        k,
        &format!("{prefix}.norm.key_norm.scale"),
        config,
        count,
        rows,
    )?;
    let v = slice_heads(
        graph,
        fused,
        attention_in * 2,
        attention_in,
        config.head_dim,
    )?;
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
        project_rows(&mut graph.program, gained, weight, None)?
    };
    let (q_even, q_odd) = rope(graph, q, inputs)?;
    let (k_even, k_odd) = rope(graph, k, inputs)?;
    let program = &mut graph.program;
    let attended = non_causal_attention_heads(
        program,
        &[(q_even, k_even), (q_odd, k_odd)],
        v,
        inv_sqrt,
        "u",
        "p",
        "d",
    )?;
    let attention_out = project_heads(
        graph,
        attended,
        &format!("{prefix}.linear2.weight"),
        config,
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
pub fn build_double_block(
    graph: &mut Graph,
    config: &DitConfig,
    inputs: &DitInputs,
    prefix: &str,
    streams: [NodeId; 2],
    modulation: [[NodeId; 6]; 2],
) -> Result<(NodeId, NodeId), TensorError> {
    double_block(graph, config, inputs, prefix, streams, modulation, None)
}

/// The constant every per-head norm divides by, built once per program.
pub fn head_norm_count(graph: &mut Graph, config: &DitConfig) -> NodeId {
    count_constant(&mut graph.program, config.head_dim)
}

/// The constant attention scores are scaled by, built once per program.
///
/// # Errors
///
/// [`TensorError`] when a node fails to build.
pub fn attention_scale(graph: &mut Graph, config: &DitConfig) -> Result<NodeId, TensorError> {
    inverse_root_constant(graph, config.head_dim)
}

/// `1 / sqrt(count)` as a rank-0 constant in the graph's activation dtype,
/// with `count` read back at the host element, which holds an integer exactly,
/// so the constant is rounded once, into the activation dtype.
///
/// # Errors
///
/// [`TensorError`] when the count does not round-trip through the host element.
pub(crate) fn inverse_root_constant(graph: &mut Graph, count: u32) -> Result<NodeId, TensorError> {
    let activation = graph.activation();
    let packed = PackedRows::from_values(&NumericFormat::host_element(), &[count as _])?;
    let mut decoded = Vec::new();
    packed.decode_all(&mut decoded)?;
    let [width] = decoded.as_slice() else {
        return Err(TensorError::Empty);
    };
    let scale = 1.0 / width.sqrt();
    Ok(scalar_constant_as(&mut graph.program, scale, activation))
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
    single_block(graph, config, inputs, prefix, x, modulation)
}

fn modulation_chunks(
    graph: &mut Graph,
    vector: NodeId,
    tensor: &str,
    width: u32,
    chunks: u32,
) -> Result<Vec<NodeId>, TensorError> {
    let weight = graph.weight(tensor, &[width * chunks, width], None);
    let projected = project_vector(&mut graph.program, vector, weight, None)?;
    (0..chunks)
        .map(|chunk| slice_range(&mut graph.program, projected, chunk * width, width))
        .collect()
}

fn exact_chunks<const COUNT: usize>(chunks: Vec<NodeId>) -> Result<[NodeId; COUNT], TensorError> {
    let found = chunks.len();
    chunks.try_into().map_err(|_| {
        TensorError::InvalidPlan(format!("expected {COUNT} modulation chunks, found {found}"))
    })
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
    let image_in = project_leaf(
        graph,
        inputs.latent,
        "img_in.weight",
        &[config.hidden, config.in_channels],
    )?;
    let text_in = project_leaf(
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
    let time = project_vector(&mut graph.program, inputs.timestep, weight, None)?;
    let time = activated(graph, time, "d->d")?;
    let weight = graph.weight(
        "time_in.out_layer.weight",
        &[config.hidden, config.hidden],
        None,
    );
    let vector = project_vector(&mut graph.program, time, weight, None)?;
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
    let image_mod: [NodeId; 6] = exact_chunks(image_mod)?;
    let text_mod: [NodeId; 6] = exact_chunks(text_mod)?;
    let single_mod: [NodeId; 3] = exact_chunks(single_mod)?;
    let final_mod: [NodeId; 2] = exact_chunks(final_mod)?;

    let mut image = image_in;
    let mut text = text_in;
    for block in 0..config.double_blocks {
        let prefix = format!("double_blocks.{block}");
        let (next_image, next_text) = double_block(
            graph,
            config,
            inputs,
            &prefix,
            [image, text],
            [image_mod, text_mod],
            (block == 0).then_some(&mut trace),
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
    for block in 0..config.single_blocks {
        let prefix = format!("single_blocks.{block}");
        stream = single_block(graph, config, inputs, &prefix, stream, single_mod)?;
        if block == 0 {
            trace.push(("s0", stream));
        }
    }

    let eps = norm_eps(graph, stream)?;
    let program = &mut graph.program;
    let normed = layer_norm_no_affine(program, stream, config.hidden, eps)?;
    let normed = modulate(program, normed, final_mod[0], final_mod[1])?;
    let image_tokens = proxima_tensor::spec::slice_rows(
        &mut graph.program,
        normed,
        inputs.text_tokens,
        inputs.image_tokens,
    )?;
    let velocity = project_leaf(
        graph,
        image_tokens,
        "final_layer.linear.weight",
        &[config.in_channels, config.hidden],
    )?;
    trace.push(("final", velocity));
    Ok(DitOutput { velocity, trace })
}

#[cfg(test)]
mod linear2_binding {
    //! The single block's fused `linear2` consumed as today's two column-window
    //! leaves (`windowed`) and as one whole leaf read twice (`whole`): the attention
    //! half through [`project_heads`], the mlp half through
    //! [`proxima_tensor::spec::project_window`]. Both forms share the surrounding
    //! `attended`/`gained` leaves and the closing [`scaled_sum`], so only the binding differs.

    use proxima_numeric::{NumericError, PackedRows, Philox, compare};
    use proxima_tensor::QuantizedBlock;
    use proxima_tensor::spec::project_window;

    use super::super::DiffusionSettings;
    use super::*;

    type Outcome = Result<(), String>;
    #[cfg(all(
        feature = "metal",
        feature = "instrument",
        feature = "metal-tiled-gemm",
        target_os = "macos"
    ))]
    type Verdict = Result<(), omega::msl::TiledGemmRejection>;

    const LINEAR2: &str = "single_blocks.0.linear2.weight";

    fn config(hidden: u32, heads: u32, head_dim: u32) -> DitConfig {
        DitConfig {
            in_channels: 128,
            hidden,
            heads,
            head_dim,
            joint_attention_dim: 7680,
            double_blocks: 0,
            single_blocks: 1,
            timestep_channels: 256,
            rope_theta: 2000,
        }
    }

    fn rows(graph: &Graph, count: usize, seed: u64) -> Result<PackedRows, NumericError> {
        PackedRows::from_values(graph.activation_format(), &Philox::new(seed).normal(count))
    }

    /// `linear2`'s closing sum for `tokens` rows; `filled` binds deterministic
    /// activations, otherwise the leaves are shape-only.
    fn fragment(
        config: &DitConfig,
        tokens: u32,
        whole: bool,
        filled: bool,
    ) -> Result<(Graph, NodeId), String> {
        let settings = DiffusionSettings::try_resolved().map_err(|why| why.to_string())?;
        let mut graph = Graph::with_settings(
            settings.dit_activation_dtype()?,
            settings,
        )?;
        let attention_in = config.attention_width();
        let mlp = config.mlp_hidden();
        let heads = attention_in / config.head_dim;
        let attended_count = (tokens * attention_in) as usize;
        let gained_count = (tokens * mlp) as usize;
        let (attended_values, gained_values) = if filled {
            (rows(&graph, attended_count, 1), rows(&graph, gained_count, 2))
        } else {
            (graph.unbound(), graph.unbound())
        };
        let attended = graph.host("attended", &[tokens, heads, config.head_dim], attended_values);
        let gained = graph.host("gained", &[tokens, mlp], gained_values);
        let (attention_out, mlp_out) = if whole {
            let total = attention_in + mlp;
            let attention_out =
                project_heads(&mut graph, attended, LINEAR2, config, total, attention_in, None)
                    .map_err(|why| why.to_string())?;
            let weight = graph.weight_slice(LINEAR2, &[config.hidden, total], None, None);
            let mlp_out = project_window(&mut graph.program, gained, weight, attention_in, mlp)
                .map_err(|why| why.to_string())?;
            (attention_out, mlp_out)
        } else {
            let attention_out = project_heads(
                &mut graph,
                attended,
                LINEAR2,
                config,
                attention_in,
                attention_in,
                Some((0, attention_in)),
            )
            .map_err(|why| why.to_string())?;
            let weight = graph.weight_slice(
                LINEAR2,
                &[config.hidden, mlp],
                None,
                Some((attention_in, attention_in + mlp)),
            );
            let mlp_out = project_rows(&mut graph.program, gained, weight, None)
                .map_err(|why| why.to_string())?;
            (attention_out, mlp_out)
        };
        let root = scaled_sum(&mut graph, attention_out, mlp_out, "td->td")
            .map_err(|why| why.to_string())?;
        Ok((graph, root))
    }

    /// The checkpoint tensor `whole` (`[hidden, columns]`, row-major) cut to a leaf's column window.
    fn leaf_values<T: Copy>(whole: &[T], columns: usize, window: Option<(u32, u32)>) -> Vec<T> {
        let (start, end) = window.map_or((0, columns), |(start, end)| (start as usize, end as usize));
        whole
            .chunks(columns)
            .flat_map(|row| row[start..end].iter().copied())
            .collect()
    }

    fn evaluated(config: &DitConfig, tokens: u32, whole: bool) -> Result<PackedRows, String> {
        let (graph, root) = fragment(config, tokens, whole, true)?;
        let columns = (config.attention_width() + config.mlp_hidden()) as usize;
        let tensor = Philox::new(3).normal(config.hidden as usize * columns);
        let leaves: Vec<Vec<_>> = graph
            .weights
            .iter()
            .map(|binding| leaf_values(&tensor, columns, binding.columns))
            .collect();
        let mut bound = graph.host_blocks(&[])?;
        for (binding, values) in graph.weights.iter().zip(&leaves) {
            bound.push((binding.leaf.as_str(), QuantizedBlock::from(values.as_slice())));
        }
        let mut pools = proxima_tensor::cpu::BufferPools::default();
        let mut validated = None;
        let result = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
            &graph.program,
            &[],
            &bound,
            &[root],
            &mut pools,
            &mut validated,
        )
        .map_err(|why| why.to_string())?;
        let value = result.get(root).ok_or("the root has no value")?;
        PackedRows::from_values(graph.activation_format(), value.0).map_err(|why| why.to_string())
    }

    #[test]
    fn fused_projection_bound_whole_equals_two_column_windows_on_small_shapes() -> Outcome {
        let small = config(64, 4, 16);
        let windowed = evaluated(&small, 8, false)?;
        let whole = evaluated(&small, 8, true)?;
        let rms = compare::rms_rel(&whole, &windowed).map_err(|why| why.to_string())?;
        let mut measured = Vec::new();
        rms.decode_all(&mut measured).map_err(|why| why.to_string())?;
        assert!(measured[0] < 1e-4, "relative rms difference {}", measured[0]);
        Ok(())
    }

    /// Every reduce that reads `linear2`, with the tiled gemm's verdict, at FLUX shapes
    /// and the checkpoint weight format.
    #[cfg(all(
        feature = "metal",
        feature = "instrument",
        feature = "metal-tiled-gemm",
        target_os = "macos"
    ))]
    fn classified(whole: bool) -> Result<Vec<(String, Verdict)>, String> {
        use std::collections::BTreeSet;

        use proxima_tensor::{BoundOpKind, Keep};

        let flux = config(3072, 24, 128);
        let (graph, root) = fragment(&flux, 4608, whole, false)?;
        let codec = DiffusionSettings::try_resolved()
            .map_err(|why| why.to_string())?
            .dit_compute;
        let shapes = proxima_tensor::infer(&graph.program, &[]).map_err(|why| why.to_string())?;
        let packed: BTreeSet<NodeId> = graph
            .program
            .iter()
            .enumerate()
            .filter(|(_, op)| op.name().is_some_and(|name| name.starts_with(LINEAR2)))
            .map(|(position, _)| NodeId(position as u32))
            .collect();
        let mut bound = proxima_tensor::bind(
            &graph.program,
            &shapes,
            &[root],
            proxima_numeric::NumericPolicy::default(),
        )
        .map_err(|why| why.to_string())?;
        proxima_tensor::correct_packed_matmul_layouts(&mut bound, &packed);
        let mut verdicts = Vec::new();
        for op in &bound {
            let BoundOpKind::Reduce {
                reduce_op,
                init,
                keep: Keep::Reduce,
                output_axes,
                ..
            } = &op.kind
            else {
                continue;
            };
            let operands = op.operands();
            let Some((weight, _, _)) = operands.iter().find(|(node, _, _)| packed.contains(node)) else {
                continue;
            };
            let quantized: Vec<Option<_>> = operands
                .iter()
                .map(|(node, _, _)| packed.contains(node).then_some(codec))
                .collect();
            let name = graph.program[weight.0 as usize].name().unwrap_or_default().to_string();
            let verdict = omega::msl::diagnose_tiled_gemm_block(op, &quantized, *reduce_op, *init, output_axes);
            verdicts.push((name, verdict));
        }
        Ok(verdicts)
    }

    #[cfg(all(
        feature = "metal",
        feature = "instrument",
        feature = "metal-tiled-gemm",
        target_os = "macos"
    ))]
    #[test]
    fn fused_leaf_bound_whole_is_tiled_for_both_windows() -> Outcome {
        for whole in [false, true] {
            let verdicts = classified(whole)?;
            assert_eq!(verdicts.len(), 2, "whole={whole}: one reduce per window, got {verdicts:?}");
            for (name, verdict) in &verdicts {
                assert_eq!(verdict, &Ok(()), "whole={whole} {name} is refused by the tiled kernel");
            }
        }
        Ok(())
    }
}
