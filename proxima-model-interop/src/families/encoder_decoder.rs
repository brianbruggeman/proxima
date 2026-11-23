//! Generic encoder-decoder graph builders: a convolutional audio front end, a
//! bidirectional encoder stack, a cross-attention KV projection and a causal
//! decoder stack. No model name and no per-model profile: every dimension,
//! layer count and weight is a value the caller read from the checkpoint.
//!
//! The builders compose the existing algebra (`elementwise`, `reduce`, `Iota`,
//! `Constant`) plus [`proxima_tensor::spec`]'s generic lowerings, so the graph
//! is the same one `proxima-autograd` differentiates. Shapes are inferred from
//! the operand maps; a caller passes only the head width and the mask choice.

use alloc::vec::Vec;

use proxima_numeric::{DType, FloatLayout};
use proxima_tensor::error::TensorError;
use proxima_tensor::op::{Extent, NodeId, Op, ReduceInit, ScalarOp};
use proxima_tensor::spec::{
    elementwise, embedding_lookup, input_leaf, layer_norm, reduce, scalar_constant,
};

/// The graph's activation format, named by `proxima-numeric`'s canonical
/// layout rather than a literal dtype. A settings-driven format replaces this
/// one constant in PROXIMAINTEROP-11.
pub const ACTIVATION: DType = DType::Layout(FloatLayout::BF32);

/// A weight matrix `[in, out]` and an optional `[out]` bias.
#[derive(Debug, Clone, Copy)]
pub struct Linear {
    pub weight: NodeId,
    pub bias: Option<NodeId>,
}

/// One attention block's four projections.
#[derive(Debug, Clone, Copy)]
pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
}

/// One encoder layer.
#[derive(Debug, Clone, Copy)]
pub struct EncoderLayer {
    pub attn_ln: Linear,
    pub attn: Attention,
    pub mlp_ln: Linear,
    pub fc1: Linear,
    pub fc2: Linear,
}

/// One decoder layer. `cross_k`/`cross_v` are the once-per-utterance encoder
/// projections and are passed to the builder, not stored here.
#[derive(Debug, Clone, Copy)]
pub struct DecoderLayer {
    pub self_ln: Linear,
    pub self_attn: Attention,
    pub cross_ln: Linear,
    pub cross_q: Linear,
    pub cross_out: Linear,
    pub mlp_ln: Linear,
    pub fc1: Linear,
    pub fc2: Linear,
}

/// The encoder's convolution stem, position table and final norm.
#[derive(Debug, Clone, Copy)]
pub struct EncoderStem {
    pub conv1: Linear,
    pub conv2: Linear,
    pub position: NodeId,
    pub layer_norm: Linear,
}

/// A decoder's embedding tables and final norm. The vocabulary projection is
/// the token table itself, as the checkpoint ties it.
#[derive(Debug, Clone, Copy)]
pub struct DecoderStem {
    pub tokens: NodeId,
    pub position: NodeId,
    pub layer_norm: Linear,
    pub head: Linear,
}

fn f32_shape(shape: &[u32]) -> Vec<Extent> {
    shape.iter().map(|&size| Extent::Static(size)).collect()
}

fn same(
    program: &mut Vec<Op>,
    body: ScalarOp,
    a: NodeId,
    axes: &str,
) -> Result<NodeId, TensorError> {
    let map = alloc::format!("{axes}->{axes}");
    elementwise(program, ACTIVATION, body, &[(a, map.as_str())])
}

fn scale(
    program: &mut Vec<Op>,
    x: NodeId,
    scalar: NodeId,
    axes: &str,
) -> Result<NodeId, TensorError> {
    let map = alloc::format!("{axes}->{axes}");
    let broadcast = alloc::format!("->{axes}");
    elementwise(
        program,
        ACTIVATION,
        ScalarOp::Multiply,
        &[(x, map.as_str()), (scalar, broadcast.as_str())],
    )
}

fn binary(
    program: &mut Vec<Op>,
    body: ScalarOp,
    a: NodeId,
    b: NodeId,
    axes: &str,
) -> Result<NodeId, TensorError> {
    let map = alloc::format!("{axes}->{axes}");
    elementwise(
        program,
        ACTIVATION,
        body,
        &[(a, map.as_str()), (b, map.as_str())],
    )
}

/// `[s, in] @ [in, out] -> [s, out]`, with an optional `[out]` bias.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn linear(program: &mut Vec<Op>, x: NodeId, w: Linear) -> Result<NodeId, TensorError> {
    let product = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Multiply,
        &[(x, "si->sio"), (w.weight, "io->sio")],
    )?;
    let out = reduce(
        program,
        ACTIVATION,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "sio->sio",
        "so->sio",
    )?;
    match w.bias {
        Some(bias) => elementwise(
            program,
            ACTIVATION,
            ScalarOp::Add,
            &[(out, "so->so"), (bias, "o->so")],
        ),
        None => Ok(out),
    }
}

/// `[s, in] @ [in, heads*head_dim] -> [s, heads, head_dim]`.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn linear_heads(
    program: &mut Vec<Op>,
    x: NodeId,
    w: Linear,
    _head_dim: u32,
) -> Result<NodeId, TensorError> {
    let product = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Multiply,
        &[(x, "s,i->shji"), (w.weight, "ihj->shji")],
    )?;
    let out = reduce(
        program,
        ACTIVATION,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "shji->shji",
        "shj->shji",
    )?;
    match w.bias {
        Some(bias) => elementwise(
            program,
            ACTIVATION,
            ScalarOp::Add,
            &[(out, "shj->shj"), (bias, "hj->shj")],
        ),
        None => Ok(out),
    }
}

/// `[s, heads, head_dim] -> [s, heads*head_dim]`, the inverse of
/// [`linear_heads`]'s reshape. A merge has no affine read map, so a one-hot
/// selector over two `Iota`s does it.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn flatten_heads(
    program: &mut Vec<Op>,
    x: NodeId,
    heads: u32,
    head_dim: u32,
) -> Result<NodeId, TensorError> {
    let iota = |program: &mut Vec<Op>, extent: u32| {
        proxima_tensor::op::append(
            program,
            Op::Iota {
                dtype: ACTIVATION,
                extent: Extent::Static(extent),
            },
        )
    };
    let target = iota(program, heads * head_dim);
    let head = iota(program, heads);
    let channel = iota(program, head_dim);
    let stride = scalar_constant(program, head_dim as f32);
    let start = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Multiply,
        &[(head, "h->hd"), (stride, "->hd")],
    )?;
    let flat = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Add,
        &[(start, "hd->hd"), (channel, "d->hd")],
    )?;
    let select = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Equal,
        &[(target, "t->thd"), (flat, "hd->thd")],
    )?;
    let product = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Multiply,
        &[(x, "shd->sthd"), (select, "thd->sthd")],
    )?;
    reduce(
        program,
        ACTIVATION,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "sthd->sthd",
        "st->sthd",
    )
}

/// `[s, heads, head_dim] @ [heads, head_dim, out] -> [s, out]`, the
/// head-major output projection. A reduce over both head axes is a merge
/// with no separate flatten node.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn project_heads(program: &mut Vec<Op>, x: NodeId, w: Linear) -> Result<NodeId, TensorError> {
    let product = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Multiply,
        &[(x, "shd->shdo"), (w.weight, "hdo->shdo")],
    )?;
    let out = reduce(
        program,
        ACTIVATION,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "shdo->shdo",
        "so->shdo",
    )?;
    match w.bias {
        Some(bias) => elementwise(
            program,
            ACTIVATION,
            ScalarOp::Add,
            &[(out, "so->so"), (bias, "o->so")],
        ),
        None => Ok(out),
    }
}

/// `[a, b] -> [b, a]`.
///
/// # Errors
///
/// [`TensorError`] when the map or node is malformed.
pub fn transpose(program: &mut Vec<Op>, x: NodeId) -> Result<NodeId, TensorError> {
    elementwise(program, ACTIVATION, ScalarOp::Identity, &[(x, "ab->ba")])
}

/// `gelu_pytorch_tanh`: `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`, the
/// tanh form llama.cpp's `ggml_gelu` computes.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn gelu_tanh(program: &mut Vec<Op>, x: NodeId, axes: &str) -> Result<NodeId, TensorError> {
    let square = binary(program, ScalarOp::Multiply, x, x, axes)?;
    let cube = binary(program, ScalarOp::Multiply, square, x, axes)?;
    let coefficient = scalar_constant(program, 0.044_715);
    let cubic = scale(program, cube, coefficient, axes)?;
    let inner = binary(program, ScalarOp::Add, x, cubic, axes)?;
    let gain = scalar_constant(program, 0.797_884_6);
    let driven = scale(program, inner, gain, axes)?;
    let squashed = same(program, ScalarOp::Tanh, driven, axes)?;
    let one = scalar_constant(program, 1.0);
    let map = alloc::format!("{axes}->{axes}");
    let broadcast = alloc::format!("->{axes}");
    let raised = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Add,
        &[(squashed, map.as_str()), (one, broadcast.as_str())],
    )?;
    let half = scalar_constant(program, 0.5);
    let doubled = binary(program, ScalarOp::Multiply, x, raised, axes)?;
    scale(program, doubled, half, axes)
}

/// The strict causal mask `key > query` and the `-inf` it selects, both built
/// over an `n`-token sequence.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn causal_mask(program: &mut Vec<Op>, n: u32) -> Result<(NodeId, NodeId), TensorError> {
    let iota = |program: &mut Vec<Op>| {
        proxima_tensor::op::append(
            program,
            Op::Iota {
                dtype: ACTIVATION,
                extent: Extent::Static(n),
            },
        )
    };
    let query = iota(program);
    let key = iota(program);
    let future = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Greater,
        &[(key, "t->st"), (query, "s->st")],
    )?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    Ok((future, neg_infinity))
}

/// Softmax attention over `q` `[s, u, i]`, `k` `[t, u, i]` and `v` `[t, u, d]`,
/// with an optional `(is_future, neg_infinity)` causal mask. The scale is
/// `head_dim^-0.5`, so a caller passes un-scaled projections.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn attention(
    program: &mut Vec<Op>,
    q: NodeId,
    k: NodeId,
    v: NodeId,
    inv_sqrt_head_dim: NodeId,
    mask: Option<(NodeId, NodeId)>,
) -> Result<NodeId, TensorError> {
    let product = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Multiply,
        &[(q, "sui->stui"), (k, "tui->stui")],
    )?;
    let scores = reduce(
        program,
        ACTIVATION,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "stui->stui",
        "stu->stui",
    )?;
    let scaled = scale(program, scores, inv_sqrt_head_dim, "stu")?;
    let masked = match mask {
        Some((future, neg_infinity)) => elementwise(
            program,
            ACTIVATION,
            ScalarOp::Select,
            &[
                (future, "st->stu"),
                (neg_infinity, "->stu"),
                (scaled, "stu->stu"),
            ],
        )?,
        None => scaled,
    };
    let peak = reduce(
        program,
        ACTIVATION,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        masked,
        "stu->stu",
        "su->stu",
    )?;
    let shifted = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Subtract,
        &[(masked, "stu->stu"), (peak, "su->stu")],
    )?;
    let exponent = same(program, ScalarOp::Exponential, shifted, "stu")?;
    let total = reduce(
        program,
        ACTIVATION,
        ScalarOp::Add,
        ReduceInit::Zero,
        exponent,
        "stu->stu",
        "su->stu",
    )?;
    let weights = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Divide,
        &[(exponent, "stu->stu"), (total, "su->stu")],
    )?;
    let mixed = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Multiply,
        &[(weights, "stu->stud"), (v, "tud->stud")],
    )?;
    reduce(
        program,
        ACTIVATION,
        ScalarOp::Add,
        ReduceInit::Zero,
        mixed,
        "stud->stud",
        "sud->stud",
    )
}

/// One encoder layer: pre-norm attention, then pre-norm FFN, both residual.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn encoder_layer(
    program: &mut Vec<Op>,
    x: NodeId,
    layer: &EncoderLayer,
    heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<NodeId, TensorError> {
    let width = heads * head_dim;
    let normed = layer_norm(
        program,
        x,
        layer.attn_ln.weight,
        layer
            .attn_ln
            .bias
            .ok_or_else(|| missing("encoder attention norm bias"))?,
        width,
        eps,
    )?;
    let q = linear_heads(program, normed, layer.attn.q, head_dim)?;
    let k = linear_heads(program, normed, layer.attn.k, head_dim)?;
    let v = linear_heads(program, normed, layer.attn.v, head_dim)?;
    let inv = scalar_constant(program, 1.0 / (head_dim as f32).sqrt());
    let attended = attention(program, q, k, v, inv, None)?;
    let projected = project_heads(program, attended, layer.attn.out)?;
    let residual = binary(program, ScalarOp::Add, x, projected, "sd")?;
    let mlp_normed = layer_norm(
        program,
        residual,
        layer.mlp_ln.weight,
        layer
            .mlp_ln
            .bias
            .ok_or_else(|| missing("encoder mlp norm bias"))?,
        width,
        eps,
    )?;
    let up = linear(program, mlp_normed, layer.fc1)?;
    let hidden = gelu_tanh(program, up, "sd")?;
    let ffn = linear(program, hidden, layer.fc2)?;
    binary(program, ScalarOp::Add, residual, ffn, "sd")
}

/// One decoder layer: causal self-attention, cross-attention to the encoder
/// memory, then the FFN, all pre-norm and residual.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
// a graph builder takes its node ids and attention dims; a struct here would only rename the call
#[allow(clippy::too_many_arguments)]
pub fn decoder_layer(
    program: &mut Vec<Op>,
    x: NodeId,
    layer: &DecoderLayer,
    cross_k: NodeId,
    cross_v: NodeId,
    heads: u32,
    head_dim: u32,
    eps: f32,
    mask: Option<(NodeId, NodeId)>,
) -> Result<NodeId, TensorError> {
    let width = heads * head_dim;
    let normed = layer_norm(
        program,
        x,
        layer.self_ln.weight,
        layer
            .self_ln
            .bias
            .ok_or_else(|| missing("decoder self norm bias"))?,
        width,
        eps,
    )?;
    let q = linear_heads(program, normed, layer.self_attn.q, head_dim)?;
    let k = linear_heads(program, normed, layer.self_attn.k, head_dim)?;
    let v = linear_heads(program, normed, layer.self_attn.v, head_dim)?;
    let inv = scalar_constant(program, 1.0 / (head_dim as f32).sqrt());
    let attended = attention(program, q, k, v, inv, mask)?;
    let projected = project_heads(program, attended, layer.self_attn.out)?;
    let residual = binary(program, ScalarOp::Add, x, projected, "sd")?;

    let cross_normed = layer_norm(
        program,
        residual,
        layer.cross_ln.weight,
        layer
            .cross_ln
            .bias
            .ok_or_else(|| missing("decoder cross norm bias"))?,
        width,
        eps,
    )?;
    let cross_q = linear_heads(program, cross_normed, layer.cross_q, head_dim)?;
    let cross = attention(program, cross_q, cross_k, cross_v, inv, None)?;
    let cross_projected = project_heads(program, cross, layer.cross_out)?;
    let cross_residual = binary(program, ScalarOp::Add, residual, cross_projected, "sd")?;

    let mlp_normed = layer_norm(
        program,
        cross_residual,
        layer.mlp_ln.weight,
        layer
            .mlp_ln
            .bias
            .ok_or_else(|| missing("decoder mlp norm bias"))?,
        width,
        eps,
    )?;
    let up = linear(program, mlp_normed, layer.fc1)?;
    let hidden = gelu_tanh(program, up, "sd")?;
    let ffn = linear(program, hidden, layer.fc2)?;
    binary(program, ScalarOp::Add, cross_residual, ffn, "sd")
}

/// The encoder stack: conv stem, position table, `layers`, final norm.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
/// The encoder's convolutional stem: `[1, mels, frames] -> [frames/2, width]`.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn encoder_conv(
    program: &mut Vec<Op>,
    mel: NodeId,
    stem: &EncoderStem,
    mels: u32,
    width: u32,
    frames: u32,
) -> Result<NodeId, TensorError> {
    // `mel` is already `[1, mels, frames]`; the caller binds the batch axis.
    let conv1 = conv(program, mel, stem.conv1, mels, width, frames, 1)?;
    let activated1 = gelu_tanh(program, conv1, "ncl")?;
    let conv2 = conv(program, activated1, stem.conv2, width, width, frames, 2)?;
    let activated2 = gelu_tanh(program, conv2, "ncl")?;
    elementwise(
        program,
        ACTIVATION,
        ScalarOp::Identity,
        &[(activated2, "0,d,l->ld")],
    )
}

/// The encoder stack: conv stem, position table, `layers`, final norm.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
#[allow(clippy::too_many_arguments)]
pub fn encoder(
    program: &mut Vec<Op>,
    mel: NodeId,
    stem: &EncoderStem,
    layers: &[EncoderLayer],
    heads: u32,
    head_dim: u32,
    eps: f32,
    mels: u32,
    frames: u32,
) -> Result<NodeId, TensorError> {
    let width = heads * head_dim;
    let time_major = encoder_conv(program, mel, stem, mels, width, frames)?;
    encoder_after_conv(program, time_major, stem, layers, heads, head_dim, eps)
}

/// The encoder from an already-convolved `[frames/2, width]` activation: the
/// position table, the `layers` and the final norm. Split out so a golden can
/// inject the reference's own conv output and isolate the stack.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn encoder_after_conv(
    program: &mut Vec<Op>,
    time_major: NodeId,
    stem: &EncoderStem,
    layers: &[EncoderLayer],
    heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<NodeId, TensorError> {
    let width = heads * head_dim;
    let positioned = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Add,
        &[(time_major, "l,d->ld"), (stem.position, "l,d->ld")],
    )?;
    let mut x = positioned;
    for layer in layers {
        x = encoder_layer(program, x, layer, heads, head_dim, eps)?;
    }
    layer_norm(
        program,
        x,
        stem.layer_norm.weight,
        stem.layer_norm
            .bias
            .ok_or_else(|| missing("encoder final norm bias"))?,
        width,
        eps,
    )
}

/// A 1-D convolution over `[1, channels, length]` with symmetric zero padding.
/// The workspace already carries this trainable lowering in `proxima-autograd`;
/// the audio stem uses it rather than a second construction.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
#[allow(clippy::too_many_arguments)]
fn conv(
    program: &mut Vec<Op>,
    signal: NodeId,
    weights: Linear,
    in_channels: u32,
    out_channels: u32,
    length: u32,
    stride: u64,
) -> Result<NodeId, TensorError> {
    let signal_shape = (1_u64, u64::from(in_channels), u64::from(length));
    let weight_shape = (u64::from(out_channels), u64::from(in_channels), 3_u64);
    proxima_autograd::conv::conv1d(
        program,
        ACTIVATION,
        signal,
        signal_shape,
        weights.weight,
        weight_shape,
        weights.bias,
        stride,
        1,
        1,
    )
    .ok_or_else(|| missing("conv1d shape"))
}

/// The decoder stack over `tokens` `[s]`: embed tokens and positions, run
/// `layers` with the causal mask, final norm, then the vocabulary head.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
// a graph builder takes its node ids and attention dims; a struct here would only rename the call
#[allow(clippy::too_many_arguments)]
pub fn decoder(
    program: &mut Vec<Op>,
    tokens: NodeId,
    positions: NodeId,
    stem: &DecoderStem,
    layers: &[DecoderLayer],
    cross: &[(NodeId, NodeId)],
    heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<NodeId, TensorError> {
    let token_rows = embedding_lookup(program, stem.tokens, tokens);
    let position_rows = embedding_lookup(program, stem.position, positions);
    let mut x = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Add,
        &[(token_rows, "sd->sd"), (position_rows, "sd->sd")],
    )?;
    let n = token_count(program, tokens)?;
    let (future, neg_infinity) = causal_mask(program, n)?;
    for (index, layer) in layers.iter().enumerate() {
        let (cross_k, cross_v) = cross
            .get(index)
            .copied()
            .ok_or_else(|| missing("a cross-KV pair per decoder layer"))?;
        x = decoder_layer(
            program,
            x,
            layer,
            cross_k,
            cross_v,
            heads,
            head_dim,
            eps,
            Some((future, neg_infinity)),
        )?;
    }
    let normed = layer_norm(
        program,
        x,
        stem.layer_norm.weight,
        stem.layer_norm
            .bias
            .ok_or_else(|| missing("decoder final norm bias"))?,
        heads * head_dim,
        eps,
    )?;
    linear(program, normed, stem.head)
}

/// The cross-attention K/V of one decoder layer, projected from `memory`
/// `[t, width]` and split into `[t, heads, head_dim]`. The reference stores K
/// scaled by `head_dim^-0.25`; this builder returns the un-scaled projection
/// because [`attention`] applies `head_dim^-0.5` to the scores.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn cross_kv(
    program: &mut Vec<Op>,
    memory: NodeId,
    k: Linear,
    v: Linear,
    head_dim: u32,
) -> Result<(NodeId, NodeId), TensorError> {
    Ok((
        linear_heads(program, memory, k, head_dim)?,
        linear_heads(program, memory, v, head_dim)?,
    ))
}

fn missing(what: &'static str) -> TensorError {
    TensorError::NotLowerable {
        node: NodeId(0),
        reason: what,
    }
}

fn token_count(program: &[Op], tokens: NodeId) -> Result<u32, TensorError> {
    let op = program
        .get(tokens.0 as usize)
        .ok_or_else(|| missing("token leaf is not in the program"))?;
    match op {
        Op::Input { shape, .. } => match shape.first() {
            Some(Extent::Static(n)) => Ok(*n),
            _ => Err(missing("token leaf has no static length")),
        },
        _ => Err(missing("token node is not an input leaf")),
    }
}

/// Declare an F32 input leaf, the shape the host binds weights or ids to.
#[must_use]
pub fn leaf(program: &mut Vec<Op>, shape: &[u32], name: &str) -> NodeId {
    input_leaf(program, ACTIVATION, f32_shape(shape), name)
}
