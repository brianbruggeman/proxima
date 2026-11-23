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
    CachedLayerRoots, PLANE_CACHE_READS, PLANE_NEW_READS, RopePairing, causal_mask, elementwise,
    embedding_lookup, input_leaf, layer_norm, non_causal_attention, reduce, scalar_constant,
    two_range_attention, unrotated_head_pair,
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

/// `[s, heads, 1, head_dim] @ [heads, head_dim, out] -> [s, out]`, the
/// head-major output projection over an attention result. A reduce over the
/// head, group and channel axes is a merge with no separate flatten node.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed.
pub fn project_heads(program: &mut Vec<Op>, x: NodeId, w: Linear) -> Result<NodeId, TensorError> {
    let product = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Multiply,
        &[(x, "sugd->sugdo"), (w.weight, "udo->sugdo")],
    )?;
    let out = reduce(
        program,
        ACTIVATION,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "sugdo->sugdo",
        "so->sugdo",
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
    let attended = non_causal_attention(program, q, k, v, inv)?;
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
/// Self-attention is `two_range_attention` over `cache` (the earlier
/// positions' key half planes and values, extent `Symbolic(1)`, or an empty
/// range) and this call's own keys, split into two plain halves of the head
/// width. `is_future` and `group_ones` are the shared causal mask and the
/// `[heads, 1]` ones leaf. Returns the next activation and this call's key
/// halves and values, which a cached driver appends to the cache.
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
    is_future: NodeId,
    group_ones: NodeId,
    cache: CachedLayerRoots,
) -> Result<(NodeId, CachedLayerRoots), TensorError> {
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
    let pairs = head_dim / 2;
    let split = RopePairing::SplitHalf { pairs };
    let (q_first, q_second) = unrotated_head_pair(program, q, 'h', split, pairs)?;
    let (k_first, k_second) = unrotated_head_pair(program, k, 'u', split, pairs)?;
    let (attended, _) = two_range_attention(
        program,
        ACTIVATION,
        (q_first, q_second),
        (k_first, k_second),
        v,
        (cache.0, cache.1),
        cache.2,
        inv,
        None,
        is_future,
        None,
        false,
        None,
        1,
        group_ones,
        PLANE_CACHE_READS,
        PLANE_NEW_READS,
    )?;
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
    let cross = non_causal_attention(program, cross_q, cross_k, cross_v, inv)?;
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
    let next = binary(program, ScalarOp::Add, cross_residual, ffn, "sd")?;
    Ok((next, (k_first, k_second, v)))
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

/// The decoder stack over `tokens` `[s]` (extent `Symbolic(0)`): embed tokens
/// and positions, run `layers` with the causal mask, final norm, then the
/// vocabulary head.
///
/// `self_cache` holds one `(k_first, k_second, v)` leaf triple per layer,
/// the earlier positions' key half planes `[t, heads, head_dim/2]` and values
/// `[t, heads, head_dim]` at `Extent::Symbolic(1)`. `None` caches an empty
/// range, so the graph is the teacher-forced one over the whole sequence.
/// Returns the logits and one `(k_first, k_second, v)` triple per layer for
/// this call's positions, which a cached driver appends to its cache.
///
/// # Errors
///
/// [`TensorError`] when a map or node is malformed, or `head_dim` is odd.
// a graph builder takes its node ids and attention dims; a struct here would only rename the call
#[allow(clippy::too_many_arguments)]
pub fn decoder(
    program: &mut Vec<Op>,
    tokens: NodeId,
    positions: NodeId,
    stem: &DecoderStem,
    layers: &[DecoderLayer],
    cross: &[(NodeId, NodeId)],
    self_cache: Option<&[CachedLayerRoots]>,
    heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<(NodeId, Vec<CachedLayerRoots>), TensorError> {
    if !head_dim.is_multiple_of(2) {
        return Err(missing("an even head width to split into two halves"));
    }
    let token_rows = embedding_lookup(program, stem.tokens, tokens);
    let position_rows = embedding_lookup(program, stem.position, positions);
    let mut x = elementwise(
        program,
        ACTIVATION,
        ScalarOp::Add,
        &[(token_rows, "sd->sd"), (position_rows, "sd->sd")],
    )?;
    let (is_future, _) = causal_mask(program)?;
    let group_ones = proxima_tensor::op::append(
        program,
        Op::Constant {
            dtype: ACTIVATION,
            shape: alloc::vec![Extent::Static(heads), Extent::Static(1)],
            value: 1.0,
        },
    );
    let empty = empty_cache(program, heads, head_dim);
    let mut appended = Vec::with_capacity(layers.len());
    for (index, layer) in layers.iter().enumerate() {
        let (cross_k, cross_v) = cross
            .get(index)
            .copied()
            .ok_or_else(|| missing("a cross-KV pair per decoder layer"))?;
        let cache = match self_cache {
            Some(leaves) => leaves
                .get(index)
                .copied()
                .ok_or_else(|| missing("a self-attention cache triple per decoder layer"))?,
            None => empty,
        };
        let (next, new_roots) = decoder_layer(
            program, x, layer, cross_k, cross_v, heads, head_dim, eps, is_future, group_ones, cache,
        )?;
        x = next;
        appended.push(new_roots);
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
    let logits = linear(program, normed, stem.head)?;
    Ok((logits, appended))
}

fn empty_cache(program: &mut Vec<Op>, heads: u32, head_dim: u32) -> CachedLayerRoots {
    let mut plane = |width: u32| {
        proxima_tensor::op::append(
            program,
            Op::Constant {
                dtype: ACTIVATION,
                shape: alloc::vec![
                    Extent::Static(0),
                    Extent::Static(heads),
                    Extent::Static(width)
                ],
                value: 0.0,
            },
        )
    };
    let first = plane(head_dim / 2);
    let second = plane(head_dim / 2);
    (first, second, plane(head_dim))
}

fn missing(what: &'static str) -> TensorError {
    TensorError::NotLowerable {
        node: NodeId(0),
        reason: what,
    }
}

/// Declare an F32 input leaf, the shape the host binds weights or ids to.
#[must_use]
pub fn leaf(program: &mut Vec<Op>, shape: &[u32], name: &str) -> NodeId {
    input_leaf(program, ACTIVATION, f32_shape(shape), name)
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::String;
    use alloc::{format, vec};
    use core::ops::{Div, Mul};

    use proxima_autograd::adjoint::differentiate_bound;
    use proxima_numeric::NumericPolicy;
    use proxima_tensor::config::TensorExecutionConfig;
    use proxima_tensor::cpu::{
        BufferPools, Evaluated, QuantizedBlock, evaluate_quantized_named_with_config,
    };

    use super::*;

    const HEADS: u32 = 2;
    const HEAD_DIM: u32 = 4;
    const WIDTH: u32 = HEADS * HEAD_DIM;
    const FFN: u32 = 16;
    const LAYERS: usize = 2;
    const SOURCE: u32 = 5;
    const VOCAB: u32 = 11;
    const POSITIONS: u32 = 8;
    const PROMPT: [u8; 3] = [1, 2, 3];
    const STEPS: usize = 6;

    struct Graph {
        program: Vec<Op>,
        declared: Vec<(String, usize)>,
    }

    fn seed_of(name: &str) -> u64 {
        name.bytes().fold(0xcbf2_9ce4_8422_2325, |state, byte| {
            (state ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    fn noise<T: From<i16> + Div<Output = T>>(name: &str, count: usize) -> Vec<T> {
        let mut state = seed_of(name);
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let raw =
                    i16::try_from((state >> 40) % 1000).expect("a value below 1000 fits") - 500;
                T::from(raw) / T::from(1000)
            })
            .collect()
    }

    fn sample<T: From<u8>>(id: u8) -> T {
        T::from(id)
    }

    fn activation_mantissa_bits() -> u8 {
        match ACTIVATION {
            DType::Layout(layout) => layout.mantissa_bits,
            _ => panic!("the activation is a layout"),
        }
    }

    // the longest sum in the decoder graph under test: the model width, the feed-forward width,
    // the cross source positions or the attended rows
    fn longest_reduction() -> u8 {
        let rows = u32::try_from(STEPS).expect("a short run");
        let longest = [WIDTH, FFN, SOURCE, rows].into_iter().max();
        u8::try_from(longest.expect("four extents")).expect("a small config")
    }

    // Higham's forward-error bound for a sum of n terms is gamma_n, about n * u, with u the unit
    // roundoff 2^-mantissa_bits of the activation layout
    fn tolerance_unit<T>() -> T
    where
        T: From<u8> + Div<Output = T> + Mul<Output = T>,
    {
        let roundoff =
            (0..activation_mantissa_bits()).fold(T::from(1), |scale, _| scale / T::from(2));
        roundoff * T::from(longest_reduction())
    }

    impl Graph {
        fn new() -> Self {
            Self {
                program: Vec::new(),
                declared: Vec::new(),
            }
        }

        fn leaf(&mut self, name: &str, shape: &[u32]) -> NodeId {
            let count = shape.iter().product::<u32>() as usize;
            self.declared.push((name.into(), count));
            leaf(&mut self.program, shape, name)
        }

        fn symbolic(&mut self, name: &str, extents: &[Extent]) -> NodeId {
            input_leaf(&mut self.program, ACTIVATION, extents.to_vec(), name)
        }

        fn norm(&mut self, name: &str) -> Linear {
            Linear {
                weight: self.leaf(&format!("{name}.weight"), &[WIDTH]),
                bias: Some(self.leaf(&format!("{name}.bias"), &[WIDTH])),
            }
        }

        fn projection(&mut self, name: &str, input: u32, output: u32) -> Linear {
            Linear {
                weight: self.leaf(&format!("{name}.weight"), &[input, output]),
                bias: Some(self.leaf(&format!("{name}.bias"), &[output])),
            }
        }

        fn heads(&mut self, name: &str, bias: bool) -> Linear {
            Linear {
                weight: self.leaf(&format!("{name}.weight"), &[WIDTH, HEADS, HEAD_DIM]),
                bias: bias.then(|| self.leaf(&format!("{name}.bias"), &[HEADS, HEAD_DIM])),
            }
        }

        fn heads_out(&mut self, name: &str) -> Linear {
            Linear {
                weight: self.leaf(&format!("{name}.weight"), &[HEADS, HEAD_DIM, WIDTH]),
                bias: Some(self.leaf(&format!("{name}.bias"), &[WIDTH])),
            }
        }

        fn layer(&mut self, index: usize) -> DecoderLayer {
            let prefix = format!("layer{index}");
            DecoderLayer {
                self_ln: self.norm(&format!("{prefix}.self_ln")),
                self_attn: Attention {
                    q: self.heads(&format!("{prefix}.self_q"), true),
                    k: self.heads(&format!("{prefix}.self_k"), false),
                    v: self.heads(&format!("{prefix}.self_v"), true),
                    out: self.heads_out(&format!("{prefix}.self_out")),
                },
                cross_ln: self.norm(&format!("{prefix}.cross_ln")),
                cross_q: self.heads(&format!("{prefix}.cross_q"), true),
                cross_out: self.heads_out(&format!("{prefix}.cross_out")),
                mlp_ln: self.norm(&format!("{prefix}.mlp_ln")),
                fc1: self.projection(&format!("{prefix}.fc1"), WIDTH, FFN),
                fc2: self.projection(&format!("{prefix}.fc2"), FFN, WIDTH),
            }
        }

        fn stem(&mut self) -> DecoderStem {
            DecoderStem {
                tokens: self.leaf("tokens_table", &[VOCAB, WIDTH]),
                position: self.leaf("position_table", &[POSITIONS, WIDTH]),
                layer_norm: self.norm("final_ln"),
                head: Linear {
                    weight: self.leaf("head.weight", &[WIDTH, VOCAB]),
                    bias: None,
                },
            }
        }

        fn cross(&mut self) -> Vec<(NodeId, NodeId)> {
            (0..LAYERS)
                .map(|index| {
                    let shape = [SOURCE, HEADS, HEAD_DIM];
                    (
                        self.leaf(&format!("cross_k_{index}"), &shape),
                        self.leaf(&format!("cross_v_{index}"), &shape),
                    )
                })
                .collect()
        }

        fn cache_leaves(&mut self) -> Vec<CachedLayerRoots> {
            let rows = Extent::Symbolic(1);
            let pairs = [rows, Extent::Static(HEADS), Extent::Static(HEAD_DIM / 2)];
            let values = [rows, Extent::Static(HEADS), Extent::Static(HEAD_DIM)];
            (0..LAYERS)
                .map(|index| {
                    (
                        self.symbolic(&format!("cache_first_{index}"), &pairs),
                        self.symbolic(&format!("cache_second_{index}"), &pairs),
                        self.symbolic(&format!("cache_value_{index}"), &values),
                    )
                })
                .collect()
        }

        fn decode(&mut self, cached: bool) -> (NodeId, Vec<CachedLayerRoots>) {
            let tokens = self.symbolic("tokens", &[Extent::Symbolic(0)]);
            let positions = self.symbolic("positions", &[Extent::Symbolic(0)]);
            let stem = self.stem();
            let layers: Vec<_> = (0..LAYERS).map(|index| self.layer(index)).collect();
            let cross = self.cross();
            let cache = cached.then(|| self.cache_leaves());
            decoder(
                &mut self.program,
                tokens,
                positions,
                &stem,
                &layers,
                &cross,
                cache.as_deref(),
                HEADS,
                HEAD_DIM,
                1e-5,
            )
            .expect("the synthetic decoder builds")
        }

        fn evaluate(
            &self,
            symbols: &[u64],
            ids: &[u8],
            first_position: u8,
            extras: &[(&str, QuantizedBlock<'_>)],
            outputs: &[NodeId],
        ) -> Evaluated {
            self.evaluate_in(&self.program, symbols, ids, first_position, extras, outputs)
        }

        // `extras` come first so a bound value there replaces the seeded weight of that name
        fn evaluate_in(
            &self,
            program: &[Op],
            symbols: &[u64],
            ids: &[u8],
            first_position: u8,
            extras: &[(&str, QuantizedBlock<'_>)],
            outputs: &[NodeId],
        ) -> Evaluated {
            let weights: Vec<(&str, Vec<_>)> = self
                .declared
                .iter()
                .map(|(name, count)| (name.as_str(), noise(name, *count)))
                .collect();
            let tokens: Vec<_> = ids.iter().map(|id| sample(*id)).collect();
            let positions: Vec<_> = (0..ids.len())
                .map(|offset| sample(first_position + u8::try_from(offset).expect("a short run")))
                .collect();
            let mut named: Vec<(&str, QuantizedBlock<'_>)> = extras.to_vec();
            named.extend(
                weights
                    .iter()
                    .map(|(name, values)| (*name, QuantizedBlock::from(values.as_slice()))),
            );
            named.push(("tokens", QuantizedBlock::from(tokens.as_slice())));
            named.push(("positions", QuantizedBlock::from(positions.as_slice())));
            // the default policy fuses a cached range into llama's f16 flash arithmetic, which
            // the unfused full recompute cannot state; the f32 online kernel can
            let config = TensorExecutionConfig::builder()
                .numeric(NumericPolicy::bit_exact().with_flash_narrowing(true))
                .build();
            let mut pools = BufferPools::default();
            let mut validated = None;
            evaluate_quantized_named_with_config(
                program,
                symbols,
                &named,
                outputs,
                (&mut pools, &mut validated),
                None,
                &config,
            )
            .expect("the synthetic graph evaluates")
        }
    }

    fn cache_names() -> Vec<String> {
        (0..LAYERS)
            .flat_map(|layer| {
                ["first", "second", "value"].map(|part| format!("cache_{part}_{layer}"))
            })
            .collect()
    }

    fn flat_outputs(logits: NodeId, appended: &[CachedLayerRoots]) -> Vec<NodeId> {
        let mut outputs = vec![logits];
        outputs.extend(
            appended
                .iter()
                .flat_map(|(first, second, value)| [*first, *second, *value]),
        );
        outputs
    }

    #[test]
    fn cached_one_token_steps_match_the_full_recompute_last_row() {
        let mut full = Graph::new();
        let (full_logits, _) = full.decode(false);
        let mut step = Graph::new();
        let (step_logits, appended) = step.decode(true);
        let outputs = flat_outputs(step_logits, &appended);
        let names = cache_names();

        let mut tokens = PROMPT.to_vec();
        let mut cache: Vec<Vec<_>> = vec![Vec::new(); LAYERS * 3];
        let mut compared = 0;
        let vocab = VOCAB as usize;
        for index in 0..STEPS {
            let rows = index + 1;
            let symbols = [rows as u64, 0];
            let evaluated_full = full.evaluate(&symbols, &tokens[..rows], 0, &[], &[full_logits]);
            let (all_rows, _) = evaluated_full.get(full_logits).expect("full logits");
            let expected = &all_rows[index * vocab..rows * vocab];

            let extras: Vec<(&str, QuantizedBlock<'_>)> = names
                .iter()
                .zip(&cache)
                .map(|(name, plane)| (name.as_str(), QuantizedBlock::from(plane.as_slice())))
                .collect();
            let first_position = u8::try_from(index).expect("a short run");
            let evaluated_step = step.evaluate(
                &[1, index as u64],
                &tokens[index..rows],
                first_position,
                &extras,
                &outputs,
            );
            let (actual, _) = evaluated_step.get(step_logits).expect("step logits");
            assert_eq!(actual.len(), vocab, "a one-token step emits one logits row");
            let unit = tolerance_unit();
            for (column, (found, wanted)) in actual.iter().zip(expected).enumerate() {
                let tolerance = unit * (1.0 + wanted.abs());
                assert!(
                    (found - wanted).abs() <= tolerance,
                    "step {index} logit {column}: cached {found} against full {wanted} (tolerance {tolerance})"
                );
            }
            let spread = expected
                .iter()
                .fold(0.0, |widest, value| value.abs().max(widest));
            assert!(
                spread > unit,
                "step {index} logits are all within roundoff of zero"
            );
            for (plane, node) in cache.iter_mut().zip(&outputs[1..]) {
                let (rows_out, _) = evaluated_step.get(*node).expect("appended rows");
                plane.extend_from_slice(rows_out);
            }
            if tokens.len() == rows {
                let (best, _) = expected
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .expect("a logits row is not empty");
                tokens.push(u8::try_from(best).expect("the vocabulary is below 256"));
            }
            compared += 1;
        }
        assert_eq!(compared, STEPS, "every step is compared");
    }

    #[test]
    fn an_emptied_cache_changes_the_logits() {
        let mut full = Graph::new();
        let (full_logits, _) = full.decode(false);
        let mut step = Graph::new();
        let (step_logits, appended) = step.decode(true);
        let outputs = flat_outputs(step_logits, &appended);

        let rows = PROMPT.len();
        let evaluated_full = full.evaluate(&[rows as u64, 0], &PROMPT, 0, &[], &[full_logits]);
        let (all_rows, _) = evaluated_full.get(full_logits).expect("full logits");
        let expected = &all_rows[(rows - 1) * VOCAB as usize..];
        let first_position = u8::try_from(rows - 1).expect("a short run");
        let names = cache_names();
        let empty: Vec<_> = Vec::new();
        let extras: Vec<(&str, QuantizedBlock<'_>)> = names
            .iter()
            .map(|name| (name.as_str(), QuantizedBlock::from(empty.as_slice())))
            .collect();
        let evaluated_step = step.evaluate(
            &[1, 0],
            &PROMPT[rows - 1..],
            first_position,
            &extras,
            &outputs,
        );
        let (actual, _) = evaluated_step.get(step_logits).expect("step logits");
        let unit = tolerance_unit();
        let widest = actual
            .iter()
            .zip(expected)
            .fold(0.0, |gap, (found, wanted)| (found - wanted).abs().max(gap));
        assert!(
            widest > unit,
            "dropping two cached positions must move the logits, widest gap {widest}"
        );
    }

    #[test]
    fn autograd_matches_central_differences_on_the_teacher_forced_decoder() {
        let mut graph = Graph::new();
        let (logits, _) = graph.decode(false);
        let squared = elementwise(
            &mut graph.program,
            ACTIVATION,
            ScalarOp::Multiply,
            &[(logits, "sv->sv"), (logits, "sv->sv")],
        )
        .expect("the squared logits build");
        let loss = reduce(
            &mut graph.program,
            ACTIVATION,
            ScalarOp::Add,
            ReduceInit::Zero,
            squared,
            "sv->sv",
            "->sv",
        )
        .expect("the loss builds");
        let symbols = [PROMPT.len() as u64, 0];
        let differentiated = differentiate_bound(&graph.program, loss, &symbols)
            .expect("the teacher-forced decoder differentiates");
        let watched = ["layer0.self_q.weight", "layer0.cross_q.weight"];
        let gradients: Vec<NodeId> = watched
            .iter()
            .map(|name| {
                differentiated
                    .gradient_of_named(name)
                    .expect("the weight feeds the loss")
            })
            .collect();
        let analytic = graph.evaluate_in(
            &differentiated.program,
            &symbols,
            &PROMPT,
            0,
            &[],
            &gradients,
        );
        let base = graph.evaluate(&symbols, &PROMPT, 0, &[], &[loss]);
        let (loss_row, _) = base.get(loss).expect("loss");
        let loss_value = loss_row[0];
        let unit = tolerance_unit();
        assert!(
            loss_value.abs() > unit,
            "the loss {loss_value} is within roundoff of zero"
        );
        // the step balances the truncation term step^2 against the roundoff term unit / step
        let step = unit.cbrt();

        let mut checked = 0;
        let mut informative = 0;
        for (name, gradient) in watched.iter().zip(&gradients) {
            let (derived, _) = analytic.get(*gradient).expect("gradient");
            let original = noise(name, derived.len());
            let loss_with = |index: usize, delta| {
                let mut weights = original.clone();
                weights[index] += delta;
                let swapped = [(*name, QuantizedBlock::from(weights.as_slice()))];
                let moved = graph.evaluate(&symbols, &PROMPT, 0, &swapped, &[loss]);
                moved.get(loss).expect("loss").0[0]
            };
            let stride = (derived.len() / 10).max(1);
            for index in (0..derived.len()).step_by(stride) {
                let numeric = (loss_with(index, step) - loss_with(index, -step)) / (step + step);
                let tolerance =
                    unit * (1.0 + loss_value.abs()) / step + step * step * (1.0 + numeric.abs());
                assert!(
                    (derived[index] - numeric).abs() <= tolerance,
                    "{name}[{index}]: autograd {} against central difference {numeric} (tolerance {tolerance})",
                    derived[index]
                );
                checked += 1;
                if numeric.abs() > tolerance {
                    informative += 1;
                }
            }
        }
        assert!(
            checked >= watched.len(),
            "at least one index per weight is checked"
        );
        assert!(
            informative > 0,
            "some checked gradient must exceed the tolerance, or the check proves nothing"
        );
    }

    #[test]
    fn an_odd_head_width_is_refused() {
        let mut graph = Graph::new();
        let tokens = graph.symbolic("tokens", &[Extent::Symbolic(0)]);
        let positions = graph.symbolic("positions", &[Extent::Symbolic(0)]);
        let stem = graph.stem();
        let refused = decoder(
            &mut graph.program,
            tokens,
            positions,
            &stem,
            &[],
            &[],
            None,
            HEADS,
            HEAD_DIM + 1,
            1e-5,
        );
        assert!(
            refused.is_err(),
            "an odd head width cannot split into two halves"
        );
    }
}
