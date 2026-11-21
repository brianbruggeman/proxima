//! Model-agnostic lowerings for diffusion-style backbones: a norm with no
//! learned affine, adaptive modulation, bidirectional attention, multi-axis
//! RoPE tables and a layer-tap stack.
//!
//! Every builder composes the existing algebra ([`elementwise_like`],
//! [`reduce_like`], [`scalar_constant_like`], [`super::fused_rope_pair`]) and nothing
//! else, so each is `Elementwise`/`Reduce`/`Iota`/`Constant` and autograd can
//! differentiate it. Element types are read off the nodes passed in, never
//! named here. Learn the pattern from [`super::rmsnorm`] (the affine, root-mean-square
//! sibling of [`layer_norm_no_affine`]) and from the attention core shared with
//! [`super::append_attention_mixer`], whose causal mask is the switch
//! [`non_causal_attention`] leaves off.

use alloc::vec::Vec;

use crate::error::TensorError;
use crate::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};
use proxima_numeric::{DType, FormatCodec};

use super::attention_forward::append_masked_softmax_mix;
use super::{
    elementwise_like, input_leaf, produced_dtype_or_binary32, reduce_like, scalar_constant_like,
};

fn row_mean(program: &mut Vec<Op>, x: NodeId, inv_width: NodeId) -> Result<NodeId, TensorError> {
    let sum = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        x,
        "sd->sd",
        "s->sd",
    )?;
    elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(sum, "s->s"), (inv_width, "->s")],
    )
}

/// `(x - mean(x)) / sqrt(var(x) + eps)` over the last axis of `x` (`[s, d]`,
/// `d == width`), with no learned scale or shift: the DiT pre-modulation norm,
/// and the `elementwise_affine=false` case of LayerNorm. Compose
/// [`modulate`] after it for adaLN; [`super::rmsnorm`] is the centred-less affine
/// sibling.
///
/// # Errors
///
/// [`TensorError`] when `x` is not a node of `program`.
pub fn layer_norm_no_affine(
    program: &mut Vec<Op>,
    x: NodeId,
    width: u32,
    eps: f32,
) -> Result<NodeId, TensorError> {
    let inv_width = scalar_constant_like(program, 1.0 / width as f32, x)?;
    let mean = row_mean(program, x, inv_width)?;
    let centered = elementwise_like(
        program,
        x,
        ScalarOp::Subtract,
        &[(x, "sd->sd"), (mean, "s->sd")],
    )?;
    let squared = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(centered, "sd->sd"), (centered, "sd->sd")],
    )?;
    let variance = row_mean(program, squared, inv_width)?;
    let eps_node = scalar_constant_like(program, eps, x)?;
    let spread = elementwise_like(
        program,
        x,
        ScalarOp::Add,
        &[(variance, "s->s"), (eps_node, "->s")],
    )?;
    let deviation = elementwise_like(program, x, ScalarOp::SquareRoot, &[(spread, "s->s")])?;
    let inverse = elementwise_like(program, x, ScalarOp::Reciprocal, &[(deviation, "s->s")])?;
    elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(centered, "sd->sd"), (inverse, "s->sd")],
    )
}

/// Affine LayerNorm: `layer_norm_no_affine(x) * weight + bias` over the last axis of
/// `x` (`[s, d]`), with `weight` and `bias` one `[d]` vector each broadcast over `s`.
/// Composes [`layer_norm_no_affine`] with an `Elementwise` multiply and add (the
/// `"d->sd"` operand map [`modulate`] uses); no new `Reduce`, so autograd differentiates
/// it with the rules it already has. Reach for [`layer_norm_no_affine`] when the norm
/// carries no learned parameters.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn layer_norm(
    program: &mut Vec<Op>,
    x: NodeId,
    weight: NodeId,
    bias: NodeId,
    width: u32,
    eps: f32,
) -> Result<NodeId, TensorError> {
    let normed = layer_norm_no_affine(program, x, width, eps)?;
    let scaled = elementwise_like(
        program,
        x,
        ScalarOp::Multiply,
        &[(normed, "sd->sd"), (weight, "d->sd")],
    )?;
    elementwise_like(
        program,
        x,
        ScalarOp::Add,
        &[(scaled, "sd->sd"), (bias, "d->sd")],
    )
}

/// adaLN modulation: `normed * (1 + scale) + shift`, with `shift` and `scale`
/// one `[d]` vector each (per sample, broadcast over the sequence axis).
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn modulate(
    program: &mut Vec<Op>,
    normed: NodeId,
    shift: NodeId,
    scale: NodeId,
) -> Result<NodeId, TensorError> {
    let one = scalar_constant_like(program, 1.0, normed)?;
    let gain = elementwise_like(
        program,
        normed,
        ScalarOp::Add,
        &[(scale, "d->d"), (one, "->d")],
    )?;
    let scaled = elementwise_like(
        program,
        normed,
        ScalarOp::Multiply,
        &[(normed, "sd->sd"), (gain, "d->sd")],
    )?;
    elementwise_like(
        program,
        normed,
        ScalarOp::Add,
        &[(scaled, "sd->sd"), (shift, "d->sd")],
    )
}

/// Softmax attention with every key visible to every query: `q` is
/// `[s, u, i]`, `k` is `[t, u, i]`, `v` is `[t, u, d]`, the result is
/// `[s, u, 1, d]` (row-major identical to `[s, u, d]`). `s` and `t` are
/// independent extents, so this also serves cross and joint attention. The
/// scale/softmax/mix is `append_masked_softmax_mix`, the very core
/// [`super::append_attention_mixer`] runs with its causal mask; here the mask is
/// `None`. `inv_sqrt_head_dim` is a rank-0 node (e.g. [`scalar_constant_like`]).
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn non_causal_attention(
    program: &mut Vec<Op>,
    q: NodeId,
    k: NodeId,
    v: NodeId,
    inv_sqrt_head_dim: NodeId,
) -> Result<NodeId, TensorError> {
    let group_dtype = produced_dtype_or_binary32(program, q);
    let single_group = op::append(
        program,
        Op::Constant {
            dtype: group_dtype,
            shape: alloc::vec![Extent::Static(1)],
            value: 1.0,
        },
    );
    let grouped = elementwise_like(
        program,
        q,
        ScalarOp::Multiply,
        &[(q, "sui->sugi"), (single_group, "g->sugi")],
    )?;
    let product = elementwise_like(
        program,
        q,
        ScalarOp::Multiply,
        &[(grouped, "sugi->stugi"), (k, "tui->stugi")],
    )?;
    let scores = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    append_masked_softmax_mix(program, scores, inv_sqrt_head_dim, None, None, 1, v)
}

/// One head-split projection of the encoder memory: `memory` is `[t, m]`,
/// `weight` is `[m, u, o]`, the result is `[t, u, o]`; an optional `bias`
/// (`[u, o]`) is added per position. A multiply and a `Reduce` over `m`, so
/// autograd differentiates it with the rules it already has.
fn project_memory(
    program: &mut Vec<Op>,
    memory: NodeId,
    weight: NodeId,
    bias: Option<NodeId>,
) -> Result<NodeId, TensorError> {
    let product = elementwise_like(
        program,
        memory,
        ScalarOp::Multiply,
        &[(memory, "tm->tmuo"), (weight, "muo->tmuo")],
    )?;
    let projected = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "tmuo->tmuo",
        "tuo->tmuo",
    )?;
    match bias {
        Some(bias) => elementwise_like(
            program,
            memory,
            ScalarOp::Add,
            &[(projected, "tuo->tuo"), (bias, "uo->tuo")],
        ),
        None => Ok(projected),
    }
}

/// The once-per-utterance cross-attention K and V: both projected from the
/// encoder `memory` (`[t, m]`) with `k_weight` (`[m, u, i]`) and `v_weight`
/// (`[m, u, d]`), giving `k` as `[t, u, i]` and `v` as `[t, u, d]`. A bias is an
/// operand or it is absent (`None`); a model whose K projection has no bias
/// passes `None`, nothing branches on a name.
///
/// Feed the pair to [`non_causal_attention`] (`s != t`). Inference binds the
/// cached `k`/`v` tensors as `Input`s in place of this builder's output;
/// training calls this builder in the same graph, so the adjoint flows through
/// the projection to the memory and the weights. The attention ops are the same
/// either way: there is no inference-only op.
///
/// # Errors
///
/// [`TensorError`] when a node is not part of `program`.
pub fn cross_kv(
    program: &mut Vec<Op>,
    memory: NodeId,
    k_weight: NodeId,
    k_bias: Option<NodeId>,
    v_weight: NodeId,
    v_bias: Option<NodeId>,
) -> Result<(NodeId, NodeId), TensorError> {
    let k = project_memory(program, memory, k_weight, k_bias)?;
    let v = project_memory(program, memory, v_weight, v_bias)?;
    Ok((k, v))
}

fn encode_table(values: &[f32], dtype: DType) -> Result<Vec<u8>, TensorError> {
    let unsupported = TensorError::NotLowerable {
        node: NodeId(0),
        reason: "rope table dtype has no scalar codec",
    };
    let format = dtype.format().ok_or(unsupported.clone())?;
    let codec = FormatCodec::for_format(&format).ok_or(unsupported.clone())?;
    let mut bytes = alloc::vec![0u8; codec.packed_len(values.len()).ok_or(unsupported)?];
    codec.encode(values, &mut bytes)?;
    Ok(bytes)
}

/// Host-side `(cos, sin)` tables for RoPE whose head dimension is split into
/// axis sections (`sections[a]` channels rotate by position stream `a` at base
/// `thetas[a]`), encoded as `dtype` bytes through [`FormatCodec`], each
/// `[rows, sum(sections) / 2]`. `positions` is `[rows, sections.len()]`
/// row-major; `dtype` is the table leaf's own element type. Feed the tables to
/// [`super::fused_rope_pair`] with `RopePairing::Interleaved`: that builder already
/// rotates adjacent channel pairs, and each axis owns a contiguous pair range,
/// so the IR is the existing one and only the table is new. Single-axis RoPE
/// is `sections.len() == 1`.
///
/// # Errors
///
/// [`TensorError::NotLowerable`] when a section is odd, `thetas` and
/// `sections` differ in length, `positions` is not a whole number of rows, or
/// `dtype` cannot be encoded.
pub fn multi_axis_rope_tables(
    positions: &[f32],
    sections: &[u32],
    thetas: &[f32],
    dtype: DType,
) -> Result<(Vec<u8>, Vec<u8>), TensorError> {
    let axes = sections.len();
    let well_formed = axes > 0
        && thetas.len() == axes
        && positions.len().is_multiple_of(axes)
        && sections.iter().all(|section| section % 2 == 0);
    if !well_formed {
        return Err(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "multi-axis rope needs even sections, a theta per axis and whole rows",
        });
    }
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for row in positions.chunks(axes) {
        for (axis, &section) in sections.iter().enumerate() {
            for pair in 0..section / 2 {
                let exponent = -((2 * pair) as f32) / section as f32;
                let angle = row[axis] * libm::powf(thetas[axis], exponent);
                cos.push(libm::cosf(angle));
                sin.push(libm::sinf(angle));
            }
        }
    }
    Ok((encode_table(&cos, dtype)?, encode_table(&sin, dtype)?))
}

/// Stacks the outputs of the chosen layers into one `[s, taps, d]` node:
/// `layer_outputs[taps[k]]` (each `[s, d]`) lands at index `k` of the middle
/// axis. A consumer that wants the concatenated feature axis reads it as
/// `[s, taps * d]` (row-major layout makes that a relabel, not a copy).
/// Each tap is routed by one `Select` on an `Iota == k` mask, so the stack is
/// plain `Elementwise` and gradients reach every tapped layer.
///
/// # Errors
///
/// [`TensorError::NotLowerable`] when `taps` is empty or names a layer past
/// `layer_outputs`.
pub fn tap_layers(
    program: &mut Vec<Op>,
    layer_outputs: &[NodeId],
    taps: &[u32],
) -> Result<NodeId, TensorError> {
    let chosen = taps
        .iter()
        .map(|&layer| layer_outputs.get(layer as usize).copied())
        .collect::<Option<Vec<NodeId>>>()
        .filter(|chosen| !chosen.is_empty())
        .ok_or(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "tap_layers needs at least one tap and every tap inside the layer list",
        })?;
    let slot_dtype = produced_dtype_or_binary32(program, chosen[0]);
    let slots = op::append(
        program,
        Op::Iota {
            dtype: slot_dtype,
            extent: Extent::Static(chosen.len() as u32),
        },
    );
    let mut stacked = scalar_constant_like(program, 0.0, chosen[0])?;
    for (slot, &layer) in chosen.iter().enumerate() {
        let index = scalar_constant_like(program, slot as f32, slots)?;
        let is_slot = elementwise_like(
            program,
            slots,
            ScalarOp::Equal,
            &[(slots, "k->k"), (index, "->k")],
        )?;
        let acc_map = if slot == 0 { "->skd" } else { "skd->skd" };
        stacked = elementwise_like(
            program,
            layer,
            ScalarOp::Select,
            &[(is_slot, "k->skd"), (layer, "sd->skd"), (stacked, acc_map)],
        )?;
    }
    Ok(stacked)
}

/// Window applied to each frame before the DFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum WindowKind {
    /// Hann, periodic form: `0.5 * (1 - cos(2*pi*i / n_fft))`.
    #[default]
    HannPeriodic,
    /// `0.5 * (1 - cos(2*pi*i / (n_fft - 1)))`.
    HannSymmetric,
    /// All ones: the bare DFT of each frame.
    Rectangular,
}

/// Every constant of [`stft_log_mel`]; the defaults are one family's reference values (sourced in
/// the spec's reference table, not here), and none is selected by a name.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct StftLogMelConfig {
    /// Window applied to each frame before the DFT.
    pub window: WindowKind,
    /// DFT length and frame length in samples, default 400.
    pub n_fft: u32,
    /// Stride between frames in samples, default 160.
    pub hop: u32,
    /// Lower bound on a mel energy before the log, default 1e-10.
    pub floor: f32,
    /// Decades below the whole-tensor peak at which log values are raised, default 8.
    pub dynamic_range: f32,
    /// Multiplier of the final affine rescale, default 0.25 (`(x + 4) / 4` with the offset).
    pub scale: f32,
    /// Addend of the final affine rescale, default 1.0.
    pub offset: f32,
}

impl Default for StftLogMelConfig {
    fn default() -> Self {
        Self {
            window: WindowKind::HannPeriodic,
            n_fft: 400,
            hop: 160,
            floor: 1e-10,
            dynamic_range: 8.0,
            scale: 0.25,
            offset: 1.0,
        }
    }
}

/// Name of the `[n_fft]` window leaf [`stft_log_mel`] declares.
pub const STFT_WINDOW_INPUT: &str = "stft_window";
/// Name of the `[n_fft, bins]` cosine-table leaf [`stft_log_mel`] declares.
pub const STFT_COS_INPUT: &str = "stft_cos";
/// Name of the `[n_fft, bins]` sine-table leaf [`stft_log_mel`] declares.
pub const STFT_SIN_INPUT: &str = "stft_sin";

/// Host-built leaf bytes for [`stft_log_mel`], each encoded as the leaf's dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StftTables {
    /// `[n_fft]`, bound to [`STFT_WINDOW_INPUT`].
    pub window: Vec<u8>,
    /// `[n_fft, bins]`, bound to [`STFT_COS_INPUT`].
    pub cos: Vec<u8>,
    /// `[n_fft, bins]`, bound to [`STFT_SIN_INPUT`].
    pub sin: Vec<u8>,
}

impl StftLogMelConfig {
    /// Frequency bins kept from the DFT: `n_fft / 2 + 1`.
    #[must_use]
    pub const fn bins(&self) -> u32 {
        self.n_fft / 2 + 1
    }

    fn frame_count(&self, samples: u32) -> Result<u32, TensorError> {
        if self.n_fft < 2 || self.hop == 0 || samples < self.n_fft {
            return Err(TensorError::NotLowerable {
                node: NodeId(0),
                reason: "stft needs n_fft >= 2, a nonzero hop and at least one whole frame",
            });
        }
        Ok((samples - self.n_fft) / self.hop + 1)
    }

    fn window_values(&self) -> Vec<f32> {
        let length = f64::from(self.n_fft);
        (0..self.n_fft)
            .map(|index| {
                let phase = f64::from(index) * core::f64::consts::TAU;
                match self.window {
                    WindowKind::HannPeriodic => 0.5 * (1.0 - libm::cos(phase / length)),
                    WindowKind::HannSymmetric => 0.5 * (1.0 - libm::cos(phase / (length - 1.0))),
                    WindowKind::Rectangular => 1.0,
                }
            })
            .map(|value| value as f32)
            .collect()
    }
}

/// Host tables for [`stft_log_mel`]: the window and the `[n_fft, bins]` cosine and sine matrices
/// (`cos(2*pi*t*k / n_fft)`, `sin(...)`), encoded as `dtype` through [`FormatCodec`] like
/// [`multi_axis_rope_tables`]. The sine table is positive; the power spectrum squares it, so the
/// sign convention of the DFT is immaterial.
///
/// # Errors
///
/// [`TensorError::NotLowerable`] when `n_fft < 2` or `dtype` cannot be encoded.
pub fn stft_tables(config: &StftLogMelConfig, dtype: DType) -> Result<StftTables, TensorError> {
    if config.n_fft < 2 {
        return Err(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "stft needs n_fft >= 2",
        });
    }
    let length = f64::from(config.n_fft);
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for tap in 0..config.n_fft {
        for bin in 0..config.bins() {
            let angle = core::f64::consts::TAU * f64::from(tap) * f64::from(bin) / length;
            cos.push(libm::cos(angle) as f32);
            sin.push(libm::sin(angle) as f32);
        }
    }
    Ok(StftTables {
        window: encode_table(&config.window_values(), dtype)?,
        cos: encode_table(&cos, dtype)?,
        sin: encode_table(&sin, dtype)?,
    })
}

/// Log-mel spectrogram of a waveform `signal` (`[samples]`) as `[mels, frames]`, with
/// `frames = (samples - n_fft) / hop + 1`: window each frame, DFT as a matmul against host
/// cos/sin tables, power, projection through `filterbank` (`[mels, bins]`, a bound operand),
/// `Maximum` floor, `log10`, a whole-tensor `Reduce Maximum` then `Maximum` for the dynamic-range
/// clamp, and the affine rescale. It declares the window, cosine and sine leaves
/// ([`STFT_WINDOW_INPUT`], [`STFT_COS_INPUT`], [`STFT_SIN_INPUT`]) in the signal's dtype; bind them
/// from [`stft_tables`]. Framing is an `Elementwise` read through the two-term window map
/// `hop*frame + tap`, which autograd differentiates by mask composition; every other step is
/// `Elementwise` or an Add/Maximum `Reduce`, so the whole front end is trainable to the waveform
/// and to the filterbank. Padding the signal is the caller's job. Compose [`layer_norm`] or a
/// convolution after it; [`non_causal_attention`] shows the same leaf-plus-table pattern.
///
/// # Errors
///
/// [`TensorError::NotLowerable`] when `n_fft < 2`, `hop == 0` or `samples < n_fft`.
pub fn stft_log_mel(
    program: &mut Vec<Op>,
    config: &StftLogMelConfig,
    signal: NodeId,
    samples: u32,
    filterbank: NodeId,
) -> Result<NodeId, TensorError> {
    let frames = config.frame_count(samples)?;
    let dtype = produced_dtype_or_binary32(program, signal);
    let table = |extents: &[u32]| {
        extents
            .iter()
            .map(|&e| Extent::Static(e))
            .collect::<Vec<_>>()
    };
    let window = input_leaf(program, dtype, table(&[config.n_fft]), STFT_WINDOW_INPUT);
    let shape = table(&[config.n_fft, config.bins()]);
    let cos = input_leaf(program, dtype, shape.clone(), STFT_COS_INPUT);
    let sin = input_leaf(program, dtype, shape, STFT_SIN_INPUT);
    let stamp = op::append(
        program,
        Op::Constant {
            dtype,
            shape: table(&[frames, config.n_fft]),
            value: 1.0,
        },
    );
    let window_map = alloc::format!("{}*f+t->ft", config.hop);
    let framed = elementwise_like(
        program,
        signal,
        ScalarOp::Multiply,
        &[(signal, window_map.as_str()), (stamp, "ft->ft")],
    )?;
    let windowed = elementwise_like(
        program,
        signal,
        ScalarOp::Multiply,
        &[(framed, "ft->ft"), (window, "t->ft")],
    )?;
    let real = dft_half(program, windowed, cos)?;
    let imaginary = dft_half(program, windowed, sin)?;
    let real_sq = elementwise_like(
        program,
        signal,
        ScalarOp::Multiply,
        &[(real, "fk->fk"), (real, "fk->fk")],
    )?;
    let imaginary_sq = elementwise_like(
        program,
        signal,
        ScalarOp::Multiply,
        &[(imaginary, "fk->fk"), (imaginary, "fk->fk")],
    )?;
    let power = elementwise_like(
        program,
        signal,
        ScalarOp::Add,
        &[(real_sq, "fk->fk"), (imaginary_sq, "fk->fk")],
    )?;
    let weighted = elementwise_like(
        program,
        signal,
        ScalarOp::Multiply,
        &[(power, "fk->mfk"), (filterbank, "mk->mfk")],
    )?;
    let mel = reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        weighted,
        "mfk->mfk",
        "mf->mfk",
    )?;
    clamp_and_rescale(program, config, mel)
}

fn dft_half(program: &mut Vec<Op>, windowed: NodeId, table: NodeId) -> Result<NodeId, TensorError> {
    let product = elementwise_like(
        program,
        windowed,
        ScalarOp::Multiply,
        &[(windowed, "ft->ftk"), (table, "tk->ftk")],
    )?;
    reduce_like(
        program,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "ftk->ftk",
        "fk->ftk",
    )
}

fn clamp_and_rescale(
    program: &mut Vec<Op>,
    config: &StftLogMelConfig,
    mel: NodeId,
) -> Result<NodeId, TensorError> {
    let floor = scalar_constant_like(program, config.floor, mel)?;
    let floored = elementwise_like(
        program,
        mel,
        ScalarOp::Maximum,
        &[(mel, "mf->mf"), (floor, "->mf")],
    )?;
    let natural = elementwise_like(program, mel, ScalarOp::Logarithm, &[(floored, "mf->mf")])?;
    let per_decade = scalar_constant_like(program, core::f32::consts::LOG10_E, mel)?;
    let decades = elementwise_like(
        program,
        mel,
        ScalarOp::Multiply,
        &[(natural, "mf->mf"), (per_decade, "->mf")],
    )?;
    let peak = reduce_like(
        program,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        decades,
        "mf->mf",
        "->mf",
    )?;
    let range = scalar_constant_like(program, config.dynamic_range, mel)?;
    let lowest = elementwise_like(
        program,
        mel,
        ScalarOp::Subtract,
        &[(peak, "->"), (range, "->")],
    )?;
    let clamped = elementwise_like(
        program,
        mel,
        ScalarOp::Maximum,
        &[(decades, "mf->mf"), (lowest, "->mf")],
    )?;
    let scale = scalar_constant_like(program, config.scale, mel)?;
    let scaled = elementwise_like(
        program,
        mel,
        ScalarOp::Multiply,
        &[(clamped, "mf->mf"), (scale, "->mf")],
    )?;
    let offset = scalar_constant_like(program, config.offset, mel)?;
    elementwise_like(
        program,
        mel,
        ScalarOp::Add,
        &[(scaled, "mf->mf"), (offset, "->mf")],
    )
}

#[cfg(test)]
mod layer_norm_tests {
    use super::*;
    use crate::spec::input_leaf;

    fn reference(x: &[f32], weight: &[f32], bias: &[f32], width: usize, eps: f32) -> Vec<f32> {
        let mut out = Vec::new();
        for row in x.chunks(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inverse = 1.0 / (variance + eps).sqrt();
            for (column, value) in row.iter().enumerate() {
                out.push((value - mean) * inverse * weight[column] + bias[column]);
            }
        }
        out
    }

    #[test]
    fn affine_layer_norm_matches_naive_reference_on_a_non_model_shape() {
        let (rows, width) = (3usize, 5usize);
        let x: Vec<f32> = (0..rows * width)
            .map(|i| ((i * 7 % 11) as f32) - 4.5)
            .collect();
        let weight = [0.5_f32, -1.0, 2.0, 1.5, 0.25];
        let bias = [0.1_f32, -0.2, 0.3, -0.4, 0.5];
        let mut program = Vec::new();
        let shape = |dims: &[u32]| dims.iter().map(|&d| Extent::Static(d)).collect::<Vec<_>>();
        let x_node = input_leaf(
            &mut program,
            DType::FLOAT32,
            shape(&[rows as u32, width as u32]),
            "x",
        );
        let w_node = input_leaf(
            &mut program,
            DType::FLOAT32,
            shape(&[width as u32]),
            "weight",
        );
        let b_node = input_leaf(&mut program, DType::FLOAT32, shape(&[width as u32]), "bias");
        let out = layer_norm(&mut program, x_node, w_node, b_node, width as u32, 1e-5)
            .expect("affine layer norm builds");
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[],
            &[("x", &x), ("weight", &weight), ("bias", &bias)],
            &[out],
        )
        .expect("affine layer norm evaluates");
        let got = evaluated.get(out).expect("output requested").0.to_vec();
        let want = reference(&x, &weight, &bias, width, 1e-5);
        assert_eq!(got.len(), rows * width);
        for (index, (actual, expected)) in got.iter().zip(&want).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "element {index}: {actual} vs {expected}"
            );
        }
    }
}

#[cfg(test)]
mod stft_log_mel_tests {
    use super::*;
    use crate::spec::input_leaf;

    const SAMPLES: usize = 64;
    const MELS: usize = 4;

    fn config() -> StftLogMelConfig {
        StftLogMelConfig {
            n_fft: 16,
            hop: 4,
            ..StftLogMelConfig::default()
        }
    }

    fn decode(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    }

    fn waveform() -> Vec<f32> {
        (0..SAMPLES)
            .map(|index| {
                let time = index as f32;
                3.0 * (0.9 * time).sin() + 1.5 * (2.1 * time).cos() + 0.1 * ((index * 5 % 7) as f32)
            })
            .collect()
    }

    fn filterbank(bins: usize, silent_row: Option<usize>) -> Vec<f32> {
        let mut rows = Vec::new();
        for mel in 0..MELS {
            for bin in 0..bins {
                let center = (mel as f32 + 0.5) * bins as f32 / MELS as f32;
                let weight = (1.0 - (bin as f32 - center).abs() / 3.0).max(0.0);
                rows.push(if silent_row == Some(mel) { 0.0 } else { weight });
            }
        }
        rows
    }

    fn reference(signal: &[f32], bank: &[f32], config: &StftLogMelConfig) -> Vec<f64> {
        let (n_fft, hop, bins) = (
            config.n_fft as usize,
            config.hop as usize,
            config.bins() as usize,
        );
        let window = config.window_values();
        let frames = (signal.len() - n_fft) / hop + 1;
        let mut decades = vec![0f64; MELS * frames];
        for frame in 0..frames {
            let power: Vec<f64> = (0..bins)
                .map(|bin| {
                    let (mut real, mut imaginary) = (0f64, 0f64);
                    for tap in 0..n_fft {
                        let sample = f64::from(signal[frame * hop + tap]) * f64::from(window[tap]);
                        let angle = core::f64::consts::TAU * (tap * bin) as f64 / n_fft as f64;
                        real += sample * angle.cos();
                        imaginary -= sample * angle.sin();
                    }
                    real * real + imaginary * imaginary
                })
                .collect();
            for mel in 0..MELS {
                let energy: f64 = (0..bins)
                    .map(|bin| power[bin] * f64::from(bank[mel * bins + bin]))
                    .sum();
                decades[mel * frames + frame] = energy.max(f64::from(config.floor)).log10();
            }
        }
        let peak = decades.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        decades
            .iter()
            .map(|value| {
                value.max(peak - f64::from(config.dynamic_range)) * f64::from(config.scale)
                    + f64::from(config.offset)
            })
            .collect()
    }

    fn run(config: &StftLogMelConfig, bank: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let bins = config.bins();
        let mut program = Vec::new();
        let signal = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(SAMPLES as u32)],
            "signal",
        );
        let fb = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(MELS as u32), Extent::Static(bins)],
            "filterbank",
        );
        let out = stft_log_mel(&mut program, config, signal, SAMPLES as u32, fb)
            .expect("stft log-mel builds");
        let tables = stft_tables(config, DType::FLOAT32).expect("tables encode");
        let (window, cos, sin) = (
            decode(&tables.window),
            decode(&tables.cos),
            decode(&tables.sin),
        );
        let wave = waveform();
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[],
            &[
                ("signal", &wave),
                ("filterbank", bank),
                (STFT_WINDOW_INPUT, &window),
                (STFT_COS_INPUT, &cos),
                (STFT_SIN_INPUT, &sin),
            ],
            &[out],
        )
        .expect("stft log-mel evaluates");
        (
            evaluated.get(out).expect("output requested").0.to_vec(),
            wave,
        )
    }

    #[test]
    fn stft_log_mel_matches_a_direct_dft_reference_on_a_64_sample_signal() {
        let config = config();
        let bank = filterbank(config.bins() as usize, None);
        let (got, wave) = run(&config, &bank);
        let want = reference(&wave, &bank, &config);
        assert_eq!(got.len(), MELS * 13);
        for (index, (actual, expected)) in got.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(*actual) - expected).abs() < 1e-4,
                "element {index}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn stft_log_mel_raises_a_bin_far_below_the_peak_to_max_minus_range() {
        let config = config();
        let bank = filterbank(config.bins() as usize, Some(3));
        let (got, wave) = run(&config, &bank);
        let frames = 13;
        let peak_decades = (0..MELS - 1)
            .flat_map(|mel| (0..frames).map(move |frame| (mel, frame)))
            .map(|(mel, frame)| (f64::from(got[mel * frames + frame]) - 1.0) / 0.25)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            peak_decades - 8.0 > -10.0,
            "silent bin must sit below the clamp"
        );
        let want = (peak_decades - 8.0) * 0.25 + 1.0;
        let silent = &got[3 * frames..4 * frames];
        assert!(
            silent
                .iter()
                .all(|value| (f64::from(*value) - want).abs() < 1e-4)
        );
        let reference = reference(&wave, &bank, &config);
        assert!((f64::from(silent[0]) - reference[3 * frames]).abs() < 1e-4);
    }

    #[test]
    fn stft_log_mel_rejects_a_signal_shorter_than_one_frame() {
        let mut program = Vec::new();
        let signal = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(8)],
            "signal",
        );
        let fb = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(1), Extent::Static(9)],
            "fb",
        );
        assert!(stft_log_mel(&mut program, &config(), signal, 8, fb).is_err());
    }
}
