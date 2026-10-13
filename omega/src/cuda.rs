//! CUDA C kernel emission — the third ISA target over the same [`BoundOp`]
//! descriptor [`crate::msl::emit`] (Metal) and [`crate::wgsl::emit_wgsl`]
//! (WGSL) already emit from. Same "runtime uniforms, not baked constants"
//! stance, same per-[`Keep`] execution model, same structural cacheability
//! property (`same_structure_different_extents_yield_identical_source`
//! below) — this module differs only in target ISA (`extern "C" __global__`
//! CUDA C instead of MSL/WGSL text) and, for now, in which of
//! [`crate::msl`]'s optimized fast paths it ports: this crate's row-blocked
//! and tiled-`simdgroup_matrix` GEMM paths are Metal-`simdgroup`-specific and
//! have no port here yet, so every op renders through the generic per-thread
//! (elementwise/scan) or per-output-element (reduce) path.
//!
//! # Emitter and driver boundary
//!
//! This module is pure string generation over an already-bound [`BoundOp`] —
//! it never links `libcuda`, `nvrtc`, or any NVIDIA driver, and has zero
//! dependencies (`cuda = []` in `Cargo.toml`). That is what lets it build and
//! its tests run on a machine with no NVIDIA GPU, including the macOS CI host
//! that gates this crate. The emitted CUDA C source is never compiled here
//! (`nvcc`/`nvrtc` are not invoked by this module), so this module's tests
//! prove emission STRUCTURE only — that the right buffers, unpack calls, and
//! control flow appear in the text — not numeric execution parity. The
//! separately gated `cuda-driver` module uses cudarc and NVRTC to compile and
//! launch the supported emitted graph subset. This module remains the source
//! EMITTER; `crate::backend::GpuDriver::Cuda` exists only when that driver
//! feature is enabled (see `crate::backend`'s own doc, §B.4 of the
//! 2026-09-04 design).
//!
//! # v1 scope
//!
//! - **Elementwise**, **`Keep::Reduce`** (serial per-output-element fold, plus
//!   a warp-shuffle cooperative fold for the associative/commutative
//!   `ScalarOp`s — see `reduce_is_cooperative`), and **`Keep::Scan`**
//!   (one thread, serial over every outer line — matching
//!   `proxima_tensor::cpu::run_scan`'s persistent accumulator, the same shape
//!   `crate::wgsl::render_scan` takes).
//! - **Gather.** A fault buffer records an out-of-range fetched index (the
//!   `atomicOr`-based flag `push_gather_fetch` emits) the same way
//!   `crate::msl`'s own gather-fetch renderer's `atomic_fetch_max` does, so
//!   `omega::execute`'s driver can turn a nonzero slot into the same
//!   `TensorError::GatherIndexOutOfRange` the CPU oracle reports.
//! - **Packed operands**: `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`/`Q4_0` plus
//!   `Float16`/`BFloat16`, ported from [`crate::msl`]'s own unpack constants
//!   — see [`Q4K_UNPACK_CUDA`] and its siblings.
//! - **`f32`/`f16`** (`type_token`) — `f16` needs `<cuda_fp16.h>`'s
//!   `__half`, this module's one CUDA-specific preamble line MSL's `half`
//!   needed no analogue for.
//!
//! # Not in v1
//!
//! - **`Iota`/`Constant`** — [`EmitError::CudaUnsupportedOpKind`], the same
//!   v1 boundary [`crate::wgsl`] draws.
//! - **The row-blocked packed-matmul and tiled `simdgroup_matrix` GEMM fast
//!   paths** `crate::msl` carries — Metal-`simdgroup`-specific amortizations
//!   with no CUDA `wmma`/tensor-core port here yet. Every packed op renders
//!   through the generic per-element unpack accessor instead.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_tensor::{
    BoundOp, BoundOpKind, ComposedBody, DType, Keep, Layout, Lookup, NodeId, NumericPolicy,
    ReduceInit, ScalarOp, StepArg,
};

use crate::error::EmitError;
use crate::grid::checked_product;
use crate::identity::{op_token, operand_codecs, reduce_epilogue_is_identity};
use crate::msl::{Binding, Codec, PackedOperands};

/// Appended to a Q4_0 matvec's CUDA entry name only when
/// `NumericPolicy::activation_quantization` is granted and the dp4a
/// recogniser matches -- the int8-activation body is a different kernel from
/// the f32 body for the same op, so the two must never share a compiled
/// module.
const ACTIVATION_QUANTIZED_MATVEC_TOKEN: &str = "_aq8dp4";

/// Every lane of one NVIDIA warp — fixed at 32 on every CUDA-capable GPU
/// generation to date, the same "hardware-family fact, never a policy knob"
/// class `crate::sized::SIMD_WIDTH` is in for Apple's SIMD-group (restated
/// rather than shared: the two are numerically equal but are DIFFERENT
/// hardware facts about different vendors' GPUs, and conflating them would
/// make a change to one silently affect the other). Not read from a device
/// at emit time — emission has no device handle, only the [`BoundOp`]'s
/// structure — so this has to be the compile-time constant a driver's
/// dispatch is built to honor unconditionally.
pub const WARP_SIZE: u64 = 32;

/// One compiled CUDA kernel: source text, its `extern "C"` entry point, the
/// buffer-index -> data mapping a driver needs to bind before launch (the
/// same [`Binding`] list [`crate::msl::emit`]/[`crate::wgsl::emit_wgsl`]
/// return), and the thread count this dispatch needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaKernel {
    pub source: String,
    pub entry: String,
    pub bindings: Vec<Binding>,
    pub grid: CudaGridSpec,
    /// Arena buffers a driver binds after the [`Binding::Uniforms`] slot, in
    /// kernel-argument order, each with its element count. Non-empty only for
    /// a kind that writes outputs beyond [`Binding::Output`] (`MoeTopK`); the
    /// list is `BoundOpKind::moe_topk_extra_outputs`, the one layout the CPU
    /// interpreter's extra sink and the Metal backend follow too.
    pub extra_outputs: Vec<(NodeId, usize)>,
    /// Kernels the driver launches, in order, on the same stream before this one.
    /// Empty for every kernel `emit_cuda_with_policy` produces today.
    pub prologue: Vec<CudaKernel>,
}

/// How many threads a driver must launch, and — for a cooperative reduce —
/// the block width the launch is pinned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaGridSpec {
    pub threads: u64,
    /// `Some(WARP_SIZE)` for a warp-shuffle cooperative reduce (see
    /// `reduce_is_cooperative`): the driver must launch blocks exactly
    /// this wide so every warp boundary lands on an output-element
    /// boundary. `None` for every other kernel.
    pub block_width: Option<u64>,
}

/// Threads per block for every kernel whose [`CudaGridSpec::block_width`] is
/// `None`: the driver's choice, named here so the emit-time grid check and the
/// launch agree on how many blocks a grid needs.
pub const DEFAULT_BLOCK_WIDTH: u64 = 256;

/// Most blocks a launch's x dimension holds (`gridDim.x <= 2^31 - 1`). The
/// kernel's own `gid` is 64-bit, so this, not a 32-bit thread index, is the
/// widest grid `emit_cuda` covers.
const MAX_GRID_BLOCKS: u64 = i32::MAX as u64;

impl CudaGridSpec {
    /// The block width this launch runs at: the pinned cooperative width, else
    /// [`DEFAULT_BLOCK_WIDTH`].
    #[must_use]
    pub fn block_width_or_default(&self) -> u64 {
        self.block_width.unwrap_or(DEFAULT_BLOCK_WIDTH).max(1)
    }

    /// Blocks needed to cover [`Self::threads`] at [`Self::block_width_or_default`].
    #[must_use]
    pub fn blocks(&self) -> u64 {
        self.threads.div_ceil(self.block_width_or_default())
    }
}

/// Emits a CUDA C kernel from a bound [`BoundOp`] — the CUDA counterpart of
/// [`crate::msl::emit`]. See the module doc for exactly which op shapes v1
/// covers.
///
/// # Errors
/// [`EmitError::UnsupportedDType`] for anything but `Float32`/`Float16`,
/// [`EmitError::CudaUnsupportedOpKind`] for `CachedAttention`,
/// [`EmitError::ArityMismatch`]/[`EmitError::ReductionBodyIsSelect`]/
/// [`EmitError::EmptyScan`] for the same structural failures
/// [`crate::msl::emit`] rejects.
///
/// # Examples
///
/// ```
/// use proxima_tensor::{DType, Extent, IndexMap, Op, ScalarOp, append, map};
///
/// let mut program = Vec::new();
/// let source = append(
///     &mut program,
///     Op::Input {
///         dtype: DType::Float32,
///         shape: vec![Extent::Static(4)],
///         name: None,
///     },
/// );
/// let activated = append(
///     &mut program,
///     Op::Elementwise {
///         dtype: DType::Float32,
///         body: ScalarOp::Tanh,
///         operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
///         name: None,
///     },
/// );
///
/// let shapes = proxima_tensor::infer(&program, &[])?;
/// // `bind` binds only what `outputs` reaches (`bind_plain`'s reachability
/// // pass, ROW 541, `proxima-tensor/docs/discipline.md`) -- an empty
/// // outputs list binds nothing.
/// let bound_ops = proxima_tensor::bind(
///     &program,
///     &shapes,
///     &[activated],
///     proxima_tensor::NumericPolicy::default(),
/// )?;
/// let packed_operands = omega::PackedOperands::new();
/// let kernel = omega::emit_cuda(&bound_ops[0], &packed_operands)?;
/// assert!(kernel.source.contains("extern \"C\" __global__"));
/// assert!(kernel.source.contains("tanhf("));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn emit_cuda(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
) -> Result<CudaKernel, EmitError> {
    emit_cuda_with_policy(resolved, packed_operands, NumericPolicy::default())
}

pub fn emit_cuda_with_policy(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
) -> Result<CudaKernel, EmitError> {
    validate(resolved)?;
    let quantized = operand_codecs(resolved, packed_operands);
    let entry = entry_name(resolved, packed_operands, &quantized, numeric_policy);
    let source = match &resolved.kind {
        BoundOpKind::Elementwise { .. } => render_elementwise(resolved, &entry, &quantized)?,
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => render_reduce(resolved, &entry, &quantized, numeric_policy)?,
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => render_scan(resolved, &entry, &quantized)?,
        BoundOpKind::Iota => render_leaf(resolved, &entry, None)?,
        BoundOpKind::Constant { value } => render_leaf(resolved, &entry, Some(*value))?,
        BoundOpKind::CachedAttention { .. } => {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: "cached_attention",
            });
        }
        BoundOpKind::GatedDeltaNet { .. } => {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: "gated_delta_net",
            });
        }
        BoundOpKind::MoeTopK { .. } => render_moe_topk(resolved, &entry)?,
        BoundOpKind::TopFractionSelect { .. } => {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: "top_fraction_select",
            });
        }
        BoundOpKind::RoundBatchedReduce { .. } => {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: "round_batched_reduce",
            });
        }
        BoundOpKind::CachedSoftmaxWeights { .. } => {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: "cached_softmax_weights",
            });
        }
        BoundOpKind::ElementwiseTwin { .. } => {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: "elementwise_twin",
            });
        }
    };
    let cooperative = reduce_is_cooperative(resolved, &quantized, numeric_policy);
    let grid = CudaGridSpec {
        threads: grid_threads(resolved, cooperative)?,
        block_width: match &resolved.kind {
            BoundOpKind::MoeTopK { expert_count, .. } => Some(moe_block_width(*expert_count)),
            _ => cooperative.then_some(WARP_SIZE),
        },
    };
    if grid.blocks() > MAX_GRID_BLOCKS {
        return Err(EmitError::GridExceedsThreadIndex {
            node: resolved.node,
            threads: grid.threads,
            limit: MAX_GRID_BLOCKS * grid.block_width_or_default(),
        });
    }
    Ok(CudaKernel {
        source,
        entry,
        bindings: bindings(resolved),
        grid,
        extra_outputs: resolved
            .kind
            .moe_topk_extra_outputs(resolved.extents.iter().product::<u64>() as usize),
        prologue: Vec::new(),
    })
}

/// one warp lane per element, so a block is exactly one warp wide.
const Q8_1_BLOCK_ELEMENTS: u64 = 32;

const Q8_1_QUANTIZE_ENTRY: &str = "omega_cuda_q8_1_quantize";

const Q8_1_QUANTIZE_SOURCE: &str = r#"#include <cuda_fp16.h>
extern "C" __global__ void omega_cuda_q8_1_quantize(const float *in, float *out, const int *u) {
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long block = gid / 32;
    unsigned int lane = (unsigned int)(gid & 31);
    if (block >= u[0]) return;
    float x = in[block * 32 + lane];
    float amax = fabsf(x);
    for (int offset = 16; offset > 0; offset >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset)); }
    float d = amax / 127.0f;
    int q = amax == 0.0f ? 0 : (int)roundf(x / d);
    int qsum = q;
    for (int offset = 16; offset > 0; offset >>= 1) { qsum += __shfl_xor_sync(0xffffffffu, qsum, offset); }
    unsigned char *bytes = (unsigned char *)out + block * 36;
    if (lane == 0) {
        unsigned short d_bits = __half_as_ushort(__float2half(d));
        unsigned short s_bits = __half_as_ushort(__float2half(d * (float)qsum));
        bytes[0] = (unsigned char)(d_bits & 0xffu);
        bytes[1] = (unsigned char)(d_bits >> 8);
        bytes[2] = (unsigned char)(s_bits & 0xffu);
        bytes[3] = (unsigned char)(s_bits >> 8);
    }
    bytes[4 + lane] = (unsigned char)(signed char)q;
}
"#;

/// The activation quantize that feeds a dp4a matvec body (S3): one warp per
/// 32-element block, written as `block_q8_1` bytes into `scratch`. Emitted but
/// not yet called from `emit_cuda_with_policy`; the driver launches it through
/// [`CudaKernel::prologue`].
pub fn emit_q8_1_quantize(
    activation: NodeId,
    scratch: NodeId,
    element_count: u64,
) -> Result<CudaKernel, EmitError> {
    if element_count == 0 || !element_count.is_multiple_of(Q8_1_BLOCK_ELEMENTS) {
        return Err(EmitError::CudaUnsupportedOpKind {
            node: activation,
            kind: "q8_1_quantize_length_not_block_multiple",
        });
    }
    let block_count = element_count / Q8_1_BLOCK_ELEMENTS;
    let grid = CudaGridSpec {
        threads: checked_product(activation, [block_count, WARP_SIZE])?,
        block_width: Some(WARP_SIZE),
    };
    if grid.blocks() > MAX_GRID_BLOCKS {
        return Err(EmitError::GridExceedsThreadIndex {
            node: activation,
            threads: grid.threads,
            limit: MAX_GRID_BLOCKS * grid.block_width_or_default(),
        });
    }
    Ok(CudaKernel {
        source: String::from(Q8_1_QUANTIZE_SOURCE),
        entry: String::from(Q8_1_QUANTIZE_ENTRY),
        bindings: vec![
            Binding::Input(activation),
            Binding::Output(scratch),
            Binding::Uniforms,
        ],
        grid,
        extra_outputs: Vec::new(),
        prologue: Vec::new(),
    })
}

/// Emits a whole bound program, moving a previous kernel across unchanged when
/// the bound op for the same node is equal to the one it was emitted from.
/// Returns the kernels in `resolved` order and how many were reused. On a
/// failed emit the error carries the failing op's node, as the cold path does.
///
/// Teaching pointer: [`emit_cuda_with_policy`] is a pure function of
/// `(resolved, packed_operands, numeric_policy)`, so an equal `BoundOp` under
/// the same packed operands and policy emits identical text. Equality is taken
/// over the whole op, not over a guessed subset of text-reaching fields, because
/// the entry name omits packed extents and cannot stand in for the source.
/// The caller owns the "same packed operands and policy" half of that premise.
pub fn emit_cuda_reusing(
    previous_bound: &[BoundOp],
    previous_kernels: Vec<CudaKernel>,
    resolved: &[BoundOp],
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
) -> Result<(Vec<CudaKernel>, usize), (NodeId, EmitError)> {
    let mut previous: BTreeMap<NodeId, (&BoundOp, Option<CudaKernel>)> = previous_bound
        .iter()
        .zip(previous_kernels)
        .map(|(bound, kernel)| (bound.node, (bound, Some(kernel))))
        .collect();
    let mut reused = 0;
    let mut kernels = Vec::with_capacity(resolved.len());
    for bound in resolved {
        let unchanged = previous
            .get_mut(&bound.node)
            .filter(|(old, _)| *old == bound)
            .and_then(|(_, kernel)| kernel.take());
        kernels.push(match unchanged {
            Some(kernel) => {
                reused += 1;
                kernel
            }
            None => emit_cuda_with_policy(bound, packed_operands, numeric_policy)
                .map_err(|error| (bound.node, error))?,
        });
    }
    Ok((kernels, reused))
}
/// Packs the runtime ABI for the f32 elementwise CUDA kernels.
///
/// CUDA's emitted `Uniforms` uses native-width `long` fields, so this is
/// deliberately separate from the wgpu i32 packer. Keeping the packer beside
/// the emitter makes the source layout and the host layout one contract.
/// `cuda_driver::launch` (this crate's own `cuda-driver` feature) is the one
/// non-test caller — the `cuda` feature alone is the dependency-free emitter
/// this module's own doc describes, with no binding/argument packer wired to
/// it yet, so this stays test-only in that build rather than dead code the
/// lint has to be silenced around.
#[cfg(any(test, feature = "cuda-driver"))]
pub(crate) fn pack_elementwise_uniforms(resolved: &BoundOp) -> Result<Vec<u8>, EmitError> {
    if !matches!(resolved.kind, BoundOpKind::Elementwise { .. }) {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "elementwise",
            found: resolved.kind.name(),
        });
    }
    let rank_len = resolved.extents.len().max(1);
    let mut bytes = Vec::new();
    let push_i64 = |bytes: &mut Vec<u8>, value: i64| bytes.extend_from_slice(&value.to_ne_bytes());
    push_i64(&mut bytes, resolved.extents.iter().product::<u64>() as i64);
    for slot in 0..rank_len {
        push_i64(
            &mut bytes,
            resolved.extents.get(slot).copied().unwrap_or(0) as i64,
        );
    }
    for (_, layout, _) in resolved.operands() {
        push_i64(&mut bytes, layout.base);
    }
    for (_, layout, _) in resolved.operands() {
        for slot in 0..rank_len {
            push_i64(&mut bytes, layout.strides.get(slot).copied().unwrap_or(0));
        }
    }
    push_cuda_gather_uniforms(&mut bytes, resolved, rank_len);
    Ok(bytes)
}

#[cfg(any(test, feature = "cuda-driver"))]
fn push_cuda_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_ne_bytes());
}

#[cfg(any(test, feature = "cuda-driver"))]
fn push_cuda_i64_row(bytes: &mut Vec<u8>, values: &[i64], width: usize) {
    for slot in 0..width {
        push_cuda_i64(bytes, values.get(slot).copied().unwrap_or(0));
    }
}

#[cfg(any(test, feature = "cuda-driver"))]
fn push_cuda_extents(bytes: &mut Vec<u8>, extents: &[u64], width: usize) {
    for slot in 0..width {
        push_cuda_i64(bytes, extents.get(slot).copied().unwrap_or(0) as i64);
    }
}

#[cfg(any(test, feature = "cuda-driver"))]
fn contiguous_strides(extents: &[u64]) -> Vec<i64> {
    let mut strides = vec![0_i64; extents.len()];
    let mut stride = 1_i64;
    for (index, extent) in extents.iter().enumerate().rev() {
        strides[index] = stride;
        stride = stride.saturating_mul(*extent as i64);
    }
    strides
}

#[cfg(any(test, feature = "cuda-driver"))]
fn push_cuda_gather_uniforms(bytes: &mut Vec<u8>, bound: &BoundOp, rank_len: usize) {
    let gathers: Vec<&Lookup> = bound
        .operands()
        .iter()
        .filter_map(|(_, _, lookup)| lookup.as_ref())
        .collect();
    for lookup in &gathers {
        push_cuda_i64(bytes, lookup.index_layout.base);
    }
    for lookup in &gathers {
        push_cuda_i64_row(bytes, &lookup.index_layout.strides, rank_len);
    }
    for lookup in &gathers {
        push_cuda_i64(bytes, lookup.element_stride);
    }
    for lookup in &gathers {
        push_cuda_i64(bytes, lookup.extent as i64);
    }
}

/// Packs every non-attention CUDA graph ABI currently emitted by this module.
/// The field order mirrors the `Uniforms` declarations in the three renderers;
/// all fields are `long`, so the host representation is fixed at i64 words.
/// See [`pack_elementwise_uniforms`]'s own doc for why this is test-only
/// without the `cuda-driver` feature.
#[cfg(any(test, feature = "cuda-driver"))]
pub(crate) fn pack_cuda_uniforms(resolved: &BoundOp) -> Result<Vec<u8>, EmitError> {
    let rank_len = resolved.extents.len().max(1);
    let mut bytes = Vec::new();
    match &resolved.kind {
        BoundOpKind::Elementwise { .. } => return pack_elementwise_uniforms(resolved),
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            out_layout,
            epilogue_operands,
            epilogue_broadcast_axes,
            ..
        } => {
            let reduce_axes = reduction_dims(resolved, output_axes);
            let is_broadcast_epilogue = !epilogue_broadcast_axes.is_empty();
            let output_rank_len = if is_broadcast_epilogue {
                rank_len
            } else {
                output_axes.len().max(1)
            };
            let reduce_rank_len = reduce_axes.len().max(1);
            let output_total = if is_broadcast_epilogue {
                resolved.extents.iter().product::<u64>()
            } else {
                output_axes
                    .iter()
                    .map(|axis| resolved.extents[*axis as usize])
                    .product::<u64>()
            };
            push_cuda_i64(&mut bytes, output_total as i64);
            push_cuda_i64(
                &mut bytes,
                reduce_axes
                    .iter()
                    .map(|axis| resolved.extents[*axis as usize])
                    .product::<u64>() as i64,
            );
            let output_extents: Vec<i64> = if is_broadcast_epilogue {
                resolved
                    .extents
                    .iter()
                    .map(|extent| *extent as i64)
                    .collect()
            } else {
                output_axes
                    .iter()
                    .map(|axis| resolved.extents[*axis as usize] as i64)
                    .collect()
            };
            let reduction_extents: Vec<i64> = reduce_axes
                .iter()
                .map(|axis| resolved.extents[*axis as usize] as i64)
                .collect();
            push_cuda_i64_row(&mut bytes, &output_extents, output_rank_len);
            push_cuda_i64_row(&mut bytes, &reduction_extents, reduce_rank_len);
            for (_, layout, _) in resolved.operands() {
                push_cuda_i64(&mut bytes, layout.base);
            }
            for (_, layout, _) in resolved.operands() {
                push_cuda_i64_row(&mut bytes, &layout.strides, rank_len);
            }
            push_cuda_i64(&mut bytes, out_layout.base);
            push_cuda_i64_row(&mut bytes, &out_layout.strides, rank_len);
            if !epilogue_operands.is_empty() {
                for (_, layout, _) in epilogue_operands {
                    push_cuda_i64(&mut bytes, layout.base);
                }
                for (_, layout, _) in epilogue_operands {
                    push_cuda_i64_row(&mut bytes, &layout.strides, output_rank_len);
                }
            }
            if is_broadcast_epilogue {
                push_cuda_i64_row(&mut bytes, &contiguous_strides(&resolved.extents), rank_len);
            }
            push_cuda_gather_uniforms(&mut bytes, resolved, rank_len);
        }
        BoundOpKind::Reduce {
            keep: Keep::Scan,
            out_layout,
            ..
        } => {
            let outer_rank = resolved.extents.len().saturating_sub(1);
            let outer_rank_len = outer_rank.max(1);
            push_cuda_i64(
                &mut bytes,
                resolved.extents[..outer_rank].iter().product::<u64>() as i64,
            );
            push_cuda_i64(
                &mut bytes,
                resolved.extents.last().copied().unwrap_or(1) as i64,
            );
            push_cuda_extents(&mut bytes, &resolved.extents[..outer_rank], outer_rank_len);
            for (_, layout, _) in resolved.operands() {
                push_cuda_i64(&mut bytes, layout.base);
            }
            for (_, layout, _) in resolved.operands() {
                push_cuda_i64_row(&mut bytes, &layout.strides, rank_len);
            }
            push_cuda_i64(&mut bytes, out_layout.base);
            push_cuda_i64_row(&mut bytes, &out_layout.strides, rank_len);
        }
        BoundOpKind::Iota | BoundOpKind::Constant { .. } | BoundOpKind::MoeTopK { .. } => {
            push_cuda_i64(&mut bytes, resolved.extents.iter().product::<u64>() as i64);
        }
        BoundOpKind::CachedAttention { .. }
        | BoundOpKind::GatedDeltaNet { .. }
        | BoundOpKind::TopFractionSelect { .. }
        | BoundOpKind::RoundBatchedReduce { .. }
        | BoundOpKind::CachedSoftmaxWeights { .. }
        | BoundOpKind::ElementwiseTwin { .. } => {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: resolved.kind.name(),
            });
        }
    }
    Ok(bytes)
}

fn render_leaf(
    resolved: &BoundOp,
    entry: &str,
    constant: Option<f32>,
) -> Result<String, EmitError> {
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let mut source = String::new();
    preamble(&mut source, element_type);
    source.push_str("struct Uniforms { long total_elements; };\n\n");
    kernel_signature(&mut source, &[], 0, 0, entry, element_type);
    source.push_str("    if (gid >= u.total_elements) { return; }\n");
    match constant {
        Some(value) => source.push_str(&format!(
            "    out[gid] = ({element_type}){};\n",
            cuda_float_literal(value)
        )),
        None => source.push_str(&format!("    out[gid] = ({element_type})gid;\n")),
    }
    source.push_str("}\n");
    Ok(source)
}

/// Most threads one CUDA block holds (`maxThreadsPerBlock`, fixed at 1024 on
/// every generation `emit_cuda` targets).
const MAX_BLOCK_WIDTH: u64 = 1024;

/// Block width of the `MoeTopK` kernel: `expert_count` rounded up to whole
/// warps, so every `__shfl_xor_sync` reduces a full warp and the padding
/// lanes past `expert_count` are masked rather than divergent.
fn moe_block_width(expert_count: u64) -> u64 {
    expert_count.div_ceil(WARP_SIZE).max(1) * WARP_SIZE
}

/// [`BoundOpKind::MoeTopK`]'s fused kernel, the CUDA port of
/// [`crate::msl`]'s `render_moe_topk`: one block per token, `top_k` rounds of
/// take-the-maximum-with-exclusion. A warp reduces with `__shfl_xor_sync`, one
/// value per warp crosses through shared memory, and two `__syncthreads` per
/// round publish the warp maxima and the warp winners.
///
/// Ties resolve to the HIGHER index (`max` over `masked == max_value ? tid : -1`),
/// and exclusion removes EVERY expert tied with the round's maximum -- both
/// exactly as the Metal arm and `proxima_tensor::cpu::run_moe_topk` do. The
/// candidate guard `tid < expert_count` keeps a padding lane (held at `-inf`)
/// from winning a row whose real scores are all `-inf`.
///
/// Arguments after the `Uniforms` slot are the `2 * top_k` extra outputs, then
/// the stacked pair when present, in `moe_topk_extra_outputs` order.
fn render_moe_topk(resolved: &BoundOp, entry: &str) -> Result<String, EmitError> {
    let BoundOpKind::MoeTopK {
        expert_count,
        top_k,
        stacked,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "moe_topk",
            found: resolved.kind.name(),
        });
    };
    let block_width = moe_block_width(*expert_count);
    if block_width > MAX_BLOCK_WIDTH {
        return Err(EmitError::WideGridUnsupported {
            node: resolved.node,
            reason: "its reduction is coherent only inside the one block it runs as, and expert_count exceeds one block",
        });
    }
    let expert_count = *expert_count;
    let top_k = *top_k;
    let extra_count = 2 * top_k;
    let num_warps = block_width / WARP_SIZE;
    let mut source = String::new();
    preamble(&mut source, "float");
    source.push_str("struct Uniforms { long unused; };\n\n");
    source.push_str(&format!(
        "extern \"C\" __global__ void {entry}(\n    const float* __restrict__ scores,\n    float* __restrict__ route0,\n    const Uniforms* __restrict__ u_ptr"
    ));
    for index in 0..extra_count {
        source.push_str(&format!(",\n    float* __restrict__ extra{index}"));
    }
    if stacked.is_some() {
        source.push_str(
            ",\n    float* __restrict__ stacked_routes,\n    float* __restrict__ stacked_weights",
        );
    }
    source.push_str(") {\n");
    let extra_params: Vec<String> = (0..extra_count)
        .map(|index| format!("extra{index}"))
        .collect();
    let stacked_writes = if stacked.is_some() {
        "            stacked_routes[(long)token * top_k + round] = (float)winner_index;\n\
         \x20           stacked_weights[(long)token * top_k + round] = weight;\n"
    } else {
        ""
    };
    source.push_str(&format!(
        "    constexpr unsigned int expert_count = {expert_count}u;\n\
         \x20   constexpr unsigned int top_k = {top_k}u;\n\
         \x20   constexpr unsigned int num_warps = {num_warps}u;\n\
         \x20   float* extras[{extra_count}] = {{ {extras} }};\n\
         \x20   const float negative_infinity = -__int_as_float(0x7f800000);\n\
         \x20   __shared__ float warp_max[num_warps];\n\
         \x20   __shared__ int warp_index[num_warps];\n\
         \x20   const unsigned int token = blockIdx.x;\n\
         \x20   const unsigned int tid = threadIdx.x;\n\
         \x20   const unsigned int warp_id = tid >> 5;\n\
         \x20   const unsigned int lane = tid & 31u;\n\
         \x20   float masked = (tid < expert_count) ? scores[(long)token * expert_count + tid] : negative_infinity;\n\
         \x20   float max0 = 0.0f;\n\
         \x20   float weight_total = 0.0f;\n\
         \x20   for (unsigned int round = 0; round < top_k; round++) {{\n\
         \x20       float lane_max = masked;\n\
         \x20       for (int offset = 16; offset > 0; offset >>= 1) {{ lane_max = fmaxf(lane_max, __shfl_xor_sync(0xffffffffu, lane_max, offset)); }}\n\
         \x20       if (lane == 0) {{ warp_max[warp_id] = lane_max; }}\n\
         \x20       __syncthreads();\n\
         \x20       float max_value = warp_max[0];\n\
         \x20       for (unsigned int warp = 1; warp < num_warps; warp++) {{ max_value = fmaxf(max_value, warp_max[warp]); }}\n\
         \x20       int candidate = (masked == max_value && tid < expert_count) ? (int)tid : -1;\n\
         \x20       for (int offset = 16; offset > 0; offset >>= 1) {{ candidate = max(candidate, __shfl_xor_sync(0xffffffffu, candidate, offset)); }}\n\
         \x20       if (lane == 0) {{ warp_index[warp_id] = candidate; }}\n\
         \x20       __syncthreads();\n\
         \x20       int winner_index = warp_index[0];\n\
         \x20       for (unsigned int warp = 1; warp < num_warps; warp++) {{ winner_index = max(winner_index, warp_index[warp]); }}\n\
         \x20       if (round == 0) {{ max0 = max_value; }}\n\
         \x20       if (tid == 0) {{\n\
         \x20           const float weight = expf(max_value - max0);\n\
         \x20           weight_total += weight;\n\
         \x20           if (round == 0) {{ route0[token] = (float)winner_index; }} else {{ extras[round - 1][token] = (float)winner_index; }}\n\
         \x20           extras[(top_k - 1) + round][token] = weight;\n\
         {stacked_writes}\
         \x20           if (round == top_k - 1) {{ extras[{extra_count} - 1][token] = weight_total; }}\n\
         \x20       }}\n\
         \x20       if (masked == max_value) {{ masked = negative_infinity; }}\n\
         \x20   }}\n\
         }}\n",
        extras = extra_params.join(", ")
    ));
    Ok(source)
}

fn cuda_float_literal(value: f32) -> String {
    if value.is_nan() {
        "NAN".to_string()
    } else if value == f32::INFINITY {
        "__int_as_float(0x7f800000)".to_string()
    } else if value == f32::NEG_INFINITY {
        "-__int_as_float(0x7f800000)".to_string()
    } else {
        format!("{value:?}f")
    }
}

#[cfg(not(feature = "float-layout-dtype"))]
fn type_token(node: NodeId, dtype: DType) -> Result<&'static str, EmitError> {
    match dtype {
        DType::Float16 => Ok("__half"),
        DType::Float32
        | DType::BFloat16
        | DType::Bool
        | DType::Int8
        | DType::UInt8
        | DType::Int32
        | DType::UInt32 => Ok("float"),
        DType::Int16
        | DType::UInt16
        | DType::Int64
        | DType::UInt64
        | DType::Int128
        | DType::UInt128
        | DType::Float64
        | DType::BFloat8
        | DType::Fp6E3M2 => Err(EmitError::UnsupportedDType { node, dtype }),
    }
}

/// `__half` and `__nv_bfloat16` where the layout's fields equal binary16's and bfloat16's, `float`
/// for every other layout (the kernel computes in binary32 and the host narrows through the
/// layout codec at the buffer boundary, as on Metal).
#[cfg(feature = "float-layout-dtype")]
fn type_token(node: NodeId, dtype: DType) -> Result<&'static str, EmitError> {
    match dtype {
        DType::Float(layout) => Ok(match crate::msl::layout_render::hardware_type(&layout) {
            Some("half") => "__half",
            Some("bfloat") => "__nv_bfloat16",
            _ => "float",
        }),
        DType::Bool | DType::Int8 | DType::UInt8 | DType::Int32 | DType::UInt32 => Ok("float"),
        DType::Int16
        | DType::UInt16
        | DType::Int64
        | DType::UInt64
        | DType::Int128
        | DType::UInt128
        | DType::Float64 => Err(EmitError::UnsupportedDType { node, dtype }),
    }
}

fn validate_body(node: NodeId, body: &ComposedBody) -> Result<(), EmitError> {
    for step in &body.steps {
        let expected = step.op.arity();
        let found = step.args.len();
        if expected != found {
            return Err(EmitError::ArityMismatch {
                node,
                expected,
                found,
            });
        }
    }
    Ok(())
}

fn validate(resolved: &BoundOp) -> Result<(), EmitError> {
    validate_body(resolved.node, resolved.element_body())?;
    if let BoundOpKind::Reduce {
        reduce_op,
        keep,
        out_scatter,
        epilogue_body,
        epilogue_operands,
        ..
    } = &resolved.kind
    {
        if out_scatter.is_some() {
            return Err(EmitError::ScatterNotSupported {
                node: resolved.node,
            });
        }
        if matches!(reduce_op, ScalarOp::Select) {
            return Err(EmitError::ReductionBodyIsSelect {
                node: resolved.node,
            });
        }
        // `render_reduce` (this module's `Keep::Reduce` renderer) now folds
        // `BoundOpKind::Reduce::epilogue_body` into its output write via
        // `push_reduce_epilogue_write`, mirroring `crate::msl`/`crate::wgsl`.
        // Scan's renderer (`render_scan`) has no such write hook yet -- same
        // "no renderer, reject" contract, scoped to the one shape still
        // missing it.
        if *keep == Keep::Scan && !reduce_epilogue_is_identity(epilogue_body, epilogue_operands) {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: "scan with a fused epilogue",
            });
        }
        if *keep == Keep::Scan && resolved.extents.is_empty() {
            return Err(EmitError::EmptyScan {
                node: resolved.node,
            });
        }
    }
    Ok(())
}

fn reduction_dims(resolved: &BoundOp, output_axes: &[u16]) -> Vec<u16> {
    (0..resolved.extents.len() as u16)
        .filter(|dim| !output_axes.contains(dim))
        .collect()
}

fn gather_count(resolved: &BoundOp) -> usize {
    resolved
        .operands()
        .iter()
        .filter(|(_, _, gather)| gather.is_some())
        .count()
}

/// For each operand, `Some(slot)` if it gathers — `slot` is its position
/// among only the gathered operands, matching the order [`bindings`] appends
/// `Indices` buffers in. Mirrors `crate::msl::gather_slots` exactly.
fn gather_slots(resolved: &BoundOp) -> Vec<Option<usize>> {
    let mut next = 0usize;
    resolved
        .operands()
        .iter()
        .map(|(_, _, gather)| {
            gather.as_ref().map(|_| {
                let slot = next;
                next += 1;
                slot
            })
        })
        .collect()
}

fn bindings(resolved: &BoundOp) -> Vec<Binding> {
    // `all_read_sources`, not `operands` -- a `BoundOpKind::Reduce` with a
    // fused epilogue reads its `epilogue_operands` too, and those need a
    // buffer bound at the exact `epi{index}` parameter position
    // `kernel_signature` declares (see that function's own doc).
    let mut bindings: Vec<Binding> = resolved
        .all_read_sources()
        .map(|(node, _, _)| Binding::Input(*node))
        .collect();
    for (_, _, gather) in resolved.operands() {
        if let Some(gather_access) = gather {
            bindings.push(Binding::Indices(gather_access.indices));
        }
    }
    bindings.push(Binding::Output(resolved.node));
    bindings.push(Binding::Uniforms);
    if gather_count(resolved) > 0 {
        bindings.push(Binding::Fault);
    }
    bindings
}

fn grid_threads(resolved: &BoundOp, cooperative: bool) -> Result<u64, EmitError> {
    let extents_product = || checked_product(resolved.node, resolved.extents.iter().copied());
    match &resolved.kind {
        BoundOpKind::Elementwise { .. } => extents_product(),
        BoundOpKind::Reduce {
            output_axes,
            epilogue_broadcast_axes,
            keep: Keep::Reduce,
            ..
        } => {
            let output_total: u64 = if epilogue_broadcast_axes.is_empty() {
                checked_product(
                    resolved.node,
                    output_axes
                        .iter()
                        .map(|dim| resolved.extents[*dim as usize]),
                )?
            } else {
                extents_product()?
            };
            if cooperative {
                checked_product(resolved.node, [output_total, WARP_SIZE])
            } else {
                Ok(output_total)
            }
        }
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => {
            let rank = resolved.extents.len();
            checked_product(
                resolved.node,
                resolved.extents[..rank.saturating_sub(1)].iter().copied(),
            )
        }
        // `CachedAttention`/`GatedDeltaNet`/`RoundBatchedReduce`/
        // `CachedSoftmaxWeights` never reach this function in practice --
        // `emit_cuda`'s own match on `resolved.kind` returns
        // `EmitError::CudaUnsupportedOpKind` for each before `grid_threads`
        // is ever called. Grouped with `Iota`/`Constant` only to satisfy
        // exhaustiveness with a harmless value, never a real dispatch shape.
        BoundOpKind::MoeTopK { expert_count, .. } => checked_product(
            resolved.node,
            [
                resolved.extents.iter().product::<u64>(),
                moe_block_width(*expert_count),
            ],
        ),
        BoundOpKind::Iota
        | BoundOpKind::Constant { .. }
        | BoundOpKind::GatedDeltaNet { .. }
        | BoundOpKind::TopFractionSelect { .. }
        | BoundOpKind::RoundBatchedReduce { .. }
        | BoundOpKind::CachedSoftmaxWeights { .. }
        | BoundOpKind::CachedAttention { .. }
        | BoundOpKind::ElementwiseTwin { .. } => extents_product(),
    }
}

/// Whether every gathered operand selects the same index across the
/// reduction. A zero index-layout stride on each reduction dim means one
/// fetched index serves all lanes; any other stride selects a different
/// row per reduction element and must stay serial. Mirrors
/// `crate::msl::gather_is_reduction_invariant`.
fn gather_is_reduction_invariant(resolved: &BoundOp, output_axes: &[u16]) -> bool {
    let reduce_dims = reduction_dims(resolved, output_axes);
    resolved.operands().iter().all(|(_, _, lookup)| {
        lookup.as_ref().is_none_or(|lookup| {
            reduce_dims
                .iter()
                .all(|&dim| lookup.index_layout.stride(dim) == 0)
        })
    })
}

/// Whether `resolved` is a `Keep::Reduce` fold whose `reduce_op` is
/// associative and commutative and whose gathered operands (if any) are
/// invariant over the reduction — the same set `crate::msl::reduce_is_cooperative`
/// picks for a SIMD-group cooperative fold: `Subtract`/`Divide` are not
/// associative, so reordering their combination across lanes is wrong, not
/// merely imprecise.
fn reduce_is_cooperative(
    resolved: &BoundOp,
    _quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
) -> bool {
    match &resolved.kind {
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            reduce_op,
            output_axes,
            ..
        } => {
            // Bit-exact mode retains the reference accumulation order for
            // packed dots. Relaxed mode explicitly permits the warp fold;
            // the packed decode is still deterministic, but its reordered
            // f32 accumulation is a numeric-policy decision rather than an
            // unconditional backend rewrite.
            numeric_policy.reassociation
                && gather_is_reduction_invariant(resolved, output_axes)
                && is_cooperative_reduce_op(*reduce_op)
        }
        _ => false,
    }
}

fn is_cooperative_reduce_op(op: ScalarOp) -> bool {
    matches!(
        op,
        ScalarOp::Add | ScalarOp::Multiply | ScalarOp::Maximum | ScalarOp::Minimum
    )
}

/// The `__shfl_down_sync` tree-reduction step for one cooperative
/// `reduce_op` — CUDA has no single builtin analogous to MSL's `simd_sum`/
/// `simd_max`, so the combine is a five-step butterfly reduction over
/// `__shfl_down_sync` instead of one call; see [`push_cooperative_reduce_body`]
/// for where this is spliced.
fn shuffle_combine_expr(
    node: NodeId,
    op: ScalarOp,
    accumulator: &str,
    shuffled: &str,
) -> Result<String, EmitError> {
    match op {
        ScalarOp::Add => Ok(format!("{accumulator} + {shuffled}")),
        ScalarOp::Multiply => Ok(format!("{accumulator} * {shuffled}")),
        ScalarOp::Maximum => Ok(format!("fmaxf({accumulator}, {shuffled})")),
        ScalarOp::Minimum => Ok(format!("fminf({accumulator}, {shuffled})")),
        ScalarOp::Identity
        | ScalarOp::Subtract
        | ScalarOp::Divide
        | ScalarOp::Negate
        | ScalarOp::Reciprocal
        | ScalarOp::Exponential
        | ScalarOp::Logarithm
        | ScalarOp::SquareRoot
        | ScalarOp::Tanh
        | ScalarOp::Erf
        | ScalarOp::Greater
        | ScalarOp::Equal
        | ScalarOp::Select => Err(EmitError::NonCooperativeReduceOp {
            node,
            op: op_token(op),
        }),
    }
}

/// This language's own [`crate::identity::kernel_identity`] — see that
/// module's doc for the full union of axes folded in and why. No CUDA
/// driver exists yet (this crate emits source only), so nothing currently
/// caches by this string, but it is still the name embedded literally in
/// emitted CUDA source, and gains the same fix `crate::wgsl::entry_name`
/// does: per-operand packed codec, which `crate::cuda`'s own `packed_
/// element_expr` already renders a distinct body for, and dtype width
/// class — this census's own finding, neither was folded in before.
///
/// The `_gid64` suffix names the index form the emitted text carries (`long gid`
/// built from `blockIdx.x * blockDim.x` in 64 bits). The entry name doubles as
/// the cache identity a driver keys compiled modules on, and this text changed
/// for every kernel at every size when the 32-bit multiply was widened, so a
/// module compiled from the earlier text can never be served for this one.
/// `_gid64c8` marks the revision whose cooperative reduce hoists Q6_K chunks.
fn entry_name(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
) -> String {
    let identity = crate::identity::kernel_identity(
        crate::identity::KernelLanguage::Cuda,
        resolved,
        packed_operands,
        crate::identity::MetalOnlyExtras::default(),
        // Cuda has no `BoundOpKind::CachedAttention` renderer
        // (`proxima_tensor::bind::bind_with_fusion`'s own doc: this driver
        // calls it with `fuse_cached_attention: false`), so a policy value
        // never actually varies this identity string -- the default is
        // correct, not a placeholder.
        proxima_tensor::NumericPolicy::default(),
    );
    let activation_token = if numeric_policy.activation_quantization
        && q4_0_matvec(resolved, quantized).is_some()
    {
        ACTIVATION_QUANTIZED_MATVEC_TOKEN
    } else {
        ""
    };
    format!("{identity}{activation_token}_gid64c8")
}

/// The `(weight, activation)` operand slots of a Q4_0-weight matvec that the
/// int8-activation `dp4a` body can serve: one `Multiply` over exactly two
/// operands, add-reduced from `Zero`, with no gather and no fused epilogue,
/// the weight packed as Q4_0 and the activation unpacked and constant across
/// every output row. Either operand order matches.
fn q4_0_matvec(resolved: &BoundOp, quantized: &[Option<Codec>]) -> Option<(usize, usize)> {
    let BoundOpKind::Reduce {
        reduce_op: ScalarOp::Add,
        init: ReduceInit::Zero,
        keep: Keep::Reduce,
        output_axes,
        epilogue_body,
        epilogue_operands,
        operands,
        ..
    } = &resolved.kind
    else {
        return None;
    };
    let [only_step] = resolved.element_body().steps.as_slice() else {
        return None;
    };
    let is_two_operand_multiply = only_step.op == ScalarOp::Multiply
        && only_step.args == [StepArg::Operand(0), StepArg::Operand(1)];
    let (weight_slot, activation_slot) = match quantized {
        [Some(Codec::Q4_0), None] => (0, 1),
        [None, Some(Codec::Q4_0)] => (1, 0),
        _ => return None,
    };
    let (_, activation_layout, _) = operands.get(activation_slot)?;
    let activation_is_broadcast = output_axes
        .iter()
        .all(|axis| activation_layout.stride(*axis) == 0);
    (is_two_operand_multiply
        && gather_count(resolved) == 0
        && reduce_epilogue_is_identity(epilogue_body, epilogue_operands)
        && activation_is_broadcast)
        .then_some((weight_slot, activation_slot))
}

fn scalar_op_expr(op: ScalarOp, args: &[&str]) -> String {
    match op {
        ScalarOp::Identity => (*args.first().unwrap_or(&"0.0f")).into(),
        ScalarOp::Add => format!("({} + {})", args[0], args[1]),
        ScalarOp::Subtract => format!("({} - {})", args[0], args[1]),
        ScalarOp::Multiply => format!("({} * {})", args[0], args[1]),
        ScalarOp::Divide => format!("({} / {})", args[0], args[1]),
        ScalarOp::Maximum => format!("fmaxf({}, {})", args[0], args[1]),
        ScalarOp::Minimum => format!("fminf({}, {})", args[0], args[1]),
        ScalarOp::Negate => format!("(-{})", args[0]),
        ScalarOp::Reciprocal => format!("(1.0f / {})", args[0]),
        ScalarOp::Exponential => format!("expf({})", args[0]),
        ScalarOp::Logarithm => format!("logf({})", args[0]),
        ScalarOp::SquareRoot => format!("sqrtf({})", args[0]),
        ScalarOp::Tanh => format!("tanhf({})", args[0]),
        ScalarOp::Erf => format!("proxima_erf({})", args[0]),
        ScalarOp::Greater => format!("(({} > {}) ? 1.0f : 0.0f)", args[0], args[1]),
        ScalarOp::Equal => format!("((fabsf({} - {}) == 0.0f) ? 1.0f : 0.0f)", args[0], args[1]),
        ScalarOp::Select => format!("(({} != 0.0f) ? {} : {})", args[0], args[1], args[2]),
    }
}

/// `(init expression, seeded-from-the-start)` — ports
/// `crate::msl::fold_init_tokens`. The infinity values use CUDA's bit-cast
/// builtin rather than relying on an `INFINITY` macro in NVRTC headers.
fn fold_init_tokens(init: ReduceInit) -> (&'static str, &'static str) {
    match init {
        ReduceInit::Zero => ("0.0f", "true"),
        ReduceInit::One => ("1.0f", "true"),
        ReduceInit::NegativeInfinity => ("-__int_as_float(0x7f800000)", "true"),
        ReduceInit::PositiveInfinity => ("__int_as_float(0x7f800000)", "true"),
        ReduceInit::FirstElement => ("0.0f", "false"),
    }
}

/// The algebraic identity `op` folds against without changing a value —
/// ports `crate::msl::cooperative_identity_token`. Every warp lane but lane
/// 0 seeds its private accumulator with this rather than the `BoundOp`'s own
/// `ReduceInit`, so folding it into the final `__shfl_down_sync` combine can
/// never perturb the result.
fn cooperative_identity_token(node: NodeId, op: ScalarOp) -> Result<&'static str, EmitError> {
    match op {
        ScalarOp::Add => Ok("0.0f"),
        ScalarOp::Multiply => Ok("1.0f"),
        ScalarOp::Maximum => Ok("-__int_as_float(0x7f800000)"),
        ScalarOp::Minimum => Ok("__int_as_float(0x7f800000)"),
        ScalarOp::Identity
        | ScalarOp::Subtract
        | ScalarOp::Divide
        | ScalarOp::Negate
        | ScalarOp::Reciprocal
        | ScalarOp::Exponential
        | ScalarOp::Logarithm
        | ScalarOp::SquareRoot
        | ScalarOp::Tanh
        | ScalarOp::Erf
        | ScalarOp::Greater
        | ScalarOp::Equal
        | ScalarOp::Select => Err(EmitError::NonCooperativeReduceOp {
            node,
            op: op_token(op),
        }),
    }
}

fn push_body_steps(
    source: &mut String,
    body: &ComposedBody,
    indent: &str,
    element_type: &str,
) -> String {
    crate::epilogue::declare_steps(
        source,
        body,
        "scratch",
        "step",
        scalar_op_expr,
        |source, index, expr| {
            source.push_str(&format!("{indent}{element_type} step{index} = {expr};\n"));
        },
    )
}

/// [`push_body_steps`]'s counterpart for [`BoundOpKind::Reduce::
/// epilogue_body`] — the CUDA sibling of `crate::msl::
/// push_epilogue_body_steps`/`crate::wgsl::push_epilogue_body_steps`, reading
/// `epi_scratch[i]`/`epi_step{k}` instead of `push_body_steps`'s
/// `scratch[i]`/`step{k}` so the epilogue's own operand table never collides
/// with the fold's.
fn push_epilogue_body_steps(
    source: &mut String,
    body: &ComposedBody,
    indent: &str,
    element_type: &str,
) -> String {
    crate::epilogue::declare_steps(
        source,
        body,
        "epi_scratch",
        "epi_step",
        scalar_op_expr,
        |source, index, expr| {
            source.push_str(&format!(
                "{indent}{element_type} epi_step{index} = {expr};\n"
            ));
        },
    )
}

/// Shared write tail for every CUDA reduce renderer that folds an output
/// element ([`push_serial_reduce_body`], [`push_cooperative_reduce_body`]) —
/// the CUDA counterpart of `crate::msl::push_reduce_epilogue_write`/
/// `crate::wgsl::push_reduce_epilogue_write`. `coord` gives the OUTPUT-
/// axis-order coordinate expression for axis `dim`. When the epilogue is the
/// untouched identity default ([`reduce_epilogue_is_identity`]), this emits
/// exactly the one-line `out[...] = accumulator;` write every caller emitted
/// before epilogue fusion existed.
#[allow(clippy::too_many_arguments)]
fn push_reduce_epilogue_write(
    source: &mut String,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    output_rank: usize,
    element_type: &str,
    indent: &str,
    coord: impl Fn(usize) -> String,
    accumulator_expr: &str,
    out_offset_expr: &str,
) {
    if reduce_epilogue_is_identity(epilogue_body, epilogue_operands) {
        source.push_str(&format!(
            "{indent}out[{out_offset_expr}] = {accumulator_expr};\n"
        ));
        return;
    }
    let epilogue_operand_count = epilogue_operands.len();
    source.push_str(&format!(
        "{indent}{element_type} epi_scratch[{}];\n",
        epilogue_operand_count + 1
    ));
    for index in 0..epilogue_operand_count {
        source.push_str(&format!(
            "{indent}long epi_off{index} = u.epilogue_operand_base[{index}];\n"
        ));
        for dim in 0..output_rank {
            source.push_str(&format!(
                "{indent}epi_off{index} += {} * u.epilogue_operand_strides[{index}][{dim}];\n",
                coord(dim)
            ));
        }
        source.push_str(&format!(
            "{indent}epi_scratch[{index}] = epi{index}[epi_off{index}];\n"
        ));
    }
    source.push_str(&format!(
        "{indent}epi_scratch[{epilogue_operand_count}] = {accumulator_expr};\n"
    ));
    let epi_value = push_epilogue_body_steps(source, epilogue_body, indent, element_type);
    source.push_str(&format!("{indent}out[{out_offset_expr}] = {epi_value};\n"));
}

/// NVRTC builtin math has no `erf` variant that matches the
/// CPU oracle's bit pattern exactly — the same Abramowitz & Stegun 7.1.26
/// polynomial `crate::msl::PROXIMA_ERF_FN`/`crate::wgsl`'s
/// `PROXIMA_ERF_FN_WGSL` port, restated in CUDA C so all three backends and
/// the CPU interpreter run the identical formula rather than three
/// "close enough" approximations.
const PROXIMA_ERF_FN_CUDA: &str = "\
__device__ __forceinline__ float proxima_erf(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float magnitude = fabsf(x);
    float t = 1.0f / fmaf(0.3275911f, magnitude, 1.0f);
    float poly = t * fmaf(fmaf(fmaf(fmaf(1.061405429f, t, -1.453152027f), t, 1.421413741f), t, -0.284496736f), t, 0.254829592f);
    return sign * fmaf(poly, -expf(-magnitude * magnitude), 1.0f);
}
";

fn preamble(source: &mut String, element_type: &str) {
    // The packed-codec helpers use CUDA's half bit-conversion intrinsics even
    // when the graph's materialized element type is f32.  Keeping this header
    // unconditional makes the emitted translation unit self-contained for
    // NVRTC; only the bfloat header follows the graph's element type.
    source.push_str("#include <cuda_fp16.h>\n");
    if element_type == "__nv_bfloat16" {
        source.push_str("#include <cuda_bf16.h>\n");
    }
    source.push('\n');
    source.push_str(PROXIMA_ERF_FN_CUDA);
    source.push('\n');
    source.push_str(Q2K_UNPACK_CUDA);
    source.push('\n');
    source.push_str(Q3K_UNPACK_CUDA);
    source.push('\n');
    source.push_str(Q4K_UNPACK_CUDA);
    source.push('\n');
    source.push_str(Q5K_UNPACK_CUDA);
    source.push('\n');
    source.push_str(Q6K_UNPACK_CUDA);
    source.push('\n');
    source.push_str(Q8_0_UNPACK_CUDA);
    source.push('\n');
    source.push_str(Q4_0_UNPACK_CUDA);
    source.push('\n');
    source.push_str(BF16_UNPACK_CUDA);
    source.push('\n');
}

const Q2K_UNPACK_CUDA: &str = r#"
__device__ __forceinline__ float q2k_element(const unsigned char *block, unsigned int index) {
    const unsigned char *scales = block;
    const unsigned char *qs = block + 16;
    unsigned short d_bits = (unsigned short)((unsigned int)block[80] | ((unsigned int)block[81] << 8));
    unsigned short dmin_bits = (unsigned short)((unsigned int)block[82] | ((unsigned int)block[83] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    float dmin = __half2float(__ushort_as_half(dmin_bits));
    unsigned int chunk = index / 128u;
    unsigned int within = index % 128u;
    unsigned int group = within / 32u;
    unsigned int local = within % 32u;
    unsigned int sub_block = chunk * 8u + group * 2u + (local >= 16u ? 1u : 0u);
    unsigned char scale_min = scales[sub_block];
    float scale = d * (float)(scale_min & 0x0Fu);
    float minimum = dmin * (float)(scale_min >> 4u);
    unsigned char level = (qs[chunk * 32u + local] >> (2u * group)) & 0x03u;
    return scale * (float)level - minimum;
}
"#;

const Q3K_UNPACK_CUDA: &str = r#"
__device__ __forceinline__ int q3k_unpack_scale(const unsigned char *scales, unsigned int sub_block) {
    unsigned char low = (sub_block < 8u) ? (scales[sub_block] & 0x0Fu) : (scales[sub_block - 8u] >> 4u);
    unsigned char high = (scales[8u + sub_block % 4u] >> (2u * (sub_block / 4u))) & 0x03u;
    return (int)(low | (high << 4u)) - 32;
}
struct q3k_header_cuda { float scale; unsigned char mask; };
__device__ __forceinline__ q3k_header_cuda q3k_header_for(const unsigned char *block, unsigned int index) {
    const unsigned char *scales = block + 96;
    unsigned short d_bits = (unsigned short)((unsigned int)block[108] | ((unsigned int)block[109] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int chunk = index / 128u;
    unsigned int rem = index % 128u;
    unsigned int j = rem / 32u;
    unsigned int local32 = rem % 32u;
    q3k_header_cuda header;
    header.scale = d * (float)q3k_unpack_scale(scales, 8u * chunk + 2u * j + (local32 < 16u ? 0u : 1u));
    header.mask = (unsigned char)(1u << (4u * chunk + j));
    return header;
}
__device__ __forceinline__ float q3k_element(const unsigned char *block, unsigned int index) {
    q3k_header_cuda header = q3k_header_for(block, index);
    const unsigned char *qs = block + 32;
    unsigned int chunk = index / 128u;
    unsigned int rem = index % 128u;
    unsigned int j = rem / 32u;
    unsigned int local32 = rem % 32u;
    unsigned char level = (qs[chunk * 32u + local32] >> (2u * j)) & 0x03u;
    float correction = (block[local32] & header.mask) != 0u ? 0.0f : 4.0f;
    return header.scale * ((float)level - correction);
}
"#;

const Q2K_BLOCK_BYTES: usize = proxima_gguf::quant::q2_k::BLOCK_BYTES;
const Q2K_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q2_k::QK_K;
const Q3K_BLOCK_BYTES: usize = proxima_gguf::quant::q3_k::BLOCK_BYTES;
const Q3K_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q3_k::QK_K;

fn kernel_signature(
    source: &mut String,
    quantized: &[Option<Codec>],
    epilogue_operand_count: usize,
    gather_count: usize,
    entry: &str,
    element_type: &str,
) {
    source.push_str(&format!("extern \"C\" __global__ void {entry}(\n"));
    for (index, &codec) in quantized.iter().enumerate() {
        let binding_type = match codec {
            None => element_type,
            Some(Codec::Float16) => "__half",
            Some(_) => "unsigned char",
        };
        source.push_str(&format!(
            "    const {binding_type}* __restrict__ in{index},\n"
        ));
    }
    // A [`proxima_tensor::BoundOpKind::Reduce::epilogue_operands`] entry is
    // always a plain, un-gathered, un-packed `element_type` buffer -- the
    // same restriction `crate::msl::kernel_signature`'s own `epi{index}`
    // params carry -- so each gets one flat `__restrict__` pointer,
    // positioned right after the fold's own operands and before anything
    // gather adds.
    for index in 0..epilogue_operand_count {
        source.push_str(&format!(
            "    const {element_type}* __restrict__ epi{index},\n"
        ));
    }
    for slot in 0..gather_count {
        source.push_str(&format!(
            "    const float* __restrict__ gather_idx{slot},\n"
        ));
    }
    source.push_str(&format!("    {element_type}* __restrict__ out,\n"));
    source.push_str("    const Uniforms* __restrict__ u_ptr");
    if gather_count > 0 {
        source.push_str(",\n    unsigned int* __restrict__ fault\n");
    } else {
        source.push('\n');
    }
    source.push_str(") {\n");
    source.push_str("    const Uniforms u = *u_ptr;\n");
    source.push_str("    long gid = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;\n");
}

fn push_gather_uniform_fields(source: &mut String, gather_count: usize, rank_len: usize) {
    if gather_count == 0 {
        return;
    }
    source.push_str(&format!("    long gather_index_base[{gather_count}];\n"));
    source.push_str(&format!(
        "    long gather_index_strides[{gather_count}][{rank_len}];\n"
    ));
    source.push_str(&format!(
        "    long gather_element_stride[{gather_count}];\n"
    ));
    source.push_str(&format!("    long gather_extent[{gather_count}];\n"));
}

/// Emits the out-of-range check for one just-fetched, not-yet-clamped
/// `fetched{operand_index}`: when it falls outside
/// `[0, u.gather_extent[gather_slot])`, sets that gathered operand's `fault`
/// flag via `atomicExch`. The nonzero payload records the offending signed
/// index (the driver decodes the bits after synchronization), which is
/// required to distinguish malformed index data from a uniform-layout bug.
/// Multiple concurrent faults are diagnostic-only: any one recorded index is
/// sufficient to prove the bounds violation, while the zero sentinel means
/// no lane reported a fault.
fn push_gather_fault_check(
    source: &mut String,
    operand_index: usize,
    gather_slot: usize,
    indent: &str,
) {
    source.push_str(&format!(
        "{indent}if (fetched{operand_index} < 0 || fetched{operand_index} >= u.gather_extent[{gather_slot}]) {{\n"
    ));
    source.push_str(&format!(
        "{indent}    atomicExch(&fault[{gather_slot}], (unsigned int)fetched{operand_index});\n"
    ));
    source.push_str(&format!("{indent}}}\n"));
}

fn push_gather_fetch(
    source: &mut String,
    operand_index: usize,
    gather_slot: usize,
    rank: usize,
    coord_var: &str,
    offset_var: &str,
) {
    source.push_str(&format!(
        "    long gather_off{operand_index} = u.gather_index_base[{gather_slot}];\n"
    ));
    for dim in 0..rank {
        source.push_str(&format!(
            "    gather_off{operand_index} += {coord_var}[{dim}] * u.gather_index_strides[{gather_slot}][{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "    long fetched{operand_index} = (long)gather_idx{gather_slot}[gather_off{operand_index}];\n"
    ));
    push_gather_fault_check(source, operand_index, gather_slot, "    ");
    source.push_str(&format!(
        "    fetched{operand_index} = max((long)0, min(fetched{operand_index}, u.gather_extent[{gather_slot}] - 1));\n"
    ));
    source.push_str(&format!(
        "    {offset_var} += fetched{operand_index} * u.gather_element_stride[{gather_slot}];\n"
    ));
}

/// Once-per-output gather fetch for a reduction-invariant gathered operand:
/// every lane loads the same index (one broadcast transaction), lane 0 alone
/// records a fault, and the element offset is kept as `gather_base{index}`
/// for the lane loop to add. Mirrors `push_cooperative_gather_fetch` in MSL.
fn push_hoisted_gather_fetch(
    source: &mut String,
    operand_index: usize,
    gather_slot: usize,
    rank: usize,
) {
    source.push_str(&format!(
        "    long gather_off{operand_index} = u.gather_index_base[{gather_slot}];\n"
    ));
    for dim in 0..rank {
        source.push_str(&format!(
            "    gather_off{operand_index} += full_coord[{dim}] * u.gather_index_strides[{gather_slot}][{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "    long fetched{operand_index} = (long)gather_idx{gather_slot}[gather_off{operand_index}];\n"
    ));
    source.push_str("    if (lane == 0u && u.reduction_total > 0) {\n");
    push_gather_fault_check(source, operand_index, gather_slot, "        ");
    source.push_str("    }\n");
    source.push_str(&format!(
        "    fetched{operand_index} = max((long)0, min(fetched{operand_index}, u.gather_extent[{gather_slot}] - 1));\n"
    ));
    source.push_str(&format!(
        "    long gather_base{operand_index} = fetched{operand_index} * u.gather_element_stride[{gather_slot}];\n"
    ));
}

/// Adds the hoisted gather base to a gathered operand's element offset, so
/// the lane loop reads the selected row. Ungathered operands emit nothing.
fn push_gather_base_add(source: &mut String, resolved: &BoundOp, index: usize, indent: &str) {
    if resolved.operands()[index].2.is_some() {
        source.push_str(&format!("{indent}off{index} += gather_base{index};\n"));
    }
}

/// How operand `index` is READ, given the element-offset expression the
/// caller already computed — the CUDA counterpart of
/// `crate::msl::operand_read`, same per-element-only scope (no row-blocked
/// header amortization; see the module doc).
fn operand_read(
    node: NodeId,
    index: usize,
    offset: &str,
    codec: Option<Codec>,
) -> Result<String, EmitError> {
    match codec {
        None => Ok(format!("in{index}[{offset}]")),
        Some(Codec::Q2K) => Ok(format!(
            "q2k_element(in{index} + ({offset} / {Q2K_BLOCK_ELEMENTS}) * {Q2K_BLOCK_BYTES}, (unsigned int)({offset} % {Q2K_BLOCK_ELEMENTS}))"
        )),
        Some(Codec::Q3K) => Ok(format!(
            "q3k_element(in{index} + ({offset} / {Q3K_BLOCK_ELEMENTS}) * {Q3K_BLOCK_BYTES}, (unsigned int)({offset} % {Q3K_BLOCK_ELEMENTS}))"
        )),
        Some(Codec::Q4K) => Ok(format!(
            "q4k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q4K_BLOCK_BYTES}, (unsigned int)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        )),
        Some(Codec::Q5K) => Ok(format!(
            "q5k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q5K_BLOCK_BYTES}, (unsigned int)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        )),
        Some(Codec::Q6K) => Ok(format!(
            "q6k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q6K_BLOCK_BYTES}, (unsigned int)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        )),
        Some(Codec::Q8_0) => Ok(format!(
            "q8_0_element(in{index} + ({offset} / {Q8_0_BLOCK_ELEMENTS}) * {Q8_0_BLOCK_BYTES}, (unsigned int)({offset} % {Q8_0_BLOCK_ELEMENTS}))"
        )),
        Some(Codec::Q4_0) => Ok(format!(
            "q4_0_element(in{index} + ({offset} / {Q4_0_BLOCK_ELEMENTS}) * {Q4_0_BLOCK_BYTES}, (unsigned int)({offset} % {Q4_0_BLOCK_ELEMENTS}))"
        )),
        // `__half` converts implicitly to `float` in CUDA C++, the same
        // "already a valid narrow-float buffer" shape MSL's `half` takes.
        Some(Codec::Float16) => Ok(format!("in{index}[{offset}]")),
        Some(Codec::BFloat16) => Ok(format!(
            "bf16_element(in{index} + ({offset} / {BFLOAT16_BLOCK_ELEMENTS}) * {BFLOAT16_BLOCK_BYTES}, (unsigned int)({offset} % {BFLOAT16_BLOCK_ELEMENTS}))"
        )),
        // a future codec must choose here; no silent raw read of packed bytes
        Some(
            Codec::Q5_1
            | Codec::Q5_0
            | Codec::Q4_1
            | Codec::Q8_1
            | Codec::Q8K
            | Codec::Iq1S
            | Codec::Iq1M
            | Codec::Iq2Xxs
            | Codec::Iq2Xs
            | Codec::Iq2S
            | Codec::Iq3Xxs
            | Codec::Iq3S
            | Codec::Iq4Nl
            | Codec::Iq4Xs
            | Codec::Tq10
            | Codec::Tq20
            | Codec::Mxfp4
            | Codec::Nvfp4
            | Codec::Q1_0
            | Codec::Q2_0
            | Codec::BFloat8
            | Codec::Bf8E5M2
            | Codec::Bf4E2M1
            | Codec::Fp6E3M2,
        ) => Err(EmitError::CudaUnsupportedCodec { node }),
    }
}

fn render_elementwise(
    resolved: &BoundOp,
    entry: &str,
    quantized: &[Option<Codec>],
) -> Result<String, EmitError> {
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;

    let mut source = String::new();
    preamble(&mut source, element_type);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long total_elements;\n");
    source.push_str(&format!("    long extents[{rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!(
        "    long operand_strides[{operand_count}][{rank_len}];\n"
    ));
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    kernel_signature(&mut source, quantized, 0, gather_count, entry, element_type);
    source.push_str("    if (gid >= u.total_elements) { return; }\n");

    if rank > 0 {
        source.push_str(&format!("    long coord[{rank_len}];\n"));
        source.push_str("    long remaining = gid;\n");
        for dim in (0..rank).rev() {
            source.push_str(&format!(
                "    coord[{dim}] = remaining % u.extents[{dim}]; remaining /= u.extents[{dim}];\n"
            ));
        }
    }

    for (index, gather_slot) in gather_slots.iter().enumerate() {
        source.push_str(&format!("    long off{index} = u.operand_base[{index}];\n"));
        for dim in 0..rank {
            source.push_str(&format!(
                "    off{index} += coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slot {
            push_gather_fetch(
                &mut source,
                index,
                *slot,
                rank,
                "coord",
                &format!("off{index}"),
            );
        }
    }

    source.push_str(&format!(
        "    {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!(
            "    scratch[{index}] = {};\n",
            operand_read(resolved.node, index, &format!("off{index}"), codec)?
        ));
    }

    let result = push_body_steps(&mut source, resolved.element_body(), "    ", element_type);
    source.push_str(&format!("    out[gid] = {result};\n"));
    source.push_str("}\n");
    Ok(source)
}

#[allow(clippy::too_many_arguments)]
/// Emit the packed weight address used by the CPU quantized-matmul lowering.
/// A quantized reduce's weight operand is a contiguous row-major `[rows, k]`
/// slab even when its logical graph layout is a transposed/interleaved view.
/// Using the logical layout offset directly would turn a stride such as
/// `[0, 64, 1, 128]` into a packed address and read only every fourth Q8_0
/// block. The CPU path first flattens the varying output axes into a row;
/// CUDA must use the same physical contract.
fn push_packed_matmul_offset(
    source: &mut String,
    index: usize,
    layout: &Layout,
    output_axes: &[u16],
    resolved: &BoundOp,
    indent: &str,
) {
    #[cfg(feature = "std")]
    if std::env::var_os("PROXIMA_CUDA_LOGICAL_PACKED_ADDRESS").is_some() {
        source.push_str(&format!(
            "{indent}long off{index} = u.operand_base[{index}];\n"
        ));
        for dim in 0..resolved.extents.len() {
            source.push_str(&format!(
                "{indent}off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        return;
    }
    source.push_str(&format!("{indent}long packed_row{index} = 0;\n"));
    for axis in output_axes {
        let axis_usize = *axis as usize;
        if layout.stride(*axis) != 0 {
            source.push_str(&format!(
                "{indent}packed_row{index} = packed_row{index} * {} + full_coord[{axis_usize}];\n",
                resolved.extents[axis_usize]
            ));
        }
    }
    source.push_str(&format!(
        "{indent}long off{index} = u.operand_base[{index}] + packed_row{index} * u.reduction_total + r;\n"
    ));
}

// mirrors `crate::msl::elementwise_reduce_core::push_serial_reduce_body`'s
// own identical allow: one CUDA C reduce body needs every one of the
// fold's axis/rank/operand facts in scope at once, and this crate's other
// renderer already carries the same shape rather than a param-bundling
// struct for read-only rendering inputs.
#[allow(
    clippy::too_many_arguments,
    reason = "the parameters mirror the Metal twin's renderer inputs field for field, and bundling them into a struct would only relocate the count"
)]
fn push_serial_reduce_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
    reduce_dims: &[u16],
    rank: usize,
    rank_len: usize,
    output_rank: usize,
    output_rank_len: usize,
    reduce_rank: usize,
    reduce_rank_len: usize,
    operand_count: usize,
    gather_slots: &[Option<usize>],
    quantized: &[Option<Codec>],
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    is_broadcast_epilogue: bool,
) -> Result<(), EmitError> {
    source.push_str("    if (gid >= u.output_total) { return; }\n");

    source.push_str(&format!("    long full_coord[{rank_len}];\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }

    if output_rank > 0 {
        source.push_str(&format!("    long output_coord[{output_rank_len}];\n"));
        source.push_str("    long remaining = gid;\n");
        for index in (0..output_rank).rev() {
            source.push_str(&format!(
                "    output_coord[{index}] = remaining % u.output_extents[{index}]; \
                 remaining /= u.output_extents[{index}];\n"
            ));
        }
        if is_broadcast_epilogue {
            for dim in 0..rank {
                source.push_str(&format!("    full_coord[{dim}] = output_coord[{dim}];\n"));
            }
        } else {
            for (index, dim) in output_axes.iter().enumerate() {
                source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
            }
        }
    }

    let (init_expr, seeded_init) = fold_init_tokens(init);
    source.push_str(&format!("    {element_type} accumulator = {init_expr};\n"));
    source.push_str(&format!("    bool seeded = {seeded_init};\n"));

    source.push_str("    for (long r = 0; r < u.reduction_total; r++) {\n");
    if reduce_rank > 0 {
        source.push_str(&format!(
            "        long reduction_coord[{reduce_rank_len}];\n"
        ));
        source.push_str("        long remaining_r = r;\n");
        for index in (0..reduce_rank).rev() {
            source.push_str(&format!(
                "        reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
                 remaining_r /= u.reduction_extents[{index}];\n"
            ));
        }
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!(
                "        full_coord[{dim}] = reduction_coord[{index}];\n"
            ));
        }
    }

    for (index, gather_slot) in gather_slots.iter().enumerate() {
        if let Some(codec) = quantized[index] {
            let (_, layout, _) = &resolved.operands()[index];
            push_packed_matmul_offset(source, index, layout, output_axes, resolved, "        ");
            let _ = codec;
        } else {
            source.push_str(&format!(
                "        long off{index} = u.operand_base[{index}];\n"
            ));
            for dim in 0..rank {
                source.push_str(&format!(
                    "        off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
                ));
            }
        }
        if let Some(slot) = gather_slot {
            push_gather_fetch(
                source,
                index,
                *slot,
                rank,
                "full_coord",
                &format!("off{index}"),
            );
        }
    }
    source.push_str(&format!(
        "        {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!(
            "        scratch[{index}] = {};\n",
            operand_read(resolved.node, index, &format!("off{index}"), codec)?
        ));
    }
    let value_expr = push_body_steps(source, resolved.element_body(), "        ", element_type);
    source.push_str(&format!("        {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "        accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    if is_broadcast_epilogue {
        // The reduction produces one value per output coordinate, while the
        // broadcast epilogue must write that value over every reduced axis.
        // Rebuild each full coordinate after the fold; the fold loop leaves
        // `full_coord` at its final reduction coordinate.
        source.push_str("    long broadcast_full_coord[");
        source.push_str(&format!("{rank_len}];\n"));
        for dim in 0..rank {
            source.push_str(&format!(
                "    broadcast_full_coord[{dim}] = output_coord[{dim}];\n"
            ));
        }
        source.push_str(
            "    for (long broadcast_r = 0; broadcast_r < u.reduction_total; broadcast_r++) {\n",
        );
        if reduce_rank > 0 {
            source.push_str("        long broadcast_remaining = broadcast_r;\n");
            for index in (0..reduce_rank).rev() {
                source.push_str(&format!(
                    "        long broadcast_coord_{index} = broadcast_remaining % u.reduction_extents[{index}]; broadcast_remaining /= u.reduction_extents[{index}];\n"
                ));
            }
            for (index, dim) in reduce_dims.iter().enumerate() {
                source.push_str(&format!(
                    "        broadcast_full_coord[{dim}] = broadcast_coord_{index};\n"
                ));
            }
        }
        source.push_str("        long broadcast_out_offset = u.out_base;\n");
        for dim in 0..rank {
            source.push_str(&format!(
                "        broadcast_out_offset += broadcast_full_coord[{dim}] * u.broadcast_out_strides[{dim}];\n"
            ));
        }
        push_reduce_epilogue_write(
            source,
            epilogue_body,
            epilogue_operands,
            rank,
            element_type,
            "        ",
            |dim| format!("broadcast_full_coord[{dim}]"),
            "accumulator",
            "broadcast_out_offset",
        );
        source.push_str("    }\n");
        return Ok(());
    }
    source.push_str("    long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "    out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_rank,
        element_type,
        "    ",
        |dim| {
            if output_rank > 0 {
                format!("output_coord[{dim}]")
            } else {
                "0".to_string()
            }
        },
        "accumulator",
        "out_offset",
    );
    Ok(())
}

fn logical_packed_address() -> bool {
    #[cfg(feature = "std")]
    {
        std::env::var_os("PROXIMA_CUDA_LOGICAL_PACKED_ADDRESS").is_some()
    }
    #[cfg(not(feature = "std"))]
    {
        false
    }
}

fn push_reduction_coord_decompose(
    source: &mut String,
    from: &str,
    reduce_rank: usize,
    reduce_rank_len: usize,
    indent: &str,
) {
    source.push_str(&format!("{indent}long reduction_coord[{reduce_rank_len}];\n"));
    source.push_str(&format!("{indent}long remaining_r = {from};\n"));
    for index in (0..reduce_rank).rev() {
        source.push_str(&format!(
            "{indent}reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
             remaining_r /= u.reduction_extents[{index}];\n"
        ));
    }
}

fn push_reduction_coord_carry(source: &mut String, reduce_rank: usize) {
    source.push_str("            reduction_coord[");
    source.push_str(&format!("{}]++;\n", reduce_rank - 1));
    for index in (1..reduce_rank).rev() {
        source.push_str(&format!(
            "            if (reduction_coord[{index}] == u.reduction_extents[{index}]) {{ \
             reduction_coord[{index}] = 0; reduction_coord[{}]++; }}\n",
            index - 1
        ));
    }
}

fn codec_slots(quantized: &[Option<Codec>], codec: Codec) -> Vec<usize> {
    quantized
        .iter()
        .enumerate()
        .filter(|(_, slot_codec)| **slot_codec == Some(codec))
        .map(|(index, _)| index)
        .collect()
}

fn push_q4_0_chunk_decode(source: &mut String, index: usize) {
    source.push_str(&format!("        float q4_0_chunk{index}[8];\n"));
    source.push_str(&format!(
        "        if ((off{index} & 7) == 0 && chunk + 8 <= u.reduction_total) {{\n\
         \x20           q4_0_chunk8(in{index} + (off{index} / {Q4_0_BLOCK_ELEMENTS}) * {Q4_0_BLOCK_BYTES}, \
         (unsigned int)(off{index} % {Q4_0_BLOCK_ELEMENTS}), q4_0_chunk{index});\n\
         \x20       }} else {{\n\
         \x20           for (int slot = 0; slot < 8 && chunk + slot < u.reduction_total; slot++) {{\n\
         \x20               long element = off{index} + slot;\n\
         \x20               q4_0_chunk{index}[slot] = q4_0_element(in{index} + (element / {Q4_0_BLOCK_ELEMENTS}) * {Q4_0_BLOCK_BYTES}, \
         (unsigned int)(element % {Q4_0_BLOCK_ELEMENTS}));\n\
         \x20           }}\n\
         \x20       }}\n"
    ));
}

fn push_q6k_chunk_decode(source: &mut String, index: usize) {
    source.push_str(&format!("        float q6k_chunk{index}[8];\n"));
    source.push_str(&format!(
        "        if ((off{index} & 7) == 0 && chunk + 8 <= u.reduction_total) {{\n\
         \x20           q6k_chunk8(in{index} + (off{index} / {Q4K_BLOCK_ELEMENTS}) * {Q6K_BLOCK_BYTES}, \
         (unsigned int)(off{index} % {Q4K_BLOCK_ELEMENTS}), q6k_chunk{index});\n\
         \x20       }} else {{\n\
         \x20           for (int slot = 0; slot < 8 && chunk + slot < u.reduction_total; slot++) {{\n\
         \x20               long element = off{index} + slot;\n\
         \x20               q6k_chunk{index}[slot] = q6k_element(in{index} + (element / {Q4K_BLOCK_ELEMENTS}) * {Q6K_BLOCK_BYTES}, \
         (unsigned int)(element % {Q4K_BLOCK_ELEMENTS}));\n\
         \x20           }}\n\
         \x20       }}\n"
    ));
}

fn push_q8_0_chunk_decode(source: &mut String, index: usize) {
    source.push_str(&format!("        float q8_0_chunk{index}[8];\n"));
    source.push_str(&format!(
        "        if ((off{index} & 7) == 0 && chunk + 8 <= u.reduction_total) {{\n\
         \x20           q8_0_chunk8(in{index} + (off{index} / {Q8_0_BLOCK_ELEMENTS}) * {Q8_0_BLOCK_BYTES}, \
         (unsigned int)(off{index} % {Q8_0_BLOCK_ELEMENTS}), q8_0_chunk{index});\n\
         \x20       }} else {{\n\
         \x20           for (int slot = 0; slot < 8 && chunk + slot < u.reduction_total; slot++) {{\n\
         \x20               long element = off{index} + slot;\n\
         \x20               q8_0_chunk{index}[slot] = q8_0_element(in{index} + (element / {Q8_0_BLOCK_ELEMENTS}) * {Q8_0_BLOCK_BYTES}, \
         (unsigned int)(element % {Q8_0_BLOCK_ELEMENTS}));\n\
         \x20           }}\n\
         \x20       }}\n"
    ));
}

/// The warp-shuffle cooperative fold — the CUDA counterpart of
/// `crate::msl::push_cooperative_reduce_body`'s non-tiled, non-row-blocked
/// arm (this module has no port of either fast path; see the module doc).
/// One warp per output element: each of the [`WARP_SIZE`] lanes strides
/// through the reduction space (`r = lane, lane + WARP_SIZE, ...`), folds its
/// own private run, then a five-step `__shfl_down_sync` butterfly combines
/// all 32 lanes' partials — `log2(32) == 5` steps, the standard warp-reduce
/// idiom.
#[allow(clippy::too_many_arguments)]
fn push_cooperative_reduce_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
    reduce_dims: &[u16],
    rank: usize,
    quantized: &[Option<Codec>],
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    is_broadcast_epilogue: bool,
) -> Result<(), EmitError> {
    let rank_len = rank.max(1);
    let output_rank = output_axes.len();
    let output_rank_len = output_rank.max(1);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);
    let operand_count = resolved.operands().len();
    // Packed cooperative reductions are already gated by the caller's
    // reassociation permission.  Give each lane a contiguous small chunk so
    // neighboring lanes read neighboring quantized bytes; the prior
    // lane-strided walk made a warp touch 32 separated elements before
    // returning to the same quantization block.
    let packed_chunked = quantized.iter().any(Option::is_some);

    source.push_str(&format!("    long output_index = gid / {WARP_SIZE};\n"));
    source.push_str("    if (output_index >= u.output_total) { return; }\n");
    source.push_str(&format!(
        "    unsigned int lane = (unsigned int)(gid % {WARP_SIZE});\n"
    ));

    source.push_str(&format!("    long full_coord[{rank_len}];\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }

    if output_rank > 0 {
        source.push_str(&format!("    long output_coord[{output_rank_len}];\n"));
        source.push_str("    long remaining = output_index;\n");
        for index in (0..output_rank).rev() {
            let extent_slot = if is_broadcast_epilogue {
                output_axes[index] as usize
            } else {
                index
            };
            source.push_str(&format!(
                "    output_coord[{index}] = remaining % u.output_extents[{extent_slot}]; \
                 remaining /= u.output_extents[{extent_slot}];\n"
            ));
        }
        for (index, dim) in output_axes.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
    }

    for (index, slot) in gather_slots(resolved).into_iter().enumerate() {
        if let Some(gather_slot) = slot {
            push_hoisted_gather_fetch(source, index, gather_slot, rank);
        }
    }

    let (init_expr, seeded_init) = fold_init_tokens(init);
    let identity = cooperative_identity_token(resolved.node, reduce_op)?;
    source.push_str(&format!("    {element_type} accumulator;\n"));
    source.push_str("    bool seeded;\n");
    source.push_str("    if (lane == 0u) {\n");
    source.push_str(&format!("        accumulator = {init_expr};\n"));
    source.push_str(&format!("        seeded = {seeded_init};\n"));
    source.push_str("    } else {\n");
    source.push_str(&format!("        accumulator = {identity};\n"));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    let chunk_hoisted = packed_chunked && !logical_packed_address();
    let q4_0_slots = if chunk_hoisted {
        codec_slots(quantized, Codec::Q4_0)
    } else {
        Vec::new()
    };
    let q8_0_slots = if chunk_hoisted {
        codec_slots(quantized, Codec::Q8_0)
    } else {
        Vec::new()
    };
    let q6k_slots = if chunk_hoisted {
        codec_slots(quantized, Codec::Q6K)
    } else {
        Vec::new()
    };
    let chunk_hoisted_index = |index: usize| {
        q4_0_slots.contains(&index) || q8_0_slots.contains(&index) || q6k_slots.contains(&index)
    };

    if packed_chunked {
        source.push_str(&format!(
            "    for (long chunk = (long)lane * 8; chunk < u.reduction_total; chunk += {} * 8) {{\n",
            WARP_SIZE
        ));
        if reduce_rank > 0 {
            push_reduction_coord_decompose(source, "chunk", reduce_rank, reduce_rank_len, "        ");
        }
        if chunk_hoisted {
            if !q4_0_slots.is_empty() || !q8_0_slots.is_empty() || !q6k_slots.is_empty() {
                source.push_str("        long r = chunk;\n");
            }
            for &index in &q4_0_slots {
                let (_, layout, _) = &resolved.operands()[index];
                push_packed_matmul_offset(source, index, layout, output_axes, resolved, "        ");
                push_gather_base_add(source, resolved, index, "        ");
                push_q4_0_chunk_decode(source, index);
            }
            for &index in &q8_0_slots {
                let (_, layout, _) = &resolved.operands()[index];
                push_packed_matmul_offset(source, index, layout, output_axes, resolved, "        ");
                push_gather_base_add(source, resolved, index, "        ");
                push_q8_0_chunk_decode(source, index);
            }
            for &index in &q6k_slots {
                let (_, layout, _) = &resolved.operands()[index];
                push_packed_matmul_offset(source, index, layout, output_axes, resolved, "        ");
                push_gather_base_add(source, resolved, index, "        ");
                push_q6k_chunk_decode(source, index);
            }
        }
        source.push_str("        for (long inner = 0; inner < 8 && chunk + inner < u.reduction_total; inner++) {\n");
        source.push_str("            long r = chunk + inner;\n");
    } else {
        source.push_str(&format!(
            "    for (long r = (long)lane; r < u.reduction_total; r += {WARP_SIZE}) {{\n"
        ));
    }
    if reduce_rank > 0 && !packed_chunked {
        push_reduction_coord_decompose(source, "r", reduce_rank, reduce_rank_len, "        ");
    }
    if reduce_rank > 0 {
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!(
                "        full_coord[{dim}] = reduction_coord[{index}];\n"
            ));
        }
    }

    for (index, &codec) in quantized.iter().enumerate() {
        if chunk_hoisted_index(index) {
            continue;
        }
        if codec.is_some() {
            let (_, layout, _) = &resolved.operands()[index];
            push_packed_matmul_offset(source, index, layout, output_axes, resolved, "        ");
        } else {
            source.push_str(&format!(
                "        long off{index} = u.operand_base[{index}];\n"
            ));
            for dim in 0..rank {
                source.push_str(&format!(
                    "        off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
                ));
            }
        }
        push_gather_base_add(source, resolved, index, "        ");
    }
    source.push_str(&format!(
        "        {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, &codec) in quantized.iter().enumerate() {
        if q4_0_slots.contains(&index) {
            source.push_str(&format!("        scratch[{index}] = q4_0_chunk{index}[inner];\n"));
            continue;
        }
        if q8_0_slots.contains(&index) {
            source.push_str(&format!("        scratch[{index}] = q8_0_chunk{index}[inner];\n"));
            continue;
        }
        if q6k_slots.contains(&index) {
            source.push_str(&format!("        scratch[{index}] = q6k_chunk{index}[inner];\n"));
            continue;
        }
        source.push_str(&format!(
            "        scratch[{index}] = {};\n",
            operand_read(resolved.node, index, &format!("off{index}"), codec)?
        ));
    }
    let value_expr = push_body_steps(source, resolved.element_body(), "        ", element_type);
    source.push_str(&format!("        {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "        accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("        seeded = true;\n");
    if packed_chunked && reduce_rank > 0 {
        push_reduction_coord_carry(source, reduce_rank);
    }
    source.push_str("    }\n");
    if packed_chunked {
        source.push_str("    }\n");
    }

    source.push_str("    #pragma unroll\n");
    source.push_str(&format!(
        "    for (int shift = {}; shift > 0; shift >>= 1) {{\n",
        WARP_SIZE / 2
    ));
    source.push_str(&format!(
        "        {element_type} shuffled = __shfl_down_sync(0xffffffffu, accumulator, shift);\n"
    ));
    let shuffle_expr = shuffle_combine_expr(resolved.node, reduce_op, "accumulator", "shuffled")?;
    source.push_str(&format!("        accumulator = {shuffle_expr};\n"));
    source.push_str("    }\n");

    if is_broadcast_epilogue {
        // Every lane participates in the broadcast write below. The
        // down-shuffle tree leaves the complete reduction only in lane 0;
        // without this broadcast, lanes 1..31 reuse their partial sums as
        // if they were the norm, corrupting every other output coordinate.
        source.push_str("    accumulator = __shfl_sync(0xffffffffu, accumulator, 0);\n");
        // Recreate the non-reduction coordinate after the fold, whose loop
        // leaves `full_coord` holding its last reduction coordinate. Each
        // lane then owns a strided share of the full output row.
        for dim in 0..rank {
            source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
        }
        for (index, dim) in output_axes.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
        source.push_str(&format!(
            "    for (long r = (long)lane; r < u.reduction_total; r += {WARP_SIZE}) {{\n"
        ));
        if reduce_rank > 0 {
            source.push_str(&format!(
                "        long broadcast_reduce_coord[{reduce_rank_len}];\n"
            ));
            source.push_str("        long broadcast_remaining = r;\n");
            for index in (0..reduce_rank).rev() {
                source.push_str(&format!(
                    "        broadcast_reduce_coord[{index}] = broadcast_remaining % u.reduction_extents[{index}]; broadcast_remaining /= u.reduction_extents[{index}];\n"
                ));
            }
            source.push_str(&format!("        long broadcast_full_coord[{rank_len}];\n"));
            for dim in 0..rank {
                source.push_str(&format!(
                    "        broadcast_full_coord[{dim}] = full_coord[{dim}];\n"
                ));
            }
            for (index, dim) in reduce_dims.iter().enumerate() {
                source.push_str(&format!(
                    "        broadcast_full_coord[{dim}] = broadcast_reduce_coord[{index}];\n"
                ));
            }
            source.push_str("        long broadcast_out_offset = u.out_base;\n");
            for dim in 0..rank {
                source.push_str(&format!(
                    "        broadcast_out_offset += broadcast_full_coord[{dim}] * u.broadcast_out_strides[{dim}];\n"
                ));
            }
            push_reduce_epilogue_write(
                source,
                epilogue_body,
                epilogue_operands,
                rank,
                element_type,
                "        ",
                |dim| format!("broadcast_full_coord[{dim}]"),
                "accumulator",
                "broadcast_out_offset",
            );
        }
        source.push_str("    }\n");
        return Ok(());
    }
    source.push_str("    if (lane != 0u) { return; }\n");
    source.push_str("    long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "    out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_rank,
        element_type,
        "    ",
        |dim| {
            if output_rank > 0 {
                format!("output_coord[{dim}]")
            } else {
                "0".to_string()
            }
        },
        "accumulator",
        "out_offset",
    );
    Ok(())
}

fn render_reduce(
    resolved: &BoundOp,
    entry: &str,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
) -> Result<String, EmitError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        epilogue_body,
        epilogue_operands,
        epilogue_broadcast_axes,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "keep::reduce fold",
            found: resolved.kind.name(),
        });
    };
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let is_broadcast_epilogue = !epilogue_broadcast_axes.is_empty();
    let output_rank = if is_broadcast_epilogue {
        rank
    } else {
        output_axes.len()
    };
    let output_rank_len = output_rank.max(1);
    let reduce_dims = reduction_dims(resolved, output_axes);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let epilogue_operand_count = epilogue_operands.len();

    let mut source = String::new();
    preamble(&mut source, element_type);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long output_total;\n");
    source.push_str("    long reduction_total;\n");
    source.push_str(&format!("    long output_extents[{output_rank_len}];\n"));
    source.push_str(&format!("    long reduction_extents[{reduce_rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!(
        "    long operand_strides[{operand_count}][{rank_len}];\n"
    ));
    source.push_str("    long out_base;\n");
    source.push_str(&format!("    long out_strides[{rank_len}];\n"));
    if epilogue_operand_count > 0 {
        source.push_str(&format!(
            "    long epilogue_operand_base[{epilogue_operand_count}];\n"
        ));
        source.push_str(&format!(
            "    long epilogue_operand_strides[{epilogue_operand_count}][{output_rank_len}];\n"
        ));
    }
    if is_broadcast_epilogue {
        source.push_str(&format!("    long broadcast_out_strides[{rank_len}];\n"));
    }
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    kernel_signature(
        &mut source,
        quantized,
        epilogue_operand_count,
        gather_count,
        entry,
        element_type,
    );

    if reduce_is_cooperative(resolved, quantized, numeric_policy) {
        push_cooperative_reduce_body(
            &mut source,
            resolved,
            *reduce_op,
            *init,
            output_axes,
            &reduce_dims,
            rank,
            quantized,
            element_type,
            epilogue_body,
            epilogue_operands,
            is_broadcast_epilogue,
        )?;
    } else {
        push_serial_reduce_body(
            &mut source,
            resolved,
            *reduce_op,
            *init,
            output_axes,
            &reduce_dims,
            rank,
            rank_len,
            output_rank,
            output_rank_len,
            reduce_rank,
            reduce_rank_len,
            operand_count,
            &gather_slots,
            quantized,
            element_type,
            epilogue_body,
            epilogue_operands,
            is_broadcast_epilogue,
        )?;
    }
    source.push_str("}\n");
    Ok(source)
}

fn render_scan(
    resolved: &BoundOp,
    entry: &str,
    quantized: &[Option<Codec>],
) -> Result<String, EmitError> {
    let BoundOpKind::Reduce {
        reduce_op, init, ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "keep::scan fold",
            found: resolved.kind.name(),
        });
    };
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let outer_rank = rank.saturating_sub(1);
    let outer_rank_len = outer_rank.max(1);
    let last_dim = rank.saturating_sub(1);
    let operand_count = resolved.operands().len();
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;

    let mut source = String::new();
    preamble(&mut source, element_type);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long outer_total;\n");
    source.push_str("    long inner_len;\n");
    source.push_str(&format!("    long outer_extents[{outer_rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!(
        "    long operand_strides[{operand_count}][{rank_len}];\n"
    ));
    source.push_str("    long out_base;\n");
    source.push_str(&format!("    long out_strides[{rank_len}];\n"));
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    kernel_signature(&mut source, quantized, 0, gather_count, entry, element_type);
    source.push_str("    if (gid != 0) { return; }\n");

    if outer_rank > 0 {
        source.push_str(&format!("    long outer_coord[{outer_rank_len}];\n"));
    }

    let (init_expr, seeded_init) = fold_init_tokens(*init);
    source.push_str(&format!("    {element_type} accumulator = {init_expr};\n"));
    source.push_str(&format!("    bool seeded = {seeded_init};\n"));

    source.push_str("    for (long outer = 0; outer < u.outer_total; outer++) {\n");
    if outer_rank > 0 {
        source.push_str("        long remaining = outer;\n");
        for dim in (0..outer_rank).rev() {
            source.push_str(&format!(
                "        outer_coord[{dim}] = remaining % u.outer_extents[{dim}]; \
                 remaining /= u.outer_extents[{dim}];\n"
            ));
        }
    }
    for (index, gather_slot) in gather_slots.iter().enumerate() {
        source.push_str(&format!(
            "        long running{index} = u.operand_base[{index}];\n"
        ));
        for dim in 0..outer_rank {
            source.push_str(&format!(
                "        running{index} += outer_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slot {
            source.push_str(&format!(
                "        long gather_running{index} = u.gather_index_base[{slot}];\n"
            ));
            for dim in 0..outer_rank {
                source.push_str(&format!(
                    "        gather_running{index} += outer_coord[{dim}] * u.gather_index_strides[{slot}][{dim}];\n"
                ));
            }
        }
    }
    source.push_str("        long out_running = u.out_base;\n");
    for dim in 0..outer_rank {
        source.push_str(&format!(
            "        out_running += outer_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }

    source.push_str("        for (long step = 0; step < u.inner_len; step++) {\n");
    source.push_str(&format!(
        "            {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, gather_slot) in gather_slots.iter().enumerate() {
        if let Some(slot) = gather_slot {
            source.push_str(&format!(
                "            long fetched{index} = (long)gather_idx{slot}[gather_running{index}];\n"
            ));
            push_gather_fault_check(&mut source, index, *slot, "            ");
            source.push_str(&format!(
                "            fetched{index} = max((long)0, min(fetched{index}, u.gather_extent[{slot}] - 1));\n"
            ));
            source.push_str(&format!(
                "            long read_off{index} = running{index} + fetched{index} * u.gather_element_stride[{slot}];\n"
            ));
            source.push_str(&format!(
                "            scratch[{index}] = {};\n",
                operand_read(
                    resolved.node,
                    index,
                    &format!("read_off{index}"),
                    quantized[index]
                )?
            ));
            source.push_str(&format!(
                "            gather_running{index} += u.gather_index_strides[{slot}][{last_dim}];\n"
            ));
        } else {
            source.push_str(&format!(
                "            scratch[{index}] = {};\n",
                operand_read(
                    resolved.node,
                    index,
                    &format!("running{index}"),
                    quantized[index]
                )?
            ));
        }
        source.push_str(&format!(
            "            running{index} += u.operand_strides[{index}][{last_dim}];\n"
        ));
    }
    let value_expr = push_body_steps(
        &mut source,
        resolved.element_body(),
        "            ",
        element_type,
    );
    source.push_str(&format!(
        "            {element_type} value = {value_expr};\n"
    ));
    let combine_expr = scalar_op_expr(*reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "            accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("            seeded = true;\n");
    source.push_str("            out[out_running] = accumulator;\n");
    source.push_str(&format!(
        "            out_running += u.out_strides[{last_dim}];\n"
    ));
    source.push_str("        }\n");
    source.push_str("    }\n");
    source.push_str("}\n");
    Ok(source)
}

/// CUDA C source for unpacking one element of a `Q4_K` super-block — ports
/// `crate::msl::Q4K_UNPACK_MSL`'s `q4k_element` exactly (see that constant's
/// own doc for the bit layout, `get_scale_min_k4`'s two easy-to-get-wrong
/// details, and the nibble-order gotcha). This module skips the header-
/// amortized `q4k_header_for`/`q4k_value`/`q4k_run8` trio: those exist only
/// to feed `crate::msl`'s row-blocked fast path, which has no CUDA port yet
/// (see the module doc) — every read here goes through the fully generic
/// per-element accessor.
pub const Q4K_UNPACK_CUDA: &str = "\
__device__ __forceinline__ unsigned char q4k_scale(const unsigned char *scales, unsigned int sub_block) {
    if (sub_block < 4u) { return scales[sub_block] & 63; }
    return (scales[sub_block + 4u] & 0x0F) | ((scales[sub_block - 4u] >> 6) << 4);
}

__device__ __forceinline__ unsigned char q4k_min(const unsigned char *scales, unsigned int sub_block) {
    if (sub_block < 4u) { return scales[sub_block + 4u] & 63; }
    return (scales[sub_block + 4u] >> 4) | ((scales[sub_block] >> 6) << 4);
}

__device__ __forceinline__ float q4k_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    unsigned short dmin_bits = (unsigned short)((unsigned int)block[2] | ((unsigned int)block[3] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    float dmin = __half2float(__ushort_as_half(dmin_bits));

    const unsigned char *scales = block + 4;
    const unsigned char *qs = block + 16;

    unsigned int group = index / 64u;
    unsigned int within = index % 64u;
    int low_nibble = within < 32u;
    unsigned int sub_block = 2u * group + (low_nibble ? 0u : 1u);
    unsigned int byte_index = group * 32u + (within % 32u);

    float scale = d * (float)q4k_scale(scales, sub_block);
    float minimum = dmin * (float)q4k_min(scales, sub_block);
    unsigned char nibble = low_nibble ? (qs[byte_index] & 0x0Fu) : (qs[byte_index] >> 4u);
    return scale * (float)nibble - minimum;
}
";

pub const Q4K_BLOCK_BYTES: usize = proxima_gguf::quant::q4_k::BLOCK_BYTES;
pub const Q4K_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q4_k::QK_K;

/// CUDA C source for unpacking one element of a `Q6_K` super-block — ports
/// `crate::msl::Q6K_UNPACK_MSL`'s `q6k_element`/`q6k_value` (see that
/// constant's own doc for the two-half/four-lane/`qh`-shared bit layout and
/// why `d` trails the block rather than leading it).
pub const Q6K_UNPACK_CUDA: &str = "\
__device__ __forceinline__ float q6k_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[208] | ((unsigned int)block[209] << 8));
    float d = __half2float(__ushort_as_half(d_bits));

    unsigned int half_index = index / 128u;
    unsigned int local = index % 128u;
    unsigned int l = local % 32u;
    unsigned int lane = local / 32u;
    unsigned int sub_block_in_half = l / 16u;

    const unsigned char *ql = block + half_index * 64u;
    const unsigned char *qh = block + 128u + half_index * 32u;
    const unsigned char *scales = block + 192u;

    unsigned char ql_byte = (lane % 2u == 0u) ? ql[l] : ql[l + 32u];
    unsigned char nibble = (lane < 2u) ? (ql_byte & 0x0Fu) : (ql_byte >> 4u);
    unsigned char high2 = (qh[l] >> (unsigned char)(lane * 2u)) & 0x03u;
    unsigned char level = nibble | (high2 << 4u);

    unsigned char scale_byte = scales[half_index * 8u + sub_block_in_half + lane * 2u];
    float scale = (float)(signed char)scale_byte;
    float quant = (float)level - 32.0f;
    return d * scale * quant;
}
__device__ __forceinline__ void q6k_chunk8(const unsigned char *block, unsigned int index, float *out) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[208] | ((unsigned int)block[209] << 8));
    float d = __half2float(__ushort_as_half(d_bits));

    unsigned int half_index = index / 128u;
    unsigned int local = index % 128u;
    unsigned int l = local % 32u;
    unsigned int lane = local / 32u;
    unsigned int sub_block_in_half = l / 16u;

    const unsigned short *ql_pairs = (const unsigned short *)(block + half_index * 64u + (lane % 2u) * 32u + l);
    const unsigned short *qh_pairs = (const unsigned short *)(block + 128u + half_index * 32u + l);
    float scale = (float)(signed char)block[192u + half_index * 8u + sub_block_in_half + lane * 2u];
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        unsigned int ql_word = (unsigned int)ql_pairs[pair];
        unsigned int qh_word = (unsigned int)qh_pairs[pair];
        #pragma unroll
        for (int byte = 0; byte < 2; byte++) {
            unsigned int ql_byte = (ql_word >> (8u * byte)) & 0xFFu;
            unsigned int qh_byte = (qh_word >> (8u * byte)) & 0xFFu;
            unsigned int nibble = (lane < 2u) ? (ql_byte & 0x0Fu) : (ql_byte >> 4u);
            unsigned int high2 = (qh_byte >> (lane * 2u)) & 0x03u;
            unsigned int level = nibble | (high2 << 4u);
            out[2 * pair + byte] = d * scale * ((float)level - 32.0f);
        }
    }
}
";

pub const Q6K_BLOCK_BYTES: usize = proxima_gguf::quant::q6_k::BLOCK_BYTES;

/// CUDA C source for unpacking one element of a `Q5_K` super-block — ports
/// `crate::msl::Q5K_UNPACK_MSL`'s `q5k_element`/`q5k_value` (see that
/// constant's own doc for the `qh` high-bit-plane layout distinct from both
/// `Q4_K` and `Q6_K`).
pub const Q5K_UNPACK_CUDA: &str = "\
__device__ __forceinline__ float q5k_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    unsigned short dmin_bits = (unsigned short)((unsigned int)block[2] | ((unsigned int)block[3] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    float dmin = __half2float(__ushort_as_half(dmin_bits));
    const unsigned char *scales = block + 4;
    const unsigned char *qh = block + 16;
    const unsigned char *qs = block + 48;

    unsigned int chunk = index / 64u;
    unsigned int within = index % 64u;
    int low = within < 32u;
    unsigned int sub_block = 2u * chunk + (low ? 0u : 1u);
    unsigned int offset = within % 32u;

    unsigned char scale = q4k_scale(scales, sub_block);
    unsigned char minimum = q4k_min(scales, sub_block);
    unsigned char mask = low ? (unsigned char)(1u << (2u * chunk)) : (unsigned char)(2u << (2u * chunk));

    unsigned char qs_byte = qs[chunk * 32u + offset];
    unsigned char nibble = low ? (qs_byte & 0x0Fu) : (qs_byte >> 4u);
    float high = (qh[offset] & mask) != 0u ? 16.0f : 0.0f;
    return d * (float)scale * ((float)nibble + high) - dmin * (float)minimum;
}
";

pub const Q5K_BLOCK_BYTES: usize = proxima_gguf::quant::q5_k::BLOCK_BYTES;

/// CUDA C source for unpacking one element of a `Q8_0` block — ports
/// `crate::msl::q8_0_unpack_msl`'s `q8_0_element` exactly: a flat 32-element
/// block, one `f16` scale, no sub-block structure.
pub const Q8_0_UNPACK_CUDA: &str = "\
__device__ __forceinline__ float q8_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    signed char level = (signed char)block[2u + index];
    return (float)level * d;
}
__device__ __forceinline__ void q8_0_chunk8(const unsigned char *block, unsigned int index, float *out) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        out[slot] = (float)((signed char)block[2u + index + slot]) * d;
    }
}
";

pub const Q8_0_BLOCK_BYTES: usize = proxima_gguf::quant::q8_0::BLOCK_BYTES;
pub const Q8_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q8_0::QK8_0;

/// CUDA C source for unpacking one element of a `Q4_0` block — ports
/// `crate::msl::Q4_0_UNPACK_MSL`'s `q4_0_element` exactly: llama.cpp's
/// simplest legacy 4-bit format, `value = (nibble - 8) * d`.
pub const Q4_0_UNPACK_CUDA: &str = "\
__device__ __forceinline__ float q4_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned char byte = block[2u + (index % 16u)];
    int nibble = (index < 16u) ? (int)(byte & 0x0Fu) : (int)(byte >> 4u);
    return (float)(nibble - 8) * d;
}
__device__ __forceinline__ void q4_0_chunk8(const unsigned char *block, unsigned int index, float *out) {
    const unsigned short *halves = (const unsigned short *)block;
    float d = __half2float(__ushort_as_half(halves[0]));
    const unsigned short *pairs = halves + 1u + (index % 16u) / 2u;
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        unsigned int word = (unsigned int)pairs[pair];
        unsigned int low = word & 0xFFu;
        unsigned int high = word >> 8u;
        int low_nibble = (index < 16u) ? (int)(low & 0x0Fu) : (int)(low >> 4u);
        int high_nibble = (index < 16u) ? (int)(high & 0x0Fu) : (int)(high >> 4u);
        out[2 * pair] = (float)(low_nibble - 8) * d;
        out[2 * pair + 1] = (float)(high_nibble - 8) * d;
    }
}
";

pub const Q4_0_BLOCK_BYTES: usize = proxima_gguf::quant::q4_0::BLOCK_BYTES;
pub const Q4_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q4_0::QK4_0;

pub const FLOAT16_BLOCK_BYTES: usize = proxima_gguf::quant::f16::BLOCK_BYTES;
pub const FLOAT16_BLOCK_ELEMENTS: usize = proxima_gguf::quant::f16::QK_F16;
pub const BFLOAT16_BLOCK_BYTES: usize = proxima_gguf::quant::bf16::BLOCK_BYTES;
pub const BFLOAT16_BLOCK_ELEMENTS: usize = proxima_gguf::quant::bf16::QK_BF16;

/// Widens one `bfloat16` element to `float` by shifting it into the high 16
/// bits of a 32-bit word and reinterpreting — ports
/// `crate::msl::layout_render`'s `bf16_element` exactly. CUDA's own
/// `__nv_bfloat16` (`<cuda_bf16.h>`) is a native storage type on modern
/// architectures, but this crate targets no minimum compute capability yet,
/// so it stays on the same portable widen-by-shift every `sm_` generation
/// supports rather than pulling in a second half-precision header.
pub const BF16_UNPACK_CUDA: &str = "\
__device__ __forceinline__ float bf16_element(const unsigned char *block, unsigned int index) {
    (void)index;
    unsigned int bits = ((unsigned int)block[0] | ((unsigned int)block[1] << 8)) << 16u;
    return __uint_as_float(bits);
}
";

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_tensor::{
        AxisTerm, DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind,
        infer, map,
    };

    use super::*;
    use proxima_tensor::test_support::Lcg;

    fn no_packed() -> PackedOperands {
        BTreeMap::new()
    }

    fn elementwise_tanh_op(extent: u32) -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(
            &program,
            &shapes,
            &[NodeId((program.len() - 1) as u32)],
            proxima_tensor::NumericPolicy::default(),
        )
        .expect("bind succeeds");
        bound.into_iter().next().expect("one bound op")
    }

    #[test]
    fn elementwise_chain_emits_cuda_with_the_expected_shape() {
        let bound = elementwise_tanh_op(8);
        let kernel = emit_cuda(&bound, &no_packed()).expect("emit succeeds");
        assert!(kernel.source.contains("extern \"C\" __global__"));
        assert!(kernel.source.contains("tanhf("));
        assert_eq!(kernel.bindings.len(), 3);
        assert_eq!(kernel.grid.threads, 8);
        assert!(kernel.grid.block_width.is_none());
    }

    #[test]
    fn leaf_iota_and_constant_emit_a_uniform_fill_kernel() {
        for leaf in [
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(8),
            },
            Op::Constant {
                dtype: DType::Float32,
                shape: vec![Extent::Static(8)],
                value: 0.25,
            },
        ] {
            let program = vec![leaf];
            let shapes = infer(&program, &[]).expect("leaf shape inference succeeds");
            let bound = bind(
                &program,
                &shapes,
                &[NodeId(0)],
                proxima_tensor::NumericPolicy::default(),
            )
            .expect("leaf binding succeeds")
            .pop()
            .expect("one leaf bound op");
            let kernel = emit_cuda(&bound, &no_packed()).expect("leaf emits");
            assert_eq!(
                kernel.bindings,
                vec![Binding::Output(NodeId(0)), Binding::Uniforms]
            );
            assert!(kernel.source.contains("u.total_elements"));
            assert_eq!(pack_cuda_uniforms(&bound).expect("leaf packs").len(), 8);
        }
    }

    fn moe_top_k_op(expert_count: u64, top_k: u64, stacked: bool) -> BoundOp {
        let nodes = |start: u32| {
            (0..top_k as u32)
                .map(|index| NodeId(start + index))
                .collect()
        };
        BoundOp {
            node: NodeId(100),
            dtype: DType::Float32,
            extents: vec![5],
            kind: BoundOpKind::MoeTopK {
                operands: vec![(
                    NodeId(1),
                    Layout {
                        base: 0,
                        strides: vec![expert_count as i64, 1].into(),
                    },
                    None,
                )],
                expert_count,
                top_k,
                routes: nodes(100),
                weights: nodes(200),
                weight_total: NodeId(300),
                stacked: stacked.then_some((NodeId(400), NodeId(401))),
            },
        }
    }

    #[test]
    fn moe_topk_emits_one_block_per_token_with_warp_shuffle_reduction_and_extra_outputs() {
        let bound = moe_top_k_op(32, 8, false);

        let kernel = emit_cuda(&bound, &no_packed()).expect("a 32 expert router emits");

        assert_eq!(kernel.grid.block_width, Some(32));
        assert_eq!(kernel.grid.threads, 5 * 32);
        assert_eq!(kernel.grid.blocks(), 5);
        assert_eq!(
            kernel.bindings,
            vec![
                Binding::Input(NodeId(1)),
                Binding::Output(NodeId(100)),
                Binding::Uniforms
            ]
        );
        assert_eq!(kernel.extra_outputs.len(), 16);
        assert!(kernel.extra_outputs.iter().all(|(_, length)| *length == 5));
        assert!(kernel.source.contains("__shfl_xor_sync"));
        assert!(
            kernel
                .source
                .contains("masked == max_value && tid < expert_count")
        );
        assert!(kernel.source.contains("float* __restrict__ extra15"));
        assert!(!kernel.source.contains("extra16"));
        assert_eq!(pack_cuda_uniforms(&bound).expect("moe packs").len(), 8);
    }

    #[test]
    fn moe_topk_pads_a_non_warp_multiple_expert_count_to_whole_warps_and_appends_stacked_outputs() {
        let kernel = emit_cuda(&moe_top_k_op(40, 6, true), &no_packed())
            .expect("a 40 expert stacked router emits");

        assert_eq!(kernel.grid.block_width, Some(64));
        assert_eq!(kernel.grid.threads, 5 * 64);
        assert_eq!(kernel.extra_outputs.len(), 12 + 2);
        assert_eq!(kernel.extra_outputs[12], (NodeId(400), 30));
        assert!(kernel.source.contains("stacked_weights"));
    }

    #[test]
    fn moe_topk_wider_than_one_block_is_refused_with_a_typed_error() {
        let refused = emit_cuda(&moe_top_k_op(1056, 8, false), &no_packed());

        assert!(
            matches!(refused, Err(EmitError::WideGridUnsupported { node, .. }) if node == NodeId(100)),
            "expected WideGridUnsupported, got {refused:?}"
        );
    }

    #[test]
    fn same_structure_different_extents_yield_identical_source() {
        let small = emit_cuda(&elementwise_tanh_op(4), &no_packed()).expect("emit succeeds");
        let large = emit_cuda(&elementwise_tanh_op(4096), &no_packed()).expect("emit succeeds");
        assert_eq!(small.source, large.source);
        assert_ne!(small.grid.threads, large.grid.threads);
    }

    fn matmul_reduce_op(rows: u32, cols: u32, reduce_op: ScalarOp) -> BoundOp {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(rows), Extent::Static(cols)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: reduce_op,
                init: ReduceInit::Zero,
                operand: lhs,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: Some("row_reduce".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(
            &program,
            &shapes,
            &[NodeId((program.len() - 1) as u32)],
            proxima_tensor::NumericPolicy::default(),
        )
        .expect("bind succeeds");
        bound.into_iter().next().expect("one bound op")
    }

    /// `weight[rows, k] * activation[k]` summed over `k`, with the weight
    /// operand first or second. Returns the bound op and the weight's node.
    fn q4_0_matvec_op(rows: u32, k: u32, weight_first: bool) -> (BoundOp, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(rows), Extent::Static(k)],
                name: None,
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k)],
                name: None,
            },
        );
        let weight_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        let activation_map = IndexMap::Affine(map::projection(2, &[1]));
        let operands = if weight_first {
            vec![(weight, weight_map), (activation, activation_map)]
        } else {
            vec![(activation, activation_map), (weight, weight_map)]
        };
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands,
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: Some("q4_0_matvec".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(
            &program,
            &shapes,
            &[NodeId((program.len() - 1) as u32)],
            NumericPolicy::default(),
        )
        .expect("bind succeeds")
        .into_iter()
        .next()
        .expect("one bound op");
        (bound, weight)
    }

    fn q4_0_packed(weight: NodeId) -> PackedOperands {
        let mut packed = PackedOperands::new();
        packed.insert(weight, Codec::Q4_0);
        packed
    }

    #[test]
    fn q4_0_matvec_recogniser_matches_both_operand_orders() {
        let (cf0, cf0_weight) = q4_0_matvec_op(8, 64, false);
        let (c0f, c0f_weight) = q4_0_matvec_op(8, 64, true);

        for (bound, weight) in [(&cf0, cf0_weight), (&c0f, c0f_weight)] {
            let codecs = operand_codecs(bound, &q4_0_packed(weight));
            let (weight_slot, activation_slot) =
                q4_0_matvec(bound, &codecs).expect("a Q4_0 matvec is recognised in either order");
            assert_eq!(
                codecs[weight_slot],
                Some(Codec::Q4_0),
                "the recognised weight slot is the Q4_0-packed operand"
            );
            assert_eq!(
                codecs[activation_slot], None,
                "the recognised activation slot is the unpacked operand"
            );
        }
    }

    #[test]
    fn q4_0_matvec_recogniser_rejects_q6k_two_q4_0_and_gathered_and_non_add() {
        let (bound, weight) = q4_0_matvec_op(8, 64, true);
        let q4_0_weight_f32_activation = operand_codecs(&bound, &q4_0_packed(weight));

        let q6k_weight = [Some(Codec::Q6K), None];
        assert_eq!(q4_0_matvec(&bound, &q6k_weight), None, "Q6_K weight rejected");

        let two_q4_0 = [Some(Codec::Q4_0), Some(Codec::Q4_0)];
        assert_eq!(q4_0_matvec(&bound, &two_q4_0), None, "two Q4_0 operands rejected");

        let mut maximum = bound.clone();
        if let BoundOpKind::Reduce { reduce_op, .. } = &mut maximum.kind {
            *reduce_op = ScalarOp::Maximum;
        }
        assert_eq!(
            q4_0_matvec(&maximum, &q4_0_weight_f32_activation),
            None,
            "non-Add reduce rejected"
        );

        let mut gathered = bound.clone();
        if let BoundOpKind::Reduce { operands, .. } = &mut gathered.kind {
            let (_, activation_layout, _) = operands[1].clone();
            operands[1].2 = Some(Lookup {
                indices: weight,
                index_layout: activation_layout,
                element_stride: 1,
                extent: 1,
            });
        }
        assert_eq!(
            q4_0_matvec(&gathered, &q4_0_weight_f32_activation),
            None,
            "gathered operand rejected"
        );
    }

    #[test]
    fn q4_0_matvec_recogniser_rejects_an_activation_that_varies_across_rows() {
        let (bound, weight) = q4_0_matvec_op(8, 64, true);
        let codecs = operand_codecs(&bound, &q4_0_packed(weight));
        assert_eq!(
            q4_0_matvec(&bound, &codecs),
            Some((0, 1)),
            "control: the broadcast activation matches before the stride is touched"
        );

        let mut row_strided = bound.clone();
        if let BoundOpKind::Reduce { operands, .. } = &mut row_strided.kind {
            operands[1].1.strides[0] = 1;
        }
        assert_eq!(
            q4_0_matvec(&row_strided, &codecs),
            None,
            "an activation with a non-zero row stride is not broadcast and is rejected"
        );
    }

    fn activation_vector(k: usize) -> Vec<f32> {
        let mut lcg = Lcg(0x5eed_1536);
        (0..k).map(|_| lcg.next_unit()).collect()
    }

    #[test]
    fn q8_1_quantize_emits_one_warp_block_per_32_elements() {
        let activation = activation_vector(1536);
        let kernel = emit_q8_1_quantize(NodeId(0), NodeId(1), activation.len() as u64)
            .expect("1536 is a whole number of q8_1 blocks");

        assert_eq!(kernel.grid.threads, 48 * WARP_SIZE);
        assert_eq!(kernel.grid.block_width, Some(WARP_SIZE));
        assert_eq!(
            kernel.bindings,
            vec![
                Binding::Input(NodeId(0)),
                Binding::Output(NodeId(1)),
                Binding::Uniforms,
            ]
        );
        assert_eq!(kernel.entry, "omega_cuda_q8_1_quantize");
        assert!(kernel.source.contains("__shfl_xor_sync"));
        assert!(kernel.source.contains("cuda_fp16.h"));
        assert!(kernel.prologue.is_empty());
    }

    #[test]
    fn q8_1_quantize_rejects_a_length_that_is_not_a_block_multiple() {
        for length in [0_u64, 1537] {
            let error = emit_q8_1_quantize(NodeId(0), NodeId(1), length)
                .expect_err("a length that is not a 32-element block multiple is rejected");
            assert!(matches!(
                error,
                EmitError::CudaUnsupportedOpKind {
                    node: NodeId(0),
                    ..
                }
            ));
        }
        assert!(emit_q8_1_quantize(NodeId(0), NodeId(1), 1536).is_ok());
    }

    #[test]
    fn emit_cuda_with_policy_leaves_prologue_empty_for_the_q4_0_matvec_flag_on_and_off() {
        let (bound, weight) = q4_0_matvec_op(8, 64, true);
        let packed = q4_0_packed(weight);
        let flag_off = emit_cuda_with_policy(&bound, &packed, NumericPolicy::default())
            .expect("q4_0 matvec emits with the flag off");
        let flag_on = emit_cuda_with_policy(
            &bound,
            &packed,
            NumericPolicy::default().with_activation_quantization(true),
        )
        .expect("q4_0 matvec emits with the flag on");

        assert!(flag_off.prologue.is_empty());
        assert!(flag_on.prologue.is_empty());
        assert_eq!(
            flag_on.source.replace(&flag_on.entry, &flag_off.entry),
            flag_off.source,
        );
        assert_ne!(flag_on.entry, flag_off.entry);
    }

    #[test]
    fn entry_name_is_unchanged_with_the_flag_off_and_differs_with_it_on() {
        let (bound, weight) = q4_0_matvec_op(8, 64, true);
        let packed = q4_0_packed(weight);
        let quantized = operand_codecs(&bound, &packed);
        let flag_off = entry_name(&bound, &packed, &quantized, NumericPolicy::default());
        let flag_on = entry_name(
            &bound,
            &packed,
            &quantized,
            NumericPolicy::default().with_activation_quantization(true),
        );

        assert_eq!(
            flag_off,
            format!(
                "{}_gid64c8",
                crate::identity::kernel_identity(
                    crate::identity::KernelLanguage::Cuda,
                    &bound,
                    &packed,
                    crate::identity::MetalOnlyExtras::default(),
                    NumericPolicy::default(),
                )
            ),
            "with the flag off the entry name is the pre-existing identity"
        );
        assert_ne!(flag_off, flag_on, "the dp4a body must not share the f32 entry name");
    }

    #[test]
    fn associative_reduce_emits_warp_shuffle_cooperative_kernel() {
        let bound = matmul_reduce_op(4, 4096, ScalarOp::Add);
        let kernel = emit_cuda_with_policy(&bound, &no_packed(), NumericPolicy::llama_relaxed())
            .expect("emit succeeds");
        assert!(kernel.source.contains("__shfl_down_sync"));
        assert_eq!(kernel.grid.block_width, Some(WARP_SIZE));
        assert_eq!(kernel.grid.threads, 4 * WARP_SIZE);
    }

    #[test]
    fn non_associative_reduce_stays_serial() {
        let bound = matmul_reduce_op(4, 8, ScalarOp::Subtract);
        let kernel = emit_cuda(&bound, &no_packed()).expect("emit succeeds");
        assert!(!kernel.source.contains("__shfl_down_sync"));
        assert!(kernel.grid.block_width.is_none());
    }

    fn scan_op(extent: u32) -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[0])),
                keep: Keep::Scan,
                name: Some("prefix_sum".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(
            &program,
            &shapes,
            &[NodeId((program.len() - 1) as u32)],
            proxima_tensor::NumericPolicy::default(),
        )
        .expect("bind succeeds");
        bound.into_iter().next().expect("one bound op")
    }

    #[test]
    fn scan_emits_single_thread_serial_kernel() {
        let bound = scan_op(16);
        let kernel = emit_cuda(&bound, &no_packed()).expect("emit succeeds");
        assert!(kernel.source.contains("if (gid != 0) { return; }"));
        assert_eq!(kernel.grid.threads, 1);
    }

    /// `table[ids[s], d]` over iteration space `(s, d)` — the same worked
    /// example `crate::msl`'s own `embedding_lookup_op` test helper uses.
    fn embedding_lookup_op(vocab: u32, dim: u32, seq: u32) -> BoundOp {
        let mut program = Vec::new();
        let table = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(vocab), Extent::Static(dim)],
                name: None,
            },
        );
        let ids = append(
            &mut program,
            Op::Input {
                dtype: DType::Int32,
                shape: vec![Extent::Static(seq)],
                name: None,
            },
        );
        let gathered_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(2, &[0]),
            base: map::IndexPattern {
                iter_rank: 2,
                axes: vec![
                    map::AxisIndex::default(),
                    map::AxisIndex {
                        terms: vec![AxisTerm::projection(1)].into(),
                        offset: 0,
                        len: None,
                    },
                ],
            },
            gathered_dim: 0,
        };
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(table, gathered_map)],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("embedding lookup infers");
        bind(
            &program,
            &shapes,
            &[NodeId((program.len() - 1) as u32)],
            proxima_tensor::NumericPolicy::default(),
        )
        .expect("embedding lookup lowers")
        .into_iter()
        .next()
        .expect("one bound emitted")
    }

    #[test]
    fn gather_emits_fault_buffer_and_atomic_exchange_check() {
        let bound = embedding_lookup_op(50_000, 8, 4);
        let kernel = emit_cuda(&bound, &no_packed()).expect("emit succeeds");
        assert!(
            kernel
                .source
                .contains("atomicExch(&fault[0], (unsigned int)fetched0);")
        );
        assert_eq!(
            kernel.bindings,
            vec![
                Binding::Input(bound.operands()[0].0),
                Binding::Indices(
                    bound.operands()[0]
                        .2
                        .as_ref()
                        .expect("operand 0 gathers")
                        .indices
                ),
                Binding::Output(bound.node),
                Binding::Uniforms,
                Binding::Fault,
            ]
        );
    }

    #[test]
    fn q2k_and_q3k_emit_distinct_cuda_unpack_paths() {
        let (q2_bound, q2_packed) =
            packed_elementwise_op(Codec::Q2K, Q2K_BLOCK_BYTES, Q2K_BLOCK_ELEMENTS);
        let (q3_bound, q3_packed) =
            packed_elementwise_op(Codec::Q3K, Q3K_BLOCK_BYTES, Q3K_BLOCK_ELEMENTS);
        let q2 = emit_cuda(&q2_bound, &q2_packed).expect("q2k emits");
        let q3 = emit_cuda(&q3_bound, &q3_packed).expect("q3k emits");
        assert!(q2.source.contains("q2k_element"));
        assert!(q3.source.contains("q3k_element"));
        assert_ne!(q2.entry, q3.entry);
    }

    fn packed_elementwise_op(
        codec: Codec,
        block_bytes: usize,
        block_elements: usize,
    ) -> (BoundOp, PackedOperands) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(block_elements as u32)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(weight, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(
            &program,
            &shapes,
            &[NodeId((program.len() - 1) as u32)],
            proxima_tensor::NumericPolicy::default(),
        )
        .expect("bind succeeds");
        let bound = bound.into_iter().next().expect("one bound op");
        let mut packed = BTreeMap::new();
        packed.insert(bound.operands()[0].0, codec);
        let _ = block_bytes;
        (bound, packed)
    }

    #[test]
    fn q4k_operand_emits_q4k_element_read() {
        let (bound, packed) =
            packed_elementwise_op(Codec::Q4K, Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS);
        let kernel = emit_cuda(&bound, &packed).expect("emit succeeds");
        assert!(kernel.source.contains("q4k_element(in0"));
        assert!(kernel.source.contains(&format!("* {Q4K_BLOCK_BYTES}")));
    }

    /// `Σ_k table[ids[s], k]` over iteration space `(s, k)`. With
    /// `index_depends_on_reduction` the index also varies with `k`, so its
    /// stride on the reduction dim is nonzero and the reduce must stay serial.
    fn gathered_reduce_op(vocab: u32, width: u32, seq: u32, index_depends_on_reduction: bool) -> BoundOp {
        let mut program = Vec::new();
        let table = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(vocab), Extent::Static(width)],
                name: None,
            },
        );
        let (index_shape, index_map) = if index_depends_on_reduction {
            (
                vec![Extent::Static(seq), Extent::Static(width)],
                map::projection(2, &[0, 1]),
            )
        } else {
            (vec![Extent::Static(seq)], map::projection(2, &[0]))
        };
        let ids = append(
            &mut program,
            Op::Input {
                dtype: DType::Int32,
                shape: index_shape,
                name: None,
            },
        );
        let gathered_map = IndexMap::Computed {
            indices: ids,
            index_map,
            base: map::IndexPattern {
                iter_rank: 2,
                axes: vec![
                    map::AxisIndex::default(),
                    map::AxisIndex {
                        terms: vec![AxisTerm::projection(1)].into(),
                        offset: 0,
                        len: None,
                    },
                ],
            },
            gathered_dim: 0,
        };
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(seq), Extent::Static(width)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (table, gathered_map),
                    (activation, IndexMap::Affine(map::projection(2, &[0, 1]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: Some("gathered_row_reduce".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("gathered reduce infers");
        bind(
            &program,
            &shapes,
            &[NodeId((program.len() - 1) as u32)],
            proxima_tensor::NumericPolicy::default(),
        )
        .expect("gathered reduce lowers")
        .into_iter()
        .next()
        .expect("one bound emitted")
    }

    #[test]
    fn reduction_invariant_gathered_reduce_is_cooperative_with_hoisted_fetch() {
        let bound = gathered_reduce_op(16, 64, 4, false);
        let kernel = emit_cuda_with_policy(&bound, &no_packed(), NumericPolicy::llama_relaxed())
            .expect("emit succeeds");
        assert_eq!(kernel.grid.block_width, Some(WARP_SIZE));
        assert_eq!(kernel.grid.threads, 4 * WARP_SIZE);
        assert!(kernel.source.contains("__shfl_down_sync"));
        let fetch = kernel
            .source
            .find("long fetched0 =")
            .expect("hoisted fetch is emitted");
        let lane_loop = kernel
            .source
            .find("for (long r = (long)lane;")
            .expect("cooperative lane loop is emitted");
        assert!(fetch < lane_loop, "index fetch must precede the lane loop");
        assert!(kernel.source.contains("gather_base0"));
    }

    #[test]
    fn gathered_reduce_with_nonzero_reduction_stride_stays_serial() {
        let bound = gathered_reduce_op(16, 64, 4, true);
        let kernel = emit_cuda_with_policy(&bound, &no_packed(), NumericPolicy::llama_relaxed())
            .expect("emit succeeds");
        assert!(kernel.grid.block_width.is_none());
        assert_eq!(kernel.grid.threads, 4);
        assert!(!kernel.source.contains("__shfl_down_sync"));
    }

    #[test]
    fn reduction_invariant_gathered_reduce_under_bit_exact_policy_stays_serial() {
        let bound = gathered_reduce_op(16, 64, 4, false);
        let kernel = emit_cuda(&bound, &no_packed()).expect("emit succeeds");
        assert!(kernel.grid.block_width.is_none());
        assert_eq!(kernel.grid.threads, 4);
        assert!(!kernel.source.contains("__shfl_down_sync"));
    }

    #[test]
    fn q8_0_gathered_reduce_adds_hoisted_gather_base_to_packed_offset() {
        let bound = gathered_reduce_op(16, 64, 4, false);
        let mut packed = BTreeMap::new();
        packed.insert(bound.operands()[0].0, Codec::Q8_0);
        let kernel = emit_cuda_with_policy(&bound, &packed, NumericPolicy::llama_relaxed())
            .expect("emit succeeds");
        assert_eq!(kernel.grid.block_width, Some(WARP_SIZE));
        assert!(kernel.source.contains("off0 += gather_base0;"));
    }

    #[test]
    fn q8_0_cooperative_reduce_emits_chunk8_decode_after_gather_base() {
        let bound = gathered_reduce_op(16, 64, 4, false);
        let mut packed = BTreeMap::new();
        packed.insert(bound.operands()[0].0, Codec::Q8_0);
        let kernel = emit_cuda_with_policy(&bound, &packed, NumericPolicy::llama_relaxed())
            .expect("emit succeeds");
        let gather_add = kernel
            .source
            .find("off0 += gather_base0;")
            .expect("hoisted gather base is added to the packed offset");
        let chunk_call = kernel
            .source
            .find("q8_0_chunk8(in0")
            .expect("aligned chunk decode is emitted");
        let lane_loop = kernel
            .source
            .find("for (long chunk = (long)lane * 8;")
            .expect("chunked lane loop is emitted");
        assert!(lane_loop < gather_add && gather_add < chunk_call);
        assert!(kernel.source.contains("q8_0_element(in0"));
        assert!(kernel.source.contains("scratch[0] = q8_0_chunk0[inner];"));
    }

    #[test]
    fn q6k_cooperative_reduce_emits_chunk8_decode_after_gather_base() {
        let bound = gathered_reduce_op(16, 64, 4, false);
        let mut packed = BTreeMap::new();
        packed.insert(bound.operands()[0].0, Codec::Q6K);
        let kernel = emit_cuda_with_policy(&bound, &packed, NumericPolicy::llama_relaxed())
            .expect("emit succeeds");
        let gather_add = kernel
            .source
            .find("off0 += gather_base0;")
            .expect("hoisted gather base is added to the packed offset");
        let chunk_call = kernel
            .source
            .find("q6k_chunk8(in0")
            .expect("aligned chunk decode is emitted");
        let lane_loop = kernel
            .source
            .find("for (long chunk = (long)lane * 8;")
            .expect("chunked lane loop is emitted");
        assert!(lane_loop < gather_add && gather_add < chunk_call);
        assert!(kernel.source.contains("q6k_element(in0"));
        assert!(kernel.source.contains("scratch[0] = q6k_chunk0[inner];"));
    }

    #[test]
    fn q8_0_chunk_decode_is_absent_without_a_packed_operand() {
        let bound = gathered_reduce_op(16, 64, 4, false);
        let kernel = emit_cuda_with_policy(&bound, &no_packed(), NumericPolicy::llama_relaxed())
            .expect("emit succeeds");
        assert!(!kernel.source.contains("float q8_0_chunk0[8];"));
        assert!(!kernel.source.contains("q8_0_chunk8(in0"));
    }

    #[test]
    fn logical_packed_address_toggle_still_adds_hoisted_gather_base() {
        let bound = gathered_reduce_op(16, 64, 4, false);
        let kernel = temp_env::with_vars(
            [("PROXIMA_CUDA_LOGICAL_PACKED_ADDRESS", Some("1"))],
            || emit_cuda_with_policy(&bound, &no_packed(), NumericPolicy::llama_relaxed()),
        )
        .expect("emit succeeds");
        assert_eq!(kernel.grid.block_width, Some(WARP_SIZE));
        assert!(kernel.source.contains("gather_base0"));
        assert!(kernel.source.contains("off0 += gather_base0;"));
    }

    #[test]
    fn q5k_operand_emits_q5k_element_read() {
        let (bound, packed) =
            packed_elementwise_op(Codec::Q5K, Q5K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS);
        let kernel = emit_cuda(&bound, &packed).expect("emit succeeds");
        assert!(kernel.source.contains("q5k_element(in0"));
    }

    #[test]
    fn q6k_operand_emits_q6k_element_read() {
        let (bound, packed) =
            packed_elementwise_op(Codec::Q6K, Q6K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS);
        let kernel = emit_cuda(&bound, &packed).expect("emit succeeds");
        assert!(kernel.source.contains("q6k_element(in0"));
    }

    #[test]
    fn q8_0_operand_emits_q8_0_element_read() {
        let (bound, packed) =
            packed_elementwise_op(Codec::Q8_0, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS);
        let kernel = emit_cuda(&bound, &packed).expect("emit succeeds");
        assert!(kernel.source.contains("q8_0_element(in0"));
    }

    #[test]
    fn q4_0_operand_emits_q4_0_element_read() {
        let (bound, packed) =
            packed_elementwise_op(Codec::Q4_0, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS);
        let kernel = emit_cuda(&bound, &packed).expect("emit succeeds");
        assert!(kernel.source.contains("q4_0_element(in0"));
    }

    #[test]
    fn bfloat16_operand_emits_bf16_element_read() {
        let (bound, packed) = packed_elementwise_op(
            Codec::BFloat16,
            BFLOAT16_BLOCK_BYTES,
            BFLOAT16_BLOCK_ELEMENTS,
        );
        let kernel = emit_cuda(&bound, &packed).expect("emit succeeds");
        assert!(kernel.source.contains("bf16_element(in0"));
    }

    #[test]
    fn preamble_always_carries_every_codec_unpack_function() {
        let bound = elementwise_tanh_op(4);
        let kernel = emit_cuda(&bound, &no_packed()).expect("emit succeeds");
        assert!(kernel.source.contains("q4k_element"));
        assert!(kernel.source.contains("q5k_element"));
        assert!(kernel.source.contains("q6k_element"));
        assert!(kernel.source.contains("q8_0_element"));
        assert!(kernel.source.contains("q4_0_element"));
        assert!(kernel.source.contains("bf16_element"));
    }

    #[test]
    fn a_float16_node_pulls_in_the_half_header() {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float16,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float16,
                body: ScalarOp::Tanh,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(
            &program,
            &shapes,
            &[NodeId((program.len() - 1) as u32)],
            proxima_tensor::NumericPolicy::default(),
        )
        .expect("bind succeeds");
        let bound = bound.into_iter().next().expect("one bound op");
        let kernel = emit_cuda(&bound, &no_packed()).expect("emit succeeds");
        assert!(kernel.source.contains("#include <cuda_fp16.h>"));
        assert!(kernel.source.contains("__half"));
    }

    #[test]
    fn an_unsupported_dtype_is_rejected() {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float64,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float64,
                body: ScalarOp::Tanh,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(
            &program,
            &shapes,
            &[NodeId((program.len() - 1) as u32)],
            proxima_tensor::NumericPolicy::default(),
        )
        .expect("bind succeeds");
        let bound = bound.into_iter().next().expect("one bound op");
        let error = emit_cuda(&bound, &no_packed()).expect_err("f64 is rejected");
        assert!(matches!(error, EmitError::UnsupportedDType { .. }));
    }

    #[test]
    fn render_reduce_rejects_an_elementwise_bound_op() {
        let bound = elementwise_tanh_op(8);
        let error = render_reduce(&bound, "entry", &[None], NumericPolicy::bit_exact())
            .expect_err("an elementwise chain is not a Reduce fold");
        assert!(matches!(
            error,
            EmitError::RenderKindMismatch {
                expected: "keep::reduce fold",
                found: "elementwise",
                ..
            }
        ));
    }

    /// `bind::BoundOpKind::Reduce::epilogue_broadcast_axes`'s own doc: a
    /// non-empty value is the RMSNorm-shaped "broadcast-reduce" epilogue
    /// (`x * inv_rms`, re-broadcasting the fold's scalar back over the
    /// reduced axis). CUDA now widens the epilogue/output ABI to full rank
    /// and emits the per-lane broadcast write loop.
    #[test]
    fn render_reduce_emits_a_broadcast_reduce_epilogue() {
        let mut bound = matmul_reduce_op(4, 8, ScalarOp::Add);
        let BoundOpKind::Reduce {
            epilogue_broadcast_axes,
            ..
        } = &mut bound.kind
        else {
            panic!("matmul_reduce_op always builds a Keep::Reduce fold")
        };
        epilogue_broadcast_axes.push(1);

        let source = render_reduce(&bound, "entry", &[None], NumericPolicy::llama_relaxed())
            .expect("broadcast emits");
        assert!(source.contains("broadcast_out_strides[2]"));
        assert!(source.contains("broadcast_full_coord[1]"));
        assert!(source.contains("__shfl_sync(0xffffffffu, accumulator, 0)"));

        let kernel = emit_cuda_with_policy(&bound, &no_packed(), NumericPolicy::llama_relaxed())
            .expect("broadcast kernel emits");
        assert_eq!(kernel.grid.threads, 4 * 8 * WARP_SIZE);
        let uniforms = pack_cuda_uniforms(&bound).expect("broadcast uniforms pack");
        assert_eq!(
            i64::from_ne_bytes(uniforms[..8].try_into().expect("i64 output total")),
            32
        );
    }

    #[test]
    fn render_scan_rejects_an_elementwise_bound_op() {
        let bound = elementwise_tanh_op(8);
        let error = render_scan(&bound, "entry", &[None])
            .expect_err("an elementwise chain is not a Reduce fold");
        assert!(matches!(
            error,
            EmitError::RenderKindMismatch {
                expected: "keep::scan fold",
                found: "elementwise",
                ..
            }
        ));
    }

    #[test]
    fn shuffle_combine_expr_rejects_a_non_cooperative_reduce_op() {
        let bound = matmul_reduce_op(4, 8, ScalarOp::Subtract);
        let error = shuffle_combine_expr(bound.node, ScalarOp::Subtract, "accumulator", "shuffled")
            .expect_err("subtract is not associative-commutative");
        assert!(matches!(
            error,
            EmitError::NonCooperativeReduceOp { op: "subtract", .. }
        ));
    }

    /// `row_sum(lhs)[rows] + bias[rows]` -- the smallest program
    /// `reduce_epilogue_candidates` (`proxima-tensor/src/bind.rs`) fuses: a
    /// sole `Add` elementwise consumer, over the reduce's own output shape,
    /// with an identity map on both operands. Structural (`ROW 294`) proof
    /// that CUDA renders the fused write text ROW 294 found this crate
    /// rejecting outright -- no CUDA toolchain on this host means this
    /// asserts source STRUCTURE only (the module doc's own "No CUDA
    /// toolchain" note), never numeric parity.
    #[cfg(feature = "reduce-epilogue-fusion")]
    fn row_sum_plus_bias_op(rows: u32, cols: u32) -> BoundOp {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(rows), Extent::Static(cols)],
                name: None,
            },
        );
        let reduce = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: lhs,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: Some("row_sum".into()),
            }),
        );
        let bias = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(rows)],
                name: None,
            },
        );
        let consumer = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![
                    (reduce, IndexMap::Affine(map::projection(1, &[0]))),
                    (bias, IndexMap::Affine(map::projection(1, &[0]))),
                ],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(
            &program,
            &shapes,
            &[consumer],
            proxima_tensor::NumericPolicy::default(),
        )
        .expect("bind succeeds");
        bound
            .into_iter()
            .next()
            .expect("fusion collapses to one bound op")
    }

    #[test]
    #[cfg(feature = "reduce-epilogue-fusion")]
    fn a_fused_reduce_plus_bias_epilogue_renders_the_epilogue_write_in_cuda_c() {
        let bound = row_sum_plus_bias_op(4, 8);
        let BoundOpKind::Reduce {
            epilogue_operands, ..
        } = &bound.kind
        else {
            panic!("row_sum_plus_bias_op must bind to a Reduce");
        };
        assert_eq!(
            epilogue_operands.len(),
            1,
            "the bias consumer must fuse into the reduce's own epilogue"
        );

        let kernel = emit_cuda(&bound, &no_packed()).expect("emit succeeds");
        assert!(
            kernel.source.contains("epi_scratch"),
            "fused epilogue must declare its own scratch array: {}",
            kernel.source
        );
        assert!(
            kernel.source.contains("const float* __restrict__ epi0"),
            "the bias operand must bind as its own epi0 buffer: {}",
            kernel.source
        );
        assert!(
            kernel.source.contains("u.epilogue_operand_base[0]"),
            "the epilogue operand's offset must read the uniforms epilogue table: {}",
            kernel.source
        );
        assert_eq!(
            kernel.bindings.len(),
            4,
            "lhs, bias, output, uniforms -- one binding per `all_read_sources` \
             entry plus output and uniforms: {:?}",
            kernel.bindings
        );
    }

    #[test]
    #[cfg(feature = "reduce-epilogue-fusion")]
    fn a_fused_reduce_epilogue_gives_a_distinct_entry_name_from_the_bare_reduce() {
        let fused = row_sum_plus_bias_op(4, 8);
        let bare = matmul_reduce_op(4, 8, ScalarOp::Add);
        let packed = no_packed();
        let policy = NumericPolicy::default();
        assert_ne!(
            entry_name(&fused, &packed, &[], policy),
            entry_name(&bare, &packed, &[], policy),
            "a fused epilogue must never collide with the bare reduce's cache key \
             (crate::wgpu_driver::pipeline_for's own caching-by-name hazard)"
        );
    }

    /// This census's own finding: `packed_element_expr` (this module's own
    /// codec-dependent body renderer) already emitted a distinct body per
    /// codec, but `entry_name` never said so before `crate::identity::
    /// kernel_identity` folded the codec digits in for every language. A
    /// future CUDA driver keying a kernel cache on this name alone would
    /// have reused a `Q4_K`-compiled kernel's body against `Q5_K` bytes.
    #[test]
    fn packed_codec_changes_emitted_source_and_the_entry_name() {
        let bound = matmul_reduce_op(4, 256, ScalarOp::Add);
        let weight_node = bound.operands()[0].0;

        let mut q4k = PackedOperands::new();
        q4k.insert(weight_node, Codec::Q4K);
        let mut q5k = PackedOperands::new();
        q5k.insert(weight_node, Codec::Q5K);

        let kernel_q4k = emit_cuda(&bound, &q4k).expect("q4k emits");
        let kernel_q5k = emit_cuda(&bound, &q5k).expect("q5k emits");

        assert!(kernel_q4k.source.contains("packed_row0"));
        assert!(
            kernel_q4k
                .source
                .contains("u.operand_base[0] + packed_row0 * u.reduction_total + r")
        );

        assert_ne!(
            kernel_q4k.source, kernel_q5k.source,
            "packed_element_expr must render a distinct body per codec"
        );
        assert_ne!(
            kernel_q4k.entry, kernel_q5k.entry,
            "two operands differing only in packed codec must never share one compiled kernel name"
        );
    }

    #[test]
    fn a_cooperative_grid_past_a_32_bit_thread_index_is_emitted_with_a_64_bit_gid() {
        let bound = matmul_reduce_op(200_000_000, 4096, ScalarOp::Add);

        let kernel = emit_cuda_with_policy(&bound, &no_packed(), NumericPolicy::llama_relaxed())
            .expect("200M cooperative rows emit");

        assert_eq!(kernel.grid.threads, 200_000_000 * WARP_SIZE);
        assert!(kernel.grid.threads > u64::from(u32::MAX));
        assert_eq!(kernel.grid.blocks(), 200_000_000);
        assert!(
            kernel
                .source
                .contains("long gid = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;"),
            "{}",
            kernel.source
        );
    }

    #[test]
    fn a_grid_with_more_blocks_than_the_x_dimension_holds_is_rejected_instead_of_truncated() {
        let bound = matmul_reduce_op(u32::MAX, 4096, ScalarOp::Add);

        let rejected = emit_cuda_with_policy(&bound, &no_packed(), NumericPolicy::llama_relaxed());

        assert!(
            matches!(
                rejected,
                Err(EmitError::GridExceedsThreadIndex { threads, limit, .. })
                    if threads == u64::from(u32::MAX) * WARP_SIZE
                        && limit == u64::from(i32::MAX as u32) * WARP_SIZE
            ),
            "4294967295 warp-wide blocks exceed gridDim.x = 2^31 - 1: {rejected:?}"
        );
    }

    #[test]
    fn a_default_width_grid_counts_its_blocks_at_the_default_block_width() {
        let grid = CudaGridSpec {
            threads: 1_000,
            block_width: None,
        };

        assert_eq!(grid.block_width_or_default(), DEFAULT_BLOCK_WIDTH);
        assert_eq!(grid.blocks(), 4);
    }

    #[test]
    fn the_entry_name_carries_the_index_form_so_a_stale_module_cannot_be_served() {
        let kernel = emit_cuda(&elementwise_tanh_op(8), &no_packed()).expect("emits");

        assert!(kernel.entry.ends_with("_gid64c8"), "{}", kernel.entry);
        assert!(
            kernel.source.contains(&kernel.entry),
            "the name a driver keys its compiled module on is the name the source declares"
        );
        assert!(
            kernel
                .source
                .contains("(long)blockIdx.x * (long)blockDim.x")
        );
    }
}
