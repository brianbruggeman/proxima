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
//! - **Packed operands**: `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`/`Q4_0`/`Q5_0` plus
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

use proxima_primitives::block_layout::BlockLayout;
use proxima_tensor::{
    BoundOp, BoundOpKind, ComposedBody, DType, Keep, Layout, Lookup, NodeId, NumericPolicy,
    NumericRewrite, ReduceInit, ScalarOp, StepArg, admit,
};

use crate::cuda_cached_attention::{
    cached_attention_entry_name, cached_attention_runtime_bounds, render_cached_attention_cuda,
};
use crate::cuda_cached_attention_tiled::{ATTENTION_TILE_MIN_ROWS, cached_attention_tile_rows};
use crate::epilogue::{
    broadcast_reduce_takes_block, epilogue_operand_is_loop_invariant, index32_reach_fits,
    reduce_block_width,
};
use crate::error::EmitError;
use crate::grid::checked_product;
use crate::identity::{op_token, operand_codecs, reduce_epilogue_is_identity};
use crate::lowering::{
    Binding, Codec, PackedOperands, dispatch_bindings, gather_count, gather_is_reduction_invariant,
    gather_slots, is_cooperative_reduce_op, reduction_dims,
};

mod block_dot;
mod block_quantize;
mod int8_gemm;
mod plane_dot;
mod radix_dot;
mod sub_block_dot;
pub(crate) mod mma_gemm;
mod row_tile;
mod split_attention;
mod staged_matvec;
use split_attention::{
    render_combine, split_chunks, split_pass, split_scratch_len, split_scratch_node,
};
pub use int8_gemm::is_int8_weight;
use block_quantize::{block_quantize_grid, block_quantize_groups, quantize_device, render_block_quantize};
use int8_gemm::INT8_MMA_INSTRUCTION;
use mma_gemm::{
    GEMM_THREADS, MMA_F16_HELPER, MMA_GEMM_HELPERS, MMA_INSTRUCTION, mma_gemm_plan, mma_gemm_threads, push_mma_gemm_body,
    reads_prequantized_activation,
};
use row_tile::{ROW_TILE, push_row_tiled_cooperative_body, row_tile_axis};
use staged_matvec::{
    STAGED_THREADS, push_staged_matvec_body, staged_matvec_plan, staged_matvec_threads, staged_prepass,
    staged_prepass_grid,
    staged_scratch_len, staged_scratch_node,
};

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
///         dtype: DType::FLOAT32,
///         shape: vec![Extent::Static(4)],
///         name: None,
///     },
/// );
/// let activated = append(
///     &mut program,
///     Op::Elementwise {
///         dtype: DType::FLOAT32,
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

/// Device facts that shape emitted launch grids; the default is an unknown device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmitDevice {
    pub multiprocessors: u32,
}

pub fn emit_cuda_with_policy(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
) -> Result<CudaKernel, EmitError> {
    emit_cuda_on_device(resolved, packed_operands, numeric_policy, EmitDevice::default())
}

/// Emits for a known device: launch grids are shaped to its multiprocessor count.
pub fn emit_cuda_on_device(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> Result<CudaKernel, EmitError> {
    validate(resolved)?;
    let quantized = operand_codecs(resolved, packed_operands);
    reject_unserved_prequantized(resolved, &quantized, numeric_policy, device)?;
    let attention_chunks = split_chunks(&resolved.kind, device);
    let entry = match cuda_cached_attention_class(&resolved.kind) {
        Some(class) => cached_attention_entry_name(resolved, class, attention_chunks, numeric_policy)?,
        None => entry_name(resolved, packed_operands, &quantized, numeric_policy, device),
    };
    let source = match &resolved.kind {
        BoundOpKind::Elementwise { .. } => render_elementwise(resolved, &entry, &quantized)?,
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => render_reduce(resolved, &entry, &quantized, numeric_policy, device)?,
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => render_scan(resolved, &entry, &quantized)?,
        BoundOpKind::Iota => render_leaf(resolved, &entry, None)?,
        BoundOpKind::Constant { value } => render_leaf(resolved, &entry, Some(*value))?,
        BoundOpKind::CachedAttention { .. } if quantized.iter().any(Option::is_some) => {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: "cached_attention_packed_operand",
            });
        }
        BoundOpKind::CachedAttention { .. } => match attention_chunks {
            Some(chunks) => render_combine(resolved, &entry, chunks)?,
            None => render_cached_attention_cuda(resolved, &entry, numeric_policy)?,
        },
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
        BoundOpKind::BlockQuantize { .. } => render_block_quantize(resolved, &entry)?,
        BoundOpKind::ElementwiseTwin { .. } => {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: "elementwise_twin",
            });
        }
    };
    let (grid, extra_outputs) = extent_fields(resolved, &quantized, numeric_policy, device)?;
    let prologue = prologue_kernels(resolved, &quantized, numeric_policy, device, &entry, &source)?;
    Ok(CudaKernel {
        source,
        entry,
        bindings: dispatch_bindings(resolved),
        grid,
        extra_outputs,
        prologue,
    })
}

/// Whether the int8 gemm would serve `resolved` if its operand in `activation_slot` held
/// activation blocks of the `activation` codec.
pub(crate) fn int8_gemm_serves(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    activation_slot: usize,
    activation: Codec,
    numeric_policy: NumericPolicy,
) -> bool {
    let mut quantized = operand_codecs(resolved, packed_operands);
    let Some(slot) = quantized.get_mut(activation_slot) else {
        return false;
    };
    *slot = Some(activation);
    mma_gemm_plan(resolved, &quantized, numeric_policy).is_some_and(|plan| plan.int8.is_some())
}

/// Whether the staged matvec would serve `resolved` reading its operand in `activation_slot` as
/// activation blocks of the `activation` codec.
pub(crate) fn staged_matvec_serves(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    activation_slot: usize,
    activation: Codec,
    numeric_policy: NumericPolicy,
) -> bool {
    let mut quantized = operand_codecs(resolved, packed_operands);
    let Some(slot) = quantized.get_mut(activation_slot) else {
        return false;
    };
    *slot = Some(activation);
    staged_matvec_plan(resolved, &quantized, numeric_policy, EmitDevice::default()).is_some_and(|plan| plan.external.is_some())
}

/// A pre-quantized activation is readable only by the int8 gemm and the external-activation staged
/// matvec; any other consumer would read its bytes as floats, so it fails here instead of emitting a wrong kernel.
fn reject_unserved_prequantized(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> Result<(), EmitError> {
    let staged = staged_matvec_plan(resolved, quantized, numeric_policy, device).is_some_and(|plan| plan.external.is_some());
    let served = staged || mma_gemm_plan(resolved, quantized, numeric_policy).is_some_and(|plan| plan.int8.is_some());
    if reads_prequantized_activation(quantized) && !served {
        return Err(EmitError::CudaUnsupportedOpKind {
            node: resolved.node,
            kind: "block_quantize_consumer",
        });
    }
    Ok(())
}

/// The kernels launched ahead of the op: they are sized by its extents like the grid, so a
/// reused kernel rebuilds them rather than keeping the previous extents' launch.
fn prologue_kernels(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
    device: EmitDevice,
    entry: &str,
    source: &str,
) -> Result<Vec<CudaKernel>, EmitError> {
    match (
        staged_matvec_plan(resolved, quantized, numeric_policy, device),
        split_chunks(&resolved.kind, device),
    ) {
        (Some(plan), _) if plan.scratch => Ok(vec![staged_prepass(resolved, &plan, entry, source)?]),
        (_, Some(chunks)) => Ok(vec![split_pass(resolved, entry, chunks)?]),
        _ => Ok(Vec::new()),
    }
}

/// The `CudaKernel` fields that follow the op's extents rather than its text:
/// the launch grid and the extra output buffers. Shared by a fresh emit and by
/// a reused kernel, so a reused kernel carries exactly what a fresh one would.
fn extent_fields(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> Result<(CudaGridSpec, Vec<(NodeId, usize)>), EmitError> {
    if matches!(resolved.kind, BoundOpKind::BlockQuantize { .. }) {
        return Ok((block_quantize_grid(resolved)?, Vec::new()));
    }
    let cooperative = reduce_is_cooperative(resolved, quantized, numeric_policy);
    let row_tile = row_tile_axis(resolved, quantized, numeric_policy);
    let gemm = mma_gemm_plan(resolved, quantized, numeric_policy);
    let reduce_width = cooperative_width(resolved, quantized);
    let staged = staged_matvec_plan(resolved, quantized, numeric_policy, device);
    let grid = match gemm {
        _ if staged.is_some() => CudaGridSpec {
            threads: staged.as_ref().map_or(0, |plan| staged_matvec_threads(resolved, plan)),
            block_width: Some(STAGED_THREADS),
        },
        Some(plan) => CudaGridSpec {
            threads: mma_gemm_threads(resolved, &plan),
            block_width: Some(GEMM_THREADS),
        },
        None => CudaGridSpec {
            threads: grid_threads(
                resolved,
                numeric_policy,
                cooperative,
                cooperative && lane_major_reduce(resolved, quantized),
                row_tile,
                reduce_width,
            )?,
            block_width: match &resolved.kind {
                BoundOpKind::MoeTopK { expert_count, .. } => Some(moe_block_width(*expert_count)),
                BoundOpKind::CachedAttention { .. } => Some(ATTENTION_BLOCK_WIDTH),
                _ => cooperative.then_some(reduce_width),
            },
        },
    };
    if grid.blocks() > MAX_GRID_BLOCKS {
        return Err(EmitError::GridExceedsThreadIndex {
            node: resolved.node,
            threads: grid.threads,
            limit: MAX_GRID_BLOCKS * grid.block_width_or_default(),
        });
    }
    let mut extra_outputs = resolved
        .kind
        .moe_topk_extra_outputs(resolved.extents.iter().product::<u64>() as usize);
    if let Some(plan) = staged.as_ref().filter(|plan| plan.scratch) {
        extra_outputs.push((staged_scratch_node(resolved.node), staged_scratch_len(plan.reduction)));
    }
    if let Some(chunks) = split_chunks(&resolved.kind, device) {
        extra_outputs.push((
            split_scratch_node(resolved.node),
            split_scratch_len(&resolved.kind, chunks),
        ));
    }
    Ok((grid, extra_outputs))
}

/// one warp lane per element, so a block is exactly one warp wide.
const Q8_1_BLOCK_ELEMENTS: u64 = 32;

const Q8_1_QUANTIZE_ENTRY: &str = "omega_cuda_q8_1_quantize";


const Q8_1_QUANTIZE_ENTRY_TEXT: &str = r#"
extern "C" __global__ void omega_cuda_q8_1_quantize(const float *in, float *out, const int *u) {
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long block = gid / 32;
    if (block >= u[0]) return;
    omega_q8_1_quantize_block(in, out, block, (unsigned int)(gid & 31));
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
        source: ["#include <cuda_fp16.h>\n", &quantize_device(&BlockLayout::Q8_1), Q8_1_QUANTIZE_ENTRY_TEXT].concat(),
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

/// Emits a whole bound program, moving a previous kernel across when the op
/// for the same node has the same text-reaching fields (`dtype` and `kind`).
/// A reused kernel keeps its source, entry and bindings and takes its grid and
/// extra outputs from the new op, so it equals a fresh emit field for field.
/// Returns the kernels in `resolved` order and how many were reused. On a
/// failed emit the error carries the failing op's node, as the cold path does.
///
/// Teaching pointer: [`emit_cuda_with_policy`] takes its text from
/// `(dtype, kind, packed_operands, numeric_policy)`; extents reach the kernel
/// only through the uniforms and the launch grid, which are recomputed here.
/// The caller owns the "same packed operands and policy" half of that premise.
pub fn emit_cuda_reusing(
    previous_bound: &[BoundOp],
    previous_kernels: Vec<CudaKernel>,
    resolved: &[BoundOp],
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    device: EmitDevice,
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
            .filter(|(old, _)| same_emitted_text(old, bound, packed_operands, numeric_policy, device))
            .and_then(|(_, kernel)| kernel.take());
        kernels.push(match unchanged {
            Some(kernel) => {
                reused += 1;
                refit_extent_fields(kernel, bound, packed_operands, numeric_policy, device)
                    .map_err(|error| (bound.node, error))?
            }
            None => emit_cuda_on_device(bound, packed_operands, numeric_policy, device)
                .map_err(|error| (bound.node, error))?,
        });
    }
    Ok((kernels, reused))
}

/// Moves one kernel from `old` to `new` in place: the text is kept when the two
/// ops emit the same CUDA C, and the grid always follows `new`'s extents.
/// Returns whether the text was kept.
///
/// # Errors
/// The emit error of `new` when the text has to be regenerated.
pub fn refit_kernel(
    old: &BoundOp,
    new: &BoundOp,
    kernel: &mut CudaKernel,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> Result<bool, EmitError> {
    if !same_emitted_text(old, new, packed_operands, numeric_policy, device) {
        *kernel = emit_cuda_on_device(new, packed_operands, numeric_policy, device)?;
        return Ok(false);
    }
    refit_extent_fields_in_place(kernel, new, packed_operands, numeric_policy, device)?;
    Ok(true)
}

/// [`refit_kernel`] for an op already known to emit the same text: only the launch grid,
/// extra outputs and prologue follow `new`'s extents.
///
/// # Errors
/// The emit error of `new` when its grid cannot be derived.
pub fn refit_grid(
    kernel: &mut CudaKernel,
    new: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> Result<(), EmitError> {
    let quantized = operand_codecs(new, packed_operands);
    let staged = staged_matvec_plan(new, &quantized, numeric_policy, device).filter(|plan| plan.scratch);
    match (staged, kernel.prologue.as_mut_slice()) {
        (Some(plan), [prepass]) => {
            let (grid, extra_outputs) = extent_fields(new, &quantized, numeric_policy, device)?;
            prepass.grid = staged_prepass_grid(&plan);
            kernel.grid = grid;
            kernel.extra_outputs = extra_outputs;
            Ok(())
        }
        _ => refit_extent_fields_in_place(kernel, new, packed_operands, numeric_policy, device),
    }
}

/// Whether two ops select the same kernel text, ignoring the extents that only reach the
/// text through packed-operand addressing: [`same_emitted_text`] without that clause.
pub fn same_text_modulo_extents(
    old: &BoundOp,
    new: &BoundOp,
    packed: &PackedOperands,
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> bool {
    old.dtype == new.dtype
        && project_text_kind(&old.kind) == project_text_kind(&new.kind)
        && same_text_plan(old, new, packed, numeric_policy, device)
}

/// Whether two ops emit the same CUDA C text. Layout stride and base values go
/// to uniforms, so only each stride's zero pattern reaches the text. Extents
/// reach the text only through packed-operand addressing, so an op that reads
/// a packed operand also has to match on extents.
fn same_emitted_text(
    old: &BoundOp,
    new: &BoundOp,
    packed: &PackedOperands,
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> bool {
    old.dtype == new.dtype
        && project_text_kind(&old.kind) == project_text_kind(&new.kind)
        && (old.extents == new.extents || !(reads_packed(old, packed) || reads_packed(new, packed)))
        && same_text_plan(old, new, packed, numeric_policy, device)
}

fn same_text_plan(
    old: &BoundOp,
    new: &BoundOp,
    packed: &PackedOperands,
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> bool {
    let old_codecs = operand_codecs(old, packed);
    let new_codecs = operand_codecs(new, packed);
    text_plan_token(old, &old_codecs, numeric_policy, device)
        == text_plan_token(new, &new_codecs, numeric_policy, device)
}

fn reads_packed(resolved: &BoundOp, packed: &PackedOperands) -> bool {
    resolved
        .all_read_sources()
        .any(|(node, _, _)| packed.contains_key(node))
}

/// The kind with every stride and base value replaced by its text-relevant
/// shape, so two kinds that differ only in those values compare equal.
fn project_text_kind(kind: &BoundOpKind) -> BoundOpKind {
    let mut projected = kind.clone();
    match &mut projected {
        BoundOpKind::CachedAttention {
            operands,
            query_rows,
            cached_key_rows,
            new_key_rows,
            new_upper_inclusive,
            cached_lower_inclusive,
            ..
        } => {
            project_operands(operands);
            if cuda_cached_attention_class(kind) == Some(CudaAttentionClass::CachedBound) {
                *cached_key_rows = 1;
            }
            if cached_attention_runtime_bounds(kind) {
                *query_rows = (*query_rows).min(ATTENTION_TILE_MIN_ROWS);
                *new_key_rows = 0;
                *new_upper_inclusive = 0;
                *cached_lower_inclusive = 0;
            }
        }
        BoundOpKind::GatedDeltaNet { operands, .. }
        | BoundOpKind::MoeTopK { operands, .. }
        | BoundOpKind::TopFractionSelect { operands, .. }
        | BoundOpKind::Elementwise { operands, .. }
        | BoundOpKind::ElementwiseTwin { operands, .. }
        | BoundOpKind::CachedSoftmaxWeights { operands, .. }
        | BoundOpKind::BlockQuantize { operands, .. } => project_operands(operands),
        BoundOpKind::Reduce {
            operands,
            out_layout,
            out_scatter,
            epilogue_operands,
            ..
        }
        | BoundOpKind::RoundBatchedReduce {
            operands,
            out_layout,
            out_scatter,
            epilogue_operands,
            ..
        } => {
            project_operands(operands);
            project_operands(epilogue_operands);
            *out_layout = text_layout(out_layout);
            if let Some(lookup) = out_scatter {
                *lookup = text_lookup(lookup);
            }
        }
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => {}
    }
    projected
}

fn project_operands(operands: &mut [(NodeId, Layout, Option<Lookup>)]) {
    for (_, layout, gather) in operands {
        *layout = text_layout(layout);
        if let Some(lookup) = gather {
            *lookup = text_lookup(lookup);
        }
    }
}

fn text_layout(layout: &Layout) -> Layout {
    Layout {
        base: 0,
        strides: layout
            .strides
            .iter()
            .map(|stride| i64::from(*stride != 0))
            .collect(),
    }
}

fn text_lookup(lookup: &Lookup) -> Lookup {
    Lookup {
        indices: lookup.indices,
        index_layout: text_layout(&lookup.index_layout),
        element_stride: 0,
        extent: 0,
    }
}

fn refit_extent_fields(
    mut kernel: CudaKernel,
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> Result<CudaKernel, EmitError> {
    refit_extent_fields_in_place(&mut kernel, bound, packed_operands, numeric_policy, device)?;
    Ok(kernel)
}

fn refit_extent_fields_in_place(
    kernel: &mut CudaKernel,
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> Result<(), EmitError> {
    let quantized = operand_codecs(bound, packed_operands);
    let (grid, extra_outputs) = extent_fields(bound, &quantized, numeric_policy, device)?;
    let prologue = prologue_kernels(bound, &quantized, numeric_policy, device, &kernel.entry, &kernel.source)?;
    kernel.grid = grid;
    kernel.extra_outputs = extra_outputs;
    kernel.prologue = prologue;
    Ok(())
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
        BoundOpKind::BlockQuantize { .. } => {
            push_cuda_i64(&mut bytes, block_quantize_groups(resolved)? as i64);
        }
        BoundOpKind::CachedAttention {
            query_rows,
            kv_heads,
            query_groups,
            new_key_rows,
            new_upper_inclusive,
            cached_lower_inclusive,
            ..
        } => {
            push_cuda_i64(&mut bytes, (query_rows * kv_heads * query_groups) as i64);
            if cached_attention_runtime_bounds(&resolved.kind) {
                push_cuda_i64(&mut bytes, *new_key_rows as i64);
                push_cuda_i64(&mut bytes, *new_upper_inclusive);
                push_cuda_i64(&mut bytes, *cached_lower_inclusive);
            }
        }
        BoundOpKind::GatedDeltaNet { .. }
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

pub(crate) fn cuda_float_literal(value: f32) -> String {
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

/// `__half` and `__nv_bfloat16` where the layout's fields equal binary16's and bfloat16's, `float`
/// for every other layout (the kernel computes in binary32 and the host narrows through the
/// layout codec at the buffer boundary, as on Metal).
fn type_token(node: NodeId, dtype: DType) -> Result<&'static str, EmitError> {
    match dtype {
        DType::Layout(layout) if layout.width() <= 32 => {
            Ok(match crate::msl::layout_render::hardware_type(&layout) {
                Some("half") => "__half",
                Some("bfloat") => "__nv_bfloat16",
                _ => "float",
            })
        }
        DType::Bool => Ok("float"),
        DType::Layout(_) => Err(EmitError::UnsupportedDType { node, dtype }),
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

/// Threads that cooperate on one output row of a cooperative reduce. A
/// broadcast-epilogue reduce over a single row (the decode-shaped norm) has one
/// warp's worth of latency to hide, so it takes a whole block; every other
/// cooperative reduce keeps one warp per output.
const MAX_BLOCK_REDUCE_WIDTH: u64 = 1024;

/// Elements each thread folds serially before the block merge; the per-element fold is
/// latency-bound, so the block widens until each thread owns this few.
const BLOCK_REDUCE_ELEMENTS_PER_THREAD: u64 = 4;

/// Warps that share one attention vector (one query row of one head): each walks a
/// strided subset of the keys and the block merges their softmax states.
pub(crate) const ATTENTION_WARPS: u64 = 4;

pub(crate) const ATTENTION_BLOCK_WIDTH: u64 = ATTENTION_WARPS * WARP_SIZE;

const LANE_MAJOR_WARPS: u64 = 4;

const LANE_MAJOR_BLOCK_WIDTH: u64 = LANE_MAJOR_WARPS * WARP_SIZE;

/// A broadcast-epilogue reduce whose reduction axis is strided for every operand while the
/// last output axis is the contiguous one: lanes then own consecutive output elements, so
/// every operand load and every epilogue store of a warp lands in one cache line, and the
/// block's warps split the reduction and merge through shared memory.
fn lane_major_reduce(resolved: &BoundOp, quantized: &[Option<Codec>]) -> bool {
    let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        output_axes,
        epilogue_body,
        epilogue_operands,
        epilogue_broadcast_axes,
        ..
    } = &resolved.kind
    else {
        return false;
    };
    let reduce_dims = reduction_dims(resolved, output_axes);
    let (Some(&last_output), [reduce_dim]) = (output_axes.last(), reduce_dims.as_slice()) else {
        return false;
    };
    let operands = resolved.operands();
    let kept_rows: u64 = output_axes
        .iter()
        .map(|dim| resolved.extents[usize::from(*dim)])
        .product();
    !epilogue_broadcast_axes.is_empty()
        && kept_rows > 1
        && quantized.iter().all(Option::is_none)
        && gather_count(resolved) == 0
        && !reduce_epilogue_is_identity(epilogue_body, epilogue_operands)
        && index32_reach_fits(resolved.extents[usize::from(*reduce_dim)])
        && operands.iter().all(|(_, layout, _)| layout.stride(*reduce_dim) != 1)
        && operands.iter().any(|(_, layout, _)| layout.stride(last_output) == 1)
}

fn cooperative_width(resolved: &BoundOp, quantized: &[Option<Codec>]) -> u64 {
    let BoundOpKind::Reduce {
        output_axes,
        keep: Keep::Reduce,
        ..
    } = &resolved.kind
    else {
        return WARP_SIZE;
    };
    if lane_major_reduce(resolved, quantized) {
        return LANE_MAJOR_BLOCK_WIDTH;
    }
    if broadcast_reduce_takes_block(resolved, quantized) {
        let reduction_total: u64 = reduction_dims(resolved, output_axes)
            .iter()
            .map(|dim| resolved.extents[*dim as usize])
            .product();
        reduce_block_width(
            reduction_total,
            BLOCK_REDUCE_ELEMENTS_PER_THREAD,
            WARP_SIZE,
            MAX_BLOCK_REDUCE_WIDTH,
        )
    } else {
        WARP_SIZE
    }
}

fn grid_threads(
    resolved: &BoundOp,
    numeric_policy: NumericPolicy,
    cooperative: bool,
    lane_major: bool,
    row_tile: Option<usize>,
    reduce_width: u64,
) -> Result<u64, EmitError> {
    let extents_product = || checked_product(resolved.node, resolved.extents.iter().copied());
    match &resolved.kind {
        BoundOpKind::Elementwise { .. } => extents_product(),
        BoundOpKind::Reduce {
            output_axes,
            epilogue_broadcast_axes,
            keep: Keep::Reduce,
            ..
        } => {
            let kept_rows: u64 = output_axes
                .iter()
                .map(|dim| resolved.extents[*dim as usize])
                .product();
            if lane_major {
                return checked_product(
                    resolved.node,
                    [kept_rows.div_ceil(WARP_SIZE), LANE_MAJOR_BLOCK_WIDTH],
                );
            }
            let row_per_warp = cooperative && kept_rows > 1;
            let output_total: u64 = if epilogue_broadcast_axes.is_empty() || row_per_warp {
                checked_product(
                    resolved.node,
                    output_axes.iter().enumerate().map(|(position, dim)| {
                        let extent = resolved.extents[*dim as usize];
                        if row_tile == Some(position) {
                            extent.div_ceil(ROW_TILE as u64)
                        } else {
                            extent
                        }
                    }),
                )?
            } else {
                extents_product()?
            };
            if cooperative {
                let warps = if epilogue_broadcast_axes.is_empty() || row_per_warp {
                    output_total
                } else {
                    kept_rows
                };
                checked_product(resolved.node, [warps, reduce_width])
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
        BoundOpKind::CachedAttention {
            query_rows,
            kv_heads,
            query_groups,
            ..
        } => match cached_attention_tile_rows(&resolved.kind, numeric_policy) {
            Some(tile_rows) => checked_product(
                resolved.node,
                [query_rows.div_ceil(tile_rows), *kv_heads, ATTENTION_BLOCK_WIDTH],
            ),
            None => checked_product(
                resolved.node,
                [*query_rows, *kv_heads, *query_groups, ATTENTION_BLOCK_WIDTH],
            ),
        },
        BoundOpKind::Iota
        | BoundOpKind::Constant { .. }
        | BoundOpKind::GatedDeltaNet { .. }
        | BoundOpKind::TopFractionSelect { .. }
        | BoundOpKind::RoundBatchedReduce { .. }
        | BoundOpKind::CachedSoftmaxWeights { .. }
        | BoundOpKind::BlockQuantize { .. }
        | BoundOpKind::ElementwiseTwin { .. } => extents_product(),
    }
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
            admit(numeric_policy, NumericRewrite::TreeReduce).is_ok()
                && gather_is_reduction_invariant(resolved, output_axes)
                && is_cooperative_reduce_op(*reduce_op)
        }
        _ => false,
    }
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
    device: EmitDevice,
) -> String {
    let identity = crate::identity::kernel_identity(
        crate::identity::KernelLanguage::Cuda,
        resolved,
        packed_operands,
        crate::identity::MetalOnlyExtras::default(),
        // the cached-attention kernel text (`cuda_cached_attention`) carries no
        // numeric-policy value, so the default never varies this identity string
        proxima_tensor::NumericPolicy::default(),
    );
    format!("{identity}{}", text_plan_token(resolved, quantized, numeric_policy, device))
}

/// The extent-dependent choices that select a reduce kernel's emitted text: which
/// matvec, gemm, row-tile or lane-major body the op's current extents qualify for.
fn text_plan_token(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> String {
    let activation_token = if numeric_policy.activation_quantization
        && q4_0_matvec(resolved, quantized).is_some()
    {
        ACTIVATION_QUANTIZED_MATVEC_TOKEN
    } else {
        ""
    };
    let revision = if let Some(plan) = staged_matvec_plan(resolved, quantized, numeric_policy, device) {
        let source = match (plan.scratch, plan.external.is_some()) {
            (_, true) => "q",
            (true, false) => "g",
            (false, false) => "",
        };
        format!("_smv{}x{}{source}_gid64c9", staged_matvec::STAGED_WARPS, plan.rows)
    } else if let Some(plan) = mma_gemm_plan(resolved, quantized, numeric_policy) {
        match &plan.int8 {
            Some(int8) => format!("_mma8x8x16s8c{}_gid64c9", int8.chunks),
            None => {
                let single = if plan.single_activation { "h" } else { "" };
                format!("_mma16x8x8{single}_gid64c9")
            }
        }
    } else if row_tile_axis(resolved, quantized, numeric_policy).is_some() {
        format!("_rt{ROW_TILE}_gid64c9")
    } else if reduce_is_cooperative(resolved, quantized, numeric_policy)
        && lane_major_reduce(resolved, quantized)
    {
        format!("_lm{LANE_MAJOR_WARPS}_gid64c9")
    } else {
        String::from("_gid64c8")
    };
    format!("{activation_token}{revision}")
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

fn push_unit_stride_broadcast_loop(
    source: &mut String,
    hoisted: &HoistedEpilogue,
    output_axes: &[u16],
    reduce_dim: u16,
    lane_start: &str,
    lane_step: &str,
) {
    source.push_str("    long out_row_base = u.out_base;\n");
    for dim in output_axes {
        source.push_str(&format!(
            "    out_row_base += full_coord[{dim}] * u.broadcast_out_strides[{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "    long out_reduce_stride = u.broadcast_out_strides[{reduce_dim}];\n"
    ));
    source.push_str(&format!(
        "    #pragma unroll 8\n    for (unsigned int r = {lane_start}; r < (unsigned int)u.reduction_total; r += {lane_step}) {{\n"
    ));
    source.push_str(&hoisted.in_loop);
    source.push_str(&format!(
        "        out[out_row_base + (long)r * out_reduce_stride] = {};\n    }}\n",
        hoisted.value
    ));
}

fn push_unit_stride_fold_header(
    source: &mut String,
    resolved: &BoundOp,
    output_axes: &[u16],
    lane_start: &str,
    lane_step: &str,
) {
    for index in 0..resolved.operands().len() {
        source.push_str(&format!(
            "    long off_base{index} = u.operand_base[{index}];\n"
        ));
        for dim in output_axes {
            source.push_str(&format!(
                "    off_base{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
    }
    source.push_str(&format!(
        "    #pragma unroll 8\n    for (unsigned int r = {lane_start}; r < (unsigned int)u.reduction_total; r += {lane_step}) {{\n"
    ));
}

struct HoistedEpilogue {
    before_loop: String,
    in_loop: String,
    value: String,
}

fn push_epilogue_operand_load(
    source: &mut String,
    index: usize,
    rank: usize,
    indent: &str,
    coord: &impl Fn(usize) -> String,
) {
    source.push_str(&format!(
        "{indent}long epi_off{index} = u.epilogue_operand_base[{index}];\n"
    ));
    for dim in 0..rank {
        source.push_str(&format!(
            "{indent}epi_off{index} += {} * u.epilogue_operand_strides[{index}][{dim}];\n",
            coord(dim)
        ));
    }
    source.push_str(&format!(
        "{indent}epi_scratch[{index}] = epi{index}[epi_off{index}];\n"
    ));
}

fn hoisted_broadcast_epilogue(
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    reduce_dims: &[u16],
    unit_reduce: bool,
    rank: usize,
    element_type: &str,
    indent: &str,
) -> HoistedEpilogue {
    let operand_count = epilogue_operands.len();
    let invariant_operand: Vec<bool> = epilogue_operands
        .iter()
        .map(|operand| epilogue_operand_is_loop_invariant(operand, reduce_dims))
        .collect();
    let mut before_loop = format!("    {element_type} epi_scratch[{}];\n", operand_count + 1);
    let mut in_loop = String::new();
    let outer_coord = |dim: usize| format!("full_coord[{dim}]");
    let inner_coord = |dim: usize| format!("broadcast_full_coord[{dim}]");
    for (index, &invariant) in invariant_operand.iter().enumerate() {
        if invariant {
            push_epilogue_operand_load(&mut before_loop, index, rank, "    ", &outer_coord);
        } else if unit_reduce {
            let reduce_dim = reduce_dims[0];
            push_epilogue_operand_load(&mut before_loop, index, rank, "    ", &outer_coord);
            before_loop.push_str(&format!("    long epi_base{index} = epi_off{index};\n"));
            in_loop.push_str(&format!(
                "{indent}epi_scratch[{index}] = epi{index}[epi_base{index} + (long)r * u.epilogue_operand_strides[{index}][{reduce_dim}]];\n"
            ));
        } else {
            push_epilogue_operand_load(&mut in_loop, index, rank, indent, &inner_coord);
        }
    }
    before_loop.push_str(&format!("    epi_scratch[{operand_count}] = accumulator;\n"));
    let value = crate::epilogue::declare_steps_partitioned(
        &mut before_loop,
        &mut in_loop,
        epilogue_body,
        "epi_scratch",
        "epi_step",
        |operand| operand == operand_count || invariant_operand[operand],
        scalar_op_expr,
        |source, index, expr| {
            source.push_str(&format!("    {element_type} epi_step{index} = {expr};\n"));
        },
        |source, index, expr| {
            source.push_str(&format!("{indent}{element_type} epi_step{index} = {expr};\n"));
        },
    );
    HoistedEpilogue { before_loop, in_loop, value }
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

pub(crate) fn preamble(source: &mut String, element_type: &str) {
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
    source.push_str(Q5_0_UNPACK_CUDA);
    source.push('\n');
    source.push_str(Q1_0_UNPACK_CUDA);
    source.push('\n');
    source.push_str(Q2_0_UNPACK_CUDA);
    source.push('\n');
    source.push_str(PTQ1_0_UNPACK_CUDA);
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
        Some(Codec::Q5_0) => Ok(format!(
            "q5_0_element(in{index} + ({offset} / {Q5_0_BLOCK_ELEMENTS}) * {Q5_0_BLOCK_BYTES}, (unsigned int)({offset} % {Q5_0_BLOCK_ELEMENTS}))"
        )),
        Some(Codec::Q1_0) => Ok(format!(
            "q1_0_element(in{index} + ({offset} / {Q1_0_BLOCK_ELEMENTS}) * {Q1_0_BLOCK_BYTES}, (unsigned int)({offset} % {Q1_0_BLOCK_ELEMENTS}))"
        )),
        Some(Codec::Q2_0) => Ok(format!(
            "q2_0_element(in{index} + ({offset} / {Q2_0_BLOCK_ELEMENTS}) * {Q2_0_BLOCK_BYTES}, (unsigned int)({offset} % {Q2_0_BLOCK_ELEMENTS}))"
        )),
        Some(Codec::Pq2_0) => Ok(format!(
            "q2_0_element(in{index} + ({offset} / {PQ2_0_BLOCK_ELEMENTS}) * {PQ2_0_BLOCK_BYTES}, (unsigned int)({offset} % {PQ2_0_BLOCK_ELEMENTS}))"
        )),
        Some(Codec::Ptq1_0) => Ok(format!(
            "ptq1_0_element(in{index} + ({offset} / {PTQ1_0_BLOCK_ELEMENTS}) * {PTQ1_0_BLOCK_BYTES}, (unsigned int)({offset} % {PTQ1_0_BLOCK_ELEMENTS}))"
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
            | Codec::Q4_1
            | Codec::Q8_1
            | Codec::Q8_1Grouped
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
        "{indent}long off{index} = u.operand_base[{index}] + packed_row{index} * u.reduction_total + {};\n",
        packed_reduction_term(index, layout, output_axes, resolved)
    ));
}

/// The packed row's in-row element offset. A weight whose reduction axes nest in the
/// iteration order reads the flat reduction index `r`; one whose checkpoint nests them
/// differently (`ssm_out`'s head axes) reads through its native strides, which
/// `correct_packed_matmul_layouts` derived from the declared layout.
fn packed_reduction_term(
    index: usize,
    layout: &Layout,
    output_axes: &[u16],
    resolved: &BoundOp,
) -> String {
    if packed_reduction_in_order(layout, output_axes, resolved) {
        return String::from("r");
    }
    packed_reduction_axes(layout, output_axes, resolved)
        .iter()
        .map(|axis| format!("full_coord[{axis}] * u.operand_strides[{index}][{axis}]"))
        .collect::<Vec<String>>()
        .join(" + ")
}

fn packed_reduction_axes(layout: &Layout, output_axes: &[u16], resolved: &BoundOp) -> Vec<usize> {
    (0..resolved.extents.len())
        .filter(|axis| !output_axes.contains(&(*axis as u16)) && layout.stride(*axis as u16) != 0)
        .collect()
}

fn packed_reduction_in_order(layout: &Layout, output_axes: &[u16], resolved: &BoundOp) -> bool {
    let mut expected = 1i64;
    packed_reduction_axes(layout, output_axes, resolved)
        .iter()
        .rev()
        .all(|axis| {
            let matches = layout.stride(*axis as u16) == expected;
            expected *= resolved.extents[*axis] as i64;
            matches
        })
}

/// Whether every packed operand's row holds the reduction in flat iteration order, the
/// contract every body that addresses a packed weight as `row + r` (chunked decode,
/// row tiles, staged int8 dots, tensor-core tiles) relies on.
fn flat_packed_rows(resolved: &BoundOp, quantized: &[Option<Codec>]) -> bool {
    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        return false;
    };
    !logical_packed_address()
        && quantized
            .iter()
            .zip(resolved.operands())
            .filter(|(codec, _)| codec.is_some())
            .all(|(_, (_, layout, _))| packed_reduction_in_order(layout, output_axes, resolved))
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
    if reduce_rank == 1 {
        source.push_str(&format!("{indent}reduction_coord[0] = {from};\n"));
        return;
    }
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

/// A packed codec whose device helpers decode eight consecutive elements at
/// once (`<name>_chunk8`), bit-identical to eight `<name>_element` calls.
#[derive(Debug, Clone, Copy)]
struct ChunkCodec {
    name: &'static str,
    block_elements: usize,
    block_bytes: usize,
}

fn chunk_codec(codec: Codec) -> Option<ChunkCodec> {
    let (name, block_elements, block_bytes) = match codec {
        Codec::Q4_0 => ("q4_0", Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES),
        Codec::Q8_0 => ("q8_0", Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES),
        Codec::Q6K => ("q6k", Q4K_BLOCK_ELEMENTS, Q6K_BLOCK_BYTES),
        Codec::Q4K => ("q4k", Q4K_BLOCK_ELEMENTS, Q4K_BLOCK_BYTES),
        Codec::Q1_0 => ("q1_0", Q1_0_BLOCK_ELEMENTS, Q1_0_BLOCK_BYTES),
        Codec::Q2_0 => ("q2_0", Q2_0_BLOCK_ELEMENTS, Q2_0_BLOCK_BYTES),
        Codec::Pq2_0 => ("q2_0", PQ2_0_BLOCK_ELEMENTS, PQ2_0_BLOCK_BYTES),
        _ => return None,
    };
    Some(ChunkCodec {
        name,
        block_elements,
        block_bytes,
    })
}

fn chunk_slots(quantized: &[Option<Codec>]) -> Vec<(usize, ChunkCodec)> {
    quantized
        .iter()
        .enumerate()
        .filter_map(|(index, codec)| Some((index, chunk_codec((*codec)?)?)))
        .collect()
}

fn push_chunk_decode(source: &mut String, index: usize, spec: ChunkCodec) {
    let ChunkCodec {
        name,
        block_elements,
        block_bytes,
    } = spec;
    source.push_str(&format!("        float {name}_chunk{index}[8];\n"));
    source.push_str(&format!(
        "        if ((off{index} & 7) == 0 && chunk + 8 <= u.reduction_total) {{\n\
         \x20           {name}_chunk8(in{index} + (off{index} / {block_elements}) * {block_bytes}, \
         (unsigned int)(off{index} % {block_elements}), {name}_chunk{index});\n\
         \x20       }} else {{\n\
         \x20           for (int slot = 0; slot < 8 && chunk + slot < u.reduction_total; slot++) {{\n\
         \x20               long element = off{index} + slot;\n\
         \x20               {name}_chunk{index}[slot] = {name}_element(in{index} + (element / {block_elements}) * {block_bytes}, \
         (unsigned int)(element % {block_elements}));\n\
         \x20           }}\n\
         \x20       }}\n"
    ));
}

fn push_packed_row_base(
    source: &mut String,
    index: usize,
    layout: &Layout,
    output_axes: &[u16],
    resolved: &BoundOp,
) {
    source.push_str(&format!("    long packed_row{index} = 0;\n"));
    for axis in output_axes {
        let axis_usize = *axis as usize;
        if layout.stride(*axis) != 0 {
            source.push_str(&format!(
                "    packed_row{index} = packed_row{index} * {} + full_coord[{axis_usize}];\n",
                resolved.extents[axis_usize]
            ));
        }
    }
    source.push_str(&format!(
        "    const long row{index} = u.operand_base[{index}] + packed_row{index} * u.reduction_total;\n"
    ));
}

fn push_plain_row_base(source: &mut String, index: usize, rank: usize, reduce_dim: u16) {
    source.push_str(&format!(
        "    long lane_base{index} = u.operand_base[{index}];\n"
    ));
    for dim in 0..rank {
        source.push_str(&format!(
            "    lane_base{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "    const long lane_step{index} = u.operand_strides[{index}][{reduce_dim}];\n"
    ));
}

const LEAN_UNROLL: usize = 4;

/// Whether the flat reduction index walks every plain operand with one stride:
/// outer reduction strides must equal the inner stride times the inner extent.
fn contiguity_guard(quantized: &[Option<Codec>], reduce_dims: &[u16]) -> String {
    let mut conditions: Vec<String> = Vec::new();
    for (index, codec) in quantized.iter().enumerate() {
        if codec.is_some() {
            continue;
        }
        for (slot, pair) in reduce_dims.windows(2).enumerate() {
            conditions.push(format!(
                "u.operand_strides[{index}][{}] == u.operand_strides[{index}][{}] * u.reduction_extents[{}]",
                pair[0],
                pair[1],
                slot + 1
            ));
        }
    }
    if conditions.is_empty() {
        conditions.push("true".to_string());
    }
    format!(
        "    const bool contiguous = {};\n    if (contiguous) {{\n",
        conditions.join(" && ")
    )
}

/// One lane's share of a rank-1 packed reduction. Every lane walks eight-element
/// chunks `WARP_SIZE * 8` apart, in the same order as the generic loop, so the
/// accumulation order (and therefore every bit of the sum) matches it; the
/// differences are that per-row address terms are hoisted out of the loop and
/// that `LEAN_UNROLL` chunks are loaded before any is folded, so the loads of
/// consecutive chunks overlap instead of each waiting on the previous fold.
#[allow(clippy::too_many_arguments)]
fn push_lean_chunk_loops(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    output_axes: &[u16],
    rank: usize,
    reduce_dim: u16,
    quantized: &[Option<Codec>],
    chunk_slots: &[(usize, ChunkCodec)],
    element_type: &str,
) -> Result<(), EmitError> {
    let span = WARP_SIZE as usize * 8;
    for (index, &codec) in quantized.iter().enumerate() {
        if codec.is_some() {
            let (_, layout, _) = &resolved.operands()[index];
            push_packed_row_base(source, index, layout, output_axes, resolved);
        } else {
            push_plain_row_base(source, index, rank, reduce_dim);
        }
    }
    source.push_str("    long start = 0;\n");
    let plain_vectors: Vec<usize> = if element_type == "float" {
        quantized
            .iter()
            .enumerate()
            .filter(|(_, codec)| codec.is_none())
            .map(|(index, _)| index)
            .collect()
    } else {
        Vec::new()
    };
    if !chunk_slots.is_empty() {
        let aligned = chunk_slots
            .iter()
            .map(|(index, _)| format!("((row{index} & 7) == 0)"))
            .chain(
                plain_vectors
                    .iter()
                    .map(|index| format!("((lane_base{index} & 3) == 0 && lane_step{index} == 1)")),
            )
            .collect::<Vec<_>>()
            .join(" && ");
        source.push_str(&format!("    if ({aligned}) {{\n"));
        source.push_str(&format!(
            "        for (; start + {} <= u.reduction_total; start += {}) {{\n",
            LEAN_UNROLL * span,
            LEAN_UNROLL * span
        ));
        for step in 0..LEAN_UNROLL {
            source.push_str(&format!(
                "            const long chunk{step} = start + {} + (long)lane * 8;\n",
                step * span
            ));
            for &(index, spec) in chunk_slots {
                let ChunkCodec {
                    name,
                    block_elements,
                    block_bytes,
                } = spec;
                source.push_str(&format!(
                    "            const long off{index}_{step} = row{index} + chunk{step};\n\
                     \x20           float {name}_chunk{index}_{step}[8];\n\
                     \x20           {name}_chunk8(in{index} + (off{index}_{step} / {block_elements}) * {block_bytes}, \
                     (unsigned int)(off{index}_{step} % {block_elements}), {name}_chunk{index}_{step});\n"
                ));
            }
            for index in &plain_vectors {
                source.push_str(&format!(
                    "            float plain_chunk{index}_{step}[8];\n\
                     \x20           {{\n\
                     \x20               const float4 low = __ldg((const float4 *)(in{index} + lane_base{index} + chunk{step}));\n\
                     \x20               const float4 high = __ldg((const float4 *)(in{index} + lane_base{index} + chunk{step} + 4));\n\
                     \x20               plain_chunk{index}_{step}[0] = low.x; plain_chunk{index}_{step}[1] = low.y;\n\
                     \x20               plain_chunk{index}_{step}[2] = low.z; plain_chunk{index}_{step}[3] = low.w;\n\
                     \x20               plain_chunk{index}_{step}[4] = high.x; plain_chunk{index}_{step}[5] = high.y;\n\
                     \x20               plain_chunk{index}_{step}[6] = high.z; plain_chunk{index}_{step}[7] = high.w;\n\
                     \x20           }}\n"
                ));
            }
        }
        for step in 0..LEAN_UNROLL {
            source.push_str("            {\n");
            source.push_str("                #pragma unroll\n");
            source.push_str("                for (int inner = 0; inner < 8; inner++) {\n");
            push_lean_fold(
                source,
                resolved,
                reduce_op,
                quantized,
                chunk_slots,
                element_type,
                &format!("chunk{step}"),
                &format!("_{step}"),
                &plain_vectors,
            )?;
            source.push_str("                }\n            }\n");
        }
        source.push_str("        }\n    }\n");
    }
    source.push_str(&format!(
        "    for (long chunk = start + (long)lane * 8; chunk < u.reduction_total; chunk += {span}) {{\n"
    ));
    if !chunk_slots.is_empty() {
        source.push_str("        long r = chunk;\n");
    }
    for &(index, spec) in chunk_slots {
        source.push_str(&format!("        long off{index} = row{index} + r;\n"));
        push_chunk_decode(source, index, spec);
    }
    source.push_str("        for (long inner = 0; inner < 8 && chunk + inner < u.reduction_total; inner++) {\n");
    push_lean_fold(
        source,
        resolved,
        reduce_op,
        quantized,
        chunk_slots,
        element_type,
        "chunk",
        "",
        &[],
    )?;
    source.push_str("        }\n    }\n");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_lean_fold(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    quantized: &[Option<Codec>],
    chunk_slots: &[(usize, ChunkCodec)],
    element_type: &str,
    chunk_var: &str,
    suffix: &str,
    plain_vectors: &[usize],
) -> Result<(), EmitError> {
    source.push_str(&format!("            long r = {chunk_var} + inner;\n"));
    let chunk_spec = |index: usize| {
        chunk_slots
            .iter()
            .find(|(slot, _)| *slot == index)
            .map(|(_, spec)| *spec)
    };
    for (index, &codec) in quantized.iter().enumerate() {
        if chunk_spec(index).is_some() || plain_vectors.contains(&index) {
            continue;
        }
        if codec.is_some() {
            source.push_str(&format!("            long off{index} = row{index} + r;\n"));
        } else {
            source.push_str(&format!(
                "            long off{index} = lane_base{index} + r * lane_step{index};\n"
            ));
        }
    }
    source.push_str(&format!(
        "            {element_type} scratch[{}];\n",
        resolved.operands().len().max(1)
    ));
    for (index, &codec) in quantized.iter().enumerate() {
        if let Some(spec) = chunk_spec(index) {
            source.push_str(&format!(
                "            scratch[{index}] = {}_chunk{index}{suffix}[inner];\n",
                spec.name
            ));
            continue;
        }
        if plain_vectors.contains(&index) {
            source.push_str(&format!(
                "            scratch[{index}] = plain_chunk{index}{suffix}[inner];\n"
            ));
            continue;
        }
        source.push_str(&format!(
            "            scratch[{index}] = {};\n",
            operand_read(resolved.node, index, &format!("off{index}"), codec)?
        ));
    }
    let value_expr = push_body_steps(source, resolved.element_body(), "            ", element_type);
    source.push_str(&format!("            {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "            accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("            seeded = true;\n");
    Ok(())
}

fn push_lane_major_merge(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    element_type: &str,
) -> Result<(), EmitError> {
    source.push_str(&format!(
        "    __shared__ {element_type} lane_partials[{LANE_MAJOR_WARPS}][{WARP_SIZE}];\n"
    ));
    source.push_str("    lane_partials[fold_warp][lane] = accumulator;\n");
    source.push_str("    __syncthreads();\n");
    source.push_str("    accumulator = lane_partials[0][lane];\n");
    for warp in 1..LANE_MAJOR_WARPS {
        let partial = format!("lane_partials[{warp}][lane]");
        let combined = shuffle_combine_expr(resolved.node, reduce_op, "accumulator", &partial)?;
        source.push_str(&format!("    accumulator = {combined};\n"));
    }
    source.push_str("    if (!in_range) { return; }\n");
    Ok(())
}

fn push_shuffle_merge(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    element_type: &str,
    identity: &str,
    block_reduce: bool,
) -> Result<(), EmitError> {
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

    if block_reduce {
        source.push_str("    if (blockDim.x > 32u) {\n");
        source.push_str(&format!(
            "        __shared__ {element_type} warp_partials[32];\n"
        ));
        source.push_str("        if (lane == 0u) { warp_partials[tid >> 5u] = accumulator; }\n");
        source.push_str("        __syncthreads();\n");
        source.push_str(&format!(
            "        accumulator = lane < (blockDim.x >> 5u) ? warp_partials[lane] : {identity};\n"
        ));
        source.push_str("        #pragma unroll\n");
        source.push_str(&format!(
            "        for (int shift = {}; shift > 0; shift >>= 1) {{\n",
            WARP_SIZE / 2
        ));
        source.push_str(&format!(
            "            {element_type} shuffled = __shfl_down_sync(0xffffffffu, accumulator, shift);\n"
        ));
        let second_expr =
            shuffle_combine_expr(resolved.node, reduce_op, "accumulator", "shuffled")?;
        source.push_str(&format!("            accumulator = {second_expr};\n"));
        source.push_str("        }\n    }\n");
    }
    Ok(())
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

    let block_reduce = is_broadcast_epilogue && !packed_chunked;
    let lane_major = lane_major_reduce(resolved, quantized);
    let (leader, lane_start, lane_stride) = if lane_major {
        ("fold_warp == 0u", "(long)fold_warp", format!("{LANE_MAJOR_WARPS}"))
    } else if block_reduce {
        ("tid == 0u", "(long)tid", "(long)blockDim.x".to_string())
    } else {
        ("lane == 0u", "(long)lane", WARP_SIZE.to_string())
    };
    if lane_major {
        source.push_str("    unsigned int tid = threadIdx.x;\n");
        source.push_str("    unsigned int lane = tid & 31u;\n");
        source.push_str("    unsigned int fold_warp = tid >> 5u;\n");
        source.push_str(&format!(
            "    long output_index = (long)blockIdx.x * {WARP_SIZE}L + (long)lane;\n"
        ));
        source.push_str("    bool in_range = output_index < u.output_total;\n");
        source.push_str("    if (!in_range) { output_index = 0; }\n");
    } else if block_reduce {
        source.push_str("    long output_index = (long)blockIdx.x;\n");
        source.push_str("    if (output_index >= u.output_total) { return; }\n");
        source.push_str("    unsigned int tid = threadIdx.x;\n");
        source.push_str("    unsigned int lane = tid & 31u;\n");
    } else {
        source.push_str(&format!("    long output_index = gid / {WARP_SIZE};\n"));
        source.push_str("    if (output_index >= u.output_total) { return; }\n");
        source.push_str(&format!(
            "    unsigned int lane = (unsigned int)(gid % {WARP_SIZE});\n"
        ));
    }

    source.push_str(&format!("    long full_coord[{rank_len}];\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }

    if output_rank > 0 {
        source.push_str(&format!("    long output_coord[{output_rank_len}];\n"));
        source.push_str("    long remaining = output_index;\n");
        for index in (0..output_rank).rev() {
            let slot = if is_broadcast_epilogue {
                usize::from(output_axes[index])
            } else {
                index
            };
            source.push_str(&format!(
                "    output_coord[{index}] = remaining % u.output_extents[{slot}]; \
                 remaining /= u.output_extents[{slot}];\n"
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
    source.push_str(&format!("    if ({leader}) {{\n"));
    source.push_str(&format!("        accumulator = {init_expr};\n"));
    source.push_str(&format!("        seeded = {seeded_init};\n"));
    source.push_str("    } else {\n");
    source.push_str(&format!("        accumulator = {identity};\n"));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    let flat_rows = flat_packed_rows(resolved, quantized);
    let chunk_slots = if packed_chunked && flat_rows {
        chunk_slots(quantized)
    } else {
        Vec::new()
    };
    let chunk_spec = |index: usize| {
        chunk_slots
            .iter()
            .find(|(slot, _)| *slot == index)
            .map(|(_, spec)| *spec)
    };
    let lean = packed_chunked
        && reduce_rank >= 1
        && flat_rows
        && gather_count(resolved) == 0;
    let guarded = lean && reduce_rank > 1;

    if lean {
        if guarded {
            source.push_str(&contiguity_guard(quantized, reduce_dims));
        }
        push_lean_chunk_loops(
            source,
            resolved,
            reduce_op,
            output_axes,
            rank,
            reduce_dims[reduce_rank - 1],
            quantized,
            &chunk_slots,
            element_type,
        )?;
        if guarded {
            source.push_str("    }\n    if (!contiguous) {\n");
        }
    }
    let unit_fold = (block_reduce || !packed_chunked)
        && reduce_rank == 1
        && gather_count(resolved) == 0
        && index32_reach_fits(resolved.extents[reduce_dims[0] as usize]);
    if !lean || guarded {
        if unit_fold {
            let (fold_start, fold_step) = if lane_major {
                ("fold_warp", format!("{LANE_MAJOR_WARPS}u"))
            } else if block_reduce {
                ("tid", "blockDim.x".to_string())
            } else {
                ("lane", "32u".to_string())
            };
            push_unit_stride_fold_header(
                source,
                resolved,
                output_axes,
                fold_start,
                &fold_step,
            );
        } else if packed_chunked {
            source.push_str(&format!(
                "    for (long chunk = (long)lane * 8; chunk < u.reduction_total; chunk += {} * 8) {{\n",
                WARP_SIZE
            ));
            if reduce_rank > 0 {
                push_reduction_coord_decompose(
                    source,
                    "chunk",
                    reduce_rank,
                    reduce_rank_len,
                    "        ",
                );
            }
            if !chunk_slots.is_empty() {
                source.push_str("        long r = chunk;\n");
            }
            for &(index, spec) in &chunk_slots {
                let (_, layout, _) = &resolved.operands()[index];
                push_packed_matmul_offset(source, index, layout, output_axes, resolved, "        ");
                push_gather_base_add(source, resolved, index, "        ");
                push_chunk_decode(source, index, spec);
            }
            source.push_str("        for (long inner = 0; inner < 8 && chunk + inner < u.reduction_total; inner++) {\n");
            source.push_str("            long r = chunk + inner;\n");
        } else {
            source.push_str(&format!(
                "    #pragma unroll 8\n    for (long r = {lane_start}; r < u.reduction_total; r += {lane_stride}) {{\n"
            ));
        }
        if reduce_rank > 0 && !packed_chunked && !unit_fold {
            push_reduction_coord_decompose(
                source,
                "r",
                reduce_rank,
                reduce_rank_len,
                "        ",
            );
        }
        if reduce_rank > 0 && !unit_fold {
            for (index, dim) in reduce_dims.iter().enumerate() {
                source.push_str(&format!(
                    "        full_coord[{dim}] = reduction_coord[{index}];\n"
                ));
            }
        }

        for (index, &codec) in quantized.iter().enumerate() {
            if chunk_spec(index).is_some() {
                continue;
            }
            if codec.is_some() {
                let (_, layout, _) = &resolved.operands()[index];
                push_packed_matmul_offset(source, index, layout, output_axes, resolved, "        ");
            } else if unit_fold {
                let reduce_dim = reduce_dims[0];
                source.push_str(&format!(
                    "        long off{index} = off_base{index} + (long)r * u.operand_strides[{index}][{reduce_dim}];\n"
                ));
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
            if let Some(spec) = chunk_spec(index) {
                source.push_str(&format!(
                    "        scratch[{index}] = {}_chunk{index}[inner];\n",
                    spec.name
                ));
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
        if guarded {
            source.push_str("    }\n");
        }
    }

    if lane_major {
        push_lane_major_merge(source, resolved, reduce_op, element_type)?;
    } else {
        push_shuffle_merge(source, resolved, reduce_op, element_type, identity, block_reduce)?;
    }

    if is_broadcast_epilogue {
        if !lane_major {
            // Every lane participates in the broadcast write below. The
            // down-shuffle tree leaves the complete reduction only in lane 0;
            // without this broadcast, lanes 1..31 reuse their partial sums as
            // if they were the norm, corrupting every other output coordinate.
            source.push_str("    accumulator = __shfl_sync(0xffffffffu, accumulator, 0);\n");
        }
        // Recreate the non-reduction coordinate after the fold, whose loop
        // leaves `full_coord` holding its last reduction coordinate. Each
        // lane then owns a strided share of the full output row.
        for dim in 0..rank {
            source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
        }
        for (index, dim) in output_axes.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
        let unit_epilogue = block_reduce
            && reduce_rank == 1
            && gather_count(resolved) == 0
            && !reduce_epilogue_is_identity(epilogue_body, epilogue_operands)
            && index32_reach_fits(resolved.extents[reduce_dims[0] as usize]);
        let hoisted = (reduce_rank > 0
            && !reduce_epilogue_is_identity(epilogue_body, epilogue_operands))
        .then(|| {
            hoisted_broadcast_epilogue(
                epilogue_body,
                epilogue_operands,
                reduce_dims,
                unit_epilogue,
                rank,
                element_type,
                "        ",
            )
        });
        if let Some(hoisted) = &hoisted {
            source.push_str(&hoisted.before_loop);
        }
        if let (true, Some(hoisted)) = (unit_epilogue, &hoisted) {
            let (start, step) = if lane_major {
                ("fold_warp", format!("{LANE_MAJOR_WARPS}u"))
            } else {
                ("tid", "blockDim.x".to_string())
            };
            push_unit_stride_broadcast_loop(
                source,
                hoisted,
                output_axes,
                reduce_dims[0],
                start,
                &step,
            );
            return Ok(());
        }
        source.push_str(&format!(
            "    #pragma unroll 8\n    for (long r = {lane_start}; r < u.reduction_total; r += {lane_stride}) {{\n"
        ));
        if reduce_rank > 0 {
            source.push_str(&format!(
                "        long broadcast_reduce_coord[{reduce_rank_len}];\n"
            ));
            if reduce_rank == 1 {
                source.push_str("        broadcast_reduce_coord[0] = r;\n");
            } else {
                source.push_str("        long broadcast_remaining = r;\n");
                for index in (0..reduce_rank).rev() {
                    source.push_str(&format!(
                        "        broadcast_reduce_coord[{index}] = broadcast_remaining % u.reduction_extents[{index}]; broadcast_remaining /= u.reduction_extents[{index}];\n"
                    ));
                }
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
            match &hoisted {
                Some(hoisted) => {
                    source.push_str(&hoisted.in_loop);
                    source.push_str(&format!(
                        "        out[broadcast_out_offset] = {};\n",
                        hoisted.value
                    ));
                }
                None => push_reduce_epilogue_write(
                    source,
                    epilogue_body,
                    epilogue_operands,
                    rank,
                    element_type,
                    "        ",
                    |dim| format!("broadcast_full_coord[{dim}]"),
                    "accumulator",
                    "broadcast_out_offset",
                ),
            }
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
    device: EmitDevice,
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

    let staged = staged_matvec_plan(resolved, quantized, numeric_policy, device);
    let gemm = mma_gemm_plan(resolved, quantized, numeric_policy).filter(|_| staged.is_none());
    let mut source = String::new();
    preamble(&mut source, element_type);
    if gemm.is_some() {
        source.push_str(MMA_INSTRUCTION);
        source.push_str(MMA_GEMM_HELPERS);
        if gemm.as_ref().is_some_and(|plan| plan.codec == Codec::Float16) {
            source.push_str(MMA_F16_HELPER);
        }
    }
    if let Some(int8) = gemm.as_ref().and_then(|plan| plan.int8.as_ref()) {
        source.push_str(INT8_MMA_INSTRUCTION);
        source.push_str(&int8.dot.helpers);
    }
    if let Some(plan) = &staged {
        source.push_str(&plan.dot.helpers);
    }

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

    if let Some(plan) = &staged {
        source = source.replacen(
            "__global__ void ",
            &format!("__global__ void __launch_bounds__({STAGED_THREADS}) "),
            1,
        );
        if plan.scratch {
            source = source.replacen(
                "const Uniforms* __restrict__ u_ptr\n",
                "const Uniforms* __restrict__ u_ptr,\n    const int* __restrict__ staged_scratch\n",
                1,
            );
        }
        push_staged_matvec_body(&mut source, resolved, plan)?;
    } else if let Some(plan) = gemm {
        source = source.replacen(
            "__global__ void ",
            &format!("__global__ void __launch_bounds__({GEMM_THREADS}, 2) "),
            1,
        );
        push_mma_gemm_body(&mut source, resolved, &plan, element_type)?;
    } else if let Some(tile_index) = row_tile_axis(resolved, quantized, numeric_policy) {
        push_row_tiled_cooperative_body(&mut source, resolved, quantized, element_type, tile_index)?;
    } else if reduce_is_cooperative(resolved, quantized, numeric_policy) {
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

__device__ __forceinline__ void q4k_chunk8(const unsigned char *block, unsigned int index, float *out) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    unsigned short dmin_bits = (unsigned short)((unsigned int)block[2] | ((unsigned int)block[3] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    float dmin = __half2float(__ushort_as_half(dmin_bits));

    const unsigned char *scales = block + 4;
    unsigned int group = index / 64u;
    unsigned int within = index % 64u;
    int low_nibble = within < 32u;
    unsigned int sub_block = 2u * group + (low_nibble ? 0u : 1u);
    unsigned int byte_index = group * 32u + (within % 32u);

    float scale = d * (float)q4k_scale(scales, sub_block);
    float minimum = dmin * (float)q4k_min(scales, sub_block);
    uint2 packed = *(const uint2 *)(block + 16u + byte_index);
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        unsigned int word = slot < 4 ? packed.x : packed.y;
        unsigned int byte = (word >> (8u * (slot & 3))) & 0xFFu;
        unsigned int nibble = low_nibble ? (byte & 0x0Fu) : (byte >> 4u);
        out[slot] = scale * (float)nibble - minimum;
    }
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
    const unsigned short *pairs = (const unsigned short *)(block + 2u + index);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        unsigned int word = (unsigned int)pairs[pair];
        out[2 * pair] = (float)((signed char)(word & 0xFFu)) * d;
        out[2 * pair + 1] = (float)((signed char)(word >> 8u)) * d;
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

/// CUDA C source for unpacking one element of a `Q5_0` block — ports
/// `crate::msl::Q5_0_UNPACK_MSL`'s `q5_0_element` exactly: 22-byte block,
/// `d` f16 at 0, `qh` at 2, `qs` nibbles at 6, `value = (level - 16) * d`.
pub const Q5_0_UNPACK_CUDA: &str = "\
__device__ __forceinline__ float q5_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8) | ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
    bool high_half = index >= 16u;
    unsigned int local = high_half ? (index - 16u) : index;
    unsigned int byte = (unsigned int)block[6u + local];
    unsigned int level;
    if (high_half) {
        level = (byte >> 4u) | ((qh >> (local + 12u)) & 0x10u);
    } else {
        level = (byte & 0x0Fu) | (((qh >> local) << 4u) & 0x10u);
    }
    return ((float)level - 16.0f) * d;
}
";

pub const Q5_0_BLOCK_BYTES: usize = proxima_gguf::quant::q5_0::BLOCK_BYTES;
pub const Q5_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q5_0::QK5_0;

/// CUDA C source for unpacking one element of a `Q1_0` block (PrismML `dequantize_row_q1_0`):
/// 18-byte block, `d` f16 at 0, one sign bit per element from byte 2 (LSB first),
/// `value = bit ? d : -d`.
pub const Q1_0_UNPACK_CUDA: &str = "\
__device__ __forceinline__ float q1_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int bit = ((unsigned int)block[2u + (index >> 3u)] >> (index & 7u)) & 1u;
    return bit ? d : -d;
}
__device__ __forceinline__ void q1_0_chunk8(const unsigned char *block, unsigned int index, float *out) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int bits = (unsigned int)block[2u + (index >> 3u)];
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        out[slot] = ((bits >> slot) & 1u) ? d : -d;
    }
}
";

pub const Q1_0_BLOCK_BYTES: usize = proxima_gguf::quant::q1_0::BLOCK_BYTES;
pub const Q1_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q1_0::QK_Q1_0;

/// CUDA C source for unpacking one element of a `Q2_0` or `PQ2_0` block (PrismML
/// `dequantize_row_q2_0` / `dequantize_row_pq2_0`): `d` f16 at 0, two bits per element from
/// byte 2 (LSB first), `value = (code - 1) * d`. The two codecs share this body and differ
/// only in the block stride the caller applies.
pub const Q2_0_UNPACK_CUDA: &str = "\
__device__ __forceinline__ float q2_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int code = ((unsigned int)block[2u + (index >> 2u)] >> ((index & 3u) * 2u)) & 3u;
    return ((float)code - 1.0f) * d;
}
__device__ __forceinline__ void q2_0_chunk8(const unsigned char *block, unsigned int index, float *out) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int bits = (unsigned int)*(const unsigned short *)(block + 2u + (index >> 2u));
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        out[slot] = ((float)((bits >> (2u * slot)) & 3u) - 1.0f) * d;
    }
}
";

pub const Q2_0_BLOCK_BYTES: usize = proxima_gguf::quant::q2_0::BLOCK_BYTES;
pub const Q2_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q2_0::QK_Q2_0;
pub const PQ2_0_BLOCK_BYTES: usize = proxima_gguf::quant::pq2_0::BLOCK_BYTES;
pub const PQ2_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::pq2_0::QK_PQ2_0;

/// CUDA C source for unpacking one element of a `PTQ1_0` block (PrismML
/// `dequantize_row_ptq1_0`): 28-byte block, `qs[24]` then `qh[2]` then `d` f16 at 26. Elements
/// 0..80 are the five trits of `qs[0..16]` (element `n * 16 + m` is trit `n` of byte `m`),
/// 80..120 the five trits of `qs[16..24]` (`n * 8 + m`), 120..128 the four trits of `qh`
/// (`n * 2 + m`). Trit `n` of byte `b` is `((b * 3^n mod 256) * 3 >> 8) - 1`, times `d`.
pub const PTQ1_0_UNPACK_CUDA: &str = "\
__device__ __forceinline__ float ptq1_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[26] | ((unsigned int)block[27] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    unsigned int byte_index;
    unsigned int position;
    if (index < 80u) {
        byte_index = index & 15u;
        position = index >> 4u;
    } else if (index < 120u) {
        byte_index = 16u + ((index - 80u) & 7u);
        position = (index - 80u) >> 3u;
    } else {
        byte_index = 24u + ((index - 120u) & 1u);
        position = (index - 120u) >> 1u;
    }
    unsigned int power = position == 0u ? 1u : position == 1u ? 3u : position == 2u ? 9u : position == 3u ? 27u : 81u;
    unsigned int scaled = ((unsigned int)block[byte_index] * power) & 255u;
    return (float)((int)((scaled * 3u) >> 8u) - 1) * d;
}
";

pub const PTQ1_0_BLOCK_BYTES: usize = proxima_gguf::quant::ptq1_0::BLOCK_BYTES;
pub const PTQ1_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::ptq1_0::QK_PTQ1_0;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaAttentionClass {
    Static,
    CachedBound,
}

pub(crate) fn cuda_cached_attention_class(kind: &BoundOpKind) -> Option<CudaAttentionClass> {
    let BoundOpKind::CachedAttention {
        operands,
        cached_key_rows,
        head_dim,
        rotary_dim,
        ..
    } = kind
    else {
        return None;
    };
    let (static_operands, bound_operands) = if rotary_dim < head_dim {
        (11, 12)
    } else {
        (8, 9)
    };
    match operands.len() {
        count if count == static_operands => Some(CudaAttentionClass::Static),
        count if count == bound_operands && *cached_key_rows > 0 => {
            Some(CudaAttentionClass::CachedBound)
        }
        _ => None,
    }
}

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
    use crate::msl::{CachedAttentionForm, cached_attention_form};
    use proxima_tensor::test_support::Lcg;

    fn no_packed() -> PackedOperands {
        BTreeMap::new()
    }

    fn attention_kind(
        operand_count: u32,
        cached_key_rows: u64,
        head_dim: u64,
        rotary_dim: u64,
    ) -> BoundOpKind {
        BoundOpKind::CachedAttention {
            operands: (0..operand_count)
                .map(|index| {
                    (
                        NodeId(index),
                        Layout {
                            base: 0,
                            strides: vec![1_i64].into(),
                        },
                        None,
                    )
                })
                .collect(),
            query_rows: 1,
            cached_key_rows,
            new_key_rows: 1,
            kv_heads: 1,
            query_groups: 8,
            head_dim,
            rotary_dim,
            scale: 1.0,
            cached_lower_inclusive: i64::MIN,
            new_upper_inclusive: 0,
                arithmetic: proxima_tensor::CachedAttentionArithmetic::OnlineF32,
        }
    }

    #[test]
    fn classifier_accepts_eight_operand_as_static() {
        let kind = attention_kind(8, 5, 256, 256);

        assert_eq!(
            cuda_cached_attention_class(&kind),
            Some(CudaAttentionClass::Static)
        );
    }

    #[test]
    fn classifier_accepts_nine_operand_nonzero_cached_as_cached_bound() {
        let kind = attention_kind(9, 5, 256, 256);

        assert_eq!(
            cuda_cached_attention_class(&kind),
            Some(CudaAttentionClass::CachedBound)
        );
    }

    #[test]
    fn classifier_declines_single_range_and_mismatched_pass_planes_and_non_attention() {
        let single_range = attention_kind(9, 0, 256, 256);
        let partial_rotary_without_pass_planes = attention_kind(8, 5, 256, 64);
        let unexpected_count = attention_kind(11, 5, 256, 256);
        let elementwise = elementwise_tanh_op(4).kind;

        assert_eq!(cuda_cached_attention_class(&single_range), None);
        assert_eq!(
            cuda_cached_attention_class(&partial_rotary_without_pass_planes),
            None
        );
        assert_eq!(cuda_cached_attention_class(&unexpected_count), None);
        assert_eq!(cuda_cached_attention_class(&elementwise), None);
    }

    #[test]
    fn classifier_accepts_partial_rotary_with_pass_planes() {
        assert_eq!(
            cuda_cached_attention_class(&attention_kind(11, 5, 256, 64)),
            Some(CudaAttentionClass::Static)
        );
        assert_eq!(
            cuda_cached_attention_class(&attention_kind(12, 5, 256, 64)),
            Some(CudaAttentionClass::CachedBound)
        );
    }

    #[test]
    fn partial_rotary_kernel_reads_the_pass_planes_after_the_cached_len_slot() {
        let op = attention_op_with(attention_kind(12, 5, 256, 64));

        let kernel = emit_cuda(&op, &no_packed()).expect("partial rotary two-range form emits");

        assert!(kernel.source.contains("pass_query = in9"));
        assert!(kernel.source.contains("pass_cached_key = in10"));
        assert!(kernel.source.contains("pass_new_key = in11"));
        assert!(kernel.source.contains("constexpr long pass_dim = head_dim - rotary_dim"));
    }

    #[test]
    fn entry_name_differs_when_rotary_dim_differs() {
        let full = attention_op_with(attention_kind(9, 32, 256, 256));
        let partial = attention_op_with(attention_kind(12, 32, 256, 64));

        let full_kernel = emit_cuda(&full, &no_packed()).expect("full rotary emits");
        let partial_kernel = emit_cuda(&partial, &no_packed()).expect("partial rotary emits");

        assert_ne!(full_kernel.entry, partial_kernel.entry);
    }

    fn attention_op_with(kind: BoundOpKind) -> BoundOp {
        BoundOp {
            node: NodeId(9),
            dtype: DType::FLOAT32,
            extents: vec![1, 1, 8, 256],
            kind,
        }
    }

    #[test]
    fn cached_bound_kernel_bindings_are_nine_inputs_output_uniforms() {
        let cached = attention_op_with(attention_kind(9, 5, 256, 256));
        let static_op = attention_op_with(attention_kind(8, 5, 256, 256));

        let cached_kernel = emit_cuda(&cached, &no_packed()).expect("two-range form emits");
        let static_kernel = emit_cuda(&static_op, &no_packed()).expect("static form emits");

        let inputs = |count: u32| -> Vec<Binding> {
            (0..count).map(|index| Binding::Input(NodeId(index))).collect()
        };
        let mut nine = inputs(9);
        nine.extend([Binding::Output(NodeId(9)), Binding::Uniforms]);
        let mut eight = inputs(8);
        eight.extend([Binding::Output(NodeId(9)), Binding::Uniforms]);
        assert_eq!(cached_kernel.bindings, nine);
        assert_eq!(static_kernel.bindings, eight);
    }

    #[test]
    fn cached_bound_entry_name_is_independent_of_bucket_capacity() {
        let bucket_32 = attention_op_with(attention_kind(9, 32, 256, 256));
        let bucket_64 = attention_op_with(attention_kind(9, 64, 256, 256));

        let kernel_32 = emit_cuda(&bucket_32, &no_packed()).expect("bucket 32 emits");
        let kernel_64 = emit_cuda(&bucket_64, &no_packed()).expect("bucket 64 emits");

        assert_eq!(kernel_32.entry, kernel_64.entry);
        assert_eq!(kernel_32.source, kernel_64.source);
    }

    #[test]
    fn entry_name_differs_when_head_dim_differs() {
        let head_64 = attention_op_with(attention_kind(9, 32, 64, 64));
        let head_256 = attention_op_with(attention_kind(9, 32, 256, 256));

        let kernel_64 = emit_cuda(&head_64, &no_packed()).expect("head 64 emits");
        let kernel_256 = emit_cuda(&head_256, &no_packed()).expect("head 256 emits");

        assert_ne!(kernel_64.entry, kernel_256.entry);
    }

    #[test]
    fn cached_bound_uniforms_hold_total_rows() {
        let mut kind = attention_kind(9, 5, 256, 256);
        let BoundOpKind::CachedAttention { query_rows, .. } = &mut kind else {
            unreachable!("attention_kind always builds a CachedAttention kind");
        };
        *query_rows = 3;
        let op = attention_op_with(kind);

        let bytes = pack_cuda_uniforms(&op).expect("uniforms pack");
        let total = i64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"));

        assert_eq!(total, 3 * 8);
        assert_eq!(bytes.len(), 32, "a multi-row form also carries its row bounds");
    }

    fn multi_row_attention_op(rows: u64) -> BoundOp {
        let mut kind = attention_kind(9, 5, 256, 256);
        let BoundOpKind::CachedAttention {
            query_rows,
            new_key_rows,
            new_upper_inclusive,
            ..
        } = &mut kind
        else {
            unreachable!("attention_kind always builds a CachedAttention kind");
        };
        *query_rows = rows;
        *new_key_rows = rows;
        *new_upper_inclusive = 0;
        attention_op_with(kind)
    }

    #[test]
    fn multi_row_kernel_text_is_independent_of_the_chunk_width() {
        let wide = multi_row_attention_op(512);
        let tail = multi_row_attention_op(277);

        let wide_kernel = emit_cuda(&wide, &no_packed()).expect("512 rows emit");
        let tail_kernel = emit_cuda(&tail, &no_packed()).expect("277 rows emit");

        assert_eq!(wide_kernel.entry, tail_kernel.entry);
        assert_eq!(wide_kernel.source, tail_kernel.source);
        assert_ne!(
            pack_cuda_uniforms(&wide).expect("wide packs"),
            pack_cuda_uniforms(&tail).expect("tail packs"),
            "the width reaches the kernel through its uniforms"
        );
    }

    #[test]
    fn single_row_kernel_keeps_its_bounds_as_constants() {
        let kernel = emit_cuda(&attention_op_with(attention_kind(9, 5, 256, 256)), &no_packed())
            .expect("decode form emits");

        assert!(kernel.source.contains("constexpr long new_key_rows = 1L;"));
    }

    #[test]
    fn cached_bound_source_reads_cached_len_and_rescales_online_softmax() {
        let op = attention_op_with(attention_kind(9, 5, 256, 256));

        let kernel = emit_cuda(&op, &no_packed()).expect("two-range form emits");

        assert!(kernel.source.contains("(long)in8[0]"));
        assert!(kernel.source.contains("expf(maximum - next_max)"));
        assert!(kernel.source.contains("weighted[local_dimension] * rescale"));
        assert!(kernel.source.contains("__shfl_xor_sync(0xffffffffu, partial, offset)"));
    }

    /// One attention vector is one block of `ATTENTION_WARPS` warps, each on a
    /// strided share of the keys, merged through shared memory.
    #[test]
    fn cached_attention_gives_each_vector_a_block_of_warps_that_merge_softmax_state() {
        let op = attention_op_with(attention_kind(9, 5, 256, 256));
        let kernel = emit_cuda(&op, &no_packed()).expect("two-range form emits");
        let bytes = pack_cuda_uniforms(&op).expect("uniforms pack");
        let vectors = i64::from_le_bytes(bytes.as_slice().try_into().expect("8 bytes")) as u64;

        assert!(vectors > 0);
        assert_eq!(kernel.grid.block_width, Some(ATTENTION_WARPS * WARP_SIZE));
        assert_eq!(kernel.grid.threads, vectors * ATTENTION_WARPS * WARP_SIZE);
        assert_eq!(kernel.grid.blocks(), vectors);
        assert!(kernel.source.contains("constexpr int warps = 4;"));
        assert!(kernel.source.contains("key += warps"));
        assert!(kernel.source.contains("__syncthreads();"));
        assert!(kernel.source.contains("shared_weighted[source * head_dim + dimension]"));
    }

    #[cfg(feature = "metal-attn-split-decode")]
    #[test]
    fn classifier_does_not_follow_metal_form_selection() {
        let kind = attention_kind(9, 32, 256, 256);
        let metal_form = cached_attention_form(&kind, NumericPolicy::llama_relaxed());

        assert!(
            matches!(metal_form, Some(CachedAttentionForm::TwoRangeDecodeSplit { .. })),
            "control needs metal to select DecodeSplit, got {metal_form:?}"
        );
        assert_eq!(
            cuda_cached_attention_class(&kind),
            Some(CudaAttentionClass::CachedBound)
        );
    }

    fn elementwise_tanh_op(extent: u32) -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
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
                dtype: DType::FLOAT32,
                extent: Extent::Static(8),
            },
            Op::Constant {
                dtype: DType::FLOAT32,
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
            dtype: DType::FLOAT32,
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

    #[test]
    fn emit_text_equal_across_kv_extents() {
        let short = emit_cuda(&row_sum_plus_bias_op(4, 8), &no_packed()).expect("emit succeeds");
        let long = emit_cuda(&row_sum_plus_bias_op(4, 16), &no_packed()).expect("emit succeeds");
        assert_eq!(short.source, long.source);
        assert_eq!(short.entry, long.entry);
    }

    #[test]
    fn reuse_matches_fresh_emit_for_every_op() {
        let pairs = [
            (elementwise_tanh_op(4), elementwise_tanh_op(4096)),
            (row_sum_plus_bias_op(4, 8), row_sum_plus_bias_op(4, 16)),
            (scan_op(8), scan_op(64)),
        ];
        for (previous, resolved) in pairs {
            let previous_kernel = emit_cuda(&previous, &no_packed()).expect("emit succeeds");
            let (kernels, reused) = emit_cuda_reusing(
                core::slice::from_ref(&previous),
                vec![previous_kernel],
                core::slice::from_ref(&resolved),
                &no_packed(),
                NumericPolicy::default(),
                EmitDevice::default(),
            )
            .expect("reuse succeeds");
            let fresh = emit_cuda(&resolved, &no_packed()).expect("emit succeeds");
            assert_eq!(reused, 1, "extent-only change must reuse the kernel");
            assert_eq!(kernels, vec![fresh]);
        }
    }

    #[test]
    fn reuse_refuses_a_kind_change_at_the_same_node() {
        let previous = matmul_reduce_op(4, 8, ScalarOp::Add);
        let resolved = matmul_reduce_op(4, 8, ScalarOp::Maximum);
        let previous_kernel = emit_cuda(&previous, &no_packed()).expect("emit succeeds");
        let (kernels, reused) = emit_cuda_reusing(
            core::slice::from_ref(&previous),
            vec![previous_kernel],
            core::slice::from_ref(&resolved),
            &no_packed(),
            NumericPolicy::default(),
            EmitDevice::default(),
        )
        .expect("reuse succeeds");
        let fresh = emit_cuda(&resolved, &no_packed()).expect("emit succeeds");
        assert_eq!(reused, 0);
        assert_eq!(kernels, vec![fresh]);
    }

    fn matmul_reduce_op(rows: u32, cols: u32, reduce_op: ScalarOp) -> BoundOp {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(rows), Extent::Static(cols)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
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
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(rows), Extent::Static(k)],
                name: None,
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::FLOAT32,
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
                dtype: DType::FLOAT32,
                body: ScalarOp::Multiply,
                operands,
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
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
        assert!(!flag_off.source.contains("blockdot_"));
        assert!(flag_on.source.contains("blockdot_"), "the flag-on q4_0 matvec reads the layout-driven int8 dot");
        assert_ne!(flag_on.entry, flag_off.entry);
    }

    #[test]
    fn entry_name_is_unchanged_with_the_flag_off_and_differs_with_it_on() {
        let (bound, weight) = q4_0_matvec_op(8, 64, true);
        let packed = q4_0_packed(weight);
        let quantized = operand_codecs(&bound, &packed);
        let flag_off = entry_name(&bound, &packed, &quantized, NumericPolicy::default(), EmitDevice::default());
        let flag_on = entry_name(
            &bound,
            &packed,
            &quantized,
            NumericPolicy::default().with_activation_quantization(true),
            EmitDevice::default(),
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
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
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
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(vocab), Extent::Static(dim)],
                name: None,
            },
        );
        let ids = append(
            &mut program,
            Op::Input {
                dtype: DType::INT32,
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
                dtype: DType::FLOAT32,
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
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(block_elements as u32)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
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
                dtype: DType::FLOAT32,
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
                dtype: DType::INT32,
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
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(seq), Extent::Static(width)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
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
                dtype: DType::FLOAT32,
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
                dtype: DType::FLOAT16,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT16,
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
                dtype: DType::FLOAT64,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT64,
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
        let error = render_reduce(&bound, "entry", &[None], NumericPolicy::bit_exact(), EmitDevice::default())
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

        let source = render_reduce(&bound, "entry", &[None], NumericPolicy::llama_relaxed(), EmitDevice::default())
            .expect("broadcast emits");
        assert!(source.contains("broadcast_out_strides[2]"));
        assert!(source.contains("broadcast_full_coord[1]"));
        assert!(source.contains("__shfl_sync(0xffffffffu, accumulator, 0)"));

        let kernel = emit_cuda_with_policy(&bound, &no_packed(), NumericPolicy::llama_relaxed())
            .expect("broadcast kernel emits");
        assert_eq!(kernel.grid.threads, 4 * WARP_SIZE);
        let uniforms = pack_cuda_uniforms(&bound).expect("broadcast uniforms pack");
        assert_eq!(
            i64::from_ne_bytes(uniforms[..8].try_into().expect("i64 output total")),
            32
        );
    }

    /// A broadcast-epilogue reduce over one row (the decode-shaped RMSNorm) is
    /// latency-bound on a single warp, so it takes a whole block and combines
    /// the warps' partials through shared memory.
    #[test]
    fn a_single_row_broadcast_reduce_takes_a_whole_block() {
        let mut bound = matmul_reduce_op(1, 1024, ScalarOp::Add);
        let BoundOpKind::Reduce {
            epilogue_broadcast_axes,
            ..
        } = &mut bound.kind
        else {
            panic!("matmul_reduce_op always builds a Keep::Reduce fold")
        };
        epilogue_broadcast_axes.push(1);

        let kernel = emit_cuda_with_policy(&bound, &no_packed(), NumericPolicy::llama_relaxed())
            .expect("broadcast kernel emits");

        assert_eq!(kernel.grid.block_width, Some(1024 / BLOCK_REDUCE_ELEMENTS_PER_THREAD));
        assert_eq!(kernel.grid.threads, 1024 / BLOCK_REDUCE_ELEMENTS_PER_THREAD);
        assert!(kernel.source.contains("__shared__ float warp_partials[32];"));
        assert!(kernel.source.contains("long output_index = (long)blockIdx.x;"));
    }

    #[test]
    fn a_multi_row_broadcast_reduce_keeps_one_warp_per_row() {
        let mut bound = matmul_reduce_op(5, 1024, ScalarOp::Add);
        let BoundOpKind::Reduce {
            epilogue_broadcast_axes,
            ..
        } = &mut bound.kind
        else {
            panic!("matmul_reduce_op always builds a Keep::Reduce fold")
        };
        epilogue_broadcast_axes.push(1);

        let kernel = emit_cuda_with_policy(&bound, &no_packed(), NumericPolicy::llama_relaxed())
            .expect("broadcast kernel emits");

        assert_eq!(kernel.grid.block_width, Some(WARP_SIZE));
        assert_eq!(kernel.grid.threads, 5 * WARP_SIZE);
    }

    #[test]
    fn a_rank_one_packed_reduce_hoists_row_bases_and_decodes_unrolled_chunks() {
        for (codec, name) in [
            (Codec::Q8_0, "q8_0"),
            (Codec::Q4K, "q4k"),
            (Codec::Q1_0, "q1_0"),
            (Codec::Q2_0, "q2_0"),
            (Codec::Pq2_0, "q2_0"),
        ] {
            let bound = matmul_reduce_op(4, 256, ScalarOp::Add);
            let mut packed = BTreeMap::new();
            packed.insert(bound.operands()[0].0, codec);

            let kernel = emit_cuda_with_policy(&bound, &packed, NumericPolicy::llama_relaxed())
                .expect("emit succeeds");

            assert!(kernel.source.contains("const long row0 ="), "{name} row base");
            assert!(
                kernel.source.contains(&format!("{name}_chunk8(in0 + (off0_3 /")),
                "{name} unrolled chunk decode"
            );
            assert!(
                !kernel.source.contains("remaining_r"),
                "{name} rank-1 reduction needs no coordinate decomposition"
            );
        }
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
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(rows), Extent::Static(cols)],
                name: None,
            },
        );
        let reduce = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
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
                dtype: DType::FLOAT32,
                shape: vec![Extent::Static(rows)],
                name: None,
            },
        );
        let consumer = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
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
            entry_name(&fused, &packed, &[], policy, EmitDevice::default()),
            entry_name(&bare, &packed, &[], policy, EmitDevice::default()),
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
