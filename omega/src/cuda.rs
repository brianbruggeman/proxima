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
//! # No CUDA toolchain on this host
//!
//! This module is pure string generation over an already-bound [`BoundOp`] —
//! it never links `libcuda`, `nvrtc`, or any NVIDIA driver, and has zero
//! dependencies (`cuda = []` in `Cargo.toml`). That is what lets it build and
//! its tests run on a machine with no NVIDIA GPU, including the macOS CI host
//! that gates this crate. The emitted CUDA C source is never compiled here
//! (`nvcc`/`nvrtc` are not available), so this crate proves emission
//! STRUCTURE only — that the right buffers, the right unpack calls, and the
//! right control flow appear in the text — never numeric execution parity
//! with a real device. A driver half (`cudarc`/`cust`, actual `cuModuleLoad`
//! and `cuLaunchKernel`) is future work behind its own std-gated,
//! dependency-bearing feature; `Backend::Cuda` stays `NotImplemented` until
//! then (see `crate::backend`'s own doc).
//!
//! # v1 scope
//!
//! - **Elementwise**, **`Keep::Reduce`** (serial per-output-element fold, plus
//!   a warp-shuffle cooperative fold for the associative/commutative
//!   `ScalarOp`s — see [`reduce_is_cooperative`]), and **`Keep::Scan`**
//!   (one thread, serial over every outer line — matching
//!   `proxima_tensor::cpu::run_scan`'s persistent accumulator, the same shape
//!   `crate::wgsl::render_scan` takes).
//! - **Gather.** A fault buffer records an out-of-range fetched index (the
//!   `atomicOr`-based flag [`push_gather_fetch`] emits) the same way
//!   [`crate::msl::push_gather_fetch`]'s `atomic_fetch_max` does, so
//!   `omega::execute`'s driver can turn a nonzero slot into the same
//!   `TensorError::GatherIndexOutOfRange` the CPU oracle reports.
//! - **Packed operands**: `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`/`Q4_0` plus
//!   `Float16`/`BFloat16`, ported from [`crate::msl`]'s own unpack constants
//!   — see [`Q4K_UNPACK_CUDA`] and its siblings.
//! - **`f32`/`f16`** ([`type_token`]) — `f16` needs `<cuda_fp16.h>`'s
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

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::{BoundOp, BoundOpKind, ComposedBody, DType, Keep, NodeId, ReduceInit, ScalarOp, StepArg};

use crate::error::EmitError;
use crate::msl::{Binding, PackedCodec, PackedOperands};

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
}

/// How many threads a driver must launch, and — for a cooperative reduce —
/// the block width the launch is pinned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaGridSpec {
    pub threads: u64,
    /// `Some(WARP_SIZE)` for a warp-shuffle cooperative reduce (see
    /// [`reduce_is_cooperative`]): the driver must launch blocks exactly
    /// this wide so every warp boundary lands on an output-element
    /// boundary. `None` for every other kernel.
    pub block_width: Option<u64>,
}

/// Emits a CUDA C kernel from a bound [`BoundOp`] — the CUDA counterpart of
/// [`crate::msl::emit`]. See the module doc for exactly which op shapes v1
/// covers.
///
/// # Errors
/// [`EmitError::UnsupportedDType`] for anything but `Float32`/`Float16`,
/// [`EmitError::CudaUnsupportedOpKind`] for `Iota`/`Constant`,
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
/// append(
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
/// let bound_ops = proxima_tensor::bind(&program, &shapes, &[])?;
/// let packed_operands = omega::PackedOperands::new();
/// let kernel = omega::emit_cuda(&bound_ops[0], &packed_operands)?;
/// assert!(kernel.source.contains("extern \"C\" __global__"));
/// assert!(kernel.source.contains("tanhf("));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn emit_cuda(resolved: &BoundOp, packed_operands: &PackedOperands) -> Result<CudaKernel, EmitError> {
    validate(resolved)?;
    let entry = entry_name(resolved);
    let quantized = operand_codecs(resolved, packed_operands);
    let source = match &resolved.kind {
        BoundOpKind::Elementwise { .. } => render_elementwise(resolved, &entry, &quantized)?,
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => render_reduce(resolved, &entry, &quantized)?,
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => render_scan(resolved, &entry, &quantized)?,
        BoundOpKind::Iota => {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: "iota",
            });
        }
        BoundOpKind::Constant { .. } => {
            return Err(EmitError::CudaUnsupportedOpKind {
                node: resolved.node,
                kind: "constant",
            });
        }
    };
    Ok(CudaKernel {
        source,
        entry,
        bindings: bindings(resolved),
        grid: CudaGridSpec {
            threads: grid_threads(resolved),
            block_width: reduce_is_cooperative(resolved).then_some(WARP_SIZE),
        },
    })
}

fn operand_codecs(resolved: &BoundOp, packed_operands: &PackedOperands) -> Vec<Option<PackedCodec>> {
    resolved
        .operands()
        .iter()
        .map(|(node, _, _)| packed_operands.get(node).copied())
        .collect()
}

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
        reduce_op, keep, ..
    } = &resolved.kind
    {
        if matches!(reduce_op, ScalarOp::Select) {
            return Err(EmitError::ReductionBodyIsSelect { node: resolved.node });
        }
        if *keep == Keep::Scan && resolved.extents.is_empty() {
            return Err(EmitError::EmptyScan { node: resolved.node });
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
    let mut bindings: Vec<Binding> = resolved
        .operands()
        .iter()
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

fn grid_threads(resolved: &BoundOp) -> u64 {
    match &resolved.kind {
        BoundOpKind::Elementwise { .. } => resolved.extents.iter().product(),
        BoundOpKind::Reduce {
            output_axes,
            keep: Keep::Reduce,
            ..
        } => {
            let output_total: u64 = output_axes.iter().map(|dim| resolved.extents[*dim as usize]).product();
            if reduce_is_cooperative(resolved) {
                output_total * WARP_SIZE
            } else {
                output_total
            }
        }
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => {
            let rank = resolved.extents.len();
            resolved.extents[..rank.saturating_sub(1)].iter().product()
        }
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => resolved.extents.iter().product(),
    }
}

/// Whether `resolved` is a `Keep::Reduce` fold whose `reduce_op` is
/// associative and commutative with no gathered operand — the same set
/// `crate::msl::reduce_is_cooperative` picks for a SIMD-group cooperative
/// fold, ported unchanged: `Subtract`/`Divide` are not associative, so
/// reordering their combination across lanes is wrong, not merely imprecise.
fn reduce_is_cooperative(resolved: &BoundOp) -> bool {
    match &resolved.kind {
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            reduce_op,
            ..
        } => gather_count(resolved) == 0 && is_cooperative_reduce_op(*reduce_op),
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
fn shuffle_combine_expr(op: ScalarOp, accumulator: &str, shuffled: &str) -> String {
    match op {
        ScalarOp::Add => format!("{accumulator} + {shuffled}"),
        ScalarOp::Multiply => format!("{accumulator} * {shuffled}"),
        ScalarOp::Maximum => format!("fmaxf({accumulator}, {shuffled})"),
        ScalarOp::Minimum => format!("fminf({accumulator}, {shuffled})"),
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
        | ScalarOp::Select => unreachable!("shuffle_combine_expr is only called for a cooperative reduce_op"),
    }
}

fn op_token(op: ScalarOp) -> &'static str {
    match op {
        ScalarOp::Identity => "identity",
        ScalarOp::Add => "add",
        ScalarOp::Subtract => "subtract",
        ScalarOp::Multiply => "multiply",
        ScalarOp::Divide => "divide",
        ScalarOp::Maximum => "maximum",
        ScalarOp::Minimum => "minimum",
        ScalarOp::Negate => "negate",
        ScalarOp::Reciprocal => "reciprocal",
        ScalarOp::Exponential => "exponential",
        ScalarOp::Logarithm => "logarithm",
        ScalarOp::SquareRoot => "square_root",
        ScalarOp::Tanh => "tanh",
        ScalarOp::Erf => "erf",
        ScalarOp::Greater => "greater",
        ScalarOp::Equal => "equal",
        ScalarOp::Select => "select",
    }
}

fn init_token(init: ReduceInit) -> &'static str {
    match init {
        ReduceInit::Zero => "zero",
        ReduceInit::One => "one",
        ReduceInit::NegativeInfinity => "negative_infinity",
        ReduceInit::PositiveInfinity => "positive_infinity",
        ReduceInit::FirstElement => "first_element",
    }
}

fn keep_token(keep: Keep) -> &'static str {
    match keep {
        Keep::Reduce => "reduce",
        Keep::Scan => "scan",
    }
}

fn is_leaf(body: &ComposedBody) -> bool {
    body.steps.len() == 1
        && body.steps[0]
            .args
            .iter()
            .enumerate()
            .all(|(index, arg)| matches!(arg, StepArg::Operand(operand) if *operand as usize == index))
}

fn body_fingerprint(body: &ComposedBody) -> String {
    body.steps
        .iter()
        .map(|step| {
            let mut token = String::from(op_token(step.op));
            for arg in &step.args {
                match arg {
                    StepArg::Operand(index) => token.push_str(&format!("_o{index}")),
                    StepArg::Step(index) => token.push_str(&format!("_s{index}")),
                }
            }
            token
        })
        .collect::<Vec<_>>()
        .join("__")
}

fn body_token(body: &ComposedBody) -> String {
    if is_leaf(body) {
        op_token(body.steps[0].op).into()
    } else {
        format!("fused_{}", body_fingerprint(body))
    }
}

/// A structural fingerprint over rank/operand-count/body/(for a reduce)
/// reduce-op/init/output-rank/gather-shape — the CUDA counterpart of
/// `crate::msl::entry_name`.
fn entry_name(resolved: &BoundOp) -> String {
    let rank = resolved.extents.len();
    let operand_count = resolved.operands().len();
    let base = match &resolved.kind {
        BoundOpKind::Elementwise { .. } => {
            let body = body_token(resolved.element_body());
            format!("omega_cuda_elementwise_r{rank}_n{operand_count}_{body}")
        }
        BoundOpKind::Reduce {
            reduce_op,
            init,
            keep,
            output_axes,
            ..
        } => {
            let body = body_token(resolved.element_body());
            let kind = keep_token(*keep);
            let reduce_body = op_token(*reduce_op);
            let init = init_token(*init);
            let output_rank = output_axes.len();
            format!("omega_cuda_{kind}_r{rank}_o{output_rank}_n{operand_count}_{body}_{reduce_body}_{init}")
        }
        BoundOpKind::Iota => format!("omega_cuda_iota_r{rank}"),
        BoundOpKind::Constant { value } => {
            format!("omega_cuda_constant_r{rank}_v{:08x}", value.to_bits())
        }
    };
    let gather_bits: String = resolved
        .operands()
        .iter()
        .map(|(_, _, gather)| if gather.is_some() { '1' } else { '0' })
        .collect();
    if gather_bits.contains('1') {
        format!("{base}_g{gather_bits}")
    } else {
        base
    }
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
/// `crate::msl::fold_init_tokens`. `NegativeInfinity`/`PositiveInfinity` use
/// `-INFINITY`/`INFINITY`, which `<math.h>`'s CUDA-provided
/// `<cuda_runtime.h>` guarantees as IEEE-754 infinities, unlike WGSL's base
/// spec.
fn fold_init_tokens(init: ReduceInit) -> (&'static str, &'static str) {
    match init {
        ReduceInit::Zero => ("0.0f", "true"),
        ReduceInit::One => ("1.0f", "true"),
        ReduceInit::NegativeInfinity => ("-INFINITY", "true"),
        ReduceInit::PositiveInfinity => ("INFINITY", "true"),
        ReduceInit::FirstElement => ("0.0f", "false"),
    }
}

/// The algebraic identity `op` folds against without changing a value —
/// ports `crate::msl::cooperative_identity_token`. Every warp lane but lane
/// 0 seeds its private accumulator with this rather than the `BoundOp`'s own
/// `ReduceInit`, so folding it into the final `__shfl_down_sync` combine can
/// never perturb the result.
fn cooperative_identity_token(op: ScalarOp) -> &'static str {
    match op {
        ScalarOp::Add => "0.0f",
        ScalarOp::Multiply => "1.0f",
        ScalarOp::Maximum => "-INFINITY",
        ScalarOp::Minimum => "INFINITY",
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
        | ScalarOp::Select => unreachable!("cooperative_identity_token is only called for a cooperative reduce_op"),
    }
}

fn push_body_steps(source: &mut String, body: &ComposedBody, indent: &str, element_type: &str) -> String {
    for (index, step) in body.steps.iter().enumerate() {
        let args: Vec<String> = step
            .args
            .iter()
            .map(|arg| match arg {
                StepArg::Operand(operand_index) => format!("scratch[{operand_index}]"),
                StepArg::Step(step_index) => format!("step{step_index}"),
            })
            .collect();
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let expr = scalar_op_expr(step.op, &arg_refs);
        source.push_str(&format!("{indent}{element_type} step{index} = {expr};\n"));
    }
    format!("step{}", body.steps.len().saturating_sub(1))
}

/// `<math.h>` (via `<cuda_runtime.h>`) has no `erf` variant that matches the
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

fn preamble(source: &mut String, needs_half: bool) {
    source.push_str("#include <cuda_runtime.h>\n");
    if needs_half {
        source.push_str("#include <cuda_fp16.h>\n");
    }
    source.push('\n');
    source.push_str(PROXIMA_ERF_FN_CUDA);
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

fn kernel_signature(
    source: &mut String,
    quantized: &[Option<PackedCodec>],
    gather_count: usize,
    entry: &str,
    element_type: &str,
) {
    source.push_str(&format!("extern \"C\" __global__ void {entry}(\n"));
    for (index, &codec) in quantized.iter().enumerate() {
        let binding_type = match codec {
            None => element_type,
            Some(PackedCodec::Float16) => "__half",
            Some(_) => "unsigned char",
        };
        source.push_str(&format!("    const {binding_type}* __restrict__ in{index},\n"));
    }
    for slot in 0..gather_count {
        source.push_str(&format!("    const float* __restrict__ gather_idx{slot},\n"));
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
    source.push_str("    long gid = (long)(blockIdx.x * blockDim.x + threadIdx.x);\n");
}

fn push_gather_uniform_fields(source: &mut String, gather_count: usize, rank_len: usize) {
    if gather_count == 0 {
        return;
    }
    source.push_str(&format!("    long gather_index_base[{gather_count}];\n"));
    source.push_str(&format!("    long gather_index_strides[{gather_count}][{rank_len}];\n"));
    source.push_str(&format!("    long gather_element_stride[{gather_count}];\n"));
    source.push_str(&format!("    long gather_extent[{gather_count}];\n"));
}

/// Emits the out-of-range check for one just-fetched, not-yet-clamped
/// `fetched{operand_index}`: when it falls outside
/// `[0, u.gather_extent[gather_slot])`, sets that gathered operand's `fault`
/// flag via `atomicOr` — unlike `crate::msl::push_gather_fault_check`'s
/// `atomic_fetch_max` (which records the offending value), this records only
/// that a fault occurred: CUDA's `atomicOr` on `unsigned int` composes
/// losslessly across concurrent threads (the OR of any set of nonzero flags
/// is still nonzero) without needing a value-carrying atomic, and the driver
/// only needs to know a fault occurred to build a
/// `TensorError::GatherIndexOutOfRange` the way it already does for Metal.
fn push_gather_fault_check(source: &mut String, operand_index: usize, gather_slot: usize, indent: &str) {
    source.push_str(&format!(
        "{indent}if (fetched{operand_index} < 0 || fetched{operand_index} >= u.gather_extent[{gather_slot}]) {{\n"
    ));
    source.push_str(&format!(
        "{indent}    atomicOr(&fault[{gather_slot}], 1u);\n"
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

/// How operand `index` is READ, given the element-offset expression the
/// caller already computed — the CUDA counterpart of
/// `crate::msl::operand_read`, same per-element-only scope (no row-blocked
/// header amortization; see the module doc).
fn operand_read(index: usize, offset: &str, codec: Option<PackedCodec>) -> String {
    match codec {
        None => format!("in{index}[{offset}]"),
        Some(PackedCodec::Q4K) => format!(
            "q4k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q4K_BLOCK_BYTES}, (unsigned int)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        Some(PackedCodec::Q5K) => format!(
            "q5k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q5K_BLOCK_BYTES}, (unsigned int)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        Some(PackedCodec::Q6K) => format!(
            "q6k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q6K_BLOCK_BYTES}, (unsigned int)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        Some(PackedCodec::Q8_0) => format!(
            "q8_0_element(in{index} + ({offset} / {Q8_0_BLOCK_ELEMENTS}) * {Q8_0_BLOCK_BYTES}, (unsigned int)({offset} % {Q8_0_BLOCK_ELEMENTS}))"
        ),
        Some(PackedCodec::Q4_0) => format!(
            "q4_0_element(in{index} + ({offset} / {Q4_0_BLOCK_ELEMENTS}) * {Q4_0_BLOCK_BYTES}, (unsigned int)({offset} % {Q4_0_BLOCK_ELEMENTS}))"
        ),
        // `__half` converts implicitly to `float` in CUDA C++, the same
        // "already a valid narrow-float buffer" shape MSL's `half` takes.
        Some(PackedCodec::Float16) => format!("in{index}[{offset}]"),
        Some(PackedCodec::BFloat16) => format!(
            "bf16_element(in{index} + ({offset} / {BFLOAT16_BLOCK_ELEMENTS}) * {BFLOAT16_BLOCK_BYTES}, (unsigned int)({offset} % {BFLOAT16_BLOCK_ELEMENTS}))"
        ),
    }
}

fn render_elementwise(resolved: &BoundOp, entry: &str, quantized: &[Option<PackedCodec>]) -> Result<String, EmitError> {
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;

    let mut source = String::new();
    preamble(&mut source, element_type == "__half");

    source.push_str("struct Uniforms {\n");
    source.push_str("    long total_elements;\n");
    source.push_str(&format!("    long extents[{rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!("    long operand_strides[{operand_count}][{rank_len}];\n"));
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    kernel_signature(&mut source, quantized, gather_count, entry, element_type);
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
            push_gather_fetch(&mut source, index, *slot, rank, "coord", &format!("off{index}"));
        }
    }

    source.push_str(&format!("    {element_type} scratch[{}];\n", operand_count.max(1)));
    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!(
            "    scratch[{index}] = {};\n",
            operand_read(index, &format!("off{index}"), codec)
        ));
    }

    let result = push_body_steps(&mut source, resolved.element_body(), "    ", element_type);
    source.push_str(&format!("    out[gid] = {result};\n"));
    source.push_str("}\n");
    Ok(source)
}

#[allow(clippy::too_many_arguments)]
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
    quantized: &[Option<PackedCodec>],
    element_type: &str,
) {
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
        for (index, dim) in output_axes.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
    }

    let (init_expr, seeded_init) = fold_init_tokens(init);
    source.push_str(&format!("    {element_type} accumulator = {init_expr};\n"));
    source.push_str(&format!("    bool seeded = {seeded_init};\n"));

    source.push_str("    for (long r = 0; r < u.reduction_total; r++) {\n");
    if reduce_rank > 0 {
        source.push_str(&format!("        long reduction_coord[{reduce_rank_len}];\n"));
        source.push_str("        long remaining_r = r;\n");
        for index in (0..reduce_rank).rev() {
            source.push_str(&format!(
                "        reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
                 remaining_r /= u.reduction_extents[{index}];\n"
            ));
        }
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!("        full_coord[{dim}] = reduction_coord[{index}];\n"));
        }
    }

    for (index, gather_slot) in gather_slots.iter().enumerate() {
        source.push_str(&format!("        long off{index} = u.operand_base[{index}];\n"));
        for dim in 0..rank {
            source.push_str(&format!(
                "        off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slot {
            push_gather_fetch(source, index, *slot, rank, "full_coord", &format!("off{index}"));
        }
    }
    source.push_str(&format!("        {element_type} scratch[{}];\n", operand_count.max(1)));
    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!(
            "        scratch[{index}] = {};\n",
            operand_read(index, &format!("off{index}"), codec)
        ));
    }
    let value_expr = push_body_steps(source, resolved.element_body(), "        ", element_type);
    source.push_str(&format!("        {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
    source.push_str(&format!("        accumulator = seeded ? {combine_expr} : value;\n"));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    source.push_str("    long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!("    out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"));
    }
    source.push_str("    out[out_offset] = accumulator;\n");
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
    quantized: &[Option<PackedCodec>],
    element_type: &str,
) {
    let rank_len = rank.max(1);
    let output_rank = output_axes.len();
    let output_rank_len = output_rank.max(1);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);
    let operand_count = resolved.operands().len();

    source.push_str(&format!("    long output_index = gid / {WARP_SIZE};\n"));
    source.push_str("    if (output_index >= u.output_total) { return; }\n");
    source.push_str(&format!("    unsigned int lane = (unsigned int)(gid % {WARP_SIZE});\n"));

    source.push_str(&format!("    long full_coord[{rank_len}];\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }

    if output_rank > 0 {
        source.push_str(&format!("    long output_coord[{output_rank_len}];\n"));
        source.push_str("    long remaining = output_index;\n");
        for index in (0..output_rank).rev() {
            source.push_str(&format!(
                "    output_coord[{index}] = remaining % u.output_extents[{index}]; \
                 remaining /= u.output_extents[{index}];\n"
            ));
        }
        for (index, dim) in output_axes.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
    }

    let (init_expr, seeded_init) = fold_init_tokens(init);
    let identity = cooperative_identity_token(reduce_op);
    source.push_str(&format!("    {element_type} accumulator;\n"));
    source.push_str("    bool seeded;\n");
    source.push_str("    if (lane == 0u) {\n");
    source.push_str(&format!("        accumulator = {init_expr};\n"));
    source.push_str(&format!("        seeded = {seeded_init};\n"));
    source.push_str("    } else {\n");
    source.push_str(&format!("        accumulator = {identity};\n"));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    source.push_str(&format!(
        "    for (long r = (long)lane; r < u.reduction_total; r += {WARP_SIZE}) {{\n"
    ));
    if reduce_rank > 0 {
        source.push_str(&format!("        long reduction_coord[{reduce_rank_len}];\n"));
        source.push_str("        long remaining_r = r;\n");
        for index in (0..reduce_rank).rev() {
            source.push_str(&format!(
                "        reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
                 remaining_r /= u.reduction_extents[{index}];\n"
            ));
        }
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!("        full_coord[{dim}] = reduction_coord[{index}];\n"));
        }
    }

    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!("        long off{index} = u.operand_base[{index}];\n"));
        for dim in 0..rank {
            source.push_str(&format!(
                "        off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        let _ = codec;
    }
    source.push_str(&format!("        {element_type} scratch[{}];\n", operand_count.max(1)));
    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!(
            "        scratch[{index}] = {};\n",
            operand_read(index, &format!("off{index}"), codec)
        ));
    }
    let value_expr = push_body_steps(source, resolved.element_body(), "        ", element_type);
    source.push_str(&format!("        {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
    source.push_str(&format!("        accumulator = seeded ? {combine_expr} : value;\n"));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    source.push_str("    #pragma unroll\n");
    source.push_str(&format!("    for (int shift = {}; shift > 0; shift >>= 1) {{\n", WARP_SIZE / 2));
    source.push_str(&format!(
        "        {element_type} shuffled = __shfl_down_sync(0xffffffffu, accumulator, shift);\n"
    ));
    let shuffle_expr = shuffle_combine_expr(reduce_op, "accumulator", "shuffled");
    source.push_str(&format!("        accumulator = {shuffle_expr};\n"));
    source.push_str("    }\n");

    source.push_str("    if (lane != 0u) { return; }\n");
    source.push_str("    long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!("    out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"));
    }
    source.push_str("    out[out_offset] = accumulator;\n");
}

fn render_reduce(resolved: &BoundOp, entry: &str, quantized: &[Option<PackedCodec>]) -> Result<String, EmitError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        ..
    } = &resolved.kind
    else {
        unreachable!("render_reduce is only called for a Keep::Reduce fold")
    };
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let output_rank = output_axes.len();
    let output_rank_len = output_rank.max(1);
    let reduce_dims = reduction_dims(resolved, output_axes);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;

    let mut source = String::new();
    preamble(&mut source, element_type == "__half");

    source.push_str("struct Uniforms {\n");
    source.push_str("    long output_total;\n");
    source.push_str("    long reduction_total;\n");
    source.push_str(&format!("    long output_extents[{output_rank_len}];\n"));
    source.push_str(&format!("    long reduction_extents[{reduce_rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!("    long operand_strides[{operand_count}][{rank_len}];\n"));
    source.push_str("    long out_base;\n");
    source.push_str(&format!("    long out_strides[{rank_len}];\n"));
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    kernel_signature(&mut source, quantized, gather_count, entry, element_type);

    if reduce_is_cooperative(resolved) {
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
        );
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
        );
    }
    source.push_str("}\n");
    Ok(source)
}

fn render_scan(resolved: &BoundOp, entry: &str, quantized: &[Option<PackedCodec>]) -> Result<String, EmitError> {
    let BoundOpKind::Reduce {
        reduce_op, init, ..
    } = &resolved.kind
    else {
        unreachable!("render_scan is only called for a Keep::Scan fold")
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
    preamble(&mut source, element_type == "__half");

    source.push_str("struct Uniforms {\n");
    source.push_str("    long outer_total;\n");
    source.push_str("    long inner_len;\n");
    source.push_str(&format!("    long outer_extents[{outer_rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!("    long operand_strides[{operand_count}][{rank_len}];\n"));
    source.push_str("    long out_base;\n");
    source.push_str(&format!("    long out_strides[{rank_len}];\n"));
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    kernel_signature(&mut source, quantized, gather_count, entry, element_type);
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
        source.push_str(&format!("        long running{index} = u.operand_base[{index}];\n"));
        for dim in 0..outer_rank {
            source.push_str(&format!(
                "        running{index} += outer_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slot {
            source.push_str(&format!("        long gather_running{index} = u.gather_index_base[{slot}];\n"));
            for dim in 0..outer_rank {
                source.push_str(&format!(
                    "        gather_running{index} += outer_coord[{dim}] * u.gather_index_strides[{slot}][{dim}];\n"
                ));
            }
        }
    }
    source.push_str("        long out_running = u.out_base;\n");
    for dim in 0..outer_rank {
        source.push_str(&format!("        out_running += outer_coord[{dim}] * u.out_strides[{dim}];\n"));
    }

    source.push_str("        for (long step = 0; step < u.inner_len; step++) {\n");
    source.push_str(&format!("            {element_type} scratch[{}];\n", operand_count.max(1)));
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
                operand_read(index, &format!("read_off{index}"), quantized[index])
            ));
            source.push_str(&format!(
                "            gather_running{index} += u.gather_index_strides[{slot}][{last_dim}];\n"
            ));
        } else {
            source.push_str(&format!(
                "            scratch[{index}] = {};\n",
                operand_read(index, &format!("running{index}"), quantized[index])
            ));
        }
        source.push_str(&format!(
            "            running{index} += u.operand_strides[{index}][{last_dim}];\n"
        ));
    }
    let value_expr = push_body_steps(&mut source, resolved.element_body(), "            ", element_type);
    source.push_str(&format!("            {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(*reduce_op, &["accumulator", "value"]);
    source.push_str(&format!("            accumulator = seeded ? {combine_expr} : value;\n"));
    source.push_str("            seeded = true;\n");
    source.push_str("            out[out_running] = accumulator;\n");
    source.push_str(&format!("            out_running += u.out_strides[{last_dim}];\n"));
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

pub const Q4K_BLOCK_BYTES: usize = 144;
pub const Q4K_BLOCK_ELEMENTS: usize = 256;

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
";

pub const Q6K_BLOCK_BYTES: usize = 210;

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

pub const Q5K_BLOCK_BYTES: usize = 176;

/// CUDA C source for unpacking one element of a `Q8_0` block — ports
/// `crate::msl::Q8_0_UNPACK_MSL`'s `q8_0_element` exactly: a flat 32-element
/// block, one `f16` scale, no sub-block structure.
pub const Q8_0_UNPACK_CUDA: &str = "\
__device__ __forceinline__ float q8_0_element(const unsigned char *block, unsigned int index) {
    unsigned short d_bits = (unsigned short)((unsigned int)block[0] | ((unsigned int)block[1] << 8));
    float d = __half2float(__ushort_as_half(d_bits));
    signed char level = (signed char)block[2u + index];
    return (float)level * d;
}
";

pub const Q8_0_BLOCK_BYTES: usize = 34;
pub const Q8_0_BLOCK_ELEMENTS: usize = 32;

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
";

pub const Q4_0_BLOCK_BYTES: usize = 18;
pub const Q4_0_BLOCK_ELEMENTS: usize = 32;

pub const FLOAT16_BLOCK_BYTES: usize = 2;
pub const FLOAT16_BLOCK_ELEMENTS: usize = 1;
pub const BFLOAT16_BLOCK_BYTES: usize = 2;
pub const BFLOAT16_BLOCK_ELEMENTS: usize = 1;

/// Widens one `bfloat16` element to `float` by shifting it into the high 16
/// bits of a 32-bit word and reinterpreting — ports
/// `crate::msl::BF16_UNPACK_MSL`'s `bf16_element` exactly. CUDA's own
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
        AxisTerm, DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind, infer, map,
    };

    use super::*;

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
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
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
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
        bound.into_iter().next().expect("one bound op")
    }

    #[test]
    fn associative_reduce_emits_warp_shuffle_cooperative_kernel() {
        let bound = matmul_reduce_op(4, 4096, ScalarOp::Add);
        let kernel = emit_cuda(&bound, &no_packed()).expect("emit succeeds");
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
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
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
        bind(&program, &shapes, &[])
            .expect("embedding lookup lowers")
            .into_iter()
            .next()
            .expect("one bound emitted")
    }

    #[test]
    fn gather_emits_fault_buffer_and_atomic_or_check() {
        let bound = embedding_lookup_op(50_000, 8, 4);
        let kernel = emit_cuda(&bound, &no_packed()).expect("emit succeeds");
        assert!(kernel.source.contains("atomicOr(&fault[0], 1u);"));
        assert_eq!(
            kernel.bindings,
            vec![
                Binding::Input(bound.operands()[0].0),
                Binding::Indices(bound.operands()[0].2.as_ref().expect("operand 0 gathers").indices),
                Binding::Output(bound.node),
                Binding::Uniforms,
                Binding::Fault,
            ]
        );
    }

    fn packed_elementwise_op(codec: PackedCodec, block_bytes: usize, block_elements: usize) -> (BoundOp, PackedOperands) {
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
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
        let bound = bound.into_iter().next().expect("one bound op");
        let mut packed = BTreeMap::new();
        packed.insert(bound.operands()[0].0, codec);
        let _ = block_bytes;
        (bound, packed)
    }

    #[test]
    fn q4k_operand_emits_q4k_element_read() {
        let (bound, packed) = packed_elementwise_op(PackedCodec::Q4K, Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS);
        let kernel = emit_cuda(&bound, &packed).expect("emit succeeds");
        assert!(kernel.source.contains("q4k_element(in0"));
        assert!(kernel.source.contains(&format!("* {Q4K_BLOCK_BYTES}")));
    }

    #[test]
    fn q5k_operand_emits_q5k_element_read() {
        let (bound, packed) = packed_elementwise_op(PackedCodec::Q5K, Q5K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS);
        let kernel = emit_cuda(&bound, &packed).expect("emit succeeds");
        assert!(kernel.source.contains("q5k_element(in0"));
    }

    #[test]
    fn q6k_operand_emits_q6k_element_read() {
        let (bound, packed) = packed_elementwise_op(PackedCodec::Q6K, Q6K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS);
        let kernel = emit_cuda(&bound, &packed).expect("emit succeeds");
        assert!(kernel.source.contains("q6k_element(in0"));
    }

    #[test]
    fn q8_0_operand_emits_q8_0_element_read() {
        let (bound, packed) = packed_elementwise_op(PackedCodec::Q8_0, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS);
        let kernel = emit_cuda(&bound, &packed).expect("emit succeeds");
        assert!(kernel.source.contains("q8_0_element(in0"));
    }

    #[test]
    fn q4_0_operand_emits_q4_0_element_read() {
        let (bound, packed) = packed_elementwise_op(PackedCodec::Q4_0, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS);
        let kernel = emit_cuda(&bound, &packed).expect("emit succeeds");
        assert!(kernel.source.contains("q4_0_element(in0"));
    }

    #[test]
    fn bfloat16_operand_emits_bf16_element_read() {
        let (bound, packed) = packed_elementwise_op(PackedCodec::BFloat16, BFLOAT16_BLOCK_BYTES, BFLOAT16_BLOCK_ELEMENTS);
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
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
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
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
        let bound = bound.into_iter().next().expect("one bound op");
        let error = emit_cuda(&bound, &no_packed()).expect_err("f64 is rejected");
        assert!(matches!(error, EmitError::UnsupportedDType { .. }));
    }
}
