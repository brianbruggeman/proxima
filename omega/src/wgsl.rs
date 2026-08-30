//! WGSL kernel emission — the portable counterpart of [`crate::msl::emit`],
//! one abstraction layer over: same [`BoundOp`] descriptor
//! (`proxima_tensor::bind`), same "runtime uniforms, not baked constants"
//! stance, same per-[`Keep`] execution model. [`emit_wgsl`] differs only in
//! target ISA (WGSL text instead of MSL text) and in scope — see the module
//! doc below for exactly what v1 covers and what it does not.
//!
//! # v1 scope
//!
//! - **Elementwise**: the sixteen direct [`ScalarOp`]s (`Identity`, `Add`,
//!   `Subtract`, `Multiply`, `Divide`, `Maximum`, `Minimum`, `Negate`,
//!   `Reciprocal`, `Exponential`, `Logarithm`, `SquareRoot`, `Tanh`,
//!   `Greater`, `Equal`, `Select`), plus `Erf` via the same
//!   Abramowitz & Stegun 7.1.26 polynomial `crate::msl::PROXIMA_ERF_FN` ports
//!   — see `PROXIMA_ERF_FN_WGSL`'s own doc.
//! - **`Keep::Reduce`** (matmul-shaped fold): one thread per OUTPUT element,
//!   a serial loop over the reduction dims inside the kernel — the same
//!   shape `crate::msl::push_serial_reduce_body` renders, with no SIMD-group
//!   cooperative path (WGSL subgroup operations are a device-capability
//!   extension, not baseline; see this module's own doc on why v1 stays
//!   serial).
//! - **`Keep::Scan`** (prefix fold): ONE thread, serial over every outer
//!   line and along the folded (innermost) dim within each — matching
//!   `proxima_tensor::cpu::run_scan`'s own accumulator, which persists
//!   across outer lines rather than resetting per line (see
//!   `render_scan`'s own doc). Not embarrassingly parallel the way a
//!   reduce is; v1 does not attempt a parallel prefix-sum reformulation.
//! - **`f32`, plus `f16` compute when the adapter offers it.** `Float16`
//!   renders through WGSL's `enable f16;` extension when [`WgslCaps::shader_f16`]
//!   is set (`crate::wgpu_driver::plan` sets it exactly when the acquired
//!   device requested `wgpu::Features::SHADER_F16`); otherwise it fails with
//!   [`EmitError::UnsupportedDType`], never silently falling back to `f32`
//!   compute. `BFloat16` collapses to `f32` unconditionally, the same
//!   `type_token` choice `crate::msl::type_token` makes (`bfloat` has no
//!   native WGSL type either way).
//!
//! `f16` here is COMPUTE only, not storage: every operand/output buffer
//! stays `array<f32>` (the wire format `crate::wgpu_driver`'s upload/readback
//! already speaks) — a `Float16`-dtype kernel casts `f32` down to `f16` at
//! the read and back up to `f32` at the write, so intermediate arithmetic
//! rounds the way half-precision compute does without widening the driver's
//! upload/readback byte format to native half storage the way
//! `crate::metal::upload_block`/`read_back` do. See `omega/tests/wgpu_parity.rs`
//! for the parity tolerance this rounding costs.
//!
//! # Gather (elementwise and `Keep::Reduce` only)
//!
//! An operand carrying a [`proxima_tensor::Lookup`] fetches an index out of
//! an `Indices` binding (a `storage, read` `array<f32>`, the same "an index
//! is an exact-integer float" convention `crate::msl::push_gather_fetch`
//! uses), clamps it into range, and records an out-of-range fetch into a
//! `Fault` binding (`storage, read_write` `array<atomic<u32>>`) via
//! `atomicMax` — the WGSL counterpart of `crate::msl::push_gather_fetch`'s
//! `atomic_fetch_max_explicit`. `crate::wgpu_driver` reads the fault buffer
//! back after every dispatch that gathers and turns a nonzero slot into
//! [`proxima_tensor::TensorError::GatherIndexOutOfRange`], exactly the error
//! `cpu::evaluate` reports for the same fetch. `Keep::Scan` does not take
//! this path yet (see [`EmitError::GatherNotSupported`]) — nothing in this
//! crate's v1 test surface needs a scanned gather.
//!
//! # Packed operands (elementwise and `Keep::Reduce` only)
//!
//! An operand named in the caller's [`crate::msl::PackedOperands`] table
//! binds as raw bytes (`array<u32>`, word-addressed — WGSL storage has no
//! byte-addressable type) instead of `array<f32>`, and every read goes
//! through the matching [`crate::msl::PackedCodec`] unpack function
//! `packed_codec_functions_wgsl` generates for that operand — a word-based
//! port of `crate::msl`'s own five packed-codec unpack functions plus
//! `BFloat16`/`Float16`, specialized per operand index because WGSL rejects a
//! storage-address-space function parameter (see that function's own doc).
//! `crate::wgpu_driver`'s upload path uploads the codec's raw packed bytes
//! unchanged rather than dequantizing on the host. `Keep::Scan` does not take
//! this path (same boundary as gather, see [`EmitError::UnsupportedOpKind`]).
//!
//! # `Iota` and `Constant`
//!
//! Both render (position-only, `render_iota`; literal-fill,
//! `render_constant`) — the real forward fixture's RoPE positions and
//! causal mask need them, and neither takes an operand, so there is no
//! gather/packed-operand interaction to scope out.
//!
//! # Runtime uniforms, not baked constants
//!
//! Exactly [`crate::msl::emit`]'s own stance: a node's extents and strides
//! are read out of a `storage, read` `Uniforms` buffer at kernel runtime,
//! never spliced into the source text as literal numbers. Two `BoundOp`
//! nodes that agree on structure (rank, operand count, body, `Keep`) but
//! differ in concrete extents/strides/buffers emit byte-identical WGSL, the
//! same cacheability property [`crate::msl::emit`]'s own doc proves for MSL.
//!
//! `Uniforms` lives in the `storage` address space rather than `uniform`:
//! WGSL's `uniform` address space imposes std140-style 16-byte array-stride
//! padding on scalar arrays, which would force every `i32` extent/stride
//! entry to occupy 16 bytes. `storage, read` has no such requirement (natural
//! 4-byte stride for `i32`), and every consumer here only ever reads it, so
//! there is nothing `uniform`'s extra guarantees would buy.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::{BoundOp, BoundOpKind, ComposedBody, DType, Keep, NodeId, ScalarOp, StepArg};

use crate::error::EmitError;
use crate::msl::{Binding, PackedCodec, PackedOperands, gather_count, gather_slots};

/// Threads per workgroup every v1 WGSL kernel dispatches with. A build-time
/// policy knob the way `crate::sized::SIMD_WIDTH` is a hardware fact — this
/// is the other kind: any positive value is legal WGSL, `64` is chosen to
/// match `crate::metal`'s own occupancy-driven default width for the
/// non-cooperative (serial) kernels this module emits exclusively. Folded
/// into [`WgslKernel::entry`] would be redundant (`entry_name` already keys
/// the pipeline cache by structure) — it is instead folded into the pipeline
/// CACHE KEY `crate::wgpu_driver` builds, since two kernels sharing a
/// structural name but a different workgroup size would need different
/// compiled pipelines.
pub const WORKGROUP_SIZE: u32 = 64;

/// One compiled WGSL kernel: source text, its `@compute` entry point, the
/// buffer-index -> data mapping a driver needs to bind before dispatch (the
/// same [`Binding`] list [`crate::msl::emit`] returns — no gather/fault
/// variant is ever constructed here, see the module doc), and how many
/// threads this dispatch needs (a driver divides by [`WORKGROUP_SIZE`] and
/// rounds up for the workgroup count it actually dispatches).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgslKernel {
    pub source: String,
    pub entry: String,
    pub bindings: Vec<Binding>,
    pub threads: u64,
    /// Threads per workgroup this dispatch actually needs — [`WORKGROUP_SIZE`]
    /// for every non-cooperative kernel, or the adapter's own subgroup width
    /// for a cooperative reduce (see [`WgslCaps::subgroup_size`]'s own doc):
    /// that kernel's `@compute @workgroup_size` is baked to exactly this
    /// value so one dispatched workgroup is exactly one subgroup, the same
    /// alignment `crate::msl::GridSpec::threadgroup_width` enforces for its
    /// own SIMD-group cooperative path.
    pub workgroup_size: u32,
}

/// Device capabilities [`emit_wgsl`] renders against — the WGSL counterpart
/// of what an adapter's `wgpu::Features` bitset already told
/// `crate::wgpu_driver::plan` at device-acquisition time. A capability the
/// device lacks is a NAMED rejection ([`EmitError::UnsupportedDType`]),
/// never a silent fallback to a lower-precision or serial path — see the
/// module doc's "f16 compute" section and [`crate::wgsl::WgslCaps::shader_f16`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WgslCaps {
    /// Whether the acquired `wgpu::Device` requested `wgpu::Features::SHADER_F16`
    /// — gates `DType::Float16` rendering through WGSL's `enable f16;`
    /// extension. `false` (the [`Default`]) rejects every `Float16` node with
    /// [`EmitError::UnsupportedDType`], the same posture v1 already took
    /// before this capability existed.
    pub shader_f16: bool,
    /// `Some(width)` when the acquired device requested `wgpu::Features::SUBGROUP`
    /// AND the adapter reports a FIXED subgroup width (`subgroup_min_size ==
    /// subgroup_max_size` — heterogeneous adapters report a range, which
    /// this module cannot pin a `@workgroup_size` to at emit time). Gates
    /// `reduce_is_cooperative`: `None` (the [`Default`]) keeps every
    /// `Keep::Reduce` fold on the portable one-thread-per-output serial
    /// path — never a silent guess at a width the device did not confirm.
    pub subgroup_size: Option<u32>,
}

/// Emits a WGSL kernel from a bound [`BoundOp`] — see the module doc for
/// exactly which op shapes this covers in v1.
///
/// # Errors
/// [`EmitError::UnsupportedDType`] for anything but `Float32` (or `Float16`
/// when `caps.shader_f16`), [`EmitError::GatherNotSupported`] for a gathered
/// operand, [`EmitError::UnsupportedOpKind`] for `Keep::Scan` over a packed operand,
/// [`EmitError::ArityMismatch`]/[`EmitError::ReductionBodyIsSelect`]/
/// [`EmitError::EmptyScan`] for the same structural failures
/// [`crate::msl::emit`] rejects.
pub fn emit_wgsl(resolved: &BoundOp, caps: WgslCaps, packed_operands: &PackedOperands) -> Result<WgslKernel, EmitError> {
    validate(resolved, packed_operands)?;
    let entry = entry_name(resolved);
    let element_type = type_token(resolved.node, resolved.dtype, caps)?;
    let quantized = operand_codecs(resolved, packed_operands);
    // `reduce_is_cooperative` is true only when `caps.subgroup_size` is
    // `Some`, but re-deriving that rather than `.expect()`-ing it keeps this
    // call site panic-free.
    let cooperative_width = if reduce_is_cooperative(resolved, caps) {
        caps.subgroup_size
    } else {
        None
    };
    let source = match &resolved.kind {
        BoundOpKind::Elementwise { .. } => render_elementwise(resolved, &entry, element_type, &quantized),
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => match cooperative_width {
            Some(width) => render_reduce_cooperative(resolved, &entry, element_type, &quantized, width),
            None => render_reduce(resolved, &entry, element_type, &quantized),
        },
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => render_scan(resolved, &entry, element_type),
        BoundOpKind::Iota => render_iota(resolved, &entry),
        BoundOpKind::Constant { value } => render_constant(resolved, &entry, *value),
    };
    let (threads, workgroup_size) = match cooperative_width {
        Some(width) => (grid_threads(resolved) * u64::from(width), width),
        None => (grid_threads(resolved), WORKGROUP_SIZE),
    };
    Ok(WgslKernel {
        source,
        entry,
        bindings: bindings(resolved),
        threads,
        workgroup_size,
    })
}

fn is_cooperative_reduce_op(op: ScalarOp) -> bool {
    matches!(op, ScalarOp::Add | ScalarOp::Multiply | ScalarOp::Maximum | ScalarOp::Minimum)
}

/// Whether `resolved` takes the SIMD-group-cooperative reduce path instead
/// of the one-thread-per-output serial fold — the WGSL counterpart of
/// `crate::msl::reduce_is_cooperative`: a `Keep::Reduce` fold, associative
/// reduce op ([`is_cooperative_reduce_op`]), no gathered operand (a
/// cooperative lane striding through the reduction would need its own
/// fault-slot contribution, which this pass does not implement — default to
/// serial when unsure), AND the device confirmed a fixed subgroup width
/// (`caps.subgroup_size`).
fn reduce_is_cooperative(resolved: &BoundOp, caps: WgslCaps) -> bool {
    caps.subgroup_size.is_some()
        && match &resolved.kind {
            BoundOpKind::Reduce {
                keep: Keep::Reduce,
                reduce_op,
                ..
            } => gather_count(resolved) == 0 && is_cooperative_reduce_op(*reduce_op),
            _ => false,
        }
}

/// The WGSL subgroup builtin that combines one lane's private accumulator
/// across the whole subgroup — the counterpart of `crate::msl::simd_combine_fn`.
/// Only called for an [`is_cooperative_reduce_op`] body.
fn subgroup_combine_fn(op: ScalarOp) -> &'static str {
    match op {
        ScalarOp::Add => "subgroupAdd",
        ScalarOp::Multiply => "subgroupMul",
        ScalarOp::Maximum => "subgroupMax",
        ScalarOp::Minimum => "subgroupMin",
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
        | ScalarOp::Select => unreachable!("subgroup_combine_fn is only called for a cooperative reduce_op"),
    }
}

/// The algebraic identity `op` folds against without changing a value — the
/// WGSL counterpart of `crate::msl::cooperative_identity_token`. Every lane
/// but lane 0 seeds its private accumulator with this (never the `BoundOp`'s
/// own `ReduceInit`, which may be `FirstElement` or otherwise mismatched
/// with `op`), so folding it into the final subgroup combine can never
/// perturb the result.
fn cooperative_identity_token(op: ScalarOp) -> &'static str {
    match op {
        ScalarOp::Add => "0.0",
        ScalarOp::Multiply => "1.0",
        ScalarOp::Maximum => "bitcast<f32>(0xff800000u)",
        ScalarOp::Minimum => "bitcast<f32>(0x7f800000u)",
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

fn type_token(node: NodeId, dtype: DType, caps: WgslCaps) -> Result<&'static str, EmitError> {
    match dtype {
        DType::Float32 | DType::BFloat16 => Ok("f32"),
        DType::Float16 if caps.shader_f16 => Ok("f16"),
        other => Err(EmitError::UnsupportedDType { node, dtype: other }),
    }
}

/// Whether operand reads/output writes for `element_type` need a cast — true
/// for every element type but `f32`, since [`crate::wgpu_driver`]'s
/// operand/output buffers always speak `array<f32>` (see the module doc's
/// "f16 here is COMPUTE only" note).
fn needs_f32_cast(element_type: &str) -> bool {
    element_type != "f32"
}

fn read_cast(element_type: &str, expr: &str) -> String {
    if needs_f32_cast(element_type) {
        format!("{element_type}({expr})")
    } else {
        expr.to_string()
    }
}

fn write_cast(element_type: &str, expr: &str) -> String {
    if needs_f32_cast(element_type) {
        format!("f32({expr})")
    } else {
        expr.to_string()
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

fn validate(resolved: &BoundOp, packed_operands: &PackedOperands) -> Result<(), EmitError> {
    validate_body(resolved.node, resolved.element_body())?;
    if let BoundOpKind::Reduce {
        reduce_op, keep, ..
    } = &resolved.kind
    {
        if matches!(reduce_op, ScalarOp::Select) {
            return Err(EmitError::ReductionBodyIsSelect { node: resolved.node });
        }
        if *keep == Keep::Scan {
            if resolved.extents.is_empty() {
                return Err(EmitError::EmptyScan { node: resolved.node });
            }
            // scan's accumulator persists across every outer line (see
            // `render_scan`'s own doc) -- gather and packed-operand support
            // there both need their own running-offset tracking
            // `crate::msl::render_scan` carries and this module's v1 has not
            // ported yet, see the module doc.
            for (node, _, gather) in resolved.operands() {
                if gather.is_some() {
                    return Err(EmitError::GatherNotSupported { node: resolved.node });
                }
                if packed_operands.contains_key(node) {
                    return Err(EmitError::UnsupportedOpKind {
                        node: resolved.node,
                        kind: "scan over a packed operand",
                    });
                }
            }
        }
    }
    Ok(())
}

/// One codec slot per operand — the WGSL counterpart of
/// `crate::msl::operand_codecs`, restated here since that helper is private
/// to `crate::msl`.
fn operand_codecs(resolved: &BoundOp, packed_operands: &PackedOperands) -> Vec<Option<PackedCodec>> {
    resolved
        .operands()
        .iter()
        .map(|(node, _, _)| packed_operands.get(node).copied())
        .collect()
}

fn bindings(resolved: &BoundOp) -> Vec<Binding> {
    let mut bindings: Vec<Binding> = resolved
        .operands()
        .iter()
        .map(|(node, _, _)| Binding::Input(*node))
        .collect();
    for (_, _, gather) in resolved.operands() {
        if let Some(lookup) = gather {
            bindings.push(Binding::Indices(lookup.indices));
        }
    }
    bindings.push(Binding::Output(resolved.node));
    bindings.push(Binding::Uniforms);
    if gather_count(resolved) > 0 {
        bindings.push(Binding::Fault);
    }
    bindings
}

fn reduction_dims(resolved: &BoundOp, output_axes: &[u16]) -> Vec<u16> {
    (0..resolved.extents.len() as u16)
        .filter(|dim| !output_axes.contains(dim))
        .collect()
}

fn grid_threads(resolved: &BoundOp) -> u64 {
    match &resolved.kind {
        BoundOpKind::Elementwise { .. } => resolved.extents.iter().product(),
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            ..
        } => output_axes.iter().map(|dim| resolved.extents[*dim as usize]).product(),
        // exactly one thread -- see `render_scan`'s own doc on why a scan's
        // accumulator persists across every outer line rather than resetting
        // per line, which rules out one thread per line.
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => 1,
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => resolved.extents.iter().product(),
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

fn keep_token(keep: Keep) -> &'static str {
    match keep {
        Keep::Reduce => "reduce",
        Keep::Scan => "scan",
    }
}

fn init_token(init: proxima_tensor::ReduceInit) -> &'static str {
    use proxima_tensor::ReduceInit;
    match init {
        ReduceInit::Zero => "zero",
        ReduceInit::One => "one",
        ReduceInit::NegativeInfinity => "negative_infinity",
        ReduceInit::PositiveInfinity => "positive_infinity",
        ReduceInit::FirstElement => "first_element",
    }
}

fn is_leaf(body: &ComposedBody) -> bool {
    body.steps.len() == 1
        && body.steps[0].args.iter().enumerate().all(
            |(index, arg)| matches!(arg, StepArg::Operand(operand) if *operand as usize == index),
        )
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
/// reduce-op/init/output-rank — the WGSL counterpart of `crate::msl::entry_name`,
/// narrower because v1 never fuses a gather bit-pattern suffix into the name
/// (gather is rejected before this is ever called).
fn entry_name(resolved: &BoundOp) -> String {
    let rank = resolved.extents.len();
    let operand_count = resolved.operands().len();
    match &resolved.kind {
        BoundOpKind::Elementwise { .. } => {
            let body = body_token(resolved.element_body());
            format!("omega_wgsl_elementwise_r{rank}_n{operand_count}_{body}")
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
            format!(
                "omega_wgsl_{kind}_r{rank}_o{output_rank}_n{operand_count}_{body}_{reduce_body}_{init}"
            )
        }
        BoundOpKind::Iota => format!("omega_wgsl_iota_r{rank}"),
        BoundOpKind::Constant { value } => {
            format!("omega_wgsl_constant_r{rank}_v{:08x}", value.to_bits())
        }
    }
}

/// `metal_stdlib` has no `erf`; WGSL's own standard library has none either
/// (no `erf` in any WGSL builtin function list). Same Abramowitz & Stegun
/// 7.1.26 approximation `crate::msl::PROXIMA_ERF_FN` ports, restated in WGSL
/// syntax (`fma` is a WGSL builtin over `f32`/vectors, so the polynomial
/// itself is unchanged) so a WGSL kernel and the CPU/Metal paths it is
/// checked against all run the identical formula.
const PROXIMA_ERF_FN_WGSL: &str = "\
fn proxima_erf(x: f32) -> f32 {
    let sign: f32 = select(1.0, -1.0, x < 0.0);
    let magnitude: f32 = abs(x);
    let t: f32 = 1.0 / fma(0.3275911, magnitude, 1.0);
    let poly: f32 = t * fma(fma(fma(fma(1.061405429, t, -1.453152027), t, 1.421413741), t, -0.284496736), t, 0.254829592);
    return sign * fma(poly, -exp(-magnitude * magnitude), 1.0);
}
";

/// `f16_bits_to_f32`, shared by every packed operand: WGSL's core
/// `unpack2x16float` built-in decodes an f16 scale directly from its raw
/// bits, no `enable f16;` extension needed (unlike [`WgslCaps::shader_f16`]'s
/// compute path) since `unpack2x16float` is baseline WGSL.
const F16_BITS_TO_F32_WGSL: &str = "\
fn f16_bits_to_f32(bits: u32) -> f32 {
    return unpack2x16float(bits).x;
}
";

/// Word-based ports of `crate::msl`'s five packed-codec unpack functions
/// (`Q4K_UNPACK_MSL`/`Q5K_UNPACK_MSL`/`Q6K_UNPACK_MSL`/`Q8_0_UNPACK_MSL`/
/// `Q4_0_UNPACK_MSL`) plus `BF16_UNPACK_MSL` and a `Float16` reader,
/// specialized to operand index `op` (`in{op}` is the only storage buffer
/// this generated text touches) — WGSL rejects a `ptr<storage, ...>`
/// function PARAMETER outright (naga: "pointer ... can't be passed into
/// functions", confirmed against this crate's own wgpu 30 dependency), so
/// unlike `crate::msl`'s one `device const uchar*`-parameterized function per
/// codec, each packed operand gets its own specialized copy that indexes
/// `in{op}` directly. Bytes: every read goes through `read_u8_{op}`
/// (`word = byte_offset / 4`, `shift = (byte_offset % 4) * 8`) since WGSL
/// storage has no byte-addressable type.
///
/// Bit-for-bit the same arithmetic as the MSL source; see each MSL
/// constant's own doc for the GGUF layout each codec ports.
fn packed_codec_functions_wgsl(op: usize) -> String {
    format!(
        "\
fn read_u8_{op}(byte_offset: i32) -> u32 {{
    let word = in{op}[byte_offset / 4];
    let shift = u32(byte_offset % 4) * 8u;
    return (word >> shift) & 0xFFu;
}}

fn read_u16le_{op}(byte_offset: i32) -> u32 {{
    return read_u8_{op}(byte_offset) | (read_u8_{op}(byte_offset + 1) << 8u);
}}

fn q4k_scale_min_{op}(scales_offset: i32, sub_block: i32) -> vec2<u32> {{
    if (sub_block < 4) {{
        let scale = read_u8_{op}(scales_offset + sub_block) & 63u;
        let minimum = read_u8_{op}(scales_offset + sub_block + 4) & 63u;
        return vec2<u32>(scale, minimum);
    }}
    let scale = (read_u8_{op}(scales_offset + sub_block + 4) & 0x0Fu) \
        | ((read_u8_{op}(scales_offset + sub_block - 4) >> 6u) << 4u);
    let minimum = (read_u8_{op}(scales_offset + sub_block + 4) >> 4u) \
        | ((read_u8_{op}(scales_offset + sub_block) >> 6u) << 4u);
    return vec2<u32>(scale, minimum);
}}

fn q4k_element_{op}(block_offset: i32, index: i32) -> f32 {{
    let d = f16_bits_to_f32(read_u16le_{op}(block_offset));
    let dmin = f16_bits_to_f32(read_u16le_{op}(block_offset + 2));
    let scales_offset = block_offset + 4;
    let qs_offset = block_offset + 16;
    let group = index / 64;
    let within = index % 64;
    let low_nibble = within < 32;
    let sub_block = 2 * group + select(1, 0, low_nibble);
    let byte_index = group * 32 + (within % 32);
    let scale_min = q4k_scale_min_{op}(scales_offset, sub_block);
    let scale = d * f32(scale_min.x);
    let minimum = dmin * f32(scale_min.y);
    let byte = read_u8_{op}(qs_offset + byte_index);
    let nibble = select(byte >> 4u, byte & 0x0Fu, low_nibble);
    return scale * f32(nibble) - minimum;
}}

fn q5k_element_{op}(block_offset: i32, index: i32) -> f32 {{
    let d = f16_bits_to_f32(read_u16le_{op}(block_offset));
    let dmin = f16_bits_to_f32(read_u16le_{op}(block_offset + 2));
    let scales_offset = block_offset + 4;
    let qh_offset = block_offset + 16;
    let qs_offset = block_offset + 48;
    let chunk = index / 64;
    let within = index % 64;
    let low = within < 32;
    let sub_block = 2 * chunk + select(1, 0, low);
    let offset = within % 32;
    let scale_min = q4k_scale_min_{op}(scales_offset, sub_block);
    let scale = d * f32(scale_min.x);
    let minimum = dmin * f32(scale_min.y);
    let mask = select(2u << u32(2 * chunk), 1u << u32(2 * chunk), low);
    let qs_byte = read_u8_{op}(qs_offset + chunk * 32 + offset);
    let nibble = select(qs_byte >> 4u, qs_byte & 0x0Fu, low);
    let qh_byte = read_u8_{op}(qh_offset + offset);
    let high = select(0.0, 16.0, (qh_byte & mask) != 0u);
    return scale * (f32(nibble) + high) - minimum;
}}

fn q6k_element_{op}(block_offset: i32, index: i32) -> f32 {{
    let d = f16_bits_to_f32(read_u16le_{op}(block_offset + 208));
    let half_index = index / 128;
    let local = index % 128;
    let l = local % 32;
    let lane = local / 32;
    let sub_block_in_half = l / 16;
    let ql_offset = block_offset + half_index * 64;
    let qh_offset = block_offset + 128 + half_index * 32;
    let scales_offset = block_offset + 192;
    let ql_byte = select(read_u8_{op}(ql_offset + l + 32), read_u8_{op}(ql_offset + l), lane % 2 == 0);
    let nibble = select(ql_byte >> 4u, ql_byte & 0x0Fu, lane < 2);
    let high2 = (read_u8_{op}(qh_offset + l) >> u32(lane * 2)) & 0x03u;
    let level = nibble | (high2 << 4u);
    let scale_byte = read_u8_{op}(scales_offset + half_index * 8 + sub_block_in_half + lane * 2);
    let scale = f32(bitcast<i32>(scale_byte << 24u) >> 24);
    let quant = f32(level) - 32.0;
    return d * scale * quant;
}}

fn q8_0_element_{op}(block_offset: i32, index: i32) -> f32 {{
    let d = f16_bits_to_f32(read_u16le_{op}(block_offset));
    let byte = read_u8_{op}(block_offset + 2 + index);
    let level = bitcast<i32>(byte << 24u) >> 24;
    return f32(level) * d;
}}

fn q4_0_element_{op}(block_offset: i32, index: i32) -> f32 {{
    let d = f16_bits_to_f32(read_u16le_{op}(block_offset));
    let byte = read_u8_{op}(block_offset + 2 + (index % 16));
    let nibble = select(i32(byte >> 4u), i32(byte & 0x0Fu), index < 16);
    return f32(nibble - 8) * d;
}}

fn f16_element_{op}(block_offset: i32, index: i32) -> f32 {{
    return f16_bits_to_f32(read_u16le_{op}(block_offset));
}}

fn bf16_element_{op}(block_offset: i32, index: i32) -> f32 {{
    return bitcast<f32>(read_u16le_{op}(block_offset) << 16u);
}}
"
    )
}

fn scalar_op_expr(op: ScalarOp, args: &[&str]) -> String {
    match op {
        ScalarOp::Identity => (*args.first().unwrap_or(&"0.0")).into(),
        ScalarOp::Add => format!("({} + {})", args[0], args[1]),
        ScalarOp::Subtract => format!("({} - {})", args[0], args[1]),
        ScalarOp::Multiply => format!("({} * {})", args[0], args[1]),
        ScalarOp::Divide => format!("({} / {})", args[0], args[1]),
        ScalarOp::Maximum => format!("max({}, {})", args[0], args[1]),
        ScalarOp::Minimum => format!("min({}, {})", args[0], args[1]),
        ScalarOp::Negate => format!("(-{})", args[0]),
        ScalarOp::Reciprocal => format!("(1.0 / {})", args[0]),
        ScalarOp::Exponential => format!("exp({})", args[0]),
        ScalarOp::Logarithm => format!("log({})", args[0]),
        ScalarOp::SquareRoot => format!("sqrt({})", args[0]),
        ScalarOp::Tanh => format!("tanh({})", args[0]),
        ScalarOp::Erf => format!("proxima_erf({})", args[0]),
        ScalarOp::Greater => format!("select(0.0, 1.0, {} > {})", args[0], args[1]),
        ScalarOp::Equal => format!("select(0.0, 1.0, abs({} - {}) == 0.0)", args[0], args[1]),
        ScalarOp::Select => format!("select({}, {}, {} != 0.0)", args[2], args[1], args[0]),
    }
}

/// `(init expression, seeded-from-the-start)` — the WGSL counterpart of
/// `crate::msl::fold_init_tokens`. `NegativeInfinity`/`PositiveInfinity` are
/// bitcast rather than a literal keyword: WGSL's base spec has no portable
/// `INFINITY` token, and `1.0 / 0.0` is not guaranteed IEEE-754 semantics at
/// WGSL const-eval time, so the exact bit pattern is spelled directly.
fn fold_init_tokens(init: proxima_tensor::ReduceInit) -> (&'static str, &'static str) {
    use proxima_tensor::ReduceInit;
    match init {
        ReduceInit::Zero => ("0.0", "true"),
        ReduceInit::One => ("1.0", "true"),
        ReduceInit::NegativeInfinity => ("bitcast<f32>(0xff800000u)", "true"),
        ReduceInit::PositiveInfinity => ("bitcast<f32>(0x7f800000u)", "true"),
        ReduceInit::FirstElement => ("0.0", "false"),
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
        source.push_str(&format!("{indent}let step{index}: {element_type} = {expr};\n"));
    }
    format!("step{}", body.steps.len().saturating_sub(1))
}

fn preamble(
    source: &mut String,
    operand_count: usize,
    gather_count: usize,
    quantized: &[Option<PackedCodec>],
    element_type: &str,
    uniforms_struct: &str,
) {
    // `enable` directives are WGSL module-scope items that MUST precede
    // every other declaration -- see the module doc's "f16 compute" section.
    if element_type == "f16" {
        source.push_str("enable f16;\n");
    }
    source.push_str(PROXIMA_ERF_FN_WGSL);
    source.push('\n');
    if quantized.iter().any(Option::is_some) {
        source.push_str(F16_BITS_TO_F32_WGSL);
        source.push('\n');
    }
    // one specialized copy of the codec-unpack functions per packed operand
    // index -- see `packed_codec_functions_wgsl`'s own doc for why WGSL
    // cannot take one function parameterized over which `in{n}` to read.
    for (index, codec) in quantized.iter().enumerate() {
        if codec.is_some() {
            source.push_str(&packed_codec_functions_wgsl(index));
            source.push('\n');
        }
    }
    source.push_str(uniforms_struct);
    source.push('\n');
    for index in 0..operand_count {
        // a packed operand's buffer is raw BYTES reinterpreted as `u32`
        // words (`read_u8_{n}`/`read_u16le_{n}` split a byte/word offset
        // back out at the read) -- see `packed_codec_functions_wgsl`'s
        // own doc.
        let binding_type = if quantized.get(index).copied().flatten().is_some() {
            "array<u32>"
        } else {
            "array<f32>"
        };
        source.push_str(&format!(
            "@group(0) @binding({index}) var<storage, read> in{index}: {binding_type};\n"
        ));
    }
    for slot in 0..gather_count {
        source.push_str(&format!(
            "@group(0) @binding({}) var<storage, read> gather_idx{slot}: array<f32>;\n",
            operand_count + slot
        ));
    }
    let output_binding = operand_count + gather_count;
    source.push_str(&format!(
        "@group(0) @binding({output_binding}) var<storage, read_write> out: array<f32>;\n"
    ));
    let uniforms_binding = output_binding + 1;
    source.push_str(&format!(
        "@group(0) @binding({uniforms_binding}) var<storage, read> u: Uniforms;\n"
    ));
    if gather_count > 0 {
        let fault_binding = uniforms_binding + 1;
        source.push_str(&format!(
            "@group(0) @binding({fault_binding}) var<storage, read_write> fault: array<atomic<u32>>;\n"
        ));
    }
    source.push('\n');
}

/// Declares the `Uniforms` fields a gather needs — the WGSL counterpart of
/// `crate::msl::push_gather_uniform_fields`. Declared only when
/// `gather_count > 0`, so a gather-free kernel's `Uniforms` struct is
/// byte-for-byte what it was before gather existed.
fn push_gather_uniform_fields(source: &mut String, gather_count: usize, rank_len: usize) {
    if gather_count == 0 {
        return;
    }
    source.push_str(&format!("    gather_index_base: array<i32, {gather_count}>,\n"));
    source.push_str(&format!(
        "    gather_index_strides: array<array<i32, {rank_len}>, {gather_count}>,\n"
    ));
    source.push_str(&format!(
        "    gather_element_stride: array<i32, {gather_count}>,\n"
    ));
    source.push_str(&format!("    gather_extent: array<i32, {gather_count}>,\n"));
}

/// Records an out-of-range fetched index into `fault[gather_slot]` — the
/// WGSL counterpart of `crate::msl::push_gather_fault_check`. `atomicMax`
/// plays the same role `atomic_fetch_max_explicit` does on the Metal side:
/// whichever value wins under concurrent invocations is still a genuine
/// fault, and the driver only needs to know one occurred and at what value.
fn push_gather_fault_check(source: &mut String, operand_index: usize, gather_slot: usize, indent: &str) {
    source.push_str(&format!(
        "{indent}if (fetched{operand_index} < 0 || fetched{operand_index} >= u.gather_extent[{gather_slot}]) {{\n"
    ));
    source.push_str(&format!(
        "{indent}    atomicMax(&fault[{gather_slot}], u32(max(fetched{operand_index}, 0)) + 1u);\n"
    ));
    source.push_str(&format!("{indent}}}\n"));
}

/// Emits the fetch for one gathered operand — the WGSL counterpart of
/// `crate::msl::push_gather_fetch`: reads the index, checks and records a
/// fault, clamps into range regardless (so the read this drives always
/// lands in bounds), and adds the resulting offset into `offset_var`.
fn push_gather_fetch(
    source: &mut String,
    operand_index: usize,
    gather_slot: usize,
    rank: usize,
    coord_var: &str,
    offset_var: &str,
) {
    source.push_str(&format!(
        "    var gather_off{operand_index}: i32 = u.gather_index_base[{gather_slot}];\n"
    ));
    for dim in 0..rank {
        source.push_str(&format!(
            "    gather_off{operand_index} += {coord_var}[{dim}] * u.gather_index_strides[{gather_slot}][{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "    var fetched{operand_index}: i32 = i32(gather_idx{gather_slot}[gather_off{operand_index}]);\n"
    ));
    push_gather_fault_check(source, operand_index, gather_slot, "    ");
    source.push_str(&format!(
        "    fetched{operand_index} = max(0, min(fetched{operand_index}, u.gather_extent[{gather_slot}] - 1));\n"
    ));
    source.push_str(&format!(
        "    {offset_var} += fetched{operand_index} * u.gather_element_stride[{gather_slot}];\n"
    ));
}

fn kernel_signature(source: &mut String, entry: &str) {
    source.push_str(&format!("@compute @workgroup_size({WORKGROUP_SIZE})\n"));
    source.push_str(&format!(
        "fn {entry}(@builtin(global_invocation_id) global_id: vec3<u32>) {{\n"
    ));
    source.push_str("    let gid: i32 = i32(global_id.x);\n");
}

/// [`kernel_signature`]'s cooperative-reduce counterpart: dispatched at
/// exactly `width` threads per workgroup (baked into `@workgroup_size` so
/// one workgroup is exactly one subgroup) and carries the extra
/// `subgroup_invocation_id` builtin every lane needs to know its own
/// position within that subgroup.
fn cooperative_kernel_signature(source: &mut String, entry: &str, width: u32) {
    source.push_str(&format!("@compute @workgroup_size({width})\n"));
    source.push_str(&format!(
        "fn {entry}(@builtin(global_invocation_id) global_id: vec3<u32>, \
         @builtin(subgroup_invocation_id) lane: u32) {{\n"
    ));
    source.push_str("    let gid: i32 = i32(global_id.x);\n");
}

fn codec_function_name(codec: PackedCodec) -> &'static str {
    match codec {
        PackedCodec::Q4K => "q4k_element",
        PackedCodec::Q5K => "q5k_element",
        PackedCodec::Q6K => "q6k_element",
        PackedCodec::Q8_0 => "q8_0_element",
        PackedCodec::Q4_0 => "q4_0_element",
        PackedCodec::Float16 => "f16_element",
        PackedCodec::BFloat16 => "bf16_element",
    }
}

/// How operand `index` is READ, given the element-offset expression the
/// caller already computed — the WGSL counterpart of `crate::msl::operand_read`.
/// A plain operand is a direct index; a packed operand's element offset
/// splits into a super-block (`offset / block_elements`, scaled to a byte
/// offset by `block_bytes`) and a position inside it (`offset % block_elements`),
/// the same split the `_{index}`-suffixed `*_element` function
/// `packed_codec_functions_wgsl` generated for this operand expects.
fn wgsl_operand_read(index: usize, offset_expr: &str, codec: Option<PackedCodec>) -> String {
    match codec {
        None => format!("in{index}[{offset_expr}]"),
        Some(codec) => {
            let elements = codec.block_elements();
            let bytes = codec.block_bytes();
            let function = codec_function_name(codec);
            format!("{function}_{index}(({offset_expr} / {elements}) * {bytes}, {offset_expr} % {elements})")
        }
    }
}

fn render_elementwise(resolved: &BoundOp, entry: &str, element_type: &str, quantized: &[Option<PackedCodec>]) -> String {
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let gather_total = gather_count(resolved);
    let slots = gather_slots(resolved);

    let mut uniforms = String::new();
    uniforms.push_str("struct Uniforms {\n");
    uniforms.push_str("    total_elements: i32,\n");
    uniforms.push_str(&format!("    extents: array<i32, {rank_len}>,\n"));
    uniforms.push_str(&format!("    operand_base: array<i32, {operand_count}>,\n"));
    uniforms.push_str(&format!(
        "    operand_strides: array<array<i32, {rank_len}>, {operand_count}>,\n"
    ));
    push_gather_uniform_fields(&mut uniforms, gather_total, rank_len);
    uniforms.push_str("};\n");

    let mut source = String::new();
    preamble(&mut source, operand_count, gather_total, quantized, element_type, &uniforms);
    kernel_signature(&mut source, entry);
    source.push_str("    if (gid >= u.total_elements) { return; }\n");

    if rank > 0 {
        source.push_str(&format!("    var coord: array<i32, {rank_len}>;\n"));
        source.push_str("    var remaining: i32 = gid;\n");
        for dim in (0..rank).rev() {
            source.push_str(&format!(
                "    coord[{dim}] = remaining % u.extents[{dim}]; remaining = remaining / u.extents[{dim}];\n"
            ));
        }
    }

    for (index, gather_slot) in slots.iter().enumerate() {
        source.push_str(&format!("    var off{index}: i32 = u.operand_base[{index}];\n"));
        for dim in 0..rank {
            source.push_str(&format!(
                "    off{index} += coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slot {
            push_gather_fetch(&mut source, index, *slot, rank, "coord", &format!("off{index}"));
        }
    }

    source.push_str(&format!(
        "    var scratch: array<{element_type}, {}>;\n",
        operand_count.max(1)
    ));
    for index in 0..operand_count {
        let codec = quantized.get(index).copied().flatten();
        let expr = wgsl_operand_read(index, &format!("off{index}"), codec);
        let read = read_cast(element_type, &expr);
        source.push_str(&format!("    scratch[{index}] = {read};\n"));
    }

    let result = push_body_steps(&mut source, resolved.element_body(), "    ", element_type);
    let stored = write_cast(element_type, &result);
    source.push_str(&format!("    out[gid] = {stored};\n"));
    source.push_str("}\n");
    source
}

fn render_reduce(resolved: &BoundOp, entry: &str, element_type: &str, quantized: &[Option<PackedCodec>]) -> String {
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
    let gather_total = gather_count(resolved);
    let slots = gather_slots(resolved);

    let mut uniforms = String::new();
    uniforms.push_str("struct Uniforms {\n");
    uniforms.push_str("    output_total: i32,\n");
    uniforms.push_str("    reduction_total: i32,\n");
    uniforms.push_str(&format!("    output_extents: array<i32, {output_rank_len}>,\n"));
    uniforms.push_str(&format!(
        "    reduction_extents: array<i32, {reduce_rank_len}>,\n"
    ));
    uniforms.push_str(&format!("    operand_base: array<i32, {operand_count}>,\n"));
    uniforms.push_str(&format!(
        "    operand_strides: array<array<i32, {rank_len}>, {operand_count}>,\n"
    ));
    uniforms.push_str("    out_base: i32,\n");
    uniforms.push_str(&format!("    out_strides: array<i32, {rank_len}>,\n"));
    push_gather_uniform_fields(&mut uniforms, gather_total, rank_len);
    uniforms.push_str("};\n");

    let mut source = String::new();
    preamble(&mut source, operand_count, gather_total, quantized, element_type, &uniforms);
    kernel_signature(&mut source, entry);
    source.push_str("    if (gid >= u.output_total) { return; }\n");

    source.push_str(&format!("    var full_coord: array<i32, {rank_len}>;\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }

    if output_rank > 0 {
        source.push_str(&format!("    var output_coord: array<i32, {output_rank_len}>;\n"));
        source.push_str("    var remaining: i32 = gid;\n");
        for index in (0..output_rank).rev() {
            source.push_str(&format!(
                "    output_coord[{index}] = remaining % u.output_extents[{index}]; \
                 remaining = remaining / u.output_extents[{index}];\n"
            ));
        }
        for (index, dim) in output_axes.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
    }

    let (init_expr, seeded_init) = fold_init_tokens(*init);
    source.push_str(&format!("    var accumulator: {element_type} = {init_expr};\n"));
    source.push_str(&format!("    var seeded: bool = {seeded_init};\n"));

    source.push_str("    for (var r: i32 = 0; r < u.reduction_total; r = r + 1) {\n");
    if reduce_rank > 0 {
        source.push_str(&format!("        var reduction_coord: array<i32, {reduce_rank_len}>;\n"));
        source.push_str("        var remaining_r: i32 = r;\n");
        for index in (0..reduce_rank).rev() {
            source.push_str(&format!(
                "        reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
                 remaining_r = remaining_r / u.reduction_extents[{index}];\n"
            ));
        }
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!("        full_coord[{dim}] = reduction_coord[{index}];\n"));
        }
    }

    for (index, gather_slot) in slots.iter().enumerate() {
        source.push_str(&format!("        var off{index}: i32 = u.operand_base[{index}];\n"));
        for dim in 0..rank {
            source.push_str(&format!(
                "        off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slot {
            push_gather_fetch(
                &mut source,
                index,
                *slot,
                rank,
                "full_coord",
                &format!("off{index}"),
            );
        }
    }
    source.push_str(&format!(
        "        var scratch: array<{element_type}, {}>;\n",
        operand_count.max(1)
    ));
    for index in 0..operand_count {
        let codec = quantized.get(index).copied().flatten();
        let expr = wgsl_operand_read(index, &format!("off{index}"), codec);
        let read = read_cast(element_type, &expr);
        source.push_str(&format!("        scratch[{index}] = {read};\n"));
    }
    let value_expr = push_body_steps(&mut source, resolved.element_body(), "        ", element_type);
    source.push_str(&format!("        let value: {element_type} = {value_expr};\n"));
    let combine_expr = scalar_op_expr(*reduce_op, &["accumulator", "value"]);
    source.push_str(&format!("        accumulator = select(value, {combine_expr}, seeded);\n"));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    source.push_str("    var out_offset: i32 = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!("    out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"));
    }
    let stored = write_cast(element_type, "accumulator");
    source.push_str(&format!("    out[out_offset] = {stored};\n"));
    source.push_str("}\n");
    source
}

/// The SIMD-group-cooperative fold: `width` lanes (one whole workgroup, see
/// [`cooperative_kernel_signature`]) split one output element's reduction
/// axis, each striding through `reduction_total` by `width` so every
/// element is visited by exactly one lane, then combine via
/// [`subgroup_combine_fn`] — the WGSL counterpart of
/// `crate::msl::push_cooperative_reduce_body`'s general (non-packed,
/// non-tiled) path. Only lane 0 writes the result, and only lane 0 seeds
/// from the `BoundOp`'s real `ReduceInit`; every other lane seeds from
/// [`cooperative_identity_token`] so the true seed folds into the group
/// exactly once. Never called with a gathered operand (see
/// `reduce_is_cooperative`'s own doc), so no fault/indices plumbing here.
fn render_reduce_cooperative(
    resolved: &BoundOp,
    entry: &str,
    element_type: &str,
    quantized: &[Option<PackedCodec>],
    width: u32,
) -> String {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        ..
    } = &resolved.kind
    else {
        unreachable!("render_reduce_cooperative is only called for a Keep::Reduce fold")
    };
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let output_rank = output_axes.len();
    let output_rank_len = output_rank.max(1);
    let reduce_dims = reduction_dims(resolved, output_axes);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);

    let mut uniforms = String::new();
    uniforms.push_str("struct Uniforms {\n");
    uniforms.push_str("    output_total: i32,\n");
    uniforms.push_str("    reduction_total: i32,\n");
    uniforms.push_str(&format!("    output_extents: array<i32, {output_rank_len}>,\n"));
    uniforms.push_str(&format!(
        "    reduction_extents: array<i32, {reduce_rank_len}>,\n"
    ));
    uniforms.push_str(&format!("    operand_base: array<i32, {operand_count}>,\n"));
    uniforms.push_str(&format!(
        "    operand_strides: array<array<i32, {rank_len}>, {operand_count}>,\n"
    ));
    uniforms.push_str("    out_base: i32,\n");
    uniforms.push_str(&format!("    out_strides: array<i32, {rank_len}>,\n"));
    uniforms.push_str("};\n");

    let mut source = String::new();
    // no gather ever reaches here (see this function's own doc), so `quantized`
    // is the only per-operand table `preamble` needs and `gather_count` is 0.
    preamble(&mut source, operand_count, 0, quantized, element_type, &uniforms);
    cooperative_kernel_signature(&mut source, entry, width);

    source.push_str(&format!("    let output_index: i32 = gid / {width};\n"));
    source.push_str("    if (output_index >= u.output_total) { return; }\n");

    source.push_str(&format!("    var full_coord: array<i32, {rank_len}>;\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }

    if output_rank > 0 {
        source.push_str(&format!("    var output_coord: array<i32, {output_rank_len}>;\n"));
        source.push_str("    var remaining: i32 = output_index;\n");
        for index in (0..output_rank).rev() {
            source.push_str(&format!(
                "    output_coord[{index}] = remaining % u.output_extents[{index}]; \
                 remaining = remaining / u.output_extents[{index}];\n"
            ));
        }
        for (index, dim) in output_axes.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
    }

    let (init_expr, seeded_init) = fold_init_tokens(*init);
    let identity = cooperative_identity_token(*reduce_op);
    source.push_str(&format!("    var accumulator: {element_type};\n"));
    source.push_str("    var seeded: bool;\n");
    source.push_str("    if (lane == 0u) {\n");
    source.push_str(&format!("        accumulator = {init_expr};\n"));
    source.push_str(&format!("        seeded = {seeded_init};\n"));
    source.push_str("    } else {\n");
    source.push_str(&format!("        accumulator = {identity};\n"));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    source.push_str(&format!(
        "    for (var r: i32 = i32(lane); r < u.reduction_total; r = r + {width}) {{\n"
    ));
    if reduce_rank > 0 {
        source.push_str(&format!("        var reduction_coord: array<i32, {reduce_rank_len}>;\n"));
        source.push_str("        var remaining_r: i32 = r;\n");
        for index in (0..reduce_rank).rev() {
            source.push_str(&format!(
                "        reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
                 remaining_r = remaining_r / u.reduction_extents[{index}];\n"
            ));
        }
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!("        full_coord[{dim}] = reduction_coord[{index}];\n"));
        }
    }

    for index in 0..operand_count {
        source.push_str(&format!("        var off{index}: i32 = u.operand_base[{index}];\n"));
        for dim in 0..rank {
            source.push_str(&format!(
                "        off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
    }
    source.push_str(&format!(
        "        var scratch: array<{element_type}, {}>;\n",
        operand_count.max(1)
    ));
    for index in 0..operand_count {
        let codec = quantized.get(index).copied().flatten();
        let expr = wgsl_operand_read(index, &format!("off{index}"), codec);
        let read = read_cast(element_type, &expr);
        source.push_str(&format!("        scratch[{index}] = {read};\n"));
    }
    let value_expr = push_body_steps(&mut source, resolved.element_body(), "        ", element_type);
    source.push_str(&format!("        let value: {element_type} = {value_expr};\n"));
    let combine_expr = scalar_op_expr(*reduce_op, &["accumulator", "value"]);
    source.push_str(&format!("        accumulator = select(value, {combine_expr}, seeded);\n"));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    let combine_fn = subgroup_combine_fn(*reduce_op);
    source.push_str(&format!("    let reduced: {element_type} = {combine_fn}(accumulator);\n"));
    source.push_str("    if (lane == 0u) {\n");
    source.push_str("        var out_offset: i32 = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "        out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    let stored = write_cast(element_type, "reduced");
    source.push_str(&format!("        out[out_offset] = {stored};\n"));
    source.push_str("    }\n");
    source.push_str("}\n");
    source
}

/// WGSL spelling for a literal `f32` — the counterpart of
/// `crate::msl::msl_literal`. NaN/infinity have no portable WGSL literal
/// spelling (WGSL float literals must be finite), so they are bitcast from
/// their exact IEEE-754 bit pattern, the same trick [`fold_init_tokens`]
/// already uses for `ReduceInit::NegativeInfinity`/`PositiveInfinity`.
fn wgsl_literal(value: f32) -> String {
    if value.is_nan() {
        return "bitcast<f32>(0x7fc00000u)".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "bitcast<f32>(0xff800000u)".to_string()
        } else {
            "bitcast<f32>(0x7f800000u)".to_string()
        };
    }
    format!("{value:?}")
}

/// [`BoundOpKind::Iota`]'s kernel: the output value at each position is the
/// thread's own grid coordinate — the WGSL counterpart of
/// `crate::msl::render_iota`. No operand buffers, no body; every kernel
/// already computes `gid`, so there is nothing to derive beyond widening it
/// to the output buffer's `f32`.
fn render_iota(resolved: &BoundOp, entry: &str) -> String {
    let mut uniforms = String::new();
    uniforms.push_str("struct Uniforms {\n    total_elements: i32,\n};\n");

    let mut source = String::new();
    preamble(&mut source, 0, 0, &[], "f32", &uniforms);
    kernel_signature(&mut source, entry);
    source.push_str("    if (gid >= u.total_elements) { return; }\n");
    source.push_str("    out[gid] = f32(gid);\n");
    source.push_str("}\n");
    let _ = resolved;
    source
}

/// [`BoundOpKind::Constant`]'s kernel: every output element is the same
/// literal `value`, no operand buffers and no per-element compute at all —
/// the WGSL counterpart of `crate::msl::render_iota`'s sibling for a
/// constant-fill node. The output buffer is always `array<f32>` regardless
/// of the node's own dtype (see the module doc's "f16 here is COMPUTE only"
/// note), so this writes the literal directly with no `element_type` cast to
/// thread through.
fn render_constant(resolved: &BoundOp, entry: &str, value: f32) -> String {
    let mut uniforms = String::new();
    uniforms.push_str("struct Uniforms {\n    total_elements: i32,\n};\n");

    let mut source = String::new();
    preamble(&mut source, 0, 0, &[], "f32", &uniforms);
    kernel_signature(&mut source, entry);
    source.push_str("    if (gid >= u.total_elements) { return; }\n");
    let literal = wgsl_literal(value);
    source.push_str(&format!("    out[gid] = {literal};\n"));
    source.push_str("}\n");
    let _ = resolved;
    source
}

fn render_scan(resolved: &BoundOp, entry: &str, element_type: &str) -> String {
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

    let mut uniforms = String::new();
    uniforms.push_str("struct Uniforms {\n");
    uniforms.push_str("    outer_total: i32,\n");
    uniforms.push_str("    inner_len: i32,\n");
    uniforms.push_str(&format!("    outer_extents: array<i32, {outer_rank_len}>,\n"));
    uniforms.push_str(&format!("    operand_base: array<i32, {operand_count}>,\n"));
    uniforms.push_str(&format!(
        "    operand_strides: array<array<i32, {rank_len}>, {operand_count}>,\n"
    ));
    uniforms.push_str("    out_base: i32,\n");
    uniforms.push_str(&format!("    out_strides: array<i32, {rank_len}>,\n"));
    uniforms.push_str("};\n");

    let mut source = String::new();
    preamble(&mut source, operand_count, 0, &alloc::vec![None; operand_count], element_type, &uniforms);
    kernel_signature(&mut source, entry);
    // `crate::msl`'s own `push_serial_reduce_body`/`run_scan` (the CPU
    // oracle) carry ONE accumulator across every outer line, not one per
    // line -- `cpu::run_scan` declares `accumulator`/`seeded` OUTSIDE its
    // `outer_flat` loop. A scan is therefore not embarrassingly parallel
    // across outer lines the way a reduce is: this dispatches exactly one
    // thread (see `grid_threads`'s own doc) that walks every outer line
    // serially, matching that accumulator-persistence exactly.
    source.push_str("    if (gid != 0) { return; }\n");

    let (init_expr, seeded_init) = fold_init_tokens(*init);
    source.push_str(&format!("    var accumulator: {element_type} = {init_expr};\n"));
    source.push_str(&format!("    var seeded: bool = {seeded_init};\n"));

    source.push_str("    for (var outer: i32 = 0; outer < u.outer_total; outer = outer + 1) {\n");
    if outer_rank > 0 {
        source.push_str(&format!("        var outer_coord: array<i32, {outer_rank_len}>;\n"));
        source.push_str("        var remaining: i32 = outer;\n");
        for dim in (0..outer_rank).rev() {
            source.push_str(&format!(
                "        outer_coord[{dim}] = remaining % u.outer_extents[{dim}]; \
                 remaining = remaining / u.outer_extents[{dim}];\n"
            ));
        }
    }
    for index in 0..operand_count {
        source.push_str(&format!("        var running{index}: i32 = u.operand_base[{index}];\n"));
        for dim in 0..outer_rank {
            source.push_str(&format!(
                "        running{index} += outer_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
    }
    source.push_str("        var out_running: i32 = u.out_base;\n");
    for dim in 0..outer_rank {
        source.push_str(&format!(
            "        out_running += outer_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }

    source.push_str("        for (var step: i32 = 0; step < u.inner_len; step = step + 1) {\n");
    source.push_str(&format!(
        "            var scratch: array<{element_type}, {}>;\n",
        operand_count.max(1)
    ));
    for index in 0..operand_count {
        let read = read_cast(element_type, &format!("in{index}[running{index}]"));
        source.push_str(&format!("            scratch[{index}] = {read};\n"));
        source.push_str(&format!(
            "            running{index} += u.operand_strides[{index}][{last_dim}];\n"
        ));
    }
    let value_expr = push_body_steps(&mut source, resolved.element_body(), "            ", element_type);
    source.push_str(&format!("            let value: {element_type} = {value_expr};\n"));
    let combine_expr = scalar_op_expr(*reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "            accumulator = select(value, {combine_expr}, seeded);\n"
    ));
    source.push_str("            seeded = true;\n");
    let stored = write_cast(element_type, "accumulator");
    source.push_str(&format!("            out[out_running] = {stored};\n"));
    source.push_str(&format!("            out_running += u.out_strides[{last_dim}];\n"));
    source.push_str("        }\n");
    source.push_str("    }\n");
    source.push_str("}\n");
    source
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_tensor::{DType, Extent, IndexMap, Op, ScalarOp, append, bind, infer, map};

    use super::*;

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
    fn elementwise_tanh_emits_wgsl_with_the_expected_shape() {
        let bound = elementwise_tanh_op(8);
        let kernel = emit_wgsl(&bound, WgslCaps::default(), &PackedOperands::new()).expect("emit succeeds");
        assert!(kernel.source.contains("@compute"));
        assert!(kernel.source.contains("tanh("));
        assert_eq!(kernel.bindings.len(), 3);
        assert_eq!(kernel.threads, 8);
    }

    #[test]
    fn same_structure_different_extents_yield_identical_source() {
        let small = emit_wgsl(&elementwise_tanh_op(4), WgslCaps::default(), &PackedOperands::new()).expect("emit succeeds");
        let large = emit_wgsl(&elementwise_tanh_op(4096), WgslCaps::default(), &PackedOperands::new()).expect("emit succeeds");
        assert_eq!(small.source, large.source);
        assert_ne!(small.threads, large.threads);
    }

    #[test]
    fn erf_emits_the_polynomial_helper() {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Erf,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
        let bound = bound.into_iter().next().expect("one bound op");
        let kernel = emit_wgsl(&bound, WgslCaps::default(), &PackedOperands::new()).expect("emit succeeds");
        assert!(kernel.source.contains("fn proxima_erf"));
        assert!(kernel.source.contains("proxima_erf(scratch[0])"));
    }

    #[test]
    fn a_float16_node_is_rejected() {
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
        let error = emit_wgsl(&bound, WgslCaps::default(), &PackedOperands::new()).expect_err("f16 is rejected without shader_f16");
        assert!(matches!(error, EmitError::UnsupportedDType { .. }));
    }

    #[test]
    fn a_float16_node_renders_through_enable_f16_when_the_capability_is_set() {
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
        let caps = WgslCaps {
            shader_f16: true,
            ..WgslCaps::default()
        };
        let kernel = emit_wgsl(&bound, caps, &PackedOperands::new()).expect("f16 emits when shader_f16 is set");
        assert!(kernel.source.starts_with("enable f16;\n"));
        assert!(kernel.source.contains("tanh("));
        assert!(kernel.source.contains("f16(in0[off0])"), "operand read must cast f32 down to f16");
        assert!(kernel.source.contains("out[gid] = f32("), "output write must cast f16 back up to f32");
    }

    #[test]
    fn a_bfloat16_node_collapses_to_f32_compute() {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::BFloat16,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::BFloat16,
                body: ScalarOp::Tanh,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
        let bound = bound.into_iter().next().expect("one bound op");
        let kernel = emit_wgsl(&bound, WgslCaps::default(), &PackedOperands::new()).expect("bf16 collapses to f32 unconditionally");
        assert!(!kernel.source.contains("enable f16"));
        assert!(kernel.source.contains("var scratch: array<f32, 1>"));
    }

    /// `sum_k lhs[m, k] * rhs[k, n]` — the matmul shape `reduce_is_cooperative`
    /// selects for.
    fn matmul_reduce_op(m: u32, k: u32, n: u32) -> BoundOp {
        use proxima_tensor::{Reduce, ReduceInit};

        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(m), Extent::Static(k)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k), Extent::Static(n)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matmul".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
        bound.into_iter().next_back().expect("one bound op (the reduce)")
    }

    #[test]
    fn a_cooperative_reduce_renders_subgroup_builtins_and_a_matching_workgroup_size() {
        let bound = matmul_reduce_op(4, 37, 3);
        let caps = WgslCaps {
            subgroup_size: Some(32),
            ..WgslCaps::default()
        };
        let kernel = emit_wgsl(&bound, caps, &PackedOperands::new()).expect("cooperative reduce emits");
        assert!(kernel.source.contains("@workgroup_size(32)"));
        assert!(kernel.source.contains("@builtin(subgroup_invocation_id) lane: u32"));
        assert!(kernel.source.contains("subgroupAdd(accumulator)"));
        assert_eq!(kernel.workgroup_size, 32);
        // one whole subgroup dispatched per output element (m * n = 12).
        assert_eq!(kernel.threads, 12 * 32);
    }

    #[test]
    fn without_a_confirmed_subgroup_width_the_same_reduce_stays_serial() {
        let bound = matmul_reduce_op(4, 37, 3);
        let kernel =
            emit_wgsl(&bound, WgslCaps::default(), &PackedOperands::new()).expect("serial reduce emits");
        assert!(!kernel.source.contains("subgroupAdd"));
        assert!(kernel.source.contains("@workgroup_size(64)"));
        assert_eq!(kernel.workgroup_size, WORKGROUP_SIZE);
        assert_eq!(kernel.threads, 12);
    }

    #[test]
    fn iota_emits_the_thread_coordinate_with_no_operand_bindings() {
        let mut program = Vec::new();
        append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(6),
            },
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
        let bound = bound.into_iter().next().expect("one bound op");
        let kernel = emit_wgsl(&bound, WgslCaps::default(), &PackedOperands::new()).expect("iota emits");
        assert!(kernel.source.contains("out[gid] = f32(gid);"));
        assert_eq!(kernel.bindings.len(), 2, "output + uniforms, no operand bindings");
        assert_eq!(kernel.threads, 6);
    }

    #[test]
    fn constant_emits_the_literal_with_no_operand_bindings() {
        let mut program = Vec::new();
        append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: vec![Extent::Static(4)],
                value: -1.0e9,
            },
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
        let bound = bound.into_iter().next().expect("one bound op");
        let kernel = emit_wgsl(&bound, WgslCaps::default(), &PackedOperands::new()).expect("constant emits");
        assert!(kernel.source.contains("out[gid] = -1000000000.0;"));
        assert_eq!(kernel.bindings.len(), 2, "output + uniforms, no operand bindings");
        assert_eq!(kernel.threads, 4);
    }

    #[test]
    fn a_nan_constant_emits_a_bitcast_not_a_bare_literal() {
        assert_eq!(wgsl_literal(f32::NAN), "bitcast<f32>(0x7fc00000u)");
        assert_eq!(wgsl_literal(f32::NEG_INFINITY), "bitcast<f32>(0xff800000u)");
        assert_eq!(wgsl_literal(f32::INFINITY), "bitcast<f32>(0x7f800000u)");
    }
}
