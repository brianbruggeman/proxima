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
//!   — see [`PROXIMA_ERF_FN_WGSL`]'s own doc.
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
//!   [`render_scan`]'s own doc). Not embarrassingly parallel the way a
//!   reduce is; v1 does not attempt a parallel prefix-sum reformulation.
//! - **`f32` only.** `Float16`/`BFloat16` (and everything [`type_token`]
//!   otherwise rejects) fail with [`EmitError::UnsupportedDType`] — WGSL's
//!   base spec has no portable narrow float type every wgpu backend
//!   supports, unlike MSL's native `half`.
//!
//! # Not in v1 (see [`EmitError::GatherNotSupported`] /
//! [`EmitError::UnsupportedOpKind`])
//!
//! - **Gather.** No indices binding, no fault buffer, no atomic
//!   out-of-range reporting — `crate::msl::push_gather_fetch`'s whole
//!   mechanism has no counterpart here yet. A `BoundOp` whose operand
//!   carries a [`proxima_tensor::Lookup`] is rejected at [`emit_wgsl`]
//!   rather than silently reading out of bounds.
//! - **`Iota`/`Constant`.** Rejected the same way; nothing in this crate's
//!   v1 test surface needs a position-only or literal-only kernel.
//! - **Quantized/packed operands.** [`emit_wgsl`] assumes every operand is a
//!   plain `f32` array — there is no [`crate::msl::PackedCodec`] table here.
//!   `crate::wgpu_driver`'s upload path is what actually enforces this: it
//!   accepts `QuantizedBlock::Float32` alone and rejects every packed codec
//!   with a named error rather than attempting a CPU-side dequantize (see
//!   that module's own doc).
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
use crate::msl::Binding;

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
}

/// Emits a WGSL kernel from a bound [`BoundOp`] — see the module doc for
/// exactly which op shapes this covers in v1.
///
/// # Errors
/// [`EmitError::UnsupportedDType`] for anything but `Float32`,
/// [`EmitError::GatherNotSupported`] for a gathered operand,
/// [`EmitError::UnsupportedOpKind`] for `Iota`/`Constant`,
/// [`EmitError::ArityMismatch`]/[`EmitError::ReductionBodyIsSelect`]/
/// [`EmitError::EmptyScan`] for the same structural failures
/// [`crate::msl::emit`] rejects.
pub fn emit_wgsl(resolved: &BoundOp) -> Result<WgslKernel, EmitError> {
    validate(resolved)?;
    let entry = entry_name(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let source = match &resolved.kind {
        BoundOpKind::Elementwise { .. } => render_elementwise(resolved, &entry, element_type),
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => render_reduce(resolved, &entry, element_type),
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => render_scan(resolved, &entry, element_type),
        BoundOpKind::Iota => {
            return Err(EmitError::UnsupportedOpKind {
                node: resolved.node,
                kind: "iota",
            });
        }
        BoundOpKind::Constant { .. } => {
            return Err(EmitError::UnsupportedOpKind {
                node: resolved.node,
                kind: "constant",
            });
        }
    };
    Ok(WgslKernel {
        source,
        entry,
        bindings: bindings(resolved),
        threads: grid_threads(resolved),
    })
}

fn type_token(node: NodeId, dtype: DType) -> Result<&'static str, EmitError> {
    match dtype {
        DType::Float32 => Ok("f32"),
        other => Err(EmitError::UnsupportedDType { node, dtype: other }),
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
    for (_, _, gather) in resolved.operands() {
        if gather.is_some() {
            return Err(EmitError::GatherNotSupported { node: resolved.node });
        }
    }
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

fn bindings(resolved: &BoundOp) -> Vec<Binding> {
    let mut bindings: Vec<Binding> = resolved
        .operands()
        .iter()
        .map(|(node, _, _)| Binding::Input(*node))
        .collect();
    bindings.push(Binding::Output(resolved.node));
    bindings.push(Binding::Uniforms);
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

fn preamble(source: &mut String, operand_count: usize, output_len: usize, uniforms_struct: &str) {
    source.push_str(PROXIMA_ERF_FN_WGSL);
    source.push('\n');
    source.push_str(uniforms_struct);
    source.push('\n');
    for index in 0..operand_count {
        source.push_str(&format!(
            "@group(0) @binding({index}) var<storage, read> in{index}: array<f32>;\n"
        ));
    }
    source.push_str(&format!(
        "@group(0) @binding({operand_count}) var<storage, read_write> out: array<f32>;\n"
    ));
    let _ = output_len;
    source.push_str(&format!(
        "@group(0) @binding({}) var<storage, read> u: Uniforms;\n\n",
        operand_count + 1
    ));
}

fn kernel_signature(source: &mut String, entry: &str) {
    source.push_str(&format!("@compute @workgroup_size({WORKGROUP_SIZE})\n"));
    source.push_str(&format!(
        "fn {entry}(@builtin(global_invocation_id) global_id: vec3<u32>) {{\n"
    ));
    source.push_str("    let gid: i32 = i32(global_id.x);\n");
}

fn render_elementwise(resolved: &BoundOp, entry: &str, element_type: &str) -> String {
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();

    let mut uniforms = String::new();
    uniforms.push_str("struct Uniforms {\n");
    uniforms.push_str("    total_elements: i32,\n");
    uniforms.push_str(&format!("    extents: array<i32, {rank_len}>,\n"));
    uniforms.push_str(&format!("    operand_base: array<i32, {operand_count}>,\n"));
    uniforms.push_str(&format!(
        "    operand_strides: array<array<i32, {rank_len}>, {operand_count}>,\n"
    ));
    uniforms.push_str("};\n");

    let mut source = String::new();
    preamble(&mut source, operand_count, 0, &uniforms);
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

    for index in 0..operand_count {
        source.push_str(&format!("    var off{index}: i32 = u.operand_base[{index}];\n"));
        for dim in 0..rank {
            source.push_str(&format!(
                "    off{index} += coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
    }

    source.push_str(&format!(
        "    var scratch: array<{element_type}, {}>;\n",
        operand_count.max(1)
    ));
    for index in 0..operand_count {
        source.push_str(&format!("    scratch[{index}] = in{index}[off{index}];\n"));
    }

    let result = push_body_steps(&mut source, resolved.element_body(), "    ", element_type);
    source.push_str(&format!("    out[gid] = {result};\n"));
    source.push_str("}\n");
    source
}

fn render_reduce(resolved: &BoundOp, entry: &str, element_type: &str) -> String {
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
    preamble(&mut source, operand_count, 0, &uniforms);
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
        source.push_str(&format!("        scratch[{index}] = in{index}[off{index}];\n"));
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
    source.push_str("    out[out_offset] = accumulator;\n");
    source.push_str("}\n");
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
    preamble(&mut source, operand_count, 0, &uniforms);
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
        source.push_str(&format!("            scratch[{index}] = in{index}[running{index}];\n"));
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
    source.push_str("            out[out_running] = accumulator;\n");
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
        let kernel = emit_wgsl(&bound).expect("emit succeeds");
        assert!(kernel.source.contains("@compute"));
        assert!(kernel.source.contains("tanh("));
        assert_eq!(kernel.bindings.len(), 3);
        assert_eq!(kernel.threads, 8);
    }

    #[test]
    fn same_structure_different_extents_yield_identical_source() {
        let small = emit_wgsl(&elementwise_tanh_op(4)).expect("emit succeeds");
        let large = emit_wgsl(&elementwise_tanh_op(4096)).expect("emit succeeds");
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
        let kernel = emit_wgsl(&bound).expect("emit succeeds");
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
        let error = emit_wgsl(&bound).expect_err("f16 is rejected in v1");
        assert!(matches!(error, EmitError::UnsupportedDType { .. }));
    }
}
