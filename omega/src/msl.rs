//! Metal Shading Language kernel emission.
//!
//! [`emit`] turns one lowered [`Nest`] into one [`Kernel`]: MSL source text,
//! an entry name, the buffer-index -> data [`Binding`] list a driver needs to
//! set up a dispatch, and the thread-count [`GridSpec`] for this invocation.
//!
//! # Runtime uniforms, not baked constants
//!
//! A `Nest`'s extents and strides are read out of a `constant Uniforms&`
//! buffer at kernel runtime — never spliced into the source text as literal
//! numbers. What *does* vary the source is `nest`'s STRUCTURE: rank (operand
//! and output coordinate arity), operand count, which [`ScalarOp`]s the body
//! and (if present) the reduction use, and whether a reduction is present at
//! all and which [`Keep`] it is. Two `Nest`s that agree on structure but
//! differ in concrete extents, strides, or which buffers they bind therefore
//! emit byte-identical source — see
//! `same_structure_different_extents_yield_identical_source` below for the
//! proof. This is what makes a kernel cacheable (and an `MTLLibrary`
//! reusable) by structure rather than by nest.
//!
//! # Execution model (v1: correctness parity with `cpu.rs`, not peak speed)
//!
//! - **Elementwise** (no reduction): one thread per output element. A
//!   thread's linear id decodes into a coordinate via the same row-major
//!   div/mod chain `cpu::unflatten` uses, each operand's read offset is
//!   `base + sum(coord[d] * stride[d])`, and the body writes directly to the
//!   dense output at its own linear id — matching `cpu::run_elementwise`.
//! - **Fused fold, `Keep::Last`** (reduce): one thread per OUTPUT element
//!   (matmul is one thread per `(i, j)`), with a serial loop over the
//!   reduction dims inside the kernel. `FoldInit` seeding — including
//!   `FirstElement`'s seed-on-first-step behavior — matches
//!   `cpu::run_reduce` exactly: the accumulator is seeded from the *first*
//!   reduction step's value rather than combined into an `init` constant.
//! - **`Keep::All`** (scan): one thread per non-folded coordinate line,
//!   serial along the folded (innermost) dim, writing every prefix through
//!   the output strides — matching `cpu::run_scan`.
//!
//! # dtype
//!
//! `Nest` carries no dtype at all (see [`proxima_tensor::nest`]'s
//! documentation) — by the time a program reaches a `Nest` it has already
//! passed `cpu::reject_non_float32` upstream in every path this crate is
//! meant to consume, so every kernel here is generated in `float`
//! unconditionally, matching `cpu.rs`'s own f32-only v1 stance.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::{FoldInit, Keep, Nest, NodeId, Reduction, ScalarOp};

use crate::error::EmitError;

/// One compiled kernel: MSL source, its entry point, the buffer-index ->
/// data mapping a driver needs to bind before dispatch, and the thread count
/// this particular `nest` needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub source: String,
    pub entry: String,
    pub bindings: Vec<Binding>,
    pub grid: GridSpec,
}

/// What buffer index `n` in [`Kernel::bindings`] is for, in dispatch order:
/// index `0..operands.len()` are inputs, then the output, then the uniforms
/// blob (extents/strides/bases for this dispatch — see the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    Input(NodeId),
    Output(NodeId),
    Uniforms,
}

/// How many threads a driver must dispatch for this `nest` — one per
/// independent unit of work (output element for elementwise/reduce, output
/// line for a scan). Unlike [`Kernel::source`], this genuinely is a function
/// of `nest`'s concrete extents, not just its structure: it is per-dispatch
/// data, the same way an argument to a function call is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSpec {
    pub threads: u64,
}

/// Emits an MSL kernel from a lowered [`Nest`] — the GPU-emission half of the
/// same descriptor [`proxima_tensor::cpu`] interprets on CPU. See the module
/// doc for the runtime-uniforms stance and the per-[`Keep`] execution model.
///
/// # Examples
///
/// ```
/// use proxima_tensor::{DType, Expr, Extent, IndexMap, ScalarOp, append, map};
///
/// let mut program = Vec::new();
/// let source = append(
///     &mut program,
///     Expr::Block {
///         dtype: DType::Float32,
///         shape: vec![Extent::Static(4)],
///         name: None,
///     },
/// );
/// append(
///     &mut program,
///     Expr::Zip {
///         dtype: DType::Float32,
///         body: ScalarOp::Tanh,
///         operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
///         name: None,
///     },
/// );
///
/// let shapes = proxima_tensor::infer(&program, &[])?;
/// let nests = proxima_tensor::lower(&program, &shapes, &[])?;
///
/// let kernel = omega::emit(&nests[0])?;
/// assert!(kernel.source.contains("kernel void"));
/// assert!(kernel.source.contains("tanh("));
/// assert_eq!(kernel.bindings.len(), 3); // one input, one output, uniforms
/// assert_eq!(kernel.grid.threads, 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn emit(nest: &Nest) -> Result<Kernel, EmitError> {
    validate(nest)?;
    let entry = entry_name(nest);
    let source = match &nest.reduction {
        None => render_elementwise(nest, &entry),
        Some(reduction) => match reduction.keep {
            Keep::Last => render_reduce(nest, reduction, &entry),
            Keep::All => render_scan(nest, reduction, &entry),
        },
    };
    Ok(Kernel {
        source,
        entry,
        bindings: bindings(nest),
        grid: GridSpec {
            threads: grid_threads(nest),
        },
    })
}

fn validate(nest: &Nest) -> Result<(), EmitError> {
    let expected = nest.body.arity();
    let found = nest.operands.len();
    if expected != found {
        return Err(EmitError::ArityMismatch {
            node: nest.node,
            expected,
            found,
        });
    }
    if let Some(reduction) = &nest.reduction {
        if matches!(reduction.body, ScalarOp::Select) {
            return Err(EmitError::ReductionBodyIsSelect { node: nest.node });
        }
        if reduction.keep == Keep::All && nest.extents.is_empty() {
            return Err(EmitError::EmptyScan { node: nest.node });
        }
    }
    Ok(())
}

/// `pub(crate)`, not private: the Metal driver's uniforms packer
/// (`crate::metal::pack_reduce_uniforms`) needs the exact same reduce-dim set
/// this rendering uses, and duplicating the filter would risk the two
/// drifting apart.
pub(crate) fn reduction_dims(nest: &Nest, output_dims: &[u16]) -> Vec<u16> {
    (0..nest.extents.len() as u16)
        .filter(|dim| !output_dims.contains(dim))
        .collect()
}

fn bindings(nest: &Nest) -> Vec<Binding> {
    let mut bindings: Vec<Binding> = nest
        .operands
        .iter()
        .map(|(node, _)| Binding::Input(*node))
        .collect();
    bindings.push(Binding::Output(nest.node));
    bindings.push(Binding::Uniforms);
    bindings
}

/// Total independent units of work `nest` needs — see [`GridSpec`]'s doc for
/// why this, unlike [`Kernel::source`], is genuinely a function of `nest`'s
/// concrete extents.
fn grid_threads(nest: &Nest) -> u64 {
    match &nest.reduction {
        None => nest.extents.iter().product(),
        Some(reduction) => match reduction.keep {
            Keep::Last => reduction
                .output_dims
                .iter()
                .map(|dim| nest.extents[*dim as usize])
                .product(),
            Keep::All => {
                let rank = nest.extents.len();
                nest.extents[..rank.saturating_sub(1)].iter().product()
            }
        },
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
        ScalarOp::Greater => "greater",
        ScalarOp::Equal => "equal",
        ScalarOp::Select => "select",
    }
}

fn init_token(init: FoldInit) -> &'static str {
    match init {
        FoldInit::Zero => "zero",
        FoldInit::One => "one",
        FoldInit::NegativeInfinity => "negative_infinity",
        FoldInit::PositiveInfinity => "positive_infinity",
        FoldInit::FirstElement => "first_element",
    }
}

fn keep_token(keep: Keep) -> &'static str {
    match keep {
        Keep::Last => "reduce",
        Keep::All => "scan",
    }
}

/// A structural fingerprint, not a hash of anything runtime: rank, operand
/// count, and every `ScalarOp`/`FoldInit`/`Keep` involved, which is exactly
/// the set of things [`emit`]'s source text depends on.
fn entry_name(nest: &Nest) -> String {
    let rank = nest.extents.len();
    let operand_count = nest.operands.len();
    let body = op_token(nest.body);
    match &nest.reduction {
        None => format!("omega_elementwise_r{rank}_n{operand_count}_{body}"),
        Some(reduction) => {
            let kind = keep_token(reduction.keep);
            let reduce_body = op_token(reduction.body);
            let init = init_token(reduction.init);
            format!("omega_{kind}_r{rank}_n{operand_count}_{body}_{reduce_body}_{init}")
        }
    }
}

fn scalar_op_expr(op: ScalarOp, args: &[&str]) -> String {
    match op {
        ScalarOp::Identity => (*args.first().unwrap_or(&"0.0f")).into(),
        ScalarOp::Add => format!("({} + {})", args[0], args[1]),
        ScalarOp::Subtract => format!("({} - {})", args[0], args[1]),
        ScalarOp::Multiply => format!("({} * {})", args[0], args[1]),
        ScalarOp::Divide => format!("({} / {})", args[0], args[1]),
        ScalarOp::Maximum => format!("max({}, {})", args[0], args[1]),
        ScalarOp::Minimum => format!("min({}, {})", args[0], args[1]),
        ScalarOp::Negate => format!("(-{})", args[0]),
        ScalarOp::Reciprocal => format!("(1.0f / {})", args[0]),
        ScalarOp::Exponential => format!("exp({})", args[0]),
        ScalarOp::Logarithm => format!("log({})", args[0]),
        ScalarOp::SquareRoot => format!("sqrt({})", args[0]),
        ScalarOp::Tanh => format!("tanh({})", args[0]),
        ScalarOp::Greater => format!("(({} > {}) ? 1.0f : 0.0f)", args[0], args[1]),
        ScalarOp::Equal => format!("((fabs({} - {}) == 0.0f) ? 1.0f : 0.0f)", args[0], args[1]),
        ScalarOp::Select => format!("(({} != 0.0f) ? {} : {})", args[0], args[1], args[2]),
    }
}

/// `(init expression, seeded-from-the-start)`. `FirstElement` mirrors
/// `cpu::initial_value`/`cpu::run_reduce`'s `seeded` flag: the accumulator
/// starts unseeded and is instead set from the first reduction step's value —
/// the init expression here is never actually read in that case.
fn fold_init_tokens(init: FoldInit) -> (&'static str, &'static str) {
    match init {
        FoldInit::Zero => ("0.0f", "true"),
        FoldInit::One => ("1.0f", "true"),
        FoldInit::NegativeInfinity => ("-INFINITY", "true"),
        FoldInit::PositiveInfinity => ("INFINITY", "true"),
        FoldInit::FirstElement => ("0.0f", "false"),
    }
}

fn scratch_args(operand_count: usize) -> Vec<String> {
    (0..operand_count)
        .map(|index| format!("scratch[{index}]"))
        .collect()
}

fn kernel_signature(source: &mut String, operand_count: usize, entry: &str) {
    source.push_str(&format!("kernel void {entry}(\n"));
    for index in 0..operand_count {
        source.push_str(&format!(
            "    device const float* in{index} [[buffer({index})]],\n"
        ));
    }
    source.push_str(&format!(
        "    device float* out [[buffer({operand_count})]],\n"
    ));
    source.push_str(&format!(
        "    constant Uniforms& u [[buffer({})]],\n",
        operand_count + 1
    ));
    source.push_str("    uint gid [[thread_position_in_grid]])\n{\n");
}

fn preamble(source: &mut String) {
    source.push_str("#include <metal_stdlib>\n");
    source.push_str("using namespace metal;\n\n");
}

fn render_elementwise(nest: &Nest, entry: &str) -> String {
    let rank = nest.extents.len();
    let rank_len = rank.max(1);
    let operand_count = nest.operands.len();

    let mut source = String::new();
    preamble(&mut source);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long total_elements;\n");
    source.push_str(&format!("    long extents[{rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!(
        "    long operand_strides[{operand_count}][{rank_len}];\n"
    ));
    source.push_str("};\n\n");

    kernel_signature(&mut source, operand_count, entry);
    source.push_str("    if ((long)gid >= u.total_elements) { return; }\n");

    if rank > 0 {
        source.push_str(&format!("    long coord[{rank_len}];\n"));
        source.push_str("    long remaining = (long)gid;\n");
        for dim in (0..rank).rev() {
            source.push_str(&format!(
                "    coord[{dim}] = remaining % u.extents[{dim}]; remaining /= u.extents[{dim}];\n"
            ));
        }
    }

    for index in 0..operand_count {
        source.push_str(&format!("    long off{index} = u.operand_base[{index}];\n"));
        for dim in 0..rank {
            source.push_str(&format!(
                "    off{index} += coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
    }

    source.push_str("    float scratch[3];\n");
    for index in 0..operand_count {
        source.push_str(&format!("    scratch[{index}] = in{index}[off{index}];\n"));
    }

    let args = scratch_args(operand_count);
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let expr = scalar_op_expr(nest.body, &arg_refs);
    source.push_str(&format!("    out[gid] = {expr};\n"));
    source.push_str("}\n");
    source
}

fn render_reduce(nest: &Nest, reduction: &Reduction, entry: &str) -> String {
    let rank = nest.extents.len();
    let rank_len = rank.max(1);
    let operand_count = nest.operands.len();
    let output_dims = &reduction.output_dims;
    let output_rank = output_dims.len();
    let output_rank_len = output_rank.max(1);
    let reduce_dims = reduction_dims(nest, output_dims);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);

    let mut source = String::new();
    preamble(&mut source);

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
    source.push_str("};\n\n");

    kernel_signature(&mut source, operand_count, entry);
    source.push_str("    if ((long)gid >= u.output_total) { return; }\n");

    source.push_str(&format!("    long full_coord[{rank_len}];\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }

    if output_rank > 0 {
        source.push_str(&format!("    long output_coord[{output_rank_len}];\n"));
        source.push_str("    long remaining = (long)gid;\n");
        for index in (0..output_rank).rev() {
            source.push_str(&format!(
                "    output_coord[{index}] = remaining % u.output_extents[{index}]; \
                 remaining /= u.output_extents[{index}];\n"
            ));
        }
        for (index, dim) in output_dims.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
    }

    let (init_expr, seeded_init) = fold_init_tokens(reduction.init);
    source.push_str(&format!("    float accumulator = {init_expr};\n"));
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

    for index in 0..operand_count {
        source.push_str(&format!(
            "        long off{index} = u.operand_base[{index}];\n"
        ));
        for dim in 0..rank {
            source.push_str(&format!(
                "        off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
    }
    source.push_str("        float scratch[3];\n");
    for index in 0..operand_count {
        source.push_str(&format!(
            "        scratch[{index}] = in{index}[off{index}];\n"
        ));
    }
    let args = scratch_args(operand_count);
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let value_expr = scalar_op_expr(nest.body, &arg_refs);
    source.push_str(&format!("        float value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(reduction.body, &["accumulator", "value"]);
    source.push_str(&format!(
        "        accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    source.push_str("    long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "    out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    source.push_str("    out[out_offset] = accumulator;\n");
    source.push_str("}\n");
    source
}

fn render_scan(nest: &Nest, reduction: &Reduction, entry: &str) -> String {
    let rank = nest.extents.len();
    let rank_len = rank.max(1);
    let outer_rank = rank.saturating_sub(1);
    let outer_rank_len = outer_rank.max(1);
    let last_dim = rank.saturating_sub(1);
    let operand_count = nest.operands.len();

    let mut source = String::new();
    preamble(&mut source);

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
    source.push_str("};\n\n");

    kernel_signature(&mut source, operand_count, entry);
    source.push_str("    if ((long)gid >= u.outer_total) { return; }\n");

    if outer_rank > 0 {
        source.push_str(&format!("    long outer_coord[{outer_rank_len}];\n"));
        source.push_str("    long remaining = (long)gid;\n");
        for dim in (0..outer_rank).rev() {
            source.push_str(&format!(
                "    outer_coord[{dim}] = remaining % u.outer_extents[{dim}]; \
                 remaining /= u.outer_extents[{dim}];\n"
            ));
        }
    }

    for index in 0..operand_count {
        source.push_str(&format!(
            "    long running{index} = u.operand_base[{index}];\n"
        ));
        for dim in 0..outer_rank {
            source.push_str(&format!(
                "    running{index} += outer_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
    }
    source.push_str("    long out_running = u.out_base;\n");
    for dim in 0..outer_rank {
        source.push_str(&format!(
            "    out_running += outer_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }

    let (init_expr, seeded_init) = fold_init_tokens(reduction.init);
    source.push_str(&format!("    float accumulator = {init_expr};\n"));
    source.push_str(&format!("    bool seeded = {seeded_init};\n"));

    source.push_str("    for (long step = 0; step < u.inner_len; step++) {\n");
    source.push_str("        float scratch[3];\n");
    for index in 0..operand_count {
        source.push_str(&format!(
            "        scratch[{index}] = in{index}[running{index}];\n"
        ));
        source.push_str(&format!(
            "        running{index} += u.operand_strides[{index}][{last_dim}];\n"
        ));
    }
    let args = scratch_args(operand_count);
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let value_expr = scalar_op_expr(nest.body, &arg_refs);
    source.push_str(&format!("        float value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(reduction.body, &["accumulator", "value"]);
    source.push_str(&format!(
        "        accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("        seeded = true;\n");
    source.push_str("        out[out_running] = accumulator;\n");
    source.push_str(&format!(
        "        out_running += u.out_strides[{last_dim}];\n"
    ));
    source.push_str("    }\n");
    source.push_str("}\n");
    source
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_tensor::{
        DType, Expr, Extent, Fold, FoldInit, IndexMap, Keep, ScalarOp, append, infer, lower, map,
    };

    use super::*;

    fn elementwise_tanh_nest(extent: u32) -> Nest {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("elementwise infers");
        lower(&program, &shapes, &[])
            .expect("elementwise lowers")
            .into_iter()
            .next()
            .expect("one nest emitted")
    }

    fn matmul_nest(m: u32, k: u32, n: u32) -> Nest {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: vec![Extent::Static(m), Extent::Static(k)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k), Extent::Static(n)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Expr::Zip {
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
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Last,
                name: Some("matmul".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("matmul infers");
        lower(&program, &shapes, &[])
            .expect("matmul lowers")
            .into_iter()
            .next()
            .expect("one fused nest emitted")
    }

    fn cumsum_nest(extent: u32) -> Nest {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        append(
            &mut program,
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[0])),
                keep: Keep::All,
                name: None,
            }),
        );
        let shapes = infer(&program, &[]).expect("cumsum infers");
        lower(&program, &shapes, &[])
            .expect("cumsum lowers")
            .into_iter()
            .next()
            .expect("one nest emitted")
    }

    #[test]
    fn elementwise_nest_emits_one_input_one_output_and_a_matching_grid() {
        let nest = elementwise_tanh_nest(10);
        let kernel = emit(&nest).expect("elementwise emits");

        assert_eq!(kernel.entry, "omega_elementwise_r1_n1_tanh");
        assert_eq!(
            kernel.bindings,
            vec![
                Binding::Input(nest.operands[0].0),
                Binding::Output(nest.node),
                Binding::Uniforms
            ]
        );
        assert!(
            kernel
                .source
                .contains("kernel void omega_elementwise_r1_n1_tanh")
        );
        assert!(kernel.source.contains("tanh(scratch[0])"));
        assert_eq!(kernel.grid.threads, 10);
    }

    #[test]
    fn fused_matmul_nest_emits_two_inputs_a_reduction_loop_and_a_row_by_col_grid() {
        let nest = matmul_nest(4, 3, 5);
        assert!(
            nest.reduction.is_some(),
            "zip must have fused into the fold"
        );
        let kernel = emit(&nest).expect("matmul emits");

        assert_eq!(kernel.entry, "omega_reduce_r3_n2_multiply_add_zero");
        assert_eq!(kernel.bindings.len(), 4, "two inputs, one output, uniforms");
        assert!(matches!(kernel.bindings[2], Binding::Output(_)));
        assert!(matches!(kernel.bindings[3], Binding::Uniforms));
        assert!(
            kernel
                .source
                .contains("kernel void omega_reduce_r3_n2_multiply_add_zero")
        );
        assert!(kernel.source.contains("reduction_total"));
        assert!(kernel.source.contains("(scratch[0] * scratch[1])"));
        assert!(kernel.source.contains("(accumulator + value)"));
        assert_eq!(kernel.grid.threads, 4 * 5, "one thread per (row, col)");
    }

    #[test]
    fn cumsum_nest_emits_a_scan_kernel_with_one_thread_per_line() {
        let nest = cumsum_nest(8);
        let kernel = emit(&nest).expect("cumsum emits");

        assert_eq!(kernel.entry, "omega_scan_r1_n1_identity_add_zero");
        assert!(kernel.source.contains("inner_len"));
        assert!(kernel.source.contains("out_running"));
        assert_eq!(
            kernel.grid.threads, 1,
            "no leading dims: a single scan line"
        );
    }

    #[test]
    fn emit_is_deterministic_byte_equal() {
        let nest = matmul_nest(4, 3, 5);
        let first = emit(&nest).expect("first emit succeeds");
        let second = emit(&nest).expect("second emit succeeds");
        assert_eq!(first, second);
    }

    #[test]
    fn same_structure_different_extents_yield_identical_source_but_different_grid() {
        let small = elementwise_tanh_nest(4);
        let large = elementwise_tanh_nest(4096);

        let small_kernel = emit(&small).expect("small emits");
        let large_kernel = emit(&large).expect("large emits");

        assert_eq!(small_kernel.source, large_kernel.source);
        assert_eq!(small_kernel.entry, large_kernel.entry);
        assert_ne!(small_kernel.grid.threads, large_kernel.grid.threads);
    }

    #[test]
    fn an_arity_mismatched_nest_is_rejected() {
        let mut nest = elementwise_tanh_nest(4);
        nest.body = ScalarOp::Add; // arity 2, but the nest still carries 1 operand

        let error = emit(&nest).expect_err("mismatched arity is rejected");
        assert!(matches!(error, EmitError::ArityMismatch { .. }), "{error}");
    }

    #[test]
    fn a_select_reduction_body_is_rejected() {
        let mut nest = matmul_nest(4, 3, 5);
        if let Some(reduction) = nest.reduction.as_mut() {
            reduction.body = ScalarOp::Select;
        }

        let error = emit(&nest).expect_err("select reduction body is rejected");
        assert!(
            matches!(error, EmitError::ReductionBodyIsSelect { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_keep_all_scan_over_zero_dims_is_rejected() {
        let mut nest = cumsum_nest(8);
        nest.extents.clear();
        if let Some(reduction) = nest.reduction.as_mut() {
            reduction.output_dims.clear();
        }

        let error = emit(&nest).expect_err("an empty scan is rejected");
        assert!(matches!(error, EmitError::EmptyScan { .. }), "{error}");
    }
}
