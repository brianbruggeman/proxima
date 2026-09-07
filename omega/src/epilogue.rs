//! The [`ComposedBody`] step-declaration walk every renderer
//! ([`crate::msl`], [`crate::wgsl`], [`crate::cuda`]) does identically: read
//! each step's args (an earlier step or an operand slot), render the step's
//! [`ScalarOp`] through the caller's own expression table, and declare the
//! result under the caller's own `step{n}` naming and syntax.
//!
//! Pulled out once ROW 294 found the same loop duplicated three times over —
//! `crate::msl::push_body_steps`/`push_epilogue_body_steps`,
//! `crate::wgsl::push_body_steps`, `crate::cuda::push_body_steps` all walked
//! `body.steps` and matched [`StepArg::Operand`]/[`StepArg::Step`]
//! identically. What differs per language is never the walk: it is the
//! `ScalarOp` -> expression text table (MSL's ternary vs WGSL's `select` vs
//! CUDA's `fmaxf`/`fminf`) and the declaration syntax (`{type} step{n} = ...;`
//! in MSL/CUDA C vs `let step{n}: {type} = ...;` in WGSL). Both stay
//! per-language, passed in as closures, so this module has no opinion on
//! either.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::{ComposedBody, ScalarOp, StepArg};

/// Declares one step per [`ComposedBody`] entry, in order, returning the
/// declared name of the body's own result (its last step). `operand_prefix`
/// names the array a [`StepArg::Operand`] indexes into (`"scratch"` for an
/// element body, `"epi_scratch"` for a fused reduce epilogue); `step_prefix`
/// names this walk's own declared values (`"step"`/`"epi_step"`) so an
/// epilogue's steps never collide with the fold's own `step{n}` slots even
/// when a real operand index matches.
///
/// `scalar_op_expr` renders one [`ScalarOp`] over already-named argument
/// strings into the target language's expression syntax.
/// `declare_step` emits the actual `{index}`-th declaration statement given
/// that expression — the one piece of true per-language syntax (the type
/// annotation position, `let` vs bare declaration, the trailing indent).
pub(crate) fn declare_steps(
    source: &mut String,
    body: &ComposedBody,
    operand_prefix: &str,
    step_prefix: &str,
    mut scalar_op_expr: impl FnMut(ScalarOp, &[&str]) -> String,
    mut declare_step: impl FnMut(&mut String, usize, &str),
) -> String {
    for (index, step) in body.steps.iter().enumerate() {
        let args: Vec<String> = step
            .args
            .iter()
            .map(|arg| match arg {
                StepArg::Operand(operand_index) => format!("{operand_prefix}[{operand_index}]"),
                StepArg::Step(step_index) => format!("{step_prefix}{step_index}"),
            })
            .collect();
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let expr = scalar_op_expr(step.op, &arg_refs);
        declare_step(source, index, &expr);
    }
    format!("{step_prefix}{}", body.steps.len().saturating_sub(1))
}

/// [`declare_steps`]'s split form for a body rendered inside a per-lane write
/// loop (ROW 370): some steps depend only on operands that do not vary across
/// loop iterations (a folded reduce scalar, a broadcast constant) while
/// others depend on an operand read at the current iteration's coordinate
/// (`x[s,d]`, `gamma[d]`). Re-declaring the whole chain every iteration wastes
/// the invariant half's arithmetic 16x over for a [1,4096]-shaped fold at
/// cooperative width 256. This walk classifies each step by
/// `is_operand_invariant` (transitively through [`StepArg::Step`]) and routes
/// its declaration to `declare_invariant_step` (emitted once, before the
/// loop) or `declare_element_step` (emitted once per iteration, inside the
/// loop) accordingly. The caller still owns loop structure and operand reads;
/// this only decides where each step's declaration text lands.
#[allow(clippy::too_many_arguments)]
pub(crate) fn declare_steps_partitioned(
    invariant_source: &mut String,
    element_source: &mut String,
    body: &ComposedBody,
    operand_prefix: &str,
    step_prefix: &str,
    is_operand_invariant: impl Fn(usize) -> bool,
    mut scalar_op_expr: impl FnMut(ScalarOp, &[&str]) -> String,
    mut declare_invariant_step: impl FnMut(&mut String, usize, &str),
    mut declare_element_step: impl FnMut(&mut String, usize, &str),
) -> String {
    let mut invariant = alloc::vec![false; body.steps.len()];
    for (index, step) in body.steps.iter().enumerate() {
        let step_is_invariant = step.args.iter().all(|arg| match arg {
            StepArg::Operand(operand_index) => is_operand_invariant(usize::from(*operand_index)),
            StepArg::Step(step_index) => invariant[usize::from(*step_index)],
        });
        invariant[index] = step_is_invariant;
        let args: Vec<String> = step
            .args
            .iter()
            .map(|arg| match arg {
                StepArg::Operand(operand_index) => format!("{operand_prefix}[{operand_index}]"),
                StepArg::Step(step_index) => format!("{step_prefix}{step_index}"),
            })
            .collect();
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        let expr = scalar_op_expr(step.op, &arg_refs);
        if step_is_invariant {
            declare_invariant_step(invariant_source, index, &expr);
        } else {
            declare_element_step(element_source, index, &expr);
        }
    }
    format!("{step_prefix}{}", body.steps.len().saturating_sub(1))
}
