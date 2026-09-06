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
