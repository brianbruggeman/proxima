//! The Core ML edge of the door: the calls that leave the sans-IO pipes.
//!
//! [`compile`] is the Verify stage's edge: it runs `xcrun coremlcompiler` on
//! the [`MilPackage`](proxima_mil::writer::MilPackage) the writing stages
//! produced, as `plan_named` is the edge to Metal for the Plan stage.

pub mod compile;
