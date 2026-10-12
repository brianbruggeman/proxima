//! The Core ML edge of the door: the calls that leave the sans-IO pipes.
//!
//! [`compile`] is the Verify stage's edge: it runs `xcrun coremlcompiler` on
//! the [`MilPackage`](proxima_mil::writer::MilPackage) the writing stages
//! produced, as `plan_named` is the edge to Metal for the Plan stage.
//!
//! [`sink`] is the door's [`Writer`](crate::door::Writer) that calls it: a
//! writer refusal is the door's Emit error, a compiler refusal its Verify error.

pub mod compile;
pub mod sink;
