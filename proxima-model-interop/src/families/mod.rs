//! Architecture families as data, inferred from a checkpoint (PROXNUM-611).
//!
//! A [`FamilyTemplate`] is a graph template with slots; [`infer`] matches a
//! checkpoint's tensor names and shapes to exactly one template and fills the
//! slots, with `config.json` as the second source and a cited default as the
//! last. Every value carries its [`Source`]. Inference is a hypothesis: the
//! golden gate (`accept`) is what turns it into a model.

mod infer;
mod pattern;
mod template;

pub use infer::{
    CheckpointSha, Fact, InferError, Inferred, ShaError, Source, TensorShape, Unmet, infer,
};
pub use pattern::{Pattern, PatternError};
pub use template::{Dim, FamilyTemplate, Knob, Probe, Stack, TemplateError, Value};

#[cfg(test)]
mod tests;
