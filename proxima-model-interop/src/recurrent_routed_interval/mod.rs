//! Header and layer configuration for qwen3.6's hybrid SSM/MoE checkpoint.

pub mod execution;
mod header;
pub mod hparams;
pub mod layer_boundary;
mod program;

pub(crate) use header::header;
pub use hparams::{RecurrentRoutedIntervalHparams, LayerKind, from_metadata};
pub use program::{
    MoeForwardProgram, MoeLayerDiagnostics, descriptor_from_architecture, recurrent_routed_interval_forward_program,
    recurrent_routed_interval_forward_program_at_width,
};
