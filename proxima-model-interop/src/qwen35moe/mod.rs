//! Header and layer configuration for qwen3.6's hybrid SSM/MoE checkpoint.

pub mod execution;
mod header;
pub mod hparams;
pub mod layer_boundary;
mod program;

pub(crate) use header::header;
pub use hparams::{Qwen35MoeHparams, LayerKind, from_metadata};
pub use program::{
    Qwen35MoeForwardProgram, Qwen35MoeLayerDiagnostics, descriptor_from_architecture, qwen35moe_forward_program,
    qwen35moe_forward_program_at_width,
};
