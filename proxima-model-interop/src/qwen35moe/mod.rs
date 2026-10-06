//! Header and layer configuration for qwen3.6's hybrid SSM/MoE checkpoint.

mod bind;
pub mod execution;
pub mod hparams;
pub mod layer_boundary;
mod program;

pub use bind::{QWEN35MOE, Qwen35MoeArch};
pub use hparams::{Architecture, LayerKind, from_metadata};
pub use program::{
    Qwen35MoeForwardProgram, Qwen35MoeLayerDiagnostics, descriptor_from_architecture, qwen35moe_forward_program,
    qwen35moe_forward_program_at_width,
};
