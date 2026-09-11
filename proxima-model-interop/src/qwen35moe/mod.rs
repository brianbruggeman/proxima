//! Header and layer configuration for qwen3.6's hybrid SSM/MoE checkpoint.

mod bind;
pub mod execution;
pub mod hparams;
pub mod layer_boundary;
mod program;
mod shared_expert;

pub use bind::{QWEN35MOE, Qwen35MoeArch, bind_qwen35moe_weights, qwen35moe_tensor_names};
pub use hparams::{Architecture, LayerKind, from_metadata};
pub use program::{
    Qwen35MoeForwardProgram, Qwen35MoeGdnPrefillTaps, Qwen35MoeLayerDiagnostics,
    qwen35moe_forward_program,
};
