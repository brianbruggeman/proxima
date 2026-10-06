//! Header configuration and registration for Gemma 4's mixture-of-experts
//! checkpoint family (`general.architecture = "gemma4"`). `Gemma4Arch::bind`
//! (`bind.rs`) is a descriptor over the generic
//! `proxima_tensor::spec::lfm2_forward_program_with_experts` engine -- there is
//! no bespoke gemma4 forward-graph module here.

mod bind;
pub mod hparams;
pub mod program;

pub use bind::{GEMMA4, Gemma4Arch, bind_gemma4_all_positions_logits};
#[cfg(feature = "std")]
pub use bind::descriptor_from_gguf;
pub use hparams::{Architecture, from_metadata};
