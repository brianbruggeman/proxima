//! Header configuration for Gemma 4's checkpoint family (`general.architecture
//! = "gemma4"`), the reader the `sliding_pattern` schedule source names
//! ([`proxima_tensor::spec::ScheduleSource::SlidingPattern`]). `bind.rs` is a
//! descriptor over the generic
//! `proxima_tensor::spec::lfm2_forward_program_with_experts` engine -- there is
//! no bespoke gemma4 forward-graph module here.

mod bind;
pub mod hparams;
pub mod program;

pub use bind::descriptor_from_gguf;
pub(crate) use bind::header;
pub use hparams::{Architecture, from_metadata};
