//! Header configuration for the checkpoint family whose `general.architecture`
//! selects the `sliding_pattern` schedule source, the reader that source names
//! ([`proxima_tensor::spec::ScheduleSource::SlidingPattern`]). `bind.rs` is a
//! descriptor over the generic
//! `proxima_tensor::spec::lfm2_forward_program_with_experts` engine -- there is
//! no bespoke sliding-pattern forward-graph module here.

mod bind;
pub mod hparams;
pub mod program;

pub use bind::descriptor_from_gguf;
pub(crate) use bind::header;
pub use hparams::{Gemma4Hparams, from_metadata};
