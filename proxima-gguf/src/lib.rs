//! A sans-IO reader for the GGUF model-weight container format.
//!
//! # Tier split
//!
//! The parser/writer core ([`parser`], [`reader`], [`types`], [`value`],
//! [`tensor`], [`error`], [`pipe`], [`sized`], [`quant`], [`writer`]) is
//! `no_std + alloc`: it operates on `&[u8]` chunks handed to it by
//! [`parser::GgufParser::push`] (or, for [`writer::write_complete`],
//! returns an owned `Vec<u8>`) and never performs IO. It compiles under
//! `--no-default-features` (this crate has no separate `alloc` feature
//! toggle to flip — alloc is the floor, `std` only adds [`edge`] and
//! [`config`]). [`sized`] holds the build-time floor constants
//! ([`sized::MAX_SUPPORTED_VERSION`], [`sized::DEFAULT_ALIGNMENT`]) that
//! [`parser::GgufParser::new`] and [`writer::write_complete`] both use;
//! [`config`]'s `GgufParserConfig` (std-only, conflaguration-backed) seeds
//! its runtime defaults from those same constants and can override them
//! per-process via [`parser::GgufParser::with_config`]. The [`edge`]
//! module is the other `std`-gated surface: a thin convenience that reads
//! a whole file into memory and hands its bytes to the parser. There is no
//! `mmap` anywhere in this crate — neither the parser nor the writer owns
//! how bytes get in front of / away from it; an mmap'd region, a `Vec<u8>`
//! from `std::fs::read` or `std::fs::write`, or bytes streamed over a
//! socket are all just `&[u8]` (or a `Vec<u8>` the writer hands back) to
//! this crate.
//!
//! # Layout source
//!
//! Read from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`,
//! present on this host): `ggml/include/gguf.h` (wire shape + type enum)
//! and `ggml/src/gguf.cpp` (`gguf_init_from_file_impl`, the exact
//! validation order this parser mirrors) plus the block-quantization
//! layouts in `ggml/src/ggml-common.h` and the `type_traits` table in
//! `ggml/src/ggml.c`. Specific `file:line` citations live on the types and
//! functions that used them.
//!
//! # Codecs
//!
//! [`quant`] unpacks/packs block-quantized tensor values ([`quant::q4_k`]
//! is the first format landed) — `f32 <-> packed bytes`, borrowed input,
//! caller-provided output, no allocation on the hot path. It is alloc-tier
//! like the rest of the parser core; only `F32`/`F16` had typed accessors
//! before this since those are direct reinterpretations, not
//! dequantization.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
pub mod config;
pub mod error;
pub mod parser;
pub mod pipe;
pub mod quant;
pub mod reader;
pub mod restack;
pub mod sized;
pub mod tensor;
pub mod types;
pub mod value;
pub mod writer;

#[cfg(feature = "std")]
pub mod edge;

pub use error::GgufError;
pub use parser::{GgufEvent, GgufParser};
pub use pipe::{ParsedGguf, parse_complete};
pub use restack::{
    RestackError, StackPlan, discover_experts, expert_tensor_name, gather_expert, plan_stack,
    restack_into,
};
pub use tensor::TensorInfo;
pub use types::{GgmlType, MetadataType};
pub use value::{MetadataArray, MetadataValue};
pub use writer::{GgufModel, TensorPayload, write_complete};

#[cfg(test)]
mod tests;
