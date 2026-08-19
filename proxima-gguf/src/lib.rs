//! A sans-IO reader for the GGUF model-weight container format.
//!
//! # Tier split
//!
//! The parser core ([`parser`], [`reader`], [`types`], [`value`], [`tensor`],
//! [`error`], [`pipe`]) is `no_std + alloc`: it operates on `&[u8]` chunks
//! handed to it by [`parser::GgufParser::feed`] and never performs IO. It
//! compiles under `--no-default-features` (this crate has no separate
//! `alloc` feature toggle to flip — alloc is the floor, `std` only adds
//! [`edge`]). The [`edge`] module is the only `std`-gated surface: a thin
//! convenience that reads a whole file into memory and hands its bytes to
//! the parser. There is no `mmap` anywhere in this crate — the parser never
//! owns how bytes get in front of it; an mmap'd region, a `Vec<u8>` from
//! `std::fs::read`, or bytes streamed off a socket are all just `&[u8]` to
//! [`parser::GgufParser::feed`].
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
//! # Out of scope
//!
//! Dequantizing packed tensor values (unpacking `Q4_K` et al. into
//! floats) is a separate, already-sized job. This crate reports each
//! tensor's [`types::GgmlType`] and byte range faithfully and hands back
//! raw bytes; only `F32`/`F16` get typed accessors since those are direct
//! reinterpretations, not dequantization.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod error;
pub mod parser;
pub mod pipe;
pub mod reader;
pub mod tensor;
pub mod types;
pub mod value;

#[cfg(feature = "std")]
pub mod edge;

pub use error::GgufError;
pub use parser::{GgufEvent, GgufParser, PollOutcome};
pub use pipe::{ParseComplete, ParsedGguf, parse_complete};
pub use tensor::TensorInfo;
pub use types::{GgmlType, MetadataType};
pub use value::{MetadataArray, MetadataValue};

#[cfg(test)]
mod tests;
