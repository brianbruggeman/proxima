//! A sans-IO byte-level BPE tokenizer.
//!
//! # Tier split
//!
//! The core ([`byte_level`], [`vocab`], [`bpe`], [`pretokenize`],
//! [`pipe`], [`error`], [`sized`]) is `no_std + alloc`: it never opens a
//! file and never performs IO. It compiles under `--no-default-features`
//! (this crate has no separate `alloc` feature toggle to flip -- alloc is
//! the floor, `std` only adds [`config`]). [`sized`] holds the build-time
//! floor constant ([`sized::MAX_INPUT_BYTES`]); [`config`]'s
//! `TokenizerConfig` (std-only, conflaguration-backed) seeds its runtime
//! default from that same constant.
//!
//! # Getting a vocab
//!
//! [`Vocab::new`] takes the token list, merge rules, and special token ids
//! directly -- the sans-IO core has no opinion on where those came from.
//! The `gguf` feature adds [`gguf::vocab_from_metadata`], which reads them
//! out of a [`proxima_gguf::ParsedGguf`]'s metadata (the real key names,
//! confirmed against a live GGUF fixture, are documented there).
//!
//! # Two encoders, selected by what the vocab declares
//!
//! Byte-level BPE over the GPT-2 alphabet ([`byte_level`]), with the
//! LLAMA3 pretokenizer ([`pretokenize`]) -- the variant `tokenizer.ggml.
//! model = "gpt2"` / `tokenizer.ggml.pre = "llama-bpe"` identify on the
//! real fixture this crate was built against
//! (`~/repos/others/llama.cpp/models/ggml-vocab-llama-bpe.gguf`).
//!
//! SentencePiece/SPM ([`unigram`]) for `tokenizer.ggml.model = "llama"`
//! vocabs (`tokenizer.ggml.scores` present, no `tokenizer.ggml.merges`),
//! confirmed against a real openchat-3.5-1210 fixture. [`pipe::encode`]/
//! [`pipe::decode`] pick the encoder from [`vocab::Vocab::is_unigram`] --
//! never a caller flag.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod bpe;
pub mod byte_level;
#[cfg(feature = "std")]
pub mod config;
pub mod error;
#[cfg(feature = "gguf")]
pub mod gguf;
pub mod pipe;
pub mod pretokenize;
pub mod sample;
pub mod sized;
pub mod unigram;
pub mod vocab;

pub use error::TokenizerError;
pub use pipe::{decode, encode, encode_with_bos_eos};
pub use sample::greedy_pick;
pub use vocab::{TokenType, Vocab};

#[cfg(test)]
mod tests;
