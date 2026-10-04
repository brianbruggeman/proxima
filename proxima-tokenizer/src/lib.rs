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
//! The `gguf` feature adds `gguf::vocab_from_metadata`, which reads them
//! out of a `proxima_gguf::ParsedGguf`'s metadata (the real key names,
//! confirmed against a live GGUF fixture, are documented there). Plain code
//! spans, not links: `gguf` is feature-gated, so a default (no-`gguf`)
//! rustdoc build never compiles the module the link would resolve into.
//!
//! # Two encoders, selected by what the vocab declares
//!
//! Byte-level BPE over the GPT-2 alphabet ([`byte_level`]), with the
//! pre-split selected by [`pretokenize::PreType`], which a GGUF names in
//! `tokenizer.ggml.pre` (`"llama-bpe"` is the LLAMA3 rule, `"qwen2"` and
//! `"qwen35"` split digits one per pretoken); `tokenizer.ggml.model = "gpt2"`
//! identifies the byte-level family itself.
//!
//! Char-level BPE for `tokenizer.ggml.model = "gemma4"`: merges keyed on raw
//! UTF-8 characters with `▁` for space, split only on newline runs
//! ([`pretokenize::pretokenize_newline_runs`]) and merged by
//! [`bpe::encode_char_pretoken`]. [`vocab::Vocab::is_char_level_bpe`] is
//! probed from the vocab's own tokens; the GPT-2 path above is untouched.
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
pub mod draft;
pub mod error;
#[cfg(feature = "gguf")]
pub mod gguf;
#[cfg(feature = "hf")]
pub mod hf;
pub mod pipe;
pub mod pretokenize;
pub mod sample;
pub mod sized;
mod pretokenize_default;
mod unicode_tables;
pub mod unigram;
pub mod vocab;

pub use draft::{NgramSimpleConfig, Verified, ngram_simple_draft, verify_greedy};
pub use error::TokenizerError;
pub use pipe::{decode, drain_lossy_utf8, encode, encode_with_bos_eos};
pub use sample::{SamplingConfig, greedy_pick, sample_next_token};
pub use vocab::{TokenType, Vocab};

#[cfg(test)]
mod tests;
