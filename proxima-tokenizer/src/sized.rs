//! Build-time constants -- the no_std+alloc floor's only configuration
//! surface (mirrors `proxima-gguf/src/sized.rs`'s tier-2 pattern: constants
//! ARE the config below `std`). At `std`, [`crate::config::TokenizerConfig`]
//! seeds its runtime default from this, never re-declaring the value, so
//! the two can never drift apart (`default_tracks_the_sized_floor` in
//! `config.rs` pins the invariant).

/// Largest single `encode` input this tokenizer accepts, in bytes. BPE
/// merging is worst-case quadratic in pretoken length, so an unbounded
/// input is a denial-of-service surface; this is a defensive ceiling, not
/// a format fact. Policy: at `std`, `TokenizerConfig::max_input_bytes`
/// seeds from this and can be overridden per-process.
pub const MAX_INPUT_BYTES: usize = 1 << 20;
