//! Build-time constants -- the no_std+alloc floor's only configuration
//! surface (conflaguration's tier-2 pattern: constants ARE the config
//! below `std`). At `std`, [`crate::config::GgufParserConfig`] seeds its
//! runtime defaults from these, never re-declaring them, so the two can
//! never drift apart (`defaults_track_the_sized_floor` in `config.rs`
//! pins the invariant).
//!
//! `MAX_DIMS` and `MAX_NAME_LEN` size `ArrayVec` const generics and
//! therefore can never be runtime config at any tier -- they stay
//! build-time-only even at `std`.

/// `ggml_tensor::ne` is a fixed 4-element array (`ggml.h:218`,
/// `GGML_MAX_DIMS`); trailing unused dimensions read as 1. Sizes the
/// `ArrayVec<u64, MAX_DIMS>` const generic -- cannot be runtime config at
/// any tier.
pub const MAX_DIMS: usize = 4;

/// `ggml_tensor::name` is `char[GGML_MAX_NAME]` (`ggml.h:225`, 64 bytes
/// including the nul terminator llama.cpp's C string carries -- the GGUF
/// wire string itself has no terminator, so the usable length is 63).
/// Bounds the wire string-length check -- cannot be runtime config at any
/// tier (there is no const-generic capacity tied to it today, but it is a
/// format-identity fact, not a policy knob).
pub const MAX_NAME_LEN: usize = 63;

/// Newest GGUF version this parser understands (`gguf.h:42`). Policy, not
/// format identity: at `std`, `GgufParserConfig::max_supported_version`
/// seeds from this and can be overridden per-process.
pub const MAX_SUPPORTED_VERSION: u32 = 3;

/// Fallback alignment when `general.alignment` is absent (`gguf.h:46`).
/// Policy: at `std`, `GgufParserConfig::default_alignment` seeds from this
/// and can be overridden per-process.
pub const DEFAULT_ALIGNMENT: u32 = 32;
