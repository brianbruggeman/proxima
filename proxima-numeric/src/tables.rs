//! Trained numeric codebooks, moved out of `proxima-gguf` because they are numeric data: a level
//! set and a magnitude grid are what a format's codes mean, not how a file is parsed.
//!
//! Every table here is copied verbatim from llama.cpp's `ggml-common.h`; the per-module doc names
//! the exact source line range (guiding-principles principle 10, attribution).

pub mod iq2xs_grid;
pub mod iq3xxs_grid;

pub use iq2xs_grid::{IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
pub use iq3xxs_grid::IQ3XXS_GRID;

/// The non-linear 4-bit codebook every IQ4_NL nibble indexes into
/// (`ggml-common.h:1077-1079`, `GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)`), copied verbatim.
/// Trained offline by llama.cpp to fit typical weight distributions better than a uniform
/// `nibble - 8` ladder; this crate only decodes, never retrains it.
pub const KVALUES_IQ4_NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
