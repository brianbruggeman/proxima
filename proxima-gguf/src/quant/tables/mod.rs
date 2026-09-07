//! Large codebook/lookup tables for the non-linear (`iq*`) quant formats,
//! kept out of their owning codec module so `iq2_xs.rs`/`iq3_xxs.rs` read
//! as decode logic, not a wall of hex literals. Each table here is copied
//! verbatim from llama.cpp's `ggml-common.h` -- see the per-module doc for
//! the exact source line range (guiding-principles principle 10).

pub mod iq2xs_grid;
pub mod iq3xxs_grid;
