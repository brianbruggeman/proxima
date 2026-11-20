//! The CPU side of the expert-grouped GEMM parity tests: the same stage formats as the kernel
//! ([`omega::msl::GroupedGemmFormats`]), applied through `proxima_numeric`'s float codec.
//!
//! The kernel rounds the weight to `weight_tile` after decode and the activation to
//! `activation_tile` on staging (round to nearest even, no scale), multiplies the two exactly
//! (a product of two 11-bit significands fits binary32), and accumulates in `accumulator`. So the
//! oracle rounds both operands through their layouts and lets the f32 evaluator multiply and
//! accumulate.

#![allow(dead_code)]

use omega::msl::GroupedGemmFormats;
use proxima_numeric::float_layout::FloatLayout;
use proxima_numeric::float_layout_codec::{decode, encode};

/// `values` rounded to `layout` (nearest, ties to even; overflow per the layout's specials) and
/// widened back to `f32`.
pub fn rounded_through(values: &[f32], layout: FloatLayout) -> Vec<f32> {
    values
        .iter()
        .map(|value| decode(encode(*value, layout), layout))
        .collect()
}

/// The weight and activation an f32 evaluator must see to match a kernel staged in `formats`.
pub fn staged_operands(
    formats: &GroupedGemmFormats,
    weight: &[f32],
    activation: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    (
        rounded_through(weight, formats.weight_tile),
        rounded_through(activation, formats.activation_tile),
    )
}

/// Worst relative difference over a row, treating a non-finite element as equal only when the
/// other side is the same kind of non-finite (same-signed infinity, or NaN); a mismatch is
/// `f32::INFINITY`.
pub fn worst_relative_row_error_with_specials(oracle: &[f32], metal: &[f32], row_width: usize) -> f32 {
    oracle
        .chunks_exact(row_width)
        .zip(metal.chunks_exact(row_width))
        .map(|(want_row, got_row)| {
            let norm = want_row
                .iter()
                .filter(|value| value.is_finite())
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            let worst = want_row
                .iter()
                .zip(got_row)
                .map(|(want, got)| element_difference(*want, *got))
                .fold(0.0f32, f32::max);
            worst / norm.max(1.0e-6)
        })
        .fold(0.0f32, f32::max)
}

fn element_difference(want: f32, got: f32) -> f32 {
    if want.is_finite() && got.is_finite() {
        (want - got).abs()
    } else if (want.is_nan() && got.is_nan()) || want == got {
        0.0
    } else {
        f32::INFINITY
    }
}
