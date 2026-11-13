//! The searched min-max fit of one block: ggml's `make_qkx2_quants` (ggml-quants.c:808-887, b12110).
//!
//! [`RangeFit`] is the data (see [`crate::encode_rule::ScaleStatistic::MinMax`]); [`fit_range`]
//! runs it. With `nstep == 0` and uniform weights the fit is the plain range fit of
//! `quantize_row_q4_1_ref`, which [`crate::block_codec`] runs without this module; with
//! `nstep > 0` the scale and the offset are refitted by weighted least squares over `nstep + 1`
//! candidate inverse scales and the best error wins. The arithmetic follows the C statement by
//! statement in `f32` (no fused multiply-add), so the result is bit-equal to the incumbent's.
//!
//! Teaching pointer: a k-quant sub-block is this fit run per 32 (or 16) elements; the two-level
//! rule that then quantizes the sub-block scales is a separate card.

use crate::encode_rule::{FitError, FitWeighting, RangeFit};

/// ggml `nearest_int` (ggml-quants.c:630-635): round half to even by adding `1.5 * 2^23` so the
/// integer lands in the low mantissa bits. The C asserts `|value| <= 4194303`; here the input
/// saturates there so a larger one cannot leave the mantissa.
#[inline(always)]
#[must_use]
pub fn nearest_even(value: f32) -> i32 {
    let biased = value.clamp(-4_194_303.0, 4_194_303.0) + 12_582_912.0;
    (biased.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

#[inline(always)]
fn weight_of(fit: &RangeFit, mean_square_root: f32, value: f32) -> f32 {
    match fit.weighting {
        FitWeighting::Uniform => 1.0,
        FitWeighting::MeanPlusAbs => mean_square_root + value.abs(),
        FitWeighting::Abs => value.abs(),
    }
}

#[inline(always)]
fn error_of(fit: &RangeFit, diff: f32) -> f32 {
    match fit.error {
        FitError::Squared => diff * diff,
        FitError::Absolute => diff.abs(),
    }
}

/// Fits `input` to codes `0..=nmax`: writes the codes and returns `(scale, offset)` with
/// `value ~ scale * code + offset` and `offset <= 0` (ggml returns `-offset` as `*the_min`).
/// A block whose range is empty once the low end is clamped to at most zero fits to scale zero
/// and codes zero, as in the C.
///
/// O(`nstep` * `ELEMENTS`); no allocation.
#[must_use]
pub fn fit_range<const ELEMENTS: usize>(
    fit: &RangeFit,
    nmax: i32,
    input: &[f32; ELEMENTS],
    codes: &mut [u8; ELEMENTS],
) -> (f32, f32) {
    let mut sum_squares = 0.0f32;
    for &value in input {
        sum_squares += value * value;
    }
    let root = libm::sqrtf(sum_squares / ELEMENTS as f32);
    let mut low = input[0];
    let mut high = input[0];
    let mut sum_w = weight_of(fit, root, input[0]);
    let mut sum_x = sum_w * input[0];
    for &value in &input[1..] {
        if value < low {
            low = value;
        }
        if value > high {
            high = value;
        }
        let weight = weight_of(fit, root, value);
        sum_w += weight;
        sum_x += weight * value;
    }
    if low > 0.0 {
        low = 0.0;
    }
    if high == low {
        codes.fill(0);
        return (0.0, low);
    }
    let levels = nmax as f32;
    let mut iscale = levels / (high - low);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for (code, &value) in codes.iter_mut().zip(input) {
        let level = nearest_even(iscale * (value - low)).clamp(0, nmax);
        *code = level as u8;
        let diff = scale * level as f32 + low - value;
        best_error += weight_of(fit, root, value) * error_of(fit, diff);
    }
    if fit.nstep < 1 {
        return (scale, low);
    }
    let mut trial = [0u8; ELEMENTS];
    for step in 0..=fit.nstep {
        iscale = (fit.rmin + fit.rdelta * step as f32 + levels) / (high - low);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for (slot, &value) in trial.iter_mut().zip(input) {
            let level = nearest_even(iscale * (value - low)).clamp(0, nmax);
            *slot = level as u8;
            let weight = weight_of(fit, root, value);
            let level = level as f32;
            sum_l += weight * level;
            sum_l2 += weight * level * level;
            sum_xl += weight * level * value;
        }
        let determinant = sum_w * sum_l2 - sum_l * sum_l;
        if determinant > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / determinant;
            let mut this_offset = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
            if this_offset > 0.0 {
                this_offset = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut error = 0.0f32;
            for (&slot, &value) in trial.iter().zip(input) {
                let diff = this_scale * slot as f32 + this_offset - value;
                error += weight_of(fit, root, value) * error_of(fit, diff);
            }
            if error < best_error {
                *codes = trial;
                best_error = error;
                scale = this_scale;
                low = this_offset;
            }
        }
    }
    (scale, low)
}

#[cfg(test)]
mod tests {
    use super::*;

    const K_QUANT: RangeFit = RangeFit { rmin: -1.0, rdelta: 0.1, nstep: 20, error: FitError::Squared, weighting: FitWeighting::MeanPlusAbs };
    const Q2_K: RangeFit = RangeFit { rmin: -0.5, rdelta: 0.1, nstep: 15, error: FitError::Absolute, weighting: FitWeighting::Abs };
    const Q5_K: RangeFit = RangeFit { rmin: -0.5, rdelta: 0.1, nstep: 15, error: FitError::Squared, weighting: FitWeighting::Uniform };

    /// the 256-value ramp of the PROXNUM-218 worked example; a permutation of multiples of 1/32.
    fn ramp(start: usize) -> [f32; 32] {
        let mut block = [0.0f32; 32];
        for (slot, index) in block.iter_mut().zip(start..) {
            *slot = (((index * 37) % 256) as f32 - 90.0) / 32.0;
        }
        block
    }

    fn case(name: &str) -> ([f32; 32], i32, RangeFit) {
        match name {
            "positive" => {
                let mut block = [0.0f32; 32];
                for (lane, slot) in block.iter_mut().enumerate() {
                    *slot = ((lane * 11) % 32 + 3) as f32 / 16.0;
                }
                (block, 15, K_QUANT)
            }
            "constant" => ([0.5f32; 32], 15, K_QUANT),
            "q2k_abs_mad" => (ramp(64), 3, Q2_K),
            "q5k_uniform" => (ramp(96), 31, Q5_K),
            "plain_nstep0" => (ramp(128), 15, RangeFit { nstep: 0, ..K_QUANT }),
            sub => (ramp(32 * sub[7..].parse::<usize>().expect("q4k_subN")), 15, K_QUANT),
        }
    }

    /// (case, bits of ggml's scale, bits of ggml's `*the_min`, codes) from `raw/harness.c`
    /// (`cc -O0 -ffp-contract=off`, ggml-quants.c b12110 make_qkx2_quants verbatim).
    const ORACLE: &[(&str, u32, u32, [u8; 32])] = &[
    ("q4k_sub0", 0x3ef102a4, 0x40327a91, [0,2,5,7,10,12,15,0,3,5,8,10,12,15,0,3,5,8,10,13,15,1,3,5,8,10,13,15,1,3,6,8]),
    ("q4k_sub1", 0x3eeea046, 0x4015973e, [10,12,15,0,2,5,7,10,12,15,0,3,5,8,10,13,15,0,3,5,8,10,13,15,1,3,6,8,11,13,15,1]),
    ("q4k_sub2", 0x3f1122ae, 0x4036979e, [4,6,8,10,12,14,2,4,6,8,10,12,14,2,4,6,8,10,12,14,2,4,6,8,10,12,0,2,4,6,8,10]),
    ("q4k_sub3", 0x3ef1ed93, 0x40282824, [14,0,2,5,7,10,12,15,0,3,5,7,10,12,15,0,3,5,8,10,13,15,1,3,5,8,10,13,15,1,3,6]),
    ("q4k_sub4", 0x3eeb7941, 0x40053627, [7,10,12,15,0,2,5,7,10,12,15,0,3,5,8,10,13,15,0,3,5,8,10,13,15,0,3,5,8,10,13,15]),
    ("q4k_sub5", 0x3f10e8e2, 0x402eae44, [2,4,6,8,10,12,14,2,4,6,8,10,12,0,2,4,6,8,10,12,0,2,4,6,8,10,12,0,2,4,6,9]),
    ("q4k_sub6", 0x3ef1b924, 0x401dbe71, [12,14,0,2,5,7,10,12,15,0,3,5,7,10,12,15,0,3,5,8,10,13,15,1,3,5,8,10,13,15,1,3]),
    ("q4k_sub7", 0x3ee8b592, 0x3fef330c, [5,7,10,12,15,0,2,5,7,10,12,15,0,2,5,7,10,13,15,0,3,5,8,10,13,15,0,3,5,8,10,13]),
    ("positive", 0x3e0ff0c1, 0x80000000, [1,6,11,2,7,12,2,7,12,3,8,12,3,8,13,4,8,13,4,9,14,4,9,14,5,10,15,5,10,15,6,11]),
    ("constant", 0x3d088889, 0x80000000, [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]),
    ("q2k_abs_mad", 0x4004daef, 0x3fa48b94, [0,1,1,2,2,3,0,0,1,1,2,2,3,0,0,1,1,2,3,3,0,0,1,1,2,3,0,0,0,1,2,2]),
    ("q5k_uniform", 0x3e6e0f0a, 0x402a0a37, [29,0,5,10,15,20,25,30,0,5,10,15,20,25,30,1,6,11,16,21,26,31,1,6,11,16,21,26,31,2,7,12]),
    ("plain_nstep0", 0x3ef66666, 0x400c0000, [7,9,12,14,0,2,5,7,10,12,14,0,3,5,7,10,12,15,0,3,5,8,10,12,15,1,3,5,8,10,13,15]),
    ];

    #[test]
    fn make_qkx2_quants_oracle_scale_min_and_codes_are_bit_equal() {
        for &(name, scale_bits, min_bits, expected) in ORACLE {
            let (block, nmax, fit) = case(name);
            let mut codes = [0u8; 32];
            let (scale, offset) = fit_range(&fit, nmax, &block, &mut codes);
            assert_eq!(scale.to_bits(), scale_bits, "{name} scale {scale}");
            assert_eq!((-offset).to_bits(), min_bits, "{name} min {}", -offset);
            assert_eq!(codes, expected, "{name} codes");
        }
    }

    #[test]
    fn an_all_zero_block_fits_to_zero_scale_and_zero_codes() {
        let mut codes = [9u8; 32];
        let (scale, offset) = fit_range(&K_QUANT, 15, &[0.0f32; 32], &mut codes);
        assert_eq!((scale, offset, codes), (0.0, 0.0, [0u8; 32]));
    }

    #[test]
    fn nearest_even_rounds_ties_to_even_not_away_from_zero() {
        let cases = [(0.5, 0), (1.5, 2), (2.5, 2), (3.5, 4), (-0.5, 0), (-1.5, -2), (-2.5, -2), (2.4999, 2), (2.5001, 3), (-7.6, -8)];
        for (value, expected) in cases {
            assert_eq!(nearest_even(value), expected, "{value}");
        }
    }
}
