//! How a block of values becomes a block of codes, as data.
//!
//! [`BlockLayout`] describes where the bits of a block are and what they mean (the decode).
//! [`EncodeRule`] is its other half: which statistic of the block's values fixes the scale, what
//! the scale is divided by, how a value over the scale becomes a code, and where the code is
//! clamped. It names no format: the constants of a layout (the bias, the code width, the
//! multiplier, the free bytes beside the scale) fill every field in, and
//! [`BlockLayout::encode_rule`] is that fill. Running the rule is somebody else's job: the host
//! loop is [`crate::block_codec`], a device kernel is rendered from the same fields, and this
//! module only says what the arithmetic is.
//!
//! Teaching pointer: [`crate::decode::decode_blocks`] reads a block by its [`BlockLayout`];
//! [`crate::format_codec::FormatCodec::encode`] writes one by the layout's [`EncodeRule`]. A layout
//! whose scale is not one statistic over one divisor (a stored min, sub-block scales, a search over
//! candidate scales, a trained codebook) has no rule, and [`BlockLayout::encode_rule`] says so with
//! `None` instead of approximating it.
//!
//! The arithmetic a rule means, in `f32` throughout:
//!
//! - `scale = statistic(values) / divisor` (a [`ScaleStatistic::MeanMagnitude`] is already the
//!   mean, so its divisor is one);
//! - `inverse = 0` when `scale == 0`, else `1 / scale`;
//! - the code of a value is chosen by [`CodeRounding`] from `value * inverse` and the layout's bias,
//!   then clamped by [`CodeClamp`] to `lowest..=highest`.

use crate::block_layout::BlockLayout;
use crate::block_codec::encode_rule_of;

/// The statistic of a block's values the scale is fitted to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ScaleStatistic {
    /// The largest magnitude, taken in element order with the C `MAX` macro, so a NaN in the last
    /// element stays NaN.
    PeakMagnitude,
    /// The signed value of largest magnitude, the first one on a tie, the zero when all are zero.
    SignedPeak,
    /// The mean magnitude: the sum of the magnitudes over the element count.
    MeanMagnitude,
    /// The range of the block: the scale is `(max - min) / divisor`, with `min` and `max` taken by
    /// strict comparison in element order. The low end is also the block's offset: see
    /// [`StoredMin`].
    MinMax,
}

/// Whether the block stores the low end of its [`ScaleStatistic::MinMax`] range, and with which sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StoredMin {
    /// No offset: a value is coded from zero.
    Absent,
    /// The layout's `min` field holds `sign * low`, with `sign` the layout's `min_multiplier`
    /// (`1` or `-1`), so a decode that adds `min_multiplier * min` recovers the low end. A code is
    /// then fitted from `value - low`.
    LowEnd { sign: i8 },
}

/// How a value over the scale becomes a code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CodeRounding {
    /// `round(value * inverse)` with ties away from zero, then the bias added.
    NearestAwayFromZero,
    /// `trunc(value * inverse + bias + 0.5)`: the bias and the half added before the cast truncates
    /// toward zero, so the code is `floor` for any in-range positive sum.
    HalfUpThenTruncate,
    /// One bit: `1` when `value >= 0`, else `0`; the scale carries all of the magnitude.
    NonNegativeBit,
}

/// Which ends of the code range clamp a code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CodeClamp {
    /// Both ends: the value is clamped to `lowest..=highest` before the bias is added.
    Both,
    /// Only the top: the biased code is `min`'d with `highest + bias`; below zero the cast decides.
    Upper,
    /// Neither: the code is one bit and cannot leave its range.
    Neither,
}

/// A block's encode, as data. See the module documentation for the arithmetic each field means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EncodeRule {
    pub statistic: ScaleStatistic,
    /// What the statistic is divided by; negative where the signed peak maps to the most negative
    /// code (`-8` for a four-bit offset-binary code), `1` for a mean.
    pub divisor: i32,
    pub rounding: CodeRounding,
    pub clamp: CodeClamp,
    /// The lowest value a code stands for, before the bias.
    pub lowest: i32,
    /// The highest value a code stands for, before the bias.
    pub highest: i32,
    /// The two bytes after the scale hold `f16(scale * sum(values of codes))`.
    pub stored_sum: bool,
    /// Whether the layout's min field holds the low end of the block's range.
    pub stored_min: StoredMin,
    /// The raw code whose value is zero (the bias of an offset-binary code, `0` of a two's
    /// complement one). A block whose finite scale stores as zero, or stores with a non-finite
    /// inverse, has every code set to this, so it decodes to exactly zero; a NaN or infinite scale
    /// is not that case and keeps the arithmetic.
    pub zero_code: u8,
}

impl BlockLayout {
    /// This layout's encode as data, or `None` when one statistic over one divisor does not say how
    /// the layout is fitted (it stores a min, carries sub-block scales, has a float or wide code,
    /// or keeps free bytes the rule does not write).
    #[must_use]
    pub const fn encode_rule(&self) -> Option<EncodeRule> {
        encode_rule_of(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_fits_the_signed_peak_over_minus_its_bias_and_truncates_after_the_half() {
        let rule = BlockLayout::Q4_0.encode_rule().expect("q4_0 has a rule");
        assert_eq!(
            rule,
            EncodeRule {
                statistic: ScaleStatistic::SignedPeak,
                divisor: -8,
                rounding: CodeRounding::HalfUpThenTruncate,
                clamp: CodeClamp::Upper,
                lowest: -8,
                highest: 7,
                stored_sum: false,
                stored_min: StoredMin::Absent,
                zero_code: 8,
            }
        );
    }

    #[test]
    fn q8_0_fits_the_peak_over_127_and_rounds_away_from_zero_inside_both_ends() {
        let rule = BlockLayout::Q8_0.encode_rule().expect("q8_0 has a rule");
        assert_eq!(rule.statistic, ScaleStatistic::PeakMagnitude);
        assert_eq!(rule.divisor, 127);
        assert_eq!(rule.rounding, CodeRounding::NearestAwayFromZero);
        assert_eq!(rule.clamp, CodeClamp::Both);
        assert!(!rule.stored_sum);
    }

    #[test]
    fn q8_1_is_the_q8_0_rule_with_a_stored_sum_and_q1_0_is_a_mean_sign_bit() {
        let q8_1 = BlockLayout::Q8_1.encode_rule().expect("q8_1 has a rule");
        assert!(q8_1.stored_sum);
        assert_eq!(q8_1.divisor, 127);
        let q1_0 = BlockLayout::Q1_0.encode_rule().expect("q1_0 has a rule");
        assert_eq!(q1_0.statistic, ScaleStatistic::MeanMagnitude);
        assert_eq!(q1_0.rounding, CodeRounding::NonNegativeBit);
        assert_eq!(q1_0.clamp, CodeClamp::Neither);
    }

    #[test]
    fn a_layout_with_sub_blocks_has_no_rule() {
        assert_eq!(BlockLayout::Q4_K.encode_rule(), None);
    }

    #[test]
    fn q4_1_and_q5_1_fit_the_range_over_their_code_span_and_store_the_low_end() {
        for (layout, span) in [(BlockLayout::Q4_1, 15), (BlockLayout::Q5_1, 31)] {
            let rule = layout.encode_rule().expect("a stored-min layout has a rule");
            assert_eq!(rule.statistic, ScaleStatistic::MinMax);
            assert_eq!(rule.divisor, span);
            assert_eq!(rule.stored_min, StoredMin::LowEnd { sign: 1 });
            assert_eq!(rule.rounding, CodeRounding::HalfUpThenTruncate);
            assert_eq!(rule.clamp, CodeClamp::Upper);
        }
    }
}
