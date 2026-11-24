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
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// [`StoredMin`]. The [`RangeFit`] says whether the scale and offset are then searched
    /// ([`RangeFit::PLAIN`] is not).
    MinMax(RangeFit),
}

/// The error a candidate scale is judged by in a [`RangeFit`] search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FitError {
    /// `diff * diff`.
    Squared,
    /// `|diff|` (ggml's `use_mad`).
    Absolute,
}

/// How each element's error is weighted in a [`RangeFit`] search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FitWeighting {
    /// Every weight is one.
    Uniform,
    /// `sqrt(mean(x^2)) + |x|` (ggml's `av_x + fabsf(x)`, Q4_K and Q5_K).
    MeanPlusAbs,
    /// `|x|` (Q2_K).
    Abs,
}

/// The search of ggml's `make_qkx2_quants` (ggml-quants.c:808-887) as data: `nstep + 1` candidate
/// inverse scales `(rmin + rdelta * step + nmax) / (max - min)`, each refitted by weighted least
/// squares, the least error kept. `nstep == 0` with [`FitWeighting::Uniform`] is no search: the
/// plain range fit of `quantize_row_q4_1_ref`, which `Q4_1` and `Q5_1` resolve to.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RangeFit {
    /// The offset added to the first candidate's level count, `-1.0` for Q4_K.
    pub rmin: f32,
    /// The step between candidates, `0.1` for the k-quants.
    pub rdelta: f32,
    /// Candidates beyond the first; zero is no search.
    pub nstep: u32,
    pub error: FitError,
    pub weighting: FitWeighting,
}

impl RangeFit {
    /// No search: the fit `Q4_1` and `Q5_1` use.
    pub const PLAIN: Self = Self { rmin: 0.0, rdelta: 0.0, nstep: 0, error: FitError::Squared, weighting: FitWeighting::Uniform };

    /// Whether the fit runs the candidate search.
    #[must_use]
    pub const fn searches(&self) -> bool {
        self.nstep > 0
    }
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
    /// ggml `nearest_int` (ggml-quants.c:630-635): ties to even through the float magic-number add,
    /// see [`crate::range_fit::nearest_even`]; then the bias added.
    NearestEven,
    /// `trunc(value * inverse + bias + 0.5)`: the bias and the half added before the cast truncates
    /// toward zero, so the code is `floor` for any in-range positive sum.
    HalfUpThenTruncate,
    /// One bit: `1` when `value >= 0`, else `0`; the scale carries all of the magnitude.
    NonNegativeBit,
    /// A float code: the stored scale times each finite code's value is compared with the input in
    /// `f32` (`|code_value * scale - value|`), and the first code in code order with the smallest
    /// error wins, so a tie takes the lowest code. This is llama.cpp's `best_index_mxfp4`.
    NearestCodeValue,
    /// A float code: `value / scale` (zero for a zero scale) is encoded by the code layout itself,
    /// nearest representable value with ties to even, saturating under the layout's specials.
    NearestEvenQuotient,
}

/// How the fitted scale is turned into the value the scale field stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ScaleRounding {
    /// The scale field's own encode of `statistic / divisor`: nearest, ties to even.
    Stored,
    /// `2^(floor(log2 peak) - floor(log2 reach))`, the OCP MX v1.0 shared exponent: a block peak
    /// above the largest code clamps to it. Needs a scale field with no mantissa.
    PowerOfTwoFloor,
    /// `2^ceil(log2(peak / reach))`: rounded away from zero, so the block peak always lands inside
    /// the code range and never clamps, at the price of a coarser grid for the rest of the block.
    /// Needs a scale field with no mantissa.
    PowerOfTwoCeil,
}

/// How a quantize step turns a block's peak into the scale and each value into a code, for the
/// integer codes a backend writes when it quantizes an activation. The two arithmetics round
/// differently where `value / scale` and `value * (reach / peak)` straddle a half.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ScaleArithmetic {
    /// `scale = peak / reach`, `code = round(value / scale)`: llama.cpp's `quantize_q8_1`
    /// (`ggml-cuda/quantize.cu`), the path a single-row (matvec) product reads.
    Quotient,
    /// `inverse = reach / peak`, `code = round(value * inverse)`, `scale = 1 / inverse`: llama.cpp's
    /// `quantize_mmq_q8_1` (`quantize.cu`), the path a many-row (matmul) product reads.
    ReciprocalMultiply,
}

impl ScaleArithmetic {
    /// The most rows llama.cpp's CUDA matvec path takes (`MMVQ_MAX_BATCH_SIZE`, `mmvq.cuh:3`); a
    /// product over more rows takes the matmul path and its quantizer.
    pub const SINGLE_ROW_LIMIT: u64 = 8;

    /// The arithmetic llama.cpp picks for an activation read by `rows` rows when `batched` asks for
    /// its batched rule; `Quotient` otherwise.
    #[must_use]
    pub const fn for_rows(rows: u64, batched: bool) -> Self {
        if batched && rows > Self::SINGLE_ROW_LIMIT {
            Self::ReciprocalMultiply
        } else {
            Self::Quotient
        }
    }

    /// The unrounded scale for a block whose largest magnitude is `peak`; zero for a zero peak.
    #[must_use]
    pub fn scale(self, peak: f32, reach: f32) -> f32 {
        if peak == 0.0 {
            return 0.0;
        }
        match self {
            Self::Quotient => peak / reach,
            Self::ReciprocalMultiply => 1.0 / (reach / peak),
        }
    }

    /// The code of `value` in a block with largest magnitude `peak`; zero for a zero peak.
    #[must_use]
    pub fn code(self, value: f32, peak: f32, reach: f32) -> i32 {
        if peak == 0.0 {
            return 0;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the rounded quotient is at most the reach in magnitude"
        )]
        let code = match self {
            Self::Quotient => libm::roundf(value / (peak / reach)) as i32,
            Self::ReciprocalMultiply => libm::roundf(value * (reach / peak)) as i32,
        };
        code
    }
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
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EncodeRule {
    pub statistic: ScaleStatistic,
    /// What the statistic is divided by; negative where the signed peak maps to the most negative
    /// code (`-8` for a four-bit offset-binary code), `1` for a mean. Zero for a float code, whose
    /// divisor is the layout's reach (the largest finite magnitude its codes stand for on both
    /// sides of zero, [`crate::encode::reach`]), which is not always an integer; a nonzero divisor
    /// on a float code replaces the reach (1024 for binary16 staging, which leaves headroom for
    /// sums of 32 staged values below the binary16 maximum).
    pub divisor: i32,
    /// The lowest exponent a power-of-two scale may take: `Some(0)` never scales a row up, so a
    /// small row is staged as it is. `None` leaves the scale field's own range.
    pub min_scale_exponent: Option<i16>,
    pub rounding: CodeRounding,
    /// How the fitted scale becomes the stored one; [`ScaleRounding::Stored`] for every integer code.
    pub scale_rounding: ScaleRounding,
    pub clamp: CodeClamp,
    /// The lowest value a code stands for, before the bias; the lowest raw code for a float code.
    pub lowest: i32,
    /// The highest value a code stands for, before the bias; the highest raw code for a float code.
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
    /// The second level of a layout with sub-blocks: see [`SubRule`].
    pub sub: Option<SubRule>,
}

/// The second level of a two-level layout, derived from its `sub` field
/// ([`crate::block_layout::SubBlocks`]) and the search attached to it. Each sub-block is fitted with
/// `fit`; its scale and min are then quantized as `nearest_even(levels * s / max_s)` clamped to
/// `levels` (`levels` is the top of the sub-block code, 63 for a six-bit one), the block scale is
/// `f16(max_s / levels)` and the block min likewise, and every element code is refitted against the
/// rounded scales. The sub-block codes are written through the description's own planes
/// ([`crate::block_layout::Plane::address`]), the inverse of [`crate::decode::decode_blocks`]' read.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SubRule {
    /// Sub-blocks per block.
    pub count: u16,
    /// The largest sub-block scale or min code.
    pub levels: u32,
    /// The per-sub-block search.
    pub fit: RangeFit,
}

impl EncodeRule {
    /// Whether the code is a small float chosen by [`CodeRounding::NearestCodeValue`] or
    /// [`CodeRounding::NearestEvenQuotient`], with one scale per run of elements.
    #[must_use]
    pub const fn is_float_code(&self) -> bool {
        matches!(self.rounding, CodeRounding::NearestCodeValue | CodeRounding::NearestEvenQuotient)
    }
}

impl BlockLayout {
    /// This layout's encode as data, or `None` when one statistic over one divisor does not say how
    /// the layout is fitted (it stores a min, carries sub-block scales, has a code wider than a
    /// byte, or keeps free bytes the rule does not write). A float code fills the rule in from its
    /// scale field: a scale with no mantissa is a shared exponent (the OCP MX rule), a scale with a
    /// mantissa stores the peak over the reach.
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
                min_scale_exponent: None,
                rounding: CodeRounding::HalfUpThenTruncate,
                scale_rounding: ScaleRounding::Stored,
                clamp: CodeClamp::Upper,
                lowest: -8,
                highest: 7,
                stored_sum: false,
                stored_min: StoredMin::Absent,
                zero_code: 8,
                sub: None,
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
    fn a_layout_with_sub_blocks_and_no_stored_min_has_no_rule() {
        assert_eq!(BlockLayout::Q6_K.encode_rule(), None);
    }

    #[test]
    fn q4_1_and_q5_1_fit_the_range_over_their_code_span_and_store_the_low_end() {
        for (layout, span) in [(BlockLayout::Q4_1, 15), (BlockLayout::Q5_1, 31)] {
            let rule = layout.encode_rule().expect("a stored-min layout has a rule");
            assert_eq!(rule.statistic, ScaleStatistic::MinMax(RangeFit::PLAIN));
            assert_eq!(rule.divisor, span);
            assert_eq!(rule.stored_min, StoredMin::LowEnd { sign: 1 });
            assert_eq!(rule.rounding, CodeRounding::HalfUpThenTruncate);
            assert_eq!(rule.clamp, CodeClamp::Upper);
        }
    }
}
