//! The one target rule for the two multiply-add shapes a block decode has: fused exactly where
//! llama.cpp's compiler fuses them, so a decode computes the expression the incumbent's build of
//! the same target runs.
//!
//! llama.cpp writes `x * d + m` (`ggml-quants.c` `dequantize_row_q4_1`, `_q5_1`) and
//! `d1 * q - m1` with `m1 = min * sub_min` (`_q2_K`, `_q4_K`, `_q5_K`) as plain C, and clang
//! contracts a multiply and an add into one `fma` wherever the target has an FMA instruction. For
//! the K formats the contracted multiply is `d1 * q`, and the min's product is negated before it
//! is multiplied: `fma(d1, q, (-min) * sub_min)`. The cross-check against the incumbent's own
//! `dequantize_row_*` fixes that form: a NaN min comes out sign-flipped, an infinite min times a
//! zero sub-min comes out as the default NaN (`0x7fc00000`, not its negation), and when `d1` and
//! the min are both NaN the min's wins, which the hardware's addend-first NaN priority gives only
//! when the min's product is the addend. Every product of a code and a scale is exact in `f32`,
//! so fused and unfused agree on every finite value; they differ only in the payload and sign of
//! a NaN.
//!
//! The rule is fused when the target has a hardware FMA and `f32::mul_add` exists: `aarch64`
//! always has `fmadd`; `x86` and `x86_64` have it only with `target_feature = "fma"` (a generic
//! x86-64 build emits `mulps` + `addps`, as the incumbent's does). `mul_add` is a `std` method,
//! so the `no_std` tier is unfused: it mirrors a build that cannot emit an `fma`, never a software
//! `fma` that would be slower than the mul and add it replaces.
//!
//! Teaching pointer: [`crate::block_codec`] applies [`mul_add`] to every layout with a min, and
//! `proxima-gguf`'s `q4_1`, `q5_1` dequantizers apply it too; `q2_k`, `q4_k`, `q5_k` compute
//! [`min_term`] once per sub-block and [`scaled_sub`] per element. The incumbent ports and the
//! layout codec share the rule, so they cannot drift apart on it.

#[cfg(all(
    feature = "std",
    any(
        target_arch = "aarch64",
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "fma")
    )
))]
mod rule {
    use core::hint::black_box;

    pub const FUSED: bool = true;

    #[inline(always)]
    pub fn mul_add(factor: f32, scale: f32, min: f32) -> f32 {
        factor.mul_add(scale, min)
    }

    #[inline(always)]
    pub fn min_term(min: f32, sub_min: f32) -> f32 {
        // llvm commutes a negate-then-multiply into multiply-then-negate, which turns the default
        // nan of an infinite min times a zero sub-min into its negation; the opaque negation keeps
        // the incumbent's order, once per sub-block
        black_box(-min) * sub_min
    }

    #[inline(always)]
    pub fn scaled_sub(scale: f32, code: f32, term: f32) -> f32 {
        scale.mul_add(code, term)
    }
}

#[cfg(not(all(
    feature = "std",
    any(
        target_arch = "aarch64",
        all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "fma")
    )
)))]
mod rule {
    pub const FUSED: bool = false;

    #[inline(always)]
    pub fn mul_add(factor: f32, scale: f32, min: f32) -> f32 {
        factor * scale + min
    }

    #[inline(always)]
    pub fn min_term(min: f32, sub_min: f32) -> f32 {
        min * sub_min
    }

    #[inline(always)]
    pub fn scaled_sub(scale: f32, code: f32, term: f32) -> f32 {
        scale * code - term
    }
}

/// `true` when this build computes a multiply and an add as one rounding, as llama.cpp's build of
/// the same target does.
pub const FUSED: bool = rule::FUSED;

/// `factor * scale + min`, in llama.cpp's operand order, fused where its compiler fuses it.
#[inline(always)]
pub fn mul_add(factor: f32, scale: f32, min: f32) -> f32 {
    rule::mul_add(factor, scale, min)
}

/// The K formats' `m1 = min * sub_min`, once per sub-block, in the form [`scaled_sub`] consumes:
/// negated before the multiply where fused, so a NaN min is sign-flipped and an infinite min times
/// a zero sub-min is the default NaN, as the incumbent's contracted code produces.
#[inline(always)]
pub fn min_term(min: f32, sub_min: f32) -> f32 {
    rule::min_term(min, sub_min)
}

/// `scale * code - m1`, per element, given the sub-block's [`min_term`]: `scale * code` fused into
/// the subtract where llama.cpp's compiler fuses it.
#[inline(always)]
pub fn scaled_sub(scale: f32, code: f32, term: f32) -> f32 {
    rule::scaled_sub(scale, code, term)
}

#[cfg(test)]
mod tests {
    use core::hint::black_box;

    use super::*;

    fn decode(scale: f32, code: f32, min: f32, sub_min: f32) -> f32 {
        scaled_sub(scale, code, min_term(min, sub_min))
    }

    #[test]
    fn finite_values_agree_with_the_unfused_expression() {
        let scale = half::f16::from_bits(0x3555).to_f32();
        let min = half::f16::from_bits(0xb2aa).to_f32();
        for code in 0..32u8 {
            let factor = f32::from(code);
            assert_eq!(mul_add(factor, scale, min).to_bits(), (factor * scale + min).to_bits());
            assert_eq!(
                decode(scale, factor, min, f32::from(code)).to_bits(),
                (scale * factor - min * f32::from(code)).to_bits()
            );
        }
    }

    #[test]
    fn zero_times_infinity_against_a_nan_min_is_the_default_nan_where_fused() {
        let nan_min = f32::from_bits(0x7fc0_1234);
        let result = mul_add(black_box(0.0), black_box(f32::INFINITY), black_box(nan_min));
        assert!(result.is_nan());
        if FUSED && cfg!(target_arch = "aarch64") {
            assert_eq!(result.to_bits(), 0x7fc0_0000);
        }
    }

    #[test]
    fn a_nan_min_comes_out_of_the_k_decode_sign_flipped_where_fused() {
        let nan_min = f32::from_bits(0x7fc4_e000);
        let result = decode(black_box(3.0), black_box(5.0), black_box(nan_min), black_box(2.0));
        assert!(result.is_nan());
        if FUSED && cfg!(target_arch = "aarch64") {
            assert_eq!(result.to_bits(), 0xffc4_e000);
        }
    }

    #[test]
    fn an_infinite_min_times_a_zero_sub_min_is_the_default_nan_not_its_negation_where_fused() {
        for min in [f32::INFINITY, f32::NEG_INFINITY] {
            let result = decode(black_box(3.0), black_box(5.0), black_box(min), black_box(0.0));
            assert!(result.is_nan());
            if FUSED && cfg!(target_arch = "aarch64") {
                assert_eq!(result.to_bits(), 0x7fc0_0000, "min {min}");
            }
        }
    }

    #[test]
    fn a_nan_min_beats_a_nan_scale_where_fused() {
        let nan_scale = f32::from_bits(0x7fe9_8000);
        let nan_min = f32::from_bits(0x7fed_c000);
        let result = decode(black_box(nan_scale), black_box(5.0), black_box(nan_min), black_box(2.0));
        assert!(result.is_nan());
        if FUSED && cfg!(target_arch = "aarch64") {
            assert_eq!(result.to_bits(), 0xffed_c000);
        }
    }

    #[test]
    fn aarch64_with_std_is_fused() {
        let fused = black_box(FUSED);
        if cfg!(all(feature = "std", target_arch = "aarch64")) {
            assert!(fused);
        }
        if cfg!(not(feature = "std")) {
            assert!(!fused);
        }
    }
}
