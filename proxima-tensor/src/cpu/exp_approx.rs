//! Polynomial exponential for the width kernels, reached only when the plan's
//! `NumericPolicy::approx_functions` is granted (`elementwise_width_generic`'s
//! `approximate_exp`).
//!
//! `exp(x) = 2^n * exp(r)` with `n = round(x * log2(e))` and `r = x - n * ln 2`
//! (`ln 2` split in two so `r` keeps its low bits), `|r| <= ln(2) / 2`. `exp(r)`
//! is the degree-6 Taylor polynomial; its truncation error at `|r| = 0.3466` is
//! `r^7 / 5040`, about 1.3e-7 relative. Every operation is a multiply, add,
//! round, convert or select, so a loop over a contiguous row vectorizes (NEON
//! `fmul`/`fadd`/`frintn`/`fcvtzs`) where libm's `expf` is a call that blocks it.
//!
//! It is a macro so the element type is whatever the caller's value is: the
//! width kernels already fix it, and a format is not spelled here. The relative
//! error bound (6e-7) is asserted by
//! `exp_approximation_stays_within_the_documented_relative_error` in `tests.rs`.

/// `exp(value)` to a 6e-7 relative error for `value` in `[-87.33, 88.72]`;
/// `0` below and `inf` above; NaN in gives NaN out.
macro_rules! exp_approximation {
    ($value:expr) => {{
        let value = $value;
        let scaled = (value * 1.442_695_04).round();
        let reduced = (value - scaled * 0.693_145_75) - scaled * 1.428_606_8e-6;
        let polynomial = 1.0
            + reduced
                * (1.0
                    + reduced
                        * (0.5
                            + reduced
                                * (1.0 / 6.0
                                    + reduced
                                        * (1.0 / 24.0 + reduced * (1.0 / 120.0 + reduced / 720.0)))));
        let exponent_bits = ((scaled as i32).clamp(-126, 127) + 127) << 23;
        // SAFETY: a 32-bit integer is a valid bit pattern for the 32-bit float
        // the target infers; the biased exponent is in 1..=254, a normal number.
        // the array ties the transmute target to `value`'s type without spelling it.
        let [_, power_of_two] = [value, unsafe { core::mem::transmute::<i32, _>(exponent_bits) }];
        if value < -87.336_55 {
            0.0
        } else if value > 88.722_84 {
            value / 0.0
        } else {
            polynomial * power_of_two
        }
    }};
}
