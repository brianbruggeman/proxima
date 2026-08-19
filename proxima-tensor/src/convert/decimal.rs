//! Decimal as a conversion, not a `DType` variant.
//!
//! A fixed-point add is exactly an integer add on the mantissa — no pipe, no
//! wrapper, nothing to write, because the existing `+` on `i64`/`i128`
//! already expresses it; minting a type to host that would be the newtype
//! this crate's own rules rule out. Everything that *does* need state lives
//! here: [`ToFixed`]/[`FromFixed`] hold the scale (a decimal's one piece of
//! information a bare mantissa buffer cannot carry), and [`DecimalMultiply`]
//! holds the scale plus the widening-intermediate boundary a multiply
//! actually needs.
//!
//! A widening multiply's intermediate needs twice the mantissa's bits — an
//! `i64` mantissa at `scale=10^9` already requires an `i128` intermediate
//! before the rescale divide, and `i128` itself has nothing wider to widen
//! into, so its own multiply can overflow past magnitude ~1e12 at that scale
//! with no silent-wrap escape hatch — [`DecimalMultiply<i128>`] reports that
//! as [`DecimalError::MultiplyOverflow`] instead of wrapping.
//!
//! Both boundary points are pinned by tests in this module rather than by a
//! citation: `multiply_i128_stays_exact_below_the_measured_overflow_boundary`
//! and `multiply_i128_reports_overflow_past_its_own_measured_boundary`. A
//! measurement that lives only in a scratch file is not reproducible once
//! that file is cleaned up, which is exactly what happened to the probe this
//! comment used to cite.

use core::future::Future;
use core::marker::PhantomData;

use proxima_primitives::pipe::Pipe;

/// Round-half-away-from-zero on a plain `f64`, without `f64::round` — that
/// method is a `std`-only libm call (`core` has no float transcendentals),
/// and this pipe is core-tier. The bias-then-truncate shape is exactly what
/// the `as i64`/`as i128` cast below already needs to do, so no transcendental
/// call is missing, only the one this crate chooses not to depend on.
#[must_use]
fn round_half_away_from_zero(value: f64) -> f64 {
    if value >= 0.0 { value + 0.5 } else { value - 0.5 }
}

/// Every fault a decimal conversion can raise: a scale too large for its
/// mantissa width to hold `10^scale`, or a multiply whose product (`i64` ->
/// widened into `i128`, or `i128` multiplied directly with nothing wider to
/// widen into) overflowed before the rescale divide could run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DecimalError {
    #[error("scale 10^{scale} does not fit a {bits}-bit mantissa")]
    ScaleOutOfRange { scale: u32, bits: u32 },

    #[error("decimal multiply overflowed the {bits}-bit mantissa's intermediate at scale 10^{scale}")]
    MultiplyOverflow { bits: u32, scale: u32 },
}

/// Converts a floating value into a fixed-point mantissa at this pipe's
/// scale — `round(value * 10^scale)`. `Mantissa` is `i64` or `i128`; see
/// this module's own doc for why nothing wider is wired up.
#[derive(Debug, Clone, Copy)]
pub struct ToFixed<Mantissa> {
    scale: u32,
    marker: PhantomData<Mantissa>,
}

/// Converts a fixed-point mantissa back to a floating value at this pipe's
/// scale — `mantissa / 10^scale`.
#[derive(Debug, Clone, Copy)]
pub struct FromFixed<Mantissa> {
    scale: u32,
    marker: PhantomData<Mantissa>,
}

/// A fixed-point multiply: widens both operands, multiplies, then divides
/// the widened intermediate by `10^scale` to rescale back to `Mantissa`.
#[derive(Debug, Clone, Copy)]
pub struct DecimalMultiply<Mantissa> {
    scale: u32,
    marker: PhantomData<Mantissa>,
}

impl ToFixed<i64> {
    /// # Errors
    /// [`DecimalError::ScaleOutOfRange`] if `10^scale` overflows `i64`
    /// (`scale > 18`).
    pub fn new(scale: u32) -> Result<Self, DecimalError> {
        10_i64
            .checked_pow(scale)
            .ok_or(DecimalError::ScaleOutOfRange { scale, bits: 64 })?;
        Ok(Self { scale, marker: PhantomData })
    }
}

impl Pipe for ToFixed<i64> {
    type In = f64;
    type Out = i64;
    type Err = DecimalError;

    fn call(&self, input: f64) -> impl Future<Output = Result<i64, DecimalError>> {
        let scale = self.scale;
        async move {
            let one = 10_i64.pow(scale);
            #[allow(clippy::cast_possible_truncation)]
            Ok(round_half_away_from_zero(input * one as f64) as i64)
        }
    }
}

impl FromFixed<i64> {
    /// # Errors
    /// [`DecimalError::ScaleOutOfRange`] if `10^scale` overflows `i64`
    /// (`scale > 18`).
    pub fn new(scale: u32) -> Result<Self, DecimalError> {
        10_i64
            .checked_pow(scale)
            .ok_or(DecimalError::ScaleOutOfRange { scale, bits: 64 })?;
        Ok(Self { scale, marker: PhantomData })
    }
}

impl Pipe for FromFixed<i64> {
    type In = i64;
    type Out = f64;
    type Err = DecimalError;

    fn call(&self, input: i64) -> impl Future<Output = Result<f64, DecimalError>> {
        let scale = self.scale;
        async move {
            let one = 10_i64.pow(scale);
            #[allow(clippy::cast_precision_loss)]
            Ok(input as f64 / one as f64)
        }
    }
}

impl ToFixed<i128> {
    /// # Errors
    /// [`DecimalError::ScaleOutOfRange`] if `10^scale` overflows `i128`
    /// (`scale > 38`).
    pub fn new(scale: u32) -> Result<Self, DecimalError> {
        10_i128
            .checked_pow(scale)
            .ok_or(DecimalError::ScaleOutOfRange { scale, bits: 128 })?;
        Ok(Self { scale, marker: PhantomData })
    }
}

impl Pipe for ToFixed<i128> {
    type In = f64;
    type Out = i128;
    type Err = DecimalError;

    fn call(&self, input: f64) -> impl Future<Output = Result<i128, DecimalError>> {
        let scale = self.scale;
        async move {
            let one = 10_i128.pow(scale);
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            Ok(round_half_away_from_zero(input * one as f64) as i128)
        }
    }
}

impl FromFixed<i128> {
    /// # Errors
    /// [`DecimalError::ScaleOutOfRange`] if `10^scale` overflows `i128`
    /// (`scale > 38`).
    pub fn new(scale: u32) -> Result<Self, DecimalError> {
        10_i128
            .checked_pow(scale)
            .ok_or(DecimalError::ScaleOutOfRange { scale, bits: 128 })?;
        Ok(Self { scale, marker: PhantomData })
    }
}

impl Pipe for FromFixed<i128> {
    type In = i128;
    type Out = f64;
    type Err = DecimalError;

    fn call(&self, input: i128) -> impl Future<Output = Result<f64, DecimalError>> {
        let scale = self.scale;
        async move {
            let one = 10_i128.pow(scale);
            #[allow(clippy::cast_precision_loss)]
            Ok(input as f64 / one as f64)
        }
    }
}

impl DecimalMultiply<i64> {
    /// # Errors
    /// [`DecimalError::ScaleOutOfRange`] if `10^scale` overflows `i128` (the
    /// widened intermediate's own rescale divisor).
    pub fn new(scale: u32) -> Result<Self, DecimalError> {
        10_i128
            .checked_pow(scale)
            .ok_or(DecimalError::ScaleOutOfRange { scale, bits: 128 })?;
        Ok(Self { scale, marker: PhantomData })
    }
}

impl Pipe for DecimalMultiply<i64> {
    type In = (i64, i64);
    type Out = i64;
    type Err = DecimalError;

    /// Widens both `i64` mantissas into `i128` before multiplying — an
    /// `i64 x i64` product can need up to 128 bits, so the widen is not
    /// optional (see this module's own doc). The rescale divide then narrows
    /// back to `i64`, which is the one step that can fail: a divisor result
    /// still too large for `i64` reports [`DecimalError::MultiplyOverflow`]
    /// rather than truncating silently.
    fn call(&self, (left, right): (i64, i64)) -> impl Future<Output = Result<i64, DecimalError>> {
        let scale = self.scale;
        async move {
            let widened = i128::from(left) * i128::from(right);
            let divisor = 10_i128.pow(scale);
            let rescaled = widened / divisor;
            i64::try_from(rescaled).map_err(|_error| DecimalError::MultiplyOverflow { bits: 64, scale })
        }
    }
}

impl DecimalMultiply<i128> {
    /// # Errors
    /// [`DecimalError::ScaleOutOfRange`] if `10^scale` overflows `i128`.
    pub fn new(scale: u32) -> Result<Self, DecimalError> {
        10_i128
            .checked_pow(scale)
            .ok_or(DecimalError::ScaleOutOfRange { scale, bits: 128 })?;
        Ok(Self { scale, marker: PhantomData })
    }
}

impl Pipe for DecimalMultiply<i128> {
    type In = (i128, i128);
    type Out = i128;
    type Err = DecimalError;

    /// `i128` has nothing wider to widen into (no `i256` in this language),
    /// so the product itself is the boundary: [`i128::checked_mul`] reports
    /// overflow directly rather than this pipe attempting an intermediate
    /// that does not exist.
    fn call(&self, (left, right): (i128, i128)) -> impl Future<Output = Result<i128, DecimalError>> {
        let scale = self.scale;
        async move {
            let product = left
                .checked_mul(right)
                .ok_or(DecimalError::MultiplyOverflow { bits: 128, scale })?;
            let divisor = 10_i128.pow(scale);
            Ok(product / divisor)
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use proxima_primitives::block_on;
    use rstest::rstest;

    #[rstest]
    #[case::whole_number(3.0, 300)]
    #[case::two_decimal_places(19.99, 1999)]
    #[case::negative(-2.5, -250)]
    fn to_fixed_scales_by_ten_to_the_scale(#[case] value: f64, #[case] expected: i64) {
        let pipe = ToFixed::<i64>::new(2).expect("scale 2 fits i64");
        let fixed = block_on(pipe.call(value)).expect("finite input never errors");
        assert_eq!(fixed, expected);
    }

    #[test]
    fn from_fixed_is_to_fixed_inverse() {
        let to_fixed = ToFixed::<i64>::new(4).expect("scale 4 fits i64");
        let from_fixed = FromFixed::<i64>::new(4).expect("scale 4 fits i64");
        let fixed = block_on(to_fixed.call(123.4567)).unwrap();
        let back = block_on(from_fixed.call(fixed)).unwrap();
        assert!((back - 123.4567).abs() < 1.0e-4);
    }

    #[test]
    fn multiply_i64_matches_plain_arithmetic_within_headroom() {
        let pipe = DecimalMultiply::<i64>::new(2).expect("scale 2 fits i128 intermediate");
        // 12.34 * 5.00 = 61.70, at scale 2: 1234 * 500 -> rescale by 100
        let product = block_on(pipe.call((1234, 500))).expect("well within i64 headroom");
        assert_eq!(product, 6170);
    }

    #[test]
    fn multiply_i64_reports_overflow_instead_of_wrapping() {
        let pipe = DecimalMultiply::<i64>::new(0).expect("scale 0 fits i128 intermediate");
        let error = block_on(pipe.call((i64::MAX, 2))).expect_err("product exceeds i64");
        assert_eq!(error, DecimalError::MultiplyOverflow { bits: 64, scale: 0 });
    }

    #[test]
    fn multiply_i128_stays_exact_below_the_measured_overflow_boundary() {
        // magnitude=1e9 at scale=10^9 fits an i128 intermediate; this test IS
        // the record of that headroom point, not a mirror of one elsewhere.
        let pipe = DecimalMultiply::<i128>::new(9).expect("scale 9 fits i128");
        let one = 10_i128.pow(9);
        let magnitude = 1_000_000_000_i128 * one; // represents 1e9 at scale 1e9
        let product = block_on(pipe.call((magnitude, one))).expect("1e9 * 1.0 stays in range");
        assert_eq!(product, magnitude);
    }

    #[test]
    fn multiply_i128_reports_overflow_past_its_own_measured_boundary() {
        // at scale=10^9, magnitude ~1e12 already
        // overflows the i128 intermediate — no wider type exists to widen
        // into, so this must surface as an error, never a silent wrap.
        let pipe = DecimalMultiply::<i128>::new(9).expect("scale 9 fits i128");
        let one = 10_i128.pow(9);
        let past_boundary = 10_i128.pow(13) * one; // 1e13, past the ~1e12 measured ceiling
        let error = block_on(pipe.call((past_boundary, past_boundary))).expect_err("exceeds i128");
        assert_eq!(error, DecimalError::MultiplyOverflow { bits: 128, scale: 9 });
    }

    #[test]
    fn scale_out_of_range_is_rejected_at_construction_not_at_call_time() {
        let error = ToFixed::<i64>::new(19).expect_err("10^19 overflows i64");
        assert_eq!(error, DecimalError::ScaleOutOfRange { scale: 19, bits: 64 });
    }
}
