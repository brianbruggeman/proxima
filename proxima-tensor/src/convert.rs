//! Element conversion as an ordinary [`Pipe`] — `In` = source scalar, `Out` =
//! target scalar, no new trait. [`Convert`] is the one generic type: a
//! `PhantomData` marker, monomorphized per pair by one concrete `Pipe` impl
//! per pair of the machine scalar types
//! ([`crate::dtype::DType`]'s `Int8`/`UInt8`/`Int32`/`UInt32`/`BFloat16`/
//! `Float16`/`Float32`, `Bool` as `bool`) the caller names. A pair not wired
//! directly (say `Int8 -> Float32`) composes for free through
//! `proxima_primitives::PipeExt::and_then` (`Int8 -> Int32 -> Float32`) —
//! that is the algebra doing its job, not a gap. `SimdConvert`'s scalar
//! fallback (below) calls straight through this same `Pipe::call` via
//! `block_on`, so there is exactly one place per pair the conversion body
//! is written.
//!
//! [`SimdConvert::convert_slice`] is the bulk sibling: a register's worth of
//! elements per step (NEON on `aarch64`, where a lane type exists for the
//! pair) instead of one `Pipe::call` per element. `i64 -> i128`/`u64 -> u128`
//! have no NEON lane type — aarch64 NEON tops out at 64-bit integer lanes —
//! so those two, and every pair this module did not hand-write a kernel for,
//! fall back to the same scalar loop `Pipe::call` already performs; that is
//! the honest ceiling, not an oversight.
//!
//! bf16 and IEEE binary16 both need a marker distinct from a bare `u16` —
//! bf16's truncated 8-bit exponent and binary16's 5-bit exponent are two
//! incompatible encodings, so `u16` alone cannot say which one a value
//! holds. This module used to mint that marker itself (`Bf16`/`F16`);
//! it now reuses [`half::bf16`]/[`half::f16`] instead — a sibling worktree
//! (`omega`) already depends on `half` for GPU marshalling, so minting a
//! second incompatible representation of the same concept would only
//! collide at merge. `half`'s conversions are also correctly
//! round-to-nearest-even on ties, which this module's own hand-rolled
//! binary16 rounding was not (see `git log` on this file for the bug).
//!
//! Decimal is a conversion, not a `DType` variant — see [`decimal`].

use core::convert::Infallible;
use core::future::Future;
use core::marker::PhantomData;

use half::{bf16, f16};
use proxima_primitives::block_on;
use proxima_primitives::pipe::Pipe;

/// A conversion between two scalar element types, as data: `In` = source,
/// `Out` = target. No state, no allocation — every instance is
/// interchangeable, so construction is a `PhantomData` marker, not a value
/// worth threading through a caller's own type.
#[derive(Debug, Clone, Copy, Default)]
pub struct Convert<From, To> {
    marker: PhantomData<fn(From) -> To>,
}

impl<From, To> Convert<From, To> {
    #[must_use]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

// EXPERIMENT (question A): `Cast<To>` deleted. Each pair now carries its
// conversion directly on a concrete `impl Pipe for Convert<From, To>` —
// `SimdConvert`'s scalar paths (below) call THROUGH this `Pipe::call`
// (via `block_on`) instead of a bare `.cast()`, so there is exactly one
// place per pair the conversion body is written, not two.
macro_rules! impl_convert_pipe_as {
    ($From:ty => $To:ty) => {
        impl Pipe for Convert<$From, $To> {
            type In = $From;
            type Out = $To;
            type Err = Infallible;

            #[inline(always)]
            fn call(&self, input: $From) -> impl Future<Output = Result<$To, Infallible>> {
                async move { Ok(input as $To) }
            }
        }
    };
}

impl_convert_pipe_as!(i8 => i16);
impl_convert_pipe_as!(i16 => i32);
impl_convert_pipe_as!(i32 => i64);
impl_convert_pipe_as!(i64 => i128);
impl_convert_pipe_as!(u8 => u16);
impl_convert_pipe_as!(u16 => u32);
impl_convert_pipe_as!(u32 => u64);
impl_convert_pipe_as!(u64 => u128);
impl_convert_pipe_as!(i16 => i8);
impl_convert_pipe_as!(i32 => i16);
impl_convert_pipe_as!(i64 => i32);
impl_convert_pipe_as!(i128 => i64);
impl_convert_pipe_as!(u16 => u8);
impl_convert_pipe_as!(u32 => u16);
impl_convert_pipe_as!(u64 => u32);
impl_convert_pipe_as!(u128 => u64);
impl_convert_pipe_as!(i8 => i32);
impl_convert_pipe_as!(i32 => i8);
impl_convert_pipe_as!(u8 => u32);
impl_convert_pipe_as!(u32 => u8);
impl_convert_pipe_as!(i32 => f32);
impl_convert_pipe_as!(f32 => i32);
impl_convert_pipe_as!(u32 => f32);
impl_convert_pipe_as!(f32 => u32);
impl_convert_pipe_as!(i8 => u8);
impl_convert_pipe_as!(u8 => i8);
impl_convert_pipe_as!(i32 => u32);
impl_convert_pipe_as!(u32 => i32);
impl_convert_pipe_as!(f32 => f64);
impl_convert_pipe_as!(f64 => f32);

impl Pipe for Convert<bool, u8> {
    type In = bool;
    type Out = u8;
    type Err = Infallible;

    fn call(&self, input: bool) -> impl Future<Output = Result<u8, Infallible>> {
        async move { Ok(input as u8) }
    }
}
impl Pipe for Convert<u8, bool> {
    type In = u8;
    type Out = bool;
    type Err = Infallible;

    fn call(&self, input: u8) -> impl Future<Output = Result<bool, Infallible>> {
        async move { Ok(input != 0) }
    }
}
impl Pipe for Convert<bool, i8> {
    type In = bool;
    type Out = i8;
    type Err = Infallible;

    fn call(&self, input: bool) -> impl Future<Output = Result<i8, Infallible>> {
        async move { Ok(input as i8) }
    }
}
impl Pipe for Convert<i8, bool> {
    type In = i8;
    type Out = bool;
    type Err = Infallible;

    fn call(&self, input: i8) -> impl Future<Output = Result<bool, Infallible>> {
        async move { Ok(input != 0) }
    }
}

impl Pipe for Convert<f32, bf16> {
    type In = f32;
    type Out = bf16;
    type Err = Infallible;

    fn call(&self, input: f32) -> impl Future<Output = Result<bf16, Infallible>> {
        async move { Ok(bf16::from_f32(input)) }
    }
}
impl Pipe for Convert<bf16, f32> {
    type In = bf16;
    type Out = f32;
    type Err = Infallible;

    fn call(&self, input: bf16) -> impl Future<Output = Result<f32, Infallible>> {
        async move { Ok(input.to_f32()) }
    }
}
impl Pipe for Convert<f32, f16> {
    type In = f32;
    type Out = f16;
    type Err = Infallible;

    fn call(&self, input: f32) -> impl Future<Output = Result<f16, Infallible>> {
        async move { Ok(f16::from_f32(input)) }
    }
}
impl Pipe for Convert<f16, f32> {
    type In = f16;
    type Out = f32;
    type Err = Infallible;

    fn call(&self, input: f16) -> impl Future<Output = Result<f32, Infallible>> {
        async move { Ok(input.to_f32()) }
    }
}

/// The bulk sibling of [`Pipe::call`]: converts a whole slice, a register's
/// worth of elements per step where a NEON lane type exists for the pair —
/// see this module's own doc for which pairs that is and why the rest fall
/// back to a scalar loop.
pub trait SimdConvert: Pipe {
    /// # Panics
    /// If `input.len() != output.len()`.
    fn convert_slice(&self, input: &[Self::In], output: &mut [Self::Out]);
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        vcvtq_f32_s32, vcvtq_f32_u32, vcvtq_s32_f32, vcvtq_u32_f32, vget_high_s16, vget_high_s32,
        vget_high_u16, vget_high_u32, vget_low_s16, vget_low_s32, vget_low_u16, vget_low_u32, vld1_s8,
        vld1_u8, vld1q_f32, vld1q_s16, vld1q_s32, vld1q_u16, vld1q_u32, vmovl_s8, vmovl_s16, vmovl_s32,
        vmovl_u8, vmovl_u16, vmovl_u32, vst1q_f32, vst1q_s16, vst1q_s32, vst1q_s64, vst1q_u16, vst1q_u32,
        vst1q_u64,
    };

    /// # Safety
    /// `input` and `output` must both be exactly 8 elements.
    pub unsafe fn i8_to_i16(input: &[i8], output: &mut [i16]) {
        unsafe {
            let wide = vmovl_s8(vld1_s8(input.as_ptr()));
            vst1q_s16(output.as_mut_ptr(), wide);
        }
    }

    /// # Safety
    /// `input` and `output` must both be exactly 8 elements.
    pub unsafe fn u8_to_u16(input: &[u8], output: &mut [u16]) {
        unsafe {
            let wide = vmovl_u8(vld1_u8(input.as_ptr()));
            vst1q_u16(output.as_mut_ptr(), wide);
        }
    }

    /// # Safety
    /// `input` and `output` must both be exactly 8 elements.
    pub unsafe fn i16_to_i32(input: &[i16], output: &mut [i32]) {
        unsafe {
            let full = vld1q_s16(input.as_ptr());
            vst1q_s32(output.as_mut_ptr(), vmovl_s16(vget_low_s16(full)));
            vst1q_s32(output.as_mut_ptr().add(4), vmovl_s16(vget_high_s16(full)));
        }
    }

    /// # Safety
    /// `input` and `output` must both be exactly 8 elements.
    pub unsafe fn u16_to_u32(input: &[u16], output: &mut [u32]) {
        unsafe {
            let full = vld1q_u16(input.as_ptr());
            vst1q_u32(output.as_mut_ptr(), vmovl_u16(vget_low_u16(full)));
            vst1q_u32(output.as_mut_ptr().add(4), vmovl_u16(vget_high_u16(full)));
        }
    }

    /// # Safety
    /// `input` and `output` must both be exactly 4 elements.
    pub unsafe fn i32_to_i64(input: &[i32], output: &mut [i64]) {
        unsafe {
            let full = vld1q_s32(input.as_ptr());
            vst1q_s64(output.as_mut_ptr(), vmovl_s32(vget_low_s32(full)));
            vst1q_s64(output.as_mut_ptr().add(2), vmovl_s32(vget_high_s32(full)));
        }
    }

    /// # Safety
    /// `input` and `output` must both be exactly 4 elements.
    pub unsafe fn u32_to_u64(input: &[u32], output: &mut [u64]) {
        unsafe {
            let full = vld1q_u32(input.as_ptr());
            vst1q_u64(output.as_mut_ptr(), vmovl_u32(vget_low_u32(full)));
            vst1q_u64(output.as_mut_ptr().add(2), vmovl_u32(vget_high_u32(full)));
        }
    }

    /// # Safety
    /// `input` and `output` must both be exactly 8 elements.
    pub unsafe fn i8_to_i32(input: &[i8], output: &mut [i32]) {
        unsafe {
            let mid = vmovl_s8(vld1_s8(input.as_ptr()));
            vst1q_s32(output.as_mut_ptr(), vmovl_s16(vget_low_s16(mid)));
            vst1q_s32(output.as_mut_ptr().add(4), vmovl_s16(vget_high_s16(mid)));
        }
    }

    /// # Safety
    /// `input` and `output` must both be exactly 8 elements.
    pub unsafe fn u8_to_u32(input: &[u8], output: &mut [u32]) {
        unsafe {
            let mid = vmovl_u8(vld1_u8(input.as_ptr()));
            vst1q_u32(output.as_mut_ptr(), vmovl_u16(vget_low_u16(mid)));
            vst1q_u32(output.as_mut_ptr().add(4), vmovl_u16(vget_high_u16(mid)));
        }
    }

    /// # Safety
    /// `input` and `output` must both be exactly 4 elements.
    pub unsafe fn i32_to_f32(input: &[i32], output: &mut [f32]) {
        unsafe {
            vst1q_f32(output.as_mut_ptr(), vcvtq_f32_s32(vld1q_s32(input.as_ptr())));
        }
    }

    /// # Safety
    /// `input` and `output` must both be exactly 4 elements.
    pub unsafe fn u32_to_f32(input: &[u32], output: &mut [f32]) {
        unsafe {
            vst1q_f32(output.as_mut_ptr(), vcvtq_f32_u32(vld1q_u32(input.as_ptr())));
        }
    }

    /// # Safety
    /// `input` and `output` must both be exactly 4 elements.
    pub unsafe fn f32_to_i32(input: &[f32], output: &mut [i32]) {
        unsafe {
            vst1q_s32(output.as_mut_ptr(), vcvtq_s32_f32(vld1q_f32(input.as_ptr())));
        }
    }

    /// # Safety
    /// `input` and `output` must both be exactly 4 elements.
    pub unsafe fn f32_to_u32(input: &[f32], output: &mut [u32]) {
        unsafe {
            vst1q_u32(output.as_mut_ptr(), vcvtq_u32_f32(vld1q_f32(input.as_ptr())));
        }
    }
}

macro_rules! impl_simd_convert_neon {
    ($From:ty => $To:ty, lanes = $lanes:expr, kernel = $kernel:path) => {
        impl SimdConvert for Convert<$From, $To> {
            #[cfg(target_arch = "aarch64")]
            fn convert_slice(&self, input: &[$From], output: &mut [$To]) {
                assert_eq!(input.len(), output.len(), "convert_slice: length mismatch");
                let lanes = $lanes;
                let mut processed = 0_usize;
                while processed + lanes <= input.len() {
                    // SAFETY: the window is exactly `lanes` wide, matching
                    // the load/store width the kernel above uses.
                    unsafe {
                        $kernel(
                            &input[processed..processed + lanes],
                            &mut output[processed..processed + lanes],
                        );
                    }
                    processed += lanes;
                }
                for index in processed..input.len() {
                    output[index] = block_on(self.call(input[index])).expect("Convert::call is Infallible");
                }
            }

            #[cfg(not(target_arch = "aarch64"))]
            fn convert_slice(&self, input: &[$From], output: &mut [$To]) {
                assert_eq!(input.len(), output.len(), "convert_slice: length mismatch");
                for (source, target) in input.iter().zip(output.iter_mut()) {
                    *target = block_on(self.call(*source)).expect("Convert::call is Infallible");
                }
            }
        }
    };
}

macro_rules! impl_simd_convert_scalar {
    ($From:ty => $To:ty) => {
        impl SimdConvert for Convert<$From, $To> {
            fn convert_slice(&self, input: &[$From], output: &mut [$To]) {
                assert_eq!(input.len(), output.len(), "convert_slice: length mismatch");
                for (source, target) in input.iter().zip(output.iter_mut()) {
                    *target = block_on(self.call(*source)).expect("Convert::call is Infallible");
                }
            }
        }
    };
}

impl_simd_convert_neon!(i8 => i16, lanes = 8, kernel = neon::i8_to_i16);
impl_simd_convert_neon!(u8 => u16, lanes = 8, kernel = neon::u8_to_u16);
impl_simd_convert_neon!(i16 => i32, lanes = 8, kernel = neon::i16_to_i32);
impl_simd_convert_neon!(u16 => u32, lanes = 8, kernel = neon::u16_to_u32);
impl_simd_convert_neon!(i32 => i64, lanes = 4, kernel = neon::i32_to_i64);
impl_simd_convert_neon!(u32 => u64, lanes = 4, kernel = neon::u32_to_u64);
impl_simd_convert_neon!(i8 => i32, lanes = 8, kernel = neon::i8_to_i32);
impl_simd_convert_neon!(u8 => u32, lanes = 8, kernel = neon::u8_to_u32);
impl_simd_convert_neon!(i32 => f32, lanes = 4, kernel = neon::i32_to_f32);
impl_simd_convert_neon!(u32 => f32, lanes = 4, kernel = neon::u32_to_f32);
impl_simd_convert_neon!(f32 => i32, lanes = 4, kernel = neon::f32_to_i32);
impl_simd_convert_neon!(f32 => u32, lanes = 4, kernel = neon::f32_to_u32);

// no aarch64 NEON lane type past 64-bit integers — i128/u128 are the honest
// scalar ceiling the task's own measurement names, not an oversight.
impl_simd_convert_scalar!(i64 => i128);
impl_simd_convert_scalar!(u64 => u128);
impl_simd_convert_scalar!(i16 => i8);
impl_simd_convert_scalar!(i32 => i16);
impl_simd_convert_scalar!(i64 => i32);
impl_simd_convert_scalar!(i128 => i64);
impl_simd_convert_scalar!(u16 => u8);
impl_simd_convert_scalar!(u32 => u16);
impl_simd_convert_scalar!(u64 => u32);
impl_simd_convert_scalar!(u128 => u64);
impl_simd_convert_scalar!(i32 => i8);
impl_simd_convert_scalar!(u32 => u8);
impl_simd_convert_scalar!(i8 => u8);
impl_simd_convert_scalar!(u8 => i8);
impl_simd_convert_scalar!(i32 => u32);
impl_simd_convert_scalar!(u32 => i32);
impl_simd_convert_scalar!(f32 => f64);
impl_simd_convert_scalar!(f64 => f32);
impl_simd_convert_scalar!(bool => u8);
impl_simd_convert_scalar!(u8 => bool);
impl_simd_convert_scalar!(bool => i8);
impl_simd_convert_scalar!(i8 => bool);
impl_simd_convert_scalar!(f32 => bf16);
impl_simd_convert_scalar!(bf16 => f32);
impl_simd_convert_scalar!(f32 => f16);
impl_simd_convert_scalar!(f16 => f32);

pub mod decimal;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use proxima_primitives::block_on;

    #[test]
    fn widening_ladder_round_trips_a_small_value() {
        let start: i8 = 42;
        let as_i16: i16 = block_on(Convert::<i8, i16>::new().call(start)).unwrap();
        let as_i32: i32 = block_on(Convert::<i16, i32>::new().call(as_i16)).unwrap();
        let as_i64: i64 = block_on(Convert::<i32, i64>::new().call(as_i32)).unwrap();
        let as_i128: i128 = block_on(Convert::<i64, i128>::new().call(as_i64)).unwrap();
        assert_eq!(as_i128, 42);
    }

    #[test]
    fn narrowing_truncates_like_a_bare_as_cast() {
        let wide: i32 = 0x1234_5678_u32 as i32;
        let narrow: i8 = block_on(Convert::<i32, i8>::new().call(300)).unwrap();
        assert_eq!(narrow, 300_i32 as i8);
        let narrow_wide: i8 = block_on(Convert::<i32, i8>::new().call(wide)).unwrap();
        assert_eq!(narrow_wide, wide as i8);
    }

    #[proxima::test]
    #[case::zero(0.0)]
    #[case::one(1.0)]
    #[case::negative(-3.5)]
    #[case::saturates_high(1.0e30)]
    #[case::saturates_low(-1.0e30)]
    async fn float_to_int_saturates_never_panics(#[case] value: f32) {
        let converted: i32 = block_on(Convert::<f32, i32>::new().call(value)).unwrap();
        assert_eq!(converted, value as i32);
    }

    #[test]
    fn convert_slice_matches_scalar_call_including_remainder() {
        // 21 is not a multiple of the i8->i32 kernel's 8-lane width, so this
        // forces the scalar remainder tail alongside the vectorized body.
        let input: Vec<i8> = (0..21_i32).map(|value| (value - 10) as i8).collect();
        let mut via_slice = vec![0_i32; input.len()];
        Convert::<i8, i32>::new().convert_slice(&input, &mut via_slice);

        let mut via_scalar = vec![0_i32; input.len()];
        for (source, target) in input.iter().zip(via_scalar.iter_mut()) {
            *target = block_on(Convert::<i8, i32>::new().call(*source)).unwrap();
        }
        assert_eq!(via_slice, via_scalar);
    }

    #[proxima::test]
    #[case::i8_i16(17)]
    #[case::exact_multiple(16)]
    #[case::single_remainder(9)]
    async fn simd_ladder_pairs_agree_with_scalar(#[case] length: usize) {
        let input: Vec<i16> = (0..length as i32).map(|value| value as i16).collect();
        let mut via_slice = vec![0_i32; length];
        Convert::<i16, i32>::new().convert_slice(&input, &mut via_slice);
        let expected: Vec<i32> = input.iter().map(|value| i32::from(*value)).collect();
        assert_eq!(via_slice, expected);
    }

    #[test]
    fn bool_pipes_round_trip() {
        let as_u8: u8 = block_on(Convert::<bool, u8>::new().call(true)).unwrap();
        assert_eq!(as_u8, 1);
        let as_bool: bool = block_on(Convert::<u8, bool>::new().call(5)).unwrap();
        assert!(as_bool, "any nonzero byte is true");
        let as_bool_zero: bool = block_on(Convert::<u8, bool>::new().call(0)).unwrap();
        assert!(!as_bool_zero);
    }

    #[proxima::test]
    #[case::zero(0.0)]
    #[case::one(1.0)]
    #[case::negative_fraction(-0.25)]
    #[case::small(1.0e-30)]
    async fn bf16_round_trips_representable_values(#[case] value: f32) {
        let narrowed: bf16 = block_on(Convert::<f32, bf16>::new().call(value)).unwrap();
        let widened: f32 = block_on(Convert::<bf16, f32>::new().call(narrowed)).unwrap();
        // bf16 keeps f32's exponent and drops 16 mantissa bits, so a value
        // whose low mantissa bits are already zero round-trips exactly.
        if value == 0.0 || value == 1.0 {
            assert_eq!(widened, value);
        } else {
            assert!((widened - value).abs() / value.abs() < 0.01);
        }
    }

    #[test]
    fn bf16_preserves_nan_and_infinity() {
        let nan: bf16 = block_on(Convert::<f32, bf16>::new().call(f32::NAN)).unwrap();
        let back: f32 = block_on(Convert::<bf16, f32>::new().call(nan)).unwrap();
        assert!(back.is_nan());

        let infinite: bf16 = block_on(Convert::<f32, bf16>::new().call(f32::INFINITY)).unwrap();
        let back_infinite: f32 = block_on(Convert::<bf16, f32>::new().call(infinite)).unwrap();
        assert_eq!(back_infinite, f32::INFINITY);
    }

    #[proxima::test]
    #[case::zero(0.0)]
    #[case::one(1.0)]
    #[case::negative(-2.5)]
    #[case::max_normal(65504.0)]
    #[case::small_subnormal(0.000_061_035_156)]
    async fn f16_round_trips_representable_values(#[case] value: f32) {
        let narrowed: f16 = block_on(Convert::<f32, f16>::new().call(value)).unwrap();
        let widened: f32 = block_on(Convert::<f16, f32>::new().call(narrowed)).unwrap();
        assert_eq!(widened, value, "every value in this table is exactly representable in f16");
    }

    #[test]
    fn f16_saturates_past_max_magnitude_to_infinity() {
        let narrowed: f16 = block_on(Convert::<f32, f16>::new().call(1.0e6)).unwrap();
        let widened: f32 = block_on(Convert::<f16, f32>::new().call(narrowed)).unwrap();
        assert_eq!(widened, f32::INFINITY);
    }

    #[test]
    fn f16_flushes_values_below_subnormal_range_to_zero() {
        let narrowed: f16 = block_on(Convert::<f32, f16>::new().call(1.0e-10)).unwrap();
        let widened: f32 = block_on(Convert::<f16, f32>::new().call(narrowed)).unwrap();
        assert_eq!(widened, 0.0);
    }

    #[test]
    fn f16_preserves_nan_and_infinity() {
        let nan: f16 = block_on(Convert::<f32, f16>::new().call(f32::NAN)).unwrap();
        let back: f32 = block_on(Convert::<f16, f32>::new().call(nan)).unwrap();
        assert!(back.is_nan());

        let infinite: f16 = block_on(Convert::<f32, f16>::new().call(f32::NEG_INFINITY)).unwrap();
        let back_infinite: f32 = block_on(Convert::<f16, f32>::new().call(infinite)).unwrap();
        assert_eq!(back_infinite, f32::NEG_INFINITY);
    }

    #[proxima::test]
    #[case::rounds_down_kept_lsb_even(1.000_488_3, 0x3c00)]
    #[case::rounds_up_kept_lsb_odd(1.002_441_4, 0x3c02)]
    async fn f16_normal_path_ties_round_to_nearest_even(
        #[case] value: f32,
        #[case] expected_bits: u16,
    ) {
        // both `value`s sit exactly halfway between two representable f16
        // values (dropped mantissa bits exactly 0x1000) — round-half-even
        // must round to the neighbor with an even kept LSB, never always up.
        let narrowed: f16 = block_on(Convert::<f32, f16>::new().call(value)).unwrap();
        assert_eq!(
            narrowed.to_bits(),
            expected_bits,
            "tie must resolve to the even neighbor, not always round up"
        );
    }

    #[test]
    fn compose_through_and_then_reaches_a_pair_with_no_direct_pipe() {
        use proxima_primitives::PipeExt;
        // Int8 -> Float32 has no direct Convert impl; it composes through
        // Int32, exactly the algebra's job, not a gap in this module.
        let chain = Convert::<i8, i32>::new().and_then(Convert::<i32, f32>::new());
        let result: f32 = block_on(chain.call(-5)).unwrap();
        assert_eq!(result, -5.0);
    }
}
