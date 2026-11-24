//! How far apart two runs of the same format are: the metrics a model parity check reports.
//!
//! Every function takes [`PackedRows`] and returns a [`PackedRows`] holding the measurement, so a
//! caller never names an element type: the inputs decode into [`PackedRows::decode_all`]'s own
//! output, the sums run in that decoded element type (the accumulation width is the decoded
//! element's, not wider), and the answer is packed into [`NumericFormat::host_element`], whatever
//! the inputs' format. A narrow input format therefore never rounds the measured difference: a
//! cosine of 0.998 between two narrow runs reads 0.998, not 1.
//!
//! Sums are compensated (Kahan), so a long run does not drift, and an element count is the run's
//! length converted to the decoded element, not a running sum of ones.
//!
//! A measurement is one element (two for [`mean_std`]). A NaN or infinite input element is
//! [`NumericError::NonFiniteInput`]: a metric over it would be meaningless, and a fold that drops
//! NaN would report a finite number. A measurement that comes out non-finite from finite input (a
//! cosine of a zero vector, a relative error against an all-zero reference) is
//! [`NumericError::NonFinite`].
//!
//! Teaching pointer: each function is one pass over two decoded vectors; the transcendentals go
//! through `libm`, so the module builds at the no_std-with-alloc tier.

use alloc::vec::Vec;
use core::ops::{Add, Sub};

use crate::error::NumericError;
use crate::format::NumericFormat;
use crate::packed_rows::PackedRows;

macro_rules! decoded {
    ($rows:expr) => {{
        let mut out = Vec::new();
        $rows.decode_all(&mut out)?;
        out
    }};
}

macro_rules! tally {
    ($terms:expr) => {
        compensated($terms, libm::fabsf(0.0))
    };
}

macro_rules! count {
    ($values:expr) => {
        libm::fabsf($values.len() as _)
    };
}

macro_rules! report {
    ($($value:expr),+) => {
        PackedRows::from_values(&NumericFormat::host_element(), &[$($value),+])
    };
}

macro_rules! finite {
    ($what:expr, $($values:expr),+) => {
        if ![$($values.iter().all(|value| value.is_finite())),+]
            .iter()
            .all(|clean| *clean)
        {
            return Err(NumericError::NonFiniteInput { what: $what });
        }
    };
}

fn compensated<T>(terms: impl Iterator<Item = T>, zero: T) -> T
where
    T: Copy + Add<Output = T> + Sub<Output = T>,
{
    let (total, _) = terms.fold((zero, zero), |(total, lost), term| {
        let corrected = term - lost;
        let next = total + corrected;
        (next, (next - total) - corrected)
    });
    total
}

fn format_label(rows: &PackedRows) -> &'static str {
    let format = rows.format();
    format.name().unwrap_or(match format {
        NumericFormat::Scalar(_) => "unnamed scalar",
        NumericFormat::Block(_) => "unnamed block",
        NumericFormat::Levels(_) => "unnamed levels",
    })
}

fn host_name() -> &'static str {
    NumericFormat::host_element()
        .name()
        .unwrap_or("host element")
}

fn check(what: &'static str, left: &PackedRows, right: &PackedRows) -> Result<(), NumericError> {
    if left.format() != right.format() {
        return Err(NumericError::FormatMismatch {
            left: format_label(left),
            right: format_label(right),
        });
    }
    if left.elements() != right.elements() {
        return Err(NumericError::OutputSizeMismatch {
            found: right.elements(),
            expected: left.elements(),
        });
    }
    non_empty(what, left)
}

fn non_empty(what: &'static str, rows: &PackedRows) -> Result<(), NumericError> {
    if rows.is_empty() {
        return Err(NumericError::EmptyRun { what });
    }
    Ok(())
}

/// The largest absolute element difference between `a` and `b`, in the host element format.
///
/// # Errors
/// [`NumericError::FormatMismatch`] when the formats differ, [`NumericError::OutputSizeMismatch`]
/// when the lengths differ, [`NumericError::EmptyRun`] for empty runs,
/// [`NumericError::NonFiniteInput`] for a NaN or infinite element, and whatever decoding or
/// encoding the result reports.
pub fn max_abs_diff(a: &PackedRows, b: &PackedRows) -> Result<PackedRows, NumericError> {
    check("max_abs_diff", a, b)?;
    let (left, right) = (decoded!(a), decoded!(b));
    finite!("max_abs_diff", left, right);
    let worst = left
        .iter()
        .zip(&right)
        .map(|(x, y)| libm::fabsf(x - y))
        .fold(libm::fabsf(0.0), libm::fmaxf);
    report!(worst)
}

/// The cosine of the angle between `a` and `b`, as vectors; 1 for a run against itself.
///
/// # Errors
/// As [`max_abs_diff`]; a zero vector has no direction and reports [`NumericError::NonFinite`].
pub fn cosine(a: &PackedRows, b: &PackedRows) -> Result<PackedRows, NumericError> {
    check("cosine", a, b)?;
    let (left, right) = (decoded!(a), decoded!(b));
    finite!("cosine", left, right);
    let dot = tally!(left.iter().zip(&right).map(|(x, y)| x * y));
    let norm_left = tally!(left.iter().map(|x| x * x));
    let norm_right = tally!(right.iter().map(|y| y * y));
    let cosine = dot / libm::sqrtf(norm_left * norm_right);
    if !cosine.is_finite() {
        return Err(NumericError::NonFinite {
            format: host_name(),
        });
    }
    report!(cosine)
}

/// The root-mean-square of `a` alone, in the host element format.
///
/// # Errors
/// [`NumericError::EmptyRun`] for an empty run, [`NumericError::NonFiniteInput`] for a NaN or
/// infinite element, and whatever decoding or encoding the result reports.
pub fn rms(a: &PackedRows) -> Result<PackedRows, NumericError> {
    non_empty("rms", a)?;
    let values = decoded!(a);
    finite!("rms", values);
    let count = count!(values);
    let squares = tally!(values.iter().map(|x| x * x));
    report!(libm::sqrtf(squares / count))
}

/// The root-mean-square of `actual - reference`, relative to the root-mean-square of `reference`.
///
/// # Errors
/// As [`max_abs_diff`]; an all-zero `reference` reports [`NumericError::NonFinite`].
pub fn rms_rel(actual: &PackedRows, reference: &PackedRows) -> Result<PackedRows, NumericError> {
    check("rms_rel", actual, reference)?;
    let (left, right) = (decoded!(actual), decoded!(reference));
    finite!("rms_rel", left, right);
    let error = tally!(left.iter().zip(&right).map(|(x, y)| (x - y) * (x - y)));
    let squares = tally!(right.iter().map(|y| y * y));
    let relative = libm::sqrtf(error / squares);
    if squares <= 0.0 || !relative.is_finite() {
        return Err(NumericError::NonFinite {
            format: host_name(),
        });
    }
    report!(relative)
}

/// The peak signal-to-noise ratio of `a` against `b`, in dB: `20 log10(full_scale / rms(a - b))`.
///
/// `full_scale` is one element, in any format. Identical runs have zero error and an infinite
/// ratio, which the host element format holds; [`NumericError::NonFinite`] is reported only when
/// the ratio is NaN (a zero scale at zero error).
///
/// # Errors
/// As [`max_abs_diff`], and [`NumericError::OutputSizeMismatch`] when `full_scale` is not exactly
/// one element.
pub fn psnr(
    a: &PackedRows,
    b: &PackedRows,
    full_scale: &PackedRows,
) -> Result<PackedRows, NumericError> {
    check("psnr", a, b)?;
    let scales = decoded!(full_scale);
    let [scale] = scales.as_slice() else {
        return Err(NumericError::OutputSizeMismatch {
            found: scales.len(),
            expected: 1,
        });
    };
    let (left, right) = (decoded!(a), decoded!(b));
    finite!("psnr", left, right, scales);
    let count = count!(left);
    let error = tally!(left.iter().zip(&right).map(|(x, y)| (x - y) * (x - y)));
    let level = libm::sqrtf(error / count);
    let ratio = 20.0 * libm::log10f(scale / level);
    if ratio.is_nan() {
        return Err(NumericError::NonFinite {
            format: host_name(),
        });
    }
    report!(ratio)
}

/// The mean and the population standard deviation of `a`, as a two-element run in the host element
/// format.
///
/// # Errors
/// [`NumericError::EmptyRun`] for an empty run, [`NumericError::NonFiniteInput`] for a NaN or
/// infinite element, and whatever decoding or encoding the result reports.
pub fn mean_std(a: &PackedRows) -> Result<PackedRows, NumericError> {
    non_empty("mean_std", a)?;
    let values = decoded!(a);
    finite!("mean_std", values);
    let count = count!(values);
    let mean = tally!(values.iter().copied()) / count;
    let spread = tally!(values.iter().map(|x| (x - mean) * (x - mean))) / count;
    report!(mean, libm::sqrtf(spread))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::NumericFormat;
    use crate::rng::Philox;

    const SEED: u64 = 42;

    macro_rules! scalar_of {
        ($rows:expr) => {{
            let mut out = Vec::new();
            $rows.decode_all(&mut out).expect("the measurement decodes");
            out[0]
        }};
    }

    macro_rules! rows {
        ($format:expr, $values:expr) => {
            PackedRows::from_values($format, $values).expect("the format encodes these values")
        };
    }

    macro_rules! close {
        ($found:expr, $expected:expr, $tolerance:expr) => {
            libm::fabsf($found - $expected) <= $tolerance
        };
    }

    fn usable_formats() -> Vec<NumericFormat> {
        let probe = Philox::new(SEED).normal(64);
        NumericFormat::ALL
            .iter()
            .copied()
            .filter(|format| format.is_float() && format.elements_per_block() == 1)
            .filter(|format| {
                let mut back = Vec::new();
                PackedRows::from_values(format, &probe)
                    .and_then(|packed| packed.decode_all(&mut back))
                    .is_ok()
            })
            .collect()
    }

    fn widest() -> NumericFormat {
        usable_formats()
            .into_iter()
            .max_by_key(NumericFormat::bits)
            .expect("at least one scalar format round-trips")
    }

    #[test]
    fn a_run_against_itself_is_zero_apart_and_cosine_one_in_every_scalar_format() {
        let formats = usable_formats();
        assert!(!formats.is_empty(), "no scalar format to test");
        let noise = Philox::new(SEED).normal(256);
        for format in &formats {
            let packed = rows!(format, &noise);
            let apart = max_abs_diff(&packed, &packed).expect("a run compares with itself");
            let angle = cosine(&packed, &packed).expect("a run has a direction");
            assert!(close!(scalar_of!(apart), 0.0, 0.0), "{format:?} apart");
            assert!(
                close!(scalar_of!(angle), 1.0, 1.0 / 32.0),
                "{format:?} cosine"
            );
        }
    }

    #[test]
    fn identical_runs_have_an_infinite_psnr_in_the_host_element() {
        let format = widest();
        let noise = Philox::new(SEED).normal(256);
        let packed = rows!(&format, &noise);
        let scale = rows!(&format, &[1.0]);
        let ratio = psnr(&packed, &packed, &scale).expect("the host element holds an infinity");
        let value = scalar_of!(ratio);
        assert!(!value.is_nan() && value > 100.0, "psnr {value}");
    }

    #[test]
    fn hand_worked_values_come_back_in_the_widest_format() {
        let format = widest();
        let reference = rows!(&format, &[0.0, 1.0, 2.0, 3.0]);
        let actual = rows!(&format, &[0.0, 1.0, 2.0, 7.0]);
        let doubled = rows!(&format, &[0.0, 2.0, 4.0, 6.0]);
        let negated = rows!(&format, &[0.0, -1.0, -2.0, -3.0]);
        let scale = rows!(&format, &[8.0]);
        let worst = max_abs_diff(&actual, &reference).expect("same length");
        let level = psnr(&actual, &reference, &scale).expect("one scale");
        let parallel = cosine(&reference, &doubled).expect("same length");
        let opposite = cosine(&reference, &negated).expect("same length");
        let spread = mean_std(&rows!(&format, &[1.0, 2.0, 3.0, 4.0])).expect("four values");
        let relative = rms_rel(&actual, &reference).expect("same length");
        let level_alone = rms(&rows!(&format, &[3.0, 4.0])).expect("two values");
        assert!(close!(scalar_of!(worst), 4.0, 0.01));
        assert!(close!(scalar_of!(level), 12.041, 0.05));
        assert!(close!(scalar_of!(parallel), 1.0, 0.01));
        assert!(close!(scalar_of!(opposite), -1.0, 0.01));
        assert!(close!(scalar_of!(relative), 2.0 / libm::sqrtf(3.5), 0.01));
        assert!(close!(scalar_of!(level_alone), libm::sqrtf(12.5), 0.01));
        let mut pair = Vec::new();
        spread.decode_all(&mut pair).expect("the pair decodes");
        assert_eq!(pair.len(), 2);
        assert!(close!(pair[0], 2.5, 0.01));
        assert!(close!(pair[1], libm::sqrtf(1.25), 0.01));
    }

    #[test]
    fn philox_normals_have_mean_zero_std_one_and_no_correlation_across_draws() {
        let format = widest();
        let mut generator = Philox::new(SEED);
        let first = rows!(&format, &generator.normal(4096));
        let second = rows!(&format, &generator.normal(4096));
        let spread = mean_std(&first).expect("a long run");
        let across = cosine(&first, &second).expect("same length");
        let mut pair = Vec::new();
        spread.decode_all(&mut pair).expect("the pair decodes");
        assert!(close!(pair[0], 0.0, 0.1), "mean {}", pair[0]);
        assert!(close!(pair[1], 1.0, 0.1), "std {}", pair[1]);
        assert!(close!(scalar_of!(across), 0.0, 0.1));
    }

    #[test]
    fn a_length_mismatch_is_refused_by_every_comparison() {
        let format = widest();
        let long = rows!(&format, &[1.0, 2.0, 3.0, 4.0]);
        let short = rows!(&format, &[1.0, 2.0, 3.0]);
        let scale = rows!(&format, &[1.0]);
        let expected = Err(NumericError::OutputSizeMismatch {
            found: 3,
            expected: 4,
        });
        assert_eq!(max_abs_diff(&long, &short).map(|_| ()), expected);
        assert_eq!(cosine(&long, &short).map(|_| ()), expected);
        assert_eq!(rms_rel(&long, &short).map(|_| ()), expected);
        assert_eq!(psnr(&long, &short, &scale).map(|_| ()), expected);
    }

    #[test]
    fn two_formats_are_refused_by_name() {
        let formats = usable_formats();
        assert!(formats.len() >= 2, "need two scalar formats");
        let left = rows!(&formats[0], &[1.0, 2.0]);
        let right = rows!(&formats[1], &[1.0, 2.0]);
        let error = max_abs_diff(&left, &right)
            .map(|_| ())
            .expect_err("formats differ");
        assert!(
            matches!(error, NumericError::FormatMismatch { .. }),
            "{error}"
        );
    }

    #[test]
    fn an_empty_run_and_a_wrong_sized_full_scale_are_refused() {
        let format = widest();
        let empty = rows!(&format, &[]);
        let pair = rows!(&format, &[1.0, 2.0]);
        let two_scales = rows!(&format, &[1.0, 1.0]);
        assert_eq!(
            max_abs_diff(&empty, &empty).map(|_| ()),
            Err(NumericError::EmptyRun {
                what: "max_abs_diff"
            })
        );
        assert_eq!(
            rms(&empty).map(|_| ()),
            Err(NumericError::EmptyRun { what: "rms" })
        );
        assert_eq!(
            mean_std(&empty).map(|_| ()),
            Err(NumericError::EmptyRun { what: "mean_std" })
        );
        assert_eq!(
            psnr(&pair, &pair, &two_scales).map(|_| ()),
            Err(NumericError::OutputSizeMismatch {
                found: 2,
                expected: 1
            })
        );
    }

    #[test]
    fn a_zero_vector_has_no_cosine() {
        let format = widest();
        let zeros = rows!(&format, &[0.0, 0.0]);
        let error = cosine(&zeros, &zeros)
            .map(|_| ())
            .expect_err("0/0 is not a number");
        assert!(matches!(error, NumericError::NonFinite { .. }), "{error}");
    }

    #[test]
    fn a_block_input_reports_its_mean_std_in_the_host_element() {
        let block = NumericFormat::ALL
            .iter()
            .filter(|format| format.elements_per_block() > 1)
            .find_map(|format| {
                let count = usize::from(format.elements_per_block());
                PackedRows::from_values(format, &Philox::new(SEED).normal(count)).ok()
            })
            .expect("at least one block format encodes");
        let spread = mean_std(&block).expect("a block decodes and the host element holds two");
        assert_eq!(spread.elements(), 2);
        assert!(spread.format().is_host_element());
    }

    macro_rules! nan {
        () => {
            libm::fabsf(0.0) / libm::fabsf(0.0)
        };
    }

    #[test]
    fn a_nan_input_is_refused_by_every_metric_that_reads_it() {
        let host = NumericFormat::host_element();
        let clean = rows!(&host, &[1.0, 2.0, 3.0]);
        let poisoned = rows!(&host, &[1.0, nan!(), 3.0]);
        let scale = rows!(&host, &[1.0]);
        let refused = |what| Err(NumericError::NonFiniteInput { what });
        assert_eq!(
            max_abs_diff(&poisoned, &clean).map(|_| ()),
            refused("max_abs_diff")
        );
        assert_eq!(rms_rel(&poisoned, &clean).map(|_| ()), refused("rms_rel"));
        assert_eq!(rms_rel(&clean, &poisoned).map(|_| ()), refused("rms_rel"));
        assert_eq!(cosine(&clean, &poisoned).map(|_| ()), refused("cosine"));
        assert_eq!(rms(&poisoned).map(|_| ()), refused("rms"));
        assert_eq!(mean_std(&poisoned).map(|_| ()), refused("mean_std"));
        assert_eq!(
            psnr(&poisoned, &clean, &scale).map(|_| ()),
            refused("psnr")
        );
    }

    #[test]
    fn an_all_zero_reference_has_no_relative_rms() {
        let host = NumericFormat::host_element();
        let actual = rows!(&host, &[1.0, 2.0]);
        let zeros = rows!(&host, &[0.0, 0.0]);
        let error = rms_rel(&actual, &zeros)
            .map(|_| ())
            .expect_err("dividing by a zero reference");
        assert!(matches!(error, NumericError::NonFinite { .. }), "{error}");
    }

    #[test]
    fn a_narrow_input_reports_its_measurement_in_the_host_element_unrounded() {
        let host = NumericFormat::host_element();
        let probe = [1.0, 0.0, 0.0625];
        let narrow: Vec<NumericFormat> = usable_formats()
            .into_iter()
            .filter(|format| format.bits() < host.bits())
            .filter(|format| {
                let mut back = Vec::new();
                rows!(format, &probe)
                    .decode_all(&mut back)
                    .is_ok_and(|()| back == probe)
            })
            .collect();
        assert!(!narrow.is_empty(), "no narrow scalar format to test");
        let expected = 1.0 / libm::sqrtf(1.0 + 0.0625 * 0.0625);
        for format in &narrow {
            let left = rows!(format, &[1.0, 0.0]);
            let right = rows!(format, &[1.0, 0.0625]);
            let angle = cosine(&left, &right).expect("same length");
            let worst = max_abs_diff(&left, &right).expect("same length");
            assert!(angle.format().is_host_element(), "{format:?} cosine");
            assert!(worst.format().is_host_element(), "{format:?} max_abs_diff");
            assert!(close!(scalar_of!(angle), expected, 1.0e-6), "{format:?}");
            assert!(close!(scalar_of!(worst), 0.0625, 0.0), "{format:?}");
        }
    }
}
