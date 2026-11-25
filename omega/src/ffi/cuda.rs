//! The CUDA driver's event-timing boundary: the one place omega receives the float the driver
//! reports between two events. It crosses as bytes and is decoded through proxima-numeric's own
//! format identity, so no scalar type is named here. Callers get a [`Duration`].

use core::time::Duration;

use cudarc::driver::{CudaEvent, DriverError};
use proxima_numeric::{NumericFormat, decode_scalar};

// the C ABI float the driver writes through `cuEventElapsedTime`
const DRIVER_FLOAT_BYTES: usize = core::mem::size_of::<core::ffi::c_float>();

/// Time between two recorded events, as the driver reports it.
///
/// # Errors
/// An event has not completed or the driver cannot time it.
pub fn event_elapsed(start: &CudaEvent, end: &CudaEvent) -> Result<Duration, DriverError> {
    start
        .elapsed_ms(end)
        .map(|millis| duration_from_driver_bytes(millis.to_ne_bytes()))
}

// the driver reports milliseconds; a negative reading clamps to zero through the saturating cast,
// and bytes that do not decode read as zero time
fn duration_from_driver_bytes(raw: [u8; DRIVER_FLOAT_BYTES]) -> Duration {
    match decode_scalar(&NumericFormat::host_element(), &raw) {
        Ok(millis) => Duration::from_micros((millis * 1_000.0).round() as u64),
        Err(_) => Duration::ZERO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_float_is_exactly_one_host_element() {
        let host = NumericFormat::host_element();
        assert!(host.is_host_element());
        assert_eq!(host.packed_len(1), Some(DRIVER_FLOAT_BYTES));
    }

    #[test]
    fn driver_reporting_one_and_a_half_milliseconds_is_1500_microseconds() {
        let raw = <core::ffi::c_float>::to_ne_bytes(1.5);
        assert_eq!(duration_from_driver_bytes(raw).as_micros(), 1_500);
    }

    #[test]
    fn a_negative_driver_reading_clamps_to_zero() {
        let raw = <core::ffi::c_float>::to_ne_bytes(-0.25);
        assert_eq!(duration_from_driver_bytes(raw), Duration::ZERO);
    }
}
