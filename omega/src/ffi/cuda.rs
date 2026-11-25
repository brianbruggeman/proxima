//! The CUDA driver's event-timing boundary: the one place omega names or converts the float the
//! driver reports between two events. Callers get a [`Duration`].

use core::time::Duration;

use cudarc::driver::{CudaEvent, DriverError};

/// Time between two recorded events, as the driver reports it.
///
/// # Errors
/// An event has not completed or the driver cannot time it.
pub fn event_elapsed(start: &CudaEvent, end: &CudaEvent) -> Result<Duration, DriverError> {
    start.elapsed_ms(end).map(duration_from_driver_millis)
}

// the driver reports milliseconds; a negative reading clamps to zero through the saturating cast
fn duration_from_driver_millis(millis: f32) -> Duration {
    Duration::from_micros((millis * 1_000.0).round() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_reporting_one_and_a_half_milliseconds_is_1500_microseconds() {
        assert_eq!(duration_from_driver_millis(1.5).as_micros(), 1_500);
    }

    #[test]
    fn a_negative_driver_reading_clamps_to_zero() {
        assert_eq!(duration_from_driver_millis(-0.25), Duration::ZERO);
    }
}
