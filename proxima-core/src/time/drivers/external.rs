//! Link-time external driver. `now` / `schedule_wake` are provided by
//! *another crate* in the final binary via two `extern "Rust"` symbols —
//! there is NO cargo dependency edge, so a per-core runtime (prime) can
//! back `proxima_core::time`'s timer without the `prime -> proxima-pipe ->
//! proxima-core -> prime` cycle a direct dep would form. This is the
//! `#[global_allocator]` pattern applied to the timer driver: upstream
//! references the symbol, downstream provides it, the linker ties them.
//!
//! Selected by the `time-driver-prime-wheel` feature (and any future
//! link-injected backend). If the feature is on but no crate exports the
//! symbols, the binary fails to LINK — the correct, loud failure for
//! "asked for prime-wheel without linking the runtime that provides it".

use core::task::Waker;
use core::time::Duration;

use crate::time::{Driver, Instant};

unsafe extern "Rust" {
    // monotonic milliseconds on the providing runtime's clock.
    fn proxima_time_external_now_millis() -> u64;
    fn proxima_time_external_schedule_wake(deadline_millis: u64, waker: Waker);
}

/// Singleton — referenced by the build.rs binding when a link-injected
/// driver (e.g. `time-driver-prime-wheel`) is active.
pub static DRIVER: ExternalDriver = ExternalDriver;

/// Zero-sized driver routing every call to the link-time symbols the
/// final binary's runtime crate provides.
pub struct ExternalDriver;

impl Driver for ExternalDriver {
    fn now(&self) -> Instant {
        let millis = unsafe { proxima_time_external_now_millis() };
        Instant::from_monotonic(Duration::from_millis(millis))
    }

    fn schedule_wake(&self, deadline: Instant, waker: Waker) {
        let deadline_millis =
            u64::try_from(deadline.into_monotonic().as_millis()).unwrap_or(u64::MAX);
        unsafe { proxima_time_external_schedule_wake(deadline_millis, waker) }
    }
}

// the crate's own test binary links no runtime crate to provide the symbols,
// yet workspace feature-unification can bind this driver anyway. self-host
// with a wall clock + wake thread so `cargo test` links and the timer tests
// exercise the real external-dispatch path. real consumers keep the loud
// link failure documented above.
#[cfg(all(test, proxima_external_driver))]
mod test_host {
    use core::task::Waker;
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    fn epoch() -> Instant {
        static EPOCH: OnceLock<Instant> = OnceLock::new();
        *EPOCH.get_or_init(Instant::now)
    }

    #[unsafe(no_mangle)]
    extern "Rust" fn proxima_time_external_now_millis() -> u64 {
        u64::try_from(epoch().elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    #[unsafe(no_mangle)]
    extern "Rust" fn proxima_time_external_schedule_wake(deadline_millis: u64, waker: Waker) {
        let now_millis = proxima_time_external_now_millis();
        let delay = Duration::from_millis(deadline_millis.saturating_sub(now_millis));
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            waker.wake();
        });
    }
}
