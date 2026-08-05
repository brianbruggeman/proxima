//! Proves the hardware-clock seam end-to-end: a `Pipe` implemented over a
//! fake memory-mapped page (a `[u64; N]` array read via
//! `core::ptr::read_volatile`) composes with [`ToUnixNanos`] with zero
//! edits to `proxima-clock` — the mmap/DMA/PTP shape a real driver would
//! use, modeled on `src/runtime.rs`'s
//! `foreign_backend_can_be_constructed_and_driven_through_runtime_selection`
//! (a foreign backend binds through the existing seam, not a new one).
//!
//! `std` is fine here: this is the TEST driving the seam, not the crate
//! under test. `proxima-clock`'s own source never allocates or touches
//! `std`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::convert::Infallible;
use core::future::Future;

use proxima_clock::anchor::{AnchorCell, ToUnixNanos};
use proxima_clock::ticks::Ticks;
use proxima_clock::unix_nanos::UnixNanos;
use proxima_primitives::block_on;
use proxima_primitives::pipe::ext::PipeExt;
use proxima_primitives::pipe::primitives::Pipe;

/// A fake memory-mapped counter register: word 0 of a page a real driver
/// would have obtained via `mmap`/DMA. `HardwareTicks` holds a raw pointer
/// into it and reads it with `read_volatile` on every call — the same
/// shape a real ARM generic timer, PTP hardware clock, or NVMe completion
/// timestamp register would use.
struct HardwareTicks {
    register: *const u64,
}

// SAFETY (test-only): `register` points at a `[u64; _]` local the test
// owns for the whole call, never mutated except via the deliberate
// `write_volatile` calls below, from the same thread.
unsafe impl Send for HardwareTicks {}

impl Pipe for HardwareTicks {
    type In = ();
    type Out = Ticks;
    type Err = Infallible;

    fn call(&self, (): ()) -> impl Future<Output = Result<Ticks, Infallible>> {
        // SAFETY: `register` points into a live `[u64; N]` for the
        // lifetime of this test; `read_volatile` matches how a real
        // mmap'd register must be read (never elided/reordered by the
        // optimizer).
        let raw = unsafe { core::ptr::read_volatile(self.register) };
        let ticks = Ticks::from_raw(raw);
        async move { Ok(ticks) }
    }
}

const ARM_GENERIC_TIMER_HZ: u64 = 24_000_000;

#[test]
fn fake_mmap_hardware_counter_drives_the_real_seam_end_to_end() {
    // word 0 = the free-running tick counter; a real driver's page would
    // carry more (control/status registers) that this test never touches.
    let page: [u64; 4] = [0, 0, 0, 0];
    let source = HardwareTicks {
        register: core::ptr::addr_of!(page[0]),
    };

    let anchor = AnchorCell::new(
        Ticks::from_raw(0),
        UnixNanos::from_nanos(1_753_500_000_000_000_000),
    );
    let wall_clock = source.and_then(ToUnixNanos::new(&anchor, ARM_GENERIC_TIMER_HZ));

    let at_anchor = block_on(Pipe::call(&wall_clock, ())).expect("hardware read never fails");
    assert_eq!(at_anchor, UnixNanos::from_nanos(1_753_500_000_000_000_000));

    // SAFETY: same page, still live, still owned by this test, no
    // concurrent access — this simulates the hardware counter advancing.
    unsafe {
        core::ptr::write_volatile(
            core::ptr::addr_of!(page[0]).cast_mut(),
            ARM_GENERIC_TIMER_HZ * 2,
        );
    }

    let two_seconds_later =
        block_on(Pipe::call(&wall_clock, ())).expect("hardware read never fails");
    assert_eq!(
        two_seconds_later,
        UnixNanos::from_nanos(1_753_500_002_000_000_000)
    );
}

#[test]
fn fake_mmap_hardware_counter_at_ptp_100mhz_and_tsc_nominal_1ghz() {
    for (frequency_hz, ticks_elapsed, expected_nanos_elapsed) in [
        (100_000_000u64, 50_000_000u64, 500_000_000u64), // PTP hardware clock, 0.5s
        (1_000_000_000u64, 7u64, 7u64),                  // TSC nominal, 1:1
    ] {
        let page: [u64; 1] = [0];
        let source = HardwareTicks {
            register: core::ptr::addr_of!(page[0]),
        };
        let anchor = AnchorCell::new(Ticks::from_raw(0), UnixNanos::from_nanos(0));
        let wall_clock = source.and_then(ToUnixNanos::new(&anchor, frequency_hz));

        // SAFETY: page is local, live, single-threaded, no aliasing.
        unsafe {
            core::ptr::write_volatile(core::ptr::addr_of!(page[0]).cast_mut(), ticks_elapsed);
        }

        let reading = block_on(Pipe::call(&wall_clock, ())).expect("hardware read never fails");
        assert_eq!(reading, UnixNanos::from_nanos(expected_nanos_elapsed));
    }
}

#[test]
fn malformed_register_value_of_all_ones_does_not_panic_or_overflow() {
    // adversarial input: a torn/garbage register read (e.g. a device reset
    // mid-read) reporting u64::MAX. the conversion must stay total.
    let page: [u64; 1] = [u64::MAX];
    let source = HardwareTicks {
        register: core::ptr::addr_of!(page[0]),
    };
    let anchor = AnchorCell::new(Ticks::from_raw(0), UnixNanos::from_nanos(u64::MAX - 1));
    let wall_clock = source.and_then(ToUnixNanos::new(&anchor, ARM_GENERIC_TIMER_HZ));

    let reading = block_on(Pipe::call(&wall_clock, ())).expect("hardware read never fails");

    assert_eq!(
        reading,
        UnixNanos::from_nanos(u64::MAX),
        "saturates at UnixNanos' representable maximum instead of wrapping or panicking"
    );
}
