//! Proves the architectural claim behind shaping tick sources as pipes:
//! arbitrating between redundant clock sources (a chrony/PTP-style
//! "several sources, prefer the best live one" setup — a GPS PPS signal,
//! a PTP grandmaster, a local oscillator) is
//! `proxima_primitives::pipe::fan_in::FanIn` composing DIRECTLY over
//! `proxima-clock`'s tick sources. No bespoke arbitration machinery in
//! this crate — the claim is falsifiable, so it is tested, not asserted.
//!
//! `FanIn`'s merged sources are `UnpinPipe<In = (), Err = Exhausted>` (see
//! `proxima-primitives/src/pipe/fan_in.rs`): a redundant clock source
//! resolving `Exhausted` means "this particular source will never answer
//! again" (GPS fix permanently lost, PTP grandmaster decommissioned) —
//! semantically real for a REDUNDANT source, unlike a single raw hardware
//! register read (`proxima_clock::coarse::TickCell`'s `Pipe` impl, or a
//! `HardwareTicks` in `hardware_mmap_seam.rs`), which never claims to
//! exhaust and stays `Err = Infallible`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::future::Future;

use proxima_clock::ticks::Ticks;
use proxima_core::markers::DropSafe;
use proxima_primitives::pipe::fan_in::{Exhausted, FanIn, Select};
use proxima_primitives::pipe::primitives::UnpinPipe;

fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
    let mut pinned = core::pin::pin!(future);
    let mut context = core::task::Context::from_waker(core::task::Waker::noop());
    loop {
        if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
            return output;
        }
    }
}

/// A redundant tick source that answers a fixed number of times, then
/// permanently exhausts — modeling a GPS PPS signal whose fix is lost, or
/// a PTP grandmaster taken offline for maintenance.
struct RedundantSource {
    reading: Ticks,
    remaining_answers: core::cell::Cell<u32>,
}

impl UnpinPipe for RedundantSource {
    type In = ();
    type Out = Ticks;
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<Ticks, Exhausted>> + Unpin {
        let remaining = self.remaining_answers.get();
        let outcome = if remaining == 0 {
            Err(Exhausted)
        } else {
            self.remaining_answers.set(remaining - 1);
            Ok(self.reading)
        };
        core::future::ready(outcome)
    }
}

impl DropSafe for RedundantSource {}

#[test]
fn fan_in_arbitrates_across_two_redundant_tick_sources_with_no_bespoke_machinery() {
    let gps_pps = RedundantSource {
        reading: Ticks::from_raw(1_000),
        remaining_answers: core::cell::Cell::new(2),
    };
    let local_oscillator = RedundantSource {
        reading: Ticks::from_raw(2_000),
        remaining_answers: core::cell::Cell::new(2),
    };

    let fan = FanIn::new([gps_pps, local_oscillator], Select::RoundRobin);

    // 4 total answers available (2 + 2); round-robin visits the GPS
    // source first (array order), so the sequence alternates until both
    // exhaust.
    let first = block_on(fan.call(())).expect("gps still live");
    assert_eq!(first, Ticks::from_raw(1_000));

    let second = block_on(fan.call(())).expect("local oscillator still live");
    assert_eq!(second, Ticks::from_raw(2_000));

    let third = block_on(fan.call(())).expect("gps still live");
    assert_eq!(third, Ticks::from_raw(1_000));

    let fourth = block_on(fan.call(())).expect("local oscillator still live");
    assert_eq!(fourth, Ticks::from_raw(2_000));

    assert_eq!(fan.live_count(), 2, "both sources still marked live before their final call");

    let fifth = block_on(fan.call(()));
    assert_eq!(
        fifth,
        Err(Exhausted),
        "both redundant sources answered their fixed count; the merge itself exhausts"
    );
    assert_eq!(fan.live_count(), 0);
}

#[test]
fn fan_in_keeps_arbitrating_when_one_redundant_source_goes_stale_first() {
    // the GPS fix is lost after one reading; the local oscillator keeps
    // answering — proving the merge does not stop just because ONE
    // redundant source exhausts, matching the real chrony-style shape.
    let gps_pps = RedundantSource {
        reading: Ticks::from_raw(1_000),
        remaining_answers: core::cell::Cell::new(1),
    };
    let local_oscillator = RedundantSource {
        reading: Ticks::from_raw(2_000),
        remaining_answers: core::cell::Cell::new(3),
    };

    let fan = FanIn::new([gps_pps, local_oscillator], Select::RoundRobin);

    // GPS answers its one reading, then exhausts on its second turn; the
    // scan falls through to the local oscillator within the SAME call, so
    // the merge never surfaces the GPS's exhaustion to the caller as a
    // gap. exhaustion is discovered when a source is ASKED and declines —
    // the local oscillator's own internal count hitting zero on its last
    // successful answer does not retroactively mark it exhausted until it
    // is asked once more.
    let readings: Vec<Ticks> = (0..4)
        .map(|_| block_on(fan.call(())).expect("at least one source still live"))
        .collect();

    assert_eq!(
        readings,
        vec![
            Ticks::from_raw(1_000),
            Ticks::from_raw(2_000),
            Ticks::from_raw(2_000),
            Ticks::from_raw(2_000),
        ],
        "GPS answers once then exhausts; the merge falls through to the local oscillator"
    );
    assert_eq!(
        fan.live_count(),
        1,
        "local oscillator's 4th answer used its last reading but is not yet KNOWN exhausted \
         until asked again"
    );

    let fifth = block_on(fan.call(()));
    assert_eq!(
        fifth,
        Err(Exhausted),
        "both redundant sources have now been asked past their fixed answer count"
    );
    assert_eq!(fan.live_count(), 0);
}
