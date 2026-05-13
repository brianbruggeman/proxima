use core::time::Duration;

use crate::pipe::capabilities::Clock;
use proxima_core::time::Sleep;

/// Production [`Clock`] backed by `proxima-time`'s link-bound driver.
///
/// `now_nanos` reads the monotonic clock; `delay` hands back proxima-time's
/// concrete `Sleep` future, so a `Retry` built over this stays unboxed and
/// no-alloc. Build the controller's `Deadline` from the same clock so the
/// nanos origins agree.
#[derive(Debug, Clone, Copy, Default)]
pub struct TimeClock;

impl Clock for TimeClock {
    type Delay = Sleep;

    fn now_nanos(&self) -> u64 {
        u64::try_from(proxima_core::time::now().into_monotonic().as_nanos()).unwrap_or(u64::MAX)
    }

    fn delay(&self, dur: Duration) -> Sleep {
        proxima_core::time::sleep(dur)
    }
}
