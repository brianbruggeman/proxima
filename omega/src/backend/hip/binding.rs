use core::cell::Cell;

use crate::ffi::hip::HipApi;

thread_local! {
    static BOUND_ORDINAL: Cell<Option<i32>> = const { Cell::new(None) };
}

/// Makes `ordinal` this host thread's current HIP device through `set_device`, skipping the call
/// when the thread already bound it. HIP's current device is per host thread, so every entry that
/// touches the device binds first. A failed bind clears the memo: the thread's device is unknown.
pub(super) fn bind_ordinal<Failure>(
    ordinal: i32,
    set_device: impl FnOnce(i32) -> Result<(), Failure>,
) -> Result<(), Failure> {
    if BOUND_ORDINAL.with(Cell::get) == Some(ordinal) {
        return Ok(());
    }
    let result = set_device(ordinal);
    BOUND_ORDINAL.with(|bound| bound.set(result.is_ok().then_some(ordinal)));
    result
}

/// Runs `release` on `ordinal`'s device. A failed bind skips the release: a free on the wrong
/// device is worse than a leak, and a `Drop` cannot report, so the runtime reclaims it at teardown.
pub(super) fn release_on_device(api: &HipApi, ordinal: i32, release: impl FnOnce()) {
    release_after_bind(ordinal, |bound| api.set_device(bound), release);
}

pub(super) fn release_after_bind<Failure>(
    ordinal: i32,
    set_device: impl FnOnce(i32) -> Result<(), Failure>,
    release: impl FnOnce(),
) {
    if bind_ordinal(ordinal, set_device).is_ok() {
        release();
    }
}
