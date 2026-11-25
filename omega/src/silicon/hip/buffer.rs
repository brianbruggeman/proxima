use std::sync::Arc;

use crate::ffi::hip::{DevicePointer, HipApi, KernelSlot};
use crate::kernel_params::KernelArgument;

use super::binding::release_on_device;
use super::error::HipDriverError;

/// A device allocation with the runtime handle that must free it.
pub struct HipBuffer {
    pub(super) api: Arc<HipApi>,
    pub(super) pointer: DevicePointer,
    pub(super) bytes: usize,
    pub(super) ordinal: i32,
}

impl core::fmt::Debug for HipBuffer {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("HipBuffer")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl HipBuffer {
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// A borrowed window of `len` bytes starting `offset` bytes into this
    /// allocation. The view cannot outlive the buffer, and the window is
    /// checked against the allocation here, so no later copy has to trust it.
    ///
    /// # Errors
    /// [`HipDriverError::ViewOutOfRange`] when `offset + len` passes the allocation.
    pub fn view(&self, offset: usize, len: usize) -> Result<HipView<'_>, HipDriverError> {
        window_end(self.bytes, offset, len).ok_or(HipDriverError::ViewOutOfRange {
            offset,
            len,
            bytes: self.bytes,
        })?;
        Ok(HipView {
            pointer: offset_pointer(self.pointer, offset),
            len,
            owner: core::marker::PhantomData,
        })
    }
}

/// The end offset of a `len`-byte window at `offset` inside `total` bytes,
/// or `None` when it overflows or passes the end.
pub(super) fn window_end(total: usize, offset: usize, len: usize) -> Option<usize> {
    offset.checked_add(len).filter(|end| *end <= total)
}

/// `base` advanced `offset` bytes. The result is only dereferenced by the
/// device, and only after [`window_end`] proved it stays inside the allocation.
pub(super) fn offset_pointer(base: DevicePointer, offset: usize) -> DevicePointer {
    base.offset(offset)
}

/// A bounds-checked window into a [`HipBuffer`], borrowed from it.
#[derive(Debug, Clone, Copy)]
pub struct HipView<'buffer> {
    pub(super) pointer: DevicePointer,
    pub(super) len: usize,
    pub(super) owner: core::marker::PhantomData<&'buffer HipBuffer>,
}

// SAFETY: the slot is the address of the `pointer` field, which holds the device pointer for as long as `self` lives.
unsafe impl KernelArgument for HipBuffer {
    fn slot(&self) -> KernelSlot {
        self.pointer.slot()
    }
}

// SAFETY: the slot is the address of the view's own `pointer` field, valid while the view is borrowed.
unsafe impl KernelArgument for HipView<'_> {
    fn slot(&self) -> KernelSlot {
        self.pointer.slot()
    }
}

impl HipView<'_> {
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            release_on_device(&self.api, self.ordinal, || {
                // SAFETY: the pointer came from `hipMalloc` on this ordinal's device, bound
                // first because the dropping thread may not have it current, and is freed once.
                unsafe { self.api.free(self.pointer) };
            });
        }
    }
}

// SAFETY: a HIP allocation is process-wide runtime state addressed by pointer; the host never
// dereferences it, so moving or sharing the owning handle across threads cannot alias host memory,
// and the device serializes on its queue. HIP's current device is per host thread, so every
// `HipDevice`/`HipRuntime` entry calls `bind` first, which makes the device it was opened for the
// calling thread's current device, so a thread other than the opener still acts on that device.
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}
