use std::sync::Arc;

use crate::ffi::hip::KernelSlot;
use crate::kernel_params::KernelArgument;

use super::buffer::{HipBuffer, HipView};
use super::error::HipDriverError;

/// The element window `range` names inside a buffer of `total` elements of `size` bytes each, as
/// `(byte offset, byte length)`. A range past the end is refused, never clamped: a kernel gets
/// only the pointer, so a clamped start would hand it an end-of-allocation pointer with no
/// length. The refusal rides the window and surfaces when the window is pushed to a launch.
pub(super) fn element_window(
    total: usize,
    range: &core::ops::Range<usize>,
    size: usize,
) -> Result<(usize, usize), WindowRefusal> {
    if range.start <= range.end && range.end <= total {
        return Ok((range.start * size, (range.end - range.start) * size));
    }
    Err(WindowRefusal {
        offset: range.start.saturating_mul(size),
        len: range.end.saturating_sub(range.start).saturating_mul(size),
        bytes: total.saturating_mul(size),
    })
}

/// An out-of-range window request, kept on the window until a launch can return it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WindowRefusal {
    pub(super) offset: usize,
    pub(super) len: usize,
    pub(super) bytes: usize,
}

impl WindowRefusal {
    pub(super) fn error(self) -> HipDriverError {
        HipDriverError::ViewOutOfRange {
            offset: self.offset,
            len: self.len,
            bytes: self.bytes,
        }
    }
}

/// A HIP device allocation of `Element`s: the untyped [`HipBuffer`] plus the element count the
/// shared executor's typed buffers carry.
pub struct HipSlice<Element> {
    pub(super) buffer: Arc<HipBuffer>,
    pub(super) len: usize,
    pub(super) element: core::marker::PhantomData<fn() -> Element>,
}

impl<Element> Clone for HipSlice<Element> {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            len: self.len,
            element: core::marker::PhantomData,
        }
    }
}

impl<Element> core::fmt::Debug for HipSlice<Element> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("HipSlice")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// A shared window of `Element`s onto a [`HipSlice`].
pub struct HipWindow<'buffer, Element> {
    pub(super) view: HipView<'buffer>,
    pub(super) refusal: Option<WindowRefusal>,
    pub(super) element: core::marker::PhantomData<fn() -> Element>,
}

/// A writable window of `Element`s onto a [`HipSlice`], exclusive for its lifetime.
pub struct HipWindowMut<'buffer, Element> {
    pub(super) view: HipView<'buffer>,
    pub(super) refusal: Option<WindowRefusal>,
    pub(super) exclusive: core::marker::PhantomData<&'buffer mut HipBuffer>,
    pub(super) element: core::marker::PhantomData<fn() -> Element>,
}

// SAFETY: delegates to the window's own view, whose slot lives inside the window.
unsafe impl<Element> KernelArgument for HipWindow<'_, Element> {
    fn slot(&self) -> KernelSlot {
        self.view.slot()
    }
}

// SAFETY: delegates to the window's own view, whose slot lives inside the window.
unsafe impl<Element> KernelArgument for HipWindowMut<'_, Element> {
    fn slot(&self) -> KernelSlot {
        self.view.slot()
    }
}
