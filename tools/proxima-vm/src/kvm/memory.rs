use std::io;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::ptr::NonNull;

use super::error::{KvmError, KvmResult};

const MEMFD_NAME: &core::ffi::CStr = c"proxima-vm-region";

pub fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(size).unwrap_or(4096)
}

#[must_use]
pub fn round_up_to_page(value: usize) -> usize {
    let page = page_size();
    value.div_ceil(page) * page
}

#[derive(Debug)]
pub struct HostMapping {
    address: NonNull<u8>,
    length: usize,
}

// SAFETY: a mapping is plain process address space with no thread affinity.
unsafe impl Send for HostMapping {}
unsafe impl Sync for HostMapping {}

impl HostMapping {
    pub fn anonymous(length: usize, reserve_swap: bool) -> KvmResult<Self> {
        let mut flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
        if !reserve_swap {
            flags |= libc::MAP_NORESERVE;
        }
        Self::map(-1, length, libc::PROT_READ | libc::PROT_WRITE, flags, "mmap anonymous")
    }

    pub fn of_fd(fd: RawFd, length: usize, shared: bool) -> KvmResult<Self> {
        let visibility = if shared { libc::MAP_SHARED } else { libc::MAP_PRIVATE };
        Self::map(fd, length, libc::PROT_READ | libc::PROT_WRITE, visibility, "mmap fd")
    }

    fn map(
        fd: RawFd,
        length: usize,
        protection: i32,
        flags: i32,
        operation: &'static str,
    ) -> KvmResult<Self> {
        // SAFETY: a fresh kernel-chosen address range; the result is checked.
        let address = unsafe { libc::mmap(core::ptr::null_mut(), length, protection, flags, fd, 0) };
        if address == libc::MAP_FAILED {
            return Err(KvmError::last_os(operation));
        }
        NonNull::new(address.cast::<u8>())
            .map(|address| Self { address, length })
            .ok_or_else(|| KvmError::guest("mmap returned a null mapping"))
    }

    #[must_use]
    pub fn as_ptr(&self) -> *mut u8 {
        self.address.as_ptr()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping is `length` bytes of live process memory.
        unsafe { core::slice::from_raw_parts(self.address.as_ptr(), self.length) }
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: exclusive borrow of the owner of a `length`-byte mapping.
        unsafe { core::slice::from_raw_parts_mut(self.address.as_ptr(), self.length) }
    }

    /// Surrenders ownership of the mapping to the caller as raw parts; pair
    /// with [`HostMapping::release_raw`].
    #[must_use]
    pub fn into_raw_parts(self) -> (*mut u8, usize) {
        let parts = (self.address.as_ptr(), self.length);
        core::mem::forget(self);
        parts
    }

    /// # Safety
    ///
    /// `address` and `length` must be exactly the parts of a mapping surrendered by
    /// [`HostMapping::into_raw_parts`] and not yet released.
    pub unsafe fn release_raw(address: *mut u8, length: usize) -> io::Result<()> {
        // SAFETY: forwarded from this function's own contract.
        match unsafe { libc::munmap(address.cast(), length) } {
            0 => Ok(()),
            _ => Err(io::Error::last_os_error()),
        }
    }

    /// First-touches every page so the cost of faulting the mapping in is paid
    /// here, and returns the wall nanoseconds it took.
    pub fn touch_every_page(&mut self) -> u64 {
        let started = std::time::Instant::now();
        let base = self.address.as_ptr();
        for offset in (0..self.length).step_by(page_size()) {
            // SAFETY: `offset` is inside the `length`-byte mapping this value owns.
            unsafe { base.add(offset).write_volatile(0) };
        }
        u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    pub fn release(self) -> io::Result<()> {
        let outcome = self.unmap();
        core::mem::forget(self);
        outcome
    }

    fn unmap(&self) -> io::Result<()> {
        // SAFETY: `address`/`length` describe exactly the mapping this value owns.
        match unsafe { libc::munmap(self.address.as_ptr().cast(), self.length) } {
            0 => Ok(()),
            _ => Err(io::Error::last_os_error()),
        }
    }
}

impl Drop for HostMapping {
    fn drop(&mut self) {
        let _ = self.unmap();
    }
}

#[derive(Debug)]
pub struct NamedRegion {
    descriptor: OwnedFd,
    primary: HostMapping,
}

impl NamedRegion {
    pub fn create(size: usize) -> KvmResult<Self> {
        let length = round_up_to_page(size.max(1));
        let descriptor = create_memfd(length)?;
        let primary = HostMapping::of_fd(descriptor.as_raw_fd(), length, true)?;
        Ok(Self { descriptor, primary })
    }

    pub fn map_view(&self, private: bool) -> KvmResult<HostMapping> {
        HostMapping::of_fd(self.descriptor.as_raw_fd(), self.primary.len(), !private)
    }

    #[must_use]
    pub fn primary(&self) -> &HostMapping {
        &self.primary
    }

    pub fn primary_mut(&mut self) -> &mut HostMapping {
        &mut self.primary
    }

    #[must_use]
    pub fn as_raw_fd(&self) -> RawFd {
        self.descriptor.as_raw_fd()
    }

    /// Surrenders the descriptor and primary view as raw parts, for a caller
    /// that keeps them in a C-shaped record; pair with
    /// [`NamedRegion::from_raw_parts`].
    #[must_use]
    pub fn into_raw_parts(self) -> (RawFd, *mut u8, usize) {
        let Self { descriptor, primary } = self;
        let (address, length) = primary.into_raw_parts();
        (descriptor.into_raw_fd(), address, length)
    }

    /// # Safety
    ///
    /// The parts must be exactly those [`NamedRegion::into_raw_parts`] returned,
    /// and not yet reclaimed.
    #[must_use]
    pub unsafe fn from_raw_parts(descriptor: RawFd, address: *mut u8, length: usize) -> Self {
        Self {
            // SAFETY: forwarded from this function's own contract.
            descriptor: unsafe { OwnedFd::from_raw_fd(descriptor) },
            primary: HostMapping { address: NonNull::new(address).unwrap_or(NonNull::dangling()), length },
        }
    }

    pub fn release(self) -> Vec<io::Error> {
        let Self { descriptor, primary } = self;
        let mut failures = Vec::new();
        failures.extend(primary.release().err());
        failures.extend(close_checked(descriptor).err());
        failures
    }
}

/// A copyable view of guest RAM for the devices that move bytes in and out of
/// it. Devices run one at a time on the vCPU's thread while the vCPU is
/// stopped at an exit, so a view never overlaps another access; the owner of
/// the mapping must outlive every view.
#[derive(Clone, Copy, Debug)]
pub struct GuestRam {
    address: *mut u8,
    length: usize,
}

impl GuestRam {
    #[must_use]
    pub fn of(mapping: &HostMapping) -> Self {
        Self { address: mapping.as_ptr(), length: mapping.len() }
    }

    /// A view of `length` bytes at `address`, for memory a caller owns outside
    /// a [`HostMapping`] (a named region's view, a layered base).
    ///
    /// # Safety
    ///
    /// `address` must stay valid for `length` bytes for as long as the view or
    /// any copy of it is used.
    #[must_use]
    pub unsafe fn from_parts(address: *mut u8, length: usize) -> Self {
        Self { address, length }
    }

    #[must_use]
    pub fn as_ptr(&self) -> *mut u8 {
        self.address
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    fn span(&self, address: u64, length: u64) -> Option<(usize, usize)> {
        let end = address.checked_add(length).filter(|end| *end <= self.length as u64)?;
        Some((usize::try_from(address).ok()?, usize::try_from(end).ok()?))
    }

    #[must_use]
    pub fn read(&self, address: u64, length: u64) -> Option<bytes::Bytes> {
        let (start, end) = self.span(address, length)?;
        // SAFETY: `span` proved `start..end` lies inside the live mapping.
        let region = unsafe { core::slice::from_raw_parts(self.address.add(start), end - start) };
        Some(bytes::Bytes::copy_from_slice(region))
    }

    /// Copies `length` bytes at `offset` of this view to the same offset of
    /// `destination`, which must be a different mapping.
    pub fn copy_to(&self, destination: &Self, offset: u64, length: u64) -> bool {
        let (Some((start, end)), Some(_)) = (self.span(offset, length), destination.span(offset, length)) else {
            return false;
        };
        // SAFETY: both spans were bounds-checked against their own live mappings, and
        // the mappings are distinct by the caller's contract.
        unsafe { core::ptr::copy_nonoverlapping(self.address.add(start), destination.address.add(start), end - start) };
        true
    }

    pub fn write(&self, address: u64, bytes: &[u8]) -> bool {
        let Some((start, _)) = self.span(address, bytes.len() as u64) else {
            return false;
        };
        // SAFETY: `span` proved the destination lies inside the live mapping, and
        // `bytes` is a separate host buffer.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.address.add(start), bytes.len()) };
        true
    }

    pub fn with_mut<R>(&self, operate: impl FnOnce(&mut [u8]) -> R) -> R {
        // SAFETY: see the type's contract: no other access to the mapping is live.
        operate(unsafe { core::slice::from_raw_parts_mut(self.address, self.length) })
    }
}

fn create_memfd(length: usize) -> KvmResult<OwnedFd> {
    // SAFETY: a valid NUL-terminated name; the result is checked.
    let raw = unsafe { libc::memfd_create(MEMFD_NAME.as_ptr(), libc::MFD_CLOEXEC) };
    if raw < 0 {
        return Err(KvmError::last_os("memfd_create"));
    }
    // SAFETY: `raw` is a fresh descriptor nothing else owns.
    let descriptor = unsafe { OwnedFd::from_raw_fd(raw) };
    let truncate_to = libc::off_t::try_from(length)
        .map_err(|_| KvmError::guest("named region length exceeds off_t"))?;
    // SAFETY: `descriptor` is a live memfd.
    if unsafe { libc::ftruncate(descriptor.as_raw_fd(), truncate_to) } < 0 {
        return Err(KvmError::last_os("ftruncate named region"));
    }
    Ok(descriptor)
}

pub fn close_checked(descriptor: OwnedFd) -> io::Result<()> {
    // SAFETY: ownership of the descriptor is surrendered to this single close.
    match unsafe { libc::close(descriptor.into_raw_fd()) } {
        0 => Ok(()),
        _ => Err(io::Error::last_os_error()),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn a_shared_view_sees_writes_through_the_primary_and_a_private_view_diverges() {
        let mut region = NamedRegion::create(8192).expect("memfd region");
        region.primary_mut().bytes_mut()[0] = 7;
        let shared = region.map_view(false).expect("shared view");
        let mut private = region.map_view(true).expect("private view");

        region.primary_mut().bytes_mut()[1] = 9;
        private.bytes_mut()[0] = 0x55;

        assert_eq!((shared.bytes()[0], shared.bytes()[1]), (7, 9));
        assert_eq!(private.bytes()[0], 0x55, "a private write is local");
        assert_eq!(region.primary().bytes()[0], 7, "and invisible to the primary");
    }

    #[test]
    fn a_region_rounds_its_size_up_to_a_page() {
        let region = NamedRegion::create(1).expect("memfd region");

        assert_eq!(region.primary().len(), page_size());
    }
}
