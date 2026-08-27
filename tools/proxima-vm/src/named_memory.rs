//! M4 — guest memory as a named object
//! (`tools/proxima-vm/ROADMAP.md`'s M4 section): `mmap(MAP_ANON)` has no
//! identity beyond the one virtual-address range one process mapped it
//! into, so it cannot be shared with a second mapper, snapshotted, or
//! forked. This module is the tier-2 std driver leaf over the named
//! backing object each platform substitutes instead — `memfd_create` +
//! `MAP_SHARED`/`MAP_PRIVATE` on the KVM lane, `vm_allocate` +
//! `mach_make_memory_entry_64` on the HVF lane
//! (`backend_linux.c`/`backend_macos.c`'s `proxima_vm_create_named_region`
//! / `proxima_vm_map_named_region` / `proxima_vm_destroy_named_region`).
//!
//! [`GuestMemoryRegion`] owns the backing object and its first
//! ([`GuestMemoryRegion::primary_slice`]) view; [`RegionView`] is a second,
//! independent host-address-space view of the same object
//! ([`GuestMemoryRegion::map_shared_view`],
//! [`GuestMemoryRegion::map_private_view`]) — the shape `crate::dispatch`'s
//! `run_dispatch_loop` now allocates its own guest memory through on both
//! backends, and the shape M7/M8's snapshot and fork milestones need a
//! second, independent mapper of the same guest RAM to exist at all.

#![cfg(feature = "std")]

use proxima_core::ProximaError;

/// FFI mirror of `proxima_vm_named_region_t` (`ffi_segment.h`). Field order
/// and width must match the C struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RawNamedRegion {
    handle: i32,
    primary_address: *mut core::ffi::c_void,
    mapped_size: usize,
}

/// Owns one named backing object plus its primary view. Dropping it tears
/// down the primary view and closes the handle (`proxima_vm_destroy_named_region`);
/// any [`RegionView`] created from it must already have been dropped or the
/// two `Drop` orders race the same backing object — callers are expected to
/// drop views before the region that created them, the same ownership order
/// Rust's own borrow checker would otherwise enforce for a `&self`-borrowing
/// view type.
#[derive(Debug)]
pub struct GuestMemoryRegion {
    raw: RawNamedRegion,
}

/// A second, independent host-address-space view of a [`GuestMemoryRegion`]'s
/// backing object. Dropping it unmaps only this view
/// (`proxima_vm_unmap_named_region_view`) — the backing object and the
/// region's own primary view are untouched.
#[derive(Debug)]
pub struct RegionView {
    address: *mut core::ffi::c_void,
    mapped_size: usize,
}

impl GuestMemoryRegion {
    /// Create a fresh, zero-filled named region of `size` bytes and map its
    /// first view.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call.
    #[must_use = "an unused GuestMemoryRegion immediately destroys the object it created"]
    pub fn create(size: usize) -> Result<Self, ProximaError> {
        let raw = platform::create(size)?;
        Ok(Self { raw })
    }

    /// The primary view's bytes, read-only.
    #[must_use]
    pub fn primary_slice(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(self.raw.primary_address.cast::<u8>(), self.raw.mapped_size)
        }
    }

    /// The primary view's bytes, mutable.
    pub fn primary_slice_mut(&mut self) -> &mut [u8] {
        unsafe {
            core::slice::from_raw_parts_mut(
                self.raw.primary_address.cast::<u8>(),
                self.raw.mapped_size,
            )
        }
    }

    /// Number of bytes the region reserves.
    #[must_use]
    pub fn len(&self) -> usize {
        self.raw.mapped_size
    }

    /// Whether the region reserves zero bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.raw.mapped_size == 0
    }

    /// Map a second, `MAP_SHARED`-equivalent view of this region's backing
    /// object: a write through either view becomes visible through the
    /// other. This is the M4 exit criterion's "two VMs map the same region
    /// and observe each other's writes" case.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call.
    #[must_use = "an unused RegionView immediately unmaps the view it created"]
    pub fn map_shared_view(&self) -> Result<RegionView, ProximaError> {
        platform::map_view(&self.raw, false)
    }

    /// Map a second, `MAP_PRIVATE`-equivalent copy-on-write view of this
    /// region's backing object: a write through this view never becomes
    /// visible through the primary view or any other view. This is the M4
    /// exit criterion's "a `MAP_PRIVATE` child on the KVM lane observes
    /// copy-on-write" case — the HVF lane has no equivalent primitive this
    /// simple and returns [`ProximaError::Upstream`] naming that gap
    /// (`backend_macos.c`'s `proxima_vm_map_named_region` doc).
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call,
    /// or the HVF lane's named unsupported-operation error.
    #[must_use = "an unused RegionView immediately unmaps the view it created"]
    pub fn map_private_view(&self) -> Result<RegionView, ProximaError> {
        platform::map_view(&self.raw, true)
    }
}

impl Drop for GuestMemoryRegion {
    fn drop(&mut self) {
        platform::destroy(&mut self.raw);
    }
}

impl RegionView {
    /// This view's bytes, read-only.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.address.cast::<u8>(), self.mapped_size) }
    }

    /// This view's bytes, mutable.
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.address.cast::<u8>(), self.mapped_size) }
    }
}

impl Drop for RegionView {
    fn drop(&mut self) {
        platform::unmap_view(self.address, self.mapped_size);
    }
}

// SAFETY: a `GuestMemoryRegion`/`RegionView` owns a host memory mapping
// with no thread affinity — the platform calls behind it (`mach_vm_map`,
// `mmap`) are safe to invoke from, and their resulting mapping is safe to
// dereference from, any thread, exactly like `Vec<u8>`'s own heap
// allocation.
unsafe impl Send for GuestMemoryRegion {}
unsafe impl Sync for GuestMemoryRegion {}
unsafe impl Send for RegionView {}
unsafe impl Sync for RegionView {}

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
mod platform {
    use std::ffi::CStr;
    use std::os::raw::c_char;

    use proxima_core::ProximaError;

    use super::RawNamedRegion;

    const ERROR_CAPACITY: usize = 512;

    unsafe extern "C" {
        fn proxima_vm_create_named_region(
            size: usize,
            region_out: *mut RawNamedRegion,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;

        fn proxima_vm_map_named_region(
            region: *const RawNamedRegion,
            want_private_view: i32,
            host_address_out: *mut *mut core::ffi::c_void,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;

        fn proxima_vm_unmap_named_region_view(
            host_address: *mut core::ffi::c_void,
            mapped_size: usize,
        );

        fn proxima_vm_destroy_named_region(region: *mut RawNamedRegion);
    }

    fn read_error(error_buffer: &[c_char]) -> String {
        unsafe { CStr::from_ptr(error_buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    pub(super) fn create(size: usize) -> Result<RawNamedRegion, ProximaError> {
        let mut region_out = RawNamedRegion {
            handle: -1,
            primary_address: core::ptr::null_mut(),
            mapped_size: 0,
        };
        let mut error_buffer = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            proxima_vm_create_named_region(
                size,
                &raw mut region_out,
                error_buffer.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        if status != 0 {
            return Err(ProximaError::Upstream(read_error(&error_buffer)));
        }
        Ok(region_out)
    }

    pub(super) fn map_view(
        region: &RawNamedRegion,
        want_private_view: bool,
    ) -> Result<super::RegionView, ProximaError> {
        let mut host_address_out = core::ptr::null_mut();
        let mut error_buffer = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            proxima_vm_map_named_region(
                region,
                i32::from(want_private_view),
                &raw mut host_address_out,
                error_buffer.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        if status != 0 {
            return Err(ProximaError::Upstream(read_error(&error_buffer)));
        }
        Ok(super::RegionView {
            address: host_address_out,
            mapped_size: region.mapped_size,
        })
    }

    pub(super) fn unmap_view(address: *mut core::ffi::c_void, mapped_size: usize) {
        unsafe { proxima_vm_unmap_named_region_view(address, mapped_size) }
    }

    pub(super) fn destroy(region: &mut RawNamedRegion) {
        unsafe { proxima_vm_destroy_named_region(&raw mut *region) }
    }
}

#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
mod platform {
    use proxima_core::ProximaError;

    use super::RawNamedRegion;

    pub(super) fn create(_size: usize) -> Result<RawNamedRegion, ProximaError> {
        Err(ProximaError::Upstream(
            "named guest memory regions are only implemented for the KVM and HVF lanes".into(),
        ))
    }

    pub(super) fn map_view(
        _region: &RawNamedRegion,
        _want_private_view: bool,
    ) -> Result<super::RegionView, ProximaError> {
        Err(ProximaError::Upstream(
            "named guest memory regions are only implemented for the KVM and HVF lanes".into(),
        ))
    }

    pub(super) fn unmap_view(_address: *mut core::ffi::c_void, _mapped_size: usize) {}

    pub(super) fn destroy(_region: &mut RawNamedRegion) {}
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::GuestMemoryRegion;

    #[cfg(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64")
    ))]
    #[proxima::test]
    async fn a_second_shared_view_observes_writes_made_through_the_primary_view() {
        let mut region =
            GuestMemoryRegion::create(4096).expect("create a named guest memory region");
        region.primary_slice_mut()[..4].copy_from_slice(&[1, 2, 3, 4]);

        let view = region.map_shared_view().expect("map a second shared view");

        assert_eq!(
            &view.as_slice()[..4],
            &[1, 2, 3, 4],
            "a second view of the same named object must observe the first view's writes"
        );
    }

    #[cfg(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64")
    ))]
    #[proxima::test]
    async fn a_write_through_a_second_shared_view_is_visible_through_the_primary_view() {
        let region = GuestMemoryRegion::create(4096).expect("create a named guest memory region");
        let mut view = region.map_shared_view().expect("map a second shared view");

        view.as_slice_mut()[..4].copy_from_slice(&[9, 8, 7, 6]);

        assert_eq!(
            &region.primary_slice()[..4],
            &[9, 8, 7, 6],
            "a write through a second shared view must be visible through the primary view -- \
             proving the two views share the same backing object, not two independent copies"
        );
    }

    #[cfg(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64")
    ))]
    #[proxima::test]
    async fn a_freshly_created_region_is_zero_filled() {
        let region = GuestMemoryRegion::create(4096).expect("create a named guest memory region");

        assert!(
            region.primary_slice().iter().all(|byte| *byte == 0),
            "a freshly created named region must read as all-zero, matching mmap(MAP_ANON)'s contract"
        );
    }

    /// KVM lane only (`ROADMAP.md`'s M4 exit criterion verbatim: "a
    /// `MAP_PRIVATE` child on the KVM lane observes copy-on-write"). This
    /// host is `macos`/`aarch64`, so `backend_linux.c` never compiles into
    /// this crate at all (`build.rs`'s `match (target_os, target_arch)`) --
    /// this test exists, cross-compiles clean under
    /// `--target x86_64-unknown-linux-gnu` (verified separately), and has
    /// never executed on real KVM hardware, the same unexecuted status
    /// `tests/dispatch_hypercall.rs`'s x86_64 lane already carries.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[proxima::test]
    async fn a_private_child_view_on_the_kvm_lane_does_not_see_its_own_writes_reflected_back() {
        let mut region =
            GuestMemoryRegion::create(4096).expect("create a named guest memory region");
        region.primary_slice_mut()[..4].copy_from_slice(&[1, 2, 3, 4]);

        let mut child = region
            .map_private_view()
            .expect("map a private copy-on-write view");
        assert_eq!(
            &child.as_slice()[..4],
            &[1, 2, 3, 4],
            "a private view must read the parent's contents before its own first write"
        );

        child.as_slice_mut()[..4].copy_from_slice(&[9, 8, 7, 6]);

        assert_eq!(
            &region.primary_slice()[..4],
            &[1, 2, 3, 4],
            "a write through a MAP_PRIVATE child view must never become visible through the \
             parent's primary view -- proven by reading the parent's bytes back, not by inference"
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[proxima::test]
    async fn a_private_view_request_on_the_hvf_lane_is_a_named_unsupported_error() {
        let region = GuestMemoryRegion::create(4096).expect("create a named guest memory region");

        let error = region
            .map_private_view()
            .expect_err("the HVF lane has no copy-on-write named-entry primitive this simple");

        let message = error.to_string();
        assert!(
            message.contains("not supported"),
            "the HVF lane's private-view rejection must name the gap, not fail silently: {message}"
        );
    }
}
