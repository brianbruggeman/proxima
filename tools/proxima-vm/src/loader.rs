//! Host-side guest memory: maps the [`crate::elf::Segment`]s
//! [`crate::elf::parse_elf`] accepted into a hypervisor guest address
//! space, each at its own `virtual_address()`, with exactly its own
//! `is_readable()` / `is_writable()` / `is_executable()` permissions —
//! honoring W^X instead of the scratch guest's RWX-everything baseline at
//! `backend_macos.c:87`.
//!
//! Tier-2 std driver leaf: [`GuestMemory`] owns the hypervisor VM handle
//! (and, on the KVM lane, the `/dev/kvm` and per-VM file descriptors) for
//! its lifetime; `Drop` unmaps every segment and tears the VM down.
//!
//! # What this module does not do
//!
//! It does not create a vCPU, set `PC`/`SP`, or run anything — that is a
//! later driver leaf (the dispatch/run component the ROADMAP's M1 section
//! names next). It cannot be exercised against a real hypervisor from an
//! in-process unit test either: `hv_vm_create` returns `HV_DENIED` for a
//! process without the `com.apple.security.hypervisor` entitlement, and
//! applying that entitlement is the same post-link `codesign` step
//! `tests/boot.rs`'s `SignedGuest` already works around for the scratch
//! guest — by signing a *subprocess binary*, not the test harness itself.
//! The real end-to-end mapping behavior is exercised by the `proxima-vm
//! run <path>` CLI subcommand's own signed-subprocess test, once that
//! lands.

#![cfg(feature = "std")]

use arrayvec::ArrayVec;
use proxima_core::ProximaError;

use crate::elf::Segment;

/// One accepted segment's host-side mapping: the guest address it landed
/// at and the size actually reserved (page-rounded up from the segment's
/// declared `memory_size()`), kept only so [`GuestMemory`]'s `Drop` can
/// unwind exactly what was mapped.
#[derive(Debug)]
struct MappedSegment {
    guest_address: u64,
    host_address: *mut core::ffi::c_void,
    mapped_size: usize,
}

/// The host-side mapping of a guest's address space.
///
/// `MAX_SEGMENTS` mirrors [`crate::elf::parse_elf`]'s own caller-chosen
/// capacity — the same const-generic-cap idiom already proven in this
/// crate, not a hidden magic number (`slot-0/AGENTS.md` principle 12).
///
/// Wraps `hv_vm_map`/`hv_vm_create` (macOS) or
/// `KVM_SET_USER_MEMORY_REGION`/`KVM_CREATE_VM` (Linux) in
/// `backend_macos.c`/`backend_linux.c`, extending the same driver-leaf
/// pattern [`crate::ScratchVm`] already proves for a single code blob to N
/// permissioned segments.
#[derive(Debug)]
pub struct GuestMemory<const MAX_SEGMENTS: usize> {
    mapped: ArrayVec<MappedSegment, MAX_SEGMENTS>,
    platform: platform::Handle,
}

impl<const MAX_SEGMENTS: usize> GuestMemory<MAX_SEGMENTS> {
    /// Map every `segment` into a freshly created guest address space,
    /// each at its own `virtual_address()`, sized to `memory_size()`
    /// (page-rounded up) and permissioned exactly by `segment`'s own
    /// `is_readable()` / `is_writable()` / `is_executable()`.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Config`] when `segments.len()` exceeds
    /// `MAX_SEGMENTS`, or [`ProximaError::Upstream`] naming the failing
    /// hypervisor call.
    #[must_use = "an unused GuestMemory immediately unmaps everything it mapped"]
    pub fn map(segments: &[Segment<'_>]) -> Result<Self, ProximaError> {
        if segments.len() > MAX_SEGMENTS {
            return Err(ProximaError::Config(format!(
                "guest has {} segments, capacity is {MAX_SEGMENTS}",
                segments.len()
            )));
        }
        let (mapped, platform) = platform::map(segments)?;
        Ok(Self { mapped, platform })
    }

    /// Guest-physical addresses of every mapped segment, in mapping order
    /// — exposed so the next driver leaf (vCPU creation) can cross-check
    /// what actually landed in guest memory before pointing `PC` at it.
    pub fn guest_addresses(&self) -> impl Iterator<Item = u64> + '_ {
        self.mapped.iter().map(|segment| segment.guest_address)
    }

    /// Number of segments currently mapped.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.mapped.len()
    }
}

impl<const MAX_SEGMENTS: usize> Drop for GuestMemory<MAX_SEGMENTS> {
    fn drop(&mut self) {
        platform::unmap(&self.mapped, &self.platform);
    }
}

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
pub(crate) use platform::RawSegment;

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
mod platform {
    use std::ffi::CStr;
    use std::os::raw::c_char;

    use arrayvec::ArrayVec;
    use proxima_core::ProximaError;

    use super::MappedSegment;
    use crate::elf::Segment;

    const ERROR_CAPACITY: usize = 512;

    /// FFI mirror of `proxima_vm_segment_t` (`ffi_segment.h`). Field order
    /// and width must match the C struct exactly.
    ///
    /// `pub(crate)` beyond this module: `dispatch.rs`'s `run_dispatch_loop`
    /// reuses this same marshaling to hand real ELF segments AND a
    /// synthetic stack region (see [`RawSegment::stack`]) to the C-side
    /// per-segment mapper, instead of hand-rolling a second FFI struct.
    #[repr(C)]
    pub(crate) struct RawSegment {
        guest_address: u64,
        data: *const u8,
        data_length: usize,
        memory_size: u64,
        readable: u8,
        writable: u8,
        executable: u8,
    }

    impl RawSegment {
        pub(crate) fn from_segment(segment: &Segment<'_>) -> Self {
            Self {
                guest_address: segment.virtual_address(),
                data: segment.data().as_ptr(),
                data_length: segment.data().len(),
                memory_size: segment.memory_size(),
                readable: u8::from(segment.is_readable()),
                writable: u8::from(segment.is_writable()),
                executable: u8::from(segment.is_executable()),
            }
        }

        /// A synthetic, data-free region: the write-only, non-executable
        /// stack reservation `guests/lambda/link.ld` carves out beyond the
        /// guest ELF's own `PT_LOAD` segments (`__stack_top` sits at the end
        /// of the linker script's 64 MiB `RAM` region; nothing between the
        /// last segment and there is backed by file content). `data` is a
        /// dangling-but-non-null empty-slice pointer — never dereferenced,
        /// since `data_length` is `0` and the C-side mapper guards the copy
        /// on that length.
        pub(crate) fn stack(guest_address: u64, memory_size: u64) -> Self {
            Self {
                guest_address,
                data: (&[] as &[u8]).as_ptr(),
                data_length: 0,
                memory_size,
                readable: 1,
                writable: 1,
                executable: 0,
            }
        }

        /// A raw, non-ELF-backed region at a caller-chosen `guest_address`
        /// offset and permission set — the shape a real kernel boot needs
        /// for the Image blob and the DTB blob (`crate::boot`), neither of
        /// which has an ELF `Segment` to marshal via [`Self::from_segment`].
        /// `data` must outlive the `proxima_vm_run_dispatch_loop` call this
        /// segment is passed into.
        pub(crate) fn raw(
            guest_address: u64,
            data: &[u8],
            memory_size: u64,
            readable: bool,
            writable: bool,
            executable: bool,
        ) -> Self {
            Self {
                guest_address,
                data: data.as_ptr(),
                data_length: data.len(),
                memory_size,
                readable: u8::from(readable),
                writable: u8::from(writable),
                executable: u8::from(executable),
            }
        }
    }

    /// FFI mirror of `proxima_vm_mapped_segment_t` (`ffi_segment.h`).
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawMappedSegment {
        guest_address: u64,
        host_address: *mut core::ffi::c_void,
        mapped_size: usize,
    }

    impl Default for RawMappedSegment {
        fn default() -> Self {
            Self {
                guest_address: 0,
                host_address: core::ptr::null_mut(),
                mapped_size: 0,
            }
        }
    }

    fn read_error(error_buffer: &[c_char]) -> String {
        unsafe { CStr::from_ptr(error_buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    mod ffi {
        use std::os::raw::c_char;

        use super::{RawMappedSegment, RawSegment};

        unsafe extern "C" {
            pub fn proxima_vm_map_guest_memory(
                segments: *const RawSegment,
                segment_count: usize,
                mapped_out: *mut RawMappedSegment,
                error_buffer: *mut c_char,
                error_capacity: usize,
            ) -> i32;

            pub fn proxima_vm_unmap_guest_memory(
                mapped: *const RawMappedSegment,
                mapped_count: usize,
            );
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    mod ffi {
        use std::os::raw::c_char;

        use super::{RawMappedSegment, RawSegment};

        unsafe extern "C" {
            pub fn proxima_vm_map_guest_memory(
                segments: *const RawSegment,
                segment_count: usize,
                mapped_out: *mut RawMappedSegment,
                kvm_fd_out: *mut i32,
                vm_fd_out: *mut i32,
                error_buffer: *mut c_char,
                error_capacity: usize,
            ) -> i32;

            pub fn proxima_vm_unmap_guest_memory(
                mapped: *const RawMappedSegment,
                mapped_count: usize,
                kvm_fd: i32,
                vm_fd: i32,
            );
        }
    }

    /// The hypervisor resources [`GuestMemory`](super::GuestMemory) must
    /// release on `Drop`, beyond the per-segment host mappings themselves.
    /// macOS's `hv_vm_create`/`hv_vm_destroy` pair is process-global and
    /// carries no handle; the KVM lane's VM is a file descriptor pair that
    /// must be closed explicitly.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[derive(Debug)]
    pub struct Handle;

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[derive(Debug)]
    pub struct Handle {
        kvm_fd: i32,
        vm_fd: i32,
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn map<const MAX_SEGMENTS: usize>(
        segments: &[Segment<'_>],
    ) -> Result<(ArrayVec<MappedSegment, MAX_SEGMENTS>, Handle), ProximaError> {
        let raw_segments: ArrayVec<RawSegment, MAX_SEGMENTS> =
            segments.iter().map(RawSegment::from_segment).collect();
        let mut mapped_out = [RawMappedSegment::default(); MAX_SEGMENTS];
        let mut error_buffer = [0_i8; ERROR_CAPACITY];

        let status = unsafe {
            ffi::proxima_vm_map_guest_memory(
                raw_segments.as_ptr(),
                raw_segments.len(),
                mapped_out.as_mut_ptr(),
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if status != 0 {
            return Err(ProximaError::Upstream(read_error(&error_buffer)));
        }

        let mapped = mapped_out[..raw_segments.len()]
            .iter()
            .map(|raw| MappedSegment {
                guest_address: raw.guest_address,
                host_address: raw.host_address,
                mapped_size: raw.mapped_size,
            })
            .collect();
        Ok((mapped, Handle))
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    pub fn map<const MAX_SEGMENTS: usize>(
        segments: &[Segment<'_>],
    ) -> Result<(ArrayVec<MappedSegment, MAX_SEGMENTS>, Handle), ProximaError> {
        let raw_segments: ArrayVec<RawSegment, MAX_SEGMENTS> =
            segments.iter().map(RawSegment::from_segment).collect();
        let mut mapped_out = [RawMappedSegment::default(); MAX_SEGMENTS];
        let mut error_buffer = [0_i8; ERROR_CAPACITY];
        let mut kvm_fd: i32 = -1;
        let mut vm_fd: i32 = -1;

        let status = unsafe {
            ffi::proxima_vm_map_guest_memory(
                raw_segments.as_ptr(),
                raw_segments.len(),
                mapped_out.as_mut_ptr(),
                &raw mut kvm_fd,
                &raw mut vm_fd,
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if status != 0 {
            return Err(ProximaError::Upstream(read_error(&error_buffer)));
        }

        let mapped = mapped_out[..raw_segments.len()]
            .iter()
            .map(|raw| MappedSegment {
                guest_address: raw.guest_address,
                host_address: raw.host_address,
                mapped_size: raw.mapped_size,
            })
            .collect();
        Ok((mapped, Handle { kvm_fd, vm_fd }))
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn unmap<const MAX_SEGMENTS: usize>(
        mapped: &ArrayVec<MappedSegment, MAX_SEGMENTS>,
        _handle: &Handle,
    ) {
        let raw: ArrayVec<RawMappedSegment, MAX_SEGMENTS> = mapped
            .iter()
            .map(|segment| RawMappedSegment {
                guest_address: segment.guest_address,
                host_address: segment.host_address,
                mapped_size: segment.mapped_size,
            })
            .collect();
        unsafe {
            ffi::proxima_vm_unmap_guest_memory(raw.as_ptr(), raw.len());
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    pub fn unmap<const MAX_SEGMENTS: usize>(
        mapped: &ArrayVec<MappedSegment, MAX_SEGMENTS>,
        handle: &Handle,
    ) {
        let raw: ArrayVec<RawMappedSegment, MAX_SEGMENTS> = mapped
            .iter()
            .map(|segment| RawMappedSegment {
                guest_address: segment.guest_address,
                host_address: segment.host_address,
                mapped_size: segment.mapped_size,
            })
            .collect();
        unsafe {
            ffi::proxima_vm_unmap_guest_memory(
                raw.as_ptr(),
                raw.len(),
                handle.kvm_fd,
                handle.vm_fd,
            );
        }
    }
}

#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
mod platform {
    use arrayvec::ArrayVec;
    use proxima_core::ProximaError;

    use super::MappedSegment;
    use crate::elf::Segment;

    #[derive(Debug)]
    pub struct Handle;

    pub fn map<const MAX_SEGMENTS: usize>(
        _segments: &[Segment<'_>],
    ) -> Result<(ArrayVec<MappedSegment, MAX_SEGMENTS>, Handle), ProximaError> {
        Err(ProximaError::Config(
            "guest memory mapping supports linux/x86_64 KVM and macos/aarch64 Hypervisor.framework only"
                .into(),
        ))
    }

    pub fn unmap<const MAX_SEGMENTS: usize>(
        _mapped: &ArrayVec<MappedSegment, MAX_SEGMENTS>,
        _handle: &Handle,
    ) {
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::elf::parse_elf;

    /// A real, valid two-segment ELF (readable+executable `.text`,
    /// readable-only `.rodata`) built with `elf.rs`'s own canonical test
    /// encoder, so `GuestMemory::map`'s capacity check runs against real
    /// `Segment` views rather than a hand-rolled struct literal. The
    /// capacity guard runs before any hypervisor call, so this test needs
    /// no signed subprocess and no real VM — see the module doc.
    #[test]
    fn map_rejects_more_segments_than_capacity() {
        let image = crate::elf::test_support::build_two_segment_elf();
        let (_entry, segments) = parse_elf::<4>(&image).expect("valid ELF parses");
        assert_eq!(segments.len(), 2);

        let error = GuestMemory::<1>::map(&segments).unwrap_err();
        assert_eq!(
            error.to_string(),
            "config: guest has 2 segments, capacity is 1"
        );
    }
}
