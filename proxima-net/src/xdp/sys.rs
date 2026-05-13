//! Thin typed wrappers over the raw AF_XDP syscalls. Each function owns
//! exactly one syscall and turns a failed return into a typed [`XdpError`]
//! carrying the kernel's errno, so [`super::umem`] and [`super::xsk`] never
//! touch `libc` directly.

use super::error::XdpError;
use super::uapi::{self, sockaddr_xdp, xdp_mmap_offsets, xdp_umem_reg};
use std::ffi::CString;
use std::io;
use std::mem::size_of;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::ptr;

fn last_errno() -> i32 {
    io::Error::last_os_error().raw_os_error().unwrap_or(-1)
}

/// Open an `AF_XDP` raw socket.
///
/// # Errors
/// [`XdpError::Socket`] with the kernel's errno if `socket(2)` fails.
pub fn xsk_socket() -> Result<OwnedFd, XdpError> {
    // SAFETY: socket(2) with plain integer arguments, no pointers involved.
    let raw = unsafe {
        libc::socket(
            i32::from(uapi::AF_XDP),
            libc::SOCK_RAW | libc::SOCK_CLOEXEC,
            0,
        )
    };
    if raw < 0 {
        return Err(XdpError::Socket(last_errno()));
    }
    // SAFETY: raw is a valid, just-created fd this call uniquely owns.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Register the UMEM region on `fd` via `setsockopt(XDP_UMEM_REG)`.
///
/// # Errors
/// [`XdpError::UmemReg`] with the kernel's errno if the option is rejected.
pub fn umem_reg(fd: RawFd, reg: &xdp_umem_reg) -> Result<(), XdpError> {
    // SAFETY: `reg` is a valid, live reference for the duration of the call;
    // setsockopt only reads `size_of::<xdp_umem_reg>()` bytes from it.
    let result = unsafe {
        libc::setsockopt(
            fd,
            uapi::SOL_XDP,
            uapi::XDP_UMEM_REG,
            ptr::from_ref(reg).cast::<libc::c_void>(),
            size_of::<xdp_umem_reg>() as libc::socklen_t,
        )
    };
    if result < 0 {
        return Err(XdpError::UmemReg(last_errno()));
    }
    Ok(())
}

/// Set one ring's entry count via `setsockopt`. `optname` is one of the
/// `XDP_{RX,TX}_RING` / `XDP_UMEM_{FILL,COMPLETION}_RING` option names;
/// `ring` names the ring in error reports.
///
/// # Errors
/// [`XdpError::RingSetup`] with the kernel's errno if the option is rejected.
pub fn set_ring_size(
    fd: RawFd,
    optname: i32,
    size: u32,
    ring: &'static str,
) -> Result<(), XdpError> {
    // SAFETY: `size` is a plain stack `u32`; setsockopt reads exactly
    // `size_of::<u32>()` bytes from it.
    let result = unsafe {
        libc::setsockopt(
            fd,
            uapi::SOL_XDP,
            optname,
            ptr::from_ref(&size).cast::<libc::c_void>(),
            size_of::<u32>() as libc::socklen_t,
        )
    };
    if result < 0 {
        return Err(XdpError::RingSetup {
            ring,
            errno: last_errno(),
        });
    }
    Ok(())
}

/// Read the ring layout via `getsockopt(XDP_MMAP_OFFSETS)`.
///
/// # Errors
/// [`XdpError::MmapOffsets`] with the kernel's errno if the option is rejected.
pub fn mmap_offsets(fd: RawFd) -> Result<xdp_mmap_offsets, XdpError> {
    let mut offsets = xdp_mmap_offsets::default();
    let mut len = size_of::<xdp_mmap_offsets>() as libc::socklen_t;
    // SAFETY: `offsets` is a valid, appropriately sized out-param and `len`
    // bounds how much getsockopt may write into it.
    let result = unsafe {
        libc::getsockopt(
            fd,
            uapi::SOL_XDP,
            uapi::XDP_MMAP_OFFSETS,
            ptr::from_mut(&mut offsets).cast::<libc::c_void>(),
            &raw mut len,
        )
    };
    if result < 0 {
        return Err(XdpError::MmapOffsets(last_errno()));
    }
    Ok(offsets)
}

/// `mmap` one ring's region at page offset `pgoff` for `len` bytes.
///
/// # Errors
/// [`XdpError::RingMmap`] with the kernel's errno if the mapping fails.
pub fn mmap_ring(
    fd: RawFd,
    pgoff: u64,
    len: usize,
    ring: &'static str,
) -> Result<*mut u8, XdpError> {
    // SAFETY: fixed-size mapping of a kernel-owned ring; MAP_SHARED so writes
    // to the producer/consumer counters reach the kernel; addr=null lets the
    // kernel place it.
    let mapped = unsafe {
        libc::mmap(
            ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED | libc::MAP_POPULATE,
            fd,
            pgoff as libc::off_t,
        )
    };
    if mapped == libc::MAP_FAILED {
        return Err(XdpError::RingMmap {
            ring,
            errno: last_errno(),
        });
    }
    Ok(mapped.cast::<u8>())
}

/// `bind(2)` the socket to `ifindex`/`queue_id` with the given `sxdp_flags`.
///
/// # Errors
/// [`XdpError::Bind`] with the kernel's errno if the bind is rejected.
pub fn bind_xdp(fd: RawFd, ifindex: u32, queue_id: u32, flags: u16) -> Result<(), XdpError> {
    let addr = sockaddr_xdp {
        sxdp_family: uapi::AF_XDP,
        sxdp_flags: flags,
        sxdp_ifindex: ifindex,
        sxdp_queue_id: queue_id,
        sxdp_shared_umem_fd: 0,
    };
    // SAFETY: `addr` is a valid, live `sockaddr_xdp`; bind reads exactly
    // `size_of::<sockaddr_xdp>()` bytes, matching the kernel's expected ABI
    // for `AF_XDP` even though the type differs from `libc::sockaddr`.
    let result = unsafe {
        libc::bind(
            fd,
            ptr::from_ref(&addr).cast::<libc::sockaddr>(),
            size_of::<sockaddr_xdp>() as libc::socklen_t,
        )
    };
    if result < 0 {
        return Err(XdpError::Bind {
            interface: ifindex.to_string(),
            queue: queue_id,
            errno: last_errno(),
        });
    }
    Ok(())
}

/// Resolve an interface name to its kernel ifindex.
///
/// # Errors
/// [`XdpError::ArgNul`] if `name` has an interior nul, or
/// [`XdpError::InterfaceNotFound`] if the kernel has no such interface.
pub fn if_nametoindex(name: &str) -> Result<u32, XdpError> {
    let cname = CString::new(name).map_err(|_| XdpError::ArgNul)?;
    // SAFETY: `cname` is a valid, nul-terminated C string alive for the call.
    let index = unsafe { libc::if_nametoindex(cname.as_ptr()) };
    if index == 0 {
        return Err(XdpError::InterfaceNotFound {
            interface: name.to_string(),
            errno: last_errno(),
        });
    }
    Ok(index)
}
