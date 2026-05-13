//! The UMEM frame region: a page-aligned anonymous `mmap` shared with the
//! kernel that backs every AF_XDP descriptor. It is a fixed-size pool of
//! equal-sized frames; [`Umem::alloc_frame`]/[`Umem::free_frame`] hand frame
//! addresses (UMEM-relative offsets) out to the fill/tx rings and take them
//! back from the rx/completion rings.

use super::error::XdpError;
use std::io;
use std::ptr::NonNull;

/// An anonymous mmap of `frame_count * frame_size` bytes plus a free-list of
/// frame addresses. Frees the mapping on drop.
pub struct Umem {
    base: NonNull<u8>,
    len: usize,
    frame_size: u32,
    free_frames: Vec<u64>,
}

// SAFETY: `base` is an exclusively-owned mapping; the kernel only ever
// touches it through descriptors this Umem handed out, and the AF_XDP model
// requires a single polling thread per socket to drive the rings that hand
// those descriptors out.
unsafe impl Send for Umem {}

impl Umem {
    /// Reserve `frame_count` frames of `frame_size` bytes as one anonymous
    /// mapping, seeding the free list with every frame's address.
    ///
    /// # Errors
    /// [`XdpError::UmemReg`] if the anonymous mapping fails.
    pub fn new(frame_count: u32, frame_size: u32) -> Result<Self, XdpError> {
        let len = frame_count as usize * frame_size as usize;
        // SAFETY: anonymous, fixed-size, non-file-backed mapping; addr=null
        // lets the kernel choose the address, fd=-1/offset=0 are required for
        // MAP_ANONYMOUS.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            let errno = io::Error::last_os_error().raw_os_error().unwrap_or(-1);
            return Err(XdpError::UmemReg(errno));
        }
        let Some(base) = NonNull::new(mapped.cast::<u8>()) else {
            return Err(XdpError::UmemReg(-1));
        };
        let free_frames = (0..u64::from(frame_count))
            .map(|index| index * u64::from(frame_size))
            .collect();
        Ok(Self {
            base,
            len,
            frame_size,
            free_frames,
        })
    }

    /// Take one free frame's UMEM-relative address, or `None` if exhausted.
    pub fn alloc_frame(&mut self) -> Option<u64> {
        self.free_frames.pop()
    }

    /// Return a frame to the free list, normalized to its chunk base.
    ///
    /// RX descriptors carry the kernel's `XDP_PACKET_HEADROOM` offset within
    /// the chunk, so `addr` is often `chunk_base + headroom`; the FILL ring
    /// (aligned mode) only accepts chunk-aligned addresses, so snap back to the
    /// chunk base before recycling.
    pub fn free_frame(&mut self, addr: u64) {
        let chunk_base = addr - (addr % u64::from(self.frame_size));
        self.free_frames.push(chunk_base);
    }

    /// The frame region base address, as registered with `XDP_UMEM_REG`.
    #[must_use]
    pub fn base_addr(&self) -> u64 {
        self.base.as_ptr() as u64
    }

    /// A live pointer to the frame at UMEM-relative `addr`.
    ///
    /// # Panics
    /// Never in practice for `addr` values this `Umem` handed out via
    /// [`Umem::alloc_frame`]; callers must not pass an out-of-range address.
    #[must_use]
    pub fn frame_ptr(&self, addr: u64) -> *mut u8 {
        // SAFETY: `addr` is a frame offset this Umem produced in `new` (all
        // multiples of `frame_size` below `len`), so the offset stays within
        // the mapped region.
        unsafe { self.base.as_ptr().add(addr as usize) }
    }

    /// Total mapped region length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the region has zero frames (always false for a constructed
    /// `Umem`; present for the `len`/`is_empty` clippy convention).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The configured per-frame chunk size.
    #[must_use]
    pub fn frame_size(&self) -> u32 {
        self.frame_size
    }
}

impl Drop for Umem {
    fn drop(&mut self) {
        // SAFETY: `base`/`len` are exactly what `new` mapped; this is the
        // only owner and drop runs at most once.
        unsafe {
            libc::munmap(self.base.as_ptr().cast::<libc::c_void>(), self.len);
        }
    }
}
