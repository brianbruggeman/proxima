use super::error::DpdkError;
use super::ffi::{self, RteMbuf, RteMempool};
use std::ffi::CString;

/// A dpdk packet-buffer pool. Lives for the EAL's lifetime; dpdk owns the
/// backing memory, so this is a non-owning handle (no free on drop — the EAL
/// reclaims pools at cleanup).
pub struct Mempool {
    raw: *mut RteMempool,
}

impl Mempool {
    /// Create a pktmbuf pool of `count` buffers on `socket_id` (use -1 for any).
    ///
    /// # Errors
    /// [`DpdkError::ArgNul`] if `name` has an interior nul, [`DpdkError::PoolCreate`]
    /// if dpdk cannot allocate the pool (out of memory or hugepages).
    pub fn create(name: &str, count: u32, socket_id: i32) -> Result<Self, DpdkError> {
        let cname = CString::new(name).map_err(|_| DpdkError::ArgNul)?;
        let raw = unsafe { ffi::pool_create(cname.as_ptr(), count, socket_id) };
        if raw.is_null() {
            return Err(DpdkError::PoolCreate);
        }
        // boot guard: prove the pure-Rust mbuf/mempool ABI against this real pool
        // before any traffic flows, so an offset mistake fails here, not later.
        unsafe { ffi::abi_self_check(raw)? };
        Ok(Self { raw })
    }

    /// Allocate an empty packet buffer from the pool, or `None` if exhausted.
    #[must_use]
    pub fn alloc(&self) -> *mut RteMbuf {
        unsafe { ffi::mbuf_alloc(self.raw) }
    }
}

/// Number of dpdk ethernet ports the EAL has probed (vdevs + bound PCI).
#[must_use]
pub fn port_count() -> u16 {
    ffi::port_count()
}

/// A started dpdk ethernet port with one RX and one TX queue.
pub struct Port {
    id: u16,
}

impl Port {
    /// Configure 1 RX + 1 TX queue on `port`, start it, enable promiscuous mode.
    ///
    /// # Errors
    /// [`DpdkError::NoPorts`] / [`DpdkError::PortOutOfRange`] if `port` is not
    /// present, or [`DpdkError::PortInit`] with dpdk's code if bring-up fails.
    pub fn init(port: u16, pool: &Mempool) -> Result<Self, DpdkError> {
        let available = port_count();
        if available == 0 {
            return Err(DpdkError::NoPorts);
        }
        if port >= available {
            return Err(DpdkError::PortOutOfRange {
                requested: port,
                available,
            });
        }
        let code = unsafe { ffi::port_init(port, pool.raw) };
        if code < 0 {
            return Err(DpdkError::PortInit(code));
        }
        Ok(Self { id: port })
    }

    #[must_use]
    pub fn id(&self) -> u16 {
        self.id
    }

    /// The port's MAC address.
    ///
    /// # Errors
    /// [`DpdkError::MacAddr`] with dpdk's code if the query fails.
    pub fn mac(&self) -> Result<[u8; 6], DpdkError> {
        let mut out = [0u8; 6];
        let code = unsafe { ffi::macaddr(self.id, out.as_mut_ptr()) };
        if code < 0 {
            return Err(DpdkError::MacAddr(code));
        }
        Ok(out)
    }

    /// Poll up to `bufs.len()` received frames into `bufs`; returns how many.
    /// The caller owns the returned mbufs (must `tx` or `free` them).
    pub fn rx_burst(&self, bufs: &mut [*mut RteMbuf]) -> u16 {
        let max = u16::try_from(bufs.len()).unwrap_or(u16::MAX);
        unsafe { ffi::rx_burst(self.id, bufs.as_mut_ptr(), max) }
    }

    /// Transmit `bufs`; returns how many were accepted. dpdk takes ownership of
    /// the accepted mbufs; the caller must `free` any tail that was not sent.
    pub fn tx_burst(&self, bufs: &mut [*mut RteMbuf]) -> u16 {
        let nb = u16::try_from(bufs.len()).unwrap_or(u16::MAX);
        unsafe { ffi::tx_burst(self.id, bufs.as_mut_ptr(), nb) }
    }
}

/// Swap ethernet src/dst in place — a minimal L2 echo (no-op on runt frames).
///
/// # Safety
/// `mbuf` must be a live mbuf the caller owns.
pub unsafe fn eth_swap(mbuf: *mut RteMbuf) {
    let frame = unsafe { frame_bytes_mut(mbuf) };
    if frame.len() < 12 {
        return;
    }
    let mut destination = [0u8; 6];
    destination.copy_from_slice(&frame[0..6]);
    frame.copy_within(6..12, 0);
    frame[6..12].copy_from_slice(&destination);
}

/// Length of the frame's first segment, in bytes.
///
/// # Safety
/// `mbuf` must be a live mbuf the caller owns.
#[must_use]
pub unsafe fn frame_len(mbuf: *mut RteMbuf) -> u16 {
    unsafe { ffi::mbuf_len(mbuf) }
}

/// Borrow the frame bytes for the lifetime of `mbuf`'s ownership.
///
/// # Safety
/// `mbuf` must be a live mbuf the caller owns and not freed for `'a`.
#[must_use]
pub unsafe fn frame_bytes<'a>(mbuf: *mut RteMbuf) -> &'a [u8] {
    let len = usize::from(unsafe { ffi::mbuf_len(mbuf) });
    let data = unsafe { ffi::mbuf_data(mbuf) };
    unsafe { core::slice::from_raw_parts(data, len) }
}

/// Mutably borrow the frame bytes for in-place rewrites (echo, ARP/ICMP reply).
///
/// # Safety
/// `mbuf` must be a live mbuf the caller owns and not freed for `'a`.
#[must_use]
pub unsafe fn frame_bytes_mut<'a>(mbuf: *mut RteMbuf) -> &'a mut [u8] {
    let len = usize::from(unsafe { ffi::mbuf_len(mbuf) });
    let data = unsafe { ffi::mbuf_data(mbuf) };
    unsafe { core::slice::from_raw_parts_mut(data, len) }
}

/// Grow `mbuf` by `len` bytes and borrow the appended region to fill.
/// Returns `None` if the buffer has no room.
///
/// # Safety
/// `mbuf` must be a live, caller-owned mbuf and not freed for `'a`.
#[must_use]
pub unsafe fn frame_append<'a>(mbuf: *mut RteMbuf, len: u16) -> Option<&'a mut [u8]> {
    let data = unsafe { ffi::mbuf_append(mbuf, len) };
    if data.is_null() {
        return None;
    }
    Some(unsafe { core::slice::from_raw_parts_mut(data, usize::from(len)) })
}

/// Return a buffer to its pool.
///
/// # Safety
/// `mbuf` must be a live mbuf the caller owns and must not be used afterwards.
pub unsafe fn free(mbuf: *mut RteMbuf) {
    unsafe { ffi::mbuf_free(mbuf) }
}
