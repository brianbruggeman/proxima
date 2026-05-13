//! The AF_XDP socket: the syscall bring-up sequence (socket → UMEM mmap →
//! `XDP_UMEM_REG` → ring-size setup → `XDP_MMAP_OFFSETS` → ring mmap → bind)
//! and the four mmap'd ring views layered over [`ProducerIndex`]/
//! [`ConsumerIndex`]. The FILL and TX rings are producers on our side; RX and
//! COMPLETION are consumers on our side — the kernel drives the other end of
//! each, so every shared counter access is an atomic load/store rather than a
//! plain read/write.

use super::error::XdpError;
use super::ring::{ConsumerIndex, ProducerIndex};
use super::sized;
use super::sys;
use super::uapi::{self, xdp_desc, xdp_ring_offset, xdp_umem_reg};
use super::umem::Umem;
use std::io;
use std::mem::size_of;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};

/// One mmap'd AF_XDP ring: its base pointer and the kernel-reported byte
/// offsets of the shared producer/consumer counters and descriptor array
/// within that mapping. Shared plumbing for all four ring flavors below.
struct RingMemory {
    base: *mut u8,
    len: usize,
    offset: xdp_ring_offset,
}

impl RingMemory {
    fn map(
        fd: RawFd,
        pgoff: u64,
        size: u32,
        offset: xdp_ring_offset,
        entry_size: usize,
        ring: &'static str,
    ) -> Result<Self, XdpError> {
        let len = offset.desc as usize + size as usize * entry_size;
        let base = sys::mmap_ring(fd, pgoff, len, ring)?;
        Ok(Self { base, len, offset })
    }

    fn producer(&self) -> &AtomicU32 {
        // SAFETY: `offset.producer` is the kernel-reported byte offset of the
        // producer counter within this live mapping, aligned to `u32` by the
        // kernel ABI; `AtomicU32` has `u32`'s size and representation.
        unsafe {
            &*self
                .base
                .add(self.offset.producer as usize)
                .cast::<AtomicU32>()
        }
    }

    fn consumer(&self) -> &AtomicU32 {
        // SAFETY: same reasoning as `producer`, at the consumer offset.
        unsafe {
            &*self
                .base
                .add(self.offset.consumer as usize)
                .cast::<AtomicU32>()
        }
    }

    fn desc_ptr<Entry>(&self) -> *mut Entry {
        // SAFETY: `offset.desc` is the kernel-reported start of the
        // descriptor array within this live mapping; `map`'s `len`
        // computation reserved `size * entry_size` bytes there.
        unsafe { self.base.add(self.offset.desc as usize).cast::<Entry>() }
    }

    fn flags(&self) -> &AtomicU32 {
        // SAFETY: `offset.flags` is the kernel-reported byte offset of the
        // ring's flags word within this live mapping, `u32`-aligned by ABI.
        unsafe {
            &*self
                .base
                .add(self.offset.flags as usize)
                .cast::<AtomicU32>()
        }
    }
}

impl Drop for RingMemory {
    fn drop(&mut self) {
        // SAFETY: `base`/`len` are exactly what `map` mapped; this is the
        // only owner and drop runs at most once.
        unsafe {
            libc::munmap(self.base.cast::<libc::c_void>(), self.len);
        }
    }
}

/// The FILL ring: we push empty UMEM frame addresses for the kernel to
/// receive into.
pub struct FillRing {
    memory: RingMemory,
    index: ProducerIndex,
}

// SAFETY: driven from a single polling thread on our side; the kernel side
// of the shared counters is accessed only through atomics.
unsafe impl Send for FillRing {}

impl FillRing {
    fn new(memory: RingMemory, size: u32) -> Result<Self, XdpError> {
        Ok(Self {
            memory,
            index: ProducerIndex::new(size)?,
        })
    }

    /// Publish one free frame address to the kernel. Returns `false` if the
    /// ring is currently full.
    pub fn push(&mut self, frame_addr: u64) -> bool {
        let live_consumer = self.memory.consumer().load(Ordering::Acquire);
        let Some(start) = self.index.reserve(1, live_consumer) else {
            return false;
        };
        let slot = self.index.slot(start);
        // SAFETY: `slot` is `start & mask`, so it is in `[0, size)`, and the
        // descriptor array holds `size` `u64` entries in the live mapping.
        unsafe {
            *self.memory.desc_ptr::<u64>().add(slot) = frame_addr;
        }
        self.memory
            .producer()
            .store(self.index.commit(), Ordering::Release);
        true
    }

    /// Publish a batch of free frame addresses with a single atomic commit,
    /// returning how many were accepted (`min(frames.len(), free)`).
    pub fn push_batch(&mut self, frames: &[u64]) -> usize {
        let want = u32::try_from(frames.len()).unwrap_or(u32::MAX);
        let live_consumer = self.memory.consumer().load(Ordering::Acquire);
        let (start, granted) = self.index.reserve_up_to(want, live_consumer);
        for offset in 0..granted {
            let slot = self.index.slot(start.wrapping_add(offset));
            // SAFETY: `slot` is in `[0, size)` and the descriptor array holds
            // `size` `u64` entries in the live mapping.
            unsafe {
                *self.memory.desc_ptr::<u64>().add(slot) = frames[offset as usize];
            }
        }
        self.memory
            .producer()
            .store(self.index.commit(), Ordering::Release);
        granted as usize
    }

    /// Whether the kernel set `XDP_RING_NEED_WAKEUP` on the FILL ring — it has
    /// run dry and needs a syscall kick after we replenish it.
    #[must_use]
    pub fn needs_wakeup(&self) -> bool {
        self.memory.flags().load(Ordering::Acquire) & uapi::XDP_RING_NEED_WAKEUP != 0
    }
}

/// The COMPLETION ring: the kernel hands back UMEM frame addresses it has
/// finished transmitting.
pub struct CompletionRing {
    memory: RingMemory,
    index: ConsumerIndex,
}

// SAFETY: driven from a single polling thread on our side; the kernel side
// of the shared counters is accessed only through atomics.
unsafe impl Send for CompletionRing {}

impl CompletionRing {
    fn new(memory: RingMemory, size: u32) -> Result<Self, XdpError> {
        Ok(Self {
            memory,
            index: ConsumerIndex::new(size)?,
        })
    }

    /// Pop one completed frame's address, or `None` if nothing is ready.
    pub fn pop(&mut self) -> Option<u64> {
        let live_producer = self.memory.producer().load(Ordering::Acquire);
        let (start, ready) = self.index.peek(1, live_producer);
        if ready == 0 {
            return None;
        }
        let slot = self.index.slot(start);
        // SAFETY: `slot` is in `[0, size)` and the descriptor array holds
        // `size` `u64` entries in the live mapping.
        let addr = unsafe { *self.memory.desc_ptr::<u64>().add(slot) };
        self.memory
            .consumer()
            .store(self.index.release(1), Ordering::Release);
        Some(addr)
    }

    /// Pop a batch of completed frame addresses into `out` with a single atomic
    /// release, returning how many were written.
    pub fn pop_batch(&mut self, out: &mut [u64]) -> usize {
        let want = u32::try_from(out.len()).unwrap_or(u32::MAX);
        let live_producer = self.memory.producer().load(Ordering::Acquire);
        let (start, ready) = self.index.peek(want, live_producer);
        for offset in 0..ready {
            let slot = self.index.slot(start.wrapping_add(offset));
            // SAFETY: `slot` is in `[0, size)` and the descriptor array holds
            // `size` `u64` entries in the live mapping.
            out[offset as usize] = unsafe { *self.memory.desc_ptr::<u64>().add(slot) };
        }
        if ready > 0 {
            self.memory
                .consumer()
                .store(self.index.release(ready), Ordering::Release);
        }
        ready as usize
    }
}

/// The RX ring: the kernel hands us descriptors for received frames.
pub struct RxRing {
    memory: RingMemory,
    index: ConsumerIndex,
}

// SAFETY: driven from a single polling thread on our side; the kernel side
// of the shared counters is accessed only through atomics.
unsafe impl Send for RxRing {}

impl RxRing {
    fn new(memory: RingMemory, size: u32) -> Result<Self, XdpError> {
        Ok(Self {
            memory,
            index: ConsumerIndex::new(size)?,
        })
    }

    /// Pop one received frame's descriptor, or `None` if nothing is ready.
    pub fn pop(&mut self) -> Option<xdp_desc> {
        let live_producer = self.memory.producer().load(Ordering::Acquire);
        let (start, ready) = self.index.peek(1, live_producer);
        if ready == 0 {
            return None;
        }
        let slot = self.index.slot(start);
        // SAFETY: `slot` is in `[0, size)` and the descriptor array holds
        // `size` `xdp_desc` entries in the live mapping.
        let desc = unsafe { *self.memory.desc_ptr::<xdp_desc>().add(slot) };
        self.memory
            .consumer()
            .store(self.index.release(1), Ordering::Release);
        Some(desc)
    }

    /// Pop a batch of received descriptors into `out` with a single atomic
    /// release, returning how many were written.
    pub fn peek_batch(&mut self, out: &mut [xdp_desc]) -> usize {
        let want = u32::try_from(out.len()).unwrap_or(u32::MAX);
        let live_producer = self.memory.producer().load(Ordering::Acquire);
        let (start, ready) = self.index.peek(want, live_producer);
        for offset in 0..ready {
            let slot = self.index.slot(start.wrapping_add(offset));
            // SAFETY: `slot` is in `[0, size)` and the descriptor array holds
            // `size` `xdp_desc` entries in the live mapping.
            out[offset as usize] = unsafe { *self.memory.desc_ptr::<xdp_desc>().add(slot) };
        }
        if ready > 0 {
            self.memory
                .consumer()
                .store(self.index.release(ready), Ordering::Release);
        }
        ready as usize
    }
}

/// The TX ring: we push descriptors for frames the kernel should send.
pub struct TxRing {
    memory: RingMemory,
    index: ProducerIndex,
}

// SAFETY: driven from a single polling thread on our side; the kernel side
// of the shared counters is accessed only through atomics.
unsafe impl Send for TxRing {}

impl TxRing {
    fn new(memory: RingMemory, size: u32) -> Result<Self, XdpError> {
        Ok(Self {
            memory,
            index: ProducerIndex::new(size)?,
        })
    }

    /// Publish one descriptor for transmission. Returns `false` if the ring
    /// is currently full.
    pub fn push(&mut self, desc: xdp_desc) -> bool {
        let live_consumer = self.memory.consumer().load(Ordering::Acquire);
        let Some(start) = self.index.reserve(1, live_consumer) else {
            return false;
        };
        let slot = self.index.slot(start);
        // SAFETY: `slot` is in `[0, size)` and the descriptor array holds
        // `size` `xdp_desc` entries in the live mapping.
        unsafe {
            *self.memory.desc_ptr::<xdp_desc>().add(slot) = desc;
        }
        self.memory
            .producer()
            .store(self.index.commit(), Ordering::Release);
        true
    }

    /// Publish a batch of descriptors for transmission with a single atomic
    /// commit, returning how many were accepted (`min(descs.len(), free)`).
    pub fn push_batch(&mut self, descs: &[xdp_desc]) -> usize {
        let want = u32::try_from(descs.len()).unwrap_or(u32::MAX);
        let live_consumer = self.memory.consumer().load(Ordering::Acquire);
        let (start, granted) = self.index.reserve_up_to(want, live_consumer);
        for offset in 0..granted {
            let slot = self.index.slot(start.wrapping_add(offset));
            // SAFETY: `slot` is in `[0, size)` and the descriptor array holds
            // `size` `xdp_desc` entries in the live mapping.
            unsafe {
                *self.memory.desc_ptr::<xdp_desc>().add(slot) = descs[offset as usize];
            }
        }
        self.memory
            .producer()
            .store(self.index.commit(), Ordering::Release);
        granted as usize
    }

    /// Whether the kernel set `XDP_RING_NEED_WAKEUP` on the TX ring — it needs
    /// a `sendto` kick to pick up queued descriptors.
    #[must_use]
    pub fn needs_wakeup(&self) -> bool {
        self.memory.flags().load(Ordering::Acquire) & uapi::XDP_RING_NEED_WAKEUP != 0
    }
}

/// The UMEM frame layout an [`XskSocket`] registers.
#[derive(Debug, Clone, Copy)]
pub struct UmemConfig {
    pub frame_count: u32,
    pub frame_size: u32,
}

/// Entry counts for the four rings; each must be a non-zero power of two.
#[derive(Debug, Clone, Copy)]
pub struct RingSizes {
    pub fill: u32,
    pub completion: u32,
    pub rx: u32,
    pub tx: u32,
}

/// A bound AF_XDP socket: the raw fd, its UMEM frame region, and the four
/// mmap'd rings over it.
pub struct XskSocket {
    fd: OwnedFd,
    umem: Umem,
    fill: FillRing,
    completion: CompletionRing,
    rx: RxRing,
    tx: TxRing,
}

impl XskSocket {
    /// Bring up an AF_XDP socket on `ifname`/`queue_id`: allocate the UMEM,
    /// register it, size the four rings, mmap them, and bind.
    ///
    /// # Errors
    /// Propagates the first failing step as a typed [`XdpError`] carrying the
    /// kernel's errno.
    pub fn bind(
        ifname: &str,
        queue_id: u32,
        umem_cfg: UmemConfig,
        ring_sizes: RingSizes,
        flags: u16,
    ) -> Result<Self, XdpError> {
        let ifindex = sys::if_nametoindex(ifname)?;
        let fd = sys::xsk_socket()?;
        let raw_fd = fd.as_raw_fd();

        let umem = Umem::new(umem_cfg.frame_count, umem_cfg.frame_size)?;

        let reg = xdp_umem_reg {
            addr: umem.base_addr(),
            len: umem.len() as u64,
            chunk_size: umem_cfg.frame_size,
            headroom: sized::UMEM_HEADROOM,
            flags: 0,
            tx_metadata_len: 0,
        };
        sys::umem_reg(raw_fd, &reg)?;

        sys::set_ring_size(raw_fd, uapi::XDP_UMEM_FILL_RING, ring_sizes.fill, "fill")?;
        sys::set_ring_size(
            raw_fd,
            uapi::XDP_UMEM_COMPLETION_RING,
            ring_sizes.completion,
            "completion",
        )?;
        sys::set_ring_size(raw_fd, uapi::XDP_RX_RING, ring_sizes.rx, "rx")?;
        sys::set_ring_size(raw_fd, uapi::XDP_TX_RING, ring_sizes.tx, "tx")?;

        let offsets = sys::mmap_offsets(raw_fd)?;

        let fill_memory = RingMemory::map(
            raw_fd,
            uapi::XDP_UMEM_PGOFF_FILL_RING,
            ring_sizes.fill,
            offsets.fill,
            size_of::<u64>(),
            "fill",
        )?;
        let completion_memory = RingMemory::map(
            raw_fd,
            uapi::XDP_UMEM_PGOFF_COMPLETION_RING,
            ring_sizes.completion,
            offsets.completion,
            size_of::<u64>(),
            "completion",
        )?;
        let rx_memory = RingMemory::map(
            raw_fd,
            uapi::XDP_PGOFF_RX_RING,
            ring_sizes.rx,
            offsets.rx,
            size_of::<xdp_desc>(),
            "rx",
        )?;
        let tx_memory = RingMemory::map(
            raw_fd,
            uapi::XDP_PGOFF_TX_RING,
            ring_sizes.tx,
            offsets.tx,
            size_of::<xdp_desc>(),
            "tx",
        )?;

        let fill = FillRing::new(fill_memory, ring_sizes.fill)?;
        let completion = CompletionRing::new(completion_memory, ring_sizes.completion)?;
        let rx = RxRing::new(rx_memory, ring_sizes.rx)?;
        let tx = TxRing::new(tx_memory, ring_sizes.tx)?;

        sys::bind_xdp(raw_fd, ifindex, queue_id, flags)?;

        Ok(Self {
            fd,
            umem,
            fill,
            completion,
            rx,
            tx,
        })
    }

    /// The raw socket file descriptor (for polling/epoll registration).
    #[must_use]
    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Kick the kernel to process the TX ring. In copy/SKB mode the kernel
    /// transmits queued descriptors only when poked by a `sendto`; `EBUSY`/
    /// `EAGAIN`/`ENOBUFS` mean "already working / retry later" and are not
    /// errors.
    ///
    /// # Errors
    /// Any other `sendto` failure as an [`io::Error`].
    pub fn kick_tx(&self) -> io::Result<()> {
        // SAFETY: a zero-length `sendto` on our own fd with null buffer/addr;
        // this is the documented AF_XDP TX wakeup and touches no user memory.
        let ret = unsafe {
            libc::sendto(
                self.fd(),
                ptr::null(),
                0,
                libc::MSG_DONTWAIT,
                ptr::null(),
                0,
            )
        };
        if ret < 0 {
            let error = io::Error::last_os_error();
            return match error.raw_os_error() {
                Some(libc::EBUSY | libc::EAGAIN | libc::ENOBUFS) => Ok(()),
                _ => Err(error),
            };
        }
        Ok(())
    }

    /// Whether the FILL ring needs a syscall kick (need-wakeup mode): the
    /// kernel ran out of fill descriptors and set the flag, so after we
    /// replenish it we must poke the RX path via [`XskSocket::wake_rx`].
    #[must_use]
    pub fn fill_needs_wakeup(&self) -> bool {
        self.fill.needs_wakeup()
    }

    /// Whether the TX ring needs a `sendto` kick (need-wakeup mode): only then
    /// must we syscall to make the kernel drain queued TX descriptors.
    #[must_use]
    pub fn tx_needs_wakeup(&self) -> bool {
        self.tx.needs_wakeup()
    }

    /// Poke the kernel to consume the FILL ring (the RX-side wakeup). A
    /// zero-length `recvfrom` is the documented AF_XDP RX wakeup; `EAGAIN`/
    /// `EBUSY`/`ENOBUFS` mean "already draining / nothing to do".
    ///
    /// # Errors
    /// Any other `recvfrom` failure as an [`io::Error`].
    pub fn wake_rx(&self) -> io::Result<()> {
        // SAFETY: zero-length `recvfrom` on our own fd with null buffer/addr;
        // the documented AF_XDP RX wakeup, touches no user memory.
        let ret = unsafe {
            libc::recvfrom(
                self.fd(),
                ptr::null_mut(),
                0,
                libc::MSG_DONTWAIT,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if ret < 0 {
            let error = io::Error::last_os_error();
            return match error.raw_os_error() {
                Some(libc::EAGAIN | libc::EBUSY | libc::ENOBUFS) => Ok(()),
                _ => Err(error),
            };
        }
        Ok(())
    }

    #[must_use]
    pub fn umem(&self) -> &Umem {
        &self.umem
    }

    pub fn umem_mut(&mut self) -> &mut Umem {
        &mut self.umem
    }

    pub fn fill_mut(&mut self) -> &mut FillRing {
        &mut self.fill
    }

    pub fn completion_mut(&mut self) -> &mut CompletionRing {
        &mut self.completion
    }

    pub fn rx_mut(&mut self) -> &mut RxRing {
        &mut self.rx
    }

    pub fn tx_mut(&mut self) -> &mut TxRing {
        &mut self.tx
    }
}
