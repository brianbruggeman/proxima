//! AF_XDP kernel UAPI — the `linux/if_xdp.h` layout the datapath drives over raw
//! syscalls. These are stable kernel ABI (unlike dpdk's version-volatile mbuf
//! internals, which is why this crate needs no build-time offset probe). Kept
//! pure so the definitions compile and are reviewable in mac CI even though only
//! the linux `xdp` feature actually issues the syscalls.

#![allow(non_camel_case_types)]

/// Address family / protocol family for XDP sockets.
pub const AF_XDP: u16 = 44;

/// `setsockopt`/`getsockopt` level for XDP options.
pub const SOL_XDP: i32 = 283;

/// `setsockopt` option names.
pub const XDP_MMAP_OFFSETS: i32 = 1;
pub const XDP_RX_RING: i32 = 2;
pub const XDP_TX_RING: i32 = 3;
pub const XDP_UMEM_REG: i32 = 4;
pub const XDP_UMEM_FILL_RING: i32 = 5;
pub const XDP_UMEM_COMPLETION_RING: i32 = 6;
pub const XDP_STATISTICS: i32 = 7;
pub const XDP_OPTIONS: i32 = 8;

/// `mmap` page offsets that select which ring to map.
pub const XDP_PGOFF_RX_RING: u64 = 0;
pub const XDP_PGOFF_TX_RING: u64 = 0x8000_0000;
pub const XDP_UMEM_PGOFF_FILL_RING: u64 = 0x1_0000_0000;
pub const XDP_UMEM_PGOFF_COMPLETION_RING: u64 = 0x1_8000_0000;

/// `sockaddr_xdp::sxdp_flags` bind flags.
pub const XDP_SHARED_UMEM: u16 = 1 << 0;
pub const XDP_COPY: u16 = 1 << 1;
pub const XDP_ZEROCOPY: u16 = 1 << 2;
pub const XDP_USE_NEED_WAKEUP: u16 = 1 << 3;

/// `xdp_ring_offset::flags` / `xdp_umem_reg::flags` bits.
pub const XDP_RING_NEED_WAKEUP: u32 = 1 << 0;
pub const XDP_UMEM_UNALIGNED_CHUNK_FLAG: u32 = 1 << 0;

/// `bind(2)` target: which interface + queue the socket attaches to.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct sockaddr_xdp {
    pub sxdp_family: u16,
    pub sxdp_flags: u16,
    pub sxdp_ifindex: u32,
    pub sxdp_queue_id: u32,
    pub sxdp_shared_umem_fd: u32,
}

/// Where one ring's `producer`, `consumer`, `desc`, and `flags` live within its
/// mmap'd region, as reported by `getsockopt(XDP_MMAP_OFFSETS)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct xdp_ring_offset {
    pub producer: u64,
    pub consumer: u64,
    pub desc: u64,
    pub flags: u64,
}

/// The offsets for all four rings, filled by `getsockopt(XDP_MMAP_OFFSETS)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct xdp_mmap_offsets {
    pub rx: xdp_ring_offset,
    pub tx: xdp_ring_offset,
    pub fill: xdp_ring_offset,
    pub completion: xdp_ring_offset,
}

/// UMEM registration passed to `setsockopt(XDP_UMEM_REG)`: the shared memory
/// region, its chunk size, and per-chunk headroom.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct xdp_umem_reg {
    pub addr: u64,
    pub len: u64,
    pub chunk_size: u32,
    pub headroom: u32,
    pub flags: u32,
    pub tx_metadata_len: u32,
}

/// An RX/TX descriptor: the UMEM-relative `addr` of the frame and its `len`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct xdp_desc {
    pub addr: u64,
    pub len: u32,
    pub options: u32,
}
