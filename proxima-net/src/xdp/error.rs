use thiserror::Error;

/// Failures bringing up or driving an AF_XDP socket. The `i32` codes are the
/// kernel's negative errno returns, surfaced verbatim so the cause is
/// recoverable from the log alone.
#[derive(Debug, Error)]
pub enum XdpError {
    #[error("a socket argument contained an interior nul byte")]
    ArgNul,

    #[error("requested a ring of {0} entries; ring size must be a non-zero power of two")]
    RingSizeNotPowerOfTwo(u32),

    #[error("socket(AF_XDP) failed (errno {0})")]
    Socket(i32),

    #[error("XDP_UMEM_REG setsockopt failed (errno {0})")]
    UmemReg(i32),

    #[error("ring-size setsockopt {ring} failed (errno {errno})")]
    RingSetup { ring: &'static str, errno: i32 },

    #[error("XDP_MMAP_OFFSETS getsockopt failed (errno {0})")]
    MmapOffsets(i32),

    #[error("mmap of the {ring} ring failed (errno {errno})")]
    RingMmap { ring: &'static str, errno: i32 },

    #[error("bind to interface {interface} queue {queue} failed (errno {errno})")]
    Bind {
        interface: String,
        queue: u32,
        errno: i32,
    },

    #[error("no free UMEM frame for the fill ring")]
    UmemExhausted,

    #[error("no such network interface {interface} (errno {errno})")]
    InterfaceNotFound { interface: String, errno: i32 },

    #[error("reactor read-readiness registration failed (errno {0})")]
    Readiness(i32),

    #[error("failed to create the xskmap (errno {0})")]
    BpfMapCreate(i32),

    #[error("failed to update xskmap[{queue}]=fd (errno {errno})")]
    BpfMapUpdate { queue: u32, errno: i32 },

    #[error("failed to load the xsk redirect program (errno {errno}): {log}")]
    BpfLoad { errno: i32, log: String },

    #[error("failed to open a netlink route socket (errno {0})")]
    Netlink(i32),

    #[error("failed to attach the xdp program to ifindex {ifindex} (errno {errno})")]
    BpfAttach { ifindex: u32, errno: i32 },
}
