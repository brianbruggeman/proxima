//! DPDK packet listener — a prime-native UDP `PacketListener` over a dpdk PMD.
//! The implementation lives in `proxima-net-dpdk` (the kernel-bypass I/O floor);
//! this is the re-export that mounts it into the proxima listener registry under
//! the `dpdk` feature. No tokio: the listener busy-polls the RX ring on its
//! prime core (a PMD has no fd to register).

pub use proxima_net::dpdk::DpdkPacketListener;
