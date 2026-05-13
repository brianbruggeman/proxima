//! DPDK stream listener — a prime-native TCP `StreamListener` over a dpdk PMD.
//! The implementation (handshake + data path + close + AsyncRead/AsyncWrite)
//! lives in `proxima-net-dpdk`; this re-exports it into the proxima listener
//! registry under the `dpdk` feature. No tokio: the listener busy-polls the RX
//! ring on its prime core.

pub use proxima_net::dpdk::{DpdkStreamConnection, DpdkStreamListener, DpdkStreamUpstream};
