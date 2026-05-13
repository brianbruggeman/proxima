//! Loss-detection constants per [RFC 9002 §A.2].
//!
//! [RFC 9002 §A.2]: https://www.rfc-editor.org/rfc/rfc9002#appendix-A.2

/// Maximum reordering in packets before packet-threshold loss is
/// declared per RFC 9002 §6.1.1. Recommended value is 3.
pub const K_PACKET_THRESHOLD: u64 = 3;

/// Time-threshold numerator (9/8 RTT per RFC 9002 §6.1.2).
pub const K_TIME_THRESHOLD_NUM: u64 = 9;

/// Time-threshold denominator (9/8 RTT per RFC 9002 §6.1.2).
pub const K_TIME_THRESHOLD_DENOM: u64 = 8;

/// Clock-granularity guard per RFC 9002 §A.2. Recommended 1 ms.
pub const K_GRANULARITY_MICROS: u64 = 1_000;

/// Default initial RTT before the first sample per RFC 9002 §6.2.2.
/// Recommended 333 ms.
pub const K_INITIAL_RTT_MICROS: u64 = 333_000;

/// Persistent-congestion threshold per RFC 9002 §7.6. Recommended 3.
/// Consumed by C15 congestion-control reset logic.
pub const K_PERSISTENT_CONGESTION_THRESHOLD: u32 = 3;

/// Per-epoch sent-packets queue capacity. Sourced from
/// `proxima-quic-proto.toml [loss].max_sent_packets` (override via
/// `PROXIMA_QUIC_PROTO_LOSS_MAX_SENT_PACKETS`). Drop-oldest on
/// overflow.
pub const MAX_SENT_PACKETS: usize = crate::quic::sized::LOSS_MAX_SENT_PACKETS;
