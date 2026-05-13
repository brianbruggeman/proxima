//! Loss detection + RTT estimation per [RFC 9002] §5 + §6.
//!
//! Composed entry point: [`LossDetection`] orchestrates per-epoch
//! sent-packet tracking + RTT updates + packet/time-threshold loss
//! declaration + PTO timer per the [C14 paper proof].
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). All storage is `arrayvec::ArrayVec`
//! no-alloc; arithmetic uses [`crate::quic::time::Duration`] saturating
//! newtypes.
//!
//! [RFC 9002]: https://www.rfc-editor.org/rfc/rfc9002
//! [C14 paper proof]: ../../docs/proxima-quic/c14-loss-detection-design.md

pub mod constants;
pub mod detector;
pub mod rtt;
pub mod sent_packet;

pub use constants::{
    K_GRANULARITY_MICROS, K_INITIAL_RTT_MICROS, K_PACKET_THRESHOLD,
    K_PERSISTENT_CONGESTION_THRESHOLD, K_TIME_THRESHOLD_DENOM, K_TIME_THRESHOLD_NUM,
    MAX_SENT_PACKETS,
};
pub use detector::{LossDetection, LossOutcome};
pub use rtt::RttEstimator;
pub use sent_packet::{SentPacket, SentPacketQueue};
