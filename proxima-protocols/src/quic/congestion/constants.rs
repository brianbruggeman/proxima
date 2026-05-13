//! Congestion-control constants per RFC 9002 §B + §7.2.

/// Loss-reduction factor numerator (1/2 per RFC 9002 §7.3.2).
pub const K_LOSS_REDUCTION_NUM: u64 = 1;

/// Loss-reduction factor denominator (1/2 per RFC 9002 §7.3.2).
pub const K_LOSS_REDUCTION_DENOM: u64 = 2;

/// Default max datagram size per RFC 9000 §14. PMTUD may raise this.
pub const DEFAULT_MAX_DATAGRAM_SIZE: u64 = 1200;

/// Initial congestion window in DATAGRAMS per RFC 9002 §7.2. The
/// recommended value is 10 × max_datagram_size, clamped at 14720 B.
pub const K_INITIAL_WINDOW_DATAGRAMS: u64 = 10;

/// Minimum congestion window in DATAGRAMS per RFC 9002 §7.2 (2 × MTU).
pub const K_MIN_WINDOW_DATAGRAMS: u64 = 2;

/// Persistent-congestion threshold per RFC 9002 §7.6 (multiplier on PTO).
pub const K_PERSISTENT_CONGESTION_THRESHOLD: u64 = 3;
