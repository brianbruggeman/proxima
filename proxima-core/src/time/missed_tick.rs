//! `MissedTickBehavior` — match tokio's three-way enum for how the
//! [`Interval`](super::Interval) recovers when a consumer falls behind
//! the period.

/// What happens when an [`Interval`](super::Interval) consumer skips
/// past a tick deadline (consumer time exceeded the period). Shape
/// matches `tokio::time::MissedTickBehavior`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MissedTickBehavior {
    /// Catch up — fire missed ticks back-to-back until the deadline
    /// converges. Useful for "every tick must run" semantics, but the
    /// burst can starve other work. Tokio's default.
    #[default]
    Burst,
    /// Delay — anchor the next deadline to (now + period). Drifts
    /// over time but never bursts. Useful for periodic maintenance
    /// that should pace from each completion.
    Delay,
    /// Skip — anchor the next deadline to (now + period), discarding
    /// the missed ticks. Useful for "latest tick wins" semantics
    /// (rate-limit sweepers, health pings). Bounded drift; no burst.
    Skip,
}
