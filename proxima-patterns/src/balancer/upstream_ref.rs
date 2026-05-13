use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use proxima_primitives::pipe::handler::{PipeHandle, ThreadLocalPipeHandle};

#[derive(Debug)]
pub struct UpstreamMetrics {
    pub in_flight: AtomicU32,
    pub recent_latency_us: AtomicU64,
    pub healthy: AtomicBool,
    pub consecutive_failures: AtomicU32,
    pub consecutive_successes: AtomicU32,
}

impl Default for UpstreamMetrics {
    fn default() -> Self {
        Self {
            in_flight: AtomicU32::new(0),
            recent_latency_us: AtomicU64::new(0),
            healthy: AtomicBool::new(true),
            consecutive_failures: AtomicU32::new(0),
            consecutive_successes: AtomicU32::new(0),
        }
    }
}

impl UpstreamMetrics {
    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn recent_latency(&self) -> Duration {
        Duration::from_micros(self.recent_latency_us.load(Ordering::Relaxed))
    }

    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    pub fn record_latency(&self, latency: Duration) {
        let micros = latency.as_micros().min(u128::from(u64::MAX)) as u64;
        let prior = self.recent_latency_us.load(Ordering::Relaxed);
        let blended = ewma_micros(prior, micros);
        self.recent_latency_us.store(blended, Ordering::Relaxed);
    }

    pub fn note_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.consecutive_successes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_failure(&self) {
        self.consecutive_successes.store(0, Ordering::Relaxed);
        self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn apply_outlier_policy(&self, policy: &OutlierPolicy) {
        let was_healthy = self.healthy.load(Ordering::Relaxed);
        let failures = self.consecutive_failures.load(Ordering::Relaxed);
        let successes = self.consecutive_successes.load(Ordering::Relaxed);
        if was_healthy && failures >= policy.eject_after_failures {
            self.healthy.store(false, Ordering::Relaxed);
        } else if !was_healthy && successes >= policy.recovery_after_successes {
            self.healthy.store(true, Ordering::Relaxed);
        }
    }
}

#[derive(Debug, Clone)]
pub struct OutlierPolicy {
    pub eject_after_failures: u32,
    pub recovery_after_successes: u32,
}

impl Default for OutlierPolicy {
    fn default() -> Self {
        Self {
            eject_after_failures: 5,
            recovery_after_successes: 2,
        }
    }
}

fn ewma_micros(prior: u64, sample: u64) -> u64 {
    if prior == 0 {
        return sample;
    }
    let prior = prior as u128;
    let sample = sample as u128;
    ((prior * 7 + sample) / 8) as u64
}

#[derive(Clone)]
pub struct UpstreamRef {
    pub pipe: PipeHandle,
    pub metrics: Arc<UpstreamMetrics>,
    pub name: String,
    pub weight: u32,
    pub outlier_policy: OutlierPolicy,
}

impl UpstreamRef {
    #[must_use]
    pub fn new(pipe: PipeHandle, name: impl Into<String>, weight: u32) -> Self {
        Self {
            pipe,
            metrics: Arc::new(UpstreamMetrics::default()),
            name: name.into(),
            weight: weight.max(1),
            outlier_policy: OutlierPolicy::default(),
        }
    }

    #[must_use]
    pub fn with_outlier_policy(mut self, policy: OutlierPolicy) -> Self {
        self.outlier_policy = policy;
        self
    }

    pub fn track_call(&self) -> CallTracker<'_> {
        self.metrics.in_flight.fetch_add(1, Ordering::Relaxed);
        CallTracker {
            metrics: &self.metrics,
            policy: &self.outlier_policy,
            started: Instant::now(),
            settled: false,
        }
    }
}

impl std::fmt::Debug for UpstreamRef {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UpstreamRef")
            .field("name", &self.name)
            .field("weight", &self.weight)
            .field("in_flight", &self.metrics.in_flight())
            .field("recent_latency", &self.metrics.recent_latency())
            .field("healthy", &self.metrics.is_healthy())
            .finish()
    }
}

/// Per-thread sibling of [`UpstreamRef`] for selection paths driven by
/// a [`crate::pipe::ThreadLocalPipe`].
#[derive(Clone)]
pub struct ThreadLocalUpstreamRef {
    pub pipe: ThreadLocalPipeHandle,
    pub metrics: Arc<UpstreamMetrics>,
    pub name: String,
    pub weight: u32,
    pub outlier_policy: OutlierPolicy,
}

impl ThreadLocalUpstreamRef {
    #[must_use]
    pub fn new(pipe: ThreadLocalPipeHandle, name: impl Into<String>, weight: u32) -> Self {
        Self {
            pipe,
            metrics: Arc::new(UpstreamMetrics::default()),
            name: name.into(),
            weight: weight.max(1),
            outlier_policy: OutlierPolicy::default(),
        }
    }

    #[must_use]
    pub fn with_outlier_policy(mut self, policy: OutlierPolicy) -> Self {
        self.outlier_policy = policy;
        self
    }

    pub fn track_call(&self) -> CallTracker<'_> {
        self.metrics.in_flight.fetch_add(1, Ordering::Relaxed);
        CallTracker {
            metrics: &self.metrics,
            policy: &self.outlier_policy,
            started: Instant::now(),
            settled: false,
        }
    }
}

impl std::fmt::Debug for ThreadLocalUpstreamRef {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ThreadLocalUpstreamRef")
            .field("name", &self.name)
            .field("weight", &self.weight)
            .field("in_flight", &self.metrics.in_flight())
            .field("recent_latency", &self.metrics.recent_latency())
            .field("healthy", &self.metrics.is_healthy())
            .finish()
    }
}

pub struct CallTracker<'parent> {
    metrics: &'parent UpstreamMetrics,
    policy: &'parent OutlierPolicy,
    started: Instant,
    settled: bool,
}

impl CallTracker<'_> {
    pub fn settle_success(mut self) {
        self.metrics.in_flight.fetch_sub(1, Ordering::Relaxed);
        self.metrics.record_latency(self.started.elapsed());
        self.metrics.note_success();
        self.metrics.apply_outlier_policy(self.policy);
        self.settled = true;
    }

    pub fn settle_failure(mut self) {
        self.metrics.in_flight.fetch_sub(1, Ordering::Relaxed);
        self.metrics.record_latency(self.started.elapsed());
        self.metrics.note_failure();
        self.metrics.apply_outlier_policy(self.policy);
        self.settled = true;
    }
}

impl Drop for CallTracker<'_> {
    fn drop(&mut self) {
        if !self.settled {
            self.metrics.in_flight.fetch_sub(1, Ordering::Relaxed);
            self.metrics.note_failure();
            self.metrics.apply_outlier_policy(self.policy);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn ewma_blends_toward_recent_sample() {
        let blended = ewma_micros(1_000_000, 100_000);
        assert!(blended < 1_000_000);
        assert!(blended > 100_000);
    }

    #[test]
    fn ewma_with_no_prior_returns_sample() {
        assert_eq!(ewma_micros(0, 250), 250);
    }

    #[test]
    fn metrics_record_latency_updates_recent() {
        let metrics = UpstreamMetrics::default();
        metrics.record_latency(Duration::from_millis(5));
        assert!(metrics.recent_latency() > Duration::ZERO);
    }

    #[test]
    fn note_success_resets_consecutive_failures() {
        let metrics = UpstreamMetrics::default();
        metrics.note_failure();
        metrics.note_failure();
        metrics.note_success();
        assert_eq!(metrics.consecutive_failures.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn call_tracker_increments_and_decrements_in_flight() {
        let metrics = UpstreamMetrics::default();
        let policy = OutlierPolicy::default();
        let tracker = CallTracker {
            metrics: &metrics,
            policy: &policy,
            started: Instant::now(),
            settled: false,
        };
        metrics.in_flight.fetch_add(1, Ordering::Relaxed);
        assert_eq!(metrics.in_flight(), 1);
        thread::sleep(Duration::from_millis(1));
        tracker.settle_success();
        assert_eq!(metrics.in_flight(), 0);
        assert!(metrics.recent_latency() > Duration::ZERO);
    }

    #[test]
    fn outlier_policy_marks_unhealthy_after_threshold() {
        let metrics = UpstreamMetrics::default();
        let policy = OutlierPolicy {
            eject_after_failures: 3,
            recovery_after_successes: 2,
        };
        for _ in 0..3 {
            metrics.note_failure();
        }
        metrics.apply_outlier_policy(&policy);
        assert!(!metrics.is_healthy());
    }

    #[test]
    fn recovery_requires_consecutive_successes_threshold() {
        let metrics = UpstreamMetrics::default();
        let policy = OutlierPolicy {
            eject_after_failures: 3,
            recovery_after_successes: 2,
        };
        for _ in 0..3 {
            metrics.note_failure();
        }
        metrics.apply_outlier_policy(&policy);
        assert!(!metrics.is_healthy(), "ejected after 3 failures");

        metrics.note_success();
        metrics.apply_outlier_policy(&policy);
        assert!(
            !metrics.is_healthy(),
            "single success below recovery threshold"
        );

        metrics.note_success();
        metrics.apply_outlier_policy(&policy);
        assert!(
            metrics.is_healthy(),
            "second success crosses recovery threshold"
        );
    }
}
