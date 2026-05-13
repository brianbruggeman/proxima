use std::sync::Arc;

use serde::Deserialize;

use crate::sched::inflight::InFlight;
use crate::sched::pacer::GridPacer;

/// conflaguration-derived config for the open-loop scheduler.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub struct RateSpec {
    pub rate_per_sec: f64,
    #[serde(default = "default_max_in_flight")]
    pub max_in_flight: u32,
}

fn default_max_in_flight() -> u32 {
    1024
}

/// ties the grid pacer to the bounded gate. constructed from config or the
/// fluent builder; the two are required to agree (see tests).
#[derive(Debug)]
pub struct Scheduler {
    pub pacer: GridPacer,
    pub in_flight: Arc<InFlight>,
    rate_per_sec: f64,
    max_in_flight: u32,
}

impl Scheduler {
    #[must_use]
    pub fn from_spec(spec: RateSpec) -> Self {
        Self {
            pacer: GridPacer::new(spec.rate_per_sec),
            in_flight: InFlight::new(spec.max_in_flight),
            rate_per_sec: spec.rate_per_sec,
            max_in_flight: spec.max_in_flight,
        }
    }

    #[must_use]
    pub fn builder() -> SchedulerBuilder {
        SchedulerBuilder::default()
    }

    #[must_use]
    pub fn rate_per_sec(&self) -> f64 {
        self.rate_per_sec
    }

    #[must_use]
    pub fn max_in_flight(&self) -> u32 {
        self.max_in_flight
    }
}

#[derive(Debug)]
pub struct SchedulerBuilder {
    rate_per_sec: f64,
    max_in_flight: u32,
}

impl Default for SchedulerBuilder {
    fn default() -> Self {
        Self {
            rate_per_sec: 0.0,
            max_in_flight: default_max_in_flight(),
        }
    }
}

impl SchedulerBuilder {
    #[must_use]
    pub fn rate_per_sec(mut self, rate_per_sec: f64) -> Self {
        self.rate_per_sec = rate_per_sec;
        self
    }

    #[must_use]
    pub fn max_in_flight(mut self, max_in_flight: u32) -> Self {
        self.max_in_flight = max_in_flight;
        self
    }

    #[must_use]
    pub fn build(self) -> Scheduler {
        Scheduler::from_spec(RateSpec {
            rate_per_sec: self.rate_per_sec,
            max_in_flight: self.max_in_flight,
        })
    }
}

#[cfg(test)]
mod tests {
    // tests assert on known states; unwrap/expect are the clearer failure here
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[proxima::test]
    async fn config_and_builder_agree() {
        let from_toml: RateSpec = toml::from_str("rate_per_sec = 1000.0\nmax_in_flight = 512").expect("parse");
        let cfg = Scheduler::from_spec(from_toml);
        let fluent = Scheduler::builder()
            .rate_per_sec(1000.0)
            .max_in_flight(512)
            .build();

        assert!((cfg.rate_per_sec() - fluent.rate_per_sec()).abs() < f64::EPSILON);
        assert_eq!(cfg.max_in_flight(), fluent.max_in_flight());
        assert_eq!(cfg.in_flight.max(), fluent.in_flight.max());
    }

    #[proxima::test]
    async fn max_in_flight_defaults_when_absent() {
        let spec: RateSpec = toml::from_str("rate_per_sec = 50.0").expect("parse");
        assert_eq!(spec.max_in_flight, 1024);
    }
}
