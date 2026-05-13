//! The throughput load as a FIRST-CLASS config — both ways.
//!
//! [`LoadPlan`] is the wrk-beating closed-loop run ([`crate::engine::drive_throughput`],
//! the `send_raw` hot loop) expressed as a value you can build TWO equivalent
//! ways, neither the lesser:
//!
//! - a `conflaguration` config surface: TOML + typed, per-key, named-default env
//!   (`REKT_TARGET`, `REKT_CONNECTIONS_PER_CORE`, `REKT_CORES`, `REKT_DURATION_SECS`)
//!   via the derived [`LoadPlan::from_env`];
//! - a fluent builder: `LoadPlan::builder().target(..).connections_per_core(..).build()`
//!   via the derived [`LoadPlan::builder`].
//!
//! Both resolve to the identical immutable `LoadPlan`, proven by the parity test
//! below, and both round-trip through serde (built -> config). The config is
//! resolved ONCE here; [`LoadPlan::run`] then drives the same monomorphized
//! `H1ClientUpstream::send_raw` loop — so the first-class config surface costs
//! nothing on the hot path. This mirrors proxima's own `H1ClientConfig` pattern.

use std::time::Duration;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

use proxima_runtime::concurrency::Concurrency;

use crate::engine::{Throughput, drive_adaptive, drive_throughput};
use crate::error::Error;

fn default_connections_per_core() -> usize {
    25
}

fn default_cores() -> usize {
    1
}

fn default_duration_secs() -> u64 {
    5
}

/// A closed-loop throughput run: where to hit, how many keep-alive connections
/// per core, how many cores, for how long. The builder result IS the config —
/// same data shape both ways.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "REKT")]
#[builder(derive(Clone, Debug))]
pub struct LoadPlan {
    /// Target URL, e.g. `"http://127.0.0.1:8080/"`. Required (no default).
    pub target: String,

    /// Keep-alive connections per core. With the default adaptive drive this is
    /// the SEED the controller starts from (and searches around); pin it as a hard
    /// cap by also setting `adaptive = false`.
    #[setting(default = 25)]
    #[serde(default = "default_connections_per_core")]
    #[builder(default = 25)]
    pub connections_per_core: usize,

    /// Worker cores the load fans across.
    #[setting(default = 1)]
    #[serde(default = "default_cores")]
    #[builder(default = 1)]
    pub cores: usize,

    /// Run duration in whole seconds.
    #[setting(default = 5)]
    #[serde(default = "default_duration_secs")]
    #[builder(default = 5)]
    pub duration_secs: u64,

    /// Adapt the per-core in-flight count to the workload. **On by default** — the
    /// same reasoning as float-default affinity: a fixed connections-per-core is
    /// wrong for every workload but the one it was tuned on, and the adaptive
    /// drive is wrk-competitive (matches wrk, ~2% under a perfectly-tuned fixed)
    /// while it auto-finds the crest and dodges the 3.8–7.5× mis-set loss. rekt
    /// drives the throughput-maximising hillclimb controller seeded at
    /// `connections_per_core` (a load-gen's intent is max throughput; a real
    /// server would default to the latency-safe gradient preset). Set
    /// `adaptive = false` to pin a fixed `connections_per_core` cap.
    #[setting(default = true)]
    #[serde(default = "default_adaptive")]
    #[builder(default = true)]
    pub adaptive: bool,
}

fn default_adaptive() -> bool {
    true
}

impl Validate for LoadPlan {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.target.is_empty() {
            errors.push(ValidationMessage::new("target", "must be non-empty"));
        }
        if self.connections_per_core == 0 {
            errors.push(ValidationMessage::new("connections_per_core", "must be >= 1"));
        }
        if self.cores == 0 {
            errors.push(ValidationMessage::new("cores", "must be >= 1"));
        }
        if errors.is_empty() { Ok(()) } else { Err(conflaguration::Error::Validation { errors }) }
    }
}

impl LoadPlan {
    /// Load from a TOML string (the file half of the config surface).
    pub fn from_toml(text: &str) -> Result<Self, Error> {
        let plan: Self = toml::from_str(text).map_err(|err| Error::Engine(err.to_string()))?;
        plan.validate()
            .map_err(|err| Error::Engine(err.to_string()))?;
        Ok(plan)
    }

    /// The resolved run duration.
    #[must_use]
    pub fn duration(&self) -> Duration {
        Duration::from_secs(self.duration_secs)
    }

    /// The resolved per-core concurrency knob — one primitive, the same type a
    /// proxima server uses for its per-core handler limit. `Fixed` when adaptive
    /// is off (back-compat); a hillclimb controller seeded at
    /// `connections_per_core` when on. Built fresh per call (the controller holds
    /// a live, non-`Clone` law), so each core gets its own.
    pub fn concurrency(&self) -> Result<Concurrency, Error> {
        if !self.adaptive {
            return Ok(Concurrency::fixed(self.connections_per_core));
        }
        let seed = self.connections_per_core.max(1);
        Concurrency::builder()
            .hillclimb()
            .start(seed)
            .bounds(1, seed.saturating_mul(8).max(8))
            .build()
            .map_err(|err| Error::Engine(err.to_string()))
    }

    /// Drive the run — the same `send_raw` closed-loop the CLI bench drives.
    /// Fixed cap → the historical monomorphic loop; adaptive → the controller
    /// drives the per-core in-flight count toward the crest each window.
    pub fn run(&self) -> Result<Throughput, Error> {
        self.validate()
            .map_err(|err| Error::Engine(err.to_string()))?;
        if self.adaptive {
            drive_adaptive(&self.target, self.connections_per_core, self.cores, self.duration())
        } else {
            drive_throughput(&self.target, self.connections_per_core, self.cores, self.duration())
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn reference() -> LoadPlan {
        LoadPlan::builder()
            .target("http://127.0.0.1:8080/".to_string())
            .connections_per_core(25)
            .cores(1)
            .duration_secs(5)
            .build()
    }

    /// P4 parity: the conflaguration config surface (TOML + typed env) and the
    /// fluent builder resolve to IDENTICAL state — neither is the lesser path.
    #[test]
    fn builder_toml_and_env_resolve_identically() {
        let via_builder = reference();

        let via_toml = LoadPlan::from_toml(
            "target = \"http://127.0.0.1:8080/\"\n\
             connections_per_core = 25\n\
             cores = 1\n\
             duration_secs = 5\n",
        )
        .expect("toml load");
        assert_eq!(via_toml, via_builder, "TOML config == fluent builder");

        // typed, per-key, named-default env overlay (the RUST_LOG-killer shape:
        // one key per setting, not a positional comma-string).
        temp_env::with_vars(
            [
                ("REKT_TARGET", Some("http://127.0.0.1:8080/")),
                ("REKT_CONNECTIONS_PER_CORE", Some("25")),
                ("REKT_CORES", Some("1")),
                ("REKT_DURATION_SECS", Some("5")),
            ],
            || {
                let via_env = LoadPlan::from_env().expect("env load");
                assert_eq!(via_env, via_builder, "typed env == fluent builder");
            },
        );
    }

    /// Built -> config round-trip: a value built fluently serializes and reloads
    /// to the same value (the wire form survives a hop).
    #[test]
    fn built_round_trips_through_config() {
        let original = reference();
        let serialized = toml::to_string(&original).expect("serialize");
        let reloaded = LoadPlan::from_toml(&serialized).expect("reload");
        assert_eq!(reloaded, original);
    }

    /// Named defaults apply when only the required field is supplied — the
    /// config never demands the operator restate the obvious.
    #[test]
    fn defaults_fill_when_only_target_given() {
        let minimal = LoadPlan::from_toml("target = \"http://localhost/\"\n").expect("toml");
        assert_eq!(minimal.connections_per_core, 25);
        assert_eq!(minimal.cores, 1);
        assert_eq!(minimal.duration_secs, 5);
        // and the fluent builder agrees on those defaults from one setter.
        let fluent = LoadPlan::builder()
            .target("http://localhost/".to_string())
            .build();
        assert_eq!(fluent, minimal);
    }

    /// Validation rejects an empty target and a zero core/conn count through
    /// the SAME `Validate` impl both surfaces run.
    #[test]
    fn validation_rejects_degenerate_plans() {
        assert!(LoadPlan::from_toml("target = \"\"\n").is_err());
        assert!(LoadPlan::from_toml("target = \"http://x/\"\ncores = 0\n").is_err());
    }

    /// Adaptive is the DEFAULT (the float-default reasoning): a bare plan resolves
    /// to a controller seeded at `connections_per_core`, not a fixed cap.
    #[test]
    fn default_resolves_to_adaptive() {
        let plan = LoadPlan::builder()
            .target("http://x/".to_string())
            .connections_per_core(25)
            .build();
        assert!(plan.adaptive, "adaptive is on by default");
        let concurrency = plan.concurrency().expect("resolve");
        assert!(matches!(concurrency, Concurrency::Adaptive(_)));
        assert_eq!(concurrency.initial(), 25, "seeded at connections_per_core");
    }

    /// Opting OUT (`adaptive = false`) pins the fixed cap — the historical drive.
    #[test]
    fn opt_out_resolves_to_fixed_cap() {
        let plan = LoadPlan::builder()
            .target("http://x/".to_string())
            .connections_per_core(25)
            .adaptive(false)
            .build();
        assert!(matches!(plan.concurrency().expect("resolve"), Concurrency::Fixed(25)));
    }

    /// `adaptive` is a first-class config key on both surfaces.
    #[test]
    fn adaptive_is_config_and_builder_first_class() {
        let via_toml = LoadPlan::from_toml("target = \"http://x/\"\nadaptive = true\n").expect("toml");
        assert!(via_toml.adaptive);
        let via_builder = LoadPlan::builder()
            .target("http://x/".to_string())
            .adaptive(true)
            .build();
        assert_eq!(via_toml, via_builder);
    }
}
