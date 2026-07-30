//! The load plan: one file, one hierarchy.
//!
//! There used to be two config types with an overlapping `target` field —
//! `Scenario` (a staged, paced run parsed from TOML) and `LoadPlan` (a
//! closed-loop throughput run, TOML *or* env *or* builder). Two ways to say
//! "where to hit", and no way to say "these three workloads".
//!
//! Collapsed: [`LoadPlan`] is the file and [`Scenario`] is a member, so a file
//! carries as many workloads as it likes. The duplicated `target` moved onto the
//! scenario that uses it, and `duration_secs` turned out to be a [`Stage`]'s
//! duration — a throughput run is just a scenario with one stage that names no
//! rate.
//!
//! That last point is the unification: **an absent `rate` means flat out.** A
//! closed-loop throughput bench and a paced open-loop run stopped being two
//! kinds of run and became one field.

use std::time::Duration;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

use crate::error::Error;

fn default_connections_per_core() -> usize {
    25
}

fn default_cores() -> usize {
    1
}

fn default_adaptive() -> bool {
    true
}

/// A whole run: how to drive it, what to drive, and the gates it must clear.
///
/// Machine knobs (`cores`, `connections_per_core`, `adaptive`) carry the
/// `conflaguration` env surface, because those are what you override per
/// invocation. The workload does not: scenarios come from the file, which is
/// where a list of them can actually be written.
#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "REKT")]
#[builder(derive(Clone, Debug))]
#[serde(deny_unknown_fields)]
pub struct LoadPlan {
    /// Worker cores the load fans across.
    #[setting(default = 1)]
    #[serde(default = "default_cores")]
    #[builder(default = 1)]
    pub cores: usize,

    /// Keep-alive connections per core. With the adaptive drive this is the SEED
    /// the controller starts from and searches around; pin it as a hard cap by
    /// also setting `adaptive = false`.
    #[setting(default = 25)]
    #[serde(default = "default_connections_per_core")]
    #[builder(default = 25)]
    pub connections_per_core: usize,

    /// Adapt the per-core in-flight count to the workload. **On by default** — a
    /// fixed connections-per-core is wrong for every workload but the one it was
    /// tuned on, and the adaptive drive is wrk-competitive while it auto-finds
    /// the crest and dodges the 3.8-7.5x mis-set loss.
    #[setting(default = true)]
    #[serde(default = "default_adaptive")]
    #[builder(default = true)]
    pub adaptive: bool,

    /// Run-wide pass/fail gates.
    #[setting(skip)]
    #[serde(rename = "threshold", default)]
    #[builder(default)]
    pub thresholds: Thresholds,

    /// The workloads. More than one is the point of the collapse.
    #[setting(skip)]
    #[serde(rename = "scenario", default)]
    #[builder(default)]
    pub scenarios: Vec<Scenario>,

    /// Opt-in arrival capture. `None` — the default — records nothing and costs
    /// nothing: the dump arm is simply absent from the measurement fan.
    #[setting(skip)]
    #[serde(rename = "dump", default)]
    pub dump: Option<Dump>,
}

/// One workload: a target, what to send it, and the stages to send it in.
///
/// `name` is not decoration — it is the label every metric this scenario records
/// is keyed by, which is what makes a multi-scenario report readable.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    /// Distinguishes this workload's metrics from its siblings'.
    #[serde(default = "unnamed")]
    pub name: String,

    /// This scenario's share of the connection pool, relative to its siblings.
    ///
    /// Weights 3 and 1 give the first scenario three quarters of the
    /// connections. It is a share of *concurrency*, not of arrivals: a stage
    /// that names a rate already says how many arrivals it wants, and a stage
    /// that names none wants as many as its connections can carry.
    #[serde(default = "one")]
    pub weight: u32,

    /// Target URL, e.g. `"http://127.0.0.1:8080/"`.
    pub url: String,

    #[serde(rename = "request", default)]
    pub payload: PayloadSpec,

    #[serde(rename = "stage", default)]
    pub stages: Vec<Stage>,
}

fn unnamed() -> String {
    "load".to_string()
}

fn one() -> u32 {
    1
}

/// What the scenario's `[request]` table describes: the workload to send, in the
/// vocabulary a person writing an HTTP load test types.
///
/// A *description* of a payload, never a payload envelope — the engine encodes
/// it to bytes once, at connect time, and the load loop never sees these fields.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadSpec {
    #[serde(default = "get")]
    pub method: String,
    #[serde(default = "root")]
    pub path: String,
    #[serde(default)]
    pub body: Option<String>,
    #[serde(default)]
    pub headers: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub query: std::collections::BTreeMap<String, String>,
}

fn get() -> String {
    "GET".to_string()
}

fn root() -> String {
    "/".to_string()
}

impl Default for PayloadSpec {
    /// `GET /` — the benchmark workload, and what an omitted `[request]` means.
    fn default() -> Self {
        Self {
            method: get(),
            path: root(),
            body: None,
            headers: std::collections::BTreeMap::new(),
            query: std::collections::BTreeMap::new(),
        }
    }
}

/// One phase of a scenario.
///
/// `rate` absent means **flat out** — the closed-loop throughput drive. Present
/// means paced open-loop at that arrival rate. One field, two modes that used to
/// be two config types.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    #[serde(rename = "rate", default, deserialize_with = "optional_rate")]
    pub rate_per_sec: Option<f64>,
    #[serde(deserialize_with = "duration", serialize_with = "as_secs_text")]
    pub duration: Duration,
    /// How the arrivals are spread across the rate. Ignored when no rate is
    /// named — flat out has no schedule to shape.
    #[serde(default)]
    pub arrival: Arrival,
}

/// The arrival process a rate is realised as.
///
/// `rate = "200/s"` says how MANY; this says WHEN. They are different questions
/// and rekt used to answer the second one silently: arrivals were a perfectly
/// even 5ms grid, which is the least realistic option in the set. Uniform
/// arrivals systematically under-stress a queue, because real traffic clusters
/// and clustering is what builds depth.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Arrival {
    /// Exactly `1/rate` apart. Deterministic, reproducible, and what a
    /// throughput bench wants when it is measuring a ceiling rather than
    /// modelling users.
    #[default]
    Even,
    /// Exponential inter-arrival gaps with mean `1/rate` — a Poisson process,
    /// the standard model for independent arrivals. Bursty by construction, and
    /// the honest choice when the question is "what does real traffic do to
    /// this", not "how fast can it go".
    ///
    /// `seed` pins the sequence: the same seed replays the same clustering, on
    /// any machine and any core count.
    Poisson {
        #[serde(default)]
        seed: u64,
    },
}

impl Arrival {
    /// Nanoseconds between arrival `index` and the one before it.
    ///
    /// Pure in `(self, mean_nanos, index)` — no state, no clock, no global RNG —
    /// so a run's arrival pattern is reproducible however the load is sharded
    /// across cores and whatever order things complete in. The *stream* that
    /// turns these gaps into arrivals in real time is `engine::Arrivals`.
    #[must_use]
    pub fn gap_nanos(&self, mean_nanos: u64, index: u64) -> u64 {
        match self {
            Arrival::Even => mean_nanos,
            Arrival::Poisson { seed } => exponential_nanos(mean_nanos, seed.wrapping_add(index)),
        }
    }
}

/// One exponential sample with the given mean, from a pure hash of `state`.
///
/// splitmix64 rather than a dependency: the requirement is determinism from a
/// seed, and four lines of well-known mixing gives that without pulling an rng
/// crate into a load generator's hot path.
fn exponential_nanos(mean_nanos: u64, state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;

    // uniform in (0, 1] — never 0, so ln is finite
    let uniform = ((z >> 11) as f64 + 1.0) / (2.0f64.powi(53) + 1.0);
    let gap = -(mean_nanos as f64) * uniform.ln();
    gap as u64
}

/// pass/fail gates. `failure_rate` is the share of arrivals that got NO reply
/// -- a 500 came back, so it is a reply and lives in its own bucket.
#[derive(Debug, Clone, PartialEq, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Thresholds {
    #[serde(default, deserialize_with = "optional_duration", serialize_with = "as_optional_ms")]
    pub p99: Option<Duration>,
    #[serde(rename = "error_rate", default, deserialize_with = "optional_percent", serialize_with = "as_optional_percent")]
    pub failure_rate: Option<f64>,
}

/// Opt-in capture of every arrival: what went out, and what came back.
///
/// Off by default and slow when on — the point is repeatability, not speed. A
/// dumped run replays: the same arrivals, at the same spacing, against a
/// controllable clock, with no target on the other end.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Dump {
    /// Where the event log goes.
    pub path: String,

    /// `bin` (zstd + postcard, the default) or `json` (one event per line).
    #[serde(default = "bin")]
    pub format: String,

    /// Events buffered before the drain worker has to keep up.
    ///
    /// The bound is the point: a load generator must never park on a slow disk,
    /// because the stall would silently lower the offered rate — the instrument
    /// changing what it measures. Overflow is dropped per `on_full` and counted
    /// in `record_dropped_total`, so a lossy dump says so instead of lying.
    #[serde(default = "default_dump_capacity")]
    pub capacity: usize,

    /// `drop_newest` (default), `drop_oldest`, or `fail_closed`.
    #[serde(default = "drop_newest")]
    pub on_full: String,

    /// Events coalesced into one durable block.
    #[serde(default = "default_dump_batch")]
    pub batch: usize,
}

fn bin() -> String {
    "bin".to_string()
}

fn drop_newest() -> String {
    "drop_newest".to_string()
}

fn default_dump_capacity() -> usize {
    16_384
}

fn default_dump_batch() -> usize {
    256
}

impl Validate for LoadPlan {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.cores == 0 {
            errors.push(ValidationMessage::new("cores", "must be >= 1"));
        }
        if self.connections_per_core == 0 {
            errors.push(ValidationMessage::new("connections_per_core", "must be >= 1"));
        }
        for (index, scenario) in self.scenarios.iter().enumerate() {
            if scenario.url.is_empty() {
                errors.push(ValidationMessage::new("scenario.url", format!("scenario {index} has an empty url")));
            }
            if scenario.weight == 0 {
                errors.push(ValidationMessage::new("scenario.weight", format!("scenario {index} has weight 0: remove it instead")));
            }
            if scenario.stages.is_empty() {
                errors.push(ValidationMessage::new("scenario.stage", format!("scenario {index} has no stages")));
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(conflaguration::Error::Validation { errors }) }
    }
}

impl LoadPlan {
    /// Connections each scenario gets, apportioned by weight.
    ///
    /// Every scenario gets at least one — a weight small enough to round to zero
    /// would otherwise silently drop a workload from the run.
    #[must_use]
    pub fn connection_shares(&self) -> Vec<usize> {
        let total: u32 = self
            .scenarios
            .iter()
            .map(|scenario| scenario.weight)
            .sum();
        if total == 0 {
            return vec![1; self.scenarios.len()];
        }
        self.scenarios
            .iter()
            .map(|scenario| {
                let share = self.connections_per_core * scenario.weight as usize / total as usize;
                share.max(1)
            })
            .collect()
    }
}

impl LoadPlan {
    /// Load from a TOML string.
    pub fn from_toml(text: &str) -> Result<Self, Error> {
        let plan: Self = toml::from_str(text)?;
        plan.validate()
            .map_err(|err| Error::Config(err.to_string()))?;
        Ok(plan)
    }

    /// Total wall clock the plan asks for, across every scenario and stage.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.scenarios
            .iter()
            .flat_map(|scenario| scenario.stages.iter())
            .map(|stage| stage.duration)
            .sum()
    }
}

// ── the units the file speaks ────────────────────────────────────────────────
//
// serde seams rather than a second struct hierarchy. Each is the existing parser
// plus a few lines of plumbing, and a bad unit fails at deserialize time with
// the file's own span reporting rather than in a hand-rolled second pass.

fn optional_rate<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
    match Option::<String>::deserialize(deserializer)? {
        Some(text) => parse_rate(&text)
            .map(Some)
            .map_err(serde::de::Error::custom),
        None => Ok(None),
    }
}

fn duration<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
    let text = String::deserialize(deserializer)?;
    parse_duration(&text).map_err(serde::de::Error::custom)
}

fn optional_duration<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<Duration>, D::Error> {
    match Option::<String>::deserialize(deserializer)? {
        Some(text) => parse_duration(&text)
            .map(Some)
            .map_err(serde::de::Error::custom),
        None => Ok(None),
    }
}

fn optional_percent<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
    match Option::<String>::deserialize(deserializer)? {
        Some(text) => parse_percent(&text)
            .map(Some)
            .map_err(serde::de::Error::custom),
        None => Ok(None),
    }
}

fn as_secs_text<S: serde::Serializer>(value: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("{}s", value.as_secs_f64()))
}

fn as_optional_ms<S: serde::Serializer>(value: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error> {
    match value {
        Some(duration) => serializer.serialize_some(&format!("{}ms", duration.as_secs_f64() * 1000.0)),
        None => serializer.serialize_none(),
    }
}

fn as_optional_percent<S: serde::Serializer>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error> {
    match value {
        Some(fraction) => serializer.serialize_some(&format!("{}%", fraction * 100.0)),
        None => serializer.serialize_none(),
    }
}

fn split_num_unit(s: &str) -> Result<(&str, &str), Error> {
    let idx = s
        .find(|c: char| c.is_ascii_alphabetic() || c == '%')
        .ok_or_else(|| Error::Config(format!("missing unit in {s:?}")))?;
    Ok((&s[..idx], &s[idx..]))
}

fn parse_duration(s: &str) -> Result<Duration, Error> {
    let s = s.trim();
    let (num, unit) = split_num_unit(s)?;
    let v: f64 = num
        .trim()
        .parse()
        .map_err(|_| Error::Config(format!("bad duration {s:?}")))?;
    let secs = match unit {
        "ms" => v / 1000.0,
        "s" => v,
        "m" => v * 60.0,
        "h" => v * 3600.0,
        other => return Err(Error::Config(format!("bad duration unit {other:?}"))),
    };
    Ok(Duration::from_secs_f64(secs))
}

fn parse_rate(s: &str) -> Result<f64, Error> {
    let s = s.trim();
    let (num, per) = s
        .split_once('/')
        .ok_or_else(|| Error::Config(format!("bad rate {s:?}, want N/s")))?;
    let n: f64 = num
        .trim()
        .parse()
        .map_err(|_| Error::Config(format!("bad rate {s:?}")))?;
    let secs = match per.trim() {
        "s" => 1.0,
        "m" => 60.0,
        "h" => 3600.0,
        other => return Err(Error::Config(format!("bad rate unit {other:?}"))),
    };
    Ok(n / secs)
}

fn parse_percent(s: &str) -> Result<f64, Error> {
    let v: f64 = s
        .trim()
        .trim_end_matches('%')
        .parse()
        .map_err(|_| Error::Config(format!("bad percent {s:?}")))?;
    Ok(v / 100.0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const TWO_SCENARIOS: &str = r#"
cores = 2
connections_per_core = 10

[threshold]
p99 = "250ms"
error_rate = "0.1%"

[[scenario]]
name = "browse"
url = "http://127.0.0.1:8080/"

[scenario.request]
method = "GET"
path = "/"

[[scenario.stage]]
rate = "100/s"
duration = "30s"

[[scenario]]
name = "checkout"
url = "http://127.0.0.1:9090/"

[scenario.request]
method = "POST"
path = "/checkout"
body = "cart=1"

[[scenario.stage]]
duration = "10s"
"#;

    #[test]
    fn a_file_carries_more_than_one_scenario() {
        let plan = LoadPlan::from_toml(TWO_SCENARIOS).expect("parses");

        assert_eq!(plan.scenarios.len(), 2);
        assert_eq!(plan.scenarios[0].name, "browse");
        assert_eq!(plan.scenarios[0].url, "http://127.0.0.1:8080/");
        assert_eq!(plan.scenarios[1].name, "checkout");
        assert_eq!(plan.scenarios[1].payload.method, "POST");
        assert_eq!(plan.scenarios[1].payload.body.as_deref(), Some("cart=1"));
    }

    #[test]
    fn an_absent_rate_means_flat_out() {
        // the unification: a closed-loop throughput bench IS a scenario whose
        // stage names no rate. It used to be a separate config type.
        let plan = LoadPlan::from_toml(TWO_SCENARIOS).expect("parses");

        assert_eq!(plan.scenarios[0].stages[0].rate_per_sec, Some(100.0), "paced open loop");
        assert_eq!(plan.scenarios[1].stages[0].rate_per_sec, None, "flat out");
    }

    #[test]
    fn machine_knobs_and_gates_live_at_the_plan() {
        let plan = LoadPlan::from_toml(TWO_SCENARIOS).expect("parses");

        assert_eq!(plan.cores, 2);
        assert_eq!(plan.connections_per_core, 10);
        assert!(plan.adaptive, "on by default");
        assert_eq!(plan.thresholds.p99, Some(Duration::from_millis(250)));
        assert_eq!(plan.thresholds.failure_rate, Some(0.001));
        assert_eq!(plan.duration(), Duration::from_secs(40), "30s + 10s across scenarios");
    }

    #[test]
    fn a_scenario_without_a_url_is_refused() {
        let text = r#"
[[scenario]]
name = "nowhere"
url = ""

[[scenario.stage]]
duration = "1s"
"#;
        let refusal = LoadPlan::from_toml(text)
            .expect_err("refused")
            .to_string();
        assert!(refusal.contains("url"), "expected a url complaint, got: {refusal}");
    }

    #[test]
    fn a_scenario_without_stages_is_refused() {
        let text = r#"
[[scenario]]
url = "http://127.0.0.1:8080/"
"#;
        let refusal = LoadPlan::from_toml(text)
            .expect_err("refused")
            .to_string();
        assert!(refusal.contains("stage"), "expected a stage complaint, got: {refusal}");
    }

    /// P4 parity, for the knobs that still carry an env surface: the
    /// conflaguration config and the fluent builder resolve identically. The
    /// workload moved to the file — `REKT_TARGET` is gone, because a target
    /// belongs to a scenario and a run can have several.
    #[test]
    fn builder_and_env_resolve_identically() {
        let via_builder = LoadPlan::builder()
            .cores(2)
            .connections_per_core(10)
            .adaptive(true)
            .build();

        temp_env::with_vars([("REKT_CORES", Some("2")), ("REKT_CONNECTIONS_PER_CORE", Some("10")), ("REKT_ADAPTIVE", Some("true"))], || {
            let via_env = LoadPlan::from_env().expect("env load");
            assert_eq!(via_env, via_builder, "typed env == fluent builder");
        });
    }
}
