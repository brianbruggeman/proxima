use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::ProximaError;
use crate::telemetry::Labels;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioPipeSpec {
    pub name: String,
    #[serde(flatten)]
    pub spec: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadSpec {
    pub target_pipe: String,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default = "default_path")]
    pub path: String,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub query: BTreeMap<String, String>,
    #[serde(default, with = "serde_body_text")]
    pub body: Option<bytes::Bytes>,
    /// closed-loop total request count. `0` (default) means use open-loop fields instead.
    #[serde(default)]
    pub requests: usize,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// open-loop target requests-per-second; paired with `duration` or `profile`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_rps: Option<u64>,
    /// open-loop run duration; required when `target_rps` is set without `profile`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<DurationSpec>,
    /// open-loop ramp/ladder profile. each step holds `rate` for `duration`.
    /// non-empty `profile` overrides `(target_rps, duration)` as the schedule.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub profile: Vec<ProfileStep>,
}

fn default_method() -> String {
    "GET".into()
}

fn default_path() -> String {
    "/".into()
}

fn default_concurrency() -> usize {
    1
}

mod serde_body_text {
    use bytes::Bytes;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(value: &Option<Bytes>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(bytes) => match std::str::from_utf8(bytes) {
                Ok(text) => serializer.serialize_some(text),
                Err(_) => Err(<S::Error as serde::ser::Error>::custom(
                    "scenario body bytes are not valid utf-8; use the programmatic api for binary",
                )),
            },
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Bytes>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw: Option<String> = serde::Deserialize::deserialize(deserializer)?;
        raw.map(|text| {
            if text.is_empty() {
                Err(<D::Error as serde::de::Error>::custom(
                    "scenario body must be non-empty",
                ))
            } else {
                Ok(Bytes::from(text.into_bytes()))
            }
        })
        .transpose()
    }
}

impl WorkloadSpec {
    #[must_use]
    pub fn new(target_pipe: impl Into<String>, requests: usize) -> Self {
        Self {
            target_pipe: target_pipe.into(),
            method: "GET".into(),
            path: "/".into(),
            headers: BTreeMap::new(),
            query: BTreeMap::new(),
            body: None,
            requests,
            concurrency: 1,
            target_rps: None,
            duration: None,
            profile: Vec::new(),
        }
    }

    /// open-loop variant: drive `target_pipe` at `rps` for `duration`.
    /// `requests` defaults to 0; supply concurrency separately if needed.
    #[must_use]
    pub fn new_open_loop(
        target_pipe: impl Into<String>,
        target_rps: u64,
        duration: DurationSpec,
    ) -> Self {
        Self {
            target_pipe: target_pipe.into(),
            method: "GET".into(),
            path: "/".into(),
            headers: BTreeMap::new(),
            query: BTreeMap::new(),
            body: None,
            requests: 0,
            concurrency: 1,
            target_rps: Some(target_rps),
            duration: Some(duration),
            profile: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.method = method.into();
        self
    }

    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = path.into();
        self
    }

    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    #[must_use]
    pub fn with_body(mut self, body: impl Into<bytes::Bytes>) -> Self {
        self.body = Some(body.into());
        self
    }

    #[must_use]
    pub fn with_target_rps(mut self, rps: u64, duration: DurationSpec) -> Self {
        self.target_rps = Some(rps);
        self.duration = Some(duration);
        self
    }

    #[must_use]
    pub fn with_profile(mut self, profile: Vec<ProfileStep>) -> Self {
        self.profile = profile;
        self
    }

    /// classify the workload as open-loop or closed-loop. errors if both
    /// closed-loop (`requests > 0`) and open-loop (`target_rps`) signals are set,
    /// or if neither is.
    pub fn mode(&self) -> Result<WorkloadMode, ProximaError> {
        let has_closed = self.requests > 0;
        let has_open = self.target_rps.is_some() || !self.profile.is_empty();
        match (has_closed, has_open) {
            (true, false) => Ok(WorkloadMode::ClosedLoop),
            (false, true) => {
                if self.profile.is_empty() && self.duration.is_none() {
                    return Err(ProximaError::Config(
                        "open-loop workload missing `duration` (or `profile`)".into(),
                    ));
                }
                Ok(WorkloadMode::OpenLoop)
            }
            (true, true) => Err(ProximaError::Config(
                "workload sets both `requests` (closed-loop) and `target_rps`/`profile` \
                 (open-loop); pick one"
                    .into(),
            )),
            (false, false) => Err(ProximaError::Config(
                "workload sets neither `requests` (closed-loop) nor `target_rps`/`profile` \
                 (open-loop)"
                    .into(),
            )),
        }
    }
}

/// duration in whole seconds. deserializes from either a string suffixed
/// with `s` / `m` / `h` (e.g. `"30s"`, `"5m"`) or a table `{ secs = N }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DurationSpec {
    pub secs: u64,
}

impl DurationSpec {
    #[must_use]
    pub const fn from_secs(secs: u64) -> Self {
        Self { secs }
    }

    #[must_use]
    pub const fn as_duration(self) -> std::time::Duration {
        std::time::Duration::from_secs(self.secs)
    }
}

impl<'de> serde::Deserialize<'de> for DurationSpec {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct DurationSpecVisitor;
        impl<'de> serde::de::Visitor<'de> for DurationSpecVisitor {
            type Value = DurationSpec;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a duration string like \"30s\" or a table { secs = N }")
            }

            fn visit_str<EnvError: serde::de::Error>(
                self,
                value: &str,
            ) -> Result<DurationSpec, EnvError> {
                parse_duration_str(value).map_err(EnvError::custom)
            }

            fn visit_map<MapAccess: serde::de::MapAccess<'de>>(
                self,
                map: MapAccess,
            ) -> Result<DurationSpec, MapAccess::Error> {
                #[derive(Deserialize)]
                struct Inner {
                    secs: u64,
                }
                let inner = Inner::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
                Ok(DurationSpec { secs: inner.secs })
            }

            fn visit_u64<EnvError: serde::de::Error>(
                self,
                value: u64,
            ) -> Result<DurationSpec, EnvError> {
                Ok(DurationSpec { secs: value })
            }
        }
        deserializer.deserialize_any(DurationSpecVisitor)
    }
}

fn parse_duration_str(value: &str) -> Result<DurationSpec, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("duration string is empty".into());
    }
    let split_at = trimmed
        .find(|character: char| character.is_alphabetic())
        .unwrap_or(trimmed.len());
    let (number_part, suffix_part) = trimmed.split_at(split_at);
    let number: u64 = number_part
        .trim()
        .parse()
        .map_err(|err| format!("invalid duration number `{number_part}`: {err}"))?;
    let secs = match suffix_part.trim() {
        "" | "s" => number,
        "m" => number
            .checked_mul(60)
            .ok_or_else(|| format!("duration `{trimmed}` overflows u64 seconds"))?,
        "h" => number
            .checked_mul(3600)
            .ok_or_else(|| format!("duration `{trimmed}` overflows u64 seconds"))?,
        other => return Err(format!("unsupported duration suffix `{other}`; use s/m/h")),
    };
    Ok(DurationSpec { secs })
}

/// single step in an open-loop ramp/ladder profile. holds `rate` rps for `duration`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileStep {
    pub rate: u64,
    pub duration: DurationSpec,
}

/// classification of a workload at dispatch time. not serialized — derived from
/// `WorkloadSpec::mode()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadMode {
    ClosedLoop,
    OpenLoop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareOp {
    Eq,
    Ge,
    Le,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Expectation {
    Counter {
        metric: String,
        #[serde(default)]
        labels: BTreeMap<String, String>,
        op: CompareOp,
        expected: u64,
    },
    HistogramP99LeMs {
        metric: String,
        #[serde(default)]
        labels: BTreeMap<String, String>,
        max_ms: f64,
    },
    SuccessRateGe {
        ratio: f64,
    },
    /// CEL expression. Bindings: `success_rate` (f64 in [0,1]),
    /// `completed` / `successes` / `failures` (u64), `metric.<name>`
    /// (u64). example: `"success_rate >= 0.99"`
    Cel {
        expression: String,
    },
    /// asserts the behavior of a `diff` middleware Pipe during the run by
    /// inspecting `proxima.diff.{identical,divergent}_total` counters.
    /// requires the `diff` middleware to have been part of the pipe graph.
    Diff {
        /// require at least one identical-result observation, and (when
        /// `true`) zero divergent observations.
        #[serde(default = "default_diff_identical")]
        identical: bool,
        /// when `Some`, asserts the recorded `DiffReport.first_diff_offset`
        /// for any divergent observation does not exceed this. currently
        /// reserved — counters do not yet carry offset; field accepted for
        /// forward compat.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_first_diff_offset: Option<usize>,
    },
}

const fn default_diff_identical() -> bool {
    true
}

impl Expectation {
    #[must_use]
    pub fn counter_with_labels(
        metric: impl Into<String>,
        labels: &Labels,
        op: CompareOp,
        expected: u64,
    ) -> Self {
        Self::Counter {
            metric: metric.into(),
            labels: label_map(labels),
            op,
            expected,
        }
    }

    #[must_use]
    pub fn histogram_p99_with_labels(
        metric: impl Into<String>,
        labels: &Labels,
        max_ms: f64,
    ) -> Self {
        Self::HistogramP99LeMs {
            metric: metric.into(),
            labels: label_map(labels),
            max_ms,
        }
    }
}

fn label_map(labels: &Labels) -> BTreeMap<String, String> {
    labels
        .entries()
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OrchestrationMode {
    /// in-process tasks under the orchestrator's tokio runtime. fast,
    /// parent reads server-side metrics directly. default.
    #[default]
    InProcess,
    /// each pipe is a child `proxima serve` process. parent drives
    /// traffic via HTTP, captures client-side metrics. use to validate
    /// cross-process behaviour (resource limits, signals, supervisor).
    Isolated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    #[serde(default)]
    pub mode: OrchestrationMode,
    #[serde(default, rename = "pipe")]
    pub pipes: Vec<ScenarioPipeSpec>,
    pub workload: WorkloadSpec,
    #[serde(default, rename = "expect")]
    pub expectations: Vec<Expectation>,
}

impl Scenario {
    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self, ProximaError> {
        let raw = std::fs::read_to_string(path.as_ref())
            .map_err(|err| ProximaError::Config(format!("read scenario.toml: {err}")))?;
        Self::from_toml_str(&raw)
    }

    pub fn from_toml_str(raw: &str) -> Result<Self, ProximaError> {
        toml::from_str(raw).map_err(|err| ProximaError::Config(format!("scenario.toml: {err}")))
    }
}

impl Scenario {
    #[must_use]
    pub fn new_programmatic(workload: WorkloadSpec) -> Self {
        Self {
            mode: OrchestrationMode::InProcess,
            pipes: Vec::new(),
            workload,
            expectations: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_mode(mut self, mode: OrchestrationMode) -> Self {
        self.mode = mode;
        self
    }

    #[must_use]
    pub fn with_pipe(mut self, name: impl Into<String>, spec: serde_json::Value) -> Self {
        self.pipes.push(ScenarioPipeSpec {
            name: name.into(),
            spec,
        });
        self
    }

    #[must_use]
    pub fn with_expectation(mut self, expectation: Expectation) -> Self {
        self.expectations.push(expectation);
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_scenario_from_toml() {
        let raw = r#"
[workload]
target_pipe = "echo"
requests = 50
concurrency = 4

[[pipe]]
name = "echo"
[pipe.synth]
status = 200
body = "hello"

[[expect]]
kind = "success_rate_ge"
ratio = 1.0
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse scenario.toml");
        assert_eq!(scenario.workload.target_pipe, "echo");
        assert_eq!(scenario.workload.requests, 50);
        assert_eq!(scenario.workload.concurrency, 4);
        assert_eq!(scenario.pipes.len(), 1);
        assert_eq!(scenario.pipes[0].name, "echo");
        assert_eq!(scenario.expectations.len(), 1);
        assert!(matches!(
            scenario.expectations[0],
            Expectation::SuccessRateGe { ratio } if (ratio - 1.0).abs() < 1e-9
        ));
    }

    #[test]
    fn parses_counter_expectation_with_labels() {
        let raw = r#"
[workload]
target_pipe = "x"
requests = 1

[[expect]]
kind = "counter"
metric = "proxima.cache.hits_total"
op = "ge"
expected = 10

[expect.labels]
target = "cache"
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse");
        match &scenario.expectations[0] {
            Expectation::Counter {
                metric,
                labels,
                op,
                expected,
            } => {
                assert_eq!(metric, "proxima.cache.hits_total");
                assert_eq!(labels.get("target"), Some(&"cache".to_string()));
                assert_eq!(*op, CompareOp::Ge);
                assert_eq!(*expected, 10);
            }
            other => panic!("expected counter expectation, got {other:?}"),
        }
    }

    #[test]
    fn parses_cel_expectation_block() {
        let raw = r#"
[workload]
target_pipe = "x"
requests = 1

[[expect]]
kind = "cel"
expression = "success_rate >= 0.95"
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse");
        match &scenario.expectations[0] {
            Expectation::Cel { expression } => {
                assert_eq!(expression, "success_rate >= 0.95");
            }
            other => panic!("expected cel expectation, got {other:?}"),
        }
    }

    #[test]
    fn missing_workload_block_returns_typed_error() {
        let raw = r#"
[[pipe]]
name = "echo"
[pipe.synth]
status = 200
body = "hi"
"#;
        let outcome = Scenario::from_toml_str(raw);
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn parses_open_loop_workload_with_string_duration() {
        let raw = r#"
[workload]
target_pipe = "echo"
target_rps = 200
duration = "5s"
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse open-loop");
        assert_eq!(scenario.workload.target_pipe, "echo");
        assert_eq!(scenario.workload.requests, 0);
        assert_eq!(scenario.workload.target_rps, Some(200));
        assert_eq!(scenario.workload.duration, Some(DurationSpec::from_secs(5)));
        let mode = scenario.workload.mode().expect("mode resolves");
        assert_eq!(mode, WorkloadMode::OpenLoop);
    }

    #[test]
    fn parses_open_loop_workload_with_minute_suffix() {
        let raw = r#"
[workload]
target_pipe = "x"
target_rps = 100
duration = "2m"
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse minutes");
        assert_eq!(
            scenario.workload.duration,
            Some(DurationSpec::from_secs(120))
        );
    }

    #[test]
    fn parses_open_loop_workload_with_table_duration() {
        let raw = r#"
[workload]
target_pipe = "x"
target_rps = 100

[workload.duration]
secs = 30
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse table duration");
        assert_eq!(
            scenario.workload.duration,
            Some(DurationSpec::from_secs(30))
        );
    }

    #[test]
    fn parses_workload_profile_steps() {
        let raw = r#"
[workload]
target_pipe = "x"
target_rps = 100

[[workload.profile]]
rate = 50
duration = "1s"

[[workload.profile]]
rate = 200
duration = "5s"
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse profile");
        assert_eq!(scenario.workload.profile.len(), 2);
        assert_eq!(scenario.workload.profile[0].rate, 50);
        assert_eq!(scenario.workload.profile[0].duration.secs, 1);
        assert_eq!(scenario.workload.profile[1].rate, 200);
        assert_eq!(scenario.workload.profile[1].duration.secs, 5);
        // open-loop classification still valid with profile present
        assert_eq!(
            scenario.workload.mode().expect("mode"),
            WorkloadMode::OpenLoop
        );
    }

    #[test]
    fn mode_rejects_both_closed_and_open_loop_fields() {
        let raw = r#"
[workload]
target_pipe = "x"
requests = 50
target_rps = 100
duration = "5s"
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse");
        let outcome = scenario.workload.mode();
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn mode_rejects_open_loop_missing_duration_and_profile() {
        let raw = r#"
[workload]
target_pipe = "x"
target_rps = 100
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse");
        let outcome = scenario.workload.mode();
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn mode_rejects_workload_with_neither_signal() {
        let raw = r#"
[workload]
target_pipe = "x"
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse");
        let outcome = scenario.workload.mode();
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn closed_loop_legacy_scenario_still_parses_and_classifies() {
        let raw = r#"
[workload]
target_pipe = "echo"
requests = 50
concurrency = 4
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse legacy");
        assert_eq!(scenario.workload.requests, 50);
        assert_eq!(
            scenario.workload.mode().expect("mode"),
            WorkloadMode::ClosedLoop
        );
    }

    #[test]
    fn parses_diff_expectation_block() {
        let raw = r#"
[workload]
target_pipe = "x"
requests = 1

[[expect]]
kind = "diff"
identical = true
"#;
        let scenario = Scenario::from_toml_str(raw).expect("parse diff");
        match &scenario.expectations[0] {
            Expectation::Diff {
                identical,
                max_first_diff_offset,
            } => {
                assert!(*identical);
                assert!(max_first_diff_offset.is_none());
            }
            other => panic!("expected diff expectation, got {other:?}"),
        }
    }

    #[test]
    fn rejects_invalid_duration_suffix() {
        let raw = r#"
[workload]
target_pipe = "x"
target_rps = 100
duration = "5d"
"#;
        let outcome = Scenario::from_toml_str(raw);
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }
}
