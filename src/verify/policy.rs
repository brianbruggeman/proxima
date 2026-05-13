//! Policy DSL parser. TOML format with three top-level sections —
//! `[shared]`, `[static]`, `[replay]` — feeding a single typed
//! `Policy` struct. The static walker consumes the `[static]`
//! section + relevant `[shared]` facts; the replay walker consumes
//! `[replay]` + the same `[shared]`.
//!
//! Custom predicate grammar is `applies_to` + `filter` + `require`
//! over four ops (`host_in`, `name_matches`, `field_present`,
//! `field_equals`). No AND / OR / NOT in v1 — wait for a second
//! consumer to justify the language extension.

use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use super::report::Level;
use crate::error::ProximaError;

/// A complete policy document. Loaded from a single TOML file; each
/// walker reads the section it needs.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Policy {
    #[serde(default)]
    pub shared: SharedSection,

    #[serde(default, rename = "static")]
    pub static_section: StaticSection,

    #[serde(default)]
    pub replay: ReplaySection,
}

impl Policy {
    /// Parse a policy from TOML text.
    pub fn parse_str(text: &str) -> Result<Self, ProximaError> {
        toml::from_str(text).map_err(|err| ProximaError::Config(format!("parse policy: {err}")))
    }

    /// Load a policy from a TOML file path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ProximaError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(ProximaError::Io)?;
        Self::parse_str(&text)
    }
}

/// Facts both walkers reference — names of external upstreams,
/// optional public-route allowlist for the static walker.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SharedSection {
    #[serde(default)]
    pub external_upstreams: Vec<String>,

    #[serde(default)]
    pub public_routes: Vec<String>,
}

/// Static-walker rules.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct StaticSection {
    /// Names of built-in invariants to run. Empty means run the
    /// default set (see `BUILT_IN_INVARIANTS` in the walker).
    #[serde(default)]
    pub invariants: Vec<String>,

    /// Custom predicates over the spec graph.
    #[serde(default)]
    pub custom: Vec<CustomRule>,
}

/// Replay-walker rules.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ReplaySection {
    /// Allowlist of upstream names the replay may call. Any other
    /// upstream call ⇒ FAIL `unauthorized_upstream_call`.
    #[serde(default)]
    pub allowed_upstreams: Vec<String>,

    /// Pipes whose responses must match the recording byte-for-byte.
    /// Drift ⇒ FAIL `byte_drift`.
    #[serde(default)]
    pub byte_identical_pipes: Vec<String>,

    /// Sources of nondeterminism declared safe (clock, uuid). A
    /// nondeterministic pipe whose name is not on this list ⇒ FAIL
    /// `nondeclared_nondeterminism`.
    #[serde(default)]
    pub nondeterministic_sources: Vec<String>,

    /// Pipes whose events must come from the recording, not from
    /// inference. Inferred events for these pipes ⇒ FAIL
    /// `inferred_not_recorded`.
    #[serde(default)]
    pub must_derive_from_record: Vec<String>,
}

/// A single custom predicate. v1 grammar: pick a target class
/// (`applies_to`), narrow with `filter`, assert with `require`.
/// `severity` selects the report level emitted when the rule fails.
/// A passing rule always emits `Level::Pass` regardless of severity.
/// Valid severity values are `"warn"` or `"fail"` — `"pass"` is
/// rejected at parse time (a rule that produces Pass on failure is
/// meaningless).
#[derive(Debug, Clone, Deserialize)]
pub struct CustomRule {
    pub name: String,

    pub applies_to: TargetClass,

    #[serde(default)]
    pub filter: PredicateTable,

    pub require: PredicateTable,

    #[serde(
        default = "default_severity",
        deserialize_with = "deserialize_violation_level"
    )]
    pub severity: Level,
}

fn default_severity() -> Level {
    Level::Fail
}

fn deserialize_violation_level<'de, D>(deserializer: D) -> Result<Level, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let level = Level::deserialize(deserializer)?;
    match level {
        Level::Warn | Level::Fail => Ok(level),
        Level::Pass => Err(D::Error::custom(
            "custom-rule severity must be 'warn' or 'fail', not 'pass'",
        )),
    }
}

/// Which kind of spec entry the custom rule walks over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TargetClass {
    Upstream,
    Pipe,
    Middleware,
    Route,
}

/// One predicate clause — a map of field name → op-value pair.
/// Multiple clauses in the same table are conjunctive (`AND`).
/// Use `field_value` for the four supported ops directly; `serde`
/// flattens the table so authors write `{ host = "x.com" }` or
/// `{ host_in = ["a","b"] }` directly.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct PredicateTable {
    pub clauses: std::collections::BTreeMap<String, Value>,
}

impl PredicateTable {
    /// Resolve the four supported ops against a target's JSON value.
    /// Returns `Ok(true)` if every clause holds, `Ok(false)` if any
    /// clause fails, or `Err` for malformed ops.
    pub fn evaluate(&self, target: &Value) -> Result<bool, ProximaError> {
        for (key, expected) in &self.clauses {
            if !evaluate_clause(key, expected, target)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn evaluate_clause(key: &str, expected: &Value, target: &Value) -> Result<bool, ProximaError> {
    // `host_in = [...]` — actual field is `host`, op is `_in`
    if let Some(field) = key.strip_suffix("_in") {
        let candidates = expected.as_array().ok_or_else(|| {
            ProximaError::Config(format!("predicate '{key}': expected array of values"))
        })?;
        let actual = match target.get(field) {
            Some(value) => value,
            None => return Ok(false),
        };
        return Ok(candidates.iter().any(|cand| cand == actual));
    }

    // `name_matches = "regex"` — actual field is `name`, op is `_matches`
    if let Some(field) = key.strip_suffix("_matches") {
        let pattern = expected.as_str().ok_or_else(|| {
            ProximaError::Config(format!("predicate '{key}': expected string regex"))
        })?;
        let actual = match target.get(field).and_then(|v| v.as_str()) {
            Some(text) => text,
            None => return Ok(false),
        };
        let regex = regex::Regex::new(pattern).map_err(|err| {
            ProximaError::Config(format!(
                "predicate '{key}': invalid regex '{pattern}': {err}"
            ))
        })?;
        return Ok(regex.is_match(actual));
    }

    // `field_present = "x"` — assert `target.x` exists and is non-null
    if key == "field_present" {
        let field = expected.as_str().ok_or_else(|| {
            ProximaError::Config("predicate 'field_present': expected field name string".into())
        })?;
        return Ok(target.get(field).is_some_and(|value| !value.is_null()));
    }

    // Default: equality. `host = "x.com"` means `target.host == "x.com"`.
    let actual = match target.get(key) {
        Some(value) => value,
        None => return Ok(false),
    };
    Ok(actual == expected)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]
    use super::*;

    #[test]
    fn empty_policy_parses_clean() {
        let policy = Policy::parse_str("").expect("parse");
        assert!(policy.shared.external_upstreams.is_empty());
        assert!(policy.static_section.invariants.is_empty());
        assert!(policy.replay.allowed_upstreams.is_empty());
    }

    #[test]
    fn shared_section_parses_external_upstreams() {
        let text = r#"
            [shared]
            external_upstreams = ["claims_api", "echo"]
        "#;
        let policy = Policy::parse_str(text).expect("parse");
        assert_eq!(
            policy.shared.external_upstreams,
            vec!["claims_api".to_string(), "echo".to_string()]
        );
    }

    #[test]
    fn static_section_parses_invariants_and_custom_rule() {
        let text = r#"
            [static]
            invariants = ["no_cycles", "all_upstreams_have_timeouts"]
            [[static.custom]]
            name = "external_hosts_allowlisted"
            applies_to = "upstream"
            filter = { type = "http", external = true }
            require = { host_in = ["api.x.com", "api.y.com"] }
            severity = "fail"
        "#;
        let policy = Policy::parse_str(text).expect("parse");
        assert_eq!(policy.static_section.invariants.len(), 2);
        assert_eq!(policy.static_section.custom.len(), 1);
        let rule = &policy.static_section.custom[0];
        assert_eq!(rule.name, "external_hosts_allowlisted");
        assert_eq!(rule.applies_to, TargetClass::Upstream);
        assert_eq!(rule.severity, Level::Fail);
    }

    #[test]
    fn replay_section_parses_all_lists() {
        let text = r#"
            [replay]
            allowed_upstreams = ["claims_api"]
            nondeterministic_sources = ["clock", "uuid_v4"]
            byte_identical_pipes = ["redact_claims_v2"]
            must_derive_from_record = ["fetch", "build"]
        "#;
        let policy = Policy::parse_str(text).expect("parse");
        assert_eq!(policy.replay.allowed_upstreams, vec!["claims_api"]);
        assert_eq!(policy.replay.nondeterministic_sources.len(), 2);
        assert_eq!(policy.replay.byte_identical_pipes, vec!["redact_claims_v2"]);
        assert_eq!(policy.replay.must_derive_from_record.len(), 2);
    }

    #[test]
    fn predicate_equals_matches_field_value() {
        let table = PredicateTable {
            clauses: [("host".into(), Value::String("api.x.com".into()))]
                .into_iter()
                .collect(),
        };
        let target = serde_json::json!({ "host": "api.x.com" });
        assert!(table.evaluate(&target).expect("evaluate"));
        let target_other = serde_json::json!({ "host": "api.y.com" });
        assert!(!table.evaluate(&target_other).expect("evaluate"));
    }

    #[test]
    fn predicate_host_in_matches_any() {
        let table = PredicateTable {
            clauses: [(
                "host_in".into(),
                serde_json::json!(["api.x.com", "api.y.com"]),
            )]
            .into_iter()
            .collect(),
        };
        let target = serde_json::json!({ "host": "api.y.com" });
        assert!(table.evaluate(&target).expect("evaluate"));
        let target_other = serde_json::json!({ "host": "api.z.com" });
        assert!(!table.evaluate(&target_other).expect("evaluate"));
    }

    #[test]
    fn predicate_name_matches_regex() {
        let table = PredicateTable {
            clauses: [("name_matches".into(), Value::String("^claims_".into()))]
                .into_iter()
                .collect(),
        };
        let target = serde_json::json!({ "name": "claims_router" });
        assert!(table.evaluate(&target).expect("evaluate"));
        let target_other = serde_json::json!({ "name": "router_claims" });
        assert!(!table.evaluate(&target_other).expect("evaluate"));
    }

    #[test]
    fn predicate_field_present_checks_existence() {
        let table = PredicateTable {
            clauses: [("field_present".into(), Value::String("timeout".into()))]
                .into_iter()
                .collect(),
        };
        let with_timeout = serde_json::json!({ "timeout": 5 });
        assert!(table.evaluate(&with_timeout).expect("evaluate"));
        let without = serde_json::json!({});
        assert!(!table.evaluate(&without).expect("evaluate"));
    }

    #[test]
    fn predicate_conjunction_requires_all_clauses() {
        let table = PredicateTable {
            clauses: [
                ("host".into(), Value::String("api.x.com".into())),
                ("external".into(), Value::Bool(true)),
            ]
            .into_iter()
            .collect(),
        };
        let both = serde_json::json!({ "host": "api.x.com", "external": true });
        let only_host = serde_json::json!({ "host": "api.x.com", "external": false });
        assert!(table.evaluate(&both).expect("evaluate"));
        assert!(!table.evaluate(&only_host).expect("evaluate"));
    }
}
