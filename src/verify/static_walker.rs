//! Static spec-graph invariants. Walks a parsed proxima spec (as a
//! `serde_json::Value`) and emits a report entry per enabled
//! invariant.
//!
//! Two built-in invariants — `no_cycles` and
//! `all_upstreams_have_timeouts` — plus the custom-predicate runner.
//! Built-ins stay focused on structural correctness and universal
//! config sanity; domain rules ("external upstreams require auth",
//! "writeback after success", "public routes covered by auth") are
//! the user's policy, expressed as custom predicates via the
//! `[[static.custom]]` section in `policy.toml`. See
//! [`crate::verify::policy`] for the grammar.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde_json::{Map, Value};

use super::policy::{CustomRule, Policy, TargetClass};
use super::report::{Level, Report, ReportEntry};

/// Names of built-in invariants users can list in `policy.static.invariants`.
/// When the list is empty in the policy file, the walker runs every
/// invariant marked `default = true`.
const BUILT_INS: &[BuiltIn] = &[
    BuiltIn {
        name: "no_cycles",
        default: true,
    },
    BuiltIn {
        name: "all_upstreams_have_timeouts",
        default: true,
    },
];

struct BuiltIn {
    name: &'static str,
    default: bool,
}

/// Run the static walker against a parsed spec value (as produced by
/// [`crate::load::load_value_from_path`] or
/// [`crate::settings::ProximaSettings::from_path`] re-serialized).
///
/// Accepts both the **named-map** form (`[pipes.api] chain = [...]`)
/// and the **array-of-tables** form (`[[pipe]] name = "api"`).
/// The walker normalizes the array-of-tables shape into the named-map
/// shape before running invariants; this is the canonical form
/// `App::load_full` uses internally.
pub fn verify_static(spec: &Value, policy: &Policy) -> Report {
    let normalized = normalize_spec(spec);
    let mut report = Report::new();

    let enabled = resolve_enabled_invariants(&policy.static_section.invariants);
    for name in enabled {
        match name {
            "no_cycles" => run_no_cycles(&normalized, &mut report),
            "all_upstreams_have_timeouts" => run_timeouts(&normalized, &mut report),
            other => {
                report.push(ReportEntry::warn(
                    "policy.unknown_invariant",
                    format!("invariant '{other}' is not implemented in v1; skipping"),
                ));
            }
        }
    }

    for rule in &policy.static_section.custom {
        run_custom_rule(&normalized, rule, &mut report);
    }

    report
}

/// Normalize the spec to the named-map shape the walker expects.
/// Converts `[[pipe]] name = "x" ...` arrays into `pipes: { "x":
/// {...} }` maps; same for `upstream`, `middleware`, `listen`.
///
/// Both forms are valid proxima config — `App::load_full` reads
/// the array form (canonical for multi-listener `proxima serve
/// --config <path>` deployments); `ProximaSettings::from_path`
/// reads the map form (used for typed builder round-trip and the
/// daemon control plane). The walker handles both.
fn normalize_spec(spec: &Value) -> Value {
    let mut out = spec.clone();
    let Some(map) = out.as_object_mut() else {
        return out;
    };
    normalize_section(map, "pipe", "pipes");
    normalize_section(map, "upstream", "upstreams");
    normalize_section(map, "middleware", "middlewares");
    normalize_section(map, "listen", "listeners");
    out
}

fn normalize_section(map: &mut Map<String, Value>, source_key: &str, target_key: &str) {
    let entries = match map.remove(source_key) {
        Some(Value::Array(arr)) => arr,
        Some(other) => {
            // not an array — put it back, leave the target alone
            map.insert(source_key.to_string(), other);
            return;
        }
        None => return,
    };
    let target = map
        .entry(target_key.to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let Value::Object(target_obj) = target else {
        return;
    };
    for entry in entries {
        let Some(name) = entry.get("name").and_then(Value::as_str).map(String::from) else {
            continue;
        };
        let mut clean = entry.clone();
        if let Some(obj) = clean.as_object_mut() {
            obj.remove("name");
        }
        target_obj.insert(name, clean);
    }
}

fn resolve_enabled_invariants(requested: &[String]) -> Vec<&'static str> {
    if requested.is_empty() {
        return BUILT_INS
            .iter()
            .filter(|b| b.default)
            .map(|b| b.name)
            .collect();
    }
    requested
        .iter()
        .map(|name| {
            BUILT_INS
                .iter()
                .find(|b| b.name == name)
                .map(|b| b.name)
                .unwrap_or("__unknown__")
        })
        .collect()
}

/// `no_cycles` — build the directed graph of named entries
/// (upstreams ∪ middlewares ∪ pipes) and edges where a `chain`
/// field references another named entry. A back edge ⇒ FAIL with
/// the cycle path.
fn run_no_cycles(spec: &Value, report: &mut Report) {
    let nodes = collect_named_entries(spec);
    if nodes.is_empty() {
        report.push(ReportEntry::pass("no_cycles"));
        return;
    }

    let edges = collect_chain_edges(spec, &nodes);

    match find_cycle(&nodes, &edges) {
        None => report.push(ReportEntry::pass("no_cycles")),
        Some(path) => {
            let rendered = path.join(" → ");
            report.push(ReportEntry::fail("no_cycles", format!("cycle: {rendered}")));
        }
    }
}

fn collect_named_entries(spec: &Value) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for section in ["upstreams", "middlewares", "pipes"] {
        if let Some(Value::Object(map)) = spec.get(section) {
            for key in map.keys() {
                names.insert(key.clone());
            }
        }
    }
    names
}

fn collect_chain_edges(spec: &Value, nodes: &BTreeSet<String>) -> BTreeMap<String, Vec<String>> {
    let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for section in ["pipes", "middlewares"] {
        let Some(Value::Object(map)) = spec.get(section) else {
            continue;
        };
        for (name, entry) in map {
            let Some(Value::Array(chain)) = entry.get("chain") else {
                continue;
            };
            let targets: Vec<String> = chain
                .iter()
                .filter_map(|item| item.as_str().map(String::from))
                .filter(|target| nodes.contains(target))
                .collect();
            if !targets.is_empty() {
                edges.insert(name.clone(), targets);
            }
        }
    }
    edges
}

fn find_cycle(
    nodes: &BTreeSet<String>,
    edges: &BTreeMap<String, Vec<String>>,
) -> Option<Vec<String>> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut on_stack: HashSet<String> = HashSet::new();
    for start in nodes {
        if visited.contains(start) {
            continue;
        }
        let mut path: Vec<String> = Vec::new();
        if let Some(cycle) = dfs_for_cycle(start, edges, &mut visited, &mut on_stack, &mut path) {
            return Some(cycle);
        }
    }
    None
}

fn dfs_for_cycle(
    node: &String,
    edges: &BTreeMap<String, Vec<String>>,
    visited: &mut HashSet<String>,
    on_stack: &mut HashSet<String>,
    path: &mut Vec<String>,
) -> Option<Vec<String>> {
    visited.insert(node.clone());
    on_stack.insert(node.clone());
    path.push(node.clone());

    if let Some(targets) = edges.get(node) {
        for target in targets {
            if on_stack.contains(target) {
                let start = path.iter().position(|name| name == target).unwrap_or(0);
                let mut cycle = path[start..].to_vec();
                cycle.push(target.clone());
                return Some(cycle);
            }
            if !visited.contains(target)
                && let Some(found) = dfs_for_cycle(target, edges, visited, on_stack, path)
            {
                return Some(found);
            }
        }
    }

    path.pop();
    on_stack.remove(node);
    None
}

/// `all_upstreams_have_timeouts` — every http-class upstream must
/// declare a `timeout`. Walks both **top-level** upstreams (the
/// `upstreams.<name>` map) and **inline** upstreams nested inside
/// pipe entries (`pipes.<name>.upstreams = [...]`, `pipes.<name>.http
/// = "..."`). Missing ⇒ WARN with location detail.
fn run_timeouts(spec: &Value, report: &mut Report) {
    let mut missing: Vec<String> = Vec::new();

    if let Some(Value::Object(upstreams)) = spec.get("upstreams") {
        for (name, entry) in upstreams {
            if upstream_needs_timeout(entry) && !has_timeout(entry) {
                missing.push(format!("upstreams.{name}"));
            }
        }
    }

    if let Some(Value::Object(pipes)) = spec.get("pipes") {
        for (pipe_name, pipe_entry) in pipes {
            // pipe with inline http sugar at root: pipes.<name>.http
            if pipe_entry.get("http").is_some() && !has_timeout(pipe_entry) {
                missing.push(format!("pipes.{pipe_name} (inline http)"));
            }
            // pipe with explicit upstreams array
            if let Some(Value::Array(inline)) = pipe_entry.get("upstreams") {
                for (index, inline_upstream) in inline.iter().enumerate() {
                    if !upstream_needs_timeout(inline_upstream) {
                        continue;
                    }
                    if has_timeout(inline_upstream) {
                        continue;
                    }
                    let label = inline_upstream
                        .get("name")
                        .and_then(Value::as_str)
                        .map(String::from)
                        .unwrap_or_else(|| index.to_string());
                    missing.push(format!("pipes.{pipe_name}.upstreams[{label}]"));
                }
            }
        }
    }

    if missing.is_empty() {
        report.push(ReportEntry::pass("all_upstreams_have_timeouts"));
    } else {
        report.push(ReportEntry::warn(
            "all_upstreams_have_timeouts",
            format!("missing timeout on: {}", missing.join(", ")),
        ));
    }
}

fn upstream_needs_timeout(entry: &Value) -> bool {
    let kind = entry
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    is_http_upstream(kind, entry)
}

fn has_timeout(entry: &Value) -> bool {
    entry.get("timeout").is_some_and(|value| !value.is_null())
}

fn is_http_upstream(kind: &str, entry: &Value) -> bool {
    // declared via `type = "http"`; also accept legacy short-hand
    // `http = "url"` which proxima sugar uses heavily.
    kind == "http" || entry.get("http").is_some()
}

fn run_custom_rule(spec: &Value, rule: &CustomRule, report: &mut Report) {
    let section = match rule.applies_to {
        TargetClass::Upstream => "upstreams",
        TargetClass::Middleware => "middlewares",
        TargetClass::Pipe => "pipes",
        TargetClass::Route => {
            report.push(ReportEntry::warn(
                format!("custom.{}", rule.name),
                "applies_to = route is not implemented in v1".to_string(),
            ));
            return;
        }
    };

    let Some(Value::Object(entries)) = spec.get(section) else {
        report.push(ReportEntry::pass(format!("custom.{}", rule.name)));
        return;
    };

    let mut failures: Vec<String> = Vec::new();
    for (name, entry) in entries {
        let filter_holds = match rule.filter.evaluate(entry) {
            Ok(value) => value,
            Err(err) => {
                report.push(ReportEntry::fail(
                    format!("custom.{}", rule.name),
                    format!("filter evaluation error on '{name}': {err}"),
                ));
                continue;
            }
        };
        if !filter_holds {
            continue;
        }
        let require_holds = match rule.require.evaluate(entry) {
            Ok(value) => value,
            Err(err) => {
                report.push(ReportEntry::fail(
                    format!("custom.{}", rule.name),
                    format!("require evaluation error on '{name}': {err}"),
                ));
                continue;
            }
        };
        if !require_holds {
            failures.push(name.clone());
        }
    }

    if failures.is_empty() {
        report.push(ReportEntry::pass(format!("custom.{}", rule.name)));
        return;
    }

    let detail = format!("violated on: {}", failures.join(", "));
    let entry = match rule.severity {
        Level::Fail => ReportEntry::fail(format!("custom.{}", rule.name), detail),
        Level::Warn => ReportEntry::warn(format!("custom.{}", rule.name), detail),
        Level::Pass => unreachable!("custom-rule severity is parse-validated to Warn|Fail"),
    };
    report.push(entry);
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
    use crate::verify::policy::Policy;
    use serde_json::json;

    fn run(spec: Value, policy: &str) -> Report {
        let policy = Policy::parse_str(policy).expect("parse policy");
        verify_static(&spec, &policy)
    }

    #[test]
    fn empty_spec_passes_no_cycles() {
        let report = run(json!({}), "");
        let names: Vec<&str> = report.entries.iter().map(|e| e.rule.as_str()).collect();
        assert!(names.contains(&"no_cycles"));
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn array_of_tables_form_normalizes_to_named_map() {
        // The real proxima.toml format: `[[pipe]] name = "api" ...`
        let spec = json!({
            "pipe": [
                { "name": "api", "chain": ["b"] },
                { "name": "b", "chain": ["api"] },
            ]
        });
        let report = run(spec, "");
        let fails: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Fail))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(
            fails.contains(&"no_cycles"),
            "cycle in [[pipe]] form must FAIL"
        );
    }

    #[test]
    fn inline_http_upstream_without_timeout_warns() {
        // 05-todos style: pipe with inline `http = "..."` sugar at root,
        // no timeout.
        let spec = json!({
            "pipe": [
                { "name": "proxy", "http": "https://backend.x" },
            ]
        });
        let report = run(spec, "");
        let warns: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Warn))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(warns.contains(&"all_upstreams_have_timeouts"));
    }

    #[test]
    fn nested_upstreams_array_walked_for_timeouts() {
        // nginx-multi style: pipe with `upstreams = [{...}]` array.
        let spec = json!({
            "pipe": [
                {
                    "name": "lb",
                    "upstreams": [
                        { "name": "be1", "type": "http", "url": "https://x", "timeout": "5s" },
                        { "name": "be2", "type": "http", "url": "https://y" },
                    ]
                }
            ]
        });
        let report = run(spec, "");
        let warn_entry = report
            .entries
            .iter()
            .find(|entry| entry.rule == "all_upstreams_have_timeouts")
            .expect("rule present");
        assert!(
            matches!(warn_entry.level, super::super::report::Level::Warn),
            "should warn on be2"
        );
        assert!(
            warn_entry.detail.contains("be2"),
            "got: {}",
            warn_entry.detail
        );
        assert!(
            !warn_entry.detail.contains("be1"),
            "be1 has timeout, should not appear"
        );
    }

    #[test]
    fn array_of_tables_with_all_timeouts_passes() {
        let spec = json!({
            "pipe": [
                {
                    "name": "lb",
                    "upstreams": [
                        { "name": "be1", "type": "http", "url": "https://x", "timeout": "5s" },
                    ]
                }
            ]
        });
        let report = run(spec, "");
        assert_eq!(report.warn_count(), 0);
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn linear_chain_passes_no_cycles() {
        let spec = json!({
            "pipes": {
                "api": { "chain": ["auth", "backend"] },
            },
            "middlewares": { "auth": { "type": "bearer_auth" } },
            "upstreams": { "backend": { "type": "http", "url": "x", "timeout": 5 } },
        });
        let report = run(spec, "");
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn cyclic_pipe_chain_fails_no_cycles() {
        let spec = json!({
            "pipes": {
                "a": { "chain": ["b"] },
                "b": { "chain": ["a"] },
            },
        });
        let report = run(spec, "");
        let fails: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Fail))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(fails.contains(&"no_cycles"));
    }

    #[test]
    fn upstreams_without_timeout_warn() {
        let spec = json!({
            "upstreams": {
                "good": { "type": "http", "url": "x", "timeout": 3 },
                "bad":  { "type": "http", "url": "y" },
            },
        });
        let report = run(spec, "");
        let warns: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Warn))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(warns.contains(&"all_upstreams_have_timeouts"));
    }

    #[test]
    fn timeouts_pass_when_all_set() {
        let spec = json!({
            "upstreams": {
                "good": { "type": "http", "url": "x", "timeout": 3 },
            },
        });
        let report = run(spec, "");
        assert_eq!(report.warn_count(), 0);
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn unknown_invariant_warns_but_does_not_fail() {
        let spec = json!({});
        let policy = r#"
            [static]
            invariants = ["no_cycles", "totally_made_up"]
        "#;
        let report = run(spec, policy);
        assert_eq!(report.fail_count(), 0);
        assert_eq!(report.warn_count(), 1);
    }

    #[test]
    fn custom_rule_fails_when_filter_matches_and_require_fails() {
        let spec = json!({
            "upstreams": {
                "claims": { "type": "http", "url": "https://claims.x.com", "host": "api.evil.com" },
            },
        });
        let policy = r#"
            [[static.custom]]
            name = "hosts_allowlisted"
            applies_to = "upstream"
            filter  = { type = "http" }
            require = { host_in = ["api.good.com"] }
            severity = "fail"
        "#;
        let report = run(spec, policy);
        let fails: Vec<&str> = report
            .entries
            .iter()
            .filter(|entry| matches!(entry.level, super::super::report::Level::Fail))
            .map(|entry| entry.rule.as_str())
            .collect();
        assert!(fails.contains(&"custom.hosts_allowlisted"));
    }

    #[test]
    fn custom_rule_passes_when_no_target_matches_filter() {
        let spec = json!({
            "upstreams": {
                "kv": { "type": "kv", "kv": "cache" },
            },
        });
        let policy = r#"
            [[static.custom]]
            name = "hosts_allowlisted"
            applies_to = "upstream"
            filter  = { type = "http" }
            require = { host_in = ["api.good.com"] }
            severity = "fail"
        "#;
        let report = run(spec, policy);
        assert_eq!(report.fail_count(), 0);
    }
}
