//! Verification reports — a flat list of entries with one of three
//! severity levels. The walker fills a `Report`; the CLI emits it as
//! grep-friendly text (default) or JSON (machine consumer).

use std::fmt;

use serde::{Deserialize, Serialize};

/// Severity of a single verification finding.
///
/// `Pass` documents the rule fired and held. `Warn` documents a
/// finding that does not gate CI by default (still reported, still
/// surfaced in `--format json`, but the CLI exits 0 unless the user
/// passed `--strict`). `Fail` always gates CI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Pass,
    Warn,
    Fail,
}

impl Level {
    fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// One finding. `rule` names the built-in invariant or custom
/// predicate that produced this entry; `detail` is the human-readable
/// explanation (cycle path, missing field, mismatched name).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportEntry {
    pub level: Level,
    pub rule: String,
    pub detail: String,
}

impl ReportEntry {
    pub fn pass(rule: impl Into<String>) -> Self {
        Self {
            level: Level::Pass,
            rule: rule.into(),
            detail: String::new(),
        }
    }

    pub fn warn(rule: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            level: Level::Warn,
            rule: rule.into(),
            detail: detail.into(),
        }
    }

    pub fn fail(rule: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            level: Level::Fail,
            rule: rule.into(),
            detail: detail.into(),
        }
    }
}

/// The full report for one `proxima verify` (or `proxima replay
/// --verify`) invocation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Report {
    pub entries: Vec<ReportEntry>,
}

impl Report {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, entry: ReportEntry) {
        self.entries.push(entry);
    }

    /// Count entries at a given level — useful for the CLI's
    /// exit-code logic.
    #[must_use]
    pub fn count(&self, level: Level) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.level == level)
            .count()
    }

    /// Convenience: total `Pass` count.
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.count(Level::Pass)
    }

    /// Convenience: total `Warn` count.
    #[must_use]
    pub fn warn_count(&self) -> usize {
        self.count(Level::Warn)
    }

    /// Convenience: total `Fail` count.
    #[must_use]
    pub fn fail_count(&self) -> usize {
        self.count(Level::Fail)
    }

    /// Grep-friendly text format. One line per entry:
    /// `LEVEL rule_name optional detail`.
    #[must_use]
    pub fn emit_text(&self) -> String {
        let mut buffer = String::new();
        for entry in &self.entries {
            if entry.detail.is_empty() {
                buffer.push_str(&format!("{} {}\n", entry.level, entry.rule));
            } else {
                buffer.push_str(&format!(
                    "{} {} {}\n",
                    entry.level, entry.rule, entry.detail
                ));
            }
        }
        buffer
    }

    /// JSON format. Inline counts at the document root so consumers
    /// can branch without iterating `entries`.
    pub fn emit_json(&self) -> Result<String, serde_json::Error> {
        let doc = serde_json::json!({
            "entries": self.entries,
            "pass": self.pass_count(),
            "warn": self.warn_count(),
            "fail": self.fail_count(),
        });
        serde_json::to_string_pretty(&doc)
    }
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
    fn empty_report_counts_are_zero() {
        let report = Report::new();
        assert_eq!(report.pass_count(), 0);
        assert_eq!(report.warn_count(), 0);
        assert_eq!(report.fail_count(), 0);
    }

    #[test]
    fn counts_partition_by_level() {
        let mut report = Report::new();
        report.push(ReportEntry::pass("no_cycles"));
        report.push(ReportEntry::pass("all_upstreams_have_timeouts"));
        report.push(ReportEntry::warn(
            "route_auth_coverage",
            "/debug/* unauthed",
        ));
        report.push(ReportEntry::fail(
            "auth_dominates_external_upstreams",
            "claims_api unauthed",
        ));
        assert_eq!(report.pass_count(), 2);
        assert_eq!(report.warn_count(), 1);
        assert_eq!(report.fail_count(), 1);
    }

    #[test]
    fn emit_text_matches_grep_friendly_format() {
        let mut report = Report::new();
        report.push(ReportEntry::pass("no_cycles"));
        report.push(ReportEntry::fail(
            "byte_drift",
            "claims_api differs at offset 42",
        ));
        let text = report.emit_text();
        assert_eq!(
            text,
            "PASS no_cycles\nFAIL byte_drift claims_api differs at offset 42\n"
        );
    }

    #[test]
    fn emit_json_has_entries_and_inline_counts() {
        let mut report = Report::new();
        report.push(ReportEntry::pass("no_cycles"));
        let json = report.emit_json().expect("emit_json");
        assert!(json.contains("\"no_cycles\""));
        assert!(json.contains("\"pass\": 1"));
        assert!(json.contains("\"fail\": 0"));
    }
}
