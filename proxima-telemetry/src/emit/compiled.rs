//! The compiled filter table — the typed core every front-end lowers to.
//!
//! A flat `Level` floor, a `RUST_LOG` string, and a TOML config all produce a
//! [`CompiledEmit`]: a longest-prefix-wins rule table over module-path targets
//! plus a named default. All the prefix sorting happens once at build time; the
//! per-record [`CompiledEmit::decide`] is a linear scan of the (small,
//! cache-resident) rule set with a byte-compare each, then a `Coord` decide —
//! no allocation. Mirrors the resolution the existing
//! [`crate::pipes::FilterByLevelPipe`] does for the flat case, generalized to
//! targets + hierarchy.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::emit::{Coord, Decision, EmitThreshold};

/// How a rule's target prefix matches a record's module path.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum MatchMode {
    /// Raw `str::starts_with` — `foo` matches `foobar`. This is exactly what
    /// `tracing-subscriber`'s `EnvFilter` does, so the `RUST_LOG` front-end uses
    /// it for byte-for-byte precedence parity (P14).
    Raw,
    /// `::`-boundary-aware — `foo` matches `foo` and `foo::bar` but NOT `foobar`.
    /// The native default: less surprising than tracing's raw prefix.
    #[default]
    Boundary,
}

/// One operator rule before compilation: a target prefix and its threshold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmitRule {
    /// Module-path prefix, e.g. `"proxima::h2"`.
    pub target: String,
    /// What to keep at/under this target.
    pub threshold: EmitThreshold,
}

impl EmitRule {
    /// Build a rule from a target and threshold.
    #[must_use]
    pub fn new(target: impl Into<String>, threshold: EmitThreshold) -> Self {
        Self {
            target: target.into(),
            threshold,
        }
    }
}

/// A compiled, immutable filter table. Built once; queried per record.
#[derive(Clone, Debug)]
pub struct CompiledEmit {
    /// Sorted by target length descending, so the FIRST match is the longest
    /// (most-specific) prefix — tracing's precedence, made explicit.
    rules: Box<[EmitRule]>,
    /// The named default — applies when no rule prefix matches. There is always
    /// a default, so resolution never falls through to nothing (fixes
    /// `RUST_LOG`'s "is there even a default?" ambiguity).
    default: EmitThreshold,
    mode: MatchMode,
}

impl CompiledEmit {
    /// Compile a rule set + named default into the queryable table. Sorts rules
    /// longest-target-first (ties broken lexicographically for determinism), so
    /// `decide` can take the first match.
    #[must_use]
    pub fn build(default: EmitThreshold, mut rules: Vec<EmitRule>, mode: MatchMode) -> Self {
        rules.sort_by(|left, right| {
            right
                .target
                .len()
                .cmp(&left.target.len())
                .then_with(|| left.target.cmp(&right.target))
        });
        Self {
            rules: rules.into_boxed_slice(),
            default,
            mode,
        }
    }

    /// The named default threshold (applies where no rule matches).
    #[must_use]
    pub const fn default_threshold(&self) -> EmitThreshold {
        self.default
    }

    /// Resolve one record: longest matching target prefix wins, else the default.
    /// Hot path — no allocation, a byte-compare per rule then a `Coord` decide.
    #[inline]
    #[must_use]
    pub fn decide(&self, target: &str, coord: Coord) -> Decision {
        for rule in &self.rules {
            if self.matches(&rule.target, target) {
                return rule.threshold.decide(coord);
            }
        }
        self.default.decide(coord)
    }

    #[inline]
    fn matches(&self, prefix: &str, target: &str) -> bool {
        match self.mode {
            MatchMode::Raw => target.starts_with(prefix),
            MatchMode::Boundary => boundary_prefix(target, prefix),
        }
    }
}

/// `::`-boundary prefix: `target == prefix`, or `target` continues with `::`
/// right after `prefix`.
#[inline]
fn boundary_prefix(target: &str, prefix: &str) -> bool {
    if !target.starts_with(prefix) {
        return false;
    }
    let rest = &target[prefix.len()..];
    rest.is_empty() || rest.starts_with("::")
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

    use alloc::vec;

    use rstest::rstest;

    use super::{CompiledEmit, EmitRule, MatchMode};
    use crate::emit::{Coord, Decision, EmitThreshold};
    use crate::level::Level;

    // the worked example from the design: real proxima module paths, a named
    // default, longest-prefix-wins, plus a verbose subtree on the deepest target.
    fn fixture() -> CompiledEmit {
        CompiledEmit::build(
            EmitThreshold::at(Coord::from(Level::WARN)), // default
            vec![
                EmitRule::new("proxima", EmitThreshold::at(Coord::from(Level::INFO))),
                EmitRule::new("proxima::h2", EmitThreshold::at(Coord::from(Level::DEBUG))),
                EmitRule::new(
                    "proxima::h2::hpack",
                    EmitThreshold::verbose(Coord::from(Level::WARN), Coord::parse("9.2").unwrap()),
                ),
            ],
            MatchMode::Boundary,
        )
    }

    #[rstest]
    // hpack matches its own rule (floor warn), info is below -> drop...
    #[case::hpack_info_dropped(
        "proxima::h2::hpack::evict",
        Coord::from(Level::INFO),
        Decision::Drop
    )]
    // ...but the 9.2 verbose subtree is kept even though band 9 < warn.
    #[case::hpack_verbose_kept("proxima::h2::hpack::evict", Coord::parse("9.2.4").unwrap(), Decision::Keep)]
    // h2 (not hpack) matches proxima::h2 = debug; debug is kept.
    #[case::h2_debug_kept("proxima::h2::frame", Coord::from(Level::DEBUG), Decision::Keep)]
    // shallow proxima matches proxima = info.
    #[case::proxima_info_kept("proxima::quic", Coord::from(Level::INFO), Decision::Keep)]
    // no rule matches -> default warn keeps error.
    #[case::default_error_kept("downstream::store", Coord::from(Level::ERROR), Decision::Keep)]
    // no rule matches -> default warn drops debug.
    #[case::default_debug_dropped("downstream::store", Coord::from(Level::DEBUG), Decision::Drop)]
    fn longest_prefix_wins(#[case] target: &str, #[case] coord: Coord, #[case] want: Decision) {
        assert_eq!(fixture().decide(target, coord), want);
    }

    // boundary mode rejects a non-boundary prefix; raw mode (tracing parity)
    // accepts it.
    #[test]
    fn match_mode_boundary_vs_raw() {
        let rules = vec![EmitRule::new(
            "foo",
            EmitThreshold::at(Coord::from(Level::DEBUG)),
        )];
        let boundary = CompiledEmit::build(
            EmitThreshold::at(Coord::from(Level::ERROR)),
            rules.clone(),
            MatchMode::Boundary,
        );
        let raw = CompiledEmit::build(
            EmitThreshold::at(Coord::from(Level::ERROR)),
            rules,
            MatchMode::Raw,
        );
        // "foobar" is NOT under "foo" by boundary, so it hits the error default.
        assert_eq!(
            boundary.decide("foobar", Coord::from(Level::DEBUG)),
            Decision::Drop
        );
        // tracing's raw prefix DOES match "foobar" against "foo".
        assert_eq!(
            raw.decide("foobar", Coord::from(Level::DEBUG)),
            Decision::Keep
        );
        // both agree on the real boundary case.
        assert_eq!(
            boundary.decide("foo::bar", Coord::from(Level::DEBUG)),
            Decision::Keep
        );
        assert_eq!(
            raw.decide("foo::bar", Coord::from(Level::DEBUG)),
            Decision::Keep
        );
    }
}
