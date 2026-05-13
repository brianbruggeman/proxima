//! `RUST_LOG` / `tracing-subscriber` `EnvFilter` compatibility front-end.
//!
//! Parses the familiar comma-separated `target=level` directive grammar into a
//! [`CompiledEmit`] so existing `RUST_LOG` strings keep working unchanged — the
//! "match the old API" half. It lowers tracing's semantics into proxima's
//! severity world: a tracing directive `mycrate=debug` ("debug and more severe")
//! maps to a proxima floor at `DEBUG`, which keeps the same set (debug, info,
//! warn, error, fatal; drops trace). Precedence is tracing's: most-specific =
//! longest target by raw `str::starts_with` ([`MatchMode::Raw`]), single winner,
//! declaration order irrelevant.
//!
//! Grounded against vendored `tracing-subscriber-0.3.23`
//! (`filter/env/directive.rs`). The exact edge-case parity (empty-level → ERROR,
//! invalid-level-drops-directive, the `ERROR` global default) is locked by a
//! parity test against the real `EnvFilter` in the bench/parity phase.

use alloc::vec::Vec;

use crate::emit::{CompiledEmit, Coord, EmitRule, EmitThreshold, MatchMode};
use crate::level::Level;

/// Parser for `RUST_LOG`-style directive strings.
pub struct EnvFilter;

impl EnvFilter {
    /// Parse a `RUST_LOG`-style string into a compiled filter. Lenient like
    /// tracing: malformed directives are dropped, never fatal.
    ///
    /// Default semantics match `tracing_subscriber::EnvFilter::new` exactly
    /// (verified, `filter/env/builder.rs`): the `ERROR` default directive is
    /// added ONLY when no directives parse (the empty-string case). A non-empty
    /// directive set with no bare-level (global) directive drops every unmatched
    /// callsite (OFF). A bare-level directive sets the global default.
    #[must_use]
    pub fn parse(input: &str) -> CompiledEmit {
        let mut default_override: Option<EmitThreshold> = None;
        let mut rules: Vec<EmitRule> = Vec::new();

        for raw in input.split(',') {
            let directive = raw.trim();
            if directive.is_empty() {
                continue;
            }
            match directive.rsplit_once('=') {
                Some((target_part, level_part)) => {
                    // `target[span{..}]=level` — strip the span/field part; the
                    // emit filter is target+level (callsite span-scope filtering
                    // is not a post-emit concern).
                    let target = target_part.split('[').next().unwrap_or("").trim();
                    match level_threshold(level_part) {
                        Some(threshold) if target.is_empty() => default_override = Some(threshold),
                        Some(threshold) => rules.push(EmitRule::new(target, threshold)),
                        None => {} // `target=BADLEVEL` drops the directive (tracing parity)
                    }
                }
                None => {
                    // a bare token: a level (the global default) or a target
                    // (enable everything for it, i.e. floor TRACE).
                    let target = directive.split('[').next().unwrap_or("").trim();
                    match level_threshold(directive) {
                        Some(threshold) => default_override = Some(threshold),
                        None if target.is_empty() => {}
                        None => rules.push(EmitRule::new(
                            target,
                            EmitThreshold::at(Coord::from(Level::TRACE)),
                        )),
                    }
                }
            }
        }

        let default = default_override.unwrap_or_else(|| {
            if rules.is_empty() {
                EmitThreshold::at(Coord::from(Level::ERROR)) // empty string → ERROR
            } else {
                EmitThreshold::at(Coord::from_severity(u8::MAX)) // directives present, no global → OFF
            }
        });

        CompiledEmit::build(default, rules, MatchMode::Raw)
    }

    /// Build a filter from an env var's value (missing/empty → `""` → ERROR,
    /// matching tracing's empty-string default). Makes `RUST_LOG` "just work".
    #[cfg(feature = "std")]
    #[must_use]
    pub fn from_env(var: &str) -> CompiledEmit {
        Self::parse(&std::env::var(var).unwrap_or_default())
    }

    /// Build a filter from the conventional `RUST_LOG` env var — the drop-in
    /// equivalent of `tracing_subscriber::EnvFilter::from_default_env`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn from_default_env() -> CompiledEmit {
        Self::from_env("RUST_LOG")
    }
}

/// Map a tracing level token to a proxima floor threshold. Accepts the names
/// (case-insensitive) and the numbers `0..=5` tracing uses, plus the
/// empty-string → `ERROR` and `off` → drop-all quirks. `None` = not a level.
fn level_threshold(token: &str) -> Option<EmitThreshold> {
    let token = token.trim();
    // `off`/`0` is a floor above every real band, so nothing is kept.
    let floor = if token.eq_ignore_ascii_case("off") || token == "0" {
        Coord::from_severity(u8::MAX)
    } else if token.eq_ignore_ascii_case("error") || token == "1" {
        Coord::from(Level::ERROR)
    } else if token.eq_ignore_ascii_case("warn") || token == "2" {
        Coord::from(Level::WARN)
    } else if token.eq_ignore_ascii_case("info") || token == "3" {
        Coord::from(Level::INFO)
    } else if token.eq_ignore_ascii_case("debug") || token == "4" {
        Coord::from(Level::DEBUG)
    } else if token.eq_ignore_ascii_case("trace") || token == "5" {
        Coord::from(Level::TRACE)
    } else if token.is_empty() {
        Coord::from(Level::ERROR) // tracing: `target=` → ERROR
    } else {
        return None;
    };
    Some(EmitThreshold::at(floor))
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

    use super::EnvFilter;
    use crate::emit::{Coord, Decision};
    use crate::level::Level;

    // the canonical RUST_LOG shape: per-target floors; unmatched callsites are
    // OFF (no global directive), matching tracing's EnvFilter::new.
    #[test]
    fn targeted_directives_drop_unmatched() {
        let filter = EnvFilter::parse("proxima::h2=debug,proxima=info");
        // h2 subtree keeps debug-and-above
        assert_eq!(
            filter.decide("proxima::h2::frame", Coord::from(Level::DEBUG)),
            Decision::Keep
        );
        assert_eq!(
            filter.decide("proxima::h2::frame", Coord::from(Level::TRACE)),
            Decision::Drop
        );
        // proxima (not h2) keeps info-and-above; longest-prefix wins (h2 vs proxima)
        assert_eq!(
            filter.decide("proxima::quic", Coord::from(Level::INFO)),
            Decision::Keep
        );
        assert_eq!(
            filter.decide("proxima::quic", Coord::from(Level::DEBUG)),
            Decision::Drop
        );
        // unmatched -> OFF: dropped even at ERROR (the ERROR default is added
        // only for the empty string, see `empty_filter_defaults_to_error`).
        assert_eq!(
            filter.decide("downstream::store", Coord::from(Level::ERROR)),
            Decision::Drop
        );
    }

    // an empty filter keeps ERROR-and-above (tracing adds the ERROR default
    // directive only when no directives parse).
    #[test]
    fn empty_filter_defaults_to_error() {
        let filter = EnvFilter::parse("");
        assert_eq!(
            filter.decide("any", Coord::from(Level::ERROR)),
            Decision::Keep
        );
        assert_eq!(
            filter.decide("any", Coord::from(Level::WARN)),
            Decision::Drop
        );
    }

    // bare level = global default; bare target = enable-all (TRACE).
    #[test]
    fn bare_level_and_bare_target() {
        let filter = EnvFilter::parse("debug,noisy_crate");
        assert_eq!(
            filter.decide("anything", Coord::from(Level::DEBUG)),
            Decision::Keep
        ); // default debug
        assert_eq!(
            filter.decide("anything", Coord::from(Level::TRACE)),
            Decision::Drop
        );
        assert_eq!(
            filter.decide("noisy_crate", Coord::from(Level::TRACE)),
            Decision::Keep
        ); // enable-all
    }

    // off disables a target entirely; numbers and case are accepted.
    #[test]
    fn off_numbers_and_case() {
        let filter = EnvFilter::parse("spammy=off,FOO=4,bar=1");
        assert_eq!(
            filter.decide("spammy", Coord::from(Level::FATAL)),
            Decision::Drop
        ); // off = drop all
        assert_eq!(
            filter.decide("FOO", Coord::from(Level::DEBUG)),
            Decision::Keep
        ); // 4 = debug
        assert_eq!(
            filter.decide("FOO", Coord::from(Level::TRACE)),
            Decision::Drop
        );
        assert_eq!(
            filter.decide("bar", Coord::from(Level::ERROR)),
            Decision::Keep
        ); // 1 = error
        assert_eq!(
            filter.decide("bar", Coord::from(Level::WARN)),
            Decision::Drop
        );
    }

    // longest-prefix wins regardless of directive order (tracing precedence).
    #[test]
    fn longest_prefix_wins_order_independent() {
        let filter = EnvFilter::parse("a::b::c=trace,a=warn");
        assert_eq!(
            filter.decide("a::b::c::d", Coord::from(Level::TRACE)),
            Decision::Keep
        ); // longest
        assert_eq!(
            filter.decide("a::other", Coord::from(Level::TRACE)),
            Decision::Drop
        ); // a=warn
        assert_eq!(
            filter.decide("a::other", Coord::from(Level::WARN)),
            Decision::Keep
        );
    }

    // a malformed level drops just that directive; the rest survive.
    #[test]
    fn invalid_level_drops_only_that_directive() {
        let filter = EnvFilter::parse("good=info,bad=bogus");
        // `bad`'s directive was dropped; with a valid `good` rule and no global
        // directive, unmatched -> OFF -> dropped even at ERROR.
        assert_eq!(
            filter.decide("bad", Coord::from(Level::INFO)),
            Decision::Drop
        );
        assert_eq!(
            filter.decide("bad", Coord::from(Level::ERROR)),
            Decision::Drop
        );
        // `good` survived
        assert_eq!(
            filter.decide("good", Coord::from(Level::INFO)),
            Decision::Keep
        );
    }

    // the RUST_LOG env var "just works" — set it, get the filter.
    #[test]
    fn from_default_env_reads_rust_log() {
        temp_env::with_var("RUST_LOG", Some("proxima::h2=debug"), || {
            let filter = EnvFilter::from_default_env();
            assert_eq!(
                filter.decide("proxima::h2::frame", Coord::from(Level::DEBUG)),
                Decision::Keep
            );
            assert_eq!(
                filter.decide("proxima::h2::frame", Coord::from(Level::TRACE)),
                Decision::Drop
            );
        });
    }
}
