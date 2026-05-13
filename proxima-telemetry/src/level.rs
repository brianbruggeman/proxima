use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

use crate::error::Error;

// built-in name table: (severity, canonical_name)
// opt-sweep target: replace FromStr custom-name lookup with a fixed-cap interner
// so that `Level::from_str("audit")` works for any custom registered at compile time.
const BUILT_INS: &[(u8, &str)] = &[
    (1, "trace"),
    (5, "debug"),
    (9, "info"),
    (13, "warn"),
    (17, "error"),
    (21, "fatal"),
];

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Level {
    severity: u8,
    name: &'static str,
}

impl Level {
    pub const TRACE: Level = Level {
        severity: 1,
        name: "trace",
    };
    pub const DEBUG: Level = Level {
        severity: 5,
        name: "debug",
    };
    pub const INFO: Level = Level {
        severity: 9,
        name: "info",
    };
    pub const WARN: Level = Level {
        severity: 13,
        name: "warn",
    };
    pub const ERROR: Level = Level {
        severity: 17,
        name: "error",
    };
    pub const FATAL: Level = Level {
        severity: 21,
        name: "fatal",
    };

    pub const fn custom(name: &'static str, severity: u8) -> Self {
        Self { severity, name }
    }

    pub const fn severity(self) -> u8 {
        self.severity
    }

    pub const fn name(self) -> &'static str {
        self.name
    }
}

impl PartialOrd for Level {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Level {
    fn cmp(&self, other: &Self) -> Ordering {
        self.severity.cmp(&other.severity)
    }
}

impl fmt::Display for Level {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name)
    }
}

impl FromStr for Level {
    type Err = Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        // case-insensitive match against built-ins
        for (severity, name) in BUILT_INS {
            if input.eq_ignore_ascii_case(name) {
                return Ok(Level {
                    severity: *severity,
                    name,
                });
            }
        }

        // numeric input: map to closest built-in name (the range covering that severity)
        // opt-sweep tweak: extend this branch to consult the fixed-cap interner for
        // custom names registered via Level::custom(name, severity) at call-site.
        if let Ok(numeric) = input.parse::<u8>() {
            let name = severity_to_builtin_name(numeric);
            return Ok(Level {
                severity: numeric,
                name,
            });
        }

        Err(Error::InvalidInput)
    }
}

// maps a numeric severity to the canonical name of the range it falls in
const fn severity_to_builtin_name(severity: u8) -> &'static str {
    match severity {
        1..=4 => "trace",
        5..=8 => "debug",
        9..=12 => "info",
        13..=16 => "warn",
        17..=20 => "error",
        21..=24 => "fatal",
        _ => "custom",
    }
}

// serialize as the canonical name ("info", "trace", ...) so a config file reads
// `floor = "info"`; deserialize routes through `FromStr` (built-in names +
// numeric severity). A custom-named level does not round-trip through config —
// `FromStr` knows only the built-ins — which is the existing `FromStr` contract.
impl serde::Serialize for Level {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name)
    }
}

impl<'de> serde::Deserialize<'de> for Level {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = alloc::string::String::deserialize(deserializer)?;
        name.parse()
            .map_err(|_| serde::de::Error::custom("expected a level name (trace..fatal) or severity"))
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

    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;
    use rstest::rstest;

    use super::Level;
    use crate::error::Error;

    // 1. built-in consts have expected (severity, name) pairs; ordering holds
    #[rstest]
    #[case::trace(Level::TRACE, 1, "trace")]
    #[case::debug(Level::DEBUG, 5, "debug")]
    #[case::info(Level::INFO, 9, "info")]
    #[case::warn(Level::WARN, 13, "warn")]
    #[case::error(Level::ERROR, 17, "error")]
    #[case::fatal(Level::FATAL, 21, "fatal")]
    fn builtin_consts_have_expected_fields(
        #[case] level: Level,
        #[case] expected_severity: u8,
        #[case] expected_name: &str,
    ) {
        assert_eq!(level.severity(), expected_severity);
        assert_eq!(level.name(), expected_name);
    }

    #[test]
    fn builtin_ordering_holds() {
        assert!(Level::TRACE < Level::DEBUG);
        assert!(Level::DEBUG < Level::INFO);
        assert!(Level::INFO < Level::WARN);
        assert!(Level::WARN < Level::ERROR);
        assert!(Level::ERROR < Level::FATAL);
    }

    // 2. custom level construction
    #[test]
    fn custom_level_constructs() {
        let audit = Level::custom("audit", 18);
        assert_eq!(audit.severity(), 18);
        assert_eq!(audit.name(), "audit");
    }

    // 3. invalid name returns Err::InvalidInput
    #[test]
    fn from_str_bogus_returns_invalid_input() {
        let result = "bogus".parse::<Level>();
        assert_eq!(result.unwrap_err(), Error::InvalidInput);
    }

    // 4. zero severity and empty name are allowed (edge)
    #[test]
    fn custom_zero_severity_allowed() {
        let edge = Level::custom("", 0);
        // zero severity is valid as a sentinel; name is caller's responsibility
        assert_eq!(edge.severity(), 0);
        assert_eq!(edge.name(), "");
    }

    // 5. Display prints the name, no extra punctuation
    #[rstest]
    #[case::trace(Level::TRACE, "trace")]
    #[case::debug(Level::DEBUG, "debug")]
    #[case::info(Level::INFO, "info")]
    #[case::warn(Level::WARN, "warn")]
    #[case::error(Level::ERROR, "error")]
    #[case::fatal(Level::FATAL, "fatal")]
    fn display_prints_lowercase_name(#[case] level: Level, #[case] expected: &str) {
        assert_eq!(level.to_string(), expected);
    }

    // 6. round-trip: from_str(level.name()) == level for each built-in
    #[rstest]
    #[case::trace(Level::TRACE)]
    #[case::debug(Level::DEBUG)]
    #[case::info(Level::INFO)]
    #[case::warn(Level::WARN)]
    #[case::error(Level::ERROR)]
    #[case::fatal(Level::FATAL)]
    fn from_str_roundtrip(#[case] level: Level) {
        let parsed = level.name().parse::<Level>().unwrap();
        assert_eq!(parsed, level);
    }

    // 7. sorted Vec of mixed built-ins + customs is in severity order
    #[test]
    fn sorted_vec_is_severity_order() {
        let audit = Level::custom("audit", 18);
        let mut levels = [
            Level::FATAL,
            audit,
            Level::TRACE,
            Level::INFO,
            Level::WARN,
            Level::DEBUG,
            Level::ERROR,
        ];
        levels.sort();
        let severities: Vec<u8> = levels.iter().map(|lv| lv.severity()).collect();
        assert_eq!(severities, vec![1u8, 5, 9, 13, 17, 18, 21]);
    }
}
